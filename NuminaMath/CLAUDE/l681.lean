import Mathlib

namespace NUMINAMATH_CALUDE_root_product_theorem_l681_68132

-- Define the polynomial f(x)
def f (x : ℂ) : ℂ := x^6 + 2*x^3 + 1

-- Define the function h(x)
def h (x : ℂ) : ℂ := x^3 - 3*x

-- State the theorem
theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ y₆ : ℂ) 
  (hf₁ : f y₁ = 0) (hf₂ : f y₂ = 0) (hf₃ : f y₃ = 0)
  (hf₄ : f y₄ = 0) (hf₅ : f y₅ = 0) (hf₆ : f y₆ = 0) :
  (h y₁) * (h y₂) * (h y₃) * (h y₄) * (h y₅) * (h y₆) = 676 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l681_68132


namespace NUMINAMATH_CALUDE_warehouse_paintable_area_l681_68106

/-- Represents the dimensions of a rectangular warehouse -/
structure Warehouse where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area of a warehouse -/
def totalPaintableArea (w : Warehouse) (windowCount : ℕ) (windowDim : Window) : ℝ :=
  let wallArea1 := 2 * (w.width * w.height) * 2  -- Both sides of width walls
  let wallArea2 := 2 * (w.length * w.height - windowCount * windowDim.width * windowDim.height) * 2  -- Both sides of length walls with windows
  let ceilingArea := w.width * w.length
  let floorArea := w.width * w.length
  wallArea1 + wallArea2 + ceilingArea + floorArea

/-- Theorem stating the total paintable area of the warehouse -/
theorem warehouse_paintable_area :
  let w : Warehouse := { width := 12, length := 15, height := 7 }
  let windowDim : Window := { width := 2, height := 3 }
  totalPaintableArea w 3 windowDim = 876 := by sorry

end NUMINAMATH_CALUDE_warehouse_paintable_area_l681_68106


namespace NUMINAMATH_CALUDE_median_sum_bounds_l681_68100

/-- Given a triangle ABC with medians m_a, m_b, m_c, and perimeter p,
    prove that the sum of the medians is between 3/2 and 2 times the perimeter. -/
theorem median_sum_bounds (m_a m_b m_c p : ℝ) (h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ p > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧
    m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (3/2) * p < m_a + m_b + m_c ∧ m_a + m_b + m_c < 2 * p := by
  sorry

end NUMINAMATH_CALUDE_median_sum_bounds_l681_68100


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l681_68173

theorem a_minus_b_equals_two (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l681_68173


namespace NUMINAMATH_CALUDE_power_of_power_l681_68160

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l681_68160


namespace NUMINAMATH_CALUDE_square_modification_l681_68169

theorem square_modification (x : ℝ) : 
  x > 0 →
  x^2 = (x - 2) * (1.2 * x) →
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_square_modification_l681_68169


namespace NUMINAMATH_CALUDE_shift_by_two_equiv_l681_68157

/-- A function that represents a vertical shift of another function -/
def verticalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- Theorem stating that f(x) + 2 is equivalent to shifting f(x) upward by 2 units -/
theorem shift_by_two_equiv (f : ℝ → ℝ) (x : ℝ) : 
  f x + 2 = verticalShift f 2 x := by sorry

end NUMINAMATH_CALUDE_shift_by_two_equiv_l681_68157


namespace NUMINAMATH_CALUDE_joe_trip_expenses_l681_68112

/-- Calculates the remaining money after expenses -/
def remaining_money (initial_savings flight_cost hotel_cost food_cost : ℕ) : ℕ :=
  initial_savings - (flight_cost + hotel_cost + food_cost)

/-- Proves that Joe has $1,000 left after his trip expenses -/
theorem joe_trip_expenses :
  remaining_money 6000 1200 800 3000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_joe_trip_expenses_l681_68112


namespace NUMINAMATH_CALUDE_jingJing_bought_four_notebooks_l681_68193

/-- Represents the purchase of stationery items -/
structure StationeryPurchase where
  carbonPens : ℕ
  notebooks : ℕ
  pencilCases : ℕ

/-- Calculates the total cost of a stationery purchase -/
def totalCost (p : StationeryPurchase) : ℚ :=
  1.8 * p.carbonPens + 3.5 * p.notebooks + 4.2 * p.pencilCases

/-- Theorem stating that Jing Jing bought 4 notebooks -/
theorem jingJing_bought_four_notebooks :
  ∃ (p : StationeryPurchase),
    p.carbonPens > 0 ∧
    p.notebooks > 0 ∧
    p.pencilCases > 0 ∧
    totalCost p = 20 ∧
    p.notebooks = 4 :=
by sorry

end NUMINAMATH_CALUDE_jingJing_bought_four_notebooks_l681_68193


namespace NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_over_three_l681_68176

theorem tan_fifteen_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_over_three_l681_68176


namespace NUMINAMATH_CALUDE_hannahs_work_hours_l681_68196

/-- Given Hannah's work conditions, prove the number of hours she worked -/
theorem hannahs_work_hours 
  (hourly_rate : ℕ) 
  (late_penalty : ℕ) 
  (times_late : ℕ) 
  (total_pay : ℕ) 
  (h1 : hourly_rate = 30)
  (h2 : late_penalty = 5)
  (h3 : times_late = 3)
  (h4 : total_pay = 525) :
  ∃ (hours_worked : ℕ), 
    hours_worked * hourly_rate - times_late * late_penalty = total_pay ∧ 
    hours_worked = 18 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_work_hours_l681_68196


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l681_68178

theorem angle_measure_in_triangle (y : ℝ) : 
  let angle_ABC : ℝ := 180
  let angle_CBD : ℝ := 115
  let angle_BAD : ℝ := 31
  angle_ABC = 180 ∧ 
  angle_CBD = 115 ∧ 
  angle_BAD = 31 ∧
  y + angle_BAD + (angle_ABC - angle_CBD) = 180
  → y = 84 := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l681_68178


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l681_68108

theorem quadratic_roots_condition (r : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (r - 4) * x₁^2 - 2*(r - 3) * x₁ + r = 0 ∧
   (r - 4) * x₂^2 - 2*(r - 3) * x₂ + r = 0 ∧
   x₁ > -1 ∧ x₂ > -1) ↔ 
  (3.5 < r ∧ r < 4.5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l681_68108


namespace NUMINAMATH_CALUDE_empty_graph_l681_68192

theorem empty_graph (x y : ℝ) : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 4*y + 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_graph_l681_68192


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l681_68181

/-- Given a map distance between two towns and a scale factor, calculate the actual distance -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale_factor : ℝ) 
  (h1 : map_distance = 45) 
  (h2 : scale_factor = 10) : 
  map_distance * scale_factor = 450 := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_distance_l681_68181


namespace NUMINAMATH_CALUDE_unique_integer_sum_l681_68124

theorem unique_integer_sum (b₃ b₄ b₅ b₆ b₇ b₈ : ℕ) : 
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧ b₃ ≠ b₈ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧ b₄ ≠ b₈ ∧
  b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧ b₅ ≠ b₈ ∧
  b₆ ≠ b₇ ∧ b₆ ≠ b₈ ∧
  b₇ ≠ b₈ →
  (11 : ℚ) / 9 = b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 →
  b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 →
  b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 25 := by
sorry


end NUMINAMATH_CALUDE_unique_integer_sum_l681_68124


namespace NUMINAMATH_CALUDE_maya_max_number_l681_68144

theorem maya_max_number : ∃ (max : ℕ), max = 600 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_maya_max_number_l681_68144


namespace NUMINAMATH_CALUDE_problem_1_l681_68182

theorem problem_1 (m n : ℤ) (h1 : 4*m + n = 90) (h2 : 2*m - 3*n = 10) :
  (m + 2*n)^2 - (3*m - n)^2 = -900 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l681_68182


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l681_68185

theorem marilyn_bottle_caps (initial_caps : ℝ) (received_caps : ℝ) : 
  initial_caps = 51.0 → received_caps = 36.0 → initial_caps + received_caps = 87.0 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l681_68185


namespace NUMINAMATH_CALUDE_stable_performance_comparison_l681_68122

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of stability based on variance -/
def more_stable (a b : ShooterStats) : Prop :=
  a.variance < b.variance

theorem stable_performance_comparison 
  (A B : ShooterStats)
  (h_avg : A.average_score = B.average_score)
  (h_var_A : A.variance = 0.4)
  (h_var_B : B.variance = 3.2) :
  more_stable A B :=
sorry

end NUMINAMATH_CALUDE_stable_performance_comparison_l681_68122


namespace NUMINAMATH_CALUDE_problem_statement_l681_68148

theorem problem_statement (a b c d : ℕ) 
  (h1 : d ∣ a^(2*b) + c) 
  (h2 : d ≥ a + c) : 
  d ≥ a + a^(1/(2*b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l681_68148


namespace NUMINAMATH_CALUDE_triangle_frame_stability_l681_68158

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides and three angles. -/
inductive Triangle : Type where
  | mk : Triangle

/-- Stability is a property of structures that resist deformation under load. -/
def Stability : Prop := sorry

/-- A bicycle frame made in the shape of a triangle provides stability. -/
theorem triangle_frame_stability (frame : BicycleFrame) (h : frame.shape = Triangle) : 
  Stability :=
sorry

end NUMINAMATH_CALUDE_triangle_frame_stability_l681_68158


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l681_68121

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for part (1)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} :=
sorry

-- Theorem for part (2)
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, m^2 - m < f x} = {m : ℝ | -1 < m ∧ m < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l681_68121


namespace NUMINAMATH_CALUDE_intersection_complement_P_and_Q_l681_68188

-- Define the set P
def P : Set ℝ := {y | ∃ x > 0, y = (1/2)^x}

-- Define the set Q
def Q : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- Define the complement of P in ℝ
def complement_P : Set ℝ := {y | y ≤ 0 ∨ y ≥ 1}

-- Theorem statement
theorem intersection_complement_P_and_Q :
  (complement_P ∩ Q) = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_P_and_Q_l681_68188


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l681_68179

theorem cookie_boxes_problem (type1_per_box type3_per_box : ℕ)
  (type1_boxes type2_boxes type3_boxes : ℕ)
  (total_cookies : ℕ)
  (h1 : type1_per_box = 12)
  (h2 : type3_per_box = 16)
  (h3 : type1_boxes = 50)
  (h4 : type2_boxes = 80)
  (h5 : type3_boxes = 70)
  (h6 : total_cookies = 3320)
  (h7 : type1_per_box * type1_boxes + type2_boxes * type2_per_box + type3_per_box * type3_boxes = total_cookies) :
  type2_per_box = 20 := by
  sorry


end NUMINAMATH_CALUDE_cookie_boxes_problem_l681_68179


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l681_68139

/-- Represents the characteristics of girls in a school -/
structure SchoolGirls where
  total : ℕ
  blueEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes -/
def brownEyedBrunettes (s : SchoolGirls) : ℕ :=
  s.brownEyed - (s.total - s.brunettes - s.blueEyedBlondes)

/-- Theorem stating the number of brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (s : SchoolGirls) 
  (h1 : s.total = 60)
  (h2 : s.blueEyedBlondes = 20)
  (h3 : s.brunettes = 35)
  (h4 : s.brownEyed = 25) :
  brownEyedBrunettes s = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, blueEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l681_68139


namespace NUMINAMATH_CALUDE_rectangle_area_l681_68105

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l681_68105


namespace NUMINAMATH_CALUDE_virus_memory_growth_l681_68123

/-- Represents the memory occupied by the virus in KB -/
def memory_occupied (t : ℕ) : ℕ := 2 * 2^t

/-- Represents the time elapsed in minutes -/
def time_elapsed (t : ℕ) : ℕ := 3 * t

theorem virus_memory_growth :
  ∃ t : ℕ, memory_occupied t = 64 * 2^10 ∧ time_elapsed t = 45 :=
sorry

end NUMINAMATH_CALUDE_virus_memory_growth_l681_68123


namespace NUMINAMATH_CALUDE_min_tangent_length_l681_68102

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l681_68102


namespace NUMINAMATH_CALUDE_jennifer_fruits_left_l681_68135

def fruits_left (pears oranges apples cherries grapes : ℕ) 
  (pears_given oranges_given apples_given cherries_given grapes_given : ℕ) : ℕ :=
  (pears - pears_given) + (oranges - oranges_given) + (apples - apples_given) + 
  (cherries - cherries_given) + (grapes - grapes_given)

theorem jennifer_fruits_left : 
  let pears : ℕ := 15
  let oranges : ℕ := 30
  let apples : ℕ := 2 * pears
  let cherries : ℕ := oranges / 2
  let grapes : ℕ := 3 * apples
  fruits_left pears oranges apples cherries grapes 3 5 5 7 3 = 157 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fruits_left_l681_68135


namespace NUMINAMATH_CALUDE_roots_sum_powers_l681_68167

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l681_68167


namespace NUMINAMATH_CALUDE_trigonometric_identity_l681_68138

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.sin α ^ 2 + Real.sin α * Real.sin β + Real.sin β ^ 2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l681_68138


namespace NUMINAMATH_CALUDE_monomial_properties_l681_68191

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (R : Type*) [CommRing R] where
  coeff : R
  exponents : List ℕ

/-- The degree of a monomial is the sum of its exponents. -/
def Monomial.degree {R : Type*} [CommRing R] (m : Monomial R) : ℕ :=
  m.exponents.sum

/-- Our specific monomial -3x^2y -/
def our_monomial : Monomial ℤ :=
  { coeff := -3
  , exponents := [2, 1] }

theorem monomial_properties :
  our_monomial.coeff = -3 ∧ our_monomial.degree = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l681_68191


namespace NUMINAMATH_CALUDE_continuous_fraction_solution_l681_68151

theorem continuous_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (6 + 2 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_solution_l681_68151


namespace NUMINAMATH_CALUDE_room_painting_problem_l681_68140

/-- The total area of a room painted by two painters working together --/
def room_area (painter1_rate : ℝ) (painter2_rate : ℝ) (slowdown : ℝ) (time : ℝ) : ℝ :=
  time * (painter1_rate + painter2_rate - slowdown)

theorem room_painting_problem :
  let painter1_rate := 1 / 6
  let painter2_rate := 1 / 8
  let slowdown := 5
  let time := 4
  room_area painter1_rate painter2_rate slowdown time = 120 := by
sorry

end NUMINAMATH_CALUDE_room_painting_problem_l681_68140


namespace NUMINAMATH_CALUDE_wire_length_proof_l681_68156

/-- The length of wire used to make an equilateral triangle plus the leftover wire -/
def total_wire_length (side_length : ℝ) (leftover : ℝ) : ℝ :=
  3 * side_length + leftover

/-- Theorem: Given an equilateral triangle with side length 19 cm and 15 cm of leftover wire,
    the total length of wire is 72 cm. -/
theorem wire_length_proof :
  total_wire_length 19 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l681_68156


namespace NUMINAMATH_CALUDE_max_tickets_buyable_l681_68194

def regular_price : ℝ := 15
def discount_threshold : ℕ := 6
def discount_rate : ℝ := 0.1
def budget : ℝ := 120

def discounted_price : ℝ := regular_price * (1 - discount_rate)

def cost (n : ℕ) : ℝ :=
  if n ≤ discount_threshold then n * regular_price
  else n * discounted_price

theorem max_tickets_buyable :
  ∀ n : ℕ, cost n ≤ budget → n ≤ 8 ∧ cost 8 ≤ budget :=
sorry

end NUMINAMATH_CALUDE_max_tickets_buyable_l681_68194


namespace NUMINAMATH_CALUDE_product_abcd_l681_68143

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 8 * d = 45) →
  (4 * (d + c) = b) →
  (4 * b + c = a) →
  (c + 1 = d) →
  (a * b * c * d = (1511 / 103) * (332 / 103) * (-7 / 103) * (96 / 103)) := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l681_68143


namespace NUMINAMATH_CALUDE_nested_series_sum_l681_68172

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 7 = 510 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l681_68172


namespace NUMINAMATH_CALUDE_water_margin_price_l681_68129

theorem water_margin_price :
  ∀ (x : ℝ),
    (x > 0) →
    (3600 / (x + 60) = (1 / 2) * (4800 / x)) →
    x = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_water_margin_price_l681_68129


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_repeated_sixes_and_seven_l681_68114

def digits_of_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

def digits_of_fours (n : ℕ) : ℕ :=
  4 * digits_of_ones n * (10^n) + 4 * digits_of_ones n

theorem sqrt_expression_equals_repeated_sixes_and_seven (n : ℕ) :
  let a := digits_of_ones n
  let fours := digits_of_fours n
  let ones := 10 * a + 1
  let sixes := 6 * a
  Real.sqrt (fours / (2 * n * (1/4)) + ones - sixes) = 6 * a + 1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_repeated_sixes_and_seven_l681_68114


namespace NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l681_68101

theorem polynomial_with_arithmetic_progression_roots (j : ℝ) : 
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + 16*x + 64 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    b - a = c - b ∧ d - c = c - b) →
  j = -160/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l681_68101


namespace NUMINAMATH_CALUDE_house_locations_contradiction_l681_68145

-- Define the directions
inductive Direction
  | North
  | South
  | East
  | West
  | Northeast
  | Northwest
  | Southeast
  | Southwest

-- Define a function to get the opposite direction
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East
  | Direction.Northeast => Direction.Southwest
  | Direction.Northwest => Direction.Southeast
  | Direction.Southeast => Direction.Northwest
  | Direction.Southwest => Direction.Northeast

-- Define the theorem
theorem house_locations_contradiction :
  ∀ (house1 house2 : Type) (direction1 direction2 : Direction),
    (direction1 = Direction.Southeast ∧ direction2 = Direction.Southwest) →
    (oppositeDirection direction1 ≠ direction2) :=
by sorry

end NUMINAMATH_CALUDE_house_locations_contradiction_l681_68145


namespace NUMINAMATH_CALUDE_inequality_proof_l681_68113

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Theorem statement
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l681_68113


namespace NUMINAMATH_CALUDE_sara_marbles_l681_68154

theorem sara_marbles (initial lost left : ℕ) : 
  lost = 7 → left = 3 → initial = lost + left → initial = 10 := by
sorry

end NUMINAMATH_CALUDE_sara_marbles_l681_68154


namespace NUMINAMATH_CALUDE_biased_dice_probability_l681_68152

def num_rolls : ℕ := 10
def num_sixes : ℕ := 4
def prob_six : ℚ := 1/3
def prob_not_six : ℚ := 2/3

theorem biased_dice_probability :
  (Nat.choose num_rolls num_sixes) * (prob_six ^ num_sixes) * (prob_not_six ^ (num_rolls - num_sixes)) = 13440/59049 :=
by sorry

end NUMINAMATH_CALUDE_biased_dice_probability_l681_68152


namespace NUMINAMATH_CALUDE_total_rooms_to_paint_l681_68195

/-- Proves that the total number of rooms to be painted is 9 -/
theorem total_rooms_to_paint (hours_per_room : ℕ) (rooms_painted : ℕ) (hours_remaining : ℕ) : 
  hours_per_room = 8 → rooms_painted = 5 → hours_remaining = 32 →
  rooms_painted + (hours_remaining / hours_per_room) = 9 :=
by
  sorry

#check total_rooms_to_paint

end NUMINAMATH_CALUDE_total_rooms_to_paint_l681_68195


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_6_with_digit_sum_12_l681_68127

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 204 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_6_with_digit_sum_12_l681_68127


namespace NUMINAMATH_CALUDE_odd_function_value_l681_68120

def f (x : ℝ) : ℝ := sorry

theorem odd_function_value (a : ℝ) : 
  (∀ x : ℝ, f x = -f (-x)) → 
  (∀ x : ℝ, x ≥ 0 → f x = 3^x - 2*x + a) → 
  f (-2) = -4 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l681_68120


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l681_68133

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  ∃ (s₁ s₂ : ℝ),
    s₁ > 0 ∧ s₂ > 0 ∧
    s₁^2 + (s₁/2)^2 = r^2 ∧  -- Square inscribed in quarter-circle
    s₂^2 = 2*r^2 ∧           -- Square inscribed in full circle
    s₁^2 / s₂^2 = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l681_68133


namespace NUMINAMATH_CALUDE_square_of_binomial_condition_l681_68168

theorem square_of_binomial_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_condition_l681_68168


namespace NUMINAMATH_CALUDE_smallest_x_with_natural_percentages_l681_68166

theorem smallest_x_with_natural_percentages :
  ∀ x : ℝ, x > 0 →
    (∃ n : ℕ, (45 / 100) * x = n) →
    (∃ m : ℕ, (24 / 100) * x = m) →
    x ≥ 100 / 3 ∧
    (∃ a b : ℕ, (45 / 100) * (100 / 3) = a ∧ (24 / 100) * (100 / 3) = b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_natural_percentages_l681_68166


namespace NUMINAMATH_CALUDE_train_speed_proof_l681_68177

/-- The speed of the train from city A -/
def speed_train_A : ℝ := 60

/-- The speed of the train from city B -/
def speed_train_B : ℝ := 75

/-- The total distance between cities A and B in km -/
def total_distance : ℝ := 465

/-- The time in hours that the train from A travels before meeting -/
def time_train_A : ℝ := 4

/-- The time in hours that the train from B travels before meeting -/
def time_train_B : ℝ := 3

theorem train_speed_proof : 
  speed_train_A * time_train_A + speed_train_B * time_train_B = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l681_68177


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l681_68134

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.4 * L) (h2 : B' * L' = 1.05 * B * L) :
  B' = 0.75 * B :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l681_68134


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l681_68165

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 5*x - 20

theorem polynomial_remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x + 2) * q x + 98 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l681_68165


namespace NUMINAMATH_CALUDE_min_value_expression_l681_68104

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 3 + 2*Real.sqrt 2 ∧ 
  ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' + 2*y' + z' = 1 → 
    (1 / (x' + y')) + (2 / (y' + z')) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l681_68104


namespace NUMINAMATH_CALUDE_hall_width_proof_l681_68126

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 15 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 30 →
  total_cost = 28500 →
  ∃ w : ℝ, w > 0 ∧ 
    (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧
    w = 15 := by
  sorry

#check hall_width_proof

end NUMINAMATH_CALUDE_hall_width_proof_l681_68126


namespace NUMINAMATH_CALUDE_irrational_sqrt_10_and_others_rational_l681_68111

theorem irrational_sqrt_10_and_others_rational : 
  (Irrational (Real.sqrt 10)) ∧ 
  (¬ Irrational (1 / 7 : ℝ)) ∧ 
  (¬ Irrational (3.5 : ℝ)) ∧ 
  (¬ Irrational (-0.3030030003 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_10_and_others_rational_l681_68111


namespace NUMINAMATH_CALUDE_batsman_average_l681_68174

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 12 →
  last_innings_score = 65 →
  average_increase = 2 →
  (∃ (prev_average : ℕ),
    (prev_average * (total_innings - 1) + last_innings_score) / total_innings = prev_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score + (total_innings - 1) * average_increase) / total_innings - average_increase) + last_innings_score) / total_innings) = 43 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l681_68174


namespace NUMINAMATH_CALUDE_simple_interest_years_l681_68189

/-- Calculates the number of years for which a sum was put at simple interest, given the principal amount and the additional interest earned with a 1% rate increase. -/
def calculateYears (principal : ℚ) (additionalInterest : ℚ) : ℚ :=
  (100 * additionalInterest) / principal

theorem simple_interest_years :
  let principal : ℚ := 2300
  let additionalInterest : ℚ := 69
  calculateYears principal additionalInterest = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_years_l681_68189


namespace NUMINAMATH_CALUDE_anna_chargers_l681_68164

theorem anna_chargers (phone_chargers : ℕ) (laptop_chargers : ℕ) : 
  laptop_chargers = 5 * phone_chargers →
  phone_chargers + laptop_chargers = 24 →
  phone_chargers = 4 := by
sorry

end NUMINAMATH_CALUDE_anna_chargers_l681_68164


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l681_68162

/-- Yan's scenario with distances and speeds -/
structure YanScenario where
  a : ℝ  -- distance from Yan to home
  b : ℝ  -- distance from Yan to mall
  w : ℝ  -- Yan's walking speed
  bike_speed : ℝ -- Yan's bicycle speed

/-- The conditions of Yan's scenario -/
def valid_scenario (s : YanScenario) : Prop :=
  s.a > 0 ∧ s.b > 0 ∧ s.w > 0 ∧
  s.bike_speed = 5 * s.w ∧
  s.b / s.w = s.a / s.w + (s.a + s.b) / s.bike_speed

/-- The theorem stating the ratio of distances -/
theorem yan_distance_ratio (s : YanScenario) (h : valid_scenario s) :
  s.a / s.b = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l681_68162


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l681_68183

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 3) + (3 * x^2 - 27) / x
  ∃ (min_sol : ℝ), min_sol = (8 - Real.sqrt 145) / 3 ∧
    f min_sol = 14 ∧
    ∀ (y : ℝ), f y = 14 → y ≥ min_sol :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l681_68183


namespace NUMINAMATH_CALUDE_min_value_quadratic_l681_68136

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 39 / 4 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39 / 4 ↔ x = 1 / 2 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l681_68136


namespace NUMINAMATH_CALUDE_carnation_percentage_l681_68150

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℝ
  pink_roses : ℝ
  red_roses : ℝ
  pink_carnations : ℝ
  red_carnations : ℝ
  yellow_tulips : ℝ

/-- Conditions for the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = 0.4 * b.total ∧
  b.red_roses + b.red_carnations = 0.4 * b.total ∧
  b.yellow_tulips = 0.2 * b.total ∧
  b.pink_roses = (2/5) * (b.pink_roses + b.pink_carnations) ∧
  b.red_carnations = (1/2) * (b.red_roses + b.red_carnations)

/-- Theorem stating that the percentage of carnations is 44% -/
theorem carnation_percentage (b : Bouquet) (h : bouquet_conditions b) : 
  (b.pink_carnations + b.red_carnations) / b.total = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l681_68150


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l681_68130

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l681_68130


namespace NUMINAMATH_CALUDE_book_arrangement_count_l681_68128

/-- Represents the number of math books -/
def num_math_books : ℕ := 3

/-- Represents the number of physics books -/
def num_physics_books : ℕ := 2

/-- Represents the number of chemistry books -/
def num_chem_books : ℕ := 1

/-- Represents the total number of books -/
def total_books : ℕ := num_math_books + num_physics_books + num_chem_books

/-- Calculates the number of arrangements of books on a shelf -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements of books on a shelf,
    where math books are adjacent and physics books are not adjacent,
    is equal to 72 -/
theorem book_arrangement_count :
  num_arrangements = 72 ∧
  num_math_books = 3 ∧
  num_physics_books = 2 ∧
  num_chem_books = 1 ∧
  total_books = num_math_books + num_physics_books + num_chem_books :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l681_68128


namespace NUMINAMATH_CALUDE_inequality_solution_range_l681_68131

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l681_68131


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l681_68147

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l681_68147


namespace NUMINAMATH_CALUDE_sum_of_roots_l681_68146

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 12) (hb : b * (b - 4) = 12) (hab : a ≠ b) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l681_68146


namespace NUMINAMATH_CALUDE_extreme_values_and_sum_l681_68153

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem extreme_values_and_sum (α β : ℝ) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x ≥ -2 ∧ f x ≤ 2) ∧
  f (α - Real.pi / 6) = 2 * Real.sqrt 5 / 5 ∧
  Real.sin (β - α) = Real.sqrt 10 / 10 ∧
  α ∈ Set.Icc (Real.pi / 4) Real.pi ∧
  β ∈ Set.Icc Real.pi (3 * Real.pi / 2) →
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = -2) ∧
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = 2) ∧
  α + β = 7 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_sum_l681_68153


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_component_l681_68103

/-- Given two 2D vectors m and n, if they are perpendicular and have specific components,
    then the x-component of m must be 2. -/
theorem perpendicular_vectors_x_component
  (m n : ℝ × ℝ)  -- m and n are 2D real vectors
  (h1 : m.1 = x ∧ m.2 = 2)  -- m = (x, 2)
  (h2 : n = (1, -1))  -- n = (1, -1)
  (h3 : m • n = 0)  -- m is perpendicular to n (dot product is zero)
  : x = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_component_l681_68103


namespace NUMINAMATH_CALUDE_largest_of_three_numbers_l681_68137

theorem largest_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_products_eq : x*y + x*z + y*z = -6)
  (product_eq : x*y*z = -8) :
  ∃ max_val : ℝ, max_val = (1 + Real.sqrt 17) / 2 ∧ 
  max_val = max x (max y z) := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_numbers_l681_68137


namespace NUMINAMATH_CALUDE_min_trips_for_elevator_l681_68117

def weights : List ℕ := [50, 51, 55, 57, 58, 59, 60, 63, 75, 140]
def max_capacity : ℕ := 180

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ max_capacity

def covers_all_weights (trips : List (List ℕ)) : Prop :=
  weights.all (λ w => ∃ t ∈ trips, w ∈ t)

theorem min_trips_for_elevator : 
  ∃ (trips : List (List ℕ)), 
    trips.length = 4 ∧ 
    (∀ t ∈ trips, is_valid_trip t) ∧
    covers_all_weights trips ∧
    (∀ (other_trips : List (List ℕ)), 
      (∀ t ∈ other_trips, is_valid_trip t) → 
      covers_all_weights other_trips → 
      other_trips.length ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_trips_for_elevator_l681_68117


namespace NUMINAMATH_CALUDE_wrappers_found_at_park_l681_68159

/-- Represents the number of bottle caps Danny found at the park. -/
def bottle_caps_found : ℕ := 58

/-- Represents the number of wrappers Danny now has in his collection. -/
def wrappers_now : ℕ := 11

/-- Represents the number of bottle caps Danny now has in his collection. -/
def bottle_caps_now : ℕ := 12

/-- Represents the difference between bottle caps and wrappers Danny has now. -/
def cap_wrapper_difference : ℕ := 1

/-- Proves that the number of wrappers Danny found at the park is 11. -/
theorem wrappers_found_at_park : ℕ := by
  sorry

end NUMINAMATH_CALUDE_wrappers_found_at_park_l681_68159


namespace NUMINAMATH_CALUDE_divisor_problem_l681_68187

theorem divisor_problem (N : ℕ) (D : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 5 ∧ N % D = 3 → D = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l681_68187


namespace NUMINAMATH_CALUDE_max_value_xyz_l681_68119

theorem max_value_xyz (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 3 * x + 2 * y + 6 * z = 1) :
  x^4 * y^3 * z^2 ≤ 1 / 372008 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_l681_68119


namespace NUMINAMATH_CALUDE_class_division_transfer_l681_68199

/-- 
Given a class divided into two groups with 26 and 22 people respectively,
prove that the number of people transferred (x) from the second group to the first
satisfies the equation x + 26 = 3(22 - x) when the first group becomes
three times the size of the second group after the transfer.
-/
theorem class_division_transfer (x : ℤ) : x + 26 = 3 * (22 - x) ↔ 
  (26 + x = 3 * (22 - x) ∧ 
   26 + x > 0 ∧
   22 - x > 0) := by
  sorry

#check class_division_transfer

end NUMINAMATH_CALUDE_class_division_transfer_l681_68199


namespace NUMINAMATH_CALUDE_min_value_expression_l681_68171

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l681_68171


namespace NUMINAMATH_CALUDE_percentage_not_receiving_muffin_l681_68141

theorem percentage_not_receiving_muffin (total_percentage : ℝ) (muffin_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : muffin_percentage = 38) : 
  total_percentage - muffin_percentage = 62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_receiving_muffin_l681_68141


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l681_68118

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = f(x)
variable (x y : ℝ)

-- Define the reflection transformation across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem reflection_across_y_axis :
  y = f x ↔ y = f (-(-x)) :=
sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l681_68118


namespace NUMINAMATH_CALUDE_opera_selection_probability_l681_68109

theorem opera_selection_probability :
  let total_operas : ℕ := 5
  let distinguished_operas : ℕ := 2
  let selection_size : ℕ := 2

  let total_combinations : ℕ := Nat.choose total_operas selection_size
  let favorable_combinations : ℕ := distinguished_operas * (total_operas - distinguished_operas)

  (favorable_combinations : ℚ) / total_combinations = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_opera_selection_probability_l681_68109


namespace NUMINAMATH_CALUDE_teacher_total_score_l681_68163

/-- Calculates the total score of a teacher based on their written test and interview scores -/
def calculate_total_score (written_score : ℝ) (interview_score : ℝ) 
  (written_weight : ℝ) (interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem: The teacher's total score is 72 points -/
theorem teacher_total_score : 
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  calculate_total_score written_score interview_score written_weight interview_weight = 72 := by
sorry

end NUMINAMATH_CALUDE_teacher_total_score_l681_68163


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l681_68175

/-- The number of cats sold during a pet store sale -/
theorem cats_sold_during_sale 
  (siamese_initial : ℕ) 
  (house_initial : ℕ) 
  (cats_left : ℕ) 
  (h1 : siamese_initial = 13)
  (h2 : house_initial = 5)
  (h3 : cats_left = 8) :
  siamese_initial + house_initial - cats_left = 10 :=
by sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l681_68175


namespace NUMINAMATH_CALUDE_inequality_holds_for_nonzero_reals_l681_68142

theorem inequality_holds_for_nonzero_reals (x : ℝ) (h : x ≠ 0) :
  (x^3 - 2*x^5 + x^6) / (x - 2*x^2 + x^4) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_nonzero_reals_l681_68142


namespace NUMINAMATH_CALUDE_solve_equation_l681_68161

theorem solve_equation (x : ℝ) : 2 * x = (26 - x) + 19 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l681_68161


namespace NUMINAMATH_CALUDE_matrix_inverse_zero_l681_68197

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -6; -2, 3]

theorem matrix_inverse_zero : 
  A⁻¹ = !![0, 0; 0, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_zero_l681_68197


namespace NUMINAMATH_CALUDE_f_at_two_l681_68190

def f (x : ℝ) : ℝ := 15 * x^5 - 24 * x^4 + 33 * x^3 - 42 * x^2 + 51 * x

theorem f_at_two : f 2 = 294 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l681_68190


namespace NUMINAMATH_CALUDE_ellipse_properties_l681_68170

/-- Given an ellipse with the following properties:
  * Equation: x²/a² + y²/b² = 1, where a > b > 0
  * Vertices: A(0,b) and C(0,-b)
  * Foci: F₁(-c,0) and F₂(c,0), where c > 0
  * A line through point E(3c,0) intersects the ellipse at another point B
  * F₁A ∥ F₂B
-/
theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a -- eccentricity
  let m := (5/3) * c
  let n := (2*Real.sqrt 2/3) * c
  ∃ (x y : ℝ),
    -- Point B on the ellipse
    x^2/a^2 + y^2/b^2 = 1 ∧
    -- B is on the line through E(3c,0)
    ∃ (t : ℝ), x = 3*c*(1-t) ∧ y = 3*c*t ∧
    -- F₁A ∥ F₂B
    (b / (-c)) = (y - 0) / (x - c) ∧
    -- Eccentricity is √3/3
    e = Real.sqrt 3 / 3 ∧
    -- Point H(m,n) is on F₂B
    n / (m - c) = y / (x - c) ∧
    -- H is on the circumcircle of AF₁C
    (m - c/2)^2 + n^2 = (3*c/2)^2 ∧
    -- Ratio n/m
    n / m = 2 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l681_68170


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_129_6_l681_68180

theorem percentage_of_360_equals_129_6 : 
  (129.6 / 360) * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_129_6_l681_68180


namespace NUMINAMATH_CALUDE_no_formula_matches_all_points_l681_68125

-- Define the table of values
def table : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 100), (3, 20), (4, -80)]

-- Define the formulas
def formula_A (x : ℤ) : ℤ := 200 - 30*x
def formula_B (x : ℤ) : ℤ := 200 - 20*x - 10*x^2
def formula_C (x : ℤ) : ℤ := 200 - 40*x + 10*x^2
def formula_D (x : ℤ) : ℤ := 200 - 10*x - 20*x^2

-- Theorem statement
theorem no_formula_matches_all_points :
  ¬(∀ (x y : ℤ), (x, y) ∈ table → 
    (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_matches_all_points_l681_68125


namespace NUMINAMATH_CALUDE_all_stars_arrangement_l681_68155

/-- The number of ways to arrange All-Stars in a row -/
def arrange_all_stars (total : ℕ) (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) (dodgers : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial cubs * Nat.factorial red_sox * Nat.factorial yankees * Nat.factorial dodgers

/-- Theorem stating the number of arrangements for the given problem -/
theorem all_stars_arrangement :
  arrange_all_stars 10 4 3 2 1 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_all_stars_arrangement_l681_68155


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l681_68110

theorem decimal_to_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 0.425875 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.425875 → c ≤ a ∧ d ≤ b → 
  (a : ℕ) + (b : ℕ) = 11407 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l681_68110


namespace NUMINAMATH_CALUDE_only_KC2H3O2_turns_pink_l681_68198

-- Define the set of solutes
inductive Solute
| NaCl
| KC2H3O2
| LiBr
| NH4NO3

-- Define a function to determine if a solution is basic
def isBasic (s : Solute) : Prop :=
  match s with
  | Solute.KC2H3O2 => True
  | _ => False

-- Define a function to check if phenolphthalein turns pink
def turnsPink (s : Solute) : Prop := isBasic s

-- Theorem statement
theorem only_KC2H3O2_turns_pink :
  ∀ s : Solute, turnsPink s ↔ s = Solute.KC2H3O2 :=
by sorry

end NUMINAMATH_CALUDE_only_KC2H3O2_turns_pink_l681_68198


namespace NUMINAMATH_CALUDE_removed_triangles_area_l681_68184

theorem removed_triangles_area (original_side : ℝ) (h_original_side : original_side = 20) :
  let smaller_side : ℝ := original_side / 2
  let removed_triangle_leg : ℝ := (original_side - smaller_side) / Real.sqrt 2
  let single_triangle_area : ℝ := removed_triangle_leg ^ 2 / 2
  4 * single_triangle_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l681_68184


namespace NUMINAMATH_CALUDE_eight_positions_l681_68149

def number : ℚ := 38.82

theorem eight_positions (n : ℚ) (h : n = number) : 
  (n - 38 = 0.82) ∧ 
  (n - 38.8 = 0.02) :=
by sorry

end NUMINAMATH_CALUDE_eight_positions_l681_68149


namespace NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l681_68186

theorem complex_roots_equilateral_triangle (z₁ z₂ p q : ℂ) : 
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l681_68186


namespace NUMINAMATH_CALUDE_function_value_determination_l681_68107

theorem function_value_determination (A : ℝ) (α : ℝ) 
  (h1 : A ≠ 0)
  (h2 : α ∈ Set.Icc 0 π)
  (h3 : A * Real.sin (α + π/4) = Real.cos (2*α))
  (h4 : Real.sin (2*α) = -7/9) :
  A = -4*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_determination_l681_68107


namespace NUMINAMATH_CALUDE_tangent_at_negative_one_a_lower_bound_l681_68115

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the derivative of g
def g' (x : ℝ) : ℝ := 2*x

-- Define the condition for the tangent line
def tangent_condition (x₁ a : ℝ) : Prop :=
  ∃ x₂, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

-- Theorem 1: When x₁ = -1, a = 3
theorem tangent_at_negative_one :
  tangent_condition (-1) 3 :=
sorry

-- Theorem 2: For all valid x₁, a ≥ -1
theorem a_lower_bound :
  ∀ x₁ a : ℝ, tangent_condition x₁ a → a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_negative_one_a_lower_bound_l681_68115


namespace NUMINAMATH_CALUDE_extremal_values_sum_l681_68116

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5, 
    S_max and S_min are the maximum and minimum values of x^2 + y^2 respectively. -/
theorem extremal_values_sum (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let S := x^2 + y^2
  let S_max := (10 : ℝ) / 3
  let S_min := (10 : ℝ) / 13
  (1 / S_max) + (1 / S_min) = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_extremal_values_sum_l681_68116
