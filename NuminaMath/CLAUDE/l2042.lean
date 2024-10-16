import Mathlib

namespace NUMINAMATH_CALUDE_intersection_implies_m_in_range_l2042_204207

/-- A line intersects a circle if the distance from the circle's center to the line is less than the circle's radius -/
def line_intersects_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : Prop :=
  (|a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)) < r

/-- The problem statement -/
theorem intersection_implies_m_in_range :
  ∃ m : ℤ, (3 ≤ m ∧ m ≤ 6) ∧
  line_intersects_circle 4 3 (2 * ↑m) (-3) 1 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_in_range_l2042_204207


namespace NUMINAMATH_CALUDE_path_length_is_pi_l2042_204240

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the path length of a dot on the center of the top face when the prism is rolled -/
def pathLength (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that the path length for a 2x2x4 cm prism is π cm -/
theorem path_length_is_pi :
  let prism := RectangularPrism.mk 4 2 2
  pathLength prism = π :=
sorry

end NUMINAMATH_CALUDE_path_length_is_pi_l2042_204240


namespace NUMINAMATH_CALUDE_children_savings_l2042_204276

def josiah_daily_savings : ℚ := 0.25
def josiah_days : ℕ := 24

def leah_daily_savings : ℚ := 0.50
def leah_days : ℕ := 20

def megan_days : ℕ := 12

def total_savings (j_daily : ℚ) (j_days : ℕ) (l_daily : ℚ) (l_days : ℕ) (m_days : ℕ) : ℚ :=
  j_daily * j_days + l_daily * l_days + 2 * l_daily * m_days

theorem children_savings : 
  total_savings josiah_daily_savings josiah_days leah_daily_savings leah_days megan_days = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_savings_l2042_204276


namespace NUMINAMATH_CALUDE_square_difference_inapplicable_l2042_204290

/-- The square difference formula cannot be directly applied to (x-y)(-x+y) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ) (c₁ c₂ c₃ c₄ : ℝ), 
    (a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    ((x - y) * (-x + y) = (a + b) * (a - b) ∨ (x - y) * (-x + y) = (a - b) * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_inapplicable_l2042_204290


namespace NUMINAMATH_CALUDE_vectors_are_parallel_l2042_204282

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ (k : ℝ), b = k • a := by sorry

end NUMINAMATH_CALUDE_vectors_are_parallel_l2042_204282


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2042_204296

theorem least_positive_integer_congruence :
  ∃! y : ℕ+, y.val + 3077 ≡ 1456 [ZMOD 15] ∧
  ∀ z : ℕ+, z.val + 3077 ≡ 1456 [ZMOD 15] → y ≤ z ∧ y.val = 14 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2042_204296


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l2042_204217

/-- Given a polynomial f and its related polynomial g, proves that g(1) = (4 - a) / (4c) -/
theorem polynomial_root_relation (a b c : ℝ) (f g : ℝ → ℝ) : 
  (∀ x, f x = x^3 + 2*a*x^2 + 3*b*x + 4*c) →
  a > 0 → b > 0 → c > 0 →
  a + b + c = 1 →
  (∀ r, f r = 0 → ∃ s, g s = 0 ∧ r * s = 1) →
  g 1 = (4 - a) / (4 * c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l2042_204217


namespace NUMINAMATH_CALUDE_inequality_proof_l2042_204200

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2042_204200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_10_l2042_204201

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_3_5 : a 3 + a 5 = 14
  sum_n : ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (a 1 + a n) / 2 = 100

/-- The theorem stating that n = 10 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_10 (seq : ArithmeticSequence) :
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (seq.a 1 + seq.a n) / 2 = 100 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_10_l2042_204201


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l2042_204232

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ → ℝ × ℝ := λ a ↦ (a, 0)
def C : ℝ × ℝ := (0, 4)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

-- Theorem statement
theorem collinear_points_imply_a_equals_four :
  ∀ a : ℝ, collinear A (B a) C → a = 4 := by sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l2042_204232


namespace NUMINAMATH_CALUDE_f_5_equals_357_l2042_204264

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + 5 * n + 7

theorem f_5_equals_357 : f 5 = 357 := by sorry

end NUMINAMATH_CALUDE_f_5_equals_357_l2042_204264


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2042_204216

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2042_204216


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l2042_204226

/-- The number of bacteria after a given number of tripling events -/
def bacteria_count (initial_count : ℕ) (tripling_events : ℕ) : ℕ :=
  initial_count * (3 ^ tripling_events)

/-- The number of tripling events in a given number of seconds -/
def tripling_events (seconds : ℕ) : ℕ :=
  seconds / 20

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count (tripling_events 180) = 275562 ∧
    initial_count = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l2042_204226


namespace NUMINAMATH_CALUDE_zeros_difference_l2042_204218

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, -2)

/-- Another point on the parabola -/
def point_on_parabola : ℝ × ℝ := (5, 14)

/-- The zeros of the parabola -/
def zeros (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem stating that the difference between the zeros of the parabola is √2 -/
theorem zeros_difference (p : Parabola) : 
  let (m, n) := zeros p
  m - n = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_zeros_difference_l2042_204218


namespace NUMINAMATH_CALUDE_area_of_triangle_RZX_l2042_204212

-- Define the square WXYZ
def Square (W X Y Z : ℝ × ℝ) : Prop :=
  -- Add conditions for a square here
  sorry

-- Define the area of a shape
def Area (shape : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) (ratio : ℝ) : Prop :=
  -- P is on AB with AP:PB = ratio:(1-ratio)
  sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  -- M is the midpoint of AB
  sorry

theorem area_of_triangle_RZX 
  (W X Y Z : ℝ × ℝ)
  (P Q R : ℝ × ℝ)
  (h_square : Square W X Y Z)
  (h_area_WXYZ : Area {W, X, Y, Z} = 144)
  (h_P_on_YZ : PointOnSegment P Y Z (1/3))
  (h_Q_mid_WP : Midpoint Q W P)
  (h_R_mid_XP : Midpoint R X P)
  (h_area_YPRQ : Area {Y, P, R, Q} = 30)
  : Area {R, Z, X} = 24 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_RZX_l2042_204212


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2042_204227

theorem rectangle_shorter_side (a b d : ℝ) : 
  a > 0 → b > 0 → d > 0 →
  (a / b = 3 / 4) →
  (a^2 + b^2 = d^2) →
  d = 9 →
  a = 5.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2042_204227


namespace NUMINAMATH_CALUDE_hex_725_equals_octal_3445_l2042_204223

-- Define a function to convert a base-16 number to base-10
def hexToDecimal (hex : String) : ℕ := sorry

-- Define a function to convert a base-10 number to base-8
def decimalToOctal (decimal : ℕ) : String := sorry

-- Theorem statement
theorem hex_725_equals_octal_3445 :
  decimalToOctal (hexToDecimal "725") = "3445" := by sorry

end NUMINAMATH_CALUDE_hex_725_equals_octal_3445_l2042_204223


namespace NUMINAMATH_CALUDE_square_side_length_l2042_204281

theorem square_side_length (w h r s : ℕ) : 
  w = 4000 →
  h = 2300 →
  2 * r + s = h →
  2 * r + 3 * s = w →
  s = 850 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2042_204281


namespace NUMINAMATH_CALUDE_exam_failure_count_l2042_204253

theorem exam_failure_count (total : ℕ) (pass_percent : ℚ) (fail_count : ℕ) : 
  total = 800 → 
  pass_percent = 35 / 100 → 
  fail_count = total - (pass_percent * total).floor → 
  fail_count = 520 := by
sorry

end NUMINAMATH_CALUDE_exam_failure_count_l2042_204253


namespace NUMINAMATH_CALUDE_ceremonial_team_arrangements_l2042_204299

def total_boys : ℕ := 48
def total_girls : ℕ := 32

def is_valid_arrangement (n : ℕ) : Prop :=
  n > 1 ∧
  total_boys % n = 0 ∧
  total_girls % n = 0 ∧
  (total_boys / n) = (total_girls / n) * 3 / 2

theorem ceremonial_team_arrangements :
  {n : ℕ | is_valid_arrangement n} = {2, 4, 8, 16} :=
by sorry

end NUMINAMATH_CALUDE_ceremonial_team_arrangements_l2042_204299


namespace NUMINAMATH_CALUDE_field_division_l2042_204220

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 := by
sorry

end NUMINAMATH_CALUDE_field_division_l2042_204220


namespace NUMINAMATH_CALUDE_exponential_inequality_l2042_204261

theorem exponential_inequality (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x - Real.log y < 1 / Real.log x - 1 / Real.log y) : 
  Real.exp (y - x) > 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2042_204261


namespace NUMINAMATH_CALUDE_find_x_l2042_204297

theorem find_x (x y z : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * z = 2 * y) 
  (h3 : x + y = 14) : 
  x = 12 := by sorry

end NUMINAMATH_CALUDE_find_x_l2042_204297


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2042_204266

theorem complex_arithmetic_equality : (1 - Complex.I) - (-3 + 2 * Complex.I) + (4 - 6 * Complex.I) = 8 - 9 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2042_204266


namespace NUMINAMATH_CALUDE_eve_gift_cost_l2042_204287

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_percentage : ℝ := 0.25
def num_nieces : ℕ := 3

def total_cost_per_niece : ℝ := hand_mitts_cost + apron_cost + utensils_cost + knife_cost

def total_cost_before_discount : ℝ := num_nieces * total_cost_per_niece

def discount_amount : ℝ := discount_percentage * total_cost_before_discount

theorem eve_gift_cost : total_cost_before_discount - discount_amount = 135 := by
  sorry

end NUMINAMATH_CALUDE_eve_gift_cost_l2042_204287


namespace NUMINAMATH_CALUDE_salt_concentration_increase_l2042_204248

/-- Given a 100 kg solution with 10% salt concentration and adding 20 kg of pure salt,
    the final salt concentration is 25%. -/
theorem salt_concentration_increase (initial_solution : ℝ) (initial_concentration : ℝ) 
    (added_salt : ℝ) (final_concentration : ℝ) : 
    initial_solution = 100 →
    initial_concentration = 0.1 →
    added_salt = 20 →
    final_concentration = (initial_solution * initial_concentration + added_salt) / 
                          (initial_solution + added_salt) →
    final_concentration = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_salt_concentration_increase_l2042_204248


namespace NUMINAMATH_CALUDE_wendy_full_face_time_l2042_204237

/-- Calculates the total time for Wendy's "full face" routine -/
def full_face_time (num_products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (num_products - 1) * wait_time + makeup_time

/-- Proves that Wendy's "full face" routine takes 50 minutes -/
theorem wendy_full_face_time :
  full_face_time 5 5 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_wendy_full_face_time_l2042_204237


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2042_204262

/-- Given a cycle bought for a certain price with a specific gain percent,
    calculate the selling price. -/
def selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for Rs. 675 with a 60% gain is Rs. 1080 -/
theorem cycle_selling_price :
  selling_price 675 60 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2042_204262


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l2042_204214

theorem concert_ticket_cost : ∀ (adult_price : ℝ),
  (adult_price > 0) →
  (3 * adult_price + 5 * (adult_price / 3) = 21) →
  (7 * adult_price + 4 * (adult_price / 3) = 37.5) :=
λ adult_price adult_price_positive eq_21 =>
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l2042_204214


namespace NUMINAMATH_CALUDE_inequality_range_l2042_204239

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2042_204239


namespace NUMINAMATH_CALUDE_gift_cost_per_parent_l2042_204271

-- Define the given values
def total_spent : ℝ := 150
def siblings_count : ℕ := 3
def cost_per_sibling : ℝ := 30

-- Define the theorem
theorem gift_cost_per_parent :
  let spent_on_siblings := siblings_count * cost_per_sibling
  let spent_on_parents := total_spent - spent_on_siblings
  let cost_per_parent := spent_on_parents / 2
  cost_per_parent = 30 := by sorry

end NUMINAMATH_CALUDE_gift_cost_per_parent_l2042_204271


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l2042_204221

theorem missing_number_in_mean (numbers : List ℕ) (missing : ℕ) : 
  numbers = [1, 22, 23, 24, 25, 26, 27] →
  numbers.length = 7 →
  (numbers.sum + missing) / 8 = 20 →
  missing = 12 := by
sorry

end NUMINAMATH_CALUDE_missing_number_in_mean_l2042_204221


namespace NUMINAMATH_CALUDE_max_k_value_l2042_204295

/-- The maximum value of k satisfying the given inequality -/
theorem max_k_value (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 1) :
  (∃ k : ℝ, ∀ a b c : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 1 →
    (a / (1 + 9*b*c + k*(b-c)^2)) + (b / (1 + 9*c*a + k*(c-a)^2)) + (c / (1 + 9*a*b + k*(a-b)^2)) ≥ 1/2) ∧
  (∀ k' : ℝ, k' > 4 →
    ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
      (a / (1 + 9*b*c + k'*(b-c)^2)) + (b / (1 + 9*c*a + k'*(c-a)^2)) + (c / (1 + 9*a*b + k'*(a-b)^2)) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2042_204295


namespace NUMINAMATH_CALUDE_milk_cartons_problem_l2042_204208

theorem milk_cartons_problem (total : ℕ) (ratio : ℚ) : total = 24 → ratio = 7/1 → ∃ regular : ℕ, regular = 3 ∧ regular * (1 + ratio) = total := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_problem_l2042_204208


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2042_204230

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 1 = 0) → (x₂^2 + 5*x₂ - 1 = 0) → (x₁ + x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2042_204230


namespace NUMINAMATH_CALUDE_weak_coffee_amount_is_one_l2042_204265

/-- The amount of coffee used per cup of water for weak coffee -/
def weak_coffee_amount : ℝ := 1

/-- The number of cups of each type of coffee made -/
def cups_per_type : ℕ := 12

/-- The total amount of coffee used in tablespoons -/
def total_coffee : ℕ := 36

/-- Theorem stating that the amount of coffee used per cup of water for weak coffee is 1 tablespoon -/
theorem weak_coffee_amount_is_one :
  weak_coffee_amount = 1 ∧
  cups_per_type * weak_coffee_amount + cups_per_type * (2 * weak_coffee_amount) = total_coffee :=
by sorry

end NUMINAMATH_CALUDE_weak_coffee_amount_is_one_l2042_204265


namespace NUMINAMATH_CALUDE_parabola_chord_constant_l2042_204288

/-- Given a parabola y = 2x^2 and a point C(0, c), if t = 1/AC + 1/BC is constant
    for all chords AB passing through C, then t = -20/(7c) -/
theorem parabola_chord_constant (c : ℝ) :
  let parabola := fun (x : ℝ) => 2 * x^2
  let C := (0, c)
  let chord_length (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  ∃ t : ℝ, ∀ A B : ℝ × ℝ,
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    (∃ m b : ℝ, ∀ x : ℝ, m * x + b = parabola x ↔ (x = A.1 ∨ x = B.1)) →
    C.2 = m * C.1 + b →
    t = 1 / chord_length A C + 1 / chord_length B C →
    t = -20 / (7 * c) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_chord_constant_l2042_204288


namespace NUMINAMATH_CALUDE_tree_planting_participants_l2042_204259

theorem tree_planting_participants : ∃ (x y : ℕ), 
  x * y = 2013 ∧ 
  (x - 5) * (y + 2) < 2013 ∧ 
  (x - 5) * (y + 3) > 2013 ∧ 
  x = 61 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_participants_l2042_204259


namespace NUMINAMATH_CALUDE_mark_fruit_count_l2042_204269

/-- The number of apples Mark has chosen -/
def num_apples : ℕ := 3

/-- The number of bananas in the bunch Mark has selected -/
def num_bananas : ℕ := 4

/-- The number of oranges Mark needs to pick out -/
def num_oranges : ℕ := 5

/-- The total number of pieces of fruit Mark is looking to buy -/
def total_fruit : ℕ := num_apples + num_bananas + num_oranges

theorem mark_fruit_count : total_fruit = 12 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_count_l2042_204269


namespace NUMINAMATH_CALUDE_profit_percent_l2042_204250

theorem profit_percent (P : ℝ) (C : ℝ) (h1 : C > 0) (h2 : (2/3) * P = 0.88 * C) :
  (P - C) / C = 0.32 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_l2042_204250


namespace NUMINAMATH_CALUDE_special_quad_integer_area_iff_conditions_l2042_204292

/-- A quadrilateral ABCD with special properties -/
structure SpecialQuad where
  AB : ℝ
  CD : ℝ
  -- AB ⊥ BC and BC ⊥ CD
  perpendicular : True
  -- BC is tangent to a circle centered at O
  tangent : True
  -- AD is the diameter of the circle
  diameter : True

/-- The area of the special quadrilateral is an integer -/
def has_integer_area (q : SpecialQuad) : Prop :=
  ∃ n : ℕ, (q.AB + q.CD) * Real.sqrt (q.AB * q.CD) = n

/-- The product of AB and CD is a perfect square -/
def is_perfect_square_product (q : SpecialQuad) : Prop :=
  ∃ m : ℕ, q.AB * q.CD = m^2

theorem special_quad_integer_area_iff_conditions (q : SpecialQuad) :
  has_integer_area q ↔ is_perfect_square_product q ∧ has_integer_area q :=
sorry

end NUMINAMATH_CALUDE_special_quad_integer_area_iff_conditions_l2042_204292


namespace NUMINAMATH_CALUDE_fraction_division_equality_l2042_204229

theorem fraction_division_equality : 
  (-1/42 : ℚ) / ((1/6 : ℚ) - (3/14 : ℚ) + (2/3 : ℚ) - (2/7 : ℚ)) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l2042_204229


namespace NUMINAMATH_CALUDE_eight_jaguars_arrangement_l2042_204274

/-- The number of ways to arrange n different objects in a line -/
def linearArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n different objects in a line with the largest and smallest at the ends -/
def arrangementsWithExtremes (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 2)

/-- Theorem: There are 1440 ways to arrange 8 different objects in a line with the largest and smallest at the ends -/
theorem eight_jaguars_arrangement :
  arrangementsWithExtremes 8 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_eight_jaguars_arrangement_l2042_204274


namespace NUMINAMATH_CALUDE_profit_percentage_l2042_204228

/-- If the cost price of 72 articles equals the selling price of 60 articles, then the percent profit is 20%. -/
theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2042_204228


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2042_204242

def q (x : ℝ) : ℝ := -x^3 + 2*x^2 + 3*x

theorem q_satisfies_conditions :
  (q 3 = 0) ∧ 
  (q (-1) = 0) ∧ 
  (∃ (a b c : ℝ), ∀ x, q x = a*x^3 + b*x^2 + c*x) ∧
  (q 4 = -20) := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2042_204242


namespace NUMINAMATH_CALUDE_no_real_solutions_greater_than_one_l2042_204235

theorem no_real_solutions_greater_than_one :
  ∀ x : ℝ, x > 1 → (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 22 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_greater_than_one_l2042_204235


namespace NUMINAMATH_CALUDE_first_same_side_proof_l2042_204252

/-- The time when two points moving on a square are first on the same side --/
def first_same_side (side_length : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  25

theorem first_same_side_proof (side_length speed_A speed_B : ℝ) 
  (h1 : side_length = 100)
  (h2 : speed_A = 5)
  (h3 : speed_B = 10) :
  first_same_side side_length speed_A speed_B = 25 := by
  sorry

end NUMINAMATH_CALUDE_first_same_side_proof_l2042_204252


namespace NUMINAMATH_CALUDE_triangle_problem_l2042_204225

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (2 * Real.sin t.B * Real.cos t.A = Real.sin (t.A + t.C)) →  -- Given condition
  (t.BC = 2) →  -- Given condition
  (1/2 * t.AB * t.AC * Real.sin t.A = Real.sqrt 3) →  -- Area condition
  (t.A = Real.pi / 3 ∧ t.AB = 2) := by  -- Conclusion
sorry  -- Proof is omitted

end NUMINAMATH_CALUDE_triangle_problem_l2042_204225


namespace NUMINAMATH_CALUDE_horner_method_v3_l2042_204244

def horner_polynomial (x : ℝ) : ℝ := 1 + 5*x + 10*x^2 + 10*x^3 + 5*x^4 + x^5

def horner_v1 (x : ℝ) : ℝ := x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 10

theorem horner_method_v3 :
  horner_v3 (-2) = 2 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2042_204244


namespace NUMINAMATH_CALUDE_courtyard_length_courtyard_length_is_25_l2042_204298

/-- Proves that the length of a rectangular courtyard is 25 meters -/
theorem courtyard_length : ℝ → ℝ → ℝ → ℝ → Prop :=
  λ (width : ℝ) (num_bricks : ℝ) (brick_length : ℝ) (brick_width : ℝ) =>
    width = 16 ∧
    num_bricks = 20000 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 →
    (num_bricks * brick_length * brick_width) / width = 25

/-- The length of the courtyard is 25 meters -/
theorem courtyard_length_is_25 :
  courtyard_length 16 20000 0.2 0.1 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_courtyard_length_is_25_l2042_204298


namespace NUMINAMATH_CALUDE_academy_entrance_problem_l2042_204205

theorem academy_entrance_problem (rect1 rect2 rect3 rect4 : ℝ) 
  (h1 : rect1 = 20)
  (h2 : rect2 = 40)
  (h3 : rect3 = 48)
  (h4 : rect4 = 42) :
  rect1 + rect2 + rect3 + rect4 = 150 :=
by sorry

end NUMINAMATH_CALUDE_academy_entrance_problem_l2042_204205


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2042_204222

def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | x^2 - x - 6 = 0}

theorem intersection_of_P_and_Q : P ∩ Q = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2042_204222


namespace NUMINAMATH_CALUDE_cubic_inequality_l2042_204267

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 - 7*x + 10 > 0 ↔ x < -2 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2042_204267


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2042_204203

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -4982 [ZMOD 9] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2042_204203


namespace NUMINAMATH_CALUDE_fraction_lower_bound_l2042_204209

theorem fraction_lower_bound (p q r s : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_lower_bound_l2042_204209


namespace NUMINAMATH_CALUDE_seeking_cause_is_necessary_condition_l2042_204280

/-- "Seeking the cause from the effect" in analytical proof -/
def seeking_cause_from_effect : Prop := sorry

/-- Necessary condition in a proposition -/
def necessary_condition : Prop := sorry

/-- Theorem stating that "seeking the cause from the effect" refers to seeking the necessary condition -/
theorem seeking_cause_is_necessary_condition : 
  seeking_cause_from_effect ↔ necessary_condition := by sorry

end NUMINAMATH_CALUDE_seeking_cause_is_necessary_condition_l2042_204280


namespace NUMINAMATH_CALUDE_cubic_square_inequality_l2042_204245

theorem cubic_square_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / (a^2 + b^2) ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_inequality_l2042_204245


namespace NUMINAMATH_CALUDE_quiz_min_correct_answers_l2042_204289

theorem quiz_min_correct_answers 
  (total_questions : ℕ) 
  (points_correct : ℕ) 
  (points_incorrect : ℕ) 
  (target_score : ℕ) 
  (min_correct : ℕ) :
  total_questions = 20 →
  points_correct = 10 →
  points_incorrect = 4 →
  target_score = 88 →
  min_correct = 12 →
  (∀ x : ℕ, x ≥ min_correct ↔ 
    points_correct * x - points_incorrect * (total_questions - x) ≥ target_score) :=
by sorry

end NUMINAMATH_CALUDE_quiz_min_correct_answers_l2042_204289


namespace NUMINAMATH_CALUDE_anglets_in_sixth_circle_is_6000_l2042_204241

-- Define constants
def full_circle_degrees : ℕ := 360
def anglets_per_degree : ℕ := 100

-- Define the number of anglets in a sixth of a circle
def anglets_in_sixth_circle : ℕ := (full_circle_degrees / 6) * anglets_per_degree

-- Theorem statement
theorem anglets_in_sixth_circle_is_6000 : anglets_in_sixth_circle = 6000 := by
  sorry

end NUMINAMATH_CALUDE_anglets_in_sixth_circle_is_6000_l2042_204241


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l2042_204294

theorem discriminant_of_specific_quadratic (a b c : ℝ) : 
  a = 1 → b = -2 → c = 1 → b^2 - 4*a*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l2042_204294


namespace NUMINAMATH_CALUDE_cost_of_mangos_rice_flour_l2042_204275

/-- The cost of mangos, rice, and flour given certain price relationships -/
theorem cost_of_mangos_rice_flour (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 21) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 237.3 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_mangos_rice_flour_l2042_204275


namespace NUMINAMATH_CALUDE_complex_i_plus_i_squared_l2042_204249

theorem complex_i_plus_i_squared : ∃ (i : ℂ), i * i = -1 ∧ i + i * i = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_i_plus_i_squared_l2042_204249


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2042_204293

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2042_204293


namespace NUMINAMATH_CALUDE_smallest_a_for_coeff_70_l2042_204256

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 1512 * a + 4725

/-- The problem statement -/
theorem smallest_a_for_coeff_70 :
  ∃ a : ℝ, (∀ b : ℝ, coeff_x4 b = 70 → a ≤ b) ∧ coeff_x4 a = 70 ∧ a = -50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_coeff_70_l2042_204256


namespace NUMINAMATH_CALUDE_extreme_points_sum_l2042_204231

/-- Given that x = 2 and x = -4 are extreme points of f(x) = x³ + px² + qx, prove that p + q = -21 -/
theorem extreme_points_sum (p q : ℝ) : 
  (∀ x : ℝ, x = 2 ∨ x = -4 → (3*x^2 + 2*p*x + q = 0)) → 
  p + q = -21 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_sum_l2042_204231


namespace NUMINAMATH_CALUDE_circle_k_range_l2042_204260

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

/-- The condition for the equation to represent a circle --/
def is_circle (k : ℝ) : Prop :=
  ∃ (h c r : ℝ), ∀ (x y : ℝ), circle_equation x y k ↔ (x - h)^2 + (y - c)^2 = r^2 ∧ r > 0

/-- The theorem stating the range of k for which the equation represents a circle --/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k > 4 ∨ k < -1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l2042_204260


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l2042_204272

/-- Triangle ABC with given side lengths and angle --/
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

/-- The number of solutions for a triangle with given side lengths and angle --/
def numSolutions (t : Triangle) : ℕ :=
  sorry

/-- Theorem stating that a triangle with a = 18, b = 24, and A = 30° has exactly two solutions --/
theorem triangle_two_solutions :
  ∀ t : Triangle, t.a = 18 ∧ t.b = 24 ∧ t.A = 30 * π / 180 → numSolutions t = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l2042_204272


namespace NUMINAMATH_CALUDE_equiangular_hexagon_side_lengths_l2042_204284

/-- An equiangular hexagon is a hexagon where all internal angles are equal. -/
structure EquiangularHexagon where
  sides : Fin 6 → ℝ
  is_equiangular : True  -- This is a placeholder for the equiangular property

/-- The theorem stating the side lengths of a specific equiangular hexagon -/
theorem equiangular_hexagon_side_lengths 
  (h : EquiangularHexagon) 
  (h1 : h.sides 0 = 3)
  (h2 : h.sides 2 = 5)
  (h3 : h.sides 3 = 4)
  (h4 : h.sides 4 = 1) :
  h.sides 5 = 6 ∧ h.sides 1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_equiangular_hexagon_side_lengths_l2042_204284


namespace NUMINAMATH_CALUDE_f_properties_l2042_204211

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a < 0 → ∀ x : ℝ, 0 < x → f a x > 0) ∧
  (0 < a → ∀ x : ℝ, 0 < x → (f a x > 0 ↔ x < 2*a)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2042_204211


namespace NUMINAMATH_CALUDE_bus_problem_l2042_204263

/-- The number of people on a bus after a stop, given the original number and the difference between those who left and those who got on. -/
def peopleOnBusAfterStop (originalCount : ℕ) (exitEnterDifference : ℕ) : ℕ :=
  originalCount - exitEnterDifference

/-- Theorem stating that given the initial conditions, the number of people on the bus after the stop is 29. -/
theorem bus_problem :
  peopleOnBusAfterStop 38 9 = 29 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2042_204263


namespace NUMINAMATH_CALUDE_correct_calculation_l2042_204233

theorem correct_calculation (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2042_204233


namespace NUMINAMATH_CALUDE_root_equation_value_l2042_204224

theorem root_equation_value (m : ℝ) : m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2042_204224


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l2042_204285

theorem tracy_candies_problem :
  ∃ (initial : ℕ) (brother_took : ℕ),
    initial > 0 ∧
    brother_took ≥ 2 ∧
    brother_took ≤ 6 ∧
    (3 * initial / 10 : ℚ) - 20 - brother_took = 6 ∧
    initial = 100 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l2042_204285


namespace NUMINAMATH_CALUDE_sara_letters_count_l2042_204247

/-- The number of letters Sara sent in January. -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February. -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March. -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent over three months. -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count : total_letters = 33 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_count_l2042_204247


namespace NUMINAMATH_CALUDE_scientific_notation_120000_l2042_204273

theorem scientific_notation_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_120000_l2042_204273


namespace NUMINAMATH_CALUDE_solvable_iff_edge_start_l2042_204215

/-- Represents a cell on the 4x4 board -/
inductive Cell
| corner : Cell
| edge : Cell
| center : Cell

/-- Represents the state of the board -/
structure Board :=
(empty_cell : Cell)
(stones : Nat)

/-- Defines a valid move on the board -/
def valid_move (b : Board) : Prop :=
  b.stones > 1 ∧ ∃ (new_empty : Cell), new_empty ≠ b.empty_cell

/-- Defines the final state with one stone -/
def final_state (b : Board) : Prop :=
  b.stones = 1

/-- The main theorem to prove -/
theorem solvable_iff_edge_start :
  ∀ (b : Board),
    (b.empty_cell = Cell.edge ∧ b.stones = 15) ↔
    (∃ (b_final : Board), 
      final_state b_final ∧
      (∃ (moves : Nat), ∀ (i : Nat), i < moves → valid_move (Board.mk b.empty_cell (b.stones - i)))) :=
sorry

end NUMINAMATH_CALUDE_solvable_iff_edge_start_l2042_204215


namespace NUMINAMATH_CALUDE_circle_in_rectangle_l2042_204278

theorem circle_in_rectangle (rectangle_side : Real) (circle_area : Real) : 
  rectangle_side = 14 →
  circle_area = 153.93804002589985 →
  (circle_area = π * (rectangle_side / 2)^2) →
  rectangle_side = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_l2042_204278


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2042_204236

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 1650) (h4 : 0.075 * x = 0.125 * y) : y = 2475 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2042_204236


namespace NUMINAMATH_CALUDE_absent_men_count_solve_work_scenario_l2042_204202

/-- Represents the work scenario with absences -/
structure WorkScenario where
  total_men : ℕ
  original_days : ℕ
  actual_days : ℕ
  absent_men : ℕ

/-- Calculates the total work in man-days -/
def total_work (s : WorkScenario) : ℕ := s.total_men * s.original_days

/-- Calculates the work done by remaining men -/
def remaining_work (s : WorkScenario) : ℕ := (s.total_men - s.absent_men) * s.actual_days

/-- Theorem stating that 8 men became absent -/
theorem absent_men_count (s : WorkScenario) 
  (h1 : s.total_men = 48)
  (h2 : s.original_days = 15)
  (h3 : s.actual_days = 18)
  (h4 : total_work s = remaining_work s) :
  s.absent_men = 8 := by
  sorry

/-- Main theorem proving the solution -/
theorem solve_work_scenario : 
  ∃ (s : WorkScenario), s.total_men = 48 ∧ s.original_days = 15 ∧ s.actual_days = 18 ∧ 
  total_work s = remaining_work s ∧ s.absent_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_absent_men_count_solve_work_scenario_l2042_204202


namespace NUMINAMATH_CALUDE_inequality_proof_l2042_204258

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2042_204258


namespace NUMINAMATH_CALUDE_hat_problem_l2042_204210

/-- The number of customers -/
def n : ℕ := 5

/-- The probability that no customer gets their own hat when n customers randomly take hats -/
def prob_no_own_hat (n : ℕ) : ℚ :=
  sorry

theorem hat_problem : prob_no_own_hat n = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_hat_problem_l2042_204210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2042_204268

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (a 1 = 1) →
  (∀ k : ℕ, a (k + 1) - a k = 3) →
  (a n = 298) →
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2042_204268


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l2042_204257

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 1 ∨ 2 < x ∧ x < 3} := by sorry

-- Theorem for (C_U A) ∪ B
theorem union_complement_A_B : (Set.compl A) ∪ B = {x | x < 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l2042_204257


namespace NUMINAMATH_CALUDE_count_hollow_circles_l2042_204270

/-- The length of the repeating sequence of circles -/
def sequence_length : ℕ := 24

/-- The number of hollow circles in each repetition of the sequence -/
def hollow_circles_per_sequence : ℕ := 5

/-- The total number of circles we're considering -/
def total_circles : ℕ := 2003

/-- The number of hollow circles in the first 2003 circles -/
def hollow_circles_count : ℕ := 446

theorem count_hollow_circles :
  (total_circles / sequence_length) * hollow_circles_per_sequence +
  (hollow_circles_per_sequence * (total_circles % sequence_length) / sequence_length) =
  hollow_circles_count :=
sorry

end NUMINAMATH_CALUDE_count_hollow_circles_l2042_204270


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2042_204291

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 3*p - 2 = 0) → 
  (q^3 - 3*q - 2 = 0) → 
  (r^3 - 3*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2042_204291


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l2042_204243

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l2042_204243


namespace NUMINAMATH_CALUDE_coefficient_b_is_zero_l2042_204283

/-- Given an equation px + qy + bz = 1 with three solutions, prove that b = 0 -/
theorem coefficient_b_is_zero
  (p q b a : ℝ)
  (h1 : q * (3 * a) + b * 1 = 1)
  (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
  (h3 : q * (3 * a) = 1) :
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_b_is_zero_l2042_204283


namespace NUMINAMATH_CALUDE_incorrect_proposition_l2042_204255

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (different : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : m ≠ n)
  : ¬(parallel_lines m n ∧ intersect α β m →
      parallel_line_plane n α ∧ parallel_line_plane n β) :=
sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l2042_204255


namespace NUMINAMATH_CALUDE_zachs_bike_savings_l2042_204277

/-- Represents the problem of calculating how much more money Zach needs to earn --/
theorem zachs_bike_savings (bike_cost : ℕ) (discount_rate : ℚ) 
  (weekly_allowance : ℕ) (lawn_mowing_min lawn_mowing_max : ℕ) 
  (garage_cleaning : ℕ) (babysitting_rate babysitting_hours : ℕ) 
  (loan_to_repay : ℕ) (current_savings : ℕ) : 
  bike_cost = 150 →
  discount_rate = 1/10 →
  weekly_allowance = 5 →
  lawn_mowing_min = 8 →
  lawn_mowing_max = 12 →
  garage_cleaning = 15 →
  babysitting_rate = 7 →
  babysitting_hours = 3 →
  loan_to_repay = 10 →
  current_savings = 65 →
  ∃ (additional_money : ℕ), additional_money = 27 ∧ 
    (bike_cost - (discount_rate * bike_cost).floor) - current_savings + loan_to_repay = 
    weekly_allowance + lawn_mowing_max + garage_cleaning + (babysitting_rate * babysitting_hours) + additional_money :=
by sorry

end NUMINAMATH_CALUDE_zachs_bike_savings_l2042_204277


namespace NUMINAMATH_CALUDE_last_page_stamps_l2042_204286

/-- The number of stamp books Jenny originally has -/
def num_books : ℕ := 8

/-- The number of pages in each stamp book -/
def pages_per_book : ℕ := 42

/-- The number of stamps on each page originally -/
def stamps_per_page_original : ℕ := 6

/-- The number of stamps on each page after reorganization -/
def stamps_per_page_new : ℕ := 10

/-- The number of completely filled books after reorganization -/
def filled_books : ℕ := 4

/-- The number of completely filled pages in the partially filled book -/
def filled_pages_partial : ℕ := 33

theorem last_page_stamps :
  (num_books * pages_per_book * stamps_per_page_original) % stamps_per_page_new = 6 :=
sorry

end NUMINAMATH_CALUDE_last_page_stamps_l2042_204286


namespace NUMINAMATH_CALUDE_quadratic_roots_not_straddling_1000_l2042_204204

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The roots of a quadratic function -/
def roots (a b c : ℝ) : Set ℝ := {x : ℝ | quadratic a b c x = 0}

theorem quadratic_roots_not_straddling_1000 (a b c : ℝ) (h : a ≠ 0) :
  quadratic a b c 1000 > 0 →
  ¬∃ (r₁ r₂ : ℝ), r₁ ∈ roots a b c ∧ r₂ ∈ roots a b c ∧ r₁ < 1000 ∧ r₂ > 1000 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_not_straddling_1000_l2042_204204


namespace NUMINAMATH_CALUDE_line_equation_specific_l2042_204279

/-- The equation of a line with given slope and y-intercept -/
def line_equation (slope : ℝ) (y_intercept : ℝ) : ℝ → ℝ := λ x => slope * x + y_intercept

/-- Theorem: The equation of a line with slope 2 and y-intercept 1 is y = 2x + 1 -/
theorem line_equation_specific : line_equation 2 1 = λ x => 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_specific_l2042_204279


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2042_204213

theorem simplify_and_rationalize (x : ℝ) : 
  (1 : ℝ) / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2042_204213


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_equals_sqrt_two_l2042_204251

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_equals_sqrt_two :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_equals_sqrt_two_l2042_204251


namespace NUMINAMATH_CALUDE_split_meal_cost_l2042_204206

def meal_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

theorem split_meal_cost :
  meal_contribution 67 4 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_split_meal_cost_l2042_204206


namespace NUMINAMATH_CALUDE_fourth_sample_is_75_l2042_204254

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (nthSample : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (nthSample - 1)

/-- Theorem: In a systematic sampling scheme with a population of 480, a sample size of 20, 
    and a first sample of 3, the fourth sample will be 75 -/
theorem fourth_sample_is_75 :
  systematicSample 480 20 3 4 = 75 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_75_l2042_204254


namespace NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l2042_204238

theorem polygon_sides_when_angles_equal : ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l2042_204238


namespace NUMINAMATH_CALUDE_expression_simplification_l2042_204246

theorem expression_simplification (b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  1/2 - 1/(1 + b/(1 - 2*b)) = (3*b - 1)/(2*(1 - b)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2042_204246


namespace NUMINAMATH_CALUDE_function_inequality_l2042_204234

def RealFunction := ℝ → ℝ

def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def IsMonotoneIncreasingOnNonnegatives (f : RealFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem function_inequality (f : RealFunction) 
  (h_even : IsEven f)
  (h_monotone : IsMonotoneIncreasingOnNonnegatives f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2042_204234


namespace NUMINAMATH_CALUDE_line_through_point_l2042_204219

theorem line_through_point (a : ℚ) : 
  (3 * a * 2 + (2 * a + 3) * (-5) = 4 * a + 6) → a = -21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2042_204219
