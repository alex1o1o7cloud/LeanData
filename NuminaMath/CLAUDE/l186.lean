import Mathlib

namespace NUMINAMATH_CALUDE_angle_sum_theorem_l186_18607

theorem angle_sum_theorem (x : ℝ) : 
  (6 * x + 7 * x + 3 * x + 4 * x) * (π / 180) = 2 * π → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l186_18607


namespace NUMINAMATH_CALUDE_tomatoes_for_sale_tuesday_l186_18619

/-- Calculates the amount of tomatoes ready for sale on Tuesday given specific conditions --/
theorem tomatoes_for_sale_tuesday 
  (initial_shipment : ℝ)
  (saturday_selling_rate : ℝ)
  (sunday_spoilage_rate : ℝ)
  (monday_shipment_multiplier : ℝ)
  (monday_selling_rate : ℝ)
  (tuesday_spoilage_rate : ℝ)
  (h1 : initial_shipment = 1000)
  (h2 : saturday_selling_rate = 0.6)
  (h3 : sunday_spoilage_rate = 0.2)
  (h4 : monday_shipment_multiplier = 1.5)
  (h5 : monday_selling_rate = 0.4)
  (h6 : tuesday_spoilage_rate = 0.15) :
  ∃ (tomatoes_tuesday : ℝ), tomatoes_tuesday = 928.2 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_for_sale_tuesday_l186_18619


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l186_18612

theorem coin_and_die_probability : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads
  let d : ℕ := 6   -- number of sides on the die
  let p_coin : ℚ := 1/2  -- probability of heads on a fair coin
  let p_die : ℚ := 1/d  -- probability of rolling a 6 on a fair die
  (Nat.choose n k * p_coin^k * (1 - p_coin)^(n - k)) * p_die = 55/6144 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l186_18612


namespace NUMINAMATH_CALUDE_ship_ratio_l186_18633

/-- Given the conditions of ships in a busy port, prove the ratio of sailboats to fishing boats -/
theorem ship_ratio : 
  ∀ (cruise cargo sailboats fishing : ℕ),
  cruise = 4 →
  cargo = 2 * cruise →
  sailboats = cargo + 6 →
  cruise + cargo + sailboats + fishing = 28 →
  sailboats / fishing = 7 ∧ fishing ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ship_ratio_l186_18633


namespace NUMINAMATH_CALUDE_tan_60_minus_sin_60_l186_18652

theorem tan_60_minus_sin_60 : Real.tan (π / 3) - Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_60_minus_sin_60_l186_18652


namespace NUMINAMATH_CALUDE_min_value_of_a_l186_18663

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := 
  fun x => if x > 0 then Real.exp x + a else -(Real.exp (-x) + a)

-- State the theorem
theorem min_value_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) →  -- f is odd
  (∀ x y : ℝ, x < y → f a x < f a y) →  -- f is strictly increasing (monotonic)
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x < f b y) → 
            b ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l186_18663


namespace NUMINAMATH_CALUDE_fraction_value_l186_18631

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l186_18631


namespace NUMINAMATH_CALUDE_equation_solution_l186_18644

theorem equation_solution (a b : ℝ) :
  b ≠ 0 →
  (∀ x, (4 * a * x + 1) / b - 5 = 3 * x / b) ↔
  (b = 0 ∧ False) ∨
  (a = 3/4 ∧ b = 1/5) ∨
  (4 * a - 3 ≠ 0 ∧ ∃! x, x = (5 * b - 1) / (4 * a - 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l186_18644


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l186_18625

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 1)
  (h2 : Real.sin a + Real.sin b = 1/2)
  (h3 : Real.tan (a - b) = 1) :
  Real.tan ((a + b) / 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l186_18625


namespace NUMINAMATH_CALUDE_quadratic_polynomial_roots_l186_18646

theorem quadratic_polynomial_roots (x y : ℝ) (t : ℝ → ℝ) : 
  x + y = 12 → x * (3 * y) = 108 → 
  (∀ r, t r = 0 ↔ r = x ∨ r = y) → 
  t = fun r ↦ r^2 - 12*r + 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_roots_l186_18646


namespace NUMINAMATH_CALUDE_dance_cost_theorem_l186_18677

/-- Represents the cost calculation for dance shoes and fans. -/
structure DanceCost where
  x : ℝ  -- Number of fans per pair of shoes
  yA : ℝ -- Cost at supermarket A
  yB : ℝ -- Cost at supermarket B

/-- Calculates the cost for dance shoes and fans given the conditions. -/
def calculate_cost (x : ℝ) : DanceCost :=
  { x := x
  , yA := 27 * x + 270
  , yB := 30 * x + 240 }

/-- Theorem stating the relationship between costs and number of fans. -/
theorem dance_cost_theorem (x : ℝ) (h : x ≥ 2) :
  let cost := calculate_cost x
  cost.yA = 27 * x + 270 ∧
  cost.yB = 30 * x + 240 ∧
  (x < 10 → cost.yB < cost.yA) ∧
  (x = 10 → cost.yB = cost.yA) ∧
  (x > 10 → cost.yA < cost.yB) := by
  sorry

#check dance_cost_theorem

end NUMINAMATH_CALUDE_dance_cost_theorem_l186_18677


namespace NUMINAMATH_CALUDE_max_positive_integer_solution_of_inequality_system_l186_18639

theorem max_positive_integer_solution_of_inequality_system :
  ∃ (x : ℝ), (3 * x - 1 > x + 1) ∧ ((4 * x - 5) / 3 ≤ x) ∧
  (∀ (y : ℤ), (3 * y - 1 > y + 1) ∧ ((4 * y - 5) / 3 ≤ y) → y ≤ 5) ∧
  (3 * 5 - 1 > 5 + 1) ∧ ((4 * 5 - 5) / 3 ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_max_positive_integer_solution_of_inequality_system_l186_18639


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l186_18609

theorem real_root_of_cubic (a b c : ℂ) (h_a_real : a.im = 0)
  (h_sum : a + b + c = 5)
  (h_sum_prod : a * b + b * c + c * a = 7)
  (h_prod : a * b * c = 2) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l186_18609


namespace NUMINAMATH_CALUDE_min_white_points_l186_18694

theorem min_white_points (total_points : ℕ) (h_total : total_points = 100) :
  ∃ (n : ℕ), n = 10 ∧ 
  (∀ (k : ℕ), k < n → k + (k.choose 3) < total_points) ∧
  (n + (n.choose 3) ≥ total_points) := by
  sorry

end NUMINAMATH_CALUDE_min_white_points_l186_18694


namespace NUMINAMATH_CALUDE_probability_three_heads_twelve_coins_l186_18690

theorem probability_three_heads_twelve_coins : 
  (Nat.choose 12 3 : ℚ) / (2^12 : ℚ) = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_twelve_coins_l186_18690


namespace NUMINAMATH_CALUDE_fraction_simplification_l186_18616

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l186_18616


namespace NUMINAMATH_CALUDE_corresponding_time_l186_18643

-- Define the ratio
def ratio : ℚ := 8 / 4

-- Define the conversion factor from seconds to minutes
def seconds_to_minutes : ℚ := 1 / 60

-- State the theorem
theorem corresponding_time (t : ℚ) : 
  ratio = 8 / t → t = 4 * seconds_to_minutes :=
by sorry

end NUMINAMATH_CALUDE_corresponding_time_l186_18643


namespace NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l186_18688

theorem x_minus_25_is_perfect_square (n : ℕ) :
  let x := 10^(2*n + 4) + 10^(n + 3) + 50
  ∃ k : ℕ, x - 25 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l186_18688


namespace NUMINAMATH_CALUDE_robins_hair_length_l186_18649

/-- Given that Robin's hair is currently 13 inches long after cutting off 4 inches,
    prove that his initial hair length was 17 inches. -/
theorem robins_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 13)
  (h2 : cut_length = 4) : 
  current_length + cut_length = 17 := by
sorry

end NUMINAMATH_CALUDE_robins_hair_length_l186_18649


namespace NUMINAMATH_CALUDE_function_composition_identity_l186_18636

/-- Piecewise function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 10 - 4 * x

/-- Theorem stating that if f(f(x)) = x for all x, then a + b = 21/4 -/
theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_identity_l186_18636


namespace NUMINAMATH_CALUDE_bottle_cap_calculation_l186_18693

theorem bottle_cap_calculation (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_calculation_l186_18693


namespace NUMINAMATH_CALUDE_additional_students_l186_18620

theorem additional_students (initial_students : ℕ) (students_per_computer : ℕ) (target_computers : ℕ) : 
  initial_students = 82 →
  students_per_computer = 2 →
  target_computers = 49 →
  (initial_students + (target_computers - initial_students / students_per_computer) * students_per_computer) - initial_students = 16 := by
sorry

end NUMINAMATH_CALUDE_additional_students_l186_18620


namespace NUMINAMATH_CALUDE_swimmers_speed_l186_18604

/-- Proves that a swimmer's speed in still water is 12 km/h given the conditions -/
theorem swimmers_speed (v s : ℝ) (h1 : s = 4) (h2 : (v - s)⁻¹ = 2 * (v + s)⁻¹) : v = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_speed_l186_18604


namespace NUMINAMATH_CALUDE_polynomial_equality_conditions_l186_18632

theorem polynomial_equality_conditions (A B C p q : ℝ) :
  (∀ x : ℝ, A * x^4 + B * x^2 + C = A * (x^2 + p * x + q) * (x^2 - p * x + q)) →
  (A * (2 * q - p^2) = B ∧ A * q^2 = C) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_conditions_l186_18632


namespace NUMINAMATH_CALUDE_switches_in_position_a_after_process_l186_18667

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 4

/-- The set of all switches -/
def switches : Finset Switch :=
  sorry

/-- The process of advancing switches -/
def advance_switches : Finset Switch → Finset Switch :=
  sorry

/-- The final state after 729 steps -/
def final_state : Finset Switch :=
  sorry

/-- Count switches in position A -/
def count_position_a (s : Finset Switch) : Nat :=
  sorry

theorem switches_in_position_a_after_process :
  count_position_a final_state = 409 := by
  sorry

end NUMINAMATH_CALUDE_switches_in_position_a_after_process_l186_18667


namespace NUMINAMATH_CALUDE_equation_solution_l186_18618

theorem equation_solution (x : ℝ) : 
  (5 * x^2 - 3) / (x + 3) - 5 / (x + 3) = 6 / (x + 3) → 
  x = Real.sqrt 70 / 5 ∨ x = -Real.sqrt 70 / 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l186_18618


namespace NUMINAMATH_CALUDE_pizza_delivery_solution_l186_18680

/-- Represents the pizza delivery problem -/
def PizzaDelivery (total_pizzas : ℕ) (total_time : ℕ) (avg_time_per_stop : ℕ) : Prop :=
  ∃ (two_pizza_stops : ℕ),
    two_pizza_stops * 2 + (total_pizzas - two_pizza_stops * 2) = total_pizzas ∧
    (two_pizza_stops + (total_pizzas - two_pizza_stops * 2)) * avg_time_per_stop = total_time

/-- Theorem stating the solution to the pizza delivery problem -/
theorem pizza_delivery_solution :
  PizzaDelivery 12 40 4 → ∃ (two_pizza_stops : ℕ), two_pizza_stops = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_pizza_delivery_solution_l186_18680


namespace NUMINAMATH_CALUDE_daves_initial_apps_l186_18601

theorem daves_initial_apps (initial_files : ℕ) (apps_left : ℕ) (files_left : ℕ) : 
  initial_files = 24 →
  apps_left = 21 →
  files_left = 4 →
  apps_left = files_left + 17 →
  ∃ initial_apps : ℕ, initial_apps = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_daves_initial_apps_l186_18601


namespace NUMINAMATH_CALUDE_s_5_value_l186_18696

/-- s(n) is a number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that s(5) equals 1491625 -/
theorem s_5_value : s 5 = 1491625 :=
  sorry

end NUMINAMATH_CALUDE_s_5_value_l186_18696


namespace NUMINAMATH_CALUDE_adults_on_bicycles_l186_18637

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem: The number of adults riding bicycles is 6 -/
theorem adults_on_bicycles : 
  ∃ (a : ℕ), a * bicycle_wheels + children_on_tricycles * tricycle_wheels = total_wheels ∧ a = 6 :=
sorry

end NUMINAMATH_CALUDE_adults_on_bicycles_l186_18637


namespace NUMINAMATH_CALUDE_max_profits_l186_18634

def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

def average_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

theorem max_profits :
  (∃ (x_max : ℕ+), ∀ (x : ℕ+), total_profit x ≤ total_profit x_max ∧ 
    total_profit x_max = 45) ∧
  (∃ (x_avg_max : ℕ+), ∀ (x : ℕ+), average_annual_profit x ≤ average_annual_profit x_avg_max ∧ 
    average_annual_profit x_avg_max = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_profits_l186_18634


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l186_18628

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_specific_sequence :
  let a₁ : ℚ := 27
  let r : ℚ := 2/3
  geometric_sequence a₁ r 8 = 128/81 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l186_18628


namespace NUMINAMATH_CALUDE_rabbit_can_escape_l186_18621

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a square with side length 1 -/
structure Square where
  center : Point
  side_length : Real := 1

/-- Represents an entity (rabbit or wolf) with a position and speed -/
structure Entity where
  position : Point
  speed : Real

/-- Theorem stating that the rabbit can escape the square -/
theorem rabbit_can_escape (s : Square) (rabbit : Entity) (wolves : Finset Entity) :
  rabbit.position = s.center →
  wolves.card = 4 →
  (∀ w ∈ wolves, w.speed = 1.4 * rabbit.speed) →
  (∀ w ∈ wolves, w.position.x = 0 ∨ w.position.x = 1) →
  (∀ w ∈ wolves, w.position.y = 0 ∨ w.position.y = 1) →
  ∃ (escape_path : Real → Point),
    (escape_path 0 = rabbit.position) ∧
    (∃ t : Real, t > 0 ∧ (escape_path t).x = 0 ∨ (escape_path t).x = 1 ∨ (escape_path t).y = 0 ∨ (escape_path t).y = 1) ∧
    (∀ w ∈ wolves, ∀ t : Real, t ≥ 0 → 
      (escape_path t).x ≠ w.position.x ∨ (escape_path t).y ≠ w.position.y) :=
sorry

end NUMINAMATH_CALUDE_rabbit_can_escape_l186_18621


namespace NUMINAMATH_CALUDE_michael_matchsticks_l186_18623

/-- The number of matchstick houses Michael creates -/
def num_houses : ℕ := 30

/-- The number of matchsticks used per house -/
def matchsticks_per_house : ℕ := 10

/-- The total number of matchsticks Michael used -/
def total_matchsticks_used : ℕ := num_houses * matchsticks_per_house

/-- Michael's original number of matchsticks -/
def original_matchsticks : ℕ := 2 * total_matchsticks_used

theorem michael_matchsticks : original_matchsticks = 600 := by
  sorry

end NUMINAMATH_CALUDE_michael_matchsticks_l186_18623


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l186_18617

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l186_18617


namespace NUMINAMATH_CALUDE_a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l186_18614

theorem a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (a + Real.log a > b + Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l186_18614


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l186_18668

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  u : ℝ
  v : ℝ
  w : ℝ

/-- The theorem statement -/
theorem area_ratio_theorem (t : TriangleWithPoints) 
  (h_PQ : t.PQ = 12)
  (h_QR : t.QR = 16)
  (h_PR : t.PR = 20)
  (h_positive : t.u > 0 ∧ t.v > 0 ∧ t.w > 0)
  (h_sum : t.u + t.v + t.w = 3/4)
  (h_sum_squares : t.u^2 + t.v^2 + t.w^2 = 1/2) :
  let area_PQR := (1/2) * t.PQ * t.QR
  let area_XYZ := area_PQR * (1 - (t.u * (1 - t.w) + t.v * (1 - t.u) + t.w * (1 - t.v)))
  area_XYZ / area_PQR = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l186_18668


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l186_18669

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![2, 3; 0, -1]) : 
  (B^2)⁻¹ = !![4, 3; 0, 1] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l186_18669


namespace NUMINAMATH_CALUDE_max_distance_complex_l186_18645

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 81 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((1 + 2*I)*z^4 - z^6) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l186_18645


namespace NUMINAMATH_CALUDE_divisibility_by_nine_implies_divisibility_by_three_l186_18638

theorem divisibility_by_nine_implies_divisibility_by_three (u v : ℤ) :
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_implies_divisibility_by_three_l186_18638


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l186_18655

/-- Represents the configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : ℝ
  h_area_ratio : area_ratio = 9
  h_outer_side : outer_side = inner_side + 2 * rect_short
  h_rect_long : rect_long + rect_short = outer_side

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio_is_two (config : SquareRectConfig) :
  config.rect_long / config.rect_short = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l186_18655


namespace NUMINAMATH_CALUDE_tan_105_degrees_l186_18606

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l186_18606


namespace NUMINAMATH_CALUDE_number_and_square_sum_l186_18657

theorem number_and_square_sum (x : ℝ) : x + x^2 = 342 → x = 18 ∨ x = -19 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l186_18657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l186_18679

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- The theorem states that for an arithmetic sequence where a₄ = 2a₃, 
    the ratio of S₇ to S₅ is equal to 14/5 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (h : a 4 = 2 * a 3) : 
  S 7 a / S 5 a = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l186_18679


namespace NUMINAMATH_CALUDE_simon_treasures_l186_18692

/-- The number of sand dollars Simon collected -/
def sand_dollars : ℕ := sorry

/-- The number of sea glass pieces Simon collected -/
def sea_glass : ℕ := sorry

/-- The number of seashells Simon collected -/
def seashells : ℕ := sorry

/-- The total number of treasures Simon collected -/
def total_treasures : ℕ := 190

theorem simon_treasures : 
  sea_glass = 3 * sand_dollars ∧ 
  seashells = 5 * sea_glass ∧
  total_treasures = sand_dollars + sea_glass + seashells →
  sand_dollars = 10 := by sorry

end NUMINAMATH_CALUDE_simon_treasures_l186_18692


namespace NUMINAMATH_CALUDE_aquarium_solution_l186_18664

def aquarium_animals (otters seals sea_lions : ℕ) : Prop :=
  (otters + seals = 7 ∨ otters = 7 ∨ seals = 7) ∧
  (sea_lions + seals = 6 ∨ sea_lions = 6 ∨ seals = 6) ∧
  (otters + sea_lions = 5 ∨ otters = 5 ∨ sea_lions = 5) ∧
  (otters ≤ seals ∨ seals ≤ otters) ∧
  (otters ≤ sea_lions ∧ seals ≤ sea_lions)

theorem aquarium_solution :
  ∃! (otters seals sea_lions : ℕ),
    aquarium_animals otters seals sea_lions ∧
    otters = 5 ∧ seals = 7 ∧ sea_lions = 6 :=
sorry

end NUMINAMATH_CALUDE_aquarium_solution_l186_18664


namespace NUMINAMATH_CALUDE_omega_value_l186_18675

-- Define the complex numbers z and ω
variable (z ω : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the conditions
axiom pure_imaginary : ∃ (y : ℝ), (1 + 3 * i) * z = i * y
axiom omega_def : ω = z / (2 + i)
axiom omega_abs : Complex.abs ω = 5 * Real.sqrt 2

-- State the theorem to be proved
theorem omega_value : ω = 7 - i ∨ ω = -(7 - i) := by sorry

end NUMINAMATH_CALUDE_omega_value_l186_18675


namespace NUMINAMATH_CALUDE_binomial_square_example_l186_18676

theorem binomial_square_example : 16^2 + 2*(16*5) + 5^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_example_l186_18676


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l186_18672

/-- Given a line passing through (10, 3) with x-intercept 4,
    the x-coordinate of the point on this line with y-coordinate -3 is -2. -/
theorem line_point_x_coordinate 
  (line : ℝ → ℝ) 
  (passes_through_10_3 : line 10 = 3)
  (x_intercept_4 : line 4 = 0) :
  ∃ x : ℝ, line x = -3 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_coordinate_l186_18672


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l186_18666

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 35 square units and y > 0, then y = 7. -/
theorem rectangle_area_theorem (y : ℝ) : y > 0 → 5 * y = 35 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l186_18666


namespace NUMINAMATH_CALUDE_bride_groom_age_difference_l186_18671

theorem bride_groom_age_difference :
  ∀ (bride_age groom_age : ℕ),
    bride_age = 102 →
    bride_age + groom_age = 185 →
    bride_age - groom_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_difference_l186_18671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l186_18698

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = ∫ x in (0 : ℝ)..2, |1 - x^2|) →
  a 4 + a 6 + a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l186_18698


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l186_18651

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price = 560) :
  (selling_price - cost_price) / cost_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l186_18651


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l186_18648

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ * x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l186_18648


namespace NUMINAMATH_CALUDE_jack_payback_l186_18684

/-- The amount borrowed by Jack -/
def principal : ℝ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def totalAmount : ℝ := principal * (1 + interestRate)

/-- Theorem stating that the total amount Jack will pay back is $1320 -/
theorem jack_payback : totalAmount = 1320 := by
  sorry

end NUMINAMATH_CALUDE_jack_payback_l186_18684


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l186_18670

theorem arithmetic_evaluation : 4 + 10 / 2 - 2 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l186_18670


namespace NUMINAMATH_CALUDE_simon_stamps_l186_18699

theorem simon_stamps (initial_stamps : ℕ) (friend1_stamps : ℕ) (friend2_stamps : ℕ) (friend3_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : friend1_stamps = 15)
  (h3 : friend2_stamps = 23)
  (h4 : initial_stamps + friend1_stamps + friend2_stamps + friend3_stamps = 61) :
  friend3_stamps = 23 ∧ friend1_stamps + friend2_stamps + friend3_stamps = 61 := by
  sorry

end NUMINAMATH_CALUDE_simon_stamps_l186_18699


namespace NUMINAMATH_CALUDE_fraction_equality_l186_18629

theorem fraction_equality (p q : ℝ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l186_18629


namespace NUMINAMATH_CALUDE_cone_section_properties_l186_18682

/-- Given a right circular cone with base radius 25 cm and slant height 42 cm,
    when cut by a plane parallel to the base such that the volumes of the two resulting parts are equal,
    the radius of the circular intersection is 25 * (1/2)^(1/3) cm
    and the height of the smaller cone is sqrt(1139) * (1/2)^(1/3) cm. -/
theorem cone_section_properties :
  let base_radius : ℝ := 25
  let slant_height : ℝ := 42
  let cone_height : ℝ := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  let section_radius : ℝ := base_radius * (1/2) ^ (1/3)
  let small_cone_height : ℝ := cone_height * (1/2) ^ (1/3)
  (1/3) * Real.pi * base_radius ^ 2 * cone_height = 2 * ((1/3) * Real.pi * section_radius ^ 2 * small_cone_height) →
  section_radius = 25 * (1/2) ^ (1/3) ∧ small_cone_height = Real.sqrt 1139 * (1/2) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_cone_section_properties_l186_18682


namespace NUMINAMATH_CALUDE_rice_dumpling_costs_l186_18674

theorem rice_dumpling_costs (total_cost_honey : ℝ) (total_cost_date : ℝ) 
  (cost_diff : ℝ) (h1 : total_cost_honey = 1300) (h2 : total_cost_date = 1000) 
  (h3 : cost_diff = 0.6) :
  ∃ (cost_date cost_honey : ℝ),
    cost_date = 2 ∧ 
    cost_honey = 2.6 ∧
    cost_honey = cost_date + cost_diff ∧
    total_cost_honey / cost_honey = total_cost_date / cost_date :=
by
  sorry

end NUMINAMATH_CALUDE_rice_dumpling_costs_l186_18674


namespace NUMINAMATH_CALUDE_oliver_final_balance_l186_18615

def oliver_money_problem (initial_amount : ℝ) (allowance_savings : ℝ) (chore_earnings : ℝ)
  (frisbee_cost : ℝ) (puzzle_cost : ℝ) (sticker_cost : ℝ)
  (movie_ticket_price : ℝ) (movie_discount_percent : ℝ)
  (snack_price : ℝ) (snack_coupon : ℝ)
  (birthday_gift : ℝ) : Prop :=
  let total_expenses := frisbee_cost + puzzle_cost + sticker_cost
  let discounted_movie_price := movie_ticket_price * (1 - movie_discount_percent / 100)
  let snack_cost := snack_price - snack_coupon
  let final_balance := initial_amount + allowance_savings + chore_earnings - 
                       total_expenses - discounted_movie_price - snack_cost + birthday_gift
  final_balance = 9

theorem oliver_final_balance :
  oliver_money_problem 9 5 6 4 3 2 10 20 3 1 8 :=
by sorry

end NUMINAMATH_CALUDE_oliver_final_balance_l186_18615


namespace NUMINAMATH_CALUDE_min_sum_of_product_l186_18641

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l186_18641


namespace NUMINAMATH_CALUDE_floor_T_equals_120_l186_18687

-- Define positive real numbers p, q, r, s
variable (p q r s : ℝ)

-- Define the conditions
axiom p_pos : p > 0
axiom q_pos : q > 0
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom sum_squares_pq : p^2 + q^2 = 2500
axiom sum_squares_rs : r^2 + s^2 = 2500
axiom product_pr : p * r = 1200
axiom product_qs : q * s = 1200

-- Define T
def T : ℝ := p + q + r + s

-- Theorem to prove
theorem floor_T_equals_120 : ⌊T p q r s⌋ = 120 := by sorry

end NUMINAMATH_CALUDE_floor_T_equals_120_l186_18687


namespace NUMINAMATH_CALUDE_equation_solution_l186_18642

theorem equation_solution (x : ℝ) : x ≠ 3 →
  (x - 7 = (4 * |x - 3|) / (x - 3)) ↔ x = 11 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l186_18642


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l186_18681

/-- Given a school with 850 boys, where 28% are Hindus, 10% are Sikhs, 
    and 136 boys belong to other communities, prove that 46% of the boys are Muslims. -/
theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 136 →
  (↑(total_boys - (total_boys * hindu_percent).floor - (total_boys * sikh_percent).floor - other_boys) / total_boys : ℚ) = 46 / 100 :=
by
  sorry

#eval (850 : ℕ) - (850 * (28 / 100 : ℚ)).floor - (850 * (10 / 100 : ℚ)).floor - 136

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l186_18681


namespace NUMINAMATH_CALUDE_percent_calculation_l186_18685

theorem percent_calculation (a b c d e : ℝ) 
  (h1 : c = 0.25 * a)
  (h2 : c = 0.1 * b)
  (h3 : d = 0.5 * b)
  (h4 : d = 0.2 * e)
  (h5 : e = 0.15 * a)
  (h6 : e = 0.05 * c)
  (h7 : a ≠ 0)
  (h8 : c ≠ 0) :
  (d * b + c * e) / (a * c) = 12.65 := by
  sorry

#check percent_calculation

end NUMINAMATH_CALUDE_percent_calculation_l186_18685


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l186_18622

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the distance between vertices
def vertex_distance : ℝ := 8

-- Theorem statement
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), hyperbola_equation x y → vertex_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l186_18622


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_l186_18650

-- Define the functions f and g
def f (x : ℝ) : ℝ := -abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x - 3) + m

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x > g x m) ↔ m < 1 :=
sorry

-- Theorem for the solution set of f(x) + a - 1 > 0
theorem solution_set (a : ℝ) :
  (∀ x : ℝ, f x + a - 1 > 0) ↔
    (a = 1 ∧ (∀ x : ℝ, x ≠ 2 → x ∈ Set.univ)) ∨
    (a > 1 ∧ (∀ x : ℝ, x ∈ Set.univ)) ∨
    (a < 1 ∧ (∀ x : ℝ, x < 1 + a ∨ x > 3 - a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_l186_18650


namespace NUMINAMATH_CALUDE_largest_k_value_l186_18695

/-- Triangle side lengths are positive real numbers that satisfy the triangle inequality --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

/-- The inequality holds for all triangles --/
def inequality_holds (k : ℝ) : Prop :=
  ∀ t : Triangle, (t.a + t.b + t.c)^3 ≥ (5/2) * (t.a^3 + t.b^3 + t.c^3) + k * t.a * t.b * t.c

/-- 39/2 is the largest real number satisfying the inequality --/
theorem largest_k_value : 
  (∀ k : ℝ, k > 39/2 → ¬(inequality_holds k)) ∧ 
  inequality_holds (39/2) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_value_l186_18695


namespace NUMINAMATH_CALUDE_k_range_when_proposition_p_false_l186_18662

theorem k_range_when_proposition_p_false (k : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, k * 4^x - k * 2^(x + 1) + 6 * (k - 5) ≠ 0) →
  k ∈ Set.Iio 5 ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_k_range_when_proposition_p_false_l186_18662


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l186_18624

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Represents the number of games per round -/
def games_per_round : ℕ := 4

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players

/-- Theorem stating the number of ways to schedule the chess tournament -/
theorem chess_tournament_schedules : 
  (num_rounds.factorial * (games_per_round.factorial ^ num_rounds)) = 7962624 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l186_18624


namespace NUMINAMATH_CALUDE_train_passing_time_l186_18605

/-- Proves that the time for train A to pass train B is 7.5 seconds given the conditions -/
theorem train_passing_time (length_A length_B : ℝ) (time_B_passes_A : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 200)
  (h3 : time_B_passes_A = 10) :
  (length_A / (length_B / time_B_passes_A)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l186_18605


namespace NUMINAMATH_CALUDE_min_value_cos_sum_l186_18635

theorem min_value_cos_sum (x : ℝ) : 
  ∃ (m : ℝ), m = -Real.sqrt 2 ∧ ∀ y : ℝ, 
    Real.cos (3*y + π/6) + Real.cos (3*y - π/3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sum_l186_18635


namespace NUMINAMATH_CALUDE_orange_cost_l186_18697

/-- Given that 4 dozen oranges cost $28.80, prove that 5 dozen oranges at the same rate cost $36.00 -/
theorem orange_cost (cost_four_dozen : ℝ) (h1 : cost_four_dozen = 28.80) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 36 :=
by sorry

end NUMINAMATH_CALUDE_orange_cost_l186_18697


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l186_18611

/-- A circle tangent to both coordinate axes with its center on the line 5x - 3y = 8 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  center_on_line : 5 * center.1 - 3 * center.2 = 8

/-- The equation of the circle is either (x-4)² + (y-4)² = 16 or (x-1)² + (y+1)² = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 4)^2 + (y - 4)^2 = 16) ∨
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l186_18611


namespace NUMINAMATH_CALUDE_inequality_proof_l186_18600

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a + b)) + (b / (b + c)) + (c / (c + a)) ≤ 3 / (1 + Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l186_18600


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l186_18654

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the projection of a tetrahedron onto a plane -/
structure TetrahedronProjection where
  area : ℝ
  has_60_degree_angle : Bool

/-- 
Given a regular tetrahedron and its projection onto a plane parallel to the line segment 
connecting the midpoints of two opposite edges, prove that the surface area of the tetrahedron 
is 2x^2 √2/3, where x is the edge length of the tetrahedron.
-/
theorem tetrahedron_surface_area 
  (t : RegularTetrahedron) 
  (p : TetrahedronProjection) 
  (h : p.has_60_degree_angle = true) : 
  ℝ :=
by
  sorry

#check tetrahedron_surface_area

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l186_18654


namespace NUMINAMATH_CALUDE_min_cyclic_fraction_sum_l186_18613

theorem min_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_cyclic_fraction_sum_l186_18613


namespace NUMINAMATH_CALUDE_hunter_can_kill_wolf_l186_18627

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  center : Point

/-- Checks if a point is within an equilateral triangle -/
def isWithinTriangle (p : Point) (t : EquilateralTriangle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Theorem: Hunter can always kill the wolf -/
theorem hunter_can_kill_wolf (t : EquilateralTriangle) 
  (h_side : t.sideLength = 100) :
  ∃ (hunter : Point), 
    ∀ (wolf : Point), 
      isWithinTriangle wolf t → 
        distance hunter wolf ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_hunter_can_kill_wolf_l186_18627


namespace NUMINAMATH_CALUDE_male_athletes_to_sample_l186_18689

def total_athletes : ℕ := 98
def female_athletes : ℕ := 42
def selection_probability : ℚ := 2/7

def male_athletes : ℕ := total_athletes - female_athletes

theorem male_athletes_to_sample :
  ⌊(male_athletes : ℚ) * selection_probability⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_male_athletes_to_sample_l186_18689


namespace NUMINAMATH_CALUDE_trigonometric_sequence_solution_l186_18630

theorem trigonometric_sequence_solution (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, 2 * (Real.cos (a n))^2 = Real.cos (a (n + 1))) →
  (∀ n, Real.cos (a (n + 1)) ≥ 0) →
  (∀ n, |Real.cos (a n)| ≤ 1 / Real.sqrt 2) →
  (∀ n, a (n + 1) = a n + d) →
  (∃ k : ℤ, d = 2 * Real.pi * ↑k ∧ k ≠ 0) →
  (∃ m : ℤ, a 1 = Real.pi / 2 + Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = Real.pi / 3 + 2 * Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = -Real.pi / 3 + 2 * Real.pi * ↑m) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sequence_solution_l186_18630


namespace NUMINAMATH_CALUDE_solution_set_inequality_l186_18653

/-- The solution set of the inequality x(9-x) > 0 is the open interval (0,9) -/
theorem solution_set_inequality (x : ℝ) : x * (9 - x) > 0 ↔ x ∈ Set.Ioo 0 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l186_18653


namespace NUMINAMATH_CALUDE_card_arrangements_sum_14_l186_18659

-- Define the card suits
inductive Suit
| Hearts
| Clubs

-- Define the card values
def CardValue := Fin 4

-- Define a card as a pair of suit and value
def Card := Suit × CardValue

-- Define the deck of 8 cards
def deck : Finset Card := sorry

-- Function to calculate the sum of card values
def sumCardValues (hand : Finset Card) : Nat := sorry

-- Function to count different arrangements
def countArrangements (hand : Finset Card) : Nat := sorry

theorem card_arrangements_sum_14 :
  (Finset.filter (fun hand => hand.card = 4 ∧ sumCardValues hand = 14)
    (Finset.powerset deck)).sum countArrangements = 396 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangements_sum_14_l186_18659


namespace NUMINAMATH_CALUDE_tan_205_in_terms_of_cos_155_l186_18658

theorem tan_205_in_terms_of_cos_155 (a : ℝ) (h : Real.cos (155 * π / 180) = a) :
  Real.tan (205 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end NUMINAMATH_CALUDE_tan_205_in_terms_of_cos_155_l186_18658


namespace NUMINAMATH_CALUDE_solve_cab_driver_income_l186_18673

def cab_driver_income_problem (day1 day2 day3 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day3 + day5
  let day4 := total - known_sum
  (day1 = 45 ∧ day2 = 50 ∧ day3 = 60 ∧ day5 = 70 ∧ average = 58) →
  day4 = 65

theorem solve_cab_driver_income :
  cab_driver_income_problem 45 50 60 70 58 :=
sorry

end NUMINAMATH_CALUDE_solve_cab_driver_income_l186_18673


namespace NUMINAMATH_CALUDE_complex_equation_solution_l186_18602

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 3 + Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l186_18602


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l186_18678

theorem sqrt_five_power_calculation : 
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * 5 ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l186_18678


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l186_18640

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 150 > 0 ∧ 
  x + x + 150 = 360 →
  x = 105 := by sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l186_18640


namespace NUMINAMATH_CALUDE_fishing_tomorrow_l186_18683

/-- Represents the fishing schedule in a coastal village --/
structure FishingVillage where
  daily : ℕ        -- Number of people fishing every day
  everyOther : ℕ   -- Number of people fishing every other day
  everyThree : ℕ   -- Number of people fishing every three days
  yesterday : ℕ    -- Number of people who fished yesterday
  today : ℕ        -- Number of people fishing today

/-- Calculates the number of people fishing tomorrow --/
def tomorrowFishers (v : FishingVillage) : ℕ :=
  v.daily + v.everyThree + (v.everyOther - (v.yesterday - v.daily))

/-- Theorem stating that given the village's fishing pattern, 
    15 people will fish tomorrow --/
theorem fishing_tomorrow (v : FishingVillage) 
  (h1 : v.daily = 7)
  (h2 : v.everyOther = 8)
  (h3 : v.everyThree = 3)
  (h4 : v.yesterday = 12)
  (h5 : v.today = 10) :
  tomorrowFishers v = 15 := by
  sorry

end NUMINAMATH_CALUDE_fishing_tomorrow_l186_18683


namespace NUMINAMATH_CALUDE_sum_equals_result_l186_18660

-- Define the sum
def sum : ℚ := 10/9 + 9/10

-- Define the result as a rational number (2 + 1/10)
def result : ℚ := 2 + 1/10

-- Theorem stating that the sum equals the result
theorem sum_equals_result : sum = result := by sorry

end NUMINAMATH_CALUDE_sum_equals_result_l186_18660


namespace NUMINAMATH_CALUDE_four_Z_three_equals_one_l186_18691

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

-- Theorem to prove
theorem four_Z_three_equals_one : Z 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_one_l186_18691


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l186_18603

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (w + rectangle_length) = 30 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l186_18603


namespace NUMINAMATH_CALUDE_goldbach_negation_equiv_l186_18647

-- Define Goldbach's Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define the negation of Goldbach's Conjecture
def not_goldbach : Prop :=
  ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Theorem stating the equivalence
theorem goldbach_negation_equiv :
  ¬goldbach_conjecture ↔ not_goldbach := by sorry

end NUMINAMATH_CALUDE_goldbach_negation_equiv_l186_18647


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l186_18610

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 = 3 →
  a 7 + a 8 = 27 →
  a 9 + a 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l186_18610


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l186_18656

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + (2*m - 1)*x + m^2 = 0

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x

def range_of_m : Set ℝ :=
  {m : ℝ | m ≤ 1/4}

def roots_relation (m : ℝ) (α β : ℝ) : Prop :=
  quadratic_equation m α ∧ quadratic_equation m β ∧ α ≠ β

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m → m ∈ range_of_m) ∧
  (∃ m : ℝ, m = -1 ∧ 
    ∃ α β : ℝ, roots_relation m α β ∧ α^2 + β^2 - α*β = 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l186_18656


namespace NUMINAMATH_CALUDE_coffee_stock_calculation_l186_18661

/-- Represents the initial amount of coffee in stock -/
def initial_stock : ℝ := sorry

/-- The fraction of initial stock that is decaffeinated -/
def initial_decaf_fraction : ℝ := 0.4

/-- The amount of new coffee purchased -/
def new_purchase : ℝ := 100

/-- The fraction of new purchase that is decaffeinated -/
def new_decaf_fraction : ℝ := 0.6

/-- The fraction of total stock that is decaffeinated after the purchase -/
def final_decaf_fraction : ℝ := 0.44

theorem coffee_stock_calculation :
  initial_stock = 400 ∧
  final_decaf_fraction * (initial_stock + new_purchase) =
    initial_decaf_fraction * initial_stock + new_decaf_fraction * new_purchase :=
sorry

end NUMINAMATH_CALUDE_coffee_stock_calculation_l186_18661


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l186_18665

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -1/2 and 2 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x ∈ Set.Ioo (-1/2 : ℝ) 2, QuadraticFunction a b c x > 0) →
  (QuadraticFunction a b c (-1/2) = 0) →
  (QuadraticFunction a b c 2 = 0) →
  (b > 0 ∧ c > 0 ∧ a + b + c > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l186_18665


namespace NUMINAMATH_CALUDE_equation_proof_l186_18686

theorem equation_proof (x : ℚ) : x = 5 → 65 + (x * 12) / 60 = 66 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l186_18686


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l186_18626

theorem absolute_difference_of_roots (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → |p - q| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l186_18626


namespace NUMINAMATH_CALUDE_area_equality_l186_18608

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define midpoints
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

-- Define intersection of lines
def is_intersection (P X₁ Y₁ X₂ Y₂ : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define area of a quadrilateral
def area_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_equality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_midpoint : is_midpoint E A B)
  (h_F_midpoint : is_midpoint F C D)
  (h_G_intersection : is_intersection G A F D E)
  (h_H_intersection : is_intersection H B F C E) :
  area_triangle A G D + area_triangle B H C = area_quadrilateral E H F G := 
sorry

end NUMINAMATH_CALUDE_area_equality_l186_18608
