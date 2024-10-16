import Mathlib

namespace NUMINAMATH_CALUDE_champion_is_C_l425_42563

-- Define the contestants
inductive Contestant : Type
  | A | B | C | D | E

-- Define the predictions
def father_prediction (c : Contestant) : Prop :=
  c = Contestant.A ∨ c = Contestant.C

def mother_prediction (c : Contestant) : Prop :=
  c ≠ Contestant.B ∧ c ≠ Contestant.C

def child_prediction (c : Contestant) : Prop :=
  c = Contestant.D ∨ c = Contestant.E

-- Define the condition that only one prediction is correct
def only_one_correct (c : Contestant) : Prop :=
  (father_prediction c ∧ ¬mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ ¬mother_prediction c ∧ child_prediction c)

-- Theorem statement
theorem champion_is_C :
  ∃ (c : Contestant), only_one_correct c → c = Contestant.C :=
sorry

end NUMINAMATH_CALUDE_champion_is_C_l425_42563


namespace NUMINAMATH_CALUDE_train_speed_l425_42509

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 200) (h2 : time = 20) :
  length / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l425_42509


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l425_42538

/-- Given two parallel vectors p and q, prove that the magnitude of their sum is √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (x : ℝ) : 
  p = (2, -3) → 
  q = (x, 6) → 
  (2 * 6 = -3 * x) →  -- parallelism condition
  ‖p + q‖ = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l425_42538


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l425_42527

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a ≠ 1 ∧ A ∪ B a = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l425_42527


namespace NUMINAMATH_CALUDE_shooting_game_probability_l425_42513

theorem shooting_game_probability (A B : Type) 
  (hit_score : ℕ) (miss_score : ℕ) 
  (A_hit_rate : ℚ) (B_hit_rate : ℚ) 
  (sum_two_prob : ℚ) :
  hit_score = 2 →
  miss_score = 0 →
  A_hit_rate = 3/5 →
  sum_two_prob = 9/20 →
  (A_hit_rate * (1 - B_hit_rate) + (1 - A_hit_rate) * B_hit_rate = sum_two_prob) →
  B_hit_rate = 3/4 := by
sorry

end NUMINAMATH_CALUDE_shooting_game_probability_l425_42513


namespace NUMINAMATH_CALUDE_battery_difference_is_13_l425_42506

/-- The number of batteries Tom used in flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in toys -/
def toy_batteries : ℕ := 15

/-- The difference between the number of batteries in toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_is_13 : battery_difference = 13 := by
  sorry

end NUMINAMATH_CALUDE_battery_difference_is_13_l425_42506


namespace NUMINAMATH_CALUDE_ratio_problem_l425_42570

theorem ratio_problem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ratio : (x + y) / (x - y) = 4 / 3) : x / y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l425_42570


namespace NUMINAMATH_CALUDE_green_to_yellow_area_ratio_l425_42588

-- Define the diameters of the circles
def small_diameter : ℝ := 2
def large_diameter : ℝ := 6

-- Define the theorem
theorem green_to_yellow_area_ratio :
  let small_radius := small_diameter / 2
  let large_radius := large_diameter / 2
  let yellow_area := π * small_radius^2
  let total_area := π * large_radius^2
  let green_area := total_area - yellow_area
  green_area / yellow_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_green_to_yellow_area_ratio_l425_42588


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l425_42553

/-- Given a circle with radius 7, perpendicular diameters EF and GH, and a chord EJ of length 12
    intersecting GH at M, prove that GM = 7 + √13 and MH = 7 - √13 -/
theorem chord_intersection_lengths (O : ℝ × ℝ) (E F G H J M : ℝ × ℝ) :
  let r : ℝ := 7
  let circle := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}
  (E ∈ circle) ∧ (F ∈ circle) ∧ (G ∈ circle) ∧ (H ∈ circle) ∧ (J ∈ circle) →
  (E.1 - F.1) * (G.2 - H.2) = 0 ∧ (E.2 - F.2) * (G.1 - H.1) = 0 →
  (E.1 - J.1)^2 + (E.2 - J.2)^2 = 12^2 →
  M.1 = (G.1 + H.1) / 2 ∧ M.2 = (G.2 + H.2) / 2 →
  (M.1 - G.1)^2 + (M.2 - G.2)^2 = (7 + Real.sqrt 13)^2 ∧
  (M.1 - H.1)^2 + (M.2 - H.2)^2 = (7 - Real.sqrt 13)^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l425_42553


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l425_42572

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l425_42572


namespace NUMINAMATH_CALUDE_smallest_x_value_l425_42571

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = x * (x + 15)) → x ≥ -15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l425_42571


namespace NUMINAMATH_CALUDE_height_difference_l425_42520

/-- Given the heights of three siblings, prove the height difference between two of them. -/
theorem height_difference (cary_height bill_height jan_height : ℕ) :
  cary_height = 72 →
  bill_height = cary_height / 2 →
  jan_height = 42 →
  jan_height - bill_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l425_42520


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l425_42522

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  Nat.minFac (3^15 + 11^21) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l425_42522


namespace NUMINAMATH_CALUDE_min_value_shifted_function_l425_42591

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) ∧
  (∀ (x : ℝ), f c x ≥ 2) ∧
  (∃ (x₁ : ℝ), f c x₁ = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f c (x - 3) ≥ m ∧ ∃ (x₀ : ℝ), f c (x₀ - 3) = m) ∧
  (∀ (x : ℝ), f c (x - 3) ≥ 2) ∧
  (∃ (x₁ : ℝ), f c (x₁ - 3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_function_l425_42591


namespace NUMINAMATH_CALUDE_geometric_sum_eight_thirds_l425_42525

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_eight_thirds : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_thirds_l425_42525


namespace NUMINAMATH_CALUDE_function_non_negative_implies_k_range_l425_42566

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + k + 3

/-- The theorem statement -/
theorem function_non_negative_implies_k_range (k : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f k x ≥ 0) → k ≥ -3/13 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_k_range_l425_42566


namespace NUMINAMATH_CALUDE_root_of_fifth_unity_l425_42585

theorem root_of_fifth_unity {p q r s t m : ℂ} (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_of_fifth_unity_l425_42585


namespace NUMINAMATH_CALUDE_michael_watermelon_weight_l425_42511

/-- The weight of Michael's watermelon in pounds -/
def michael_watermelon : ℝ := 8

/-- The weight of Clay's watermelon in pounds -/
def clay_watermelon : ℝ := 3 * michael_watermelon

/-- The weight of John's watermelon in pounds -/
def john_watermelon : ℝ := 12

theorem michael_watermelon_weight :
  michael_watermelon = 8 ∧
  clay_watermelon = 3 * michael_watermelon ∧
  john_watermelon = clay_watermelon / 2 ∧
  john_watermelon = 12 := by
  sorry

end NUMINAMATH_CALUDE_michael_watermelon_weight_l425_42511


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l425_42579

theorem opposite_of_negative_three : 
  (∃ x : ℤ, -3 + x = 0) → (∃ x : ℤ, -3 + x = 0 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l425_42579


namespace NUMINAMATH_CALUDE_incorrect_reasoning_l425_42500

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- Define the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ l α A, ¬(line_on_plane l α) → on_line A l → ¬(on_plane A α)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_reasoning_l425_42500


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l425_42548

theorem smallest_undefined_value : 
  let f (x : ℝ) := (x - 3) / (9*x^2 - 90*x + 225)
  ∃ (y : ℝ), (∀ (x : ℝ), x < y → f x ≠ 0⁻¹) ∧ f y = 0⁻¹ ∧ y = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l425_42548


namespace NUMINAMATH_CALUDE_determinant_inequality_l425_42590

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det2x2 7 (x^2) 2 1 > det2x2 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_determinant_inequality_l425_42590


namespace NUMINAMATH_CALUDE_solution_set_inequality_l425_42568

theorem solution_set_inequality (x : ℝ) :
  x^2 * (x - 4) ≥ 0 ↔ x ≥ 4 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l425_42568


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l425_42577

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x > 1 ∧ y > 2) → x + y > 3) ∧
   ∃ x y, x + y > 3 ∧ ¬(x > 1 ∧ y > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l425_42577


namespace NUMINAMATH_CALUDE_delta_quotient_on_curve_l425_42552

/-- Given a point (1,3) on the curve y = x^2 + 2, and a nearby point (1 + Δx, 3 + Δy) on the same curve,
    prove that Δy / Δx = 2 + Δx. -/
theorem delta_quotient_on_curve (Δx Δy : ℝ) : 
  (3 + Δy = (1 + Δx)^2 + 2) → (Δy / Δx = 2 + Δx) := by
  sorry

end NUMINAMATH_CALUDE_delta_quotient_on_curve_l425_42552


namespace NUMINAMATH_CALUDE_total_turtles_is_30_l425_42501

/-- The number of turtles Kristen has -/
def kristens_turtles : ℕ := 12

/-- The number of turtles Kris has -/
def kris_turtles : ℕ := kristens_turtles / 4

/-- The number of turtles Trey has -/
def treys_turtles : ℕ := 5 * kris_turtles

/-- The total number of turtles -/
def total_turtles : ℕ := kristens_turtles + kris_turtles + treys_turtles

theorem total_turtles_is_30 : total_turtles = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_is_30_l425_42501


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l425_42510

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l425_42510


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l425_42518

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90) 
  (h2 : both = 22) 
  (h3 : with_car = 44) 
  (h4 : bike_only = 35) : 
  total - (with_car + bike_only + both - both) = 11 := by
  sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l425_42518


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l425_42581

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Given equation
  (Real.sqrt 3 * Real.sin B + b * Real.cos A = c) →
  -- Prove angle B
  (B = π / 6) ∧
  -- Prove area when a = √3 * c and b = 2
  (a = Real.sqrt 3 * c ∧ b = 2 → 
   (1 / 2) * a * b * Real.sin C = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l425_42581


namespace NUMINAMATH_CALUDE_ticket_difference_l425_42576

theorem ticket_difference (fair_tickets : ℕ) (baseball_tickets : ℕ)
  (h1 : fair_tickets = 25)
  (h2 : baseball_tickets = 56) :
  2 * baseball_tickets - fair_tickets = 87 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l425_42576


namespace NUMINAMATH_CALUDE_hardcover_probability_l425_42537

theorem hardcover_probability (total_books : Nat) (hardcover_books : Nat) (selected_books : Nat) :
  total_books = 15 →
  hardcover_books = 5 →
  selected_books = 3 →
  (Nat.choose hardcover_books selected_books * Nat.choose (total_books - hardcover_books) (selected_books - hardcover_books) +
   Nat.choose hardcover_books (selected_books - 1) * Nat.choose (total_books - hardcover_books) 1 +
   Nat.choose hardcover_books selected_books) / Nat.choose total_books selected_books = 67 / 91 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_probability_l425_42537


namespace NUMINAMATH_CALUDE_final_temperature_is_correct_l425_42508

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let temp1 := initial * 2
  let temp2 := temp1 - 30
  let temp3 := temp2 * 0.7
  let temp4 := temp3 + 24
  let temp5 := temp4 * 0.9
  let temp6 := temp5 + 8
  let temp7 := temp6 * 1.2
  temp7 - 15

/-- Theorem stating that the final temperature is 58.32 degrees --/
theorem final_temperature_is_correct : 
  abs (finalTemperature 40 - 58.32) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_temperature_is_correct_l425_42508


namespace NUMINAMATH_CALUDE_three_line_hexagon_angle_sum_l425_42589

/-- A hexagon formed by the intersection of three lines -/
structure ThreeLineHexagon where
  -- Define the six angles of the hexagon
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  angle6 : ℝ

/-- The sum of angles in a hexagon formed by three intersecting lines is 360° -/
theorem three_line_hexagon_angle_sum (h : ThreeLineHexagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_three_line_hexagon_angle_sum_l425_42589


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l425_42523

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 → b = 32 → c^2 = a^2 + b^2 → c = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l425_42523


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l425_42536

def Q (n : ℕ) : ℚ := 4 / ((n + 2) * (n + 3))

theorem smallest_n_for_Q_less_than_threshold : 
  (∃ n : ℕ, Q n < 1/4022) ∧ 
  (∀ m : ℕ, m < 62 → Q m ≥ 1/4022) ∧ 
  Q 62 < 1/4022 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l425_42536


namespace NUMINAMATH_CALUDE_order_of_a_ab2_ab_l425_42503

theorem order_of_a_ab2_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_order_of_a_ab2_ab_l425_42503


namespace NUMINAMATH_CALUDE_cricket_average_increase_l425_42551

/-- Represents the problem of calculating the increase in average runs -/
def calculateAverageIncrease (initialMatches : ℕ) (initialAverage : ℚ) (nextMatchRuns : ℕ) : ℚ :=
  let totalInitialRuns := initialMatches * initialAverage
  let totalMatches := initialMatches + 1
  let totalRuns := totalInitialRuns + nextMatchRuns
  (totalRuns / totalMatches) - initialAverage

/-- The theorem stating the solution to the cricket player's average problem -/
theorem cricket_average_increase :
  calculateAverageIncrease 10 32 76 = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_increase_l425_42551


namespace NUMINAMATH_CALUDE_r_amount_unchanged_l425_42562

/-- Represents the financial situation of three friends P, Q, and R. -/
structure FriendFinances where
  p : ℝ  -- Amount with P
  q : ℝ  -- Amount with Q
  r : ℝ  -- Amount with R

/-- The total amount among the three friends is 4000. -/
def total_amount (f : FriendFinances) : Prop :=
  f.p + f.q + f.r = 4000

/-- R has two-thirds of the total amount with P and Q. -/
def r_two_thirds_pq (f : FriendFinances) : Prop :=
  f.r = (2/3) * (f.p + f.q)

/-- The ratio of amount with P to amount with Q is 3:2. -/
def p_q_ratio (f : FriendFinances) : Prop :=
  f.p / f.q = 3/2

/-- 10% of P's amount will be donated to charity. -/
def charity_donation (f : FriendFinances) : ℝ :=
  0.1 * f.p

/-- Theorem stating that R's amount remains unchanged after P's charity donation. -/
theorem r_amount_unchanged (f : FriendFinances) 
  (h1 : total_amount f) 
  (h2 : r_two_thirds_pq f) 
  (h3 : p_q_ratio f) : 
  f.r = 1600 :=
sorry

end NUMINAMATH_CALUDE_r_amount_unchanged_l425_42562


namespace NUMINAMATH_CALUDE_park_tree_count_l425_42567

/-- Calculates the final number of trees in a park after cutting --/
def final_tree_count (initial_oak initial_maple oak_cut maple_cut : ℕ) : ℕ × ℕ × ℕ :=
  let final_oak := initial_oak - oak_cut
  let final_maple := initial_maple - maple_cut
  let total := final_oak + final_maple
  (final_oak, final_maple, total)

/-- Theorem stating the final tree count after cutting in the park --/
theorem park_tree_count :
  final_tree_count 57 43 13 8 = (44, 35, 79) := by
  sorry

end NUMINAMATH_CALUDE_park_tree_count_l425_42567


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l425_42502

/-- 
Given a quadratic equation ax^2 + bx + c = 0,
prove that it has two distinct real roots when a = 1, b = 2, and c = -5.
-/
theorem quadratic_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -5
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ (x y : ℝ), x ≠ y ∧ 
    a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l425_42502


namespace NUMINAMATH_CALUDE_johnson_family_reunion_l425_42534

theorem johnson_family_reunion (num_children : ℕ) (num_adults : ℕ) (num_blue_adults : ℕ) : 
  num_children = 45 →
  num_adults = num_children / 3 →
  num_blue_adults = num_adults / 3 →
  num_adults - num_blue_adults = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_l425_42534


namespace NUMINAMATH_CALUDE_min_value_ab_l425_42595

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 3 / b = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2 / x + 3 / y = Real.sqrt (x * y) → a * b ≤ x * y :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l425_42595


namespace NUMINAMATH_CALUDE_linear_regression_at_25_l425_42599

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

/-- Theorem: Given the linear regression equation y = 0.50x - 0.81, when x = 25, y = 11.69 -/
theorem linear_regression_at_25 : linear_regression 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_at_25_l425_42599


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l425_42598

theorem five_sixths_of_twelve_fifths (a b c d : ℚ) : 
  a = 5 ∧ b = 6 ∧ c = 12 ∧ d = 5 → (a / b) * (c / d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l425_42598


namespace NUMINAMATH_CALUDE_boys_joining_group_l425_42583

theorem boys_joining_group (total : ℕ) (initial_boys : ℕ) (initial_girls : ℕ) (boys_joining : ℕ) :
  total = 48 →
  initial_boys + initial_girls = total →
  initial_boys * 5 = initial_girls * 3 →
  (initial_boys + boys_joining) * 3 = initial_girls * 5 →
  boys_joining = 32 := by
sorry

end NUMINAMATH_CALUDE_boys_joining_group_l425_42583


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l425_42535

theorem base_10_to_base_7 (n : ℕ) (h : n = 947) :
  ∃ (a b c d : ℕ),
    n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 5 ∧ c = 2 ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l425_42535


namespace NUMINAMATH_CALUDE_butterfly_equation_equal_roots_l425_42575

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) that satisfies the "butterfly" condition (a - b + c = 0) and has two equal real roots implies a = c. -/
theorem butterfly_equation_equal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (a - b + c = 0) →  -- Butterfly condition
  (b^2 - 4*a*c = 0) →  -- Condition for two equal real roots (discriminant = 0)
  a = c := by
  sorry

end NUMINAMATH_CALUDE_butterfly_equation_equal_roots_l425_42575


namespace NUMINAMATH_CALUDE_expression_equals_five_l425_42543

theorem expression_equals_five :
  (π + Real.sqrt 3) ^ 0 + (-2) ^ 2 + |(-1/2)| - Real.sin (30 * π / 180) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_l425_42543


namespace NUMINAMATH_CALUDE_x_power_y_value_l425_42544

theorem x_power_y_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end NUMINAMATH_CALUDE_x_power_y_value_l425_42544


namespace NUMINAMATH_CALUDE_same_lunch_group_probability_l425_42554

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 6

/-- The number of students in each lunch group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

/-- The number of specific students we're interested in -/
def num_specific_students : ℕ := 4

theorem same_lunch_group_probability :
  (prob_single_student ^ (num_specific_students - 1) : ℚ) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_same_lunch_group_probability_l425_42554


namespace NUMINAMATH_CALUDE_root_product_equation_l425_42541

theorem root_product_equation (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) →
  (β^2 + p*β + 2 = 0) →
  (γ^2 + q*γ + 2 = 0) →
  (δ^2 + q*δ + 2 = 0) →
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 4 + 2*(p^2 - q^2) :=
by sorry

end NUMINAMATH_CALUDE_root_product_equation_l425_42541


namespace NUMINAMATH_CALUDE_sarahs_friends_ages_sum_l425_42574

theorem sarahs_friends_ages_sum :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit integers
    a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
    a * b = 36 →                -- product of two ages is 36
    c ∣ 36 →                    -- third age is a factor of 36
    c ≠ a ∧ c ≠ b →             -- third age is not one of the first two
    a + b + c = 16 :=           -- sum of all three ages is 16
by sorry

end NUMINAMATH_CALUDE_sarahs_friends_ages_sum_l425_42574


namespace NUMINAMATH_CALUDE_combined_population_l425_42582

/-- The combined population of New England and New York given their relative populations -/
theorem combined_population (new_england_pop : ℕ) (new_york_pop : ℕ) :
  new_england_pop = 2100000 →
  new_york_pop = (2 : ℕ) * new_england_pop / 3 →
  new_england_pop + new_york_pop = 3500000 :=
by sorry

end NUMINAMATH_CALUDE_combined_population_l425_42582


namespace NUMINAMATH_CALUDE_granddaughter_age_l425_42515

/-- Represents a family with three generations -/
structure Family where
  betty_age : ℕ
  daughter_age : ℕ
  granddaughter_age : ℕ

/-- The age relationship in the family -/
def valid_family_ages (f : Family) : Prop :=
  f.betty_age = 60 ∧
  f.daughter_age = f.betty_age - (f.betty_age * 40 / 100) ∧
  f.granddaughter_age = f.daughter_age / 3

/-- Theorem stating the granddaughter's age in the family -/
theorem granddaughter_age (f : Family) (h : valid_family_ages f) : 
  f.granddaughter_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_granddaughter_age_l425_42515


namespace NUMINAMATH_CALUDE_z_squared_minus_four_z_is_real_l425_42592

/-- Given a real number a and a complex number z = 2 + ai, 
    prove that z^2 - 4z is a real number. -/
theorem z_squared_minus_four_z_is_real (a : ℝ) : 
  let z : ℂ := 2 + a * Complex.I
  (z^2 - 4*z).im = 0 := by sorry

end NUMINAMATH_CALUDE_z_squared_minus_four_z_is_real_l425_42592


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l425_42573

theorem max_value_abc_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l425_42573


namespace NUMINAMATH_CALUDE_girls_entered_classroom_l425_42565

theorem girls_entered_classroom (initial_boys initial_girls boys_left final_total : ℕ) :
  initial_boys = 5 →
  initial_girls = 4 →
  boys_left = 3 →
  final_total = 8 →
  ∃ girls_entered : ℕ, girls_entered = 2 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end NUMINAMATH_CALUDE_girls_entered_classroom_l425_42565


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l425_42528

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l425_42528


namespace NUMINAMATH_CALUDE_motorcyclists_travel_time_l425_42526

/-- 
Two motorcyclists start simultaneously from opposite points A and B.
They meet at some point between A and B.
The first motorcyclist (from A to B) arrives at B 2.5 hours after meeting.
The second motorcyclist (from B to A) arrives at A 1.6 hours after meeting.
This theorem proves that their total travel times are 4.5 hours and 3.6 hours respectively.
-/
theorem motorcyclists_travel_time (s : ℝ) (h : s > 0) : 
  ∃ (t : ℝ), t > 0 ∧ 
    (s / (t + 2.5) * 2.5 = s / (t + 1.6) * t) ∧ 
    (t + 2.5 = 4.5) ∧ 
    (t + 1.6 = 3.6) := by
  sorry

#check motorcyclists_travel_time

end NUMINAMATH_CALUDE_motorcyclists_travel_time_l425_42526


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l425_42555

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.44 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l425_42555


namespace NUMINAMATH_CALUDE_max_composite_sum_l425_42545

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- The sum of a list of natural numbers. -/
def ListSum (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

/-- A list of natural numbers is a valid decomposition if all its elements are composite
    and their sum is 2013. -/
def IsValidDecomposition (L : List ℕ) : Prop :=
  (∀ n ∈ L, IsComposite n) ∧ ListSum L = 2013

theorem max_composite_sum :
  (∃ L : List ℕ, IsValidDecomposition L ∧ L.length = 502) ∧
  (∀ L : List ℕ, IsValidDecomposition L → L.length ≤ 502) := by
  sorry

end NUMINAMATH_CALUDE_max_composite_sum_l425_42545


namespace NUMINAMATH_CALUDE_possible_m_values_l425_42596

theorem possible_m_values (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1} →
  B = {x | m * x = 1} →
  A ∪ B = A →
  m = 0 ∨ m = 1 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_possible_m_values_l425_42596


namespace NUMINAMATH_CALUDE_special_function_inequality_l425_42593

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  has_derivative : Differentiable ℝ f
  symmetric : ∀ x, f x = 6 * x^2 - f (-x)
  derivative_bound : ∀ x, x < 0 → 2 * deriv f x + 1 < 12 * x

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  ∀ m : ℝ, sf.f (m + 2) ≤ sf.f (-2 * m) + 12 * m + 12 - 9 * m^2 ↔ m ≥ - 2/3 :=
by sorry

end NUMINAMATH_CALUDE_special_function_inequality_l425_42593


namespace NUMINAMATH_CALUDE_weather_forecast_probability_l425_42561

/-- The probability of success for a single trial -/
def p : ℝ := 0.8

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem weather_forecast_probability :
  binomial_probability n k p = 0.384 := by
  sorry

end NUMINAMATH_CALUDE_weather_forecast_probability_l425_42561


namespace NUMINAMATH_CALUDE_games_for_512_players_l425_42556

/-- A single-elimination tournament with a given number of initial players. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  initial_players_pos : initial_players > 0

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.initial_players - 1

/-- Theorem stating that a single-elimination tournament with 512 initial players
    requires 511 games to determine the champion. -/
theorem games_for_512_players :
  ∀ (t : SingleEliminationTournament), t.initial_players = 512 → games_played t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_512_players_l425_42556


namespace NUMINAMATH_CALUDE_solutions_are_correct_l425_42584

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 = 49
def equation2 (x : ℝ) : Prop := (2*x + 3)^2 = 4*(2*x + 3)
def equation3 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0
def equation4 (x : ℝ) : Prop := (x + 8)*(x + 1) = -12

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 7 ∧ equation1 (-7)) ∧
  (equation2 (-3/2) ∧ equation2 (1/2)) ∧
  (equation3 ((-2 + Real.sqrt 10) / 2) ∧ equation3 ((-2 - Real.sqrt 10) / 2)) ∧
  (equation4 (-4) ∧ equation4 (-5)) := by sorry

end NUMINAMATH_CALUDE_solutions_are_correct_l425_42584


namespace NUMINAMATH_CALUDE_sum_of_ages_l425_42550

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 20 →
  jill_age = 13 →
  henry_age - 6 = 2 * (jill_age - 6) →
  henry_age + jill_age = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l425_42550


namespace NUMINAMATH_CALUDE_alissa_earrings_l425_42540

/-- Represents the number of pairs of earrings Barbie bought -/
def barbie_pairs : ℕ := 12

/-- Represents the number of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := barbie_pairs * 2 / 2

/-- Represents Alissa's total number of earrings after receiving the gift -/
def alissa_total : ℕ := 3 * earrings_given

theorem alissa_earrings : alissa_total = 36 := by
  sorry

end NUMINAMATH_CALUDE_alissa_earrings_l425_42540


namespace NUMINAMATH_CALUDE_factorization_equality_l425_42530

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l425_42530


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l425_42569

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- The initial "T" shaped configuration -/
def initial_config : TileConfiguration :=
  { tiles := 6, perimeter := 12 }

/-- The number of tiles added -/
def added_tiles : ℕ := 3

/-- A function that calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added : ℕ) : ℕ :=
  sorry

theorem perimeter_after_adding_tiles :
  new_perimeter initial_config added_tiles = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l425_42569


namespace NUMINAMATH_CALUDE_grocery_store_problem_l425_42547

/-- Represents the price of an item after applying discount and tax -/
structure ItemPrice where
  base : ℝ
  discount : ℝ
  tax : ℝ

/-- Calculates the final price of an item after applying discount and tax -/
def finalPrice (item : ItemPrice) : ℝ :=
  item.base * (1 - item.discount) * (1 + item.tax)

/-- Represents the grocery store problem -/
theorem grocery_store_problem :
  let spam : ItemPrice := { base := 3, discount := 0.1, tax := 0 }
  let peanutButter : ItemPrice := { base := 5, discount := 0, tax := 0.05 }
  let bread : ItemPrice := { base := 2, discount := 0, tax := 0 }
  let milk : ItemPrice := { base := 4, discount := 0.2, tax := 0.08 }
  let eggs : ItemPrice := { base := 3, discount := 0.05, tax := 0 }
  
  let totalAmount :=
    12 * finalPrice spam +
    3 * finalPrice peanutButter +
    4 * finalPrice bread +
    2 * finalPrice milk +
    1 * finalPrice eggs
  
  totalAmount = 65.92 := by sorry

end NUMINAMATH_CALUDE_grocery_store_problem_l425_42547


namespace NUMINAMATH_CALUDE_point_distance_product_l425_42560

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-5 - 4)^2 + (y₁ - 5)^2 = 12^2) →
  ((-5 - 4)^2 + (y₂ - 5)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -38 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l425_42560


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l425_42529

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (x^2 - 4) / (2*x - 4) = 0 ∧ 2*x - 4 ≠ 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l425_42529


namespace NUMINAMATH_CALUDE_passing_percentage_l425_42512

theorem passing_percentage (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ) 
  (h1 : student_marks = 175)
  (h2 : failed_by = 56)
  (h3 : max_marks = 700) :
  (((student_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
sorry

end NUMINAMATH_CALUDE_passing_percentage_l425_42512


namespace NUMINAMATH_CALUDE_expand_expression_l425_42558

theorem expand_expression (x : ℝ) : 4 * (5 * x^3 - 3 * x^2 + 7 * x - 2) = 20 * x^3 - 12 * x^2 + 28 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l425_42558


namespace NUMINAMATH_CALUDE_arielle_age_l425_42594

theorem arielle_age (elvie_age : ℕ) (total : ℕ) : 
  elvie_age = 10 →
  (∃ (arielle_age : ℕ), 
    elvie_age + arielle_age + elvie_age * arielle_age = total ∧
    total = 131) →
  ∃ (arielle_age : ℕ), arielle_age = 11 :=
by sorry

end NUMINAMATH_CALUDE_arielle_age_l425_42594


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l425_42586

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l425_42586


namespace NUMINAMATH_CALUDE_hotel_room_occupancy_l425_42557

theorem hotel_room_occupancy (num_rooms : ℕ) (towels_per_person : ℕ) (total_towels : ℕ) 
  (h1 : num_rooms = 10)
  (h2 : towels_per_person = 2)
  (h3 : total_towels = 60) :
  total_towels / towels_per_person / num_rooms = 3 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_occupancy_l425_42557


namespace NUMINAMATH_CALUDE_equation_solution_l425_42546

theorem equation_solution (a b : ℝ) (h : (a^2 * b^2) / (a^4 - 2*b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l425_42546


namespace NUMINAMATH_CALUDE_tvs_on_auction_site_l425_42517

def tvs_in_person : ℕ := 8
def tvs_online_multiplier : ℕ := 3
def total_tvs : ℕ := 42

theorem tvs_on_auction_site :
  let tvs_online := tvs_online_multiplier * tvs_in_person
  let tvs_before_auction := tvs_in_person + tvs_online
  total_tvs - tvs_before_auction = 10 := by
sorry

end NUMINAMATH_CALUDE_tvs_on_auction_site_l425_42517


namespace NUMINAMATH_CALUDE_equation_solution_l425_42532

theorem equation_solution : 
  ∃ x : ℝ, (8 * 5.4 - 0.6 * x / 1.2 = 31.000000000000004) ∧ (x = 24.4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l425_42532


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l425_42580

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the line
def line (x : ℝ) : Prop := x = -1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (4, 0)

-- Define points A and B
variable (A B : ℝ × ℝ)

-- State the theorem
theorem parabola_intersection_length :
  parabola B.1 B.2 →
  line A.1 →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1 - t) • focus + t • A) →
  (A - focus) = 5 • (B - focus) →
  ‖A - B‖ = 28 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l425_42580


namespace NUMINAMATH_CALUDE_transylvanian_must_be_rational_l425_42533

/-- Represents the state of a person's mind -/
inductive MindState
| Rational
| Lost

/-- Represents a person -/
structure Person where
  mindState : MindState

/-- Represents the claim made by a person -/
def claim (p : Person) : Prop :=
  p.mindState = MindState.Lost

/-- A person with a lost mind cannot make a truthful claim about their condition -/
axiom lost_mind_cannot_claim : ∀ (p : Person), p.mindState = MindState.Lost → ¬(claim p)

/-- The theorem to be proved -/
theorem transylvanian_must_be_rational (p : Person) (makes_claim : claim p) :
  p.mindState = MindState.Rational := by
  sorry

end NUMINAMATH_CALUDE_transylvanian_must_be_rational_l425_42533


namespace NUMINAMATH_CALUDE_order_relation_l425_42531

theorem order_relation (a b c d : ℝ) 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l425_42531


namespace NUMINAMATH_CALUDE_crispy_red_plum_pricing_l425_42597

theorem crispy_red_plum_pricing (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x > 5) 
  (first_batch_cost : ℝ := 12000)
  (second_batch_cost : ℝ := 11000)
  (price_difference : ℝ := 5)
  (quantity_difference : ℝ := 40) :
  first_batch_cost / x = second_batch_cost / (x - price_difference) - quantity_difference := by
sorry

end NUMINAMATH_CALUDE_crispy_red_plum_pricing_l425_42597


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l425_42539

def set_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def set_B : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_of_A_and_B : set_A ∩ (set_B.image (coe : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l425_42539


namespace NUMINAMATH_CALUDE_dan_marbles_remaining_l425_42505

/-- The number of violet marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Dan has 50 marbles left -/
theorem dan_marbles_remaining : remaining_marbles 64 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_remaining_l425_42505


namespace NUMINAMATH_CALUDE_max_value_theorem_l425_42542

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (M : ℝ), M = 15 ∧ ∀ (a b : ℝ), 2 * a^2 - 6 * a + b^2 = 0 → a^2 + b^2 + 2 * a ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l425_42542


namespace NUMINAMATH_CALUDE_two_diamonds_balance_l425_42524

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℝ
  diamond : ℝ
  circle : ℝ

/-- The given balance conditions -/
def balance_conditions (b : Balance) : Prop :=
  3 * b.triangle + b.diamond = 9 * b.circle ∧
  b.triangle = b.diamond + 2 * b.circle

/-- The theorem to prove -/
theorem two_diamonds_balance (b : Balance) :
  balance_conditions b → 2 * b.diamond = 1.5 * b.circle := by
  sorry

end NUMINAMATH_CALUDE_two_diamonds_balance_l425_42524


namespace NUMINAMATH_CALUDE_julieta_total_spent_l425_42564

-- Define the original prices and price changes
def original_backpack_price : ℕ := 50
def original_binder_price : ℕ := 20
def backpack_price_increase : ℕ := 5
def binder_price_reduction : ℕ := 2
def number_of_binders : ℕ := 3

-- Define the theorem
theorem julieta_total_spent :
  let new_backpack_price := original_backpack_price + backpack_price_increase
  let new_binder_price := original_binder_price - binder_price_reduction
  let total_spent := new_backpack_price + number_of_binders * new_binder_price
  total_spent = 109 := by sorry

end NUMINAMATH_CALUDE_julieta_total_spent_l425_42564


namespace NUMINAMATH_CALUDE_boat_stream_speed_l425_42578

/-- Proves that the speed of the stream is 6 kmph given the conditions of the boat problem -/
theorem boat_stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ)
  (h_boat_speed : boat_speed = 8)
  (h_distance : distance = 210)
  (h_total_time : total_time = 120)
  (h_equation : (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = total_time)
  : stream_speed = 6 := by
  sorry

#check boat_stream_speed

end NUMINAMATH_CALUDE_boat_stream_speed_l425_42578


namespace NUMINAMATH_CALUDE_tan_x_equals_zero_l425_42514

theorem tan_x_equals_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x/2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_x_equals_zero_l425_42514


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l425_42559

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of ways to arrange n crates with given counts of each orientation -/
def arrangementCount (n a b c : ℕ) : ℕ := sorry

/-- The probability of stacking crates to achieve a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates targetHeight : ℕ) : ℚ :=
  let totalArrangements := 3^numCrates
  let validArrangements := 
    arrangementCount numCrates 8 0 4 + 
    arrangementCount numCrates 6 3 3 + 
    arrangementCount numCrates 4 6 2 + 
    arrangementCount numCrates 2 9 1 + 
    arrangementCount numCrates 0 12 0
  validArrangements / totalArrangements

theorem crate_stacking_probability : 
  let dimensions := CrateDimensions.mk 3 4 6
  stackProbability dimensions 12 48 = 37522 / 531441 := by sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l425_42559


namespace NUMINAMATH_CALUDE_sphere_surface_area_l425_42587

theorem sphere_surface_area (a b c : ℝ) (h1 : a * b * c = Real.sqrt 6) 
  (h2 : a * b = Real.sqrt 2) (h3 : b * c = Real.sqrt 3) : 
  4 * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^2 = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l425_42587


namespace NUMINAMATH_CALUDE_two_thousand_plus_sqrt_two_thousand_one_in_A_l425_42519

-- Define the set A
variable (A : Set ℝ)

-- Define the conditions
axiom one_in_A : 1 ∈ A
axiom square_in_A : ∀ x : ℝ, x ∈ A → x^2 ∈ A
axiom inverse_square_in_A : ∀ x : ℝ, (x^2 - 4*x + 4) ∈ A → x ∈ A

-- State the theorem
theorem two_thousand_plus_sqrt_two_thousand_one_in_A :
  (2000 + Real.sqrt 2001) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_plus_sqrt_two_thousand_one_in_A_l425_42519


namespace NUMINAMATH_CALUDE_ten_thousand_scientific_notation_l425_42507

/-- Scientific notation representation of 10,000 -/
def scientific_notation_10000 : ℝ := 1 * (10 ^ 4)

/-- Theorem stating that 10,000 is equal to its scientific notation representation -/
theorem ten_thousand_scientific_notation : 
  (10000 : ℝ) = scientific_notation_10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_scientific_notation_l425_42507


namespace NUMINAMATH_CALUDE_positive_rational_function_uniqueness_l425_42549

/-- A function from positive rationals to positive rationals -/
def PositiveRationalFunction := {f : ℚ → ℚ // ∀ x, 0 < x → 0 < f x}

/-- The property that f(x+1) = f(x) + 1 for all positive rationals x -/
def HasUnitPeriod (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (x + 1) = f.val x + 1

/-- The property that f(1/x) = 1/f(x) for all positive rationals x -/
def HasInverseProperty (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (1 / x) = 1 / f.val x

/-- The main theorem: if a function satisfies both properties, it must be the identity function -/
theorem positive_rational_function_uniqueness (f : PositiveRationalFunction) 
    (h1 : HasUnitPeriod f) (h2 : HasInverseProperty f) : 
    ∀ x : ℚ, 0 < x → f.val x = x := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_function_uniqueness_l425_42549


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l425_42516

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n % 5 = 0) ∧
  (n % 11 = 7) ∧
  (n % 7 = 4) ∧
  (n % 9 = 4) ∧
  (∀ m : ℕ, 
    (m ≥ 1000 ∧ m < 10000) ∧ 
    (m % 5 = 0) ∧
    (m % 11 = 7) ∧
    (m % 7 = 4) ∧
    (m % 9 = 4) →
    n ≤ m) ∧
  n = 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l425_42516


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l425_42521

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l425_42521


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l425_42504

-- Define the radius of the circle in feet
def radius : ℝ := 14

-- Define the length of a pencil in feet
def pencil_length : ℝ := 0.5

-- Theorem: The number of pencils that can be placed end-to-end across the diameter is 56
theorem pencils_across_diameter : 
  ⌊(2 * radius) / pencil_length⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l425_42504
