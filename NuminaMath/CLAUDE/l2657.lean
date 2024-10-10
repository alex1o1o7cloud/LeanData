import Mathlib

namespace smallest_n_for_roots_of_unity_l2657_265782

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f(z) -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n-th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- Theorem: 9 is the smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  ∃ (n : ℕ), n > 0 ∧ roots_of_f ⊆ nth_roots_of_unity n ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(roots_of_f ⊆ nth_roots_of_unity m) ∧
  n = 9 :=
sorry

end smallest_n_for_roots_of_unity_l2657_265782


namespace toms_weekly_fluid_intake_l2657_265732

/-- The number of cans of soda Tom drinks per day -/
def soda_cans_per_day : ℕ := 5

/-- The number of ounces in each can of soda -/
def oz_per_soda_can : ℕ := 12

/-- The number of ounces of water Tom drinks per day -/
def water_oz_per_day : ℕ := 64

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Tom's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℕ := 
  (soda_cans_per_day * oz_per_soda_can + water_oz_per_day) * days_in_week

theorem toms_weekly_fluid_intake : weekly_fluid_intake = 868 := by
  sorry

end toms_weekly_fluid_intake_l2657_265732


namespace vector_addition_l2657_265735

/-- Given two 2D vectors a and b, prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by sorry

end vector_addition_l2657_265735


namespace socks_selection_with_red_l2657_265770

def total_socks : ℕ := 10
def red_socks : ℕ := 1
def socks_to_choose : ℕ := 4

theorem socks_selection_with_red :
  (Nat.choose total_socks socks_to_choose) - 
  (Nat.choose (total_socks - red_socks) socks_to_choose) = 84 := by
  sorry

end socks_selection_with_red_l2657_265770


namespace carmen_paint_area_l2657_265766

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in Carmen's house -/
def total_paint_area (num_rooms : ℕ) (room_dims : RoomDimensions) (unpainted_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area := wall_area - unpainted_area
  (num_rooms : ℝ) * paintable_area

/-- Theorem stating that the total area to be painted in Carmen's house is 1408 square feet -/
theorem carmen_paint_area :
  let room_dims : RoomDimensions := ⟨15, 12, 8⟩
  let num_rooms : ℕ := 4
  let unpainted_area : ℝ := 80
  total_paint_area num_rooms room_dims unpainted_area = 1408 := by
  sorry

end carmen_paint_area_l2657_265766


namespace k_range_l2657_265741

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k < 0 := by
sorry

end k_range_l2657_265741


namespace sum_of_roots_l2657_265730

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2000*x = 1997)
  (hy : y^3 - 3*y^2 + 2000*y = 1999) : 
  x + y = 2 := by
  sorry

end sum_of_roots_l2657_265730


namespace equation_represents_pair_of_straight_lines_l2657_265799

/-- The equation representing the graph -/
def equation (x y : ℝ) : Prop := 9 * x^2 - y^2 - 6 * x = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end equation_represents_pair_of_straight_lines_l2657_265799


namespace soccer_ball_max_height_l2657_265778

/-- The height function of a soccer ball's path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the soccer ball -/
def max_height : ℝ := 40

theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end soccer_ball_max_height_l2657_265778


namespace change_in_expression_l2657_265728

/-- The original function -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

/-- The change in f when x is replaced by x + b -/
def delta_plus (x b : ℝ) : ℝ := f (x + b) - f x

/-- The change in f when x is replaced by x - b -/
def delta_minus (x b : ℝ) : ℝ := f (x - b) - f x

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  (delta_plus x b = 3*x^2*b + 3*x*b^2 + b^3 - 2*b) ∧
  (delta_minus x b = -3*x^2*b + 3*x*b^2 - b^3 + 2*b) := by
  sorry

end change_in_expression_l2657_265728


namespace orange_price_is_60_cents_l2657_265723

/-- Represents the price and quantity of fruits -/
structure FruitInfo where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  final_avg_price : ℚ
  removed_oranges : ℕ

/-- Theorem stating that given the conditions, the price of each orange is 60 cents -/
theorem orange_price_is_60_cents (info : FruitInfo) 
    (h1 : info.apple_price = 40/100)
    (h2 : info.total_fruits = 10)
    (h3 : info.initial_avg_price = 54/100)
    (h4 : info.final_avg_price = 48/100)
    (h5 : info.removed_oranges = 5) :
    info.orange_price = 60/100 := by
  sorry

#check orange_price_is_60_cents

end orange_price_is_60_cents_l2657_265723


namespace liam_commute_speed_l2657_265713

theorem liam_commute_speed (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) 
  (h1 : distance = 40)
  (h2 : actual_speed = 60)
  (h3 : early_time = 4/60) :
  let ideal_speed := actual_speed - 5
  let actual_time := distance / actual_speed
  let ideal_time := distance / ideal_speed
  ideal_time - actual_time = early_time := by sorry

end liam_commute_speed_l2657_265713


namespace chess_players_lost_to_ai_l2657_265739

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 120 →
  never_lost_fraction = 2 / 5 →
  (total_players : ℚ) * (1 - never_lost_fraction) = 72 := by
  sorry

end chess_players_lost_to_ai_l2657_265739


namespace free_throw_probability_convergence_l2657_265709

/-- Represents the number of successful shots for a given total number of shots -/
def makes : ℕ → ℕ
| 50 => 28
| 100 => 49
| 150 => 78
| 200 => 102
| 300 => 153
| 400 => 208
| 500 => 255
| _ => 0  -- For any other number of shots, we don't have data

/-- Represents the total number of shots taken -/
def shots : List ℕ := [50, 100, 150, 200, 300, 400, 500]

/-- Calculate the make frequency for a given number of shots -/
def makeFrequency (n : ℕ) : ℚ :=
  (makes n : ℚ) / n

/-- The statement to be proved -/
theorem free_throw_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ n ∈ shots, n ≥ N → |makeFrequency n - 51/100| < ε :=
sorry

end free_throw_probability_convergence_l2657_265709


namespace sports_equipment_pricing_and_discount_l2657_265708

theorem sports_equipment_pricing_and_discount (soccer_price basketball_price : ℝ)
  (h1 : 2 * soccer_price + 3 * basketball_price = 410)
  (h2 : 5 * soccer_price + 2 * basketball_price = 530)
  (h3 : ∃ discount_rate : ℝ, 
    discount_rate * (5 * soccer_price + 5 * basketball_price) = 680 ∧ 
    0 < discount_rate ∧ 
    discount_rate < 1) :
  soccer_price = 70 ∧ basketball_price = 90 ∧ 
  ∃ discount_rate : ℝ, discount_rate * (5 * 70 + 5 * 90) = 680 ∧ discount_rate = 0.85 := by
sorry

end sports_equipment_pricing_and_discount_l2657_265708


namespace side_c_values_simplify_expression_l2657_265769

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the specific triangle with given conditions
def SpecificTriangle : Triangle → Prop
  | t => t.a = 4 ∧ t.b = 6 ∧ t.a + t.b + t.c < 18 ∧ Even (t.a + t.b + t.c)

-- Theorem 1: If the perimeter is less than 18 and even, then c = 4 or c = 6
theorem side_c_values (t : Triangle) (h : SpecificTriangle t) :
  t.c = 4 ∨ t.c = 6 := by
  sorry

-- Theorem 2: Simplification of |a+b-c|+|c-a-b|
theorem simplify_expression (t : Triangle) :
  |t.a + t.b - t.c| + |t.c - t.a - t.b| = 2*t.a + 2*t.b - 2*t.c := by
  sorry

end side_c_values_simplify_expression_l2657_265769


namespace students_liking_both_subjects_l2657_265765

theorem students_liking_both_subjects (total : ℕ) (math : ℕ) (english : ℕ) (neither : ℕ) 
  (h1 : total = 48)
  (h2 : math = 38)
  (h3 : english = 36)
  (h4 : neither = 4) :
  math + english - (total - neither) = 30 := by
  sorry

end students_liking_both_subjects_l2657_265765


namespace combined_share_A_and_C_l2657_265700

def total_amount : ℚ := 15800
def charity_percentage : ℚ := 10 / 100
def savings_percentage : ℚ := 8 / 100
def distribution_ratio : List ℚ := [5, 9, 6, 5]

def remaining_amount : ℚ := total_amount * (1 - charity_percentage - savings_percentage)

def share (ratio : ℚ) : ℚ := (ratio / (distribution_ratio.sum)) * remaining_amount

theorem combined_share_A_and_C : 
  share (distribution_ratio[0]!) + share (distribution_ratio[2]!) = 5700.64 := by
  sorry

end combined_share_A_and_C_l2657_265700


namespace test_scores_l2657_265793

theorem test_scores (keith_score : Real) (larry_multiplier : Real) (danny_difference : Real)
  (h1 : keith_score = 3.5)
  (h2 : larry_multiplier = 3.2)
  (h3 : danny_difference = 5.7) :
  let larry_score := keith_score * larry_multiplier
  let danny_score := larry_score + danny_difference
  keith_score + larry_score + danny_score = 31.6 := by
  sorry

end test_scores_l2657_265793


namespace employee_pay_percentage_l2657_265701

/-- Proof that X is paid 120% of Y's pay given the conditions -/
theorem employee_pay_percentage (total pay_y : ℕ) (pay_x : ℕ) 
  (h1 : total = 880)
  (h2 : pay_y = 400)
  (h3 : pay_x + pay_y = total) :
  (pay_x : ℚ) / pay_y = 120 / 100 := by
  sorry

end employee_pay_percentage_l2657_265701


namespace marbles_lost_found_difference_l2657_265760

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem marbles_lost_found_difference (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 4)
  (h2 : lost = 16)
  (h3 : found = 8) :
  lost - found = 8 := by
  sorry

end marbles_lost_found_difference_l2657_265760


namespace sum_of_pairwise_products_of_cubic_roots_l2657_265755

theorem sum_of_pairwise_products_of_cubic_roots (p q r : ℝ) : 
  (6 * p^3 - 9 * p^2 + 17 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 17 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 17 * r - 12 = 0) →
  p * q + q * r + r * p = 17 / 6 := by
sorry

end sum_of_pairwise_products_of_cubic_roots_l2657_265755


namespace geometric_sequence_sum_l2657_265731

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  (a 5 + a 7 = 6 ∨ a 5 + a 7 = -6) :=
by sorry

end geometric_sequence_sum_l2657_265731


namespace sequence_properties_l2657_265785

/-- Given two sequences a and b with no equal items, and S_n as the sum of the first n terms of a. -/
def Sequence (a b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * b n = S n + 1

theorem sequence_properties
  (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : Sequence a b S)
  (h_a1 : a 1 = 1)
  (h_bn : ∀ n, b n = n / 2)
  (h_geometric : ∃ q ≠ 1, ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d, ∀ n, b (n + 1) = b n + d)
  (h_nonzero : ∀ n, a n ≠ 0) :
  (∃ q ≠ 1, ∀ n, (b n + 1 / (1 - q)) = (b 1 + 1 / (1 - q)) * q^(n - 1)) ∧
  (∀ n ≥ 2, a (n + 1) - a n = a n - a (n - 1) ↔ d = 1 / 2) :=
sorry

#check sequence_properties

end sequence_properties_l2657_265785


namespace ellipse_k_range_l2657_265736

-- Define the curve
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

-- Define the conditions for an ellipse
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
sorry

end ellipse_k_range_l2657_265736


namespace parabola_equation_l2657_265756

/-- A parabola with vertex at the origin and axis of symmetry x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = -2px -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The axis of symmetry is x = 2 -/
  axis_of_symmetry : ∀ y, equation 2 y ↔ equation 2 (-y)

/-- The equation of the parabola is y² = -8x -/
theorem parabola_equation (p : Parabola) : 
  p.equation = fun x y => y^2 = -8*x :=
sorry

end parabola_equation_l2657_265756


namespace min_distance_C₁_C₂_l2657_265705

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop :=
  (x - Real.sqrt 3 / 2)^2 + (y - 1/2)^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 8 = 0

-- State the theorem
theorem min_distance_C₁_C₂ :
  ∃ d : ℝ, d = 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end min_distance_C₁_C₂_l2657_265705


namespace intersection_distance_squared_l2657_265721

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 9
def circle3 (x y : ℝ) : Prop := (x - 5)^2 + (y - 2)^2 = 16

-- Define the intersection points
def intersection (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem intersection_distance_squared :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection x1 y1 ∧
    intersection x2 y2 ∧
    circle3 x1 y1 ∧
    circle3 x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 224 / 9 :=
by sorry

end intersection_distance_squared_l2657_265721


namespace square_neg_sqrt_three_eq_three_l2657_265788

theorem square_neg_sqrt_three_eq_three : (-Real.sqrt 3)^2 = 3 := by
  sorry

end square_neg_sqrt_three_eq_three_l2657_265788


namespace binomial_cube_constant_l2657_265711

theorem binomial_cube_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → 
  a = 8 := by
sorry

end binomial_cube_constant_l2657_265711


namespace f_one_intersection_l2657_265737

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (3*a - 2)*x + a - 1

/-- Theorem stating the condition for f(x) to have exactly one intersection with x-axis in (-1,3) -/
theorem f_one_intersection (a : ℝ) : 
  (∃! x : ℝ, x > -1 ∧ x < 3 ∧ f a x = 0) ↔ (a ≤ -1/5 ∨ a ≥ 1) :=
sorry

end f_one_intersection_l2657_265737


namespace go_board_sales_solution_l2657_265745

/-- Represents the sales data for a month -/
structure MonthlySales where
  typeA : ℕ
  typeB : ℕ
  revenue : ℕ

/-- Represents the Go board sales problem -/
structure GoBoardSales where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  month1 : MonthlySales
  month2 : MonthlySales
  totalBudget : ℕ
  totalSets : ℕ

/-- Theorem stating the solution to the Go board sales problem -/
theorem go_board_sales_solution (sales : GoBoardSales)
  (h1 : sales.purchasePriceA = 200)
  (h2 : sales.purchasePriceB = 170)
  (h3 : sales.month1 = ⟨3, 5, 1800⟩)
  (h4 : sales.month2 = ⟨4, 10, 3100⟩)
  (h5 : sales.totalBudget = 5400)
  (h6 : sales.totalSets = 30) :
  ∃ (sellingPriceA sellingPriceB maxTypeA : ℕ),
    sellingPriceA = 250 ∧
    sellingPriceB = 210 ∧
    maxTypeA = 10 ∧
    maxTypeA * sales.purchasePriceA + (sales.totalSets - maxTypeA) * sales.purchasePriceB ≤ sales.totalBudget ∧
    maxTypeA * (sellingPriceA - sales.purchasePriceA) + (sales.totalSets - maxTypeA) * (sellingPriceB - sales.purchasePriceB) = 1300 :=
by sorry


end go_board_sales_solution_l2657_265745


namespace cube_volume_derivative_half_surface_area_l2657_265758

-- Define a cube with edge length x
def cube_volume (x : ℝ) : ℝ := x^3
def cube_surface_area (x : ℝ) : ℝ := 6 * x^2

-- State the theorem
theorem cube_volume_derivative_half_surface_area :
  ∀ x : ℝ, (deriv cube_volume) x = (1/2) * cube_surface_area x :=
by
  sorry

end cube_volume_derivative_half_surface_area_l2657_265758


namespace min_value_trigonometric_expression_l2657_265754

theorem min_value_trigonometric_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 235.97 := by
  sorry

end min_value_trigonometric_expression_l2657_265754


namespace function_characterization_l2657_265784

/-- A function from ℚ × ℚ to ℚ satisfying the given property -/
def FunctionProperty (f : ℚ × ℚ → ℚ) : Prop :=
  ∀ x y z : ℚ, f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)

/-- The theorem stating the form of functions satisfying the property -/
theorem function_characterization (f : ℚ × ℚ → ℚ) (h : FunctionProperty f) :
    ∃ a b : ℚ, ∀ x y : ℚ, f (x, y) = a * y^2 + 2 * a * x * y + b * y :=
  sorry

end function_characterization_l2657_265784


namespace triangle_side_length_l2657_265787

/-- Given a triangle ABC with angle A = 60°, area = √3, and b + c = 6, prove that side length a = 2√6 -/
theorem triangle_side_length (b c : ℝ) (h1 : b + c = 6) (h2 : (1/2) * b * c * (Real.sqrt 3 / 2) = Real.sqrt 3) : 
  Real.sqrt (b^2 + c^2 - b * c) = 2 * Real.sqrt 6 := by
  sorry

end triangle_side_length_l2657_265787


namespace clock_angle_at_four_l2657_265738

/-- The number of degrees in a complete circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour mark on a clock -/
def degrees_per_hour : ℕ := circle_degrees / clock_hours

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- The smaller angle formed by the clock hands at the target hour -/
def smaller_angle (h : ℕ) : ℕ := min (h * degrees_per_hour) (circle_degrees - h * degrees_per_hour)

theorem clock_angle_at_four :
  smaller_angle target_hour = 120 := by sorry

end clock_angle_at_four_l2657_265738


namespace max_students_equal_distribution_l2657_265716

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 640) (h_pencils : pencils = 520) :
  Nat.gcd pens pencils = 40 := by
  sorry

end max_students_equal_distribution_l2657_265716


namespace pure_imaginary_ratio_l2657_265734

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -5/3 := by
sorry

end pure_imaginary_ratio_l2657_265734


namespace fourth_sample_is_31_l2657_265725

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  known_samples : Finset Nat

/-- Calculates the sample interval for a systematic sampling. -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.total_students / s.sample_size

/-- Theorem: In a systematic sampling of 4 from 56 students, if 3, 17, and 45 are sampled, the fourth sample is 31. -/
theorem fourth_sample_is_31 (s : SystematicSampling) 
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = {3, 17, 45}) :
  ∃ (fourth_sample : Nat), fourth_sample ∈ s.known_samples ∪ {31} ∧ 
  (s.known_samples ∪ {fourth_sample}).card = s.sample_size :=
by
  sorry


end fourth_sample_is_31_l2657_265725


namespace tyler_saltwater_animals_l2657_265797

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

/-- Theorem stating that the total number of saltwater animals Tyler has is 512 -/
theorem tyler_saltwater_animals : total_animals = 512 := by
  sorry

end tyler_saltwater_animals_l2657_265797


namespace cell_growth_proof_l2657_265748

/-- The time interval between cell divisions in minutes -/
def division_interval : ℕ := 20

/-- The total time elapsed in minutes -/
def total_time : ℕ := 3 * 60 + 20

/-- The number of cells after one division -/
def cells_after_division : ℕ := 2

/-- The number of cells after a given number of divisions -/
def cells_after_divisions (n : ℕ) : ℕ := cells_after_division ^ n

theorem cell_growth_proof :
  cells_after_divisions (total_time / division_interval) = 1024 :=
by sorry

end cell_growth_proof_l2657_265748


namespace symmetric_circle_equation_l2657_265775

/-- Given a circle with equation (x-1)^2+(y+1)^2=4, its symmetric circle with respect to the origin has the equation (x+1)^2+(y-1)^2=4 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + (y + 1)^2 = 4) →
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 4) :=
by sorry

end symmetric_circle_equation_l2657_265775


namespace even_function_implies_m_equals_one_l2657_265726

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = x^2 + (m-1)x - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m-1)*x - 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end even_function_implies_m_equals_one_l2657_265726


namespace specific_arithmetic_series_sum_l2657_265757

/-- Sum of an arithmetic series with given parameters -/
def arithmeticSeriesSum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series (-42) + (-40) + ⋯ + 0 is -462 -/
theorem specific_arithmetic_series_sum :
  arithmeticSeriesSum (-42) 0 2 = -462 := by
  sorry

end specific_arithmetic_series_sum_l2657_265757


namespace minimum_children_for_shared_birthday_l2657_265727

theorem minimum_children_for_shared_birthday (n : ℕ) : 
  (∀ f : Fin n → Fin 366, ∃ d : Fin 366, (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ f i = f j ∧ f j = f k)) ↔ 
  n ≥ 733 :=
sorry

end minimum_children_for_shared_birthday_l2657_265727


namespace birds_on_fence_two_plus_four_birds_l2657_265706

/-- The number of birds on a fence after more birds join -/
theorem birds_on_fence (initial : Nat) (joined : Nat) : 
  initial + joined = initial + joined :=
by sorry

/-- The specific case of 2 initial birds and 4 joined birds -/
theorem two_plus_four_birds : 2 + 4 = 6 :=
by sorry

end birds_on_fence_two_plus_four_birds_l2657_265706


namespace power_of_product_l2657_265724

theorem power_of_product (a b : ℝ) : (-5 * a^3 * b)^2 = 25 * a^6 * b^2 := by
  sorry

end power_of_product_l2657_265724


namespace unique_solution_l2657_265768

def satisfiesConditions (n : ℕ) : Prop :=
  ∃ k m p : ℕ, n = 2 * k + 1 ∧ n = 3 * m - 1 ∧ n = 5 * p + 2

theorem unique_solution : 
  satisfiesConditions 47 ∧ 
  (¬ satisfiesConditions 39) ∧ 
  (¬ satisfiesConditions 40) ∧ 
  (¬ satisfiesConditions 49) ∧ 
  (¬ satisfiesConditions 53) :=
sorry

end unique_solution_l2657_265768


namespace brandon_cash_sales_l2657_265749

theorem brandon_cash_sales (total_sales : ℝ) (credit_ratio : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_ratio = 2/5 →
  cash_sales = total_sales * (1 - credit_ratio) →
  cash_sales = 48 := by
sorry

end brandon_cash_sales_l2657_265749


namespace arithmetic_sequence_difference_difference_1004_1001_l2657_265710

/-- Given an arithmetic sequence with first term 3 and common difference 7,
    the positive difference between the 1004th term and the 1001st term is 21. -/
theorem arithmetic_sequence_difference : ℕ → ℕ :=
  fun n => 3 + (n - 1) * 7

#check arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21

theorem difference_1004_1001 :
  arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21 := by
  sorry

end arithmetic_sequence_difference_difference_1004_1001_l2657_265710


namespace product_of_fractions_l2657_265719

theorem product_of_fractions : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end product_of_fractions_l2657_265719


namespace phil_quarters_l2657_265715

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_amount : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem phil_quarters : 
  ⌊remaining_amount * quarters_in_dollar⌋ = 97 := by
  sorry

end phil_quarters_l2657_265715


namespace parallel_line_slope_l2657_265773

/-- Given a line parallel to 3x - 6y = 21, its slope is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : a * x - b * y = c) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a, b, c) = (3 * k, 6 * k, 21 * k)) :
  (a / b : ℝ) = 1 / 2 := by
sorry

end parallel_line_slope_l2657_265773


namespace tan_value_from_trig_ratio_l2657_265786

theorem tan_value_from_trig_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) :
  Real.tan α = -12/5 := by
  sorry

end tan_value_from_trig_ratio_l2657_265786


namespace garden_flowers_equality_l2657_265795

/-- Given a garden with white and red flowers, calculate the number of additional red flowers needed to make their quantities equal. -/
def additional_red_flowers (white : ℕ) (red : ℕ) : ℕ :=
  if white > red then white - red else 0

/-- Theorem: In a garden with 555 white flowers and 347 red flowers, 208 additional red flowers are needed to make their quantities equal. -/
theorem garden_flowers_equality : additional_red_flowers 555 347 = 208 := by
  sorry

end garden_flowers_equality_l2657_265795


namespace no_half_parallel_diagonals_l2657_265776

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- n ≥ 3 for a valid polygon
  sides_ge_three : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon n) : ℕ :=
  n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon -/
def num_parallel_diagonals (p : RegularPolygon n) : ℕ :=
  if n % 2 = 0 then
    (n / 2 - 1)
  else
    0

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (n : ℕ) (p : RegularPolygon n) :
  2 * (num_parallel_diagonals p) ≠ num_diagonals p :=
sorry

end no_half_parallel_diagonals_l2657_265776


namespace grid_toothpick_count_l2657_265789

/-- Calculates the number of toothpicks in a rectangular grid with a missing row and column -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_lines := height
  let vertical_lines := width
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * (height - 1)
  horizontal_toothpicks + vertical_toothpicks

/-- Theorem stating that a 25x15 grid with a missing row and column uses 735 toothpicks -/
theorem grid_toothpick_count : toothpick_count 25 15 = 735 := by
  sorry

#eval toothpick_count 25 15

end grid_toothpick_count_l2657_265789


namespace book_problem_solution_l2657_265743

def book_problem (total_cost selling_price_1 cost_1 loss_percent : ℚ) : Prop :=
  let cost_2 := total_cost - cost_1
  let selling_price_2 := selling_price_1
  let gain_percent := (selling_price_2 - cost_2) / cost_2 * 100
  (total_cost = 540) ∧
  (cost_1 = 315) ∧
  (selling_price_1 = cost_1 * (1 - loss_percent / 100)) ∧
  (loss_percent = 15) ∧
  (gain_percent = 19)

theorem book_problem_solution :
  ∃ (total_cost selling_price_1 cost_1 loss_percent : ℚ),
    book_problem total_cost selling_price_1 cost_1 loss_percent :=
sorry

end book_problem_solution_l2657_265743


namespace cheeseBalls35ozBarrel_l2657_265798

/-- Calculates the number of cheese balls in a barrel given its size in ounces -/
def cheeseBallsInBarrel (barrelSize : ℕ) : ℕ :=
  let servingsIn24oz : ℕ := 60
  let cheeseBallsPerServing : ℕ := 12
  let cheeseBallsPer24oz : ℕ := servingsIn24oz * cheeseBallsPerServing
  let cheeseBallsPerOz : ℕ := cheeseBallsPer24oz / 24
  barrelSize * cheeseBallsPerOz

theorem cheeseBalls35ozBarrel :
  cheeseBallsInBarrel 35 = 1050 :=
by sorry

end cheeseBalls35ozBarrel_l2657_265798


namespace inequality_solution_l2657_265707

theorem inequality_solution (x : ℝ) : 
  1 / (x^3 + 1) > 4 / x + 2 / 5 ↔ -1 < x ∧ x < 0 :=
sorry

end inequality_solution_l2657_265707


namespace special_polynomial_value_l2657_265762

/-- A polynomial function of degree n satisfying f(k) = k/(k+1) for k = 0, 1, ..., n -/
def SpecialPolynomial (n : ℕ) (f : ℝ → ℝ) : Prop :=
  (∃ p : Polynomial ℝ, Polynomial.degree p = n ∧ f = p.eval) ∧
  (∀ k : ℕ, k ≤ n → f k = k / (k + 1))

/-- The main theorem stating the value of f(n+1) for a SpecialPolynomial -/
theorem special_polynomial_value (n : ℕ) (f : ℝ → ℝ) 
  (h : SpecialPolynomial n f) : 
  f (n + 1) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end special_polynomial_value_l2657_265762


namespace andys_future_age_ratio_l2657_265750

def rahims_current_age : ℕ := 6
def age_difference : ℕ := 1
def years_in_future : ℕ := 5

theorem andys_future_age_ratio :
  (rahims_current_age + age_difference + years_in_future) / rahims_current_age = 2 := by
  sorry

end andys_future_age_ratio_l2657_265750


namespace marble_distribution_l2657_265752

theorem marble_distribution (y : ℚ) : 
  (4 * y + 2) + (2 * y) + (y + 3) = 31 → y = 26 / 7 := by
sorry

end marble_distribution_l2657_265752


namespace parallel_lines_a_value_l2657_265712

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x + y + b₂ = 0) ↔ m₁ = -m₂

/-- The value of a when two lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 2 * x - y + 1 = 0 ↔ x + a * y + 2 = 0) → a = -1/2 := by
  sorry

end parallel_lines_a_value_l2657_265712


namespace intersection_points_concyclic_and_share_radical_axis_l2657_265746

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Given two circles and two lines, returns the new circle formed by the intersection of chords -/
def newCircle (C₁ C₂ : Circle) (L₁ L₂ : Line) : Circle :=
  sorry

/-- Checks if three circles share a common radical axis -/
def shareCommonRadicalAxis (C₁ C₂ C₃ : Circle) : Prop :=
  sorry

/-- Main theorem: The four intersection points of chords lie on a new circle that shares
    a common radical axis with the original two circles -/
theorem intersection_points_concyclic_and_share_radical_axis
  (C₁ C₂ : Circle) (L₁ L₂ : Line) :
  let C := newCircle C₁ C₂ L₁ L₂
  shareCommonRadicalAxis C₁ C₂ C :=
by
  sorry

end intersection_points_concyclic_and_share_radical_axis_l2657_265746


namespace circle_power_theorem_l2657_265772

/-- Given a circle with center O and radius R, and points A and B on the circle,
    for any point P on line AB, PA * PB = OP^2 - R^2 in terms of algebraic lengths -/
theorem circle_power_theorem (O : ℝ × ℝ) (R : ℝ) (A B P : ℝ × ℝ) :
  (∀ X : ℝ × ℝ, dist O X = R → (X = A ∨ X = B)) →
  (∃ t : ℝ, P = (1 - t) • A + t • B) →
  (dist P A * dist P B : ℝ) = dist O P ^ 2 - R ^ 2 := by
  sorry

end circle_power_theorem_l2657_265772


namespace inequality_proof_l2657_265740

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (a^2 / (1 + b * c)) + (b^2 / (1 + c * a)) + (c^2 / (1 + a * b)) ≥ 3/2 := by
  sorry

end inequality_proof_l2657_265740


namespace tulip_ratio_l2657_265763

/-- Given the number of red tulips for eyes and smile, and the total number of tulips,
    prove that the ratio of yellow tulips in the background to red tulips in the smile is 9:1 -/
theorem tulip_ratio (red_tulips_per_eye : ℕ) (red_tulips_smile : ℕ) (total_tulips : ℕ) :
  red_tulips_per_eye = 8 →
  red_tulips_smile = 18 →
  total_tulips = 196 →
  (total_tulips - (2 * red_tulips_per_eye + red_tulips_smile)) / red_tulips_smile = 9 := by
  sorry

end tulip_ratio_l2657_265763


namespace ceiling_distance_to_square_existence_l2657_265774

theorem ceiling_distance_to_square_existence : 
  ∃ (A : ℝ), ∀ (n : ℕ), 
    ∃ (m : ℕ), (⌈A^n⌉ : ℝ) - (m^2 : ℝ) = 2 ∧ 
    ∀ (k : ℕ), k > m → (k^2 : ℝ) > ⌈A^n⌉ :=
by sorry

end ceiling_distance_to_square_existence_l2657_265774


namespace vector_angle_difference_l2657_265796

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (ha : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (hb : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h_eq : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
  sorry

end vector_angle_difference_l2657_265796


namespace cat_and_mouse_positions_l2657_265720

/-- Represents the position of the cat or mouse -/
inductive Position
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The number of moves in the problem -/
def totalMoves : ℕ := 315

/-- The length of the cat's movement cycle -/
def catCycleLength : ℕ := 4

/-- The length of the mouse's movement cycle -/
def mouseCycleLength : ℕ := 8

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : Position :=
  match moves % catCycleLength with
  | 0 => Position.TopLeft
  | 1 => Position.TopRight
  | 2 => Position.BottomRight
  | 3 => Position.BottomLeft
  | _ => Position.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : Position :=
  match moves % mouseCycleLength with
  | 0 => Position.TopMiddle
  | 1 => Position.TopRight
  | 2 => Position.RightMiddle
  | 3 => Position.BottomRight
  | 4 => Position.BottomMiddle
  | 5 => Position.BottomLeft
  | 6 => Position.LeftMiddle
  | 7 => Position.TopLeft
  | _ => Position.TopMiddle  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions : 
  catPosition totalMoves = Position.BottomRight ∧ 
  mousePosition totalMoves = Position.RightMiddle := by
  sorry

end cat_and_mouse_positions_l2657_265720


namespace ball_selection_probabilities_l2657_265771

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from a bag -/
def probRed (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The bags used in the problem -/
def bagA : Bag := ⟨8, 4⟩
def bagB : Bag := ⟨6, 6⟩

/-- The theorem to be proved -/
theorem ball_selection_probabilities :
  /- The probability of selecting two balls of the same color is 1/2 -/
  (probWhite bagA * probWhite bagB + probRed bagA * probRed bagB = 1/2) ∧
  /- The probability of selecting at least one red ball is 2/3 -/
  (1 - probWhite bagA * probWhite bagB = 2/3) := by
  sorry


end ball_selection_probabilities_l2657_265771


namespace ray_gave_ratio_l2657_265703

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The initial amount Ray has in cents -/
def initial_amount : ℕ := 95

/-- The amount Ray gives to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving to both Peter and Randi -/
def nickels_left : ℕ := 4

/-- The ratio of the amount Ray gave to Randi to the amount he gave to Peter -/
def ratio_randi_to_peter : ℚ := 2 / 1

theorem ray_gave_ratio :
  let initial_nickels := initial_amount / nickel_value
  let nickels_to_peter := amount_to_peter / nickel_value
  let nickels_to_randi := initial_nickels - nickels_to_peter - nickels_left
  let amount_to_randi := nickels_to_randi * nickel_value
  (amount_to_randi : ℚ) / amount_to_peter = ratio_randi_to_peter :=
by sorry

end ray_gave_ratio_l2657_265703


namespace total_pets_is_54_l2657_265751

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_total : ℕ := teddy_dogs + teddy_cats
  let ben_total : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_total : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_total + ben_total + dave_total

/-- Theorem stating that the total number of pets is 54 -/
theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end total_pets_is_54_l2657_265751


namespace fraction_product_simplification_l2657_265780

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 * 7 / 9 = 1 / 3 := by sorry

end fraction_product_simplification_l2657_265780


namespace systematic_sampling_interval_l2657_265714

/-- The segment interval for systematic sampling given a population and sample size -/
def segment_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The segment interval for systematic sampling of 100 students from a population of 2400 is 24 -/
theorem systematic_sampling_interval :
  segment_interval 2400 100 = 24 := by
  sorry

end systematic_sampling_interval_l2657_265714


namespace intersection_point_k_value_l2657_265759

theorem intersection_point_k_value (k : ℝ) : 
  (∃ x y : ℝ, x - 2*y - 2*k = 0 ∧ 2*x - 3*y - k = 0 ∧ 3*x - y = 0) → k = 0 :=
by sorry

end intersection_point_k_value_l2657_265759


namespace total_onions_grown_l2657_265729

/-- The total number of onions grown by Sara, Sally, and Fred is 18. -/
theorem total_onions_grown (sara_onions : ℕ) (sally_onions : ℕ) (fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 :=
by sorry

end total_onions_grown_l2657_265729


namespace carpet_border_area_l2657_265783

/-- Calculates the area of a carpet border in a rectangular room -/
theorem carpet_border_area 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (border_width : ℝ) 
  (h1 : room_length = 12) 
  (h2 : room_width = 10) 
  (h3 : border_width = 2) : 
  room_length * room_width - (room_length - 2 * border_width) * (room_width - 2 * border_width) = 72 := by
  sorry

#check carpet_border_area

end carpet_border_area_l2657_265783


namespace exists_coprime_in_ten_consecutive_integers_l2657_265791

theorem exists_coprime_in_ten_consecutive_integers (n : ℤ) :
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end exists_coprime_in_ten_consecutive_integers_l2657_265791


namespace paths_from_A_to_D_l2657_265747

/-- Represents a point in the network -/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents the number of direct paths between two points -/
def direct_paths (p q : Point) : ℕ :=
  match p, q with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ := 10

theorem paths_from_A_to_D :
  total_paths = 
    (direct_paths Point.A Point.B * direct_paths Point.B Point.C * direct_paths Point.C Point.D) +
    (direct_paths Point.A Point.C * direct_paths Point.C Point.D) :=
by sorry

end paths_from_A_to_D_l2657_265747


namespace optimal_selling_price_l2657_265767

/-- Represents the problem of finding the optimal selling price for a product --/
structure PricingProblem where
  costPrice : ℝ        -- Cost price per kilogram
  initialPrice : ℝ     -- Initial selling price per kilogram
  initialSales : ℝ     -- Initial monthly sales in kilograms
  salesDecrease : ℝ    -- Decrease in sales per 1 yuan price increase
  availableCapital : ℝ -- Available capital
  targetProfit : ℝ     -- Target profit

/-- Calculates the profit for a given selling price --/
def calculateProfit (p : PricingProblem) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  (sellingPrice - p.costPrice) * salesVolume

/-- Checks if the capital required for a given selling price is within the available capital --/
def isCapitalSufficient (p : PricingProblem) (sellingPrice : ℝ) : Prop :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  p.costPrice * salesVolume ≤ p.availableCapital

/-- Theorem stating that the optimal selling price is 80 yuan --/
theorem optimal_selling_price (p : PricingProblem) 
  (h1 : p.costPrice = 40)
  (h2 : p.initialPrice = 50)
  (h3 : p.initialSales = 500)
  (h4 : p.salesDecrease = 10)
  (h5 : p.availableCapital = 10000)
  (h6 : p.targetProfit = 8000) :
  ∃ (x : ℝ), x = 80 ∧ 
    calculateProfit p x = p.targetProfit ∧ 
    isCapitalSufficient p x ∧
    ∀ (y : ℝ), y ≠ x → calculateProfit p y = p.targetProfit → ¬(isCapitalSufficient p y) := by
  sorry


end optimal_selling_price_l2657_265767


namespace odd_function_a_value_l2657_265792

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then 2^x - a*x else -(2^(-x)) - a*x

-- State the theorem
theorem odd_function_a_value :
  -- f is an odd function
  (∀ x, f a x = -(f a (-x))) →
  -- f(x) = 2^x - ax when x < 0
  (∀ x, x < 0 → f a x = 2^x - a*x) →
  -- f(2) = 2
  f a 2 = 2 →
  -- Then a = -9/8
  a = -9/8 :=
sorry

end odd_function_a_value_l2657_265792


namespace mans_swimming_speed_l2657_265777

/-- The speed of a man swimming in still water, given his downstream and upstream performances. -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 42) (h2 : upstream_distance = 18) (h3 : time = 3) : 
  ∃ (v_m v_s : ℝ), v_m = 10 ∧ 
    downstream_distance = (v_m + v_s) * time ∧ 
    upstream_distance = (v_m - v_s) * time :=
by
  sorry

#check mans_swimming_speed

end mans_swimming_speed_l2657_265777


namespace qian_receives_23_yuan_l2657_265761

/-- Represents the amount of money paid by each person for each meal -/
structure MealPayments where
  zhao_lunch : ℕ
  qian_lunch : ℕ
  sun_lunch : ℕ
  zhao_dinner : ℕ
  qian_dinner : ℕ

/-- Calculates the amount Qian should receive from Li -/
def amount_qian_receives (payments : MealPayments) : ℕ :=
  let total_cost := payments.zhao_lunch + payments.qian_lunch + payments.sun_lunch +
                    payments.zhao_dinner + payments.qian_dinner
  let cost_per_person := total_cost / 4
  let qian_paid := payments.qian_lunch + payments.qian_dinner
  qian_paid - cost_per_person

/-- The main theorem stating that Qian should receive 23 yuan from Li -/
theorem qian_receives_23_yuan (payments : MealPayments) 
  (h1 : payments.zhao_lunch = 23)
  (h2 : payments.qian_lunch = 41)
  (h3 : payments.sun_lunch = 56)
  (h4 : payments.zhao_dinner = 48)
  (h5 : payments.qian_dinner = 32) :
  amount_qian_receives payments = 23 := by
  sorry


end qian_receives_23_yuan_l2657_265761


namespace seashells_total_l2657_265717

theorem seashells_total (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end seashells_total_l2657_265717


namespace det_cyclic_matrix_zero_l2657_265733

theorem det_cyclic_matrix_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) →
  (b^4 + p*b^2 + q*b + r = 0) →
  (c^4 + p*c^2 + q*c + r = 0) →
  (d^4 + p*d^2 + q*d + r = 0) →
  Matrix.det (
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]]
  ) = 0 := by
  sorry

end det_cyclic_matrix_zero_l2657_265733


namespace prob_missing_one_equals_two_prob_decreasing_sequence_l2657_265722

-- Define the number of items in the collection
def n : ℕ := 10

-- Define the probability of finding each item
def p : ℝ := 0.1

-- Define the probability of missing exactly k items in the second set
-- when the first set is completed
noncomputable def p_k (k : ℕ) : ℝ := sorry

-- Theorem 1: p_1 = p_2
theorem prob_missing_one_equals_two : p_k 1 = p_k 2 := sorry

-- Theorem 2: p_2 > p_3 > p_4 > ... > p_10
theorem prob_decreasing_sequence : 
  ∀ k₁ k₂ : ℕ, 2 ≤ k₁ → k₁ < k₂ → k₂ ≤ n → p_k k₁ > p_k k₂ := sorry

end prob_missing_one_equals_two_prob_decreasing_sequence_l2657_265722


namespace total_count_is_six_l2657_265704

def problem (total_count : ℕ) (group1_count group2_count group3_count : ℕ) 
  (total_avg group1_avg group2_avg group3_avg : ℚ) : Prop :=
  total_count = group1_count + group2_count + group3_count ∧
  group1_count = 2 ∧
  group2_count = 2 ∧
  group3_count = 2 ∧
  total_avg = 3.95 ∧
  group1_avg = 3.6 ∧
  group2_avg = 3.85 ∧
  group3_avg = 4.400000000000001

theorem total_count_is_six :
  ∃ (total_count : ℕ) (group1_count group2_count group3_count : ℕ)
    (total_avg group1_avg group2_avg group3_avg : ℚ),
  problem total_count group1_count group2_count group3_count
    total_avg group1_avg group2_avg group3_avg ∧
  total_count = 6 :=
by sorry

end total_count_is_six_l2657_265704


namespace quadratic_inequality_solution_set_l2657_265781

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x > 3/2 ∨ x < -1} := by
  sorry

end quadratic_inequality_solution_set_l2657_265781


namespace art_club_election_l2657_265744

theorem art_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 18) 
  (h2 : past_officers = 8) 
  (h3 : positions = 6) :
  (Nat.choose total_candidates positions) - 
  (Nat.choose (total_candidates - past_officers) positions) = 18354 := by
  sorry

end art_club_election_l2657_265744


namespace danny_shorts_count_l2657_265764

/-- Represents the number of clothes washed by Cally and Danny -/
structure ClothesWashed where
  cally_white_shirts : Nat
  cally_colored_shirts : Nat
  cally_shorts : Nat
  cally_pants : Nat
  danny_white_shirts : Nat
  danny_colored_shirts : Nat
  danny_pants : Nat
  total_clothes : Nat

/-- Theorem stating that Danny washed 10 pairs of shorts -/
theorem danny_shorts_count (cw : ClothesWashed)
    (h1 : cw.cally_white_shirts = 10)
    (h2 : cw.cally_colored_shirts = 5)
    (h3 : cw.cally_shorts = 7)
    (h4 : cw.cally_pants = 6)
    (h5 : cw.danny_white_shirts = 6)
    (h6 : cw.danny_colored_shirts = 8)
    (h7 : cw.danny_pants = 6)
    (h8 : cw.total_clothes = 58) :
    ∃ (danny_shorts : Nat), danny_shorts = 10 ∧
    cw.total_clothes = cw.cally_white_shirts + cw.cally_colored_shirts + cw.cally_shorts + cw.cally_pants +
                       cw.danny_white_shirts + cw.danny_colored_shirts + danny_shorts + cw.danny_pants :=
  by sorry


end danny_shorts_count_l2657_265764


namespace sum_of_A_and_B_l2657_265718

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A

-- State the theorem
theorem sum_of_A_and_B 
  (A B : ℝ) 
  (h1 : A ≠ B) 
  (h2 : B - A = 2) 
  (h3 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = -1/2 := by
sorry

end sum_of_A_and_B_l2657_265718


namespace integral_sqrt_minus_sin_equals_pi_l2657_265702

open Set
open MeasureTheory
open Interval

theorem integral_sqrt_minus_sin_equals_pi :
  ∫ x in (Icc (-1) 1), (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end integral_sqrt_minus_sin_equals_pi_l2657_265702


namespace rectangle_area_l2657_265794

theorem rectangle_area (square_area : ℝ) (h1 : square_area = 36) : ∃ (rect_width rect_length rect_area : ℝ),
  rect_width ^ 2 = square_area ∧
  rect_length = 3 * rect_width ∧
  rect_area = rect_width * rect_length ∧
  rect_area = 108 := by
sorry

end rectangle_area_l2657_265794


namespace min_sum_xyz_l2657_265742

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ (a b c : ℤ), (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c :=
by sorry

end min_sum_xyz_l2657_265742


namespace virus_reaches_64MB_l2657_265753

/-- Represents the memory occupation of a virus over time -/
def virusMemory (t : ℕ) : ℕ :=
  2 * 2^t

/-- Represents the time in minutes since boot -/
def timeInMinutes (n : ℕ) : ℕ :=
  3 * n

/-- Theorem stating that the virus occupies 64MB after 45 minutes -/
theorem virus_reaches_64MB :
  ∃ n : ℕ, virusMemory n = 64 * 2^10 ∧ timeInMinutes n = 45 := by
  sorry

end virus_reaches_64MB_l2657_265753


namespace sum_abs_difference_l2657_265790

theorem sum_abs_difference : ∀ (a b : ℤ), a = -5 ∧ b = -4 → abs a + abs b - (a + b) = 18 := by
  sorry

end sum_abs_difference_l2657_265790


namespace infinitely_many_rational_pairs_sum_equals_product_l2657_265779

theorem infinitely_many_rational_pairs_sum_equals_product :
  ∃ f : ℚ → ℚ × ℚ, Function.Injective f ∧ ∀ z, (f z).1 + (f z).2 = (f z).1 * (f z).2 :=
sorry

end infinitely_many_rational_pairs_sum_equals_product_l2657_265779
