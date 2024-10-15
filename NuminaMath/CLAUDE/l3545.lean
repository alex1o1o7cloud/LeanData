import Mathlib

namespace NUMINAMATH_CALUDE_leftover_coin_value_l3545_354559

def quarters_per_roll : ℕ := 45
def dimes_per_roll : ℕ := 55
def james_quarters : ℕ := 95
def james_dimes : ℕ := 173
def lindsay_quarters : ℕ := 140
def lindsay_dimes : ℕ := 285
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 5.30 := by sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l3545_354559


namespace NUMINAMATH_CALUDE_last_day_pages_for_specific_book_l3545_354554

/-- Calculates the number of pages read on the last day to complete a book -/
def pages_on_last_day (total_pages : ℕ) (pages_per_day : ℕ) (break_interval : ℕ) : ℕ :=
  let pages_per_cycle := pages_per_day * (break_interval - 1)
  let full_cycles := (total_pages / pages_per_cycle : ℕ)
  let pages_read_in_full_cycles := full_cycles * pages_per_cycle
  total_pages - pages_read_in_full_cycles

theorem last_day_pages_for_specific_book :
  pages_on_last_day 575 37 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_last_day_pages_for_specific_book_l3545_354554


namespace NUMINAMATH_CALUDE_inverse_direct_proportionality_l3545_354505

/-- Given two real numbers are inversely proportional -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

/-- Given two real numbers are directly proportional -/
def directly_proportional (z y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ z = k * y

/-- Main theorem -/
theorem inverse_direct_proportionality
  (x y z : ℝ → ℝ)
  (h_inv : ∀ t, inversely_proportional (x t) (y t))
  (h_dir : ∀ t, directly_proportional (z t) (y t))
  (h_x : x 9 = 40)
  (h_z : z 10 = 45) :
  x 20 = 18 ∧ z 20 = 90 := by
  sorry


end NUMINAMATH_CALUDE_inverse_direct_proportionality_l3545_354505


namespace NUMINAMATH_CALUDE_base_number_proof_l3545_354521

theorem base_number_proof (x : ℝ) (k : ℕ+) 
  (h1 : x^(k : ℝ) = 4) 
  (h2 : x^(2*(k : ℝ) + 2) = 64) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l3545_354521


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_five_l3545_354507

theorem sqrt_plus_square_zero_implies_diff_five (x y : ℝ) 
  (h : Real.sqrt (x - 3) + (y + 2)^2 = 0) : x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_five_l3545_354507


namespace NUMINAMATH_CALUDE_no_real_roots_l3545_354520

/-- Given a function f and constants a and b, prove that f(ax + b) has no real roots -/
theorem no_real_roots (f : ℝ → ℝ) (a b : ℝ) : 
  (∀ x, f x = x^2 + 2*x + a) →
  (∀ x, f (b*x) = 9*x - 6*x + 2) →
  (∀ x, f (a*x + b) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l3545_354520


namespace NUMINAMATH_CALUDE_work_completion_time_l3545_354527

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B would take to complete the work alone. -/
def days_for_B (days_together days_A_alone : ℚ) : ℚ :=
  1 / (1 / days_together - 1 / days_A_alone)

/-- Theorem stating that if A and B can complete a work in 12 days, and A alone can complete
    the work in 20 days, then B alone will complete the work in 30 days. -/
theorem work_completion_time :
  days_for_B 12 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3545_354527


namespace NUMINAMATH_CALUDE_pressure_calculation_l3545_354562

/-- Prove that given the ideal gas law and specific conditions, the pressure is 1125000 Pa -/
theorem pressure_calculation (v R T V : ℝ) (h1 : v = 30)
  (h2 : R = 8.31) (h3 : T = 300) (h4 : V = 0.06648) :
  v * R * T / V = 1125000 :=
by sorry

end NUMINAMATH_CALUDE_pressure_calculation_l3545_354562


namespace NUMINAMATH_CALUDE_smallest_base_for_90_in_three_digits_l3545_354550

theorem smallest_base_for_90_in_three_digits : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 90 ∧ 90 < b^3) → b ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_90_in_three_digits_l3545_354550


namespace NUMINAMATH_CALUDE_steinburg_marching_band_max_size_l3545_354596

theorem steinburg_marching_band_max_size :
  ∀ n : ℕ,
  (30 * n) % 34 = 6 →
  30 * n < 1200 →
  (∀ m : ℕ, (30 * m) % 34 = 6 → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 720 := by
sorry

end NUMINAMATH_CALUDE_steinburg_marching_band_max_size_l3545_354596


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3545_354546

theorem right_triangle_increase_sides_acute (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → c^2 = a^2 + b^2 → 
  (a + x)^2 + (b + x)^2 > (c + x)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3545_354546


namespace NUMINAMATH_CALUDE_football_tickets_problem_l3545_354512

/-- Given a ticket price and budget, calculates the maximum number of tickets that can be purchased. -/
def max_tickets (price : ℕ) (budget : ℕ) : ℕ :=
  (budget / price : ℕ)

/-- Proves that given a ticket price of 15 and a budget of 120, the maximum number of tickets that can be purchased is 8. -/
theorem football_tickets_problem :
  max_tickets 15 120 = 8 := by
  sorry

end NUMINAMATH_CALUDE_football_tickets_problem_l3545_354512


namespace NUMINAMATH_CALUDE_concert_attendance_difference_l3545_354558

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_difference_l3545_354558


namespace NUMINAMATH_CALUDE_min_xy_value_l3545_354570

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  xy ≥ 64 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x*y = 64 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l3545_354570


namespace NUMINAMATH_CALUDE_cole_gum_count_l3545_354532

/-- The number of people sharing the gum -/
def num_people : ℕ := 3

/-- The number of pieces of gum John has -/
def john_gum : ℕ := 54

/-- The number of pieces of gum Aubrey has -/
def aubrey_gum : ℕ := 0

/-- The number of pieces each person gets after sharing -/
def shared_gum : ℕ := 33

/-- Cole's initial number of pieces of gum -/
def cole_gum : ℕ := num_people * shared_gum - john_gum - aubrey_gum

theorem cole_gum_count : cole_gum = 45 := by
  sorry

end NUMINAMATH_CALUDE_cole_gum_count_l3545_354532


namespace NUMINAMATH_CALUDE_sum_of_coefficients_of_fifth_power_l3545_354534

theorem sum_of_coefficients_of_fifth_power (a b : ℕ) (h : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2) : a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_of_fifth_power_l3545_354534


namespace NUMINAMATH_CALUDE_age_difference_proof_l3545_354530

theorem age_difference_proof (jack_age bill_age : ℕ) : 
  jack_age = 3 * bill_age →
  (jack_age + 3) = 2 * (bill_age + 3) →
  jack_age - bill_age = 6 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3545_354530


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l3545_354508

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem stating that the profit-maximizing price is 35 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l3545_354508


namespace NUMINAMATH_CALUDE_complex_magnitude_three_fifths_minus_four_sevenths_i_l3545_354524

theorem complex_magnitude_three_fifths_minus_four_sevenths_i :
  Complex.abs (3/5 - (4/7)*Complex.I) = 29/35 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_three_fifths_minus_four_sevenths_i_l3545_354524


namespace NUMINAMATH_CALUDE_unique_representation_l3545_354579

theorem unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_representation_l3545_354579


namespace NUMINAMATH_CALUDE_anya_hair_growth_l3545_354567

/-- The number of hairs Anya washes down the drain -/
def hairs_washed : ℕ := 32

/-- The number of hairs Anya brushes out -/
def hairs_brushed : ℕ := hairs_washed / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := 49

/-- The total number of additional hairs Anya wants to have -/
def additional_hairs : ℕ := hairs_washed + hairs_brushed + hairs_to_grow

theorem anya_hair_growth :
  additional_hairs = 97 := by sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l3545_354567


namespace NUMINAMATH_CALUDE_sum_equals_point_nine_six_repeating_l3545_354573

/-- Represents a repeating decimal where the digit 8 repeats infinitely -/
def repeating_eight : ℚ := 8/9

/-- Represents the decimal 0.07 -/
def seven_hundredths : ℚ := 7/100

/-- Theorem stating that the sum of 0.8̇ and 0.07 is equal to 0.96̇ -/
theorem sum_equals_point_nine_six_repeating :
  repeating_eight + seven_hundredths = 29/30 := by sorry

end NUMINAMATH_CALUDE_sum_equals_point_nine_six_repeating_l3545_354573


namespace NUMINAMATH_CALUDE_conditional_statement_b_is_content_when_met_l3545_354569

/-- Represents the structure of a conditional statement -/
structure ConditionalStatement where
  condition : Prop
  contentWhenMet : Prop
  contentWhenNotMet : Prop

/-- Theorem stating that B in a conditional statement represents the content executed when the condition is met -/
theorem conditional_statement_b_is_content_when_met (stmt : ConditionalStatement) :
  stmt.contentWhenMet = stmt.contentWhenMet := by sorry

end NUMINAMATH_CALUDE_conditional_statement_b_is_content_when_met_l3545_354569


namespace NUMINAMATH_CALUDE_shaded_area_equals_1150_l3545_354568

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The main theorem stating the area of the shaded region -/
theorem shaded_area_equals_1150 :
  let square_side : ℝ := 40
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨20, 0⟩
  let p3 : Point := ⟨40, 30⟩
  let p4 : Point := ⟨40, 40⟩
  let p5 : Point := ⟨10, 40⟩
  let p6 : Point := ⟨0, 10⟩
  let square_area := square_side * square_side
  let triangle1_area := triangleArea p2 ⟨40, 0⟩ p3
  let triangle2_area := triangleArea p6 ⟨0, 40⟩ p5
  square_area - (triangle1_area + triangle2_area) = 1150 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_equals_1150_l3545_354568


namespace NUMINAMATH_CALUDE_gcd_problem_l3545_354587

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 431) :
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3545_354587


namespace NUMINAMATH_CALUDE_trolleybus_problem_l3545_354501

/-- Trolleybus Problem -/
theorem trolleybus_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (∀ z : ℝ, z > 0 → y * z = 6 * (y - x) ∧ y * z = 3 * (y + x)) →
  (∃ z : ℝ, z = 4 ∧ x = y / 3) :=
by sorry

end NUMINAMATH_CALUDE_trolleybus_problem_l3545_354501


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_perimeter_l3545_354536

theorem isosceles_trapezoid_perimeter 
  (base1 : ℝ) (base2 : ℝ) (altitude : ℝ)
  (h1 : base1 = Real.log 3)
  (h2 : base2 = Real.log 192)
  (h3 : altitude = Real.log 16)
  (h4 : ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 
    perimeter = Real.log (2^p * 3^q)) :
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 
    perimeter = Real.log (2^p * 3^q) ∧ p + q = 18 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_perimeter_l3545_354536


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l3545_354516

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, Real.sqrt x > Real.sqrt y → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(Real.sqrt x > Real.sqrt y)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l3545_354516


namespace NUMINAMATH_CALUDE_fraction_simplification_l3545_354538

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3545_354538


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3545_354522

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the line passing through A(0,-2) and B(t,0)
def line (t x y : ℝ) : Prop := y = (2/t)*x - 2

-- Define the condition for no intersection
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → ¬(line t x y)

-- Theorem statement
theorem parabola_line_intersection (t : ℝ) :
  no_intersection t ↔ t < -1 ∨ t > 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3545_354522


namespace NUMINAMATH_CALUDE_max_abs_value_l3545_354592

theorem max_abs_value (x y : ℝ) 
  (h1 : x + y - 2 ≤ 0) 
  (h2 : x - y + 4 ≥ 0) 
  (h3 : y ≥ 0) : 
  ∃ (z : ℝ), z = |x - 2*y + 2| ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = |x - 2*y + 2| → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_abs_value_l3545_354592


namespace NUMINAMATH_CALUDE_expression_evaluation_l3545_354555

theorem expression_evaluation : 
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3545_354555


namespace NUMINAMATH_CALUDE_wilson_gained_money_l3545_354502

def watch_problem (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) : Prop :=
  let cost_price1 := selling_price / (1 + profit_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_revenue := 2 * selling_price
  total_revenue > total_cost

theorem wilson_gained_money : watch_problem 150 25 15 := by
  sorry

end NUMINAMATH_CALUDE_wilson_gained_money_l3545_354502


namespace NUMINAMATH_CALUDE_shaded_area_is_fifty_l3545_354504

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length and partitioning points -/
structure PartitionedSquare where
  sideLength : ℝ
  pointA : Point
  pointB : Point

/-- Calculates the area of the shaded diamond region in the partitioned square -/
def shadedAreaInPartitionedSquare (square : PartitionedSquare) : ℝ :=
  sorry

/-- The theorem stating that the shaded area in the given partitioned square is 50 square cm -/
theorem shaded_area_is_fifty (square : PartitionedSquare) 
  (h1 : square.sideLength = 10)
  (h2 : square.pointA = ⟨10/3, 10⟩)
  (h3 : square.pointB = ⟨20/3, 0⟩) : 
  shadedAreaInPartitionedSquare square = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_fifty_l3545_354504


namespace NUMINAMATH_CALUDE_quadratic_equation_from_condition_l3545_354523

theorem quadratic_equation_from_condition (a b : ℝ) :
  a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0 →
  ∃ (x : ℝ → ℝ), (x a = 0 ∧ x b = 0) ∧ (∀ y, x y = y^2 - 3*y + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_condition_l3545_354523


namespace NUMINAMATH_CALUDE_sum_of_qp_at_points_l3545_354576

def p (x : ℝ) : ℝ := |x^2 - 4|

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_at_points :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_at_points_l3545_354576


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l3545_354542

def sequence_property (s : Fin 10 → ℕ) : Prop :=
  ∀ n : Fin 10, n.val ≥ 2 → s n = s (n - 1) * s (n - 2)

theorem first_number_in_sequence 
  (s : Fin 10 → ℕ) 
  (h_property : sequence_property s)
  (h_last_three : s 7 = 81 ∧ s 8 = 6561 ∧ s 9 = 43046721) :
  s 0 = 3486784401 :=
sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l3545_354542


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l3545_354539

theorem roots_sum_reciprocals (p q : ℝ) (x₁ x₂ : ℝ) (hx₁ : x₁^2 + p*x₁ + q = 0) (hx₂ : x₂^2 + p*x₂ + q = 0) (hq : q ≠ 0) :
  x₁/x₂ + x₂/x₁ = (p^2 - 2*q) / q :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l3545_354539


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3545_354583

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 2x + 1 is 28 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ),
    (x₁^2 + 2*x₁ + 1 = 7) ∧ 
    (x₂^2 + 2*x₂ + 1 = 7) ∧ 
    ((x₂ - x₁)^2 = 28) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3545_354583


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l3545_354526

theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_depth : ℝ := 10
  let hole1_diameter : ℝ := 5
  let hole2_depth : ℝ := 12
  let hole2_diameter : ℝ := 4
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let hole1_volume := π * (hole1_diameter / 2)^2 * hole1_depth
  let hole2_volume := π * (hole2_diameter / 2)^2 * hole2_depth
  sphere_volume - hole1_volume - hole2_volume = 10556.17 * π :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l3545_354526


namespace NUMINAMATH_CALUDE_discount_order_difference_l3545_354535

/-- Calculates the final price after applying discounts and tax -/
def final_price (initial_price : ℚ) (flat_discount : ℚ) (percent_discount : ℚ) (tax_rate : ℚ) (flat_first : Bool) : ℚ :=
  let price_after_flat := initial_price - flat_discount
  let price_after_percent := initial_price * (1 - percent_discount)
  let discounted_price := if flat_first then
    price_after_flat * (1 - percent_discount)
  else
    price_after_percent - flat_discount
  discounted_price * (1 + tax_rate)

/-- The difference in final price between two discount application orders -/
def price_difference (initial_price flat_discount percent_discount tax_rate : ℚ) : ℚ :=
  (final_price initial_price flat_discount percent_discount tax_rate true) -
  (final_price initial_price flat_discount percent_discount tax_rate false)

theorem discount_order_difference :
  price_difference 30 5 (25/100) (10/100) = 1375/1000 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l3545_354535


namespace NUMINAMATH_CALUDE_log_cutting_problem_l3545_354598

/-- Represents the number of cuts needed to divide logs into 1-meter pieces -/
def num_cuts (x y : ℕ) : ℕ := 2 * x + 3 * y

theorem log_cutting_problem :
  ∃ (x y : ℕ),
    x + y = 30 ∧
    3 * x + 4 * y = 100 ∧
    num_cuts x y = 70 :=
by sorry

end NUMINAMATH_CALUDE_log_cutting_problem_l3545_354598


namespace NUMINAMATH_CALUDE_f_sum_positive_l3545_354565

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l3545_354565


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_l3545_354563

theorem units_digit_of_large_power (n : ℕ) : n % 10 = (7^(3^(5^2))) % 10 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_l3545_354563


namespace NUMINAMATH_CALUDE_divides_prime_expression_l3545_354588

theorem divides_prime_expression (p : Nat) (h1 : p.Prime) (h2 : p > 3) :
  (42 * p) ∣ (3^p - 2^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_prime_expression_l3545_354588


namespace NUMINAMATH_CALUDE_complex_modulus_l3545_354506

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3545_354506


namespace NUMINAMATH_CALUDE_fifth_figure_perimeter_l3545_354574

/-- Represents the outer perimeter of a figure in the sequence -/
def outer_perimeter (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

/-- The outer perimeter of the fifth figure in the sequence is 20 -/
theorem fifth_figure_perimeter :
  outer_perimeter 5 = 20 := by
  sorry

#check fifth_figure_perimeter

end NUMINAMATH_CALUDE_fifth_figure_perimeter_l3545_354574


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l3545_354572

-- Define the ellipse G
def ellipse_G (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 6 / 3

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = 2 * Real.sqrt 2 ∧ y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m : ℝ), y = x + m

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the isosceles triangle
def isosceles_triangle (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), P = (-3, 2) ∧
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

-- Theorem statement
theorem ellipse_and_line_equations :
  ∀ (A B : ℝ × ℝ) (e : ℝ),
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  eccentricity e ∧
  right_focus (2 * Real.sqrt 2) 0 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  intersection_points A B ∧
  isosceles_triangle A B →
  (∀ (x y : ℝ), ellipse_G x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∀ (x y : ℝ), line_l x y ↔ x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equations_l3545_354572


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_2_range_of_m_l3545_354525

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

-- Part I
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x > 7 - |x - 1|} = {x : ℝ | x < -4 ∨ x > 5} := by sorry

-- Part II
theorem range_of_m :
  {m : ℝ | ∃ x : ℝ, f m x > 7 + |x - 1|} = {m : ℝ | m < -6 ∨ m > 8} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_2_range_of_m_l3545_354525


namespace NUMINAMATH_CALUDE_negative_three_halves_less_than_negative_one_l3545_354529

theorem negative_three_halves_less_than_negative_one :
  -((3 : ℚ) / 2) < -1 := by sorry

end NUMINAMATH_CALUDE_negative_three_halves_less_than_negative_one_l3545_354529


namespace NUMINAMATH_CALUDE_soap_cost_two_years_l3545_354541

-- Define the cost of one bar of soap
def cost_per_bar : ℕ := 4

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Define the number of years
def years : ℕ := 2

-- Define the function to calculate total cost
def total_cost (cost_per_bar months_per_year years : ℕ) : ℕ :=
  cost_per_bar * months_per_year * years

-- Theorem statement
theorem soap_cost_two_years :
  total_cost cost_per_bar months_per_year years = 96 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_two_years_l3545_354541


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_33_l3545_354589

-- Define a flippy number
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
    (n = a * 10000 + b * 1000 + a * 100 + b * 10 + a ∨
     n = b * 10000 + a * 1000 + b * 100 + a * 10 + b)

-- Define a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

-- Theorem statement
theorem no_five_digit_flippy_divisible_by_33 :
  ¬ ∃ (n : ℕ), is_five_digit n ∧ is_flippy n ∧ n % 33 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_33_l3545_354589


namespace NUMINAMATH_CALUDE_range_of_m_l3545_354514

theorem range_of_m (P S : Set ℝ) (m : ℝ) : 
  P = {x : ℝ | x^2 - 8*x - 20 ≤ 0} →
  S = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} →
  S.Nonempty →
  (∀ x, x ∉ P → x ∉ S) →
  (∃ x, x ∉ P ∧ x ∈ S) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ S', S' = {x : ℝ | 1 - k ≤ x ∧ x ≤ 1 + k} ∧
    S'.Nonempty ∧
    (∀ x, x ∉ P → x ∉ S') ∧
    (∃ x, x ∉ P ∧ x ∈ S') :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3545_354514


namespace NUMINAMATH_CALUDE_september_births_percentage_l3545_354553

theorem september_births_percentage 
  (total_people : ℕ) 
  (september_births : ℕ) 
  (h1 : total_people = 120) 
  (h2 : september_births = 12) : 
  (september_births : ℚ) / total_people * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_september_births_percentage_l3545_354553


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3545_354560

theorem rationalize_denominator :
  let x : ℝ := Real.rpow 3 (1/3)
  (1 / (x + Real.rpow 27 (1/3) - Real.rpow 9 (1/3))) = (x^2 + 3*x + 3) / (3 * 21) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3545_354560


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l3545_354547

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_multiplication_division_equality : 
  (binary_to_nat "1100101" * binary_to_nat "101101" * binary_to_nat "110") / 
  binary_to_nat "100" = binary_to_nat "1111101011011011000" := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l3545_354547


namespace NUMINAMATH_CALUDE_marathon_remainder_l3545_354518

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon_length : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 5

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := number_of_marathons * marathon_length.miles + (number_of_marathons * marathon_length.yards) / yards_per_mile,
      yards := (number_of_marathons * marathon_length.yards) % yards_per_mile }) 
  (h_range : y < yards_per_mile) : 
  y = 165 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_l3545_354518


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l3545_354585

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - m - 1 = 0 → m^2 - m = 1 := by
sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l3545_354585


namespace NUMINAMATH_CALUDE_transaction_period_is_one_year_l3545_354500

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℝ
  borrow_rate : ℝ
  lend_rate : ℝ
  gain_per_year : ℝ

/-- Calculates the number of years for the transaction -/
def transaction_years (t : Transaction) : ℝ :=
  1

/-- Theorem stating that the transaction period is 1 year -/
theorem transaction_period_is_one_year (t : Transaction) 
  (h1 : t.principal = 5000)
  (h2 : t.borrow_rate = 0.04)
  (h3 : t.lend_rate = 0.08)
  (h4 : t.gain_per_year = 200) :
  transaction_years t = 1 := by
  sorry

end NUMINAMATH_CALUDE_transaction_period_is_one_year_l3545_354500


namespace NUMINAMATH_CALUDE_factorization_problems_l3545_354510

theorem factorization_problems :
  (∀ x : ℝ, 4*x^2 - 16 = 4*(x+2)*(x-2)) ∧
  (∀ x y : ℝ, 2*x^3 - 12*x^2*y + 18*x*y^2 = 2*x*(x-3*y)^2) := by
sorry

end NUMINAMATH_CALUDE_factorization_problems_l3545_354510


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l3545_354597

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 7) :
  (Nat.choose n k) * Nat.factorial (k - 1) = 25920 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l3545_354597


namespace NUMINAMATH_CALUDE_beth_sells_80_coins_l3545_354515

/-- Calculates the number of coins Beth sells given her initial coins and a gift -/
def coins_sold (initial : ℕ) (gift : ℕ) : ℕ :=
  (initial + gift) / 2

/-- Proves that Beth sells 80 coins given her initial 125 coins and Carl's gift of 35 coins -/
theorem beth_sells_80_coins : coins_sold 125 35 = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_sells_80_coins_l3545_354515


namespace NUMINAMATH_CALUDE_arrangements_equal_42_l3545_354552

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people returning after training -/
def num_returning : ℕ := 2

/-- The maximum number of people that can be accommodated in each department -/
def max_per_department : ℕ := 1

/-- A function that calculates the number of different arrangements -/
def num_arrangements (n d r m : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_equal_42 : 
  num_arrangements num_departments num_departments num_returning max_per_department = 42 :=
sorry

end NUMINAMATH_CALUDE_arrangements_equal_42_l3545_354552


namespace NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3545_354586

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  ∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2) + ∫ (x : ℝ) in (1)..(2), 1/x = π/4 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3545_354586


namespace NUMINAMATH_CALUDE_least_integer_with_8_factors_l3545_354557

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- The property of being the least positive integer with exactly 8 factors -/
def is_least_with_8_factors (n : ℕ) : Prop :=
  count_factors n = 8 ∧ ∀ m : ℕ, m > 0 ∧ m < n → count_factors m ≠ 8

theorem least_integer_with_8_factors :
  is_least_with_8_factors 24 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_8_factors_l3545_354557


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3545_354575

theorem complex_fraction_simplification :
  (3 + 3 * Complex.I) / (-4 + 5 * Complex.I) = 3 / 41 - 27 / 41 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3545_354575


namespace NUMINAMATH_CALUDE_cubic_sum_l3545_354509

theorem cubic_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c →
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_l3545_354509


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l3545_354571

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l3545_354571


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_problem_l3545_354540

theorem dividend_divisor_quotient_remainder_problem 
  (y1 y2 z1 z2 r1 x1 x2 : ℤ)
  (hy1 : y1 = 2)
  (hy2 : y2 = 3)
  (hz1 : z1 = 3)
  (hz2 : z2 = 5)
  (hr1 : r1 = 1)
  (hx1 : x1 = 4)
  (hx2 : x2 = 6)
  (y : ℤ) (hy : y = 3*(y1 + y2) + 4)
  (z : ℤ) (hz : z = 2*z1^2 - z2)
  (r : ℤ) (hr : r = 3*r1 + 2)
  (x : ℤ) (hx : x = 2*x1*y1 - x2 + 10) :
  x = 20 ∧ y = 19 ∧ z = 13 ∧ r = 5 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_remainder_problem_l3545_354540


namespace NUMINAMATH_CALUDE_symmetric_point_l3545_354531

/-- Given a line l: x + y = 1 and two points P and Q, 
    this function checks if Q is symmetric to P with respect to l --/
def is_symmetric (P Q : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  (qy - py) / (qx - px) = -1 ∧ -- Perpendicular condition
  (px + qx) / 2 + (py + qy) / 2 = 1 -- Midpoint on the line condition

/-- Theorem stating that Q(-4, -1) is symmetric to P(2, 5) with respect to the line x + y = 1 --/
theorem symmetric_point : is_symmetric (2, 5) (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_l3545_354531


namespace NUMINAMATH_CALUDE_function_symmetry_l3545_354517

/-- Given a function f: ℝ → ℝ, if the graph of f(x-1) is symmetric to the curve y = e^x 
    with respect to the y-axis, then f(x) = e^(-x-1) -/
theorem function_symmetry (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = Real.exp (-x)) → 
  (∀ x : ℝ, f x = Real.exp (-x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3545_354517


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3545_354593

theorem complex_power_modulus : Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 1369 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3545_354593


namespace NUMINAMATH_CALUDE_tangent_line_value_l3545_354561

/-- The line x + y = c is tangent to the circle x^2 + y^2 = 8, where c is a positive real number. -/
def is_tangent_line (c : ℝ) : Prop :=
  c > 0 ∧ ∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c ∧
  ∀ (x' y' : ℝ), x' + y' = c → x'^2 + y'^2 ≥ 8

theorem tangent_line_value :
  ∀ c : ℝ, is_tangent_line c → c = 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_value_l3545_354561


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3545_354577

/-- Theorem: When the length of a rectangle is halved and its breadth is tripled, 
    the percentage change in area is a 50% increase. -/
theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let new_area := (L / 2) * (3 * B)
  let percent_change := (new_area - original_area) / original_area * 100
  percent_change = 50 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_change_l3545_354577


namespace NUMINAMATH_CALUDE_solve_equation_l3545_354511

theorem solve_equation (x y : ℚ) : 
  y = 2 / (4 * x + 2) → y = 1/2 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3545_354511


namespace NUMINAMATH_CALUDE_pie_piece_price_l3545_354503

/-- Represents the price of a single piece of pie -/
def price_per_piece : ℝ := 3.83

/-- Represents the number of pieces a single pie is divided into -/
def pieces_per_pie : ℕ := 3

/-- Represents the number of pies the bakery can make in one hour -/
def pies_per_hour : ℕ := 12

/-- Represents the cost to create one pie -/
def cost_per_pie : ℝ := 0.5

/-- Represents the total revenue from selling all pie pieces -/
def total_revenue : ℝ := 138

theorem pie_piece_price :
  price_per_piece * (pieces_per_pie * pies_per_hour) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_pie_piece_price_l3545_354503


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3545_354537

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} = 
  {(-1, 0), (-1, 1), (0, -1), (0, 2), (1, -1), (1, 2), (2, 0), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3545_354537


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l3545_354594

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = 1) : 
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l3545_354594


namespace NUMINAMATH_CALUDE_marlon_lollipops_l3545_354551

theorem marlon_lollipops (initial_lollipops : ℕ) (kept_lollipops : ℕ) (lou_lollipops : ℕ) :
  initial_lollipops = 42 →
  kept_lollipops = 4 →
  lou_lollipops = 10 →
  (initial_lollipops - kept_lollipops - lou_lollipops : ℚ) / initial_lollipops = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_marlon_lollipops_l3545_354551


namespace NUMINAMATH_CALUDE_absent_laborers_l3545_354544

theorem absent_laborers (W : ℝ) : 
  let L := 17.5
  let original_days := 6
  let actual_days := 10
  let absent := L * (1 - (original_days : ℝ) / (actual_days : ℝ))
  absent = 14 := by sorry

end NUMINAMATH_CALUDE_absent_laborers_l3545_354544


namespace NUMINAMATH_CALUDE_fixed_points_bisector_range_l3545_354548

noncomputable def f (a b x : ℝ) : ℝ := a * x + b + 1

theorem fixed_points_bisector_range (a b : ℝ) :
  (0 < a) → (a < 2) →
  (∃ x₀ : ℝ, f a b x₀ = x₀) →
  (∃ A B : ℝ × ℝ, 
    (f a b A.1 = A.2 ∧ f a b B.1 = B.2) ∧
    (∀ x y : ℝ, y = x + 1 / (2 * a^2 + 1) ↔ 
      ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
       2 * x = A.1 + B.1 ∧ 2 * y = A.2 + B.2))) →
  b ∈ Set.Icc (-Real.sqrt 2 / 4) 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_bisector_range_l3545_354548


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3545_354580

/-- A line passing through the point (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (-1, 2)
  let given_line (x y : ℝ) := 2 * x - 3 * y + 4 = 0
  let perpendicular_line (x y : ℝ) := 3 * x + 2 * y - 1 = 0
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (x = point.1 ∧ y = point.2 ∨ 
     ∃ t : ℝ, x = point.1 + 3 * t ∧ y = point.2 + 2 * t)) ∧
  (∀ x y : ℝ, given_line x y → 
    ∀ x' y' : ℝ, perpendicular_line x' y' → 
      (x - x') * 2 + (y - y') * 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l3545_354580


namespace NUMINAMATH_CALUDE_problem_solution_l3545_354590

theorem problem_solution : 
  let x := 0.47 * 1442 - 0.36 * 1412
  ∃ y, x + y = 3 ∧ y = -166.42 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3545_354590


namespace NUMINAMATH_CALUDE_average_listening_time_is_55_minutes_l3545_354528

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  quarter_listeners_percent : ℚ
  half_listeners_percent : ℚ
  three_quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the given audience distribution --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem average_listening_time_is_55_minutes 
  (dist : AudienceDistribution)
  (h1 : dist.total_audience = 200)
  (h2 : dist.lecture_duration = 90)
  (h3 : dist.full_listeners_percent = 30 / 100)
  (h4 : dist.sleepers_percent = 15 / 100)
  (h5 : dist.quarter_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h6 : dist.half_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h7 : dist.three_quarter_listeners_percent = 1 - dist.full_listeners_percent - dist.sleepers_percent - dist.quarter_listeners_percent - dist.half_listeners_percent)
  : average_listening_time dist = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_listening_time_is_55_minutes_l3545_354528


namespace NUMINAMATH_CALUDE_correct_propositions_are_123_l3545_354519

-- Define the type for propositions
inductive GeometricProposition
  | frustum_def
  | frustum_edges
  | cone_def
  | hemisphere_rotation

-- Define a function to check if a proposition is correct
def is_correct_proposition (p : GeometricProposition) : Prop :=
  match p with
  | GeometricProposition.frustum_def => True
  | GeometricProposition.frustum_edges => True
  | GeometricProposition.cone_def => True
  | GeometricProposition.hemisphere_rotation => False

-- Define the set of all propositions
def all_propositions : Set GeometricProposition :=
  {GeometricProposition.frustum_def, GeometricProposition.frustum_edges, 
   GeometricProposition.cone_def, GeometricProposition.hemisphere_rotation}

-- Define the set of correct propositions
def correct_propositions : Set GeometricProposition :=
  {p ∈ all_propositions | is_correct_proposition p}

-- Theorem to prove
theorem correct_propositions_are_123 :
  correct_propositions = {GeometricProposition.frustum_def, 
                          GeometricProposition.frustum_edges, 
                          GeometricProposition.cone_def} := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_are_123_l3545_354519


namespace NUMINAMATH_CALUDE_floor_abs_sum_equality_l3545_354582

theorem floor_abs_sum_equality : ⌊|(-7.3 : ℝ)|⌋ + |⌊(-7.3 : ℝ)⌋| = 15 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equality_l3545_354582


namespace NUMINAMATH_CALUDE_banana_cost_l3545_354591

/-- The cost of a bunch of bananas can be expressed as $5 minus the cost of a dozen apples -/
theorem banana_cost (apple_cost banana_cost : ℝ) : 
  apple_cost + banana_cost = 5 → banana_cost = 5 - apple_cost := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l3545_354591


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3545_354543

theorem isosceles_triangle_condition 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a = 2 * b * Real.cos C)
  (h6 : a > 0 ∧ b > 0 ∧ c > 0)
  : B = C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3545_354543


namespace NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l3545_354599

/-- Given a cylinder with diameter 6 cm and height 6 cm, if spheres of equal volume are made from the same material, the diameter of each sphere is equal to the cube root of (162 * π) cm. -/
theorem sphere_diameter_from_cylinder (π : ℝ) (h : π > 0) :
  let cylinder_diameter : ℝ := 6
  let cylinder_height : ℝ := 6
  let cylinder_volume : ℝ := π * (cylinder_diameter / 2)^2 * cylinder_height
  let sphere_volume : ℝ := cylinder_volume
  let sphere_diameter : ℝ := 2 * (3 * sphere_volume / (4 * π))^(1/3)
  sphere_diameter = (162 * π)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l3545_354599


namespace NUMINAMATH_CALUDE_debate_team_girls_l3545_354584

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (girls : ℕ) : 
  boys = 26 → 
  groups = 8 → 
  group_size = 9 → 
  groups * group_size = boys + girls → 
  girls = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l3545_354584


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3545_354578

/-- The number of ways to choose k elements from n elements -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets (who must be included in the starting lineup) -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

theorem basketball_team_selection :
  binomial (total_players - quadruplets) (starters - quadruplets) = 364 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3545_354578


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3545_354556

theorem negation_of_proposition :
  (¬ ∀ (x y : ℝ), xy = 0 → x = 0) ↔ (∃ (x y : ℝ), xy = 0 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3545_354556


namespace NUMINAMATH_CALUDE_egg_carton_problem_l3545_354581

theorem egg_carton_problem (abigail_eggs beatrice_eggs carson_eggs carton_size : ℕ) 
  (h1 : abigail_eggs = 48)
  (h2 : beatrice_eggs = 63)
  (h3 : carson_eggs = 27)
  (h4 : carton_size = 15) :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  (total_eggs % carton_size = 3) ∧ (total_eggs / carton_size = 9) := by
  sorry

end NUMINAMATH_CALUDE_egg_carton_problem_l3545_354581


namespace NUMINAMATH_CALUDE_ratio_problem_l3545_354533

theorem ratio_problem (x y z : ℝ) 
  (h : y / z = z / x ∧ z / x = x / y ∧ x / y = 1 / 2) : 
  (x / (y * z)) / (y / (z * x)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3545_354533


namespace NUMINAMATH_CALUDE_lunchroom_students_l3545_354513

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l3545_354513


namespace NUMINAMATH_CALUDE_broken_bamboo_equation_l3545_354595

theorem broken_bamboo_equation (x : ℝ) : 
  (0 ≤ x) ∧ (x ≤ 10) →
  x^2 + 3^2 = (10 - x)^2 :=
by sorry

/- Explanation of the Lean 4 statement:
   - We import Mathlib to access necessary mathematical definitions and theorems.
   - We define a theorem named 'broken_bamboo_equation'.
   - The theorem takes a real number 'x' as input, representing the height of the broken part.
   - The condition (0 ≤ x) ∧ (x ≤ 10) ensures that x is between 0 and 10 chi.
   - The equation x^2 + 3^2 = (10 - x)^2 represents the Pythagorean theorem applied to the scenario.
   - We use 'by sorry' to skip the proof, as requested.
-/

end NUMINAMATH_CALUDE_broken_bamboo_equation_l3545_354595


namespace NUMINAMATH_CALUDE_crypto_encoding_theorem_l3545_354566

/-- Represents the digits in the cryptographic encoding -/
inductive CryptoDigit
| V
| W
| X
| Y
| Z

/-- Represents a number in the cryptographic encoding -/
def CryptoNumber := List CryptoDigit

/-- Converts a CryptoNumber to its base 5 representation -/
def toBase5 : CryptoNumber → Nat := sorry

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 : Nat → Nat := sorry

/-- The theorem to be proved -/
theorem crypto_encoding_theorem 
  (encode : Nat → CryptoNumber) 
  (n : Nat) :
  encode n = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.Z] ∧
  encode (n + 1) = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.X] ∧
  encode (n + 2) = [CryptoDigit.V, CryptoDigit.V, CryptoDigit.W] →
  base5ToBase10 (toBase5 [CryptoDigit.X, CryptoDigit.Y, CryptoDigit.Z]) = 108 := by
  sorry

end NUMINAMATH_CALUDE_crypto_encoding_theorem_l3545_354566


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3545_354549

theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3545_354549


namespace NUMINAMATH_CALUDE_min_value_on_line_l3545_354545

theorem min_value_on_line (m n : ℝ) : 
  m + 2 * n = 1 → 2^m + 4^n ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l3545_354545


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3545_354564

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x² + (m-1)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x

theorem even_function_implies_m_equals_one (m : ℝ) :
  IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3545_354564
