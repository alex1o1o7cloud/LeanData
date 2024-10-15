import Mathlib

namespace NUMINAMATH_CALUDE_fraction_subtraction_l260_26053

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l260_26053


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l260_26002

/-- Given a rectangular plot with the following properties:
  - The length is 60 meters
  - The length is 20 meters more than the breadth
  - The total cost of fencing is Rs. 5300
  Prove that the cost of fencing per meter is Rs. 26.50 -/
theorem fencing_cost_per_meter
  (length : ℝ)
  (breadth : ℝ)
  (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : length = breadth + 20)
  (h3 : total_cost = 5300) :
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l260_26002


namespace NUMINAMATH_CALUDE_a_3_value_l260_26070

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

/-- Definition of a_n in terms of S_n -/
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_3_value : a 3 = 1 / 20 := by sorry

end NUMINAMATH_CALUDE_a_3_value_l260_26070


namespace NUMINAMATH_CALUDE_minimal_circle_and_intersecting_line_l260_26025

-- Define the right-angled triangle
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 / 4 + p.2 / 2 ≤ 1}

-- Define the circle equation
def circle_equation (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Define the line equation
def line_equation (slope : ℝ) (intercept : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = slope * point.1 + intercept

theorem minimal_circle_and_intersecting_line :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (intercept : ℝ),
    (∀ p ∈ triangle, circle_equation center radius p) ∧
    (∀ c, ∀ r, (∀ p ∈ triangle, circle_equation c r p) → r ≥ radius) ∧
    center = (2, 1) ∧
    radius^2 = 5 ∧
    (intercept = -1 - Real.sqrt 5 ∨ intercept = -1 + Real.sqrt 5) ∧
    (∃ A B : ℝ × ℝ,
      A ≠ B ∧
      circle_equation center radius A ∧
      circle_equation center radius B ∧
      line_equation 1 intercept A ∧
      line_equation 1 intercept B ∧
      ((A.1 - center.1) * (B.1 - center.1) + (A.2 - center.2) * (B.2 - center.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimal_circle_and_intersecting_line_l260_26025


namespace NUMINAMATH_CALUDE_riding_mower_rate_riding_mower_rate_is_two_l260_26075

theorem riding_mower_rate (total_area : ℝ) (riding_mower_fraction : ℝ) 
  (push_mower_rate : ℝ) (total_time : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : total_area = 8 := by sorry
  have h2 : riding_mower_fraction = 3/4 := by sorry
  have h3 : push_mower_rate = 1 := by sorry
  have h4 : total_time = 5 := by sorry

  -- Calculate the area mowed by each mower
  let riding_mower_area := total_area * riding_mower_fraction
  let push_mower_area := total_area * (1 - riding_mower_fraction)

  -- Calculate the time spent with the push mower
  let push_mower_time := push_mower_area / push_mower_rate

  -- Calculate the time spent with the riding mower
  let riding_mower_time := total_time - push_mower_time

  -- Calculate and return the riding mower rate
  exact riding_mower_area / riding_mower_time
  
-- The theorem statement proves that the riding mower rate is 2 acres per hour
theorem riding_mower_rate_is_two : 
  riding_mower_rate 8 (3/4) 1 5 = 2 := by sorry

end NUMINAMATH_CALUDE_riding_mower_rate_riding_mower_rate_is_two_l260_26075


namespace NUMINAMATH_CALUDE_a_equals_b_l260_26099

theorem a_equals_b (a b : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 6 → 
  b = 1 / (Real.sqrt 6 - Real.sqrt 5) → 
  a = b := by sorry

end NUMINAMATH_CALUDE_a_equals_b_l260_26099


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l260_26028

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_90_l260_26028


namespace NUMINAMATH_CALUDE_A_completes_in_15_days_l260_26085

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the rate at which A and B work
variable (A_rate B_rate : ℝ)

-- Define the time it takes for A and B to complete the work alone
variable (A_time B_time : ℝ)

-- Conditions from the problem
axiom B_time_18 : B_time = 18
axiom B_rate_def : B_rate = W / B_time
axiom work_split : A_rate * 5 + B_rate * 12 = W
axiom A_rate_def : A_rate = W / A_time

-- Theorem to prove
theorem A_completes_in_15_days : A_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_A_completes_in_15_days_l260_26085


namespace NUMINAMATH_CALUDE_a_5_value_l260_26027

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -1) (h_7 : a 7 = -9) : a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l260_26027


namespace NUMINAMATH_CALUDE_min_different_numbers_l260_26000

theorem min_different_numbers (total : ℕ) (max_freq : ℕ) (min_diff : ℕ) : 
  total = 2019 →
  max_freq = 10 →
  min_diff = 225 →
  (∀ k : ℕ, k < min_diff → k * (max_freq - 1) + max_freq < total) ∧
  (min_diff * (max_freq - 1) + max_freq ≥ total) := by
  sorry

end NUMINAMATH_CALUDE_min_different_numbers_l260_26000


namespace NUMINAMATH_CALUDE_sqrt_transformation_l260_26026

theorem sqrt_transformation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_transformation_l260_26026


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l260_26047

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  tip_amount = 9 →
  total_tips = 27 →
  total_customers - (total_tips / tip_amount) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l260_26047


namespace NUMINAMATH_CALUDE_trig_expression_equality_l260_26061

theorem trig_expression_equality : 
  (1 - Real.cos (10 * π / 180)^2) / 
  (Real.cos (800 * π / 180) * Real.sqrt (1 - Real.cos (20 * π / 180))) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l260_26061


namespace NUMINAMATH_CALUDE_square_side_length_l260_26097

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l260_26097


namespace NUMINAMATH_CALUDE_sum_of_remainders_mod_500_l260_26037

def remainders : Finset ℕ := Finset.image (fun n => (3^n) % 500) (Finset.range 101)

def T : ℕ := Finset.sum remainders id

theorem sum_of_remainders_mod_500 : T % 500 = (Finset.sum (Finset.range 101) (fun n => (3^n) % 500)) % 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_mod_500_l260_26037


namespace NUMINAMATH_CALUDE_parabola_directrix_l260_26077

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop := y = -1

/-- Theorem: The directrix equation of the parabola x^2 = 4y is y = -1 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_equation x y → directrix_equation y := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l260_26077


namespace NUMINAMATH_CALUDE_aliyah_vivienne_phone_difference_l260_26062

theorem aliyah_vivienne_phone_difference :
  ∀ (aliyah_phones : ℕ) (vivienne_phones : ℕ),
    vivienne_phones = 40 →
    (aliyah_phones + vivienne_phones) * 400 = 36000 →
    aliyah_phones - vivienne_phones = 10 := by
  sorry

end NUMINAMATH_CALUDE_aliyah_vivienne_phone_difference_l260_26062


namespace NUMINAMATH_CALUDE_correct_calculation_l260_26014

theorem correct_calculation (x y : ℝ) : -2 * x^2 * y - 3 * y * x^2 = -5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l260_26014


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l260_26046

/-- Given that Sandy took $310 for shopping and had $217 left after spending,
    prove that she spent 30% of her money. -/
theorem sandy_spending_percentage (money_taken : ℝ) (money_left : ℝ) : 
  money_taken = 310 → money_left = 217 → 
  (money_taken - money_left) / money_taken * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l260_26046


namespace NUMINAMATH_CALUDE_driver_speed_driver_speed_proof_l260_26096

/-- The actual average speed of a driver, given that increasing the speed by 12 miles per hour
would have reduced the travel time by 1/3. -/
theorem driver_speed : ℝ → Prop :=
  fun v : ℝ =>
    ∀ t d : ℝ,
      t > 0 → d > 0 →
      d = v * t →
      d = (v + 12) * (2/3 * t) →
      v = 24

-- The proof is omitted
theorem driver_speed_proof : driver_speed 24 := by sorry

end NUMINAMATH_CALUDE_driver_speed_driver_speed_proof_l260_26096


namespace NUMINAMATH_CALUDE_seven_minus_three_times_number_l260_26068

theorem seven_minus_three_times_number (n : ℝ) (c : ℝ) : 
  n = 3 → 7 * n = 3 * n + c → 7 * n - 3 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_three_times_number_l260_26068


namespace NUMINAMATH_CALUDE_share_ratio_l260_26049

theorem share_ratio (total money : ℕ) (a_share : ℕ) (x : ℚ) :
  total = 600 →
  a_share = 240 →
  a_share = x * (total - a_share) →
  (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))))) = total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))) →
  (a_share : ℚ) / (total - a_share) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l260_26049


namespace NUMINAMATH_CALUDE_solve_linear_equation_l260_26040

theorem solve_linear_equation :
  ∀ x : ℚ, -4 * x - 15 = 12 * x + 5 → x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l260_26040


namespace NUMINAMATH_CALUDE_sum_of_specific_T_values_l260_26036

def T (n : ℕ) : ℤ :=
  (-1 : ℤ) + 4 - 3 + 8 - 5 + ((-1)^n * (2*n : ℤ)) + ((-1)^(n+1) * (n : ℤ))

theorem sum_of_specific_T_values :
  T 27 + T 43 + T 60 = -84 ∨
  T 27 + T 43 + T 60 = -42 ∨
  T 27 + T 43 + T 60 = 0 ∨
  T 27 + T 43 + T 60 = 42 ∨
  T 27 + T 43 + T 60 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_values_l260_26036


namespace NUMINAMATH_CALUDE_two_vertical_asymptotes_l260_26020

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x + 2
def denominator (x : ℝ) : ℝ := x^2 + 8*x + 15

-- Define a function to check if a given x-value is a vertical asymptote
def is_vertical_asymptote (x : ℝ) : Prop :=
  denominator x = 0 ∧ numerator x ≠ 0

-- Theorem stating that there are exactly 2 vertical asymptotes
theorem two_vertical_asymptotes :
  ∃ (a b : ℝ), a ≠ b ∧
    is_vertical_asymptote a ∧
    is_vertical_asymptote b ∧
    ∀ (x : ℝ), is_vertical_asymptote x → (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_two_vertical_asymptotes_l260_26020


namespace NUMINAMATH_CALUDE_marks_pond_depth_l260_26021

/-- Given that Peter's pond is 5 feet deep and Mark's pond is 4 feet deeper than 3 times Peter's pond,
    prove that the depth of Mark's pond is 19 feet. -/
theorem marks_pond_depth (peters_depth : ℕ) (marks_depth : ℕ) 
  (h1 : peters_depth = 5)
  (h2 : marks_depth = 3 * peters_depth + 4) :
  marks_depth = 19 := by
  sorry

end NUMINAMATH_CALUDE_marks_pond_depth_l260_26021


namespace NUMINAMATH_CALUDE_inequality_properties_l260_26010

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : 
  (a + b < a * b) ∧ (a * b < b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l260_26010


namespace NUMINAMATH_CALUDE_candy_cost_l260_26056

theorem candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l260_26056


namespace NUMINAMATH_CALUDE_actual_weight_loss_percentage_l260_26089

-- Define the weight loss challenge scenario
def weight_loss_challenge (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ) : Prop :=
  let final_weight := W * (1 - actual_loss_percent / 100 + clothes_add_percent / 100)
  final_weight = W * (1 - measured_loss_percent / 100)

-- Theorem statement
theorem actual_weight_loss_percentage 
  (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ)
  (h1 : W > 0)
  (h2 : clothes_add_percent = 2)
  (h3 : measured_loss_percent = 8.2)
  (h4 : weight_loss_challenge W actual_loss_percent clothes_add_percent measured_loss_percent) :
  actual_loss_percent = 10.2 := by
sorry


end NUMINAMATH_CALUDE_actual_weight_loss_percentage_l260_26089


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l260_26079

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l260_26079


namespace NUMINAMATH_CALUDE_max_insurmountable_questions_max_insurmountable_questions_is_10_l260_26039

theorem max_insurmountable_questions :
  ∀ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 →
    3 * x₃ + 2 * x₂ + x₁ = 64 →
    x₂ = 2 * x₀ →
    x₀ ≤ 10 :=
by
  sorry

theorem max_insurmountable_questions_is_10 :
  ∃ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 ∧
    3 * x₃ + 2 * x₂ + x₁ = 64 ∧
    x₂ = 2 * x₀ ∧
    x₀ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_max_insurmountable_questions_max_insurmountable_questions_is_10_l260_26039


namespace NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l260_26008

theorem continuous_function_zero_on_interval 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l260_26008


namespace NUMINAMATH_CALUDE_geometric_problem_l260_26084

/-- Given a parabola and an ellipse with specific properties, prove the coordinates of intersection points, 
    the equation of a hyperbola, and the maximum area of a triangle. -/
theorem geometric_problem (a t : ℝ) (h_a_pos : a > 0) (h_a_range : a ∈ Set.Icc 1 2) (h_t : t > 4) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  let C := {(x, y) : ℝ × ℝ | x^2/(2*a^2) + y^2/a^2 = 1}
  let l := {(x, y) : ℝ × ℝ | y = x - a}
  let P := (4*a/3, a/3)
  let Q := ((3 - 2*Real.sqrt 2)*a, (2 - 2*Real.sqrt 2)*a)
  let A := (t, 0)
  let H := {(x, y) : ℝ × ℝ | 7*x^2 - 13*y^2 = 11*a^2}
  (P ∈ C ∧ P ∈ l) ∧
  (Q ∈ C₁ ∧ Q ∈ l) ∧
  (∃ Q' ∈ H, ∃ d : ℝ, d = 4*a ∧ (Q'.1 - Q.1)^2 + (Q'.2 - Q.2)^2 = d^2) ∧
  (∀ a' ∈ Set.Icc 1 2, 
    let S := abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2
    S ≤ (Real.sqrt 2 - 5/6)*(2*t - 4)) ∧
  (∃ S : ℝ, S = (Real.sqrt 2 - 5/6)*(2*t - 4) ∧
    S = abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2 ∧
    a = 2) :=
by sorry


end NUMINAMATH_CALUDE_geometric_problem_l260_26084


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l260_26003

/-- For a nonreal complex number z with |z| = 2, 
    the real part of 1/(2-z) is (2-x)/(8-4x+x^2), where x is the real part of z -/
theorem real_part_reciprocal (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) 
  (h3 : z.re = x) : 
  Complex.re (1 / (2 - z)) = (2 - x) / (8 - 4*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l260_26003


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l260_26044

/-- Given two points in polar coordinates, returns their midpoint. -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let A : ℝ × ℝ := (5, π/4)
  let B : ℝ × ℝ := (5, 3*π/4)
  let M : ℝ × ℝ := polar_midpoint A.1 A.2 B.1 B.2
  M.1 = 5*Real.sqrt 2/2 ∧ M.2 = 3*π/8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l260_26044


namespace NUMINAMATH_CALUDE_castle_extension_l260_26057

theorem castle_extension (a : ℝ) (ha : a > 0) :
  let original_perimeter := 4 * a
  let new_perimeter := 4 * a + 2 * (0.2 * a)
  let original_area := a ^ 2
  let new_area := a ^ 2 + (0.2 * a) ^ 2
  (new_perimeter = 1.1 * original_perimeter) →
  ((new_area - original_area) / original_area = 0.04) :=
by sorry

end NUMINAMATH_CALUDE_castle_extension_l260_26057


namespace NUMINAMATH_CALUDE_square_independence_of_p_l260_26013

theorem square_independence_of_p (m n p k : ℕ) : 
  m > 0 → n > 0 → p.Prime → p > m → 
  m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2 → 
  ∃ f : ℕ → ℕ, ∀ q : ℕ, q.Prime → q > m → 
    m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = (f q)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_independence_of_p_l260_26013


namespace NUMINAMATH_CALUDE_complex_sum_powers_l260_26042

theorem complex_sum_powers (z : ℂ) (h : z^2 - z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l260_26042


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l260_26034

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellowToRed : ℕ := 10
def redToBlue : ℕ := 10

/-- Number of yellow tickets needed to win a Bible -/
def yellowToWin : ℕ := 10

/-- Tom's current tickets -/
def tomCurrentTickets : TomTickets :=
  { yellow := 8, red := 3, blue := 7 }

/-- Calculate the total number of blue tickets Tom has -/
def totalBlueTickets (t : TomTickets) : ℕ :=
  t.yellow * yellowToRed * redToBlue + t.red * redToBlue + t.blue

/-- Calculate the number of blue tickets needed to win -/
def blueTicketsToWin : ℕ := yellowToWin * yellowToRed * redToBlue

/-- Theorem: Tom needs 163 more blue tickets to win a Bible -/
theorem tom_needs_163_blue_tickets :
  blueTicketsToWin - totalBlueTickets tomCurrentTickets = 163 := by
  sorry


end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l260_26034


namespace NUMINAMATH_CALUDE_weight_loss_problem_l260_26088

/-- Given four people who lost weight, prove that the last two people each lost 28 kg. -/
theorem weight_loss_problem (total_loss weight_loss1 weight_loss2 weight_loss3 weight_loss4 : ℕ) :
  total_loss = 103 →
  weight_loss1 = 27 →
  weight_loss2 = weight_loss1 - 7 →
  weight_loss3 = weight_loss4 →
  total_loss = weight_loss1 + weight_loss2 + weight_loss3 + weight_loss4 →
  weight_loss3 = 28 ∧ weight_loss4 = 28 := by
  sorry


end NUMINAMATH_CALUDE_weight_loss_problem_l260_26088


namespace NUMINAMATH_CALUDE_goldbach_refutation_l260_26086

theorem goldbach_refutation (n : ℕ) : 
  (∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) → 
  ¬(∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
by sorry

end NUMINAMATH_CALUDE_goldbach_refutation_l260_26086


namespace NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l260_26090

theorem function_always_positive_implies_x_range 
  (x : ℝ) 
  (h : ∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) : 
  x < 1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l260_26090


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l260_26033

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m+6)

/-- Theorem stating the condition for f to have both a maximum and a minimum -/
theorem f_has_max_and_min (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ m < -3 ∨ m > 6 := by
  sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l260_26033


namespace NUMINAMATH_CALUDE_selected_number_in_fourth_group_l260_26007

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Calculates the selected number for a given group in the systematic sampling -/
def selectedNumber (sampling : SystematicSampling) (groupIndex : Nat) : Nat :=
  sampling.startingNumber + (groupIndex - 1) * (sampling.totalStudents / sampling.sampleSize)

theorem selected_number_in_fourth_group (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 1200)
  (h2 : sampling.sampleSize = 80)
  (h3 : sampling.startingNumber = 6) :
  selectedNumber sampling 4 = 51 := by
  sorry

end NUMINAMATH_CALUDE_selected_number_in_fourth_group_l260_26007


namespace NUMINAMATH_CALUDE_investment_inconsistency_l260_26048

theorem investment_inconsistency :
  ¬ ∃ (r x y : ℝ), 
    x + y = 10000 ∧ 
    x > y ∧ 
    y > 0 ∧ 
    0.05 * y = 6000 ∧ 
    r * x = 0.05 * y + 160 ∧ 
    r > 0 := by
  sorry

end NUMINAMATH_CALUDE_investment_inconsistency_l260_26048


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l260_26073

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 5) :
  a / c = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l260_26073


namespace NUMINAMATH_CALUDE_min_sum_perpendicular_sides_right_triangle_l260_26005

theorem min_sum_perpendicular_sides_right_triangle (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) (h_area : a * b / 2 = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_perpendicular_sides_right_triangle_l260_26005


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l260_26072

theorem polynomial_coefficients 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (x - 1)^3 + (x + 1)^4 = x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) : 
  a₁ = 5 ∧ a₂ + a₃ + a₄ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l260_26072


namespace NUMINAMATH_CALUDE_min_value_a_l260_26069

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + 2*x*y ≤ a*(x^2 + y^2)) ↔ 
  a ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l260_26069


namespace NUMINAMATH_CALUDE_triangle_gp_common_ratio_bounds_l260_26054

/-- The common ratio of a geometric progression forming the sides of a triangle -/
def common_ratio_triangle_gp : Set ℝ :=
  {q : ℝ | (Real.sqrt 5 - 1) / 2 ≤ q ∧ q ≤ (Real.sqrt 5 + 1) / 2}

/-- Theorem: The common ratio of a geometric progression forming the sides of a triangle
    is bounded by (√5 - 1)/2 and (√5 + 1)/2 -/
theorem triangle_gp_common_ratio_bounds (a : ℝ) (q : ℝ) 
    (h_a : a > 0) (h_q : q ≥ 1) 
    (h_triangle : a + a*q > a*q^2 ∧ a + a*q^2 > a*q ∧ a*q + a*q^2 > a) :
  q ∈ common_ratio_triangle_gp := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_common_ratio_bounds_l260_26054


namespace NUMINAMATH_CALUDE_defective_units_shipped_l260_26018

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.1 →
  shipped_rate = 0.05 →
  (defective_rate * shipped_rate * 100) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l260_26018


namespace NUMINAMATH_CALUDE_larger_circle_radius_l260_26064

/-- A system of two circles with specific properties -/
structure CircleSystem where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  longest_chord : ℝ  -- Length of the longest chord in the larger circle

/-- Properties of the circle system -/
def circle_system_properties (cs : CircleSystem) : Prop :=
  cs.longest_chord = 24 ∧  -- The longest chord of the larger circle is 24
  cs.r = cs.R / 2 ∧  -- The radius of the smaller circle is half the radius of the larger circle
  cs.R > 0 ∧  -- The radius of the larger circle is positive
  cs.r > 0  -- The radius of the smaller circle is positive

/-- Theorem stating that the radius of the larger circle is 12 -/
theorem larger_circle_radius (cs : CircleSystem) 
  (h : circle_system_properties cs) : cs.R = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l260_26064


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l260_26030

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of times Terrell must lift the new weights to match the total weight -/
def new_lifts : ℕ := 25

theorem terrell_weight_lifting :
  num_weights * original_weight * original_lifts =
  num_weights * new_weight * new_lifts := by
  sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l260_26030


namespace NUMINAMATH_CALUDE_oil_tank_capacity_l260_26035

theorem oil_tank_capacity (C : ℝ) (h1 : C > 0) :
  (C / 6 : ℝ) / C = 1 / 6 ∧ (C / 6 + 4) / C = 1 / 3 → C = 24 := by
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_l260_26035


namespace NUMINAMATH_CALUDE_smallest_factor_correct_l260_26032

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_factor : ℕ := 1848

/-- Theorem stating that the smallest_factor is correct -/
theorem smallest_factor_correct :
  (∀ k : ℕ, k > 0 → 112 ∣ (k * 43 * 62 * 1311) → 33 ∣ (k * 43 * 62 * 1311) → k ≥ smallest_factor) ∧
  (112 ∣ (smallest_factor * 43 * 62 * 1311)) ∧
  (33 ∣ (smallest_factor * 43 * 62 * 1311)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_correct_l260_26032


namespace NUMINAMATH_CALUDE_youngest_age_proof_l260_26022

theorem youngest_age_proof (n : ℕ) (current_avg : ℝ) (past_avg : ℝ) :
  n = 7 ∧ current_avg = 30 ∧ past_avg = 27 →
  (n : ℝ) * current_avg - (n - 1 : ℝ) * past_avg = 48 := by
  sorry

end NUMINAMATH_CALUDE_youngest_age_proof_l260_26022


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l260_26087

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Parabola equation y^2 = x + 4 -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = p.x + 4

/-- Perpendicular lines have product of slopes equal to -1 -/
def Perpendicular (a b c : Point) : Prop :=
  (b.y - a.y) * (c.y - b.y) = -(b.x - a.x) * (c.x - b.x)

/-- The main theorem -/
theorem parabola_perpendicular_range :
  ∀ (b c : Point),
    OnParabola b → OnParabola c →
    Perpendicular ⟨0, 2⟩ b c →
    c.y ≤ 0 ∨ c.y ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l260_26087


namespace NUMINAMATH_CALUDE_derivative_tangent_line_existence_no_derivative_no_tangent_slope_l260_26078

-- Define a real-valued function f
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define the existence of a derivative at x₀
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - L * (x - x₀)| ≤ ε * |x - x₀|

-- Define the existence of a tangent line at x₀
def has_tangent_line_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m b, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - (m * x + b)| < ε * |x - x₀|

-- Define the existence of a slope of the tangent line at x₀
def has_tangent_slope_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - m * (x - x₀)| < ε * |x - x₀|

-- Theorem 1: Non-existence of derivative doesn't imply non-existence of tangent line
theorem derivative_tangent_line_existence (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → (has_tangent_line_at f x₀ ∨ ¬(has_tangent_line_at f x₀)) :=
sorry

-- Theorem 2: Non-existence of derivative implies non-existence of tangent slope
theorem no_derivative_no_tangent_slope (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → ¬(has_tangent_slope_at f x₀) :=
sorry

end NUMINAMATH_CALUDE_derivative_tangent_line_existence_no_derivative_no_tangent_slope_l260_26078


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l260_26017

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l260_26017


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l260_26093

theorem shaded_fraction_of_rectangle : ∀ (length width : ℕ) (shaded_fraction : ℚ),
  length = 15 →
  width = 20 →
  shaded_fraction = 1/4 →
  (shaded_fraction * (1/2 : ℚ)) * (length * width : ℚ) = (1/8 : ℚ) * (length * width : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l260_26093


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l260_26050

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Check if a point lies on a line segment -/
def point_on_segment (P A B : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (circle : Circle) (A B : Point) : Prop := sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem tangential_quadrilateral_theorem 
  (A B C D : Point) 
  (circle1 circle2 : Circle) :
  are_concyclic A B C D →
  point_on_segment circle2.center A B →
  circle_tangent_to_segment circle2 B C →
  circle_tangent_to_segment circle2 C D →
  circle_tangent_to_segment circle2 D A →
  distance A D + distance B C = distance A B := by
  sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l260_26050


namespace NUMINAMATH_CALUDE_polynomial_factorization_l260_26071

theorem polynomial_factorization (x y : ℝ) :
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8 =
  (x^2 - x*y + y^2) * (x^6 - x^3*y^3 + y^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l260_26071


namespace NUMINAMATH_CALUDE_find_T_l260_26001

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l260_26001


namespace NUMINAMATH_CALUDE_forces_equilibrium_l260_26045

/-- A 2D vector representing a force -/
structure Force where
  x : ℝ
  y : ℝ

/-- Add two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

instance : Add Force :=
  ⟨Force.add⟩

/-- The zero force -/
def Force.zero : Force :=
  ⟨0, 0⟩

instance : Zero Force :=
  ⟨Force.zero⟩

theorem forces_equilibrium (f₁ f₂ f₃ f₄ : Force) 
    (h₁ : f₁ = ⟨-2, -1⟩)
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    f₁ + f₂ + f₃ + f₄ = 0 := by
  sorry

end NUMINAMATH_CALUDE_forces_equilibrium_l260_26045


namespace NUMINAMATH_CALUDE_fraction_problem_l260_26029

theorem fraction_problem (x : ℚ) (h1 : x * 180 = 36) (h2 : x < 0.3) : x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l260_26029


namespace NUMINAMATH_CALUDE_expression_evaluation_l260_26067

theorem expression_evaluation (c d : ℤ) (hc : c = 2) (hd : d = 3) :
  (c^3 + d^2)^2 - (c^3 - d^2)^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l260_26067


namespace NUMINAMATH_CALUDE_range_of_a_l260_26009

-- Define the conditions p and q as functions
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x))  -- ¬p is necessary for ¬q
  (h3 : ∃ x, ¬(p a x) ∧ q x)     -- ¬p is not sufficient for ¬q
  : a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l260_26009


namespace NUMINAMATH_CALUDE_amy_work_hours_l260_26094

/-- Calculates the required weekly hours for a given total earnings, number of weeks, and hourly rate -/
def required_weekly_hours (total_earnings : ℚ) (num_weeks : ℚ) (hourly_rate : ℚ) : ℚ :=
  total_earnings / (num_weeks * hourly_rate)

/-- Represents Amy's work scenario -/
theorem amy_work_hours 
  (summer_weekly_hours : ℚ) 
  (summer_weeks : ℚ) 
  (summer_earnings : ℚ) 
  (school_weeks : ℚ) 
  (school_earnings : ℚ)
  (h1 : summer_weekly_hours = 45)
  (h2 : summer_weeks = 8)
  (h3 : summer_earnings = 3600)
  (h4 : school_weeks = 24)
  (h5 : school_earnings = 3600) :
  required_weekly_hours school_earnings school_weeks 
    (summer_earnings / (summer_weekly_hours * summer_weeks)) = 15 := by
  sorry

#eval required_weekly_hours 3600 24 (3600 / (45 * 8))

end NUMINAMATH_CALUDE_amy_work_hours_l260_26094


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l260_26038

open Real

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 1 2, x^2 - a > 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∃ a : ℝ, a < 2 ∧ ¬(P a)) ∧
  (∀ a : ℝ, P a → a < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l260_26038


namespace NUMINAMATH_CALUDE_overweight_condition_equiv_l260_26031

/-- Ideal weight formula -/
def ideal_weight (h : ℝ) : ℝ := 22 * h^2

/-- Overweight threshold -/
def overweight_threshold (h : ℝ) : ℝ := 1.1 * ideal_weight h

/-- Overweight condition -/
def is_overweight (W h : ℝ) : Prop := W > overweight_threshold h

/-- Quadratic overweight condition -/
def quadratic_overweight (c d e : ℝ) (W h : ℝ) : Prop := W > c * h^2 + d * h + e

theorem overweight_condition_equiv :
  ∃ c d e : ℝ, ∀ W h : ℝ, is_overweight W h ↔ quadratic_overweight c d e W h :=
sorry

end NUMINAMATH_CALUDE_overweight_condition_equiv_l260_26031


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_l260_26023

theorem sqrt_sum_equals_six : 
  Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_l260_26023


namespace NUMINAMATH_CALUDE_adjacent_rectangles_area_l260_26051

/-- The total area of two adjacent rectangles -/
theorem adjacent_rectangles_area 
  (u v w z : Real) 
  (hu : u > 0) 
  (hv : v > 0) 
  (hw : w > 0) 
  (hz : z > w) : 
  let first_rectangle := (u + v) * w
  let second_rectangle := (u + v) * (z - w)
  first_rectangle + second_rectangle = (u + v) * z :=
by sorry

end NUMINAMATH_CALUDE_adjacent_rectangles_area_l260_26051


namespace NUMINAMATH_CALUDE_hat_price_calculation_l260_26066

theorem hat_price_calculation (total_hats green_hats : ℕ) (blue_price green_price : ℚ) 
  (h1 : total_hats = 85)
  (h2 : green_hats = 38)
  (h3 : blue_price = 6)
  (h4 : green_price = 7) :
  let blue_hats := total_hats - green_hats
  (blue_hats * blue_price + green_hats * green_price : ℚ) = 548 := by
  sorry

end NUMINAMATH_CALUDE_hat_price_calculation_l260_26066


namespace NUMINAMATH_CALUDE_divisibility_condition_l260_26059

/-- Represents a four-digit number MCUD -/
structure FourDigitNumber where
  M : Nat
  C : Nat
  D : Nat
  U : Nat
  h_M : M < 10
  h_C : C < 10
  h_D : D < 10
  h_U : U < 10

/-- Calculates the value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.M + 100 * n.C + 10 * n.D + n.U

/-- Calculates the remainders r₁, r₂, and r₃ for a given divisor -/
def calculateRemainders (A : Nat) : Nat × Nat × Nat :=
  let r₁ := 10 % A
  let r₂ := (10 * r₁) % A
  let r₃ := (10 * r₂) % A
  (r₁, r₂, r₃)

/-- The main theorem stating the divisibility condition -/
theorem divisibility_condition (n : FourDigitNumber) (A : Nat) (hA : A > 0) :
  A ∣ n.value ↔ A ∣ (n.U + n.D * (calculateRemainders A).1 + n.C * (calculateRemainders A).2.1 + n.M * (calculateRemainders A).2.2) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l260_26059


namespace NUMINAMATH_CALUDE_complex_modulus_l260_26065

theorem complex_modulus (z : ℂ) (h : z * (3 - 4*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l260_26065


namespace NUMINAMATH_CALUDE_two_number_difference_l260_26083

theorem two_number_difference (x y : ℝ) : 
  x + y = 40 → 3 * y - 4 * x = 20 → |y - x| = 11.42 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l260_26083


namespace NUMINAMATH_CALUDE_superhero_movie_count_l260_26019

/-- The number of movies watched by Dalton -/
def dalton_movies : ℕ := 7

/-- The number of movies watched by Hunter -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched -/
def total_different_movies : ℕ := dalton_movies + hunter_movies + alex_movies - 2 * movies_watched_together

theorem superhero_movie_count : total_different_movies = 32 := by
  sorry

end NUMINAMATH_CALUDE_superhero_movie_count_l260_26019


namespace NUMINAMATH_CALUDE_saree_price_l260_26091

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 144 ∧ discount1 = 0.20 ∧ discount2 = 0.10 →
  ∃ original_price : ℝ, 
    original_price = 200 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_saree_price_l260_26091


namespace NUMINAMATH_CALUDE_magician_card_decks_l260_26058

/-- A problem about a magician selling magic card decks. -/
theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) (initial_decks : ℕ) : 
  price = 2 →
  decks_left = 3 →
  earnings = 4 →
  initial_decks = earnings / price + decks_left →
  initial_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_magician_card_decks_l260_26058


namespace NUMINAMATH_CALUDE_circumscribed_polygon_similarity_l260_26098

/-- A circumscribed n-gon (n > 3) divided by non-intersecting diagonals into triangles -/
structure CircumscribedPolygon (n : ℕ) :=
  (n_gt_three : n > 3)
  (divided_into_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Predicate to check if all triangles are similar to at least one other triangle -/
def all_triangles_similar (p : CircumscribedPolygon n) : Prop := sorry

/-- The set of possible n values for which the described situation is possible -/
def possible_n_values : Set ℕ := {n | n = 4 ∨ n > 5}

/-- Theorem stating the possible values of n for which the described situation is possible -/
theorem circumscribed_polygon_similarity (n : ℕ) (p : CircumscribedPolygon n) :
  all_triangles_similar p ↔ n ∈ possible_n_values :=
sorry

end NUMINAMATH_CALUDE_circumscribed_polygon_similarity_l260_26098


namespace NUMINAMATH_CALUDE_pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l260_26052

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in an aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem verifying the relationship between swordfish and pufferfish -/
theorem swordfish_pufferfish_relation : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_swordfish + num_pufferfish = total_fish := by sorry

end NUMINAMATH_CALUDE_pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l260_26052


namespace NUMINAMATH_CALUDE_fraction_inequality_l260_26080

theorem fraction_inequality (x : ℝ) : x / (x + 3) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-3) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l260_26080


namespace NUMINAMATH_CALUDE_range_of_m_l260_26041

def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 = 0}

theorem range_of_m :
  ∀ m : ℝ, (A m ∩ {1, 3} = A m) ↔ ((-2 * Real.sqrt 3 < m ∧ m < 2 * Real.sqrt 3) ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l260_26041


namespace NUMINAMATH_CALUDE_max_value_of_f_l260_26092

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

theorem max_value_of_f (a : ℝ) :
  (∀ x : ℝ, f a x = f a (π / 3 - x)) →
  (∃ m : ℝ, ∀ x : ℝ, f a x ≤ m ∧ ∃ x₀ : ℝ, f a x₀ = m) →
  (∃ x₀ : ℝ, f a x₀ = 1) ∧ (∀ x : ℝ, f a x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l260_26092


namespace NUMINAMATH_CALUDE_infinite_52_divisible_cells_l260_26012

/-- Represents a position in the grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- The value at a node given its position in the spiral -/
def spiral_value (p : Position) : ℕ := sorry

/-- The sum of values at the four corners of a cell -/
def cell_sum (p : Position) : ℕ :=
  spiral_value p + spiral_value ⟨p.x + 1, p.y⟩ + 
  spiral_value ⟨p.x + 1, p.y + 1⟩ + spiral_value ⟨p.x, p.y + 1⟩

/-- Predicate for whether a number is divisible by 52 -/
def divisible_by_52 (n : ℕ) : Prop := n % 52 = 0

/-- The main theorem to be proved -/
theorem infinite_52_divisible_cells :
  ∀ n : ℕ, ∃ p : Position, p.x ≥ n ∧ p.y ≥ n ∧ divisible_by_52 (cell_sum p) :=
sorry

end NUMINAMATH_CALUDE_infinite_52_divisible_cells_l260_26012


namespace NUMINAMATH_CALUDE_rachels_homework_difference_l260_26081

/-- Rachel's homework problem -/
theorem rachels_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 3) 
  (h2 : reading_pages = 4) : 
  reading_pages - math_pages = 1 := by
  sorry

end NUMINAMATH_CALUDE_rachels_homework_difference_l260_26081


namespace NUMINAMATH_CALUDE_josh_marbles_count_l260_26063

def final_marbles (initial found traded broken : ℕ) : ℕ :=
  initial + found - traded - broken

theorem josh_marbles_count : final_marbles 357 146 32 10 = 461 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l260_26063


namespace NUMINAMATH_CALUDE_difference_of_squares_l260_26076

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l260_26076


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l260_26055

/-- The axis of symmetry of the parabola y = 2x² is the line x = 0 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  ∀ x y : ℝ, f (x) = f (-x) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l260_26055


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l260_26095

/-- A linear function from ℝ to ℝ -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is decreasing if for any x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  IsDecreasing (LinearFunction m b) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l260_26095


namespace NUMINAMATH_CALUDE_power_equation_solution_l260_26004

theorem power_equation_solution : 2^4 - 7 = 3^3 + (-18) := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l260_26004


namespace NUMINAMATH_CALUDE_original_cost_was_75_l260_26015

/-- Represents the selling price of a key chain -/
def selling_price : ℝ := 100

/-- Represents the original profit percentage -/
def original_profit_percentage : ℝ := 0.25

/-- Represents the new profit percentage -/
def new_profit_percentage : ℝ := 0.50

/-- Represents the new manufacturing cost -/
def new_manufacturing_cost : ℝ := 50

/-- Calculates the original manufacturing cost based on the given conditions -/
def original_manufacturing_cost : ℝ := selling_price * (1 - original_profit_percentage)

/-- Theorem stating that the original manufacturing cost was $75 -/
theorem original_cost_was_75 : 
  original_manufacturing_cost = 75 := by sorry

end NUMINAMATH_CALUDE_original_cost_was_75_l260_26015


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l260_26082

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l260_26082


namespace NUMINAMATH_CALUDE_range_of_m_given_p_q_l260_26074

/-- The range of m given the conditions of p and q -/
theorem range_of_m_given_p_q :
  ∀ (m : ℝ),
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → (x - (1 - m)) * (x - (1 + m)) > 0) ∧
  (∃ x : ℝ, (x - (1 - m)) * (x - (1 + m)) > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  m > 0 →
  0 < m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_p_q_l260_26074


namespace NUMINAMATH_CALUDE_oliver_puzzle_cost_l260_26006

/-- The amount of money Oliver spent on the puzzle -/
def puzzle_cost (initial_amount savings frisbee_cost birthday_gift final_amount : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost + birthday_gift - final_amount

theorem oliver_puzzle_cost : 
  puzzle_cost 9 5 4 8 15 = 3 := by sorry

end NUMINAMATH_CALUDE_oliver_puzzle_cost_l260_26006


namespace NUMINAMATH_CALUDE_shifted_sine_equivalence_shift_amount_l260_26016

/-- Proves that the given function is equivalent to a shifted sine function -/
theorem shifted_sine_equivalence (x : ℝ) : 
  (1/2 : ℝ) * Real.sin (4*x) - (Real.sqrt 3 / 2) * Real.cos (4*x) = Real.sin (4*x - π/3) :=
by sorry

/-- Proves that the shift is π/12 units to the right -/
theorem shift_amount : 
  ∃ (k : ℝ), ∀ (x : ℝ), Real.sin (4*x - π/3) = Real.sin (4*(x - k)) ∧ k = π/12 :=
by sorry

end NUMINAMATH_CALUDE_shifted_sine_equivalence_shift_amount_l260_26016


namespace NUMINAMATH_CALUDE_complex_square_roots_l260_26060

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -91 - 49 * I ↔ z = (7 * Real.sqrt 2) / 2 - 7 * Real.sqrt 2 * I ∨ 
                         z = -(7 * Real.sqrt 2) / 2 + 7 * Real.sqrt 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l260_26060


namespace NUMINAMATH_CALUDE_brownie_problem_l260_26024

def initial_brownies : ℕ := 16

theorem brownie_problem (B : ℕ) (h1 : B = initial_brownies) :
  let remaining_after_children : ℚ := 3/4 * B
  let remaining_after_family : ℚ := 1/2 * remaining_after_children
  let final_remaining : ℚ := remaining_after_family - 1
  final_remaining = 5 := by sorry

end NUMINAMATH_CALUDE_brownie_problem_l260_26024


namespace NUMINAMATH_CALUDE_vector_sum_zero_implies_parallel_l260_26011

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_implies_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + b = 0 → parallel a b) ∧ ¬(parallel a b → a + b = 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_implies_parallel_l260_26011


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l260_26043

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 2 * a n - 2) : a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l260_26043
