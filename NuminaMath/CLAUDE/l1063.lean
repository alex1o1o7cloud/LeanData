import Mathlib

namespace product_of_roots_plus_two_l1063_106309

theorem product_of_roots_plus_two (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end product_of_roots_plus_two_l1063_106309


namespace square_sum_equals_eight_l1063_106365

theorem square_sum_equals_eight (a b : ℝ) 
  (h1 : (a + b)^2 = 11) 
  (h2 : (a - b)^2 = 5) : 
  a^2 + b^2 = 8 := by
sorry

end square_sum_equals_eight_l1063_106365


namespace card_sets_l1063_106339

def is_valid_card_set (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 9 ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· = 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· < 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· > 9)).length = 2)

theorem card_sets :
  ∀ a b c d : ℕ,
    is_valid_card_set a b c d ↔
      (a = 1 ∧ b = 2 ∧ c = 7 ∧ d = 8) ∨
      (a = 1 ∧ b = 3 ∧ c = 6 ∧ d = 8) ∨
      (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 8) ∨
      (a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 7) ∨
      (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 7) ∨
      (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) :=
by sorry

end card_sets_l1063_106339


namespace boys_camp_total_l1063_106349

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℝ) * 0.2 * 0.7 = 42) : total = 300 := by
  sorry

#check boys_camp_total

end boys_camp_total_l1063_106349


namespace alpha_values_l1063_106318

theorem alpha_values (α : Real) 
  (h1 : 0 < α ∧ α < 2 * Real.pi)
  (h2 : Real.sin α = Real.cos α)
  (h3 : (Real.sin α > 0 ∧ Real.cos α > 0) ∨ (Real.sin α < 0 ∧ Real.cos α < 0)) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 := by
sorry

end alpha_values_l1063_106318


namespace sqrt_sum_equality_l1063_106307

theorem sqrt_sum_equality (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end sqrt_sum_equality_l1063_106307


namespace company_average_service_l1063_106399

/-- Represents a department in the company -/
structure Department where
  employees : ℕ
  total_service : ℕ

/-- The company with two departments -/
structure Company where
  dept_a : Department
  dept_b : Department

/-- Average years of service for a department -/
def avg_service (d : Department) : ℚ :=
  d.total_service / d.employees

/-- Average years of service for the entire company -/
def company_avg_service (c : Company) : ℚ :=
  (c.dept_a.total_service + c.dept_b.total_service) / (c.dept_a.employees + c.dept_b.employees)

theorem company_average_service (k : ℕ) (h_k : k > 0) :
  let c : Company := {
    dept_a := { employees := 7 * k, total_service := 56 * k },
    dept_b := { employees := 5 * k, total_service := 30 * k }
  }
  avg_service c.dept_a = 8 ∧
  avg_service c.dept_b = 6 ∧
  company_avg_service c = 7 + 1/6 :=
by sorry

end company_average_service_l1063_106399


namespace rectangular_field_dimensions_l1063_106324

theorem rectangular_field_dimensions (m : ℕ) : 
  (3 * m + 10) * (m - 5) = 72 → m = 7 := by sorry

end rectangular_field_dimensions_l1063_106324


namespace abc_fraction_equals_twelve_l1063_106314

theorem abc_fraction_equals_twelve
  (a b c m : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hsum : a + b + c = m)
  (hsquare_sum : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 :=
by sorry

end abc_fraction_equals_twelve_l1063_106314


namespace quadratic_equation_problem_l1063_106317

theorem quadratic_equation_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m + 2 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ + x₂ + x₁ * x₂ = 2 →
  m = 3 := by
sorry

end quadratic_equation_problem_l1063_106317


namespace shaded_area_calculation_l1063_106397

/-- The area of a square with side length 12 cm, minus the area of four quarter circles 
    with radius 4 cm (one-third of the square's side length) drawn at each corner, 
    is equal to 144 - 16π cm². -/
theorem shaded_area_calculation (π : Real) : 
  let square_side : Real := 12
  let circle_radius : Real := square_side / 3
  let square_area : Real := square_side ^ 2
  let quarter_circles_area : Real := π * circle_radius ^ 2
  square_area - quarter_circles_area = 144 - 16 * π := by
  sorry

end shaded_area_calculation_l1063_106397


namespace combined_return_is_ten_percent_l1063_106367

/-- The combined yearly return percentage of two investments -/
def combined_return_percentage (investment1 investment2 return1 return2 : ℚ) : ℚ :=
  ((investment1 * return1 + investment2 * return2) / (investment1 + investment2)) * 100

/-- Theorem: The combined yearly return percentage of a $500 investment with 7% return
    and a $1500 investment with 11% return is 10% -/
theorem combined_return_is_ten_percent :
  combined_return_percentage 500 1500 (7/100) (11/100) = 10 := by
  sorry

end combined_return_is_ten_percent_l1063_106367


namespace rectangle_square_ratio_l1063_106393

/-- Configuration of rectangles around a square -/
structure RectangleSquareConfig where
  /-- Side length of the inner square -/
  inner_side : ℝ
  /-- Shorter side of each rectangle -/
  rect_short : ℝ
  /-- Longer side of each rectangle -/
  rect_long : ℝ

/-- Theorem: If the area of the outer square is 9 times that of the inner square,
    then the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_square_ratio (config : RectangleSquareConfig) 
    (h_area : (config.inner_side + 2 * config.rect_short)^2 = 9 * config.inner_side^2) :
    config.rect_long / config.rect_short = 2 := by
  sorry


end rectangle_square_ratio_l1063_106393


namespace original_number_proof_l1063_106358

theorem original_number_proof (x : ℝ) : 
  x * 16 = 3408 → 0.16 * 2.13 = 0.3408 → x = 213 := by
  sorry

end original_number_proof_l1063_106358


namespace intersection_of_A_and_B_l1063_106386

def A : Set ℝ := {x | Real.tan x > Real.sqrt 3}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : 
  A ∩ B = Set.Ioo (-2) (-Real.pi/2) ∪ Set.Ioo (Real.pi/3) (Real.pi/2) := by
  sorry

end intersection_of_A_and_B_l1063_106386


namespace sixteenth_row_seats_l1063_106313

/-- 
Represents the number of seats in a row of an auditorium where:
- The first row has 5 seats
- Each subsequent row increases by 2 seats
-/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 3

/-- 
Theorem: The 16th row of the auditorium has 35 seats
-/
theorem sixteenth_row_seats : seats_in_row 16 = 35 := by
  sorry

end sixteenth_row_seats_l1063_106313


namespace parabola_equation_l1063_106353

-- Define the parabola
def Parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 4 * p * x}

-- Define the focus of the parabola
def Focus (p : ℝ) : ℝ × ℝ := (p, 0)

-- Theorem statement
theorem parabola_equation (p : ℝ) (h : p = 2) :
  Parabola p = {(x, y) : ℝ × ℝ | y^2 = 8 * x} :=
sorry

end parabola_equation_l1063_106353


namespace negation_of_existence_negation_of_exponential_inequality_l1063_106306

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_exponential_inequality_l1063_106306


namespace deal_or_no_deal_probability_l1063_106331

/-- The total number of boxes in the game --/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $250,000 --/
def high_value_boxes : ℕ := 6

/-- The number of boxes to eliminate --/
def boxes_to_eliminate : ℕ := 8

/-- The probability of selecting a high-value box after elimination --/
def probability_high_value : ℚ := 1 / 3

theorem deal_or_no_deal_probability :
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) = probability_high_value :=
sorry

end deal_or_no_deal_probability_l1063_106331


namespace line_passes_through_P_and_parallel_to_tangent_l1063_106326

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 : ℝ) * M.1 - 4

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Theorem statement
theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  (∀ x y : ℝ, line_equation x y → (y - P.2) = m * (x - P.1)) :=
sorry

end line_passes_through_P_and_parallel_to_tangent_l1063_106326


namespace sum_of_squares_problem_l1063_106384

theorem sum_of_squares_problem (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 52)
  (h_sum_products : x*y + y*z + z*x = 24) :
  x + y + z = 10 := by
sorry

end sum_of_squares_problem_l1063_106384


namespace monkey_ladder_min_steps_l1063_106325

/-- The minimum number of steps for the monkey's ladder. -/
def min_steps : ℕ := 26

/-- Represents the possible movements of the monkey. -/
inductive Movement
| up : Movement
| down : Movement

/-- The number of steps the monkey moves in each direction. -/
def step_count (m : Movement) : ℤ :=
  match m with
  | Movement.up => 18
  | Movement.down => -10

/-- A sequence of movements that allows the monkey to reach the top and return to the ground. -/
def valid_sequence : List Movement := 
  [Movement.up, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down]

theorem monkey_ladder_min_steps :
  (∀ (seq : List Movement), 
    (seq.foldl (λ acc m => acc + step_count m) 0 = 0) →
    (seq.foldl (λ acc m => max acc (acc + step_count m)) 0 ≥ min_steps)) ∧
  (valid_sequence.foldl (λ acc m => acc + step_count m) 0 = 0) ∧
  (valid_sequence.foldl (λ acc m => max acc (acc + step_count m)) 0 = min_steps) := by
  sorry

#check monkey_ladder_min_steps

end monkey_ladder_min_steps_l1063_106325


namespace gcd_840_1764_l1063_106395

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1063_106395


namespace linear_function_y_axis_intersection_l1063_106347

/-- The coordinates of the intersection point of y = (1/2)x + 1 with the y-axis -/
theorem linear_function_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ (1/2) * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) := by
  sorry

end linear_function_y_axis_intersection_l1063_106347


namespace equation_solution_l1063_106336

theorem equation_solution :
  ∃! x : ℚ, (x^2 + 2*x + 2) / (x + 2) = x + 3 :=
by
  use (-4/3)
  sorry

end equation_solution_l1063_106336


namespace pyramid_volume_l1063_106340

/-- The volume of a pyramid with a right triangular base of side length 2 and height 2 is 4/3 -/
theorem pyramid_volume (s h : ℝ) (hs : s = 2) (hh : h = 2) :
  (1 / 3 : ℝ) * (1 / 2 * s * s) * h = 4 / 3 := by
  sorry

end pyramid_volume_l1063_106340


namespace rational_sum_of_three_cubes_l1063_106341

theorem rational_sum_of_three_cubes (t : ℚ) : 
  ∃ (x y z : ℚ), t = x^3 + y^3 + z^3 := by
  sorry

end rational_sum_of_three_cubes_l1063_106341


namespace lottery_prize_probability_l1063_106392

/-- The probability of getting a prize in a lottery with 10 prizes and 25 blanks -/
theorem lottery_prize_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  let probability : ℚ := num_prizes / total_outcomes
  probability = 2 / 7 := by
sorry

end lottery_prize_probability_l1063_106392


namespace product_sum_equality_l1063_106301

theorem product_sum_equality : 1520 * 1997 * 0.152 * 100 + 152^2 = 46161472 := by
  sorry

end product_sum_equality_l1063_106301


namespace inequality_proof_l1063_106354

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end inequality_proof_l1063_106354


namespace sine_of_pi_thirds_minus_two_theta_l1063_106391

theorem sine_of_pi_thirds_minus_two_theta (θ : ℝ) 
  (h : Real.tan (θ + π / 12) = 2) : 
  Real.sin (π / 3 - 2 * θ) = -3 / 5 := by
sorry

end sine_of_pi_thirds_minus_two_theta_l1063_106391


namespace line_perp_parallel_planes_l1063_106321

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane if they intersect at right angles -/
def line_perp_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel if they do not intersect -/
def planes_parallel (α β : Plane) : Prop := sorry

theorem line_perp_parallel_planes (α β : Plane) (m : Line) :
  different_planes α β →
  line_perp_plane m β →
  planes_parallel α β →
  line_perp_plane m α :=
sorry

end line_perp_parallel_planes_l1063_106321


namespace flower_cost_ratio_l1063_106343

/-- Given the conditions of Nadia's flower purchase, prove the ratio of lily cost to rose cost. -/
theorem flower_cost_ratio :
  ∀ (roses : ℕ) (lilies : ℚ) (rose_cost lily_cost total_cost : ℚ),
    roses = 20 →
    lilies = (3 / 4) * roses →
    rose_cost = 5 →
    total_cost = 250 →
    total_cost = roses * rose_cost + lilies * lily_cost →
    lily_cost / rose_cost = 2 := by
  sorry

end flower_cost_ratio_l1063_106343


namespace library_visitors_theorem_l1063_106310

/-- The average number of visitors on non-Sunday days in a library -/
def average_visitors_non_sunday (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) : ℚ :=
  let sundays := total_days / 7 + 1
  let other_days := total_days - sundays
  (total_days * month_avg - sundays * sunday_avg) / other_days

/-- Theorem stating the average number of visitors on non-Sunday days -/
theorem library_visitors_theorem :
  average_visitors_non_sunday 630 30 305 = 240 := by
  sorry

#eval average_visitors_non_sunday 630 30 305

end library_visitors_theorem_l1063_106310


namespace prime_power_sum_l1063_106355

theorem prime_power_sum (a b c d e : ℕ) :
  2^a * 3^b * 5^c * 7^d * 11^e = 27720 →
  2*a + 3*b + 5*c + 7*d + 11*e = 35 := by
  sorry

end prime_power_sum_l1063_106355


namespace square_plate_nails_l1063_106323

/-- The number of nails on each side of the square plate -/
def nails_per_side : ℕ := 25

/-- The number of sides of a square -/
def sides_of_square : ℕ := 4

/-- The number of corners in a square -/
def corners_of_square : ℕ := 4

/-- The total number of nails used to fix the square plate -/
def total_nails : ℕ := nails_per_side * sides_of_square - corners_of_square

theorem square_plate_nails :
  total_nails = 96 := by sorry

end square_plate_nails_l1063_106323


namespace intersection_range_l1063_106364

theorem intersection_range (k₁ k₂ t p q m n : ℝ) : 
  k₁ > 0 → k₂ > 0 → 
  k₁ * 1 = k₂ / 1 →
  t ≠ 0 → t ≠ -2 →
  p = k₁ * t →
  q = k₁ * (t + 2) →
  m = k₂ / t →
  n = k₂ / (t + 2) →
  (p - m) * (q - n) < 0 ↔ (-3 < t ∧ t < -2) ∨ (0 < t ∧ t < 1) :=
by sorry

end intersection_range_l1063_106364


namespace dodge_trucks_count_l1063_106304

theorem dodge_trucks_count (ford dodge toyota vw : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = 2 * toyota)
  (h3 : vw = toyota / 2)
  (h4 : vw = 5) : 
  dodge = 60 := by
  sorry

end dodge_trucks_count_l1063_106304


namespace sum_P_eq_4477547_l1063_106362

/-- P(n) is the product of all non-zero digits of the positive integer n -/
def P (n : ℕ+) : ℕ := sorry

/-- The sum of P(n) for n from 1 to 2009 -/
def sum_P : ℕ := (Finset.range 2009).sum (fun i => P ⟨i + 1, Nat.succ_pos i⟩)

/-- Theorem stating that the sum of P(n) for n from 1 to 2009 is 4477547 -/
theorem sum_P_eq_4477547 : sum_P = 4477547 := by sorry

end sum_P_eq_4477547_l1063_106362


namespace logans_score_l1063_106372

theorem logans_score (total_students : ℕ) (average_without_logan : ℚ) (average_with_logan : ℚ) :
  total_students = 20 →
  average_without_logan = 85 →
  average_with_logan = 86 →
  (total_students * average_with_logan - (total_students - 1) * average_without_logan : ℚ) = 105 :=
by sorry

end logans_score_l1063_106372


namespace parabola_rectangle_problem_l1063_106300

/-- The parabola equation -/
def parabola_equation (k x y : ℝ) : Prop := y = k^2 - x^2

/-- Rectangle ABCD properties -/
structure Rectangle (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  parallel_to_axes : Prop
  A_on_x_axis : A.2 = 0
  D_on_x_axis : D.2 = 0
  V_midpoint_BC : (B.1 + C.1) / 2 = 0 ∧ (B.2 + C.2) / 2 = k^2

/-- Perimeter of the rectangle -/
def perimeter (rect : Rectangle k) : ℝ :=
  2 * (|rect.A.1 - rect.B.1| + |rect.A.2 - rect.B.2|)

/-- Main theorem -/
theorem parabola_rectangle_problem (k : ℝ) 
  (h_pos : k > 0)
  (rect : Rectangle k)
  (h_perimeter : perimeter rect = 48) :
  k = 4 := by sorry

end parabola_rectangle_problem_l1063_106300


namespace consignment_total_items_l1063_106338

/-- Represents the price and quantity of items in a consignment shop. -/
structure ConsignmentItems where
  camera_price : ℕ
  clock_price : ℕ
  pen_price : ℕ
  receiver_price : ℕ
  camera_quantity : ℕ

/-- Conditions for the consignment shop problem -/
def ConsignmentConditions (items : ConsignmentItems) : Prop :=
  -- Total value of all items is 240 rubles
  (3 * items.camera_quantity * items.pen_price + 
   items.camera_quantity * items.clock_price + 
   items.camera_quantity * items.receiver_price + 
   items.camera_quantity * items.camera_price = 240) ∧
  -- Sum of receiver and clock prices is 4 rubles more than sum of camera and pen prices
  (items.receiver_price + items.clock_price = items.camera_price + items.pen_price + 4) ∧
  -- Sum of clock and pen prices is 24 rubles less than sum of camera and receiver prices
  (items.clock_price + items.pen_price + 24 = items.camera_price + items.receiver_price) ∧
  -- Pen price is an integer not exceeding 6 rubles
  (items.pen_price ≤ 6) ∧
  -- Number of cameras equals camera price divided by 10
  (items.camera_quantity = items.camera_price / 10) ∧
  -- Number of clocks equals number of receivers and number of cameras
  (items.camera_quantity = items.camera_quantity) ∧
  -- Number of pens is three times the number of cameras
  (3 * items.camera_quantity = 3 * items.camera_quantity)

/-- The theorem stating that under the given conditions, the total number of items is 18 -/
theorem consignment_total_items (items : ConsignmentItems) 
  (h : ConsignmentConditions items) : 
  (6 * items.camera_quantity = 18) := by
  sorry


end consignment_total_items_l1063_106338


namespace largest_prime_factor_l1063_106374

def numbers : List Nat := [85, 57, 119, 143, 169]

def has_largest_prime_factor (n : Nat) (ns : List Nat) : Prop :=
  ∀ m ∈ ns, ∀ p : Nat, p.Prime → p ∣ m → ∃ q : Nat, q.Prime ∧ q ∣ n ∧ q ≥ p

theorem largest_prime_factor :
  has_largest_prime_factor 57 numbers := by sorry

end largest_prime_factor_l1063_106374


namespace exam_failure_rate_l1063_106303

/-- Examination results -/
structure ExamResults where
  total_candidates : ℕ
  num_girls : ℕ
  boys_math_pass_rate : ℚ
  boys_science_pass_rate : ℚ
  boys_lang_pass_rate : ℚ
  girls_math_pass_rate : ℚ
  girls_science_pass_rate : ℚ
  girls_lang_pass_rate : ℚ

/-- Calculate the failure rate given exam results -/
def calculate_failure_rate (results : ExamResults) : ℚ :=
  let num_boys := results.total_candidates - results.num_girls
  let boys_passing := min (results.boys_math_pass_rate * num_boys)
                          (min (results.boys_science_pass_rate * num_boys)
                               (results.boys_lang_pass_rate * num_boys))
  let girls_passing := min (results.girls_math_pass_rate * results.num_girls)
                           (min (results.girls_science_pass_rate * results.num_girls)
                                (results.girls_lang_pass_rate * results.num_girls))
  let total_passing := boys_passing + girls_passing
  let total_failing := results.total_candidates - total_passing
  total_failing / results.total_candidates

/-- The main theorem about the examination failure rate -/
theorem exam_failure_rate :
  let results := ExamResults.mk 2500 1100 (42/100) (39/100) (36/100) (35/100) (32/100) (40/100)
  calculate_failure_rate results = 6576/10000 := by
  sorry


end exam_failure_rate_l1063_106303


namespace convention_center_distance_l1063_106322

/-- The distance from Elena's home to the convention center -/
def distance : ℝ := sorry

/-- Elena's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in Elena's speed for the rest of the journey -/
def speed_increase : ℝ := 20

/-- The time Elena would be late if she continued at the initial speed -/
def late_time : ℝ := 0.75

/-- The time Elena arrives early after increasing her speed -/
def early_time : ℝ := 0.25

/-- The actual time needed to arrive on time -/
def actual_time : ℝ := sorry

theorem convention_center_distance :
  (distance = initial_speed * (actual_time + late_time)) ∧
  (distance - initial_speed = (initial_speed + speed_increase) * (actual_time - 1 - early_time)) ∧
  (distance = 191.25) := by sorry

end convention_center_distance_l1063_106322


namespace prime_dividing_polynomial_congruence_l1063_106396

theorem prime_dividing_polynomial_congruence (n : ℕ) (p : ℕ) (hn : n > 0) (hp : Nat.Prime p) :
  p ∣ (5^(4*n) - 5^(3*n) + 5^(2*n) - 5^n + 1) → p % 4 = 1 := by
  sorry

end prime_dividing_polynomial_congruence_l1063_106396


namespace max_fraction_sum_l1063_106330

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The fraction (A+B)/(C+D) is an integer -/
def is_integer_fraction (a b c d : Digit) : Prop :=
  ∃ k : ℕ, k * (c.val + d.val) = a.val + b.val

/-- The fraction (A+B)/(C+D) is maximized -/
def is_maximized (a b c d : Digit) : Prop :=
  ∀ w x y z : Digit, distinct w x y z →
    is_integer_fraction w x y z →
    (a.val + b.val : ℚ) / (c.val + d.val) ≥ (w.val + x.val : ℚ) / (y.val + z.val)

theorem max_fraction_sum (a b c d : Digit) :
  distinct a b c d →
  is_integer_fraction a b c d →
  is_maximized a b c d →
  a.val + b.val = 17 :=
sorry

end max_fraction_sum_l1063_106330


namespace triangle_inequality_with_altitudes_l1063_106308

/-- Given a triangle with sides a > b and corresponding altitudes h_a and h_b,
    prove that a + h_a ≥ b + h_b with equality iff the angle between a and b is 90° -/
theorem triangle_inequality_with_altitudes (a b h_a h_b : ℝ) (S : ℝ) (γ : ℝ) :
  a > b →
  S = (1/2) * a * h_a →
  S = (1/2) * b * h_b →
  S = (1/2) * a * b * Real.sin γ →
  (a + h_a ≥ b + h_b) ∧ (a + h_a = b + h_b ↔ γ = Real.pi / 2) :=
by sorry

end triangle_inequality_with_altitudes_l1063_106308


namespace polynomial_value_l1063_106366

theorem polynomial_value (x y : ℝ) (h : x - y = 5) :
  (x - y)^2 + 2*(x - y) - 10 = 25 := by
  sorry

end polynomial_value_l1063_106366


namespace smallest_prime_perimeter_scalene_triangle_l1063_106356

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers are consecutive primes -/
def areConsecutivePrimes (a b c : ℕ) : Prop := sorry

/-- A function that checks if three side lengths can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop := sorry

/-- The smallest perimeter of a scalene triangle with consecutive prime side lengths and a prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    areConsecutivePrimes a b c ∧
    canFormTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      areConsecutivePrimes x y z ∧
      canFormTriangle x y z ∧
      isPrime (x + y + z) →
      (x + y + z ≥ 23)) :=
sorry

end smallest_prime_perimeter_scalene_triangle_l1063_106356


namespace complex_fraction_simplification_l1063_106387

theorem complex_fraction_simplification :
  ((1 - Complex.I) * (1 + 2 * Complex.I)) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end complex_fraction_simplification_l1063_106387


namespace max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l1063_106329

/-- Represents the profit function for desk lamp sales -/
def profit_function (x : ℝ) : ℝ :=
  (x - 30) * (600 - 10 * (x - 40))

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit_selling_price :
  ∃ (max_price max_profit : ℝ),
    max_price = 65 ∧
    max_profit = 12250 ∧
    ∀ (x : ℝ), profit_function x ≤ max_profit :=
by
  sorry

/-- Theorem stating that 15,000 yuan daily profit is not achievable -/
theorem daily_profit_unachievable :
  ∀ (x : ℝ), profit_function x < 15000 :=
by
  sorry

/-- Theorem stating the selling prices for 10,000 yuan monthly profit -/
theorem monthly_profit_prices :
  ∃ (price1 price2 : ℝ),
    price1 = 80 ∧
    price2 = 50 ∧
    profit_function price1 = 10000 ∧
    profit_function price2 = 10000 ∧
    ∀ (x : ℝ), profit_function x = 10000 → (x = price1 ∨ x = price2) :=
by
  sorry

end max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l1063_106329


namespace product_equality_l1063_106315

theorem product_equality : 1500 * 451 * 0.0451 * 25 = 7627537500 := by
  sorry

end product_equality_l1063_106315


namespace marathon_distance_yards_l1063_106369

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- The number of yards in a mile -/
def yardsPerMile : ℕ := 1760

/-- The distance of a single marathon -/
def marathonDistance : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- The number of marathons Leila has run -/
def marathonCount : ℕ := 8

/-- Calculates the total distance run in multiple marathons -/
def totalMarathonDistance (marathonDist : MarathonDistance) (count : ℕ) : TotalDistance :=
  { miles := marathonDist.miles * count,
    yards := marathonDist.yards * count }

/-- Converts a TotalDistance to a normalized form where yards < yardsPerMile -/
def normalizeDistance (dist : TotalDistance) : TotalDistance :=
  { miles := dist.miles + dist.yards / yardsPerMile,
    yards := dist.yards % yardsPerMile }

theorem marathon_distance_yards :
  (normalizeDistance (totalMarathonDistance marathonDistance marathonCount)).yards = 1400 := by
  sorry

end marathon_distance_yards_l1063_106369


namespace square_root_of_square_l1063_106394

theorem square_root_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end square_root_of_square_l1063_106394


namespace inequality_proof_l1063_106368

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ 
  a + b + c + (2 * a - b - c)^2 / (a + b + c) := by
  sorry

end inequality_proof_l1063_106368


namespace triangle_3_4_6_l1063_106398

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 6 can form a triangle -/
theorem triangle_3_4_6 : can_form_triangle 3 4 6 := by
  sorry

end triangle_3_4_6_l1063_106398


namespace cylindrical_bucket_height_l1063_106327

/-- The height of a cylindrical bucket given its radius and the dimensions of a conical heap formed when emptied -/
theorem cylindrical_bucket_height (r_cylinder r_cone h_cone : ℝ) (h_cylinder : ℝ) : 
  r_cylinder = 21 →
  r_cone = 63 →
  h_cone = 12 →
  r_cylinder^2 * h_cylinder = (1/3) * r_cone^2 * h_cone →
  h_cylinder = 36 := by
  sorry

end cylindrical_bucket_height_l1063_106327


namespace four_objects_two_groups_l1063_106388

theorem four_objects_two_groups : ∃ (n : ℕ), n = 14 ∧ 
  n = (Nat.choose 4 1) + (Nat.choose 4 2) + (Nat.choose 4 3) :=
sorry

end four_objects_two_groups_l1063_106388


namespace right_angled_triangle_exists_l1063_106348

/-- A color type with exactly three colors -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the cartesian grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each grid point -/
def Coloring := GridPoint → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightAngled (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0 ∨
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0 ∨
  (p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) = 0

/-- Main theorem: There always exists a right-angled triangle with vertices of different colors -/
theorem right_angled_triangle_exists (f : Coloring)
  (h1 : ∃ p : GridPoint, f p = Color.Red)
  (h2 : ∃ p : GridPoint, f p = Color.Green)
  (h3 : ∃ p : GridPoint, f p = Color.Blue) :
  ∃ p1 p2 p3 : GridPoint,
    isRightAngled p1 p2 p3 ∧
    f p1 ≠ f p2 ∧ f p2 ≠ f p3 ∧ f p1 ≠ f p3 :=
by
  sorry


end right_angled_triangle_exists_l1063_106348


namespace sum_of_roots_cubic_sum_of_roots_specific_cubic_l1063_106320

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = -b / a :=
sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 25 * x^3 - 50 * x^2 + 35 * x + 7
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = 2 :=
sorry

end sum_of_roots_cubic_sum_of_roots_specific_cubic_l1063_106320


namespace guitar_picks_problem_l1063_106383

theorem guitar_picks_problem (total : ℕ) (red blue yellow : ℕ) : 
  total > 0 ∧ 
  red = total / 2 ∧ 
  blue = total / 3 ∧ 
  yellow = 6 ∧ 
  red + blue + yellow = total → 
  blue = 12 := by
sorry

end guitar_picks_problem_l1063_106383


namespace train_speed_calculation_l1063_106344

/-- The speed of a train given the lengths of two trains, the speed of the other train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 : ℝ) (length2 : ℝ) (speed2 : ℝ) (cross_time : ℝ) :
  length1 = 270 →
  length2 = 230 →
  speed2 = 80 →
  cross_time = 9 / 3600 →
  (length1 + length2) / 1000 / cross_time - speed2 = 120 :=
by sorry

end train_speed_calculation_l1063_106344


namespace maria_coffee_shop_visits_l1063_106333

/-- 
Given that Maria orders 3 cups of coffee each time she goes to the coffee shop
and orders 6 cups of coffee per day, prove that she goes to the coffee shop 2 times per day.
-/
theorem maria_coffee_shop_visits 
  (cups_per_visit : ℕ) 
  (cups_per_day : ℕ) 
  (h1 : cups_per_visit = 3)
  (h2 : cups_per_day = 6) :
  cups_per_day / cups_per_visit = 2 := by
  sorry

end maria_coffee_shop_visits_l1063_106333


namespace simplify_expression_l1063_106371

theorem simplify_expression (a : ℝ) : (1 : ℝ) * (3 * a) * (5 * a^2) * (7 * a^3) * (9 * a^4) = 945 * a^10 := by
  sorry

end simplify_expression_l1063_106371


namespace rectangular_solid_surface_area_l1063_106316

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid is given by 2(lw + wh + hl). -/
def surfaceArea (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (hl : isPrime l) 
  (hw : isPrime w) 
  (hh : isPrime h) 
  (hv : volume l w h = 437) : 
  surfaceArea l w h = 958 := by
  sorry

#check rectangular_solid_surface_area

end rectangular_solid_surface_area_l1063_106316


namespace right_triangle_power_equality_l1063_106350

theorem right_triangle_power_equality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_n_gt_2 : n > 2)
  (h_equality : (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n))) :
  n = 4 := by sorry

end right_triangle_power_equality_l1063_106350


namespace abs_sum_lt_abs_diff_for_opposite_signs_l1063_106346

theorem abs_sum_lt_abs_diff_for_opposite_signs (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end abs_sum_lt_abs_diff_for_opposite_signs_l1063_106346


namespace mark_balloon_cost_l1063_106385

/-- Represents a bag of water balloons -/
structure BalloonBag where
  price : ℕ
  quantity : ℕ

/-- The available bag sizes -/
def availableBags : List BalloonBag := [
  { price := 4, quantity := 50 },
  { price := 6, quantity := 75 },
  { price := 12, quantity := 200 }
]

/-- The total number of balloons Mark wants to buy -/
def targetBalloons : ℕ := 400

/-- Calculates the minimum cost to buy the target number of balloons -/
def minCost (bags : List BalloonBag) (target : ℕ) : ℕ :=
  sorry

theorem mark_balloon_cost :
  minCost availableBags targetBalloons = 24 :=
sorry

end mark_balloon_cost_l1063_106385


namespace lewis_harvest_weeks_l1063_106352

/-- The number of weeks Lewis works during the harvest -/
def harvest_weeks (total_earnings weekly_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proof that Lewis works 5 weeks during the harvest -/
theorem lewis_harvest_weeks :
  harvest_weeks 460 92 = 5 := by
  sorry

end lewis_harvest_weeks_l1063_106352


namespace square_inscribed_in_circle_sum_of_cube_and_reciprocal_polygon_diagonals_distance_between_points_l1063_106363

-- Problem G10.1
theorem square_inscribed_in_circle (d : ℝ) (A : ℝ) (h : d = 10) :
  A = (d^2) / 2 → A = 50 := by sorry

-- Problem G10.2
theorem sum_of_cube_and_reciprocal (a : ℝ) (S : ℝ) (h : a + 1/a = 2) :
  S = a^3 + 1/(a^3) → S = 2 := by sorry

-- Problem G10.3
theorem polygon_diagonals (n : ℕ) :
  n * (n - 3) / 2 = 14 → n = 7 := by sorry

-- Problem G10.4
theorem distance_between_points (d : ℝ) :
  d = Real.sqrt ((2 - (-1))^2 + (3 - 7)^2) → d = 5 := by sorry

end square_inscribed_in_circle_sum_of_cube_and_reciprocal_polygon_diagonals_distance_between_points_l1063_106363


namespace equation_solution_l1063_106360

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end equation_solution_l1063_106360


namespace rectangle_area_l1063_106373

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y)
  (h3 : (x + 4) * (y - 2) = x * y) :
  x * y = 36 := by
sorry

end rectangle_area_l1063_106373


namespace fourth_selection_is_65_l1063_106337

/-- Systematic sampling function -/
def systematicSample (totalParts : ℕ) (sampleSize : ℕ) (firstSelection : ℕ) (selectionNumber : ℕ) : ℕ :=
  let samplingInterval := totalParts / sampleSize
  firstSelection + (selectionNumber - 1) * samplingInterval

/-- Theorem: In the given systematic sampling scenario, the fourth selection is part number 65 -/
theorem fourth_selection_is_65 :
  let totalParts := 200
  let sampleSize := 10
  let firstSelection := 5
  let fourthSelection := 4
  systematicSample totalParts sampleSize firstSelection fourthSelection = 65 := by
  sorry

#eval systematicSample 200 10 5 4  -- Should output 65

end fourth_selection_is_65_l1063_106337


namespace magnitude_sum_vectors_l1063_106335

/-- Given two planar vectors a and b, prove that |a + 2b| = 2√2 -/
theorem magnitude_sum_vectors (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  ‖b‖ = 1 →             -- |b| = 1
  a • b = 0 →           -- angle between a and b is 90°
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
sorry


end magnitude_sum_vectors_l1063_106335


namespace existence_of_person_with_few_amicable_foes_l1063_106359

structure Society where
  n : ℕ  -- number of persons
  q : ℕ  -- number of amicable pairs
  is_valid : q ≤ n * (n - 1) / 2  -- maximum possible number of pairs

def is_hostile (S : Society) (a b : Fin S.n) : Prop := sorry

def is_amicable (S : Society) (a b : Fin S.n) : Prop := ¬(is_hostile S a b)

axiom society_property (S : Society) :
  ∀ (a b c : Fin S.n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    is_hostile S a b ∨ is_hostile S b c ∨ is_hostile S a c

def foes (S : Society) (a : Fin S.n) : Set (Fin S.n) :=
  {b | is_hostile S a b}

def amicable_pairs_among_foes (S : Society) (a : Fin S.n) : ℕ := sorry

theorem existence_of_person_with_few_amicable_foes (S : Society) :
  ∃ (a : Fin S.n), amicable_pairs_among_foes S a ≤ S.q * (1 - 4 * S.q / (S.n * S.n)) :=
sorry

end existence_of_person_with_few_amicable_foes_l1063_106359


namespace equal_numbers_iff_odd_l1063_106311

/-- Represents a square table of numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Initial state of the table with ones on the diagonal and zeros elsewhere -/
def initialTable (n : ℕ) : Table n :=
  λ i j => if i = j then 1 else 0

/-- Represents a closed path of a rook on the table -/
def RookPath (n : ℕ) := List (Fin n × Fin n)

/-- Checks if a path is valid (closed and non-self-intersecting) -/
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

/-- Applies the transformation along a given path -/
def applyTransformation (n : ℕ) (table : Table n) (path : RookPath n) : Table n := sorry

/-- Checks if all numbers in the table are equal -/
def allEqual (n : ℕ) (table : Table n) : Prop := sorry

/-- Main theorem: It's possible to make all numbers equal if and only if n is odd -/
theorem equal_numbers_iff_odd (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    (∀ path ∈ transformations, isValidPath n path) ∧ 
    allEqual n (transformations.foldl (applyTransformation n) (initialTable n))) 
  ↔ n % 2 = 1 := by sorry

end equal_numbers_iff_odd_l1063_106311


namespace elisa_current_amount_l1063_106334

def current_amount (target : ℕ) (needed : ℕ) : ℕ :=
  target - needed

theorem elisa_current_amount :
  let target : ℕ := 53
  let needed : ℕ := 16
  current_amount target needed = 37 := by
  sorry

end elisa_current_amount_l1063_106334


namespace solve_system_l1063_106381

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20)
  (eq2 : 6 * p + 5 * q = 29) :
  q = -25 / 11 := by
  sorry

end solve_system_l1063_106381


namespace terminating_decimal_count_l1063_106375

theorem terminating_decimal_count : 
  let n_range := Finset.range 449
  let divisible_by_nine := n_range.filter (λ n => (n + 1) % 9 = 0)
  divisible_by_nine.card = 49 := by
  sorry

end terminating_decimal_count_l1063_106375


namespace johns_friends_count_l1063_106342

def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

theorem johns_friends_count : 
  (total_cost / cost_per_person) - 1 = 10 := by sorry

end johns_friends_count_l1063_106342


namespace cubic_function_extrema_l1063_106332

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 + x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + 1

theorem cubic_function_extrema (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = 1/6 ∧ b = -3/4) ∧
  (IsLocalMax (f a b) 1 ∧ IsLocalMin (f a b) 2) :=
sorry

end cubic_function_extrema_l1063_106332


namespace apple_division_l1063_106382

theorem apple_division (total_apples : ℕ) (total_weight : ℚ) (portions : ℕ) 
  (h1 : total_apples = 28)
  (h2 : total_weight = 3)
  (h3 : portions = 7) :
  (1 : ℚ) / portions = 1 / 7 ∧ total_weight / portions = 3 / 7 := by
  sorry

end apple_division_l1063_106382


namespace third_side_possible_length_l1063_106377

/-- Given a triangle with two sides of lengths 3 and 7, 
    prove that 6 is a possible length for the third side. -/
theorem third_side_possible_length :
  ∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end third_side_possible_length_l1063_106377


namespace add_9999_seconds_to_5_45_00_l1063_106380

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 5, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 8, minutes := 31, seconds := 39 }

theorem add_9999_seconds_to_5_45_00 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end add_9999_seconds_to_5_45_00_l1063_106380


namespace percentage_relationship_l1063_106305

theorem percentage_relationship (x y : ℝ) : 
  Real.sqrt (0.3 * (x - y)) = Real.sqrt (0.2 * (x + y)) → y = 0.2 * x := by
  sorry

end percentage_relationship_l1063_106305


namespace first_team_pies_l1063_106351

/-- Given a catering problem with three teams making pies, prove the number of pies made by the first team. -/
theorem first_team_pies (total_pies : ℕ) (team2_pies : ℕ) (team3_pies : ℕ)
  (h_total : total_pies = 750)
  (h_team2 : team2_pies = 275)
  (h_team3 : team3_pies = 240) :
  total_pies - team2_pies - team3_pies = 235 := by
  sorry

#check first_team_pies

end first_team_pies_l1063_106351


namespace sqrt_meaningful_iff_x_gt_one_l1063_106390

theorem sqrt_meaningful_iff_x_gt_one (x : ℝ) : 
  (∃ y : ℝ, y * y = 1 / (x - 1)) ↔ x > 1 :=
by sorry

end sqrt_meaningful_iff_x_gt_one_l1063_106390


namespace unique_four_digit_number_l1063_106370

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  (n / 10) % 10 = n % 10 + 2 ∧
  (n / 1000) = (n / 100) % 10 + 2 ∧
  n = 9742 :=
by sorry

end unique_four_digit_number_l1063_106370


namespace peggy_stickers_count_l1063_106345

/-- The number of folders Peggy buys -/
def num_folders : Nat := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : Nat := 10

/-- The number of stickers on each sheet in the red folder -/
def red_stickers_per_sheet : Nat := 3

/-- The number of stickers on each sheet in the green folder -/
def green_stickers_per_sheet : Nat := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : Nat := 1

/-- The total number of stickers Peggy uses -/
def total_stickers : Nat := 
  sheets_per_folder * red_stickers_per_sheet +
  sheets_per_folder * green_stickers_per_sheet +
  sheets_per_folder * blue_stickers_per_sheet

theorem peggy_stickers_count : total_stickers = 60 := by
  sorry

end peggy_stickers_count_l1063_106345


namespace vector_parallel_condition_l1063_106357

-- Define the plane vectors
def a (m : ℝ) : Fin 2 → ℝ := ![1, m]
def b : Fin 2 → ℝ := ![2, 5]
def c (m : ℝ) : Fin 2 → ℝ := ![m, 3]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a m + c m) (a m - b) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end vector_parallel_condition_l1063_106357


namespace bouquet_cost_55_l1063_106389

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  (30 : ℚ) * n / 24

theorem bouquet_cost_55 : bouquet_cost 55 = (68750 : ℚ) / 1000 := by
  sorry

end bouquet_cost_55_l1063_106389


namespace largest_n_value_l1063_106361

def base_8_to_10 (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

def base_9_to_10 (c b a : ℕ) : ℕ := 81 * c + 9 * b + a

theorem largest_n_value (n : ℕ) (a b c : ℕ) :
  (n > 0) →
  (a < 8 ∧ b < 8 ∧ c < 8) →
  (a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8) →
  (n = base_8_to_10 a b c) →
  (n = base_9_to_10 c b a) →
  (∀ m, m > 0 ∧ 
    (∃ x y z, x < 8 ∧ y < 8 ∧ z < 8 ∧ m = base_8_to_10 x y z) ∧
    (∃ x y z, x ≤ 8 ∧ y ≤ 8 ∧ z ≤ 8 ∧ m = base_9_to_10 z y x) →
    m ≤ n) →
  n = 511 := by
  sorry

end largest_n_value_l1063_106361


namespace parallel_x_implies_parallel_y_implies_on_bisector_implies_l1063_106319

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, 2)
def B (b : ℝ) : ℝ × ℝ := (-3, b + 1)

-- Define the conditions
def parallel_to_x_axis (a b : ℝ) : Prop := (A a).2 = (B b).2
def parallel_to_y_axis (a b : ℝ) : Prop := (A a).1 = (B b).1
def on_bisector (a b : ℝ) : Prop := (A a).1 = (A a).2 ∧ (B b).1 = (B b).2

-- Theorem statements
theorem parallel_x_implies (a b : ℝ) : parallel_to_x_axis a b → a ≠ -2 ∧ b = 1 := by sorry

theorem parallel_y_implies (a b : ℝ) : parallel_to_y_axis a b → a = -2 ∧ b ≠ 1 := by sorry

theorem on_bisector_implies (a b : ℝ) : on_bisector a b → a = 3 ∧ b = -4 := by sorry

end parallel_x_implies_parallel_y_implies_on_bisector_implies_l1063_106319


namespace gcd_count_for_product_600_l1063_106379

theorem gcd_count_for_product_600 : 
  ∃ (S : Finset Nat), 
    (∀ d ∈ S, ∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) ∧
    (∀ d : Nat, (∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) → d ∈ S) ∧
    S.card = 14 := by
  sorry

end gcd_count_for_product_600_l1063_106379


namespace donuts_left_for_coworkers_l1063_106376

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := 18
def chocolate_gluten_free : ℕ := 6
def plain_gluten_free : ℕ := 6
def chocolate_regular : ℕ := 11
def plain_regular : ℕ := 7

def eaten_while_driving_gluten_free : ℕ := 1
def eaten_while_driving_regular : ℕ := 1

def afternoon_snack_regular : ℕ := 3
def afternoon_snack_gluten_free : ℕ := 3

theorem donuts_left_for_coworkers :
  total_donuts - 
  (eaten_while_driving_gluten_free + eaten_while_driving_regular + 
   afternoon_snack_regular + afternoon_snack_gluten_free) = 23 := by
  sorry

end donuts_left_for_coworkers_l1063_106376


namespace min_troupe_size_l1063_106328

theorem min_troupe_size : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 8 ∣ m ∧ 10 ∣ m ∧ 12 ∣ m) → n ≤ m :=
by
  use 120
  sorry

end min_troupe_size_l1063_106328


namespace isosceles_right_triangle_area_l1063_106312

/-- 
Given an isosceles right triangle that, when folded twice along the altitude to its hypotenuse, 
results in a smaller isosceles right triangle with leg length 2 cm, 
prove that the area of the original triangle is 4 square centimeters.
-/
theorem isosceles_right_triangle_area (a : ℝ) (h1 : a > 0) : 
  (a / Real.sqrt 2 = 2) → (1 / 2 * a * a = 4) := by sorry

end isosceles_right_triangle_area_l1063_106312


namespace lcm_inequality_l1063_106302

theorem lcm_inequality (n : ℕ) (k : ℕ) (a : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, n ≥ a i)
  (h2 : ∀ i j : Fin k, i < j → a i > a j)
  (h3 : ∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) :
  ∀ i : Fin k, (i.val + 1) * a i ≤ n := by
  sorry

end lcm_inequality_l1063_106302


namespace difference_from_sum_and_difference_of_squares_l1063_106378

theorem difference_from_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end difference_from_sum_and_difference_of_squares_l1063_106378
