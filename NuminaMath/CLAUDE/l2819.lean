import Mathlib

namespace NUMINAMATH_CALUDE_temperature_difference_l2819_281921

def january_temp : ℝ := -3
def march_temp : ℝ := 2

theorem temperature_difference : march_temp - january_temp = 5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2819_281921


namespace NUMINAMATH_CALUDE_store_products_theorem_l2819_281936

-- Define the universe of products in the store
variable (Product : Type)

-- Define a predicate for products that are for sale
variable (for_sale : Product → Prop)

-- Define a predicate for products that are displayed
variable (displayed : Product → Prop)

-- Theorem: If not all displayed products are for sale, then some displayed products are not for sale
-- and not all displayed products are for sale
theorem store_products_theorem :
  (¬ ∀ p, displayed p → for_sale p) →
  ((∃ p, displayed p ∧ ¬ for_sale p) ∧
   ¬ (∀ p, displayed p → for_sale p)) :=
by sorry

end NUMINAMATH_CALUDE_store_products_theorem_l2819_281936


namespace NUMINAMATH_CALUDE_tysons_three_pointers_l2819_281954

/-- Tyson's basketball scoring problem -/
theorem tysons_three_pointers (x : ℕ) : 
  (3 * x + 2 * 12 + 1 * 6 = 75) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_tysons_three_pointers_l2819_281954


namespace NUMINAMATH_CALUDE_hyperbola_parameter_l2819_281949

/-- Given a parabola y^2 = 16x and a hyperbola (x^2/a^2) - (y^2/b^2) = 1 where:
    1. The right focus of the hyperbola coincides with the focus of the parabola (4, 0)
    2. The left directrix of the hyperbola is x = -3
    Then a^2 = 12 -/
theorem hyperbola_parameter (a b : ℝ) : 
  (∃ (x y : ℝ), y^2 = 16*x) → -- Parabola exists
  (∃ (x y : ℝ), (x^2/a^2) - (y^2/b^2) = 1) → -- Hyperbola exists
  (4 : ℝ) = a^2/(2*a) → -- Right focus of hyperbola is (4, 0)
  (-3 : ℝ) = -a^2/(2*a) → -- Left directrix of hyperbola is x = -3
  a^2 = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_l2819_281949


namespace NUMINAMATH_CALUDE_positive_interval_l2819_281961

theorem positive_interval (x : ℝ) : 
  (x + 3) * (x - 1) > 0 ↔ x < (1 - Real.sqrt 13) / 2 ∨ x > (1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_l2819_281961


namespace NUMINAMATH_CALUDE_lunch_sharing_l2819_281973

theorem lunch_sharing (y : ℝ) : 
  let sam_portion := y
  let lee_portion := 1.5 * y
  let sam_initial_eaten := (2/3) * sam_portion
  let lee_initial_eaten := (2/3) * lee_portion
  let sam_remaining := sam_portion - sam_initial_eaten
  let lee_remaining := lee_portion - lee_initial_eaten
  let lee_gives_sam := (1/2) * lee_remaining
  let sam_final_eaten := sam_initial_eaten + lee_gives_sam
  let lee_final_eaten := lee_initial_eaten - lee_gives_sam
  sam_final_eaten = lee_final_eaten →
  sam_portion + lee_portion = 2.5 * y :=
by sorry

end NUMINAMATH_CALUDE_lunch_sharing_l2819_281973


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l2819_281952

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l2819_281952


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2819_281934

theorem angle_bisector_theorem (a b : Real) (h : b - a = 100) :
  (b / 2) - (a / 2) = 50 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2819_281934


namespace NUMINAMATH_CALUDE_r_earnings_l2819_281904

def daily_earnings (p q r : ℝ) : Prop :=
  9 * (p + q + r) = 1980 ∧
  5 * (p + r) = 600 ∧
  7 * (q + r) = 910

theorem r_earnings (p q r : ℝ) (h : daily_earnings p q r) : r = 30 := by
  sorry

end NUMINAMATH_CALUDE_r_earnings_l2819_281904


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l2819_281967

/-- The number of grandchildren --/
def n : ℕ := 12

/-- The probability of a child being male or female --/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters --/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l2819_281967


namespace NUMINAMATH_CALUDE_max_quarters_l2819_281978

theorem max_quarters (total : ℚ) (q : ℕ) : 
  total = 4.55 →
  (0.25 * q + 0.05 * q + 0.1 * (q / 2 : ℚ) = total) →
  (∀ n : ℕ, (0.25 * n + 0.05 * n + 0.1 * (n / 2 : ℚ) ≤ total)) →
  q = 13 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_l2819_281978


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l2819_281902

/-- Given a koala that absorbs 30% of the fiber it eats and absorbed 15 ounces of fiber in one day,
    prove that the total amount of fiber eaten is 50 ounces. -/
theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) : 
  absorption_rate = 0.30 →
  absorbed_fiber = 15 →
  total_fiber * absorption_rate = absorbed_fiber →
  total_fiber = 50 := by
  sorry


end NUMINAMATH_CALUDE_koala_fiber_consumption_l2819_281902


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2819_281944

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2819_281944


namespace NUMINAMATH_CALUDE_division_problem_l2819_281908

theorem division_problem (L S q : ℕ) : 
  L - S = 1365 → 
  L = 1634 → 
  L = S * q + 20 → 
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2819_281908


namespace NUMINAMATH_CALUDE_turner_tickets_l2819_281956

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1

def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def ferris_wheel_cost : ℕ := 1

def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_tickets : total_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_turner_tickets_l2819_281956


namespace NUMINAMATH_CALUDE_line_count_theorem_l2819_281988

-- Define the angle between two lines
def angle_between_lines (a b : Line) : ℝ := sorry

-- Define a function to count the number of lines satisfying the conditions
def count_lines (a b : Line) (P : Point) (α : ℝ) : ℕ := sorry

-- Main theorem
theorem line_count_theorem (a b : Line) (P : Point) (α : ℝ) :
  angle_between_lines a b = 60 →
  0 < α ∧ α < 90 →
  count_lines a b P α = 
    if α < 30 then 0
    else if α = 30 then 1
    else if α < 60 then 2
    else if α = 60 then 3
    else 4 :=
by sorry

end NUMINAMATH_CALUDE_line_count_theorem_l2819_281988


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l2819_281995

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → additional_storks = 6 →
  initial_birds + initial_storks + additional_storks = 13 := by
sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l2819_281995


namespace NUMINAMATH_CALUDE_constant_product_rule_l2819_281994

theorem constant_product_rule (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  a * b = (k * a) * (b / k) :=
by sorry

end NUMINAMATH_CALUDE_constant_product_rule_l2819_281994


namespace NUMINAMATH_CALUDE_triangular_difference_2015_l2819_281974

theorem triangular_difference_2015 : ∃ (n k : ℕ), 
  1000 ≤ n * (n + 1) / 2 ∧ n * (n + 1) / 2 < 10000 ∧
  1000 ≤ k * (k + 1) / 2 ∧ k * (k + 1) / 2 < 10000 ∧
  n * (n + 1) / 2 - k * (k + 1) / 2 = 2015 :=
by sorry


end NUMINAMATH_CALUDE_triangular_difference_2015_l2819_281974


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2819_281972

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_50 : 
  units_digit (sum_factorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2819_281972


namespace NUMINAMATH_CALUDE_transformed_sine_value_l2819_281953

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem transformed_sine_value 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 ≤ φ ∧ φ < π/2) 
  (h_transform : ∀ x, Real.sin x = Real.sin (2 * ω * (x - π/6) + φ)) :
  f ω φ (π/6) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_transformed_sine_value_l2819_281953


namespace NUMINAMATH_CALUDE_count_arrangements_eq_six_l2819_281959

/-- The number of different four-digit numbers that can be formed by arranging the digits in 5006 -/
def count_arrangements : ℕ :=
  let digits : List ℕ := [5, 0, 0, 6]
  let valid_first_digits : List ℕ := digits.filter (λ d => d ≠ 0)
  let remaining_digits : ℕ := digits.length - 1
  valid_first_digits.length * (Nat.factorial remaining_digits / Nat.factorial (remaining_digits - (digits.length - digits.dedup.length)))

theorem count_arrangements_eq_six : count_arrangements = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_eq_six_l2819_281959


namespace NUMINAMATH_CALUDE_log_sum_abs_l2819_281966

theorem log_sum_abs (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_abs_l2819_281966


namespace NUMINAMATH_CALUDE_gcd_of_sequence_l2819_281924

theorem gcd_of_sequence (n : ℕ) : 
  ∃ d : ℕ, d > 0 ∧ 
  (∀ m : ℕ, d ∣ (7^(m+2) + 8^(2*m+1))) ∧
  (∀ k : ℕ, k > 0 → (∀ m : ℕ, k ∣ (7^(m+2) + 8^(2*m+1))) → k ≤ d) ∧
  d = 57 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_sequence_l2819_281924


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_roots_l2819_281963

/-- Given a, b, and c are side lengths of a triangle, 
    the quadratic equation (a+b)x^2 + 2cx + a+b = 0 has no real roots -/
theorem triangle_quadratic_no_roots (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  ∀ x : ℝ, (a + b) * x^2 + 2 * c * x + (a + b) ≠ 0 := by
  sorry

#check triangle_quadratic_no_roots

end NUMINAMATH_CALUDE_triangle_quadratic_no_roots_l2819_281963


namespace NUMINAMATH_CALUDE_product_xy_l2819_281920

theorem product_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_l2819_281920


namespace NUMINAMATH_CALUDE_scooter_depreciation_l2819_281927

theorem scooter_depreciation (initial_value : ℝ) : 
  (((initial_value * (3/4)) * (3/4)) = 22500) → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_depreciation_l2819_281927


namespace NUMINAMATH_CALUDE_inequality_condition_l2819_281946

-- Define the conditions
def has_solutions (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0

def condition_q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- State the theorem
theorem inequality_condition :
  (∀ a : ℝ, condition_q a → has_solutions a) ∧
  ¬(∀ a : ℝ, has_solutions a → condition_q a) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l2819_281946


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l2819_281980

/-- Definition of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Function to check if a line is a common external tangent to two circles -/
def isCommonExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept : 
  let c1 : Circle := { center := (2, 4), radius := 4 }
  let c2 : Circle := { center := (14, 9), radius := 9 }
  ∃ l : Line, l.slope > 0 ∧ isCommonExternalTangent l c1 c2 ∧ l.intercept = 912 / 119 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l2819_281980


namespace NUMINAMATH_CALUDE_two_solutions_l2819_281976

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x * (x - 6) = 7

-- Theorem statement
theorem two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ 
  quadratic_equation a ∧ 
  quadratic_equation b ∧
  ∀ (c : ℝ), quadratic_equation c → (c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l2819_281976


namespace NUMINAMATH_CALUDE_rent_increase_problem_l2819_281931

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.16) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1250 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l2819_281931


namespace NUMINAMATH_CALUDE_living_room_set_cost_l2819_281917

theorem living_room_set_cost (coach_cost sectional_cost paid_amount : ℚ)
  (h1 : coach_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : paid_amount = 7200)
  (discount_rate : ℚ)
  (h4 : discount_rate = 0.1) :
  ∃ (additional_cost : ℚ),
    paid_amount = (1 - discount_rate) * (coach_cost + sectional_cost + additional_cost) ∧
    additional_cost = 2000 := by
sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l2819_281917


namespace NUMINAMATH_CALUDE_sin_600_degrees_l2819_281979

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l2819_281979


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2819_281937

theorem triangle_angle_problem (x : ℝ) :
  (∃ (A B C : ℝ), 
    A = 40 ∧ 
    B = x ∧ 
    C = 2*x ∧ 
    A + B + C = 180) → 
  x = 140/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2819_281937


namespace NUMINAMATH_CALUDE_periodic_scaled_function_l2819_281943

-- Define a real-valued function with period T
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define F(x) = f(αx)
def F (f : ℝ → ℝ) (α : ℝ) (x : ℝ) : ℝ := f (α * x)

-- Theorem statement
theorem periodic_scaled_function
  (f : ℝ → ℝ) (T α : ℝ) (h_periodic : is_periodic f T) (h_pos : α > 0) :
  is_periodic (F f α) (T / α) :=
sorry

end NUMINAMATH_CALUDE_periodic_scaled_function_l2819_281943


namespace NUMINAMATH_CALUDE_student_mistake_difference_l2819_281990

theorem student_mistake_difference : (5/6 : ℚ) * 96 - (5/16 : ℚ) * 96 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l2819_281990


namespace NUMINAMATH_CALUDE_paint_cube_cost_l2819_281998

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)        -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)    -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)         -- Length of cube side in feet
  (h1 : paint_cost = 20)  -- Paint costs Rs. 20 per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)    -- Cube has sides of 5 feet
  : ℝ :=
by
  -- The proof would go here
  sorry

#check paint_cube_cost

end NUMINAMATH_CALUDE_paint_cube_cost_l2819_281998


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2819_281912

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 17.6 →
  a = 15 →
  b = 20 →
  c = 22 →
  d = e →
  d * e = 240.25 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2819_281912


namespace NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l2819_281913

/-- The number of sides in a polygon where each exterior angle measures 40 degrees -/
def polygon_sides : ℕ :=
  (360 : ℕ) / 40

/-- Theorem: A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  polygon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l2819_281913


namespace NUMINAMATH_CALUDE_congruence_solution_l2819_281909

theorem congruence_solution (n : ℤ) : 19 * n ≡ 13 [ZMOD 47] → n ≡ 25 [ZMOD 47] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2819_281909


namespace NUMINAMATH_CALUDE_macaron_distribution_theorem_l2819_281901

/-- The number of kids who receive macarons given the conditions of macaron production and distribution -/
def kids_receiving_macarons (mitch_total : ℕ) (mitch_burnt : ℕ) (joshua_extra : ℕ) 
  (joshua_undercooked : ℕ) (renz_burnt : ℕ) (leah_total : ℕ) (leah_undercooked : ℕ) 
  (first_kids : ℕ) (first_kids_macarons : ℕ) (remaining_kids_macarons : ℕ) : ℕ :=
  let miles_total := 2 * (mitch_total + joshua_extra)
  let renz_total := (3 * miles_total) / 4 - 1
  let total_good_macarons := (mitch_total - mitch_burnt) + 
    (mitch_total + joshua_extra - joshua_undercooked) + 
    miles_total + (renz_total - renz_burnt) + 
    (leah_total - leah_undercooked)
  let remaining_macarons := total_good_macarons - (first_kids * first_kids_macarons)
  first_kids + (remaining_macarons / remaining_kids_macarons)

theorem macaron_distribution_theorem : 
  kids_receiving_macarons 20 2 6 3 4 35 5 10 3 2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_macaron_distribution_theorem_l2819_281901


namespace NUMINAMATH_CALUDE_almond_butter_servings_l2819_281971

/-- Represents a mixed number as a whole number part and a fraction part -/
structure MixedNumber where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Converts a mixed number to a rational number -/
def mixedNumberToRational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem almond_butter_servings 
  (container_amount : MixedNumber) 
  (serving_size : ℚ) 
  (h1 : container_amount = ⟨37, 2, 3, by norm_num⟩) 
  (h2 : serving_size = 3) : 
  ∃ (result : MixedNumber), 
    mixedNumberToRational result = 
      mixedNumberToRational container_amount / serving_size ∧
    result = ⟨12, 5, 9, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l2819_281971


namespace NUMINAMATH_CALUDE_loss_equals_five_balls_cost_l2819_281960

/-- The number of balls whose cost price equals the loss when selling 17 balls -/
def loss_in_balls (total_balls : ℕ) (selling_price : ℕ) (cost_per_ball : ℕ) : ℕ :=
  (total_balls * cost_per_ball - selling_price) / cost_per_ball

theorem loss_equals_five_balls_cost : 
  loss_in_balls 17 720 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_equals_five_balls_cost_l2819_281960


namespace NUMINAMATH_CALUDE_least_positive_integer_l2819_281915

theorem least_positive_integer (x : ℕ) : x = 6 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*41*(2*y) + 41^2) % 53 = 0) ∧
  ((2*x)^2 + 2*41*(2*x) + 41^2) % 53 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_l2819_281915


namespace NUMINAMATH_CALUDE_range_of_x_l2819_281930

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) :
  (∀ x, p x ∨ q x) → (∀ x, ¬q x) → x ≤ 0 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2819_281930


namespace NUMINAMATH_CALUDE_kieras_envelopes_l2819_281919

theorem kieras_envelopes :
  ∀ (yellow : ℕ),
  let blue := 14
  let green := 3 * yellow
  yellow < blue →
  blue + yellow + green = 46 →
  blue - yellow = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_kieras_envelopes_l2819_281919


namespace NUMINAMATH_CALUDE_sum_2018_terms_equals_1009_l2819_281987

/-- Definition of an arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Theorem: Sum of 2018 terms of specific arithmetic sequence -/
theorem sum_2018_terms_equals_1009 (a : ℕ → ℚ) (d : ℚ) :
  arithmetic_sequence a d →
  a 1 = 1 →
  d = -1 / 2017 →
  sum_arithmetic_sequence a 2018 = 1009 := by sorry

end NUMINAMATH_CALUDE_sum_2018_terms_equals_1009_l2819_281987


namespace NUMINAMATH_CALUDE_machine_A_time_l2819_281905

/-- The time it takes for machines A, B, and C to finish a job together -/
def combined_time : ℝ := 2.181818181818182

/-- The time it takes for machine B to finish the job alone -/
def time_B : ℝ := 12

/-- The time it takes for machine C to finish the job alone -/
def time_C : ℝ := 8

/-- Theorem stating that if machines A, B, and C working together can finish a job in 
    2.181818181818182 hours, machine B alone takes 12 hours, and machine C alone takes 8 hours, 
    then machine A alone takes 4 hours to finish the job -/
theorem machine_A_time : 
  ∃ (time_A : ℝ), 
    1 / time_A + 1 / time_B + 1 / time_C = 1 / combined_time ∧ 
    time_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_time_l2819_281905


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l2819_281923

/-- Given a three-digit number satisfying specific conditions, prove it equals 824 -/
theorem three_digit_number_proof (x y z : ℕ) : 
  z^2 = x * y →
  y = (x + z) / 6 →
  100 * x + 10 * y + z - 396 = 100 * z + 10 * y + x →
  100 * x + 10 * y + z = 824 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l2819_281923


namespace NUMINAMATH_CALUDE_max_y_value_l2819_281938

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2819_281938


namespace NUMINAMATH_CALUDE_convenience_store_choices_l2819_281935

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1 : Nat) (set2 : Nat) : Nat :=
  set1 * set2

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem convenience_store_choices :
  choose_one_from_each 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_choices_l2819_281935


namespace NUMINAMATH_CALUDE_monomial_existence_l2819_281932

/-- A monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_power : ℕ
  b_power : ℕ

/-- Multiplication of monomials -/
def mul_monomial (x y : Monomial) : Monomial :=
  { coeff := x.coeff * y.coeff,
    a_power := x.a_power + y.a_power,
    b_power := x.b_power + y.b_power }

/-- Addition of monomials -/
def add_monomial (x y : Monomial) : Option Monomial :=
  if x.a_power = y.a_power ∧ x.b_power = y.b_power then
    some { coeff := x.coeff + y.coeff,
           a_power := x.a_power,
           b_power := x.b_power }
  else
    none

theorem monomial_existence : ∃ (x y : Monomial),
  (mul_monomial x y = { coeff := -12, a_power := 4, b_power := 2 }) ∧
  (∃ (z : Monomial), add_monomial x y = some z ∧ z.coeff = 1) :=
sorry

end NUMINAMATH_CALUDE_monomial_existence_l2819_281932


namespace NUMINAMATH_CALUDE_williams_points_l2819_281965

/-- The number of classes in the contest -/
def num_classes : ℕ := 4

/-- Points scored by Mr. Adams' class -/
def adams_points : ℕ := 57

/-- Points scored by Mrs. Brown's class -/
def brown_points : ℕ := 49

/-- Points scored by Mrs. Daniel's class -/
def daniel_points : ℕ := 57

/-- The mean of the number of points scored -/
def mean_points : ℚ := 53.3

/-- Theorem stating that Mrs. William's class scored 50 points -/
theorem williams_points : ℕ := by
  sorry

end NUMINAMATH_CALUDE_williams_points_l2819_281965


namespace NUMINAMATH_CALUDE_share_of_b_l2819_281993

theorem share_of_b (A B C : ℕ) : 
  A = 3 * B → 
  B = C + 25 → 
  A + B + C = 645 → 
  B = 134 := by
sorry

end NUMINAMATH_CALUDE_share_of_b_l2819_281993


namespace NUMINAMATH_CALUDE_parabola_intersection_l2819_281970

/-- The parabola y = x^2 - 2x - 3 intersects the x-axis at (-1, 0) and (3, 0) -/
theorem parabola_intersection (x : ℝ) :
  let y := x^2 - 2*x - 3
  (y = 0 ∧ x = -1) ∨ (y = 0 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2819_281970


namespace NUMINAMATH_CALUDE_point_on_graph_l2819_281951

/-- A linear function passing through (0, -3) with slope 2 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The point (2, 1) lies on the graph of f -/
theorem point_on_graph : f 2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l2819_281951


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l2819_281925

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (13/7, 41/14, 55/7) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (13/7, 41/14, 55/7) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l2819_281925


namespace NUMINAMATH_CALUDE_closed_grid_path_even_length_l2819_281940

/-- A closed path on a grid -/
structure GridPath where
  up : ℕ
  down : ℕ
  right : ℕ
  left : ℕ
  closed : up = down ∧ right = left

/-- The length of a grid path -/
def GridPath.length (p : GridPath) : ℕ :=
  p.up + p.down + p.right + p.left

/-- Theorem: The length of any closed grid path is even -/
theorem closed_grid_path_even_length (p : GridPath) : 
  Even p.length := by
sorry

end NUMINAMATH_CALUDE_closed_grid_path_even_length_l2819_281940


namespace NUMINAMATH_CALUDE_min_value_of_x_l2819_281996

theorem min_value_of_x (x : ℝ) : 
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) → 
  x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_min_value_of_x_l2819_281996


namespace NUMINAMATH_CALUDE_remaining_land_to_clean_l2819_281999

theorem remaining_land_to_clean 
  (total_land : ℕ) 
  (lizzie_group : ℕ) 
  (other_group : ℕ) 
  (h1 : total_land = 900) 
  (h2 : lizzie_group = 250) 
  (h3 : other_group = 265) : 
  total_land - (lizzie_group + other_group) = 385 := by
sorry

end NUMINAMATH_CALUDE_remaining_land_to_clean_l2819_281999


namespace NUMINAMATH_CALUDE_hexagonal_diagram_impossible_l2819_281992

/-- Represents a hexagonal diagram filled with numbers -/
structure HexagonalDiagram :=
  (first_row : Fin 6 → ℕ)
  (is_valid : ∀ i : Fin 6, first_row i ∈ Finset.range 22)

/-- Calculates the sum of all numbers in the hexagonal diagram -/
def hexagon_sum (h : HexagonalDiagram) : ℕ :=
  6 * h.first_row 0 + 20 * h.first_row 1 + 34 * h.first_row 2 +
  34 * h.first_row 3 + 20 * h.first_row 4 + 6 * h.first_row 5

/-- The sum of numbers from 1 to 21 -/
def sum_1_to_21 : ℕ := (21 * 22) / 2

/-- Theorem stating the impossibility of filling the hexagonal diagram -/
theorem hexagonal_diagram_impossible :
  ¬ ∃ (h : HexagonalDiagram), hexagon_sum h = sum_1_to_21 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_diagram_impossible_l2819_281992


namespace NUMINAMATH_CALUDE_line_intersection_l2819_281969

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 7*x + 12

-- Define the linear function
def g (m b x : ℝ) : ℝ := m*x + b

-- Define the distance between two points on the same vertical line
def distance (k m b : ℝ) : ℝ := |f k - g m b k|

theorem line_intersection (m b : ℝ) : 
  (∃ k, distance k m b = 8) ∧ 
  g m b 2 = 7 ∧ 
  b ≠ 0 →
  (m = 1 ∧ b = 5) ∨ (m = 5 ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_l2819_281969


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2819_281957

/-- Calculates the number of rods in a triangle with given number of rows -/
def num_rods (n : ℕ) : ℕ := n * (n + 1) * 3

/-- Calculates the number of connectors in a triangle with given number of rows -/
def num_connectors (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The total number of pieces in a triangle with given number of rows -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 366 ∧
  num_rods 3 = 18 ∧
  num_connectors 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2819_281957


namespace NUMINAMATH_CALUDE_ratio_equivalence_l2819_281900

theorem ratio_equivalence (x : ℝ) : 
  (20 / 10 = 25 / x) → x = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l2819_281900


namespace NUMINAMATH_CALUDE_division_problem_l2819_281922

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 12)
  (h2 : divisor = 17)
  (h3 : remainder = 10)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2819_281922


namespace NUMINAMATH_CALUDE_intersection_and_subset_l2819_281907

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | Real.sqrt (x - 1) ≥ 1}

theorem intersection_and_subset : 
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (∀ a : ℝ, (A ∩ B) ⊆ {x | x ≥ a} ↔ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l2819_281907


namespace NUMINAMATH_CALUDE_a_cubed_congruence_l2819_281945

theorem a_cubed_congruence (n : ℕ+) (a : ℤ) 
  (h1 : a * a ≡ 1 [ZMOD n])
  (h2 : a ≡ -1 [ZMOD n]) :
  a^3 ≡ -1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_a_cubed_congruence_l2819_281945


namespace NUMINAMATH_CALUDE_random_selection_theorem_l2819_281981

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table --/
structure Position where
  row : Nat
  column : Nat

/-- Represents a selected participant --/
structure Participant where
  number : Nat

/-- Function to select participants using a random number table --/
def selectParticipants (table : RandomNumberTable) (startPos : Position) (totalStudents : Nat) (numToSelect : Nat) : List Participant :=
  sorry

/-- The given random number table --/
def givenTable : RandomNumberTable :=
  [[03, 47, 43, 73, 86, 36, 96, 47, 36, 61, 46, 98, 63, 71, 62, 33, 26, 16, 80, 45, 60, 11, 14, 10, 95],
   [97, 74, 24, 67, 62, 42, 81, 14, 57, 20, 42, 53, 32, 37, 32, 27, 07, 36, 07, 51, 24, 51, 79, 89, 73],
   [16, 76, 62, 27, 66, 56, 50, 26, 71, 07, 32, 90, 79, 78, 53, 13, 55, 38, 58, 59, 88, 97, 54, 14, 10],
   [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76],
   [55, 59, 56, 35, 64, 38, 54, 82, 46, 22, 31, 62, 43, 09, 90, 06, 18, 44, 32, 53, 23, 83, 01, 30, 30]]

theorem random_selection_theorem :
  let startPos : Position := ⟨4, 9⟩
  let totalStudents : Nat := 247
  let numToSelect : Nat := 4
  let selectedParticipants := selectParticipants givenTable startPos totalStudents numToSelect
  selectedParticipants = [⟨050⟩, ⟨121⟩, ⟨014⟩, ⟨218⟩] :=
by sorry

end NUMINAMATH_CALUDE_random_selection_theorem_l2819_281981


namespace NUMINAMATH_CALUDE_chord_distance_from_center_l2819_281982

theorem chord_distance_from_center (R : ℝ) (chord_length : ℝ) (h1 : R = 13) (h2 : chord_length = 10) :
  ∃ d : ℝ, d = 12 ∧ d^2 + (chord_length/2)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_from_center_l2819_281982


namespace NUMINAMATH_CALUDE_purely_imaginary_m_l2819_281962

theorem purely_imaginary_m (m : ℝ) : 
  (m^2 - m : ℂ) + 3*I = (0 : ℝ) + I * (3 : ℝ) → m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_l2819_281962


namespace NUMINAMATH_CALUDE_mixture_composition_l2819_281903

theorem mixture_composition (alcohol_water_ratio : ℚ) (alcohol_fraction : ℚ) :
  alcohol_water_ratio = 1/2 →
  alcohol_fraction = 1/7 →
  1 - alcohol_fraction = 2/7 :=
by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2819_281903


namespace NUMINAMATH_CALUDE_min_value_theorem_l2819_281914

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_mnp : m * n * p = 8) (h_xyz : x * y * z = 8) :
  let f := x^2 + y^2 + z^2 + m*x*y + n*x*z + p*y*z
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → f ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (m = 2 ∧ n = 2 ∧ p = 2 → ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    36 ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    6 * (2^(1/3 : ℝ)) * (m^(2/3 : ℝ) + n^(2/3 : ℝ) + p^(2/3 : ℝ)) ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2819_281914


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_l2819_281955

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(x^4 - 3*x + 7) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  p 1 = -42 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_l2819_281955


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l2819_281910

theorem tan_sum_product_equals_one :
  ∀ (x y : Real),
  (x = 17 * π / 180 ∧ y = 28 * π / 180) →
  (∀ (A B : Real), Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B)) →
  (x + y = π / 4) →
  (Real.tan (π / 4) = 1) →
  Real.tan x + Real.tan y + Real.tan x * Real.tan y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l2819_281910


namespace NUMINAMATH_CALUDE_smallest_factor_b_l2819_281947

theorem smallest_factor_b : 
  ∀ b : ℕ+, 
    (∃ (p q : ℤ), (∀ x : ℝ, x^2 + b * x + 2016 = (x + p) * (x + q))) →
    b ≥ 92 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_b_l2819_281947


namespace NUMINAMATH_CALUDE_one_solution_condition_l2819_281984

theorem one_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ -4 ∧ x ≠ 1 ∧ |x + 1| = |x - 4| + a) ↔ a ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_one_solution_condition_l2819_281984


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_is_correct_l2819_281997

/-- The cost of a zoo entry ticket per person -/
def zoo_ticket_cost : ℝ := 5

/-- The one-way bus fare per person -/
def bus_fare : ℝ := 1.5

/-- The total amount of money brought -/
def total_amount : ℝ := 40

/-- The amount left after buying tickets and paying for bus fare -/
def amount_left : ℝ := 24

/-- The number of people -/
def num_people : ℕ := 2

theorem zoo_ticket_cost_is_correct : 
  zoo_ticket_cost = (total_amount - amount_left - 2 * num_people * bus_fare) / num_people := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_is_correct_l2819_281997


namespace NUMINAMATH_CALUDE_student_sample_total_prove_student_sample_size_l2819_281906

/-- Represents the composition of students in a high school sample -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The theorem stating the total number of students in the sample -/
theorem student_sample_total (s : StudentSample) : s.total = 800 :=
  by
  have h1 : s.juniors = (28 : ℕ) * s.total / 100 := sorry
  have h2 : s.sophomores = (25 : ℕ) * s.total / 100 := sorry
  have h3 : s.seniors = 160 := sorry
  have h4 : s.freshmen = s.sophomores + 16 := sorry
  have h5 : s.total = s.freshmen + s.sophomores + s.juniors + s.seniors := sorry
  sorry

/-- The main theorem proving the total number of students -/
theorem prove_student_sample_size : ∃ s : StudentSample, s.total = 800 :=
  by
  sorry

end NUMINAMATH_CALUDE_student_sample_total_prove_student_sample_size_l2819_281906


namespace NUMINAMATH_CALUDE_bag_probability_l2819_281928

theorem bag_probability (n : ℕ) : 
  (5 : ℚ) / (n + 5) = 1 / 3 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_bag_probability_l2819_281928


namespace NUMINAMATH_CALUDE_prime_natural_equation_solutions_l2819_281929

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) :=
by sorry

end NUMINAMATH_CALUDE_prime_natural_equation_solutions_l2819_281929


namespace NUMINAMATH_CALUDE_water_depth_is_60_feet_l2819_281918

def ron_height : ℝ := 12

def water_depth : ℝ := 5 * ron_height

theorem water_depth_is_60_feet : water_depth = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_60_feet_l2819_281918


namespace NUMINAMATH_CALUDE_marbles_per_bag_is_ten_l2819_281950

/-- The number of marbles in each bag of blue marbles --/
def marbles_per_bag : ℕ := sorry

/-- The initial number of green marbles --/
def initial_green : ℕ := 26

/-- The number of bags of blue marbles bought --/
def blue_bags : ℕ := 6

/-- The number of green marbles given away --/
def green_gift : ℕ := 6

/-- The number of blue marbles given away --/
def blue_gift : ℕ := 8

/-- The total number of marbles Janelle has after giving away the gift --/
def final_total : ℕ := 72

theorem marbles_per_bag_is_ten :
  (initial_green - green_gift) + (blue_bags * marbles_per_bag - blue_gift) = final_total →
  marbles_per_bag = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_per_bag_is_ten_l2819_281950


namespace NUMINAMATH_CALUDE_money_distribution_l2819_281968

theorem money_distribution (m l n : ℚ) (h1 : m > 0) (h2 : l > 0) (h3 : n > 0) 
  (h4 : m / 5 = l / 3) (h5 : m / 5 = n / 2) : 
  (3 * (m / 5)) / (m + l + n) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2819_281968


namespace NUMINAMATH_CALUDE_martha_final_cards_l2819_281926

-- Define the initial number of cards Martha has
def initial_cards : ℝ := 76.0

-- Define the number of cards Martha gives away
def cards_given_away : ℝ := 3.0

-- Theorem statement
theorem martha_final_cards : 
  initial_cards - cards_given_away = 73.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_final_cards_l2819_281926


namespace NUMINAMATH_CALUDE_f_2023_of_5_eq_57_l2819_281933

def f (x : ℚ) : ℚ := (2 + x) / (1 - 2 * x)

def f_n : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => f (f_n n x)

theorem f_2023_of_5_eq_57 : f_n 2023 5 = 57 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_of_5_eq_57_l2819_281933


namespace NUMINAMATH_CALUDE_square_root_difference_squared_l2819_281958

theorem square_root_difference_squared : 
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3))^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_squared_l2819_281958


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2819_281975

/-- The x-coordinate of the point on the x-axis equidistant from A(-4, 0) and B(2, 6) is 2 -/
theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), (x + 4)^2 = (x - 2)^2 + 36 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2819_281975


namespace NUMINAMATH_CALUDE_original_gross_profit_percentage_l2819_281939

theorem original_gross_profit_percentage
  (old_price new_price : ℝ)
  (new_profit_percentage : ℝ)
  (cost : ℝ)
  (h1 : old_price = 88)
  (h2 : new_price = 92)
  (h3 : new_profit_percentage = 0.15)
  (h4 : new_price = cost * (1 + new_profit_percentage)) :
  (old_price - cost) / cost = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_original_gross_profit_percentage_l2819_281939


namespace NUMINAMATH_CALUDE_bicycle_distance_l2819_281948

theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time_minutes : ℝ) :
  motorcycle_speed = 90 →
  bicycle_speed_ratio = 2 / 3 →
  time_minutes = 15 →
  (bicycle_speed_ratio * motorcycle_speed) * (time_minutes / 60) = 15 := by
sorry

end NUMINAMATH_CALUDE_bicycle_distance_l2819_281948


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6_l2819_281941

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6_l2819_281941


namespace NUMINAMATH_CALUDE_angle_problem_l2819_281985

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle2 + angle3 + angle4 = 180)
  (h3 : angle1 = 70)
  (h4 : angle3 = 40) : 
  angle4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l2819_281985


namespace NUMINAMATH_CALUDE_simplify_expression_l2819_281964

theorem simplify_expression : ((4 + 6) * 2) / 4 - 1 / 4 = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2819_281964


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l2819_281983

/-- A natural number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬(Nat.Prime n)

/-- The number of factors of a natural number -/
def numFactors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

/-- Theorem: Any composite number has at least 3 factors -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
  numFactors n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l2819_281983


namespace NUMINAMATH_CALUDE_square_plus_n_plus_one_is_odd_l2819_281916

theorem square_plus_n_plus_one_is_odd (n : ℤ) : Odd (n^2 + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_n_plus_one_is_odd_l2819_281916


namespace NUMINAMATH_CALUDE_power_division_rule_l2819_281977

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l2819_281977


namespace NUMINAMATH_CALUDE_light_distance_half_year_l2819_281911

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 299792

/-- The number of days in half a year -/
def half_year_days : ℝ := 182.5

/-- The distance light travels in half a year -/
def light_distance : ℝ := speed_of_light * half_year_days * 24 * 3600

theorem light_distance_half_year :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 * 10^12 ∧ 
  |light_distance - 4.73 * 10^12| < ε :=
sorry

end NUMINAMATH_CALUDE_light_distance_half_year_l2819_281911


namespace NUMINAMATH_CALUDE_train_length_calculation_l2819_281991

theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 95) (h2 : v2 = 85) (h3 : t = 6) :
  let relative_speed := (v1 + v2) * (5/18)
  let total_length := relative_speed * t
  let train_length := total_length / 2
  train_length = 150 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2819_281991


namespace NUMINAMATH_CALUDE_systematic_sample_fifth_seat_l2819_281989

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 4 → ℕ
  (class_size_pos : class_size > 0)
  (sample_size_pos : sample_size > 0)
  (sample_size_le_class : sample_size ≤ class_size)
  (known_seats_valid : ∀ i, known_seats i ≤ class_size)
  (known_seats_ordered : ∀ i j, i < j → known_seats i < known_seats j)

/-- The theorem to be proved -/
theorem systematic_sample_fifth_seat
  (s : SystematicSample)
  (h1 : s.class_size = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.known_seats 0 = 3)
  (h4 : s.known_seats 1 = 15)
  (h5 : s.known_seats 2 = 39)
  (h6 : s.known_seats 3 = 51) :
  ∃ (fifth_seat : ℕ), fifth_seat = 27 ∧
    (∀ i j, i ≠ j → s.known_seats i ≠ fifth_seat) ∧
    fifth_seat ≤ s.class_size :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fifth_seat_l2819_281989


namespace NUMINAMATH_CALUDE_log_exponent_sum_l2819_281986

theorem log_exponent_sum (a b : ℝ) (h1 : a = Real.log 25) (h2 : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l2819_281986


namespace NUMINAMATH_CALUDE_paiges_math_problems_l2819_281942

theorem paiges_math_problems (total_problems math_problems science_problems finished_problems left_problems : ℕ) :
  science_problems = 12 →
  finished_problems = 44 →
  left_problems = 11 →
  total_problems = math_problems + science_problems →
  total_problems = finished_problems + left_problems →
  math_problems = 43 := by
sorry

end NUMINAMATH_CALUDE_paiges_math_problems_l2819_281942
