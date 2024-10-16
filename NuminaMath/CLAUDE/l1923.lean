import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bounds_l1923_192336

/-- Given a triangle with sides a ≤ b ≤ c and corresponding altitudes ma ≥ mb ≥ mc,
    the radius ρ of the inscribed circle satisfies mc/3 ≤ ρ ≤ ma/3 -/
theorem inscribed_circle_radius_bounds (a b c ma mb mc ρ : ℝ) 
  (h_sides : a ≤ b ∧ b ≤ c)
  (h_altitudes : ma ≥ mb ∧ mb ≥ mc)
  (h_inradius : ρ > 0)
  (h_area : ρ * (a + b + c) = a * ma)
  (h_area_alt : ρ * (a + b + c) = c * mc) :
  mc / 3 ≤ ρ ∧ ρ ≤ ma / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bounds_l1923_192336


namespace NUMINAMATH_CALUDE_harriet_trip_time_l1923_192382

theorem harriet_trip_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 90)
  (h3 : return_speed = 160) :
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed) / outbound_speed * 60 = 192 := by
  sorry

end NUMINAMATH_CALUDE_harriet_trip_time_l1923_192382


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l1923_192385

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + 
  Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l1923_192385


namespace NUMINAMATH_CALUDE_brand_x_pen_price_l1923_192320

/-- The price of a brand X pen satisfies the given conditions -/
theorem brand_x_pen_price :
  ∀ (total_pens brand_x_pens : ℕ) (brand_y_price total_cost brand_x_price : ℚ),
    total_pens = 12 →
    brand_x_pens = 8 →
    brand_y_price = 14/5 →
    total_cost = 40 →
    brand_x_price * brand_x_pens + brand_y_price * (total_pens - brand_x_pens) = total_cost →
    brand_x_price = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_brand_x_pen_price_l1923_192320


namespace NUMINAMATH_CALUDE_train_speed_l1923_192342

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 300 → 
  time = 18 → 
  speed = (length / time) * 3.6 → 
  speed = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_l1923_192342


namespace NUMINAMATH_CALUDE_triangle_properties_l1923_192397

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 6)
  (h2 : Real.sin t.A - Real.sin t.C = Real.sin (t.A - t.B))
  (h3 : t.b = 2 * Real.sqrt 7) :
  t.B = π / 3 ∧ 
  (t.a * t.c * Real.sin t.B / 2 = 3 * Real.sqrt 3 ∨ 
   t.a * t.c * Real.sin t.B / 2 = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1923_192397


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l1923_192302

theorem snooker_tournament_ticket_difference :
  ∀ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = 320 →
    40 * vip_tickets + 10 * general_tickets = 7500 →
    general_tickets - vip_tickets = 34 := by
  sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l1923_192302


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1923_192321

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1923_192321


namespace NUMINAMATH_CALUDE_tax_problem_l1923_192324

/-- Proves that the monthly gross income is 127,500 HUF when the tax equals 30% of the annual income --/
theorem tax_problem (x : ℝ) (h1 : x > 1050000) : 
  (267000 + 0.4 * (x - 1050000) = 0.3 * x) → (x / 12 = 127500) := by
  sorry

end NUMINAMATH_CALUDE_tax_problem_l1923_192324


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1923_192308

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < π / 3) 
  (h3 : Real.sqrt 3 * Real.sin α + Real.cos α = Real.sqrt 6 / 2) : 
  (Real.cos (α + π / 6) = Real.sqrt 10 / 4) ∧ 
  (Real.cos (2 * α + 7 * π / 12) = (Real.sqrt 2 - Real.sqrt 30) / 8) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1923_192308


namespace NUMINAMATH_CALUDE_floor_paving_cost_l1923_192396

/-- Calculates the total cost of paving a floor with different types of slabs -/
theorem floor_paving_cost (room_length room_width : ℝ)
  (square_slab_side square_slab_cost square_slab_percentage : ℝ)
  (rect_slab_length rect_slab_width rect_slab_cost rect_slab_percentage : ℝ)
  (tri_slab_height tri_slab_base tri_slab_cost tri_slab_percentage : ℝ) :
  room_length = 5.5 →
  room_width = 3.75 →
  square_slab_side = 1 →
  square_slab_cost = 800 →
  square_slab_percentage = 0.4 →
  rect_slab_length = 1.5 →
  rect_slab_width = 1 →
  rect_slab_cost = 1000 →
  rect_slab_percentage = 0.35 →
  tri_slab_height = 1 →
  tri_slab_base = 1 →
  tri_slab_cost = 1200 →
  tri_slab_percentage = 0.25 →
  square_slab_percentage + rect_slab_percentage + tri_slab_percentage = 1 →
  (room_length * room_width) * 
    (square_slab_percentage * square_slab_cost + 
     rect_slab_percentage * rect_slab_cost + 
     tri_slab_percentage * tri_slab_cost) = 20006.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l1923_192396


namespace NUMINAMATH_CALUDE_son_age_l1923_192314

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 25 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end NUMINAMATH_CALUDE_son_age_l1923_192314


namespace NUMINAMATH_CALUDE_point_on_curve_iff_f_eq_zero_l1923_192338

-- Define a function f representing the curve
variable (f : ℝ → ℝ → ℝ)

-- Define a point P
variable (x₀ y₀ : ℝ)

-- Theorem stating the necessary and sufficient condition
theorem point_on_curve_iff_f_eq_zero :
  (∃ (x y : ℝ), f x y = 0 ∧ x = x₀ ∧ y = y₀) ↔ f x₀ y₀ = 0 := by sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_f_eq_zero_l1923_192338


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l1923_192374

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ (M : ℝ), M = 10 ∧ ∀ (a b : ℝ), a^2 + b^2 - 2*a + 4*b = 0 → a - 2*b ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l1923_192374


namespace NUMINAMATH_CALUDE_zephyr_orbit_distance_l1923_192300

/-- Given an elliptical orbit with perigee 3 AU and apogee 15 AU, 
    the distance from a point vertically above the center of the ellipse to the focus (sun) is 3√5 + 6 AU -/
theorem zephyr_orbit_distance (perigee apogee : ℝ) (h1 : perigee = 3) (h2 : apogee = 15) :
  let semi_major_axis := (apogee + perigee) / 2
  let semi_minor_axis := Real.sqrt (semi_major_axis^2 - (semi_major_axis - perigee)^2)
  semi_minor_axis + (semi_major_axis - perigee) = 3 * Real.sqrt 5 + 6 := by
sorry

end NUMINAMATH_CALUDE_zephyr_orbit_distance_l1923_192300


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1923_192332

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 13 →
  a * b = 142.40786495134319 →
  d^2 = a^2 + b^2 →
  d = 17 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1923_192332


namespace NUMINAMATH_CALUDE_function_bound_l1923_192357

theorem function_bound (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1)
  (h2 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1) :
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l1923_192357


namespace NUMINAMATH_CALUDE_log_sum_equals_one_implies_product_equals_ten_l1923_192395

theorem log_sum_equals_one_implies_product_equals_ten (a b : ℝ) (h : Real.log a + Real.log b = 1) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_one_implies_product_equals_ten_l1923_192395


namespace NUMINAMATH_CALUDE_surplus_shortage_equation_l1923_192373

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  people : ℕ
  itemPrice : ℕ

/-- Defines the concept of surplus or shortage in a group purchase -/
def contributionDifference (g : GroupPurchase) (contribution : ℕ) : ℤ :=
  (g.people * contribution : ℤ) - g.itemPrice

/-- The main theorem representing the "Surplus and Shortage" problem -/
theorem surplus_shortage_equation (g : GroupPurchase) :
  contributionDifference g 9 = 11 ∧ contributionDifference g 6 = -16 →
  9 * g.people - 11 = 6 * g.people + 16 :=
by sorry

end NUMINAMATH_CALUDE_surplus_shortage_equation_l1923_192373


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1923_192361

theorem log_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 8 = Real.log (p - q) / Real.log 18) : 
  q / p = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1923_192361


namespace NUMINAMATH_CALUDE_find_number_l1923_192390

theorem find_number : ∃ x : ℝ, 1.35 + 0.321 + x = 1.794 ∧ x = 0.123 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1923_192390


namespace NUMINAMATH_CALUDE_area_of_stacked_squares_l1923_192329

/-- The area of a 24-sided polygon formed by stacking three identical square sheets -/
theorem area_of_stacked_squares (side_length : ℝ) (h : side_length = 8) :
  let diagonal := side_length * Real.sqrt 2
  let radius := diagonal / 2
  let triangle_area := (1/2) * radius^2 * Real.sin (π/6)
  let total_area := 12 * triangle_area
  total_area = 96 := by sorry

end NUMINAMATH_CALUDE_area_of_stacked_squares_l1923_192329


namespace NUMINAMATH_CALUDE_total_people_on_boats_l1923_192369

/-- The number of boats in the lake -/
def num_boats : ℕ := 5

/-- The number of people on each boat -/
def people_per_boat : ℕ := 3

/-- The total number of people on boats in the lake -/
def total_people : ℕ := num_boats * people_per_boat

theorem total_people_on_boats : total_people = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_people_on_boats_l1923_192369


namespace NUMINAMATH_CALUDE_second_highest_coefficient_of_g_l1923_192335

/-- Given a polynomial g(x) satisfying g(x + 1) - g(x) = 6x^2 + 4x + 2 for all x,
    prove that the second highest coefficient of g(x) is 2/3 -/
theorem second_highest_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x^2 + 4 * x + 2) :
  ∃ a b c d : ℝ, (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧ b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_second_highest_coefficient_of_g_l1923_192335


namespace NUMINAMATH_CALUDE_expression_value_l1923_192341

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : m = 2 ∨ m = -2) : 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = 11 ∨ 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = -21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1923_192341


namespace NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l1923_192317

theorem sum_of_a_values_for_single_solution (a : ℝ) : 
  let equation := fun (x : ℝ) => 2 * x^2 + a * x + 6 * x + 7
  let discriminant := (a + 6)^2 - 4 * 2 * 7
  let sum_of_a_values := -(12 : ℝ)
  (∃ (a₁ a₂ : ℝ), 
    (∀ x, equation x = 0 → discriminant = 0) ∧ 
    (a₁ ≠ a₂) ∧ 
    (a₁ + a₂ = sum_of_a_values)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l1923_192317


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1923_192384

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q)) ↔ b = 43 :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1923_192384


namespace NUMINAMATH_CALUDE_disease_mortality_percentage_l1923_192348

theorem disease_mortality_percentage (population : ℝ) 
  (h1 : population > 0) 
  (affected_percentage : ℝ) 
  (h2 : affected_percentage = 15) 
  (death_percentage : ℝ) 
  (h3 : death_percentage = 8) : 
  (affected_percentage / 100) * (death_percentage / 100) * 100 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_disease_mortality_percentage_l1923_192348


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1923_192323

/-- The quadratic equation x^2 + 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

/-- The two roots of the quadratic equation -/
noncomputable def root1 : ℝ := sorry
noncomputable def root2 : ℝ := sorry

/-- Proposition p: The two roots have opposite signs -/
def p : Prop := root1 * root2 < 0

/-- Proposition q: The sum of the two roots is 3 -/
def q : Prop := root1 + root2 = 3

theorem quadratic_roots_properties : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1923_192323


namespace NUMINAMATH_CALUDE_circle_radius_calculation_l1923_192303

-- Define the circles and triangle
def circleA : ℝ := 13  -- radius of circle A
def circleB : ℝ := 4   -- radius of circle B
def circleC : ℝ := 3   -- radius of circle C

-- Define the theorem
theorem circle_radius_calculation (r : ℝ) : 
  -- Right triangle T inscribed in circle A
  -- Circle B internally tangent to A at one vertex of T
  -- Circle C internally tangent to A at another vertex of T
  -- Circles B and C externally tangent to circle E with radius r
  -- Angle between radii of A touching vertices related to B and C is 90°
  r = (Real.sqrt 181 - 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_calculation_l1923_192303


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l1923_192381

theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^2 + a*b + c^2 - b*c = 2*a*c) : 
  a = c ∨ a = b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l1923_192381


namespace NUMINAMATH_CALUDE_routes_count_l1923_192377

/-- The number of routes from A to B with 6 horizontal and 6 vertical moves -/
def num_routes : ℕ := 924

/-- The total number of moves -/
def total_moves : ℕ := 12

/-- The number of horizontal moves -/
def horizontal_moves : ℕ := 6

/-- The number of vertical moves -/
def vertical_moves : ℕ := 6

theorem routes_count : 
  num_routes = Nat.choose total_moves horizontal_moves :=
by sorry

end NUMINAMATH_CALUDE_routes_count_l1923_192377


namespace NUMINAMATH_CALUDE_circle_area_three_fourths_l1923_192343

/-- Given a circle where three times the reciprocal of its circumference 
    equals its diameter, prove that its area is 3/4 -/
theorem circle_area_three_fourths (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 2 * r) : 
  π * r^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_three_fourths_l1923_192343


namespace NUMINAMATH_CALUDE_difference_sum_rational_product_irrational_l1923_192315

theorem difference_sum_rational_product_irrational : 
  let a : ℝ := 8
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3 - 1
  let d : ℝ := 3 * Real.sqrt 3
  (a + b) - (c * d) = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_difference_sum_rational_product_irrational_l1923_192315


namespace NUMINAMATH_CALUDE_pyramid_volume_change_specific_pyramid_volume_l1923_192368

/-- Given a pyramid with a triangular base and initial volume V, 
    if the base height is doubled, base dimensions are tripled, 
    and the pyramid's height is increased by 40%, 
    then the new volume is 8.4 * V. -/
theorem pyramid_volume_change (V : ℝ) : 
  V > 0 → 
  (2 * 3 * 3 * 1.4) * V = 8.4 * V :=
by sorry

/-- The new volume of the specific pyramid is 604.8 cubic inches. -/
theorem specific_pyramid_volume : 
  (8.4 : ℝ) * 72 = 604.8 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_specific_pyramid_volume_l1923_192368


namespace NUMINAMATH_CALUDE_boys_who_quit_l1923_192352

theorem boys_who_quit (initial_girls : ℕ) (initial_boys : ℕ) (girls_joined : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → 
  initial_boys = 15 → 
  girls_joined = 7 → 
  final_total = 36 → 
  initial_boys - (final_total - (initial_girls + girls_joined)) = 4 := by
sorry

end NUMINAMATH_CALUDE_boys_who_quit_l1923_192352


namespace NUMINAMATH_CALUDE_stone_skipping_l1923_192309

/-- Represents the number of skips for each throw --/
structure Throws where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Defines the conditions of the stone-skipping problem --/
def validThrows (t : Throws) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = 8 ∧
  t.first + t.second + t.third + t.fourth + t.fifth = 33

/-- The theorem to be proved --/
theorem stone_skipping (t : Throws) (h : validThrows t) : 
  t.fifth - t.fourth = 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_skipping_l1923_192309


namespace NUMINAMATH_CALUDE_football_game_attendance_l1923_192346

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total revenue. -/
theorem football_game_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000)
  : ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1923_192346


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l1923_192398

theorem ring_toss_total_earnings (first_44_days : ℕ) (remaining_10_days : ℕ) (total : ℕ) :
  first_44_days = 382 →
  remaining_10_days = 374 →
  total = first_44_days + remaining_10_days →
  total = 756 := by sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l1923_192398


namespace NUMINAMATH_CALUDE_divisibility_condition_l1923_192354

theorem divisibility_condition (n : ℕ) : 
  n > 0 ∧ (n - 1) ∣ (n^3 + 4) ↔ n = 2 ∨ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1923_192354


namespace NUMINAMATH_CALUDE_sample_size_is_number_of_individuals_l1923_192389

/-- Definition of a sample in statistics -/
structure Sample (α : Type) where
  elements : List α

/-- Definition of sample size -/
def sampleSize {α : Type} (s : Sample α) : ℕ :=
  s.elements.length

/-- Theorem: The sample size is the number of individuals in the sample -/
theorem sample_size_is_number_of_individuals {α : Type} (s : Sample α) :
  sampleSize s = s.elements.length := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_number_of_individuals_l1923_192389


namespace NUMINAMATH_CALUDE_total_fishes_caught_l1923_192350

/-- The number of fishes caught by Hazel and her father in Lake Erie -/
theorem total_fishes_caught (hazel_fishes : Nat) (father_fishes : Nat)
  (h1 : hazel_fishes = 48)
  (h2 : father_fishes = 46) :
  hazel_fishes + father_fishes = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_caught_l1923_192350


namespace NUMINAMATH_CALUDE_decimal_place_of_13_over_17_l1923_192370

/-- The decimal representation of 13/17 repeats every 17 digits -/
def decimal_period : ℕ := 17

/-- The repeating sequence of digits in the decimal representation of 13/17 -/
def repeating_sequence : List ℕ := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 250

theorem decimal_place_of_13_over_17 :
  (repeating_sequence.get! ((target_position - 1) % decimal_period)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_of_13_over_17_l1923_192370


namespace NUMINAMATH_CALUDE_multiplication_and_division_problem_l1923_192356

theorem multiplication_and_division_problem : (-12 * 3) + ((-15 * -5) / 3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_division_problem_l1923_192356


namespace NUMINAMATH_CALUDE_no_consecutive_solution_l1923_192322

theorem no_consecutive_solution : ¬ ∃ (a b c d e f : ℕ), 
  (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3) ∧ (e = a + 4) ∧ (f = a + 5) ∧
  (a * b^c * d + e^f * a * b = 2015) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_solution_l1923_192322


namespace NUMINAMATH_CALUDE_longest_segment_l1923_192375

-- Define the triangle ABD
structure TriangleABD where
  angleABD : ℝ
  angleADB : ℝ
  hab : angleABD = 30
  had : angleADB = 70

-- Define the triangle BCD
structure TriangleBCD where
  angleCBD : ℝ
  angleBDC : ℝ
  hcb : angleCBD = 45
  hbd : angleBDC = 60

-- Define the lengths of the segments
variables {AB AD BD BC CD : ℝ}

-- State the theorem
theorem longest_segment (abd : TriangleABD) (bcd : TriangleBCD) :
  CD > BC ∧ BC > BD ∧ BD > AB ∧ AB > AD :=
sorry

end NUMINAMATH_CALUDE_longest_segment_l1923_192375


namespace NUMINAMATH_CALUDE_factorization_equality_l1923_192326

theorem factorization_equality (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1923_192326


namespace NUMINAMATH_CALUDE_kohens_apples_l1923_192376

/-- Kohen's Apple Business Theorem -/
theorem kohens_apples (boxes : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ) 
  (h1 : boxes = 10)
  (h2 : apples_per_box = 300)
  (h3 : sold_fraction = 3/4) : 
  boxes * apples_per_box - (sold_fraction * (boxes * apples_per_box)).num = 750 := by
  sorry

end NUMINAMATH_CALUDE_kohens_apples_l1923_192376


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l1923_192366

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ π) :
  (∀ x : Real, -1 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ + x * (1 + x) - (1 + x)^2 * Real.sin θ < 0) ↔
  (π / 2 < θ ∧ θ < π) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l1923_192366


namespace NUMINAMATH_CALUDE_halving_period_correct_l1923_192358

/-- The number of years it takes for the cost of a ticket to Mars to be halved. -/
def halving_period : ℕ := 10

/-- The initial cost of a ticket to Mars in dollars. -/
def initial_cost : ℕ := 1000000

/-- The cost of a ticket to Mars after 30 years in dollars. -/
def cost_after_30_years : ℕ := 125000

/-- The number of years passed. -/
def years_passed : ℕ := 30

/-- Theorem stating that the halving period is correct given the initial conditions. -/
theorem halving_period_correct : 
  initial_cost / (2 ^ (years_passed / halving_period)) = cost_after_30_years :=
sorry

end NUMINAMATH_CALUDE_halving_period_correct_l1923_192358


namespace NUMINAMATH_CALUDE_inequality_range_l1923_192330

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (∀ m : ℝ, (1 / x + 4 / y ≥ m) ↔ m ≤ 9 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1923_192330


namespace NUMINAMATH_CALUDE_overtime_pay_ratio_l1923_192306

/-- Given Bill's pay structure, prove the ratio of overtime to regular pay rate --/
theorem overtime_pay_ratio (initial_rate : ℝ) (total_pay : ℝ) (total_hours : ℕ) (regular_hours : ℕ) :
  initial_rate = 20 →
  total_pay = 1200 →
  total_hours = 50 →
  regular_hours = 40 →
  (total_pay - initial_rate * regular_hours) / (total_hours - regular_hours) / initial_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_ratio_l1923_192306


namespace NUMINAMATH_CALUDE_oblique_view_isosceles_implies_right_trapezoid_l1923_192311

/-- A plane figure. -/
structure PlaneFigure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- An oblique view of a plane figure. -/
structure ObliqueView where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents that a figure is an isosceles trapezoid. -/
def is_isosceles_trapezoid (f : ObliqueView) : Prop :=
  sorry

/-- Represents the base angle of a figure. -/
def base_angle (f : ObliqueView) : ℝ :=
  sorry

/-- Represents that a figure is a right trapezoid. -/
def is_right_trapezoid (f : PlaneFigure) : Prop :=
  sorry

/-- 
If the oblique view of a plane figure is an isosceles trapezoid 
with a base angle of 45°, then the original figure is a right trapezoid.
-/
theorem oblique_view_isosceles_implies_right_trapezoid 
  (f : PlaneFigure) (v : ObliqueView) :
  is_isosceles_trapezoid v → base_angle v = 45 → is_right_trapezoid f :=
by
  sorry

end NUMINAMATH_CALUDE_oblique_view_isosceles_implies_right_trapezoid_l1923_192311


namespace NUMINAMATH_CALUDE_factory_non_defective_percentage_l1923_192355

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  { production_percentage := 0.25, defective_percentage := 0.02 },
  { production_percentage := 0.35, defective_percentage := 0.04 },
  { production_percentage := 0.40, defective_percentage := 0.05 }
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating that the percentage of non-defective products is 96.1% -/
theorem factory_non_defective_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

end NUMINAMATH_CALUDE_factory_non_defective_percentage_l1923_192355


namespace NUMINAMATH_CALUDE_greatest_lower_bound_system_l1923_192333

theorem greatest_lower_bound_system (x y z u : ℕ+) 
  (h1 : x ≥ y) 
  (h2 : x + y = z + u) 
  (h3 : 2 * x * y = z * u) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ a b c d : ℕ+, a ≥ b → a + b = c + d → 2 * a * b = c * d → (a : ℝ) / b ≥ m) ∧
  (∀ ε > 0, ∃ a b c d : ℕ+, a ≥ b ∧ a + b = c + d ∧ 2 * a * b = c * d ∧ (a : ℝ) / b < m + ε) :=
sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_system_l1923_192333


namespace NUMINAMATH_CALUDE_chord_length_sqrt3_line_l1923_192387

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- A circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The length of a chord formed by the intersection of a line and a circle --/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

theorem chord_length_sqrt3_line (c : Circle) :
  c.h = 2 ∧ c.k = 0 ∧ c.r = 2 →
  chordLength { m := Real.sqrt 3 } c = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_sqrt3_line_l1923_192387


namespace NUMINAMATH_CALUDE_triangle_area_l1923_192353

theorem triangle_area (A B C : ℝ) (b : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  Real.sin (2 * A) + Real.sin (A - C) - Real.sin B = 0 →  -- Given equation
  (1 / 2) * b * b * Real.sin B = Real.sqrt 3  -- Area of the triangle
  := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1923_192353


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_only_l1923_192340

theorem females_with_advanced_degrees_only (total_employees : ℕ) 
  (female_employees : ℕ) (employees_with_advanced_degrees : ℕ)
  (employees_with_college_only : ℕ) (employees_with_multiple_degrees : ℕ)
  (males_with_college_only : ℕ) (males_with_multiple_degrees : ℕ)
  (females_with_multiple_degrees : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : employees_with_college_only = 55)
  (h5 : employees_with_multiple_degrees = 15)
  (h6 : males_with_college_only = 31)
  (h7 : males_with_multiple_degrees = 8)
  (h8 : females_with_multiple_degrees = 10) :
  total_employees - female_employees - males_with_college_only - males_with_multiple_degrees +
  employees_with_advanced_degrees - females_with_multiple_degrees - males_with_multiple_degrees = 35 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_only_l1923_192340


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1923_192349

theorem max_value_of_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≤ 1/27 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1923_192349


namespace NUMINAMATH_CALUDE_T_is_three_rays_l1923_192360

/-- The set T of points in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               ((4 = x + 3 ∧ y - 5 ≤ 4) ∨
                (4 = y - 5 ∧ x + 3 ≤ 4) ∨
                (x + 3 = y - 5 ∧ 4 ≤ x + 3))}

/-- Definition of a ray starting from a point -/
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

/-- The three rays that should compose T -/
def ThreeRays : Set (ℝ × ℝ) :=
  Ray (1, 9) (0, -1) ∪ Ray (1, 9) (-1, 0) ∪ Ray (1, 9) (1, 1)

theorem T_is_three_rays : T = ThreeRays := by sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l1923_192360


namespace NUMINAMATH_CALUDE_range_of_a_for_circle_condition_l1923_192383

/-- The range of 'a' for which there exists a point M on the circle (x-a)^2 + (y-a+2)^2 = 1
    such that |MA| = 2|MO|, where A is (0, -3) and O is the origin. -/
theorem range_of_a_for_circle_condition (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ 
    (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) ↔ 
  0 ≤ a ∧ a ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_for_circle_condition_l1923_192383


namespace NUMINAMATH_CALUDE_triangle_special_angle_l1923_192331

/-- Given a triangle ABC where b = c and a² = 2b²(1 - sin A), prove that A = π/4 -/
theorem triangle_special_angle (a b c : ℝ) (A : ℝ) 
  (h1 : b = c) 
  (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l1923_192331


namespace NUMINAMATH_CALUDE_shaded_square_covers_all_rows_l1923_192318

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (n + 2)

def covers_all_remainders (n : ℕ) : Prop :=
  ∀ k : Fin 12, ∃ m : ℕ, m ≤ n ∧ shaded_sequence m % 12 = k

theorem shaded_square_covers_all_rows :
  covers_all_remainders 11 ∧ shaded_sequence 11 = 144 ∧
  ∀ k < 11, ¬covers_all_remainders k :=
sorry

end NUMINAMATH_CALUDE_shaded_square_covers_all_rows_l1923_192318


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l1923_192344

/-- Given a cuboid with edges of 4 cm, x cm, and 6 cm, and a volume of 120 cm³, prove that x = 5 cm. -/
theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 4 * x * 6 = 120 → x = 5 := by sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l1923_192344


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1923_192367

theorem trigonometric_identity : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1923_192367


namespace NUMINAMATH_CALUDE_angelinas_speed_l1923_192363

theorem angelinas_speed (v : ℝ) 
  (home_to_grocery : 840 / v = 510 / (1.5 * v) + 40)
  (grocery_to_library : 510 / (1.5 * v) = 480 / (2 * v) + 20) :
  2 * v = 25 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_l1923_192363


namespace NUMINAMATH_CALUDE_speech_contest_probabilities_l1923_192305

def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_selected

theorem speech_contest_probabilities :
  let p_two_boys := (num_boys.choose num_selected : ℚ) / total_combinations
  let p_one_girl := (num_boys.choose 1 * num_girls.choose 1 : ℚ) / total_combinations
  let p_at_least_one_girl := 1 - p_two_boys
  (p_two_boys = 2/5) ∧
  (p_one_girl = 8/15) ∧
  (p_at_least_one_girl = 3/5) := by sorry

end NUMINAMATH_CALUDE_speech_contest_probabilities_l1923_192305


namespace NUMINAMATH_CALUDE_equivalent_angle_for_negative_463_l1923_192347

-- Define the angle equivalence relation
def angle_equivalent (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

-- State the theorem
theorem equivalent_angle_for_negative_463 :
  ∀ k : ℤ, angle_equivalent (-463) (k * 360 + 257) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_angle_for_negative_463_l1923_192347


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1923_192334

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 4) / 9 = 4 / (x - 9) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1923_192334


namespace NUMINAMATH_CALUDE_four_letter_initials_count_l1923_192379

theorem four_letter_initials_count : 
  let letter_count : ℕ := 10
  let initial_length : ℕ := 4
  let order_matters : Bool := true
  let allow_repetition : Bool := true
  (letter_count ^ initial_length : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_four_letter_initials_count_l1923_192379


namespace NUMINAMATH_CALUDE_greatest_common_divisor_450_90_under_60_l1923_192304

theorem greatest_common_divisor_450_90_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 450 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧ 
  ∀ (m : ℕ), m ∣ 450 ∧ m < 60 ∧ m ∣ 90 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_450_90_under_60_l1923_192304


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l1923_192337

/-- Proposition p: -x^2 + 8x + 20 ≥ 0 -/
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

/-- Proposition q: x^2 + 2x + 1 - 4m^2 ≤ 0 -/
def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - 4*m^2 ≤ 0

/-- If ¬p is a necessary but not sufficient condition for ¬q when m > 0, then m ≥ 11/2 -/
theorem not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2 :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, (¬q x m → ¬p x) ∧ (∃ x : ℝ, ¬p x ∧ q x m)) →
  m ≥ 11/2 :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l1923_192337


namespace NUMINAMATH_CALUDE_equation_solution_l1923_192392

theorem equation_solution (x : ℝ) :
  (1 : ℝ) = 1 / (4 * x^2 + 2 * x + 1) →
  x = 0 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1923_192392


namespace NUMINAMATH_CALUDE_equation_solution_l1923_192394

theorem equation_solution :
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = 7 / 9 ↔ y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1923_192394


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1923_192378

theorem regular_polygon_sides (n₁ n₂ : ℕ) : 
  n₁ % 2 = 0 → 
  n₂ % 2 = 0 → 
  (n₁ - 2) * 180 + (n₂ - 2) * 180 = 1800 → 
  ((n₁ = 4 ∧ n₂ = 10) ∨ (n₁ = 10 ∧ n₂ = 4) ∨ (n₁ = 6 ∧ n₂ = 8) ∨ (n₁ = 8 ∧ n₂ = 6)) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1923_192378


namespace NUMINAMATH_CALUDE_money_distribution_l1923_192316

theorem money_distribution (total : ℝ) (share_d : ℝ) :
  let proportion_sum := 5 + 2 + 4 + 3
  let proportion_d := 3
  let proportion_c := 4
  share_d = 1500 →
  share_d = (proportion_d / proportion_sum) * total →
  let share_c := (proportion_c / proportion_sum) * total
  share_c - share_d = 500 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l1923_192316


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1923_192365

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1923_192365


namespace NUMINAMATH_CALUDE_odd_function_theorem_l1923_192391

/-- A function f: ℝ → ℝ is odd if f(x) = -f(-x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- The main theorem: if f is odd and satisfies the given functional equation,
    then f is the zero function -/
theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_eq : ∀ x y, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2) : 
    ∀ x, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_theorem_l1923_192391


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1923_192345

theorem unique_triple_solution : 
  ∃! (a b c : ℤ), (|a - b| + c = 23) ∧ (a^2 - b*c = 119) :=
sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1923_192345


namespace NUMINAMATH_CALUDE_ab_values_l1923_192388

theorem ab_values (a b : ℝ) (h : a^2*b^2 + a^2 + b^2 + 1 = 4*a*b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_ab_values_l1923_192388


namespace NUMINAMATH_CALUDE_mike_ride_distance_l1923_192393

-- Define the taxi fare structure for each route
structure TaxiFare :=
  (initial_fee : ℚ)
  (per_mile_rate : ℚ)
  (extra_fee : ℚ)

-- Define the routes
def route_a : TaxiFare := ⟨2.5, 0.25, 3⟩
def route_b : TaxiFare := ⟨2.5, 0.3, 4⟩
def route_c : TaxiFare := ⟨2.5, 0.25, 9⟩ -- Combined bridge toll and traffic surcharge

-- Calculate the fare for a given route and distance
def calculate_fare (route : TaxiFare) (miles : ℚ) : ℚ :=
  route.initial_fee + route.per_mile_rate * miles + route.extra_fee

-- Theorem statement
theorem mike_ride_distance :
  let annie_miles : ℚ := 14
  let annie_fare := calculate_fare route_c annie_miles
  ∃ (mike_miles : ℚ), 
    (calculate_fare route_a mike_miles = annie_fare) ∧
    (mike_miles = 38) :=
by
  sorry


end NUMINAMATH_CALUDE_mike_ride_distance_l1923_192393


namespace NUMINAMATH_CALUDE_first_week_cases_l1923_192327

/-- Given the number of coronavirus cases in New York over three weeks,
    prove that the number of cases in the first week was 3750. -/
theorem first_week_cases (first_week : ℕ) : 
  (first_week + first_week / 2 + (first_week / 2 + 2000) = 9500) → 
  first_week = 3750 := by
  sorry

end NUMINAMATH_CALUDE_first_week_cases_l1923_192327


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l1923_192399

theorem jenny_easter_eggs (n : ℕ) : 
  n ∣ 30 ∧ n ∣ 45 ∧ n ≥ 5 → n ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l1923_192399


namespace NUMINAMATH_CALUDE_complex_sum_powers_l1923_192307

theorem complex_sum_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 451.625 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l1923_192307


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l1923_192371

-- Define the polynomial Q(x)
def Q (a b c d e f : ℝ) (x : ℝ) : ℝ :=
  x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

-- Define the property of having six distinct x-intercepts
def has_six_distinct_intercepts (a b c d e f : ℝ) : Prop :=
  ∃ p q r s t : ℝ, 
    (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ (p ≠ 0) ∧
    (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ (q ≠ 0) ∧
    (r ≠ s) ∧ (r ≠ t) ∧ (r ≠ 0) ∧
    (s ≠ t) ∧ (s ≠ 0) ∧
    (t ≠ 0) ∧
    (Q a b c d e f p = 0) ∧ (Q a b c d e f q = 0) ∧ 
    (Q a b c d e f r = 0) ∧ (Q a b c d e f s = 0) ∧ 
    (Q a b c d e f t = 0) ∧ (Q a b c d e f 0 = 0)

-- Theorem statement
theorem coefficient_d_nonzero 
  (a b c d e f : ℝ) 
  (h : has_six_distinct_intercepts a b c d e f) : 
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l1923_192371


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l1923_192386

theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l1923_192386


namespace NUMINAMATH_CALUDE_smallest_angle_greater_than_36_degrees_l1923_192310

/-- Represents the angles of a convex pentagon in arithmetic progression -/
structure ConvexPentagonAngles where
  α : ℝ  -- smallest angle
  γ : ℝ  -- common difference
  convex : α > 0 ∧ γ ≥ 0 ∧ α + 4*γ < π  -- convexity condition
  sum : α + (α + γ) + (α + 2*γ) + (α + 3*γ) + (α + 4*γ) = 3*π  -- sum of angles

/-- 
Theorem: In a convex pentagon with angles in arithmetic progression, 
the smallest angle is greater than π/5 radians (36 degrees).
-/
theorem smallest_angle_greater_than_36_degrees (p : ConvexPentagonAngles) : 
  p.α > π/5 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_greater_than_36_degrees_l1923_192310


namespace NUMINAMATH_CALUDE_probability_heart_spade_queen_value_l1923_192328

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (valid : cards.card = 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
| Hearts | Spades | Diamonds | Clubs

/-- Represents a rank in a deck of cards -/
inductive Rank
| Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

/-- A function to check if a card is a heart -/
def is_heart (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a spade -/
def is_spade (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a queen -/
def is_queen (card : Nat × Nat) : Prop := sorry

/-- The probability of drawing a heart first, a spade second, and a queen third -/
def probability_heart_spade_queen (d : Deck) : ℚ := sorry

/-- Theorem stating the probability of drawing a heart first, a spade second, and a queen third -/
theorem probability_heart_spade_queen_value (d : Deck) : 
  probability_heart_spade_queen d = 221 / 44200 := by sorry

end NUMINAMATH_CALUDE_probability_heart_spade_queen_value_l1923_192328


namespace NUMINAMATH_CALUDE_area_G1G2G3_l1923_192319

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point P inside triangle ABC
variable (P : ℝ × ℝ)

-- Define G1, G2, G3 as centroids of triangles PBC, PCA, PAB respectively
def G1 : ℝ × ℝ := sorry
def G2 : ℝ × ℝ := sorry
def G3 : ℝ × ℝ := sorry

-- Define the area function
def area (a b c : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_G1G2G3 (h : area A B C = 24) :
  area G1 G2 G3 = 8/3 := by sorry

end NUMINAMATH_CALUDE_area_G1G2G3_l1923_192319


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l1923_192301

def is_target (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 18 = 6

theorem greatest_integer_with_gcd_six :
  ∃ (m : ℕ), is_target m ∧ ∀ (k : ℕ), is_target k → k ≤ m :=
by
  use 144
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l1923_192301


namespace NUMINAMATH_CALUDE_range_of_cosine_composition_l1923_192359

theorem range_of_cosine_composition (x : ℝ) :
  0.5 ≤ Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ∧
  Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_cosine_composition_l1923_192359


namespace NUMINAMATH_CALUDE_correct_operation_l1923_192339

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - 3 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1923_192339


namespace NUMINAMATH_CALUDE_inequality_relationship_l1923_192362

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def monotone_decreasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → y ≤ 1 → f y < f x

theorem inequality_relationship (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_period_two f) 
  (h3 : monotone_decreasing_on_unit_interval f) : 
  f (-1) < f 2.5 ∧ f 2.5 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l1923_192362


namespace NUMINAMATH_CALUDE_smallest_special_number_l1923_192313

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ (m : ℕ), 
    (100 ≤ m ∧ m < n) →
    (¬(∃ (k : ℕ), m = 2 * k) ∨
     ¬(∃ (k : ℕ), m + 1 = 3 * k) ∨
     ¬(∃ (k : ℕ), m + 2 = 4 * k) ∨
     ¬(∃ (k : ℕ), m + 3 = 5 * k) ∨
     ¬(∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1923_192313


namespace NUMINAMATH_CALUDE_spring_mass_for_27cm_unique_mass_for_27cm_l1923_192325

-- Define the relationship between spring length and mass
def spring_length (mass : ℝ) : ℝ := 16 + 2 * mass

-- Theorem stating that when the spring length is 27 cm, the mass is 5.5 kg
theorem spring_mass_for_27cm : spring_length 5.5 = 27 := by
  sorry

-- Theorem stating the uniqueness of the solution
theorem unique_mass_for_27cm (mass : ℝ) : 
  spring_length mass = 27 → mass = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_mass_for_27cm_unique_mass_for_27cm_l1923_192325


namespace NUMINAMATH_CALUDE_remainder_problem_l1923_192380

theorem remainder_problem (N : ℤ) : N % 221 = 43 → N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1923_192380


namespace NUMINAMATH_CALUDE_cylinder_volume_constant_l1923_192364

/-- Given a cube with side length 3 and a cylinder with the same surface area,
    if the volume of the cylinder is (M * sqrt(6)) / sqrt(π),
    then M = 9 * sqrt(6) * π -/
theorem cylinder_volume_constant (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  ∃ (r h : ℝ),
    (2 * π * r^2 + 2 * π * r * h = cube_surface_area) ∧ 
    (π * r^2 * h = (M * Real.sqrt 6) / Real.sqrt π) →
    M = 9 * Real.sqrt 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_constant_l1923_192364


namespace NUMINAMATH_CALUDE_track_distance_proof_l1923_192351

/-- The distance Albert needs to run in total, in meters. -/
def total_distance : ℝ := 99

/-- The number of laps Albert has already completed. -/
def completed_laps : ℕ := 6

/-- The number of additional laps Albert will run. -/
def additional_laps : ℕ := 5

/-- The distance around the track, in meters. -/
def track_distance : ℝ := 9

theorem track_distance_proof : 
  (completed_laps + additional_laps : ℝ) * track_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_track_distance_proof_l1923_192351


namespace NUMINAMATH_CALUDE_gcd_problem_l1923_192372

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1923_192372


namespace NUMINAMATH_CALUDE_cattle_train_speed_calculation_l1923_192312

/-- The speed of the cattle train in miles per hour -/
def cattle_train_speed : ℝ := 93.33333333333333

/-- The speed of the diesel train in miles per hour -/
def diesel_train_speed (x : ℝ) : ℝ := x - 33

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 6

/-- The travel time of the diesel train in hours -/
def diesel_travel_time : ℝ := 12

/-- The total distance between the trains after the diesel train's travel -/
def total_distance : ℝ := 1284

theorem cattle_train_speed_calculation :
  time_difference * cattle_train_speed +
  diesel_travel_time * cattle_train_speed +
  diesel_travel_time * (diesel_train_speed cattle_train_speed) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_cattle_train_speed_calculation_l1923_192312
