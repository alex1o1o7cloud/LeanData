import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l1807_180704

theorem unique_solution : ∃! (x : ℝ), x ≥ 0 ∧ x + 10 * Real.sqrt x = 39 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1807_180704


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1807_180790

/-- Hyperbola with given asymptotes and passing point -/
structure Hyperbola where
  -- Asymptotes are y = ±√2x
  asymptote_slope : ℝ
  asymptote_slope_sq : asymptote_slope^2 = 2
  -- Passes through (3, -2√3)
  passes_through : (3 : ℝ)^2 / 3 - (-2 * Real.sqrt 3)^2 / 6 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 / 6 = 1

/-- Point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola_equation h x y

/-- Foci of the hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Angle between foci and point on hyperbola -/
def angle_F₁PF₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the angle

/-- Area of triangle formed by foci and point on hyperbola -/
def area_PF₁F₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the area

/-- Main theorem -/
theorem hyperbola_theorem (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) :
  hyperbola_equation h p.x p.y ∧
  (angle_F₁PF₂ h f p = π / 3 → area_PF₁F₂ h f p = 6 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1807_180790


namespace NUMINAMATH_CALUDE_goods_train_length_l1807_180723

/-- The length of a goods train passing a man in another train --/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) 
  (passing_time : ℝ) : 
  man_train_speed = 36 →
  goods_train_speed = 50.4 →
  passing_time = 10 →
  (man_train_speed + goods_train_speed) * (1000 / 3600) * passing_time = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l1807_180723


namespace NUMINAMATH_CALUDE_arc_length_calculation_l1807_180748

/-- 
Given an arc with radius π cm and central angle 120°, 
prove that its arc length is (2/3)π² cm.
-/
theorem arc_length_calculation (r : ℝ) (θ_degrees : ℝ) (l : ℝ) : 
  r = π → θ_degrees = 120 → l = (2/3) * π^2 → 
  l = r * (θ_degrees * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l1807_180748


namespace NUMINAMATH_CALUDE_tourist_catch_up_l1807_180746

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

theorem tourist_catch_up 
  (v_bicycle : ℝ) 
  (v_motorcycle : ℝ) 
  (initial_ride_time : ℝ) 
  (break_time : ℝ) 
  (delay_time : ℝ) :
  v_bicycle = 16 →
  v_motorcycle = 56 →
  initial_ride_time = 1.5 →
  break_time = 1.5 →
  delay_time = 4 →
  ∃ t : ℝ, 
    t > 0 ∧ 
    v_bicycle * (initial_ride_time + t) = 
    v_motorcycle * t + v_bicycle * initial_ride_time ∧
    v_bicycle * (initial_ride_time + t) = catch_up_distance :=
by sorry

end NUMINAMATH_CALUDE_tourist_catch_up_l1807_180746


namespace NUMINAMATH_CALUDE_relationship_abc_l1807_180783

theorem relationship_abc :
  let a : ℤ := -2 * 3^2
  let b : ℤ := (-2 * 3)^2
  let c : ℤ := -(2 * 3)^2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1807_180783


namespace NUMINAMATH_CALUDE_count_equations_l1807_180706

-- Define a function to check if an expression is an equation
def is_equation (expr : String) : Bool :=
  match expr with
  | "5 + 3 = 8" => false
  | "a = 0" => true
  | "y^2 - 2y" => false
  | "x - 3 = 8" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["5 + 3 = 8", "a = 0", "y^2 - 2y", "x - 3 = 8"]

-- Theorem to prove
theorem count_equations :
  (expressions.filter is_equation).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_equations_l1807_180706


namespace NUMINAMATH_CALUDE_least_number_divisible_l1807_180747

theorem least_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((1076 + m) % 23 = 0 ∧ (1076 + m) % 29 = 0 ∧ (1076 + m) % 31 = 0)) ∧ 
  ((1076 + n) % 23 = 0 ∧ (1076 + n) % 29 = 0 ∧ (1076 + n) % 31 = 0) → 
  n = 19601 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisible_l1807_180747


namespace NUMINAMATH_CALUDE_distance_between_blue_lights_l1807_180756

/-- Represents the sequence of lights -/
inductive Light
| Blue
| Yellow

/-- The pattern of lights -/
def light_pattern : List Light := [Light.Blue, Light.Blue, Light.Yellow, Light.Yellow, Light.Yellow]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth blue light -/
def blue_light_position (n : ℕ) : ℕ :=
  sorry

/-- Calculates the distance between two positions in feet -/
def distance_in_feet (pos1 pos2 : ℕ) : ℚ :=
  sorry

theorem distance_between_blue_lights :
  distance_in_feet (blue_light_position 4) (blue_light_position 26) = 100/3 :=
sorry

end NUMINAMATH_CALUDE_distance_between_blue_lights_l1807_180756


namespace NUMINAMATH_CALUDE_perpendicular_travel_time_l1807_180739

theorem perpendicular_travel_time 
  (adam_speed : ℝ) 
  (simon_speed : ℝ) 
  (distance : ℝ) 
  (h1 : adam_speed = 10)
  (h2 : simon_speed = 5)
  (h3 : distance = 75) :
  ∃ (time : ℝ), 
    time = 3 * Real.sqrt 5 ∧ 
    distance^2 = (adam_speed * time)^2 + (simon_speed * time)^2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_travel_time_l1807_180739


namespace NUMINAMATH_CALUDE_chair_cost_l1807_180763

theorem chair_cost (total_spent : ℕ) (num_chairs : ℕ) (cost_per_chair : ℚ)
  (h1 : total_spent = 180)
  (h2 : num_chairs = 12)
  (h3 : (cost_per_chair : ℚ) * (num_chairs : ℚ) = total_spent) :
  cost_per_chair = 15 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l1807_180763


namespace NUMINAMATH_CALUDE_neg_two_star_neg_one_l1807_180743

/-- Custom binary operation ※ -/
def star (a b : ℤ) : ℤ := b^2 - a*b

/-- Theorem stating that (-2) ※ (-1) = -1 -/
theorem neg_two_star_neg_one : star (-2) (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_neg_two_star_neg_one_l1807_180743


namespace NUMINAMATH_CALUDE_inequality_proof_l1807_180762

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ c + a ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ¬(∀ x y z : ℝ, x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) →
    x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1807_180762


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1807_180773

theorem rational_equation_solution (x : ℚ) : 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 2*x - 24) → x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1807_180773


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1807_180789

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 4 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1807_180789


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1807_180749

def f (x : ℝ) : ℝ := x^2 * (x + 1)

theorem f_derivative_at_negative_one :
  (deriv f) (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1807_180749


namespace NUMINAMATH_CALUDE_food_box_shipment_l1807_180794

theorem food_box_shipment (total_food : ℝ) (max_shipping_weight : ℝ) :
  total_food = 777.5 ∧ max_shipping_weight = 2 →
  ⌊total_food / max_shipping_weight⌋ = 388 := by
  sorry

end NUMINAMATH_CALUDE_food_box_shipment_l1807_180794


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l1807_180709

theorem root_difference_quadratic (x : ℝ) : 
  let eq := fun x : ℝ => x^2 + 42*x + 360 + 49
  let roots := {r : ℝ | eq r = 0}
  let diff := fun (a b : ℝ) => |a - b|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ diff r₁ r₂ = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l1807_180709


namespace NUMINAMATH_CALUDE_triangle_properties_l1807_180750

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 36) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 = 54) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1807_180750


namespace NUMINAMATH_CALUDE_bridge_length_is_954_l1807_180726

/-- The length of a bridge given train parameters -/
def bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The length of the bridge is 954 meters -/
theorem bridge_length_is_954 :
  bridge_length 90 36 29 = 954 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_954_l1807_180726


namespace NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l1807_180770

/-- A regular polygon with an exterior angle of 45° has 8 sides. -/
theorem regular_polygon_with_45_degree_exterior_angle_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 45 →
    (360 : ℝ) / exterior_angle = n →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l1807_180770


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_37_l1807_180738

theorem three_digit_divisibility_by_37 (A B C : ℕ) (h_three_digit : 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000) (h_divisible : (100 * A + 10 * B + C) % 37 = 0) :
  ∃ M : ℕ, M = 100 * B + 10 * C + A ∧ 100 ≤ M ∧ M < 1000 ∧ M % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_37_l1807_180738


namespace NUMINAMATH_CALUDE_prob_four_blue_exact_l1807_180702

-- Define the number of blue pens, red pens, and total draws
def blue_pens : ℕ := 5
def red_pens : ℕ := 4
def total_draws : ℕ := 7

-- Define the probability of picking a blue pen in a single draw
def prob_blue : ℚ := blue_pens / (blue_pens + red_pens)

-- Define the probability of picking a red pen in a single draw
def prob_red : ℚ := red_pens / (blue_pens + red_pens)

-- Define the number of ways to choose 4 blue pens out of 7 draws
def ways_to_choose : ℕ := Nat.choose total_draws 4

-- Define the probability of picking exactly 4 blue pens in 7 draws
def prob_four_blue : ℚ := ways_to_choose * (prob_blue ^ 4 * prob_red ^ 3)

-- Theorem statement
theorem prob_four_blue_exact :
  prob_four_blue = 35 * 40000 / 4782969 := by sorry

end NUMINAMATH_CALUDE_prob_four_blue_exact_l1807_180702


namespace NUMINAMATH_CALUDE_complementary_events_l1807_180717

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def AtLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 < 2 ∨ outcome.2 < 2

def BothRed (outcome : SampleSpace) : Prop :=
  outcome.1 ≥ 2 ∧ outcome.2 ≥ 2

-- Theorem statement
theorem complementary_events :
  ∀ (outcome : SampleSpace), AtLeastOneBlack outcome ↔ ¬BothRed outcome := by
  sorry

end NUMINAMATH_CALUDE_complementary_events_l1807_180717


namespace NUMINAMATH_CALUDE_conference_center_occupancy_l1807_180736

theorem conference_center_occupancy (rooms : Nat) (capacity : Nat) (occupancy_ratio : Rat) : 
  rooms = 12 →
  capacity = 150 →
  occupancy_ratio = 5/7 →
  (rooms * capacity * occupancy_ratio).floor = 1285 := by
  sorry

end NUMINAMATH_CALUDE_conference_center_occupancy_l1807_180736


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1807_180708

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (M : ℝ), M = 4/9 ∧ ∀ (a b : ℝ), Real.sin a + Real.sin b = 1/3 →
  Real.sin b - Real.cos a ^ 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1807_180708


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l1807_180707

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [true, true, false, true, false]) = [3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l1807_180707


namespace NUMINAMATH_CALUDE_proposition_logic_l1807_180752

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 ≥ 3)) (hq : q ↔ (3 > 4)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l1807_180752


namespace NUMINAMATH_CALUDE_barbed_wire_rate_proof_l1807_180760

-- Define the given constants
def field_area : ℝ := 3136
def gate_width : ℝ := 1
def num_gates : ℕ := 2
def total_cost : ℝ := 799.20

-- Define the theorem
theorem barbed_wire_rate_proof :
  let side_length : ℝ := Real.sqrt field_area
  let perimeter : ℝ := 4 * side_length
  let wire_length : ℝ := perimeter - (↑num_gates * gate_width)
  let rate_per_meter : ℝ := total_cost / wire_length
  rate_per_meter = 3.60 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_proof_l1807_180760


namespace NUMINAMATH_CALUDE_gecko_sale_price_l1807_180721

/-- The amount Brandon sold the geckos for -/
def brandon_sale_price : ℝ := 100

/-- The pet store's selling price -/
def pet_store_price (x : ℝ) : ℝ := 3 * x + 5

/-- The pet store's profit -/
def pet_store_profit : ℝ := 205

theorem gecko_sale_price :
  pet_store_price brandon_sale_price - brandon_sale_price = pet_store_profit :=
by sorry

end NUMINAMATH_CALUDE_gecko_sale_price_l1807_180721


namespace NUMINAMATH_CALUDE_parallelogram_area_l1807_180781

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1807_180781


namespace NUMINAMATH_CALUDE_least_n_radios_l1807_180705

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
def RadioProblem (n d : ℕ) : Prop :=
  d > 0 ∧  -- d is a positive integer
  (2 * d + (n - 4) * (d + 10 * n)) = n * (d + 100)  -- profit equation

/-- The least possible value of n that satisfies the RadioProblem. -/
theorem least_n_radios : 
  ∀ n d, RadioProblem n d → n ≥ 14 :=
sorry

end NUMINAMATH_CALUDE_least_n_radios_l1807_180705


namespace NUMINAMATH_CALUDE_probability_three_common_books_l1807_180761

def total_books : ℕ := 12
def books_to_select : ℕ := 4

theorem probability_three_common_books :
  (Nat.choose total_books 3 * Nat.choose (total_books - 3) 1 * Nat.choose (total_books - 4) 1) /
  (Nat.choose total_books books_to_select * Nat.choose total_books books_to_select) =
  32 / 495 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l1807_180761


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l1807_180776

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l1807_180776


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l1807_180713

theorem salary_decrease_percentage (x : ℝ) : 
  (100 - x) / 100 * 130 / 100 = 65 / 100 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l1807_180713


namespace NUMINAMATH_CALUDE_inverse_power_of_three_l1807_180744

theorem inverse_power_of_three : 3⁻¹ = (1 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_inverse_power_of_three_l1807_180744


namespace NUMINAMATH_CALUDE_carlo_thursday_practice_l1807_180716

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  thursday : ℕ  -- Minutes practiced on Thursday
  wednesday : ℕ := thursday + 5  -- Minutes practiced on Wednesday
  tuesday : ℕ := wednesday - 10  -- Minutes practiced on Tuesday
  monday : ℕ := 2 * tuesday  -- Minutes practiced on Monday
  friday : ℕ := 60  -- Minutes practiced on Friday

/-- Calculates the total practice time for the week --/
def totalPracticeTime (schedule : PracticeSchedule) : ℕ :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Theorem stating that Carlo practiced for 50 minutes on Thursday --/
theorem carlo_thursday_practice :
  ∃ (schedule : PracticeSchedule), totalPracticeTime schedule = 300 ∧ schedule.thursday = 50 := by
  sorry

end NUMINAMATH_CALUDE_carlo_thursday_practice_l1807_180716


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1807_180791

theorem geometric_sequence_product (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ = 8/3 ∧ a₅ = 27/2 ∧ 
  (∃ q : ℝ, q ≠ 0 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q) →
  |a₂ * a₃ * a₄| = 216 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1807_180791


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l1807_180786

theorem base_2_representation_of_125 :
  ∃ (a b c d e f g : Nat),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    125 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l1807_180786


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_bound_l1807_180741

theorem quadratic_inequality_implies_m_bound (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m ≤ 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_bound_l1807_180741


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1807_180771

/-- Given that i² = -1, prove that (3 - 2i) / (4 + 5i) = 2/41 - 23/41 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (3 - 2*i) / (4 + 5*i) = 2/41 - 23/41 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1807_180771


namespace NUMINAMATH_CALUDE_ramu_car_price_l1807_180714

/-- Proves that given the conditions of Ramu's car purchase, repair, and sale,
    the original price he paid for the car is 42000 rupees. -/
theorem ramu_car_price :
  let repair_cost : ℝ := 12000
  let selling_price : ℝ := 64900
  let profit_percent : ℝ := 20.185185185185187
  let original_price : ℝ := 42000
  (selling_price = original_price + repair_cost + (original_price + repair_cost) * (profit_percent / 100)) →
  original_price = 42000 :=
by
  sorry

#check ramu_car_price

end NUMINAMATH_CALUDE_ramu_car_price_l1807_180714


namespace NUMINAMATH_CALUDE_height_to_ad_l1807_180798

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- AB length
  ab : ℝ
  -- BC length
  bc : ℝ
  -- Height dropped to CD
  height_cd : ℝ
  -- Parallelogram property
  is_parallelogram : ab > 0 ∧ bc > 0 ∧ height_cd > 0

/-- Theorem: In a parallelogram ABCD where AB = 6, BC = 8, and the height dropped to CD is 4,
    the height dropped to AD is 3 -/
theorem height_to_ad (p : Parallelogram) 
    (h_ab : p.ab = 6)
    (h_bc : p.bc = 8)
    (h_height_cd : p.height_cd = 4) :
  ∃ (height_ad : ℝ), height_ad = 3 ∧ p.ab * p.height_cd = p.bc * height_ad :=
by sorry

end NUMINAMATH_CALUDE_height_to_ad_l1807_180798


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1807_180725

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition : ∃ x > 2, x + 4 / (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1807_180725


namespace NUMINAMATH_CALUDE_max_full_books_read_l1807_180722

def pages_per_hour : ℕ := 120
def pages_per_book : ℕ := 360
def available_hours : ℕ := 8

def books_read : ℕ := available_hours * pages_per_hour / pages_per_book

theorem max_full_books_read :
  books_read = 2 :=
sorry

end NUMINAMATH_CALUDE_max_full_books_read_l1807_180722


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1807_180731

theorem simplify_sqrt_expression (y : ℝ) (h : y ≥ 5/2) :
  Real.sqrt (y + 2 + 3 * Real.sqrt (2 * y - 5)) - Real.sqrt (y - 2 + Real.sqrt (2 * y - 5)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1807_180731


namespace NUMINAMATH_CALUDE_simplify_fraction_l1807_180799

theorem simplify_fraction (a : ℚ) (h : a = 2) : 15 * a^4 / (45 * a^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1807_180799


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l1807_180712

/-- A right triangle with sides 9, 12, and 15 units -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_right_triangle : side_a^2 + side_b^2 = side_c^2
  side_a_eq : side_a = 9
  side_b_eq : side_b = 12
  side_c_eq : side_c = 15

/-- A circle with radius 2 units -/
def circle_radius : ℝ := 2

/-- The inner triangle formed by the path of the circle's center -/
def inner_triangle (t : RightTriangle) : ℝ × ℝ × ℝ :=
  (t.side_a - 2 * circle_radius, t.side_b - 2 * circle_radius, t.side_c - 2 * circle_radius)

/-- Theorem: The perimeter of the inner triangle is 24 units -/
theorem inner_triangle_perimeter (t : RightTriangle) :
  let (a, b, c) := inner_triangle t
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l1807_180712


namespace NUMINAMATH_CALUDE_expression_simplification_l1807_180740

theorem expression_simplification (x y z : ℝ) : 
  (x + (y - z)) - ((x + z) - y) = 2 * y - 2 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1807_180740


namespace NUMINAMATH_CALUDE_sum_of_integers_l1807_180745

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 4)
  (eq2 : q - r + s = 5)
  (eq3 : r - s + p = 7)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1807_180745


namespace NUMINAMATH_CALUDE_fly_probabilities_l1807_180742

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def fly_probability (n m : ℕ) : ℚ :=
  (binomial (n + m) n : ℚ) / (2 ^ (n + m))

def fly_probability_through_segment (n1 m1 n2 m2 : ℕ) : ℚ :=
  ((binomial (n1 + m1) n1 : ℚ) * (binomial (n2 + m2) n2)) / (2 ^ (n1 + m1 + n2 + m2 + 1))

def fly_probability_through_circle (n m r : ℕ) : ℚ :=
  let total_steps := n + m
  let mid_steps := total_steps / 2
  (2 * (binomial mid_steps 2 : ℚ) * (binomial mid_steps (mid_steps - 2)) +
   2 * (binomial mid_steps 3 : ℚ) * (binomial mid_steps (mid_steps - 3)) +
   (binomial mid_steps 4 : ℚ) * (binomial mid_steps (mid_steps - 4))) /
  (2 ^ total_steps)

theorem fly_probabilities :
  fly_probability 8 10 = (binomial 18 8 : ℚ) / (2^18) ∧
  fly_probability_through_segment 5 6 2 4 = ((binomial 11 5 : ℚ) * (binomial 6 2)) / (2^18) ∧
  fly_probability_through_circle 8 10 3 = 
    (2 * (binomial 9 2 : ℚ) * (binomial 9 6) + 
     2 * (binomial 9 3 : ℚ) * (binomial 9 5) + 
     (binomial 9 4 : ℚ) * (binomial 9 4)) / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l1807_180742


namespace NUMINAMATH_CALUDE_fair_coin_three_tosses_one_head_l1807_180711

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a fair coin tossed 3 times, the probability
    of getting exactly 1 head and 2 tails is 3/8. -/
theorem fair_coin_three_tosses_one_head (p : ℝ) (h : fair_coin p) :
  binomial_probability 3 1 p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_tosses_one_head_l1807_180711


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1807_180774

theorem quadratic_solution_implies_value (a : ℝ) :
  (2^2 - 3*2 + a = 0) → (2*a - 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1807_180774


namespace NUMINAMATH_CALUDE_problem_solution_l1807_180727

theorem problem_solution (P Q R : ℚ) : 
  (5 / 8 = P / 56) → 
  (5 / 8 = 80 / Q) → 
  (R = P - 4) → 
  (Q + R = 159) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1807_180727


namespace NUMINAMATH_CALUDE_total_crayons_is_18_l1807_180777

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 3

/-- The number of children -/
def number_of_children : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := crayons_per_child * number_of_children

theorem total_crayons_is_18 : total_crayons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_is_18_l1807_180777


namespace NUMINAMATH_CALUDE_student_allocation_arrangements_l1807_180780

theorem student_allocation_arrangements : 
  let n : ℕ := 4  -- number of students
  let m : ℕ := 3  -- number of locations
  let arrangements := {f : Fin n → Fin m | ∀ i : Fin m, ∃ j : Fin n, f j = i}
  Fintype.card arrangements = 36 := by
sorry

end NUMINAMATH_CALUDE_student_allocation_arrangements_l1807_180780


namespace NUMINAMATH_CALUDE_odd_sum_probability_l1807_180734

theorem odd_sum_probability (n : Nat) (h : n = 16) :
  let grid_size := 4
  let total_arrangements := n.factorial
  let valid_arrangements := (grid_size.choose 2) * (n / 2).factorial * (n / 2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 2150 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l1807_180734


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1807_180779

/-- If ax^2 + 28x + 9 is the square of a binomial, then a = 196/9 -/
theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, a * x^2 + 28 * x + 9 = (p * x + q)^2) → 
  a = 196 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1807_180779


namespace NUMINAMATH_CALUDE_f_monotone_and_bounded_l1807_180766

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 - a * Real.sin x - 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x - a * Real.cos x

theorem f_monotone_and_bounded (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ M : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f_deriv a x| ≤ M) →
    ∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f a x| ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_bounded_l1807_180766


namespace NUMINAMATH_CALUDE_annie_brownies_l1807_180782

def brownies_problem (total : ℕ) : Prop :=
  let after_admin : ℕ := total / 2
  let after_carl : ℕ := after_admin / 2
  let after_simon : ℕ := after_carl - 2
  (after_simon = 3) ∧ (total > 0)

theorem annie_brownies : ∃ (total : ℕ), brownies_problem total ∧ total = 20 := by
  sorry

end NUMINAMATH_CALUDE_annie_brownies_l1807_180782


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1807_180795

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) ∧ 
  ∀ m : ℕ, (m : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1807_180795


namespace NUMINAMATH_CALUDE_newsstand_profit_optimization_l1807_180754

/-- Newsstand profit optimization problem -/
theorem newsstand_profit_optimization :
  let buying_price : ℚ := 60 / 100
  let selling_price : ℚ := 1
  let return_price : ℚ := 10 / 100
  let high_demand_days : ℕ := 20
  let low_demand_days : ℕ := 10
  let high_demand_sales : ℕ := 400
  let low_demand_sales : ℕ := 250
  let total_days : ℕ := high_demand_days + low_demand_days
  let profit_per_sold (x : ℕ) : ℚ := 
    (high_demand_days * (min x high_demand_sales) + 
     low_demand_days * (min x low_demand_sales)) * (selling_price - buying_price)
  let loss_per_unsold (x : ℕ) : ℚ := 
    (high_demand_days * (x - min x high_demand_sales) + 
     low_demand_days * (x - min x low_demand_sales)) * (buying_price - return_price)
  let total_profit (x : ℕ) : ℚ := profit_per_sold x - loss_per_unsold x
  ∀ x : ℕ, total_profit x ≤ total_profit high_demand_sales ∧ 
           total_profit high_demand_sales = 2450 / 100 := by
  sorry

end NUMINAMATH_CALUDE_newsstand_profit_optimization_l1807_180754


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1807_180792

theorem common_number_in_overlapping_lists (numbers : List ℝ) : 
  numbers.length = 8 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 5).sum / 3 = 10 →
  numbers.sum / 8 = 8 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 5, x = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1807_180792


namespace NUMINAMATH_CALUDE_division_problem_l1807_180755

theorem division_problem : (62976 : ℕ) / 512 = 123 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1807_180755


namespace NUMINAMATH_CALUDE_bicycle_time_saved_l1807_180793

/-- The time in minutes it takes Mike to walk to school -/
def walking_time : ℕ := 98

/-- The time in minutes Mike saved by riding a bicycle -/
def time_saved : ℕ := 34

/-- Theorem: The time saved by riding a bicycle compared to walking is 34 minutes -/
theorem bicycle_time_saved : time_saved = 34 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_time_saved_l1807_180793


namespace NUMINAMATH_CALUDE_sharks_winning_percentage_l1807_180732

theorem sharks_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (1 + k : ℚ) / (4 + k) < 9 / 10) ∧
  (1 + N : ℚ) / (4 + N) ≥ 9 / 10 →
  N = 26 :=
sorry

end NUMINAMATH_CALUDE_sharks_winning_percentage_l1807_180732


namespace NUMINAMATH_CALUDE_product_inequality_l1807_180784

theorem product_inequality (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 3) 
  (h_x_pos : ∀ i ∈ Finset.range (n - 1), x (i + 2) > 0)
  (h_x_prod : (Finset.range (n - 1)).prod (λ i => x (i + 2)) = 1) :
  (Finset.range (n - 1)).prod (λ i => (1 + x (i + 2)) ^ (i + 2)) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1807_180784


namespace NUMINAMATH_CALUDE_package_cost_l1807_180751

/-- The cost to mail each package, given the total amount spent, cost per letter, number of letters, and relationship between letters and packages. -/
theorem package_cost (total_spent : ℚ) (letter_cost : ℚ) (num_letters : ℕ) 
  (h1 : total_spent = 4.49)
  (h2 : letter_cost = 0.37)
  (h3 : num_letters = 5)
  (h4 : num_letters = num_packages + 2) : 
  (total_spent - num_letters * letter_cost) / (num_letters - 2) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_package_cost_l1807_180751


namespace NUMINAMATH_CALUDE_complex_fraction_imaginary_l1807_180787

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (k : ℝ), (2 + I) / (a - I) = k * I) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_imaginary_l1807_180787


namespace NUMINAMATH_CALUDE_octal_sum_l1807_180703

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of 444₈, 44₈, and 4₈ in base 8 is 514₈ --/
theorem octal_sum : 
  decimal_to_octal (octal_to_decimal 444 + octal_to_decimal 44 + octal_to_decimal 4) = 514 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_l1807_180703


namespace NUMINAMATH_CALUDE_football_players_l1807_180775

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 460)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  total - neither - cricket + both = 325 :=
by sorry

end NUMINAMATH_CALUDE_football_players_l1807_180775


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l1807_180757

theorem subset_implies_m_equals_one (m : ℝ) :
  let A : Set ℝ := {-1, 2, 2*m - 1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l1807_180757


namespace NUMINAMATH_CALUDE_all_propositions_true_l1807_180724

-- Proposition 1
def expanded_terms (a b c d p q r m n : ℕ) : ℕ := 24

-- Proposition 2
def five_digit_numbers : ℕ := 36

-- Proposition 3
def seating_arrangements : ℕ := 24

-- Proposition 4
def odd_coefficients (x : ℝ) : ℕ := 2

theorem all_propositions_true :
  (∀ a b c d p q r m n, expanded_terms a b c d p q r m n = 24) ∧
  (five_digit_numbers = 36) ∧
  (seating_arrangements = 24) ∧
  (∀ x, odd_coefficients x = 2) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1807_180724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1807_180764

/-- 
Given an arithmetic sequence {a_n} where the first three terms are a-1, a-1, and 2a+3,
prove that the general term formula is a_n = 2n-3.
-/
theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℝ) 
  (a : ℝ) 
  (h1 : a_n 1 = a - 1) 
  (h2 : a_n 2 = a - 1) 
  (h3 : a_n 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) :
  ∀ n : ℕ, a_n n = 2*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1807_180764


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1807_180710

structure RightTriangle :=
  (O X Y : ℝ × ℝ)
  (is_right : (X.1 - O.1) * (Y.1 - O.1) + (X.2 - O.2) * (Y.2 - O.2) = 0)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (M_midpoint : M = ((X.1 + O.1) / 2, (X.2 + O.2) / 2))
  (N_midpoint : N = ((Y.1 + O.1) / 2, (Y.2 + O.2) / 2))
  (XN_length : Real.sqrt ((X.1 - N.1)^2 + (X.2 - N.2)^2) = 19)
  (YM_length : Real.sqrt ((Y.1 - M.1)^2 + (Y.2 - M.2)^2) = 22)

theorem right_triangle_hypotenuse (t : RightTriangle) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1807_180710


namespace NUMINAMATH_CALUDE_original_area_l1807_180788

/-- In an oblique dimetric projection, given a regular triangle as the intuitive diagram -/
structure ObliqueTriangle where
  /-- Side length of the intuitive diagram -/
  side_length : ℝ
  /-- Area ratio of original to intuitive -/
  area_ratio : ℝ
  /-- Side length is positive -/
  side_length_pos : 0 < side_length
  /-- Area ratio is positive -/
  area_ratio_pos : 0 < area_ratio

/-- Theorem: Area of the original figure in oblique dimetric projection -/
theorem original_area (t : ObliqueTriangle) (h1 : t.side_length = 2) (h2 : t.area_ratio = 2 * Real.sqrt 2) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_original_area_l1807_180788


namespace NUMINAMATH_CALUDE_expected_total_rain_l1807_180719

/-- Represents the possible rain outcomes for a day --/
inductive RainOutcome
  | NoRain
  | ThreeInches
  | EightInches

/-- Probability of each rain outcome --/
def rainProbability (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0.5
  | RainOutcome.ThreeInches => 0.3
  | RainOutcome.EightInches => 0.2

/-- Amount of rain for each outcome in inches --/
def rainAmount (outcome : RainOutcome) : ℝ :=
  match outcome with
  | RainOutcome.NoRain => 0
  | RainOutcome.ThreeInches => 3
  | RainOutcome.EightInches => 8

/-- Number of days in the forecast --/
def forecastDays : ℕ := 5

/-- Expected value of rain for a single day --/
def dailyExpectedRain : ℝ :=
  (rainProbability RainOutcome.NoRain * rainAmount RainOutcome.NoRain) +
  (rainProbability RainOutcome.ThreeInches * rainAmount RainOutcome.ThreeInches) +
  (rainProbability RainOutcome.EightInches * rainAmount RainOutcome.EightInches)

/-- Theorem: The expected value of the total amount of rain for the forecast period is 12.5 inches --/
theorem expected_total_rain :
  forecastDays * dailyExpectedRain = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_expected_total_rain_l1807_180719


namespace NUMINAMATH_CALUDE_average_tickets_sold_l1807_180765

/-- Given a total sales amount of $960 over 3 days, with each ticket costing $4,
    prove that the average number of tickets sold per day is 80. -/
theorem average_tickets_sold (total_sales : ℕ) (days : ℕ) (ticket_price : ℕ) 
  (h1 : total_sales = 960)
  (h2 : days = 3)
  (h3 : ticket_price = 4) :
  total_sales / (days * ticket_price) = 80 := by
  sorry

#check average_tickets_sold

end NUMINAMATH_CALUDE_average_tickets_sold_l1807_180765


namespace NUMINAMATH_CALUDE_negation_of_negative_square_positive_is_false_l1807_180718

theorem negation_of_negative_square_positive_is_false : 
  ¬(∀ x : ℝ, x < 0 → x^2 > 0) = False := by sorry

end NUMINAMATH_CALUDE_negation_of_negative_square_positive_is_false_l1807_180718


namespace NUMINAMATH_CALUDE_game_result_l1807_180759

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 10
  else if n % 2 = 0 then 5
  else 0

def allieRolls : List ℕ := [2, 3, 6, 4]
def bettyRolls : List ℕ := [2, 1, 5, 6]

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : totalPoints allieRolls + totalPoints bettyRolls = 45 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1807_180759


namespace NUMINAMATH_CALUDE_triangle_property_l1807_180733

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  b = 2 →
  a + c = 4 →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧ 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l1807_180733


namespace NUMINAMATH_CALUDE_hyperbola_C_equation_l1807_180769

/-- A hyperbola passing through a point and sharing asymptotes with another hyperbola -/
def hyperbola_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧ 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧
  (3^2 / a^2 - 2 / b^2 = 1) ∧
  (a^2 / b^2 = 3)

/-- Theorem stating the standard equation of hyperbola C -/
theorem hyperbola_C_equation :
  ∀ x y : ℝ, hyperbola_C x y → (x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_C_equation_l1807_180769


namespace NUMINAMATH_CALUDE_specific_cube_stack_surface_area_l1807_180715

/-- Represents a three-dimensional shape formed by stacking cubes -/
structure CubeStack where
  num_cubes : ℕ
  edge_length : ℝ
  num_layers : ℕ

/-- Calculates the surface area of a cube stack -/
def surface_area (stack : CubeStack) : ℝ :=
  sorry

/-- Theorem stating that a specific cube stack has a surface area of 72 square meters -/
theorem specific_cube_stack_surface_area :
  let stack : CubeStack := {
    num_cubes := 30,
    edge_length := 1,
    num_layers := 4
  }
  surface_area stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_stack_surface_area_l1807_180715


namespace NUMINAMATH_CALUDE_line_equation_l1807_180767

/-- A line passing through (2,3) with opposite-sign intercepts -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2,3)
  passes_through : 3 = m * 2 + b
  -- The line has opposite-sign intercepts
  opposite_intercepts : (b ≠ 0 ∧ (-b/m) * b < 0) ∨ (b = 0 ∧ m ≠ 0)

/-- The equation of the line is either 3x - 2y = 0 or x - y + 1 = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 3/2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = -1) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1807_180767


namespace NUMINAMATH_CALUDE_dog_treats_duration_l1807_180778

theorem dog_treats_duration (treats_per_day : ℕ) (cost_per_treat : ℚ) (total_spent : ℚ) : 
  treats_per_day = 2 → cost_per_treat = 1/10 → total_spent = 6 → 
  (total_spent / cost_per_treat) / treats_per_day = 30 := by
  sorry

end NUMINAMATH_CALUDE_dog_treats_duration_l1807_180778


namespace NUMINAMATH_CALUDE_work_completion_time_l1807_180737

theorem work_completion_time (x : ℝ) : 
  x > 0 → 
  (8 * (1 / x + 1 / 20) = 14 / 15) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1807_180737


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l1807_180730

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (percent_not_covering_politics : ℚ)
  (h2 : percent_not_covering_politics = 1/2)
  (percent_not_covering_local_politics : ℚ)
  (h3 : percent_not_covering_local_politics = 3/10)
  : (↑total_reporters - (percent_not_covering_politics * ↑total_reporters) -
     (percent_not_covering_local_politics * (↑total_reporters - (percent_not_covering_politics * ↑total_reporters))))
    / ↑total_reporters = 7/20 :=
by sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l1807_180730


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coeff_difference_l1807_180701

theorem quadratic_roots_to_coeff_difference (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ (x = -1/2 ∨ x = 1/3)) → 
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coeff_difference_l1807_180701


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1807_180700

theorem triangle_angle_sum (a b c : ℝ) : 
  b = 2 * a →
  c = a - 40 →
  a + b + c = 180 →
  a + c = 70 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1807_180700


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radii_l1807_180785

theorem tetrahedron_sphere_radii (r : ℝ) (R : ℝ) :
  r = Real.sqrt 2 - 1 →
  R = Real.sqrt 6 + 1 →
  ∃ (a : ℝ),
    r = (a * Real.sqrt 2) / 4 ∧
    R = (a * Real.sqrt 6) / 4 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radii_l1807_180785


namespace NUMINAMATH_CALUDE_sweettarts_distribution_l1807_180796

theorem sweettarts_distribution (total_sweettarts : ℕ) (num_friends : ℕ) (sweettarts_per_friend : ℕ) :
  total_sweettarts = 15 →
  num_friends = 3 →
  total_sweettarts = num_friends * sweettarts_per_friend →
  sweettarts_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_sweettarts_distribution_l1807_180796


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1807_180797

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0)
  (pos_q : q > 0)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq_1 : 2 * b = p + c)
  (arith_seq_2 : 2 * c = b + q) :
  (2 * a)^2 - 4 * b * c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1807_180797


namespace NUMINAMATH_CALUDE_find_m_l1807_180753

def U : Set Nat := {1, 2, 3, 4}

def A (m : ℤ) : Set Nat := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℤ, (U \ A m) = {1, 4} ∧ m = 6 := by sorry

end NUMINAMATH_CALUDE_find_m_l1807_180753


namespace NUMINAMATH_CALUDE_graph_translation_l1807_180720

/-- Translating the graph of f(x) = cos(2x - π/3) to the left by π/6 units results in y = cos(2x) -/
theorem graph_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  let h : ℝ → ℝ := λ x => f (x + π / 6)
  h x = g x := by sorry

end NUMINAMATH_CALUDE_graph_translation_l1807_180720


namespace NUMINAMATH_CALUDE_min_value_of_function_l1807_180729

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (y : ℝ), y = x + 1 / (x + 1) ∧
  ∀ (z : ℝ), z > -1 → z + 1 / (z + 1) ≥ x + 1 / (x + 1) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1807_180729


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1807_180772

/-- The point symmetric to P(-2, 1) with respect to the line x + y - 3 = 0 is (3, 4). -/
theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-2, 1)
  let line (x y : ℝ) := x + y - 3 = 0
  let Q : ℝ × ℝ := (3, 4)
  let midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (∀ x y, line x y ↔ line (midpoint P Q).1 (midpoint P Q).2) ∧
  (Q.2 - P.2) / (Q.1 - P.1) = -1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1807_180772


namespace NUMINAMATH_CALUDE_smallest_square_ending_644_l1807_180735

theorem smallest_square_ending_644 :
  ∀ n : ℕ+, n.val < 194 → (n.val ^ 2) % 1000 ≠ 644 ∧ (194 ^ 2) % 1000 = 644 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_ending_644_l1807_180735


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1807_180768

/-- The number of ways to arrange books on a shelf --/
def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial math_books * Nat.factorial english_books

/-- Theorem stating the number of ways to arrange 4 math books, 7 English books, and 1 journal --/
theorem book_arrangement_count :
  arrange_books 4 7 = 725760 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1807_180768


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1807_180728

theorem square_sum_theorem (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90) :
  (x + y)^2 = 130 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1807_180728


namespace NUMINAMATH_CALUDE_randy_initial_money_l1807_180758

/-- Calculates the initial amount of money Randy had in his piggy bank. -/
def initial_money (cost_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) (money_left : ℕ) : ℕ :=
  cost_per_trip * trips_per_month * months + money_left

/-- Proves that Randy started with $200 given the problem conditions. -/
theorem randy_initial_money :
  initial_money 2 4 12 104 = 200 := by
  sorry

end NUMINAMATH_CALUDE_randy_initial_money_l1807_180758
