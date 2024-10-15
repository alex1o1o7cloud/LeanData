import Mathlib

namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3382_338261

theorem complex_exponential_to_rectangular : 
  ∃ (z : ℂ), z = Real.sqrt 2 * Complex.exp (Complex.I * (13 * Real.pi / 6)) ∧ 
             z = (Real.sqrt 6 / 2 : ℂ) + Complex.I * (Real.sqrt 2 / 2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3382_338261


namespace NUMINAMATH_CALUDE_intersection_A_and_naturals_l3382_338203

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_A_and_naturals :
  A ∩ Set.univ.image (Nat.cast : ℕ → ℝ) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_and_naturals_l3382_338203


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_9_l3382_338223

theorem ceiling_neg_sqrt_64_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_9_l3382_338223


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l3382_338230

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for complement of A ∪ B in U
theorem complement_union_A_B : (A ∪ B)ᶜ = {x | x < -3 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for (complement of A in U) ∪ B
theorem complement_A_union_B : Aᶜ ∪ B = {x | x ≤ 2 ∨ 3 ≤ x} ∩ U := by sorry

-- Theorem for A ∩ (complement of B in U)
theorem A_intersection_complement_B : A ∩ Bᶜ = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for (complement of A in U) ∪ (complement of B in U)
theorem complement_A_union_complement_B : Aᶜ ∪ Bᶜ = {x | x ≤ -2 ∨ 2 < x} ∩ U := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_complement_A_union_B_A_intersection_complement_B_complement_A_union_complement_B_l3382_338230


namespace NUMINAMATH_CALUDE_jimmy_shorts_count_l3382_338226

def senior_discount : ℚ := 10 / 100
def shorts_price : ℚ := 15
def shirt_price : ℚ := 17
def num_shirts : ℕ := 5
def total_paid : ℚ := 117

def num_shorts : ℕ := 2

theorem jimmy_shorts_count :
  let shirts_cost := shirt_price * num_shirts
  let discount := shirts_cost * senior_discount
  let irene_total := shirts_cost - discount
  let remaining := total_paid - irene_total
  (remaining / shorts_price).floor = num_shorts := by sorry

end NUMINAMATH_CALUDE_jimmy_shorts_count_l3382_338226


namespace NUMINAMATH_CALUDE_inequality_implies_range_l3382_338246

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l3382_338246


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l3382_338260

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit positive integer -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The theorem stating the largest number in the sequence -/
theorem largest_non_prime_sequence :
  ∃ (n : ℕ), 
    (∀ k : ℕ, k ∈ Finset.range 7 → is_two_digit (n - k)) ∧ 
    (∀ k : ℕ, k ∈ Finset.range 7 → n - k < 50) ∧
    (∀ k : ℕ, k ∈ Finset.range 7 → ¬(is_prime (n - k))) ∧
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l3382_338260


namespace NUMINAMATH_CALUDE_mexican_food_pricing_l3382_338281

/-- Given the pricing conditions for Mexican food items, prove the cost of a specific combination. -/
theorem mexican_food_pricing
  (enchilada_price taco_price burrito_price : ℚ)
  (h1 : 2 * enchilada_price + 3 * taco_price + burrito_price = 5)
  (h2 : 3 * enchilada_price + 2 * taco_price + 2 * burrito_price = 15/2) :
  2 * enchilada_price + 2 * taco_price + 3 * burrito_price = 85/8 := by
  sorry

end NUMINAMATH_CALUDE_mexican_food_pricing_l3382_338281


namespace NUMINAMATH_CALUDE_two_digit_multiplication_trick_l3382_338219

theorem two_digit_multiplication_trick (a b c : ℕ) 
  (h1 : b + c = 10) 
  (h2 : 0 ≤ a ∧ a ≤ 9) 
  (h3 : 0 ≤ b ∧ b ≤ 9) 
  (h4 : 0 ≤ c ∧ c ≤ 9) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_trick_l3382_338219


namespace NUMINAMATH_CALUDE_sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l3382_338205

theorem sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two :
  Real.sqrt 2 - (Real.sqrt 2) / 2 = (Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l3382_338205


namespace NUMINAMATH_CALUDE_divisibility_of_powers_l3382_338277

theorem divisibility_of_powers (a : ℕ) (ha : a > 0) :
  ∃ b : ℕ, b > a ∧ (1 + 2^b + 3^b) % (1 + 2^a + 3^a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_powers_l3382_338277


namespace NUMINAMATH_CALUDE_distance_between_points_l3382_338259

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 18)
  let p2 : ℝ × ℝ := (13, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 269 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3382_338259


namespace NUMINAMATH_CALUDE_bridget_fruits_count_bridget_fruits_proof_l3382_338212

theorem bridget_fruits_count : ℕ → ℕ → Prop :=
  fun apples oranges =>
    apples / oranges = 2 →
    apples / 2 - 3 = 4 →
    oranges - 3 = 5 →
    apples + oranges = 21

theorem bridget_fruits_proof : ∃ a o : ℕ, bridget_fruits_count a o := by
  sorry

end NUMINAMATH_CALUDE_bridget_fruits_count_bridget_fruits_proof_l3382_338212


namespace NUMINAMATH_CALUDE_enrollment_difference_l3382_338282

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : ℕ

/-- Theorem: The positive difference between the maximum and minimum enrollments is 750 --/
theorem enrollment_difference (schools : List School) 
  (h1 : schools.length = 5)
  (h2 : ∃ s ∈ schools, s.name = "Varsity" ∧ s.enrollment = 1680)
  (h3 : ∃ s ∈ schools, s.name = "Northwest" ∧ s.enrollment = 1170)
  (h4 : ∃ s ∈ schools, s.name = "Central" ∧ s.enrollment = 1840)
  (h5 : ∃ s ∈ schools, s.name = "Greenbriar" ∧ s.enrollment = 1090)
  (h6 : ∃ s ∈ schools, s.name = "Eastside" ∧ s.enrollment = 1450) :
  (schools.map (·.enrollment)).maximum?.get! - (schools.map (·.enrollment)).minimum?.get! = 750 := by
  sorry


end NUMINAMATH_CALUDE_enrollment_difference_l3382_338282


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3382_338289

theorem cubic_equation_root (a b : ℚ) : 
  (1 + Real.sqrt 5)^3 + a * (1 + Real.sqrt 5)^2 + b * (1 + Real.sqrt 5) - 60 = 0 → 
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3382_338289


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3382_338210

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (r : ℕ), r < d ∧ (n - r) % d = 0 ∧ ∀ (s : ℕ), s < r → (n - s) % d ≠ 0 := by
  sorry

theorem problem_solution :
  ∃ (r : ℕ), r = 43 ∧ (62575 - r) % 99 = 0 ∧ ∀ (s : ℕ), s < r → (62575 - s) % 99 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3382_338210


namespace NUMINAMATH_CALUDE_leah_peeled_18_l3382_338274

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  leah_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Leah peeled -/
def leah_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_left := scenario.total_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.leah_rate
  let combined_time := potatoes_left / combined_rate
  scenario.leah_rate * combined_time

/-- The theorem stating that Leah peeled 18 potatoes -/
theorem leah_peeled_18 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 50)
  (h2 : scenario.homer_rate = 3)
  (h3 : scenario.leah_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  leah_potatoes scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_leah_peeled_18_l3382_338274


namespace NUMINAMATH_CALUDE_quadrilateral_area_rational_l3382_338225

/-- The area of a quadrilateral with integer coordinates is rational -/
theorem quadrilateral_area_rational
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ) :
  ∃ (q : ℚ), q = |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)| / 2 +
              |x₁ * (y₃ - y₄) + x₃ * (y₄ - y₁) + x₄ * (y₁ - y₃)| / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_rational_l3382_338225


namespace NUMINAMATH_CALUDE_area_is_100_l3382_338276

/-- The area enclosed by the graph of |x| + |2y| = 10 -/
def area_enclosed : ℝ := 100

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs x + abs (2 * y) = 10

/-- The graph is symmetric across the x-axis -/
axiom symmetric_x_axis : ∀ x y : ℝ, graph_equation x y → graph_equation x (-y)

/-- The graph is symmetric across the y-axis -/
axiom symmetric_y_axis : ∀ x y : ℝ, graph_equation x y → graph_equation (-x) y

/-- The graph forms four congruent triangles -/
axiom four_congruent_triangles : ∃ A : ℝ, A > 0 ∧ area_enclosed = 4 * A

/-- Theorem: The area enclosed by the graph of |x| + |2y| = 10 is 100 square units -/
theorem area_is_100 : area_enclosed = 100 := by sorry

end NUMINAMATH_CALUDE_area_is_100_l3382_338276


namespace NUMINAMATH_CALUDE_distance_city_A_to_B_l3382_338290

/-- The distance between city A and city B given the travel times and speeds of Eddy and Freddy -/
theorem distance_city_A_to_B 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (distance_AC : ℝ) 
  (speed_ratio : ℝ) : 
  time_eddy = 3 → 
  time_freddy = 4 → 
  distance_AC = 300 → 
  speed_ratio = 2.1333333333333333 → 
  time_eddy * (speed_ratio * (distance_AC / time_freddy)) = 480 :=
by sorry

end NUMINAMATH_CALUDE_distance_city_A_to_B_l3382_338290


namespace NUMINAMATH_CALUDE_square_over_fraction_equals_324_l3382_338254

theorem square_over_fraction_equals_324 : (45^2 : ℚ) / (7 - 3/4) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_over_fraction_equals_324_l3382_338254


namespace NUMINAMATH_CALUDE_calculate_total_exports_l3382_338298

/-- Calculates the total yearly exports of a country given the percentage of fruit exports,
    the percentage of orange exports within fruit exports, and the revenue from orange exports. -/
theorem calculate_total_exports (fruit_export_percent : ℝ) (orange_export_fraction : ℝ) (orange_export_revenue : ℝ) :
  fruit_export_percent = 0.20 →
  orange_export_fraction = 1 / 6 →
  orange_export_revenue = 4.25 →
  (orange_export_revenue / orange_export_fraction) / fruit_export_percent = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_exports_l3382_338298


namespace NUMINAMATH_CALUDE_divide_into_eight_parts_l3382_338240

-- Define a type for geometric figures
inductive Figure
  | Cube
  | Rectangle

-- Define a function to check if a figure can be divided into 8 identical parts
def canDivideIntoEightParts (f : Figure) : Prop :=
  match f with
  | Figure.Cube => true
  | Figure.Rectangle => true

-- Theorem stating that any cube or rectangle can be divided into 8 identical parts
theorem divide_into_eight_parts (f : Figure) : canDivideIntoEightParts f := by
  sorry

#check divide_into_eight_parts

end NUMINAMATH_CALUDE_divide_into_eight_parts_l3382_338240


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3382_338269

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

-- Define y-intercept
def is_y_intercept (y : ℝ) : Prop :=
  line_equation 0 y

-- Theorem statement
theorem y_intercept_of_line :
  is_y_intercept (-2) :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3382_338269


namespace NUMINAMATH_CALUDE_vincent_stickers_l3382_338287

theorem vincent_stickers (yesterday : ℕ) (today_extra : ℕ) : 
  yesterday = 15 → today_extra = 10 → yesterday + (yesterday + today_extra) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l3382_338287


namespace NUMINAMATH_CALUDE_two_roots_sum_greater_than_2a_l3382_338229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 2

theorem two_roots_sum_greater_than_2a (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_two_roots_sum_greater_than_2a_l3382_338229


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3382_338224

theorem sum_of_three_numbers (p q r M : ℚ) 
  (h1 : p + q + r = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : r / 5 = M) :
  M = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3382_338224


namespace NUMINAMATH_CALUDE_journey_time_bounds_l3382_338262

/-- Represents the bus journey from Kimovsk to Moscow -/
structure BusJourney where
  speed : ℝ
  kimovsk_novomoskovsk : ℝ
  novomoskovsk_tula : ℝ
  tula_moscow : ℝ
  kimovsk_tula_time : ℝ
  novomoskovsk_moscow_time : ℝ

/-- The conditions of the bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.speed ≤ 60 ∧
  j.kimovsk_novomoskovsk = 35 ∧
  j.novomoskovsk_tula = 60 ∧
  j.tula_moscow = 200 ∧
  j.kimovsk_tula_time = 2 ∧
  j.novomoskovsk_moscow_time = 5

/-- The theorem stating the bounds of the total journey time -/
theorem journey_time_bounds (j : BusJourney) 
  (h : journey_conditions j) : 
  ∃ (t : ℝ), 5 + 7/12 ≤ t ∧ t ≤ 6 ∧ 
  t = (j.kimovsk_novomoskovsk + j.novomoskovsk_tula + j.tula_moscow) / j.speed :=
sorry

end NUMINAMATH_CALUDE_journey_time_bounds_l3382_338262


namespace NUMINAMATH_CALUDE_pipe_ratio_proof_l3382_338291

theorem pipe_ratio_proof (total_length longer_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : longer_length = 118)
  (h3 : ∃ k : ℕ, k * (total_length - longer_length) = longer_length) :
  longer_length / (total_length - longer_length) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_ratio_proof_l3382_338291


namespace NUMINAMATH_CALUDE_janet_waiting_time_l3382_338288

/-- Proves that Janet waits 3 hours for her sister to cross the lake -/
theorem janet_waiting_time (lake_width : ℝ) (janet_speed : ℝ) (sister_speed : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  sister_speed = 12 →
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_waiting_time_l3382_338288


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3382_338297

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line l: x + y - 1 = 0 -/
def line_l (p : Point) : Prop := p.x + p.y - 1 = 0

/-- The specific condition x=2 and y=-1 -/
def specific_condition (p : Point) : Prop := p.x = 2 ∧ p.y = -1

/-- Theorem stating that the specific condition is sufficient but not necessary -/
theorem sufficient_not_necessary :
  (∀ p : Point, specific_condition p → line_l p) ∧
  ¬(∀ p : Point, line_l p → specific_condition p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3382_338297


namespace NUMINAMATH_CALUDE_zongzi_prices_l3382_338256

-- Define variables
variable (x : ℝ) -- Purchase price of egg yolk zongzi
variable (y : ℝ) -- Purchase price of red bean zongzi
variable (m : ℝ) -- Selling price of egg yolk zongzi

-- Define conditions
def first_purchase : Prop := 60 * x + 90 * y = 4800
def second_purchase : Prop := 40 * x + 80 * y = 3600
def initial_sales : Prop := m = 70 ∧ (70 - 50) * 20 = 400
def sales_change : Prop := ∀ p, (p - 50) * (20 + 5 * (70 - p)) = 220 → p = 52

-- Theorem statement
theorem zongzi_prices :
  first_purchase x y ∧ second_purchase x y ∧ initial_sales m ∧ sales_change →
  x = 50 ∧ y = 20 ∧ m = 52 := by
  sorry

end NUMINAMATH_CALUDE_zongzi_prices_l3382_338256


namespace NUMINAMATH_CALUDE_factor_count_of_n_l3382_338207

-- Define the number we're working with
def n : ℕ := 8^2 * 9^3 * 10^4

-- Define a function to count distinct natural-number factors
def count_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem factor_count_of_n : count_factors n = 385 := by sorry

end NUMINAMATH_CALUDE_factor_count_of_n_l3382_338207


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3382_338264

theorem arithmetic_mean_problem : ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 9 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3382_338264


namespace NUMINAMATH_CALUDE_multiplicative_inverse_289_mod_391_l3382_338273

theorem multiplicative_inverse_289_mod_391 
  (h : 136^2 + 255^2 = 289^2) : 
  (289 * 18) % 391 = 1 ∧ 0 ≤ 18 ∧ 18 < 391 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_289_mod_391_l3382_338273


namespace NUMINAMATH_CALUDE_bus_fare_payment_possible_l3382_338267

/-- Represents a person with their initial money and final payment -/
structure Person where
  initial_money : ℕ
  final_payment : ℕ

/-- Represents the bus fare payment scenario -/
def BusFareScenario (fare : ℕ) (people : List Person) : Prop :=
  (people.length = 3) ∧
  (∀ p ∈ people, p.final_payment = fare) ∧
  (∃ total : ℕ, total = people.foldl (λ sum person => sum + person.initial_money) 0) ∧
  (∃ payer : Person, payer ∈ people ∧ payer.initial_money ≥ 3 * fare)

/-- Theorem stating that it's possible to pay the bus fare -/
theorem bus_fare_payment_possible (fare : ℕ) (people : List Person) 
  (h : BusFareScenario fare people) : 
  ∃ (final_money : List ℕ), 
    final_money.length = people.length ∧ 
    final_money.sum = people.foldl (λ sum person => sum + person.initial_money) 0 - 3 * fare :=
sorry

end NUMINAMATH_CALUDE_bus_fare_payment_possible_l3382_338267


namespace NUMINAMATH_CALUDE_right_triangle_pq_length_l3382_338283

/-- Given a right triangle PQR with ∠P = 90°, QR = 15, and tan R = 5 cos Q, prove that PQ = 6√6 -/
theorem right_triangle_pq_length (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let cos_q := pq / qr
  let tan_r := pq / pr
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at P
  qr = 15 →
  tan_r = 5 * cos_q →
  pq = 6 * Real.sqrt 6 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_pq_length_l3382_338283


namespace NUMINAMATH_CALUDE_car_travel_distance_l3382_338237

theorem car_travel_distance (speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) (distance : ℝ) : 
  speed = 80 →
  speed_increase = 40 →
  time_decrease = 0.5 →
  distance / speed - distance / (speed + speed_increase) = time_decrease →
  distance = 120 := by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l3382_338237


namespace NUMINAMATH_CALUDE_final_women_count_room_population_problem_l3382_338222

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Represents the changes in population -/
structure PopulationChange where
  menEntered : ℕ
  womenLeft : ℕ
  womenMultiplier : ℕ

/-- The theorem to prove -/
theorem final_women_count 
  (initialRatio : Rat) 
  (changes : PopulationChange) 
  (finalMenCount : ℕ) : ℕ :=
  by
    sorry

/-- The main theorem that encapsulates the problem -/
theorem room_population_problem : 
  final_women_count (7/8) ⟨4, 5, 3⟩ 16 = 27 :=
  by
    sorry

end NUMINAMATH_CALUDE_final_women_count_room_population_problem_l3382_338222


namespace NUMINAMATH_CALUDE_system_solutions_l3382_338252

theorem system_solutions (x y z : ℝ) : 
  (x * (3 * y^2 + 1) = y * (y^2 + 3) ∧
   y * (3 * z^2 + 1) = z * (z^2 + 3) ∧
   z * (3 * x^2 + 1) = x * (x^2 + 3)) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
   (x = -1 ∧ y = -1 ∧ z = -1) ∨ 
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3382_338252


namespace NUMINAMATH_CALUDE_age_gap_ratio_l3382_338243

/-- Given the birth years of family members, prove the ratio of age gaps -/
theorem age_gap_ratio (older_brother_birth : ℕ) (older_sister_birth : ℕ) (grandmother_birth : ℕ)
  (h1 : older_brother_birth = 1932)
  (h2 : older_sister_birth = 1936)
  (h3 : grandmother_birth = 1944) :
  (grandmother_birth - older_sister_birth) / (older_sister_birth - older_brother_birth) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_gap_ratio_l3382_338243


namespace NUMINAMATH_CALUDE_fraction_simplification_l3382_338268

theorem fraction_simplification (x : ℝ) : (2*x + 3)/4 + (5 - 4*x)/3 = (-10*x + 29)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3382_338268


namespace NUMINAMATH_CALUDE_min_broken_line_length_l3382_338244

/-- Given points A and C in the coordinate plane, and point B on the x-axis,
    the minimum length of the broken line ABC is 7.5 -/
theorem min_broken_line_length :
  let A : ℝ × ℝ := (-3, -4)
  let C : ℝ × ℝ := (1.5, -2)
  ∃ B : ℝ × ℝ, B.2 = 0 ∧
    ∀ B' : ℝ × ℝ, B'.2 = 0 →
      Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
      Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≤
      Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2) +
      Real.sqrt ((C.1 - B'.1)^2 + (C.2 - B'.2)^2) ∧
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_min_broken_line_length_l3382_338244


namespace NUMINAMATH_CALUDE_solution_set_satisfies_inequalities_l3382_338285

def S : Set ℝ := {x | 0 < x ∧ x < 1}

theorem solution_set_satisfies_inequalities :
  ∀ x ∈ S, x * (x + 2) > 0 ∧ |x| < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_satisfies_inequalities_l3382_338285


namespace NUMINAMATH_CALUDE_zoo_field_trip_remaining_individuals_l3382_338247

/-- Represents the number of individuals from a school -/
structure SchoolGroup :=
  (students : ℕ)
  (parents : ℕ)
  (teachers : ℕ)

/-- Calculates the total number of individuals in a school group -/
def SchoolGroup.total (sg : SchoolGroup) : ℕ :=
  sg.students + sg.parents + sg.teachers

theorem zoo_field_trip_remaining_individuals
  (school_a : SchoolGroup)
  (school_b : SchoolGroup)
  (school_c : SchoolGroup)
  (school_d : SchoolGroup)
  (h1 : school_a = ⟨10, 5, 2⟩)
  (h2 : school_b = ⟨12, 3, 2⟩)
  (h3 : school_c = ⟨15, 3, 0⟩)
  (h4 : school_d = ⟨20, 4, 0⟩)
  (left_students_ab : ℕ)
  (left_students_c : ℕ)
  (left_students_d : ℕ)
  (left_parents_a : ℕ)
  (left_parents_c : ℕ)
  (h5 : left_students_ab = 10)
  (h6 : left_students_c = 6)
  (h7 : left_students_d = 9)
  (h8 : left_parents_a = 2)
  (h9 : left_parents_c = 1)
  : (school_a.total + school_b.total + school_c.total + school_d.total) -
    (left_students_ab + left_students_c + left_students_d + left_parents_a + left_parents_c) = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_zoo_field_trip_remaining_individuals_l3382_338247


namespace NUMINAMATH_CALUDE_building_occupancy_ratio_l3382_338216

/-- Calculates the occupancy ratio of a building given the number of units,
    monthly rent per unit, and total annual rent received. -/
theorem building_occupancy_ratio
  (num_units : ℕ)
  (monthly_rent : ℝ)
  (annual_rent_received : ℝ)
  (h1 : num_units = 100)
  (h2 : monthly_rent = 400)
  (h3 : annual_rent_received = 360000) :
  annual_rent_received / (num_units * monthly_rent * 12) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_building_occupancy_ratio_l3382_338216


namespace NUMINAMATH_CALUDE_modifiedLucas_50th_term_mod_5_l3382_338251

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modifiedLucas_50th_term_mod_5 :
  modifiedLucas 49 % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_modifiedLucas_50th_term_mod_5_l3382_338251


namespace NUMINAMATH_CALUDE_kristy_work_hours_l3382_338255

/-- Proves that given the conditions of Kristy's salary structure and earnings,
    she worked 160 hours in the month. -/
theorem kristy_work_hours :
  let hourly_rate : ℝ := 7.5
  let commission_rate : ℝ := 0.16
  let total_sales : ℝ := 25000
  let insurance_amount : ℝ := 260
  let insurance_rate : ℝ := 0.05
  let commission : ℝ := commission_rate * total_sales
  let total_earnings : ℝ := insurance_amount / insurance_rate
  let hours_worked : ℝ := (total_earnings - commission) / hourly_rate
  hours_worked = 160 := by sorry

end NUMINAMATH_CALUDE_kristy_work_hours_l3382_338255


namespace NUMINAMATH_CALUDE_positive_A_value_l3382_338249

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3382_338249


namespace NUMINAMATH_CALUDE_least_perimeter_of_triangle_l3382_338221

theorem least_perimeter_of_triangle (a b c : ℕ) : 
  a = 24 → b = 51 → c > 0 → a + b > c → a + c > b → b + c > a → 
  ∀ x : ℕ, (x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a) → a + b + c ≤ a + b + x →
  a + b + c = 103 := by sorry

end NUMINAMATH_CALUDE_least_perimeter_of_triangle_l3382_338221


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3382_338218

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3382_338218


namespace NUMINAMATH_CALUDE_power_of_five_congruences_and_digits_l3382_338245

theorem power_of_five_congruences_and_digits : 
  (∃ k : ℕ, 5^500 = 1000 * k + 1) ∧ 
  (∃ m : ℕ, 5^10000 = 1000 * m + 1) ∧
  5^10000 % 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_five_congruences_and_digits_l3382_338245


namespace NUMINAMATH_CALUDE_triangle_formation_l3382_338250

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (5, 6, 10), (1, 2, 3)]

theorem triangle_formation :
  ∃! (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3382_338250


namespace NUMINAMATH_CALUDE_total_salary_proof_l3382_338209

def salary_n : ℝ := 260

def salary_m : ℝ := 1.2 * salary_n

def total_salary : ℝ := salary_m + salary_n

theorem total_salary_proof : total_salary = 572 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_proof_l3382_338209


namespace NUMINAMATH_CALUDE_last_four_digits_of_perfect_square_l3382_338227

theorem last_four_digits_of_perfect_square (n : ℕ) : 
  (∃ d : ℕ, d < 10 ∧ n^2 % 1000 = d * 111) → 
  (n^2 % 10000 = 0 ∨ n^2 % 10000 = 1444) :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_perfect_square_l3382_338227


namespace NUMINAMATH_CALUDE_expression_simplification_l3382_338214

theorem expression_simplification (a x : ℝ) 
  (h1 : x ≠ a / 3) (h2 : x ≠ -a / 3) (h3 : x ≠ -a) : 
  (3 * a^2 + 2 * a * x - x^2) / ((3 * x + a) * (a + x)) - 2 + 
  10 * (a * x - 3 * x^2) / (a^2 - 9 * x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3382_338214


namespace NUMINAMATH_CALUDE_players_sold_is_two_l3382_338233

/-- Represents the financial transactions of a football club --/
def football_club_transactions 
  (initial_balance : ℚ) 
  (selling_price : ℚ) 
  (buying_price : ℚ) 
  (players_bought : ℕ) 
  (final_balance : ℚ) : Prop :=
  ∃ (players_sold : ℕ), 
    initial_balance + (selling_price * players_sold) - (buying_price * players_bought) = final_balance

/-- Theorem stating that the number of players sold is 2 --/
theorem players_sold_is_two : 
  football_club_transactions 100 10 15 4 60 → 
  ∃ (players_sold : ℕ), players_sold = 2 := by
  sorry

end NUMINAMATH_CALUDE_players_sold_is_two_l3382_338233


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3382_338253

theorem contrapositive_equivalence (f : ℝ → ℝ) (a b : ℝ) :
  (¬(f a + f b ≥ f (-a) + f (-b)) → ¬(a + b ≥ 0)) ↔
  (f a + f b < f (-a) + f (-b) → a + b < 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3382_338253


namespace NUMINAMATH_CALUDE_tank_emptying_time_l3382_338257

/-- Represents the time (in minutes) it takes to empty a water tank -/
def empty_tank_time (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

theorem tank_emptying_time :
  let initial_fill : ℚ := 4/5
  let fill_pipe_rate : ℚ := 1/10
  let empty_pipe_rate : ℚ := 1/6
  empty_tank_time initial_fill fill_pipe_rate empty_pipe_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l3382_338257


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3382_338204

theorem divide_eight_by_repeating_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3382_338204


namespace NUMINAMATH_CALUDE_circle_condition_tangent_lines_perpendicular_intersection_l3382_338294

-- Define the equation of circle C
def C (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + a = 0

-- Define the line l
def l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem 1: C represents a circle iff a ∈ (-∞, 8)
theorem circle_condition (a : ℝ) : 
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), C x y a ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ 
  a < 8 :=
sorry

-- Theorem 2: Tangent lines when a = -17
theorem tangent_lines : 
  (∀ (x y : ℝ), C x y (-17) → (39*x + 80*y - 207 = 0 ∨ x = 7)) ∧
  C 7 (-6) (-17) ∧
  (∃ (x y : ℝ), C x y (-17) ∧ 39*x + 80*y - 207 = 0) ∧
  (∃ (y : ℝ), C 7 y (-17)) :=
sorry

-- Theorem 3: Value of a when OA ⊥ OB
theorem perpendicular_intersection :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C x₁ y₁ (-6/5) ∧ C x₂ y₂ (-6/5) ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_tangent_lines_perpendicular_intersection_l3382_338294


namespace NUMINAMATH_CALUDE_triangle_properties_l3382_338292

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation holds for the triangle -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) / (t.a^2 + t.b^2 - t.c^2) = t.c / (Real.sqrt 2 * t.a - t.c)

theorem triangle_properties (t : Triangle) (h : satisfies_equation t) :
  t.B = π/4 ∧ 
  (t.b = 1 → ∃ (max_area : ℝ), max_area = (Real.sqrt 2 + 1) / 4 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.c * Real.sin t.B → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3382_338292


namespace NUMINAMATH_CALUDE_wednesday_temperature_l3382_338293

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C under the given conditions. -/
theorem wednesday_temperature (M T W : ℤ) : 
  T = M + 4 →   -- Tuesday's temperature is 4°C warmer than Monday's
  W = M - 6 →   -- Wednesday's temperature is 6°C cooler than Monday's
  T = 22 →      -- Tuesday's temperature is 22°C
  W = 12 :=     -- Prove: Wednesday's temperature is 12°C
by sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l3382_338293


namespace NUMINAMATH_CALUDE_inequality_proof_l3382_338279

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3382_338279


namespace NUMINAMATH_CALUDE_variance_equality_and_percentile_l3382_338239

-- Define the sequences x_i and y_i
def x : Fin 10 → ℝ := fun i => 2 * (i.val + 1)
def y : Fin 10 → ℝ := fun i => x i - 20

-- Define variance function
def variance (s : Fin 10 → ℝ) : ℝ := sorry

-- Define percentile function
def percentile (p : ℝ) (s : Fin 10 → ℝ) : ℝ := sorry

theorem variance_equality_and_percentile :
  (variance x = variance y) ∧ (percentile 0.3 y = -13) := by sorry

end NUMINAMATH_CALUDE_variance_equality_and_percentile_l3382_338239


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3382_338228

theorem complex_equation_solution (z : ℂ) :
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3382_338228


namespace NUMINAMATH_CALUDE_cos_42_cos_78_minus_sin_42_sin_78_l3382_338211

theorem cos_42_cos_78_minus_sin_42_sin_78 :
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) -
  Real.sin (42 * π / 180) * Real.sin (78 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_78_minus_sin_42_sin_78_l3382_338211


namespace NUMINAMATH_CALUDE_set_B_equals_l3382_338232

def A : Set ℝ := {x | x^2 ≤ 4}

def B : Set ℕ := {x | x > 0 ∧ (x - 1 : ℝ) ∈ A}

theorem set_B_equals : B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l3382_338232


namespace NUMINAMATH_CALUDE_jess_walk_distance_l3382_338234

/-- The number of blocks Jess must walk to complete her errands and arrive at work -/
def remaining_blocks (post_office store gallery library work already_walked : ℕ) : ℕ :=
  post_office + store + gallery + library + work - already_walked

/-- Theorem stating the number of blocks Jess must walk given the problem conditions -/
theorem jess_walk_distance :
  remaining_blocks 24 18 15 14 22 9 = 84 := by
  sorry

end NUMINAMATH_CALUDE_jess_walk_distance_l3382_338234


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3382_338265

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 5*x ↔ x = 0 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3382_338265


namespace NUMINAMATH_CALUDE_calculation_proofs_l3382_338296

theorem calculation_proofs :
  (6 / (-1/2 + 1/3) = -36) ∧
  ((-14/17) * 99 + (13/17) * 99 - (16/17) * 99 = -99) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3382_338296


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_eight_l3382_338286

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_sum_equals_negative_eight :
  ∃ (a b : ℝ), f a = 4 ∧ f b = -100 ∧ a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_eight_l3382_338286


namespace NUMINAMATH_CALUDE_remaining_pets_count_l3382_338231

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : ℕ
  kittens : ℕ
  rabbits : ℕ
  guineaPigs : ℕ
  chameleons : ℕ
  parrots : ℕ

/-- Calculates the total number of pets -/
def totalPets (counts : PetCounts) : ℕ :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guineaPigs + counts.chameleons + counts.parrots

/-- Represents the pet store transactions throughout the day -/
def petStoreTransactions (initial : PetCounts) : PetCounts :=
  { puppies := initial.puppies - 2 - 1 + 3 - 1 - 1,
    kittens := initial.kittens - 1 - 2 + 2 - 1 + 1 - 1,
    rabbits := initial.rabbits - 1 - 1 + 1 - 1 - 1,
    guineaPigs := initial.guineaPigs - 1 - 2 - 1 - 1,
    chameleons := initial.chameleons + 1 + 2 - 1,
    parrots := initial.parrots - 1 }

/-- The main theorem stating that after all transactions, 16 pets remain -/
theorem remaining_pets_count (initial : PetCounts)
    (h_initial : initial = { puppies := 7, kittens := 6, rabbits := 4,
                             guineaPigs := 5, chameleons := 3, parrots := 2 }) :
    totalPets (petStoreTransactions initial) = 16 := by
  sorry


end NUMINAMATH_CALUDE_remaining_pets_count_l3382_338231


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3382_338275

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_inverse : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h_nonzero_x : x₁ ≠ 0 ∧ x₂ ≠ 0) (h_nonzero_y : y₁ ≠ 0 ∧ y₂ ≠ 0) (h_ratio_x : x₁ / x₂ = 4 / 5) 
    (h_correspond : x₁ * y₁ = x₂ * y₂) : y₁ / y₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3382_338275


namespace NUMINAMATH_CALUDE_angle_A_is_30_degrees_l3382_338213

/-- In a triangle ABC, given that the side opposite to angle B is twice the length of the side opposite to angle A, and angle B is 60° greater than angle A, prove that angle A measures 30°. -/
theorem angle_A_is_30_degrees (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 * a ∧  -- Given condition
  B = A + π / 3 →  -- B = A + 60° (in radians)
  A = π / 6 :=  -- A = 30° (in radians)
by sorry

end NUMINAMATH_CALUDE_angle_A_is_30_degrees_l3382_338213


namespace NUMINAMATH_CALUDE_last_digit_of_fraction_l3382_338263

def last_digit (n : ℚ) : ℕ := sorry

theorem last_digit_of_fraction :
  last_digit (1 / (2^10 * 3^10)) = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_fraction_l3382_338263


namespace NUMINAMATH_CALUDE_evaluate_expression_l3382_338235

theorem evaluate_expression : 8^3 + 4*(8^2) + 6*8 + 3 = 1000 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3382_338235


namespace NUMINAMATH_CALUDE_smallest_nonnegative_a_l3382_338266

theorem smallest_nonnegative_a (b : ℝ) (a : ℝ) (h1 : b = π / 4) 
  (h2 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 0 ∧ a = 17 - π / 4 ∧ ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ a :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_a_l3382_338266


namespace NUMINAMATH_CALUDE_butter_production_theorem_l3382_338284

/-- Represents the problem of determining butter production from milk --/
structure MilkButterProblem where
  milk_price : ℚ
  butter_price : ℚ
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  total_earnings : ℚ

/-- Calculates the number of sticks of butter that can be made from one gallon of milk --/
def sticks_per_gallon (p : MilkButterProblem) : ℚ :=
  let total_milk := p.num_cows * p.milk_per_cow
  let sold_milk := p.num_customers * p.milk_per_customer
  let milk_revenue := sold_milk * p.milk_price
  let butter_revenue := p.total_earnings - milk_revenue
  let milk_for_butter := total_milk - sold_milk
  let total_butter_sticks := butter_revenue / p.butter_price
  total_butter_sticks / milk_for_butter

/-- Theorem stating that for the given problem conditions, 2 sticks of butter can be made per gallon of milk --/
theorem butter_production_theorem (p : MilkButterProblem) 
  (h1 : p.milk_price = 3)
  (h2 : p.butter_price = 3/2)
  (h3 : p.num_cows = 12)
  (h4 : p.milk_per_cow = 4)
  (h5 : p.num_customers = 6)
  (h6 : p.milk_per_customer = 6)
  (h7 : p.total_earnings = 144) :
  sticks_per_gallon p = 2 := by
  sorry

end NUMINAMATH_CALUDE_butter_production_theorem_l3382_338284


namespace NUMINAMATH_CALUDE_parabola_vertex_l3382_338280

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 + ay + bx + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_vertex :
  let p : Parabola := { equation := fun y x _ => y^2 - 4*y + 3*x + 7 }
  vertex p = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3382_338280


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3382_338201

theorem largest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9985 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l3382_338201


namespace NUMINAMATH_CALUDE_weeks_of_feed_l3382_338241

-- Define the given quantities
def boxes_bought : ℕ := 3
def boxes_in_pantry : ℕ := 5
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Calculate total boxes and total grams
def total_boxes : ℕ := boxes_bought + boxes_in_pantry
def total_grams : ℕ := total_boxes * grams_per_box

-- Calculate weekly consumption
def weekly_consumption : ℕ := parrot_consumption + cockatiel_consumption

-- Theorem to prove
theorem weeks_of_feed : (total_grams / weekly_consumption : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_weeks_of_feed_l3382_338241


namespace NUMINAMATH_CALUDE_area_covered_by_specific_strips_l3382_338236

/-- Calculates the area covered by four rectangular strips on a table. -/
def areaCoveredByStrips (lengths : List Nat) (width : Nat) (overlaps : Nat) : Nat :=
  let totalArea := (lengths.sum * width)
  let overlapArea := overlaps * width
  totalArea - overlapArea

/-- Theorem: The area covered by four specific strips is 33. -/
theorem area_covered_by_specific_strips :
  areaCoveredByStrips [12, 10, 8, 6] 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_specific_strips_l3382_338236


namespace NUMINAMATH_CALUDE_valid_numbers_are_unique_l3382_338206

/-- Represents a six-digit number of the form 387abc --/
def SixDigitNumber (a b c : Nat) : Nat :=
  387000 + a * 100 + b * 10 + c

/-- Checks if a natural number is divisible by 5, 6, and 7 --/
def isDivisibleBy567 (n : Nat) : Prop :=
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0

/-- The set of valid six-digit numbers --/
def ValidNumbers : Set Nat :=
  {387000, 387210, 387420, 387630, 387840}

/-- Theorem stating that the ValidNumbers are the only six-digit numbers
    of the form 387abc that are divisible by 5, 6, and 7 --/
theorem valid_numbers_are_unique :
  ∀ a b c : Nat, a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy567 (SixDigitNumber a b c) ↔ SixDigitNumber a b c ∈ ValidNumbers :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_are_unique_l3382_338206


namespace NUMINAMATH_CALUDE_least_repeating_digits_7_13_is_6_l3382_338270

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_digits_7_13 : ℕ :=
  6

/-- Theorem stating that the least number of digits in a repeating block 
    of the decimal expansion of 7/13 is 6 -/
theorem least_repeating_digits_7_13_is_6 :
  least_repeating_digits_7_13 = 6 := by sorry

end NUMINAMATH_CALUDE_least_repeating_digits_7_13_is_6_l3382_338270


namespace NUMINAMATH_CALUDE_partner_numbers_problem_l3382_338215

/-- Definition of "partner numbers" -/
def partner_numbers (m n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    m = 100 * a + 10 * b + c ∧
    n = 100 * d + 10 * e + f ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    1 ≤ e ∧ e ≤ 9 ∧
    1 ≤ f ∧ f ≤ 9 ∧
    ∃ (k : ℤ), k * (b - c) = a + 4 * d + 4 * e + 4 * f

theorem partner_numbers_problem (x y z : ℕ) 
  (hx : x ≤ 3) 
  (hy : 0 < y ∧ y ≤ 4) 
  (hz : 3 < z ∧ z ≤ 9) 
  (h_partner : partner_numbers (467 + 110 * x) (200 * y + z + 37))
  (h_sum : (2 * y + z + 1) % 12 = 0) :
  467 + 110 * x = 467 ∨ 467 + 110 * x = 687 := by
  sorry

end NUMINAMATH_CALUDE_partner_numbers_problem_l3382_338215


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3382_338272

/-- A random vector following a normal distribution with mean 3 and variance 1 -/
def X : Type := Real

/-- The probability density function of X -/
noncomputable def pdf (x : X) : Real := sorry

/-- The cumulative distribution function of X -/
noncomputable def cdf (x : X) : Real := sorry

/-- The probability that X is greater than a given value -/
noncomputable def P_greater (a : Real) : Real := 1 - cdf a

/-- The probability that X is less than a given value -/
noncomputable def P_less (a : Real) : Real := cdf a

/-- The theorem stating that if P(X > 2c - 1) = P(X < c + 3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : Real) :
  P_greater (2 * c - 1) = P_less (c + 3) → c = 4/3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3382_338272


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3382_338202

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 8 ∧ ∀ (m : ℕ), m > 0 → (3 * m) % 26 = 8 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3382_338202


namespace NUMINAMATH_CALUDE_function_maximum_value_l3382_338295

theorem function_maximum_value (x : ℝ) (h : x > 4) :
  (fun x => -x + 1 / (4 - x)) x ≤ -6 :=
by sorry

end NUMINAMATH_CALUDE_function_maximum_value_l3382_338295


namespace NUMINAMATH_CALUDE_continuous_functions_integral_bound_l3382_338299

open Set
open MeasureTheory
open Interval

theorem continuous_functions_integral_bound 
  (f g : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g)
  (hf_integral : ∫ x in (Icc 0 1), (f x)^2 = 1)
  (hg_integral : ∫ x in (Icc 0 1), (g x)^2 = 1) :
  ∃ c ∈ Icc 0 1, f c + g c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_continuous_functions_integral_bound_l3382_338299


namespace NUMINAMATH_CALUDE_system_solutions_product_l3382_338208

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x^3 - 5*x*y^2 = 21 ∧ y^3 - 5*x^2*y = 28

-- Define the theorem
theorem system_solutions_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  system x₁ y₁ ∧ system x₂ y₂ ∧ system x₃ y₃ ∧
  (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃) →
  (11 - x₁/y₁) * (11 - x₂/y₂) * (11 - x₃/y₃) = 1729 :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_product_l3382_338208


namespace NUMINAMATH_CALUDE_team_points_distribution_l3382_338278

theorem team_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x ∧ 
  y ≤ 24 ∧ 
  ∀ (z : ℕ), z ≤ 8 → (y : ℝ) / 8 ≤ 3 →
  y = 17 :=
sorry

end NUMINAMATH_CALUDE_team_points_distribution_l3382_338278


namespace NUMINAMATH_CALUDE_exponent_power_rule_l3382_338238

theorem exponent_power_rule (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_power_rule_l3382_338238


namespace NUMINAMATH_CALUDE_local_min_condition_l3382_338242

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a^2 - 2*a)*x

/-- Theorem stating the condition for x = 0 to be a local minimum of f -/
theorem local_min_condition (a : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → f a x ≥ f a 0) ↔ a < 0 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_local_min_condition_l3382_338242


namespace NUMINAMATH_CALUDE_normal_vector_of_det_equation_l3382_338258

/-- The determinant equation of a line -/
def det_equation (x y : ℝ) : Prop := x * 1 - y * 2 = 0

/-- Definition of a normal vector -/
def is_normal_vector (n : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line_eq x y → n.1 * x + n.2 * y = 0

/-- Theorem: (1, -2) is a normal vector of the line represented by |x 2; y 1| = 0 -/
theorem normal_vector_of_det_equation :
  is_normal_vector (1, -2) det_equation :=
sorry

end NUMINAMATH_CALUDE_normal_vector_of_det_equation_l3382_338258


namespace NUMINAMATH_CALUDE_pen_pencil_difference_is_1500_l3382_338217

/-- Represents the stationery order problem --/
structure StationeryOrder where
  pencilBoxes : ℕ
  pencilsPerBox : ℕ
  penCost : ℕ
  pencilCost : ℕ
  totalCost : ℕ

/-- Calculates the difference between pens and pencils ordered --/
def penPencilDifference (order : StationeryOrder) : ℕ :=
  let totalPencils := order.pencilBoxes * order.pencilsPerBox
  let totalPenCost := order.totalCost - order.pencilCost * totalPencils
  let totalPens := totalPenCost / order.penCost
  totalPens - totalPencils

/-- Theorem stating the difference between pens and pencils ordered --/
theorem pen_pencil_difference_is_1500 (order : StationeryOrder) 
  (h1 : order.pencilBoxes = 15)
  (h2 : order.pencilsPerBox = 80)
  (h3 : order.penCost = 5)
  (h4 : order.pencilCost = 4)
  (h5 : order.totalCost = 18300)
  (h6 : order.penCost * (penPencilDifference order + order.pencilBoxes * order.pencilsPerBox) > 
        2 * order.pencilCost * (order.pencilBoxes * order.pencilsPerBox)) :
  penPencilDifference order = 1500 := by
  sorry

#eval penPencilDifference { pencilBoxes := 15, pencilsPerBox := 80, penCost := 5, pencilCost := 4, totalCost := 18300 }

end NUMINAMATH_CALUDE_pen_pencil_difference_is_1500_l3382_338217


namespace NUMINAMATH_CALUDE_pyramid_division_volumes_l3382_338220

/-- Right quadrangular pyramid with inscribed prism and dividing plane -/
structure PyramidWithPrism where
  /-- Side length of the pyramid's base -/
  a : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Side length of the prism's base -/
  b : ℝ
  /-- Height of the prism -/
  h₀ : ℝ
  /-- Condition: The side length of the pyramid's base is 8√2 -/
  ha : a = 8 * Real.sqrt 2
  /-- Condition: The height of the pyramid is 4 -/
  hh : h = 4
  /-- Condition: The side length of the prism's base is 2 -/
  hb : b = 2
  /-- Condition: The height of the prism is 1 -/
  hh₀ : h₀ = 1

/-- Theorem stating the volumes of the parts divided by plane γ -/
theorem pyramid_division_volumes (p : PyramidWithPrism) :
  ∃ (v₁ v₂ : ℝ), v₁ = 512 / 15 ∧ v₂ = 2048 / 15 ∧
  v₁ + v₂ = (1 / 3) * p.a^2 * p.h :=
sorry

end NUMINAMATH_CALUDE_pyramid_division_volumes_l3382_338220


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3382_338200

/-- Represents the probability of selecting a blue marble from a bag with specific conditions. -/
theorem blue_marble_probability (total : ℕ) (yellow : ℕ) (h1 : total = 60) (h2 : yellow = 20) :
  let green := yellow / 2
  let remaining := total - yellow - green
  let blue := remaining / 2
  (blue : ℚ) / total = 1/4 := by sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l3382_338200


namespace NUMINAMATH_CALUDE_solve_equation_l3382_338271

theorem solve_equation (x : ℝ) : 1 - 2 / (1 + x) = 1 / (1 + x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3382_338271


namespace NUMINAMATH_CALUDE_real_part_of_Z_l3382_338248

theorem real_part_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.abs (3 + 4 * Complex.I)) : 
  Z.re = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_Z_l3382_338248
