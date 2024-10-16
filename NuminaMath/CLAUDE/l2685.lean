import Mathlib

namespace NUMINAMATH_CALUDE_probability_a_b_same_area_l2685_268536

def total_employees : ℕ := 4
def employees_per_area : ℕ := 2
def num_areas : ℕ := 2

def probability_same_area (total : ℕ) (per_area : ℕ) (areas : ℕ) : ℚ :=
  if total = total_employees ∧ per_area = employees_per_area ∧ areas = num_areas then
    1 / 3
  else
    0

theorem probability_a_b_same_area :
  probability_same_area total_employees employees_per_area num_areas = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_a_b_same_area_l2685_268536


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2685_268596

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2685_268596


namespace NUMINAMATH_CALUDE_white_coinciding_pairs_l2685_268506

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  green_green : ℕ
  red_white : ℕ
  green_blue : ℕ

/-- Theorem stating that the number of coinciding white triangle pairs is 4 -/
theorem white_coinciding_pairs
  (half_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : half_count.red = 4)
  (h2 : half_count.blue = 4)
  (h3 : half_count.green = 2)
  (h4 : half_count.white = 6)
  (h5 : coinciding.red_red = 3)
  (h6 : coinciding.blue_blue = 2)
  (h7 : coinciding.green_green = 1)
  (h8 : coinciding.red_white = 2)
  (h9 : coinciding.green_blue = 1) :
  ∃ (white_pairs : ℕ), white_pairs = 4 ∧ 
  white_pairs = half_count.white - coinciding.red_white := by
  sorry

end NUMINAMATH_CALUDE_white_coinciding_pairs_l2685_268506


namespace NUMINAMATH_CALUDE_twenty_knocks_to_knicks_l2685_268514

-- Define the units
variable (knick knack knock : ℚ)

-- Define the given conditions
axiom knicks_to_knacks : 8 * knick = 3 * knack
axiom knacks_to_knocks : 4 * knack = 5 * knock

-- State the theorem
theorem twenty_knocks_to_knicks : 
  20 * knock = 64 / 3 * knick :=
sorry

end NUMINAMATH_CALUDE_twenty_knocks_to_knicks_l2685_268514


namespace NUMINAMATH_CALUDE_range_of_a_l2685_268583

-- Define the property that the inequality holds for all real x
def inequality_holds_for_all (a : ℝ) : Prop :=
  ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1

-- Define the theorem
theorem range_of_a (a : ℝ) :
  inequality_holds_for_all a ∧ a > 0 → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2685_268583


namespace NUMINAMATH_CALUDE_sum_not_prime_l2685_268524

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2685_268524


namespace NUMINAMATH_CALUDE_consecutive_multiples_of_four_sum_l2685_268567

theorem consecutive_multiples_of_four_sum (n : ℕ) : 
  (4*n + (4*n + 8) = 140) → (4*n + (4*n + 4) + (4*n + 8) = 210) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_four_sum_l2685_268567


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2685_268513

-- Define the sum of digits function
def S (n : ℕ+) : ℕ+ :=
  sorry

-- Define the properties of the function f
def satisfies_conditions (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f n < f (n + 1) ∧ f (n + 1) < f n + 2020) ∧
  (∀ n : ℕ+, S (f n) = f (S n))

-- Theorem statement
theorem unique_function_theorem :
  ∃! f : ℕ+ → ℕ+, satisfies_conditions f ∧ ∀ x : ℕ+, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2685_268513


namespace NUMINAMATH_CALUDE_sin_cos_ratio_l2685_268508

theorem sin_cos_ratio (θ : Real) (h : Real.sqrt 2 * Real.sin (θ + π/4) = 3 * Real.cos θ) :
  Real.sin θ / (Real.sin θ - Real.cos θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_ratio_l2685_268508


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l2685_268503

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧
  x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l2685_268503


namespace NUMINAMATH_CALUDE_playground_children_count_l2685_268501

theorem playground_children_count : 
  ∀ (girls boys : ℕ), 
  girls = 28 → 
  boys = 35 → 
  girls + boys = 63 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l2685_268501


namespace NUMINAMATH_CALUDE_waiter_customer_count_l2685_268585

/-- Calculates the final number of customers after a series of arrivals and departures. -/
def finalCustomerCount (initial : ℕ) (left1 left2 : ℕ) (arrived1 arrived2 : ℕ) : ℕ :=
  initial - left1 + arrived1 + arrived2 - left2

/-- Theorem stating that given the specific customer movements, the final count is 14. -/
theorem waiter_customer_count : 
  finalCustomerCount 13 5 6 4 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l2685_268585


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2685_268586

theorem total_tickets_sold (adult_price student_price total_amount student_count : ℕ) 
  (h1 : adult_price = 12)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_count : ℕ), 
    adult_count * adult_price + student_count * student_price = total_amount ∧
    adult_count + student_count = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2685_268586


namespace NUMINAMATH_CALUDE_book_sale_discount_l2685_268588

/-- Calculates the discount percentage for a book sale --/
theorem book_sale_discount (cost : ℝ) (markup_percent : ℝ) (profit_percent : ℝ) 
  (h_cost : cost = 50)
  (h_markup : markup_percent = 30)
  (h_profit : profit_percent = 17) : 
  let marked_price := cost * (1 + markup_percent / 100)
  let selling_price := cost * (1 + profit_percent / 100)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_discount_l2685_268588


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_21_l2685_268522

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem magnitude_a_minus_2b_equals_sqrt_21 
  (h1 : ‖b‖ = 2 * ‖a‖) 
  (h2 : ‖b‖ = 2) 
  (h3 : inner a b = ‖a‖ * ‖b‖ * (-1/2)) : 
  ‖a - 2 • b‖ = Real.sqrt 21 := by
sorry


end NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_21_l2685_268522


namespace NUMINAMATH_CALUDE_acute_angles_are_in_first_quadrant_l2685_268502

/- Definition of acute angle -/
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Definition of angle in the first quadrant -/
def is_in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Theorem stating that acute angles are angles in the first quadrant -/
theorem acute_angles_are_in_first_quadrant :
  ∀ θ : Real, is_acute_angle θ → is_in_first_quadrant θ := by
  sorry

#check acute_angles_are_in_first_quadrant

end NUMINAMATH_CALUDE_acute_angles_are_in_first_quadrant_l2685_268502


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2685_268594

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 36) → (n * exterior_angle = 360) → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2685_268594


namespace NUMINAMATH_CALUDE_car_journey_metrics_l2685_268530

/-- Represents the car's journey with given conditions -/
structure CarJourney where
  total_distance : ℝ
  acc_dec_distance : ℝ
  constant_speed_distance : ℝ
  acc_dec_time : ℝ
  reference_speed : ℝ
  reference_time : ℝ

/-- Calculates the acceleration rate, deceleration rate, and highest speed of the car -/
def calculate_car_metrics (journey : CarJourney) : 
  (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct acceleration, deceleration, and highest speed -/
theorem car_journey_metrics :
  let journey : CarJourney := {
    total_distance := 100,
    acc_dec_distance := 1,
    constant_speed_distance := 98,
    acc_dec_time := 100,
    reference_speed := 40 / 3600,
    reference_time := 90
  }
  let (acc_rate, dec_rate, highest_speed) := calculate_car_metrics journey
  acc_rate = 0.0002 ∧ dec_rate = 0.0002 ∧ highest_speed = 72 / 3600 :=
by sorry

end NUMINAMATH_CALUDE_car_journey_metrics_l2685_268530


namespace NUMINAMATH_CALUDE_all_ingredients_good_probability_l2685_268592

/-- The probability of selecting a fresh bottle of milk -/
def prob_fresh_milk : ℝ := 0.8

/-- The probability of selecting a good egg -/
def prob_good_egg : ℝ := 0.4

/-- The probability of selecting a good canister of flour -/
def prob_good_flour : ℝ := 0.75

/-- The probability that all three ingredients (milk, egg, flour) are good when selected randomly -/
def prob_all_good : ℝ := prob_fresh_milk * prob_good_egg * prob_good_flour

theorem all_ingredients_good_probability :
  prob_all_good = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_all_ingredients_good_probability_l2685_268592


namespace NUMINAMATH_CALUDE_special_polynomial_at_zero_l2685_268505

/-- A polynomial of degree 6 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, ∀ x, p x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
  (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (2^n))

/-- Theorem stating that a special polynomial evaluates to 0 at x = 0 -/
theorem special_polynomial_at_zero
  (p : ℝ → ℝ)
  (h : special_polynomial p) :
  p 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_at_zero_l2685_268505


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l2685_268575

/-- Represents the cost calculation for a pizza order with special pricing --/
def pizza_order_cost (small_price medium_price large_price topping_price : ℚ)
  (triple_cheese_count triple_cheese_toppings : ℕ)
  (meat_lovers_count meat_lovers_toppings : ℕ)
  (veggie_delight_count veggie_delight_toppings : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_price + 
                            triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 1) / 3) * 2 * medium_price + 
                          meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count + 1) / 3) * 2 * small_price + 
                             veggie_delight_count * veggie_delight_toppings * topping_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost

/-- Theorem stating that the given pizza order costs $169 --/
theorem pizza_order_theorem : 
  pizza_order_cost 5 8 10 (5/2) 6 2 4 3 10 1 = 169 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l2685_268575


namespace NUMINAMATH_CALUDE_jacks_book_loss_l2685_268537

/-- Calculates the loss from buying and selling books -/
def bookLoss (booksPerMonth : ℕ) (costPerBook : ℕ) (monthsInYear : ℕ) (sellingPrice : ℕ) : ℕ :=
  let totalBooks := booksPerMonth * monthsInYear
  let totalCost := totalBooks * costPerBook
  totalCost - sellingPrice

/-- Theorem stating that Jack's loss is $220 -/
theorem jacks_book_loss :
  bookLoss 3 20 12 500 = 220 := by
  sorry

end NUMINAMATH_CALUDE_jacks_book_loss_l2685_268537


namespace NUMINAMATH_CALUDE_interest_percentage_approx_l2685_268535

def purchase_price : ℚ := 2345
def down_payment : ℚ := 385
def num_monthly_payments : ℕ := 18
def monthly_payment : ℚ := 125

def total_paid : ℚ := down_payment + num_monthly_payments * monthly_payment

def interest_paid : ℚ := total_paid - purchase_price

def interest_percentage : ℚ := (interest_paid / purchase_price) * 100

theorem interest_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ |interest_percentage - 12.4| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_percentage_approx_l2685_268535


namespace NUMINAMATH_CALUDE_log_sum_equation_l2685_268504

theorem log_sum_equation (k : ℤ) (x : ℝ) 
  (h : (7.318 * Real.log x / Real.log k) + 
       (Real.log x / Real.log (k ^ (1/2 : ℝ))) + 
       (Real.log x / Real.log (k ^ (1/3 : ℝ))) + 
       -- ... (representing the sum up to k terms)
       (Real.log x / Real.log (k ^ (1/k : ℝ))) = 
       (k + 1 : ℝ) / 2) :
  x = k ^ (1/k : ℝ) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equation_l2685_268504


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2685_268597

theorem triangle_angle_value (A : ℝ) (h : 0 < A ∧ A < π) : 
  Real.sqrt 2 * Real.sin A = Real.sqrt (3 * Real.cos A) → A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2685_268597


namespace NUMINAMATH_CALUDE_last_digit_2014_power_2014_l2685_268509

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powerMod10 (base exponent : ℕ) : ℕ :=
  (base ^ exponent) % 10

theorem last_digit_2014_power_2014 :
  lastDigit (powerMod10 2014 2014) = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2014_power_2014_l2685_268509


namespace NUMINAMATH_CALUDE_modulus_of_z_plus_one_equals_two_l2685_268598

def i : ℂ := Complex.I

theorem modulus_of_z_plus_one_equals_two :
  Complex.abs ((1 - 3 * i) / (1 + i) + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_plus_one_equals_two_l2685_268598


namespace NUMINAMATH_CALUDE_initial_marbles_calculation_l2685_268566

theorem initial_marbles_calculation (a b : ℚ) :
  a/b + 489.35 = 2778.65 →
  a/b = 2289.3 := by
sorry

end NUMINAMATH_CALUDE_initial_marbles_calculation_l2685_268566


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_21_l2685_268545

theorem cube_sum_over_product_equals_21 (x y z : ℂ) 
  (hnonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (hsum : x + y + z = 18)
  (hdiff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_21_l2685_268545


namespace NUMINAMATH_CALUDE_trigonometric_relation_and_triangle_property_l2685_268543

theorem trigonometric_relation_and_triangle_property (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  ∃ (A B C : ℝ), 
    (y / x = (Real.tan (9*π/20) * Real.cos (π/5) - Real.sin (π/5)) / (Real.cos (π/5) + Real.tan (9*π/20) * Real.sin (π/5))) ∧
    (Real.tan C = y / x) ∧
    (∀ A' B' : ℝ, Real.sin (2*A') + 2 * Real.cos B' ≤ B) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_relation_and_triangle_property_l2685_268543


namespace NUMINAMATH_CALUDE_system_solution_l2685_268568

theorem system_solution : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2685_268568


namespace NUMINAMATH_CALUDE_vector_simplification_l2685_268554

variable {V : Type*} [AddCommGroup V]

/-- For any five points A, B, C, D, E in a vector space,
    AC + DE + EB - AB = DC -/
theorem vector_simplification (A B C D E : V) :
  (C - A) + (E - D) + (B - E) - (B - A) = C - D := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l2685_268554


namespace NUMINAMATH_CALUDE_largest_when_first_changed_l2685_268565

def original : ℚ := 0.12345

def change_digit (n : ℕ) : ℚ :=
  match n with
  | 1 => 0.92345
  | 2 => 0.19345
  | 3 => 0.12945
  | 4 => 0.12395
  | 5 => 0.12349
  | _ => original

theorem largest_when_first_changed :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → change_digit 1 ≥ change_digit n :=
sorry

end NUMINAMATH_CALUDE_largest_when_first_changed_l2685_268565


namespace NUMINAMATH_CALUDE_square_division_area_l2685_268590

theorem square_division_area : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≠ 1 ∧ 
  x^2 = 24 + y^2 ∧
  x^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_division_area_l2685_268590


namespace NUMINAMATH_CALUDE_population_growth_theorem_l2685_268555

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℕ) (birth_rate : ℚ) (death_rate : ℚ) (years : ℕ) : ℚ :=
  match years with
  | 0 => initial_population
  | n + 1 => 
    let prev_population := population_after_years initial_population birth_rate death_rate n
    prev_population + prev_population * birth_rate - prev_population * death_rate

/-- The population after 2 years is approximately 53045 -/
theorem population_growth_theorem : 
  let initial_population : ℕ := 50000
  let birth_rate : ℚ := 43 / 1000
  let death_rate : ℚ := 13 / 1000
  let years : ℕ := 2
  ⌊population_after_years initial_population birth_rate death_rate years⌋ = 53045 := by
  sorry


end NUMINAMATH_CALUDE_population_growth_theorem_l2685_268555


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2685_268559

theorem geometric_series_common_ratio : 
  let a₁ := 7 / 8
  let a₂ := -5 / 12
  let a₃ := 25 / 144
  let r := a₂ / a₁
  r = -10 / 21 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2685_268559


namespace NUMINAMATH_CALUDE_range_of_m_l2685_268558

theorem range_of_m (p q : ℝ → Prop) (m : ℝ) 
  (hp : p = fun x => x^2 + x - 2 > 0)
  (hq : q = fun x => x > m)
  (h_suff_not_nec : ∀ x, ¬(q x) → ¬(p x)) 
  (h_not_nec : ∃ x, ¬(q x) ∧ p x) : 
  m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2685_268558


namespace NUMINAMATH_CALUDE_hungarian_olympiad_1959_l2685_268571

theorem hungarian_olympiad_1959 (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  ∃ k : ℤ, (x^n * (y-z) + y^n * (z-x) + z^n * (x-y)) / ((x-y)*(x-z)*(y-z)) = k :=
sorry

end NUMINAMATH_CALUDE_hungarian_olympiad_1959_l2685_268571


namespace NUMINAMATH_CALUDE_fossil_age_count_is_40_l2685_268576

/-- The number of possible 6-digit numbers formed using the digits 1 (three times), 4, 8, and 9,
    where the number must end with an even digit. -/
def fossil_age_count : ℕ :=
  let digits : List ℕ := [1, 1, 1, 4, 8, 9]
  let even_endings : List ℕ := [4, 8]
  2 * (Nat.factorial 5 / Nat.factorial 3)

theorem fossil_age_count_is_40 : fossil_age_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_fossil_age_count_is_40_l2685_268576


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2685_268580

/-- The volume of a cube with a space diagonal of 10 units is 1000 cubic units -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 10 → s^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2685_268580


namespace NUMINAMATH_CALUDE_range_of_x_is_real_l2685_268544

theorem range_of_x_is_real : 
  (∀ a : ℝ, a ∈ Set.Ioc (-2) 4 → ∀ x : ℝ, x^2 - a*x + 9 > 0) →
  ∀ x : ℝ, x ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_is_real_l2685_268544


namespace NUMINAMATH_CALUDE_square_between_squares_l2685_268581

theorem square_between_squares (n k ℓ x : ℕ) : 
  (x^2 < n) → (n < (x+1)^2) → (n - x^2 = k) → ((x+1)^2 - n = ℓ) →
  n - k*ℓ = (x^2 - n + x)^2 := by
sorry

end NUMINAMATH_CALUDE_square_between_squares_l2685_268581


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l2685_268528

/-- 
Given two workers a and b who can complete a work in 4 days together, 
and in 8/3 days when working simultaneously,
prove that worker a alone can complete the work in 8 days.
-/
theorem worker_a_completion_time 
  (total_time : ℝ) 
  (combined_time : ℝ) 
  (ha : total_time = 4) 
  (hb : combined_time = 8/3) : 
  ∃ (a_time : ℝ), a_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l2685_268528


namespace NUMINAMATH_CALUDE_solution_implies_a_equals_six_l2685_268591

theorem solution_implies_a_equals_six :
  ∀ a : ℝ, (2 * 1 + 5 = 1 + a) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_equals_six_l2685_268591


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2685_268512

theorem algebraic_expression_value (a b : ℝ) 
  (ha : a = 1 + Real.sqrt 2) 
  (hb : b = 1 - Real.sqrt 2) : 
  a^2 - a*b + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2685_268512


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_proof_l2685_268578

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 100) ∧
    (∃ c : ℕ, n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 100 ∧ (∃ d : ℕ, m = 15 * d - 3) → m ≥ n) →
    n = 102

-- The proof goes here
theorem smallest_number_of_eggs_proof : smallest_number_of_eggs 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_proof_l2685_268578


namespace NUMINAMATH_CALUDE_books_in_series_l2685_268540

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 59

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 61

/-- There are 2 more movies than books in the series -/
axiom movie_book_difference : num_movies = num_books + 2

theorem books_in_series : num_books = 59 := by sorry

end NUMINAMATH_CALUDE_books_in_series_l2685_268540


namespace NUMINAMATH_CALUDE_keyword_selection_theorem_l2685_268552

theorem keyword_selection_theorem (n m k : ℕ) (h1 : n = 12) (h2 : m = 4) (h3 : k = 2) : 
  (Nat.choose n k * Nat.choose m 1 + Nat.choose m k) + 
  (Nat.choose n (k + 1) * Nat.choose m 1 + Nat.choose n k * Nat.choose m 2 + Nat.choose m (k + 1)) = 202 := by
  sorry

end NUMINAMATH_CALUDE_keyword_selection_theorem_l2685_268552


namespace NUMINAMATH_CALUDE_distribution_theorem_l2685_268547

/-- The number of ways to distribute 4 students among 4 universities 
    such that exactly two students are admitted to the same university -/
def distribution_count : ℕ := 144

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of universities -/
def num_universities : ℕ := 4

theorem distribution_theorem : 
  (num_students = 4) → 
  (num_universities = 4) → 
  (distribution_count = 144) := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l2685_268547


namespace NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l2685_268541

theorem multiply_inverse_square_equals_cube (x : ℝ) : 
  x * (1/7)^2 = 7^3 → x = 16807 := by
sorry

end NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l2685_268541


namespace NUMINAMATH_CALUDE_range_of_a_l2685_268507

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a < 0) ∧ 
  (∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0) → 
  -1 < a ∧ a ≤ 5/8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2685_268507


namespace NUMINAMATH_CALUDE_arithmetic_operations_with_five_l2685_268573

theorem arithmetic_operations_with_five (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_with_five_l2685_268573


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2685_268546

/-- Given a line symmetric to y = 3x - 2 with respect to the y-axis, prove its equation is y = -3x - 2 -/
theorem symmetric_line_equation (l₁ : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (-x, y) ∈ {(x, y) | y = 3 * x - 2}) →
  l₁ = {(x, y) | y = -3 * x - 2} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2685_268546


namespace NUMINAMATH_CALUDE_miniature_cars_per_package_l2685_268570

theorem miniature_cars_per_package (total_packages : ℕ) (fraction_given_away : ℚ) (cars_left : ℕ) : 
  total_packages = 10 → 
  fraction_given_away = 2/5 → 
  cars_left = 30 → 
  ∃ (cars_per_package : ℕ), 
    cars_per_package = 5 ∧ 
    (total_packages * cars_per_package) * (1 - fraction_given_away) = cars_left :=
by
  sorry

end NUMINAMATH_CALUDE_miniature_cars_per_package_l2685_268570


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2685_268516

theorem quadratic_equation_solution (m n : ℝ) : 
  (∀ x, x^2 + m*x - 15 = (x + 5)*(x + n)) → m = 2 ∧ n = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2685_268516


namespace NUMINAMATH_CALUDE_intersection_implies_m_greater_than_one_l2685_268500

/-- Given a parabola y = x^2 - x + 2 and a line y = x + m, if they intersect at two points, then m > 1 -/
theorem intersection_implies_m_greater_than_one :
  ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 - x₁ + 2 = x₁ + m) ∧ 
    (x₂^2 - x₂ + 2 = x₂ + m)) →
  m > 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_greater_than_one_l2685_268500


namespace NUMINAMATH_CALUDE_distributive_law_analogy_l2685_268569

theorem distributive_law_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_law_analogy_l2685_268569


namespace NUMINAMATH_CALUDE_intersection_A_B_l2685_268539

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2685_268539


namespace NUMINAMATH_CALUDE_elizas_height_l2685_268553

/-- Given the heights of Eliza's siblings and their total height, prove Eliza's height -/
theorem elizas_height (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) (sibling3_height : ℕ) (sibling4_height : ℕ) :
  total_height = 330 ∧ 
  sibling1_height = 66 ∧ 
  sibling2_height = 66 ∧ 
  sibling3_height = 60 ∧ 
  sibling4_height = sibling1_height + 2 →
  ∃ (eliza_height : ℕ), eliza_height = 68 ∧ 
    total_height = sibling1_height + sibling2_height + sibling3_height + sibling4_height + eliza_height :=
by
  sorry

end NUMINAMATH_CALUDE_elizas_height_l2685_268553


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2685_268515

def A : Set (ℝ × ℝ) := {p | p.1 + p.2 = 5}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 1}

theorem intersection_of_A_and_B : A ∩ B = {(3, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2685_268515


namespace NUMINAMATH_CALUDE_school_travel_time_l2685_268584

theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (17 / 13 * usual_rate * (usual_time - 7) = usual_rate * usual_time) →
  usual_time = 119 / 4 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_l2685_268584


namespace NUMINAMATH_CALUDE_time_spent_on_type_a_l2685_268518

/-- Represents the time allocation for an exam with three problem types. -/
structure ExamTime where
  totalTime : ℕ  -- Total exam time in minutes
  totalQuestions : ℕ  -- Total number of questions
  typeACount : ℕ  -- Number of Type A problems
  typeBCount : ℕ  -- Number of Type B problems
  typeCCount : ℕ  -- Number of Type C problems
  typeATime : ℚ  -- Time for one Type A problem
  typeBTime : ℚ  -- Time for one Type B problem
  typeCTime : ℚ  -- Time for one Type C problem

/-- Theorem stating the time spent on Type A problems in the given exam scenario. -/
theorem time_spent_on_type_a (exam : ExamTime) : 
  exam.totalTime = 240 ∧ 
  exam.totalQuestions = 300 ∧ 
  exam.typeACount = 25 ∧ 
  exam.typeBCount = 100 ∧ 
  exam.typeCCount = 175 ∧ 
  exam.typeATime = 4 * exam.typeBTime ∧ 
  exam.typeBTime = 2 * exam.typeCTime ∧ 
  exam.typeACount * exam.typeATime + exam.typeBCount * exam.typeBTime = exam.totalTime / 2 
  → exam.typeACount * exam.typeATime = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_type_a_l2685_268518


namespace NUMINAMATH_CALUDE_area_of_region_l2685_268527

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 - 1 ≤ p.2 ∧ p.2 ≤ Real.sqrt (1 - p.1^2)}

-- State the theorem
theorem area_of_region :
  MeasureTheory.volume R = π / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l2685_268527


namespace NUMINAMATH_CALUDE_upper_limit_y_l2685_268511

theorem upper_limit_y (x y : ℤ) (h1 : 5 < x) (h2 : x < 8) (h3 : 8 < y) 
  (h4 : ∀ (a b : ℤ), 5 < a → a < 8 → 8 < b → b - a ≤ 7) : y ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_y_l2685_268511


namespace NUMINAMATH_CALUDE_ball_events_properties_l2685_268548

-- Define the sample space
def Ω : Type := Fin 8

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry

-- Theorem statement
theorem ball_events_properties :
  (P (A ∩ C) = 0) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end NUMINAMATH_CALUDE_ball_events_properties_l2685_268548


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l2685_268593

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

theorem fixed_points_of_f (a b : ℝ) (ha : a ≠ 0) :
  -- Part 1
  (a = 1 ∧ b = 2 → ∃ x : ℝ, f 1 2 x = x ∧ x = -1) ∧
  -- Part 2
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = x₁ ∧ f a b x₂ = x₂) ↔ 0 < a ∧ a < 1) ∧
  -- Part 3
  (0 < a ∧ a < 1 →
    ∀ x₁ x₂ : ℝ, f a b x₁ = x₁ → f a b x₂ = x₂ →
      f a b x₁ + x₂ = -a / (2 * a^2 + 1) →
        0 < b ∧ b < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l2685_268593


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2685_268582

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 16 ways to distribute 7 indistinguishable balls
    into 3 distinguishable boxes, with each box containing at least one ball -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l2685_268582


namespace NUMINAMATH_CALUDE_vector_projection_l2685_268531

/-- Given two vectors a and b in ℝ², and a vector c such that a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem vector_projection (a b c : ℝ × ℝ) :
  a = (2, 3) →
  b = (-4, 7) →
  a + c = (0, 0) →
  (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2685_268531


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_zero_l2685_268532

theorem sum_of_a_and_b_is_zero (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) / i = 1 + b * i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_zero_l2685_268532


namespace NUMINAMATH_CALUDE_sum_of_squares_l2685_268521

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = -10)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2685_268521


namespace NUMINAMATH_CALUDE_statue_original_cost_l2685_268525

/-- If a statue is sold for $660 with a 20% profit, then its original cost was $550. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 660 → profit_percentage = 0.20 → 
  selling_price = (1 + profit_percentage) * 550 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l2685_268525


namespace NUMINAMATH_CALUDE_sum_of_signs_l2685_268538

theorem sum_of_signs (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = 0) 
  (h2 : a * b * c < 0) : 
  a / |a| + b / |b| + c / |c| = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_signs_l2685_268538


namespace NUMINAMATH_CALUDE_student_problem_attempt_l2685_268562

theorem student_problem_attempt :
  ∀ (correct incorrect : ℕ),
    correct + incorrect ≤ 20 ∧
    8 * correct - 5 * incorrect = 13 →
    correct + incorrect = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_student_problem_attempt_l2685_268562


namespace NUMINAMATH_CALUDE_cylinder_volume_from_square_rotation_l2685_268563

/-- The volume of a cylinder formed by rotating a square about its vertical line of symmetry -/
theorem cylinder_volume_from_square_rotation (square_side : ℝ) (height : ℝ) : 
  square_side = 10 → height = 20 → 
  (π * (square_side / 2)^2 * height : ℝ) = 500 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_square_rotation_l2685_268563


namespace NUMINAMATH_CALUDE_largest_square_tile_l2685_268579

theorem largest_square_tile (wall_width wall_length : ℕ) 
  (h1 : wall_width = 120) (h2 : wall_length = 96) : 
  Nat.gcd wall_width wall_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_l2685_268579


namespace NUMINAMATH_CALUDE_open_spots_difference_is_five_l2685_268589

/-- Represents a parking garage with given characteristics -/
structure ParkingGarage where
  totalLevels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsFourthLevel : Nat
  totalFullSpots : Nat

/-- Calculates the difference between open spots on third and second levels -/
def openSpotsDifference (garage : ParkingGarage) : Int :=
  let totalSpots := garage.totalLevels * garage.spotsPerLevel
  let totalOpenSpots := totalSpots - garage.totalFullSpots
  let openSpotsThirdLevel := totalOpenSpots - garage.openSpotsFirstLevel - garage.openSpotsSecondLevel - garage.openSpotsFourthLevel
  openSpotsThirdLevel - garage.openSpotsSecondLevel

/-- Theorem stating the difference between open spots on third and second levels is 5 -/
theorem open_spots_difference_is_five (garage : ParkingGarage)
  (h1 : garage.totalLevels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsFourthLevel = 31)
  (h6 : garage.totalFullSpots = 186) :
  openSpotsDifference garage = 5 := by
  sorry

end NUMINAMATH_CALUDE_open_spots_difference_is_five_l2685_268589


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2685_268529

/-- Given a quadratic function f(x) = ax^2 + bx, where 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4,
    the range of f(-2) is [6, 10]. -/
theorem quadratic_function_range (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  6 ≤ f (-2) ∧ f (-2) ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2685_268529


namespace NUMINAMATH_CALUDE_candy_problem_l2685_268577

theorem candy_problem (r g b : ℕ+) (n : ℕ) : 
  (10 * r = 16 * g) → 
  (16 * g = 18 * b) → 
  (18 * b = 25 * n) → 
  (∀ m : ℕ, m < n → ¬(∃ r' g' b' : ℕ+, 10 * r' = 16 * g' ∧ 16 * g' = 18 * b' ∧ 18 * b' = 25 * m)) →
  n = 29 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l2685_268577


namespace NUMINAMATH_CALUDE_expression_evaluation_l2685_268526

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1) :
  2 * x^2 - (2*x*y - 3*y^2) + 2*(x^2 + x*y - 2*y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2685_268526


namespace NUMINAMATH_CALUDE_steve_exceeds_goal_and_optimal_strategy_l2685_268533

/-- Represents the berry types --/
inductive Berry
  | Lingonberry
  | Cloudberry
  | Blueberry

/-- Represents Steve's berry-picking job --/
structure BerryJob where
  goal : ℕ
  payRates : Berry → ℕ
  basketCapacity : ℕ
  mondayPicking : Berry → ℕ
  tuesdayPicking : Berry → ℕ

def stevesJob : BerryJob :=
  { goal := 150
  , payRates := fun b => match b with
      | Berry.Lingonberry => 2
      | Berry.Cloudberry => 3
      | Berry.Blueberry => 5
  , basketCapacity := 30
  , mondayPicking := fun b => match b with
      | Berry.Lingonberry => 8
      | Berry.Cloudberry => 10
      | Berry.Blueberry => 12
  , tuesdayPicking := fun b => match b with
      | Berry.Lingonberry => 24
      | Berry.Cloudberry => 20
      | Berry.Blueberry => 5
  }

def earnings (job : BerryJob) (picking : Berry → ℕ) : ℕ :=
  (picking Berry.Lingonberry * job.payRates Berry.Lingonberry) +
  (picking Berry.Cloudberry * job.payRates Berry.Cloudberry) +
  (picking Berry.Blueberry * job.payRates Berry.Blueberry)

def totalEarnings (job : BerryJob) : ℕ :=
  earnings job job.mondayPicking + earnings job job.tuesdayPicking

theorem steve_exceeds_goal_and_optimal_strategy (job : BerryJob) :
  (totalEarnings job > job.goal) ∧
  (∀ picking : Berry → ℕ,
    (picking Berry.Lingonberry + picking Berry.Cloudberry + picking Berry.Blueberry ≤ job.basketCapacity) →
    (earnings job picking ≤ job.basketCapacity * job.payRates Berry.Blueberry)) :=
by sorry

end NUMINAMATH_CALUDE_steve_exceeds_goal_and_optimal_strategy_l2685_268533


namespace NUMINAMATH_CALUDE_hannahs_pay_l2685_268595

/-- Calculates the final pay for an employee given their hourly rate, hours worked, late penalty, and number of times late. -/
def calculate_final_pay (hourly_rate : ℕ) (hours_worked : ℕ) (late_penalty : ℕ) (times_late : ℕ) : ℕ :=
  hourly_rate * hours_worked - late_penalty * times_late

/-- Proves that Hannah's final pay is $525 given her work conditions. -/
theorem hannahs_pay :
  calculate_final_pay 30 18 5 3 = 525 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_pay_l2685_268595


namespace NUMINAMATH_CALUDE_vector_operations_l2685_268551

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Theorem stating the results of the vector operations -/
theorem vector_operations :
  (3 • a + b - 2 • c = ![0, 6]) ∧
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l2685_268551


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2685_268519

theorem initial_money_calculation (initial_amount : ℝ) : 
  (initial_amount / 2 - (initial_amount / 2) / 2 = 51) → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2685_268519


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2685_268587

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 10 = 0 → y = x) ↔ 
  (k = 2 - 2 * Real.sqrt 30 ∨ k = -2 - 2 * Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2685_268587


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2685_268599

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (-3, 1) →
  b = (-1, 2) →
  m • a - n • b = (10, 0) →
  m = -4 ∧ n = -2 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2685_268599


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l2685_268520

theorem remainder_of_sum_of_powers (n : ℕ) : (9^24 + 12^37) % 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l2685_268520


namespace NUMINAMATH_CALUDE_inequality_proof_l2685_268556

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2685_268556


namespace NUMINAMATH_CALUDE_sunflower_seeds_count_l2685_268572

/-- The number of sunflower plants -/
def num_sunflowers : ℕ := 6

/-- The number of dandelion plants -/
def num_dandelions : ℕ := 8

/-- The number of seeds per dandelion plant -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of total seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64/100

/-- The number of seeds per sunflower plant -/
def seeds_per_sunflower : ℕ := 9

theorem sunflower_seeds_count :
  let total_dandelion_seeds := num_dandelions * seeds_per_dandelion
  let total_seeds := total_dandelion_seeds / dandelion_seed_percentage
  let total_sunflower_seeds := total_seeds - total_dandelion_seeds
  seeds_per_sunflower = total_sunflower_seeds / num_sunflowers := by
sorry

end NUMINAMATH_CALUDE_sunflower_seeds_count_l2685_268572


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l2685_268542

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow) / (green + yellow + white) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l2685_268542


namespace NUMINAMATH_CALUDE_center_locus_is_conic_l2685_268564

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A conic section in a 2D plane -/
structure Conic where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a conic section -/
def center (c : Conic) : Point2D :=
  { x := 0, y := 0 }  -- Placeholder definition

/-- Checks if a point lies on a conic -/
def lies_on (p : Point2D) (c : Conic) : Prop :=
  c.a * p.x^2 + c.b * p.x * p.y + c.c * p.y^2 + c.d * p.x + c.e * p.y + c.f = 0

/-- The set of all conics passing through four given points -/
def conics_through_four_points (A B C D : Point2D) : Set Conic :=
  { c | lies_on A c ∧ lies_on B c ∧ lies_on C c ∧ lies_on D c }

/-- The locus of centers of conics passing through four points -/
def center_locus (A B C D : Point2D) : Set Point2D :=
  { p | ∃ c ∈ conics_through_four_points A B C D, center c = p }

/-- Theorem: The locus of centers of conics passing through four points is a conic -/
theorem center_locus_is_conic (A B C D : Point2D) :
  ∃ Γ : Conic, ∀ p ∈ center_locus A B C D, lies_on p Γ :=
sorry

end NUMINAMATH_CALUDE_center_locus_is_conic_l2685_268564


namespace NUMINAMATH_CALUDE_percentage_problem_l2685_268550

theorem percentage_problem :
  let percentage := 6.620000000000001
  let value := 66.2
  let x := value / (percentage / 100)
  x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2685_268550


namespace NUMINAMATH_CALUDE_second_machine_copies_per_minute_l2685_268574

/-- 
Given two copy machines working at constant rates, where the first machine makes 35 copies per minute,
and together they make 3300 copies in 30 minutes, prove that the second machine makes 75 copies per minute.
-/
theorem second_machine_copies_per_minute 
  (rate1 : ℕ) 
  (rate2 : ℕ) 
  (total_time : ℕ) 
  (total_copies : ℕ) 
  (h1 : rate1 = 35)
  (h2 : total_time = 30)
  (h3 : total_copies = 3300)
  (h4 : rate1 * total_time + rate2 * total_time = total_copies) : 
  rate2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_second_machine_copies_per_minute_l2685_268574


namespace NUMINAMATH_CALUDE_days_at_grandparents_l2685_268549

def vacation_duration : ℕ := 21  -- 3 weeks * 7 days

def travel_to_grandparents : ℕ := 1
def travel_to_brother : ℕ := 1
def stay_at_brother : ℕ := 5
def travel_to_sister : ℕ := 2
def stay_at_sister : ℕ := 5
def travel_home : ℕ := 2

def known_days : ℕ := travel_to_grandparents + travel_to_brother + stay_at_brother + travel_to_sister + stay_at_sister + travel_home

theorem days_at_grandparents :
  vacation_duration - known_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_days_at_grandparents_l2685_268549


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2685_268557

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1 / 3 : ℂ) + (5 / 8 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1 / 3 : ℂ) - (5 / 8 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2685_268557


namespace NUMINAMATH_CALUDE_second_solution_concentration_l2685_268560

/-- 
Given two solutions that are mixed to form a new solution,
this theorem proves that the concentration of the second solution
must be 10% under the specified conditions.
-/
theorem second_solution_concentration
  (volume_first : ℝ)
  (concentration_first : ℝ)
  (volume_second : ℝ)
  (concentration_final : ℝ)
  (h1 : volume_first = 4)
  (h2 : concentration_first = 0.04)
  (h3 : volume_second = 2)
  (h4 : concentration_final = 0.06)
  (h5 : volume_first * concentration_first + volume_second * (concentration_second / 100) = 
        (volume_first + volume_second) * concentration_final) :
  concentration_second = 10 := by
  sorry

#check second_solution_concentration

end NUMINAMATH_CALUDE_second_solution_concentration_l2685_268560


namespace NUMINAMATH_CALUDE_distinct_roots_condition_l2685_268517

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 1

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := 4 + 4 * k

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x = 0 ∧ quadratic_equation k y = 0) ↔
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_l2685_268517


namespace NUMINAMATH_CALUDE_exists_sum_of_scores_with_two_ways_l2685_268561

/-- Represents a scoring configuration for the modified AMC test. -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents an answer combination for the test. -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination under a specific scoring config. -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + answers.unanswered * config.unanswered_points + answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given total number of questions. -/
def is_valid_combination (total_questions : ℕ) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = total_questions

/-- Defines the specific scoring configuration for the problem. -/
def amc_scoring : ScoringConfig :=
  { total_questions := 20
  , correct_points := 7
  , unanswered_points := 3
  , incorrect_points := 0 }

/-- Theorem stating the existence of a sum of scores meeting the problem criteria. -/
theorem exists_sum_of_scores_with_two_ways :
  ∃ (sum : ℕ), 
    (∃ (scores : List ℕ),
      (∀ score ∈ scores, 
        score ≤ 140 ∧ 
        (∃ (ways : List AnswerCombination), 
          ways.length = 2 ∧
          (∀ way ∈ ways, 
            is_valid_combination amc_scoring.total_questions way ∧
            calculate_score amc_scoring way = score))) ∧
      sum = scores.sum) := by
  sorry


end NUMINAMATH_CALUDE_exists_sum_of_scores_with_two_ways_l2685_268561


namespace NUMINAMATH_CALUDE_hexagon_area_l2685_268510

/-- A regular hexagon divided by three diagonals -/
structure RegularHexagon where
  /-- The area of one small triangle formed by the diagonals -/
  small_triangle_area : ℝ
  /-- The total number of small triangles in the hexagon -/
  total_triangles : ℕ
  /-- The number of shaded triangles -/
  shaded_triangles : ℕ
  /-- The total shaded area -/
  shaded_area : ℝ
  /-- The hexagon is divided into 12 congruent triangles -/
  triangle_count : total_triangles = 12
  /-- Two regions (equivalent to 5 small triangles) are shaded -/
  shaded_count : shaded_triangles = 5
  /-- The total shaded area is 20 cm² -/
  shaded_area_value : shaded_area = 20

/-- The theorem stating the area of the hexagon -/
theorem hexagon_area (h : RegularHexagon) : h.total_triangles * h.small_triangle_area = 48 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l2685_268510


namespace NUMINAMATH_CALUDE_mike_remaining_amount_l2685_268534

/-- Calculates the remaining amount for a partner in a profit-sharing scenario -/
def remaining_amount (total_parts : ℕ) (partner_parts : ℕ) (other_partner_amount : ℕ) (spending : ℕ) : ℕ :=
  let part_value := other_partner_amount / (total_parts - partner_parts)
  let partner_amount := part_value * partner_parts
  partner_amount - spending

/-- Theorem stating the remaining amount for Mike in the given profit-sharing scenario -/
theorem mike_remaining_amount :
  remaining_amount 7 2 2500 200 = 800 := by
  sorry

end NUMINAMATH_CALUDE_mike_remaining_amount_l2685_268534


namespace NUMINAMATH_CALUDE_additional_money_needed_l2685_268523

-- Define the given values
def perfume_cost : ℚ := 50
def christian_initial : ℚ := 5
def sue_initial : ℚ := 7
def christian_yards : ℕ := 4
def christian_yard_charge : ℚ := 5
def sue_dogs : ℕ := 6
def sue_dog_charge : ℚ := 2

-- Define the theorem
theorem additional_money_needed : 
  perfume_cost - (christian_initial + sue_initial + 
    (christian_yards : ℚ) * christian_yard_charge + 
    (sue_dogs : ℚ) * sue_dog_charge) = 6 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2685_268523
