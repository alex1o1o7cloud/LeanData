import Mathlib

namespace NUMINAMATH_CALUDE_diana_wins_probability_l803_80353

-- Define the type for a die roll (1 to 6)
def DieRoll : Type := Fin 6

-- Define the type for a pair of dice rolls
def DicePair : Type := DieRoll × DieRoll

-- Function to calculate the sum of a pair of dice rolls
def diceSum (pair : DicePair) : Nat :=
  pair.1.val + pair.2.val + 2

-- Define the sample space of all possible outcomes
def sampleSpace : Finset (DicePair × DicePair) :=
  sorry

-- Define the event where Diana's sum exceeds Apollo's by at least 2
def favorableEvent : Finset (DicePair × DicePair) :=
  sorry

-- Theorem to prove
theorem diana_wins_probability :
  (favorableEvent.card : Rat) / sampleSpace.card = 47 / 432 := by
  sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l803_80353


namespace NUMINAMATH_CALUDE_white_washing_cost_l803_80331

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem white_washing_cost
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_cost_per_sqft : cost_per_sqft = 10) :
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + num_windows * window_length * window_width)) * cost_per_sqft = 9060 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_l803_80331


namespace NUMINAMATH_CALUDE_chocolates_in_cost_price_l803_80308

/-- The number of chocolates in the cost price -/
def n : ℕ := sorry

/-- The cost price of one chocolate -/
def C : ℝ := sorry

/-- The selling price of one chocolate -/
def S : ℝ := sorry

/-- The cost price of n chocolates equals the selling price of 16 chocolates -/
axiom cost_price_eq_selling_price : n * C = 16 * S

/-- The gain percent is 50% -/
axiom gain_percent : S = 1.5 * C

theorem chocolates_in_cost_price : n = 24 := by sorry

end NUMINAMATH_CALUDE_chocolates_in_cost_price_l803_80308


namespace NUMINAMATH_CALUDE_product_modulo_seven_l803_80386

theorem product_modulo_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_seven_l803_80386


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l803_80344

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    (R.1 - P.1)^2 + (R.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2

-- State the theorem
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' →
      Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) ≤ 
      Real.sqrt ((Q'.1 - P'.1)^2 + (Q'.2 - P'.2)^2) ∧
    Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = Real.sqrt 207 + Real.sqrt 132 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l803_80344


namespace NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l803_80328

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

def numbers_to_check : List ℕ := [5, 7, 9, 10, 11]

theorem only_eleven_not_sum_of_two_primes :
  ∀ n ∈ numbers_to_check, n ≠ 11 → is_sum_of_two_primes n ∧
  ¬(is_sum_of_two_primes 11) :=
sorry

end NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l803_80328


namespace NUMINAMATH_CALUDE_closest_to_fraction_l803_80337

def options : List ℝ := [500, 1000, 2000, 2100, 4000]

theorem closest_to_fraction (options : List ℝ) :
  2100 = (options.filter (λ x => ∀ y ∈ options, |850 / 0.42 - x| ≤ |850 / 0.42 - y|)).head! :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l803_80337


namespace NUMINAMATH_CALUDE_rhombus_area_scaling_l803_80390

theorem rhombus_area_scaling (d1 d2 : ℝ) :
  d1 > 0 → d2 > 0 → (d1 * d2) / 2 = 3 → ((5 * d1) * (5 * d2)) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_scaling_l803_80390


namespace NUMINAMATH_CALUDE_smallest_divisible_by_72_l803_80334

/-- Concatenates the digits of all positive integers from 1 to n -/
def concatenateDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_divisible_by_72 :
  ∃ (n : ℕ), n > 0 ∧ isDivisibleBy (concatenateDigits n) 72 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬isDivisibleBy (concatenateDigits m) 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_72_l803_80334


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l803_80322

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l803_80322


namespace NUMINAMATH_CALUDE_museum_artifacts_per_wing_l803_80355

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_paintings : ℕ
  small_painting_wings : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing -/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_paintings + m.small_painting_wings * m.small_paintings_per_wing
  let total_artifacts := total_paintings * m.artifact_multiplier
  let artifact_wings := m.total_wings - m.painting_wings
  (total_artifacts + artifact_wings - 1) / artifact_wings

/-- Theorem stating the number of artifacts in each artifact wing for the given museum -/
theorem museum_artifacts_per_wing :
  let m : Museum := {
    total_wings := 16,
    painting_wings := 6,
    large_paintings := 2,
    small_painting_wings := 4,
    small_paintings_per_wing := 20,
    artifact_multiplier := 8
  }
  artifacts_per_wing m = 66 := by sorry

end NUMINAMATH_CALUDE_museum_artifacts_per_wing_l803_80355


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l803_80366

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I)^2 = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l803_80366


namespace NUMINAMATH_CALUDE_log_equation_solution_l803_80359

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l803_80359


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l803_80332

theorem opposite_numbers_equation (a b : ℝ) : a + b = 0 → a - (2 - b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l803_80332


namespace NUMINAMATH_CALUDE_blocks_remaining_l803_80368

theorem blocks_remaining (initial_blocks : ℕ) (first_tower : ℕ) (second_tower : ℕ)
  (h1 : initial_blocks = 78)
  (h2 : first_tower = 19)
  (h3 : second_tower = 25) :
  initial_blocks - first_tower - second_tower = 34 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_l803_80368


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l803_80306

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n ≥ 1 -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsGeometric a →
  (a 1)^2 - 2*(a 1) - 3 = 0 →
  (a 4)^2 - 2*(a 4) - 3 = 0 →
  a 2 * a 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l803_80306


namespace NUMINAMATH_CALUDE_least_number_satisfying_conditions_l803_80301

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_9 n ∧
  (∀ d : ℕ, 3 ≤ d ∧ d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions :
  satisfies_conditions 6302 ∧
  ∀ m : ℕ, m < 6302 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_number_satisfying_conditions_l803_80301


namespace NUMINAMATH_CALUDE_gregs_shopping_expenditure_l803_80300

theorem gregs_shopping_expenditure (total_spent : ℕ) (shoe_price_difference : ℕ) :
  total_spent = 300 →
  shoe_price_difference = 9 →
  ∃ (shirt_price shoe_price : ℕ),
    shirt_price + shoe_price = total_spent ∧
    shoe_price = 2 * shirt_price + shoe_price_difference ∧
    shirt_price = 97 :=
by sorry

end NUMINAMATH_CALUDE_gregs_shopping_expenditure_l803_80300


namespace NUMINAMATH_CALUDE_one_large_pizza_sufficient_l803_80327

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSizes where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of pizzas ordered for each dietary restriction --/
structure PizzaOrder where
  gluten_free_small : Nat
  dairy_free_medium : Nat
  large : Nat

/-- Calculates if the pizza order is sufficient for both brothers --/
def is_sufficient_order (sizes : PizzaSizes) (order : PizzaOrder) : Prop :=
  let gluten_free_slices := order.gluten_free_small * sizes.small + order.large * sizes.large
  let dairy_free_slices := order.dairy_free_medium * sizes.medium
  gluten_free_slices ≥ 15 ∧ dairy_free_slices ≥ 15

/-- Theorem stating that ordering 1 large pizza is sufficient --/
theorem one_large_pizza_sufficient 
  (sizes : PizzaSizes)
  (h_large : sizes.large = 14)
  (h_medium : sizes.medium = 10)
  (h_small : sizes.small = 8) :
  is_sufficient_order sizes { gluten_free_small := 1, dairy_free_medium := 2, large := 1 } :=
by sorry


end NUMINAMATH_CALUDE_one_large_pizza_sufficient_l803_80327


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l803_80356

/-- The distance from the center of the circle ρ = 4cos θ to the line tan θ = 1 is √2 -/
theorem distance_circle_center_to_line : 
  ∀ (θ : ℝ) (ρ : ℝ → ℝ) (x y : ℝ),
  (ρ θ = 4 * Real.cos θ) →  -- Circle equation
  (Real.tan θ = 1) →        -- Line equation
  (x - 2)^2 + y^2 = 4 →     -- Standard form of circle equation
  x - y = 0 →               -- Line equation in rectangular coordinates
  Real.sqrt 2 = |x - 2| / Real.sqrt ((1:ℝ)^2 + (-1:ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l803_80356


namespace NUMINAMATH_CALUDE_problem_17_l803_80373

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def p (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → log_a a (x + 3) > log_a a (y + 3)

def q (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*a - 3)*x₂ + 1 = 0

theorem problem_17 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi (5/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_17_l803_80373


namespace NUMINAMATH_CALUDE_remaining_tickets_l803_80345

/-- Given an initial number of tickets, the number of tickets lost, and the number of tickets spent,
    the remaining number of tickets is equal to the initial number minus the lost tickets minus the spent tickets. -/
theorem remaining_tickets (initial lost spent : ℝ) : 
  initial - lost - spent = initial - (lost + spent) := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l803_80345


namespace NUMINAMATH_CALUDE_total_time_two_trips_l803_80363

/-- Represents the time in minutes for a round trip to the beauty parlor -/
structure RoundTrip where
  to_parlor : ℕ
  from_parlor : ℕ
  delay : ℕ
  additional_time : ℕ

/-- Calculates the total time for a round trip -/
def total_time (trip : RoundTrip) : ℕ :=
  trip.to_parlor + trip.from_parlor + trip.delay + trip.additional_time

/-- Represents Naomi's two round trips to the beauty parlor -/
def naomi_trips : (RoundTrip × RoundTrip) :=
  ({ to_parlor := 60
   , from_parlor := 120
   , delay := 15
   , additional_time := 10 }
  ,{ to_parlor := 60
   , from_parlor := 120
   , delay := 20
   , additional_time := 30 })

/-- Theorem stating that the total time for both round trips is 435 minutes -/
theorem total_time_two_trips : 
  total_time naomi_trips.1 + total_time naomi_trips.2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_total_time_two_trips_l803_80363


namespace NUMINAMATH_CALUDE_application_methods_for_five_graduates_three_universities_l803_80346

/-- The number of different application methods for high school graduates to universities -/
def application_methods (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: Given 5 high school graduates and 3 universities, where each graduate can only apply to one university, the total number of different application methods is 3^5 -/
theorem application_methods_for_five_graduates_three_universities :
  application_methods 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_application_methods_for_five_graduates_three_universities_l803_80346


namespace NUMINAMATH_CALUDE_quadratic_roots_and_graph_point_l803_80376

theorem quadratic_roots_and_graph_point (a b c : ℝ) (x : ℝ) 
  (h1 : a ≠ 0)
  (h2 : Real.tan x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h3 : Real.tan (π/4 - x) = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  : a * 1^2 + b * 1 - c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_graph_point_l803_80376


namespace NUMINAMATH_CALUDE_john_reading_capacity_l803_80303

/-- Represents the reading speed ratio between John and his brother -/
def johnSpeedRatio : ℝ := 1.6

/-- Time taken by John's brother to read one book (in hours) -/
def brotherReadTime : ℝ := 8

/-- Available time for John to read (in hours) -/
def availableTime : ℝ := 15

/-- Number of books John can read in the available time -/
def johnBooksRead : ℕ := 3

theorem john_reading_capacity : 
  ⌊availableTime / (brotherReadTime / johnSpeedRatio)⌋ = johnBooksRead := by
  sorry

end NUMINAMATH_CALUDE_john_reading_capacity_l803_80303


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l803_80343

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 140 45 30 = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l803_80343


namespace NUMINAMATH_CALUDE_josh_shopping_spending_l803_80391

/-- The problem of calculating Josh's total spending at the shopping center -/
theorem josh_shopping_spending :
  let num_films : ℕ := 9
  let num_books : ℕ := 4
  let num_cds : ℕ := 6
  let cost_per_film : ℕ := 5
  let cost_per_book : ℕ := 4
  let cost_per_cd : ℕ := 3
  let total_spent := 
    num_films * cost_per_film + 
    num_books * cost_per_book + 
    num_cds * cost_per_cd
  total_spent = 79 := by
  sorry

end NUMINAMATH_CALUDE_josh_shopping_spending_l803_80391


namespace NUMINAMATH_CALUDE_parabola_coefficients_l803_80375

/-- A parabola with vertex (4, 3), vertical axis of symmetry, passing through (2, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a * 4^2 + b * 4 + c = 3
  symmetry : b = -2 * a * 4
  point : a * 2^2 + b * 2 + c = 1

/-- The coefficients of the parabola are (-1/2, 4, -5) -/
theorem parabola_coefficients (p : Parabola) : p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l803_80375


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_l803_80388

theorem consecutive_integers_product_812 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_l803_80388


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l803_80325

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬ (9 ∣ (4499 + m))) ∧ (9 ∣ (4499 + n)) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l803_80325


namespace NUMINAMATH_CALUDE_equal_pairwise_products_l803_80309

theorem equal_pairwise_products (n : ℕ) : 
  (¬ ∃ n : ℕ, n > 0 ∧ n < 1000 ∧ n^2 - 1000*n + 499500 = 0) ∧
  (∃ n : ℕ, n > 0 ∧ n < 10000 ∧ n^2 - 10000*n + 49995000 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_pairwise_products_l803_80309


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_forty_l803_80347

/-- If x is 20 percent greater than 40, then x equals 48. -/
theorem twenty_percent_greater_than_forty (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_forty_l803_80347


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l803_80340

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 + x - 12 ≤ 0 ∧ x ≥ -4 → -4 ≤ x ∧ x ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l803_80340


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l803_80311

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (longer_base : ℝ)
  (base_angle : ℝ)

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.longer_base = 20 ∧
    t.base_angle = Real.arcsin 0.6 →
    area t = 72 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l803_80311


namespace NUMINAMATH_CALUDE_parallel_vectors_l803_80372

/-- Two vectors in R² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Given vectors a and b, if they are parallel, then the x-component of a is 1/2 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.2 = 1 ∧ b = (2, 4)) :
  parallel a b → a.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l803_80372


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_proof_short_sleeve_shirts_l803_80319

theorem short_sleeve_shirts_count : ℕ → ℕ → ℕ → Prop :=
  fun total_shirts long_sleeve_shirts short_sleeve_shirts =>
    total_shirts = long_sleeve_shirts + short_sleeve_shirts →
    total_shirts = 30 →
    long_sleeve_shirts = 21 →
    short_sleeve_shirts = 8

-- The proof is omitted
theorem proof_short_sleeve_shirts : short_sleeve_shirts_count 30 21 8 := by
  sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_proof_short_sleeve_shirts_l803_80319


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_second_or_third_quadrant_l803_80317

/-- A linear function f(x) = kx + b is monotonically decreasing on ℝ -/
def is_monotonically_decreasing (k b : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → k * x + b > k * y + b

/-- The point (k, b) is in the second or third quadrant -/
def is_in_second_or_third_quadrant (k b : ℝ) : Prop :=
  k < 0 ∧ (b > 0 ∨ b < 0)

/-- If a linear function y = kx + b is monotonically decreasing on ℝ,
    then the point (k, b) is in the second or third quadrant -/
theorem monotonically_decreasing_implies_second_or_third_quadrant (k b : ℝ) :
  is_monotonically_decreasing k b → is_in_second_or_third_quadrant k b :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_second_or_third_quadrant_l803_80317


namespace NUMINAMATH_CALUDE_antonella_coins_l803_80399

theorem antonella_coins (num_coins : ℕ) (loonie_value toonie_value : ℚ) 
  (frappuccino_cost remaining_money : ℚ) :
  num_coins = 10 →
  loonie_value = 1 →
  toonie_value = 2 →
  frappuccino_cost = 3 →
  remaining_money = 11 →
  ∃ (num_loonies num_toonies : ℕ),
    num_loonies + num_toonies = num_coins ∧
    num_loonies * loonie_value + num_toonies * toonie_value = 
      remaining_money + frappuccino_cost ∧
    num_toonies = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonella_coins_l803_80399


namespace NUMINAMATH_CALUDE_price_comparison_l803_80383

theorem price_comparison (x : ℝ) (h : x > 0) : x * 1.1 * 0.9 < x := by
  sorry

end NUMINAMATH_CALUDE_price_comparison_l803_80383


namespace NUMINAMATH_CALUDE_evening_rice_fraction_l803_80349

/-- 
Given:
- Rose initially has 10 kg of rice
- She cooks 9/10 kg in the morning
- She has 750 g left at the end
Prove that the fraction of remaining rice cooked in the evening is 1/4
-/
theorem evening_rice_fraction (initial_rice : ℝ) (morning_cooked : ℝ) (final_rice : ℝ) :
  initial_rice = 10 →
  morning_cooked = 9/10 →
  final_rice = 750/1000 →
  (initial_rice - morning_cooked - final_rice) / (initial_rice - morning_cooked) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_evening_rice_fraction_l803_80349


namespace NUMINAMATH_CALUDE_smallest_positive_integer_properties_l803_80380

theorem smallest_positive_integer_properties : ∃ a : ℕ, 
  (∀ n : ℕ, n > 0 → a ≤ n) ∧ 
  (a^3 + 1 = 2) ∧ 
  ((a + 1) * (a^2 - a + 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_properties_l803_80380


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l803_80361

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (a = 3) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l803_80361


namespace NUMINAMATH_CALUDE_paperback_cost_is_twelve_l803_80333

/-- Represents the book club's annual fee collection --/
structure BookClub where
  members : ℕ
  snackFeePerMember : ℕ
  hardcoverBooksPerMember : ℕ
  hardcoverBookPrice : ℕ
  paperbackBooksPerMember : ℕ
  totalCollected : ℕ

/-- Calculates the cost per paperback book --/
def costPerPaperback (club : BookClub) : ℚ :=
  let snackTotal := club.members * club.snackFeePerMember
  let hardcoverTotal := club.members * club.hardcoverBooksPerMember * club.hardcoverBookPrice
  let paperbackTotal := club.totalCollected - snackTotal - hardcoverTotal
  paperbackTotal / (club.members * club.paperbackBooksPerMember)

/-- Theorem stating that the cost per paperback book is $12 --/
theorem paperback_cost_is_twelve (club : BookClub) 
    (h1 : club.members = 6)
    (h2 : club.snackFeePerMember = 150)
    (h3 : club.hardcoverBooksPerMember = 6)
    (h4 : club.hardcoverBookPrice = 30)
    (h5 : club.paperbackBooksPerMember = 6)
    (h6 : club.totalCollected = 2412) :
    costPerPaperback club = 12 := by
  sorry


end NUMINAMATH_CALUDE_paperback_cost_is_twelve_l803_80333


namespace NUMINAMATH_CALUDE_least_sum_of_valid_pair_l803_80305

def is_valid_pair (a b : ℕ+) : Prop :=
  Nat.gcd (a + b) 330 = 1 ∧
  (a : ℕ) ^ (a : ℕ) % (b : ℕ) ^ (b : ℕ) = 0 ∧
  (a : ℕ) % (b : ℕ) ≠ 0

theorem least_sum_of_valid_pair :
  ∃ (a b : ℕ+), is_valid_pair a b ∧
    ∀ (a' b' : ℕ+), is_valid_pair a' b' → a + b ≤ a' + b' ∧
    a + b = 357 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_valid_pair_l803_80305


namespace NUMINAMATH_CALUDE_original_profit_percentage_l803_80381

theorem original_profit_percentage (cost selling_price : ℝ) 
  (h1 : cost > 0) 
  (h2 : selling_price > cost) 
  (h3 : selling_price - (1.12 * cost) = 0.552 * selling_price) : 
  (selling_price - cost) / cost = 1.5 := by
sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l803_80381


namespace NUMINAMATH_CALUDE_stratified_sampling_third_grade_l803_80393

def total_students : ℕ := 270000
def third_grade_students : ℕ := 81000
def sample_size : ℕ := 3000

theorem stratified_sampling_third_grade :
  (third_grade_students * sample_size) / total_students = 900 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_grade_l803_80393


namespace NUMINAMATH_CALUDE_warriors_truth_count_l803_80398

theorem warriors_truth_count :
  ∀ (total_warriors : ℕ) 
    (sword_yes spear_yes axe_yes bow_yes : ℕ),
  total_warriors = 33 →
  sword_yes = 13 →
  spear_yes = 15 →
  axe_yes = 20 →
  bow_yes = 27 →
  ∃ (truth_tellers : ℕ),
    truth_tellers = 12 ∧
    truth_tellers + (total_warriors - truth_tellers) * 3 = 
      sword_yes + spear_yes + axe_yes + bow_yes :=
by sorry

end NUMINAMATH_CALUDE_warriors_truth_count_l803_80398


namespace NUMINAMATH_CALUDE_bangles_per_box_l803_80314

def total_pairs : ℕ := 240
def num_boxes : ℕ := 20

theorem bangles_per_box :
  (total_pairs * 2) / num_boxes = 24 := by
  sorry

end NUMINAMATH_CALUDE_bangles_per_box_l803_80314


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l803_80377

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l803_80377


namespace NUMINAMATH_CALUDE_angle_C_measure_l803_80378

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
  sum_180 : A + B + C = 180

-- Theorem statement
theorem angle_C_measure (t : ScaleneTriangle) 
  (h1 : t.B = t.A + 20)  -- Angle B is 20 degrees larger than angle A
  (h2 : t.C = 2 * t.A)   -- Angle C is twice the size of angle A
  : t.C = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l803_80378


namespace NUMINAMATH_CALUDE_insurance_cost_calculation_l803_80316

def apartment_cost : ℝ := 7000000
def loan_amount : ℝ := 4000000
def interest_rate : ℝ := 0.101
def property_insurance_rate : ℝ := 0.0009
def life_health_insurance_female : ℝ := 0.0017
def life_health_insurance_male : ℝ := 0.0019
def title_insurance_rate : ℝ := 0.0027
def svetlana_ratio : ℝ := 0.2
def dmitry_ratio : ℝ := 0.8

def total_insurance_cost : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_insurance := total_loan * property_insurance_rate
  let title_insurance := total_loan * title_insurance_rate
  let svetlana_insurance := total_loan * svetlana_ratio * life_health_insurance_female
  let dmitry_insurance := total_loan * dmitry_ratio * life_health_insurance_male
  property_insurance + title_insurance + svetlana_insurance + dmitry_insurance

theorem insurance_cost_calculation :
  total_insurance_cost = 24045.84 := by sorry

end NUMINAMATH_CALUDE_insurance_cost_calculation_l803_80316


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l803_80384

theorem mean_of_three_numbers (a b c : ℝ) : 
  (a + b + c + 105) / 4 = 90 →
  (a + b + c) / 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l803_80384


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l803_80320

/-- 
Given an article with a selling price of 600 and a cost price of 375,
prove that the profit percentage is 60%.
-/
theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 600)
  (h2 : cost_price = 375) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l803_80320


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l803_80367

/-- The minimum diagonal of a rectangle with perimeter 24 -/
theorem min_diagonal_rectangle (l w : ℝ) (h_perimeter : l + w = 12) :
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
  (∀ (l' w' : ℝ), l' + w' = 12 → Real.sqrt (l'^2 + w'^2) ≥ d) ∧
  d = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l803_80367


namespace NUMINAMATH_CALUDE_chord_length_30_60_l803_80352

theorem chord_length_30_60 : 
  let A : ℝ × ℝ := (Real.cos (π / 6), Real.sin (π / 6))
  let B : ℝ × ℝ := (Real.cos (π / 3), Real.sin (π / 3))
  let chord_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  chord_length = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
  sorry

#check chord_length_30_60

end NUMINAMATH_CALUDE_chord_length_30_60_l803_80352


namespace NUMINAMATH_CALUDE_kelly_gave_away_64_games_l803_80307

/-- The number of games Kelly gave away -/
def games_given_away (initial_games final_games : ℕ) : ℕ :=
  initial_games - final_games

/-- Theorem: Kelly gave away 64 games -/
theorem kelly_gave_away_64_games :
  games_given_away 106 42 = 64 := by
  sorry

end NUMINAMATH_CALUDE_kelly_gave_away_64_games_l803_80307


namespace NUMINAMATH_CALUDE_percentage_non_defective_m3_l803_80397

theorem percentage_non_defective_m3 (m1_percentage : Real) (m2_percentage : Real)
  (m1_defective : Real) (m2_defective : Real) (total_defective : Real) :
  m1_percentage = 0.4 →
  m2_percentage = 0.3 →
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  total_defective = 0.036 →
  ∃ (m3_non_defective : Real),
    m3_non_defective = 0.93 ∧
    m1_percentage * m1_defective + m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end NUMINAMATH_CALUDE_percentage_non_defective_m3_l803_80397


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l803_80315

theorem angle_sum_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_eq : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) :
  α + β = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l803_80315


namespace NUMINAMATH_CALUDE_max_integer_difference_l803_80385

theorem max_integer_difference (x y : ℤ) (hx : -6 < x ∧ x < -2) (hy : 4 < y ∧ y < 10) :
  (∀ (a b : ℤ), -6 < a ∧ a < -2 ∧ 4 < b ∧ b < 10 → b - a ≤ y - x) →
  y - x = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l803_80385


namespace NUMINAMATH_CALUDE_square_of_sum_31_3_l803_80371

theorem square_of_sum_31_3 : 31^2 + 2*(31*3) + 3^2 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_31_3_l803_80371


namespace NUMINAMATH_CALUDE_triangle_properties_l803_80336

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 - b^2 + a*c = 0 →
  (∃ (p : ℝ), p = a + b + c) →
  B = 2*π/3 ∧ (b = 2*Real.sqrt 3 → ∃ (p_max : ℝ), p_max = 4 + 2*Real.sqrt 3 ∧ ∀ p, p ≤ p_max) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l803_80336


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l803_80324

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l803_80324


namespace NUMINAMATH_CALUDE_functional_equation_solution_l803_80395

theorem functional_equation_solution (f g : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l803_80395


namespace NUMINAMATH_CALUDE_projection_a_on_b_l803_80357

def a : ℝ × ℝ := (-8, 1)
def b : ℝ × ℝ := (3, 4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l803_80357


namespace NUMINAMATH_CALUDE_inequality_solution_set_l803_80350

-- Define the set of real numbers satisfying the inequality
def S : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

-- State the theorem
theorem inequality_solution_set : S = Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l803_80350


namespace NUMINAMATH_CALUDE_inequality_not_true_range_l803_80364

theorem inequality_not_true_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - a| + |x - 12| < 6)) ↔ (a ≤ 6 ∨ a ≥ 18) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_range_l803_80364


namespace NUMINAMATH_CALUDE_derivative_at_two_l803_80389

theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 3 * (deriv f 2) * x) : 
  deriv f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l803_80389


namespace NUMINAMATH_CALUDE_first_number_is_30_l803_80358

def fibonacci_like_sequence (a₁ a₂ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | 1 => a₂
  | (n+2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n+1)

theorem first_number_is_30 (a₁ a₂ : ℤ) :
  fibonacci_like_sequence a₁ a₂ 6 = 5 ∧
  fibonacci_like_sequence a₁ a₂ 7 = 14 ∧
  fibonacci_like_sequence a₁ a₂ 8 = 33 →
  a₁ = 30 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_30_l803_80358


namespace NUMINAMATH_CALUDE_correct_calculation_l803_80379

theorem correct_calculation (x : ℤ) : 
  (713 + x = 928) → (713 - x = 498) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l803_80379


namespace NUMINAMATH_CALUDE_water_bottles_per_day_l803_80326

theorem water_bottles_per_day 
  (total_bottles : ℕ) 
  (total_days : ℕ) 
  (h1 : total_bottles = 28) 
  (h2 : total_days = 4) 
  (h3 : total_days ≠ 0) : 
  total_bottles / total_days = 7 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_per_day_l803_80326


namespace NUMINAMATH_CALUDE_albert_sequence_theorem_l803_80318

/-- Represents the sequence of positive integers starting with 1 or 2 in increasing order -/
def albert_sequence : ℕ → ℕ := sorry

/-- Returns the nth digit in Albert's sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def target_number : ℕ := 100 * (nth_digit 1498) + 10 * (nth_digit 1499) + (nth_digit 1500)

theorem albert_sequence_theorem : target_number = 121 := by sorry

end NUMINAMATH_CALUDE_albert_sequence_theorem_l803_80318


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l803_80339

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

-- Theorem statement
theorem intersection_implies_a_zero (a : ℝ) : A a ∩ B a = {-1} → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l803_80339


namespace NUMINAMATH_CALUDE_marble_selection_ways_l803_80335

def total_marbles : ℕ := 15
def specific_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem marble_selection_ways : 
  specific_marbles * choose (total_marbles - specific_marbles) (marbles_to_choose - 1) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l803_80335


namespace NUMINAMATH_CALUDE_correct_sum_after_change_l803_80374

def number1 : ℕ := 935641
def number2 : ℕ := 471850
def incorrect_sum : ℕ := 1417491
def digit_to_change : ℕ := 7
def new_digit : ℕ := 8

theorem correct_sum_after_change :
  ∃ (changed_number2 : ℕ),
    (changed_number2 ≠ number2) ∧
    (∃ (pos : ℕ),
      (number2 / 10^pos) % 10 = digit_to_change ∧
      changed_number2 = number2 + (new_digit - digit_to_change) * 10^pos) ∧
    (number1 + changed_number2 = incorrect_sum) :=
  sorry

end NUMINAMATH_CALUDE_correct_sum_after_change_l803_80374


namespace NUMINAMATH_CALUDE_product_pricing_and_purchase_l803_80382

-- Define variables
variable (x : ℝ) -- Price of product A
variable (y : ℝ) -- Price of product B
variable (m : ℝ) -- Number of units of product A to be purchased

-- Define the conditions
def condition1 : Prop := 2 * x + 3 * y = 690
def condition2 : Prop := x + 4 * y = 720
def condition3 : Prop := m * x + (40 - m) * y ≤ 5400
def condition4 : Prop := m ≤ 3 * (40 - m)

-- State the theorem
theorem product_pricing_and_purchase (h1 : condition1 x y) (h2 : condition2 x y) 
  (h3 : condition3 x y m) (h4 : condition4 m) : 
  x = 120 ∧ y = 150 ∧ 20 ≤ m ∧ m ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_and_purchase_l803_80382


namespace NUMINAMATH_CALUDE_inequality_proof_l803_80396

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (3 * a + 2 * b + c) +
  b / Real.sqrt (3 * b + 2 * c + a) +
  c / Real.sqrt (3 * c + 2 * a + b) ≤
  (1 / Real.sqrt 2) * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l803_80396


namespace NUMINAMATH_CALUDE_unique_assignment_l803_80341

-- Define the friends and professions as enums
inductive Friend : Type
  | Ivanov | Petrenko | Sidorchuk | Grishin | Altman

inductive Profession : Type
  | Painter | Miller | Carpenter | Postman | Barber

-- Define the assignment of professions to friends
def assignment : Friend → Profession
  | Friend.Ivanov => Profession.Barber
  | Friend.Petrenko => Profession.Miller
  | Friend.Sidorchuk => Profession.Postman
  | Friend.Grishin => Profession.Carpenter
  | Friend.Altman => Profession.Painter

-- Define the conditions
def conditions (a : Friend → Profession) : Prop :=
  -- Each friend has a unique profession
  (∀ f1 f2, f1 ≠ f2 → a f1 ≠ a f2) ∧
  -- Petrenko and Grishin have never used a painter's brush
  (a Friend.Petrenko ≠ Profession.Painter ∧ a Friend.Grishin ≠ Profession.Painter) ∧
  -- Ivanov and Grishin visited the miller
  (a Friend.Ivanov ≠ Profession.Miller ∧ a Friend.Grishin ≠ Profession.Miller) ∧
  -- Petrenko and Altman live in the same house as the postman
  (a Friend.Petrenko ≠ Profession.Postman ∧ a Friend.Altman ≠ Profession.Postman) ∧
  -- Sidorchuk attended Petrenko's wedding and the wedding of his barber friend's daughter
  (a Friend.Sidorchuk ≠ Profession.Barber ∧ a Friend.Petrenko ≠ Profession.Barber) ∧
  -- Ivanov and Petrenko often play dominoes with the carpenter and the painter
  (a Friend.Ivanov ≠ Profession.Carpenter ∧ a Friend.Ivanov ≠ Profession.Painter ∧
   a Friend.Petrenko ≠ Profession.Carpenter ∧ a Friend.Petrenko ≠ Profession.Painter) ∧
  -- Grishin and Altman go to their barber friend's shop to get shaved
  (a Friend.Grishin ≠ Profession.Barber ∧ a Friend.Altman ≠ Profession.Barber) ∧
  -- The postman shaves himself
  (∀ f, a f = Profession.Postman → a f ≠ Profession.Barber)

-- Theorem statement
theorem unique_assignment : 
  ∀ a : Friend → Profession, conditions a → a = assignment :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l803_80341


namespace NUMINAMATH_CALUDE_pink_notebook_cost_l803_80304

def total_notebooks : ℕ := 4
def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def green_notebook_cost : ℕ := 10

theorem pink_notebook_cost :
  total_notebooks = green_notebooks + black_notebooks + pink_notebooks →
  total_cost = green_notebooks * green_notebook_cost + black_notebook_cost + pink_notebooks * 10 := by
  sorry

end NUMINAMATH_CALUDE_pink_notebook_cost_l803_80304


namespace NUMINAMATH_CALUDE_first_grade_enrollment_proof_l803_80387

theorem first_grade_enrollment_proof :
  ∃! a : ℕ,
    200 ≤ a ∧ a ≤ 300 ∧
    (∃ R : ℕ, a = 25 * R + 10) ∧
    (∃ L : ℕ, a = 30 * L - 15) ∧
    a = 285 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_proof_l803_80387


namespace NUMINAMATH_CALUDE_sqrt_15_div_sqrt_5_eq_sqrt_3_l803_80302

theorem sqrt_15_div_sqrt_5_eq_sqrt_3 : Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_div_sqrt_5_eq_sqrt_3_l803_80302


namespace NUMINAMATH_CALUDE_product_125_sum_31_l803_80342

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l803_80342


namespace NUMINAMATH_CALUDE_bold_o_lit_cells_l803_80310

/-- Represents a 5x5 grid with boolean values indicating lit (true) or unlit (false) cells. -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- The initial configuration of the letter 'o' on the grid. -/
def initial_o : Grid := sorry

/-- The number of lit cells in the initial 'o' configuration. -/
def initial_lit_cells : Nat := 12

/-- Makes a letter bold by lighting cells to the right of lit cells. -/
def make_bold (g : Grid) : Grid := sorry

/-- Counts the number of lit cells in a grid. -/
def count_lit_cells (g : Grid) : Nat := sorry

/-- Theorem stating that the number of lit cells in a bold 'o' is 24. -/
theorem bold_o_lit_cells :
  count_lit_cells (make_bold initial_o) = 24 := by sorry

end NUMINAMATH_CALUDE_bold_o_lit_cells_l803_80310


namespace NUMINAMATH_CALUDE_binomial_product_l803_80354

theorem binomial_product (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l803_80354


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_division_l803_80394

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ
  base1_positive : base1 > 0
  base2_positive : base2 > 0
  leg_positive : leg > 0

/-- A division of an isosceles right triangle into trapezoids -/
def TriangleDivision (t : IsoscelesRightTriangle) := 
  List IsoscelesTrapezoid

theorem isosceles_right_triangle_division (t : IsoscelesRightTriangle) :
  ∃ (d : TriangleDivision t), d.length = 7 :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_division_l803_80394


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l803_80360

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 6 = 5) ∧ 
  (a % 8 = 7) ∧ 
  (∀ b : ℕ, b > 0 → b % 6 = 5 → b % 8 = 7 → a ≤ b) ∧
  (a = 23) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l803_80360


namespace NUMINAMATH_CALUDE_midpoint_fraction_l803_80392

theorem midpoint_fraction : 
  let a := 3/4
  let b := 5/6
  (a + b) / 2 = 19/24 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l803_80392


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l803_80330

/-- Proves that the first interest rate is 10% given the problem conditions --/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (total_profit : ℕ)
  (second_rate : ℚ)
  (h1 : total_amount = 50000)
  (h2 : first_part = 30000)
  (h3 : second_part = total_amount - first_part)
  (h4 : total_profit = 7000)
  (h5 : second_rate = 20 / 100)
  : ∃ (r : ℚ), r = 10 / 100 ∧ 
    total_profit = (first_part * r).floor + (second_part * second_rate).floor :=
by sorry


end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l803_80330


namespace NUMINAMATH_CALUDE_sin_double_alpha_l803_80323

/-- Given that the terminal side of angle α intersects the unit circle at point P(-√3/2, 1/2),
    prove that sin 2α = -√3/2 -/
theorem sin_double_alpha (α : Real) 
  (h : ∃ P : Real × Real, P.1 = -Real.sqrt 3 / 2 ∧ P.2 = 1 / 2 ∧ 
       P.1^2 + P.2^2 = 1 ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.sin (2 * α) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l803_80323


namespace NUMINAMATH_CALUDE_fraction_equality_l803_80362

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 8/11) : 
  (7*x + 11*y) / (49*x*y) = 231/56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l803_80362


namespace NUMINAMATH_CALUDE_root_equation_problem_l803_80365

theorem root_equation_problem (a b c m : ℝ) : 
  (a^2 - 4*a + m = 0 ∧ b^2 - 4*b + m = 0) →
  (b^2 - 8*b + 5*m = 0 ∧ c^2 - 8*c + 5*m = 0) →
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l803_80365


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l803_80338

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (a ≥ 3) →
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) → a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l803_80338


namespace NUMINAMATH_CALUDE_inscribed_decagon_area_proof_l803_80351

/-- The area of a decagon inscribed in a square with perimeter 150 cm, 
    where the vertices of the decagon divide each side of the square into five equal segments. -/
def inscribed_decagon_area : ℝ := 1181.25

/-- The perimeter of the square. -/
def square_perimeter : ℝ := 150

/-- The number of equal segments each side of the square is divided into. -/
def num_segments : ℕ := 5

/-- The number of triangles removed from the square to form the decagon. -/
def num_triangles : ℕ := 8

theorem inscribed_decagon_area_proof :
  let side_length := square_perimeter / 4
  let segment_length := side_length / num_segments
  let triangle_area := (1 / 2) * segment_length * segment_length
  let total_triangle_area := num_triangles * triangle_area
  let square_area := side_length * side_length
  square_area - total_triangle_area = inscribed_decagon_area := by sorry

end NUMINAMATH_CALUDE_inscribed_decagon_area_proof_l803_80351


namespace NUMINAMATH_CALUDE_complex_modulus_l803_80369

theorem complex_modulus (a b : ℝ) (h : b^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l803_80369


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l803_80313

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 32 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    a' * b' * c' * d' = 8 ∧ 
    e' * f' * g' * h' = 16 ∧ 
    (a' * e')^2 + (b' * f')^2 + (c' * g')^2 + (d' * h')^2 = 32 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l803_80313


namespace NUMINAMATH_CALUDE_f_g_deriv_neg_l803_80370

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_neg (x : ℝ) (h : x < 0) : deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_deriv_neg_l803_80370


namespace NUMINAMATH_CALUDE_equation_equivalence_l803_80312

theorem equation_equivalence (x : ℝ) : 6 - (x - 2) / 2 = x ↔ 12 - x + 2 = 2 * x :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l803_80312


namespace NUMINAMATH_CALUDE_binomial_square_constant_l803_80348

theorem binomial_square_constant (c : ℚ) : 
  (∃ a b : ℚ, ∀ x, 9 * x^2 + 27 * x + c = (a * x + b)^2) → c = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l803_80348


namespace NUMINAMATH_CALUDE_one_point_three_six_billion_scientific_notation_l803_80329

/-- Proves that 1.36 billion is equal to 1.36 × 10^9 -/
theorem one_point_three_six_billion_scientific_notation :
  (1.36 : ℝ) * (10 ^ 9 : ℝ) = 1.36e9 := by sorry

end NUMINAMATH_CALUDE_one_point_three_six_billion_scientific_notation_l803_80329


namespace NUMINAMATH_CALUDE_cost_price_is_47_5_l803_80321

/-- Given an article with a marked price and discount rate, calculates the cost price -/
def calculate_cost_price (marked_price : ℚ) (discount_rate : ℚ) (profit_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  selling_price / (1 + profit_rate)

/-- Theorem stating that the cost price of the article is 47.5 given the conditions -/
theorem cost_price_is_47_5 :
  let marked_price : ℚ := 74.21875
  let discount_rate : ℚ := 0.20
  let profit_rate : ℚ := 0.25
  calculate_cost_price marked_price discount_rate profit_rate = 47.5 := by
  sorry

#eval calculate_cost_price 74.21875 0.20 0.25

end NUMINAMATH_CALUDE_cost_price_is_47_5_l803_80321
