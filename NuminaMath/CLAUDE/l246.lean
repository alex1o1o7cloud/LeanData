import Mathlib

namespace missing_village_population_l246_24696

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 1249]
def total_villages : ℕ := 7
def average_population : ℕ := 1000

theorem missing_village_population :
  (village_populations.sum + (total_villages * average_population - village_populations.sum)) / total_villages = average_population ∧
  total_villages * average_population - village_populations.sum = 980 :=
by sorry

end missing_village_population_l246_24696


namespace coefficient_x_squared_l246_24685

theorem coefficient_x_squared (x : ℝ) : 
  let expansion := (1 + 1/x + 1/x^2) * (1 + x^2)^5
  ∃ a b c d e : ℝ, expansion = a*x^2 + b*x + c + d/x + e/x^2 ∧ a = 15 := by
sorry

end coefficient_x_squared_l246_24685


namespace quiz_score_problem_l246_24659

theorem quiz_score_problem (initial_students : ℕ) (dropped_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 ∧ 
  dropped_students = 3 ∧ 
  initial_average = 62.5 ∧ 
  new_average = 62 →
  (initial_students * initial_average - 
   (initial_students - dropped_students) * new_average : ℚ) = 194 := by
  sorry

end quiz_score_problem_l246_24659


namespace altitude_polynomial_exists_l246_24607

/-- Given a triangle whose side lengths are the roots of a cubic polynomial
    with rational coefficients, there exists a polynomial of sixth degree
    with rational coefficients whose roots are the altitudes of this triangle. -/
theorem altitude_polynomial_exists (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ), ∀ x : ℝ,
    p * x^6 + q * x^5 + s * x^4 + t * x^3 + u * x^2 + v * x + w = 0 ↔
    x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₁ * (r₂ + r₃ - r₁))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₂ * (r₃ + r₁ - r₂))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₃ * (r₁ + r₂ - r₃)) :=
by
  sorry

end altitude_polynomial_exists_l246_24607


namespace not_right_triangle_l246_24666

theorem not_right_triangle : ∃ (a b c : ℝ), 
  (a = Real.sqrt 3 ∧ b = 4 ∧ c = 5) ∧ 
  (a^2 + b^2 ≠ c^2) ∧
  (∀ (x y z : ℝ), 
    ((x = 1 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 3) ∨
     (x = 7 ∧ y = 24 ∧ z = 25) ∨
     (x = 5 ∧ y = 12 ∧ z = 13)) →
    (x^2 + y^2 = z^2)) :=
by sorry

end not_right_triangle_l246_24666


namespace equation_roots_l246_24681

theorem equation_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -x₂) ∧ 
  (x₁ = Real.sqrt 2 ∨ x₁ = -Real.sqrt 2) ∧
  (x₂ = Real.sqrt 2 ∨ x₂ = -Real.sqrt 2) ∧
  (x₃ = 1/2) ∧
  (2 * x₁^5 - x₁^4 - 2 * x₁^3 + x₁^2 - 4 * x₁ + 2 = 0) ∧
  (2 * x₂^5 - x₂^4 - 2 * x₂^3 + x₂^2 - 4 * x₂ + 2 = 0) ∧
  (2 * x₃^5 - x₃^4 - 2 * x₃^3 + x₃^2 - 4 * x₃ + 2 = 0) := by
  sorry

#check equation_roots

end equation_roots_l246_24681


namespace will_initial_money_l246_24626

-- Define the initial amount of money Will had
def initial_money : ℕ := sorry

-- Define the cost of the game
def game_cost : ℕ := 47

-- Define the number of toys bought
def num_toys : ℕ := 9

-- Define the cost of each toy
def toy_cost : ℕ := 4

-- Theorem to prove
theorem will_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end will_initial_money_l246_24626


namespace first_tract_width_l246_24613

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    calculates the width of the first tract. -/
theorem first_tract_width (length1 : ℝ) (length2 width2 : ℝ) (combined_area : ℝ) : 
  length1 = 300 →
  length2 = 250 →
  width2 = 630 →
  combined_area = 307500 →
  combined_area = length1 * (combined_area - length2 * width2) / length1 + length2 * width2 →
  (combined_area - length2 * width2) / length1 = 500 := by
sorry

end first_tract_width_l246_24613


namespace arithmetic_calculation_l246_24645

theorem arithmetic_calculation : 3521 + 480 / 60 * 3 - 521 = 3024 := by
  sorry

end arithmetic_calculation_l246_24645


namespace value_of_3a_plus_6b_l246_24664

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end value_of_3a_plus_6b_l246_24664


namespace candy_cost_l246_24683

def amount_given : ℚ := 1
def change_received : ℚ := 0.46

theorem candy_cost : amount_given - change_received = 0.54 := by
  sorry

end candy_cost_l246_24683


namespace least_value_quadratic_inequality_l246_24610

theorem least_value_quadratic_inequality :
  ∃ (x : ℝ), x = 4 ∧ (∀ y : ℝ, -y^2 + 9*y - 20 ≤ 0 → y ≥ x) := by
  sorry

end least_value_quadratic_inequality_l246_24610


namespace mean_proportional_81_100_l246_24622

theorem mean_proportional_81_100 : ∃ x : ℝ, x^2 = 81 * 100 ∧ x = 90 := by
  sorry

end mean_proportional_81_100_l246_24622


namespace decimal_sum_to_fraction_l246_24617

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 + 0.000007 = 234567 / 1000000 := by
  sorry

end decimal_sum_to_fraction_l246_24617


namespace pool_cost_is_90000_l246_24621

/-- The cost to fill a rectangular pool with bottled water -/
def pool_fill_cost (length width depth : ℝ) (liters_per_cubic_foot : ℝ) (cost_per_liter : ℝ) : ℝ :=
  length * width * depth * liters_per_cubic_foot * cost_per_liter

/-- Theorem: The cost to fill the specified pool is $90,000 -/
theorem pool_cost_is_90000 :
  pool_fill_cost 20 6 10 25 3 = 90000 := by
  sorry

#eval pool_fill_cost 20 6 10 25 3

end pool_cost_is_90000_l246_24621


namespace derivative_of_constant_cosine_l246_24648

theorem derivative_of_constant_cosine (x : ℝ) : 
  deriv (λ _ : ℝ => Real.cos (π / 3)) x = 0 :=
by sorry

end derivative_of_constant_cosine_l246_24648


namespace not_proportional_l246_24619

theorem not_proportional (x y : ℝ) : 
  (3 * x - y = 7 ∨ 4 * x + 3 * y = 13) → 
  ¬(∃ (k₁ k₂ : ℝ), (y = k₁ * x ∨ x * y = k₂)) :=
by sorry

end not_proportional_l246_24619


namespace second_discount_percentage_l246_24670

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 200 →
  first_discount = 10 →
  final_price = 171 →
  (initial_price * (1 - first_discount / 100) * (1 - (initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100))) = final_price) ∧
  ((initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100)) * 100 = 5) :=
by sorry

end second_discount_percentage_l246_24670


namespace a_6_equals_2_l246_24641

/-- An arithmetic sequence with common difference 2 where a₁, a₃, and a₄ form a geometric sequence -/
def ArithGeomSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 3)^2 = a 1 * a 4

theorem a_6_equals_2 (a : ℕ → ℝ) (h : ArithGeomSequence a) : a 6 = 2 := by
  sorry

end a_6_equals_2_l246_24641


namespace two_from_four_is_six_l246_24603

/-- The number of ways to choose 2 items from a set of 4 distinct items -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- Theorem stating that choosing 2 items from 4 distinct items results in 6 possibilities -/
theorem two_from_four_is_six : choose_two_from_four = 6 := by
  sorry

end two_from_four_is_six_l246_24603


namespace franks_daily_work_hours_l246_24630

/-- Given that Frank worked a total of 8.0 hours over 4.0 days, with equal time worked each day,
    prove that he worked 2.0 hours per day. -/
theorem franks_daily_work_hours (total_hours : ℝ) (total_days : ℝ) (hours_per_day : ℝ)
    (h1 : total_hours = 8.0)
    (h2 : total_days = 4.0)
    (h3 : hours_per_day * total_days = total_hours) :
    hours_per_day = 2.0 := by
  sorry

end franks_daily_work_hours_l246_24630


namespace point_outside_circle_l246_24661

theorem point_outside_circle (r d : ℝ) (hr : r = 2) (hd : d = 3) :
  d > r :=
by sorry

end point_outside_circle_l246_24661


namespace chord_length_for_max_distance_l246_24649

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the chord AB
def Chord (A B : ℝ × ℝ) := A ∈ Circle ∧ B ∈ Circle

-- Define the semicircle ACB
def Semicircle (A B C : ℝ × ℝ) := 
  Chord A B ∧ 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the point C as the farthest point on semicircle ACB from O
def FarthestPoint (O A B C : ℝ × ℝ) := 
  Semicircle A B C ∧
  ∀ D, Semicircle A B D → (C.1 - O.1)^2 + (C.2 - O.2)^2 ≥ (D.1 - O.1)^2 + (D.2 - O.2)^2

-- Define OC perpendicular to AB
def Perpendicular (O A B C : ℝ × ℝ) := 
  (C.1 - O.1) * (B.1 - A.1) + (C.2 - O.2) * (B.2 - A.2) = 0

-- The main theorem
theorem chord_length_for_max_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  Chord A B →
  FarthestPoint O A B C →
  Perpendicular O A B C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 :=
sorry

end chord_length_for_max_distance_l246_24649


namespace marc_total_spent_l246_24665

/-- Calculates the total amount Marc spent on his purchases --/
def total_spent (model_car_price : ℝ) (paint_price : ℝ) (paintbrush_price : ℝ)
                (display_case_price : ℝ) (model_car_discount : ℝ) (paint_coupon : ℝ)
                (gift_card : ℝ) (first_tax_rate : ℝ) (second_tax_rate : ℝ) : ℝ :=
  let model_cars_cost := 5 * model_car_price * (1 - model_car_discount)
  let paint_cost := 5 * paint_price - paint_coupon
  let paintbrushes_cost := 7 * paintbrush_price
  let first_subtotal := model_cars_cost + paint_cost + paintbrushes_cost - gift_card
  let first_transaction := first_subtotal * (1 + first_tax_rate)
  let display_cases_cost := 3 * display_case_price
  let second_transaction := display_cases_cost * (1 + second_tax_rate)
  first_transaction + second_transaction

/-- Theorem stating that Marc's total spent is $187.02 --/
theorem marc_total_spent :
  total_spent 20 10 2 15 0.1 5 20 0.08 0.06 = 187.02 := by
  sorry

end marc_total_spent_l246_24665


namespace complex_equality_l246_24677

theorem complex_equality (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 * Complex.I)
  Complex.re z = Complex.im z → a = -1 := by
  sorry

end complex_equality_l246_24677


namespace unique_quadratic_solution_l246_24656

theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + 30 * x + 5 = 0) → 
  (∃ x, a * x^2 + 30 * x + 5 = 0 ∧ x = -1/3) :=
by sorry

end unique_quadratic_solution_l246_24656


namespace range_of_m_l246_24679

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
by sorry

end range_of_m_l246_24679


namespace total_earnings_l246_24650

def price_per_kg : ℝ := 0.50
def rooster1_weight : ℝ := 30
def rooster2_weight : ℝ := 40

theorem total_earnings :
  price_per_kg * rooster1_weight + price_per_kg * rooster2_weight = 35 := by
sorry

end total_earnings_l246_24650


namespace side_altitude_inequality_l246_24674

/-- Triangle ABC with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  hₐ : ℝ
  hb : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_hₐ : 0 < hₐ
  pos_hb : 0 < hb

/-- Theorem: In a triangle, a ≥ b if and only if a + hₐ ≥ b + hb -/
theorem side_altitude_inequality (t : Triangle) : t.a ≥ t.b ↔ t.a + t.hₐ ≥ t.b + t.hb := by
  sorry

end side_altitude_inequality_l246_24674


namespace board_sum_always_odd_l246_24637

theorem board_sum_always_odd (n : ℕ) (h : n = 1966) :
  let initial_sum := n * (n + 1) / 2
  ∀ (operations : ℕ), ∃ (final_sum : ℤ),
    final_sum ≡ initial_sum [ZMOD 2] ∧ final_sum ≠ 0 := by
  sorry

end board_sum_always_odd_l246_24637


namespace system_of_equations_solution_range_l246_24606

theorem system_of_equations_solution_range (a x y : ℝ) : 
  x + 3*y = 3 - a →
  2*x + y = 1 + 3*a →
  x + y > 3*a + 4 →
  a < -3/2 := by
sorry

end system_of_equations_solution_range_l246_24606


namespace rational_terms_not_adjacent_probability_l246_24642

theorem rational_terms_not_adjacent_probability :
  let total_terms : ℕ := 9
  let rational_terms : ℕ := 3
  let irrational_terms : ℕ := 6
  let total_arrangements := Nat.factorial total_terms
  let favorable_arrangements := Nat.factorial irrational_terms * (Nat.factorial (irrational_terms + 1)).choose rational_terms
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end rational_terms_not_adjacent_probability_l246_24642


namespace correct_front_view_l246_24657

def StackColumn := List Nat

def frontView (stacks : List StackColumn) : List Nat :=
  stacks.map (List.foldl max 0)

theorem correct_front_view (stacks : List StackColumn) :
  stacks = [[3, 5], [2, 6, 4], [1, 1, 3, 8], [5, 2]] →
  frontView stacks = [5, 6, 8, 5] := by
  sorry

end correct_front_view_l246_24657


namespace sum_of_fractions_inequality_l246_24680

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 1 := by
  sorry

end sum_of_fractions_inequality_l246_24680


namespace intercepted_segment_length_l246_24697

-- Define the polar equations
def line_equation (p θ : ℝ) : Prop := p * Real.cos θ = 1
def circle_equation (p θ : ℝ) : Prop := p = 4 * Real.cos θ

-- Define the theorem
theorem intercepted_segment_length :
  ∃ (p₁ θ₁ p₂ θ₂ : ℝ),
    line_equation p₁ θ₁ ∧
    line_equation p₂ θ₂ ∧
    circle_equation p₁ θ₁ ∧
    circle_equation p₂ θ₂ ∧
    (p₁ * Real.cos θ₁ - p₂ * Real.cos θ₂)^2 + (p₁ * Real.sin θ₁ - p₂ * Real.sin θ₂)^2 = 12 :=
sorry

end intercepted_segment_length_l246_24697


namespace monotonic_increasing_sufficient_not_necessary_l246_24689

-- Define a monotonically increasing function on ℝ
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Define the existence of x₁ < x₂ such that f(x₁) < f(x₂)
def ExistsStrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem stating that monotonic increasing is sufficient but not necessary
-- for the existence of strictly increasing points
theorem monotonic_increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (MonotonicIncreasing f → ExistsStrictlyIncreasing f) ∧
  ¬(ExistsStrictlyIncreasing f → MonotonicIncreasing f) :=
sorry

end monotonic_increasing_sufficient_not_necessary_l246_24689


namespace number_subtraction_problem_l246_24601

theorem number_subtraction_problem :
  ∀ x : ℤ, x - 2 = 6 → x = 8 := by
  sorry

end number_subtraction_problem_l246_24601


namespace first_number_is_five_l246_24644

/-- A sequence where each term is obtained by adding 9 to the previous term -/
def arithmeticSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => arithmeticSequence a₁ n + 9

/-- The property that 2012 is in the sequence -/
def contains2012 (a₁ : ℕ) : Prop :=
  ∃ n : ℕ, arithmeticSequence a₁ n = 2012

theorem first_number_is_five :
  ∃ a₁ : ℕ, a₁ < 10 ∧ contains2012 a₁ ∧ a₁ = 5 :=
by sorry

end first_number_is_five_l246_24644


namespace polynomial_multiplication_l246_24602

theorem polynomial_multiplication :
  ∀ x : ℝ, (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
    35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := by
  sorry

end polynomial_multiplication_l246_24602


namespace equation_proof_l246_24662

theorem equation_proof : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end equation_proof_l246_24662


namespace probability_of_pink_flower_l246_24636

-- Define the contents of the bags
def bag_A_red : ℕ := 6
def bag_A_pink : ℕ := 3
def bag_B_red : ℕ := 2
def bag_B_pink : ℕ := 7

-- Define the total number of flowers in each bag
def total_A : ℕ := bag_A_red + bag_A_pink
def total_B : ℕ := bag_B_red + bag_B_pink

-- Define the probability of choosing a pink flower from each bag
def prob_pink_A : ℚ := bag_A_pink / total_A
def prob_pink_B : ℚ := bag_B_pink / total_B

-- Theorem statement
theorem probability_of_pink_flower :
  (prob_pink_A + prob_pink_B) / 2 = 5 / 9 := by sorry

end probability_of_pink_flower_l246_24636


namespace fourth_circle_radius_l246_24686

theorem fourth_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 23) (h₂ : r₂ = 35) (h₃ : r₃ = Real.sqrt 1754) :
  π * r₃^2 = π * r₁^2 + π * r₂^2 := by
  sorry

end fourth_circle_radius_l246_24686


namespace dormitory_to_city_distance_l246_24608

theorem dormitory_to_city_distance : 
  ∀ (total_distance : ℝ),
    (1/5 : ℝ) * total_distance + (2/3 : ℝ) * total_distance + 4 = total_distance →
    total_distance = 30 := by
  sorry

end dormitory_to_city_distance_l246_24608


namespace scale_drawing_conversion_l246_24635

/-- Proves that a 6.5 cm line segment in a scale drawing where 1 cm represents 250 meters
    is equivalent to 5332.125 feet, given that 1 meter equals approximately 3.281 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (length : ℝ) (meter_to_feet : ℝ) :
  scale = 250 →
  length = 6.5 →
  meter_to_feet = 3.281 →
  length * scale * meter_to_feet = 5332.125 := by
sorry

end scale_drawing_conversion_l246_24635


namespace terrell_weight_lifting_l246_24653

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 16

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def new_lifts : ℕ := (num_weights * original_lifts * original_weight) / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 40 :=
sorry

end terrell_weight_lifting_l246_24653


namespace roots_relation_l246_24682

-- Define the polynomials f and g
def f (x : ℝ) := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ r : ℝ, f r = 0 → g (r^2) b c d = 0) →
  b = -2 ∧ c = 1 ∧ d = -12 :=
sorry

end roots_relation_l246_24682


namespace value_of_y_l246_24623

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 15 ∧ y = 35 := by sorry

end value_of_y_l246_24623


namespace reflect_x_coordinates_l246_24671

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across x-axis operation
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem reflect_x_coordinates (x y : ℝ) :
  reflect_x (x, y) = (x, -y) := by
  sorry

end reflect_x_coordinates_l246_24671


namespace sin_product_theorem_l246_24627

theorem sin_product_theorem (x : ℝ) (h : Real.sin (5 * Real.pi / 2 - x) = 3 / 5) :
  Real.sin (x / 2) * Real.sin (5 * x / 2) = 86 / 125 := by
  sorry

end sin_product_theorem_l246_24627


namespace absolute_value_equation_solution_l246_24611

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 8| = 4 - 3*x :=
by
  -- The unique solution is x = -4/5
  use -4/5
  sorry

end absolute_value_equation_solution_l246_24611


namespace average_speed_two_hours_car_average_speed_l246_24690

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 ≥ 0 → d2 ≥ 0 → (d1 + d2) / 2 = (d1 / 1 + d2 / 1) / 2 := by
  sorry

/-- The average speed of a car traveling 10 km in the first hour and 60 km in the second hour is 35 km/h -/
theorem car_average_speed : 
  let d1 : ℝ := 10  -- Distance traveled in the first hour
  let d2 : ℝ := 60  -- Distance traveled in the second hour
  (d1 + d2) / 2 = 35 := by
  sorry

end average_speed_two_hours_car_average_speed_l246_24690


namespace intersection_segment_length_l246_24629

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 / 7 := by sorry

end intersection_segment_length_l246_24629


namespace vector_parallel_tangent_l246_24609

/-- Given points A, B, and C in a 2D Cartesian coordinate system,
    prove that vector AB equals (1, √3) and tan x equals √3 when AB is parallel to OC. -/
theorem vector_parallel_tangent (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (Real.cos x, Real.sin x)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OC : ℝ × ℝ := C
  AB.2 / AB.1 = OC.2 / OC.1 →
  AB = (1, Real.sqrt 3) ∧ Real.tan x = Real.sqrt 3 := by
  sorry

end vector_parallel_tangent_l246_24609


namespace equivalent_operations_l246_24669

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/3) = x * (15/12) :=
by sorry

end equivalent_operations_l246_24669


namespace threeTangentLines_l246_24687

/-- Represents a circle in the 2D plane --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The first circle: x^2 + y^2 + 4x - 4y + 7 = 0 --/
def circle1 : Circle := { a := 1, b := 1, c := 4, d := -4, e := 7 }

/-- The second circle: x^2 + y^2 - 4x - 10y + 13 = 0 --/
def circle2 : Circle := { a := 1, b := 1, c := -4, d := -10, e := 13 }

/-- Count the number of lines tangent to both circles --/
def countTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 lines tangent to both circles --/
theorem threeTangentLines : countTangentLines circle1 circle2 = 3 := by
  sorry

end threeTangentLines_l246_24687


namespace sum_of_roots_quadratic_l246_24652

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l246_24652


namespace arccos_cos_fifteen_l246_24639

theorem arccos_cos_fifteen (x : Real) (h : x = 15) :
  Real.arccos (Real.cos x) = x % (2 * Real.pi) := by
  sorry

end arccos_cos_fifteen_l246_24639


namespace equation_solutions_l246_24698

def solutions : Set (ℤ × ℤ) := {(-13,-2), (-4,-1), (-1,0), (2,3), (3,6), (4,15), (6,-21), (7,-12), (8,-9), (11,-6), (14,-5), (23,-4)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x * y + 3 * x - 5 * y = -3) ↔ (x, y) ∈ solutions :=
by sorry

end equation_solutions_l246_24698


namespace inequality_and_equalities_l246_24693

theorem inequality_and_equalities : 
  ((-3)^2 ≠ -3^2) ∧ 
  (|-5| = -(-5)) ∧ 
  (-Real.sqrt 4 = -2) ∧ 
  ((-1)^3 = -1^3) := by
  sorry

end inequality_and_equalities_l246_24693


namespace apple_cost_calculation_l246_24615

/-- If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem apple_cost_calculation (cost_four_dozen : ℝ) (h : cost_four_dozen = 31.20) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by sorry

end apple_cost_calculation_l246_24615


namespace cos_2alpha_plus_4pi_3_l246_24605

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α * Real.cos α = 1 / 2) :
  Real.cos (2 * α + 4 * π / 3) = -7 / 8 := by
  sorry

end cos_2alpha_plus_4pi_3_l246_24605


namespace efqs_equals_qrst_l246_24699

/-- Assigns a value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- Calculates the product of values assigned to a list of characters -/
def list_product (s : List Char) : ℕ :=
  s.map letter_value |>.foldl (·*·) 1

/-- Checks if a list of characters contains distinct elements -/
def distinct_chars (s : List Char) : Prop :=
  s.toFinset.card = s.length

theorem efqs_equals_qrst : ∃ (e f q s : Char), 
  distinct_chars ['E', 'F', 'Q', 'S'] ∧
  list_product ['E', 'F', 'Q', 'S'] = list_product ['Q', 'R', 'S', 'T'] :=
by sorry

end efqs_equals_qrst_l246_24699


namespace min_A_mats_l246_24673

/-- Represents the purchase and sale of bamboo mats -/
structure BambooMatSale where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  sale_price_A : ℝ
  sale_price_B : ℝ

/-- The conditions of the bamboo mat sale problem -/
def bamboo_mat_conditions (s : BambooMatSale) : Prop :=
  10 * s.purchase_price_A + 15 * s.purchase_price_B = 3600 ∧
  25 * s.purchase_price_A + 30 * s.purchase_price_B = 8100 ∧
  s.sale_price_A = 260 ∧
  s.sale_price_B = 180

/-- The profit calculation for a given number of mats A -/
def profit (s : BambooMatSale) (num_A : ℝ) : ℝ :=
  (s.sale_price_A - s.purchase_price_A) * num_A +
  (s.sale_price_B - s.purchase_price_B) * (60 - num_A)

/-- The main theorem stating the minimum number of A mats to purchase -/
theorem min_A_mats (s : BambooMatSale) 
  (h : bamboo_mat_conditions s) : 
  ∃ (n : ℕ), n = 40 ∧ 
  (∀ (m : ℕ), m ≥ 40 → profit s m ≥ 4400) ∧
  (∀ (m : ℕ), m < 40 → profit s m < 4400) := by
  sorry

end min_A_mats_l246_24673


namespace kenny_book_purchase_l246_24640

/-- Calculates the number of books Kenny can buy after mowing lawns and purchasing video games -/
def books_kenny_can_buy (lawn_price : ℕ) (video_game_price : ℕ) (book_price : ℕ) 
                        (lawns_mowed : ℕ) (video_games_to_buy : ℕ) : ℕ :=
  let total_earnings := lawn_price * lawns_mowed
  let video_games_cost := video_game_price * video_games_to_buy
  let remaining_money := total_earnings - video_games_cost
  remaining_money / book_price

/-- Theorem stating that Kenny can buy 60 books given the problem conditions -/
theorem kenny_book_purchase :
  books_kenny_can_buy 15 45 5 35 5 = 60 := by
  sorry

#eval books_kenny_can_buy 15 45 5 35 5

end kenny_book_purchase_l246_24640


namespace rectangle_fold_ef_length_l246_24675

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the fold
structure Fold :=
  (distanceFromB : ℝ)
  (distanceFromC : ℝ)

-- Define the theorem
theorem rectangle_fold_ef_length 
  (rect : Rectangle)
  (fold : Fold)
  (h1 : rect.AB = 4)
  (h2 : rect.BC = 8)
  (h3 : fold.distanceFromB = 3)
  (h4 : fold.distanceFromC = 5)
  (h5 : fold.distanceFromB + fold.distanceFromC = rect.BC) :
  let EF := Real.sqrt ((rect.AB ^ 2) + (fold.distanceFromB ^ 2))
  EF = 5 := by sorry

end rectangle_fold_ef_length_l246_24675


namespace x_intercept_is_six_l246_24614

-- Define the line equation
def line_equation (x y : ℚ) : Prop := 4 * x - 3 * y = 24

-- Define x-intercept
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

-- Theorem statement
theorem x_intercept_is_six : is_x_intercept 6 := by
  sorry

end x_intercept_is_six_l246_24614


namespace tangent_line_and_function_inequality_l246_24633

open Real

theorem tangent_line_and_function_inequality (a b m : ℝ) : 
  (∀ x, x = -π/4 → (tan x = a*x + b + π/2)) →
  (∀ x, x ∈ Set.Icc 1 2 → m ≤ (exp x + b*x^2 + a) ∧ (exp x + b*x^2 + a) ≤ m^2 - 2) →
  (∃ m_max : ℝ, m_max = exp 1 + 1 ∧ m ≤ m_max) :=
by sorry

end tangent_line_and_function_inequality_l246_24633


namespace min_style_A_purchase_correct_l246_24695

/-- Represents the clothing store problem -/
structure ClothingStore where
  total_pieces : ℕ
  total_cost : ℕ
  unit_price_A : ℕ
  unit_price_B : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  other_store_purchase : ℕ
  other_store_min_profit : ℕ

/-- The minimum number of style A clothing pieces to be purchased by another store -/
def min_style_A_purchase (store : ClothingStore) : ℕ :=
  23

/-- Theorem stating that the minimum number of style A clothing pieces to be purchased
    by another store is correct given the conditions -/
theorem min_style_A_purchase_correct (store : ClothingStore)
  (h1 : store.total_pieces = 100)
  (h2 : store.total_cost = 11200)
  (h3 : store.unit_price_A = 120)
  (h4 : store.unit_price_B = 100)
  (h5 : store.selling_price_A = 200)
  (h6 : store.selling_price_B = 140)
  (h7 : store.other_store_purchase = 60)
  (h8 : store.other_store_min_profit = 3300) :
  ∀ m : ℕ, m ≥ min_style_A_purchase store →
    (store.selling_price_A - store.unit_price_A) * m +
    (store.selling_price_B - store.unit_price_B) * (store.other_store_purchase - m) ≥
    store.other_store_min_profit :=
  sorry

end min_style_A_purchase_correct_l246_24695


namespace trees_chopped_per_day_l246_24631

/-- Represents the number of blocks of wood Ragnar gets per tree -/
def blocks_per_tree : ℕ := 3

/-- Represents the total number of blocks of wood Ragnar gets in 5 days -/
def total_blocks : ℕ := 30

/-- Represents the number of days Ragnar works -/
def days : ℕ := 5

/-- Theorem stating the number of trees Ragnar chops each day -/
theorem trees_chopped_per_day : 
  (total_blocks / days) / blocks_per_tree = 2 := by sorry

end trees_chopped_per_day_l246_24631


namespace tangent_line_implies_a_value_l246_24663

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x + 3

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + a

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x + 1  -- We use b = 1 as it's not relevant for finding a

-- Theorem statement
theorem tangent_line_implies_a_value :
  ∀ a : ℝ, (curve_derivative a 1 = 1) → a = -3 := by sorry

end tangent_line_implies_a_value_l246_24663


namespace ellipse_slope_product_l246_24692

theorem ellipse_slope_product (a b m n x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (x^2 / a^2 + y^2 / b^2 = 1) →
  (m^2 / a^2 + n^2 / b^2 = 1) →
  ((y - n) / (x - m)) * ((y + n) / (x + m)) = -b^2 / a^2 :=
by sorry

end ellipse_slope_product_l246_24692


namespace function_properties_l246_24625

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (is_decreasing_on f 0 1) ∧
  (f 2014 = f 0) := by
  sorry

end function_properties_l246_24625


namespace function_characterization_l246_24634

def is_positive_integer (n : ℕ) : Prop := n > 0

def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ n, is_positive_integer n → f (f n) + f n = 2 * n + 6

theorem function_characterization (f : ℕ → ℕ) :
  (∀ n, is_positive_integer n → is_positive_integer (f n)) →
  satisfies_equation f →
  ∀ n, is_positive_integer n → f n = n + 2 :=
by sorry

end function_characterization_l246_24634


namespace max_value_expression_l246_24604

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 182 :=
sorry

end max_value_expression_l246_24604


namespace x_intercepts_count_l246_24620

theorem x_intercepts_count (x : ℝ) :
  (∃! x, (x - 5) * (x^2 + 6*x + 10) = 0) :=
sorry

end x_intercepts_count_l246_24620


namespace police_catch_thief_time_police_catch_thief_time_equals_two_l246_24632

/-- Proves that the time taken for a police officer to catch a thief is 2 hours,
    given specific initial conditions. -/
theorem police_catch_thief_time (thief_speed : ℝ) (police_speed : ℝ) 
  (initial_distance : ℝ) (delay_time : ℝ) : ℝ :=
  by
  -- Define the conditions
  have h1 : thief_speed = 20 := by sorry
  have h2 : police_speed = 40 := by sorry
  have h3 : initial_distance = 60 := by sorry
  have h4 : delay_time = 1 := by sorry

  -- Calculate the distance covered by the thief during the delay
  let thief_distance := thief_speed * delay_time

  -- Calculate the remaining distance between the police and thief
  let remaining_distance := initial_distance - thief_distance

  -- Calculate the relative speed between police and thief
  let relative_speed := police_speed - thief_speed

  -- Calculate the time taken to catch the thief
  let catch_time := remaining_distance / relative_speed

  -- Prove that catch_time equals 2
  sorry

/-- The time taken for the police officer to catch the thief -/
def catch_time : ℝ := 2

-- Proof that the theorem result equals the defined catch_time
theorem police_catch_thief_time_equals_two :
  police_catch_thief_time 20 40 60 1 = catch_time := by sorry

end police_catch_thief_time_police_catch_thief_time_equals_two_l246_24632


namespace quadratic_equation_properties_l246_24668

/-- A quadratic equation that is divisible by (x - 1) and has a constant term of 2 -/
def quadratic_equation (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem quadratic_equation_properties : 
  (∃ (q : ℝ → ℝ), ∀ x, quadratic_equation x = (x - 1) * q x) ∧ 
  (quadratic_equation 0 = 2) := by
  sorry

#check quadratic_equation_properties

end quadratic_equation_properties_l246_24668


namespace smallest_n_is_83_l246_24672

def candy_problem (money : ℕ) : Prop :=
  ∃ (r g b : ℕ),
    money = 18 * r ∧
    money = 20 * g ∧
    money = 22 * b ∧
    money = 24 * 83 ∧
    ∀ (n : ℕ), n < 83 → money ≠ 24 * n

theorem smallest_n_is_83 :
  ∃ (money : ℕ), candy_problem money :=
sorry

end smallest_n_is_83_l246_24672


namespace complex_equation_result_l246_24600

theorem complex_equation_result (x y : ℝ) (i : ℂ) 
  (h1 : x * i + 2 = y - i) 
  (h2 : i^2 = -1) : 
  x - y = -3 := by
sorry

end complex_equation_result_l246_24600


namespace expression_equality_l246_24655

theorem expression_equality : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : ℤ) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := by
  sorry

end expression_equality_l246_24655


namespace fraction_equality_l246_24618

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 1 / 2 := by
  sorry

end fraction_equality_l246_24618


namespace triangulation_count_l246_24638

/-- A triangulation of a non-self-intersecting n-gon using m interior vertices -/
structure Triangulation where
  n : ℕ  -- number of vertices in the n-gon
  m : ℕ  -- number of interior vertices
  N : ℕ  -- number of triangles in the triangulation
  h1 : N > 0  -- there is at least one triangle
  h2 : n ≥ 3  -- n-gon has at least 3 vertices

/-- The number of triangles in a triangulation of an n-gon with m interior vertices -/
theorem triangulation_count (T : Triangulation) : T.N = T.n + 2 * T.m - 2 := by
  sorry

end triangulation_count_l246_24638


namespace fifth_term_value_l246_24684

/-- A geometric sequence with positive terms satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  first_second_sum : a 1 + 2 * a 2 = 4
  fourth_squared : a 4 ^ 2 = 4 * a 3 * a 7

/-- The fifth term of the geometric sequence is 1/8 -/
theorem fifth_term_value (seq : GeometricSequence) : seq.a 5 = 1/8 := by
  sorry

end fifth_term_value_l246_24684


namespace trigonometric_identity_l246_24624

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 2 / Real.cos (10 * π / 180) := by
  sorry

end trigonometric_identity_l246_24624


namespace parallel_line_slope_l246_24628

/-- The slope of any line parallel to 3x + 6y = -21 is -1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a + 6 * b = -21) :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x y : ℝ), 3 * x + 6 * y = -21 → y = m * x + c :=
sorry

end parallel_line_slope_l246_24628


namespace total_selling_price_is_correct_l246_24691

def cycle_price : ℕ := 2000
def scooter_price : ℕ := 25000
def bike_price : ℕ := 60000

def cycle_loss_percent : ℚ := 10 / 100
def scooter_loss_percent : ℚ := 15 / 100
def bike_loss_percent : ℚ := 5 / 100

def selling_price (price : ℕ) (loss_percent : ℚ) : ℚ :=
  price - (price * loss_percent)

def total_selling_price : ℚ :=
  selling_price cycle_price cycle_loss_percent +
  selling_price scooter_price scooter_loss_percent +
  selling_price bike_price bike_loss_percent

theorem total_selling_price_is_correct :
  total_selling_price = 80050 := by
  sorry

end total_selling_price_is_correct_l246_24691


namespace square_perimeter_problem_l246_24660

theorem square_perimeter_problem (perimeter_A perimeter_B : ℝ) 
  (h1 : perimeter_A = 20)
  (h2 : perimeter_B = 40)
  (h3 : ∀ (side_A side_B : ℝ), 
    perimeter_A = 4 * side_A → 
    perimeter_B = 4 * side_B → 
    ∃ (perimeter_C : ℝ), perimeter_C = 4 * (side_A + side_B)) :
  ∃ (perimeter_C : ℝ), perimeter_C = 60 := by
  sorry

end square_perimeter_problem_l246_24660


namespace f_not_monotonic_l246_24658

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

-- State that f is an even function
axiom f_even (m : ℝ) : ∀ x, f m x = f m (-x)

-- Define the derivative of f
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * (m - 1) * x - 2 * m

-- Theorem: f is not monotonic on (-∞, 3)
theorem f_not_monotonic (m : ℝ) : 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x < f m y) ∧ 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x > f m y) :=
sorry

end f_not_monotonic_l246_24658


namespace equilateral_triangle_area_perimeter_ratio_l246_24651

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l246_24651


namespace solution_set_l246_24667

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f(x+1) is an odd function
axiom f_odd : ∀ x : ℝ, f (x + 1) = -f (-x - 1)

-- For any unequal real numbers x₁, x₂: x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁)
axiom f_inequality : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Theorem: The solution set of f(2-x) < 0 is (1,+∞)
theorem solution_set : {x : ℝ | f (2 - x) < 0} = Set.Ioi 1 := by
  sorry

end solution_set_l246_24667


namespace f_13_equals_223_l246_24654

/-- Define the function f for natural numbers -/
def f (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem stating that f(13) equals 223 -/
theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end f_13_equals_223_l246_24654


namespace crafts_club_beads_l246_24646

/-- The number of beads needed for a group of people making necklaces -/
def total_beads (num_members : ℕ) (necklaces_per_member : ℕ) (beads_per_necklace : ℕ) : ℕ :=
  num_members * necklaces_per_member * beads_per_necklace

theorem crafts_club_beads : 
  total_beads 9 2 50 = 900 := by
  sorry

end crafts_club_beads_l246_24646


namespace probability_specific_order_correct_l246_24647

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Represents the number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Represents the number of cards to be drawn -/
def cardsDrawn : ℕ := 4

/-- Calculates the probability of drawing one card from each suit in a specific order -/
def probabilitySpecificOrder : ℚ :=
  (cardsPerSuit : ℚ) / standardDeck *
  (cardsPerSuit : ℚ) / (standardDeck - 1) *
  (cardsPerSuit : ℚ) / (standardDeck - 2) *
  (cardsPerSuit : ℚ) / (standardDeck - 3)

/-- Theorem: The probability of drawing one card from each suit in a specific order is 2197/499800 -/
theorem probability_specific_order_correct :
  probabilitySpecificOrder = 2197 / 499800 := by
  sorry

end probability_specific_order_correct_l246_24647


namespace wall_bricks_count_l246_24678

def wall_problem (initial_courses : ℕ) (bricks_per_course : ℕ) (added_courses : ℕ) : ℕ :=
  let total_courses := initial_courses + added_courses
  let total_bricks := total_courses * bricks_per_course
  let removed_bricks := bricks_per_course / 2
  total_bricks - removed_bricks

theorem wall_bricks_count :
  wall_problem 3 400 2 = 1800 := by
  sorry

end wall_bricks_count_l246_24678


namespace convex_number_probability_l246_24676

-- Define the set of digits
def Digits : Finset Nat := {1, 2, 3, 4}

-- Define a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  digit_in_range : hundreds ∈ Digits ∧ tens ∈ Digits ∧ units ∈ Digits

-- Define a convex number
def isConvex (n : ThreeDigitNumber) : Prop :=
  n.hundreds < n.tens ∧ n.tens > n.units

-- Define the set of all possible three-digit numbers
def allNumbers : Finset ThreeDigitNumber := sorry

-- Define the set of convex numbers
def convexNumbers : Finset ThreeDigitNumber := sorry

-- Theorem to prove
theorem convex_number_probability :
  (Finset.card convexNumbers : Rat) / (Finset.card allNumbers : Rat) = 1 / 3 := by sorry

end convex_number_probability_l246_24676


namespace gpa_probability_at_least_3_6_l246_24688

/-- Represents the possible grades a student can receive. -/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value. -/
def gradeToPoints (g : Grade) : ℕ :=
  match g with
  | Grade.A => 4
  | Grade.B => 3
  | Grade.C => 2
  | Grade.D => 1

/-- Calculates the GPA given a list of grades. -/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 5

/-- Represents the probability distribution of grades for a class. -/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades. -/
def englishProb : GradeProbability :=
  { probA := 1/4, probB := 1/3, probC := 5/12, probD := 0 }

/-- The probability distribution for History grades. -/
def historyProb : GradeProbability :=
  { probA := 1/3, probB := 1/4, probC := 5/12, probD := 0 }

/-- Theorem stating the probability of achieving a GPA of at least 3.6. -/
theorem gpa_probability_at_least_3_6 :
  let allGrades := [Grade.A, Grade.A, Grade.A] -- Math, Science, Art
  let probAtLeast3_6 := (
    englishProb.probA * historyProb.probA +
    englishProb.probA * historyProb.probB +
    englishProb.probB * historyProb.probA +
    englishProb.probB * historyProb.probB
  )
  probAtLeast3_6 = 49/144 := by sorry

end gpa_probability_at_least_3_6_l246_24688


namespace no_valid_decagon_labeling_l246_24612

/-- Represents a labeling of a regular decagon with center -/
def DecagonLabeling := Fin 11 → Fin 10

/-- The sum of digits on a line through the center of the decagon -/
def line_sum (l : DecagonLabeling) (i j : Fin 11) : ℕ :=
  l i + l j + l 10

/-- Checks if a labeling is valid according to the problem constraints -/
def is_valid_labeling (l : DecagonLabeling) : Prop :=
  (∀ i j : Fin 11, i ≠ j → l i ≠ l j) ∧
  (line_sum l 0 4 = line_sum l 1 5) ∧
  (line_sum l 0 4 = line_sum l 2 6) ∧
  (line_sum l 0 4 = line_sum l 3 7) ∧
  (line_sum l 0 4 = line_sum l 4 8)

theorem no_valid_decagon_labeling :
  ¬∃ l : DecagonLabeling, is_valid_labeling l :=
sorry

end no_valid_decagon_labeling_l246_24612


namespace end_zeros_imply_n_greater_than_seven_l246_24643

theorem end_zeros_imply_n_greater_than_seven (m n : ℕ+) 
  (h1 : m > n) 
  (h2 : (22220038 ^ m.val - 22220038 ^ n.val) % (10 ^ 8) = 0) : 
  n > 7 := by
  sorry

end end_zeros_imply_n_greater_than_seven_l246_24643


namespace prop_two_prop_three_prop_one_false_prop_four_false_l246_24616

-- Proposition ②
theorem prop_two (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

-- Proposition ③
theorem prop_three (a b : ℝ) : a > b → a^3 > b^3 := by sorry

-- Proposition ① is false
theorem prop_one_false : ∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2) := by sorry

-- Proposition ④ is false
theorem prop_four_false : ∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2) := by sorry

end prop_two_prop_three_prop_one_false_prop_four_false_l246_24616


namespace complex_modulus_problem_l246_24694

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I) * z = 1 - 2 * Complex.I^3 →
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_problem_l246_24694
