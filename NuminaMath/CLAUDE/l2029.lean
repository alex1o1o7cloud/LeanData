import Mathlib

namespace m_range_l2029_202936

def p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

theorem m_range (m : ℝ) (h : p m ∧ q m) : m ≤ 1 := by
  sorry

end m_range_l2029_202936


namespace hyperbola_eccentricity_l2029_202940

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, foci at (-c, 0) and (c, 0),
    and an isosceles right triangle with hypotenuse connecting the foci,
    if the midpoint of the legs of this triangle lies on the hyperbola,
    then c/a = (√10 + √2)/2 -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ∧ 
              x = -c/2 ∧ y = c/2) →
  c/a = (Real.sqrt 10 + Real.sqrt 2)/2 := by
sorry

end hyperbola_eccentricity_l2029_202940


namespace no_real_roots_composition_l2029_202965

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem no_real_roots_composition (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) :=
by
  sorry

end no_real_roots_composition_l2029_202965


namespace f_f_zero_l2029_202960

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end f_f_zero_l2029_202960


namespace inverse_f_128_l2029_202901

/-- Given a function f: ℝ → ℝ satisfying f(4) = 2 and f(2x) = 2f(x) for all x,
    prove that f⁻¹(128) = 256 -/
theorem inverse_f_128 (f : ℝ → ℝ) (h1 : f 4 = 2) (h2 : ∀ x, f (2 * x) = 2 * f x) :
  f⁻¹ 128 = 256 := by
  sorry

end inverse_f_128_l2029_202901


namespace room_extension_ratio_l2029_202984

/-- Given a room with original length, width, and an extension to the length,
    prove that the ratio of the new total length to the new perimeter is 35:100. -/
theorem room_extension_ratio (original_length width extension : ℕ) 
  (h1 : original_length = 25)
  (h2 : width = 15)
  (h3 : extension = 10) :
  (original_length + extension) * 100 = 35 * (2 * (original_length + extension + width)) :=
by sorry

end room_extension_ratio_l2029_202984


namespace no_maximum_b_plus_c_l2029_202966

/-- A cubic function f(x) = x^3 + bx^2 + cx + d -/
def cubic_function (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of the cubic function -/
def cubic_derivative (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem no_maximum_b_plus_c :
  ∀ b c d : ℝ,
  (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b c x ≤ 0) →
  ¬∃ M : ℝ, ∀ b' c' : ℝ, 
    (∀ x ∈ Set.Icc (-1) 2, cubic_derivative b' c' x ≤ 0) →
    b' + c' ≤ M :=
by sorry

end no_maximum_b_plus_c_l2029_202966


namespace cubic_function_properties_l2029_202951

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≤ 20) →
  (a = 3 ∧ b = 9) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = 13 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ 13) ∨
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a b c y ≥ -7) :=
by sorry

end cubic_function_properties_l2029_202951


namespace factorization_problem_1_l2029_202967

theorem factorization_problem_1 (a : ℝ) :
  3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 := by
  sorry

end factorization_problem_1_l2029_202967


namespace sum_of_coefficients_is_five_l2029_202998

/-- Given two real-valued functions f and h, where f is linear and h is affine,
    and a condition relating their composition to a linear function,
    prove that the sum of the coefficients of f is 5. -/
theorem sum_of_coefficients_is_five
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h : ℝ → ℝ)
  (h_def : ∀ x, h x = 3 * x - 6)
  (f_def : ∀ x, f x = a * x + b)
  (composition_condition : ∀ x, h (f x) = 4 * x + 5) :
  a + b = 5 := by
sorry

end sum_of_coefficients_is_five_l2029_202998


namespace russian_pairing_probability_l2029_202995

def total_players : ℕ := 10
def russian_players : ℕ := 4

theorem russian_pairing_probability :
  let remaining_players := total_players - 1
  let remaining_russian_players := russian_players - 1
  let first_pair_prob := remaining_russian_players / remaining_players
  let second_pair_prob := 1 / (remaining_players - 1)
  first_pair_prob * second_pair_prob = 1 / 21 := by
  sorry

end russian_pairing_probability_l2029_202995


namespace oil_containers_per_box_l2029_202964

theorem oil_containers_per_box :
  let trucks_with_20_boxes : ℕ := 7
  let boxes_per_truck_20 : ℕ := 20
  let trucks_with_12_boxes : ℕ := 5
  let boxes_per_truck_12 : ℕ := 12
  let total_trucks_after_redistribution : ℕ := 10
  let containers_per_truck_after_redistribution : ℕ := 160

  let total_boxes : ℕ := trucks_with_20_boxes * boxes_per_truck_20 + trucks_with_12_boxes * boxes_per_truck_12
  let total_containers : ℕ := total_trucks_after_redistribution * containers_per_truck_after_redistribution

  (total_containers / total_boxes : ℚ) = 8 := by sorry

end oil_containers_per_box_l2029_202964


namespace rhombus_area_from_diagonals_l2029_202904

/-- The area of a rhombus given its diagonals -/
theorem rhombus_area_from_diagonals (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  (1 / 2 : ℝ) * d1 * d2 = 192 := by
  sorry

end rhombus_area_from_diagonals_l2029_202904


namespace ferry_journey_difference_l2029_202913

/-- Represents the properties of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The ferry problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : FerryJourney),
    -- Ferry P properties
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    -- Ferry Q properties
    Q.distance = 2 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    -- The time difference is 1 hour
    Q.time - P.time = 1

/-- Theorem stating the solution to the ferry problem -/
theorem ferry_journey_difference : ferryProblem := by
  sorry

end ferry_journey_difference_l2029_202913


namespace points_difference_l2029_202939

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of matches in a round-robin tournament -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_matches num_teams * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := total_matches num_teams * 2 * tie_points

/-- The theorem stating the difference between maximum and minimum total points -/
theorem points_difference :
  max_total_points - min_total_points = 30 := by sorry

end points_difference_l2029_202939


namespace mini_van_capacity_correct_l2029_202922

/-- Represents the capacity of a mini-van's tank in liters -/
def mini_van_capacity : ℝ := 65

/-- Represents the service cost per vehicle in dollars -/
def service_cost : ℝ := 2.10

/-- Represents the fuel cost per liter in dollars -/
def fuel_cost : ℝ := 0.60

/-- Represents the number of mini-vans -/
def num_mini_vans : ℕ := 3

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the total cost in dollars -/
def total_cost : ℝ := 299.1

/-- Represents the ratio of truck tank capacity to mini-van tank capacity -/
def truck_capacity_ratio : ℝ := 2.2

theorem mini_van_capacity_correct :
  service_cost * (num_mini_vans + num_trucks) +
  fuel_cost * (num_mini_vans * mini_van_capacity + num_trucks * (truck_capacity_ratio * mini_van_capacity)) =
  total_cost := by sorry

end mini_van_capacity_correct_l2029_202922


namespace tax_rate_calculation_l2029_202937

theorem tax_rate_calculation (total_value tax_free_allowance tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_allowance = 600 →
  tax_paid = 78.4 →
  (tax_paid / (total_value - tax_free_allowance)) * 100 = 7 := by
sorry

end tax_rate_calculation_l2029_202937


namespace geometric_sequence_a4_l2029_202925

/-- A geometric sequence with real terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a → a 2 = 9 → a 6 = 1 → a 4 = 3 := by
  sorry

end geometric_sequence_a4_l2029_202925


namespace cube_iff_diagonal_perpendicular_l2029_202971

/-- A rectangular parallelepiped -/
structure RectangularParallelepiped where
  -- Add necessary fields and properties here

/-- Predicate for a rectangular parallelepiped being a cube -/
def is_cube (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Predicate for the diagonal perpendicularity property -/
def diagonal_perpendicular_property (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Theorem stating the equivalence of the cube property and the diagonal perpendicularity property -/
theorem cube_iff_diagonal_perpendicular (S : RectangularParallelepiped) :
  is_cube S ↔ diagonal_perpendicular_property S :=
sorry

end cube_iff_diagonal_perpendicular_l2029_202971


namespace ariels_age_multiplier_l2029_202970

theorem ariels_age_multiplier :
  let current_age : ℕ := 5
  let years_passed : ℕ := 15
  let future_age : ℕ := current_age + years_passed
  ∃ (multiplier : ℕ), future_age = multiplier * current_age ∧ multiplier = 4 :=
by sorry

end ariels_age_multiplier_l2029_202970


namespace gold_cube_profit_l2029_202957

-- Define the cube's side length in cm
def cube_side : ℝ := 6

-- Define the density of gold in g/cm³
def gold_density : ℝ := 19

-- Define the buying price per gram in dollars
def buying_price : ℝ := 60

-- Define the selling price multiplier
def selling_multiplier : ℝ := 1.5

-- Theorem statement
theorem gold_cube_profit :
  let volume : ℝ := cube_side ^ 3
  let mass : ℝ := gold_density * volume
  let cost : ℝ := mass * buying_price
  let selling_price : ℝ := cost * selling_multiplier
  selling_price - cost = 123120 := by
  sorry

end gold_cube_profit_l2029_202957


namespace factorization_equality_l2029_202933

theorem factorization_equality (a b x y : ℝ) :
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = a * b * (x - y)^2 * (a * x - a * y - b) := by
  sorry

end factorization_equality_l2029_202933


namespace min_value_fraction_sum_l2029_202905

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ (4 / a₀) + (9 / b₀) = 25 :=
by sorry

end min_value_fraction_sum_l2029_202905


namespace dealer_net_profit_dealer_net_profit_is_97_20_l2029_202983

/-- Calculates the dealer's net profit from selling a desk --/
theorem dealer_net_profit (purchase_price : ℝ) (markup_rate : ℝ) (discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let selling_price := purchase_price / (1 - markup_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let total_payment := discounted_price * (1 + sales_tax_rate)
  let commission := discounted_price * commission_rate
  total_payment - purchase_price - commission

/-- Proves that the dealer's net profit is $97.20 under the given conditions --/
theorem dealer_net_profit_is_97_20 :
  dealer_net_profit 150 0.5 0.2 0.05 0.02 = 97.20 := by
  sorry

end dealer_net_profit_dealer_net_profit_is_97_20_l2029_202983


namespace pizza_distribution_l2029_202990

/-- Calculates the number of slices each person gets in a group pizza order -/
def slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

/-- Proves that given 18 people, 6 pizzas with 9 slices each, each person gets 3 slices -/
theorem pizza_distribution :
  slices_per_person 18 6 9 = 3 := by
  sorry

#eval slices_per_person 18 6 9

end pizza_distribution_l2029_202990


namespace problem_solution_l2029_202938

/-- Check if a number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Check if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k, n = k * k

/-- Get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- Get the hundreds digit of a number -/
def hundredsDigit (n : ℕ) : ℕ := (n / 100) % 10

/-- Create a new number by placing one digit in front of another number -/
def placeDigitInFront (digit : ℕ) (n : ℕ) : ℕ := digit * 100 + n

theorem problem_solution :
  let a : ℕ := 3
  let b : ℕ := 44
  let c : ℕ := 149
  (a < 10 ∧ b < 100 ∧ b ≥ 10 ∧ c < 1000 ∧ c ≥ 100) ∧
  (isOdd a ∧ isEven b ∧ isOdd c) ∧
  (lastTwoDigits (a * b * c) = 68) ∧
  (isPerfectSquare (a + b + c)) ∧
  (isPerfectSquare (placeDigitInFront (hundredsDigit c) b) ∧ isPerfectSquare (c % 100)) := by
  sorry


end problem_solution_l2029_202938


namespace monic_quartic_with_specific_roots_l2029_202907

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 14*x^3 + 57*x^2 - 132*x + 36

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 14*x^3 + 57*x^2 - 132*x + 36) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 4 - √7 is a root
  p (4 - Real.sqrt 7) = 0 :=
sorry

end monic_quartic_with_specific_roots_l2029_202907


namespace percentage_fraction_equality_l2029_202932

theorem percentage_fraction_equality : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by
sorry

end percentage_fraction_equality_l2029_202932


namespace expression_equality_l2029_202979

theorem expression_equality : (8 * 10^10) / (2 * 10^5 * 4) = 100000 := by
  sorry

end expression_equality_l2029_202979


namespace twelfth_term_equals_three_over_512_l2029_202902

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfthTerm : ℚ :=
  geometricSequenceTerm 12 (1/2) 12

theorem twelfth_term_equals_three_over_512 :
  twelfthTerm = 3/512 := by
  sorry

end twelfth_term_equals_three_over_512_l2029_202902


namespace paving_cost_calculation_l2029_202952

-- Define the room dimensions
def room_length : ℝ := 10
def room_width : ℝ := 4.75

-- Define the paving rate
def paving_rate : ℝ := 900

-- Calculate the area of the room
def room_area : ℝ := room_length * room_width

-- Calculate the total cost of paving
def paving_cost : ℝ := room_area * paving_rate

-- Theorem to prove
theorem paving_cost_calculation : paving_cost = 42750 := by
  sorry

end paving_cost_calculation_l2029_202952


namespace sufficient_but_not_necessary_l2029_202977

-- Define a proposition P to represent the given condition
variable (P : Prop)

-- Define a proposition Q to represent the conclusion
variable (Q : Prop)

-- Theorem stating that P is sufficient but not necessary for Q
theorem sufficient_but_not_necessary : (P → Q) ∧ ¬(Q → P) := by
  sorry

end sufficient_but_not_necessary_l2029_202977


namespace hcd_8100_270_minus_8_l2029_202906

theorem hcd_8100_270_minus_8 : Nat.gcd 8100 270 - 8 = 262 := by
  sorry

end hcd_8100_270_minus_8_l2029_202906


namespace clothes_to_earnings_ratio_l2029_202929

/-- Proves that the ratio of clothes spending to initial earnings is 1:2 given the conditions --/
theorem clothes_to_earnings_ratio 
  (initial_earnings : ℚ)
  (clothes_spending : ℚ)
  (book_spending : ℚ)
  (remaining : ℚ)
  (h1 : initial_earnings = 600)
  (h2 : book_spending = (initial_earnings - clothes_spending) / 2)
  (h3 : remaining = initial_earnings - clothes_spending - book_spending)
  (h4 : remaining = 150) :
  clothes_spending / initial_earnings = 1 / 2 := by
sorry

end clothes_to_earnings_ratio_l2029_202929


namespace womans_age_multiple_l2029_202987

theorem womans_age_multiple (W S k : ℕ) : 
  W = k * S + 3 →
  W + S = 84 →
  S = 27 →
  k = 2 := by
sorry

end womans_age_multiple_l2029_202987


namespace maria_berry_purchase_l2029_202980

/-- The number of cartons Maria needs to buy -/
def cartons_to_buy (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

/-- Theorem stating that Maria needs to buy 9 more cartons of berries -/
theorem maria_berry_purchase : cartons_to_buy 21 4 8 = 9 := by
  sorry

end maria_berry_purchase_l2029_202980


namespace greatest_common_divisor_of_84_and_n_l2029_202988

theorem greatest_common_divisor_of_84_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
    {d | d > 0 ∧ d ∣ 84 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ 84 ∧ d ∣ n ∧ 
    ∀ (k : ℕ), k > 0 ∧ k ∣ 84 ∧ k ∣ n → k ≤ d) →
  4 = (Nat.gcd 84 n) :=
sorry

end greatest_common_divisor_of_84_and_n_l2029_202988


namespace mother_age_is_36_l2029_202991

/-- Petra's age -/
def petra_age : ℕ := 11

/-- The sum of Petra's and her mother's ages -/
def age_sum : ℕ := 47

/-- Petra's mother's age -/
def mother_age : ℕ := age_sum - petra_age

/-- Theorem: Petra's mother is 36 years old -/
theorem mother_age_is_36 : mother_age = 36 := by
  sorry

end mother_age_is_36_l2029_202991


namespace second_meeting_at_5_4_minutes_l2029_202985

/-- Represents the race scenario between George and Henry --/
structure RaceScenario where
  pool_length : ℝ
  george_start_time : ℝ
  henry_start_time : ℝ
  first_meeting_time : ℝ
  first_meeting_distance : ℝ

/-- Calculates the time of the second meeting given a race scenario --/
def second_meeting_time (scenario : RaceScenario) : ℝ :=
  sorry

/-- The main theorem stating that the second meeting occurs 5.4 minutes after George's start --/
theorem second_meeting_at_5_4_minutes (scenario : RaceScenario) 
  (h1 : scenario.pool_length = 50)
  (h2 : scenario.george_start_time = 0)
  (h3 : scenario.henry_start_time = 1)
  (h4 : scenario.first_meeting_time = 3)
  (h5 : scenario.first_meeting_distance = 25) : 
  second_meeting_time scenario = 5.4 := by
  sorry

end second_meeting_at_5_4_minutes_l2029_202985


namespace arithmetic_sequence_20th_term_l2029_202978

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
  sorry

end arithmetic_sequence_20th_term_l2029_202978


namespace expected_sixes_is_one_third_l2029_202911

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the probability of rolling a 6 on a single die
def prob_six : ℚ := 1 / die_sides

-- Define the probability of not rolling a 6 on a single die
def prob_not_six : ℚ := 1 - prob_six

-- Define the expected number of 6's when rolling two dice
def expected_sixes : ℚ := 
  2 * (prob_six * prob_six) + 
  1 * (2 * prob_six * prob_not_six) + 
  0 * (prob_not_six * prob_not_six)

-- Theorem statement
theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 :=
sorry

end expected_sixes_is_one_third_l2029_202911


namespace shooting_probability_l2029_202968

/-- The probability of person A hitting the target in a single shot -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target in a single shot -/
def prob_B : ℚ := 4/5

/-- The probability that A has taken two shots when they stop shooting -/
def prob_A_two_shots : ℚ := 19/400

theorem shooting_probability :
  let p1 := (1 - prob_A) * (1 - prob_B) * prob_A
  let p2 := (1 - prob_A) * (1 - prob_B) * (1 - prob_A) * prob_B
  p1 + p2 = prob_A_two_shots := by sorry

end shooting_probability_l2029_202968


namespace unique_solution_l2029_202997

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem -/
def AdditionProblem (A B C : Digit) : Prop :=
  (C.val * 100 + C.val * 10 + A.val) + (B.val * 100 + 2 * 10 + B.val) = A.val * 100 + 8 * 10 + 8

theorem unique_solution :
  ∃! (A B C : Digit), AdditionProblem A B C ∧ A.val * B.val * C.val = 42 :=
sorry

end unique_solution_l2029_202997


namespace average_MTWT_is_48_l2029_202910

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 43

/-- The temperature on Friday -/
def temp_Friday : ℝ := 35

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = average_some_days :=
by sorry

end average_MTWT_is_48_l2029_202910


namespace volume_increase_when_radius_doubled_l2029_202945

/-- The volume increase of a right circular cylinder when its radius is doubled -/
theorem volume_increase_when_radius_doubled (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 6 → 
  π * (2*r)^2 * h - π * r^2 * h = 18 := by
  sorry

end volume_increase_when_radius_doubled_l2029_202945


namespace dog_weight_gain_l2029_202935

/-- Given a golden retriever that:
    - Is 8 years old
    - Currently weighs 88 pounds
    - Weighed 40 pounds at 1 year old
    Prove that the average yearly weight gain is 6 pounds. -/
theorem dog_weight_gain (current_weight : ℕ) (age : ℕ) (initial_weight : ℕ) 
  (h1 : current_weight = 88)
  (h2 : age = 8)
  (h3 : initial_weight = 40) :
  (current_weight - initial_weight) / (age - 1) = 6 :=
sorry

end dog_weight_gain_l2029_202935


namespace unique_prime_satisfying_conditions_l2029_202928

theorem unique_prime_satisfying_conditions :
  ∃! (n : ℕ), n.Prime ∧ 
    (n^2 + 10).Prime ∧ 
    (n^2 - 2).Prime ∧ 
    (n^3 + 6).Prime ∧ 
    (n^5 + 36).Prime ∧ 
    n = 7 := by
  sorry

end unique_prime_satisfying_conditions_l2029_202928


namespace total_rainfall_three_days_l2029_202973

/-- Calculates the total rainfall over three days given specific conditions --/
theorem total_rainfall_three_days 
  (monday_hours : ℕ) 
  (monday_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (tuesday_rate : ℕ) 
  (wednesday_hours : ℕ) 
  (h_monday : monday_hours = 7 ∧ monday_rate = 1)
  (h_tuesday : tuesday_hours = 4 ∧ tuesday_rate = 2)
  (h_wednesday : wednesday_hours = 2)
  (h_wednesday_rate : wednesday_hours * (2 * tuesday_rate) = 8) :
  monday_hours * monday_rate + 
  tuesday_hours * tuesday_rate + 
  wednesday_hours * (2 * tuesday_rate) = 23 := by
sorry


end total_rainfall_three_days_l2029_202973


namespace bill_after_30_days_l2029_202963

/-- The amount owed after applying late charges -/
def amount_owed (initial_bill : ℝ) (late_charge_rate : ℝ) (days : ℕ) : ℝ :=
  initial_bill * (1 + late_charge_rate) ^ (days / 10)

/-- Theorem stating the amount owed after 30 days -/
theorem bill_after_30_days (initial_bill : ℝ) (late_charge_rate : ℝ) :
  initial_bill = 500 →
  late_charge_rate = 0.02 →
  amount_owed initial_bill late_charge_rate 30 = 530.604 :=
by
  sorry

#eval amount_owed 500 0.02 30

end bill_after_30_days_l2029_202963


namespace vector_difference_magnitude_l2029_202942

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by sorry

end vector_difference_magnitude_l2029_202942


namespace inverse_proportion_problem_l2029_202993

/-- Given two real numbers a and b that are inversely proportional,
    prove that if a + b = 30 and a - b = 8, then when a = 6, b = 209/6 -/
theorem inverse_proportion_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
    (h2 : a + b = 30) (h3 : a - b = 8) : 
    (a = 6) → (b = 209 / 6) := by
  sorry

end inverse_proportion_problem_l2029_202993


namespace tree_leaves_problem_l2029_202999

/-- The number of leaves remaining after dropping 1/10 of leaves n times -/
def leavesRemaining (initialLeaves : ℕ) (n : ℕ) : ℚ :=
  initialLeaves * (9/10)^n

/-- The proposition that a tree with the given leaf-dropping pattern initially had 311 leaves -/
theorem tree_leaves_problem : ∃ (initialLeaves : ℕ),
  (leavesRemaining initialLeaves 4).num = 204 * (leavesRemaining initialLeaves 4).den ∧
  initialLeaves = 311 := by
  sorry


end tree_leaves_problem_l2029_202999


namespace ellipse_equation_l2029_202924

theorem ellipse_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : c / a = 2 / 3) (h5 : a = 3) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 9 + y^2 / 5 = 1) :=
by sorry

end ellipse_equation_l2029_202924


namespace x_squared_minus_five_is_quadratic_l2029_202989

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5

/-- Theorem: x² - 5 = 0 is a quadratic equation -/
theorem x_squared_minus_five_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_minus_five_is_quadratic_l2029_202989


namespace square_plus_reciprocal_square_l2029_202915

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end square_plus_reciprocal_square_l2029_202915


namespace triangle_not_right_angle_l2029_202986

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A + B + C = 180) (h5 : A / 3 = B / 4) (h6 : A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end triangle_not_right_angle_l2029_202986


namespace product_seven_consecutive_divisible_by_ten_l2029_202976

/-- The product of any seven consecutive positive integers is divisible by 10 -/
theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
sorry

end product_seven_consecutive_divisible_by_ten_l2029_202976


namespace circle_division_sum_l2029_202931

/-- The sum of numbers on a circle after n steps of division -/
def circleSum (n : ℕ) : ℕ :=
  2 * 3^n

/-- The process of dividing the circle and summing numbers -/
def divideAndSum : ℕ → ℕ
  | 0 => 2  -- Initial sum: 1 + 1
  | n + 1 => 3 * divideAndSum n

theorem circle_division_sum (n : ℕ) :
  divideAndSum n = circleSum n := by
  sorry

end circle_division_sum_l2029_202931


namespace tan_negative_405_degrees_l2029_202961

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end tan_negative_405_degrees_l2029_202961


namespace saltwater_animals_per_aquarium_l2029_202941

theorem saltwater_animals_per_aquarium :
  ∀ (num_aquariums : ℕ) (total_animals : ℕ) (animals_per_aquarium : ℕ),
    num_aquariums = 26 →
    total_animals = 52 →
    total_animals = num_aquariums * animals_per_aquarium →
    animals_per_aquarium = 2 := by
  sorry

end saltwater_animals_per_aquarium_l2029_202941


namespace simplify_fraction_l2029_202918

theorem simplify_fraction : (121 : ℚ) / 13310 = 1 / 110 := by sorry

end simplify_fraction_l2029_202918


namespace smallest_number_600_times_prime_divisors_l2029_202948

theorem smallest_number_600_times_prime_divisors :
  ∃ (N : ℕ), N > 1 ∧
  (∀ p : ℕ, Nat.Prime p → p ∣ N → N ≥ 600 * p) ∧
  (∀ M : ℕ, M > 1 → (∀ q : ℕ, Nat.Prime q → q ∣ M → M ≥ 600 * q) → M ≥ N) ∧
  N = 1944 :=
by sorry

end smallest_number_600_times_prime_divisors_l2029_202948


namespace time_after_1457_minutes_l2029_202953

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Converts a number to a 24-hour time -/
def minutesToTime (m : ℕ) : Time :=
  sorry

theorem time_after_1457_minutes :
  let start_time : Time := ⟨3, 0, sorry⟩
  let added_minutes : ℕ := 1457
  let end_time : Time := addMinutes start_time added_minutes
  end_time = ⟨3, 17, sorry⟩ :=
sorry

end time_after_1457_minutes_l2029_202953


namespace train_platform_crossing_time_l2029_202912

/-- Calculates the time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_ms : ℝ)
  (time_to_pass_man : ℝ)
  (platform_length : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_ms = 20)
  (h3 : time_to_pass_man = 16)
  (h4 : platform_length = 280)
  (h5 : train_speed_ms = train_speed_kmph * 1000 / 3600) :
  let train_length := train_speed_ms * time_to_pass_man
  let total_distance := train_length + platform_length
  total_distance / train_speed_ms = 30 := by
  sorry

end train_platform_crossing_time_l2029_202912


namespace gensokyo_tennis_club_meeting_day_l2029_202950

/-- The Gensokyo Tennis Club problem -/
theorem gensokyo_tennis_club_meeting_day :
  let total_players : ℕ := 2016
  let total_courts : ℕ := 1008
  let reimu_start : ℕ := 123
  let marisa_start : ℕ := 876
  let winner_move (court : ℕ) : ℕ := if court > 1 then court - 1 else 1
  let loser_move (court : ℕ) : ℕ := if court < total_courts then court + 1 else total_courts
  let reimu_path (day : ℕ) : ℕ := if day < reimu_start then reimu_start - day else 1
  let marisa_path (day : ℕ) : ℕ :=
    if day ≤ (total_courts - marisa_start) then
      marisa_start + day
    else
      total_courts - (day - (total_courts - marisa_start))
  ∃ (n : ℕ), n > 0 ∧ reimu_path n = marisa_path n ∧ 
    ∀ (m : ℕ), m > 0 ∧ m < n → reimu_path m ≠ marisa_path m :=
by
  sorry

end gensokyo_tennis_club_meeting_day_l2029_202950


namespace equation_is_quadratic_l2029_202949

theorem equation_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∀ x, 3 * (x + 1)^2 = 2 * (x - 2) ↔ a * x^2 + b * x + c = 0 :=
by sorry

end equation_is_quadratic_l2029_202949


namespace expand_equality_l2029_202947

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := (binomial 10 2 : ℝ) * x^8 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := (binomial 10 3 : ℝ) * x^7 * y^3

-- Main theorem
theorem expand_equality (p q : ℝ) 
  (h_pos_p : p > 0) 
  (h_pos_q : q > 0) 
  (h_sum : p + q = 2) 
  (h_equal : third_term p q = fourth_term p q) : 
  p = 16/11 := by
sorry

end expand_equality_l2029_202947


namespace super_ball_distance_l2029_202954

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  -- Definition of total distance calculation
  sorry

/-- Theorem stating the total distance traveled by the ball -/
theorem super_ball_distance :
  let initialHeight : ℝ := 150
  let reboundRatio : ℝ := 2/3
  let bounces : ℕ := 5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |totalDistance initialHeight reboundRatio bounces - 591.67| < ε :=
by
  sorry


end super_ball_distance_l2029_202954


namespace line_equation_through_M_intersecting_C_l2029_202944

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = -1 + 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ

-- Define the point M
def point_M : ℝ × ℝ := (-1, 2)

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- State the theorem
theorem line_equation_through_M_intersecting_C :
  ∀ A B : ℝ × ℝ,
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  A ≠ B →
  line_through_points point_M.1 point_M.2 A.1 A.2 B.1 B.2 →
  point_M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ x y : ℝ,
    (Real.sqrt 15 * x - 5 * y + Real.sqrt 15 + 10 = 0) ∨
    (Real.sqrt 15 * x + 5 * y + Real.sqrt 15 - 10 = 0) :=
by sorry

end line_equation_through_M_intersecting_C_l2029_202944


namespace problem_statement_l2029_202908

theorem problem_statement (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^4374 - 1/x^4374 = -Complex.I := by
  sorry

end problem_statement_l2029_202908


namespace correct_calculation_l2029_202994

theorem correct_calculation : ∃ x : ℝ, 5 * x = 40 ∧ 2 * x = 16 := by
  sorry

end correct_calculation_l2029_202994


namespace binomial_constant_term_l2029_202917

theorem binomial_constant_term (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 4*n = 5*r) ↔ n = 10 := by sorry

end binomial_constant_term_l2029_202917


namespace sector_max_area_l2029_202943

/-- Given a sector with perimeter 20 cm, prove that the area is maximized when the central angle is 2 radians and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (θ : ℝ) :
  r > 0 →
  r * θ + 2 * r = 20 →
  0 < θ →
  θ ≤ 2 * π →
  (∀ r' θ', r' > 0 → r' * θ' + 2 * r' = 20 → 0 < θ' → θ' ≤ 2 * π → 
    1/2 * r * r * θ ≥ 1/2 * r' * r' * θ') →
  θ = 2 ∧ 1/2 * r * r * θ = 25 :=
sorry

end sector_max_area_l2029_202943


namespace floor_product_equals_45_l2029_202981

theorem floor_product_equals_45 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Ico 7.5 (46 / 6) :=
sorry

end floor_product_equals_45_l2029_202981


namespace student_distribution_l2029_202919

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with at least one object in each box -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 college students -/
def num_students : ℕ := 5

/-- There are 3 factories -/
def num_factories : ℕ := 3

/-- The theorem stating that there are 150 ways to distribute 5 students among 3 factories
    with at least one student in each factory -/
theorem student_distribution : distribute num_students num_factories = 150 := by sorry

end student_distribution_l2029_202919


namespace max_leap_years_in_period_l2029_202923

/-- A calendrical system where leap years occur every 5 years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period_length : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period_length / c.leap_year_interval

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end max_leap_years_in_period_l2029_202923


namespace frame_sales_ratio_l2029_202958

/-- Given:
  - Dorothy sells glass frames at half the price of Jemma
  - Jemma sells glass frames at 5 dollars each
  - Jemma sold 400 frames
  - They made 2500 dollars together in total
Prove that the ratio of frames Jemma sold to frames Dorothy sold is 2:1 -/
theorem frame_sales_ratio (jemma_price : ℚ) (jemma_sold : ℕ) (total_revenue : ℚ) 
    (h1 : jemma_price = 5)
    (h2 : jemma_sold = 400)
    (h3 : total_revenue = 2500) : 
  ∃ (dorothy_sold : ℕ), jemma_sold = 2 * dorothy_sold := by
  sorry

#check frame_sales_ratio

end frame_sales_ratio_l2029_202958


namespace divide_by_reciprocal_twelve_divided_by_one_twelfth_l2029_202920

theorem divide_by_reciprocal (a b : ℚ) (h : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end divide_by_reciprocal_twelve_divided_by_one_twelfth_l2029_202920


namespace exclusive_albums_count_l2029_202969

/-- The number of albums that are in either Andrew's or Bella's collection, but not both. -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (bella_unique : ℕ) : ℕ :=
  (andrew_total - shared) + bella_unique

/-- Theorem stating that the number of exclusive albums is 17 given the problem conditions. -/
theorem exclusive_albums_count :
  exclusive_albums 15 23 9 = 17 := by
  sorry

end exclusive_albums_count_l2029_202969


namespace number_problem_l2029_202975

theorem number_problem (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end number_problem_l2029_202975


namespace green_face_prob_five_eighths_l2029_202996

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  green_faces : ℕ
  purple_faces : ℕ
  total_faces : ℕ
  face_sum : green_faces + purple_faces = total_faces
  is_octahedron : total_faces = 8

/-- The probability of rolling a green face on a colored octahedron -/
def green_face_probability (o : ColoredOctahedron) : ℚ :=
  o.green_faces / o.total_faces

/-- Theorem: The probability of rolling a green face on a regular octahedron 
    with 5 green faces and 3 purple faces is 5/8 -/
theorem green_face_prob_five_eighths :
  ∀ (o : ColoredOctahedron), 
    o.green_faces = 5 → 
    o.purple_faces = 3 → 
    green_face_probability o = 5/8 :=
by
  sorry

end green_face_prob_five_eighths_l2029_202996


namespace x_axis_symmetry_y_axis_symmetry_l2029_202972

-- Define the region
def region (x y : ℝ) : Prop := abs (x + 2*y) + abs (2*x - y) ≤ 8

-- Theorem: The region is symmetric about the x-axis
theorem x_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region x (-y) :=
sorry

-- Theorem: The region is symmetric about the y-axis
theorem y_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region (-x) y :=
sorry

end x_axis_symmetry_y_axis_symmetry_l2029_202972


namespace diagonal_pigeonhole_l2029_202900

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths in a regular n-gon -/
def distinct_lengths (n : ℕ) : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n (n : ℕ) : ℕ := distinct_lengths n + 1

theorem diagonal_pigeonhole :
  smallest_n n = 1008 :=
sorry

end diagonal_pigeonhole_l2029_202900


namespace average_b_c_l2029_202946

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : a - c = 90) : 
  (b + c) / 2 = 70 := by
sorry

end average_b_c_l2029_202946


namespace second_divisor_l2029_202909

theorem second_divisor (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 7 = 3) 
  (d : ℕ) (h5 : d > 0) (h6 : k % d = 5) : d = 12 := by
  sorry

end second_divisor_l2029_202909


namespace soccer_lineup_combinations_l2029_202982

def total_players : ℕ := 16
def rookie_players : ℕ := 4
def goalkeeper_count : ℕ := 1
def defender_count : ℕ := 4
def midfielder_count : ℕ := 4
def forward_count : ℕ := 3

def lineup_combinations : ℕ := 
  total_players * 
  (Nat.choose (total_players - 1) defender_count) * 
  (Nat.choose (total_players - 1 - defender_count) midfielder_count) * 
  (Nat.choose rookie_players 2 * Nat.choose (total_players - rookie_players - goalkeeper_count - defender_count - midfielder_count) 1)

theorem soccer_lineup_combinations : 
  lineup_combinations = 21508800 := by sorry

end soccer_lineup_combinations_l2029_202982


namespace ellipse_semi_minor_axis_l2029_202916

/-- Given an ellipse with center at the origin, one focus at (0, -2), and one endpoint
    of a semi-major axis at (0, 5), its semi-minor axis has length √21. -/
theorem ellipse_semi_minor_axis (c a b : ℝ) : 
  c = 2 →  -- distance from center to focus
  a = 5 →  -- length of semi-major axis
  b^2 = a^2 - c^2 →  -- relationship between a, b, and c in an ellipse
  b = Real.sqrt 21 := by
sorry

end ellipse_semi_minor_axis_l2029_202916


namespace power_plus_mod_five_l2029_202956

theorem power_plus_mod_five : (2^2018 + 2019) % 5 = 3 := by
  sorry

end power_plus_mod_five_l2029_202956


namespace factorization_proof_l2029_202903

theorem factorization_proof (x y : ℝ) : 
  y^2 * (x - 2) + 16 * (2 - x) = (x - 2) * (y + 4) * (y - 4) := by
  sorry

end factorization_proof_l2029_202903


namespace integer_pair_property_l2029_202927

theorem integer_pair_property (a b : ℤ) :
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d ∣ a^n + b^n + 1)) ↔
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)) ∨
  ((a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2)) ∨
  ((a % 6 = 1 ∧ b % 6 = 4) ∨ (a % 6 = 4 ∧ b % 6 = 1)) :=
by sorry

end integer_pair_property_l2029_202927


namespace water_tank_capacity_l2029_202959

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) :
  (1 / 5 : ℝ) * c + 6 = (1 / 3 : ℝ) * c → c = 45 := by
  sorry

end water_tank_capacity_l2029_202959


namespace fourth_power_sum_l2029_202955

theorem fourth_power_sum (a b t : ℝ) 
  (h1 : a + b = t) 
  (h2 : a^2 + b^2 = t) 
  (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := by sorry

end fourth_power_sum_l2029_202955


namespace simplify_and_evaluate_l2029_202930

theorem simplify_and_evaluate : 
  let x : ℚ := -4
  let y : ℚ := 1/2
  (x + 2*y)^2 - x*(x + 3*y) - 4*y^2 = -2 := by
  sorry

end simplify_and_evaluate_l2029_202930


namespace circle_center_transformation_l2029_202992

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let S : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x S
  let final := translate_up reflected 10
  final = (-2, 4) := by sorry

end circle_center_transformation_l2029_202992


namespace intersection_P_complement_Q_l2029_202962

-- Define the sets P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}

-- Define the universal set U as the set of real numbers
def U : Type := ℝ

-- Theorem statement
theorem intersection_P_complement_Q :
  P ∩ (Set.univ \ Q) = {1, 2} := by
  sorry

end intersection_P_complement_Q_l2029_202962


namespace triangle_properties_l2029_202934

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A) : 
  t.A = Real.pi / 3 ∧ 
  (∃ (max : Real), max = Real.sqrt 3 ∧ 
    ∀ (x : Real), x = Real.sin t.B + Real.sin t.C → x ≤ max) ∧
  (t.A = t.B ∧ t.B = t.C) := by
  sorry

#check triangle_properties

end triangle_properties_l2029_202934


namespace painted_cube_theorem_l2029_202926

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) → n = 8 := by
  sorry

end painted_cube_theorem_l2029_202926


namespace microtron_stock_price_l2029_202921

/-- Represents the stock market scenario with Microtron and Dynaco stocks -/
structure StockMarket where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares_sold : ℕ
  average_price : ℝ
  dynaco_shares_sold : ℕ

/-- Theorem stating the price of Microtron stock given the market conditions -/
theorem microtron_stock_price (market : StockMarket) 
  (h1 : market.dynaco_price = 44)
  (h2 : market.total_shares_sold = 300)
  (h3 : market.average_price = 40)
  (h4 : market.dynaco_shares_sold = 150) :
  market.microtron_price = 36 := by
  sorry

end microtron_stock_price_l2029_202921


namespace percent_of_a_l2029_202914

theorem percent_of_a (a b c : ℝ) (h1 : c = 0.1 * b) (h2 : b = 2.5 * a) : c = 0.25 * a := by
  sorry

end percent_of_a_l2029_202914


namespace sum_of_cubes_l2029_202974

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85/2 := by
  sorry

end sum_of_cubes_l2029_202974
