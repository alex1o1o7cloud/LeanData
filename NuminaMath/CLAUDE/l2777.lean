import Mathlib

namespace NUMINAMATH_CALUDE_selling_price_ratio_l2777_277704

theorem selling_price_ratio (C : ℝ) (S1 S2 : ℝ) 
  (h1 : S1 = C + 0.60 * C) 
  (h2 : S2 = C + 3.20 * C) : 
  S2 / S1 = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l2777_277704


namespace NUMINAMATH_CALUDE_root_sum_squares_plus_product_l2777_277725

theorem root_sum_squares_plus_product (a b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + b^2 + a = 0) → 
  (x₂^2 + b*x₂ + b^2 + a = 0) → 
  x₁^2 + x₁*x₂ + x₂^2 + a = 0 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_plus_product_l2777_277725


namespace NUMINAMATH_CALUDE_circle_points_m_value_l2777_277709

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def onSameCircle (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (D E F : ℝ),
    p1.x^2 + p1.y^2 + D*p1.x + E*p1.y + F = 0 ∧
    p2.x^2 + p2.y^2 + D*p2.x + E*p2.y + F = 0 ∧
    p3.x^2 + p3.y^2 + D*p3.x + E*p3.y + F = 0 ∧
    p4.x^2 + p4.y^2 + D*p4.x + E*p4.y + F = 0

/-- Theorem: If (2,1), (4,2), (3,4), and (1,m) lie on the same circle, then m = 2 or m = 3 -/
theorem circle_points_m_value :
  ∀ (m : ℝ),
    onSameCircle
      (Point.mk 2 1)
      (Point.mk 4 2)
      (Point.mk 3 4)
      (Point.mk 1 m) →
    m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_m_value_l2777_277709


namespace NUMINAMATH_CALUDE_pencils_left_ashtons_pencils_l2777_277755

/-- Given two boxes of pencils with fourteen pencils each, after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given_away : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given_away

/-- Ashton's pencil problem -/
theorem ashtons_pencils : pencils_left 2 14 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_ashtons_pencils_l2777_277755


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2777_277779

theorem polynomial_factorization (a b : ℝ) : 
  a^2 - b^2 + 2*a + 1 = (a-b+1)*(a+b+1) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2777_277779


namespace NUMINAMATH_CALUDE_opera_house_earnings_correct_l2777_277796

/-- Calculates the earnings of an opera house for a single show. -/
def opera_house_earnings (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℕ) (percent_empty : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let occupied_seats := total_seats - (total_seats * percent_empty / 100)
  occupied_seats * ticket_price

/-- Theorem stating that the opera house earnings for the given conditions equal $12000. -/
theorem opera_house_earnings_correct : opera_house_earnings 150 10 10 20 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_opera_house_earnings_correct_l2777_277796


namespace NUMINAMATH_CALUDE_equation_solution_l2777_277715

theorem equation_solution (x : ℝ) : 
  x ≠ (1 / 3) → x ≠ -3 → 
  ((3 * x + 2) / (3 * x^2 + 8 * x - 3) = (3 * x) / (3 * x - 1)) ↔ 
  (x = -1 + Real.sqrt 15 / 3 ∨ x = -1 - Real.sqrt 15 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2777_277715


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2777_277791

theorem polynomial_factorization (y : ℝ) :
  1 + 5*y^2 + 25*y^4 + 125*y^6 + 625*y^8 = 
  (5*y^2 + ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 + ((5-Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5-Real.sqrt 5)*y)/2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2777_277791


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_17_l2777_277736

theorem half_abs_diff_squares_21_17 : (1/2 : ℚ) * |21^2 - 17^2| = 76 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_17_l2777_277736


namespace NUMINAMATH_CALUDE_parabola_vertex_vertex_of_specific_parabola_l2777_277789

/-- The vertex of a parabola y = ax^2 + bx + c is (h, k) where h = -b/(2a) and k = f(h) -/
theorem parabola_vertex (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let vertex_y : ℝ := f vertex_x
  (∀ x, f x ≥ vertex_y) ∨ (∀ x, f x ≤ vertex_y) :=
sorry

theorem vertex_of_specific_parabola :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 5
  let vertex : ℝ × ℝ := (1, 2)
  (∀ x, f x ≥ 2) ∧ f 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_vertex_of_specific_parabola_l2777_277789


namespace NUMINAMATH_CALUDE_no_polynomial_exists_l2777_277780

theorem no_polynomial_exists : ¬∃ (P : ℤ → ℤ) (a b c d : ℤ),
  (∀ n : ℕ, ∃ k : ℤ, P n = k) ∧  -- P has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4 := by
sorry

end NUMINAMATH_CALUDE_no_polynomial_exists_l2777_277780


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l2777_277792

/-- The number of handshakes in a gathering of couples -/
def num_handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands
    with everyone except their spouse and one other person, 
    the total number of handshakes is 54. -/
theorem six_couples_handshakes :
  num_handshakes 6 = 54 := by
  sorry

#eval num_handshakes 6  -- Should output 54

end NUMINAMATH_CALUDE_six_couples_handshakes_l2777_277792


namespace NUMINAMATH_CALUDE_max_value_fraction_l2777_277764

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 → 3 ≤ y' ∧ y' ≤ 6 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2777_277764


namespace NUMINAMATH_CALUDE_selectPeopleCount_l2777_277772

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select 4 people from 4 boys and 3 girls, 
    ensuring both boys and girls are included -/
def selectPeople : ℕ :=
  choose 4 3 * choose 3 1 + 
  choose 4 2 * choose 3 2 + 
  choose 4 1 * choose 3 3

theorem selectPeopleCount : selectPeople = 34 := by sorry

end NUMINAMATH_CALUDE_selectPeopleCount_l2777_277772


namespace NUMINAMATH_CALUDE_smallest_a1_l2777_277739

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 13 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ in the sequence. -/
theorem smallest_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
    ∀ a₁ : ℝ, a 1 ≥ a₁ → a₁ ≥ 25 / 72 :=
  sorry

end NUMINAMATH_CALUDE_smallest_a1_l2777_277739


namespace NUMINAMATH_CALUDE_sleeper_probability_l2777_277710

def total_delegates : ℕ := 9
def mexico_delegates : ℕ := 2
def canada_delegates : ℕ := 3
def us_delegates : ℕ := 4
def sleepers : ℕ := 3

theorem sleeper_probability :
  let total_outcomes := Nat.choose total_delegates sleepers
  let favorable_outcomes := 
    Nat.choose mexico_delegates 2 * Nat.choose canada_delegates 1 +
    Nat.choose mexico_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose canada_delegates 2 * Nat.choose us_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose mexico_delegates 1 +
    Nat.choose us_delegates 2 * Nat.choose canada_delegates 1
  (favorable_outcomes : ℚ) / total_outcomes = 55 / 84 := by
  sorry

end NUMINAMATH_CALUDE_sleeper_probability_l2777_277710


namespace NUMINAMATH_CALUDE_orange_calculation_l2777_277753

/-- Calculates the total number and weight of oranges given the number of children,
    oranges per child, and average weight per orange. -/
theorem orange_calculation (num_children : ℕ) (oranges_per_child : ℕ) (avg_weight : ℚ) :
  num_children = 4 →
  oranges_per_child = 3 →
  avg_weight = 3/10 →
  (num_children * oranges_per_child = 12 ∧
   (num_children * oranges_per_child : ℚ) * avg_weight = 18/5) :=
by sorry

end NUMINAMATH_CALUDE_orange_calculation_l2777_277753


namespace NUMINAMATH_CALUDE_sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2777_277729

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to calculate the exponent of a prime factor in n!
def primeExponentInFactorial (n : ℕ) (p : ℕ) : ℕ :=
  if p.Prime then
    (List.range (n + 1)).foldl (λ acc k => acc + k / p^k) 0
  else
    0

-- Define a function to get the largest even number not exceeding n
def largestEvenNotExceeding (n : ℕ) : ℕ :=
  if n % 2 = 0 then n else n - 1

-- Define the main theorem
theorem sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial :
  (let n := 15
   let primes := [2, 3, 5, 7]
   let exponents := primes.map (λ p => largestEvenNotExceeding (primeExponentInFactorial n p) / 2)
   exponents.sum) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_of_sqrt_largest_perfect_square_15_factorial_l2777_277729


namespace NUMINAMATH_CALUDE_percentage_of_160_to_50_l2777_277762

theorem percentage_of_160_to_50 : ∀ x : ℝ, (160 / 50) * 100 = x → x = 320 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_160_to_50_l2777_277762


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2777_277795

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - 4 * (1 + 2 + 3) / (5 + 10 + 15) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2777_277795


namespace NUMINAMATH_CALUDE_distinct_integers_sum_l2777_277782

theorem distinct_integers_sum (b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℕ) : 
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧ b₂ ≠ b₈ ∧ b₂ ≠ b₉ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧ b₃ ≠ b₈ ∧ b₃ ≠ b₉ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧ b₄ ≠ b₈ ∧ b₄ ≠ b₉ ∧
  b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧ b₅ ≠ b₈ ∧ b₅ ≠ b₉ ∧
  b₆ ≠ b₇ ∧ b₆ ≠ b₈ ∧ b₆ ≠ b₉ ∧
  b₇ ≠ b₈ ∧ b₇ ≠ b₉ ∧
  b₈ ≠ b₉ →
  (7 : ℚ) / 11 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 + b₉ / 362880 →
  0 ≤ b₂ ∧ b₂ < 2 →
  0 ≤ b₃ ∧ b₃ < 3 →
  0 ≤ b₄ ∧ b₄ < 4 →
  0 ≤ b₅ ∧ b₅ < 5 →
  0 ≤ b₆ ∧ b₆ < 6 →
  0 ≤ b₇ ∧ b₇ < 7 →
  0 ≤ b₈ ∧ b₈ < 8 →
  0 ≤ b₉ ∧ b₉ < 9 →
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ + b₉ = 16 := by
sorry

end NUMINAMATH_CALUDE_distinct_integers_sum_l2777_277782


namespace NUMINAMATH_CALUDE_water_leaked_l2777_277777

/-- Calculates the amount of water leaked from a bucket given the initial and remaining amounts. -/
theorem water_leaked (initial : ℚ) (remaining : ℚ) (h1 : initial = 0.75) (h2 : remaining = 0.5) :
  initial - remaining = 0.25 := by
  sorry

#check water_leaked

end NUMINAMATH_CALUDE_water_leaked_l2777_277777


namespace NUMINAMATH_CALUDE_mcgees_bakery_pies_l2777_277731

theorem mcgees_bakery_pies (smiths_pies mcgees_pies : ℕ) : 
  smiths_pies = 70 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  mcgees_pies = 16 := by
sorry

end NUMINAMATH_CALUDE_mcgees_bakery_pies_l2777_277731


namespace NUMINAMATH_CALUDE_fraction_value_l2777_277714

theorem fraction_value (a b : ℝ) (h : a / b = 3 / 5) : (2 * a + b) / (2 * a - b) = 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2777_277714


namespace NUMINAMATH_CALUDE_square_property_implies_equality_l2777_277799

theorem square_property_implies_equality (n : ℕ) (a : ℕ) (a_list : List ℕ) 
  (h : ∀ k : ℕ, ∃ m : ℕ, a * k + 1 = m ^ 2 → 
    ∃ (i : ℕ) (hi : i < a_list.length) (p : ℕ), a_list[i] * k + 1 = p ^ 2) :
  a ∈ a_list := by
  sorry

end NUMINAMATH_CALUDE_square_property_implies_equality_l2777_277799


namespace NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l2777_277744

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem least_number_divisible_by_first_five_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ first_five_primes, n % p = 0) → n ≥ 2310) ∧
  (∀ p ∈ first_five_primes, 2310 % p = 0) :=
sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l2777_277744


namespace NUMINAMATH_CALUDE_polynomial_equality_l2777_277787

def polynomial (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) (x : ℝ) : ℝ :=
  a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8

theorem polynomial_equality 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 3)^3 * (2*x + 1)^5 = polynomial a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ x) →
  (a₀ = -27 ∧ a₀ + a₂ + a₄ + a₆ + a₈ = -940) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2777_277787


namespace NUMINAMATH_CALUDE_square_sum_xy_l2777_277798

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : 1 / x^2 + 1 / y^2 = 7)
  (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l2777_277798


namespace NUMINAMATH_CALUDE_not_p_and_not_q_l2777_277773

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := (∀ a b : ℝ, a * b > 1 → (a > 1 ∧ b > 1)) ∧ 
                ¬(∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1)

-- Theorem to prove
theorem not_p_and_not_q : ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_l2777_277773


namespace NUMINAMATH_CALUDE_expression_value_l2777_277761

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2777_277761


namespace NUMINAMATH_CALUDE_no_upper_bound_for_a_l2777_277717

/-- The number of different representations of n as the sum of different divisors -/
def a (n : ℕ) : ℕ := sorry

/-- There is no upper bound M for a(n) that holds for all n -/
theorem no_upper_bound_for_a : ∀ M : ℕ, ∃ n : ℕ, a n > M := by sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_a_l2777_277717


namespace NUMINAMATH_CALUDE_smallest_satisfying_arrangement_l2777_277794

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_guests : ℕ

/-- Checks if a seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ (i : ℕ), i < seating.total_chairs →
    ∃ (j : ℕ), j < seating.seated_guests ∧
      (i % (seating.total_chairs / seating.seated_guests) = 0 ∨
       (i + 1) % (seating.total_chairs / seating.seated_guests) = 0)

/-- The main theorem to be proved -/
theorem smallest_satisfying_arrangement :
  ∀ (n : ℕ), n < 20 →
    ¬(satisfies_condition { total_chairs := 120, seated_guests := n }) ∧
    satisfies_condition { total_chairs := 120, seated_guests := 20 } :=
by sorry


end NUMINAMATH_CALUDE_smallest_satisfying_arrangement_l2777_277794


namespace NUMINAMATH_CALUDE_sum_reciprocals_l2777_277711

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l2777_277711


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2777_277740

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 12½π square inches,
    then the length of a chord of the larger circle tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) 
  (h2 : π * a^2 - π * b^2 = 25/2 * π) : 
  ∃ (c : ℝ), c^2 = 50 ∧ c = (2 * a^2 - 2 * b^2).sqrt := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2777_277740


namespace NUMINAMATH_CALUDE_hire_purchase_monthly_installment_l2777_277775

/-- Calculate the monthly installment for a hire-purchase car agreement -/
theorem hire_purchase_monthly_installment
  (cash_price : ℝ)
  (deposit_rate : ℝ)
  (num_installments : ℕ)
  (annual_interest_rate : ℝ)
  (h1 : cash_price = 22000)
  (h2 : deposit_rate = 0.1)
  (h3 : num_installments = 60)
  (h4 : annual_interest_rate = 0.12) :
  ∃ (monthly_installment : ℝ),
    monthly_installment = 528 ∧
    monthly_installment * num_installments =
      (cash_price - deposit_rate * cash_price) * (1 + annual_interest_rate * (num_installments / 12)) :=
by sorry

end NUMINAMATH_CALUDE_hire_purchase_monthly_installment_l2777_277775


namespace NUMINAMATH_CALUDE_percentage_to_number_l2777_277752

theorem percentage_to_number (x : ℝ) (h : x = 209) :
  x / 100 * 100 = 209 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_number_l2777_277752


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l2777_277716

theorem percentage_increase_problem (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  decrease_percent = 40 →
  final = 1080 →
  final = initial * (1 + increase_percent / 100) * (1 - decrease_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l2777_277716


namespace NUMINAMATH_CALUDE_tournament_rankings_l2777_277760

/-- Represents a team in the volleyball tournament -/
inductive Team : Type
| E | F | G | H | I | J

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  saturday_matches : Vector Match 3
  no_ties : Bool

/-- Calculates the number of possible ranking sequences -/
def possible_rankings (t : Tournament) : Nat :=
  6 * 6

/-- Theorem: The number of possible six-team ranking sequences is 36 -/
theorem tournament_rankings (t : Tournament) :
  t.no_ties → possible_rankings t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_rankings_l2777_277760


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2777_277737

theorem complex_modulus_problem (z : ℂ) : z * Complex.I ^ 2018 = 3 + 4 * Complex.I → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2777_277737


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2777_277747

theorem exponent_multiplication (x : ℝ) : x^2 * x * x^4 = x^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2777_277747


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2777_277767

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 54 →
  a * d + b * c = 105 →
  c * d = 50 →
  a^2 + b^2 + c^2 + d^2 ≤ 124 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2777_277767


namespace NUMINAMATH_CALUDE_fraction_simplification_l2777_277719

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hyd : y - 2/x ≠ 0) :
  (2*x - 3/y) / (3*y - 2/x) = (2*x*y - 3) / (3*x*y - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2777_277719


namespace NUMINAMATH_CALUDE_b_completion_time_l2777_277706

-- Define the work rates and time worked by A
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def a_time_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem b_completion_time :
  let work_done_by_a : ℚ := a_rate * a_time_worked
  let remaining_work : ℚ := total_work - work_done_by_a
  remaining_work / b_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l2777_277706


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_are_parallel_l2777_277774

/-- A line in 3D space -/
structure Line3D where
  -- You might want to add more properties here to fully define a line

/-- A plane in 3D space -/
structure Plane3D where
  -- You might want to add more properties here to fully define a plane

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (p1 p2 : Plane3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Main theorem: If m ⊥ α, n ⊥ β, α ∥ β, then m ∥ n -/
theorem perpendicular_to_parallel_planes_are_parallel 
  (m n : Line3D) (α β : Plane3D) :
  perpendicular_line_plane m α →
  perpendicular_line_plane n β →
  parallel_planes α β →
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_are_parallel_l2777_277774


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2777_277797

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2777_277797


namespace NUMINAMATH_CALUDE_parking_space_area_l2777_277781

theorem parking_space_area (l w : ℝ) (h1 : l = 9) (h2 : 2 * w + l = 37) : l * w = 126 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_area_l2777_277781


namespace NUMINAMATH_CALUDE_distance_between_projections_l2777_277734

/-- Given a point A(-1, 2, -3) in ℝ³, prove that the distance between its projection
    onto the yOz plane and its projection onto the x-axis is √14. -/
theorem distance_between_projections :
  let A : ℝ × ℝ × ℝ := (-1, 2, -3)
  let P₁ : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)  -- projection onto yOz plane
  let P₂ : ℝ × ℝ × ℝ := (A.1, 0, 0)        -- projection onto x-axis
  (P₁.1 - P₂.1)^2 + (P₁.2.1 - P₂.2.1)^2 + (P₁.2.2 - P₂.2.2)^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_projections_l2777_277734


namespace NUMINAMATH_CALUDE_angle_relation_l2777_277726

-- Define the structure for a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define a function to calculate the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if two triangles are anti-similar
def antiSimilar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to find the intersection of perpendicular bisectors
def perpendicularBisectorIntersection (A B C D : Point) : Point := sorry

-- Main theorem
theorem angle_relation 
  (A B C D X : Point)
  (Y : Point := perpendicularBisectorIntersection A B C D)
  (h1 : antiSimilar ⟨B, X, C⟩ ⟨A, X, D⟩)
  (h2 : isConvex ⟨A, B, C, D⟩)
  (h3 : angle A D X = angle B C X)
  (h4 : angle D A X = angle C B X)
  (h5 : angle A D X < π/2)
  (h6 : angle D A X < π/2)
  (h7 : angle B C X < π/2)
  (h8 : angle C B X < π/2) :
  angle A Y B = 2 * angle A D X := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l2777_277726


namespace NUMINAMATH_CALUDE_root_equation_solution_l2777_277765

theorem root_equation_solution (p q r : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/p) * (M^(1/q))^(1/p) * ((M^(1/r))^(1/q))^(1/p))^p = M^(15/24)) :
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l2777_277765


namespace NUMINAMATH_CALUDE_cake_and_bread_weight_l2777_277713

/-- Given the weight of 4 cakes and the weight difference between a cake and a piece of bread,
    calculate the total weight of 3 cakes and 5 pieces of bread. -/
theorem cake_and_bread_weight (cake_weight : ℕ) (bread_weight : ℕ) : 
  (4 * cake_weight = 800) →
  (cake_weight = bread_weight + 100) →
  (3 * cake_weight + 5 * bread_weight = 1100) :=
by sorry

end NUMINAMATH_CALUDE_cake_and_bread_weight_l2777_277713


namespace NUMINAMATH_CALUDE_four_liars_in_group_l2777_277702

/-- Represents a person who is either a knight or a liar -/
inductive Person
  | Knight
  | Liar

/-- Represents an answer to the question "How many liars are among you?" -/
def Answer := Fin 5

/-- A function that determines whether a person is telling the truth given their answer and the actual number of liars -/
def isTellingTruth (p : Person) (answer : Answer) (actualLiars : Nat) : Prop :=
  match p with
  | Person.Knight => answer.val + 1 = actualLiars
  | Person.Liar => answer.val + 1 ≠ actualLiars

/-- The main theorem -/
theorem four_liars_in_group (group : Fin 5 → Person) (answers : Fin 5 → Answer) 
    (h_distinct : ∀ i j, i ≠ j → answers i ≠ answers j) :
    (∃ (actualLiars : Nat), actualLiars = 4 ∧ 
      ∀ i, isTellingTruth (group i) (answers i) actualLiars) := by
  sorry

end NUMINAMATH_CALUDE_four_liars_in_group_l2777_277702


namespace NUMINAMATH_CALUDE_rectangles_and_triangles_on_4x3_grid_l2777_277703

/-- The number of rectangles on an m × n grid -/
def count_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

/-- The number of right-angled triangles (with right angles at grid points) on an m × n grid -/
def count_right_triangles (m n : ℕ) : ℕ := 2 * (m - 1) * (n - 1)

/-- The total number of rectangles and right-angled triangles on a 4×3 grid is 30 -/
theorem rectangles_and_triangles_on_4x3_grid :
  count_rectangles 4 3 + count_right_triangles 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_and_triangles_on_4x3_grid_l2777_277703


namespace NUMINAMATH_CALUDE_sum_of_squares_2005_squared_l2777_277790

theorem sum_of_squares_2005_squared :
  ∃ (a b c d e f g h : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
    2005^2 = a^2 + b^2 ∧
    2005^2 = c^2 + d^2 ∧
    2005^2 = e^2 + f^2 ∧
    2005^2 = g^2 + h^2 ∧
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (a, b) ≠ (g, h) ∧
    (c, d) ≠ (e, f) ∧ (c, d) ≠ (g, h) ∧
    (e, f) ≠ (g, h) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_2005_squared_l2777_277790


namespace NUMINAMATH_CALUDE_orange_percentage_l2777_277771

/-- Given a box of fruit with initial oranges and kiwis, and additional kiwis added,
    calculate the percentage of oranges in the final mixture. -/
theorem orange_percentage
  (initial_oranges : ℕ)
  (initial_kiwis : ℕ)
  (added_kiwis : ℕ)
  (h1 : initial_oranges = 24)
  (h2 : initial_kiwis = 30)
  (h3 : added_kiwis = 26) :
  (initial_oranges : ℚ) / (initial_oranges + initial_kiwis + added_kiwis) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_orange_percentage_l2777_277771


namespace NUMINAMATH_CALUDE_group_size_problem_l2777_277786

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2777_277786


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2777_277756

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter n is √6 -/
theorem ellipse_hyperbola_same_foci (n : ℝ) :
  n > 0 →
  (∀ x y : ℝ, x^2 / 16 + y^2 / n^2 = 1 ↔ x^2 / n^2 - y^2 / 4 = 1) →
  n = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2777_277756


namespace NUMINAMATH_CALUDE_geometric_sum_example_l2777_277748

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- Proof of the sum of the first eight terms of a specific geometric sequence -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l2777_277748


namespace NUMINAMATH_CALUDE_greatest_number_l2777_277743

theorem greatest_number (A B C : ℤ) : 
  A = 95 - 35 →
  B = A + 12 →
  C = B - 19 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  B > A ∧ B > C :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l2777_277743


namespace NUMINAMATH_CALUDE_smallest_equal_flock_size_l2777_277785

theorem smallest_equal_flock_size (duck_flock_size crane_flock_size : ℕ) 
  (duck_flock_size_pos : duck_flock_size > 0)
  (crane_flock_size_pos : crane_flock_size > 0)
  (duck_flock_size_eq : duck_flock_size = 13)
  (crane_flock_size_eq : crane_flock_size = 17) :
  ∃ n : ℕ, n > 0 ∧ 
    n % duck_flock_size = 0 ∧ 
    n % crane_flock_size = 0 ∧
    (∀ m : ℕ, m > 0 ∧ m % duck_flock_size = 0 ∧ m % crane_flock_size = 0 → m ≥ n) ∧
    n = 221 :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_flock_size_l2777_277785


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l2777_277708

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ+, (k : ℝ) / (3 : ℝ) ^ (k : ℕ)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l2777_277708


namespace NUMINAMATH_CALUDE_valid_arrangements_l2777_277759

/-- Represents the number of plates of each color -/
structure PlateCount where
  yellow : Nat
  blue : Nat
  red : Nat
  purple : Nat

/-- Calculates the total number of plates -/
def totalPlates (count : PlateCount) : Nat :=
  count.yellow + count.blue + count.red + count.purple

/-- Calculates the number of circular arrangements -/
def circularArrangements (count : PlateCount) : Nat :=
  sorry

/-- Calculates the number of circular arrangements with red plates adjacent -/
def redAdjacentArrangements (count : PlateCount) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements (count : PlateCount) 
  (h1 : count.yellow = 4)
  (h2 : count.blue = 3)
  (h3 : count.red = 2)
  (h4 : count.purple = 1) :
  circularArrangements count - redAdjacentArrangements count = 980 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2777_277759


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l2777_277768

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(2*x+y)/z⌋ + ⌊(2*y+z)/x⌋ + ⌊(2*z+x)/y⌋ ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l2777_277768


namespace NUMINAMATH_CALUDE_money_division_l2777_277718

theorem money_division (a b c : ℚ) 
  (h1 : a = (2/3) * (b + c))
  (h2 : b = (6/9) * (a + c))
  (h3 : a = 280) : 
  a + b + c = 700 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2777_277718


namespace NUMINAMATH_CALUDE_fraction_equality_l2777_277732

theorem fraction_equality : (1/3 - 1/4) / (1/2 - 1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2777_277732


namespace NUMINAMATH_CALUDE_compute_fraction_power_l2777_277746

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l2777_277746


namespace NUMINAMATH_CALUDE_fibonacci_determinant_identity_l2777_277707

def fibonacci : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![fibonacci (n + 1), fibonacci n; fibonacci n, fibonacci (n - 1)]

theorem fibonacci_determinant_identity (n : ℕ) :
  fibonacci (n + 1) * fibonacci (n - 1) - fibonacci n ^ 2 = (-1) ^ n :=
sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_identity_l2777_277707


namespace NUMINAMATH_CALUDE_min_distance_point_is_circumcenter_l2777_277722

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the foot of the perpendicular from a point to a line segment -/
def perpendicularFoot (p : Point) (a b : Point) : Point :=
  sorry

/-- Calculates the circumcenter of a triangle -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Main theorem: The point that minimizes the sum of squared distances to the sides
    of an acute triangle is its circumcenter -/
theorem min_distance_point_is_circumcenter (t : Triangle) (h : isAcute t) :
  ∀ P : Point,
    let L := perpendicularFoot P t.B t.C
    let M := perpendicularFoot P t.C t.A
    let N := perpendicularFoot P t.A t.B
    squaredDistance P L + squaredDistance P M + squaredDistance P N ≥
    let C := circumcenter t
    let CL := perpendicularFoot C t.B t.C
    let CM := perpendicularFoot C t.C t.A
    let CN := perpendicularFoot C t.A t.B
    squaredDistance C CL + squaredDistance C CM + squaredDistance C CN :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_is_circumcenter_l2777_277722


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2777_277776

theorem divisibility_theorem (n : ℕ) (x : ℤ) (h : n ≥ 1) :
  ∃ k : ℤ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = k * (x-1)^3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2777_277776


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l2777_277735

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l2777_277735


namespace NUMINAMATH_CALUDE_matrix_equality_l2777_277783

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![20/3, 4/3], ![-8/3, 8/3]]) : 
  B * A = ![![20/3, 4/3], ![-8/3, 8/3]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l2777_277783


namespace NUMINAMATH_CALUDE_function_zeros_l2777_277766

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def count_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

theorem function_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f (2 * Real.pi))
  (h_zero_3 : f 3 = 0)
  (h_zero_4 : f 4 = 0) :
  count_zeros f 0 10 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_function_zeros_l2777_277766


namespace NUMINAMATH_CALUDE_lattice_polygon_extension_l2777_277738

/-- A point with integer coordinates -/
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A polygon with all vertices being lattice points -/
def LatticePolygon (vertices : List (ℝ × ℝ)) : Prop :=
  ∀ v ∈ vertices, LatticePoint v

/-- A convex polygon -/
def ConvexPolygon (vertices : List (ℝ × ℝ)) : Prop :=
  sorry  -- Definition of convex polygon

/-- Theorem: For any convex lattice polygon, there exists another convex lattice polygon
    that contains it and has exactly one additional vertex -/
theorem lattice_polygon_extension
  (Γ : List (ℝ × ℝ))
  (h_lattice : LatticePolygon Γ)
  (h_convex : ConvexPolygon Γ) :
  ∃ (Γ' : List (ℝ × ℝ)),
    LatticePolygon Γ' ∧
    ConvexPolygon Γ' ∧
    (∀ v ∈ Γ, v ∈ Γ') ∧
    (∃! v, v ∈ Γ' ∧ v ∉ Γ) :=
  sorry

end NUMINAMATH_CALUDE_lattice_polygon_extension_l2777_277738


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2777_277724

-- Define the parameters of the problem
def initial_tagged : ℕ := 30
def second_catch : ℕ := 50
def total_fish : ℕ := 750

-- Define the theorem
theorem tagged_fish_in_second_catch :
  ∃ (T : ℕ), (T : ℚ) / second_catch = initial_tagged / total_fish ∧ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2777_277724


namespace NUMINAMATH_CALUDE_tangent_intersection_theorem_l2777_277793

/-- The x-coordinate of the point where a line tangent to two circles intersects the x-axis -/
def tangent_intersection_x : ℝ := 4.5

/-- The radius of the first circle -/
def r1 : ℝ := 3

/-- The radius of the second circle -/
def r2 : ℝ := 5

/-- The x-coordinate of the center of the second circle -/
def c2_x : ℝ := 12

theorem tangent_intersection_theorem :
  let x := tangent_intersection_x
  x > 0 ∧ 
  x / (c2_x - x) = r1 / r2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_theorem_l2777_277793


namespace NUMINAMATH_CALUDE_veridux_female_employees_l2777_277788

/-- Proves that the number of female employees at Veridux Corporation is 90 -/
theorem veridux_female_employees :
  let total_employees : ℕ := 250
  let total_managers : ℕ := 40
  let male_associates : ℕ := 160
  let female_managers : ℕ := 40
  let total_associates : ℕ := total_employees - total_managers
  let female_associates : ℕ := total_associates - male_associates
  let female_employees : ℕ := female_managers + female_associates
  female_employees = 90 := by
  sorry


end NUMINAMATH_CALUDE_veridux_female_employees_l2777_277788


namespace NUMINAMATH_CALUDE_raisin_cookies_sold_l2777_277730

theorem raisin_cookies_sold (raisin oatmeal : ℕ) : 
  (raisin : ℚ) / oatmeal = 6 / 1 →
  raisin + oatmeal = 49 →
  raisin = 42 := by
sorry

end NUMINAMATH_CALUDE_raisin_cookies_sold_l2777_277730


namespace NUMINAMATH_CALUDE_cos_48_degrees_l2777_277712

theorem cos_48_degrees : Real.cos (48 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l2777_277712


namespace NUMINAMATH_CALUDE_bike_sharing_selection_l2777_277770

theorem bike_sharing_selection (yellow_bikes : ℕ) (blue_bikes : ℕ) (inspect_yellow : ℕ) (inspect_blue : ℕ) :
  yellow_bikes = 6 →
  blue_bikes = 4 →
  inspect_yellow = 4 →
  inspect_blue = 4 →
  (Nat.choose blue_bikes 2 * Nat.choose yellow_bikes 2 +
   Nat.choose blue_bikes 3 * Nat.choose yellow_bikes 1 +
   Nat.choose blue_bikes 4) = 115 :=
by sorry

end NUMINAMATH_CALUDE_bike_sharing_selection_l2777_277770


namespace NUMINAMATH_CALUDE_triangle_properties_l2777_277727

theorem triangle_properties (x : ℝ) (h : x > 0) :
  let a := 5*x
  let b := 12*x
  let c := 13*x
  (a^2 + b^2 = c^2) ∧ (∃ q : ℚ, (a / b : ℝ) = q) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2777_277727


namespace NUMINAMATH_CALUDE_interest_difference_l2777_277757

/-- Calculates the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  let compound_interest := principal * (1 + rate)^time - principal
  let simple_interest := principal * rate * time
  principal = 6500 ∧ rate = 0.04 ∧ time = 2 →
  compound_interest - simple_interest = 9.40 := by sorry

end NUMINAMATH_CALUDE_interest_difference_l2777_277757


namespace NUMINAMATH_CALUDE_unique_expression_value_l2777_277750

theorem unique_expression_value (m n : ℤ) : 
  (∃! z : ℤ, m * n + 13 * m + 13 * n - m^2 - n^2 = z ∧ 
   (∀ k l : ℤ, k * l + 13 * k + 13 * l - k^2 - l^2 = z → k = m ∧ l = n)) →
  m * n + 13 * m + 13 * n - m^2 - n^2 = 169 :=
by sorry

end NUMINAMATH_CALUDE_unique_expression_value_l2777_277750


namespace NUMINAMATH_CALUDE_slices_left_for_era_l2777_277745

/-- The number of burgers Era had -/
def num_burgers : ℕ := 5

/-- The number of friends Era has -/
def num_friends : ℕ := 4

/-- The number of slices each burger is cut into -/
def slices_per_burger : ℕ := 2

/-- The number of slices given to the first friend -/
def slices_friend1 : ℕ := 1

/-- The number of slices given to the second friend -/
def slices_friend2 : ℕ := 2

/-- The number of slices given to the third friend -/
def slices_friend3 : ℕ := 3

/-- The number of slices given to the fourth friend -/
def slices_friend4 : ℕ := 3

/-- Theorem stating that the number of slices left for Era is 1 -/
theorem slices_left_for_era :
  num_burgers * slices_per_burger - (slices_friend1 + slices_friend2 + slices_friend3 + slices_friend4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_slices_left_for_era_l2777_277745


namespace NUMINAMATH_CALUDE_cubic_difference_l2777_277700

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2777_277700


namespace NUMINAMATH_CALUDE_mixed_nuts_cost_l2777_277742

/-- Represents the price and amount of a type of nut -/
structure NutInfo where
  price : ℚ  -- Price in dollars
  amount : ℚ  -- Amount in ounces
  deriving Repr

/-- Calculates the discounted price per ounce -/
def discountedPricePerOz (info : NutInfo) (discount : ℚ) : ℚ :=
  (info.price / info.amount) * (1 - discount)

/-- Calculates the cost of a nut in the mix -/
def nutCostInMix (pricePerOz : ℚ) (proportion : ℚ) : ℚ :=
  pricePerOz * proportion

/-- The main theorem stating the minimum cost of the mixed nuts -/
theorem mixed_nuts_cost
  (almond_info : NutInfo)
  (cashew_info : NutInfo)
  (walnut_info : NutInfo)
  (almond_discount cashew_discount walnut_discount : ℚ)
  (h_almond_price : almond_info.price = 18)
  (h_almond_amount : almond_info.amount = 32)
  (h_cashew_price : cashew_info.price = 45/2)
  (h_cashew_amount : cashew_info.amount = 28)
  (h_walnut_price : walnut_info.price = 15)
  (h_walnut_amount : walnut_info.amount = 24)
  (h_almond_discount : almond_discount = 1/10)
  (h_cashew_discount : cashew_discount = 3/20)
  (h_walnut_discount : walnut_discount = 1/5)
  : ∃ (cost : ℕ), cost = 56 ∧ 
    cost * (1/100 : ℚ) ≥ 
      nutCostInMix (discountedPricePerOz almond_info almond_discount) (1/2) +
      nutCostInMix (discountedPricePerOz cashew_info cashew_discount) (3/10) +
      nutCostInMix (discountedPricePerOz walnut_info walnut_discount) (1/5) :=
sorry

end NUMINAMATH_CALUDE_mixed_nuts_cost_l2777_277742


namespace NUMINAMATH_CALUDE_train_crossing_time_l2777_277733

/-- Given a train and platform with specific dimensions and crossing time, 
    calculate the time it takes for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 450)
  (h3 : platform_crossing_time = 45)
  : ∃ (signal_pole_time : ℝ), 
    (signal_pole_time ≥ 17.9 ∧ signal_pole_time ≤ 18.1) := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2777_277733


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l2777_277769

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def T_size : ℕ := 15

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17/35

theorem hexagon_segment_probability :
  (num_sides / T_size) * ((num_sides - 1) / (T_size - 1)) +
  (num_diagonals / T_size) * ((num_diagonals - 1) / (T_size - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l2777_277769


namespace NUMINAMATH_CALUDE_abc_log_sum_l2777_277741

theorem abc_log_sum (A B C : ℕ+) (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) :
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_log_sum_l2777_277741


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2777_277720

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  a = 2 * Real.sqrt 3 →
  c = 2 * Real.sqrt 2 →
  A = π / 3 →
  (a / Real.sin A = c / Real.sin C) →
  C < π / 2 →
  C = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2777_277720


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l2777_277723

def A (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; a, b]

theorem matrix_is_own_inverse (a b : ℚ) :
  A a b * A a b = 1 ↔ a = 15/2 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l2777_277723


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2777_277749

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + a * c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2777_277749


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l2777_277758

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let numAxles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll numAxles = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l2777_277758


namespace NUMINAMATH_CALUDE_equation_solution_l2777_277728

theorem equation_solution :
  ∀ x : ℝ, (2*x - 3)^2 = (x - 2)^2 ↔ x = 1 ∨ x = 5/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2777_277728


namespace NUMINAMATH_CALUDE_number_difference_l2777_277763

theorem number_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) : 
  |x - y| = 7 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2777_277763


namespace NUMINAMATH_CALUDE_days_in_year_l2777_277751

theorem days_in_year (a b c : ℕ+) (h : 29 * a + 30 * b + 31 * c = 366) : 
  19 * a + 20 * b + 21 * c = 246 := by
  sorry

end NUMINAMATH_CALUDE_days_in_year_l2777_277751


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2777_277721

theorem angle_measure_proof (x : ℝ) : 
  (x = 21) → 
  (90 - x = 3 * x + 6) ∧
  (x + (90 - x) = 90) :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2777_277721


namespace NUMINAMATH_CALUDE_boat_distance_proof_l2777_277784

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed stream_speed time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 63 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l2777_277784


namespace NUMINAMATH_CALUDE_train_length_l2777_277701

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 100 → time_s = 9 → 
  ∃ length_m : ℝ, abs (length_m - 250.02) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2777_277701


namespace NUMINAMATH_CALUDE_root_implies_inequality_l2777_277778

theorem root_implies_inequality (a b : ℝ) 
  (h : ∃ x, (x + a) * (x + b) = 9 ∧ x = a + b) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_inequality_l2777_277778


namespace NUMINAMATH_CALUDE_abs_sum_diff_inequality_l2777_277705

theorem abs_sum_diff_inequality (x y : ℝ) :
  (abs x < 1 ∧ abs y < 1) ↔ abs (x + y) + abs (x - y) < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_diff_inequality_l2777_277705


namespace NUMINAMATH_CALUDE_factory_production_l2777_277754

/-- The number of computers produced per day by a factory -/
def computers_per_day : ℕ := 1500

/-- The selling price of each computer in dollars -/
def price_per_computer : ℕ := 150

/-- The revenue from one week's production in dollars -/
def weekly_revenue : ℕ := 1575000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem factory_production :
  computers_per_day * price_per_computer * days_in_week = weekly_revenue :=
by sorry

end NUMINAMATH_CALUDE_factory_production_l2777_277754
