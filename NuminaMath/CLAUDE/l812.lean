import Mathlib

namespace seating_arrangements_seven_people_l812_81203

/-- The number of ways to arrange n people around a circular table, considering rotations as identical -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n blocks around a circular table, with one block containing 3 fixed people -/
def arrangementsWithFixedBlock (n : ℕ) : ℕ := 
  circularArrangements (n - 2) * 2

theorem seating_arrangements_seven_people : 
  arrangementsWithFixedBlock 7 = 240 := by
  sorry

end seating_arrangements_seven_people_l812_81203


namespace inequality_solution_set_l812_81289

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - 4| ≥ a) ↔ a ≤ 1 :=
sorry

end inequality_solution_set_l812_81289


namespace distance_proof_l812_81201

/-- The distance between points A and B in kilometers -/
def distance : ℝ := 180

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- The velocity of the stream in km/h -/
def stream_velocity : ℝ := 4

/-- The speed of the boat in still water in km/h -/
def boat_speed : ℝ := 14

/-- The downstream speed of the boat in km/h -/
def downstream_speed : ℝ := boat_speed + stream_velocity

/-- The upstream speed of the boat in km/h -/
def upstream_speed : ℝ := boat_speed - stream_velocity

theorem distance_proof :
  distance / downstream_speed + (distance / 2) / upstream_speed = total_time :=
sorry

end distance_proof_l812_81201


namespace isosceles_hyperbola_l812_81224

/-- 
Given that C ≠ 0 and A and B do not vanish simultaneously,
the equation A x(x^2 - y^2) - (A^2 - B^2) x y = C represents
an isosceles hyperbola with asymptotes A x + B y = 0 and B x - A y = 0
-/
theorem isosceles_hyperbola (A B C : ℝ) (h1 : C ≠ 0) (h2 : ¬(A = 0 ∧ B = 0)) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, A * (x t) * ((x t)^2 - (y t)^2) - (A^2 - B^2) * (x t) * (y t) = C) ∧ 
    (∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ 
      A * (x t1) + B * (y t1) = 0 ∧ 
      B * (x t2) - A * (y t2) = 0) :=
by sorry

end isosceles_hyperbola_l812_81224


namespace pushup_sequence_sum_l812_81213

theorem pushup_sequence_sum (a : ℕ → ℕ) :
  (a 0 = 10) →
  (∀ n : ℕ, a (n + 1) = a n + 5) →
  (a 0 + a 1 + a 2 = 45) := by
  sorry

end pushup_sequence_sum_l812_81213


namespace two_numbers_difference_l812_81228

theorem two_numbers_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 30) (h3 : x * y = 200) :
  x - y = 10 :=
by sorry

end two_numbers_difference_l812_81228


namespace range_of_a_l812_81243

-- Define the condition that x > 2 is sufficient but not necessary for x^2 > a
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, x > 2 → x^2 > a) ∧ 
  ¬(∀ x : ℝ, x^2 > a → x > 2)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a ↔ a ≤ 4 :=
sorry

end range_of_a_l812_81243


namespace percentage_difference_l812_81295

theorem percentage_difference (x : ℝ) : 
  (60 / 100 * 50 = 30) →
  (30 = x / 100 * 30 + 17.4) →
  x = 42 := by
sorry

end percentage_difference_l812_81295


namespace complex_magnitude_problem_l812_81262

theorem complex_magnitude_problem (z : ℂ) (h : (Complex.I / (1 + Complex.I)) * z = 1) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l812_81262


namespace squares_below_line_l812_81246

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line := { a := 5, b := 152, c := 1520 }

/-- The theorem to be proved --/
theorem squares_below_line : countSquaresBelowLine problemLine = 1363 := by
  sorry

end squares_below_line_l812_81246


namespace circle_passes_through_origin_l812_81233

/-- A circle is defined by its center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- A point is defined by its coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point (x, y) is on a circle if and only if (x-a)^2 + (y-b)^2 = r^2 -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.a)^2 + (p.y - c.b)^2 = c.r^2

/-- Theorem: A circle passes through the origin if and only if a^2 + b^2 = r^2 -/
theorem circle_passes_through_origin (c : Circle) :
  isOnCircle origin c ↔ c.a^2 + c.b^2 = c.r^2 := by
  sorry

end circle_passes_through_origin_l812_81233


namespace triangle_area_from_squares_l812_81219

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end triangle_area_from_squares_l812_81219


namespace solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l812_81268

-- System of linear equations 1
theorem solve_system_1 (x y : ℝ) : 
  x = 7 * y ∧ 2 * x + y = 30 → x = 14 ∧ y = 2 := by sorry

-- System of linear equations 2
theorem solve_system_2 (x y : ℝ) : 
  x / 2 + y / 3 = 7 ∧ x / 3 - y / 4 = -1 → x = 6 ∧ y = 12 := by sorry

-- Linear inequality
theorem solve_inequality (x : ℝ) :
  4 + 3 * (x - 1) > -5 ↔ x > -2 := by sorry

-- System of linear inequalities
theorem solve_inequality_system (x : ℝ) :
  (1 / 2 * (x - 2) + 3 > 7 ∧ -1 / 3 * (x + 3) - 4 > -10) ↔ (x > 10 ∧ x < 15) := by sorry

end solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l812_81268


namespace medium_supermarkets_sample_l812_81269

/-- Represents the number of supermarkets to be sampled in a stratified sampling method. -/
def stratified_sample (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) : ℕ :=
  let total := total_large + total_medium + total_small
  (sample_size * total_medium) / total

/-- Theorem stating that the number of medium-sized supermarkets to be sampled is 20. -/
theorem medium_supermarkets_sample :
  stratified_sample 200 400 1400 100 = 20 := by
  sorry

end medium_supermarkets_sample_l812_81269


namespace smallest_prime_after_four_nonprimes_l812_81277

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all nonprime -/
def fourConsecutiveNonPrime (n : ℕ) : Prop :=
  ¬(isPrime n) ∧ ¬(isPrime (n + 1)) ∧ ¬(isPrime (n + 2)) ∧ ¬(isPrime (n + 3))

/-- The theorem stating that 29 is the smallest prime after four consecutive nonprimes -/
theorem smallest_prime_after_four_nonprimes :
  ∃ n : ℕ, fourConsecutiveNonPrime n ∧ isPrime (n + 4) ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutiveNonPrime m ∧ isPrime (m + 4)) :=
sorry

end smallest_prime_after_four_nonprimes_l812_81277


namespace geoff_total_spending_l812_81296

/-- Geoff's spending on sneakers over three days -/
def sneaker_spending (monday_spend : ℝ) : ℝ :=
  let tuesday_spend := 4 * monday_spend
  let wednesday_spend := 5 * monday_spend
  monday_spend + tuesday_spend + wednesday_spend

/-- Theorem stating that Geoff's total spending over three days is $600 -/
theorem geoff_total_spending :
  sneaker_spending 60 = 600 := by
  sorry

end geoff_total_spending_l812_81296


namespace cheerleader_group_composition_cheerleader_group_composition_result_l812_81298

theorem cheerleader_group_composition 
  (total_females : Nat) 
  (males_chose_malt : Nat) 
  (females_chose_malt : Nat) : Nat :=
  let total_malt := males_chose_malt + females_chose_malt
  let total_coke := total_malt / 2
  let total_cheerleaders := total_malt + total_coke
  let total_males := total_cheerleaders - total_females
  
  have h1 : total_females = 16 := by sorry
  have h2 : males_chose_malt = 6 := by sorry
  have h3 : females_chose_malt = 8 := by sorry
  
  total_males

theorem cheerleader_group_composition_result : 
  cheerleader_group_composition 16 6 8 = 5 := by sorry

end cheerleader_group_composition_cheerleader_group_composition_result_l812_81298


namespace reciprocal_counterexample_l812_81229

theorem reciprocal_counterexample : ∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end reciprocal_counterexample_l812_81229


namespace largest_three_digit_special_divisibility_l812_81260

theorem largest_three_digit_special_divisibility : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n % 11 = 0) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) → 
    (∀ d : ℕ, d ∈ m.digits 10 → d ≠ 0 → m % d = 0) →
    (m % 11 = 0) → m ≤ n) ∧
  n = 924 :=
by sorry

end largest_three_digit_special_divisibility_l812_81260


namespace min_value_theorem_l812_81206

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (2 * x^2 - x + 1) / (x * y) ≥ 2 * Real.sqrt 2 + 1 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧
    (2 * x₀^2 - x₀ + 1) / (x₀ * y₀) = 2 * Real.sqrt 2 + 1 :=
by sorry

end min_value_theorem_l812_81206


namespace function_equation_solution_l812_81299

theorem function_equation_solution (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = 2^x + (1/2)^x := by
  sorry

end function_equation_solution_l812_81299


namespace d_t_eventually_two_exists_n_d_t_two_from_m_l812_81218

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The t-th iteration of d applied to n -/
def d_t (t n : ℕ) : ℕ :=
  match t with
  | 0 => n
  | t + 1 => d (d_t t n)

/-- For any n > 1, the sequence d_t(n) eventually becomes 2 -/
theorem d_t_eventually_two (n : ℕ) (h : n > 1) :
  ∃ k, ∀ t, t ≥ k → d_t t n = 2 := by sorry

/-- For any m, there exists an n such that d_t(n) becomes 2 from the m-th term onwards -/
theorem exists_n_d_t_two_from_m (m : ℕ) :
  ∃ n, ∀ t, t ≥ m → d_t t n = 2 := by sorry

end d_t_eventually_two_exists_n_d_t_two_from_m_l812_81218


namespace product_of_odd_primes_below_16_mod_32_l812_81222

def odd_primes_below_16 : List Nat := [3, 5, 7, 11, 13]

theorem product_of_odd_primes_below_16_mod_32 :
  (List.prod odd_primes_below_16) % (2^5) = 7 := by
  sorry

end product_of_odd_primes_below_16_mod_32_l812_81222


namespace product_of_fractions_and_root_l812_81200

theorem product_of_fractions_and_root : 
  (2 : ℝ) / 3 * (3 : ℝ) / 5 * ((4 : ℝ) / 7) ^ (1 / 2) = 4 * Real.sqrt 7 / 35 := by
  sorry

end product_of_fractions_and_root_l812_81200


namespace unique_solution_cubic_system_l812_81278

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ),
    x = y^3 + y - 8 ∧
    y = z^3 + z - 8 ∧
    z = x^3 + x - 8 ∧
    x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end unique_solution_cubic_system_l812_81278


namespace machine_year_production_l812_81256

/-- A machine that produces items at a constant rate. -/
structure Machine where
  production_rate : ℕ  -- Items produced per hour

/-- Represents a year with a fixed number of days. -/
structure Year where
  days : ℕ

/-- Calculates the total number of units produced by a machine in a year. -/
def units_produced (m : Machine) (y : Year) : ℕ :=
  m.production_rate * y.days * 24

/-- Theorem stating that a machine producing one item per hour will make 8760 units in a year of 365 days. -/
theorem machine_year_production :
  ∀ (m : Machine) (y : Year),
    m.production_rate = 1 →
    y.days = 365 →
    units_produced m y = 8760 :=
by
  sorry


end machine_year_production_l812_81256


namespace waysToChooseIsCorrect_l812_81237

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people -/
def waysToChoose : ℕ :=
  let totalPeople : ℕ := 10
  let committeeSize : ℕ := 3
  totalPeople * Nat.choose (totalPeople - 1) committeeSize

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people is 840 -/
theorem waysToChooseIsCorrect : waysToChoose = 840 := by
  sorry

end waysToChooseIsCorrect_l812_81237


namespace tangency_quad_area_theorem_l812_81275

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The area of the quadrilateral formed by tangency points -/
  tangencyQuadArea : ℝ
  /-- Assumption that the trapezoid is circumscribed around the circle -/
  isCircumscribed : Prop
  /-- Assumption that the trapezoid is isosceles -/
  isIsosceles : Prop

/-- Theorem stating the relationship between the areas -/
theorem tangency_quad_area_theorem (t : CircumscribedTrapezoid)
  (h1 : t.radius = 1)
  (h2 : t.trapezoidArea = 5)
  : t.tangencyQuadArea = 1.6 := by
  sorry

end tangency_quad_area_theorem_l812_81275


namespace solution_inequality1_solution_inequality2_l812_81297

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def solution_set2 : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := |1 - (2*x - 1)/3| ≤ 2
def inequality2 (x : ℝ) : Prop := (2 - x)*(x + 3) < 2 - x

-- Theorem statements
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = solution_set2 := by sorry

end solution_inequality1_solution_inequality2_l812_81297


namespace expansion_properties_l812_81261

def binomial_sum (n : ℕ) : ℕ := 2^n

def constant_term (n : ℕ) : ℕ := Nat.choose n (n / 2)

theorem expansion_properties :
  ∃ (n : ℕ), 
    binomial_sum n = 64 ∧ 
    constant_term n = 15 := by
  sorry

end expansion_properties_l812_81261


namespace all_statements_false_l812_81207

theorem all_statements_false :
  (¬ (∀ x : ℝ, x^(1/3) = x → x = 0 ∨ x = 1)) ∧
  (¬ (∀ a : ℝ, Real.sqrt (a^2) = a)) ∧
  (¬ ((-8 : ℝ)^(1/3) = 2 ∨ (-8 : ℝ)^(1/3) = -2)) ∧
  (¬ (Real.sqrt (Real.sqrt 81) = 9)) :=
sorry

end all_statements_false_l812_81207


namespace spending_limit_ratio_l812_81276

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sally_cards_conditions (cards : SallysCards) : Prop :=
  cards.gold.balance = (1/3) * cards.gold.limit ∧
  cards.platinum.balance = (1/4) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (5/12) * cards.platinum.limit

/-- The theorem stating the ratio of spending limits -/
theorem spending_limit_ratio (cards : SallysCards) 
  (h : sally_cards_conditions cards) : 
  cards.platinum.limit = (1/2) * cards.gold.limit := by
  sorry

#check spending_limit_ratio

end spending_limit_ratio_l812_81276


namespace max_sqrt_sum_l812_81258

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ d : ℝ, d = 6 ∧ ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end max_sqrt_sum_l812_81258


namespace second_smallest_divisible_by_all_less_than_9_sum_of_digits_l812_81227

def is_divisible_by_all_less_than_9 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 9 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_9_sum_of_digits :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_9 N ∧ sum_of_digits N = 15 := by
  sorry

end second_smallest_divisible_by_all_less_than_9_sum_of_digits_l812_81227


namespace sqrt_nine_factorial_over_108_l812_81249

theorem sqrt_nine_factorial_over_108 : 
  Real.sqrt (Nat.factorial 9 / 108) = 8 * Real.sqrt 35 := by
  sorry

end sqrt_nine_factorial_over_108_l812_81249


namespace prom_attendance_l812_81234

theorem prom_attendance (total_students : ℕ) (couples : ℕ) (solo_students : ℕ) : 
  total_students = 123 → couples = 60 → solo_students = total_students - 2 * couples →
  solo_students = 3 := by
  sorry

end prom_attendance_l812_81234


namespace degrees_to_radians_l812_81259

theorem degrees_to_radians (degrees : ℝ) (radians : ℝ) : 
  degrees = 12 → radians = degrees * (π / 180) → radians = π / 15 := by
  sorry

end degrees_to_radians_l812_81259


namespace least_number_divisible_by_five_primes_l812_81236

theorem least_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n ∧ p₅ ∣ n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, 
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m ∧ q₅ ∣ m)) ∧
  n = 2310 :=
by sorry

end least_number_divisible_by_five_primes_l812_81236


namespace shaded_area_problem_l812_81209

/-- Given a square FGHI with area 80 and points J, K, L, M on its sides
    such that FK = GL = HM = IJ and FK = 3KG, 
    the area of the quadrilateral JKLM is 50. -/
theorem shaded_area_problem (F G H I J K L M : ℝ × ℝ) : 
  (∃ s : ℝ, s > 0 ∧ (G.1 - F.1)^2 + (G.2 - F.2)^2 = s^2 ∧ s^2 = 80) →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = (L.1 - G.1)^2 + (L.2 - G.2)^2 ∧
   (L.1 - G.1)^2 + (L.2 - G.2)^2 = (M.1 - H.1)^2 + (M.2 - H.2)^2 ∧
   (M.1 - H.1)^2 + (M.2 - H.2)^2 = (J.1 - I.1)^2 + (J.2 - I.2)^2 →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = 9 * ((G.1 - K.1)^2 + (G.2 - K.2)^2) →
  (K.1 - J.1)^2 + (K.2 - J.2)^2 = 50 :=
by sorry


end shaded_area_problem_l812_81209


namespace floor_abs_sum_l812_81291

theorem floor_abs_sum (x : ℝ) (h : x = -5.7) : 
  ⌊|x|⌋ + |⌊x⌋| = 11 := by
sorry

end floor_abs_sum_l812_81291


namespace open_box_volume_l812_81216

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ)
  (h_length : sheet_length = 52)
  (h_width : sheet_width = 36)
  (h_cut : cut_size = 8) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5760 :=
by sorry

end open_box_volume_l812_81216


namespace package_not_qualified_l812_81226

/-- The standard net weight of a biscuit in grams -/
def standard_weight : ℝ := 350

/-- The acceptable deviation from the standard weight in grams -/
def acceptable_deviation : ℝ := 5

/-- The weight of the package in question in grams -/
def package_weight : ℝ := 358

/-- A package is qualified if its weight is within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  (standard_weight - acceptable_deviation ≤ weight) ∧
  (weight ≤ standard_weight + acceptable_deviation)

/-- Theorem stating that the package with weight 358 grams is not qualified -/
theorem package_not_qualified : ¬(is_qualified package_weight) := by
  sorry

end package_not_qualified_l812_81226


namespace fraction_zero_implies_x_equals_two_l812_81264

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (2 - |x|) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end fraction_zero_implies_x_equals_two_l812_81264


namespace samuel_remaining_amount_samuel_remaining_amount_proof_l812_81232

theorem samuel_remaining_amount 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  ℕ :=
  let samuel_received : ℚ := total * samuel_fraction
  let samuel_spent : ℚ := total * spent_fraction
  let samuel_remaining : ℚ := samuel_received - samuel_spent
  132

theorem samuel_remaining_amount_proof 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  samuel_remaining_amount total samuel_fraction spent_fraction h1 h2 h3 = 132 := by
  sorry

end samuel_remaining_amount_samuel_remaining_amount_proof_l812_81232


namespace binomial_12_10_l812_81241

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by sorry

end binomial_12_10_l812_81241


namespace alchemist_safe_combinations_l812_81285

/-- The number of different herbs available to the alchemist. -/
def num_herbs : ℕ := 4

/-- The number of different gems available to the alchemist. -/
def num_gems : ℕ := 6

/-- The number of unstable combinations of herbs and gems. -/
def num_unstable : ℕ := 3

/-- The total number of possible combinations of herbs and gems. -/
def total_combinations : ℕ := num_herbs * num_gems

/-- The number of safe combinations available for the alchemist's elixir. -/
def safe_combinations : ℕ := total_combinations - num_unstable

theorem alchemist_safe_combinations :
  safe_combinations = 21 :=
sorry

end alchemist_safe_combinations_l812_81285


namespace two_integers_problem_l812_81265

theorem two_integers_problem (x y : ℕ+) :
  (x / Nat.gcd x y + y / Nat.gcd x y : ℚ) = 18 →
  Nat.lcm x y = 975 →
  (x = 75 ∧ y = 195) ∨ (x = 195 ∧ y = 75) := by
  sorry

end two_integers_problem_l812_81265


namespace cats_remaining_l812_81205

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end cats_remaining_l812_81205


namespace no_cube_sum_equals_cube_l812_81211

theorem no_cube_sum_equals_cube : ∀ m n : ℕ+, m^3 + 11^3 ≠ n^3 := by
  sorry

end no_cube_sum_equals_cube_l812_81211


namespace carwash_problem_l812_81282

/-- Represents the carwash problem with modified constraints to ensure consistency --/
theorem carwash_problem 
  (car_price SUV_price truck_price motorcycle_price bus_price : ℕ)
  (total_raised : ℕ)
  (num_SUVs num_trucks num_motorcycles : ℕ)
  (max_vehicles : ℕ)
  (h1 : car_price = 7)
  (h2 : SUV_price = 12)
  (h3 : truck_price = 10)
  (h4 : motorcycle_price = 15)
  (h5 : bus_price = 18)
  (h6 : total_raised = 500)
  (h7 : num_SUVs = 3)
  (h8 : num_trucks = 8)
  (h9 : num_motorcycles = 5)
  (h10 : max_vehicles = 20)  -- Modified to make the problem consistent
  : ∃ (num_cars num_buses : ℕ), 
    (num_cars + num_buses + num_SUVs + num_trucks + num_motorcycles ≤ max_vehicles) ∧ 
    (num_cars % 2 = 0) ∧ 
    (num_buses % 2 = 1) ∧
    (car_price * num_cars + bus_price * num_buses + 
     SUV_price * num_SUVs + truck_price * num_trucks + 
     motorcycle_price * num_motorcycles = total_raised) := by
  sorry


end carwash_problem_l812_81282


namespace felix_trees_chopped_l812_81217

/-- Calculates the minimum number of trees chopped given the total spent on sharpening,
    cost per sharpening, and trees chopped before resharpening is needed. -/
def min_trees_chopped (total_spent : ℕ) (cost_per_sharpening : ℕ) (trees_per_sharpening : ℕ) : ℕ :=
  (total_spent / cost_per_sharpening) * trees_per_sharpening

/-- Proves that Felix has chopped down at least 150 trees given the problem conditions. -/
theorem felix_trees_chopped :
  let total_spent : ℕ := 48
  let cost_per_sharpening : ℕ := 8
  let trees_per_sharpening : ℕ := 25
  min_trees_chopped total_spent cost_per_sharpening trees_per_sharpening = 150 := by
  sorry

#eval min_trees_chopped 48 8 25  -- Should output 150

end felix_trees_chopped_l812_81217


namespace point_on_translated_line_l812_81250

/-- The original line -/
def original_line (x : ℝ) : ℝ := x

/-- The translated line -/
def translated_line (x : ℝ) : ℝ := x + 2

/-- Theorem stating that (2, 4) lies on the translated line -/
theorem point_on_translated_line : translated_line 2 = 4 := by sorry

end point_on_translated_line_l812_81250


namespace work_completion_time_l812_81212

/-- Given workers A and B, where:
  * A can complete the entire work in 15 days
  * A works for 5 days and then leaves
  * B completes the remaining work in 6 days
  This theorem proves that B alone can complete the entire work in 9 days -/
theorem work_completion_time (a_total_days b_completion_days : ℕ) 
  (a_worked_days : ℕ) (h1 : a_total_days = 15) (h2 : a_worked_days = 5) 
  (h3 : b_completion_days = 6) : 
  (b_completion_days : ℚ) / ((a_total_days - a_worked_days : ℚ) / a_total_days) = 9 := by
  sorry

end work_completion_time_l812_81212


namespace fraction_subtraction_result_l812_81290

theorem fraction_subtraction_result : 
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - 
  (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 := by
  sorry

end fraction_subtraction_result_l812_81290


namespace sum_of_ninth_powers_l812_81252

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^9 + b^9 = 76 -/
theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^9 + b^9 = 76 := by
  sorry

#check sum_of_ninth_powers

end sum_of_ninth_powers_l812_81252


namespace number_comparisons_l812_81251

theorem number_comparisons :
  (31^11 < 17^14) ∧
  (33^75 > 63^60) ∧
  (82^33 > 26^44) ∧
  (29^31 > 80^23) := by
  sorry

end number_comparisons_l812_81251


namespace relationship_xyz_l812_81238

noncomputable def x : ℝ := Real.sqrt 2
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 0.7 / Real.log 5

theorem relationship_xyz : z < y ∧ y < x := by sorry

end relationship_xyz_l812_81238


namespace quadrilateral_perimeter_l812_81280

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ :=
  sorry

-- Define the perpendicular property
def perpendicular (v w : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_perimeter :
  ∀ (q : Quadrilateral),
    perpendicular (q.B - q.A) (q.C - q.B) →
    perpendicular (q.C - q.D) (q.C - q.B) →
    ‖q.B - q.A‖ = 9 →
    ‖q.D - q.C‖ = 4 →
    ‖q.C - q.B‖ = 12 →
    perimeter q = 38 :=
by
  sorry

end quadrilateral_perimeter_l812_81280


namespace count_divisors_of_M_l812_81247

/-- The number of positive divisors of M, where M = 2^3 * 3^4 * 5^3 * 7^1 -/
def num_divisors : ℕ :=
  (3 + 1) * (4 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^3 * 3^4 * 5^3 * 7^1 -/
def M : ℕ := 2^3 * 3^4 * 5^3 * 7^1

theorem count_divisors_of_M :
  num_divisors = 160 ∧ num_divisors = (Finset.filter (· ∣ M) (Finset.range (M + 1))).card :=
sorry

end count_divisors_of_M_l812_81247


namespace morks_tax_rate_l812_81286

theorem morks_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) : 
  mork_tax_rate > 0 →
  mork_income > 0 →
  let mindy_income := 3 * mork_income
  let mindy_tax_rate := 0.3
  let total_income := mork_income + mindy_income
  let total_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_tax_rate := total_tax / total_income
  combined_tax_rate = 0.325 →
  mork_tax_rate = 0.4 := by
sorry

end morks_tax_rate_l812_81286


namespace polar_equation_is_circle_l812_81223

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    is_circle x y h k r :=
sorry

end polar_equation_is_circle_l812_81223


namespace art_dealer_etchings_sold_l812_81294

theorem art_dealer_etchings_sold (total_earnings : ℕ) (price_low : ℕ) (price_high : ℕ) (num_low : ℕ) :
  total_earnings = 630 →
  price_low = 35 →
  price_high = 45 →
  num_low = 9 →
  ∃ (num_high : ℕ), num_low * price_low + num_high * price_high = total_earnings ∧ num_low + num_high = 16 :=
by sorry

end art_dealer_etchings_sold_l812_81294


namespace all_points_same_number_l812_81240

-- Define a type for points in the plane
structure Point := (x : ℝ) (y : ℝ)

-- Define a function that assigns a real number to each point
def assign : Point → ℝ := sorry

-- Define a predicate for the inscribed circle property
def inscribedCircleProperty (assign : Point → ℝ) : Prop :=
  ∀ A B C : Point,
  ∃ I : Point,
  assign I = (assign A + assign B + assign C) / 3

-- Theorem statement
theorem all_points_same_number
  (h : inscribedCircleProperty assign) :
  ∀ P Q : Point, assign P = assign Q :=
sorry

end all_points_same_number_l812_81240


namespace simplify_expression_l812_81231

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  10 * x^3 * y^2 / (15 * x^2 * y^3) = 5 / 3 := by
  sorry

end simplify_expression_l812_81231


namespace min_value_expression_min_value_achievable_l812_81254

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_z : z = (x + y) / 2) :
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 3 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ z = (x + y) / 2 ∧
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) = 3 / 2 :=
by sorry

end min_value_expression_min_value_achievable_l812_81254


namespace caterpillar_final_position_l812_81281

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction as a unit vector -/
inductive Direction
  | West
  | North
  | East
  | South

/-- Represents the state of the caterpillar -/
structure CaterpillarState where
  position : Point
  direction : Direction
  moveDistance : Nat

/-- Performs a single move and turn -/
def move (state : CaterpillarState) : CaterpillarState :=
  sorry

/-- Performs n moves and turns -/
def moveNTimes (initialState : CaterpillarState) (n : Nat) : CaterpillarState :=
  sorry

/-- The main theorem to prove -/
theorem caterpillar_final_position :
  let initialState : CaterpillarState := {
    position := { x := 15, y := -15 },
    direction := Direction.West,
    moveDistance := 1
  }
  let finalState := moveNTimes initialState 1010
  finalState.position = { x := -491, y := 489 } :=
sorry

end caterpillar_final_position_l812_81281


namespace jacobs_gift_budget_l812_81279

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 :=
by sorry

end jacobs_gift_budget_l812_81279


namespace unique_valid_multiplication_l812_81244

def is_valid_multiplication (a b : Nat) : Prop :=
  a % 10 = 5 ∧
  b % 10 = 5 ∧
  (a * (b / 10 % 10)) % 100 = 25 ∧
  (a / 10 % 10) % 2 = 0 ∧
  b / 10 % 10 < 3 ∧
  1000 ≤ a * b ∧ a * b < 10000

theorem unique_valid_multiplication :
  ∀ a b : Nat, is_valid_multiplication a b → (a = 365 ∧ b = 25) :=
sorry

end unique_valid_multiplication_l812_81244


namespace correct_stratified_sample_l812_81287

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (population : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPopulation := population.freshmen + population.sophomores + population.juniors
  let ratio := totalSample / totalPopulation
  { freshmen := population.freshmen * ratio,
    sophomores := population.sophomores * ratio,
    juniors := population.juniors * ratio }

theorem correct_stratified_sample :
  let population := GradePopulation.mk 560 540 520
  let sample := stratifiedSample population 81
  sample = SampleSize.mk 28 27 26 := by sorry

end correct_stratified_sample_l812_81287


namespace initial_deposit_calculation_l812_81284

-- Define the initial deposit
variable (P : ℝ)

-- Define the interest rates
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.15

-- Define the final amount
def final_amount : ℝ := 690

-- Theorem statement
theorem initial_deposit_calculation :
  (P * (1 + first_year_rate) / 2) * (1 + second_year_rate) = final_amount →
  P = 1000 := by
  sorry


end initial_deposit_calculation_l812_81284


namespace find_c_minus_d_l812_81263

-- Define the functions
def f (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g (x : ℝ) : ℝ := -4 * x + 6
def h (c d : ℝ) (x : ℝ) : ℝ := f c d (g x)

-- State the theorem
theorem find_c_minus_d (c d : ℝ) :
  (∀ x, h c d x = x - 8) →
  (∀ x, h c d (x + 8) = x) →
  c - d = 25/4 := by
sorry

end find_c_minus_d_l812_81263


namespace tangent_curves_l812_81267

theorem tangent_curves (m : ℝ) : 
  (∃ x y : ℝ, y = x^3 + 2 ∧ y^2 - m*x = 1 ∧ 
   ∀ x' : ℝ, x' ≠ x → (x'^3 + 2)^2 - m*x' ≠ 1) ↔ 
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end tangent_curves_l812_81267


namespace ten_men_absent_l812_81274

/-- Represents the work scenario with men and days -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ

/-- Calculates the number of absent men given a work scenario -/
def absentMen (w : WorkScenario) : ℕ :=
  w.totalMen - (w.totalMen * w.originalDays) / w.actualDays

/-- The theorem stating that 10 men became absent in the given scenario -/
theorem ten_men_absent : absentMen ⟨60, 50, 60⟩ = 10 := by
  sorry

end ten_men_absent_l812_81274


namespace magic_8_ball_probability_l812_81253

/-- The probability of getting exactly k positive answers out of n questions
    when each question has a probability p of getting a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers when asking 6 questions
    to a Magic 8 Ball, where each question has a 1/2 chance of getting a positive answer. -/
theorem magic_8_ball_probability : binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end magic_8_ball_probability_l812_81253


namespace base8_to_base7_conversion_l812_81235

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : Nat) : Nat :=
  (n / 343) * 1000 + ((n / 49) % 7) * 100 + ((n / 7) % 7) * 10 + (n % 7)

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 563) = 1162 := by
  sorry

end base8_to_base7_conversion_l812_81235


namespace johns_score_less_than_winning_score_l812_81270

/-- In a blackjack game, given the scores of three players and the winning score,
    prove that the score of the player who didn't win is less than the winning score. -/
theorem johns_score_less_than_winning_score 
  (theodore_score : ℕ) 
  (zoey_score : ℕ) 
  (john_score : ℕ) 
  (winning_score : ℕ) 
  (h1 : theodore_score = 13)
  (h2 : zoey_score = 19)
  (h3 : winning_score = 19)
  (h4 : zoey_score = winning_score)
  (h5 : john_score ≠ zoey_score) : 
  john_score < winning_score :=
sorry

end johns_score_less_than_winning_score_l812_81270


namespace lucy_money_problem_l812_81272

theorem lucy_money_problem (initial_amount : ℚ) : 
  (initial_amount * (2/3) * (3/4) = 15) → initial_amount = 30 := by
  sorry

end lucy_money_problem_l812_81272


namespace georgia_carnation_problem_l812_81292

/-- The number of teachers Georgia sent a dozen carnations to -/
def num_teachers : ℕ := 4

/-- The cost of a single carnation in cents -/
def single_carnation_cost : ℕ := 50

/-- The cost of a dozen carnations in cents -/
def dozen_carnation_cost : ℕ := 400

/-- The number of Georgia's friends -/
def num_friends : ℕ := 14

/-- The total amount Georgia spent in cents -/
def total_spent : ℕ := 2500

theorem georgia_carnation_problem :
  num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost ≤ total_spent ∧
  (num_teachers + 1) * dozen_carnation_cost + num_friends * single_carnation_cost > total_spent :=
by sorry

end georgia_carnation_problem_l812_81292


namespace jake_weight_is_152_l812_81271

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 212

theorem jake_weight_is_152 :
  (jake_weight - 32 = 2 * sister_weight) →
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 152 := by sorry

end jake_weight_is_152_l812_81271


namespace min_value_theorem_l812_81293

theorem min_value_theorem (a b c : ℝ) :
  (∀ x y : ℝ, 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) →
  2 ≤ a + b - c :=
by sorry

end min_value_theorem_l812_81293


namespace olympic_mascot_problem_l812_81248

/-- Olympic Mascot Problem -/
theorem olympic_mascot_problem (m : ℝ) : 
  -- Conditions
  (3000 / m = 2400 / (m - 30)) →
  -- Definitions
  let bing_price := m
  let shuey_price := m - 30
  let bing_sell := 190
  let shuey_sell := 140
  let total_mascots := 200
  let profit (x : ℝ) := (bing_sell - bing_price) * x + (shuey_sell - shuey_price) * (total_mascots - x)
  -- Theorem statements
  (m = 150 ∧ 
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_mascots ∧ (total_mascots - x ≥ (2/3) * x) →
     profit x ≤ profit 120) := by sorry

end olympic_mascot_problem_l812_81248


namespace total_dress_designs_l812_81210

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve types available -/
def num_sleeve_types : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_types = 60 := by
  sorry

end total_dress_designs_l812_81210


namespace production_days_calculation_l812_81215

/-- Given the average daily production for n days and the production on an additional day,
    calculate the number of days n. -/
theorem production_days_calculation (n : ℕ) : 
  (n * 70 + 90) / (n + 1) = 75 → n = 3 := by
  sorry

end production_days_calculation_l812_81215


namespace jerk_tuna_fish_count_l812_81208

theorem jerk_tuna_fish_count (jerk_tuna : ℕ) (tall_tuna : ℕ) : 
  tall_tuna = 2 * jerk_tuna → 
  jerk_tuna + tall_tuna = 432 → 
  jerk_tuna = 144 := by
sorry

end jerk_tuna_fish_count_l812_81208


namespace min_value_a_l812_81242

theorem min_value_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ a) → 
  (∀ b : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ b) → a ≤ b) → 
  a = -1 := by
sorry

end min_value_a_l812_81242


namespace common_difference_is_negative_two_l812_81214

def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

theorem common_difference_is_negative_two :
  ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = -2 := by
  sorry

end common_difference_is_negative_two_l812_81214


namespace hyperbola_parameter_value_l812_81255

/-- Represents a hyperbola with parameter a -/
structure Hyperbola (a : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 / (a - 3) + y^2 / (1 - a) = 1)

/-- Condition that the foci lie on the x-axis -/
def foci_on_x_axis (h : Hyperbola a) : Prop :=
  a > 1 ∧ a > 3

/-- Condition that the focal distance is 4 -/
def focal_distance_is_4 (h : Hyperbola a) : Prop :=
  ∃ (c : ℝ), c^2 = (a - 3) - (1 - a) ∧ 2 * c = 4

/-- Theorem stating that for a hyperbola with the given conditions, a = 4 -/
theorem hyperbola_parameter_value
  (a : ℝ)
  (h : Hyperbola a)
  (h_foci : foci_on_x_axis h)
  (h_focal : focal_distance_is_4 h) :
  a = 4 :=
sorry

end hyperbola_parameter_value_l812_81255


namespace price_reduction_proof_optimal_price_increase_proof_l812_81202

/-- Initial price in yuan per kilogram -/
def initial_price : ℝ := 50

/-- Final price after two reductions in yuan per kilogram -/
def final_price : ℝ := 32

/-- Initial profit in yuan per kilogram -/
def initial_profit : ℝ := 10

/-- Initial daily sales in kilograms -/
def initial_sales : ℝ := 500

/-- Maximum allowed price increase in yuan per kilogram -/
def max_price_increase : ℝ := 8

/-- Sales decrease per yuan of price increase in kilograms -/
def sales_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_profit : ℝ := 6000

/-- Percentage reduction after each price cut -/
def reduction_percentage : ℝ := 0.2

/-- Price increase to achieve target profit -/
def optimal_price_increase : ℝ := 5

theorem price_reduction_proof :
  initial_price * (1 - reduction_percentage)^2 = final_price :=
sorry

theorem optimal_price_increase_proof :
  (initial_profit + optimal_price_increase) * 
  (initial_sales - sales_decrease_rate * optimal_price_increase) = target_profit ∧
  0 < optimal_price_increase ∧
  optimal_price_increase ≤ max_price_increase :=
sorry

end price_reduction_proof_optimal_price_increase_proof_l812_81202


namespace product_pure_imaginary_implies_a_eq_neg_one_l812_81266

/-- Given complex numbers z₁ and z₂, prove that if z₁ · z₂ is purely imaginary, then a = -1 -/
theorem product_pure_imaginary_implies_a_eq_neg_one (a : ℝ) :
  let z₁ : ℂ := a - Complex.I
  let z₂ : ℂ := 1 + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I) → a = -1 := by
  sorry

end product_pure_imaginary_implies_a_eq_neg_one_l812_81266


namespace friend_walking_problem_l812_81230

/-- 
Given two friends walking towards each other on a trail:
- The trail length is 33 km
- They start at opposite ends at the same time
- One friend's speed is 20% faster than the other's
Prove that the faster friend will have walked 18 km when they meet.
-/
theorem friend_walking_problem (v : ℝ) (h_v_pos : v > 0) :
  let trail_length : ℝ := 33
  let speed_ratio : ℝ := 1.2
  let t : ℝ := trail_length / (v * (1 + speed_ratio))
  speed_ratio * v * t = 18 := by sorry

end friend_walking_problem_l812_81230


namespace unicorn_flower_bloom_l812_81245

theorem unicorn_flower_bloom :
  let num_unicorns : ℕ := 12
  let journey_length : ℕ := 15000  -- in meters
  let step_length : ℕ := 3  -- in meters
  let flowers_per_step : ℕ := 7
  
  (journey_length / step_length) * num_unicorns * flowers_per_step = 420000 :=
by
  sorry

end unicorn_flower_bloom_l812_81245


namespace min_value_theorem_l812_81204

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_theorem_l812_81204


namespace simplify_sqrt_x6_plus_x3_l812_81239

theorem simplify_sqrt_x6_plus_x3 (x : ℝ) : 
  Real.sqrt (x^6 + x^3) = |x| * Real.sqrt |x| * Real.sqrt (x^3 + 1) := by
  sorry

end simplify_sqrt_x6_plus_x3_l812_81239


namespace tan_105_degrees_l812_81220

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l812_81220


namespace pear_mango_weight_equivalence_l812_81257

/-- Given that 9 pears weigh the same as 6 mangoes, 
    prove that 36 pears weigh the same as 24 mangoes. -/
theorem pear_mango_weight_equivalence 
  (pear_weight mango_weight : ℝ) 
  (h : 9 * pear_weight = 6 * mango_weight) : 
  36 * pear_weight = 24 * mango_weight := by
  sorry

end pear_mango_weight_equivalence_l812_81257


namespace triangle_area_theorem_l812_81288

/-- Triangle ABC with given properties --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  angle_ABC : Real

/-- Angle bisector AD in triangle ABC --/
structure AngleBisector (T : Triangle) where
  AD : ℝ
  is_bisector : Bool

/-- Area of triangle ADC --/
def area_ADC (T : Triangle) (AB : AngleBisector T) : ℝ := sorry

/-- Main theorem --/
theorem triangle_area_theorem (T : Triangle) (AB : AngleBisector T) :
  T.angle_ABC = 90 ∧ T.AB = 90 ∧ T.BC = 56 ∧ T.AC = 2 * T.BC - 6 ∧ AB.is_bisector = true →
  abs (area_ADC T AB - 1363) < 1 := by
  sorry

end triangle_area_theorem_l812_81288


namespace probability_at_least_one_from_A_l812_81221

/-- Represents the number of classes in each school -/
structure SchoolClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of classes sampled from each school -/
structure SampledClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The total number of classes to be sampled -/
def totalSampled : ℕ := 6

/-- The number of classes to be randomly selected for comparison -/
def comparisonClasses : ℕ := 2

/-- Calculate the probability of selecting at least one class from school A 
    when randomly choosing 2 out of 6 sampled classes -/
def probabilityAtLeastOneFromA (classes : SchoolClasses) (sampled : SampledClasses) : ℚ :=
  sorry

/-- Theorem stating the probability is 3/5 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let classes : SchoolClasses := ⟨12, 6, 18⟩
  let sampled : SampledClasses := ⟨2, 1, 3⟩
  probabilityAtLeastOneFromA classes sampled = 3/5 :=
sorry

end probability_at_least_one_from_A_l812_81221


namespace extended_triangle_pc_length_l812_81225

/-- Triangle ABC with sides AB, BC, CA, extended to point P -/
structure ExtendedTriangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  PC : ℝ

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : ExtendedTriangle) : Prop :=
  t.PC / (t.PC + t.BC) = t.CA / t.AB

theorem extended_triangle_pc_length 
  (t : ExtendedTriangle) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t) : 
  t.PC = 31.5 := by
  sorry

end extended_triangle_pc_length_l812_81225


namespace crabapple_sequences_l812_81283

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients possible in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 248832 := by
  sorry

end crabapple_sequences_l812_81283


namespace min_max_abs_quadratic_l812_81273

theorem min_max_abs_quadratic (p q : ℝ) :
  (∃ (M : ℝ), M ≥ 1/2 ∧ ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) → M ≥ 1/2) :=
sorry

end min_max_abs_quadratic_l812_81273
