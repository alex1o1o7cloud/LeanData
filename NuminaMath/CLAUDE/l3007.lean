import Mathlib

namespace profit_increase_calculation_l3007_300748

/-- Proves that given a 40% increase followed by a 20% decrease, 
    a final increase that results in an overall 68% increase must be a 50% increase. -/
theorem profit_increase_calculation (P : ℝ) (h : P > 0) : 
  let april_profit := 1.40 * P
  let may_profit := 0.80 * april_profit
  let june_profit := 1.68 * P
  (june_profit / may_profit - 1) * 100 = 50 := by
  sorry

end profit_increase_calculation_l3007_300748


namespace sandy_fish_count_l3007_300726

/-- The number of pet fish Sandy has after buying more -/
def total_fish (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sandy now has 32 pet fish -/
theorem sandy_fish_count : total_fish 26 6 = 32 := by
  sorry

end sandy_fish_count_l3007_300726


namespace sum_square_value_l3007_300766

theorem sum_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 36) 
  (h2 : y * (x + y) = 72) : 
  (x + y)^2 = 108 := by
sorry

end sum_square_value_l3007_300766


namespace simplify_fraction_l3007_300736

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end simplify_fraction_l3007_300736


namespace m_range_theorem_l3007_300763

-- Define the conditions
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 4 = 0 ∧ x₂^2 + m*x₂ + 4 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ (Set.Ioo 1 3) ∪ (Set.Ioi 4) :=
sorry

end m_range_theorem_l3007_300763


namespace factor_expression_l3007_300799

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factor_expression_l3007_300799


namespace expression_evaluation_l3007_300768

theorem expression_evaluation : 
  (3 - 4 * (5 - 6)⁻¹)⁻¹ * (1 - 2⁻¹) = (1 : ℚ) / 14 := by sorry

end expression_evaluation_l3007_300768


namespace repeating_decimal_division_l3007_300764

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 →
  b = 36 / 99 →
  a / b = 9 / 4 := by
sorry

end repeating_decimal_division_l3007_300764


namespace runners_meet_time_l3007_300717

/-- The time in seconds for runner P to complete one round -/
def P_time : ℕ := 252

/-- The time in seconds for runner Q to complete one round -/
def Q_time : ℕ := 198

/-- The time in seconds for runner R to complete one round -/
def R_time : ℕ := 315

/-- The time after which all runners meet at the starting point -/
def meet_time : ℕ := 13860

/-- Theorem stating that the meet time is the least common multiple of individual round times -/
theorem runners_meet_time : 
  Nat.lcm (Nat.lcm P_time Q_time) R_time = meet_time := by sorry

end runners_meet_time_l3007_300717


namespace corrected_mean_is_36_4_l3007_300757

/-- Calculates the corrected mean of a set of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  let original_sum := n * original_mean
  let difference := correct_value - wrong_value
  let corrected_sum := original_sum + difference
  corrected_sum / n

/-- Proves that the corrected mean is 36.4 given the specified conditions --/
theorem corrected_mean_is_36_4 :
  corrected_mean 50 36 23 43 = 36.4 := by
sorry

end corrected_mean_is_36_4_l3007_300757


namespace equation_solution_l3007_300733

theorem equation_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
sorry

end equation_solution_l3007_300733


namespace gcf_of_40_120_80_l3007_300724

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by sorry

end gcf_of_40_120_80_l3007_300724


namespace dispersion_measures_l3007_300732

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the statistics
def standardDeviation (s : Sample) : Real := sorry
def median (s : Sample) : Real := sorry
def range (s : Sample) : Real := sorry
def mean (s : Sample) : Real := sorry

-- Define a measure of dispersion
def measuresDispersion (f : Sample → Real) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  (measuresDispersion (standardDeviation)) ∧
  (measuresDispersion (range)) ∧
  (¬ measuresDispersion (median)) ∧
  (¬ measuresDispersion (mean)) :=
sorry

end dispersion_measures_l3007_300732


namespace opposite_of_negative_twelve_l3007_300704

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_negative_twelve : opposite (-12) = 12 := by
  sorry

end opposite_of_negative_twelve_l3007_300704


namespace smallest_k_for_divisibility_property_l3007_300780

theorem smallest_k_for_divisibility_property (n : ℕ) :
  let M := Finset.range n
  (∃ k : ℕ, k > 0 ∧
    (∀ S : Finset ℕ, S ⊆ M → S.card = k →
      ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
    (∀ k' : ℕ, k' < k →
      ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
        ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b))) →
  let k := ⌈(n : ℚ) / 2⌉₊ + 1
  (∀ S : Finset ℕ, S ⊆ M → S.card = k →
    ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Finset ℕ, S ⊆ M ∧ S.card = k' ∧
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b)) :=
by sorry

end smallest_k_for_divisibility_property_l3007_300780


namespace simple_interest_rate_l3007_300703

/-- Calculate the simple interest rate given principal, final amount, and time -/
theorem simple_interest_rate (principal final_amount time : ℝ) 
  (h_principal : principal > 0)
  (h_final : final_amount > principal)
  (h_time : time > 0) :
  (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end simple_interest_rate_l3007_300703


namespace min_value_polynomial_l3007_300708

theorem min_value_polynomial (x y : ℝ) : 
  x^2 + y^2 - 6*x + 8*y + 7 ≥ -18 :=
by sorry

end min_value_polynomial_l3007_300708


namespace right_triangle_hypotenuse_and_area_l3007_300723

theorem right_triangle_hypotenuse_and_area 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 60) (h_b : b = 80) : 
  c = 100 ∧ (1/2 * a * b) = 2400 := by
  sorry

end right_triangle_hypotenuse_and_area_l3007_300723


namespace triangle_with_60_degree_angle_l3007_300742

/-- In a triangle with sides 4, 2√3, and 2 + 2√2, one of the angles is 60°. -/
theorem triangle_with_60_degree_angle :
  ∃ (a b c : ℝ) (α β γ : ℝ),
    a = 4 ∧ 
    b = 2 * Real.sqrt 3 ∧ 
    c = 2 + 2 * Real.sqrt 2 ∧
    α + β + γ = π ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos β ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧
    β = π/3 :=
by sorry


end triangle_with_60_degree_angle_l3007_300742


namespace distance_to_line_not_greater_than_two_l3007_300721

/-- A structure representing a line in a plane -/
structure Line :=
  (points : Set Point)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The distance from a point to a line -/
def distanceToLine (p : Point) (l : Line) : ℝ := sorry

/-- Theorem: If a point P is outside a line l, and there are three points A, B, and C on l
    such that PA = 2, PB = 2.5, and PC = 3, then the distance from P to l is not greater than 2 -/
theorem distance_to_line_not_greater_than_two
  (P : Point) (l : Line) (A B C : Point)
  (h_P_outside : P ∉ l.points)
  (h_ABC_on_l : A ∈ l.points ∧ B ∈ l.points ∧ C ∈ l.points)
  (h_PA : distance P A = 2)
  (h_PB : distance P B = 2.5)
  (h_PC : distance P C = 3) :
  distanceToLine P l ≤ 2 := by sorry

end distance_to_line_not_greater_than_two_l3007_300721


namespace gumball_probability_l3007_300755

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  blue_prob + pink_prob = 1 →
  blue_prob * blue_prob = 16 / 36 →
  pink_prob = 1 / 3 :=
by sorry

end gumball_probability_l3007_300755


namespace union_A_B_complement_A_intersect_B_a_greater_than_one_l3007_300782

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem 2: (∁ₗA) ∩ B = {x | 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 1
theorem a_greater_than_one (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 1 := by sorry

end union_A_B_complement_A_intersect_B_a_greater_than_one_l3007_300782


namespace function_transformation_l3007_300789

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 1 = -2) : f (-(-1)) + 1 = -1 := by
  sorry

end function_transformation_l3007_300789


namespace clock_digit_sum_probability_l3007_300745

def total_times : ℕ := 1440
def times_with_sum_23 : ℕ := 4

theorem clock_digit_sum_probability :
  (times_with_sum_23 : ℚ) / total_times = 1 / 360 := by
  sorry

end clock_digit_sum_probability_l3007_300745


namespace lukes_mother_ten_bills_l3007_300744

def school_fee : ℕ := 350

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2

def father_fifty : ℕ := 4
def father_twenty : ℕ := 1
def father_ten : ℕ := 1

theorem lukes_mother_ten_bills (mother_ten : ℕ) :
  mother_fifty * 50 + mother_twenty * 20 + mother_ten * 10 +
  father_fifty * 50 + father_twenty * 20 + father_ten * 10 = school_fee →
  mother_ten = 3 := by
  sorry

end lukes_mother_ten_bills_l3007_300744


namespace infinitely_many_non_representable_primes_l3007_300750

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- Evaluate a polynomial at a given point -/
def eval (p : IntPolynomial) (x : ℕ) : ℤ := sorry

/-- The set of values that can be represented by a list of polynomials -/
def representableSet (polys : List IntPolynomial) : Set ℕ :=
  {n : ℕ | ∃ (p : IntPolynomial) (a : ℕ), p ∈ polys ∧ eval p a = n}

/-- The main theorem -/
theorem infinitely_many_non_representable_primes
  (n : ℕ)
  (polys : List IntPolynomial)
  (h_degree : ∀ p ∈ polys, degree p ≥ 2)
  : Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ representableSet polys} := by
  sorry

end infinitely_many_non_representable_primes_l3007_300750


namespace pants_and_belt_price_difference_l3007_300700

def price_difference (total_cost pants_price : ℝ) : ℝ :=
  total_cost - 2 * pants_price

theorem pants_and_belt_price_difference :
  let total_cost : ℝ := 70.93
  let pants_price : ℝ := 34
  pants_price < (total_cost - pants_price) →
  price_difference total_cost pants_price = 2.93 := by
sorry

end pants_and_belt_price_difference_l3007_300700


namespace average_chapters_per_book_l3007_300749

theorem average_chapters_per_book (total_chapters : Real) (total_books : Real) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) :
  total_chapters / total_books = 4.25 := by
  sorry

end average_chapters_per_book_l3007_300749


namespace binomial_coefficient_problem_l3007_300710

theorem binomial_coefficient_problem (a : ℝ) : 
  (6 : ℕ).choose 1 * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end binomial_coefficient_problem_l3007_300710


namespace flensburgian_iff_even_l3007_300783

/-- A set of equations is Flensburgian if there exists an i ∈ {1, 2, 3} such that
    for every solution where all variables are pairwise different, x_i > x_j for all j ≠ i -/
def IsFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, f x y z → x ≠ y → y ≠ z → z ≠ x →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The set of equations a^n + b = a and c^(n+1) + b^2 = ab -/
def EquationSet (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a * b

theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  IsFlensburgian (EquationSet n) ↔ Even n :=
sorry

end flensburgian_iff_even_l3007_300783


namespace unique_solution_l3007_300793

theorem unique_solution : ∃! x : ℝ, x^29 * 4^15 = 2 * 10^29 := by sorry

end unique_solution_l3007_300793


namespace denominator_of_0_34_l3007_300737

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem denominator_of_0_34 :
  (decimal_to_fraction 0.34).2 = 100 := by sorry

end denominator_of_0_34_l3007_300737


namespace divisibility_problem_l3007_300781

theorem divisibility_problem (N : ℕ) : 
  N % 44 = 0 → N % 39 = 15 → N / 44 = 3 := by
sorry

end divisibility_problem_l3007_300781


namespace city_population_l3007_300739

/-- Proves that given the conditions about women and retail workers in a city,
    the total population is 6,000,000 -/
theorem city_population (total_population : ℕ) : 
  (total_population / 2 : ℚ) = (total_population : ℚ) * (1 / 2 : ℚ) →
  ((total_population / 2 : ℚ) * (1 / 3 : ℚ) : ℚ) = 1000000 →
  total_population = 6000000 := by
  sorry

end city_population_l3007_300739


namespace greatest_five_digit_multiple_of_6_l3007_300790

def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def uses_digits (n : ℕ) (digits : List ℕ) : Prop :=
  (n.digits 10).toFinset = digits.toFinset

theorem greatest_five_digit_multiple_of_6 :
  ∃ (n : ℕ),
    n ≥ 10000 ∧
    n < 100000 ∧
    is_multiple_of_6 n ∧
    uses_digits n [2, 5, 6, 8, 9] ∧
    ∀ (m : ℕ),
      m ≥ 10000 →
      m < 100000 →
      is_multiple_of_6 m →
      uses_digits m [2, 5, 6, 8, 9] →
      m ≤ n ∧
    n = 98652 :=
  sorry

end greatest_five_digit_multiple_of_6_l3007_300790


namespace second_person_receives_345_l3007_300741

/-- The total amount of money distributed -/
def total_amount : ℕ := 1000

/-- The sequence of distributions -/
def distribution_sequence (n : ℕ) : ℕ := n

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The largest n such that the sum of the first n natural numbers is at most the total amount -/
def max_n : ℕ := 44

/-- The amount received by the second person (Bernardo) -/
def amount_received_by_second : ℕ := 345

/-- Theorem stating that the second person (Bernardo) receives 345 reais -/
theorem second_person_receives_345 :
  (∀ n : ℕ, n ≤ max_n → sum_first_n n ≤ total_amount) →
  (∀ k : ℕ, k ≤ 15 → distribution_sequence (3*k - 1) ≤ max_n) →
  amount_received_by_second = 345 := by
  sorry

end second_person_receives_345_l3007_300741


namespace inequality_property_l3007_300709

theorem inequality_property (a b c d : ℝ) : a > b → c > d → a - d > b - c := by
  sorry

end inequality_property_l3007_300709


namespace abs_eq_sqrt_sq_l3007_300795

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end abs_eq_sqrt_sq_l3007_300795


namespace fraction_to_decimal_l3007_300770

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end fraction_to_decimal_l3007_300770


namespace price_increase_percentage_l3007_300758

def lowest_price : ℝ := 18
def highest_price : ℝ := 24

theorem price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 33.33333333333333 := by
sorry

end price_increase_percentage_l3007_300758


namespace vase_transport_problem_l3007_300712

theorem vase_transport_problem (x : ℕ) : 
  (∃ C : ℝ, 
    (10 * (x - 50) - C = -300) ∧ 
    (12 * (x - 50) - C = 800)) → 
  x = 600 := by
  sorry

end vase_transport_problem_l3007_300712


namespace simplify_sqrt_expression_l3007_300797

theorem simplify_sqrt_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l3007_300797


namespace new_average_age_l3007_300707

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → initial_avg = 14 → new_person_age = 34 → 
  ((n : ℝ) * initial_avg + new_person_age) / ((n : ℝ) + 1) = 16 := by
  sorry

end new_average_age_l3007_300707


namespace inequality_proof_l3007_300713

theorem inequality_proof (n : ℕ) (a b : ℝ) 
  (h1 : n ≠ 1) (h2 : a > b) (h3 : b > 0) : 
  ((a + b) / 2) ^ n < (a ^ n + b ^ n) / 2 := by
  sorry

end inequality_proof_l3007_300713


namespace textbook_selling_price_l3007_300731

/-- The selling price of a textbook, given its cost price and profit -/
theorem textbook_selling_price (cost_price profit : ℝ) (h1 : cost_price = 44) (h2 : profit = 11) :
  cost_price + profit = 55 := by
  sorry

#check textbook_selling_price

end textbook_selling_price_l3007_300731


namespace sum_of_fourth_powers_l3007_300760

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) : 
  x^4 + y^4 = 82 := by
sorry

end sum_of_fourth_powers_l3007_300760


namespace three_sides_form_triangle_l3007_300772

/-- A polygon circumscribed around a circle -/
structure CircumscribedPolygon where
  n : ℕ
  sides : Fin n → ℝ
  is_circumscribed : Bool

/-- The theorem stating that in any polygon circumscribed around a circle 
    with at least 4 sides, there exist three sides that can form a triangle -/
theorem three_sides_form_triangle (P : CircumscribedPolygon) 
  (h : P.n ≥ 4) (h_circ : P.is_circumscribed = true) :
  ∃ (i j k : Fin P.n), 
    P.sides i + P.sides j > P.sides k ∧
    P.sides j + P.sides k > P.sides i ∧
    P.sides k + P.sides i > P.sides j :=
sorry


end three_sides_form_triangle_l3007_300772


namespace teal_survey_l3007_300722

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_green : more_green = 90)
  (h_both : both = 40)
  (h_neither : neither = 25) :
  total - (more_green - both + both + neither) = 75 :=
sorry

end teal_survey_l3007_300722


namespace sameColorPairsTheorem_l3007_300791

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
def sameColorPairs (white green brown blue : ℕ) : ℕ :=
  Nat.choose white 2 + Nat.choose green 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: Given a drawer with 16 distinguishable socks (6 white, 4 green, 4 brown, and 2 blue),
    the number of ways to choose a pair of socks of the same color is 28. -/
theorem sameColorPairsTheorem :
  sameColorPairs 6 4 4 2 = 28 := by
  sorry

end sameColorPairsTheorem_l3007_300791


namespace major_premise_incorrect_l3007_300794

theorem major_premise_incorrect : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end major_premise_incorrect_l3007_300794


namespace sum_of_digits_2008_5009_7_l3007_300746

theorem sum_of_digits_2008_5009_7 :
  ∃ (n : ℕ), n = 2^2008 * 5^2009 * 7 ∧ (List.sum (Nat.digits 10 n) = 5) :=
by sorry

end sum_of_digits_2008_5009_7_l3007_300746


namespace floor_ceiling_difference_l3007_300792

theorem floor_ceiling_difference : 
  ⌊(14 : ℝ) / 5 * (31 : ℝ) / 4⌋ - ⌈(14 : ℝ) / 5 * ⌈(31 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end floor_ceiling_difference_l3007_300792


namespace f_upper_bound_l3007_300752

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 3

theorem f_upper_bound :
  ∃ (a : ℝ), a = 3 ∧ (∀ x ∈ Set.Icc (-2) 2, f x ≤ a) ∧
  (∀ b < a, ∃ x ∈ Set.Icc (-2) 2, f x > b) :=
sorry

end f_upper_bound_l3007_300752


namespace simplify_and_rationalize_l3007_300769

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 6 / Real.sqrt 8) = Real.sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l3007_300769


namespace smallest_perfect_square_divisible_by_4_and_5_l3007_300740

theorem smallest_perfect_square_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (n % 4 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → (k % 4 = 0) → (k % 5 = 0) → k ≥ n) ∧
  n = 400 :=
by sorry

end smallest_perfect_square_divisible_by_4_and_5_l3007_300740


namespace no_solution_equation_l3007_300761

theorem no_solution_equation : 
  ¬∃ (x : ℝ), (2 / (x + 1) + 3 / (x - 1) = 6 / (x^2 - 1)) := by
  sorry

end no_solution_equation_l3007_300761


namespace solve_star_equation_l3007_300728

-- Define the * operation
def star_op (a b : ℚ) : ℚ :=
  if a ≥ b then a^2 * b else a * b^2

-- Theorem statement
theorem solve_star_equation :
  ∃! m : ℚ, star_op 3 m = 48 ∧ m > 0 :=
sorry

end solve_star_equation_l3007_300728


namespace band_members_count_l3007_300788

theorem band_members_count :
  ∃! N : ℕ, 100 < N ∧ N ≤ 200 ∧
  (∃ k : ℕ, N + 2 = 8 * k) ∧
  (∃ m : ℕ, N + 3 = 9 * m) :=
by sorry

end band_members_count_l3007_300788


namespace triangle_ratio_bound_l3007_300705

/-- For any triangle with perimeter p, circumradius R, and inradius r,
    the expression p/R * (1 - r/(3R)) is at most 5√3/2 -/
theorem triangle_ratio_bound (p R r : ℝ) (hp : p > 0) (hR : R > 0) (hr : r > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p = a + b + c ∧
    R = (a * b * c) / (4 * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
    r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * p)) :
  p / R * (1 - r / (3 * R)) ≤ 5 * Real.sqrt 3 / 2 :=
sorry

end triangle_ratio_bound_l3007_300705


namespace only_log23_not_computable_l3007_300720

-- Define the given logarithm values
def log27 : ℝ := 1.4314
def log32 : ℝ := 1.5052

-- Define a function to represent the computability of a logarithm
def is_computable (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log27 log32 = Real.log x

-- State the theorem
theorem only_log23_not_computable :
  ¬(is_computable 23) ∧ 
  (is_computable (9/8)) ∧ 
  (is_computable 28) ∧ 
  (is_computable 800) ∧ 
  (is_computable 0.45) := by
  sorry

end only_log23_not_computable_l3007_300720


namespace sum_of_squares_zero_l3007_300784

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c form a basis in space V, and real numbers x, y, z 
    satisfy the equation x*a + y*b + z*c = 0, then x^2 + y^2 + z^2 = 0 -/
theorem sum_of_squares_zero (a b c : V) (x y z : ℝ) 
  (h_basis : LinearIndependent ℝ ![a, b, c]) 
  (h_eq : x • a + y • b + z • c = 0) : 
  x^2 + y^2 + z^2 = 0 := by
  sorry

end sum_of_squares_zero_l3007_300784


namespace sibling_discount_calculation_l3007_300767

/-- Represents the tuition cost at the music school -/
def regular_tuition : ℕ := 45

/-- Represents the discounted cost for both children -/
def discounted_cost : ℕ := 75

/-- Represents the number of children -/
def num_children : ℕ := 2

/-- Calculates the sibling discount -/
def sibling_discount : ℕ :=
  regular_tuition * num_children - discounted_cost

theorem sibling_discount_calculation :
  sibling_discount = 15 := by sorry

end sibling_discount_calculation_l3007_300767


namespace unique_satisfying_pair_l3007_300725

/-- A pair of real numbers satisfying both arithmetic and geometric progression conditions -/
def SatisfyingPair (a b : ℝ) : Prop :=
  -- Arithmetic progression condition
  (15 : ℝ) - a = a - b ∧ a - b = b - (a * b) ∧
  -- Geometric progression condition
  ∃ r : ℝ, a * b = 15 * r^3 ∧ r > 0

/-- Theorem stating that (15, 15) is the only pair satisfying both conditions -/
theorem unique_satisfying_pair :
  ∀ a b : ℝ, SatisfyingPair a b → a = 15 ∧ b = 15 :=
by sorry

end unique_satisfying_pair_l3007_300725


namespace tan_negative_4095_degrees_l3007_300718

theorem tan_negative_4095_degrees : Real.tan ((-4095 : ℝ) * Real.pi / 180) = 1 := by
  sorry

end tan_negative_4095_degrees_l3007_300718


namespace min_students_l3007_300734

/-- Represents the number of students in each income group -/
structure IncomeGroups where
  low : ℕ
  middle : ℕ
  high : ℕ

/-- Represents the lowest salary in each income range -/
structure LowestSalaries where
  low : ℝ
  middle : ℝ
  high : ℝ

/-- Represents the median salary in each income range -/
def medianSalaries (lowest : LowestSalaries) : LowestSalaries :=
  { low := lowest.low + 50000
  , middle := lowest.middle + 40000
  , high := lowest.high + 30000 }

/-- The conditions of the problem -/
structure GraduatingClass where
  groups : IncomeGroups
  lowest : LowestSalaries
  salary_range : ℝ
  average_salary : ℝ
  median_salary : ℝ
  (high_twice_low : groups.high = 2 * groups.low)
  (middle_sum_others : groups.middle = groups.low + groups.high)
  (salary_range_constant : ∀ (r : LowestSalaries → ℝ), r (medianSalaries lowest) - r lowest = salary_range)
  (average_above_median : average_salary = median_salary + 20000)

/-- The theorem to prove -/
theorem min_students (c : GraduatingClass) : 
  c.groups.low + c.groups.middle + c.groups.high ≥ 6 :=
sorry

end min_students_l3007_300734


namespace best_fit_highest_r_squared_l3007_300787

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  r_squared : ℝ
  h_nonneg : 0 ≤ r_squared
  h_le_one : r_squared ≤ 1

/-- Determines if a model has better fit than another based on R² values -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.r_squared > model2.r_squared

/-- Theorem: The model with the highest R² value has the best fitting effect -/
theorem best_fit_highest_r_squared (models : List RegressionModel) (best_model : RegressionModel) 
    (h_best_in_models : best_model ∈ models)
    (h_best_r_squared : ∀ model ∈ models, model.r_squared ≤ best_model.r_squared) :
    ∀ model ∈ models, better_fit best_model model ∨ best_model = model :=
  sorry

end best_fit_highest_r_squared_l3007_300787


namespace quadratic_roots_ratio_l3007_300756

theorem quadratic_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = -10 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 + 10*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 24 := by
sorry

end quadratic_roots_ratio_l3007_300756


namespace jinho_money_problem_l3007_300701

theorem jinho_money_problem (initial_money : ℕ) : 
  (initial_money / 2 + 300 + 
   ((initial_money - (initial_money / 2 + 300)) / 2 + 400) = initial_money) → 
  initial_money = 2200 :=
by sorry

end jinho_money_problem_l3007_300701


namespace square_of_sum_l3007_300715

theorem square_of_sum (x y : ℝ) 
  (h1 : x * (2 * x + y) = 36)
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 108 := by
sorry

end square_of_sum_l3007_300715


namespace correct_dice_configuration_l3007_300798

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  h : dots ≥ 1 ∧ dots ≤ 6

/-- Represents the configuration of four dice -/
structure DiceConfiguration where
  faceA : DieFace
  faceB : DieFace
  faceC : DieFace
  faceD : DieFace

/-- Theorem stating the correct number of dots on each face -/
theorem correct_dice_configuration :
  ∃ (config : DiceConfiguration),
    config.faceA.dots = 3 ∧
    config.faceB.dots = 5 ∧
    config.faceC.dots = 6 ∧
    config.faceD.dots = 5 := by
  sorry

end correct_dice_configuration_l3007_300798


namespace sufficient_not_necessary_l3007_300775

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, (x + 4) * (x + 3) ≥ 0 → x^2 + y^2 + 4*x + 3 ≤ 0) ∧
  (∃ x y, x^2 + y^2 + 4*x + 3 ≤ 0 ∧ (x + 4) * (x + 3) < 0) :=
sorry

end sufficient_not_necessary_l3007_300775


namespace expansion_coefficient_l3007_300786

/-- The coefficient of x³y²z in the expansion of (x+y+z)⁶ -/
def coefficient_x3y2z : ℕ :=
  Nat.choose 6 3 * Nat.choose 3 2

theorem expansion_coefficient :
  coefficient_x3y2z = 60 := by
  sorry

end expansion_coefficient_l3007_300786


namespace distributive_property_implication_l3007_300743

theorem distributive_property_implication (a b c : ℝ) (h : c ≠ 0) :
  (∀ x y z : ℝ, (x + y) * z = x * z + y * z) →
  (a + b) / c = a / c + b / c :=
by sorry

end distributive_property_implication_l3007_300743


namespace rock_mist_distance_l3007_300714

/-- The distance from the city to Sky Falls in miles -/
def distance_to_sky_falls : ℝ := 8

/-- The factor by which Rock Mist Mountains are farther from the city than Sky Falls -/
def rock_mist_factor : ℝ := 50

/-- The distance from the city to Rock Mist Mountains in miles -/
def distance_to_rock_mist : ℝ := distance_to_sky_falls * rock_mist_factor

theorem rock_mist_distance : distance_to_rock_mist = 400 := by
  sorry

end rock_mist_distance_l3007_300714


namespace unique_prime_between_30_and_40_with_remainder_1_mod_6_l3007_300776

theorem unique_prime_between_30_and_40_with_remainder_1_mod_6 :
  ∃! n : ℕ, 30 < n ∧ n < 40 ∧ Nat.Prime n ∧ n % 6 = 1 :=
by
  -- The proof goes here
  sorry

end unique_prime_between_30_and_40_with_remainder_1_mod_6_l3007_300776


namespace gcd_102_238_l3007_300747

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l3007_300747


namespace root_squared_plus_root_plus_one_equals_two_l3007_300751

theorem root_squared_plus_root_plus_one_equals_two (a : ℝ) : 
  a^2 + a - 1 = 0 → a^2 + a + 1 = 2 := by
  sorry

end root_squared_plus_root_plus_one_equals_two_l3007_300751


namespace flight_duration_sum_l3007_300796

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for time zone difference -/
def timeDifference (departure : Time) (arrival : Time) (timeZoneDiff : ℤ) : ℕ :=
  let totalMinutes := (arrival.hours - departure.hours) * 60 + arrival.minutes - departure.minutes
  (totalMinutes + timeZoneDiff * 60).toNat

/-- Theorem stating the flight duration property -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
    (h : ℕ) (m : ℕ) (mPos : 0 < m) (mLt60 : m < 60) :
    departureTime.hours = 15 ∧ departureTime.minutes = 15 →
    arrivalTime.hours = 16 ∧ arrivalTime.minutes = 50 →
    timeDifference departureTime arrivalTime (-1) = h * 60 + m →
    h + m = 36 := by
  sorry

end flight_duration_sum_l3007_300796


namespace radiator_water_fraction_l3007_300759

/-- Represents the fraction of water remaining after a number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (replacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ replacements

/-- Proves that the fraction of water in the radiator after 5 replacements is 243/1024 -/
theorem radiator_water_fraction :
  waterFraction 20 5 5 = 243 / 1024 := by
  sorry

#eval waterFraction 20 5 5

end radiator_water_fraction_l3007_300759


namespace X_is_element_of_Y_l3007_300735

def X : Set Nat := {0, 1}

def Y : Set (Set Nat) := {s | s ⊆ X}

theorem X_is_element_of_Y : X ∈ Y := by sorry

end X_is_element_of_Y_l3007_300735


namespace euler_line_equation_l3007_300729

/-- Triangle ABC with vertices A(1,3) and B(2,1), and |AC| = |BC| -/
structure Triangle :=
  (C : ℝ × ℝ)
  (ac_eq_bc : (C.1 - 1)^2 + (C.2 - 3)^2 = (C.2 - 2)^2 + (C.2 - 1)^2)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0}

/-- Theorem: The Euler line of triangle ABC is 2x - 4y + 5 = 0 -/
theorem euler_line_equation (t : Triangle) : 
  EulerLine t = {p : ℝ × ℝ | 2 * p.1 - 4 * p.2 + 5 = 0} := by
  sorry

end euler_line_equation_l3007_300729


namespace log_cos_sum_squared_l3007_300738

theorem log_cos_sum_squared : 
  (Real.log (Real.cos (20 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (40 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (80 * π / 180)) / Real.log (Real.sqrt 2)) ^ 2 = 16 := by
  sorry

end log_cos_sum_squared_l3007_300738


namespace solve_equation_l3007_300771

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end solve_equation_l3007_300771


namespace avery_donation_l3007_300727

/-- The number of shirts Avery puts in the donation box -/
def num_shirts : ℕ := 4

/-- The number of pants Avery puts in the donation box -/
def num_pants : ℕ := 2 * num_shirts

/-- The number of shorts Avery puts in the donation box -/
def num_shorts : ℕ := num_pants / 2

/-- The total number of clothes Avery is donating -/
def total_clothes : ℕ := num_shirts + num_pants + num_shorts

theorem avery_donation : total_clothes = 16 := by
  sorry

end avery_donation_l3007_300727


namespace fifteen_customers_tipped_l3007_300753

/-- Calculates the number of customers who left a tip --/
def customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : ℕ :=
  initial_customers + additional_customers - non_tipping_customers

/-- Theorem: Given the conditions, prove that 15 customers left a tip --/
theorem fifteen_customers_tipped :
  customers_who_tipped 29 20 34 = 15 := by
  sorry

end fifteen_customers_tipped_l3007_300753


namespace projectile_meeting_time_l3007_300719

theorem projectile_meeting_time (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  distance = 1455 →
  speed1 = 470 →
  speed2 = 500 →
  (distance / (speed1 + speed2)) * 60 = 90 := by
sorry

end projectile_meeting_time_l3007_300719


namespace purchase_price_l3007_300774

/-- The purchase price of an article given markup conditions -/
theorem purchase_price (M : ℝ) (P : ℝ) : M = 0.30 * P + 12 → M = 55 → P = 143.33 := by
  sorry

end purchase_price_l3007_300774


namespace geometric_sequence_max_first_term_l3007_300754

theorem geometric_sequence_max_first_term 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 ≥ 1)
  (h_a2 : a 2 ≤ 2)
  (h_a3 : a 3 ≥ 3) :
  a 1 ≤ 4/3 :=
sorry

end geometric_sequence_max_first_term_l3007_300754


namespace max_profit_at_half_l3007_300730

/-- The profit function for a souvenir sale after process improvement -/
def profit_function (x : ℝ) : ℝ := 500 * (1 + 4*x - x^2 - 4*x^3)

/-- The theorem stating the maximum profit and the corresponding price increase -/
theorem max_profit_at_half :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 0 < x → x < 1 → profit_function x ≤ max_profit) ∧
    profit_function (1/2) = max_profit ∧
    max_profit = 11125 := by
  sorry

end max_profit_at_half_l3007_300730


namespace quadratic_roots_triangle_range_l3007_300765

/-- Given a quadratic equation x^2 - 2x + m = 0 with two real roots a and b,
    where a, b, and 1 can form the sides of a triangle, prove that 3/4 < m ≤ 1 --/
theorem quadratic_roots_triangle_range (m : ℝ) (a b : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ x = a ∨ x = b) → 
  (a + b > 1 ∧ a > 0 ∧ b > 0 ∧ 1 > 0 ∧ a + 1 > b ∧ b + 1 > a) →
  (3/4 < m ∧ m ≤ 1) := by
  sorry


end quadratic_roots_triangle_range_l3007_300765


namespace sequence_nonpositive_l3007_300778

theorem sequence_nonpositive (N : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i ∈ Finset.range (N - 1), a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
sorry

end sequence_nonpositive_l3007_300778


namespace largest_x_and_ratio_l3007_300779

theorem largest_x_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (5 * x / 7 + 1 = 3 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-7 + 21 * Real.sqrt 22) / 10) ∧
  (a * c * d / b = -70) :=
by sorry

end largest_x_and_ratio_l3007_300779


namespace melissa_total_score_l3007_300762

/-- Given a player who scores the same number of points in each game,
    calculate their total score across multiple games. -/
def totalPoints (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Theorem: A player scoring 120 points per game for 10 games
    will have a total score of 1200 points. -/
theorem melissa_total_score :
  totalPoints 120 10 = 1200 := by
  sorry

end melissa_total_score_l3007_300762


namespace circle_center_l3007_300785

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 12 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle given by x^2 - 6x + y^2 + 2y - 12 = 0 is (3, -1) -/
theorem circle_center : 
  ∃ (h k : ℝ), CircleCenter h k = (3, -1) ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 12 + 6*h - 2*k) :=
sorry

end circle_center_l3007_300785


namespace pie_eating_contest_l3007_300773

theorem pie_eating_contest (a b c d : ℤ) (h1 : a = 7 ∧ b = 8) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b - (c : ℚ) / d = 1 / 24 := by
  sorry

end pie_eating_contest_l3007_300773


namespace quinary_324_equals_binary_1011001_l3007_300716

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun ⟨i, d⟩ acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem quinary_324_equals_binary_1011001 :
  decimal_to_binary (quinary_to_decimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end quinary_324_equals_binary_1011001_l3007_300716


namespace p_minus_q_value_l3007_300711

theorem p_minus_q_value (p q : ℚ) 
  (h1 : -6 / p = 3/2) 
  (h2 : 8 / q = -1/4) : 
  p - q = 28 := by
sorry

end p_minus_q_value_l3007_300711


namespace rectangle_area_change_l3007_300706

/-- Theorem: Area change of a rectangle after length decrease and width increase -/
theorem rectangle_area_change
  (l w : ℝ)  -- l: original length, w: original width
  (hl : l > 0)  -- length is positive
  (hw : w > 0)  -- width is positive
  : (0.9 * l) * (1.3 * w) = 1.17 * (l * w) :=
by sorry

end rectangle_area_change_l3007_300706


namespace complex_magnitude_example_l3007_300702

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end complex_magnitude_example_l3007_300702


namespace sum_digits_base7_999_l3007_300777

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base7_999 : sumList (toBase7 999) = 15 := by
  sorry

end sum_digits_base7_999_l3007_300777
