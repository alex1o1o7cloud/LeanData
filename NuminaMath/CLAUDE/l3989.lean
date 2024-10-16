import Mathlib

namespace NUMINAMATH_CALUDE_smallest_divisible_k_l3989_398921

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem stating that 120 is the smallest positive integer k such that p(z) divides f(k)(z) -/
theorem smallest_divisible_k : 
  (∀ z : ℂ, p z ∣ f 120 z) ∧ 
  (∀ k : ℕ, k < 120 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_k_l3989_398921


namespace NUMINAMATH_CALUDE_lines_perpendicular_l3989_398905

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -4 / 3

-- Define the equations of the two lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: The two lines are perpendicular
theorem lines_perpendicular : slope1 * slope2 = -1 := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l3989_398905


namespace NUMINAMATH_CALUDE_minji_clothes_combinations_l3989_398985

theorem minji_clothes_combinations (tops : ℕ) (bottoms : ℕ) 
  (h1 : tops = 3) (h2 : bottoms = 5) : tops * bottoms = 15 := by
  sorry

end NUMINAMATH_CALUDE_minji_clothes_combinations_l3989_398985


namespace NUMINAMATH_CALUDE_product_of_primes_with_sum_85_l3989_398944

theorem product_of_primes_with_sum_85 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 85 → p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_with_sum_85_l3989_398944


namespace NUMINAMATH_CALUDE_store_pricing_l3989_398916

theorem store_pricing (h n : ℝ) 
  (eq1 : 4 * h + 5 * n = 10.45)
  (eq2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l3989_398916


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l3989_398950

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) : 
  Real.cos (Real.pi - 2*α) = -1/9 := by sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l3989_398950


namespace NUMINAMATH_CALUDE_production_growth_l3989_398906

theorem production_growth (a : ℝ) (x : ℕ) (y : ℝ) (h : x > 0) :
  y = a * (1 + 0.05) ^ x ↔ 
  (∀ n : ℕ, n ≤ x → 
    (n = 0 → y = a) ∧ 
    (n > 0 → y = a * (1 + 0.05) ^ n)) :=
sorry

end NUMINAMATH_CALUDE_production_growth_l3989_398906


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l3989_398945

theorem number_satisfying_equation : ∃ x : ℝ, x^2 + 4 = 5*x ∧ (x = 4 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l3989_398945


namespace NUMINAMATH_CALUDE_first_test_score_l3989_398952

theorem first_test_score (second_score average : ℝ) (h1 : second_score = 84) (h2 : average = 81) :
  let first_score := 2 * average - second_score
  first_score = 78 := by
sorry

end NUMINAMATH_CALUDE_first_test_score_l3989_398952


namespace NUMINAMATH_CALUDE_function_inequality_l3989_398902

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → (x + 1) * f x + x * f' x ≥ 0) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3989_398902


namespace NUMINAMATH_CALUDE_granger_age_multiple_l3989_398970

/-- The multiple of Mr. Granger's son's age last year that Mr. Granger's age last year was 4 years less than -/
def multiple_last_year (grangers_age : ℕ) (sons_age : ℕ) : ℚ :=
  (grangers_age - 1) / (sons_age - 1)

/-- Mr. Granger's current age -/
def grangers_age : ℕ := 42

/-- Mr. Granger's son's current age -/
def sons_age : ℕ := 16

theorem granger_age_multiple : multiple_last_year grangers_age sons_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_granger_age_multiple_l3989_398970


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3989_398907

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - (x - 2)/3 > 1) ↔ (x < 1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3989_398907


namespace NUMINAMATH_CALUDE_shortest_side_length_l3989_398968

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the shortest side (opposite to 30° angle) -/
  short : ℝ
  /-- The length of the middle side (opposite to 60° angle) -/
  middle : ℝ
  /-- The length of the hypotenuse (opposite to 90° angle) -/
  hypotenuse : ℝ
  /-- The ratio of sides in a 30-60-90 triangle -/
  ratio_prop : short = middle / Real.sqrt 3 ∧ middle = hypotenuse / 2

/-- Theorem: In a 30-60-90 triangle with hypotenuse 30 units, the shortest side is 15 units -/
theorem shortest_side_length (t : Triangle30_60_90) (h : t.hypotenuse = 30) : t.short = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l3989_398968


namespace NUMINAMATH_CALUDE_inverse_function_problem_l3989_398908

/-- Given a function h and its inverse f⁻¹, prove that 7c + 7d = 2 -/
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, (7 * x - 6 : ℝ) = (Function.invFun (fun x ↦ c * x + d) x - 5)) →
  7 * c + 7 * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l3989_398908


namespace NUMINAMATH_CALUDE_horner_v₃_value_l3989_398909

-- Define the polynomial coefficients
def a₀ : ℝ := 2
def a₁ : ℝ := 0.3
def a₂ : ℝ := 1
def a₃ : ℝ := 0
def a₄ : ℝ := 6
def a₅ : ℝ := -5
def a₆ : ℝ := 1

-- Define x
def x : ℝ := -2

-- Define Horner's method steps
def v₀ : ℝ := a₆
def v₁ : ℝ := x * v₀ + a₅
def v₂ : ℝ := x * v₁ + a₄
def v₃ : ℝ := x * v₂ + a₃

-- Theorem statement
theorem horner_v₃_value : v₃ = -40 := by
  sorry

end NUMINAMATH_CALUDE_horner_v₃_value_l3989_398909


namespace NUMINAMATH_CALUDE_prob_level_b_part1_prob_not_qualifying_part2_l3989_398930

-- Define the probability of success for a single attempt
def p_success : ℚ := 1/2

-- Define the number of attempts for part 1
def attempts_part1 : ℕ := 4

-- Define the number of successes required for level B
def level_b_successes : ℕ := 3

-- Define the maximum number of attempts for part 2
def max_attempts_part2 : ℕ := 5

-- Part 1: Probability of exactly 3 successes in 4 attempts
theorem prob_level_b_part1 :
  (Nat.choose attempts_part1 level_b_successes : ℚ) * p_success^level_b_successes * (1 - p_success)^(attempts_part1 - level_b_successes) = 3/16 := by
  sorry

-- Part 2: Probability of not qualifying as level B or A player
theorem prob_not_qualifying_part2 :
  let seq := List.cons p_success (List.cons p_success (List.cons (1 - p_success) (List.cons (1 - p_success) [])))
  let p_exactly_3 := (Nat.choose 4 2 : ℚ) * p_success^3 * (1 - p_success)^2
  let p_exactly_2 := p_success^2 * (1 - p_success)^2 + 3 * p_success^2 * (1 - p_success)^3
  let p_exactly_1 := p_success * (1 - p_success)^2 + p_success * (1 - p_success)^3
  let p_exactly_0 := (1 - p_success)^2
  p_exactly_3 + p_exactly_2 + p_exactly_1 + p_exactly_0 = 25/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_level_b_part1_prob_not_qualifying_part2_l3989_398930


namespace NUMINAMATH_CALUDE_second_half_revenue_l3989_398913

/-- Represents the ticket categories --/
inductive TicketCategory
  | A
  | B
  | C

/-- Calculates the total revenue from ticket sales --/
def calculate_revenue (tickets : Nat) (price : Nat) : Nat :=
  tickets * price

/-- Represents the ticket sales data for Richmond Tigers --/
structure TicketSalesData where
  total_tickets : Nat
  first_half_total : Nat
  first_half_A : Nat
  first_half_B : Nat
  first_half_C : Nat
  price_A : Nat
  price_B : Nat
  price_C : Nat

/-- Theorem: The total revenue from the second half of the season is $154,510 --/
theorem second_half_revenue (data : TicketSalesData) 
  (h1 : data.total_tickets = 9570)
  (h2 : data.first_half_total = 3867)
  (h3 : data.first_half_A = 1350)
  (h4 : data.first_half_B = 1150)
  (h5 : data.first_half_C = 1367)
  (h6 : data.price_A = 50)
  (h7 : data.price_B = 40)
  (h8 : data.price_C = 30) :
  calculate_revenue data.first_half_A data.price_A + 
  calculate_revenue data.first_half_B data.price_B + 
  calculate_revenue data.first_half_C data.price_C = 154510 := by
  sorry


end NUMINAMATH_CALUDE_second_half_revenue_l3989_398913


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l3989_398938

theorem crazy_silly_school_books (books_read books_unread : ℕ) 
  (h1 : books_read = 13) 
  (h2 : books_unread = 8) : 
  books_read + books_unread = 21 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l3989_398938


namespace NUMINAMATH_CALUDE_cube_net_opposite_face_l3989_398925

-- Define the faces of the cube
inductive Face : Type
  | W | X | Y | Z | V | z

-- Define the concept of opposite faces
def opposite (f1 f2 : Face) : Prop := sorry

-- Define the concept of adjacent faces in the net
def adjacent_in_net (f1 f2 : Face) : Prop := sorry

-- Define the concept of a valid cube net
def valid_cube_net (net : List Face) : Prop := sorry

-- Theorem statement
theorem cube_net_opposite_face (net : List Face) 
  (h_valid : valid_cube_net net)
  (h_z_central : adjacent_in_net Face.z Face.W ∧ 
                 adjacent_in_net Face.z Face.X ∧ 
                 adjacent_in_net Face.z Face.Y)
  (h_v_not_adjacent : ¬adjacent_in_net Face.z Face.V) :
  opposite Face.z Face.V := by sorry

end NUMINAMATH_CALUDE_cube_net_opposite_face_l3989_398925


namespace NUMINAMATH_CALUDE_next_two_juicy_numbers_l3989_398903

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * b * c * d = n ∧ 1 = 1/a + 1/b + 1/c + 1/d

theorem next_two_juicy_numbers :
  (∀ n < 6, ¬ is_juicy n) ∧
  is_juicy 6 ∧
  is_juicy 12 ∧
  is_juicy 20 ∧
  (∀ n, 6 < n ∧ n < 12 → ¬ is_juicy n) ∧
  (∀ n, 12 < n ∧ n < 20 → ¬ is_juicy n) :=
sorry

end NUMINAMATH_CALUDE_next_two_juicy_numbers_l3989_398903


namespace NUMINAMATH_CALUDE_school_fundraiser_distribution_l3989_398989

theorem school_fundraiser_distribution (total_amount : ℚ) (num_charities : ℕ) 
  (h1 : total_amount = 3109)
  (h2 : num_charities = 25) :
  total_amount / num_charities = 124.36 := by
  sorry

end NUMINAMATH_CALUDE_school_fundraiser_distribution_l3989_398989


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3989_398924

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3989_398924


namespace NUMINAMATH_CALUDE_fortieth_term_is_210_l3989_398933

/-- A function that checks if a number contains the digit 2 --/
def containsTwo (n : ℕ) : Bool :=
  sorry

/-- A function that generates the sequence of positive multiples of 3 containing at least one digit 2 --/
def sequenceGenerator : ℕ → ℕ :=
  sorry

/-- The theorem stating that the 40th term of the sequence is 210 --/
theorem fortieth_term_is_210 : sequenceGenerator 40 = 210 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_term_is_210_l3989_398933


namespace NUMINAMATH_CALUDE_new_rectangle_area_l3989_398912

/-- Given a rectangle with sides a and b, prove the area of a new rectangle constructed from it. -/
theorem new_rectangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l3989_398912


namespace NUMINAMATH_CALUDE_percentage_relation_l3989_398918

theorem percentage_relation (A B C : ℝ) (h1 : A = 0.07 * C) (h2 : A = 0.5 * B) :
  B = 0.14 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3989_398918


namespace NUMINAMATH_CALUDE_min_hypotenuse_max_inscribed_circle_radius_l3989_398956

/-- A right-angled triangle with perimeter 1 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  perimeter_eq_one : a + b + hypotenuse = 1
  right_angle : a^2 + b^2 = hypotenuse^2
  positive : 0 < a ∧ 0 < b

/-- The minimum length of the hypotenuse in a right-angled triangle with perimeter 1 -/
theorem min_hypotenuse (t : RightTriangle) : t.hypotenuse ≥ Real.sqrt 2 - 1 := by
  sorry

/-- The maximum radius of the inscribed circle in a right-angled triangle with perimeter 1 -/
theorem max_inscribed_circle_radius (t : RightTriangle) : 
  t.a * t.b / (t.a + t.b + t.hypotenuse) ≤ 3/2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_hypotenuse_max_inscribed_circle_radius_l3989_398956


namespace NUMINAMATH_CALUDE_expression_simplification_l3989_398999

theorem expression_simplification (x y a b c : ℝ) :
  (2 - y) * 24 * (x - y) + 2 * ((a - 2 - 3 * c) * a - 2 * b + c) = 2 + 4 * b^2 - a * b - c^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3989_398999


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l3989_398914

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 2800) 
  (h2 : males_below_50 = 490) : 
  (males_below_50 : ℝ) / ((70 : ℝ) / 100 * total_employees) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l3989_398914


namespace NUMINAMATH_CALUDE_total_earnings_is_4350_l3989_398980

/-- Given investment ratios and return ratios for three investors a, b, and c,
    calculates their total earnings. -/
def total_earnings (invest_a invest_b invest_c : ℚ)
                   (return_a return_b return_c : ℚ)
                   (diff_b_a : ℚ) : ℚ :=
  let earnings_a := invest_a * return_a
  let earnings_b := invest_b * return_b
  let earnings_c := invest_c * return_c
  earnings_a + earnings_b + earnings_c

/-- Theorem stating that under given conditions, the total earnings are 4350. -/
theorem total_earnings_is_4350 :
  ∃ (x y : ℚ),
    let invest_a := 3 * x
    let invest_b := 4 * x
    let invest_c := 5 * x
    let return_a := 6 * y
    let return_b := 5 * y
    let return_c := 4 * y
    invest_b * return_b - invest_a * return_a = 150 ∧
    total_earnings invest_a invest_b invest_c return_a return_b return_c 150 = 4350 :=
by
  sorry

#check total_earnings_is_4350

end NUMINAMATH_CALUDE_total_earnings_is_4350_l3989_398980


namespace NUMINAMATH_CALUDE_disney_banquet_revenue_l3989_398931

/-- Calculates the total revenue from ticket sales for a Disney banquet --/
theorem disney_banquet_revenue :
  let total_attendees : ℕ := 586
  let resident_price : ℚ := 12.95
  let non_resident_price : ℚ := 17.95
  let num_residents : ℕ := 219
  let num_non_residents : ℕ := total_attendees - num_residents
  let resident_revenue : ℚ := num_residents * resident_price
  let non_resident_revenue : ℚ := num_non_residents * non_resident_price
  let total_revenue : ℚ := resident_revenue + non_resident_revenue
  total_revenue = 9423.70 := by
  sorry

end NUMINAMATH_CALUDE_disney_banquet_revenue_l3989_398931


namespace NUMINAMATH_CALUDE_min_comparisons_for_max_l3989_398969

/-- Represents a list of n pairwise distinct numbers -/
def DistinctNumbers (n : ℕ) := { l : List ℝ // l.length = n ∧ l.Pairwise (· ≠ ·) }

/-- Represents a comparison between two numbers -/
def Comparison := ℝ × ℝ

/-- A function that finds the maximum number in a list using pairwise comparisons -/
def FindMax (n : ℕ) (numbers : DistinctNumbers n) : 
  { comparisons : List Comparison // comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max } :=
sorry

theorem min_comparisons_for_max (n : ℕ) (numbers : DistinctNumbers n) :
  (∀ comparisons : List Comparison, 
    (∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) → 
    comparisons.length ≥ n - 1) ∧
  (∃ comparisons : List Comparison, 
    comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) :=
sorry

end NUMINAMATH_CALUDE_min_comparisons_for_max_l3989_398969


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3989_398900

theorem greatest_integer_for_all_real_domain (a : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 15 ≠ 0)) ↔ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3989_398900


namespace NUMINAMATH_CALUDE_clara_loses_prob_l3989_398986

/-- The probability of Clara's coin landing heads -/
def clara_heads_prob : ℚ := 2/3

/-- The probability of Ethan's coin landing heads -/
def ethan_heads_prob : ℚ := 1/4

/-- The probability of both Clara and Ethan getting tails in one round -/
def both_tails_prob : ℚ := (1 - clara_heads_prob) * (1 - ethan_heads_prob)

/-- The game where Clara and Ethan alternately toss coins until one gets a head and loses -/
def coin_toss_game : Prop :=
  ∃ (p : ℚ), p = clara_heads_prob * (1 / (1 - both_tails_prob))

/-- The theorem stating that the probability of Clara losing is 8/9 -/
theorem clara_loses_prob : 
  coin_toss_game → (∃ (p : ℚ), p = 8/9 ∧ p = clara_heads_prob * (1 / (1 - both_tails_prob))) :=
by sorry

end NUMINAMATH_CALUDE_clara_loses_prob_l3989_398986


namespace NUMINAMATH_CALUDE_apple_cost_for_two_weeks_l3989_398951

/-- Represents the cost of apples for Irene and her dog for 2 weeks -/
def appleCost (daysPerWeek : ℕ) (weeks : ℕ) (appleWeight : ℚ) (pricePerPound : ℚ) : ℚ :=
  let totalDays : ℕ := daysPerWeek * weeks
  let totalApples : ℕ := totalDays
  let totalWeight : ℚ := appleWeight * totalApples
  totalWeight * pricePerPound

/-- Theorem stating that the cost of apples for 2 weeks is $7.00 -/
theorem apple_cost_for_two_weeks :
  appleCost 7 2 (1/4) 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_apple_cost_for_two_weeks_l3989_398951


namespace NUMINAMATH_CALUDE_total_pools_calculation_l3989_398995

/-- The number of Pat's Pool Supply stores -/
def pool_supply_stores : ℕ := 4

/-- The number of Pat's Ark & Athletic Wear stores -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 5

/-- The initial number of pools at one Pat's Ark & Athletic Wear store -/
def initial_pools : ℕ := 200

/-- The number of pools sold at one Pat's Ark & Athletic Wear store -/
def pools_sold : ℕ := 8

/-- The number of pools returned to one Pat's Ark & Athletic Wear store -/
def pools_returned : ℕ := 3

/-- The total number of swimming pools across all stores -/
def total_pools : ℕ := 5070

theorem total_pools_calculation :
  let current_pools := initial_pools - pools_sold + pools_returned
  let supply_store_pools := pool_ratio * current_pools
  total_pools = ark_athletic_stores * current_pools + pool_supply_stores * supply_store_pools := by
  sorry

end NUMINAMATH_CALUDE_total_pools_calculation_l3989_398995


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_property_l3989_398934

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_property_l3989_398934


namespace NUMINAMATH_CALUDE_symmetric_point_in_first_quadrant_l3989_398988

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Symmetry about the x-axis -/
def symmetric_about_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P -/
def P : Point :=
  ⟨2, -3⟩

theorem symmetric_point_in_first_quadrant :
  is_in_first_quadrant (symmetric_about_x_axis P) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_in_first_quadrant_l3989_398988


namespace NUMINAMATH_CALUDE_expression_evaluation_l3989_398983

theorem expression_evaluation : 
  let c : ℕ := 4
  (c^c - 2*c*(c-2)^c + c^2)^c = 431441456 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3989_398983


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3989_398964

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 →                  -- One leg measures 8 meters
  (1/2) * a * b = 48 →     -- Area is 48 square meters
  a^2 + b^2 = c^2 →        -- Pythagorean theorem for right triangle
  c = 4 * Real.sqrt 13 :=  -- Hypotenuse length is 4√13 meters
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3989_398964


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3989_398928

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_prod : a 5 * a 11 = 3)
  (h_sum : a 3 + a 13 = 4) :
  ∃ r : ℝ, (r = 3 ∨ r = -3) ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3989_398928


namespace NUMINAMATH_CALUDE_normal_prob_theorem_l3989_398911

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The probability density function of a normal distribution -/
def normal_pdf (X : NormalRV) (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
def normal_cdf (X : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable is in a given interval -/
def prob_interval (X : NormalRV) (a b : ℝ) : ℝ := normal_cdf X b - normal_cdf X a

theorem normal_prob_theorem (X : NormalRV) (h : X.μ = 2) :
  prob_interval X 1 3 = 0.4 → normal_cdf X 1 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_prob_theorem_l3989_398911


namespace NUMINAMATH_CALUDE_expression_evaluation_l3989_398922

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3989_398922


namespace NUMINAMATH_CALUDE_probability_two_A_grades_l3989_398976

/-- The probability of achieving an A grade in exactly two out of three subjects. -/
theorem probability_two_A_grades
  (p_politics : ℝ)
  (p_history : ℝ)
  (p_geography : ℝ)
  (hp_politics : p_politics = 4/5)
  (hp_history : p_history = 3/5)
  (hp_geography : p_geography = 2/5)
  (hprob_politics : 0 ≤ p_politics ∧ p_politics ≤ 1)
  (hprob_history : 0 ≤ p_history ∧ p_history ≤ 1)
  (hprob_geography : 0 ≤ p_geography ∧ p_geography ≤ 1) :
  p_politics * p_history * (1 - p_geography) +
  p_politics * (1 - p_history) * p_geography +
  (1 - p_politics) * p_history * p_geography = 58/125 := by
sorry

end NUMINAMATH_CALUDE_probability_two_A_grades_l3989_398976


namespace NUMINAMATH_CALUDE_grocery_store_costs_l3989_398904

/-- Calculates the money paid for orders given total costs and fractions for salary and delivery --/
def money_paid_for_orders (total_costs : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) : ℝ :=
  let salary := salary_fraction * total_costs
  let remaining := total_costs - salary
  let delivery := delivery_fraction * remaining
  total_costs - salary - delivery

/-- Proves that given the specified conditions, the money paid for orders is $1800 --/
theorem grocery_store_costs : 
  money_paid_for_orders 4000 (2/5) (1/4) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l3989_398904


namespace NUMINAMATH_CALUDE_second_group_size_l3989_398955

/-- Represents a choir split into three groups -/
structure Choir :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (sum_eq_total : group1 + group2 + group3 = total)

/-- Theorem: Given a choir with 70 total members, 25 in the first group,
    and 15 in the third group, the second group must have 30 members -/
theorem second_group_size (c : Choir)
  (h1 : c.total = 70)
  (h2 : c.group1 = 25)
  (h3 : c.group3 = 15) :
  c.group2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l3989_398955


namespace NUMINAMATH_CALUDE_line_contains_point_l3989_398997

/-- Proves that for the line equation 3 - kx = -4y, if the point (3, -2) lies on the line, then k = -5/3 -/
theorem line_contains_point (k : ℚ) : 
  (3 - k * 3 = -4 * (-2)) → k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l3989_398997


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3989_398973

theorem quadratic_one_solution (q : ℝ) (h : q ≠ 0) : 
  (q = 64/9) ↔ (∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3989_398973


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3989_398917

theorem average_of_remaining_numbers
  (n : ℕ)
  (total_avg : ℚ)
  (first_three_avg : ℚ)
  (next_three_avg : ℚ)
  (h1 : n = 8)
  (h2 : total_avg = 4.5)
  (h3 : first_three_avg = 5.2)
  (h4 : next_three_avg = 3.6) :
  (n * total_avg - 3 * first_three_avg - 3 * next_three_avg) / 2 = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3989_398917


namespace NUMINAMATH_CALUDE_power_relation_l3989_398923

theorem power_relation (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 9) : x^(3*a - b) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3989_398923


namespace NUMINAMATH_CALUDE_three_year_officer_pays_51_l3989_398991

/-- The price of duty shoes for an officer who has served at least three years -/
def price_for_three_year_officer : ℝ :=
  let full_price : ℝ := 85
  let first_year_discount : ℝ := 0.20
  let three_year_discount : ℝ := 0.25
  let discounted_price : ℝ := full_price * (1 - first_year_discount)
  discounted_price * (1 - three_year_discount)

/-- Theorem stating that an officer who has served at least three years pays $51 for duty shoes -/
theorem three_year_officer_pays_51 :
  price_for_three_year_officer = 51 := by
  sorry

end NUMINAMATH_CALUDE_three_year_officer_pays_51_l3989_398991


namespace NUMINAMATH_CALUDE_jamie_father_burns_500_calories_l3989_398920

/-- The number of calories in a pound of body fat -/
def calories_per_pound : ℕ := 3500

/-- The number of pounds Jamie's father wants to lose -/
def pounds_to_lose : ℕ := 5

/-- The number of days it takes Jamie's father to burn off the weight -/
def days_to_burn : ℕ := 35

/-- The number of calories Jamie's father eats per day -/
def calories_eaten_daily : ℕ := 2000

/-- The number of calories Jamie's father burns daily through light exercise -/
def calories_burned_daily : ℕ := (pounds_to_lose * calories_per_pound) / days_to_burn

theorem jamie_father_burns_500_calories :
  calories_burned_daily = 500 :=
sorry

end NUMINAMATH_CALUDE_jamie_father_burns_500_calories_l3989_398920


namespace NUMINAMATH_CALUDE_parallelogram_height_l3989_398937

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 416)
  (h_base : base = 26)
  (h_formula : area = base * height) : 
  height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3989_398937


namespace NUMINAMATH_CALUDE_exists_common_divisor_l3989_398901

/-- A function from positive integers to positive integers greater than 1 -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The theorem stating that there exists a common divisor greater than 1 for all values of f -/
theorem exists_common_divisor (f : PositiveIntegerFunction) 
  (h : HasDivisibilityProperty f) : 
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n := by
  sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l3989_398901


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3989_398992

theorem ice_cream_combinations :
  (5 : ℕ) * (Nat.choose 7 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3989_398992


namespace NUMINAMATH_CALUDE_combined_ratio_theorem_l3989_398941

/-- Represents the ratio of liquids in a vessel -/
structure LiquidRatio :=
  (water : ℚ)
  (milk : ℚ)
  (syrup : ℚ)

/-- Represents a vessel with its volume and liquid ratio -/
structure Vessel :=
  (volume : ℚ)
  (ratio : LiquidRatio)

def combine_vessels (vessels : List Vessel) : LiquidRatio :=
  let total_water := vessels.map (λ v => v.volume * v.ratio.water) |>.sum
  let total_milk := vessels.map (λ v => v.volume * v.ratio.milk) |>.sum
  let total_syrup := vessels.map (λ v => v.volume * v.ratio.syrup) |>.sum
  { water := total_water, milk := total_milk, syrup := total_syrup }

theorem combined_ratio_theorem (v1 v2 v3 : Vessel)
  (h1 : v1.volume = 3 ∧ v2.volume = 5 ∧ v3.volume = 7)
  (h2 : v1.ratio = { water := 1/6, milk := 1/3, syrup := 1/2 })
  (h3 : v2.ratio = { water := 2/7, milk := 4/7, syrup := 1/7 })
  (h4 : v3.ratio = { water := 1/2, milk := 1/6, syrup := 1/3 }) :
  let combined := combine_vessels [v1, v2, v3]
  combined.water / (combined.water + combined.milk + combined.syrup) = 228 / 630 ∧
  combined.milk / (combined.water + combined.milk + combined.syrup) = 211 / 630 ∧
  combined.syrup / (combined.water + combined.milk + combined.syrup) = 191 / 630 := by
  sorry

#check combined_ratio_theorem

end NUMINAMATH_CALUDE_combined_ratio_theorem_l3989_398941


namespace NUMINAMATH_CALUDE_josette_purchase_cost_l3989_398947

/-- Calculates the total cost of mineral water bottles with a discount --/
def total_cost (small_count : ℕ) (large_count : ℕ) (small_price : ℚ) (large_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_count := small_count + large_count
  let subtotal := small_count * small_price + large_count * large_price
  if total_count ≥ 5 then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- The total cost for Josette's purchase is €8.37 --/
theorem josette_purchase_cost :
  total_cost 3 2 (3/2) (12/5) (1/10) = 837/100 := by sorry

end NUMINAMATH_CALUDE_josette_purchase_cost_l3989_398947


namespace NUMINAMATH_CALUDE_permutations_of_six_l3989_398957

theorem permutations_of_six (n : Nat) : n = 6 → Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_l3989_398957


namespace NUMINAMATH_CALUDE_middle_part_value_l3989_398910

theorem middle_part_value (total : ℚ) (r1 r2 r3 : ℚ) (p1 p2 p3 : ℚ) : 
  total = 120 →
  r1 = (1 : ℚ) / 2 →
  r2 = (1 : ℚ) / 4 →
  r3 = (1 : ℚ) / 8 →
  p1 + p2 + p3 = total →
  p1 / r1 = p2 / r2 →
  p2 / r2 = p3 / r3 →
  p2 = 480 / 14 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_value_l3989_398910


namespace NUMINAMATH_CALUDE_function_monotonicity_l3989_398984

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3989_398984


namespace NUMINAMATH_CALUDE_only_two_digit_divisor_with_remainder_four_l3989_398936

theorem only_two_digit_divisor_with_remainder_four (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ 143 % d = 4 → d = 139 :=
by sorry

end NUMINAMATH_CALUDE_only_two_digit_divisor_with_remainder_four_l3989_398936


namespace NUMINAMATH_CALUDE_alyssa_fruit_expenses_l3989_398975

theorem alyssa_fruit_expenses : 
  let grapes_cost : ℚ := 12.08
  let cherries_cost : ℚ := 9.85
  grapes_cost + cherries_cost = 21.93 := by sorry

end NUMINAMATH_CALUDE_alyssa_fruit_expenses_l3989_398975


namespace NUMINAMATH_CALUDE_system_solution_l3989_398929

theorem system_solution (a b c d : ℚ) 
  (eq1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * (d + c) = b)
  (eq3 : 4 * b + 2 * c = a)
  (eq4 : c + 1 = d) :
  a + b + c + d = 513 / 37 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3989_398929


namespace NUMINAMATH_CALUDE_app_cost_is_four_l3989_398954

/-- The average cost of an app given the total budget, remaining amount, and number of apps. -/
def average_app_cost (total_budget : ℚ) (remaining : ℚ) (num_apps : ℕ) : ℚ :=
  (total_budget - remaining) / num_apps

/-- Theorem stating that the average cost of an app is $4 given the problem conditions. -/
theorem app_cost_is_four :
  let total_budget : ℚ := 66
  let remaining : ℚ := 6
  let num_apps : ℕ := 15
  average_app_cost total_budget remaining num_apps = 4 := by
  sorry

end NUMINAMATH_CALUDE_app_cost_is_four_l3989_398954


namespace NUMINAMATH_CALUDE_max_blue_points_l3989_398949

theorem max_blue_points (total_spheres : ℕ) (h : total_spheres = 2016) :
  ∃ (red_spheres : ℕ), 
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : ℕ), x ≤ total_spheres → 
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_blue_points_l3989_398949


namespace NUMINAMATH_CALUDE_books_sold_on_wednesday_l3989_398994

theorem books_sold_on_wednesday 
  (initial_stock : ℕ) 
  (sold_monday : ℕ) 
  (sold_tuesday : ℕ) 
  (sold_thursday : ℕ) 
  (sold_friday : ℕ) 
  (unsold : ℕ) 
  (h1 : initial_stock = 800)
  (h2 : sold_monday = 60)
  (h3 : sold_tuesday = 10)
  (h4 : sold_thursday = 44)
  (h5 : sold_friday = 66)
  (h6 : unsold = 600) :
  initial_stock - unsold - (sold_monday + sold_tuesday + sold_thursday + sold_friday) = 20 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_on_wednesday_l3989_398994


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l3989_398987

/-- The cost of the cheaper rice variety in Rs per kg -/
def cost_cheap : ℚ := 9/2

/-- The cost of the more expensive rice variety in Rs per kg -/
def cost_expensive : ℚ := 35/4

/-- The ratio of cheaper rice to more expensive rice in the mixture -/
def mixture_ratio : ℚ := 5/12

/-- The cost of the mixture per kg -/
def mixture_cost : ℚ := 23/4

theorem rice_mixture_cost :
  (cost_cheap * mixture_ratio + cost_expensive * 1) / (mixture_ratio + 1) = mixture_cost := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l3989_398987


namespace NUMINAMATH_CALUDE_floor_length_percentage_more_than_breadth_l3989_398953

theorem floor_length_percentage_more_than_breadth 
  (length : Real) 
  (area : Real) 
  (h1 : length = 13.416407864998739)
  (h2 : area = 60) :
  let breadth := area / length
  (length - breadth) / breadth * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_percentage_more_than_breadth_l3989_398953


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l3989_398935

/-- Proves that mixing 28 ounces of 40% salt solution with 112 ounces of 90% salt solution
    results in a 140-ounce mixture that is 80% salt -/
theorem salt_mixture_proof :
  let solution_a_amount : ℝ := 28
  let solution_b_amount : ℝ := 112
  let solution_a_concentration : ℝ := 0.4
  let solution_b_concentration : ℝ := 0.9
  let total_amount : ℝ := solution_a_amount + solution_b_amount
  let target_concentration : ℝ := 0.8
  let mixture_salt_amount : ℝ := solution_a_amount * solution_a_concentration +
                                  solution_b_amount * solution_b_concentration
  (total_amount = 140) ∧
  (mixture_salt_amount / total_amount = target_concentration) :=
by
  sorry


end NUMINAMATH_CALUDE_salt_mixture_proof_l3989_398935


namespace NUMINAMATH_CALUDE_book_club_combinations_l3989_398942

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the book club --/
def total_people : ℕ := 5

/-- The number of people who lead the discussion --/
def discussion_leaders : ℕ := 3

theorem book_club_combinations :
  choose total_people discussion_leaders = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_club_combinations_l3989_398942


namespace NUMINAMATH_CALUDE_possible_r_value_l3989_398959

-- Define sets A and B
def A (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * (p.1 - 1) + p.2 * (p.2 - 1) ≤ r}

def B (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ r^2}

-- Theorem statement
theorem possible_r_value :
  ∃ (r : ℝ), (A r ⊆ B r) ∧ (r = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_possible_r_value_l3989_398959


namespace NUMINAMATH_CALUDE_quadratic_transform_coefficient_l3989_398966

/-- Given a quadratic equation 7x - 3 = 2x², prove that when transformed
    to general form ax² + bx + c = 0 with c = 3, the coefficient of x (b) is -7 -/
theorem quadratic_transform_coefficient (x : ℝ) : 
  (7 * x - 3 = 2 * x^2) → 
  ∃ (a b : ℝ), (a * x^2 + b * x + 3 = 0) ∧ (b = -7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_transform_coefficient_l3989_398966


namespace NUMINAMATH_CALUDE_inradius_right_triangle_l3989_398962

/-- The inradius of a right triangle with side lengths 9, 40, and 41 is 4. -/
theorem inradius_right_triangle : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_inradius_right_triangle_l3989_398962


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_l3989_398948

theorem simplify_and_evaluate_1 (m : ℤ) :
  m = -2023 → 7 * m^2 + 4 - 2 * m^2 - 3 * m - 5 * m^2 - 5 + 4 * m = -2024 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_l3989_398948


namespace NUMINAMATH_CALUDE_toms_total_amount_l3989_398981

/-- Tom's initial amount in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def earned_amount : ℕ := 86

/-- Theorem stating Tom's total amount after washing cars -/
theorem toms_total_amount : initial_amount + earned_amount = 160 := by
  sorry

end NUMINAMATH_CALUDE_toms_total_amount_l3989_398981


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3989_398919

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (a-1)*x + a*y + 1 = 0) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ + 2*a*y₁ - 1 = 0 ∧ (a-1)*x₂ + a*y₂ + 1 = 0 ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3989_398919


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3989_398977

/-- 
Given a rectangular box with face areas of 36, 18, and 8 square inches,
prove that its volume is 72 cubic inches.
-/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3989_398977


namespace NUMINAMATH_CALUDE_genuine_coin_remains_l3989_398979

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Fake

/-- Represents the state of the coin selection process -/
structure CoinState where
  total : Nat
  genuine : Nat
  fake : Nat
  moves : Nat

/-- The initial state of coins -/
def initialState : CoinState :=
  { total := 2022
  , genuine := 1012  -- More than half of 2022
  , fake := 1010     -- Less than half of 2022
  , moves := 0 }

/-- Simulates a single move in the coin selection process -/
def move (state : CoinState) : CoinState :=
  { state with
    total := state.total - 1
    moves := state.moves + 1
    genuine := state.genuine - 1  -- Worst case: remove a genuine coin
  }

/-- Applies the move function n times -/
def applyMoves (n : Nat) (state : CoinState) : CoinState :=
  match n with
  | 0 => state
  | n + 1 => move (applyMoves n state)

theorem genuine_coin_remains : 
  (applyMoves 2021 initialState).genuine > 0 := by
  sorry

#check genuine_coin_remains

end NUMINAMATH_CALUDE_genuine_coin_remains_l3989_398979


namespace NUMINAMATH_CALUDE_brick_width_is_10cm_l3989_398990

/-- Proves that the width of each brick is 10 centimeters, given the courtyard dimensions,
    brick length, and total number of bricks. -/
theorem brick_width_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 18) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 14400) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width * 100 * 100 = 
    brick_length * brick_width * total_bricks * 10000 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_10cm_l3989_398990


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l3989_398958

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 0.50 * (x - 2 : ℝ)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧
    constant = 1.50 := by sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l3989_398958


namespace NUMINAMATH_CALUDE_journey_mpg_l3989_398971

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer : ℕ) (final_odometer : ℕ) (odometer_error : ℕ) 
                (initial_fuel : ℕ) (refill1 : ℕ) (refill2 : ℕ) : ℚ :=
  let actual_distance := (final_odometer - odometer_error) - initial_odometer
  let total_fuel := initial_fuel + refill1 + refill2
  (actual_distance : ℚ) / total_fuel

/-- Theorem stating that the average miles per gallon for the given journey is 20.8 -/
theorem journey_mpg : 
  let mpg := average_mpg 68300 69350 10 10 15 25
  ∃ (n : ℕ), (n : ℚ) / 10 = mpg ∧ n = 208 :=
by sorry

end NUMINAMATH_CALUDE_journey_mpg_l3989_398971


namespace NUMINAMATH_CALUDE_zero_exponent_l3989_398939

theorem zero_exponent (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l3989_398939


namespace NUMINAMATH_CALUDE_find_x_l3989_398974

theorem find_x : ∃ x : ℝ, 
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3989_398974


namespace NUMINAMATH_CALUDE_triangle_third_angle_l3989_398915

theorem triangle_third_angle (a b : ℝ) (ha : a = 37) (hb : b = 75) :
  180 - a - b = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l3989_398915


namespace NUMINAMATH_CALUDE_buffer_water_requirement_l3989_398982

/-- Given a buffer solution where water constitutes 1/3 of the total volume,
    prove that 0.72 liters of the buffer solution requires 0.24 liters of water. -/
theorem buffer_water_requirement (total_volume : ℝ) (water_fraction : ℝ) 
    (h1 : total_volume = 0.72)
    (h2 : water_fraction = 1/3) : 
  total_volume * water_fraction = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_buffer_water_requirement_l3989_398982


namespace NUMINAMATH_CALUDE_circle_alignment_exists_l3989_398943

/-- Represents a circle with a circumference of 100 cm -/
structure Circle :=
  (circumference : ℝ)
  (h_circumference : circumference = 100)

/-- Represents a set of marked points on a circle -/
structure MarkedPoints :=
  (circle : Circle)
  (num_points : ℕ)

/-- Represents a set of arcs on a circle -/
structure Arcs :=
  (circle : Circle)
  (total_length : ℝ)
  (h_length : total_length < 1)

/-- Represents an alignment of two circles -/
def Alignment := ℝ

/-- Checks if a marked point coincides with an arc for a given alignment -/
def coincides (mp : MarkedPoints) (a : Arcs) (alignment : Alignment) : Prop :=
  sorry

theorem circle_alignment_exists (c1 c2 : Circle) 
  (mp : MarkedPoints) (a : Arcs) 
  (h_mp : mp.circle = c1) (h_a : a.circle = c2) 
  (h_num_points : mp.num_points = 100) :
  ∃ (alignment : Alignment), ∀ (p : ℕ) (h_p : p < mp.num_points), 
    ¬ coincides mp a alignment :=
sorry

end NUMINAMATH_CALUDE_circle_alignment_exists_l3989_398943


namespace NUMINAMATH_CALUDE_ben_remaining_money_l3989_398927

def calculate_remaining_money (initial amount : ℕ) (cheque debtor_payment maintenance_cost : ℕ) : ℕ :=
  initial - cheque + debtor_payment - maintenance_cost

theorem ben_remaining_money :
  calculate_remaining_money 2000 600 800 1200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ben_remaining_money_l3989_398927


namespace NUMINAMATH_CALUDE_sqrt_39_equals_33_l3989_398961

theorem sqrt_39_equals_33 : Real.sqrt 39 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_39_equals_33_l3989_398961


namespace NUMINAMATH_CALUDE_speed_calculation_l3989_398998

/-- Given a distance of 3.0 miles and a time of 1.5 hours, prove that the speed is 2.0 miles per hour. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 3.0) 
    (h2 : time = 1.5) 
    (h3 : speed = distance / time) : speed = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l3989_398998


namespace NUMINAMATH_CALUDE_annual_increase_fraction_l3989_398946

theorem annual_increase_fraction (initial_amount final_amount : ℝ) (f : ℝ) 
  (h1 : initial_amount = 65000)
  (h2 : final_amount = 82265.625)
  (h3 : final_amount = initial_amount * (1 + f)^2) :
  f = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_increase_fraction_l3989_398946


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3989_398965

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 - b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) :
  (a + b + c = 0) ∧ (a < 0) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3989_398965


namespace NUMINAMATH_CALUDE_rps_win_on_sixth_game_l3989_398972

/-- The probability of a tie in a single game of Rock Paper Scissors -/
def tie_prob : ℚ := 1 / 3

/-- The probability of not tying (i.e., someone wins) in a single game -/
def win_prob : ℚ := 1 - tie_prob

/-- The number of consecutive ties before the winning game -/
def num_ties : ℕ := 5

theorem rps_win_on_sixth_game : 
  tie_prob ^ num_ties * win_prob = 2 / 729 := by sorry

end NUMINAMATH_CALUDE_rps_win_on_sixth_game_l3989_398972


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3989_398963

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_under_500 : 
  (∀ n : ℕ, n < 500 → is_mersenne_prime n → n ≤ 127) ∧ 
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3989_398963


namespace NUMINAMATH_CALUDE_train_speed_l3989_398932

/-- Calculates the speed of a train in km/hr given its length and time to pass a tree -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 11) :
  (length / time) * 3.6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3989_398932


namespace NUMINAMATH_CALUDE_monomial_properties_l3989_398996

-- Define the structure of a monomial
structure Monomial (α : Type*) [Field α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

-- Define the given monomial
def given_monomial : Monomial ℚ := {
  coeff := -1/7,
  x_exp := 2,
  y_exp := 1
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

-- Define the degree of a monomial
def degree (m : Monomial ℚ) : ℕ := m.x_exp + m.y_exp

-- Theorem statement
theorem monomial_properties :
  coefficient given_monomial = -1/7 ∧ degree given_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3989_398996


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_five_halves_l3989_398960

theorem sum_of_solutions_eq_five_halves :
  let f : ℝ → ℝ := λ x => (4*x + 6)*(3*x - 12)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_five_halves_l3989_398960


namespace NUMINAMATH_CALUDE_remainder_problem_l3989_398978

theorem remainder_problem (a b : ℤ) 
  (ha : a % 98 = 92) 
  (hb : b % 147 = 135) : 
  (3 * a + b) % 49 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3989_398978


namespace NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3989_398993

/-- Given a triangle with perimeter p, the perimeter of the triangle formed by 
    connecting the midpoints of its sides is p/2. -/
theorem midpoint_triangle_perimeter (p : ℝ) (h : p > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧ 
  (a/2 + b/2 + c/2) = p/2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3989_398993


namespace NUMINAMATH_CALUDE_keanu_destination_distance_l3989_398967

/-- Represents the distance to Keanu's destination -/
def destination_distance : ℝ := 280

/-- Represents the capacity of Keanu's motorcycle's gas tank in liters -/
def tank_capacity : ℝ := 8

/-- Represents the distance Keanu's motorcycle can travel with one full tank in miles -/
def miles_per_tank : ℝ := 40

/-- Represents the number of times Keanu refills his motorcycle for a round trip -/
def refills : ℕ := 14

/-- Theorem stating that the distance to Keanu's destination is 280 miles -/
theorem keanu_destination_distance :
  destination_distance = (refills : ℝ) * miles_per_tank / 2 :=
sorry

end NUMINAMATH_CALUDE_keanu_destination_distance_l3989_398967


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l3989_398940

theorem arithmetic_square_root_of_nine :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 9 ∧ (∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l3989_398940


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l3989_398926

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 3 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_zero_l3989_398926
