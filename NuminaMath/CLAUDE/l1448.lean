import Mathlib

namespace NUMINAMATH_CALUDE_add_7777_seconds_to_11pm_l1448_144838

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a 12-hour time (with PM indicator) to 24-hour format -/
def to24Hour (hours : Nat) (isPM : Bool) : Nat :=
  sorry

theorem add_7777_seconds_to_11pm :
  let startTime := Time.mk (to24Hour 11 true) 0 0
  let endTime := addSeconds startTime 7777
  endTime = Time.mk 1 9 37 :=
sorry

end NUMINAMATH_CALUDE_add_7777_seconds_to_11pm_l1448_144838


namespace NUMINAMATH_CALUDE_total_installments_count_l1448_144893

/-- Proves that the total number of installments is 52 given the specified payment conditions -/
theorem total_installments_count (first_25_payment : ℝ) (remaining_payment : ℝ) (average_payment : ℝ) :
  first_25_payment = 500 →
  remaining_payment = 600 →
  average_payment = 551.9230769230769 →
  ∃ n : ℕ, n = 52 ∧ 
    n * average_payment = 25 * first_25_payment + (n - 25) * remaining_payment :=
by sorry

end NUMINAMATH_CALUDE_total_installments_count_l1448_144893


namespace NUMINAMATH_CALUDE_stating_lunch_potatoes_count_l1448_144896

/-- Represents the number of potatoes used for different purposes -/
structure PotatoUsage where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- 
Theorem stating that given a total of 7 potatoes and 2 used for dinner,
the number of potatoes used for lunch must be 5.
-/
theorem lunch_potatoes_count (usage : PotatoUsage) 
    (h1 : usage.total = 7)
    (h2 : usage.dinner = 2)
    (h3 : usage.total = usage.lunch + usage.dinner) : 
  usage.lunch = 5 := by
  sorry

end NUMINAMATH_CALUDE_stating_lunch_potatoes_count_l1448_144896


namespace NUMINAMATH_CALUDE_bacteria_growth_l1448_144879

/-- The factor by which the bacteria population increases each minute -/
def growth_factor : ℕ := 2

/-- The number of minutes that pass -/
def time : ℕ := 4

/-- The function that calculates the population after n minutes -/
def population (n : ℕ) : ℕ := growth_factor ^ n

/-- Theorem stating that after 4 minutes, the population is 16 times the original -/
theorem bacteria_growth :
  population time = 16 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1448_144879


namespace NUMINAMATH_CALUDE_money_division_l1448_144871

/-- Given an amount of money divided between A and B in the ratio 1:2, where A receives $200,
    prove that the total amount to be divided is $600. -/
theorem money_division (a b total : ℕ) : 
  (a : ℚ) / b = 1 / 2 →  -- The ratio of A's share to B's share is 1:2
  a = 200 →              -- A gets $200
  total = a + b →        -- Total is the sum of A's and B's shares
  total = 600 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l1448_144871


namespace NUMINAMATH_CALUDE_grill_runtime_proof_l1448_144848

/-- Represents the burning rate of coals in a grill -/
structure BurningRate :=
  (coals : ℕ)
  (minutes : ℕ)

/-- Represents a bag of coals -/
structure CoalBag :=
  (coals : ℕ)

def grill_running_time (rate : BurningRate) (bags : ℕ) (bag : CoalBag) : ℕ :=
  (bags * bag.coals * rate.minutes) / rate.coals

theorem grill_runtime_proof (rate : BurningRate) (bags : ℕ) (bag : CoalBag)
  (h1 : rate.coals = 15)
  (h2 : rate.minutes = 20)
  (h3 : bags = 3)
  (h4 : bag.coals = 60) :
  grill_running_time rate bags bag = 240 :=
by
  sorry

#check grill_runtime_proof

end NUMINAMATH_CALUDE_grill_runtime_proof_l1448_144848


namespace NUMINAMATH_CALUDE_health_codes_survey_is_comprehensive_l1448_144807

/-- Represents a survey option -/
inductive SurveyOption
  | MovieViewing
  | SeedGermination
  | RiverWaterQuality
  | StudentHealthCodes

/-- Characteristics of a survey that make it suitable for a comprehensive survey (census) -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.StudentHealthCodes => true
  | _ => false

/-- Theorem stating that the survey on health codes of students during an epidemic
    is suitable for a comprehensive survey (census) -/
theorem health_codes_survey_is_comprehensive :
  isSuitableForComprehensiveSurvey SurveyOption.StudentHealthCodes :=
by sorry

end NUMINAMATH_CALUDE_health_codes_survey_is_comprehensive_l1448_144807


namespace NUMINAMATH_CALUDE_cos_18_degrees_l1448_144857

open Real

theorem cos_18_degrees : cos (18 * π / 180) = (Real.sqrt (10 + 2 * Real.sqrt 5)) / 4 := by sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l1448_144857


namespace NUMINAMATH_CALUDE_front_view_length_l1448_144824

theorem front_view_length
  (body_diagonal : ℝ)
  (side_view : ℝ)
  (top_view : ℝ)
  (h1 : body_diagonal = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) :
  ∃ front_view : ℝ,
    front_view = Real.sqrt 41 ∧
    side_view ^ 2 + top_view ^ 2 + front_view ^ 2 = body_diagonal ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_front_view_length_l1448_144824


namespace NUMINAMATH_CALUDE_cubic_inequality_l1448_144802

theorem cubic_inequality (x : ℝ) : x^3 - 16*x^2 + 73*x > 84 ↔ x > 13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1448_144802


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_production_l1448_144852

/-- The number of bottles produced per minute by a single machine -/
def bottles_per_machine_per_minute (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

/-- The number of bottles produced per minute by a given number of machines -/
def bottles_per_minute (bottles_per_machine : ℕ) (num_machines : ℕ) : ℕ :=
  bottles_per_machine * num_machines

/-- The total number of bottles produced in a given time -/
def total_bottles (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines will produce 1800 bottles in 4 minutes -/
theorem ten_machines_four_minutes_production 
  (h : bottles_per_minute (bottles_per_machine_per_minute 270 6) 10 = 450) :
  total_bottles 450 4 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ten_machines_four_minutes_production_l1448_144852


namespace NUMINAMATH_CALUDE_smallest_n_divisors_not_multiple_of_ten_l1448_144810

def is_perfect_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

def is_perfect_seventh (m : ℕ) : Prop := ∃ k : ℕ, m = k^7

def count_non_ten_divisors (n : ℕ) : ℕ := 
  (Finset.filter (fun d => ¬(10 ∣ d)) (Nat.divisors n)).card

theorem smallest_n_divisors_not_multiple_of_ten :
  ∃ n : ℕ, 
    (∀ m < n, ¬(is_perfect_cube (m / 2) ∧ is_perfect_square (m / 3) ∧ is_perfect_seventh (m / 5))) ∧
    is_perfect_cube (n / 2) ∧
    is_perfect_square (n / 3) ∧
    is_perfect_seventh (n / 5) ∧
    count_non_ten_divisors n = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisors_not_multiple_of_ten_l1448_144810


namespace NUMINAMATH_CALUDE_x_range_theorem_l1448_144803

theorem x_range_theorem (x : ℝ) : 
  (∀ (a b : ℝ), a > 0 → b > 0 → |x + 1| + |x - 2| ≤ (a + 1/b) * (1/a + b)) →
  -3/2 ≤ x ∧ x ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1448_144803


namespace NUMINAMATH_CALUDE_eulers_formula_l1448_144881

/-- A convex polyhedron is represented by its number of vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices + p.faces = p.edges + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1448_144881


namespace NUMINAMATH_CALUDE_area_bounded_by_curve_l1448_144819

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sqrt (4 - x^2)

theorem area_bounded_by_curve : ∫ x in (0)..(2), f x = π := by sorry

end NUMINAMATH_CALUDE_area_bounded_by_curve_l1448_144819


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1448_144844

theorem quadratic_factorization (a : ℕ+) :
  (∃ m n p q : ℤ, (21 : ℤ) * x^2 + (a : ℤ) * x + 21 = (m * x + n) * (p * x + q)) →
  ∃ k : ℕ+, a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1448_144844


namespace NUMINAMATH_CALUDE_linear_functions_properties_l1448_144849

/-- Linear function y₁ -/
def y₁ (x : ℝ) : ℝ := 50 + 2 * x

/-- Linear function y₂ -/
def y₂ (x : ℝ) : ℝ := 5 * x

theorem linear_functions_properties :
  (∃ x : ℝ, y₁ x > y₂ x) ∧ 
  (∃ x : ℝ, y₁ x < y₂ x) ∧
  (∀ x dx : ℝ, y₁ (x + dx) - y₁ x = 2 * dx) ∧
  (∀ x dx : ℝ, y₂ (x + dx) - y₂ x = 5 * dx) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ y₂ x₂ ≥ 100 ∧ y₁ x₁ ≥ 100 ∧ x₂ < x₁ ∧
    ∀ x : ℝ, x ≥ 1 → y₂ x ≥ 100 → x ≥ x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_functions_properties_l1448_144849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_nine_l1448_144812

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a_5 = 2, S_9 = 18 -/
theorem arithmetic_sequence_sum_nine 
  (seq : ArithmeticSequence) 
  (h : seq.a 5 = 2) : 
  seq.S 9 = 18 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_nine_l1448_144812


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l1448_144868

theorem multiplication_problem_solution :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    10 ≤ a * 8 ∧ a * 8 < 100 ∧
    100 ≤ a * 9 ∧ a * 9 < 1000 ∧
    a * b = 1068 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l1448_144868


namespace NUMINAMATH_CALUDE_pictures_hung_vertically_l1448_144805

theorem pictures_hung_vertically (total : ℕ) (horizontal : ℕ) (haphazard : ℕ) (vertical : ℕ) : 
  total = 30 → 
  horizontal = total / 2 → 
  haphazard = 5 → 
  vertical + horizontal + haphazard = total → 
  vertical = 10 := by
sorry

end NUMINAMATH_CALUDE_pictures_hung_vertically_l1448_144805


namespace NUMINAMATH_CALUDE_porter_monthly_earnings_l1448_144897

/-- Calculates the monthly earnings of a worker with overtime -/
def monthlyEarningsWithOvertime (dailyRate : ℕ) (regularDaysPerWeek : ℕ) (overtimeRatePercent : ℕ) (weeksInMonth : ℕ) : ℕ :=
  let regularWeeklyEarnings := dailyRate * regularDaysPerWeek
  let overtimeDailyRate := dailyRate * overtimeRatePercent / 100
  let overtimeWeeklyEarnings := dailyRate + overtimeDailyRate
  (regularWeeklyEarnings + overtimeWeeklyEarnings) * weeksInMonth

/-- Theorem stating that under given conditions, monthly earnings with overtime equal $208 -/
theorem porter_monthly_earnings :
  monthlyEarningsWithOvertime 8 5 150 4 = 208 := by
  sorry

#eval monthlyEarningsWithOvertime 8 5 150 4

end NUMINAMATH_CALUDE_porter_monthly_earnings_l1448_144897


namespace NUMINAMATH_CALUDE_place_five_in_three_l1448_144827

/-- The number of ways to place n distinct objects into k distinct containers -/
def place_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: Placing 5 distinct objects into 3 distinct containers results in 3^5 ways -/
theorem place_five_in_three : place_objects 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_place_five_in_three_l1448_144827


namespace NUMINAMATH_CALUDE_death_rate_is_eleven_l1448_144828

/-- Given a birth rate, net growth rate, and initial population, calculates the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Proves that given the specified conditions, the death rate is 11. -/
theorem death_rate_is_eleven :
  let birth_rate : ℝ := 32
  let net_growth_rate : ℝ := 0.021
  let initial_population : ℝ := 1000
  calculate_death_rate birth_rate net_growth_rate initial_population = 11 := by
  sorry

#eval calculate_death_rate 32 0.021 1000

end NUMINAMATH_CALUDE_death_rate_is_eleven_l1448_144828


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1448_144899

/-- Theorem: For a hyperbola x^2 - y^2/a^2 = 1 with a > 0, if its asymptotes are y = ± 2x, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 - y^2/a^2 = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1448_144899


namespace NUMINAMATH_CALUDE_line_opposite_sides_range_l1448_144876

/-- The range of 'a' for a line x + y - a = 0 with (0, 0) and (1, 1) on opposite sides -/
theorem line_opposite_sides_range (a : ℝ) : 
  (∀ x y : ℝ, x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_line_opposite_sides_range_l1448_144876


namespace NUMINAMATH_CALUDE_at_least_one_passes_l1448_144861

theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l1448_144861


namespace NUMINAMATH_CALUDE_inequality_proof_l1448_144856

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (Real.arctan ((a * d - b * c) / (a * c + b * d)))^2 ≥ 2 * (1 - (a * c + b * d) / Real.sqrt ((a^2 + b^2) * (c^2 + d^2))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1448_144856


namespace NUMINAMATH_CALUDE_emily_walk_distance_l1448_144818

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Proves that walking 8 blocks west and 10 blocks south, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem emily_walk_distance : total_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_emily_walk_distance_l1448_144818


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1448_144874

theorem complex_fraction_equality : 
  let numerator := ((5 / 2) ^ 2 / (1 / 2) ^ 3) * (5 / 2) ^ 2
  let denominator := ((5 / 3) ^ 4 * (1 / 2) ^ 2) / (2 / 3) ^ 3
  numerator / denominator = 48 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1448_144874


namespace NUMINAMATH_CALUDE_cube_division_impossibility_l1448_144809

/-- Represents a rectangular parallelepiped with dimensions (n, n+1, n+2) --/
structure Parallelepiped where
  n : ℕ

/-- The volume of a parallelepiped --/
def volume (p : Parallelepiped) : ℕ := p.n * (p.n + 1) * (p.n + 2)

/-- Theorem: It's impossible to divide a cube of volume 8000 into parallelepipeds
    with consecutive natural number dimensions --/
theorem cube_division_impossibility :
  ¬ ∃ (parallelepipeds : List Parallelepiped),
    (parallelepipeds.map volume).sum = 8000 :=
sorry

end NUMINAMATH_CALUDE_cube_division_impossibility_l1448_144809


namespace NUMINAMATH_CALUDE_correct_selection_schemes_l1448_144850

/-- Represents the number of translators for each language --/
structure TranslatorCounts where
  total : ℕ
  english : ℕ
  japanese : ℕ
  both : ℕ

/-- Represents the required team sizes --/
structure TeamSizes where
  english : ℕ
  japanese : ℕ

/-- Calculates the number of different selection schemes --/
def selectionSchemes (counts : TranslatorCounts) (sizes : TeamSizes) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem correct_selection_schemes :
  let counts : TranslatorCounts := ⟨11, 5, 4, 2⟩
  let sizes : TeamSizes := ⟨4, 4⟩
  selectionSchemes counts sizes = 144 := by sorry

end NUMINAMATH_CALUDE_correct_selection_schemes_l1448_144850


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l1448_144888

/-- Represents a parabola of the form y = -x^2 + 2x + c -/
def Parabola (c : ℝ) := {p : ℝ × ℝ | p.2 = -p.1^2 + 2*p.1 + c}

theorem parabola_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : (0, y₁) ∈ Parabola c)
  (h₂ : (1, y₂) ∈ Parabola c)
  (h₃ : (3, y₃) ∈ Parabola c) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l1448_144888


namespace NUMINAMATH_CALUDE_rabbit_carrots_l1448_144884

theorem rabbit_carrots : ∀ (rabbit_holes fox_holes : ℕ),
  rabbit_holes = fox_holes + 2 →
  5 * rabbit_holes = 6 * fox_holes →
  5 * rabbit_holes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l1448_144884


namespace NUMINAMATH_CALUDE_toy_spending_ratio_l1448_144833

theorem toy_spending_ratio (initial_amount : ℕ) (toy_cost : ℕ) (final_amount : ℕ) :
  initial_amount = 204 →
  final_amount = 51 →
  toy_cost + (initial_amount - toy_cost) / 2 + final_amount = initial_amount →
  toy_cost * 2 = initial_amount :=
by sorry

end NUMINAMATH_CALUDE_toy_spending_ratio_l1448_144833


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1448_144889

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > 5 ∧ x > a) ↔ x > 5) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1448_144889


namespace NUMINAMATH_CALUDE_square_9801_difference_of_squares_l1448_144875

theorem square_9801_difference_of_squares (x : ℤ) (h : x^2 = 9801) :
  (x + 1) * (x - 1) = 9800 := by
sorry

end NUMINAMATH_CALUDE_square_9801_difference_of_squares_l1448_144875


namespace NUMINAMATH_CALUDE_composite_divides_factorial_l1448_144836

theorem composite_divides_factorial (n : ℕ) (h1 : n > 4) (h2 : ¬ Nat.Prime n) :
  n ∣ Nat.factorial (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_divides_factorial_l1448_144836


namespace NUMINAMATH_CALUDE_sum_x_y_equals_two_l1448_144830

-- Define the function f(t) = t^3 + 2003t
def f (t : ℝ) : ℝ := t^3 + 2003*t

-- State the theorem
theorem sum_x_y_equals_two (x y : ℝ) 
  (hx : f (x - 1) = -1) 
  (hy : f (y - 1) = 1) : 
  x + y = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_x_y_equals_two_l1448_144830


namespace NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_series_l1448_144885

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_last_two_digits_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_series_l1448_144885


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1448_144886

theorem polynomial_remainder (f : ℝ → ℝ) (a b c d : ℝ) (h : a ≠ b) :
  (∃ g : ℝ → ℝ, ∀ x, f x = (x - a) * g x + c) →
  (∃ h : ℝ → ℝ, ∀ x, f x = (x - b) * h x + d) →
  ∃ k : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * k x + ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1448_144886


namespace NUMINAMATH_CALUDE_keith_total_cost_l1448_144847

def rabbit_toy_cost : ℚ := 6.51
def pet_food_cost : ℚ := 5.79
def cage_cost : ℚ := 12.51
def found_money : ℚ := 1.00

theorem keith_total_cost :
  rabbit_toy_cost + pet_food_cost + cage_cost - found_money = 23.81 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_cost_l1448_144847


namespace NUMINAMATH_CALUDE_tom_age_ratio_l1448_144858

/-- Tom's current age -/
def T : ℕ := sorry

/-- Number of years ago mentioned in the second condition -/
def N : ℕ := 5

/-- Sum of the current ages of Tom's three children -/
def children_sum : ℕ := T / 2

/-- Tom's age N years ago -/
def tom_age_N_years_ago : ℕ := T - N

/-- Sum of the ages of Tom's children N years ago -/
def children_sum_N_years_ago : ℕ := children_sum - 3 * N

/-- The theorem stating the ratio of T to N -/
theorem tom_age_ratio : T / N = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_age_ratio_l1448_144858


namespace NUMINAMATH_CALUDE_coin_arrangement_strategy_exists_l1448_144821

/-- Represents a strategy for arranging coins by weight --/
structure CoinArrangementStrategy where
  /-- Function that decides which coins to compare at each step --/
  compareCoins : ℕ → (ℕ × ℕ)
  /-- Maximum number of comparisons needed --/
  maxComparisons : ℕ

/-- Represents the expected number of comparisons for a strategy --/
def expectedComparisons (strategy : CoinArrangementStrategy) : ℚ :=
  sorry

/-- There exists a strategy to arrange 4 coins with expected comparisons less than 4.8 --/
theorem coin_arrangement_strategy_exists :
  ∃ (strategy : CoinArrangementStrategy),
    strategy.maxComparisons ≤ 4 ∧ expectedComparisons strategy < 24/5 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_strategy_exists_l1448_144821


namespace NUMINAMATH_CALUDE_basketball_score_total_l1448_144804

/-- Given the scores of three basketball players Tim, Joe, and Ken, prove their total score. -/
theorem basketball_score_total (tim joe ken : ℕ) : 
  tim = joe + 20 →   -- Tim scored 20 points more than Joe
  tim = ken / 2 →    -- Tim scored half as many points as Ken
  tim = 30 →         -- Tim scored 30 points
  tim + joe + ken = 100 := by  -- The total score of all three players is 100
sorry

end NUMINAMATH_CALUDE_basketball_score_total_l1448_144804


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1448_144870

/-- An arithmetic sequence satisfying a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ d : ℝ, ∀ k : ℕ, a (k + 1) = a k + d

/-- The specific condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SequenceCondition a) :
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1448_144870


namespace NUMINAMATH_CALUDE_x_sum_less_than_2m_l1448_144841

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 5 - a / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

theorem x_sum_less_than_2m (a : ℝ) (m x₁ x₂ : ℝ) 
  (h1 : m ≥ 1) 
  (h2 : x₁ < m) 
  (h3 : x₂ > m) 
  (h4 : g a x₁ + g a x₂ = 2 * g a m) : 
  x₁ + x₂ < 2 * m := by
  sorry

end NUMINAMATH_CALUDE_x_sum_less_than_2m_l1448_144841


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1448_144834

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1221 → ¬(∃ k : ℕ, n^3 + 99 = k * (n + 11)) ∧ 
  ∃ k : ℕ, 1221^3 + 99 = k * (1221 + 11) :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1448_144834


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_eq_neg_e_l1448_144883

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x
  else if x = 0 then 0
  else Real.exp (-x) - a * x

-- State the theorem
theorem three_zeros_implies_a_eq_neg_e (a : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a = -Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_three_zeros_implies_a_eq_neg_e_l1448_144883


namespace NUMINAMATH_CALUDE_peters_remaining_money_l1448_144882

/-- Calculates Peter's remaining money after market purchases -/
theorem peters_remaining_money
  (initial_amount : ℕ)
  (potato_quantity potato_price : ℕ)
  (tomato_quantity tomato_price : ℕ)
  (cucumber_quantity cucumber_price : ℕ)
  (banana_quantity banana_price : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_quantity = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_quantity = 9)
  (h5 : tomato_price = 3)
  (h6 : cucumber_quantity = 5)
  (h7 : cucumber_price = 4)
  (h8 : banana_quantity = 3)
  (h9 : banana_price = 5) :
  initial_amount - (potato_quantity * potato_price +
                    tomato_quantity * tomato_price +
                    cucumber_quantity * cucumber_price +
                    banana_quantity * banana_price) = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l1448_144882


namespace NUMINAMATH_CALUDE_john_yearly_expenses_l1448_144878

/-- Calculates the total amount John needs to pay for his EpiPens and additional medical expenses for a year. -/
def total_yearly_expenses (epipen_cost : ℚ) (first_epipen_coverage : ℚ) (second_epipen_coverage : ℚ) (yearly_medical_expenses : ℚ) (medical_expenses_coverage : ℚ) : ℚ :=
  let first_epipen_payment := epipen_cost * (1 - first_epipen_coverage)
  let second_epipen_payment := epipen_cost * (1 - second_epipen_coverage)
  let total_epipen_cost := first_epipen_payment + second_epipen_payment
  let medical_expenses_payment := yearly_medical_expenses * (1 - medical_expenses_coverage)
  total_epipen_cost + medical_expenses_payment

/-- Theorem stating that John's total yearly expenses are $725 given the problem conditions. -/
theorem john_yearly_expenses :
  total_yearly_expenses 500 0.75 0.6 2000 0.8 = 725 := by
  sorry

end NUMINAMATH_CALUDE_john_yearly_expenses_l1448_144878


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1448_144815

/-- A rhombus with area 192 and diagonal ratio 4:3 has longest diagonal of length 16√2 -/
theorem rhombus_longest_diagonal (d₁ d₂ : ℝ) : 
  d₁ * d₂ / 2 = 192 →  -- Area formula
  d₁ / d₂ = 4 / 3 →    -- Diagonal ratio
  max d₁ d₂ = 16 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1448_144815


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_five_is_smallest_smallest_n_zero_l1448_144822

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by sorry

theorem five_is_smallest (n : ℕ) (h : n < 5) : 2^n ≤ n^2 := by sorry

theorem smallest_n_zero : ∃ (n₀ : ℕ), (∀ (n : ℕ), n ≥ n₀ → 2^n > n^2) ∧ 
  (∀ (m : ℕ), m < n₀ → 2^m ≤ m^2) := by
  use 5
  constructor
  · exact power_two_greater_than_square
  · exact five_is_smallest

end NUMINAMATH_CALUDE_power_two_greater_than_square_five_is_smallest_smallest_n_zero_l1448_144822


namespace NUMINAMATH_CALUDE_carpenter_rate_proof_l1448_144892

def carpenter_hourly_rate (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ) : ℚ :=
  (total_estimate - material_cost) / job_duration

theorem carpenter_rate_proof (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ)
  (h1 : total_estimate = 980)
  (h2 : material_cost = 560)
  (h3 : job_duration = 15) :
  carpenter_hourly_rate total_estimate material_cost job_duration = 28 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_rate_proof_l1448_144892


namespace NUMINAMATH_CALUDE_age_difference_proof_l1448_144811

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h1 : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1448_144811


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1448_144860

/-- A point P with coordinates (m+2, m+1) lies on the y-axis if and only if its coordinates are (0, -1) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 2 = 0 ∧ ∃ y, (0, y) = (m + 2, m + 1)) ↔ (0, -1) = (m + 2, m + 1) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1448_144860


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_4_minus_sqrt_2_l1448_144873

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution_is_4_minus_sqrt_2 :
  ∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) ∧
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_4_minus_sqrt_2_l1448_144873


namespace NUMINAMATH_CALUDE_special_function_period_l1448_144800

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = (1 + f x) / (1 - f x)

/-- The period of a real function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The main theorem: if f is a SpecialFunction with parameter a, then it has period 4|a| -/
theorem special_function_period (f : ℝ → ℝ) (a : ℝ) 
    (hf : SpecialFunction f a) : HasPeriod f (4 * |a|) := by
  sorry

end NUMINAMATH_CALUDE_special_function_period_l1448_144800


namespace NUMINAMATH_CALUDE_labourer_monthly_income_l1448_144801

/-- Represents the financial situation of a labourer over a 10-month period --/
structure LabourerFinances where
  monthlyIncome : ℝ
  firstSixMonthsExpenditure : ℝ
  nextFourMonthsExpenditure : ℝ
  savings : ℝ

/-- Theorem stating the labourer's monthly income given the problem conditions --/
theorem labourer_monthly_income 
  (finances : LabourerFinances)
  (h1 : finances.firstSixMonthsExpenditure = 90 * 6)
  (h2 : finances.monthlyIncome * 6 < finances.firstSixMonthsExpenditure)
  (h3 : finances.nextFourMonthsExpenditure = 60 * 4)
  (h4 : finances.monthlyIncome * 4 = finances.nextFourMonthsExpenditure + finances.savings)
  (h5 : finances.savings = 30) :
  finances.monthlyIncome = 81 := by
  sorry

end NUMINAMATH_CALUDE_labourer_monthly_income_l1448_144801


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1448_144869

theorem tangent_line_problem (a : ℝ) : 
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 3 * x^2 = m)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 2 * a * x + 15/4 = m)))))
  → a = -25/64 ∨ a = -1 := by
  sorry

#check tangent_line_problem

end NUMINAMATH_CALUDE_tangent_line_problem_l1448_144869


namespace NUMINAMATH_CALUDE_problem_statement_l1448_144839

theorem problem_statement (x : ℝ) : 
  (1/5)^35 * (1/4)^18 = 1/(x*(10)^35) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1448_144839


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1448_144826

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_periodic_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_period : hasPeriod f 2) : 
  isIncreasingOn f 0 1 ↔ isDecreasingOn f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1448_144826


namespace NUMINAMATH_CALUDE_complex_multiplication_l1448_144895

theorem complex_multiplication (P F G : ℂ) : 
  P = 3 + 4*Complex.I ∧ 
  F = 2*Complex.I ∧ 
  G = 3 - 4*Complex.I → 
  (P + F) * G = 21 + 6*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1448_144895


namespace NUMINAMATH_CALUDE_train_speed_l1448_144866

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 12) :
  length / time = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1448_144866


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1448_144864

theorem complex_arithmetic_equality : (5 - 5*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I) = -10*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1448_144864


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l1448_144863

/-- Given odds of 5:8 for pulling a prize, the probability of not pulling the prize is 8/13 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h_odds : favorable_outcomes = 5 ∧ unfavorable_outcomes = 8) :
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 8 / 13 := by
  sorry


end NUMINAMATH_CALUDE_probability_not_pulling_prize_l1448_144863


namespace NUMINAMATH_CALUDE_range_of_f_on_I_l1448_144845

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x

-- Define the interval
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f_on_I :
  {y | ∃ x ∈ I, f x = y} = {y | -1/4 ≤ y ∧ y ≤ 12} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_on_I_l1448_144845


namespace NUMINAMATH_CALUDE_compute_fraction_power_l1448_144867

theorem compute_fraction_power : 9 * (1/7)^4 = 9/2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l1448_144867


namespace NUMINAMATH_CALUDE_sin_15_plus_cos_15_l1448_144887

theorem sin_15_plus_cos_15 : Real.sin (15 * π / 180) + Real.cos (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_plus_cos_15_l1448_144887


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1448_144891

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 3) :
  (2 - x) / (x - 3) + 1 / (3 - x) = 1 ↔ x = 2 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1448_144891


namespace NUMINAMATH_CALUDE_brads_money_l1448_144823

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ) :
  total = 68 →
  josh_brad_ratio = 2 →
  josh_doug_ratio = 3/4 →
  ∃ (brad josh doug : ℚ),
    brad + josh + doug = total ∧
    josh = josh_brad_ratio * brad ∧
    josh = josh_doug_ratio * doug ∧
    brad = 12 :=
by sorry

end NUMINAMATH_CALUDE_brads_money_l1448_144823


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l1448_144865

theorem geometric_sequence_b_value (b : ℝ) (h_positive : b > 0) 
  (h_sequence : ∃ r : ℝ, 250 * r = b ∧ b * r = 81 / 50) : 
  b = 9 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l1448_144865


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1448_144808

-- Define the first quadrant
def first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

-- Define the fourth quadrant
def fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) (h : first_quadrant α) :
  fourth_quadrant (360 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1448_144808


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l1448_144806

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = -1/12 ∧ d = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l1448_144806


namespace NUMINAMATH_CALUDE_problem_1_l1448_144854

theorem problem_1 : 12 - (-10) + 7 = 29 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1448_144854


namespace NUMINAMATH_CALUDE_local_extrema_condition_l1448_144851

/-- A function f with parameter a that we want to analyze -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a + 6

theorem local_extrema_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), IsLocalMin (f a) x₁ ∧ IsLocalMax (f a) x₂) ↔ (a ≤ -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_local_extrema_condition_l1448_144851


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l1448_144825

theorem existence_of_special_polynomial :
  ∃ (f : Polynomial ℤ), 
    (∀ (i : ℕ), (f.coeff i = 1 ∨ f.coeff i = -1)) ∧ 
    (∃ (g : Polynomial ℤ), f = g * (X - 1) ^ 2013) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l1448_144825


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1448_144872

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 400 * π → (4 / 3) * π * r^3 = (4000 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1448_144872


namespace NUMINAMATH_CALUDE_find_a_over_b_l1448_144846

-- Define the region
def region (x y : ℝ) : Prop :=
  x ≥ 1 ∧ x + y ≤ 4 ∧ ∃ a b : ℝ, a * x + b * y + 2 ≥ 0

-- Define the objective function
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem find_a_over_b :
  ∃ a b : ℝ,
    (∀ x y : ℝ, region x y → z x y ≤ 7) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 7) ∧
    (∀ x y : ℝ, region x y → z x y ≥ 1) ∧
    (∃ x y : ℝ, region x y ∧ z x y = 1) ∧
    a / b = -1 :=
sorry

end NUMINAMATH_CALUDE_find_a_over_b_l1448_144846


namespace NUMINAMATH_CALUDE_derivative_of_x_minus_sin_l1448_144859

open Real

theorem derivative_of_x_minus_sin (x : ℝ) : 
  deriv (fun x => x - sin x) x = 1 - cos x := by
sorry

end NUMINAMATH_CALUDE_derivative_of_x_minus_sin_l1448_144859


namespace NUMINAMATH_CALUDE_max_y_value_l1448_144817

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * Real.log (y / x) - y * Real.exp x + x * (x + 1) ≥ 0) : 
  y ≤ 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l1448_144817


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_1001_l1448_144855

theorem three_digit_squares_ending_1001 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → (n^2 % 10000 = 1001 ↔ n = 501 ∨ n = 749) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_1001_l1448_144855


namespace NUMINAMATH_CALUDE_red_triangles_in_colored_graph_l1448_144813

/-- A coloring of a complete graph is a function that assigns either red or blue to each edge. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- The set of vertices connected to a given vertex by red edges. -/
def RedNeighborhood (n : ℕ) (c : Coloring n) (v : Fin n) : Finset (Fin n) :=
  Finset.filter (fun u => c v u) (Finset.univ.erase v)

/-- A red triangle in a colored complete graph. -/
def RedTriangle (n : ℕ) (c : Coloring n) (v1 v2 v3 : Fin n) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ c v1 v2 ∧ c v2 v3 ∧ c v1 v3

theorem red_triangles_in_colored_graph (k : ℕ) (h : k ≥ 3) :
  ∀ (c : Coloring (3*k+2)),
  (∀ v, (RedNeighborhood (3*k+2) c v).card ≥ k+2) →
  (∀ v w, ¬c v w → (RedNeighborhood (3*k+2) c v ∪ RedNeighborhood (3*k+2) c w).card ≥ 2*k+2) →
  ∃ (S : Finset (Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2))),
    S.card ≥ k+2 ∧ ∀ (t : Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2)), t ∈ S → RedTriangle (3*k+2) c t.1 t.2.1 t.2.2 :=
by sorry

end NUMINAMATH_CALUDE_red_triangles_in_colored_graph_l1448_144813


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1448_144832

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax² for some a ∈ ℚ -/
theorem functional_equation_solution :
  ∀ f : ℚ → ℚ, SatisfiesFunctionalEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l1448_144832


namespace NUMINAMATH_CALUDE_knight_moves_equal_for_7x7_l1448_144890

/-- Represents a position on a chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a knight's move on a chessboard -/
inductive KnightMove : Position → Position → Prop where
  | move_1 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- The minimum number of moves for a knight to reach a target position from a start position -/
def minKnightMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equal_for_7x7 :
  let start := Position.mk 0 0
  let upperRight := Position.mk 6 6
  let lowerRight := Position.mk 6 0
  minKnightMoves start upperRight = minKnightMoves start lowerRight :=
by
  sorry

end NUMINAMATH_CALUDE_knight_moves_equal_for_7x7_l1448_144890


namespace NUMINAMATH_CALUDE_items_left_in_store_l1448_144843

/-- Given the number of items ordered, sold, and in the storeroom, 
    calculate the total number of items left in the whole store. -/
theorem items_left_in_store 
  (items_ordered : ℕ) 
  (items_sold : ℕ) 
  (items_in_storeroom : ℕ) 
  (h1 : items_ordered = 4458)
  (h2 : items_sold = 1561)
  (h3 : items_in_storeroom = 575) :
  items_ordered - items_sold + items_in_storeroom = 3472 :=
by sorry

end NUMINAMATH_CALUDE_items_left_in_store_l1448_144843


namespace NUMINAMATH_CALUDE_max_weeks_correct_l1448_144877

/-- Represents a weekly ranking of 10 songs -/
def Ranking := Fin 10 → Fin 10

/-- The maximum number of weeks the same 10 songs can remain in the ranking -/
def max_weeks : ℕ := 46

/-- A function that represents the ranking change from one week to the next -/
def next_week (r : Ranking) : Ranking := sorry

/-- Predicate to check if a song's ranking has dropped -/
def has_dropped (r1 r2 : Ranking) (song : Fin 10) : Prop :=
  r2 song > r1 song

theorem max_weeks_correct (initial : Ranking) :
  ∀ (sequence : ℕ → Ranking),
    (∀ n, sequence (n + 1) = next_week (sequence n)) →
    (∀ n, sequence n ≠ sequence (n + 1)) →
    (∀ n m song, n < m → has_dropped (sequence n) (sequence m) song →
      ∀ k > m, has_dropped (sequence m) (sequence k) song ∨ sequence m song = sequence k song) →
    (∃ n ≤ max_weeks, ∃ song, sequence 0 song ≠ sequence n song) ∧
    (∀ n > max_weeks, ∃ song, sequence 0 song ≠ sequence n song) :=
  sorry


end NUMINAMATH_CALUDE_max_weeks_correct_l1448_144877


namespace NUMINAMATH_CALUDE_subset_range_equivalence_l1448_144853

theorem subset_range_equivalence (a : ℝ) :
  ({x : ℝ | x^2 + 2*(1-a)*x + (3-a) ≤ 0} ⊆ Set.Icc 0 3) ↔ (-1 ≤ a ∧ a ≤ 18/7) := by
  sorry

end NUMINAMATH_CALUDE_subset_range_equivalence_l1448_144853


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1448_144862

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,3}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1448_144862


namespace NUMINAMATH_CALUDE_garden_dimensions_l1448_144840

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  breadth : ℝ

/-- Checks if the given dimensions satisfy the garden constraints. -/
def satisfiesConstraints (d : GardenDimensions) : Prop :=
  d.length = (3 / 5) * d.breadth ∧
  d.length * d.breadth = 600 ∧
  2 * (d.length + d.breadth) ≤ 120

/-- Theorem stating the correct dimensions of the garden. -/
theorem garden_dimensions :
  ∃ (d : GardenDimensions),
    satisfiesConstraints d ∧
    d.length = 6 * Real.sqrt 10 ∧
    d.breadth = 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_garden_dimensions_l1448_144840


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l1448_144831

/-- The length of the generatrix of a cone with base radius √2 and lateral surface forming a semicircle when unfolded is 2√2. -/
theorem cone_generatrix_length :
  ∀ (base_radius : ℝ) (generatrix_length : ℝ),
  base_radius = Real.sqrt 2 →
  2 * Real.pi * base_radius = Real.pi * generatrix_length →
  generatrix_length = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l1448_144831


namespace NUMINAMATH_CALUDE_integral_cos_quadratic_l1448_144814

theorem integral_cos_quadratic (f : ℝ → ℝ) :
  (∫ x in (0)..(2 * Real.pi), (1 - 8 * x^2) * Real.cos (4 * x)) = -2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_integral_cos_quadratic_l1448_144814


namespace NUMINAMATH_CALUDE_hotel_expenditure_l1448_144816

theorem hotel_expenditure (num_persons : ℕ) (regular_spend : ℕ) (extra_spend : ℕ) : 
  num_persons = 9 →
  regular_spend = 12 →
  extra_spend = 8 →
  (num_persons - 1) * regular_spend + 
  (((num_persons - 1) * regular_spend + (regular_spend + extra_spend)) / num_persons + extra_spend) = 117 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l1448_144816


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1448_144820

open Set

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1448_144820


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l1448_144842

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l1448_144842


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1448_144880

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8 > 0) →
  (m - 3 > 0) →
  (3 * m + 8) * (m - 3) = 85 →
  m = (1 + Real.sqrt 1309) / 6 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1448_144880


namespace NUMINAMATH_CALUDE_factor_expression_l1448_144894

theorem factor_expression (x y : ℝ) : 100 - 25 * x^2 + 16 * y^2 = (10 - 5*x + 4*y) * (10 + 5*x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1448_144894


namespace NUMINAMATH_CALUDE_number_of_adults_at_play_l1448_144898

/-- The number of adults attending a play, given ticket prices and conditions. -/
theorem number_of_adults_at_play : ℕ :=
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let extra_adult_cost : ℕ := 50
  9

#check number_of_adults_at_play

end NUMINAMATH_CALUDE_number_of_adults_at_play_l1448_144898


namespace NUMINAMATH_CALUDE_problem_1_l1448_144837

theorem problem_1 : (-5) + (-2) + 9 - (-8) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1448_144837


namespace NUMINAMATH_CALUDE_factor_expression_l1448_144835

theorem factor_expression (x : ℝ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1448_144835


namespace NUMINAMATH_CALUDE_calculate_expression_l1448_144829

theorem calculate_expression : 
  Real.sqrt 5 * 5^(1/3) + 15 / 5 * 3 - 9^(5/2) = 5^(5/6) - 234 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1448_144829
