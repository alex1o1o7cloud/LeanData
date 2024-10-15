import Mathlib

namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1692_169245

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1692_169245


namespace NUMINAMATH_CALUDE_middle_number_problem_l1692_169269

theorem middle_number_problem :
  ∃! n : ℕ, 
    (n - 1)^2 + n^2 + (n + 1)^2 = 2030 ∧
    7 ∣ (n^3 - n^2) ∧
    n = 26 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l1692_169269


namespace NUMINAMATH_CALUDE_henry_trips_problem_l1692_169270

def henry_trips (carry_capacity : ℕ) (table1_trays : ℕ) (table2_trays : ℕ) : ℕ :=
  (table1_trays + table2_trays + carry_capacity - 1) / carry_capacity

theorem henry_trips_problem : henry_trips 9 29 52 = 9 := by
  sorry

end NUMINAMATH_CALUDE_henry_trips_problem_l1692_169270


namespace NUMINAMATH_CALUDE_leftover_seashells_proof_l1692_169250

/-- The number of leftover seashells after packaging -/
def leftover_seashells (derek_shells : ℕ) (emily_shells : ℕ) (fiona_shells : ℕ) (package_size : ℕ) : ℕ :=
  (derek_shells + emily_shells + fiona_shells) % package_size

theorem leftover_seashells_proof (derek_shells emily_shells fiona_shells package_size : ℕ) 
  (h_package_size : package_size > 0) :
  leftover_seashells derek_shells emily_shells fiona_shells package_size = 
  (derek_shells + emily_shells + fiona_shells) % package_size :=
by
  sorry

#eval leftover_seashells 58 73 31 10

end NUMINAMATH_CALUDE_leftover_seashells_proof_l1692_169250


namespace NUMINAMATH_CALUDE_card_ratio_l1692_169258

/-- Prove that given the conditions in the problem, the ratio of football cards to hockey cards is 4:1 -/
theorem card_ratio (total_cards : ℕ) (hockey_cards : ℕ) (s : ℕ) :
  total_cards = 1750 →
  hockey_cards = 200 →
  total_cards = (s * hockey_cards - 50) + (s * hockey_cards) + hockey_cards →
  (s * hockey_cards) / hockey_cards = 4 :=
by sorry

end NUMINAMATH_CALUDE_card_ratio_l1692_169258


namespace NUMINAMATH_CALUDE_triangle_side_length_l1692_169203

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (ha : t.a = Real.sqrt 5) 
  (hc : t.c = 2) 
  (hcosA : Real.cos t.A = 2/3) : 
  t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1692_169203


namespace NUMINAMATH_CALUDE_min_semi_focal_distance_l1692_169224

/-- The minimum semi-focal distance of a hyperbola satisfying certain conditions -/
theorem min_semi_focal_distance (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  c^2 = a^2 + b^2 →  -- Definition of semi-focal distance for hyperbola
  (1/3 * c + 1) * c = a * b →  -- Condition on distance from origin to line
  c ≥ 6 := by sorry

end NUMINAMATH_CALUDE_min_semi_focal_distance_l1692_169224


namespace NUMINAMATH_CALUDE_existence_of_single_root_quadratic_l1692_169235

/-- Given a quadratic polynomial with leading coefficient 1 and exactly one root,
    there exists a point (p, q) such that x^2 + px + q also has exactly one root. -/
theorem existence_of_single_root_quadratic 
  (b c : ℝ) 
  (h1 : b^2 - 4*c = 0) : 
  ∃ p q : ℝ, p^2 - 4*q = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_single_root_quadratic_l1692_169235


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1692_169240

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 32) 
  (h2 : height = 8) 
  (h3 : area = base * height) : 
  base = 4 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1692_169240


namespace NUMINAMATH_CALUDE_erica_pie_fraction_l1692_169205

theorem erica_pie_fraction (apple_fraction : ℚ) : 
  (apple_fraction + 3/4 = 95/100) → apple_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_erica_pie_fraction_l1692_169205


namespace NUMINAMATH_CALUDE_mike_washed_nine_cars_l1692_169278

/-- Time in minutes to wash one car -/
def wash_time : ℕ := 10

/-- Time in minutes to change oil on one car -/
def oil_change_time : ℕ := 15

/-- Time in minutes to change one set of tires -/
def tire_change_time : ℕ := 30

/-- Number of cars Mike changed oil on -/
def oil_changes : ℕ := 6

/-- Number of sets of tires Mike changed -/
def tire_changes : ℕ := 2

/-- Total time Mike worked in minutes -/
def total_work_time : ℕ := 4 * 60

/-- Function to calculate the number of cars Mike washed -/
def cars_washed : ℕ :=
  (total_work_time - (oil_changes * oil_change_time + tire_changes * tire_change_time)) / wash_time

/-- Theorem stating that Mike washed 9 cars -/
theorem mike_washed_nine_cars : cars_washed = 9 := by
  sorry

end NUMINAMATH_CALUDE_mike_washed_nine_cars_l1692_169278


namespace NUMINAMATH_CALUDE_water_remaining_l1692_169276

theorem water_remaining (total : ℚ) (used : ℚ) (h1 : total = 3) (h2 : used = 4/3) :
  total - used = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l1692_169276


namespace NUMINAMATH_CALUDE_range_of_a_l1692_169252

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1692_169252


namespace NUMINAMATH_CALUDE_domain_of_f_l1692_169233

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (2 * x + 1) / Real.log (1/2))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 < x ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l1692_169233


namespace NUMINAMATH_CALUDE_initial_roses_l1692_169296

theorem initial_roses (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 4)
  (h2 : added = 25)
  (h3 : final = 23)
  (h4 : initial - thrown_away + added = final) : initial = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_roses_l1692_169296


namespace NUMINAMATH_CALUDE_divisors_of_2700_l1692_169206

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_2700 : number_of_divisors 2700 = 36 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2700_l1692_169206


namespace NUMINAMATH_CALUDE_linear_function_second_quadrant_increasing_l1692_169262

/-- A linear function passing through the second quadrant with increasing y as x increases -/
def LinearFunctionSecondQuadrantIncreasing (k b : ℝ) : Prop :=
  k > 0 ∧ b > 0

/-- The property of a function passing through the second quadrant -/
def PassesThroughSecondQuadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x < 0 ∧ y > 0 ∧ f x = y

/-- The property of a function being increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Theorem stating that a linear function with positive slope and y-intercept
    passes through the second quadrant and is increasing -/
theorem linear_function_second_quadrant_increasing (k b : ℝ) :
  LinearFunctionSecondQuadrantIncreasing k b ↔
  PassesThroughSecondQuadrant (λ x => k * x + b) ∧
  IsIncreasing (λ x => k * x + b) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_second_quadrant_increasing_l1692_169262


namespace NUMINAMATH_CALUDE_proposition_truths_l1692_169210

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := 
  (∃ (q : ℚ), a + b = q) → (∃ (r s : ℚ), a = r ∧ b = s)

def proposition2 (a b : ℝ) : Prop :=
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)

def proposition3 (a b : ℝ) : Prop :=
  ∀ x, a * x + b > 0 ↔ x > -b / a

def proposition4 (a b c : ℝ) : Prop :=
  (∃ x, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0

-- Theorem stating which propositions are true
theorem proposition_truths :
  (∃ a b : ℝ, ¬ proposition1 a b) ∧
  (∀ a b : ℝ, proposition2 a b) ∧
  (∃ a b : ℝ, ¬ proposition3 a b) ∧
  (∀ a b c : ℝ, proposition4 a b c) :=
sorry

end NUMINAMATH_CALUDE_proposition_truths_l1692_169210


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l1692_169232

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l1692_169232


namespace NUMINAMATH_CALUDE_fourth_week_sugar_l1692_169284

def sugar_reduction (initial_amount : ℚ) (weeks : ℕ) : ℚ :=
  initial_amount / (2 ^ weeks)

theorem fourth_week_sugar : sugar_reduction 24 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_sugar_l1692_169284


namespace NUMINAMATH_CALUDE_first_term_value_l1692_169280

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 40 terms is 180
  sum_first_40 : (40 : ℚ) / 2 * (2 * a + 39 * d) = 180
  -- Sum of next 40 terms (41st to 80th) is 2200
  sum_next_40 : (40 : ℚ) / 2 * (2 * (a + 40 * d) + 39 * d) = 2200
  -- 20th term is 75
  term_20 : a + 19 * d = 75

/-- The first term of the arithmetic sequence with given properties is 51.0125 -/
theorem first_term_value (seq : ArithmeticSequence) : seq.a = 51.0125 := by
  sorry

end NUMINAMATH_CALUDE_first_term_value_l1692_169280


namespace NUMINAMATH_CALUDE_michael_born_in_1979_l1692_169219

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The number of AMC 8 competitions Michael has taken -/
def michaels_amc8_number : ℕ := 10

/-- Michael's age when he took his AMC 8 -/
def michaels_age : ℕ := 15

/-- Function to calculate the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Michael's birth year -/
def michaels_birth_year : ℕ := amc8_year michaels_amc8_number - michaels_age

theorem michael_born_in_1979 : michaels_birth_year = 1979 := by
  sorry

end NUMINAMATH_CALUDE_michael_born_in_1979_l1692_169219


namespace NUMINAMATH_CALUDE_certain_number_proof_l1692_169275

theorem certain_number_proof (N x : ℝ) (h1 : N / (1 + 3 / x) = 1) (h2 : x = 1) : N = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1692_169275


namespace NUMINAMATH_CALUDE_b_current_age_l1692_169221

/-- Given two people A and B, where in 10 years A will be twice as old as B was 10 years ago,
    and A is currently 7 years older than B, prove that B's current age is 37 years. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 7) → 
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_b_current_age_l1692_169221


namespace NUMINAMATH_CALUDE_number_of_boys_l1692_169226

theorem number_of_boys (total_amount : ℕ) (total_children : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
    boys * boy_amount + (total_children - boys) * girl_amount = total_amount :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l1692_169226


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1692_169238

/-- Represents the repeating decimal 0.42̄157 -/
def repeating_decimal : ℚ := 42157 / 100000 + (157 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 4207359 / 99900

/-- Theorem stating that the repeating decimal 0.42̄157 is equal to 4207359/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1692_169238


namespace NUMINAMATH_CALUDE_complex_equation_system_l1692_169229

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 10)
  (eq5 : s + t + u = 6) :
  s * t * u = 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_system_l1692_169229


namespace NUMINAMATH_CALUDE_apple_difference_apple_problem_solution_l1692_169200

/-- Proves that Mark has 13 fewer apples than Susan given the conditions of the problem -/
theorem apple_difference : ℕ → Prop := fun total_apples =>
  ∀ (greg_sarah_apples susan_apples mark_apples mom_pie_apples mom_leftover_apples : ℕ),
    greg_sarah_apples = 18 →
    susan_apples = 2 * (greg_sarah_apples / 2) →
    mom_pie_apples = 40 →
    mom_leftover_apples = 9 →
    total_apples = mom_pie_apples + mom_leftover_apples →
    mark_apples = total_apples - susan_apples →
    susan_apples - mark_apples = 13

/-- The main theorem stating that there exists a total number of apples satisfying the conditions -/
theorem apple_problem_solution : ∃ total_apples : ℕ, apple_difference total_apples := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_apple_problem_solution_l1692_169200


namespace NUMINAMATH_CALUDE_hexagon_tile_difference_l1692_169298

/-- Given a hexagonal figure with initial red and yellow tiles, prove the difference
    between yellow and red tiles after adding a border of yellow tiles. -/
theorem hexagon_tile_difference (initial_red : ℕ) (initial_yellow : ℕ) 
    (sides : ℕ) (tiles_per_side : ℕ) :
  initial_red = 15 →
  initial_yellow = 9 →
  sides = 6 →
  tiles_per_side = 4 →
  let new_yellow := initial_yellow + sides * tiles_per_side
  new_yellow - initial_red = 18 := by
sorry

end NUMINAMATH_CALUDE_hexagon_tile_difference_l1692_169298


namespace NUMINAMATH_CALUDE_fraction_in_lowest_terms_l1692_169285

theorem fraction_in_lowest_terms (n : ℤ) (h : Odd n) :
  Nat.gcd (Int.natAbs (2 * n + 2)) (Int.natAbs (3 * n + 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_in_lowest_terms_l1692_169285


namespace NUMINAMATH_CALUDE_factory_assignment_l1692_169277

-- Define the workers and machines
inductive Worker : Type
  | Dan : Worker
  | Emma : Worker
  | Fiona : Worker

inductive Machine : Type
  | A : Machine
  | B : Machine
  | C : Machine

-- Define the assignment of workers to machines
def Assignment := Worker → Machine

-- Define the conditions
def condition1 (a : Assignment) : Prop := a Worker.Emma ≠ Machine.A
def condition2 (a : Assignment) : Prop := a Worker.Dan = Machine.C
def condition3 (a : Assignment) : Prop := a Worker.Fiona = Machine.B

-- Define the correct assignment
def correct_assignment : Assignment :=
  fun w => match w with
    | Worker.Dan => Machine.C
    | Worker.Emma => Machine.A
    | Worker.Fiona => Machine.B

-- Theorem statement
theorem factory_assignment :
  ∀ (a : Assignment),
    (a Worker.Dan ≠ a Worker.Emma ∧ a Worker.Dan ≠ a Worker.Fiona ∧ a Worker.Emma ≠ a Worker.Fiona) →
    ((condition1 a ∧ ¬condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ ¬condition2 a ∧ condition3 a)) →
    a = correct_assignment :=
  sorry

end NUMINAMATH_CALUDE_factory_assignment_l1692_169277


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1692_169256

-- Define the quadratic expression
def quadratic (k x : ℝ) : ℝ := x^2 - (k - 4)*x - k + 7

-- State the theorem
theorem quadratic_always_positive (k : ℝ) :
  (∀ x, quadratic k x > 0) ↔ k > -2 ∧ k < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1692_169256


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l1692_169273

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement to prove
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l1692_169273


namespace NUMINAMATH_CALUDE_inequality_proof_l1692_169217

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  (1/(1-a)) + (1/(1-b)) ≥ 4 ∧ ((1/(1-a)) + (1/(1-b)) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1692_169217


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1692_169292

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 3| - |x - 1| = 4*x - 3 :=
by
  -- The unique solution is 7/3
  use 7/3
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1692_169292


namespace NUMINAMATH_CALUDE_no_equal_partition_for_2002_l1692_169271

theorem no_equal_partition_for_2002 :
  ¬ ∃ (S : Finset ℕ),
    S ⊆ Finset.range 2003 ∧
    S.sum id = ((Finset.range 2003).sum id) / 2 :=
by sorry

end NUMINAMATH_CALUDE_no_equal_partition_for_2002_l1692_169271


namespace NUMINAMATH_CALUDE_nellie_legos_l1692_169214

theorem nellie_legos (L : ℕ) : 
  L - 57 - 24 = 299 → L = 380 := by
sorry

end NUMINAMATH_CALUDE_nellie_legos_l1692_169214


namespace NUMINAMATH_CALUDE_problem_solution_l1692_169234

theorem problem_solution : 
  (1 + 3/4 - 3/8 + 5/6) / (-1/24) = -53 ∧ 
  -2^2 + (-4) / 2 * (1/2) + |(-3)| = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1692_169234


namespace NUMINAMATH_CALUDE_smaller_field_area_l1692_169209

/-- Given a field of 500 hectares divided into two parts, where the difference
    of the areas is one-fifth of their average, the area of the smaller part
    is 225 hectares. -/
theorem smaller_field_area (x y : ℝ) (h1 : x + y = 500)
    (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 := by
  sorry

end NUMINAMATH_CALUDE_smaller_field_area_l1692_169209


namespace NUMINAMATH_CALUDE_message_reconstruction_existence_l1692_169261

/-- Represents a text as a list of characters -/
def Text := List Char

/-- Represents a permutation of characters -/
def Permutation := Char → Char

/-- Represents a substitution of characters -/
def Substitution := Char → Char

/-- Apply a permutation to a text -/
def applyPermutation (p : Permutation) (t : Text) : Text :=
  t.map p

/-- Apply a substitution to a text -/
def applySubstitution (s : Substitution) (t : Text) : Text :=
  t.map s

/-- Check if a substitution is bijective -/
def isBijectiveSubstitution (s : Substitution) : Prop :=
  Function.Injective s ∧ Function.Surjective s

theorem message_reconstruction_existence :
  ∃ (original : Text) (p : Permutation) (s : Substitution),
    let text1 := "МИМОПРАСТЕТИРАСИСПДАИСАФЕИИБОЕТКЖРГЛЕОЛОИШИСАННСЙСАООЛТЛЕЯТУИЦВЫИПИЯДПИЩПЬПСЕЮЯ".data
    let text2 := "УЩФМШПДРЕЦЧЕШЮЧДАКЕЧМДВКШБЕЕЧДФЭПЙЩГШФЩЦЕЮЩФПМЕЧПМРРМЕОЧХЕШРГИФРЯЯЛКДФФЕЕ".data
    applyPermutation p original = text1 ∧
    applySubstitution s original = text2 ∧
    isBijectiveSubstitution s ∧
    original = "ШЕСТАЯОЛИМПИАДАПОКРИПТОГРАФИИПОСВЯЩЕННАЯСЕМЬДЕСЯТИПЯТИЛЕТИЮСПЕЦИАЛЬНОЙСЛУЖБЫРОССИИ".data :=
by sorry


end NUMINAMATH_CALUDE_message_reconstruction_existence_l1692_169261


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1692_169230

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 = (1 : ℂ) + (Complex.I : ℂ) * 2 * a + b → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1692_169230


namespace NUMINAMATH_CALUDE_min_value_and_integral_bound_l1692_169247

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x * Real.log x

-- Define the integral G
def G (a b : ℝ) : ℝ := ∫ x in a..b, |Real.log x - Real.log ((a + b) / 2)|

-- State the theorem
theorem min_value_and_integral_bound 
  (k : ℝ) (a b : ℝ) (h1 : k ≠ 0) (h2 : 0 < a) (h3 : a < b) :
  (∃ (x : ℝ), f k x = -1 / Real.exp 1 ∧ ∀ (y : ℝ), f k y ≥ -1 / Real.exp 1) →
  (k = 1 ∧ 
   G a b = a * Real.log a + b * Real.log b - (a + b) * Real.log ((a + b) / 2) ∧
   G a b / (b - a) < Real.log 2) := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_and_integral_bound_l1692_169247


namespace NUMINAMATH_CALUDE_ratio_problem_l1692_169204

theorem ratio_problem (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1692_169204


namespace NUMINAMATH_CALUDE_charm_cost_calculation_l1692_169255

/-- The cost of a single charm used in Tim's necklace business -/
def charm_cost : ℚ := 15

/-- The number of charms used in each necklace -/
def charms_per_necklace : ℕ := 10

/-- The selling price of each necklace -/
def necklace_price : ℚ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit from selling 30 necklaces -/
def total_profit : ℚ := 1500

theorem charm_cost_calculation :
  charm_cost * (charms_per_necklace : ℚ) * (necklaces_sold : ℚ) =
  necklace_price * (necklaces_sold : ℚ) - total_profit := by sorry

end NUMINAMATH_CALUDE_charm_cost_calculation_l1692_169255


namespace NUMINAMATH_CALUDE_quadratic_not_in_third_quadrant_l1692_169291

/-- A linear function passing through the first, third, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  passes_first_quadrant : ∃ x > 0, -a * x + b > 0
  passes_third_quadrant : ∃ x < 0, -a * x + b < 0
  passes_fourth_quadrant : ∃ x > 0, -a * x + b < 0

/-- The corresponding quadratic function -/
def quadratic_function (f : LinearFunction) (x : ℝ) : ℝ :=
  -f.a * x^2 + f.b * x

/-- Theorem stating that the quadratic function does not pass through the third quadrant -/
theorem quadratic_not_in_third_quadrant (f : LinearFunction) :
  ¬∃ x < 0, quadratic_function f x < 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_not_in_third_quadrant_l1692_169291


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1692_169248

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1692_169248


namespace NUMINAMATH_CALUDE_Q_on_circle_25_line_AB_equation_l1692_169208

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define that Q is outside circle P
def Q_outside_P (a b : ℝ) : Prop := a^2 + b^2 > 16

-- Define circle M with diameter PQ intersecting circle P at A and B
def circle_M_intersects_P (a b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧ 
  ((x1 - a)^2 + (y1 - b)^2 = (x1^2 + y1^2) / 4) ∧
  ((x2 - a)^2 + (y2 - b)^2 = (x2^2 + y2^2) / 4)

-- Theorem 1: When QA = QB = 3, Q lies on x^2 + y^2 = 25
theorem Q_on_circle_25 (a b : ℝ) 
  (h1 : Q_outside_P a b) 
  (h2 : circle_M_intersects_P a b) 
  (h3 : ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
        (x1 - a)^2 + (y1 - b)^2 = 9 ∧ (x2 - a)^2 + (y2 - b)^2 = 9) :
  a^2 + b^2 = 25 :=
sorry

-- Theorem 2: When Q(4, 6), the equation of line AB is 2x + 3y - 8 = 0
theorem line_AB_equation 
  (h1 : Q_outside_P 4 6) 
  (h2 : circle_M_intersects_P 4 6) :
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
  2 * x1 + 3 * y1 - 8 = 0 ∧ 2 * x2 + 3 * y2 - 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_Q_on_circle_25_line_AB_equation_l1692_169208


namespace NUMINAMATH_CALUDE_classroom_tables_l1692_169281

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students in base 7 notation -/
def studentsBase7 : Nat := 321

/-- The number of students per table -/
def studentsPerTable : Nat := 3

/-- Theorem: The number of tables in the classroom is 54 -/
theorem classroom_tables :
  (base7ToBase10 studentsBase7) / studentsPerTable = 54 := by
  sorry


end NUMINAMATH_CALUDE_classroom_tables_l1692_169281


namespace NUMINAMATH_CALUDE_pool_capacity_l1692_169267

theorem pool_capacity (C : ℝ) 
  (h1 : 0.4 * C + 300 = 0.7 * C)  -- Adding 300 gallons fills to 70%
  (h2 : 300 = 0.3 * (0.4 * C))    -- 300 gallons is a 30% increase
  : C = 1000 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l1692_169267


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_equals_zero_l1692_169289

theorem cos_ninety_degrees_equals_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_equals_zero_l1692_169289


namespace NUMINAMATH_CALUDE_integral_x_power_five_minus_one_to_one_equals_zero_l1692_169266

theorem integral_x_power_five_minus_one_to_one_equals_zero :
  ∫ x in (-1)..1, x^5 = 0 := by sorry

end NUMINAMATH_CALUDE_integral_x_power_five_minus_one_to_one_equals_zero_l1692_169266


namespace NUMINAMATH_CALUDE_spending_ratio_l1692_169216

def initial_amount : ℕ := 200
def spent_on_books : ℕ := 30
def spent_on_clothes : ℕ := 55
def spent_on_snacks : ℕ := 25
def spent_on_gift : ℕ := 20
def spent_on_electronics : ℕ := 40

def total_spent : ℕ := spent_on_books + spent_on_clothes + spent_on_snacks + spent_on_gift + spent_on_electronics
def unspent : ℕ := initial_amount - total_spent

theorem spending_ratio : 
  (total_spent : ℚ) / (unspent : ℚ) = 17 / 3 := by sorry

end NUMINAMATH_CALUDE_spending_ratio_l1692_169216


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1692_169242

theorem right_triangle_perimeter : ∃ (a c : ℕ), 
  11^2 + a^2 = c^2 ∧ 11 + a + c = 132 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1692_169242


namespace NUMINAMATH_CALUDE_intersection_of_M_and_S_l1692_169201

-- Define the set M
def M : Set ℕ := {x | 0 < x ∧ x < 4}

-- Define the set S
def S : Set ℕ := {2, 3, 5}

-- Theorem statement
theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_S_l1692_169201


namespace NUMINAMATH_CALUDE_hens_not_laying_eggs_l1692_169295

theorem hens_not_laying_eggs 
  (total_chickens : ℕ)
  (roosters : ℕ)
  (eggs_per_hen : ℕ)
  (total_eggs : ℕ)
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : eggs_per_hen = 3)
  (h4 : total_eggs = 1158) :
  total_chickens - roosters - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_hens_not_laying_eggs_l1692_169295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1692_169299

/-- Given an arithmetic sequence {a_n}, if a₂ + 4a₇ + a₁₂ = 96, then 2a₃ + a₁₅ = 48 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + 4 * a 7 + a 12 = 96 →                       -- given condition
  2 * a 3 + a 15 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1692_169299


namespace NUMINAMATH_CALUDE_polynomial_properties_l1692_169218

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

-- Define the polynomial equality
def poly_eq (x : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- Theorem statement
theorem polynomial_properties :
  (∀ x, poly_eq a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1692_169218


namespace NUMINAMATH_CALUDE_ghee_mixture_proof_l1692_169225

/-- Proves that adding 20 kg of pure ghee to a 30 kg mixture of 50% pure ghee and 50% vanaspati
    results in a mixture where vanaspati constitutes 30% of the total. -/
theorem ghee_mixture_proof (original_quantity : ℝ) (pure_ghee_added : ℝ) 
  (h1 : original_quantity = 30)
  (h2 : pure_ghee_added = 20) : 
  let initial_vanaspati := 0.5 * original_quantity
  let total_after_addition := original_quantity + pure_ghee_added
  initial_vanaspati / total_after_addition = 0.3 := by
  sorry

#check ghee_mixture_proof

end NUMINAMATH_CALUDE_ghee_mixture_proof_l1692_169225


namespace NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l1692_169279

theorem z_squared_in_second_quadrant :
  let z : ℂ := Complex.exp (75 * π / 180 * Complex.I)
  (z^2).re < 0 ∧ (z^2).im > 0 :=
by sorry

end NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l1692_169279


namespace NUMINAMATH_CALUDE_island_area_l1692_169283

/-- The area of a rectangular island with width 5 miles and length 10 miles is 50 square miles. -/
theorem island_area : 
  let width : ℝ := 5
  let length : ℝ := 10
  width * length = 50 := by sorry

end NUMINAMATH_CALUDE_island_area_l1692_169283


namespace NUMINAMATH_CALUDE_sum_to_135_mod_7_l1692_169213

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the sum of integers from 1 to 135, when divided by 7, has a remainder of 3 -/
theorem sum_to_135_mod_7 : sum_to 135 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_to_135_mod_7_l1692_169213


namespace NUMINAMATH_CALUDE_f_properties_l1692_169254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) ^ (a * x^2 - 4*x + 3)

theorem f_properties :
  (∀ x > 2, ∀ y > x, f 1 y < f 1 x) ∧
  (∃ x, f 1 x = 2 → 1 = 1) ∧
  (∀ a, (∀ x < 2, ∀ y < x, f a y < f a x) → 0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1692_169254


namespace NUMINAMATH_CALUDE_identity_value_l1692_169220

theorem identity_value (a b c : ℝ) (m n : ℤ) :
  (∀ x : ℝ, (x^n + c)^m = (a*x^m + 1)*(b*x^m + 1)) →
  |a + b + c| = 3 :=
by sorry

end NUMINAMATH_CALUDE_identity_value_l1692_169220


namespace NUMINAMATH_CALUDE_book_selling_price_l1692_169286

theorem book_selling_price (cost_price : ℚ) 
  (h1 : cost_price * (1 + 1/10) = 550) 
  (h2 : ∃ original_price : ℚ, original_price = cost_price * (1 - 1/10)) : 
  ∃ original_price : ℚ, original_price = 450 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l1692_169286


namespace NUMINAMATH_CALUDE_nearest_integer_to_two_plus_sqrt_three_fourth_l1692_169264

theorem nearest_integer_to_two_plus_sqrt_three_fourth (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℤ), n = 194 ∧ |((2 : ℝ) + Real.sqrt 3)^4 - (n : ℝ)| < (1/2 : ℝ) + ε :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_two_plus_sqrt_three_fourth_l1692_169264


namespace NUMINAMATH_CALUDE_eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l1692_169251

theorem eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3 :
  8 * (Real.cos (25 * π / 180))^2 - Real.tan (40 * π / 180) - 4 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_cos_squared_25_minus_tan_40_minus_4_equals_sqrt_3_l1692_169251


namespace NUMINAMATH_CALUDE_parabola_intersection_circle_radius_squared_l1692_169211

theorem parabola_intersection_circle_radius_squared (x y : ℝ) : 
  y = (x - 2)^2 ∧ x + 6 = (y - 5)^2 → 
  (x - 5/2)^2 + (y - 9/2)^2 = 83/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_circle_radius_squared_l1692_169211


namespace NUMINAMATH_CALUDE_jose_join_time_l1692_169260

/-- Represents the problem of determining when Jose joined Tom's business --/
theorem jose_join_time (tom_investment jose_investment total_profit jose_profit : ℚ) 
  (h1 : tom_investment = 3000)
  (h2 : jose_investment = 4500)
  (h3 : total_profit = 5400)
  (h4 : jose_profit = 3000) :
  let x := (12 * tom_investment * (total_profit - jose_profit)) / 
           (jose_investment * jose_profit) - 12
  x = 2 := by sorry

end NUMINAMATH_CALUDE_jose_join_time_l1692_169260


namespace NUMINAMATH_CALUDE_square_area_problem_l1692_169297

theorem square_area_problem (x : ℝ) (h : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l1692_169297


namespace NUMINAMATH_CALUDE_parabola_translation_l1692_169288

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola := ⟨1, 0, 0⟩

/-- The final parabola y = (x+4)^2 - 5 -/
def final_parabola : Parabola := ⟨1, 8, 11⟩

/-- Translate a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b - 2 * p.a * d, p.a * d^2 + p.b * d + p.c⟩

/-- Translate a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + d⟩

/-- Theorem: The final parabola can be obtained by translating the standard parabola
    4 units to the left and then 5 units downward -/
theorem parabola_translation :
  translate_vertical (translate_horizontal standard_parabola (-4)) (-5) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1692_169288


namespace NUMINAMATH_CALUDE_triangle_with_lattice_point_is_equilateral_l1692_169246

/-- A triangle in a plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := sorry

/-- Whether a point is a lattice point --/
def is_lattice_point (p : ℝ × ℝ) : Prop := sorry

/-- Whether a point is on or inside a triangle --/
def point_in_triangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Whether two triangles are congruent --/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Whether a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop := sorry

theorem triangle_with_lattice_point_is_equilateral (t : Triangle) :
  perimeter t = 3 + 2 * Real.sqrt 3 →
  (∀ t' : Triangle, congruent t t' → ∃ p : ℝ × ℝ, is_lattice_point p ∧ point_in_triangle p t') →
  is_equilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_with_lattice_point_is_equilateral_l1692_169246


namespace NUMINAMATH_CALUDE_total_salary_proof_l1692_169237

def salary_B : ℝ := 232

def salary_A : ℝ := 1.5 * salary_B

def total_salary : ℝ := salary_A + salary_B

theorem total_salary_proof : total_salary = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_proof_l1692_169237


namespace NUMINAMATH_CALUDE_area_product_eq_volume_squared_l1692_169249

/-- Represents a rectangular box with dimensions x, y, and z, and diagonal d. -/
structure RectBox where
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ
  h_positive : x > 0 ∧ y > 0 ∧ z > 0
  h_diagonal : d^2 = x^2 + y^2 + z^2

/-- The volume of a rectangular box. -/
def volume (box : RectBox) : ℝ := box.x * box.y * box.z

/-- The product of the areas of the bottom, side, and front of a rectangular box. -/
def areaProduct (box : RectBox) : ℝ := (box.x * box.y) * (box.y * box.z) * (box.z * box.x)

/-- Theorem stating that the product of the areas is equal to the square of the volume. -/
theorem area_product_eq_volume_squared (box : RectBox) :
  areaProduct box = (volume box)^2 := by sorry

end NUMINAMATH_CALUDE_area_product_eq_volume_squared_l1692_169249


namespace NUMINAMATH_CALUDE_satellite_survey_is_census_l1692_169212

/-- Represents a survey type -/
inductive SurveyType
| Sample
| Census

/-- Represents a survey option -/
structure SurveyOption where
  description : String
  type : SurveyType

/-- Determines if a survey option is suitable for a census -/
def isSuitableForCensus (survey : SurveyOption) : Prop :=
  survey.type = SurveyType.Census

/-- The satellite component quality survey -/
def satelliteComponentSurvey : SurveyOption :=
  { description := "Investigating the quality of components of the satellite \"Zhangheng-1\""
    type := SurveyType.Census }

/-- Theorem stating that the satellite component survey is suitable for a census -/
theorem satellite_survey_is_census : 
  isSuitableForCensus satelliteComponentSurvey := by
  sorry


end NUMINAMATH_CALUDE_satellite_survey_is_census_l1692_169212


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1692_169215

open Real

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = -f x)

-- State the theorem
theorem odd_periodic_function_property 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h_f : is_odd_and_periodic f) 
  (h_α : tan α = 2) : 
  f (15 * sin α * cos α) = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1692_169215


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1692_169239

def numbers : List Nat := [55, 63, 85, 94, 133]

def has_largest_prime_factor (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, ∃ p q : Nat, 
    Nat.Prime p ∧ 
    n = p * q ∧ 
    ∀ r s : Nat, (Nat.Prime r ∧ m = r * s) → r ≤ p

theorem largest_prime_factor : 
  has_largest_prime_factor 94 numbers := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1692_169239


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l1692_169241

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The conditions of the vegan restaurant menu problem -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 10 ∧
  m.beans_lentils = 2 ∧
  m.beans_seitan = 2 ∧
  m.only_beans = (m.total_dishes - m.beans_lentils - m.beans_seitan) / 2 ∧
  m.only_beans = 3 * m.only_seitan

/-- Theorem stating that the number of dishes including lentils is 2 -/
theorem lentil_dishes_count (m : VeganMenu) (h : menu_conditions m) : 
  m.beans_lentils + m.only_lentils = 2 := by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l1692_169241


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1692_169243

theorem complex_equation_solution (c d x : ℂ) : 
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1692_169243


namespace NUMINAMATH_CALUDE_molecular_weight_7_moles_AlOH3_l1692_169223

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of aluminum atoms in Al(OH)3 -/
def num_Al : ℕ := 1

/-- The number of oxygen atoms in Al(OH)3 -/
def num_O : ℕ := 3

/-- The number of hydrogen atoms in Al(OH)3 -/
def num_H : ℕ := 3

/-- The number of moles of Al(OH)3 -/
def num_moles : ℝ := 7

/-- The molecular weight of Al(OH)3 in g/mol -/
def molecular_weight_AlOH3 : ℝ :=
  num_Al * atomic_weight_Al + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_7_moles_AlOH3 :
  num_moles * molecular_weight_AlOH3 = 546.07 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_7_moles_AlOH3_l1692_169223


namespace NUMINAMATH_CALUDE_jacob_phoebe_age_fraction_l1692_169290

/-- Represents the ages and relationships of Rehana, Phoebe, and Jacob -/
structure AgeRelationship where
  rehana_current_age : ℕ
  jacob_current_age : ℕ
  years_until_comparison : ℕ
  rehana_phoebe_ratio : ℕ

/-- The fraction of Phoebe's age that Jacob's age represents -/
def age_fraction (ar : AgeRelationship) : ℚ :=
  ar.jacob_current_age / (ar.rehana_current_age + ar.years_until_comparison - ar.years_until_comparison * ar.rehana_phoebe_ratio)

/-- Theorem stating that given the conditions, Jacob's age is 3/5 of Phoebe's age -/
theorem jacob_phoebe_age_fraction :
  ∀ (ar : AgeRelationship),
  ar.rehana_current_age = 25 →
  ar.jacob_current_age = 3 →
  ar.years_until_comparison = 5 →
  ar.rehana_phoebe_ratio = 3 →
  age_fraction ar = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_jacob_phoebe_age_fraction_l1692_169290


namespace NUMINAMATH_CALUDE_journey_distance_l1692_169207

/-- Represents the man's journey with different speeds and times for each segment -/
structure Journey where
  flat_walk_speed : ℝ
  downhill_run_speed : ℝ
  hilly_walk_speed : ℝ
  hilly_run_speed : ℝ
  flat_walk_time : ℝ
  downhill_run_time : ℝ
  hilly_walk_time : ℝ
  hilly_run_time : ℝ

/-- Calculates the total distance traveled during the journey -/
def total_distance (j : Journey) : ℝ :=
  j.flat_walk_speed * j.flat_walk_time +
  j.downhill_run_speed * j.downhill_run_time +
  j.hilly_walk_speed * j.hilly_walk_time +
  j.hilly_run_speed * j.hilly_run_time

/-- Theorem stating that the total distance traveled is 90 km -/
theorem journey_distance :
  let j : Journey := {
    flat_walk_speed := 8,
    downhill_run_speed := 24,
    hilly_walk_speed := 6,
    hilly_run_speed := 18,
    flat_walk_time := 3,
    downhill_run_time := 1.5,
    hilly_walk_time := 2,
    hilly_run_time := 1
  }
  total_distance j = 90 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l1692_169207


namespace NUMINAMATH_CALUDE_remainder_of_3_600_mod_19_l1692_169244

theorem remainder_of_3_600_mod_19 : 3^600 % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_600_mod_19_l1692_169244


namespace NUMINAMATH_CALUDE_sequence_theorem_l1692_169274

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0) ∧ (∀ n : ℕ+, a (n + 1) + 1 / (a n) < 2)

theorem sequence_theorem (a : ℕ+ → ℝ) (h : sequence_property a) :
  (∀ n : ℕ+, a (n + 2) < a (n + 1) ∧ a (n + 1) < 2) ∧
  (∀ n : ℕ+, a n > 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l1692_169274


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1692_169202

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The length of the major axis is twice the length of the minor axis -/
  major_twice_minor : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (2, -6) -/
  passes_through_2_neg6 : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (3, 0) -/
  passes_through_3_0 : ℝ → ℝ → Prop
  /-- The eccentricity of the ellipse is √6/3 -/
  eccentricity_sqrt6_div_3 : ℝ → ℝ → Prop

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating that a SpecialEllipse satisfies one of two standard equations -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, standard_equation 3 (Real.sqrt 3) x y) ∨
  (∀ x y, standard_equation (Real.sqrt 27) 3 x y) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1692_169202


namespace NUMINAMATH_CALUDE_non_student_ticket_cost_l1692_169272

theorem non_student_ticket_cost
  (total_tickets : ℕ)
  (student_ticket_cost : ℚ)
  (total_amount : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 193)
  (h2 : student_ticket_cost = 1/2)
  (h3 : total_amount = 412/2)
  (h4 : student_tickets = 83) :
  (total_amount - student_ticket_cost * student_tickets) / (total_tickets - student_tickets) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_non_student_ticket_cost_l1692_169272


namespace NUMINAMATH_CALUDE_fence_perimeter_l1692_169253

/-- The number of posts in the fence -/
def total_posts : ℕ := 24

/-- The width of each post in inches -/
def post_width : ℚ := 5

/-- The space between adjacent posts in feet -/
def post_spacing : ℚ := 6

/-- The number of posts on each side of the square fence -/
def posts_per_side : ℕ := 7

/-- The length of one side of the square fence in feet -/
def side_length : ℚ := post_spacing * 6 + posts_per_side * (post_width / 12)

/-- The outer perimeter of the square fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 156 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l1692_169253


namespace NUMINAMATH_CALUDE_distance_between_points_on_parabola_l1692_169268

/-- The distance between two points on a parabola -/
theorem distance_between_points_on_parabola
  (a b c x₁ x₂ : ℝ) :
  let y₁ := a * x₁^2 + b * x₁ + c
  let y₂ := a * x₂^2 + b * x₂ + c
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = |x₂ - x₁| * Real.sqrt (1 + (a * (x₂ + x₁) + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_parabola_l1692_169268


namespace NUMINAMATH_CALUDE_fourth_ball_black_probability_l1692_169282

theorem fourth_ball_black_probability 
  (total_balls : Nat) 
  (black_balls : Nat) 
  (red_balls : Nat) 
  (h1 : total_balls = black_balls + red_balls)
  (h2 : black_balls = 4)
  (h3 : red_balls = 4) :
  (black_balls : ℚ) / total_balls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_ball_black_probability_l1692_169282


namespace NUMINAMATH_CALUDE_sallys_number_l1692_169228

theorem sallys_number (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∀ d : ℕ, 2 ≤ d ∧ d ≤ 9 → n % d = 1) ↔ 
  (n = 2521 ∨ n = 5041 ∨ n = 7561) :=
sorry

end NUMINAMATH_CALUDE_sallys_number_l1692_169228


namespace NUMINAMATH_CALUDE_orange_cost_l1692_169263

/-- Given Alexander's shopping scenario, prove the cost of each orange. -/
theorem orange_cost (apple_price : ℝ) (apple_count : ℕ) (orange_count : ℕ) (total_spent : ℝ) :
  apple_price = 1 →
  apple_count = 5 →
  orange_count = 2 →
  total_spent = 9 →
  (total_spent - apple_price * apple_count) / orange_count = 2 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_l1692_169263


namespace NUMINAMATH_CALUDE_evaluate_expression_l1692_169227

theorem evaluate_expression (x y : ℤ) (hx : x = 5) (hy : y = -3) :
  y * (y - 2 * x + 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1692_169227


namespace NUMINAMATH_CALUDE_smallest_cube_ending_528_l1692_169259

theorem smallest_cube_ending_528 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 528 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 528 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_528_l1692_169259


namespace NUMINAMATH_CALUDE_circle_radius_difference_l1692_169265

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 13^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9^2}
def P : ℝ × ℝ := (5, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_radius_difference (k : ℝ) : 
  P ∈ larger_circle ∧ 
  S k ∈ smaller_circle ∧
  (13 : ℝ) - 9 = 4 →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_circle_radius_difference_l1692_169265


namespace NUMINAMATH_CALUDE_min_digits_to_remove_l1692_169236

def original_number : ℕ := 123454321

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def remove_digits (n : ℕ) (indices : List ℕ) : ℕ :=
  let digits := digits n
  let new_digits := (List.enum digits).filter (λ (i, _) => ¬ indices.contains i)
  new_digits.foldl (λ acc (_, d) => acc * 10 + d) 0

theorem min_digits_to_remove :
  ∃ (indices : List ℕ),
    indices.length = 2 ∧
    is_divisible_by_9 (remove_digits original_number indices) ∧
    ∀ (other_indices : List ℕ),
      other_indices.length < 2 →
      ¬ is_divisible_by_9 (remove_digits original_number other_indices) :=
by sorry

end NUMINAMATH_CALUDE_min_digits_to_remove_l1692_169236


namespace NUMINAMATH_CALUDE_event_probability_range_l1692_169287

/-- The probability of event A occurring in a single trial -/
def p : ℝ := sorry

/-- The number of independent trials -/
def n : ℕ := 4

/-- The probability of event A occurring exactly k times in n trials -/
def prob_k (k : ℕ) : ℝ := sorry

theorem event_probability_range :
  (0 ≤ p ∧ p ≤ 1) →  -- Probability is between 0 and 1
  (prob_k 1 ≤ prob_k 2) →  -- Probability of occurring once ≤ probability of occurring twice
  (2/5 ≤ p ∧ p ≤ 1) :=  -- The range of probability p is [2/5, 1]
sorry

end NUMINAMATH_CALUDE_event_probability_range_l1692_169287


namespace NUMINAMATH_CALUDE_third_cube_edge_l1692_169222

-- Define the cube volume function
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Define the given cubes
def cube1_edge : ℝ := 4
def cube2_edge : ℝ := 5
def final_cube_edge : ℝ := 6

-- Theorem statement
theorem third_cube_edge :
  ∃ (third_edge : ℝ),
    cube_volume third_edge + cube_volume cube1_edge + cube_volume cube2_edge
    = cube_volume final_cube_edge ∧ third_edge = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_cube_edge_l1692_169222


namespace NUMINAMATH_CALUDE_factors_72_l1692_169293

/-- The number of distinct positive factors of 72 -/
def num_factors_72 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 72 is 12 -/
theorem factors_72 : num_factors_72 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_72_l1692_169293


namespace NUMINAMATH_CALUDE_friends_basketball_score_l1692_169294

theorem friends_basketball_score 
  (total_points : ℕ) 
  (edward_points : ℕ) 
  (h1 : total_points = 13) 
  (h2 : edward_points = 7) : 
  total_points - edward_points = 6 := by
  sorry

end NUMINAMATH_CALUDE_friends_basketball_score_l1692_169294


namespace NUMINAMATH_CALUDE_family_size_l1692_169257

def total_spent : ℕ := 119
def adult_ticket_price : ℕ := 21
def child_ticket_price : ℕ := 14
def adult_tickets_purchased : ℕ := 4

theorem family_size :
  ∃ (child_tickets : ℕ),
    adult_tickets_purchased * adult_ticket_price + child_tickets * child_ticket_price = total_spent ∧
    adult_tickets_purchased + child_tickets = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_family_size_l1692_169257


namespace NUMINAMATH_CALUDE_power_of_two_multiplication_l1692_169231

theorem power_of_two_multiplication : 2^4 * 2^4 * 2^4 = 2^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_multiplication_l1692_169231
