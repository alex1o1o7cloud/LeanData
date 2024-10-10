import Mathlib

namespace oranges_in_box_l2403_240373

/-- The number of oranges Jonathan takes from the box -/
def oranges_taken : ℕ := 45

/-- The number of oranges left in the box after Jonathan takes some -/
def oranges_left : ℕ := 51

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem oranges_in_box : initial_oranges = 96 := by
  sorry

end oranges_in_box_l2403_240373


namespace repeated_roots_coincide_l2403_240387

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial has a repeated root -/
def has_repeated_root (p : QuadraticPolynomial) : Prop :=
  ∃ r : ℝ, p.eval r = 0 ∧ (∀ x : ℝ, p.eval x = p.a * (x - r)^2)

/-- The sum of two quadratic polynomials -/
def add_poly (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

/-- Theorem: If P and Q are quadratic polynomials with repeated roots, 
    and P + Q also has a repeated root, then all these roots are equal -/
theorem repeated_roots_coincide (P Q : QuadraticPolynomial) 
  (hP : has_repeated_root P) 
  (hQ : has_repeated_root Q) 
  (hPQ : has_repeated_root (add_poly P Q)) : 
  ∃ r : ℝ, (∀ x : ℝ, P.eval x = P.a * (x - r)^2) ∧ 
            (∀ x : ℝ, Q.eval x = Q.a * (x - r)^2) ∧ 
            (∀ x : ℝ, (add_poly P Q).eval x = (P.a + Q.a) * (x - r)^2) := by
  sorry


end repeated_roots_coincide_l2403_240387


namespace base_n_representation_l2403_240319

theorem base_n_representation (n : ℕ) : 
  n > 0 ∧ 
  (∃ a b c : ℕ, 
    a < n ∧ b < n ∧ c < n ∧ 
    1998 = a * n^2 + b * n + c ∧ 
    a + b + c = 24) → 
  n = 15 ∨ n = 22 ∨ n = 43 := by
sorry

end base_n_representation_l2403_240319


namespace product_digit_sum_l2403_240334

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_digit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem product_digit_sum (n : ℕ) : 
  n ≥ 1 → digit_sum (5 * repeated_digit 5 n) ≥ 500 ↔ n ≥ 72 := by sorry

end product_digit_sum_l2403_240334


namespace perpendicular_vectors_l2403_240380

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (6, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If vectors a and b(k) are perpendicular, then k = 9 -/
theorem perpendicular_vectors (k : ℝ) : 
  dot_product a (b k) = 0 → k = 9 := by
  sorry

end perpendicular_vectors_l2403_240380


namespace stream_current_speed_l2403_240393

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RowerScenario where
  distance : ℝ
  rower_speed : ℝ
  current_speed : ℝ
  time_diff : ℝ

/-- Represents the scenario when the rower increases their speed -/
structure IncreasedSpeedScenario extends RowerScenario where
  speed_increase : ℝ
  new_time_diff : ℝ

/-- The theorem stating the speed of the stream's current given the conditions -/
theorem stream_current_speed 
  (scenario : RowerScenario)
  (increased : IncreasedSpeedScenario)
  (h1 : scenario.distance = 18)
  (h2 : scenario.time_diff = 4)
  (h3 : increased.speed_increase = 0.5)
  (h4 : increased.new_time_diff = 2)
  (h5 : scenario.distance / (scenario.rower_speed + scenario.current_speed) + scenario.time_diff = 
        scenario.distance / (scenario.rower_speed - scenario.current_speed))
  (h6 : scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed + scenario.current_speed) + 
        increased.new_time_diff = 
        scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed - scenario.current_speed))
  : scenario.current_speed = 2.5 := by
  sorry

end stream_current_speed_l2403_240393


namespace jenny_bus_time_l2403_240390

/-- Represents the schedule of Jenny's day --/
structure Schedule where
  wakeUpTime : Nat  -- in minutes after midnight
  busToSchoolTime : Nat  -- in minutes after midnight
  numClasses : Nat
  classDuration : Nat  -- in minutes
  lunchDuration : Nat  -- in minutes
  extracurricularDuration : Nat  -- in minutes
  busHomeTime : Nat  -- in minutes after midnight

/-- Calculates the time Jenny spent on the bus given her schedule --/
def timeBusSpent (s : Schedule) : Nat :=
  (s.busHomeTime - s.busToSchoolTime) - 
  (s.numClasses * s.classDuration + s.lunchDuration + s.extracurricularDuration)

/-- Jenny's actual schedule --/
def jennySchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busToSchoolTime := 8 * 60
    numClasses := 5
    classDuration := 45
    lunchDuration := 45
    extracurricularDuration := 90
    busHomeTime := 17 * 60 }

theorem jenny_bus_time : timeBusSpent jennySchedule = 180 := by
  sorry

end jenny_bus_time_l2403_240390


namespace coin_problem_l2403_240347

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

def total_coins : ℕ := 13
def total_value : ℕ := 163

theorem coin_problem (pennies nickels dimes quarters half_dollars : ℕ) 
  (h1 : pennies + nickels + dimes + quarters + half_dollars = total_coins)
  (h2 : pennies * penny_value + nickels * nickel_value + dimes * dime_value + 
        quarters * quarter_value + half_dollars * half_dollar_value = total_value)
  (h3 : pennies ≥ 1)
  (h4 : nickels ≥ 1)
  (h5 : dimes ≥ 1)
  (h6 : quarters ≥ 1)
  (h7 : half_dollars ≥ 1) :
  dimes = 3 := by
sorry

end coin_problem_l2403_240347


namespace optimal_method_is_random_then_stratified_l2403_240365

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Stratified
  | RandomThenStratified
  | StratifiedThenRandom

/-- Represents a school with first-year classes -/
structure School where
  num_classes : Nat
  male_female_ratio : Real

/-- Represents the sampling scenario -/
structure SamplingScenario where
  school : School
  num_classes_to_sample : Nat

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that the optimal sampling method for the given scenario
    is to use random sampling first, then stratified sampling -/
theorem optimal_method_is_random_then_stratified 
  (scenario : SamplingScenario) 
  (h1 : scenario.school.num_classes = 16) 
  (h2 : scenario.num_classes_to_sample = 2) :
  optimal_sampling_method scenario = SamplingMethod.RandomThenStratified :=
sorry

end optimal_method_is_random_then_stratified_l2403_240365


namespace min_sum_squares_l2403_240371

def S : Set Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
              f ≠ g ∧ f ≠ h ∧
              g ≠ h)
  (in_set : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S)
  (sum_condition : e + f + g + h = 9) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 106 := by
  sorry

end min_sum_squares_l2403_240371


namespace ratio_in_specific_arithmetic_sequence_l2403_240391

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (a m b : ℝ) : Prop :=
  s 0 = a ∧ s 1 = m ∧ s 2 = b ∧ s 3 = 3*m

-- State the theorem
theorem ratio_in_specific_arithmetic_sequence (s : ℕ → ℝ) (a m b : ℝ) :
  is_arithmetic_sequence s → our_sequence s a m b → b / a = -2 :=
by sorry

end ratio_in_specific_arithmetic_sequence_l2403_240391


namespace forty_percent_of_number_l2403_240337

theorem forty_percent_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 := by
  sorry

end forty_percent_of_number_l2403_240337


namespace percentage_of_singles_l2403_240342

def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 2
def doubles : ℕ := 8

def non_singles : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_singles

theorem percentage_of_singles :
  (singles : ℚ) / total_hits * 100 = 73 := by sorry

end percentage_of_singles_l2403_240342


namespace largest_product_in_S_largest_product_is_attained_l2403_240399

def S : Set Int := {-8, -3, 0, 2, 4}

theorem largest_product_in_S (a b : Int) : 
  a ∈ S → b ∈ S → a * b ≤ 24 := by sorry

theorem largest_product_is_attained : 
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = 24 := by sorry

end largest_product_in_S_largest_product_is_attained_l2403_240399


namespace two_star_three_equals_one_l2403_240364

-- Define the ã — operation
def star_op (a b : ℤ) : ℤ := 2 * a - 3 * b + a * b

-- State the theorem
theorem two_star_three_equals_one :
  star_op 2 3 = 1 :=
by sorry

end two_star_three_equals_one_l2403_240364


namespace f_has_three_roots_l2403_240397

def f (x : ℝ) := x^3 - 64*x

theorem f_has_three_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry

end f_has_three_roots_l2403_240397


namespace total_trophies_after_seven_years_l2403_240370

def michael_initial_trophies : ℕ := 100
def michael_yearly_increase : ℕ := 200
def years : ℕ := 7
def jack_multiplier : ℕ := 20

def michael_final_trophies : ℕ := michael_initial_trophies + michael_yearly_increase * years
def jack_final_trophies : ℕ := jack_multiplier * michael_initial_trophies + michael_final_trophies

theorem total_trophies_after_seven_years :
  michael_final_trophies + jack_final_trophies = 5000 := by
  sorry

end total_trophies_after_seven_years_l2403_240370


namespace min_value_quadratic_equation_l2403_240376

theorem min_value_quadratic_equation (a b : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a*x + b - 3 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a'*x + b' - 3 = 0) → 
    a^2 + (b - 4)^2 ≤ a'^2 + (b' - 4)^2) →
  a^2 + (b - 4)^2 = 2 :=
by sorry

end min_value_quadratic_equation_l2403_240376


namespace total_vehicles_proof_l2403_240326

/-- The number of vehicles involved in accidents last year -/
def accidents : ℕ := 2000

/-- The number of vehicles per 100 million that are involved in accidents -/
def accident_rate : ℕ := 100

/-- The total number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

/-- Theorem stating that the total number of vehicles is correct given the accident rate and number of accidents -/
theorem total_vehicles_proof :
  accidents * (100000000 / accident_rate) = total_vehicles :=
sorry

end total_vehicles_proof_l2403_240326


namespace unit_digit_4137_pow_1289_l2403_240358

/-- The unit digit of a number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The unit digit pattern for powers of 7 repeats every 4 steps -/
def unitDigitPattern : Fin 4 → ℕ
  | 0 => 7
  | 1 => 9
  | 2 => 3
  | 3 => 1

theorem unit_digit_4137_pow_1289 :
  unitDigit ((4137 : ℕ) ^ 1289) = 7 := by
  sorry

end unit_digit_4137_pow_1289_l2403_240358


namespace car_travel_time_l2403_240375

theorem car_travel_time (distance : ℝ) (speed : ℝ) (time_ratio : ℝ) (initial_time : ℝ) : 
  distance = 324 →
  speed = 36 →
  time_ratio = 3 / 2 →
  distance = speed * (time_ratio * initial_time) →
  initial_time = 6 := by
sorry

end car_travel_time_l2403_240375


namespace prob_miss_at_least_once_prob_A_twice_B_once_l2403_240388

-- Define the probabilities of hitting the target
def prob_hit_A : ℚ := 2/3
def prob_hit_B : ℚ := 3/4

-- Define the number of shots for each part
def shots_part1 : ℕ := 3
def shots_part2 : ℕ := 2

-- Assume independence of shots
axiom independence : ∀ (n : ℕ), (prob_hit_A ^ n) = prob_hit_A * (prob_hit_A ^ (n - 1))

-- Part 1: Probability that Person A misses at least once in 3 shots
theorem prob_miss_at_least_once : 
  1 - (prob_hit_A ^ shots_part1) = 19/27 := by sorry

-- Part 2: Probability that A hits exactly twice and B hits exactly once in 2 shots each
theorem prob_A_twice_B_once :
  (prob_hit_A ^ 2) * (2 * prob_hit_B * (1 - prob_hit_B)) = 1/6 := by sorry

end prob_miss_at_least_once_prob_A_twice_B_once_l2403_240388


namespace f_of_3_equals_9_l2403_240303

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1) + 1

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end f_of_3_equals_9_l2403_240303


namespace polynomial_irreducibility_equivalence_l2403_240368

theorem polynomial_irreducibility_equivalence 
  (f : Polynomial ℤ) : 
  Irreducible f ↔ Irreducible (f.map (algebraMap ℤ ℚ)) :=
sorry

end polynomial_irreducibility_equivalence_l2403_240368


namespace complex_symmetry_l2403_240344

theorem complex_symmetry (z₁ z₂ : ℂ) : 
  (z₁ = 2 - 3*I) → (z₁ = -z₂) → (z₂ = -2 + 3*I) := by
  sorry

end complex_symmetry_l2403_240344


namespace perfect_square_discriminant_implies_rational_roots_rational_roots_implies_perfect_square_discriminant_all_odd_coefficients_no_rational_roots_l2403_240360

-- Define a structure for quadratic equations
structure QuadraticEquation where
  a : Int
  b : Int
  c : Int
  a_nonzero : a ≠ 0

-- Define the discriminant
def discriminant (eq : QuadraticEquation) : Int :=
  eq.b * eq.b - 4 * eq.a * eq.c

-- Define a perfect square
def is_perfect_square (n : Int) : Prop :=
  ∃ m : Int, n = m * m

-- Define rational roots
def has_rational_roots (eq : QuadraticEquation) : Prop :=
  ∃ p q : Int, q ≠ 0 ∧ eq.a * p * p + eq.b * p * q + eq.c * q * q = 0

-- Theorem 1
theorem perfect_square_discriminant_implies_rational_roots
  (eq : QuadraticEquation)
  (h : is_perfect_square (discriminant eq)) :
  has_rational_roots eq :=
sorry

-- Theorem 2
theorem rational_roots_implies_perfect_square_discriminant
  (eq : QuadraticEquation)
  (h : has_rational_roots eq) :
  is_perfect_square (discriminant eq) :=
sorry

-- Theorem 3
theorem all_odd_coefficients_no_rational_roots
  (eq : QuadraticEquation)
  (h1 : eq.a % 2 = 1)
  (h2 : eq.b % 2 = 1)
  (h3 : eq.c % 2 = 1) :
  ¬(has_rational_roots eq) :=
sorry

end perfect_square_discriminant_implies_rational_roots_rational_roots_implies_perfect_square_discriminant_all_odd_coefficients_no_rational_roots_l2403_240360


namespace equation_solution_l2403_240386

theorem equation_solution : ∃ r : ℚ, 23 - 5 = 3 * r + 2 ∧ r = 16 / 3 := by
  sorry

end equation_solution_l2403_240386


namespace discarded_number_proof_l2403_240328

theorem discarded_number_proof (numbers : Finset ℕ) (sum : ℕ) (x : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 50 →
  55 ∈ numbers →
  x ∈ numbers →
  x ≠ 55 →
  (sum - 55 - x) / 48 = 50 →
  x = 45 :=
by sorry

end discarded_number_proof_l2403_240328


namespace largest_three_digit_square_cube_l2403_240315

/-- The largest three-digit number that is both a perfect square and a perfect cube -/
def largest_square_cube : ℕ := 729

/-- A number is a three-digit number if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is a perfect square if there exists an integer whose square is that number -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is a perfect cube if there exists an integer whose cube is that number -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

theorem largest_three_digit_square_cube :
  is_three_digit largest_square_cube ∧
  is_perfect_square largest_square_cube ∧
  is_perfect_cube largest_square_cube ∧
  ∀ n : ℕ, is_three_digit n → is_perfect_square n → is_perfect_cube n → n ≤ largest_square_cube :=
sorry

end largest_three_digit_square_cube_l2403_240315


namespace librarian_shelves_l2403_240317

/-- The number of books on the top shelf -/
def first_term : ℕ := 3

/-- The difference in the number of books between each consecutive shelf -/
def common_difference : ℕ := 3

/-- The total number of books on all shelves -/
def total_books : ℕ := 225

/-- The number of shelves used by the librarian -/
def num_shelves : ℕ := 15

theorem librarian_shelves :
  ∃ (n : ℕ), n = num_shelves ∧
  n * (2 * first_term + (n - 1) * common_difference) = 2 * total_books :=
sorry

end librarian_shelves_l2403_240317


namespace train_length_problem_l2403_240305

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 18) : ∃ (train_length : ℝ), 
  train_length = 50 ∧ 
  train_length * 1000 = (faster_speed - slower_speed) * passing_time / 3600 := by
  sorry

end train_length_problem_l2403_240305


namespace kyle_age_l2403_240350

/-- Given the relationships between Kyle, Julian, Frederick, and Tyson's ages, prove Kyle's age. -/
theorem kyle_age (tyson_age : ℕ) (kyle_julian : ℕ) (julian_frederick : ℕ) (frederick_tyson : ℕ) :
  tyson_age = 20 →
  kyle_julian = 5 →
  julian_frederick = 20 →
  frederick_tyson = 2 →
  tyson_age * frederick_tyson - julian_frederick + kyle_julian = 25 :=
by sorry

end kyle_age_l2403_240350


namespace least_integer_with_leading_six_and_fraction_l2403_240301

theorem least_integer_with_leading_six_and_fraction (x : ℕ) : x ≥ 625 →
  (∃ n : ℕ, ∃ y : ℕ, 
    x = 6 * 10^n + y ∧ 
    y < 10^n ∧ 
    y = x / 25) →
  x = 625 :=
sorry

end least_integer_with_leading_six_and_fraction_l2403_240301


namespace min_time_for_all_flickers_l2403_240302

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time taken for one flicker (in seconds) -/
def flicker_time : ℕ := 5

/-- The interval time between flickers (in seconds) -/
def interval_time : ℕ := 5

/-- The total number of possible flickers -/
def total_flickers : ℕ := Nat.factorial num_lights

theorem min_time_for_all_flickers :
  (total_flickers * flicker_time) + ((total_flickers - 1) * interval_time) = 1195 := by
  sorry

end min_time_for_all_flickers_l2403_240302


namespace complement_of_A_l2403_240324

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 2}

-- Statement to prove
theorem complement_of_A (x : Nat) : x ∈ (U \ A) ↔ x = 3 ∨ x = 4 := by
  sorry

end complement_of_A_l2403_240324


namespace frank_meets_quota_l2403_240336

/-- Represents the sales data for Frank's car sales challenge -/
structure SalesData where
  quota : Nat
  days : Nat
  first3DaysSales : Nat
  next4DaysSales : Nat
  bonusCars : Nat
  remainingInventory : Nat
  oddDaySales : Nat
  evenDaySales : Nat

/-- Calculates the total sales and remaining inventory based on the given sales data -/
def calculateSales (data : SalesData) : (Nat × Nat) :=
  let initialSales := data.first3DaysSales * 3 + data.next4DaysSales * 4 + data.bonusCars
  let remainingDays := data.days - 7
  let oddDays := remainingDays / 2
  let evenDays := remainingDays - oddDays
  let potentialRemainingDaySales := data.oddDaySales * oddDays + data.evenDaySales * evenDays
  let actualRemainingDaySales := min potentialRemainingDaySales data.remainingInventory
  let totalSales := min (initialSales + actualRemainingDaySales) data.quota
  let remainingInventory := data.remainingInventory - (totalSales - initialSales)
  (totalSales, remainingInventory)

/-- Theorem stating that Frank will meet his quota and have 22 cars left in inventory -/
theorem frank_meets_quota (data : SalesData)
  (h1 : data.quota = 50)
  (h2 : data.days = 30)
  (h3 : data.first3DaysSales = 5)
  (h4 : data.next4DaysSales = 3)
  (h5 : data.bonusCars = 5)
  (h6 : data.remainingInventory = 40)
  (h7 : data.oddDaySales = 2)
  (h8 : data.evenDaySales = 3) :
  calculateSales data = (50, 22) := by
  sorry


end frank_meets_quota_l2403_240336


namespace correct_mark_calculation_l2403_240379

theorem correct_mark_calculation (n : ℕ) (initial_avg correct_avg wrong_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  (n : ℝ) * initial_avg - wrong_mark + (n : ℝ) * correct_avg - (n : ℝ) * initial_avg = 10 := by
  sorry

end correct_mark_calculation_l2403_240379


namespace imaginary_part_of_pure_imaginary_complex_l2403_240355

theorem imaginary_part_of_pure_imaginary_complex (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.im z = 2 := by
  sorry

end imaginary_part_of_pure_imaginary_complex_l2403_240355


namespace parallel_planes_perpendicular_line_l2403_240325

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_perpendicular_line 
  (α β : Plane) (m : Line) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) : 
  perpendicular m β :=
sorry

end parallel_planes_perpendicular_line_l2403_240325


namespace case_cost_l2403_240320

theorem case_cost (pen ink case : ℝ) 
  (total_cost : pen + ink + case = 2.30)
  (pen_cost : pen = ink + 1.50)
  (case_cost : case = 0.5 * ink) :
  case = 0.1335 := by
sorry

end case_cost_l2403_240320


namespace polynomial_factorization_l2403_240354

theorem polynomial_factorization (a b m n : ℝ) 
  (h : |m - 4| + (n^2 - 8*n + 16) = 0) : 
  a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2) := by
  sorry

end polynomial_factorization_l2403_240354


namespace inequality_proof_l2403_240332

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := by
  sorry

end inequality_proof_l2403_240332


namespace smallest_number_range_l2403_240382

theorem smallest_number_range (a b c d e : ℝ) 
  (distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (sum1 : a + b = 20)
  (sum2 : a + c = 200)
  (sum3 : d + e = 2014)
  (sum4 : c + e = 2000) :
  -793 < a ∧ a < 10 := by
sorry

end smallest_number_range_l2403_240382


namespace four_digit_integers_with_4_or_5_l2403_240366

/-- The number of four-digit positive integers -/
def four_digit_count : ℕ := 9000

/-- The number of options for the first digit when excluding 4 and 5 -/
def first_digit_options : ℕ := 7

/-- The number of options for each of the other three digits when excluding 4 and 5 -/
def other_digit_options : ℕ := 8

/-- The count of four-digit numbers without a 4 or 5 -/
def numbers_without_4_or_5 : ℕ := first_digit_options * other_digit_options * other_digit_options * other_digit_options

/-- The count of four-digit positive integers with at least one digit that is a 4 or a 5 -/
def numbers_with_4_or_5 : ℕ := four_digit_count - numbers_without_4_or_5

theorem four_digit_integers_with_4_or_5 : numbers_with_4_or_5 = 5416 := by
  sorry

end four_digit_integers_with_4_or_5_l2403_240366


namespace solve_for_q_l2403_240363

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end solve_for_q_l2403_240363


namespace simple_interest_calculation_l2403_240323

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 22.5)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 90 := by
  sorry

end simple_interest_calculation_l2403_240323


namespace fruit_bowl_problem_l2403_240330

/-- Represents the number of fruits in a bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit bowl problem -/
def validFruitBowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that a valid fruit bowl contains 9 bananas -/
theorem fruit_bowl_problem (bowl : FruitBowl) : 
  validFruitBowl bowl → bowl.bananas = 9 := by
  sorry


end fruit_bowl_problem_l2403_240330


namespace average_speeding_percentage_l2403_240311

def zone_a_speeding_percentage : ℝ := 30
def zone_b_speeding_percentage : ℝ := 20
def zone_c_speeding_percentage : ℝ := 25

def number_of_zones : ℕ := 3

theorem average_speeding_percentage :
  (zone_a_speeding_percentage + zone_b_speeding_percentage + zone_c_speeding_percentage) / number_of_zones = 25 := by
  sorry

end average_speeding_percentage_l2403_240311


namespace abc_sum_problem_l2403_240367

theorem abc_sum_problem (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  100 * A + 10 * B + C + 10 * A + B + A = C →
  C = 1 := by sorry

end abc_sum_problem_l2403_240367


namespace min_m_value_l2403_240359

theorem min_m_value (m : ℝ) (h_m : m > 0) :
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |m * x^3 - Real.log x| ≥ 1) →
  m ≥ (1/3) * Real.exp 2 :=
by sorry

end min_m_value_l2403_240359


namespace max_value_of_trigonometric_expression_l2403_240361

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + π/4) - Real.cos (x + π/3) + Real.sin (x + π/6)
  let domain : Set ℝ := {x | -π/4 ≤ x ∧ x ≤ 0}
  ∃ x ∈ domain, f x = 1 ∧ ∀ y ∈ domain, f y ≤ f x := by
  sorry

end max_value_of_trigonometric_expression_l2403_240361


namespace cubic_tangent_line_l2403_240362

/-- Given a cubic function f(x) = ax³ + bx + 1, if the tangent line
    at the point (1, f(1)) has the equation 4x - y - 1 = 0,
    then a + b = 2. -/
theorem cubic_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + b
  (f' 1 = 4 ∧ f 1 = 3) → a + b = 2 := by
sorry

end cubic_tangent_line_l2403_240362


namespace spiral_notebook_cost_l2403_240322

theorem spiral_notebook_cost :
  let personal_planner_cost : ℝ := 10
  let discount_rate : ℝ := 0.2
  let total_cost_with_discount : ℝ := 112
  let spiral_notebook_cost : ℝ := 15
  (1 - discount_rate) * (4 * spiral_notebook_cost + 8 * personal_planner_cost) = total_cost_with_discount :=
by sorry

end spiral_notebook_cost_l2403_240322


namespace inequality_system_solution_l2403_240321

theorem inequality_system_solution (x : ℝ) : 
  ((-x + 3) / 2 < x) ∧ (2 * (x + 6) ≥ 5 * x) → 1 < x ∧ x ≤ 4 := by
  sorry

end inequality_system_solution_l2403_240321


namespace mabel_transactions_l2403_240357

theorem mabel_transactions : ∃ M : ℕ,
  let A := (11 * M) / 10  -- Anthony's transactions
  let C := (2 * A) / 3    -- Cal's transactions
  let J := C + 15         -- Jade's transactions
  J = 81 ∧ M = 90 := by
  sorry

end mabel_transactions_l2403_240357


namespace normal_distribution_probability_l2403_240339

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Expected value of a random variable -/
def expected_value (ξ : NormalRandomVariable) : ℝ := ξ.μ

/-- Variance of a random variable -/
def variance (ξ : NormalRandomVariable) : ℝ := ξ.σ^2

/-- Probability of a random variable falling within a certain range -/
def probability (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h1 : expected_value ξ = 3) 
  (h2 : variance ξ = 1) 
  (h3 : probability ξ (ξ.μ - ξ.σ) (ξ.μ + ξ.σ) = 0.683) : 
  probability ξ 2 4 = 0.683 := by
  sorry

end normal_distribution_probability_l2403_240339


namespace rosalina_total_gifts_l2403_240389

/-- The number of gifts Rosalina received from Emilio -/
def gifts_from_emilio : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def gifts_from_jorge : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def gifts_from_pedro : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := gifts_from_emilio + gifts_from_jorge + gifts_from_pedro

theorem rosalina_total_gifts : total_gifts = 21 := by
  sorry

end rosalina_total_gifts_l2403_240389


namespace patsy_appetizers_needed_l2403_240310

def appetizers_per_guest : ℕ := 6
def number_of_guests : ℕ := 30
def deviled_eggs_dozens : ℕ := 3
def pigs_in_blanket_dozens : ℕ := 2
def kebabs_dozens : ℕ := 2
def items_per_dozen : ℕ := 12

theorem patsy_appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (deviled_eggs_dozens + pigs_in_blanket_dozens + kebabs_dozens) * items_per_dozen) / items_per_dozen = 8 := by
  sorry

end patsy_appetizers_needed_l2403_240310


namespace max_value_of_trigonometric_function_l2403_240304

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  ∃ (max_y : ℝ), max_y = 11 / 6 * Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc (-5 * Real.pi / 12) (-Real.pi / 3), y x ≤ max_y :=
by
  sorry

end max_value_of_trigonometric_function_l2403_240304


namespace beef_weight_before_processing_l2403_240394

theorem beef_weight_before_processing 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : loss_percentage = 40) 
  (h2 : final_weight = 240) 
  (h3 : final_weight = initial_weight * (1 - loss_percentage / 100)) : 
  initial_weight = 400 := by
sorry

end beef_weight_before_processing_l2403_240394


namespace no_real_solutions_l2403_240396

theorem no_real_solutions : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end no_real_solutions_l2403_240396


namespace units_digit_2137_power_753_l2403_240314

def units_digit (n : ℕ) : ℕ := n % 10

def power_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit (units_digit base ^ exp)

def cycle_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case will never be reached

theorem units_digit_2137_power_753 :
  power_units_digit 2137 753 = 7 :=
by
  sorry

end units_digit_2137_power_753_l2403_240314


namespace system_solution_l2403_240392

theorem system_solution : 
  ∃! (x y : ℝ), (3 * x + y = 2 ∧ 2 * x - 3 * y = 27) ∧ x = 3 ∧ y = -7 := by
  sorry

end system_solution_l2403_240392


namespace factorial_calculation_l2403_240352

theorem factorial_calculation : (Nat.factorial 9 * Nat.factorial 5 * Nat.factorial 2) / (Nat.factorial 8 * Nat.factorial 6) = 3 := by
  sorry

end factorial_calculation_l2403_240352


namespace smallest_b_value_l2403_240327

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 6) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 9) :
  b.val ≥ 3 ∧ ∃ (a' b' : ℕ+), b'.val = 3 ∧ a'.val - b'.val = 6 ∧ 
    Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 9 :=
by sorry


end smallest_b_value_l2403_240327


namespace square_difference_squared_l2403_240308

theorem square_difference_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end square_difference_squared_l2403_240308


namespace quadratic_equation_roots_l2403_240346

theorem quadratic_equation_roots :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ 
  (∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = r1 ∨ x = r2) := by
  sorry

end quadratic_equation_roots_l2403_240346


namespace hyperbola_asymptote_l2403_240340

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3x/5, then a = 5. -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3 * x / 5) →
  a = 5 := by
sorry

end hyperbola_asymptote_l2403_240340


namespace same_solution_c_value_l2403_240348

theorem same_solution_c_value (x : ℚ) (c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x + 8 = 6) → c = 3/2 := by
  sorry

end same_solution_c_value_l2403_240348


namespace f_eq_g_l2403_240369

/-- The given polynomial function f(x, y, z) -/
def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) +
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) +
  (x^2 - y^2) * (1 + y*z) * (1 + x*z)

/-- The factored form of the polynomial -/
def g (x y z : ℝ) : ℝ :=
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z)

/-- Theorem stating that f and g are equivalent for all real x, y, and z -/
theorem f_eq_g : ∀ x y z : ℝ, f x y z = g x y z := by
  sorry

end f_eq_g_l2403_240369


namespace women_decrease_l2403_240349

theorem women_decrease (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  initial_women - 3 = 24 →
  initial_women - 24 = 3 := by
sorry

end women_decrease_l2403_240349


namespace trig_expression_simplification_l2403_240377

theorem trig_expression_simplification (θ : ℝ) :
  (Real.tan (2 * Real.pi - θ) * Real.sin (-2 * Real.pi - θ) * Real.cos (6 * Real.pi - θ)) /
  (Real.cos (θ - Real.pi) * Real.sin (5 * Real.pi + θ)) = Real.tan θ :=
by sorry

end trig_expression_simplification_l2403_240377


namespace tan_105_degrees_l2403_240307

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -Real.sqrt 3 - 2 := by
  sorry

end tan_105_degrees_l2403_240307


namespace person_age_is_54_l2403_240335

/-- Represents the age of a person and their eldest son, satisfying given conditions --/
structure AgeRelation where
  Y : ℕ  -- Current age of the person
  S : ℕ  -- Current age of the eldest son
  age_relation_past : Y - 9 = 5 * (S - 9)  -- Relation 9 years ago
  age_relation_present : Y = 3 * S         -- Current relation

/-- Theorem stating that given the conditions, the person's current age is 54 --/
theorem person_age_is_54 (ar : AgeRelation) : ar.Y = 54 := by
  sorry

end person_age_is_54_l2403_240335


namespace inverse_g_inverse_g_14_l2403_240374

def g (x : ℝ) : ℝ := 5 * x - 3

theorem inverse_g_inverse_g_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 32 / 25 := by
  sorry

end inverse_g_inverse_g_14_l2403_240374


namespace complex_equation_solution_l2403_240383

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z = 1 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l2403_240383


namespace problem_solution_l2403_240356

theorem problem_solution (a b n : ℤ) : 
  a % 50 = 24 →
  b % 50 = 95 →
  150 ≤ n ∧ n ≤ 200 →
  (a - b) % 50 = n % 50 →
  n % 4 = 3 →
  n = 179 := by
sorry

end problem_solution_l2403_240356


namespace even_function_shift_l2403_240331

/-- Given a function f and a real number a, proves that if f(x+a) is even and a is in (0,π/2), then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (f = λ x => 3 * Real.sin (2 * x - π/3)) →
  (∀ x, f (x + a) = f (-x - a)) →
  (0 < a) →
  (a < π/2) →
  a = 5*π/12 := by
sorry

end even_function_shift_l2403_240331


namespace macaroon_solution_l2403_240329

/-- Represents the problem of calculating the remaining weight of macaroons --/
def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) : Prop :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  let remaining_bags := num_bags - 1
  let remaining_weight := remaining_bags * weight_per_bag
  remaining_weight = 45

/-- Theorem stating the solution to the macaroon problem --/
theorem macaroon_solution : macaroon_problem 12 5 4 := by
  sorry

end macaroon_solution_l2403_240329


namespace cardinality_of_S_l2403_240341

/-- The number of elements in a set -/
def C (A : Set ℝ) : ℕ := sorry

/-- The operation * defined on sets -/
def star (A B : Set ℝ) : ℕ :=
  if C A ≥ C B then C A - C B else C B - C A

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ :=
  {x : ℝ | (x + a) * (x^3 + a*x^2 + 2*x) = 0}

/-- The set A -/
def A : Set ℝ := {1, 2}

/-- The set S of all possible values of a -/
def S : Set ℝ :=
  {a : ℝ | star A (B a) = 1 ∧ C A = 2}

theorem cardinality_of_S : C S = 3 := by sorry

end cardinality_of_S_l2403_240341


namespace line_intersects_circle_l2403_240395

/-- The line l intersects with the circle C if the distance from the center of C to l is less than the radius of C. -/
theorem line_intersects_circle (m : ℝ) : 
  let l : Set (ℝ × ℝ) := {(x, y) | m * x - y + 1 = 0}
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y-1)^2 = 5}
  let center : ℝ × ℝ := (0, 1)
  let radius : ℝ := Real.sqrt 5
  let distance_to_line (p : ℝ × ℝ) : ℝ := 
    abs (m * p.1 - p.2 + 1) / Real.sqrt (m^2 + 1)
  distance_to_line center < radius → 
  ∃ p, p ∈ l ∧ p ∈ C := by
sorry

end line_intersects_circle_l2403_240395


namespace second_order_eq_circle_iff_l2403_240345

/-- A general second-order equation in two variables -/
structure SecondOrderEquation where
  a11 : ℝ
  a12 : ℝ
  a22 : ℝ
  a13 : ℝ
  a23 : ℝ
  a33 : ℝ

/-- Predicate to check if a second-order equation represents a circle -/
def IsCircle (eq : SecondOrderEquation) : Prop :=
  eq.a11 = eq.a22 ∧ eq.a12 = 0

/-- Theorem stating the conditions for a second-order equation to represent a circle -/
theorem second_order_eq_circle_iff (eq : SecondOrderEquation) :
  IsCircle eq ↔ ∃ (h k : ℝ) (r : ℝ), r > 0 ∧
    ∀ (x y : ℝ), eq.a11 * x^2 + 2*eq.a12 * x*y + eq.a22 * y^2 + 2*eq.a13 * x + 2*eq.a23 * y + eq.a33 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 :=
  sorry


end second_order_eq_circle_iff_l2403_240345


namespace sandy_clothes_cost_l2403_240316

-- Define the costs of individual items
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

-- Define the total cost
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

-- Theorem statement
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end sandy_clothes_cost_l2403_240316


namespace store_inventory_l2403_240313

theorem store_inventory (ties belts black_shirts : ℕ) 
  (h1 : ties = 34)
  (h2 : belts = 40)
  (h3 : black_shirts = 63)
  (h4 : ∃ white_shirts : ℕ, 
    ∃ jeans : ℕ, 
    ∃ scarves : ℕ,
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33) :
  ∃ white_shirts : ℕ, white_shirts = 42 := by
sorry

end store_inventory_l2403_240313


namespace arithmetic_sequence_roots_iff_l2403_240318

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for a cubic equation having three real roots in arithmetic sequence -/
def has_arithmetic_sequence_roots (eq : CubicEquation) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    x₃ - x₂ = x₂ - x₁ ∧
    x₁^3 + eq.a * x₁^2 + eq.b * x₁ + eq.c = 0 ∧
    x₂^3 + eq.a * x₂^2 + eq.b * x₂ + eq.c = 0 ∧
    x₃^3 + eq.a * x₃^2 + eq.b * x₃ + eq.c = 0

/-- The necessary and sufficient conditions for a cubic equation to have three real roots in arithmetic sequence -/
theorem arithmetic_sequence_roots_iff (eq : CubicEquation) :
  has_arithmetic_sequence_roots eq ↔ 
  (2 * eq.a^3 - 9 * eq.a * eq.b + 27 * eq.c = 0) ∧ (eq.a^2 - 3 * eq.b ≥ 0) :=
by sorry

end arithmetic_sequence_roots_iff_l2403_240318


namespace relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2403_240378

-- Define the relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ :=
  |a - n| + |b - n|

-- Theorem 1
theorem relative_relationship_value_example : 
  relative_relationship_value 2 (-5) 2 = 7 := by sorry

-- Theorem 2
theorem max_sum_given_relative_relationship_value :
  ∀ m n : ℚ, relative_relationship_value m n 2 = 2 → 
  m + n ≤ 6 := by sorry

-- Theorem to show that 6 is indeed achievable
theorem max_sum_achievable :
  ∃ m n : ℚ, relative_relationship_value m n 2 = 2 ∧ m + n = 6 := by sorry

end relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2403_240378


namespace solution_characterization_l2403_240309

def solution_set (a : ℕ) : Set ℝ :=
  {x : ℝ | a * x^2 + 2 * |x - a| - 20 < 0}

def inequality1_set : Set ℝ :=
  {x : ℝ | x^2 + x - 2 < 0}

def inequality2_set : Set ℝ :=
  {x : ℝ | |2*x - 1| < x + 2}

theorem solution_characterization :
  ∀ a : ℕ, (inequality1_set ⊆ solution_set a ∧ inequality2_set ⊆ solution_set a) ↔ 
    a ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) :=
by sorry

end solution_characterization_l2403_240309


namespace line_circle_intersection_l2403_240343

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ↔ 
  (k = 1 → ∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ∧ 
  (∃ k' : ℝ, k' ≠ 1 ∧ ∃ x y : ℝ, x - y + k' = 0 ∧ x^2 + y^2 = 1) :=
sorry

end line_circle_intersection_l2403_240343


namespace intersection_of_A_and_B_l2403_240381

def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {-5, -4, -3, 1}

theorem intersection_of_A_and_B : A ∩ B = {-5, -4} := by
  sorry

end intersection_of_A_and_B_l2403_240381


namespace power_function_through_2_4_l2403_240338

/-- A power function that passes through the point (2, 4) is equivalent to f(x) = x^2 -/
theorem power_function_through_2_4 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 2 = 4 →           -- f passes through the point (2, 4)
  ∀ x, f x = x^2 :=   -- f is equivalent to x^2
by sorry

end power_function_through_2_4_l2403_240338


namespace nancys_savings_in_euros_l2403_240300

/-- Calculates the amount of money Nancy has in euros given her savings and the exchange rate. -/
def nancys_euros_savings (quarters : ℕ) (five_dollar_bills : ℕ) (dimes : ℕ) (exchange_rate : ℚ) : ℚ :=
  let dollars : ℚ := (quarters * (1 / 4) + five_dollar_bills * 5 + dimes * (1 / 10))
  dollars / exchange_rate

/-- Proves that Nancy has €18.21 in euros given her savings and the exchange rate. -/
theorem nancys_savings_in_euros :
  nancys_euros_savings 12 3 24 (112 / 100) = 1821 / 100 := by
  sorry

end nancys_savings_in_euros_l2403_240300


namespace square_difference_l2403_240333

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end square_difference_l2403_240333


namespace disjoint_triangles_exist_l2403_240312

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are disjoint -/
def disjoint (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c

/-- The main theorem -/
theorem disjoint_triangles_exist (n : ℕ) (points : Fin (3 * n) → Point) 
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ triangles : Fin n → Triangle, 
    (∀ i, ∃ j k l, triangles i = ⟨points j, points k, points l⟩) ∧ 
    (∀ i j, i ≠ j → disjoint (triangles i) (triangles j)) :=
  sorry


end disjoint_triangles_exist_l2403_240312


namespace fourth_root_64_times_cube_root_27_times_sqrt_9_l2403_240351

theorem fourth_root_64_times_cube_root_27_times_sqrt_9 :
  (64 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 18 * (2 : ℝ) ^ (1/2) := by
  sorry

end fourth_root_64_times_cube_root_27_times_sqrt_9_l2403_240351


namespace tangent_line_at_one_max_value_condition_a_range_condition_l2403_240384

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

-- Define e as the base of natural logarithms
def e : ℝ := Real.exp 1

-- Theorem 1: Tangent line equation when a = 1
theorem tangent_line_at_one (x y : ℝ) :
  f 1 1 = 1 → 2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1) :=
sorry

-- Theorem 2: Value of a when maximum of f(x) is -2
theorem max_value_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = -2) → a = -e :=
sorry

-- Theorem 3: Range of a when a < 0 and f(x) ≤ g(x) for x ∈ [1,e]
theorem a_range_condition (a : ℝ) :
  a < 0 ∧ (∀ x ∈ Set.Icc 1 e, f a x ≤ g a x) →
  a ∈ Set.Icc ((1 - 2*e) / (e^2 - e)) 0 :=
sorry

end tangent_line_at_one_max_value_condition_a_range_condition_l2403_240384


namespace chair_carrying_trips_l2403_240353

/-- Proves that given 5 students, each carrying 5 chairs per trip, and a total of 250 chairs moved, the number of trips each student made is 10 -/
theorem chair_carrying_trips 
  (num_students : ℕ) 
  (chairs_per_trip : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5)
  (h2 : chairs_per_trip = 5)
  (h3 : total_chairs = 250) :
  (total_chairs / (num_students * chairs_per_trip) : ℕ) = 10 := by
  sorry

end chair_carrying_trips_l2403_240353


namespace complex_subtraction_l2403_240306

theorem complex_subtraction : (6 : ℂ) + 2*I - (3 - 5*I) = 3 + 7*I := by
  sorry

end complex_subtraction_l2403_240306


namespace sequence_properties_l2403_240385

def S (n : ℕ) : ℤ := 2 * n^2 - 10 * n

def a (n : ℕ) : ℤ := 4 * n - 5

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∃ n : ℕ, ∀ m : ℕ, S m ≥ S n) ∧
  (∃ n : ℕ, S n = -12 ∧ ∀ m : ℕ, S m ≥ S n) :=
sorry

end sequence_properties_l2403_240385


namespace partridge_family_allowance_l2403_240398

/-- The total weekly allowance for the Partridge family children -/
theorem partridge_family_allowance : 
  ∀ (younger_children older_children : ℕ) 
    (younger_allowance older_allowance : ℚ),
  younger_children = 3 →
  older_children = 2 →
  younger_allowance = 8 →
  older_allowance = 13 →
  (younger_children : ℚ) * younger_allowance + (older_children : ℚ) * older_allowance = 50 :=
by
  sorry

end partridge_family_allowance_l2403_240398


namespace y_derivative_l2403_240372

noncomputable def y (x : ℝ) : ℝ := 
  (1/2) * Real.log ((1 + Real.cos x) / (1 - Real.cos x)) - 1 / Real.cos x - 1 / (3 * (Real.cos x)^3)

theorem y_derivative (x : ℝ) (h : Real.cos x ≠ 0) (h' : Real.sin x ≠ 0) : 
  deriv y x = -1 / (Real.sin x * (Real.cos x)^4) :=
sorry

end y_derivative_l2403_240372
