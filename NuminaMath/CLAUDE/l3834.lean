import Mathlib

namespace sedan_acceleration_l3834_383489

def v (t : ℝ) : ℝ := t^2 + 3

theorem sedan_acceleration : 
  let a (t : ℝ) := (deriv v) t
  a 3 = 6 := by sorry

end sedan_acceleration_l3834_383489


namespace intersection_of_A_and_B_l3834_383432

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l3834_383432


namespace three_W_five_l3834_383488

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem three_W_five : W 3 5 = 23 := by
  sorry

end three_W_five_l3834_383488


namespace fifth_term_of_specific_arithmetic_sequence_l3834_383497

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ := 3
  let a₂ := 7
  let a₃ := 11
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 5 = 19 := by sorry

end fifth_term_of_specific_arithmetic_sequence_l3834_383497


namespace ice_cream_problem_l3834_383413

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def quarters : ℕ := 26
def double_scoop_cost : ℚ := 3
def leftover : ℚ := 48/100

def total_amount : ℚ := 
  pennies * 1/100 + nickels * 5/100 + dimes * 10/100 + quarters * 25/100

theorem ice_cream_problem : 
  ∃ (n : ℕ), n * double_scoop_cost = total_amount - leftover ∧ n = 5 := by
  sorry

end ice_cream_problem_l3834_383413


namespace certain_number_value_l3834_383443

theorem certain_number_value : ∃! x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 90 ∧
  (128 + 255 + 511 + 1023 + x) / 5 = 423 ∧
  x = 198 := by
  sorry

end certain_number_value_l3834_383443


namespace polynomial_identity_l3834_383457

theorem polynomial_identity (x : ℝ) : 
  let P (x : ℝ) := (x - 1/2)^2001 + 1/2
  P x + P (1 - x) = 1 := by sorry

end polynomial_identity_l3834_383457


namespace log_properties_l3834_383474

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- State the properties to be proven
theorem log_properties :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) :=
by sorry

end log_properties_l3834_383474


namespace sum_g_32_neg_32_l3834_383441

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

/-- Theorem stating that the sum of g(32) and g(-32) equals 6 -/
theorem sum_g_32_neg_32 (a b c : ℝ) (h : g a b c 32 = 3) :
  g a b c 32 + g a b c (-32) = 6 := by
  sorry

end sum_g_32_neg_32_l3834_383441


namespace complex_number_in_second_quadrant_l3834_383412

theorem complex_number_in_second_quadrant (z : ℂ) (a : ℝ) :
  z = a + Complex.I * Real.sqrt 3 →
  (Complex.re z < 0 ∧ Complex.im z > 0) →
  Complex.abs z = 2 →
  z = -1 + Complex.I * Real.sqrt 3 := by
sorry

end complex_number_in_second_quadrant_l3834_383412


namespace infinite_solutions_equation_l3834_383433

theorem infinite_solutions_equation (A B C : ℚ) : 
  (∀ x : ℚ, (x + B) * (A * x + 40) = 3 * (x + C) * (x + 10)) →
  (A = 3 ∧ B = 10/9 ∧ C = 40/9 ∧ 
   (- 40/9) + (-10) = -130/9) :=
by sorry

end infinite_solutions_equation_l3834_383433


namespace max_cables_is_150_l3834_383493

/-- Represents the maximum number of cables that can be installed between
    Brand A and Brand B computers under given conditions. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) 
               (connectable_brand_b : ℕ) : ℕ :=
  brand_a_computers * connectable_brand_b

/-- Theorem stating that the maximum number of cables is 150 under the given conditions. -/
theorem max_cables_is_150 :
  max_cables 50 15 35 10 = 150 :=
by
  sorry

end max_cables_is_150_l3834_383493


namespace infinitely_many_L_for_fibonacci_ratio_l3834_383446

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- The sequence defined in the problem -/
def a (L : ℕ) : ℕ → ℚ
  | 0 => 0
  | (n + 1) => 1 / (L - a L n)

theorem infinitely_many_L_for_fibonacci_ratio :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k, ∃ i j,
    ∀ n, a (lucas (f k)) n = (fib i) / (fib j) := by
  sorry

end infinitely_many_L_for_fibonacci_ratio_l3834_383446


namespace line_through_AB_l3834_383406

-- Define the lines and points
def line1 (a₁ b₁ x y : ℝ) : Prop := a₁ * x + b₁ * y + 1 = 0
def line2 (a₂ b₂ x y : ℝ) : Prop := a₂ * x + b₂ * y + 1 = 0
def point_P : ℝ × ℝ := (2, 3)
def point_A (a₁ b₁ : ℝ) : ℝ × ℝ := (a₁, b₁)
def point_B (a₂ b₂ : ℝ) : ℝ × ℝ := (a₂, b₂)

-- Define the theorem
theorem line_through_AB (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : line1 a₁ b₁ (point_P.1) (point_P.2))
  (h2 : line2 a₂ b₂ (point_P.1) (point_P.2))
  (h3 : a₁ ≠ a₂) :
  ∃ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (y - b₁) / (x - a₁) = (b₂ - b₁) / (a₂ - a₁) :=
by sorry

end line_through_AB_l3834_383406


namespace cheese_cost_for_order_l3834_383417

/-- Represents the cost of cheese for a Mexican restaurant order --/
def cheese_cost (burrito_count : ℕ) (taco_count : ℕ) (enchilada_count : ℕ) : ℚ :=
  let cheddar_per_burrito : ℚ := 4
  let cheddar_per_taco : ℚ := 9
  let mozzarella_per_enchilada : ℚ := 5
  let cheddar_cost_per_ounce : ℚ := 4/5
  let mozzarella_cost_per_ounce : ℚ := 1
  (burrito_count * cheddar_per_burrito + taco_count * cheddar_per_taco) * cheddar_cost_per_ounce +
  (enchilada_count * mozzarella_per_enchilada) * mozzarella_cost_per_ounce

/-- Theorem stating the total cost of cheese for a specific order --/
theorem cheese_cost_for_order :
  cheese_cost 7 1 3 = 446/10 :=
sorry

end cheese_cost_for_order_l3834_383417


namespace incorrect_statement_l3834_383495

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q → False) → (p → False) ∧ (q → False)) := by
  sorry

end incorrect_statement_l3834_383495


namespace friends_meeting_problem_l3834_383436

/-- Two friends walk in opposite directions and then run towards each other -/
theorem friends_meeting_problem 
  (misha_initial_speed : ℝ) 
  (vasya_initial_speed : ℝ) 
  (initial_walk_time : ℝ) 
  (speed_increase_factor : ℝ) :
  misha_initial_speed = 8 →
  vasya_initial_speed = misha_initial_speed / 2 →
  initial_walk_time = 3/4 →
  speed_increase_factor = 3/2 →
  ∃ (meeting_time total_distance : ℝ),
    meeting_time = 1/2 ∧ 
    total_distance = 18 :=
by sorry

end friends_meeting_problem_l3834_383436


namespace polyhedron_property_l3834_383465

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  h : ℕ  -- Number of hexagonal faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  H : ℕ  -- Number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 30
  face_types : t + h = F
  edge_count : E = (3 * t + 6 * h) / 2
  vertex_count : V = 3 * t / T
  triangle_hex_relation : T = 1 ∧ H = 2

/-- Theorem stating the specific property of the polyhedron -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.V = 270 := by
  sorry

end polyhedron_property_l3834_383465


namespace least_coins_seventeen_coins_coins_in_jar_l3834_383470

theorem least_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) → n ≥ 17 :=
by
  sorry

theorem seventeen_coins : 
  (17 % 7 = 3) ∧ (17 % 4 = 1) ∧ (17 % 6 = 5) :=
by
  sorry

theorem coins_in_jar : 
  ∃ (n : ℕ), (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) ∧ 
  (∀ (m : ℕ), (m % 7 = 3) ∧ (m % 4 = 1) ∧ (m % 6 = 5) → m ≥ n) :=
by
  sorry

end least_coins_seventeen_coins_coins_in_jar_l3834_383470


namespace specific_ellipse_foci_distance_l3834_383440

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end specific_ellipse_foci_distance_l3834_383440


namespace average_student_headcount_l3834_383419

def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

theorem average_student_headcount : 
  (student_headcount.sum / student_headcount.length : ℚ) = 11029 := by
  sorry

end average_student_headcount_l3834_383419


namespace smallest_factor_for_perfect_square_l3834_383408

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 2 * 3^2 * 5^2 * 7) :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (b : ℕ), b > 0 → ∃ (k : ℕ), n * b = k^2 → a ≤ b) ∧
  (∃ (k : ℕ), n * a = k^2) ∧
  a = 14 := by
sorry

end smallest_factor_for_perfect_square_l3834_383408


namespace vowel_classification_l3834_383437

-- Define the set of all English letters
def EnglishLetters : Type := Fin 26

-- Define the categories
inductive Category
| one
| two
| three
| four
| five

-- Define the classification function
def classify : EnglishLetters → Category := sorry

-- Define the vowels
def vowels : Fin 5 → EnglishLetters := sorry

-- Theorem statement
theorem vowel_classification :
  (classify (vowels 0) = Category.four) ∧
  (classify (vowels 1) = Category.three) ∧
  (classify (vowels 2) = Category.one) ∧
  (classify (vowels 3) = Category.one) ∧
  (classify (vowels 4) = Category.four) := by
  sorry

end vowel_classification_l3834_383437


namespace buffet_meal_combinations_l3834_383455

theorem buffet_meal_combinations : 
  (Nat.choose 4 2) * (Nat.choose 5 3) * (Nat.choose 5 2) = 600 := by
  sorry

end buffet_meal_combinations_l3834_383455


namespace additional_week_rate_is_12_l3834_383464

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate (first_week_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost - first_week_rate * 7) / (total_days - 7)

/-- Theorem stating that the additional week rate is $12.00 per day -/
theorem additional_week_rate_is_12 :
  additional_week_rate 18 23 318 = 12 := by
  sorry

end additional_week_rate_is_12_l3834_383464


namespace appropriate_sampling_methods_l3834_383498

/-- Represents a sampling task with a population size and sample size -/
structure SamplingTask where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a stratified population with different group sizes -/
structure StratifiedPopulation where
  group_sizes : List ℕ

/-- Enumeration of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Determines the most appropriate sampling method for a given task -/
def most_appropriate_sampling_method (task : SamplingTask) (stratified_info : Option StratifiedPopulation) : SamplingMethod :=
  sorry

/-- The three sampling tasks from the problem -/
def yogurt_task : SamplingTask := { population_size := 10, sample_size := 3 }
def attendees_task : SamplingTask := { population_size := 1280, sample_size := 32 }
def staff_task : SamplingTask := { population_size := 160, sample_size := 20 }

/-- The stratified population information for the staff task -/
def staff_stratified : StratifiedPopulation := { group_sizes := [120, 16, 24] }

theorem appropriate_sampling_methods :
  most_appropriate_sampling_method yogurt_task none = SamplingMethod.SimpleRandom ∧
  most_appropriate_sampling_method attendees_task none = SamplingMethod.Systematic ∧
  most_appropriate_sampling_method staff_task (some staff_stratified) = SamplingMethod.Stratified :=
sorry

end appropriate_sampling_methods_l3834_383498


namespace one_adult_in_family_l3834_383463

/-- Represents the cost of tickets for a family visit to an aquarium -/
structure AquariumTickets where
  adultPrice : ℕ
  childPrice : ℕ
  numChildren : ℕ
  totalCost : ℕ

/-- Calculates the number of adults in the family based on ticket prices and total cost -/
def calculateAdults (tickets : AquariumTickets) : ℕ :=
  (tickets.totalCost - tickets.childPrice * tickets.numChildren) / tickets.adultPrice

/-- Theorem stating that for the given ticket prices and family composition, there is 1 adult -/
theorem one_adult_in_family (tickets : AquariumTickets) 
  (h1 : tickets.adultPrice = 35)
  (h2 : tickets.childPrice = 20)
  (h3 : tickets.numChildren = 6)
  (h4 : tickets.totalCost = 155) : 
  calculateAdults tickets = 1 := by
  sorry

#eval calculateAdults { adultPrice := 35, childPrice := 20, numChildren := 6, totalCost := 155 }

end one_adult_in_family_l3834_383463


namespace day_90_of_year_N_minus_1_is_thursday_l3834_383444

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

/-- Checks if a given year is a leap year -/
def isLeapYear (year : ℤ) : Bool :=
  sorry

theorem day_90_of_year_N_minus_1_is_thursday
  (N : ℤ)
  (h1 : dayOfWeek ⟨N, 150⟩ = DayOfWeek.Sunday)
  (h2 : dayOfWeek ⟨N + 2, 220⟩ = DayOfWeek.Sunday) :
  dayOfWeek ⟨N - 1, 90⟩ = DayOfWeek.Thursday :=
by sorry

end day_90_of_year_N_minus_1_is_thursday_l3834_383444


namespace greatest_valid_number_l3834_383494

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10) * (n % 10) = 12 ∧
  (n / 10) < (n % 10)

theorem greatest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≤ 34 :=
by sorry

end greatest_valid_number_l3834_383494


namespace sum_of_digits_1197_l3834_383434

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem sum_of_digits_1197 : digit_sum 1197 = 18 := by
  sorry

end sum_of_digits_1197_l3834_383434


namespace bobs_age_l3834_383442

theorem bobs_age (alice : ℝ) (bob : ℝ) 
  (h1 : bob = 3 * alice - 20) 
  (h2 : bob + alice = 70) : 
  bob = 47.5 := by
  sorry

end bobs_age_l3834_383442


namespace rals_current_age_l3834_383400

-- Define Suri's and Ral's ages as natural numbers
def suris_age : ℕ := sorry
def rals_age : ℕ := sorry

-- State the theorem
theorem rals_current_age : 
  (rals_age = 3 * suris_age) →   -- Ral is three times as old as Suri
  (suris_age + 3 = 16) →         -- In 3 years, Suri's current age will be 16
  rals_age = 39 :=               -- Ral's current age is 39
by sorry

end rals_current_age_l3834_383400


namespace product_of_integers_l3834_383490

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 3 * (a * b) + 4 * a = 5 * b + 318 → a * b = 56 := by
  sorry

end product_of_integers_l3834_383490


namespace value_of_x_l3834_383467

theorem value_of_x : (2015^2 - 2015) / 2015 = 2014 := by sorry

end value_of_x_l3834_383467


namespace min_red_tulips_for_arrangement_l3834_383415

/-- Represents the number of tulips in a bouquet -/
structure Bouquet where
  white : ℕ
  red : ℕ

/-- Represents the total number of tulips and bouquets -/
structure TulipArrangement where
  whiteTotal : ℕ
  redTotal : ℕ
  bouquetCount : ℕ

/-- Checks if a TulipArrangement is valid according to the problem constraints -/
def isValidArrangement (arr : TulipArrangement) : Prop :=
  ∃ (b : Bouquet),
    arr.whiteTotal = arr.bouquetCount * b.white ∧
    arr.redTotal = arr.bouquetCount * b.red

/-- The main theorem to prove -/
theorem min_red_tulips_for_arrangement :
  ∀ (arr : TulipArrangement),
    arr.whiteTotal = 21 →
    arr.bouquetCount = 7 →
    isValidArrangement arr →
    arr.redTotal ≥ 7 :=
by sorry

end min_red_tulips_for_arrangement_l3834_383415


namespace max_consecutive_semi_primes_l3834_383405

/-- A natural number greater than 25 is semi-prime if it is the sum of two different prime numbers -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive natural numbers that are semi-prime is 5 -/
theorem max_consecutive_semi_primes :
  ∀ k : ℕ, (∀ n : ℕ, ∀ i : ℕ, i < k → IsSemiPrime (n + i)) → k ≤ 5 :=
sorry

end max_consecutive_semi_primes_l3834_383405


namespace reach_50_from_49_l3834_383492

def double (n : ℕ) : ℕ := n * 2

def eraseLast (n : ℕ) : ℕ := n / 10

def canReach (start target : ℕ) : Prop :=
  ∃ (steps : ℕ), ∃ (moves : Fin steps → Bool),
    (start = target) ∨
    (∃ (intermediate : Fin (steps + 1) → ℕ),
      intermediate 0 = start ∧
      intermediate (Fin.last steps) = target ∧
      ∀ i : Fin steps,
        (moves i = true → intermediate (i.succ) = double (intermediate i)) ∧
        (moves i = false → intermediate (i.succ) = eraseLast (intermediate i)))

theorem reach_50_from_49 : canReach 49 50 := by
  sorry

end reach_50_from_49_l3834_383492


namespace sum_of_last_two_digits_is_eight_l3834_383487

def sixDigitNumber (x y : ℕ) : ℕ := 123400 + 10 * x + y

theorem sum_of_last_two_digits_is_eight 
  (x y : ℕ) 
  (h1 : x < 10 ∧ y < 10) 
  (h2 : sixDigitNumber x y % 8 = 0) 
  (h3 : sixDigitNumber x y % 9 = 0) :
  x + y = 8 := by
sorry

end sum_of_last_two_digits_is_eight_l3834_383487


namespace tonyas_age_l3834_383485

/-- Proves Tonya's age given the conditions of the problem -/
theorem tonyas_age (john mary tonya : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john = tonya / 2)
  (h3 : (john + mary + tonya) / 3 = 35) :
  tonya = 60 := by sorry

end tonyas_age_l3834_383485


namespace vacation_cost_l3834_383486

/-- The total cost of a vacation satisfying specific conditions -/
theorem vacation_cost : ∃ (C P : ℝ), 
  C = 5 * P ∧ 
  C = 7 * (P - 40) ∧ 
  C = 8 * (P - 60) ∧ 
  C = 700 := by
  sorry

end vacation_cost_l3834_383486


namespace dwarf_heights_l3834_383461

/-- The heights of Mr. Ticháček's dwarfs -/
theorem dwarf_heights :
  ∀ (F J M : ℕ),
  (J + F = M) →
  (M + F = J + 34) →
  (M + J = F + 72) →
  (F = 17 ∧ J = 36 ∧ M = 53) :=
by
  sorry

end dwarf_heights_l3834_383461


namespace most_frequent_digit_l3834_383421

/-- The digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n - 1) % 9 + 1

/-- The count of occurrences of each digit (1-9) in the digital roots of numbers from 1 to 1,000,000 -/
def digitCounts : Fin 9 → ℕ
| ⟨i, _⟩ => if i = 0 then 111112 else 111111

theorem most_frequent_digit :
  ∃ (d : Fin 9), ∀ (d' : Fin 9), digitCounts d ≥ digitCounts d' ∧
  (d = ⟨0, by norm_num⟩ ∨ digitCounts d > digitCounts d') :=
sorry

end most_frequent_digit_l3834_383421


namespace curve_C_cartesian_to_polar_l3834_383428

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The polar equation of curve C -/
def polar_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The relationship between Cartesian and polar coordinates -/
def polar_to_cartesian (ρ θ x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

theorem curve_C_cartesian_to_polar :
  ∀ x y ρ θ : ℝ, 
    polar_to_cartesian ρ θ x y →
    (C x y ↔ polar_C ρ θ) :=
by sorry

end curve_C_cartesian_to_polar_l3834_383428


namespace abs_ac_plus_bd_le_one_l3834_383427

theorem abs_ac_plus_bd_le_one (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : 
  |a*c + b*d| ≤ 1 := by sorry

end abs_ac_plus_bd_le_one_l3834_383427


namespace rectangle_equation_l3834_383479

/-- A rectangle centered at the origin with width 2a and height 2b can be described by the equation
    √x * √(a - x) * √y * √(b - y) = 0, where 0 ≤ x ≤ a and 0 ≤ y ≤ b. -/
theorem rectangle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧
    Real.sqrt x * Real.sqrt (a - x) * Real.sqrt y * Real.sqrt (b - y) = 0} =
  {p : ℝ × ℝ | -a ≤ p.1 ∧ p.1 ≤ a ∧ -b ≤ p.2 ∧ p.2 ≤ b} :=
by sorry

end rectangle_equation_l3834_383479


namespace rationalize_denominator_l3834_383430

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 4 ∧ B = 3 ∧ C = -1 ∧ D = -1 ∧ E = 42 ∧ F = 10 :=
by sorry

end rationalize_denominator_l3834_383430


namespace adults_on_bus_l3834_383471

/-- Given a bus with 60 passengers where children make up 25% of the riders,
    prove that there are 45 adults on the bus. -/
theorem adults_on_bus (total_passengers : ℕ) (children_percentage : ℚ) : 
  total_passengers = 60 →
  children_percentage = 25 / 100 →
  (total_passengers : ℚ) * (1 - children_percentage) = 45 := by
sorry

end adults_on_bus_l3834_383471


namespace log_157489_between_consecutive_integers_l3834_383481

theorem log_157489_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 157489 / Real.log 10 ∧ Real.log 157489 / Real.log 10 < (d : ℝ) ∧ c + d = 11 := by
  sorry

end log_157489_between_consecutive_integers_l3834_383481


namespace area_of_triangle_ABC_l3834_383424

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 10
def AB : ℝ := 17
def DC : ℝ := 6

-- Define coplanarity
def coplanar (A B C D : ℝ × ℝ) : Prop := sorry

-- Define right angle
def right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC (h1 : coplanar A B C D) 
                             (h2 : right_angle A D C) :
  triangle_area A B C = 84 := by sorry

end area_of_triangle_ABC_l3834_383424


namespace expression_equality_l3834_383469

theorem expression_equality : 
  (Real.sqrt 3 - Real.sqrt 2) * (-Real.sqrt 3 - Real.sqrt 2) + (3 + 2 * Real.sqrt 5)^2 = 28 + 12 * Real.sqrt 5 := by
  sorry

end expression_equality_l3834_383469


namespace candle_burning_theorem_l3834_383459

/-- Represents the state of a burning candle -/
structure BurningCandle where
  burn_rate : ℝ
  remaining : ℝ

/-- Represents the state of three burning candles -/
structure ThreeCandles where
  candle1 : BurningCandle
  candle2 : BurningCandle
  candle3 : BurningCandle

/-- 
Given three candles burning at constant rates, if 2/5 of the second candle
and 3/7 of the third candle remain when the first candle burns out, then
1/21 of the third candle will remain when the second candle burns out.
-/
theorem candle_burning_theorem (candles : ThreeCandles) 
  (h1 : candles.candle1.burn_rate > 0)
  (h2 : candles.candle2.burn_rate > 0)
  (h3 : candles.candle3.burn_rate > 0)
  (h4 : candles.candle2.remaining = 2/5)
  (h5 : candles.candle3.remaining = 3/7) :
  candles.candle3.remaining - (candles.candle2.remaining / candles.candle2.burn_rate) * candles.candle3.burn_rate = 1/21 := by
  sorry

end candle_burning_theorem_l3834_383459


namespace infinitely_many_special_pairs_l3834_383458

theorem infinitely_many_special_pairs :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a : ℤ) > 0 ∧ (b : ℤ) > 0 ∧
    (∃ k : ℤ, (a : ℤ) * b + 1 = k * ((a : ℤ) + b)) ∧
    (∃ m : ℤ, (a : ℤ) * b - 1 = m * ((a : ℤ) - b)) ∧
    (b : ℤ) > 1 ∧
    (a : ℤ) > (b : ℤ) * Real.sqrt 3 - 1 :=
by sorry

end infinitely_many_special_pairs_l3834_383458


namespace blue_balls_count_l3834_383425

theorem blue_balls_count (total : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 35 →
  yellow + blue = total →
  4 * blue = 3 * yellow →
  blue = 15 := by
sorry

end blue_balls_count_l3834_383425


namespace pet_store_inventory_l3834_383473

/-- Represents the number of birds of each type in a cage -/
structure BirdCage where
  parrots : ℕ
  parakeets : ℕ
  canaries : ℕ
  cockatiels : ℕ
  lovebirds : ℕ
  finches : ℕ

/-- The pet store inventory -/
def petStore : List BirdCage :=
  (List.replicate 7 ⟨3, 5, 4, 0, 0, 0⟩) ++
  (List.replicate 6 ⟨0, 0, 0, 2, 3, 1⟩) ++
  (List.replicate 2 ⟨0, 0, 0, 0, 0, 0⟩)

/-- Calculate the total number of birds of each type -/
def totalBirds (store : List BirdCage) : BirdCage :=
  store.foldl (fun acc cage =>
    ⟨acc.parrots + cage.parrots,
     acc.parakeets + cage.parakeets,
     acc.canaries + cage.canaries,
     acc.cockatiels + cage.cockatiels,
     acc.lovebirds + cage.lovebirds,
     acc.finches + cage.finches⟩)
    ⟨0, 0, 0, 0, 0, 0⟩

theorem pet_store_inventory :
  totalBirds petStore = ⟨21, 35, 28, 12, 18, 6⟩ := by
  sorry

end pet_store_inventory_l3834_383473


namespace petrol_price_equation_l3834_383466

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The equation representing the price reduction scenario -/
theorem petrol_price_equation : (300 / P + 7) * (0.85 * P) = 300 := by sorry

end petrol_price_equation_l3834_383466


namespace log_inequality_l3834_383409

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ :=
  sorry

theorem log_inequality (n : ℕ+) :
  Real.log n ≥ (num_distinct_prime_factors n : ℝ) * Real.log 2 :=
sorry

end log_inequality_l3834_383409


namespace infinite_geometric_series_first_term_l3834_383411

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 24) :
  let a := S * (1 - r)
  a = 18 := by
sorry

end infinite_geometric_series_first_term_l3834_383411


namespace f_value_at_8pi_3_l3834_383431

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_3 :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + π) = f x) →  -- f has period π
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sqrt 3 * Real.tan x - 1) →  -- definition of f on [0, π/2)
  f (8*π/3) = 2 := by sorry

end f_value_at_8pi_3_l3834_383431


namespace sequence_properties_l3834_383460

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of the first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of the arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of T_n as the sum of the first n terms of a_n * b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  (∀ k, S k + a k = 1) ∧
  (b 1 + b 2 = b 3) ∧
  (b 3 = 3) →
  (S n = 1 - (1/2)^n) ∧
  (T n = 2 - (n + 2) * (1/2)^n) := by
  sorry

end sequence_properties_l3834_383460


namespace cookie_solution_l3834_383491

def cookie_problem (initial_cookies : ℕ) : Prop :=
  let andy_ate : ℕ := 3
  let brother_ate : ℕ := 5
  let team_size : ℕ := 8
  let team_sequence : List ℕ := List.range team_size |>.map (λ n => 2*n + 1)
  let team_ate : ℕ := team_sequence.sum
  initial_cookies = andy_ate + brother_ate + team_ate

theorem cookie_solution : 
  ∃ (initial_cookies : ℕ), cookie_problem initial_cookies ∧ initial_cookies = 72 := by
  sorry

end cookie_solution_l3834_383491


namespace degree_to_radian_conversion_l3834_383456

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) :
  angle_in_degrees = 1440 →
  (angle_in_degrees * (π / 180)) = 8 * π :=
by sorry

end degree_to_radian_conversion_l3834_383456


namespace max_area_and_optimal_length_l3834_383477

/-- Represents the dimensions and cost of a rectangular house. -/
structure House where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of the house
  coloredSteelPrice : ℝ  -- Price per meter of colored steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof

/-- Calculates the material cost of the house. -/
def materialCost (h : House) : ℝ :=
  2 * h.x * h.coloredSteelPrice * h.h +
  2 * h.y * h.compositeSteelPrice * h.h +
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : House) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : House)
    (height_constraint : h.h = 2.5)
    (colored_steel_price : h.coloredSteelPrice = 450)
    (composite_steel_price : h.compositeSteelPrice = 200)
    (roof_price : h.roofPrice = 200)
    (cost_constraint : materialCost h ≤ 32000) :
    (∃ (max_area : ℝ) (optimal_x : ℝ),
      max_area = 100 ∧
      optimal_x = 20 / 3 ∧
      area h ≤ max_area ∧
      (area h = max_area ↔ h.x = optimal_x)) := by
  sorry


end max_area_and_optimal_length_l3834_383477


namespace flower_beds_count_l3834_383402

theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) : 
  seeds_per_bed = 10 → total_seeds = 60 → num_beds * seeds_per_bed = total_seeds → num_beds = 6 :=
by
  sorry

end flower_beds_count_l3834_383402


namespace bus_speed_without_stoppages_l3834_383401

theorem bus_speed_without_stoppages 
  (speed_with_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stoppages = 45) 
  (h2 : stop_time = 10) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 54 :=
by
  sorry

end bus_speed_without_stoppages_l3834_383401


namespace tournament_players_l3834_383454

-- Define the number of Asian players
variable (n : ℕ)

-- Define the number of European players as 2n
def european_players := 2 * n

-- Define the total number of matches
def total_matches := n * (n - 1) / 2 + (2 * n) * (2 * n - 1) / 2 + 2 * n^2

-- Define the number of matches won by Europeans
def european_wins (x : ℕ) := (2 * n) * (2 * n - 1) / 2 + x

-- Define the number of matches won by Asians
def asian_wins (x : ℕ) := n * (n - 1) / 2 + 2 * n^2 - x

-- State the theorem
theorem tournament_players :
  ∃ x : ℕ, european_wins n x = (5 / 7) * asian_wins n x ∧ n = 3 ∧ n + european_players n = 9 := by
  sorry


end tournament_players_l3834_383454


namespace burger_cost_is_110_l3834_383499

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 110

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

theorem burger_cost_is_110 :
  (∃ (s : ℕ), 4 * burger_cost + 3 * s = 440 ∧ 3 * burger_cost + 2 * s = 330) →
  burger_cost = 110 := by
  sorry

end burger_cost_is_110_l3834_383499


namespace simplify_fraction_l3834_383448

theorem simplify_fraction : 
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7) = 
  -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7) := by
  sorry

end simplify_fraction_l3834_383448


namespace grape_rate_specific_grape_rate_l3834_383476

/-- The rate of grapes per kg given the following conditions:
  1. 8 kg of grapes were purchased at an unknown rate
  2. 9 kg of mangoes were purchased at 50 rupees per kg
  3. The total amount paid was 1010 rupees -/
theorem grape_rate : ℕ → ℕ → ℕ → ℕ → Prop :=
  λ grape_quantity mango_quantity mango_rate total_paid =>
    ∃ (G : ℕ),
      grape_quantity * G + mango_quantity * mango_rate = total_paid ∧
      G = 70

/-- The specific instance of the problem -/
theorem specific_grape_rate : grape_rate 8 9 50 1010 := by
  sorry

end grape_rate_specific_grape_rate_l3834_383476


namespace probability_defective_smartphones_l3834_383450

/-- Represents the probability of selecting two defective smartphones --/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones --/
theorem probability_defective_smartphones :
  let total := 250
  let type_a_total := 100
  let type_a_defective := 30
  let type_b_total := 80
  let type_b_defective := 25
  let type_c_total := 70
  let type_c_defective := 21
  let total_defective := type_a_defective + type_b_defective + type_c_defective
  abs (probability_two_defective total total_defective - 0.0916) < 0.0001 :=
sorry

end probability_defective_smartphones_l3834_383450


namespace path_time_equality_implies_distance_ratio_l3834_383420

/-- Given two points A and B, and a point P between them, 
    if the time to go directly from P to B equals the time to go from P to A 
    and then from A to B at 6 times the speed, 
    then the ratio of PA to PB is 5/7 -/
theorem path_time_equality_implies_distance_ratio 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h_between : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is between A and B
  (speed : ℝ) -- walking speed
  (h_speed_pos : speed > 0) -- speed is positive
  : (dist P B / speed = dist P A / speed + (dist A B) / (6 * speed)) → 
    (dist P A / dist P B = 5 / 7) :=
by sorry

#check path_time_equality_implies_distance_ratio

end path_time_equality_implies_distance_ratio_l3834_383420


namespace keychain_cost_decrease_l3834_383482

theorem keychain_cost_decrease (P : ℝ) : 
  P > 0 →                           -- Selling price is positive
  P - 50 = 0.5 * P →                -- New profit is 50% of selling price
  P - 0.75 * P = 0.25 * P →         -- Initial profit was 25% of selling price
  0.75 * P = 75 :=                  -- Initial cost was $75
by
  sorry

end keychain_cost_decrease_l3834_383482


namespace factor_expression_l3834_383472

theorem factor_expression (a : ℝ) : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) := by
  sorry

end factor_expression_l3834_383472


namespace log_base_4_properties_l3834_383416

noncomputable def y (x : ℝ) : ℝ := Real.log x / Real.log 4

theorem log_base_4_properties :
  (∀ x : ℝ, x = 1 → y x = 0) ∧
  (∀ x : ℝ, x = 4 → y x = 1) ∧
  (∀ x : ℝ, x = -4 → ¬ ∃ (r : ℝ), y x = r) ∧
  (∀ x : ℝ, 0 < x → x < 1 → y x < 0 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x', 0 < x' → x' < δ → y x' < -ε) :=
by sorry

end log_base_4_properties_l3834_383416


namespace genevieve_cherries_l3834_383475

/-- The number of kilograms of cherries Genevieve bought -/
def cherries_bought : ℕ := 277

/-- The original price of cherries per kilogram in cents -/
def original_price : ℕ := 800

/-- The discount percentage on cherries -/
def discount_percentage : ℚ := 1 / 10

/-- The amount Genevieve was short in cents -/
def short_amount : ℕ := 40000

/-- The amount Genevieve had in cents -/
def genevieve_amount : ℕ := 160000

/-- Theorem stating that given the conditions, Genevieve bought 277 kilograms of cherries -/
theorem genevieve_cherries :
  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_price : ℕ := genevieve_amount + short_amount
  (total_price : ℚ) / discounted_price = cherries_bought := by sorry

end genevieve_cherries_l3834_383475


namespace correct_addition_result_l3834_383429

theorem correct_addition_result 
  (correct_addend : ℕ)
  (mistaken_addend : ℕ)
  (other_addend : ℕ)
  (mistaken_result : ℕ)
  (h1 : correct_addend = 420)
  (h2 : mistaken_addend = 240)
  (h3 : mistaken_result = mistaken_addend + other_addend)
  : correct_addend + other_addend = 570 :=
by
  sorry

end correct_addition_result_l3834_383429


namespace cos_sin_inequalities_l3834_383403

theorem cos_sin_inequalities (x : ℝ) (h : x > 0) : 
  (Real.cos x > 1 - x^2 / 2) ∧ (Real.sin x > x - x^3 / 6) := by sorry

end cos_sin_inequalities_l3834_383403


namespace alex_sandwich_production_l3834_383404

/-- Given that Alex can prepare 18 sandwiches using 3 loaves of bread,
    this theorem proves that he can make 60 sandwiches with 10 loaves of bread. -/
theorem alex_sandwich_production (sandwiches_per_three_loaves : ℕ) 
    (h1 : sandwiches_per_three_loaves = 18) : 
    (sandwiches_per_three_loaves / 3) * 10 = 60 := by
  sorry

#check alex_sandwich_production

end alex_sandwich_production_l3834_383404


namespace triangle_inequality_with_sum_zero_l3834_383438

theorem triangle_inequality_with_sum_zero (a b c p q r : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) → 
  (p + q + r = 0) → 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end triangle_inequality_with_sum_zero_l3834_383438


namespace min_value_implies_a_eq_one_l3834_383447

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x - 2 + 2 * a * Real.log x

theorem min_value_implies_a_eq_one (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a x = 0) →
  a = 1 := by sorry

end min_value_implies_a_eq_one_l3834_383447


namespace total_students_l3834_383462

theorem total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : 
  boys_ratio = 8 → girls_ratio = 5 → num_girls = 160 → 
  (boys_ratio + girls_ratio) * (num_girls / girls_ratio) = 416 := by
sorry

end total_students_l3834_383462


namespace loan_split_l3834_383451

theorem loan_split (total : ℝ) (years1 rate1 years2 rate2 : ℝ) 
  (h1 : total = 2704)
  (h2 : years1 = 8)
  (h3 : rate1 = 0.03)
  (h4 : years2 = 3)
  (h5 : rate2 = 0.05)
  (h6 : ∃ x : ℝ, x * years1 * rate1 = (total - x) * years2 * rate2) :
  ∃ y : ℝ, y = total - 1664 ∧ y * years1 * rate1 = (total - y) * years2 * rate2 := by
  sorry

end loan_split_l3834_383451


namespace derivative_implies_power_l3834_383407

/-- Given a function f(x) = m * x^(m-n) where its derivative f'(x) = 8 * x^3,
    prove that m^n = 1/4 -/
theorem derivative_implies_power (m n : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m * x^(m-n))
  (h2 : ∀ x, deriv f x = 8 * x^3) :
  m^n = 1/4 := by
  sorry

end derivative_implies_power_l3834_383407


namespace three_over_x_equals_one_l3834_383414

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end three_over_x_equals_one_l3834_383414


namespace linear_function_slope_l3834_383423

theorem linear_function_slope (x₁ x₂ y₁ y₂ m : ℝ) :
  x₁ > x₂ →
  y₁ > y₂ →
  y₁ = (m - 3) * x₁ - 4 →
  y₂ = (m - 3) * x₂ - 4 →
  m > 3 := by
  sorry

end linear_function_slope_l3834_383423


namespace train_speed_calculation_l3834_383480

/-- Proves that given a round trip of 240 miles total, where the return trip speed is 38.71 miles per hour, 
and the total travel time is 5.5 hours, the speed of the first leg of the trip is 50 miles per hour. -/
theorem train_speed_calculation (total_distance : ℝ) (return_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 240)
  (h2 : return_speed = 38.71)
  (h3 : total_time = 5.5) :
  ∃ (outbound_speed : ℝ), outbound_speed = 50 ∧ 
  (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed = total_time :=
by
  sorry

#check train_speed_calculation

end train_speed_calculation_l3834_383480


namespace squares_ending_in_identical_digits_l3834_383483

def endsIn (n : ℤ) (d : ℤ) : Prop := n % (10 ^ (d.natAbs + 1)) = d

theorem squares_ending_in_identical_digits :
  (∀ n : ℤ, (endsIn n 12 ∨ endsIn n 38 ∨ endsIn n 62 ∨ endsIn n 88) → endsIn (n^2) 44) ∧
  (∀ m : ℤ, (endsIn m 038 ∨ endsIn m 462 ∨ endsIn m 538 ∨ endsIn m 962) → endsIn (m^2) 444) :=
by sorry

end squares_ending_in_identical_digits_l3834_383483


namespace taxicab_distance_properties_l3834_383439

/-- Taxicab distance between two points -/
def taxicab_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- Check if a point is on a line segment -/
def on_segment (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ c = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2))

/-- The set of points equidistant from two given points -/
def equidistant_set (m n : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | taxicab_distance p m = taxicab_distance p n}

theorem taxicab_distance_properties :
  (∀ a b c : ℝ × ℝ, on_segment a b c → taxicab_distance a c + taxicab_distance c b = taxicab_distance a b) ∧
  ¬(∀ a b c : ℝ × ℝ, taxicab_distance a c + taxicab_distance c b > taxicab_distance a b) ∧
  equidistant_set (-1, 0) (1, 0) = {p : ℝ × ℝ | p.1 = 0} ∧
  (∀ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 → taxicab_distance (0, 0) p ≥ 2 * Real.sqrt 5) ∧
  (∃ p : ℝ × ℝ, p.1 + p.2 = 2 * Real.sqrt 5 ∧ taxicab_distance (0, 0) p = 2 * Real.sqrt 5) := by
  sorry

end taxicab_distance_properties_l3834_383439


namespace no_positive_real_roots_l3834_383435

theorem no_positive_real_roots (m : ℝ) : 
  (∀ x > 0, (x^2 + (5-2*m)*x + m-3) / (x-1) ≠ 2*x + m) ↔ m = 3 :=
by sorry

end no_positive_real_roots_l3834_383435


namespace proposition_p_false_and_q_true_l3834_383418

theorem proposition_p_false_and_q_true :
  (∃ x : ℝ, 2^x ≤ x^2) ∧
  ((∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
   (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1))) :=
by sorry

end proposition_p_false_and_q_true_l3834_383418


namespace total_slices_equals_twelve_l3834_383496

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem total_slices_equals_twelve : total_slices = 12 := by
  sorry

end total_slices_equals_twelve_l3834_383496


namespace shaded_square_area_ratio_l3834_383445

/-- The ratio of the area of a square with vertices at (2,3), (4,3), (4,5), and (2,5) 
    to the area of a 6x6 square is 1/9. -/
theorem shaded_square_area_ratio : 
  let grid_size : ℕ := 6
  let vertex1 : (ℕ × ℕ) := (2, 3)
  let vertex2 : (ℕ × ℕ) := (4, 3)
  let vertex3 : (ℕ × ℕ) := (4, 5)
  let vertex4 : (ℕ × ℕ) := (2, 5)
  let shaded_square_side : ℕ := vertex2.1 - vertex1.1
  let shaded_square_area : ℕ := shaded_square_side * shaded_square_side
  let grid_area : ℕ := grid_size * grid_size
  (shaded_square_area : ℚ) / grid_area = 1 / 9 := by
sorry

end shaded_square_area_ratio_l3834_383445


namespace projectile_max_height_l3834_383452

def f (t : ℝ) : ℝ := -8 * t^2 + 64 * t + 36

theorem projectile_max_height :
  ∃ (max : ℝ), max = 164 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end projectile_max_height_l3834_383452


namespace triangle_dot_product_l3834_383453

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area √3, prove AB · AC = ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  (AB.1 * AC.1 + AB.2 * AC.2 = 2) ∨ (AB.1 * AC.1 + AB.2 * AC.2 = -2) :=
by sorry

end triangle_dot_product_l3834_383453


namespace class_average_problem_l3834_383484

/-- Given a class of 25 students where 10 students average 88% and the overall average is 79%,
    this theorem proves that the average percentage of the remaining 15 students is 73%. -/
theorem class_average_problem (total_students : Nat) (group_a_students : Nat) (group_b_students : Nat)
    (group_b_average : ℝ) (overall_average : ℝ) :
    total_students = 25 →
    group_a_students = 15 →
    group_b_students = 10 →
    group_b_average = 88 →
    overall_average = 79 →
    (group_a_students * x + group_b_students * group_b_average) / total_students = overall_average →
    x = 73 :=
  by sorry


end class_average_problem_l3834_383484


namespace regular_polygon_sides_l3834_383478

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n ≥ 3 → 
  exterior_angle = 45 →
  (360 : ℝ) / exterior_angle = n →
  n = 8 := by
  sorry

end regular_polygon_sides_l3834_383478


namespace sum_of_coefficients_l3834_383426

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^3 - (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = -54 := by
sorry

end sum_of_coefficients_l3834_383426


namespace inequality_equivalence_l3834_383449

theorem inequality_equivalence (x : ℝ) :
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by
sorry

end inequality_equivalence_l3834_383449


namespace isosceles_triangles_same_perimeter_area_l3834_383410

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ  -- length of equal sides
  base : ℕ  -- length of the base
  isIsosceles : leg > 0 ∧ base > 0

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ)^2 - ((t.base : ℝ)/2)^2).sqrt) / 2

/-- Theorem: There exist two non-congruent isosceles triangles with integer side lengths,
    having the same perimeter and area, where the ratio of their bases is 5:4,
    and we can determine the minimum possible value of their common perimeter. -/
theorem isosceles_triangles_same_perimeter_area :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter t1 ≤ perimeter s1 := by
  sorry

end isosceles_triangles_same_perimeter_area_l3834_383410


namespace smallest_max_sum_l3834_383422

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_constraint : a + b + c + d + e = 3015) : 
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧ 
   ∀ N : ℕ, (N = max (a + b) (max (b + c) (max (c + d) (d + e))) → M ≤ N) ∧
   M = 755) := by
  sorry

end smallest_max_sum_l3834_383422


namespace noahs_closet_capacity_l3834_383468

theorem noahs_closet_capacity (ali_capacity : ℕ) (noah_total_capacity : ℕ) : 
  ali_capacity = 200 → noah_total_capacity = 100 → 
  (noah_total_capacity / 2 : ℚ) / ali_capacity = 1/4 := by
  sorry

end noahs_closet_capacity_l3834_383468
