import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2286_228692

theorem equation_solution : 
  ∃ (x : ℝ), x ≥ 0 ∧ 2021 * x = 2022 * (x^2021)^(1/2022) - 1 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2286_228692


namespace NUMINAMATH_CALUDE_crabapple_sequences_count_l2286_228685

/-- The number of students in each class -/
def students_per_class : ℕ := 8

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The number of classes -/
def number_of_classes : ℕ := 2

/-- The total number of sequences of crabapple recipients for both classes in a week -/
def total_sequences : ℕ := (students_per_class ^ meetings_per_week) ^ number_of_classes

/-- Theorem stating that the total number of sequences is 262,144 -/
theorem crabapple_sequences_count : total_sequences = 262144 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_count_l2286_228685


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2286_228641

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a row contains 2, 3, and 4 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g row 0, g row 1, g row 2}

/-- Checks if a column contains 2, 3, and 4 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ({2, 3, 4} : Finset Nat) = {g 0 col, g 1 col, g 2 col}

/-- Checks if the grid satisfies all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  g 0 0 = 2 ∧
  g 1 1 = 3

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) : g 2 0 + g 0 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2286_228641


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2286_228688

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2286_228688


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2286_228648

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) : 
  distance = 500 ∧ 
  speed_increase = 10 ∧ 
  time_decrease = 2 →
  ∃ (v : ℝ), v > 0 ∧ 
    distance / v - distance / (v + speed_increase) = time_decrease ∧ 
    v = 45.25 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2286_228648


namespace NUMINAMATH_CALUDE_stratified_sampling_under_40_l2286_228635

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 400

/-- Represents the number of teachers under 40 -/
def teachers_under_40 : ℕ := 250

/-- Represents the total sample size -/
def sample_size : ℕ := 80

/-- Calculates the number of teachers under 40 in the sample -/
def sample_under_40 : ℕ := (teachers_under_40 * sample_size) / total_teachers

theorem stratified_sampling_under_40 :
  sample_under_40 = 50 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_under_40_l2286_228635


namespace NUMINAMATH_CALUDE_circle_radius_l2286_228651

/-- The radius of the circle defined by x^2 + y^2 - 8x = 0 is 4 -/
theorem circle_radius (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2286_228651


namespace NUMINAMATH_CALUDE_coral_reading_pages_l2286_228608

def pages_night1 : ℕ := 30

def pages_night2 : ℕ := 2 * pages_night1 - 2

def pages_night3 : ℕ := pages_night1 + pages_night2 + 3

def total_pages : ℕ := pages_night1 + pages_night2 + pages_night3

theorem coral_reading_pages : total_pages = 179 := by
  sorry

end NUMINAMATH_CALUDE_coral_reading_pages_l2286_228608


namespace NUMINAMATH_CALUDE_number_problem_l2286_228632

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n / sum = quotient ∧ n % sum = 30 → n = 220030 :=
by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2286_228632


namespace NUMINAMATH_CALUDE_prob_receive_one_out_of_two_prob_receive_at_least_ten_l2286_228628

/-- The probability of receiving a red envelope for each recipient -/
def prob_receive : ℚ := 1 / 3

/-- The probability of not receiving a red envelope for each recipient -/
def prob_not_receive : ℚ := 2 / 3

/-- The number of recipients -/
def num_recipients : ℕ := 3

/-- The number of red envelopes sent in the first scenario -/
def num_envelopes_1 : ℕ := 2

/-- The number of red envelopes sent in the second scenario -/
def num_envelopes_2 : ℕ := 3

/-- The amounts in the red envelopes for the second scenario -/
def envelope_amounts : List ℚ := [5, 5, 10]

/-- Theorem 1: Probability of receiving exactly one envelope out of two -/
theorem prob_receive_one_out_of_two :
  let p := prob_receive
  let q := prob_not_receive
  p * q + q * p = 4 / 9 := by sorry

/-- Theorem 2: Probability of receiving at least 10 yuan out of three envelopes -/
theorem prob_receive_at_least_ten :
  let p := prob_receive
  let q := prob_not_receive
  p^2 * q + 2 * p^2 * q + p^3 = 11 / 27 := by sorry

end NUMINAMATH_CALUDE_prob_receive_one_out_of_two_prob_receive_at_least_ten_l2286_228628


namespace NUMINAMATH_CALUDE_gift_payment_l2286_228619

theorem gift_payment (total : ℝ) (alice bob carlos : ℝ) : 
  total = 120 ∧ 
  alice = (1/3) * (bob + carlos) ∧ 
  bob = (1/4) * (alice + carlos) ∧ 
  total = alice + bob + carlos → 
  carlos = 72 := by
sorry

end NUMINAMATH_CALUDE_gift_payment_l2286_228619


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l2286_228650

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l2286_228650


namespace NUMINAMATH_CALUDE_base6_addition_l2286_228659

-- Define a function to convert a base-6 number (represented as a list of digits) to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to its base-6 representation
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition :
  base6ToNat [4, 5, 1, 2] + base6ToNat [2, 3, 4, 5, 3] = base6ToNat [3, 4, 4, 0, 5] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l2286_228659


namespace NUMINAMATH_CALUDE_total_time_in_hours_l2286_228668

def laundry_time : ℕ := 30
def bathroom_cleaning_time : ℕ := 15
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40

def minutes_per_hour : ℕ := 60

theorem total_time_in_hours :
  (laundry_time + bathroom_cleaning_time + room_cleaning_time + homework_time) / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_time_in_hours_l2286_228668


namespace NUMINAMATH_CALUDE_percentage_of_apples_after_adding_l2286_228627

/-- Given a basket of fruits with the following conditions:
  * x is the initial number of apples
  * y is the initial number of oranges
  * z is the number of oranges added
  * w is the number of apples added
  * The sum of initial apples and oranges is 30
  * The sum of added oranges and apples is 12
  * The ratio of initial apples to initial oranges is 2:1
  * The ratio of added apples to added oranges is 3:1
  Prove that the percentage of apples in the basket after adding extra fruits is (29/42) * 100 -/
theorem percentage_of_apples_after_adding (x y z w : ℕ) : 
  x + y = 30 →
  z + w = 12 →
  x = 2 * y →
  w = 3 * z →
  (x + w : ℚ) / (x + y + z + w) * 100 = 29 / 42 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_apples_after_adding_l2286_228627


namespace NUMINAMATH_CALUDE_complex_equality_l2286_228647

theorem complex_equality (z : ℂ) : z = -15/8 + 5/4*I → Complex.abs (z - 2*I) = Complex.abs (z + 4) ∧ Complex.abs (z - 2*I) = Complex.abs (z + I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2286_228647


namespace NUMINAMATH_CALUDE_diagonal_length_l2286_228621

structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)
  (diagonal_bisects : sorry)
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_AD : dist B C = dist A D)
  (AB_length : dist A B = 5)
  (BC_length : dist B C = 3)

/-- The length of the diagonal AC in the given parallelogram is 5√2 -/
theorem diagonal_length (p : Parallelogram) : dist p.A p.C = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l2286_228621


namespace NUMINAMATH_CALUDE_min_value_of_f_l2286_228667

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- Define the closed interval [-1, 0]
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 0}

-- Theorem statement
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ I ∧ f x = -1 ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2286_228667


namespace NUMINAMATH_CALUDE_sqrt_20_in_terms_of_a_and_b_l2286_228617

theorem sqrt_20_in_terms_of_a_and_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 20 = a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_in_terms_of_a_and_b_l2286_228617


namespace NUMINAMATH_CALUDE_fish_pond_population_l2286_228695

/-- Proves that given the conditions of the fish tagging problem, the approximate number of fish in the pond is 1250 -/
theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 50)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch) :
  total_fish = 1250 := by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l2286_228695


namespace NUMINAMATH_CALUDE_median_could_be_16_l2286_228642

/-- Represents the age distribution of the school band --/
structure AgeDist where
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Calculates the total number of members in the band --/
def totalMembers (dist : AgeDist) : Nat :=
  dist.age13 + dist.age14 + dist.age15 + dist.age16

/-- Checks if a given age is the median of the distribution --/
def isMedian (dist : AgeDist) (age : Nat) : Prop :=
  let total := totalMembers dist
  let halfTotal := total / 2
  let countBelow := 
    if age == 13 then 0
    else if age == 14 then dist.age13
    else if age == 15 then dist.age13 + dist.age14
    else dist.age13 + dist.age14 + dist.age15
  countBelow < halfTotal ∧ countBelow + (if age == 16 then dist.age16 else 0) ≥ halfTotal

/-- The main theorem stating that 16 could be the median --/
theorem median_could_be_16 (dist : AgeDist) : 
  dist.age13 = 5 → dist.age14 = 7 → dist.age15 = 13 → ∃ n : Nat, isMedian { age13 := 5, age14 := 7, age15 := 13, age16 := n } 16 :=
sorry

end NUMINAMATH_CALUDE_median_could_be_16_l2286_228642


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2286_228630

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 6 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n > 120) ∧ 
  (n < 209) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2286_228630


namespace NUMINAMATH_CALUDE_books_on_shelves_l2286_228671

/-- The number of ways to place n distinct books onto k shelves with no empty shelf -/
def place_books (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

theorem books_on_shelves :
  place_books 10 3 * Nat.factorial 10 = 55980 * Nat.factorial 10 :=
by sorry

end NUMINAMATH_CALUDE_books_on_shelves_l2286_228671


namespace NUMINAMATH_CALUDE_juice_consumption_l2286_228604

theorem juice_consumption (total_juice : ℚ) (sam_fraction : ℚ) (alex_fraction : ℚ) :
  total_juice = 3/4 ∧ sam_fraction = 1/2 ∧ alex_fraction = 1/4 →
  sam_fraction * total_juice + alex_fraction * total_juice = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_juice_consumption_l2286_228604


namespace NUMINAMATH_CALUDE_rectangle_perimeter_increase_l2286_228624

theorem rectangle_perimeter_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let initial_perimeter := 2 * (l + w)
  let new_perimeter := 2 * (1.1 * l + 1.1 * w)
  new_perimeter / initial_perimeter = 1.1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_increase_l2286_228624


namespace NUMINAMATH_CALUDE_kimmie_earnings_l2286_228636

theorem kimmie_earnings (kimmie_earnings : ℚ) : 
  (kimmie_earnings / 2 + (2 / 3 * kimmie_earnings) / 2 = 375) → 
  kimmie_earnings = 450 := by
  sorry

end NUMINAMATH_CALUDE_kimmie_earnings_l2286_228636


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2286_228610

theorem digit_sum_problem (a b c x s z : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → x ≠ 0 → s ≠ 0 → z ≠ 0 →
  a + b = x →
  x + c = s →
  s + a = z →
  b + c + z = 16 →
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2286_228610


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l2286_228626

/-- The number of children sharing the cupcakes -/
def num_children : ℕ := 8

/-- The number of cupcakes each child gets when shared equally -/
def cupcakes_per_child : ℕ := 12

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := num_children * cupcakes_per_child

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l2286_228626


namespace NUMINAMATH_CALUDE_some_athletes_not_honor_society_l2286_228607

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (h2 : ∀ x, HonorSocietyMember x → Disciplined x)

-- Theorem to prove
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end NUMINAMATH_CALUDE_some_athletes_not_honor_society_l2286_228607


namespace NUMINAMATH_CALUDE_ternary_decimal_conversion_decimal_base7_conversion_l2286_228645

-- Define a function to convert from base 3 to base 10
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (t.length - 1 - i))) 0

-- Define a function to convert from base 10 to base 7
def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem ternary_decimal_conversion :
  ternary_to_decimal [1, 0, 2, 1, 2] = 104 := by sorry

theorem decimal_base7_conversion :
  decimal_to_base7 1234 = [3, 4, 1, 2] := by sorry

end NUMINAMATH_CALUDE_ternary_decimal_conversion_decimal_base7_conversion_l2286_228645


namespace NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l2286_228660

theorem greatest_integer_solution (x : ℤ) : (5 - 4 * x > 17) ↔ x < -3 :=
  sorry

theorem greatest_integer_value : ∀ x : ℤ, (5 - 4 * x > 17) → x ≤ -4 :=
  sorry

theorem minus_four_is_solution : 5 - 4 * (-4) > 17 :=
  sorry

theorem minus_four_is_greatest : ∀ x : ℤ, x > -4 → ¬(5 - 4 * x > 17) :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l2286_228660


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2286_228665

/-- Given vectors a and b, prove that a is perpendicular to b iff n = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) :
  a = (3, 2) → b.1 = 2 → a.1 * b.1 + a.2 * b.2 = 0 ↔ b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2286_228665


namespace NUMINAMATH_CALUDE_ten_trees_road_length_l2286_228640

/-- The length of a road with trees planted at equal intervals --/
def road_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters --/
theorem ten_trees_road_length :
  road_length 10 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ten_trees_road_length_l2286_228640


namespace NUMINAMATH_CALUDE_probability_is_half_l2286_228677

/-- An equilateral triangle divided by two medians -/
structure TriangleWithMedians where
  /-- The number of regions formed by drawing two medians in an equilateral triangle -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- The total number of regions is 6 -/
  h_total : total_regions = 6
  /-- The number of shaded regions is 3 -/
  h_shaded : shaded_regions = 3

/-- The probability of a point landing in a shaded region -/
def probability (t : TriangleWithMedians) : ℚ :=
  t.shaded_regions / t.total_regions

theorem probability_is_half (t : TriangleWithMedians) :
  probability t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l2286_228677


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2286_228655

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x where its derivative f'(x) is an even function,
    the equation of the tangent line to the curve y = f(x) at the origin is y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^3 + a*x^2 + (a - 2)*x) →
  (∀ x, (deriv f) x = f' x) →
  (∀ x, f' x = f' (-x)) →
  (∀ x, x * (-2) = f x - f 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2286_228655


namespace NUMINAMATH_CALUDE_mikaela_hourly_rate_l2286_228646

/-- Mikaela's tutoring earnings problem -/
theorem mikaela_hourly_rate :
  ∀ (hourly_rate : ℝ),
  let first_month_hours : ℝ := 35
  let second_month_hours : ℝ := first_month_hours + 5
  let total_hours : ℝ := first_month_hours + second_month_hours
  let total_earnings : ℝ := total_hours * hourly_rate
  let personal_needs_fraction : ℝ := 4/5
  let savings : ℝ := 150
  (personal_needs_fraction * total_earnings + savings = total_earnings) →
  hourly_rate = 10 := by
sorry


end NUMINAMATH_CALUDE_mikaela_hourly_rate_l2286_228646


namespace NUMINAMATH_CALUDE_square_roots_to_N_l2286_228613

theorem square_roots_to_N (m : ℝ) (N : ℝ) : 
  (3 * m - 4) ^ 2 = N ∧ (7 - 4 * m) ^ 2 = N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_to_N_l2286_228613


namespace NUMINAMATH_CALUDE_petya_win_probability_l2286_228663

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The state of the game -/
structure GameState where
  stones : Nat
  currentPlayer : Player

/-- The outcome of the game -/
inductive GameOutcome
  | PetyaWins
  | ComputerWins

/-- A strategy for playing the game -/
def Strategy := GameState → Nat

/-- The random strategy that Petya uses -/
def randomStrategy : Strategy := sorry

/-- The optimal strategy that the computer uses -/
def optimalStrategy : Strategy := sorry

/-- Play the game with given strategies -/
def playGame (petyaStrategy : Strategy) (computerStrategy : Strategy) : GameOutcome := sorry

/-- The probability of Petya winning -/
def petyaWinProbability : ℚ := sorry

/-- Main theorem: The probability of Petya winning is 1/256 -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 1, 4⟩
  petyaWinProbability = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_l2286_228663


namespace NUMINAMATH_CALUDE_probability_penny_dime_halfdollar_heads_l2286_228699

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def total_outcomes : ℕ := 64

/-- Predicate for the desired outcome (penny, dime, and half-dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.half_dollar = CoinFlip.Heads

/-- The number of outcomes satisfying the desired condition -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating the probability of the desired outcome -/
theorem probability_penny_dime_halfdollar_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_penny_dime_halfdollar_heads_l2286_228699


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2286_228696

theorem triangle_angle_c (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  3 * Real.sin A + 4 * Real.cos B = 6 ∧
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2286_228696


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2286_228686

theorem complex_equation_solution (z : ℂ) : 3 + 2 * Complex.I * z = 7 - 4 * Complex.I * z ↔ z = -2 * Complex.I / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2286_228686


namespace NUMINAMATH_CALUDE_existence_of_x_l2286_228611

/-- A sequence of nonnegative integers satisfying the given condition -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The main theorem -/
theorem existence_of_x (a : ℕ → ℕ) (h : SequenceCondition a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_l2286_228611


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l2286_228622

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 
  24 ∣ n^2 ∧ 
  1024 ∣ n^3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 1024 ∣ m^3)) → 
  n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l2286_228622


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2286_228676

theorem expand_and_simplify (y : ℝ) : 5 * (4 * y^2 - 3 * y + 2) = 20 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2286_228676


namespace NUMINAMATH_CALUDE_article_cost_price_l2286_228698

/-- Given an article with a 15% markup, sold at Rs. 456 after a 26.570048309178745% discount,
    prove that the cost price of the article is Rs. 540. -/
theorem article_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (discount_percentage : ℝ)
    (h1 : markup_percentage = 15)
    (h2 : selling_price = 456)
    (h3 : discount_percentage = 26.570048309178745) :
    ∃ (cost_price : ℝ),
      cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = selling_price ∧
      cost_price = 540 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2286_228698


namespace NUMINAMATH_CALUDE_charitable_woman_age_l2286_228612

theorem charitable_woman_age (x : ℚ) : 
  (x / 2 + 1) + ((x / 2 - 1) / 2 + 2) + ((x / 4 - 3 / 2) / 2 + 3) + 1 = x → x = 38 :=
by sorry

end NUMINAMATH_CALUDE_charitable_woman_age_l2286_228612


namespace NUMINAMATH_CALUDE_pokemon_game_l2286_228681

theorem pokemon_game (n : ℕ) : 
  (∃ (m : ℕ), 
    n * m + 11 * (m + 6) = n^2 + 3*n - 2 ∧ 
    m > 0 ∧ 
    (m + 6) > 0) → 
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_pokemon_game_l2286_228681


namespace NUMINAMATH_CALUDE_line_equation_with_slope_and_area_l2286_228687

theorem line_equation_with_slope_and_area (x y : ℝ) :
  ∃ (b : ℝ), (3 * x - 4 * y + 12 * b = 0 ∨ 3 * x - 4 * y - 12 * b = 0) ∧
  (3 / 4 : ℝ) = (y - 0) / (x - 0) ∧
  6 = (1 / 2) * |0 - x| * |0 - y| :=
sorry

end NUMINAMATH_CALUDE_line_equation_with_slope_and_area_l2286_228687


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2286_228670

theorem sum_of_squares_and_square_of_sum : (5 + 7)^2 + (5^2 + 7^2) = 218 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2286_228670


namespace NUMINAMATH_CALUDE_total_points_is_24_l2286_228674

/-- Calculates points earned based on pounds recycled and points per set of pounds -/
def calculatePoints (pounds : ℕ) (poundsPerSet : ℕ) (pointsPerSet : ℕ) : ℕ :=
  (pounds / poundsPerSet) * pointsPerSet

/-- Represents the recycling problem and calculates total points -/
def recyclingProblem : ℕ :=
  let gwenPoints := calculatePoints 12 4 2
  let lisaPoints := calculatePoints 25 5 3
  let jackPoints := calculatePoints 21 7 1
  gwenPoints + lisaPoints + jackPoints

/-- Theorem stating that the total points earned is 24 -/
theorem total_points_is_24 : recyclingProblem = 24 := by
  sorry


end NUMINAMATH_CALUDE_total_points_is_24_l2286_228674


namespace NUMINAMATH_CALUDE_sum_of_negatives_l2286_228643

theorem sum_of_negatives : (-3) + (-9) = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_negatives_l2286_228643


namespace NUMINAMATH_CALUDE_candidate_score_approx_45_l2286_228633

-- Define the maximum marks for Paper I
def max_marks : ℝ := 127.27

-- Define the passing percentage
def passing_percentage : ℝ := 0.55

-- Define the margin by which the candidate failed
def failing_margin : ℝ := 25

-- Define the candidate's score
def candidate_score : ℝ := max_marks * passing_percentage - failing_margin

-- Theorem to prove
theorem candidate_score_approx_45 : 
  ∃ ε > 0, abs (candidate_score - 45) < ε :=
sorry

end NUMINAMATH_CALUDE_candidate_score_approx_45_l2286_228633


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2286_228644

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs -/
def pairSequence : ℕ → IntPair :=
  sorry

/-- The 60th pair in the sequence -/
def sixtiethPair : IntPair :=
  pairSequence 60

theorem sixtieth_pair_is_five_seven :
  sixtiethPair = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2286_228644


namespace NUMINAMATH_CALUDE_order_relation_l2286_228657

theorem order_relation (a b c : ℝ) : 
  a = (1 : ℝ) / 2023 →
  b = Real.exp (-(2022 : ℝ) / 2023) →
  c = Real.cos ((1 : ℝ) / 2023) / 2023 →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l2286_228657


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l2286_228606

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) 
  (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l2286_228606


namespace NUMINAMATH_CALUDE_waitress_tips_l2286_228662

theorem waitress_tips (salary : ℝ) (tips : ℝ) (h1 : salary > 0) (h2 : tips > 0) :
  tips / (salary + tips) = 1/3 → tips / salary = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_waitress_tips_l2286_228662


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l2286_228614

/-- The floor on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  let steps_per_floor := petya_steps / 2
  1 + vasya_steps / steps_per_floor

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l2286_228614


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2286_228618

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 953 ∧ (218791 - x) % 953 = 0 ∧ ∀ y : ℕ, y < x → (218791 - y) % 953 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2286_228618


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_seconds_l2286_228615

-- Define the motion equation
def s (t : ℝ) : ℝ := (2 * t + 3) ^ 2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 4 * (2 * t + 3)

-- Theorem statement
theorem instantaneous_velocity_at_2_seconds : v 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_seconds_l2286_228615


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l2286_228625

/-- The sum of internal angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The sum of the five known angles in the hexagon -/
def known_angles_sum : ℝ := 130 + 100 + 105 + 115 + 95

/-- Theorem: In a hexagon where five of the internal angles measure 130°, 100°, 105°, 115°, and 95°,
    the measure of the sixth angle is 175°. -/
theorem sixth_angle_measure :
  hexagon_angle_sum - known_angles_sum = 175 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l2286_228625


namespace NUMINAMATH_CALUDE_eighth_pentagon_shaded_fraction_l2286_228666

/-- Triangular number sequence -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Pentagonal number sequence -/
def pentagonal (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- Total sections in the nth pentagon -/
def total_sections (n : ℕ) : ℕ := n^2

/-- Shaded sections in the nth pentagon -/
def shaded_sections (n : ℕ) : ℕ :=
  if n % 2 = 1 then triangular (n / 2 + 1)
  else pentagonal (n / 2)

theorem eighth_pentagon_shaded_fraction :
  (shaded_sections 8 : ℚ) / (total_sections 8 : ℚ) = 11 / 32 := by
  sorry

end NUMINAMATH_CALUDE_eighth_pentagon_shaded_fraction_l2286_228666


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l2286_228678

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 1

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 13 → n ≤ 7111111 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l2286_228678


namespace NUMINAMATH_CALUDE_doll_completion_time_l2286_228691

/-- Time in minutes to craft one doll -/
def craft_time : ℕ := 105

/-- Break time in minutes -/
def break_time : ℕ := 30

/-- Number of dolls to be made -/
def num_dolls : ℕ := 10

/-- Number of dolls after which a break is taken -/
def dolls_per_break : ℕ := 3

/-- Start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

theorem doll_completion_time :
  let total_craft_time := num_dolls * craft_time
  let total_breaks := (num_dolls / dolls_per_break) * break_time
  let total_time := total_craft_time + total_breaks
  let completion_time := (start_time + total_time) % (24 * 60)
  completion_time = 5 * 60 :=
by sorry

end NUMINAMATH_CALUDE_doll_completion_time_l2286_228691


namespace NUMINAMATH_CALUDE_square_root_23_minus_one_expression_l2286_228694

theorem square_root_23_minus_one_expression : 
  let x : ℝ := Real.sqrt 23 - 1
  x^2 + 2*x + 2 = 24 := by sorry

end NUMINAMATH_CALUDE_square_root_23_minus_one_expression_l2286_228694


namespace NUMINAMATH_CALUDE_problem_statement_l2286_228683

theorem problem_statement (x : ℝ) : 
  (1/5)^35 * (1/4)^18 = 1/(x*(10)^35) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2286_228683


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l2286_228616

theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 132 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ wholesale_price : ℝ,
    wholesale_price = 99 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l2286_228616


namespace NUMINAMATH_CALUDE_duck_flying_days_l2286_228639

/-- The number of days it takes a duck to fly south during winter -/
def days_south : ℕ := 40

/-- The number of days it takes a duck to fly north during summer -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes a duck to fly east during spring -/
def days_east : ℕ := 60

/-- The total number of days a duck flies during winter, summer, and spring -/
def total_flying_days : ℕ := days_south + days_north + days_east

/-- Theorem stating that the total number of days a duck flies during winter, summer, and spring is 180 -/
theorem duck_flying_days : total_flying_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_duck_flying_days_l2286_228639


namespace NUMINAMATH_CALUDE_max_n_value_l2286_228637

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c)) :
  n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l2286_228637


namespace NUMINAMATH_CALUDE_exam_score_percentage_l2286_228634

theorem exam_score_percentage : 
  let score1 : ℕ := 42
  let score2 : ℕ := 33
  let total_score : ℕ := score1 + score2
  (score1 : ℚ) / (total_score : ℚ) * 100 = 56 := by
sorry

end NUMINAMATH_CALUDE_exam_score_percentage_l2286_228634


namespace NUMINAMATH_CALUDE_selection_theorem_l2286_228603

/-- The number of ways to select 4 students from 7 students (4 boys and 3 girls), 
    ensuring that the selection includes both boys and girls -/
def selection_ways : ℕ :=
  Nat.choose 7 4 - Nat.choose 4 4

theorem selection_theorem : selection_ways = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2286_228603


namespace NUMINAMATH_CALUDE_existence_of_comparable_indices_l2286_228680

theorem existence_of_comparable_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_comparable_indices_l2286_228680


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_constant_l2286_228693

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, S n = n^2 + 2*n + (S 1 - 3)) →
  S 1 - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_constant_l2286_228693


namespace NUMINAMATH_CALUDE_triangle_problem_l2286_228658

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 * Real.sqrt 3 →
  B = π / 3 ∧ 
  (∀ (a' c' : ℝ), a' * c' ≤ 12) ∧
  (∃ (a' c' : ℝ), a' * c' = 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2286_228658


namespace NUMINAMATH_CALUDE_range_of_function_l2286_228654

theorem range_of_function (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2286_228654


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2286_228689

theorem partial_fraction_decomposition (A B : ℝ) :
  (∀ x : ℝ, (2*x + 1) / ((x + 1) * (x + 2)) = A / (x + 1) + B / (x + 2)) →
  A = -1 ∧ B = 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2286_228689


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_is_zero_l2286_228609

/-- The decimal representation of 1/13 -/
def decimal_1_13 : ℚ := 1 / 13

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1 / 11

/-- The sum of the decimal representations of 1/13 and 1/11 -/
def sum_decimals : ℚ := decimal_1_13 + decimal_1_11

/-- The function that returns the nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 25th digit after the decimal point in the sum of 1/13 and 1/11 is 0 -/
theorem twenty_fifth_digit_is_zero : nth_digit_after_decimal sum_decimals 25 = 0 := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_is_zero_l2286_228609


namespace NUMINAMATH_CALUDE_percentage_relationship_l2286_228602

theorem percentage_relationship (x : ℝ) (p : ℝ) :
  x = 120 →
  5.76 = p * (0.4 * x) →
  p = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2286_228602


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2286_228605

theorem function_inequality_implies_a_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ 1) :
  (∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a / x₁ ≥ x₂ - Real.log x₂) →
  Real.exp 1 - 2 ≤ a ∧ a ≤ 1 := by
  sorry

#check function_inequality_implies_a_range

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2286_228605


namespace NUMINAMATH_CALUDE_san_antonio_bound_passes_two_austin_bound_l2286_228682

/-- Represents the direction of travel for a bus -/
inductive Direction
  | AustinToSanAntonio
  | SanAntonioToAustin

/-- Represents a bus schedule -/
structure BusSchedule where
  direction : Direction
  departureInterval : ℕ  -- in hours
  departureOffset : ℕ    -- in hours

/-- Represents the bus system between Austin and San Antonio -/
structure BusSystem where
  travelTime : ℕ
  austinToSanAntonioSchedule : BusSchedule
  sanAntonioToAustinSchedule : BusSchedule

/-- Counts the number of buses passed during a journey -/
def countPassedBuses (system : BusSystem) : ℕ :=
  sorry

/-- The main theorem stating that a San Antonio-bound bus passes exactly 2 Austin-bound buses -/
theorem san_antonio_bound_passes_two_austin_bound :
  ∀ (system : BusSystem),
    system.travelTime = 3 ∧
    system.austinToSanAntonioSchedule.direction = Direction.AustinToSanAntonio ∧
    system.austinToSanAntonioSchedule.departureInterval = 2 ∧
    system.austinToSanAntonioSchedule.departureOffset = 0 ∧
    system.sanAntonioToAustinSchedule.direction = Direction.SanAntonioToAustin ∧
    system.sanAntonioToAustinSchedule.departureInterval = 2 ∧
    system.sanAntonioToAustinSchedule.departureOffset = 1 →
    countPassedBuses system = 2 :=
  sorry

end NUMINAMATH_CALUDE_san_antonio_bound_passes_two_austin_bound_l2286_228682


namespace NUMINAMATH_CALUDE_harolds_marbles_l2286_228679

/-- Given that Harold has 100 marbles, keeps 20 for himself, and shares the rest evenly among 5 friends,
    prove that each friend receives 16 marbles. -/
theorem harolds_marbles (total : ℕ) (kept : ℕ) (friends : ℕ) 
    (h1 : total = 100) 
    (h2 : kept = 20)
    (h3 : friends = 5) :
    (total - kept) / friends = 16 := by
  sorry

end NUMINAMATH_CALUDE_harolds_marbles_l2286_228679


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2286_228629

/-- Given an arithmetic sequence where the first term is 4/7 and the seventeenth term is 5/6,
    the ninth term is equal to 59/84. -/
theorem ninth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h1 : a 1 = 4/7)
  (h17 : a 17 = 5/6)
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  a 9 = 59/84 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2286_228629


namespace NUMINAMATH_CALUDE_particular_number_problem_l2286_228638

theorem particular_number_problem : ∃! x : ℚ, 2 * (67 - (x / 23)) = 102 := by sorry

end NUMINAMATH_CALUDE_particular_number_problem_l2286_228638


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l2286_228672

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l2286_228672


namespace NUMINAMATH_CALUDE_multiple_of_p_plus_q_l2286_228697

theorem multiple_of_p_plus_q (p q : ℚ) (h : p / q = 3 / 11) :
  ∃ m : ℤ, m * p + q = 17 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_p_plus_q_l2286_228697


namespace NUMINAMATH_CALUDE_square_mod_five_l2286_228600

theorem square_mod_five (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_five_l2286_228600


namespace NUMINAMATH_CALUDE_infinite_sum_equals_one_fourth_l2286_228656

/-- The sum of the series (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) from n=1 to infinity equals 1/4 -/
theorem infinite_sum_equals_one_fourth :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_one_fourth_l2286_228656


namespace NUMINAMATH_CALUDE_large_cube_surface_area_l2286_228675

theorem large_cube_surface_area 
  (num_small_cubes : ℕ) 
  (small_cube_edge : ℝ) 
  (large_cube_edge : ℝ) :
  num_small_cubes = 27 →
  small_cube_edge = 4 →
  large_cube_edge = small_cube_edge * (num_small_cubes ^ (1/3 : ℝ)) →
  6 * large_cube_edge^2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_surface_area_l2286_228675


namespace NUMINAMATH_CALUDE_eesha_travel_time_l2286_228664

/-- Eesha's usual time to reach her office -/
def usual_time : ℝ := 60

/-- The additional time taken when driving slower -/
def additional_time : ℝ := 20

/-- The ratio of slower speed to usual speed -/
def speed_ratio : ℝ := 0.75

theorem eesha_travel_time :
  usual_time = 60 ∧
  additional_time = usual_time / speed_ratio - usual_time :=
by sorry

end NUMINAMATH_CALUDE_eesha_travel_time_l2286_228664


namespace NUMINAMATH_CALUDE_triangle_area_l2286_228669

/-- The area of the triangle formed by two lines intersecting at (3,3) with slopes 1/3 and 3, 
    and a third line x + y = 12 -/
theorem triangle_area : ℝ := by
  -- Define the lines
  let line1 : ℝ → ℝ := fun x ↦ (1/3) * x + 2
  let line2 : ℝ → ℝ := fun x ↦ 3 * x - 6
  let line3 : ℝ → ℝ := fun x ↦ 12 - x

  -- Define the intersection points
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (4.5, 7.5)
  let C : ℝ × ℝ := (7.5, 4.5)

  -- Calculate the area of the triangle
  have area_formula : ℝ :=
    (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Assert that the area is equal to 9
  have area_eq_9 : area_formula = 9 := by sorry

  exact 9

end NUMINAMATH_CALUDE_triangle_area_l2286_228669


namespace NUMINAMATH_CALUDE_fiftieth_term_is_346_l2286_228601

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_is_346 : 
  arithmetic_sequence 3 7 50 = 346 := by
sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_346_l2286_228601


namespace NUMINAMATH_CALUDE_prob_draw_club_is_one_fourth_l2286_228623

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a deck -/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The probability of drawing a club from the top of a shuffled deck -/
def prob_draw_club (d : Deck) : ℚ :=
  cards_per_suit / total_cards

theorem prob_draw_club_is_one_fourth (d : Deck) :
  prob_draw_club d = 1 / 4 := by
  sorry

#check prob_draw_club_is_one_fourth

end NUMINAMATH_CALUDE_prob_draw_club_is_one_fourth_l2286_228623


namespace NUMINAMATH_CALUDE_garden_dimensions_l2286_228684

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

end NUMINAMATH_CALUDE_garden_dimensions_l2286_228684


namespace NUMINAMATH_CALUDE_book_pages_maximum_l2286_228661

theorem book_pages_maximum (pages : ℕ) : pages ≤ 208 :=
by
  have h1 : pages ≤ 13 * 16 := by sorry
  have h2 : pages ≤ 11 * 20 := by sorry
  sorry

#check book_pages_maximum

end NUMINAMATH_CALUDE_book_pages_maximum_l2286_228661


namespace NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l2286_228653

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l2286_228653


namespace NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_for_nonnegative_f_l2286_228652

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 0} = {x : ℝ | x ≤ -1/3} := by sorry

-- Part 2: Range of a for f(x) ≥ 0 when x ≥ -1
theorem range_of_a_for_nonnegative_f :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ -1 → f a x ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_for_nonnegative_f_l2286_228652


namespace NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l2286_228631

theorem base_conversion_256_to_base_5 :
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0) = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l2286_228631


namespace NUMINAMATH_CALUDE_is_cylinder_l2286_228649

/-- Represents the shape of a view in an orthographic projection --/
inductive ViewShape
  | Circle
  | Rectangle

/-- Represents the three orthographic views of a solid --/
structure OrthographicViews where
  top : ViewShape
  front : ViewShape
  side : ViewShape

/-- Represents different types of solids --/
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | Cuboid

/-- Given the three orthographic views of a solid, determine if it is a cylinder --/
theorem is_cylinder (views : OrthographicViews) :
  views.top = ViewShape.Circle ∧ 
  views.front = ViewShape.Rectangle ∧ 
  views.side = ViewShape.Rectangle → 
  ∃ (s : Solid), s = Solid.Cylinder :=
sorry

end NUMINAMATH_CALUDE_is_cylinder_l2286_228649


namespace NUMINAMATH_CALUDE_cars_without_features_l2286_228620

theorem cars_without_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : steering = 45)
  (h3 : windows = 25)
  (h4 : both = 17) :
  total - (steering + windows - both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_features_l2286_228620


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_not_all_intersecting_l2286_228673

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = -x
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x - 2

-- Theorem statement
theorem equation_represents_three_lines_not_all_intersecting :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (¬ (line1 p3.1 p3.2 ∧ line2 p3.1 p3.2 ∧ line3 p3.1 p3.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_three_lines_not_all_intersecting_l2286_228673


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l2286_228690

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, -2; 0, -1]) : 
  (B^3)⁻¹ = !![27, -24; 0, -1] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l2286_228690
