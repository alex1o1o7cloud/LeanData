import Mathlib

namespace NUMINAMATH_CALUDE_jenny_spent_fraction_l1814_181481

theorem jenny_spent_fraction (initial_amount : ℚ) : 
  (initial_amount / 2 = 21) →
  (initial_amount - 24 > 0) →
  ((initial_amount - 24) / initial_amount = 3/7) := by
  sorry

end NUMINAMATH_CALUDE_jenny_spent_fraction_l1814_181481


namespace NUMINAMATH_CALUDE_min_draws_for_twenty_l1814_181480

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 25, yellow := 22, blue := 15, white := 12, black := 10 }

/-- The theorem to be proved -/
theorem min_draws_for_twenty (counts : BallCounts) :
  counts = problemCounts → minDraws counts 20 = 95 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_twenty_l1814_181480


namespace NUMINAMATH_CALUDE_area_of_triangle_OBA_l1814_181495

/-- Given two points A and B in polar coordinates, prove that the area of triangle OBA is 6 --/
theorem area_of_triangle_OBA (A B : ℝ × ℝ) (h_A : A = (3, π/3)) (h_B : B = (4, π/6)) : 
  let O : ℝ × ℝ := (0, 0)
  let area := (1/2) * (A.1 * B.1) * Real.sin (B.2 - A.2)
  area = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_OBA_l1814_181495


namespace NUMINAMATH_CALUDE_valentines_count_l1814_181423

theorem valentines_count (initial : ℕ) (given_away : ℕ) (received : ℕ) : 
  initial = 60 → given_away = 16 → received = 5 → 
  initial - given_away + received = 49 := by
  sorry

end NUMINAMATH_CALUDE_valentines_count_l1814_181423


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1814_181494

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ m % 6 = 2 ∧ m % 8 = 3 → n ≤ m) ∧
  n > 0 ∧ n % 6 = 2 ∧ n % 8 = 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1814_181494


namespace NUMINAMATH_CALUDE_seashell_theorem_l1814_181403

def seashell_problem (mary_shells jessica_shells : ℕ) : Prop :=
  let kevin_shells := 3 * mary_shells
  let laura_shells := jessica_shells / 2
  mary_shells + jessica_shells + kevin_shells + laura_shells = 134

theorem seashell_theorem :
  seashell_problem 18 41 := by sorry

end NUMINAMATH_CALUDE_seashell_theorem_l1814_181403


namespace NUMINAMATH_CALUDE_sequence_2015th_term_l1814_181420

/-- Given a sequence {a_n} satisfying the conditions:
    1) a₁ = 1
    2) a₂ = 1/2
    3) 2/a_{n+1} = 1/a_n + 1/a_{n+2} for all n ∈ ℕ*
    Prove that a₂₀₁₅ = 1/2015 -/
theorem sequence_2015th_term (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → 2 / (a (n + 1)) = 1 / (a n) + 1 / (a (n + 2))) :
  a 2015 = 1 / 2015 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2015th_term_l1814_181420


namespace NUMINAMATH_CALUDE_pet_store_total_l1814_181462

/-- The number of dogs for sale in the pet store -/
def num_dogs : ℕ := 12

/-- The number of cats for sale in the pet store -/
def num_cats : ℕ := num_dogs / 3

/-- The number of birds for sale in the pet store -/
def num_birds : ℕ := 4 * num_dogs

/-- The number of fish for sale in the pet store -/
def num_fish : ℕ := 5 * num_dogs

/-- The number of reptiles for sale in the pet store -/
def num_reptiles : ℕ := 2 * num_dogs

/-- The number of rodents for sale in the pet store -/
def num_rodents : ℕ := num_dogs

/-- The total number of animals for sale in the pet store -/
def total_animals : ℕ := num_dogs + num_cats + num_birds + num_fish + num_reptiles + num_rodents

theorem pet_store_total : total_animals = 160 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_total_l1814_181462


namespace NUMINAMATH_CALUDE_joan_sold_26_books_l1814_181493

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Theorem: Joan sold 26 books in the yard sale -/
theorem joan_sold_26_books :
  books_sold 33 7 = 26 := by
  sorry

end NUMINAMATH_CALUDE_joan_sold_26_books_l1814_181493


namespace NUMINAMATH_CALUDE_star_equation_solutions_l1814_181407

-- Define the * operation
def star (a b : ℝ) : ℝ := a * (a + b) + b

-- Theorem statement
theorem star_equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  star a₁ 2.5 = 28.5 ∧ 
  star a₂ 2.5 = 28.5 ∧
  (a₁ = 4 ∨ a₁ = -13/2) ∧
  (a₂ = 4 ∨ a₂ = -13/2) :=
sorry

end NUMINAMATH_CALUDE_star_equation_solutions_l1814_181407


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l1814_181484

theorem greatest_integer_problem : 
  (∃ m : ℕ, 
    0 < m ∧ 
    m < 150 ∧ 
    (∃ a : ℤ, m = 10 * a - 2) ∧ 
    (∃ b : ℤ, m = 9 * b - 4) ∧
    (∀ n : ℕ, 
      (0 < n ∧ 
       n < 150 ∧ 
       (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
       (∃ b' : ℤ, n = 9 * b' - 4)) → 
      n ≤ m)) ∧
  (∀ m : ℕ, 
    (0 < m ∧ 
     m < 150 ∧ 
     (∃ a : ℤ, m = 10 * a - 2) ∧ 
     (∃ b : ℤ, m = 9 * b - 4) ∧
     (∀ n : ℕ, 
       (0 < n ∧ 
        n < 150 ∧ 
        (∃ a' : ℤ, n = 10 * a' - 2) ∧ 
        (∃ b' : ℤ, n = 9 * b' - 4)) → 
       n ≤ m)) → 
    m = 68) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l1814_181484


namespace NUMINAMATH_CALUDE_min_distance_sum_l1814_181474

theorem min_distance_sum (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1814_181474


namespace NUMINAMATH_CALUDE_product_of_fractions_l1814_181413

theorem product_of_fractions : 
  (10 : ℚ) / 6 * 4 / 20 * 20 / 12 * 16 / 32 * 40 / 24 * 8 / 40 * 60 / 36 * 32 / 64 = 25 / 324 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1814_181413


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1814_181472

theorem sqrt_sum_equality : Real.sqrt 1 + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1814_181472


namespace NUMINAMATH_CALUDE_stating_pacific_area_rounded_l1814_181433

/-- The area of the Pacific Ocean in square kilometers -/
def pacific_area : ℕ := 19996800

/-- Conversion factor from square kilometers to ten thousand square kilometers -/
def ten_thousand : ℕ := 10000

/-- Rounds a natural number to the nearest multiple of ten thousand -/
def round_to_nearest_ten_thousand (n : ℕ) : ℕ :=
  (n + 5000) / 10000 * 10000

/-- 
Theorem stating that the area of the Pacific Ocean, when converted to 
ten thousand square kilometers and rounded to the nearest ten thousand, 
is equal to 2000 ten thousand square kilometers
-/
theorem pacific_area_rounded : 
  round_to_nearest_ten_thousand (pacific_area / ten_thousand) = 2000 := by
  sorry


end NUMINAMATH_CALUDE_stating_pacific_area_rounded_l1814_181433


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l1814_181412

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to define any fields, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of corners in a regular tetrahedron -/
def num_corners (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: The sum of edges, corners, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_corners t + num_faces t = 14 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l1814_181412


namespace NUMINAMATH_CALUDE_star_equation_solution_l1814_181432

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a + 2 * b

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 (star 4 y) = -2 ∧ y = -11.5 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1814_181432


namespace NUMINAMATH_CALUDE_minimum_class_size_minimum_class_size_is_21_l1814_181450

theorem minimum_class_size : ℕ → Prop :=
  fun n =>
    ∃ (boys girls : ℕ),
      boys > 0 ∧ girls > 0 ∧
      (3 * boys = 4 * ((2 * girls) / 3)) ∧
      n = boys + girls + 4 ∧
      ∀ m, m < n →
        ¬∃ (b g : ℕ),
          b > 0 ∧ g > 0 ∧
          (3 * b = 4 * ((2 * g) / 3)) ∧
          m = b + g + 4

theorem minimum_class_size_is_21 :
  minimum_class_size 21 := by
  sorry

end NUMINAMATH_CALUDE_minimum_class_size_minimum_class_size_is_21_l1814_181450


namespace NUMINAMATH_CALUDE_book_sorting_terminates_and_sorts_width_l1814_181425

/-- Represents a book with height and width -/
structure Book where
  height : ℕ
  width : ℕ

/-- The state of the bookshelf -/
structure BookshelfState where
  books : List Book
  n : ℕ

/-- Predicate to check if books are sorted by increasing width -/
def sortedByWidth (state : BookshelfState) : Prop :=
  ∀ i j, i < j → i < state.n → j < state.n →
    (state.books.get ⟨i, by sorry⟩).width < (state.books.get ⟨j, by sorry⟩).width

/-- Predicate to check if a swap is valid -/
def canSwap (state : BookshelfState) (i : ℕ) : Prop :=
  i + 1 < state.n ∧
  (state.books.get ⟨i, by sorry⟩).width > (state.books.get ⟨i + 1, by sorry⟩).width ∧
  (state.books.get ⟨i, by sorry⟩).height < (state.books.get ⟨i + 1, by sorry⟩).height

/-- The main theorem -/
theorem book_sorting_terminates_and_sorts_width
  (initial : BookshelfState)
  (h_n : initial.n ≥ 2)
  (h_unique : ∀ i j, i ≠ j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height ≠ (initial.books.get ⟨j, by sorry⟩).height ∧
    (initial.books.get ⟨i, by sorry⟩).width ≠ (initial.books.get ⟨j, by sorry⟩).width)
  (h_initial_height : ∀ i j, i < j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height < (initial.books.get ⟨j, by sorry⟩).height) :
  ∃ (final : BookshelfState),
    (∀ i, ¬canSwap final i) ∧
    sortedByWidth final :=
by sorry

end NUMINAMATH_CALUDE_book_sorting_terminates_and_sorts_width_l1814_181425


namespace NUMINAMATH_CALUDE_max_value_implies_a_f_leq_g_implies_a_range_l1814_181477

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 * a * x + 1
def g (x : ℝ) : ℝ := x * (Real.exp x + 1)

-- Part (Ⅰ)
theorem max_value_implies_a (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = 0) →
  a = -1/2 :=
sorry

-- Part (Ⅱ)
theorem f_leq_g_implies_a_range (a : ℝ) :
  (∀ x > 0, f a x ≤ g x) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_f_leq_g_implies_a_range_l1814_181477


namespace NUMINAMATH_CALUDE_queue_probabilities_l1814_181436

/-- Probabilities of different numbers of people queuing -/
structure QueueProbabilities where
  p0 : ℝ  -- Probability of 0 people
  p1 : ℝ  -- Probability of 1 person
  p2 : ℝ  -- Probability of 2 people
  p3 : ℝ  -- Probability of 3 people
  p4 : ℝ  -- Probability of 4 people
  p5 : ℝ  -- Probability of 5 or more people
  sum_to_one : p0 + p1 + p2 + p3 + p4 + p5 = 1
  all_nonneg : 0 ≤ p0 ∧ 0 ≤ p1 ∧ 0 ≤ p2 ∧ 0 ≤ p3 ∧ 0 ≤ p4 ∧ 0 ≤ p5

/-- The probabilities for the specific scenario -/
def scenario : QueueProbabilities where
  p0 := 0.1
  p1 := 0.16
  p2 := 0.3
  p3 := 0.3
  p4 := 0.1
  p5 := 0.04
  sum_to_one := by sorry
  all_nonneg := by sorry

theorem queue_probabilities (q : QueueProbabilities) :
  (q.p0 + q.p1 + q.p2 = 0.56) ∧ 
  (q.p3 + q.p4 + q.p5 = 0.44) :=
by sorry

end NUMINAMATH_CALUDE_queue_probabilities_l1814_181436


namespace NUMINAMATH_CALUDE_inconsistent_fraction_problem_l1814_181404

theorem inconsistent_fraction_problem :
  ¬ ∃ (f : ℚ), (f * 4 = 8) ∧ ((1/8) * 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_fraction_problem_l1814_181404


namespace NUMINAMATH_CALUDE_vector_subtraction_l1814_181409

def a : Fin 3 → ℝ := ![(-5 : ℝ), 1, 3]
def b : Fin 3 → ℝ := ![(3 : ℝ), -1, 2]

theorem vector_subtraction :
  a - 2 • b = ![(-11 : ℝ), 3, -1] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1814_181409


namespace NUMINAMATH_CALUDE_jordan_stats_l1814_181478

/-- Represents the statistics of a basketball player in a game -/
structure BasketballStats where
  total_points : ℕ
  total_shots : ℕ
  total_hits : ℕ
  three_pointers : ℕ
  two_pointers : ℕ
  free_throws : ℕ

/-- Checks if the given basketball stats are valid -/
def valid_stats (stats : BasketballStats) : Prop :=
  stats.total_points = 3 * stats.three_pointers + 2 * stats.two_pointers + stats.free_throws ∧
  stats.total_hits = stats.three_pointers + stats.two_pointers + stats.free_throws ∧
  stats.total_shots ≥ stats.three_pointers + stats.two_pointers

/-- Theorem: Given Jordan's basketball stats, prove that he made 8 two-pointers and 3 free throws -/
theorem jordan_stats :
  ∀ (stats : BasketballStats),
    stats.total_points = 28 →
    stats.total_shots = 24 →
    stats.total_hits = 14 →
    stats.three_pointers = 3 →
    valid_stats stats →
    stats.two_pointers = 8 ∧ stats.free_throws = 3 := by
  sorry

end NUMINAMATH_CALUDE_jordan_stats_l1814_181478


namespace NUMINAMATH_CALUDE_zero_unique_for_multiplication_and_division_l1814_181417

theorem zero_unique_for_multiplication_and_division :
  ∀ x : ℝ, (∀ a : ℝ, x * a = x ∧ (a ≠ 0 → x / a = x)) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_unique_for_multiplication_and_division_l1814_181417


namespace NUMINAMATH_CALUDE_fox_invasion_count_l1814_181454

/-- The number of foxes that invaded the forest region --/
def num_foxes : ℕ := 3

/-- The initial number of rodents in the forest --/
def initial_rodents : ℕ := 150

/-- The number of rodents each fox catches per week --/
def rodents_per_fox_per_week : ℕ := 6

/-- The number of weeks the foxes hunted --/
def weeks : ℕ := 3

/-- The number of rodents remaining after the foxes hunted --/
def remaining_rodents : ℕ := 96

theorem fox_invasion_count :
  num_foxes * (rodents_per_fox_per_week * weeks) = initial_rodents - remaining_rodents :=
by sorry

end NUMINAMATH_CALUDE_fox_invasion_count_l1814_181454


namespace NUMINAMATH_CALUDE_discontinuous_when_limit_not_equal_value_l1814_181444

-- Define a multivariable function type
def MultivariableFunction (α : Type*) (β : Type*) := α → β

-- Define the concept of a limit for a multivariable function
def HasLimit (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - L) < ε

-- Define continuity for a multivariable function
def IsContinuousAt (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, HasLimit f x₀ L ∧ f x₀ = L

-- Theorem statement
theorem discontinuous_when_limit_not_equal_value
  (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) :
  HasLimit f x₀ L → f x₀ ≠ L → ¬(IsContinuousAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_discontinuous_when_limit_not_equal_value_l1814_181444


namespace NUMINAMATH_CALUDE_students_without_A_l1814_181455

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_history : history = 12)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l1814_181455


namespace NUMINAMATH_CALUDE_no_2000_digit_square_with_1999_fives_l1814_181410

theorem no_2000_digit_square_with_1999_fives : 
  ¬ ∃ n : ℕ, 
    (10^1999 ≤ n) ∧ (n < 10^2000) ∧  -- 2000-digit integer
    (∃ k : ℕ, n = k^2) ∧              -- perfect square
    (∃ d : ℕ, d < 10 ∧                -- at least 1999 digits of "5"
      (n / 10 = 5 * (10^1998 - 1) / 9 + d * 10^1998 ∨
       n % 10 ≠ 5 ∧ n / 10 = 5 * (10^1999 - 1) / 9)) :=
by sorry

end NUMINAMATH_CALUDE_no_2000_digit_square_with_1999_fives_l1814_181410


namespace NUMINAMATH_CALUDE_north_southland_population_increase_l1814_181419

/-- The number of hours between each birth in North Southland -/
def hours_between_births : ℕ := 6

/-- The number of deaths per day in North Southland -/
def deaths_per_day : ℕ := 2

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The population increase in North Southland per year -/
def population_increase : ℕ := 
  (24 / hours_between_births - deaths_per_day) * days_per_year

/-- The population increase in North Southland rounded to the nearest hundred -/
def rounded_population_increase : ℕ := 
  (population_increase + 50) / 100 * 100

theorem north_southland_population_increase :
  rounded_population_increase = 700 := by sorry

end NUMINAMATH_CALUDE_north_southland_population_increase_l1814_181419


namespace NUMINAMATH_CALUDE_line_properties_l1814_181426

def line_equation (x y : ℝ) : Prop := 3 * y = 4 * x - 9

theorem line_properties :
  (∃ m : ℝ, m = 4/3 ∧ ∀ x y : ℝ, line_equation x y → y = m * x + (-3)) ∧
  line_equation 3 1 :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1814_181426


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fifth_l1814_181439

theorem cos_eight_arccos_one_fifth :
  Real.cos (8 * Real.arccos (1/5)) = -15647/390625 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fifth_l1814_181439


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1814_181430

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (h1 : man_age = son_age + 26) (h2 : son_age = 24) :
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1814_181430


namespace NUMINAMATH_CALUDE_race_finish_order_l1814_181438

/-- Represents a sprinter in the race -/
inductive Sprinter
  | A
  | B
  | C

/-- Represents the order of sprinters -/
def RaceOrder := List Sprinter

/-- Represents the number of position changes for each sprinter -/
def PositionChanges := Sprinter → Nat

/-- Determines if a sprinter started later than another -/
def StartedLater (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter finished before another -/
def FinishedBefore (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter was delayed at the start -/
def DelayedAtStart (s : Sprinter) : Prop := sorry

theorem race_finish_order :
  ∀ (changes : PositionChanges),
    changes Sprinter.C = 6 →
    changes Sprinter.A = 5 →
    StartedLater Sprinter.B Sprinter.A →
    FinishedBefore Sprinter.B Sprinter.A →
    DelayedAtStart Sprinter.C →
    ∃ (order : RaceOrder),
      order = [Sprinter.B, Sprinter.A, Sprinter.C] :=
by sorry

end NUMINAMATH_CALUDE_race_finish_order_l1814_181438


namespace NUMINAMATH_CALUDE_vw_toyota_ratio_l1814_181489

/-- The number of Dodge trucks in the parking lot -/
def dodge_trucks : ℕ := 60

/-- The number of Volkswagen Bugs in the parking lot -/
def vw_bugs : ℕ := 5

/-- The number of Ford trucks in the parking lot -/
def ford_trucks : ℕ := dodge_trucks / 3

/-- The number of Toyota trucks in the parking lot -/
def toyota_trucks : ℕ := ford_trucks / 2

/-- The ratio of Volkswagen Bugs to Toyota trucks is 1:2 -/
theorem vw_toyota_ratio : 
  (vw_bugs : ℚ) / toyota_trucks = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_vw_toyota_ratio_l1814_181489


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l1814_181471

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = 1) : a + b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀ + b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l1814_181471


namespace NUMINAMATH_CALUDE_triangle_lines_l1814_181466

structure Triangle where
  B : ℝ × ℝ
  altitude_AB : ℝ → ℝ → ℝ
  angle_bisector_A : ℝ → ℝ → ℝ

def line_AB (t : Triangle) : ℝ → ℝ → ℝ := fun x y => 2*x + y - 8
def line_AC (t : Triangle) : ℝ → ℝ → ℝ := fun x y => x + 2*y + 2

theorem triangle_lines (t : Triangle) 
  (hB : t.B = (3, 2))
  (hAlt : t.altitude_AB = fun x y => x - 2*y + 2)
  (hBis : t.angle_bisector_A = fun x y => x + y - 2) :
  (line_AB t = fun x y => 2*x + y - 8) ∧ 
  (line_AC t = fun x y => x + 2*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_lines_l1814_181466


namespace NUMINAMATH_CALUDE_range_of_k_l1814_181402

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) ↔ k ∈ Set.Ioo 0 2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l1814_181402


namespace NUMINAMATH_CALUDE_hooper_bay_lobster_ratio_l1814_181499

/-- The ratio of lobster in Hooper Bay to other harbors -/
theorem hooper_bay_lobster_ratio :
  let total_lobster : ℕ := 480
  let other_harbors_lobster : ℕ := 80 + 80
  let hooper_bay_lobster : ℕ := total_lobster - other_harbors_lobster
  (hooper_bay_lobster : ℚ) / other_harbors_lobster = 2 := by
  sorry

end NUMINAMATH_CALUDE_hooper_bay_lobster_ratio_l1814_181499


namespace NUMINAMATH_CALUDE_two_sets_satisfying_union_condition_l1814_181485

theorem two_sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    S.card = 2 ∧ 
    ∀ M ∈ S, M ∪ {1} = {1, 2, 3} ∧
    ∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S :=
by sorry

end NUMINAMATH_CALUDE_two_sets_satisfying_union_condition_l1814_181485


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1814_181421

def set_of_numbers : List ℕ := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

theorem arithmetic_mean_of_special_set :
  let n := set_of_numbers.length
  let sum := set_of_numbers.sum
  sum / n = 98765432 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1814_181421


namespace NUMINAMATH_CALUDE_three_heads_after_three_tails_probability_l1814_181486

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of coin flips -/
def FlipSequence := List CoinFlip

/-- A fair coin has equal probability of heads and tails -/
def isFairCoin (p : CoinFlip → ℝ) : Prop :=
  p CoinFlip.Heads = 1/2 ∧ p CoinFlip.Tails = 1/2

/-- Checks if a sequence ends with three heads in a row -/
def endsWithThreeHeads : FlipSequence → Bool := sorry

/-- Checks if a sequence contains three tails before three heads -/
def hasThreeTailsBeforeThreeHeads : FlipSequence → Bool := sorry

/-- The probability of a specific flip sequence occurring -/
def sequenceProbability (s : FlipSequence) (p : CoinFlip → ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem three_heads_after_three_tails_probability 
  (p : CoinFlip → ℝ) (h : isFairCoin p) :
  (∃ s : FlipSequence, endsWithThreeHeads s ∧ hasThreeTailsBeforeThreeHeads s ∧
    sequenceProbability s p = 1/192) :=
by sorry

end NUMINAMATH_CALUDE_three_heads_after_three_tails_probability_l1814_181486


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1814_181442

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define the property of a_1 and a_2 being roots of the equation
def roots_property (a : ℕ → ℝ) : Prop :=
  (a 1)^2 - (a 3) * (a 1) + (a 4) = 0 ∧
  (a 2)^2 - (a 3) * (a 2) + (a 4) = 0

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → roots_property a → ∀ n : ℕ, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1814_181442


namespace NUMINAMATH_CALUDE_midpoint_chain_l1814_181487

/-- Given points X, Y, G, H, I, J on a line where:
  G is the midpoint of XY
  H is the midpoint of XG
  I is the midpoint of XH
  J is the midpoint of XI
  XJ = 4
  Prove that XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) 
  (hG : G = (X + Y) / 2)
  (hH : H = (X + G) / 2)
  (hI : I = (X + H) / 2)
  (hJ : J = (X + I) / 2)
  (hXJ : J - X = 4) : 
  Y - X = 64 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1814_181487


namespace NUMINAMATH_CALUDE_mopping_time_is_30_l1814_181458

def vacuum_time : ℕ := 45
def dusting_time : ℕ := 60
def cat_brushing_time : ℕ := 5
def num_cats : ℕ := 3
def total_free_time : ℕ := 3 * 60
def remaining_free_time : ℕ := 30

def total_cleaning_time : ℕ := total_free_time - remaining_free_time

def other_tasks_time : ℕ := vacuum_time + dusting_time + (cat_brushing_time * num_cats)

theorem mopping_time_is_30 : 
  total_cleaning_time - other_tasks_time = 30 := by sorry

end NUMINAMATH_CALUDE_mopping_time_is_30_l1814_181458


namespace NUMINAMATH_CALUDE_ellipse_equation_l1814_181422

/-- An ellipse with center at the origin, focus on the y-axis, eccentricity 1/2, and focal length 8 has the equation y²/64 + x²/48 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let focus_on_y_axis := true
  let eccentricity := (1 : ℝ) / 2
  let focal_length := (8 : ℝ)
  (y^2 / 64 + x^2 / 48 = 1) ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = focal_length / 2 ∧
      eccentricity = c / a ∧
      b^2 = a^2 - c^2 ∧
      y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1814_181422


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1814_181431

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - Real.sqrt 2 * i) / (Real.sqrt 2 + 1) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1814_181431


namespace NUMINAMATH_CALUDE_sakshi_tanya_efficiency_increase_l1814_181468

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Theorem stating the efficiency increase from Sakshi to Tanya -/
theorem sakshi_tanya_efficiency_increase :
  let sakshi_rate : ℚ := 1/5
  let tanya_rate : ℚ := 1/4
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_sakshi_tanya_efficiency_increase_l1814_181468


namespace NUMINAMATH_CALUDE_sports_equipment_purchasing_l1814_181456

/-- Represents the price and quantity information for sports equipment --/
structure EquipmentInfo where
  price_a : ℕ
  price_b : ℕ
  total_budget : ℕ
  total_units : ℕ

/-- Theorem about sports equipment purchasing --/
theorem sports_equipment_purchasing (info : EquipmentInfo) 
  (h1 : 3 * info.price_a + info.price_b = 500)
  (h2 : info.price_a + 2 * info.price_b = 250)
  (h3 : info.total_budget = 2700)
  (h4 : info.total_units = 25) :
  info.price_a = 150 ∧ 
  info.price_b = 50 ∧
  (∀ m : ℕ, m * info.price_a + (info.total_units - m) * info.price_b ≤ info.total_budget → m ≤ 14) ∧
  (∀ m : ℕ, 12 ≤ m → m ≤ 14 → m * info.price_a + (info.total_units - m) * info.price_b ≥ 2450) := by
  sorry

end NUMINAMATH_CALUDE_sports_equipment_purchasing_l1814_181456


namespace NUMINAMATH_CALUDE_sum_first_15_even_positive_l1814_181492

/-- The sum of the first n even positive integers -/
def sum_first_n_even_positive (n : ℕ) : ℕ :=
  n * (n + 1)

/-- Theorem: The sum of the first 15 even positive integers is 240 -/
theorem sum_first_15_even_positive :
  sum_first_n_even_positive 15 = 240 := by
  sorry

#eval sum_first_n_even_positive 15  -- This should output 240

end NUMINAMATH_CALUDE_sum_first_15_even_positive_l1814_181492


namespace NUMINAMATH_CALUDE_inverse_sqrt_problem_l1814_181408

-- Define the relationship between x and y
def inverse_sqrt_relation (x y : ℝ) (k : ℝ) : Prop :=
  y * Real.sqrt x = k

-- Define the theorem
theorem inverse_sqrt_problem (x y k : ℝ) :
  inverse_sqrt_relation x y k →
  inverse_sqrt_relation 2 4 k →
  y = 1 →
  x = 32 := by sorry

end NUMINAMATH_CALUDE_inverse_sqrt_problem_l1814_181408


namespace NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1814_181451

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the result of tossing two coins simultaneously -/
def TwoCoinsResult := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads),
                                         (CoinOutcome.Heads, CoinOutcome.Tails),
                                         (CoinOutcome.Tails, CoinOutcome.Heads),
                                         (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting at most 1 head -/
def atMostOneHead : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Tails),
                                           (CoinOutcome.Tails, CoinOutcome.Heads),
                                           (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting exactly 2 heads -/
def exactlyTwoHeads : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsResult) : Prop := A ∩ B = ∅

theorem atMostOneHead_exactlyTwoHeads_mutuallyExclusive :
  mutuallyExclusive atMostOneHead exactlyTwoHeads := by
  sorry

end NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1814_181451


namespace NUMINAMATH_CALUDE_sector_area_l1814_181415

/-- The area of a sector with radius 2 and perimeter equal to the circumference of its circle is 4π - 2 -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 2 → 
  2 * r + r * θ = 2 * π * r → 
  (1/2) * r^2 * θ = 4 * π - 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1814_181415


namespace NUMINAMATH_CALUDE_relay_team_orders_l1814_181476

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 5 team members -/
def team_size : ℕ := 5

/-- Lara always runs the last lap, so we need to arrange the other 4 runners -/
def runners_to_arrange : ℕ := team_size - 1

theorem relay_team_orders : permutations runners_to_arrange = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_orders_l1814_181476


namespace NUMINAMATH_CALUDE_max_xy_value_l1814_181453

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1814_181453


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1814_181406

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1814_181406


namespace NUMINAMATH_CALUDE_ryan_learning_days_l1814_181401

/-- Given that Ryan spends 4 hours on Chinese daily and a total of 24 hours on Chinese,
    prove that the number of days he learns is 6. -/
theorem ryan_learning_days :
  ∀ (hours_chinese_per_day : ℕ) (total_hours_chinese : ℕ),
    hours_chinese_per_day = 4 →
    total_hours_chinese = 24 →
    total_hours_chinese / hours_chinese_per_day = 6 :=
by sorry

end NUMINAMATH_CALUDE_ryan_learning_days_l1814_181401


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l1814_181470

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (2 * b) + b / (4 * c) + c / (8 * a)) ≥ (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l1814_181470


namespace NUMINAMATH_CALUDE_rational_sqrt5_zero_quadratic_roots_sum_difference_l1814_181482

theorem rational_sqrt5_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 := by
  sorry

theorem quadratic_roots_sum_difference (k : ℝ) :
  k ≠ 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧
    4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0 ∧
    x₁^2 + x₂^2 - 2 * x₁ * x₂ = 1/2) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt5_zero_quadratic_roots_sum_difference_l1814_181482


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l1814_181445

/-- Calculates the sample size for a Student Congress given the number of classes and students selected per class. -/
def sampleSize (numClasses : ℕ) (studentsPerClass : ℕ) : ℕ :=
  numClasses * studentsPerClass

/-- Theorem stating that for a school with 40 classes, where each class selects 3 students
    for the Student Congress, the sample size is 120 students. -/
theorem student_congress_sample_size :
  sampleSize 40 3 = 120 := by
  sorry

#eval sampleSize 40 3

end NUMINAMATH_CALUDE_student_congress_sample_size_l1814_181445


namespace NUMINAMATH_CALUDE_dad_catch_is_27_l1814_181463

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The total number of salmons caught by Hazel and her dad -/
def total_catch : ℕ := 51

/-- The number of salmons Hazel's dad caught -/
def dad_catch : ℕ := total_catch - hazel_catch

theorem dad_catch_is_27 : dad_catch = 27 := by
  sorry

end NUMINAMATH_CALUDE_dad_catch_is_27_l1814_181463


namespace NUMINAMATH_CALUDE_complex_fraction_real_l1814_181497

theorem complex_fraction_real (a : ℝ) : 
  ((-a + Complex.I) / (1 - Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l1814_181497


namespace NUMINAMATH_CALUDE_squared_inequality_condition_l1814_181440

theorem squared_inequality_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_squared_inequality_condition_l1814_181440


namespace NUMINAMATH_CALUDE_smallest_m_value_l1814_181441

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_m_value :
  ∃ (m x y : ℕ),
    m = x * y * (10 * x + y) ∧
    100 ≤ m ∧ m < 1000 ∧
    x < 10 ∧ y < 10 ∧
    x ≠ y ∧
    is_prime (10 * x + y) ∧
    is_prime (x + y) ∧
    (∀ (m' x' y' : ℕ),
      m' = x' * y' * (10 * x' + y') →
      100 ≤ m' ∧ m' < 1000 →
      x' < 10 ∧ y' < 10 →
      x' ≠ y' →
      is_prime (10 * x' + y') →
      is_prime (x' + y') →
      m ≤ m') ∧
    m = 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_value_l1814_181441


namespace NUMINAMATH_CALUDE_bugs_eat_seventeen_flowers_l1814_181411

/-- Represents the number of flowers eaten by each type of bug -/
structure BugEating where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Represents the number of bugs of each type -/
structure BugCount where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Calculates the total number of flowers eaten by all bugs -/
def totalFlowersEaten (eating : BugEating) (count : BugCount) : Nat :=
  eating.typeA * count.typeA + eating.typeB * count.typeB + eating.typeC * count.typeC

theorem bugs_eat_seventeen_flowers : 
  let eating : BugEating := { typeA := 2, typeB := 3, typeC := 5 }
  let count : BugCount := { typeA := 3, typeB := 2, typeC := 1 }
  totalFlowersEaten eating count = 17 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eat_seventeen_flowers_l1814_181411


namespace NUMINAMATH_CALUDE_inequalities_hold_l1814_181443

theorem inequalities_hold (m n l : ℝ) (h1 : m > n) (h2 : n > l) : 
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1814_181443


namespace NUMINAMATH_CALUDE_inequality_equiv_range_l1814_181449

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- The theorem stating the equivalence between the inequality and the range of x -/
theorem inequality_equiv_range :
  ∀ x : ℝ, (f (1 - x) + f (2 * x) > 2) ↔ x > -1 :=
sorry

end NUMINAMATH_CALUDE_inequality_equiv_range_l1814_181449


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1814_181491

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def target_number : ℕ := 101000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.01
    exponent := 5
    coeff_range := by sorry }

theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = target_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1814_181491


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l1814_181498

/-- Represents a scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the results of a math competition -/
structure CompetitionResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_questions : Nat
  new_score : Int
  old_score : Int

def new_system : ScoringSystem := ⟨6, 0, 3, 0⟩
def old_system : ScoringSystem := ⟨5, -2, 0, 20⟩

/-- Calculates the score based on a given scoring system and competition result -/
def calculate_score (system : ScoringSystem) (result : CompetitionResult) : Int :=
  system.initial + 
  system.correct * result.correct + 
  system.wrong * result.wrong + 
  system.unanswered * result.unanswered

theorem alice_unanswered_questions 
  (result : CompetitionResult)
  (h1 : result.new_score = 105)
  (h2 : result.old_score = 75)
  (h3 : result.total_questions = 30)
  (h4 : calculate_score new_system result = result.new_score)
  (h5 : calculate_score old_system result = result.old_score)
  (h6 : result.correct + result.wrong + result.unanswered = result.total_questions) :
  result.unanswered = 5 := by
  sorry

#check alice_unanswered_questions

end NUMINAMATH_CALUDE_alice_unanswered_questions_l1814_181498


namespace NUMINAMATH_CALUDE_y_divisibility_l1814_181459

def y : ℕ := 58 + 104 + 142 + 184 + 304 + 368 + 3304

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 4 * k) ∧
  ¬(∀ k : ℕ, y = 8 * k) ∧
  ¬(∀ k : ℕ, y = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l1814_181459


namespace NUMINAMATH_CALUDE_triangle_area_heron_l1814_181488

/-- Given a triangle ABC with sides a, b, c and area S, prove that under certain conditions, 
    the area S calculated using Heron's formula is equal to 15√7/4 -/
theorem triangle_area_heron (a b c : ℝ) (S : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 * Real.sin C = 24 * Real.sin A →
  a * (Real.sin C - Real.sin B) * (c + b) = (27 - a^2) * Real.sin A →
  S = Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2) / 2)^2)) →
  S = 15 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_heron_l1814_181488


namespace NUMINAMATH_CALUDE_average_equals_seven_implies_x_equals_twelve_l1814_181428

theorem average_equals_seven_implies_x_equals_twelve :
  let numbers : List ℝ := [1, 2, 4, 5, 6, 9, 9, 10, 12, x]
  (List.sum numbers) / (List.length numbers) = 7 →
  x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_average_equals_seven_implies_x_equals_twelve_l1814_181428


namespace NUMINAMATH_CALUDE_set_A_determination_l1814_181483

def U : Set ℕ := {0, 1, 2, 4}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {1, 2}) : A = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_A_determination_l1814_181483


namespace NUMINAMATH_CALUDE_subtracted_amount_l1814_181429

theorem subtracted_amount (chosen_number : ℕ) (subtracted_amount : ℕ) : 
  chosen_number = 208 → 
  (chosen_number / 2 : ℚ) - subtracted_amount = 4 → 
  subtracted_amount = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1814_181429


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l1814_181405

/-- Given that f(x) = x^3 - ax is monotonically increasing on [1, +∞), 
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → (x₁^3 - a*x₁) < (x₂^3 - a*x₂)) →
  a ≤ 3 ∧ ∀ b > a, ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ (x₁^3 - b*x₁) ≥ (x₂^3 - b*x₂) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l1814_181405


namespace NUMINAMATH_CALUDE_circle_properties_l1814_181469

-- Define the circle C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y - 1)^2 = 9}

-- Define the line intercepting the chord
def L : Set (ℝ × ℝ) := {(x, y) | 12*x - 5*y - 8 = 0}

-- Define a general line through the origin
def l (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x}

-- Define point Q
def Q : ℝ × ℝ := (1, 2)

theorem circle_properties :
  -- Part 1: Length of the chord
  ∃ (A B : ℝ × ℝ), A ∈ C ∩ L ∧ B ∈ C ∩ L ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 ∧
  -- Part 2: Sum of reciprocals of y-coordinates is constant
  ∀ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 → A ∈ C ∩ l k → B ∈ C ∩ l k → A ≠ B →
    1 / A.2 + 1 / B.2 = -1/4 ∧
  -- Part 3: Slope of line l when sum of squared distances is 22
  ∃ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 ∧ A ∈ C ∩ l k ∧ B ∈ C ∩ l k ∧ A ≠ B ∧
    (A.1 - Q.1)^2 + (A.2 - Q.2)^2 + (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 22 ∧ k = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1814_181469


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1814_181475

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1814_181475


namespace NUMINAMATH_CALUDE_ten_consecutive_composites_l1814_181427

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem ten_consecutive_composites :
  ∃ (start : ℕ), 
    start + 9 < 500 ∧
    (∀ i : ℕ, i ∈ Finset.range 10 → isComposite (start + i)) ∧
    start + 9 = 489 := by
  sorry

end NUMINAMATH_CALUDE_ten_consecutive_composites_l1814_181427


namespace NUMINAMATH_CALUDE_box_probability_l1814_181460

theorem box_probability (a : ℕ) (h1 : a > 0) : 
  (4 : ℝ) / a = (1 : ℝ) / 5 → a = 20 := by
sorry

end NUMINAMATH_CALUDE_box_probability_l1814_181460


namespace NUMINAMATH_CALUDE_student_tickets_sold_l1814_181479

theorem student_tickets_sold (total_tickets : ℕ) (student_price non_student_price total_amount : ℚ) :
  total_tickets = 193 →
  student_price = 1/2 →
  non_student_price = 3/2 →
  total_amount = 825/4 →
  ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_tickets * student_price + non_student_tickets * non_student_price = total_amount ∧
    student_tickets = 83 := by
  sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l1814_181479


namespace NUMINAMATH_CALUDE_z3_magnitude_range_l1814_181447

/-- Given complex numbers satisfying certain conditions, prove the range of the magnitude of z₃ -/
theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 2)
  (h2 : Complex.abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : Complex.abs (z₁ + z₂ - z₃) = 2) :
  0 ≤ Complex.abs z₃ ∧ Complex.abs z₃ ≤ 4 := by sorry

end NUMINAMATH_CALUDE_z3_magnitude_range_l1814_181447


namespace NUMINAMATH_CALUDE_arrangement_count_four_objects_five_positions_l1814_181461

theorem arrangement_count : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * arrangement_count n

theorem four_objects_five_positions :
  arrangement_count 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_four_objects_five_positions_l1814_181461


namespace NUMINAMATH_CALUDE_car_journey_speed_l1814_181457

def car_journey (v : ℝ) : Prop :=
  let first_part_time : ℝ := 1
  let first_part_speed : ℝ := 40
  let second_part_time : ℝ := 0.5
  let third_part_time : ℝ := 2
  let total_time : ℝ := first_part_time + second_part_time + third_part_time
  let total_distance : ℝ := first_part_speed * first_part_time + v * (second_part_time + third_part_time)
  let average_speed : ℝ := 54.285714285714285
  total_distance / total_time = average_speed

theorem car_journey_speed : car_journey 60 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_speed_l1814_181457


namespace NUMINAMATH_CALUDE_parabola_properties_l1814_181490

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 22

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (3, 5)

-- Define a point that the parabola passes through
def point : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through the given point
  parabola_equation point.1 = point.2 ∧
  -- The vertex of the parabola is at (3, 5)
  (∀ x, parabola_equation x ≤ parabola_equation vertex.1) ∧
  -- The axis of symmetry is vertical (x = 3)
  (∀ x, parabola_equation (2 * vertex.1 - x) = parabola_equation x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1814_181490


namespace NUMINAMATH_CALUDE_range_of_f_l1814_181416

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 5

-- Define the domain of x
def domain : Set ℝ := { x | -3 ≤ x ∧ x < 2 }

-- State the theorem
theorem range_of_f :
  ∃ (y_min y_max : ℝ), y_min = -7 ∧ y_max = 11 ∧
  ∀ y, (∃ x ∈ domain, f x = y) ↔ y_min ≤ y ∧ y < y_max :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1814_181416


namespace NUMINAMATH_CALUDE_symmetric_point_example_l1814_181424

/-- Given a line ax + by + c = 0 and two points P and Q, this function checks if Q is symmetric to P with respect to the line. -/
def is_symmetric_point (a b c : ℝ) (px py qx qy : ℝ) : Prop :=
  let mx := (px + qx) / 2
  let my := (py + qy) / 2
  (a * mx + b * my + c = 0) ∧ (a * (qx - px) + b * (qy - py) = 0)

/-- The point (3, 2) is symmetric to (-1, -2) with respect to the line x + y = 1 -/
theorem symmetric_point_example : is_symmetric_point 1 1 (-1) (-1) (-2) 3 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l1814_181424


namespace NUMINAMATH_CALUDE_olivia_earnings_l1814_181473

def hourly_wage : ℕ := 9
def monday_hours : ℕ := 4
def wednesday_hours : ℕ := 3
def friday_hours : ℕ := 6

theorem olivia_earnings : 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_earnings_l1814_181473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1814_181496

/-- Given a sequence of real numbers satisfying the condition
    |a_m + a_n - a_(m+n)| ≤ 1 / (m + n) for all m and n,
    prove that the sequence is arithmetic with a_k = k * a_1 for all k. -/
theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
    (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
  ∀ k : ℕ, a k = k * a 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1814_181496


namespace NUMINAMATH_CALUDE_g_6_eq_1_l1814_181434

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the conditions on f
axiom f_1 : f 1 = 1
axiom f_add_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_add_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1 - x

-- State the theorem to be proved
theorem g_6_eq_1 : g 6 = 1 := by sorry

end NUMINAMATH_CALUDE_g_6_eq_1_l1814_181434


namespace NUMINAMATH_CALUDE_seed_cost_calculation_l1814_181400

/-- Given that 2 pounds of seed cost $44.68, prove that 6 pounds of seed will cost $134.04. -/
theorem seed_cost_calculation (cost_for_two_pounds : ℝ) (pounds_needed : ℝ) : 
  cost_for_two_pounds = 44.68 → pounds_needed = 6 → 
  (pounds_needed / 2) * cost_for_two_pounds = 134.04 := by
sorry

end NUMINAMATH_CALUDE_seed_cost_calculation_l1814_181400


namespace NUMINAMATH_CALUDE_max_value_abc_l1814_181467

theorem max_value_abc (a b c : ℝ) (h : a^2 + b^2/4 + c^2/9 = 1) :
  a + b + c ≤ Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l1814_181467


namespace NUMINAMATH_CALUDE_inequality_properties_l1814_181464

theorem inequality_properties (a b c : ℝ) : 
  (a^2 > b^2 → abs a > abs b) ∧ 
  (a > b ↔ a + c > b + c) := by sorry

end NUMINAMATH_CALUDE_inequality_properties_l1814_181464


namespace NUMINAMATH_CALUDE_journey_time_approx_24_hours_l1814_181448

/-- Represents a segment of the journey --/
structure Segment where
  distance : Float
  speed : Float
  stay : Float

/-- Calculates the time taken for a segment --/
def segmentTime (s : Segment) : Float :=
  s.distance / s.speed + s.stay

/-- Represents Manex's journey --/
def manexJourney : List Segment := [
  { distance := 70, speed := 60, stay := 1 },
  { distance := 50, speed := 35, stay := 3 },
  { distance := 20, speed := 60, stay := 0 },
  { distance := 20, speed := 30, stay := 2 },
  { distance := 30, speed := 40, stay := 0 },
  { distance := 60, speed := 70, stay := 2.5 },
  { distance := 60, speed := 35, stay := 0.75 }
]

/-- Calculates the total outbound distance --/
def outboundDistance : Float :=
  (manexJourney.map (·.distance)).sum

/-- Represents the return journey --/
def returnJourney : Segment :=
  { distance := outboundDistance + 100, speed := 55, stay := 0 }

/-- Calculates the total journey time --/
def totalJourneyTime : Float :=
  (manexJourney.map segmentTime).sum + segmentTime returnJourney

/-- Theorem stating that the total journey time is approximately 24 hours --/
theorem journey_time_approx_24_hours :
  (totalJourneyTime).round = 24 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_approx_24_hours_l1814_181448


namespace NUMINAMATH_CALUDE_expected_final_set_size_l1814_181465

/-- The set of elements Marisa is working with -/
def S : Finset Nat := Finset.range 8

/-- The initial number of subsets in Marisa's collection -/
def initial_subsets : Nat := 2^8 - 1

/-- The number of steps in Marisa's process -/
def num_steps : Nat := 2^8 - 2

/-- The probability of an element being in a randomly chosen subset -/
def prob_in_subset : ℚ := 128 / 255

/-- The expected size of the final set in Marisa's subset collection process -/
theorem expected_final_set_size :
  (S.card : ℚ) * prob_in_subset = 1024 / 255 := by sorry

end NUMINAMATH_CALUDE_expected_final_set_size_l1814_181465


namespace NUMINAMATH_CALUDE_triangle_property_l1814_181437

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = (3/5)*c, then tan(A)/tan(B) = 4 and max(tan(A-B)) = 3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = (3/5) * c) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (Real.tan A / Real.tan B = 4) ∧
  (∀ x y : ℝ, Real.tan (A - B) ≤ (3/4)) ∧
  (∃ x y : ℝ, Real.tan (A - B) = (3/4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1814_181437


namespace NUMINAMATH_CALUDE_james_comics_count_l1814_181414

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The number of years James writes comics -/
def numYears : ℕ := 4

/-- The frequency of James writing comics (every other day) -/
def comicFrequency : ℕ := 2

/-- The total number of comics James writes in 4 non-leap years -/
def totalComics : ℕ := (daysInYear * numYears) / comicFrequency

theorem james_comics_count : totalComics = 730 := by
  sorry

end NUMINAMATH_CALUDE_james_comics_count_l1814_181414


namespace NUMINAMATH_CALUDE_fraction_simplification_l1814_181452

theorem fraction_simplification :
  let x := 5 / (1 + (32 * (Real.cos (15 * π / 180))^4 - 10 - 8 * Real.sqrt 3)^(1/3))
  x = 1 - 4^(1/3) + 16^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1814_181452


namespace NUMINAMATH_CALUDE_prism_21_edges_l1814_181435

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := sorry

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := sorry

/-- Theorem: A prism with 21 edges has 9 faces and 7 vertices -/
theorem prism_21_edges (p : Prism) (h : p.edges = 21) : 
  num_faces p = 9 ∧ num_vertices p = 7 := by sorry

end NUMINAMATH_CALUDE_prism_21_edges_l1814_181435


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1814_181418

theorem inequality_implies_range (a : ℝ) : (1^2 * a + 2 * 1 + 1 < 0) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1814_181418


namespace NUMINAMATH_CALUDE_roy_daily_sports_hours_l1814_181446

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of hours Roy spends on sports in a week when he misses 2 days -/
def sports_hours_with_missed_days : ℕ := 6

/-- The number of days Roy misses in a week -/
def missed_days : ℕ := 2

/-- The number of hours Roy spends on sports activities in school every day -/
def daily_sports_hours : ℚ := 2

theorem roy_daily_sports_hours :
  daily_sports_hours = sports_hours_with_missed_days / (school_days_per_week - missed_days) :=
by sorry

end NUMINAMATH_CALUDE_roy_daily_sports_hours_l1814_181446
