import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_difference_l3034_303473

theorem chocolate_difference (friend1 friend2 friend3 : ℚ)
  (h1 : friend1 = 5/6)
  (h2 : friend2 = 2/3)
  (h3 : friend3 = 7/9) :
  max friend1 (max friend2 friend3) - min friend1 (min friend2 friend3) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_difference_l3034_303473


namespace NUMINAMATH_CALUDE_triangle_problem_l3034_303493

open Real

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  sin A / a = sin B / b ∧ sin A / a = sin C / c ∧  -- Law of sines
  sqrt 3 * c * cos A - a * cos C + b - 2 * c = 0 →
  A = π / 3 ∧ 
  sqrt 3 / 2 < cos B + cos C ∧ cos B + cos C ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3034_303493


namespace NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l3034_303468

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddysCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Thursday --/
def thursdayPurchase (cards : BuddysCards) : ℕ :=
  cards.thursday - cards.wednesday

/-- The theorem stating the ratio of Thursday's purchase to Tuesday's amount --/
theorem thursday_to_tuesday_ratio (cards : BuddysCards) :
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.wednesday = cards.tuesday + 12 →
  cards.thursday = 32 →
  thursdayPurchase cards * 3 = cards.tuesday := by
  sorry

end NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l3034_303468


namespace NUMINAMATH_CALUDE_fraction_value_l3034_303481

theorem fraction_value (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3034_303481


namespace NUMINAMATH_CALUDE_jake_bitcoin_theorem_l3034_303471

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_bitcoin_theorem_l3034_303471


namespace NUMINAMATH_CALUDE_problem_solution_l3034_303444

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 7)
  (h_eq2 : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3034_303444


namespace NUMINAMATH_CALUDE_chord_length_l3034_303414

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus at 135°
def line (x y : ℝ) : Prop := y = -x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ ‖A - B‖ = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3034_303414


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l3034_303453

/-- Given a restaurant menu with vegan and allergen constraints, 
    prove the fraction of edible dishes for a vegan with allergies -/
theorem restaurant_menu_fraction (total_vegan : ℕ) (vegan_fraction : ℚ) (allergen_vegan : ℕ) :
  total_vegan = 6 →
  vegan_fraction = 1/6 →
  allergen_vegan = 4 →
  (total_vegan - allergen_vegan : ℚ) / (total_vegan / vegan_fraction) = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l3034_303453


namespace NUMINAMATH_CALUDE_circle_symmetry_l3034_303411

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 5

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 5

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g (-x) (-y)

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_origin original_circle symmetric_circle :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3034_303411


namespace NUMINAMATH_CALUDE_probability_neither_orange_nor_white_l3034_303412

theorem probability_neither_orange_nor_white (orange black white : ℕ) 
  (h_orange : orange = 8) (h_black : black = 7) (h_white : white = 6) :
  (black : ℚ) / (orange + black + white) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_neither_orange_nor_white_l3034_303412


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3034_303479

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 20*x - 8

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 44 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3034_303479


namespace NUMINAMATH_CALUDE_complex_magnitude_l3034_303441

theorem complex_magnitude (z : ℂ) : z = -2 - I → Complex.abs (z + I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3034_303441


namespace NUMINAMATH_CALUDE_min_value_expression_l3034_303409

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  3 * a^2 + 1 / (a * (a - b)) + 1 / (a * b) - 6 * a * c + 9 * c^2 ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3034_303409


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3034_303451

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ z : ℂ, z = (m^2 - 4 : ℝ) + (m + 2 : ℝ) * I ∧ z.re = 0 ∧ m + 2 ≠ 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3034_303451


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3034_303462

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℚ, (3 * x₁^2 - 5 * x₁ - 2 = 0 ∧ x₁ = 2) ∧
                (3 * x₂^2 - 5 * x₂ - 2 = 0 ∧ x₂ = -1/3)) ∧
  (∃ y₁ y₂ : ℚ, (3 * y₁ * (y₁ - 1) = 2 - 2 * y₁ ∧ y₁ = 1) ∧
                (3 * y₂ * (y₂ - 1) = 2 - 2 * y₂ ∧ y₂ = -2/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3034_303462


namespace NUMINAMATH_CALUDE_supermarket_turnover_equation_l3034_303480

/-- Represents the equation for a supermarket's quarterly turnover with monthly growth rate -/
theorem supermarket_turnover_equation (x : ℝ) : 
  200 * (1 + (1 + x) + (1 + x)^2) = 1000 ↔ 
  (2 * (1 + x + (1 + x)^2) = 10 ∧ 
   2 > 0 ∧ 
   10 > 0 ∧ 
   (∀ m : ℕ, m < 3 → (1 + x)^m > 0)) := by
  sorry

end NUMINAMATH_CALUDE_supermarket_turnover_equation_l3034_303480


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3034_303459

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  muffin_price > 0 →
  banana_price > 0 →
  4 * muffin_price + 3 * banana_price > 0 →
  2 * (4 * muffin_price + 3 * banana_price) = 2 * muffin_price + 16 * banana_price →
  muffin_price / banana_price = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3034_303459


namespace NUMINAMATH_CALUDE_angelina_walk_speeds_l3034_303431

theorem angelina_walk_speeds (v : ℝ) :
  v > 0 ∧
  960 / v - 40 = 480 / (2 * v) ∧
  480 / (2 * v) - 20 = 720 / (3 * v) →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by sorry

end NUMINAMATH_CALUDE_angelina_walk_speeds_l3034_303431


namespace NUMINAMATH_CALUDE_shortest_distance_to_origin_l3034_303495

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := sorry

-- Define a point P on the right branch of the hyperbola
def point_P : ℝ × ℝ := sorry

-- Define point A satisfying the orthogonality condition
def point_A : ℝ × ℝ := sorry

-- State the theorem
theorem shortest_distance_to_origin :
  ∀ (A : ℝ × ℝ),
    (∃ (P : ℝ × ℝ), hyperbola P.1 P.2 ∧ 
      ((A.1 - P.1) * (A.1 - left_focus.1) + (A.2 - P.2) * (A.2 - left_focus.2) = 0)) →
    (∃ (d : ℝ), d = Real.sqrt 3 ∧ 
      ∀ (B : ℝ × ℝ), Real.sqrt (B.1^2 + B.2^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_origin_l3034_303495


namespace NUMINAMATH_CALUDE_not_sum_of_consecutive_iff_power_of_two_l3034_303486

/-- A natural number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- A natural number can be expressed as the sum of consecutive natural numbers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start : ℕ) (length : ℕ+), n = (length : ℕ) * (2 * start + length - 1) / 2

/-- 
Theorem: A natural number cannot be expressed as the sum of consecutive natural numbers 
if and only if it is a power of 2
-/
theorem not_sum_of_consecutive_iff_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n := by sorry

end NUMINAMATH_CALUDE_not_sum_of_consecutive_iff_power_of_two_l3034_303486


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3034_303498

theorem quadratic_solution_property :
  ∀ p q : ℝ,
  (5 * p^2 - 20 * p + 15 = 0) →
  (5 * q^2 - 20 * q + 15 = 0) →
  p ≠ q →
  (p * q - 3)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3034_303498


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3034_303446

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧ 
  (∃ a b : ℝ, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3034_303446


namespace NUMINAMATH_CALUDE_second_agency_daily_charge_proof_l3034_303415

/-- The daily charge of the first agency in dollars -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency in dollars -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency in dollars -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25.0

/-- The daily charge of the second agency in dollars -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_proof :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_second_agency_daily_charge_proof_l3034_303415


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3034_303449

theorem quadratic_factorization (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3034_303449


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l3034_303488

/-- Given a point P (-2, -3), prove that (2, 3) is its symmetric point about the origin -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-2, -3)
  let Q : ℝ × ℝ := (2, 3)
  (∀ (x y : ℝ), (x, y) = P → (-x, -y) = Q) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l3034_303488


namespace NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l3034_303450

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l3034_303450


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3034_303418

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3034_303418


namespace NUMINAMATH_CALUDE_average_weight_solution_l3034_303430

def average_weight_problem (a b c : ℝ) : Prop :=
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  b = 31

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c → (a + b + c) / 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l3034_303430


namespace NUMINAMATH_CALUDE_participation_schemes_count_l3034_303467

/-- The number of students to choose from -/
def totalStudents : Nat := 4

/-- The number of students to be selected -/
def selectedStudents : Nat := 3

/-- The number of subjects -/
def subjects : Nat := 3

/-- Represents that student A must participate -/
def studentAMustParticipate : Prop := True

/-- The total number of different participation schemes -/
def participationSchemes : Nat := 18

theorem participation_schemes_count :
  studentAMustParticipate →
  participationSchemes = (Nat.choose (totalStudents - 1) (selectedStudents - 1)) * (Nat.factorial selectedStudents) :=
by sorry

end NUMINAMATH_CALUDE_participation_schemes_count_l3034_303467


namespace NUMINAMATH_CALUDE_paige_mp3_songs_l3034_303419

theorem paige_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) : 
  initial = 11 → deleted = 9 → added = 8 → initial - deleted + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_paige_mp3_songs_l3034_303419


namespace NUMINAMATH_CALUDE_temperature_at_14_minutes_l3034_303478

/-- Represents the temperature change over time -/
structure TemperatureChange where
  initialTemp : ℝ
  rate : ℝ

/-- Calculates the temperature at a given time -/
def temperature (tc : TemperatureChange) (t : ℝ) : ℝ :=
  tc.initialTemp + tc.rate * t

/-- Theorem: The temperature at 14 minutes is 52°C given the conditions -/
theorem temperature_at_14_minutes (tc : TemperatureChange) 
    (h1 : tc.initialTemp = 10)
    (h2 : tc.rate = 3) : 
    temperature tc 14 = 52 := by
  sorry

#eval temperature { initialTemp := 10, rate := 3 } 14

end NUMINAMATH_CALUDE_temperature_at_14_minutes_l3034_303478


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3034_303405

theorem rational_equation_solution (x : ℝ) : 
  x ≠ 3 → x ≠ -3 → (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3034_303405


namespace NUMINAMATH_CALUDE_binomial_expansion_x_squared_term_l3034_303464

theorem binomial_expansion_x_squared_term (x : ℝ) (n : ℕ) :
  (∃ r : ℕ, r ≤ n ∧ (n.choose r) * x^((5*r)/2 - n) = x^2) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_x_squared_term_l3034_303464


namespace NUMINAMATH_CALUDE_square_of_sum_l3034_303442

theorem square_of_sum (x y : ℝ) : (x + 2*y)^2 = x^2 + 4*x*y + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l3034_303442


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3034_303433

theorem trigonometric_equality : 
  (2 * Real.sin (47 * π / 180) - Real.sqrt 3 * Real.sin (17 * π / 180)) / Real.cos (17 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3034_303433


namespace NUMINAMATH_CALUDE_exists_number_with_reversed_digits_and_middle_zero_l3034_303474

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  d : ℕ
  e : ℕ
  f : ℕ
  d_lt_base : d < base
  e_lt_base : e < base
  f_lt_base : f < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.d * base^2 + n.e * base + n.f

theorem exists_number_with_reversed_digits_and_middle_zero :
  ∃ (n : ThreeDigitNumber 6) (m : ThreeDigitNumber 8),
    to_nat n = to_nat m ∧
    n.d = m.f ∧
    n.e = 0 ∧
    n.e = m.e ∧
    n.f = m.d :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_reversed_digits_and_middle_zero_l3034_303474


namespace NUMINAMATH_CALUDE_b_age_is_eighteen_l3034_303485

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 47
    Prove that b is 18 years old. -/
theorem b_age_is_eighteen (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 47) : 
  b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_eighteen_l3034_303485


namespace NUMINAMATH_CALUDE_precious_stone_cost_l3034_303408

theorem precious_stone_cost (num_stones : ℕ) (total_amount : ℕ) (h1 : num_stones = 8) (h2 : total_amount = 14280) :
  total_amount / num_stones = 1785 := by
sorry

end NUMINAMATH_CALUDE_precious_stone_cost_l3034_303408


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l3034_303448

/-- Given 72 Lemon Heads distributed equally among 6 friends, prove that each friend receives 12 Lemon Heads. -/
theorem lemon_heads_distribution (total : ℕ) (friends : ℕ) (each : ℕ) 
  (h1 : total = 72) 
  (h2 : friends = 6) 
  (h3 : total = friends * each) : 
  each = 12 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l3034_303448


namespace NUMINAMATH_CALUDE_book_distribution_l3034_303435

theorem book_distribution (x : ℕ) (total_books : ℕ) : 
  (9 * x + 7 ≤ total_books) ∧ (total_books < 11 * x) →
  (9 * x + 7 = total_books) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l3034_303435


namespace NUMINAMATH_CALUDE_norma_cards_l3034_303413

theorem norma_cards (initial_cards : ℕ) (lost_fraction : ℚ) (remaining_cards : ℕ) : 
  initial_cards = 88 → 
  lost_fraction = 3/4 → 
  remaining_cards = initial_cards - (initial_cards * lost_fraction).floor → 
  remaining_cards = 22 := by
sorry

end NUMINAMATH_CALUDE_norma_cards_l3034_303413


namespace NUMINAMATH_CALUDE_sophias_book_length_l3034_303456

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 →
  total_pages = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l3034_303456


namespace NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l3034_303426

theorem infinite_solutions_and_sum_of_exceptions :
  let A : ℚ := 3
  let B : ℚ := 5
  let C : ℚ := 40/3
  let f (x : ℚ) := (x + B) * (A * x + 40) / ((x + C) * (x + 5))
  (∀ x, x ≠ -C → x ≠ -5 → f x = 3) ∧
  (-5 + (-C) = -55/3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l3034_303426


namespace NUMINAMATH_CALUDE_swimmers_passing_count_l3034_303423

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnTime : ℝ

/-- Calculates the number of times two swimmers pass each other --/
def calculatePassings (poolLength : ℝ) (totalTime : ℝ) (swimmerA : Swimmer) (swimmerB : Swimmer) : ℕ :=
  sorry

/-- The main theorem stating the number of times the swimmers pass each other --/
theorem swimmers_passing_count :
  let poolLength : ℝ := 120
  let totalTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmerA : Swimmer := { speed := 4, turnTime := 0 }
  let swimmerB : Swimmer := { speed := 3, turnTime := 2 }
  calculatePassings poolLength totalTime swimmerA swimmerB = 51 :=
by sorry

end NUMINAMATH_CALUDE_swimmers_passing_count_l3034_303423


namespace NUMINAMATH_CALUDE_seed_distribution_l3034_303425

theorem seed_distribution (n : ℕ) : 
  (n * (n + 1) / 2 : ℚ) + 100 = n * (3 * n + 1) / 2 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_seed_distribution_l3034_303425


namespace NUMINAMATH_CALUDE_ball_bounce_height_l3034_303417

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h_target : ℝ) (k : ℕ) 
  (h_initial : h₀ = 1500)
  (h_ratio : r = 2/3)
  (h_target_def : h_target = 2) :
  (∀ n : ℕ, n < k → h₀ * r^n ≥ h_target) ∧ 
  (h₀ * r^k < h_target) ↔ 
  k = 19 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l3034_303417


namespace NUMINAMATH_CALUDE_chord_line_equation_l3034_303439

/-- Given a circle with equation x^2 + y^2 = 10 and a chord with midpoint P(1, 1),
    the equation of the line containing this chord is x + y - 2 = 0 -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 + y^2 = 10) →
  (∃ (t : ℝ), x = 1 + t ∧ y = 1 - t) →
  (x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l3034_303439


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3034_303458

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' + y' + 5 = 0 → Real.sqrt (x'^2 + y'^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3034_303458


namespace NUMINAMATH_CALUDE_parabola_focus_property_l3034_303438

/-- Parabola with equation y^2 = 16x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 16 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (4, 0)

/-- Point on y-axis with |OA| = |OF| -/
def A : ℝ × ℝ := (0, 4) -- We choose the positive y-coordinate

/-- Intersection of directrix and x-axis -/
def B : ℝ × ℝ := (-4, 0)

/-- Vector from F to A -/
def FA : ℝ × ℝ := (A.1 - F.1, A.2 - F.2)

/-- Vector from A to B -/
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_focus_property :
  F ∈ Parabola ∧
  A.1 = 0 ∧
  (A.1 - 0)^2 + (A.2 - 0)^2 = (F.1 - 0)^2 + (F.2 - 0)^2 ∧
  B.2 = 0 →
  dot_product FA AB = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_property_l3034_303438


namespace NUMINAMATH_CALUDE_prob_at_least_seven_stay_value_l3034_303470

def num_friends : ℕ := 8
def num_unsure : ℕ := 5
def num_certain : ℕ := 3
def prob_unsure_stay : ℚ := 3/7

def prob_at_least_seven_stay : ℚ :=
  Nat.choose num_unsure 3 * (prob_unsure_stay ^ 3) * ((1 - prob_unsure_stay) ^ 2) +
  prob_unsure_stay ^ num_unsure

theorem prob_at_least_seven_stay_value :
  prob_at_least_seven_stay = 4563/16807 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_seven_stay_value_l3034_303470


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3034_303416

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (average : ℕ) 
  (h1 : english = 96) 
  (h2 : mathematics = 95) 
  (h3 : physics = 82) 
  (h4 : chemistry = 97) 
  (h5 : average = 93) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 95 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3034_303416


namespace NUMINAMATH_CALUDE_length_ae_is_10_l3034_303400

/-- A quadrilateral with the properties of an isosceles trapezoid and a rectangle -/
structure QuadrilateralABCDE where
  /-- AB is a side of the quadrilateral -/
  ab : ℝ
  /-- EC is a side of the quadrilateral -/
  ec : ℝ
  /-- ABCE is an isosceles trapezoid -/
  abce_isosceles_trapezoid : Bool
  /-- ACDE is a rectangle -/
  acde_rectangle : Bool

/-- The length of AE in the quadrilateral ABCDE -/
def length_ae (q : QuadrilateralABCDE) : ℝ :=
  sorry

/-- Theorem stating that the length of AE is 10 under given conditions -/
theorem length_ae_is_10 (q : QuadrilateralABCDE) 
  (h1 : q.ab = 10) 
  (h2 : q.ec = 20) 
  (h3 : q.abce_isosceles_trapezoid = true) 
  (h4 : q.acde_rectangle = true) : 
  length_ae q = 10 :=
sorry

end NUMINAMATH_CALUDE_length_ae_is_10_l3034_303400


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3034_303452

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3034_303452


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3034_303496

/-- Given that the inequality ax^2 + x + 1 < 0 has a non-empty solution set for x,
    prove that the range of a is a < 1/4 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + x + 1 < 0) → a < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3034_303496


namespace NUMINAMATH_CALUDE_decrease_six_l3034_303434

def temperature_change : ℝ → ℝ := id

axiom positive_rise (x : ℝ) : x > 0 → temperature_change x > 0

axiom rise_three : temperature_change 3 = 3

theorem decrease_six : temperature_change (-6) = -6 := by sorry

end NUMINAMATH_CALUDE_decrease_six_l3034_303434


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l3034_303445

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l3034_303445


namespace NUMINAMATH_CALUDE_expression_evaluation_l3034_303455

theorem expression_evaluation : (π - 2023)^0 + |1 - Real.sqrt 3| + Real.sqrt 8 - Real.tan (π / 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3034_303455


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3034_303482

-- Define the point type
structure Point := (x : ℝ) (y : ℝ)

-- Define the problem statement
theorem point_B_coordinates (A B : Point) (h1 : A.x = 1 ∧ A.y = -1) 
  (h2 : (B.x - A.x)^2 + (B.y - A.y)^2 = 3^2) 
  (h3 : B.x = A.x) : 
  (B = Point.mk 1 (-4) ∨ B = Point.mk 1 2) := by
  sorry


end NUMINAMATH_CALUDE_point_B_coordinates_l3034_303482


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3034_303401

/-- Represents the total number of staff in each category -/
structure StaffCount where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (staff : StaffCount) (managementSample : ℕ) : ℕ :=
  let totalStaff := staff.business + staff.management + staff.logistics
  let samplingFraction := managementSample / staff.management
  totalStaff * samplingFraction

/-- Theorem: Given the staff counts and management sample, the total sample size is 20 -/
theorem stratified_sample_size 
  (staff : StaffCount) 
  (h1 : staff.business = 120) 
  (h2 : staff.management = 24) 
  (h3 : staff.logistics = 16) 
  (h4 : calculateSampleSize staff 3 = 20) : 
  calculateSampleSize staff 3 = 20 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3034_303401


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_roots_in_unit_interval_l3034_303432

theorem min_a_for_quadratic_roots_in_unit_interval :
  ∀ (a b c : ℤ) (α β : ℝ),
    a > 0 →
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = α ∨ x = β) →
    0 < α →
    α < β →
    β < 1 →
    a ≥ 5 ∧ ∃ (a₀ b₀ c₀ : ℤ) (α₀ β₀ : ℝ),
      a₀ = 5 ∧
      a₀ > 0 ∧
      (∀ x : ℝ, a₀ * x^2 + b₀ * x + c₀ = 0 ↔ x = α₀ ∨ x = β₀) ∧
      0 < α₀ ∧
      α₀ < β₀ ∧
      β₀ < 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_roots_in_unit_interval_l3034_303432


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3034_303436

/-- Given a sphere with surface area 16π cm², prove its volume is 32π/3 cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 16 * π → (4/3) * π * r^3 = (32 * π)/3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3034_303436


namespace NUMINAMATH_CALUDE_king_ducats_distribution_l3034_303469

theorem king_ducats_distribution (n : ℕ) (total_ducats : ℕ) :
  (∃ (a : ℕ),
    -- The eldest son receives 'a' ducats in the first round
    a + n = 21 ∧
    -- Total ducats in the first round
    n * a - (n - 1) * n / 2 +
    -- Total ducats in the second round
    n * (n + 1) / 2 = total_ducats) →
  n = 7 ∧ total_ducats = 105 := by
sorry

end NUMINAMATH_CALUDE_king_ducats_distribution_l3034_303469


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3034_303461

theorem age_ratio_problem (amy jeremy chris : ℕ) : 
  amy + jeremy + chris = 132 →
  amy = jeremy / 3 →
  jeremy = 66 →
  ∃ k : ℕ, chris = k * amy →
  chris / amy = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3034_303461


namespace NUMINAMATH_CALUDE_election_votes_calculation_l3034_303454

theorem election_votes_calculation (total_votes : ℕ) : 
  (85 : ℚ) / 100 * ((85 : ℚ) / 100 * total_votes) = 404600 →
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l3034_303454


namespace NUMINAMATH_CALUDE_rams_weight_increase_l3034_303420

theorem rams_weight_increase (ram_weight shyam_weight : ℝ) : 
  ram_weight / shyam_weight = 6 / 5 →
  ∃ (ram_increase : ℝ),
    ram_weight * (1 + ram_increase) + shyam_weight * 1.21 = 82.8 ∧
    (ram_weight * (1 + ram_increase) + shyam_weight * 1.21) / (ram_weight + shyam_weight) = 1.15 →
    ram_increase = 1.48 := by
  sorry

end NUMINAMATH_CALUDE_rams_weight_increase_l3034_303420


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3034_303427

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3034_303427


namespace NUMINAMATH_CALUDE_james_is_25_l3034_303477

-- Define the ages as natural numbers
def james_age : ℕ := sorry
def john_age : ℕ := sorry
def tim_age : ℕ := 79

-- Define the relationships between ages
axiom age_difference : john_age = james_age + 12
axiom tim_age_relation : tim_age = 2 * john_age - 5

-- Theorem to prove
theorem james_is_25 : james_age = 25 := by sorry

end NUMINAMATH_CALUDE_james_is_25_l3034_303477


namespace NUMINAMATH_CALUDE_train_true_speed_l3034_303428

/-- The true speed of a train given its length, crossing time, and opposing wind speed -/
theorem train_true_speed (train_length : ℝ) (crossing_time : ℝ) (wind_speed : ℝ) :
  train_length = 200 →
  crossing_time = 20 →
  wind_speed = 5 →
  (train_length / crossing_time) + wind_speed = 15 := by
  sorry


end NUMINAMATH_CALUDE_train_true_speed_l3034_303428


namespace NUMINAMATH_CALUDE_Y_subset_X_l3034_303403

-- Define set X
def X : Set ℕ := {n : ℕ | ∃ m : ℕ, (3^n + 4^n) / 5 = m}

-- Define set Y
def Y : Set ℕ := {t : ℕ | ∃ k : ℕ, t = (2*k - 1)^2 + 1}

-- Theorem statement
theorem Y_subset_X : Y ⊆ X := by sorry

end NUMINAMATH_CALUDE_Y_subset_X_l3034_303403


namespace NUMINAMATH_CALUDE_ice_cream_theorem_ice_cream_distribution_count_l3034_303463

def ice_cream_distribution (n : ℕ) : ℕ :=
  (Nat.choose (n + 2) 2)

theorem ice_cream_theorem :
  ice_cream_distribution 62 = 2016 :=
by sorry

/-- Given:
    - 62 trainees choose from 5 ice cream flavors
    - Bubblegum flavor (r) at least as popular as tabasco (t)
    - Number of students choosing cactus flavor (a) is a multiple of 6
    - At most 5 students chose lemon basil flavor (b)
    - At most 1 student chose foie gras flavor (c)
    Prove: The number of possible distributions is 2016 -/
theorem ice_cream_distribution_count :
  ∃ (r t a b c : ℕ),
    r + t + a + b + c = 62 ∧
    r ≥ t ∧
    a % 6 = 0 ∧
    b ≤ 5 ∧
    c ≤ 1 ∧
    ice_cream_distribution 62 = 2016 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_ice_cream_distribution_count_l3034_303463


namespace NUMINAMATH_CALUDE_exponent_base_proof_l3034_303447

theorem exponent_base_proof (m : ℤ) (x : ℝ) : 
  ((-2)^(2*m) = x^(12-m)) → (m = 4) → (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_exponent_base_proof_l3034_303447


namespace NUMINAMATH_CALUDE_optimal_solution_is_valid_and_minimal_l3034_303440

/-- Represents a nail in the painting hanging problem -/
inductive Nail
| a₁ : Nail
| a₂ : Nail
| a₃ : Nail
| a₄ : Nail

/-- Represents a sequence of nails and their inverses -/
inductive NailSequence
| empty : NailSequence
| cons : Nail → NailSequence → NailSequence
| inv : Nail → NailSequence → NailSequence

/-- Counts the number of symbols in a nail sequence -/
def symbolCount : NailSequence → Nat
| NailSequence.empty => 0
| NailSequence.cons _ s => 1 + symbolCount s
| NailSequence.inv _ s => 1 + symbolCount s

/-- Checks if a nail sequence falls when a given nail is removed -/
def fallsWhenRemoved (s : NailSequence) (n : Nail) : Prop := sorry

/-- Represents the optimal solution [[a₁, a₂], [a₃, a₄]] -/
def optimalSolution : NailSequence := sorry

/-- Theorem: The optimal solution is valid and minimal -/
theorem optimal_solution_is_valid_and_minimal :
  (∀ n : Nail, fallsWhenRemoved optimalSolution n) ∧
  (∀ s : NailSequence, (∀ n : Nail, fallsWhenRemoved s n) → symbolCount optimalSolution ≤ symbolCount s) := by
  sorry

end NUMINAMATH_CALUDE_optimal_solution_is_valid_and_minimal_l3034_303440


namespace NUMINAMATH_CALUDE_sum_distances_constant_l3034_303429

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the triangle -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- A point inside an equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) where
  /-- The distance from the point to the first side -/
  dist1 : ℝ
  /-- The distance from the point to the second side -/
  dist2 : ℝ
  /-- The distance from the point to the third side -/
  dist3 : ℝ
  /-- Assumption that all distances are non-negative -/
  dist_nonneg : dist1 ≥ 0 ∧ dist2 ≥ 0 ∧ dist3 ≥ 0
  /-- Assumption that the point is inside the triangle -/
  inside : dist1 + dist2 + dist3 < t.side_length * Real.sqrt 3 / 2

/-- The theorem stating that the sum of distances is constant -/
theorem sum_distances_constant (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.dist1 + p.dist2 + p.dist3 = t.side_length * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_distances_constant_l3034_303429


namespace NUMINAMATH_CALUDE_square_area_error_l3034_303406

def error_in_area (excess_error : Real) (deficit_error : Real) : Real :=
  let correct_factor := (1 + excess_error) * (1 - deficit_error)
  (1 - correct_factor) * 100

theorem square_area_error :
  error_in_area 0.03 0.04 = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l3034_303406


namespace NUMINAMATH_CALUDE_set_a_range_l3034_303489

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | 6 * x + a > 0}
  1 ∉ A → a ∈ Set.Iic (-6) :=
by sorry

end NUMINAMATH_CALUDE_set_a_range_l3034_303489


namespace NUMINAMATH_CALUDE_mans_rate_l3034_303466

def with_stream : ℝ := 25
def against_stream : ℝ := 13

theorem mans_rate (with_stream against_stream : ℝ) :
  with_stream = 25 →
  against_stream = 13 →
  (with_stream + against_stream) / 2 = 19 := by
sorry

end NUMINAMATH_CALUDE_mans_rate_l3034_303466


namespace NUMINAMATH_CALUDE_campers_hiking_morning_l3034_303494

theorem campers_hiking_morning (morning_rowers afternoon_rowers total_rowers : ℕ)
  (h1 : morning_rowers = 13)
  (h2 : afternoon_rowers = 21)
  (h3 : total_rowers = 34)
  (h4 : morning_rowers + afternoon_rowers = total_rowers) :
  total_rowers - (morning_rowers + afternoon_rowers) = 0 :=
by sorry

end NUMINAMATH_CALUDE_campers_hiking_morning_l3034_303494


namespace NUMINAMATH_CALUDE_kamal_age_problem_l3034_303487

/-- Kamal's age problem -/
theorem kamal_age_problem (k s : ℕ) : 
  k - 12 = 7 * (s - 12) →  -- 12 years ago, Kamal was 7 times as old as his son
  k + 6 = 3 * (s + 6) →    -- In 6 years, Kamal will be thrice as old as his son
  k = 75                   -- Kamal's present age is 75
  := by sorry

end NUMINAMATH_CALUDE_kamal_age_problem_l3034_303487


namespace NUMINAMATH_CALUDE_twelve_people_round_table_l3034_303465

/-- The number of distinct seating arrangements for n people around a round table,
    where rotations are considered the same. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct seating arrangements for 12 people around a round table,
    where rotations are considered the same, is equal to 11!. -/
theorem twelve_people_round_table : roundTableArrangements 12 = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_twelve_people_round_table_l3034_303465


namespace NUMINAMATH_CALUDE_x_gets_thirty_paisa_l3034_303402

/-- Represents the share of each person in rupees -/
structure Share where
  w : ℝ
  x : ℝ
  y : ℝ

/-- The total amount distributed -/
def total_amount : ℝ := 15

/-- The share of w in rupees -/
def w_share : ℝ := 10

/-- The amount y gets for each rupee w gets, in rupees -/
def y_per_w : ℝ := 0.20

/-- Theorem stating that x gets 0.30 rupees for each rupee w gets -/
theorem x_gets_thirty_paisa (s : Share) 
  (h1 : s.w = w_share)
  (h2 : s.y = y_per_w * s.w)
  (h3 : s.w + s.x + s.y = total_amount) : 
  s.x / s.w = 0.30 := by sorry

end NUMINAMATH_CALUDE_x_gets_thirty_paisa_l3034_303402


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3034_303483

theorem max_value_quadratic :
  (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ 111/4) ∧
  (∃ x : ℝ, -3 * x^2 + 15 * x + 9 = 111/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3034_303483


namespace NUMINAMATH_CALUDE_range_of_a_l3034_303472

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 * x else x^2 + 5 * x + 2

/-- Function g(x) defined as f(x) - 2x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * x

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    (∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z)) →
  a ∈ Set.Icc (-1) 2 ∧ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3034_303472


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l3034_303497

-- Define a triangle in Euclidean space
def Triangle : Type := ℝ × ℝ × ℝ

-- Define the function that calculates the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := sorry

-- Theorem stating that the sum of interior angles of any triangle is 180°
theorem sum_of_interior_angles_is_180 (t : Triangle) :
  sum_of_interior_angles t = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l3034_303497


namespace NUMINAMATH_CALUDE_intersection_A_B_l3034_303407

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3034_303407


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3034_303492

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 288 →
  profit_percentage = 0.20 →
  selling_price = (1 + profit_percentage) * (selling_price / (1 + profit_percentage)) →
  selling_price / (1 + profit_percentage) = 240 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3034_303492


namespace NUMINAMATH_CALUDE_square_root_equation_l3034_303490

theorem square_root_equation (n : ℕ+) :
  Real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / ((n + 1) : ℝ)^2) = 1 + 1 / ((n : ℝ) * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3034_303490


namespace NUMINAMATH_CALUDE_women_stockbrokers_increase_l3034_303424

/-- Calculates the final number of women stockbrokers after a percentage increase -/
def final_number (initial : ℕ) (percent_increase : ℕ) : ℕ :=
  initial + (initial * percent_increase) / 100

/-- Theorem: Given 10,000 initial women stockbrokers and a 100% increase, 
    the final number is 20,000 -/
theorem women_stockbrokers_increase : 
  final_number 10000 100 = 20000 := by sorry

end NUMINAMATH_CALUDE_women_stockbrokers_increase_l3034_303424


namespace NUMINAMATH_CALUDE_nurses_count_l3034_303491

theorem nurses_count (total_staff : ℕ) (doctor_ratio nurse_ratio : ℕ) : 
  total_staff = 200 → 
  doctor_ratio = 4 → 
  nurse_ratio = 6 → 
  (nurse_ratio : ℚ) / (doctor_ratio + nurse_ratio : ℚ) * total_staff = 120 := by
sorry

end NUMINAMATH_CALUDE_nurses_count_l3034_303491


namespace NUMINAMATH_CALUDE_factorization_proof_l3034_303404

theorem factorization_proof (a b c : ℝ) :
  4 * a * b + 2 * a * c = 2 * a * (2 * b + c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3034_303404


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3034_303443

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the second inequality
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x + 1) + c - 3 * a * x

theorem quadratic_inequality_solution 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : solution_set a b c = Set.Ioo (-2) 1) :
  {x : ℝ | g a b c x < 0} = Set.Iic 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3034_303443


namespace NUMINAMATH_CALUDE_range_of_difference_l3034_303484

theorem range_of_difference (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = x^2 - 2*x) →
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_difference_l3034_303484


namespace NUMINAMATH_CALUDE_zach_rental_cost_l3034_303422

/-- Calculates the total cost of renting a car given the base cost, per-mile cost, and miles driven -/
def total_rental_cost (base_cost : ℝ) (per_mile_cost : ℝ) (miles_monday : ℝ) (miles_thursday : ℝ) : ℝ :=
  base_cost + per_mile_cost * (miles_monday + miles_thursday)

/-- Theorem: Given the rental conditions, Zach's total cost is $832 -/
theorem zach_rental_cost :
  total_rental_cost 150 0.5 620 744 = 832 := by
  sorry

#eval total_rental_cost 150 0.5 620 744

end NUMINAMATH_CALUDE_zach_rental_cost_l3034_303422


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3034_303475

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  asymptotes_tangent_to_parabola : Bool

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x - 1

-- Define the eccentricity of a hyperbola
def eccentricity (h : Hyperbola) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (C : Hyperbola) :
  C.center = (0, 0) →
  C.foci_on_x_axis = true →
  C.asymptotes_tangent_to_parabola = true →
  eccentricity C = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3034_303475


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3034_303457

theorem sum_of_roots_quadratic (x : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 6
  let sum_of_roots := -b / a
  2 * x^2 - 8 * x + 6 = 0 → sum_of_roots = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3034_303457


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3034_303421

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3034_303421


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3034_303476

theorem polygon_interior_angles_sum (n : ℕ) (h : n > 2) :
  (360 / 36 : ℝ) = n →
  (n - 2) * 180 = 1440 := by
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3034_303476


namespace NUMINAMATH_CALUDE_custom_mult_example_l3034_303410

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := a * b + b ^ 2

/-- Theorem stating that 4 * (-2) = -4 using the custom multiplication -/
theorem custom_mult_example : custom_mult 4 (-2) = -4 := by sorry

end NUMINAMATH_CALUDE_custom_mult_example_l3034_303410


namespace NUMINAMATH_CALUDE_min_value_condition_inequality_condition_l3034_303499

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x + m|

-- Theorem for part 1
theorem min_value_condition (m : ℝ) :
  (∃ (x : ℝ), f x m = 2 ∧ ∀ (y : ℝ), f y m ≥ 2) ↔ (m = 3 ∨ m = -1) :=
sorry

-- Theorem for part 2
theorem inequality_condition (m : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x m ≤ 2 * x + 3) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_condition_inequality_condition_l3034_303499


namespace NUMINAMATH_CALUDE_one_third_between_one_fourth_one_sixth_l3034_303460

/-- The fraction one-third of the way from a to b -/
def one_third_between (a b : ℚ) : ℚ := (2 * a + b) / 3

/-- Prove that the fraction one-third of the way from 1/4 to 1/6 is equal to 2/9 -/
theorem one_third_between_one_fourth_one_sixth :
  one_third_between (1/4) (1/6) = 2/9 := by sorry

end NUMINAMATH_CALUDE_one_third_between_one_fourth_one_sixth_l3034_303460


namespace NUMINAMATH_CALUDE_tinas_money_left_l3034_303437

/-- Calculates the amount of money Tina has left after saving and spending --/
theorem tinas_money_left (june_savings july_savings august_savings : ℕ) 
  (book_expense shoe_expense : ℕ) : 
  june_savings = 27 →
  july_savings = 14 →
  august_savings = 21 →
  book_expense = 5 →
  shoe_expense = 17 →
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense) = 40 := by
sorry


end NUMINAMATH_CALUDE_tinas_money_left_l3034_303437
