import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4_451

theorem complex_modulus_problem (z : ℂ) : 
  z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4_451


namespace NUMINAMATH_CALUDE_finance_club_probability_l4_484

theorem finance_club_probability (total_students : ℕ) (interested_fraction : ℚ) 
  (h1 : total_students = 20)
  (h2 : interested_fraction = 3/4) :
  let interested_students := (interested_fraction * total_students).num
  let not_interested_students := total_students - interested_students
  1 - (not_interested_students / total_students) * ((not_interested_students - 1) / (total_students - 1)) = 18/19 := by
sorry

end NUMINAMATH_CALUDE_finance_club_probability_l4_484


namespace NUMINAMATH_CALUDE_quadratic_positivity_quadratic_positivity_range_l4_471

/-- Given a quadratic function f(x) = x^2 + 2x + a, if f(x) > 0 for all x ≥ 1,
    then a > -3. -/
theorem quadratic_positivity (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0) → a > -3 := by
  sorry

/-- The range of a for which f(x) = x^2 + 2x + a is positive for all x ≥ 1
    is the open interval (-3, +∞). -/
theorem quadratic_positivity_range :
  {a : ℝ | ∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0} = Set.Ioi (-3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_positivity_quadratic_positivity_range_l4_471


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l4_493

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ := current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l4_493


namespace NUMINAMATH_CALUDE_vector_problem_l4_496

/-- Given four points in a plane, prove that if certain vector conditions are met,
    then the coordinates of point D and the value of k are as specified. -/
theorem vector_problem (A B C D : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (2, -2) →
  C = (4, -1) →
  B - A = D - C →
  ∃ (t : ℝ), t • (k • (B - A) - (C - B)) = (B - A) + 3 • (C - B) →
  D = (5, -6) ∧ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l4_496


namespace NUMINAMATH_CALUDE_parabola_intersection_l4_450

/-- Proves that (-3, 55) and (4, -8) are the only intersection points of the parabolas
    y = 3x^2 - 12x - 8 and y = 2x^2 - 10x + 4 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 8
  let g (x : ℝ) := 2 * x^2 - 10 * x + 4
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -3 ∧ y = 55) ∨ (x = 4 ∧ y = -8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l4_450


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l4_456

def f (x : ℝ) : ℝ := x - x^3 - 1

theorem tangent_line_perpendicular (a : ℝ) : 
  (∃ k : ℝ, k = (deriv f) 1 ∧ k * (-4/a) = -1) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l4_456


namespace NUMINAMATH_CALUDE_markup_is_twenty_percent_l4_464

/-- Calculates the markup percentage given cost price, discount, and profit percentage. -/
def markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage) - discount
  let markup := selling_price - cost_price
  (markup / cost_price) * 100

/-- Theorem stating that under the given conditions, the markup percentage is 20%. -/
theorem markup_is_twenty_percent :
  markup_percentage 180 50 (20/100) = 20 := by
sorry

end NUMINAMATH_CALUDE_markup_is_twenty_percent_l4_464


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l4_479

/-- The number of players in a chess tournament -/
def num_players : ℕ := 19

/-- The total number of games played in the tournament -/
def total_games : ℕ := 342

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  2 * num_players * (num_players - 1) = total_games := by
  sorry

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l4_479


namespace NUMINAMATH_CALUDE_darius_age_is_8_l4_433

-- Define the ages of Jenna and Darius
def jenna_age : ℕ := 13
def darius_age : ℕ := 21 - jenna_age

-- Theorem statement
theorem darius_age_is_8 :
  (jenna_age > darius_age) ∧ 
  (jenna_age + darius_age = 21) ∧
  (jenna_age = 13) →
  darius_age = 8 := by
sorry

end NUMINAMATH_CALUDE_darius_age_is_8_l4_433


namespace NUMINAMATH_CALUDE_power_of_two_digit_sum_five_l4_474

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem power_of_two_digit_sum_five (n : ℕ) : 
  sum_of_digits (2^n) = 5 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_power_of_two_digit_sum_five_l4_474


namespace NUMINAMATH_CALUDE_abs_less_of_even_increasing_fn_l4_462

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- State the theorem
theorem abs_less_of_even_increasing_fn (a b : ℝ) 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_nonneg f) 
  (h_less : f a < f b) : 
  |a| < |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_less_of_even_increasing_fn_l4_462


namespace NUMINAMATH_CALUDE_negation_equivalence_l4_439

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4_439


namespace NUMINAMATH_CALUDE_equation_solution_l4_475

theorem equation_solution (x y : ℤ) (hy : y ≠ 0) :
  (2 : ℝ) ^ ((x - y : ℝ) / y) - (3 / 2 : ℝ) * y = 1 ↔
  ∃ n : ℕ, x = ((2 * n + 1) * (2 ^ (2 * n + 1) - 2)) / 3 ∧
           y = (2 ^ (2 * n + 1) - 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4_475


namespace NUMINAMATH_CALUDE_fishing_problem_l4_494

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus relative to Patrick -/
def angus_more_than_patrick : ℕ := 4

/-- The number of fish Ollie caught fewer than Angus -/
def ollie_fewer_than_angus : ℕ := 7

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

theorem fishing_problem :
  ollie_fish + ollie_fewer_than_angus - angus_more_than_patrick = patrick_fish := by
  sorry

end NUMINAMATH_CALUDE_fishing_problem_l4_494


namespace NUMINAMATH_CALUDE_eight_solutions_for_triple_f_l4_468

def f (x : ℝ) : ℝ := |1 - 2*x|

theorem eight_solutions_for_triple_f (x : ℝ) :
  x ∈ Set.Icc 0 1 →
  ∃! (solutions : Finset ℝ),
    (∀ s ∈ solutions, f (f (f s)) = (1/2) * s) ∧
    Finset.card solutions = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_solutions_for_triple_f_l4_468


namespace NUMINAMATH_CALUDE_four_digit_integer_proof_l4_458

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_proof (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 17)
  (h3 : middle_digits_sum n = 8)
  (h4 : thousands_minus_units n = 3)
  (h5 : n % 7 = 0) :
  n = 6443 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integer_proof_l4_458


namespace NUMINAMATH_CALUDE_andrei_apple_spending_l4_482

/-- Calculates Andrei's monthly spending on apples given the original price, price increase percentage, discount percentage, and amount bought per month. -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease / 100)
  let discountedPrice := newPrice * (1 - discount / 100)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles under the given conditions. -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 10 10 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_andrei_apple_spending_l4_482


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l4_445

theorem solution_satisfies_equations :
  let x : ℚ := 67 / 9
  let y : ℚ := 22 / 3
  (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l4_445


namespace NUMINAMATH_CALUDE_a_most_stable_l4_446

/-- Represents a person's shooting performance data -/
structure ShootingData where
  name : String
  variance : Real

/-- Defines stability of shooting performance based on variance -/
def isMoreStable (a b : ShootingData) : Prop :=
  a.variance < b.variance

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable (a b c d : ShootingData)
  (ha : a.name = "A" ∧ a.variance = 0.6)
  (hb : b.name = "B" ∧ b.variance = 1.1)
  (hc : c.name = "C" ∧ c.variance = 0.9)
  (hd : d.name = "D" ∧ d.variance = 1.2) :
  isMoreStable a b ∧ isMoreStable a c ∧ isMoreStable a d :=
sorry

end NUMINAMATH_CALUDE_a_most_stable_l4_446


namespace NUMINAMATH_CALUDE_highest_score_is_242_l4_403

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  average : ℚ
  score_difference : ℕ
  average_drop : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highest_score (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the highest score is 242 -/
theorem highest_score_is_242 (stats : BatsmanStats) 
  (h1 : stats.total_innings = 60)
  (h2 : stats.average = 55)
  (h3 : stats.score_difference = 200)
  (h4 : stats.average_drop = 3) :
  highest_score stats = 242 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_is_242_l4_403


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l4_473

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l4_473


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4_490

-- Problem 1
theorem problem_1 : (Real.sqrt 7 - Real.sqrt 3) * (Real.sqrt 7 + Real.sqrt 3) - (Real.sqrt 6 + Real.sqrt 2)^2 = -4 - 4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (3 * Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4_490


namespace NUMINAMATH_CALUDE_red_other_side_probability_l4_485

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def total_cards : ℕ := 9
def black_both_sides : ℕ := 4
def black_red : ℕ := 2
def red_both_sides : ℕ := 3

def is_red (side : Bool) : Prop := side = true

theorem red_other_side_probability :
  let cards : List Card := 
    (List.replicate black_both_sides ⟨false, false⟩) ++
    (List.replicate black_red ⟨false, true⟩) ++
    (List.replicate red_both_sides ⟨true, true⟩)
  let total_red_sides := red_both_sides * 2 + black_red
  let red_both_sides_count := red_both_sides * 2
  (red_both_sides_count : ℚ) / total_red_sides = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_red_other_side_probability_l4_485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4_459

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h3 : ∃ d, arithmetic_sequence a d) :
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4_459


namespace NUMINAMATH_CALUDE_tan_double_alpha_l4_454

theorem tan_double_alpha (α β : Real) 
  (h1 : Real.tan (α + β) = 3) 
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (2 * α) = -1 := by
sorry

end NUMINAMATH_CALUDE_tan_double_alpha_l4_454


namespace NUMINAMATH_CALUDE_chess_class_percentage_l4_430

theorem chess_class_percentage 
  (total_students : ℕ) 
  (swimming_students : ℕ) 
  (chess_to_swimming_ratio : ℚ) :
  total_students = 1000 →
  swimming_students = 125 →
  chess_to_swimming_ratio = 1/2 →
  (↑swimming_students : ℚ) / (chess_to_swimming_ratio * ↑total_students) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_chess_class_percentage_l4_430


namespace NUMINAMATH_CALUDE_soccer_lineup_selections_l4_466

/-- The number of players in the soccer team -/
def team_size : ℕ := 16

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select the starting lineup -/
def lineup_selections : ℕ := 409500

/-- Theorem: The number of ways to select a starting lineup of 5 players from a team of 16,
    where one player (utility) cannot be selected for a specific position (goalkeeper),
    is equal to 409,500. -/
theorem soccer_lineup_selections :
  (team_size - 1) *  -- Goalkeeper selection (excluding utility player)
  (team_size - 1) *  -- Defender selection
  (team_size - 2) *  -- Midfielder selection
  (team_size - 3) *  -- Forward selection
  (team_size - 4)    -- Utility player selection (excluding goalkeeper)
  = lineup_selections := by sorry

end NUMINAMATH_CALUDE_soccer_lineup_selections_l4_466


namespace NUMINAMATH_CALUDE_problem_statement_l4_441

theorem problem_statement (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_a : a ≥ 1/a + 2/b) (h_b : b ≥ 3/a + 2/b) :
  (a + b ≥ 4) ∧ 
  (a^2 + b^2 ≥ 3 + 2*Real.sqrt 6) ∧ 
  (1/a + 1/b < 1 + Real.sqrt 2 / 2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4_441


namespace NUMINAMATH_CALUDE_dog_adoption_rate_is_half_l4_429

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialCats : ℕ
  initialLizards : ℕ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterMonth : ℕ

/-- Calculates the dog adoption rate given the shelter scenario --/
def dogAdoptionRate (shelter : AnimalShelter) : ℚ :=
  let totalInitial := shelter.initialDogs + shelter.initialCats + shelter.initialLizards
  let adoptedCats := shelter.catAdoptionRate * shelter.initialCats
  let adoptedLizards := shelter.lizardAdoptionRate * shelter.initialLizards
  let remainingPets := shelter.totalPetsAfterMonth - shelter.newPetsPerMonth
  ((totalInitial - remainingPets) - (adoptedCats + adoptedLizards)) / shelter.initialDogs

/-- Theorem stating that the dog adoption rate is 50% for the given scenario --/
theorem dog_adoption_rate_is_half (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialCats = 28)
  (h3 : shelter.initialLizards = 20)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterMonth = 65) :
  dogAdoptionRate shelter = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_dog_adoption_rate_is_half_l4_429


namespace NUMINAMATH_CALUDE_square_difference_equals_eight_xy_l4_419

theorem square_difference_equals_eight_xy (x y A : ℝ) :
  (x + 2*y)^2 = (x - 2*y)^2 + A → A = 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_eight_xy_l4_419


namespace NUMINAMATH_CALUDE_min_rooms_in_apartment_l4_483

/-- Represents an apartment with rooms and doors. -/
structure Apartment where
  rooms : ℕ
  doors : ℕ
  at_most_one_door_between_rooms : Bool
  at_most_one_door_to_outside : Bool

/-- Checks if the apartment configuration is valid. -/
def is_valid_apartment (a : Apartment) : Prop :=
  a.at_most_one_door_between_rooms ∧
  a.at_most_one_door_to_outside ∧
  a.doors = 12

/-- Theorem: The minimum number of rooms in a valid apartment is 5. -/
theorem min_rooms_in_apartment (a : Apartment) 
  (h : is_valid_apartment a) : a.rooms ≥ 5 := by
  sorry

#check min_rooms_in_apartment

end NUMINAMATH_CALUDE_min_rooms_in_apartment_l4_483


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4_412

/-- Given a geometric sequence {aₙ} where the first three terms are x, 2x+2, and 3x+3 respectively,
    prove that the fourth term a₄ = -27/2. -/
theorem geometric_sequence_fourth_term (x : ℝ) (a : ℕ → ℝ) :
  a 1 = x ∧ a 2 = 2*x + 2 ∧ a 3 = 3*x + 3 ∧ 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) →
  a 4 = -27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4_412


namespace NUMINAMATH_CALUDE_certain_number_operations_l4_476

theorem certain_number_operations (x : ℝ) : 
  (((x + 5) * 2) / 5) - 5 = 62.5 / 2 → x = 85.625 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_operations_l4_476


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l4_415

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) → 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) ∧ 
  n = 1007 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l4_415


namespace NUMINAMATH_CALUDE_joan_has_nine_balloons_l4_436

/-- The number of blue balloons that Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons that Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := 16

/-- The number of blue balloons that Joan has -/
def joan_balloons : ℕ := total_balloons - (sally_balloons + jessica_balloons)

theorem joan_has_nine_balloons : joan_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_nine_balloons_l4_436


namespace NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l4_413

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) :
  b / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l4_413


namespace NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l4_418

/-- In an isosceles triangle DEF, G is the point where the altitude from D meets EF. -/
structure IsoscelesTriangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  is_isosceles : dist D E = dist D F
  altitude : (G.1 - D.1) * (E.1 - F.1) + (G.2 - D.2) * (E.2 - F.2) = 0
  on_base : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = (1 - t) • E + t • F

/-- The length of EF in the isosceles triangle DEF. -/
def EF_length (triangle : IsoscelesTriangle) : ℝ :=
  dist triangle.E triangle.F

/-- The theorem stating the length of EF in the specific isosceles triangle. -/
theorem isosceles_triangle_ef_length 
  (triangle : IsoscelesTriangle)
  (de_length : dist triangle.D triangle.E = 5)
  (eg_gf_ratio : dist triangle.E triangle.G = 4 * dist triangle.G triangle.F) :
  EF_length triangle = (5 * Real.sqrt 10) / 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l4_418


namespace NUMINAMATH_CALUDE_prime_square_plus_two_l4_443

theorem prime_square_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_l4_443


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4_447

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 1 ∧ b = -8 ∧ c = 16) :
  ∃! x : ℝ, x^2 - 8*x + 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4_447


namespace NUMINAMATH_CALUDE_simplify_expression_value_given_condition_value_given_equations_l4_434

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3*(x+y)^2 - 7*(x+y) + 8*(x+y)^2 + 6*(x+y) = 11*(x+y)^2 - (x+y) := by sorry

-- Part 2
theorem value_given_condition (a : ℝ) (h : a^2 + 2*a = 3) :
  3*a^2 + 6*a - 14 = -5 := by sorry

-- Part 3
theorem value_given_equations (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b + c = 5) (h3 : c - 4*d = -7) :
  (a - 2*b) - (3*b - c) - (c + 4*d) = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_value_given_condition_value_given_equations_l4_434


namespace NUMINAMATH_CALUDE_prime_binomial_divisibility_l4_422

theorem prime_binomial_divisibility (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end NUMINAMATH_CALUDE_prime_binomial_divisibility_l4_422


namespace NUMINAMATH_CALUDE_soda_distribution_l4_428

theorem soda_distribution (boxes : Nat) (cans_per_box : Nat) (discarded : Nat) (cartons : Nat) :
  boxes = 7 →
  cans_per_box = 16 →
  discarded = 13 →
  cartons = 8 →
  (boxes * cans_per_box - discarded) % cartons = 3 := by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_l4_428


namespace NUMINAMATH_CALUDE_intersection_right_triangle_l4_467

/-- Given a line and a circle in the Cartesian plane, if they intersect at two points
    forming a right triangle with the circle's center, then the parameter of the line
    and circle equations must be -1. -/
theorem intersection_right_triangle (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (a * A.1 + A.2 - 2 = 0 ∧ (A.1 - 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 + B.2 - 2 = 0 ∧ (B.1 - 1)^2 + (B.2 - a)^2 = 16) ∧
    A ≠ B ∧
    let C : ℝ × ℝ := (1, a)
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) →
  a = -1 := by
sorry


end NUMINAMATH_CALUDE_intersection_right_triangle_l4_467


namespace NUMINAMATH_CALUDE_no_natural_solution_l4_491

theorem no_natural_solution (x y z : ℕ) : (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l4_491


namespace NUMINAMATH_CALUDE_restaurant_bill_share_l4_495

/-- Calculate each person's share of a restaurant bill with tip -/
theorem restaurant_bill_share 
  (total_bill : ℝ) 
  (num_people : ℕ) 
  (tip_percentage : ℝ) 
  (h1 : total_bill = 211)
  (h2 : num_people = 5)
  (h3 : tip_percentage = 0.15) : 
  (total_bill * (1 + tip_percentage)) / num_people = 48.53 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_share_l4_495


namespace NUMINAMATH_CALUDE_correct_proposition_l4_409

-- Define proposition p₁
def p₁ : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l4_409


namespace NUMINAMATH_CALUDE_binomial_10_3_l4_406

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l4_406


namespace NUMINAMATH_CALUDE_accurate_to_tenths_l4_472

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : Int
  fractionalPart : Nat
  fractionalDigits : Nat

/-- Defines accuracy to a certain decimal place -/
def accurateTo (n : DecimalNumber) (place : Nat) : Prop :=
  n.fractionalDigits ≥ place

/-- The decimal number 3.72 -/
def number : DecimalNumber :=
  { integerPart := 3,
    fractionalPart := 72,
    fractionalDigits := 2 }

/-- The tenths place -/
def tenthsPlace : Nat := 1

theorem accurate_to_tenths :
  accurateTo number tenthsPlace := by sorry

end NUMINAMATH_CALUDE_accurate_to_tenths_l4_472


namespace NUMINAMATH_CALUDE_office_payroll_is_75000_l4_469

/-- Calculates the total monthly payroll for office workers given the following conditions:
  * There are 15 factory workers with a total monthly payroll of $30,000
  * There are 30 office workers
  * The average monthly salary of an office worker exceeds that of a factory worker by $500
-/
def office_workers_payroll (
  factory_workers : ℕ)
  (factory_payroll : ℕ)
  (office_workers : ℕ)
  (salary_difference : ℕ) : ℕ :=
  let factory_avg_salary := factory_payroll / factory_workers
  let office_avg_salary := factory_avg_salary + salary_difference
  office_workers * office_avg_salary

/-- Theorem stating that the total monthly payroll for office workers is $75,000 -/
theorem office_payroll_is_75000 :
  office_workers_payroll 15 30000 30 500 = 75000 := by
  sorry

end NUMINAMATH_CALUDE_office_payroll_is_75000_l4_469


namespace NUMINAMATH_CALUDE_P_homogeneous_P_symmetry_P_normalization_P_unique_l4_420

/-- A binary polynomial that satisfies the given conditions -/
def P (n : ℕ+) (x y : ℝ) : ℝ := x^(n : ℕ) - y^(n : ℕ)

/-- P is homogeneous of degree n -/
theorem P_homogeneous (n : ℕ+) (t x y : ℝ) :
  P n (t * x) (t * y) = t^(n : ℕ) * P n x y := by sorry

/-- P satisfies the symmetry condition -/
theorem P_symmetry (n : ℕ+) (a b c : ℝ) :
  P n (a + b) c + P n (b + c) a + P n (c + a) b = 0 := by sorry

/-- P satisfies the normalization condition -/
theorem P_normalization (n : ℕ+) :
  P n 1 0 = 1 := by sorry

/-- P is the unique polynomial satisfying all conditions -/
theorem P_unique (n : ℕ+) (Q : ℝ → ℝ → ℝ) 
  (h_homogeneous : ∀ t x y, Q (t * x) (t * y) = t^(n : ℕ) * Q x y)
  (h_symmetry : ∀ a b c, Q (a + b) c + Q (b + c) a + Q (c + a) b = 0)
  (h_normalization : Q 1 0 = 1) :
  ∀ x y, Q x y = P n x y := by sorry

end NUMINAMATH_CALUDE_P_homogeneous_P_symmetry_P_normalization_P_unique_l4_420


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l4_486

/-- Represents a text messaging plan with a per-message cost and a monthly fee. -/
structure TextPlan where
  perMessageCost : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages under a specific plan. -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.perMessageCost * messages + plan.monthlyFee

/-- The number of messages at which all plans have the same cost. -/
def equalCostMessages (planA planB planC : TextPlan) : ℚ :=
  60

theorem equal_cost_at_60_messages (planA planB planC : TextPlan) 
    (hA : planA = ⟨0.25, 9⟩) 
    (hB : planB = ⟨0.40, 0⟩)
    (hC : planC = ⟨0.20, 12⟩) : 
    let messages := equalCostMessages planA planB planC
    totalCost planA messages = totalCost planB messages ∧ 
    totalCost planA messages = totalCost planC messages :=
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l4_486


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4_410

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4_410


namespace NUMINAMATH_CALUDE_cubic_equation_root_magnitude_l4_465

theorem cubic_equation_root_magnitude (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_magnitude_l4_465


namespace NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_composite_l4_411

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

-- Part 2
theorem find_value (a b : ℝ) (h : a^2 - 2*b = 4) :
  3*a^2 - 6*b - 21 = -9 := by
  sorry

-- Part 3
theorem evaluate_composite (a b c d : ℝ) 
  (h1 : a - 5*b = 3) 
  (h2 : 5*b - 3*c = -5) 
  (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_composite_l4_411


namespace NUMINAMATH_CALUDE_boxes_with_neither_l4_489

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 4)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l4_489


namespace NUMINAMATH_CALUDE_extreme_values_and_sum_l4_449

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem extreme_values_and_sum (α β : ℝ) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x ≥ -2 ∧ f x ≤ 2) ∧
  f (α - Real.pi / 6) = 2 * Real.sqrt 5 / 5 ∧
  Real.sin (β - α) = Real.sqrt 10 / 10 ∧
  α ∈ Set.Icc (Real.pi / 4) Real.pi ∧
  β ∈ Set.Icc Real.pi (3 * Real.pi / 2) →
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = -2) ∧
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = 2) ∧
  α + β = 7 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_sum_l4_449


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l4_416

theorem opposite_of_negative_five : -((-5 : ℤ)) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l4_416


namespace NUMINAMATH_CALUDE_final_amount_after_15_years_l4_414

/-- Calculate the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the final amount after 15 years -/
theorem final_amount_after_15_years :
  simpleInterest 800000 0.07 15 = 1640000 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_15_years_l4_414


namespace NUMINAMATH_CALUDE_max_cookies_eaten_l4_461

/-- Given two people sharing 30 cookies, where one eats twice as many as the other,
    the maximum number of cookies the person eating fewer could have eaten is 10. -/
theorem max_cookies_eaten (total : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ) : 
  total = 30 →
  bella_cookies = 2 * andy_cookies →
  total = andy_cookies + bella_cookies →
  andy_cookies ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l4_461


namespace NUMINAMATH_CALUDE_value_of_one_item_l4_487

/-- Given two persons with equal capitals, each consisting of items of equal value and coins,
    prove that the value of one item is (p - m) / (a - b) --/
theorem value_of_one_item
  (a b : ℕ) (m p : ℝ) (h : a ≠ b)
  (equal_capitals : a * x + m = b * x + p)
  (x : ℝ) :
  x = (p - m) / (a - b) :=
by sorry

end NUMINAMATH_CALUDE_value_of_one_item_l4_487


namespace NUMINAMATH_CALUDE_zero_in_M_l4_427

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l4_427


namespace NUMINAMATH_CALUDE_number_of_observations_l4_426

theorem number_of_observations (initial_mean new_mean : ℝ) 
  (wrong_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 36 →
  wrong_value = 23 →
  correct_value = 34 →
  new_mean = 36.5 →
  (n : ℝ) * initial_mean + (correct_value - wrong_value) = (n : ℝ) * new_mean →
  n = 22 := by
  sorry

#check number_of_observations

end NUMINAMATH_CALUDE_number_of_observations_l4_426


namespace NUMINAMATH_CALUDE_unique_solution_value_l4_455

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant b^2 - 4ac must be zero -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 4x^2 + mx + 16 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  4*x^2 + m*x + 16 = 0

theorem unique_solution_value :
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, has_one_solution 4 m 16) ∧ m = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_value_l4_455


namespace NUMINAMATH_CALUDE_sum_and_operations_l4_425

/-- Given three numbers a, b, and c, and a value M, such that:
    1. a + b + c = 100
    2. a - 10 = M
    3. b + 10 = M
    4. 10 * c = M
    Prove that M = 1000/21 -/
theorem sum_and_operations (a b c M : ℚ) 
  (sum_eq : a + b + c = 100)
  (a_dec : a - 10 = M)
  (b_inc : b + 10 = M)
  (c_mul : 10 * c = M) :
  M = 1000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_operations_l4_425


namespace NUMINAMATH_CALUDE_complex_fraction_equals_one_tenth_l4_417

-- Define the expression
def complex_fraction : ℚ :=
  (⌈(23 / 9 : ℚ) - ⌈(35 / 23 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 9 : ℚ) + ⌈(9 * 23 / 35 : ℚ)⌉⌉ : ℚ)

-- State the theorem
theorem complex_fraction_equals_one_tenth : complex_fraction = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_one_tenth_l4_417


namespace NUMINAMATH_CALUDE_color_change_probability_is_0_15_l4_404

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probability_of_color_change (cycle : TrafficLightCycle) (observation_window : ℕ) : ℚ :=
  let total_cycle_time := cycle.green + cycle.yellow + cycle.blue + cycle.red
  let favorable_time := 3 * observation_window  -- 3 color transitions
  (favorable_time : ℚ) / total_cycle_time

/-- Theorem stating the probability of observing a color change is 0.15 for the given cycle -/
theorem color_change_probability_is_0_15 :
  let cycle := TrafficLightCycle.mk 45 5 10 40
  let observation_window := 5
  probability_of_color_change cycle observation_window = 15 / 100 := by
  sorry


end NUMINAMATH_CALUDE_color_change_probability_is_0_15_l4_404


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4_431

/-- Given a geometric sequence {a_n}, prove that if a_1 = 2 and a_9 = 8, then a_5 = 4 -/
theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) 
  (h_first : a 1 = 2) 
  (h_ninth : a 9 = 8) : 
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4_431


namespace NUMINAMATH_CALUDE_vector_coordinates_l4_421

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (0, 3)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_coordinates :
  vector A B = (-4, 3) ∧
  vector B C = (1, -2) ∧
  vector A C = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_l4_421


namespace NUMINAMATH_CALUDE_unique_solution_square_sum_product_l4_405

theorem unique_solution_square_sum_product : 
  ∃! (a b : ℕ+), a^2 + b^2 = a * b * (a + b) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_sum_product_l4_405


namespace NUMINAMATH_CALUDE_g_fixed_points_l4_460

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_fixed_points_l4_460


namespace NUMINAMATH_CALUDE_janice_earnings_l4_435

/-- Calculates the total earnings for Janice's work week -/
def calculate_earnings (days_worked : ℕ) (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_shifts : ℕ) : ℕ :=
  days_worked * daily_rate + overtime_shifts * overtime_rate

/-- Proves that Janice's earnings for the week equal $195 -/
theorem janice_earnings : calculate_earnings 5 30 15 3 = 195 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l4_435


namespace NUMINAMATH_CALUDE_bag_of_balls_l4_401

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 8)
  (h4 : red = 5)
  (h5 : purple = 7)
  (h6 : (white + green + yellow : ℝ) / (white + green + yellow + red + purple) = 0.8) :
  white + green + yellow + red + purple = 60 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l4_401


namespace NUMINAMATH_CALUDE_biased_dice_probability_l4_448

def num_rolls : ℕ := 10
def num_sixes : ℕ := 4
def prob_six : ℚ := 1/3
def prob_not_six : ℚ := 2/3

theorem biased_dice_probability :
  (Nat.choose num_rolls num_sixes) * (prob_six ^ num_sixes) * (prob_not_six ^ (num_rolls - num_sixes)) = 13440/59049 :=
by sorry

end NUMINAMATH_CALUDE_biased_dice_probability_l4_448


namespace NUMINAMATH_CALUDE_scientific_notation_of_given_number_l4_463

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number in millions -/
def givenNumber : ℝ := 141260

theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 1.4126 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_given_number_l4_463


namespace NUMINAMATH_CALUDE_min_c_plus_d_l4_453

theorem min_c_plus_d (a b c d : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  a < b ∧ b < c ∧ c < d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (n : ℕ), a + b + c + d = n^2 →
  11 ≤ c + d ∧ ∃ (a' b' c' d' : ℕ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' ∧
    ∃ (m : ℕ), a' + b' + c' + d' = m^2 ∧
    c' + d' = 11 :=
by sorry

end NUMINAMATH_CALUDE_min_c_plus_d_l4_453


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l4_470

/-- 
Given a bucket that weighs p kilograms when three-fourths full of water
and q kilograms when one-third full of water, this theorem proves that
the weight of the bucket when full is (8p - 3q) / 5 kilograms.
-/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- The proof of the bucket_weight theorem. -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l4_470


namespace NUMINAMATH_CALUDE_rectangular_region_area_l4_499

/-- The area of a rectangular region enclosed by lines derived from given equations -/
theorem rectangular_region_area (a : ℝ) (ha : a > 0) :
  let eq1 (x y : ℝ) := (2 * x - a * y)^2 = 25 * a^2
  let eq2 (x y : ℝ) := (5 * a * x + 2 * y)^2 = 36 * a^2
  let area := (120 * a^2) / Real.sqrt (100 * a^2 + 16 + 100 * a^4)
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    eq1 x1 y1 ∧ eq1 x2 y2 ∧ eq1 x3 y3 ∧ eq1 x4 y4 ∧
    eq2 x1 y1 ∧ eq2 x2 y2 ∧ eq2 x3 y3 ∧ eq2 x4 y4 ∧
    (x1 - x2) * (y1 - y3) = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_region_area_l4_499


namespace NUMINAMATH_CALUDE_triangle_side_length_l4_452

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b^2 - 6*b + 8 = 0 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4_452


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l4_478

theorem max_value_trig_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) ≤ Real.sqrt (a^2 + b^2 + 2 * c^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) = Real.sqrt (a^2 + b^2 + 2 * c^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l4_478


namespace NUMINAMATH_CALUDE_sqrt_sum_equal_product_equal_l4_488

-- Problem 1
theorem sqrt_sum_equal : Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24 = 3 * Real.sqrt 6 := by sorry

-- Problem 2
theorem product_equal : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equal_product_equal_l4_488


namespace NUMINAMATH_CALUDE_vector_addition_l4_481

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![3, -7]
  let v2 : Fin 2 → ℝ := ![-6, 11]
  v1 + v2 = ![(-3), 4] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l4_481


namespace NUMINAMATH_CALUDE_olly_ferrets_l4_497

theorem olly_ferrets (num_dogs : ℕ) (num_cats : ℕ) (total_shoes : ℕ) :
  num_dogs = 3 →
  num_cats = 2 →
  total_shoes = 24 →
  ∃ (num_ferrets : ℕ),
    num_ferrets * 4 + num_dogs * 4 + num_cats * 4 = total_shoes ∧
    num_ferrets = 1 :=
by sorry

end NUMINAMATH_CALUDE_olly_ferrets_l4_497


namespace NUMINAMATH_CALUDE_power_of_power_three_l4_477

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l4_477


namespace NUMINAMATH_CALUDE_parabola_directrix_l4_457

/-- Given a parabola with equation y = ax^2 and directrix y = 1, prove that a = -1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Equation of the parabola
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y ≠ a * x^2) →  -- Condition 2: Equation of the directrix
  a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4_457


namespace NUMINAMATH_CALUDE_f_properties_l4_498

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x, f x ≤ 0) ∧
  (f 0 = 0) ∧
  (∀ x, f x ≥ -4) ∧
  (f 2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4_498


namespace NUMINAMATH_CALUDE_baseball_team_selection_l4_442

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets that must be included -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 9

theorem baseball_team_selection :
  choose (total_players - quadruplets) (starters - quadruplets) = 2002 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_selection_l4_442


namespace NUMINAMATH_CALUDE_starting_team_combinations_l4_444

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The total number of team members --/
def totalMembers : ℕ := 20

/-- The number of players in the starting team --/
def startingTeamSize : ℕ := 9

/-- The number of goalkeepers --/
def numGoalkeepers : ℕ := 2

/-- Theorem stating the number of ways to choose the starting team --/
theorem starting_team_combinations : 
  (choose totalMembers numGoalkeepers) * (choose (totalMembers - numGoalkeepers) (startingTeamSize - numGoalkeepers)) = 6046560 := by
  sorry

end NUMINAMATH_CALUDE_starting_team_combinations_l4_444


namespace NUMINAMATH_CALUDE_rectangle_segment_length_l4_402

/-- Given a rectangle with dimensions 10 units by 5 units, prove that the total length
    of segments in a new figure formed by removing three sides is 15 units. The remaining
    segments include two full heights and two parts of the width (3 units and 2 units). -/
theorem rectangle_segment_length :
  let original_width : ℕ := 10
  let original_height : ℕ := 5
  let remaining_width_part1 : ℕ := 3
  let remaining_width_part2 : ℕ := 2
  let total_length : ℕ := 2 * original_height + remaining_width_part1 + remaining_width_part2
  total_length = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_segment_length_l4_402


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l4_480

theorem sum_of_two_numbers : ∃ (a b : ℤ), 
  (a = |(-10)| + 1) ∧ 
  (b = -(2) - 1) ∧ 
  (a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l4_480


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4_423

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4_423


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4_407

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →  -- Measure of angle D is 80 degrees
  E = 4 * F + 10 →  -- Measure of angle E is 10 degrees more than four times the measure of angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 18 :=  -- Measure of angle F is 18 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4_407


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l4_424

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l4_424


namespace NUMINAMATH_CALUDE_function_growth_l4_408

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < deriv f x) (a : ℝ) (ha : 0 < a) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l4_408


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4_400

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) :
  a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4_400


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l4_440

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l4_440


namespace NUMINAMATH_CALUDE_product_of_multiples_l4_432

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem product_of_multiples : 
  smallest_two_digit_multiple_of_5 * smallest_three_digit_multiple_of_7 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_product_of_multiples_l4_432


namespace NUMINAMATH_CALUDE_cricket_score_product_l4_437

def first_ten_scores : List Nat := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem cricket_score_product :
  ∀ (score_11 score_12 : Nat),
    score_11 < 15 →
    score_12 < 15 →
    is_integer ((List.sum first_ten_scores + score_11) / 11) →
    is_integer ((List.sum first_ten_scores + score_11 + score_12) / 12) →
    score_11 * score_12 = 14 :=
by sorry

end NUMINAMATH_CALUDE_cricket_score_product_l4_437


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l4_438

/-- Represents the cistern problem -/
theorem cistern_emptying_time 
  (volume : ℝ) 
  (time_with_tap : ℝ) 
  (tap_rate : ℝ) 
  (h1 : volume = 480) 
  (h2 : time_with_tap = 24) 
  (h3 : tap_rate = 4) : 
  (volume / (volume / time_with_tap - tap_rate) = 30) := by
  sorry

#check cistern_emptying_time

end NUMINAMATH_CALUDE_cistern_emptying_time_l4_438


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l4_492

theorem complex_modulus_equation (t : ℝ) (h1 : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 2 → t = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l4_492
