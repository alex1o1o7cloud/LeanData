import Mathlib

namespace NUMINAMATH_CALUDE_function_is_identity_l3874_387469

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (a : ℝ), f (f x - y) = f x + f (f y - f a) + x

theorem function_is_identity (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l3874_387469


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3874_387496

/-- A random vector following a normal distribution with mean 3 and variance 1 -/
def X : Type := Real

/-- The probability density function of X -/
noncomputable def pdf (x : X) : Real := sorry

/-- The cumulative distribution function of X -/
noncomputable def cdf (x : X) : Real := sorry

/-- The probability that X is greater than a given value -/
noncomputable def P_greater (a : Real) : Real := 1 - cdf a

/-- The probability that X is less than a given value -/
noncomputable def P_less (a : Real) : Real := cdf a

/-- The theorem stating that if P(X > 2c - 1) = P(X < c + 3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : Real) :
  P_greater (2 * c - 1) = P_less (c + 3) → c = 4/3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3874_387496


namespace NUMINAMATH_CALUDE_dog_weights_l3874_387457

theorem dog_weights (y z : ℝ) : 
  let dog_weights : List ℝ := [25, 31, 35, 33, y, z]
  (dog_weights.take 4).sum / 4 = dog_weights.sum / 6 →
  y + z = 62 := by
sorry

end NUMINAMATH_CALUDE_dog_weights_l3874_387457


namespace NUMINAMATH_CALUDE_no_tangent_line_with_slope_three_halves_for_sine_l3874_387476

theorem no_tangent_line_with_slope_three_halves_for_sine :
  ¬∃ (x : ℝ), Real.cos x = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_tangent_line_with_slope_three_halves_for_sine_l3874_387476


namespace NUMINAMATH_CALUDE_cyclist_wait_time_correct_l3874_387413

/-- The time (in minutes) the cyclist stops to wait after passing the hiker -/
def cyclist_wait_time : ℝ := 3.6667

/-- The hiker's speed in miles per hour -/
def hiker_speed : ℝ := 4

/-- The cyclist's speed in miles per hour -/
def cyclist_speed : ℝ := 15

/-- The time (in minutes) the cyclist waits for the hiker to catch up -/
def catch_up_time : ℝ := 13.75

theorem cyclist_wait_time_correct :
  cyclist_wait_time * (cyclist_speed / 60) = catch_up_time * (hiker_speed / 60) := by
  sorry

#check cyclist_wait_time_correct

end NUMINAMATH_CALUDE_cyclist_wait_time_correct_l3874_387413


namespace NUMINAMATH_CALUDE_ron_pick_frequency_l3874_387404

/-- Represents a book club with a given number of members -/
structure BookClub where
  members : ℕ

/-- Calculates how many times a member gets to pick a book in a year -/
def pickFrequency (club : BookClub) (weeksInYear : ℕ) : ℕ :=
  weeksInYear / club.members

theorem ron_pick_frequency :
  let couples := 3
  let singlePeople := 5
  let ronAndWife := 2
  let weeksInYear := 52
  let club := BookClub.mk (couples * 2 + singlePeople + ronAndWife)
  pickFrequency club weeksInYear = 4 := by
  sorry

end NUMINAMATH_CALUDE_ron_pick_frequency_l3874_387404


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3874_387416

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3874_387416


namespace NUMINAMATH_CALUDE_expected_points_is_16_l3874_387441

/-- The probability of a successful free throw -/
def free_throw_probability : ℝ := 0.8

/-- The number of free throw opportunities in a game -/
def opportunities : ℕ := 10

/-- The number of attempts per free throw opportunity -/
def attempts_per_opportunity : ℕ := 2

/-- The number of points awarded for each successful hit -/
def points_per_hit : ℕ := 1

/-- The expected number of points scored in a game -/
def expected_points : ℝ :=
  (opportunities : ℝ) * (attempts_per_opportunity : ℝ) * free_throw_probability * (points_per_hit : ℝ)

theorem expected_points_is_16 : expected_points = 16 := by sorry

end NUMINAMATH_CALUDE_expected_points_is_16_l3874_387441


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l3874_387477

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The angles are in a ratio of 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l3874_387477


namespace NUMINAMATH_CALUDE_inequality_solution_l3874_387487

def inequality_solution_set : Set ℝ :=
  {x | x < -3 ∨ (-3 < x ∧ x < 3)}

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x + 3) < 0} = inequality_solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3874_387487


namespace NUMINAMATH_CALUDE_abs_rational_inequality_l3874_387417

theorem abs_rational_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_abs_rational_inequality_l3874_387417


namespace NUMINAMATH_CALUDE_coin_inverted_after_two_rolls_l3874_387479

/-- Represents the orientation of a coin -/
inductive CoinOrientation
  | Upright
  | Inverted

/-- Represents a single roll of the coin -/
def single_roll_rotation : ℕ := 270

/-- The total rotation after two equal rolls -/
def total_rotation : ℕ := 2 * single_roll_rotation

/-- Function to determine the final orientation after a given rotation -/
def final_orientation (initial : CoinOrientation) (rotation : ℕ) : CoinOrientation :=
  if rotation % 360 = 180 then
    match initial with
    | CoinOrientation.Upright => CoinOrientation.Inverted
    | CoinOrientation.Inverted => CoinOrientation.Upright
  else initial

/-- Theorem stating that after two equal rolls, the coin will be inverted -/
theorem coin_inverted_after_two_rolls (initial : CoinOrientation) :
  final_orientation initial total_rotation = CoinOrientation.Inverted :=
sorry

end NUMINAMATH_CALUDE_coin_inverted_after_two_rolls_l3874_387479


namespace NUMINAMATH_CALUDE_function_property_l3874_387428

/-- Iterated function application -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ x y, iterate f (x + 1) y + iterate f (y + 1) x = 2 * f (x + y)) :
  ∀ n, f (f n) = f (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3874_387428


namespace NUMINAMATH_CALUDE_selection_methods_count_l3874_387414

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of boys in the class -/
def num_boys : ℕ := 4

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select 4 members from 4 boys and 2 girls, with at least 1 girl -/
def num_selections : ℕ := 
  binomial num_girls 1 * binomial num_boys 3 + 
  binomial num_girls 2 * binomial num_boys 2

theorem selection_methods_count : num_selections = 14 := by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3874_387414


namespace NUMINAMATH_CALUDE_five_by_five_to_fifty_l3874_387456

/-- Represents a square cut into pieces --/
structure CutSquare :=
  (side : ℕ)
  (pieces : ℕ)

/-- Represents the result of reassembling cut pieces --/
structure ReassembledSquares :=
  (count : ℕ)
  (side : ℚ)

/-- Function that cuts a square and reassembles the pieces --/
def cut_and_reassemble (s : CutSquare) : ReassembledSquares :=
  sorry

/-- Theorem stating that a 5x5 square can be cut and reassembled into 50 equal squares --/
theorem five_by_five_to_fifty :
  ∃ (cs : CutSquare) (rs : ReassembledSquares),
    cs.side = 5 ∧
    rs = cut_and_reassemble cs ∧
    rs.count = 50 ∧
    rs.side * rs.side * rs.count = cs.side * cs.side :=
  sorry

end NUMINAMATH_CALUDE_five_by_five_to_fifty_l3874_387456


namespace NUMINAMATH_CALUDE_factorial_ratio_100_98_l3874_387440

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_100_98 : factorial 100 / factorial 98 = 9900 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_100_98_l3874_387440


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3874_387473

theorem solution_set_inequality (x : ℝ) : 
  (x - 1)^2 > 4 ↔ x < -1 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3874_387473


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l3874_387442

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l3874_387442


namespace NUMINAMATH_CALUDE_exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l3874_387443

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  snakes : Nat
  parrots : Nat

/-- Defines the conditions given in the problem -/
def validPetCounts (p : PetCounts) : Prop :=
  p.dogs = 2 ∧
  p.snakes > p.cats ∧
  p.parrots = p.cats - 1 ∧
  p.dogs + p.cats = 6 ∧
  p.dogs + p.cats + p.snakes + p.parrots = 19

/-- Theorem stating that there exists a valid pet count configuration -/
theorem exists_valid_pet_counts : ∃ p : PetCounts, validPetCounts p :=
  sorry

/-- Theorem proving the uniqueness of the valid pet count configuration -/
theorem unique_valid_pet_counts (p q : PetCounts) 
  (hp : validPetCounts p) (hq : validPetCounts q) : p = q :=
  sorry

/-- Main theorem proving that the total number of pets is 19 -/
theorem total_pets_is_nineteen (p : PetCounts) (h : validPetCounts p) :
  p.dogs + p.cats + p.snakes + p.parrots = 19 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l3874_387443


namespace NUMINAMATH_CALUDE_cross_number_puzzle_solution_l3874_387490

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of (base : ℕ) (n : ℕ) : Prop := ∃ m : ℕ, n = base ^ m

theorem cross_number_puzzle_solution :
  ∃! d : ℕ, d < 10 ∧
    (∃ n₃ n₇ : ℕ,
      is_three_digit n₃ ∧
      is_three_digit n₇ ∧
      is_power_of 3 n₃ ∧
      is_power_of 7 n₇ ∧
      (∃ k₃ k₇ : ℕ, n₃ % 10^k₃ / 10^(k₃-1) = d ∧ n₇ % 10^k₇ / 10^(k₇-1) = d)) :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_solution_l3874_387490


namespace NUMINAMATH_CALUDE_jerrys_age_l3874_387498

/-- Given that Mickey's age is 18 and Mickey's age is 4 years less than 400% of Jerry's age,
    prove that Jerry's age is 5.5 years. -/
theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 18 ∧ 
  mickey_age = 4 * jerry_age - 4 → 
  jerry_age = 5.5 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3874_387498


namespace NUMINAMATH_CALUDE_anya_lost_games_l3874_387446

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents a game of table tennis --/
structure Game where
  number : Nat
  players : Fin 2 → Girl
  loser : Girl

/-- The total number of games played --/
def total_games : Nat := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- Predicate to check if a girl lost a specific game --/
def lost_game (g : Girl) (n : Nat) : Prop := ∃ game : Game, game.number = n ∧ game.loser = g

/-- The main theorem to prove --/
theorem anya_lost_games :
  (lost_game Girl.Anya 4) ∧
  (lost_game Girl.Anya 8) ∧
  (lost_game Girl.Anya 12) ∧
  (lost_game Girl.Anya 16) ∧
  (∀ n : Nat, n ≤ total_games → n ≠ 4 → n ≠ 8 → n ≠ 12 → n ≠ 16 → ¬(lost_game Girl.Anya n)) :=
sorry

end NUMINAMATH_CALUDE_anya_lost_games_l3874_387446


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3874_387471

theorem linear_equation_solution (a b : ℝ) :
  (3 : ℝ) * a + (-2 : ℝ) * b = -1 → 3 * a - 2 * b + 2024 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3874_387471


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3874_387488

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x := by
sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3874_387488


namespace NUMINAMATH_CALUDE_complex_number_location_l3874_387425

theorem complex_number_location : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3874_387425


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3874_387464

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ ∃ x : ℝ, x^2 + 2*x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3874_387464


namespace NUMINAMATH_CALUDE_second_grade_sample_l3874_387482

def stratified_sample (total_sample : ℕ) (ratios : List ℕ) : List ℕ :=
  let total_ratio := ratios.sum
  ratios.map (λ r => (r * total_sample) / total_ratio)

theorem second_grade_sample :
  let grades_ratio := [3, 3, 4]
  let sample_size := 50
  let samples := stratified_sample sample_size grades_ratio
  samples[1] = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_sample_l3874_387482


namespace NUMINAMATH_CALUDE_mode_median_determinable_l3874_387497

/-- Represents the age distribution of students in the model aviation interest group --/
structure AgeDistribution where
  total : Nat
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Conditions of the problem --/
def aviation_group : AgeDistribution where
  total := 50
  age13 := 5
  age14 := 23
  age15 := 0  -- Unknown, represented as 0
  age16 := 0  -- Unknown, represented as 0

/-- Definition of mode --/
def mode (ad : AgeDistribution) : Nat :=
  max (max ad.age13 ad.age14) (max ad.age15 ad.age16)

/-- Definition of median for even number of students --/
def median (ad : AgeDistribution) : Nat :=
  if ad.age13 + ad.age14 ≥ ad.total / 2 then 14 else 15

/-- Main theorem --/
theorem mode_median_determinable (ad : AgeDistribution) 
  (h1 : ad.total = 50)
  (h2 : ad.age13 = 5)
  (h3 : ad.age14 = 23)
  (h4 : ad.age15 + ad.age16 = ad.total - ad.age13 - ad.age14) :
  (∃ (m : Nat), mode ad = m) ∧ 
  (∃ (n : Nat), median ad = n) ∧
  (¬ ∃ (mean : ℚ), true) ∧  -- Mean cannot be determined
  (¬ ∃ (variance : ℚ), true) :=  -- Variance cannot be determined
sorry


end NUMINAMATH_CALUDE_mode_median_determinable_l3874_387497


namespace NUMINAMATH_CALUDE_boys_percentage_l3874_387432

theorem boys_percentage (total : ℕ) (boys : ℕ) (girls : ℕ) (additional_boys : ℕ) : 
  total = 50 →
  boys + girls = total →
  additional_boys = 50 →
  girls = (total + additional_boys) / 20 →
  (boys : ℚ) / total = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_boys_percentage_l3874_387432


namespace NUMINAMATH_CALUDE_johns_new_height_l3874_387493

/-- Calculates the new height in feet after a growth spurt -/
def new_height_in_feet (initial_height_inches : ℕ) (growth_rate_inches_per_month : ℕ) (growth_duration_months : ℕ) : ℚ :=
  (initial_height_inches + growth_rate_inches_per_month * growth_duration_months) / 12

/-- Proves that John's new height is 6 feet -/
theorem johns_new_height :
  new_height_in_feet 66 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_johns_new_height_l3874_387493


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3874_387408

/-- Calculates the average speed of a round trip given the outbound speed and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h : outbound_speed = 45) : 
  let return_time := 2 * (1 / outbound_speed)
  let total_time := 1 / outbound_speed + return_time
  let total_distance := 2
  (total_distance / total_time) = 30 := by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3874_387408


namespace NUMINAMATH_CALUDE_container_production_l3874_387448

/-- Represents the production rate of containers per worker per hour -/
def container_rate : ℝ := by sorry

/-- Represents the production rate of covers per worker per hour -/
def cover_rate : ℝ := by sorry

/-- The number of containers produced by 80 workers in 2 hours -/
def containers_80_2 : ℝ := 320

/-- The number of covers produced by 80 workers in 2 hours -/
def covers_80_2 : ℝ := 160

/-- The number of containers produced by 100 workers in 3 hours -/
def containers_100_3 : ℝ := 450

/-- The number of covers produced by 100 workers in 3 hours -/
def covers_100_3 : ℝ := 300

/-- The number of covers produced by 40 workers in 4 hours -/
def covers_40_4 : ℝ := 160

theorem container_production :
  80 * 2 * container_rate = containers_80_2 ∧
  80 * 2 * cover_rate = covers_80_2 ∧
  100 * 3 * container_rate = containers_100_3 ∧
  100 * 3 * cover_rate = covers_100_3 ∧
  40 * 4 * cover_rate = covers_40_4 →
  40 * 4 * container_rate = 160 := by sorry

end NUMINAMATH_CALUDE_container_production_l3874_387448


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l3874_387491

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), x^3 + 2*y^2 + 4*z = n := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l3874_387491


namespace NUMINAMATH_CALUDE_find_a_solution_set_g_solution_set_h_l3874_387411

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x^2 - 4 * x + 6

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ -3 < x ∧ x < 1

-- Theorem 1
theorem find_a (a : ℝ) (h : solution_set_condition a) : a = 3 :=
sorry

-- Define the quadratic function for part 2
def g (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Theorem 2
theorem solution_set_g :
  ∀ x, g x > 0 ↔ x < -1 ∨ x > 3/2 :=
sorry

-- Define the quadratic function for part 3
def h (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + b * x + 3

-- Theorem 3
theorem solution_set_h (b : ℝ) :
  (∀ x, h b x ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_find_a_solution_set_g_solution_set_h_l3874_387411


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l3874_387452

def P (n : ℕ) : ℚ := 3 / ((n + 3) * (n + 4))

theorem smallest_n_for_P_less_than_threshold : 
  (∃ n : ℕ, P n < 1 / 2010) ∧ 
  (∀ m : ℕ, m < 23 → P m ≥ 1 / 2010) ∧ 
  (P 23 < 1 / 2010) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l3874_387452


namespace NUMINAMATH_CALUDE_only_B_is_certain_l3874_387420

-- Define the type for events
inductive Event : Type
  | A : Event  -- It will be sunny on New Year's Day in 2020
  | B : Event  -- The sun rises from the east
  | C : Event  -- The TV is turned on and broadcasting the news
  | D : Event  -- Drawing a red ball from a box without any red balls

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.B => True
  | _ => False

-- Theorem statement
theorem only_B_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.B :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_certain_l3874_387420


namespace NUMINAMATH_CALUDE_fourth_term_is_five_l3874_387480

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (b d : ℝ), ∀ n, a n = b + (n - 1) * d

theorem fourth_term_is_five
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 5 = 10) :
  a 4 = 5 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_is_five_l3874_387480


namespace NUMINAMATH_CALUDE_sum_positive_implies_both_positive_is_false_l3874_387424

theorem sum_positive_implies_both_positive_is_false : 
  ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_both_positive_is_false_l3874_387424


namespace NUMINAMATH_CALUDE_exactly_one_B_divisible_by_7_l3874_387474

def is_multiple_of_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

def number_47B (B : ℕ) : ℕ :=
  400 + 70 + B

theorem exactly_one_B_divisible_by_7 :
  ∃! B : ℕ, B ≤ 9 ∧ is_multiple_of_7 (number_47B B) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_B_divisible_by_7_l3874_387474


namespace NUMINAMATH_CALUDE_trajectory_equation_l3874_387468

/-- The trajectory of point M satisfying the given conditions -/
def trajectory_of_M (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ y = (1/4) * x^2

/-- Point P -/
def P : ℝ × ℝ := (0, -3)

/-- Point A on the x-axis -/
def A (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Point Q on the positive y-axis -/
def Q (b : ℝ) : ℝ × ℝ := (0, b)

/-- Condition: Q is on the positive half of y-axis -/
def Q_positive (b : ℝ) : Prop := b > 0

/-- Vector PA -/
def vec_PA (a : ℝ) : ℝ × ℝ := (a - P.1, 0 - P.2)

/-- Vector AM -/
def vec_AM (a x y : ℝ) : ℝ × ℝ := (x - a, y)

/-- Vector MQ -/
def vec_MQ (x y b : ℝ) : ℝ × ℝ := (0 - x, b - y)

/-- Dot product of PA and AM is zero -/
def PA_dot_AM_zero (a x y : ℝ) : Prop :=
  (vec_PA a).1 * (vec_AM a x y).1 + (vec_PA a).2 * (vec_AM a x y).2 = 0

/-- AM = -3/2 * MQ -/
def AM_eq_neg_three_half_MQ (a x y b : ℝ) : Prop :=
  vec_AM a x y = (-3/2 : ℝ) • vec_MQ x y b

/-- The main theorem: given the conditions, prove that M follows the trajectory equation -/
theorem trajectory_equation (x y a b : ℝ) : 
  Q_positive b →
  PA_dot_AM_zero a x y →
  AM_eq_neg_three_half_MQ a x y b →
  trajectory_of_M x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3874_387468


namespace NUMINAMATH_CALUDE_company_a_bottles_company_a_bottles_proof_l3874_387483

/-- Proves that Company A sold 300 bottles given the problem conditions -/
theorem company_a_bottles : ℕ :=
  let company_a_price : ℚ := 4
  let company_b_price : ℚ := 7/2
  let company_b_bottles : ℕ := 350
  let revenue_difference : ℚ := 25
  300

theorem company_a_bottles_proof (company_a_price : ℚ) (company_b_price : ℚ) 
  (company_b_bottles : ℕ) (revenue_difference : ℚ) :
  company_a_price = 4 →
  company_b_price = 7/2 →
  company_b_bottles = 350 →
  revenue_difference = 25 →
  company_a_price * company_a_bottles = 
    company_b_price * company_b_bottles + revenue_difference :=
by sorry

end NUMINAMATH_CALUDE_company_a_bottles_company_a_bottles_proof_l3874_387483


namespace NUMINAMATH_CALUDE_remainder_10_pow_23_minus_7_mod_6_l3874_387406

theorem remainder_10_pow_23_minus_7_mod_6 :
  (10^23 - 7) % 6 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_10_pow_23_minus_7_mod_6_l3874_387406


namespace NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3874_387422

/-- A permutation of numbers from 1 to n satisfies the average condition if for any three indices
    i < j < k, the average of the i-th and k-th elements is not equal to the j-th element. -/
def satisfies_average_condition (n : ℕ) (perm : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (perm i + perm k) / 2 ≠ perm j

/-- For any positive integer n, there exists a permutation of the numbers 1 to n
    that satisfies the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ+) :
  ∃ perm : Fin n → ℕ, Function.Injective perm ∧ Set.range perm = Finset.range n ∧
    satisfies_average_condition n perm :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3874_387422


namespace NUMINAMATH_CALUDE_min_value_function_l3874_387467

theorem min_value_function (x : ℝ) (h : x > 3) : 
  (1 / (x - 3)) + x ≥ 5 ∧ ∃ y > 3, (1 / (y - 3)) + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3874_387467


namespace NUMINAMATH_CALUDE_ribbon_length_proof_l3874_387400

theorem ribbon_length_proof (R : ℝ) : 
  (R / 2 + 2000 = R - ((R / 2 - 2000) / 2 + 2000)) → R = 12000 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_proof_l3874_387400


namespace NUMINAMATH_CALUDE_smallest_tax_price_integer_l3874_387402

theorem smallest_tax_price_integer (n : ℕ) : n = 21 ↔ 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → 
    ¬∃ x : ℕ, (105 * x : ℚ) / 100 = m) ∧
  ∃ x : ℕ, (105 * x : ℚ) / 100 = n :=
by sorry

end NUMINAMATH_CALUDE_smallest_tax_price_integer_l3874_387402


namespace NUMINAMATH_CALUDE_division_theorem_l3874_387401

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 132)
  (h2 : divisor = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l3874_387401


namespace NUMINAMATH_CALUDE_inequality_minimum_a_l3874_387439

theorem inequality_minimum_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_minimum_a_l3874_387439


namespace NUMINAMATH_CALUDE_channel_count_is_164_l3874_387466

/-- Calculates the final number of channels after a series of changes --/
def final_channels (initial : ℕ) : ℕ :=
  let after_first := initial - 20 + 12
  let after_second := after_first - 10 + 8
  let after_third := after_second + 15 - 5
  let overlap := (after_third * 10) / 100
  let after_fourth := after_third + (25 - overlap)
  after_fourth + 7 - 3

/-- Theorem stating that given the initial number of channels and the series of changes, 
    the final number of channels is 164 --/
theorem channel_count_is_164 : final_channels 150 = 164 := by
  sorry

end NUMINAMATH_CALUDE_channel_count_is_164_l3874_387466


namespace NUMINAMATH_CALUDE_boys_girls_difference_l3874_387470

/-- The number of girls on the playground -/
def num_girls : ℝ := 28.0

/-- The number of boys on the playground -/
def num_boys : ℝ := 35.0

/-- The difference between the number of boys and girls -/
def difference : ℝ := num_boys - num_girls

theorem boys_girls_difference : difference = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_boys_girls_difference_l3874_387470


namespace NUMINAMATH_CALUDE_episodes_per_season_l3874_387438

theorem episodes_per_season 
  (days : ℕ) 
  (episodes_per_day : ℕ) 
  (seasons : ℕ) 
  (h1 : days = 10)
  (h2 : episodes_per_day = 6)
  (h3 : seasons = 4)
  (h4 : (days * episodes_per_day) % seasons = 0) : 
  (days * episodes_per_day) / seasons = 15 := by
sorry

end NUMINAMATH_CALUDE_episodes_per_season_l3874_387438


namespace NUMINAMATH_CALUDE_equation_solution_l3874_387484

theorem equation_solution (a b x : ℝ) : 
  (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 4) ∨ 
  (b = Real.sqrt 2 * a ∧ ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 4) ∨
  (b = -Real.sqrt 2 * a ∧ ∃ k : ℤ, x = (2 * k + 1) * Real.pi) := by
sorry


end NUMINAMATH_CALUDE_equation_solution_l3874_387484


namespace NUMINAMATH_CALUDE_product_increased_by_amount_l3874_387423

theorem product_increased_by_amount (x y : ℝ) (h1 : x = 3) (h2 : 5 * x + y = 19) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_by_amount_l3874_387423


namespace NUMINAMATH_CALUDE_solve_system_l3874_387429

theorem solve_system (A B C D : ℤ) 
  (eq1 : A + C = 15)
  (eq2 : A - B = 1)
  (eq3 : C + C = A)
  (eq4 : B - D = 2)
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3874_387429


namespace NUMINAMATH_CALUDE_tax_increase_l3874_387462

/-- Calculates the tax amount based on income and tax rates -/
def calculate_tax (income : ℕ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  if income ≤ 500000 then
    (income : ℚ) * rate1
  else if income ≤ 1000000 then
    500000 * rate1 + ((income - 500000) : ℚ) * rate2
  else
    500000 * rate1 + 500000 * rate2 + ((income - 1000000) : ℚ) * rate2

/-- Represents the tax system change and income increase -/
theorem tax_increase :
  let old_tax := calculate_tax 1000000 (20/100) (25/100)
  let new_main_tax := calculate_tax 1500000 (30/100) (35/100)
  let rental_income : ℚ := 100000
  let rental_deduction : ℚ := 10/100
  let taxable_rental := rental_income * (1 - rental_deduction)
  let rental_tax := taxable_rental * (35/100)
  let new_total_tax := new_main_tax + rental_tax
  new_total_tax - old_tax = 306500 := by sorry

end NUMINAMATH_CALUDE_tax_increase_l3874_387462


namespace NUMINAMATH_CALUDE_both_correct_undetermined_l3874_387437

/-- Represents a class of students and their test performance -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Predicate to check if the number of students who answered both questions correctly is determinable -/
def both_correct_determinable (c : ClassTestResults) : Prop :=
  ∃ (n : ℕ), n ≤ c.correct_q1 ∧ n ≤ c.correct_q2 ∧ n = c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that the number of students who answered both questions correctly is undetermined -/
theorem both_correct_undetermined (c : ClassTestResults)
  (h1 : c.total_students = 25)
  (h2 : c.correct_q1 = 22)
  (h3 : c.absent = 3)
  (h4 : c.correct_q2 ≤ c.total_students - c.absent)
  (h5 : c.correct_q2 > 0) :
  ¬ both_correct_determinable c := by
  sorry


end NUMINAMATH_CALUDE_both_correct_undetermined_l3874_387437


namespace NUMINAMATH_CALUDE_smallest_p_value_l3874_387433

theorem smallest_p_value (p q : ℕ+) (h1 : (5 : ℚ) / 8 < p / q) (h2 : p / q < (7 : ℚ) / 8) (h3 : p + q = 2005) :
  p.val ≥ 772 ∧ (∀ (p' : ℕ+), p'.val ≥ 772 → (5 : ℚ) / 8 < p' / (2005 - p') → p' / (2005 - p') < (7 : ℚ) / 8 → p'.val ≤ p.val) :=
sorry

end NUMINAMATH_CALUDE_smallest_p_value_l3874_387433


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l3874_387418

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l3874_387418


namespace NUMINAMATH_CALUDE_max_angle_is_90_deg_l3874_387403

/-- A regular quadrilateral prism with height half the side length of its base -/
structure RegularQuadPrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- A point on the edge AB of the prism -/
def PointOnAB (prism : RegularQuadPrism) := {x : ℝ // 0 ≤ x ∧ x ≤ prism.base_side}

/-- The angle A₁MC₁ where M is a point on AB -/
def angleA1MC1 (prism : RegularQuadPrism) (m : PointOnAB prism) : ℝ := sorry

/-- The maximum value of angle A₁MC₁ is 90° -/
theorem max_angle_is_90_deg (prism : RegularQuadPrism) :
  ∃ (m : PointOnAB prism), angleA1MC1 prism m = π / 2 ∧
  ∀ (m' : PointOnAB prism), angleA1MC1 prism m' ≤ π / 2 :=
sorry

end NUMINAMATH_CALUDE_max_angle_is_90_deg_l3874_387403


namespace NUMINAMATH_CALUDE_range_of_m_l3874_387486

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

def p (m : ℝ) : Prop := ∀ x ≥ 2, Monotone (f m)

def q (m : ℝ) : Prop := ∀ x, m*x^2 + 2*(m-2)*x + 1 > 0

theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Iic 1 ∪ Set.Ioo 2 4) ↔ (p m ∨ q m) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3874_387486


namespace NUMINAMATH_CALUDE_max_absolute_value_quadratic_l3874_387492

theorem max_absolute_value_quadratic (a b : ℝ) :
  (∃ m : ℝ, ∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) ∧
  (∀ m : ℝ, (∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_quadratic_l3874_387492


namespace NUMINAMATH_CALUDE_bob_eats_one_more_than_george_l3874_387409

/-- Represents the number of slices in different pizza sizes and quantities purchased --/
structure PizzaOrder where
  small_slices : ℕ := 4
  large_slices : ℕ := 8
  small_count : ℕ := 3
  large_count : ℕ := 2

/-- Represents the pizza consumption of different people --/
structure PizzaConsumption where
  george : ℕ := 3
  bill : ℕ := 3
  fred : ℕ := 3
  mark : ℕ := 3
  leftover : ℕ := 10

/-- Theorem stating that Bob eats one more slice than George --/
theorem bob_eats_one_more_than_george (order : PizzaOrder) (consumption : PizzaConsumption) : 
  ∃ (bob : ℕ) (susie : ℕ), 
    susie = bob / 2 ∧ 
    bob = consumption.george + 1 ∧
    order.small_slices * order.small_count + order.large_slices * order.large_count = 
      consumption.george + bob + susie + consumption.bill + consumption.fred + consumption.mark + consumption.leftover :=
by
  sorry

end NUMINAMATH_CALUDE_bob_eats_one_more_than_george_l3874_387409


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3874_387431

/-- Given two similar triangles with an area ratio of 1:9 and the smaller triangle
    having a height of 5 cm, prove that the corresponding height of the larger triangle
    is 15 cm. -/
theorem similar_triangles_height (small_height large_height : ℝ) :
  small_height = 5 →
  (9 : ℝ) * small_height^2 = large_height^2 →
  large_height = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3874_387431


namespace NUMINAMATH_CALUDE_metro_earnings_proof_l3874_387419

/-- Calculates the earnings from ticket sales given the ticket price, average tickets sold per minute, and duration in minutes. -/
def calculate_earnings (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ) : ℝ :=
  ticket_price * tickets_per_minute * duration

/-- Proves that the earnings from ticket sales for the given conditions equal $90. -/
theorem metro_earnings_proof (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ)
  (h1 : ticket_price = 3)
  (h2 : tickets_per_minute = 5)
  (h3 : duration = 6) :
  calculate_earnings ticket_price tickets_per_minute duration = 90 := by
  sorry

end NUMINAMATH_CALUDE_metro_earnings_proof_l3874_387419


namespace NUMINAMATH_CALUDE_min_xy_value_l3874_387489

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x * y ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3874_387489


namespace NUMINAMATH_CALUDE_system_solution_l3874_387445

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 9) 
  (eq2 : 3 * x + 2 * y = 11) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3874_387445


namespace NUMINAMATH_CALUDE_complex_distance_l3874_387434

theorem complex_distance (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_distance_l3874_387434


namespace NUMINAMATH_CALUDE_train_speed_l3874_387410

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3874_387410


namespace NUMINAMATH_CALUDE_outside_point_distance_l3874_387499

/-- A circle with center O and radius 5 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 5)

/-- A point P outside the circle -/
structure OutsidePoint (c : Circle) :=
  (P : ℝ × ℝ)
  (h_outside : dist P c.O > c.radius)

/-- The statement to prove -/
theorem outside_point_distance {c : Circle} (p : OutsidePoint c) :
  dist p.P c.O > 5 := by sorry

end NUMINAMATH_CALUDE_outside_point_distance_l3874_387499


namespace NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l3874_387460

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of calcium atoms in a molecule of calcium oxide -/
def num_Ca_atoms : ℕ := 1

/-- The number of oxygen atoms in a molecule of calcium oxide -/
def num_O_atoms : ℕ := 1

/-- The molecular weight of calcium oxide in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_O * num_O_atoms

theorem calcium_oxide_molecular_weight :
  molecular_weight_CaO = 56.08 := by sorry

end NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l3874_387460


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l3874_387455

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l3874_387455


namespace NUMINAMATH_CALUDE_fraction_doubled_l3874_387495

theorem fraction_doubled (a b : ℝ) (h : a ≠ b) : 
  (2*a * 2*b) / (2*a - 2*b) = 2 * (a * b / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_doubled_l3874_387495


namespace NUMINAMATH_CALUDE_meal_base_cost_is_28_l3874_387436

/-- Represents the cost structure of a meal --/
structure MealCost where
  baseCost : ℝ
  taxRate : ℝ
  tipRate : ℝ
  totalCost : ℝ

/-- Calculates the total cost of a meal given its base cost, tax rate, and tip rate --/
def calculateTotalCost (m : MealCost) : ℝ :=
  m.baseCost * (1 + m.taxRate + m.tipRate)

/-- Theorem stating that given the specified conditions, the base cost of the meal is $28 --/
theorem meal_base_cost_is_28 (m : MealCost) 
  (h1 : m.taxRate = 0.08)
  (h2 : m.tipRate = 0.18)
  (h3 : m.totalCost = 35.20)
  (h4 : calculateTotalCost m = m.totalCost) :
  m.baseCost = 28 := by
  sorry

#eval (28 : ℚ) * (1 + 0.08 + 0.18)

end NUMINAMATH_CALUDE_meal_base_cost_is_28_l3874_387436


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3874_387458

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) = (2 + Complex.I) / (1 + a * Complex.I) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3874_387458


namespace NUMINAMATH_CALUDE_exists_convex_polyhedron_with_triangular_section_l3874_387454

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- A cross-section of a polyhedron -/
structure CrossSection where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Predicate to check if a cross-section is triangular -/
def is_triangular (cs : CrossSection) : Prop :=
  sorry

/-- Predicate to check if a cross-section passes through vertices -/
def passes_through_vertices (p : ConvexPolyhedron) (cs : CrossSection) : Prop :=
  sorry

/-- Function to count the number of edges meeting at a vertex -/
def edges_at_vertex (p : ConvexPolyhedron) (v : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of a convex polyhedron with the specified properties -/
theorem exists_convex_polyhedron_with_triangular_section :
  ∃ (p : ConvexPolyhedron) (cs : CrossSection),
    is_triangular cs ∧
    ¬passes_through_vertices p cs ∧
    ∀ (v : ℕ), edges_at_vertex p v = 5 :=
  sorry

end NUMINAMATH_CALUDE_exists_convex_polyhedron_with_triangular_section_l3874_387454


namespace NUMINAMATH_CALUDE_scientific_notation_of_175_billion_l3874_387459

theorem scientific_notation_of_175_billion : ∃ (a : ℝ) (n : ℤ), 
  175000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.75 ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_175_billion_l3874_387459


namespace NUMINAMATH_CALUDE_least_common_multiple_of_denominators_l3874_387478

theorem least_common_multiple_of_denominators : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_denominators_l3874_387478


namespace NUMINAMATH_CALUDE_derivative_y_l3874_387407

noncomputable def y (x : ℝ) : ℝ :=
  (1/2) * Real.tanh x + (1/(4*Real.sqrt 2)) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_y (x : ℝ) :
  deriv y x = 1 / (Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l3874_387407


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l3874_387449

/-- An arithmetic sequence is represented by its sums of first n, 2n, and 3n terms. -/
structure ArithmeticSequenceSums where
  S : ℝ  -- Sum of first n terms
  T : ℝ  -- Sum of first 2n terms
  R : ℝ  -- Sum of first 3n terms

/-- 
For any arithmetic sequence, given the sums of its first n, 2n, and 3n terms,
the sum of the first 3n terms equals three times the difference between
the sum of the first 2n terms and the sum of the first n terms.
-/
theorem arithmetic_sequence_sum_relation (seq : ArithmeticSequenceSums) : 
  seq.R = 3 * (seq.T - seq.S) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l3874_387449


namespace NUMINAMATH_CALUDE_dwarfs_truth_count_l3874_387465

theorem dwarfs_truth_count :
  -- Total number of dwarfs
  ∀ (total : ℕ),
  -- Number of dwarfs who raised hands for each ice cream type
  ∀ (vanilla chocolate fruit : ℕ),
  -- Conditions
  total = 10 →
  vanilla = total →
  chocolate = total / 2 →
  fruit = 1 →
  -- Conclusion
  ∃ (truthful : ℕ),
    truthful = 4 ∧
    truthful + (total - truthful) = total ∧
    truthful + 2 * (total - truthful) = vanilla + chocolate + fruit :=
by
  sorry

end NUMINAMATH_CALUDE_dwarfs_truth_count_l3874_387465


namespace NUMINAMATH_CALUDE_equation_solutions_l3874_387447

theorem equation_solutions (x m : ℝ) : 
  ((3 * x - m) / 2 - (x + m) / 3 = 5 / 6) →
  (m = -1 → x = 0) ∧
  (x = 5 → (1 / 2) * m^2 + 2 * m = 30) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3874_387447


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l3874_387475

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 - Real.sqrt 145) / 8) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 8) + u = 0) → 
  u = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l3874_387475


namespace NUMINAMATH_CALUDE_problem_solution_l3874_387463

def f (x : ℝ) : ℝ := 2 * x^2 + 7

def g (x : ℝ) : ℝ := x^3 - 4

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 23) : 
  a = (2 * Real.sqrt 2 + 4) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3874_387463


namespace NUMINAMATH_CALUDE_max_candies_eaten_l3874_387427

/-- Represents the state of the board and the total candies eaten -/
structure BoardState where
  numbers : List Nat
  candies : Nat

/-- The process of combining two numbers on the board -/
def combineNumbers (state : BoardState) : BoardState :=
  match state.numbers with
  | x :: y :: rest => {
      numbers := (x + y) :: rest,
      candies := state.candies + x * y
    }
  | _ => state

/-- Theorem stating the maximum number of candies that can be eaten -/
theorem max_candies_eaten :
  ∃ (final : BoardState),
    (combineNumbers^[48] {numbers := List.replicate 49 1, candies := 0}) = final ∧
    final.candies = 1176 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l3874_387427


namespace NUMINAMATH_CALUDE_gold_foil_thickness_scientific_notation_l3874_387481

theorem gold_foil_thickness_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000092 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 9.2 ∧ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_gold_foil_thickness_scientific_notation_l3874_387481


namespace NUMINAMATH_CALUDE_children_on_bus_l3874_387451

theorem children_on_bus (initial_children : ℕ) (children_who_got_on : ℕ) : 
  initial_children = 18 → children_who_got_on = 7 → 
  initial_children + children_who_got_on = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_l3874_387451


namespace NUMINAMATH_CALUDE_melted_to_spending_value_ratio_l3874_387472

-- Define the weight of a quarter in ounces
def quarter_weight : ℚ := 1/5

-- Define the value of melted gold per ounce in dollars
def melted_gold_value_per_ounce : ℚ := 100

-- Define the spending value of a quarter in dollars
def quarter_spending_value : ℚ := 1/4

-- Theorem statement
theorem melted_to_spending_value_ratio : 
  (melted_gold_value_per_ounce / quarter_weight) / (1 / quarter_spending_value) = 80 := by
  sorry

end NUMINAMATH_CALUDE_melted_to_spending_value_ratio_l3874_387472


namespace NUMINAMATH_CALUDE_baseball_season_games_l3874_387450

/-- The number of baseball games in a season -/
def games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: There are 14 baseball games in a season -/
theorem baseball_season_games :
  games_in_season 7 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_baseball_season_games_l3874_387450


namespace NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l3874_387412

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ)
  (h_length : field_length = 75)
  (h_width : field_width = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_unit : cost_per_unit = 2) :
  let area := path_area field_length field_width path_width
  let cost := construction_cost area cost_per_unit
  area = 675 ∧ cost = 1350 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l3874_387412


namespace NUMINAMATH_CALUDE_system_solution_l3874_387430

theorem system_solution (x y z : ℝ) : 
  (4 * x * y * z = (x + y) * (x * y + 2) ∧ 
   4 * x * y * z = (x + z) * (x * z + 2) ∧ 
   4 * x * y * z = (y + z) * (y * z + 2)) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3874_387430


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l3874_387453

theorem min_value_of_sequence (n : ℝ) : 
  ∃ (m : ℝ), ∀ (n : ℝ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℝ), k^2 - 8*k + 5 = m :=
by
  -- The minimum value is -11
  use -11
  sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l3874_387453


namespace NUMINAMATH_CALUDE_cycling_competition_problem_l3874_387461

/-- Represents the distance Natalia rode on each day of the week -/
structure CyclingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- The cycling competition problem -/
theorem cycling_competition_problem (d : CyclingDistance) : 
  d.tuesday = 50 ∧ 
  d.wednesday = 0.5 * d.tuesday ∧ 
  d.thursday = d.monday + d.wednesday ∧ 
  d.monday + d.tuesday + d.wednesday + d.thursday = 180 →
  d.monday = 40 := by
sorry


end NUMINAMATH_CALUDE_cycling_competition_problem_l3874_387461


namespace NUMINAMATH_CALUDE_all_zero_function_l3874_387444

-- Define the type of our function
def IntFunction := Nat → Nat

-- Define the conditions
def satisfiesConditions (f : IntFunction) : Prop :=
  (∀ m n : Nat, m > 0 ∧ n > 0 → f (m * n) = f m + f n) ∧
  (f 2008 = 0) ∧
  (∀ n : Nat, n > 0 ∧ n % 2008 = 39 → f n = 0)

-- State the theorem
theorem all_zero_function (f : IntFunction) :
  satisfiesConditions f → ∀ n : Nat, n > 0 → f n = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_all_zero_function_l3874_387444


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3874_387415

/-- A curve in the rectangular coordinate system (xOy) -/
structure Curve where
  -- The equation of the curve is implicitly defined by this function
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a curve -/
def eccentricity (c : Curve) : ℝ := sorry

/-- Whether a point lies on a curve -/
def lies_on (p : ℝ × ℝ) (c : Curve) : Prop :=
  c.equation p.1 p.2

/-- The standard equation of a hyperbola -/
def is_standard_hyperbola_equation (c : Curve) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, c.equation x y ↔ y^2 - x^2 = a^2

theorem hyperbola_equation (c : Curve) 
  (h_ecc : eccentricity c = Real.sqrt 2)
  (h_point : lies_on (1, Real.sqrt 2) c) :
  is_standard_hyperbola_equation c ∧ 
  ∃ x y : ℝ, c.equation x y ↔ y^2 - x^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3874_387415


namespace NUMINAMATH_CALUDE_mr_slinkums_shipment_count_l3874_387485

theorem mr_slinkums_shipment_count : ∀ (total : ℕ), 
  (75 : ℚ) / 100 * total = 150 → total = 200 := by
  sorry

end NUMINAMATH_CALUDE_mr_slinkums_shipment_count_l3874_387485


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3874_387435

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3874_387435


namespace NUMINAMATH_CALUDE_sum_of_variables_l3874_387426

theorem sum_of_variables (a b c d e : ℝ) 
  (eq1 : 3*a + 2*b + 4*d = 10)
  (eq2 : 6*a + 5*b + 4*c + 3*d + 2*e = 8)
  (eq3 : a + b + 2*c + 5*e = 3)
  (eq4 : 2*c + 3*d + 3*e = 4)
  (eq5 : a + 2*b + 3*c + d = 7) :
  a + b + c + d + e = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l3874_387426


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3874_387405

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def P : Set ℕ := {3,4,5}
def Q : Set ℕ := {1,3,6}

theorem complement_intersection_equals_set : (U \ P) ∩ (U \ Q) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3874_387405


namespace NUMINAMATH_CALUDE_cake_slices_kept_l3874_387494

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (kept_slices : ℕ) : 
  total_slices = 12 →
  eaten_fraction = 1/4 →
  kept_slices = total_slices - (total_slices * (eaten_fraction.num / eaten_fraction.den).toNat) →
  kept_slices = 9 := by
  sorry

end NUMINAMATH_CALUDE_cake_slices_kept_l3874_387494


namespace NUMINAMATH_CALUDE_fraction_inequality_l3874_387421

theorem fraction_inequality (x : ℝ) : 
  x ∈ Set.Icc (-2 : ℝ) 2 → 
  (6 * x + 1 > 7 - 4 * x ↔ 3 / 5 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3874_387421
