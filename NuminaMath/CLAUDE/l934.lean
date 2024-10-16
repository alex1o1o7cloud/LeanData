import Mathlib

namespace NUMINAMATH_CALUDE_four_intersection_points_range_l934_93466

/-- Parabola C: x^2 = 4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle M: x^2 + (y-4)^2 = r^2 -/
def circle_M (x y r : ℝ) : Prop := x^2 + (y-4)^2 = r^2

/-- The number of intersection points between C and M -/
noncomputable def intersection_count (r : ℝ) : ℕ := sorry

theorem four_intersection_points_range (r : ℝ) :
  r > 0 ∧ intersection_count r = 4 → 2 * Real.sqrt 3 < r ∧ r < 4 := by sorry

end NUMINAMATH_CALUDE_four_intersection_points_range_l934_93466


namespace NUMINAMATH_CALUDE_roots_sum_zero_l934_93474

/-- Given two quadratic trinomials with specific properties, prove their product's roots sum to 0 -/
theorem roots_sum_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  (∃ z₁ z₂ z₃ : ℝ, 
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = z₁ ∨ x = z₂ ∨ x = z₃)) ∧
    z₁ + z₂ + z₃ = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l934_93474


namespace NUMINAMATH_CALUDE_birth_death_rate_decisive_l934_93448

/-- Represents the various characteristics of a population -/
inductive PopulationCharacteristic
  | Density
  | AgeComposition
  | SexRatio
  | BirthRate
  | DeathRate
  | ImmigrationRate
  | EmigrationRate

/-- Represents the impact of a characteristic on population size change -/
inductive Impact
  | Decisive
  | Indirect
  | Basic

/-- Function that maps a population characteristic to its impact on population size change -/
def characteristicImpact : PopulationCharacteristic → Impact
  | PopulationCharacteristic.Density => Impact.Basic
  | PopulationCharacteristic.AgeComposition => Impact.Indirect
  | PopulationCharacteristic.SexRatio => Impact.Indirect
  | PopulationCharacteristic.BirthRate => Impact.Decisive
  | PopulationCharacteristic.DeathRate => Impact.Decisive
  | PopulationCharacteristic.ImmigrationRate => Impact.Decisive
  | PopulationCharacteristic.EmigrationRate => Impact.Decisive

theorem birth_death_rate_decisive :
  ∀ c : PopulationCharacteristic,
    characteristicImpact c = Impact.Decisive →
    c = PopulationCharacteristic.BirthRate ∨ c = PopulationCharacteristic.DeathRate :=
by sorry

end NUMINAMATH_CALUDE_birth_death_rate_decisive_l934_93448


namespace NUMINAMATH_CALUDE_car_sale_profit_l934_93499

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.95 * P
  let selling_price := 1.52 * P
  (selling_price - buying_price) / buying_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_l934_93499


namespace NUMINAMATH_CALUDE_olympic_system_matches_l934_93486

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  is_single_elimination : Bool

/-- Calculates the number of matches in a single-elimination tournament. -/
def matches_played (t : Tournament) : ℕ :=
  if t.is_single_elimination then t.num_teams - 1 else 0

/-- Theorem: A single-elimination tournament with 30 teams has 29 matches. -/
theorem olympic_system_matches :
  ∀ t : Tournament, t.num_teams = 30 ∧ t.is_single_elimination → matches_played t = 29 := by
  sorry

end NUMINAMATH_CALUDE_olympic_system_matches_l934_93486


namespace NUMINAMATH_CALUDE_gnollish_sentences_l934_93423

/-- The number of words in the Gnollish language -/
def num_words : ℕ := 4

/-- The length of a sentence in the Gnollish language -/
def sentence_length : ℕ := 3

/-- The number of invalid sentence patterns due to the restriction -/
def num_invalid_patterns : ℕ := 2

/-- The number of choices for the unrestricted word in an invalid pattern -/
def choices_for_unrestricted : ℕ := num_words

theorem gnollish_sentences :
  (num_words ^ sentence_length) - (num_invalid_patterns * choices_for_unrestricted) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gnollish_sentences_l934_93423


namespace NUMINAMATH_CALUDE_marys_average_speed_l934_93456

/-- Mary's round trip walking problem -/
theorem marys_average_speed (uphill_distance downhill_distance : ℝ)
                             (uphill_time downhill_time : ℝ)
                             (h1 : uphill_distance = 1.5)
                             (h2 : downhill_distance = 1.5)
                             (h3 : uphill_time = 45 / 60)
                             (h4 : downhill_time = 15 / 60) :
  (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end NUMINAMATH_CALUDE_marys_average_speed_l934_93456


namespace NUMINAMATH_CALUDE_sum_345_75_base6_l934_93495

/-- Converts a natural number from base 10 to base 6 -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 6 -/
def addBase6 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 345 and 75 in base 6 is 1540 -/
theorem sum_345_75_base6 :
  addBase6 (toBase6 345) (toBase6 75) = [1, 5, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_345_75_base6_l934_93495


namespace NUMINAMATH_CALUDE_count_integers_in_range_l934_93427

theorem count_integers_in_range : 
  (Finset.range (513 - 2)).card = 511 := by sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l934_93427


namespace NUMINAMATH_CALUDE_certain_number_problem_l934_93457

theorem certain_number_problem (y : ℝ) : 0.5 * 10 = 0.05 * y - 20 → y = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l934_93457


namespace NUMINAMATH_CALUDE_remainder_1234567_div_123_l934_93437

theorem remainder_1234567_div_123 : 1234567 % 123 = 129 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_123_l934_93437


namespace NUMINAMATH_CALUDE_complex_equation_solution_l934_93442

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) :
  (1 - i)^2 / z = 1 + i → z = -1 - i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l934_93442


namespace NUMINAMATH_CALUDE_broken_calculator_multiplication_l934_93493

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem broken_calculator_multiplication :
  ∀ a b : ℕ, is_two_digit a → is_two_digit b →
  (a * b = 1001 ∨ a * b = 1100) ↔ 
  ((a = 11 ∧ b = 91) ∨ (a = 91 ∧ b = 11) ∨
   (a = 13 ∧ b = 77) ∨ (a = 77 ∧ b = 13) ∨
   (a = 25 ∧ b = 44) ∨ (a = 44 ∧ b = 25)) :=
by sorry

end NUMINAMATH_CALUDE_broken_calculator_multiplication_l934_93493


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l934_93417

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x + 3 - 4*x
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 6 ∧ x₂ = 3 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l934_93417


namespace NUMINAMATH_CALUDE_tan_sum_identity_l934_93479

theorem tan_sum_identity : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 
  2 + Real.tan (23 * π / 180) * Real.tan (22 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l934_93479


namespace NUMINAMATH_CALUDE_employed_males_percentage_l934_93494

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 25)
  (h3 : total_population > 0) :
  let employed := (employed_percentage / 100) * total_population
  let employed_females := (employed_females_percentage / 100) * employed
  let employed_males := employed - employed_females
  (employed_males / total_population) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l934_93494


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l934_93497

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l934_93497


namespace NUMINAMATH_CALUDE_paint_cube_cost_l934_93463

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (cube_side_length : ℝ)
  (h1 : paint_cost_per_kg = 60)
  (h2 : paint_coverage_per_kg = 20)
  (h3 : cube_side_length = 10) :
  cube_side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end NUMINAMATH_CALUDE_paint_cube_cost_l934_93463


namespace NUMINAMATH_CALUDE_box_value_proof_l934_93467

theorem box_value_proof : ∃ x : ℝ, (1 + 1.1 + 1.11 + x = 4.44) ∧ (x = 1.23) := by
  sorry

end NUMINAMATH_CALUDE_box_value_proof_l934_93467


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l934_93482

open Set Real

def P : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def Q : Set ℝ := {x | log x < 1}

theorem intersection_P_complement_Q : P ∩ (univ \ Q) = {-3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l934_93482


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l934_93406

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l934_93406


namespace NUMINAMATH_CALUDE_circle_equation_proof_l934_93483

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle equation
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation_proof :
  ∃ (c : Circle),
    (∃ (x : ℝ), line1 x 0 ∧ c.center = (x, 0)) ∧
    (∀ (x y : ℝ), line2 x y → ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)) →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x + 1)^2 + y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l934_93483


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l934_93408

theorem triangle_abc_properties (A B C : Real) (a b : Real) (S : Real) :
  A = 30 * Real.pi / 180 →
  B = 45 * Real.pi / 180 →
  a = Real.sqrt 2 →
  b = a * Real.sin B / Real.sin A →
  C = Real.pi - A - B →
  S = 1/2 * a * b * Real.sin C →
  b = 2 ∧ S = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l934_93408


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l934_93462

def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem complement_A_intersect_B : (U \ A) ∩ B = {6} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l934_93462


namespace NUMINAMATH_CALUDE_roots_properties_l934_93432

theorem roots_properties (z₁ z₂ : ℂ) (h : x^2 + x + 1 = 0 ↔ x = z₁ ∨ x = z₂) :
  z₁ * z₂ = 1 ∧ z₁^3 = 1 ∧ z₂^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l934_93432


namespace NUMINAMATH_CALUDE_mat_cost_per_square_meter_l934_93447

/-- Calculates the cost per square meter of mat for a rectangular hall -/
theorem mat_cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (h_length : length = 20) 
  (h_width : width = 15) 
  (h_height : height = 5) 
  (h_expenditure : total_expenditure = 38000) : 
  total_expenditure / (length * width + 2 * (length * height + width * height)) = 58.46 := by
  sorry

end NUMINAMATH_CALUDE_mat_cost_per_square_meter_l934_93447


namespace NUMINAMATH_CALUDE_total_games_in_season_l934_93436

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  games_per_pair * (n * (n - 1) / 2)

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem total_games_in_season :
  hockey_league_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l934_93436


namespace NUMINAMATH_CALUDE_solutions_count_l934_93439

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (¬ ∃ x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (x + 3) = a * x + 2 ∧ Real.sqrt (y + 3) = a * y + 2) :=
  by sorry

end NUMINAMATH_CALUDE_solutions_count_l934_93439


namespace NUMINAMATH_CALUDE_empty_seats_in_theater_l934_93401

theorem empty_seats_in_theater (total_seats people_watching : ℕ) 
  (h1 : total_seats = 750)
  (h2 : people_watching = 532) :
  total_seats - people_watching = 218 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_in_theater_l934_93401


namespace NUMINAMATH_CALUDE_additional_passengers_proof_l934_93488

/-- The number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- The number of seats in each carriage -/
def seats_per_carriage : ℕ := 25

/-- The total number of passengers that can be accommodated in 3 trains with additional capacity -/
def total_passengers : ℕ := 420

/-- The number of trains -/
def num_trains : ℕ := 3

/-- The additional number of passengers each carriage can accommodate -/
def additional_passengers : ℕ := 10

theorem additional_passengers_proof :
  additional_passengers = 
    (total_passengers - num_trains * carriages_per_train * seats_per_carriage) / 
    (num_trains * carriages_per_train) :=
by sorry

end NUMINAMATH_CALUDE_additional_passengers_proof_l934_93488


namespace NUMINAMATH_CALUDE_company_employees_l934_93470

/-- If a company has 15% more employees in December than in January,
    and it has 490 employees in December, then it had 426 employees in January. -/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 490 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 426 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l934_93470


namespace NUMINAMATH_CALUDE_moon_arrangements_l934_93440

def word : String := "MOON"

theorem moon_arrangements :
  (List.permutations (word.toList)).length = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_l934_93440


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l934_93420

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (1/x₁ + 1/x₂ = -3) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l934_93420


namespace NUMINAMATH_CALUDE_equation_solutions_count_l934_93481

theorem equation_solutions_count : 
  ∃! (pairs : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ pairs ↔ (1 : ℚ) / y - (1 : ℚ) / (y + 2) = (1 : ℚ) / (3 * 2^x)) ∧
    pairs.length = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l934_93481


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l934_93419

theorem set_inclusion_implies_a_value (a : ℝ) :
  let A : Set ℝ := {x | |x| = 1}
  let B : Set ℝ := {x | a * x = 1}
  A ⊇ B →
  a = -1 ∨ a = 0 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l934_93419


namespace NUMINAMATH_CALUDE_circle_center_sum_l934_93430

theorem circle_center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 = 6*X - 8*Y + 24 ↔ (X - x)^2 + (Y - y)^2 = (x^2 + y^2 - 6*x + 8*y - 24)) →
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l934_93430


namespace NUMINAMATH_CALUDE_train_ride_total_time_l934_93460

def train_ride_duration (reading_time eating_time movie_time nap_time : ℕ) : ℕ :=
  reading_time + eating_time + movie_time + nap_time

theorem train_ride_total_time :
  let reading_time : ℕ := 2
  let eating_time : ℕ := 1
  let movie_time : ℕ := 3
  let nap_time : ℕ := 3
  train_ride_duration reading_time eating_time movie_time nap_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_ride_total_time_l934_93460


namespace NUMINAMATH_CALUDE_constant_function_theorem_l934_93410

def is_average_of_neighbors (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

theorem constant_function_theorem (f : ℤ × ℤ → ℕ) 
  (h : is_average_of_neighbors f) : 
  ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l934_93410


namespace NUMINAMATH_CALUDE_horner_v2_value_l934_93475

-- Define the polynomial
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

-- Define Horner's method for the first two steps
def horner_v2 (a : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * a + 1
  v1 * a

-- Theorem statement
theorem horner_v2_value :
  horner_v2 2 = 10 :=
sorry

end NUMINAMATH_CALUDE_horner_v2_value_l934_93475


namespace NUMINAMATH_CALUDE_x_range_l934_93433

theorem x_range (m : ℝ) (h1 : 0 < m) (h2 : m ≤ 5) :
  (∀ x : ℝ, x^2 + (2*m - 1)*x > 4*x + 2*m - 4) →
  (∀ x : ℝ, x < -6 ∨ x > 4) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l934_93433


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l934_93429

theorem walnut_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) :
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l934_93429


namespace NUMINAMATH_CALUDE_club_officer_selection_l934_93458

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (experienced_members : ℕ) : ℕ :=
  experienced_members * (experienced_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem stating the number of ways to choose officers in the given club scenario --/
theorem club_officer_selection :
  let total_members : ℕ := 12
  let experienced_members : ℕ := 4
  choose_officers total_members experienced_members = 1080 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l934_93458


namespace NUMINAMATH_CALUDE_sin_axis_of_symmetry_l934_93496

/-- Proves that x = π/12 is one of the axes of symmetry for the function y = sin(2x + π/3) -/
theorem sin_axis_of_symmetry :
  ∃ (k : ℤ), 2 * (π/12 : ℝ) + π/3 = π/2 + k*π := by sorry

end NUMINAMATH_CALUDE_sin_axis_of_symmetry_l934_93496


namespace NUMINAMATH_CALUDE_cube_constructions_l934_93491

/-- The number of rotational symmetries of a cube -/
def cubeRotations : ℕ := 24

/-- The total number of ways to place 3 blue cubes in 8 positions -/
def totalPlacements : ℕ := Nat.choose 8 3

/-- The number of invariant configurations under 180° rotation around edge axes -/
def edgeRotationInvariants : ℕ := 4

/-- The number of invariant configurations under 180° rotation around face axes -/
def faceRotationInvariants : ℕ := 4

/-- The sum of all fixed points under different rotations -/
def sumFixedPoints : ℕ := totalPlacements + 6 * edgeRotationInvariants + 3 * faceRotationInvariants

/-- The number of unique constructions of a 2x2x2 cube with 5 white and 3 blue unit cubes -/
def uniqueConstructions : ℕ := sumFixedPoints / cubeRotations

theorem cube_constructions : uniqueConstructions = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_constructions_l934_93491


namespace NUMINAMATH_CALUDE_gaochun_temperature_difference_l934_93413

def temperature_difference (low high : Int) : Int :=
  high - low

theorem gaochun_temperature_difference :
  let low : Int := -2
  let high : Int := 9
  temperature_difference low high = 11 := by
  sorry

end NUMINAMATH_CALUDE_gaochun_temperature_difference_l934_93413


namespace NUMINAMATH_CALUDE_prob_one_male_correct_prob_at_least_one_female_correct_l934_93434

/-- Represents the number of female students in the group -/
def num_females : ℕ := 2

/-- Represents the number of male students in the group -/
def num_males : ℕ := 3

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_females + num_males

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one male student -/
def prob_one_male : ℚ := 3 / 5

/-- Calculates the probability of selecting at least one female student -/
def prob_at_least_one_female : ℚ := 7 / 10

/-- Proves that the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_correct : 
  prob_one_male = (num_females * num_males : ℚ) / (total_students.choose num_selected : ℚ) := by
  sorry

/-- Proves that the probability of selecting at least one female student is 7/10 -/
theorem prob_at_least_one_female_correct :
  prob_at_least_one_female = 1 - ((num_males.choose num_selected : ℚ) / (total_students.choose num_selected : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_prob_one_male_correct_prob_at_least_one_female_correct_l934_93434


namespace NUMINAMATH_CALUDE_total_pets_l934_93418

theorem total_pets (taylor_pets : ℕ) (friends_with_double : ℕ) (friends_with_two : ℕ) : 
  taylor_pets = 4 → 
  friends_with_double = 3 → 
  friends_with_two = 2 → 
  taylor_pets + friends_with_double * (2 * taylor_pets) + friends_with_two * 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l934_93418


namespace NUMINAMATH_CALUDE_platform_length_l934_93405

/-- Given a train of length 300 meters that crosses a platform in 45 seconds
    and crosses a signal pole in 18 seconds, prove that the platform length is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let total_distance := train_speed * platform_crossing_time
  train_length + (total_distance - train_length) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l934_93405


namespace NUMINAMATH_CALUDE_transportation_problem_l934_93438

/-- Transportation problem between warehouses and factories -/
theorem transportation_problem 
  (warehouse_a warehouse_b : ℕ)
  (factory_a factory_b : ℕ)
  (cost_a_to_a cost_a_to_b cost_b_to_a cost_b_to_b : ℕ)
  (total_cost : ℕ)
  (h1 : warehouse_a = 20)
  (h2 : warehouse_b = 6)
  (h3 : factory_a = 10)
  (h4 : factory_b = 16)
  (h5 : cost_a_to_a = 400)
  (h6 : cost_a_to_b = 800)
  (h7 : cost_b_to_a = 300)
  (h8 : cost_b_to_b = 500)
  (h9 : total_cost = 16000) :
  ∃ (x y : ℕ),
    x + (warehouse_b - y) = factory_a ∧
    (warehouse_a - x) + y = factory_b ∧
    cost_a_to_a * x + cost_a_to_b * (warehouse_a - x) + 
    cost_b_to_a * (warehouse_b - y) + cost_b_to_b * y = total_cost ∧
    x = 5 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_transportation_problem_l934_93438


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l934_93441

/-- A geometric sequence {a_n} with a_1 = 3 and a_4 = 81 has the general term formula a_n = 3^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∀ n : ℕ, a n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l934_93441


namespace NUMINAMATH_CALUDE_black_squares_eaten_l934_93411

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)
  (black_squares : ℕ)

/-- Represents the eaten portion of the chessboard --/
structure EatenPortion :=
  (black_squares : ℕ)

/-- The theorem stating the number of black squares eaten --/
theorem black_squares_eaten 
  (board : Chessboard) 
  (eaten : EatenPortion) : 
  board.size = 8 → 
  board.black_squares = 32 → 
  eaten.black_squares = 12 :=
by sorry

end NUMINAMATH_CALUDE_black_squares_eaten_l934_93411


namespace NUMINAMATH_CALUDE_parabola_shift_left_l934_93449

/-- The analytical expression of a parabola shifted to the left -/
theorem parabola_shift_left (x y : ℝ) :
  (∀ x, y = x^2) →  -- Original parabola
  (∀ x, y = (x + 1)^2) -- Parabola shifted 1 unit left
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_left_l934_93449


namespace NUMINAMATH_CALUDE_inequality_solution_l934_93400

def solution_set : Set ℝ :=
  {x | x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)}

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l934_93400


namespace NUMINAMATH_CALUDE_isosceles_triangle_ratio_l934_93409

/-- Two isosceles triangles with equal perimeters -/
structure IsoscelesTrianglePair :=
  (base₁ : ℝ)
  (leg₁ : ℝ)
  (base₂ : ℝ)
  (leg₂ : ℝ)
  (base₁_pos : 0 < base₁)
  (leg₁_pos : 0 < leg₁)
  (base₂_pos : 0 < base₂)
  (leg₂_pos : 0 < leg₂)
  (equal_perimeters : base₁ + 2 * leg₁ = base₂ + 2 * leg₂)
  (base_relation : base₂ = 1.15 * base₁)
  (leg_relation : leg₂ = 0.95 * leg₁)

/-- The ratio of the base to the leg of the first triangle is 2:3 -/
theorem isosceles_triangle_ratio (t : IsoscelesTrianglePair) : t.base₁ / t.leg₁ = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ratio_l934_93409


namespace NUMINAMATH_CALUDE_two_power_ten_minus_one_factors_l934_93469

theorem two_power_ten_minus_one_factors : 
  ∃ (p q r : Nat), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (2^10 - 1 : Nat) = p * q * r ∧
    p + q + r = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_two_power_ten_minus_one_factors_l934_93469


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l934_93450

theorem modulus_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l934_93450


namespace NUMINAMATH_CALUDE_max_area_difference_line_l934_93473

/-- The line that maximizes the area difference when passing through P(1,1) and dividing the circle (x-2)^2+y^2 ≤ 4 -/
theorem max_area_difference_line (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 2)^2 + b^2 ≤ 4 → 
    (x - y = 0 → 
      ∀ (m n : ℝ), (m + n - 2 = 0 ∨ y - 1 = 0 ∨ m + 3*n - 4 = 0) → 
        (abs ((a - 2)^2 + b^2 - ((a - x)^2 + (b - y)^2)) ≤ 
         abs ((a - 2)^2 + b^2 - ((a - m)^2 + (b - n)^2))))) :=
by sorry

end NUMINAMATH_CALUDE_max_area_difference_line_l934_93473


namespace NUMINAMATH_CALUDE_tShirts_per_package_example_l934_93421

/-- Given a total number of white t-shirts and a number of packages,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (total : ℕ) (packages : ℕ) : ℕ :=
  total / packages

/-- Theorem: Given 70 white t-shirts in 14 packages,
    prove that each package contains 5 t-shirts. -/
theorem tShirts_per_package_example :
  tShirtsPerPackage 70 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tShirts_per_package_example_l934_93421


namespace NUMINAMATH_CALUDE_transylvanian_truth_telling_l934_93443

-- Define the types
inductive Being
| Human
| Vampire

-- Define the properties
def declares (b : Being) (x : Prop) : Prop :=
  match b with
  | Being.Human => x
  | Being.Vampire => ¬x

theorem transylvanian_truth_telling (b : Being) (x : Prop) :
  (b = Being.Human → (declares b x → x)) ∧
  (b = Being.Vampire → (declares b x → ¬x)) :=
by sorry

end NUMINAMATH_CALUDE_transylvanian_truth_telling_l934_93443


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l934_93468

/-- The number of coins in the final round of distribution -/
def x : ℕ := sorry

/-- The sum of coins Pete gives himself in each round -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution :
  -- Paul ends up with x coins
  -- Pete ends up with 5x coins
  -- Pete's coins follow the pattern 1 + 2 + 3 + ... + x
  -- The total number of coins is 54
  x + 5 * x = 54 ∧ petes_coins x = 5 * x := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l934_93468


namespace NUMINAMATH_CALUDE_initial_speed_is_50_l934_93446

/-- Represents the journey with increasing speed -/
structure Journey where
  distance : ℝ  -- Total distance in km
  time : ℝ      -- Total time in hours
  speedIncrease : ℝ  -- Speed increase in km/h
  intervalTime : ℝ   -- Time interval for speed increase in hours

/-- Calculates the initial speed given a journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the initial speed is 50 km/h -/
theorem initial_speed_is_50 : 
  let j : Journey := {
    distance := 52,
    time := 48 / 60,  -- 48 minutes converted to hours
    speedIncrease := 10,
    intervalTime := 12 / 60  -- 12 minutes converted to hours
  }
  calculateInitialSpeed j = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_50_l934_93446


namespace NUMINAMATH_CALUDE_A_2022_coordinates_l934_93476

/-- The companion point transformation --/
def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

/-- The sequence of points starting from A1 --/
def A : ℕ → ℝ × ℝ
  | 0 => (2, 4)
  | n + 1 => companion_point (A n)

/-- The main theorem --/
theorem A_2022_coordinates :
  A 2021 = (-3, 3) := by
  sorry

end NUMINAMATH_CALUDE_A_2022_coordinates_l934_93476


namespace NUMINAMATH_CALUDE_student_number_problem_l934_93404

theorem student_number_problem (x : ℤ) : x = 60 ↔ 4 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l934_93404


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l934_93428

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l934_93428


namespace NUMINAMATH_CALUDE_max_pons_is_11_l934_93416

/-- Represents the number of items purchased -/
structure Purchase where
  pans : Nat
  pins : Nat
  pons : Nat

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : Nat :=
  3 * p.pans + 5 * p.pins + 8 * p.pons

/-- Checks if a purchase is valid (at least one of each item and total cost is $100) -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pans ≥ 1 ∧ p.pins ≥ 1 ∧ p.pons ≥ 1 ∧ totalCost p = 100

/-- Theorem: The maximum number of pons in a valid purchase is 11 -/
theorem max_pons_is_11 :
  ∀ p : Purchase, isValidPurchase p → p.pons ≤ 11 ∧ 
  ∃ q : Purchase, isValidPurchase q ∧ q.pons = 11 :=
by sorry

#check max_pons_is_11

end NUMINAMATH_CALUDE_max_pons_is_11_l934_93416


namespace NUMINAMATH_CALUDE_relationship_abc_l934_93412

theorem relationship_abc (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x * y) 
  (hc : c = x^2 + y^2) : 
  a^2 = c + 2*b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l934_93412


namespace NUMINAMATH_CALUDE_max_distance_ellipse_line_l934_93490

/-- The maximum distance between a point on the ellipse x²/12 + y²/4 = 1 and the line x + √3y - 6 = 0 is √6 + 3, occurring at the point (-√6, -√2) -/
theorem max_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 12 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let distance (p : ℝ × ℝ) := |p.1 + Real.sqrt 3 * p.2 - 6| / 2
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q ∈ ellipse, distance q ≤ distance p) ∧
    distance p = Real.sqrt 6 + 3 ∧
    p = (-Real.sqrt 6, -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_line_l934_93490


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l934_93403

def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def N : Set ℝ := {-1, 2, 3}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, -3, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l934_93403


namespace NUMINAMATH_CALUDE_unique_base_conversion_l934_93455

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : ℕ) (b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion : 
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 45 = baseBToBase10 113 b :=
by sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l934_93455


namespace NUMINAMATH_CALUDE_distance_AB_is_7_l934_93484

/-- Represents the distance between two points A and B, given the conditions of the pedestrian problem. -/
def distance_AB : ℝ :=
  let v1 : ℝ := 4  -- Speed of the first pedestrian in km/hr
  let v2 : ℝ := 3  -- Speed of the second pedestrian in km/hr
  let t_meet : ℝ := 1.5  -- Time until meeting in hours
  let d1_before : ℝ := v1 * t_meet  -- Distance covered by first pedestrian before meeting
  let d2_before : ℝ := v2 * t_meet  -- Distance covered by second pedestrian before meeting
  let d1_after : ℝ := v1 * 0.75  -- Distance covered by first pedestrian after meeting
  let d2_after : ℝ := v2 * (4/3)  -- Distance covered by second pedestrian after meeting
  d1_before + d2_before  -- Total distance

/-- Theorem stating that the distance between points A and B is 7 km, given the conditions of the pedestrian problem. -/
theorem distance_AB_is_7 : distance_AB = 7 := by
  sorry  -- Proof is omitted as per instructions

#eval distance_AB  -- This will evaluate to 7

end NUMINAMATH_CALUDE_distance_AB_is_7_l934_93484


namespace NUMINAMATH_CALUDE_arrangements_with_separation_l934_93489

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the problem. -/
def numberOfPeople : ℕ := 5

/-- Theorem stating that the number of arrangements with at least one person between A and B is 72. -/
theorem arrangements_with_separation :
  totalArrangements numberOfPeople - adjacentArrangements numberOfPeople = 72 := by
  sorry

#eval totalArrangements numberOfPeople - adjacentArrangements numberOfPeople

end NUMINAMATH_CALUDE_arrangements_with_separation_l934_93489


namespace NUMINAMATH_CALUDE_decagon_adjacent_probability_l934_93425

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of ways to choose 2 distinct vertices from a decagon -/
def totalChoices (d : Decagon) : ℕ := (d.val.choose 2)

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacentChoices (d : Decagon) : ℕ := 2 * d.val

/-- The probability of choosing two adjacent vertices in a decagon -/
def adjacentProbability (d : Decagon) : ℚ :=
  (adjacentChoices d : ℚ) / (totalChoices d : ℚ)

theorem decagon_adjacent_probability (d : Decagon) :
  adjacentProbability d = 4/9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_probability_l934_93425


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l934_93402

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l934_93402


namespace NUMINAMATH_CALUDE_point_on_x_axis_l934_93485

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis -/
def xAxis : Set Point :=
  {p : Point | p.y = 0}

theorem point_on_x_axis (a : ℝ) :
  let P : Point := ⟨4, 2*a + 6⟩
  P ∈ xAxis → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l934_93485


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l934_93465

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l934_93465


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l934_93471

def S : Set (ℕ+ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n : ℕ+, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1 : ℝ) * f (2 * n)}

theorem smallest_upper_bound :
  ∃ M : ℕ+, (∀ f ∈ S, ∀ n : ℕ+, f n < M) ∧
  (∀ M' : ℕ+, M' < M → ∃ f ∈ S, ∃ n : ℕ+, f n ≥ M') :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l934_93471


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l934_93415

theorem triangle_abc_properties (a b c A B C : ℝ) : 
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b →
  a > b →
  b = Real.sqrt 13 →
  a + c = 4 →
  B = π / 6 ∧ 
  (1/2) * a * c * Real.sin B = (6 - 3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l934_93415


namespace NUMINAMATH_CALUDE_first_group_size_l934_93492

/-- Given a work that takes some men 80 days to complete and 20 men 32 days to complete,
    prove that the number of men in the first group is 8. -/
theorem first_group_size (work : ℕ) : ∃ (x : ℕ), x * 80 = 20 * 32 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_first_group_size_l934_93492


namespace NUMINAMATH_CALUDE_subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l934_93407

-- Statement 1
theorem subset_if_intersection_eq (A B : Set α) : A ∩ B = A → A ⊆ B := by sorry

-- Statement 2
theorem elem_of_union_if_elem_of_intersection {A B : Set α} {x : α} :
  x ∈ A ∩ B → x ∈ A ∪ B := by sorry

-- Statement 3
theorem fraction_inequality_necessary_not_sufficient {a b : ℝ} :
  (a < b ∧ b < 0) → b / a < a / b := by sorry

-- Statement 4
theorem exists_non_positive_square : ∃ x : ℤ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l934_93407


namespace NUMINAMATH_CALUDE_find_divisor_l934_93444

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 25 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l934_93444


namespace NUMINAMATH_CALUDE_non_negative_product_l934_93480

theorem non_negative_product (a b c d e f g h : ℝ) :
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_product_l934_93480


namespace NUMINAMATH_CALUDE_inequality_proof_l934_93453

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l934_93453


namespace NUMINAMATH_CALUDE_traveler_distance_l934_93452

/-- The straight-line distance from start to end point given net northward and westward distances -/
theorem traveler_distance (north west : ℝ) (h_north : north = 12) (h_west : west = 12) :
  Real.sqrt (north ^ 2 + west ^ 2) = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_traveler_distance_l934_93452


namespace NUMINAMATH_CALUDE_exists_point_on_line_with_sum_of_distances_l934_93498

-- Define the line l
variable (l : Line)

-- Define points A and B
variable (A B : Point)

-- Define the given segment length
variable (a : ℝ)

-- Define the property that A and B are on the same side of l
def sameSideOfLine (A B : Point) (l : Line) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define what it means for a point to be on a line
def onLine (P : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem exists_point_on_line_with_sum_of_distances
  (h_same_side : sameSideOfLine A B l) :
  ∃ M : Point, onLine M l ∧ distance M A + distance M B = a := sorry

end NUMINAMATH_CALUDE_exists_point_on_line_with_sum_of_distances_l934_93498


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_2999_l934_93451

theorem largest_prime_factor_of_2999 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2999 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2999 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_2999_l934_93451


namespace NUMINAMATH_CALUDE_m_range_theorem_l934_93478

/-- Proposition p: There exists x ∈ ℝ, such that 2x² + (m-1)x + 1/2 ≤ 0 -/
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

/-- Proposition q: The curve C₁: x²/m² + y²/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := m ≠ 0 ∧ 2 * m + 8 > 0 ∧ m^2 > 2 * m + 8

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (3 ≤ m ∧ m ≤ 4) ∨ (-2 ≤ m ∧ m ≤ -1) ∨ m ≤ -4

theorem m_range_theorem (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l934_93478


namespace NUMINAMATH_CALUDE_movie_date_communication_l934_93459

theorem movie_date_communication (p : ℝ) (h1 : p = 0.9) :
  p * p + (1 - p) * (1 - p) = 0.82 := by
  sorry

end NUMINAMATH_CALUDE_movie_date_communication_l934_93459


namespace NUMINAMATH_CALUDE_sophia_transactions_l934_93424

theorem sophia_transactions (mabel anthony cal jade sophia : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  sophia = jade + jade / 2 →
  sophia = 128 :=
by sorry

end NUMINAMATH_CALUDE_sophia_transactions_l934_93424


namespace NUMINAMATH_CALUDE_circle_constant_value_l934_93454

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 8*y + c = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop :=
  x = -2 ∧ y = -4

-- Define the radius of the circle
def circle_radius (r : ℝ) : Prop :=
  r = 5

-- Theorem statement
theorem circle_constant_value :
  ∀ (c : ℝ), 
  (∀ (x y : ℝ), circle_equation x y c → 
    ∃ (h k : ℝ), circle_center h k ∧ 
    ∃ (r : ℝ), circle_radius r ∧ 
    (x - h)^2 + (y - k)^2 = r^2) →
  c = -5 := by sorry

end NUMINAMATH_CALUDE_circle_constant_value_l934_93454


namespace NUMINAMATH_CALUDE_total_pears_l934_93414

-- Define the number of pears sold
def sold : ℕ := 20

-- Define the number of pears poached in terms of sold
def poached : ℕ := sold / 2

-- Define the number of pears canned in terms of poached
def canned : ℕ := poached + poached / 5

-- Theorem statement
theorem total_pears : sold + poached + canned = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l934_93414


namespace NUMINAMATH_CALUDE_fran_average_speed_l934_93487

/-- Calculates the average speed required for Fran to cover the same distance as Joann -/
theorem fran_average_speed (joann_speed1 joann_time1 joann_speed2 joann_time2 fran_time : ℝ) 
  (h1 : joann_speed1 = 15)
  (h2 : joann_time1 = 4)
  (h3 : joann_speed2 = 12)
  (h4 : joann_time2 = 0.5)
  (h5 : fran_time = 4) :
  (joann_speed1 * joann_time1 + joann_speed2 * joann_time2) / fran_time = 16.5 := by
  sorry

#check fran_average_speed

end NUMINAMATH_CALUDE_fran_average_speed_l934_93487


namespace NUMINAMATH_CALUDE_train_length_l934_93431

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 24 →
  train_speed = 50 →
  train_speed * crossing_time - bridge_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l934_93431


namespace NUMINAMATH_CALUDE_dinnerCostTheorem_l934_93445

/-- Represents the cost breakdown of a dinner -/
structure DinnerCost where
  preTax : ℝ
  taxRate : ℝ
  tipRate : ℝ
  total : ℝ

/-- The combined pre-tax cost of two dinners -/
def combinedPreTaxCost (d1 d2 : DinnerCost) : ℝ :=
  d1.preTax + d2.preTax

/-- Calculates the total cost of a dinner including tax and tip -/
def calculateTotal (d : DinnerCost) : ℝ :=
  d.preTax * (1 + d.taxRate + d.tipRate)

theorem dinnerCostTheorem (johnDinner sarahDinner : DinnerCost) :
  johnDinner.taxRate = 0.12 →
  johnDinner.tipRate = 0.16 →
  sarahDinner.taxRate = 0.09 →
  sarahDinner.tipRate = 0.10 →
  johnDinner.total = 35.20 →
  sarahDinner.total = 22.00 →
  calculateTotal johnDinner = johnDinner.total →
  calculateTotal sarahDinner = sarahDinner.total →
  combinedPreTaxCost johnDinner sarahDinner = 46 := by
  sorry

#eval 46  -- This line is added to ensure the statement can be built successfully

end NUMINAMATH_CALUDE_dinnerCostTheorem_l934_93445


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l934_93422

/-- Definition of the ⊕ operation for real numbers -/
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

/-- Theorem stating that 4 ⊕ 2 = 18 -/
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l934_93422


namespace NUMINAMATH_CALUDE_problem_statement_l934_93435

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem problem_statement :
  (∀ x : ℝ, f x > a^2) →
  (∀ m n : ℝ, m ∈ T → n ∈ T → Real.sqrt 3 * |m + n| < |m * n + 3|) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l934_93435


namespace NUMINAMATH_CALUDE_escalator_steps_l934_93464

theorem escalator_steps (n : ℕ) : 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 6 ∧ 
  n % 20 = 19 ∧ 
  n < 1000 → 
  n = 839 :=
by sorry

end NUMINAMATH_CALUDE_escalator_steps_l934_93464


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l934_93426

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {(0, 0), (1, 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l934_93426


namespace NUMINAMATH_CALUDE_two_digit_number_product_l934_93461

theorem two_digit_number_product (n : ℕ) (tens units : ℕ) : 
  n = tens * 10 + units →
  n = 24 →
  units = tens + 2 →
  n * (tens + units) = 144 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_product_l934_93461


namespace NUMINAMATH_CALUDE_max_value_of_function_l934_93472

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  ∃ (max_y : ℝ), max_y = 1/8 ∧ ∀ y, y = x * (1 - 2*x) → y ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l934_93472


namespace NUMINAMATH_CALUDE_liars_on_black_chairs_l934_93477

def room_scenario (total_people : ℕ) (initial_black_claims : ℕ) (final_white_claims : ℕ) : Prop :=
  -- Total number of people is positive
  total_people > 0 ∧
  -- Initially, all people claim to be on black chairs
  initial_black_claims = total_people ∧
  -- After rearrangement, some people claim to be on white chairs
  final_white_claims > 0 ∧ final_white_claims < total_people

theorem liars_on_black_chairs 
  (total_people : ℕ) 
  (initial_black_claims : ℕ) 
  (final_white_claims : ℕ) 
  (h : room_scenario total_people initial_black_claims final_white_claims) :
  -- The number of liars on black chairs after rearrangement
  (final_white_claims / 2) = 8 :=
sorry

end NUMINAMATH_CALUDE_liars_on_black_chairs_l934_93477
