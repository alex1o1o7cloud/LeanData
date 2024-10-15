import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l924_92478

def problem (m : ℝ) : Prop :=
  let a : Fin 2 → ℝ := ![m + 2, 1]
  let b : Fin 2 → ℝ := ![1, -2*m]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- a ⊥ b condition
  ‖(a 0 + b 0, a 1 + b 1)‖ = Real.sqrt 34

theorem problem_solution :
  ∃ m : ℝ, problem m := by sorry

end NUMINAMATH_CALUDE_problem_solution_l924_92478


namespace NUMINAMATH_CALUDE_floor_times_self_110_l924_92432

theorem floor_times_self_110 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 110 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_110_l924_92432


namespace NUMINAMATH_CALUDE_new_trailers_correct_l924_92431

/-- Represents the trailer park scenario -/
structure TrailerPark where
  initial_count : ℕ
  initial_avg_age : ℕ
  years_passed : ℕ
  current_avg_age : ℕ

/-- Calculates the number of new trailers added -/
def new_trailers (park : TrailerPark) : ℕ :=
  13

/-- Theorem stating that the calculated number of new trailers is correct -/
theorem new_trailers_correct (park : TrailerPark) 
  (h1 : park.initial_count = 30)
  (h2 : park.initial_avg_age = 10)
  (h3 : park.years_passed = 5)
  (h4 : park.current_avg_age = 12) :
  new_trailers park = 13 := by
  sorry

#check new_trailers_correct

end NUMINAMATH_CALUDE_new_trailers_correct_l924_92431


namespace NUMINAMATH_CALUDE_bicycle_race_finishers_l924_92497

theorem bicycle_race_finishers :
  let initial_racers : ℕ := 50
  let joined_racers : ℕ := 30
  let dropped_racers : ℕ := 30
  let racers_after_joining := initial_racers + joined_racers
  let racers_after_doubling := 2 * racers_after_joining
  let finishers := racers_after_doubling - dropped_racers
  finishers = 130 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_race_finishers_l924_92497


namespace NUMINAMATH_CALUDE_function_growth_l924_92413

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_growth (hf : ∀ x, f x < f' x) :
  (f 1 > Real.exp 1 * f 0) ∧ (f 2023 > Real.exp 2023 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l924_92413


namespace NUMINAMATH_CALUDE_z_less_than_y_l924_92474

/-- 
Given:
- w is 40% less than u, so w = 0.6u
- u is 40% less than y, so u = 0.6y
- z is greater than w by 50% of w, so z = 1.5w

Prove that z is 46% less than y, which means z = 0.54y
-/
theorem z_less_than_y (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by
  sorry

end NUMINAMATH_CALUDE_z_less_than_y_l924_92474


namespace NUMINAMATH_CALUDE_fishes_from_ontario_erie_l924_92481

/-- The number of fishes taken from Lake Huron and Michigan -/
def huron_michigan : ℕ := 30

/-- The number of fishes taken from Lake Superior -/
def superior : ℕ := 44

/-- The total number of fishes brought home -/
def total : ℕ := 97

/-- The number of fishes taken from Lake Ontario and Erie -/
def ontario_erie : ℕ := total - (huron_michigan + superior)

theorem fishes_from_ontario_erie : ontario_erie = 23 := by
  sorry

end NUMINAMATH_CALUDE_fishes_from_ontario_erie_l924_92481


namespace NUMINAMATH_CALUDE_cards_thrown_away_l924_92406

theorem cards_thrown_away (cards_per_deck : ℕ) (half_full_decks : ℕ) (full_decks : ℕ) (remaining_cards : ℕ) : 
  cards_per_deck = 52 →
  half_full_decks = 3 →
  full_decks = 3 →
  remaining_cards = 200 →
  (cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks) - remaining_cards = 34 :=
by sorry

end NUMINAMATH_CALUDE_cards_thrown_away_l924_92406


namespace NUMINAMATH_CALUDE_units_digit_G_100_l924_92408

-- Define G_n
def G (n : ℕ) : ℕ := 2^(5^n) + 1

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l924_92408


namespace NUMINAMATH_CALUDE_compound_interest_rate_l924_92417

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 3025) :
  r = 25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l924_92417


namespace NUMINAMATH_CALUDE_min_t_value_l924_92426

/-- Ellipse C with eccentricity sqrt(2)/2 passing through (1, sqrt(2)/2) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (2, 0)

/-- Line containing point P -/
def line_P (x y : ℝ) : Prop := x + y = 1

/-- Vector relation between OA, OB, and OP -/
def vector_relation (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * P.1 ∧ A.2 + B.2 = t * P.2

/-- Main theorem: Minimum value of t -/
theorem min_t_value :
  ∃ (t_min : ℝ), 
    (∀ (A B P : ℝ × ℝ) (t : ℝ),
      ellipse_C A.1 A.2 → 
      ellipse_C B.1 B.2 → 
      line_P P.1 P.2 →
      vector_relation A B P t →
      t ≥ t_min) ∧
    t_min = 2 - Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_t_value_l924_92426


namespace NUMINAMATH_CALUDE_mary_baking_cake_l924_92489

theorem mary_baking_cake (total_flour total_sugar remaining_flour_diff : ℕ) 
  (h1 : total_flour = 9)
  (h2 : total_sugar = 6)
  (h3 : remaining_flour_diff = 7) :
  total_sugar - (total_flour - remaining_flour_diff) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l924_92489


namespace NUMINAMATH_CALUDE_wayne_blocks_problem_l924_92469

theorem wayne_blocks_problem (initial_blocks : ℕ) (father_blocks : ℕ) : 
  initial_blocks = 9 →
  father_blocks = 6 →
  (3 * (initial_blocks + father_blocks)) - (initial_blocks + father_blocks) = 30 :=
by sorry

end NUMINAMATH_CALUDE_wayne_blocks_problem_l924_92469


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l924_92496

def die_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (a b : ℕ) : Prop := is_prime (a + b)

def favorable_outcomes : ℕ := 23

def total_outcomes : ℕ := die_sides * die_sides

theorem probability_prime_sum_two_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 23 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l924_92496


namespace NUMINAMATH_CALUDE_composite_product_ratio_l924_92464

def first_twelve_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21]

def product_first_six : ℕ := (first_twelve_composites.take 6).prod

def product_next_six : ℕ := (first_twelve_composites.drop 6).prod

theorem composite_product_ratio : 
  (product_first_six : ℚ) / product_next_six = 2 / 245 := by sorry

end NUMINAMATH_CALUDE_composite_product_ratio_l924_92464


namespace NUMINAMATH_CALUDE_min_packs_for_100_cans_l924_92473

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cans_in_pack (p : PackSize) : Nat :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 14
  | PackSize.large => 28

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def total_cans (c : PackCombination) : Nat :=
  c.small * cans_in_pack PackSize.small +
  c.medium * cans_in_pack PackSize.medium +
  c.large * cans_in_pack PackSize.large

/-- Calculates the total number of packs in a combination -/
def total_packs (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- Predicate to check if a combination is valid (exactly 100 cans) -/
def is_valid_combination (c : PackCombination) : Prop :=
  total_cans c = 100

/-- Theorem: The minimum number of packs to buy exactly 100 cans is 5 -/
theorem min_packs_for_100_cans :
  ∃ (c : PackCombination),
    is_valid_combination c ∧
    total_packs c = 5 ∧
    (∀ (c' : PackCombination), is_valid_combination c' → total_packs c' ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_packs_for_100_cans_l924_92473


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l924_92407

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + (1 : ℝ) / 18) = (Real.sqrt 26) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l924_92407


namespace NUMINAMATH_CALUDE_train_length_calculation_l924_92436

/-- The length of each train in meters -/
def train_length : ℝ := 79.92

/-- The speed of the faster train in km/hr -/
def faster_speed : ℝ := 52

/-- The speed of the slower train in km/hr -/
def slower_speed : ℝ := 36

/-- The time it takes for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := (faster_speed - slower_speed) * 1000 / 3600
  2 * train_length = relative_speed * passing_time := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l924_92436


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l924_92421

theorem angle_sum_is_pi_over_two (a b : ℝ) 
  (h_acute_a : 0 < a ∧ a < π / 2) 
  (h_acute_b : 0 < b ∧ b < π / 2)
  (h1 : 4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1)
  (h2 : 4 * Real.sin (2 * a) - 3 * Real.sin (2 * b) = 0) : 
  2 * a + b = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l924_92421


namespace NUMINAMATH_CALUDE_ababa_binary_bits_l924_92416

/-- The decimal representation of ABABA₁₆ -/
def ababa_decimal : ℕ := 701162

/-- The number of bits in the binary representation of ABABA₁₆ -/
def num_bits : ℕ := 20

theorem ababa_binary_bits :
  (2 ^ (num_bits - 1) : ℕ) ≤ ababa_decimal ∧ ababa_decimal < 2 ^ num_bits :=
by sorry

end NUMINAMATH_CALUDE_ababa_binary_bits_l924_92416


namespace NUMINAMATH_CALUDE_rectangular_field_length_l924_92420

theorem rectangular_field_length (width : ℝ) (length : ℝ) : 
  width = 13.5 → length = 2 * width - 3 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l924_92420


namespace NUMINAMATH_CALUDE_square_diagonal_triangle_l924_92445

theorem square_diagonal_triangle (s : ℝ) (h : s = 12) :
  let square_side := s
  let triangle_leg := s
  let triangle_hypotenuse := s * Real.sqrt 2
  let triangle_area := (s^2) / 2
  (triangle_leg = 12 ∧ 
   triangle_hypotenuse = 12 * Real.sqrt 2 ∧ 
   triangle_area = 72) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_triangle_l924_92445


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l924_92470

/-- Given a line passing through points (-6, 0) and (-3, 3) with direction vector (3, b), prove b = 3 -/
theorem direction_vector_b_value (b : ℝ) : b = 3 :=
  by
  -- Define the two points on the line
  let p1 : Fin 2 → ℝ := ![- 6, 0]
  let p2 : Fin 2 → ℝ := ![- 3, 3]
  
  -- Define the direction vector of the line
  let dir : Fin 2 → ℝ := ![3, b]
  
  -- Assert that the direction vector is parallel to the vector between the two points
  have h : ∃ (k : ℝ), k ≠ 0 ∧ (λ i => p2 i - p1 i) = (λ i => k * dir i) := by sorry
  
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l924_92470


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l924_92462

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (8 * x^2 + 10 * x - 16 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l924_92462


namespace NUMINAMATH_CALUDE_hearty_beads_count_l924_92441

/-- The number of beads Hearty has in total -/
def total_beads (blue_packages red_packages beads_per_package : ℕ) : ℕ :=
  (blue_packages + red_packages) * beads_per_package

/-- Proof that Hearty has 320 beads in total -/
theorem hearty_beads_count :
  total_beads 3 5 40 = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l924_92441


namespace NUMINAMATH_CALUDE_sqrt_inequality_l924_92460

theorem sqrt_inequality (a : ℝ) (h : a > 2) : Real.sqrt (a + 2) + Real.sqrt (a - 2) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l924_92460


namespace NUMINAMATH_CALUDE_natasha_maria_earnings_l924_92484

theorem natasha_maria_earnings (t : ℚ) : 
  (t - 4) * (3 * t - 4) = (3 * t - 12) * (t + 2) → t = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_natasha_maria_earnings_l924_92484


namespace NUMINAMATH_CALUDE_probability_at_least_one_success_l924_92429

theorem probability_at_least_one_success (p : ℝ) (n : ℕ) (h1 : p = 3/10) (h2 : n = 2) :
  1 - (1 - p)^n = 51/100 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_success_l924_92429


namespace NUMINAMATH_CALUDE_girls_without_pets_girls_without_pets_proof_l924_92480

theorem girls_without_pets (total_students : ℕ) (boys_fraction : ℚ) 
  (girls_with_dogs : ℚ) (girls_with_cats : ℚ) : ℕ :=
  let girls_fraction := 1 - boys_fraction
  let total_girls := (total_students : ℚ) * girls_fraction
  let girls_without_pets_fraction := 1 - girls_with_dogs - girls_with_cats
  let girls_without_pets := total_girls * girls_without_pets_fraction
  8

theorem girls_without_pets_proof :
  girls_without_pets 30 (1/3) (2/5) (1/5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_girls_without_pets_girls_without_pets_proof_l924_92480


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l924_92456

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l924_92456


namespace NUMINAMATH_CALUDE_triangle_side_length_l924_92468

/-- Given a triangle ABC with angle A = π/6, side a = 1, and side b = √3, 
    the length of side c is either 2 or 1. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  (c = 2 ∨ c = 1) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l924_92468


namespace NUMINAMATH_CALUDE_power_of_power_of_five_l924_92444

theorem power_of_power_of_five : (5^4)^2 = 390625 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_five_l924_92444


namespace NUMINAMATH_CALUDE_pyramid_height_l924_92453

theorem pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : base_perimeter = 40) (h_apex_dist : apex_to_vertex = 12) :
  let side_length := base_perimeter / 4
  let diagonal := side_length * Real.sqrt 2
  let half_diagonal := diagonal / 2
  Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2) = Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l924_92453


namespace NUMINAMATH_CALUDE_negation_of_tangent_equality_l924_92430

theorem negation_of_tangent_equality (x : ℝ) :
  (¬ ∀ x : ℝ, Real.tan (-x) = Real.tan x) ↔ (∃ x : ℝ, Real.tan (-x) ≠ Real.tan x) := by sorry

end NUMINAMATH_CALUDE_negation_of_tangent_equality_l924_92430


namespace NUMINAMATH_CALUDE_last_year_winner_ounces_l924_92493

/-- The amount of ounces in each hamburger -/
def hamburger_ounces : ℕ := 4

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat : ℕ := 22

/-- Theorem: The amount of ounces eaten by last year's winner is 88 -/
theorem last_year_winner_ounces : 
  hamburger_ounces * hamburgers_to_beat - hamburger_ounces = 88 := by
  sorry

end NUMINAMATH_CALUDE_last_year_winner_ounces_l924_92493


namespace NUMINAMATH_CALUDE_expected_value_ten_sided_die_l924_92422

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Finset ℕ := Finset.range 10

/-- The expected value of rolling the die -/
def ExpectedValue : ℚ := (Finset.sum TenSidedDie (λ i => i + 1)) / 10

/-- Theorem: The expected value of rolling a fair 10-sided die with faces numbered from 1 to 10 is 5.5 -/
theorem expected_value_ten_sided_die : ExpectedValue = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_ten_sided_die_l924_92422


namespace NUMINAMATH_CALUDE_f_composed_eq_6_has_three_solutions_l924_92476

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 7

-- Define the composite function f(f(x))
noncomputable def f_composed (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem f_composed_eq_6_has_three_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ f_composed x = 6 :=
sorry

end NUMINAMATH_CALUDE_f_composed_eq_6_has_three_solutions_l924_92476


namespace NUMINAMATH_CALUDE_daily_harvest_l924_92495

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

/-- Theorem stating that the total number of sacks harvested daily is 360 -/
theorem daily_harvest : total_sacks = 360 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l924_92495


namespace NUMINAMATH_CALUDE_lawn_width_is_60_l924_92446

/-- Represents a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  road_area : ℝ

/-- Theorem: The width of the lawn is 60 meters -/
theorem lawn_width_is_60 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.road_width = 10)
  (h3 : lawn.road_area = 1300)
  : lawn.width = 60 := by
  sorry

#check lawn_width_is_60

end NUMINAMATH_CALUDE_lawn_width_is_60_l924_92446


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l924_92435

/-- The focal length of a hyperbola with given properties -/
theorem hyperbola_focal_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := 2  -- eccentricity
  let d := Real.sqrt 3  -- distance from focus to asymptote
  2 * Real.sqrt (a^2 + b^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l924_92435


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l924_92438

theorem imaginary_part_of_z (z : ℂ) : z = (1 : ℂ) / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l924_92438


namespace NUMINAMATH_CALUDE_percentage_sum_proof_l924_92439

theorem percentage_sum_proof : 
  ∃ (x : ℝ), x * 400 + 0.45 * 250 = 224.5 ∧ x = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_proof_l924_92439


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l924_92419

theorem product_and_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 / y) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l924_92419


namespace NUMINAMATH_CALUDE_heartsuit_three_four_l924_92499

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_four_l924_92499


namespace NUMINAMATH_CALUDE_prime_solution_equation_l924_92414

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l924_92414


namespace NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l924_92483

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l924_92483


namespace NUMINAMATH_CALUDE_equation_solution_l924_92467

theorem equation_solution :
  let f (x : ℝ) := x^2 * (x - 2) - (4 * x^2 + 4)
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l924_92467


namespace NUMINAMATH_CALUDE_y_increases_with_x_on_positive_slope_line_l924_92466

/-- Given two points on a line with a positive slope, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is less than the y-coordinate of the second point. -/
theorem y_increases_with_x_on_positive_slope_line 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 3 * x₁ + 4) 
  (h2 : y₂ = 3 * x₂ + 4) 
  (h3 : x₁ < x₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_increases_with_x_on_positive_slope_line_l924_92466


namespace NUMINAMATH_CALUDE_point_A_coordinates_l924_92459

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line x + y + 3 = 0 -/
def line (p : Point) : Prop :=
  p.x + p.y + 3 = 0

/-- Two points are symmetric about a line if their midpoint lies on the line
    and the line is perpendicular to the line segment connecting the points -/
def symmetric_about (a b : Point) : Prop :=
  let midpoint : Point := ⟨(a.x + b.x) / 2, (a.y + b.y) / 2⟩
  line midpoint ∧ (a.y - b.y) = (a.x - b.x)

/-- The main theorem -/
theorem point_A_coordinates :
  ∀ (A : Point),
    symmetric_about A ⟨1, 2⟩ →
    A.x = -5 ∧ A.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l924_92459


namespace NUMINAMATH_CALUDE_negation_equivalence_l924_92472

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 2*x + x + 1 ≤ 0) ↔ (∀ x : ℤ, 2*x + x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l924_92472


namespace NUMINAMATH_CALUDE_income_ratio_uma_bala_l924_92443

theorem income_ratio_uma_bala (uma_income : ℕ) (uma_expenditure bala_expenditure : ℕ) 
  (h1 : uma_income = 16000)
  (h2 : uma_expenditure = 7 * bala_expenditure / 6)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000)
  : uma_income / (uma_income - 2000) = 8 / 7 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_uma_bala_l924_92443


namespace NUMINAMATH_CALUDE_serenity_shoes_pairs_serenity_bought_three_pairs_l924_92411

theorem serenity_shoes_pairs : ℕ → ℕ → ℕ → Prop :=
  fun total_shoes shoes_per_pair pairs_bought =>
    total_shoes = 6 ∧ shoes_per_pair = 2 →
    pairs_bought = total_shoes / shoes_per_pair ∧
    pairs_bought = 3

-- Proof
theorem serenity_bought_three_pairs : serenity_shoes_pairs 6 2 3 := by
  sorry

end NUMINAMATH_CALUDE_serenity_shoes_pairs_serenity_bought_three_pairs_l924_92411


namespace NUMINAMATH_CALUDE_cow_herd_division_l924_92457

theorem cow_herd_division (n : ℕ) : 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 5 : ℚ) + 7 = n → n = 140 := by
  sorry

end NUMINAMATH_CALUDE_cow_herd_division_l924_92457


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l924_92447

theorem greatest_divisor_with_remainders : Nat.gcd (3589 - 23) (5273 - 41) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l924_92447


namespace NUMINAMATH_CALUDE_tuesday_wednesday_most_available_l924_92440

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

-- Define the students
inductive Student
| alice
| bob
| cindy
| david
| eva

-- Define the availability function
def availability (s : Student) (d : Day) : Bool :=
  match s, d with
  | Student.alice, Day.monday => false
  | Student.alice, Day.tuesday => true
  | Student.alice, Day.wednesday => false
  | Student.alice, Day.thursday => true
  | Student.alice, Day.friday => true
  | Student.alice, Day.saturday => false
  | Student.bob, Day.monday => true
  | Student.bob, Day.tuesday => false
  | Student.bob, Day.wednesday => true
  | Student.bob, Day.thursday => false
  | Student.bob, Day.friday => false
  | Student.bob, Day.saturday => true
  | Student.cindy, Day.monday => false
  | Student.cindy, Day.tuesday => false
  | Student.cindy, Day.wednesday => true
  | Student.cindy, Day.thursday => false
  | Student.cindy, Day.friday => false
  | Student.cindy, Day.saturday => true
  | Student.david, Day.monday => true
  | Student.david, Day.tuesday => true
  | Student.david, Day.wednesday => false
  | Student.david, Day.thursday => false
  | Student.david, Day.friday => true
  | Student.david, Day.saturday => false
  | Student.eva, Day.monday => false
  | Student.eva, Day.tuesday => true
  | Student.eva, Day.wednesday => true
  | Student.eva, Day.thursday => true
  | Student.eva, Day.friday => false
  | Student.eva, Day.saturday => false

-- Count available students for a given day
def availableStudents (d : Day) : Nat :=
  (Student.alice :: Student.bob :: Student.cindy :: Student.david :: Student.eva :: []).filter (fun s => availability s d) |>.length

-- Theorem stating that Tuesday and Wednesday have the most available students
theorem tuesday_wednesday_most_available :
  (availableStudents Day.tuesday = availableStudents Day.wednesday) ∧
  (∀ d : Day, availableStudents d ≤ availableStudents Day.tuesday) :=
by sorry

end NUMINAMATH_CALUDE_tuesday_wednesday_most_available_l924_92440


namespace NUMINAMATH_CALUDE_lending_years_calculation_l924_92455

/-- Proves that the number of years the first part is lent is 5 -/
theorem lending_years_calculation (total_sum : ℝ) (second_part : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (second_years : ℕ) :
  total_sum = 2665 →
  second_part = 1332.5 →
  first_rate = 0.03 →
  second_rate = 0.05 →
  second_years = 3 →
  let first_part := total_sum - second_part
  let first_interest := first_part * first_rate
  let second_interest := second_part * second_rate * second_years
  first_interest * (5 : ℝ) = second_interest :=
by sorry

end NUMINAMATH_CALUDE_lending_years_calculation_l924_92455


namespace NUMINAMATH_CALUDE_ac_cube_l924_92454

theorem ac_cube (a b c : ℝ) (h1 : a * b = 1) (h2 : b + c = 0) : (a * c)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ac_cube_l924_92454


namespace NUMINAMATH_CALUDE_cost_calculation_l924_92452

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 20.50

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  ∀ mango_cost rice_cost : ℝ,
  mango_rice_relation mango_cost rice_cost →
  flour_rice_relation rice_cost →
  total_cost mango_cost rice_cost = 311.60 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l924_92452


namespace NUMINAMATH_CALUDE_quadratic_inequality_l924_92498

theorem quadratic_inequality (x : ℝ) : 
  (2 * x^2 - 5 * x - 12 > 0 ↔ x < -3/2 ∨ x > 4) ∧
  (2 * x^2 - 5 * x - 12 < 0 ↔ -3/2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l924_92498


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l924_92442

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_theorem (a b c : ℝ) :
  (f a b c 0 = 0) →
  (∀ x, f a b c (x + 1) = f a b c x + x + 1) →
  (∀ x, f a b c x = x^2 / 2 + x / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l924_92442


namespace NUMINAMATH_CALUDE_min_sum_cotangents_l924_92400

theorem min_sum_cotangents (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
  (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
  (hSum : A + B + C = π) (hSin : 2 * Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2) : 
  (∀ A' B' C', A' + B' + C' = π → 2 * Real.sin A' ^ 2 + Real.sin B' ^ 2 = 2 * Real.sin C' ^ 2 →
    1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C ≤ 
    1 / Real.tan A' + 1 / Real.tan B' + 1 / Real.tan C') ∧
  1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C = Real.sqrt 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_cotangents_l924_92400


namespace NUMINAMATH_CALUDE_pencil_cost_l924_92448

/-- Given a pen and a pencil where the pen costs half the price of the pencil,
    and their total cost is $12, prove that the pencil costs $8. -/
theorem pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = pencil_cost / 2 →
  pen_cost + pencil_cost = 12 →
  pencil_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_l924_92448


namespace NUMINAMATH_CALUDE_simplify_like_terms_l924_92437

theorem simplify_like_terms (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l924_92437


namespace NUMINAMATH_CALUDE_mono_properties_l924_92427

/-- Represents a monomial with coefficient and variables --/
structure Monomial where
  coeff : ℤ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial --/
def degree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -4mn^5 --/
def mono : Monomial :=
  { coeff := -4
    vars := [('m', 1), ('n', 5)] }

theorem mono_properties : (mono.coeff = -4) ∧ (degree mono = 6) := by
  sorry

end NUMINAMATH_CALUDE_mono_properties_l924_92427


namespace NUMINAMATH_CALUDE_train_problem_solution_l924_92405

/-- Represents the train problem scenario -/
structure TrainProblem where
  total_distance : ℝ
  train_a_speed : ℝ
  train_b_speed : ℝ
  separation_distance : ℝ

/-- The time when trains are at the given separation distance -/
def separation_time (p : TrainProblem) : Set ℝ :=
  { t : ℝ | t = (p.total_distance - p.separation_distance) / (p.train_a_speed + p.train_b_speed) ∨
             t = (p.total_distance + p.separation_distance) / (p.train_a_speed + p.train_b_speed) }

/-- Theorem stating the solution to the train problem -/
theorem train_problem_solution (p : TrainProblem) 
    (h1 : p.total_distance = 840)
    (h2 : p.train_a_speed = 68.5)
    (h3 : p.train_b_speed = 71.5)
    (h4 : p.separation_distance = 210) :
    separation_time p = {4.5, 7.5} := by
  sorry

end NUMINAMATH_CALUDE_train_problem_solution_l924_92405


namespace NUMINAMATH_CALUDE_garage_sale_books_sold_l924_92494

def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

theorem garage_sale_books_sold :
  let initial_books : ℕ := 108
  let remaining_books : ℕ := 66
  books_sold initial_books remaining_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_books_sold_l924_92494


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l924_92491

theorem quadratic_equation_c_value (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = (-8 + Real.sqrt 20) / 4 ∨ x = (-8 - Real.sqrt 20) / 4) →
  c = 5.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l924_92491


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l924_92477

theorem not_divisible_by_169 (x : ℤ) : ¬(169 ∣ (x^2 + 5*x + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l924_92477


namespace NUMINAMATH_CALUDE_statement_equivalence_l924_92492

theorem statement_equivalence (P Q R : Prop) :
  (P → (Q ∧ ¬R)) ↔ ((¬Q ∨ R) → ¬P) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l924_92492


namespace NUMINAMATH_CALUDE_janet_dermatologist_distance_l924_92471

def dermatologist_distance (x : ℝ) := x
def gynecologist_distance : ℝ := 50
def car_efficiency : ℝ := 20
def gas_used : ℝ := 8

theorem janet_dermatologist_distance :
  ∃ x : ℝ, 
    dermatologist_distance x = 30 ∧ 
    2 * dermatologist_distance x + 2 * gynecologist_distance = car_efficiency * gas_used :=
by sorry

end NUMINAMATH_CALUDE_janet_dermatologist_distance_l924_92471


namespace NUMINAMATH_CALUDE_streetlights_per_square_l924_92475

theorem streetlights_per_square 
  (total_streetlights : ℕ) 
  (num_squares : ℕ) 
  (unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : num_squares = 15) 
  (h3 : unused_streetlights = 20) : 
  (total_streetlights - unused_streetlights) / num_squares = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_streetlights_per_square_l924_92475


namespace NUMINAMATH_CALUDE_max_value_expression_l924_92404

theorem max_value_expression (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x ≤ 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ∧
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x = 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ↔ x = Real.sqrt a :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l924_92404


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l924_92482

/-- Calculate the number of toothpicks needed for a rectangular grid -/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A rectangular grid with length 20 and width 10 requires 430 toothpicks -/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l924_92482


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l924_92418

-- Define a function f over the real numbers
variable (f : ℝ → ℝ)

-- Define the reflection operation
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Theorem statement
theorem reflection_across_x_axis (x y : ℝ) :
  (y = f x) ↔ (-y = (reflect f) x) :=
sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l924_92418


namespace NUMINAMATH_CALUDE_cos_2theta_minus_15_deg_l924_92461

theorem cos_2theta_minus_15_deg (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ + π / 12) = 4 / 5) : 
  Real.cos (2 * θ - π / 12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_minus_15_deg_l924_92461


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l924_92434

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → 
  (n + 2) + (n + 3) + (n + 4) = 66 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l924_92434


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l924_92428

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d, 
    its area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l924_92428


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l924_92415

/-- Represents the price of type A Kiwi in yuan -/
def a : ℝ := 35

/-- Represents the price of type B Kiwi in yuan -/
def b : ℝ := 50

/-- The cost of 2 type A and 1 type B Kiwi is 120 yuan -/
axiom cost_equation_1 : 2 * a + b = 120

/-- The cost of 3 type A and 2 type B Kiwi is 205 yuan -/
axiom cost_equation_2 : 3 * a + 2 * b = 205

/-- The cost price of each type B Kiwi is 40 yuan -/
def cost_B : ℝ := 40

/-- Daily sales of type B Kiwi at price b -/
def initial_sales : ℝ := 100

/-- Decrease in sales for each yuan increase in price -/
def sales_decrease : ℝ := 5

/-- Daily profit function for type B Kiwi -/
def profit (x : ℝ) : ℝ := (x - cost_B) * (initial_sales - sales_decrease * (x - b))

/-- The optimal selling price for type B Kiwi -/
def optimal_price : ℝ := 55

/-- The maximum daily profit for type B Kiwi -/
def max_profit : ℝ := 1125

/-- Theorem stating that the optimal price maximizes the profit -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit x ≤ profit optimal_price ∧ profit optimal_price = max_profit :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l924_92415


namespace NUMINAMATH_CALUDE_line_equation_l924_92409

/-- Given a line with inclination angle π/3 and y-intercept 2, its equation is √3x - y + 2 = 0 -/
theorem line_equation (x y : ℝ) :
  let angle : ℝ := π / 3
  let y_intercept : ℝ := 2
  (Real.sqrt 3 * x - y + y_intercept = 0) ↔ 
    (y = Real.tan angle * x + y_intercept) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l924_92409


namespace NUMINAMATH_CALUDE_midpoint_theorem_ap_twice_pb_theorem_l924_92403

-- Define the line and points
def Line := ℝ → ℝ → Prop
def Point := ℝ × ℝ

-- Define the given point P
def P : Point := (-3, 1)

-- Define the properties of points A and B
def on_x_axis (A : Point) : Prop := A.2 = 0
def on_y_axis (B : Point) : Prop := B.1 = 0

-- Define the property of P being the midpoint of AB
def is_midpoint (P A B : Point) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define the property of AP = 2PB
def ap_twice_pb (P A B : Point) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Define the equations of the lines
def line_eq1 (x y : ℝ) : Prop := x - 3*y + 6 = 0
def line_eq2 (x y : ℝ) : Prop := x - 6*y + 9 = 0

-- Theorem 1
theorem midpoint_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → is_midpoint P A B →
  (∀ x y, l x y ↔ line_eq1 x y) :=
sorry

-- Theorem 2
theorem ap_twice_pb_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → ap_twice_pb P A B →
  (∀ x y, l x y ↔ line_eq2 x y) :=
sorry

end NUMINAMATH_CALUDE_midpoint_theorem_ap_twice_pb_theorem_l924_92403


namespace NUMINAMATH_CALUDE_cut_polygon_perimeter_decrease_l924_92479

/-- Represents a polygon -/
structure Polygon where
  perimeter : ℝ
  perim_pos : perimeter > 0

/-- Represents the result of cutting a polygon with a straight line -/
structure CutPolygon where
  original : Polygon
  part1 : Polygon
  part2 : Polygon

/-- Theorem: The perimeter of each part resulting from cutting a polygon
    with a straight line is less than the perimeter of the original polygon -/
theorem cut_polygon_perimeter_decrease (cp : CutPolygon) :
  cp.part1.perimeter < cp.original.perimeter ∧
  cp.part2.perimeter < cp.original.perimeter := by
  sorry

end NUMINAMATH_CALUDE_cut_polygon_perimeter_decrease_l924_92479


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l924_92425

/-- Given a function f(x) = ax² + b/x where a > 0 and b > 0, 
    and its tangent line at x = 1 passes through (3/2, 1/2),
    prove that the minimum value of 1/a + 1/b is 9 -/
theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => a * x^2 + b / x
  let f' := fun x => 2 * a * x - b / x^2
  let tangent_slope := f' 1
  let tangent_point := (1, f 1)
  (tangent_slope * (3/2 - 1) = 1/2 - f 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 9) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 1/b' = 9) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l924_92425


namespace NUMINAMATH_CALUDE_ribbon_division_theorem_l924_92458

theorem ribbon_division_theorem (p q r s : ℝ) :
  p + q + r + s = 36 →
  (p + q) / 2 + (r + s) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_theorem_l924_92458


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l924_92486

theorem at_least_one_non_negative (x : ℝ) : 
  let m := x^2 - 1
  let n := 2*x + 2
  max m n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l924_92486


namespace NUMINAMATH_CALUDE_certain_number_problem_l924_92490

theorem certain_number_problem : 
  ∃ N : ℕ, (N / 5 + N + 5 = 65) ∧ (N = 50) :=
by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l924_92490


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l924_92465

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}

theorem complement_of_A_in_U :
  Set.compl A = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l924_92465


namespace NUMINAMATH_CALUDE_leonard_younger_than_nina_l924_92433

/-- Given the ages of Leonard, Nina, and Jerome, prove that Leonard is 4 years younger than Nina. -/
theorem leonard_younger_than_nina :
  ∀ (leonard nina jerome : ℕ),
    leonard = 6 →
    nina = jerome / 2 →
    leonard + nina + jerome = 36 →
    nina - leonard = 4 :=
by sorry

end NUMINAMATH_CALUDE_leonard_younger_than_nina_l924_92433


namespace NUMINAMATH_CALUDE_divisibility_by_three_l924_92463

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l924_92463


namespace NUMINAMATH_CALUDE_min_Q_value_l924_92401

def is_special_number (m : ℕ) : Prop :=
  m ≥ 10 ∧ m < 100 ∧ (m / 10) ≠ (m % 10) ∧ (m / 10) ≠ 0 ∧ (m % 10) ≠ 0

def F (m : ℕ) : ℤ :=
  let m₁ := (m % 10) * 10 + (m / 10)
  (m * 100 + m₁ - (m₁ * 100 + m)) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s : ℚ) / s

theorem min_Q_value (s t : ℕ) (a b x y : ℕ) :
  is_special_number s →
  is_special_number t →
  s = 10 * a + b →
  t = 10 * x + y →
  1 ≤ b →
  b < a →
  a ≤ 7 →
  1 ≤ x →
  x ≤ 8 →
  1 ≤ y →
  y ≤ 8 →
  F s % 5 = 1 →
  F t - F s + 18 * x = 36 →
  ∀ (s' t' : ℕ), is_special_number s' → is_special_number t' → Q s' t' ≥ Q s t →
  Q s t = -42 / 73 :=
sorry

end NUMINAMATH_CALUDE_min_Q_value_l924_92401


namespace NUMINAMATH_CALUDE_effective_average_reduction_l924_92402

theorem effective_average_reduction (initial_price : ℝ) (reduction_percent : ℝ) : 
  reduction_percent = 36 → 
  ∃ (effective_reduction : ℝ), 
    (1 - effective_reduction / 100)^2 * initial_price = 
    (1 - reduction_percent / 100)^2 * initial_price ∧
    effective_reduction = 20 := by
  sorry

end NUMINAMATH_CALUDE_effective_average_reduction_l924_92402


namespace NUMINAMATH_CALUDE_linear_increasing_positive_slope_l924_92451

def f (k : ℝ) (x : ℝ) : ℝ := k * x - 100

theorem linear_increasing_positive_slope (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y, x < y → f k x < f k y) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_increasing_positive_slope_l924_92451


namespace NUMINAMATH_CALUDE_andrews_apples_l924_92412

theorem andrews_apples (n : ℕ) : 
  (6 * n = 5 * (n + 2)) → (6 * n = 60) := by
  sorry

end NUMINAMATH_CALUDE_andrews_apples_l924_92412


namespace NUMINAMATH_CALUDE_special_calculator_problem_l924_92488

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Applies the calculator's operation to a two-digit number -/
def calculator_operation (x : ℕ) : ℕ :=
  reverse_digits (2 * x) + 2

theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 → calculator_operation x = 27 → x = 26 := by
sorry

end NUMINAMATH_CALUDE_special_calculator_problem_l924_92488


namespace NUMINAMATH_CALUDE_triangle_properties_l924_92487

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following properties when certain conditions are met. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a - b = 2 →
  c = 4 →
  Real.sin A = 2 * Real.sin B →
  (a = 4 ∧ b = 2 ∧ Real.cos B = 7/8) ∧
  Real.sin (2*B - π/6) = (21 * Real.sqrt 5 - 17) / 64 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l924_92487


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l924_92410

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 80000 → 
  (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 33) % 7 = 0 → 
  n ≤ 79993 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l924_92410


namespace NUMINAMATH_CALUDE_green_balls_count_l924_92485

/-- Represents the contents and properties of a bag of colored balls. -/
structure BagOfBalls where
  total : Nat
  white : Nat
  yellow : Nat
  red : Nat
  purple : Nat
  prob_not_red_purple : Rat

/-- Calculates the number of green balls in the bag. -/
def green_balls (bag : BagOfBalls) : Nat :=
  bag.total - bag.white - bag.yellow - bag.red - bag.purple

/-- Theorem stating the number of green balls in the specific bag described in the problem. -/
theorem green_balls_count (bag : BagOfBalls) 
  (h1 : bag.total = 60)
  (h2 : bag.white = 22)
  (h3 : bag.yellow = 5)
  (h4 : bag.red = 6)
  (h5 : bag.purple = 9)
  (h6 : bag.prob_not_red_purple = 3/4) :
  green_balls bag = 18 := by
  sorry

#eval green_balls { 
  total := 60, 
  white := 22, 
  yellow := 5, 
  red := 6, 
  purple := 9, 
  prob_not_red_purple := 3/4 
}

end NUMINAMATH_CALUDE_green_balls_count_l924_92485


namespace NUMINAMATH_CALUDE_find_a_l924_92449

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- State the theorem
theorem find_a : ∃ a : ℝ, (A a ∪ B a = {0, 1, 2, 4, 16}) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l924_92449


namespace NUMINAMATH_CALUDE_line_intersects_plane_l924_92450

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships between points, lines, and planes
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_intersects_plane (l : Line) (α : Plane) :
  (∃ p q : Point, on_line p l ∧ on_line q l ∧ in_plane p α ∧ ¬in_plane q α) →
  intersects l α :=
sorry

end NUMINAMATH_CALUDE_line_intersects_plane_l924_92450


namespace NUMINAMATH_CALUDE_inequality_proof_l924_92423

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l924_92423


namespace NUMINAMATH_CALUDE_jakes_initial_money_l924_92424

theorem jakes_initial_money (M : ℝ) : 
  (M - 2800 - (M - 2800) / 2) * 3 / 4 = 825 → M = 5000 := by
  sorry

end NUMINAMATH_CALUDE_jakes_initial_money_l924_92424
