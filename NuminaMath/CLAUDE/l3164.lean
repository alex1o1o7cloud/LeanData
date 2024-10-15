import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_rotation_common_volume_l3164_316420

theorem tetrahedron_rotation_common_volume
  (V : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  ∃ (common_volume : ℝ),
    common_volume = V * (1 + Real.tan (α/2)^2) / (1 + Real.tan (α/2))^2 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_rotation_common_volume_l3164_316420


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l3164_316433

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l3164_316433


namespace NUMINAMATH_CALUDE_probability_is_two_thirds_l3164_316484

/-- Given four evenly spaced points A, B, C, D on a number line with an interval of 1,
    this function calculates the probability that a randomly chosen point E on AD
    has a sum of distances to B and C less than 2. -/
def probability_sum_distances_less_than_two (A B C D : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the probability is 2/3 -/
theorem probability_is_two_thirds (A B C D : ℝ) 
  (h1 : B - A = 1) 
  (h2 : C - B = 1) 
  (h3 : D - C = 1) : 
  probability_sum_distances_less_than_two A B C D = 2/3 :=
sorry

end NUMINAMATH_CALUDE_probability_is_two_thirds_l3164_316484


namespace NUMINAMATH_CALUDE_triangle_area_l3164_316409

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (f A = 2) ∧
  (a = Real.sqrt 7) ∧
  (Real.sin B = 2 * Real.sin C) ∧
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (1/2 * a * b * Real.sin C) = 7 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3164_316409


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3164_316466

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3164_316466


namespace NUMINAMATH_CALUDE_yahs_to_bahs_l3164_316427

-- Define the units
variable (bah rah yah : ℕ → ℚ)

-- Define the conversion rates
axiom bah_to_rah : ∀ x, bah x = rah (2 * x)
axiom rah_to_yah : ∀ x, rah x = yah (2 * x)

-- State the theorem
theorem yahs_to_bahs : yah 1200 = bah 300 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_l3164_316427


namespace NUMINAMATH_CALUDE_unique_four_digit_square_repeated_digits_l3164_316442

-- Define a four-digit number with repeated digits
def fourDigitRepeated (x y : Nat) : Nat :=
  1100 * x + 11 * y

-- Theorem statement
theorem unique_four_digit_square_repeated_digits :
  ∃! n : Nat, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (∃ x y : Nat, n = fourDigitRepeated x y) ∧  -- repeated digits
    (∃ m : Nat, n = m ^ 2) ∧  -- perfect square
    n = 7744 := by
  sorry


end NUMINAMATH_CALUDE_unique_four_digit_square_repeated_digits_l3164_316442


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3164_316404

theorem shaded_area_fraction (a r : ℝ) (h1 : a = 1/4) (h2 : r = 1/16) :
  let S := a / (1 - r)
  S = 4/15 := by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3164_316404


namespace NUMINAMATH_CALUDE_parabola_equation_l3164_316460

/-- A parabola with specified properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : a ≠ 0 → b = -4 * a
  tangent_line : ∃ x : ℝ, a * x^2 + b * x + c = 2 * x + 1 ∧
                 ∀ y : ℝ, y ≠ x → a * y^2 + b * y + c > 2 * y + 1
  y_intercepts : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                 a * x₁^2 + b * x₁ + c = 0 ∧
                 a * x₂^2 + b * x₂ + c = 0 ∧
                 (x₁ - x₂)^2 = 8

/-- The parabola equation is one of the two specified forms -/
theorem parabola_equation (p : Parabola) : 
  (p.a = 1 ∧ p.b = 4 ∧ p.c = 2) ∨ (p.a = 1/2 ∧ p.b = 2 ∧ p.c = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3164_316460


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l3164_316444

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l3164_316444


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3164_316429

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances := by
  sorry

#check player_positions_satisfy_distances

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3164_316429


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l3164_316489

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_slope_implies_a (a : ℝ) :
  curve_derivative a 1 = 2 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l3164_316489


namespace NUMINAMATH_CALUDE_fraction_relations_l3164_316400

theorem fraction_relations (x y : ℚ) (h : x / y = 2 / 5) :
  (x + y) / y = 7 / 5 ∧ 
  y / (y - x) = 5 / 3 ∧ 
  x / (3 * y) = 2 / 15 ∧ 
  (x + 3 * y) / x ≠ 17 / 2 ∧ 
  (x - y) / y ≠ 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relations_l3164_316400


namespace NUMINAMATH_CALUDE_line_above_function_l3164_316414

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1/a) - a*x

theorem line_above_function (a : ℝ) (h : a ≠ 0) :
  (∀ x, a*x > f a x) ↔ a > Real.exp 1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_above_function_l3164_316414


namespace NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3164_316428

/-- Represents the cookie exchange system set up by Patty --/
structure CookieExchange where
  cookie_per_chore : ℕ
  chores_per_week : ℕ
  money_available : ℕ
  cookies_per_pack : ℕ
  cost_per_pack : ℕ

/-- Calculates the number of weeks Patty can avoid chores --/
def weeks_without_chores (ce : CookieExchange) : ℕ :=
  let packs_bought := ce.money_available / ce.cost_per_pack
  let total_cookies := packs_bought * ce.cookies_per_pack
  let cookies_per_week := ce.cookie_per_chore * ce.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that Patty can avoid chores for 10 weeks --/
theorem patty_avoids_chores_for_ten_weeks :
  let ce : CookieExchange := {
    cookie_per_chore := 3,
    chores_per_week := 4,
    money_available := 15,
    cookies_per_pack := 24,
    cost_per_pack := 3
  }
  weeks_without_chores ce = 10 := by
  sorry

end NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3164_316428


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3164_316497

/-- Given a hyperbola with eccentricity 2 and the same foci as the ellipse x²/25 + y²/9 = 1,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2/25 + y^2/9 = 1 → 
      ∃ (c : ℝ), c = 4 ∧ 
        (∀ (x y : ℝ), (x + c)^2 + y^2 = 25 ∨ (x - c)^2 + y^2 = 25)) ∧
    (∃ (c : ℝ), c/a = 2 ∧ c = 4)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3164_316497


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3164_316436

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y + 1 / x + y ≥ y / x + 1 / y + x ↔ (x - y) * (x - 1) * (1 - y) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3164_316436


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3164_316424

theorem trigonometric_identities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (((2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) = 5) ∧
  ((Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3)) = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3164_316424


namespace NUMINAMATH_CALUDE_red_light_probability_l3164_316487

theorem red_light_probability (p : ℝ) (h1 : p = 1 / 3) :
  let probability_green := 1 - p
  let probability_red := p
  let probability_first_red_at_second := probability_green * probability_red
  probability_first_red_at_second = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_red_light_probability_l3164_316487


namespace NUMINAMATH_CALUDE_factorization_200_perfect_square_factors_200_l3164_316467

/-- A function that returns the number of positive factors of n that are perfect squares -/
def perfect_square_factors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 200 is 2^3 * 5^2 -/
theorem factorization_200 : 200 = 2^3 * 5^2 := sorry

/-- Theorem stating that the number of positive factors of 200 that are perfect squares is 4 -/
theorem perfect_square_factors_200 : perfect_square_factors 200 = 4 := by sorry

end NUMINAMATH_CALUDE_factorization_200_perfect_square_factors_200_l3164_316467


namespace NUMINAMATH_CALUDE_smores_group_size_l3164_316407

/-- Given the conditions for S'mores supplies, prove the number of people in the group. -/
theorem smores_group_size :
  ∀ (smores_per_person : ℕ) 
    (cost_per_set : ℕ) 
    (smores_per_set : ℕ) 
    (total_cost : ℕ),
  smores_per_person = 3 →
  cost_per_set = 3 →
  smores_per_set = 4 →
  total_cost = 18 →
  (total_cost / cost_per_set) * smores_per_set / smores_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_smores_group_size_l3164_316407


namespace NUMINAMATH_CALUDE_derivative_of_2_sqrt_x_cubed_l3164_316464

theorem derivative_of_2_sqrt_x_cubed (x : ℝ) (h : x > 0) :
  deriv (λ x => 2 * Real.sqrt (x^3)) x = 3 * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_2_sqrt_x_cubed_l3164_316464


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3164_316471

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), c = -1/12 ∧ d = 7/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3164_316471


namespace NUMINAMATH_CALUDE_fraction_comparison_l3164_316453

theorem fraction_comparison (x y : ℝ) (h1 : x > y) (h2 : y > 2) :
  y / (y^2 - y + 1) > x / (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3164_316453


namespace NUMINAMATH_CALUDE_tile_square_side_length_l3164_316454

/-- Given tiles with width 16 and length 24, proves that the side length of a square
    formed by a minimum of 6 tiles is 48. -/
theorem tile_square_side_length
  (tile_width : ℕ) (tile_length : ℕ) (min_tiles : ℕ)
  (hw : tile_width = 16)
  (hl : tile_length = 24)
  (hm : min_tiles = 6) :
  2 * tile_length = 3 * tile_width ∧ 2 * tile_length = 48 := by
  sorry

#check tile_square_side_length

end NUMINAMATH_CALUDE_tile_square_side_length_l3164_316454


namespace NUMINAMATH_CALUDE_barrel_leak_percentage_l3164_316486

theorem barrel_leak_percentage (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 220 →
  remaining_volume = 198 →
  (initial_volume - remaining_volume) / initial_volume * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_barrel_leak_percentage_l3164_316486


namespace NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l3164_316473

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The proposition that the line y = a intersects the graph of f(x) at three distinct points -/
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

/-- The theorem stating that the line y = a intersects the graph of f(x) at three distinct points
    if and only if a is in the open interval (-2, 2) -/
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_distinct_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l3164_316473


namespace NUMINAMATH_CALUDE_binomial_600_0_l3164_316440

theorem binomial_600_0 : (600 : ℕ).choose 0 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_600_0_l3164_316440


namespace NUMINAMATH_CALUDE_philips_weekly_mileage_l3164_316418

/-- Calculate Philip's car's mileage for a typical week -/
theorem philips_weekly_mileage (school_distance : ℝ) (market_distance : ℝ)
  (school_trips_per_day : ℕ) (school_days_per_week : ℕ) (market_trips_per_week : ℕ)
  (h1 : school_distance = 2.5)
  (h2 : market_distance = 2)
  (h3 : school_trips_per_day = 2)
  (h4 : school_days_per_week = 4)
  (h5 : market_trips_per_week = 1) :
  school_distance * 2 * ↑school_trips_per_day * ↑school_days_per_week +
  market_distance * 2 * ↑market_trips_per_week = 44 := by
  sorry

#check philips_weekly_mileage

end NUMINAMATH_CALUDE_philips_weekly_mileage_l3164_316418


namespace NUMINAMATH_CALUDE_mike_total_spent_l3164_316422

def trumpet_cost : ℚ := 145.16
def songbook_cost : ℚ := 5.84

theorem mike_total_spent :
  trumpet_cost + songbook_cost = 151 := by sorry

end NUMINAMATH_CALUDE_mike_total_spent_l3164_316422


namespace NUMINAMATH_CALUDE_train_length_calculation_l3164_316496

-- Define the given constants
def train_speed : Real := 72 -- km/hr
def bridge_length : Real := 142 -- meters
def crossing_time : Real := 12.598992080633549 -- seconds

-- Define the theorem
theorem train_length_calculation :
  let speed_ms : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 110 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3164_316496


namespace NUMINAMATH_CALUDE_impossible_three_similar_parts_l3164_316441

theorem impossible_three_similar_parts : 
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by sorry

end NUMINAMATH_CALUDE_impossible_three_similar_parts_l3164_316441


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3164_316419

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) (h : IsGeometricSequence a q) 
  (h_eq : 16 * a 6 = a 2) :
  q = 1/2 ∨ q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3164_316419


namespace NUMINAMATH_CALUDE_alice_bob_distance_difference_l3164_316476

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed_a : ℝ) (speed_b : ℝ) (time : ℝ) : ℝ :=
  (speed_a - speed_b) * time

/-- Theorem: Alice bikes 30 miles more than Bob after 6 hours -/
theorem alice_bob_distance_difference :
  distance_difference 15 10 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_distance_difference_l3164_316476


namespace NUMINAMATH_CALUDE_probability_at_least_one_tenth_grade_l3164_316438

/-- The number of volunteers from the 10th grade -/
def tenth_grade_volunteers : ℕ := 2

/-- The number of volunteers from the 11th grade -/
def eleventh_grade_volunteers : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := tenth_grade_volunteers + eleventh_grade_volunteers

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 2

/-- The probability of selecting at least one volunteer from the 10th grade -/
theorem probability_at_least_one_tenth_grade :
  (1 : ℚ) - (Nat.choose eleventh_grade_volunteers selected_volunteers : ℚ) / 
  (Nat.choose total_volunteers selected_volunteers : ℚ) = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_tenth_grade_l3164_316438


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l3164_316430

/-- The number of distinct sequences possible when flipping a coin n times -/
def coinFlipSequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences possible when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coinFlipSequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l3164_316430


namespace NUMINAMATH_CALUDE_skylar_donation_l3164_316474

/-- Calculates the total donation amount given starting age, current age, and annual donation amount. -/
def totalDonation (startAge currentAge annualDonation : ℕ) : ℕ :=
  (currentAge - startAge) * annualDonation

/-- Proves that Skylar's total donation is 100k -/
theorem skylar_donation :
  let startAge : ℕ := 13
  let currentAge : ℕ := 33
  let annualDonation : ℕ := 5000
  totalDonation startAge currentAge annualDonation = 100000 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_l3164_316474


namespace NUMINAMATH_CALUDE_prob_white_after_transfer_l3164_316458

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a white ball from a bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.black)

theorem prob_white_after_transfer : 
  let bag_a := Bag.mk 4 6
  let bag_b := Bag.mk 4 5
  let new_bag_b := Bag.mk (bag_b.white + 1) bag_b.black
  prob_white new_bag_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_after_transfer_l3164_316458


namespace NUMINAMATH_CALUDE_puzzle_solution_l3164_316401

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) Nat

-- Define the constraint for black dots (ratio of 2)
def blackDotConstraint (a b : Nat) : Prop := a = 2 * b ∨ b = 2 * a

-- Define the constraint for white dots (difference of 1)
def whiteDotConstraint (a b : Nat) : Prop := a = b + 1 ∨ b = a + 1

-- Define the property of having no repeated numbers in a row or column
def noRepeats (g : Grid) : Prop :=
  ∀ i j : Fin 6, i ≠ j → 
    (∀ k : Fin 6, g i k ≠ g j k) ∧ 
    (∀ k : Fin 6, g k i ≠ g k j)

-- Define the property that all numbers are between 1 and 6
def validNumbers (g : Grid) : Prop :=
  ∀ i j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

-- Define the specific constraints for this puzzle
def puzzleConstraints (g : Grid) : Prop :=
  blackDotConstraint (g 0 0) (g 0 1) ∧
  whiteDotConstraint (g 0 4) (g 0 5) ∧
  blackDotConstraint (g 1 2) (g 1 3) ∧
  whiteDotConstraint (g 2 1) (g 2 2) ∧
  blackDotConstraint (g 3 0) (g 3 1) ∧
  whiteDotConstraint (g 3 2) (g 3 3) ∧
  blackDotConstraint (g 4 4) (g 4 5) ∧
  whiteDotConstraint (g 5 3) (g 5 4)

-- Theorem statement
theorem puzzle_solution :
  ∀ g : Grid,
    noRepeats g →
    validNumbers g →
    puzzleConstraints g →
    g 3 0 = 2 ∧ g 3 1 = 1 ∧ g 3 2 = 4 ∧ g 3 3 = 3 ∧ g 3 4 = 6 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3164_316401


namespace NUMINAMATH_CALUDE_library_books_total_l3164_316437

theorem library_books_total (initial_books : ℕ) (additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
sorry

end NUMINAMATH_CALUDE_library_books_total_l3164_316437


namespace NUMINAMATH_CALUDE_temperature_80_degrees_l3164_316468

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- State the theorem
theorem temperature_80_degrees :
  ∃ t₁ t₂ : ℝ, 
    t₁ = 5 + 3 * Real.sqrt 5 ∧ 
    t₂ = 5 - 3 * Real.sqrt 5 ∧ 
    temperature t₁ = 80 ∧ 
    temperature t₂ = 80 ∧ 
    (∀ t : ℝ, temperature t = 80 → t = t₁ ∨ t = t₂) := by
  sorry

end NUMINAMATH_CALUDE_temperature_80_degrees_l3164_316468


namespace NUMINAMATH_CALUDE_vector_dot_product_l3164_316423

/-- Given two vectors a and b in ℝ², where a is parallel to (a + b), prove that their dot product is 4. -/
theorem vector_dot_product (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, x]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • (a + b)) → 
  (a • b = 4) := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3164_316423


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l3164_316495

theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (sheep_to_horse_ratio : ℚ) :
  sheep_count = 48 →
  total_horse_food = 12880 →
  sheep_to_horse_ratio = 6 / 7 →
  (total_horse_food / (sheep_count * (1 / sheep_to_horse_ratio))) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l3164_316495


namespace NUMINAMATH_CALUDE_chord_rotation_in_unit_circle_l3164_316478

/-- Chord rotation in a unit circle -/
theorem chord_rotation_in_unit_circle :
  -- Define the circle
  let circle_radius : ℝ := 1
  -- Define the chord length (side of inscribed equilateral triangle)
  let chord_length : ℝ := Real.sqrt 3
  -- Define the rotation angle (90 degrees in radians)
  let rotation_angle : ℝ := π / 2
  -- Define the area of the full circle
  let circle_area : ℝ := π * circle_radius ^ 2

  -- Statement 1: Area swept by chord during 90° rotation
  let area_swept : ℝ := (7 * π / 16) - 1 / 4

  -- Statement 2: Angle to sweep half of circle's area
  let angle_half_area : ℝ := (4 * π + 6 * Real.sqrt 3) / 9

  -- Prove the following:
  True →
    -- 1. The area swept by the chord during a 90° rotation
    (area_swept = (7 * π / 16) - 1 / 4) ∧
    -- 2. The angle required to sweep exactly half of the circle's area
    (angle_half_area = (4 * π + 6 * Real.sqrt 3) / 9) ∧
    -- Additional verification: the swept area at angle_half_area is indeed half the circle's area
    (2 * ((angle_half_area / (2 * π)) * circle_area - 
     (Real.sqrt (1 - (chord_length / 2) ^ 2) * (chord_length / 2))) = circle_area) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_rotation_in_unit_circle_l3164_316478


namespace NUMINAMATH_CALUDE_chord_length_line_circle_specific_chord_length_l3164_316492

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length_line_circle (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let circle := {(x, y) : ℝ × ℝ | (x - d)^2 + (y - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e - c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y = 7}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}
  let center := (2, 0)
  let radius := 2
  let dist_center_to_line := |3 * 2 + 4 * 0 - 7| / Real.sqrt (3^2 + 4^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_specific_chord_length_l3164_316492


namespace NUMINAMATH_CALUDE_episode_length_l3164_316421

def total_days : ℕ := 5
def episodes : ℕ := 20
def daily_hours : ℕ := 2
def minutes_per_hour : ℕ := 60

theorem episode_length :
  (total_days * daily_hours * minutes_per_hour) / episodes = 30 := by
  sorry

end NUMINAMATH_CALUDE_episode_length_l3164_316421


namespace NUMINAMATH_CALUDE_trig_identity_l3164_316434

theorem trig_identity (α : Real) (h : Real.tan (π + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α)^2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3164_316434


namespace NUMINAMATH_CALUDE_fair_game_conditions_l3164_316491

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  white : Nat
  green : Nat
  black : Nat

/-- Checks if the game is fair given the bag contents -/
def isFairGame (bag : BagContents) : Prop :=
  bag.green = bag.black

/-- Theorem stating the conditions for a fair game -/
theorem fair_game_conditions (x : Nat) :
  let bag : BagContents := {
    total := 15,
    white := x,
    green := 2 * x,
    black := 15 - x - 2 * x
  }
  isFairGame bag ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fair_game_conditions_l3164_316491


namespace NUMINAMATH_CALUDE_smallest_number_with_same_factors_l3164_316452

def alice_number : Nat := 30

-- Bob's number must have all prime factors of Alice's number
def has_all_prime_factors (m n : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → (p ∣ n → p ∣ m)

-- The theorem to prove
theorem smallest_number_with_same_factors (n : Nat) (h : n = alice_number) :
  ∃ m : Nat, has_all_prime_factors m n ∧ 
  (∀ k : Nat, has_all_prime_factors k n → m ≤ k) ∧
  m = n :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_factors_l3164_316452


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3164_316494

/-- Proves that the cost of an adult ticket is $9, given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_tickets : ℕ) :
  child_ticket_cost = 6 →
  total_tickets = 225 →
  total_revenue = 1875 →
  children_tickets = 50 →
  (total_revenue - child_ticket_cost * children_tickets) / (total_tickets - children_tickets) = 9 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3164_316494


namespace NUMINAMATH_CALUDE_domain_of_h_l3164_316465

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-10) 3

-- Define the function h
def h (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of h
def domain_h : Set ℝ := Set.Ici (10/3)

-- Theorem statement
theorem domain_of_h :
  ∀ x : ℝ, x ∈ domain_h ↔ -3 * x ∈ domain_f :=
sorry

end NUMINAMATH_CALUDE_domain_of_h_l3164_316465


namespace NUMINAMATH_CALUDE_fraction_power_multiplication_compute_fraction_power_l3164_316417

theorem fraction_power_multiplication (a b c : ℚ) (n : ℕ) :
  a * (b / c)^n = (a * b^n) / c^n :=
by sorry

theorem compute_fraction_power : 7 * (1 / 5)^3 = 7 / 125 :=
by sorry

end NUMINAMATH_CALUDE_fraction_power_multiplication_compute_fraction_power_l3164_316417


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_l3164_316415

theorem fourth_term_coefficient : 
  let a := (1/2 : ℚ)
  let b := (2/3 : ℚ)
  let n := 6
  let k := 4
  (n.choose (k-1)) * a^(n-(k-1)) * b^(k-1) = 20 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_l3164_316415


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l3164_316405

def g_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → g x + g (1 / x) = x^2

theorem largest_domain_of_g :
  ∃! (S : Set ℝ), S.Nonempty ∧
    (∀ T : Set ℝ, (∃ g : ℝ → ℝ, (∀ x ∈ T, x ≠ 0 ∧ g_condition g) → T ⊆ S)) ∧
    S = {-1, 1} :=
  sorry

end NUMINAMATH_CALUDE_largest_domain_of_g_l3164_316405


namespace NUMINAMATH_CALUDE_potato_difference_l3164_316416

/-- The number of potato wedges Cynthia makes -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips Cynthia makes -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_difference_l3164_316416


namespace NUMINAMATH_CALUDE_fourth_board_score_l3164_316408

/-- Represents a dartboard with its score -/
structure Dartboard :=
  (score : ℕ)

/-- Represents the set of four dartboards -/
def Dartboards := Fin 4 → Dartboard

theorem fourth_board_score (boards : Dartboards) 
  (h1 : boards 0 = ⟨30⟩)
  (h2 : boards 1 = ⟨38⟩)
  (h3 : boards 2 = ⟨41⟩)
  (identical : ∀ (i j : Fin 4), (boards i).score + (boards j).score = 2 * ((boards 0).score + (boards 1).score) / 2) :
  (boards 3).score = 34 := by
  sorry

end NUMINAMATH_CALUDE_fourth_board_score_l3164_316408


namespace NUMINAMATH_CALUDE_trig_problem_l3164_316443

theorem trig_problem (θ : ℝ) 
  (h : (2 * Real.cos ((3/2) * Real.pi + θ) + Real.cos (Real.pi + θ)) / 
       (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5/2) * Real.pi + θ)) = 1/5) : 
  Real.tan θ = 1 ∧ Real.sin θ^2 + 3 * Real.sin θ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3164_316443


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3164_316479

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a * b * c) : 
  (a^4 + b^4) / (a * b * (a^3 + b^3)) + 
  (b^4 + c^4) / (b * c * (b^3 + c^3)) + 
  (c^4 + a^4) / (c * a * (c^3 + a^3)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3164_316479


namespace NUMINAMATH_CALUDE_fathers_age_multiplier_l3164_316477

theorem fathers_age_multiplier (father_age son_age : ℕ) (h_sum : father_age + son_age = 75)
  (h_son : son_age = 27) (h_father : father_age = 48) :
  ∃ (M : ℕ), M * (son_age - (father_age - son_age)) = father_age ∧ M = 8 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_multiplier_l3164_316477


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3164_316448

/-- The x-coordinate of a point on a parabola at a given distance from the directrix -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) -- x and y coordinates of point M
  (h1 : y^2 = 4*x) -- point M is on the parabola y² = 4x
  (h2 : |x + 1| = 3) -- distance from M to the directrix x = -1 is 3
  : x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3164_316448


namespace NUMINAMATH_CALUDE_kayak_production_sum_l3164_316475

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 5  -- Initial production in February
  let r := 3  -- Growth ratio
  let n := 6  -- Number of months (February to July)
  geometric_sum a r n = 1820 := by
sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l3164_316475


namespace NUMINAMATH_CALUDE_cricket_bat_price_l3164_316413

theorem cricket_bat_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) : 
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 222 →
  ∃ (cost_price_A : ℝ), cost_price_A = 148 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l3164_316413


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l3164_316447

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2*b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
by sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l3164_316447


namespace NUMINAMATH_CALUDE_poes_speed_l3164_316402

theorem poes_speed (teena_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ) (time : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  final_distance = 15 →
  time = 1.5 →
  ∃ (poe_speed : ℝ), 
    poe_speed = 40 ∧
    teena_speed * time - poe_speed * time = initial_distance + final_distance :=
by
  sorry

end NUMINAMATH_CALUDE_poes_speed_l3164_316402


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3164_316483

theorem complex_fraction_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3164_316483


namespace NUMINAMATH_CALUDE_group_size_l3164_316431

/-- The number of people in the group -/
def n : ℕ := sorry

/-- The original weight of each person in kg -/
def original_weight : ℝ := 50

/-- The weight of the new person in kg -/
def new_person_weight : ℝ := 70

/-- The average weight increase in kg -/
def average_increase : ℝ := 2.5

theorem group_size :
  (n : ℝ) * (original_weight + average_increase) = n * original_weight + (new_person_weight - original_weight) →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_l3164_316431


namespace NUMINAMATH_CALUDE_searchlight_probability_l3164_316439

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℚ := 2

/-- The time in seconds for one complete revolution of the searchlight -/
def revolution_time : ℚ := 60 / revolutions_per_minute

/-- The minimum time in seconds a man needs to stay in the dark -/
def min_dark_time : ℚ := 10

/-- The probability of a man staying in the dark for at least the minimum time -/
def dark_probability : ℚ := min_dark_time / revolution_time

theorem searchlight_probability :
  dark_probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_searchlight_probability_l3164_316439


namespace NUMINAMATH_CALUDE_eighth_of_two_power_44_l3164_316463

theorem eighth_of_two_power_44 (x : ℤ) :
  (2^44 : ℚ) / 8 = 2^x → x = 41 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_two_power_44_l3164_316463


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l3164_316488

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents Alice's usage of stationery --/
def alice_usage (box : StationeryBox) : Prop :=
  box.sheets - 2 * box.envelopes = 80

/-- Represents Bob's usage of stationery --/
def bob_usage (box : StationeryBox) : Prop :=
  4 * box.envelopes = box.sheets ∧ box.envelopes ≥ 35

theorem stationery_box_sheets :
  ∃ (box : StationeryBox), alice_usage box ∧ bob_usage box ∧ box.sheets = 160 := by
  sorry

#check stationery_box_sheets

end NUMINAMATH_CALUDE_stationery_box_sheets_l3164_316488


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3164_316457

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  2/a + 1/b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3164_316457


namespace NUMINAMATH_CALUDE_proportional_segments_l3164_316461

theorem proportional_segments (a b c d : ℝ) : 
  b = 3 → c = 6 → d = 9 → (a / b = c / d) → a = 2 := by sorry

end NUMINAMATH_CALUDE_proportional_segments_l3164_316461


namespace NUMINAMATH_CALUDE_red_ball_removal_l3164_316449

theorem red_ball_removal (total : ℕ) (initial_red_percent : ℚ) (final_red_percent : ℚ) 
  (removed : ℕ) (h_total : total = 600) (h_initial_red : initial_red_percent = 70/100) 
  (h_final_red : final_red_percent = 60/100) (h_removed : removed = 150) : 
  (initial_red_percent * total - removed) / (total - removed) = final_red_percent := by
  sorry

end NUMINAMATH_CALUDE_red_ball_removal_l3164_316449


namespace NUMINAMATH_CALUDE_no_zero_root_l3164_316472

theorem no_zero_root : 
  (∀ x : ℝ, 4 * x^2 - 3 = 49 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - x - 20 = 0 → x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_zero_root_l3164_316472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3164_316485

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_property (b : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence b)
  (h_incr : ∀ n : ℕ, b n < b (n + 1))
  (h_prod : b 4 * b 7 = 24) :
  b 3 * b 8 = 200 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3164_316485


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3164_316459

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 1000.0000000000001)
  (h2 : final_salary = 1045) :
  ∃ P : ℝ, 
    (P = 10) ∧ 
    (final_salary = initial_salary * (1 + P / 100) * (1 - 5 / 100)) := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3164_316459


namespace NUMINAMATH_CALUDE_line_circle_intersection_x_intercept_l3164_316411

/-- The x-intercept of a line that intersects a circle --/
theorem line_circle_intersection_x_intercept
  (m : ℝ)  -- Slope of the line
  (h1 : ∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → x^2 + y^2 = 12 → 
         ∃ A B : ℝ × ℝ, A ≠ B ∧ 
         m * A.1 + A.2 + 3 * m - Real.sqrt 3 = 0 ∧
         A.1^2 + A.2^2 = 12 ∧
         m * B.1 + B.2 + 3 * m - Real.sqrt 3 = 0 ∧
         B.1^2 + B.2^2 = 12)
  (h2 : ∃ A B : ℝ × ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) :
  ∃ x : ℝ, x = -6 ∧ m * x + 3 * m - Real.sqrt 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_x_intercept_l3164_316411


namespace NUMINAMATH_CALUDE_eighteenth_prime_l3164_316455

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ :=
  sorry

-- State the theorem
theorem eighteenth_prime :
  (nthPrime 7 = 17) → (nthPrime 18 = 67) :=
by sorry

end NUMINAMATH_CALUDE_eighteenth_prime_l3164_316455


namespace NUMINAMATH_CALUDE_max_b_value_l3164_316445

/-- Given a box with volume 360 cubic units and integer dimensions a, b, and c
    satisfying 1 < c < b < a, the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 → 
  1 < c → c < b → b < a → 
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l3164_316445


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3164_316412

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3164_316412


namespace NUMINAMATH_CALUDE_blue_ball_probability_l3164_316426

theorem blue_ball_probability (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue : ℚ) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l3164_316426


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3164_316450

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (x, 2) (1, 6) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3164_316450


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3164_316481

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3164_316481


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3164_316446

theorem arithmetic_calculations : 
  ((1 : Int) * (-11) + 8 + (-14) = -17) ∧ 
  (13 - (-12) + (-21) = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3164_316446


namespace NUMINAMATH_CALUDE_log_ratio_equality_l3164_316435

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_ratio_equality (m n : ℝ) 
  (h1 : log10 2 = m) 
  (h2 : log10 3 = n) : 
  (log10 12) / (log10 15) = (2*m + n) / (1 - m + n) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equality_l3164_316435


namespace NUMINAMATH_CALUDE_smallest_term_divisible_by_billion_l3164_316470

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_billion (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10^9

theorem smallest_term_divisible_by_billion :
  let a₁ := 5 / 8
  let a₂ := 50
  (∀ n < 9, ¬ is_divisible_by_billion (geometric_sequence a₁ a₂ n)) ∧
  is_divisible_by_billion (geometric_sequence a₁ a₂ 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_term_divisible_by_billion_l3164_316470


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l3164_316462

/-- Given three jars of marbles with specific conditions, prove the ratio of marbles in Jar C to Jar B -/
theorem marble_jar_ratio :
  let jar_a : ℕ := 28
  let jar_b : ℕ := jar_a + 12
  let total : ℕ := 148
  let jar_c : ℕ := total - (jar_a + jar_b)
  (jar_c : ℚ) / jar_b = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l3164_316462


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l3164_316403

/-- Given that Alice takes 30 minutes to clean her room and Bob takes 1/3 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time bob_time : ℚ) : 
  alice_time = 30 → bob_time = (1/3) * alice_time → bob_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l3164_316403


namespace NUMINAMATH_CALUDE_min_value_polynomial_l3164_316493

theorem min_value_polynomial (x y : ℝ) : 
  ∀ a b : ℝ, 5 * a^2 - 4 * a * b + 4 * b^2 + 12 * a + 25 ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l3164_316493


namespace NUMINAMATH_CALUDE_wendys_cookies_l3164_316456

/-- Represents the number of pastries in various categories -/
structure Pastries where
  cupcakes : ℕ
  cookies : ℕ
  taken_home : ℕ
  sold : ℕ

/-- The theorem statement for Wendy's bake sale problem -/
theorem wendys_cookies (w : Pastries) 
  (h1 : w.cupcakes = 4)
  (h2 : w.taken_home = 24)
  (h3 : w.sold = 9)
  (h4 : w.cupcakes + w.cookies = w.taken_home + w.sold) :
  w.cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_wendys_cookies_l3164_316456


namespace NUMINAMATH_CALUDE_a_greater_than_one_l3164_316490

-- Define an increasing function on the real numbers
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem a_greater_than_one
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inequality : f (a + 1) < f (2 * a)) :
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_a_greater_than_one_l3164_316490


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l3164_316480

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1050) :
  210 = Nat.gcd q r ∧ ∀ x : ℕ, x < 210 → x ≠ Nat.gcd q r :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l3164_316480


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3164_316482

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3164_316482


namespace NUMINAMATH_CALUDE_prop_2_prop_3_l3164_316498

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (containedIn : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem prop_2 
  (m : Line) (α β : Plane) 
  (h1 : planeParallel α β) 
  (h2 : containedIn m α) : 
  parallel m β :=
sorry

-- Theorem for proposition ③
theorem prop_3 
  (m n : Line) (α β : Plane)
  (h1 : perpendicular n α)
  (h2 : perpendicular n β)
  (h3 : perpendicular m α) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_l3164_316498


namespace NUMINAMATH_CALUDE_arithmetic_progression_includes_1999_l3164_316432

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_progression_includes_1999
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_positive : d > 0)
  (h_arithmetic : IsArithmeticProgression a d)
  (h_7 : ∃ n, a n = 7)
  (h_15 : ∃ n, a n = 15)
  (h_27 : ∃ n, a n = 27) :
  ∃ n, a n = 1999 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_includes_1999_l3164_316432


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l3164_316451

theorem a_eq_one_sufficient_not_necessary_for_a_sq_eq_one :
  ∃ (a : ℝ), (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l3164_316451


namespace NUMINAMATH_CALUDE_empty_solution_implies_a_leq_5_l3164_316410

theorem empty_solution_implies_a_leq_5 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_implies_a_leq_5_l3164_316410


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3164_316425

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / 2^n = 220 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3164_316425


namespace NUMINAMATH_CALUDE_max_m_value_l3164_316499

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 2/a + 1/b = 1/4) (h_ineq : ∀ m : ℝ, 2*a + b ≥ 4*m) : 
  ∃ m_max : ℝ, m_max = 9 ∧ ∀ m : ℝ, (∀ x : ℝ, 2*a + b ≥ 4*x → m ≤ x) → m ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3164_316499


namespace NUMINAMATH_CALUDE_fermat_prime_l3164_316469

theorem fermat_prime (n : ℕ) (p : ℕ) (h1 : p = 2^n + 1) 
  (h2 : (3^((p-1)/2) + 1) % p = 0) : Nat.Prime p := by
  sorry

end NUMINAMATH_CALUDE_fermat_prime_l3164_316469


namespace NUMINAMATH_CALUDE_equation_solution_l3164_316406

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) = 1) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3164_316406
