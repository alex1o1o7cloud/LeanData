import Mathlib

namespace NUMINAMATH_CALUDE_sister_age_problem_l3183_318344

theorem sister_age_problem (younger_current_age older_current_age : ℕ) 
  (h1 : younger_current_age = 18)
  (h2 : older_current_age = 26)
  (h3 : ∃ k : ℕ, younger_current_age - k + older_current_age - k = 20) :
  ∃ k : ℕ, older_current_age - k = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_sister_age_problem_l3183_318344


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3183_318359

theorem min_value_sum_fractions (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  (∃ (x : ℝ), 0 < x → (x + x + k) / x + (x + x + k) / x + (x + x + k) / x = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3183_318359


namespace NUMINAMATH_CALUDE_principal_calculation_l3183_318361

/-- Given a principal amount P at simple interest for 3 years, 
    if increasing the interest rate by 1% results in Rs. 72 more interest, 
    then P = 2400. -/
theorem principal_calculation (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3183_318361


namespace NUMINAMATH_CALUDE_three_non_congruent_triangles_l3183_318313

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles : 
  ∃ (t1 t2 t3 : IntTriangle), 
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬(are_congruent t1 t2) ∧
    ¬(are_congruent t1 t3) ∧
    ¬(are_congruent t2 t3) ∧
    ∀ (t : IntTriangle), t ∈ triangles_with_perimeter_11 → 
      (are_congruent t t1 ∨ are_congruent t t2 ∨ are_congruent t t3) :=
by
  sorry

end NUMINAMATH_CALUDE_three_non_congruent_triangles_l3183_318313


namespace NUMINAMATH_CALUDE_remainder_sum_l3183_318304

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 75)
  (hd : d % 120 = 117) : 
  (c + d) % 40 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3183_318304


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_60_bottles_6_samples_l3183_318302

/-- Systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- The problem statement -/
theorem systematic_sampling_interval_for_60_bottles_6_samples :
  systematicSamplingInterval 60 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_60_bottles_6_samples_l3183_318302


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l3183_318366

/-- Proves that given a train with a speed of 90 kmph including stoppages
    and stopping for 15 minutes per hour, the speed of the train excluding
    stoppages is 120 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_with_stoppages = 90)
  (h2 : stopping_time = 15/60) :
  let running_time := 1 - stopping_time
  speed_with_stoppages * 1 = speed_with_stoppages * running_time →
  speed_with_stoppages / running_time = 120 :=
by
  sorry

#check train_speed_excluding_stoppages

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l3183_318366


namespace NUMINAMATH_CALUDE_jason_nickels_l3183_318314

theorem jason_nickels : 
  ∀ (n q : ℕ), 
    n = q + 10 → 
    5 * n + 10 * q = 680 → 
    n = 52 := by
  sorry

end NUMINAMATH_CALUDE_jason_nickels_l3183_318314


namespace NUMINAMATH_CALUDE_fountain_distance_is_30_l3183_318306

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℕ :=
  total_distance / num_trips

/-- Theorem stating that the water fountain is 30 feet from Mrs. Hilt's desk -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fountain_distance_is_30_l3183_318306


namespace NUMINAMATH_CALUDE_zero_vector_length_l3183_318328

theorem zero_vector_length (n : Type*) [NormedAddCommGroup n] : ‖(0 : n)‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_vector_length_l3183_318328


namespace NUMINAMATH_CALUDE_petya_run_time_l3183_318397

/-- Time (in seconds) for Petya to run from 4th to 1st floor -/
def petya_4_to_1 : ℝ := sorry

/-- Time (in seconds) for Mom's elevator ride from 4th to 1st floor -/
def mom_elevator : ℝ := sorry

/-- Time (in seconds) for Petya to run from 5th to 1st floor -/
def petya_5_to_1 : ℝ := sorry

/-- Time (in seconds) for Petya to run one flight of stairs -/
def petya_one_flight : ℝ := sorry

theorem petya_run_time : 
  (petya_4_to_1 + 2 = mom_elevator) ∧ 
  (mom_elevator + 2 = petya_5_to_1) ∧
  (petya_one_flight * 3 = petya_4_to_1) ∧
  (petya_one_flight * 4 = petya_5_to_1) →
  petya_4_to_1 = 12 := by sorry

end NUMINAMATH_CALUDE_petya_run_time_l3183_318397


namespace NUMINAMATH_CALUDE_extra_flowers_l3183_318386

def tulips : ℕ := 4
def roses : ℕ := 11
def used_flowers : ℕ := 11

theorem extra_flowers :
  tulips + roses - used_flowers = 4 :=
by sorry

end NUMINAMATH_CALUDE_extra_flowers_l3183_318386


namespace NUMINAMATH_CALUDE_crescent_moon_area_l3183_318360

/-- The area of a crescent moon formed by two circles -/
theorem crescent_moon_area :
  let large_circle_radius : ℝ := 4
  let small_circle_radius : ℝ := 2
  let large_quarter_circle_area : ℝ := π * large_circle_radius^2 / 4
  let small_half_circle_area : ℝ := π * small_circle_radius^2 / 2
  large_quarter_circle_area - small_half_circle_area = 2 * π := by
sorry

end NUMINAMATH_CALUDE_crescent_moon_area_l3183_318360


namespace NUMINAMATH_CALUDE_max_negative_integers_l3183_318372

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    (∃ (neg_set : Finset ℤ),
      neg_set ⊆ {a, b, c, d, e, f} ∧
      neg_set.card = neg_count ∧
      ∀ x ∈ neg_set, x < 0) ∧
    ∀ (other_neg_set : Finset ℤ),
      other_neg_set ⊆ {a, b, c, d, e, f} →
      (∀ x ∈ other_neg_set, x < 0) →
      other_neg_set.card ≤ neg_count :=
by sorry

end NUMINAMATH_CALUDE_max_negative_integers_l3183_318372


namespace NUMINAMATH_CALUDE_water_tank_evaporation_l3183_318358

/-- Calculates the remaining water in a tank after evaporation --/
def remaining_water (initial_amount : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_amount - evaporation_rate * days

/-- Proves that 450 gallons remain after 50 days of evaporation --/
theorem water_tank_evaporation :
  remaining_water 500 1 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_evaporation_l3183_318358


namespace NUMINAMATH_CALUDE_milkman_profit_is_90_l3183_318357

/-- Calculates the profit of a milkman selling a milk-water mixture --/
def milkman_profit (total_milk : ℕ) (milk_in_mixture : ℕ) (water_in_mixture : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let total_mixture := milk_in_mixture + water_in_mixture
  let selling_price := total_mixture * cost_per_liter
  let cost_of_milk_used := milk_in_mixture * cost_per_liter
  selling_price - cost_of_milk_used

/-- Proves that the milkman's profit is 90 under given conditions --/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

#eval milkman_profit 30 20 5 18

end NUMINAMATH_CALUDE_milkman_profit_is_90_l3183_318357


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l3183_318362

def total_missiles : ℕ := 70
def selected_missiles : ℕ := 7

def systematic_sample (start : ℕ) (interval : ℕ) : List ℕ :=
  List.range selected_missiles |>.map (fun i => start + i * interval)

theorem correct_systematic_sample :
  ∃ (start : ℕ), start ≤ total_missiles ∧
  systematic_sample start (total_missiles / selected_missiles) =
    [3, 13, 23, 33, 43, 53, 63] :=
by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l3183_318362


namespace NUMINAMATH_CALUDE_f_of_f_3_l3183_318382

def f (x : ℝ) : ℝ := 3 * x^2 + 3 * x - 2

theorem f_of_f_3 : f (f 3) = 3568 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l3183_318382


namespace NUMINAMATH_CALUDE_not_necessarily_regular_l3183_318383

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk ::

/-- Predicate to check if all edges of a polyhedron are equal -/
def all_edges_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all dihedral angles of a polyhedron are equal -/
def all_dihedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if all polyhedral angles of a polyhedron are equal -/
def all_polyhedral_angles_equal (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Predicate to check if a polyhedron is regular -/
def is_regular (p : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a convex polyhedron with equal edges and either equal dihedral angles
    or equal polyhedral angles is not necessarily regular -/
theorem not_necessarily_regular :
  ∃ p : ConvexPolyhedron,
    (all_edges_equal p ∧ all_dihedral_angles_equal p ∧ ¬is_regular p) ∨
    (all_edges_equal p ∧ all_polyhedral_angles_equal p ∧ ¬is_regular p) :=
  sorry

end NUMINAMATH_CALUDE_not_necessarily_regular_l3183_318383


namespace NUMINAMATH_CALUDE_harry_fish_count_l3183_318373

/-- The number of fish Sam has -/
def sam_fish : ℕ := 7

/-- The number of fish Joe has relative to Sam -/
def joe_multiplier : ℕ := 8

/-- The number of fish Harry has relative to Joe -/
def harry_multiplier : ℕ := 4

/-- The number of fish Joe has -/
def joe_fish : ℕ := joe_multiplier * sam_fish

/-- The number of fish Harry has -/
def harry_fish : ℕ := harry_multiplier * joe_fish

theorem harry_fish_count : harry_fish = 224 := by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_l3183_318373


namespace NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l3183_318308

-- Define the original line
def original_line (x y : ℝ) : Prop := y = 3 * x - 4

-- Define the point P₀
def P₀ : ℝ × ℝ := (1, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := y = 3 * x - 1

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := y = -1/3 * x + 7/3

-- Theorem for the parallel line
theorem parallel_line_correct :
  (parallel_line P₀.1 P₀.2) ∧
  (∀ x y z : ℝ, original_line x y → original_line (x + 1) z → z - y = 3) :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (perpendicular_line P₀.1 P₀.2) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, original_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (y₂ - y₁) = -(1/3) * (x₂ - x₁)) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_line_correct_l3183_318308


namespace NUMINAMATH_CALUDE_no_divisibility_by_15_and_11_exists_divisibility_by_11_l3183_318336

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def construct_number (n : ℕ) : ℕ :=
  80000 + n * 1000 + 642

theorem no_divisibility_by_15_and_11 :
  ¬ ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 15 = 0 ∧ 
    (construct_number n) % 11 = 0 :=
sorry

theorem exists_divisibility_by_11 :
  ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_divisibility_by_15_and_11_exists_divisibility_by_11_l3183_318336


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l3183_318377

theorem consecutive_integers_average (c d : ℝ) : 
  (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6 = d →
  ((d-1) + d + (d+1) + (d+2) + (d+3) + (d+4)) / 6 = c + 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l3183_318377


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3183_318394

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3183_318394


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l3183_318363

/-- The series sum from n=1 to infinity of (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) equals 1/4 -/
theorem series_sum_equals_one_fourth :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l3183_318363


namespace NUMINAMATH_CALUDE_quadratic_always_has_distinct_roots_l3183_318393

theorem quadratic_always_has_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 1 = 0 ∧ x₂^2 + m*x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_has_distinct_roots_l3183_318393


namespace NUMINAMATH_CALUDE_even_sum_probability_l3183_318316

/-- Probability of obtaining an even sum when spinning two wheels -/
theorem even_sum_probability (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ)
  (h1 : wheel1_total = 6)
  (h2 : wheel1_even = 2)
  (h3 : wheel2_total = 5)
  (h4 : wheel2_even = 3) :
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total +
  ((wheel1_total - wheel1_even) : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total =
  7 / 15 :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3183_318316


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l3183_318369

/-- Represents the state of the game -/
structure GameState :=
  (dominoes : Finset Nat)
  (current_score : Nat)
  (last_move : Nat)

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : Nat) : Prop :=
  move ∈ state.dominoes ∧ move ≠ state.last_move

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.current_score = 37 ∨ state.current_score > 37

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Nat

/-- Defines a winning strategy for the first player -/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (initial_state : GameState),
    initial_state.dominoes = {1, 2, 3, 4, 5} →
    initial_state.current_score = 0 →
    ∃ (final_state : GameState),
      is_winning_state final_state ∧
      (∀ (opponent_move : Nat),
        valid_move initial_state opponent_move →
        valid_move (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move) 
        (s (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move)))

/-- Theorem stating that there exists a winning strategy for the first player -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy), winning_strategy s :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l3183_318369


namespace NUMINAMATH_CALUDE_xyz_value_l3183_318326

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 16 * Real.rpow 4 (1/3))
  (h2 : x * z = 28 * Real.rpow 4 (1/3))
  (h3 : y * z = 112 / Real.rpow 4 (1/3)) :
  x * y * z = 112 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3183_318326


namespace NUMINAMATH_CALUDE_decagon_area_decagon_area_specific_l3183_318310

/-- The area of a decagon inscribed in a rectangle with specific properties. -/
theorem decagon_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : ℝ :=
  let length := (3 * perimeter) / (10 : ℝ)
  let width := (2 * perimeter) / (10 : ℝ)
  let rectangle_area := length * width
  let triangle_area_long := (1 / 2 : ℝ) * (length / 5) * (length / 5)
  let triangle_area_short := (1 / 2 : ℝ) * (width / 5) * (width / 5)
  let total_removed_area := 4 * triangle_area_long + 4 * triangle_area_short
  rectangle_area - total_removed_area

/-- 
  The area of a decagon inscribed in a rectangle is 1984 square centimeters, given:
  - The vertices of the decagon divide the sides of the rectangle into five equal parts
  - The perimeter of the rectangle is 200 centimeters
  - The ratio of length to width of the rectangle is 3:2
-/
theorem decagon_area_specific : decagon_area 200 3 2 = 1984 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_decagon_area_specific_l3183_318310


namespace NUMINAMATH_CALUDE_no_perfect_squares_sum_l3183_318374

theorem no_perfect_squares_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_sum_l3183_318374


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3183_318321

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + 2 * y = 20) 
  (eq2 : 2 * x + 4 * y = 16) : 
  4 * x^2 + 12 * x * y + 12 * y^2 = 292 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3183_318321


namespace NUMINAMATH_CALUDE_local_integrability_from_difference_l3183_318398

open MeasureTheory

theorem local_integrability_from_difference (f : ℝ → ℝ) 
  (h_meas : Measurable f) 
  (h_diff_int : ∀ t, LocallyIntegrable (fun x => f (x + t) - f x) volume) : 
  LocallyIntegrable f volume :=
sorry

end NUMINAMATH_CALUDE_local_integrability_from_difference_l3183_318398


namespace NUMINAMATH_CALUDE_intersection_sum_l3183_318387

/-- Two lines intersect at a point -/
def intersect_at (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + 6 ∧ y = 4 * x + b

/-- The theorem statement -/
theorem intersection_sum (m b : ℝ) :
  intersect_at m b 8 14 → b + m = -17 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3183_318387


namespace NUMINAMATH_CALUDE_housewife_spending_l3183_318319

theorem housewife_spending (initial_amount : ℚ) (spent_fraction : ℚ) :
  initial_amount = 150 →
  spent_fraction = 2/3 →
  initial_amount * (1 - spent_fraction) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_housewife_spending_l3183_318319


namespace NUMINAMATH_CALUDE_sixteen_radii_ten_circles_regions_l3183_318323

/-- Calculates the number of regions created by radii and concentric circles within a larger circle -/
def regions_in_circle (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem sixteen_radii_ten_circles_regions :
  regions_in_circle 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_radii_ten_circles_regions_l3183_318323


namespace NUMINAMATH_CALUDE_completing_square_l3183_318346

theorem completing_square (x : ℝ) : x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l3183_318346


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l3183_318307

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4  -- radius of smaller sphere
  let r₂ : ℝ := 9  -- radius of larger sphere
  let v₁ := (4/3) * π * r₁^3  -- volume of smaller sphere
  let v₂ := (4/3) * π * r₂^3  -- volume of larger sphere
  v₂ - v₁ = (2656/3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l3183_318307


namespace NUMINAMATH_CALUDE_pencil_case_combinations_l3183_318355

theorem pencil_case_combinations :
  let n : ℕ := 6
  2^n = 64 :=
by sorry

end NUMINAMATH_CALUDE_pencil_case_combinations_l3183_318355


namespace NUMINAMATH_CALUDE_joey_work_hours_l3183_318301

/-- Calculates the number of hours Joey needs to work to buy sneakers -/
def hours_needed (sneaker_cost lawn_count lawn_pay figure_count figure_pay hourly_wage : ℕ) : ℕ :=
  let lawn_income := lawn_count * lawn_pay
  let figure_income := figure_count * figure_pay
  let total_income := lawn_income + figure_income
  let remaining_cost := sneaker_cost - total_income
  remaining_cost / hourly_wage

/-- Proves that Joey needs to work 10 hours to buy the sneakers -/
theorem joey_work_hours : 
  hours_needed 92 3 8 2 9 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_joey_work_hours_l3183_318301


namespace NUMINAMATH_CALUDE_future_cup_analysis_l3183_318376

/-- Represents a class's defensive performance in the "Future Cup" football match --/
structure DefensivePerformance where
  average_goals_conceded : ℝ
  standard_deviation : ℝ

/-- The defensive performance of Class A --/
def class_a : DefensivePerformance :=
  { average_goals_conceded := 1.9,
    standard_deviation := 0.3 }

/-- The defensive performance of Class B --/
def class_b : DefensivePerformance :=
  { average_goals_conceded := 1.3,
    standard_deviation := 1.2 }

theorem future_cup_analysis :
  (class_b.average_goals_conceded < class_a.average_goals_conceded) ∧
  (class_b.standard_deviation > class_a.standard_deviation) ∧
  (class_a.average_goals_conceded + class_a.standard_deviation < 
   class_b.average_goals_conceded + class_b.standard_deviation) :=
by sorry

end NUMINAMATH_CALUDE_future_cup_analysis_l3183_318376


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3183_318337

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 3^2 + 4^2 = c^2) →
  (∃ (k : ℝ), k = b/a ∧ 4/3 = k) →
  a^2 = 9 ∧ b^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3183_318337


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3183_318333

theorem sin_n_eq_cos_810 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) (h3 : Real.sin (n * π / 180) = Real.cos (810 * π / 180)) :
  n = 0 ∨ n = 180 ∨ n = -180 := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_810_l3183_318333


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l3183_318340

theorem not_divides_power_diff (m n : ℕ) : 
  m ≥ 3 → n ≥ 3 → Odd m → Odd n → ¬(2^m - 1 ∣ 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l3183_318340


namespace NUMINAMATH_CALUDE_rectangular_window_area_l3183_318381

/-- The area of a rectangular window with length 47.3 cm and width 24 cm is 1135.2 cm². -/
theorem rectangular_window_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  let area : ℝ := length * width
  area = 1135.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_window_area_l3183_318381


namespace NUMINAMATH_CALUDE_circle_center_sum_l3183_318390

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x + 4*y + 9

/-- The center of a circle given its equation -/
def CircleCenter (eq : (ℝ → ℝ → Prop)) : ℝ × ℝ :=
  sorry

/-- The sum of coordinates of a point -/
def SumOfCoordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

theorem circle_center_sum :
  SumOfCoordinates (CircleCenter CircleEquation) = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3183_318390


namespace NUMINAMATH_CALUDE_total_limes_is_57_l3183_318329

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem: The total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_57_l3183_318329


namespace NUMINAMATH_CALUDE_daughters_return_days_l3183_318384

/-- Represents the return frequency of each daughter in days -/
structure DaughterReturnFrequency where
  eldest : Nat
  middle : Nat
  youngest : Nat

/-- Calculates the number of days at least one daughter returns home -/
def daysAtLeastOneDaughterReturns (freq : DaughterReturnFrequency) (period : Nat) : Nat :=
  sorry

theorem daughters_return_days (freq : DaughterReturnFrequency) (period : Nat) :
  freq.eldest = 5 →
  freq.middle = 4 →
  freq.youngest = 3 →
  period = 100 →
  daysAtLeastOneDaughterReturns freq period = 60 := by
  sorry

end NUMINAMATH_CALUDE_daughters_return_days_l3183_318384


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l3183_318324

/-- Given m ∈ ℝ, the curve C is represented by the equation (2-m)x² + (m+1)y² = 1.
    This theorem states that when m is between 1/2 and 2 (exclusive),
    the curve C represents an ellipse with foci on the x-axis. -/
theorem curve_is_ellipse (m : ℝ) (h1 : 1/2 < m) (h2 : m < 2) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (2-m)*x^2 + (m+1)*y^2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l3183_318324


namespace NUMINAMATH_CALUDE_five_balls_two_boxes_l3183_318392

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 32 ways to put 5 distinguishable balls in 2 distinguishable boxes -/
theorem five_balls_two_boxes : distributeBalls 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_two_boxes_l3183_318392


namespace NUMINAMATH_CALUDE_choose_two_from_eleven_l3183_318327

theorem choose_two_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 2 → Nat.choose n k = 55 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_eleven_l3183_318327


namespace NUMINAMATH_CALUDE_sequence_general_term_l3183_318341

/-- Given a sequence {a_n} where S_n is the sum of the first n terms 
    and S_n = (1/2)(1 - a_n), prove that a_n = (1/3)^n -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = (1/2) * (1 - a n)) :
  ∀ n, a n = (1/3)^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3183_318341


namespace NUMINAMATH_CALUDE_solve_for_c_l3183_318318

theorem solve_for_c (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (b_eq : b = (a + c) / 5) :
  c = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l3183_318318


namespace NUMINAMATH_CALUDE_college_class_period_length_l3183_318332

/-- Given a total time, number of periods, and time between periods, 
    calculate the length of each period. -/
def period_length (total_time : ℕ) (num_periods : ℕ) (time_between : ℕ) : ℕ :=
  (total_time - (num_periods - 1) * time_between) / num_periods

/-- Theorem stating that under the given conditions, each period is 40 minutes long. -/
theorem college_class_period_length : 
  period_length 220 5 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_college_class_period_length_l3183_318332


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3183_318368

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 1300)
  (h2 : new_price = 988) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3183_318368


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3183_318309

theorem gcd_of_three_numbers : Nat.gcd 12357 (Nat.gcd 15498 21726) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3183_318309


namespace NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1km_l3183_318388

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1km :
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1km_l3183_318388


namespace NUMINAMATH_CALUDE_matrix_equality_l3183_318339

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equality (x y z w : ℝ) (h1 : A * B x y z w = B x y z w * A) (h2 : 4 * y ≠ z) :
  (x - w) / (z - 4 * y) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l3183_318339


namespace NUMINAMATH_CALUDE_k_range_k_trapezoid_l3183_318367

-- Define the circles and lines
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2
def circle_N (x y : ℝ) : Prop := x^2 + (y - 8)^2 = 40

def line_l1 (k x y : ℝ) : Prop := y = k * x
def line_l2 (k x y : ℝ) : Prop := y = -1/k * x

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
def conditions (k : ℝ) : Prop :=
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ C ≠ D ∧
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    circle_N C.1 C.2 ∧ circle_N D.1 D.2 ∧
    line_l1 k A.1 A.2 ∧ line_l1 k B.1 B.2 ∧
    line_l2 k C.1 C.2 ∧ line_l2 k D.1 D.2

-- Define the trapezoid condition
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := sorry

-- Theorem for the range of k
theorem k_range (k : ℝ) (h : conditions k) : 
  2 - Real.sqrt 3 < k ∧ k < Real.sqrt 15 / 3 := by sorry

-- Theorem for k when ABCD is a trapezoid
theorem k_trapezoid (k : ℝ) (h : conditions k) :
  (∃ (A B C D : ℝ × ℝ), is_trapezoid A B C D) → k = 1 := by sorry

end NUMINAMATH_CALUDE_k_range_k_trapezoid_l3183_318367


namespace NUMINAMATH_CALUDE_lcm_36_100_l3183_318391

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l3183_318391


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3183_318320

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3183_318320


namespace NUMINAMATH_CALUDE_basketball_competition_equation_l3183_318364

/-- Represents the number of matches in a basketball competition where each pair of classes plays once --/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for 10 total matches, the equation x(x-1)/2 = 10 correctly represents the situation --/
theorem basketball_competition_equation (x : ℕ) (h : number_of_matches x = 10) : 
  x * (x - 1) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_competition_equation_l3183_318364


namespace NUMINAMATH_CALUDE_jade_transactions_l3183_318380

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  jade = 85 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l3183_318380


namespace NUMINAMATH_CALUDE_sin_45_degrees_l3183_318345

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l3183_318345


namespace NUMINAMATH_CALUDE_miriam_initial_marbles_l3183_318338

/-- The number of marbles Miriam initially had --/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam currently has --/
def current_marbles : ℕ := 30

/-- The number of marbles Miriam gave to her brother --/
def brother_marbles : ℕ := 60

/-- The number of marbles Miriam gave to her sister --/
def sister_marbles : ℕ := 2 * brother_marbles

/-- The number of marbles Miriam gave to her friend Savanna --/
def savanna_marbles : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = current_marbles + brother_marbles + sister_marbles + savanna_marbles ∧
  initial_marbles = 300 := by
  sorry

end NUMINAMATH_CALUDE_miriam_initial_marbles_l3183_318338


namespace NUMINAMATH_CALUDE_largest_possible_a_l3183_318342

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 150) :
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧
    a' < 3 * b' ∧
    b' < 4 * c' ∧
    c' < 5 * d' ∧
    d' < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l3183_318342


namespace NUMINAMATH_CALUDE_certain_value_problem_l3183_318399

theorem certain_value_problem (number : ℝ) (value : ℝ) : 
  number = 45 → (1/3 : ℝ) * number - value = 10 → value = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l3183_318399


namespace NUMINAMATH_CALUDE_students_in_biology_or_chemistry_l3183_318300

theorem students_in_biology_or_chemistry (both : ℕ) (biology : ℕ) (chemistry_only : ℕ) : 
  both = 15 → biology = 35 → chemistry_only = 18 → 
  (biology - both) + chemistry_only = 38 := by
sorry

end NUMINAMATH_CALUDE_students_in_biology_or_chemistry_l3183_318300


namespace NUMINAMATH_CALUDE_factorization_equality_l3183_318375

theorem factorization_equality (a b : ℝ) : a * b^2 + 10 * a * b + 25 * a = a * (b + 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3183_318375


namespace NUMINAMATH_CALUDE_probability_theorem_l3183_318356

/-- The number of days the performance lasts -/
def total_days : ℕ := 8

/-- The number of consecutive days Resident A watches -/
def watch_days : ℕ := 3

/-- The number of days we're interested in (first to fourth day) -/
def interest_days : ℕ := 4

/-- The total number of ways to choose 3 consecutive days out of 8 days -/
def total_choices : ℕ := total_days - watch_days + 1

/-- The number of ways to choose 3 consecutive days within the first 4 days -/
def interest_choices : ℕ := interest_days - watch_days + 1

/-- The probability of choosing 3 consecutive days within the first 4 days out of 8 total days -/
theorem probability_theorem : 
  (interest_choices : ℚ) / total_choices = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3183_318356


namespace NUMINAMATH_CALUDE_cube_sum_of_symmetric_polynomials_l3183_318325

theorem cube_sum_of_symmetric_polynomials (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = -3) 
  (h3 : a * b * c = 9) : 
  a^3 + b^3 + c^3 = 22 := by sorry

end NUMINAMATH_CALUDE_cube_sum_of_symmetric_polynomials_l3183_318325


namespace NUMINAMATH_CALUDE_rain_probability_three_days_l3183_318317

theorem rain_probability_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.40) 
  (h2 : prob_saturday = 0.60) 
  (h3 : prob_sunday = 0.35) 
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1) 
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1) 
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.084 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_three_days_l3183_318317


namespace NUMINAMATH_CALUDE_emily_score_emily_score_proof_l3183_318396

/-- Calculates Emily's score in a dodgeball game -/
theorem emily_score (total_players : ℕ) (total_points : ℕ) (other_player_score : ℕ) : ℕ :=
  let other_players := total_players - 1
  let other_players_total := other_players * other_player_score
  total_points - other_players_total

/-- Proves Emily's score given the game conditions -/
theorem emily_score_proof :
  emily_score 8 39 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_emily_score_emily_score_proof_l3183_318396


namespace NUMINAMATH_CALUDE_prob_all_even_is_one_tenth_and_half_l3183_318315

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)
  (sides_pos : sides > 0)

/-- The number of even outcomes on a die -/
def evenOutcomes (d : Die) : ℕ :=
  d.sides / 2

/-- The probability of rolling an even number on a die -/
def probEven (d : Die) : ℚ :=
  evenOutcomes d / d.sides

/-- The three dice in the problem -/
def die1 : Die := ⟨6, by norm_num⟩
def die2 : Die := ⟨7, by norm_num⟩
def die3 : Die := ⟨9, by norm_num⟩

/-- The theorem to be proved -/
theorem prob_all_even_is_one_tenth_and_half :
  probEven die1 * probEven die2 * probEven die3 = 1 / (10 : ℚ) + 1 / (20 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_all_even_is_one_tenth_and_half_l3183_318315


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_under_1000_l3183_318343

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_under_1000_l3183_318343


namespace NUMINAMATH_CALUDE_pirate_coin_sharing_l3183_318370

/-- The number of coins Pete gives himself in the final round -/
def x : ℕ := 9

/-- The total number of coins Pete has at the end -/
def petes_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- The total number of coins Paul has at the end -/
def pauls_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has 5 times as many coins as Paul -/
def pete_five_times_paul (x : ℕ) : Prop :=
  petes_coins x = 5 * pauls_coins x

/-- The total number of coins shared -/
def total_coins (x : ℕ) : ℕ := petes_coins x + pauls_coins x

theorem pirate_coin_sharing :
  pete_five_times_paul x ∧ total_coins x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_sharing_l3183_318370


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l3183_318352

/-- Proves that given a 6-liter solution with an unknown initial alcohol percentage,
    adding 1.8 liters of pure alcohol to create a 50% alcohol solution
    implies that the initial alcohol percentage was 35%. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.8)
  (h3 : final_percentage = 50)
  (h4 : final_percentage / 100 * (initial_volume + added_alcohol) = 
        (initial_volume * x / 100) + added_alcohol) :
  x = 35 :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_percentage_l3183_318352


namespace NUMINAMATH_CALUDE_parallelepiped_net_squares_l3183_318348

/-- Represents a paper parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the net of an unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- The function that unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- The theorem to be proved -/
theorem parallelepiped_net_squares (p : Parallelepiped) (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  unfold p = n →
  n.squares - 1 = 9 →
  n.squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_net_squares_l3183_318348


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l3183_318349

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l3183_318349


namespace NUMINAMATH_CALUDE_compare_base_numbers_l3183_318312

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem compare_base_numbers : 
  let base6_num := [3, 0, 4]
  let base8_num := [7, 1, 2]
  base_to_decimal base6_num 6 > base_to_decimal base8_num 8 := by
sorry

end NUMINAMATH_CALUDE_compare_base_numbers_l3183_318312


namespace NUMINAMATH_CALUDE_bus_train_speed_ratio_l3183_318305

/-- Proves that the fraction of bus speed to train speed is 3/4 -/
theorem bus_train_speed_ratio :
  let train_car_speed_ratio : ℚ := 16 / 15
  let bus_distance : ℕ := 480
  let bus_time : ℕ := 8
  let car_distance : ℕ := 450
  let car_time : ℕ := 6
  let bus_speed : ℚ := bus_distance / bus_time
  let car_speed : ℚ := car_distance / car_time
  let train_speed : ℚ := car_speed * train_car_speed_ratio
  bus_speed / train_speed = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_train_speed_ratio_l3183_318305


namespace NUMINAMATH_CALUDE_pool_length_calculation_l3183_318378

/-- Calculates the length of a rectangular pool given its draining rate, width, depth, initial capacity, and time to drain. -/
theorem pool_length_calculation (drain_rate : ℝ) (width depth : ℝ) (initial_capacity : ℝ) (drain_time : ℝ) :
  drain_rate = 60 →
  width = 40 →
  depth = 10 →
  initial_capacity = 0.8 →
  drain_time = 800 →
  (drain_rate * drain_time) / initial_capacity / (width * depth) = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_length_calculation_l3183_318378


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l3183_318311

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l3183_318311


namespace NUMINAMATH_CALUDE_minimum_distance_theorem_l3183_318371

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the position and speed of a character -/
structure Character where
  start : Point
  speed : ℝ

/-- The minimum distance between two characters chasing a target -/
def minDistance (char1 char2 : Character) (target : Point) : ℝ :=
  sorry

/-- Garfield's initial position and speed -/
def garfield : Character :=
  { start := { x := 0, y := 0 }, speed := 7 }

/-- Odie's initial position and speed -/
def odie : Character :=
  { start := { x := 25, y := 0 }, speed := 10 }

/-- The target point both characters are chasing -/
def target : Point :=
  { x := 9, y := 12 }

theorem minimum_distance_theorem :
  minDistance garfield odie target = 10 / Real.sqrt 149 :=
sorry

end NUMINAMATH_CALUDE_minimum_distance_theorem_l3183_318371


namespace NUMINAMATH_CALUDE_inequality_implication_l3183_318347

theorem inequality_implication (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) : 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3183_318347


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l3183_318322

/-- Proves that given a total amount of Rs. 15,500 after 4 years with a simple interest rate of 6% per annum, the initial sum of money (principal) is Rs. 12,500. -/
theorem initial_sum_calculation (total_amount : ℝ) (time : ℝ) (rate : ℝ) (principal : ℝ)
  (h1 : total_amount = 15500)
  (h2 : time = 4)
  (h3 : rate = 6)
  (h4 : total_amount = principal + (principal * rate * time / 100)) :
  principal = 12500 := by
sorry

end NUMINAMATH_CALUDE_initial_sum_calculation_l3183_318322


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3183_318379

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x : ℝ | 3 * x^2 + 6 * x - 4 = x^2 + 2 * x + 1}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  x^2 + 2 * x + 1

/-- The set of intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ intersection_x ∧ p.2 = intersection_y p.1}

theorem parabolas_intersection :
  intersection_points = {(-5, 16), (1/2, 9/4)} :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3183_318379


namespace NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l3183_318335

-- Define the triangles
def triangle_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def triangle_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Define a function to calculate the area of a right triangle
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b : ℚ) / 2

-- Theorem statement
theorem area_ratio_GHI_JKL :
  let (g1, g2, _) := triangle_GHI
  let (j1, j2, _) := triangle_JKL
  (area_right_triangle g1 g2) / (area_right_triangle j1 j2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l3183_318335


namespace NUMINAMATH_CALUDE_largest_non_sum_of_multiple_30_and_composite_l3183_318395

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_of_multiple_30_and_composite :
  (∀ n > 211, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_multiple_30_and_composite_l3183_318395


namespace NUMINAMATH_CALUDE_complex_roots_power_sum_l3183_318330

theorem complex_roots_power_sum (α β : ℂ) (p : ℕ) : 
  (2 * α^4 - 6 * α^3 + 11 * α^2 - 6 * α - 4 = 0) →
  (2 * β^4 - 6 * β^3 + 11 * β^2 - 6 * β - 4 = 0) →
  p ≥ 5 →
  α^p + β^p = (α + β)^p := by sorry

end NUMINAMATH_CALUDE_complex_roots_power_sum_l3183_318330


namespace NUMINAMATH_CALUDE_f_of_f_of_one_eq_31_l3183_318365

def f (x : ℝ) : ℝ := 4 * x^3 + 2 * x^2 - 5 * x + 1

theorem f_of_f_of_one_eq_31 : f (f 1) = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_one_eq_31_l3183_318365


namespace NUMINAMATH_CALUDE_megan_zoo_pictures_l3183_318389

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan has left after deleting -/
def remaining_pictures : ℕ := 2

/-- Theorem stating that Megan took 15 pictures at the zoo -/
theorem megan_zoo_pictures : 
  zoo_pictures = 15 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures := sorry
  sorry

end NUMINAMATH_CALUDE_megan_zoo_pictures_l3183_318389


namespace NUMINAMATH_CALUDE_undefined_expression_l3183_318350

theorem undefined_expression (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 1) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 :=
sorry

end NUMINAMATH_CALUDE_undefined_expression_l3183_318350


namespace NUMINAMATH_CALUDE_triangle_inequality_l3183_318334

theorem triangle_inequality (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : a + f = b + c ∧ b + c = d + e) : 
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (c^2 - c*d + d^2) > Real.sqrt (e^2 - e*f + f^2) ∧
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (c^2 - c*d + d^2) ∧
  Real.sqrt (c^2 - c*d + d^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (a^2 - a*b + b^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3183_318334


namespace NUMINAMATH_CALUDE_liam_picked_40_oranges_l3183_318303

/-- The number of oranges Liam picked -/
def liam_oranges : ℕ := sorry

/-- The price of 2 of Liam's oranges in cents -/
def liam_price : ℕ := 250

/-- The number of oranges Claire picked -/
def claire_oranges : ℕ := 30

/-- The price of each of Claire's oranges in cents -/
def claire_price : ℕ := 120

/-- The total amount saved in cents -/
def total_saved : ℕ := 8600

theorem liam_picked_40_oranges :
  liam_oranges = 40 ∧
  liam_price = 250 ∧
  claire_oranges = 30 ∧
  claire_price = 120 ∧
  total_saved = 8600 ∧
  (liam_oranges * liam_price / 2 + claire_oranges * claire_price = total_saved) :=
by sorry

end NUMINAMATH_CALUDE_liam_picked_40_oranges_l3183_318303


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3183_318385

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 3*x + 1 = 0 ∧ 
   ∀ y : ℝ, k*y^2 - 3*y + 1 = 0 → y = x) → 
  k = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3183_318385


namespace NUMINAMATH_CALUDE_millet_majority_on_wednesday_l3183_318331

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Rat
  other_seeds : Rat

/-- Calculates the next day's feeder state -/
def next_day (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet * (4/5),
    other_seeds := 0 }

/-- Adds new seeds to the feeder (every other day) -/
def add_seeds (state : FeederState) : FeederState :=
  { day := state.day,
    millet := state.millet + 2/5,
    other_seeds := state.other_seeds + 3/5 }

/-- Initial state of the feeder on Monday -/
def initial_state : FeederState :=
  { day := 1, millet := 2/5, other_seeds := 3/5 }

/-- Theorem: On Wednesday (day 3), millet is more than half of total seeds -/
theorem millet_majority_on_wednesday :
  let state_wednesday := add_seeds (next_day (next_day initial_state))
  state_wednesday.millet > (state_wednesday.millet + state_wednesday.other_seeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_wednesday_l3183_318331


namespace NUMINAMATH_CALUDE_extremum_implies_deriv_root_exists_deriv_root_without_extremum_l3183_318351

-- Define a differentiable function on the real line
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f'(x) = 0 to have a real root
def deriv_has_root (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, deriv f x = 0

-- Statement 1: If f has an extremum, then f'(x) = 0 has a real root
theorem extremum_implies_deriv_root :
  has_extremum f → deriv_has_root f :=
sorry

-- Statement 2: There exists a function f such that f'(x) = 0 has a real root,
-- but f does not have an extremum
theorem exists_deriv_root_without_extremum :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv_has_root f ∧ ¬has_extremum f :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_deriv_root_exists_deriv_root_without_extremum_l3183_318351


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3183_318354

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  (b - a = c - b) →  -- arithmetic sequence
  (a / c = b / a) →  -- geometric sequence
  (a + 3*b + c = 10) →  -- sum condition
  a = -4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3183_318354


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3183_318353

theorem binomial_divisibility (p k : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ m : ℤ, (p : ℤ)^3 * m = (Nat.choose (k * p) p : ℤ) - k := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3183_318353
