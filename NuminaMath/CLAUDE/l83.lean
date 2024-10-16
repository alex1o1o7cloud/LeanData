import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l83_8345

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l83_8345


namespace NUMINAMATH_CALUDE_eggs_in_park_l83_8331

theorem eggs_in_park (total_eggs club_house_eggs town_hall_eggs : ℕ) 
  (h1 : total_eggs = 20)
  (h2 : club_house_eggs = 12)
  (h3 : town_hall_eggs = 3) :
  total_eggs - club_house_eggs - town_hall_eggs = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_in_park_l83_8331


namespace NUMINAMATH_CALUDE_min_value_of_expression_l83_8358

theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 2 * m - n * (-2) - 2 = 0 → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') → 
  1 / m + 2 / n = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l83_8358


namespace NUMINAMATH_CALUDE_digit_sum_divisibility_l83_8359

theorem digit_sum_divisibility (n k : ℕ) (hn : n > 0) (hk : k ≥ n) (h3 : ¬3 ∣ n) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (∃ digits : List ℕ, m.digits 10 = digits ∧ digits.sum = k) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_divisibility_l83_8359


namespace NUMINAMATH_CALUDE_f_extrema_and_inequality_l83_8327

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 + (1/2) * x^2

theorem f_extrema_and_inequality :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ 1 + (Real.exp 1)^2) ∧
  (∀ x ∈ Set.Ioi 1, f x < g x) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_inequality_l83_8327


namespace NUMINAMATH_CALUDE_carlos_baseball_cards_l83_8341

theorem carlos_baseball_cards :
  ∀ (jorge matias carlos : ℕ),
    jorge = matias →
    matias = carlos - 6 →
    jorge + matias + carlos = 48 →
    carlos = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_carlos_baseball_cards_l83_8341


namespace NUMINAMATH_CALUDE_min_value_absolute_difference_l83_8306

theorem min_value_absolute_difference (x : ℝ) :
  ((2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2) →
  (∃ y : ℝ, y = |x - 1| - |x + 3| ∧ 
   (∀ z : ℝ, ((2 * z - 1) / 3 - 1 ≥ z - (5 - 3 * z) / 2) → y ≤ |z - 1| - |z + 3|) ∧
   y = -2 - 8 / 11) :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_difference_l83_8306


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l83_8391

/-- The line equation passes through a fixed point for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check line_passes_through_fixed_point

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l83_8391


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l83_8354

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_of_P_in_U : {2} = U \ P := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l83_8354


namespace NUMINAMATH_CALUDE_octahedron_inscribed_in_cube_l83_8353

/-- A cube in 3D space -/
structure Cube where
  edge_length : ℝ
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- An octahedron in 3D space -/
structure Octahedron where
  vertices : Fin 6 → ℝ × ℝ × ℝ

/-- Predicate to check if a point lies on an edge of a cube -/
def point_on_cube_edge (c : Cube) (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (i j : Fin 8), i ≠ j ∧ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      p = ((1 - t) • (c.vertices i) + t • (c.vertices j))

/-- Theorem stating that an octahedron can be inscribed in a cube 
    with its vertices on the cube's edges -/
theorem octahedron_inscribed_in_cube : 
  ∃ (c : Cube) (o : Octahedron), 
    ∀ (i : Fin 6), point_on_cube_edge c (o.vertices i) :=
  sorry

end NUMINAMATH_CALUDE_octahedron_inscribed_in_cube_l83_8353


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l83_8372

theorem repeating_decimal_problem (n : ℕ) :
  n < 1000 ∧
  (∃ (a b c d e f : ℕ), (1 : ℚ) / n = (a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) / 999999) ∧
  (∃ (w x y z : ℕ), (1 : ℚ) / (n + 5) = (w * 1000 + x * 100 + y * 10 + z) / 9999) →
  151 ≤ n ∧ n ≤ 300 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l83_8372


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l83_8377

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (h : r > 0) : a r = 8 * r + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l83_8377


namespace NUMINAMATH_CALUDE_locus_is_circumcircle_l83_8325

/-- Triangle represented by its vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Distance from a point to a line segment -/
def distToSide (P : Point) (A B : Point) : ℝ := sorry

/-- Distance between two points -/
def dist (P Q : Point) : ℝ := sorry

/-- Circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A point lies on the circumcircle of a triangle -/
def onCircumcircle (P : Point) (t : Triangle) : Prop :=
  P ∈ circumcircle t

theorem locus_is_circumcircle (t : Triangle) (P : Point) :
  (distToSide P t.A t.B * dist P t.C = distToSide P t.A t.C * dist P t.B) →
  onCircumcircle P t := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circumcircle_l83_8325


namespace NUMINAMATH_CALUDE_minimum_bags_in_warehouse_A_minimum_bags_proof_l83_8302

theorem minimum_bags_in_warehouse_A : ℕ → ℕ → Prop :=
  fun x y =>
    (∃ k : ℕ, 
      (y + 90 = 2 * (x - 90)) ∧
      (x + k = 6 * (y - k)) ∧
      (x ≥ 139) ∧
      (∀ z : ℕ, z < x → 
        ¬(∃ w k : ℕ, 
          (w + 90 = 2 * (z - 90)) ∧
          (z + k = 6 * (w - k))))) →
    x = 139

-- The proof goes here
theorem minimum_bags_proof : 
  ∃ x y : ℕ, minimum_bags_in_warehouse_A x y :=
sorry

end NUMINAMATH_CALUDE_minimum_bags_in_warehouse_A_minimum_bags_proof_l83_8302


namespace NUMINAMATH_CALUDE_decagon_triangles_l83_8364

def regularDecagonVertices : ℕ := 10

def trianglesFromDecagon : ℕ :=
  Nat.choose regularDecagonVertices 3

theorem decagon_triangles :
  trianglesFromDecagon = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l83_8364


namespace NUMINAMATH_CALUDE_sonya_falls_count_l83_8329

/-- The number of times Sonya fell while ice skating --/
def sonya_falls (steven_falls stephanie_falls : ℕ) : ℕ :=
  (stephanie_falls / 2) - 2

/-- Proof that Sonya fell 6 times given the conditions --/
theorem sonya_falls_count :
  ∀ (steven_falls stephanie_falls : ℕ),
    steven_falls = 3 →
    stephanie_falls = steven_falls + 13 →
    sonya_falls steven_falls stephanie_falls = 6 := by
  sorry

end NUMINAMATH_CALUDE_sonya_falls_count_l83_8329


namespace NUMINAMATH_CALUDE_high_school_ten_games_count_l83_8309

def num_teams : ℕ := 10
def games_against_each_team : ℕ := 2
def non_conference_games_per_team : ℕ := 6

theorem high_school_ten_games_count :
  (num_teams * (num_teams - 1) / 2) * games_against_each_team + 
  num_teams * non_conference_games_per_team = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_games_count_l83_8309


namespace NUMINAMATH_CALUDE_mark_change_factor_l83_8304

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℝ) (h1 : n = 12) (h2 : original_avg = 36) (h3 : new_avg = 72) :
  ∃ (factor : ℝ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l83_8304


namespace NUMINAMATH_CALUDE_lte_lemma_largest_power_of_two_dividing_difference_l83_8375

-- Define the valuation function v_2
def v_2 (n : ℕ) : ℕ := sorry

-- Define the Lifting The Exponent Lemma
theorem lte_lemma (a b : ℕ) (h : Odd a ∧ Odd b) :
  v_2 (a^4 - b^4) = v_2 (a - b) + v_2 4 + v_2 (a + b) - 1 := sorry

-- Main theorem
theorem largest_power_of_two_dividing_difference :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) := by sorry

end NUMINAMATH_CALUDE_lte_lemma_largest_power_of_two_dividing_difference_l83_8375


namespace NUMINAMATH_CALUDE_product_mod_seven_l83_8371

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l83_8371


namespace NUMINAMATH_CALUDE_exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l83_8320

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event "Exactly one head is up" -/
def exactlyOneHead : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads)}

/-- The event "Exactly two heads are up" -/
def exactlyTwoHeads : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set TwoCoinsOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary :
  mutuallyExclusive exactlyOneHead exactlyTwoHeads ∧
  ¬complementary exactlyOneHead exactlyTwoHeads :=
by sorry

end NUMINAMATH_CALUDE_exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l83_8320


namespace NUMINAMATH_CALUDE_perpendicular_condition_l83_8303

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (l m n : Line) (α : Plane) 
  (h1 : in_plane m α) 
  (h2 : in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧ 
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l83_8303


namespace NUMINAMATH_CALUDE_intersection_A_B_l83_8328

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l83_8328


namespace NUMINAMATH_CALUDE_tree_growth_fraction_l83_8389

theorem tree_growth_fraction (initial_height : ℝ) (yearly_growth : ℝ) :
  initial_height = 4 →
  yearly_growth = 0.4 →
  let height_4th_year := initial_height + 4 * yearly_growth
  let height_6th_year := initial_height + 6 * yearly_growth
  (height_6th_year - height_4th_year) / height_4th_year = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_fraction_l83_8389


namespace NUMINAMATH_CALUDE_nancy_soap_purchase_l83_8367

/-- Calculates the total number of soap bars Nancy bought from three brands. -/
def total_soap_bars (brand_a_packs : ℕ) (brand_a_bars_per_pack : ℕ)
                    (brand_b_packs : ℕ) (brand_b_bars_per_pack : ℕ)
                    (brand_c_packs : ℕ) (brand_c_bars_per_pack : ℕ)
                    (brand_c_free_pack_bars : ℕ) : ℕ :=
  brand_a_packs * brand_a_bars_per_pack +
  brand_b_packs * brand_b_bars_per_pack +
  brand_c_packs * brand_c_bars_per_pack +
  brand_c_free_pack_bars

theorem nancy_soap_purchase :
  total_soap_bars 4 3 3 5 2 6 4 = 43 := by
  sorry

end NUMINAMATH_CALUDE_nancy_soap_purchase_l83_8367


namespace NUMINAMATH_CALUDE_average_time_per_mile_l83_8335

/-- Proves that the average time per mile is 9 minutes for a 24-mile run completed in 3 hours and 36 minutes -/
theorem average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : 
  distance = 24 ∧ hours = 3 ∧ minutes = 36 → 
  (hours * 60 + minutes) / distance = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_time_per_mile_l83_8335


namespace NUMINAMATH_CALUDE_calculation_proof_l83_8305

theorem calculation_proof :
  (1 * (-8) - 9 - (-3) + (-6) = -20) ∧
  (-2^2 + 3 * (-1)^2023 - |1 - 5| / 2 = -9) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l83_8305


namespace NUMINAMATH_CALUDE_min_balls_theorem_l83_8301

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The total number of balls in the box -/
def total_balls (counts : BallCounts) : Nat :=
  counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black

/-- The minimum number of balls to draw to ensure at least n are of the same color -/
def min_balls_to_draw (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_theorem (counts : BallCounts) (n : Nat) :
  counts.red = 28 →
  counts.green = 20 →
  counts.yellow = 12 →
  counts.blue = 20 →
  counts.white = 10 →
  counts.black = 10 →
  total_balls counts = 100 →
  min_balls_to_draw counts 15 = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_theorem_l83_8301


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l83_8308

def remaining_problems (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  let problems_per_worksheet : ℕ := 3
  let total_worksheets : ℕ := 15
  let graded_worksheets : ℕ := 7
  remaining_problems problems_per_worksheet total_worksheets graded_worksheets = 24 := by
sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l83_8308


namespace NUMINAMATH_CALUDE_fermat_number_prime_count_l83_8396

/-- Fermat number defined as F_n = 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- There are at least n+1 distinct prime numbers less than or equal to F_n -/
theorem fermat_number_prime_count (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n + 1 ∧ (∀ p ∈ S, Nat.Prime p ∧ p ≤ fermat_number n) :=
sorry

end NUMINAMATH_CALUDE_fermat_number_prime_count_l83_8396


namespace NUMINAMATH_CALUDE_expression_value_at_three_l83_8342

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l83_8342


namespace NUMINAMATH_CALUDE_james_has_more_balloons_l83_8307

/-- James has 1222 balloons -/
def james_balloons : ℕ := 1222

/-- Amy has 513 balloons -/
def amy_balloons : ℕ := 513

/-- The difference in balloon count between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

/-- Theorem stating that James has 709 more balloons than Amy -/
theorem james_has_more_balloons : balloon_difference = 709 := by
  sorry

end NUMINAMATH_CALUDE_james_has_more_balloons_l83_8307


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l83_8318

theorem grasshopper_jump_distance (frog_jump : ℕ) (difference : ℕ) : 
  frog_jump = 40 → difference = 15 → frog_jump - difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l83_8318


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_square_sum_digits_l83_8374

def isValidNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (let digits := [n / 100, (n / 10) % 10, n % 10]
   digits.toFinset.card = 3 ∧
   n % ((digits.sum)^2) = 0)

theorem three_digit_divisible_by_square_sum_digits :
  {n : ℕ | isValidNumber n} =
  {162, 243, 324, 405, 512, 648, 729, 810, 972} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_square_sum_digits_l83_8374


namespace NUMINAMATH_CALUDE_root_between_roots_l83_8343

theorem root_between_roots (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) 
  (hr : a * r^2 + b * r + c = 0) 
  (hs : -a * s^2 + b * s + c = 0) : 
  ∃ t, (t > min r s ∧ t < max r s) ∧ (a / 2) * t^2 + b * t + c = 0 :=
sorry

end NUMINAMATH_CALUDE_root_between_roots_l83_8343


namespace NUMINAMATH_CALUDE_square_root_problem_l83_8380

theorem square_root_problem (a x : ℝ) 
  (h1 : Real.sqrt a = x + 3) 
  (h2 : Real.sqrt a = 3 * x - 11) : 
  2 * a - 1 = 199 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l83_8380


namespace NUMINAMATH_CALUDE_arccos_negative_half_l83_8346

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l83_8346


namespace NUMINAMATH_CALUDE_toms_journey_to_virgo_l83_8344

theorem toms_journey_to_virgo (
  train_ride : ℝ)
  (first_layover : ℝ)
  (bus_ride : ℝ)
  (second_layover : ℝ)
  (first_flight : ℝ)
  (third_layover : ℝ)
  (fourth_layover : ℝ)
  (car_drive : ℝ)
  (first_boat_ride : ℝ)
  (fifth_layover : ℝ)
  (final_walk : ℝ)
  (h1 : train_ride = 5)
  (h2 : first_layover = 1.5)
  (h3 : bus_ride = 4)
  (h4 : second_layover = 0.5)
  (h5 : first_flight = 6)
  (h6 : third_layover = 2)
  (h7 : fourth_layover = 3)
  (h8 : car_drive = 3.5)
  (h9 : first_boat_ride = 1.5)
  (h10 : fifth_layover = 0.75)
  (h11 : final_walk = 1.25) :
  train_ride + first_layover + bus_ride + second_layover + first_flight + 
  third_layover + (3 * bus_ride) + fourth_layover + car_drive + 
  first_boat_ride + fifth_layover + (2 * first_boat_ride - 0.5) + final_walk = 44 := by
  sorry


end NUMINAMATH_CALUDE_toms_journey_to_virgo_l83_8344


namespace NUMINAMATH_CALUDE_frame_diameter_l83_8370

theorem frame_diameter (d_y : ℝ) (uncovered_fraction : ℝ) (d_x : ℝ) : 
  d_y = 12 →
  uncovered_fraction = 0.4375 →
  d_x = 16 →
  (π * (d_x / 2)^2) = (π * (d_y / 2)^2) + uncovered_fraction * (π * (d_x / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_frame_diameter_l83_8370


namespace NUMINAMATH_CALUDE_max_d_is_one_l83_8349

def a (n : ℕ) : ℕ := 105 + n^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one :
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ d m = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l83_8349


namespace NUMINAMATH_CALUDE_dave_apps_problem_l83_8395

theorem dave_apps_problem (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 15)
  (h2 : final_apps = 14)
  (h3 : ∃ (added deleted : ℕ), initial_apps + added - deleted = final_apps ∧ deleted = added + 1) :
  ∃ (added : ℕ), added = 0 ∧ initial_apps + added - (added + 1) = final_apps :=
by sorry

end NUMINAMATH_CALUDE_dave_apps_problem_l83_8395


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l83_8356

/-- The revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The optimal price that maximizes revenue -/
def optimal_price : ℝ := 18.75

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℝ, p ≤ 30 → revenue p ≤ revenue optimal_price := by
  sorry


end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l83_8356


namespace NUMINAMATH_CALUDE_fabric_per_shirt_is_two_l83_8340

/-- Represents the daily production and fabric usage in a tailoring business -/
structure TailoringBusiness where
  shirts_per_day : ℕ
  pants_per_day : ℕ
  fabric_per_pants : ℕ
  total_fabric_3days : ℕ

/-- Calculates the amount of fabric used for each shirt -/
def fabric_per_shirt (tb : TailoringBusiness) : ℚ :=
  let total_pants_3days := tb.pants_per_day * 3
  let fabric_for_pants := total_pants_3days * tb.fabric_per_pants
  let fabric_for_shirts := tb.total_fabric_3days - fabric_for_pants
  let total_shirts_3days := tb.shirts_per_day * 3
  fabric_for_shirts / total_shirts_3days

/-- Theorem stating that the amount of fabric per shirt is 2 yards -/
theorem fabric_per_shirt_is_two (tb : TailoringBusiness) 
    (h1 : tb.shirts_per_day = 3)
    (h2 : tb.pants_per_day = 5)
    (h3 : tb.fabric_per_pants = 5)
    (h4 : tb.total_fabric_3days = 93) :
    fabric_per_shirt tb = 2 := by
  sorry

#eval fabric_per_shirt { shirts_per_day := 3, pants_per_day := 5, fabric_per_pants := 5, total_fabric_3days := 93 }

end NUMINAMATH_CALUDE_fabric_per_shirt_is_two_l83_8340


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l83_8312

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 1014 ∧
  ∀ m : ℕ, m < 1014 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l83_8312


namespace NUMINAMATH_CALUDE_two_digit_number_property_l83_8322

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l83_8322


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_2023_l83_8378

/-- A function that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The theorem stating that there are no two-digit factors of 2023 -/
theorem no_two_digit_factors_of_2023 : 
  ¬ ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ a * b = 2023 := by
  sorry

#check no_two_digit_factors_of_2023

end NUMINAMATH_CALUDE_no_two_digit_factors_of_2023_l83_8378


namespace NUMINAMATH_CALUDE_neon_sign_blink_interval_l83_8361

theorem neon_sign_blink_interval (t1 t2 : ℕ) : 
  t1 = 9 → 
  t1.lcm t2 = 45 → 
  t2 = 15 := by
sorry

end NUMINAMATH_CALUDE_neon_sign_blink_interval_l83_8361


namespace NUMINAMATH_CALUDE_f_2017_equals_3_l83_8333

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2017_equals_3 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = -3) :
  f 2017 = 3 := by
sorry

end NUMINAMATH_CALUDE_f_2017_equals_3_l83_8333


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l83_8360

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 180 → a + b + c = 26 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l83_8360


namespace NUMINAMATH_CALUDE_jack_euros_l83_8321

/-- Calculates the number of euros Jack has given his dollar amount, 
    the exchange rate, and his total amount in dollars. -/
def calculate_euros (dollars : ℕ) (exchange_rate : ℕ) (total : ℕ) : ℕ :=
  (total - dollars) / exchange_rate

/-- Proves that Jack has 36 euros given the problem conditions. -/
theorem jack_euros : calculate_euros 45 2 117 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jack_euros_l83_8321


namespace NUMINAMATH_CALUDE_parallel_intersection_l83_8376

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection
  (l₁ l₂ l₃ : Line) (α β : Plane)
  (h1 : parallel l₁ l₂)
  (h2 : subset l₁ α)
  (h3 : subset l₂ β)
  (h4 : intersection α β = l₃) :
  parallel l₁ l₃ :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_l83_8376


namespace NUMINAMATH_CALUDE_egypt_promotion_theorem_l83_8315

/-- The number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- The number of tourists who came on their own -/
def solo_tourists : ℕ := 13

/-- The number of tourists who did not bring anyone -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_theorem :
  ∃ (total_tourists : ℕ),
    total_tourists = solo_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists = 29 := by
  sorry

end NUMINAMATH_CALUDE_egypt_promotion_theorem_l83_8315


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l83_8399

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f) x₀ = 3 →
  x₀ = 1 ∨ x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l83_8399


namespace NUMINAMATH_CALUDE_bus_remaining_distance_l83_8386

def distance_between_points (z : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
  (z / 2) / (z - 19.2) = x ∧
  (z - 12) / (z / 2) = x

theorem bus_remaining_distance (z : ℝ) (h : distance_between_points z) :
  z - z * (4/5) = 6.4 :=
sorry

end NUMINAMATH_CALUDE_bus_remaining_distance_l83_8386


namespace NUMINAMATH_CALUDE_joggers_meeting_time_l83_8350

structure Jogger where
  name : String
  lap_time : ℕ

def lisa_effective_lap_time (lisa_lap_time : ℕ) (break_time : ℕ) : ℚ :=
  (2 * lisa_lap_time + break_time) / 2

def earliest_meeting_time (betty : Jogger) (charles : Jogger) (lisa : Jogger) (lisa_break_time : ℕ) : ℕ :=
  let lisa_eff_time := lisa_effective_lap_time lisa.lap_time lisa_break_time
  let lcm_betty_charles := Nat.lcm betty.lap_time charles.lap_time
  Nat.lcm lcm_betty_charles (Nat.lcm lisa.lap_time lisa_break_time) / 2

theorem joggers_meeting_time 
  (betty : Jogger)
  (charles : Jogger)
  (lisa : Jogger)
  (lisa_break_time : ℕ)
  (h_betty : betty.lap_time = 5)
  (h_charles : charles.lap_time = 8)
  (h_lisa : lisa.lap_time = 9)
  (h_lisa_break : lisa_break_time = 3) :
  earliest_meeting_time betty charles lisa lisa_break_time = 420 :=
by sorry

end NUMINAMATH_CALUDE_joggers_meeting_time_l83_8350


namespace NUMINAMATH_CALUDE_final_weight_is_16_l83_8379

/-- The weight of the box after each step of adding ingredients --/
structure BoxWeight where
  initial : ℝ
  afterBrownies : ℝ
  afterMoreJellyBeans : ℝ
  final : ℝ

/-- The process of creating the care package --/
def createCarePackage : BoxWeight :=
  { initial := 2,
    afterBrownies := 2 * 3,
    afterMoreJellyBeans := 2 * 3 + 2,
    final := (2 * 3 + 2) * 2 }

/-- The theorem stating the final weight of the care package --/
theorem final_weight_is_16 :
  (createCarePackage.final : ℝ) = 16 := by sorry

end NUMINAMATH_CALUDE_final_weight_is_16_l83_8379


namespace NUMINAMATH_CALUDE_inequality_equivalence_l83_8388

theorem inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -6 ↔ x ∈ Set.Ioo (-9/2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l83_8388


namespace NUMINAMATH_CALUDE_binomial_12_9_l83_8363

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l83_8363


namespace NUMINAMATH_CALUDE_unique_pairs_sum_product_l83_8338

theorem unique_pairs_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ + y₁ = S ∧ x₁ * y₁ = P) ∧
    (x₂ + y₂ = S ∧ x₂ * y₂ = P) ∧
    x₁ = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₁ = S - x₁ ∧
    x₂ = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₂ = S - x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_sum_product_l83_8338


namespace NUMINAMATH_CALUDE_saree_price_calculation_l83_8392

theorem saree_price_calculation (final_price : ℝ) 
  (h : final_price = 378.675) : ∃ (original_price : ℝ), 
  original_price * 0.85 * 0.90 = final_price ∧ 
  original_price = 495 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l83_8392


namespace NUMINAMATH_CALUDE_cloth_selling_price_l83_8387

/-- Calculates the total selling price of cloth given the quantity, cost price, and loss per metre. -/
def totalSellingPrice (quantity : ℕ) (costPrice lossPerMetre : ℚ) : ℚ :=
  quantity * (costPrice - lossPerMetre)

/-- Proves that the total selling price for 500 metres of cloth with a cost price of 41 and a loss of 5 per metre is 18000. -/
theorem cloth_selling_price :
  totalSellingPrice 500 41 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l83_8387


namespace NUMINAMATH_CALUDE_binary_table_theorem_l83_8332

/-- Represents a table filled with 0s and 1s -/
def BinaryTable := List (List Bool)

/-- Checks if all rows in the table are unique -/
def allRowsUnique (table : BinaryTable) : Prop :=
  ∀ i j, i ≠ j → table.get! i ≠ table.get! j

/-- Checks if any 4×2 sub-table has two identical rows -/
def anySubTableHasTwoIdenticalRows (table : BinaryTable) : Prop :=
  ∀ c₁ c₂ r₁ r₂ r₃ r₄, 
    c₁ < table.head!.length → c₂ < table.head!.length → c₁ ≠ c₂ →
    r₁ < table.length → r₂ < table.length → r₃ < table.length → r₄ < table.length →
    r₁ ≠ r₂ → r₁ ≠ r₃ → r₁ ≠ r₄ → r₂ ≠ r₃ → r₂ ≠ r₄ → r₃ ≠ r₄ →
    ∃ i j, i ≠ j ∧ 
      (table.get! i).get! c₁ = (table.get! j).get! c₁ ∧
      (table.get! i).get! c₂ = (table.get! j).get! c₂

/-- Checks if a column has exactly one occurrence of a number -/
def columnHasExactlyOneOccurrence (table : BinaryTable) (col : Nat) : Prop :=
  (table.map (λ row => row.get! col)).count true = 1 ∨
  (table.map (λ row => row.get! col)).count false = 1

theorem binary_table_theorem (table : BinaryTable) 
  (h1 : allRowsUnique table)
  (h2 : anySubTableHasTwoIdenticalRows table) :
  ∃ col, columnHasExactlyOneOccurrence table col := by
  sorry


end NUMINAMATH_CALUDE_binary_table_theorem_l83_8332


namespace NUMINAMATH_CALUDE_paige_homework_problem_l83_8334

/-- The number of problems Paige has left to do for homework -/
def problems_left (math science history language_arts finished_at_school unfinished_math : ℕ) : ℕ :=
  math + science + history + language_arts - finished_at_school + unfinished_math

theorem paige_homework_problem :
  problems_left 43 12 10 5 44 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_problem_l83_8334


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l83_8314

theorem subtraction_of_fractions : (12 : ℚ) / 30 - 1 / 7 = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l83_8314


namespace NUMINAMATH_CALUDE_distance_before_break_l83_8316

/-- Proves the distance walked before the break given initial, final, and total distances -/
theorem distance_before_break 
  (initial_distance : ℕ) 
  (final_distance : ℕ) 
  (total_distance : ℕ) 
  (h1 : initial_distance = 3007)
  (h2 : final_distance = 840)
  (h3 : total_distance = 6030) :
  total_distance - (initial_distance + final_distance) = 2183 := by
  sorry

#check distance_before_break

end NUMINAMATH_CALUDE_distance_before_break_l83_8316


namespace NUMINAMATH_CALUDE_workshop_production_theorem_l83_8393

/-- Represents the factory workshop setup and production requirements -/
structure Workshop where
  total_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  type_a_required : ℕ
  type_b_required : ℕ
  type_a_cost : ℕ
  type_b_cost : ℕ

/-- Calculates the number of workers assigned to type A parts -/
def workers_for_type_a (w : Workshop) : ℕ :=
  sorry

/-- Calculates the total processing cost for all workers in one day -/
def total_processing_cost (w : Workshop) : ℕ :=
  sorry

/-- The main theorem stating the correct number of workers for type A and total cost -/
theorem workshop_production_theorem (w : Workshop) 
  (h1 : w.total_workers = 50)
  (h2 : w.type_a_production = 30)
  (h3 : w.type_b_production = 20)
  (h4 : w.type_a_required = 7)
  (h5 : w.type_b_required = 2)
  (h6 : w.type_a_cost = 10)
  (h7 : w.type_b_cost = 12) :
  workers_for_type_a w = 35 ∧ total_processing_cost w = 14100 :=
by sorry

end NUMINAMATH_CALUDE_workshop_production_theorem_l83_8393


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_110_l83_8317

theorem largest_multiple_of_9_less_than_110 : 
  ∀ n : ℕ, n % 9 = 0 → n < 110 → n ≤ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_110_l83_8317


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l83_8365

theorem sqrt_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 - x) + Real.sqrt x
  ∀ x : ℝ, f x = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l83_8365


namespace NUMINAMATH_CALUDE_shaded_area_l83_8362

/-- The area of the shaded part in a figure containing an equilateral triangle and a regular hexagon -/
theorem shaded_area (triangle_area hexagon_area : ℝ) : 
  triangle_area = 960 →
  hexagon_area = 840 →
  ∃ (shaded_area : ℝ), shaded_area = 735 :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_l83_8362


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l83_8397

/-- Given that point A is an intersection point of y = ax and y = (4-a)/x with x-coordinate 1,
    prove that the y-coordinate of A is 2. -/
theorem intersection_point_y_coordinate (a : ℝ) :
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1) →
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1 ∧ A.2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l83_8397


namespace NUMINAMATH_CALUDE_triangle_properties_l83_8355

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l83_8355


namespace NUMINAMATH_CALUDE_existence_of_57_multiple_non_existence_of_58_multiple_l83_8330

/-- Removes the first digit of a positive integer -/
def removeFirstDigit (n : ℕ) : ℕ := sorry

/-- Checks if a number satisfies the condition A = k * B, where B is A with its first digit removed -/
def satisfiesCondition (A : ℕ) (k : ℕ) : Prop :=
  A = k * removeFirstDigit A

theorem existence_of_57_multiple :
  ∃ A : ℕ, A > 0 ∧ satisfiesCondition A 57 := by sorry

theorem non_existence_of_58_multiple :
  ¬∃ A : ℕ, A > 0 ∧ satisfiesCondition A 58 := by sorry

end NUMINAMATH_CALUDE_existence_of_57_multiple_non_existence_of_58_multiple_l83_8330


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l83_8366

/-- Given a paint mixture with ratio blue:green:white as 5:3:7,
    prove that using 21 quarts of white paint requires 9 quarts of green paint. -/
theorem paint_mixture_ratio (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : green = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l83_8366


namespace NUMINAMATH_CALUDE_custom_op_example_l83_8373

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem custom_op_example : custom_op 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l83_8373


namespace NUMINAMATH_CALUDE_balloons_left_after_distribution_l83_8383

def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def num_friends : ℕ := 10

theorem balloons_left_after_distribution :
  (red_balloons + blue_balloons + green_balloons + yellow_balloons) % num_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_balloons_left_after_distribution_l83_8383


namespace NUMINAMATH_CALUDE_counterexample_exists_l83_8394

theorem counterexample_exists : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬(Nat.Prime n) ∧ 
  ¬(Nat.Prime (n + 2)) ∧ 
  n = 14 :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l83_8394


namespace NUMINAMATH_CALUDE_order_of_abc_l83_8368

theorem order_of_abc : 
  let a := (Real.exp 0.6)⁻¹
  let b := 0.4
  let c := (Real.log 1.4) / 1.4
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l83_8368


namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l83_8311

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

-- Define the fractions in their respective bases
def fraction1_numerator : List Nat := [4, 6, 2]
def fraction1_denominator : List Nat := [2, 1]
def fraction2_numerator : List Nat := [4, 4, 1]
def fraction2_denominator : List Nat := [3, 3]

-- Define the bases
def base1 : Nat := 8
def base2 : Nat := 4
def base3 : Nat := 5

-- State the theorem
theorem base_conversion_sum_equality :
  (baseToDecimal fraction1_numerator base1 / baseToDecimal fraction1_denominator base2 : ℚ) +
  (baseToDecimal fraction2_numerator base3 / baseToDecimal fraction2_denominator base2 : ℚ) =
  499 / 15 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l83_8311


namespace NUMINAMATH_CALUDE_bike_ride_percentage_increase_l83_8300

theorem bike_ride_percentage_increase (d1 d2 d3 : ℝ) : 
  d2 = 12 →                   -- Second hour distance is 12 miles
  d2 = 1.2 * d1 →             -- Second hour is 20% farther than first hour
  d1 + d2 + d3 = 37 →         -- Total distance is 37 miles
  (d3 - d2) / d2 * 100 = 25   -- Percentage increase from second to third hour is 25%
  := by sorry

end NUMINAMATH_CALUDE_bike_ride_percentage_increase_l83_8300


namespace NUMINAMATH_CALUDE_modulus_of_z_l83_8382

theorem modulus_of_z (z : ℂ) (h : z / (Real.sqrt 3 - Complex.I) = 1 + Real.sqrt 3 * Complex.I) : 
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l83_8382


namespace NUMINAMATH_CALUDE_walkers_meet_at_corner_d_l83_8324

/-- Represents the corners of the rectangular area -/
inductive Corner
| A
| B
| C
| D

/-- Represents the rectangular area -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a person walking along the perimeter -/
structure Walker where
  speed : ℚ
  startCorner : Corner
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The meeting point of two walkers -/
def meetingPoint (rect : Rectangle) (walker1 walker2 : Walker) : Corner :=
  sorry

/-- The theorem to be proved -/
theorem walkers_meet_at_corner_d 
  (rect : Rectangle)
  (jane hector : Walker)
  (h_rect_dims : rect.length = 10 ∧ rect.width = 4)
  (h_start : jane.startCorner = Corner.A ∧ hector.startCorner = Corner.A)
  (h_directions : jane.direction = false ∧ hector.direction = true)
  (h_speeds : jane.speed = 2 * hector.speed) :
  meetingPoint rect jane hector = Corner.D :=
sorry

end NUMINAMATH_CALUDE_walkers_meet_at_corner_d_l83_8324


namespace NUMINAMATH_CALUDE_randy_piano_expert_age_l83_8390

/-- Proves that Randy must start practicing piano at age 12 to become an expert by age 20 --/
theorem randy_piano_expert_age (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) 
  (expert_hours : ℕ) (target_age : ℕ) : 
  hours_per_day = 5 →
  days_per_week = 5 →
  weeks_per_year = 50 →
  expert_hours = 10000 →
  target_age = 20 →
  target_age - (expert_hours / (hours_per_day * days_per_week * weeks_per_year)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_randy_piano_expert_age_l83_8390


namespace NUMINAMATH_CALUDE_inequality_proof_l83_8369

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) + 
  (2*b + c + a)^2 / (2*b^2 + (c + a)^2) + 
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l83_8369


namespace NUMINAMATH_CALUDE_kobe_initial_order_proof_l83_8313

/-- Represents the number of pieces of fried chicken Kobe initially ordered -/
def kobe_initial_order : ℕ := 5

/-- Represents the number of pieces of fried chicken Pau initially ordered -/
def pau_initial_order : ℕ := 2 * kobe_initial_order

/-- Represents the total number of pieces of fried chicken Pau ate -/
def pau_total : ℕ := 20

theorem kobe_initial_order_proof :
  pau_initial_order + pau_initial_order = pau_total :=
by sorry

end NUMINAMATH_CALUDE_kobe_initial_order_proof_l83_8313


namespace NUMINAMATH_CALUDE_time_after_317h_58m_30s_l83_8348

def hours_to_12hour_clock (h : ℕ) : ℕ :=
  h % 12

def add_time (start_hour start_minute start_second : ℕ) 
             (add_hours add_minutes add_seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_seconds := start_second + add_seconds
  let total_minutes := start_minute + add_minutes + total_seconds / 60
  let total_hours := start_hour + add_hours + total_minutes / 60
  (hours_to_12hour_clock total_hours, total_minutes % 60, total_seconds % 60)

theorem time_after_317h_58m_30s : 
  let (A, B, C) := add_time 3 0 0 317 58 30
  A + B + C = 96 := by sorry

end NUMINAMATH_CALUDE_time_after_317h_58m_30s_l83_8348


namespace NUMINAMATH_CALUDE_egyptian_fraction_1991_l83_8336

theorem egyptian_fraction_1991 : ∃ (k l m : ℕ), 
  Odd k ∧ Odd l ∧ Odd m ∧ 
  (1 : ℚ) / 1991 = 1 / k + 1 / l + 1 / m := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_1991_l83_8336


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l83_8319

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l83_8319


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l83_8381

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + I) / (1 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l83_8381


namespace NUMINAMATH_CALUDE_white_balls_in_box_l83_8398

theorem white_balls_in_box (orange_balls black_balls : ℕ) 
  (prob_not_orange_or_white : ℚ) (white_balls : ℕ) : 
  orange_balls = 8 → 
  black_balls = 7 → 
  prob_not_orange_or_white = 38095238095238093 / 100000000000000000 →
  (black_balls : ℚ) / (orange_balls + black_balls + white_balls : ℚ) = prob_not_orange_or_white →
  white_balls = 3 := by
sorry

end NUMINAMATH_CALUDE_white_balls_in_box_l83_8398


namespace NUMINAMATH_CALUDE_function_machine_output_l83_8347

/-- Function machine operation -/
def function_machine (input : ℕ) : ℕ :=
  let doubled := input * 2
  if doubled ≤ 15 then
    doubled * 3
  else
    doubled * 3

/-- Theorem: The function machine outputs 90 for an input of 15 -/
theorem function_machine_output : function_machine 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l83_8347


namespace NUMINAMATH_CALUDE_sum_f_negative_l83_8384

/-- The function f(x) = 2x³ + 4x -/
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

/-- Theorem: Given f(x) = 2x³ + 4x and a + b < 0, b + c < 0, c + a < 0, then f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) :
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l83_8384


namespace NUMINAMATH_CALUDE_percentage_to_total_l83_8337

/-- If 25% of an amount is 75 rupees, then the total amount is 300 rupees. -/
theorem percentage_to_total (amount : ℝ) : (25 / 100) * amount = 75 → amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_total_l83_8337


namespace NUMINAMATH_CALUDE_total_students_l83_8357

theorem total_students (scavenger_hunting : ℕ) (skiing : ℕ) : 
  scavenger_hunting = 4000 → 
  skiing = 2 * scavenger_hunting → 
  scavenger_hunting + skiing = 12000 := by
sorry

end NUMINAMATH_CALUDE_total_students_l83_8357


namespace NUMINAMATH_CALUDE_certain_number_for_prime_squared_l83_8351

theorem certain_number_for_prime_squared (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃! n : ℕ, (p^2 + n) % 12 = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_for_prime_squared_l83_8351


namespace NUMINAMATH_CALUDE_thursday_loaves_l83_8385

def bakery_sequence : List ℕ := [5, 11, 10, 14, 19, 25]

def alternating_differences (seq : List ℕ) : List ℕ :=
  List.zipWith (λ a b => b - a) seq (seq.tail)

theorem thursday_loaves :
  let seq := bakery_sequence
  let diffs := alternating_differences seq
  (seq[1] = 11 ∧
   diffs[0] = diffs[2] + 1 ∧
   diffs[1] = diffs[3] - 1 ∧
   diffs[2] = diffs[4] + 1) →
  seq[1] = 11 := by sorry

end NUMINAMATH_CALUDE_thursday_loaves_l83_8385


namespace NUMINAMATH_CALUDE_square_root_of_625_l83_8323

theorem square_root_of_625 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 625) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_625_l83_8323


namespace NUMINAMATH_CALUDE_stating_rest_day_alignment_l83_8310

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of consecutive work days -/
def workDays : ℕ := 8

/-- Represents the number of consecutive rest days -/
def restDays : ℕ := 2

/-- Represents the total days in a work-rest cycle -/
def cycleDays : ℕ := workDays + restDays

/-- 
Theorem stating that it takes 7 weeks for the rest day to align with Sunday again 
given the work-rest cycle and initial conditions
-/
theorem rest_day_alignment : 
  ∀ n : ℕ, 
  (n * daysInWeek) % cycleDays = restDays - 1 → 
  n = 7 := by sorry

end NUMINAMATH_CALUDE_stating_rest_day_alignment_l83_8310


namespace NUMINAMATH_CALUDE_equation_solution_l83_8339

theorem equation_solution (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l83_8339


namespace NUMINAMATH_CALUDE_smallest_percentage_both_drinks_l83_8326

/-- The percentage of adults who drink coffee -/
def coffee_drinkers : ℝ := 90

/-- The percentage of adults who drink tea -/
def tea_drinkers : ℝ := 85

/-- The smallest possible percentage of adults who drink both coffee and tea -/
def both_drinkers : ℝ := 75

theorem smallest_percentage_both_drinks (coffee_drinkers tea_drinkers both_drinkers : ℝ) 
  (h1 : coffee_drinkers = 90) 
  (h2 : tea_drinkers = 85) : 
  both_drinkers ≥ 75 ∧ ∃ (x : ℝ), x ≥ 75 ∧ 
  coffee_drinkers + tea_drinkers - x ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_percentage_both_drinks_l83_8326


namespace NUMINAMATH_CALUDE_solution_set_l83_8352

theorem solution_set (x : ℝ) : 4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioc (5/2) (20/7) :=
  sorry

end NUMINAMATH_CALUDE_solution_set_l83_8352
