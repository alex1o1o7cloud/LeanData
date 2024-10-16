import Mathlib

namespace NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_l2722_272242

theorem line_through_first_and_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_l2722_272242


namespace NUMINAMATH_CALUDE_log_problem_l2722_272232

theorem log_problem (x k : ℝ) 
  (h1 : Real.log x * (Real.log 10 / Real.log k) = 4)
  (h2 : k^2 = 100) : 
  x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2722_272232


namespace NUMINAMATH_CALUDE_xiang_lake_one_millionth_closest_to_study_room_l2722_272263

/-- The combined area of Phase I and Phase II of Xiang Lake in square kilometers -/
def xiang_lake_area : ℝ := 10.6

/-- One million as a real number -/
def one_million : ℝ := 1000000

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

/-- Approximate area of a typical study room in square meters -/
def typical_study_room_area : ℝ := 10.6

/-- Theorem stating that one-millionth of Xiang Lake's area is closest to a typical study room's area -/
theorem xiang_lake_one_millionth_closest_to_study_room :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |xiang_lake_area * km2_to_m2 / one_million - typical_study_room_area| < ε :=
sorry

end NUMINAMATH_CALUDE_xiang_lake_one_millionth_closest_to_study_room_l2722_272263


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l2722_272219

/-- Represents a linear regression model for height based on age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts the height for a given age using the model -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_at_10 (model : HeightModel)
  (h_slope : model.slope = 7.19)
  (h_intercept : model.intercept = 73.93) :
  ∃ ε > 0, |predict_height model 10 - 145.83| < ε :=
sorry

#check height_prediction_at_10

end NUMINAMATH_CALUDE_height_prediction_at_10_l2722_272219


namespace NUMINAMATH_CALUDE_probability_sum_10_l2722_272294

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when throwing two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The total number of possible outcomes -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The sum of two numbers -/
def sum (pair : ℕ × ℕ) : ℕ := pair.1 + pair.2

/-- The set of favorable outcomes (sum equals 10) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun pair => sum pair = 10)

/-- The number of favorable outcomes -/
def numFavorableOutcomes : ℕ := favorableOutcomes.card

theorem probability_sum_10 :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 5 / 36 := by
  sorry

#eval numFavorableOutcomes -- Should output 5
#eval totalOutcomes -- Should output 36

end NUMINAMATH_CALUDE_probability_sum_10_l2722_272294


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l2722_272260

theorem rectangular_field_fence (L W : ℝ) : 
  L > 0 ∧ W > 0 →  -- Positive dimensions
  L * W = 210 →    -- Area condition
  L + 2 * W = 41 → -- Fencing condition
  L = 21 :=        -- Conclusion: uncovered side length
by sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l2722_272260


namespace NUMINAMATH_CALUDE_f_composition_value_l2722_272228

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l2722_272228


namespace NUMINAMATH_CALUDE_journey_time_comparison_l2722_272280

/-- Represents the speed of walking -/
def walking_speed : ℝ := 1

/-- Represents the speed of cycling -/
def cycling_speed : ℝ := 2 * walking_speed

/-- Represents the speed of the bus -/
def bus_speed : ℝ := 5 * cycling_speed

/-- Represents half the total journey distance -/
def half_journey : ℝ := 1

theorem journey_time_comparison : 
  (half_journey / bus_speed + half_journey / walking_speed) > (2 * half_journey) / cycling_speed :=
sorry

end NUMINAMATH_CALUDE_journey_time_comparison_l2722_272280


namespace NUMINAMATH_CALUDE_polynomial_value_l2722_272271

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2722_272271


namespace NUMINAMATH_CALUDE_vector_simplification_l2722_272276

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) : 
  ((B - A) - (D - C)) - ((C - A) - (D - B)) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l2722_272276


namespace NUMINAMATH_CALUDE_cakes_baked_yesterday_prove_cakes_baked_yesterday_l2722_272224

def cakes_baked_today : ℕ := 5
def cakes_sold_dinner : ℕ := 6
def cakes_left : ℕ := 2

theorem cakes_baked_yesterday : ℕ :=
  cakes_sold_dinner - cakes_baked_today + cakes_left

theorem prove_cakes_baked_yesterday :
  cakes_baked_yesterday = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_baked_yesterday_prove_cakes_baked_yesterday_l2722_272224


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l2722_272241

def original_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

def new_ellipse (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

def same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ c : ℝ, (∀ x y : ℝ, e1 x y ↔ (x - c)^2/(9 - 4) + y^2/4 = 1) ∧
           (∀ x y : ℝ, e2 x y ↔ (x - c)^2/(15 - 10) + y^2/10 = 1)

theorem ellipse_equation_proof :
  (new_ellipse 3 (-2)) ∧
  (same_foci original_ellipse new_ellipse) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l2722_272241


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2722_272201

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2722_272201


namespace NUMINAMATH_CALUDE_projection_closed_l2722_272278

open Set
open Topology

-- Define the projection function
def proj_y (p : ℝ × ℝ) : ℝ := p.2

-- State the theorem
theorem projection_closed {a b : ℝ} {S : Set (ℝ × ℝ)} 
  (hS : IsClosed S) 
  (hSub : S ⊆ {p : ℝ × ℝ | a < p.1 ∧ p.1 < b}) :
  IsClosed (proj_y '' S) := by
  sorry

end NUMINAMATH_CALUDE_projection_closed_l2722_272278


namespace NUMINAMATH_CALUDE_tuna_salmon_ratio_l2722_272282

/-- Proves that the ratio of tuna weight to salmon weight is 2:1 given specific conditions --/
theorem tuna_salmon_ratio (trout_weight salmon_weight tuna_weight : ℝ) : 
  trout_weight = 200 →
  salmon_weight = trout_weight * 1.5 →
  trout_weight + salmon_weight + tuna_weight = 1100 →
  tuna_weight / salmon_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuna_salmon_ratio_l2722_272282


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2722_272261

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 4) :
  (1 / a + 4 / b + 9 / c) ≥ 9 ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 4 ∧ 1 / a₀ + 4 / b₀ + 9 / c₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2722_272261


namespace NUMINAMATH_CALUDE_hawks_score_l2722_272221

theorem hawks_score (total_points eagles_margin hawks_min_score : ℕ) 
  (h1 : total_points = 82)
  (h2 : eagles_margin = 18)
  (h3 : hawks_min_score = 9)
  (h4 : ∃ (hawks_score : ℕ), 
    hawks_score ≥ hawks_min_score ∧ 
    hawks_score + (hawks_score + eagles_margin) = total_points) :
  ∃ (hawks_score : ℕ), hawks_score = 32 :=
by sorry

end NUMINAMATH_CALUDE_hawks_score_l2722_272221


namespace NUMINAMATH_CALUDE_prime_sum_and_squares_l2722_272206

theorem prime_sum_and_squares (p q r s : ℕ) : 
  p.Prime ∧ q.Prime ∧ r.Prime ∧ s.Prime ∧  -- p, q, r, s are prime
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  (p + q + r + s).Prime ∧  -- their sum is prime
  ∃ a, p^2 + q*s = a^2 ∧  -- p² + qs is a perfect square
  ∃ b, p^2 + q*r = b^2  -- p² + qr is a perfect square
  →
  ((p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_and_squares_l2722_272206


namespace NUMINAMATH_CALUDE_min_radius_value_l2722_272247

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on the circle satisfying the given condition -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2
  condition : point.2^2 ≥ 4 * point.1

/-- The theorem stating the minimum value of r -/
theorem min_radius_value (c : Circle) (p : PointOnCircle c) :
  c.center.1 = c.radius + 1 ∧ c.center.2 = 0 → c.radius ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_radius_value_l2722_272247


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l2722_272277

theorem decimal_fraction_equality (b : ℕ+) : 
  (5 * b + 17 : ℚ) / (7 * b + 12) = 85 / 100 ↔ b = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l2722_272277


namespace NUMINAMATH_CALUDE_divisibility_by_six_l2722_272297

theorem divisibility_by_six (a x : ℤ) : 
  (∃ k : ℤ, a * (x^3 + a^2 * x^2 + a^2 - 1) = 6 * k) ↔ 
  (∃ t : ℤ, x = 3 * t ∨ x = 3 * t - a^2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l2722_272297


namespace NUMINAMATH_CALUDE_fraction_simplification_l2722_272227

theorem fraction_simplification (x : ℝ) : 
  (2*x^2 + 3)/4 - (5 - 4*x^2)/6 = (14*x^2 - 1)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2722_272227


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2722_272257

theorem fractional_equation_solution :
  ∃ x : ℚ, x ≠ 0 ∧ x ≠ -3 ∧ (1 / x = 6 / (x + 3)) ∧ x = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2722_272257


namespace NUMINAMATH_CALUDE_meeting_day_is_wednesday_l2722_272262

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brothers
inductive Brother
| Tralalala
| Trulala

def lies (b : Brother) (d : Day) : Prop :=
  match b with
  | Brother.Tralalala => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday
  | Brother.Trulala => d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

theorem meeting_day_is_wednesday :
  ∃ (b1 b2 : Brother) (d : Day),
    b1 ≠ b2 ∧
    (lies b1 Day.Saturday ↔ lies b1 d) ∧
    (lies b2 (next_day d) ↔ ¬(lies b2 d)) ∧
    (lies b1 Day.Sunday ↔ lies b1 d) ∧
    d = Day.Wednesday :=
  sorry


end NUMINAMATH_CALUDE_meeting_day_is_wednesday_l2722_272262


namespace NUMINAMATH_CALUDE_hockey_league_teams_l2722_272248

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 17

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1360

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l2722_272248


namespace NUMINAMATH_CALUDE_initial_milk_amount_l2722_272205

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                             (target_water_content : Real)
                             (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  target_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk * initial_water_content = 
      (initial_milk + pure_milk_added) * target_water_content ∧
    initial_milk = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_amount_l2722_272205


namespace NUMINAMATH_CALUDE_max_successful_teams_16_l2722_272256

/-- Represents a football championship --/
structure Championship :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Definition of a successful team --/
def is_successful (c : Championship) (points : ℕ) : Prop :=
  points ≥ (c.teams - 1) * c.points_for_win / 2

/-- The maximum number of successful teams in the championship --/
def max_successful_teams (c : Championship) : ℕ := sorry

/-- The main theorem --/
theorem max_successful_teams_16 :
  ∀ c : Championship,
    c.teams = 16 ∧
    c.points_for_win = 3 ∧
    c.points_for_draw = 1 ∧
    c.points_for_loss = 0 →
    max_successful_teams c = 15 := by sorry

end NUMINAMATH_CALUDE_max_successful_teams_16_l2722_272256


namespace NUMINAMATH_CALUDE_eliot_votes_l2722_272254

/-- Given the vote distribution in a school election, prove that Eliot got 160 votes. -/
theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 → 
  shaun_votes = 5 * randy_votes → 
  eliot_votes = 2 * shaun_votes → 
  eliot_votes = 160 := by
sorry


end NUMINAMATH_CALUDE_eliot_votes_l2722_272254


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2722_272203

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2722_272203


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l2722_272281

theorem point_on_unit_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2*t^3) / (t^3 + 1)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l2722_272281


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2722_272288

/-- 
For a quadratic equation kx² + 2x - 1 = 0 to have two distinct real roots,
k must satisfy k > -1 and k ≠ 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ - 1 = 0 ∧ k * x₂^2 + 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2722_272288


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_parallelepiped_l2722_272223

/-- The surface area of a sphere that circumscribes a rectangular parallelepiped with edge lengths 3, 4, and 5 is equal to 50π. -/
theorem sphere_surface_area_of_inscribed_parallelepiped (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diameter := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diameter / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_parallelepiped_l2722_272223


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_plus_minus_one_l2722_272272

theorem fraction_zero_implies_x_plus_minus_one (x : ℝ) :
  (x^2 - 1) / x = 0 → x ≠ 0 → (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_plus_minus_one_l2722_272272


namespace NUMINAMATH_CALUDE_part_one_part_two_l2722_272238

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ Real.cos t.B = 3/5

-- Part 1
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_b : t.b = 4) :
  Real.sin t.A = 2/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 4) :
  t.b = Real.sqrt 17 ∧ t.c = 5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2722_272238


namespace NUMINAMATH_CALUDE_income_mean_difference_l2722_272287

theorem income_mean_difference (T : ℝ) (n : ℕ) : 
  n = 500 → 
  (T + 1100000) / n - (T + 110000) / n = 1980 :=
by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l2722_272287


namespace NUMINAMATH_CALUDE_sum_a_d_equals_one_l2722_272284

theorem sum_a_d_equals_one (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_one_l2722_272284


namespace NUMINAMATH_CALUDE_sachin_gain_is_487_50_l2722_272215

/-- Calculates Sachin's gain in one year based on given borrowing and lending conditions. -/
def sachinsGain (X R1 R2 R3 : ℚ) : ℚ :=
  let interestFromRahul := X * R2 / 100
  let interestFromRavi := X * R3 / 100
  let interestPaid := X * R1 / 100
  interestFromRahul + interestFromRavi - interestPaid

/-- Theorem stating that Sachin's gain in one year is 487.50 rupees. -/
theorem sachin_gain_is_487_50 :
  sachinsGain 5000 4 (25/4) (15/2) = 487.5 := by
  sorry

#eval sachinsGain 5000 4 (25/4) (15/2)

end NUMINAMATH_CALUDE_sachin_gain_is_487_50_l2722_272215


namespace NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l2722_272274

open Real

theorem floor_x_floor_x_eq_48 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 48 ↔ 8 ≤ x ∧ x < 49 / 6 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l2722_272274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2722_272214

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 6 4 n = 206 ∧ n = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2722_272214


namespace NUMINAMATH_CALUDE_seven_from_five_twos_l2722_272290

theorem seven_from_five_twos : ∃ (a b c d e f g h i j : ℕ),
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
  (f = 2 ∧ g = 2 ∧ h = 2 ∧ i = 2 ∧ j = 2) ∧
  (a * b * c - d / e = 7) ∧
  (f + g + h + i / j = 7) ∧
  ((10 * a + b) / c - d * e = 7) :=
by sorry

end NUMINAMATH_CALUDE_seven_from_five_twos_l2722_272290


namespace NUMINAMATH_CALUDE_blue_parrots_count_l2722_272225

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 160 → 
  green_fraction = 5/8 → 
  (1 - green_fraction) * total = 60 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l2722_272225


namespace NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_p_q_l2722_272273

theorem log_ten_seven_in_terms_of_p_q (p q : ℝ) 
  (hp : Real.log 3 / Real.log 4 = p)
  (hq : Real.log 7 / Real.log 5 = q) :
  Real.log 7 / Real.log 10 = (2 * p * q + 2 * p) / (1 + 2 * p) := by
  sorry

end NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_p_q_l2722_272273


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l2722_272249

/-- Given that x³ varies inversely with y⁴, and x = 2 when y = 4,
    prove that x³ = 1/2 when y = 8 -/
theorem inverse_variation_cube_fourth (k : ℝ) :
  (∀ x y : ℝ, x^3 * y^4 = k) →
  (2^3 * 4^4 = k) →
  ∃ x : ℝ, x^3 * 8^4 = k ∧ x^3 = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l2722_272249


namespace NUMINAMATH_CALUDE_complex_magnitude_l2722_272202

theorem complex_magnitude (x y : ℝ) (h : x * (1 + Complex.I) = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2722_272202


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l2722_272239

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.monday_hours + schedule.tuesday_hours + 
                     schedule.wednesday_hours + schedule.thursday_hours + 
                     schedule.friday_hours
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  let sheila_schedule : WorkSchedule := {
    monday_hours := 8,
    tuesday_hours := 6,
    wednesday_hours := 8,
    thursday_hours := 6,
    friday_hours := 8,
    weekly_earnings := 288
  }
  hourly_rate sheila_schedule = 8 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l2722_272239


namespace NUMINAMATH_CALUDE_georges_initial_money_l2722_272255

theorem georges_initial_money (shirt_cost sock_cost money_left : ℕ) :
  shirt_cost = 24 →
  sock_cost = 11 →
  money_left = 65 →
  shirt_cost + sock_cost + money_left = 100 :=
by sorry

end NUMINAMATH_CALUDE_georges_initial_money_l2722_272255


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2722_272250

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  (f m < 0) → (f (m - 1) > 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2722_272250


namespace NUMINAMATH_CALUDE_walter_seal_time_l2722_272283

/-- The time Walter spends at the zoo -/
def total_time : ℕ := 260

/-- Walter's initial time spent looking at seals -/
def initial_seal_time : ℕ := 20

/-- Time spent looking at penguins -/
def penguin_time (s : ℕ) : ℕ := 8 * s

/-- Time spent looking at elephants -/
def elephant_time : ℕ := 13

/-- Time spent on second visit to seals -/
def second_seal_time (s : ℕ) : ℕ := s / 2

/-- Time spent at giraffe exhibit -/
def giraffe_time (s : ℕ) : ℕ := 3 * s

/-- Total time spent looking at seals -/
def total_seal_time (s : ℕ) : ℕ := s + (s / 2)

theorem walter_seal_time :
  total_seal_time initial_seal_time = 30 ∧
  initial_seal_time + penguin_time initial_seal_time + elephant_time +
  second_seal_time initial_seal_time + giraffe_time initial_seal_time = total_time :=
sorry

end NUMINAMATH_CALUDE_walter_seal_time_l2722_272283


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2722_272230

theorem quadratic_always_positive 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hseq : b / a = c / b) : 
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2722_272230


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2023_l2722_272293

theorem units_digit_of_7_to_2023 : 7^2023 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2023_l2722_272293


namespace NUMINAMATH_CALUDE_mrs_brown_payment_l2722_272265

/-- Calculates the final price after applying multiple discounts --/
def calculate_final_price (base_price : ℝ) (mother_discount : ℝ) (child_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  let price_after_mother := base_price * (1 - mother_discount)
  let price_after_child := price_after_mother * (1 - child_discount)
  price_after_child * (1 - vip_discount)

/-- Theorem stating that Mrs. Brown's final payment amount is $201.10 --/
theorem mrs_brown_payment : 
  let shoes_price : ℝ := 125
  let handbag_price : ℝ := 75
  let scarf_price : ℝ := 45
  let total_price : ℝ := shoes_price + handbag_price + scarf_price
  let mother_discount : ℝ := 0.10
  let child_discount : ℝ := 0.04
  let vip_discount : ℝ := 0.05
  calculate_final_price total_price mother_discount child_discount vip_discount = 201.10 := by
  sorry


end NUMINAMATH_CALUDE_mrs_brown_payment_l2722_272265


namespace NUMINAMATH_CALUDE_optimal_bus_rental_plan_l2722_272252

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  modelA : ℕ
  modelB : ℕ

/-- Calculates the total capacity of a bus rental plan -/
def totalCapacity (plan : BusRentalPlan) : ℕ :=
  40 * plan.modelA + 55 * plan.modelB

/-- Calculates the total cost of a bus rental plan -/
def totalCost (plan : BusRentalPlan) : ℕ :=
  600 * plan.modelA + 700 * plan.modelB

/-- Checks if a bus rental plan is valid -/
def isValidPlan (plan : BusRentalPlan) : Prop :=
  plan.modelA + plan.modelB = 10 ∧ 
  plan.modelA ≥ 1 ∧ 
  plan.modelB ≥ 1 ∧
  totalCapacity plan ≥ 502

/-- Theorem stating the properties of the optimal bus rental plan -/
theorem optimal_bus_rental_plan :
  ∃ (optimalPlan : BusRentalPlan),
    isValidPlan optimalPlan ∧
    optimalPlan.modelA = 3 ∧
    optimalPlan.modelB = 7 ∧
    totalCost optimalPlan = 6700 ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → totalCost plan ≥ totalCost optimalPlan) ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → plan.modelA ≤ 3) :=
  sorry


end NUMINAMATH_CALUDE_optimal_bus_rental_plan_l2722_272252


namespace NUMINAMATH_CALUDE_city_male_population_l2722_272299

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 800 →
  num_parts = 4 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 :=
by sorry

end NUMINAMATH_CALUDE_city_male_population_l2722_272299


namespace NUMINAMATH_CALUDE_y_value_approximation_l2722_272243

noncomputable def x : ℝ := 3.87

theorem y_value_approximation :
  let y := 2 * (Real.log x)^3 - (5 / 3)
  ∃ ε > 0, |y + 1.2613| < ε ∧ ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_y_value_approximation_l2722_272243


namespace NUMINAMATH_CALUDE_min_value_product_l2722_272240

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 33/4 := by
sorry

end NUMINAMATH_CALUDE_min_value_product_l2722_272240


namespace NUMINAMATH_CALUDE_digit_145_of_49_div_686_l2722_272296

/-- The decimal expansion of 49/686 has a period of 6 -/
def period : ℕ := 6

/-- The repeating sequence in the decimal expansion of 49/686 -/
def repeating_sequence : Fin 6 → ℕ
| 0 => 0
| 1 => 7
| 2 => 1
| 3 => 4
| 4 => 2
| 5 => 8

/-- The 145th digit after the decimal point in the decimal expansion of 49/686 is 8 -/
theorem digit_145_of_49_div_686 : 
  repeating_sequence ((145 - 1) % period) = 8 := by sorry

end NUMINAMATH_CALUDE_digit_145_of_49_div_686_l2722_272296


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2722_272236

-- Define the probability of having a boy or a girl
def p_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2722_272236


namespace NUMINAMATH_CALUDE_range_of_a_l2722_272237

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2722_272237


namespace NUMINAMATH_CALUDE_opposites_and_reciprocals_problem_l2722_272222

theorem opposites_and_reciprocals_problem 
  (a b x y : ℝ) 
  (h1 : a + b = 0)      -- a and b are opposites
  (h2 : x * y = 1)      -- x and y are reciprocals
  : 5 * |a + b| - 5 * x * y = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposites_and_reciprocals_problem_l2722_272222


namespace NUMINAMATH_CALUDE_prove_theta_value_l2722_272245

-- Define the angles in degrees
def angle_VEK : ℝ := 70
def angle_KEW : ℝ := 40
def angle_EVG : ℝ := 110

-- Define θ as a real number
def θ : ℝ := 40

-- Theorem statement
theorem prove_theta_value :
  angle_VEK = 70 ∧
  angle_KEW = 40 ∧
  angle_EVG = 110 →
  θ = 40 := by
  sorry


end NUMINAMATH_CALUDE_prove_theta_value_l2722_272245


namespace NUMINAMATH_CALUDE_percentage_calculation_l2722_272209

theorem percentage_calculation : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2722_272209


namespace NUMINAMATH_CALUDE_first_person_speed_l2722_272269

/-- Two persons walk in opposite directions for a given time, ending up at a specific distance apart. -/
def opposite_walk (x : ℝ) (time : ℝ) (distance : ℝ) : Prop :=
  (x + 7) * time = distance

/-- The theorem states that given the conditions of the problem, the speed of the first person is 6 km/hr. -/
theorem first_person_speed : ∃ x : ℝ, opposite_walk x 3.5 45.5 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_person_speed_l2722_272269


namespace NUMINAMATH_CALUDE_total_length_of_sticks_l2722_272218

/-- The total length of 5 sticks with specific length relationships -/
theorem total_length_of_sticks : ∀ (stick1 stick2 stick3 stick4 stick5 : ℝ),
  stick1 = 3 →
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick4 = stick3 / 2 →
  stick5 = 4 * stick4 →
  stick1 + stick2 + stick3 + stick4 + stick5 = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_total_length_of_sticks_l2722_272218


namespace NUMINAMATH_CALUDE_book_selection_combinations_l2722_272200

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of books in the library -/
def total_books : ℕ := 15

/-- The number of books to be selected -/
def selected_books : ℕ := 3

/-- Theorem: The number of ways to choose 3 books from 15 books is 455 -/
theorem book_selection_combinations :
  choose total_books selected_books = 455 := by sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l2722_272200


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2722_272234

noncomputable def f (θ : ℝ) : ℝ := (Real.sin θ) / (2 + Real.cos θ)

theorem derivative_f_at_zero :
  deriv f 0 = 1/3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2722_272234


namespace NUMINAMATH_CALUDE_inverse_value_of_symmetrical_function_l2722_272207

-- Define a function that is symmetrical about a point
def SymmetricalAboutPoint (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Define the existence of an inverse function
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_value_of_symmetrical_function
  (f : ℝ → ℝ)
  (h_sym : SymmetricalAboutPoint f (1, 2))
  (h_inv : HasInverse f)
  (h_f4 : f 4 = 0) :
  ∃ f_inv : ℝ → ℝ, HasInverse f ∧ f_inv 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_value_of_symmetrical_function_l2722_272207


namespace NUMINAMATH_CALUDE_tour_group_composition_l2722_272285

/-- Represents the number of people in a tour group -/
structure TourGroup where
  total : ℕ
  children : ℕ

/-- Represents the ticket prices -/
structure TicketPrices where
  adult : ℕ
  child : ℕ

/-- The main theorem statement -/
theorem tour_group_composition 
  (group_a group_b : TourGroup) 
  (prices : TicketPrices) : 
  (group_b.total = group_a.total + 4) →
  (group_a.total + group_b.total = 18 * (group_b.total - group_a.total)) →
  (group_b.children = 3 * group_a.children - 2) →
  (prices.adult = 100) →
  (prices.child = prices.adult * 3 / 5) →
  (prices.adult * (group_a.total - group_a.children) + prices.child * group_a.children = 
   prices.adult * (group_b.total - group_b.children) + prices.child * group_b.children) →
  (group_a.total = 34 ∧ group_a.children = 6 ∧ 
   group_b.total = 38 ∧ group_b.children = 16) :=
by sorry

end NUMINAMATH_CALUDE_tour_group_composition_l2722_272285


namespace NUMINAMATH_CALUDE_blocks_added_l2722_272258

/-- 
Given:
- initial_blocks: The initial number of blocks in Adolfo's tower
- final_blocks: The final number of blocks in Adolfo's tower

Prove that the number of blocks added is equal to the difference between 
the final and initial number of blocks.
-/
theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) : 
  final_blocks - initial_blocks = 30 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l2722_272258


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2722_272266

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : (a^6 + b^6) / (a + b)^6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2722_272266


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2722_272235

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -6 ∨ x > 2/3} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2722_272235


namespace NUMINAMATH_CALUDE_min_socks_theorem_l2722_272231

/-- Represents a collection of socks with at least 5 different colors -/
structure SockCollection where
  colors : Nat
  min_socks_per_color : Nat
  colors_ge_5 : colors ≥ 5
  min_socks_ge_40 : min_socks_per_color ≥ 40

/-- The smallest number of socks that must be selected to guarantee at least 15 pairs -/
def min_socks_for_15_pairs (sc : SockCollection) : Nat :=
  38

theorem min_socks_theorem (sc : SockCollection) :
  min_socks_for_15_pairs sc = 38 := by
  sorry

#check min_socks_theorem

end NUMINAMATH_CALUDE_min_socks_theorem_l2722_272231


namespace NUMINAMATH_CALUDE_puppy_feeding_theorem_l2722_272264

/-- Given the number of puppies, total portions of formula, and number of days,
    calculate the number of times each puppy should be fed per day. -/
def feeding_frequency (num_puppies : ℕ) (total_portions : ℕ) (num_days : ℕ) : ℕ :=
  (total_portions / num_days) / num_puppies

/-- Theorem stating that for 7 puppies, 105 portions of formula, and 5 days,
    the feeding frequency is 3 times per day. -/
theorem puppy_feeding_theorem :
  feeding_frequency 7 105 5 = 3 := by
  sorry

#eval feeding_frequency 7 105 5

end NUMINAMATH_CALUDE_puppy_feeding_theorem_l2722_272264


namespace NUMINAMATH_CALUDE_volunteer_distribution_l2722_272267

theorem volunteer_distribution (n : ℕ) (h : n = 5) :
  (n.choose 1) * ((n - 1).choose 2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l2722_272267


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2722_272226

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2722_272226


namespace NUMINAMATH_CALUDE_inequality_theta_range_l2722_272279

theorem inequality_theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  ∃ k : ℤ, θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theta_range_l2722_272279


namespace NUMINAMATH_CALUDE_oliver_stickers_l2722_272212

theorem oliver_stickers (initial_stickers : ℕ) (remaining_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : remaining_stickers = 54)
  (h3 : ∃ x : ℚ, 0 ≤ x ∧ x < 1 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor) :
  ∃ x : ℚ, x = 1/3 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor :=
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l2722_272212


namespace NUMINAMATH_CALUDE_mary_performance_l2722_272275

theorem mary_performance (total_days : ℕ) (adequate_rate : ℕ) (outstanding_rate : ℕ) (total_amount : ℕ) :
  total_days = 15 ∧ 
  adequate_rate = 4 ∧ 
  outstanding_rate = 7 ∧ 
  total_amount = 85 →
  ∃ (adequate_days outstanding_days : ℕ),
    adequate_days + outstanding_days = total_days ∧
    adequate_days * adequate_rate + outstanding_days * outstanding_rate = total_amount ∧
    outstanding_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_performance_l2722_272275


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2722_272220

/-- A parabola with vertex at (-2, 5) passing through (2, 9) has coefficients a = 1/4, b = 1, c = 6 -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 = a * (-2)^2 + b * (-2) + c) →
  (∀ x : ℝ, a * (x + 2)^2 + 5 = a * x^2 + b * x + c) →
  (9 = a * 2^2 + b * 2 + c) →
  (a = 1/4 ∧ b = 1 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2722_272220


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2722_272289

theorem arithmetic_and_geometric_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2722_272289


namespace NUMINAMATH_CALUDE_infinite_divisible_factorial_exponents_l2722_272286

/-- νₚ(n) is the exponent of p in the prime factorization of n! -/
def ν (p : Nat) (n : Nat) : Nat :=
  sorry

theorem infinite_divisible_factorial_exponents
  (d : Nat) (primes : Finset Nat) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ S : Set Nat, Set.Infinite S ∧
    ∀ n ∈ S, ∀ p ∈ primes, d ∣ ν p n :=
  sorry

end NUMINAMATH_CALUDE_infinite_divisible_factorial_exponents_l2722_272286


namespace NUMINAMATH_CALUDE_tangent_line_implies_function_values_l2722_272233

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_implies_function_values :
  (∀ y, y = f 5 ↔ y = -5 + 8) →  -- Tangent line equation at x = 5
  (f 5 = 3 ∧ deriv f 5 = -1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_function_values_l2722_272233


namespace NUMINAMATH_CALUDE_team_cautions_l2722_272229

theorem team_cautions (total_players : ℕ) (red_cards : ℕ) (yellow_per_red : ℕ) :
  total_players = 11 →
  red_cards = 3 →
  yellow_per_red = 2 →
  ∃ (no_caution players_with_yellow : ℕ),
    no_caution + players_with_yellow = total_players ∧
    players_with_yellow = red_cards * yellow_per_red ∧
    no_caution = 5 :=
by sorry

end NUMINAMATH_CALUDE_team_cautions_l2722_272229


namespace NUMINAMATH_CALUDE_smallest_x_for_inequality_l2722_272246

theorem smallest_x_for_inequality :
  ∃ (x : ℝ), x = 49 ∧
  (∀ (a : ℝ), a ≥ 0 → a ≥ 14 * Real.sqrt a - x) ∧
  (∀ (y : ℝ), y < x → ∃ (a : ℝ), a ≥ 0 ∧ a < 14 * Real.sqrt a - y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_inequality_l2722_272246


namespace NUMINAMATH_CALUDE_cross_in_square_l2722_272244

theorem cross_in_square (s : ℝ) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l2722_272244


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l2722_272204

/-- Represents the number of people who purchased only book A -/
def C : ℕ := 1000

/-- Represents the number of people who purchased both books A and B -/
def AB : ℕ := 500

/-- Represents the total number of people who purchased book A -/
def A : ℕ := C + AB

/-- Represents the total number of people who purchased book B -/
def B : ℕ := AB + (A / 2 - AB)

theorem book_purchase_ratio : (AB : ℚ) / (B - AB : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l2722_272204


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l2722_272216

theorem greatest_integer_with_gcf_four (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 4 → n ≤ 148 := by
  sorry

theorem exists_unique_greatest : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n := by
  sorry

theorem solution_is_148 : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n ∧ n = 148 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l2722_272216


namespace NUMINAMATH_CALUDE_min_value_expression_l2722_272253

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (2 * a / b) + (3 * b / c) + (4 * c / a) ≥ 9 ∧
  ((2 * a / b) + (3 * b / c) + (4 * c / a) = 9 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2722_272253


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2722_272295

/-- The equation of a circle with center (-1, 1) that is tangent to the line x - y = 0 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), (x + 1)^2 + (y - 1)^2 = r^2 ∧ 
  r = |(-1 - 1 + 0)| / Real.sqrt (1^2 + (-1)^2) ∧
  r > 0) ↔ 
  (x + 1)^2 + (y - 1)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2722_272295


namespace NUMINAMATH_CALUDE_mary_has_more_euros_l2722_272298

-- Define initial amounts
def michelle_initial : ℚ := 30
def alice_initial : ℚ := 18
def marco_initial : ℚ := 24
def mary_initial : ℚ := 15

-- Define conversion rate
def usd_to_eur : ℚ := 0.85

-- Define transactions
def marco_to_mary : ℚ := marco_initial / 2
def michelle_to_alice : ℚ := michelle_initial * (40 / 100)
def mary_spend : ℚ := 5
def alice_convert : ℚ := 10

-- Calculate final amounts
def marco_final : ℚ := marco_initial - marco_to_mary
def mary_final : ℚ := mary_initial + marco_to_mary - mary_spend
def alice_final_usd : ℚ := alice_initial + michelle_to_alice - alice_convert
def alice_final_eur : ℚ := alice_convert * usd_to_eur

-- Theorem statement
theorem mary_has_more_euros :
  mary_final = marco_final + alice_final_eur + (3/2) := by sorry

end NUMINAMATH_CALUDE_mary_has_more_euros_l2722_272298


namespace NUMINAMATH_CALUDE_rate_of_current_l2722_272291

/-- Proves that given a man's downstream speed, upstream speed, and still water speed, 
    the rate of current can be calculated. -/
theorem rate_of_current 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : downstream_speed = 45) 
  (h2 : upstream_speed = 23) 
  (h3 : still_water_speed = 34) : 
  downstream_speed - still_water_speed = 11 := by
  sorry

#check rate_of_current

end NUMINAMATH_CALUDE_rate_of_current_l2722_272291


namespace NUMINAMATH_CALUDE_difference_squared_equals_negative_sixteen_l2722_272251

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : a^2 + 8 > 0)  -- Ensure a^2 + 8 is positive to avoid division by zero
variable (h2 : a * b = 12)

-- State the theorem
theorem difference_squared_equals_negative_sixteen : (a - b)^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_difference_squared_equals_negative_sixteen_l2722_272251


namespace NUMINAMATH_CALUDE_equation_solution_l2722_272208

theorem equation_solution :
  ∃! y : ℚ, 7 * (4 * y + 3) - 3 = -3 * (2 - 5 * y) ∧ y = -24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2722_272208


namespace NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l2722_272213

theorem sum_of_powers_implies_sum_power (a b : ℝ) : 
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l2722_272213


namespace NUMINAMATH_CALUDE_intersection_condition_l2722_272292

-- Define the curves
def C₁ (x : ℝ) : Prop := ∃ y : ℝ, y = x^2 ∧ -2 ≤ x ∧ x ≤ 2

def C₂ (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Theorem statement
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, C₁ x ∧ C₂ m x y) ↔ -1/4 ≤ m ∧ m ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2722_272292


namespace NUMINAMATH_CALUDE_cos_greater_when_sin_greater_in_second_quadrant_l2722_272259

theorem cos_greater_when_sin_greater_in_second_quadrant 
  (α β : Real) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : π/2 < β ∧ β < π) 
  (h3 : Real.sin α > Real.sin β) : 
  Real.cos α > Real.cos β := by
sorry

end NUMINAMATH_CALUDE_cos_greater_when_sin_greater_in_second_quadrant_l2722_272259


namespace NUMINAMATH_CALUDE_linear_system_ratio_l2722_272211

theorem linear_system_ratio (x y a b : ℝ) 
  (eq1 : 4 * x - 6 * y = a)
  (eq2 : 9 * x - 6 * y = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = 2 := by
sorry

end NUMINAMATH_CALUDE_linear_system_ratio_l2722_272211


namespace NUMINAMATH_CALUDE_not_all_zero_iff_at_least_one_nonzero_l2722_272268

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_iff_at_least_one_nonzero_l2722_272268


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_arithmetic_l2722_272217

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_fifth_sum_arithmetic (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  a 6 = 33 →
  a 7 = 38 →
  a 4 + a 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_fifth_sum_arithmetic_l2722_272217


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2722_272270

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧ 
  ¬(monotonic_on (f a) 1 2 → a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2722_272270


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2722_272210

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x + 3 * y = 9) 
  (h2 : x * y = -15) : 
  x^2 + 9 * y^2 = 171 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2722_272210
