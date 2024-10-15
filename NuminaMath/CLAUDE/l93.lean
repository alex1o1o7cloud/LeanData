import Mathlib

namespace NUMINAMATH_CALUDE_min_value_cos_sin_min_value_cos_sin_achieved_l93_9380

theorem min_value_cos_sin (x : ℝ) : 2 * (Real.cos x)^2 - Real.sin (2 * x) ≥ 1 - Real.sqrt 2 := by
  sorry

theorem min_value_cos_sin_achieved : ∃ x : ℝ, 2 * (Real.cos x)^2 - Real.sin (2 * x) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_min_value_cos_sin_achieved_l93_9380


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l93_9393

/-- The minimum distance between an ellipse and a line -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = (15 : ℝ) / Real.sqrt 41 ∧
  ∀ (x y : ℝ), 
    (x^2 / 25 + y^2 / 9 = 1) →
    (∀ (x' y' : ℝ), (4*x' - 5*y' + 40 = 0) → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l93_9393


namespace NUMINAMATH_CALUDE_max_movies_watched_l93_9329

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270
def wednesday_movie_multiplier : ℕ := 2

theorem max_movies_watched (movie_duration : ℕ) (tuesday_watch_time : ℕ) (wednesday_movie_multiplier : ℕ) :
  movie_duration = 90 →
  tuesday_watch_time = 270 →
  wednesday_movie_multiplier = 2 →
  (tuesday_watch_time / movie_duration + wednesday_movie_multiplier * (tuesday_watch_time / movie_duration)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_movies_watched_l93_9329


namespace NUMINAMATH_CALUDE_expression_simplification_l93_9337

theorem expression_simplification (x y a b c : ℝ) :
  (2 - y) * 24 * (x - y) + 2 * ((a - 2 - 3 * c) * a - 2 * b + c) = 2 + 4 * b^2 - a * b - c^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l93_9337


namespace NUMINAMATH_CALUDE_polynomial_factorization_l93_9332

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l93_9332


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l93_9391

theorem square_difference_fourth_power : (6^2 - 3^2)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l93_9391


namespace NUMINAMATH_CALUDE_sector_angle_l93_9346

/-- A circular sector with area 1 cm² and perimeter 4 cm has a central angle of 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  (1/2 * α * r^2 = 1) → (2*r + α*r = 4) → α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l93_9346


namespace NUMINAMATH_CALUDE_ratio_to_percentage_difference_l93_9330

theorem ratio_to_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_difference_l93_9330


namespace NUMINAMATH_CALUDE_equation_solutions_l93_9312

theorem equation_solutions : 
  let f (x : ℂ) := (x - 2)^4 + (x - 6)^4 + 16
  ∀ x : ℂ, f x = 0 ↔ 
    x = 4 + 2*I*Real.sqrt 3 ∨ 
    x = 4 - 2*I*Real.sqrt 3 ∨ 
    x = 4 + I*Real.sqrt 2 ∨ 
    x = 4 - I*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l93_9312


namespace NUMINAMATH_CALUDE_percentage_relation_l93_9348

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l93_9348


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_3_min_m_value_l93_9363

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 1|

-- Part I
theorem solution_set_for_m_eq_3 :
  {x : ℝ | f 3 x > 4} = {x : ℝ | x < -4/5 ∨ x > 4/5} :=
sorry

-- Part II
theorem min_m_value (m : ℝ) (h1 : 0 < m) (h2 : m < 2) 
  (h3 : ∀ x : ℝ, f m x ≥ 3 / (2 * m)) :
  m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_3_min_m_value_l93_9363


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l93_9359

theorem arithmetic_square_root_of_nine :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 9 ∧ (∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l93_9359


namespace NUMINAMATH_CALUDE_min_value_theorem_l93_9321

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ 
  ∀ (z : ℝ), z = (1 / (x - y)) + (9 / (x + 5*y)) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l93_9321


namespace NUMINAMATH_CALUDE_niles_win_probability_l93_9311

/-- Represents a die with six faces. -/
structure Die :=
  (faces : Fin 6 → ℕ)

/-- Billie's die -/
def billie_die : Die :=
  { faces := λ i => i.val + 1 }

/-- Niles' die -/
def niles_die : Die :=
  { faces := λ i => if i.val < 3 then 4 else 5 }

/-- The probability that Niles wins when rolling against Billie -/
def niles_win_prob : ℚ :=
  7 / 12

theorem niles_win_probability :
  let p := niles_win_prob.num
  let q := niles_win_prob.den
  7 * p + 11 * q = 181 := by sorry

end NUMINAMATH_CALUDE_niles_win_probability_l93_9311


namespace NUMINAMATH_CALUDE_S_bounds_l93_9376

def is_permutation (x : Fin 10 → ℕ) : Prop :=
  ∀ n : ℕ, n < 10 → ∃ i : Fin 10, x i = n

def S (x : Fin 10 → ℕ) : ℕ :=
  x 1 + x 2 + x 3 + x 4 + x 6 + x 7 + x 8

theorem S_bounds (x : Fin 10 → ℕ) (h : is_permutation x) : 
  21 ≤ S x ∧ S x ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_S_bounds_l93_9376


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l93_9340

/-- Given two circles where one has a diameter of 80 cm and its radius is 4 times
    the radius of the other, prove that the radius of the smaller circle is 10 cm. -/
theorem smaller_circle_radius (d : ℝ) (r₁ r₂ : ℝ) : 
  d = 80 → r₁ = d / 2 → r₁ = 4 * r₂ → r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l93_9340


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l93_9396

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (a-1)*x + a*y + 1 = 0) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ + 2*a*y₁ - 1 = 0 ∧ (a-1)*x₂ + a*y₂ + 1 = 0 ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l93_9396


namespace NUMINAMATH_CALUDE_injective_function_equality_l93_9328

def injective (f : ℕ → ℝ) : Prop := ∀ n m : ℕ, f n = f m → n = m

theorem injective_function_equality (f : ℕ → ℝ) (n m : ℕ) 
  (h_inj : injective f) 
  (h_eq : 1 / f n + 1 / f m = 4 / (f n + f m)) : 
  n = m := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equality_l93_9328


namespace NUMINAMATH_CALUDE_sandwiches_al_can_order_correct_l93_9314

/-- Represents the types of ingredients available at the deli -/
structure DeliIngredients where
  breads : Nat
  meats : Nat
  cheeses : Nat

/-- Represents the specific ingredients mentioned in the problem -/
structure SpecificIngredients where
  turkey : Bool
  salami : Bool
  swissCheese : Bool
  multiGrainBread : Bool

/-- Calculates the number of sandwiches Al can order -/
def sandwichesAlCanOrder (d : DeliIngredients) (s : SpecificIngredients) : Nat :=
  d.breads * d.meats * d.cheeses - d.breads - d.cheeses

/-- The theorem stating the number of sandwiches Al can order -/
theorem sandwiches_al_can_order_correct (d : DeliIngredients) (s : SpecificIngredients) :
  d.breads = 5 → d.meats = 7 → d.cheeses = 6 →
  s.turkey = true → s.salami = true → s.swissCheese = true → s.multiGrainBread = true →
  sandwichesAlCanOrder d s = 199 := by
  sorry

#check sandwiches_al_can_order_correct

end NUMINAMATH_CALUDE_sandwiches_al_can_order_correct_l93_9314


namespace NUMINAMATH_CALUDE_age_difference_proof_l93_9320

theorem age_difference_proof (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ b)
  (h4 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  (10 * a + b) - (10 * b + a) = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l93_9320


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l93_9353

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (p : Point) 
  (l1 : Line) 
  (l2 : Line) : 
  p.liesOn l2 ∧ l2.isParallelTo l1 → 
  l2 = Line.mk 1 (-2) 7 :=
by
  sorry

#check line_through_point_parallel_to_line 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 3) 
  (Line.mk 1 (-2) 7)

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l93_9353


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l93_9368

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l93_9368


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l93_9389

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |2 * n + 8| = 3 * n - 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l93_9389


namespace NUMINAMATH_CALUDE_not_red_card_probability_l93_9370

/-- Given a deck of cards where the odds of drawing a red card are 1:3,
    the probability of drawing a card that is not red is 3/4. -/
theorem not_red_card_probability (odds : ℚ) (h : odds = 1/3) :
  1 - odds / (1 + odds) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_not_red_card_probability_l93_9370


namespace NUMINAMATH_CALUDE_original_salary_approximation_l93_9361

/-- Calculates the final salary after applying a sequence of percentage changes --/
def final_salary (original : ℝ) : ℝ :=
  original * 1.12 * 0.93 * 1.09 * 0.94

/-- Theorem stating that the original salary is approximately 981.47 --/
theorem original_salary_approximation :
  ∃ (S : ℝ), S > 0 ∧ final_salary S = 1212 ∧ abs (S - 981.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_approximation_l93_9361


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l93_9310

theorem volleyball_team_selection (n : ℕ) (k : ℕ) : n = 16 ∧ k = 7 → Nat.choose n k = 11440 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l93_9310


namespace NUMINAMATH_CALUDE_car_travel_distance_l93_9342

theorem car_travel_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) :
  train_speed = 90 →
  car_speed_ratio = 5/6 →
  travel_time_minutes = 45 →
  let car_speed := car_speed_ratio * train_speed
  let travel_time_hours := travel_time_minutes / 60
  car_speed * travel_time_hours = 56.25 := by
sorry

end NUMINAMATH_CALUDE_car_travel_distance_l93_9342


namespace NUMINAMATH_CALUDE_square_sum_xy_l93_9323

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x - y = 5) :
  (x + y)^2 = 130 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l93_9323


namespace NUMINAMATH_CALUDE_total_fish_l93_9378

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l93_9378


namespace NUMINAMATH_CALUDE_cycling_speed_rectangular_park_l93_9316

/-- Calculates the cycling speed around a rectangular park -/
theorem cycling_speed_rectangular_park 
  (L B : ℝ) 
  (h1 : B = 3 * L) 
  (h2 : L * B = 120000) 
  (h3 : (2 * L + 2 * B) / 8 = 200) : 
  (200 : ℝ) * 60 / 1000 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_rectangular_park_l93_9316


namespace NUMINAMATH_CALUDE_defective_units_shipped_for_sale_l93_9347

/-- 
Given that 4% of units produced are defective and 0.16% of units produced
are defective units shipped for sale, prove that 4% of defective units
are shipped for sale.
-/
theorem defective_units_shipped_for_sale 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (defective_shipped_rate : ℝ) 
  (h1 : defective_rate = 0.04) 
  (h2 : defective_shipped_rate = 0.0016) : 
  defective_shipped_rate / defective_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_for_sale_l93_9347


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l93_9390

theorem girls_to_boys_ratio (total_students : ℕ) 
  (girls boys : ℕ) 
  (girls_with_dogs : ℚ) 
  (boys_with_dogs : ℚ) 
  (total_with_dogs : ℕ) :
  total_students = 100 →
  girls + boys = total_students →
  girls_with_dogs = 1/5 →
  boys_with_dogs = 1/10 →
  total_with_dogs = 15 →
  girls_with_dogs * girls + boys_with_dogs * boys = total_with_dogs →
  girls = boys :=
by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l93_9390


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l93_9367

theorem least_subtraction_for_divisibility (x : ℕ) : 
  (x = 26 ∧ (12702 - x) % 99 = 0) ∧ 
  ∀ y : ℕ, y < x → (12702 - y) % 99 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l93_9367


namespace NUMINAMATH_CALUDE_weight_range_proof_l93_9375

/-- Given the weights of Tracy, John, and Jake, prove the range of their weights. -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) : 
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end NUMINAMATH_CALUDE_weight_range_proof_l93_9375


namespace NUMINAMATH_CALUDE_stream_speed_l93_9383

/-- Given a boat's travel times and distances, calculate the stream speed -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 90) 
  (h2 : upstream_distance = 72)
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l93_9383


namespace NUMINAMATH_CALUDE_dog_legs_on_street_l93_9388

theorem dog_legs_on_street (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2 / 3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 :=
by sorry

end NUMINAMATH_CALUDE_dog_legs_on_street_l93_9388


namespace NUMINAMATH_CALUDE_correct_probability_l93_9319

-- Define the set of balls
inductive Ball : Type
| Red1 : Ball
| Red2 : Ball
| Red3 : Ball
| White2 : Ball
| White3 : Ball

-- Define a function to check if two balls have different colors and numbers
def differentColorAndNumber (b1 b2 : Ball) : Prop :=
  match b1, b2 with
  | Ball.Red1, Ball.White2 => True
  | Ball.Red1, Ball.White3 => True
  | Ball.Red2, Ball.White3 => True
  | Ball.Red3, Ball.White2 => True
  | _, _ => False

-- Define the probability of drawing two balls with different colors and numbers
def probabilityDifferentColorAndNumber : ℚ :=
  2 / 5

-- State the theorem
theorem correct_probability :
  probabilityDifferentColorAndNumber = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_correct_probability_l93_9319


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l93_9369

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_5 = 2S_4 + 3 and a_6 = 2S_5 + 3, then q = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 5 = 2 * S 4 + 3 →
  a 6 = 2 * S 5 + 3 →
  q = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l93_9369


namespace NUMINAMATH_CALUDE_bike_distance_theorem_l93_9357

/-- Calculates the distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 3 m/s for 7 seconds covers 21 meters -/
theorem bike_distance_theorem :
  let speed : ℝ := 3
  let time : ℝ := 7
  distance_traveled speed time = 21 := by sorry

end NUMINAMATH_CALUDE_bike_distance_theorem_l93_9357


namespace NUMINAMATH_CALUDE_provisions_last_20_days_l93_9338

/-- Calculates the number of days provisions will last after reinforcement -/
def provisions_duration (initial_men : ℕ) (initial_days : ℕ) (days_passed : ℕ) (reinforcement : ℕ) : ℚ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := total_provisions - (initial_men * days_passed)
  let new_total_men := initial_men + reinforcement
  remaining_provisions / new_total_men

/-- Proves that given the initial conditions, the provisions will last for 20 days after reinforcement -/
theorem provisions_last_20_days :
  provisions_duration 2000 54 21 1300 = 20 := by
  sorry

end NUMINAMATH_CALUDE_provisions_last_20_days_l93_9338


namespace NUMINAMATH_CALUDE_climb_10_stairs_l93_9315

/-- The number of ways to climb n stairs -/
def climbWays : ℕ → ℕ
  | 0 => 1  -- base case for 0 stairs
  | 1 => 1  -- given condition
  | 2 => 2  -- given condition
  | (n + 3) => climbWays (n + 2) + climbWays (n + 1)

/-- Theorem stating that there are 89 ways to climb 10 stairs -/
theorem climb_10_stairs : climbWays 10 = 89 := by
  sorry

/-- Lemma: The number of ways to climb n stairs is the sum of ways to climb (n-1) and (n-2) stairs -/
lemma climb_recursive (n : ℕ) (h : n ≥ 3) : climbWays n = climbWays (n - 1) + climbWays (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_climb_10_stairs_l93_9315


namespace NUMINAMATH_CALUDE_simplify_power_expression_l93_9371

theorem simplify_power_expression (x : ℝ) : (3 * (2 * x)^5)^4 = 84934656 * x^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l93_9371


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l93_9300

theorem opposite_of_negative_fraction :
  let x : ℚ := -4/5
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l93_9300


namespace NUMINAMATH_CALUDE_math_club_election_l93_9381

theorem math_club_election (total_candidates : ℕ) (positions : ℕ) (past_officers : ℕ) 
  (h1 : total_candidates = 20)
  (h2 : positions = 5)
  (h3 : past_officers = 10) :
  (Nat.choose total_candidates positions) - (Nat.choose (total_candidates - past_officers) positions) = 15252 := by
sorry

end NUMINAMATH_CALUDE_math_club_election_l93_9381


namespace NUMINAMATH_CALUDE_alyssa_fruit_expenses_l93_9397

theorem alyssa_fruit_expenses : 
  let grapes_cost : ℚ := 12.08
  let cherries_cost : ℚ := 9.85
  grapes_cost + cherries_cost = 21.93 := by sorry

end NUMINAMATH_CALUDE_alyssa_fruit_expenses_l93_9397


namespace NUMINAMATH_CALUDE_complex_equation_solution_l93_9394

theorem complex_equation_solution :
  ∀ b : ℝ, (6 - b * I) / (1 + 2 * I) = 2 - 2 * I → b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l93_9394


namespace NUMINAMATH_CALUDE_quadratic_factoring_l93_9308

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The result of factoring a quadratic equation -/
inductive FactoredForm
  | Product : FactoredForm

/-- Factoring a quadratic equation results in a product form -/
theorem quadratic_factoring (eq : QuadraticEquation) : ∃ (f : FactoredForm), f = FactoredForm.Product := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l93_9308


namespace NUMINAMATH_CALUDE_unique_root_quadratic_theorem_l93_9305

/-- A quadratic polynomial with exactly one root -/
def UniqueRootQuadratic (g : ℝ → ℝ) : Prop :=
  (∃ x₀, g x₀ = 0) ∧ (∀ x y, g x = 0 → g y = 0 → x = y)

theorem unique_root_quadratic_theorem
  (g : ℝ → ℝ)
  (h_unique : UniqueRootQuadratic g)
  (a b c d : ℝ)
  (h_ac : a ≠ c)
  (h_composed : UniqueRootQuadratic (fun x ↦ g (a * x + b) + g (c * x + d))) :
  ∃ x₀, g x₀ = 0 ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_theorem_l93_9305


namespace NUMINAMATH_CALUDE_remainder_theorem_l93_9392

theorem remainder_theorem : ∃ q : ℕ, 2^202 + 202 = (2^101 + 2^51 + 1) * q + 201 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l93_9392


namespace NUMINAMATH_CALUDE_fundraiser_hourly_rate_l93_9377

/-- Proves that if 8 volunteers working 40 hours each at $18 per hour raise the same total amount
    as 12 volunteers working 32 hours each, then the hourly rate for the second group is $15. -/
theorem fundraiser_hourly_rate
  (volunteers_last_week : ℕ)
  (hours_last_week : ℕ)
  (rate_last_week : ℚ)
  (volunteers_this_week : ℕ)
  (hours_this_week : ℕ)
  (h1 : volunteers_last_week = 8)
  (h2 : hours_last_week = 40)
  (h3 : rate_last_week = 18)
  (h4 : volunteers_this_week = 12)
  (h5 : hours_this_week = 32)
  (h6 : volunteers_last_week * hours_last_week * rate_last_week =
        volunteers_this_week * hours_this_week * (15 : ℚ)) :
  15 = (volunteers_last_week * hours_last_week * rate_last_week) /
       (volunteers_this_week * hours_this_week) :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_hourly_rate_l93_9377


namespace NUMINAMATH_CALUDE_faster_train_speed_l93_9355

/-- Proves that the speed of the faster train is 20/3 m/s given the specified conditions -/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h_length : train_length = 100)
  (h_time : crossing_time = 20)
  (h_speed_ratio : ∃ (v : ℝ), v > 0 ∧ faster_speed = 2 * v ∧ slower_speed = v)
  (h_relative_speed : relative_speed = faster_speed + slower_speed)
  (h_distance : total_distance = 2 * train_length)
  (h_speed_formula : relative_speed = total_distance / crossing_time) :
  faster_speed = 20 / 3 :=
sorry

end NUMINAMATH_CALUDE_faster_train_speed_l93_9355


namespace NUMINAMATH_CALUDE_min_a_value_l93_9349

noncomputable def f (x : ℝ) : ℝ := (2 * 2023^x) / (2023^x + 1)

theorem min_a_value (a : ℝ) :
  (∀ x : ℝ, x > 0 → f (a * Real.exp x) ≥ 2 - f (Real.log a - Real.log x)) →
  a ≥ 1 / Real.exp 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → f (b * Real.exp x) ≥ 2 - f (Real.log b - Real.log x)) → b ≥ 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l93_9349


namespace NUMINAMATH_CALUDE_wood_measurement_correct_l93_9301

/-- Represents the system of equations for the wood measurement problem from "The Mathematical Classic of Sunzi" --/
def wood_measurement_system (x y : ℝ) : Prop :=
  (x - y = 4.5) ∧ (y - (1/2) * x = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem --/
theorem wood_measurement_correct (x y : ℝ) :
  (x > y) ∧                         -- rope is longer than wood
  (x - y = 4.5) ∧                   -- 4.5 feet of rope left when measuring
  (y > (1/2) * x) ∧                 -- wood is longer than half the rope
  (y - (1/2) * x = 1) →             -- rope falls short by 1 foot when folded
  wood_measurement_system x y := by
  sorry


end NUMINAMATH_CALUDE_wood_measurement_correct_l93_9301


namespace NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l93_9365

theorem unique_solution_sin_cos_equation :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ Real.sin (Real.cos x) = Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l93_9365


namespace NUMINAMATH_CALUDE_borrowed_amount_l93_9358

theorem borrowed_amount (P : ℝ) (interest_rate : ℝ) (total_repayment : ℝ) : 
  interest_rate = 0.1 →
  total_repayment = 1320 →
  total_repayment = P * (1 + interest_rate) →
  P = 1200 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l93_9358


namespace NUMINAMATH_CALUDE_probability_two_pairs_l93_9325

def total_socks : ℕ := 10
def drawn_socks : ℕ := 4
def distinct_pairs : ℕ := 5

theorem probability_two_pairs : 
  (Nat.choose distinct_pairs 2) / (Nat.choose total_socks drawn_socks) = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_pairs_l93_9325


namespace NUMINAMATH_CALUDE_insulation_project_proof_l93_9386

/-- The daily completion rate of Team A in square meters -/
def team_a_rate : ℝ := 200

/-- The daily completion rate of Team B in square meters -/
def team_b_rate : ℝ := 1.5 * team_a_rate

/-- The total area to be insulated in square meters -/
def total_area : ℝ := 9000

/-- The difference in completion time between Team A and Team B in days -/
def time_difference : ℝ := 15

theorem insulation_project_proof :
  (total_area / team_a_rate) - (total_area / team_b_rate) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_insulation_project_proof_l93_9386


namespace NUMINAMATH_CALUDE_train_a_speed_l93_9350

/-- The speed of Train A in miles per hour -/
def speed_train_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_train_b : ℝ := 36

/-- The time difference in hours between Train A and Train B's departure -/
def time_difference : ℝ := 2

/-- The distance in miles at which Train B overtakes Train A -/
def overtake_distance : ℝ := 360

/-- Theorem stating that the speed of Train A is 30 mph given the conditions -/
theorem train_a_speed :
  ∃ (t : ℝ), 
    t > time_difference ∧
    speed_train_a * t = overtake_distance ∧
    speed_train_b * (t - time_difference) = overtake_distance ∧
    speed_train_a = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_a_speed_l93_9350


namespace NUMINAMATH_CALUDE_exponent_equality_and_inequalities_l93_9327

theorem exponent_equality_and_inequalities : 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  ((-2 : ℤ)^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^3 ≠ -2^3) :=
by sorry

end NUMINAMATH_CALUDE_exponent_equality_and_inequalities_l93_9327


namespace NUMINAMATH_CALUDE_division_problem_l93_9399

theorem division_problem (x : ℚ) : 
  (2976 / x - 240 = 8) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l93_9399


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l93_9362

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 6*x₁ + 5 = 0) → (x₂^2 - 6*x₂ + 5 = 0) → x₁ + x₂ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l93_9362


namespace NUMINAMATH_CALUDE_manuscript_cost_is_860_l93_9317

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (revisedOnce : ℕ) (revisedTwice : ℕ) 
  (firstTypeCost : ℕ) (revisionCost : ℕ) : ℕ :=
  totalPages * firstTypeCost + revisedOnce * revisionCost + revisedTwice * 2 * revisionCost

/-- Proves that the total cost of typing a 100-page manuscript with given revision parameters is $860. -/
theorem manuscript_cost_is_860 : 
  manuscriptTypingCost 100 35 15 6 4 = 860 := by
  sorry

#eval manuscriptTypingCost 100 35 15 6 4

end NUMINAMATH_CALUDE_manuscript_cost_is_860_l93_9317


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l93_9384

theorem arithmetic_sequence_sum (n : ℕ) (s : ℕ → ℝ) :
  (∀ k, s (k + 1) - s k = s (k + 2) - s (k + 1)) →  -- arithmetic sequence condition
  s n = 48 →                                        -- sum of first n terms
  s (2 * n) = 60 →                                  -- sum of first 2n terms
  s (3 * n) = 36 :=                                 -- sum of first 3n terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l93_9384


namespace NUMINAMATH_CALUDE_comics_in_box_l93_9352

theorem comics_in_box (pages_per_comic : ℕ) (found_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  found_pages = 150 →
  untorn_comics = 5 →
  (found_pages / pages_per_comic + untorn_comics : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_comics_in_box_l93_9352


namespace NUMINAMATH_CALUDE_part_one_part_two_l93_9331

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a))

-- Theorem for part 2
theorem part_two :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l93_9331


namespace NUMINAMATH_CALUDE_combined_ratio_theorem_l93_9360

/-- Represents the ratio of liquids in a vessel -/
structure LiquidRatio :=
  (water : ℚ)
  (milk : ℚ)
  (syrup : ℚ)

/-- Represents a vessel with its volume and liquid ratio -/
structure Vessel :=
  (volume : ℚ)
  (ratio : LiquidRatio)

def combine_vessels (vessels : List Vessel) : LiquidRatio :=
  let total_water := vessels.map (λ v => v.volume * v.ratio.water) |>.sum
  let total_milk := vessels.map (λ v => v.volume * v.ratio.milk) |>.sum
  let total_syrup := vessels.map (λ v => v.volume * v.ratio.syrup) |>.sum
  { water := total_water, milk := total_milk, syrup := total_syrup }

theorem combined_ratio_theorem (v1 v2 v3 : Vessel)
  (h1 : v1.volume = 3 ∧ v2.volume = 5 ∧ v3.volume = 7)
  (h2 : v1.ratio = { water := 1/6, milk := 1/3, syrup := 1/2 })
  (h3 : v2.ratio = { water := 2/7, milk := 4/7, syrup := 1/7 })
  (h4 : v3.ratio = { water := 1/2, milk := 1/6, syrup := 1/3 }) :
  let combined := combine_vessels [v1, v2, v3]
  combined.water / (combined.water + combined.milk + combined.syrup) = 228 / 630 ∧
  combined.milk / (combined.water + combined.milk + combined.syrup) = 211 / 630 ∧
  combined.syrup / (combined.water + combined.milk + combined.syrup) = 191 / 630 := by
  sorry

#check combined_ratio_theorem

end NUMINAMATH_CALUDE_combined_ratio_theorem_l93_9360


namespace NUMINAMATH_CALUDE_quadratic_equation_for_complex_roots_l93_9309

theorem quadratic_equation_for_complex_roots (ω : ℂ) (α β : ℂ) 
  (h1 : ω^8 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : α = ω + ω^3 + ω^5) 
  (h4 : β = ω^2 + ω^4 + ω^6 + ω^7) :
  α^2 + α + 3 = 0 ∧ β^2 + β + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_for_complex_roots_l93_9309


namespace NUMINAMATH_CALUDE_solve_fish_problem_l93_9303

def fish_problem (current_fish : ℕ) (added_fish : ℕ) (caught_fish : ℕ) : Prop :=
  let original_fish := current_fish - added_fish
  (caught_fish < original_fish) ∧ (original_fish - caught_fish = 4)

theorem solve_fish_problem :
  ∃ (caught_fish : ℕ), fish_problem 20 8 caught_fish :=
by sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l93_9303


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l93_9326

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l93_9326


namespace NUMINAMATH_CALUDE_arrangement_count_l93_9333

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of elderly people --/
def num_elderly : ℕ := 2

/-- The total number of units to arrange (volunteers + elderly unit) --/
def total_units : ℕ := num_volunteers + 1

/-- The number of possible positions for the elderly unit --/
def elderly_positions : ℕ := total_units - 2

theorem arrangement_count :
  (elderly_positions * Nat.factorial num_volunteers * Nat.factorial num_elderly) = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l93_9333


namespace NUMINAMATH_CALUDE_propositions_p_and_q_are_true_l93_9344

theorem propositions_p_and_q_are_true :
  (∀ (m : ℝ), (∀ (x : ℝ), x^2 + x + m > 0) → m > 1/4) ∧
  (∀ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
    (A > B ↔ Real.sin A > Real.sin B)) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_are_true_l93_9344


namespace NUMINAMATH_CALUDE_only_solution_is_two_l93_9374

theorem only_solution_is_two : 
  ∃! (n : ℤ), n + 13 > 15 ∧ -6*n > -18 :=
by sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l93_9374


namespace NUMINAMATH_CALUDE_mans_age_l93_9318

theorem mans_age (P : ℝ) 
  (h1 : P = 1.25 * (P - 10)) 
  (h2 : P = (250 / 300) * (P + 10)) : 
  P = 50 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_l93_9318


namespace NUMINAMATH_CALUDE_area_FYH_specific_l93_9382

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Calculates the area of triangle FYH in a trapezoid -/
def area_FYH (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle FYH in the specific trapezoid -/
theorem area_FYH_specific : 
  let t : Trapezoid := { base1 := 24, base2 := 36, area := 360 }
  area_FYH t = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_area_FYH_specific_l93_9382


namespace NUMINAMATH_CALUDE_remaining_insects_l93_9307

def playground_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem remaining_insects : 
  playground_insects 3 12 8 2 = 21 := by sorry

end NUMINAMATH_CALUDE_remaining_insects_l93_9307


namespace NUMINAMATH_CALUDE_curve_symmetry_l93_9322

theorem curve_symmetry (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) ↔ x = (p * (-y) + q) / (r * (-y) + s)) →
  p = s :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l93_9322


namespace NUMINAMATH_CALUDE_min_value_expression_l93_9356

theorem min_value_expression (a b c k m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk : k > 0) (hm : m > 0) (hn : n > 0) : 
  (k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a ≥ 6 * k ∧
  ((k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a = 6 * k ↔ 
    k = m ∧ m = n ∧ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l93_9356


namespace NUMINAMATH_CALUDE_expansion_term_count_l93_9335

/-- The number of terms in the expansion of (a+b+c)(a+d+e+f+g) -/
def expansion_terms : ℕ := 15

/-- The first polynomial (a+b+c) has 3 terms -/
def first_poly_terms : ℕ := 3

/-- The second polynomial (a+d+e+f+g) has 5 terms -/
def second_poly_terms : ℕ := 5

/-- Theorem stating that the expansion of (a+b+c)(a+d+e+f+g) has 15 terms -/
theorem expansion_term_count :
  expansion_terms = first_poly_terms * second_poly_terms := by
  sorry

end NUMINAMATH_CALUDE_expansion_term_count_l93_9335


namespace NUMINAMATH_CALUDE_parabola_shift_up_two_l93_9398

/-- Represents a vertical shift transformation of a parabola -/
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x => x^2

/-- Theorem: Shifting the parabola y = x^2 up by 2 units results in y = x^2 + 2 -/
theorem parabola_shift_up_two :
  verticalShift originalParabola 2 = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_up_two_l93_9398


namespace NUMINAMATH_CALUDE_speed_calculation_l93_9336

/-- Given a distance of 3.0 miles and a time of 1.5 hours, prove that the speed is 2.0 miles per hour. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 3.0) 
    (h2 : time = 1.5) 
    (h3 : speed = distance / time) : speed = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l93_9336


namespace NUMINAMATH_CALUDE_decimal_sum_l93_9324

/-- The sum of 0.403, 0.0007, and 0.07 is equal to 0.4737 -/
theorem decimal_sum : 0.403 + 0.0007 + 0.07 = 0.4737 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l93_9324


namespace NUMINAMATH_CALUDE_golden_ratio_function_l93_9354

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_function (f : ℝ → ℝ) :
  (∀ x > 0, Monotone f) →
  (∀ x > 0, f x > 0) →
  (∀ x > 0, f x * f (f x + 1 / x) = 1) →
  f 1 = φ := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_function_l93_9354


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l93_9379

/-- The number of possible six-ring arrangements on four fingers -/
def ring_arrangements : ℕ := 618854400

/-- The number of distinguishable rings -/
def total_rings : ℕ := 10

/-- The number of rings to be arranged -/
def arranged_rings : ℕ := 6

/-- The number of fingers (excluding thumb) -/
def fingers : ℕ := 4

theorem ring_arrangement_count :
  ring_arrangements = (total_rings.choose arranged_rings) * (arranged_rings.factorial) * (fingers ^ arranged_rings) :=
sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l93_9379


namespace NUMINAMATH_CALUDE_sum_of_squares_l93_9372

theorem sum_of_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^2 + 9) / a = (b^2 + 9) / b ∧ (b^2 + 9) / b = (c^2 + 9) / c) : 
  a^2 + b^2 + c^2 = -27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l93_9372


namespace NUMINAMATH_CALUDE_minimum_pastries_for_trick_l93_9341

/-- Represents a pastry with two fillings -/
structure Pastry where
  filling1 : Fin 10
  filling2 : Fin 10
  h : filling1 ≠ filling2

/-- The set of all possible pastries -/
def allPastries : Finset Pastry :=
  sorry

theorem minimum_pastries_for_trick :
  ∀ n : ℕ,
    (n < 36 →
      ∃ (remaining : Finset Pastry),
        remaining ⊆ allPastries ∧
        remaining.card = 45 - n ∧
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (q : Pastry),
              q ∈ remaining ∧ q ≠ p ∧
              (p.filling1 = q.filling1 ∨ p.filling1 = q.filling2 ∨
               p.filling2 = q.filling1 ∨ p.filling2 = q.filling2)) ∧
    (n = 36 →
      ∀ (remaining : Finset Pastry),
        remaining ⊆ allPastries →
        remaining.card = 45 - n →
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (broken : Finset Pastry),
              broken ⊆ allPastries ∧
              broken.card = n ∧
              (p.filling1 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2 ∨
               p.filling2 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_pastries_for_trick_l93_9341


namespace NUMINAMATH_CALUDE_josette_purchase_cost_l93_9364

/-- Calculates the total cost of mineral water bottles with a discount --/
def total_cost (small_count : ℕ) (large_count : ℕ) (small_price : ℚ) (large_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_count := small_count + large_count
  let subtotal := small_count * small_price + large_count * large_price
  if total_count ≥ 5 then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- The total cost for Josette's purchase is €8.37 --/
theorem josette_purchase_cost :
  total_cost 3 2 (3/2) (12/5) (1/10) = 837/100 := by sorry

end NUMINAMATH_CALUDE_josette_purchase_cost_l93_9364


namespace NUMINAMATH_CALUDE_mothers_age_l93_9345

theorem mothers_age (eunji_current_age eunji_past_age mother_past_age : ℕ) 
  (h1 : eunji_current_age = 16)
  (h2 : eunji_past_age = 8)
  (h3 : mother_past_age = 35) : 
  mother_past_age + (eunji_current_age - eunji_past_age) = 43 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l93_9345


namespace NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l93_9334

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem sesame_seed_weight_scientific_notation :
  toScientificNotation 0.00000201 = ScientificNotation.mk 2.01 (-6) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l93_9334


namespace NUMINAMATH_CALUDE_simplify_and_multiply_l93_9313

theorem simplify_and_multiply :
  (3 / 504 - 17 / 72) * (5 / 7) = -145 / 882 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_multiply_l93_9313


namespace NUMINAMATH_CALUDE_ellipse_and_circle_properties_l93_9306

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1/4

-- Theorem statement
theorem ellipse_and_circle_properties :
  -- The eccentricity of ellipse C is √3/2
  (∃ e : ℝ, e = Real.sqrt 3 / 2 ∧
    ∀ x y : ℝ, ellipse_C x y → 
      e = Real.sqrt (1 - (Real.sqrt (1 - x^2/4))^2) / 2) ∧
  -- Circle D lies entirely inside ellipse C
  (∀ x y : ℝ, circle_D x y → ellipse_C x y) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_properties_l93_9306


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_40_integers_from_7_l93_9385

theorem arithmetic_mean_of_40_integers_from_7 :
  let start : ℕ := 7
  let count : ℕ := 40
  let sequence := (fun i => start + i - 1)
  let sum := (sequence 1 + sequence count) * count / 2
  (sum : ℚ) / count = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_40_integers_from_7_l93_9385


namespace NUMINAMATH_CALUDE_positive_solutions_conditions_l93_9351

theorem positive_solutions_conditions (a m x y z : ℝ) : 
  (x + y - z = 2 * a) →
  (x^2 + y^2 = z^2) →
  (m * (x + y) = x * y) →
  (x > 0 ∧ y > 0 ∧ z > 0) ↔ 
  (a / 2 * (2 + Real.sqrt 2) ≤ m ∧ m ≤ 2 * a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_solutions_conditions_l93_9351


namespace NUMINAMATH_CALUDE_sum_three_consecutive_not_prime_l93_9343

theorem sum_three_consecutive_not_prime (n : ℕ) : ¬ Prime (3 * (n + 1)) := by
  sorry

#check sum_three_consecutive_not_prime

end NUMINAMATH_CALUDE_sum_three_consecutive_not_prime_l93_9343


namespace NUMINAMATH_CALUDE_system_solution_l93_9366

theorem system_solution :
  let f (x y : ℝ) := 7 * x^2 + 7 * y^2 - 3 * x^2 * y^2
  let g (x y : ℝ) := x^4 + y^4 - x^2 * y^2
  ∀ x y : ℝ, (f x y = 7 ∧ g x y = 37) ↔
    ((x = Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = -Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7)) ∨
     (x = -Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l93_9366


namespace NUMINAMATH_CALUDE_percentage_relation_l93_9395

theorem percentage_relation (A B C : ℝ) (h1 : A = 0.07 * C) (h2 : A = 0.5 * B) :
  B = 0.14 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l93_9395


namespace NUMINAMATH_CALUDE_modulus_of_z_l93_9373

-- Define the complex number z
def z : ℂ := 3 + 4 * Complex.I

-- State the theorem
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l93_9373


namespace NUMINAMATH_CALUDE_problem_solution_l93_9302

theorem problem_solution : let M := 2021 / 3
                           let N := M / 4
                           let Y := M + N
                           Y = 843 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l93_9302


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l93_9387

theorem percentage_of_male_students
  (total_percentage : ℝ)
  (male_percentage : ℝ)
  (female_percentage : ℝ)
  (male_older_25 : ℝ)
  (female_older_25 : ℝ)
  (prob_younger_25 : ℝ)
  (h1 : total_percentage = male_percentage + female_percentage)
  (h2 : total_percentage = 100)
  (h3 : male_older_25 = 40)
  (h4 : female_older_25 = 20)
  (h5 : prob_younger_25 = 0.72)
  (h6 : prob_younger_25 = (1 - male_older_25 / 100) * male_percentage / 100 +
                          (1 - female_older_25 / 100) * female_percentage / 100) :
  male_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l93_9387


namespace NUMINAMATH_CALUDE_two_person_subcommittees_l93_9339

theorem two_person_subcommittees (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_l93_9339


namespace NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l93_9304

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initial_stones : Nat
  min_take : Nat
  max_take : Nat

/-- Player types -/
inductive Player
  | Petya
  | Computer

/-- Game state -/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- Optimal play function for the computer -/
def optimal_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- Random play function for Petya -/
def random_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- The probability of Petya winning the game -/
def petya_win_probability (game : HeapOfStones) : Real :=
  sorry

/-- Theorem stating the probability of Petya winning -/
theorem petya_win_probability_is_1_256 (game : HeapOfStones) :
  game.initial_stones = 16 ∧ 
  game.min_take = 1 ∧ 
  game.max_take = 4 →
  petya_win_probability game = 1 / 256 :=
by sorry

end NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l93_9304
