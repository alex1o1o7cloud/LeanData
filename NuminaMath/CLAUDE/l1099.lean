import Mathlib

namespace NUMINAMATH_CALUDE_book_length_l1099_109978

theorem book_length (pages_read : ℚ) (pages_remaining : ℚ) (total_pages : ℚ) : 
  pages_read = (2 : ℚ) / 3 * total_pages →
  pages_remaining = (1 : ℚ) / 3 * total_pages →
  pages_read = pages_remaining + 30 →
  total_pages = 90 := by
sorry

end NUMINAMATH_CALUDE_book_length_l1099_109978


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1099_109935

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2*x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1099_109935


namespace NUMINAMATH_CALUDE_fraction_division_l1099_109963

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l1099_109963


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l1099_109969

theorem complex_imaginary_solution (a : ℝ) : 
  (a - (5 : ℂ) / (2 - Complex.I)).im = (a - (5 : ℂ) / (2 - Complex.I)).re → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l1099_109969


namespace NUMINAMATH_CALUDE_room_width_proof_l1099_109932

/-- Given a rectangular room with length 5.5 meters, prove that its width is 4 meters
    when the cost of paving is 850 rupees per square meter and the total cost is 18,700 rupees. -/
theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost / cost_per_sqm / length = 4 := by
sorry

end NUMINAMATH_CALUDE_room_width_proof_l1099_109932


namespace NUMINAMATH_CALUDE_hardcover_probability_l1099_109946

theorem hardcover_probability (total_books : Nat) (hardcover_books : Nat) (selected_books : Nat) :
  total_books = 15 →
  hardcover_books = 5 →
  selected_books = 3 →
  (Nat.choose hardcover_books selected_books * Nat.choose (total_books - hardcover_books) (selected_books - hardcover_books) +
   Nat.choose hardcover_books (selected_books - 1) * Nat.choose (total_books - hardcover_books) 1 +
   Nat.choose hardcover_books selected_books) / Nat.choose total_books selected_books = 67 / 91 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_probability_l1099_109946


namespace NUMINAMATH_CALUDE_part_one_part_two_l1099_109948

/-- Given real numbers a and b, define the functions f and g -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

/-- Define the derivatives of f and g -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

/-- Define consistent monotonicity on an interval -/
def consistent_monotonicity (a b : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f' a x * g' b x ≥ 0

/-- Part 1: Prove that if a > 0 and f, g have consistent monotonicity on [-1, +∞), then b ≥ 2 -/
theorem part_one (a b : ℝ) (ha : a > 0)
  (h_cons : consistent_monotonicity a b { x | x ≥ -1 }) : b ≥ 2 := by
  sorry

/-- Part 2: Prove that if a < 0, a ≠ b, and f, g have consistent monotonicity on (min a b, max a b),
    then |a - b| ≤ 1/3 -/
theorem part_two (a b : ℝ) (ha : a < 0) (hab : a ≠ b)
  (h_cons : consistent_monotonicity a b (Set.Ioo (min a b) (max a b))) : |a - b| ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1099_109948


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1099_109976

theorem condition_necessary_not_sufficient (b : ℝ) (hb : b ≠ 0) :
  (∃ a : ℝ, a > b ∧ Real.log (a - b) ≤ 0) ∧
  (∀ a : ℝ, Real.log (a - b) > 0 → a > b) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1099_109976


namespace NUMINAMATH_CALUDE_not_divides_for_all_m_l1099_109940

theorem not_divides_for_all_m : ∀ m : ℕ, ¬((1000^m - 1) ∣ (1978^m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_for_all_m_l1099_109940


namespace NUMINAMATH_CALUDE_keith_added_scissors_l1099_109913

/-- The number of scissors Keith added to the drawer -/
def scissors_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Keith added 22 scissors to the drawer -/
theorem keith_added_scissors : scissors_added 54 76 = 22 := by
  sorry

end NUMINAMATH_CALUDE_keith_added_scissors_l1099_109913


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1099_109954

theorem quadratic_factorization : 
  ∃ (c d : ℤ), (∀ y : ℝ, 4 * y^2 + 4 * y - 32 = (4 * y + c) * (y + d)) ∧ c - d = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1099_109954


namespace NUMINAMATH_CALUDE_sin_inequality_l1099_109985

open Real

theorem sin_inequality (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  let f := fun θ => (sin θ)^3 / (2 * θ - sin (2 * θ))
  f α > f β := by
sorry

end NUMINAMATH_CALUDE_sin_inequality_l1099_109985


namespace NUMINAMATH_CALUDE_ordering_of_exp_and_log_l1099_109902

theorem ordering_of_exp_and_log : 
  (Real.exp 0.1 - 1 : ℝ) > (0.1 : ℝ) ∧ (0.1 : ℝ) > Real.log 1.1 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_exp_and_log_l1099_109902


namespace NUMINAMATH_CALUDE_rainfall_difference_l1099_109983

/-- The number of Mondays -/
def num_mondays : ℕ := 13

/-- The rainfall on each Monday in centimeters -/
def rain_per_monday : ℝ := 1.75

/-- The number of Tuesdays -/
def num_tuesdays : ℕ := 16

/-- The rainfall on each Tuesday in centimeters -/
def rain_per_tuesday : ℝ := 2.65

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (num_tuesdays : ℝ) * rain_per_tuesday - (num_mondays : ℝ) * rain_per_monday = 19.65 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l1099_109983


namespace NUMINAMATH_CALUDE_one_third_repeating_one_seventh_repeating_one_ninth_repeating_l1099_109944

def repeating_decimal (n : ℕ) (d : ℕ) (period : List ℕ) : ℚ :=
  (n : ℚ) / (d : ℚ)

theorem one_third_repeating : repeating_decimal 1 3 [3] = 1 / 3 := by sorry

theorem one_seventh_repeating : repeating_decimal 1 7 [1, 4, 2, 8, 5, 7] = 1 / 7 := by sorry

theorem one_ninth_repeating : repeating_decimal 1 9 [1] = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_one_third_repeating_one_seventh_repeating_one_ninth_repeating_l1099_109944


namespace NUMINAMATH_CALUDE_power_division_l1099_109995

theorem power_division (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1099_109995


namespace NUMINAMATH_CALUDE_bacteria_growth_l1099_109900

theorem bacteria_growth (quadruple_time : ℕ) (total_time : ℕ) (final_count : ℕ) 
  (h1 : quadruple_time = 20)
  (h2 : total_time = 4 * 60)
  (h3 : final_count = 1048576)
  (h4 : (total_time / quadruple_time : ℚ) = 12) :
  ∃ (initial_count : ℚ), 
    initial_count * (4 ^ (total_time / quadruple_time)) = final_count ∧ 
    initial_count = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1099_109900


namespace NUMINAMATH_CALUDE_chess_piece_position_l1099_109942

/-- Represents a position on a chess board -/
structure ChessPosition :=
  (column : Nat)
  (row : Nat)

/-- Converts a ChessPosition to a pair of natural numbers -/
def ChessPosition.toPair (pos : ChessPosition) : Nat × Nat :=
  (pos.column, pos.row)

theorem chess_piece_position :
  let piece : ChessPosition := ⟨3, 7⟩
  ChessPosition.toPair piece = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_position_l1099_109942


namespace NUMINAMATH_CALUDE_equation_solution_l1099_109981

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (10 * x)^3 → x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1099_109981


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1099_109933

theorem cos_300_degrees (θ : Real) : 
  θ = 300 * Real.pi / 180 → Real.cos θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1099_109933


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l1099_109950

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 3 * Real.pi / 4) - Real.tan (x + Real.pi / 4) + Real.sin (x + Real.pi / 4)
  ∃ (max_y : ℝ), max_y = Real.sqrt 2 / 2 ∧
    ∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ -Real.pi / 2 → y x ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l1099_109950


namespace NUMINAMATH_CALUDE_election_winner_votes_l1099_109910

/-- The number of candidates in the election -/
def num_candidates : ℕ := 4

/-- The percentage of votes received by the winning candidate -/
def winner_percentage : ℚ := 468 / 1000

/-- The percentage of votes received by the second-place candidate -/
def second_percentage : ℚ := 326 / 1000

/-- The margin of victory in number of votes -/
def margin : ℕ := 752

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 5296

/-- The number of votes received by the winning candidate -/
def winner_votes : ℕ := 2479

theorem election_winner_votes :
  num_candidates = 4 ∧
  winner_percentage = 468 / 1000 ∧
  second_percentage = 326 / 1000 ∧
  margin = 752 ∧
  total_votes = 5296 →
  winner_votes = 2479 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1099_109910


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1099_109951

theorem power_fraction_simplification :
  (10 ^ 0.7) * (10 ^ 0.4) / ((10 ^ 0.2) * (10 ^ 0.6) * (10 ^ 0.3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1099_109951


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l1099_109977

/-- A circle in the Euclidean plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A circle is tangent to the y-axis if there exists exactly one point that is both on the circle and on the y-axis -/
def tangent_to_y_axis (c : Circle) : Prop :=
  ∃! p : ℝ × ℝ, c.equation p.1 p.2 ∧ on_y_axis p

/-- The main theorem -/
theorem circle_tangent_to_y_axis :
  let c := Circle.mk (-2, 3) 2
  c.equation x y ↔ (x + 2)^2 + (y - 3)^2 = 4 ∧ tangent_to_y_axis c :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l1099_109977


namespace NUMINAMATH_CALUDE_final_depth_calculation_l1099_109926

/-- Calculates the final depth aimed to dig given initial and new working conditions -/
theorem final_depth_calculation 
  (initial_men : ℕ) 
  (initial_hours : ℕ) 
  (initial_depth : ℕ) 
  (extra_men : ℕ) 
  (new_hours : ℕ) : 
  initial_men = 75 → 
  initial_hours = 8 → 
  initial_depth = 50 → 
  extra_men = 65 → 
  new_hours = 6 → 
  (initial_men + extra_men) * new_hours * initial_depth = initial_men * initial_hours * 70 := by
  sorry

#check final_depth_calculation

end NUMINAMATH_CALUDE_final_depth_calculation_l1099_109926


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1099_109934

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other. -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m-1, -3) and B(2, n) are symmetric with respect to the origin,
    then m + n = 2. -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1099_109934


namespace NUMINAMATH_CALUDE_barkley_bones_l1099_109907

/-- The number of new dog bones Barkley gets at the beginning of each month -/
def monthly_new_bones : ℕ := sorry

/-- The number of months -/
def months : ℕ := 5

/-- The number of bones available after 5 months -/
def available_bones : ℕ := 8

/-- The number of bones buried after 5 months -/
def buried_bones : ℕ := 42

theorem barkley_bones : monthly_new_bones = 10 := by
  sorry

end NUMINAMATH_CALUDE_barkley_bones_l1099_109907


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1099_109989

theorem inequality_solution_set : ∀ x : ℝ, 
  (2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5) ↔ (5 / 2 < x ∧ x ≤ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1099_109989


namespace NUMINAMATH_CALUDE_sum_first_100_odd_integers_l1099_109937

theorem sum_first_100_odd_integers : 
  (Finset.range 100).sum (fun i => 2 * (i + 1) - 1) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_odd_integers_l1099_109937


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1099_109947

/-- Given two parallel vectors p and q, prove that the magnitude of their sum is √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (x : ℝ) : 
  p = (2, -3) → 
  q = (x, 6) → 
  (2 * 6 = -3 * x) →  -- parallelism condition
  ‖p + q‖ = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1099_109947


namespace NUMINAMATH_CALUDE_franks_breakfast_cost_l1099_109997

/-- The cost of Frank's breakfast shopping -/
def breakfast_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  bun_price * bun_quantity + milk_price * milk_quantity + (milk_price * egg_price_multiplier)

/-- Theorem stating that Frank's breakfast shopping costs $11 -/
theorem franks_breakfast_cost :
  breakfast_cost 0.1 10 2 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_franks_breakfast_cost_l1099_109997


namespace NUMINAMATH_CALUDE_min_value_expression_l1099_109960

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℝ) / 2 + 50 / m = 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1099_109960


namespace NUMINAMATH_CALUDE_simplify_expression_l1099_109924

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^3 + b^2) - 2 * b^3 = 9 * b^4 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1099_109924


namespace NUMINAMATH_CALUDE_four_digit_square_decrease_theorem_l1099_109917

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def all_digits_decreasable (n k : ℕ) : Prop :=
  ∀ d, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≥ k

def decrease_all_digits (n k : ℕ) : ℕ :=
  (n / 1000 - k) * 1000 + ((n / 100) % 10 - k) * 100 + ((n / 10) % 10 - k) * 10 + (n % 10 - k)

theorem four_digit_square_decrease_theorem :
  ∀ n : ℕ, is_four_digit n → is_perfect_square n →
  (∃ k : ℕ, k > 0 ∧ all_digits_decreasable n k ∧
   is_four_digit (decrease_all_digits n k) ∧ is_perfect_square (decrease_all_digits n k)) →
  n = 3136 ∨ n = 4489 := by sorry

end NUMINAMATH_CALUDE_four_digit_square_decrease_theorem_l1099_109917


namespace NUMINAMATH_CALUDE_local_min_condition_l1099_109921

open Real

/-- A function f with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + b

/-- The theorem statement -/
theorem local_min_condition (b : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (f b) x) → b ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_local_min_condition_l1099_109921


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1099_109975

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ -1 < x ∧ x < 3) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1099_109975


namespace NUMINAMATH_CALUDE_race_time_proof_l1099_109931

/-- Given a race with the following conditions:
    - The race distance is 240 meters
    - Runner A beats runner B by either 56 meters or 7 seconds
    This theorem proves that runner A's time to complete the race is 23 seconds. -/
theorem race_time_proof (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 240 ∧ distance_diff = 56 ∧ time_diff = 7 →
  ∃ (time_A : ℝ), time_A = 23 ∧
    (race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff)) :=
by sorry

end NUMINAMATH_CALUDE_race_time_proof_l1099_109931


namespace NUMINAMATH_CALUDE_truck_speed_theorem_l1099_109988

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 47

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 53

/-- The initial distance between Truck X and Truck Y in miles -/
def initial_distance : ℝ := 13

/-- The time it takes for Truck Y to overtake and get ahead of Truck X in hours -/
def overtake_time : ℝ := 3

/-- The distance Truck Y is ahead of Truck X after overtaking in miles -/
def final_distance : ℝ := 5

theorem truck_speed_theorem :
  truck_x_speed * overtake_time + initial_distance + final_distance = truck_y_speed * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_truck_speed_theorem_l1099_109988


namespace NUMINAMATH_CALUDE_monochromatic_cycle_exists_l1099_109903

/-- A complete bipartite graph K_{n,n} -/
structure CompleteBipartiteGraph (n : ℕ) where
  left : Fin n
  right : Fin n

/-- A 2-coloring of edges -/
def Coloring (n : ℕ) := CompleteBipartiteGraph n → Bool

/-- A 4-cycle in the graph -/
structure Cycle4 (n : ℕ) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  v4 : Fin n

/-- Check if a 4-cycle is monochromatic under a given coloring -/
def isMonochromatic (n : ℕ) (c : Coloring n) (cycle : Cycle4 n) : Prop :=
  let color1 := c ⟨cycle.v1, cycle.v2⟩
  let color2 := c ⟨cycle.v2, cycle.v3⟩
  let color3 := c ⟨cycle.v3, cycle.v4⟩
  let color4 := c ⟨cycle.v4, cycle.v1⟩
  color1 = color2 ∧ color2 = color3 ∧ color3 = color4

/-- The main theorem: Any 2-coloring of K_{5,5} contains a monochromatic 4-cycle -/
theorem monochromatic_cycle_exists :
  ∀ (c : Coloring 5), ∃ (cycle : Cycle4 5), isMonochromatic 5 c cycle :=
sorry

end NUMINAMATH_CALUDE_monochromatic_cycle_exists_l1099_109903


namespace NUMINAMATH_CALUDE_peaches_before_equals_34_l1099_109906

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The current total number of peaches at the stand -/
def current_total : ℕ := 86

/-- The number of peaches Mike had left at the stand before picking more -/
def peaches_before : ℕ := current_total - peaches_picked

theorem peaches_before_equals_34 : peaches_before = 34 := by sorry

end NUMINAMATH_CALUDE_peaches_before_equals_34_l1099_109906


namespace NUMINAMATH_CALUDE_sports_camp_coach_age_l1099_109972

theorem sports_camp_coach_age (total_members : ℕ) (avg_age : ℕ) 
  (num_girls num_boys num_coaches : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 30 →
  avg_age = 20 →
  num_girls = 10 →
  num_boys = 15 →
  num_coaches = 5 →
  avg_age_girls = 18 →
  avg_age_boys = 19 →
  (total_members * avg_age - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_coaches = 27 :=
by sorry

end NUMINAMATH_CALUDE_sports_camp_coach_age_l1099_109972


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1099_109941

theorem arithmetic_equation : 8 / 4 - 3^2 + 4 * 5 = 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1099_109941


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l1099_109952

/-- Given a segment with endpoints A and B, extended to point C such that BC = 1/2 * AB,
    prove that C has the calculated coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (1, -3) → 
  B = (11, 3) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (16, 6) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l1099_109952


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1099_109901

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a7 (a : ℕ → ℚ) :
  GeometricSequence a →
  a 5 = 1/2 →
  4 * a 3 + a 7 = 2 →
  a 7 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l1099_109901


namespace NUMINAMATH_CALUDE_max_parts_three_planes_exists_eight_parts_l1099_109929

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition not provided as it's not necessary for the statement

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem max_parts_three_planes :
  ∀ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] ≤ 8 :=
sorry

/-- Theorem: There exists a configuration of three planes that divides 3D space into exactly 8 parts -/
theorem exists_eight_parts :
  ∃ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] = 8 :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_exists_eight_parts_l1099_109929


namespace NUMINAMATH_CALUDE_unique_natural_number_with_specific_properties_l1099_109920

theorem unique_natural_number_with_specific_properties :
  ∀ (x n : ℕ),
    x = 5^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 3124 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_specific_properties_l1099_109920


namespace NUMINAMATH_CALUDE_sci_fi_section_pages_per_book_l1099_109982

/-- Given a library section with a number of books and a total number of pages,
    calculate the number of pages per book. -/
def pages_per_book (num_books : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / num_books

/-- Theorem stating that in a library section with 8 books and 3824 total pages,
    each book has 478 pages. -/
theorem sci_fi_section_pages_per_book :
  pages_per_book 8 3824 = 478 := by
  sorry

end NUMINAMATH_CALUDE_sci_fi_section_pages_per_book_l1099_109982


namespace NUMINAMATH_CALUDE_cone_volume_l1099_109945

/-- Given a cone with base area 2π and lateral area 6π, its volume is 8π/3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) : 
  (π * r^2 = 2*π) → 
  (π * r * l = 6*π) → 
  (h^2 + r^2 = l^2) →
  (1/3 * π * r^2 * h = 8*π/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1099_109945


namespace NUMINAMATH_CALUDE_medals_award_ways_l1099_109998

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Canadian sprinters
def canadian_sprinters : ℕ := 4

-- Define the number of non-Canadian sprinters
def non_canadian_sprinters : ℕ := total_sprinters - canadian_sprinters

-- Define the number of medals
def num_medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def ways_to_award_medals : ℕ := 
  -- Case 1: No Canadians get a medal
  (non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)) + 
  -- Case 2: Exactly one Canadian gets a medal
  (canadian_sprinters * num_medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1))

-- Theorem statement
theorem medals_award_ways : 
  ways_to_award_medals = 360 := by sorry

end NUMINAMATH_CALUDE_medals_award_ways_l1099_109998


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1099_109905

/-- The ratio of cylinder volumes formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio :
  let l₁ : ℝ := 6
  let l₂ : ℝ := 9
  let v₁ : ℝ := l₁ * l₂^2 / (4 * Real.pi)
  let v₂ : ℝ := l₂ * l₁^2 / (4 * Real.pi)
  max v₁ v₂ / min v₁ v₂ = 3/2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1099_109905


namespace NUMINAMATH_CALUDE_correlation_coefficient_equals_height_variation_total_variation_is_one_l1099_109923

/-- The correlation coefficient between height and weight -/
def correlation_coefficient : ℝ := 0.76

/-- The proportion of weight variation explained by height -/
def height_explained_variation : ℝ := 0.76

/-- The proportion of weight variation explained by random errors -/
def random_error_variation : ℝ := 0.24

/-- Theorem stating that the correlation coefficient is equal to the proportion of variation explained by height -/
theorem correlation_coefficient_equals_height_variation :
  correlation_coefficient = height_explained_variation :=
by sorry

/-- Theorem stating that the sum of variations explained by height and random errors is 1 -/
theorem total_variation_is_one :
  height_explained_variation + random_error_variation = 1 :=
by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_equals_height_variation_total_variation_is_one_l1099_109923


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1099_109908

theorem complex_equation_solution (i z : ℂ) (hi : i * i = -1) (hz : (2 * i) / z = 1 - i) : z = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1099_109908


namespace NUMINAMATH_CALUDE_video_game_earnings_l1099_109955

def total_games : ℕ := 10
def non_working_games : ℕ := 2
def price_per_game : ℕ := 4

theorem video_game_earnings :
  (total_games - non_working_games) * price_per_game = 32 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l1099_109955


namespace NUMINAMATH_CALUDE_product_of_roots_l1099_109984

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1099_109984


namespace NUMINAMATH_CALUDE_squareable_numbers_l1099_109916

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val.succ + (i.val.succ) = k * k

theorem squareable_numbers : 
  isSquareable 9 ∧ 
  isSquareable 15 ∧ 
  ¬isSquareable 7 ∧ 
  ¬isSquareable 11 :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l1099_109916


namespace NUMINAMATH_CALUDE_absolute_value_of_z_squared_minus_two_z_l1099_109964

theorem absolute_value_of_z_squared_minus_two_z (z : ℂ) : 
  z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_squared_minus_two_z_l1099_109964


namespace NUMINAMATH_CALUDE_no_all_prime_arrangement_l1099_109949

/-- A card with two digits -/
structure Card where
  digit1 : Nat
  digit2 : Nat
  h_different : digit1 ≠ digit2
  h_range : digit1 < 10 ∧ digit2 < 10

/-- Function to form a two-digit number from two digits -/
def formNumber (tens : Nat) (ones : Nat) : Nat :=
  10 * tens + ones

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Main theorem statement -/
theorem no_all_prime_arrangement :
  ¬∃ (card1 card2 : Card),
    card1.digit1 ≠ card2.digit1 ∧
    card1.digit1 ≠ card2.digit2 ∧
    card1.digit2 ≠ card2.digit1 ∧
    card1.digit2 ≠ card2.digit2 ∧
    (∀ (d1 d2 : Nat),
      (d1 = card1.digit1 ∨ d1 = card1.digit2 ∨ d1 = card2.digit1 ∨ d1 = card2.digit2) →
      (d2 = card1.digit1 ∨ d2 = card1.digit2 ∨ d2 = card2.digit1 ∨ d2 = card2.digit2) →
      isPrime (formNumber d1 d2)) :=
sorry

end NUMINAMATH_CALUDE_no_all_prime_arrangement_l1099_109949


namespace NUMINAMATH_CALUDE_marble_count_theorem_l1099_109936

/-- Represents the count of marbles of each color in a bag -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleCount := { red := 2, blue := 4, green := 6 }

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 42

/-- Theorem stating the correct count of marbles given the ratio and green marble count -/
theorem marble_count_theorem (ratio : MarbleCount) (green_count : ℕ) :
  ratio = marbleRatio →
  green_count = greenMarbleCount →
  ∃ (count : MarbleCount),
    count.red = 14 ∧
    count.blue = 28 ∧
    count.green = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l1099_109936


namespace NUMINAMATH_CALUDE_joan_initial_dimes_l1099_109943

/-- The number of dimes Joan spent -/
def dimes_spent : ℕ := 2

/-- The number of dimes Joan has left -/
def dimes_left : ℕ := 3

/-- The initial number of dimes Joan had -/
def initial_dimes : ℕ := dimes_spent + dimes_left

theorem joan_initial_dimes : initial_dimes = 5 := by sorry

end NUMINAMATH_CALUDE_joan_initial_dimes_l1099_109943


namespace NUMINAMATH_CALUDE_sin_half_theta_l1099_109928

theorem sin_half_theta (θ : Real) (h1 : |Real.cos θ| = 1/5) (h2 : 5*Real.pi/2 < θ) (h3 : θ < 3*Real.pi) :
  Real.sin (θ/2) = -Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_half_theta_l1099_109928


namespace NUMINAMATH_CALUDE_bills_max_papers_l1099_109967

/-- Represents the number of items Bill can buy -/
structure BillsPurchase where
  pens : ℕ
  pencils : ℕ
  papers : ℕ

/-- The cost of Bill's purchase -/
def cost (b : BillsPurchase) : ℕ := 3 * b.pens + 5 * b.pencils + 9 * b.papers

/-- A purchase is valid if it meets the given conditions -/
def isValid (b : BillsPurchase) : Prop :=
  b.pens ≥ 2 ∧ b.pencils ≥ 1 ∧ cost b = 72

/-- The maximum number of papers Bill can buy -/
def maxPapers : ℕ := 6

theorem bills_max_papers :
  ∀ b : BillsPurchase, isValid b → b.papers ≤ maxPapers ∧
  ∃ b' : BillsPurchase, isValid b' ∧ b'.papers = maxPapers :=
sorry

end NUMINAMATH_CALUDE_bills_max_papers_l1099_109967


namespace NUMINAMATH_CALUDE_max_self_intersections_l1099_109925

/-- A polygonal chain on a graph paper -/
structure PolygonalChain where
  segments : ℕ
  closed : Bool
  on_graph_paper : Bool
  no_segments_on_same_line : Bool

/-- The number of self-intersection points of a polygonal chain -/
def self_intersection_points (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points for a closed 14-segment polygonal chain 
    on a graph paper, where no two segments lie on the same line, is 17 -/
theorem max_self_intersections (chain : PolygonalChain) :
  chain.segments = 14 ∧ 
  chain.closed ∧ 
  chain.on_graph_paper ∧ 
  chain.no_segments_on_same_line →
  self_intersection_points chain ≤ 17 ∧ 
  ∃ (chain' : PolygonalChain), 
    chain'.segments = 14 ∧ 
    chain'.closed ∧ 
    chain'.on_graph_paper ∧ 
    chain'.no_segments_on_same_line ∧
    self_intersection_points chain' = 17 :=
sorry

end NUMINAMATH_CALUDE_max_self_intersections_l1099_109925


namespace NUMINAMATH_CALUDE_deposit_amount_is_34_l1099_109922

/-- Represents a bank account with transactions --/
structure BankAccount where
  initial_balance : ℕ
  last_month_deposit : ℕ
  current_balance : ℕ

/-- Calculates the deposit amount this month --/
def deposit_this_month (account : BankAccount) : ℕ :=
  account.current_balance - account.initial_balance

/-- Theorem: The deposit amount this month is $34 --/
theorem deposit_amount_is_34 (account : BankAccount) 
  (h1 : account.initial_balance = 150)
  (h2 : account.last_month_deposit = 17)
  (h3 : account.current_balance = account.initial_balance + 16) :
  deposit_this_month account = 34 := by
  sorry

#eval deposit_this_month { initial_balance := 150, last_month_deposit := 17, current_balance := 166 }

end NUMINAMATH_CALUDE_deposit_amount_is_34_l1099_109922


namespace NUMINAMATH_CALUDE_base7_246_equals_132_l1099_109991

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_246_equals_132 :
  base7ToBase10 [6, 4, 2] = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_equals_132_l1099_109991


namespace NUMINAMATH_CALUDE_triangle_BC_length_l1099_109979

/-- Triangle ABC with given properties -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  AM : ℝ  -- Median from A to midpoint of BC
  area : ℝ
  BC : ℝ

/-- The triangle satisfies the given conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.AB = 6 ∧ t.AC = 8 ∧ t.AM = 5 ∧ t.area = 24

/-- Theorem: If a triangle satisfies the given conditions, its BC side length is 10 -/
theorem triangle_BC_length (t : Triangle) (h : satisfies_conditions t) : t.BC = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l1099_109979


namespace NUMINAMATH_CALUDE_art_show_ratio_l1099_109986

theorem art_show_ratio (total_painted : ℚ) (sold : ℚ) 
  (h1 : total_painted = 180.5)
  (h2 : sold = 76.3) :
  (total_painted - sold) / sold = 1042 / 763 := by
  sorry

end NUMINAMATH_CALUDE_art_show_ratio_l1099_109986


namespace NUMINAMATH_CALUDE_count_nine_digit_integers_l1099_109957

/-- The number of different 9-digit positive integers -/
def nine_digit_integers : ℕ := 9 * (10 ^ 8)

theorem count_nine_digit_integers : nine_digit_integers = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_count_nine_digit_integers_l1099_109957


namespace NUMINAMATH_CALUDE_common_measure_proof_l1099_109965

def segment1 : ℚ := 1/5
def segment2 : ℚ := 1/3
def commonMeasure : ℚ := 1/15

theorem common_measure_proof :
  (∃ (n m : ℕ), n * commonMeasure = segment1 ∧ m * commonMeasure = segment2) ∧
  (∀ (x : ℚ), x > 0 → (∃ (n m : ℕ), n * x = segment1 ∧ m * x = segment2) → x ≤ commonMeasure) :=
by sorry

end NUMINAMATH_CALUDE_common_measure_proof_l1099_109965


namespace NUMINAMATH_CALUDE_problem_1_l1099_109959

theorem problem_1 : 2 * Real.tan (45 * π / 180) + (-1/2)^0 + |Real.sqrt 3 - 1| = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1099_109959


namespace NUMINAMATH_CALUDE_intersection_sum_l1099_109909

-- Define the constants and variables
variable (n c : ℝ)
variable (x y : ℝ)

-- Define the two lines
def line1 (x : ℝ) : ℝ := n * x + 5
def line2 (x : ℝ) : ℝ := 4 * x + c

-- State the theorem
theorem intersection_sum (h1 : line1 5 = 15) (h2 : line2 5 = 15) : c + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1099_109909


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_two_l1099_109968

theorem x_over_y_is_negative_two (x y : ℝ) (h1 : 1 < (x - y) / (x + y))
  (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_two_l1099_109968


namespace NUMINAMATH_CALUDE_cube_volumes_from_surface_area_l1099_109990

theorem cube_volumes_from_surface_area :
  ∀ a b c : ℕ,
  (6 * (a^2 + b^2 + c^2) = 564) →
  (a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volumes_from_surface_area_l1099_109990


namespace NUMINAMATH_CALUDE_min_corners_8x8_grid_l1099_109999

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Represents a seven-cell corner --/
structure SevenCellCorner

/-- The number of cells in a seven-cell corner --/
def SevenCellCorner.cells : Nat := 7

/-- Checks if a given number of seven-cell corners can fit in the grid --/
def can_fit (g : Grid) (n : Nat) : Prop :=
  g.size * g.size ≥ n * SevenCellCorner.cells

/-- Checks if after clipping n seven-cell corners, no more can be clipped --/
def no_more_corners (g : Grid) (n : Nat) : Prop :=
  can_fit g n ∧ ¬can_fit g (n + 1)

/-- The main theorem: The minimum number of seven-cell corners that can be clipped from an 8x8 grid such that no more can be clipped is 3 --/
theorem min_corners_8x8_grid :
  ∃ (n : Nat), n = 3 ∧ no_more_corners (Grid.mk 8) n ∧ ∀ m < n, ¬no_more_corners (Grid.mk 8) m :=
sorry

end NUMINAMATH_CALUDE_min_corners_8x8_grid_l1099_109999


namespace NUMINAMATH_CALUDE_ice_cream_sales_l1099_109993

theorem ice_cream_sales (chocolate : ℕ) (mango : ℕ) 
  (h1 : chocolate = 50) 
  (h2 : mango = 54) : 
  chocolate + mango - (chocolate * 3 / 5 + mango * 2 / 3) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l1099_109993


namespace NUMINAMATH_CALUDE_point_reflection_x_axis_l1099_109911

/-- Given a point A(2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (2,-3). -/
theorem point_reflection_x_axis : 
  let A : ℝ × ℝ := (2, 3)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x A = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_reflection_x_axis_l1099_109911


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l1099_109939

def A (a : ℝ) : Set ℝ := {-1, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {9, 5-a, 4-a}

theorem intersection_implies_a_values (a : ℝ) :
  A a ∩ B a = {9} → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l1099_109939


namespace NUMINAMATH_CALUDE_inequality_proof_l1099_109930

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3)) + (y^n / (x^3 + y)) ≥ (2^(4-n)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1099_109930


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1099_109980

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 4 * a + k = 0) ∧ 
    (5 * b^2 + 4 * b + k = 0) ∧ 
    (|a - b| = a^2 + b^2)) ↔ 
  (k = 3/5 ∨ k = -12/5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1099_109980


namespace NUMINAMATH_CALUDE_rental_shop_problem_l1099_109961

/-- Rental shop problem -/
theorem rental_shop_problem 
  (first_hour_rate : ℝ) 
  (additional_hour_rate : ℝ)
  (sales_tax_rate : ℝ)
  (total_paid : ℝ)
  (h : ℕ)
  (h_def : h = (total_paid / (1 + sales_tax_rate) - first_hour_rate) / additional_hour_rate)
  (first_hour_rate_def : first_hour_rate = 25)
  (additional_hour_rate_def : additional_hour_rate = 10)
  (sales_tax_rate_def : sales_tax_rate = 0.08)
  (total_paid_def : total_paid = 125) :
  h + 1 = 10 := by
sorry


end NUMINAMATH_CALUDE_rental_shop_problem_l1099_109961


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_196_l1099_109996

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_196_l1099_109996


namespace NUMINAMATH_CALUDE_lcm_of_5_8_10_27_l1099_109987

theorem lcm_of_5_8_10_27 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 27)) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_8_10_27_l1099_109987


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l1099_109919

/-- Proves that for a complex number z with argument 60°, 
    if |z-1| is the geometric mean of |z| and |z-2|, then |z| = √2 + 1 -/
theorem complex_magnitude_proof (z : ℂ) :
  Complex.arg z = π / 3 →
  Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2) →
  Complex.abs z = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l1099_109919


namespace NUMINAMATH_CALUDE_trapezoid_area_l1099_109956

/-- Given an outer equilateral triangle with area 64 and an inner equilateral triangle
    with area 4, where the space between them is divided into three congruent trapezoids,
    prove that the area of one trapezoid is 20. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 64) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1099_109956


namespace NUMINAMATH_CALUDE_spending_difference_l1099_109970

def akeno_spending : ℕ := 2985

def lev_spending : ℕ := akeno_spending / 3

def ambrocio_spending : ℕ := lev_spending - 177

def total_difference : ℕ := akeno_spending - (lev_spending + ambrocio_spending)

theorem spending_difference :
  total_difference = 1172 :=
by sorry

end NUMINAMATH_CALUDE_spending_difference_l1099_109970


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1099_109958

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) :
  ∃ k : ℤ, (2*a - 1)^3 - (2*b - 1)^3 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1099_109958


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1099_109974

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  (total_players - throwers) % 3 = 0 →
  56 = throwers + (total_players - throwers) - (total_players - throwers) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1099_109974


namespace NUMINAMATH_CALUDE_car_stopping_distance_l1099_109927

/-- Represents the distance traveled by a car in a given second -/
def distance_per_second (n : ℕ) : ℕ :=
  max (40 - 10 * n) 0

/-- Calculates the total distance traveled by the car -/
def total_distance : ℕ :=
  (List.range 5).map distance_per_second |>.sum

/-- Theorem: The total distance traveled by the car is 100 feet -/
theorem car_stopping_distance : total_distance = 100 := by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_car_stopping_distance_l1099_109927


namespace NUMINAMATH_CALUDE_b_spending_percentage_l1099_109992

/-- Proves that B spends 85% of his salary given the specified conditions -/
theorem b_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (a_spending_rate : ℝ) 
  (h1 : total_salary = 2000)
  (h2 : a_salary = 1500)
  (h3 : a_spending_rate = 0.95)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  b_spending_rate = 0.85 := by
  sorry

#check b_spending_percentage

end NUMINAMATH_CALUDE_b_spending_percentage_l1099_109992


namespace NUMINAMATH_CALUDE_jungkook_has_most_apples_l1099_109971

def jungkook_initial : ℕ := 6
def jungkook_additional : ℕ := 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

def jungkook_total : ℕ := jungkook_initial + jungkook_additional

theorem jungkook_has_most_apples :
  jungkook_total > yoongi_apples ∧ jungkook_total > yuna_apples :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_most_apples_l1099_109971


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1099_109953

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1099_109953


namespace NUMINAMATH_CALUDE_max_value_of_a_l1099_109962

-- Define the operation
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem max_value_of_a :
  (∀ x : ℝ, matrix_op (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∀ b : ℝ, (∀ x : ℝ, matrix_op (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ 3/2) ∧
  (∃ x : ℝ, matrix_op (x - 1) (3/2 - 2) (3/2 + 1) x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1099_109962


namespace NUMINAMATH_CALUDE_geometric_progression_a10_l1099_109912

/-- A geometric progression with given conditions -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_a10 (a : ℕ → ℝ) :
  geometric_progression a → a 2 = 2 → a 6 = 162 → a 10 = 13122 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_a10_l1099_109912


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l1099_109994

/-- Represents the sum of the first k terms of a geometric sequence -/
noncomputable def S (a₁ q : ℝ) (k : ℕ) : ℝ :=
  if q = 1 then k * a₁ else a₁ * (1 - q^k) / (1 - q)

theorem geometric_sequence_sum_problem
  (a₁ q : ℝ)
  (h_pos : ∀ n : ℕ, 0 < a₁ * q^n)
  (h_Sn : S a₁ q n = 2)
  (h_S3n : S a₁ q (3*n) = 14) :
  S a₁ q (4*n) = 30 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l1099_109994


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l1099_109904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 918

-- Define a function that returns the molecular weight for any number of moles
def weight_for_moles (moles : ℝ) : ℝ := molecular_weight

-- Theorem stating that the molecular weight is constant regardless of the number of moles
theorem constant_molecular_weight (moles : ℝ) :
  weight_for_moles moles = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l1099_109904


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l1099_109966

theorem bucket_capacity_proof (x : ℝ) : 
  (12 * x = 132 * 5) → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l1099_109966


namespace NUMINAMATH_CALUDE_seats_per_bus_l1099_109938

/-- Given a school trip scenario with students and buses, calculate the number of seats per bus. -/
theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) :
  students / buses = 3 := by
  sorry


end NUMINAMATH_CALUDE_seats_per_bus_l1099_109938


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1099_109914

/-- A trinomial x^2 + 2ax + 9 is a perfect square if and only if a = ±3 -/
theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 2*a*x + 9 = (x + b)^2) ↔ (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1099_109914


namespace NUMINAMATH_CALUDE_multiply_by_112_equals_70000_l1099_109915

theorem multiply_by_112_equals_70000 (x : ℝ) : 112 * x = 70000 → x = 625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_112_equals_70000_l1099_109915


namespace NUMINAMATH_CALUDE_tan_255_degrees_l1099_109918

theorem tan_255_degrees : Real.tan (255 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_255_degrees_l1099_109918


namespace NUMINAMATH_CALUDE_return_journey_percentage_l1099_109973

/-- Represents the distance of a one-way trip -/
def one_way_distance : ℝ := 1

/-- Represents the total round-trip distance -/
def round_trip_distance : ℝ := 2 * one_way_distance

/-- Represents the percentage of the round-trip completed -/
def round_trip_completed_percentage : ℝ := 0.75

/-- Represents the distance traveled in the round-trip -/
def distance_traveled : ℝ := round_trip_completed_percentage * round_trip_distance

/-- Represents the distance traveled on the return journey -/
def return_journey_traveled : ℝ := distance_traveled - one_way_distance

theorem return_journey_percentage :
  return_journey_traveled / one_way_distance = 0.5 := by sorry

end NUMINAMATH_CALUDE_return_journey_percentage_l1099_109973
