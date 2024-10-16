import Mathlib

namespace NUMINAMATH_CALUDE_amber_guppies_l2016_201643

/-- The number of guppies in Amber's pond -/
theorem amber_guppies (initial_adults : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) :
  initial_adults + (first_batch_dozens * 12) + second_batch =
  initial_adults + first_batch_dozens * 12 + second_batch :=
by sorry

end NUMINAMATH_CALUDE_amber_guppies_l2016_201643


namespace NUMINAMATH_CALUDE_bicycle_discount_price_l2016_201650

theorem bicycle_discount_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 →
  discount1 = 0.4 →
  discount2 = 0.2 →
  original_price * (1 - discount1) * (1 - discount2) = 96 := by
sorry

end NUMINAMATH_CALUDE_bicycle_discount_price_l2016_201650


namespace NUMINAMATH_CALUDE_de_moivre_and_rationality_l2016_201644

/-- De Moivre's formula and its implication on rationality of trigonometric functions -/
theorem de_moivre_and_rationality (θ : ℝ) (n : ℕ) :
  (Complex.exp (θ * Complex.I))^n = Complex.exp (n * θ * Complex.I) ∧
  (∀ (a b : ℚ), Complex.exp (θ * Complex.I) = ↑a + ↑b * Complex.I →
    ∃ (c d : ℚ), Complex.exp (n * θ * Complex.I) = ↑c + ↑d * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_de_moivre_and_rationality_l2016_201644


namespace NUMINAMATH_CALUDE_tangent_surface_area_l2016_201686

/-- Given a sphere of radius R and a point S at distance 2R from the center,
    the surface area formed by tangent lines from S to the sphere is 3πR^2/2 -/
theorem tangent_surface_area (R : ℝ) (h : R > 0) :
  let sphere_radius := R
  let point_distance := 2 * R
  let surface_area := (3 / 2) * π * R^2
  surface_area = (3 / 2) * π * sphere_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_surface_area_l2016_201686


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_calculation_l2016_201616

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width height water_depth : ℝ) : ℝ :=
  length * width + 2 * (length * water_depth) + 2 * (width * water_depth)

/-- Theorem stating that the wet surface area of the given cistern is 387.5 m² -/
theorem cistern_wet_surface_area_calculation :
  cistern_wet_surface_area 15 10 8 4.75 = 387.5 := by
  sorry

#eval cistern_wet_surface_area 15 10 8 4.75

end NUMINAMATH_CALUDE_cistern_wet_surface_area_calculation_l2016_201616


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2016_201649

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (t : ℝ) :
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -5 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2016_201649


namespace NUMINAMATH_CALUDE_fibConversionAccuracy_l2016_201694

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the Fibonacci representation of a number
def fibRep (n : ℕ) : List ℕ := sorry

-- Define the conversion function using Fibonacci representation
def kmToMilesFib (km : ℕ) : ℕ := sorry

-- Define the exact conversion from km to miles
def kmToMilesExact (km : ℕ) : ℚ :=
  (km : ℚ) / 1.609

-- Main theorem
theorem fibConversionAccuracy :
  ∀ n : ℕ, n ≤ 100 →
    |((kmToMilesFib n : ℚ) - kmToMilesExact n)| < 2/3 := by sorry

end NUMINAMATH_CALUDE_fibConversionAccuracy_l2016_201694


namespace NUMINAMATH_CALUDE_population_trend_l2016_201648

theorem population_trend (P k : ℝ) (h1 : P > 0) (h2 : -1 < k) (h3 : k < 0) :
  ∀ n : ℕ, (P * (1 + k)^(n + 1)) < (P * (1 + k)^n) := by
  sorry

end NUMINAMATH_CALUDE_population_trend_l2016_201648


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2016_201602

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 3) ↔ (0 < x ∧ x < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2016_201602


namespace NUMINAMATH_CALUDE_problem_solution_l2016_201671

theorem problem_solution (a b c : ℝ) 
  (h1 : (6 * a + 34) ^ (1/3 : ℝ) = 4)
  (h2 : (5 * a + b - 2) ^ (1/2 : ℝ) = 5)
  (h3 : c = 9 ^ (1/2 : ℝ)) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2016_201671


namespace NUMINAMATH_CALUDE_particle_diameter_scientific_notation_l2016_201687

/-- Converts a decimal number to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem particle_diameter_scientific_notation :
  to_scientific_notation 0.00000021 = (2.1, -7) :=
sorry

end NUMINAMATH_CALUDE_particle_diameter_scientific_notation_l2016_201687


namespace NUMINAMATH_CALUDE_percentOutsideC_eq_61_11_l2016_201642

def gradeScale : List (Char × (Int × Int)) := [
  ('A', (94, 100)),
  ('B', (86, 93)),
  ('C', (76, 85)),
  ('D', (65, 75)),
  ('F', (0, 64))
]

def scores : List Int := [98, 73, 55, 100, 76, 93, 88, 72, 77, 65, 82, 79, 68, 85, 91, 56, 81, 89]

def isOutsideC (score : Int) : Bool :=
  score < 76 || score > 85

def countOutsideC : Nat :=
  scores.filter isOutsideC |>.length

theorem percentOutsideC_eq_61_11 :
  (countOutsideC : ℚ) / scores.length * 100 = 61.11 := by
  sorry

end NUMINAMATH_CALUDE_percentOutsideC_eq_61_11_l2016_201642


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2016_201637

-- Define the ratio between p and k
def ratio_p_k (p k : ℝ) : Prop := p / k = Real.sqrt 3

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop := 2 / Real.sqrt (k^2 + 1) = 1

-- Define the theorem
theorem p_sufficient_not_necessary (p q : ℝ) : 
  (∃ k, ratio_p_k p k ∧ is_tangent k) → 
  (∃ k, ratio_p_k q k ∧ is_tangent k) → 
  (∃ k, ratio_p_k p k ∧ is_tangent k → ∃ k', ratio_p_k q k' ∧ is_tangent k') ∧ 
  ¬(∀ k, ratio_p_k q k ∧ is_tangent k → ∃ k', ratio_p_k p k' ∧ is_tangent k') :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2016_201637


namespace NUMINAMATH_CALUDE_milan_phone_bill_l2016_201654

/-- Calculates the number of minutes billed given the total bill, monthly fee, and per-minute rate. -/
def minutes_billed (total_bill monthly_fee per_minute_rate : ℚ) : ℚ :=
  (total_bill - monthly_fee) / per_minute_rate

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : 
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let per_minute_rate : ℚ := 0.12
  minutes_billed total_bill monthly_fee per_minute_rate = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l2016_201654


namespace NUMINAMATH_CALUDE_odd_divides_power_two_minus_one_l2016_201680

theorem odd_divides_power_two_minus_one (a : ℕ) (h : Odd a) :
  ∃ b : ℕ, a ∣ (2^b - 1) := by sorry

end NUMINAMATH_CALUDE_odd_divides_power_two_minus_one_l2016_201680


namespace NUMINAMATH_CALUDE_current_rate_calculation_l2016_201689

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time_minutes : ℝ) 
  (h1 : boat_speed = 30) 
  (h2 : downstream_distance = 22.2) 
  (h3 : downstream_time_minutes = 36) : 
  ∃ (current_rate : ℝ), 
    (boat_speed + current_rate) * (downstream_time_minutes / 60) = downstream_distance ∧ 
    current_rate = 7 :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l2016_201689


namespace NUMINAMATH_CALUDE_luke_connor_sleep_difference_l2016_201630

theorem luke_connor_sleep_difference (connor_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  puppy_sleep = 16 →
  puppy_sleep = 2 * (connor_sleep + (puppy_sleep / 2 - connor_sleep)) →
  puppy_sleep / 2 - connor_sleep = 2 :=
by sorry

end NUMINAMATH_CALUDE_luke_connor_sleep_difference_l2016_201630


namespace NUMINAMATH_CALUDE_statement_d_not_always_true_l2016_201659

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h1 : ¬ parallel m n)
variable (h2 : α ≠ β)

-- State the theorem
theorem statement_d_not_always_true :
  ¬ (∀ (m : Line) (α β : Plane),
    plane_perpendicular α β →
    contained_in m α →
    perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_statement_d_not_always_true_l2016_201659


namespace NUMINAMATH_CALUDE_lisa_children_count_l2016_201657

/-- The number of Lisa's children -/
def num_children : ℕ := 4

/-- The number of spoons in the new cutlery set -/
def new_cutlery_spoons : ℕ := 25

/-- The number of decorative spoons -/
def decorative_spoons : ℕ := 2

/-- The number of baby spoons per child -/
def baby_spoons_per_child : ℕ := 3

/-- The total number of spoons Lisa has -/
def total_spoons : ℕ := 39

/-- Theorem stating that the number of Lisa's children is 4 -/
theorem lisa_children_count : 
  num_children * baby_spoons_per_child + new_cutlery_spoons + decorative_spoons = total_spoons :=
by sorry

end NUMINAMATH_CALUDE_lisa_children_count_l2016_201657


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2016_201631

theorem sum_x_y_equals_negative_one (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_one_l2016_201631


namespace NUMINAMATH_CALUDE_average_length_is_10_over_3_l2016_201669

-- Define the lengths of the strings
def string1_length : ℚ := 2
def string2_length : ℚ := 5
def string3_length : ℚ := 3

-- Define the number of strings
def num_strings : ℕ := 3

-- Define the average length calculation
def average_length : ℚ := (string1_length + string2_length + string3_length) / num_strings

-- Theorem statement
theorem average_length_is_10_over_3 : average_length = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_length_is_10_over_3_l2016_201669


namespace NUMINAMATH_CALUDE_cube_opposite_face_l2016_201632

-- Define a cube face
inductive Face : Type
| A | B | C | D | E | F

-- Define the property of being adjacent
def adjacent (x y : Face) : Prop := sorry

-- Define the property of sharing an edge
def sharesEdge (x y : Face) : Prop := sorry

-- Define the property of being opposite
def opposite (x y : Face) : Prop := sorry

-- Theorem statement
theorem cube_opposite_face :
  -- Conditions
  (sharesEdge Face.B Face.A) →
  (adjacent Face.C Face.B) →
  (¬ adjacent Face.C Face.A) →
  (sharesEdge Face.D Face.A) →
  (sharesEdge Face.D Face.F) →
  -- Conclusion
  (opposite Face.C Face.E) := by
sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l2016_201632


namespace NUMINAMATH_CALUDE_carla_games_won_l2016_201678

theorem carla_games_won (total_games : ℕ) (frankie_games : ℕ) (carla_games : ℕ) : 
  total_games = 30 →
  frankie_games = carla_games / 2 →
  frankie_games + carla_games = total_games →
  carla_games = 20 := by
  sorry

end NUMINAMATH_CALUDE_carla_games_won_l2016_201678


namespace NUMINAMATH_CALUDE_sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l2016_201633

theorem sum_of_squares_modulo_prime (p n : ℤ) (hp : Prime p) (hp5 : p > 5) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = n % p :=
sorry

theorem sum_of_squares_zero_modulo_prime_1mod4 (p : ℤ) (hp : Prime p) (hp1 : p % 4 = 1) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = 0 :=
sorry

theorem sum_of_squares_nonzero_modulo_prime_3mod4 (p : ℤ) (hp : Prime p) (hp3 : p % 4 = 3) :
  ∀ x y : ℤ, x % p ≠ 0 → y % p ≠ 0 → (x^2 + y^2) % p ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l2016_201633


namespace NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l2016_201624

theorem max_value_of_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁ + r₂ = r₁^n + r₂^n) →
  (∃ M : ℝ, M = (1 : ℝ) / r₁^2010 + (1 : ℝ) / r₂^2010 ∧ 
   ∀ t' q' r₁' r₂' : ℝ, 
     (∀ x, x^2 - t'*x + q' = 0 ↔ x = r₁' ∨ x = r₂') →
     (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁' + r₂' = r₁'^n + r₂'^n) →
     (1 : ℝ) / r₁'^2010 + (1 : ℝ) / r₂'^2010 ≤ M) →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l2016_201624


namespace NUMINAMATH_CALUDE_third_team_wins_l2016_201603

/-- Represents the amount of wood processed by a team of lumberjacks -/
structure WoodProcessed where
  amount : ℝ
  amount_pos : amount > 0

/-- The competition between three teams of lumberjacks -/
structure LumberjackCompetition where
  team1 : WoodProcessed
  team2 : WoodProcessed
  team3 : WoodProcessed
  first_third_twice_second : team1.amount + team3.amount = 2 * team2.amount
  second_third_thrice_first : team2.amount + team3.amount = 3 * team1.amount

/-- The third team processes the most wood in the competition -/
theorem third_team_wins (comp : LumberjackCompetition) : 
  comp.team3.amount > comp.team1.amount ∧ comp.team3.amount > comp.team2.amount := by
  sorry

#check third_team_wins

end NUMINAMATH_CALUDE_third_team_wins_l2016_201603


namespace NUMINAMATH_CALUDE_woman_completes_in_40_days_l2016_201653

-- Define the efficiency ratio between man and woman
def efficiency_ratio : ℝ := 1.25

-- Define the number of days it takes the man to complete the task
def man_days : ℝ := 32

-- Define the function to calculate the woman's days
def woman_days : ℝ := efficiency_ratio * man_days

-- Theorem to prove
theorem woman_completes_in_40_days : 
  woman_days = 40 := by sorry

end NUMINAMATH_CALUDE_woman_completes_in_40_days_l2016_201653


namespace NUMINAMATH_CALUDE_team_total_score_l2016_201675

/-- Given a team of 10 people in a shooting competition, prove that their total score is 905 points. -/
theorem team_total_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best : ℕ) 
  (hypothetical_average : ℕ) :
  team_size = 10 →
  best_score = 95 →
  hypothetical_best = 110 →
  hypothetical_average = 92 →
  (hypothetical_best - best_score + (team_size * hypothetical_average)) = (team_size * 905) :=
by sorry

end NUMINAMATH_CALUDE_team_total_score_l2016_201675


namespace NUMINAMATH_CALUDE_find_x_value_l2016_201672

theorem find_x_value (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l2016_201672


namespace NUMINAMATH_CALUDE_system_has_three_solutions_l2016_201699

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations -/
def system (x : ℝ) : Prop :=
  3 * x^2 - 45 * (floor x) + 60 = 0 ∧ 2 * x - 3 * (floor x) + 1 = 0

/-- The theorem stating that the system has exactly 3 real solutions -/
theorem system_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ system x :=
sorry

end NUMINAMATH_CALUDE_system_has_three_solutions_l2016_201699


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2016_201612

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ b, c * b^2 < a * b^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2016_201612


namespace NUMINAMATH_CALUDE_divisor_product_theorem_l2016_201656

/-- d(n) is the number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- s(n) is the sum of positive divisors of n -/
def s (n : ℕ) : ℕ := (Nat.divisors n).sum id

/-- The main theorem: s(x) * d(x) = 96 if and only if x is 14, 15, or 47 -/
theorem divisor_product_theorem (x : ℕ) : s x * d x = 96 ↔ x = 14 ∨ x = 15 ∨ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_divisor_product_theorem_l2016_201656


namespace NUMINAMATH_CALUDE_apple_purchase_l2016_201634

theorem apple_purchase (cecile_apples diane_apples : ℕ) : 
  diane_apples = cecile_apples + 20 →
  cecile_apples + diane_apples = 50 →
  cecile_apples = 15 := by
sorry

end NUMINAMATH_CALUDE_apple_purchase_l2016_201634


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2016_201683

/-- The number of ways to arrange 2 men and 4 women in a row, 
    such that no two men or two women are adjacent -/
def arrangement_count : ℕ := 240

/-- The number of positions between and at the ends of the men -/
def women_positions : ℕ := 5

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 4

theorem arrangement_theorem : 
  arrangement_count = women_positions.choose num_women * num_women.factorial * num_men.factorial :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2016_201683


namespace NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l2016_201645

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_three_squares 
  (small_square_area : ℝ) 
  (large_square_side_multiplier : ℝ) : 
  small_square_area = 1 →
  large_square_side_multiplier = 3 →
  2 * small_square_area + (large_square_side_multiplier^2 * small_square_area) = 11 :=
by
  sorry

#check rectangle_area_with_three_squares

end NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l2016_201645


namespace NUMINAMATH_CALUDE_river_width_is_8km_l2016_201685

/-- Represents the boat's journey across the river -/
structure RiverCrossing where
  boat_speed : ℝ
  current_speed : ℝ
  crossing_time : ℝ

/-- Calculates the width of the river based on the given conditions -/
def river_width (rc : RiverCrossing) : ℝ :=
  rc.boat_speed * rc.crossing_time

/-- Theorem stating that the width of the river is 8 km under the given conditions -/
theorem river_width_is_8km (rc : RiverCrossing) 
  (h1 : rc.boat_speed = 4)
  (h2 : rc.current_speed = 3)
  (h3 : rc.crossing_time = 2) : 
  river_width rc = 8 := by
  sorry

end NUMINAMATH_CALUDE_river_width_is_8km_l2016_201685


namespace NUMINAMATH_CALUDE_paco_marble_purchase_l2016_201676

theorem paco_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_paco_marble_purchase_l2016_201676


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l2016_201627

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 17.5 and standard deviation 2.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := ⟨17.5, 2.5, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_two_std_dev_below_mean_l2016_201627


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l2016_201628

def K : ℚ := (1 : ℚ) / 1 + 1 / 3 + 1 / 5 + 1 / 7 + 1 / 9

def T (n : ℕ) : ℚ := n * (5 ^ (n - 1)) * K

def is_integer (q : ℚ) : Prop := ∃ (m : ℤ), q = m

theorem smallest_n_for_integer_T :
  ∃ (n : ℕ), n > 0 ∧ is_integer (T n) ∧ ∀ (m : ℕ), 0 < m ∧ m < n → ¬is_integer (T m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l2016_201628


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l2016_201673

/-- Represents the cost and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silver_change_for_gold_payment (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak with 14 gold coins -/
theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1 = { silver_paid := 20, gold_change := 4 })
  (h2 : t2 = { silver_paid := 15, gold_change := 1 }) :
  silver_change_for_gold_payment t1 t2 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l2016_201673


namespace NUMINAMATH_CALUDE_platform_length_l2016_201658

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 54 →
  crossing_time = 25 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length = 175 := by sorry

end NUMINAMATH_CALUDE_platform_length_l2016_201658


namespace NUMINAMATH_CALUDE_impossible_heart_and_club_l2016_201622

-- Define a standard deck of cards
def StandardDeck : Type := Fin 52

-- Define suits
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

-- Define a function to get the suit of a card
def getSuit : StandardDeck → Suit := sorry

-- Theorem: The probability of drawing a card that is both Hearts and Clubs is 0
theorem impossible_heart_and_club (card : StandardDeck) : 
  ¬(getSuit card = Suit.Hearts ∧ getSuit card = Suit.Clubs) := by
  sorry

end NUMINAMATH_CALUDE_impossible_heart_and_club_l2016_201622


namespace NUMINAMATH_CALUDE_square_root_sum_l2016_201663

theorem square_root_sum (x : ℝ) (h : x + x⁻¹ = 3) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l2016_201663


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l2016_201655

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 82 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 63 := by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l2016_201655


namespace NUMINAMATH_CALUDE_bold_o_lit_cells_l2016_201610

/-- Represents a 5x5 grid with boolean values indicating lit (true) or unlit (false) cells. -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- The initial configuration of the letter 'o' on the grid. -/
def initial_o : Grid := sorry

/-- The number of lit cells in the initial 'o' configuration. -/
def initial_lit_cells : Nat := 12

/-- Makes a letter bold by lighting cells to the right of lit cells. -/
def make_bold (g : Grid) : Grid := sorry

/-- Counts the number of lit cells in a grid. -/
def count_lit_cells (g : Grid) : Nat := sorry

/-- Theorem stating that the number of lit cells in a bold 'o' is 24. -/
theorem bold_o_lit_cells :
  count_lit_cells (make_bold initial_o) = 24 := by sorry

end NUMINAMATH_CALUDE_bold_o_lit_cells_l2016_201610


namespace NUMINAMATH_CALUDE_five_to_fifth_sum_five_times_l2016_201619

theorem five_to_fifth_sum_five_times (n : ℕ) : 5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_five_to_fifth_sum_five_times_l2016_201619


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2016_201614

/-- Proves that given the specified investment conditions, the unknown interest rate is 8% --/
theorem investment_interest_rate : 
  ∀ (x y r : ℚ),
  x + y = 2000 →
  y = 650 →
  x * (1/10) - y * r = 83 →
  r = 8/100 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2016_201614


namespace NUMINAMATH_CALUDE_solution_pairs_l2016_201635

theorem solution_pairs : 
  {p : ℕ × ℕ | let (m, n) := p; m^2 + 2 * 3^n = m * (2^(n+1) - 1)} = 
  {(9, 3), (6, 3), (9, 5), (54, 5)} :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l2016_201635


namespace NUMINAMATH_CALUDE_meet_on_same_side_time_l2016_201647

/-- The time when two points moving on a square meet on the same side for the first time -/
def time_to_meet_on_same_side (side_length : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  35

/-- Theorem stating that the time to meet on the same side is 35 seconds under given conditions -/
theorem meet_on_same_side_time :
  time_to_meet_on_same_side 100 5 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_meet_on_same_side_time_l2016_201647


namespace NUMINAMATH_CALUDE_two_color_distance_l2016_201677

/-- A type representing colors --/
inductive Color
| Red
| Blue

/-- A two-coloring of the plane --/
def Coloring := ℝ × ℝ → Color

/-- Predicate to check if both colors are used in a coloring --/
def BothColorsUsed (c : Coloring) : Prop :=
  (∃ p : ℝ × ℝ, c p = Color.Red) ∧ (∃ p : ℝ × ℝ, c p = Color.Blue)

/-- The main theorem --/
theorem two_color_distance (c : Coloring) (h : BothColorsUsed c) (a : ℝ) (ha : a > 0) :
  ∃ p q : ℝ × ℝ, c p ≠ c q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = a :=
sorry

end NUMINAMATH_CALUDE_two_color_distance_l2016_201677


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2016_201625

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > x + 1 ∧ (4 * x - 5) / 3 ≤ x) ↔ (1 < x ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2016_201625


namespace NUMINAMATH_CALUDE_reading_time_difference_l2016_201611

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference 
  (jonathan_speed : ℝ) 
  (alice_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : jonathan_speed = 150) 
  (h2 : alice_speed = 75) 
  (h3 : book_pages = 450) : 
  (book_pages / alice_speed - book_pages / jonathan_speed) * 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2016_201611


namespace NUMINAMATH_CALUDE_jenny_meal_combinations_l2016_201620

/-- Represents the number of choices for each meal component -/
structure MealChoices where
  mainDishes : Nat
  drinks : Nat
  desserts : Nat
  sideDishes : Nat

/-- Calculates the total number of possible meal combinations -/
def totalMealCombinations (choices : MealChoices) : Nat :=
  choices.mainDishes * choices.drinks * choices.desserts * choices.sideDishes

/-- Theorem stating that Jenny can arrange 48 distinct possible meals -/
theorem jenny_meal_combinations :
  let jennyChoices : MealChoices := {
    mainDishes := 4,
    drinks := 2,
    desserts := 2,
    sideDishes := 3
  }
  totalMealCombinations jennyChoices = 48 := by
  sorry

end NUMINAMATH_CALUDE_jenny_meal_combinations_l2016_201620


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2016_201697

theorem intersection_point_of_lines : ∃! p : ℚ × ℚ, 
  (3 * p.2 = -2 * p.1 + 6) ∧ (4 * p.2 = 3 * p.1 - 4) ∧ 
  p = (36/17, 10/17) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2016_201697


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l2016_201682

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 6}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals_set : M ∩ (I \ N) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l2016_201682


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l2016_201607

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 1.20

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 1.70

/-- The number of color copies -/
def num_copies : ℕ := 70

/-- The difference in charge between print shop Y and print shop X for a given number of copies -/
def charge_difference (n : ℕ) : ℚ := n * (cost_y - cost_x)

theorem print_shop_charge_difference : 
  charge_difference num_copies = 35 := by sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l2016_201607


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2016_201661

theorem chess_tournament_participants (total_games : ℕ) : total_games = 378 →
  ∃! n : ℕ, n * (n - 1) / 2 = total_games :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2016_201661


namespace NUMINAMATH_CALUDE_angle_S_measure_l2016_201692

/-- Represents a convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angle measures
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  -- Convexity and angle sum property
  sum_angles : P + Q + R + S + T = 540
  -- Angle congruence properties
  PQR_congruent : P = Q ∧ Q = R
  ST_congruent : S = T
  -- Relation between P and S
  P_less_than_S : P + 30 = S

/-- 
Theorem: In a convex pentagon with the given properties, 
the measure of angle S is 126 degrees.
-/
theorem angle_S_measure (pentagon : ConvexPentagon) : pentagon.S = 126 := by
  sorry

end NUMINAMATH_CALUDE_angle_S_measure_l2016_201692


namespace NUMINAMATH_CALUDE_least_froods_for_more_points_l2016_201660

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n Froods -/
def eating_points (n : ℕ) : ℕ := 5 * n

/-- Proposition: 10 is the least positive integer n for which 
    dropping n Froods earns more points than eating them -/
theorem least_froods_for_more_points : 
  ∀ n : ℕ, n > 0 → (
    (n < 10 → sum_first_n n ≤ eating_points n) ∧
    (sum_first_n 10 > eating_points 10)
  ) := by sorry

end NUMINAMATH_CALUDE_least_froods_for_more_points_l2016_201660


namespace NUMINAMATH_CALUDE_max_ellipse_area_in_rectangle_l2016_201623

/-- The maximum area of an ellipse inside a rectangle -/
theorem max_ellipse_area_in_rectangle (π : ℝ) (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 18 ∧ rectangle_width = 14 →
  let semi_major_axis := rectangle_length / 2
  let semi_minor_axis := rectangle_width / 2
  let max_area := π * semi_major_axis * semi_minor_axis
  max_area = π * 63 :=
by sorry

end NUMINAMATH_CALUDE_max_ellipse_area_in_rectangle_l2016_201623


namespace NUMINAMATH_CALUDE_unique_solution_implies_sqrt_three_l2016_201601

theorem unique_solution_implies_sqrt_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 3 = 0) → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_sqrt_three_l2016_201601


namespace NUMINAMATH_CALUDE_area_difference_zero_l2016_201639

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- A point inside the polygon -/
def InteriorPoint (p : RegularPolygon n) := ℝ × ℝ

/-- The area difference function between black and white triangles -/
def areaDifference (p : RegularPolygon n) (point : InteriorPoint p) : ℝ := sorry

/-- Theorem stating that the area difference is always zero -/
theorem area_difference_zero (n : ℕ) (p : RegularPolygon n) (point : InteriorPoint p) :
  areaDifference p point = 0 := by sorry

end NUMINAMATH_CALUDE_area_difference_zero_l2016_201639


namespace NUMINAMATH_CALUDE_statue_selling_price_l2016_201667

/-- The selling price of a statue given its cost and profit percentage -/
def selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

/-- Theorem: The selling price of a statue that costs $536 and is sold at a 25% profit is $670 -/
theorem statue_selling_price : 
  selling_price 536 0.25 = 670 := by
  sorry

end NUMINAMATH_CALUDE_statue_selling_price_l2016_201667


namespace NUMINAMATH_CALUDE_unique_bases_sum_l2016_201652

def recurring_decimal (a b : ℕ) (base : ℕ) : ℚ :=
  (a : ℚ) / (base ^ 2 - 1 : ℚ) * base + (b : ℚ) / (base ^ 2 - 1 : ℚ)

theorem unique_bases_sum :
  ∃! (R₁ R₂ : ℕ), 
    R₁ > 1 ∧ R₂ > 1 ∧
    recurring_decimal 3 7 R₁ = recurring_decimal 2 5 R₂ ∧
    recurring_decimal 7 3 R₁ = recurring_decimal 5 2 R₂ ∧
    R₁ + R₂ = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_bases_sum_l2016_201652


namespace NUMINAMATH_CALUDE_homework_problem_l2016_201609

theorem homework_problem (total : ℕ) (finished_ratio unfinished_ratio : ℕ) 
  (h_total : total = 65)
  (h_ratio : finished_ratio = 9 ∧ unfinished_ratio = 4) :
  (finished_ratio * total) / (finished_ratio + unfinished_ratio) = 45 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l2016_201609


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2016_201608

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x - 1 ∧ x - 1 ≤ 4}
def N : Set ℝ := {x | x^2 < 25}

-- Define the complement of M in ℝ
def C_R_M : Set ℝ := {x | x ∉ M}

-- State the theorem
theorem complement_M_intersect_N :
  (C_R_M ∩ N) = {x : ℝ | -5 < x ∧ x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2016_201608


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2016_201666

theorem polynomial_evaluation (y : ℝ) (h1 : y > 0) (h2 : y^2 - 3*y - 9 = 0) :
  y^3 - 3*y^2 - 9*y + 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2016_201666


namespace NUMINAMATH_CALUDE_total_votes_is_330_l2016_201615

/-- Proves that the total number of votes is 330 given the specified conditions -/
theorem total_votes_is_330 :
  ∀ (total_votes votes_for votes_against : ℕ),
    votes_for = votes_against + 66 →
    votes_against = (40 * total_votes) / 100 →
    total_votes = votes_for + votes_against →
    total_votes = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_is_330_l2016_201615


namespace NUMINAMATH_CALUDE_total_transport_is_405_l2016_201688

/-- Calculates the total number of people transported by two boats over two days -/
def total_people_transported (boat_a_capacity : ℕ) (boat_b_capacity : ℕ)
  (day1_a_trips : ℕ) (day1_b_trips : ℕ)
  (day2_a_trips : ℕ) (day2_b_trips : ℕ) : ℕ :=
  (boat_a_capacity * day1_a_trips + boat_b_capacity * day1_b_trips) +
  (boat_a_capacity * day2_a_trips + boat_b_capacity * day2_b_trips)

/-- Theorem stating that the total number of people transported is 405 -/
theorem total_transport_is_405 :
  total_people_transported 20 15 7 5 5 6 = 405 := by
  sorry

#eval total_people_transported 20 15 7 5 5 6

end NUMINAMATH_CALUDE_total_transport_is_405_l2016_201688


namespace NUMINAMATH_CALUDE_min_surface_area_to_volume_ratio_cylinder_in_sphere_l2016_201606

/-- For a right circular cylinder inscribed in a sphere of radius R,
    the minimum ratio of surface area to volume is (((4^(1/3)) + 1)^(3/2)) / R. -/
theorem min_surface_area_to_volume_ratio_cylinder_in_sphere (R : ℝ) (h : R > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ r^2 + (h/2)^2 = R^2 ∧
    ∀ (r' h' : ℝ), r' > 0 → h' > 0 → r'^2 + (h'/2)^2 = R^2 →
      (2 * π * r * (r + h)) / (π * r^2 * h) ≥ (((4^(1/3) : ℝ) + 1)^(3/2)) / R :=
by sorry

end NUMINAMATH_CALUDE_min_surface_area_to_volume_ratio_cylinder_in_sphere_l2016_201606


namespace NUMINAMATH_CALUDE_work_earnings_equality_l2016_201662

theorem work_earnings_equality (t : ℚ) : 
  (t + 3) * (3 * t - 1) = (3 * t - 7) * (t + 4) + 5 → t = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l2016_201662


namespace NUMINAMATH_CALUDE_fuel_used_calculation_l2016_201605

/-- Calculates the total fuel used given the initial capacity, intermediate reading, and final reading after refill -/
def total_fuel_used (initial_capacity : ℝ) (intermediate_reading : ℝ) (final_reading : ℝ) : ℝ :=
  (initial_capacity - intermediate_reading) + (initial_capacity - final_reading)

/-- Theorem stating that the total fuel used is 4582 L given the specific readings -/
theorem fuel_used_calculation :
  let initial_capacity : ℝ := 3000
  let intermediate_reading : ℝ := 180
  let final_reading : ℝ := 1238
  total_fuel_used initial_capacity intermediate_reading final_reading = 4582 := by
  sorry

#eval total_fuel_used 3000 180 1238

end NUMINAMATH_CALUDE_fuel_used_calculation_l2016_201605


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2016_201651

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence a_k defined recursively -/
def sequenceA (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => sequenceA A k + sumOfDigits (sequenceA A k)

/-- A number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The sequence eventually becomes constant -/
def eventuallyConstant (A : ℕ) : Prop := ∃ N : ℕ, ∀ k ≥ N, sequenceA A k = sequenceA A N

/-- Main theorem -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  eventuallyConstant A ↔ isPerfectSquare A := by sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2016_201651


namespace NUMINAMATH_CALUDE_amelia_win_probability_l2016_201681

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 2/7

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Amelia getting two heads in one turn -/
def p_amelia_win_turn : ℚ := p_amelia ^ 2

/-- Probability of Blaine getting two heads in one turn -/
def p_blaine_win_turn : ℚ := p_blaine ^ 2

/-- Probability of neither player winning in one round -/
def p_no_win_round : ℚ := (1 - p_amelia_win_turn) * (1 - p_blaine_win_turn)

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win_turn / (1 - p_no_win_round)) = 4/9 :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l2016_201681


namespace NUMINAMATH_CALUDE_jellybean_probability_l2016_201626

/-- The number of jellybean colors -/
def num_colors : ℕ := 5

/-- The number of jellybeans in the sample -/
def sample_size : ℕ := 5

/-- The probability of selecting exactly 2 distinct colors when randomly choosing
    5 jellybeans from a set of 5 equally proportioned colors -/
theorem jellybean_probability : 
  (num_colors.choose 2 * (2^sample_size - 2)) / (num_colors^sample_size) = 12/125 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2016_201626


namespace NUMINAMATH_CALUDE_problem_statement_l2016_201693

theorem problem_statement (x y : ℝ) (hx : x = 1/3) (hy : y = 3) :
  (1/4) * x^3 * y^8 = 60.75 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2016_201693


namespace NUMINAMATH_CALUDE_product_profit_l2016_201679

theorem product_profit (original_price : ℝ) (cost_price : ℝ) : 
  cost_price > 0 →
  original_price > 0 →
  (0.8 * original_price = 1.2 * cost_price) →
  (original_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_product_profit_l2016_201679


namespace NUMINAMATH_CALUDE_dividend_calculation_l2016_201638

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 140 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2016_201638


namespace NUMINAMATH_CALUDE_media_team_selection_count_l2016_201690

/-- The number of domestic media teams -/
def domestic_teams : ℕ := 6

/-- The number of foreign media teams -/
def foreign_teams : ℕ := 3

/-- The total number of teams to be selected -/
def selected_teams : ℕ := 3

/-- Represents whether domestic teams can ask questions consecutively -/
def consecutive_domestic : Prop := False

theorem media_team_selection_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_media_team_selection_count_l2016_201690


namespace NUMINAMATH_CALUDE_unique_factorial_equation_l2016_201646

theorem unique_factorial_equation : ∃! (N : ℕ), N > 0 ∧ ∃ (m : ℕ), m > 0 ∧ (7 : ℕ).factorial * (11 : ℕ).factorial = 20 * m * N.factorial := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_equation_l2016_201646


namespace NUMINAMATH_CALUDE_equation_solutions_l2016_201618

theorem equation_solutions (x : ℝ) : 
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) ↔ 
  (x = 10 ∨ x = -1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2016_201618


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l2016_201684

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) → 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l2016_201684


namespace NUMINAMATH_CALUDE_james_toy_ratio_l2016_201696

/-- Given that James buys toy soldiers and toy cars, with 20 toy cars and a total of 60 toys,
    prove that the ratio of toy soldiers to toy cars is 2:1. -/
theorem james_toy_ratio :
  let total_toys : ℕ := 60
  let toy_cars : ℕ := 20
  let toy_soldiers : ℕ := total_toys - toy_cars
  (toy_soldiers : ℚ) / toy_cars = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_james_toy_ratio_l2016_201696


namespace NUMINAMATH_CALUDE_plant_structure_l2016_201668

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total count of parts in the plant (main stem + branches + small branches). -/
def Plant.totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- The plant satisfies the given conditions. -/
def validPlant (p : Plant) : Prop :=
  p.branches = p.smallBranchesPerBranch ∧ p.totalCount = 43

theorem plant_structure : ∃ (p : Plant), validPlant p ∧ p.smallBranchesPerBranch = 6 := by
  sorry

end NUMINAMATH_CALUDE_plant_structure_l2016_201668


namespace NUMINAMATH_CALUDE_sum_of_digits_A_squared_l2016_201600

/-- For a number with n digits, all being 9 -/
def A (n : ℕ) : ℕ := 10^n - 1

/-- Sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_A_squared :
  sum_of_digits ((A 221)^2) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_A_squared_l2016_201600


namespace NUMINAMATH_CALUDE_last_digit_sum_l2016_201674

def is_valid_pair (a b : Nat) : Prop :=
  (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

def valid_sequence (s : List Nat) : Prop :=
  s.length = 2000 ∧
  s.head? = some 3 ∧
  ∀ i, i < 1999 → is_valid_pair (s.get! i) (s.get! (i + 1))

theorem last_digit_sum (s : List Nat) (a b : Nat) :
  valid_sequence s →
  (s.getLast? = some a ∨ s.getLast? = some b) →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_last_digit_sum_l2016_201674


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2016_201691

/-- Represents a seating arrangement of adults and children -/
def SeatingArrangement := Fin 6 → Bool

/-- Checks if a seating arrangement is valid (no two children sit next to each other) -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 6, arrangement i → arrangement ((i + 1) % 6) → False

/-- The number of valid seating arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 72 valid seating arrangements -/
theorem seating_arrangements_count :
  num_valid_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2016_201691


namespace NUMINAMATH_CALUDE_batsman_highest_score_l2016_201665

theorem batsman_highest_score 
  (total_innings : ℕ)
  (total_average : ℚ)
  (score_difference : ℕ)
  (remaining_innings : ℕ)
  (remaining_average : ℚ)
  (h : total_innings = 46)
  (i : total_average = 61)
  (j : score_difference = 150)
  (k : remaining_innings = 44)
  (l : remaining_average = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    highest_score + lowest_score = total_innings * total_average - remaining_innings * remaining_average ∧
    highest_score = 202 := by
  sorry

#check batsman_highest_score

end NUMINAMATH_CALUDE_batsman_highest_score_l2016_201665


namespace NUMINAMATH_CALUDE_sine_sum_equality_l2016_201613

theorem sine_sum_equality (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l2016_201613


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2016_201664

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2016_201664


namespace NUMINAMATH_CALUDE_f_continuous_not_bounded_variation_l2016_201670

/-- The function f(x) = x sin(1/x) for x ≠ 0 and f(0) = 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x * Real.sin (1 / x) else 0

/-- The interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

theorem f_continuous_not_bounded_variation :
  ContinuousOn f I ∧ ¬ BoundedVariationOn f I := by sorry

end NUMINAMATH_CALUDE_f_continuous_not_bounded_variation_l2016_201670


namespace NUMINAMATH_CALUDE_triangle_properties_l2016_201629

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 - b^2 + a*c = 0 →
  (∃ (p : ℝ), p = a + b + c) →
  B = 2*π/3 ∧ (b = 2*Real.sqrt 3 → ∃ (p_max : ℝ), p_max = 4 + 2*Real.sqrt 3 ∧ ∀ p, p ≤ p_max) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2016_201629


namespace NUMINAMATH_CALUDE_negative_result_operation_only_A_is_negative_l2016_201698

theorem negative_result_operation : ℤ → Prop :=
  fun x => x < 0

theorem only_A_is_negative :
  negative_result_operation ((-1) + (-3)) ∧
  ¬negative_result_operation (6 - (-3)) ∧
  ¬negative_result_operation ((-3) * (-2)) ∧
  ¬negative_result_operation (0 / (-7)) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_result_operation_only_A_is_negative_l2016_201698


namespace NUMINAMATH_CALUDE_least_x_value_l2016_201604

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 12 * p * q) : 
  x ≥ 72 ∧ (∃ x₀ : ℕ, x₀ ≥ 72 → 
    (∃ p₀ q₀ : ℕ, Nat.Prime p₀ ∧ Nat.Prime q₀ ∧ q₀ % 2 = 1 ∧ x₀ = 12 * p₀ * q₀) → x₀ ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_least_x_value_l2016_201604


namespace NUMINAMATH_CALUDE_x_value_l2016_201640

theorem x_value : ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((4 * x) / 3) = x ∧ x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2016_201640


namespace NUMINAMATH_CALUDE_expression_evaluation_l2016_201621

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 4 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2016_201621


namespace NUMINAMATH_CALUDE_min_value_cyclic_sum_l2016_201636

theorem min_value_cyclic_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 ∧
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_sum_l2016_201636


namespace NUMINAMATH_CALUDE_largest_initial_number_l2016_201617

/-- Represents a sequence of five additions -/
structure FiveAdditions (n : ℕ) :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)
  (sum_eq : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100)
  (not_div₁ : ¬(n % a₁ = 0))
  (not_div₂ : ¬((n + a₁) % a₂ = 0))
  (not_div₃ : ¬((n + a₁ + a₂) % a₃ = 0))
  (not_div₄ : ¬((n + a₁ + a₂ + a₃) % a₄ = 0))
  (not_div₅ : ¬((n + a₁ + a₂ + a₃ + a₄) % a₅ = 0))

/-- The main theorem stating that 89 is the largest initial number -/
theorem largest_initial_number :
  (∃ (f : FiveAdditions 89), True) ∧
  (∀ n > 89, ¬∃ (f : FiveAdditions n), True) :=
sorry

end NUMINAMATH_CALUDE_largest_initial_number_l2016_201617


namespace NUMINAMATH_CALUDE_find_t_l2016_201695

-- Define vectors in R²
def AB : Fin 2 → ℝ := ![2, 3]
def AC : ℝ → Fin 2 → ℝ := λ t => ![3, t]

-- Define the dot product of two vectors in R²
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the perpendicular condition
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w = 0

-- Theorem statement
theorem find_t : ∃ t : ℝ, 
  perpendicular AB (λ i => AC t i - AB i) ∧ t = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l2016_201695


namespace NUMINAMATH_CALUDE_max_prize_winners_l2016_201641

/-- Represents a tournament with given number of players and point thresholds. -/
structure Tournament :=
  (num_players : ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)
  (prize_threshold : ℕ)

/-- Calculates the total number of games in a round-robin tournament. -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total points available in the tournament. -/
def total_points (t : Tournament) : ℕ :=
  total_games t.num_players * t.win_points

/-- Theorem stating the maximum number of prize winners in the specific tournament. -/
theorem max_prize_winners (t : Tournament) 
  (h1 : t.num_players = 15)
  (h2 : t.win_points = 2)
  (h3 : t.draw_points = 1)
  (h4 : t.loss_points = 0)
  (h5 : t.prize_threshold = 20) :
  ∃ (max_winners : ℕ), max_winners = 9 ∧ 
  (∀ (n : ℕ), n > max_winners → 
    n * t.prize_threshold > total_points t) :=
sorry

end NUMINAMATH_CALUDE_max_prize_winners_l2016_201641
