import Mathlib

namespace NUMINAMATH_CALUDE_a5_b5_ratio_l1402_140204

def S (n : ℕ+) : ℝ := sorry
def T (n : ℕ+) : ℝ := sorry
def a : ℕ+ → ℝ := sorry
def b : ℕ+ → ℝ := sorry

axiom arithmetic_sum_property (n : ℕ+) : S n / T n = (n + 1) / (2 * n - 1)

theorem a5_b5_ratio : a 5 / b 5 = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_a5_b5_ratio_l1402_140204


namespace NUMINAMATH_CALUDE_black_car_overtakes_l1402_140262

/-- Represents the scenario of three cars racing on a highway -/
structure CarRace where
  red_speed : ℝ
  green_speed : ℝ
  black_speed : ℝ
  red_black_distance : ℝ
  black_green_distance : ℝ

/-- Theorem stating the condition for the black car to overtake the red car before the green car overtakes the black car -/
theorem black_car_overtakes (race : CarRace) 
  (h1 : race.red_speed = 40)
  (h2 : race.green_speed = 60)
  (h3 : race.red_black_distance = 10)
  (h4 : race.black_green_distance = 5)
  (h5 : race.black_speed > 40) :
  race.black_speed > 53.33 ↔ 
  (10 / (race.black_speed - 40) < 5 / (60 - race.black_speed)) := by
  sorry

end NUMINAMATH_CALUDE_black_car_overtakes_l1402_140262


namespace NUMINAMATH_CALUDE_living_room_to_bedroom_ratio_l1402_140278

/-- Energy usage of lights in Noah's house -/
def energy_usage (bedroom_watts_per_hour : ℝ) (hours : ℝ) (total_watts : ℝ) : Prop :=
  let bedroom_energy := bedroom_watts_per_hour * hours
  let office_energy := 3 * bedroom_energy
  let living_room_energy := total_watts - bedroom_energy - office_energy
  (living_room_energy / bedroom_energy = 4)

/-- Theorem: The ratio of living room light energy to bedroom light energy is 4:1 -/
theorem living_room_to_bedroom_ratio :
  energy_usage 6 2 96 := by
  sorry

end NUMINAMATH_CALUDE_living_room_to_bedroom_ratio_l1402_140278


namespace NUMINAMATH_CALUDE_rest_of_body_length_l1402_140282

theorem rest_of_body_length (total_height : ℝ) (leg_ratio : ℝ) (head_ratio : ℝ) 
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1/3)
  (h3 : head_ratio = 1/4) :
  total_height - (leg_ratio * total_height) - (head_ratio * total_height) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rest_of_body_length_l1402_140282


namespace NUMINAMATH_CALUDE_sugar_packet_weight_l1402_140244

-- Define the number of packets sold per week
def packets_per_week : ℕ := 20

-- Define the total weight of sugar sold per week in kilograms
def total_weight_kg : ℕ := 2

-- Define the conversion factor from kilograms to grams
def kg_to_g : ℕ := 1000

-- Theorem stating that each packet weighs 100 grams
theorem sugar_packet_weight :
  (total_weight_kg * kg_to_g) / packets_per_week = 100 := by
sorry

end NUMINAMATH_CALUDE_sugar_packet_weight_l1402_140244


namespace NUMINAMATH_CALUDE_log_equation_sum_l1402_140209

theorem log_equation_sum (A B C : ℕ+) (h_coprime : Nat.Coprime A B ∧ Nat.Coprime A C ∧ Nat.Coprime B C)
  (h_eq : A * Real.log 5 / Real.log 180 + B * Real.log 3 / Real.log 180 + C * Real.log 2 / Real.log 180 = 1) :
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l1402_140209


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l1402_140226

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7, where a, b, and c are constants,
    if f(-2011) = -17, then f(2011) = 31. -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  f (-2011) = -17 → f 2011 = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l1402_140226


namespace NUMINAMATH_CALUDE_prob_hit_third_shot_prob_hit_at_least_once_l1402_140283

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the probability of hitting the target on the 3rd shot
theorem prob_hit_third_shot : 
  hit_probability = 0.9 := by sorry

-- Theorem for the probability of hitting the target at least once
theorem prob_hit_at_least_once : 
  1 - (1 - hit_probability) ^ num_shots = 1 - 0.1 ^ 4 := by sorry

end NUMINAMATH_CALUDE_prob_hit_third_shot_prob_hit_at_least_once_l1402_140283


namespace NUMINAMATH_CALUDE_monochromatic_4cycle_exists_l1402_140232

/-- A color for an edge -/
inductive Color
| Red
| Blue

/-- A graph with 6 vertices -/
def Graph6 := Fin 6 → Fin 6 → Color

/-- A 4-cycle in a graph -/
def IsCycle4 (g : Graph6) (v1 v2 v3 v4 : Fin 6) (c : Color) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v4 ∧ v4 ≠ v1 ∧
  g v1 v2 = c ∧ g v2 v3 = c ∧ g v3 v4 = c ∧ g v4 v1 = c

/-- The main theorem: every 6-vertex complete graph with red/blue edges contains a monochromatic 4-cycle -/
theorem monochromatic_4cycle_exists (g : Graph6) 
  (complete : ∀ u v : Fin 6, u ≠ v → (g u v = Color.Red ∨ g u v = Color.Blue)) :
  ∃ (v1 v2 v3 v4 : Fin 6) (c : Color), IsCycle4 g v1 v2 v3 v4 c :=
sorry

end NUMINAMATH_CALUDE_monochromatic_4cycle_exists_l1402_140232


namespace NUMINAMATH_CALUDE_proposition_evaluations_l1402_140214

theorem proposition_evaluations :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 + x - 6 ≥ 0) ∧
  (∃ x : ℝ, x ≠ 2 ∧ x^2 - 5*x + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluations_l1402_140214


namespace NUMINAMATH_CALUDE_purum_elementary_students_l1402_140256

theorem purum_elementary_students (total : ℕ) (difference : ℕ) : total = 41 → difference = 3 →
  ∃ purum non_purum : ℕ, purum = non_purum + difference ∧ purum + non_purum = total ∧ purum = 22 :=
by sorry

end NUMINAMATH_CALUDE_purum_elementary_students_l1402_140256


namespace NUMINAMATH_CALUDE_sixth_power_of_sqrt_two_plus_sqrt_two_l1402_140246

theorem sixth_power_of_sqrt_two_plus_sqrt_two :
  (Real.sqrt (2 + Real.sqrt 2)) ^ 6 = 16 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_sqrt_two_plus_sqrt_two_l1402_140246


namespace NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l1402_140249

/-- Proves the equivalence of rectangular and polar coordinate equations --/
theorem rectangular_to_polar_equivalence 
  (x y ρ θ : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) : 
  y^2 = 12*x ↔ ρ * Real.sin θ^2 = 12 * Real.cos θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_equivalence_l1402_140249


namespace NUMINAMATH_CALUDE_kevins_stamps_l1402_140234

theorem kevins_stamps (carl_stamps : ℕ) (difference : ℕ) (h1 : carl_stamps = 89) (h2 : difference = 32) :
  carl_stamps - difference = 57 := by
  sorry

end NUMINAMATH_CALUDE_kevins_stamps_l1402_140234


namespace NUMINAMATH_CALUDE_apple_running_rate_l1402_140247

/-- Given Mac's and Apple's running rates, prove that Apple's rate is 3 miles per hour -/
theorem apple_running_rate (mac_rate apple_rate : ℝ) : 
  mac_rate = 4 →  -- Mac's running rate is 4 miles per hour
  (24 / mac_rate) * 60 + 120 = (24 / apple_rate) * 60 →  -- Mac runs 24 miles 120 minutes faster than Apple
  apple_rate = 3 :=  -- Apple's running rate is 3 miles per hour
by
  sorry


end NUMINAMATH_CALUDE_apple_running_rate_l1402_140247


namespace NUMINAMATH_CALUDE_value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l1402_140248

/-- A function is value-preserving on an interval [a, b] if it is monotonic
and its range on [a, b] is exactly [a, b] -/
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The function f(x) = 2x² has a unique value-preserving interval [0, 1/2] -/
theorem value_preserving_2x_squared :
  ∃! (a b : ℝ), is_value_preserving (fun x ↦ 2 * x^2) a b ∧ a = 0 ∧ b = 1/2 :=
sorry

/-- The function g(x) = x² - 2x + m has value-preserving intervals
if and only if m ∈ [1, 5/4) ∪ [2, 9/4) -/
theorem value_preserving_x_squared_minus_2x_plus_m (m : ℝ) :
  (∃ a b, is_value_preserving (fun x ↦ x^2 - 2*x + m) a b) ↔ 
  (1 ≤ m ∧ m < 5/4) ∨ (2 ≤ m ∧ m < 9/4) :=
sorry

end NUMINAMATH_CALUDE_value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l1402_140248


namespace NUMINAMATH_CALUDE_unique_solution_iff_sqrt_three_l1402_140270

/-- The function f(x) = x^2 + a|x| + a^2 - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x| + a^2 - 3

/-- The theorem stating that the equation f(x) = 0 has a unique real solution iff a = √3 -/
theorem unique_solution_iff_sqrt_three (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ↔ a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_sqrt_three_l1402_140270


namespace NUMINAMATH_CALUDE_range_of_a_l1402_140277

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a ≥ 0) ∧
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) →
  a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1402_140277


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1402_140227

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (W : Point)
  (X : Point)
  (Y : Point)
  (Z : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two line segments intersect at right angles -/
def intersect_at_right_angle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if two line segments bisect each other -/
def bisect_each_other (p1 p2 p3 p4 : Point) : Prop :=
  sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral)
  (h1 : is_convex q)
  (h2 : distance q.W q.Y = 15)
  (h3 : distance q.X q.Z = 20)
  (h4 : distance q.W q.X = 18)
  (P : Point)
  (h5 : intersect_at_right_angle q.W q.X q.Y q.Z)
  (h6 : bisect_each_other q.W q.X q.Y q.Z) :
  distance q.W P = 9 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1402_140227


namespace NUMINAMATH_CALUDE_bus_passengers_l1402_140254

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : 
  initial = 28 → got_on = 7 → got_off = 9 → 
  initial + got_on - got_off = 26 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1402_140254


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1402_140221

/-- A bounded real-valued function satisfying a specific functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  (∀ x y, f (x * f y) + y * f x = x * f y + f (x * y))

/-- The theorem stating the only possible forms of f satisfying the functional equation. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ 
  (∀ x, x < 0 → f x = -2*x) ∧ (∀ x, x ≥ 0 → f x = 0) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1402_140221


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l1402_140291

/-- Represents a number in base 8 -/
def BaseEight : Type := ℕ

/-- Converts a base 8 number to its decimal representation -/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Converts a decimal number to its base 8 representation -/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtracts two base 8 numbers -/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

/-- Theorem stating that 46₈ - 27₈ = 17₈ in base 8 -/
theorem base_eight_subtraction :
  base_eight_sub (to_base_eight 38) (to_base_eight 23) = to_base_eight 15 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l1402_140291


namespace NUMINAMATH_CALUDE_sector_area_l1402_140274

/-- Given a circular sector with perimeter 6 cm and central angle 1 radian, its area is 3 cm² -/
theorem sector_area (r : ℝ) (h1 : r + r + r = 6) (h2 : 1 = 1) : r * r / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1402_140274


namespace NUMINAMATH_CALUDE_car_average_speed_l1402_140235

/-- Proves that the average speed of a car traveling 90 km/h for the first hour
    and 60 km/h for the second hour is 75 km/h. -/
theorem car_average_speed :
  let speed1 : ℝ := 90  -- Speed in the first hour (km/h)
  let speed2 : ℝ := 60  -- Speed in the second hour (km/h)
  let time : ℝ := 2     -- Total time (hours)
  let total_distance : ℝ := speed1 + speed2  -- Total distance traveled (km)
  let average_speed : ℝ := total_distance / time  -- Average speed (km/h)
  average_speed = 75
  := by sorry

end NUMINAMATH_CALUDE_car_average_speed_l1402_140235


namespace NUMINAMATH_CALUDE_passenger_arrangement_l1402_140217

def arrange_passengers (n : ℕ) (r : ℕ) : ℕ :=
  -- Define the function to calculate the number of arrangements
  sorry

theorem passenger_arrangement :
  arrange_passengers 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_passenger_arrangement_l1402_140217


namespace NUMINAMATH_CALUDE_sum_of_specific_coefficients_l1402_140237

/-- The coefficient of x^m * y^n in the expansion of (1+x)^4 * (1+y)^6 -/
def P (m n : ℕ) : ℕ := Nat.choose 4 m * Nat.choose 6 n

/-- The sum of coefficients of x^2*y^1 and x^1*y^2 in the expansion of (1+x)^4 * (1+y)^6 is 96 -/
theorem sum_of_specific_coefficients : P 2 1 + P 1 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_coefficients_l1402_140237


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l1402_140269

theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11) 
  (h2 : team_avg_age = 26) 
  (h3 : wicket_keeper_age_diff = 3) 
  (h4 : remaining_avg_age = 23) : 
  team_avg_age - ((team_size * team_avg_age - (team_avg_age + wicket_keeper_age_diff + team_avg_age)) / (team_size - 2)) = 0.33 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l1402_140269


namespace NUMINAMATH_CALUDE_area_of_corner_squares_l1402_140251

/-- The total area of four smaller squares inscribed in the corners of a 2x2 square with an inscribed circle -/
theorem area_of_corner_squares (s : ℝ) : 
  s > 0 ∧ 
  s^2 - 4*s + 2 = 0 ∧ 
  (∃ (r : ℝ), r = 1 ∧ r^2 + r^2 = s^2) →
  4 * s^2 = (48 - 32 * Real.sqrt 2) / 18 :=
by sorry

end NUMINAMATH_CALUDE_area_of_corner_squares_l1402_140251


namespace NUMINAMATH_CALUDE_tunnel_length_l1402_140289

/-- The length of a tunnel given train passage information -/
theorem tunnel_length (train_length : ℝ) (total_time : ℝ) (inside_time : ℝ) : 
  train_length = 300 →
  total_time = 60 →
  inside_time = 30 →
  ∃ (tunnel_length : ℝ) (train_speed : ℝ),
    tunnel_length + train_length = total_time * train_speed ∧
    tunnel_length - train_length = inside_time * train_speed ∧
    tunnel_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l1402_140289


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l1402_140210

-- Define the grid type
def Grid := Matrix (Fin 5) (Fin 5) ℕ

-- Define the sum of a list of numbers
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

-- Define the property that a grid contains numbers 1 to 12
def contains_one_to_twelve (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ∃ i j, g i j = n

-- Define the sum of central columns
def central_columns_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 3 2] = 26 ∧
  list_sum [g 0 3, g 1 3, g 2 3, g 3 3] = 26

-- Define the sum of central rows
def central_rows_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 2 1, g 2 2, g 2 3] = 26 ∧
  list_sum [g 3 0, g 3 1, g 3 2, g 3 3] = 26

-- Define the sum of roses pattern
def roses_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 2 3] = 26

-- Define the sum of shamrocks pattern
def shamrocks_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 3 1, g 4 2, g 1 2] = 26

-- Define the sum of thistle pattern
def thistle_sum (g : Grid) : Prop :=
  list_sum [g 2 2, g 3 2, g 3 3] = 26

-- The main theorem
theorem exists_valid_arrangement :
  ∃ g : Grid,
    contains_one_to_twelve g ∧
    central_columns_sum g ∧
    central_rows_sum g ∧
    roses_sum g ∧
    shamrocks_sum g ∧
    thistle_sum g := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l1402_140210


namespace NUMINAMATH_CALUDE_hyperbola_equation_prove_hyperbola_equation_l1402_140260

/-- The standard equation of a hyperbola with given foci and passing through a specific point. -/
theorem hyperbola_equation (h : ℝ → ℝ → Prop) (f : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  -- Given hyperbola with equation x^2/16 - y^2/9 = 1
  (∀ x y, h x y ↔ x^2/16 - y^2/9 = 1) →
  -- The new hyperbola has the same foci as the given one
  (∃ c : ℝ, c^2 = 25 ∧ f = (c, 0) ∨ f = (-c, 0)) →
  -- The new hyperbola passes through the point P
  (p = (-Real.sqrt 5 / 2, -Real.sqrt 6)) →
  -- The standard equation of the new hyperbola is x^2/1 - y^2/24 = 1
  (∀ x y, (x^2/1 - y^2/24 = 1) ↔ 
    ((x - f.1)^2 + y^2)^(1/2) - ((x + f.1)^2 + y^2)^(1/2) = 2 * Real.sqrt (f.1^2 - 1))

/-- Proof of the hyperbola equation -/
theorem prove_hyperbola_equation : ∃ h f p, hyperbola_equation h f p := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_prove_hyperbola_equation_l1402_140260


namespace NUMINAMATH_CALUDE_james_to_remaining_ratio_l1402_140258

def total_slices : ℕ := 8
def friend_eats : ℕ := 2
def james_eats : ℕ := 3

def slices_after_friend : ℕ := total_slices - friend_eats

theorem james_to_remaining_ratio :
  (james_eats : ℚ) / slices_after_friend = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_james_to_remaining_ratio_l1402_140258


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l1402_140295

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_to_asymptote : ℝ
  asymptote_slope : ℝ
  foci_distance : foci_to_asymptote = 2
  asymptote_parallel : asymptote_slope = 1/2

/-- The theorem stating the specific dimensions of the hyperbola -/
theorem hyperbola_dimensions (h : Hyperbola) : h.a = 4 ∧ h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l1402_140295


namespace NUMINAMATH_CALUDE_dot_product_equation_is_line_l1402_140223

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- Theorem stating that the equation r ⋅ a = m represents a line -/
theorem dot_product_equation_is_line (a : Vec2D) (m : ℝ) :
  ∃ (A B C : ℝ), ∀ (r : Vec2D), dot r a = m ↔ A * r.x + B * r.y + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equation_is_line_l1402_140223


namespace NUMINAMATH_CALUDE_inequality_condition_l1402_140230

theorem inequality_condition (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1402_140230


namespace NUMINAMATH_CALUDE_exactly_one_correct_statement_l1402_140293

/-- Rules of the oblique projection drawing method -/
structure ObliqueProjectionRules where
  parallelism_preserved : Bool
  x_axis_length_preserved : Bool
  y_axis_length_halved : Bool

/-- Statements about intuitive diagrams -/
structure IntuitiveDiagramStatements where
  equal_angles_preserved : Bool
  equal_segments_preserved : Bool
  longest_segment_preserved : Bool
  midpoint_preserved : Bool

/-- Theorem: Exactly one statement is correct given the oblique projection rules -/
theorem exactly_one_correct_statement 
  (rules : ObliqueProjectionRules)
  (statements : IntuitiveDiagramStatements) :
  rules.parallelism_preserved ∧
  rules.x_axis_length_preserved ∧
  rules.y_axis_length_halved →
  (statements.equal_angles_preserved = false) ∧
  (statements.equal_segments_preserved = false) ∧
  (statements.longest_segment_preserved = false) ∧
  (statements.midpoint_preserved = true) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_statement_l1402_140293


namespace NUMINAMATH_CALUDE_remaining_calories_l1402_140285

-- Define the given conditions
def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

-- Define the theorem
theorem remaining_calories :
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calories_l1402_140285


namespace NUMINAMATH_CALUDE_minimum_correct_answers_l1402_140279

def test_score (correct : ℕ) : ℤ :=
  4 * correct - (25 - correct)

theorem minimum_correct_answers : 
  ∀ x : ℕ, x ≤ 25 → test_score x > 70 → x ≥ 19 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_correct_answers_l1402_140279


namespace NUMINAMATH_CALUDE_february_first_is_sunday_l1402_140273

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to check if a given day is Monday -/
def isMonday (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Monday => true
  | _ => false

/-- Theorem: In a leap year, if February has exactly four Mondays, then February 1st must be a Sunday -/
theorem february_first_is_sunday (february : List FebruaryDate) 
  (leap_year : february.length = 29)
  (four_mondays : (february.filter (fun d => isMonday d.dayOfWeek)).length = 4) :
  (february.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Sunday) :=
by
  sorry


end NUMINAMATH_CALUDE_february_first_is_sunday_l1402_140273


namespace NUMINAMATH_CALUDE_ellipse_b_value_l1402_140286

/-- Definition of an ellipse with foci and a point on it -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The vectors PF1 and PF2 are perpendicular -/
def vectors_perpendicular (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area_is_9 (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem -/
theorem ellipse_b_value (e : Ellipse) 
  (h_perp : vectors_perpendicular e) 
  (h_area : triangle_area_is_9 e) : 
  e.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l1402_140286


namespace NUMINAMATH_CALUDE_closest_point_l1402_140261

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2 + 5*t
  | 1 => -3 + 7*t
  | 2 => -3 - 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 7
  | 2 => -2

theorem closest_point :
  let t := 43 / 78
  (v t - a) • direction = 0 ∧
  ∀ s, s ≠ t → ‖v s - a‖ > ‖v t - a‖ :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1402_140261


namespace NUMINAMATH_CALUDE_pizza_slices_ordered_l1402_140216

/-- The number of friends Ron ate pizza with -/
def num_friends : ℕ := 2

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating pizza (Ron + his friends) -/
def total_people : ℕ := num_friends + 1

/-- Theorem: The total number of pizza slices ordered is at least 12 -/
theorem pizza_slices_ordered (num_friends : ℕ) (slices_per_person : ℕ) (total_people : ℕ) :
  num_friends = 2 →
  slices_per_person = 4 →
  total_people = num_friends + 1 →
  total_people * slices_per_person ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_ordered_l1402_140216


namespace NUMINAMATH_CALUDE_jar_water_problem_l1402_140200

theorem jar_water_problem (small_capacity large_capacity water_amount : ℝ) 
  (h1 : water_amount = (1/6) * small_capacity)
  (h2 : water_amount = (1/3) * large_capacity)
  (h3 : small_capacity > 0)
  (h4 : large_capacity > 0) :
  water_amount / large_capacity = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jar_water_problem_l1402_140200


namespace NUMINAMATH_CALUDE_product_first_two_terms_of_specific_sequence_l1402_140257

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_first_two_terms_of_specific_sequence :
  ∃ (a₁ : ℝ),
    arithmetic_sequence a₁ 1 5 = 11 ∧
    arithmetic_sequence a₁ 1 1 * arithmetic_sequence a₁ 1 2 = 56 :=
by sorry

end NUMINAMATH_CALUDE_product_first_two_terms_of_specific_sequence_l1402_140257


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1402_140220

open Real

/-- For the function f(x) = ln x - ax, where a ∈ ℝ and x ∈ (1, +∞),
    the inequality f(x) + a < 0 holds for all x in (1, +∞) if and only if a ≥ 1 -/
theorem function_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x > 1 → (log x - a * x + a < 0)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1402_140220


namespace NUMINAMATH_CALUDE_towel_shrinkage_l1402_140250

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) :
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ new_breadth : ℝ, new_breadth = 0.9 * B ∧ new_length * new_breadth = new_area :=
by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l1402_140250


namespace NUMINAMATH_CALUDE_no_combination_for_3_4_meters_l1402_140272

theorem no_combination_for_3_4_meters :
  ¬ ∃ (a b : ℕ), 0.7 * (a : ℝ) + 0.8 * (b : ℝ) = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_no_combination_for_3_4_meters_l1402_140272


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1402_140202

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 2186 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1402_140202


namespace NUMINAMATH_CALUDE_octagon_handshake_distance_l1402_140255

theorem octagon_handshake_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) :
  let points := n
  let radius := r
  let connections_per_point := n - 3
  let angle_between_points := 2 * Real.pi / n
  let distance_to_third := radius * Real.sqrt (2 - Real.sqrt 2)
  let total_distance := n * connections_per_point * distance_to_third
  total_distance = 1600 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_handshake_distance_l1402_140255


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l1402_140236

def bill : ℚ := 314.16
def payment_per_person : ℚ := 34.91
def total_payment : ℚ := 314.19

theorem restaurant_bill_split :
  ∃ (n : ℕ), n > 0 ∧ 
  (n : ℚ) * payment_per_person ≥ bill ∧
  (n : ℚ) * payment_per_person < bill + 1 ∧
  n * payment_per_person = total_payment ∧
  n = 8 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_split_l1402_140236


namespace NUMINAMATH_CALUDE_sarah_pizza_consumption_l1402_140239

theorem sarah_pizza_consumption (total_slices : ℕ) (eaten_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 20 →
  eaten_slices = 3 →
  shared_slice = 1/3 →
  (eaten_slices : ℚ) / total_slices + shared_slice / total_slices = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pizza_consumption_l1402_140239


namespace NUMINAMATH_CALUDE_yard_area_l1402_140228

/-- The area of a rectangular yard with a square cut-out --/
theorem yard_area (length width cut_side : ℕ) 
  (h1 : length = 20) 
  (h2 : width = 16) 
  (h3 : cut_side = 4) : 
  length * width - cut_side * cut_side = 304 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l1402_140228


namespace NUMINAMATH_CALUDE_sampling_probabilities_equal_l1402_140253

/-- The total number of parts -/
def total_parts : ℕ := 160

/-- The number of first-class products -/
def first_class : ℕ := 48

/-- The number of second-class products -/
def second_class : ℕ := 64

/-- The number of third-class products -/
def third_class : ℕ := 32

/-- The number of substandard products -/
def substandard : ℕ := 16

/-- The sample size -/
def sample_size : ℕ := 20

/-- The probability of selection in simple random sampling -/
def p₁ : ℚ := sample_size / total_parts

/-- The probability of selection in stratified sampling -/
def p₂ : ℚ := sample_size / total_parts

/-- The probability of selection in systematic sampling -/
def p₃ : ℚ := sample_size / total_parts

theorem sampling_probabilities_equal :
  p₁ = p₂ ∧ p₂ = p₃ ∧ p₁ = 1/8 :=
sorry

end NUMINAMATH_CALUDE_sampling_probabilities_equal_l1402_140253


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l1402_140287

theorem line_passes_through_quadrants 
  (a b c : ℝ) 
  (h1 : a * b < 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y + c = 0) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l1402_140287


namespace NUMINAMATH_CALUDE_matching_polygons_l1402_140207

def is_matching (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons :
  ∀ n m : ℕ, n > 2 ∧ m > 2 →
    is_matching n m ↔ ((n = 3 ∧ m = 9) ∨ (n = 4 ∧ m = 6) ∨ (n = 5 ∧ m = 5) ∨ (n = 8 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_matching_polygons_l1402_140207


namespace NUMINAMATH_CALUDE_book_cost_l1402_140211

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one := cost_of_three / 3
  8 * cost_of_one = 120 := by sorry

end NUMINAMATH_CALUDE_book_cost_l1402_140211


namespace NUMINAMATH_CALUDE_simplify_polynomial_product_l1402_140241

theorem simplify_polynomial_product (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^10 + 4*x^9 + x^8) =
  15*x^13 - x^12 + 15*x^11 - 2*x^10 - 5*x^9 - 2*x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_product_l1402_140241


namespace NUMINAMATH_CALUDE_f_100_equals_2_l1402_140271

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else Real.log x

-- Theorem statement
theorem f_100_equals_2 : f 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_100_equals_2_l1402_140271


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1402_140265

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -3) →
  (k = 1) →
  (c = Real.sqrt 41) →
  (a = 4) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1402_140265


namespace NUMINAMATH_CALUDE_sqrt_identity_l1402_140212

theorem sqrt_identity (θ : Real) (h : θ = 40 * π / 180) :
  Real.sqrt (16 - 12 * Real.sin θ) = 4 + Real.sqrt 3 * (1 / Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l1402_140212


namespace NUMINAMATH_CALUDE_equilateral_condition_isosceles_condition_l1402_140259

-- Define a triangle ABC with side lengths a, b, c
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem 1
theorem equilateral_condition (t : Triangle) :
  abs (t.a - t.b) + abs (t.b - t.c) = 0 → is_equilateral t :=
by sorry

-- Theorem 2
theorem isosceles_condition (t : Triangle) :
  (t.a - t.b) * (t.b - t.c) = 0 → is_isosceles t :=
by sorry

end NUMINAMATH_CALUDE_equilateral_condition_isosceles_condition_l1402_140259


namespace NUMINAMATH_CALUDE_textbook_order_cost_l1402_140252

/-- Calculates the total cost of a textbook order with discounts applied --/
def calculate_order_cost (quantities : List Nat) (prices : List Float) (discount_threshold : Nat) (discount_rate : Float) : Float :=
  let total_cost := List.sum (List.zipWith (λ q p => q.toFloat * p) quantities prices)
  let discounted_cost := List.sum (List.zipWith 
    (λ q p => 
      if q ≥ discount_threshold then
        q.toFloat * p * (1 - discount_rate)
      else
        q.toFloat * p
    ) quantities prices)
  discounted_cost

theorem textbook_order_cost : 
  let quantities := [35, 35, 20, 30, 25, 15]
  let prices := [7.50, 10.50, 12.00, 9.50, 11.25, 6.75]
  let discount_threshold := 30
  let discount_rate := 0.1
  calculate_order_cost quantities prices discount_threshold discount_rate = 1446.00 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l1402_140252


namespace NUMINAMATH_CALUDE_range_of_m_l1402_140290

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := -2 * x^2 + b * x + c

-- State the theorem
theorem range_of_m (b c : ℝ) :
  (∀ x, f x b c > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, -1 ≤ x ∧ x ≤ 0 → ∃ m, f x b c + m ≥ 4) →
  ∃ m₀, ∀ m, m ≥ m₀ ↔ (∀ x, -1 ≤ x ∧ x ≤ 0 → f x b c + m ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1402_140290


namespace NUMINAMATH_CALUDE_sin_2012_deg_l1402_140243

theorem sin_2012_deg : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_2012_deg_l1402_140243


namespace NUMINAMATH_CALUDE_fibonacci_rabbit_problem_l1402_140276

/-- Fibonacci sequence representing the number of adult rabbit pairs -/
def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

/-- The number of adult rabbit pairs after n months -/
def adult_rabbits (n : ℕ) : ℕ := fibonacci n

theorem fibonacci_rabbit_problem :
  adult_rabbits 12 = 233 := by sorry

end NUMINAMATH_CALUDE_fibonacci_rabbit_problem_l1402_140276


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_by_11_l1402_140268

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that in any 39 consecutive natural numbers, 
    there is at least one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_by_11 (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 38 ∧ (sum_of_digits k) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_by_11_l1402_140268


namespace NUMINAMATH_CALUDE_game_terminates_l1402_140206

/-- Represents the state of the game at each step -/
structure GameState where
  x : ℕ  -- First number on the blackboard
  y : ℕ  -- Second number on the blackboard
  r : ℕ  -- Lower bound of the possible range for the unknown number
  s : ℕ  -- Upper bound of the possible range for the unknown number

/-- The game terminates when the range becomes invalid (r > s) -/
def is_terminal (state : GameState) : Prop :=
  state.r > state.s

/-- The next state of the game after a question is asked -/
def next_state (state : GameState) : GameState :=
  { x := state.x
  , y := state.y
  , r := state.y - state.s
  , s := state.x - state.r }

/-- The main theorem: the game terminates in a finite number of steps -/
theorem game_terminates (a b : ℕ) (h : a > 0 ∧ b > 0) :
  ∃ n : ℕ, is_terminal (n.iterate next_state (GameState.mk (min (a + b) (a + b + 1)) (max (a + b) (a + b + 1)) 0 (a + b))) :=
sorry

end NUMINAMATH_CALUDE_game_terminates_l1402_140206


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l1402_140222

theorem x_plus_y_equals_negative_eight 
  (h1 : |x| + x - y = 16) 
  (h2 : x - |y| + y = -8) : 
  x + y = -8 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l1402_140222


namespace NUMINAMATH_CALUDE_doctors_visit_insurance_coverage_percentage_l1402_140294

def doctors_visit_cost : ℝ := 300
def cats_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60
def total_paid_after_insurance : ℝ := 135

theorem doctors_visit_insurance_coverage_percentage :
  let total_cost := doctors_visit_cost + cats_visit_cost
  let total_insurance_coverage := total_cost - total_paid_after_insurance
  let doctors_visit_coverage := total_insurance_coverage - pet_insurance_coverage
  doctors_visit_coverage / doctors_visit_cost = 0.75 := by sorry

end NUMINAMATH_CALUDE_doctors_visit_insurance_coverage_percentage_l1402_140294


namespace NUMINAMATH_CALUDE_magazine_revenue_calculation_l1402_140224

/-- Calculates the revenue from magazine sales given the total sales, newspaper sales, prices, and total revenue -/
theorem magazine_revenue_calculation 
  (total_items : ℕ) 
  (newspaper_count : ℕ) 
  (newspaper_price : ℚ) 
  (magazine_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_items = 425)
  (h2 : newspaper_count = 275)
  (h3 : newspaper_price = 5/2)
  (h4 : magazine_price = 19/4)
  (h5 : total_revenue = 123025/100)
  (h6 : newspaper_count ≤ total_items) :
  (total_items - newspaper_count) * magazine_price = 54275/100 := by
  sorry

end NUMINAMATH_CALUDE_magazine_revenue_calculation_l1402_140224


namespace NUMINAMATH_CALUDE_max_ab_and_min_fraction_l1402_140240

theorem max_ab_and_min_fraction (a b x y : ℝ) : 
  a > 0 → b > 0 → 4 * a + b = 1 → 
  x > 0 → y > 0 → x + y = 1 → 
  (∀ a' b', a' > 0 → b' > 0 → 4 * a' + b' = 1 → a * b ≥ a' * b') ∧ 
  (∀ x' y', x' > 0 → y' > 0 → x' + y' = 1 → 4 / x + 9 / y ≤ 4 / x' + 9 / y') ∧
  a * b = 1 / 16 ∧ 
  4 / x + 9 / y = 25 := by
sorry

end NUMINAMATH_CALUDE_max_ab_and_min_fraction_l1402_140240


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1402_140280

-- Define the base 7 number as a list of digits
def base7_number : List Nat := [2, 5, 3, 4]

-- Define the conversion function from base 7 to base 10
def base7_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem base7_to_base10_conversion :
  base7_to_base10 base7_number = 956 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1402_140280


namespace NUMINAMATH_CALUDE_c_to_a_ratio_l1402_140231

/-- Represents the share of money for each person in Rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions of the problem -/
def ProblemConditions (s : Shares) : Prop :=
  s.c = 56 ∧ 
  s.a + s.b + s.c = 287 ∧ 
  s.b = 0.65 * s.a

/-- Theorem stating the ratio of C's share to A's share in paisa -/
theorem c_to_a_ratio (s : Shares) 
  (h : ProblemConditions s) : (s.c * 100) / (s.a * 100) = 0.4 := by
  sorry

#check c_to_a_ratio

end NUMINAMATH_CALUDE_c_to_a_ratio_l1402_140231


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1402_140296

theorem absolute_value_inequality (y : ℝ) : 
  |((8 - 2*y) / 4)| < 3 ↔ -2 < y ∧ y < 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1402_140296


namespace NUMINAMATH_CALUDE_john_classes_l1402_140266

theorem john_classes (packs_per_student : ℕ) (students_per_class : ℕ) (total_packs : ℕ) 
  (h1 : packs_per_student = 2)
  (h2 : students_per_class = 30)
  (h3 : total_packs = 360) :
  total_packs / (packs_per_student * students_per_class) = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_classes_l1402_140266


namespace NUMINAMATH_CALUDE_competition_score_l1402_140233

theorem competition_score (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (total_score : ℤ) (score_difference : ℤ) :
  total_questions = 10 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 58 →
  score_difference = 14 →
  ∃ (a_correct : ℕ) (b_correct : ℕ),
    a_correct + b_correct ≤ total_questions ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points +
    b_correct * correct_points + (total_questions - b_correct) * incorrect_points = total_score ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points -
    (b_correct * correct_points + (total_questions - b_correct) * incorrect_points) = score_difference ∧
    a_correct = 8 :=
by sorry

end NUMINAMATH_CALUDE_competition_score_l1402_140233


namespace NUMINAMATH_CALUDE_speed_conversion_l1402_140284

/-- Conversion factor from kilometers per hour to meters per second -/
def kmph_to_ms : ℝ := 0.277778

/-- Given speed in kilometers per hour -/
def given_speed : ℝ := 252

/-- Equivalent speed in meters per second -/
def equivalent_speed : ℝ := 70

/-- Theorem stating that the given speed in kmph is equal to the equivalent speed in m/s -/
theorem speed_conversion :
  given_speed * kmph_to_ms = equivalent_speed := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l1402_140284


namespace NUMINAMATH_CALUDE_facial_tissue_price_decrease_l1402_140218

/-- The percent decrease in price per box of facial tissue during a sale -/
theorem facial_tissue_price_decrease (original_price sale_price : ℚ) : 
  original_price = 5 / 4 →
  sale_price = 4 / 5 →
  abs ((original_price - sale_price) / original_price - 9 / 25) < 1 / 100 := by
  sorry

#eval (5/4 : ℚ) -- Original price per box
#eval (4/5 : ℚ) -- Sale price per box
#eval ((5/4 - 4/5) / (5/4) : ℚ) -- Actual percent decrease

end NUMINAMATH_CALUDE_facial_tissue_price_decrease_l1402_140218


namespace NUMINAMATH_CALUDE_hindi_speakers_count_l1402_140205

/-- Represents the number of children who can speak a given language or combination of languages -/
structure LanguageCount where
  total : ℕ
  onlyEnglish : ℕ
  onlyHindi : ℕ
  onlySpanish : ℕ
  englishAndHindi : ℕ
  englishAndSpanish : ℕ
  hindiAndSpanish : ℕ
  allThree : ℕ

/-- Calculates the number of children who can speak Hindi -/
def hindiSpeakers (c : LanguageCount) : ℕ :=
  c.onlyHindi + c.englishAndHindi + c.hindiAndSpanish + c.allThree

/-- Theorem stating that the number of Hindi speakers is 45 given the conditions -/
theorem hindi_speakers_count (c : LanguageCount)
  (h_total : c.total = 90)
  (h_onlyEnglish : c.onlyEnglish = 90 * 25 / 100)
  (h_onlyHindi : c.onlyHindi = 90 * 15 / 100)
  (h_onlySpanish : c.onlySpanish = 90 * 10 / 100)
  (h_englishAndHindi : c.englishAndHindi = 90 * 20 / 100)
  (h_englishAndSpanish : c.englishAndSpanish = 90 * 15 / 100)
  (h_hindiAndSpanish : c.hindiAndSpanish = 90 * 10 / 100)
  (h_allThree : c.allThree = 90 * 5 / 100) :
  hindiSpeakers c = 45 := by
  sorry


end NUMINAMATH_CALUDE_hindi_speakers_count_l1402_140205


namespace NUMINAMATH_CALUDE_sasha_questions_per_hour_l1402_140264

theorem sasha_questions_per_hour 
  (initial_questions : ℕ)
  (work_hours : ℕ)
  (remaining_questions : ℕ)
  (h1 : initial_questions = 60)
  (h2 : work_hours = 2)
  (h3 : remaining_questions = 30) :
  (initial_questions - remaining_questions) / work_hours = 15 := by
sorry

end NUMINAMATH_CALUDE_sasha_questions_per_hour_l1402_140264


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_S_l1402_140238

def S : Set ℕ := {n : ℕ | ∃ (d₁ d₂ d₃ : ℕ), d₁ > d₂ ∧ d₂ > d₃ ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₁ ≠ n ∧ d₂ ≠ n ∧ d₃ ≠ n ∧ d₁ + d₂ + d₃ > n}

theorem largest_common_divisor_of_S : ∀ n ∈ S, 6 ∣ n ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_S_l1402_140238


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l1402_140208

/-- The number of beaded necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used by Kylie -/
def total_beads : ℕ := 325

/-- Theorem stating that 10 beads are needed to make one beaded bracelet -/
theorem beads_per_bracelet : 
  (total_beads - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / wednesday_bracelets = 10 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l1402_140208


namespace NUMINAMATH_CALUDE_coupon_discount_proof_l1402_140229

/-- Calculates the discount given the costs and final amount paid -/
def calculate_discount (magazine_cost pencil_cost final_amount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - final_amount

theorem coupon_discount_proof :
  let magazine_cost : ℚ := 85/100
  let pencil_cost : ℚ := 1/2
  let final_amount : ℚ := 1
  calculate_discount magazine_cost pencil_cost final_amount = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_coupon_discount_proof_l1402_140229


namespace NUMINAMATH_CALUDE_stop_after_fourth_draw_l1402_140213

/-- The probability of stopping after the fourth draw in a box with 5 black and 4 white balls -/
theorem stop_after_fourth_draw (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = black_balls + white_balls →
  black_balls = 5 →
  white_balls = 4 →
  (black_balls / total_balls : ℚ)^3 * (white_balls / total_balls : ℚ) = (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stop_after_fourth_draw_l1402_140213


namespace NUMINAMATH_CALUDE_triangle_midline_lengths_l1402_140275

/-- Given a triangle with side lengths a, b, and c, the lengths of its midlines are half the lengths of the opposite sides. -/
theorem triangle_midline_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (s_a s_b s_c : ℝ),
    s_a = (1/2) * b ∧
    s_b = (1/2) * c ∧
    s_c = (1/2) * a :=
by sorry

end NUMINAMATH_CALUDE_triangle_midline_lengths_l1402_140275


namespace NUMINAMATH_CALUDE_add_decimals_l1402_140263

theorem add_decimals : (7.56 : ℝ) + (4.29 : ℝ) = 11.85 := by sorry

end NUMINAMATH_CALUDE_add_decimals_l1402_140263


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l1402_140203

theorem greatest_integer_inequality : ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l1402_140203


namespace NUMINAMATH_CALUDE_simplify_expression_l1402_140281

theorem simplify_expression (x : ℚ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1402_140281


namespace NUMINAMATH_CALUDE_pam_current_age_l1402_140288

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the current state -/
structure CurrentState where
  pam_age : Age
  rena_age : Age

/-- Represents the future state after 10 years -/
structure FutureState where
  pam_age : Age
  rena_age : Age

/-- The conditions of the problem -/
def problem_conditions (current : CurrentState) (future : FutureState) : Prop :=
  (current.pam_age.years * 2 = current.rena_age.years) ∧
  (future.rena_age.years = future.pam_age.years + 5) ∧
  (future.pam_age.years = current.pam_age.years + 10) ∧
  (future.rena_age.years = current.rena_age.years + 10)

/-- The theorem to prove -/
theorem pam_current_age
  (current : CurrentState)
  (future : FutureState)
  (h : problem_conditions current future) :
  current.pam_age.years = 5 := by
  sorry

end NUMINAMATH_CALUDE_pam_current_age_l1402_140288


namespace NUMINAMATH_CALUDE_abs_plus_one_minimum_l1402_140242

theorem abs_plus_one_minimum :
  ∃ (min : ℝ) (x₀ : ℝ), (∀ x : ℝ, min ≤ |x| + 1) ∧ (min = |x₀| + 1) ∧ (min = 1 ∧ x₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_plus_one_minimum_l1402_140242


namespace NUMINAMATH_CALUDE_parallel_condition_l1402_140225

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) : 
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧ 
  (∃ α β a, line_plane_parallel a β ∧ ¬plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1402_140225


namespace NUMINAMATH_CALUDE_functions_satisfy_equation_l1402_140201

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := a * x^2 + b * x + c
def h (x : ℝ) : ℝ := a * x + b

theorem functions_satisfy_equation :
  ∀ (x y : ℝ), f a b c x - g a b c y = (x - y) * h a b (x + y) := by sorry

end NUMINAMATH_CALUDE_functions_satisfy_equation_l1402_140201


namespace NUMINAMATH_CALUDE_tan_3_expression_zero_l1402_140298

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_expression_zero_l1402_140298


namespace NUMINAMATH_CALUDE_linear_system_solution_l1402_140297

theorem linear_system_solution (x y z : ℚ) : 
  x + 2 * y = 12 ∧ 
  y + 3 * z = 15 ∧ 
  3 * x - z = 6 → 
  x = 54 / 17 ∧ y = 75 / 17 ∧ z = 60 / 17 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1402_140297


namespace NUMINAMATH_CALUDE_constant_dot_product_l1402_140219

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + 3*y^2/4 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a tangent line to circle O
def tangent_line (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m ∧ 1 + k^2 = m^2

-- Define the intersection points of the tangent line and ellipse C
def intersection_points (k m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
  tangent_line k m x₁ y₁ ∧ tangent_line k m x₂ y₂

-- Theorem statement
theorem constant_dot_product :
  ∀ (k m x₁ y₁ x₂ y₂ : ℝ),
  intersection_points k m x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l1402_140219


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1402_140215

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x ≥ Real.sin x)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x < Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1402_140215


namespace NUMINAMATH_CALUDE_rachel_steps_up_correct_l1402_140299

/-- The number of steps Rachel climbed going up the Eiffel Tower -/
def steps_up : ℕ := 567

/-- The number of steps Rachel climbed going down the Eiffel Tower -/
def steps_down : ℕ := 325

/-- The total number of steps Rachel climbed -/
def total_steps : ℕ := 892

/-- Theorem: The number of steps Rachel climbed going up is correct -/
theorem rachel_steps_up_correct : steps_up + steps_down = total_steps := by
  sorry

end NUMINAMATH_CALUDE_rachel_steps_up_correct_l1402_140299


namespace NUMINAMATH_CALUDE_exists_special_function_l1402_140245

theorem exists_special_function :
  ∃ f : ℕ+ → ℕ+,
    (∀ m n : ℕ+, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l1402_140245


namespace NUMINAMATH_CALUDE_min_value_f_min_value_expression_l1402_140292

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f x ≥ m :=
sorry

-- Theorem for the minimum value of 1/(a²+1) + 4/(b²+1)
theorem min_value_expression (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ min_val : ℝ, min_val = 9/4 ∧
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_expression_l1402_140292


namespace NUMINAMATH_CALUDE_martha_butterflies_l1402_140267

/-- The number of black butterflies in Martha's collection --/
def black_butterflies (total blue yellow : ℕ) : ℕ :=
  total - blue - yellow

/-- Theorem stating the number of black butterflies in Martha's collection --/
theorem martha_butterflies :
  ∀ (total blue yellow : ℕ),
    total = 11 →
    blue = 4 →
    blue = 2 * yellow →
    black_butterflies total blue yellow = 5 := by
  sorry

end NUMINAMATH_CALUDE_martha_butterflies_l1402_140267
