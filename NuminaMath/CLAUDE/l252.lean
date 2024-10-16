import Mathlib

namespace NUMINAMATH_CALUDE_item_list_price_l252_25264

/-- The list price of an item -/
def list_price : ℝ := 33

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Charles's selling price -/
def charles_price (x : ℝ) : ℝ := x - 18

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Charles's commission rate -/
def charles_rate : ℝ := 0.18

theorem item_list_price :
  alice_rate * alice_price list_price = charles_rate * charles_price list_price :=
by sorry

end NUMINAMATH_CALUDE_item_list_price_l252_25264


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l252_25289

theorem circle_center_polar_coordinates :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 1}
  let center := (1, 1)
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  (center ∈ circle) ∧
  (r * Real.cos θ = center.1 - 0) ∧
  (r * Real.sin θ = center.2 - 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l252_25289


namespace NUMINAMATH_CALUDE_gecko_hatched_eggs_l252_25228

/-- Theorem: Number of hatched eggs for a gecko --/
theorem gecko_hatched_eggs (total_eggs : ℕ) (infertile_rate : ℚ) (calcification_rate : ℚ)
  (h_total : total_eggs = 30)
  (h_infertile : infertile_rate = 1/5)
  (h_calcification : calcification_rate = 1/3) :
  (total_eggs : ℚ) * (1 - infertile_rate) * (1 - calcification_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gecko_hatched_eggs_l252_25228


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l252_25276

/-- Calculates the cost per trip to an amusement park given the number of sons,
    cost per pass, and number of trips by each son. -/
def cost_per_trip (num_sons : ℕ) (cost_per_pass : ℚ) (trips_oldest : ℕ) (trips_youngest : ℕ) : ℚ :=
  (num_sons * cost_per_pass) / (trips_oldest + trips_youngest)

/-- Theorem stating that for the given inputs, the cost per trip is $4.00 -/
theorem amusement_park_cost_per_trip :
  cost_per_trip 2 100 35 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l252_25276


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l252_25292

-- Define the circle radius
def circle_radius : ℝ := 6

-- Define the relationship between rectangle and circle areas
def area_ratio : ℝ := 3

-- Theorem statement
theorem rectangle_longer_side (circle_radius : ℝ) (area_ratio : ℝ) :
  circle_radius = 6 →
  area_ratio = 3 →
  let circle_area := π * circle_radius^2
  let rectangle_area := area_ratio * circle_area
  let shorter_side := 2 * circle_radius
  rectangle_area / shorter_side = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l252_25292


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l252_25298

theorem binary_arithmetic_equality : 
  (0b10110 : Nat) + 0b1101 - 0b11100 + 0b11101 + 0b101 = 0b101101 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l252_25298


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l252_25257

def aquarium (num_stingrays : ℕ) : ℕ → Prop :=
  fun total_fish =>
    ∃ num_sharks : ℕ,
      num_sharks = 2 * num_stingrays ∧
      total_fish = num_sharks + num_stingrays

theorem aquarium_fish_count : aquarium 28 84 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l252_25257


namespace NUMINAMATH_CALUDE_total_donuts_three_days_l252_25231

def monday_donuts : ℕ := 14

def tuesday_donuts : ℕ := monday_donuts / 2

def wednesday_donuts : ℕ := 4 * monday_donuts

theorem total_donuts_three_days : 
  monday_donuts + tuesday_donuts + wednesday_donuts = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_donuts_three_days_l252_25231


namespace NUMINAMATH_CALUDE_alcohol_dilution_l252_25208

/-- Proves that mixing 50 ml of 30% alcohol after-shave lotion with 30 ml of pure water
    results in a solution with 18.75% alcohol content. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (water_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_percentage = 30)
  (h3 : water_volume = 30) :
  let alcohol_volume : ℝ := initial_volume * (initial_percentage / 100)
  let total_volume : ℝ := initial_volume + water_volume
  let new_percentage : ℝ := (alcohol_volume / total_volume) * 100
  new_percentage = 18.75 := by
sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l252_25208


namespace NUMINAMATH_CALUDE_investment_quoted_price_l252_25277

/-- Calculates the quoted price of shares given investment details -/
def quoted_price (total_investment : ℚ) (nominal_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * nominal_value
  let number_of_shares := annual_income / dividend_per_share
  total_investment / number_of_shares

/-- Theorem stating that given the investment details, the quoted price is 9.5 -/
theorem investment_quoted_price :
  quoted_price 4940 10 14 728 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_investment_quoted_price_l252_25277


namespace NUMINAMATH_CALUDE_child_sold_seven_apples_l252_25265

/-- Represents the number of apples sold by a child given the initial conditions and final count -/
def apples_sold (num_children : ℕ) (apples_per_child : ℕ) (eating_children : ℕ) (apples_eaten_each : ℕ) (apples_left : ℕ) : ℕ :=
  num_children * apples_per_child - eating_children * apples_eaten_each - apples_left

/-- Theorem stating that given the conditions in the problem, the child sold 7 apples -/
theorem child_sold_seven_apples :
  apples_sold 5 15 2 4 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_child_sold_seven_apples_l252_25265


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l252_25296

theorem largest_triangle_perimeter : 
  ∀ y : ℤ, 
  (y > 0) → 
  (7 + 9 > y) → 
  (7 + y > 9) → 
  (9 + y > 7) → 
  (∀ z : ℤ, (z > 0) → (7 + 9 > z) → (7 + z > 9) → (9 + z > 7) → (7 + 9 + y ≥ 7 + 9 + z)) →
  (7 + 9 + y = 31) :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l252_25296


namespace NUMINAMATH_CALUDE_mode_of_shoe_sizes_l252_25216

def shoe_sizes : List ℝ := [24, 24.5, 25, 25.5, 26]
def sales : List ℕ := [2, 5, 3, 6, 4]

def mode (sizes : List ℝ) (counts : List ℕ) : ℝ :=
  let pairs := List.zip sizes counts
  let max_count := pairs.map Prod.snd |>.maximum?
  match max_count with
  | none => 0  -- Default value if the list is empty
  | some mc => (pairs.filter (fun p => p.2 = mc)).head!.1

theorem mode_of_shoe_sizes :
  mode shoe_sizes sales = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_shoe_sizes_l252_25216


namespace NUMINAMATH_CALUDE_max_utility_problem_l252_25223

theorem max_utility_problem (s : ℝ) : 
  s ≥ 0 ∧ s * (10 - s) = (3 - s) * (s + 4) → s = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_utility_problem_l252_25223


namespace NUMINAMATH_CALUDE_soccer_substitutions_modulo_l252_25255

def num_players : ℕ := 22
def starting_players : ℕ := 11
def max_substitutions : ℕ := 4

def substitution_ways : ℕ → ℕ
| 0 => 1
| 1 => starting_players * starting_players
| n+1 => substitution_ways n * (starting_players - n) * (starting_players - n)

def total_substitution_ways : ℕ := 
  (List.range (max_substitutions + 1)).map substitution_ways |> List.sum

theorem soccer_substitutions_modulo :
  total_substitution_ways % 1000 = 722 := by sorry

end NUMINAMATH_CALUDE_soccer_substitutions_modulo_l252_25255


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l252_25219

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l252_25219


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_on_circle_l252_25212

-- Define a homogeneous polynomial
def IsHomogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (c x y : ℝ), P (c * x) (c * y) = c^n * P x y

-- Define the theorem
theorem homogeneous_polynomial_on_circle (P : ℝ → ℝ → ℝ) (n : ℕ) :
  IsHomogeneous P n →
  (∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ ∀ x y : ℝ, P x y = (x^2 + y^2)^k :=
by sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_on_circle_l252_25212


namespace NUMINAMATH_CALUDE_root_product_equals_27_l252_25241

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l252_25241


namespace NUMINAMATH_CALUDE_coin_problem_l252_25269

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →  -- Total number of coins
  2 * x + 5 * y = 125 →  -- Total amount of money
  y = 15  -- Number of 5-dollar coins
  := by sorry

end NUMINAMATH_CALUDE_coin_problem_l252_25269


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l252_25244

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (bridge_length : ℝ) 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 200)
  (h2 : train_length = 100)
  (h3 : train_speed = 5) : 
  (bridge_length + train_length) / train_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l252_25244


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l252_25271

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (team_average : ℝ),
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℝ) * team_average = 
    (team_size - 2 : ℝ) * (team_average - 1) + 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) →
  team_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l252_25271


namespace NUMINAMATH_CALUDE_single_colony_days_l252_25203

/-- Represents the growth of bacteria colonies -/
def BacteriaGrowth : Type :=
  { n : ℕ // n > 0 }

/-- The number of days it takes for two colonies to reach the habitat limit -/
def two_colony_days : BacteriaGrowth := ⟨15, by norm_num⟩

/-- Calculates the size of a colony after n days, given its initial size -/
def colony_size (initial : ℕ) (days : ℕ) : ℕ :=
  initial * 2^days

/-- Theorem stating that a single colony takes 16 days to reach the habitat limit -/
theorem single_colony_days :
  ∃ (limit : ℕ), limit > 0 ∧
    colony_size 1 (two_colony_days.val + 1) = limit ∧
    colony_size 2 two_colony_days.val = limit := by
  sorry

end NUMINAMATH_CALUDE_single_colony_days_l252_25203


namespace NUMINAMATH_CALUDE_complex_modulus_one_l252_25235

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l252_25235


namespace NUMINAMATH_CALUDE_inequality_may_not_hold_l252_25260

theorem inequality_may_not_hold (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, a / c ≤ b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_may_not_hold_l252_25260


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l252_25222

theorem prime_triplet_equation :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p * (p - 7) + q * (q - 7) = r * (r - 7) →
  ((p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 7 ∧ q = 5 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 2) ∨
   (p = 7 ∧ q = 5 ∧ r = 2) ∨
   (p = 7 ∧ q = 3 ∧ r = 3) ∨
   (p = 3 ∧ q = 7 ∧ r = 3) ∨
   (Prime p ∧ q = 7 ∧ r = p) ∨
   (p = 7 ∧ Prime q ∧ r = q)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l252_25222


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l252_25240

/-- The value depletion rate of a machine given its initial value and value after 2 years -/
theorem machine_value_depletion_rate 
  (initial_value : ℝ) 
  (value_after_two_years : ℝ) 
  (h1 : initial_value = 700) 
  (h2 : value_after_two_years = 567) : 
  ∃ (r : ℝ), 
    value_after_two_years = initial_value * (1 - r)^2 ∧ 
    r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l252_25240


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1560_l252_25236

theorem sum_of_largest_and_smallest_prime_factors_of_1560 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1560 ∧ largest ∣ 1560 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≥ smallest) ∧
    smallest + largest = 15 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1560_l252_25236


namespace NUMINAMATH_CALUDE_decreasing_function_characterization_l252_25288

open Set

/-- A function f is decreasing on an open interval (a, b) if for all x₁, x₂ in (a, b),
    x₁ < x₂ implies f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_characterization
  (f : ℝ → ℝ) (a b : ℝ) (h : a < b)
  (h_domain : ∀ x, x ∈ Ioo a b → f x ∈ range f)
  (h_inequality : ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b →
    (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  DecreasingOn f a b := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_characterization_l252_25288


namespace NUMINAMATH_CALUDE_eight_lines_theorem_l252_25220

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Representation of a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- The given set of lines -/
def given_lines : LineSet :=
  { num_lines := 8
  , no_parallel := true
  , no_concurrent := true }

theorem eight_lines_theorem (lines : LineSet) (h1 : lines.num_lines = 8) 
    (h2 : lines.no_parallel) (h3 : lines.no_concurrent) : 
  num_regions lines.num_lines = 37 := by
  sorry

#eval num_regions 8

end NUMINAMATH_CALUDE_eight_lines_theorem_l252_25220


namespace NUMINAMATH_CALUDE_sequence_properties_l252_25253

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℕ := 2^(n+1) - 2

/-- The n-th term of sequence a_n -/
def a (n : ℕ) : ℕ := 2^n

/-- The n-th term of sequence b_n -/
def b (n : ℕ) : ℕ := n * a n

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℕ := (n-1) * 2^(n+1) + 2

theorem sequence_properties (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, a k = 2^k) ∧
  T n = (n-1) * 2^(n+1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l252_25253


namespace NUMINAMATH_CALUDE_certain_number_problem_l252_25238

theorem certain_number_problem : ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l252_25238


namespace NUMINAMATH_CALUDE_distance_traveled_downstream_l252_25221

/-- Calculate the distance traveled downstream by a boat -/
theorem distance_traveled_downstream 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 5)
  (h3 : time_minutes = 24) :
  let effective_speed := boat_speed + current_speed
  let time_hours := time_minutes / 60
  effective_speed * time_hours = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_traveled_downstream_l252_25221


namespace NUMINAMATH_CALUDE_kids_on_soccer_field_l252_25256

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who joined the soccer field -/
def joined_kids : ℕ := 22

/-- The total number of kids on the soccer field after more kids joined -/
def total_kids : ℕ := initial_kids + joined_kids

theorem kids_on_soccer_field : total_kids = 36 := by
  sorry

end NUMINAMATH_CALUDE_kids_on_soccer_field_l252_25256


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_simplify_third_expression_l252_25213

-- First expression
theorem simplify_first_expression (a b : ℝ) (h : (a - b)^2 + a*b ≠ 0) :
  (a^3 + b^3) / ((a - b)^2 + a*b) = a + b := by sorry

-- Second expression
theorem simplify_second_expression (x a : ℝ) (h : x^2 - 4*a^2 ≠ 0) :
  (x^2 - 4*a*x + 4*a^2) / (x^2 - 4*a^2) = (x - 2*a) / (x + 2*a) := by sorry

-- Third expression
theorem simplify_third_expression (x y : ℝ) (h : x*y - 2*x ≠ 0) :
  (x*y - 2*x - 3*y + 6) / (x*y - 2*x) = (x - 3) / x := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_simplify_third_expression_l252_25213


namespace NUMINAMATH_CALUDE_all_lines_have_inclination_angle_not_necessarily_slope_l252_25297

-- Define what a line is (this is a simplified representation)
structure Line where
  -- You might add more properties here in a real implementation
  dummy : Unit

-- Define the concept of an inclination angle
def has_inclination_angle (l : Line) : Prop := sorry

-- Define the concept of a slope
def has_slope (l : Line) : Prop := sorry

-- The theorem to prove
theorem all_lines_have_inclination_angle_not_necessarily_slope :
  (∀ l : Line, has_inclination_angle l) ∧
  (∃ l : Line, ¬ has_slope l) := by sorry

end NUMINAMATH_CALUDE_all_lines_have_inclination_angle_not_necessarily_slope_l252_25297


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l252_25214

/-- The perimeter of a semi-circle with radius 14 cm is 14π + 28 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let diameter : ℝ := 2 * r
  let half_circumference : ℝ := π * r
  let perimeter : ℝ := half_circumference + diameter
  perimeter = 14 * π + 28 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l252_25214


namespace NUMINAMATH_CALUDE_no_valid_numbers_with_19x_relation_l252_25258

/-- Checks if a natural number is composed only of digits 2, 3, 4, and 9 -/
def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [2, 3, 4, 9]

/-- The main theorem stating the impossibility of finding two numbers
    with the given properties -/
theorem no_valid_numbers_with_19x_relation :
  ¬∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ b = 19 * a :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_with_19x_relation_l252_25258


namespace NUMINAMATH_CALUDE_march_total_distance_l252_25200

/-- Represents Emberly's walking distance for a single day -/
structure DailyWalk where
  day : Nat
  distance : Float

/-- Emberly's walking pattern for March -/
def marchWalks : List DailyWalk := [
  ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨6, 0⟩, ⟨7, 0⟩,
  ⟨8, 5⟩, ⟨9, 2.5⟩, ⟨10, 5⟩, ⟨11, 5⟩, ⟨12, 2.5⟩, ⟨13, 2.5⟩, ⟨14, 0⟩,
  ⟨15, 6⟩, ⟨16, 6⟩, ⟨17, 0⟩, ⟨18, 0⟩, ⟨19, 0⟩, ⟨20, 4⟩, ⟨21, 4⟩, ⟨22, 3.5⟩,
  ⟨23, 4.5⟩, ⟨24, 0⟩, ⟨25, 4.5⟩, ⟨26, 0⟩, ⟨27, 4.5⟩, ⟨28, 0⟩, ⟨29, 4.5⟩, ⟨30, 0⟩, ⟨31, 0⟩
]

/-- Calculate the total distance walked in March -/
def totalDistance (walks : List DailyWalk) : Float :=
  walks.foldl (fun acc walk => acc + walk.distance) 0

/-- Theorem: The total distance Emberly walked in March is 82 miles -/
theorem march_total_distance : totalDistance marchWalks = 82 := by
  sorry


end NUMINAMATH_CALUDE_march_total_distance_l252_25200


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l252_25254

theorem polynomial_divisibility (m n p : ℕ) : 
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m) + X^(n+1) + X^(p+2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l252_25254


namespace NUMINAMATH_CALUDE_discount_rate_example_l252_25218

/-- Given a bag with a marked price and a selling price, calculate the discount rate. -/
def discount_rate (marked_price selling_price : ℚ) : ℚ :=
  (marked_price - selling_price) / marked_price * 100

/-- Theorem: The discount rate for a bag marked at $80 and sold for $68 is 15%. -/
theorem discount_rate_example : discount_rate 80 68 = 15 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_example_l252_25218


namespace NUMINAMATH_CALUDE_power_of_two_l252_25283

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l252_25283


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l252_25201

theorem adult_tickets_sold (children_tickets : ℕ) (children_price : ℕ) (adult_price : ℕ) (total_earnings : ℕ) : 
  children_tickets = 210 →
  children_price = 25 →
  adult_price = 50 →
  total_earnings = 5950 →
  (total_earnings - children_tickets * children_price) / adult_price = 14 :=
by sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l252_25201


namespace NUMINAMATH_CALUDE_integer_solutions_inequalities_l252_25237

theorem integer_solutions_inequalities (x : ℤ) : 
  ((x - 2) / 2 ≤ -x / 2 + 2 ∧ 4 - 7*x < -3) ↔ (x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_inequalities_l252_25237


namespace NUMINAMATH_CALUDE_winning_bet_amount_l252_25280

def initial_amount : ℕ := 400

def bet_multiplier : ℕ := 2

theorem winning_bet_amount (initial : ℕ) (multiplier : ℕ) :
  initial = initial_amount →
  multiplier = bet_multiplier →
  initial + (multiplier * initial) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_winning_bet_amount_l252_25280


namespace NUMINAMATH_CALUDE_range_of_a_l252_25268

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + a + 2 = 0 has solutions -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

/-- If propositions p and q are both true, then a ∈ (-∞, -1] -/
theorem range_of_a (a : ℝ) (h_p : prop_p a) (h_q : prop_q a) : a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l252_25268


namespace NUMINAMATH_CALUDE_equal_roots_iff_n_eq_neg_one_l252_25239

/-- The equation has equal roots if and only if n = -1 -/
theorem equal_roots_iff_n_eq_neg_one (n : ℝ) : 
  (∃! x : ℝ, x ≠ 2 ∧ (x * (x - 2) - (n + 2)) / ((x - 2) * (n - 2)) = x / n) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_iff_n_eq_neg_one_l252_25239


namespace NUMINAMATH_CALUDE_unique_solution_l252_25266

def is_valid_number (α β : ℕ) : Prop :=
  0 ≤ α ∧ α ≤ 9 ∧ 0 ≤ β ∧ β ≤ 9

def number_value (α β : ℕ) : ℕ :=
  62000000 + α * 10000 + β * 1000 + 427

theorem unique_solution (α β : ℕ) :
  is_valid_number α β →
  (number_value α β) % 99 = 0 →
  α = 2 ∧ β = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l252_25266


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l252_25286

noncomputable def triangle_abc (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a - b) * (Real.sin A + Real.sin B) = c * (Real.sin A - Real.sin C) ∧
  b = 2 ∧
  a = 2 * Real.sqrt 6 / 3

theorem triangle_abc_properties {a b c A B C : ℝ} 
  (h : triangle_abc a b c A B C) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) ∧ 
  (∃ (area : ℝ), area = Real.sqrt 3 / 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l252_25286


namespace NUMINAMATH_CALUDE_age_of_B_l252_25291

/-- Given the ages of four people A, B, C, and D, prove that B's age is 27 years. -/
theorem age_of_B (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 28 →
  (a + c) / 2 = 29 →
  (2 * b + 3 * d) / 5 = 27 →
  b = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_of_B_l252_25291


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l252_25247

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 31) + Real.sqrt (17 - x) + Real.sqrt x ≤ 12 ∧
  ∃ x₀, x₀ = 13 ∧ Real.sqrt (x₀ + 31) + Real.sqrt (17 - x₀) + Real.sqrt x₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l252_25247


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l252_25232

def polynomial (x : ℝ) : ℝ := -2 * (x^7 - x^4 + 3*x^2 - 5) + 4*(x^3 + 2*x) - 3*(x^5 - 4)

theorem sum_of_coefficients : 
  (polynomial 1) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l252_25232


namespace NUMINAMATH_CALUDE_symmetry_implies_m_sqrt3_l252_25270

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line mx - y + 1 = 0 -/
def line_of_symmetry (m : ℝ) (p : Point) : Prop :=
  m * p.x - p.y + 1 = 0

/-- The line x + y = 0 -/
def line_xy (p : Point) : Prop :=
  p.x + p.y = 0

/-- Two points are symmetric with respect to a line -/
def symmetric_points (m : ℝ) (p q : Point) : Prop :=
  ∃ (mid : Point), line_of_symmetry m mid ∧
    mid.x = (p.x + q.x) / 2 ∧
    mid.y = (p.y + q.y) / 2

theorem symmetry_implies_m_sqrt3 :
  ∀ (m : ℝ) (N : Point),
    symmetric_points m (Point.mk 1 0) N →
    line_xy N →
    m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_sqrt3_l252_25270


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l252_25282

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0 --/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 1 = 0 --/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

theorem equation_is_quadratic : IsQuadraticEquation f := by
  sorry


end NUMINAMATH_CALUDE_equation_is_quadratic_l252_25282


namespace NUMINAMATH_CALUDE_square_of_1023_l252_25295

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l252_25295


namespace NUMINAMATH_CALUDE_sum_of_roots_is_six_l252_25225

-- Define the quadratic polynomials
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem sum_of_roots_is_six 
  (a b c d : ℝ) 
  (hf : ∃ r₁ r₂ : ℝ, ∀ x, f a b x = (x - r₁) * (x - r₂))
  (hg : ∃ s₁ s₂ : ℝ, ∀ x, g c d x = (x - s₁) * (x - s₂))
  (h_eq1 : f a b 1 = g c d 2)
  (h_eq2 : g c d 1 = f a b 2) :
  ∃ r₁ r₂ s₁ s₂ : ℝ, r₁ + r₂ + s₁ + s₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_six_l252_25225


namespace NUMINAMATH_CALUDE_power_product_equals_l252_25263

theorem power_product_equals : (3 : ℕ)^6 * (4 : ℕ)^6 = 2985984 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_l252_25263


namespace NUMINAMATH_CALUDE_distinct_collections_count_l252_25217

/-- Represents the number of occurrences of each letter in "CALCULATIONS" --/
def letterCounts : Fin 26 → ℕ
| 0 => 3  -- A
| 2 => 3  -- C
| 8 => 1  -- I
| 11 => 3 -- L
| 13 => 1 -- N
| 14 => 1 -- O
| 18 => 1 -- S
| 19 => 1 -- T
| 20 => 1 -- U
| _ => 0

/-- Predicate for vowels --/
def isVowel (n : Fin 26) : Bool :=
  n = 0 ∨ n = 4 ∨ n = 8 ∨ n = 14 ∨ n = 20

/-- The number of distinct collections of three vowels and three consonants --/
def distinctCollections : ℕ := sorry

/-- Theorem stating that the number of distinct collections is 126 --/
theorem distinct_collections_count :
  distinctCollections = 126 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l252_25217


namespace NUMINAMATH_CALUDE_curve_is_line_l252_25248

/-- The curve defined by the polar equation θ = 5π/6 is a line -/
theorem curve_is_line : ∀ (r : ℝ) (θ : ℝ), 
  θ = (5 * Real.pi) / 6 → 
  ∃ (a b : ℝ), ∀ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ → 
  a * x + b * y = 0 :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_l252_25248


namespace NUMINAMATH_CALUDE_trig_identity_l252_25209

theorem trig_identity (θ : ℝ) (h : θ ≠ 0) (h' : θ ≠ π/2) : 
  (Real.sin θ + 1 / Real.sin θ)^2 + (Real.cos θ + 1 / Real.cos θ)^2 = 
  6 + 2 * ((Real.sin θ / Real.cos θ)^2 + (Real.cos θ / Real.sin θ)^2) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l252_25209


namespace NUMINAMATH_CALUDE_remainder_3_pow_20_mod_5_l252_25202

theorem remainder_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_20_mod_5_l252_25202


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l252_25252

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := 3, c := -4 }
  let l2 : Line := { a := 3, b := -2, c := 2 }
  perpendicular l1 l2 ∧ point_on_line 0 1 l2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l252_25252


namespace NUMINAMATH_CALUDE_kennel_arrangement_l252_25261

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  (Nat.factorial 3) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats)

/-- Theorem: The number of ways to arrange 3 chickens, 3 dogs, and 4 cats
    in a row of 10 cages, with animals of each type in adjacent cages,
    is 5184 -/
theorem kennel_arrangement : arrange_animals 3 3 4 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_kennel_arrangement_l252_25261


namespace NUMINAMATH_CALUDE_first_group_size_l252_25226

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of men in the second group -/
def second_group_men : ℕ := 25

/-- The number of days it takes the second group to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the first group -/
def first_group_men : ℕ := first_group_days * second_group_men * second_group_days / first_group_days

theorem first_group_size :
  first_group_men = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l252_25226


namespace NUMINAMATH_CALUDE_sum_of_even_integers_102_to_200_l252_25229

theorem sum_of_even_integers_102_to_200 :
  let first_term : ℕ := 102
  let last_term : ℕ := 200
  let num_terms : ℕ := 50
  (num_terms : ℚ) / 2 * (first_term + last_term) = 7550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_102_to_200_l252_25229


namespace NUMINAMATH_CALUDE_mikail_birthday_money_l252_25281

/-- Mikail's age tomorrow -/
def mikail_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Cost of the video game -/
def game_cost : ℕ := 80

/-- Theorem stating Mikail's situation -/
theorem mikail_birthday_money :
  (mikail_age = 3 * 3) ∧
  (mikail_age * money_per_year = 45) ∧
  (mikail_age * money_per_year < game_cost) :=
by sorry

end NUMINAMATH_CALUDE_mikail_birthday_money_l252_25281


namespace NUMINAMATH_CALUDE_function_behavior_l252_25204

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l252_25204


namespace NUMINAMATH_CALUDE_product_of_fractions_l252_25267

theorem product_of_fractions : 
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l252_25267


namespace NUMINAMATH_CALUDE_multiplication_equation_l252_25243

theorem multiplication_equation :
  ∀ (multiplier multiplicand product : ℕ),
    multiplier = 6 →
    multiplicand = product - 140 →
    multiplier * multiplicand = product →
    (multiplier = 6 ∧ multiplicand = 28 ∧ product = 168) :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_l252_25243


namespace NUMINAMATH_CALUDE_sequence_ratio_l252_25227

/-- Given two sequences where (-1, a₁, a₂, 8) form an arithmetic sequence
and (-1, b₁, b₂, b₃, -4) form a geometric sequence,
prove that (a₁ * a₂) / b₂ = -5 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) → 
  (a₂ - a₁ = 8 - a₂) → 
  (b₁ / (-1 : ℝ) = b₂ / b₁) → 
  (b₂ / b₁ = b₃ / b₂) → 
  (b₃ / b₂ = (-4 : ℝ) / b₃) → 
  (a₁ * a₂) / b₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l252_25227


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l252_25224

theorem arithmetic_sequence_problem (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = a 3 - a 2) →  -- arithmetic sequence property
  a 3 - a 2 = -2 →                     -- given condition
  a 7 = -2 →                           -- given condition
  a 9 = -6 :=                          -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l252_25224


namespace NUMINAMATH_CALUDE_unique_expansion_terms_l252_25287

def expansion_terms (N : ℕ) : ℕ := Nat.choose N 5

theorem unique_expansion_terms : 
  ∃! N : ℕ, N > 0 ∧ expansion_terms N = 231 :=
by sorry

end NUMINAMATH_CALUDE_unique_expansion_terms_l252_25287


namespace NUMINAMATH_CALUDE_f_properties_l252_25245

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + 2 * Real.sin x * Real.cos x - Real.cos x ^ 4

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f x ≥ -2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → x < y → f x < f y) ∧
  (∀ (x : ℝ), x ∈ Set.Ioc (Real.pi / 2) Real.pi → ∀ (y : ℝ), y ∈ Set.Ioc (Real.pi / 2) Real.pi → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l252_25245


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l252_25285

theorem sum_of_squares_of_roots (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁^4 + 6*r₁^3 + 11*r₁^2 + 6*r₁ + 1 = 0) →
  (r₂^4 + 6*r₂^3 + 11*r₂^2 + 6*r₂ + 1 = 0) →
  (r₃^4 + 6*r₃^3 + 11*r₃^2 + 6*r₃ + 1 = 0) →
  (r₄^4 + 6*r₄^3 + 11*r₄^2 + 6*r₄ + 1 = 0) →
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l252_25285


namespace NUMINAMATH_CALUDE_max_profit_plan_l252_25275

/-- Represents the production plan for cars -/
structure CarProduction where
  a : ℕ  -- number of A type cars
  b : ℕ  -- number of B type cars

/-- Calculates the total cost of production -/
def total_cost (p : CarProduction) : ℕ := 30 * p.a + 40 * p.b

/-- Calculates the total revenue from sales -/
def total_revenue (p : CarProduction) : ℕ := 35 * p.a + 50 * p.b

/-- Calculates the profit from a production plan -/
def profit (p : CarProduction) : ℤ := total_revenue p - total_cost p

/-- Theorem stating that the maximum profit is achieved with 5 A type cars and 35 B type cars -/
theorem max_profit_plan :
  ∀ p : CarProduction,
    p.a + p.b = 40 →
    total_cost p ≤ 1550 →
    profit p ≥ 365 →
    profit p ≤ profit { a := 5, b := 35 } :=
by sorry

end NUMINAMATH_CALUDE_max_profit_plan_l252_25275


namespace NUMINAMATH_CALUDE_unique_seating_arrangement_l252_25234

-- Define the types of representatives
inductive Representative
| Martian
| Venusian
| Earthling

-- Define the seating arrangement as a function from chair number to representative
def SeatingArrangement := Fin 10 → Representative

-- Define the rules for valid seating arrangements
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  -- Martian must occupy chair 1
  arr 0 = Representative.Martian ∧
  -- Earthling must occupy chair 10
  arr 9 = Representative.Earthling ∧
  -- Representatives must be arranged in clockwise order: Martian, Venusian, Earthling, repeating
  (∀ i : Fin 10, arr i = Representative.Martian → arr ((i + 1) % 10) = Representative.Venusian) ∧
  (∀ i : Fin 10, arr i = Representative.Venusian → arr ((i + 1) % 10) = Representative.Earthling) ∧
  (∀ i : Fin 10, arr i = Representative.Earthling → arr ((i + 1) % 10) = Representative.Martian)

-- Theorem stating that there is exactly one valid seating arrangement
theorem unique_seating_arrangement :
  ∃! arr : SeatingArrangement, is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_unique_seating_arrangement_l252_25234


namespace NUMINAMATH_CALUDE_factors_of_12_correct_ratio_exists_in_factors_l252_25242

def is_factor (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def factors_of_12 : Set ℕ := {1, 2, 3, 4, 6, 12}

theorem factors_of_12_correct :
  ∀ n : ℕ, n ∈ factors_of_12 ↔ is_factor 12 n := by sorry

theorem ratio_exists_in_factors :
  ∃ a b c d : ℕ, a ∈ factors_of_12 ∧ b ∈ factors_of_12 ∧ c ∈ factors_of_12 ∧ d ∈ factors_of_12 ∧
  a * d = b * c ∧ a ≠ 0 ∧ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_factors_of_12_correct_ratio_exists_in_factors_l252_25242


namespace NUMINAMATH_CALUDE_max_value_function_l252_25259

theorem max_value_function (x y : ℝ) :
  (2*x + 3*y + 4) / Real.sqrt (x^2 + 2*y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l252_25259


namespace NUMINAMATH_CALUDE_company_valuation_l252_25230

theorem company_valuation (P A B : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 2 * B) : 
  P / (A + B) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_company_valuation_l252_25230


namespace NUMINAMATH_CALUDE_polynomial_roots_inequality_l252_25250

theorem polynomial_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    12 * x^3 + a * x^2 + b * x + c = 0 ∧ 
    12 * y^3 + a * y^2 + b * y + c = 0 ∧ 
    12 * z^3 + a * z^2 + b * z + c = 0) →
  (∀ x : ℝ, (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c ≠ 0) →
  2001^3 + a * 2001^2 + b * 2001 + c > 1/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_inequality_l252_25250


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l252_25233

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem extreme_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 3 ∧ f b = -17 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l252_25233


namespace NUMINAMATH_CALUDE_expand_product_l252_25279

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10*y + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l252_25279


namespace NUMINAMATH_CALUDE_sequence_errors_l252_25246

-- Part (a)
def sequence_a (x y z : ℝ) : Prop :=
  (225 / 25 + 75 = 100 - 16) ∧
  (25 * (9 / (1 + 3)) = 84) ∧
  (25 * 12 = 84) ∧
  (25 = 7)

-- Part (b)
def sequence_b (x y z : ℝ) : Prop :=
  (5005 - 2002 = 35 * 143 - 143 * 14) ∧
  (5005 - 35 * 143 = 2002 - 143 * 14) ∧
  (5 * (1001 - 7 * 143) = 2 * (1001 - 7 * 143)) ∧
  (5 = 2)

theorem sequence_errors :
  ¬(∃ x y z : ℝ, sequence_a x y z) ∧
  ¬(∃ x y z : ℝ, sequence_b x y z) :=
sorry

end NUMINAMATH_CALUDE_sequence_errors_l252_25246


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l252_25205

theorem fourth_root_of_256000000 : Real.sqrt (Real.sqrt 256000000) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l252_25205


namespace NUMINAMATH_CALUDE_min_copy_paste_actions_l252_25278

theorem min_copy_paste_actions (n : ℕ) : (2^n - 1 ≥ 1000) ∧ (∀ m : ℕ, m < n → 2^m - 1 < 1000) ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_copy_paste_actions_l252_25278


namespace NUMINAMATH_CALUDE_function_monotonicity_implies_a_value_l252_25211

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - a*x

/-- The function f is decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The function f is increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- If f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem function_monotonicity_implies_a_value (a : ℝ) :
  decreasing_on_left a → increasing_on_right a → a = 4 := by sorry

end NUMINAMATH_CALUDE_function_monotonicity_implies_a_value_l252_25211


namespace NUMINAMATH_CALUDE_quadratic_inequality_l252_25272

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- State the theorem
theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ < 1) 
  (h₂ : 2 ≤ x₂ ∧ x₂ < 3) 
  (hy₁ : y₁ = f x₁) 
  (hy₂ : y₂ = f x₂) : 
  y₁ ≥ y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l252_25272


namespace NUMINAMATH_CALUDE_apple_distribution_l252_25290

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 3

/-- The number of apples Kevin has -/
def kevins_apples : ℕ := 2 * jackies_apples

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 8

/-- The total number of apples Adam, Jackie, and Kevin have -/
def total_apples : ℕ := jackies_apples + kevins_apples + adams_apples

/-- The number of apples He has -/
def his_apples : ℕ := 3 * total_apples

theorem apple_distribution :
  total_apples = 20 ∧ his_apples = 60 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l252_25290


namespace NUMINAMATH_CALUDE_dust_storm_coverage_l252_25210

/-- The dust storm problem -/
theorem dust_storm_coverage (total_prairie : ℕ) (untouched : ℕ) (covered : ℕ) : 
  total_prairie = 64013 → untouched = 522 → covered = total_prairie - untouched → covered = 63491 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_coverage_l252_25210


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l252_25284

/-- The volume of a cylinder formed by rotating a square around one of its sides. -/
theorem volume_cylinder_from_square_rotation (side_length : Real) (volume : Real) : 
  side_length = 2 → volume = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l252_25284


namespace NUMINAMATH_CALUDE_four_correct_statements_l252_25207

theorem four_correct_statements : 
  (∀ x : ℝ, Irrational x → ¬ (∃ q : ℚ, x = ↑q)) ∧ 
  ({x : ℝ | x^2 = 4} = {2, -2}) ∧
  ({x : ℝ | x^3 = x} = {-1, 0, 1}) ∧
  (∀ x : ℝ, ∃! p : ℝ, p = x) := by
  sorry

end NUMINAMATH_CALUDE_four_correct_statements_l252_25207


namespace NUMINAMATH_CALUDE_y_value_l252_25215

theorem y_value (y : ℚ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l252_25215


namespace NUMINAMATH_CALUDE_cone_properties_l252_25273

/-- A cone with vertex P, base radius √3, and lateral area 2√3π -/
structure Cone where
  vertex : Point
  base_radius : ℝ
  lateral_area : ℝ
  h_base_radius : base_radius = Real.sqrt 3
  h_lateral_area : lateral_area = 2 * Real.sqrt 3 * Real.pi

/-- The length of the generatrix of the cone -/
def generatrix_length (c : Cone) : ℝ := sorry

/-- The angle between the generatrix and the base of the cone -/
def generatrix_base_angle (c : Cone) : ℝ := sorry

theorem cone_properties (c : Cone) : 
  generatrix_length c = 2 ∧ generatrix_base_angle c = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_properties_l252_25273


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l252_25262

/-- Given two digits A and B in base d > 6, if AB_d + AA_d = 172_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h_d : d > 6)
  (h_digits : A < d ∧ B < d)
  (h_sum : d * B + A + d * A + A = d^2 + 7 * d + 2) :
  A - B = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l252_25262


namespace NUMINAMATH_CALUDE_no_fixed_points_composition_l252_25251

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + x

-- Theorem statement
theorem no_fixed_points_composition
  (a b : ℝ)
  (h : ∀ x : ℝ, f a b x ≠ x) :
  ∀ x : ℝ, f a b (f a b x) ≠ x :=
by sorry

end NUMINAMATH_CALUDE_no_fixed_points_composition_l252_25251


namespace NUMINAMATH_CALUDE_subtract_negative_two_from_three_l252_25293

theorem subtract_negative_two_from_three : 3 - (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_two_from_three_l252_25293


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l252_25274

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The statement to prove -/
theorem geometric_sequence_sum :
  (S 4 = 4) → (S 8 = 12) → (S 16 = 60) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l252_25274


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l252_25294

theorem decorative_window_area_ratio :
  let base : ℝ := 40
  let length : ℝ := (4/3) * base
  let semi_major_axis : ℝ := base / 2
  let semi_minor_axis : ℝ := base / 4
  let rectangle_area : ℝ := length * base
  let ellipse_area : ℝ := π * semi_major_axis * semi_minor_axis
  let triangle_area : ℝ := (1/2) * base * semi_minor_axis
  rectangle_area / (ellipse_area + triangle_area) = 32 / (3 * (π + 1)) :=
by sorry

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l252_25294


namespace NUMINAMATH_CALUDE_point_transformation_l252_25206

def initial_point : ℝ × ℝ × ℝ := (2, 3, -1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (reflect_xz (rotate_z_90 p))

theorem point_transformation :
  transform initial_point = (3, -2, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l252_25206


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l252_25299

/-- Given two rectangles with equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a length of 5 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (length1 width1 length2 width2 : ℝ) : 
  area = length1 * width1 → -- Area of the first rectangle
  area = length2 * width2 → -- Area of the second rectangle
  length1 = 4 →             -- Length of the first rectangle
  width1 = 30 →             -- Width of the first rectangle
  length2 = 5 →             -- Length of the second rectangle
  width2 = 24 :=            -- Width of the second rectangle (to be proved)
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l252_25299


namespace NUMINAMATH_CALUDE_complete_square_sum_l252_25249

theorem complete_square_sum (x : ℝ) : 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0) → 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0 ∧ d + e = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l252_25249
