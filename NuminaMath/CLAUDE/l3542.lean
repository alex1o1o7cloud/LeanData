import Mathlib

namespace NUMINAMATH_CALUDE_scarves_per_box_l3542_354257

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_pieces : ℕ) : 
  num_boxes = 3 → 
  mittens_per_box = 4 → 
  total_pieces = 21 → 
  (total_pieces - num_boxes * mittens_per_box) / num_boxes = 3 :=
by sorry

end NUMINAMATH_CALUDE_scarves_per_box_l3542_354257


namespace NUMINAMATH_CALUDE_complex_equation_solution_complex_inequality_range_l3542_354286

-- Problem 1
theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * I → z = -15 + 8 * I := by sorry

-- Problem 2
theorem complex_inequality_range (a : ℝ) :
  Complex.abs (3 + a * I) < 4 → -Real.sqrt 7 < a ∧ a < Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_complex_inequality_range_l3542_354286


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_value_l3542_354273

theorem parallel_vectors_fraction_value (α : ℝ) :
  let a : ℝ × ℝ := (Real.sin α, Real.cos α - 2 * Real.sin α)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_value_l3542_354273


namespace NUMINAMATH_CALUDE_min_sum_squares_l3542_354221

def S : Finset ℤ := {-8, -6, -4, -1, 1, 3, 5, 12}

theorem min_sum_squares (p q r s t u v w : ℤ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  2 ≤ (p + q + r + s)^2 + (t + u + v + w)^2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3542_354221


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3542_354201

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3542_354201


namespace NUMINAMATH_CALUDE_f_value_at_3_l3542_354233

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) :
  f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3542_354233


namespace NUMINAMATH_CALUDE_one_more_square_possible_l3542_354207

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a square that can be cut out from the grid -/
structure Square :=
  (size : ℕ)

/-- The number of squares already cut out -/
def cut_squares : ℕ := 15

/-- The function that determines if it's possible to cut out one more square -/
def can_cut_one_more (g : Grid) (s : Square) (n : ℕ) : Prop :=
  ∃ (remaining : ℕ), remaining > 0

/-- The theorem statement -/
theorem one_more_square_possible (g : Grid) (s : Square) :
  g.size = 11 → s.size = 2 → can_cut_one_more g s cut_squares :=
sorry

end NUMINAMATH_CALUDE_one_more_square_possible_l3542_354207


namespace NUMINAMATH_CALUDE_earth_inhabitable_surface_l3542_354261

theorem earth_inhabitable_surface (exposed_land : ℚ) (inhabitable_land : ℚ) 
  (h1 : exposed_land = 3 / 8)
  (h2 : inhabitable_land = 2 / 3) :
  exposed_land * inhabitable_land = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_surface_l3542_354261


namespace NUMINAMATH_CALUDE_min_value_a_plus_3b_l3542_354231

theorem min_value_a_plus_3b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_eq : a + 3*b = 1/a + 3/b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 1/x + 3/y → a + 3*b ≤ x + 3*y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 1/x + 3/y ∧ x + 3*y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_3b_l3542_354231


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3542_354226

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3542_354226


namespace NUMINAMATH_CALUDE_race_overtake_points_l3542_354298

-- Define the race parameters
def kelly_head_start : ℝ := 3
def kelly_speed : ℝ := 9
def abel_speed : ℝ := 9.5
def chris_speed : ℝ := 10
def chris_start_behind : ℝ := 2
def abel_loss_distance : ℝ := 0.75

-- Define the overtake points
def abel_overtake_kelly : ℝ := 54.75
def chris_overtake_both : ℝ := 56

-- Theorem statement
theorem race_overtake_points : 
  kelly_head_start = 3 ∧ 
  kelly_speed = 9 ∧ 
  abel_speed = 9.5 ∧ 
  chris_speed = 10 ∧ 
  chris_start_behind = 2 ∧
  abel_loss_distance = 0.75 →
  (abel_overtake_kelly = 54.75 ∧ chris_overtake_both = 56) := by
  sorry

end NUMINAMATH_CALUDE_race_overtake_points_l3542_354298


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l3542_354262

theorem fraction_irreducibility (n : ℕ) : 
  (Nat.gcd (2*n^2 + 11*n - 18) (n + 7) = 1) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l3542_354262


namespace NUMINAMATH_CALUDE_halloween_candy_percentage_l3542_354217

theorem halloween_candy_percentage (maggie_candy : ℕ) (neil_percentage : ℚ) (neil_candy : ℕ) :
  maggie_candy = 50 →
  neil_percentage = 40 / 100 →
  neil_candy = 91 →
  ∃ harper_percentage : ℚ,
    harper_percentage = 30 / 100 ∧
    neil_candy = (1 + neil_percentage) * (maggie_candy + harper_percentage * maggie_candy) :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_percentage_l3542_354217


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l3542_354299

theorem complex_exp_conversion :
  (Complex.exp (13 * Real.pi * Complex.I / 4)) * (Complex.ofReal (Real.sqrt 2)) = -1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l3542_354299


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3542_354208

def set_A : Set ℝ := {x | x + 2 = 0}
def set_B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3542_354208


namespace NUMINAMATH_CALUDE_sum_of_areas_is_1168_l3542_354235

/-- A square in the xy-plane with three vertices having x-coordinates 2, 0, and 18 -/
structure SquareXY where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4
  x_coords : {2, 0, 18} ⊆ vertices.image Prod.fst

/-- The sum of all possible areas of the square -/
def sum_of_possible_areas (s : SquareXY) : ℝ :=
  -- Definition of sum_of_possible_areas
  sorry

/-- Theorem: The sum of all possible areas of the square is 1168 -/
theorem sum_of_areas_is_1168 (s : SquareXY) : sum_of_possible_areas s = 1168 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_areas_is_1168_l3542_354235


namespace NUMINAMATH_CALUDE_weather_report_totals_l3542_354236

/-- Represents the different weather conditions --/
inductive WeatherCondition
  | Rain
  | Overcast
  | Sunshine
  | Thunderstorm

/-- Represents a day's weather report --/
structure DayReport where
  hours : WeatherCondition → Nat
  total_hours : (hours Rain) + (hours Overcast) + (hours Sunshine) + (hours Thunderstorm) = 12

/-- The weather report for the three days --/
def three_day_report : Vector DayReport 3 :=
  ⟨[
    { hours := λ c => match c with
                      | WeatherCondition.Rain => 6
                      | WeatherCondition.Overcast => 6
                      | _ => 0,
      total_hours := sorry },
    { hours := λ c => match c with
                      | WeatherCondition.Sunshine => 6
                      | WeatherCondition.Overcast => 4
                      | WeatherCondition.Rain => 2
                      | _ => 0,
      total_hours := sorry },
    { hours := λ c => match c with
                      | WeatherCondition.Thunderstorm => 2
                      | WeatherCondition.Overcast => 4
                      | WeatherCondition.Sunshine => 6
                      | _ => 0,
      total_hours := sorry }
  ], sorry⟩

/-- The total hours for each weather condition over the three days --/
def total_hours (c : WeatherCondition) : Nat :=
  (three_day_report.get 0).hours c + (three_day_report.get 1).hours c + (three_day_report.get 2).hours c

/-- The main theorem to prove --/
theorem weather_report_totals :
  total_hours WeatherCondition.Rain = 8 ∧
  total_hours WeatherCondition.Overcast = 14 ∧
  total_hours WeatherCondition.Sunshine = 12 ∧
  total_hours WeatherCondition.Thunderstorm = 2 := by
  sorry

end NUMINAMATH_CALUDE_weather_report_totals_l3542_354236


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l3542_354224

theorem quadratic_radical_equality (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a + 2 = k * (3 * a)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l3542_354224


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l3542_354270

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l3542_354270


namespace NUMINAMATH_CALUDE_leftover_coins_value_l3542_354279

/-- The number of nickels in a complete roll -/
def nickels_per_roll : ℕ := 40

/-- The number of pennies in a complete roll -/
def pennies_per_roll : ℕ := 50

/-- The number of nickels Sarah has -/
def sarah_nickels : ℕ := 132

/-- The number of pennies Sarah has -/
def sarah_pennies : ℕ := 245

/-- The number of nickels Tom has -/
def tom_nickels : ℕ := 98

/-- The number of pennies Tom has -/
def tom_pennies : ℕ := 203

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The total value of leftover coins after combining and rolling -/
def leftover_value : ℚ :=
  (((sarah_nickels + tom_nickels) % nickels_per_roll : ℚ) * nickel_value) +
  (((sarah_pennies + tom_pennies) % pennies_per_roll : ℚ) * penny_value)

theorem leftover_coins_value :
  leftover_value = 1.98 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l3542_354279


namespace NUMINAMATH_CALUDE_school_average_age_l3542_354285

theorem school_average_age (total_students : ℕ) (boys_avg_age girls_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 158 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_school_average_age_l3542_354285


namespace NUMINAMATH_CALUDE_blue_tile_probability_l3542_354211

/-- A function that determines if a number is congruent to 3 mod 7 -/
def isBlue (n : ℕ) : Bool :=
  n % 7 = 3

/-- The total number of tiles in the box -/
def totalTiles : ℕ := 70

/-- The number of blue tiles in the box -/
def blueTiles : ℕ := (List.range totalTiles).filter isBlue |>.length

/-- The probability of selecting a blue tile -/
def probabilityBlue : ℚ := blueTiles / totalTiles

theorem blue_tile_probability :
  probabilityBlue = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_blue_tile_probability_l3542_354211


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l3542_354267

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 12 (Nat.gcd 18 30)) → 
  (B = Nat.lcm 12 (Nat.lcm 18 30)) → 
  2 * A + B = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l3542_354267


namespace NUMINAMATH_CALUDE_sum_of_squares_l3542_354228

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (power_eq : a^7 + b^7 + c^7 = a^9 + b^9 + c^9) :
  a^2 + b^2 + c^2 = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3542_354228


namespace NUMINAMATH_CALUDE_smallest_number_meeting_criteria_l3542_354275

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def count_even_digits (n : ℕ) : ℕ :=
  (if n % 2 = 0 then 1 else 0) +
  (if (n / 10) % 2 = 0 then 1 else 0) +
  (if (n / 100) % 2 = 0 then 1 else 0) +
  (if (n / 1000) % 2 = 0 then 1 else 0)

def count_odd_digits (n : ℕ) : ℕ := 4 - count_even_digits n

def meets_criteria (n : ℕ) : Prop :=
  is_four_digit n ∧
  divisible_by_9 n ∧
  count_even_digits n = 3 ∧
  count_odd_digits n = 1

theorem smallest_number_meeting_criteria :
  ∀ n : ℕ, meets_criteria n → n ≥ 2043 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_meeting_criteria_l3542_354275


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3542_354219

theorem arithmetic_calculations :
  ((1 : ℤ) - 4 + 8 - 5 = -1) ∧ 
  (24 / (-3 : ℤ) - (-2)^3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3542_354219


namespace NUMINAMATH_CALUDE_function_inequality_l3542_354251

open Real

theorem function_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : -1 < log x₁) (h₄ : -1 < log x₂) : 
  let f := fun (x : ℝ) => x * log x
  x₁ * f x₁ + x₂ * f x₂ > 2 * x₂ * f x₁ := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3542_354251


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3542_354266

/-- The distance between the foci of an ellipse with center (3, 2), semi-major axis 7, and semi-minor axis 3 is 4√10. -/
theorem ellipse_foci_distance :
  ∀ (center : ℝ × ℝ) (semi_major semi_minor : ℝ),
    center = (3, 2) →
    semi_major = 7 →
    semi_minor = 3 →
    let c := Real.sqrt (semi_major^2 - semi_minor^2)
    2 * c = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3542_354266


namespace NUMINAMATH_CALUDE_park_visit_cost_family_park_visit_cost_l3542_354247

/-- Calculates the total cost for a family visiting a park with given conditions -/
theorem park_visit_cost (entrance_fee : ℝ) (kid_attraction_fee : ℝ) (adult_attraction_fee : ℝ)
  (entrance_discount_rate : ℝ) (senior_attraction_discount_rate : ℝ)
  (num_children num_parents num_grandparents : ℕ) : ℝ :=
  let total_people := num_children + num_parents + num_grandparents
  let entrance_discount := if total_people ≥ 6 then entrance_discount_rate else 0
  let entrance_cost := (1 - entrance_discount) * entrance_fee * total_people
  let attraction_cost_children := kid_attraction_fee * num_children
  let attraction_cost_parents := adult_attraction_fee * num_parents
  let attraction_cost_grandparents := 
    adult_attraction_fee * (1 - senior_attraction_discount_rate) * num_grandparents
  let total_cost := entrance_cost + attraction_cost_children + 
                    attraction_cost_parents + attraction_cost_grandparents
  total_cost

/-- The total cost for the family visit is $49.50 -/
theorem family_park_visit_cost : 
  park_visit_cost 5 2 4 0.1 0.5 4 2 1 = 49.5 := by
  sorry

end NUMINAMATH_CALUDE_park_visit_cost_family_park_visit_cost_l3542_354247


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l3542_354252

/-- Given a sales price and gross profit, calculate the percentage of gross profit relative to the cost -/
theorem gross_profit_percentage 
  (sales_price : ℝ) 
  (gross_profit : ℝ) 
  (h1 : sales_price = 81)
  (h2 : gross_profit = 51) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 170 := by
sorry


end NUMINAMATH_CALUDE_gross_profit_percentage_l3542_354252


namespace NUMINAMATH_CALUDE_minimum_greenhouse_dimensions_l3542_354206

/-- Represents the dimensions of a rectangular greenhouse. -/
structure Greenhouse where
  height : ℝ
  width : ℝ

/-- Checks if the greenhouse satisfies the given conditions. -/
def satisfiesConditions (g : Greenhouse) : Prop :=
  g.width = 2 * g.height ∧ g.height * g.width ≥ 800

/-- Theorem stating the minimum dimensions of the greenhouse. -/
theorem minimum_greenhouse_dimensions :
  ∃ (g : Greenhouse), satisfiesConditions g ∧
    ∀ (g' : Greenhouse), satisfiesConditions g' → g.height ≤ g'.height ∧ g.width ≤ g'.width :=
  sorry

end NUMINAMATH_CALUDE_minimum_greenhouse_dimensions_l3542_354206


namespace NUMINAMATH_CALUDE_max_vector_sum_on_circle_l3542_354295

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

def point_on_circle (p : ℝ × ℝ) : Prop := circle_C p.1 p.2

theorem max_vector_sum_on_circle (A B : ℝ × ℝ) :
  point_on_circle A →
  point_on_circle B →
  ‖(A.1 - B.1, A.2 - B.2)‖ = 2 * Real.sqrt 3 →
  ∃ (max : ℝ), max = 8 ∧ ∀ (A' B' : ℝ × ℝ),
    point_on_circle A' →
    point_on_circle B' →
    ‖(A'.1 - B'.1, A'.2 - B'.2)‖ = 2 * Real.sqrt 3 →
    ‖(A'.1 + B'.1, A'.2 + B'.2)‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_on_circle_l3542_354295


namespace NUMINAMATH_CALUDE_max_servings_is_56_l3542_354230

/-- Represents the ingredients required for one serving of salad -/
structure ServingRequirement where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the warehouse -/
structure WarehouseStock where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (req : ServingRequirement) (stock : WarehouseStock) : ℕ :=
  min
    (stock.cucumbers / req.cucumbers)
    (min
      (stock.tomatoes / req.tomatoes)
      (min
        (stock.brynza / req.brynza)
        (stock.peppers / req.peppers)))

/-- Theorem stating the maximum number of servings that can be made -/
theorem max_servings_is_56 :
  let req := ServingRequirement.mk 2 2 75 1
  let stock := WarehouseStock.mk 117 116 4200 60
  maxServings req stock = 56 := by
  sorry

#eval maxServings (ServingRequirement.mk 2 2 75 1) (WarehouseStock.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_servings_is_56_l3542_354230


namespace NUMINAMATH_CALUDE_graduating_students_average_score_l3542_354200

theorem graduating_students_average_score 
  (total_students : ℕ) 
  (overall_average : ℝ) 
  (graduating_students : ℕ) 
  (non_graduating_students : ℕ) 
  (graduating_average : ℝ) 
  (non_graduating_average : ℝ) :
  total_students = 100 →
  overall_average = 100 →
  non_graduating_students = (3 : ℝ) / 2 * graduating_students →
  graduating_average = (3 : ℝ) / 2 * non_graduating_average →
  total_students = graduating_students + non_graduating_students →
  (graduating_students : ℝ) * graduating_average + 
    (non_graduating_students : ℝ) * non_graduating_average = 
    (total_students : ℝ) * overall_average →
  graduating_average = 125 := by
sorry


end NUMINAMATH_CALUDE_graduating_students_average_score_l3542_354200


namespace NUMINAMATH_CALUDE_soccer_tournament_theorem_l3542_354216

/-- Represents a soccer tournament -/
structure SoccerTournament where
  n : ℕ  -- Total number of teams
  m : ℕ  -- Number of teams placed last
  h1 : n > m
  h2 : m ≥ 1

/-- Checks if the given n and m satisfy the tournament conditions -/
def validTournament (t : SoccerTournament) : Prop :=
  ∃ k : ℕ, k ≥ 1 ∧ t.n = (k + 1)^2 ∧ t.m = k * (k + 1) / 2

/-- The main theorem stating that only specific values of n and m are possible -/
theorem soccer_tournament_theorem (t : SoccerTournament) : validTournament t :=
  sorry

end NUMINAMATH_CALUDE_soccer_tournament_theorem_l3542_354216


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3542_354218

/-- Given a rectangle and a triangle where one side of the rectangle is the base of the triangle
    and one vertex of the triangle is on the opposite side of the rectangle,
    the ratio of the area of the rectangle to the area of the triangle is 2:1 -/
theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ) (rectangle_area triangle_area : ℝ),
    L > 0 → W > 0 →
    rectangle_area = L * W →
    triangle_area = (1 / 2) * L * W →
    rectangle_area / triangle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3542_354218


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l3542_354297

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem triangular_array_coin_sum :
  ∃ (N : ℕ), triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l3542_354297


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3542_354242

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3542_354242


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3542_354292

def regular_ticket_cost : ℝ := 10

theorem museum_ticket_cost :
  let discounted_ticket := 0.7 * regular_ticket_cost
  let full_price_ticket := regular_ticket_cost
  let total_spent := 44
  2 * discounted_ticket + 3 * full_price_ticket = total_spent :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3542_354292


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3542_354225

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5)^3 + a * (3 + Real.sqrt 5)^2 + b * (3 + Real.sqrt 5) - 40 = 0 → b = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3542_354225


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3542_354241

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

/-- Theorem: For a geometric sequence with a₁ = 2 and a₄ = 16, the common ratio q is 2 -/
theorem geometric_sequence_ratio : 
  ∀ (q : ℝ), 
    (geometric_sequence 2 q 1 = 2) ∧ 
    (geometric_sequence 2 q 4 = 16) → 
    q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3542_354241


namespace NUMINAMATH_CALUDE_parking_space_painted_sides_sum_l3542_354238

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  /-- The length of the unpainted side in feet -/
  unpainted_side : ℝ
  /-- The area of the parking space in square feet -/
  area : ℝ
  /-- The width of the parking space in feet -/
  width : ℝ
  /-- Assertion that the area equals length times width -/
  area_eq : area = unpainted_side * width

/-- The sum of the lengths of the painted sides of a parking space -/
def sum_painted_sides (p : ParkingSpace) : ℝ :=
  2 * p.width + p.unpainted_side

/-- Theorem stating that for a parking space with an unpainted side of 9 feet
    and an area of 125 square feet, the sum of the painted sides is 37 feet -/
theorem parking_space_painted_sides_sum :
  ∀ (p : ParkingSpace),
  p.unpainted_side = 9 ∧ p.area = 125 →
  sum_painted_sides p = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_space_painted_sides_sum_l3542_354238


namespace NUMINAMATH_CALUDE_constant_function_proof_l3542_354255

theorem constant_function_proof (g : ℝ → ℝ) 
  (h1 : ∃ x, g x ≠ 0)
  (h2 : ∀ a b : ℝ, g (a + b) + g (a - b) = g a + g b) :
  ∃ k : ℝ, ∀ x : ℝ, g x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3542_354255


namespace NUMINAMATH_CALUDE_chandler_can_buy_bike_l3542_354229

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 500

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 50 + 35 + 15

/-- Chandler's weekly earnings from the paper route in dollars -/
def weekly_earnings : ℕ := 16

/-- The number of weeks required to save enough money for the bike -/
def weeks_to_save : ℕ := 25

/-- Theorem stating that Chandler can buy the bike after saving for 25 weeks -/
theorem chandler_can_buy_bike : 
  birthday_money + weekly_earnings * weeks_to_save = bike_cost :=
sorry

end NUMINAMATH_CALUDE_chandler_can_buy_bike_l3542_354229


namespace NUMINAMATH_CALUDE_min_rods_eq_2n_minus_2_l3542_354268

/-- A puzzle is an n × n grid with n cells removed, no two in the same row or column -/
structure Puzzle (n : ℕ) where
  (n_ge_two : n ≥ 2)

/-- A rod is a 1 × k or k × 1 subgrid where k is a positive integer -/
inductive Rod
  | horizontal : ℕ+ → Rod
  | vertical : ℕ+ → Rod

/-- m(A) is the minimum number of rods needed to partition puzzle A -/
def min_rods (n : ℕ) (A : Puzzle n) : ℕ := sorry

/-- The main theorem: For any n × n puzzle A, m(A) = 2n - 2 -/
theorem min_rods_eq_2n_minus_2 (n : ℕ) (A : Puzzle n) : 
  min_rods n A = 2 * n - 2 := by sorry

end NUMINAMATH_CALUDE_min_rods_eq_2n_minus_2_l3542_354268


namespace NUMINAMATH_CALUDE_supermarket_bread_count_l3542_354248

/-- Calculates the final number of loaves in a supermarket after sales and delivery -/
def final_loaves (initial : ℕ) (sold : ℕ) (delivered : ℕ) : ℕ :=
  initial - sold + delivered

/-- Theorem stating that given the specific numbers from the problem, 
    the final number of loaves is 2215 -/
theorem supermarket_bread_count : 
  final_loaves 2355 629 489 = 2215 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_bread_count_l3542_354248


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l3542_354265

theorem common_root_of_polynomials (a b c d e f g : ℚ) :
  ∃ k : ℚ, k < 0 ∧ k ≠ ⌊k⌋ ∧
  (90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0) ∧
  (18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0) :=
by
  use -1/6
  sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l3542_354265


namespace NUMINAMATH_CALUDE_pentagon_area_given_equal_perimeter_square_l3542_354258

theorem pentagon_area_given_equal_perimeter_square (s : ℝ) (p : ℝ) : 
  s > 0 →
  p > 0 →
  4 * s = 5 * p →
  s^2 = 16 →
  abs ((5 * p^2 * Real.tan (3 * Real.pi / 10)) / 4 - 15.26) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_given_equal_perimeter_square_l3542_354258


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3542_354283

theorem arithmetic_expression_evaluation : 8 + 15 / 3 - 2^3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3542_354283


namespace NUMINAMATH_CALUDE_has_extremum_if_a_less_than_neg_one_l3542_354287

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

/-- Theorem stating that if a < -1, then f has an extremum -/
theorem has_extremum_if_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∃ x : ℝ, f_derivative a x = 0 :=
sorry

end NUMINAMATH_CALUDE_has_extremum_if_a_less_than_neg_one_l3542_354287


namespace NUMINAMATH_CALUDE_election_votes_l3542_354209

/-- Represents the total number of votes in an election --/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes for candidate A --/
def percent_A : ℚ := 45/100

/-- Represents the percentage of votes for candidate B --/
def percent_B : ℚ := 35/100

/-- Represents the percentage of votes for candidate C --/
def percent_C : ℚ := 1 - percent_A - percent_B

/-- The difference between votes for A and B --/
def diff_AB : ℕ := 500

/-- The difference between votes for B and C --/
def diff_BC : ℕ := 350

theorem election_votes : 
  (percent_A * total_votes : ℚ) - (percent_B * total_votes : ℚ) = diff_AB ∧
  (percent_B * total_votes : ℚ) - (percent_C * total_votes : ℚ) = diff_BC →
  total_votes = 5000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3542_354209


namespace NUMINAMATH_CALUDE_age_difference_l3542_354290

theorem age_difference (ana_age bonita_age : ℕ) : 
  (ana_age - 1 = 3 * (bonita_age - 1)) →  -- Last year's condition
  (ana_age = 2 * bonita_age + 3) →        -- This year's condition
  (ana_age - bonita_age = 8) :=           -- Age difference is 8
by
  sorry


end NUMINAMATH_CALUDE_age_difference_l3542_354290


namespace NUMINAMATH_CALUDE_x_value_proof_l3542_354284

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3542_354284


namespace NUMINAMATH_CALUDE_orange_weight_l3542_354245

theorem orange_weight (apple_weight : ℕ) (bag_capacity : ℕ) (num_bags : ℕ) (total_apple_weight : ℕ) :
  apple_weight = 4 →
  bag_capacity = 49 →
  num_bags = 3 →
  total_apple_weight = 84 →
  ∃ (orange_weight : ℕ),
    orange_weight * (total_apple_weight / apple_weight) = total_apple_weight ∧
    orange_weight = 4 :=
by
  sorry

#check orange_weight

end NUMINAMATH_CALUDE_orange_weight_l3542_354245


namespace NUMINAMATH_CALUDE_quadratic_problem_l3542_354294

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_problem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 4) →  -- Minimum value is 4
  (f a b c 2 = 4) →  -- Minimum occurs at x = 2
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 5 = 31) :=  -- Passes through (5, 31)
by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l3542_354294


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l3542_354203

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l3542_354203


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l3542_354296

theorem sum_of_squares_divisible_by_seven (a b : ℤ) :
  (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l3542_354296


namespace NUMINAMATH_CALUDE_inverse_sqrt_difference_equals_sum_l3542_354234

theorem inverse_sqrt_difference_equals_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  1 / (Real.sqrt a - Real.sqrt b) = Real.sqrt a + Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_inverse_sqrt_difference_equals_sum_l3542_354234


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l3542_354227

/-- 
Given a point P(a,b) on the graph of y = 4x + 3, 
prove that the value of 4a - b - 2 is -5
-/
theorem point_on_linear_graph (a b : ℝ) : 
  b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l3542_354227


namespace NUMINAMATH_CALUDE_sector_area_l3542_354282

theorem sector_area (θ : ℝ) (p : ℝ) (h1 : θ = 2) (h2 : p = 4) :
  let r := (p - θ) / 2
  let area := r^2 * θ / 2
  area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3542_354282


namespace NUMINAMATH_CALUDE_janet_needs_775_l3542_354277

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (months_advance : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * months_advance + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 :
  additional_money_needed 2225 1250 2 500 = 775 :=
by sorry

end NUMINAMATH_CALUDE_janet_needs_775_l3542_354277


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3542_354253

/-- A cyclic quadrilateral with angles in arithmetic progression -/
structure CyclicQuadrilateral where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic progression
  d : ℝ
  -- Ensures the quadrilateral is cyclic (opposite angles sum to 180°)
  cyclic : a + (a + 3*d) = 180 ∧ (a + d) + (a + 2*d) = 180
  -- Ensures the angles form an arithmetic sequence
  arithmetic_seq : true
  -- The largest angle is 140°
  largest_angle : a + 3*d = 140

/-- 
In a cyclic quadrilateral where the angles form an arithmetic sequence 
and the largest angle is 140°, the smallest angle measures 40°
-/
theorem smallest_angle_measure (q : CyclicQuadrilateral) : q.a = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3542_354253


namespace NUMINAMATH_CALUDE_inequality_proof_l3542_354272

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) ≤ 
  3 / 8 * (1 / a + 1 / b + 1 / c + 1 / d)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3542_354272


namespace NUMINAMATH_CALUDE_rodney_lifts_118_l3542_354222

/-- Represents the weight-lifting abilities of Rebecca, Rodney, Roger, and Ron. -/
structure WeightLifters where
  rebecca : ℕ
  rodney : ℕ
  roger : ℕ
  ron : ℕ

/-- The conditions of the weight-lifting problem. -/
def weightLiftingConditions (w : WeightLifters) : Prop :=
  w.rebecca + w.rodney + w.roger + w.ron = 375 ∧
  w.rodney = 2 * w.roger ∧
  w.roger = w.ron + 5 ∧
  w.rebecca = 3 * w.ron - 20

/-- Theorem stating that under the given conditions, Rodney can lift 118 pounds. -/
theorem rodney_lifts_118 (w : WeightLifters) :
  weightLiftingConditions w → w.rodney = 118 := by
  sorry


end NUMINAMATH_CALUDE_rodney_lifts_118_l3542_354222


namespace NUMINAMATH_CALUDE_books_read_theorem_l3542_354212

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the number of books read in a week
def books_read_in_week : ℕ :=
  2 + (fib 0) + (fib 1) + (fib 2) + (fib 3) + (fib 4) + (fib 5)

-- Theorem statement
theorem books_read_theorem : books_read_in_week = 22 := by
  sorry

end NUMINAMATH_CALUDE_books_read_theorem_l3542_354212


namespace NUMINAMATH_CALUDE_square_area_ratio_l3542_354232

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3542_354232


namespace NUMINAMATH_CALUDE_henrys_money_l3542_354243

theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (game_cost : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → game_cost = 10 →
  initial_amount + birthday_gift - game_cost = 19 := by
  sorry

end NUMINAMATH_CALUDE_henrys_money_l3542_354243


namespace NUMINAMATH_CALUDE_divisible_by_six_l3542_354263

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n : ℤ)^3 + 5*n = 6*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l3542_354263


namespace NUMINAMATH_CALUDE_remaining_shirt_cost_l3542_354244

/-- Given a set of shirts with known prices, calculate the price of the remaining shirts -/
theorem remaining_shirt_cost (total_shirts : ℕ) (total_cost : ℚ) (known_shirt_count : ℕ) (known_shirt_cost : ℚ) :
  total_shirts = 5 →
  total_cost = 85 →
  known_shirt_count = 3 →
  known_shirt_cost = 15 →
  (total_cost - (known_shirt_count * known_shirt_cost)) / (total_shirts - known_shirt_count) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_shirt_cost_l3542_354244


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l3542_354210

theorem two_numbers_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6) : 
  x = 18 ∧ y = 12 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l3542_354210


namespace NUMINAMATH_CALUDE_helen_cookies_yesterday_l3542_354278

def cookies_this_morning : ℕ := 270
def cookies_day_before_yesterday : ℕ := 419
def cookies_till_last_night : ℕ := 450

theorem helen_cookies_yesterday :
  cookies_day_before_yesterday + cookies_this_morning - cookies_till_last_night = 239 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_yesterday_l3542_354278


namespace NUMINAMATH_CALUDE_oil_per_cylinder_l3542_354280

theorem oil_per_cylinder (cylinders : ℕ) (oil_added : ℕ) (oil_needed : ℕ) :
  cylinders = 6 →
  oil_added = 16 →
  oil_needed = 32 →
  (oil_added + oil_needed) / cylinders = 8 := by
  sorry

end NUMINAMATH_CALUDE_oil_per_cylinder_l3542_354280


namespace NUMINAMATH_CALUDE_first_prime_of_nine_sum_100_l3542_354291

theorem first_prime_of_nine_sum_100 (primes : List Nat) : 
  primes.length = 9 ∧ 
  (∀ p ∈ primes, Nat.Prime p) ∧ 
  primes.sum = 100 →
  primes.head? = some 2 := by
sorry

end NUMINAMATH_CALUDE_first_prime_of_nine_sum_100_l3542_354291


namespace NUMINAMATH_CALUDE_maintenance_check_time_l3542_354288

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time + (1/3) * initial_time = 60) → initial_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l3542_354288


namespace NUMINAMATH_CALUDE_new_cube_edge_theorem_l3542_354205

/-- The edge length of a cube formed by melting five cubes -/
def new_cube_edge (a b c d e : ℝ) : ℝ :=
  (a^3 + b^3 + c^3 + d^3 + e^3) ^ (1/3)

/-- Theorem stating that the edge of the new cube is the cube root of the sum of volumes -/
theorem new_cube_edge_theorem :
  new_cube_edge 6 8 10 12 14 = (6^3 + 8^3 + 10^3 + 12^3 + 14^3) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_new_cube_edge_theorem_l3542_354205


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l3542_354254

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 4 < 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4)) / 5)) → n ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l3542_354254


namespace NUMINAMATH_CALUDE_box_length_l3542_354276

/-- Given a box with specified dimensions and cube properties, prove its length --/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  width = 16 →
  height = 6 →
  cube_volume = 3 →
  min_cubes = 384 →
  (min_cubes : ℝ) * cube_volume / (width * height) = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_box_length_l3542_354276


namespace NUMINAMATH_CALUDE_median_mean_equality_l3542_354223

theorem median_mean_equality (n : ℝ) : 
  let s := {n, n + 2, n + 7, n + 10, n + 16}
  n + 7 = 10 → (Finset.sum s id) / 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_median_mean_equality_l3542_354223


namespace NUMINAMATH_CALUDE_pitcher_distribution_percentage_l3542_354213

/-- Represents the contents of a pitcher -/
structure Pitcher :=
  (capacity : ℝ)
  (orange_juice : ℝ)
  (apple_juice : ℝ)

/-- Represents the distribution of the pitcher contents into cups -/
structure Distribution :=
  (pitcher : Pitcher)
  (num_cups : ℕ)

/-- The theorem stating the percentage of the pitcher's capacity in each cup -/
theorem pitcher_distribution_percentage (d : Distribution) 
  (h1 : d.pitcher.orange_juice = 2/3 * d.pitcher.capacity)
  (h2 : d.pitcher.apple_juice = 1/3 * d.pitcher.capacity)
  (h3 : d.num_cups = 6)
  (h4 : d.pitcher.capacity > 0) :
  (d.pitcher.capacity / d.num_cups) / d.pitcher.capacity = 1/6 := by
  sorry

#check pitcher_distribution_percentage

end NUMINAMATH_CALUDE_pitcher_distribution_percentage_l3542_354213


namespace NUMINAMATH_CALUDE_prob_even_product_three_dice_l3542_354259

-- Define a six-sided die
def SixSidedDie : Type := Fin 6

-- Define the probability of rolling an even number on a single die
def probEvenOnDie : ℚ := 1/2

-- Define the probability of rolling at least one even number on two dice
def probAtLeastOneEvenOnTwoDice : ℚ := 1 - (1 - probEvenOnDie) ^ 2

-- The main theorem
theorem prob_even_product_three_dice :
  probAtLeastOneEvenOnTwoDice = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_even_product_three_dice_l3542_354259


namespace NUMINAMATH_CALUDE_specific_pyramid_properties_l3542_354250

/-- Represents a straight pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  height : ℝ
  side_face_area : ℝ

/-- Calculates the base edge length of the pyramid -/
def base_edge_length (p : EquilateralPyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def volume (p : EquilateralPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem specific_pyramid_properties :
  let p : EquilateralPyramid := { height := 11, side_face_area := 210 }
  base_edge_length p = 30 ∧ volume p = 825 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_properties_l3542_354250


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3542_354269

theorem no_perfect_squares (x y : ℕ+) : 
  ¬(∃ (a b : ℕ), (x^2 + y + 2 : ℕ) = a^2 ∧ (y^2 + 4*x : ℕ) = b^2) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3542_354269


namespace NUMINAMATH_CALUDE_range_of_a_l3542_354237

-- Define the sets A, B, and C
def A : Set ℝ := {x | (2 - x) / (3 + x) ≥ 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a*(a + 1) < 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3542_354237


namespace NUMINAMATH_CALUDE_peach_difference_l3542_354220

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 17) 
  (h2 : green_peaches = 16) : 
  red_peaches - green_peaches = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3542_354220


namespace NUMINAMATH_CALUDE_mark_soup_cans_l3542_354215

/-- The number of cans of soup Mark bought -/
def soup_cans : ℕ := sorry

/-- The cost of one can of soup -/
def soup_cost : ℕ := 2

/-- The number of loaves of bread Mark bought -/
def bread_loaves : ℕ := 2

/-- The cost of one loaf of bread -/
def bread_cost : ℕ := 5

/-- The number of boxes of cereal Mark bought -/
def cereal_boxes : ℕ := 2

/-- The cost of one box of cereal -/
def cereal_cost : ℕ := 3

/-- The number of gallons of milk Mark bought -/
def milk_gallons : ℕ := 2

/-- The cost of one gallon of milk -/
def milk_cost : ℕ := 4

/-- The number of $10 bills Mark used to pay -/
def ten_dollar_bills : ℕ := 4

theorem mark_soup_cans : soup_cans = 8 := by sorry

end NUMINAMATH_CALUDE_mark_soup_cans_l3542_354215


namespace NUMINAMATH_CALUDE_snowdrift_depth_ratio_l3542_354274

theorem snowdrift_depth_ratio (initial_depth second_day_depth third_day_snow fourth_day_snow final_depth : ℝ) :
  initial_depth = 20 →
  third_day_snow = 6 →
  fourth_day_snow = 18 →
  final_depth = 34 →
  second_day_depth + third_day_snow + fourth_day_snow = final_depth →
  second_day_depth / initial_depth = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_snowdrift_depth_ratio_l3542_354274


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3542_354246

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 125 }
  let p2 : ConePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestDistanceOnCone cone p1 p2 = 125 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3542_354246


namespace NUMINAMATH_CALUDE_greg_travel_distance_l3542_354202

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let workplace_to_market : ℝ := 30  -- Distance in miles
  let market_to_home_time : ℝ := 0.5  -- Time in hours
  let market_to_home_speed : ℝ := 20  -- Speed in miles per hour
  let market_to_home : ℝ := market_to_home_speed * market_to_home_time
  let total_distance : ℝ := workplace_to_market + market_to_home
  total_distance = 40 := by
sorry


end NUMINAMATH_CALUDE_greg_travel_distance_l3542_354202


namespace NUMINAMATH_CALUDE_measure_45_minutes_l3542_354249

/-- Represents a string that can be burned --/
structure BurnableString where
  burnTime : ℝ
  nonUniformRate : Bool

/-- Represents a lighter --/
structure Lighter

/-- Represents the state of burning strings --/
inductive BurningState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Function to measure time using burnable strings and a lighter --/
def measureTime (strings : List BurnableString) (lighter : Lighter) : ℝ :=
  sorry

/-- Theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes :
  ∃ (strings : List BurnableString) (lighter : Lighter),
    strings.length = 2 ∧
    (∀ s ∈ strings, s.burnTime = 1 ∧ s.nonUniformRate = true) ∧
    measureTime strings lighter = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_measure_45_minutes_l3542_354249


namespace NUMINAMATH_CALUDE_temperature_problem_l3542_354260

def temperature_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 1) = x n + x (n + 2)

theorem temperature_problem (x : ℕ → ℤ) 
  (h_seq : temperature_sequence x)
  (h_3 : x 3 = 5)
  (h_31 : x 31 = 2) :
  x 25 = -3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_problem_l3542_354260


namespace NUMINAMATH_CALUDE_bag_of_balls_l3542_354289

theorem bag_of_balls (num_black : ℕ) (prob_black : ℚ) (total : ℕ) : 
  num_black = 4 → prob_black = 1/3 → total = num_black / prob_black → total = 12 := by
sorry

end NUMINAMATH_CALUDE_bag_of_balls_l3542_354289


namespace NUMINAMATH_CALUDE_min_value_on_line_l3542_354204

/-- The minimum value of 2^x + 4^y for points (x, y) on the line through (3, 0) and (1, 1) -/
theorem min_value_on_line : 
  ∀ (x y : ℝ), (x + 2*y = 3) → (2^x + 4^y ≥ 4 * Real.sqrt 2) ∧ 
  ∃ (x₀ y₀ : ℝ), (x₀ + 2*y₀ = 3) ∧ (2^x₀ + 4^y₀ = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l3542_354204


namespace NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_l3542_354264

def standard_deck_size : ℕ := 52
def jack_count : ℕ := 4
def queen_count : ℕ := 4

def probability_two_queens_or_at_least_one_jack : ℚ :=
  217 / 882

theorem prob_two_queens_or_at_least_one_jack :
  probability_two_queens_or_at_least_one_jack = 
    (Nat.choose queen_count 2 * (standard_deck_size - queen_count) + 
     (standard_deck_size - jack_count).choose 2 * jack_count + 
     (standard_deck_size - jack_count).choose 1 * Nat.choose jack_count 2 + 
     Nat.choose jack_count 3) / 
    Nat.choose standard_deck_size 3 :=
by
  sorry

#eval probability_two_queens_or_at_least_one_jack

end NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_l3542_354264


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l3542_354240

theorem different_color_chip_probability :
  let total_chips : ℕ := 7 + 5
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l3542_354240


namespace NUMINAMATH_CALUDE_system_solution_l3542_354271

theorem system_solution (x y : ℝ) : 
  (x^3 * y + x * y^3 = 10 ∧ x^4 + y^4 = 17) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3542_354271


namespace NUMINAMATH_CALUDE_factorization_equality_l3542_354239

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 4 * a * b + 2 * b^2 = 2 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3542_354239


namespace NUMINAMATH_CALUDE_intersection_point_l3542_354214

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 5 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem intersection_point : 
  ∃! (x y : ℝ), C1 x y ∧ C2 x y ∧ x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3542_354214


namespace NUMINAMATH_CALUDE_problem_statement_l3542_354281

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_property (f : ℝ → ℝ) : Prop := ∀ x, f (2 + x) = f (-x)

theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f)
  (h_f1 : f 1 = 2) : 
  f 2018 + f 2019 = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3542_354281


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l3542_354293

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l3542_354293


namespace NUMINAMATH_CALUDE_value_of_a_l3542_354256

theorem value_of_a (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) : 
  a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l3542_354256
