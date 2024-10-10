import Mathlib

namespace smallest_integer_in_consecutive_set_l1712_171292

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 4 < 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4)) / 5)) → n ≥ 1 :=
by
  sorry

end smallest_integer_in_consecutive_set_l1712_171292


namespace third_square_perimeter_l1712_171238

/-- Given two squares with perimeters 40 cm and 32 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 24 cm. -/
theorem third_square_perimeter (square1 square2 square3 : Real → Real → Real) :
  (∀ s, square1 s s = s * s) →
  (∀ s, square2 s s = s * s) →
  (∀ s, square3 s s = s * s) →
  (4 * 10 = 40) →
  (4 * 8 = 32) →
  (square1 10 10 - square2 8 8 = square3 6 6) →
  (4 * 6 = 24) := by
sorry

end third_square_perimeter_l1712_171238


namespace january_text_messages_l1712_171215

-- Define the sequence
def text_message_sequence : ℕ → ℕ
| 0 => 1  -- November (first month)
| n + 1 => 2 * text_message_sequence n  -- Each subsequent month

-- Theorem statement
theorem january_text_messages : text_message_sequence 2 = 4 := by
  sorry

end january_text_messages_l1712_171215


namespace negative_three_squared_plus_negative_two_cubed_l1712_171210

theorem negative_three_squared_plus_negative_two_cubed : -3^2 + (-2)^3 = -17 := by
  sorry

end negative_three_squared_plus_negative_two_cubed_l1712_171210


namespace weather_report_totals_l1712_171272

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

end weather_report_totals_l1712_171272


namespace saree_sale_price_l1712_171218

/-- Calculates the final price of a saree after discounts and tax -/
def finalSalePrice (initialPrice : ℝ) (discount1 discount2 discount3 taxRate : ℝ) : ℝ :=
  let price1 := initialPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

/-- The final sale price of a saree is approximately 298.55 Rs -/
theorem saree_sale_price :
  ∃ ε > 0, abs (finalSalePrice 560 0.2 0.3 0.15 0.12 - 298.55) < ε :=
sorry

end saree_sale_price_l1712_171218


namespace number_equation_solution_l1712_171206

theorem number_equation_solution : ∃ x : ℝ, (27 + 2 * x = 39) ∧ (x = 6) := by
  sorry

end number_equation_solution_l1712_171206


namespace halloween_candy_percentage_l1712_171290

theorem halloween_candy_percentage (maggie_candy : ℕ) (neil_percentage : ℚ) (neil_candy : ℕ) :
  maggie_candy = 50 →
  neil_percentage = 40 / 100 →
  neil_candy = 91 →
  ∃ harper_percentage : ℚ,
    harper_percentage = 30 / 100 ∧
    neil_candy = (1 + neil_percentage) * (maggie_candy + harper_percentage * maggie_candy) :=
by sorry

end halloween_candy_percentage_l1712_171290


namespace abs_z_squared_value_l1712_171214

theorem abs_z_squared_value (z : ℂ) (h : z^2 + Complex.abs z^2 = 7 + 6*I) : 
  Complex.abs z^2 = 85/14 := by
  sorry

end abs_z_squared_value_l1712_171214


namespace justin_tim_emily_games_l1712_171219

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in the larger game -/
def larger_game_players : ℕ := 7

/-- The number of specific players (Justin, Tim, and Emily) -/
def specific_players : ℕ := 3

theorem justin_tim_emily_games (h : total_players = 12 ∧ larger_game_players = 7 ∧ specific_players = 3) :
  Nat.choose (total_players - specific_players) (larger_game_players - specific_players) = 126 := by
  sorry

end justin_tim_emily_games_l1712_171219


namespace max_servings_is_56_l1712_171275

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

end max_servings_is_56_l1712_171275


namespace modulus_of_z_squared_l1712_171209

theorem modulus_of_z_squared (z : ℂ) (h : z^2 = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_squared_l1712_171209


namespace election_votes_l1712_171267

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

end election_votes_l1712_171267


namespace dogsled_race_time_difference_l1712_171222

theorem dogsled_race_time_difference 
  (course_length : ℝ) 
  (speed_T : ℝ) 
  (speed_difference : ℝ) :
  course_length = 300 →
  speed_T = 20 →
  speed_difference = 5 →
  let speed_A := speed_T + speed_difference
  let time_T := course_length / speed_T
  let time_A := course_length / speed_A
  time_T - time_A = 3 := by
sorry

end dogsled_race_time_difference_l1712_171222


namespace minimum_greenhouse_dimensions_l1712_171297

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

end minimum_greenhouse_dimensions_l1712_171297


namespace point_on_linear_graph_l1712_171254

/-- 
Given a point P(a,b) on the graph of y = 4x + 3, 
prove that the value of 4a - b - 2 is -5
-/
theorem point_on_linear_graph (a b : ℝ) : 
  b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end point_on_linear_graph_l1712_171254


namespace inverse_sqrt_difference_equals_sum_l1712_171247

theorem inverse_sqrt_difference_equals_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  1 / (Real.sqrt a - Real.sqrt b) = Real.sqrt a + Real.sqrt b :=
by sorry

end inverse_sqrt_difference_equals_sum_l1712_171247


namespace human_family_members_l1712_171211

/-- Represents the number of feet for each type of animal and the alien pet. -/
structure AnimalFeet where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Represents the number of heads for each type of animal and the alien pet. -/
structure AnimalHeads where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Calculates the total number of feet for all animals and the alien pet. -/
def totalAnimalFeet (af : AnimalFeet) : ℕ :=
  af.birds + af.dogs + af.cats + af.alien

/-- Calculates the total number of heads for all animals and the alien pet. -/
def totalAnimalHeads (ah : AnimalHeads) : ℕ :=
  ah.birds + ah.dogs + ah.cats + ah.alien

/-- Theorem stating the number of human family members. -/
theorem human_family_members :
  ∃ (h : ℕ),
    let af : AnimalFeet := ⟨7, 13, 74, 6⟩
    let ah : AnimalHeads := ⟨4, 3, 18, 1⟩
    totalAnimalFeet af + 2 * h = totalAnimalHeads ah + h + 108 ∧ h = 34 := by
  sorry

end human_family_members_l1712_171211


namespace supermarket_bread_count_l1712_171266

/-- Calculates the final number of loaves in a supermarket after sales and delivery -/
def final_loaves (initial : ℕ) (sold : ℕ) (delivered : ℕ) : ℕ :=
  initial - sold + delivered

/-- Theorem stating that given the specific numbers from the problem, 
    the final number of loaves is 2215 -/
theorem supermarket_bread_count : 
  final_loaves 2355 629 489 = 2215 := by
  sorry

end supermarket_bread_count_l1712_171266


namespace haley_candy_count_l1712_171201

/-- The number of candy pieces Haley has at the end -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that Haley's final candy count is 35 -/
theorem haley_candy_count :
  final_candy_count 33 17 19 = 35 := by
  sorry

end haley_candy_count_l1712_171201


namespace A_intersect_B_equals_open_2_closed_3_l1712_171217

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}

-- Define set B (domain of log(-x^2 + 6x - 8))
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- Theorem to prove
theorem A_intersect_B_equals_open_2_closed_3 : A ∩ B = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end A_intersect_B_equals_open_2_closed_3_l1712_171217


namespace factorization_equality_l1712_171261

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 4 * a * b + 2 * b^2 = 2 * (a - b)^2 := by
  sorry

end factorization_equality_l1712_171261


namespace min_value_on_line_l1712_171253

/-- The minimum value of 2^x + 4^y for points (x, y) on the line through (3, 0) and (1, 1) -/
theorem min_value_on_line : 
  ∀ (x y : ℝ), (x + 2*y = 3) → (2^x + 4^y ≥ 4 * Real.sqrt 2) ∧ 
  ∃ (x₀ y₀ : ℝ), (x₀ + 2*y₀ = 3) ∧ (2^x₀ + 4^y₀ = 4 * Real.sqrt 2) := by
  sorry

end min_value_on_line_l1712_171253


namespace min_top_managers_bound_l1712_171236

/-- Represents the structure of a company with its employees and order distribution system. -/
structure Company where
  total_employees : ℕ
  direct_connections : ℕ
  distribution_days : ℕ
  (total_employees_positive : total_employees > 0)
  (direct_connections_positive : direct_connections > 0)
  (distribution_days_positive : distribution_days > 0)

/-- Calculates the minimum number of top-level managers in the company. -/
def min_top_managers (c : Company) : ℕ :=
  ((c.total_employees - 1) / (c.direct_connections^(c.distribution_days + 1) - 1)) + 1

/-- Theorem stating that a company with 50,000 employees, 7 direct connections per employee, 
    and 4 distribution days has at least 28 top-level managers. -/
theorem min_top_managers_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.direct_connections = 7)
  (h3 : c.distribution_days = 4) :
  min_top_managers c ≥ 28 := by
  sorry

#eval min_top_managers ⟨50000, 7, 4, by norm_num, by norm_num, by norm_num⟩

end min_top_managers_bound_l1712_171236


namespace system_solution_l1712_171249

theorem system_solution (x y : ℝ) : 
  (x^3 * y + x * y^3 = 10 ∧ x^4 + y^4 = 17) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1)) :=
sorry

end system_solution_l1712_171249


namespace linear_functions_through_point_l1712_171212

theorem linear_functions_through_point :
  ∃ (x₀ y₀ : ℝ) (k b : Fin 10 → ℕ),
    (∀ i : Fin 10, 1 ≤ k i ∧ k i ≤ 20 ∧ 1 ≤ b i ∧ b i ≤ 20) ∧
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j ∧ b i ≠ b j) ∧
    (∀ i : Fin 10, y₀ = k i * x₀ + b i) := by
  sorry

end linear_functions_through_point_l1712_171212


namespace mixture_weight_l1712_171223

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 800 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  (((ratio_a : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_a +
   ((ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_b) / 1000 = 3.44 := by
  sorry

#check mixture_weight

end mixture_weight_l1712_171223


namespace soup_feeding_theorem_l1712_171207

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : Nat
  children : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) : Nat :=
  let cansForChildren := childrenFed / canCapacity.children
  let remainingCans := totalCans - cansForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given the conditions, 20 adults can be fed with the remaining soup -/
theorem soup_feeding_theorem (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 8 →
  childrenFed = 40 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end soup_feeding_theorem_l1712_171207


namespace different_color_chip_probability_l1712_171268

theorem different_color_chip_probability :
  let total_chips : ℕ := 7 + 5
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 :=
by sorry

end different_color_chip_probability_l1712_171268


namespace line_parallel_perpendicular_planes_l1712_171240

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_planes 
  (m : Line) (α β : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
by sorry

end line_parallel_perpendicular_planes_l1712_171240


namespace sqrt_eight_div_sqrt_two_equals_two_l1712_171257

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end sqrt_eight_div_sqrt_two_equals_two_l1712_171257


namespace strip_sum_unique_l1712_171202

def strip_sum (T : ℕ) : List ℕ → Prop
  | [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] =>
    a₁ = 2021 ∧ a₈ = 2021 ∧
    (∀ i ∈ [1, 2, 3, 4, 5, 6, 7], 
      (List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] i + 
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] (i+1) = T ∨
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] i + 
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] (i+1) = T+1)) ∧
    a₇ + a₈ = T
  | _ => False

theorem strip_sum_unique : 
  ∃! T, ∃ (l : List ℕ), strip_sum T l ∧ T = 4045 := by sorry

end strip_sum_unique_l1712_171202


namespace binary_1010_is_10_l1712_171259

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end binary_1010_is_10_l1712_171259


namespace parking_space_painted_sides_sum_l1712_171260

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

end parking_space_painted_sides_sum_l1712_171260


namespace unique_solution_for_equation_l1712_171228

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 3 :=
by
  -- The proof goes here
  sorry

end unique_solution_for_equation_l1712_171228


namespace earth_inhabitable_surface_l1712_171263

theorem earth_inhabitable_surface (exposed_land : ℚ) (inhabitable_land : ℚ) 
  (h1 : exposed_land = 3 / 8)
  (h2 : inhabitable_land = 2 / 3) :
  exposed_land * inhabitable_land = 1 / 4 := by
  sorry

end earth_inhabitable_surface_l1712_171263


namespace tangent_values_l1712_171234

/-- Two linear functions with parallel non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  hf : f = λ x => a * x + b
  hg : g = λ x => a * x + c
  ha : a ≠ 0

/-- The property that (f x)^2 is tangent to -12(g x) -/
def is_tangent_f_g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -12 * (p.g x)

/-- The main theorem -/
theorem tangent_values (p : ParallelLinearFunctions) 
  (h : is_tangent_f_g p) :
  ∃ A : Set ℝ, A = {0, 12} ∧ 
  ∀ a : ℝ, a ∈ A ↔ ∃! x, (p.g x)^2 = a * (p.f x) := by
  sorry

end tangent_values_l1712_171234


namespace greg_travel_distance_l1712_171258

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let workplace_to_market : ℝ := 30  -- Distance in miles
  let market_to_home_time : ℝ := 0.5  -- Time in hours
  let market_to_home_speed : ℝ := 20  -- Speed in miles per hour
  let market_to_home : ℝ := market_to_home_speed * market_to_home_time
  let total_distance : ℝ := workplace_to_market + market_to_home
  total_distance = 40 := by
sorry


end greg_travel_distance_l1712_171258


namespace circle_tangent_sum_radii_l1712_171231

theorem circle_tangent_sum_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end circle_tangent_sum_radii_l1712_171231


namespace not_all_equilateral_triangles_have_same_perimeter_l1712_171250

-- Define an equilateral triangle
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Properties of equilateral triangles
def EquilateralTriangle.isEquiangular (t : EquilateralTriangle) : Prop :=
  -- All angles are 60 degrees
  true

def EquilateralTriangle.isIsosceles (t : EquilateralTriangle) : Prop :=
  -- At least two sides are equal (all sides are equal in this case)
  true

def EquilateralTriangle.isRegularPolygon (t : EquilateralTriangle) : Prop :=
  -- All sides equal and all angles equal
  true

def EquilateralTriangle.isSimilarTo (t1 t2 : EquilateralTriangle) : Prop :=
  -- All equilateral triangles are similar
  true

def EquilateralTriangle.perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.sideLength

-- Theorem to prove
theorem not_all_equilateral_triangles_have_same_perimeter :
  ∃ t1 t2 : EquilateralTriangle, t1.perimeter ≠ t2.perimeter ∧
    t1.isEquiangular ∧ t2.isEquiangular ∧
    t1.isIsosceles ∧ t2.isIsosceles ∧
    t1.isRegularPolygon ∧ t2.isRegularPolygon ∧
    t1.isSimilarTo t2 :=
  sorry

end not_all_equilateral_triangles_have_same_perimeter_l1712_171250


namespace one_more_square_possible_l1712_171298

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

end one_more_square_possible_l1712_171298


namespace constant_function_proof_l1712_171245

theorem constant_function_proof (g : ℝ → ℝ) 
  (h1 : ∃ x, g x ≠ 0)
  (h2 : ∀ a b : ℝ, g (a + b) + g (a - b) = g a + g b) :
  ∃ k : ℝ, ∀ x : ℝ, g x = k := by
  sorry

end constant_function_proof_l1712_171245


namespace existence_of_same_sum_opposite_signs_l1712_171251

theorem existence_of_same_sum_opposite_signs :
  ∃ (y : ℝ) (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁^4 + x₁^5 = y ∧ x₂^4 + x₂^5 = y :=
by sorry

end existence_of_same_sum_opposite_signs_l1712_171251


namespace sum_of_areas_is_1168_l1712_171271

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

end sum_of_areas_is_1168_l1712_171271


namespace rectangle_triangle_area_ratio_l1712_171256

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

end rectangle_triangle_area_ratio_l1712_171256


namespace problem_statement_l1712_171296

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_property (f : ℝ → ℝ) : Prop := ∀ x, f (2 + x) = f (-x)

theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f)
  (h_f1 : f 1 = 2) : 
  f 2018 + f 2019 = -2 := by
sorry

end problem_statement_l1712_171296


namespace gravel_path_cost_l1712_171233

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) : 
  plot_length = 110 →
  plot_width = 65 →
  path_width = 2.5 →
  cost_per_sqm = 0.7 →
  ((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - plot_length * plot_width) * cost_per_sqm = 630 := by
  sorry

end gravel_path_cost_l1712_171233


namespace min_reciprocal_sum_l1712_171286

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 := by sorry

end min_reciprocal_sum_l1712_171286


namespace triangle_properties_l1712_171241

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧
  t.c = 2 ∧
  t.a + t.b + t.c = 2 * Real.sqrt 3 + 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C : ℝ) = 2 * Real.sqrt 3 / 3 := by
  sorry


end triangle_properties_l1712_171241


namespace books_read_theorem_l1712_171295

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

end books_read_theorem_l1712_171295


namespace set_operations_l1712_171225

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  (A ∩ B = {x | -2 < x ∧ x < 3}) ∧
  (Set.univ \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  ((Set.univ \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by
  sorry

end set_operations_l1712_171225


namespace jeff_tennis_time_l1712_171230

/-- Proves that Jeff played tennis for 2 hours given the conditions -/
theorem jeff_tennis_time (
  points_per_match : ℕ) 
  (minutes_per_point : ℕ) 
  (matches_won : ℕ) 
  (h1 : points_per_match = 8)
  (h2 : minutes_per_point = 5)
  (h3 : matches_won = 3)
  : (points_per_match * matches_won * minutes_per_point) / 60 = 2 := by
  sorry

end jeff_tennis_time_l1712_171230


namespace leftover_coins_value_l1712_171291

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

end leftover_coins_value_l1712_171291


namespace parallel_vectors_fraction_value_l1712_171274

theorem parallel_vectors_fraction_value (α : ℝ) :
  let a : ℝ × ℝ := (Real.sin α, Real.cos α - 2 * Real.sin α)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -5/3 := by
  sorry

end parallel_vectors_fraction_value_l1712_171274


namespace rodney_lifts_118_l1712_171280

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


end rodney_lifts_118_l1712_171280


namespace infinite_sum_equals_five_twentyfourths_l1712_171220

/-- The infinite sum of n / (n^4 - 4n^2 + 8) from n=1 to infinity equals 5/24 -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5/24 := by
  sorry

end infinite_sum_equals_five_twentyfourths_l1712_171220


namespace temperature_problem_l1712_171244

def temperature_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 1) = x n + x (n + 2)

theorem temperature_problem (x : ℕ → ℤ) 
  (h_seq : temperature_sequence x)
  (h_3 : x 3 = 5)
  (h_31 : x 31 = 2) :
  x 25 = -3 := by
  sorry

end temperature_problem_l1712_171244


namespace min_value_a_plus_3b_l1712_171276

theorem min_value_a_plus_3b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_eq : a + 3*b = 1/a + 3/b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 1/x + 3/y → a + 3*b ≤ x + 3*y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 1/x + 3/y ∧ x + 3*y = 4 :=
sorry

end min_value_a_plus_3b_l1712_171276


namespace prob_even_product_three_dice_l1712_171243

-- Define a six-sided die
def SixSidedDie : Type := Fin 6

-- Define the probability of rolling an even number on a single die
def probEvenOnDie : ℚ := 1/2

-- Define the probability of rolling at least one even number on two dice
def probAtLeastOneEvenOnTwoDice : ℚ := 1 - (1 - probEvenOnDie) ^ 2

-- The main theorem
theorem prob_even_product_three_dice :
  probAtLeastOneEvenOnTwoDice = 3/4 := by sorry

end prob_even_product_three_dice_l1712_171243


namespace inverse_proportion_order_l1712_171284

/-- Given points A(-2,a), B(-1,b), and C(3,c) on the graph of y = 4/x, prove b < a < c -/
theorem inverse_proportion_order (a b c : ℝ) : 
  ((-2 : ℝ) * a = 4) → ((-1 : ℝ) * b = 4) → ((3 : ℝ) * c = 4) → b < a ∧ a < c := by
  sorry

end inverse_proportion_order_l1712_171284


namespace function_inequality_l1712_171252

open Real

theorem function_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : -1 < log x₁) (h₄ : -1 < log x₂) : 
  let f := fun (x : ℝ) => x * log x
  x₁ * f x₁ + x₂ * f x₂ > 2 * x₂ * f x₁ := by
sorry

end function_inequality_l1712_171252


namespace least_common_multiple_problem_l1712_171221

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 24 = 0 ∧ n % 32 = 0 ∧ n % 36 = 0 ∧ n % 54 = 0

theorem least_common_multiple_problem : 
  ∃! x : ℕ, (is_divisible_by_all (856 + x) ∧ 
    ∀ y : ℕ, y < x → ¬is_divisible_by_all (856 + y)) ∧ 
  x = 8 := by
  sorry

end least_common_multiple_problem_l1712_171221


namespace four_n_plus_two_not_in_M_l1712_171242

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- Theorem stating that 4n+2 is not in M for any integer n -/
theorem four_n_plus_two_not_in_M (n : ℤ) : (4*n + 2) ∉ M := by
  sorry

end four_n_plus_two_not_in_M_l1712_171242


namespace g_equals_zero_at_negative_one_l1712_171205

/-- The function g(x) as defined in the problem -/
def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

/-- Theorem stating that g(-1) = 0 when r = 14 -/
theorem g_equals_zero_at_negative_one (r : ℝ) : g (-1) r = 0 ↔ r = 14 := by sorry

end g_equals_zero_at_negative_one_l1712_171205


namespace value_of_a_l1712_171246

theorem value_of_a (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) : 
  a = 1 ∨ a = 2 :=
by sorry

end value_of_a_l1712_171246


namespace parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l1712_171200

-- Define a straight line in a coordinate plane
structure Line where
  slope : Option ℝ
  point : ℝ × ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

-- Theorem 1: If the slopes of two lines do not exist, then the two lines are parallel
theorem parallel_non_existent_slopes (l1 l2 : Line) :
  l1.slope = none ∧ l2.slope = none → parallel l1 l2 := by sorry

-- Theorem 2: If one of two lines has a non-existent slope and the other has a slope, 
-- then the two lines intersect
theorem intersect_one_non_existent_slope (l1 l2 : Line) :
  (l1.slope = none ∧ l2.slope ≠ none) ∨ (l1.slope ≠ none ∧ l2.slope = none) 
  → intersect l1 l2 := by sorry

-- Theorem 3: The equation of the line passing through any two different points 
-- P₁(x₁, y₁), P₂(x₂, y₂) is (x₂-x₁)(y-y₁)=(y₂-y₁)(x-x₁)
theorem line_equation_through_two_points (P1 P2 : ℝ × ℝ) (x y : ℝ) :
  P1 ≠ P2 → (P2.1 - P1.1) * (y - P1.2) = (P2.2 - P1.2) * (x - P1.1) := by sorry

end parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l1712_171200


namespace min_sum_squares_l1712_171279

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

end min_sum_squares_l1712_171279


namespace brendas_blisters_l1712_171208

/-- The number of blisters Brenda has on each arm -/
def blisters_per_arm : ℕ := 60

/-- The number of blisters Brenda has on the rest of her body -/
def blisters_on_body : ℕ := 80

/-- The number of arms Brenda has -/
def number_of_arms : ℕ := 2

/-- The total number of blisters Brenda has -/
def total_blisters : ℕ := blisters_per_arm * number_of_arms + blisters_on_body

theorem brendas_blisters : total_blisters = 200 := by
  sorry

end brendas_blisters_l1712_171208


namespace no_rational_roots_l1712_171277

theorem no_rational_roots :
  ∀ (q : ℚ), 3 * q^4 - 2 * q^3 - 15 * q^2 + 6 * q + 3 ≠ 0 := by
  sorry

end no_rational_roots_l1712_171277


namespace inequality_proof_l1712_171273

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) ≤ 
  3 / 8 * (1 / a + 1 / b + 1 / c + 1 / d)^2 := by
sorry

end inequality_proof_l1712_171273


namespace min_values_ab_l1712_171283

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (ab ≥ 8) ∧ (a + b ≥ 3 + 2 * Real.sqrt 2) := by
  sorry

end min_values_ab_l1712_171283


namespace bookshelf_cost_price_l1712_171226

/-- The cost price of a bookshelf sold at a loss and would have made a profit with additional revenue -/
theorem bookshelf_cost_price (C : ℝ) : C = 1071.43 :=
  let SP := 0.76 * C
  have h1 : SP = 0.76 * C := by rfl
  have h2 : SP + 450 = 1.18 * C := by sorry
  sorry

end bookshelf_cost_price_l1712_171226


namespace soccer_tournament_theorem_l1712_171289

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

end soccer_tournament_theorem_l1712_171289


namespace geometric_sequence_common_ratio_l1712_171213

/-- 
Given a geometric sequence {a_n} where a₁ = -1 and a₄ = 8,
prove that the common ratio is -2.
-/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = -1 →
  a 4 = 8 →
  a 2 / a 1 = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l1712_171213


namespace polynomial_evaluation_l1712_171237

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 9 = 0 ∧ 
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = (65 + 81 * Real.sqrt 5) / 2 := by
  sorry

end polynomial_evaluation_l1712_171237


namespace scarves_per_box_l1712_171281

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_pieces : ℕ) : 
  num_boxes = 3 → 
  mittens_per_box = 4 → 
  total_pieces = 21 → 
  (total_pieces - num_boxes * mittens_per_box) / num_boxes = 3 :=
by sorry

end scarves_per_box_l1712_171281


namespace min_rods_eq_2n_minus_2_l1712_171288

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

end min_rods_eq_2n_minus_2_l1712_171288


namespace helen_cookies_yesterday_l1712_171262

def cookies_this_morning : ℕ := 270
def cookies_day_before_yesterday : ℕ := 419
def cookies_till_last_night : ℕ := 450

theorem helen_cookies_yesterday :
  cookies_day_before_yesterday + cookies_this_morning - cookies_till_last_night = 239 := by
  sorry

end helen_cookies_yesterday_l1712_171262


namespace graduating_students_average_score_l1712_171299

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


end graduating_students_average_score_l1712_171299


namespace p_true_and_q_false_l1712_171239

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end p_true_and_q_false_l1712_171239


namespace gcf_lcm_sum_l1712_171287

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 12 (Nat.gcd 18 30)) → 
  (B = Nat.lcm 12 (Nat.lcm 18 30)) → 
  2 * A + B = 192 :=
by
  sorry

end gcf_lcm_sum_l1712_171287


namespace candy_left_is_49_l1712_171264

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some to his sister -/
def candy_left : ℕ :=
  let kit_kats := 5
  let hershey_kisses := 3 * kit_kats
  let nerds := 8
  let lollipops := 11
  let baby_ruths := 10
  let reese_cups := baby_ruths / 2
  let total_candy := kit_kats + hershey_kisses + nerds + lollipops + baby_ruths + reese_cups
  let given_away := 5
  total_candy - given_away

theorem candy_left_is_49 : candy_left = 49 := by
  sorry

end candy_left_is_49_l1712_171264


namespace pulley_centers_distance_l1712_171278

/-- Given two circular pulleys with an uncrossed belt, prove the distance between their centers. -/
theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 14)
  (h₂ : r₂ = 4)
  (h₃ : contact_distance = 24) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 26 := by
  sorry

end pulley_centers_distance_l1712_171278


namespace oil_per_cylinder_l1712_171285

theorem oil_per_cylinder (cylinders : ℕ) (oil_added : ℕ) (oil_needed : ℕ) :
  cylinders = 6 →
  oil_added = 16 →
  oil_needed = 32 →
  (oil_added + oil_needed) / cylinders = 8 := by
  sorry

end oil_per_cylinder_l1712_171285


namespace two_numbers_sum_and_difference_l1712_171293

theorem two_numbers_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6) : 
  x = 18 ∧ y = 12 := by
sorry

end two_numbers_sum_and_difference_l1712_171293


namespace park_visit_cost_family_park_visit_cost_l1712_171265

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

end park_visit_cost_family_park_visit_cost_l1712_171265


namespace janet_needs_775_l1712_171248

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (months_advance : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * months_advance + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 :
  additional_money_needed 2225 1250 2 500 = 775 :=
by sorry

end janet_needs_775_l1712_171248


namespace problem_solution_l1712_171203

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end problem_solution_l1712_171203


namespace tshirt_production_l1712_171204

/-- The number of minutes in an hour -/
def minutesPerHour : ℕ := 60

/-- The rate of t-shirt production in the first hour (minutes per t-shirt) -/
def rateFirstHour : ℕ := 12

/-- The rate of t-shirt production in the second hour (minutes per t-shirt) -/
def rateSecondHour : ℕ := 6

/-- The total number of t-shirts produced in two hours -/
def totalTShirts : ℕ := minutesPerHour / rateFirstHour + minutesPerHour / rateSecondHour

theorem tshirt_production : totalTShirts = 15 := by
  sorry

end tshirt_production_l1712_171204


namespace geometric_sequence_ratio_l1712_171269

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

end geometric_sequence_ratio_l1712_171269


namespace polar_coordinates_of_point_l1712_171232

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1/2 ∧ y = -Real.sqrt 3 / 2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = Real.arccos (x / ρ) + (if y < 0 then 2 * Real.pi else 0) →
  ρ = 1 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end polar_coordinates_of_point_l1712_171232


namespace down_payment_calculation_l1712_171216

/-- Given a loan with the following conditions:
  * The loan has 0% interest
  * The loan is to be paid back in 5 years
  * Monthly payments are $600.00
  * The total loan amount (including down payment) is $46,000
  This theorem proves that the down payment is $10,000 -/
theorem down_payment_calculation (loan_amount : ℝ) (years : ℕ) (monthly_payment : ℝ) :
  loan_amount = 46000 ∧ 
  years = 5 ∧ 
  monthly_payment = 600 →
  loan_amount - (years * 12 : ℝ) * monthly_payment = 10000 :=
by sorry

end down_payment_calculation_l1712_171216


namespace max_value_quadratic_function_l1712_171227

theorem max_value_quadratic_function :
  let f : ℝ → ℝ := fun x ↦ -x^2 + 2*x + 1
  ∃ (m : ℝ), m = 2 ∧ ∀ x, f x ≤ m :=
by
  sorry

end max_value_quadratic_function_l1712_171227


namespace factor_implies_q_value_l1712_171235

theorem factor_implies_q_value (m q : ℤ) : 
  (∃ k : ℤ, m^2 - q*m - 24 = (m - 8) * k) → q = 5 := by
sorry

end factor_implies_q_value_l1712_171235


namespace sum_of_squares_l1712_171255

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (power_eq : a^7 + b^7 + c^7 = a^9 + b^9 + c^9) :
  a^2 + b^2 + c^2 = 14/9 := by
  sorry

end sum_of_squares_l1712_171255


namespace beads_per_necklace_l1712_171224

theorem beads_per_necklace (total_beads : ℕ) (total_necklaces : ℕ) 
  (h1 : total_beads = 20) 
  (h2 : total_necklaces = 4) : 
  total_beads / total_necklaces = 5 :=
by sorry

end beads_per_necklace_l1712_171224


namespace reflection_sum_coordinates_l1712_171229

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The sum of coordinates of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates :
  let C : Point := { x := 3, y := 8 }
  let D : Point := reflect_over_y_axis C
  sum_of_coordinates C D = 16 := by
  sorry

end reflection_sum_coordinates_l1712_171229


namespace no_perfect_squares_l1712_171270

theorem no_perfect_squares (x y : ℕ+) : 
  ¬(∃ (a b : ℕ), (x^2 + y + 2 : ℕ) = a^2 ∧ (y^2 + 4*x : ℕ) = b^2) :=
sorry

end no_perfect_squares_l1712_171270


namespace blue_tile_probability_l1712_171294

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

end blue_tile_probability_l1712_171294


namespace pentagon_area_given_equal_perimeter_square_l1712_171282

theorem pentagon_area_given_equal_perimeter_square (s : ℝ) (p : ℝ) : 
  s > 0 →
  p > 0 →
  4 * s = 5 * p →
  s^2 = 16 →
  abs ((5 * p^2 * Real.tan (3 * Real.pi / 10)) / 4 - 15.26) < 0.01 :=
by
  sorry

end pentagon_area_given_equal_perimeter_square_l1712_171282
