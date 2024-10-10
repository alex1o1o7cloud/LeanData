import Mathlib

namespace prakash_copies_five_pages_l3432_343253

/-- Represents the number of pages a person can copy in a given time -/
structure CopyingRate where
  pages : ℕ
  hours : ℕ

/-- Subash's copying rate -/
def subash_rate : CopyingRate := ⟨50, 10⟩

/-- Combined copying rate of Subash and Prakash -/
def combined_rate : CopyingRate := ⟨300, 40⟩

/-- Calculate the number of pages Prakash can copy in 2 hours -/
def prakash_pages : ℕ :=
  let subash_40_hours := (subash_rate.pages * combined_rate.hours) / subash_rate.hours
  let prakash_40_hours := combined_rate.pages - subash_40_hours
  (prakash_40_hours * 2) / combined_rate.hours

theorem prakash_copies_five_pages : prakash_pages = 5 := by
  sorry

end prakash_copies_five_pages_l3432_343253


namespace min_value_sum_l3432_343239

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - x - 2 * y = 0) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_sum_l3432_343239


namespace third_grade_students_l3432_343256

/-- The number of story books to be distributed -/
def total_books : ℕ := 90

/-- Proves that the number of third-grade students is 60 -/
theorem third_grade_students :
  ∃ n : ℕ, n > 0 ∧ n < total_books ∧ total_books - n = n / 2 :=
by
  -- The proof goes here
  sorry

end third_grade_students_l3432_343256


namespace mean_temperature_l3432_343242

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 0]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -10 / 7 := by
  sorry

end mean_temperature_l3432_343242


namespace derivative_limit_equality_l3432_343211

theorem derivative_limit_equality (f : ℝ → ℝ) (h : HasDerivAt f (-2) 2) :
  Filter.Tendsto (fun x => (f x - f 2) / (x - 2)) (Filter.atTop.comap (fun x => |x - 2|)) (nhds (-2)) := by
  sorry

end derivative_limit_equality_l3432_343211


namespace ice_cost_theorem_l3432_343238

/-- The cost of ice for enterprise A given the specified conditions -/
theorem ice_cost_theorem 
  (a : ℝ) -- Price of ice from B in rubles per ton
  (p : ℝ) -- Transportation cost in rubles per ton-kilometer
  (n : ℝ) -- Ice melting rate (n/1000 of mass per kilometer)
  (s : ℝ) -- Distance from B to C through A in kilometers
  (h1 : 0 < a) -- Price is positive
  (h2 : 0 < p) -- Transportation cost is positive
  (h3 : 0 < n) -- Melting rate is positive
  (h4 : 0 < s) -- Distance is positive
  (h5 : n * s < 2000) -- Ensure denominator is positive
  : ∃ (z : ℝ), z = (2.5 * a + p * s) * 1000 / (2000 - n * s) ∧ 
    z * (2 - n * s / 1000) = 2.5 * a + p * s := by
  sorry

end ice_cost_theorem_l3432_343238


namespace roberto_outfits_l3432_343252

def trousers : ℕ := 5
def shirts : ℕ := 6
def jackets : ℕ := 3
def ties : ℕ := 2

theorem roberto_outfits : trousers * shirts * jackets * ties = 180 := by
  sorry

end roberto_outfits_l3432_343252


namespace hyperbola_foci_distance_l3432_343295

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ), x * y = 4 →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧
    (Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 8) :=
by sorry


end hyperbola_foci_distance_l3432_343295


namespace expression_equals_three_l3432_343222

theorem expression_equals_three :
  (-1)^2023 + Real.sqrt 9 - π^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

end expression_equals_three_l3432_343222


namespace value_of_a_minus_b_l3432_343254

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2025)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = 1515 := by
  sorry

end value_of_a_minus_b_l3432_343254


namespace initial_birds_count_l3432_343273

theorem initial_birds_count (B : ℕ) : 
  (B + 4 - 3 + 6 = 12) → B = 5 := by
sorry

end initial_birds_count_l3432_343273


namespace triangle_side_length_l3432_343243

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if b = 6, c = 4, and A = 2B, then a = 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → c = 4 → A = 2 * B → 
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  a = 2 * Real.sqrt 15 := by
sorry

end triangle_side_length_l3432_343243


namespace rhombus_perimeter_l3432_343261

/-- The perimeter of a rhombus with diagonals measuring 20 feet and 16 feet is 8√41 feet. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 20) (h₂ : d₂ = 16) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 8 * Real.sqrt 41 := by
  sorry

end rhombus_perimeter_l3432_343261


namespace complex_distance_range_l3432_343284

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 1) :
  0 ≤ Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ∧
  Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ≤ 3 :=
sorry

end complex_distance_range_l3432_343284


namespace geometric_sequence_condition_l3432_343294

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_condition
  (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 3 * a 5 = 16 → a 4 = 4) ∧ 
  ¬(a 4 = 4 → a 3 * a 5 = 16) :=
sorry

end geometric_sequence_condition_l3432_343294


namespace product_divisibility_l3432_343208

theorem product_divisibility : ∃ k : ℕ, 86 * 87 * 88 * 89 * 90 * 91 * 92 = 7 * k := by
  sorry

#check product_divisibility

end product_divisibility_l3432_343208


namespace marble_selection_with_blue_l3432_343266

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 3
def selection_size : ℕ := 4

theorem marble_selection_with_blue (total_marbles red_marbles blue_marbles green_marbles selection_size : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : total_marbles = 10)
  (h3 : red_marbles = 3)
  (h4 : blue_marbles = 4)
  (h5 : green_marbles = 3)
  (h6 : selection_size = 4) :
  (Nat.choose total_marbles selection_size) - (Nat.choose (total_marbles - blue_marbles) selection_size) = 195 :=
by sorry

end marble_selection_with_blue_l3432_343266


namespace fraction_power_product_l3432_343215

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end fraction_power_product_l3432_343215


namespace union_of_A_and_B_l3432_343221

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1} := by
  sorry

end union_of_A_and_B_l3432_343221


namespace function_range_equivalence_l3432_343282

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_range_equivalence (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ f a x > 0 ∧ f a y < 0) ↔
  a ∈ Set.Ioo (-1) (-1/3) :=
sorry

end function_range_equivalence_l3432_343282


namespace canoe_downstream_speed_l3432_343210

/-- Given a canoe's upstream speed and the stream speed, calculates the downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  2 * upstream_speed + 3 * stream_speed

/-- Theorem stating that for a canoe with upstream speed 8 km/hr and stream speed 2 km/hr, 
    the downstream speed is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed := 8
  let stream_speed := 2
  downstream_speed upstream_speed stream_speed = 12 := by
  sorry

end canoe_downstream_speed_l3432_343210


namespace prime_triple_product_sum_l3432_343286

theorem prime_triple_product_sum : 
  ∀ x y z : ℕ, 
    Prime x → Prime y → Prime z →
    x * y * z = 5 * (x + y + z) →
    ((x = 2 ∧ y = 5 ∧ z = 7) ∨ (x = 5 ∧ y = 2 ∧ z = 7)) :=
by
  sorry

end prime_triple_product_sum_l3432_343286


namespace function_bound_l3432_343271

/-- Given real-valued functions f and g defined on the real line,
    if f(x + y) + f(x - y) = 2f(x)g(y) for all x and y,
    f is not identically zero, and |f(x)| ≤ 1 for all x,
    then |g(x)| ≤ 1 for all x. -/
theorem function_bound (f g : ℝ → ℝ)
    (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
    (h2 : ∃ x, f x ≠ 0)
    (h3 : ∀ x, |f x| ≤ 1) :
    ∀ x, |g x| ≤ 1 := by
  sorry

end function_bound_l3432_343271


namespace electricity_pricing_l3432_343293

/-- Represents the electricity pricing problem -/
theorem electricity_pricing
  (a : ℝ) -- annual electricity consumption in kilowatt-hours
  (x : ℝ) -- new electricity price per kilowatt-hour
  (h1 : 0 < a) -- assumption that consumption is positive
  (h2 : 0.55 ≤ x ∧ x ≤ 0.75) -- new price range
  : ((0.2 * a / (x - 0.40) + a) * (x - 0.30) ≥ 0.60 * a) ↔ (x ≥ 0.60) :=
by sorry

end electricity_pricing_l3432_343293


namespace rearrangements_without_substring_l3432_343285

def word : String := "HMMTHMMT"

def total_permutations : ℕ := 420

def permutations_with_substring : ℕ := 60

theorem rearrangements_without_substring :
  (total_permutations - permutations_with_substring + 1 : ℕ) = 361 := by sorry

end rearrangements_without_substring_l3432_343285


namespace divides_power_difference_l3432_343214

theorem divides_power_difference (n : ℕ) : n ∣ 2^(2*n.factorial) - 2^(n.factorial) := by
  sorry

end divides_power_difference_l3432_343214


namespace compound_molecular_weight_l3432_343291

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := carbon_weight * carbon_count + oxygen_weight * oxygen_count

theorem compound_molecular_weight :
  molecular_weight = 28.01 := by sorry

end compound_molecular_weight_l3432_343291


namespace multiple_sum_properties_l3432_343279

theorem multiple_sum_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, a + b = 2 * n) ∧ 
  (¬ ∀ p : ℤ, a + b = 6 * p) ∧
  (¬ ∀ q : ℤ, a + b = 8 * q) ∧
  (∃ r s : ℤ, a = 6 * r ∧ b = 8 * s ∧ ¬ ∃ t : ℤ, a + b = 8 * t) :=
by sorry

end multiple_sum_properties_l3432_343279


namespace shirt_price_theorem_l3432_343236

/-- The price of a shirt when the total cost of the shirt and a coat is $600,
    and the shirt costs one-third the price of the coat. -/
def shirt_price : ℝ := 150

/-- The price of a coat when the total cost of the shirt and the coat is $600,
    and the shirt costs one-third the price of the coat. -/
def coat_price : ℝ := 3 * shirt_price

theorem shirt_price_theorem :
  shirt_price + coat_price = 600 ∧ shirt_price = (1/3) * coat_price →
  shirt_price = 150 := by
sorry

end shirt_price_theorem_l3432_343236


namespace bugs_meet_time_l3432_343263

/-- The time (in minutes) it takes for two bugs to meet again at the starting point,
    given they start on two tangent circles with radii 7 and 3 inches,
    crawling at speeds of 4π and 3π inches per minute respectively. -/
def meeting_time : ℝ :=
  let r₁ : ℝ := 7  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let v₁ : ℝ := 4 * Real.pi  -- speed of bug on larger circle
  let v₂ : ℝ := 3 * Real.pi  -- speed of bug on smaller circle
  let t₁ : ℝ := (2 * Real.pi * r₁) / v₁  -- time for full circle on larger circle
  let t₂ : ℝ := (2 * Real.pi * r₂) / v₂  -- time for full circle on smaller circle
  14  -- the actual meeting time

theorem bugs_meet_time :
  meeting_time = 14 := by sorry

end bugs_meet_time_l3432_343263


namespace new_apples_grown_l3432_343257

theorem new_apples_grown (initial_apples picked_apples current_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : picked_apples = 7)
  (h3 : current_apples = 6) :
  current_apples - (initial_apples - picked_apples) = 2 :=
by sorry

end new_apples_grown_l3432_343257


namespace inequality_proof_l3432_343281

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end inequality_proof_l3432_343281


namespace inequality_solution_l3432_343224

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x ≤ -3 * (1 + x)) ↔ -4 ≤ x ∧ x ≤ -3/2 := by
  sorry

end inequality_solution_l3432_343224


namespace distributive_law_l3432_343217

theorem distributive_law (a b c : ℝ) : (a + b) * c = a * c + b * c := by
  sorry

end distributive_law_l3432_343217


namespace inscribed_cylinder_radius_l3432_343207

/-- Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular
cylinder whose height is twice its radius, the radius of the cylinder is 56/15. -/
theorem inscribed_cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) (cylinder_radius : ℝ) :
  cone_diameter = 14 →
  cone_altitude = 16 →
  (∃ (cylinder_height : ℝ), cylinder_height = 2 * cylinder_radius) →
  cylinder_radius = 56 / 15 := by
  sorry

end inscribed_cylinder_radius_l3432_343207


namespace three_solutions_condition_l3432_343201

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((x - 5) * Real.sin a - (y - 5) * Real.cos a = 0) ∧
  (((x + 1)^2 + (y + 1)^2 - 4) * ((x + 1)^2 + (y + 1)^2 - 16) = 0)

-- Define the condition for three solutions
def has_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system x₁ y₁ a ∧ system x₂ y₂ a ∧ system x₃ y₃ a ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    ∀ (x y : ℝ), system x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃)

-- Theorem statement
theorem three_solutions_condition (a : ℝ) :
  has_three_solutions a ↔ ∃ (n : ℤ), a = π/4 + Real.arcsin (Real.sqrt 2 / 6) + n * π ∨
                                     a = π/4 - Real.arcsin (Real.sqrt 2 / 6) + n * π :=
sorry

end three_solutions_condition_l3432_343201


namespace vasya_always_wins_l3432_343228

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  coloredPoints : Point → Option Color

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a winning condition is met -/
def isWinningState (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1 = some c ∧
    state.coloredPoints p2 = some c ∧
    state.coloredPoints p3 = some c

/-- The main theorem stating that Vasya always has a winning strategy -/
theorem vasya_always_wins :
  ∀ (initialState : GameState),
  ∃ (finalState : GameState),
    (∀ p, Option.isSome (finalState.coloredPoints p)) ∧
    isWinningState finalState :=
sorry

end vasya_always_wins_l3432_343228


namespace halfway_between_fractions_average_of_fractions_l3432_343275

theorem halfway_between_fractions :
  (1/8 : ℚ) + (1/10 : ℚ) = (9/40 : ℚ) :=
by sorry

theorem average_of_fractions :
  ((1/8 : ℚ) + (1/10 : ℚ)) / 2 = (9/80 : ℚ) :=
by sorry

end halfway_between_fractions_average_of_fractions_l3432_343275


namespace hours_per_day_l3432_343265

/-- Given that there are 8760 hours in a year and 365 days in a year,
    prove that there are 24 hours in a day. -/
theorem hours_per_day :
  let hours_per_year : ℕ := 8760
  let days_per_year : ℕ := 365
  (hours_per_year / days_per_year : ℚ) = 24 := by
sorry

end hours_per_day_l3432_343265


namespace vector_calculation_l3432_343200

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem vector_calculation (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (5 * a.1 - 3 * (b m).1, 5 * a.2 - 3 * (b m).2) = (-7, -16) := by
  sorry

end vector_calculation_l3432_343200


namespace randy_cheese_purchase_l3432_343276

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices : ℕ := 12

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices : ℕ := 28

/-- The smallest number of slices of each type that Randy could have bought -/
def smallest_equal_slices : ℕ := 84

theorem randy_cheese_purchase :
  smallest_equal_slices = Nat.lcm cheddar_slices swiss_slices ∧
  smallest_equal_slices % cheddar_slices = 0 ∧
  smallest_equal_slices % swiss_slices = 0 ∧
  ∀ n : ℕ, (n % cheddar_slices = 0 ∧ n % swiss_slices = 0) → n ≥ smallest_equal_slices := by
  sorry

end randy_cheese_purchase_l3432_343276


namespace power_multiplication_l3432_343297

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l3432_343297


namespace selling_price_correct_l3432_343237

/-- Calculates the selling price of a television after applying discounts -/
def selling_price (a : ℝ) : ℝ :=
  0.9 * (a - 100)

/-- Theorem stating that the selling price function correctly applies the discounts -/
theorem selling_price_correct (a : ℝ) : 
  selling_price a = 0.9 * (a - 100) := by
  sorry

end selling_price_correct_l3432_343237


namespace smallest_sum_of_sequence_l3432_343240

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 →
  (∃ d : ℤ, Z - Y = Y - X) →  -- arithmetic sequence condition
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →  -- geometric sequence condition
  (Z : ℚ) / Y = 7 / 4 →
  X + Y + Z + W ≥ 97 :=
sorry

end smallest_sum_of_sequence_l3432_343240


namespace james_weekly_beats_l3432_343202

/-- The number of beats heard in a week given a music speed and daily listening time -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) : ℕ :=
  beats_per_minute * (hours_per_day * 60) * 7

/-- Theorem: James hears 168,000 beats per week -/
theorem james_weekly_beats :
  beats_per_week 200 2 = 168000 := by
  sorry

end james_weekly_beats_l3432_343202


namespace function_transformation_l3432_343289

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = 3 * y + 4) →
  f x = 3 * x + 1 := by
  sorry

end function_transformation_l3432_343289


namespace blood_donation_selection_l3432_343213

theorem blood_donation_selection (m n k : ℕ) (hm : m = 3) (hn : n = 6) (hk : k = 5) :
  (Nat.choose (m + n) k) - (Nat.choose n k) = 120 := by
  sorry

end blood_donation_selection_l3432_343213


namespace secretary_work_time_l3432_343235

theorem secretary_work_time 
  (ratio : Fin 3 → ℕ)
  (total_time : ℕ) :
  ratio 0 = 2 →
  ratio 1 = 3 →
  ratio 2 = 5 →
  total_time = 110 →
  (ratio 0 + ratio 1 + ratio 2) * (total_time / (ratio 0 + ratio 1 + ratio 2)) = total_time →
  ratio 2 * (total_time / (ratio 0 + ratio 1 + ratio 2)) = 55 :=
by sorry

end secretary_work_time_l3432_343235


namespace geography_history_difference_l3432_343250

/-- Represents the number of pages in each textbook --/
structure TextbookPages where
  history : ℕ
  geography : ℕ
  math : ℕ
  science : ℕ

/-- Conditions for Suzanna's textbooks --/
def suzanna_textbooks (t : TextbookPages) : Prop :=
  t.history = 160 ∧
  t.geography > t.history ∧
  t.math = (t.history + t.geography) / 2 ∧
  t.science = 2 * t.history ∧
  t.history + t.geography + t.math + t.science = 905

/-- Theorem stating the difference in pages between geography and history textbooks --/
theorem geography_history_difference (t : TextbookPages) 
  (h : suzanna_textbooks t) : t.geography - t.history = 70 := by
  sorry


end geography_history_difference_l3432_343250


namespace trapezoid_EN_squared_l3432_343298

/-- Trapezoid ABCD with given side lengths and point N -/
structure Trapezoid :=
  (A B C D E M N : ℝ × ℝ)
  (AB_parallel_CD : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (CD_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 10)
  (DA_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
  (E_on_BC : ∃ t, E = (1 - t) • B + t • C)
  (E_on_DA : ∃ s, E = (1 - s) • D + s • A)
  (M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  (N_on_BMC : (N.1 - B.1)^2 + (N.2 - B.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - C.1)^2 + (N.2 - C.2)^2)
  (N_on_DMA : (N.1 - D.1)^2 + (N.2 - D.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - A.1)^2 + (N.2 - A.2)^2)
  (N_not_M : N ≠ M)

/-- The main theorem -/
theorem trapezoid_EN_squared (t : Trapezoid) : 
  (t.E.1 - t.N.1)^2 + (t.E.2 - t.N.2)^2 = 900 / 11 := by
  sorry

end trapezoid_EN_squared_l3432_343298


namespace A_is_singleton_floor_sum_property_l3432_343292

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the set A
def A : Set ℝ :=
  {x | x^2 - (floor x : ℝ) - 1 = 0 ∧ -1 < x ∧ x < 2}

-- Theorem 1: A is a singleton set
theorem A_is_singleton : ∃! x, x ∈ A := by sorry

-- Theorem 2: Floor function property
theorem floor_sum_property (x : ℝ) :
  (floor x : ℝ) + (floor (x + 1/2) : ℝ) = (floor (2*x) : ℝ) := by sorry

end A_is_singleton_floor_sum_property_l3432_343292


namespace percentage_women_without_retirement_plan_l3432_343277

theorem percentage_women_without_retirement_plan 
  (total_workers : ℕ)
  (workers_without_plan : ℕ)
  (men_with_plan : ℕ)
  (total_men : ℕ)
  (total_women : ℕ)
  (h1 : workers_without_plan = total_workers / 3)
  (h2 : men_with_plan = (total_workers - workers_without_plan) * 2 / 5)
  (h3 : total_men = 120)
  (h4 : total_women = 120)
  (h5 : total_workers = total_men + total_women) :
  (workers_without_plan - (total_men - men_with_plan)) * 100 / total_women = 20 := by
sorry

end percentage_women_without_retirement_plan_l3432_343277


namespace heathers_weight_l3432_343268

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) 
  (h1 : emily_weight = 9)
  (h2 : weight_difference = 78) :
  emily_weight + weight_difference = 87 := by
  sorry

end heathers_weight_l3432_343268


namespace jame_practice_weeks_l3432_343203

def regular_cards_per_tear : ℕ := 30
def thick_cards_per_tear : ℕ := 25
def cards_per_regular_deck : ℕ := 52
def cards_per_thick_deck : ℕ := 55
def tears_per_week : ℕ := 4
def regular_decks_bought : ℕ := 27
def thick_decks_bought : ℕ := 14

def total_cards : ℕ := regular_decks_bought * cards_per_regular_deck + thick_decks_bought * cards_per_thick_deck

def cards_torn_per_week : ℕ := (regular_cards_per_tear + thick_cards_per_tear) * (tears_per_week / 2)

theorem jame_practice_weeks :
  (total_cards / cards_torn_per_week : ℕ) = 19 := by sorry

end jame_practice_weeks_l3432_343203


namespace binomial_coefficient_product_l3432_343220

theorem binomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end binomial_coefficient_product_l3432_343220


namespace three_digit_five_times_smaller_l3432_343230

/-- A three-digit number -/
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

/-- The condition that a number becomes five times smaller when the first digit is removed -/
def FiveTimesSmallerWithoutFirstDigit (a b c : ℕ) : Prop :=
  5 * (b * 10 + c) = a * 100 + b * 10 + c

/-- The theorem stating that 125, 250, and 375 are the only three-digit numbers
    that become five times smaller when the first digit is removed -/
theorem three_digit_five_times_smaller :
  ∀ a b c : ℕ,
  ThreeDigitNumber a b c ∧ FiveTimesSmallerWithoutFirstDigit a b c ↔
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 3 ∧ b = 7 ∧ c = 5) :=
by sorry


end three_digit_five_times_smaller_l3432_343230


namespace cost_price_per_meter_l3432_343246

/-- Given the selling price and profit per meter of cloth, calculate the cost price per meter. -/
theorem cost_price_per_meter
  (selling_price : ℚ)
  (cloth_length : ℚ)
  (profit_per_meter : ℚ)
  (h1 : selling_price = 8925)
  (h2 : cloth_length = 85)
  (h3 : profit_per_meter = 25) :
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 := by
  sorry

end cost_price_per_meter_l3432_343246


namespace carol_carrots_l3432_343255

-- Define the variables
def total_carrots : ℕ := 38 + 7
def mom_carrots : ℕ := 16

-- State the theorem
theorem carol_carrots : total_carrots - mom_carrots = 29 := by
  sorry

end carol_carrots_l3432_343255


namespace decimal_power_equivalence_l3432_343219

theorem decimal_power_equivalence : (1 / 10 : ℝ) ^ 2 = 0.010000000000000002 := by
  sorry

end decimal_power_equivalence_l3432_343219


namespace min_value_problem_l3432_343259

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and the minimum value of (1/4)a² + (1/9)b² + c² is 8/7 -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end min_value_problem_l3432_343259


namespace robin_bracelet_cost_l3432_343204

def cost_per_bracelet : ℕ := 2

def friend_names : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_letters (names : List String) : ℕ :=
  names.map String.length |>.sum

def total_cost (names : List String) (cost : ℕ) : ℕ :=
  (total_letters names) * cost

theorem robin_bracelet_cost :
  total_cost friend_names cost_per_bracelet = 44 := by
  sorry

end robin_bracelet_cost_l3432_343204


namespace infinitely_many_twin_pretty_numbers_l3432_343229

-- Define what it means for a number to be "pretty"
def isPrettyNumber (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

-- Define a pair of twin pretty numbers
def isTwinPrettyPair (n m : ℕ) : Prop :=
  isPrettyNumber n ∧ isPrettyNumber m ∧ m = n + 1

-- Theorem statement
theorem infinitely_many_twin_pretty_numbers :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ isTwinPrettyPair n m :=
sorry

end infinitely_many_twin_pretty_numbers_l3432_343229


namespace sum_of_weighted_variables_l3432_343258

theorem sum_of_weighted_variables (x y z : ℝ) 
  (eq1 : x + y + z = 20) 
  (eq2 : x + 2*y + 3*z = 16) : 
  x + 3*y + 5*z = 12 := by
  sorry

end sum_of_weighted_variables_l3432_343258


namespace book_arrangement_l3432_343278

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (n.factorial / k.factorial : ℕ) = 120 := by
  sorry

end book_arrangement_l3432_343278


namespace cristina_pace_race_scenario_l3432_343251

/-- Cristina's pace in a race with Nicky --/
theorem cristina_pace (head_start : ℝ) (nicky_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let nicky_distance := head_start * nicky_pace + catch_up_time * nicky_pace
  nicky_distance / catch_up_time

/-- The race scenario --/
theorem race_scenario : cristina_pace 12 3 30 = 4.2 := by
  sorry

end cristina_pace_race_scenario_l3432_343251


namespace quadratic_function_properties_l3432_343212

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- The theorem stating the properties of the quadratic function and the range of m -/
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-3 : ℝ) 1, f x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 1 ∪ Set.Iio (-3 : ℝ), f x > 0) ∧
  (f 2 = 5) ∧
  (∀ m : ℝ, (∃ x : ℝ, f x = 9*m + 3) ↔ m ≥ -7/9) :=
by sorry

end quadratic_function_properties_l3432_343212


namespace card_game_problem_l3432_343272

/-- The card game problem -/
theorem card_game_problem (T : ℚ) :
  -- Initial ratios
  let initial_aldo : ℚ := 7 / 18 * T
  let initial_bernardo : ℚ := 6 / 18 * T
  let initial_carlos : ℚ := 5 / 18 * T
  -- Final ratios
  let final_aldo : ℚ := 6 / 15 * T
  let final_bernardo : ℚ := 5 / 15 * T
  let final_carlos : ℚ := 4 / 15 * T
  -- One player won 12 reais
  (∃ (winner : ℚ), winner - (winner - 12) = 12) →
  -- The changes in amounts
  (final_aldo - initial_aldo = 12 ∨
   final_bernardo - initial_bernardo = 12 ∨
   final_carlos - initial_carlos = 12) →
  -- Prove the final amounts
  (final_aldo = 432 ∧ final_bernardo = 360 ∧ final_carlos = 288) := by
sorry


end card_game_problem_l3432_343272


namespace second_polygon_sides_l3432_343226

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times that of the other, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →                             -- Assume positive side length
  50 * (3 * s) = n * s →              -- Same perimeter condition
  n = 150 := by
sorry

end second_polygon_sides_l3432_343226


namespace complex_number_modulus_one_l3432_343287

theorem complex_number_modulus_one (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  Complex.abs z = 1 → a = 1 ∨ a = -1 := by
  sorry

end complex_number_modulus_one_l3432_343287


namespace real_part_of_inverse_difference_l3432_343274

theorem real_part_of_inverse_difference (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1/4 :=
sorry

end real_part_of_inverse_difference_l3432_343274


namespace abs_fraction_less_than_one_l3432_343264

theorem abs_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
  sorry

end abs_fraction_less_than_one_l3432_343264


namespace some_number_calculation_l3432_343223

theorem some_number_calculation : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end some_number_calculation_l3432_343223


namespace souvenir_walk_distance_l3432_343245

theorem souvenir_walk_distance (total : ℝ) (hotel_to_postcard : ℝ) (postcard_to_tshirt : ℝ)
  (h1 : total = 0.89)
  (h2 : hotel_to_postcard = 0.11)
  (h3 : postcard_to_tshirt = 0.11) :
  total - (hotel_to_postcard + postcard_to_tshirt) = 0.67 := by
sorry

end souvenir_walk_distance_l3432_343245


namespace calculation_proof_l3432_343262

def mixed_to_improper (whole : Int) (num : Int) (denom : Int) : Rat :=
  (whole * denom + num) / denom

theorem calculation_proof :
  let a := mixed_to_improper 2 3 7
  let b := mixed_to_improper 5 1 3
  let c := mixed_to_improper 3 1 5
  let d := mixed_to_improper 2 1 6
  75 * (a - b) / (c + d) = -208 - 7/9 := by sorry

end calculation_proof_l3432_343262


namespace dans_marbles_l3432_343296

/-- The number of marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan's remaining marbles is the difference between initial and given away -/
theorem dans_marbles (initial : ℕ) (given_away : ℕ) :
  remaining_marbles initial given_away = initial - given_away :=
by sorry

end dans_marbles_l3432_343296


namespace dans_limes_l3432_343247

theorem dans_limes (initial_limes : ℝ) (given_limes : ℝ) (remaining_limes : ℝ) : 
  initial_limes = 9 → given_limes = 4.5 → remaining_limes = initial_limes - given_limes → remaining_limes = 4.5 := by
  sorry

end dans_limes_l3432_343247


namespace arithmetic_sequence_a5_value_l3432_343205

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_nonzero : ∃ n : ℕ, a n ≠ 0)
  (h_eq : a 5 ^ 2 - a 3 - a 7 = 0) :
  a 5 = 2 := by
sorry

end arithmetic_sequence_a5_value_l3432_343205


namespace parallel_lines_solution_l3432_343248

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∀ x y : ℝ, m₁ * x - y = 0 ↔ m₂ * x - y = 0) ↔ m₁ = m₂

/-- Given two parallel lines ax - y + a = 0 and (2a - 3)x + ay - a = 0, prove that a = -3 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x - y + a = 0 ↔ (2 * a - 3) * x + a * y - a = 0) → a = -3 :=
by sorry

end parallel_lines_solution_l3432_343248


namespace parentheses_placement_l3432_343288

theorem parentheses_placement :
  (1 : ℚ) / (2 / (3 / (4 / (5 / (6 / (7 / (8 / (9 / 10)))))))) = 7 := by
  sorry

end parentheses_placement_l3432_343288


namespace floor_plus_self_unique_solution_l3432_343290

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 14.5 :=
by
  -- The proof goes here
  sorry

end floor_plus_self_unique_solution_l3432_343290


namespace equation_solution_l3432_343241

theorem equation_solution (x y : ℝ) :
  y^2 - 2*y = x^2 + 2*x ↔ y = x + 2 ∨ y = -x := by
  sorry

end equation_solution_l3432_343241


namespace negative_numbers_l3432_343216

theorem negative_numbers (x y z : ℝ) 
  (h1 : 2 * x - y < 0) 
  (h2 : 3 * y - 2 * z < 0) 
  (h3 : 4 * z - 3 * x < 0) : 
  x < 0 ∧ y < 0 ∧ z < 0 := by
  sorry

end negative_numbers_l3432_343216


namespace not_yellow_houses_l3432_343280

/-- Represents the number of houses Isabella has of each color --/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses --/
def isabellas_houses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.yellow = h.red - 40 ∧
  h.green = 90

/-- Theorem stating the number of houses that are not yellow --/
theorem not_yellow_houses (h : Houses) (hcond : isabellas_houses h) :
  h.green + h.red = 160 :=
sorry

end not_yellow_houses_l3432_343280


namespace square_sum_from_means_l3432_343249

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = 16) : 
  a^2 + b^2 = 1088 := by
sorry

end square_sum_from_means_l3432_343249


namespace seating_arrangements_special_guest_seating_l3432_343232

theorem seating_arrangements (n : Nat) (k : Nat) (h : n > k) :
  (n : Nat) * (n - 1).factorial = n * (n - 1 : Nat).factorial :=
by sorry

theorem special_guest_seating :
  8 * 7 * 6 * 5 * 4 * 3 * 2 = 20160 :=
by sorry

end seating_arrangements_special_guest_seating_l3432_343232


namespace complement_union_and_intersection_range_of_a_l3432_343234

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for part (1)
theorem complement_union_and_intersection :
  (Set.univ \ (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by sorry

-- Theorem for part (2)
theorem range_of_a (h : A ∩ C a ≠ ∅) : a > 3 := by sorry

end complement_union_and_intersection_range_of_a_l3432_343234


namespace shortest_path_on_specific_frustum_l3432_343227

/-- Represents a truncated circular right cone (frustum) -/
structure Frustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- Calculates the shortest path on the surface of a frustum -/
def shortest_path (f : Frustum) (upper_travel : ℝ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem shortest_path_on_specific_frustum :
  let f : Frustum := {
    lower_circumference := 10,
    upper_circumference := 9,
    inclination_angle := Real.pi / 3  -- 60 degrees in radians
  }
  shortest_path f 3 = 5 * Real.sqrt 3 / Real.pi :=
sorry

end shortest_path_on_specific_frustum_l3432_343227


namespace entree_dessert_cost_difference_l3432_343231

/-- Given Hannah's restaurant bill, prove the cost difference between entree and dessert -/
theorem entree_dessert_cost_difference 
  (total_cost : ℕ) 
  (entree_cost : ℕ) 
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14) :
  entree_cost - (total_cost - entree_cost) = 5 := by
  sorry

#check entree_dessert_cost_difference

end entree_dessert_cost_difference_l3432_343231


namespace draw_probability_standard_deck_l3432_343244

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (clubs : Nat)
  (spades : Nat)

/-- A standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52,
    hearts := 13,
    clubs := 13,
    spades := 13 }

/-- The probability of drawing a heart, then a club, then a spade from a standard deck -/
def drawProbability (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.cards *
  (d.clubs : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

theorem draw_probability_standard_deck :
  drawProbability standardDeck = 2197 / 132600 := by
  sorry

end draw_probability_standard_deck_l3432_343244


namespace potato_fetching_time_l3432_343267

/-- Represents the problem of calculating how long it takes a dog to fetch a launched potato. -/
theorem potato_fetching_time 
  (football_fields : ℕ) -- number of football fields the potato is launched
  (yards_per_field : ℕ) -- length of a football field in yards
  (dog_speed : ℕ) -- dog's speed in feet per minute
  (h1 : football_fields = 6)
  (h2 : yards_per_field = 200)
  (h3 : dog_speed = 400) :
  (football_fields * yards_per_field * 3) / dog_speed = 9 := by
  sorry

#check potato_fetching_time

end potato_fetching_time_l3432_343267


namespace fourth_square_area_l3432_343260

-- Define the triangles and their properties
structure Triangle :=
  (P Q R : ℝ × ℝ)
  (isRightAngle : Bool)

-- Define the squares on the sides
structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Theorem statement
theorem fourth_square_area
  (PQR PRM : Triangle)
  (square1 square2 square3 : Square)
  (h1 : PQR.isRightAngle = true)
  (h2 : PRM.isRightAngle = true)
  (h3 : square1.area = 25)
  (h4 : square2.area = 81)
  (h5 : square3.area = 64)
  : ∃ (square4 : Square), square4.area = 8 := by
  sorry

end fourth_square_area_l3432_343260


namespace total_supervisors_count_l3432_343269

/-- The number of buses -/
def num_buses : ℕ := 7

/-- The number of supervisors per bus -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

theorem total_supervisors_count : total_supervisors = 21 := by
  sorry

end total_supervisors_count_l3432_343269


namespace runners_speed_difference_l3432_343283

/-- Given two runners starting at the same point, with one going north at 8 miles per hour
    and the other going east, if they are 5 miles apart after 1/2 hour,
    then the difference in their speeds is 2 miles per hour. -/
theorem runners_speed_difference (v : ℝ) : 
  (v ≥ 0) →  -- Ensuring non-negative speed
  ((8 * (1/2))^2 + (v * (1/2))^2 = 5^2) → 
  (8 - v = 2) :=
by sorry

end runners_speed_difference_l3432_343283


namespace quadratic_function_properties_l3432_343218

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem quadratic_function_properties :
  (∀ x, f x = x^2 - 2*x + 3) ∧
  (f 0 = 3) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x ≤ 1, ∀ y ≥ x, f x ≥ f y) ∧
  (∀ x ≥ 1, ∀ y ≥ x, f x ≤ f y) ∧
  (∀ x, f x ≥ 2) ∧
  (f 1 = 2) := by
sorry

end quadratic_function_properties_l3432_343218


namespace price_reduction_equation_l3432_343206

theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 28 ∧ 
    final_price = 16 ∧ 
    final_price = original_price * (1 - x)^2) →
  28 * (1 - x)^2 = 16 :=
by sorry

end price_reduction_equation_l3432_343206


namespace ending_number_proof_l3432_343233

def starting_number : ℕ := 100
def multiples_count : ℚ := 13.5

theorem ending_number_proof :
  ∃ (n : ℕ), n ≥ starting_number ∧ 
  (n - starting_number) / 8 + 1 = multiples_count ∧
  n = 204 :=
sorry

end ending_number_proof_l3432_343233


namespace johns_cows_value_increase_l3432_343209

/-- Calculates the increase in value of cows after weight gain -/
def cow_value_increase (initial_weights : Fin 3 → ℝ) (increase_factors : Fin 3 → ℝ) (price_per_pound : ℝ) : ℝ :=
  let new_weights := fun i => initial_weights i * increase_factors i
  let initial_values := fun i => initial_weights i * price_per_pound
  let new_values := fun i => new_weights i * price_per_pound
  (Finset.sum Finset.univ new_values) - (Finset.sum Finset.univ initial_values)

/-- The increase in value of John's cows after weight gain -/
theorem johns_cows_value_increase :
  let initial_weights : Fin 3 → ℝ := ![732, 845, 912]
  let increase_factors : Fin 3 → ℝ := ![1.35, 1.28, 1.4]
  let price_per_pound : ℝ := 2.75
  cow_value_increase initial_weights increase_factors price_per_pound = 2358.40 := by
  sorry

end johns_cows_value_increase_l3432_343209


namespace average_run_time_l3432_343270

/-- Represents the average minutes run per day for each grade -/
structure GradeRunTime where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the number of students in each grade -/
structure GradePopulation where
  seventh : ℝ
  sixth : ℝ
  eighth : ℝ

/-- Represents the number of days each grade runs per week -/
structure RunDays where
  sixth : ℕ
  seventh : ℕ
  eighth : ℕ

theorem average_run_time 
  (run_time : GradeRunTime)
  (population : GradePopulation)
  (days : RunDays)
  (h1 : run_time.sixth = 10)
  (h2 : run_time.seventh = 12)
  (h3 : run_time.eighth = 8)
  (h4 : population.sixth = 3 * population.seventh)
  (h5 : population.eighth = population.seventh / 2)
  (h6 : days.sixth = 2)
  (h7 : days.seventh = 2)
  (h8 : days.eighth = 1) :
  (run_time.sixth * population.sixth * days.sixth +
   run_time.seventh * population.seventh * days.seventh +
   run_time.eighth * population.eighth * days.eighth) /
  (population.sixth + population.seventh + population.eighth) /
  7 = 176 / 9 := by
  sorry


end average_run_time_l3432_343270


namespace shooting_team_composition_l3432_343299

theorem shooting_team_composition (x y : ℕ) : 
  x > 0 → y > 0 →
  (22 * x + 47 * y) / (x + y) = 41 →
  (y : ℚ) / (x + y) = 19 / 25 := by
sorry

end shooting_team_composition_l3432_343299


namespace cookie_distribution_l3432_343225

theorem cookie_distribution (total_cookies : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 28) (h2 : num_children = 6) : 
  total_cookies - (num_children * (total_cookies / num_children)) = 4 := by
  sorry

end cookie_distribution_l3432_343225
