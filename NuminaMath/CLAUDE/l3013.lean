import Mathlib

namespace hno3_concentration_after_addition_l3013_301348

/-- Calculates the final concentration of HNO3 after adding pure HNO3 to a solution -/
theorem hno3_concentration_after_addition
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (pure_hno3_added : ℝ)
  (h1 : initial_volume = 60)
  (h2 : initial_concentration = 0.35)
  (h3 : pure_hno3_added = 18) :
  let final_volume := initial_volume + pure_hno3_added
  let initial_hno3 := initial_volume * initial_concentration
  let final_hno3 := initial_hno3 + pure_hno3_added
  let final_concentration := final_hno3 / final_volume
  final_concentration = 0.5 := by sorry

end hno3_concentration_after_addition_l3013_301348


namespace total_problems_solved_l3013_301331

theorem total_problems_solved (initial_problems : Nat) (additional_problems : Nat) : 
  initial_problems = 45 → additional_problems = 18 → initial_problems + additional_problems = 63 :=
by sorry

end total_problems_solved_l3013_301331


namespace gcf_of_60_90_150_l3013_301325

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by sorry

end gcf_of_60_90_150_l3013_301325


namespace lattice_points_on_curve_l3013_301313

theorem lattice_points_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 15) ∧ 
    points.card = 4 := by
  sorry

end lattice_points_on_curve_l3013_301313


namespace no_integer_solution_for_cornelia_age_l3013_301303

theorem no_integer_solution_for_cornelia_age :
  ∀ (C : ℕ) (K : ℕ),
    K = 30 →
    C + 20 = 2 * (K + 20) →
    (K - 5)^2 = 3 * (C - 5) →
    False :=
by
  sorry

end no_integer_solution_for_cornelia_age_l3013_301303


namespace binomial_coefficient_12_5_l3013_301395

theorem binomial_coefficient_12_5 : Nat.choose 12 5 = 792 := by sorry

end binomial_coefficient_12_5_l3013_301395


namespace fib_150_mod_9_l3013_301345

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 -/
def fibMod9 (n : ℕ) : Fin 9 :=
  (fib n).mod 9

/-- The period of Fibonacci sequence modulo 9 -/
def fibMod9Period : ℕ := 24

theorem fib_150_mod_9 :
  fibMod9 150 = 8 := by sorry

end fib_150_mod_9_l3013_301345


namespace kathleen_store_visits_l3013_301324

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := 25

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := total_bottle_caps / bottle_caps_per_visit

theorem kathleen_store_visits : store_visits = 5 := by
  sorry

end kathleen_store_visits_l3013_301324


namespace product_n_n_plus_one_is_even_l3013_301364

theorem product_n_n_plus_one_is_even (n : ℕ) : Even (n * (n + 1)) := by
  sorry

end product_n_n_plus_one_is_even_l3013_301364


namespace superior_rainbow_max_quantity_l3013_301380

/-- Represents the mixing ratios for Superior Rainbow paint -/
structure MixingRatios where
  red : Rat
  white : Rat
  blue : Rat
  yellow : Rat

/-- Represents the available paint quantities -/
structure AvailablePaint where
  red : Nat
  white : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the maximum quantity of Superior Rainbow paint -/
def maxSuperiorRainbow (ratios : MixingRatios) (available : AvailablePaint) : Nat :=
  sorry

/-- Theorem: The maximum quantity of Superior Rainbow paint is 121 pints -/
theorem superior_rainbow_max_quantity :
  let ratios : MixingRatios := ⟨3/4, 2/3, 1/4, 1/6⟩
  let available : AvailablePaint := ⟨50, 45, 20, 15⟩
  maxSuperiorRainbow ratios available = 121 := by
  sorry

end superior_rainbow_max_quantity_l3013_301380


namespace parabola_zeros_difference_l3013_301357

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Checks if a point (x, y) is on the parabola -/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  p.y x = y

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.containsPoint 3 (-9) →
  p.containsPoint 5 7 →
  ∃ m n : ℝ, m ∈ p.zeros ∧ n ∈ p.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end parabola_zeros_difference_l3013_301357


namespace abc_inequality_l3013_301368

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end abc_inequality_l3013_301368


namespace andreas_living_room_area_andreas_living_room_area_is_48_l3013_301312

/-- The area of Andrea's living room floor given a carpet covering 75% of it -/
theorem andreas_living_room_area (carpet_width : ℝ) (carpet_length : ℝ) 
  (carpet_coverage_percentage : ℝ) : ℝ :=
  let carpet_area := carpet_width * carpet_length
  let floor_area := carpet_area / carpet_coverage_percentage
  floor_area

/-- Proof of Andrea's living room floor area -/
theorem andreas_living_room_area_is_48 :
  andreas_living_room_area 4 9 0.75 = 48 := by
  sorry

end andreas_living_room_area_andreas_living_room_area_is_48_l3013_301312


namespace max_rectangle_area_l3013_301394

theorem max_rectangle_area (l w : ℝ) (h_perimeter : l + w = 10) :
  l * w ≤ 25 :=
sorry

end max_rectangle_area_l3013_301394


namespace rectangle_area_with_tangent_circle_l3013_301302

/-- Given a rectangle ABCD with a circle of radius r tangent to sides AB, AD, and CD,
    and passing through a point one-third the distance from A to C along diagonal AC,
    the area of the rectangle is (2√2)/3 * r^2. -/
theorem rectangle_area_with_tangent_circle (r : ℝ) (h : r > 0) :
  ∃ (w h : ℝ),
    w > 0 ∧ h > 0 ∧
    h = r ∧
    (w^2 + h^2) = 9 * r^2 ∧
    w * h = (2 * Real.sqrt 2 / 3) * r^2 :=
by sorry

end rectangle_area_with_tangent_circle_l3013_301302


namespace tissues_left_is_1060_l3013_301307

/-- The number of tissues Tucker has left after all actions. -/
def tissues_left : ℕ :=
  let brand_a_per_box := 160
  let brand_b_per_box := 180
  let brand_c_per_box := 200
  let brand_a_boxes := 4
  let brand_b_boxes := 6
  let brand_c_boxes := 2
  let brand_a_used := 250
  let brand_b_used := 410
  let brand_c_used := 150
  let brand_b_given := 2
  let brand_c_received := 110

  let brand_a_left := brand_a_per_box * brand_a_boxes - brand_a_used
  let brand_b_left := brand_b_per_box * brand_b_boxes - brand_b_used - brand_b_per_box * brand_b_given
  let brand_c_left := brand_c_per_box * brand_c_boxes - brand_c_used + brand_c_received

  brand_a_left + brand_b_left + brand_c_left

theorem tissues_left_is_1060 : tissues_left = 1060 := by
  sorry

end tissues_left_is_1060_l3013_301307


namespace sum_of_fractions_geq_one_l3013_301367

theorem sum_of_fractions_geq_one (x y z : ℝ) :
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end sum_of_fractions_geq_one_l3013_301367


namespace tea_mixture_profit_l3013_301323

/-- Proves that the given tea mixture achieves the desired profit -/
theorem tea_mixture_profit (x y : ℝ) : 
  x + y = 100 →
  0.32 * x + 0.40 * y = 34.40 →
  x = 70 ∧ y = 30 ∧ 
  (0.43 * 100 / (0.32 * x + 0.40 * y) - 1) * 100 = 25 := by
  sorry

end tea_mixture_profit_l3013_301323


namespace inverse_38_mod_53_l3013_301377

theorem inverse_38_mod_53 (h : (16⁻¹ : ZMod 53) = 20) : (38⁻¹ : ZMod 53) = 25 := by
  sorry

end inverse_38_mod_53_l3013_301377


namespace right_triangle_segment_ratio_l3013_301384

theorem right_triangle_segment_ratio (x y z u v : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ v > 0 →
  x^2 + y^2 = z^2 →
  x / y = 2 / 5 →
  u * z = x^2 →
  v * z = y^2 →
  u + v = z →
  u / v = 4 / 25 := by
  sorry

end right_triangle_segment_ratio_l3013_301384


namespace product_of_sines_equals_one_fourth_l3013_301376

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π/8)) * (1 - Real.sin (3*π/8)) * (1 - Real.sin (5*π/8)) * (1 - Real.sin (7*π/8)) = 1/4 := by
  sorry

end product_of_sines_equals_one_fourth_l3013_301376


namespace convention_center_tables_l3013_301361

/-- The number of tables in the convention center. -/
def num_tables : ℕ := 26

/-- The number of chairs around each table. -/
def chairs_per_table : ℕ := 8

/-- The number of legs each chair has. -/
def legs_per_chair : ℕ := 4

/-- The number of legs each table has. -/
def legs_per_table : ℕ := 5

/-- The number of extra chairs not linked with any table. -/
def extra_chairs : ℕ := 10

/-- The total number of legs from tables and chairs. -/
def total_legs : ℕ := 1010

theorem convention_center_tables :
  num_tables * chairs_per_table * legs_per_chair +
  num_tables * legs_per_table +
  extra_chairs * legs_per_chair = total_legs :=
by sorry

end convention_center_tables_l3013_301361


namespace angle_C_measure_l3013_301339

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the conditions
axiom scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
axiom angle_sum : A + B + C = 180
axiom angle_relation : C = A + 40
axiom angle_B : B = 2 * A

-- Theorem to prove
theorem angle_C_measure : C = 75 := by
  sorry

end angle_C_measure_l3013_301339


namespace min_abs_a_for_solvable_equation_l3013_301343

theorem min_abs_a_for_solvable_equation :
  ∀ (a b : ℤ),
  (a + 2 * b = 32) →
  (∀ a' : ℤ, a' > 0 ∧ (∃ b' : ℤ, a' + 2 * b' = 32) → a' ≥ 4) →
  (∃ b'' : ℤ, (-2) + 2 * b'' = 32) →
  (∃ a₀ : ℤ, |a₀| = 2 ∧ (∃ b₀ : ℤ, a₀ + 2 * b₀ = 32) ∧
    ∀ a' : ℤ, (∃ b' : ℤ, a' + 2 * b' = 32) → |a'| ≥ 2) :=
by sorry

end min_abs_a_for_solvable_equation_l3013_301343


namespace cone_lateral_surface_area_l3013_301314

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π. -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π :=
by sorry

end cone_lateral_surface_area_l3013_301314


namespace nines_in_range_70_l3013_301354

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (if n % 10 ≥ 9 then 1 else 0)

theorem nines_in_range_70 : count_nines 70 = 7 := by
  sorry

end nines_in_range_70_l3013_301354


namespace total_birds_and_storks_l3013_301335

def birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : ℕ :=
  initial_birds + initial_storks + additional_storks

theorem total_birds_and_storks :
  birds_and_storks 3 4 6 = 13 := by
  sorry

end total_birds_and_storks_l3013_301335


namespace geometric_series_second_term_l3013_301366

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the second term is 7.5 -/
theorem geometric_series_second_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 40 → 
  a * (1/4) = 7.5 := by
sorry

end geometric_series_second_term_l3013_301366


namespace number_division_problem_l3013_301363

theorem number_division_problem (x : ℝ) : x / 5 = 80 + x / 6 ↔ x = 2400 := by
  sorry

end number_division_problem_l3013_301363


namespace length_of_segment_l3013_301310

/-- Given a line segment AB divided by points P and Q, prove that AB has length 25 -/
theorem length_of_segment (A B P Q : ℝ) : 
  (P - A) / (B - A) = 3 / 5 →  -- P divides AB in ratio 3:2
  (Q - A) / (B - A) = 2 / 5 →  -- Q divides AB in ratio 2:3
  Q - P = 5 →                  -- Distance between P and Q is 5 units
  B - A = 25 := by             -- Length of AB is 25 units
sorry

end length_of_segment_l3013_301310


namespace road_repair_hours_l3013_301330

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 42)
  (h2 : people2 = 30)
  (h3 : days1 = 12)
  (h4 : days2 = 14)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
  sorry

end road_repair_hours_l3013_301330


namespace min_value_of_sum_min_value_is_4_plus_4sqrt3_l3013_301342

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) → 
  ∀ a b : ℝ, a > 0 → b > 0 → (1 / (a + 1) + 1 / (b + 1) = 1 / 2) → 
  x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_4_plus_4sqrt3 (x y : ℝ) :
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) →
  x + 3 * y = 4 + 4 * Real.sqrt 3 :=
by sorry

end min_value_of_sum_min_value_is_4_plus_4sqrt3_l3013_301342


namespace tim_attend_probability_l3013_301352

-- Define the probability of rain
def prob_rain : ℝ := 0.6

-- Define the probability of sun (complementary to rain)
def prob_sun : ℝ := 1 - prob_rain

-- Define the probability Tim attends if it rains
def prob_attend_rain : ℝ := 0.25

-- Define the probability Tim attends if it's sunny
def prob_attend_sun : ℝ := 0.7

-- Theorem statement
theorem tim_attend_probability :
  prob_rain * prob_attend_rain + prob_sun * prob_attend_sun = 0.43 := by
sorry

end tim_attend_probability_l3013_301352


namespace geometric_sequence_common_ratio_l3013_301373

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = a n * (a 2 / a 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) : 
  a 2 / a 1 = -2 := by
sorry

end geometric_sequence_common_ratio_l3013_301373


namespace correct_calculation_l3013_301359

theorem correct_calculation (a b x p q : ℝ) : 
  (∀ a ≠ 0, (a * b^4)^4 ≠ a * b^8) ∧ 
  (∀ p q, (-3 * p * q)^2 ≠ -6 * p^2 * q^2) ∧ 
  (∀ x, x^2 - 1/2 * x + 1/4 ≠ (x - 1/2)^2) ∧ 
  (∀ a, 3 * (a^2)^3 - 6 * a^6 = -3 * a^6) := by
sorry

end correct_calculation_l3013_301359


namespace football_players_count_l3013_301390

/-- Represents the number of players for each sport type -/
structure PlayerCounts where
  cricket : Nat
  hockey : Nat
  softball : Nat
  total : Nat

/-- Calculates the number of football players given the counts of other players -/
def footballPlayers (counts : PlayerCounts) : Nat :=
  counts.total - (counts.cricket + counts.hockey + counts.softball)

/-- Theorem stating that the number of football players is 11 given the specific counts -/
theorem football_players_count (counts : PlayerCounts)
  (h1 : counts.cricket = 12)
  (h2 : counts.hockey = 17)
  (h3 : counts.softball = 10)
  (h4 : counts.total = 50) :
  footballPlayers counts = 11 := by
  sorry

end football_players_count_l3013_301390


namespace isosceles_triangle_perimeter_l3013_301338

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 12 :=  -- The perimeter is 12
by
  sorry


end isosceles_triangle_perimeter_l3013_301338


namespace quadratic_roots_and_reciprocals_l3013_301381

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * (k + 1) * x + k - 1

-- Theorem statement
theorem quadratic_roots_and_reciprocals (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ 
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1 :=
sorry

end quadratic_roots_and_reciprocals_l3013_301381


namespace largest_stamps_per_page_l3013_301346

theorem largest_stamps_per_page : Nat.gcd 840 1008 = 168 := by
  sorry

end largest_stamps_per_page_l3013_301346


namespace valid_arrangements_count_l3013_301333

/-- The number of ways to arrange the digits 3, 0, 5, 7, 0 into a 5-digit number -/
def digit_arrangements : ℕ :=
  let digits : Multiset ℕ := {3, 0, 5, 7, 0}
  let total_arrangements := Nat.factorial 5 / (Nat.factorial 2)  -- Total permutations with repetition
  let arrangements_starting_with_zero := Nat.factorial 4 / (Nat.factorial 2)  -- Arrangements starting with 0
  total_arrangements - arrangements_starting_with_zero

/-- The theorem stating that the number of valid arrangements is 48 -/
theorem valid_arrangements_count : digit_arrangements = 48 := by
  sorry

end valid_arrangements_count_l3013_301333


namespace sum_of_squares_and_products_l3013_301344

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52)
  (h5 : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by
sorry

end sum_of_squares_and_products_l3013_301344


namespace quadratic_function_range_l3013_301393

theorem quadratic_function_range (k m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + m > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4 * x₁ + k = 0 ∧ x₂^2 - 4 * x₂ + k = 0) →
  (∀ k' : ℤ, k' > k → 
    (∃ x : ℝ, 2 * x^2 - 2 * k' * x + m ≤ 0) ∨
    (∀ x₁ x₂ : ℝ, x₁ = x₂ ∨ x₁^2 - 4 * x₁ + k' ≠ 0 ∨ x₂^2 - 4 * x₂ + k' ≠ 0)) →
  m > 9/2 :=
by sorry

end quadratic_function_range_l3013_301393


namespace handicraft_sale_properties_l3013_301391

/-- Represents the daily profit function for a handicraft item sale --/
def daily_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

/-- Represents the daily sales volume function --/
def daily_sales (x : ℝ) : ℝ :=
  50 + 5 * (100 - x)

/-- Theorem stating the properties of the handicraft item sale --/
theorem handicraft_sale_properties :
  let cost : ℝ := 50
  let base_price : ℝ := 100
  let base_sales : ℝ := 50
  ∀ x : ℝ, cost ≤ x ∧ x ≤ base_price →
    (daily_profit x = (x - cost) * daily_sales x) ∧
    (∃ max_profit max_price, 
      max_profit = daily_profit max_price ∧
      max_price = 80 ∧ 
      max_profit = 4500 ∧
      ∀ y, cost ≤ y ∧ y ≤ base_price → daily_profit y ≤ max_profit) ∧
    (∃ min_total_cost,
      min_total_cost = 5000 ∧
      ∀ z, cost ≤ z ∧ z ≤ base_price →
        daily_profit z ≥ 4000 → cost * daily_sales z ≥ min_total_cost) := by
  sorry


end handicraft_sale_properties_l3013_301391


namespace valid_colorings_count_l3013_301347

/-- A color used for vertex coloring -/
inductive Color
| Red
| White
| Blue

/-- A vertex in the triangle structure -/
structure Vertex :=
  (id : ℕ)
  (color : Color)

/-- A triangle in the structure -/
structure Triangle :=
  (vertices : Fin 3 → Vertex)

/-- The entire structure of three connected triangles -/
structure TriangleStructure :=
  (triangles : Fin 3 → Triangle)
  (middle_restricted : Vertex)

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (s : TriangleStructure) : Prop :=
  ∀ i j : Fin 3, ∀ k l : Fin 3,
    (s.triangles i).vertices k ≠ (s.triangles j).vertices l →
    ((s.triangles i).vertices k).color ≠ ((s.triangles j).vertices l).color

/-- Predicate to check if the middle restricted vertex is colored correctly -/
def is_middle_restricted_valid (s : TriangleStructure) : Prop :=
  s.middle_restricted.color = Color.Red ∨ s.middle_restricted.color = Color.White

/-- The number of valid colorings for the triangle structure -/
def num_valid_colorings : ℕ := 36

/-- Theorem stating that the number of valid colorings is 36 -/
theorem valid_colorings_count :
  ∀ s : TriangleStructure,
    is_valid_coloring s →
    is_middle_restricted_valid s →
    num_valid_colorings = 36 :=
sorry

end valid_colorings_count_l3013_301347


namespace correct_flight_distance_l3013_301304

/-- The total distance Peter needs to fly from Germany to Russia and then back to Spain -/
def total_flight_distance (spain_russia_distance spain_germany_distance : ℕ) : ℕ :=
  (spain_russia_distance - spain_germany_distance) + 2 * spain_germany_distance

/-- Theorem stating the correct total flight distance given the problem conditions -/
theorem correct_flight_distance :
  total_flight_distance 7019 1615 = 8634 := by
  sorry

end correct_flight_distance_l3013_301304


namespace marble_remainder_l3013_301336

theorem marble_remainder (n m : ℤ) : ∃ k : ℤ, (8*n + 5) + (8*m + 7) + 6 = 8*k + 2 := by
  sorry

end marble_remainder_l3013_301336


namespace calculation_proof_l3013_301375

theorem calculation_proof :
  (3 / (-1/2) - (2/5 - 1/3) * 15 = -7) ∧
  ((-3)^2 - (-2)^3 * (-1/4) - (-1 + 6) = 2) := by
sorry

end calculation_proof_l3013_301375


namespace car_gasoline_theorem_l3013_301371

/-- Represents the relationship between remaining gasoline and distance traveled for a car --/
def gasoline_function (x : ℝ) : ℝ := 50 - 0.1 * x

/-- Represents the valid range for the distance traveled --/
def valid_distance (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 500

theorem car_gasoline_theorem :
  ∀ x : ℝ,
  valid_distance x →
  (∀ y : ℝ, y = gasoline_function x → y = 50 - 0.1 * x) ∧
  (x = 200 → gasoline_function x = 30) :=
by sorry

end car_gasoline_theorem_l3013_301371


namespace min_value_cyclic_fraction_l3013_301360

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 ∧ 
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end min_value_cyclic_fraction_l3013_301360


namespace isosceles_triangle_base_length_l3013_301397

/-- An isosceles triangle with specific height measurements -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the equal sides
  side : ℝ
  -- The height to the base
  height_to_base : ℝ
  -- The height to a lateral side
  height_to_side : ℝ
  -- Conditions for an isosceles triangle
  isosceles : side > 0
  base_positive : base > 0
  height_to_base_positive : height_to_base > 0
  height_to_side_positive : height_to_side > 0
  -- Pythagorean theorem for the height to the base
  pythagorean_base : side^2 = height_to_base^2 + (base/2)^2
  -- Pythagorean theorem for the height to the side
  pythagorean_side : side^2 = height_to_side^2 + (base/2)^2

/-- Theorem: If the height to the base is 10 and the height to the side is 12,
    then the base of the isosceles triangle is 15 -/
theorem isosceles_triangle_base_length
  (triangle : IsoscelesTriangle)
  (h1 : triangle.height_to_base = 10)
  (h2 : triangle.height_to_side = 12) :
  triangle.base = 15 := by
  sorry

end isosceles_triangle_base_length_l3013_301397


namespace angle_bisector_theorem_l3013_301349

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define point D on the angle bisector of A
variable (D : EuclideanSpace ℝ (Fin 2))

-- Assumption that ABC is a triangle
variable (h_triangle : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))

-- Assumption that D is on BC
variable (h_D_on_BC : D ∈ LineSegment B C)

-- Assumption that AD is the angle bisector of angle BAC
variable (h_angle_bisector : AngleBisector A B C D)

-- Theorem statement
theorem angle_bisector_theorem :
  (dist A B) / (dist A C) = (dist B D) / (dist C D) := by sorry

end angle_bisector_theorem_l3013_301349


namespace exists_same_color_rectangle_l3013_301305

/-- A color type representing red, black, and blue -/
inductive Color
  | Red
  | Black
  | Blue

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that assigns a color to each point in the plane -/
def colorFunction : Point → Color := sorry

/-- A type representing a rectangle in the plane -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- A predicate that checks if all vertices of a rectangle have the same color -/
def sameColorVertices (rect : Rectangle) : Prop :=
  colorFunction rect.topLeft = colorFunction rect.topRight ∧
  colorFunction rect.topLeft = colorFunction rect.bottomLeft ∧
  colorFunction rect.topLeft = colorFunction rect.bottomRight

/-- Theorem stating that there exists a rectangle with vertices of the same color -/
theorem exists_same_color_rectangle : ∃ (rect : Rectangle), sameColorVertices rect := by
  sorry


end exists_same_color_rectangle_l3013_301305


namespace smallest_number_with_given_remainders_l3013_301318

theorem smallest_number_with_given_remainders : ∃ (a : ℕ), (
  (a % 3 = 1) ∧
  (a % 6 = 3) ∧
  (a % 7 = 4) ∧
  (∀ b : ℕ, b < a → (b % 3 ≠ 1 ∨ b % 6 ≠ 3 ∨ b % 7 ≠ 4))
) ∧ a = 39 := by
  sorry

end smallest_number_with_given_remainders_l3013_301318


namespace eighth_term_is_23_l3013_301332

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighth_term_is_23 :
  arithmetic_sequence 2 3 8 = 23 := by
  sorry

end eighth_term_is_23_l3013_301332


namespace range_of_a_l3013_301329

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

-- State the theorem
theorem range_of_a : 
  (∃ a : ℝ, C a ∪ (Set.univ \ B) = Set.univ) ↔ 
  (∃ a : ℝ, a ≤ -3 ∧ C a ∪ (Set.univ \ B) = Set.univ) :=
sorry

end range_of_a_l3013_301329


namespace trampoline_jumps_l3013_301365

/-- The number of times Ronald jumped on the trampoline -/
def ronald_jumps : ℕ := 157

/-- The additional number of times Rupert jumped compared to Ronald -/
def rupert_additional_jumps : ℕ := 86

/-- The number of times Rupert jumped on the trampoline -/
def rupert_jumps : ℕ := ronald_jumps + rupert_additional_jumps

/-- The average number of jumps between the two brothers -/
def average_jumps : ℕ := (ronald_jumps + rupert_jumps) / 2

/-- The total number of jumps made by both Rupert and Ronald -/
def total_jumps : ℕ := ronald_jumps + rupert_jumps

theorem trampoline_jumps :
  average_jumps = 200 ∧ total_jumps = 400 := by
  sorry

end trampoline_jumps_l3013_301365


namespace exists_m_for_all_n_l3013_301387

theorem exists_m_for_all_n : ∀ (n : ℤ), ∃ (m : ℤ), n * m = m := by
  sorry

end exists_m_for_all_n_l3013_301387


namespace min_sum_y_intersections_l3013_301383

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through a given point with a slope -/
structure Line where
  point : Point
  slope : ℝ

/-- Represents a parabola of the form x^2 = 2y -/
def Parabola : Type := Unit

/-- Returns the y-coordinate of a point on the given line -/
def lineY (l : Line) (x : ℝ) : ℝ :=
  l.point.y + l.slope * (x - l.point.x)

/-- Returns true if the given point lies on the parabola -/
def onParabola (p : Point) : Prop :=
  p.x^2 = 2 * p.y

/-- Returns true if the given point lies on the given line -/
def onLine (l : Line) (p : Point) : Prop :=
  p.y = lineY l p.x

/-- Theorem stating that the minimum sum of y-coordinates of intersection points is 2 -/
theorem min_sum_y_intersections (p : Parabola) :
  ∀ l : Line,
    l.point = Point.mk 0 1 →
    ∃ A B : Point,
      onParabola A ∧ onLine l A ∧
      onParabola B ∧ onLine l B ∧
      A ≠ B →
      (∀ C D : Point,
        onParabola C ∧ onLine l C ∧
        onParabola D ∧ onLine l D ∧
        C ≠ D →
        A.y + B.y ≤ C.y + D.y) →
      A.y + B.y = 2 :=
sorry

end min_sum_y_intersections_l3013_301383


namespace complex_in_fourth_quadrant_l3013_301389

theorem complex_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_in_fourth_quadrant_l3013_301389


namespace cos_A_minus_B_l3013_301385

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 5/3) : 
  Real.cos (A - B) = 8/9 := by
  sorry

end cos_A_minus_B_l3013_301385


namespace nellie_uncle_rolls_l3013_301337

/-- Prove that Nellie sold 10 rolls to her uncle -/
theorem nellie_uncle_rolls : 
  ∀ (total_rolls grandmother_rolls neighbor_rolls remaining_rolls : ℕ),
  total_rolls = 45 →
  grandmother_rolls = 1 →
  neighbor_rolls = 6 →
  remaining_rolls = 28 →
  total_rolls - remaining_rolls - grandmother_rolls - neighbor_rolls = 10 :=
by
  sorry

end nellie_uncle_rolls_l3013_301337


namespace reciprocal_sum_l3013_301320

theorem reciprocal_sum : (1 / (1 / 4 + 1 / 6) : ℚ) = 12 / 5 := by
  sorry

end reciprocal_sum_l3013_301320


namespace probability_three_heads_five_tosses_l3013_301315

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1/2)^k * (1/2)^(n-k)

/-- The probability of getting exactly 3 heads in 5 tosses of a fair coin is 5/16 -/
theorem probability_three_heads_five_tosses :
  probability_k_heads 5 3 = 5/16 := by
  sorry

end probability_three_heads_five_tosses_l3013_301315


namespace circle_line_intersection_range_l3013_301328

theorem circle_line_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + Real.sqrt 3 * y + m = 0 ∧ 
   (x + Real.sqrt 3 * y + m + 1)^2 + y^2 = 4 * ((x + Real.sqrt 3 * y + m - 1)^2 + y^2)) → 
  -13/3 ≤ m ∧ m ≤ 1 := by
  sorry

end circle_line_intersection_range_l3013_301328


namespace sufficient_not_necessary_l3013_301327

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end sufficient_not_necessary_l3013_301327


namespace total_teachers_l3013_301316

theorem total_teachers (num_departments : ℕ) (teachers_per_department : ℕ) 
  (h1 : num_departments = 15) 
  (h2 : teachers_per_department = 35) : 
  num_departments * teachers_per_department = 525 := by
  sorry

end total_teachers_l3013_301316


namespace integer_solutions_equation_l3013_301326

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = 
  {(-1, -1), (-1, 0), (0, -1), (0, 0), (5, 2), (-6, 2)} := by
  sorry

end integer_solutions_equation_l3013_301326


namespace ellipse_triangle_perimeter_l3013_301392

/-- Perimeter of triangle PF₁F₂ for a specific ellipse -/
theorem ellipse_triangle_perimeter :
  ∀ (a b c : ℝ) (P F₁ F₂ : ℝ × ℝ),
  -- Ellipse equation
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | x^2 / a^2 + y^2 / 2 = 1} → x^2 / a^2 + y^2 / 2 = 1) →
  -- F₁ and F₂ are foci
  F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0 →
  -- P is on the ellipse
  P ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 2 = 1} →
  -- F₁ is symmetric to y = -x at P
  P.1 = -F₁.2 ∧ P.2 = -F₁.1 →
  -- Perimeter of triangle PF₁F₂
  dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end ellipse_triangle_perimeter_l3013_301392


namespace toys_ratio_l3013_301369

theorem toys_ratio (s : ℚ) : 
  (s * 20 = (142 - 20 - s * 20) - 2) →
  (s * 20 + (142 - 20 - s * 20) + 20 = 142) →
  (s = 3) :=
by sorry

end toys_ratio_l3013_301369


namespace ellipse_foci_distance_l3013_301301

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 900}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ ‖f₁ - f₂‖ = 40 * Real.sqrt 2 :=
by sorry

end ellipse_foci_distance_l3013_301301


namespace carpet_cut_length_l3013_301351

theorem carpet_cut_length (square_area : ℝ) (room_area : ℝ) : 
  square_area = 169 →
  room_area = 143 →
  (Real.sqrt square_area - room_area / Real.sqrt square_area) = 2 := by
  sorry

end carpet_cut_length_l3013_301351


namespace g_is_linear_l3013_301378

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom integral_condition : ∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t

axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem g_is_linear : (∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t) → 
                      (∀ x : ℝ, f (-x) = -f x) → 
                      (∀ x : ℝ, g x = 1 + x) :=
sorry

end g_is_linear_l3013_301378


namespace existence_of_floor_representation_l3013_301308

def is_valid_sequence (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, 1 ≤ i → 1 ≤ j → i + j ≤ 1997 →
    f i + f j ≤ f (i + j) ∧ f (i + j) ≤ f i + f j + 1

theorem existence_of_floor_representation (f : ℕ → ℕ) :
  is_valid_sequence f →
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → f n = ⌊n * x⌋ :=
sorry

end existence_of_floor_representation_l3013_301308


namespace not_all_axially_symmetric_figures_have_one_axis_l3013_301382

/-- A type representing geometric figures -/
structure Figure where
  -- Add necessary fields here
  
/-- Predicate to check if a figure is axially symmetric -/
def is_axially_symmetric (f : Figure) : Prop :=
  sorry

/-- Function to count the number of axes of symmetry for a figure -/
def count_axes_of_symmetry (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that not all axially symmetric figures have only one axis of symmetry -/
theorem not_all_axially_symmetric_figures_have_one_axis :
  ¬ (∀ f : Figure, is_axially_symmetric f → count_axes_of_symmetry f = 1) :=
sorry

end not_all_axially_symmetric_figures_have_one_axis_l3013_301382


namespace min_max_values_l3013_301388

theorem min_max_values : 
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 2 → a + 2 * b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 2 ∧ a + 2 * b = 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a + b ≤ Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = 1 ∧ a + b = Real.sqrt 2) :=
by sorry

end min_max_values_l3013_301388


namespace sin_squared_value_l3013_301350

theorem sin_squared_value (θ : Real) 
  (h : Real.cos θ ^ 4 + Real.sin θ ^ 4 + (Real.cos θ * Real.sin θ) ^ 4 + 
       1 / (Real.cos θ ^ 4 + Real.sin θ ^ 4) = 41 / 16) : 
  Real.sin θ ^ 2 = 1 / 2 := by
  sorry

end sin_squared_value_l3013_301350


namespace a_10_equals_1023_l3013_301322

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n + 2^(n + 1)

theorem a_10_equals_1023 : sequence_a 9 = 1023 := by
  sorry

end a_10_equals_1023_l3013_301322


namespace triangle_theorem_l3013_301306

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem --/
theorem triangle_theorem (t : Triangle) :
  (t.b / (t.a + t.c) = (t.a + t.b - t.c) / (t.a + t.b)) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 15 ∧ t.b = 10 → Real.cos t.B = Real.sqrt 6 / 3) := by
  sorry


end triangle_theorem_l3013_301306


namespace x_n_root_bound_l3013_301386

theorem x_n_root_bound (n : ℕ) (a : ℝ) (x : ℕ → ℝ) (α : ℕ → ℝ)
  (hn : n > 1)
  (ha : a ≥ 1)
  (hx1 : x 1 = 1)
  (hxi : ∀ i ∈ Finset.range n, i ≥ 2 → x i / x (i-1) = a + α i)
  (hαi : ∀ i ∈ Finset.range n, i ≥ 2 → α i ≤ 1 / (i * (i + 1))) :
  (x n) ^ (1 / (n - 1 : ℝ)) < a + 1 / (n - 1 : ℝ) := by
  sorry

end x_n_root_bound_l3013_301386


namespace unique_solution_quadratic_l3013_301309

/-- For a quadratic equation qx^2 - 8x + 2 = 0 with q ≠ 0, it has only one solution iff q = 8 -/
theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end unique_solution_quadratic_l3013_301309


namespace abs_lt_sufficient_not_necessary_l3013_301356

theorem abs_lt_sufficient_not_necessary (a b : ℝ) (ha : a > 0) :
  (∀ a b, (abs a < b) → (-a < b)) ∧
  (∃ a b, a > 0 ∧ (-a < b) ∧ (abs a ≥ b)) :=
sorry

end abs_lt_sufficient_not_necessary_l3013_301356


namespace bounded_sequence_with_lcm_condition_l3013_301334

theorem bounded_sequence_with_lcm_condition (n : ℕ) (k : ℕ) (a : Fin k → ℕ) :
  (∀ i : Fin k, 1 ≤ a i) →
  (∀ i j : Fin k, i < j → a i < a j) →
  (∀ i : Fin k, a i ≤ n) →
  (∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) →
  k ≤ 2 * Int.floor (Real.sqrt n) :=
by sorry

end bounded_sequence_with_lcm_condition_l3013_301334


namespace max_points_32_points_32_achievable_l3013_301398

/-- Represents a basketball game where a player only attempts three-point and two-point shots -/
structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  threePointSuccessRate : ℚ
  twoPointSuccessRate : ℚ

/-- Calculates the total points scored in a basketball game -/
def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccessRate * game.threePointAttempts +
  2 * game.twoPointSuccessRate * game.twoPointAttempts

/-- Theorem stating that under the given conditions, the maximum points scored is 32 -/
theorem max_points_32 (game : BasketballGame) 
    (h1 : game.threePointAttempts + game.twoPointAttempts = 40)
    (h2 : game.threePointSuccessRate = 1/4)
    (h3 : game.twoPointSuccessRate = 2/5) :
  totalPoints game ≤ 32 := by
  sorry

/-- Theorem stating that 32 points can be achieved -/
theorem points_32_achievable : 
  ∃ (game : BasketballGame), 
    game.threePointAttempts + game.twoPointAttempts = 40 ∧
    game.threePointSuccessRate = 1/4 ∧
    game.twoPointSuccessRate = 2/5 ∧
    totalPoints game = 32 := by
  sorry

end max_points_32_points_32_achievable_l3013_301398


namespace parabola_single_intersection_l3013_301396

/-- A parabola y = x^2 + 4x + 5 - m intersects the x-axis at only one point if and only if m = 1 -/
theorem parabola_single_intersection (m : ℝ) : 
  (∃! x, x^2 + 4*x + 5 - m = 0) ↔ m = 1 := by
sorry

end parabola_single_intersection_l3013_301396


namespace reciprocal_of_negative_half_l3013_301340

theorem reciprocal_of_negative_half :
  (1 : ℚ) / (-1/2 : ℚ) = -2 := by sorry

end reciprocal_of_negative_half_l3013_301340


namespace choir_size_l3013_301319

/-- Given an orchestra with female and male students, and a choir with three times
    the number of people in the orchestra, calculate the number of people in the choir. -/
theorem choir_size (female_students male_students : ℕ) 
  (h1 : female_students = 18) 
  (h2 : male_students = 25) : 
  3 * (female_students + male_students) = 129 := by
  sorry

end choir_size_l3013_301319


namespace refrigerator_cash_price_l3013_301317

/-- The cash price of a refrigerator given installment payment details --/
theorem refrigerator_cash_price 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (cash_savings : ℕ) : 
  deposit = 3000 →
  num_installments = 30 →
  installment_amount = 300 →
  cash_savings = 4000 →
  deposit + num_installments * installment_amount - cash_savings = 8000 := by
  sorry

end refrigerator_cash_price_l3013_301317


namespace equilateral_triangle_area_perimeter_ratio_l3013_301399

theorem equilateral_triangle_area_perimeter_ratio :
  ∀ s : ℝ,
  s > 0 →
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  s = 6 →
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l3013_301399


namespace intersection_distance_is_2b_l3013_301379

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- Represents a parabola with focus (p, 0) and directrix x = -p -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The distance between intersection points of an ellipse and a parabola -/
def intersection_distance (e : Ellipse) (p : Parabola) : ℝ :=
  sorry

/-- Theorem stating the distance between intersection points -/
theorem intersection_distance_is_2b 
  (e : Ellipse) 
  (p : Parabola) 
  (h1 : e.a = 5 ∧ e.b = 4)  -- Ellipse equation condition
  (h2 : p.p = 3)  -- Shared focus condition
  (h3 : ∃ b : ℝ, b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1)  -- Intersection condition
  : 
  ∃ b : ℝ, intersection_distance e p = 2 * b ∧ 
    b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1 :=
sorry

end intersection_distance_is_2b_l3013_301379


namespace second_train_length_second_train_length_solution_l3013_301374

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 80 65 7.596633648618456 141 - 165.12| < ε :=
sorry

end second_train_length_second_train_length_solution_l3013_301374


namespace age_difference_l3013_301311

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 23 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 25 := by
  sorry

end age_difference_l3013_301311


namespace online_store_commission_l3013_301355

/-- Calculates the commission percentage of an online store given the cost price,
    desired profit percentage, and final observed price. -/
theorem online_store_commission
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (observed_price : ℝ)
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.1)
  (h3 : observed_price = 19.8) :
  let distributor_price := cost_price * (1 + profit_percentage)
  let commission_percentage := (observed_price / distributor_price - 1) * 100
  commission_percentage = 20 := by
sorry

end online_store_commission_l3013_301355


namespace inequality_solution_l3013_301321

theorem inequality_solution :
  let ineq1 : ℝ → Prop := λ x => x > 1
  let ineq2 : ℝ → Prop := λ x => x > 4
  let ineq3 : ℝ → Prop := λ x => 2 - x > -1
  let ineq4 : ℝ → Prop := λ x => x < 2
  (∀ x : ℤ, (ineq1 x ∧ ineq3 x) ↔ x = 2) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq2 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq3 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq3 x ∧ ineq4 x ∧ x = 2)) :=
by sorry

end inequality_solution_l3013_301321


namespace golden_ratio_cubic_l3013_301358

theorem golden_ratio_cubic (p q : ℚ) : 
  let x : ℝ := (Real.sqrt 5 - 1) / 2
  x^3 + p * x + q = 0 → p + q = -1 := by
sorry

end golden_ratio_cubic_l3013_301358


namespace cube_root_nested_l3013_301341

theorem cube_root_nested (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/3))^(1/3) = N^(13/27) := by
  sorry

end cube_root_nested_l3013_301341


namespace equal_sum_squared_distances_exist_l3013_301372

-- Define a triangle as a tuple of three points in a plane
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define a function to calculate the sum of squared distances from a point to triangle vertices
def sumSquaredDistances (p : ℝ × ℝ) (t : Triangle) : ℝ :=
  let (a, b, c) := t
  (p.1 - a.1)^2 + (p.2 - a.2)^2 +
  (p.1 - b.1)^2 + (p.2 - b.2)^2 +
  (p.1 - c.1)^2 + (p.2 - c.2)^2

-- State the theorem
theorem equal_sum_squared_distances_exist (t1 t2 t3 : Triangle) :
  ∃ (p : ℝ × ℝ), sumSquaredDistances p t1 = sumSquaredDistances p t2 ∧
                 sumSquaredDistances p t2 = sumSquaredDistances p t3 :=
sorry

end equal_sum_squared_distances_exist_l3013_301372


namespace fraction_value_l3013_301353

theorem fraction_value (x y : ℝ) (h1 : -1 < (y - x) / (x + y)) (h2 : (y - x) / (x + y) < 2) 
  (h3 : ∃ n : ℤ, y / x = n) : y / x = 1 := by
  sorry

end fraction_value_l3013_301353


namespace stan_playlist_additional_time_l3013_301370

/-- The number of additional minutes needed for a playlist -/
def additional_minutes_needed (three_minute_songs : ℕ) (two_minute_songs : ℕ) (total_run_time : ℕ) : ℕ :=
  total_run_time - (three_minute_songs * 3 + two_minute_songs * 2)

/-- Theorem: Given Stan's playlist and run time, he needs 40 more minutes of songs -/
theorem stan_playlist_additional_time :
  additional_minutes_needed 10 15 100 = 40 := by
  sorry

#eval additional_minutes_needed 10 15 100

end stan_playlist_additional_time_l3013_301370


namespace equation_solution_l3013_301362

theorem equation_solution :
  let f (x : ℂ) := -x^2 - (4*x + 2)/(x + 2)
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∃ (a b : ℂ), s = {-1, a, b} ∧ a + b = -2 ∧ a * b = 0) :=
by sorry

end equation_solution_l3013_301362


namespace decryption_theorem_l3013_301300

-- Define the encryption functions
def encrypt_a (a : ℤ) : ℤ := a + 1
def encrypt_b (a b : ℤ) : ℤ := 2 * b + a
def encrypt_c (c : ℤ) : ℤ := 3 * c - 4

-- Define the theorem
theorem decryption_theorem (a b c : ℤ) :
  encrypt_a a = 21 ∧ encrypt_b a b = 22 ∧ encrypt_c c = 23 →
  a = 20 ∧ b = 1 ∧ c = 9 := by
  sorry

end decryption_theorem_l3013_301300
