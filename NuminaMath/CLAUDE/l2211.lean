import Mathlib

namespace NUMINAMATH_CALUDE_first_channel_ends_earlier_l2211_221108

/-- Represents the runtime of a film on a TV channel with commercials -/
structure ChannelRuntime where
  segment_length : ℕ
  commercial_length : ℕ
  num_segments : ℕ

/-- Calculates the total runtime for a channel -/
def total_runtime (c : ChannelRuntime) : ℕ :=
  c.segment_length * c.num_segments + c.commercial_length * (c.num_segments - 1)

/-- The theorem to be proved -/
theorem first_channel_ends_earlier (film_length : ℕ) :
  ∃ (n : ℕ), 
    let channel1 := ChannelRuntime.mk 20 2 n
    let channel2 := ChannelRuntime.mk 10 1 (2 * n)
    film_length = 20 * n ∧ 
    film_length = 10 * (2 * n) ∧
    total_runtime channel1 < total_runtime channel2 := by
  sorry

end NUMINAMATH_CALUDE_first_channel_ends_earlier_l2211_221108


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l2211_221131

theorem minimum_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/x + 4/(1 - 2*x)) ≥ 6 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l2211_221131


namespace NUMINAMATH_CALUDE_units_digit_problem_l2211_221138

theorem units_digit_problem : ∃ n : ℕ, 
  33 * 83^1001 * 7^1002 * 13^1003 ≡ 9 [ZMOD 10] ∧ n * 10 + 9 = 33 * 83^1001 * 7^1002 * 13^1003 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2211_221138


namespace NUMINAMATH_CALUDE_cone_base_radius_l2211_221178

/-- Represents a cone with given surface area and net shape -/
structure Cone where
  surfaceArea : ℝ
  netIsSemicircle : Prop

/-- Theorem: Given a cone with surface area 12π cm² and semicircular net, its base radius is 2 cm -/
theorem cone_base_radius (c : Cone) 
  (h1 : c.surfaceArea = 12 * Real.pi) 
  (h2 : c.netIsSemicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r * r * Real.pi + 2 * r * r * Real.pi = c.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2211_221178


namespace NUMINAMATH_CALUDE_pass_percentage_second_set_l2211_221180

theorem pass_percentage_second_set
  (students_set1 : ℕ)
  (students_set2 : ℕ)
  (students_set3 : ℕ)
  (pass_percentage_set1 : ℚ)
  (pass_percentage_set3 : ℚ)
  (overall_pass_percentage : ℚ)
  (h1 : students_set1 = 40)
  (h2 : students_set2 = 50)
  (h3 : students_set3 = 60)
  (h4 : pass_percentage_set1 = 100)
  (h5 : pass_percentage_set3 = 80)
  (h6 : overall_pass_percentage = 88.66666666666667)
  : ∃ (pass_percentage_set2 : ℚ),
    pass_percentage_set2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_pass_percentage_second_set_l2211_221180


namespace NUMINAMATH_CALUDE_glasses_fraction_after_tripling_l2211_221101

theorem glasses_fraction_after_tripling (n : ℝ) (h : n > 0) :
  let initial_with_glasses := (2/3 : ℝ) * n
  let initial_without_glasses := (1/3 : ℝ) * n
  let new_without_glasses := 3 * initial_without_glasses
  let new_total := initial_with_glasses + new_without_glasses
  (initial_with_glasses / new_total) = (2/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_glasses_fraction_after_tripling_l2211_221101


namespace NUMINAMATH_CALUDE_community_average_age_l2211_221185

theorem community_average_age 
  (k : ℕ) 
  (h_k : k > 0) 
  (women : ℕ := 7 * k) 
  (men : ℕ := 8 * k) 
  (women_avg_age : ℚ := 30) 
  (men_avg_age : ℚ := 35) : 
  (women_avg_age * women + men_avg_age * men) / (women + men) = 98 / 3 := by
sorry

end NUMINAMATH_CALUDE_community_average_age_l2211_221185


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2211_221198

/-- A point P with coordinates (m+3, m-2) lies on the x-axis if and only if its coordinates are (5,0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 2 = 0 ∧ (m + 3, m - 2).1 = m + 3 ∧ (m + 3, m - 2).2 = m - 2) ↔ 
  (m + 3, m - 2) = (5, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2211_221198


namespace NUMINAMATH_CALUDE_percentage_of_indian_women_l2211_221149

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (indian_men_percentage : ℚ) (indian_children_percentage : ℚ) (non_indian_percentage : ℚ)
  (h_total_men : total_men = 700)
  (h_total_women : total_women = 500)
  (h_total_children : total_children = 800)
  (h_indian_men : indian_men_percentage = 20 / 100)
  (h_indian_children : indian_children_percentage = 10 / 100)
  (h_non_indian : non_indian_percentage = 79 / 100) :
  (((1 - non_indian_percentage) * (total_men + total_women + total_children)
    - indian_men_percentage * total_men
    - indian_children_percentage * total_children)
   / total_women) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_women_l2211_221149


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_is_correct_l2211_221129

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 882

theorem largest_even_digit_multiple_of_9_is_correct :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 1000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_is_correct_l2211_221129


namespace NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l2211_221118

theorem z_squared_in_second_quadrant (z : ℂ) :
  z = Complex.exp (75 * Real.pi / 180 * Complex.I) →
  Complex.arg (z^2) > Real.pi / 2 ∧ Complex.arg (z^2) < Real.pi :=
sorry

end NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l2211_221118


namespace NUMINAMATH_CALUDE_train_length_is_400_l2211_221110

/-- Calculates the length of a train given its speed, the speed and length of a platform
    moving in the opposite direction, and the time taken to cross the platform. -/
def trainLength (trainSpeed : ℝ) (platformSpeed : ℝ) (platformLength : ℝ) (crossingTime : ℝ) : ℝ :=
  (trainSpeed + platformSpeed) * crossingTime - platformLength

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_is_400 :
  let trainSpeed : ℝ := 20
  let platformSpeed : ℝ := 5
  let platformLength : ℝ := 250
  let crossingTime : ℝ := 26
  trainLength trainSpeed platformSpeed platformLength crossingTime = 400 := by
sorry

end NUMINAMATH_CALUDE_train_length_is_400_l2211_221110


namespace NUMINAMATH_CALUDE_max_crates_third_trip_l2211_221113

/-- The weight of each crate in kilograms -/
def crate_weight : ℝ := 1250

/-- The maximum weight capacity of the trailer in kilograms -/
def max_weight : ℝ := 6250

/-- The number of crates on the first trip -/
def first_trip_crates : ℕ := 3

/-- The number of crates on the second trip -/
def second_trip_crates : ℕ := 4

/-- Theorem: The maximum number of crates that can be carried on the third trip is 5 -/
theorem max_crates_third_trip :
  ∃ (x : ℕ), x ≤ 5 ∧
  (∀ y : ℕ, y > x → y * crate_weight > max_weight) ∧
  x * crate_weight ≤ max_weight :=
sorry

end NUMINAMATH_CALUDE_max_crates_third_trip_l2211_221113


namespace NUMINAMATH_CALUDE_game_ends_in_three_rounds_l2211_221115

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 12
    | Player.B => 11
    | Player.C => 10
    | Player.D => 9 }

/-- Determines if the game has ended -/
def game_ended (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry  -- Implementation details omitted

/-- The number of rounds played before the game ends -/
def rounds_played (state : GameState) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem stating that the game ends after exactly 3 rounds -/
theorem game_ends_in_three_rounds :
  rounds_played initial_state = 3 :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_three_rounds_l2211_221115


namespace NUMINAMATH_CALUDE_thabo_owns_280_books_l2211_221187

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 55
  paperback_nonfiction := 55 + 20
  paperback_fiction := 2 * (55 + 20)

/-- The total number of books in a collection -/
def total_books (bc : BookCollection) : ℕ :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction

/-- Theorem stating that Thabo owns 280 books in total -/
theorem thabo_owns_280_books : total_books thabos_books = 280 := by
  sorry

end NUMINAMATH_CALUDE_thabo_owns_280_books_l2211_221187


namespace NUMINAMATH_CALUDE_weight_of_b_l2211_221186

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l2211_221186


namespace NUMINAMATH_CALUDE_part_one_part_two_l2211_221152

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b

def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := by
  sorry

-- Theorem for part (2)
theorem part_two : (∀ a b : ℝ, ∃ c : ℝ, 2 * A a b - B a b = c) → (∀ a : ℝ, 2 * A a 2 - B a 2 = 2 * A 0 2 - B 0 2) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2211_221152


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2211_221171

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the slope of the line parallel to 3x + y = 0
def k : ℝ := -3

-- Define the point of tangency
def x₀ : ℝ := 1
def y₀ : ℝ := f x₀

-- State the theorem
theorem tangent_line_equation :
  ∃ (x y : ℝ), 3*x + y - 1 = 0 ∧
  y - y₀ = k * (x - x₀) ∧
  f' x₀ = k ∧
  y₀ = f x₀ := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2211_221171


namespace NUMINAMATH_CALUDE_john_total_weight_l2211_221175

def john_workout (initial_back_squat front_squat_percentage back_squat_increase triple_percentage : ℝ) : ℝ :=
  let new_back_squat := initial_back_squat + back_squat_increase
  let front_squat := front_squat_percentage * new_back_squat
  let triple_weight := triple_percentage * front_squat
  3 * triple_weight

theorem john_total_weight :
  john_workout 200 0.8 50 0.9 = 540 := by
  sorry

end NUMINAMATH_CALUDE_john_total_weight_l2211_221175


namespace NUMINAMATH_CALUDE_intersection_M_N_l2211_221194

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2211_221194


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2211_221172

/-- Given integers a, b, c satisfying the equation (x - a)(x - 5) + 1 = (x + b)(x + c)
    and either (b + 5)(c + 5) = 1 or (b + 5)(c + 5) = 4,
    prove that the possible values of a are 2, 3, 4, and 7. -/
theorem possible_values_of_a (a b c : ℤ) 
  (h1 : ∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c))
  (h2 : (b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) :
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 := by
  sorry


end NUMINAMATH_CALUDE_possible_values_of_a_l2211_221172


namespace NUMINAMATH_CALUDE_caroline_lassis_l2211_221167

/-- Given that Caroline can make 15 lassis from 3 mangoes, prove that she can make 90 lassis from 18 mangoes. -/
theorem caroline_lassis (mangoes_small : ℕ) (lassis_small : ℕ) (mangoes_large : ℕ) :
  mangoes_small = 3 →
  lassis_small = 15 →
  mangoes_large = 18 →
  (lassis_small * mangoes_large) / mangoes_small = 90 :=
by sorry

end NUMINAMATH_CALUDE_caroline_lassis_l2211_221167


namespace NUMINAMATH_CALUDE_prime_solution_equation_l2211_221173

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l2211_221173


namespace NUMINAMATH_CALUDE_box_fits_blocks_l2211_221140

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_blocks :
  let box : Dimensions := { height := 8, width := 10, length := 12 }
  let block : Dimensions := { height := 3, width := 2, length := 4 }
  fitCount box block = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_fits_blocks_l2211_221140


namespace NUMINAMATH_CALUDE_cow_manure_plant_height_cow_manure_plant_height_is_90_l2211_221199

theorem cow_manure_plant_height : ℝ → Prop :=
  fun height_cow_manure =>
    let height_control : ℝ := 36
    let height_bone_meal : ℝ := 1.25 * height_control
    height_cow_manure = 2 * height_bone_meal

-- The proof
theorem cow_manure_plant_height_is_90 : cow_manure_plant_height 90 := by
  sorry

end NUMINAMATH_CALUDE_cow_manure_plant_height_cow_manure_plant_height_is_90_l2211_221199


namespace NUMINAMATH_CALUDE_container_volume_increase_three_gallon_to_twentyfour_gallon_l2211_221144

theorem container_volume_increase (original_volume : ℝ) (scale_factor : ℝ) :
  original_volume > 0 →
  scale_factor = 2 →
  scale_factor * scale_factor * scale_factor * original_volume = 8 * original_volume :=
by sorry

theorem three_gallon_to_twentyfour_gallon :
  let original_volume : ℝ := 3
  let scale_factor : ℝ := 2
  let new_volume : ℝ := scale_factor * scale_factor * scale_factor * original_volume
  new_volume = 24 :=
by sorry

end NUMINAMATH_CALUDE_container_volume_increase_three_gallon_to_twentyfour_gallon_l2211_221144


namespace NUMINAMATH_CALUDE_factor_theorem_quadratic_l2211_221148

theorem factor_theorem_quadratic (k : ℚ) : 
  (∀ m : ℚ, (m - 8) ∣ (m^2 - k*m - 24)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_quadratic_l2211_221148


namespace NUMINAMATH_CALUDE_rectangle_area_18_l2211_221143

def rectangle_pairs : Set (ℕ × ℕ) :=
  {p | p.1 * p.2 = 18 ∧ p.1 < p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem rectangle_area_18 :
  rectangle_pairs = {(1, 18), (2, 9), (3, 6)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l2211_221143


namespace NUMINAMATH_CALUDE_work_completion_time_l2211_221109

/-- Represents the time it takes for a worker to complete a task alone -/
structure WorkTime where
  days : ℝ
  work_rate : ℝ
  inv_days_eq_work_rate : work_rate = 1 / days

/-- Represents a work scenario with two workers -/
structure WorkScenario where
  x : WorkTime
  y : WorkTime
  total_days : ℝ
  x_solo_days : ℝ
  both_days : ℝ
  total_days_eq_sum : total_days = x_solo_days + both_days

/-- The theorem to be proved -/
theorem work_completion_time (w : WorkScenario) (h1 : w.y.days = 12) 
  (h2 : w.x_solo_days = 4) (h3 : w.total_days = 10) : w.x.days = 20 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l2211_221109


namespace NUMINAMATH_CALUDE_max_value_expression_l2211_221163

theorem max_value_expression (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x ≤ 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ∧
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x = 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ↔ x = Real.sqrt a :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2211_221163


namespace NUMINAMATH_CALUDE_anas_dresses_l2211_221133

theorem anas_dresses (ana lisa : ℕ) : 
  lisa = ana + 18 → 
  ana + lisa = 48 → 
  ana = 15 := by
sorry

end NUMINAMATH_CALUDE_anas_dresses_l2211_221133


namespace NUMINAMATH_CALUDE_triangle_area_is_64_l2211_221150

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 -/
def triangleArea : ℝ := 64

/-- The first bounding line of the triangle -/
def line1 (x : ℝ) : ℝ := x

/-- The second bounding line of the triangle -/
def line2 (x : ℝ) : ℝ := -x

/-- The third bounding line of the triangle -/
def line3 : ℝ := 8

theorem triangle_area_is_64 :
  triangleArea = (1/2) * (line3 - line1 0) * (line3 - line2 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_64_l2211_221150


namespace NUMINAMATH_CALUDE_wand_price_l2211_221114

theorem wand_price (P : ℚ) : (P * (1/8) = 8) → P = 64 := by
  sorry

end NUMINAMATH_CALUDE_wand_price_l2211_221114


namespace NUMINAMATH_CALUDE_remainder_problem_l2211_221142

theorem remainder_problem (N : ℤ) : N % 296 = 75 → N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2211_221142


namespace NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_l2211_221116

theorem tan_fifteen_degree_fraction : 
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_l2211_221116


namespace NUMINAMATH_CALUDE_range_of_a_l2211_221135

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 + 1 ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 1 = 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(p a) ∨ ¬(q a))) → (a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2211_221135


namespace NUMINAMATH_CALUDE_evaluate_expression_l2211_221106

theorem evaluate_expression (a x : ℝ) (h : x = a + 9) : x - a + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2211_221106


namespace NUMINAMATH_CALUDE_playhouse_siding_cost_l2211_221130

/-- Calculates the cost of siding for a playhouse with given dimensions --/
theorem playhouse_siding_cost
  (wall_width : ℝ)
  (wall_height : ℝ)
  (roof_width : ℝ)
  (roof_height : ℝ)
  (siding_width : ℝ)
  (siding_height : ℝ)
  (siding_cost : ℝ)
  (h_wall_width : wall_width = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_width : roof_width = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_width : siding_width = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35) :
  ⌈(wall_width * wall_height + 2 * roof_width * roof_height) / (siding_width * siding_height)⌉ * siding_cost = 70 :=
by sorry

end NUMINAMATH_CALUDE_playhouse_siding_cost_l2211_221130


namespace NUMINAMATH_CALUDE_keith_bought_22_cards_l2211_221146

/-- The number of baseball cards Keith bought -/
def cards_bought (initial_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem stating that Keith bought 22 baseball cards -/
theorem keith_bought_22_cards : cards_bought 40 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_keith_bought_22_cards_l2211_221146


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l2211_221139

/-- Given a triangle with sides 40, 60, and 80 units, prove that the larger segment
    cut off by an altitude to the side of length 80 is 52.5 units long. -/
theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 ∧ b = 60 ∧ c = 80 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l2211_221139


namespace NUMINAMATH_CALUDE_discount_effect_l2211_221158

/-- Represents the sales discount as a percentage -/
def discount : ℝ := 10

/-- Represents the increase in the number of items sold as a percentage -/
def items_increase : ℝ := 30

/-- Represents the increase in gross income as a percentage -/
def income_increase : ℝ := 17

theorem discount_effect (P N : ℝ) (h₁ : P > 0) (h₂ : N > 0) : 
  P * (1 - discount / 100) * N * (1 + items_increase / 100) = 
  P * N * (1 + income_increase / 100) := by
  sorry

#check discount_effect

end NUMINAMATH_CALUDE_discount_effect_l2211_221158


namespace NUMINAMATH_CALUDE_line_through_points_l2211_221161

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The theorem statement -/
theorem line_through_points :
  let p1 : Point := ⟨2, 0⟩
  let p2 : Point := ⟨0, -3⟩
  let l : Line := ⟨1/2, -1/3, -1⟩
  pointOnLine p1 l ∧ pointOnLine p2 l :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l2211_221161


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l2211_221151

/-- Represents the teaching experience of the four teachers --/
structure TeachingExperience where
  james : ℕ
  sarah : ℕ
  robert : ℕ
  emily : ℕ

/-- Calculates the combined teaching experience --/
def combined_experience (te : TeachingExperience) : ℕ :=
  te.james + te.sarah + te.robert + te.emily

/-- Theorem stating the combined teaching experience --/
theorem combined_teaching_experience :
  ∃ (te : TeachingExperience),
    te.james = 40 ∧
    te.sarah = te.james - 10 ∧
    te.robert = 2 * te.sarah ∧
    te.emily = 3 * te.sarah - 5 ∧
    combined_experience te = 215 := by
  sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l2211_221151


namespace NUMINAMATH_CALUDE_gravel_bags_per_truckload_l2211_221159

/-- Represents the roadwork company's asphalt paving problem -/
def roadwork_problem (road_length : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ)
  (day1_miles : ℕ) (day2_miles : ℕ) (remaining_pitch : ℕ) : Prop :=
  let total_paved := day1_miles + day2_miles
  let remaining_miles := road_length - total_paved
  let remaining_truckloads := remaining_miles * truckloads_per_mile
  let pitch_per_truckload : ℚ := remaining_pitch / remaining_truckloads
  let gravel_per_truckload := gravel_pitch_ratio * pitch_per_truckload
  gravel_per_truckload = 2

/-- The main theorem stating that the number of bags of gravel per truckload is 2 -/
theorem gravel_bags_per_truckload :
  roadwork_problem 16 5 3 4 7 6 :=
sorry

end NUMINAMATH_CALUDE_gravel_bags_per_truckload_l2211_221159


namespace NUMINAMATH_CALUDE_factorization_equality_l2211_221189

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y = 3 * y * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2211_221189


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2211_221128

/-- Given a regular polygon where each exterior angle measures 40°,
    prove that the sum of its interior angles is 1260°. -/
theorem sum_interior_angles_regular_polygon :
  ∀ (n : ℕ), n > 2 →
  (360 : ℝ) / (40 : ℝ) = n →
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2211_221128


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l2211_221181

theorem x_plus_2y_equals_10 (x y : ℝ) (eq1 : x + y = 19) (eq2 : x + 3*y = 1) : 
  x + 2*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l2211_221181


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l2211_221112

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time
  (num_blacksmiths : ℕ)
  (num_horses : ℕ)
  (time_per_hoof : ℕ)
  (hooves_per_horse : ℕ)
  (h1 : num_blacksmiths = 48)
  (h2 : num_horses = 60)
  (h3 : time_per_hoof = 5)
  (h4 : hooves_per_horse = 4) :
  (num_horses * hooves_per_horse * time_per_hoof) / num_blacksmiths = 25 := by
  sorry

#eval (60 * 4 * 5) / 48  -- Should output 25

end NUMINAMATH_CALUDE_minimum_shoeing_time_l2211_221112


namespace NUMINAMATH_CALUDE_g_minus_one_eq_zero_l2211_221111

/-- The function g(x) as defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

/-- Theorem stating that g(-1) = 0 when s = -4 -/
theorem g_minus_one_eq_zero :
  g (-1) (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_one_eq_zero_l2211_221111


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l2211_221156

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relationships
def skew (l1 l2 : Line) : Prop := sorry
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def oblique_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ (p1 p2 : Plane) (l1 l2 : Line), 
  p1 ≠ p2 → in_plane l1 p1 → in_plane l2 p2 → skew l1 l2

def prop2 : Prop := ∀ (p1 p2 : Plane) (l : Line), 
  oblique_to_plane l p1 → 
  (perpendicular p1 p2 ∧ in_plane l p2) → 
  ∀ (p3 : Plane), perpendicular p1 p3 ∧ in_plane l p3 → p2 = p3

def prop3 : Prop := ∀ (p1 p2 p3 : Plane), 
  perpendicular p1 p2 → perpendicular p1 p3 → parallel p2 p3

-- Theorem statement
theorem exactly_one_true_proposition : 
  (prop1 = False ∧ prop2 = True ∧ prop3 = False) := by sorry

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l2211_221156


namespace NUMINAMATH_CALUDE_largest_perimeter_right_triangle_l2211_221145

theorem largest_perimeter_right_triangle (x : ℕ) : 
  -- Right triangle condition (using Pythagorean theorem)
  x * x ≤ 8 * 8 + 9 * 9 →
  -- Triangle inequality conditions
  8 + 9 > x →
  8 + x > 9 →
  9 + x > 8 →
  -- Perimeter definition
  let perimeter := 8 + 9 + x
  -- Theorem statement
  perimeter ≤ 29 ∧ ∃ y : ℕ, y * y ≤ 8 * 8 + 9 * 9 ∧ 8 + 9 > y ∧ 8 + y > 9 ∧ 9 + y > 8 ∧ 8 + 9 + y = 29 :=
by sorry

end NUMINAMATH_CALUDE_largest_perimeter_right_triangle_l2211_221145


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l2211_221193

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l2211_221193


namespace NUMINAMATH_CALUDE_weight_to_lose_in_may_l2211_221165

/-- Given Michael's weight loss goal and the amounts he lost in March and April,
    prove that the weight he needs to lose in May is the difference between
    his goal and the sum of weight lost in March and April. -/
theorem weight_to_lose_in_may
  (total_goal : ℕ)
  (march_loss : ℕ)
  (april_loss : ℕ)
  (may_loss : ℕ)
  (h1 : total_goal = 10)
  (h2 : march_loss = 3)
  (h3 : april_loss = 4)
  (h4 : may_loss = total_goal - (march_loss + april_loss)) :
  may_loss = 3 :=
by sorry

end NUMINAMATH_CALUDE_weight_to_lose_in_may_l2211_221165


namespace NUMINAMATH_CALUDE_parabola_directrix_l2211_221179

/-- The directrix of the parabola y = (x^2 - 8x + 12) / 16 is y = -17/4 -/
theorem parabola_directrix : 
  let f : ℝ → ℝ := λ x => (x^2 - 8*x + 12) / 16
  ∃ (a h k : ℝ), 
    (∀ x, f x = a * (x - h)^2 + k) ∧ 
    (k - 1 / (4*a) = -17/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2211_221179


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2211_221100

-- Define a triangle ABC
structure Triangle where
  a : Real
  b : Real
  c : Real
  A : Real
  B : Real
  C : Real

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (hb : t.b = Real.sqrt 3)
  (hc : t.c = 1)
  (hB : t.B = 60 * π / 180) : -- Convert 60° to radians
  t.a = 2 ∧ t.A = π / 2 ∧ t.C = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2211_221100


namespace NUMINAMATH_CALUDE_log_stack_sum_l2211_221127

/-- Sum of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

theorem log_stack_sum :
  let a₁ : ℕ := 5  -- first term (top row)
  let aₙ : ℕ := 15 -- last term (bottom row)
  let n : ℕ := aₙ - a₁ + 1 -- number of terms
  sum_arithmetic_sequence a₁ aₙ n = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2211_221127


namespace NUMINAMATH_CALUDE_total_students_count_l2211_221170

/-- The number of students wishing to go on a scavenger hunting trip -/
def scavenger_hunting : ℕ := 4000

/-- The number of students wishing to go on a skiing trip -/
def skiing : ℕ := 2 * scavenger_hunting

/-- The number of students wishing to go on a camping trip -/
def camping : ℕ := skiing + (skiing * 15 / 100)

/-- The total number of students wishing to go on any trip -/
def total_students : ℕ := scavenger_hunting + skiing + camping

theorem total_students_count : total_students = 21200 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l2211_221170


namespace NUMINAMATH_CALUDE_rachel_essay_editing_time_l2211_221190

/-- Rachel's essay writing problem -/
theorem rachel_essay_editing_time 
  (writing_rate : ℕ → ℕ)  -- Function mapping pages to minutes
  (research_time : ℕ)     -- Time spent researching in minutes
  (total_pages : ℕ)       -- Total pages written
  (total_time : ℕ)        -- Total time spent on the essay in minutes
  (h1 : writing_rate 1 = 30)  -- Writing rate is 1 page per 30 minutes
  (h2 : research_time = 45)   -- 45 minutes spent researching
  (h3 : total_pages = 6)      -- 6 pages written in total
  (h4 : total_time = 5 * 60)  -- Total time is 5 hours (300 minutes)
  : total_time - (research_time + writing_rate total_pages) = 75 := by
  sorry

#check rachel_essay_editing_time

end NUMINAMATH_CALUDE_rachel_essay_editing_time_l2211_221190


namespace NUMINAMATH_CALUDE_tank_capacity_l2211_221184

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that the tank's capacity is 30 liters given the conditions -/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l2211_221184


namespace NUMINAMATH_CALUDE_julians_boy_friends_percentage_l2211_221137

theorem julians_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (boyd_boys_percentage : ℚ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_girls_percentage = 40/100)
  (h3 : boyd_total_friends = 100)
  (h4 : boyd_boys_percentage = 36/100)
  (h5 : (boyd_total_friends : ℚ) * (1 - boyd_boys_percentage) = 2 * (julian_total_friends : ℚ) * julian_girls_percentage) :
  (julian_total_friends : ℚ) * (1 - julian_girls_percentage) / julian_total_friends = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_julians_boy_friends_percentage_l2211_221137


namespace NUMINAMATH_CALUDE_segment_length_proof_l2211_221121

theorem segment_length_proof (A B O P M : Real) : 
  -- Conditions
  (0 ≤ A) ∧ (A < O) ∧ (O < M) ∧ (M < P) ∧ (P < B) ∧  -- Points lie on the line segment in order
  (O - A = 4/5 * (B - A)) ∧                           -- AO = 4/5 * AB
  (B - P = 2/3 * (B - A)) ∧                           -- BP = 2/3 * AB
  (M - A = 1/2 * (B - A)) ∧                           -- M is the midpoint of AB
  (M - O = 2) →                                       -- OM = 2
  (P - M = 10/9)                                      -- PM = 10/9

:= by sorry

end NUMINAMATH_CALUDE_segment_length_proof_l2211_221121


namespace NUMINAMATH_CALUDE_photos_per_remaining_page_l2211_221168

theorem photos_per_remaining_page (total_photos : ℕ) (total_pages : ℕ) 
  (first_15_photos : ℕ) (next_15_photos : ℕ) (following_10_photos : ℕ) :
  total_photos = 500 →
  total_pages = 60 →
  first_15_photos = 3 →
  next_15_photos = 4 →
  following_10_photos = 5 →
  (17 : ℕ) = (total_photos - (15 * first_15_photos + 15 * next_15_photos + 10 * following_10_photos)) / (total_pages - 40) :=
by sorry

end NUMINAMATH_CALUDE_photos_per_remaining_page_l2211_221168


namespace NUMINAMATH_CALUDE_library_books_remaining_l2211_221134

/-- The number of books remaining in a library after a series of events --/
def remaining_books (initial : ℕ) (taken_out : ℕ) (returned : ℕ) (withdrawn : ℕ) : ℕ :=
  initial - taken_out + returned - withdrawn

/-- Theorem stating that given the specific events, 150 books remain in the library --/
theorem library_books_remaining : remaining_books 250 120 35 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_library_books_remaining_l2211_221134


namespace NUMINAMATH_CALUDE_shelter_dogs_l2211_221153

theorem shelter_dogs (dogs cats : ℕ) : 
  dogs * 7 = cats * 15 → 
  dogs * 11 = (cats + 8) * 15 → 
  dogs = 30 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l2211_221153


namespace NUMINAMATH_CALUDE_debbie_tape_usage_l2211_221107

/-- The amount of tape needed to pack boxes of different sizes --/
def total_tape_used (large_boxes medium_boxes small_boxes : ℕ) : ℕ :=
  let large_tape := 5 * large_boxes  -- 4 feet for sealing + 1 foot for label
  let medium_tape := 3 * medium_boxes  -- 2 feet for sealing + 1 foot for label
  let small_tape := 2 * small_boxes  -- 1 foot for sealing + 1 foot for label
  large_tape + medium_tape + small_tape

/-- Theorem stating that Debbie used 44 feet of tape --/
theorem debbie_tape_usage : total_tape_used 2 8 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_debbie_tape_usage_l2211_221107


namespace NUMINAMATH_CALUDE_playlist_repetitions_l2211_221160

def song1_duration : ℕ := 3
def song2_duration : ℕ := 2
def song3_duration : ℕ := 3
def ride_duration : ℕ := 40

def playlist_duration : ℕ := song1_duration + song2_duration + song3_duration

theorem playlist_repetitions :
  ride_duration / playlist_duration = 5 := by sorry

end NUMINAMATH_CALUDE_playlist_repetitions_l2211_221160


namespace NUMINAMATH_CALUDE_division_problem_l2211_221154

theorem division_problem (divisor quotient remainder dividend : ℕ) 
  (h1 : divisor = 21)
  (h2 : remainder = 7)
  (h3 : dividend = 301)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 14 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2211_221154


namespace NUMINAMATH_CALUDE_rowing_distance_l2211_221182

/-- The distance to a destination given rowing conditions and round trip time -/
theorem rowing_distance (v : ℝ) (w : ℝ) (c : ℝ) (t : ℝ) (h1 : v > 0) (h2 : w ≥ 0) (h3 : c ≥ 0) (h4 : t > 0) :
  let d := (t * (v + c) * (v + c - w)) / ((v + c) + (v + c - w))
  d = 45 / 11 ↔ v = 4 ∧ w = 1 ∧ c = 2 ∧ t = 3/2 := by
  sorry

#check rowing_distance

end NUMINAMATH_CALUDE_rowing_distance_l2211_221182


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l2211_221119

theorem largest_x_abs_equation : ∃ (x_max : ℝ), (∀ (x : ℝ), |x - 5| = 12 → x ≤ x_max) ∧ |x_max - 5| = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l2211_221119


namespace NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2211_221196

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2001_and_1015_l2211_221196


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l2211_221136

/-- Given a triangle ABC with points D and E as described, prove that the intersection P of BE and AD
    has the vector representation P = (8/14)A + (1/14)B + (4/14)C -/
theorem intersection_point_coordinates (A B C D E P : ℝ × ℝ) : 
  (∃ (k : ℝ), D = k • C + (1 - k) • B ∧ k = 5/4) →  -- BD:DC = 4:1
  (∃ (m : ℝ), E = m • A + (1 - m) • C ∧ m = 2/3) →  -- AE:EC = 2:1
  (∃ (t : ℝ), P = t • A + (1 - t) • D) →            -- P is on AD
  (∃ (s : ℝ), P = s • B + (1 - s) • E) →            -- P is on BE
  (∃ (x y z : ℝ), P = x • A + y • B + z • C ∧ x + y + z = 1) →
  P = (8/14) • A + (1/14) • B + (4/14) • C :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_coordinates_l2211_221136


namespace NUMINAMATH_CALUDE_inequality_proof_l2211_221191

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2211_221191


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2211_221126

theorem degree_to_radian_conversion (π : Real) : 
  (180 : Real) = π → (750 : Real) = (25 / 6 : Real) * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2211_221126


namespace NUMINAMATH_CALUDE_housing_boom_result_l2211_221166

/-- Represents the housing development in Lawrence County -/
structure LawrenceCountyHousing where
  initial_houses : ℕ
  developer_a_rate : ℕ
  developer_a_months : ℕ
  developer_b_rate : ℕ
  developer_b_months : ℕ
  developer_c_rate : ℕ
  developer_c_months : ℕ
  final_houses : ℕ

/-- Calculates the total number of houses built by developers -/
def total_houses_built (h : LawrenceCountyHousing) : ℕ :=
  h.developer_a_rate * h.developer_a_months +
  h.developer_b_rate * h.developer_b_months +
  h.developer_c_rate * h.developer_c_months

/-- Theorem stating that the total houses built by developers is 405 -/
theorem housing_boom_result (h : LawrenceCountyHousing)
  (h_initial : h.initial_houses = 1426)
  (h_dev_a : h.developer_a_rate = 25 ∧ h.developer_a_months = 6)
  (h_dev_b : h.developer_b_rate = 15 ∧ h.developer_b_months = 9)
  (h_dev_c : h.developer_c_rate = 30 ∧ h.developer_c_months = 4)
  (h_final : h.final_houses = 2000) :
  total_houses_built h = 405 := by
  sorry


end NUMINAMATH_CALUDE_housing_boom_result_l2211_221166


namespace NUMINAMATH_CALUDE_total_animals_bought_l2211_221124

/-- The number of guppies Rick bought -/
def rickGuppies : ℕ := 30

/-- The number of clowns Tim bought -/
def timClowns : ℕ := 2 * rickGuppies

/-- The number of tetras I bought -/
def myTetras : ℕ := 4 * timClowns

/-- The total number of animals bought -/
def totalAnimals : ℕ := myTetras + timClowns + rickGuppies

theorem total_animals_bought : totalAnimals = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_bought_l2211_221124


namespace NUMINAMATH_CALUDE_square_rotation_overlap_area_l2211_221177

theorem square_rotation_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < π/2) (h3 : Real.sin β = 3/5) :
  let side_length : Real := 2
  let overlap_area := 2 * (1/2 * side_length * (side_length * ((1 - Real.tan (β/2)) / (1 + Real.tan (β/2)))))
  overlap_area = 2 := by
sorry

end NUMINAMATH_CALUDE_square_rotation_overlap_area_l2211_221177


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2211_221103

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2211_221103


namespace NUMINAMATH_CALUDE_train_problem_solution_l2211_221164

/-- Represents the train problem scenario -/
structure TrainProblem where
  total_distance : ℝ
  train_a_speed : ℝ
  train_b_speed : ℝ
  separation_distance : ℝ

/-- The time when trains are at the given separation distance -/
def separation_time (p : TrainProblem) : Set ℝ :=
  { t : ℝ | t = (p.total_distance - p.separation_distance) / (p.train_a_speed + p.train_b_speed) ∨
             t = (p.total_distance + p.separation_distance) / (p.train_a_speed + p.train_b_speed) }

/-- Theorem stating the solution to the train problem -/
theorem train_problem_solution (p : TrainProblem) 
    (h1 : p.total_distance = 840)
    (h2 : p.train_a_speed = 68.5)
    (h3 : p.train_b_speed = 71.5)
    (h4 : p.separation_distance = 210) :
    separation_time p = {4.5, 7.5} := by
  sorry

end NUMINAMATH_CALUDE_train_problem_solution_l2211_221164


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l2211_221132

-- Define the parameters of the taxi service
def initial_fee : ℚ := 2.05
def charge_per_increment : ℚ := 0.35
def miles_per_increment : ℚ := 2 / 5
def trip_distance : ℚ := 3.6

-- Define the function to calculate the total charge
def total_charge (init_fee charge_per_incr miles_per_incr dist : ℚ) : ℚ :=
  init_fee + charge_per_incr * (dist / miles_per_incr)

-- Theorem statement
theorem taxi_charge_proof :
  total_charge initial_fee charge_per_increment miles_per_increment trip_distance = 5.20 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_proof_l2211_221132


namespace NUMINAMATH_CALUDE_unique_solution_system_l2211_221120

theorem unique_solution_system (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! (x y z : ℝ), 
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧
    y = (c + a) / 2 ∧
    z = (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2211_221120


namespace NUMINAMATH_CALUDE_min_production_cost_l2211_221192

/-- Raw material requirements for products A and B --/
structure RawMaterial where
  a : ℕ  -- kg of material A required
  b : ℕ  -- kg of material B required

/-- Available raw materials and production constraints --/
structure ProductionConstraints where
  total_units : ℕ        -- Total units to be produced
  available_a : ℕ        -- Available kg of material A
  available_b : ℕ        -- Available kg of material B
  product_a : RawMaterial  -- Raw material requirements for product A
  product_b : RawMaterial  -- Raw material requirements for product B

/-- Cost information for products --/
structure CostInfo where
  cost_a : ℕ  -- Cost per unit of product A
  cost_b : ℕ  -- Cost per unit of product B

/-- Main theorem stating the minimum production cost --/
theorem min_production_cost 
  (constraints : ProductionConstraints)
  (costs : CostInfo)
  (h_constraints : constraints.total_units = 50 ∧ 
                   constraints.available_a = 360 ∧ 
                   constraints.available_b = 290 ∧
                   constraints.product_a = ⟨9, 4⟩ ∧
                   constraints.product_b = ⟨3, 10⟩)
  (h_costs : costs.cost_a = 70 ∧ costs.cost_b = 90) :
  ∃ (x : ℕ), x = 32 ∧ 
    (constraints.total_units - x) = 18 ∧
    costs.cost_a * x + costs.cost_b * (constraints.total_units - x) = 3860 :=
sorry

end NUMINAMATH_CALUDE_min_production_cost_l2211_221192


namespace NUMINAMATH_CALUDE_vectors_form_basis_l2211_221104

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (1, -2) ∧ b = (3, 5) → 
  (∃ (x y : ℝ), ∀ v : ℝ × ℝ, v = x • a + y • b) ∧ 
  (¬ ∃ (k : ℝ), a = k • b) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l2211_221104


namespace NUMINAMATH_CALUDE_range_of_m_l2211_221102

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define the theorem
theorem range_of_m (f : ℝ → ℝ) (h_odd : is_odd f) 
  (h_decreasing : is_monotone_decreasing f 0 2)
  (h_domain : ∀ x, x ∈ Set.Icc (-2) 2 → f x ≠ 0) :
  ∀ m, (f m + f (m - 1) > 0) → m ∈ Set.Ioc (-1) (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2211_221102


namespace NUMINAMATH_CALUDE_gcd_1617_1225_l2211_221195

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 := by sorry

end NUMINAMATH_CALUDE_gcd_1617_1225_l2211_221195


namespace NUMINAMATH_CALUDE_root_difference_ratio_l2211_221155

noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 3
noncomputable def f₂ (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - b
noncomputable def f₃ (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + (2 - 2*a)*x - 6 - b
noncomputable def f₄ (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + (4 - a)*x - 3 - 2*b

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (a^2 + 12)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (4 + 4*b)
noncomputable def C (a b : ℝ) : ℝ := Real.sqrt ((2 - 2*a)^2 + 12*(6 + b)) / 3
noncomputable def D (a b : ℝ) : ℝ := Real.sqrt ((4 - a)^2 + 12*(3 + 2*b)) / 3

theorem root_difference_ratio (a b : ℝ) (h : C a b ≠ D a b) :
  (A a ^ 2 - B b ^ 2) / (C a b ^ 2 - D a b ^ 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l2211_221155


namespace NUMINAMATH_CALUDE_only_two_satisfies_condition_l2211_221147

def is_quadratic_residue (a p : ℕ) : Prop :=
  ∃ x, x^2 ≡ a [MOD p]

def all_quadratic_residues (p : ℕ) : Prop :=
  ∀ k ∈ Finset.range p, is_quadratic_residue (2 * (p / k) - 1) p

theorem only_two_satisfies_condition :
  ∀ p, Nat.Prime p → (all_quadratic_residues p ↔ p = 2) := by sorry

end NUMINAMATH_CALUDE_only_two_satisfies_condition_l2211_221147


namespace NUMINAMATH_CALUDE_solid_is_frustum_l2211_221105

/-- A solid with specified view characteristics -/
structure Solid where
  top_view : Bool
  bottom_view : Bool
  front_view : Bool
  side_view : Bool

/-- Definition of a frustum based on its views -/
def is_frustum (s : Solid) : Prop :=
  s.top_view = true ∧ 
  s.bottom_view = true ∧ 
  s.front_view = true ∧ 
  s.side_view = true

/-- Theorem: A solid with circular top and bottom views, and trapezoidal front and side views, is a frustum -/
theorem solid_is_frustum (s : Solid) 
  (h_top : s.top_view = true)
  (h_bottom : s.bottom_view = true)
  (h_front : s.front_view = true)
  (h_side : s.side_view = true) : 
  is_frustum s := by
  sorry

end NUMINAMATH_CALUDE_solid_is_frustum_l2211_221105


namespace NUMINAMATH_CALUDE_parabola_above_l2211_221188

theorem parabola_above (k : ℝ) : 
  (∀ x : ℝ, 2*x^2 - 2*k*x + (k^2 + 2*k + 2) > x^2 + 2*k*x - 2*k^2 - 1) ↔ 
  (-1 < k ∧ k < 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_above_l2211_221188


namespace NUMINAMATH_CALUDE_intersection_M_N_l2211_221125

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2211_221125


namespace NUMINAMATH_CALUDE_front_yard_eggs_count_l2211_221123

/-- The number of eggs in June's front yard nest -/
def front_yard_eggs : ℕ := sorry

/-- The total number of eggs June found -/
def total_eggs : ℕ := 17

/-- The number of nests in the first tree -/
def nests_in_first_tree : ℕ := 2

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_first_tree : ℕ := 5

/-- The number of nests in the second tree -/
def nests_in_second_tree : ℕ := 1

/-- The number of eggs in the nest in the second tree -/
def eggs_in_second_tree : ℕ := 3

theorem front_yard_eggs_count :
  front_yard_eggs = total_eggs - (nests_in_first_tree * eggs_per_nest_first_tree + nests_in_second_tree * eggs_in_second_tree) :=
by sorry

end NUMINAMATH_CALUDE_front_yard_eggs_count_l2211_221123


namespace NUMINAMATH_CALUDE_equation_solution_l2211_221122

theorem equation_solution :
  let f (x : ℂ) := -x^2 * (x + 2) - (2 * x + 4)
  ∀ x : ℂ, x ≠ -2 → (f x = 0 ↔ x = -2 ∨ x = 2*I ∨ x = -2*I) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2211_221122


namespace NUMINAMATH_CALUDE_equal_distance_point_exists_l2211_221169

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (P Q O A : Plane)

-- Define the speed
variable (v : ℝ)

-- Define the distance function
variable (dist : Plane → Plane → ℝ)

-- Define the time
variable (t : ℝ)

-- Define the lines as functions of time
variable (line_P line_Q : ℝ → Plane)

-- State the theorem
theorem equal_distance_point_exists :
  (∀ t, dist O (line_P t) = v * t) →  -- P moves with constant speed v
  (∀ t, dist O (line_Q t) = v * t) →  -- Q moves with constant speed v
  (∃ t₀, line_P t₀ = O ∧ line_Q t₀ = O) →  -- The lines intersect at O
  ∃ A : Plane, ∀ t, dist A (line_P t) = dist A (line_Q t) :=
by sorry

end NUMINAMATH_CALUDE_equal_distance_point_exists_l2211_221169


namespace NUMINAMATH_CALUDE_rental_cost_equality_l2211_221183

/-- Represents the rental cost scenario for two computers -/
structure RentalCost where
  B : ℝ  -- Hourly rate for computer B
  T : ℝ  -- Time taken by computer A to complete the job

/-- The total cost is the same for both computers and equals 70 times the hourly rate of computer B -/
theorem rental_cost_equality (rc : RentalCost) : 
  1.4 * rc.B * rc.T = rc.B * (rc.T + 20) ∧ 
  1.4 * rc.B * rc.T = 70 * rc.B :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l2211_221183


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2211_221157

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2211_221157


namespace NUMINAMATH_CALUDE_bus_problem_l2211_221162

theorem bus_problem (initial_students : ℕ) (remaining_fraction : ℚ) : 
  initial_students = 64 →
  remaining_fraction = 2/3 →
  (initial_students : ℚ) * remaining_fraction^3 = 512/27 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2211_221162


namespace NUMINAMATH_CALUDE_last_locker_opened_l2211_221141

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : Prop :=
  ∃ (lastLocker : ℕ),
    -- There are 2048 lockers
    n = 2048 ∧
    -- The last locker opened is 2041
    lastLocker = 2041 ∧
    -- The pattern follows the described rules
    (∀ k : ℕ, k ≤ n → ∃ (trip : ℕ),
      -- Each trip opens lockers based on the trip number
      (k % trip = 0 → k ≠ lastLocker) ∧
      -- The last locker is only opened in the final trip
      (k = lastLocker → ∀ j < trip, k % j ≠ 0))

/-- Theorem stating that the last locker opened is 2041 -/
theorem last_locker_opened (n : ℕ) (h : lockerOpeningPattern n) :
  ∃ (lastLocker : ℕ), lastLocker = 2041 ∧ 
  ∀ k : ℕ, k ≤ n → k ≠ lastLocker → ∃ (trip : ℕ), k % trip = 0 :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l2211_221141


namespace NUMINAMATH_CALUDE_circle_diameter_l2211_221176

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2211_221176


namespace NUMINAMATH_CALUDE_min_lines_correct_l2211_221117

/-- A line in a 2D plane represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The quadrants a line passes through -/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines required to guarantee at least two lines pass through the same quadrants -/
def min_lines : ℕ := 7

/-- Theorem stating that the minimum number of lines is correct -/
theorem min_lines_correct :
  ∀ (lines : Finset Line),
    lines.card ≥ min_lines →
    ∃ (l1 l2 : Line), l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ quadrants l1 = quadrants l2 :=
  sorry

end NUMINAMATH_CALUDE_min_lines_correct_l2211_221117


namespace NUMINAMATH_CALUDE_average_salary_is_8000_l2211_221174

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_people : ℕ := 5

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

theorem average_salary_is_8000 :
  total_salary / num_people = 8000 := by sorry

end NUMINAMATH_CALUDE_average_salary_is_8000_l2211_221174


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l2211_221197

theorem opposite_sign_sum (a b : ℝ) : 
  (|a + 1| + |b + 2| = 0) → (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l2211_221197
