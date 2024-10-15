import Mathlib

namespace NUMINAMATH_CALUDE_least_prime_factor_of_11_pow_5_minus_11_pow_2_l3502_350233

theorem least_prime_factor_of_11_pow_5_minus_11_pow_2 :
  Nat.minFac (11^5 - 11^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_11_pow_5_minus_11_pow_2_l3502_350233


namespace NUMINAMATH_CALUDE_log_square_equals_twenty_l3502_350250

theorem log_square_equals_twenty (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 8 / Real.log y)
  (h_product : x * y = 128) : 
  (Real.log (x / y) / Real.log 2)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_log_square_equals_twenty_l3502_350250


namespace NUMINAMATH_CALUDE_parabola_directrix_l3502_350246

/-- Given a parabola y² = 2px and a point M(1, m) on the parabola,
    with the distance from M to its focus being 5,
    prove that the equation of the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) : 
  m^2 = 2*p  -- M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 25  -- Distance from M to focus is 5
  → -p/2 = -4  -- Equation of directrix is x = -p/2
  := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3502_350246


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l3502_350247

theorem least_positive_angle_theorem : ∃ θ : ℝ,
  θ > 0 ∧
  θ ≤ 90 ∧
  Real.cos (10 * π / 180) = Real.sin (30 * π / 180) + Real.sin (θ * π / 180) ∧
  ∀ φ : ℝ, φ > 0 ∧ φ < θ →
    Real.cos (10 * π / 180) ≠ Real.sin (30 * π / 180) + Real.sin (φ * π / 180) ∧
  θ = 80 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l3502_350247


namespace NUMINAMATH_CALUDE_variance_transform_l3502_350259

/-- The variance of a dataset -/
def variance (data : List ℝ) : ℝ := sorry

/-- Transform a dataset by multiplying each element by a and adding b -/
def transform (data : List ℝ) (a b : ℝ) : List ℝ := sorry

theorem variance_transform (data : List ℝ) (a b : ℝ) :
  variance data = 3 →
  variance (transform data a b) = 12 →
  |a| = 2 := by sorry

end NUMINAMATH_CALUDE_variance_transform_l3502_350259


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3502_350206

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 20) → (∃ y : ℝ, y^2 = 7*y - 20 ∧ x + y = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3502_350206


namespace NUMINAMATH_CALUDE_storage_unit_blocks_l3502_350209

/-- Represents the dimensions of the storage unit --/
def storage_unit_side : ℕ := 8

/-- Represents the thickness of the walls, floor, and ceiling --/
def wall_thickness : ℕ := 1

/-- Calculates the number of blocks required for the storage unit construction --/
def blocks_required : ℕ :=
  storage_unit_side ^ 3 - (storage_unit_side - 2 * wall_thickness) ^ 3

/-- Theorem stating that 296 blocks are required for the storage unit construction --/
theorem storage_unit_blocks : blocks_required = 296 := by
  sorry

end NUMINAMATH_CALUDE_storage_unit_blocks_l3502_350209


namespace NUMINAMATH_CALUDE_trip_duration_l3502_350266

-- Define the type for time
structure Time where
  hours : ℕ
  minutes : ℕ

-- Define the function to calculate the angle between clock hands
def angleBetweenHands (t : Time) : ℝ := sorry

-- Define the function to find the time when hands are at a specific angle
def timeAtAngle (startHour startMinute : ℕ) (angle : ℝ) : Time := sorry

-- Define the function to calculate time difference
def timeDifference (t1 t2 : Time) : Time := sorry

-- The main theorem
theorem trip_duration : 
  let startTime := timeAtAngle 7 0 90
  let endTime := timeAtAngle 15 0 270
  let duration := timeDifference startTime endTime
  duration = Time.mk 8 29 := by sorry

end NUMINAMATH_CALUDE_trip_duration_l3502_350266


namespace NUMINAMATH_CALUDE_ball_count_after_500_steps_l3502_350223

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list -/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem ball_count_after_500_steps : sumDigits (toBase3 500) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_after_500_steps_l3502_350223


namespace NUMINAMATH_CALUDE_staircase_perimeter_l3502_350245

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- Number of congruent sides -/
  num_sides : ℕ
  /-- Length of each congruent side -/
  side_length : ℝ
  /-- Area of the region -/
  area : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.num_sides = 12)
  (h2 : s.side_length = 1)
  (h3 : s.area = 89) :
  perimeter s = 43 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l3502_350245


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3502_350203

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3502_350203


namespace NUMINAMATH_CALUDE_simplify_expression_1_l3502_350221

theorem simplify_expression_1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l3502_350221


namespace NUMINAMATH_CALUDE_otimes_equation_roots_l3502_350260

-- Define the new operation
def otimes (a b : ℝ) : ℝ := a * b^2 - b

-- Theorem statement
theorem otimes_equation_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ otimes 1 x = k ∧ otimes 1 y = k) ↔ k > -1/4 :=
sorry

end NUMINAMATH_CALUDE_otimes_equation_roots_l3502_350260


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3502_350239

theorem retailer_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : cost_price > 0) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3502_350239


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3502_350216

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = 2 or x = -1 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (x - 1, 1)
  parallel a b → x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3502_350216


namespace NUMINAMATH_CALUDE_shooting_competition_stability_l3502_350295

/-- Represents a participant in the shooting competition -/
structure Participant where
  name : String
  variance : ℝ

/-- Defines when a participant has more stable performance -/
def more_stable (p1 p2 : Participant) : Prop :=
  p1.variance < p2.variance

theorem shooting_competition_stability :
  let A : Participant := ⟨"A", 3⟩
  let B : Participant := ⟨"B", 1.2⟩
  more_stable B A := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_stability_l3502_350295


namespace NUMINAMATH_CALUDE_town_population_problem_l3502_350228

theorem town_population_problem (original_population : ℕ) : 
  (original_population + 1200 : ℕ) * 89 / 100 = original_population - 32 →
  original_population = 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3502_350228


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l3502_350212

theorem product_mod_seventeen : (2022 * 2023 * 2024 * 2025) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l3502_350212


namespace NUMINAMATH_CALUDE_liberty_middle_school_math_competition_l3502_350293

theorem liberty_middle_school_math_competition (sixth_graders seventh_graders : ℕ) : 
  (3 * sixth_graders = 7 * seventh_graders) →
  (sixth_graders + seventh_graders = 140) →
  sixth_graders = 61 := by
  sorry

end NUMINAMATH_CALUDE_liberty_middle_school_math_competition_l3502_350293


namespace NUMINAMATH_CALUDE_angle_supplement_complement_relation_l3502_350268

theorem angle_supplement_complement_relation (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 40) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_complement_relation_l3502_350268


namespace NUMINAMATH_CALUDE_jayden_coins_l3502_350243

theorem jayden_coins (jason_coins jayden_coins total_coins : ℕ) 
  (h1 : jason_coins = jayden_coins + 60)
  (h2 : jason_coins + jayden_coins = total_coins)
  (h3 : total_coins = 660) : 
  jayden_coins = 300 := by
sorry

end NUMINAMATH_CALUDE_jayden_coins_l3502_350243


namespace NUMINAMATH_CALUDE_jackson_decorations_to_friend_l3502_350215

/-- Represents the number of Christmas decorations Mrs. Jackson gives to her friend. -/
def decorations_to_friend (total_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ) (given_to_neighbor : ℕ) : ℕ :=
  total_boxes * decorations_per_box - used_decorations - given_to_neighbor

/-- Proves that Mrs. Jackson gives 17 decorations to her friend under the given conditions. -/
theorem jackson_decorations_to_friend :
  decorations_to_friend 6 25 58 75 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_to_friend_l3502_350215


namespace NUMINAMATH_CALUDE_power_comparison_l3502_350224

theorem power_comparison : 2^444 = 4^222 ∧ 2^444 < 3^333 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l3502_350224


namespace NUMINAMATH_CALUDE_champagne_glasses_per_guest_l3502_350208

/-- Calculates the number of champagne glasses per guest at Ashley's wedding. -/
theorem champagne_glasses_per_guest :
  let num_guests : ℕ := 120
  let servings_per_bottle : ℕ := 6
  let num_bottles : ℕ := 40
  let total_servings : ℕ := num_bottles * servings_per_bottle
  let glasses_per_guest : ℕ := total_servings / num_guests
  glasses_per_guest = 2 := by
  sorry

end NUMINAMATH_CALUDE_champagne_glasses_per_guest_l3502_350208


namespace NUMINAMATH_CALUDE_xiaohua_school_time_l3502_350277

-- Define a custom type for time
structure SchoolTime where
  hours : ℕ
  minutes : ℕ
  is_pm : Bool

-- Define a function to calculate the time difference in minutes
def time_diff (t1 t2 : SchoolTime) : ℕ :=
  let total_minutes1 := t1.hours * 60 + t1.minutes + (if t1.is_pm then 12 * 60 else 0)
  let total_minutes2 := t2.hours * 60 + t2.minutes + (if t2.is_pm then 12 * 60 else 0)
  total_minutes2 - total_minutes1

-- Define Xiaohua's schedule
def morning_arrival : SchoolTime := ⟨7, 20, false⟩
def morning_departure : SchoolTime := ⟨11, 45, false⟩
def afternoon_arrival : SchoolTime := ⟨1, 50, true⟩
def afternoon_departure : SchoolTime := ⟨5, 15, true⟩

-- Theorem statement
theorem xiaohua_school_time :
  time_diff morning_arrival morning_departure +
  time_diff afternoon_arrival afternoon_departure = 7 * 60 + 50 := by
  sorry

end NUMINAMATH_CALUDE_xiaohua_school_time_l3502_350277


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l3502_350210

theorem ratio_a_to_b (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l3502_350210


namespace NUMINAMATH_CALUDE_triangle_side_length_l3502_350256

/-- In a triangle DEF, given angle E, side DE, and side DF, prove the length of EF --/
theorem triangle_side_length (E D F : ℝ) (hE : E = 45 * π / 180) 
  (hDE : D = 100) (hDF : F = 100 * Real.sqrt 2) : 
  ∃ (EF : ℝ), abs (EF - Real.sqrt (10000 + 5176.4)) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3502_350256


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l3502_350285

/-- Calculates the average speed of a car trip with multiple segments and delays -/
theorem car_trip_average_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (gravel_distance : ℝ) (gravel_speed : ℝ)
  (highway_distance : ℝ) (highway_speed : ℝ)
  (traffic_delay : ℝ) (obstruction_delay : ℝ)
  (h_local : local_distance = 60 ∧ local_speed = 30)
  (h_gravel : gravel_distance = 10 ∧ gravel_speed = 20)
  (h_highway : highway_distance = 105 ∧ highway_speed = 60)
  (h_traffic : traffic_delay = 0.25)
  (h_obstruction : obstruction_delay = 0.1667)
  : ∃ (average_speed : ℝ), 
    abs (average_speed - 37.5) < 0.1 ∧
    average_speed = (local_distance + gravel_distance + highway_distance) / 
      (local_distance / local_speed + gravel_distance / gravel_speed + 
       highway_distance / highway_speed + traffic_delay + obstruction_delay) :=
by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l3502_350285


namespace NUMINAMATH_CALUDE_min_value_expression_l3502_350222

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3502_350222


namespace NUMINAMATH_CALUDE_marly_bills_denomination_l3502_350290

theorem marly_bills_denomination (x : ℕ) : 
  (10 * 20 + 8 * x + 4 * 5 = 3 * 100) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_marly_bills_denomination_l3502_350290


namespace NUMINAMATH_CALUDE_power_set_of_A_l3502_350226

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set (Set ℕ) := {x | x ⊆ A}

-- Theorem statement
theorem power_set_of_A : B = {∅, {1}, {2}, {1, 2}} := by
  sorry

end NUMINAMATH_CALUDE_power_set_of_A_l3502_350226


namespace NUMINAMATH_CALUDE_tan_value_for_given_point_l3502_350241

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) :
  (∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) →
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_point_l3502_350241


namespace NUMINAMATH_CALUDE_david_total_cost_l3502_350244

/-- Calculates the total cost of a cell phone plan given usage and plan details -/
def calculateTotalCost (baseCost monthlyTexts monthlyHours monthlyData : ℕ)
                       (extraTextCost extraMinuteCost extraGBCost : ℚ)
                       (usedTexts usedHours usedData : ℕ) : ℚ :=
  let extraTexts := max (usedTexts - monthlyTexts) 0
  let extraMinutes := max (usedHours * 60 - monthlyHours * 60) 0
  let extraData := max (usedData - monthlyData) 0
  baseCost + extraTextCost * extraTexts + extraMinuteCost * extraMinutes + extraGBCost * extraData

/-- Theorem stating that David's total cost is $54.50 -/
theorem david_total_cost :
  calculateTotalCost 25 200 40 3 (3/100) (15/100) 10 250 42 4 = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_david_total_cost_l3502_350244


namespace NUMINAMATH_CALUDE_principal_amount_l3502_350227

/-- Proves that given the specified conditions, the principal amount is 2600 --/
theorem principal_amount (rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  rate = 4/100 → 
  time = 5 → 
  interest_difference = 2080 → 
  (∃ (principal : ℚ), 
    principal * rate * time = principal - interest_difference ∧ 
    principal = 2600) := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l3502_350227


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3502_350231

/-- The ratio of the side length of a regular pentagon to the width of a rectangle is 6/5, 
    given that both shapes have a perimeter of 60 inches and the rectangle's length is twice its width. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width rectangle_length : ℝ),
  pentagon_side * 5 = 60 →
  rectangle_width * 2 + rectangle_length * 2 = 60 →
  rectangle_length = 2 * rectangle_width →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3502_350231


namespace NUMINAMATH_CALUDE_sugar_added_indeterminate_l3502_350291

-- Define the recipe requirements
def total_flour : ℕ := 9
def total_sugar : ℕ := 5

-- Define Mary's current actions
def flour_added : ℕ := 3
def flour_to_add : ℕ := 6

-- Define a variable for the unknown amount of sugar added
variable (sugar_added : ℕ)

-- Theorem stating that sugar_added cannot be uniquely determined
theorem sugar_added_indeterminate : 
  ∀ (x y : ℕ), x ≠ y → 
  (x ≤ total_sugar ∧ y ≤ total_sugar) → 
  (∃ (state₁ state₂ : ℕ × ℕ), 
    state₁.1 = flour_added ∧ 
    state₁.2 = x ∧ 
    state₂.1 = flour_added ∧ 
    state₂.2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sugar_added_indeterminate_l3502_350291


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3502_350218

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 40 →           -- One angle is 40°
  c = 3 * b →        -- The other two angles are in the ratio 1:3
  min a (min b c) = 35 :=  -- The smallest angle is 35°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3502_350218


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_37_l3502_350202

theorem inverse_of_3_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (3 * x) % 37 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_37_l3502_350202


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l3502_350219

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin x - Real.cos x) 
  (h2 : f α = 1) : 
  Real.sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l3502_350219


namespace NUMINAMATH_CALUDE_problem_statement_l3502_350263

theorem problem_statement (m : ℤ) : 
  2^2000 - 3 * 2^1998 + 5 * 2^1996 - 2^1995 = m * 2^1995 → m = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3502_350263


namespace NUMINAMATH_CALUDE_polar_to_cartesian_and_intersection_l3502_350232

-- Define the polar equations
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the standard equations
def standard_line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def standard_circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the theorem
theorem polar_to_cartesian_and_intersection :
  (∀ ρ θ : ℝ, line_l ρ θ ↔ ∃ x y : ℝ, standard_line_l x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ ρ θ : ℝ, circle_C ρ θ ↔ ∃ x y : ℝ, standard_circle_C x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ A B : ℝ × ℝ, 
    standard_line_l A.1 A.2 ∧ standard_circle_C A.1 A.2 ∧
    standard_line_l B.1 B.2 ∧ standard_circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19) :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_and_intersection_l3502_350232


namespace NUMINAMATH_CALUDE_sequence_inequality_l3502_350297

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) :
  n ≥ 2 →
  (∀ k, a k > 0) →
  (∀ k ∈ Finset.range (n - 1), (a (k - 1) + a k) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) →
  a n < 1 / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3502_350297


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3502_350251

theorem fraction_to_decimal : (125 : ℚ) / 144 = 0.78125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3502_350251


namespace NUMINAMATH_CALUDE_sandys_pumpkins_l3502_350234

/-- Sandy and Mike grew pumpkins. This theorem proves how many pumpkins Sandy grew. -/
theorem sandys_pumpkins (mike_pumpkins total_pumpkins : ℕ) 
  (h1 : mike_pumpkins = 23)
  (h2 : mike_pumpkins + sandy_pumpkins = total_pumpkins)
  (h3 : total_pumpkins = 74) :
  sandy_pumpkins = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_sandys_pumpkins_l3502_350234


namespace NUMINAMATH_CALUDE_special_power_function_unique_m_l3502_350237

/-- A power function with exponent (m^2 - 2m - 3) that has no intersection with axes and is symmetric about the origin -/
def special_power_function (m : ℕ+) : ℝ → ℝ := fun x ↦ x ^ (m.val ^ 2 - 2 * m.val - 3)

/-- The function has no intersection with x-axis and y-axis -/
def no_axis_intersection (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≠ 0) ∧ (f 0 ≠ 0)

/-- The function is symmetric about the origin -/
def origin_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Main theorem: If the special power function satisfies the conditions, then m = 2 -/
theorem special_power_function_unique_m (m : ℕ+) :
  no_axis_intersection (special_power_function m) ∧
  origin_symmetry (special_power_function m) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_power_function_unique_m_l3502_350237


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3502_350265

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (sum_eq : a + b = 5) 
  (cube_sum_eq : a^3 + b^3 = 125) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3502_350265


namespace NUMINAMATH_CALUDE_second_rectangle_weight_l3502_350271

-- Define the properties of the rectangles
def length1 : ℝ := 4
def width1 : ℝ := 3
def weight1 : ℝ := 18
def length2 : ℝ := 6
def width2 : ℝ := 4

-- Theorem to prove
theorem second_rectangle_weight :
  ∀ (density : ℝ),
  density > 0 →
  let area1 := length1 * width1
  let area2 := length2 * width2
  let weight2 := (area2 / area1) * weight1
  weight2 = 36 := by
sorry

end NUMINAMATH_CALUDE_second_rectangle_weight_l3502_350271


namespace NUMINAMATH_CALUDE_certain_multiple_remainder_l3502_350207

theorem certain_multiple_remainder (m : ℤ) (h : m % 5 = 2) :
  (∃ k : ℕ+, k * m % 5 = 1) ∧ (∀ k : ℕ+, k * m % 5 = 1 → k ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_certain_multiple_remainder_l3502_350207


namespace NUMINAMATH_CALUDE_star_sum_24_five_pointed_star_24_l3502_350258

/-- Represents the vertices of a five-pointed star -/
inductive StarVertex
| A | B | C | D | E | F | G | H | J | K

/-- Assignment of numbers to the vertices of the star -/
def star_assignment : StarVertex → ℤ
| StarVertex.A => 1
| StarVertex.B => 2
| StarVertex.C => 3
| StarVertex.D => 4
| StarVertex.E => 5
| StarVertex.F => 10
| StarVertex.G => 12
| StarVertex.H => 9
| StarVertex.J => 6
| StarVertex.K => 8

/-- The set of all straight lines in the star -/
def star_lines : List (List StarVertex) := [
  [StarVertex.E, StarVertex.F, StarVertex.H, StarVertex.J],
  [StarVertex.F, StarVertex.G, StarVertex.K, StarVertex.J],
  [StarVertex.H, StarVertex.J, StarVertex.K, StarVertex.B],
  [StarVertex.J, StarVertex.E, StarVertex.K, StarVertex.C],
  [StarVertex.A, StarVertex.J, StarVertex.G, StarVertex.B]
]

/-- Theorem stating that the sum of numbers on each straight line equals 24 -/
theorem star_sum_24 : ∀ line ∈ star_lines, 
  (line.map star_assignment).sum = 24 := by sorry

/-- Main theorem proving the existence of a valid assignment -/
theorem five_pointed_star_24 : 
  ∃ (f : StarVertex → ℤ), ∀ line ∈ star_lines, (line.map f).sum = 24 := by
  use star_assignment
  exact star_sum_24

end NUMINAMATH_CALUDE_star_sum_24_five_pointed_star_24_l3502_350258


namespace NUMINAMATH_CALUDE_integer_roots_count_l3502_350249

/-- Represents a fourth-degree polynomial with integer coefficients -/
structure IntPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of an IntPolynomial, counting multiplicity -/
def num_integer_roots (p : IntPolynomial) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem integer_roots_count (p : IntPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_count_l3502_350249


namespace NUMINAMATH_CALUDE_pacos_countertop_marble_weight_l3502_350276

theorem pacos_countertop_marble_weight : 
  let weights : List ℝ := [0.33, 0.33, 0.08, 0.25, 0.02, 0.12, 0.15]
  weights.sum = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_pacos_countertop_marble_weight_l3502_350276


namespace NUMINAMATH_CALUDE_pyramid_section_ratio_l3502_350273

/-- Represents a pyramid with a side edge and two points on it -/
structure Pyramid where
  -- Side edge length
  ab : ℝ
  -- Position of point K from A
  ak : ℝ
  -- Position of point M from A
  am : ℝ
  -- Conditions
  ab_pos : 0 < ab
  k_on_ab : 0 ≤ ak ∧ ak ≤ ab
  m_on_ab : 0 ≤ am ∧ am ≤ ab
  ak_eq_bm : ak = ab - am
  sections_area : (ak / ab)^2 + (am / ab)^2 = 2/3

/-- The main theorem -/
theorem pyramid_section_ratio (p : Pyramid) : (p.am - p.ak) / p.ab = 1 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_section_ratio_l3502_350273


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3502_350242

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x < 1 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3502_350242


namespace NUMINAMATH_CALUDE_graph_properties_of_y_squared_equals_sin_x_squared_l3502_350252

theorem graph_properties_of_y_squared_equals_sin_x_squared :
  ∃ f : ℝ → Set ℝ, 
    (∀ x y, y ∈ f x ↔ y^2 = Real.sin (x^2)) ∧ 
    (0 ∈ f 0) ∧ 
    (∀ x y, y ∈ f x → -y ∈ f x) ∧
    (∀ x, (∃ y, y ∈ f x) → Real.sin (x^2) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_graph_properties_of_y_squared_equals_sin_x_squared_l3502_350252


namespace NUMINAMATH_CALUDE_existsNonIsoscelesWithFourEqualAreas_l3502_350284

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Function to create smaller triangles by connecting P to vertices and drawing perpendiculars
def createSmallerTriangles (P : Point) (t : Triangle) : List Triangle := sorry

-- Function to check if 4 out of 6 triangles have equal areas
def fourEqualAreas (triangles : List Triangle) : Prop := sorry

-- The main theorem
theorem existsNonIsoscelesWithFourEqualAreas : 
  ∃ (t : Triangle) (P : Point), 
    isInside P t ∧ 
    ¬isIsosceles t ∧ 
    fourEqualAreas (createSmallerTriangles P t) := sorry

end NUMINAMATH_CALUDE_existsNonIsoscelesWithFourEqualAreas_l3502_350284


namespace NUMINAMATH_CALUDE_einstein_soda_sales_l3502_350296

def goal : ℝ := 500
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def remaining : ℝ := 258

theorem einstein_soda_sales :
  ∃ (soda_sold : ℕ),
    goal = pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold + remaining ∧
    soda_sold = 25 := by
  sorry

end NUMINAMATH_CALUDE_einstein_soda_sales_l3502_350296


namespace NUMINAMATH_CALUDE_system_solution_unique_l3502_350282

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + 2 * y = 5 ∧ x - 2 * y = 11 ∧ x = 4 ∧ y = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3502_350282


namespace NUMINAMATH_CALUDE_doughnut_cost_theorem_l3502_350286

/-- Calculate the total cost of doughnuts for a class --/
theorem doughnut_cost_theorem (total_students : ℕ) 
  (chocolate_students : ℕ) (glazed_students : ℕ) (maple_students : ℕ) (strawberry_students : ℕ)
  (chocolate_cost : ℚ) (glazed_cost : ℚ) (maple_cost : ℚ) (strawberry_cost : ℚ) :
  total_students = 25 →
  chocolate_students = 10 →
  glazed_students = 8 →
  maple_students = 5 →
  strawberry_students = 2 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  maple_cost = (3/2) →
  strawberry_cost = (5/2) →
  (chocolate_students : ℚ) * chocolate_cost + 
  (glazed_students : ℚ) * glazed_cost + 
  (maple_students : ℚ) * maple_cost + 
  (strawberry_students : ℚ) * strawberry_cost = (81/2) := by
  sorry

#eval (81/2 : ℚ)

end NUMINAMATH_CALUDE_doughnut_cost_theorem_l3502_350286


namespace NUMINAMATH_CALUDE_cylinder_properties_l3502_350229

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  ∀ (h r : ℝ),
  h = 15 →
  r = 5 →
  (2 * π * r^2 + 2 * π * r * h = 200 * π) ∧
  (π * r^2 * h = 375 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_properties_l3502_350229


namespace NUMINAMATH_CALUDE_expression_evaluation_l3502_350201

theorem expression_evaluation : 1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 = 190 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3502_350201


namespace NUMINAMATH_CALUDE_stratified_sampling_third_group_size_l3502_350274

/-- Proves that in a stratified sampling scenario, given specific conditions, 
    the size of the third group is 1040. -/
theorem stratified_sampling_third_group_size 
  (total_sample : ℕ) 
  (grade11_sample : ℕ) 
  (grade10_pop : ℕ) 
  (grade11_pop : ℕ) 
  (h1 : total_sample = 81)
  (h2 : grade11_sample = 30)
  (h3 : grade10_pop = 1000)
  (h4 : grade11_pop = 1200) :
  ∃ n : ℕ, 
    (grade11_sample : ℚ) / total_sample = 
    grade11_pop / (grade10_pop + grade11_pop + n) ∧ 
    n = 1040 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_group_size_l3502_350274


namespace NUMINAMATH_CALUDE_angle_AMB_largest_l3502_350200

/-- Given a right angle XOY with OA = a and OB = b (a < b) on side OY, 
    and a point M on OX such that OM = x, 
    prove that the angle AMB is largest when x = √(ab) -/
theorem angle_AMB_largest (a b x : ℝ) (h_ab : 0 < a ∧ a < b) :
  let φ := Real.arctan ((b - a) * x / (x^2 + a * b))
  ∀ y : ℝ, y > 0 → φ ≤ Real.arctan ((b - a) * y / (y^2 + a * b)) →
  x = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_angle_AMB_largest_l3502_350200


namespace NUMINAMATH_CALUDE_tim_kittens_l3502_350279

theorem tim_kittens (initial : ℕ) (given_away : ℕ) (received : ℕ) :
  initial = 6 →
  given_away = 3 →
  received = 9 →
  initial - given_away + received = 12 := by
sorry

end NUMINAMATH_CALUDE_tim_kittens_l3502_350279


namespace NUMINAMATH_CALUDE_max_profit_at_three_l3502_350248

/-- Represents the annual operating cost for a given year -/
def annual_cost (n : ℕ) : ℚ := 2 * n

/-- Represents the total operating cost for n years -/
def total_cost (n : ℕ) : ℚ := n^2 + n

/-- Represents the annual operating income -/
def annual_income : ℚ := 11

/-- Represents the initial cost of the car -/
def initial_cost : ℚ := 9

/-- Represents the annual average profit for n years -/
def annual_average_profit (n : ℕ+) : ℚ := 
  annual_income - (total_cost n + initial_cost) / n

/-- Theorem stating that the annual average profit is maximized when n = 3 -/
theorem max_profit_at_three : 
  ∀ (m : ℕ+), annual_average_profit 3 ≥ annual_average_profit m :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_three_l3502_350248


namespace NUMINAMATH_CALUDE_max_cubes_fit_l3502_350238

theorem max_cubes_fit (large_side : ℕ) (small_edge : ℕ) : large_side = 10 ∧ small_edge = 2 →
  (large_side ^ 3) / (small_edge ^ 3) = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_fit_l3502_350238


namespace NUMINAMATH_CALUDE_zero_unique_additive_multiplicative_property_l3502_350255

theorem zero_unique_additive_multiplicative_property :
  ∀ x : ℤ, (∀ z : ℤ, z + x = z) ∧ (∀ z : ℤ, z * x = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_unique_additive_multiplicative_property_l3502_350255


namespace NUMINAMATH_CALUDE_max_triangle_perimeter_l3502_350280

theorem max_triangle_perimeter (x : ℕ) : 
  x > 0 ∧ x < 17 ∧ 8 + x > 9 ∧ 9 + x > 8 → 
  ∀ y : ℕ, y > 0 ∧ y < 17 ∧ 8 + y > 9 ∧ 9 + y > 8 → 
  8 + 9 + x ≥ 8 + 9 + y ∧ 
  8 + 9 + x ≤ 33 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_perimeter_l3502_350280


namespace NUMINAMATH_CALUDE_exists_triangle_no_isosceles_triangle_l3502_350254

/-- The set of stick lengths -/
def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

/-- Function to check if three lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three lengths can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem stating that a triangle can be formed from the given stick lengths -/
theorem exists_triangle : ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c :=
sorry

/-- Theorem stating that an isosceles triangle cannot be formed from the given stick lengths -/
theorem no_isosceles_triangle : ¬∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_no_isosceles_triangle_l3502_350254


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3502_350269

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The total volume of a mixture -/
def Mixture.volume (m : Mixture) : ℚ := m.milk + m.water

/-- The ratio of milk to water in a mixture -/
def Mixture.ratio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_problem (m1 m2 : Mixture) :
  m1.volume = m2.volume →
  m1.ratio = 7/2 →
  (Mixture.mk (m1.milk + m2.milk) (m1.water + m2.water)).ratio = 5 →
  m2.ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3502_350269


namespace NUMINAMATH_CALUDE_gummy_bear_production_l3502_350220

/-- The number of gummy bears in each packet -/
def bears_per_packet : ℕ := 50

/-- The number of packets filled in 40 minutes -/
def packets_filled : ℕ := 240

/-- The time taken to fill the packets (in minutes) -/
def time_taken : ℕ := 40

/-- The number of gummy bears manufactured per minute -/
def bears_per_minute : ℕ := packets_filled * bears_per_packet / time_taken

theorem gummy_bear_production :
  bears_per_minute = 300 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_production_l3502_350220


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3502_350230

/-- The probability of getting exactly k positive answers out of n questions,
    where each question has a p probability of a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 7 questions
    from a Magic 8 Ball, where each question has a 3/7 chance of a positive answer. -/
theorem magic_8_ball_probability :
  binomial_probability 7 3 (3/7) = 242520/823543 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3502_350230


namespace NUMINAMATH_CALUDE_bus_cost_relationship_l3502_350214

/-- The functional relationship between the number of large buses purchased and the total cost -/
theorem bus_cost_relationship (x : ℝ) (y : ℝ) : y = 22 * x + 800 ↔ 
  y = 62 * x + 40 * (20 - x) := by sorry

end NUMINAMATH_CALUDE_bus_cost_relationship_l3502_350214


namespace NUMINAMATH_CALUDE_triangle_problem_l3502_350292

theorem triangle_problem (A B C : Real) (a b c : Real) :
  C = π / 3 →
  b = 8 →
  (1 / 2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  c = 7 ∧ Real.cos (B - C) = 13 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3502_350292


namespace NUMINAMATH_CALUDE_min_nSn_l3502_350288

/-- An arithmetic sequence with sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_10 : S 10 = 0
  sum_15 : S 15 = 25

/-- The product of n and S_n for an arithmetic sequence -/
def nSn (seq : ArithmeticSequence) (n : ℕ) : ℝ := n * seq.S n

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (min : ℝ), min = -49 ∧ ∀ (n : ℕ), n ≠ 0 → min ≤ nSn seq n :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l3502_350288


namespace NUMINAMATH_CALUDE_special_integer_count_l3502_350278

/-- Count of positive integers less than 100,000 with at most two different digits -/
def count_special_integers : ℕ :=
  let single_digit_count := 45
  let two_digit_count_no_zero := 1872
  let two_digit_count_with_zero := 234
  single_digit_count + two_digit_count_no_zero + two_digit_count_with_zero

/-- The count of positive integers less than 100,000 with at most two different digits is 2151 -/
theorem special_integer_count : count_special_integers = 2151 := by
  sorry

end NUMINAMATH_CALUDE_special_integer_count_l3502_350278


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3502_350272

theorem no_positive_integer_solutions :
  ¬ ∃ (x : ℕ), 15 < 3 - 2 * (x : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3502_350272


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l3502_350275

/-- Function f(x) = |x - 1| + |x + 2| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

/-- Function g(x) = |x + 1| - |x - a| + a -/
def g (a x : ℝ) : ℝ := |x + 1| - |x - a| + a

/-- The solution set of f(x) + g(x) < 6 when a = 1 is (-4, 1) -/
theorem solution_set_theorem :
  {x : ℝ | f x + g 1 x < 6} = Set.Ioo (-4) 1 := by sorry

/-- For any real numbers x₁ and x₂, f(x₁) ≥ g(x₂) if and only if a ∈ (-∞, 1] -/
theorem range_of_a_theorem :
  ∀ (a : ℝ), (∀ (x₁ x₂ : ℝ), f x₁ ≥ g a x₂) ↔ a ∈ Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l3502_350275


namespace NUMINAMATH_CALUDE_probability_two_same_pair_l3502_350211

/-- The number of students participating in the events -/
def num_students : ℕ := 3

/-- The number of events available -/
def num_events : ℕ := 3

/-- The number of events each student chooses -/
def events_per_student : ℕ := 2

/-- The total number of possible combinations for all students' choices -/
def total_combinations : ℕ := num_students ^ num_events

/-- The number of ways to choose 2 students out of 3 -/
def ways_to_choose_2_students : ℕ := 3

/-- The number of ways to choose 1 pair of events out of 3 possible pairs -/
def ways_to_choose_event_pair : ℕ := 3

/-- The number of choices for the remaining student -/
def choices_for_remaining_student : ℕ := 2

/-- The number of favorable outcomes (where exactly two students choose the same pair) -/
def favorable_outcomes : ℕ := ways_to_choose_2_students * ways_to_choose_event_pair * choices_for_remaining_student

/-- The probability of exactly two students choosing the same pair of events -/
theorem probability_two_same_pair : 
  (favorable_outcomes : ℚ) / total_combinations = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_two_same_pair_l3502_350211


namespace NUMINAMATH_CALUDE_equal_intercepts_condition_l3502_350298

/-- The line equation ax + y - 2 - a = 0 has equal intercepts on x and y axes iff a = -2 or a = 1 -/
theorem equal_intercepts_condition (a : ℝ) : 
  (∃ (x y : ℝ), (a * x + y - 2 - a = 0 ∧ 
                ((x = 0 ∨ y = 0) ∧ 
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ x' = 0 → y' = y) ∧
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ y' = 0 → x' = x))))
  ↔ (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_condition_l3502_350298


namespace NUMINAMATH_CALUDE_cake_radius_increase_l3502_350299

theorem cake_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 30) (h₂ : c₂ = 37.5) :
  (c₂ / (2 * Real.pi)) - (c₁ / (2 * Real.pi)) = 7.5 / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cake_radius_increase_l3502_350299


namespace NUMINAMATH_CALUDE_percent_relation_l3502_350267

theorem percent_relation (x y z : ℝ) 
  (hxy : x = 1.20 * y) 
  (hyz : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3502_350267


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3502_350240

theorem quadratic_inequality_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l3502_350240


namespace NUMINAMATH_CALUDE_cylinder_radius_calculation_l3502_350257

theorem cylinder_radius_calculation (shadow_length : ℝ) (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (h1 : shadow_length = 12)
  (h2 : flagpole_height = 1.5)
  (h3 : flagpole_shadow = 3)
  (h4 : flagpole_shadow > 0) -- To avoid division by zero
  : ∃ (radius : ℝ), radius = shadow_length * (flagpole_height / flagpole_shadow) ∧ radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_calculation_l3502_350257


namespace NUMINAMATH_CALUDE_project_hours_ratio_l3502_350281

/-- Proves that given the conditions of the project hours, the ratio of Pat's time to Kate's time is 4:3 -/
theorem project_hours_ratio :
  ∀ (pat kate mark : ℕ),
  pat + kate + mark = 189 →
  ∃ (r : ℚ), pat = r * kate →
  pat = (1 : ℚ) / 3 * mark →
  mark = kate + 105 →
  r = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_project_hours_ratio_l3502_350281


namespace NUMINAMATH_CALUDE_arrangement_count_l3502_350217

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose num_teachers teachers_per_group * choose num_students students_per_group

theorem arrangement_count :
  total_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3502_350217


namespace NUMINAMATH_CALUDE_problem_solution_l3502_350204

theorem problem_solution : (12346 * 24689 * 37033 + 12347 * 37034) / 12345^2 = 74072 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3502_350204


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3502_350283

theorem no_perfect_squares (a b : ℕ) : ¬(∃k m : ℕ, (a^2 + 2*b^2 = k^2) ∧ (b^2 + 2*a = m^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3502_350283


namespace NUMINAMATH_CALUDE_ladder_problem_l3502_350205

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length ^ 2 = height ^ 2 + base ^ 2) : 
  base = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l3502_350205


namespace NUMINAMATH_CALUDE_f_is_algebraic_fraction_l3502_350235

/-- An algebraic fraction is a ratio of algebraic expressions. -/
def is_algebraic_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, d x ≠ 0 → f x = (n x) / (d x)

/-- The function f(x) = 2/(x+3) for x ≠ -3 -/
def f (x : ℚ) : ℚ := 2 / (x + 3)

/-- Theorem: f(x) = 2/(x+3) is an algebraic fraction -/
theorem f_is_algebraic_fraction : is_algebraic_fraction f :=
sorry

end NUMINAMATH_CALUDE_f_is_algebraic_fraction_l3502_350235


namespace NUMINAMATH_CALUDE_bill_difference_l3502_350289

theorem bill_difference : 
  ∀ (alice_bill bob_bill : ℝ),
  alice_bill * 0.25 = 5 →
  bob_bill * 0.10 = 4 →
  bob_bill - alice_bill = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l3502_350289


namespace NUMINAMATH_CALUDE_intersection_subset_iff_m_eq_two_l3502_350270

/-- Sets A, B, and C as defined in the problem -/
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x > 1 ∨ x < -5}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

/-- Theorem stating that A ∩ B ⊆ C(m) if and only if m = 2 -/
theorem intersection_subset_iff_m_eq_two :
  ∀ m : ℝ, (A ∩ B) ⊆ C m ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_subset_iff_m_eq_two_l3502_350270


namespace NUMINAMATH_CALUDE_two_piggy_banks_value_l3502_350294

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "dime" => 10
  | _ => 0

/-- Represents the number of coins in a piggy bank -/
def coins_in_bank (coin : String) : ℕ :=
  match coin with
  | "penny" => 100
  | "dime" => 50
  | _ => 0

/-- Calculates the total value in cents for a single piggy bank -/
def piggy_bank_value : ℕ :=
  coin_value "penny" * coins_in_bank "penny" +
  coin_value "dime" * coins_in_bank "dime"

/-- Calculates the total value in dollars for two piggy banks -/
def total_value : ℚ :=
  (2 * piggy_bank_value : ℚ) / 100

theorem two_piggy_banks_value : total_value = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_piggy_banks_value_l3502_350294


namespace NUMINAMATH_CALUDE_cube_sum_square_not_prime_product_l3502_350287

theorem cube_sum_square_not_prime_product (a b : ℕ+) (h : ∃ (u : ℕ), (a.val ^ 3 + b.val ^ 3 : ℕ) = u ^ 2) :
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a.val + b.val = p * q :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_square_not_prime_product_l3502_350287


namespace NUMINAMATH_CALUDE_infinite_divisibility_sequence_l3502_350213

theorem infinite_divisibility_sequence : 
  ∃ (a : ℕ → ℕ), ∀ n, (a n)^2 ∣ (2^(a n) + 3^(a n)) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisibility_sequence_l3502_350213


namespace NUMINAMATH_CALUDE_t_shaped_area_l3502_350253

/-- The area of a T-shaped region formed by subtracting two squares and a rectangle from a larger square --/
theorem t_shaped_area (side_large : ℝ) (side_small : ℝ) (rect_length rect_width : ℝ) : 
  side_large = side_small + rect_length →
  side_large = 6 →
  side_small = 2 →
  rect_length = 4 →
  rect_width = 2 →
  side_large^2 - (2 * side_small^2 + rect_length * rect_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_l3502_350253


namespace NUMINAMATH_CALUDE_c_nonzero_l3502_350225

def Q (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem c_nonzero (a b c d e : ℝ) :
  (∀ x : ℝ, x = 0 ∨ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2 → Q a b c d e x = 0) →
  c ≠ 0 := by sorry

end NUMINAMATH_CALUDE_c_nonzero_l3502_350225


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3502_350264

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3502_350264


namespace NUMINAMATH_CALUDE_min_disks_required_l3502_350236

def total_files : ℕ := 40
def disk_capacity : ℚ := 2
def files_1mb : ℕ := 4
def files_0_9mb : ℕ := 16
def file_size_1mb : ℚ := 1
def file_size_0_9mb : ℚ := 9/10
def file_size_0_5mb : ℚ := 1/2

theorem min_disks_required :
  let remaining_files := total_files - files_1mb - files_0_9mb
  let total_size := files_1mb * file_size_1mb + 
                    files_0_9mb * file_size_0_9mb + 
                    remaining_files * file_size_0_5mb
  let min_disks := Int.ceil (total_size / disk_capacity)
  min_disks = 16 := by sorry

end NUMINAMATH_CALUDE_min_disks_required_l3502_350236


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3502_350261

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3502_350261


namespace NUMINAMATH_CALUDE_solution_comparison_l3502_350262

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q / p > -q' / p') ↔ (q / p < q' / p') :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l3502_350262
