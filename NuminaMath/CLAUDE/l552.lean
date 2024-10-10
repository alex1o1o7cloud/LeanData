import Mathlib

namespace simplified_ratio_of_boys_to_girls_l552_55201

def number_of_boys : ℕ := 12
def number_of_girls : ℕ := 18

theorem simplified_ratio_of_boys_to_girls :
  (number_of_boys : ℚ) / (number_of_girls : ℚ) = 2 / 3 := by
  sorry

end simplified_ratio_of_boys_to_girls_l552_55201


namespace integral_of_constant_one_equals_one_l552_55259

-- Define the constant function f(x) = 1
def f : ℝ → ℝ := λ x => 1

-- State the theorem
theorem integral_of_constant_one_equals_one :
  ∫ x in (0:ℝ)..1, f x = 1 := by sorry

end integral_of_constant_one_equals_one_l552_55259


namespace bill_with_tip_divisibility_l552_55273

theorem bill_with_tip_divisibility (x : ℕ) : ∃ k : ℕ, (11 * x) = (10 * k) := by
  sorry

end bill_with_tip_divisibility_l552_55273


namespace parabola_focus_l552_55247

/-- The focus of the parabola y^2 = 8x is at the point (2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (∀ x y, y^2 = 8*x ↔ (x - 2)^2 + y^2 = 4) := by
  sorry

end parabola_focus_l552_55247


namespace largest_integer_in_interval_l552_55284

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9) → y ≤ x :=
by sorry

end largest_integer_in_interval_l552_55284


namespace bridgette_dog_baths_l552_55224

/-- The number of times Bridgette bathes her dogs each month -/
def dog_baths_per_month : ℕ := sorry

/-- The number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- The number of cats Bridgette has -/
def num_cats : ℕ := 3

/-- The number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- The number of times Bridgette bathes her cats each month -/
def cat_baths_per_month : ℕ := 1

/-- The number of times Bridgette bathes her birds each month -/
def bird_baths_per_month : ℚ := 1/4

/-- The total number of baths Bridgette gives in a year -/
def total_baths_per_year : ℕ := 96

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem bridgette_dog_baths : 
  dog_baths_per_month = 2 :=
by sorry

end bridgette_dog_baths_l552_55224


namespace right_triangle_hypotenuse_l552_55205

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  c = 10 * Real.sqrt 10 :=
by sorry

end right_triangle_hypotenuse_l552_55205


namespace wrong_observation_value_l552_55240

theorem wrong_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : correct_value = 45)
  (h4 : new_mean = 40.66)
  : ∃ (wrong_value : ℝ),
    n * new_mean - n * initial_mean = correct_value - wrong_value ∧
    wrong_value = 12 :=
by sorry

end wrong_observation_value_l552_55240


namespace mystery_compound_is_nh4_l552_55223

/-- Represents the atomic weight of an element -/
structure AtomicWeight where
  value : ℝ
  positive : value > 0

/-- Represents a chemical compound -/
structure Compound where
  molecularWeight : ℝ
  nitrogenCount : ℕ
  otherElementCount : ℕ
  otherElementWeight : AtomicWeight

/-- The atomic weight of nitrogen -/
def nitrogenWeight : AtomicWeight :=
  { value := 14.01, positive := by norm_num }

/-- The atomic weight of hydrogen -/
def hydrogenWeight : AtomicWeight :=
  { value := 1.01, positive := by norm_num }

/-- The compound in question -/
def mysteryCompound : Compound :=
  { molecularWeight := 18,
    nitrogenCount := 1,
    otherElementCount := 4,
    otherElementWeight := hydrogenWeight }

/-- Theorem stating that the mystery compound must be NH₄ -/
theorem mystery_compound_is_nh4 :
  ∀ (c : Compound),
    c.molecularWeight = 18 →
    c.nitrogenCount = 1 →
    c.otherElementWeight.value * c.otherElementCount + nitrogenWeight.value = c.molecularWeight →
    c = mysteryCompound :=
  sorry

end mystery_compound_is_nh4_l552_55223


namespace polygon_with_six_diagonals_has_nine_vertices_l552_55239

/-- The number of vertices in a polygon given the number of diagonals from one vertex -/
def vertices_from_diagonals (diagonals : ℕ) : ℕ := diagonals + 3

/-- Theorem: A polygon with 6 diagonals drawn from one vertex has 9 vertices -/
theorem polygon_with_six_diagonals_has_nine_vertices :
  vertices_from_diagonals 6 = 9 := by
  sorry

end polygon_with_six_diagonals_has_nine_vertices_l552_55239


namespace inverse_as_polynomial_of_N_l552_55296

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_as_polynomial_of_N :
  let c : ℚ := 1 / 36
  let d : ℚ := -1 / 12
  N⁻¹ = c • (N ^ 2) + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by sorry

end inverse_as_polynomial_of_N_l552_55296


namespace right_triangle_from_conditions_l552_55290

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that it is a right triangle. -/
theorem right_triangle_from_conditions (a b c : ℝ) (h1 : a + c = 2 * b) (h2 : c - a = 1 / 2 * b) :
  c^2 = a^2 + b^2 := by
  sorry

end right_triangle_from_conditions_l552_55290


namespace tangent_line_constraint_l552_55207

/-- Given a cubic function f(x) = x³ - (1/2)x² + bx + c, 
    if f has a tangent line parallel to y = 1, then b ≤ 1/12 -/
theorem tangent_line_constraint (b c : ℝ) : 
  (∃ x : ℝ, (3*x^2 - x + b) = 1) → b ≤ 1/12 := by
sorry

end tangent_line_constraint_l552_55207


namespace third_butcher_packages_l552_55221

/-- Represents the number of packages delivered by each butcher and their delivery times -/
structure Delivery :=
  (x y z : ℕ)
  (t1 t2 t3 : ℕ)

/-- Defines the conditions of the delivery problem -/
def DeliveryProblem (d : Delivery) : Prop :=
  d.x = 10 ∧
  d.y = 7 ∧
  d.t1 = 8 ∧
  d.t2 = 10 ∧
  d.t3 = 18 ∧
  4 * d.x + 4 * d.y + 4 * d.z = 100

/-- Theorem stating that under the given conditions, the third butcher delivered 8 packages -/
theorem third_butcher_packages (d : Delivery) (h : DeliveryProblem d) : d.z = 8 := by
  sorry

end third_butcher_packages_l552_55221


namespace philatelist_stamps_problem_l552_55293

theorem philatelist_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end philatelist_stamps_problem_l552_55293


namespace intersection_and_union_range_of_a_l552_55217

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (C a ⊆ B) ↔ (2 ≤ a ∧ a ≤ 8) := by sorry

end intersection_and_union_range_of_a_l552_55217


namespace two_digit_number_property_l552_55244

theorem two_digit_number_property : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by
sorry

end two_digit_number_property_l552_55244


namespace passengers_in_nine_buses_l552_55204

/-- Given that 110 passengers fit in 5 buses, prove that 198 passengers fit in 9 buses. -/
theorem passengers_in_nine_buses :
  ∀ (passengers_per_bus : ℕ),
    110 = 5 * passengers_per_bus →
    9 * passengers_per_bus = 198 := by
  sorry

end passengers_in_nine_buses_l552_55204


namespace complement_union_M_N_l552_55218

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p ≠ (2, 3)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {(2, 3)} := by sorry

end complement_union_M_N_l552_55218


namespace number_division_problem_l552_55297

theorem number_division_problem (n : ℕ) : 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 40) → 
  n = 220040 := by
sorry

end number_division_problem_l552_55297


namespace sam_above_average_l552_55234

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := bridget_count - 2

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by the three observers -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

/-- Theorem stating that Sam counted 2 more shooting stars than the average -/
theorem sam_above_average : sam_count - average_count = 2 := by
  sorry

end sam_above_average_l552_55234


namespace trigonometric_identity_l552_55212

theorem trigonometric_identity (x y z : Real) 
  (hm : m = Real.sin x / Real.sin (y - z))
  (hn : n = Real.sin y / Real.sin (z - x))
  (hp : p = Real.sin z / Real.sin (x - y)) :
  m * n + n * p + p * m = -1 :=
by sorry

end trigonometric_identity_l552_55212


namespace fraction_to_decimal_l552_55230

theorem fraction_to_decimal : (51 : ℚ) / 160 = 0.31875 := by
  sorry

end fraction_to_decimal_l552_55230


namespace matrix_multiplication_result_l552_55265

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 6, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![9, -3; 2, 2]
  A * B = !![25, -11; 46, -26] := by
  sorry

end matrix_multiplication_result_l552_55265


namespace absolute_value_inequality_solution_set_l552_55249

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≥ 5} = {x : ℝ | x ≥ 6 ∨ x ≤ -4} := by
  sorry

end absolute_value_inequality_solution_set_l552_55249


namespace quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l552_55285

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + x + 2 ≠ 0 :=
by
  sorry

-- Auxiliary definitions and theorems
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem no_real_roots_iff_negative_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ discriminant a b c < 0 :=
by
  sorry

end quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l552_55285


namespace dinner_task_assignments_l552_55250

theorem dinner_task_assignments (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) = 30 := by
  sorry

end dinner_task_assignments_l552_55250


namespace total_cost_is_correct_l552_55215

/-- Calculates the total cost of tickets for a high school musical performance. -/
def calculate_total_cost (adult_price : ℚ) (child_price : ℚ) (senior_price : ℚ) (student_price : ℚ)
  (num_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) (num_students : ℕ) : ℚ :=
  let adult_cost := num_adults * adult_price
  let child_cost := (num_children - 1) * child_price  -- Family package applied
  let senior_cost := num_seniors * senior_price * (1 - 1/10)  -- 10% senior discount
  let student_cost := 2 * student_price + (student_price / 2)  -- Student promotion
  adult_cost + child_cost + senior_cost + student_cost

/-- Theorem stating that the total cost for the given scenario is $103.30. -/
theorem total_cost_is_correct :
  calculate_total_cost 12 10 8 9 4 3 2 3 = 1033/10 := by sorry

end total_cost_is_correct_l552_55215


namespace football_players_count_l552_55251

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) :
  total_players = 55 →
  cricket_players = 15 →
  hockey_players = 12 →
  softball_players = 15 →
  total_players = cricket_players + hockey_players + softball_players + 13 :=
by
  sorry

end football_players_count_l552_55251


namespace bananas_undetermined_l552_55292

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  totalOranges : ℕ
  orangeGroups : ℕ
  orangesPerGroup : ℕ
  bananaGroups : ℕ

/-- Philip's actual fruit collection -/
def philipsCollection : FruitCollection := {
  totalOranges := 384,
  orangeGroups := 16,
  orangesPerGroup := 24,
  bananaGroups := 345
}

/-- Predicate to check if the number of bananas can be determined -/
def canDetermineBananas (c : FruitCollection) : Prop :=
  ∃ (bananasPerGroup : ℕ), True  -- Placeholder, always true

/-- Theorem stating that the number of bananas cannot be determined -/
theorem bananas_undetermined (c : FruitCollection) 
  (h1 : c.totalOranges = c.orangeGroups * c.orangesPerGroup) :
  ¬ canDetermineBananas c := by
  sorry

#check bananas_undetermined philipsCollection

end bananas_undetermined_l552_55292


namespace ant_path_theorem_l552_55243

/-- Represents the three concentric square paths -/
structure SquarePaths where
  a : ℝ  -- Side length of the smallest square
  b : ℝ  -- Side length of the middle square
  c : ℝ  -- Side length of the largest square
  h1 : 0 < a
  h2 : a < b
  h3 : b < c

/-- Represents the positions of the three ants -/
structure AntPositions (p : SquarePaths) where
  mu : ℝ  -- Distance traveled by Mu
  ra : ℝ  -- Distance traveled by Ra
  vey : ℝ  -- Distance traveled by Vey
  h1 : mu = p.c  -- Mu reaches the lower-right corner of the largest square
  h2 : ra = p.c - 1  -- Ra's position on the right side of the middle square
  h3 : vey = 2 * (p.c - p.b + 1)  -- Vey's position on the right side of the smallest square

/-- The main theorem stating the conditions and the result -/
theorem ant_path_theorem (p : SquarePaths) (pos : AntPositions p) :
  (p.c - p.b = p.b - p.a) ∧ (p.b - p.a = 2) →
  p.a = 4 ∧ p.b = 6 ∧ p.c = 8 := by
  sorry

end ant_path_theorem_l552_55243


namespace power_of_three_division_l552_55226

theorem power_of_three_division : (3 : ℕ) ^ 2023 / 9 = (3 : ℕ) ^ 2021 := by
  sorry

end power_of_three_division_l552_55226


namespace time_to_return_home_l552_55253

/-- The time it takes Eric to go to the park -/
def time_to_park : ℕ := 20 + 10

/-- The factor by which the return trip is longer than the trip to the park -/
def return_factor : ℕ := 3

/-- Theorem: The time it takes Eric to return home is 90 minutes -/
theorem time_to_return_home : time_to_park * return_factor = 90 := by
  sorry

end time_to_return_home_l552_55253


namespace dilation_of_negative_i_l552_55277

def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_of_negative_i :
  let c : ℂ := 2 - 3*I
  let k : ℝ := 3
  let z : ℂ := -I
  dilation c k z = -4 + 3*I := by
  sorry

end dilation_of_negative_i_l552_55277


namespace coyote_prints_time_l552_55210

/-- The time elapsed since the coyote left the prints -/
def time_elapsed : ℝ := 2

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The speed of Darrel in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote in hours -/
def catch_up_time : ℝ := 1

theorem coyote_prints_time :
  time_elapsed * coyote_speed = darrel_speed * catch_up_time :=
sorry

end coyote_prints_time_l552_55210


namespace same_color_probability_l552_55238

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 12

/-- Represents the number of red sides on each die -/
def redSides : ℕ := 3

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 4

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 2

/-- Theorem stating the probability of rolling the same color on both dice -/
theorem same_color_probability : 
  (redSides^2 + blueSides^2 + greenSides^2 + purpleSides^2) / totalSides^2 = 19 / 72 := by
  sorry

end same_color_probability_l552_55238


namespace land_conversion_equation_l552_55291

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : True :=
  let original_forest : ℝ := 108
  let original_arable : ℝ := 54
  let conversion_percentage : ℝ := 0.2
  let new_forest : ℝ := original_forest + x
  let new_arable : ℝ := original_arable - x
  let equation := (new_arable = conversion_percentage * new_forest)
by
  sorry

end land_conversion_equation_l552_55291


namespace equivalent_operations_l552_55276

theorem equivalent_operations (x : ℚ) : 
  (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end equivalent_operations_l552_55276


namespace divisibility_by_27_l552_55271

theorem divisibility_by_27 (t : ℤ) : 
  27 ∣ (7 * (27 * t + 16)^4 + 19 * (27 * t + 16) + 25) := by
sorry

end divisibility_by_27_l552_55271


namespace quadratic_roots_l552_55298

theorem quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 2 * a^2 + a * a + b = 0 ∧ 2 * b^2 + a * b + b = 0) : 
  a = 1/2 ∧ b = -3/4 := by
  sorry

end quadratic_roots_l552_55298


namespace marathon_distance_theorem_l552_55282

/-- Represents the length of a marathon in miles and yards. -/
structure MarathonLength where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance in miles and yards after running multiple marathons. -/
def totalDistance (marathonLength : MarathonLength) (numMarathons : ℕ) : MarathonLength :=
  let totalMiles := marathonLength.miles * numMarathons
  let totalYards := marathonLength.yards * numMarathons
  let extraMiles := totalYards / 1760
  let remainingYards := totalYards % 1760
  { miles := totalMiles + extraMiles, yards := remainingYards }

theorem marathon_distance_theorem :
  let marathonLength : MarathonLength := { miles := 26, yards := 385 }
  let numMarathons : ℕ := 15
  let result := totalDistance marathonLength numMarathons
  result.miles = 393 ∧ result.yards = 495 := by
  sorry

end marathon_distance_theorem_l552_55282


namespace circle_on_parabola_passes_through_focus_l552_55203

/-- A circle with center on the parabola y^2 = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_passes_through_focus (c : ℝ × ℝ) (r : ℝ) :
  c.2^2 = 8 * c.1 →  -- center is on the parabola y^2 = 8x
  r = c.1 + 2 →      -- circle is tangent to x + 2 = 0
  (c.1 - 2)^2 + c.2^2 = r^2  -- point (2, 0) is on the circle
  := by sorry

end circle_on_parabola_passes_through_focus_l552_55203


namespace no_four_digit_sum12_div11and5_l552_55256

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem no_four_digit_sum12_div11and5 :
  ¬ ∃ n : ℕ, is_four_digit n ∧ digit_sum n = 12 ∧ n % 11 = 0 ∧ n % 5 = 0 :=
sorry

end no_four_digit_sum12_div11and5_l552_55256


namespace maddie_tshirt_cost_l552_55248

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs : ℕ) (white_per_pack blue_per_pack : ℕ) (cost_per_shirt : ℕ) : ℕ :=
  let total_shirts := white_packs * white_per_pack + blue_packs * blue_per_pack
  total_shirts * cost_per_shirt

/-- Theorem stating that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end maddie_tshirt_cost_l552_55248


namespace set_intersection_example_l552_55252

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end set_intersection_example_l552_55252


namespace find_p_l552_55288

theorem find_p : ∃ (d q : ℝ), ∀ (x : ℝ),
  (4 * x^2 - 2 * x + 5/2) * (d * x^2 + p * x + q) = 12 * x^4 - 7 * x^3 + 12 * x^2 - 15/2 * x + 10/2 →
  p = -1/4 := by
  sorry

end find_p_l552_55288


namespace cylinder_surface_area_l552_55236

theorem cylinder_surface_area (V : Real) (d : Real) (h : Real) : 
  V = 500 * Real.pi / 3 →  -- Volume of the sphere
  d = 8 →                  -- Diameter of the cylinder base
  h = 6 →                  -- Height of the cylinder (derived from the problem)
  2 * Real.pi * (d/2) * h + 2 * Real.pi * (d/2)^2 = 80 * Real.pi := by
  sorry

end cylinder_surface_area_l552_55236


namespace coefficient_of_b_squared_l552_55274

theorem coefficient_of_b_squared (a : ℝ) : 
  (∃ b₁ b₂ : ℝ, b₁ + b₂ = 4.5 ∧ 
    (∀ b : ℝ, 4 * b^4 - a * b^2 + 100 = 0 → b ≤ b₁ ∧ b ≤ b₂) ∧
    (4 * b₁^4 - a * b₁^2 + 100 = 0) ∧ 
    (4 * b₂^4 - a * b₂^2 + 100 = 0)) →
  a = 4.5 := by
sorry

end coefficient_of_b_squared_l552_55274


namespace at_least_one_not_less_than_two_l552_55289

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l552_55289


namespace parabola_coefficient_sum_l552_55263

/-- A parabola with equation x = ay² + by + c, vertex at (3, -6), and passing through (2, -4) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : 3 = a * (-6)^2 + b * (-6) + c
  point_condition : 2 = a * (-4)^2 + b * (-4) + c

/-- The sum of coefficients a, b, and c for the given parabola is -25/4 -/
theorem parabola_coefficient_sum (p : Parabola) : p.a + p.b + p.c = -25/4 := by
  sorry

#check parabola_coefficient_sum

end parabola_coefficient_sum_l552_55263


namespace remaining_problems_calculation_l552_55232

/-- Given a number of worksheets, problems per worksheet, and graded worksheets,
    calculate the number of remaining problems to grade. -/
def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem remaining_problems_calculation :
  remaining_problems 16 4 8 = 32 := by
  sorry

end remaining_problems_calculation_l552_55232


namespace compare_expressions_l552_55237

theorem compare_expressions : -|(-3/4)| < -(-4/5) := by
  sorry

end compare_expressions_l552_55237


namespace solution_equation1_solution_equation2_l552_55242

-- Define the first equation
def equation1 (x : ℝ) : Prop := 3 * x - 5 = 6 * x - 8

-- Define the second equation
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - (2 * x - 1) / 3 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end solution_equation1_solution_equation2_l552_55242


namespace egg_grouping_l552_55206

theorem egg_grouping (total_eggs : ℕ) (group_size : ℕ) (h1 : total_eggs = 9) (h2 : group_size = 3) :
  total_eggs / group_size = 3 := by
  sorry

end egg_grouping_l552_55206


namespace chocolate_division_l552_55222

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 35 / 4 ∧
  num_piles = 5 ∧
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 7 / 2 := by
  sorry

end chocolate_division_l552_55222


namespace simplify_expression_1_simplify_expression_2_l552_55279

-- Problem 1
theorem simplify_expression_1 :
  Real.sqrt 8 + 2 * Real.sqrt 3 - (Real.sqrt 27 - Real.sqrt 2) = 3 * Real.sqrt 2 - Real.sqrt 3 :=
by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (ha : a > 0) :
  Real.sqrt (4 * a^2 * b^3) = 2 * a * b * Real.sqrt b :=
by sorry

end simplify_expression_1_simplify_expression_2_l552_55279


namespace right_triangle_area_l552_55286

theorem right_triangle_area (a b c : ℕ) : 
  a = 7 →                  -- One leg is 7
  a * a + b * b = c * c →  -- Pythagorean theorem (right triangle)
  a * b = 168 →            -- Area is 84 (2 * 84 = 168)
  (∃ (S : ℕ), S = 84 ∧ S = a * b / 2) :=
by sorry

end right_triangle_area_l552_55286


namespace inequality_equivalence_l552_55268

theorem inequality_equivalence (x y : ℝ) :
  (2 * y + 3 * x > Real.sqrt (9 * x^2)) ↔
  ((x ≥ 0 ∧ y > 0) ∨ (x < 0 ∧ y > -3 * x)) :=
by sorry

end inequality_equivalence_l552_55268


namespace product_of_roots_l552_55295

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end product_of_roots_l552_55295


namespace train_passing_jogger_train_passes_jogger_in_39_seconds_l552_55261

/-- The time taken for a train to pass a jogger -/
theorem train_passing_jogger (jogger_speed : ℝ) (train_speed : ℝ) 
  (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 39 seconds -/
theorem train_passes_jogger_in_39_seconds : 
  train_passing_jogger 9 45 120 270 = 39 := by
  sorry

end train_passing_jogger_train_passes_jogger_in_39_seconds_l552_55261


namespace remainder_when_consecutive_primes_l552_55208

theorem remainder_when_consecutive_primes (n : ℕ) :
  Nat.Prime (n + 3) ∧ Nat.Prime (n + 7) → n % 6 = 4 := by
  sorry

end remainder_when_consecutive_primes_l552_55208


namespace root_product_value_l552_55219

theorem root_product_value : 
  ∀ (a b c d : ℝ), 
  (a^2 + 2000*a + 1 = 0) → 
  (b^2 + 2000*b + 1 = 0) → 
  (c^2 - 2008*c + 1 = 0) → 
  (d^2 - 2008*d + 1 = 0) → 
  (a+c)*(b+c)*(a-d)*(b-d) = 32064 := by
sorry

end root_product_value_l552_55219


namespace eight_people_line_up_with_pair_l552_55235

/-- The number of ways to arrange n people in a line. -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line, 
    with 2 specific people always standing together. -/
def arrangementsWithPair (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 1)

/-- Theorem: There are 10080 ways for 8 people to line up
    with 2 specific people always standing together. -/
theorem eight_people_line_up_with_pair : 
  arrangementsWithPair 8 = 10080 := by
  sorry


end eight_people_line_up_with_pair_l552_55235


namespace customers_without_tip_waiter_tip_problem_l552_55211

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  let total_customers := initial_customers + additional_customers
  total_customers - customers_with_tip

theorem waiter_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end customers_without_tip_waiter_tip_problem_l552_55211


namespace woman_birth_year_l552_55255

/-- A woman born in the latter half of the nineteenth century was y years old in the year y^2. -/
theorem woman_birth_year (y : ℕ) (h1 : 1850 ≤ y^2 - y) (h2 : y^2 - y < 1900) (h3 : y^2 = y + 1892) : 
  y^2 - y = 1892 := by
  sorry

end woman_birth_year_l552_55255


namespace negation_equivalence_l552_55241

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) :=
by sorry

end negation_equivalence_l552_55241


namespace expand_expression_l552_55246

theorem expand_expression (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

end expand_expression_l552_55246


namespace dispatch_plans_count_l552_55269

-- Define the total number of students
def total_students : ℕ := 6

-- Define the number of students needed for each day
def sunday_students : ℕ := 2
def friday_students : ℕ := 1
def saturday_students : ℕ := 1

-- Define the total number of students needed
def total_needed : ℕ := sunday_students + friday_students + saturday_students

-- Theorem statement
theorem dispatch_plans_count : 
  (Nat.choose total_students sunday_students) * 
  (Nat.choose (total_students - sunday_students) friday_students) * 
  (Nat.choose (total_students - sunday_students - friday_students) saturday_students) = 180 :=
by sorry

end dispatch_plans_count_l552_55269


namespace opposite_pairs_l552_55294

theorem opposite_pairs : 
  ¬((-2 : ℝ) = -(1/2)) ∧ 
  ¬(|(-1)| = -1) ∧ 
  ¬(((-3)^2 : ℝ) = -(3^2)) ∧ 
  (-5 : ℝ) = -(-(-5)) := by sorry

end opposite_pairs_l552_55294


namespace correct_mark_is_90_l552_55275

/-- Proves that the correct mark is 90 given the problem conditions --/
theorem correct_mark_is_90 (n : ℕ) (initial_avg correct_avg wrong_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  correct_avg = 96 →
  wrong_mark = 50 →
  ∃ x : ℚ, (n * initial_avg - wrong_mark + x) / n = correct_avg ∧ x = 90 :=
by sorry

end correct_mark_is_90_l552_55275


namespace area_triangle_AOB_l552_55216

/-- Given a sector AOB with area 2π/3 and radius 2, the area of triangle AOB is √3. -/
theorem area_triangle_AOB (S : ℝ) (r : ℝ) (h1 : S = 2 * π / 3) (h2 : r = 2) :
  (1 / 2) * r^2 * Real.sin (S / r^2) = Real.sqrt 3 := by
  sorry

end area_triangle_AOB_l552_55216


namespace cement_calculation_l552_55220

theorem cement_calculation (initial bought total : ℕ) 
  (h1 : initial = 98)
  (h2 : bought = 215)
  (h3 : total = 450) :
  total - (initial + bought) = 137 := by
  sorry

end cement_calculation_l552_55220


namespace units_digit_2019_power_2019_l552_55209

theorem units_digit_2019_power_2019 : (2019^2019) % 10 = 9 := by
  sorry

end units_digit_2019_power_2019_l552_55209


namespace cost_per_book_l552_55283

def total_books : ℕ := 14
def total_spent : ℕ := 224

theorem cost_per_book : total_spent / total_books = 16 := by
  sorry

end cost_per_book_l552_55283


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l552_55262

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  let r₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  r₁ + r₂ = 5 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l552_55262


namespace nancy_albums_l552_55228

theorem nancy_albums (total_pictures : ℕ) (first_album : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pictures = 51)
  (h2 : first_album = 11)
  (h3 : pics_per_album = 5) :
  (total_pictures - first_album) / pics_per_album = 8 := by
  sorry

end nancy_albums_l552_55228


namespace reflect_triangle_xy_l552_55280

/-- A triangle in a 2D coordinate plane -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Reflection of a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection of a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Composition of reflections over x-axis and y-axis -/
def reflect_xy (p : ℝ × ℝ) : ℝ × ℝ := reflect_y (reflect_x p)

/-- Theorem: Reflecting a triangle over x-axis then y-axis negates both coordinates -/
theorem reflect_triangle_xy (t : Triangle) :
  let t' := Triangle.mk (reflect_xy t.v1) (reflect_xy t.v2) (reflect_xy t.v3)
  t'.v1 = (-t.v1.1, -t.v1.2) ∧
  t'.v2 = (-t.v2.1, -t.v2.2) ∧
  t'.v3 = (-t.v3.1, -t.v3.2) := by
  sorry

end reflect_triangle_xy_l552_55280


namespace athletes_simultaneous_return_l552_55213

/-- The time in minutes for Athlete A to complete one lap -/
def timeA : ℕ := 4

/-- The time in minutes for Athlete B to complete one lap -/
def timeB : ℕ := 5

/-- The time in minutes for Athlete C to complete one lap -/
def timeC : ℕ := 6

/-- The length of the circular track in meters -/
def trackLength : ℕ := 1000

/-- The time when all athletes simultaneously return to the starting point -/
def simultaneousReturnTime : ℕ := 60

theorem athletes_simultaneous_return :
  Nat.lcm (Nat.lcm timeA timeB) timeC = simultaneousReturnTime :=
sorry

end athletes_simultaneous_return_l552_55213


namespace sum_of_repeating_decimals_l552_55267

-- Define the repeating decimals
def repeating_six : ℚ := 2/3
def repeating_seven : ℚ := 7/9

-- State the theorem
theorem sum_of_repeating_decimals : 
  repeating_six + repeating_seven = 13/9 := by sorry

end sum_of_repeating_decimals_l552_55267


namespace sqrt_primes_not_arithmetic_progression_l552_55233

theorem sqrt_primes_not_arithmetic_progression (a b c : ℕ) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (d : ℝ), (Real.sqrt (a : ℝ) + d = Real.sqrt (b : ℝ) ∧ 
               Real.sqrt (b : ℝ) + d = Real.sqrt (c : ℝ)) := by
  sorry

end sqrt_primes_not_arithmetic_progression_l552_55233


namespace max_sum_same_color_as_center_l552_55200

/-- Represents a 5x5 checkerboard grid with alternating colors -/
def Grid := Fin 5 → Fin 5 → Bool

/-- A valid numbering of the grid satisfies the adjacent consecutive property -/
def ValidNumbering (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : Prop := sorry

/-- The sum of numbers in squares of the same color as the center square -/
def SumSameColorAsCenter (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : ℕ := sorry

/-- The maximum sum of numbers in squares of the same color as the center square -/
def MaxSumSameColorAsCenter (g : Grid) : ℕ := sorry

theorem max_sum_same_color_as_center (g : Grid) :
  MaxSumSameColorAsCenter g = 169 := by sorry

end max_sum_same_color_as_center_l552_55200


namespace triangle_area_l552_55287

/-- The area of a triangle with vertices at (-4,3), (0,6), and (2,-2) is 19 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-4, 3)
  let B : ℝ × ℝ := (0, 6)
  let C : ℝ × ℝ := (2, -2)
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 19 := by
  sorry


end triangle_area_l552_55287


namespace book_page_words_l552_55227

theorem book_page_words (total_pages : ℕ) (words_per_page : ℕ) : 
  total_pages = 150 →
  50 ≤ words_per_page →
  words_per_page ≤ 150 →
  (total_pages * words_per_page) % 221 = 217 →
  words_per_page = 135 := by
sorry

end book_page_words_l552_55227


namespace defective_shipped_percentage_l552_55202

theorem defective_shipped_percentage
  (total_units : ℕ)
  (defective_rate : ℚ)
  (shipped_rate : ℚ)
  (h1 : defective_rate = 5 / 100)
  (h2 : shipped_rate = 4 / 100) :
  (defective_rate * shipped_rate) * 100 = 0.2 := by
sorry

end defective_shipped_percentage_l552_55202


namespace no_solution_for_equation_l552_55214

theorem no_solution_for_equation : ¬ ∃ (x : ℝ), (x - 8) / (x - 7) - 8 = 1 / (7 - x) := by
  sorry

end no_solution_for_equation_l552_55214


namespace square_difference_equals_product_l552_55281

theorem square_difference_equals_product : (15 + 7)^2 - (7^2 + 15^2) = 210 := by
  sorry

end square_difference_equals_product_l552_55281


namespace optimal_pricing_strategy_l552_55260

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal marked price for the merchant's pricing strategy -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry


end optimal_pricing_strategy_l552_55260


namespace arithmetic_sequence_problem_l552_55231

/-- Given an arithmetic sequence {aₙ}, prove that a₁₈ = 8 when a₄ + a₈ = 10 and a₁₀ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 4 + a 8 = 10 →
  a 10 = 6 →
  a 18 = 8 := by
sorry

end arithmetic_sequence_problem_l552_55231


namespace x_value_theorem_l552_55257

theorem x_value_theorem (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) :
  x = 2*y^2 + 6*y + 4 := by
  sorry

end x_value_theorem_l552_55257


namespace large_shoes_count_l552_55266

/-- The number of pairs of large-size shoes initially stocked by the shop -/
def L : ℕ := sorry

/-- The number of pairs of medium-size shoes initially stocked by the shop -/
def medium_shoes : ℕ := 50

/-- The number of pairs of small-size shoes initially stocked by the shop -/
def small_shoes : ℕ := 24

/-- The number of pairs of shoes sold by the shop -/
def sold_shoes : ℕ := 83

/-- The number of pairs of shoes left after selling -/
def left_shoes : ℕ := 13

theorem large_shoes_count : L = 22 := by
  sorry

end large_shoes_count_l552_55266


namespace ab_greater_ac_l552_55278

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end ab_greater_ac_l552_55278


namespace tile_perimeter_change_l552_55264

/-- Represents a shape made of square tiles -/
structure TileShape where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a shape and returns the new perimeter -/
def add_tiles (shape : TileShape) (new_tiles : ℕ) : Set ℕ :=
  sorry

theorem tile_perimeter_change (initial_shape : TileShape) :
  initial_shape.tiles = 10 →
  initial_shape.perimeter = 16 →
  ∃ (new_perimeter : Set ℕ),
    new_perimeter = add_tiles initial_shape 2 ∧
    new_perimeter = {23, 25} :=
by sorry

end tile_perimeter_change_l552_55264


namespace square_difference_divided_by_nine_l552_55245

theorem square_difference_divided_by_nine : (109^2 - 100^2) / 9 = 209 := by
  sorry

end square_difference_divided_by_nine_l552_55245


namespace imaginary_part_of_complex_fraction_l552_55272

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (7 + i) / (3 + 4 * i)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_fraction_l552_55272


namespace equation_solution_l552_55270

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (15 * x)^3 → x = 27 / 5 := by
  sorry

end equation_solution_l552_55270


namespace min_sum_squared_eccentricities_l552_55254

/-- Given an ellipse and a hyperbola sharing the same foci, with one of their
    intersection points P forming an angle ∠F₁PF₂ = 60°, and their respective
    eccentricities e₁ and e₂, the minimum value of e₁² + e₂² is 1 + √3/2. -/
theorem min_sum_squared_eccentricities (e₁ e₂ : ℝ) 
  (h_ellipse : e₁ ∈ Set.Ioo 0 1)
  (h_hyperbola : e₂ > 1)
  (h_shared_foci : True)  -- Represents the condition that the ellipse and hyperbola share foci
  (h_intersection : True)  -- Represents the condition that P is an intersection point
  (h_angle : True)  -- Represents the condition that ∠F₁PF₂ = 60°
  : (∀ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 → ε₂ > 1 → ε₁^2 + ε₂^2 ≥ 1 + Real.sqrt 3 / 2) ∧ 
    (∃ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 ∧ ε₂ > 1 ∧ ε₁^2 + ε₂^2 = 1 + Real.sqrt 3 / 2) := by
  sorry

end min_sum_squared_eccentricities_l552_55254


namespace ball_problem_l552_55229

theorem ball_problem (x : ℕ) : 
  (x > 0) →                                      -- Ensure x is positive
  ((x + 1) / (2 * x + 1) - x / (2 * x) = 1 / 22) →  -- Probability condition
  (2 * x = 10) :=                                -- Conclusion
by sorry

end ball_problem_l552_55229


namespace polynomial_equality_l552_55299

theorem polynomial_equality (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + 1) ^ 4 = a + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  a - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end polynomial_equality_l552_55299


namespace no_relationship_between_mites_and_wilt_resistance_l552_55258

def total_plants : ℕ := 88
def infected_plants : ℕ := 33
def resistant_infected : ℕ := 19
def susceptible_infected : ℕ := 14
def not_infected_plants : ℕ := 55
def resistant_not_infected : ℕ := 28
def susceptible_not_infected : ℕ := 27

def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 3841 / 1000

theorem no_relationship_between_mites_and_wilt_resistance :
  chi_square total_plants resistant_infected resistant_not_infected 
             susceptible_infected susceptible_not_infected < critical_value := by
  sorry

end no_relationship_between_mites_and_wilt_resistance_l552_55258


namespace inequality_solution_set_l552_55225

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 15) ↔ 
  (x < -3/2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3)) :=
by sorry

end inequality_solution_set_l552_55225
