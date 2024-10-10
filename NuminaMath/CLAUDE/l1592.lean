import Mathlib

namespace rectangle_max_volume_l1592_159234

def bar_length : ℝ := 18

theorem rectangle_max_volume (length width height : ℝ) :
  length > 0 ∧ width > 0 ∧ height > 0 →
  length = 2 * width →
  2 * (length + width) = bar_length →
  length = 2 ∧ width = 1 ∧ height = 1.5 →
  ∀ (l w h : ℝ), l > 0 ∧ w > 0 ∧ h > 0 →
    l = 2 * w →
    2 * (l + w) = bar_length →
    l * w * h ≤ length * width * height :=
by sorry

end rectangle_max_volume_l1592_159234


namespace complex_modulus_problem_l1592_159295

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l1592_159295


namespace no_solution_for_equation_l1592_159224

theorem no_solution_for_equation : ¬∃ (a b : ℕ+), 
  a * b + 100 = 25 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end no_solution_for_equation_l1592_159224


namespace linear_function_properties_l1592_159229

-- Define the linear function
def f (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties
  (k b : ℝ)
  (h1 : f k b 1 = 0)
  (h2 : f k b 0 = 2)
  (m : ℝ)
  (h3 : -2 < m)
  (h4 : m ≤ 3) :
  k = -2 ∧ b = 2 ∧ -4 ≤ f k b m ∧ f k b m < 6 :=
by sorry

end linear_function_properties_l1592_159229


namespace total_wheels_in_parking_lot_l1592_159293

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_parking_lot : 
  total_cars * wheels_per_car = 48 := by
  sorry

end total_wheels_in_parking_lot_l1592_159293


namespace five_by_five_decomposition_l1592_159210

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Checks if a list of rectangles can fit in a grid -/
def canFitInGrid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  (rectangles.map (λ r => r.width * r.height)).sum = grid.size * grid.size

/-- Theorem: A 5x5 grid can be decomposed into 1x3 and 1x4 rectangles -/
theorem five_by_five_decomposition :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, r.width = 1 ∧ (r.height = 3 ∨ r.height = 4)) ∧
    canFitInGrid { size := 5 } rectangles :=
  sorry

end five_by_five_decomposition_l1592_159210


namespace prepend_append_divisible_by_72_l1592_159270

theorem prepend_append_divisible_by_72 : ∃ (a b : Nat), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 100 + b = 4104) ∧ 
  (4104 % 72 = 0) := by
  sorry

end prepend_append_divisible_by_72_l1592_159270


namespace gorilla_to_cat_dog_ratio_l1592_159240

/-- Represents the lengths of animal videos and their ratio -/
structure AnimalVideos where
  cat_length : ℕ
  dog_length : ℕ
  total_time : ℕ
  gorilla_length : ℕ
  ratio : Rat

/-- Theorem stating the ratio of gorilla video length to combined cat and dog video length -/
theorem gorilla_to_cat_dog_ratio (v : AnimalVideos) 
  (h1 : v.cat_length = 4)
  (h2 : v.dog_length = 2 * v.cat_length)
  (h3 : v.total_time = 36)
  (h4 : v.gorilla_length = v.total_time - (v.cat_length + v.dog_length))
  (h5 : v.ratio = v.gorilla_length / (v.cat_length + v.dog_length)) :
  v.ratio = 2 := by
  sorry

end gorilla_to_cat_dog_ratio_l1592_159240


namespace sphere_volume_from_surface_area_l1592_159262

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 : ℝ) = 400 * π → 
    (4 / 3 : ℝ) * π * r^3 = (4000 / 3 : ℝ) * π := by
  sorry

end sphere_volume_from_surface_area_l1592_159262


namespace emily_final_score_l1592_159245

def emily_game (round1 round2 round3 round4 round5 round6_initial : Int) : Int :=
  let round6 := round6_initial - (2 * round5) / 3
  round1 + round2 + round3 + round4 + round5 + round6

theorem emily_final_score :
  emily_game 16 33 (-25) 46 12 30 = 104 := by
  sorry

end emily_final_score_l1592_159245


namespace min_wins_for_playoffs_l1592_159223

theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points min_points : ℕ) :
  total_games = 22 →
  win_points = 2 →
  loss_points = 1 →
  min_points = 36 →
  (∃ (wins : ℕ), 
    wins ≤ total_games ∧ 
    wins * win_points + (total_games - wins) * loss_points ≥ min_points ∧
    ∀ (w : ℕ), w < wins → w * win_points + (total_games - w) * loss_points < min_points) →
  14 = (min_points - total_games * loss_points) / (win_points - loss_points) := by
sorry

end min_wins_for_playoffs_l1592_159223


namespace complex_square_eq_neg_two_i_l1592_159216

theorem complex_square_eq_neg_two_i (z : ℂ) (a b : ℝ) :
  z = Complex.mk a b → z^2 = Complex.I * (-2) → a + b = 0 := by
  sorry

end complex_square_eq_neg_two_i_l1592_159216


namespace discount_percentage_l1592_159201

theorem discount_percentage (original_price : ℝ) (discount_rate : ℝ) 
  (h1 : discount_rate = 0.8) : 
  original_price * (1 - discount_rate) = original_price * 0.8 := by
  sorry

#check discount_percentage

end discount_percentage_l1592_159201


namespace computers_needed_after_increase_problem_solution_l1592_159282

theorem computers_needed_after_increase (initial_students : ℕ) 
  (students_per_computer : ℕ) (additional_students : ℕ) : ℕ :=
  let initial_computers := initial_students / students_per_computer
  let additional_computers := additional_students / students_per_computer
  initial_computers + additional_computers

theorem problem_solution :
  computers_needed_after_increase 82 2 16 = 49 := by
  sorry

end computers_needed_after_increase_problem_solution_l1592_159282


namespace homework_problems_left_l1592_159287

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) : 
  math_problems + science_problems - finished_problems = 15 := by
sorry

end homework_problems_left_l1592_159287


namespace sine_cosine_increasing_interval_l1592_159260

theorem sine_cosine_increasing_interval :
  ∀ (a b : ℝ), (a = -π / 2 ∧ b = 0) ↔ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y)) ∧
    (¬(∀ x y, -π ≤ x ∧ x < y ∧ y ≤ -π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, π / 2 ≤ x ∧ x < y ∧ y ≤ π → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) :=
by sorry

end sine_cosine_increasing_interval_l1592_159260


namespace annual_production_exceeds_plan_l1592_159274

/-- Represents the annual car production plan and actual quarterly production --/
structure CarProduction where
  annual_plan : ℝ
  first_quarter : ℝ
  second_quarter : ℝ
  third_quarter : ℝ
  fourth_quarter : ℝ

/-- Conditions for car production --/
def production_conditions (p : CarProduction) : Prop :=
  p.first_quarter = 0.25 * p.annual_plan ∧
  p.second_quarter = 1.08 * p.first_quarter ∧
  ∃ (k : ℝ), p.second_quarter = 11.25 * k ∧
              p.third_quarter = 12 * k ∧
              p.fourth_quarter = 13.5 * k

/-- Theorem stating that the annual production exceeds the plan by 13.2% --/
theorem annual_production_exceeds_plan (p : CarProduction) 
  (h : production_conditions p) : 
  (p.first_quarter + p.second_quarter + p.third_quarter + p.fourth_quarter) / p.annual_plan = 1.132 :=
sorry

end annual_production_exceeds_plan_l1592_159274


namespace total_students_is_thirteen_l1592_159258

/-- The total number of students in a presentation lineup, given Eunjung's position and the number of students after her. -/
def total_students (eunjung_position : ℕ) (students_after : ℕ) : ℕ :=
  eunjung_position + students_after

/-- Theorem stating that the total number of students is 13, given the conditions from the problem. -/
theorem total_students_is_thirteen :
  total_students 6 7 = 13 := by
  sorry

end total_students_is_thirteen_l1592_159258


namespace age_problem_l1592_159253

theorem age_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end age_problem_l1592_159253


namespace quarters_put_aside_l1592_159266

theorem quarters_put_aside (original_quarters : ℕ) (remaining_quarters : ℕ) : 
  (5 * original_quarters = 350) →
  (remaining_quarters + 350 = 392) →
  (original_quarters - remaining_quarters : ℚ) / original_quarters = 2 / 5 :=
by sorry

end quarters_put_aside_l1592_159266


namespace P_greater_than_Q_l1592_159243

theorem P_greater_than_Q (a : ℝ) : (a^2 + 2*a) > (3*a - 1) := by
  sorry

end P_greater_than_Q_l1592_159243


namespace quadratic_real_roots_l1592_159259

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_l1592_159259


namespace point_opposite_sides_line_value_range_l1592_159204

/-- Given that the points (3,1) and (-4,6) lie on opposite sides of the line 3x - 2y + a = 0,
    prove that the value range of a is -7 < a < 24. -/
theorem point_opposite_sides_line_value_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 → -7 < a ∧ a < 24 := by
  sorry

end point_opposite_sides_line_value_range_l1592_159204


namespace radish_carrot_ratio_l1592_159214

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  radishes / carrots = 5 := by
sorry

end radish_carrot_ratio_l1592_159214


namespace inequality_proof_l1592_159285

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end inequality_proof_l1592_159285


namespace algebraic_expression_value_l1592_159261

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a * b = 2) : 
  a^2 * b + 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end algebraic_expression_value_l1592_159261


namespace complement_of_29_45_l1592_159267

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The complement of an angle is the angle that when added to the original angle results in 90° -/
def complement (a : Angle) : Angle :=
  sorry

theorem complement_of_29_45 :
  complement ⟨29, 45⟩ = ⟨60, 15⟩ :=
sorry

end complement_of_29_45_l1592_159267


namespace trapezoid_ab_length_l1592_159286

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Assumption that AB + CD = 150
  sum_sides : ab + cd = 150
  -- Assumption that AB = 3CD
  ab_triple_cd : ab = 3 * cd
  -- Assumption that area ratio of ABC to ADC is 4:1
  area_ratio_def : area_ratio = 4 / 1

/-- Theorem stating that under given conditions, AB = 120 cm -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 120 := by
  sorry

end trapezoid_ab_length_l1592_159286


namespace island_perimeter_l1592_159235

/-- The perimeter of an island consisting of an equilateral triangle and two half circles -/
theorem island_perimeter (base : ℝ) (h : base = 4) : 
  let triangle_perimeter := 3 * base
  let half_circles_perimeter := 2 * π * base
  triangle_perimeter + half_circles_perimeter = 12 + 4 * π := by
  sorry

end island_perimeter_l1592_159235


namespace james_cd_purchase_total_l1592_159256

def cd_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_price (prices : List ℝ) (discount_rate : ℝ) : ℝ :=
  (prices.map (λ p => cd_price p discount_rate)).sum

theorem james_cd_purchase_total :
  let prices : List ℝ := [10, 10, 15, 6, 18]
  let discount_rate : ℝ := 0.1
  total_price prices discount_rate = 53.10 := by
  sorry

end james_cd_purchase_total_l1592_159256


namespace hayden_ironing_time_l1592_159248

/-- The total ironing time over 4 weeks given daily ironing times and weekly frequency -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Theorem stating that Hayden's total ironing time over 4 weeks is 160 minutes -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

#eval total_ironing_time 5 3 5 4

end hayden_ironing_time_l1592_159248


namespace meal_combinations_l1592_159272

theorem meal_combinations (entrees drinks desserts : ℕ) 
  (h_entrees : entrees = 4)
  (h_drinks : drinks = 4)
  (h_desserts : desserts = 2) : 
  entrees * drinks * desserts = 32 := by
  sorry

end meal_combinations_l1592_159272


namespace hundred_thousand_eq_scientific_notation_l1592_159250

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Definition of the number 100,000 -/
def hundred_thousand : ℕ := 100000

/-- The scientific notation of 100,000 -/
def hundred_thousand_scientific : ScientificNotation :=
  ⟨1, 5, by {sorry}⟩

/-- Theorem stating that 100,000 is equal to its scientific notation representation -/
theorem hundred_thousand_eq_scientific_notation :
  (hundred_thousand : ℝ) = hundred_thousand_scientific.coefficient * (10 : ℝ) ^ hundred_thousand_scientific.exponent :=
by sorry

end hundred_thousand_eq_scientific_notation_l1592_159250


namespace extremum_implies_f_of_2_l1592_159226

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem extremum_implies_f_of_2 (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≥ f a b 1) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = -2 →
  f a b 2 = 3 :=
sorry

end extremum_implies_f_of_2_l1592_159226


namespace sport_formulation_comparison_l1592_159212

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : DrinkRatio :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_formulation_comparison : 
  (sport.flavoring / sport.water) / (standard.flavoring / standard.water) = 1/2 := by
  sorry

end sport_formulation_comparison_l1592_159212


namespace wool_price_calculation_l1592_159244

theorem wool_price_calculation (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (profit : ℕ) : 
  num_sheep = 200 → 
  shearing_cost = 2000 → 
  wool_per_sheep = 10 → 
  profit = 38000 → 
  (profit + shearing_cost) / (num_sheep * wool_per_sheep) = 20 := by
sorry

end wool_price_calculation_l1592_159244


namespace third_vertex_coordinates_l1592_159249

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem third_vertex_coordinates (y : ℝ) :
  y < 0 →
  triangleArea ⟨8, 6⟩ ⟨0, 0⟩ ⟨0, y⟩ = 48 →
  y = -12 := by
  sorry

end third_vertex_coordinates_l1592_159249


namespace stationery_shop_restocking_l1592_159294

/-- Calculates the total number of pencils and rulers after restocking in a stationery shop. -/
theorem stationery_shop_restocking
  (initial_pencils : ℕ)
  (initial_pens : ℕ)
  (initial_rulers : ℕ)
  (sold_pencils : ℕ)
  (sold_pens : ℕ)
  (given_rulers : ℕ)
  (pencil_restock_factor : ℕ)
  (ruler_restock_factor : ℕ)
  (h1 : initial_pencils = 112)
  (h2 : initial_pens = 78)
  (h3 : initial_rulers = 46)
  (h4 : sold_pencils = 32)
  (h5 : sold_pens = 56)
  (h6 : given_rulers = 12)
  (h7 : pencil_restock_factor = 5)
  (h8 : ruler_restock_factor = 3)
  : (initial_pencils - sold_pencils + pencil_restock_factor * (initial_pencils - sold_pencils)) +
    (initial_rulers - given_rulers + ruler_restock_factor * (initial_rulers - given_rulers)) = 616 := by
  sorry

#check stationery_shop_restocking

end stationery_shop_restocking_l1592_159294


namespace inequality_proof_l1592_159279

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end inequality_proof_l1592_159279


namespace tetrahedron_coloring_tetrahedron_coloring_converse_l1592_159268

/-- The number of distinct colorings of a regular tetrahedron -/
def distinct_colorings (n : ℕ) : ℚ := (n^4 + 11*n^2) / 12

/-- The theorem stating the possible values of n -/
theorem tetrahedron_coloring (n : ℕ) (hn : n > 0) :
  distinct_colorings n = n^3 → n = 1 ∨ n = 11 := by
sorry

/-- The converse of the theorem -/
theorem tetrahedron_coloring_converse (n : ℕ) (hn : n > 0) :
  (n = 1 ∨ n = 11) → distinct_colorings n = n^3 := by
sorry

end tetrahedron_coloring_tetrahedron_coloring_converse_l1592_159268


namespace intersection_nonempty_implies_a_greater_than_negative_one_l1592_159228

theorem intersection_nonempty_implies_a_greater_than_negative_one 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | -1 ≤ x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  (A ∩ B).Nonempty →
  a > -1 := by
sorry

end intersection_nonempty_implies_a_greater_than_negative_one_l1592_159228


namespace boys_camp_total_l1592_159298

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 28) : 
  total = 200 := by
sorry

end boys_camp_total_l1592_159298


namespace gcf_of_lcm_equals_five_l1592_159252

theorem gcf_of_lcm_equals_five : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_equals_five_l1592_159252


namespace power_relation_l1592_159217

theorem power_relation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := by
  sorry

end power_relation_l1592_159217


namespace first_hour_rate_l1592_159205

def shift_duration : ℕ := 4 -- hours
def masks_per_shift : ℕ := 45
def later_rate : ℕ := 6 -- minutes per mask after the first hour

-- x is the time (in minutes) to make one mask in the first hour
theorem first_hour_rate (x : ℕ) : x = 4 ↔ 
  (60 / x : ℚ) + (shift_duration - 1) * (60 / later_rate : ℚ) = masks_per_shift :=
by sorry

end first_hour_rate_l1592_159205


namespace median_squares_sum_l1592_159231

/-- Given a triangle with sides a, b, c and corresponding medians s_a, s_b, s_c,
    the sum of the squares of the medians is equal to 3/4 times the sum of the squares of the sides. -/
theorem median_squares_sum (a b c s_a s_b s_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_a : s_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_s_b : s_b^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_s_c : s_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  s_a^2 + s_b^2 + s_c^2 = 3/4 * (a^2 + b^2 + c^2) := by
  sorry


end median_squares_sum_l1592_159231


namespace ace_probabilities_l1592_159213

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The probability of drawing an Ace twice without replacement from a standard deck -/
def prob_two_aces : ℚ := 1 / 221

/-- The conditional probability of drawing an Ace on the second draw, given that the first card drawn is an Ace -/
def prob_second_ace_given_first : ℚ := 1 / 17

/-- Theorem stating the probabilities for drawing Aces from a standard deck -/
theorem ace_probabilities :
  (prob_two_aces = (num_aces : ℚ) / deck_size * (num_aces - 1) / (deck_size - 1)) ∧
  (prob_second_ace_given_first = (num_aces - 1 : ℚ) / (deck_size - 1)) :=
sorry

end ace_probabilities_l1592_159213


namespace sum_256_130_in_base6_l1592_159254

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem sum_256_130_in_base6 :
  toBase6 (256 + 130) = [1, 0, 4, 2] :=
sorry

end sum_256_130_in_base6_l1592_159254


namespace simplify_and_add_square_roots_l1592_159206

theorem simplify_and_add_square_roots :
  let x := Real.sqrt 726 / Real.sqrt 484
  let y := Real.sqrt 245 / Real.sqrt 147
  let z := Real.sqrt 1089 / Real.sqrt 441
  x + y + z = (87 + 14 * Real.sqrt 15) / 42 := by
  sorry

end simplify_and_add_square_roots_l1592_159206


namespace sum_of_three_consecutive_integers_l1592_159241

theorem sum_of_three_consecutive_integers (a : ℤ) (h : a = 29) :
  a + (a + 1) + (a + 2) = 90 := by
  sorry

end sum_of_three_consecutive_integers_l1592_159241


namespace harmonic_mean_of_square_sides_l1592_159221

theorem harmonic_mean_of_square_sides (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  3 / (1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) = 360 / 49 := by
  sorry

end harmonic_mean_of_square_sides_l1592_159221


namespace absolute_difference_of_powers_greater_than_half_l1592_159278

theorem absolute_difference_of_powers_greater_than_half :
  |2^3000 - 3^2006| > (1/2 : ℝ) := by sorry

end absolute_difference_of_powers_greater_than_half_l1592_159278


namespace derivative_of_2ln_derivative_of_exp_div_x_l1592_159290

-- Function 1: f(x) = 2ln(x)
theorem derivative_of_2ln (x : ℝ) (h : x > 0) : 
  deriv (fun x => 2 * Real.log x) x = 2 / x := by sorry

-- Function 2: f(x) = e^x / x
theorem derivative_of_exp_div_x (x : ℝ) (h : x ≠ 0) : 
  deriv (fun x => Real.exp x / x) x = (Real.exp x * x - Real.exp x) / x^2 := by sorry

end derivative_of_2ln_derivative_of_exp_div_x_l1592_159290


namespace class_average_score_l1592_159219

theorem class_average_score (total_students : Nat) (score1 score2 : Nat) (other_avg : Nat) : 
  total_students = 40 →
  score1 = 98 →
  score2 = 100 →
  other_avg = 79 →
  (other_avg * (total_students - 2) + score1 + score2) / total_students = 80 :=
by sorry

end class_average_score_l1592_159219


namespace sum_of_integers_l1592_159233

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val ^ 2 + y.val ^ 2 = 130)
  (h2 : x.val * y.val = 36)
  (h3 : x.val - y.val = 4) :
  x.val + y.val = 4 := by
sorry

end sum_of_integers_l1592_159233


namespace binomial_expansions_l1592_159255

theorem binomial_expansions (x a b : ℝ) : 
  ((x + 1) * (x + 2) = x^2 + 3*x + 2) ∧
  ((x + 1) * (x - 2) = x^2 - x - 2) ∧
  ((x - 1) * (x + 2) = x^2 + x - 2) ∧
  ((x - 1) * (x - 2) = x^2 - 3*x + 2) ∧
  ((x + a) * (x + b) = x^2 + (a + b)*x + a*b) :=
by sorry

end binomial_expansions_l1592_159255


namespace greatest_fleet_number_l1592_159269

/-- A ship is a set of connected unit squares on a grid. -/
def Ship := Set (Nat × Nat)

/-- A fleet is a set of vertex-disjoint ships. -/
def Fleet := Set Ship

/-- The grid size. -/
def gridSize : Nat := 10

/-- Checks if two ships are vertex-disjoint. -/
def vertexDisjoint (s1 s2 : Ship) : Prop := sorry

/-- Checks if a fleet is valid (all ships are vertex-disjoint). -/
def validFleet (f : Fleet) : Prop := sorry

/-- Checks if a ship is within the grid bounds. -/
def inGridBounds (s : Ship) : Prop := sorry

/-- Checks if a fleet configuration is valid for a given partition. -/
def validFleetForPartition (n : Nat) (partition : List Nat) (f : Fleet) : Prop := sorry

/-- The main theorem stating that 25 is the greatest number satisfying the fleet condition. -/
theorem greatest_fleet_number : 
  (∀ (partition : List Nat), partition.sum = 25 → 
    ∃ (f : Fleet), validFleet f ∧ validFleetForPartition 25 partition f) ∧
  (∀ (n : Nat), n > 25 → 
    ∃ (partition : List Nat), partition.sum = n ∧
      ¬∃ (f : Fleet), validFleet f ∧ validFleetForPartition n partition f) :=
sorry

end greatest_fleet_number_l1592_159269


namespace james_payment_l1592_159291

theorem james_payment (adoption_fee : ℝ) (friend_percentage : ℝ) (james_payment : ℝ) : 
  adoption_fee = 200 →
  friend_percentage = 0.25 →
  james_payment = adoption_fee - (adoption_fee * friend_percentage) →
  james_payment = 150 := by
sorry

end james_payment_l1592_159291


namespace equation_solution_l1592_159265

/-- Given the equation and values for a, b, c, and d, prove that x equals 26544.74 -/
theorem equation_solution :
  let a : ℝ := 3
  let b : ℝ := 5
  let c : ℝ := 2
  let d : ℝ := 4
  let x : ℝ := ((a^2 * b * (47 / 100 * 1442)) - (c * d * (36 / 100 * 1412))) + 63
  x = 26544.74 := by
  sorry

end equation_solution_l1592_159265


namespace inequality_proof_l1592_159264

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end inequality_proof_l1592_159264


namespace f_increasing_on_interval_l1592_159263

open Real

noncomputable def f (x : ℝ) : ℝ := exp x + 1 / (exp x)

theorem f_increasing_on_interval (x : ℝ) (h : x > 1/exp 1) : 
  deriv f x > 0 := by
  sorry

end f_increasing_on_interval_l1592_159263


namespace sqrt_eight_and_nine_sixteenths_l1592_159296

theorem sqrt_eight_and_nine_sixteenths : 
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by
  sorry

end sqrt_eight_and_nine_sixteenths_l1592_159296


namespace anns_sledding_speed_l1592_159222

/-- Given the conditions of Mary and Ann's sledding trip, prove Ann's speed -/
theorem anns_sledding_speed 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : time_difference = 13)
  : ∃ (ann_speed : ℝ), ann_speed = 40 ∧ 
    ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference :=
by sorry

end anns_sledding_speed_l1592_159222


namespace subset_implies_a_equals_three_l1592_159208

theorem subset_implies_a_equals_three (A B : Set ℝ) (a : ℝ) : 
  A = {2, 3} → B = {1, 2, a} → A ⊆ B → a = 3 := by sorry

end subset_implies_a_equals_three_l1592_159208


namespace square_coins_problem_l1592_159236

theorem square_coins_problem (perimeter_coins : ℕ) (h : perimeter_coins = 240) :
  let side_length := (perimeter_coins + 4) / 4
  side_length * side_length = 3721 := by
  sorry

end square_coins_problem_l1592_159236


namespace vector_addition_l1592_159277

theorem vector_addition (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a + b = (-1, 5) := by
  sorry

end vector_addition_l1592_159277


namespace horner_v3_equals_neg_57_l1592_159232

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

/-- Theorem: V₃ in Horner's method for f(x) at x = -4 is -57 -/
theorem horner_v3_equals_neg_57 :
  (horner (f.reverse.take 4) (-4) : ℝ) = -57 := by
  sorry

end horner_v3_equals_neg_57_l1592_159232


namespace ships_within_visibility_range_l1592_159237

/-- Two ships traveling on perpendicular courses -/
structure Ship where
  x : ℝ
  y : ℝ
  v : ℝ

/-- The problem setup -/
def ship_problem (v : ℝ) : Prop :=
  let ship1 : Ship := ⟨20, 0, v⟩
  let ship2 : Ship := ⟨0, 15, v⟩
  ∃ t : ℝ, t ≥ 0 ∧ 
    ((20 - v * t)^2 + (15 - v * t)^2) ≤ 4^2

/-- The main theorem -/
theorem ships_within_visibility_range (v : ℝ) (h : v > 0) : 
  ship_problem v :=
sorry

end ships_within_visibility_range_l1592_159237


namespace quadratic_form_sum_l1592_159292

/-- For the quadratic x^2 - 24x + 60, when written as (x+b)^2 + c, b+c equals -96 -/
theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 60 = (x+b)^2 + c) → b + c = -96 := by
  sorry

end quadratic_form_sum_l1592_159292


namespace shortened_tripod_height_l1592_159220

/-- Represents a tripod with potentially unequal legs -/
structure Tripod where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ

/-- Calculates the height of a tripod given its leg lengths -/
def tripodHeight (t : Tripod) : ℝ := sorry

/-- The original tripod with equal legs -/
def originalTripod : Tripod := ⟨5, 5, 5⟩

/-- The height of the original tripod -/
def originalHeight : ℝ := 4

/-- The tripod with one shortened leg -/
def shortenedTripod : Tripod := ⟨4, 5, 5⟩

/-- The theorem to be proved -/
theorem shortened_tripod_height :
  tripodHeight shortenedTripod = 144 / Real.sqrt (5 * 317) := by sorry

end shortened_tripod_height_l1592_159220


namespace counterexample_exists_l1592_159200

theorem counterexample_exists : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬(Nat.Prime n) ∧ 
  ¬(Nat.Prime (n + 2)) ∧ 
  n = 14 :=
sorry

end counterexample_exists_l1592_159200


namespace product_equality_l1592_159297

theorem product_equality : 100 * 19.98 * 2.998 * 1000 = 5994004 := by
  sorry

end product_equality_l1592_159297


namespace farmer_seeds_l1592_159247

theorem farmer_seeds (final_buckets sowed_buckets : ℝ) 
  (h1 : final_buckets = 6)
  (h2 : sowed_buckets = 2.75) : 
  final_buckets + sowed_buckets = 8.75 := by
  sorry

end farmer_seeds_l1592_159247


namespace riding_to_total_ratio_l1592_159273

/-- Given a group of horses and men with specific conditions, 
    prove the ratio of riding owners to total owners --/
theorem riding_to_total_ratio 
  (total_horses : ℕ) 
  (total_men : ℕ) 
  (legs_on_ground : ℕ) 
  (h1 : total_horses = 16)
  (h2 : total_men = total_horses)
  (h3 : legs_on_ground = 80) : 
  (total_horses - (legs_on_ground - 4 * total_horses) / 2) / total_horses = 1 / 2 := by
  sorry

end riding_to_total_ratio_l1592_159273


namespace statue_final_weight_l1592_159207

/-- Calculates the final weight of a statue after three weeks of carving. -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_first_week := initial_weight * (1 - 0.3)
  let weight_after_second_week := weight_after_first_week * (1 - 0.3)
  let weight_after_third_week := weight_after_second_week * (1 - 0.15)
  weight_after_third_week

/-- Theorem stating that the final weight of the statue is 124.95 kg. -/
theorem statue_final_weight :
  final_statue_weight 300 = 124.95 := by
  sorry

end statue_final_weight_l1592_159207


namespace cosine_equality_l1592_159202

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (865 * π / 180) → n = 35 ∨ n = 145 := by
  sorry

end cosine_equality_l1592_159202


namespace no_matrix_satisfies_condition_l1592_159215

theorem no_matrix_satisfies_condition : 
  ∀ (N : Matrix (Fin 2) (Fin 2) ℝ),
    (∀ (w x y z : ℝ), 
      N * !![w, x; y, z] = !![x, w; z, y]) → 
    N = 0 := by sorry

end no_matrix_satisfies_condition_l1592_159215


namespace system_solution_sum_of_squares_l1592_159238

theorem system_solution_sum_of_squares (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) :
  x^2 + y^2 = 80 := by
sorry

end system_solution_sum_of_squares_l1592_159238


namespace jesse_gift_amount_l1592_159246

/-- Prove that Jesse received $50 as a gift -/
theorem jesse_gift_amount (novel_cost lunch_cost remaining_amount : ℕ) : 
  novel_cost = 7 →
  lunch_cost = 2 * novel_cost →
  remaining_amount = 29 →
  novel_cost + lunch_cost + remaining_amount = 50 := by
  sorry

end jesse_gift_amount_l1592_159246


namespace inequality_solutions_l1592_159242

theorem inequality_solutions :
  (∀ x : ℝ, |x - 6| ≤ 2 ↔ 4 ≤ x ∧ x ≤ 8) ∧
  (∀ x : ℝ, (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2) ∧
  (∀ x : ℝ, |x| > x ↔ x < 0) ∧
  (∀ x : ℝ, |x^2 - 4*x - 5| > x^2 - 4*x - 5 ↔ -1 < x ∧ x < 5) :=
by sorry

end inequality_solutions_l1592_159242


namespace time_to_destination_l1592_159288

-- Define the walking speeds and distances
def your_speed : ℝ := 2
def harris_speed : ℝ := 1
def harris_time : ℝ := 2
def distance_ratio : ℝ := 3

-- Theorem statement
theorem time_to_destination : 
  your_speed * harris_speed = 2 → 
  (your_speed * (distance_ratio * harris_time)) / your_speed = 3 :=
by
  sorry

end time_to_destination_l1592_159288


namespace power_mod_seven_l1592_159227

theorem power_mod_seven : 5^1986 % 7 = 1 := by
  sorry

end power_mod_seven_l1592_159227


namespace hockey_helmets_l1592_159225

theorem hockey_helmets (red blue : ℕ) : 
  red = blue + 6 → 
  red * 3 = blue * 5 → 
  red + blue = 24 := by
sorry

end hockey_helmets_l1592_159225


namespace sams_candy_count_l1592_159203

/-- Represents the candy count for each friend -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total candy count for all friends -/
def totalCandy : ℕ := 50

/-- The given candy counts for Bob, Mary, John, and Sue -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 5
  sue := 20
  sam := 0  -- We don't know Sam's count yet, so we initialize it to 0

/-- Theorem stating that Sam's candy count is equal to the total minus the sum of others -/
theorem sams_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.sam = totalCandy - (c.bob + c.mary + c.john + c.sue) :=
by
  sorry

#check sams_candy_count

end sams_candy_count_l1592_159203


namespace runner_meetings_l1592_159284

/-- The number of meetings between two runners on a circular track -/
def number_of_meetings (speed1 speed2 : ℝ) (laps : ℕ) : ℕ :=
  sorry

/-- The theorem stating the number of meetings for the given problem -/
theorem runner_meetings :
  let speed1 := (4 : ℝ)
  let speed2 := (10 : ℝ)
  let laps := 28
  number_of_meetings speed1 speed2 laps = 77 :=
sorry

end runner_meetings_l1592_159284


namespace cot_thirty_degrees_l1592_159283

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end cot_thirty_degrees_l1592_159283


namespace upgraded_fraction_is_one_ninth_l1592_159281

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration. -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) : 
    upgraded_fraction s = 1 / 9 := by
  sorry

end upgraded_fraction_is_one_ninth_l1592_159281


namespace expression_value_l1592_159230

theorem expression_value (a b : ℝ) (h : a + b = 1) : a^2 - b^2 + 2*b + 9 = 10 := by
  sorry

end expression_value_l1592_159230


namespace one_not_identity_for_star_l1592_159209

/-- The set of all non-zero real numbers -/
def S : Set ℝ := {x : ℝ | x ≠ 0}

/-- The binary operation * on S -/
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

/-- Theorem: 1 is not an identity element for * in S -/
theorem one_not_identity_for_star :
  ¬(∀ a : ℝ, a ∈ S → (star 1 a = a ∧ star a 1 = a)) :=
sorry

end one_not_identity_for_star_l1592_159209


namespace journey_mpg_approx_30_3_l1592_159289

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) (gas_fills : List ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The average miles per gallon for the given journey is approximately 30.3 -/
theorem journey_mpg_approx_30_3 :
  let initial_odometer := 34650
  let final_odometer := 35800
  let gas_fills := [8, 10, 15, 5]
  let mpg := average_mpg initial_odometer final_odometer gas_fills
  ∃ ε > 0, abs (mpg - 30.3) < ε ∧ ε < 0.1 := by
  sorry

#eval average_mpg 34650 35800 [8, 10, 15, 5]

end journey_mpg_approx_30_3_l1592_159289


namespace escalator_least_time_l1592_159218

/-- The least time needed for people to go up an escalator with variable speed -/
theorem escalator_least_time (n l α : ℝ) (hn : n > 0) (hl : l > 0) (hα : α > 0) :
  let speed (m : ℝ) := m ^ (-α)
  let time_one_by_one := n * l
  let time_all_together := l * n ^ α
  min time_one_by_one time_all_together = l * n ^ min α 1 := by
  sorry

end escalator_least_time_l1592_159218


namespace diana_paint_remaining_l1592_159211

/-- The amount of paint required for each statue in gallons -/
def paint_per_statue : ℚ := 1 / 16

/-- The number of statues Diana can paint -/
def number_of_statues : ℕ := 14

/-- The total amount of paint Diana has remaining in gallons -/
def total_paint : ℚ := paint_per_statue * number_of_statues

/-- Theorem stating that the total paint Diana has remaining is 7/8 gallon -/
theorem diana_paint_remaining : total_paint = 7 / 8 := by sorry

end diana_paint_remaining_l1592_159211


namespace remaining_work_completion_time_l1592_159271

theorem remaining_work_completion_time 
  (a_completion_time b_completion_time b_work_days : ℝ) 
  (ha : a_completion_time = 6)
  (hb : b_completion_time = 15)
  (hbw : b_work_days = 10) : 
  (1 - b_work_days / b_completion_time) / (1 / a_completion_time) = 2 := by
  sorry

end remaining_work_completion_time_l1592_159271


namespace all_positive_integers_expressible_l1592_159276

theorem all_positive_integers_expressible (n : ℕ+) :
  ∃ (a b c : ℤ), (n : ℤ) = a^2 + b^2 + c^2 + c := by sorry

end all_positive_integers_expressible_l1592_159276


namespace scores_statistics_l1592_159280

def scores : List ℕ := [85, 95, 85, 80, 80, 85]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

/-- The range of a list of natural numbers -/
def range (l : List ℕ) : ℕ := sorry

theorem scores_statistics :
  mode scores = 85 ∧
  mean scores = 85 ∧
  median scores = 85 ∧
  range scores = 15 := by sorry

end scores_statistics_l1592_159280


namespace lollipops_kept_by_winnie_l1592_159257

def cherry_lollipops : ℕ := 53
def wintergreen_lollipops : ℕ := 130
def grape_lollipops : ℕ := 12
def shrimp_cocktail_lollipops : ℕ := 240
def number_of_friends : ℕ := 13

def total_lollipops : ℕ := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops

theorem lollipops_kept_by_winnie :
  total_lollipops % number_of_friends = 6 :=
by sorry

end lollipops_kept_by_winnie_l1592_159257


namespace sine_inequality_solution_l1592_159299

theorem sine_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) ∧ 
  (|a| + |b| ≥ 2 / Real.sqrt 3) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry

end sine_inequality_solution_l1592_159299


namespace characterize_satisfying_functions_l1592_159239

/-- A function satisfying the given inequality for all real numbers x < y < z -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤ f ((x + z) / 2) - (f x + f z) / 2

/-- The characterization of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔
    ∃ a b c : ℝ, a ≤ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c :=
by sorry

end characterize_satisfying_functions_l1592_159239


namespace function_inequality_implies_upper_bound_l1592_159275

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end function_inequality_implies_upper_bound_l1592_159275


namespace towels_remaining_l1592_159251

/-- The number of green towels Maria bought -/
def green_bought : ℕ := 35

/-- The number of white towels Maria bought -/
def white_bought : ℕ := 21

/-- The number of blue towels Maria bought -/
def blue_bought : ℕ := 15

/-- The number of green towels Maria gave to her mother -/
def green_given : ℕ := 22

/-- The number of white towels Maria gave to her mother -/
def white_given : ℕ := 14

/-- The number of blue towels Maria gave to her mother -/
def blue_given : ℕ := 6

/-- The total number of towels Maria gave to her mother -/
def total_given : ℕ := 42

theorem towels_remaining : 
  (green_bought + white_bought + blue_bought) - total_given = 29 := by
  sorry

end towels_remaining_l1592_159251
