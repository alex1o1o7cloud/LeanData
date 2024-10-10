import Mathlib

namespace sin_graph_shift_symmetry_l23_2381

open Real

theorem sin_graph_shift_symmetry (φ : ℝ) :
  (∀ x, ∃ y, y = sin (2*x + φ)) →
  (abs φ < π) →
  (∀ x, ∃ y, y = sin (2*(x + π/6) + φ)) →
  (∀ x, sin (2*(x + π/6) + φ) = -sin (2*(-x + π/6) + φ)) →
  (φ = -π/3 ∨ φ = 2*π/3) := by
sorry

end sin_graph_shift_symmetry_l23_2381


namespace inequality_proof_l23_2369

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end inequality_proof_l23_2369


namespace five_bikes_in_driveway_l23_2398

/-- Calculates the number of bikes in the driveway given the total number of wheels and other vehicles --/
def number_of_bikes (total_wheels car_count tricycle_count trash_can_count roller_skate_wheels : ℕ) : ℕ :=
  let car_wheels := 4 * car_count
  let tricycle_wheels := 3 * tricycle_count
  let remaining_wheels := total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels)
  let bike_and_trash_can_wheels := remaining_wheels - (2 * trash_can_count)
  bike_and_trash_can_wheels / 2

/-- Theorem stating that there are 5 bikes in the driveway --/
theorem five_bikes_in_driveway :
  number_of_bikes 25 2 1 1 4 = 5 := by
  sorry

end five_bikes_in_driveway_l23_2398


namespace prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l23_2395

/-- The probability that three students do not dine in the same cafeteria when randomly choosing between two cafeterias -/
theorem prob_not_same_cafeteria : ℚ :=
  let num_cafeterias : ℕ := 2
  let num_students : ℕ := 3
  let total_choices : ℕ := num_cafeterias ^ num_students
  let same_cafeteria_choices : ℕ := num_cafeterias
  let diff_cafeteria_choices : ℕ := total_choices - same_cafeteria_choices
  (diff_cafeteria_choices : ℚ) / total_choices

theorem prob_not_same_cafeteria_is_three_fourths :
  prob_not_same_cafeteria = 3 / 4 := by sorry

end prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l23_2395


namespace watermelon_customers_l23_2382

theorem watermelon_customers (total : ℕ) (one_melon : ℕ) (three_melons : ℕ) :
  total = 46 →
  one_melon = 17 →
  three_melons = 3 →
  ∃ (two_melons : ℕ),
    two_melons * 2 + one_melon * 1 + three_melons * 3 = total ∧
    two_melons = 10 :=
by sorry

end watermelon_customers_l23_2382


namespace correct_product_l23_2341

theorem correct_product (x : ℝ) (h : 21 * x = 27 * x - 48) : 27 * x = 27 * x := by
  sorry

end correct_product_l23_2341


namespace quarter_percentage_approx_l23_2311

def dimes : ℕ := 60
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarter_value_total : ℕ := quarters * quarter_value

theorem quarter_percentage_approx (ε : ℝ) (h : ε > 0) :
  ∃ (p : ℝ), abs (p - 48.4) < ε ∧ p = (quarter_value_total : ℝ) / total_value * 100 :=
sorry

end quarter_percentage_approx_l23_2311


namespace candle_recycling_l23_2342

def original_candle_weight : ℝ := 20
def wax_percentage : ℝ := 0.1
def num_candles : ℕ := 5
def new_candle_weight : ℝ := 5

theorem candle_recycling :
  (↑num_candles * original_candle_weight * wax_percentage) / new_candle_weight = 3 := by
  sorry

end candle_recycling_l23_2342


namespace second_smallest_hot_dog_packs_l23_2315

theorem second_smallest_hot_dog_packs : ∃ (n : ℕ), n > 0 ∧
  (12 * n) % 8 = 6 ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → (12 * m) % 8 ≠ 6) ∧
  (∃ (k : ℕ), k > 0 ∧ k < n ∧ (12 * k) % 8 = 6) ∧
  n = 7 := by
sorry

end second_smallest_hot_dog_packs_l23_2315


namespace sphere_plane_intersection_area_l23_2309

theorem sphere_plane_intersection_area (r : ℝ) (h : r = 1) :
  ∃ (d : ℝ), 0 < d ∧ d < r ∧
  (2 * π * r * (r - d) = π * r^2) ∧
  (2 * π * r * d = 3 * π * r^2) ∧
  π * (r^2 - d^2) = (3 * π) / 4 := by
sorry

end sphere_plane_intersection_area_l23_2309


namespace production_value_decrease_l23_2354

theorem production_value_decrease (a : ℝ) :
  let increase_percent := a
  let decrease_percent := |a / (100 + a)|
  increase_percent > -100 →
  decrease_percent = |1 - 1 / (1 + a / 100)| :=
by sorry

end production_value_decrease_l23_2354


namespace min_reciprocal_sum_l23_2390

theorem min_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 3) :
  (1/x + 1/y) ≥ 1 + (2*Real.sqrt 2)/3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3 ∧ 1/x₀ + 1/y₀ = 1 + (2*Real.sqrt 2)/3 :=
sorry

end min_reciprocal_sum_l23_2390


namespace polynomial_evaluation_l23_2343

theorem polynomial_evaluation (x : ℝ) (h : x = 1 + Real.sqrt 2) : 
  x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by sorry

end polynomial_evaluation_l23_2343


namespace tree_planting_change_l23_2319

/-- Represents the road with tree planting configuration -/
structure RoadConfig where
  length : ℕ
  initial_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of trees for a given spacing -/
def trees_count (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the change in number of holes -/
def hole_change (config : RoadConfig) : ℤ :=
  (trees_count config config.new_spacing : ℤ) - (trees_count config config.initial_spacing : ℤ)

theorem tree_planting_change (config : RoadConfig) 
  (h_length : config.length = 240)
  (h_initial : config.initial_spacing = 8)
  (h_new : config.new_spacing = 6) :
  hole_change config = 10 ∧ max (-(hole_change config)) 0 = 0 :=
sorry

end tree_planting_change_l23_2319


namespace candy_necklace_blocks_l23_2387

/-- The number of friends receiving candy necklaces -/
def num_friends : ℕ := 8

/-- The number of candy pieces per necklace -/
def pieces_per_necklace : ℕ := 10

/-- The number of candy pieces produced by one block -/
def pieces_per_block : ℕ := 30

/-- The minimum number of whole blocks needed to make necklaces for all friends -/
def min_blocks_needed : ℕ := 3

theorem candy_necklace_blocks :
  (num_friends * pieces_per_necklace + pieces_per_block - 1) / pieces_per_block = min_blocks_needed :=
sorry

end candy_necklace_blocks_l23_2387


namespace geometric_sequence_problem_l23_2355

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) (n : ℕ) : Prop :=
  geometric_sequence a ∧
  a 1 * a 2 * a 3 = 4 ∧
  a 4 * a 5 * a 6 = 12 ∧
  a (n - 1) * a n * a (n + 1) = 324

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  problem_conditions a n → n = 14 :=
by
  sorry

end geometric_sequence_problem_l23_2355


namespace rainfall_difference_l23_2323

/-- Rainfall data for Tropical Storm Sally -/
structure RainfallData where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Tropical Storm Sally's rainfall -/
def sallysRainfall : RainfallData where
  day1 := 4
  day2 := 5 * 4
  day3 := 18

/-- Theorem: The difference between the sum of the first two days' rainfall and the third day's rainfall is 6 inches -/
theorem rainfall_difference (data : RainfallData := sallysRainfall) :
  (data.day1 + data.day2) - data.day3 = 6 := by
  sorry

end rainfall_difference_l23_2323


namespace infinitely_many_rationals_between_one_sixth_and_five_sixths_l23_2332

theorem infinitely_many_rationals_between_one_sixth_and_five_sixths :
  ∃ (S : Set ℚ), Set.Infinite S ∧ ∀ q ∈ S, 1/6 < q ∧ q < 5/6 :=
sorry

end infinitely_many_rationals_between_one_sixth_and_five_sixths_l23_2332


namespace newer_train_theorem_l23_2345

/-- Calculates the distance traveled by a newer train given the distance of an older train and the percentage increase in distance. -/
def newer_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem stating that a newer train traveling 30% farther than an older train that goes 180 miles will travel 234 miles. -/
theorem newer_train_theorem :
  newer_train_distance 180 0.3 = 234 := by
  sorry

#eval newer_train_distance 180 0.3

end newer_train_theorem_l23_2345


namespace largest_integer_x_l23_2330

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def fraction (x : ℤ) : ℚ := (x^2 + 3*x + 8) / (x - 2)

theorem largest_integer_x : 
  (∀ x : ℤ, x > 1 → ¬ is_integer (fraction x)) ∧ 
  is_integer (fraction 1) :=
sorry

end largest_integer_x_l23_2330


namespace theo_donut_holes_l23_2347

/-- Represents a worker coating donut holes -/
structure Worker where
  name : String
  radius : ℕ

/-- Calculates the surface area of a spherical donut hole -/
def surfaceArea (r : ℕ) : ℕ := 4 * r * r

/-- Calculates the number of donut holes coated by a worker when all workers finish simultaneously -/
def donutHolesCoated (workers : List Worker) (w : Worker) : ℕ :=
  let surfaces := workers.map (λ worker => surfaceArea worker.radius)
  let lcm := surfaces.foldl Nat.lcm 1
  lcm / (surfaceArea w.radius)

/-- The main theorem stating the number of donut holes Theo will coat -/
theorem theo_donut_holes (workers : List Worker) :
  workers = [
    ⟨"Niraek", 5⟩,
    ⟨"Theo", 7⟩,
    ⟨"Akshaj", 9⟩,
    ⟨"Mira", 11⟩
  ] →
  donutHolesCoated workers (Worker.mk "Theo" 7) = 1036830 := by
  sorry

end theo_donut_holes_l23_2347


namespace det_transformation_l23_2320

/-- Given a 2x2 matrix with determinant 7, prove that a specific transformation of this matrix also has determinant 7. -/
theorem det_transformation (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end det_transformation_l23_2320


namespace scientific_notation_274_million_l23_2335

theorem scientific_notation_274_million :
  274000000 = 2.74 * (10 : ℝ)^8 := by sorry

end scientific_notation_274_million_l23_2335


namespace total_red_balloons_l23_2362

/-- The total number of red balloons given the number of balloons each person has -/
def total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) : ℕ :=
  fred_balloons + sam_balloons + dan_balloons

/-- Theorem stating that the total number of red balloons is 72 -/
theorem total_red_balloons : 
  total_balloons 10 46 16 = 72 := by
  sorry

end total_red_balloons_l23_2362


namespace cookie_boxes_theorem_l23_2303

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 8 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 9 := by
sorry

end cookie_boxes_theorem_l23_2303


namespace vehicle_distance_after_three_minutes_l23_2363

/-- The distance between two vehicles after a given time, given their speeds -/
def distance_between (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem vehicle_distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between truck_speed car_speed time_in_hours = 1 := by
  sorry

end vehicle_distance_after_three_minutes_l23_2363


namespace min_value_expression_min_value_attained_l23_2385

theorem min_value_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (1 / x + 2 * x / (1 - x)) ≥ 1 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_attained (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ (1 / x₀ + 2 * x₀ / (1 - x₀)) = 1 + 2 * Real.sqrt 2 := by
  sorry

end min_value_expression_min_value_attained_l23_2385


namespace rectangle_to_square_width_third_l23_2353

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Theorem: Given a 9x27 rectangle that can be cut into two congruent hexagons
    which can be repositioned to form a square, one third of the rectangle's width is 9 -/
theorem rectangle_to_square_width_third (rect : Rectangle) (sq : Square) :
  rect.width = 27 ∧ 
  rect.height = 9 ∧ 
  sq.side ^ 2 = rect.width * rect.height →
  rect.width / 3 = 9 := by
  sorry

end rectangle_to_square_width_third_l23_2353


namespace donut_distribution_ways_l23_2301

/-- The number of types of donuts available --/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased --/
def total_donuts : ℕ := 8

/-- The number of donuts that must be purchased of the first type --/
def first_type_min : ℕ := 2

/-- The number of donuts that must be purchased of each other type --/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to be distributed after mandatory purchases --/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (num_types - 1) * other_types_min)

theorem donut_distribution_ways : 
  (Nat.choose (remaining_donuts + num_types - 1) (num_types - 1)) = 15 := by
  sorry

end donut_distribution_ways_l23_2301


namespace quadratic_root_relation_l23_2359

theorem quadratic_root_relation (b c : ℝ) : 
  (∃ p q : ℝ, 2 * p^2 - 4 * p - 6 = 0 ∧ 2 * q^2 - 4 * q - 6 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = p - 3 ∨ x = q - 3)) →
  c = 3 := by
sorry

end quadratic_root_relation_l23_2359


namespace doughnut_costs_9_l23_2317

/-- The price of a cake in Kč -/
def cake_price : ℕ := sorry

/-- The price of a doughnut in Kč -/
def doughnut_price : ℕ := sorry

/-- The amount of pocket money Honzík has in Kč -/
def pocket_money : ℕ := sorry

/-- Theorem stating the price of one doughnut is 9 Kč -/
theorem doughnut_costs_9 
  (h1 : pocket_money - 4 * cake_price = 5)
  (h2 : 5 * cake_price - pocket_money = 6)
  (h3 : 2 * cake_price + 3 * doughnut_price = pocket_money) :
  doughnut_price = 9 := by sorry

end doughnut_costs_9_l23_2317


namespace bankers_discount_calculation_l23_2329

/-- Banker's discount calculation -/
theorem bankers_discount_calculation (face_value : ℝ) (interest_rate : ℝ) (true_discount : ℝ)
  (h1 : face_value = 74500)
  (h2 : interest_rate = 0.15)
  (h3 : true_discount = 11175) :
  face_value * interest_rate = true_discount :=
by sorry

end bankers_discount_calculation_l23_2329


namespace milk_water_ratio_problem_l23_2306

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The problem statement -/
theorem milk_water_ratio_problem 
  (initial : CanContents)
  (h_capacity : initial.milk + initial.water + 20 = 60)
  (h_ratio_after : (initial.milk + 20) / initial.water = 3) :
  initial.milk / initial.water = 5 / 3 := by
  sorry

end milk_water_ratio_problem_l23_2306


namespace percentage_difference_l23_2360

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem percentage_difference (h1 : z = 400) (h2 : y = 1.2 * z) (h3 : x + y + z = 1480) :
  (x - y) / y = 0.25 := by
  sorry

end percentage_difference_l23_2360


namespace hyperbola_eccentricity_l23_2318

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and an asymptote 2x - √3y = 0,
    prove that its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ x y : ℝ, 2 * x - Real.sqrt 3 * y = 0 → 
    (x^2 / a^2 - y^2 / b^2 = 1 ↔ x = 0 ∧ y = 0)) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end hyperbola_eccentricity_l23_2318


namespace negation_of_forall_positive_l23_2394

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end negation_of_forall_positive_l23_2394


namespace increasing_function_equivalence_l23_2312

/-- A function f is increasing on ℝ -/
def IncreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_equivalence (f : ℝ → ℝ) (h : IncreasingOnReals f) :
  ∀ a b : ℝ, (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end increasing_function_equivalence_l23_2312


namespace line_tangent_to_parabola_l23_2375

-- Define the line equation
def line (x y : ℝ) : Prop := 4 * x + 7 * y + 49 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define what it means for a line to be tangent to a parabola
def is_tangent (l : ℝ → ℝ → Prop) (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), l x₀ y₀ ∧ p x₀ y₀ ∧
    ∀ (x y : ℝ), l x y ∧ p x y → (x, y) = (x₀, y₀)

-- Theorem statement
theorem line_tangent_to_parabola :
  is_tangent line parabola :=
sorry

end line_tangent_to_parabola_l23_2375


namespace man_son_age_ratio_l23_2389

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the initial conditions. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

#check man_son_age_ratio

end man_son_age_ratio_l23_2389


namespace franks_money_duration_l23_2397

/-- The duration (in weeks) that Frank's money will last given his earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Frank's money will last for 9 weeks given his earnings and spending. -/
theorem franks_money_duration :
  money_duration 5 58 7 = 9 := by
  sorry

end franks_money_duration_l23_2397


namespace max_area_is_one_l23_2316

/-- A right triangle with legs 3 and 4, and hypotenuse 5 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2
  leg1_is_3 : leg1 = 3
  leg2_is_4 : leg2 = 4
  hypotenuse_is_5 : hypotenuse = 5

/-- A rectangle inscribed in the right triangle with one side along the hypotenuse -/
structure InscribedRectangle (t : RightTriangle) where
  base : ℝ  -- Length of the rectangle's side along the hypotenuse
  height : ℝ -- Height of the rectangle
  is_inscribed : height ≤ t.leg2 * (1 - base / t.hypotenuse)
  on_hypotenuse : base ≤ t.hypotenuse

/-- The area of an inscribed rectangle -/
def area (t : RightTriangle) (r : InscribedRectangle t) : ℝ :=
  r.base * r.height

/-- The maximum area of an inscribed rectangle is 1 -/
theorem max_area_is_one (t : RightTriangle) : 
  ∃ (r : InscribedRectangle t), ∀ (r' : InscribedRectangle t), area t r ≥ area t r' ∧ area t r = 1 :=
sorry

end max_area_is_one_l23_2316


namespace boys_to_girls_ratio_l23_2314

/-- Given a class with 100 students where there are 20 more boys than girls,
    prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (total : ℕ) (difference : ℕ) : 
  total = 100 → difference = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys = girls + difference ∧
    boys / girls = 3 / 2 := by
  sorry

end boys_to_girls_ratio_l23_2314


namespace positive_number_square_sum_l23_2300

theorem positive_number_square_sum (n : ℝ) : n > 0 → n^2 + n = 245 → n = 14 := by
  sorry

end positive_number_square_sum_l23_2300


namespace book_has_two_chapters_l23_2372

/-- A book with chapters and pages -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- The number of chapters in a book -/
def num_chapters (b : Book) : ℕ :=
  if b.first_chapter_pages + b.second_chapter_pages = b.total_pages then 2 else 0

theorem book_has_two_chapters (b : Book) 
  (h1 : b.total_pages = 81) 
  (h2 : b.first_chapter_pages = 13) 
  (h3 : b.second_chapter_pages = 68) : 
  num_chapters b = 2 := by
  sorry

end book_has_two_chapters_l23_2372


namespace cubic_sum_problem_l23_2344

theorem cubic_sum_problem (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end cubic_sum_problem_l23_2344


namespace same_solution_l23_2392

theorem same_solution (x y : ℝ) : 
  (4 * x - 8 * y - 5 = 0) ↔ (8 * x - 16 * y - 10 = 0) := by
  sorry

end same_solution_l23_2392


namespace polynomial_negative_roots_l23_2337

theorem polynomial_negative_roots (q : ℝ) (hq : q > 1/2) :
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 3*x₁^2 + q*x₁ + 9 = 0 ∧
  x₂^4 + q*x₂^3 + 3*x₂^2 + q*x₂ + 9 = 0 :=
by sorry

end polynomial_negative_roots_l23_2337


namespace problem_statement_l23_2348

theorem problem_statement (x : ℝ) (h : x^2 + 8 * (x / (x - 3))^2 = 53) :
  ((x - 3)^3 * (x + 4)) / (2 * x - 5) = 17000 / 21 := by
  sorry

end problem_statement_l23_2348


namespace first_week_gain_l23_2364

/-- Proves that the percentage gain in the first week was 25% --/
theorem first_week_gain (initial_investment : ℝ) (final_value : ℝ) : 
  initial_investment = 400 →
  final_value = 750 →
  ∃ (x : ℝ), 
    (initial_investment + x / 100 * initial_investment) * 1.5 = final_value ∧
    x = 25 := by
  sorry

#check first_week_gain

end first_week_gain_l23_2364


namespace smallest_n_for_inequality_l23_2378

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end smallest_n_for_inequality_l23_2378


namespace min_period_sin_2x_plus_pi_third_l23_2336

/-- The minimum positive period of y = sin(2x + π/3) is π -/
theorem min_period_sin_2x_plus_pi_third (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) ∧
    T = π :=
by sorry

end min_period_sin_2x_plus_pi_third_l23_2336


namespace divisibility_of_sum_of_cubes_l23_2365

theorem divisibility_of_sum_of_cubes (n m : ℕ+) 
  (h : n^3 + (n+1)^3 + (n+2)^3 = m^3) : 
  4 ∣ (n+1) := by
  sorry

end divisibility_of_sum_of_cubes_l23_2365


namespace prime_expressions_solution_l23_2380

def f (n : ℤ) : ℤ := |n^3 - 4*n^2 + 3*n - 35|
def g (n : ℤ) : ℤ := |n^2 + 4*n + 8|

theorem prime_expressions_solution :
  {n : ℤ | Nat.Prime (f n).natAbs ∧ Nat.Prime (g n).natAbs} = {-3, -1, 5} := by
sorry

end prime_expressions_solution_l23_2380


namespace largest_reciprocal_l23_2324

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = -1/4 → b = 2/7 → c = -2 → d = 3 → e = -3/2 → 
  (1/b > 1/a ∧ 1/b > 1/c ∧ 1/b > 1/d ∧ 1/b > 1/e) := by
  sorry

end largest_reciprocal_l23_2324


namespace expand_and_simplify_l23_2322

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_l23_2322


namespace quadratic_transformation_l23_2377

theorem quadratic_transformation (y m n : ℝ) : 
  (2 * y^2 - 2 = 4 * y) → 
  ((y - m)^2 = n) → 
  (m - n)^2023 = -1 := by
sorry

end quadratic_transformation_l23_2377


namespace range_of_a_l23_2325

theorem range_of_a (p q : Prop) (h1 : ∀ x : ℝ, x > 0 → x + 1/x > a → a < 2)
  (h2 : (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) → (a ≤ -1 ∨ a ≥ 1))
  (h3 : q) (h4 : ¬p) : a ≥ 2 := by
  sorry

end range_of_a_l23_2325


namespace milk_set_cost_l23_2328

/-- The cost of a set of 2 packs of 500 mL milk -/
def set_cost : ℝ := 2.50

/-- The cost of an individual pack of 500 mL milk -/
def individual_cost : ℝ := 1.30

/-- The total savings when buying ten sets of 2 packs -/
def total_savings : ℝ := 1

theorem milk_set_cost :
  set_cost = 2 * individual_cost - total_savings / 10 :=
by sorry

end milk_set_cost_l23_2328


namespace solve_for_a_l23_2333

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → 
  a = 1 := by
  sorry

end solve_for_a_l23_2333


namespace distance_after_five_hours_l23_2361

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: The distance between two people walking in opposite directions for 5 hours
    with speeds 5 km/hr and 10 km/hr is 75 km -/
theorem distance_after_five_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end distance_after_five_hours_l23_2361


namespace store_posters_l23_2350

theorem store_posters (P : ℕ) : 
  (2 : ℚ) / 5 * P + (1 : ℚ) / 2 * P + 5 = P → P = 50 := by
  sorry

end store_posters_l23_2350


namespace jumps_before_cleaning_l23_2321

-- Define the pool characteristics
def pool_capacity : ℝ := 1200  -- in liters
def splash_out_volume : ℝ := 0.2  -- in liters (200 ml = 0.2 L)
def cleaning_threshold : ℝ := 0.8  -- 80% capacity

-- Define the number of jumps
def number_of_jumps : ℕ := 1200

-- Theorem statement
theorem jumps_before_cleaning :
  ⌊(pool_capacity - pool_capacity * cleaning_threshold) / splash_out_volume⌋ = number_of_jumps := by
  sorry

end jumps_before_cleaning_l23_2321


namespace inverse_mod_103_l23_2383

theorem inverse_mod_103 (h : (7⁻¹ : ZMod 103) = 55) : (49⁻¹ : ZMod 103) = 38 := by
  sorry

end inverse_mod_103_l23_2383


namespace first_question_percentage_l23_2399

theorem first_question_percentage (second : ℝ) (neither : ℝ) (both : ℝ)
  (h1 : second = 50)
  (h2 : neither = 20)
  (h3 : both = 33)
  : ∃ first : ℝ, first = 63 ∧ first + second - both + neither = 100 :=
by sorry

end first_question_percentage_l23_2399


namespace det_B_squared_minus_3B_l23_2370

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B ^ 2 - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l23_2370


namespace complement_of_union_eq_nonpositive_l23_2346

-- Define the sets U, P, and Q
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

-- State the theorem
theorem complement_of_union_eq_nonpositive :
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by
  sorry

end complement_of_union_eq_nonpositive_l23_2346


namespace special_collection_loans_l23_2357

theorem special_collection_loans (initial_count : ℕ) (return_rate : ℚ) (final_count : ℕ) 
  (h1 : initial_count = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_count = 60) :
  ∃ (loaned_out : ℕ), loaned_out = 50 ∧ 
    initial_count - (1 - return_rate) * loaned_out = final_count :=
by sorry

end special_collection_loans_l23_2357


namespace b_share_is_seven_fifteenths_l23_2393

/-- A partnership with four partners A, B, C, and D -/
structure Partnership where
  total_capital : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ
  total_profit : ℝ
  a_profit : ℝ

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.a_share = (1/3) * p.total_capital ∧
  p.c_share = (1/5) * p.total_capital ∧
  p.d_share = p.total_capital - (p.a_share + p.b_share + p.c_share) ∧
  p.total_profit = 2430 ∧
  p.a_profit = 810

/-- Theorem stating B's share of the capital -/
theorem b_share_is_seven_fifteenths (p : Partnership) 
  (h : partnership_conditions p) : 
  p.b_share = (7/15) * p.total_capital := by
  sorry

end b_share_is_seven_fifteenths_l23_2393


namespace sin_2000_in_terms_of_tan_160_l23_2368

theorem sin_2000_in_terms_of_tan_160 (a : ℝ) (h : Real.tan (160 * π / 180) = a) :
  Real.sin (2000 * π / 180) = -a / Real.sqrt (1 + a^2) := by
  sorry

end sin_2000_in_terms_of_tan_160_l23_2368


namespace number_composition_l23_2386

def place_value (digit : ℕ) (place : ℕ) : ℕ :=
  digit * (10 ^ place)

theorem number_composition :
  let tens_of_millions := place_value 4 7
  let hundreds_of_thousands := place_value 6 5
  let hundreds := place_value 5 2
  tens_of_millions + hundreds_of_thousands + hundreds = 46000500 := by
  sorry

end number_composition_l23_2386


namespace number_of_white_balls_l23_2327

/-- Given a bag with red and white balls, prove the number of white balls when probability of drawing red is known -/
theorem number_of_white_balls (n : ℕ) : 
  (8 : ℚ) / (8 + n) = (2 : ℚ) / 5 → n = 12 := by
  sorry

end number_of_white_balls_l23_2327


namespace school_gender_difference_l23_2304

theorem school_gender_difference (initial_girls boys additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - boys = 687 := by
  sorry

end school_gender_difference_l23_2304


namespace complex_fraction_evaluation_l23_2391

theorem complex_fraction_evaluation :
  (3/2 : ℚ) * (8/3 * (15/8 - 5/6)) / ((7/8 + 11/6) / (13/4)) = 5 := by
  sorry

end complex_fraction_evaluation_l23_2391


namespace walking_problem_solution_l23_2352

/-- Two people walking in opposite directions --/
structure WalkingProblem where
  time : ℝ
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of our specific problem --/
def problem : WalkingProblem where
  time := 5
  distance := 75
  speed1 := 10
  speed2 := 5 -- This is what we want to prove

theorem walking_problem_solution (p : WalkingProblem) 
  (h1 : p.time * (p.speed1 + p.speed2) = p.distance)
  (h2 : p.time = 5)
  (h3 : p.distance = 75)
  (h4 : p.speed1 = 10) :
  p.speed2 = 5 := by
  sorry

#check walking_problem_solution problem

end walking_problem_solution_l23_2352


namespace plumber_salary_percentage_l23_2349

-- Define the daily salaries and total labor cost
def construction_worker_salary : ℝ := 100
def electrician_salary : ℝ := 2 * construction_worker_salary
def total_labor_cost : ℝ := 650

-- Define the number of workers
def num_construction_workers : ℕ := 2
def num_electricians : ℕ := 1
def num_plumbers : ℕ := 1

-- Calculate the plumber's salary
def plumber_salary : ℝ :=
  total_labor_cost - (num_construction_workers * construction_worker_salary + num_electricians * electrician_salary)

-- Define the theorem
theorem plumber_salary_percentage :
  plumber_salary / construction_worker_salary * 100 = 250 := by
  sorry


end plumber_salary_percentage_l23_2349


namespace simplify_expression_simplify_and_evaluate_l23_2338

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  4 * a^2 + 3 * b^2 + 2 * a * b - 3 * a^2 - 3 * b * a - a^2 = a^2 - a * b + 3 * b^2 := by
  sorry

-- Problem 2
theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  3 * x - 4 * x^2 + 7 - 3 * x + 2 * x^2 + 1 = -10 := by
  sorry

end simplify_expression_simplify_and_evaluate_l23_2338


namespace zoo_viewing_time_is_75_minutes_l23_2384

/-- Calculates the total viewing time for a zoo visit -/
def total_zoo_viewing_time (original_times new_times : List ℕ) (break_time : ℕ) : ℕ :=
  let total_viewing_time := original_times.sum + new_times.sum
  let total_break_time := break_time * (original_times.length + new_times.length - 1)
  total_viewing_time + total_break_time

/-- Theorem: The total time required to see all 9 animal types is 75 minutes -/
theorem zoo_viewing_time_is_75_minutes :
  total_zoo_viewing_time [4, 6, 7, 5, 9] [3, 7, 8, 10] 2 = 75 := by
  sorry

end zoo_viewing_time_is_75_minutes_l23_2384


namespace f_properties_l23_2388

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)

theorem f_properties :
  (∀ x : ℝ, f x > 0 ↔ x > 0) ∧
  (∀ s t : ℝ, s > 0 → t > 0 → f (s + t) > f s + f t) := by
  sorry

end f_properties_l23_2388


namespace yellow_curlers_count_l23_2366

/-- Given the total number of curlers and the proportions of different types,
    prove that the number of extra-large yellow curlers is 18. -/
theorem yellow_curlers_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) : 
  total = 120 →
  pink = total / 5 →
  blue = 2 * pink →
  green = total / 4 →
  yellow = total - pink - blue - green →
  yellow = 18 := by
sorry

end yellow_curlers_count_l23_2366


namespace pool_filling_time_l23_2379

/-- Calculates the time in hours required to fill a pool given its capacity and the rate of water flow. -/
theorem pool_filling_time 
  (pool_capacity : ℚ)  -- Pool capacity in gallons
  (num_hoses : ℕ)      -- Number of hoses
  (flow_rate : ℚ)      -- Flow rate per hose in gallons per minute
  (h : pool_capacity = 36000 ∧ num_hoses = 6 ∧ flow_rate = 3) :
  (pool_capacity / (↑num_hoses * flow_rate * 60)) = 100 / 3 := by
sorry

end pool_filling_time_l23_2379


namespace x_values_when_two_in_set_l23_2308

theorem x_values_when_two_in_set (x : ℝ) : 2 ∈ ({1, x^2 + x} : Set ℝ) → x = 1 ∨ x = -2 := by
  sorry

end x_values_when_two_in_set_l23_2308


namespace c_work_time_l23_2371

-- Define the work rates for each worker
def work_rate_a : ℚ := 1 / 36
def work_rate_b : ℚ := 1 / 18

-- Define the combined work rate
def combined_work_rate : ℚ := 1 / 4

-- Define the relationship between c and d's work rates
def d_work_rate (c : ℚ) : ℚ := c / 2

-- Theorem statement
theorem c_work_time :
  ∃ (c : ℚ), 
    work_rate_a + work_rate_b + c + d_work_rate c = combined_work_rate ∧
    c = 1 / 9 :=
by sorry

end c_work_time_l23_2371


namespace unique_prime_solution_l23_2307

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p → Prime q → Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end unique_prime_solution_l23_2307


namespace blue_beads_count_l23_2396

theorem blue_beads_count (total_beads blue_beads yellow_beads : ℕ) : 
  yellow_beads = 16 →
  total_beads = blue_beads + yellow_beads →
  total_beads % 3 = 0 →
  (total_beads / 3 - 10) * 2 = 6 →
  blue_beads = 23 := by
sorry

end blue_beads_count_l23_2396


namespace inequality_solution_set_l23_2340

/-- The solution set of the inequality x^2 - ax + a - 1 ≤ 0 for real a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 2 then Set.Icc (a - 1) 1
  else if a = 2 then {1}
  else Set.Icc 1 (a - 1)

/-- Theorem stating the solution set of the inequality x^2 - ax + a - 1 ≤ 0 -/
theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ x^2 - a*x + a - 1 ≤ 0 := by
  sorry

end inequality_solution_set_l23_2340


namespace smallest_number_divisibility_l23_2339

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(((m + 1) % 12 = 0) ∧ ((m + 1) % 18 = 0) ∧ ((m + 1) % 24 = 0) ∧ ((m + 1) % 32 = 0) ∧ ((m + 1) % 40 = 0))) ∧
  ((n + 1) % 12 = 0) ∧ ((n + 1) % 18 = 0) ∧ ((n + 1) % 24 = 0) ∧ ((n + 1) % 32 = 0) ∧ ((n + 1) % 40 = 0) →
  n = 2879 :=
by sorry

end smallest_number_divisibility_l23_2339


namespace total_cost_theorem_l23_2313

/-- Calculates the total cost of items with tax --/
def total_cost_with_tax (prices : List ℝ) (tax_rate : ℝ) : ℝ :=
  let subtotal := prices.sum
  let tax_amount := subtotal * tax_rate
  subtotal + tax_amount

/-- Theorem: The total cost of three items with given prices and 5% tax is $15.75 --/
theorem total_cost_theorem :
  let prices := [4.20, 7.60, 3.20]
  let tax_rate := 0.05
  total_cost_with_tax prices tax_rate = 15.75 := by
  sorry

end total_cost_theorem_l23_2313


namespace lassie_bones_problem_l23_2331

/-- The number of bones Lassie started with before Saturday -/
def initial_bones : ℕ := 50

/-- The number of bones Lassie has after eating on Saturday -/
def bones_after_saturday : ℕ := initial_bones / 2

/-- The number of bones Lassie receives on Sunday -/
def bones_received_sunday : ℕ := 10

/-- The total number of bones Lassie has after Sunday -/
def total_bones_after_sunday : ℕ := 35

theorem lassie_bones_problem :
  bones_after_saturday + bones_received_sunday = total_bones_after_sunday :=
by sorry

end lassie_bones_problem_l23_2331


namespace bread_distribution_l23_2356

theorem bread_distribution (a d : ℚ) : 
  d > 0 ∧ 
  (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100 ∧ 
  (a + (a + d) + (a + 2*d)) = (1/7) * ((a - 2*d) + (a - d)) →
  a - 2*d = 5/3 := by
sorry

end bread_distribution_l23_2356


namespace lines_perpendicular_to_plane_are_parallel_l23_2326

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (α : Plane) (a b : Line) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end lines_perpendicular_to_plane_are_parallel_l23_2326


namespace f_is_quadratic_l23_2374

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we're checking -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l23_2374


namespace triangle_perimeter_l23_2305

-- Define the triangle DEF
structure Triangle (D E F : ℝ × ℝ) : Prop where
  right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0
  de_length : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 15^2

-- Define the squares DEFG and EFHI
structure OuterSquares (D E F G H I : ℝ × ℝ) : Prop where
  square_defg : (G.1 - D.1) = (E.1 - D.1) ∧ (G.2 - D.2) = (E.2 - D.2)
  square_efhi : (I.1 - E.1) = (F.1 - E.1) ∧ (I.2 - E.2) = (F.2 - E.2)

-- Define the circle passing through G, H, I, F
structure CircleGHIF (G H I F : ℝ × ℝ) : Prop where
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (G.1 - center.1)^2 + (G.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (I.1 - center.1)^2 + (I.2 - center.2)^2 = radius^2 ∧
    (F.1 - center.1)^2 + (F.2 - center.2)^2 = radius^2

-- Define the point J on DF
def PointJ (D F J : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, J = (D.1 + t * (F.1 - D.1), D.2 + t * (F.2 - D.2)) ∧ t ≠ 1

theorem triangle_perimeter 
  (D E F G H I J : ℝ × ℝ)
  (triangle : Triangle D E F)
  (squares : OuterSquares D E F G H I)
  (circle : CircleGHIF G H I F)
  (j_on_df : PointJ D F J)
  (jf_length : (J.1 - F.1)^2 + (J.2 - F.2)^2 = 3^2) :
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let fd := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  de + ef + fd = 15 + 15 * Real.sqrt 2 := by
  sorry

end triangle_perimeter_l23_2305


namespace additional_deductible_calculation_l23_2373

/-- Calculates the additional deductible amount for an average family --/
def additional_deductible_amount (
  current_deductible : ℝ)
  (plan_a_increase : ℝ)
  (plan_b_increase : ℝ)
  (plan_c_increase : ℝ)
  (plan_a_percentage : ℝ)
  (plan_b_percentage : ℝ)
  (plan_c_percentage : ℝ)
  (inflation_rate : ℝ) : ℝ :=
  let plan_a_additional := current_deductible * plan_a_increase
  let plan_b_additional := current_deductible * plan_b_increase
  let plan_c_additional := current_deductible * plan_c_increase
  let weighted_additional := plan_a_additional * plan_a_percentage +
                             plan_b_additional * plan_b_percentage +
                             plan_c_additional * plan_c_percentage
  weighted_additional * (1 + inflation_rate)

/-- Theorem stating the additional deductible amount for an average family --/
theorem additional_deductible_calculation :
  additional_deductible_amount 3000 (2/3) (1/2) (3/5) 0.4 0.3 0.3 0.03 = 1843.70 := by
  sorry

end additional_deductible_calculation_l23_2373


namespace simplify_expression_l23_2358

theorem simplify_expression : 
  Real.sqrt 2 * 2^(1/2 : ℝ) + 18 / 3 * 3 - 8^(3/2 : ℝ) = 20 - 16 * Real.sqrt 2 := by
  sorry

end simplify_expression_l23_2358


namespace no_natural_number_with_three_prime_divisors_l23_2334

theorem no_natural_number_with_three_prime_divisors :
  ¬ ∃ (m p q r : ℕ),
    (Prime p ∧ Prime q ∧ Prime r) ∧
    (∃ (a b c : ℕ), m = p^a * q^b * r^c) ∧
    (p - 1 ∣ m) ∧
    (q * r - 1 ∣ m) ∧
    ¬(q - 1 ∣ m) ∧
    ¬(r - 1 ∣ m) ∧
    ¬(3 ∣ q + r) := by
  sorry

end no_natural_number_with_three_prime_divisors_l23_2334


namespace rental_cost_is_165_l23_2367

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℝ) (mile_rate : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  daily_rate * (days : ℝ) + mile_rate * (miles : ℝ)

/-- Theorem stating that under the given conditions, the total rental cost is $165. -/
theorem rental_cost_is_165 :
  let daily_rate : ℝ := 30
  let mile_rate : ℝ := 0.15
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mile_rate days miles = 165 := by
sorry


end rental_cost_is_165_l23_2367


namespace age_difference_l23_2376

theorem age_difference :
  ∀ (a b : ℕ),
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) →
  (10 * a + b) - (10 * b + a) = 18 := by
  sorry

end age_difference_l23_2376


namespace sequence_a_property_sequence_a_formula_l23_2310

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sorry

theorem sequence_a_property (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * sequence_a (n + 1) = n * (n - 1) * sequence_a n - (n - 2) * sequence_a (n - 1) := by sorry

theorem sequence_a_formula (n : ℕ) (h : n ≥ 2) : sequence_a n = 1 / n.factorial := by sorry

end sequence_a_property_sequence_a_formula_l23_2310


namespace mirror_wall_height_l23_2302

def hall_of_mirrors (wall1_width wall2_width wall3_width total_area : ℝ) : Prop :=
  ∃ (height : ℝ),
    wall1_width * height + wall2_width * height + wall3_width * height = total_area

theorem mirror_wall_height :
  hall_of_mirrors 30 30 20 960 →
  ∃ (height : ℝ), height = 12 := by
sorry

end mirror_wall_height_l23_2302


namespace hash_four_six_l23_2351

-- Define the operation #
def hash (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem hash_four_six : hash 4 6 = 4 := by
  sorry

end hash_four_six_l23_2351
