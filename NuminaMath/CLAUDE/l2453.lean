import Mathlib

namespace max_area_inscribed_triangle_l2453_245319

/-- The maximum area of a right-angled isosceles triangle inscribed in a 12x15 rectangle -/
theorem max_area_inscribed_triangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  let max_area := Real.sqrt (min a b ^ 2 / 2)
  ∃ (x y : ℝ), x ≤ a ∧ y ≤ b ∧ x = y ∧ x * y / 2 = max_area ^ 2 ∧ max_area ^ 2 = 72 := by
  sorry


end max_area_inscribed_triangle_l2453_245319


namespace fraction_simplification_l2453_245398

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end fraction_simplification_l2453_245398


namespace polynomial_sum_theorem_l2453_245373

theorem polynomial_sum_theorem (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (10 * d - 3 + 16 * d^2) + (4 * d + 7) = a * d + b + c * d^2 ∧ a + b + c = 34 := by
  sorry

end polynomial_sum_theorem_l2453_245373


namespace total_books_l2453_245393

/-- The total number of books Sandy, Benny, and Tim have together is 67. -/
theorem total_books (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) :
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end total_books_l2453_245393


namespace equal_area_rectangles_length_l2453_245377

/-- Given two rectangles of equal area, where one rectangle has dimensions 12 inches by 10 inches,
    and the other has a width of 5 inches, prove that the length of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_length (area jordan_length jordan_width carol_width : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : jordan_length = 12)
    (h3 : jordan_width = 10)
    (h4 : carol_width = 5)
    (h5 : area = carol_width * (area / carol_width)) :
    area / carol_width = 24 := by
  sorry

end equal_area_rectangles_length_l2453_245377


namespace min_value_theorem_l2453_245335

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + 3*b - 1 = 0) :
  ∃ (min : ℝ), min = 3 + 2*Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = 1/(2*a + b) + 1/(a + b) → x ≥ min :=
by sorry

end min_value_theorem_l2453_245335


namespace approx_small_number_to_large_place_l2453_245310

/-- Given a real number less than 10000, the highest meaningful place value
    for approximation is the hundreds place when attempting to approximate
    to the ten thousand place. -/
theorem approx_small_number_to_large_place (x : ℝ) : 
  x < 10000 → 
  ∃ (approx : ℝ), 
    (approx = 100 * ⌊x / 100⌋) ∧ 
    (∀ (y : ℝ), y = 1000 * ⌊x / 1000⌋ ∨ y = 10000 * ⌊x / 10000⌋ → |x - approx| ≤ |x - y|) :=
by sorry

end approx_small_number_to_large_place_l2453_245310


namespace inverse_proportion_l2453_245349

theorem inverse_proportion (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 1500 * 0.25 = k) :
  3000 * b = k → b = 0.125 := by sorry

end inverse_proportion_l2453_245349


namespace perimeter_of_remaining_figure_l2453_245300

/-- The perimeter of a rectangle after cutting out squares --/
def perimeter_after_cuts (length width num_cuts cut_size : ℕ) : ℕ :=
  2 * (length + width) + num_cuts * (4 * cut_size - 2 * cut_size)

/-- Theorem stating the perimeter of the remaining figure after cuts --/
theorem perimeter_of_remaining_figure :
  perimeter_after_cuts 40 30 10 5 = 240 := by
  sorry

end perimeter_of_remaining_figure_l2453_245300


namespace polar_to_hyperbola_l2453_245316

/-- Theorem: The polar equation ρ² cos(2θ) = 1 represents a hyperbola in Cartesian coordinates -/
theorem polar_to_hyperbola (ρ θ x y : ℝ) : 
  (ρ^2 * (Real.cos (2 * θ)) = 1) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  (x^2 - y^2 = 1) :=
by sorry

end polar_to_hyperbola_l2453_245316


namespace nancy_seeds_l2453_245308

/-- Calculates the total number of seeds Nancy started with. -/
def total_seeds (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Proves that Nancy started with 52 seeds given the problem conditions. -/
theorem nancy_seeds :
  let big_garden_seeds : ℕ := 28
  let small_gardens : ℕ := 6
  let seeds_per_small_garden : ℕ := 4
  total_seeds big_garden_seeds small_gardens seeds_per_small_garden = 52 := by
  sorry

#eval total_seeds 28 6 4

end nancy_seeds_l2453_245308


namespace arithmetic_equations_l2453_245305

theorem arithmetic_equations : 
  (12 * 12 / (12 + 12) = 6) ∧ ((12 * 12 + 12) / 12 = 13) := by
  sorry

end arithmetic_equations_l2453_245305


namespace oven_capacity_is_two_l2453_245354

/-- Represents the pizza-making process with given constraints -/
structure PizzaMaking where
  dough_time : ℕ  -- Time to make one batch of dough (in minutes)
  cook_time : ℕ   -- Time to cook pizzas in the oven (in minutes)
  pizzas_per_batch : ℕ  -- Number of pizzas one batch of dough can make
  total_time : ℕ  -- Total time to make all pizzas (in minutes)
  total_pizzas : ℕ  -- Total number of pizzas to be made

/-- Calculates the number of pizzas that can fit in the oven at once -/
def oven_capacity (pm : PizzaMaking) : ℕ :=
  let dough_making_time := (pm.total_pizzas / pm.pizzas_per_batch) * pm.dough_time
  let baking_time := pm.total_time - dough_making_time
  let baking_intervals := baking_time / pm.cook_time
  pm.total_pizzas / baking_intervals

/-- Theorem stating that given the conditions, the oven capacity is 2 pizzas -/
theorem oven_capacity_is_two (pm : PizzaMaking)
  (h1 : pm.dough_time = 30)
  (h2 : pm.cook_time = 30)
  (h3 : pm.pizzas_per_batch = 3)
  (h4 : pm.total_time = 300)  -- 5 hours = 300 minutes
  (h5 : pm.total_pizzas = 12) :
  oven_capacity pm = 2 := by
  sorry  -- Proof omitted

end oven_capacity_is_two_l2453_245354


namespace routes_on_3x2_grid_l2453_245395

/-- The number of routes on a grid with more right moves than down moves -/
def num_routes (right down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The grid dimensions -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_width grid_height = 21 :=
sorry

end routes_on_3x2_grid_l2453_245395


namespace stamp_collection_total_l2453_245337

theorem stamp_collection_total (foreign : ℕ) (old : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : foreign = 90)
  (h2 : old = 60)
  (h3 : both = 20)
  (h4 : neither = 70) :
  foreign + old - both + neither = 200 := by
  sorry

end stamp_collection_total_l2453_245337


namespace tenth_term_of_sequence_l2453_245370

def f (n : ℕ) (a : ℝ) : ℝ := (-2) ^ (n - 1) * a ^ n

theorem tenth_term_of_sequence (a : ℝ) : f 10 a = -2^9 * a^10 := by
  sorry

end tenth_term_of_sequence_l2453_245370


namespace amy_jeremy_age_ratio_l2453_245389

/-- Proves that the ratio of Amy's age to Jeremy's age is 1:3 given the specified conditions -/
theorem amy_jeremy_age_ratio :
  ∀ (amy_age jeremy_age chris_age : ℕ),
    jeremy_age = 66 →
    amy_age + jeremy_age + chris_age = 132 →
    chris_age = 2 * amy_age →
    (amy_age : ℚ) / jeremy_age = 1 / 3 :=
by
  sorry

end amy_jeremy_age_ratio_l2453_245389


namespace discount_percentage_retailer_discount_approx_25_percent_l2453_245302

/-- Calculates the discount percentage given markup and profit percentages -/
theorem discount_percentage (markup : ℝ) (actual_profit : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let actual_selling_price := 1 + actual_profit
  let discount := marked_price - actual_selling_price
  (discount / marked_price) * 100

/-- Proves that the discount percentage is approximately 25% given the specified markup and profit -/
theorem retailer_discount_approx_25_percent :
  ∀ (ε : ℝ), ε > 0 →
  abs (discount_percentage 0.60 0.20000000000000018 - 25) < ε :=
sorry

end discount_percentage_retailer_discount_approx_25_percent_l2453_245302


namespace intersection_of_M_and_N_l2453_245394

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {0} := by sorry

end intersection_of_M_and_N_l2453_245394


namespace parabola_vertex_on_negative_x_axis_l2453_245344

/-- Given a parabola y = x^2 - bx + 8, if its vertex lies on the negative half-axis of the x-axis, then b = -4√2 -/
theorem parabola_vertex_on_negative_x_axis (b : ℝ) :
  (∃ x, x < 0 ∧ x^2 - b*x + 8 = 0 ∧ ∀ y, y ≠ x → (y^2 - b*y + 8 > 0)) →
  b = -4 * Real.sqrt 2 := by
sorry

end parabola_vertex_on_negative_x_axis_l2453_245344


namespace toast_costs_one_pound_l2453_245392

/-- The cost of a slice of toast -/
def toast_cost : ℝ := sorry

/-- The cost of an egg -/
def egg_cost : ℝ := 3

/-- Dale's breakfast cost -/
def dale_breakfast : ℝ := 2 * toast_cost + 2 * egg_cost

/-- Andrew's breakfast cost -/
def andrew_breakfast : ℝ := toast_cost + 2 * egg_cost

/-- The total cost of both breakfasts -/
def total_cost : ℝ := 15

theorem toast_costs_one_pound :
  dale_breakfast + andrew_breakfast = total_cost →
  toast_cost = 1 := by sorry

end toast_costs_one_pound_l2453_245392


namespace family_theater_cost_l2453_245303

/-- Represents the cost of a theater ticket --/
structure TicketCost where
  full : ℝ
  senior : ℝ
  student : ℝ

/-- Calculates the total cost of tickets for a family group --/
def totalCost (t : TicketCost) : ℝ :=
  3 * t.senior + 3 * t.full + 3 * t.student

/-- Theorem: Given the specified discounts and senior ticket cost, 
    the total cost for all family members is $90 --/
theorem family_theater_cost : 
  ∀ (t : TicketCost), 
    t.senior = 10 ∧ 
    t.senior = 0.8 * t.full ∧ 
    t.student = 0.6 * t.full → 
    totalCost t = 90 := by
  sorry


end family_theater_cost_l2453_245303


namespace cos_2A_plus_cos_2B_l2453_245332

theorem cos_2A_plus_cos_2B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 := by
  sorry

end cos_2A_plus_cos_2B_l2453_245332


namespace chord_equation_l2453_245380

/-- The equation of the line on which the chord common to two circles lies -/
theorem chord_equation (r : ℝ) (ρ θ : ℝ) (h : r > 0) :
  (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end chord_equation_l2453_245380


namespace max_value_theorem_l2453_245327

theorem max_value_theorem (a b : ℝ) (h : a^2 - b^2 = -1) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x y : ℝ), x^2 - y^2 = -1 → (|x| + 1) / y ≤ M :=
sorry

end max_value_theorem_l2453_245327


namespace truck_speed_l2453_245320

/-- Proves that a truck traveling 600 meters in 40 seconds has a speed of 54 kilometers per hour -/
theorem truck_speed : ∀ (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ),
  distance = 600 →
  time = 40 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 54 := by
  sorry

#check truck_speed

end truck_speed_l2453_245320


namespace no_solutions_squared_l2453_245330

theorem no_solutions_squared (n : ℕ) (h : n > 2) :
  (∀ x y z : ℕ+, x^n + y^n ≠ z^n) →
  (∀ x y z : ℕ+, x^(2*n) + y^(2*n) ≠ z^2) :=
by sorry

end no_solutions_squared_l2453_245330


namespace vehicles_with_only_cd_player_l2453_245359

/-- Represents the percentage of vehicles with specific features -/
structure VehicleFeatures where
  power_windows : ℝ
  anti_lock_brakes : ℝ
  cd_player : ℝ
  power_windows_and_anti_lock : ℝ
  anti_lock_and_cd : ℝ
  power_windows_and_cd : ℝ

/-- The theorem stating the percentage of vehicles with only a CD player -/
theorem vehicles_with_only_cd_player (v : VehicleFeatures)
  (h1 : v.power_windows = 60)
  (h2 : v.anti_lock_brakes = 25)
  (h3 : v.cd_player = 75)
  (h4 : v.power_windows_and_anti_lock = 10)
  (h5 : v.anti_lock_and_cd = 15)
  (h6 : v.power_windows_and_cd = 22)
  (h7 : v.power_windows_and_anti_lock + v.anti_lock_and_cd + v.power_windows_and_cd ≤ v.cd_player) :
  v.cd_player - (v.power_windows_and_cd + v.anti_lock_and_cd) = 38 := by
  sorry

end vehicles_with_only_cd_player_l2453_245359


namespace sum_of_roots_l2453_245383

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = -13 := by
sorry

end sum_of_roots_l2453_245383


namespace system_of_equations_solution_system_of_inequalities_solution_l2453_245391

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y = 32 ∧ 2 * x - y = 0) → (x = 8 ∧ y = 16) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x - 1 < 5 - 2 * x ∧ 5 * x + 1 ≥ 2 * x + 3) →
  (2 / 3 ≤ x ∧ x < 6 / 5) := by sorry

end system_of_equations_solution_system_of_inequalities_solution_l2453_245391


namespace football_players_count_l2453_245361

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9)
  (h5 : total = (football - both) + (tennis - both) + both + neither) :
  football = 26 := by
  sorry

end football_players_count_l2453_245361


namespace inequality_proof_l2453_245386

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hne : m ≠ n) :
  (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by
  sorry

end inequality_proof_l2453_245386


namespace not_sufficient_nor_necessary_l2453_245345

/-- A quadratic function with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The condition that f has only one zero -/
def has_one_zero (a : ℝ) : Prop := ∃! x, f a x = 0

/-- The statement to be proved -/
theorem not_sufficient_nor_necessary :
  (∃ a, a ≤ -2 ∧ ¬(has_one_zero a)) ∧ 
  (∃ a, a > -2 ∧ has_one_zero a) :=
sorry

end not_sufficient_nor_necessary_l2453_245345


namespace absolute_sum_sequence_minimum_sum_l2453_245347

/-- An absolute sum sequence with given initial term and absolute public sum. -/
def AbsoluteSumSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => if n = 1 then a₁ else sorry

/-- The sum of the first n terms of an absolute sum sequence. -/
def SequenceSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem absolute_sum_sequence_minimum_sum :
  ∀ a : ℕ → ℝ,
  (a 1 = 2) →
  (∀ n : ℕ, |a (n + 1)| + |a n| = 3) →
  SequenceSum a 2019 ≥ -3025 ∧
  ∃ a : ℕ → ℝ, (a 1 = 2) ∧ (∀ n : ℕ, |a (n + 1)| + |a n| = 3) ∧ SequenceSum a 2019 = -3025 :=
by sorry

end absolute_sum_sequence_minimum_sum_l2453_245347


namespace smoothie_proportion_l2453_245368

/-- Given that 13 smoothies can be made from 3 bananas, prove that 65 smoothies can be made from 15 bananas. -/
theorem smoothie_proportion (make_smoothie : ℕ → ℕ) 
    (h : make_smoothie 3 = 13) : make_smoothie 15 = 65 := by
  sorry

#check smoothie_proportion

end smoothie_proportion_l2453_245368


namespace intersection_A_complement_B_l2453_245325

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {1, 2, 3, 4} := by sorry

end intersection_A_complement_B_l2453_245325


namespace max_value_real_complex_l2453_245309

theorem max_value_real_complex (α β : ℝ) :
  (∃ (M : ℝ), ∀ (x y : ℝ), abs x ≤ 1 → abs y ≤ 1 →
    abs (α * x + β * y) + abs (α * x - β * y) ≤ M ∧
    M = 2 * Real.sqrt 2 * Real.sqrt (α^2 + β^2)) ∧
  (∃ (N : ℝ), ∀ (x y : ℂ), Complex.abs x ≤ 1 → Complex.abs y ≤ 1 →
    Complex.abs (α * x + β * y) + Complex.abs (α * x - β * y) ≤ N ∧
    N = 2 * abs α + 2 * abs β) :=
by sorry

end max_value_real_complex_l2453_245309


namespace burger_non_filler_percentage_l2453_245367

/-- Given a burger with total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_non_filler_percentage 
  (total_weight : ℝ) 
  (filler_weight : ℝ) 
  (h1 : total_weight = 120)
  (h2 : filler_weight = 30) : 
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end burger_non_filler_percentage_l2453_245367


namespace smallest_added_number_l2453_245397

theorem smallest_added_number (n : ℤ) (x : ℕ) 
  (h1 : n % 25 = 4)
  (h2 : (n + x) % 5 = 4)
  (h3 : x > 0) :
  x = 5 := by
  sorry

end smallest_added_number_l2453_245397


namespace quadratic_roots_difference_l2453_245355

theorem quadratic_roots_difference (p : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -(2*p + 1)
  let c : ℝ := p*(p + 1)
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  max root1 root2 - min root1 root2 = 1 := by
sorry

end quadratic_roots_difference_l2453_245355


namespace gdp_2010_calculation_gdp_2010_l2453_245353

def gdp_2008 : ℝ := 1050
def growth_rate : ℝ := 0.132

theorem gdp_2010_calculation : 
  gdp_2008 * (1 + growth_rate)^2 = gdp_2008 * (1 + growth_rate) * (1 + growth_rate) :=
by sorry

theorem gdp_2010 : ℝ := gdp_2008 * (1 + growth_rate)^2

end gdp_2010_calculation_gdp_2010_l2453_245353


namespace complex_number_quadrant_l2453_245339

theorem complex_number_quadrant (z : ℂ) (h : (2 - I) * z = 5) :
  0 < z.re ∧ 0 < z.im := by sorry

end complex_number_quadrant_l2453_245339


namespace markese_earnings_l2453_245304

/-- Given Evan's earnings E, Markese's earnings (E - 5), and their total earnings of 37,
    prove that Markese earned 16 dollars. -/
theorem markese_earnings (E : ℕ) : E + (E - 5) = 37 → E - 5 = 16 := by
  sorry

end markese_earnings_l2453_245304


namespace evaluate_expression_l2453_245357

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l2453_245357


namespace fraction_of_three_fourths_is_one_fifth_l2453_245351

theorem fraction_of_three_fourths_is_one_fifth (x : ℚ) : x * (3 / 4 : ℚ) = (1 / 5 : ℚ) → x = (4 / 15 : ℚ) := by
  sorry

end fraction_of_three_fourths_is_one_fifth_l2453_245351


namespace hall_ratio_l2453_245360

theorem hall_ratio (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width * length = 288 →
  length - width = 12 →
  width / length = 1 / 2 := by
sorry

end hall_ratio_l2453_245360


namespace system_a_solution_l2453_245362

theorem system_a_solution (x y z t : ℝ) : 
  x - 3*y + 2*z - t = 3 ∧
  2*x + 4*y - 3*z + t = 5 ∧
  4*x - 2*y + z + t = 3 ∧
  3*x + y + z - 2*t = 10 →
  x = 2 ∧ y = -1 ∧ z = -3 ∧ t = -4 := by
sorry


end system_a_solution_l2453_245362


namespace sector_area_from_arc_length_l2453_245376

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) : 
  r * 2 = 4 → (1 / 2) * r^2 * 2 = 4 := by
  sorry

end sector_area_from_arc_length_l2453_245376


namespace sequence_comparison_theorem_l2453_245314

theorem sequence_comparison_theorem (a b : ℕ → ℕ) :
  ∃ r s : ℕ, r ≠ s ∧ a r ≥ a s ∧ b r ≥ b s := by
  sorry

end sequence_comparison_theorem_l2453_245314


namespace height_difference_is_4b_minus_8_l2453_245313

/-- A circle inside a parabola y = 4x^2, tangent at two points -/
structure TangentCircle where
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- x-coordinate of one tangent point (the other is -a) -/
  a : ℝ
  /-- The point (a, 4a^2) lies on the parabola -/
  tangent_on_parabola : 4 * a^2 = 4 * a^2
  /-- The point (a, 4a^2) lies on the circle -/
  tangent_on_circle : a^2 + (4 * a^2 - b)^2 = (b - 4 * a^2)^2 + a^2
  /-- Relation between a and b derived from tangency condition -/
  tangency_relation : 4 * b - a^2 = 8

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ := c.b - 4 * c.a^2

/-- Theorem: The height difference is always 4b - 8 -/
theorem height_difference_is_4b_minus_8 (c : TangentCircle) :
  height_difference c = 4 * c.b - 8 := by
  sorry

end height_difference_is_4b_minus_8_l2453_245313


namespace expression_simplification_l2453_245322

theorem expression_simplification (x y : ℝ) :
  3 * y + 4 * y^2 + 2 - (7 - 3 * y - 4 * y^2 + 2 * x) = 8 * y^2 + 6 * y - 2 * x - 5 := by
  sorry

end expression_simplification_l2453_245322


namespace pet_ownership_problem_l2453_245365

/-- Represents the number of students in each section of the Venn diagram -/
structure PetOwnership where
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_cats : ℕ
  cats_other : ℕ
  dogs_other : ℕ
  all_three : ℕ

/-- The main theorem to prove -/
theorem pet_ownership_problem (po : PetOwnership) : po.all_three = 4 :=
  by
  have total_students : ℕ := 40
  have dog_fraction : Rat := 5 / 8
  have cat_fraction : Rat := 1 / 4
  have other_pet_count : ℕ := 8
  have no_pet_count : ℕ := 4

  have dogs_only : po.dogs_only = 15 := by sorry
  have cats_only : po.cats_only = 3 := by sorry
  have other_only : po.other_only = 2 := by sorry

  have dog_eq : po.dogs_only + po.dogs_cats + po.dogs_other + po.all_three = (total_students : ℚ) * dog_fraction := by sorry
  have cat_eq : po.cats_only + po.dogs_cats + po.cats_other + po.all_three = (total_students : ℚ) * cat_fraction := by sorry
  have other_eq : po.other_only + po.cats_other + po.dogs_other + po.all_three = other_pet_count := by sorry
  have total_eq : po.dogs_only + po.cats_only + po.other_only + po.dogs_cats + po.cats_other + po.dogs_other + po.all_three = total_students - no_pet_count := by sorry

  sorry

end pet_ownership_problem_l2453_245365


namespace rose_jasmine_distance_l2453_245384

/-- Represents the positions of trees and flowers on a straight line -/
structure ForestLine where
  -- Positions of trees
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  -- Ensure trees are in order
  ab_pos : a < b
  bc_pos : b < c
  cd_pos : c < d
  de_pos : d < e
  -- Total distance between A and E is 28
  ae_dist : e - a = 28
  -- Positions of flowers
  daisy : ℝ
  rose : ℝ
  jasmine : ℝ
  carnation : ℝ
  -- Flowers at midpoints
  daisy_mid : daisy = (a + b) / 2
  rose_mid : rose = (b + c) / 2
  jasmine_mid : jasmine = (c + d) / 2
  carnation_mid : carnation = (d + e) / 2
  -- Distance between daisy and carnation is 20
  daisy_carnation_dist : carnation - daisy = 20

/-- The distance between the rose bush and the jasmine is 6 meters -/
theorem rose_jasmine_distance (f : ForestLine) : f.jasmine - f.rose = 6 := by
  sorry

end rose_jasmine_distance_l2453_245384


namespace cube_volume_problem_l2453_245315

theorem cube_volume_problem (a : ℝ) : 
  (a + 2) * (a + 2) * (a - 2) = a^3 - 16 → a^3 = 9 + 12 * Real.sqrt 5 := by
sorry

end cube_volume_problem_l2453_245315


namespace amy_soup_count_l2453_245343

/-- The number of cans of chicken soup Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of cans of tomato soup Amy bought -/
def tomato_soup : ℕ := 3

/-- The number of cans of vegetable soup Amy bought -/
def vegetable_soup : ℕ := 4

/-- The number of cans of clam chowder Amy bought -/
def clam_chowder : ℕ := 2

/-- The number of cans of French onion soup Amy bought -/
def french_onion_soup : ℕ := 1

/-- The number of cans of minestrone soup Amy bought -/
def minestrone_soup : ℕ := 5

/-- The total number of cans of soup Amy bought -/
def total_soups : ℕ := chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup

theorem amy_soup_count : total_soups = 21 := by
  sorry

end amy_soup_count_l2453_245343


namespace square_area_proof_l2453_245364

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 15 - 2 * x) → 
  ((3 * x - 12)^2 : ℝ) = 441 / 25 := by
  sorry

end square_area_proof_l2453_245364


namespace right_triangle_hypotenuse_l2453_245329

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end right_triangle_hypotenuse_l2453_245329


namespace smallest_addition_for_multiple_of_five_l2453_245324

theorem smallest_addition_for_multiple_of_five :
  ∀ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end smallest_addition_for_multiple_of_five_l2453_245324


namespace fourth_smallest_is_six_probability_l2453_245321

def S : Finset ℕ := Finset.range 15

def probability_fourth_smallest_is_six (n : ℕ) : ℚ :=
  let total_combinations := Nat.choose 15 8
  let favorable_outcomes := Nat.choose 5 3 * Nat.choose 9 5
  (favorable_outcomes : ℚ) / total_combinations

theorem fourth_smallest_is_six_probability :
  probability_fourth_smallest_is_six 6 = 4 / 21 := by
  sorry

#eval probability_fourth_smallest_is_six 6

end fourth_smallest_is_six_probability_l2453_245321


namespace healthcare_worker_identity_l2453_245318

/-- Represents the number of healthcare workers of each type -/
structure HealthcareWorkers where
  male_doctors : ℕ
  female_doctors : ℕ
  female_nurses : ℕ
  male_nurses : ℕ

/-- Checks if the given numbers satisfy all conditions -/
def satisfies_conditions (hw : HealthcareWorkers) : Prop :=
  hw.male_doctors + hw.female_doctors + hw.female_nurses + hw.male_nurses = 17 ∧
  hw.male_doctors + hw.female_doctors ≥ hw.female_nurses + hw.male_nurses ∧
  hw.female_nurses > hw.male_doctors ∧
  hw.male_doctors > hw.female_doctors ∧
  hw.male_nurses ≥ 2

/-- The unique solution that satisfies all conditions -/
def solution : HealthcareWorkers :=
  { male_doctors := 5
    female_doctors := 4
    female_nurses := 6
    male_nurses := 2 }

/-- The statement to be proved -/
theorem healthcare_worker_identity :
  satisfies_conditions solution ∧
  satisfies_conditions { male_doctors := solution.male_doctors,
                         female_doctors := solution.female_doctors - 1,
                         female_nurses := solution.female_nurses,
                         male_nurses := solution.male_nurses } ∧
  ∀ (hw : HealthcareWorkers), satisfies_conditions hw → hw = solution :=
sorry

end healthcare_worker_identity_l2453_245318


namespace max_volume_at_10cm_l2453_245369

/-- The side length of the original square sheet of metal in centimeters -/
def a : ℝ := 60

/-- The volume of the box as a function of the cut-out square's side length -/
def volume (x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 3600 - 480*x + 12*x^2

theorem max_volume_at_10cm :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  volume_derivative x = 0 ∧
  (∀ y : ℝ, y > 0 → y < a/2 → volume y ≤ volume x) ∧
  x = 10 :=
sorry

end max_volume_at_10cm_l2453_245369


namespace quadratic_roots_properties_l2453_245326

theorem quadratic_roots_properties :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := -42
  let product_of_roots := c / a
  let sum_of_roots := -b / a
  product_of_roots = -42 ∧ sum_of_roots = -4 := by
  sorry

end quadratic_roots_properties_l2453_245326


namespace min_triangle_area_l2453_245372

theorem min_triangle_area (p q : ℤ) : ∃ (min_area : ℚ), 
  min_area = 1 ∧ 
  ∀ (area : ℚ), area = (1 : ℚ) / 2 * |10 * p - 24 * q| → area ≥ min_area :=
sorry

end min_triangle_area_l2453_245372


namespace perfect_square_trinomial_l2453_245352

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 6*x + k = (x + a)^2) → k = 9 := by
  sorry

end perfect_square_trinomial_l2453_245352


namespace divisor_of_99_l2453_245379

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisor_of_99 (k : ℕ) 
  (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : 
  k ∣ 99 := by sorry

end divisor_of_99_l2453_245379


namespace simple_interest_rate_l2453_245381

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 800)
  (h2 : final_amount = 950)
  (h3 : time = 5)
  : (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end simple_interest_rate_l2453_245381


namespace prime_equation_solution_l2453_245301

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  (2 * p * q * r + 50 * p * q = A) ∧
  (7 * p * q * r + 55 * p * r = A) ∧
  (8 * p * q * r + 12 * q * r = A) →
  A = 1980 := by
sorry

end prime_equation_solution_l2453_245301


namespace larger_complementary_angle_measure_l2453_245385

def complementary_angles (a b : ℝ) : Prop := a + b = 90

theorem larger_complementary_angle_measure :
  ∀ (x y : ℝ),
    complementary_angles x y →
    x / y = 4 / 3 →
    x > y →
    x = 51 + 3 / 7 :=
by sorry

end larger_complementary_angle_measure_l2453_245385


namespace consecutive_page_numbers_l2453_245358

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 283 := by
  sorry

end consecutive_page_numbers_l2453_245358


namespace tan_2alpha_values_l2453_245396

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end tan_2alpha_values_l2453_245396


namespace preimage_of_3_1_l2453_245317

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
by
  sorry

end preimage_of_3_1_l2453_245317


namespace largest_fraction_l2453_245341

theorem largest_fraction : 
  let fractions : List ℚ := [2/3, 3/4, 2/5, 11/15]
  (3/4 : ℚ) = fractions.maximum := by sorry

end largest_fraction_l2453_245341


namespace videocassette_recorder_fraction_l2453_245333

theorem videocassette_recorder_fraction 
  (cable_fraction : Real) 
  (cable_and_vcr_fraction : Real) 
  (neither_fraction : Real) :
  cable_fraction = 1/5 →
  cable_and_vcr_fraction = 1/3 * cable_fraction →
  neither_fraction = 0.7666666666666667 →
  ∃ (vcr_fraction : Real),
    vcr_fraction = 1/10 ∧
    vcr_fraction + cable_fraction - cable_and_vcr_fraction + neither_fraction = 1 :=
by sorry

end videocassette_recorder_fraction_l2453_245333


namespace exists_m_with_infinite_solutions_l2453_245374

/-- The equation we're considering -/
def equation (m a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c)

/-- The existence of m with infinitely many solutions -/
theorem exists_m_with_infinite_solutions :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+, a > n ∧ b > n ∧ c > n ∧ equation m a b c :=
sorry

end exists_m_with_infinite_solutions_l2453_245374


namespace integral_cos_plus_exp_l2453_245342

theorem integral_cos_plus_exp : 
  ∫ x in -Real.pi..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp Real.pi := by
  sorry

end integral_cos_plus_exp_l2453_245342


namespace bus_problem_l2453_245371

theorem bus_problem (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 36 →
  got_off = 68 →
  final = 12 →
  got_off - (initial - got_off + final) = 24 :=
by sorry

end bus_problem_l2453_245371


namespace nested_square_root_fourth_power_l2453_245388

theorem nested_square_root_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1))))^4 = 2 + 2 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end nested_square_root_fourth_power_l2453_245388


namespace xyz_inequality_l2453_245331

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end xyz_inequality_l2453_245331


namespace dodecahedron_interior_diagonals_l2453_245328

/-- A dodecahedron is a polyhedron with 12 pentagonal faces and 20 vertices,
    where three faces meet at each vertex. -/
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_per_vertex : Nat
  faces_are_pentagonal : faces = 12
  vertex_count : vertices = 20
  three_faces_per_vertex : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on the same face. -/
def interior_diagonal (d : Dodecahedron) := Unit

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : Nat :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  num_interior_diagonals d = 160 := by
  sorry

#check dodecahedron_interior_diagonals

end dodecahedron_interior_diagonals_l2453_245328


namespace larger_cube_volume_l2453_245340

-- Define the volume of a smaller cube
def small_cube_volume : ℝ := 8

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 2

-- Theorem statement
theorem larger_cube_volume :
  ∀ (small_edge : ℝ) (large_edge : ℝ),
  small_edge > 0 →
  large_edge > 0 →
  small_edge^3 = small_cube_volume →
  num_small_cubes * small_edge = large_edge →
  large_edge^3 = 64 := by
sorry

end larger_cube_volume_l2453_245340


namespace min_box_value_l2453_245387

theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 45 * x^2 + box * x + 45) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ box_min : ℤ, box_min = 106 ∧ box ≥ box_min ∧
    (∀ a' b' box' : ℤ, 
      (∀ x, (a' * x + b') * (b' * x + a') = 45 * x^2 + box' * x + 45) →
      a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
      box' ≥ box_min)) :=
sorry

end min_box_value_l2453_245387


namespace log_base_three_seven_l2453_245382

theorem log_base_three_seven (a b : ℝ) (h1 : Real.log 2 / Real.log 3 = a) (h2 : Real.log 7 / Real.log 2 = b) :
  Real.log 7 / Real.log 3 = a * b := by
  sorry

end log_base_three_seven_l2453_245382


namespace plant_supplier_pots_cost_l2453_245306

/-- The cost of new pots for a plant supplier --/
theorem plant_supplier_pots_cost :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let remaining_money : ℕ := 1145
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + remaining_money
  total_earnings - total_expenses = 150 :=
by sorry

end plant_supplier_pots_cost_l2453_245306


namespace acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l2453_245307

/-- A triangle with side lengths a, b, and c is acute if and only if a² + b² > c², where c is the longest side. -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  c ≥ a ∧ c ≥ b ∧ a^2 + b^2 > c^2

/-- The theorem states that for an acute triangle with side lengths x²+4, 4x, and x²+6,
    where x is a positive real number, x must be greater than √(15)/3. -/
theorem acute_triangle_side_constraint (x : ℝ) :
  x > 0 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) →
  x > Real.sqrt 15 / 3 :=
by sorry

/-- The converse of the theorem: if x > √(15)/3, then the triangle with side lengths
    x²+4, 4x, and x²+6 is acute. -/
theorem acute_triangle_side_constraint_converse (x : ℝ) :
  x > Real.sqrt 15 / 3 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) :=
by sorry

end acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l2453_245307


namespace shells_per_friend_eq_l2453_245348

/-- The number of shells each friend gets when Jillian, Savannah, and Clayton
    distribute their shells evenly among F friends. -/
def shellsPerFriend (F : ℕ+) : ℚ :=
  let J : ℕ := 29  -- Jillian's shells
  let S : ℕ := 17  -- Savannah's shells
  let C : ℕ := 8   -- Clayton's shells
  (J + S + C) / F

/-- Theorem stating that the number of shells each friend gets is 54 / F. -/
theorem shells_per_friend_eq (F : ℕ+) : shellsPerFriend F = 54 / F := by
  sorry

end shells_per_friend_eq_l2453_245348


namespace team_pays_seventy_percent_l2453_245363

/-- Represents the archer's arrow usage and costs --/
structure ArcherData where
  shots_per_day : ℕ
  days_per_week : ℕ
  recovery_rate : ℚ
  arrow_cost : ℚ
  weekly_spending : ℚ

/-- Calculates the percentage of arrow costs paid by the team --/
def team_payment_percentage (data : ArcherData) : ℚ :=
  let total_shots := data.shots_per_day * data.days_per_week
  let unrecovered_arrows := total_shots * (1 - data.recovery_rate)
  let total_cost := unrecovered_arrows * data.arrow_cost
  let team_contribution := total_cost - data.weekly_spending
  (team_contribution / total_cost) * 100

/-- Theorem stating that the team pays 70% of the archer's arrow costs --/
theorem team_pays_seventy_percent (data : ArcherData)
  (h1 : data.shots_per_day = 200)
  (h2 : data.days_per_week = 4)
  (h3 : data.recovery_rate = 1/5)
  (h4 : data.arrow_cost = 11/2)
  (h5 : data.weekly_spending = 1056) :
  team_payment_percentage data = 70 := by
  sorry

end team_pays_seventy_percent_l2453_245363


namespace cube_surface_area_l2453_245399

/-- Given a cube where the sum of all edge lengths is 180 cm, 
    prove that its surface area is 1350 cm². -/
theorem cube_surface_area (edge_sum : ℝ) (h_edge_sum : edge_sum = 180) :
  let edge_length := edge_sum / 12
  6 * edge_length^2 = 1350 := by sorry

end cube_surface_area_l2453_245399


namespace mms_given_to_sister_correct_l2453_245346

/-- The number of m&m's Cheryl gave to her sister -/
def mms_given_to_sister (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) : ℕ :=
  initial - (eaten_lunch + eaten_dinner)

/-- Theorem stating that the number of m&m's given to sister is correct -/
theorem mms_given_to_sister_correct (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) 
  (h1 : initial ≥ eaten_lunch + eaten_dinner) :
  mms_given_to_sister initial eaten_lunch eaten_dinner = initial - (initial - (eaten_lunch + eaten_dinner)) :=
by
  sorry

#eval mms_given_to_sister 25 7 5

end mms_given_to_sister_correct_l2453_245346


namespace solution_part1_solution_part2_l2453_245356

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part 1: Prove the solution set for f(x) + |2x-3| > 0 when a = 2
theorem solution_part1 : 
  {x : ℝ | f 2 x + |2*x - 3| > 0} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} := by sorry

-- Part 2: Prove the range of a for which f(x) > |x-3| has solutions
theorem solution_part2 : 
  {a : ℝ | ∃ x, f a x > |x - 3|} = {a : ℝ | a < 2 ∨ a > 4} := by sorry

end solution_part1_solution_part2_l2453_245356


namespace divisible_by_six_l2453_245334

theorem divisible_by_six (n : ℕ) : 
  6 ∣ (n + 20) * (n + 201) * (n + 2020) := by
  sorry

end divisible_by_six_l2453_245334


namespace proposition_relationship_l2453_245375

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end proposition_relationship_l2453_245375


namespace range_of_a_theorem_l2453_245390

/-- The range of a, given the conditions in the problem -/
def range_of_a : Set ℝ :=
  {a | a ≤ -2 ∨ a = 1}

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

/-- Proposition q: There exists x₀ ∈ ℝ such that x₀^2 + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The main theorem stating that given the conditions, the range of a is as defined -/
theorem range_of_a_theorem (a : ℝ) :
  (prop_p a ∧ prop_q a) → a ∈ range_of_a :=
sorry

end range_of_a_theorem_l2453_245390


namespace inheritance_solution_l2453_245366

/-- Represents the inheritance problem with given conditions --/
def inheritance_problem (total : ℝ) : Prop :=
  ∃ (x : ℝ),
    x > 0 ∧
    (total - x) > 0 ∧
    0.05 * x + 0.065 * (total - x) = 227 ∧
    total - x = 1800

/-- The theorem stating the solution to the inheritance problem --/
theorem inheritance_solution :
  ∃ (total : ℝ), inheritance_problem total ∧ total = 4000 := by
  sorry

end inheritance_solution_l2453_245366


namespace chicken_cost_per_person_l2453_245323

def total_cost : ℝ := 16
def beef_price_per_pound : ℝ := 4
def beef_pounds : ℝ := 3
def oil_price : ℝ := 1
def number_of_people : ℕ := 3

theorem chicken_cost_per_person : 
  (total_cost - (beef_price_per_pound * beef_pounds + oil_price)) / number_of_people = 1 := by
  sorry

end chicken_cost_per_person_l2453_245323


namespace simplify_polynomial_expression_l2453_245378

theorem simplify_polynomial_expression (x : ℝ) :
  (3 * x - 4) * (x + 9) + (x + 6) * (3 * x + 2) = 6 * x^2 + 43 * x - 24 := by
  sorry

end simplify_polynomial_expression_l2453_245378


namespace no_solution_x6_2y2_plus_2_l2453_245311

theorem no_solution_x6_2y2_plus_2 : ∀ (x y : ℤ), x^6 ≠ 2*y^2 + 2 := by
  sorry

end no_solution_x6_2y2_plus_2_l2453_245311


namespace three_digit_twice_divisible_by_1001_l2453_245336

theorem three_digit_twice_divisible_by_1001 (a : ℕ) : 
  100 ≤ a ∧ a < 1000 → ∃ k : ℕ, 1000 * a + a = 1001 * k := by
sorry

end three_digit_twice_divisible_by_1001_l2453_245336


namespace complex_modulus_equality_l2453_245350

theorem complex_modulus_equality (x : ℝ) :
  x > 0 → (Complex.abs (5 + x * Complex.I) = 13 ↔ x = 12) := by sorry

end complex_modulus_equality_l2453_245350


namespace fraction_sum_l2453_245312

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end fraction_sum_l2453_245312


namespace power_fraction_simplification_l2453_245338

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end power_fraction_simplification_l2453_245338
