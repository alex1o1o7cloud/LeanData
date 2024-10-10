import Mathlib

namespace square_area_is_four_l1052_105242

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a division of the square
structure SquareDivision where
  square : Square
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  sum_areas : area1 + area2 + area3 + area4 = square.side ^ 2
  perpendicular_division : True  -- This is a placeholder for the perpendicular division condition

-- Theorem statement
theorem square_area_is_four 
  (div : SquareDivision) 
  (h1 : div.area1 = 1) 
  (h2 : div.area2 = 1) 
  (h3 : div.area3 = 1) : 
  div.square.side ^ 2 = 4 := by
  sorry


end square_area_is_four_l1052_105242


namespace f_max_value_f_no_real_roots_l1052_105218

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 10

-- Theorem for the maximum value
theorem f_max_value :
  ∃ (x_max : ℝ), f x_max = -2 ∧ ∀ (x : ℝ), f x ≤ f x_max ∧ x_max = 2 :=
sorry

-- Theorem for no real roots
theorem f_no_real_roots :
  ∀ (x : ℝ), f x ≠ 0 :=
sorry

end f_max_value_f_no_real_roots_l1052_105218


namespace only_event3_mutually_exclusive_l1052_105200

-- Define the set of numbers
def NumberSet : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define the sample space
def SampleSpace : Set (Nat × Nat) :=
  {pair | pair.1 ∈ NumberSet ∧ pair.2 ∈ NumberSet ∧ pair.1 ≠ pair.2}

-- Define event ①
def Event1 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 0 ∧ pair.2 % 2 = 1) ∨ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 0)

-- Define event ②
def Event2 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 1)

-- Define event ③
def Event3 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∧ pair.2 % 2 = 0)

-- Define event ④
def Event4 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∨ pair.2 % 2 = 0)

-- Theorem stating that only Event3 is mutually exclusive with other events
theorem only_event3_mutually_exclusive :
  ∀ (pair : Nat × Nat), pair ∈ SampleSpace →
    (¬(Event1 pair ∧ Event3 pair) ∧
     ¬(Event2 pair ∧ Event3 pair) ∧
     ¬(Event4 pair ∧ Event3 pair)) ∧
    ((Event1 pair ∧ Event2 pair) ∨
     (Event1 pair ∧ Event4 pair) ∨
     (Event2 pair ∧ Event4 pair)) :=
by sorry

end only_event3_mutually_exclusive_l1052_105200


namespace smallest_next_divisor_after_221_l1052_105273

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ (∀ (x : ℕ), 221 < x ∧ x < d → m % x ≠ 0) → d = 247 :=
sorry

end smallest_next_divisor_after_221_l1052_105273


namespace coupon1_best_discount_l1052_105280

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 30

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

/-- Theorem stating the condition for Coupon 1 to offer the greatest discount -/
theorem coupon1_best_discount (x : ℝ) :
  (coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x) ↔
  (200 < x ∧ x < 471.43) :=
sorry

end coupon1_best_discount_l1052_105280


namespace soap_calculation_l1052_105265

/-- Given a number of packs and bars per pack, calculates the total number of bars -/
def total_bars (packs : ℕ) (bars_per_pack : ℕ) : ℕ := packs * bars_per_pack

/-- Theorem stating that 6 packs with 5 bars each results in 30 total bars -/
theorem soap_calculation : total_bars 6 5 = 30 := by
  sorry

end soap_calculation_l1052_105265


namespace small_poster_price_is_six_l1052_105262

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  days_per_week : ℕ

/-- Calculates the selling price of small posters --/
def small_poster_price (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let daily_profit := business.weekly_profit / business.days_per_week
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let daily_large_poster_profit := large_poster_profit * business.large_posters_per_day
  let daily_small_poster_profit := daily_profit - daily_large_poster_profit
  let small_poster_profit := daily_small_poster_profit / small_posters_per_day
  small_poster_profit + business.small_poster_cost

/-- Theorem stating that the small poster price is $6 --/
theorem small_poster_price_is_six (business : PosterBusiness) 
    (h1 : business.total_posters_per_day = 5)
    (h2 : business.large_posters_per_day = 2)
    (h3 : business.large_poster_price = 10)
    (h4 : business.large_poster_cost = 5)
    (h5 : business.small_poster_cost = 3)
    (h6 : business.weekly_profit = 95)
    (h7 : business.days_per_week = 5) :
  small_poster_price business = 6 := by
  sorry

#eval small_poster_price {
  total_posters_per_day := 5,
  large_posters_per_day := 2,
  large_poster_price := 10,
  large_poster_cost := 5,
  small_poster_cost := 3,
  weekly_profit := 95,
  days_per_week := 5
}

end small_poster_price_is_six_l1052_105262


namespace dans_cards_l1052_105285

theorem dans_cards (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 27 → bought = 20 → total = 88 → total - bought - initial = 41 := by
  sorry

end dans_cards_l1052_105285


namespace symmetry_about_x_axis_periodicity_symmetry_about_origin_l1052_105271

-- Define a real-valued function on reals
variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetry_about_x_axis (x : ℝ) : 
  f (-1 - x) = f (-(x - 1)) := by sorry

-- Statement 2
theorem periodicity (x : ℝ) : 
  f (1 + x) = f (x - 1) → f (x + 2) = f x := by sorry

-- Statement 3
theorem symmetry_about_origin (x : ℝ) : 
  f (1 - x) = -f (x - 1) → f (-x) = -f x := by sorry

end symmetry_about_x_axis_periodicity_symmetry_about_origin_l1052_105271


namespace semicircle_perimeter_approx_l1052_105261

/-- The perimeter of a semicircle with radius 11 is approximately 56.56 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 11
  let π_approx : ℝ := 3.14159
  let semicircle_perimeter := π_approx * r + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 56.56) < ε :=
by sorry

end semicircle_perimeter_approx_l1052_105261


namespace custom_mult_three_four_l1052_105289

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

/-- Theorem stating that 3 * 4 = 12 under the custom multiplication -/
theorem custom_mult_three_four : custom_mult 3 4 = 12 := by
  sorry

end custom_mult_three_four_l1052_105289


namespace point_on_h_graph_and_coordinate_sum_l1052_105292

/-- Given a function g where g(4) = 8, and h defined as h(x) = 2(g(x))^3,
    prove that (4,1024) is on the graph of h and the sum of its coordinates is 1028 -/
theorem point_on_h_graph_and_coordinate_sum 
  (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = 2 * (g x)^3)
  (g_value : g 4 = 8) :
  h 4 = 1024 ∧ 4 + 1024 = 1028 := by
  sorry

end point_on_h_graph_and_coordinate_sum_l1052_105292


namespace negation_of_union_membership_l1052_105299

theorem negation_of_union_membership (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end negation_of_union_membership_l1052_105299


namespace opposite_implies_sum_l1052_105226

theorem opposite_implies_sum (x : ℝ) : 
  (3 - x) = -2 → x + 1 = 6 := by
  sorry

end opposite_implies_sum_l1052_105226


namespace coins_player1_l1052_105294

/-- Represents the number of sectors and players -/
def n : ℕ := 9

/-- Represents the number of rotations -/
def rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def coins_player4 : ℕ := 90

/-- Represents the coins received by player 8 -/
def coins_player8 : ℕ := 35

/-- Theorem stating the number of coins received by player 1 -/
theorem coins_player1 (h1 : n = 9) (h2 : rotations = 11) 
  (h3 : coins_player4 = 90) (h4 : coins_player8 = 35) : 
  ∃ (coins_player1 : ℕ), coins_player1 = 57 :=
sorry


end coins_player1_l1052_105294


namespace parabola_directrix_l1052_105291

/-- Given a parabola with equation x^2 = -1/8 * y, its directrix equation is y = 1/32 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = -1/8 * y) → (∃ (k : ℝ), k = 1/32 ∧ k = y) :=
by sorry

end parabola_directrix_l1052_105291


namespace quadratic_constant_term_l1052_105254

theorem quadratic_constant_term (a : ℝ) : 
  ((∀ x, (a + 2) * x^2 - 3 * a * x + a - 6 = 0 → (a + 2) ≠ 0) ∧ 
   a - 6 = 0) → 
  a = 6 := by
sorry

end quadratic_constant_term_l1052_105254


namespace max_profit_at_zero_optimal_investment_l1052_105232

/-- Profit function --/
def profit (m : ℝ) : ℝ := 28 - 3 * m

/-- Theorem: The profit function achieves its maximum when m = 0, given m ≥ 0 --/
theorem max_profit_at_zero (m : ℝ) (h : m ≥ 0) : profit 0 ≥ profit m := by
  sorry

/-- Corollary: The optimal investment for maximum profit is 0 --/
theorem optimal_investment : ∃ (m : ℝ), m = 0 ∧ ∀ (n : ℝ), n ≥ 0 → profit m ≥ profit n := by
  sorry

end max_profit_at_zero_optimal_investment_l1052_105232


namespace right_building_shorter_l1052_105251

def middle_height : ℝ := 100
def left_height : ℝ := 0.8 * middle_height
def total_height : ℝ := 340

theorem right_building_shorter : 
  (middle_height + left_height) - (total_height - (middle_height + left_height)) = 20 := by
  sorry

end right_building_shorter_l1052_105251


namespace problem_solution_l1052_105263

theorem problem_solution (a b c d e x : ℝ) 
  (h : ((x + a) ^ b) / c - d = e / 2) : 
  x = (c * e / 2 + c * d) ^ (1 / b) - a := by
sorry

end problem_solution_l1052_105263


namespace closest_point_is_vertex_l1052_105278

/-- Given a parabola y² = -2x and a point A(m, 0), if the point on the parabola 
closest to A is the vertex of the parabola, then m ∈ [-1, +∞). -/
theorem closest_point_is_vertex (m : ℝ) : 
  (∀ x y : ℝ, y^2 = -2*x → 
    (∀ x' y' : ℝ, y'^2 = -2*x' → (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2) → 
    x = 0 ∧ y = 0) → 
  m ≥ -1 := by
sorry

end closest_point_is_vertex_l1052_105278


namespace f_of_five_eq_six_elevenths_l1052_105286

/-- Given a function f(x) = (x+1) / (3x-4), prove that f(5) = 6/11 -/
theorem f_of_five_eq_six_elevenths :
  let f : ℝ → ℝ := λ x ↦ (x + 1) / (3 * x - 4)
  f 5 = 6 / 11 := by
  sorry

end f_of_five_eq_six_elevenths_l1052_105286


namespace average_after_addition_l1052_105266

theorem average_after_addition (numbers : List ℝ) (initial_average : ℝ) (addition : ℝ) : 
  numbers.length = 15 →
  initial_average = 40 →
  addition = 12 →
  (numbers.map (· + addition)).sum / numbers.length = 52 := by
  sorry

end average_after_addition_l1052_105266


namespace lcm_factor_problem_l1052_105252

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) (hcf : Nat.gcd A B = 42) 
  (lcm : Nat.lcm A B = 42 * X * 14) (a_val : A = 588) (a_greater : A > B) : X = 1 := by
  sorry

end lcm_factor_problem_l1052_105252


namespace omega_range_l1052_105202

/-- Given a function f(x) = sin(ωx + π/4) where ω > 0, 
    if f(x) is monotonically decreasing in the interval (π/2, π),
    then 1/2 ≤ ω ≤ 5/4 -/
theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  (∀ x ∈ Set.Ioo (π / 2) π, ∀ y ∈ Set.Ioo (π / 2) π, x < y → f x > f y) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end omega_range_l1052_105202


namespace max_production_years_l1052_105211

/-- The cumulative production function after n years -/
def f (n : ℕ) : ℚ := (1/2) * n * (n + 1) * (2 * n + 1)

/-- The annual production function -/
def annual_production (n : ℕ) : ℚ := 
  if n = 1 then f 1 else f n - f (n - 1)

/-- The maximum allowed annual production -/
def max_allowed_production : ℚ := 150

/-- The maximum number of years the production line can operate -/
def max_years : ℕ := 7

theorem max_production_years : 
  (∀ n : ℕ, n ≤ max_years → annual_production n ≤ max_allowed_production) ∧
  (annual_production (max_years + 1) > max_allowed_production) :=
sorry

end max_production_years_l1052_105211


namespace principal_calculation_l1052_105257

/-- Proves that given specific conditions, the principal amount is 1200 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2 + 2 / 5 →
  amount = 1344 →
  amount = (1200 : ℝ) * (1 + rate * time) :=
by sorry

end principal_calculation_l1052_105257


namespace blue_paint_calculation_l1052_105256

/-- Given the total amount of paint and the amount of white paint used,
    calculate the amount of blue paint used. -/
theorem blue_paint_calculation (total_paint white_paint : ℕ) 
    (h1 : total_paint = 6689)
    (h2 : white_paint = 660) :
    total_paint - white_paint = 6029 := by
  sorry

end blue_paint_calculation_l1052_105256


namespace bus_average_speed_with_stoppages_l1052_105260

/-- Calculates the average speed of a bus including stoppages -/
theorem bus_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 50 →
  stoppage_time = 12 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 40 :=
by sorry

end bus_average_speed_with_stoppages_l1052_105260


namespace cards_per_page_l1052_105275

/-- Given Will's baseball card organization problem, prove that he puts 3 cards on each page. -/
theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end cards_per_page_l1052_105275


namespace contradiction_elements_correct_l1052_105282

/-- Elements used in the method of contradiction -/
inductive ContradictionElement
  | assumption
  | originalCondition
  | axiomTheoremDefinition

/-- The set of elements used in the method of contradiction -/
def contradictionElements : Set ContradictionElement :=
  {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition}

/-- Theorem stating that the set of elements used in the method of contradiction
    is exactly the set containing assumptions, original conditions, and axioms/theorems/definitions -/
theorem contradiction_elements_correct :
  contradictionElements = {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition} := by
  sorry


end contradiction_elements_correct_l1052_105282


namespace max_y_coordinate_difference_l1052_105293

-- Define the two functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := 2 + x^2 + x^3

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Define the y-coordinates of the intersection points
def y_coordinates : Set ℝ := {y : ℝ | ∃ x ∈ intersection_points, f x = y}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (y1 y2 : ℝ), y1 ∈ y_coordinates ∧ y2 ∈ y_coordinates ∧
  ∀ (z1 z2 : ℝ), z1 ∈ y_coordinates → z2 ∈ y_coordinates →
  |y1 - y2| ≥ |z1 - z2| ∧ |y1 - y2| = 2 :=
sorry

end max_y_coordinate_difference_l1052_105293


namespace polygon_sides_count_l1052_105208

theorem polygon_sides_count (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end polygon_sides_count_l1052_105208


namespace decimal_division_proof_l1052_105290

theorem decimal_division_proof : 
  (0.182 : ℚ) / (0.0021 : ℚ) = 86 + 14 / 21 := by sorry

end decimal_division_proof_l1052_105290


namespace probability_one_or_two_first_20_rows_l1052_105274

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n ≥ 1 then 2 * n - 1 else 0

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def countTwos (n : ℕ) : ℕ := if n ≥ 3 then 2 * (n - 2) else 0

/-- The probability of selecting either 1 or 2 from the first n rows of Pascal's Triangle -/
def probabilityOneOrTwo (n : ℕ) : ℚ :=
  (countOnes n + countTwos n : ℚ) / (totalElements n : ℚ)

theorem probability_one_or_two_first_20_rows :
  probabilityOneOrTwo 20 = 5 / 14 := by sorry

end probability_one_or_two_first_20_rows_l1052_105274


namespace units_digit_of_five_consecutive_integers_l1052_105298

theorem units_digit_of_five_consecutive_integers (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 := by
sorry

end units_digit_of_five_consecutive_integers_l1052_105298


namespace train_speed_theorem_l1052_105217

theorem train_speed_theorem (passing_pole_time passing_train_time stationary_train_length : ℝ) 
  (h1 : passing_pole_time = 12)
  (h2 : passing_train_time = 27)
  (h3 : stationary_train_length = 300) :
  let train_length := (passing_train_time * stationary_train_length) / (passing_train_time - passing_pole_time)
  let train_speed := train_length / passing_pole_time
  train_speed = 20 := by sorry

end train_speed_theorem_l1052_105217


namespace division_problem_l1052_105249

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 729 ∧ quotient = 19 ∧ remainder = 7 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 38 := by
sorry

end division_problem_l1052_105249


namespace divisibility_quotient_l1052_105233

theorem divisibility_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_div : (a * b) ∣ (a^2 + b^2 + 1)) : 
  (a^2 + b^2 + 1) / (a * b) = 3 := by
  sorry

end divisibility_quotient_l1052_105233


namespace quadratic_inequality_range_l1052_105213

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end quadratic_inequality_range_l1052_105213


namespace cube_root_of_number_with_given_square_roots_l1052_105288

theorem cube_root_of_number_with_given_square_roots (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 1)^2 = x ∧ (a + 11)^2 = x) →
  ∃ (y : ℝ), y^3 = x ∧ y = 4 := by
  sorry

end cube_root_of_number_with_given_square_roots_l1052_105288


namespace power_division_23_l1052_105296

theorem power_division_23 : (23 : ℕ)^11 / (23 : ℕ)^8 = 12167 := by
  sorry

end power_division_23_l1052_105296


namespace car_distance_proof_l1052_105224

/-- Proves that the initial distance covered by a car is 180 km, given the conditions of the problem. -/
theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 20 →
  ∃ (D : ℝ),
    D = new_speed * (3/2 * initial_time) ∧
    D = 180 :=
by
  sorry

#check car_distance_proof

end car_distance_proof_l1052_105224


namespace three_equidistant_points_l1052_105244

/-- A color type with two possible values -/
inductive Color
| red
| blue

/-- A point on a straight line -/
structure Point where
  x : ℝ

/-- A coloring function that assigns a color to each point on the line -/
def Coloring := Point → Color

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

theorem three_equidistant_points (c : Coloring) :
  ∃ (A B C : Point), c A = c B ∧ c B = c C ∧ distance A B = distance B C :=
sorry

end three_equidistant_points_l1052_105244


namespace cube_equation_solution_l1052_105246

theorem cube_equation_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end cube_equation_solution_l1052_105246


namespace arithmetic_geometric_mean_product_product_of_means_2_8_l1052_105236

theorem arithmetic_geometric_mean_product (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) :
  let arithmetic_mean := (x + y) / 2
  let geometric_mean := Real.sqrt (x * y)
  arithmetic_mean * geometric_mean = (x + y) * Real.sqrt (x * y) / 2 :=
by sorry

theorem product_of_means_2_8 :
  let arithmetic_mean := (2 + 8) / 2
  let geometric_mean := Real.sqrt (2 * 8)
  (arithmetic_mean * geometric_mean = 20 ∨ arithmetic_mean * geometric_mean = -20) :=
by sorry

end arithmetic_geometric_mean_product_product_of_means_2_8_l1052_105236


namespace quadratic_root_value_l1052_105297

/-- Given a quadratic equation with real coefficients x^2 + px + q = 0,
    if b+i and 2-ai (where a and b are real) are its roots, then q = 5 -/
theorem quadratic_root_value (p q a b : ℝ) : 
  (∀ x : ℂ, x^2 + p*x + q = 0 ↔ x = b + I ∨ x = 2 - a*I) →
  q = 5 := by sorry

end quadratic_root_value_l1052_105297


namespace tree_height_problem_l1052_105210

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end tree_height_problem_l1052_105210


namespace dinner_time_l1052_105279

theorem dinner_time (total_time homework_time cleaning_time trash_time dishwasher_time : ℕ)
  (h1 : total_time = 120)
  (h2 : homework_time = 30)
  (h3 : cleaning_time = 30)
  (h4 : trash_time = 5)
  (h5 : dishwasher_time = 10) :
  total_time - (homework_time + cleaning_time + trash_time + dishwasher_time) = 45 := by
  sorry

end dinner_time_l1052_105279


namespace square_area_12m_l1052_105248

theorem square_area_12m (side_length : ℝ) (area : ℝ) : 
  side_length = 12 → area = side_length^2 → area = 144 := by sorry

end square_area_12m_l1052_105248


namespace bailey_towel_cost_l1052_105216

/-- Calculates the total cost of towel sets after discount -/
def towel_cost_after_discount (guest_sets : ℕ) (master_sets : ℕ) 
                               (guest_price : ℚ) (master_price : ℚ) 
                               (discount_percent : ℚ) : ℚ :=
  let total_cost := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_percent * total_cost
  total_cost - discount_amount

/-- Theorem stating that Bailey's total cost for towel sets is $224.00 -/
theorem bailey_towel_cost :
  towel_cost_after_discount 2 4 40 50 (20 / 100) = 224 :=
by sorry

end bailey_towel_cost_l1052_105216


namespace y_intercept_of_f_l1052_105229

/-- A linear function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The y-intercept of f is the point (0, 1) -/
theorem y_intercept_of_f :
  f 0 = 1 := by
  sorry

end y_intercept_of_f_l1052_105229


namespace diesel_cost_per_gallon_l1052_105238

/-- The cost of diesel fuel per gallon, given weekly spending and bi-weekly usage -/
theorem diesel_cost_per_gallon 
  (weekly_spending : ℝ) 
  (biweekly_usage : ℝ) 
  (h1 : weekly_spending = 36) 
  (h2 : biweekly_usage = 24) : 
  weekly_spending * 2 / biweekly_usage = 3 := by
sorry

end diesel_cost_per_gallon_l1052_105238


namespace candy_box_price_l1052_105277

/-- Proves that the current price of a candy box is 15 pounds given the conditions -/
theorem candy_box_price (
  soda_price : ℝ)
  (candy_increase : ℝ)
  (soda_increase : ℝ)
  (original_total : ℝ)
  (h1 : soda_price = 6)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.50)
  (h4 : original_total = 16) :
  ∃ (candy_price : ℝ), candy_price = 15 := by
  sorry

end candy_box_price_l1052_105277


namespace cubic_equations_common_root_l1052_105214

/-- Given real numbers a, b, c, if every pair of equations from 
    x³ - ax² + b = 0, x³ - bx² + c = 0, x³ - cx² + a = 0 has a common root, 
    then a = b = c. -/
theorem cubic_equations_common_root (a b c : ℝ) 
  (h1 : ∃ x : ℝ, x^3 - a*x^2 + b = 0 ∧ x^3 - b*x^2 + c = 0)
  (h2 : ∃ x : ℝ, x^3 - b*x^2 + c = 0 ∧ x^3 - c*x^2 + a = 0)
  (h3 : ∃ x : ℝ, x^3 - c*x^2 + a = 0 ∧ x^3 - a*x^2 + b = 0) :
  a = b ∧ b = c := by
  sorry

end cubic_equations_common_root_l1052_105214


namespace cubic_factorization_l1052_105272

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end cubic_factorization_l1052_105272


namespace simplify_expression_l1052_105223

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end simplify_expression_l1052_105223


namespace max_value_3cos_minus_sin_l1052_105219

theorem max_value_3cos_minus_sin :
  ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 ∧
  ∃ x : ℝ, 3 * Real.cos x - Real.sin x = Real.sqrt 10 :=
by sorry

end max_value_3cos_minus_sin_l1052_105219


namespace expression_evaluation_l1052_105268

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^4 + 1) / x^2 * (y^4 + 1) / y^2 - (x^4 - 1) / y^2 * (y^4 - 1) / x^2 = 2 * x^2 / y^2 + 2 * y^2 / x^2 := by
  sorry

end expression_evaluation_l1052_105268


namespace coffee_per_day_l1052_105204

/-- The number of times Maria goes to the coffee shop per day. -/
def visits_per_day : ℕ := 2

/-- The number of cups of coffee Maria orders each visit. -/
def cups_per_visit : ℕ := 3

/-- Theorem: Maria orders 6 cups of coffee per day. -/
theorem coffee_per_day : visits_per_day * cups_per_visit = 6 := by
  sorry

end coffee_per_day_l1052_105204


namespace set_operation_equality_l1052_105207

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}

def M : Set (Fin 5) := {0, 3}

def N : Set (Fin 5) := {0, 2, 4}

theorem set_operation_equality :
  M ∪ (Mᶜ ∩ N) = {0, 2, 3, 4} :=
by sorry

end set_operation_equality_l1052_105207


namespace max_value_of_f_l1052_105287

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 :=
by sorry

end max_value_of_f_l1052_105287


namespace polynomial_divisibility_l1052_105295

theorem polynomial_divisibility (a : ℤ) : 
  ∃ k : ℤ, (3*a + 5)^2 - 4 = (a + 1) * k := by
sorry

end polynomial_divisibility_l1052_105295


namespace hotel_cost_per_night_l1052_105264

theorem hotel_cost_per_night (nights : ℕ) (discount : ℕ) (total_paid : ℕ) (cost_per_night : ℕ) : 
  nights = 3 → 
  discount = 100 → 
  total_paid = 650 → 
  nights * cost_per_night - discount = total_paid → 
  cost_per_night = 250 := by
sorry

end hotel_cost_per_night_l1052_105264


namespace oil_demand_scientific_notation_l1052_105237

theorem oil_demand_scientific_notation :
  (735000000 : ℝ) = 7.35 * (10 ^ 8) := by
  sorry

end oil_demand_scientific_notation_l1052_105237


namespace trig_ratios_for_point_l1052_105267

theorem trig_ratios_for_point (m : ℝ) (α : ℝ) (h : m < 0) :
  let x : ℝ := 3 * m
  let y : ℝ := -2 * m
  let r : ℝ := Real.sqrt (x^2 + y^2)
  (x, y) = (3 * m, -2 * m) →
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧
  Real.cos α = -(3 * Real.sqrt 13 / 13) ∧
  Real.tan α = -2/3 :=
by
  sorry

end trig_ratios_for_point_l1052_105267


namespace regular_polygon_sides_l1052_105228

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 160 → n * (180 - interior_angle) = 360 → n = 18 := by
  sorry

end regular_polygon_sides_l1052_105228


namespace satisfactory_fraction_is_four_fifths_l1052_105245

/-- Represents the distribution of grades in a classroom --/
structure GradeDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  f : ℕ

/-- Calculates the fraction of satisfactory grades --/
def satisfactoryFraction (g : GradeDistribution) : ℚ :=
  let satisfactory := g.a + g.b + g.c + g.d
  let total := satisfactory + g.f
  satisfactory / total

/-- Theorem stating that for the given grade distribution, 
    the fraction of satisfactory grades is 4/5 --/
theorem satisfactory_fraction_is_four_fifths :
  let g : GradeDistribution := ⟨8, 7, 5, 4, 6⟩
  satisfactoryFraction g = 4/5 := by
  sorry

end satisfactory_fraction_is_four_fifths_l1052_105245


namespace potato_bag_weight_l1052_105258

/-- If a bag of potatoes weighs 12 lbs divided by half of its weight, then the weight of the bag is 24 lbs. -/
theorem potato_bag_weight (w : ℝ) (h : w = 12 / (w / 2)) : w = 24 :=
sorry

end potato_bag_weight_l1052_105258


namespace shaded_triangle_probability_l1052_105284

/-- Given a set of triangles, some of which are shaded, this theorem proves
    the probability of selecting a shaded triangle. -/
theorem shaded_triangle_probability
  (total_triangles : ℕ)
  (shaded_triangles : ℕ)
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
sorry

end shaded_triangle_probability_l1052_105284


namespace square_rectangle_area_relation_l1052_105205

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 2) * (x₁ + 5) = 3 * (x₁ - 3)^2 ∧
    (x₂ - 2) * (x₂ + 5) = 3 * (x₂ - 3)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 21/2 := by
  sorry

end square_rectangle_area_relation_l1052_105205


namespace mary_walking_distance_approx_l1052_105203

/-- Represents the journey Mary took to her sister's house -/
structure Journey where
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_portion : ℝ
  total_time : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  (1 - j.bike_portion) * j.total_distance

/-- The theorem stating that Mary's walking distance is approximately 0.3 km -/
theorem mary_walking_distance_approx (j : Journey) 
  (h1 : j.bike_speed = 15)
  (h2 : j.walk_speed = 4)
  (h3 : j.bike_portion = 0.4)
  (h4 : j.total_time = 0.6) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (walking_distance j - 0.3) < ε := by
  sorry

#check mary_walking_distance_approx

end mary_walking_distance_approx_l1052_105203


namespace students_correct_both_experiments_l1052_105283

/-- Given a group of students performing physics and chemistry experiments, 
    calculate the number of students who conducted both experiments correctly. -/
theorem students_correct_both_experiments 
  (total : ℕ) 
  (physics_correct : ℕ) 
  (chemistry_correct : ℕ) 
  (both_incorrect : ℕ) 
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 5) :
  physics_correct + chemistry_correct + both_incorrect - total = 26 := by
  sorry

#eval 40 + 31 + 5 - 50  -- Should output 26

end students_correct_both_experiments_l1052_105283


namespace equation_solution_l1052_105234

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ x : ℝ, (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔
  ∃ k : ℤ, x = π/4 + π * k :=
by sorry

end equation_solution_l1052_105234


namespace exactly_two_correct_propositions_l1052_105222

-- Define the basic geometric concepts
def Line : Type := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def supplementary_angle (α : ℝ) : ℝ := sorry
def adjacent_angle (l1 l2 : Line) (α : ℝ) : ℝ := sorry
def alternate_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def same_side_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def angle_bisector (l : Line) (α : ℝ) : Line := sorry
def complementary (α β : ℝ) : Prop := sorry

-- Define the four propositions
def proposition1 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, adjacent_angle l1 l2 α = adjacent_angle l1 l2 (supplementary_angle α) →
      perpendicular l1 l2

def proposition2 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, α = supplementary_angle α →
      perpendicular l1 l2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    alternate_interior_angles l1 l2 α β → α = β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

def proposition4 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    same_side_interior_angles l1 l2 α β → complementary α β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

-- The main theorem
theorem exactly_two_correct_propositions :
  (proposition1 = False ∧
   proposition2 = True ∧
   proposition3 = False ∧
   proposition4 = True) :=
sorry

end exactly_two_correct_propositions_l1052_105222


namespace jeff_bought_seven_one_yuan_socks_l1052_105209

/-- Represents the number of sock pairs at each price point -/
structure SockPurchase where
  one_yuan : ℕ
  three_yuan : ℕ
  four_yuan : ℕ

/-- Checks if a SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.one_yuan + p.three_yuan + p.four_yuan = 12 ∧
  p.one_yuan * 1 + p.three_yuan * 3 + p.four_yuan * 4 = 24 ∧
  p.one_yuan ≥ 1 ∧ p.three_yuan ≥ 1 ∧ p.four_yuan ≥ 1

/-- The main theorem stating that the only valid purchase has 7 pairs of 1-yuan socks -/
theorem jeff_bought_seven_one_yuan_socks :
  ∀ p : SockPurchase, is_valid_purchase p → p.one_yuan = 7 := by
  sorry

end jeff_bought_seven_one_yuan_socks_l1052_105209


namespace problem_statement_l1052_105281

theorem problem_statement (a b : ℝ) (h : |a + 2| + Real.sqrt (b - 4) = 0) : a / b = -1/2 := by
  sorry

end problem_statement_l1052_105281


namespace log_product_reciprocal_l1052_105206

theorem log_product_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  Real.log a / Real.log b * (Real.log b / Real.log a) = 1 :=
by sorry

end log_product_reciprocal_l1052_105206


namespace system_solution_l1052_105250

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 8 - 2*x)
  (eq2 : x + z = 10 - 2*y)
  (eq3 : x + y = 14 - 2*z) :
  2*x + 2*y + 2*z = 16 := by
sorry

end system_solution_l1052_105250


namespace geometric_sequence_ratio_l1052_105259

/-- Prove that in a geometric sequence with first term 1 and fourth term 64, the common ratio is 4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 = 1 →                     -- First term is 1
  a 4 = 64 →                    -- Fourth term is 64
  q = 4 := by
sorry

end geometric_sequence_ratio_l1052_105259


namespace franks_filled_boxes_l1052_105231

/-- Given that Frank had 13 boxes initially and 5 boxes are left unfilled,
    prove that the number of boxes he filled with toys is 8. -/
theorem franks_filled_boxes (total : ℕ) (unfilled : ℕ) (filled : ℕ) : 
  total = 13 → unfilled = 5 → filled = total - unfilled → filled = 8 := by sorry

end franks_filled_boxes_l1052_105231


namespace family_change_is_71_l1052_105225

/-- Represents a family member with their age and ticket price. -/
structure FamilyMember where
  age : ℕ
  ticketPrice : ℕ

/-- Calculates the change received after a family visit to an amusement park. -/
def amusementParkChange (family : List FamilyMember) (regularPrice discountAmount paidAmount : ℕ) : ℕ :=
  let totalCost := family.foldl (fun acc member => acc + member.ticketPrice) 0
  paidAmount - totalCost

/-- Theorem: The family receives $71 in change. -/
theorem family_change_is_71 :
  let family : List FamilyMember := [
    { age := 6, ticketPrice := 114 },
    { age := 10, ticketPrice := 114 },
    { age := 13, ticketPrice := 129 },
    { age := 8, ticketPrice := 114 },
    { age := 30, ticketPrice := 129 },  -- Assuming parent age
    { age := 30, ticketPrice := 129 }   -- Assuming parent age
  ]
  let regularPrice := 129
  let discountAmount := 15
  let paidAmount := 800
  amusementParkChange family regularPrice discountAmount paidAmount = 71 := by
sorry

end family_change_is_71_l1052_105225


namespace jimmy_garden_servings_l1052_105220

/-- The number of servings produced by a carrot plant -/
def carrot_servings : ℕ := 4

/-- The number of green bean plants -/
def green_bean_plants : ℕ := 10

/-- The number of carrot plants -/
def carrot_plants : ℕ := 8

/-- The number of corn plants -/
def corn_plants : ℕ := 12

/-- The number of tomato plants -/
def tomato_plants : ℕ := 15

/-- The number of servings produced by a corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- The number of servings produced by a green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- The number of servings produced by a tomato plant -/
def tomato_servings : ℕ := carrot_servings + 3

/-- The total number of servings in Jimmy's garden -/
def total_servings : ℕ :=
  green_bean_plants * green_bean_servings +
  carrot_plants * carrot_servings +
  corn_plants * corn_servings +
  tomato_plants * tomato_servings

theorem jimmy_garden_servings :
  total_servings = 477 := by sorry

end jimmy_garden_servings_l1052_105220


namespace logarithm_expression_equals_one_l1052_105241

-- Define the binary logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by
  sorry

end logarithm_expression_equals_one_l1052_105241


namespace problem_statement_l1052_105269

theorem problem_statement (a b c t : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (ht : t ≥ 1) 
  (sum_eq : a + b + c = 1/2) 
  (sqrt_eq : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 2 := by
  sorry

end problem_statement_l1052_105269


namespace wall_clock_interval_l1052_105230

/-- Represents a wall clock that rings at regular intervals -/
structure WallClock where
  rings_per_day : ℕ
  first_ring : ℕ
  hours_in_day : ℕ

/-- Calculates the interval between rings for a given wall clock -/
def ring_interval (clock : WallClock) : ℚ :=
  clock.hours_in_day / clock.rings_per_day

/-- Theorem: If a clock rings 8 times in a 24-hour day, starting at 1 A.M., 
    then the interval between each ring is 3 hours -/
theorem wall_clock_interval (clock : WallClock) 
    (h1 : clock.rings_per_day = 8) 
    (h2 : clock.first_ring = 1) 
    (h3 : clock.hours_in_day = 24) : 
    ring_interval clock = 3 := by
  sorry

end wall_clock_interval_l1052_105230


namespace cylinder_diagonal_angle_l1052_105240

theorem cylinder_diagonal_angle (m n : ℝ) (h : m > 0 ∧ n > 0) :
  let α := if m / n < Real.pi / 4 
           then 2 * Real.arctan (4 * m / (Real.pi * n))
           else 2 * Real.arctan (Real.pi * n / (4 * m))
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧ 
    (Real.pi * R^2) / (2 * R * H) = m / n ∧
    α = Real.arctan (2 * R / H) + Real.arctan (2 * R / H) :=
by sorry

end cylinder_diagonal_angle_l1052_105240


namespace problem_solution_l1052_105221

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 4 ≤ x ∧ x ≤ 3 * m + 2}

theorem problem_solution :
  (∀ m : ℝ, A ∪ B m = B m → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, A ∩ B m = B m → m < -3) := by
  sorry

end problem_solution_l1052_105221


namespace max_graduates_proof_l1052_105235

theorem max_graduates_proof (x : ℕ) : 
  x ≤ 210 ∧ 
  (49 + ((x - 50) / 8) * 7 : ℝ) / x > 0.9 ∧ 
  ∀ y : ℕ, y > 210 → (49 + ((y - 50) / 8) * 7 : ℝ) / y ≤ 0.9 := by
  sorry

end max_graduates_proof_l1052_105235


namespace corey_candies_l1052_105276

theorem corey_candies :
  let total_candies : ℝ := 66.5
  let tapanga_extra : ℝ := 8.25
  let corey_candies : ℝ := (total_candies - tapanga_extra) / 2
  corey_candies = 29.125 :=
by
  sorry

end corey_candies_l1052_105276


namespace hyperbola_focus_distance_l1052_105227

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 12 * x

-- Define the focus of the parabola
def parabola_focus (x : ℝ) : Prop :=
  x = 3

-- Define the point M on the hyperbola
def point_M (x y : ℝ) : Prop :=
  x = -3 ∧ y = Real.sqrt 6 / 2

-- Define the line F2M
def line_F2M (x y : ℝ) : Prop :=
  y = -Real.sqrt 6 / 12 * x + Real.sqrt 6 / 4

-- State the theorem
theorem hyperbola_focus_distance (a b x y : ℝ) :
  hyperbola a b x y →
  asymptote x y →
  parabola_focus a →
  point_M x y →
  line_F2M x y →
  (6 : ℝ) / 5 = abs (-Real.sqrt 6 / 12 * (-3) + Real.sqrt 6 / 4) / Real.sqrt (1 + 6 / 144) :=
by sorry

end hyperbola_focus_distance_l1052_105227


namespace money_duration_l1052_105239

def mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end money_duration_l1052_105239


namespace min_value_quadratic_l1052_105212

theorem min_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 18 * x + 7
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = -20) := by
  sorry

end min_value_quadratic_l1052_105212


namespace composite_sum_product_l1052_105255

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8
def fourth_composite : ℕ := 9
def fifth_composite : ℕ := 10

theorem composite_sum_product : 
  (first_composite * second_composite * third_composite) + 
  (fourth_composite * fifth_composite) = 282 := by
sorry

end composite_sum_product_l1052_105255


namespace parabola_tangent_theorem_l1052_105215

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line that point A is on
def line_A (x y : ℝ) : Prop := x - 2*y + 13 = 0

-- Define that A is not on the y-axis
def A_not_on_y_axis (x y : ℝ) : Prop := x ≠ 0

-- Define points M and N as tangent points on the parabola
def M_N_tangent_points (xm ym xn yn : ℝ) : Prop :=
  parabola xm ym ∧ parabola xn yn

-- Define B and C as intersection points of AM and AN with y-axis
def B_C_intersection_points (xb yb xc yc : ℝ) : Prop :=
  xb = 0 ∧ xc = 0

-- Theorem statement
theorem parabola_tangent_theorem
  (xa ya xm ym xn yn xb yb xc yc : ℝ)
  (h1 : line_A xa ya)
  (h2 : A_not_on_y_axis xa ya)
  (h3 : M_N_tangent_points xm ym xn yn)
  (h4 : B_C_intersection_points xb yb xc yc) :
  -- 1. Line MN passes through (13, 8)
  ∃ (t : ℝ), xm + t * (xn - xm) = 13 ∧ ym + t * (yn - ym) = 8 ∧
  -- 2. Circumcircle of ABC passes through (2, 0)
  (xa - 2)^2 + ya^2 = (xb - 2)^2 + yb^2 ∧ (xa - 2)^2 + ya^2 = (xc - 2)^2 + yc^2 ∧
  -- 3. Minimum radius of circumcircle is (3√5)/2
  ∃ (r : ℝ), r ≥ (3 * Real.sqrt 5) / 2 ∧
    (xa - 2)^2 + ya^2 = 4 * r^2 ∧ (xb - 2)^2 + yb^2 = 4 * r^2 ∧ (xc - 2)^2 + yc^2 = 4 * r^2 :=
by sorry

end parabola_tangent_theorem_l1052_105215


namespace problem_solution_l1052_105270

theorem problem_solution :
  (195 * 205 = 39975) ∧
  (9 * 11 * 101 * 10001 = 99999999) ∧
  (∀ a : ℝ, a^2 - 6*a + 8 = (a - 2)*(a - 4)) :=
by sorry

end problem_solution_l1052_105270


namespace intersection_of_symmetric_lines_l1052_105247

/-- Two lines that are symmetric about the x-axis -/
structure SymmetricLines where
  k : ℝ
  b : ℝ
  l₁ : ℝ → ℝ := fun x ↦ k * x + 2
  l₂ : ℝ → ℝ := fun x ↦ -x + b
  symmetric : l₁ 0 = -l₂ 0

/-- The intersection point of two symmetric lines is (-2, 0) -/
theorem intersection_of_symmetric_lines (lines : SymmetricLines) :
  ∃ x y, lines.l₁ x = lines.l₂ x ∧ x = -2 ∧ y = 0 := by
  sorry

end intersection_of_symmetric_lines_l1052_105247


namespace dartboard_angle_l1052_105243

/-- Given a circular dartboard, if the probability of a dart landing in a particular region is 1/4,
    then the measure of the central angle of that region is 90 degrees. -/
theorem dartboard_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end dartboard_angle_l1052_105243


namespace subsets_with_even_l1052_105201

def S : Finset Nat := {1, 2, 3, 4}

theorem subsets_with_even (A : Finset (Finset Nat)) : 
  A = {s : Finset Nat | s ⊆ S ∧ ∃ n ∈ s, Even n} → Finset.card A = 12 := by
  sorry

end subsets_with_even_l1052_105201


namespace factorial_divisibility_l1052_105253

theorem factorial_divisibility (n : ℕ) : 
  (∃ (p q : ℕ), p ≤ n ∧ q ≤ n ∧ n + 2 = p * q) ∨ 
  (∃ (p : ℕ), p ≥ 3 ∧ Prime p ∧ n + 2 = p^2) ↔ 
  (n + 2) ∣ n! :=
sorry

end factorial_divisibility_l1052_105253
