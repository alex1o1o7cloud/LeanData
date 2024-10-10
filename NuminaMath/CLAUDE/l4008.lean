import Mathlib

namespace monthly_profit_calculation_l4008_400890

/-- Calculates the monthly profit for John's computer assembly business --/
theorem monthly_profit_calculation (cost_per_computer : ℝ) (markup : ℝ) 
  (computers_per_month : ℕ) (monthly_rent : ℝ) (monthly_non_rent_expenses : ℝ) :
  cost_per_computer = 800 →
  markup = 1.4 →
  computers_per_month = 60 →
  monthly_rent = 5000 →
  monthly_non_rent_expenses = 3000 →
  let selling_price := cost_per_computer * markup
  let total_revenue := selling_price * computers_per_month
  let total_component_cost := cost_per_computer * computers_per_month
  let total_expenses := monthly_rent + monthly_non_rent_expenses
  let profit := total_revenue - total_component_cost - total_expenses
  profit = 11200 := by
sorry

end monthly_profit_calculation_l4008_400890


namespace parabola_directrix_l4008_400800

/-- Given a parabola y = ax^2 with directrix y = -2, prove that a = 1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 ∧ y = -2 → a = 1/8) :=
by sorry

end parabola_directrix_l4008_400800


namespace f_has_two_zeros_l4008_400884

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else x^2 - 1

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end f_has_two_zeros_l4008_400884


namespace parabola_directrix_l4008_400817

/-- The equation of the directrix of the parabola y² = 8x is x = -2 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, y^2 = 8*x → ∃ p, p = 4 ∧ x = -p/2) → 
  ∃ k, k = -2 ∧ (∀ x y, y^2 = 8*x → x = k) :=
sorry

end parabola_directrix_l4008_400817


namespace water_speed_calculation_l4008_400829

/-- Given a person who can swim in still water at 10 km/h and takes 2 hours to swim 12 km against
    the current, prove that the speed of the water is 4 km/h. -/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 10 →
  distance = 12 →
  time = 2 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 4 := by
  sorry

end water_speed_calculation_l4008_400829


namespace train_speed_crossing_bridge_l4008_400893

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 150) 
  (h3 : crossing_time = 20) : 
  (train_length + bridge_length) / crossing_time = 20 := by
  sorry

end train_speed_crossing_bridge_l4008_400893


namespace oddFactorsOf252_eq_six_l4008_400862

/-- The number of odd factors of 252 -/
def oddFactorsOf252 : ℕ :=
  let n : ℕ := 252
  let primeFactors : List (ℕ × ℕ) := [(3, 2), (7, 1)]  -- List of (prime, exponent) pairs for odd primes
  (primeFactors.map (fun (p, e) => e + 1)).prod

/-- Theorem: The number of odd factors of 252 is 6 -/
theorem oddFactorsOf252_eq_six : oddFactorsOf252 = 6 := by
  sorry

end oddFactorsOf252_eq_six_l4008_400862


namespace square_perimeter_equals_area_l4008_400889

theorem square_perimeter_equals_area (x : ℝ) (h : x > 0) :
  4 * x = x^2 → x = 4 := by
  sorry

end square_perimeter_equals_area_l4008_400889


namespace three_digit_number_operation_l4008_400845

theorem three_digit_number_operation (a b c : ℕ) : 
  a = c - 3 → 
  0 ≤ a ∧ a < 10 → 
  0 ≤ b ∧ b < 10 → 
  0 ≤ c ∧ c < 10 → 
  (2 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 1 :=
by sorry

end three_digit_number_operation_l4008_400845


namespace complement_B_intersect_A_range_of_a_l4008_400870

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 18 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 / (x + 1) ≤ -1}

-- Define the complement of B
def complement_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (complement_B ∩ A) = {x : ℝ | (-6 ≤ x ∧ x < -2) ∨ (-1 ≤ x ∧ x ≤ 3)} :=
sorry

-- Define set C
def C (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (B ∪ C a = B) ↔ (a ≥ 1) :=
sorry

end complement_B_intersect_A_range_of_a_l4008_400870


namespace inequality_proof_l4008_400846

theorem inequality_proof (x y : ℝ) (h : x > y) : 
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
sorry

end inequality_proof_l4008_400846


namespace perpendicular_line_equation_l4008_400816

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- The theorem to prove
theorem perpendicular_line_equation :
  ∃ (x y : ℝ), intersection_point x y ∧
  perpendicular 2 3 5 2 3 (-7) ∧
  (2 * x + 3 * y - 7 = 0) :=
sorry

end perpendicular_line_equation_l4008_400816


namespace line_increase_theorem_l4008_400803

/-- Represents a line in a Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The increase in y for a given increase in x -/
def y_increase (l : Line) (x_increase : ℝ) : ℝ :=
  l.slope * x_increase

/-- Theorem: For a line with the given properties, an increase of 20 units in x
    from the point (1, 2) results in an increase of 41.8 units in y -/
theorem line_increase_theorem (l : Line) 
    (h1 : l.slope = 11 / 5)
    (h2 : 2 = l.slope * 1 + l.y_intercept) : 
    y_increase l 20 = 41.8 := by
  sorry

end line_increase_theorem_l4008_400803


namespace min_value_and_range_l4008_400855

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

def e : ℝ := Real.exp 1

theorem min_value_and_range (t : ℝ) (h : t > 0) :
  (∃ (x : ℝ), x ∈ Set.Icc t (t + 2) ∧
    (∀ (y : ℝ), y ∈ Set.Icc t (t + 2) → f x ≤ f y) ∧
    ((0 < t ∧ t < 1/e → f x = -1/e) ∧
     (t ≥ 1/e → f x = t * Real.log t))) ∧
  (∀ (a : ℝ), (∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1/e) e ∧ 2 * f x₀ ≥ g a x₀) →
    a ≤ -2 + 1/e + 3*e) :=
sorry

end min_value_and_range_l4008_400855


namespace solve_equation_l4008_400835

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 := by
  sorry

end solve_equation_l4008_400835


namespace stream_speed_stream_speed_is_one_l4008_400882

/-- Given a man's swimming speed and the relative time to swim upstream vs downstream, 
    calculate the speed of the stream. -/
theorem stream_speed (mans_speed : ℝ) (upstream_time_ratio : ℝ) : ℝ :=
  let stream_speed := (mans_speed * (upstream_time_ratio - 1)) / (upstream_time_ratio + 1)
  stream_speed

/-- Prove that given the conditions, the stream speed is 1 km/h -/
theorem stream_speed_is_one :
  stream_speed 3 2 = 1 := by
  sorry

end stream_speed_stream_speed_is_one_l4008_400882


namespace jacobs_flock_total_l4008_400811

/-- Represents the composition of Jacob's flock -/
structure Flock where
  goats : ℕ
  sheep : ℕ

/-- Theorem stating the total number of animals in Jacob's flock -/
theorem jacobs_flock_total (f : Flock) 
  (h1 : f.goats = f.sheep / 2)  -- One third of animals are goats, so goats = (sheep + goats) / 3 = sheep / 2
  (h2 : f.sheep = f.goats + 12) -- There are 12 more sheep than goats
  : f.goats + f.sheep = 36 := by
  sorry


end jacobs_flock_total_l4008_400811


namespace expression_value_l4008_400888

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by
  sorry

end expression_value_l4008_400888


namespace sales_tax_difference_l4008_400874

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def tax_rate_high : ℝ := 0.075
def tax_rate_low : ℝ := 0.07

-- Define the sales tax calculation function
def sales_tax (price : ℝ) (rate : ℝ) : ℝ := price * rate

-- Theorem statement
theorem sales_tax_difference :
  sales_tax item_price tax_rate_high - sales_tax item_price tax_rate_low = 0.25 := by
  sorry


end sales_tax_difference_l4008_400874


namespace prime_sum_floor_squared_l4008_400892

theorem prime_sum_floor_squared : ∃! (p₁ p₂ : ℕ), 
  Prime p₁ ∧ Prime p₂ ∧ p₁ ≠ p₂ ∧
  (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val^2 : ℚ) / 5⌋) ∧
  (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val^2 : ℚ) / 5⌋) ∧
  p₁ + p₂ = 52 := by
sorry

end prime_sum_floor_squared_l4008_400892


namespace complex_modulus_equation_l4008_400876

theorem complex_modulus_equation (m : ℝ) (h1 : m > 0) :
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end complex_modulus_equation_l4008_400876


namespace prime_power_of_two_l4008_400838

theorem prime_power_of_two (n : ℕ) : 
  Prime (2^n + 1) → ∃ k : ℕ, n = 2^k :=
by sorry

end prime_power_of_two_l4008_400838


namespace estimate_larger_than_actual_l4008_400825

theorem estimate_larger_than_actual (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  ⌈x⌉ - ⌊y⌋ > x - y :=
sorry

end estimate_larger_than_actual_l4008_400825


namespace bike_rental_cost_l4008_400887

theorem bike_rental_cost 
  (daily_rate : ℝ) 
  (mileage_rate : ℝ) 
  (rental_days : ℕ) 
  (miles_biked : ℝ) 
  (h1 : daily_rate = 15)
  (h2 : mileage_rate = 0.1)
  (h3 : rental_days = 3)
  (h4 : miles_biked = 300) :
  daily_rate * ↑rental_days + mileage_rate * miles_biked = 75 :=
by sorry

end bike_rental_cost_l4008_400887


namespace roots_sum_l4008_400878

theorem roots_sum (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3/2 := by
sorry

end roots_sum_l4008_400878


namespace not_certain_rain_beijing_no_rain_shanghai_l4008_400830

-- Define the probabilities of rainfall
def probability_rain_beijing : ℝ := 0.8
def probability_rain_shanghai : ℝ := 0.2

-- Theorem to prove
theorem not_certain_rain_beijing_no_rain_shanghai :
  ¬(probability_rain_beijing = 1 ∧ probability_rain_shanghai = 0) :=
sorry

end not_certain_rain_beijing_no_rain_shanghai_l4008_400830


namespace quadratic_equation_roots_l4008_400831

theorem quadratic_equation_roots (x : ℝ) :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x = r₁ ∨ x = r₂) ↔ x^2 - 2*x - 6 = 0 :=
by sorry

end quadratic_equation_roots_l4008_400831


namespace inequality_problem_l4008_400818

/-- Given an inequality and its solution set, prove the values of a and b and solve another inequality -/
theorem inequality_problem (a b c : ℝ) : 
  (∀ x, (a * x^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) →
  c > 2 →
  (a = 1 ∧ b = 2) ∧
  (∀ x, (a * x^2 - (a*c + b)*x + b*c < 0) ↔ (2 < x ∧ x < c)) :=
by sorry

end inequality_problem_l4008_400818


namespace factoring_expression_l4008_400895

theorem factoring_expression (x : ℝ) : x * (x + 4) + 2 * (x + 4) + (x + 4) = (x + 3) * (x + 4) := by
  sorry

end factoring_expression_l4008_400895


namespace line_equation_l4008_400850

/-- Given a line passing through (a, 0) and cutting a triangular region
    with area T in the first quadrant, prove its equation. -/
theorem line_equation (a T : ℝ) (h₁ : a ≠ 0) (h₂ : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0) →
    (y = m * x + b ↔ a^2 * y + 2 * T * x - 2 * a * T = 0) :=
sorry

end line_equation_l4008_400850


namespace equation_solution_l4008_400840

theorem equation_solution : ∃ y : ℝ, y > 0 ∧ 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4) ∧ y = 1296 := by
  sorry

end equation_solution_l4008_400840


namespace fraction_equality_l4008_400886

theorem fraction_equality : (5 * 7 + 3) / (3 * 5) = 38 / 15 := by
  sorry

end fraction_equality_l4008_400886


namespace unique_solution_implies_negative_a_l4008_400891

theorem unique_solution_implies_negative_a (a : ℝ) :
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) → a < 0 := by sorry

end unique_solution_implies_negative_a_l4008_400891


namespace q_div_p_equals_225_l4008_400864

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/- Probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / choose total_cards cards_drawn

/- Probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / choose total_cards cards_drawn

/- Theorem stating the ratio of q to p -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end q_div_p_equals_225_l4008_400864


namespace probability_different_with_three_l4008_400827

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (different numbers with one being 3) -/
def favorableOutcomes : ℕ := 2 * (numFaces - 1)

/-- The probability of getting different numbers on two fair dice with one showing 3 -/
def probabilityDifferentWithThree : ℚ := favorableOutcomes / totalOutcomes

theorem probability_different_with_three :
  probabilityDifferentWithThree = 5 / 18 := by
  sorry

end probability_different_with_three_l4008_400827


namespace sin_210_degrees_l4008_400826

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l4008_400826


namespace water_consumption_l4008_400832

theorem water_consumption (initial_water : ℝ) : 
  initial_water > 0 →
  let remaining_day1 := initial_water * (1 - 7/15)
  let remaining_day2 := remaining_day1 * (1 - 5/8)
  let remaining_day3 := remaining_day2 * (1 - 2/3)
  remaining_day3 = 2.6 →
  initial_water = 39 := by
sorry

end water_consumption_l4008_400832


namespace integer_root_of_cubic_l4008_400867

-- Define the polynomial
def cubic_polynomial (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x + q

-- State the theorem
theorem integer_root_of_cubic (p q : ℚ) : 
  (∃ (n : ℤ), cubic_polynomial p q n = 0) →
  (cubic_polynomial p q (3 - Real.sqrt 5) = 0) →
  (∃ (n : ℤ), cubic_polynomial p q n = 0 ∧ n = -6) :=
by sorry

end integer_root_of_cubic_l4008_400867


namespace percentage_problem_l4008_400808

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 150 → 
  P * x = 0.20 * 487.50 → 
  P = 0.65 := by
sorry

end percentage_problem_l4008_400808


namespace absolute_curve_sufficient_not_necessary_l4008_400880

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the property of being on the curve y = |x|
def onAbsoluteCurve (p : Point2D) : Prop :=
  p.y = |p.x|

-- Define the property of equal distance to both axes
def equalDistanceToAxes (p : Point2D) : Prop :=
  |p.x| = |p.y|

-- Theorem statement
theorem absolute_curve_sufficient_not_necessary :
  (∀ p : Point2D, onAbsoluteCurve p → equalDistanceToAxes p) ∧
  (∃ p : Point2D, equalDistanceToAxes p ∧ ¬onAbsoluteCurve p) :=
sorry

end absolute_curve_sufficient_not_necessary_l4008_400880


namespace product_of_special_n_values_l4008_400837

theorem product_of_special_n_values : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) ∧ 
  (∀ n : ℕ, (∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) → n ∈ S) ∧
  S.card > 0 ∧
  (S.prod id = 396) := by
sorry

end product_of_special_n_values_l4008_400837


namespace muffins_per_box_l4008_400861

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) (muffins_per_box : ℕ) : 
  total_muffins = 96 →
  num_boxes = 8 →
  total_muffins = num_boxes * muffins_per_box →
  muffins_per_box = 12 := by
  sorry

end muffins_per_box_l4008_400861


namespace smallest_integer_satisfying_congruences_l4008_400847

theorem smallest_integer_satisfying_congruences : ∃ b : ℕ, b > 0 ∧
  b % 3 = 2 ∧
  b % 4 = 3 ∧
  b % 5 = 4 ∧
  b % 6 = 5 ∧
  ∀ k : ℕ, k > 0 ∧ k % 3 = 2 ∧ k % 4 = 3 ∧ k % 5 = 4 ∧ k % 6 = 5 → k ≥ b :=
by
  -- Proof goes here
  sorry

#eval 59 % 3  -- Should output 2
#eval 59 % 4  -- Should output 3
#eval 59 % 5  -- Should output 4
#eval 59 % 6  -- Should output 5

end smallest_integer_satisfying_congruences_l4008_400847


namespace min_value_sum_reciprocals_l4008_400848

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 5) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 5 → 
    9/x + 16/y + 25/z ≤ 9/a + 16/b + 25/c) ∧
  9/x + 16/y + 25/z = 28.8 := by
sorry

end min_value_sum_reciprocals_l4008_400848


namespace number_is_nine_l4008_400899

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def xiao_qian_statement (n : ℕ) : Prop := is_perfect_square n ∧ n < 5

def xiao_lu_statement (n : ℕ) : Prop := n < 7 ∧ n ≥ 10

def xiao_dai_statement (n : ℕ) : Prop := is_perfect_square n ∧ n ≥ 5

def one_all_true (n : ℕ) : Prop :=
  (xiao_qian_statement n) ∨ (xiao_lu_statement n) ∨ (xiao_dai_statement n)

def one_all_false (n : ℕ) : Prop :=
  (¬xiao_qian_statement n) ∨ (¬xiao_lu_statement n) ∨ (¬xiao_dai_statement n)

def one_true_one_false (n : ℕ) : Prop :=
  (is_perfect_square n ∧ ¬(n < 5)) ∨
  ((n < 7) ∧ ¬(n ≥ 10)) ∨
  ((is_perfect_square n) ∧ ¬(n ≥ 5))

theorem number_is_nine :
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 99 ∧ one_all_true n ∧ one_all_false n ∧ one_true_one_false n ∧ n = 9 :=
by sorry

end number_is_nine_l4008_400899


namespace least_cube_divisible_by_17280_l4008_400821

theorem least_cube_divisible_by_17280 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(17280 ∣ y^3)) ∧ (17280 ∣ x^3) ↔ x = 120 := by
  sorry

end least_cube_divisible_by_17280_l4008_400821


namespace least_xy_value_l4008_400842

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 9 → (a : ℕ) * b ≥ (x : ℕ) * y) ∧
  (x : ℕ) * y = 108 :=
sorry

end least_xy_value_l4008_400842


namespace more_non_products_than_products_l4008_400860

/-- The number of ten-digit numbers -/
def ten_digit_count : ℕ := 9 * 10^9

/-- The number of five-digit numbers -/
def five_digit_count : ℕ := 90000

/-- The estimated number of products of two five-digit numbers that are ten-digit numbers -/
def ten_digit_products : ℕ := (five_digit_count * (five_digit_count - 1) / 2 + five_digit_count) / 2

theorem more_non_products_than_products : ten_digit_count - ten_digit_products > ten_digit_products := by
  sorry

end more_non_products_than_products_l4008_400860


namespace solve_equation_l4008_400856

theorem solve_equation (x : ℝ) : 4 * (x - 1) - 5 * (1 + x) = 3 ↔ x = -12 := by
  sorry

end solve_equation_l4008_400856


namespace floor_painting_rate_l4008_400879

/-- Proves that the painting rate for a rectangular floor is 5 Rs/sq m given specific conditions -/
theorem floor_painting_rate (length : ℝ) (total_cost : ℝ) : 
  length = 13.416407864998739 →
  total_cost = 300 →
  ∃ (breadth : ℝ), 
    length = 3 * breadth ∧ 
    (5 : ℝ) = total_cost / (length * breadth) := by
  sorry

end floor_painting_rate_l4008_400879


namespace cubic_roots_coefficients_relation_l4008_400809

theorem cubic_roots_coefficients_relation 
  (a b c d : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : a ≠ 0) 
  (h_roots : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) : 
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
sorry

end cubic_roots_coefficients_relation_l4008_400809


namespace study_time_problem_l4008_400820

/-- The study time problem -/
theorem study_time_problem 
  (kwame_time : ℝ) 
  (lexia_time : ℝ) 
  (h1 : kwame_time = 2.5)
  (h2 : lexia_time = 97 / 60)
  (h3 : kwame_time + connor_time = lexia_time + 143 / 60) :
  connor_time = 1.5 := by
  sorry

end study_time_problem_l4008_400820


namespace square_perimeter_l4008_400815

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ strip_perimeter : ℝ, 
    strip_perimeter = 2 * (s + s / 4) ∧ 
    strip_perimeter = 40) →
  4 * s = 64 := by
sorry

end square_perimeter_l4008_400815


namespace car_speed_problem_l4008_400868

/-- Proves that Car B's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (speed_A speed_B initial_distance overtake_time final_distance : ℝ) :
  speed_A = 58 ∧ 
  initial_distance = 16 ∧ 
  overtake_time = 3 ∧ 
  final_distance = 8 ∧
  speed_A * overtake_time = speed_B * overtake_time + initial_distance + final_distance →
  speed_B = 50 := by
sorry

end car_speed_problem_l4008_400868


namespace sequence_general_term_l4008_400894

theorem sequence_general_term (n : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ k, S k = 2 * k^2 - 3 * k) →
  (∀ k, k ≥ 1 → a k = S k - S (k - 1)) →
  (∀ k, a k = 4 * k - 5) :=
by sorry

end sequence_general_term_l4008_400894


namespace pencil_cost_l4008_400853

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 286)
  (eq2 : 3 * x + 4 * y = 204) :
  y = 12 := by
  sorry

end pencil_cost_l4008_400853


namespace negation_at_most_one_obtuse_angle_l4008_400804

/-- Definition of a triangle -/
def Triangle : Type := Unit

/-- Definition of an obtuse angle in a triangle -/
def HasObtuseAngle (t : Triangle) : Prop := sorry

/-- Statement: There is at most one obtuse angle in a triangle -/
def AtMostOneObtuseAngle : Prop :=
  ∀ t : Triangle, ∃! a : ℕ, a ≤ 3 ∧ HasObtuseAngle t

/-- Theorem: The negation of "There is at most one obtuse angle in a triangle"
    is equivalent to "There are at least two obtuse angles." -/
theorem negation_at_most_one_obtuse_angle :
  ¬AtMostOneObtuseAngle ↔ ∃ t : Triangle, ∃ a b : ℕ, a ≠ b ∧ a ≤ 3 ∧ b ≤ 3 ∧ HasObtuseAngle t ∧ HasObtuseAngle t :=
by sorry

end negation_at_most_one_obtuse_angle_l4008_400804


namespace only_A_has_zero_constant_term_l4008_400885

def equation_A (x : ℝ) : ℝ := x^2 + x
def equation_B (x : ℝ) : ℝ := 2*x^2 - x - 12
def equation_C (x : ℝ) : ℝ := 2*(x^2 - 1) - 3*(x - 1)
def equation_D (x : ℝ) : ℝ := 2*(x^2 + 1) - (x + 4)

def has_zero_constant_term (f : ℝ → ℝ) : Prop := f 0 = 0

theorem only_A_has_zero_constant_term :
  has_zero_constant_term equation_A ∧
  ¬has_zero_constant_term equation_B ∧
  ¬has_zero_constant_term equation_C ∧
  ¬has_zero_constant_term equation_D :=
by sorry

end only_A_has_zero_constant_term_l4008_400885


namespace phi_equals_theta_is_plane_l4008_400897

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A generalized plane in 3D space -/
structure GeneralizedPlane where
  equation : SphericalCoord → Prop

/-- The specific equation φ = θ -/
def phiEqualsThetaPlane : GeneralizedPlane where
  equation := fun coord => coord.φ = coord.θ

/-- Theorem: The equation φ = θ in spherical coordinates describes a generalized plane -/
theorem phi_equals_theta_is_plane : 
  ∃ (p : GeneralizedPlane), p = phiEqualsThetaPlane :=
sorry

end phi_equals_theta_is_plane_l4008_400897


namespace smallest_solution_congruence_l4008_400824

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l4008_400824


namespace overall_loss_percentage_l4008_400849

/-- Calculate the overall loss percentage on three articles with given purchase prices, exchange rates, shipping fees, selling prices, and sales tax. -/
theorem overall_loss_percentage
  (purchase_a purchase_b purchase_c : ℝ)
  (exchange_eur exchange_gbp : ℝ)
  (shipping_fee : ℝ)
  (sell_a sell_b sell_c : ℝ)
  (sales_tax_rate : ℝ)
  (h_purchase_a : purchase_a = 100)
  (h_purchase_b : purchase_b = 200)
  (h_purchase_c : purchase_c = 300)
  (h_exchange_eur : exchange_eur = 1.1)
  (h_exchange_gbp : exchange_gbp = 1.3)
  (h_shipping_fee : shipping_fee = 10)
  (h_sell_a : sell_a = 110)
  (h_sell_b : sell_b = 250)
  (h_sell_c : sell_c = 330)
  (h_sales_tax_rate : sales_tax_rate = 0.05) :
  ∃ (loss_percentage : ℝ), 
    abs (loss_percentage - 0.0209) < 0.0001 ∧
    loss_percentage = 
      (((sell_a + sell_b + sell_c) * (1 + sales_tax_rate) - 
        (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) / 
       (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) * (-100) :=
by sorry

end overall_loss_percentage_l4008_400849


namespace arrangement_theorem_l4008_400810

def number_of_arrangements (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let group_of_four_two_men := Nat.choose num_men 2 * Nat.choose num_women 2
  let group_of_four_one_man := Nat.choose num_men 1 * Nat.choose num_women 3
  group_of_four_two_men + group_of_four_one_man

theorem arrangement_theorem :
  number_of_arrangements 5 4 = 80 :=
by sorry

end arrangement_theorem_l4008_400810


namespace solution_set_is_two_l4008_400857

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := log10 (2 * x + 1) + log10 x = 1

-- Theorem statement
theorem solution_set_is_two :
  ∃! x : ℝ, x > 0 ∧ 2 * x + 1 > 0 ∧ equation x := by sorry

end solution_set_is_two_l4008_400857


namespace circumradius_of_special_triangle_l4008_400836

/-- The radius of the circumcircle of a triangle with sides 8, 15, and 17 is 17/2 -/
theorem circumradius_of_special_triangle :
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (area * 4) / (a * b * c) = 2 / 17 := by
  sorry

end circumradius_of_special_triangle_l4008_400836


namespace quotient_minus_fraction_number_plus_half_l4008_400869

-- Question 1
theorem quotient_minus_fraction (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (c / a) / (c / b) - c = c * (b - a) / a / b :=
sorry

-- Question 2
theorem number_plus_half (x : ℚ) :
  x + (1/2) * x = 12/5 ↔ x = (12/5 - 1/2) / (3/2) :=
sorry

end quotient_minus_fraction_number_plus_half_l4008_400869


namespace angle_properties_l4008_400875

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, -4/5), prove properties about α and β. -/
theorem angle_properties (α β : Real) : 
  (∃ (P : Real × Real), P.1 = -3/5 ∧ P.2 = -4/5 ∧ 
   Real.cos α = -3/5 ∧ Real.sin α = -4/5) →
  Real.sin (α + π) = 4/5 ∧
  (Real.sin (α + β) = 5/13 → Real.cos β = -56/65 ∨ Real.cos β = 16/65) := by
  sorry

end angle_properties_l4008_400875


namespace ceiling_floor_sum_l4008_400822

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end ceiling_floor_sum_l4008_400822


namespace sqrt_plus_square_zero_implies_diff_l4008_400843

theorem sqrt_plus_square_zero_implies_diff (x y : ℝ) :
  Real.sqrt (y - 3) + (2 * x - 4)^2 = 0 → 2 * x - y = 1 := by
  sorry

end sqrt_plus_square_zero_implies_diff_l4008_400843


namespace cement_mixture_weight_l4008_400841

theorem cement_mixture_weight (sand water gravel cement limestone total : ℚ) : 
  sand = 2/9 →
  water = 5/18 →
  gravel = 1/6 →
  cement = 7/36 →
  limestone = 1 - (sand + water + gravel + cement) →
  limestone * total = 12 →
  total = 86.4 := by
sorry

end cement_mixture_weight_l4008_400841


namespace license_advantages_18_vs_30_l4008_400833

/-- Represents the age at which a person gets a driver's license -/
inductive LicenseAge
| Age18 : LicenseAge
| Age30 : LicenseAge

/-- Represents the advantages of getting a driver's license -/
structure LicenseAdvantages where
  insuranceCostSavings : Bool
  rentalCarFlexibility : Bool
  employmentOpportunities : Bool

/-- Theorem stating that getting a license at 18 has more advantages than at 30 -/
theorem license_advantages_18_vs_30 :
  ∃ (adv18 adv30 : LicenseAdvantages),
    (adv18.insuranceCostSavings = true ∧
     adv18.rentalCarFlexibility = true ∧
     adv18.employmentOpportunities = true) ∧
    (adv30.insuranceCostSavings = false ∨
     adv30.rentalCarFlexibility = false ∨
     adv30.employmentOpportunities = false) :=
by sorry

end license_advantages_18_vs_30_l4008_400833


namespace bus_max_capacity_l4008_400896

/-- Represents the seating capacity of a bus with specific arrangements -/
structure BusCapacity where
  left_regular : Nat
  left_priority : Nat
  right_regular : Nat
  right_priority : Nat
  back_row : Nat
  standing : Nat
  regular_capacity : Nat
  priority_capacity : Nat

/-- Calculates the total capacity of the bus -/
def total_capacity (bus : BusCapacity) : Nat :=
  bus.left_regular * bus.regular_capacity +
  bus.left_priority * bus.priority_capacity +
  bus.right_regular * bus.regular_capacity +
  bus.right_priority * bus.priority_capacity +
  bus.back_row +
  bus.standing

/-- Theorem stating that the maximum capacity of the bus is 94 -/
theorem bus_max_capacity :
  ∀ (bus : BusCapacity),
    bus.left_regular = 12 →
    bus.left_priority = 3 →
    bus.right_regular = 9 →
    bus.right_priority = 2 →
    bus.back_row = 7 →
    bus.standing = 14 →
    bus.regular_capacity = 3 →
    bus.priority_capacity = 2 →
    total_capacity bus = 94 := by
  sorry


end bus_max_capacity_l4008_400896


namespace four_integers_sum_problem_l4008_400823

theorem four_integers_sum_problem :
  ∀ a b c d : ℕ,
    0 < a ∧ a < b ∧ b < c ∧ c < d →
    a + b + c = 6 →
    a + b + d = 7 →
    a + c + d = 8 →
    b + c + d = 9 →
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by sorry

end four_integers_sum_problem_l4008_400823


namespace total_cost_is_649_70_l4008_400881

/-- Calculates the total cost of a guitar and amplifier in dollars --/
def total_cost_in_dollars (guitar_price : ℝ) (amplifier_price : ℝ) 
  (guitar_discount : ℝ) (amplifier_discount : ℝ) (vat_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_guitar := guitar_price * (1 - guitar_discount)
  let discounted_amplifier := amplifier_price * (1 - amplifier_discount)
  let total_with_vat := (discounted_guitar + discounted_amplifier) * (1 + vat_rate)
  total_with_vat * exchange_rate

/-- Theorem stating that the total cost is equal to $649.70 --/
theorem total_cost_is_649_70 :
  total_cost_in_dollars 330 220 0.10 0.05 0.07 1.20 = 649.70 := by
  sorry

end total_cost_is_649_70_l4008_400881


namespace remainder_sum_l4008_400814

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 72)
  (hd : d % 120 = 112) :
  (c + d) % 40 = 24 := by
sorry

end remainder_sum_l4008_400814


namespace james_played_five_rounds_l4008_400858

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  pointsPerCorrectAnswer : ℕ
  questionsPerRound : ℕ
  bonusPoints : ℕ
  totalPoints : ℕ
  missedQuestions : ℕ

/-- Calculates the number of rounds played given the quiz bowl parameters -/
def calculateRounds (qb : QuizBowl) : ℕ :=
  sorry

/-- Theorem stating that James played 5 rounds -/
theorem james_played_five_rounds (qb : QuizBowl) 
  (h1 : qb.pointsPerCorrectAnswer = 2)
  (h2 : qb.questionsPerRound = 5)
  (h3 : qb.bonusPoints = 4)
  (h4 : qb.totalPoints = 66)
  (h5 : qb.missedQuestions = 1) :
  calculateRounds qb = 5 := by
  sorry

end james_played_five_rounds_l4008_400858


namespace range_of_m_l4008_400863

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - m)^x < (3 - m)^y

-- Define the theorem
theorem range_of_m : 
  ∃ m_min m_max : ℝ, 
    (m_min = 1 ∧ m_max = 2) ∧ 
    (∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (m_min ≤ m ∧ m < m_max)) :=
sorry

end range_of_m_l4008_400863


namespace opposite_of_negative_one_half_l4008_400844

theorem opposite_of_negative_one_half :
  -(-(1/2)) = 1/2 :=
by sorry

end opposite_of_negative_one_half_l4008_400844


namespace y_in_terms_of_x_l4008_400883

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by sorry

end y_in_terms_of_x_l4008_400883


namespace travis_apple_sales_proof_l4008_400834

/-- Calculates the total money Travis will take home from selling apples -/
def travis_apple_sales (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 from selling his apples -/
theorem travis_apple_sales_proof :
  travis_apple_sales 10000 50 35 = 7000 := by
  sorry

end travis_apple_sales_proof_l4008_400834


namespace arcsin_sin_eq_x_div_3_l4008_400807

theorem arcsin_sin_eq_x_div_3 (x : ℝ) :
  -3 * π / 2 ≤ x ∧ x ≤ 3 * π / 2 →
  (Real.arcsin (Real.sin x) = x / 3 ↔ 
    x = -3 * π / 2 ∨ x = 0 ∨ x = 3 * π / 4 ∨ x = 3 * π / 2) := by
  sorry

end arcsin_sin_eq_x_div_3_l4008_400807


namespace line_intersects_or_tangent_circle_l4008_400813

/-- A line in 2D space defined by the equation (x+1)m + (y-1)n = 0 --/
structure Line where
  m : ℝ
  n : ℝ

/-- A circle in 2D space defined by the equation x^2 + y^2 = 2 --/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 2}

/-- The point (-1, 1) --/
def M : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the line either intersects or is tangent to the circle --/
theorem line_intersects_or_tangent_circle (l : Line) : 
  (∃ p : ℝ × ℝ, p ∈ Circle ∧ (p.1 + 1) * l.m + (p.2 - 1) * l.n = 0) := by
  sorry

#check line_intersects_or_tangent_circle

end line_intersects_or_tangent_circle_l4008_400813


namespace smallest_student_count_l4008_400866

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ
  twelfth : ℕ

/-- The ratios between 9th grade and other grades --/
def ratios : GradeCount → Prop
  | ⟨n, t, e, w⟩ => 3 * t = 2 * n ∧ 5 * e = 4 * n ∧ 7 * w = 6 * n

/-- The total number of students --/
def total_students (g : GradeCount) : ℕ :=
  g.ninth + g.tenth + g.eleventh + g.twelfth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (g : GradeCount), ratios g ∧ total_students g = 349 ∧
  (∀ (h : GradeCount), ratios h → total_students h ≥ 349) :=
sorry

end smallest_student_count_l4008_400866


namespace lewis_harvest_earnings_l4008_400802

/-- Calculates the total earnings during a harvest season given regular weekly earnings, overtime weekly earnings, and the number of weeks. -/
def total_harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating that Lewis's total earnings during the harvest season equal $1,055,497 -/
theorem lewis_harvest_earnings :
  total_harvest_earnings 28 939 1091 = 1055497 := by
  sorry

#eval total_harvest_earnings 28 939 1091

end lewis_harvest_earnings_l4008_400802


namespace inequality_proof_l4008_400871

theorem inequality_proof (t : ℝ) (h : t > 0) : (1 + 2/t) * Real.log (1 + t) > 2 := by
  sorry

end inequality_proof_l4008_400871


namespace employee_pay_l4008_400839

/-- Given two employees with a total pay of 528 and one paid 120% of the other, prove the lower-paid employee's wage --/
theorem employee_pay (x y : ℝ) (h1 : x + y = 528) (h2 : x = 1.2 * y) : y = 240 := by
  sorry

end employee_pay_l4008_400839


namespace train_length_calculation_l4008_400877

-- Define the given parameters
def bridge_length : ℝ := 120
def crossing_time : ℝ := 20
def train_speed : ℝ := 66.6

-- State the theorem
theorem train_length_calculation :
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 1212 := by sorry

end train_length_calculation_l4008_400877


namespace quadratic_inequality_range_l4008_400812

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 := by
sorry

end quadratic_inequality_range_l4008_400812


namespace combination_permutation_equality_l4008_400865

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) :
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) →
  n = 9 := by
sorry

end combination_permutation_equality_l4008_400865


namespace arrange_four_men_five_women_l4008_400805

/-- The number of ways to arrange people into groups -/
def arrange_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let three_person_group := Nat.choose num_men 2 * Nat.choose num_women 1
  let first_two_person_group := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 1
  three_person_group * first_two_person_group * 1

/-- Theorem stating the number of ways to arrange 4 men and 5 women into specific groups -/
theorem arrange_four_men_five_women :
  arrange_groups 4 5 = 240 := by
  sorry


end arrange_four_men_five_women_l4008_400805


namespace quadratic_equation_root_zero_l4008_400828

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - 1 = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - 1 = 0) → 
  k = -1 := by
  sorry

end quadratic_equation_root_zero_l4008_400828


namespace fencing_match_prob_increase_correct_l4008_400872

def fencing_match_prob_increase 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  Nat.choose (k + l) k * p^k * (1 - p)^(l + 1)

theorem fencing_match_prob_increase_correct 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) :
  fencing_match_prob_increase k l hk hl p hp = 
    Nat.choose (k + l) k * p^k * (1 - p)^(l + 1) := by
  sorry

end fencing_match_prob_increase_correct_l4008_400872


namespace intersection_point_is_unique_solution_l4008_400806

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (1/8, -3/4)

/-- First line equation: y = -6x -/
def line1 (x y : ℚ) : Prop := y = -6 * x

/-- Second line equation: y + 3 = 18x -/
def line2 (x y : ℚ) : Prop := y + 3 = 18 * x

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique solution -/
theorem intersection_point_is_unique_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ (x' y' : ℚ), line1 x' y' → line2 x' y' → (x', y') = intersection_point := by
  sorry

end intersection_point_is_unique_solution_l4008_400806


namespace locus_of_P_B_l4008_400873

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
def Point := ℝ × ℝ

-- Define the given circle and points
variable (c : Circle)
variable (A : Point)
variable (B : Point)

-- Define P_B as a function of B
def P_B (B : Point) : Point := sorry

-- Define the condition that A and B are on the circle
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the condition that B is not on line OA
def not_on_line_OA (B : Point) (c : Circle) (A : Point) : Prop := sorry

-- Define the condition that P_B is on the internal bisector of ∠AOB
def on_internal_bisector (P : Point) (O : Point) (A : Point) (B : Point) : Prop := sorry

-- State the theorem
theorem locus_of_P_B (c : Circle) (A B : Point) 
  (h1 : on_circle A c)
  (h2 : on_circle B c)
  (h3 : not_on_line_OA B c A)
  (h4 : on_internal_bisector (P_B B) c.center A B) :
  ∃ (r : ℝ), ∀ B, on_circle (P_B B) { center := c.center, radius := r } :=
sorry

end locus_of_P_B_l4008_400873


namespace max_value_of_y_l4008_400819

noncomputable section

def angle_alpha : ℝ := Real.arctan (-Real.sqrt 3 / 3)

def point_P : ℝ × ℝ := (-3, Real.sqrt 3)

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def f (x : ℝ) : ℝ := 
  determinant (Real.cos (x + angle_alpha)) (-Real.sin angle_alpha) (Real.sin (x + angle_alpha)) (Real.cos angle_alpha)

def y (x : ℝ) : ℝ := Real.sqrt 3 * f (Real.pi / 2 - 2 * x) + 2 * f x ^ 2

theorem max_value_of_y :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) ∧ 
  y x = 3 ∧ 
  ∀ (z : ℝ), z ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → y z ≤ y x :=
sorry

end

end max_value_of_y_l4008_400819


namespace greatest_b_satisfying_inequality_l4008_400852

def quadratic_inequality (b : ℝ) : Prop :=
  b^2 - 14*b + 45 ≤ 0

theorem greatest_b_satisfying_inequality :
  ∃ (b : ℝ), quadratic_inequality b ∧
    ∀ (x : ℝ), quadratic_inequality x → x ≤ b :=
by
  -- The proof goes here
  sorry

end greatest_b_satisfying_inequality_l4008_400852


namespace lucky_years_2020_to_2024_l4008_400859

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2020_to_2024 :
  isLuckyYear 2020 ∧
  isLuckyYear 2021 ∧
  isLuckyYear 2022 ∧
  ¬isLuckyYear 2023 ∧
  isLuckyYear 2024 := by
  sorry

end lucky_years_2020_to_2024_l4008_400859


namespace sqrt_6_plus_sqrt_6_equals_3_l4008_400854

theorem sqrt_6_plus_sqrt_6_equals_3 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (6 + x) → x = 3 := by
  sorry

end sqrt_6_plus_sqrt_6_equals_3_l4008_400854


namespace min_production_avoids_loss_min_production_is_minimal_l4008_400801

/-- The daily production cost function for a shoe factory -/
def cost (n : ℕ) : ℝ := 4000 + 50 * n

/-- The daily revenue function for a shoe factory -/
def revenue (n : ℕ) : ℝ := 90 * n

/-- The daily profit function for a shoe factory -/
def profit (n : ℕ) : ℝ := revenue n - cost n

/-- The minimum number of pairs of shoes that must be produced daily to avoid loss -/
def min_production : ℕ := 100

theorem min_production_avoids_loss :
  ∀ n : ℕ, n ≥ min_production → profit n ≥ 0 :=
sorry

theorem min_production_is_minimal :
  ∀ m : ℕ, (∀ n : ℕ, n ≥ m → profit n ≥ 0) → m ≥ min_production :=
sorry

end min_production_avoids_loss_min_production_is_minimal_l4008_400801


namespace greatest_k_for_100_factorial_l4008_400851

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log2) 0

def highest_power_of_5 (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (Nat.log 5 (x + 1))) 0

theorem greatest_k_for_100_factorial (a b : ℕ) (k : ℕ) :
  a = factorial 100 →
  b = 100^k →
  (∀ m : ℕ, m > k → ¬(100^m ∣ a)) →
  (100^k ∣ a) →
  k = 12 := by sorry

end greatest_k_for_100_factorial_l4008_400851


namespace elizabeth_granola_profit_l4008_400898

/-- Calculates the net profit for Elizabeth's granola bag sales --/
theorem elizabeth_granola_profit : 
  let full_price : ℝ := 6.00
  let low_cost : ℝ := 2.50
  let high_cost : ℝ := 3.50
  let low_cost_bags : ℕ := 10
  let high_cost_bags : ℕ := 10
  let full_price_low_cost_sold : ℕ := 7
  let full_price_high_cost_sold : ℕ := 8
  let discounted_low_cost_bags : ℕ := 3
  let discounted_high_cost_bags : ℕ := 2
  let low_cost_discount : ℝ := 0.20
  let high_cost_discount : ℝ := 0.30

  let total_cost : ℝ := low_cost * low_cost_bags + high_cost * high_cost_bags
  let full_price_revenue : ℝ := full_price * (full_price_low_cost_sold + full_price_high_cost_sold)
  let discounted_low_price : ℝ := full_price * (1 - low_cost_discount)
  let discounted_high_price : ℝ := full_price * (1 - high_cost_discount)
  let discounted_revenue : ℝ := discounted_low_price * discounted_low_cost_bags + 
                                 discounted_high_price * discounted_high_cost_bags
  let total_revenue : ℝ := full_price_revenue + discounted_revenue
  let net_profit : ℝ := total_revenue - total_cost

  net_profit = 52.80 := by sorry

end elizabeth_granola_profit_l4008_400898
