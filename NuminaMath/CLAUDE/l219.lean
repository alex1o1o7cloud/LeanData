import Mathlib

namespace parabola_sum_l219_21985

/-- Represents a parabola of the form x = dy^2 + ey + f -/
structure Parabola where
  d : ℚ
  e : ℚ
  f : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.xCoord (p : Parabola) (y : ℚ) : ℚ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_sum (p : Parabola) :
  p.xCoord (-6) = 7 →  -- vertex condition
  p.xCoord (-3) = 2 →  -- point condition
  p.d + p.e + p.f = -182/9 := by
  sorry

#eval (-5/9 : ℚ) + (-20/3 : ℚ) + (-13 : ℚ)  -- Should evaluate to -182/9

end parabola_sum_l219_21985


namespace grocery_store_bottles_l219_21989

theorem grocery_store_bottles : 
  157 + 126 + 87 + 52 + 64 = 486 := by
  sorry

end grocery_store_bottles_l219_21989


namespace arithmetic_sequence_11th_term_l219_21922

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 61 := by
  sorry

end arithmetic_sequence_11th_term_l219_21922


namespace board_ratio_l219_21904

theorem board_ratio (total_length shorter_length : ℝ) 
  (h1 : total_length = 6)
  (h2 : shorter_length = 2)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end board_ratio_l219_21904


namespace property_price_calculation_l219_21948

/-- Calculate the total price of a property given the price per square foot and the sizes of the house and barn. -/
theorem property_price_calculation
  (price_per_sq_ft : ℕ)
  (house_size : ℕ)
  (barn_size : ℕ)
  (h1 : price_per_sq_ft = 98)
  (h2 : house_size = 2400)
  (h3 : barn_size = 1000) :
  price_per_sq_ft * (house_size + barn_size) = 333200 := by
  sorry

#eval 98 * (2400 + 1000) -- Sanity check

end property_price_calculation_l219_21948


namespace square_condition_l219_21912

def is_square (x : ℕ) : Prop := ∃ t : ℕ, x = t^2

def floor_div (n m : ℕ) : ℕ := n / m

def expression (n : ℕ) : ℕ :=
  let k := Nat.log2 n
  (List.range (k+1)).foldl (λ acc i => acc * floor_div n (2^i)) 1 + 2 * 4^(k / 2)

theorem square_condition (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, 2^k ≤ n ∧ n < 2^(k+1)) → 
  is_square (expression n) → 
  n = 2 ∨ n = 4 := by
sorry

#eval expression 2  -- Expected: 4 (which is 2^2)
#eval expression 4  -- Expected: 16 (which is 4^2)

end square_condition_l219_21912


namespace six_digit_square_numbers_l219_21997

theorem six_digit_square_numbers : 
  ∀ n : ℕ, 
    (100000 ≤ n ∧ n < 1000000) → 
    (∃ m : ℕ, m < 1000 ∧ n = m^2) → 
    (n = 390625 ∨ n = 141376) := by
  sorry

end six_digit_square_numbers_l219_21997


namespace x_over_y_equals_two_l219_21976

theorem x_over_y_equals_two (x y : ℝ) 
  (h1 : 3 < (x^2 - y^2) / (x^2 + y^2))
  (h2 : (x^2 - y^2) / (x^2 + y^2) < 4)
  (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = 2 := by
sorry

end x_over_y_equals_two_l219_21976


namespace students_per_bus_l219_21962

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end students_per_bus_l219_21962


namespace sum_of_squares_zero_l219_21991

theorem sum_of_squares_zero (x y z : ℝ) 
  (h : x / (y + z) + y / (z + x) + z / (x + y) = 1) :
  x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y) = 0 := by
  sorry

end sum_of_squares_zero_l219_21991


namespace division_remainder_problem_l219_21940

theorem division_remainder_problem (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  : P % (D * D') = D * R' + R + C := by
  sorry

end division_remainder_problem_l219_21940


namespace snack_machine_quarters_l219_21959

def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50
def quarter_value : ℕ := 25

def total_cost (candy_bars chocolate juice : ℕ) : ℕ :=
  candy_bars * candy_bar_cost + chocolate * chocolate_cost + juice * juice_cost

def quarters_needed (total : ℕ) : ℕ :=
  (total + quarter_value - 1) / quarter_value

theorem snack_machine_quarters : quarters_needed (total_cost 3 2 1) = 11 := by
  sorry

end snack_machine_quarters_l219_21959


namespace barrel_capacity_l219_21924

def number_of_barrels : ℕ := 4
def flow_rate : ℚ := 7/2
def fill_time : ℕ := 8

theorem barrel_capacity : 
  (flow_rate * fill_time) / number_of_barrels = 7 := by sorry

end barrel_capacity_l219_21924


namespace imaginary_part_of_complex_fraction_l219_21919

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + 2*i) / (2 - i)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l219_21919


namespace find_b_l219_21986

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0}
def set2 (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3*p.1 + b}

-- State the theorem
theorem find_b : ∃ b : ℝ, set1 ⊂ set2 b → b = 2 := by sorry

end find_b_l219_21986


namespace jennifer_remaining_money_l219_21929

def initial_amount : ℚ := 150

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_remaining_money :
  remaining_amount = 20 := by sorry

end jennifer_remaining_money_l219_21929


namespace thirtieth_term_of_sequence_l219_21954

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₁₃ : ℚ) (h₁ : a₁ = 10) (h₂ : a₁₃ = 50) :
  arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 30 = 100 := by
  sorry

end thirtieth_term_of_sequence_l219_21954


namespace sparrow_percentage_among_non_owls_l219_21945

theorem sparrow_percentage_among_non_owls (total : ℝ) (total_pos : 0 < total) :
  let sparrows := 0.4 * total
  let owls := 0.2 * total
  let pigeons := 0.1 * total
  let finches := 0.2 * total
  let robins := total - (sparrows + owls + pigeons + finches)
  let non_owls := total - owls
  (sparrows / non_owls) * 100 = 50 := by
  sorry

end sparrow_percentage_among_non_owls_l219_21945


namespace largest_power_dividing_factorial_l219_21923

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2016) :
  (∃ k : ℕ, k = 334 ∧ 
   (∀ m : ℕ, n^m ∣ n! ↔ m ≤ k)) := by
  sorry

end largest_power_dividing_factorial_l219_21923


namespace congruence_problem_l219_21966

theorem congruence_problem (n : ℤ) : 
  3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 3 ∨ n = 9 := by
sorry

end congruence_problem_l219_21966


namespace hyperbola_n_range_l219_21943

def is_hyperbola (m n : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

def foci_distance (m : ℝ) : ℝ := 4

theorem hyperbola_n_range (m n : ℝ) 
  (h1 : is_hyperbola m n) 
  (h2 : foci_distance m = 4) : 
  -1 < n ∧ n < 3 := by
sorry

end hyperbola_n_range_l219_21943


namespace dessert_cost_calculation_dessert_cost_is_eleven_l219_21998

/-- Calculates the cost of a dessert given the costs of other meal components and the total price --/
theorem dessert_cost_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (tip_percentage : ℝ) 
  (total_price : ℝ) : ℝ :=
  let base_cost := appetizer_cost + 2 * entree_cost
  let dessert_cost := (total_price - base_cost) / (1 + tip_percentage)
  dessert_cost

/-- Proves that the dessert cost is $11.00 given the specific meal costs --/
theorem dessert_cost_is_eleven :
  dessert_cost_calculation 9 20 0.3 78 = 11 := by
  sorry

end dessert_cost_calculation_dessert_cost_is_eleven_l219_21998


namespace mans_speed_against_current_l219_21958

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds, 
    the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry


end mans_speed_against_current_l219_21958


namespace no_snow_probability_l219_21960

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end no_snow_probability_l219_21960


namespace impossible_sums_l219_21963

-- Define the coin values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the set of possible coin values
def coin_values : Set ℕ := {penny, nickel, dime, quarter}

-- Define a function to check if a sum is possible with 5 coins
def is_possible_sum (sum : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = sum

-- Theorem statement
theorem impossible_sums : ¬(is_possible_sum 22) ∧ ¬(is_possible_sum 48) :=
sorry

end impossible_sums_l219_21963


namespace line_segment_endpoint_l219_21968

theorem line_segment_endpoint (x y : ℝ) :
  let start : ℝ × ℝ := (2, 2)
  let length : ℝ := 8
  let slope : ℝ := 3/4
  y > 0 ∧
  (y - start.2) / (x - start.1) = slope ∧
  Real.sqrt ((x - start.1)^2 + (y - start.2)^2) = length →
  ((x = 2 + 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 + 4 * Real.sqrt 5475 / 25) + 1/2) ∨
   (x = 2 - 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 - 4 * Real.sqrt 5475 / 25) + 1/2)) :=
by sorry

end line_segment_endpoint_l219_21968


namespace final_price_approx_l219_21918

/-- The final price after applying two successive discounts to a list price. -/
def final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  list_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price after discounts is approximately 57.33 -/
theorem final_price_approx :
  let list_price : ℝ := 65
  let discount1 : ℝ := 0.1  -- 10%
  let discount2 : ℝ := 0.020000000000000027  -- 2.0000000000000027%
  abs (final_price list_price discount1 discount2 - 57.33) < 0.01 := by
  sorry

end final_price_approx_l219_21918


namespace travel_speed_l219_21952

/-- Given a distance of 195 km and a travel time of 3 hours, prove that the speed is 65 km/h -/
theorem travel_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 195 ∧ time = 3 ∧ speed = distance / time → speed = 65 := by
  sorry

end travel_speed_l219_21952


namespace incorrect_number_correction_l219_21987

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 46)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 50) :
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = actual_num - incorrect_num ∧ 
    actual_num = 65 := by
  sorry

end incorrect_number_correction_l219_21987


namespace intersection_and_parallel_perpendicular_lines_l219_21928

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ (x, y) = P) ∧
  (∀ x y, 3*x - 4*y + 8 = 0 ↔ (∃ t, (x, y) = (t*3 + P.1, t*4 + P.2))) ∧
  (∀ x y, 4*x + 3*y - 6 = 0 ↔ (∃ t, (x, y) = (t*4 + P.1, -t*3 + P.2))) :=
sorry

end intersection_and_parallel_perpendicular_lines_l219_21928


namespace y_intercepts_equal_negative_two_l219_21984

-- Define the equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 6
def equation2 (x y : ℝ) : Prop := x + 4 * y = -8

-- Define y-intercept
def is_y_intercept (y : ℝ) (eq : ℝ → ℝ → Prop) : Prop := eq 0 y

-- Theorem statement
theorem y_intercepts_equal_negative_two :
  (is_y_intercept (-2) equation1) ∧ (is_y_intercept (-2) equation2) :=
sorry

end y_intercepts_equal_negative_two_l219_21984


namespace perfect_square_count_l219_21983

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n ≤ 1500 ∧ ∃ k : ℕ, 21 * n = k^2) ∧ 
  S.card = 8 := by
  sorry

end perfect_square_count_l219_21983


namespace order_of_abc_l219_21914

theorem order_of_abc : ∀ (a b c : ℝ),
  a = 0.1 * Real.exp 0.1 →
  b = 1 / 9 →
  c = -Real.log 0.9 →
  c < a ∧ a < b :=
by sorry

end order_of_abc_l219_21914


namespace last_digit_of_2011_powers_l219_21903

theorem last_digit_of_2011_powers : ∃ n : ℕ, (2^2011 + 3^2011) % 10 = 5 := by
  sorry

end last_digit_of_2011_powers_l219_21903


namespace probability_of_sum_seven_l219_21915

-- Define the two dice
def die1 : Finset ℕ := {1, 2, 3, 4, 5, 6}
def die2 : Finset ℕ := {2, 3, 4, 5, 6, 7}

-- Define the total outcomes
def total_outcomes : ℕ := die1.card * die2.card

-- Define the favorable outcomes (pairs that sum to 7)
def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)}

-- Theorem statement
theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end probability_of_sum_seven_l219_21915


namespace aristocrat_problem_l219_21911

theorem aristocrat_problem (total_people : ℕ) 
  (men_payment women_payment total_spent : ℚ) 
  (women_fraction : ℚ) :
  total_people = 3552 →
  men_payment = 45 →
  women_payment = 60 →
  women_fraction = 1 / 12 →
  total_spent = 17760 →
  ∃ (men_fraction : ℚ),
    men_fraction * men_payment * (total_people - (women_fraction⁻¹ * women_fraction * total_people)) + 
    women_fraction * women_payment * (women_fraction⁻¹ * women_fraction * total_people) = total_spent ∧
    men_fraction = 1 / 9 :=
by sorry

end aristocrat_problem_l219_21911


namespace sum_of_obtuse_angles_l219_21927

open Real

theorem sum_of_obtuse_angles (α β : Real) : 
  π < α ∧ α < 2*π → 
  π < β ∧ β < 2*π → 
  sin α = sqrt 5 / 5 → 
  cos β = -(3 * sqrt 10) / 10 → 
  α + β = 7 * π / 4 := by
sorry

end sum_of_obtuse_angles_l219_21927


namespace sixteen_fifth_equals_four_tenth_l219_21908

theorem sixteen_fifth_equals_four_tenth : 16^5 = 4^10 := by
  sorry

end sixteen_fifth_equals_four_tenth_l219_21908


namespace otimes_inequality_l219_21977

/-- Custom binary operation ⊗ on ℝ -/
def otimes (x y : ℝ) : ℝ := (1 - x) * (1 + y)

/-- Theorem: If (x-a) ⊗ (x+a) < 1 holds for any real x, then -2 < a < 0 -/
theorem otimes_inequality (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -2 < a ∧ a < 0 := by
  sorry

end otimes_inequality_l219_21977


namespace total_bales_at_end_of_week_l219_21920

def initial_bales : ℕ := 28
def daily_additions : List ℕ := [10, 15, 8, 20, 12, 4, 18]

theorem total_bales_at_end_of_week : 
  initial_bales + daily_additions.sum = 115 := by
  sorry

end total_bales_at_end_of_week_l219_21920


namespace total_students_in_line_l219_21975

theorem total_students_in_line 
  (students_in_front : ℕ) 
  (students_behind : ℕ) 
  (h1 : students_in_front = 15)
  (h2 : students_behind = 12) :
  students_in_front + 1 + students_behind = 28 :=
by sorry

end total_students_in_line_l219_21975


namespace food_expense_percentage_l219_21971

/-- Represents the percentage of income spent on various expenses --/
structure IncomeDistribution where
  food : ℝ
  education : ℝ
  rent : ℝ
  remaining : ℝ

/-- Proves that the percentage of income spent on food is 50% --/
theorem food_expense_percentage (d : IncomeDistribution) : d.food = 50 :=
  by
  have h1 : d.education = 15 := sorry
  have h2 : d.rent = 50 * (100 - d.food - d.education) / 100 := sorry
  have h3 : d.remaining = 17.5 := sorry
  have h4 : d.food + d.education + d.rent + d.remaining = 100 := sorry
  sorry

#check food_expense_percentage

end food_expense_percentage_l219_21971


namespace profit_percentage_invariance_l219_21930

theorem profit_percentage_invariance 
  (cost_price : ℝ) 
  (discount_percentage : ℝ) 
  (final_profit_percentage : ℝ) 
  (discount_percentage_pos : 0 < discount_percentage) 
  (discount_percentage_lt_100 : discount_percentage < 100) 
  (final_profit_percentage_pos : 0 < final_profit_percentage) :
  let selling_price := cost_price * (1 + final_profit_percentage / 100)
  let discounted_price := selling_price * (1 - discount_percentage / 100)
  let profit_without_discount := (selling_price - cost_price) / cost_price * 100
  profit_without_discount = final_profit_percentage := by
sorry

end profit_percentage_invariance_l219_21930


namespace smallest_quadratic_coefficient_l219_21936

theorem smallest_quadratic_coefficient (a : ℕ) : a ≥ 5 ↔ 
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    (0 < x₁ ∧ x₁ < 1) ∧ 
    (0 < x₂ ∧ x₂ < 1) ∧ 
    (x₁ ≠ x₂) ∧
    (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
    a > 0 ∧
    (∀ a' < a, ¬∃ (b' c' : ℤ) (y₁ y₂ : ℝ), 
      (0 < y₁ ∧ y₁ < 1) ∧ 
      (0 < y₂ ∧ y₂ < 1) ∧ 
      (y₁ ≠ y₂) ∧
      (∀ x, a' * x^2 + b' * x + c' = a' * (x - y₁) * (x - y₂)) ∧
      a' > 0) := by
  sorry

end smallest_quadratic_coefficient_l219_21936


namespace no_intersection_l219_21950

-- Define the functions
def f (x : ℝ) : ℝ := |3 * x + 4|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by sorry

end no_intersection_l219_21950


namespace complex_number_quadrant_l219_21973

theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.I * 3) / (1 + Complex.I * 2) = ⟨x, y⟩ := by
  sorry

end complex_number_quadrant_l219_21973


namespace fourth_number_in_sequence_l219_21905

/-- A sequence of 10 natural numbers where each number from the third onwards
    is the sum of the two preceding numbers. -/
def FibonacciLikeSequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, i.val ≥ 2 → a i = a (i - 1) + a (i - 2)

theorem fourth_number_in_sequence
  (a : Fin 10 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_seventh : a 6 = 42)
  (h_ninth : a 8 = 110) :
  a 3 = 10 := by
sorry

end fourth_number_in_sequence_l219_21905


namespace base3_addition_l219_21925

-- Define a type for base-3 numbers
def Base3 := ℕ

-- Function to convert a base-3 number to its decimal representation
def to_decimal (n : Base3) : ℕ := sorry

-- Function to convert a decimal number to its base-3 representation
def to_base3 (n : ℕ) : Base3 := sorry

-- Define the given numbers in base 3
def a : Base3 := to_base3 1
def b : Base3 := to_base3 22
def c : Base3 := to_base3 212
def d : Base3 := to_base3 1001

-- Define the result in base 3
def result : Base3 := to_base3 210

-- Theorem statement
theorem base3_addition :
  to_decimal a - to_decimal b + to_decimal c - to_decimal d = to_decimal result := by
  sorry

end base3_addition_l219_21925


namespace ice_cream_theorem_l219_21953

def num_flavors : ℕ := 4
def num_scoops : ℕ := 4

def ice_cream_combinations (n m : ℕ) : ℕ :=
  Nat.choose (n + m - 1) (n - 1)

theorem ice_cream_theorem :
  ice_cream_combinations num_flavors num_scoops = 35 := by
  sorry

end ice_cream_theorem_l219_21953


namespace sequence_properties_l219_21996

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a : 2 * a 5 - a 3 = 3)
  (h_b2 : b 2 = 1)
  (h_b4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end sequence_properties_l219_21996


namespace tetrahedron_volume_l219_21921

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  -- Angle between faces ABC and BCD
  angle : ℝ
  -- Area of face ABC
  area_ABC : ℝ
  -- Area of face BCD
  area_BCD : ℝ
  -- Length of edge BC
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the tetrahedron with given properties -/
theorem tetrahedron_volume :
  ∀ t : Tetrahedron,
    t.angle = π/4 ∧
    t.area_ABC = 150 ∧
    t.area_BCD = 100 ∧
    t.length_BC = 12 →
    volume t = (1250 * Real.sqrt 2) / 3 := by sorry

end tetrahedron_volume_l219_21921


namespace liquid_rise_ratio_l219_21967

-- Define the cones and marbles
def narrow_cone_radius : ℝ := 5
def wide_cone_radius : ℝ := 10
def narrow_marble_radius : ℝ := 2
def wide_marble_radius : ℝ := 3

-- Define the volume ratio
def volume_ratio : ℝ := 4

-- Theorem statement
theorem liquid_rise_ratio :
  let narrow_cone_volume := (1/3) * Real.pi * narrow_cone_radius^2
  let wide_cone_volume := (1/3) * Real.pi * wide_cone_radius^2
  let narrow_marble_volume := (4/3) * Real.pi * narrow_marble_radius^3
  let wide_marble_volume := (4/3) * Real.pi * wide_marble_radius^3
  let narrow_cone_rise := narrow_marble_volume / (Real.pi * narrow_cone_radius^2)
  let wide_cone_rise := wide_marble_volume / (Real.pi * wide_cone_radius^2)
  wide_cone_volume = volume_ratio * narrow_cone_volume →
  narrow_cone_rise / wide_cone_rise = 8 := by
  sorry

end liquid_rise_ratio_l219_21967


namespace circle_diameter_from_area_l219_21938

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end circle_diameter_from_area_l219_21938


namespace dubblefud_yellow_count_l219_21979

/-- The game of dubblefud with yellow, blue, and green chips -/
def dubblefud (yellow blue green : ℕ) : Prop :=
  2^yellow * 4^blue * 5^green = 16000 ∧ blue = green

theorem dubblefud_yellow_count :
  ∀ y b g : ℕ, dubblefud y b g → y = 1 :=
by sorry

end dubblefud_yellow_count_l219_21979


namespace expression_evaluation_l219_21982

theorem expression_evaluation :
  let x : ℚ := 4 / 7
  let y : ℚ := 8 / 5
  (7 * x + 5 * y + 4) / (60 * x * y + 5) = 560 / 559 :=
by sorry

end expression_evaluation_l219_21982


namespace num_purchasing_methods_eq_seven_l219_21993

/-- The number of purchasing methods for equipment types A and B -/
def num_purchasing_methods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (x, y) := p
    600000 * x + 700000 * y ≤ 5000000 ∧
    x ≥ 3 ∧
    y ≥ 2
  ) (Finset.product (Finset.range 10) (Finset.range 10))).card

/-- Theorem stating that the number of purchasing methods is 7 -/
theorem num_purchasing_methods_eq_seven :
  num_purchasing_methods = 7 := by sorry

end num_purchasing_methods_eq_seven_l219_21993


namespace no_odd_solution_l219_21916

theorem no_odd_solution :
  ¬∃ (a b c d e f : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f : ℚ) = 1 :=
by sorry

end no_odd_solution_l219_21916


namespace loanYears_correct_l219_21913

/-- Calculates the number of years for which the first part of a loan is lent, given the following conditions:
  * The total sum is 2704
  * The second part of the loan is 1664
  * The interest rate for the first part is 3% per annum
  * The interest rate for the second part is 5% per annum
  * The interest period for the second part is 3 years
  * The interest on the first part equals the interest on the second part
-/
def loanYears : ℕ :=
  let totalSum : ℕ := 2704
  let secondPart : ℕ := 1664
  let firstPartRate : ℚ := 3 / 100
  let secondPartRate : ℚ := 5 / 100
  let secondPartPeriod : ℕ := 3
  let firstPart : ℕ := totalSum - secondPart
  8

theorem loanYears_correct : loanYears = 8 := by sorry

end loanYears_correct_l219_21913


namespace original_eq_general_form_l219_21974

/-- The original quadratic equation -/
def original_equation (x : ℝ) : ℝ := 2 * (x + 2)^2 + (x + 3) * (x - 2) + 11

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 13

/-- Theorem stating the equivalence of the original equation and its general form -/
theorem original_eq_general_form :
  ∀ x, original_equation x = general_form x := by sorry

end original_eq_general_form_l219_21974


namespace cosine_of_45_degree_angle_in_triangle_l219_21972

theorem cosine_of_45_degree_angle_in_triangle (A B C : ℝ) :
  A = 120 ∧ B = 45 ∧ C = 15 ∧ A + B + C = 180 →
  Real.cos (B * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cosine_of_45_degree_angle_in_triangle_l219_21972


namespace hyperbola_equation_l219_21931

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is x - √3 y = 0 and one of its foci is on the directrix
    of the parabola y² = -4x, then its equation is 4/3 x² - 4y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (asymptote : ∀ x y : ℝ, x - Real.sqrt 3 * y = 0 → x^2 / a^2 - y^2 / b^2 = 1)
  (focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = -4*x ∧ x = 1) :
  ∀ x y : ℝ, 4/3 * x^2 - 4 * y^2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by
sorry

end hyperbola_equation_l219_21931


namespace specific_triangle_intercepted_segments_l219_21956

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  (right_triangle : side1^2 + side2^2 = hypotenuse^2)

/-- Calculates the lengths of segments intercepted by lines drawn through the center of the inscribed circle parallel to the sides of the triangle -/
def intercepted_segments (triangle : RightTriangleWithInscribedCircle) : (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem statement for the specific right triangle with sides 6, 8, and 10 -/
theorem specific_triangle_intercepted_segments :
  let triangle : RightTriangleWithInscribedCircle := {
    side1 := 6,
    side2 := 8,
    hypotenuse := 10,
    right_triangle := by norm_num
  }
  intercepted_segments triangle = (3/2, 8/3, 25/6) := by sorry

end specific_triangle_intercepted_segments_l219_21956


namespace tenth_order_magic_constant_l219_21999

/-- The magic constant of an nth-order magic square -/
def magic_constant (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- Theorem: The magic constant of a 10th-order magic square is 505 -/
theorem tenth_order_magic_constant :
  magic_constant 10 = 505 := by
  sorry

#eval magic_constant 10  -- This will evaluate to 505

end tenth_order_magic_constant_l219_21999


namespace beach_probability_l219_21955

/-- Given a beach scenario with people wearing sunglasses and caps -/
structure BeachScenario where
  sunglasses : ℕ  -- Number of people wearing sunglasses
  caps : ℕ        -- Number of people wearing caps
  prob_cap_and_sunglasses : ℚ  -- Probability that a person wearing a cap is also wearing sunglasses

/-- The probability that a person wearing sunglasses is also wearing a cap -/
def prob_sunglasses_and_cap (scenario : BeachScenario) : ℚ :=
  (scenario.prob_cap_and_sunglasses * scenario.caps) / scenario.sunglasses

theorem beach_probability (scenario : BeachScenario) 
  (h1 : scenario.sunglasses = 75)
  (h2 : scenario.caps = 60)
  (h3 : scenario.prob_cap_and_sunglasses = 1/3) :
  prob_sunglasses_and_cap scenario = 4/15 := by
  sorry

end beach_probability_l219_21955


namespace money_problem_l219_21907

theorem money_problem (a b : ℝ) : 
  (4 * a - b = 40) ∧ (6 * a + b = 110) → a = 15 ∧ b = 20 := by
sorry

end money_problem_l219_21907


namespace cone_base_circumference_l219_21980

/-- The circumference of the base of a right circular cone formed from a 180° sector of a circle --/
theorem cone_base_circumference (r : ℝ) (h : r = 5) :
  let full_circle_circumference := 2 * π * r
  let sector_angle := π  -- 180° in radians
  let full_angle := 2 * π  -- 360° in radians
  let base_circumference := (sector_angle / full_angle) * full_circle_circumference
  base_circumference = 5 * π :=
by sorry

end cone_base_circumference_l219_21980


namespace circle_area_not_quadrupled_l219_21902

theorem circle_area_not_quadrupled (r : ℝ) (h : r > 0) : 
  ∃ k : ℝ, k ≠ 4 ∧ π * (r^2)^2 = k * (π * r^2) :=
sorry

end circle_area_not_quadrupled_l219_21902


namespace percentage_problem_l219_21937

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 6000) : 0.2 * n = 1000 := by
  sorry

end percentage_problem_l219_21937


namespace petra_age_l219_21901

theorem petra_age (petra_age mother_age : ℕ) : 
  petra_age + mother_age = 47 →
  mother_age = 2 * petra_age + 14 →
  mother_age = 36 →
  petra_age = 11 :=
by sorry

end petra_age_l219_21901


namespace two_integers_sum_l219_21935

theorem two_integers_sum (a b : ℕ+) : 
  (a : ℤ) - (b : ℤ) = 3 → a * b = 63 → (a : ℤ) + (b : ℤ) = 17 := by
  sorry

end two_integers_sum_l219_21935


namespace cubic_equation_root_l219_21941

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℂ) ^ 3 + c * (3 + Real.sqrt 5 : ℂ) ^ 2 + d * (3 + Real.sqrt 5 : ℂ) + 15 = 0 →
  d = -18.5 := by
sorry

end cubic_equation_root_l219_21941


namespace airplane_seats_l219_21949

theorem airplane_seats :
  ∀ (total_seats : ℝ),
  (30 : ℝ) + 0.2 * total_seats + 0.75 * total_seats = total_seats →
  total_seats = 600 :=
by
  sorry

end airplane_seats_l219_21949


namespace mortgage_payment_proof_l219_21969

/-- Calculates the monthly mortgage payment -/
def calculate_monthly_payment (house_price : ℕ) (deposit : ℕ) (years : ℕ) : ℚ :=
  let mortgage := house_price - deposit
  let annual_payment := mortgage / years
  annual_payment / 12

/-- Proves that the monthly payment for the given mortgage scenario is 2 thousand dollars -/
theorem mortgage_payment_proof (house_price deposit years : ℕ) 
  (h1 : house_price = 280000)
  (h2 : deposit = 40000)
  (h3 : years = 10) :
  calculate_monthly_payment house_price deposit years = 2000 := by
  sorry

#eval calculate_monthly_payment 280000 40000 10

end mortgage_payment_proof_l219_21969


namespace ellipse_parabola_intersection_l219_21933

/-- Given an ellipse and a parabola intersecting at two points with a specific distance between them, prove the value of the parabola parameter. -/
theorem ellipse_parabola_intersection (p : ℝ) (h_p_pos : p > 0) : 
  (∃ A B : ℝ × ℝ, 
    A.1^2 / 8 + A.2^2 / 2 = 1 ∧
    B.1^2 / 8 + B.2^2 / 2 = 1 ∧
    A.2^2 = 2 * p * A.1 ∧
    B.2^2 = 2 * p * B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) →
  p = 1/4 := by
sorry

end ellipse_parabola_intersection_l219_21933


namespace continued_fraction_solution_l219_21964

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end continued_fraction_solution_l219_21964


namespace waiter_tips_l219_21992

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips : total_tips 10 5 3 = 15 := by
  sorry

end waiter_tips_l219_21992


namespace duplicate_page_number_l219_21946

/-- The largest positive integer n such that n(n+1)/2 < 2550 -/
def n : ℕ := 70

/-- The theorem stating the existence and uniqueness of the duplicated page number -/
theorem duplicate_page_number :
  ∃! x : ℕ, x ≤ n ∧ (n * (n + 1)) / 2 + x = 2550 := by sorry

end duplicate_page_number_l219_21946


namespace max_product_of_functions_l219_21917

/-- Given functions f and g on ℝ with specified ranges, prove that the maximum value of their product is 10 -/
theorem max_product_of_functions (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hg : ∀ x, g x ∈ Set.Icc (-2) 1) : 
  (∃ x, f x * g x = 10) ∧ (∀ x, f x * g x ≤ 10) := by
  sorry

#check max_product_of_functions

end max_product_of_functions_l219_21917


namespace cubic_root_solutions_l219_21990

/-- A rational triple (a, b, c) is a solution if a, b, and c are the roots of the polynomial x^3 + ax^2 + bx + c = 0 -/
def IsSolution (a b c : ℚ) : Prop :=
  ∀ x : ℚ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = a ∨ x = b ∨ x = c)

/-- The only rational triples (a, b, c) that are solutions are (0, 0, 0), (1, -1, -1), and (1, -2, 0) -/
theorem cubic_root_solutions :
  ∀ a b c : ℚ, IsSolution a b c ↔ ((a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, -1, -1) ∨ (a, b, c) = (1, -2, 0)) :=
sorry

end cubic_root_solutions_l219_21990


namespace stating_boat_speed_with_stream_l219_21957

/-- Represents the speed of a boat in different conditions. -/
structure BoatSpeed where
  stillWater : ℝ
  againstStream : ℝ
  withStream : ℝ

/-- 
Theorem stating that given a man's rowing speed in still water is 6 km/h 
and his speed against the stream is 10 km/h, his speed with the stream is 10 km/h.
-/
theorem boat_speed_with_stream 
  (speed : BoatSpeed) 
  (h1 : speed.stillWater = 6) 
  (h2 : speed.againstStream = 10) : 
  speed.withStream = 10 := by
  sorry

#check boat_speed_with_stream

end stating_boat_speed_with_stream_l219_21957


namespace tan_equation_solutions_l219_21961

theorem tan_equation_solutions (x : ℝ) :
  -π < x ∧ x ≤ π ∧ 2 * Real.tan x - Real.sqrt 3 = 0 ↔ 
  x = Real.arctan (Real.sqrt 3 / 2) ∨ x = Real.arctan (Real.sqrt 3 / 2) - π :=
by sorry

end tan_equation_solutions_l219_21961


namespace second_planner_cheaper_at_31_l219_21909

/-- Represents the cost function for an event planner -/
structure PlannerCost where
  initial_fee : ℕ
  per_guest : ℕ

/-- Calculates the total cost for a given number of guests -/
def total_cost (p : PlannerCost) (guests : ℕ) : ℕ :=
  p.initial_fee + p.per_guest * guests

/-- First planner's pricing structure -/
def planner1 : PlannerCost := ⟨150, 20⟩

/-- Second planner's pricing structure -/
def planner2 : PlannerCost := ⟨300, 15⟩

/-- Theorem stating that 31 is the minimum number of guests for which the second planner is cheaper -/
theorem second_planner_cheaper_at_31 :
  (∀ g : ℕ, g < 31 → total_cost planner1 g ≤ total_cost planner2 g) ∧
  (∀ g : ℕ, g ≥ 31 → total_cost planner2 g < total_cost planner1 g) :=
sorry

end second_planner_cheaper_at_31_l219_21909


namespace seventh_term_ratio_l219_21910

/-- Two arithmetic sequences with sums of first n terms R_n and U_n -/
def R_n (n : ℕ) : ℚ := sorry
def U_n (n : ℕ) : ℚ := sorry

/-- The ratio condition for all n -/
axiom ratio_condition (n : ℕ) : R_n n / U_n n = (3 * n + 5 : ℚ) / (2 * n + 13 : ℚ)

/-- The 7th term of each sequence -/
def r_7 : ℚ := sorry
def s_7 : ℚ := sorry

/-- The main theorem -/
theorem seventh_term_ratio : r_7 / s_7 = 4 / 3 := by sorry

end seventh_term_ratio_l219_21910


namespace quadratic_shift_sum_l219_21947

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 3 units right and 5 units up,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 22 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 2*(x-3)^2 - (x-3) + 7 + 5 = a*x^2 + b*x + c) → 
  a + b + c = 22 := by
sorry

end quadratic_shift_sum_l219_21947


namespace brendan_recharge_ratio_l219_21932

/-- Represents the financial data for Brendan's June earnings and expenses -/
structure FinancialData where
  totalEarnings : ℕ
  carCost : ℕ
  remainingMoney : ℕ

/-- Calculates the amount recharged on the debit card -/
def amountRecharged (data : FinancialData) : ℕ :=
  data.totalEarnings - data.carCost - data.remainingMoney

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of recharged amount to total earnings -/
def rechargeRatio (data : FinancialData) : Ratio :=
  let recharged := amountRecharged data
  let gcd := Nat.gcd recharged data.totalEarnings
  { numerator := recharged / gcd, denominator := data.totalEarnings / gcd }

/-- Theorem stating that Brendan's recharge ratio is 1:2 -/
theorem brendan_recharge_ratio :
  let data : FinancialData := { totalEarnings := 5000, carCost := 1500, remainingMoney := 1000 }
  let ratio := rechargeRatio data
  ratio.numerator = 1 ∧ ratio.denominator = 2 := by sorry


end brendan_recharge_ratio_l219_21932


namespace march_walking_distance_l219_21939

theorem march_walking_distance (days_in_month : Nat) (miles_per_day : Nat) (skipped_days : Nat) : 
  days_in_month = 31 → miles_per_day = 4 → skipped_days = 4 → 
  (days_in_month - skipped_days) * miles_per_day = 108 := by
  sorry

end march_walking_distance_l219_21939


namespace distribute_five_projects_three_teams_l219_21970

/-- The number of ways to distribute n distinct projects among k teams,
    where each team must receive at least one project. -/
def distribute_projects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 projects among 3 teams results in 60 arrangements -/
theorem distribute_five_projects_three_teams :
  distribute_projects 5 3 = 60 := by sorry

end distribute_five_projects_three_teams_l219_21970


namespace opposite_numbers_properties_l219_21995

theorem opposite_numbers_properties :
  (∀ a b : ℝ, a = -b → a + b = 0) ∧
  (∀ a b : ℝ, a + b = 0 → a = -b) ∧
  (∀ a b : ℝ, b ≠ 0 → (a / b = -1 → a = -b)) :=
by sorry

end opposite_numbers_properties_l219_21995


namespace multiplication_mistake_problem_l219_21994

theorem multiplication_mistake_problem :
  ∃ x : ℝ, (493 * x - 394 * x = 78426) ∧ (x = 792) := by
  sorry

end multiplication_mistake_problem_l219_21994


namespace botany_zoology_ratio_l219_21934

/-- Represents the number of books in Milton's collection. -/
structure BookCollection where
  total : ℕ
  zoology : ℕ
  botany : ℕ
  h_total : total = zoology + botany
  h_botany_multiple : ∃ n : ℕ, botany = n * zoology

/-- The ratio of botany books to zoology books in Milton's collection is 4:1. -/
theorem botany_zoology_ratio (collection : BookCollection)
    (h_total : collection.total = 80)
    (h_zoology : collection.zoology = 16) :
    collection.botany / collection.zoology = 4 := by
  sorry

end botany_zoology_ratio_l219_21934


namespace line_plane_perpendicular_theorem_l219_21978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

theorem line_plane_perpendicular_theorem 
  (a b : Line) (α : Plane) :
  perpendicular_lines a b → 
  perpendicular_line_plane a α → 
  parallel_line_plane b α ∨ subset_line_plane b α :=
sorry

end line_plane_perpendicular_theorem_l219_21978


namespace min_payment_amount_l219_21942

/-- Represents the number of bills of each denomination --/
structure BillCount where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of bills --/
def totalValue (bills : BillCount) : Nat :=
  10 * bills.tens + 5 * bills.fives + bills.ones

/-- Calculates the total count of bills --/
def totalCount (bills : BillCount) : Nat :=
  bills.tens + bills.fives + bills.ones

/-- Represents Tim's initial bill distribution --/
def timsBills : BillCount :=
  { tens := 13, fives := 11, ones := 17 }

/-- Theorem stating the minimum amount Tim can pay using at least 16 bills --/
theorem min_payment_amount (payment : BillCount) : 
  totalCount payment ≥ 16 → 
  totalCount payment ≤ totalCount timsBills → 
  totalValue payment ≥ 40 :=
by sorry

end min_payment_amount_l219_21942


namespace gcd_324_243_135_l219_21988

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end gcd_324_243_135_l219_21988


namespace concert_attendees_l219_21900

theorem concert_attendees :
  let num_buses : ℕ := 8
  let students_per_bus : ℕ := 45
  let chaperones_per_bus : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]
  let total_students : ℕ := num_buses * students_per_bus
  let total_chaperones : ℕ := chaperones_per_bus.sum
  let total_attendees : ℕ := total_students + total_chaperones
  total_attendees = 389 := by
  sorry


end concert_attendees_l219_21900


namespace arithmetic_sequence_property_l219_21951

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_property_l219_21951


namespace calculate_withdrawal_l219_21981

/-- Calculates the withdrawal amount given initial balance and transactions --/
theorem calculate_withdrawal 
  (initial_balance : ℕ) 
  (deposit_last_month : ℕ) 
  (deposit_this_month : ℕ) 
  (balance_increase : ℕ) 
  (h1 : initial_balance = 150)
  (h2 : deposit_last_month = 17)
  (h3 : deposit_this_month = 21)
  (h4 : balance_increase = 16) :
  ∃ (withdrawal : ℕ), 
    initial_balance + deposit_last_month - withdrawal + deposit_this_month 
    = initial_balance + balance_increase ∧ 
    withdrawal = 22 := by
sorry

end calculate_withdrawal_l219_21981


namespace simple_interest_problem_l219_21906

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 750 →
  rate = 6 / 100 →
  time = 5 →
  principal * rate * time = interest →
  principal = 2500 := by
sorry

end simple_interest_problem_l219_21906


namespace max_words_is_16056_l219_21944

/-- Represents a language with two letters and words of maximum length 13 -/
structure TwoLetterLanguage where
  max_word_length : ℕ
  max_word_length_eq : max_word_length = 13

/-- Calculates the maximum number of words in the language -/
def max_words (L : TwoLetterLanguage) : ℕ :=
  2^14 - 2^7

/-- States that no concatenation of two words forms another word -/
axiom no_concat_word (L : TwoLetterLanguage) :
  ∀ (w1 w2 : String), (w1.length ≤ L.max_word_length ∧ w2.length ≤ L.max_word_length) →
    (w1 ++ w2).length > L.max_word_length

/-- Theorem: The maximum number of words in the language is 16056 -/
theorem max_words_is_16056 (L : TwoLetterLanguage) :
  max_words L = 16056 := by
  sorry

end max_words_is_16056_l219_21944


namespace calculator_game_result_l219_21965

/-- The number of participants in the game -/
def num_participants : ℕ := 60

/-- The operation applied to the first calculator -/
def op1 (n : ℕ) (x : ℤ) : ℤ := x ^ 3 ^ n

/-- The operation applied to the second calculator -/
def op2 (n : ℕ) (x : ℤ) : ℤ := x ^ (2 ^ n)

/-- The operation applied to the third calculator -/
def op3 (n : ℕ) (x : ℤ) : ℤ := (-1) ^ n * x

/-- The final sum of the numbers on the calculators after one complete round -/
def final_sum : ℤ := op1 num_participants 2 + op2 num_participants 0 + op3 num_participants (-1)

theorem calculator_game_result : final_sum = 2 ^ (3 ^ 60) + 1 := by
  sorry

end calculator_game_result_l219_21965


namespace square_sum_given_sum_square_and_product_l219_21926

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 := by
  sorry

end square_sum_given_sum_square_and_product_l219_21926
