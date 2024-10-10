import Mathlib

namespace pablo_book_pages_l3139_313968

/-- The number of books Pablo reads -/
def num_books : ℕ := 12

/-- The total amount of money Pablo earned in cents -/
def total_earned : ℕ := 1800

/-- The number of pages in each book -/
def pages_per_book : ℕ := total_earned / num_books

theorem pablo_book_pages :
  pages_per_book = 150 :=
by sorry

end pablo_book_pages_l3139_313968


namespace proposition_p_equivalence_l3139_313951

theorem proposition_p_equivalence (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - m - 1 < 0) ↔ m > -2 := by
  sorry

end proposition_p_equivalence_l3139_313951


namespace triangle_theorem_l3139_313969

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C + Real.cos t.A * Real.cos t.B = Real.sqrt 3 * Real.sin t.A * Real.cos t.B)
  (h2 : t.a + t.c = 1)
  (h3 : 0 < t.B)
  (h4 : t.B < Real.pi)
  (h5 : 0 < t.a)
  (h6 : t.a < 1) :
  Real.cos t.B = 1/2 ∧ 1/2 ≤ t.b ∧ t.b < 1 := by
  sorry

end triangle_theorem_l3139_313969


namespace complex_equation_solution_l3139_313956

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end complex_equation_solution_l3139_313956


namespace star_properties_l3139_313982

-- Define the * operation for rational numbers
def star (a b : ℚ) : ℚ := (a + b) - abs (b - a)

-- Theorem statement
theorem star_properties :
  (star (-3) 2 = -6) ∧ (star (star 4 3) (-5) = -10) := by
  sorry

end star_properties_l3139_313982


namespace tan_seven_pi_sixths_l3139_313937

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_seven_pi_sixths_l3139_313937


namespace square_root_expression_value_l3139_313941

theorem square_root_expression_value :
  let x : ℝ := Real.sqrt 6 - Real.sqrt 2
  2 * x^2 + 4 * Real.sqrt 2 * x = 8 := by
sorry

end square_root_expression_value_l3139_313941


namespace probability_white_given_popped_l3139_313947

-- Define the probabilities
def P_white : ℝ := 0.4
def P_yellow : ℝ := 0.4
def P_red : ℝ := 0.2
def P_pop_given_white : ℝ := 0.7
def P_pop_given_yellow : ℝ := 0.5
def P_pop_given_red : ℝ := 0

-- Define the theorem
theorem probability_white_given_popped :
  let P_popped : ℝ := P_pop_given_white * P_white + P_pop_given_yellow * P_yellow + P_pop_given_red * P_red
  (P_pop_given_white * P_white) / P_popped = 7 / 12 := by
  sorry

end probability_white_given_popped_l3139_313947


namespace reflect_F_coordinates_l3139_313928

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The composition of reflecting over y-axis and then x-axis -/
def reflect_yx (p : ℝ × ℝ) : ℝ × ℝ := reflect_x (reflect_y p)

theorem reflect_F_coordinates :
  reflect_yx (6, -4) = (-6, 4) := by sorry

end reflect_F_coordinates_l3139_313928


namespace no_repeating_subsequence_l3139_313975

/-- Count the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Define the sequence a_n based on the parity of the number of 1's in the binary representation -/
def a (n : ℕ) : ℕ := 
  if countOnes n % 2 = 0 then 0 else 1

/-- The main theorem stating that there are no positive integers k and m satisfying the condition -/
theorem no_repeating_subsequence : 
  ¬ ∃ (k m : ℕ+), ∀ (j : ℕ), j < m → 
    a (k + j) = a (k + m + j) ∧ a (k + j) = a (k + 2*m + j) := by
  sorry

end no_repeating_subsequence_l3139_313975


namespace sector_central_angle_l3139_313911

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians. -/
theorem sector_central_angle (r θ : ℝ) : 
  (2 * r + r * θ = 4) →  -- perimeter condition
  ((1 / 2) * r^2 * θ = 1) →  -- area condition
  θ = 2 := by
sorry

end sector_central_angle_l3139_313911


namespace parabola_chord_theorem_l3139_313930

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (s : ℝ) : Prop := parabola s 4

-- Define perpendicular lines
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Define a point (x,y) on line AB
def point_on_AB (x y y1 y2 : ℝ) : Prop := y + 4 = (4 / (y1 + y2)) * (x - 8)

theorem parabola_chord_theorem :
  ∀ y1 y2 : ℝ,
  point_on_parabola 4 →
  parabola (y1^2 / 4) y1 →
  parabola (y2^2 / 4) y2 →
  perpendicular ((y1 - 4) / ((y1^2 - 16) / 4)) ((y2 - 4) / ((y2^2 - 16) / 4)) →
  point_on_AB 8 (-4) y1 y2 :=
by sorry

end parabola_chord_theorem_l3139_313930


namespace y_divisibility_l3139_313961

def y : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem y_divisibility :
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  (∃ k : ℕ, y = 32 * k) ∧
  (∃ k : ℕ, y = 64 * k) :=
by sorry

end y_divisibility_l3139_313961


namespace meaningful_fraction_l3139_313912

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_l3139_313912


namespace area_between_parabola_and_line_l3139_313905

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := 1
  let lower_bound := -1
  let upper_bound := 1
  (∫ (x : ℝ) in lower_bound..upper_bound, g x - f x) = 4/3 := by
sorry

end area_between_parabola_and_line_l3139_313905


namespace second_student_speed_l3139_313938

/-- Given two students walking in opposite directions, this theorem proves
    the speed of the second student given the conditions of the problem. -/
theorem second_student_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 60)
  (h3 : speed1 = 6)
  (h4 : distance = (speed1 + speed2) * time) :
  speed2 = 9 :=
by
  sorry

#check second_student_speed

end second_student_speed_l3139_313938


namespace average_rate_of_change_l3139_313940

def f (x : ℝ) : ℝ := 2 * x - 1

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = 2 :=
by sorry

end average_rate_of_change_l3139_313940


namespace brownie_count_l3139_313919

def tray_length : ℕ := 24
def tray_width : ℕ := 15
def brownie_side : ℕ := 3

theorem brownie_count : 
  (tray_length * tray_width) / (brownie_side * brownie_side) = 40 := by
  sorry

end brownie_count_l3139_313919


namespace cyrus_remaining_pages_l3139_313918

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the remaining pages to be written --/
def remainingPages (total : ℕ) (daily : DailyPages) : ℕ :=
  total - (daily.day1 + daily.day2 + daily.day3 + daily.day4)

/-- Theorem stating the number of remaining pages Cyrus needs to write --/
theorem cyrus_remaining_pages :
  let total := 500
  let daily := DailyPages.mk 25 (25 * 2) ((25 * 2) * 2) 10
  remainingPages total daily = 315 := by
  sorry

end cyrus_remaining_pages_l3139_313918


namespace function_with_given_derivative_l3139_313993

/-- Given a differentiable function f on ℝ with f'(x) = 1 + sin x,
    prove that there exists a constant C such that f(x) = x - cos x + C. -/
theorem function_with_given_derivative
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hf' : ∀ x, deriv f x = 1 + Real.sin x) :
  ∃ C, ∀ x, f x = x - Real.cos x + C := by
  sorry

end function_with_given_derivative_l3139_313993


namespace trapezoid_area_is_correct_l3139_313910

/-- The area of a trapezoid bounded by y = x, y = 10, y = 5, and the y-axis -/
def trapezoidArea : ℝ := 37.5

/-- The line y = x -/
def lineYeqX (x : ℝ) : ℝ := x

/-- The line y = 10 -/
def lineY10 (x : ℝ) : ℝ := 10

/-- The line y = 5 -/
def lineY5 (x : ℝ) : ℝ := 5

/-- The y-axis (x = 0) -/
def yAxis : Set ℝ := {x | x = 0}

theorem trapezoid_area_is_correct :
  trapezoidArea = 37.5 := by sorry

end trapezoid_area_is_correct_l3139_313910


namespace tom_seashells_l3139_313926

theorem tom_seashells (yesterday : ℕ) (today : ℕ) 
  (h1 : yesterday = 7) (h2 : today = 4) : 
  yesterday + today = 11 := by
  sorry

end tom_seashells_l3139_313926


namespace blanket_average_price_l3139_313978

/-- Given the following conditions:
    - A man purchased 8 blankets in total
    - 1 blanket costs Rs. 100
    - 5 blankets cost Rs. 150 each
    - 2 blankets cost Rs. 650 in total
    Prove that the average price of all blankets is Rs. 187.50 -/
theorem blanket_average_price :
  let total_blankets : ℕ := 8
  let price_of_one : ℕ := 100
  let price_of_five : ℕ := 150
  let price_of_two : ℕ := 650
  let total_cost : ℕ := price_of_one + 5 * price_of_five + price_of_two
  (total_cost : ℚ) / total_blankets = 187.5 := by
  sorry

end blanket_average_price_l3139_313978


namespace pens_given_to_friends_l3139_313936

def initial_pens : ℕ := 56
def remaining_pens : ℕ := 34

theorem pens_given_to_friends :
  initial_pens - remaining_pens = 22 := by
  sorry

end pens_given_to_friends_l3139_313936


namespace midpoint_trajectory_l3139_313942

/-- The trajectory of the midpoint of a segment between a fixed point and a point on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 16 ∧ x = (px + 12) / 2 ∧ y = py / 2) → 
  (x - 6)^2 + y^2 = 4 := by
sorry

end midpoint_trajectory_l3139_313942


namespace relative_error_approximation_l3139_313960

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  let f := fun x => 1 / (1 + x)
  let approx := fun x => 1 - x
  let relative_error := fun x => (f x - approx x) / f x
  relative_error y = y^2 := by
  sorry

end relative_error_approximation_l3139_313960


namespace steve_answerable_questions_l3139_313927

theorem steve_answerable_questions (total_questions : ℕ) (difference : ℕ) : 
  total_questions = 45 → difference = 7 → total_questions - difference = 38 := by
sorry

end steve_answerable_questions_l3139_313927


namespace sum_of_quadratic_roots_l3139_313924

theorem sum_of_quadratic_roots (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 0) → 
  (∃ r s : ℝ, (2 * r^2 - 8 * r - 10 = 0) ∧ 
              (2 * s^2 - 8 * s - 10 = 0) ∧ 
              (r + s = 4)) :=
by sorry

end sum_of_quadratic_roots_l3139_313924


namespace prob_red_card_standard_deck_l3139_313906

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Represents the properties of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    black_suits := 2 }

/-- Calculates the probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.ranks * d.red_suits : ℚ) / d.total_cards

/-- Theorem stating that the probability of drawing a red card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck : 
  prob_red_card standard_deck = 1/2 := by sorry

end prob_red_card_standard_deck_l3139_313906


namespace decimal_to_fraction_l3139_313908

theorem decimal_to_fraction (n d : ℕ) (h : n = 16) :
  (n : ℚ) / d = 32 / 100 → d = 50 := by sorry

end decimal_to_fraction_l3139_313908


namespace product_of_y_coordinates_l3139_313966

def point_P (y : ℝ) : ℝ × ℝ := (-3, y)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem product_of_y_coordinates (k : ℝ) (h : k > 0) :
  ∃ y1 y2 : ℝ,
    distance_squared (point_P y1) (5, 2) = k^2 ∧
    distance_squared (point_P y2) (5, 2) = k^2 ∧
    y1 * y2 = 68 - k^2 :=
  sorry

end product_of_y_coordinates_l3139_313966


namespace clothing_store_profit_l3139_313920

/-- Profit function for a clothing store --/
def profit_function (x : ℝ) : ℝ := 20 * x + 4000

/-- Maximum profit under cost constraint --/
def max_profit : ℝ := 5500

/-- Discount value for maximum profit under new conditions --/
def discount_value : ℝ := 9

/-- Theorem stating the main results --/
theorem clothing_store_profit :
  (∀ x : ℝ, x ≥ 60 → x ≤ 100 → profit_function x = 20 * x + 4000) ∧
  (∀ x : ℝ, x ≥ 60 → x ≤ 75 → 160 * x + 120 * (100 - x) ≤ 15000 → profit_function x ≤ max_profit) ∧
  (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 160 * x + 120 * (100 - x) ≤ 15000 ∧ profit_function x = max_profit) ∧
  (∀ a : ℝ, 0 < a → a < 20 → 
    (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 
      ((20 - a) * x + 100 * a + 3600 = 4950) → a = discount_value)) :=
by sorry

end clothing_store_profit_l3139_313920


namespace factorial_division_l3139_313974

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end factorial_division_l3139_313974


namespace composite_sum_of_squares_l3139_313900

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end composite_sum_of_squares_l3139_313900


namespace number_of_refills_l3139_313992

/-- Proves that the number of refills is 4 given the total spent and cost per refill -/
theorem number_of_refills (total_spent : ℕ) (cost_per_refill : ℕ) 
  (h1 : total_spent = 40) 
  (h2 : cost_per_refill = 10) : 
  total_spent / cost_per_refill = 4 := by
  sorry

end number_of_refills_l3139_313992


namespace extremum_values_of_e_l3139_313971

theorem extremum_values_of_e (a b c d e : ℝ) 
  (h1 : 3*a + 2*b - c + 4*d + Real.sqrt 133 * e = Real.sqrt 133)
  (h2 : 2*a^2 + 3*b^2 + 3*c^2 + d^2 + 6*e^2 = 60) :
  ∃ (e_min e_max : ℝ), 
    e_min = (1 - Real.sqrt 19) / 2 ∧ 
    e_max = (1 + Real.sqrt 19) / 2 ∧
    e_min ≤ e ∧ e ≤ e_max ∧
    (e = e_min ∨ e = e_max → 
      ∃ (k : ℝ), a = 3*k/8 ∧ b = k/6 ∧ c = -k/12 ∧ d = k) :=
by sorry

end extremum_values_of_e_l3139_313971


namespace quadratic_inequality_l3139_313907

/-- A quadratic function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a * x + b

/-- The composition of f with itself -/
def f_comp (a b x : ℝ) : ℝ := f a b (f a b x)

/-- Theorem: If f(f(x)) = 0 has four distinct real solutions and
    the sum of two of these solutions is -1, then b ≤ -1/4 -/
theorem quadratic_inequality (a b : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f_comp a b w = 0 ∧ f_comp a b x = 0 ∧ f_comp a b y = 0 ∧ f_comp a b z = 0) →
  (∃ p q : ℝ, f_comp a b p = 0 ∧ f_comp a b q = 0 ∧ p + q = -1) →
  b ≤ -1/4 := by
  sorry

end quadratic_inequality_l3139_313907


namespace area_of_triangle_OAB_is_one_l3139_313997

/-- Given vector a and b in ℝ², prove that the area of triangle OAB is 1 -/
theorem area_of_triangle_OAB_is_one 
  (a b : ℝ × ℝ)
  (h_a : a = (-1/2, Real.sqrt 3/2))
  (h_OA : (a.1 - b.1, a.2 - b.2) = (a.1 - b.1, a.2 - b.2))
  (h_OB : (a.1 + b.1, a.2 + b.2) = (a.1 + b.1, a.2 + b.2))
  (h_isosceles : ‖(a.1 - b.1, a.2 - b.2)‖ = ‖(a.1 + b.1, a.2 + b.2)‖)
  (h_right_angle : (a.1 - b.1, a.2 - b.2) • (a.1 + b.1, a.2 + b.2) = 0) :
  (1/2) * ‖(a.1 - b.1, a.2 - b.2)‖ * ‖(a.1 + b.1, a.2 + b.2)‖ = 1 :=
by sorry


end area_of_triangle_OAB_is_one_l3139_313997


namespace book_profit_percentage_l3139_313972

/-- Calculates the profit percentage on the cost price for a book sale --/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (discount_rate : ℝ) 
  (h1 : cost_price = 47.50)
  (h2 : marked_price = 69.85)
  (h3 : discount_rate = 0.15) : 
  ∃ (profit_percentage : ℝ), 
    abs (profit_percentage - 24.99) < 0.01 ∧ 
    profit_percentage = (marked_price * (1 - discount_rate) - cost_price) / cost_price * 100 := by
  sorry


end book_profit_percentage_l3139_313972


namespace monotonic_increasing_interval_of_f_l3139_313981

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x ≤ y → f x ≤ f y :=
by sorry

end monotonic_increasing_interval_of_f_l3139_313981


namespace four_Y_three_equals_49_l3139_313959

-- Define the new operation Y
def Y (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem four_Y_three_equals_49 : Y 4 3 = 49 := by
  sorry

end four_Y_three_equals_49_l3139_313959


namespace polynomial_factorization_l3139_313904

/-- For all real numbers a and b, a²b - 25b = b(a + 5)(a - 5) -/
theorem polynomial_factorization (a b : ℝ) : a^2 * b - 25 * b = b * (a + 5) * (a - 5) := by
  sorry

end polynomial_factorization_l3139_313904


namespace monomial_like_terms_sum_l3139_313914

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c₁ c₂ : ℚ, a x y = c₁ ∧ b x y = c₂

theorem monomial_like_terms_sum (m n : ℕ) :
  like_terms (fun x y => 5 * x^m * y) (fun x y => -3 * x^2 * y^n) →
  m + n = 3 := by
sorry

end monomial_like_terms_sum_l3139_313914


namespace product_pricing_l3139_313922

/-- Given a cost per unit, original markup percentage, and current price percentage,
    calculate the current selling price and profit per unit. -/
theorem product_pricing (a : ℝ) (h : a > 0) :
  let original_price := a * (1 + 0.22)
  let current_price := original_price * 0.85
  let profit := current_price - a
  (current_price = 1.037 * a) ∧ (profit = 0.037 * a) := by
  sorry

end product_pricing_l3139_313922


namespace multiplication_table_odd_fraction_l3139_313909

theorem multiplication_table_odd_fraction :
  let n : ℕ := 15
  let total_products : ℕ := (n + 1) * (n + 1)
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  odd_products / total_products = 1 / 4 := by
sorry

end multiplication_table_odd_fraction_l3139_313909


namespace cordelia_hair_coloring_time_l3139_313953

/-- Represents the hair coloring process -/
structure HairColoring where
  bleaching_time : ℝ
  dyeing_time : ℝ

/-- The total time for the hair coloring process -/
def total_time (hc : HairColoring) : ℝ :=
  hc.bleaching_time + hc.dyeing_time

/-- Theorem stating the total time for Cordelia's hair coloring process -/
theorem cordelia_hair_coloring_time :
  ∃ (hc : HairColoring),
    hc.bleaching_time = 3 ∧
    hc.dyeing_time = 2 * hc.bleaching_time ∧
    total_time hc = 9 := by
  sorry

end cordelia_hair_coloring_time_l3139_313953


namespace larger_number_proof_l3139_313955

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end larger_number_proof_l3139_313955


namespace vector_calculation_l3139_313967

/-- Given vectors AB and BC in 2D space, prove that -1/2 * AC equals the specified vector -/
theorem vector_calculation (AB BC : Fin 2 → ℝ) 
  (h1 : AB = ![3, 7])
  (h2 : BC = ![-2, 3]) :
  (-1/2 : ℝ) • (AB + BC) = ![-1/2, -5] := by
  sorry

end vector_calculation_l3139_313967


namespace cos_shift_l3139_313965

theorem cos_shift (x : ℝ) : 
  Real.cos (x / 2 - π / 3) = Real.cos ((x - 2 * π / 3) / 2) := by sorry

end cos_shift_l3139_313965


namespace complex_magnitude_problem_l3139_313987

theorem complex_magnitude_problem (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l3139_313987


namespace mexican_restaurant_bill_solution_l3139_313934

/-- Represents the cost of items at a Mexican restaurant -/
structure MexicanRestaurantCosts where
  T : ℝ  -- Cost of a taco
  E : ℝ  -- Cost of an enchilada
  B : ℝ  -- Cost of a burrito

/-- The bills for three friends at a Mexican restaurant -/
def friend_bills (c : MexicanRestaurantCosts) : Prop :=
  2 * c.T + 3 * c.E = 7.80 ∧
  3 * c.T + 5 * c.E = 12.70 ∧
  4 * c.T + 2 * c.E + c.B = 15.40

/-- The theorem stating the unique solution for the Mexican restaurant bill problem -/
theorem mexican_restaurant_bill_solution :
  ∃! c : MexicanRestaurantCosts, friend_bills c ∧ c.T = 0.90 ∧ c.E = 2.00 ∧ c.B = 7.80 :=
by sorry

end mexican_restaurant_bill_solution_l3139_313934


namespace number_between_fractions_l3139_313985

theorem number_between_fractions : 0.2012 > (1 : ℚ) / 5 ∧ 0.2012 < (1 : ℚ) / 4 := by
  sorry

end number_between_fractions_l3139_313985


namespace gala_trees_l3139_313996

/-- Represents the orchard with Fuji and Gala apple trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- Conditions of the orchard -/
def validOrchard (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

theorem gala_trees (o : Orchard) (h : validOrchard o) : o.gala = 50 := by
  sorry

end gala_trees_l3139_313996


namespace katie_marbles_l3139_313902

/-- The number of pink marbles Katie has -/
def pink_marbles : ℕ := 13

/-- The number of orange marbles Katie has -/
def orange_marbles : ℕ := pink_marbles - 9

/-- The number of purple marbles Katie has -/
def purple_marbles : ℕ := 4 * orange_marbles

/-- The total number of marbles Katie has -/
def total_marbles : ℕ := 33

theorem katie_marbles : 
  pink_marbles + orange_marbles + purple_marbles = total_marbles ∧ 
  orange_marbles = pink_marbles - 9 ∧
  purple_marbles = 4 * orange_marbles ∧
  pink_marbles = 13 := by sorry

end katie_marbles_l3139_313902


namespace cube_volume_from_diagonal_l3139_313983

theorem cube_volume_from_diagonal (diagonal : ℝ) (h : diagonal = 6 * Real.sqrt 2) :
  ∃ (side : ℝ), side > 0 ∧ side^3 = 48 * Real.sqrt 6 := by
  sorry

end cube_volume_from_diagonal_l3139_313983


namespace andrews_dog_foreign_objects_l3139_313917

/-- The number of burrs on Andrew's dog -/
def num_burrs : ℕ := 12

/-- The ratio of ticks to burrs on Andrew's dog -/
def tick_to_burr_ratio : ℕ := 6

/-- The total number of foreign objects (burrs and ticks) on Andrew's dog -/
def total_foreign_objects : ℕ := num_burrs + num_burrs * tick_to_burr_ratio

theorem andrews_dog_foreign_objects :
  total_foreign_objects = 84 :=
by sorry

end andrews_dog_foreign_objects_l3139_313917


namespace exam_score_calculation_l3139_313901

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 160)
  (h3 : correct_answers = 44)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end exam_score_calculation_l3139_313901


namespace max_salary_is_368000_l3139_313990

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a soccer team -/
def max_player_salary (team : SoccerTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem: The maximum possible salary for a single player in the given conditions is 368000 -/
theorem max_salary_is_368000 :
  let team : SoccerTeam := ⟨25, 18000, 800000⟩
  max_player_salary team = 368000 := by
  sorry

#eval max_player_salary ⟨25, 18000, 800000⟩

end max_salary_is_368000_l3139_313990


namespace cylinder_lateral_surface_area_l3139_313991

-- Define the regular quadrilateral pyramid
structure RegularQuadPyramid where
  a : ℝ  -- base edge
  h : ℝ  -- height
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = (5/2) * a

-- Define the cylinder
structure Cylinder (P : RegularQuadPyramid) where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cylinder

-- Theorem statement
theorem cylinder_lateral_surface_area 
  (P : RegularQuadPyramid) 
  (C : Cylinder P) :
  ∃ (S : ℝ), S = (π * P.a^2 * Real.sqrt 46) / 9 ∧ 
  S = 2 * π * C.r * C.h :=
sorry

end cylinder_lateral_surface_area_l3139_313991


namespace square_value_l3139_313958

theorem square_value (square : ℚ) : 8/12 = square/3 → square = 2 := by
  sorry

end square_value_l3139_313958


namespace ellipse_parabola_intersection_distance_l3139_313945

/-- The distance between intersection points of an ellipse and a parabola -/
theorem ellipse_parabola_intersection_distance : 
  ∀ (ellipse : (ℝ × ℝ) → Prop) (parabola : (ℝ × ℝ) → Prop) 
    (focus : ℝ × ℝ) (directrix : ℝ → ℝ × ℝ),
  (∀ x y, ellipse (x, y) ↔ x^2 / 16 + y^2 / 36 = 1) →
  (∃ c, ∀ x, directrix x = (c, x)) →
  (∃ x₁ y₁ x₂ y₂, ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
                   ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
                   (x₁, y₁) ≠ (x₂, y₂)) →
  (∃ x y, focus = (x, y) ∧ parabola (x, y)) →
  (∃ x y, focus = (x, y) ∧ ellipse (x, y)) →
  ∃ x₁ y₁ x₂ y₂, 
    ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
    ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 24 * Real.sqrt 5 / Real.sqrt (9 + 5 * Real.sqrt 5) :=
by sorry

end ellipse_parabola_intersection_distance_l3139_313945


namespace extreme_value_at_zero_tangent_line_equation_decreasing_condition_l3139_313986

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

def f_prime (a : ℝ) (x : ℝ) : ℝ := (-3 * x^2 + (6 - a) * x + a) / Real.exp x

theorem extreme_value_at_zero (a : ℝ) :
  f_prime a 0 = 0 → a = 0 := by sorry

theorem tangent_line_equation (a : ℝ) :
  a = 0 → ∀ x y : ℝ, y = f a x → (3 * x - Real.exp 1 * y = 0 ↔ x = 1) := by sorry

theorem decreasing_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 3 → f_prime a x ≤ 0) ↔ a ≥ -9/2 := by sorry

end extreme_value_at_zero_tangent_line_equation_decreasing_condition_l3139_313986


namespace f_increasing_condition_f_max_min_on_interval_log_inequality_l3139_313988

noncomputable section

variables (a : ℝ) (x : ℝ) (n : ℕ)

def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

theorem f_increasing_condition (h : a > 0) :
  (∀ x ≥ 1, Monotone (f a)) ↔ a ≥ 1 := by sorry

theorem f_max_min_on_interval (h : a = 1) :
  (∀ x ∈ Set.Icc (1/2) 2, f a x ≤ 1 - Real.log 2) ∧
  (∀ x ∈ Set.Icc (1/2) 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f a x = 1 - Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f a x = 0) := by sorry

theorem log_inequality (h : a = 1) (hn : n > 1) :
  Real.log (n / (n - 1 : ℝ)) > 1 / n := by sorry

end

end f_increasing_condition_f_max_min_on_interval_log_inequality_l3139_313988


namespace complex_equation_solution_l3139_313948

theorem complex_equation_solution (i z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1 - i) :
  z = -i - 1 := by sorry

end complex_equation_solution_l3139_313948


namespace symmetric_line_l3139_313944

/-- Given a line L1 with equation x - 4y + 2 = 0 and an axis of symmetry x = -2,
    the symmetric line L2 has the equation x + 4y + 2 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∃ L1 : Set (ℝ × ℝ), L1 = {(x, y) | x - 4*y + 2 = 0}) →
  (∃ axis : Set (ℝ × ℝ), axis = {(x, y) | x = -2}) →
  (∃ L2 : Set (ℝ × ℝ), L2 = {(x, y) | x + 4*y + 2 = 0}) :=
by sorry

end symmetric_line_l3139_313944


namespace inequality_solution_set_l3139_313943

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 4 < 6) ↔ (x < 5) := by sorry

end inequality_solution_set_l3139_313943


namespace equal_roots_quadratic_l3139_313962

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 8*x - 4*k = 0 ∧ 
   ∀ y : ℝ, y^2 - 8*y - 4*k = 0 → y = x) → 
  k = -4 :=
by sorry

end equal_roots_quadratic_l3139_313962


namespace problem_solution_l3139_313984

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

theorem problem_solution (m : ℝ) (α : ℝ) :
  (∀ x, determinant (x + m) 2 1 x < 0 ↔ x ∈ Set.Ioo (-1) 2) →
  m * Real.cos α + 2 * Real.sin α = 0 →
  m = -1 ∧ Real.tan (2 * α - Real.pi / 4) = 1 / 7 := by
  sorry

end problem_solution_l3139_313984


namespace loss_percent_calculation_l3139_313994

theorem loss_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 600)
  (h2 : selling_price = 450) :
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end loss_percent_calculation_l3139_313994


namespace arlettes_age_l3139_313939

/-- Given the ages of three people: Omi, Kimiko, and Arlette, prove that Arlette's age is 21 years. -/
theorem arlettes_age (omi kimiko arlette : ℕ) : 
  omi = 2 * kimiko →  -- Omi's age is twice Kimiko's age
  kimiko = 28 →       -- Kimiko's age is 28 years
  (omi + kimiko + arlette) / 3 = 35 →  -- The average age of the three is 35 years
  arlette = 21 := by
sorry

end arlettes_age_l3139_313939


namespace cos_sin_sum_equality_l3139_313903

theorem cos_sin_sum_equality : 
  Real.cos (16 * π / 180) * Real.cos (61 * π / 180) + 
  Real.sin (16 * π / 180) * Real.sin (61 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end cos_sin_sum_equality_l3139_313903


namespace bus_ride_cost_l3139_313995

-- Define the cost of bus and train rides
def bus_cost : ℝ := sorry
def train_cost : ℝ := sorry

-- State the theorem
theorem bus_ride_cost :
  (train_cost = bus_cost + 6.85) →
  (train_cost + bus_cost = 9.65) →
  (bus_cost = 1.40) := by
  sorry

end bus_ride_cost_l3139_313995


namespace prime_power_sum_implies_power_of_three_l3139_313976

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end prime_power_sum_implies_power_of_three_l3139_313976


namespace equation_solution_l3139_313979

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 9 / (y / 3) → y = 3 * Real.sqrt 54 ∨ y = -3 * Real.sqrt 54 := by
  sorry

end equation_solution_l3139_313979


namespace minimum_buses_required_l3139_313957

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) : 
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 :=
by sorry

end minimum_buses_required_l3139_313957


namespace grocery_cost_l3139_313954

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ) : 
  (10 * mango_cost = 24 * rice_cost) →
  (flour_cost = 2 * rice_cost) →
  (flour_cost = 24) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 271.2) := by
sorry

end grocery_cost_l3139_313954


namespace sandys_comic_books_l3139_313929

/-- Sandy's comic book problem -/
theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 : ℚ) + 6 = 13 → initial = 14 := by
  sorry

end sandys_comic_books_l3139_313929


namespace second_quadrant_necessary_not_sufficient_for_obtuse_l3139_313998

/-- An angle is in the second quadrant if it's between 90° and 180° or between -270° and -180° --/
def is_second_quadrant_angle (α : Real) : Prop :=
  (90 < α ∧ α ≤ 180) ∨ (-270 < α ∧ α ≤ -180)

/-- An angle is obtuse if it's between 90° and 180° --/
def is_obtuse_angle (α : Real) : Prop :=
  90 < α ∧ α < 180

/-- Theorem stating that "α is a second quadrant angle" is a necessary but not sufficient condition for "α is an obtuse angle" --/
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α : Real, is_obtuse_angle α → is_second_quadrant_angle α) ∧
  (∃ α : Real, is_second_quadrant_angle α ∧ ¬is_obtuse_angle α) :=
by sorry

end second_quadrant_necessary_not_sufficient_for_obtuse_l3139_313998


namespace one_chief_physician_probability_l3139_313970

theorem one_chief_physician_probability 
  (total_male_doctors : ℕ) 
  (total_female_doctors : ℕ) 
  (male_chief_physicians : ℕ) 
  (female_chief_physicians : ℕ) 
  (selected_male_doctors : ℕ) 
  (selected_female_doctors : ℕ) :
  total_male_doctors = 4 →
  total_female_doctors = 5 →
  male_chief_physicians = 1 →
  female_chief_physicians = 1 →
  selected_male_doctors = 3 →
  selected_female_doctors = 2 →
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) (selected_male_doctors - 1) *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) (selected_female_doctors - 1)) /
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors) = 6 / 17 := by
  sorry

end one_chief_physician_probability_l3139_313970


namespace doug_marbles_l3139_313923

theorem doug_marbles (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) 
  (h1 : ed_initial = doug_initial + 12)
  (h2 : ed_lost = 20)
  (h3 : ed_current = 17)
  (h4 : ed_initial = ed_current + ed_lost) : 
  doug_initial = 25 := by
sorry

end doug_marbles_l3139_313923


namespace quadratic_inequality_always_true_l3139_313916

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 := by sorry

end quadratic_inequality_always_true_l3139_313916


namespace increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l3139_313950

-- Define a constantly increasing function
def constantlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a constantly decreasing function
def constantlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define a function that is bounded above
def boundedAbove (f : ℝ → ℝ) : Prop :=
  ∃ M, ∀ x, f x ≤ M

-- Define a function that is bounded below
def boundedBelow (f : ℝ → ℝ) : Prop :=
  ∃ m, ∀ x, f x ≥ m

-- Theorem statement
theorem increasing_not_always_unbounded_and_decreasing_not_always_unbounded :
  (∃ f : ℝ → ℝ, constantlyIncreasing f ∧ boundedAbove f) ∧
  (∃ g : ℝ → ℝ, constantlyDecreasing g ∧ boundedBelow g) :=
sorry

end increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l3139_313950


namespace gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l3139_313925

/-- The nth triangular number -/
def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

/-- Theorem: The greatest common divisor of 6Tn and n+1 is at most 3 -/
theorem gcd_6Tn_nplus1_le_3 (n : ℕ+) :
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

/-- Theorem: There exists an n such that the greatest common divisor of 6Tn and n+1 is exactly 3 -/
theorem exists_gcd_6Tn_nplus1_eq_3 :
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l3139_313925


namespace garland_arrangement_count_l3139_313952

def blue_bulbs : ℕ := 5
def red_bulbs : ℕ := 6
def white_bulbs : ℕ := 7

def total_non_white_bulbs : ℕ := blue_bulbs + red_bulbs
def total_spaces : ℕ := total_non_white_bulbs + 1

theorem garland_arrangement_count :
  (Nat.choose total_non_white_bulbs blue_bulbs) * (Nat.choose total_spaces white_bulbs) = 365904 :=
sorry

end garland_arrangement_count_l3139_313952


namespace function_properties_l3139_313980

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.sqrt a / (a^x + Real.sqrt a)

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a x + f a (1 - x) = -1) ∧
  (f a (-2) + f a (-1) + f a 0 + f a 1 + f a 2 + f a 3 = -3) := by
  sorry

end function_properties_l3139_313980


namespace intersection_of_A_and_B_l3139_313973

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l3139_313973


namespace cos_150_degrees_l3139_313963

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l3139_313963


namespace water_depth_in_cistern_l3139_313999

theorem water_depth_in_cistern (length width total_wet_area : ℝ) 
  (h_length : length = 7)
  (h_width : width = 4)
  (h_total_wet_area : total_wet_area = 55.5)
  : ∃ depth : ℝ, 
    depth = 1.25 ∧ 
    total_wet_area = length * width + 2 * length * depth + 2 * width * depth :=
by sorry

end water_depth_in_cistern_l3139_313999


namespace consecutive_integers_sum_l3139_313933

theorem consecutive_integers_sum (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end consecutive_integers_sum_l3139_313933


namespace pencil_distribution_ways_l3139_313921

def distribute_pencils (n : ℕ) (k : ℕ) (min_first : ℕ) (min_others : ℕ) : ℕ :=
  Nat.choose (n - (min_first + (k - 1) * min_others) + k - 1) (k - 1)

theorem pencil_distribution_ways : 
  distribute_pencils 8 4 2 1 = 20 := by sorry

end pencil_distribution_ways_l3139_313921


namespace consecutive_color_probability_value_l3139_313964

/-- Represents the number of green chips in the bag -/
def green_chips : ℕ := 4

/-- Represents the number of orange chips in the bag -/
def orange_chips : ℕ := 3

/-- Represents the number of blue chips in the bag -/
def blue_chips : ℕ := 5

/-- Represents the total number of chips in the bag -/
def total_chips : ℕ := green_chips + orange_chips + blue_chips

/-- The probability of drawing all chips such that each color group is drawn consecutively -/
def consecutive_color_probability : ℚ :=
  (Nat.factorial 3 * Nat.factorial green_chips * Nat.factorial orange_chips * Nat.factorial blue_chips) /
  Nat.factorial total_chips

theorem consecutive_color_probability_value :
  consecutive_color_probability = 1 / 4620 := by
  sorry

end consecutive_color_probability_value_l3139_313964


namespace carousel_horse_ratio_l3139_313932

theorem carousel_horse_ratio :
  ∀ (blue purple green gold : ℕ),
    blue = 3 →
    purple = 3 * blue →
    gold = green / 6 →
    blue + purple + green + gold = 33 →
    (green : ℚ) / purple = 2 / 1 :=
by
  sorry

end carousel_horse_ratio_l3139_313932


namespace intersection_of_sets_l3139_313989

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 0, 1, 2}
  let B : Set ℤ := {-2, 1, 2}
  A ∩ B = {1, 2} := by
sorry

end intersection_of_sets_l3139_313989


namespace sum_of_roots_quadratic_l3139_313931

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = 3 := by
  sorry

end sum_of_roots_quadratic_l3139_313931


namespace inscribed_square_area_l3139_313913

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (3, -1)

/-- Predicate to check if a point is on the parabola -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Predicate to check if a point is on the x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- Definition of a square inscribed in the parabola region -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (vertex_on_parabola : on_parabola (center.1 - side_length/2, center.2 - side_length/2))
  (bottom_left_on_x_axis : on_x_axis (center.1 - side_length/2, center.2 + side_length/2))
  (bottom_right_on_x_axis : on_x_axis (center.1 + side_length/2, center.2 + side_length/2))
  (top_right_on_parabola : on_parabola (center.1 + side_length/2, center.2 - side_length/2))

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side_length^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end inscribed_square_area_l3139_313913


namespace complement_union_theorem_l3139_313949

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset ℕ := {1, 2, 4}

-- Define set N
def N : Finset ℕ := {2, 3}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ M) ∪ N = {0, 2, 3} := by sorry

end complement_union_theorem_l3139_313949


namespace emily_beads_count_l3139_313935

/-- Given that Emily makes necklaces where each necklace requires 12 beads,
    and she made 7 necklaces, prove that the total number of beads she had is 84. -/
theorem emily_beads_count :
  let beads_per_necklace : ℕ := 12
  let necklaces_made : ℕ := 7
  beads_per_necklace * necklaces_made = 84 :=
by sorry

end emily_beads_count_l3139_313935


namespace sum_of_powers_of_i_l3139_313915

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 1 + i + i^2 + i^3 = i := by sorry

end sum_of_powers_of_i_l3139_313915


namespace additional_water_for_two_tanks_l3139_313946

/-- Calculates the additional water needed to fill two tanks with equal capacity -/
theorem additional_water_for_two_tanks
  (capacity : ℝ)  -- Capacity of each tank
  (filled1 : ℝ)   -- Amount of water in the first tank
  (filled2 : ℝ)   -- Amount of water in the second tank
  (h1 : filled1 = 300)  -- First tank has 300 liters
  (h2 : filled2 = 450)  -- Second tank has 450 liters
  (h3 : filled2 / capacity = 0.45)  -- Second tank is 45% filled
  : capacity - filled1 + capacity - filled2 = 1250 :=
by sorry

end additional_water_for_two_tanks_l3139_313946


namespace A_inter_B_eq_l3139_313977

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem A_inter_B_eq : A ∩ B = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end A_inter_B_eq_l3139_313977
