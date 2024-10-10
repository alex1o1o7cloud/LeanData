import Mathlib

namespace factorization_equality_l1117_111764

theorem factorization_equality (m n : ℝ) : -8*m^2 + 2*m*n = -2*m*(4*m - n) := by
  sorry

end factorization_equality_l1117_111764


namespace janes_daily_vase_arrangement_l1117_111745

def total_vases : ℕ := 248
def last_day_vases : ℕ := 8

theorem janes_daily_vase_arrangement :
  ∃ (daily_vases : ℕ),
    daily_vases > 0 ∧
    daily_vases = last_day_vases ∧
    (total_vases - last_day_vases) % daily_vases = 0 :=
by sorry

end janes_daily_vase_arrangement_l1117_111745


namespace odd_numbers_sum_greater_than_20000_l1117_111709

/-- The count of odd numbers between 200 and 405 whose sum is greater than 20000 -/
def count_odd_numbers_with_large_sum : ℕ :=
  let first_odd := 201
  let last_odd := 403
  let count := (last_odd - first_odd) / 2 + 1
  count

theorem odd_numbers_sum_greater_than_20000 :
  count_odd_numbers_with_large_sum = 102 :=
sorry


end odd_numbers_sum_greater_than_20000_l1117_111709


namespace sum_and_reciprocal_sum_l1117_111768

theorem sum_and_reciprocal_sum (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x)^2 = 10.25 → x + (1/x) = 3.5 := by
  sorry

end sum_and_reciprocal_sum_l1117_111768


namespace equation_has_real_roots_l1117_111778

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end equation_has_real_roots_l1117_111778


namespace constant_expression_l1117_111708

theorem constant_expression (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = 4) 
  (h2 : x * y * z = 6) : 
  (x*y - 3/2*(x+y)) * (y*z - 3/2*(y+z)) * (z*x - 3/2*(z+x)) = 81/4 := by
  sorry

end constant_expression_l1117_111708


namespace degree_of_q_l1117_111776

-- Define polynomials p, q, and i
variable (p q i : Polynomial ℝ)

-- Define the relationship between i, p, and q
def poly_relation (p q i : Polynomial ℝ) : Prop :=
  i = p.comp q ^ 2 - q ^ 3

-- State the theorem
theorem degree_of_q (hp : Polynomial.degree p = 4)
                    (hi : Polynomial.degree i = 12)
                    (h_rel : poly_relation p q i) :
  Polynomial.degree q = 4 := by
  sorry

end degree_of_q_l1117_111776


namespace contrapositive_equivalence_l1117_111729

theorem contrapositive_equivalence (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 4 → x^2 - 7*x + 12 ≠ 0) ↔ (x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) :=
by sorry

end contrapositive_equivalence_l1117_111729


namespace factors_of_1320_l1117_111725

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 32 := by
  sorry

end factors_of_1320_l1117_111725


namespace hot_water_bottle_price_l1117_111769

/-- Proves that the price of a hot-water bottle is 6 dollars given the problem conditions --/
theorem hot_water_bottle_price :
  let thermometer_price : ℚ := 2
  let total_sales : ℚ := 1200
  let thermometer_to_bottle_ratio : ℕ := 7
  let bottles_sold : ℕ := 60
  let thermometers_sold : ℕ := thermometer_to_bottle_ratio * bottles_sold
  let bottle_price : ℚ := (total_sales - (thermometer_price * thermometers_sold)) / bottles_sold
  bottle_price = 6 :=
by sorry

end hot_water_bottle_price_l1117_111769


namespace sum_of_xy_l1117_111739

theorem sum_of_xy (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq : x + y + x * y = 94) : x + y = 22 := by
sorry

end sum_of_xy_l1117_111739


namespace second_agency_daily_charge_correct_l1117_111791

/-- The daily charge of the first agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25

/-- The daily charge of the second agency -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_correct :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end second_agency_daily_charge_correct_l1117_111791


namespace quadratic_sum_of_constants_l1117_111747

/-- The quadratic function f(x) = 15x^2 + 75x + 225 -/
def f (x : ℝ) : ℝ := 15 * x^2 + 75 * x + 225

/-- The constants a, b, and c in the form a(x+b)^2+c -/
def a : ℝ := 15
def b : ℝ := 2.5
def c : ℝ := 131.25

/-- The quadratic function g(x) in the form a(x+b)^2+c -/
def g (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_constants :
  (∀ x, f x = g x) → a + b + c = 148.75 := by sorry

end quadratic_sum_of_constants_l1117_111747


namespace inequality_solution_set_l1117_111736

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end inequality_solution_set_l1117_111736


namespace dave_first_six_l1117_111733

/-- The probability of tossing a six on a single throw -/
def prob_six : ℚ := 1 / 6

/-- The probability of not tossing a six on a single throw -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Dave in each round -/
def players_before_dave : ℕ := 3

/-- The total number of players -/
def total_players : ℕ := 4

/-- The probability that Dave is the first to toss a six -/
theorem dave_first_six : 
  (prob_six * prob_not_six ^ players_before_dave) / 
  (1 - prob_not_six ^ total_players) = 125 / 671 := by
  sorry

end dave_first_six_l1117_111733


namespace students_passing_both_tests_l1117_111766

theorem students_passing_both_tests (total : ℕ) (long_jump : ℕ) (shot_put : ℕ) (failed_both : ℕ) :
  total = 50 →
  long_jump = 40 →
  shot_put = 31 →
  failed_both = 4 →
  ∃ x : ℕ, x = 25 ∧ total = (long_jump - x) + (shot_put - x) + x + failed_both :=
by sorry

end students_passing_both_tests_l1117_111766


namespace incorrect_steps_count_l1117_111785

theorem incorrect_steps_count (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  ∃ (s1 s2 s3 : Prop),
    (s1 ↔ (a * c > b * c ∧ b * c > b * d)) ∧
    (s2 ↔ (a * c > b * c ∧ b * c > b * d → a * c > b * d)) ∧
    (s3 ↔ (a * c > b * d → a / d > b / c)) ∧
    (¬s1 ∧ s2 ∧ ¬s3) :=
by sorry


end incorrect_steps_count_l1117_111785


namespace smallest_dual_base_representation_l1117_111790

def is_valid_base_6_digit (n : ℕ) : Prop := n < 6
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8

def value_in_base_6 (a : ℕ) : ℕ := 6 * a + a
def value_in_base_8 (b : ℕ) : ℕ := 8 * b + b

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ),
    is_valid_base_6_digit a ∧
    is_valid_base_8_digit b ∧
    value_in_base_6 a = 63 ∧
    value_in_base_8 b = 63 ∧
    (∀ (x y : ℕ),
      is_valid_base_6_digit x ∧
      is_valid_base_8_digit y ∧
      value_in_base_6 x = value_in_base_8 y →
      value_in_base_6 x ≥ 63) :=
by sorry

end smallest_dual_base_representation_l1117_111790


namespace x_value_l1117_111711

theorem x_value : ∃ x : ℝ, (x = 90 * (1 + 11/100)) ∧ (x = 99.9) := by sorry

end x_value_l1117_111711


namespace first_valid_year_is_2028_l1117_111757

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2020 ∧ sum_of_digits year = 10

theorem first_valid_year_is_2028 :
  ∀ year : ℕ, year < 2028 → ¬(is_valid_year year) ∧ is_valid_year 2028 :=
sorry

end first_valid_year_is_2028_l1117_111757


namespace cell_division_result_l1117_111796

/-- Represents the cell division process over time -/
def cellDivision (initialOrganisms : ℕ) (initialCellsPerOrganism : ℕ) (divisionRatio : ℕ) (daysBetweenDivisions : ℕ) (totalDays : ℕ) : ℕ :=
  let initialCells := initialOrganisms * initialCellsPerOrganism
  let numDivisions := totalDays / daysBetweenDivisions
  initialCells * divisionRatio ^ numDivisions

/-- Theorem stating the result of the cell division process -/
theorem cell_division_result :
  cellDivision 8 4 3 3 12 = 864 := by
  sorry

end cell_division_result_l1117_111796


namespace absolute_value_inequality_l1117_111777

theorem absolute_value_inequality (a b c : ℝ) :
  |a + c| < b → |a| < |b| - |c| := by
  sorry

end absolute_value_inequality_l1117_111777


namespace quadratic_form_minimum_quadratic_form_minimum_attainable_l1117_111713

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ 8 :=
by sorry

theorem quadratic_form_minimum_attainable :
  ∃ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 = 8 :=
by sorry

end quadratic_form_minimum_quadratic_form_minimum_attainable_l1117_111713


namespace overtime_hours_is_eight_l1117_111756

/-- Represents the payment structure and work hours for a worker --/
structure WorkerPayment where
  ordinary_rate : ℚ  -- Rate for ordinary hours in cents
  overtime_rate : ℚ  -- Rate for overtime hours in cents
  total_hours : ℕ    -- Total hours worked
  total_pay : ℚ      -- Total pay in cents

/-- Calculates the number of overtime hours --/
def calculate_overtime_hours (w : WorkerPayment) : ℚ :=
  (w.total_pay - w.ordinary_rate * w.total_hours) / (w.overtime_rate - w.ordinary_rate)

/-- Theorem stating that under given conditions, the overtime hours are 8 --/
theorem overtime_hours_is_eight :
  let w := WorkerPayment.mk 60 90 50 3240
  calculate_overtime_hours w = 8 := by sorry

end overtime_hours_is_eight_l1117_111756


namespace circle_center_and_radius_l1117_111704

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given equation of the circle --/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 16

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, GivenEquation x y ↔ CircleEquation c x y) ∧
                  c.center = (-4, 2) ∧
                  c.radius = 6 := by
  sorry

end circle_center_and_radius_l1117_111704


namespace max_product_other_sides_l1117_111728

/-- Given a triangle with one side of length 4 and the opposite angle of 60°,
    the maximum product of the lengths of the other two sides is 16. -/
theorem max_product_other_sides (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  A = π / 3 →
  0 < b ∧ 0 < c →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c ≤ 16 :=
sorry

end max_product_other_sides_l1117_111728


namespace fruit_shop_apples_l1117_111781

theorem fruit_shop_apples (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (4 : ℚ) / 10 * total = 140) : total = 200 := by
  sorry

end fruit_shop_apples_l1117_111781


namespace part_one_part_two_l1117_111705

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := |x - c|

-- Part I: Prove that f(x) + f(-1/x) ≥ 2 for any real x and c
theorem part_one (c : ℝ) (x : ℝ) : f c x + f c (-1/x) ≥ 2 :=
sorry

-- Part II: Prove that for c = 4, the solution set of |f(1/2x+c) - 1/2f(x)| ≤ 1 is {x | 1 ≤ x ≤ 3}
theorem part_two :
  let c : ℝ := 4
  ∀ x : ℝ, |f c (1/2 * x + c) - 1/2 * f c x| ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 :=
sorry

end part_one_part_two_l1117_111705


namespace divisibility_condition_l1117_111793

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end divisibility_condition_l1117_111793


namespace sally_buttons_l1117_111734

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday tuesday wednesday buttons_per_shirt : ℕ) : ℕ :=
  (monday + tuesday + wednesday) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all shirts -/
theorem sally_buttons : total_buttons 4 3 2 5 = 45 := by
  sorry

end sally_buttons_l1117_111734


namespace range_of_a_l1117_111754

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → 
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end range_of_a_l1117_111754


namespace complex_fraction_opposite_parts_l1117_111784

theorem complex_fraction_opposite_parts (b : ℝ) : 
  let z₁ : ℂ := 1 + b * I
  let z₂ : ℂ := -2 + I
  (((z₁ / z₂).re = -(z₁ / z₂).im) → b = -1/3) ∧ 
  (b = -1/3 → (z₁ / z₂).re = -(z₁ / z₂).im) := by
sorry

end complex_fraction_opposite_parts_l1117_111784


namespace sqrt_two_nine_two_equals_six_l1117_111738

theorem sqrt_two_nine_two_equals_six : Real.sqrt (2 * 9 * 2) = 6 := by
  sorry

end sqrt_two_nine_two_equals_six_l1117_111738


namespace triangle_side_length_l1117_111787

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Ensure positive side lengths
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a + b + c = A + B + C →  -- Triangle angle sum theorem
  a / Real.sin A = b / Real.sin B →  -- Sine rule
  a = 3 := by
sorry

end triangle_side_length_l1117_111787


namespace set_intersection_equality_l1117_111761

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : M ∩ N = intersection := by
  sorry

end set_intersection_equality_l1117_111761


namespace integral_of_f_equals_seven_sixths_l1117_111742

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := f'₁ * x^2 + x + 1

-- State the theorem
theorem integral_of_f_equals_seven_sixths :
  ∃ (f'₁ : ℝ), (∫ x in (0:ℝ)..(1:ℝ), f x f'₁) = 7/6 := by
  sorry

end integral_of_f_equals_seven_sixths_l1117_111742


namespace slope_intercept_sum_l1117_111727

/-- Given points P, Q, R in a plane, and G as the midpoint of PQ,
    prove that the sum of the slope and y-intercept of line RG is 9/2. -/
theorem slope_intercept_sum (P Q R G : ℝ × ℝ) : 
  P = (0, 10) →
  Q = (0, 0) →
  R = (10, 0) →
  G = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  let slope := (G.2 - R.2) / (G.1 - R.1)
  let y_intercept := G.2 - slope * G.1
  slope + y_intercept = 9/2 := by
sorry

end slope_intercept_sum_l1117_111727


namespace largest_integer_in_interval_l1117_111712

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/12 ∧ 
  ∀ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/6 → (z : ℚ)/6 < 7/12 → z ≤ y :=
by sorry

end largest_integer_in_interval_l1117_111712


namespace sum_of_powers_of_i_is_zero_l1117_111792

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 -/
theorem sum_of_powers_of_i_is_zero :
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 := by sorry

end sum_of_powers_of_i_is_zero_l1117_111792


namespace jacks_estimate_is_larger_l1117_111744

theorem jacks_estimate_is_larger (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (ha : a > 0) (hb : b > 0) : 
  (x + a) - (y - b) > x - y := by
  sorry

end jacks_estimate_is_larger_l1117_111744


namespace fourth_term_of_specific_gp_l1117_111759

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_specific_gp :
  let a₁ := 2
  let a₂ := 2 * Real.sqrt 2
  let a₃ := 4
  let r := a₂ / a₁
  let a₄ := geometric_progression a₁ r 4
  a₄ = 4 * Real.sqrt 2 := by sorry

end fourth_term_of_specific_gp_l1117_111759


namespace equation_equality_l1117_111720

theorem equation_equality : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end equation_equality_l1117_111720


namespace e₁_e₂_form_basis_l1117_111750

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

def are_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def form_basis (v w : ℝ × ℝ) : Prop :=
  ¬(are_collinear v w)

theorem e₁_e₂_form_basis : form_basis e₁ e₂ := by
  sorry

end e₁_e₂_form_basis_l1117_111750


namespace movie_ticket_theorem_l1117_111758

def movie_ticket_problem (child_ticket_price adult_ticket_price : ℚ) : Prop :=
  let total_spent : ℚ := 30
  let num_child_tickets : ℕ := 4
  let num_adult_tickets : ℕ := 2
  let discount : ℚ := 2
  child_ticket_price = 4.25 ∧
  adult_ticket_price > child_ticket_price ∧
  num_child_tickets + num_adult_tickets > 3 ∧
  num_child_tickets * child_ticket_price + num_adult_tickets * adult_ticket_price - discount = total_spent ∧
  adult_ticket_price - child_ticket_price = 3.25

theorem movie_ticket_theorem :
  ∃ (adult_ticket_price : ℚ), movie_ticket_problem 4.25 adult_ticket_price :=
sorry

end movie_ticket_theorem_l1117_111758


namespace toy_store_shelves_l1117_111788

/-- Calculates the number of shelves needed for a given number of items and shelf capacity -/
def shelves_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Proves that the total number of shelves needed for bears and rabbits is 6 -/
theorem toy_store_shelves : 
  let initial_bears : ℕ := 17
  let initial_rabbits : ℕ := 20
  let new_bears : ℕ := 10
  let new_rabbits : ℕ := 15
  let sold_bears : ℕ := 5
  let sold_rabbits : ℕ := 7
  let bear_shelf_capacity : ℕ := 9
  let rabbit_shelf_capacity : ℕ := 12
  let remaining_bears : ℕ := initial_bears + new_bears - sold_bears
  let remaining_rabbits : ℕ := initial_rabbits + new_rabbits - sold_rabbits
  let bear_shelves : ℕ := shelves_needed remaining_bears bear_shelf_capacity
  let rabbit_shelves : ℕ := shelves_needed remaining_rabbits rabbit_shelf_capacity
  bear_shelves + rabbit_shelves = 6 :=
by sorry

end toy_store_shelves_l1117_111788


namespace cookies_prepared_l1117_111762

theorem cookies_prepared (cookies_per_guest : ℕ) (num_guests : ℕ) (total_cookies : ℕ) :
  cookies_per_guest = 19 →
  num_guests = 2 →
  total_cookies = cookies_per_guest * num_guests →
  total_cookies = 38 := by
sorry

end cookies_prepared_l1117_111762


namespace arithmetic_sequence_problem_l1117_111743

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (seq : ℕ → ℝ) :
  (∃ a₁ d : ℝ, seq = ArithmeticSequence a₁ d ∧ 
    seq 3 = 14 ∧ seq 6 = 32) →
  seq 10 = 56 ∧ (∃ d : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = d ∧ d = 6) :=
by
  sorry


end arithmetic_sequence_problem_l1117_111743


namespace fraction_inequality_l1117_111710

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / ((a - c)^2) > e / ((b - d)^2) := by
  sorry

end fraction_inequality_l1117_111710


namespace ellipse_major_axis_length_l1117_111799

/-- The length of the major axis of the ellipse x²/5 + y²/2 = 1 is 2√5 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 5 + y^2 / 2 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 2 * Real.sqrt 5 :=
by sorry

end ellipse_major_axis_length_l1117_111799


namespace fault_line_exists_l1117_111723

/-- Represents a 6x6 grid covered by 18 1x2 dominoes -/
structure DominoCoveredGrid :=
  (grid : Fin 6 → Fin 6 → Bool)
  (dominoes : Fin 18 → (Fin 6 × Fin 6) × (Fin 6 × Fin 6))
  (cover_complete : ∀ i j, ∃ k, (dominoes k).1 = (i, j) ∨ (dominoes k).2 = (i, j))
  (domino_size : ∀ k, 
    ((dominoes k).1.1 = (dominoes k).2.1 ∧ (dominoes k).2.2 = (dominoes k).1.2.succ) ∨
    ((dominoes k).1.2 = (dominoes k).2.2 ∧ (dominoes k).2.1 = (dominoes k).1.1.succ))

/-- A fault line is a row or column that doesn't intersect any domino -/
def has_fault_line (g : DominoCoveredGrid) : Prop :=
  (∃ i : Fin 6, ∀ k, (g.dominoes k).1.1 ≠ i ∧ (g.dominoes k).2.1 ≠ i) ∨
  (∃ j : Fin 6, ∀ k, (g.dominoes k).1.2 ≠ j ∧ (g.dominoes k).2.2 ≠ j)

/-- Theorem: Every 6x6 grid covered by 18 1x2 dominoes has a fault line -/
theorem fault_line_exists (g : DominoCoveredGrid) : has_fault_line g :=
sorry

end fault_line_exists_l1117_111723


namespace choose_four_from_six_l1117_111798

theorem choose_four_from_six : Nat.choose 6 4 = 15 := by
  sorry

end choose_four_from_six_l1117_111798


namespace range_of_a_l1117_111703

def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

theorem range_of_a (a : ℝ) : (A ∪ B a = Set.univ) → a ∈ Set.Icc (-3) 1 := by
  sorry

end range_of_a_l1117_111703


namespace pizza_toppings_combinations_l1117_111726

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end pizza_toppings_combinations_l1117_111726


namespace pond_volume_l1117_111771

/-- The volume of a rectangular prism with given dimensions is 1000 cubic meters. -/
theorem pond_volume (length width depth : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : depth = 5) :
  length * width * depth = 1000 :=
by sorry

end pond_volume_l1117_111771


namespace sin_cos_product_l1117_111714

theorem sin_cos_product (θ : Real) (h : Real.tan (θ + Real.pi / 2) = 2) :
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end sin_cos_product_l1117_111714


namespace nimathur_prime_l1117_111748

/-- Definition of a-nimathur -/
def is_a_nimathur (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ ∀ n : ℕ, n ≥ b / a →
    (a * n + 1) ∣ (Nat.choose (a * n) b - 1)

/-- Main theorem -/
theorem nimathur_prime (a b : ℕ) :
  is_a_nimathur a b ∧ ¬is_a_nimathur a (b + 2) → Nat.Prime (b + 1) :=
by sorry

end nimathur_prime_l1117_111748


namespace sqrt_yz_times_sqrt_xy_l1117_111773

theorem sqrt_yz_times_sqrt_xy (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  Real.sqrt (y * z) * Real.sqrt (x * y) = 4 * Real.sqrt 15 := by
  sorry

end sqrt_yz_times_sqrt_xy_l1117_111773


namespace final_sum_after_operations_l1117_111740

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end final_sum_after_operations_l1117_111740


namespace table_runners_area_l1117_111753

theorem table_runners_area (table_area : ℝ) (covered_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) :
  table_area = 175 →
  covered_percentage = 0.8 →
  two_layer_area = 24 →
  three_layer_area = 24 →
  ∃ (total_area : ℝ), total_area = 188 ∧ 
    total_area = (covered_percentage * table_area - 2 * three_layer_area - two_layer_area) + 
                 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end table_runners_area_l1117_111753


namespace sqrt_meaningful_range_l1117_111789

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end sqrt_meaningful_range_l1117_111789


namespace license_plate_difference_l1117_111752

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible New York license plates -/
def new_york_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of license plates between California and New York -/
theorem license_plate_difference :
  california_plates - new_york_plates = 28121600 := by
  sorry

end license_plate_difference_l1117_111752


namespace simplify_trig_expression_l1117_111735

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end simplify_trig_expression_l1117_111735


namespace arithmetic_sequence_a3_value_l1117_111706

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a3_value
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 2 * a 3 + 1)
  (h_a4 : a 4 = 2 * a 3 + 7) :
  a 3 = -4 :=
sorry

end arithmetic_sequence_a3_value_l1117_111706


namespace rhombus_longer_diagonal_l1117_111786

/-- Given a rhombus with diagonal ratio 2:3 and area 12 cm², prove the longer diagonal is 6 cm -/
theorem rhombus_longer_diagonal (d1 d2 : ℝ) : 
  d1 / d2 = 2 / 3 →  -- ratio of diagonals
  d1 * d2 / 2 = 12 →  -- area of rhombus
  d2 = 6 := by
  sorry

end rhombus_longer_diagonal_l1117_111786


namespace interest_rate_is_five_percent_l1117_111765

/-- Calculates the interest rate given the principal, time, and simple interest -/
def calculate_interest_rate (principal time simple_interest : ℚ) : ℚ :=
  (simple_interest * 100) / (principal * time)

/-- Proof that the interest rate is 5% given the specified conditions -/
theorem interest_rate_is_five_percent :
  let principal : ℚ := 16065
  let time : ℚ := 5
  let simple_interest : ℚ := 4016.25
  calculate_interest_rate principal time simple_interest = 5 := by
  sorry

#eval calculate_interest_rate 16065 5 4016.25

end interest_rate_is_five_percent_l1117_111765


namespace fraction_sum_equality_l1117_111731

theorem fraction_sum_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  x / y + y / x = (x^2 * y^2 - 4 * x * y + 1) / (x * y) := by
  sorry

end fraction_sum_equality_l1117_111731


namespace system_of_equations_solution_l1117_111746

theorem system_of_equations_solution :
  ∃ (x y : ℝ), 2*x - 3*y = -5 ∧ 5*x - 2*y = 4 :=
by
  use 2, 3
  sorry

end system_of_equations_solution_l1117_111746


namespace rainy_days_count_l1117_111749

theorem rainy_days_count (n : ℤ) (R : ℕ) (NR : ℕ) : 
  n * R + 3 * NR = 26 →  -- Total cups equation
  3 * NR - n * R = 10 →  -- Difference in cups equation
  R + NR = 7 →           -- Total days equation
  R = 1 :=                -- Conclusion: 1 rainy day
by sorry

end rainy_days_count_l1117_111749


namespace grape_rate_calculation_l1117_111763

theorem grape_rate_calculation (grape_quantity : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  mango_quantity = 9 →
  mango_rate = 45 →
  total_paid = 965 →
  ∃ (grape_rate : ℕ), grape_rate * grape_quantity + mango_rate * mango_quantity = total_paid ∧ grape_rate = 70 := by
  sorry

end grape_rate_calculation_l1117_111763


namespace value_of_fraction_l1117_111715

theorem value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x/y + y/x = 4) :
  (x + 2*y) / (x - 2*y) = Real.sqrt 33 / 3 := by
  sorry

end value_of_fraction_l1117_111715


namespace journey_time_difference_l1117_111716

/-- Represents the speed of the bus in miles per hour -/
def speed : ℝ := 60

/-- Represents the distance of the first journey in miles -/
def distance1 : ℝ := 360

/-- Represents the distance of the second journey in miles -/
def distance2 : ℝ := 420

/-- Theorem stating the difference in travel time between the two journeys -/
theorem journey_time_difference : 
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end journey_time_difference_l1117_111716


namespace jeff_purchases_total_l1117_111724

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ↑(⌊x⌋) < 1/2 then ⌊x⌋ else ⌈x⌉

theorem jeff_purchases_total :
  let purchase1 : ℚ := 245/100
  let purchase2 : ℚ := 375/100
  let purchase3 : ℚ := 856/100
  let discount : ℚ := 50/100
  round_to_nearest_dollar purchase1 +
  round_to_nearest_dollar purchase2 +
  round_to_nearest_dollar (purchase3 - discount) = 14 := by
  sorry

end jeff_purchases_total_l1117_111724


namespace factorization_ax2_minus_a_l1117_111730

theorem factorization_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end factorization_ax2_minus_a_l1117_111730


namespace tangent_line_cubic_l1117_111737

/-- The equation of the tangent line to y = x³ at (2, 8) is y = 12x - 16 -/
theorem tangent_line_cubic (x y : ℝ) :
  (y = x^3) →  -- curve equation
  (∃ (m b : ℝ), y - 8 = m * (x - 2) ∧ y = m * x + b) →  -- point-slope form of tangent line
  (y = 12 * x - 16) :=
by sorry

end tangent_line_cubic_l1117_111737


namespace guys_age_proof_l1117_111794

theorem guys_age_proof :
  ∃ (age : ℕ), 
    (((age + 8) * 8 - (age - 8) * 8) / 2 = age) ∧ 
    (age = 64) := by
  sorry

end guys_age_proof_l1117_111794


namespace max_d_is_zero_l1117_111755

/-- Represents a 6-digit number of the form 6d6,33e -/
def SixDigitNumber (d e : Nat) : Nat :=
  606330 + d * 1000 + e

theorem max_d_is_zero :
  ∀ d e : Nat,
    d < 10 →
    e < 10 →
    SixDigitNumber d e % 33 = 0 →
    d ≤ 0 :=
by sorry

end max_d_is_zero_l1117_111755


namespace quadratic_real_roots_condition_l1117_111795

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_condition_l1117_111795


namespace triangle_formation_l1117_111702

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 2 4) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 3 4 8) ∧
  ¬(can_form_triangle 4 5 10) :=
sorry

end triangle_formation_l1117_111702


namespace share_face_value_l1117_111779

/-- Given a share with the following properties:
  * dividend_rate: The dividend rate of the share (9%)
  * desired_return: The desired return on investment (12%)
  * market_value: The market value of the share in Rs. (15)
  
  This theorem proves that the face value of the share is Rs. 20. -/
theorem share_face_value
  (dividend_rate : ℝ)
  (desired_return : ℝ)
  (market_value : ℝ)
  (h1 : dividend_rate = 0.09)
  (h2 : desired_return = 0.12)
  (h3 : market_value = 15) :
  (desired_return * market_value) / dividend_rate = 20 := by
  sorry

#eval (0.12 * 15) / 0.09  -- Expected output: 20

end share_face_value_l1117_111779


namespace chord_equation_l1117_111774

/-- Given a circle with equation x^2 + y^2 = 9 and a chord PQ with midpoint (1, 2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} → 
    (P ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} ∧ 
     Q ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9})) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ 
                  a * Q.1 + b * Q.2 + c = 0 ∧
                  a = 1 ∧ b = 2 ∧ c = -5 :=
by sorry

end chord_equation_l1117_111774


namespace divisibility_by_240_l1117_111751

theorem divisibility_by_240 (a b c d : ℕ) : 
  240 ∣ (a^(4*b+d) - a^(4*c+d)) := by sorry

end divisibility_by_240_l1117_111751


namespace friends_contribution_impossibility_l1117_111783

theorem friends_contribution_impossibility (a b c d e : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 →
  a + b + c + d + e > 0 →
  a + b < (a + b + c + d + e) / 3 →
  b + c < (a + b + c + d + e) / 3 →
  c + d < (a + b + c + d + e) / 3 →
  d + e < (a + b + c + d + e) / 3 →
  e + a < (a + b + c + d + e) / 3 →
  False :=
by sorry

end friends_contribution_impossibility_l1117_111783


namespace complex_modulus_problem_l1117_111718

theorem complex_modulus_problem (z : ℂ) (h : z^2 = -4) : 
  Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l1117_111718


namespace sum_of_solutions_square_equation_l1117_111780

theorem sum_of_solutions_square_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 49 ∧ (x₂ - 8)^2 = 49 ∧ x₁ + x₂ = 16 := by
  sorry

end sum_of_solutions_square_equation_l1117_111780


namespace largest_n_for_product_l1117_111700

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product (a b : ℕ → ℤ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 1764) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) :=
by sorry

end largest_n_for_product_l1117_111700


namespace largest_number_proof_l1117_111797

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ k : ℕ, k ∣ a → k ∣ b → k ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ k : ℕ, a ∣ k → b ∣ k → l ∣ k

theorem largest_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  is_hcf a b 23 → (∃ l : ℕ, is_lcm a b l ∧ 13 ∣ l ∧ 14 ∣ l) → max a b = 322 :=
by sorry

end largest_number_proof_l1117_111797


namespace problem_1_problem_2_l1117_111722

-- Problem 1
theorem problem_1 : (-1)^4 + (1 - 1/2) / 3 * (2 - 2^3) = 2 := by
  sorry

-- Problem 2
theorem problem_2 : (-3/4 - 5/9 + 7/12) / (1/36) = -26 := by
  sorry

end problem_1_problem_2_l1117_111722


namespace sorcerer_elixir_combinations_l1117_111717

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals that are incompatible with some herbs. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs that each incompatible crystal cannot be used with. -/
def num_incompatible_herbs_per_crystal : ℕ := 3

/-- The total number of valid combinations for the sorcerer's elixir. -/
def valid_combinations : ℕ := 18

theorem sorcerer_elixir_combinations :
  (num_herbs * num_crystals) - (num_incompatible_crystals * num_incompatible_herbs_per_crystal) = valid_combinations :=
by sorry

end sorcerer_elixir_combinations_l1117_111717


namespace solve_fraction_equation_l1117_111732

theorem solve_fraction_equation :
  ∀ x : ℚ, (1 / 4 : ℚ) - (1 / 6 : ℚ) = 1 / x → x = 12 := by
  sorry

end solve_fraction_equation_l1117_111732


namespace rhombus_other_diagonal_l1117_111721

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- Theorem: In a rhombus with one diagonal of 20 cm and an area of 250 cm², the other diagonal is 25 cm -/
theorem rhombus_other_diagonal
  (r : Rhombus)
  (h1 : r.diagonal1 = 20)
  (h2 : r.area = 250)
  (h3 : r.area = r.diagonal1 * r.diagonal2 / 2) :
  r.diagonal2 = 25 := by
  sorry

end rhombus_other_diagonal_l1117_111721


namespace baker_cakes_problem_l1117_111782

theorem baker_cakes_problem (pastries_made : ℕ) (pastries_sold : ℕ) (cakes_sold : ℕ) :
  pastries_made = 114 →
  pastries_sold = 154 →
  cakes_sold = 78 →
  pastries_sold = cakes_sold + 76 →
  cakes_sold = 78 :=
by sorry

end baker_cakes_problem_l1117_111782


namespace count_primes_with_squares_between_5000_and_8000_l1117_111741

theorem count_primes_with_squares_between_5000_and_8000 :
  (Finset.filter (fun p => 5000 < p^2 ∧ p^2 < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card = 5 := by
  sorry

end count_primes_with_squares_between_5000_and_8000_l1117_111741


namespace unique_divisor_with_remainder_l1117_111707

theorem unique_divisor_with_remainder (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ (145 % d = 4) → d = 47 := by
  sorry

end unique_divisor_with_remainder_l1117_111707


namespace magician_trick_possible_magician_trick_smallest_l1117_111760

/-- Represents a sequence of digits -/
def DigitSequence (n : ℕ) := Fin n → Fin 10

/-- Represents a pair of adjacent positions in a sequence -/
structure AdjacentPair (n : ℕ) where
  first : Fin n
  second : Fin n
  adjacent : second = first.succ

/-- 
Given a sequence of digits and a pair of adjacent positions,
returns the sequence with those positions covered
-/
def coverDigits (seq : DigitSequence n) (pair : AdjacentPair n) : 
  Fin (n - 2) → Fin 10 := sorry

/-- 
States that for any sequence of 101 digits, covering any two adjacent digits
still allows for unique determination of the original sequence
-/
theorem magician_trick_possible : 
  ∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
  ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair :=
sorry

/-- 
States that 101 is the smallest number for which the magician's trick is always possible
-/
theorem magician_trick_smallest : 
  (∀ n < 101, ¬(∀ (seq : DigitSequence n) (pair : AdjacentPair n),
    ∃! (original : DigitSequence n), coverDigits original pair = coverDigits seq pair)) ∧
  (∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
    ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair) :=
sorry

end magician_trick_possible_magician_trick_smallest_l1117_111760


namespace square_difference_l1117_111719

theorem square_difference (x : ℤ) (h : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end square_difference_l1117_111719


namespace beacon_population_l1117_111772

def richmond_population : ℕ := 3000
def richmond_victoria_difference : ℕ := 1000
def victoria_beacon_ratio : ℕ := 4

theorem beacon_population : 
  ∃ (victoria_population beacon_population : ℕ),
    richmond_population = victoria_population + richmond_victoria_difference ∧
    victoria_population = victoria_beacon_ratio * beacon_population ∧
    beacon_population = 500 := by
  sorry

end beacon_population_l1117_111772


namespace tims_income_percentage_l1117_111701

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : mart = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end tims_income_percentage_l1117_111701


namespace bench_press_calculation_l1117_111775

theorem bench_press_calculation (initial_weight : ℝ) (injury_decrease : ℝ) (training_increase : ℝ) : 
  initial_weight = 500 →
  injury_decrease = 0.8 →
  training_increase = 3 →
  (initial_weight * (1 - injury_decrease) * training_increase) = 300 := by
sorry

end bench_press_calculation_l1117_111775


namespace fraction_equality_l1117_111767

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 := by
  sorry

end fraction_equality_l1117_111767


namespace maggie_bouncy_balls_l1117_111770

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 160 := by
  sorry

end maggie_bouncy_balls_l1117_111770
