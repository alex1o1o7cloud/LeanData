import Mathlib

namespace NUMINAMATH_CALUDE_smallest_x_value_exists_solution_l4047_404708

theorem smallest_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≥ 0 :=
by sorry

theorem exists_solution : 
  ∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_exists_solution_l4047_404708


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l4047_404712

theorem fixed_point_of_function (n : ℤ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => x^n + a^(x-1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l4047_404712


namespace NUMINAMATH_CALUDE_tammy_climbing_speed_l4047_404797

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed :
  ∀ (v : ℝ), -- v represents the speed on the first day
  v > 0 →
  v * 7 + (v + 0.5) * 5 + (v + 1.5) * 8 = 85 →
  7 + 5 + 8 = 20 →
  (v + 0.5) = 4.025 :=
by
  sorry

end NUMINAMATH_CALUDE_tammy_climbing_speed_l4047_404797


namespace NUMINAMATH_CALUDE_rectangle_length_l4047_404713

/-- Given a rectangle with perimeter 700 and breadth 100, its length is 250. -/
theorem rectangle_length (perimeter breadth length : ℝ) : 
  perimeter = 700 →
  breadth = 100 →
  perimeter = 2 * (length + breadth) →
  length = 250 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l4047_404713


namespace NUMINAMATH_CALUDE_max_homework_time_l4047_404754

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Calculates the total time spent on homework given the conditions -/
def total_homework_time (t : HomeworkTime) : ℕ :=
  t.biology + t.history + t.geography

/-- Theorem stating that Max's total homework time is 180 minutes -/
theorem max_homework_time :
  ∀ t : HomeworkTime,
  t.biology = 20 ∧
  t.history = 2 * t.biology ∧
  t.geography = 3 * t.history →
  total_homework_time t = 180 :=
by
  sorry

#check max_homework_time

end NUMINAMATH_CALUDE_max_homework_time_l4047_404754


namespace NUMINAMATH_CALUDE_circular_table_seating_l4047_404717

-- Define the number of people and seats
def total_people : ℕ := 9
def table_seats : ℕ := 7

-- Define the function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n (n - k)) * (Nat.factorial (k - 1))

-- Theorem statement
theorem circular_table_seating :
  seating_arrangements total_people table_seats = 25920 :=
sorry

end NUMINAMATH_CALUDE_circular_table_seating_l4047_404717


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l4047_404753

def expand_polynomial (x : ℝ) : ℝ := (2*x+5)*(3*x^2+x+6) - 4*(x^3-3*x^2+5*x-1)

theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expand_polynomial x = a*x^3 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l4047_404753


namespace NUMINAMATH_CALUDE_existence_of_another_max_sequence_l4047_404714

/-- Represents a sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def countOccurrences (strip : BinarySequence) (seq : BinarySequence) : ℕ := sorry

theorem existence_of_another_max_sequence 
  (n : ℕ) 
  (h_n : n > 5) 
  (strip : BinarySequence) 
  (h_strip : strip.length > n) 
  (M : ℕ) 
  (h_M_max : ∀ seq : BinarySequence, seq.length = n → countOccurrences strip seq ≤ M) 
  (seq_max : BinarySequence) 
  (h_seq_max : seq_max = [true, true] ++ List.replicate (n - 2) false) 
  (h_M_reached : countOccurrences strip seq_max = M) 
  (seq_min : BinarySequence) 
  (h_seq_min : seq_min = List.replicate (n - 2) false ++ [true, true]) 
  (h_min_reached : ∀ seq : BinarySequence, seq.length = n → 
    countOccurrences strip seq ≥ countOccurrences strip seq_min) :
  ∃ (seq : BinarySequence), seq.length = n ∧ seq ≠ seq_max ∧ countOccurrences strip seq = M :=
sorry

end NUMINAMATH_CALUDE_existence_of_another_max_sequence_l4047_404714


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l4047_404794

theorem sqrt_fifth_power_cubed : (((5 : ℝ) ^ (1/2)) ^ 4) ^ (1/2) ^ 3 = 125 := by sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l4047_404794


namespace NUMINAMATH_CALUDE_cone_volume_proof_l4047_404744

noncomputable def cone_volume (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) : ℝ :=
  (Real.sqrt 3 / 3) * Real.pi

theorem cone_volume_proof (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) 
  (h1 : slant_height = 2)
  (h2 : lateral_surface_is_semicircle) :
  cone_volume slant_height lateral_surface_is_semicircle = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_proof_l4047_404744


namespace NUMINAMATH_CALUDE_t_of_f_6_l4047_404798

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)

noncomputable def f (x : ℝ) : ℝ := 6 - t x

theorem t_of_f_6 : t (f 6) = Real.sqrt 26 - 2 := by
  sorry

end NUMINAMATH_CALUDE_t_of_f_6_l4047_404798


namespace NUMINAMATH_CALUDE_original_price_calculation_l4047_404715

/-- Proves that if an article is sold for $130 with a 30% gain, then its original price was $100. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) : 
  sale_price = 130 ∧ gain_percent = 30 → 
  sale_price = (100 : ℝ) * (1 + gain_percent / 100) := by
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l4047_404715


namespace NUMINAMATH_CALUDE_solid_color_not_yellow_percentage_l4047_404730

-- Define the total percentage of marbles
def total_percentage : ℝ := 100

-- Define the percentage of solid color marbles
def solid_color_percentage : ℝ := 90

-- Define the percentage of solid yellow marbles
def solid_yellow_percentage : ℝ := 5

-- Theorem to prove
theorem solid_color_not_yellow_percentage :
  solid_color_percentage - solid_yellow_percentage = 85 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_not_yellow_percentage_l4047_404730


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4047_404781

/-- The parabola y = 2x^2 intersects with the line y = kx + 2 at points A and B.
    M is the midpoint of AB, and N is the foot of the perpendicular from M to the x-axis.
    If the dot product of NA and NB is zero, then k = ±4√3. -/
theorem parabola_line_intersection (k : ℝ) : 
  let C : ℝ → ℝ := λ x => 2 * x^2
  let L : ℝ → ℝ := λ x => k * x + 2
  let A : ℝ × ℝ := (x₁, C x₁)
  let B : ℝ × ℝ := (x₂, C x₂)
  let M : ℝ × ℝ := ((x₁ + x₂)/2, (C x₁ + C x₂)/2)
  let N : ℝ × ℝ := (M.1, 0)
  C x₁ = L x₁ ∧ C x₂ = L x₂ ∧ x₁ ≠ x₂ →
  (A.1 - N.1) * (B.1 - N.1) + (A.2 - N.2) * (B.2 - N.2) = 0 →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4047_404781


namespace NUMINAMATH_CALUDE_total_dog_weight_l4047_404700

/-- The weight of Evan's dog in pounds -/
def evans_dog_weight : ℝ := 63

/-- The ratio of Evan's dog weight to Ivan's dog weight -/
def weight_ratio : ℝ := 7

/-- Theorem: The total weight of Evan's and Ivan's dogs is 72 pounds -/
theorem total_dog_weight : evans_dog_weight + evans_dog_weight / weight_ratio = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_weight_l4047_404700


namespace NUMINAMATH_CALUDE_value_of_expression_l4047_404776

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l4047_404776


namespace NUMINAMATH_CALUDE_T_equals_x_to_fourth_l4047_404742

theorem T_equals_x_to_fourth (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_x_to_fourth_l4047_404742


namespace NUMINAMATH_CALUDE_two_special_right_triangles_l4047_404783

/-- A right-angled triangle with integer sides where the area equals the perimeter -/
structure SpecialRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a^2 + b^2 = c^2  -- Pythagorean theorem
  h2 : a * b = 2 * (a + b + c)  -- Area equals perimeter

/-- The set of all SpecialRightTriangles -/
def specialRightTriangles : Set SpecialRightTriangle :=
  {t : SpecialRightTriangle | True}

theorem two_special_right_triangles :
  ∃ (t1 t2 : SpecialRightTriangle),
    specialRightTriangles = {t1, t2} ∧
    ((t1.a = 5 ∧ t1.b = 12 ∧ t1.c = 13) ∨ (t1.a = 12 ∧ t1.b = 5 ∧ t1.c = 13)) ∧
    ((t2.a = 6 ∧ t2.b = 8 ∧ t2.c = 10) ∨ (t2.a = 8 ∧ t2.b = 6 ∧ t2.c = 10)) :=
  sorry

#check two_special_right_triangles

end NUMINAMATH_CALUDE_two_special_right_triangles_l4047_404783


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l4047_404770

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l4047_404770


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l4047_404775

theorem ordering_of_expressions : 
  Real.exp 0.1 > Real.sqrt 1.2 ∧ Real.sqrt 1.2 > 1 + Real.log 1.1 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l4047_404775


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l4047_404773

/-- The x-coordinate of the intersection point of two linear functions -/
theorem intersection_point_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l4047_404773


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l4047_404764

theorem polynomial_product_expansion (x : ℝ) :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l4047_404764


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4047_404743

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 105 → n = 15 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4047_404743


namespace NUMINAMATH_CALUDE_unique_n_solution_l4047_404719

def is_not_divisible_by_cube_of_prime (x : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^3 ∣ x)

theorem unique_n_solution :
  ∃! n : ℕ, n ≥ 1 ∧
    ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧
      is_not_divisible_by_cube_of_prime (a^2 + b + 3) ∧
      n = (a * b + 3 * b + 8) / (a^2 + b + 3) ∧
      n = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_n_solution_l4047_404719


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l4047_404727

/-- The number of pages that can be copied given the cost per page and available money. -/
def pages_copied (cost_per_page : ℚ) (available_money : ℚ) : ℚ :=
  (available_money * 100) / cost_per_page

/-- Theorem: Given a cost of 5 cents per page and $15 available, 300 pages can be copied. -/
theorem pages_copied_for_fifteen_dollars :
  pages_copied (5 : ℚ) (15 : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l4047_404727


namespace NUMINAMATH_CALUDE_badge_making_contest_tables_l4047_404733

theorem badge_making_contest_tables (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) : 
  stools_per_table = 7 → 
  stool_legs = 4 → 
  table_legs = 5 → 
  total_legs = 658 → 
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    total_legs = stool_legs * stools_per_table * num_tables + table_legs * num_tables :=
by sorry

end NUMINAMATH_CALUDE_badge_making_contest_tables_l4047_404733


namespace NUMINAMATH_CALUDE_problem_statement_l4047_404777

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4047_404777


namespace NUMINAMATH_CALUDE_abhay_speed_l4047_404769

theorem abhay_speed (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) : 
  distance = 24 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 →
  abhay_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_abhay_speed_l4047_404769


namespace NUMINAMATH_CALUDE_xiaojun_original_money_l4047_404705

/-- The amount of money Xiaojun originally had -/
def original_money : ℝ := 30

/-- The daily allowance Xiaojun receives from his dad -/
def daily_allowance : ℝ := 5

/-- The number of days Xiaojun can last when spending 10 yuan per day -/
def days_at_10 : ℝ := 6

/-- The number of days Xiaojun can last when spending 15 yuan per day -/
def days_at_15 : ℝ := 3

/-- The daily spending when Xiaojun lasts for 6 days -/
def spending_10 : ℝ := 10

/-- The daily spending when Xiaojun lasts for 3 days -/
def spending_15 : ℝ := 15

theorem xiaojun_original_money :
  (days_at_10 * spending_10 - days_at_10 * daily_allowance = original_money) ∧
  (days_at_15 * spending_15 - days_at_15 * daily_allowance = original_money) :=
by sorry

end NUMINAMATH_CALUDE_xiaojun_original_money_l4047_404705


namespace NUMINAMATH_CALUDE_lisa_dvd_rental_l4047_404767

theorem lisa_dvd_rental (total_cost : ℚ) (cost_per_dvd : ℚ) (h1 : total_cost = 4.80) (h2 : cost_per_dvd = 1.20) :
  total_cost / cost_per_dvd = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_dvd_rental_l4047_404767


namespace NUMINAMATH_CALUDE_max_profit_transport_plan_l4047_404786

/-- Represents the transportation problem for fruits A, B, and C. -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ
  min_trucks_per_fruit : ℕ

/-- Calculates the profit for a given transportation plan. -/
def calculate_profit (ft : FruitTransport) (x y : ℕ) : ℕ :=
  ft.profit_per_ton_A * ft.tons_per_truck_A * x +
  ft.profit_per_ton_B * ft.tons_per_truck_B * y +
  ft.profit_per_ton_C * ft.tons_per_truck_C * (ft.total_trucks - x - y)

/-- States that the given transportation plan maximizes profit. -/
theorem max_profit_transport_plan (ft : FruitTransport)
  (h_total_trucks : ft.total_trucks = 20)
  (h_total_tons : ft.total_tons = 100)
  (h_tons_A : ft.tons_per_truck_A = 6)
  (h_tons_B : ft.tons_per_truck_B = 5)
  (h_tons_C : ft.tons_per_truck_C = 4)
  (h_profit_A : ft.profit_per_ton_A = 500)
  (h_profit_B : ft.profit_per_ton_B = 600)
  (h_profit_C : ft.profit_per_ton_C = 400)
  (h_min_trucks : ft.min_trucks_per_fruit = 2) :
  ∃ (x y : ℕ),
    x = 2 ∧
    y = 16 ∧
    ft.total_trucks - x - y = 2 ∧
    calculate_profit ft x y = 57200 ∧
    ∀ (x' y' : ℕ),
      x' ≥ ft.min_trucks_per_fruit →
      y' ≥ ft.min_trucks_per_fruit →
      ft.total_trucks - x' - y' ≥ ft.min_trucks_per_fruit →
      calculate_profit ft x' y' ≤ calculate_profit ft x y :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_transport_plan_l4047_404786


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l4047_404772

-- Define the number of grandchildren
def n : ℕ := 12

-- Define the probability of a child being male or female
def p : ℚ := 1/2

-- Define the probability of having an equal number of grandsons and granddaughters
def prob_equal : ℚ := (n.choose (n/2)) / (2^n)

-- Theorem statement
theorem unequal_grandchildren_probability :
  1 - prob_equal = 793/1024 := by sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l4047_404772


namespace NUMINAMATH_CALUDE_space_diagonal_of_rectangular_solid_l4047_404766

theorem space_diagonal_of_rectangular_solid (l w h : ℝ) (hl : l = 12) (hw : w = 4) (hh : h = 3) :
  Real.sqrt (l^2 + w^2 + h^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_space_diagonal_of_rectangular_solid_l4047_404766


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l4047_404771

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l4047_404771


namespace NUMINAMATH_CALUDE_increase_data_effect_l4047_404796

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- Represents the operation of increasing each data point by a fixed value -/
def increase_data (d : DataSet) (inc : ℝ) : DataSet :=
  { average := d.average + inc, variance := d.variance }

/-- Theorem stating the effect of increasing each data point on the average and variance -/
theorem increase_data_effect (d : DataSet) (inc : ℝ) :
  d.average = 2 ∧ d.variance = 3 ∧ inc = 60 →
  (increase_data d inc).average = 62 ∧ (increase_data d inc).variance = 3 := by
  sorry

end NUMINAMATH_CALUDE_increase_data_effect_l4047_404796


namespace NUMINAMATH_CALUDE_second_derivative_at_one_l4047_404780

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

-- State the theorem
theorem second_derivative_at_one (x : ℝ) : 
  (deriv (deriv f)) 1 = 60 := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_one_l4047_404780


namespace NUMINAMATH_CALUDE_line_AB_not_through_point_B_l4047_404707

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition that (a, b) is on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line AB
def line_AB (a b x y : ℝ) : Prop := (2*a - 2)*x + 2*b*y - 3 = 0

-- Theorem statement
theorem line_AB_not_through_point_B (a b : ℝ) (h : M_on_C a b) :
  ¬ line_AB a b (1/2) (1/2) :=
sorry

end NUMINAMATH_CALUDE_line_AB_not_through_point_B_l4047_404707


namespace NUMINAMATH_CALUDE_ratio_problem_l4047_404736

theorem ratio_problem (second_term : ℝ) (ratio_percent : ℝ) (first_term : ℝ) :
  second_term = 25 →
  ratio_percent = 60 →
  first_term / second_term = ratio_percent / 100 →
  first_term = 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4047_404736


namespace NUMINAMATH_CALUDE_systematic_sampling_second_group_l4047_404759

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  (groupNumber - 1) * interval + 1

theorem systematic_sampling_second_group
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h1 : totalStudents = 160)
  (h2 : sampleSize = 20)
  (h3 : systematicSample totalStudents sampleSize 16 = 123) :
  systematicSample totalStudents sampleSize 2 = 11 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_second_group_l4047_404759


namespace NUMINAMATH_CALUDE_inequality_proof_l4047_404722

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : a + b + Real.sqrt 2 * c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4047_404722


namespace NUMINAMATH_CALUDE_tino_jellybeans_l4047_404763

/-- Proves that Tino has 34 jellybeans given the conditions -/
theorem tino_jellybeans (arnold_jellybeans : ℕ) (lee_jellybeans : ℕ) (tino_jellybeans : ℕ)
  (h1 : arnold_jellybeans = 5)
  (h2 : arnold_jellybeans * 2 = lee_jellybeans)
  (h3 : tino_jellybeans = lee_jellybeans + 24) :
  tino_jellybeans = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_jellybeans_l4047_404763


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4047_404760

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4047_404760


namespace NUMINAMATH_CALUDE_vegetable_sale_mass_l4047_404725

theorem vegetable_sale_mass (carrots zucchini broccoli : ℝ) 
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_sale_mass_l4047_404725


namespace NUMINAMATH_CALUDE_max_player_salary_l4047_404710

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 18000 →
  max_total = 1000000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 568000 :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l4047_404710


namespace NUMINAMATH_CALUDE_bucket_fill_theorem_l4047_404711

/-- Given two buckets P and Q, where P has thrice the capacity of Q,
    and P alone takes 60 turns to fill a drum, prove that P and Q together
    take 45 turns to fill the same drum. -/
theorem bucket_fill_theorem (p q : ℕ) (drum : ℕ) : 
  p = 3 * q →  -- Bucket P has thrice the capacity of bucket Q
  60 * p = drum →  -- It takes 60 turns for bucket P to fill the drum
  45 * (p + q) = drum :=  -- It takes 45 turns for both buckets to fill the drum
by sorry

end NUMINAMATH_CALUDE_bucket_fill_theorem_l4047_404711


namespace NUMINAMATH_CALUDE_car_ac_price_difference_l4047_404734

/-- Given that the price of a car and AC are in the ratio 3:2, and the AC costs $1500,
    prove that the car costs $750 more than the AC. -/
theorem car_ac_price_difference :
  ∀ (car_price ac_price : ℕ),
    car_price / ac_price = 3 / 2 →
    ac_price = 1500 →
    car_price - ac_price = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_car_ac_price_difference_l4047_404734


namespace NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l4047_404735

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_25_over_3_l4047_404735


namespace NUMINAMATH_CALUDE_polynomials_common_factor_l4047_404748

def p1 (x : ℝ) : ℝ := 16 * x^5 - x
def p2 (x : ℝ) : ℝ := (x - 1)^2 - 4 * (x - 1) + 4
def p3 (x : ℝ) : ℝ := (x + 1)^2 - 4 * x * (x + 1) + 4 * x^2
def p4 (x : ℝ) : ℝ := -4 * x^2 - 1 + 4 * x

theorem polynomials_common_factor :
  ∃ (f : ℝ → ℝ) (g1 g4 : ℝ → ℝ),
    (∀ x, p1 x = f x * g1 x) ∧
    (∀ x, p4 x = f x * g4 x) ∧
    (∀ x, f x ≠ 0) ∧
    (∀ x, f x ≠ 1) ∧
    (∀ x, f x ≠ -1) ∧
    (∀ (h2 h3 : ℝ → ℝ),
      (∀ x, p2 x ≠ f x * h2 x) ∧
      (∀ x, p3 x ≠ f x * h3 x)) :=
by sorry

end NUMINAMATH_CALUDE_polynomials_common_factor_l4047_404748


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l4047_404762

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equation (x : ℝ) :
  dot_product a (b x) = 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l4047_404762


namespace NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_l4047_404723

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m^2 - m - 2 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

-- Theorem for part (I)
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

-- Theorem for part (II)
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_l4047_404723


namespace NUMINAMATH_CALUDE_total_cookies_kept_l4047_404790

def oatmeal_baked : ℕ := 40
def sugar_baked : ℕ := 28
def chocolate_baked : ℕ := 55

def oatmeal_given : ℕ := 26
def sugar_given : ℕ := 17
def chocolate_given : ℕ := 34

def cookies_kept (baked given : ℕ) : ℕ := baked - given

theorem total_cookies_kept :
  cookies_kept oatmeal_baked oatmeal_given +
  cookies_kept sugar_baked sugar_given +
  cookies_kept chocolate_baked chocolate_given = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_kept_l4047_404790


namespace NUMINAMATH_CALUDE_total_cost_is_54_l4047_404720

/-- The total cost of Léa's purchases -/
def total_cost : ℝ :=
  let book_cost : ℝ := 16
  let binder_cost : ℝ := 2
  let notebook_cost : ℝ := 1
  let pen_cost : ℝ := 0.5
  let calculator_cost : ℝ := 12
  let num_binders : ℕ := 3
  let num_notebooks : ℕ := 6
  let num_pens : ℕ := 4
  let num_calculators : ℕ := 2
  book_cost + 
  (binder_cost * num_binders) + 
  (notebook_cost * num_notebooks) + 
  (pen_cost * num_pens) + 
  (calculator_cost * num_calculators)

theorem total_cost_is_54 : total_cost = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_54_l4047_404720


namespace NUMINAMATH_CALUDE_A_3_2_equals_29_l4047_404757

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_29 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_equals_29_l4047_404757


namespace NUMINAMATH_CALUDE_incorrect_yeast_experiment_method_l4047_404787

/-- Represents an experiment exploring dynamic changes of yeast cell numbers --/
structure YeastExperiment where
  /-- Whether the experiment requires repeated trials --/
  requires_repeated_trials : Bool
  /-- Whether the experiment needs a control group --/
  needs_control_group : Bool

/-- Theorem stating that the incorrect method for yeast cell number experiments 
    is the one claiming no need for repeated trials or control group --/
theorem incorrect_yeast_experiment_method :
  ∀ (e : YeastExperiment), 
    (e.requires_repeated_trials = true) → 
    ¬(e.requires_repeated_trials = false ∧ e.needs_control_group = false) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_yeast_experiment_method_l4047_404787


namespace NUMINAMATH_CALUDE_population_growth_rate_l4047_404782

/-- Calculates the average percent increase per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := (total_increase : ℚ) / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase per year for the given population change is 7%. -/
theorem population_growth_rate : 
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end NUMINAMATH_CALUDE_population_growth_rate_l4047_404782


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l4047_404795

theorem function_inequality_implies_parameter_range :
  ∀ (a : ℝ),
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (|x + a| + |x - 2| ≤ |x - 3|)) →
  (a ∈ Set.Icc (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l4047_404795


namespace NUMINAMATH_CALUDE_jackie_apple_count_l4047_404761

/-- Given that Adam has 9 apples and 3 more apples than Jackie, prove that Jackie has 6 apples. -/
theorem jackie_apple_count (adam_apple_count : ℕ) (adam_extra_apples : ℕ) (jackie_apple_count : ℕ)
  (h1 : adam_apple_count = 9)
  (h2 : adam_apple_count = jackie_apple_count + adam_extra_apples)
  (h3 : adam_extra_apples = 3) :
  jackie_apple_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_jackie_apple_count_l4047_404761


namespace NUMINAMATH_CALUDE_sequences_count_l4047_404755

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of selecting 3 students from a group of 15 students,
    where each student can be selected at most once -/
def num_sequences : ℕ :=
  num_students * (num_students - 1) * (num_students - 2)

theorem sequences_count :
  num_sequences = 2730 := by
  sorry

end NUMINAMATH_CALUDE_sequences_count_l4047_404755


namespace NUMINAMATH_CALUDE_scientific_notation_34_million_l4047_404793

theorem scientific_notation_34_million :
  (34 : ℝ) * 1000000 = 3.4 * (10 : ℝ) ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_34_million_l4047_404793


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l4047_404750

def is_not_divisible_by_five (b : ℤ) : Prop :=
  ¬ (5 ∣ (2 * b^3 - 2 * b^2))

theorem base_b_not_divisible_by_five :
  ∀ b : ℤ, b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
    (is_not_divisible_by_five b ↔ b ∈ ({4, 7, 8} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l4047_404750


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l4047_404758

theorem curve_is_hyperbola (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A - y^2 / B = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l4047_404758


namespace NUMINAMATH_CALUDE_joan_sandwiches_l4047_404740

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  ham : ℕ
  grilledCheese : ℕ

/-- Represents the amount of cheese slices used -/
structure CheeseUsed where
  cheddar : ℕ
  swiss : ℕ
  gouda : ℕ

/-- Calculates the total cheese used for a given number of sandwiches -/
def totalCheeseUsed (s : Sandwiches) : CheeseUsed :=
  { cheddar := s.ham + 2 * s.grilledCheese,
    swiss := s.ham,
    gouda := s.grilledCheese }

/-- The main theorem to prove -/
theorem joan_sandwiches :
  ∃ (s : Sandwiches),
    s.ham = 8 ∧
    totalCheeseUsed s = { cheddar := 40, swiss := 20, gouda := 30 } ∧
    s.grilledCheese = 16 := by
  sorry


end NUMINAMATH_CALUDE_joan_sandwiches_l4047_404740


namespace NUMINAMATH_CALUDE_difference_105th_100th_term_l4047_404728

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem difference_105th_100th_term :
  let a₁ := 3
  let d := 5
  (arithmeticSequence a₁ d 105) - (arithmeticSequence a₁ d 100) = 25 := by
  sorry

end NUMINAMATH_CALUDE_difference_105th_100th_term_l4047_404728


namespace NUMINAMATH_CALUDE_probability_specific_coin_sequence_l4047_404716

/-- The probability of getting a specific sequence of coin flips -/
def probability_specific_sequence (n : ℕ) (p : ℚ) : ℚ :=
  p^n

/-- The number of coin flips -/
def num_flips : ℕ := 10

/-- The probability of getting tails on a single flip -/
def prob_tails : ℚ := 1/2

/-- Theorem: The probability of getting the sequence TTT HHHH THT in 10 coin flips -/
theorem probability_specific_coin_sequence :
  probability_specific_sequence num_flips prob_tails = 1/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_coin_sequence_l4047_404716


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l4047_404774

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q) (h_pos : a 1 > 0) :
  (increasing_sequence a → q > 0) ∧
  ¬(q > 0 → increasing_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l4047_404774


namespace NUMINAMATH_CALUDE_square_division_perimeter_counterexample_l4047_404741

theorem square_division_perimeter_counterexample :
  ∃ (s : ℚ), 
    s > 0 ∧ 
    (∃ (w h : ℚ), w > 0 ∧ h > 0 ∧ w + h = s ∧ (2 * (w + h)).isInt) ∧ 
    ¬(4 * s).isInt :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeter_counterexample_l4047_404741


namespace NUMINAMATH_CALUDE_saucer_surface_area_l4047_404785

/-- The surface area of a saucer with given dimensions -/
theorem saucer_surface_area (radius : ℝ) (rim_thickness : ℝ) (cap_height : ℝ) 
  (h1 : radius = 3)
  (h2 : rim_thickness = 1)
  (h3 : cap_height = 1.5) :
  2 * Real.pi * radius * cap_height + Real.pi * (radius^2 - (radius - rim_thickness)^2) = 14 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_saucer_surface_area_l4047_404785


namespace NUMINAMATH_CALUDE_sports_meeting_participation_l4047_404752

theorem sports_meeting_participation (field_events track_events both : ℕ) 
  (h1 : field_events = 15)
  (h2 : track_events = 13)
  (h3 : both = 5) :
  field_events + track_events - both = 23 :=
by sorry

end NUMINAMATH_CALUDE_sports_meeting_participation_l4047_404752


namespace NUMINAMATH_CALUDE_marikas_fathers_age_l4047_404765

/-- Given that Marika was 10 years old in 2006 and her father's age was five times her age,
    prove that the year when Marika's father's age will be twice her age is 2036. -/
theorem marikas_fathers_age (marika_birth_year : ℕ) (father_birth_year : ℕ) : 
  marika_birth_year = 1996 →
  father_birth_year = 1956 →
  ∃ (year : ℕ), year = 2036 ∧ 
    (year - father_birth_year) = 2 * (year - marika_birth_year) :=
by sorry

end NUMINAMATH_CALUDE_marikas_fathers_age_l4047_404765


namespace NUMINAMATH_CALUDE_fraction_reducibility_implies_determinant_divisibility_l4047_404709

theorem fraction_reducibility_implies_determinant_divisibility
  (a b c d l k : ℤ) 
  (h : ∃ (m n : ℤ), a * l + b = k * m ∧ c * l + d = k * n) :
  k ∣ (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducibility_implies_determinant_divisibility_l4047_404709


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l4047_404702

/-- Given a point A with coordinates (-2,4), this theorem states that the point
    symmetric to A with respect to the y-axis has coordinates (2,4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (- A.1, A.2)
  symmetric_point = (2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l4047_404702


namespace NUMINAMATH_CALUDE_simple_interest_difference_l4047_404792

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal rate time : ℕ) : 
  principal = 2800 → 
  rate = 4 → 
  time = 5 → 
  principal - (principal * rate * time) / 100 = 2240 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l4047_404792


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4047_404706

theorem least_positive_integer_with_remainders : ∃! a : ℕ,
  a > 0 ∧
  a % 2 = 1 ∧
  a % 3 = 2 ∧
  a % 4 = 3 ∧
  a % 5 = 4 ∧
  ∀ b : ℕ, b > 0 ∧ b % 2 = 1 ∧ b % 3 = 2 ∧ b % 4 = 3 ∧ b % 5 = 4 → a ≤ b :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4047_404706


namespace NUMINAMATH_CALUDE_cubic_expression_value_l4047_404718

theorem cubic_expression_value (m n : ℝ) 
  (h1 : m^2 = n + 2) 
  (h2 : n^2 = m + 2) 
  (h3 : m ≠ n) : 
  m^3 - 2*m*n + n^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l4047_404718


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l4047_404747

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.53247247247

/-- The fraction representation we're aiming for -/
def fraction : ℚ := 53171 / 99900

/-- Theorem stating that the decimal equals the fraction -/
theorem decimal_equals_fraction : decimal = fraction := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l4047_404747


namespace NUMINAMATH_CALUDE_squares_concurrency_l4047_404779

-- Define the vertices of the squares as complex numbers
variable (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ)

-- Define the condition for equally oriented squares
def equally_oriented (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (w t : ℂ), Complex.abs w = 1 ∧
    zA₁ = w * zA + t ∧
    zB₁ = w * zB + t ∧
    zC₁ = w * zC + t ∧
    zD₁ = w * zD + t

-- Define the concurrency condition
def concurrent (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (P : ℂ),
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zC₁ - zC) ∧
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zD₁ - zD)

-- State the theorem
theorem squares_concurrency
  (h : equally_oriented zA zB zC zD zA₁ zB₁ zC₁ zD₁) :
  concurrent zA zB zC zD zA₁ zB₁ zC₁ zD₁ :=
by sorry

end NUMINAMATH_CALUDE_squares_concurrency_l4047_404779


namespace NUMINAMATH_CALUDE_sum_outside_angles_inscribed_pentagon_l4047_404749

/-- A pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- Define the circle
  circle : Set (ℝ × ℝ)
  -- Define the pentagon
  pentagon : Set (ℝ × ℝ)
  -- Ensure the pentagon is inscribed in the circle
  is_inscribed : pentagon ⊆ circle

/-- An angle inscribed in a segment outside the pentagon -/
def OutsideAngle (p : InscribedPentagon) : Type :=
  { θ : ℝ // 0 ≤ θ ∧ θ ≤ 2 * Real.pi }

/-- The theorem stating that the sum of angles inscribed in the five segments
    outside an inscribed pentagon is equal to 5π/2 radians (900°) -/
theorem sum_outside_angles_inscribed_pentagon (p : InscribedPentagon) 
  (α β γ δ ε : OutsideAngle p) : 
  α.val + β.val + γ.val + δ.val + ε.val = 5 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_outside_angles_inscribed_pentagon_l4047_404749


namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_four_l4047_404745

-- Define the Fibonacci sequence modulo 7
def fibMod7 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fibMod7 n + fibMod7 (n + 1)) % 7

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod7 k = d

-- Define a function to check if all digits from 0 to 6 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 6 → digitAppeared d n

-- The main theorem
theorem last_digit_to_appear_is_four :
  ∃ n, allDigitsAppeared n ∧ ¬(digitAppeared 4 (n - 1)) :=
sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_four_l4047_404745


namespace NUMINAMATH_CALUDE_alley_width_l4047_404724

/-- The width of a narrow alley given a ladder's length and angles -/
theorem alley_width (b : ℝ) (h_b_pos : b > 0) : ∃ w : ℝ,
  w = b * (1 + Real.sqrt 3) / 2 ∧
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x = b * Real.cos (π / 3) ∧
    y = b * Real.cos (π / 6) ∧
    w = x + y :=
by sorry

end NUMINAMATH_CALUDE_alley_width_l4047_404724


namespace NUMINAMATH_CALUDE_b_investment_is_60000_l4047_404738

/-- Represents the investment and profit sharing structure of a business partnership --/
structure BusinessPartnership where
  total_profit : ℝ
  a_investment : ℝ
  b_investment : ℝ
  a_management_share : ℝ
  a_total_share : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    B's investment is 60,000 --/
theorem b_investment_is_60000 (bp : BusinessPartnership)
  (h1 : bp.total_profit = 8800)
  (h2 : bp.a_investment = 50000)
  (h3 : bp.a_management_share = 0.125 * bp.total_profit)
  (h4 : bp.a_total_share = 4600)
  (h5 : bp.a_total_share = bp.a_management_share +
        (bp.total_profit - bp.a_management_share) * (bp.a_investment / (bp.a_investment + bp.b_investment)))
  : bp.b_investment = 60000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_60000_l4047_404738


namespace NUMINAMATH_CALUDE_max_discount_rate_l4047_404732

theorem max_discount_rate (cost_price : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 →
  original_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 60 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

#check max_discount_rate

end NUMINAMATH_CALUDE_max_discount_rate_l4047_404732


namespace NUMINAMATH_CALUDE_six_digit_permutations_eq_60_l4047_404788

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 6 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 6 is equal to 60 -/
theorem six_digit_permutations_eq_60 : six_digit_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_permutations_eq_60_l4047_404788


namespace NUMINAMATH_CALUDE_basketball_team_selection_l4047_404778

def num_players : ℕ := 12
def team_size : ℕ := 6
def captain_count : ℕ := 1

theorem basketball_team_selection :
  (num_players.choose captain_count) * ((num_players - captain_count).choose (team_size - captain_count)) = 5544 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l4047_404778


namespace NUMINAMATH_CALUDE_fraction_subtraction_l4047_404731

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x^2 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l4047_404731


namespace NUMINAMATH_CALUDE_sandwich_bread_count_l4047_404751

/-- The number of pieces of bread needed for a given number of regular and double meat sandwiches -/
def breadNeeded (regularCount : ℕ) (doubleMeatCount : ℕ) : ℕ :=
  2 * regularCount + 3 * doubleMeatCount

/-- Theorem stating that 14 regular sandwiches and 12 double meat sandwiches require 64 pieces of bread -/
theorem sandwich_bread_count : breadNeeded 14 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_count_l4047_404751


namespace NUMINAMATH_CALUDE_sqrt_simplification_algebraic_simplification_l4047_404789

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 6 * Real.sqrt (2/3) / Real.sqrt 2 = Real.sqrt 2 := by sorry

-- Problem 2
theorem algebraic_simplification :
  (Real.sqrt 2 + Real.sqrt 5)^2 - (Real.sqrt 2 + Real.sqrt 5)*(Real.sqrt 2 - Real.sqrt 5) = 10 + 2*Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_sqrt_simplification_algebraic_simplification_l4047_404789


namespace NUMINAMATH_CALUDE_simplify_radical_product_l4047_404721

theorem simplify_radical_product (y : ℝ) (h : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (27 * y) * Real.sqrt (14 * y) * Real.sqrt (10 * y) = 12 * y * Real.sqrt (1260 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l4047_404721


namespace NUMINAMATH_CALUDE_sin_alpha_fourth_quadrant_l4047_404791

theorem sin_alpha_fourth_quadrant (α : Real) : 
  (π/2 < α ∧ α < 2*π) →  -- α is in the fourth quadrant
  (Real.tan (π - α) = 5/12) → 
  (Real.sin α = -5/13) := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_fourth_quadrant_l4047_404791


namespace NUMINAMATH_CALUDE_peach_distribution_l4047_404729

/-- Proves that given 60 peaches distributed among two equal-sized containers and one smaller container,
    where the smaller container holds half as many peaches as each of the equal-sized containers,
    the number of peaches in the smaller container is 12. -/
theorem peach_distribution (total_peaches : ℕ) (cloth_bag : ℕ) (knapsack : ℕ) : 
  total_peaches = 60 →
  2 * cloth_bag + knapsack = total_peaches →
  knapsack = cloth_bag / 2 →
  knapsack = 12 := by
sorry

end NUMINAMATH_CALUDE_peach_distribution_l4047_404729


namespace NUMINAMATH_CALUDE_sixty_percent_of_40_minus_four_fifths_of_25_l4047_404726

theorem sixty_percent_of_40_minus_four_fifths_of_25 : (60 / 100 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_of_40_minus_four_fifths_of_25_l4047_404726


namespace NUMINAMATH_CALUDE_board_numbers_l4047_404703

theorem board_numbers (N : ℕ) (numbers : Finset ℝ) : 
  (N ≥ 9) →
  (Finset.card numbers = N) →
  (∀ x ∈ numbers, 0 ≤ x ∧ x < 1) →
  (∀ subset : Finset ℝ, subset ⊆ numbers → Finset.card subset = 8 → 
    ∃ y ∈ numbers, y ∉ subset ∧ 
    ∃ z : ℤ, (Finset.sum subset (λ i => i) + y = z)) →
  N = 9 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_l4047_404703


namespace NUMINAMATH_CALUDE_system_of_equations_l4047_404739

theorem system_of_equations (x y a : ℝ) : 
  (3 * x + y = a + 1) → 
  (x + 3 * y = 3) → 
  (x + y > 5) → 
  (a > 16) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l4047_404739


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l4047_404768

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ages of Bipin, Alok, and Chandan -/
structure Ages where
  bipin : Age
  alok : Age
  chandan : Age

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alok.years = 5 ∧
  ages.chandan.years = 10 ∧
  ages.bipin.years + 10 = 2 * (ages.chandan.years + 10)

/-- The theorem to prove -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  (ages.bipin.years : ℚ) / ages.alok.years = 6 / 1 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l4047_404768


namespace NUMINAMATH_CALUDE_problem_statement_l4047_404756

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≠ y)
  (h2 : x^2 * (y + z) = 2019)
  (h3 : y^2 * (z + x) = 2019) :
  z^2 * (x + y) - x * y * z = 4038 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4047_404756


namespace NUMINAMATH_CALUDE_binary_51_l4047_404737

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_51 : toBinary 51 = [true, true, false, false, true, true] := by
  sorry

#eval toBinary 51

end NUMINAMATH_CALUDE_binary_51_l4047_404737


namespace NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l4047_404704

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of 8-digit positive integers with digits in increasing order -/
def M : ℕ := 9 * stars_and_bars 7 10

theorem eight_digit_increasing_remainder :
  M % 1000 = 960 := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l4047_404704


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l4047_404799

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of green balls in the bag -/
def num_green_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_green_balls

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball : ℚ := num_red_balls / total_balls

theorem probability_of_red_ball :
  prob_red_ball = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l4047_404799


namespace NUMINAMATH_CALUDE_no_x_squared_term_l4047_404746

/-- Given an algebraic expression (x^2 + mx)(x - 3), if the simplified form does not contain the term x^2, then m = 3 -/
theorem no_x_squared_term (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x - 3) = x^3 + (m - 3)*x^2 - 3*m*x) →
  (m - 3 = 0) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l4047_404746


namespace NUMINAMATH_CALUDE_product_14_sum_5_or_minus_5_l4047_404701

theorem product_14_sum_5_or_minus_5 (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 14 →
  a + b + c + d = 5 ∨ a + b + c + d = -5 := by
sorry

end NUMINAMATH_CALUDE_product_14_sum_5_or_minus_5_l4047_404701


namespace NUMINAMATH_CALUDE_emily_big_garden_seeds_l4047_404784

def emily_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds :
  emily_garden_problem 42 3 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_emily_big_garden_seeds_l4047_404784
