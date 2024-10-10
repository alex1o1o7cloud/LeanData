import Mathlib

namespace stratified_sampling_female_count_l1883_188334

theorem stratified_sampling_female_count 
  (total_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : female_students = 800)
  (h3 : sample_size = 50) :
  (sample_size * female_students) / total_students = 20 := by
  sorry

end stratified_sampling_female_count_l1883_188334


namespace derivative_y_l1883_188395

def y (x : ℝ) : ℝ := x^2 - 5*x + 4

theorem derivative_y (x : ℝ) : 
  deriv y x = 2*x - 5 := by sorry

end derivative_y_l1883_188395


namespace angle_B_is_pi_over_3_max_area_is_3_sqrt_3_l1883_188357

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B

-- Theorem for part 1
theorem angle_B_is_pi_over_3 (t : Triangle) (h : condition t) : t.B = π / 3 :=
sorry

-- Theorem for part 2
theorem max_area_is_3_sqrt_3 (t : Triangle) (h1 : condition t) (h2 : t.b = 2 * Real.sqrt 3) :
  (∀ s : Triangle, condition s → s.b = 2 * Real.sqrt 3 → 
    1/2 * s.a * s.c * Real.sin s.B ≤ 3 * Real.sqrt 3) ∧ 
  (∃ s : Triangle, condition s ∧ s.b = 2 * Real.sqrt 3 ∧ 
    1/2 * s.a * s.c * Real.sin s.B = 3 * Real.sqrt 3) :=
sorry

end angle_B_is_pi_over_3_max_area_is_3_sqrt_3_l1883_188357


namespace pascal_triangle_value_l1883_188346

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : Nat := 47

/-- The position of the number we're looking for in the row (1-indexed) -/
def target_position : Nat := 45

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : Nat := row_length - 1

/-- The binomial coefficient we need to calculate -/
def pascal_number : Nat := Nat.choose row_number (target_position - 1)

theorem pascal_triangle_value : pascal_number = 1035 := by sorry

end pascal_triangle_value_l1883_188346


namespace simplify_expression_l1883_188368

theorem simplify_expression (x : ℝ) : 3 * x^2 - 1 - 2*x - 5 + 3*x - x = 3 * x^2 - 6 := by
  sorry

end simplify_expression_l1883_188368


namespace min_value_of_function_l1883_188391

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  2 * x + 1 / x^6 ≥ 3 ∧ ∃ y > 0, 2 * y + 1 / y^6 = 3 :=
sorry

end min_value_of_function_l1883_188391


namespace large_cube_volume_l1883_188365

/-- The volume of a cube constructed from smaller cubes -/
theorem large_cube_volume (n : ℕ) (edge : ℝ) (h : n = 125) (h_edge : edge = 2) :
  (n : ℝ) * (edge ^ 3) = 1000 := by
  sorry

end large_cube_volume_l1883_188365


namespace paddyfield_warbler_percentage_l1883_188312

/-- Represents the composition of birds in a nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawk_percent : ℝ
  other_percent : ℝ
  kingfisher_to_warbler_ratio : ℝ

/-- Theorem about the percentage of paddyfield-warblers among non-hawks -/
theorem paddyfield_warbler_percentage
  (pop : BirdPopulation)
  (h1 : pop.hawk_percent = 0.3)
  (h2 : pop.other_percent = 0.35)
  (h3 : pop.kingfisher_to_warbler_ratio = 0.25)
  : (((1 - pop.hawk_percent - pop.other_percent) * pop.total) / ((1 - pop.hawk_percent) * pop.total)) = 0.4 := by
  sorry

end paddyfield_warbler_percentage_l1883_188312


namespace number_wall_solution_l1883_188314

/-- Represents a number wall with the given base numbers -/
structure NumberWall (m : ℤ) :=
  (base : Fin 4 → ℤ)
  (base_values : base 0 = m ∧ base 1 = 6 ∧ base 2 = -3 ∧ base 3 = 4)

/-- Calculates the value at the top of the number wall -/
def top_value (w : NumberWall m) : ℤ :=
  let level1_0 := w.base 0 + w.base 1
  let level1_1 := w.base 1 + w.base 2
  let level1_2 := w.base 2 + w.base 3
  let level2_0 := level1_0 + level1_1
  let level2_1 := level1_1 + level1_2
  level2_0 + level2_1

/-- The theorem to be proved -/
theorem number_wall_solution (m : ℤ) (w : NumberWall m) :
  top_value w = 20 → m = 7 := by sorry

end number_wall_solution_l1883_188314


namespace divisibility_by_two_in_odd_base_system_l1883_188390

theorem divisibility_by_two_in_odd_base_system (d : ℕ) (h_odd : Odd d) :
  ∀ (x : ℕ) (digits : List ℕ),
    (x = digits.foldr (λ a acc => a + d * acc) 0) →
    (x % 2 = 0 ↔ digits.sum % 2 = 0) := by
  sorry

end divisibility_by_two_in_odd_base_system_l1883_188390


namespace A_infinite_B_infinite_unique_representation_l1883_188324

/-- Two infinite sets of non-negative integers -/
def A : Set ℕ := sorry

/-- Two infinite sets of non-negative integers -/
def B : Set ℕ := sorry

/-- A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

/-- B is infinite -/
theorem B_infinite : Set.Infinite B := by sorry

/-- Every non-negative integer can be uniquely represented as a sum of elements from A and B -/
theorem unique_representation :
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b := by sorry

end A_infinite_B_infinite_unique_representation_l1883_188324


namespace shirt_price_proof_l1883_188356

/-- Proves that if a shirt's price after a 15% discount is $68, then its original price was $80. -/
theorem shirt_price_proof (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 68 → discount_rate = 0.15 → 
  discounted_price = (1 - discount_rate) * 80 := by
sorry

end shirt_price_proof_l1883_188356


namespace product_PQRS_l1883_188310

theorem product_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 := by
  sorry

end product_PQRS_l1883_188310


namespace solution_set_l1883_188388

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_condition (x : ℕ) : Prop :=
  is_prime (3 * x + 1) ∧ 70 ≤ (3 * x + 1) ∧ (3 * x + 1) ≤ 110

theorem solution_set :
  {x : ℕ | satisfies_condition x} = {24, 26, 32, 34, 36} :=
sorry

end solution_set_l1883_188388


namespace exists_number_plus_digit_sum_equals_2014_l1883_188392

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number κ such that κ plus the sum of its digits equals 2014 -/
theorem exists_number_plus_digit_sum_equals_2014 : ∃ κ : ℕ, κ + sum_of_digits κ = 2014 := by
  sorry

end exists_number_plus_digit_sum_equals_2014_l1883_188392


namespace tangent_lines_at_k_zero_equal_angles_point_l1883_188304

-- Define the curve C and the line
def C (x y : ℝ) : Prop := x^2 = 4*y
def L (k a x y : ℝ) : Prop := y = k*x + a

-- Define the intersection points M and N
def intersection_points (k a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ C x y ∧ L k a x y}

-- Theorem for tangent lines when k = 0
theorem tangent_lines_at_k_zero (a : ℝ) (ha : a > 0) :
  ∃ (M N : ℝ × ℝ), M ∈ intersection_points 0 a ∧ N ∈ intersection_points 0 a ∧
  (∃ (x y : ℝ), M = (x, y) ∧ Real.sqrt a * x - y - a = 0) ∧
  (∃ (x y : ℝ), N = (x, y) ∧ Real.sqrt a * x + y + a = 0) :=
sorry

-- Theorem for the existence of point P
theorem equal_angles_point (a : ℝ) (ha : a > 0) :
  ∃ (P : ℝ × ℝ), P.1 = 0 ∧
  ∀ (k : ℝ) (M N : ℝ × ℝ), M ∈ intersection_points k a → N ∈ intersection_points k a →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), M = (x₁, y₁) ∧ N = (x₂, y₂) ∧
   (y₁ - P.2) / x₁ = -(y₂ - P.2) / x₂) :=
sorry

end tangent_lines_at_k_zero_equal_angles_point_l1883_188304


namespace amelia_dinner_l1883_188329

def dinner_problem (initial_amount : ℝ) (first_course : ℝ) (second_course_extra : ℝ) (dessert_percentage : ℝ) : Prop :=
  let second_course := first_course + second_course_extra
  let dessert := dessert_percentage * second_course
  let total_cost := first_course + second_course + dessert
  let money_left := initial_amount - total_cost
  money_left = 20

theorem amelia_dinner :
  dinner_problem 60 15 5 0.25 := by
  sorry

end amelia_dinner_l1883_188329


namespace min_value_f_range_of_t_l1883_188373

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t) ↔ (t ≤ -2 ∨ t ≥ 3) := by sorry

end min_value_f_range_of_t_l1883_188373


namespace modulus_of_z_equals_one_l1883_188354

open Complex

theorem modulus_of_z_equals_one : 
  let z : ℂ := (1 - I) / (1 + I) + 2 * I
  abs z = 1 := by sorry

end modulus_of_z_equals_one_l1883_188354


namespace min_value_of_sum_of_ratios_l1883_188325

theorem min_value_of_sum_of_ratios (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y, x > 0 ∧ y > 0 → (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

end min_value_of_sum_of_ratios_l1883_188325


namespace total_spectators_l1883_188352

theorem total_spectators (men : ℕ) (children : ℕ) (women : ℕ) 
  (h1 : men = 7000)
  (h2 : children = 2500)
  (h3 : children = 5 * women) :
  men + children + women = 10000 := by
  sorry

end total_spectators_l1883_188352


namespace negative_sqrt_of_squared_negative_nine_equals_negative_nine_l1883_188336

theorem negative_sqrt_of_squared_negative_nine_equals_negative_nine :
  -Real.sqrt ((-9)^2) = -9 := by
  sorry

end negative_sqrt_of_squared_negative_nine_equals_negative_nine_l1883_188336


namespace initial_men_correct_l1883_188398

/-- The number of men initially working in a garment industry -/
def initial_men : ℕ := 12

/-- The number of hours worked per day in the initial scenario -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked in the initial scenario -/
def initial_days : ℕ := 10

/-- The number of men in the second scenario -/
def second_men : ℕ := 24

/-- The number of hours worked per day in the second scenario -/
def second_hours_per_day : ℕ := 5

/-- The number of days worked in the second scenario -/
def second_days : ℕ := 8

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  initial_men * initial_hours_per_day * initial_days = 
  second_men * second_hours_per_day * second_days :=
by
  sorry

#check initial_men_correct

end initial_men_correct_l1883_188398


namespace equation_roots_sum_l1883_188341

theorem equation_roots_sum (a b c m : ℝ) : 
  (∃ x y : ℝ, 
    (x^2 - (b+1)*x) / (2*a*x - c) = (2*m-3) / (2*m+1) ∧
    (y^2 - (b+1)*y) / (2*a*y - c) = (2*m-3) / (2*m+1) ∧
    x + y = b + 1) →
  m = 1.5 := by sorry

end equation_roots_sum_l1883_188341


namespace nth_equation_sum_l1883_188340

theorem nth_equation_sum (n : ℕ) (h : n > 0) :
  (Finset.range (2 * n - 1)).sum (λ i => n + i) = (2 * n - 1)^2 := by
  sorry

end nth_equation_sum_l1883_188340


namespace average_selling_price_l1883_188384

def initial_stock : ℝ := 100
def morning_sale_weight : ℝ := 50
def morning_sale_price : ℝ := 1.2
def noon_sale_weight : ℝ := 30
def noon_sale_price : ℝ := 1
def afternoon_sale_weight : ℝ := 20
def afternoon_sale_price : ℝ := 0.8

theorem average_selling_price :
  let total_revenue := morning_sale_weight * morning_sale_price +
                       noon_sale_weight * noon_sale_price +
                       afternoon_sale_weight * afternoon_sale_price
  let total_weight := morning_sale_weight + noon_sale_weight + afternoon_sale_weight
  total_revenue / total_weight = 1.06 := by
  sorry

end average_selling_price_l1883_188384


namespace probability_different_rooms_l1883_188309

theorem probability_different_rooms (n : ℕ) (h : n = 2) : 
  (n - 1 : ℚ) / n = 1 / 2 := by
  sorry

#check probability_different_rooms

end probability_different_rooms_l1883_188309


namespace max_value_implies_m_eq_one_min_value_of_y_l1883_188339

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 1 / x - m

-- Part 1: Prove that if the maximum value of f(x) is -1, then m = 1
theorem max_value_implies_m_eq_one (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f m x ≤ f m x₀) ∧ f m (1 / m) = -1 → m = 1 :=
sorry

-- Part 2: Prove that the minimum value of y is 2 / (1 + e)
theorem min_value_of_y :
  ∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 →
  f 1 x₁ = 0 ∧ f 1 x₂ = 0 →
  Real.exp x₁ ≤ x₂ →
  (x₁ - x₂) * f_derivative 1 (x₁ + x₂) ≥ 2 / (1 + Real.exp 1) :=
sorry

end

end max_value_implies_m_eq_one_min_value_of_y_l1883_188339


namespace largest_of_three_consecutive_evens_l1883_188333

theorem largest_of_three_consecutive_evens (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- a, b, c are consecutive even integers
  a + b + c = 1194 →                               -- their sum is 1194
  c = 400                                          -- the largest (c) is 400
:= by sorry

end largest_of_three_consecutive_evens_l1883_188333


namespace second_shop_payment_l1883_188332

/-- The amount paid for books from the second shop -/
def amount_second_shop (books_first_shop : ℕ) (books_second_shop : ℕ) 
  (price_first_shop : ℚ) (average_price : ℚ) : ℚ := 
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_first_shop

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_payment : 
  amount_second_shop 65 50 1160 (18088695652173913 / 1000000000000000) = 920 := by
  sorry

end second_shop_payment_l1883_188332


namespace parabola_equation_l1883_188317

/-- A parabola with vertex at the origin and focus at (2, 0) has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (p (0, 0)  -- vertex at origin
   ∧ (∀ x y, p (x, y) → (x - 2)^2 + y^2 = 4)  -- focus at (2, 0)
   ∧ (∀ x y, p (x, y) → y^2 = 4 * 2 * x)) :=  -- standard form of parabola with p = 2
by sorry

end parabola_equation_l1883_188317


namespace gym_towels_l1883_188301

theorem gym_towels (first_hour : ℕ) (second_hour_increase : ℚ) 
  (third_hour_increase : ℚ) (fourth_hour_increase : ℚ) 
  (total_towels : ℕ) : 
  first_hour = 50 →
  second_hour_increase = 1/5 →
  third_hour_increase = 1/4 →
  fourth_hour_increase = 1/3 →
  total_towels = 285 →
  let second_hour := first_hour + (first_hour * second_hour_increase).floor
  let third_hour := second_hour + (second_hour * third_hour_increase).floor
  let fourth_hour := third_hour + (third_hour * fourth_hour_increase).floor
  first_hour + second_hour + third_hour + fourth_hour = total_towels :=
by sorry

end gym_towels_l1883_188301


namespace max_value_expression_l1883_188308

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^4 + y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end max_value_expression_l1883_188308


namespace triangle_max_area_l1883_188382

/-- Given a triangle ABC with sides a, b, c and area S, where S = a² - (b-c)² 
    and the circumference of its circumcircle is 17π, 
    prove that the maximum value of S is 64. -/
theorem triangle_max_area (a b c S : ℝ) (h1 : S = a^2 - (b - c)^2) 
  (h2 : 2 * Real.pi * (a / (2 * Real.sin (Real.arcsin (8/17)))) = 17 * Real.pi) :
  S ≤ 64 :=
sorry

end triangle_max_area_l1883_188382


namespace midline_triangle_area_sum_l1883_188376

/-- The sum of areas of an infinite series of triangles, where each triangle is constructed 
    from the midlines of the previous triangle, given the area of the original triangle. -/
theorem midline_triangle_area_sum (t : ℝ) (h : t > 0) : 
  ∃ (S : ℝ), S = (∑' n, t * (3/4)^n) ∧ S = 4 * t :=
sorry

end midline_triangle_area_sum_l1883_188376


namespace some_number_solution_l1883_188320

theorem some_number_solution : 
  ∃ x : ℝ, 45 - (28 - (x - (15 - 15))) = 54 ∧ x = 37 := by sorry

end some_number_solution_l1883_188320


namespace ellipse_major_axis_length_l1883_188360

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem stating the length of the major axis for the given conditions --/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.4 = 5.6 := by
  sorry

end ellipse_major_axis_length_l1883_188360


namespace smallest_common_multiple_of_9_and_6_l1883_188396

theorem smallest_common_multiple_of_9_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  -- The proof goes here
  sorry

end smallest_common_multiple_of_9_and_6_l1883_188396


namespace tangent_line_to_parabola_l1883_188370

/-- The line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + c → y'^2 ≥ 12 * x') ↔ c = 1 := by
  sorry

end tangent_line_to_parabola_l1883_188370


namespace ray_gave_25_cents_to_peter_l1883_188344

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the initial amount Ray had in cents -/
def initial_amount : ℕ := 95

/-- Represents the number of nickels Ray had left -/
def nickels_left : ℕ := 4

/-- Represents the amount given to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- Proves that Ray gave 25 cents to Peter given the initial conditions -/
theorem ray_gave_25_cents_to_peter :
  let total_given := initial_amount - (nickels_left * nickel_value)
  let amount_to_randi := 2 * amount_to_peter
  total_given = amount_to_peter + amount_to_randi :=
by sorry

end ray_gave_25_cents_to_peter_l1883_188344


namespace base_of_negative_four_cubed_l1883_188386

def base_of_power (x : ℤ) (n : ℕ) : ℤ := x

theorem base_of_negative_four_cubed :
  base_of_power (-4) 3 = -4 := by sorry

end base_of_negative_four_cubed_l1883_188386


namespace specific_tetrahedron_volume_l1883_188374

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (AB BC CD DA AC BD : ℝ) : ℝ :=
  -- Definition of volume calculation goes here
  sorry

/-- Theorem: The volume of the specific tetrahedron is √66/2 -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 1 (2 * Real.sqrt 6) 5 7 5 7 = Real.sqrt 66 / 2 := by
  sorry

end specific_tetrahedron_volume_l1883_188374


namespace num_ways_to_select_is_186_l1883_188351

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6
def red_ball_score : ℕ := 2
def white_ball_score : ℕ := 1
def total_balls_to_take : ℕ := 5
def min_total_score : ℕ := 7

def score (red white : ℕ) : ℕ :=
  red * red_ball_score + white * white_ball_score

def valid_selection (red white : ℕ) : Prop :=
  red + white = total_balls_to_take ∧ 
  red ≤ num_red_balls ∧ 
  white ≤ num_white_balls ∧ 
  score red white ≥ min_total_score

def num_ways_to_select : ℕ := 
  (Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1) +
  (Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2) +
  (Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3)

theorem num_ways_to_select_is_186 : num_ways_to_select = 186 := by
  sorry

end num_ways_to_select_is_186_l1883_188351


namespace rectangle_perimeter_area_sum_l1883_188300

def Rectangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def vertices : Rectangle := ((1, 1), (1, 5), (6, 5), (6, 1))

def length (r : Rectangle) : ℝ := 
  let ((x1, _), (_, _), (x2, _), _) := r
  |x2 - x1|

def width (r : Rectangle) : ℝ := 
  let ((_, y1), (_, y2), _, _) := r
  |y2 - y1|

def perimeter (r : Rectangle) : ℝ := 2 * (length r + width r)

def area (r : Rectangle) : ℝ := length r * width r

theorem rectangle_perimeter_area_sum :
  perimeter vertices + area vertices = 38 := by sorry

end rectangle_perimeter_area_sum_l1883_188300


namespace citizenship_test_study_time_l1883_188331

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 90
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let essay_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  let essay_time : ℕ := 45            -- minutes per question
  
  let total_time_minutes : ℕ := 
    multiple_choice_questions * multiple_choice_time +
    fill_in_blank_questions * fill_in_blank_time +
    essay_questions * essay_time

  let total_time_hours : ℚ := (total_time_minutes : ℚ) / 60

  total_questions = multiple_choice_questions + fill_in_blank_questions + essay_questions →
  total_time_hours = 42.5 := by
  sorry

end citizenship_test_study_time_l1883_188331


namespace age_difference_is_27_l1883_188318

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h : tens < 10 ∧ ones < 10)

def Age.value (a : Age) : Nat := 10 * a.tens + a.ones

def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h.symm⟩

theorem age_difference_is_27 (alan_age bob_age : Age) : 
  (alan_age.reverse = bob_age) →
  (bob_age.value = alan_age.value / 2 + 6) →
  (alan_age.value + 2 = 5 * (bob_age.value - 4)) →
  (alan_age.value - bob_age.value = 27) :=
sorry

end age_difference_is_27_l1883_188318


namespace fifth_number_in_list_l1883_188379

theorem fifth_number_in_list (numbers : List ℕ) : 
  numbers.length = 9 ∧ 
  numbers.sum = 207 * 9 ∧
  201 ∈ numbers ∧ 
  202 ∈ numbers ∧ 
  204 ∈ numbers ∧ 
  205 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  210 ∈ numbers ∧ 
  212 ∈ numbers ∧ 
  212 ∈ numbers →
  ∃ (fifth : ℕ), fifth ∈ numbers ∧ fifth = 211 := by
sorry

end fifth_number_in_list_l1883_188379


namespace a_minus_b_equals_negative_seven_l1883_188326

theorem a_minus_b_equals_negative_seven
  (a b : ℝ)
  (h1 : |a| = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0) :
  a - b = -7 := by
sorry

end a_minus_b_equals_negative_seven_l1883_188326


namespace point_not_in_fourth_quadrant_l1883_188367

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m - 1 > 0 ∧ m + 3 < 0) :=
by sorry

end point_not_in_fourth_quadrant_l1883_188367


namespace cuboidal_box_surface_area_l1883_188387

/-- A cuboidal box with given face areas has a specific total surface area -/
theorem cuboidal_box_surface_area (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 120 → w * h = 72 → l * h = 60 →
  2 * (l * w + w * h + l * h) = 504 := by
sorry

end cuboidal_box_surface_area_l1883_188387


namespace coplanar_vectors_k_value_l1883_188362

def a : ℝ × ℝ × ℝ := (1, -1, 2)
def b : ℝ × ℝ × ℝ := (-2, 1, 0)
def c (k : ℝ) : ℝ × ℝ × ℝ := (-3, 1, k)

theorem coplanar_vectors_k_value :
  ∀ k : ℝ, (∃ x y : ℝ, c k = x • a + y • b) → k = 2 := by
  sorry

end coplanar_vectors_k_value_l1883_188362


namespace complex_roots_circle_radius_l1883_188335

theorem complex_roots_circle_radius : 
  ∀ z : ℂ, (z - 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end complex_roots_circle_radius_l1883_188335


namespace binomial_identities_l1883_188381

theorem binomial_identities (n k : ℕ+) :
  (Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1)) ∧
  (Nat.choose n k = (n / k) * Nat.choose (n - 1) (k - 1)) := by
  sorry

end binomial_identities_l1883_188381


namespace sum_and_equal_numbers_l1883_188311

theorem sum_and_equal_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 150)
  (equal_numbers : a - 3 = b + 4 ∧ b + 4 = 4 * c) : 
  a = 631 / 9 := by
sorry

end sum_and_equal_numbers_l1883_188311


namespace jim_current_age_l1883_188399

/-- Represents the ages of Jim, Fred, and Sam -/
structure Ages where
  jim : ℕ
  fred : ℕ
  sam : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.jim = 2 * ages.fred ∧
  ages.fred = ages.sam + 9 ∧
  ages.jim - 6 = 5 * (ages.sam - 6)

/-- The theorem stating Jim's current age -/
theorem jim_current_age :
  ∃ ages : Ages, satisfiesConditions ages ∧ ages.jim = 46 :=
sorry

end jim_current_age_l1883_188399


namespace successful_pair_existence_l1883_188371

/-- A pair of natural numbers is successful if their arithmetic mean and geometric mean are both natural numbers. -/
def IsSuccessfulPair (a b : ℕ) : Prop :=
  ∃ m g : ℕ, 2 * m = a + b ∧ g * g = a * b

theorem successful_pair_existence (m n k : ℕ) (h1 : m > n) (h2 : n > 0) (h3 : m > k) (h4 : k > 0)
  (h5 : IsSuccessfulPair (m + n) (m - n)) (h6 : m^2 - n^2 = k^2) :
  ∃ (a b : ℕ), a ≠ b ∧ IsSuccessfulPair a b ∧ 2 * m = a + b ∧ (a ≠ m + n ∨ b ≠ m - n) := by
  sorry

end successful_pair_existence_l1883_188371


namespace music_festival_children_avg_age_l1883_188306

/-- Represents the demographics and age statistics of a music festival. -/
structure MusicFestival where
  total_participants : ℕ
  num_women : ℕ
  num_men : ℕ
  num_children : ℕ
  overall_avg_age : ℚ
  women_avg_age : ℚ
  men_avg_age : ℚ

/-- Calculates the average age of children in the music festival. -/
def children_avg_age (festival : MusicFestival) : ℚ :=
  (festival.total_participants * festival.overall_avg_age
   - festival.num_women * festival.women_avg_age
   - festival.num_men * festival.men_avg_age) / festival.num_children

/-- Theorem stating that for the given music festival data, the average age of children is 13. -/
theorem music_festival_children_avg_age :
  let festival : MusicFestival := {
    total_participants := 50,
    num_women := 22,
    num_men := 18,
    num_children := 10,
    overall_avg_age := 20,
    women_avg_age := 24,
    men_avg_age := 19
  }
  children_avg_age festival = 13 := by sorry

end music_festival_children_avg_age_l1883_188306


namespace sequence_contains_large_number_l1883_188345

theorem sequence_contains_large_number 
  (seq : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → seq i ≠ seq j) 
  (consecutive_product_square : ∀ i : Fin 19, ∃ k : ℕ, seq i * seq (i.succ) = k * k) 
  (first_num : seq 0 = 42) :
  ∃ i : Fin 20, seq i > 16000 := by
  sorry

end sequence_contains_large_number_l1883_188345


namespace solve_equation_one_solve_equation_two_l1883_188343

-- Equation 1
theorem solve_equation_one (x : ℝ) : 4 * (x - 2) = 2 * x ↔ x = 4 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x + 1) / 4 = 1 - (1 - x) / 3 ↔ x = -5 := by
  sorry

end solve_equation_one_solve_equation_two_l1883_188343


namespace circle_passes_through_fixed_point_tangent_circles_l1883_188366

/-- The parametric equation of a circle -/
def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point (a : ℝ) :
  circle_equation a 4 (-2) := by sorry

theorem tangent_circles (a : ℝ) :
  (∃ x y : ℝ, circle_equation a x y ∧ fixed_circle x y) ↔ 
  (a = 1 + Real.sqrt 5 / 5 ∨ a = 1 - Real.sqrt 5 / 5) := by sorry

end circle_passes_through_fixed_point_tangent_circles_l1883_188366


namespace games_spent_proof_l1883_188361

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/4
def apps_fraction : ℚ := 3/10
def snacks_fraction : ℚ := 1/5

def books_spent : ℚ := total_allowance * books_fraction
def apps_spent : ℚ := total_allowance * apps_fraction
def snacks_spent : ℚ := total_allowance * snacks_fraction

def other_expenses : ℚ := books_spent + apps_spent + snacks_spent

theorem games_spent_proof : total_allowance - other_expenses = 25/2 := by sorry

end games_spent_proof_l1883_188361


namespace allocation_schemes_count_l1883_188378

/-- The number of intern teachers --/
def num_teachers : ℕ := 5

/-- The number of classes --/
def num_classes : ℕ := 3

/-- The minimum number of teachers per class --/
def min_teachers_per_class : ℕ := 1

/-- The maximum number of teachers per class --/
def max_teachers_per_class : ℕ := 2

/-- A function that calculates the number of ways to allocate teachers to classes --/
def allocation_schemes (n_teachers : ℕ) (n_classes : ℕ) (min_per_class : ℕ) (max_per_class : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 90 --/
theorem allocation_schemes_count :
  allocation_schemes num_teachers num_classes min_teachers_per_class max_teachers_per_class = 90 :=
sorry

end allocation_schemes_count_l1883_188378


namespace ab_positive_iff_hyperbola_l1883_188389

-- Define the condition for a hyperbola
def is_hyperbola (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 - b * y^2 = 1

-- State the theorem
theorem ab_positive_iff_hyperbola (a b : ℝ) :
  a * b > 0 ↔ is_hyperbola a b :=
sorry

end ab_positive_iff_hyperbola_l1883_188389


namespace min_value_theorem_l1883_188377

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min : ℝ), min = -9/8 ∧ ∀ (z : ℝ), z = x + y + x * y → z ≥ min :=
sorry

end min_value_theorem_l1883_188377


namespace calculate_expression_l1883_188319

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end calculate_expression_l1883_188319


namespace remaining_area_formula_l1883_188383

/-- The area of a rectangular field with dimensions (x + 8) and (x + 6), 
    excluding a rectangular patch with dimensions (2x - 4) and (x - 3) -/
def remaining_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3)

/-- Theorem stating that the remaining area is equal to -x^2 + 24x + 36 -/
theorem remaining_area_formula (x : ℝ) : 
  remaining_area x = -x^2 + 24*x + 36 := by
  sorry

end remaining_area_formula_l1883_188383


namespace fast_food_order_l1883_188350

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 5

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℕ := 4

/-- The total cost of the order in dollars -/
def total_cost : ℕ := 17

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

theorem fast_food_order :
  burger_cost + sandwich_cost + smoothie_cost * num_smoothies = total_cost := by
  sorry

end fast_food_order_l1883_188350


namespace find_k_l1883_188302

-- Define the binary linear equation
def binary_linear_equation (x y t : ℝ) : Prop := 3 * x - 2 * y = t

-- Define the theorem
theorem find_k (m n : ℝ) (h1 : binary_linear_equation m n 5) 
  (h2 : binary_linear_equation (m + 2) (n - 2) k) : k = 15 := by
  sorry

end find_k_l1883_188302


namespace percentage_calculation_l1883_188359

theorem percentage_calculation (x : ℝ) (h : 0.035 * x = 700) : 0.024 * (1.5 * x) = 720 := by
  sorry

end percentage_calculation_l1883_188359


namespace oplus_five_two_l1883_188337

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem statement
theorem oplus_five_two : oplus 5 2 = 30 := by
  sorry

end oplus_five_two_l1883_188337


namespace yard_length_with_26_trees_32m_apart_l1883_188338

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

theorem yard_length_with_26_trees_32m_apart :
  yardLength 26 32 = 800 := by
  sorry

end yard_length_with_26_trees_32m_apart_l1883_188338


namespace integral_tan_over_trig_expression_l1883_188307

theorem integral_tan_over_trig_expression :
  let f := fun x : ℝ => (Real.tan x) / (Real.sin x ^ 2 - 5 * Real.cos x ^ 2 + 4)
  let a := Real.pi / 4
  let b := Real.arccos (1 / Real.sqrt 3)
  ∫ x in a..b, f x = (1 / 10) * Real.log (9 / 4) :=
by sorry

end integral_tan_over_trig_expression_l1883_188307


namespace intersection_condition_for_singleton_zero_l1883_188380

theorem intersection_condition_for_singleton_zero (A : Set ℕ) :
  (A = {0} → A ∩ {0, 1} = {0}) ∧
  ∃ A : Set ℕ, A ∩ {0, 1} = {0} ∧ A ≠ {0} :=
by sorry

end intersection_condition_for_singleton_zero_l1883_188380


namespace equation_one_l1883_188347

theorem equation_one (x : ℝ) : x * |x| = 4 ↔ x = 2 := by sorry

end equation_one_l1883_188347


namespace sequence_progression_l1883_188316

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem sequence_progression (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end sequence_progression_l1883_188316


namespace train_cars_count_l1883_188369

-- Define the given conditions
def cars_counted : ℕ := 10
def initial_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the theorem
theorem train_cars_count :
  let rate : ℚ := cars_counted / initial_time
  rate * total_time = 140 := by
  sorry

end train_cars_count_l1883_188369


namespace train_speed_Q_l1883_188321

/-- The distance between stations P and Q in kilometers -/
def distance_PQ : ℝ := 65

/-- The speed of the train starting from station P in kilometers per hour -/
def speed_P : ℝ := 20

/-- The time difference between the start of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total time until the trains meet in hours -/
def total_time : ℝ := 2

/-- The speed of the train starting from station Q in kilometers per hour -/
def speed_Q : ℝ := 25

theorem train_speed_Q : speed_Q = (distance_PQ - speed_P * total_time) / (total_time - time_difference) :=
sorry

end train_speed_Q_l1883_188321


namespace total_cost_before_markup_is_47_l1883_188322

/-- The markup percentage as a decimal -/
def markup : ℚ := 0.10

/-- The selling prices of the three books -/
def sellingPrices : List ℚ := [11.00, 16.50, 24.20]

/-- Calculate the original price before markup -/
def originalPrice (sellingPrice : ℚ) : ℚ := sellingPrice / (1 + markup)

/-- Calculate the total cost before markup -/
def totalCostBeforeMarkup : ℚ := (sellingPrices.map originalPrice).sum

/-- Theorem stating that the total cost before markup is $47.00 -/
theorem total_cost_before_markup_is_47 : totalCostBeforeMarkup = 47 := by
  sorry

end total_cost_before_markup_is_47_l1883_188322


namespace smallest_divisor_and_quadratic_form_l1883_188323

theorem smallest_divisor_and_quadratic_form : ∃ k : ℕ,
  (∃ n : ℕ, (2^n + 15) % k = 0) ∧
  (∃ x y : ℤ, k = 3*x^2 - 4*x*y + 3*y^2) ∧
  (∀ m : ℕ, m < k →
    (∃ n : ℕ, (2^n + 15) % m = 0) ∧
    (∃ x y : ℤ, m = 3*x^2 - 4*x*y + 3*y^2) →
    False) ∧
  k = 23 := by
sorry

end smallest_divisor_and_quadratic_form_l1883_188323


namespace boric_acid_mixture_concentration_l1883_188342

/-- Given two boric acid solutions with concentrations and volumes, 
    calculate the concentration of the resulting mixture --/
theorem boric_acid_mixture_concentration 
  (c1 : ℝ) (c2 : ℝ) (v1 : ℝ) (v2 : ℝ) 
  (h1 : c1 = 0.01) -- 1% concentration
  (h2 : c2 = 0.05) -- 5% concentration
  (h3 : v1 = 15) -- 15 mL of first solution
  (h4 : v2 = 15) -- 15 mL of second solution
  : (c1 * v1 + c2 * v2) / (v1 + v2) = 0.03 := by
  sorry

#check boric_acid_mixture_concentration

end boric_acid_mixture_concentration_l1883_188342


namespace dima_walking_speed_l1883_188327

/-- Represents the time in hours and minutes -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hour * 60 + t2.minute) - (t1.hour * 60 + t1.minute)

/-- Represents the problem setup -/
structure ProblemSetup where
  scheduledArrival : Time
  actualArrival : Time
  carSpeed : Nat
  earlyArrivalTime : Nat

/-- Calculates Dima's walking speed -/
def calculateWalkingSpeed (setup : ProblemSetup) : Rat :=
  sorry

theorem dima_walking_speed (setup : ProblemSetup) 
  (h1 : setup.scheduledArrival = ⟨18, 0⟩)
  (h2 : setup.actualArrival = ⟨17, 5⟩)
  (h3 : setup.carSpeed = 60)
  (h4 : setup.earlyArrivalTime = 10) :
  calculateWalkingSpeed setup = 6 := by
  sorry

end dima_walking_speed_l1883_188327


namespace existence_and_uniqueness_l1883_188328

open Real

/-- The differential equation y' = y - x^2 + 2x - 2 -/
def diff_eq (x y : ℝ) : ℝ := y - x^2 + 2*x - 2

/-- A solution to the differential equation -/
def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv f) x = diff_eq x (f x)

theorem existence_and_uniqueness :
  ∀ (x₀ y₀ : ℝ), ∃! f : ℝ → ℝ,
    is_solution f ∧ f x₀ = y₀ :=
sorry

end existence_and_uniqueness_l1883_188328


namespace divisibility_condition_l1883_188315

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end divisibility_condition_l1883_188315


namespace studentB_is_optimal_l1883_188348

-- Define the structure for a student
structure Student where
  name : String
  average : ℝ
  variance : ℝ

-- Define the students
def studentA : Student := { name := "A", average := 92, variance := 3.6 }
def studentB : Student := { name := "B", average := 95, variance := 3.6 }
def studentC : Student := { name := "C", average := 95, variance := 7.4 }
def studentD : Student := { name := "D", average := 95, variance := 8.1 }

-- Define the list of all students
def students : List Student := [studentA, studentB, studentC, studentD]

-- Function to determine if one student is better than another
def isBetterStudent (s1 s2 : Student) : Prop :=
  (s1.average > s2.average) ∨ (s1.average = s2.average ∧ s1.variance < s2.variance)

-- Theorem stating that student B is the optimal choice
theorem studentB_is_optimal : 
  ∀ s ∈ students, s.name ≠ "B" → isBetterStudent studentB s :=
by sorry

end studentB_is_optimal_l1883_188348


namespace school_fee_calculation_l1883_188364

/-- Represents the amount of money given by Luke's mother -/
def mother_contribution : ℕ :=
  50 + 2 * 20 + 3 * 10

/-- Represents the amount of money given by Luke's father -/
def father_contribution : ℕ :=
  4 * 50 + 20 + 10

/-- Represents the total school fee -/
def school_fee : ℕ :=
  mother_contribution + father_contribution

theorem school_fee_calculation :
  school_fee = 350 :=
by sorry

end school_fee_calculation_l1883_188364


namespace ratio_w_to_y_l1883_188363

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw_x : w / x = 5 / 2)
  (hy_z : y / z = 2 / 3)
  (hx_z : x / z = 10) :
  w / y = 37.5 := by
sorry

end ratio_w_to_y_l1883_188363


namespace quadratic_function_theorem_l1883_188372

def is_valid_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def satisfies_bounds (f : ℝ → ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |f x| ≤ 1

def derivative_max (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |(deriv f) x| ≤ K

def max_attained (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∃ x₀, x₀ ∈ Set.Icc (-1) 1 ∧ |(deriv f) x₀| = K

theorem quadratic_function_theorem (f : ℝ → ℝ) (K : ℝ) :
  is_valid_quadratic f →
  satisfies_bounds f →
  derivative_max f K →
  max_attained f K →
  (∃ (ε : ℝ), ε = 1 ∨ ε = -1) ∧ (∀ x, f x = ε * (2 * x^2 - 1)) :=
sorry

end quadratic_function_theorem_l1883_188372


namespace earliest_meeting_time_l1883_188349

def lapTime1 : ℕ := 5
def lapTime2 : ℕ := 8
def lapTime3 : ℕ := 9
def startTime : ℕ := 7 * 60  -- 7:00 AM in minutes since midnight

def meetingTime : ℕ := startTime + Nat.lcm (Nat.lcm lapTime1 lapTime2) lapTime3

theorem earliest_meeting_time :
  meetingTime = 13 * 60  -- 1:00 PM in minutes since midnight
  := by sorry

end earliest_meeting_time_l1883_188349


namespace mixed_groups_count_l1883_188330

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end mixed_groups_count_l1883_188330


namespace shane_garret_age_ratio_l1883_188303

theorem shane_garret_age_ratio : 
  let shane_current_age : ℕ := 44
  let garret_current_age : ℕ := 12
  let years_ago : ℕ := 20
  let shane_past_age : ℕ := shane_current_age - years_ago
  shane_past_age / garret_current_age = 2 :=
by sorry

end shane_garret_age_ratio_l1883_188303


namespace unique_arrangement_l1883_188355

structure Ball :=
  (color : String)

structure Box :=
  (color : String)
  (balls : List Ball)

def valid_arrangement (boxes : List Box) : Prop :=
  boxes.length = 3 ∧
  (∃ red_box white_box yellow_box,
    boxes = [red_box, white_box, yellow_box] ∧
    red_box.color = "red" ∧ white_box.color = "white" ∧ yellow_box.color = "yellow" ∧
    (∀ ball ∈ yellow_box.balls, ball.color = "white") ∧
    (∀ ball ∈ white_box.balls, ball.color = "red") ∧
    (∀ ball ∈ red_box.balls, ball.color = "yellow") ∧
    yellow_box.balls.length > (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "yellow"))).join.length ∧
    red_box.balls.length ≠ (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "white"))).join.length ∧
    (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "white"))).join.length < white_box.balls.length)

theorem unique_arrangement (boxes : List Box) :
  valid_arrangement boxes →
  ∃ red_box white_box yellow_box,
    boxes = [red_box, white_box, yellow_box] ∧
    (∀ ball ∈ red_box.balls, ball.color = "yellow") ∧
    (∀ ball ∈ white_box.balls, ball.color = "red") ∧
    (∀ ball ∈ yellow_box.balls, ball.color = "white") :=
by sorry

end unique_arrangement_l1883_188355


namespace max_square_plots_l1883_188313

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fence length for internal fencing -/
def available_fence : ℕ := 2200

/-- Calculates the number of square plots given the number of plots in a column -/
def num_plots (n : ℕ) : ℕ := n * (11 * n / 6)

/-- Calculates the required fence length for a given number of plots in a column -/
def required_fence (n : ℕ) : ℕ := 187 * n - 132

/-- The maximum number of square plots that can partition the field -/
def max_plots : ℕ := 264

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
  (h1 : field.length = 36) 
  (h2 : field.width = 66) : 
  (∀ n : ℕ, num_plots n ≤ max_plots ∧ required_fence n ≤ available_fence) ∧ 
  (∃ n : ℕ, num_plots n = max_plots ∧ required_fence n ≤ available_fence) :=
sorry

end max_square_plots_l1883_188313


namespace survey_sample_size_l1883_188358

/-- Represents a survey with a given population size and number of selected participants. -/
structure Survey where
  population_size : ℕ
  selected_participants : ℕ

/-- Calculates the sample size of a given survey. -/
def sample_size (s : Survey) : ℕ := s.selected_participants

/-- Theorem stating that for a survey with 4000 students and 500 randomly selected,
    the sample size is 500. -/
theorem survey_sample_size :
  let s : Survey := { population_size := 4000, selected_participants := 500 }
  sample_size s = 500 := by sorry

end survey_sample_size_l1883_188358


namespace infinitely_many_solutions_l1883_188305

/-- Represents the quantities of sugar types A, B, and C -/
structure SugarQuantities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given quantities satisfy the problem constraints -/
def satisfiesConstraints (q : SugarQuantities) : Prop :=
  q.a ≥ 0 ∧ q.b ≥ 0 ∧ q.c ≥ 0 ∧
  q.a + q.b + q.c = 1500 ∧
  (8 * q.a + 15 * q.b + 20 * q.c) / 1500 = 14

/-- There are infinitely many solutions to the sugar problem -/
theorem infinitely_many_solutions :
  ∀ ε > 0, ∃ q₁ q₂ : SugarQuantities,
    satisfiesConstraints q₁ ∧
    satisfiesConstraints q₂ ∧
    q₁ ≠ q₂ ∧
    ‖q₁.a - q₂.a‖ < ε ∧
    ‖q₁.b - q₂.b‖ < ε ∧
    ‖q₁.c - q₂.c‖ < ε :=
by sorry

end infinitely_many_solutions_l1883_188305


namespace cube_parallel_edge_pairs_l1883_188353

/-- A cube is a three-dimensional geometric shape with 12 edges. -/
structure Cube where
  edges : Fin 12
  dimensions : Fin 3

/-- A pair of parallel edges in a cube. -/
structure ParallelEdgePair where
  edge1 : Fin 12
  edge2 : Fin 12

/-- The number of parallel edge pairs in a cube. -/
def parallel_edge_pairs (c : Cube) : ℕ := 18

/-- Theorem: A cube has 18 pairs of parallel edges. -/
theorem cube_parallel_edge_pairs (c : Cube) : 
  parallel_edge_pairs c = 18 := by sorry

end cube_parallel_edge_pairs_l1883_188353


namespace worker_b_days_l1883_188393

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  wage_a : ℚ
  wage_b : ℚ
  wage_c : ℚ
  days_a : ℕ
  days_b : ℕ
  days_c : ℕ

/-- Calculates the total earnings of the workers -/
def totalEarnings (data : WorkerData) : ℚ :=
  data.wage_a * data.days_a + data.wage_b * data.days_b + data.wage_c * data.days_c

theorem worker_b_days (data : WorkerData) 
  (h1 : data.days_a = 6)
  (h2 : data.days_c = 4)
  (h3 : data.wage_a / data.wage_b = 3 / 4)
  (h4 : data.wage_b / data.wage_c = 4 / 5)
  (h5 : totalEarnings data = 1702)
  (h6 : data.wage_c = 115) :
  data.days_b = 9 := by
sorry

end worker_b_days_l1883_188393


namespace middle_card_is_five_l1883_188397

def is_valid_trio (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 16

def leftmost_uncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_trio a b₁ c₁ ∧ is_valid_trio a b₂ c₂

def rightmost_uncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_trio a₁ b₁ c ∧ is_valid_trio a₂ b₂ c

def middle_uncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, (a₁ ≠ a₂ ∨ c₁ ≠ c₂) ∧ is_valid_trio a₁ b c₁ ∧ is_valid_trio a₂ b c₂

theorem middle_card_is_five :
  ∀ a b c : ℕ,
    is_valid_trio a b c →
    leftmost_uncertain a →
    rightmost_uncertain c →
    middle_uncertain b →
    b = 5 := by sorry

end middle_card_is_five_l1883_188397


namespace max_reciprocal_eccentricity_sum_l1883_188394

theorem max_reciprocal_eccentricity_sum (e₁ e₂ : ℝ) : 
  e₁ > 0 → e₂ > 0 → 
  (∃ b c : ℝ, b > 0 ∧ c > b ∧ 
    e₁ = c / Real.sqrt (c^2 + (2*b)^2) ∧ 
    e₂ = c / Real.sqrt (c^2 - b^2)) → 
  1/e₁^2 + 4/e₂^2 = 5 → 
  1/e₁ + 1/e₂ ≤ 5/2 :=
by sorry

end max_reciprocal_eccentricity_sum_l1883_188394


namespace berry_theorem_l1883_188385

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) (raspberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries + raspberries)

theorem berry_theorem : berry_problem 50 18 12 7 = 13 := by
  sorry

end berry_theorem_l1883_188385


namespace cos_540_degrees_l1883_188375

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end cos_540_degrees_l1883_188375
