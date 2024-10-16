import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_inequality_l3020_302041

theorem right_triangle_inscribed_circle_inequality (a b r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0)
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 2)
  (h_inscribed_circle : r = a * b / (a + b + Real.sqrt (a^2 + b^2))) :
  2 + Real.sqrt 2 ≤ (2 * a * b) / ((a + b) * r) ∧ (2 * a * b) / ((a + b) * r) < 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_inequality_l3020_302041


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3020_302076

-- Define the parametric equations
def x (t : ℝ) : ℝ := 1 + t
def y (t : ℝ) : ℝ := 1 - t

-- Define the line using the parametric equations
def line : Set (ℝ × ℝ) := {(x t, y t) | t : ℝ}

-- State the theorem
theorem line_inclination_angle :
  let slope := (y 1 - y 0) / (x 1 - x 0)
  let inclination_angle := Real.arctan slope
  inclination_angle = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3020_302076


namespace NUMINAMATH_CALUDE_park_visitors_l3020_302091

theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l3020_302091


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3020_302018

theorem quadratic_factorization (x : ℝ) : 9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3020_302018


namespace NUMINAMATH_CALUDE_pool_filling_cost_l3020_302054

/-- The cost to fill Toby's swimming pool -/
theorem pool_filling_cost 
  (fill_time : ℕ) 
  (flow_rate : ℕ) 
  (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate * water_cost : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_pool_filling_cost_l3020_302054


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l3020_302034

/-- The inverse relationship between 5y and x^2 -/
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 5 * y = k / (x ^ 2)

theorem inverse_variation_solution (x₀ y₀ x₁ : ℝ) 
  (h₀ : inverse_relation x₀ y₀)
  (h₁ : x₀ = 2)
  (h₂ : y₀ = 4)
  (h₃ : x₁ = 4) :
  ∃ y₁ : ℝ, inverse_relation x₁ y₁ ∧ y₁ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l3020_302034


namespace NUMINAMATH_CALUDE_largest_number_l3020_302064

theorem largest_number (S : Finset ℕ) (h : S = {5, 8, 4, 3, 2}) : 
  Finset.max' S (by simp [h]) = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3020_302064


namespace NUMINAMATH_CALUDE_prob_more_heads_ten_coins_l3020_302049

/-- The number of coins being flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting more heads than tails when flipping n fair coins -/
def prob_more_heads (n : ℕ) (p : ℚ) : ℚ :=
  1/2 * (1 - (n.choose (n/2)) / (2^n))

theorem prob_more_heads_ten_coins :
  prob_more_heads n p = 193/512 := by
  sorry

#eval prob_more_heads n p

end NUMINAMATH_CALUDE_prob_more_heads_ten_coins_l3020_302049


namespace NUMINAMATH_CALUDE_rectangle_area_l3020_302043

theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 56 →
  area = length * breadth →
  area = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3020_302043


namespace NUMINAMATH_CALUDE_art_performance_probability_l3020_302020

def artDepartment : Finset Nat := {1, 2, 3, 4}
def firstGrade : Finset Nat := {1, 2}
def secondGrade : Finset Nat := {3, 4}

theorem art_performance_probability :
  let totalSelections := Finset.powerset artDepartment |>.filter (λ s => s.card = 2)
  let differentGradeSelections := totalSelections.filter (λ s => s ∩ firstGrade ≠ ∅ ∧ s ∩ secondGrade ≠ ∅)
  (differentGradeSelections.card : ℚ) / totalSelections.card = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_art_performance_probability_l3020_302020


namespace NUMINAMATH_CALUDE_no_simultaneous_roots_one_and_neg_one_l3020_302047

theorem no_simultaneous_roots_one_and_neg_one :
  ¬ ∃ (a b : ℝ), (1 : ℝ)^3 + a * (1 : ℝ)^2 + b = 0 ∧ (-1 : ℝ)^3 + a * (-1 : ℝ)^2 + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_roots_one_and_neg_one_l3020_302047


namespace NUMINAMATH_CALUDE_kayla_apples_l3020_302046

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 200 →
  total = kylie + kayla →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l3020_302046


namespace NUMINAMATH_CALUDE_truck_load_after_deliveries_l3020_302074

theorem truck_load_after_deliveries :
  let initial_load : ℝ := 50000
  let first_unload_percentage : ℝ := 0.1
  let second_unload_percentage : ℝ := 0.2
  let after_first_delivery := initial_load * (1 - first_unload_percentage)
  let final_load := after_first_delivery * (1 - second_unload_percentage)
  final_load = 36000 := by sorry

end NUMINAMATH_CALUDE_truck_load_after_deliveries_l3020_302074


namespace NUMINAMATH_CALUDE_least_upper_bound_and_greatest_lower_bound_l3020_302029

theorem least_upper_bound_and_greatest_lower_bound 
  (A : Set ℝ) (hA : A.Nonempty) :
  (∃ b : ℝ, ∀ a ∈ A, a ≤ b) → 
  (∃! x : ℝ, (∀ a ∈ A, a ≤ x) ∧ (∀ y : ℝ, (∀ a ∈ A, a ≤ y) → x ≤ y)) ∧
  (∃ b : ℝ, ∀ a ∈ A, b ≤ a) → 
  (∃! x : ℝ, (∀ a ∈ A, x ≤ a) ∧ (∀ y : ℝ, (∀ a ∈ A, y ≤ a) → y ≤ x)) :=
by sorry

end NUMINAMATH_CALUDE_least_upper_bound_and_greatest_lower_bound_l3020_302029


namespace NUMINAMATH_CALUDE_total_kittens_l3020_302082

/-- Given an initial number of kittens and additional kittens, 
    prove that the total number of kittens is their sum. -/
theorem total_kittens (initial additional : ℕ) :
  initial + additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_total_kittens_l3020_302082


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l3020_302052

theorem simplify_product_of_radicals (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (30 * x) = 30 * x * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l3020_302052


namespace NUMINAMATH_CALUDE_haruto_tomatoes_l3020_302033

def tomato_problem (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (given : ℕ) (left : ℕ) : Prop :=
  (initial - eaten = remaining) ∧
  (remaining / 2 = given) ∧
  (remaining - given = left)

theorem haruto_tomatoes : tomato_problem 127 19 108 54 54 := by
  sorry

end NUMINAMATH_CALUDE_haruto_tomatoes_l3020_302033


namespace NUMINAMATH_CALUDE_expected_red_balls_l3020_302088

/-- Given a bag of balls with some red and some white, prove that the expected
    number of red balls is proportional to the number of red draws in a series
    of random draws with replacement. -/
theorem expected_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  (total_balls : ℚ) * (red_draws : ℚ) / (total_draws : ℚ) = 3 :=
sorry

end NUMINAMATH_CALUDE_expected_red_balls_l3020_302088


namespace NUMINAMATH_CALUDE_log8_three_point_five_equals_512_l3020_302058

-- Define the logarithm base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log8_three_point_five_equals_512 :
  ∀ x : ℝ, x > 0 → log8 x = 3.5 → x = 512 := by
  sorry

end NUMINAMATH_CALUDE_log8_three_point_five_equals_512_l3020_302058


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3020_302042

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

theorem perpendicular_vectors (k : ℝ) : 
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3020_302042


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3020_302013

/-- A circle passing through three given points -/
def CircleThroughPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 - 7*p.1 - 3*p.2 + 2) = 0}

/-- Theorem stating that the circle passes through the given points -/
theorem circle_passes_through_points :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (4, -2)
  A ∈ CircleThroughPoints A B C ∧
  B ∈ CircleThroughPoints A B C ∧
  C ∈ CircleThroughPoints A B C := by
  sorry

#check circle_passes_through_points

end NUMINAMATH_CALUDE_circle_passes_through_points_l3020_302013


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3020_302096

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ 
   x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 1 ∧
   y₁*(y₁ - 2) - y₁ + 2 = 0 ∧ y₂*(y₂ - 2) - y₂ + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3020_302096


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l3020_302008

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = 
  {(-1, -1), (-1, 0), (0, -1), (0, 0), (5, 2), (-6, 2)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l3020_302008


namespace NUMINAMATH_CALUDE_max_xy_value_l3020_302057

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a/3 + b/4 = 1 → x*y ≥ a*b ∧ x*y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3020_302057


namespace NUMINAMATH_CALUDE_sum_of_digits_twice_square_222222222_l3020_302099

def n : ℕ := 9

def Y : ℕ := 2 * (10^n - 1) / 9

def Y_squared : ℕ := Y * Y

def doubled_Y_squared : ℕ := 2 * Y_squared

def sum_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else x % 10 + sum_of_digits (x / 10)

theorem sum_of_digits_twice_square_222222222 :
  sum_of_digits doubled_Y_squared = 126 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_twice_square_222222222_l3020_302099


namespace NUMINAMATH_CALUDE_fifth_basket_price_l3020_302068

/-- Given 4 baskets with an average cost of $4 and a fifth basket that makes
    the average cost of all 5 baskets $4.8, the price of the fifth basket is $8. -/
theorem fifth_basket_price (num_initial_baskets : Nat) (initial_avg_cost : ℝ)
    (total_baskets : Nat) (new_avg_cost : ℝ) :
    num_initial_baskets = 4 →
    initial_avg_cost = 4 →
    total_baskets = 5 →
    new_avg_cost = 4.8 →
    (total_baskets * new_avg_cost - num_initial_baskets * initial_avg_cost) = 8 := by
  sorry

#check fifth_basket_price

end NUMINAMATH_CALUDE_fifth_basket_price_l3020_302068


namespace NUMINAMATH_CALUDE_sallys_cards_l3020_302044

/-- Sally's card counting problem -/
theorem sallys_cards (initial : ℕ) (dans_gift : ℕ) (sallys_purchase : ℕ) : 
  initial = 27 → dans_gift = 41 → sallys_purchase = 20 → 
  initial + dans_gift + sallys_purchase = 88 := by
  sorry

end NUMINAMATH_CALUDE_sallys_cards_l3020_302044


namespace NUMINAMATH_CALUDE_rhombus_area_l3020_302038

/-- The area of a rhombus with side length 13 units and one interior angle of 60 degrees is (169√3)/2 square units. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 13) (h2 : θ = π / 3) :
  s^2 * Real.sin θ = (169 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3020_302038


namespace NUMINAMATH_CALUDE_sidewalk_snow_volume_l3020_302005

theorem sidewalk_snow_volume (length width height : ℝ) 
  (h1 : length = 15)
  (h2 : width = 3)
  (h3 : height = 0.6) :
  length * width * height = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_sidewalk_snow_volume_l3020_302005


namespace NUMINAMATH_CALUDE_cos_135_degrees_l3020_302045

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l3020_302045


namespace NUMINAMATH_CALUDE_light_distance_500_years_l3020_302003

theorem light_distance_500_years :
  let distance_one_year : ℝ := 5870000000000
  let years : ℕ := 500
  (distance_one_year * years : ℝ) = 2.935 * (10 ^ 15) :=
by sorry

end NUMINAMATH_CALUDE_light_distance_500_years_l3020_302003


namespace NUMINAMATH_CALUDE_x_squared_greater_than_x_l3020_302065

theorem x_squared_greater_than_x (x : ℝ) :
  (x > 1 → x^2 > x) ∧ ¬(x^2 > x → x > 1) := by sorry

end NUMINAMATH_CALUDE_x_squared_greater_than_x_l3020_302065


namespace NUMINAMATH_CALUDE_num_factors_180_multiple_15_eq_6_l3020_302026

/-- The number of positive factors of 180 that are also multiples of 15 -/
def num_factors_180_multiple_15 : ℕ :=
  (Finset.filter (λ x => 180 % x = 0 ∧ x % 15 = 0) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 that are also multiples of 15 is 6 -/
theorem num_factors_180_multiple_15_eq_6 : num_factors_180_multiple_15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_180_multiple_15_eq_6_l3020_302026


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l3020_302031

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), p = 73 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ (Nat.choose 150 75) ∧
    ∀ q : ℕ, Prime q → 10 ≤ q → q < 100 → q ∣ (Nat.choose 150 75) → q ≤ p :=
by sorry


end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l3020_302031


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l3020_302077

theorem mod_eight_equivalence (m : ℕ) : 
  13^7 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l3020_302077


namespace NUMINAMATH_CALUDE_only_setC_in_proportion_l3020_302094

-- Define a structure for a set of four line segments
structure FourSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property of being in proportion
def isInProportion (segments : FourSegments) : Prop :=
  segments.a * segments.d = segments.b * segments.c

-- Define the four sets of line segments
def setA : FourSegments := ⟨3, 5, 6, 9⟩
def setB : FourSegments := ⟨3, 5, 8, 9⟩
def setC : FourSegments := ⟨3, 9, 10, 30⟩
def setD : FourSegments := ⟨3, 6, 7, 9⟩

-- State the theorem
theorem only_setC_in_proportion :
  isInProportion setC ∧
  ¬isInProportion setA ∧
  ¬isInProportion setB ∧
  ¬isInProportion setD :=
sorry

end NUMINAMATH_CALUDE_only_setC_in_proportion_l3020_302094


namespace NUMINAMATH_CALUDE_difference_of_squares_l3020_302050

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3020_302050


namespace NUMINAMATH_CALUDE_union_p_complement_q_l3020_302048

-- Define the set P
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 4*x - 3)}

-- Define the set Q
def Q : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem union_p_complement_q :
  P ∪ (Set.univ \ Q) = Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_union_p_complement_q_l3020_302048


namespace NUMINAMATH_CALUDE_movie_expenses_split_l3020_302000

theorem movie_expenses_split (num_friends : ℕ) (ticket_price popcorn_price parking_fee milk_tea_price candy_bar_price : ℚ)
  (num_tickets num_popcorn num_milk_tea num_candy_bars : ℕ) :
  num_friends = 4 ∧
  ticket_price = 7 ∧
  popcorn_price = 3/2 ∧
  parking_fee = 4 ∧
  milk_tea_price = 3 ∧
  candy_bar_price = 2 ∧
  num_tickets = 4 ∧
  num_popcorn = 2 ∧
  num_milk_tea = 3 ∧
  num_candy_bars = 4 →
  (num_tickets * ticket_price + num_popcorn * popcorn_price + parking_fee +
   num_milk_tea * milk_tea_price + num_candy_bars * candy_bar_price) / num_friends = 13 :=
by sorry

end NUMINAMATH_CALUDE_movie_expenses_split_l3020_302000


namespace NUMINAMATH_CALUDE_highest_score_l3020_302010

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (bc_gt_ad : b + c > a + d)
  (a_gt_bd : a > b + d) :
  c > a ∧ c > b ∧ c > d :=
sorry

end NUMINAMATH_CALUDE_highest_score_l3020_302010


namespace NUMINAMATH_CALUDE_price_increase_percentage_l3020_302021

/-- Proves that the percentage increase in prices is 15% given the problem conditions -/
theorem price_increase_percentage (orange_price : ℝ) (mango_price : ℝ) (new_total_cost : ℝ) :
  orange_price = 40 →
  mango_price = 50 →
  new_total_cost = 1035 →
  10 * (orange_price * (1 + 15 / 100)) + 10 * (mango_price * (1 + 15 / 100)) = new_total_cost :=
by sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l3020_302021


namespace NUMINAMATH_CALUDE_basic_computer_price_is_correct_l3020_302093

/-- The price of the basic computer -/
def basic_computer_price : ℝ := 1040

/-- The price of the printer -/
def printer_price : ℝ := 2500 - basic_computer_price

/-- The total price of the basic computer and printer -/
def total_price : ℝ := 2500

/-- The price of the first enhanced computer -/
def enhanced_computer1_price : ℝ := basic_computer_price + 800

/-- The price of the second enhanced computer -/
def enhanced_computer2_price : ℝ := basic_computer_price + 1100

/-- The price of the third enhanced computer -/
def enhanced_computer3_price : ℝ := basic_computer_price + 1500

theorem basic_computer_price_is_correct :
  basic_computer_price + printer_price = total_price ∧
  enhanced_computer1_price + (1/5) * (enhanced_computer1_price + printer_price) = total_price ∧
  enhanced_computer2_price + (1/8) * (enhanced_computer2_price + printer_price) = total_price ∧
  enhanced_computer3_price + (1/10) * (enhanced_computer3_price + printer_price) = total_price :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_is_correct_l3020_302093


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3020_302079

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_sa : ℝ) : 
  l = 12 → w = 3 → h = 24 → 
  cube_sa = 6 * (l * w * h) ^ (2/3) →
  cube_sa = 545.02 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3020_302079


namespace NUMINAMATH_CALUDE_max_m_value_l3020_302081

/-- A point in the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Definition of a valid configuration -/
def ValidConfig (n : ℕ) (m : ℕ) (points : Fin (m + 2) → Point) : Prop :=
  (n % 2 = 1) ∧ 
  (points 0 = ⟨0, 1⟩) ∧ 
  (points (Fin.last m) = ⟨n + 1, n⟩) ∧ 
  (∀ i : Fin m, 1 ≤ (points i.succ).x ∧ (points i.succ).x ≤ n ∧ 
                1 ≤ (points i.succ).y ∧ (points i.succ).y ≤ n) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 0 → (points i).y = (points i.succ).y) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 1 → (points i).x = (points i.succ).x) ∧
  (∀ i j : Fin (m + 1), i < j → 
    ((points i).x = (points i.succ).x ∧ (points j).x = (points j.succ).x → 
      (points i).x ≠ (points j).x) ∨
    ((points i).y = (points i.succ).y ∧ (points j).y = (points j.succ).y → 
      (points i).y ≠ (points j).y))

/-- The main theorem -/
theorem max_m_value (n : ℕ) : 
  (n % 2 = 1) → (∃ m : ℕ, ∃ points : Fin (m + 2) → Point, ValidConfig n m points) → 
  (∀ k : ℕ, ∀ points : Fin (k + 2) → Point, ValidConfig n k points → k ≤ n * (n - 1)) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3020_302081


namespace NUMINAMATH_CALUDE_mango_juice_cost_l3020_302024

/-- The cost of a big bottle of mango juice in pesetas -/
def big_bottle_cost : ℕ := 2700

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℕ := 30

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℕ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℕ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def saving : ℕ := 300

theorem mango_juice_cost :
  big_bottle_cost = 
    (big_bottle_volume / small_bottle_volume) * small_bottle_cost - saving :=
by sorry

end NUMINAMATH_CALUDE_mango_juice_cost_l3020_302024


namespace NUMINAMATH_CALUDE_alicia_local_tax_cents_l3020_302086

/-- Calculates the amount of cents per hour used to pay local taxes given an hourly wage in dollars and a tax rate percentage. -/
def localTaxCents (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Proves that given Alicia's hourly wage of 25 dollars and a 2% local tax deduction, 
    the amount of cents per hour used to pay local taxes is 50. -/
theorem alicia_local_tax_cents : 
  localTaxCents 25 2 = 50 := by
  sorry

#eval localTaxCents 25 2

end NUMINAMATH_CALUDE_alicia_local_tax_cents_l3020_302086


namespace NUMINAMATH_CALUDE_stickers_after_birthday_l3020_302072

def initial_stickers : ℕ := 39
def birthday_stickers : ℕ := 22

theorem stickers_after_birthday :
  initial_stickers + birthday_stickers = 61 := by
  sorry

end NUMINAMATH_CALUDE_stickers_after_birthday_l3020_302072


namespace NUMINAMATH_CALUDE_cubic_root_b_value_l3020_302023

theorem cubic_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) →
  b = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_b_value_l3020_302023


namespace NUMINAMATH_CALUDE_power_sum_equals_three_l3020_302012

theorem power_sum_equals_three (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3) :
  a^2008 + b^2008 + c^2008 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_three_l3020_302012


namespace NUMINAMATH_CALUDE_number_relationship_l3020_302089

theorem number_relationship (A B C : ℕ) : 
  A + B + C = 660 → A = 2 * B → B = 180 → C = A - 240 := by sorry

end NUMINAMATH_CALUDE_number_relationship_l3020_302089


namespace NUMINAMATH_CALUDE_trash_bin_charge_is_10_l3020_302071

/-- Represents the garbage bill calculation -/
def garbage_bill (T : ℚ) : Prop :=
  let weeks : ℕ := 4
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let recycling_charge : ℚ := 5
  let discount_rate : ℚ := 0.18
  let fine : ℚ := 20
  let final_bill : ℚ := 102

  let pre_discount := weeks * (trash_bins * T + recycling_bins * recycling_charge)
  let discount := discount_rate * pre_discount
  let post_discount := pre_discount - discount
  let total_bill := post_discount + fine

  total_bill = final_bill

/-- Theorem stating that the charge per trash bin is $10 -/
theorem trash_bin_charge_is_10 : garbage_bill 10 := by
  sorry

end NUMINAMATH_CALUDE_trash_bin_charge_is_10_l3020_302071


namespace NUMINAMATH_CALUDE_darla_electricity_payment_l3020_302007

/-- The number of watts of electricity Darla needs to pay for -/
def watts : ℝ := 300

/-- The cost per watt of electricity in dollars -/
def cost_per_watt : ℝ := 4

/-- The late fee in dollars -/
def late_fee : ℝ := 150

/-- The total payment in dollars -/
def total_payment : ℝ := 1350

theorem darla_electricity_payment :
  cost_per_watt * watts + late_fee = total_payment := by
  sorry

end NUMINAMATH_CALUDE_darla_electricity_payment_l3020_302007


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3020_302067

theorem fraction_equivalence : 
  ∀ (n : ℚ), (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3020_302067


namespace NUMINAMATH_CALUDE_system_solution_l3020_302062

theorem system_solution (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 2*b)
  (eq3 : a + b = 9 - 5*c) :
  3*a + 3*b + 3*c = 18/17 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3020_302062


namespace NUMINAMATH_CALUDE_range_of_u_l3020_302022

theorem range_of_u (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  1 ≤ |2*x + y - 4| + |3 - x - 2*y| ∧ |2*x + y - 4| + |3 - x - 2*y| ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_u_l3020_302022


namespace NUMINAMATH_CALUDE_gift_purchase_solution_l3020_302069

/-- Pricing function based on quantity --/
def price (q : ℕ) : ℚ :=
  if q ≤ 120 then 3.5
  else if q ≤ 300 then 3.2
  else 3

/-- Total cost for a given quantity --/
def total_cost (q : ℕ) : ℚ :=
  if q ≤ 120 then q * price q
  else if q ≤ 300 then 120 * 3.5 + (q - 120) * price q
  else 120 * 3.5 + 180 * 3.2 + (q - 300) * price q

/-- Theorem stating the correctness of the solution --/
theorem gift_purchase_solution :
  let xiaoli_units : ℕ := 290
  let xiaowang_units : ℕ := 110
  xiaoli_units + xiaowang_units = 400 ∧
  xiaoli_units > 280 ∧
  total_cost xiaoli_units + total_cost xiaowang_units = 1349 :=
by sorry

end NUMINAMATH_CALUDE_gift_purchase_solution_l3020_302069


namespace NUMINAMATH_CALUDE_pipe_A_rate_correct_l3020_302028

/-- Represents the rate at which pipe A fills the tank -/
def pipe_A_rate : ℝ := 40

/-- Represents the rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- Represents the rate at which pipe C drains the tank -/
def pipe_C_rate : ℝ := 20

/-- Represents the capacity of the tank -/
def tank_capacity : ℝ := 850

/-- Represents the time it takes to fill the tank -/
def fill_time : ℝ := 51

/-- Represents the duration of one cycle -/
def cycle_duration : ℝ := 3

/-- Theorem stating that pipe A's rate satisfies the given conditions -/
theorem pipe_A_rate_correct : 
  (fill_time / cycle_duration) * (pipe_A_rate + pipe_B_rate - pipe_C_rate) = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_pipe_A_rate_correct_l3020_302028


namespace NUMINAMATH_CALUDE_max_d_value_l3020_302078

def a (n : ℕ+) : ℕ := 120 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 481) ∧ (∀ (n : ℕ+), d n ≤ 481) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3020_302078


namespace NUMINAMATH_CALUDE_even_function_coefficient_l3020_302060

theorem even_function_coefficient (a : ℝ) :
  (∀ x : ℝ, (fun x => x^2 + a*x + 1) x = (fun x => x^2 + a*x + 1) (-x)) →
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_coefficient_l3020_302060


namespace NUMINAMATH_CALUDE_correct_substitution_l3020_302025

/-- Given a system of equations { y = 1 - x, x - 2y = 4 }, 
    the correct substitution using the substitution method is x - 2 + 2x = 4 -/
theorem correct_substitution (x y : ℝ) : 
  (y = 1 - x ∧ x - 2*y = 4) → (x - 2 + 2*x = 4) :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_l3020_302025


namespace NUMINAMATH_CALUDE_john_cookies_problem_l3020_302055

theorem john_cookies_problem (cookies_left : ℕ) (cookies_eaten : ℕ) (dozen : ℕ) :
  cookies_left = 21 →
  cookies_eaten = 3 →
  dozen = 12 →
  (cookies_left + cookies_eaten) / dozen = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_cookies_problem_l3020_302055


namespace NUMINAMATH_CALUDE_bead_division_problem_l3020_302002

/-- The number of equal parts into which the beads were divided -/
def n : ℕ := sorry

/-- The total number of beads -/
def total_beads : ℕ := 23 + 16

/-- The number of beads in each part after division but before removal -/
def beads_per_part : ℚ := total_beads / n

/-- The number of beads in each part after removal but before doubling -/
def beads_after_removal : ℚ := beads_per_part - 10

/-- The final number of beads in each part after doubling -/
def final_beads : ℕ := 6

theorem bead_division_problem :
  2 * beads_after_removal = final_beads ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_bead_division_problem_l3020_302002


namespace NUMINAMATH_CALUDE_faster_train_speed_l3020_302075

theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time taken to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3020_302075


namespace NUMINAMATH_CALUDE_population_growth_l3020_302059

theorem population_growth (initial_population : ℝ) :
  let growth_factor1 := 1 + 0.05
  let growth_factor2 := 1 + 0.10
  let growth_factor3 := 1 + 0.15
  let final_population := initial_population * growth_factor1 * growth_factor2 * growth_factor3
  (final_population - initial_population) / initial_population * 100 = 33.075 := by
sorry

end NUMINAMATH_CALUDE_population_growth_l3020_302059


namespace NUMINAMATH_CALUDE_expression_simplification_l3020_302056

/-- Proves that the given expression simplifies to 1 when a = 1 and b = -2 -/
theorem expression_simplification (a b : ℤ) (ha : a = 1) (hb : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_expression_simplification_l3020_302056


namespace NUMINAMATH_CALUDE_abc_inequality_l3020_302061

theorem abc_inequality (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) > 
  a^(α + 2) * (b + c - a) + b^(α + 2) * (a - b + c) + c^(α + 2) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3020_302061


namespace NUMINAMATH_CALUDE_second_bouquet_carnations_proof_l3020_302006

/-- The number of carnations in the second bouquet -/
def second_bouquet_carnations : ℕ := 14

/-- The number of bouquets -/
def num_bouquets : ℕ := 3

/-- The number of carnations in the first bouquet -/
def first_bouquet_carnations : ℕ := 9

/-- The number of carnations in the third bouquet -/
def third_bouquet_carnations : ℕ := 13

/-- The average number of carnations per bouquet -/
def average_carnations : ℕ := 12

theorem second_bouquet_carnations_proof :
  (first_bouquet_carnations + second_bouquet_carnations + third_bouquet_carnations) / num_bouquets = average_carnations :=
by sorry

end NUMINAMATH_CALUDE_second_bouquet_carnations_proof_l3020_302006


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3020_302014

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 5*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-71, -42, -24, -14, 4, 14, 22, 41} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3020_302014


namespace NUMINAMATH_CALUDE_total_days_2010_to_2013_l3020_302011

/-- A year is a leap year if it's divisible by 4, except for century years,
    which must be divisible by 400 to be a leap year. -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- The number of days in a given year -/
def daysInYear (year : ℕ) : ℕ :=
  if isLeapYear year then 366 else 365

/-- The range of years we're considering -/
def yearRange : List ℕ := [2010, 2011, 2012, 2013]

theorem total_days_2010_to_2013 :
  (yearRange.map daysInYear).sum = 1461 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2013_l3020_302011


namespace NUMINAMATH_CALUDE_rabbit_hit_probability_l3020_302090

/-- The probability that at least one hunter hits the rabbit. -/
def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem: Given three hunters with hit probabilities 0.6, 0.5, and 0.4,
    the probability that the rabbit is hit is 0.88. -/
theorem rabbit_hit_probability :
  probability_hit 0.6 0.5 0.4 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_hit_probability_l3020_302090


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3020_302066

/-- Given a geometric sequence with first term b₁ = 2, the minimum value of 3b₂ + 6b₃ is -3/4,
    where b₂ and b₃ are the second and third terms of the sequence respectively. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ r : ℝ, b₂ = 2 * r → b₃ = 2 * r^2 → 3 * b₂ + 6 * b₃ ≥ -3/4) ∧ 
  (∃ r : ℝ, b₂ = 2 * r ∧ b₃ = 2 * r^2 ∧ 3 * b₂ + 6 * b₃ = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3020_302066


namespace NUMINAMATH_CALUDE_only_solution_is_3_2_l3020_302084

/-- The expression is an integer for the given prime pair -/
def is_integer_expr (p q : ℕ) : Prop :=
  ∃ k : ℤ, (↑((p + q)^(p + q) * (p - q)^(p - q) - 1) : ℤ) = 
           k * (↑((p + q)^(p - q) * (p - q)^(p + q) - 1) : ℤ)

/-- The main theorem stating that (3, 2) is the only solution -/
theorem only_solution_is_3_2 :
  ∀ p q : ℕ, Prime p → Prime q → p > q → is_integer_expr p q → (p = 3 ∧ q = 2) :=
by sorry

end NUMINAMATH_CALUDE_only_solution_is_3_2_l3020_302084


namespace NUMINAMATH_CALUDE_simplify_fraction_l3020_302035

theorem simplify_fraction (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  (18 * a * b^3 * c^2) / (12 * a^2 * b * c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3020_302035


namespace NUMINAMATH_CALUDE_movie_savings_theorem_l3020_302016

/-- Represents the savings calculation for a movie outing --/
def movie_savings (regular_price : ℚ) (student_discount : ℚ) (senior_discount : ℚ) 
  (early_discount : ℚ) (popcorn_price : ℚ) (popcorn_discount : ℚ) 
  (nachos_price : ℚ) (nachos_discount : ℚ) (hotdog_price : ℚ) 
  (hotdog_discount : ℚ) (combo_discount : ℚ) : ℚ :=
  let regular_tickets := 2 * regular_price
  let student_ticket := regular_price - student_discount
  let senior_ticket := regular_price - senior_discount
  let early_factor := 1 - early_discount
  let early_tickets := (regular_tickets + student_ticket + senior_ticket) * early_factor
  let ticket_savings := regular_tickets + student_ticket + senior_ticket - early_tickets
  let food_regular := popcorn_price + nachos_price + hotdog_price
  let food_discounted := popcorn_price * (1 - popcorn_discount) + 
                         nachos_price * (1 - nachos_discount) + 
                         hotdog_price * (1 - hotdog_discount)
  let food_combo := popcorn_price * (1 - popcorn_discount) + 
                    nachos_price * (1 - nachos_discount) + 
                    hotdog_price * (1 - hotdog_discount) * (1 - combo_discount)
  let food_savings := food_regular - food_combo
  ticket_savings + food_savings

/-- The total savings for the movie outing is $16.80 --/
theorem movie_savings_theorem : 
  movie_savings 10 2 3 (1/5) 10 (1/2) 8 (3/10) 6 (1/5) (1/4) = 84/5 := by
  sorry

end NUMINAMATH_CALUDE_movie_savings_theorem_l3020_302016


namespace NUMINAMATH_CALUDE_sally_nickels_l3020_302095

-- Define the initial state and gifts
def initial_nickels : ℕ := 7
def dad_gift : ℕ := 9
def mom_gift : ℕ := 2

-- Theorem to prove
theorem sally_nickels : initial_nickels + dad_gift + mom_gift = 18 := by
  sorry

end NUMINAMATH_CALUDE_sally_nickels_l3020_302095


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_6_l3020_302083

theorem gcf_lcm_sum_4_6 : Nat.gcd 4 6 + Nat.lcm 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_6_l3020_302083


namespace NUMINAMATH_CALUDE_johns_investment_l3020_302040

theorem johns_investment (total_investment : ℝ) (rate_a rate_b : ℝ) (investment_a : ℝ) (final_amount : ℝ) :
  total_investment = 1500 →
  rate_a = 0.04 →
  rate_b = 0.06 →
  investment_a = 750 →
  final_amount = 1575 →
  investment_a * (1 + rate_a) + (total_investment - investment_a) * (1 + rate_b) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_johns_investment_l3020_302040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3020_302092

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  a4_eq : a 4 = -2
  S10_eq : S 10 = 25
  arith_seq : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 14) ∧
  (seq.S 4 = -26 ∧ ∀ n : ℕ, seq.S n ≥ -26) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3020_302092


namespace NUMINAMATH_CALUDE_other_number_is_two_l3020_302001

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem other_number_is_two :
  ∃ n : ℕ, factorial 8 / factorial (8 - n) = 56 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_two_l3020_302001


namespace NUMINAMATH_CALUDE_expedition_max_distance_l3020_302019

/-- Represents the state of the expedition --/
structure ExpeditionState where
  participants : Nat
  distance : Nat
  fuel_per_car : Nat

/-- Calculates the maximum distance the expedition can travel --/
def max_distance (initial_state : ExpeditionState) : Nat :=
  sorry

/-- Theorem stating the maximum distance the expedition can travel --/
theorem expedition_max_distance :
  let initial_state : ExpeditionState := {
    participants := 9,
    distance := 0,
    fuel_per_car := 10  -- 1 gallon in tank + 9 additional cans
  }
  max_distance initial_state = 360 := by
  sorry

end NUMINAMATH_CALUDE_expedition_max_distance_l3020_302019


namespace NUMINAMATH_CALUDE_petes_number_l3020_302037

theorem petes_number (x : ℚ) : 3 * (x + 15) - 5 = 125 → x = 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l3020_302037


namespace NUMINAMATH_CALUDE_championship_probability_l3020_302039

def is_win_for_A (n : ℕ) : Bool :=
  n ≤ 5

def count_wins_A (numbers : List ℕ) : ℕ :=
  (numbers.filter is_win_for_A).length

def estimate_probability (wins : ℕ) (total : ℕ) : ℚ :=
  ↑wins / ↑total

def generated_numbers : List ℕ := [1, 9, 2, 9, 0, 7, 9, 6, 6, 9, 2, 5, 2, 7, 1, 9, 3, 2, 8, 1, 2, 6, 7, 3, 9, 3, 1, 2, 7, 5, 5, 6, 4, 8, 8, 7, 3, 0, 1, 1, 3, 5, 3, 7, 9, 8, 9, 4, 3, 1]

theorem championship_probability :
  estimate_probability (count_wins_A generated_numbers) generated_numbers.length = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_championship_probability_l3020_302039


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3020_302087

theorem sum_product_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3020_302087


namespace NUMINAMATH_CALUDE_cubic_root_product_l3020_302063

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if the product of any two roots equals 3, then c = 3a -/
theorem cubic_root_product (a b c d : ℝ) (ha : a ≠ 0) :
  (∃ r s t : ℝ, r * s = 3 ∧ r * t = 3 ∧ s * t = 3 ∧
    a * r^3 + b * r^2 + c * r + d = 0 ∧
    a * s^3 + b * s^2 + c * s + d = 0 ∧
    a * t^3 + b * t^2 + c * t + d = 0) →
  c = 3 * a :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3020_302063


namespace NUMINAMATH_CALUDE_fraction_simplification_l3020_302027

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3020_302027


namespace NUMINAMATH_CALUDE_men_in_first_group_l3020_302015

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 6

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 12

/-- The theorem stating that the number of men in the first group is 10 -/
theorem men_in_first_group : 
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group * hours_per_day = 
    men_second_group * days_second_group * hours_per_day ∧
    men_first_group = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_men_in_first_group_l3020_302015


namespace NUMINAMATH_CALUDE_arrange_5_balls_4_boxes_l3020_302070

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def arrange_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem arrange_5_balls_4_boxes : arrange_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_arrange_5_balls_4_boxes_l3020_302070


namespace NUMINAMATH_CALUDE_total_seeds_equals_685_l3020_302004

/- Morning plantings -/
def mike_morning_tomato : ℕ := 50
def mike_morning_pepper : ℕ := 30
def ted_morning_tomato : ℕ := 2 * mike_morning_tomato
def ted_morning_pepper : ℕ := mike_morning_pepper / 2
def sarah_morning_tomato : ℕ := mike_morning_tomato + 30
def sarah_morning_pepper : ℕ := mike_morning_pepper + 30

/- Afternoon plantings -/
def mike_afternoon_tomato : ℕ := 60
def mike_afternoon_pepper : ℕ := 40
def ted_afternoon_tomato : ℕ := mike_afternoon_tomato - 20
def ted_afternoon_pepper : ℕ := mike_afternoon_pepper
def sarah_afternoon_tomato : ℕ := sarah_morning_tomato + 20
def sarah_afternoon_pepper : ℕ := sarah_morning_pepper + 10

/- Total seeds planted -/
def total_seeds : ℕ := 
  mike_morning_tomato + mike_morning_pepper + 
  ted_morning_tomato + ted_morning_pepper + 
  sarah_morning_tomato + sarah_morning_pepper + 
  mike_afternoon_tomato + mike_afternoon_pepper + 
  ted_afternoon_tomato + ted_afternoon_pepper + 
  sarah_afternoon_tomato + sarah_afternoon_pepper

theorem total_seeds_equals_685 : total_seeds = 685 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_equals_685_l3020_302004


namespace NUMINAMATH_CALUDE_simplify_tan_product_l3020_302053

-- Define the tangent function
noncomputable def tan (x : Real) : Real := Real.tan x

-- State the theorem
theorem simplify_tan_product : 
  (1 + tan (10 * Real.pi / 180)) * (1 + tan (35 * Real.pi / 180)) = 2 := by
  -- Assuming the angle addition formula for tangent
  have angle_addition_formula : ∀ a b, 
    tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry
  
  -- Assuming tan 45° = 1
  have tan_45_deg : tan (45 * Real.pi / 180) = 1 := by sorry

  sorry -- The proof goes here

end NUMINAMATH_CALUDE_simplify_tan_product_l3020_302053


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3020_302032

theorem reciprocal_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1 / x < 1 / y := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3020_302032


namespace NUMINAMATH_CALUDE_second_chapter_pages_l3020_302051

theorem second_chapter_pages (total_pages first_chapter third_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : third_chapter = 24) :
  total_pages - first_chapter - third_chapter = 59 :=
by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l3020_302051


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l3020_302085

theorem cookie_ratio_proof (initial_white : ℕ) (total_remaining : ℕ) : 
  initial_white = 80 →
  total_remaining = 85 →
  let initial_black := initial_white + 50
  let remaining_white := initial_white / 4
  let remaining_black := total_remaining - remaining_white
  let black_eaten := initial_black - remaining_black
  (black_eaten : ℚ) / initial_black = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l3020_302085


namespace NUMINAMATH_CALUDE_egg_grouping_l3020_302036

theorem egg_grouping (total_eggs : ℕ) (eggs_per_group : ℕ) (groups : ℕ) : 
  total_eggs = 8 → eggs_per_group = 2 → groups = total_eggs / eggs_per_group → groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_egg_grouping_l3020_302036


namespace NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_sequence_l3020_302097

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem twelfth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a 2) 
  (h_first : a 1 = 1) : 
  a 12 = 23 := by
sorry

end NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_sequence_l3020_302097


namespace NUMINAMATH_CALUDE_smallest_y_absolute_value_l3020_302030

theorem smallest_y_absolute_value (y : ℝ) : 
  (|y - 8| = 15) → (∀ z, |z - 8| = 15 → y ≤ z) → y = -7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_absolute_value_l3020_302030


namespace NUMINAMATH_CALUDE_spotted_cats_ratio_l3020_302080

/-- Proves that the ratio of spotted cats to total cats is 1:3 -/
theorem spotted_cats_ratio (total_cats : ℕ) (spotted_fluffy : ℕ) :
  total_cats = 120 →
  spotted_fluffy = 10 →
  (4 : ℚ) * spotted_fluffy = total_spotted →
  (total_spotted : ℚ) / total_cats = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_spotted_cats_ratio_l3020_302080


namespace NUMINAMATH_CALUDE_total_animals_l3020_302073

theorem total_animals (L C P R Q : ℕ) : 
  L = 10 → 
  C = 2 * L + 4 → 
  ∃ G : ℕ, G = 2 * (L + 3) + Q → 
  (L + C + P) + ((L + 3) + R * (L + 3) + G) = 73 + P + R * 13 + Q :=
by sorry

end NUMINAMATH_CALUDE_total_animals_l3020_302073


namespace NUMINAMATH_CALUDE_bookstore_change_percentage_l3020_302017

def book_prices : List ℝ := [10, 8, 6, 4, 3, 5]
def discount_rate : ℝ := 0.1
def payment_amount : ℝ := 50

theorem bookstore_change_percentage :
  let total_price := book_prices.sum
  let discounted_price := total_price * (1 - discount_rate)
  let change := payment_amount - discounted_price
  let change_percentage := (change / payment_amount) * 100
  change_percentage = 35.2 := by sorry

end NUMINAMATH_CALUDE_bookstore_change_percentage_l3020_302017


namespace NUMINAMATH_CALUDE_smallest_x_value_l3020_302098

theorem smallest_x_value (x y : ℕ+) (h : (0.8 : ℚ) = y / (186 + x)) : 
  x ≥ 4 ∧ ∃ (y' : ℕ+), (0.8 : ℚ) = y' / (186 + 4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3020_302098


namespace NUMINAMATH_CALUDE_max_terms_sum_to_target_l3020_302009

/-- The sequence of odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2*n + 1)

/-- The sum we're aiming for -/
def targetSum : Nat := 1949

/-- The maximum number of terms that sum to the target -/
def maxTerms : Nat := 44

theorem max_terms_sum_to_target :
  ∃ (subset : List Nat),
    subset.toFinset ⊆ oddSequence.toFinset ∧
    subset.sum = targetSum ∧
    subset.length = maxTerms ∧
    ∀ (otherSubset : List Nat),
      otherSubset.toFinset ⊆ oddSequence.toFinset →
      otherSubset.sum = targetSum →
      otherSubset.length ≤ maxTerms :=
by sorry

end NUMINAMATH_CALUDE_max_terms_sum_to_target_l3020_302009
