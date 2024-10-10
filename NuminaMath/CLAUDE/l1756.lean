import Mathlib

namespace max_value_of_operation_l1756_175629

theorem max_value_of_operation : 
  ∃ (max : ℕ), max = 1200 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 3 * (500 - n) ≤ max :=
by sorry

end max_value_of_operation_l1756_175629


namespace polynomial_factorization_l1756_175679

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 9*x + 20) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 6) * (x^2 + 6*x + 3) := by
sorry

end polynomial_factorization_l1756_175679


namespace modified_pyramid_volume_l1756_175695

/-- Given a pyramid with a square base and volume of 60 cubic inches, 
    if the base side length is tripled and the height is decreased by 25%, 
    the new volume will be 405 cubic inches. -/
theorem modified_pyramid_volume 
  (s : ℝ) (h : ℝ) 
  (original_volume : (1/3 : ℝ) * s^2 * h = 60) 
  (s_positive : s > 0) 
  (h_positive : h > 0) : 
  (1/3 : ℝ) * (3*s)^2 * (0.75*h) = 405 :=
by sorry

end modified_pyramid_volume_l1756_175695


namespace rectangle_to_square_l1756_175678

/-- A rectangle can be divided into two parts that form a square -/
theorem rectangle_to_square (length width : ℝ) (h1 : length = 9) (h2 : width = 4) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 6 := by
  sorry

end rectangle_to_square_l1756_175678


namespace expression_value_l1756_175641

theorem expression_value (x y : ℝ) (h : x^2 - 4*x + 4 + |y - 1| = 0) :
  (2*x - y)^2 - 2*(2*x - y)*(x + 2*y) + (x + 2*y)^2 = 1 := by
  sorry

end expression_value_l1756_175641


namespace octagon_diagonals_l1756_175635

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end octagon_diagonals_l1756_175635


namespace art_fair_customers_one_painting_l1756_175647

/-- The number of customers who bought one painting each at Tracy's art fair booth -/
def customers_one_painting (total_customers : ℕ) (two_painting_customers : ℕ) (four_painting_customers : ℕ) (total_paintings_sold : ℕ) : ℕ :=
  total_paintings_sold - (2 * two_painting_customers + 4 * four_painting_customers)

/-- Theorem stating that the number of customers who bought one painting each is 12 -/
theorem art_fair_customers_one_painting :
  customers_one_painting 20 4 4 36 = 12 := by
  sorry

#eval customers_one_painting 20 4 4 36

end art_fair_customers_one_painting_l1756_175647


namespace systematic_sample_theorem_l1756_175661

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  let interval := populationSize / sampleSize
  List.range sampleSize |>.map (fun i => start + i * interval)

/-- Theorem: In a systematic sample of size 4 from 56 items, if 6, 20, and 48 are in the sample, then 34 is the fourth number -/
theorem systematic_sample_theorem :
  ∀ (sample : List ℕ),
    sample = systematicSample 56 4 6 →
    sample.length = 4 →
    6 ∈ sample →
    20 ∈ sample →
    48 ∈ sample →
    34 ∈ sample := by
  sorry

#eval systematicSample 56 4 6

end systematic_sample_theorem_l1756_175661


namespace triangle_inequality_l1756_175670

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end triangle_inequality_l1756_175670


namespace product_equality_l1756_175613

theorem product_equality : 2.05 * 4.1 = 20.5 * 0.41 := by
  sorry

end product_equality_l1756_175613


namespace min_cookies_eaten_is_five_l1756_175664

/-- Represents the number of cookies Paco had, ate, and bought -/
structure CookieCount where
  initial : ℕ
  eaten_first : ℕ
  bought : ℕ
  eaten_second : ℕ

/-- The conditions of the cookie problem -/
def cookie_problem (c : CookieCount) : Prop :=
  c.initial = 25 ∧
  c.bought = 3 ∧
  c.eaten_second = c.bought + 2

/-- The minimum number of cookies Paco ate -/
def min_cookies_eaten (c : CookieCount) : ℕ :=
  c.eaten_second

/-- Theorem stating that the minimum number of cookies Paco ate is 5 -/
theorem min_cookies_eaten_is_five :
  ∀ c : CookieCount, cookie_problem c → min_cookies_eaten c = 5 :=
by
  sorry

end min_cookies_eaten_is_five_l1756_175664


namespace ram_ravi_selection_probability_l1756_175662

theorem ram_ravi_selection_probability :
  let p_ram : ℝ := 5/7
  let p_both : ℝ := 0.14285714285714288
  let p_ravi : ℝ := p_both / p_ram
  p_ravi = 0.2 := by sorry

end ram_ravi_selection_probability_l1756_175662


namespace greatest_number_with_odd_factors_under_200_l1756_175671

theorem greatest_number_with_odd_factors_under_200 : 
  ∃ n : ℕ, n = 196 ∧ 
  n < 200 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  (∀ m : ℕ, m < 200 → (∃ j : ℕ, m = j^2) → m ≤ n) :=
by sorry

end greatest_number_with_odd_factors_under_200_l1756_175671


namespace smallest_area_special_square_l1756_175648

/-- A square with two vertices on a line and two on a parabola -/
structure SpecialSquare where
  /-- The y-intercept of the line containing two vertices of the square -/
  k : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- Two vertices of the square lie on the line y = 3x - 5 -/
  line_constraint : ∃ (x₁ x₂ : ℝ), y = 3 * x₁ - 5 ∧ y = 3 * x₂ - 5
  /-- Two vertices of the square lie on the parabola y = x^2 + 4 -/
  parabola_constraint : ∃ (x₁ x₂ : ℝ), y = x₁^2 + 4 ∧ y = x₂^2 + 4
  /-- The square's sides are parallel/perpendicular to coordinate axes -/
  axis_aligned : True
  /-- The area of the square is s^2 -/
  area_eq : s^2 = 10 * (25 + 4 * k)

/-- The theorem stating the smallest possible area of the special square -/
theorem smallest_area_special_square :
  ∀ (sq : SpecialSquare), sq.s^2 ≥ 200 :=
sorry

end smallest_area_special_square_l1756_175648


namespace rectangular_field_dimensions_l1756_175600

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 := by
  sorry

end rectangular_field_dimensions_l1756_175600


namespace oddDigitSequence_157th_l1756_175682

/-- A function that generates the nth number in the sequence of positive integers formed only by odd digits -/
def oddDigitSequence (n : ℕ) : ℕ :=
  sorry

/-- The set of odd digits -/
def oddDigits : Set ℕ := {1, 3, 5, 7, 9}

/-- A predicate to check if a number consists only of odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ∈ oddDigits

/-- The main theorem stating that the 157th number in the sequence is 1113 -/
theorem oddDigitSequence_157th :
  oddDigitSequence 157 = 1113 ∧ hasOnlyOddDigits (oddDigitSequence 157) :=
sorry

end oddDigitSequence_157th_l1756_175682


namespace student_arrangement_equality_l1756_175663

theorem student_arrangement_equality (n : ℕ) : 
  n = 48 → 
  (Nat.factorial n) = (Nat.factorial n) :=
by
  sorry

end student_arrangement_equality_l1756_175663


namespace nicks_sister_age_difference_l1756_175681

theorem nicks_sister_age_difference (nick_age : ℕ) (sister_age_diff : ℕ) : 
  nick_age = 13 →
  (nick_age + sister_age_diff) / 2 + 5 = 21 →
  sister_age_diff = 19 := by
  sorry

end nicks_sister_age_difference_l1756_175681


namespace inequality_range_of_p_l1756_175690

-- Define the inequality function
def inequality (x p : ℝ) : Prop := x^2 + p*x + 1 > 2*x + p

-- Define the theorem
theorem inequality_range_of_p :
  ∀ p : ℝ, (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → inequality x p) → p > -1 := by
  sorry

end inequality_range_of_p_l1756_175690


namespace abs_function_domain_range_intersection_l1756_175602

def A : Set ℝ := {-1, 0, 1}

def f (x : ℝ) : ℝ := |x|

theorem abs_function_domain_range_intersection :
  (A ∩ (f '' A)) = {0, 1} := by sorry

end abs_function_domain_range_intersection_l1756_175602


namespace cube_sum_divisibility_l1756_175612

theorem cube_sum_divisibility (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (9 ∣ (a₁^3 + a₂^3 + a₃^3 + a₄^3 + a₅^3)) → (3 ∣ (a₁ * a₂ * a₃ * a₄ * a₅)) := by
  sorry

end cube_sum_divisibility_l1756_175612


namespace expression_percentage_l1756_175630

theorem expression_percentage (x : ℝ) (h : x > 0) : 
  (x / 50 + x / 25 - x / 10 + x / 5) / x = 16 / 100 := by
  sorry

end expression_percentage_l1756_175630


namespace probability_both_boys_given_one_boy_l1756_175633

/-- Represents the gender of a child -/
inductive Gender
  | Boy
  | Girl

/-- Represents a family with two children -/
structure Family :=
  (child1 : Gender)
  (child2 : Gender)

/-- The set of all possible families with two children -/
def allFamilies : Finset Family :=
  sorry

/-- The set of families with at least one boy -/
def familiesWithBoy : Finset Family :=
  sorry

/-- The set of families with two boys -/
def familiesWithTwoBoys : Finset Family :=
  sorry

theorem probability_both_boys_given_one_boy :
    (familiesWithTwoBoys.card : ℚ) / familiesWithBoy.card = 1 / 2 := by
  sorry

end probability_both_boys_given_one_boy_l1756_175633


namespace car_distance_l1756_175614

/-- Given a total distance of 40 kilometers, if 1/4 of the distance is traveled by foot
    and 1/2 of the distance is traveled by bus, then the remaining distance traveled
    by car is 10 kilometers. -/
theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) 
    (h1 : total_distance = 40)
    (h2 : foot_fraction = 1/4)
    (h3 : bus_fraction = 1/2) :
    total_distance - (foot_fraction * total_distance) - (bus_fraction * total_distance) = 10 := by
  sorry


end car_distance_l1756_175614


namespace probability_both_selected_l1756_175624

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/7) 
  (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/35 := by
  sorry

end probability_both_selected_l1756_175624


namespace first_floor_units_count_l1756_175654

/-- A building with a specified number of floors and apartments -/
structure Building where
  floors : ℕ
  firstFloorUnits : ℕ
  otherFloorUnits : ℕ

/-- The total number of apartment units in a building -/
def totalUnits (b : Building) : ℕ :=
  b.firstFloorUnits + (b.floors - 1) * b.otherFloorUnits

theorem first_floor_units_count (b1 b2 : Building) :
  b1 = b2 ∧ 
  b1.floors = 4 ∧ 
  b1.otherFloorUnits = 5 ∧ 
  totalUnits b1 + totalUnits b2 = 34 →
  b1.firstFloorUnits = 2 :=
sorry

end first_floor_units_count_l1756_175654


namespace new_lift_count_correct_l1756_175668

/-- The number of times Terrell must lift the new weight configuration to match the total weight of the original configuration -/
def new_lift_count : ℕ := 12

/-- The weight of each item in the original configuration -/
def original_weight : ℕ := 12

/-- The number of weights in the original configuration -/
def original_count : ℕ := 3

/-- The number of times Terrell lifts the original configuration -/
def original_lifts : ℕ := 20

/-- The weights in the new configuration -/
def new_weights : List ℕ := [18, 18, 24]

theorem new_lift_count_correct :
  new_lift_count * (new_weights.sum) = original_weight * original_count * original_lifts :=
by sorry

end new_lift_count_correct_l1756_175668


namespace starting_lineup_with_twins_l1756_175640

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_with_twins (total_players : ℕ) (lineup_size : ℕ) (twin_count : ℕ) :
  total_players = 12 →
  lineup_size = 5 →
  twin_count = 2 →
  choose (total_players - twin_count) (lineup_size - twin_count) = 120 := by
  sorry

end starting_lineup_with_twins_l1756_175640


namespace order_of_products_and_square_l1756_175649

theorem order_of_products_and_square (x a b : ℝ) 
  (h1 : x < a) (h2 : a < b) (h3 : b < 0) : 
  b * x > a * x ∧ a * x > a ^ 2 :=
by sorry

end order_of_products_and_square_l1756_175649


namespace price_decrease_sales_increase_ratio_l1756_175657

theorem price_decrease_sales_increase_ratio (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end price_decrease_sales_increase_ratio_l1756_175657


namespace water_width_after_drop_l1756_175677

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -2*y

-- Define the point that the parabola passes through
def parabola_point : ℝ × ℝ := (2, -2)

-- Theorem to prove
theorem water_width_after_drop :
  parabola parabola_point.1 parabola_point.2 →
  ∀ x : ℝ, parabola x (-3) → 2 * |x| = 2 * Real.sqrt 6 :=
by sorry

end water_width_after_drop_l1756_175677


namespace imaginary_part_of_z_l1756_175622

/-- The imaginary part of 1 / (1 + i) is -1/2 -/
theorem imaginary_part_of_z (z : ℂ) : z = 1 / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end imaginary_part_of_z_l1756_175622


namespace min_value_abc_min_value_equals_one_over_nine_to_nine_l1756_175644

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 → 
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 :=
by sorry

theorem min_value_equals_one_over_nine_to_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  a^4 * b^3 * c^2 = 1 / 9^9 :=
by sorry

end min_value_abc_min_value_equals_one_over_nine_to_nine_l1756_175644


namespace f_recursive_relation_l1756_175699

/-- The smallest integer such that any permutation on n elements, repeated f(n) times, gives the identity. -/
def f (n : ℕ) : ℕ := sorry

/-- Checks if a number is a prime power -/
def isPrimePower (n : ℕ) : Prop := sorry

/-- The prime base of a prime power -/
def primeBase (n : ℕ) : ℕ := sorry

theorem f_recursive_relation (n : ℕ) :
  (isPrimePower n → f n = primeBase n * f (n - 1)) ∧
  (¬isPrimePower n → f n = f (n - 1)) := by sorry

end f_recursive_relation_l1756_175699


namespace intersection_conditions_l1756_175685

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_conditions (a : ℝ) :
  (9 ∈ A a ∩ B a ↔ a = 5 ∨ a = -3) ∧
  ({9} = A a ∩ B a ↔ a = -3) :=
sorry

end intersection_conditions_l1756_175685


namespace simplify_P_P_value_on_inverse_proportion_l1756_175697

/-- Simplification of the expression P -/
theorem simplify_P (a b : ℝ) :
  (2*a + 3*b)^2 - (2*a + b)*(2*a - b) - 2*b*(3*a + 5*b) = 6*a*b := by sorry

/-- Value of P when (a,b) lies on y = -2/x -/
theorem P_value_on_inverse_proportion (a b : ℝ) (h : a*b = -2) :
  6*a*b = -12 := by sorry

end simplify_P_P_value_on_inverse_proportion_l1756_175697


namespace g_zero_l1756_175638

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: g(6/5) = 0 -/
theorem g_zero : g (6 / 5) = 0 := by
  sorry

end g_zero_l1756_175638


namespace no_integer_roots_l1756_175619

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0) (ha_even : Even a) (hb_even : Even b) (hc_odd : Odd c) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end no_integer_roots_l1756_175619


namespace equidistant_point_on_y_axis_l1756_175684

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, 
    ((-3 : ℝ) - 0)^2 + (0 - y)^2 = ((-2 : ℝ) - 0)^2 + (5 - y)^2 ∧ 
    y = 2 := by
  sorry

end equidistant_point_on_y_axis_l1756_175684


namespace continuity_at_one_l1756_175691

def f (x : ℝ) := -3 * x^2 - 6

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end continuity_at_one_l1756_175691


namespace sum_of_coefficients_l1756_175607

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 5)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129 := by
sorry

end sum_of_coefficients_l1756_175607


namespace product_local_abs_value_l1756_175609

-- Define the complex number
def z : ℂ := 564823 + 3*Complex.I

-- Define the digit of interest
def digit : ℕ := 4

-- Define the local value of the digit in the complex number
def local_value : ℕ := 4000

-- Define the absolute value of the digit
def abs_digit : ℕ := 4

-- Theorem to prove
theorem product_local_abs_value : 
  local_value * abs_digit = 16000 := by sorry

end product_local_abs_value_l1756_175609


namespace gcd_71_19_l1756_175645

theorem gcd_71_19 : Nat.gcd 71 19 = 1 := by sorry

end gcd_71_19_l1756_175645


namespace arithmetic_sequence_inequality_l1756_175646

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end arithmetic_sequence_inequality_l1756_175646


namespace right_triangle_set_l1756_175680

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only one of the given sets forms a right triangle -/
theorem right_triangle_set :
  ¬(is_right_triangle 2 4 3) ∧
  ¬(is_right_triangle 6 8 9) ∧
  ¬(is_right_triangle 3 4 6) ∧
  is_right_triangle 1 1 (Real.sqrt 2) :=
sorry

end right_triangle_set_l1756_175680


namespace trader_donations_l1756_175655

theorem trader_donations (total_profit : ℝ) (goal_amount : ℝ) (above_goal : ℝ) : 
  total_profit = 960 → 
  goal_amount = 610 → 
  above_goal = 180 → 
  (goal_amount + above_goal) - (total_profit / 2) = 310 := by
sorry

end trader_donations_l1756_175655


namespace hyperbola_equation_l1756_175611

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (c : ℝ), c^2 = 5 * a^2 ∧ 
   ∃ (S : ℝ), S = 20 ∧ S = (1/2) * c * (4 * c)) →
  a^2 = 2 ∧ b^2 = 8 := by sorry

end hyperbola_equation_l1756_175611


namespace unique_integer_solution_l1756_175608

/-- The function f(x) = -x^2 + x + m + 2 -/
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

/-- The solution set of f(x) ≥ |x| -/
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | f x m ≥ |x|}

/-- The set of integers in the solution set -/
def integer_solutions (m : ℝ) : Set ℤ := {i : ℤ | (i : ℝ) ∈ solution_set m}

theorem unique_integer_solution (m : ℝ) :
  (∃! (i : ℤ), (i : ℝ) ∈ solution_set m) → -2 < m ∧ m < -1 :=
sorry

end unique_integer_solution_l1756_175608


namespace negation_equivalence_l1756_175672

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) ↔
  (∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) :=
by sorry

end negation_equivalence_l1756_175672


namespace arithmetic_sequence_middle_term_l1756_175669

theorem arithmetic_sequence_middle_term (a₁ a₃ y : ℤ) :
  a₁ = 3^2 →
  a₃ = 3^4 →
  y = (a₁ + a₃) / 2 →
  y = 45 :=
by sorry

end arithmetic_sequence_middle_term_l1756_175669


namespace probability_neither_cake_nor_muffin_l1756_175618

def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_buyers : ℕ := 16

theorem probability_neither_cake_nor_muffin :
  let buyers_of_at_least_one := cake_buyers + muffin_buyers - both_buyers
  let buyers_of_neither := total_buyers - buyers_of_at_least_one
  (buyers_of_neither : ℚ) / total_buyers = 26 / 100 := by
  sorry

end probability_neither_cake_nor_muffin_l1756_175618


namespace distribute_8_balls_3_boxes_l1756_175667

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes --/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball --/
def distribute_balls_nonempty (n : ℕ) (k : ℕ) : ℕ := distribute_balls (n - k) k

theorem distribute_8_balls_3_boxes : distribute_balls_nonempty 8 3 = 21 := by
  sorry

end distribute_8_balls_3_boxes_l1756_175667


namespace inequality_with_negative_multiplication_l1756_175603

theorem inequality_with_negative_multiplication 
  (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end inequality_with_negative_multiplication_l1756_175603


namespace inverse_proposition_correct_l1756_175658

/-- The statement of a geometric proposition -/
structure GeometricProposition :=
  (hypothesis : String)
  (conclusion : String)

/-- The inverse of a geometric proposition -/
def inverse_proposition (p : GeometricProposition) : GeometricProposition :=
  { hypothesis := p.conclusion,
    conclusion := p.hypothesis }

/-- The original proposition -/
def original_prop : GeometricProposition :=
  { hypothesis := "Triangles are congruent",
    conclusion := "Corresponding angles are equal" }

/-- Theorem stating that the inverse proposition is correct -/
theorem inverse_proposition_correct : 
  inverse_proposition original_prop = 
  { hypothesis := "Corresponding angles are equal",
    conclusion := "Triangles are congruent" } := by
  sorry


end inverse_proposition_correct_l1756_175658


namespace only_rectangle_both_symmetric_l1756_175643

-- Define the shape type
inductive Shape
  | EquilateralTriangle
  | Angle
  | Rectangle
  | Parallelogram

-- Define axisymmetry property
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Angle => true
  | Shape.Rectangle => true
  | Shape.Parallelogram => false

-- Define central symmetry property
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Angle => false
  | Shape.Rectangle => true
  | Shape.Parallelogram => true

-- Theorem stating that only Rectangle is both axisymmetric and centrally symmetric
theorem only_rectangle_both_symmetric :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end only_rectangle_both_symmetric_l1756_175643


namespace min_face_sum_is_16_l1756_175642

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- Check if a given arrangement satisfies the condition that the sum of any three vertices on a face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- Calculate the sum of numbers on a given face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The main theorem stating that the minimal possible sum on any face is 16 -/
theorem min_face_sum_is_16 :
  ∃ (arr : CubeArrangement), ValidArrangement arr ∧
    (∀ (arr' : CubeArrangement), ValidArrangement arr' →
      ∀ (face : Fin 6), FaceSum arr face ≤ FaceSum arr' face) ∧
    (∃ (face : Fin 6), FaceSum arr face = 16) :=
  sorry

end min_face_sum_is_16_l1756_175642


namespace cubic_sum_minus_product_l1756_175674

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end cubic_sum_minus_product_l1756_175674


namespace sum_of_r_p_x_is_negative_eleven_l1756_175692

def p (x : ℝ) : ℝ := |x| - 2

def r (x : ℝ) : ℝ := -|p x - 1|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_r_p_x_is_negative_eleven :
  (x_values.map (λ x => r (p x))).sum = -11 := by sorry

end sum_of_r_p_x_is_negative_eleven_l1756_175692


namespace expression_simplification_l1756_175683

theorem expression_simplification :
  (((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4)) = 125 / 12 := by
  sorry

end expression_simplification_l1756_175683


namespace smallest_reunion_time_l1756_175660

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_time (t : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.length ≥ 4 ∧ 
    subset.all (λ x => x ∈ horse_lap_times) ∧
    subset.all (λ x => t % x = 0)

theorem smallest_reunion_time :
  ∃ (T : ℕ), T > 0 ∧ is_valid_time T ∧
    ∀ (t : ℕ), 0 < t ∧ t < T → ¬is_valid_time t :=
  sorry

end smallest_reunion_time_l1756_175660


namespace fuel_cost_solution_l1756_175628

/-- Represents the fuel cost calculation problem --/
def fuel_cost_problem (truck_capacity : ℝ) (car_capacity : ℝ) (hybrid_capacity : ℝ)
  (truck_fullness : ℝ) (car_fullness : ℝ) (hybrid_fullness : ℝ)
  (diesel_price : ℝ) (gas_price : ℝ)
  (diesel_discount : ℝ) (gas_discount : ℝ) : Prop :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  let hybrid_to_fill := hybrid_capacity * (1 - hybrid_fullness)
  let diesel_discounted := diesel_price - diesel_discount
  let gas_discounted := gas_price - gas_discount
  let total_cost := truck_to_fill * diesel_discounted +
                    car_to_fill * gas_discounted +
                    hybrid_to_fill * gas_discounted
  total_cost = 95.88

/-- The main theorem stating the solution to the fuel cost problem --/
theorem fuel_cost_solution :
  fuel_cost_problem 25 15 10 0.5 (1/3) 0.25 3.5 3.2 0.1 0.15 :=
by sorry

end fuel_cost_solution_l1756_175628


namespace polynomial_factorization_l1756_175652

theorem polynomial_factorization (x : ℝ) : 
  75 * x^7 - 175 * x^13 = 25 * x^7 * (3 - 7 * x^6) := by
  sorry

end polynomial_factorization_l1756_175652


namespace roots_of_equation_l1756_175687

/-- The equation for which we need to find roots -/
def equation (x : ℝ) : Prop :=
  15 / (x^2 - 4) - 2 / (x - 2) = 1

/-- Theorem stating that -3 and 5 are the roots of the equation -/
theorem roots_of_equation :
  equation (-3) ∧ equation 5 :=
by sorry

end roots_of_equation_l1756_175687


namespace equation_solution_l1756_175686

theorem equation_solution (y : ℚ) : 
  (1 : ℚ) / 3 + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end equation_solution_l1756_175686


namespace average_percentages_correct_l1756_175621

-- Define the subjects
inductive Subject
  | English
  | Mathematics
  | Physics
  | Chemistry
  | Biology
  | History
  | Geography

-- Define the marks and total marks for each subject
def marks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 76
  | Subject.Mathematics => 65
  | Subject.Physics => 82
  | Subject.Chemistry => 67
  | Subject.Biology => 85
  | Subject.History => 92
  | Subject.Geography => 58

def totalMarks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 120
  | Subject.Mathematics => 150
  | Subject.Physics => 100
  | Subject.Chemistry => 80
  | Subject.Biology => 100
  | Subject.History => 150
  | Subject.Geography => 75

-- Define the average percentage calculation
def averagePercentage (s : Subject) : ℚ :=
  (marks s : ℚ) / (totalMarks s : ℚ) * 100

-- Theorem to prove the correctness of average percentages
theorem average_percentages_correct :
  averagePercentage Subject.English = 63.33 ∧
  averagePercentage Subject.Mathematics = 43.33 ∧
  averagePercentage Subject.Physics = 82 ∧
  averagePercentage Subject.Chemistry = 83.75 ∧
  averagePercentage Subject.Biology = 85 ∧
  averagePercentage Subject.History = 61.33 ∧
  averagePercentage Subject.Geography = 77.33 := by
  sorry


end average_percentages_correct_l1756_175621


namespace problem_statement_l1756_175653

theorem problem_statement (h : 125 = 5^3) : (125 : ℝ)^(2/3) * 2 = 50 := by
  sorry

end problem_statement_l1756_175653


namespace hyperbola_focus_implies_m_l1756_175606

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 / m - x^2 / 9 = 1

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (0, 5)

/-- Theorem: If F(0,5) is a focus of the hyperbola y^2/m - x^2/9 = 1, then m = 16 -/
theorem hyperbola_focus_implies_m (m : ℝ) :
  (∀ x y, hyperbola_equation x y m → (x - focus.1)^2 + (y - focus.2)^2 = (x + focus.1)^2 + (y - focus.2)^2) →
  m = 16 := by
  sorry

end hyperbola_focus_implies_m_l1756_175606


namespace max_value_theorem_l1756_175627

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ (16 * Real.sqrt 3 + 2 * Real.sqrt 33) / 3 :=
by sorry

end max_value_theorem_l1756_175627


namespace ceiling_sum_sqrt_l1756_175676

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end ceiling_sum_sqrt_l1756_175676


namespace largest_gold_coin_distribution_l1756_175666

theorem largest_gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 110 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 110 → m ≤ n) →
  n = 107 := by
  sorry

end largest_gold_coin_distribution_l1756_175666


namespace S_intersect_T_eq_T_l1756_175689

def S : Set ℝ := { y | ∃ x, y = 3^x }
def T : Set ℝ := { y | ∃ x, y = x^2 + 1 }

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l1756_175689


namespace equal_roots_quadratic_l1756_175659

theorem equal_roots_quadratic (h : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - 4 * y + h / 3 = 0 → y = x) ↔ h = 4 := by
sorry

end equal_roots_quadratic_l1756_175659


namespace intersection_of_M_and_N_l1756_175604

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end intersection_of_M_and_N_l1756_175604


namespace nabla_calculation_l1756_175673

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_calculation : nabla (nabla 2 3) 2 = 124 := by
  sorry

end nabla_calculation_l1756_175673


namespace markus_family_ages_l1756_175636

/-- Given a family where:
  * Markus is twice the age of his son
  * Markus's son is twice the age of Markus's grandson
  * Markus's grandson is three times the age of Markus's great-grandson
  * The sum of their ages is 140 years
Prove that Markus's great-grandson's age is 140/22 years. -/
theorem markus_family_ages (markus son grandson great_grandson : ℚ)
  (h1 : markus = 2 * son)
  (h2 : son = 2 * grandson)
  (h3 : grandson = 3 * great_grandson)
  (h4 : markus + son + grandson + great_grandson = 140) :
  great_grandson = 140 / 22 := by
  sorry

end markus_family_ages_l1756_175636


namespace common_root_of_quadratic_equations_l1756_175639

theorem common_root_of_quadratic_equations (p q : ℝ) (x : ℝ) :
  (2017 * x^2 + p * x + q = 0) ∧ 
  (p * x^2 + q * x + 2017 = 0) →
  x = 1 := by
sorry

end common_root_of_quadratic_equations_l1756_175639


namespace quadratic_function_range_l1756_175651

theorem quadratic_function_range (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  (5 ≤ f (-2) ∧ f (-2) ≤ 10) := by
  sorry

end quadratic_function_range_l1756_175651


namespace unique_train_journey_l1756_175632

/-- Represents a day of the week -/
inductive DayOfWeek
| Saturday
| Sunday
| Monday

/-- Represents the train journey details -/
structure TrainJourney where
  carNumber : Nat
  seatNumber : Nat
  saturdayDate : Nat
  mondayDate : Nat

/-- Checks if the journey satisfies all given conditions -/
def isValidJourney (journey : TrainJourney) : Prop :=
  journey.seatNumber < journey.carNumber ∧
  journey.saturdayDate > journey.carNumber ∧
  journey.mondayDate = journey.carNumber

theorem unique_train_journey : 
  ∀ (journey : TrainJourney), 
    isValidJourney journey → 
    journey.carNumber = 2 ∧ journey.seatNumber = 1 :=
by sorry

end unique_train_journey_l1756_175632


namespace relationship_functions_l1756_175631

-- Define the relationships
def relationA (x : ℝ) : ℝ := 180 - x
def relationB (x : ℝ) : ℝ := 60 + 3 * x
def relationC (x : ℝ) : ℝ := x ^ 2
def relationD (x : ℝ) : Set ℝ := {y | y ^ 2 = x ∧ x ≥ 0}

-- Theorem stating that A, B, and C are functions, while D is not
theorem relationship_functions :
  (∀ x : ℝ, ∃! y : ℝ, y = relationA x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationB x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationC x) ∧
  ¬(∀ x : ℝ, ∃! y : ℝ, y ∈ relationD x) :=
by sorry

end relationship_functions_l1756_175631


namespace range_of_a_l1756_175625

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 2*a*x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_of_a_l1756_175625


namespace triangle_abc_properties_l1756_175610

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  C = 2 * Real.pi / 3 →
  c = 5 →
  a = Real.sqrt 5 * b * Real.sin A →
  b = 2 * Real.sqrt 15 / 3 ∧
  Real.tan (B + Real.pi / 4) = 3 :=
by sorry

end triangle_abc_properties_l1756_175610


namespace max_grain_mass_on_platform_l1756_175617

/-- Represents a rectangular platform --/
structure Platform where
  length : ℝ
  width : ℝ

/-- Represents the properties of grain --/
structure Grain where
  density : ℝ
  max_angle : ℝ

/-- Calculates the maximum mass of grain that can be loaded onto a platform --/
def max_grain_mass (p : Platform) (g : Grain) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform --/
theorem max_grain_mass_on_platform :
  let p : Platform := { length := 10, width := 5 }
  let g : Grain := { density := 1200, max_angle := 45 }
  max_grain_mass p g = 175000 := by
  sorry

end max_grain_mass_on_platform_l1756_175617


namespace total_calories_burned_first_week_l1756_175601

def calories_per_hour_walking : ℕ := 300

def calories_per_hour_dancing : ℕ := 2 * calories_per_hour_walking

def calories_per_hour_swimming : ℕ := (3 * calories_per_hour_walking) / 2

def calories_per_hour_cycling : ℕ := calories_per_hour_walking

def dancing_hours_per_week : ℕ := 3 * (2 * 1/2) + 1

def swimming_hours_per_week : ℕ := 2 * 3/2

def cycling_hours_per_week : ℕ := 2

def total_calories_burned : ℕ := 
  calories_per_hour_dancing * dancing_hours_per_week +
  calories_per_hour_swimming * swimming_hours_per_week +
  calories_per_hour_cycling * cycling_hours_per_week

theorem total_calories_burned_first_week : 
  total_calories_burned = 4350 := by sorry

end total_calories_burned_first_week_l1756_175601


namespace crate_stacking_probability_l1756_175696

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the stack configuration -/
structure StackConfiguration where
  num_3ft : ℕ
  num_4ft : ℕ
  num_5ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 5 }

def num_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.num_3ft + config.num_4ft + config.num_5ft = num_crates ∧
  3 * config.num_3ft + 4 * config.num_4ft + 5 * config.num_5ft = target_height

def num_valid_configurations : ℕ := 33616

def total_possible_configurations : ℕ := 3^num_crates

theorem crate_stacking_probability :
  (num_valid_configurations : ℚ) / (total_possible_configurations : ℚ) = 80 / 1593 := by
  sorry

#check crate_stacking_probability

end crate_stacking_probability_l1756_175696


namespace triangle_problem_l1756_175693

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < C ∧ C < π / 2 →
  a * Real.sin A = b * Real.sin B * Real.sin C →
  b = Real.sqrt 2 * a →
  C = π / 6 ∧ c^2 / a^2 = 3 - Real.sqrt 6 := by
sorry

end triangle_problem_l1756_175693


namespace ln_x_over_x_decreasing_l1756_175688

theorem ln_x_over_x_decreasing (a b c : ℝ) : 
  a = (Real.log 3) / 3 → 
  b = (Real.log 5) / 5 → 
  c = (Real.log 6) / 6 → 
  a > b ∧ b > c := by
  sorry

end ln_x_over_x_decreasing_l1756_175688


namespace calculation_problem_l1756_175698

theorem calculation_problem (x : ℝ) : 10 * 1.8 - (2 * x / 0.3) = 50 ↔ x = -4.8 := by
  sorry

end calculation_problem_l1756_175698


namespace correct_position_probability_l1756_175620

/-- The number of books -/
def n : ℕ := 9

/-- The number of books to be in the correct position -/
def k : ℕ := 6

/-- The probability of exactly k books being in their correct position when n books are randomly rearranged -/
def probability (n k : ℕ) : ℚ := sorry

theorem correct_position_probability : probability n k = 1 / 2160 := by sorry

end correct_position_probability_l1756_175620


namespace spurs_basketball_count_l1756_175616

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let basketballs_per_player : ℕ := 11
  num_players * basketballs_per_player = 242 :=
by sorry

end spurs_basketball_count_l1756_175616


namespace tan_thirteen_pi_fourth_l1756_175605

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = 1 := by
  sorry

end tan_thirteen_pi_fourth_l1756_175605


namespace x_power_y_equals_243_l1756_175656

theorem x_power_y_equals_243 (x y : ℝ) : 
  y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 5 → x^y = 243 := by sorry

end x_power_y_equals_243_l1756_175656


namespace ace_then_diamond_probability_l1756_175675

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumberOfAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumberOfDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a diamond as the second card -/
def ProbabilityAceThenDiamond : ℚ := 1 / StandardDeck

theorem ace_then_diamond_probability :
  ProbabilityAceThenDiamond = 1 / StandardDeck := by
  sorry

end ace_then_diamond_probability_l1756_175675


namespace marco_strawberry_weight_l1756_175665

theorem marco_strawberry_weight (total_weight : ℕ) (weight_difference : ℕ) :
  total_weight = 47 →
  weight_difference = 13 →
  ∃ (marco_weight dad_weight : ℕ),
    marco_weight + dad_weight = total_weight ∧
    marco_weight = dad_weight + weight_difference ∧
    marco_weight = 30 :=
by sorry

end marco_strawberry_weight_l1756_175665


namespace inequality_theorem_l1756_175623

theorem inequality_theorem (a b c : ℝ) : a > -b → c - a < c + b := by
  sorry

end inequality_theorem_l1756_175623


namespace perpendicular_line_through_point_l1756_175694

-- Define the type for a point in 2D space
def Point := ℝ × ℝ

-- Define the type for a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define a function to check if two lines are perpendicular
def areLinesPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point 
  (l : Line) 
  (h1 : isPointOnLine (-1, 2) l) 
  (h2 : areLinesPerpendicular l ⟨1, -3, 5⟩) : 
  l = ⟨3, 1, 1⟩ :=
sorry

end perpendicular_line_through_point_l1756_175694


namespace not_isosceles_if_distinct_sides_l1756_175637

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem not_isosceles_if_distinct_sides (t : Triangle) 
  (distinct_sides : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) : 
  ¬(is_isosceles t) := by
  sorry

end not_isosceles_if_distinct_sides_l1756_175637


namespace escalator_length_l1756_175634

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ) (length : ℝ),
  escalator_speed = 12 →
  person_speed = 2 →
  time = 14 →
  length = (escalator_speed + person_speed) * time →
  length = 196 := by
sorry

end escalator_length_l1756_175634


namespace slope_intercept_sum_l1756_175615

/-- Given points A(0,6), B(0,0), C(8,0), and D the midpoint of AB, 
    prove that the sum of the slope and y-intercept of the line passing through C and D is 21/8 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → B = (0, 0) → C = (8, 0) → D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 21 / 8 := by
sorry


end slope_intercept_sum_l1756_175615


namespace equilateral_triangles_with_squares_l1756_175626

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  sorry

/-- Constructs a square externally on a side of a triangle -/
def construct_external_square (t : Triangle) (side : Fin 3) : Square :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem equilateral_triangles_with_squares
  (ABC BCD : Triangle)
  (ABEF : Square)
  (CDGH : Square)
  (h1 : is_equilateral ABC)
  (h2 : is_equilateral BCD)
  (h3 : ABEF = construct_external_square ABC 0)
  (h4 : CDGH = construct_external_square BCD 1)
  : distance ABEF.C CDGH.C / distance ABC.B ABC.C = 3 :=
by sorry

end equilateral_triangles_with_squares_l1756_175626


namespace maria_car_trip_l1756_175650

theorem maria_car_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (second_stop_fraction : ℝ) :
  total_distance = 560 ∧ 
  first_stop_fraction = 1/2 ∧ 
  second_stop_fraction = 1/4 →
  total_distance - (first_stop_fraction * total_distance) - 
    (second_stop_fraction * (total_distance - first_stop_fraction * total_distance)) = 210 := by
  sorry

end maria_car_trip_l1756_175650
