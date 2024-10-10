import Mathlib

namespace blue_socks_count_l2035_203510

/-- The number of red socks -/
def red_socks : ℕ := 2

/-- The number of black socks -/
def black_socks : ℕ := 2

/-- The number of white socks -/
def white_socks : ℕ := 2

/-- The probability of drawing two socks of the same color -/
def same_color_prob : ℚ := 1/5

theorem blue_socks_count (x : ℕ) (hx : x > 0) :
  let total := red_socks + black_socks + white_socks + x
  (3 * 2 + x * (x - 1)) / (total * (total - 1)) = same_color_prob →
  x = 4 := by sorry

end blue_socks_count_l2035_203510


namespace largest_odd_integer_sum_30_l2035_203540

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def consecutive_odd_integers (m : ℕ) : List ℕ := [m - 4, m - 2, m, m + 2, m + 4]

theorem largest_odd_integer_sum_30 :
  ∃ m : ℕ, 
    (sum_first_n_odd 30 = (consecutive_odd_integers m).sum) ∧
    (List.maximum (consecutive_odd_integers m) = some 184) := by
  sorry

end largest_odd_integer_sum_30_l2035_203540


namespace restaurant_bill_calculation_l2035_203572

/-- Calculate the total cost for a group to eat at a restaurant -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℕ) 
  (total_people : ℕ) 
  (kids_count : ℕ) 
  (h1 : adult_meal_cost = 3)
  (h2 : total_people = 12)
  (h3 : kids_count = 7) :
  (total_people - kids_count) * adult_meal_cost = 15 := by
  sorry

#check restaurant_bill_calculation

end restaurant_bill_calculation_l2035_203572


namespace average_of_four_numbers_l2035_203539

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
sorry

end average_of_four_numbers_l2035_203539


namespace olympic_items_problem_l2035_203580

/-- Olympic Commemorative Items Problem -/
theorem olympic_items_problem 
  (total_items : ℕ) 
  (figurine_cost pendant_cost : ℚ) 
  (total_spent : ℚ) 
  (figurine_price pendant_price : ℚ) 
  (min_profit : ℚ) 
  (h1 : total_items = 180)
  (h2 : figurine_cost = 80)
  (h3 : pendant_cost = 50)
  (h4 : total_spent = 11400)
  (h5 : figurine_price = 100)
  (h6 : pendant_price = 60)
  (h7 : min_profit = 2900) :
  ∃ (figurines pendants max_pendants : ℕ),
    figurines + pendants = total_items ∧
    figurine_cost * figurines + pendant_cost * pendants = total_spent ∧
    figurines = 80 ∧
    pendants = 100 ∧
    max_pendants = 70 ∧
    ∀ m : ℕ, m ≤ max_pendants →
      (pendant_price - pendant_cost) * m + 
      (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit :=
by
  sorry


end olympic_items_problem_l2035_203580


namespace fraction_equality_l2035_203547

theorem fraction_equality (a b c : ℝ) (h1 : a/3 = b) (h2 : b/4 = c) : a*b/c^2 = 48 := by
  sorry

end fraction_equality_l2035_203547


namespace z_in_first_quadrant_l2035_203582

theorem z_in_first_quadrant (z : ℂ) : 
  ((1 + 2*Complex.I) / (z - 3) = -Complex.I) → 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end z_in_first_quadrant_l2035_203582


namespace board_numbers_product_l2035_203505

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end board_numbers_product_l2035_203505


namespace spider_journey_l2035_203501

theorem spider_journey (r : ℝ) (leg : ℝ) (h1 : r = 65) (h2 : leg = 90) :
  let diameter := 2 * r
  let hypotenuse := diameter
  let other_leg := Real.sqrt (hypotenuse ^ 2 - leg ^ 2)
  hypotenuse + leg + other_leg = 220 + 20 * Real.sqrt 22 := by
  sorry

end spider_journey_l2035_203501


namespace cube_difference_factorization_l2035_203514

theorem cube_difference_factorization (t : ℝ) : t^3 - 8 = (t - 2) * (t^2 + 2*t + 4) := by
  sorry

end cube_difference_factorization_l2035_203514


namespace arithmetic_sequence_a2_l2035_203581

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) :
  arithmetic_sequence a (-2) →
  (a 1 + a 5) / 2 = -1 →
  a 2 = 1 := by
sorry

end arithmetic_sequence_a2_l2035_203581


namespace triangle_rotation_path_length_l2035_203548

/-- Represents a triangle -/
structure Triangle where
  side_length : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- Calculates the path length of a vertex of a triangle rotating inside a square -/
def path_length (t : Triangle) (s : Square) : ℝ :=
  sorry

/-- Theorem stating the path length for the given triangle and square -/
theorem triangle_rotation_path_length :
  let t : Triangle := { side_length := 3 }
  let s : Square := { side_length := 6 }
  path_length t s = 24 * Real.pi :=
sorry

end triangle_rotation_path_length_l2035_203548


namespace quadratic_root_problem_l2035_203566

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y + 6 = 0 ∧ y = -3) :=
by sorry

end quadratic_root_problem_l2035_203566


namespace smallest_other_integer_l2035_203579

theorem smallest_other_integer (m n x : ℕ+) : 
  (m = 50 ∨ n = 50) →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = x.val * (x.val + 5) →
  (m ≠ 50 → m ≥ 10) ∧ (n ≠ 50 → n ≥ 10) :=
by sorry

end smallest_other_integer_l2035_203579


namespace xy_value_l2035_203598

theorem xy_value (x y : ℝ) (h1 : |x| = 2) (h2 : y = 3) (h3 : x * y < 0) : x * y = -6 := by
  sorry

end xy_value_l2035_203598


namespace greatest_power_under_500_l2035_203529

/-- For positive integers a and b, where b > 1, if a^b is the greatest possible value less than 500, then a + b = 24 -/
theorem greatest_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) 
  (h_greatest : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → a^b ≥ x^y) 
  (h_less_500 : a^b < 500) : a + b = 24 := by
  sorry


end greatest_power_under_500_l2035_203529


namespace periodic_function_value_l2035_203522

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2014 = 5 → f 2015 = 3 := by
  sorry

end periodic_function_value_l2035_203522


namespace rectangular_plot_dimensions_l2035_203585

theorem rectangular_plot_dimensions :
  ∀ (width length area : ℕ),
    length = width + 1 →
    area = width * length →
    1000 ≤ area ∧ area < 10000 →
    (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ area = 1000 * a + 100 * a + 10 * b + b) →
    width ∈ ({33, 66, 99} : Set ℕ) :=
by sorry

end rectangular_plot_dimensions_l2035_203585


namespace dollar_three_minus_four_l2035_203568

-- Define the custom operation $
def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y + x

-- Theorem statement
theorem dollar_three_minus_four : dollar 3 (-4) = -15 := by
  sorry

end dollar_three_minus_four_l2035_203568


namespace cars_meeting_time_l2035_203595

/-- Given a scenario with two cars and a train, calculate the time for the cars to meet -/
theorem cars_meeting_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_between_encounters : ℝ)
  (time_pass_A : ℝ) (time_pass_B : ℝ)
  (h1 : train_length = 180)
  (h2 : train_speed_kmh = 60)
  (h3 : time_between_encounters = 5)
  (h4 : time_pass_A = 30 / 60)
  (h5 : time_pass_B = 6 / 60) :
  let train_speed := train_speed_kmh * 1000 / 3600
  let car_A_speed := train_speed - train_length / time_pass_A
  let car_B_speed := train_length / time_pass_B - train_speed
  let distance := time_between_encounters * (train_speed - car_A_speed)
  distance / (car_A_speed + car_B_speed) = 1.25 := by
  sorry

end cars_meeting_time_l2035_203595


namespace gel_pen_price_ratio_l2035_203553

theorem gel_pen_price_ratio (x y : ℕ) (b g : ℝ) :
  x > 0 ∧ y > 0 ∧ b > 0 ∧ g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b :=
by
  sorry

end gel_pen_price_ratio_l2035_203553


namespace g_equals_zero_at_negative_one_l2035_203576

/-- Given a function g(x) = 3x^4 + 2x^3 - x^2 - 4x + s, prove that g(-1) = 0 when s = -4 -/
theorem g_equals_zero_at_negative_one (s : ℝ) : 
  let g : ℝ → ℝ := λ x => 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s
  g (-1) = 0 ↔ s = -4 := by
  sorry

end g_equals_zero_at_negative_one_l2035_203576


namespace find_b_l2035_203513

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 3 * x - 7)
  (hq : ∀ x, q x = 3 * x - b)
  (h : p (q 5) = 11) :
  b = 9 := by sorry

end find_b_l2035_203513


namespace unique_prime_pair_sum_59_l2035_203588

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_pair_sum_59 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 59 :=
by sorry

end unique_prime_pair_sum_59_l2035_203588


namespace multiply_and_simplify_l2035_203538

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = 25 / 8 * x^2 := by
  sorry

end multiply_and_simplify_l2035_203538


namespace complex_equation_solution_l2035_203599

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), determinant 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I := by
  sorry

end complex_equation_solution_l2035_203599


namespace at_most_one_perfect_square_l2035_203535

-- Define the sequence (a_n)
def a : ℕ → ℤ
  | 0 => sorry  -- We don't know the initial value, so we use sorry
  | n + 1 => (a n)^3 + 1999

-- Define what it means for an integer to be a perfect square
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

-- State the theorem
theorem at_most_one_perfect_square :
  ∃! n : ℕ, is_perfect_square (a n) :=
sorry

end at_most_one_perfect_square_l2035_203535


namespace abs_square_of_complex_l2035_203578

theorem abs_square_of_complex (z : ℂ) : z = 5 + 2*I → Complex.abs (z^2) = 29 := by
  sorry

end abs_square_of_complex_l2035_203578


namespace motorcycles_in_parking_lot_l2035_203593

theorem motorcycles_in_parking_lot :
  let total_wheels : ℕ := 117
  let num_cars : ℕ := 19
  let wheels_per_car : ℕ := 5
  let wheels_per_motorcycle : ℕ := 2
  let num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle
  num_motorcycles = 11 := by
  sorry

end motorcycles_in_parking_lot_l2035_203593


namespace region_contains_point_c_l2035_203592

def point_in_region (x y : ℝ) : Prop :=
  x + 2*y - 1 > 0 ∧ x - y + 3 < 0

theorem region_contains_point_c :
  point_in_region 0 4 ∧
  ¬point_in_region (-4) 1 ∧
  ¬point_in_region 2 2 ∧
  ¬point_in_region (-2) 1 := by
  sorry

#check region_contains_point_c

end region_contains_point_c_l2035_203592


namespace events_mutually_exclusive_l2035_203558

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- The event of drawing a specific combination of balls -/
structure Event where
  red : ℕ
  white : ℕ

/-- The bag in the problem -/
def problem_bag : Bag := { red := 5, white := 5 }

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The event of drawing 3 red balls -/
def event_all_red : Event := { red := 3, white := 0 }

/-- The event of drawing at least 1 white ball -/
def event_at_least_one_white : Event := { red := drawn_balls - 1, white := 1 }

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Event) : Prop :=
  e1.red + e1.white = drawn_balls ∧ e2.red + e2.white = drawn_balls ∧ 
  (e1.red + e2.red > problem_bag.red ∨ e1.white + e2.white > problem_bag.white)

/-- The main theorem stating that the two events are mutually exclusive -/
theorem events_mutually_exclusive : 
  mutually_exclusive event_all_red event_at_least_one_white :=
sorry

end events_mutually_exclusive_l2035_203558


namespace research_team_composition_l2035_203549

/-- Represents the composition of a research team -/
structure ResearchTeam where
  total : Nat
  male : Nat
  female : Nat

/-- Represents the company's employee composition -/
def company : ResearchTeam :=
  { total := 60,
    male := 45,
    female := 15 }

/-- The size of the research team -/
def team_size : Nat := 4

/-- The probability of an employee being selected for the research team -/
def selection_probability : Rat := team_size / company.total

/-- The composition of the research team -/
def research_team : ResearchTeam :=
  { total := team_size,
    male := 3,
    female := 1 }

/-- The probability of selecting exactly one female when choosing two employees from the research team -/
def prob_one_female : Rat := 1 / 2

theorem research_team_composition :
  selection_probability = 1 / 15 ∧
  research_team.male = 3 ∧
  research_team.female = 1 ∧
  prob_one_female = 1 / 2 := by sorry

end research_team_composition_l2035_203549


namespace quadratic_function_unique_l2035_203526

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 3) 
  (h3 : ∀ x, f (x + 2) - f x = 4 * x + 2) : 
  ∀ x, f x = x^2 - x + 3 := by
sorry


end quadratic_function_unique_l2035_203526


namespace cube_split_theorem_l2035_203512

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) : 
  m^2 - m + 1 = 73 → m = 9 :=
by
  sorry

#check cube_split_theorem

end cube_split_theorem_l2035_203512


namespace other_color_students_l2035_203555

theorem other_color_students (total : ℕ) (blue_percent red_percent green_percent : ℚ) : 
  total = 800 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  (total : ℚ) * (1 - (blue_percent + red_percent + green_percent)) = 136 := by
  sorry

end other_color_students_l2035_203555


namespace max_d_value_l2035_203507

def a (n : ℕ+) : ℕ := n.val ^ 2 + 1000

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 4001 ∧ ∀ (n : ℕ+), d n ≤ 4001 :=
sorry

end max_d_value_l2035_203507


namespace book_cart_total_l2035_203554

/-- Represents the number of books in each category on the top section of the cart -/
structure TopSection where
  history : ℕ
  romance : ℕ
  poetry : ℕ

/-- Represents the number of books in each category on the bottom section of the cart -/
structure BottomSection where
  western : ℕ
  biography : ℕ
  scifi : ℕ

/-- Represents the entire book cart -/
structure BookCart where
  top : TopSection
  bottom : BottomSection
  mystery : ℕ

def total_books (cart : BookCart) : ℕ :=
  cart.top.history + cart.top.romance + cart.top.poetry +
  cart.bottom.western + cart.bottom.biography + cart.bottom.scifi +
  cart.mystery

theorem book_cart_total (cart : BookCart) :
  cart.top.history = 12 →
  cart.top.romance = 8 →
  cart.top.poetry = 4 →
  cart.bottom.western = 5 →
  cart.bottom.biography = 6 →
  cart.bottom.scifi = 3 →
  cart.mystery = 2 * (cart.bottom.western + cart.bottom.biography + cart.bottom.scifi) →
  total_books cart = 66 := by
  sorry

#check book_cart_total

end book_cart_total_l2035_203554


namespace sum_of_real_and_imaginary_parts_l2035_203517

theorem sum_of_real_and_imaginary_parts : ∃ (z : ℂ), z = 3 - 4*I ∧ z.re + z.im = -1 :=
sorry

end sum_of_real_and_imaginary_parts_l2035_203517


namespace committee_selection_l2035_203583

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end committee_selection_l2035_203583


namespace f_neg_two_equals_ten_l2035_203573

/-- Given a function f(x) = x^2 - 3x, prove that f(-2) = 10 -/
theorem f_neg_two_equals_ten (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 3*x) : f (-2) = 10 := by
  sorry

end f_neg_two_equals_ten_l2035_203573


namespace isosceles_triangle_angle_b_l2035_203503

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle_a : ℝ
  angle_b : ℝ
  is_isosceles : (angle_a = angle_b) ∨ (angle_a + 2 * angle_b = 180) ∨ (2 * angle_a + angle_b = 180)

-- Define the theorem
theorem isosceles_triangle_angle_b (t : IsoscelesTriangle) 
  (h : t.angle_a = 70) : 
  t.angle_b = 55 ∨ t.angle_b = 70 ∨ t.angle_b = 40 := by
  sorry


end isosceles_triangle_angle_b_l2035_203503


namespace bus_speed_is_45_l2035_203586

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

def initial_number : TwoDigitNumber := sorry

def one_hour_later : ThreeDigitNumber := sorry

def two_hours_later : ThreeDigitNumber := sorry

/-- The speed of the bus in km/h -/
def bus_speed : Nat := sorry

theorem bus_speed_is_45 :
  (one_hour_later.hundreds = initial_number.ones) ∧
  (one_hour_later.tens = 0) ∧
  (one_hour_later.ones = initial_number.tens) ∧
  (two_hours_later.hundreds = one_hour_later.hundreds) ∧
  (two_hours_later.ones = one_hour_later.ones) ∧
  (two_hours_later.tens ≠ 0) →
  bus_speed = 45 := by sorry

end bus_speed_is_45_l2035_203586


namespace complement_intersection_theorem_l2035_203511

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {4, 7, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 2, 6} := by sorry

end complement_intersection_theorem_l2035_203511


namespace volume_expanded_parallelepiped_eq_l2035_203575

/-- The volume of a set of points inside or within one unit of a 2x3x4 rectangular parallelepiped -/
def volume_expanded_parallelepiped : ℝ := sorry

/-- The dimension of the parallelepiped along the x-axis -/
def x_dim : ℝ := 2

/-- The dimension of the parallelepiped along the y-axis -/
def y_dim : ℝ := 3

/-- The dimension of the parallelepiped along the z-axis -/
def z_dim : ℝ := 4

/-- The radius of the expanded region around the parallelepiped -/
def expansion_radius : ℝ := 1

theorem volume_expanded_parallelepiped_eq :
  volume_expanded_parallelepiped = (228 + 31 * Real.pi) / 3 := by sorry

end volume_expanded_parallelepiped_eq_l2035_203575


namespace locus_of_symmetric_point_l2035_203551

/-- Given a parabola y = x^2 and a fixed point A(a, 0) where a ≠ 0, 
    the locus of point Q symmetric to A with respect to a point on the parabola 
    is described by the equation y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ (x y : ℝ), (y = x^2) → 
      ∃ (qx qy : ℝ), 
        (qx + x = 2 * a ∧ qy + y = 0) → 
        f qx = qy ∧ f qx = (1/2) * (qx + a)^2) := by
  sorry

end locus_of_symmetric_point_l2035_203551


namespace business_investment_l2035_203559

theorem business_investment (A B : ℕ) (t : ℕ) (r : ℚ) : 
  A = 45000 →
  t = 2 →
  r = 2 / 1 →
  (A * t : ℚ) / (B * t : ℚ) = r →
  B = 22500 := by
  sorry

end business_investment_l2035_203559


namespace q_value_at_two_l2035_203520

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define q(x) as p(p(x))
def q (x : ℝ) : ℝ := p (p x)

-- Theorem statement
theorem q_value_at_two (h : ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ q a = 0 ∧ q b = 0 ∧ q c = 0) :
  q 2 = -1 := by
  sorry


end q_value_at_two_l2035_203520


namespace oranges_thrown_away_l2035_203544

/-- Proves that 2 old oranges were thrown away given the initial, added, and final orange counts. -/
theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) 
    (h1 : initial = 5)
    (h2 : added = 28)
    (h3 : final = 31) :
  initial - (initial + added - final) = 2 := by
  sorry

end oranges_thrown_away_l2035_203544


namespace degree_three_polynomial_l2035_203562

/-- Polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- Polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x + x^2 - 6*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that -5/11 is the value of c that makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -5/11 ∧ 
  (∀ (x : ℝ), h c x = (1 + 3*c) + (-12 - 2*c)*x + (3 + c)*x^2 + (-4 - 6*c)*x^3) :=
sorry

end degree_three_polynomial_l2035_203562


namespace solve_for_q_l2035_203597

theorem solve_for_q (k l q : ℚ) : 
  (7 / 8 = k / 96) → 
  (7 / 8 = (k + l) / 112) → 
  (7 / 8 = (q - l) / 144) → 
  q = 140 := by
sorry

end solve_for_q_l2035_203597


namespace unique_determination_by_digit_sums_l2035_203542

/-- Given a natural number, compute the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number N, generate a sequence of digit sums for consecutive numbers starting from N+1 -/
def digit_sum_sequence (N : ℕ) (length : ℕ) : List ℕ := sorry

/-- Theorem: For any natural number N, there exists a finite sequence of digit sums that uniquely determines N -/
theorem unique_determination_by_digit_sums (N : ℕ) : 
  ∃ (length : ℕ), ∀ (M : ℕ), M ≠ N → 
    digit_sum_sequence N length ≠ digit_sum_sequence M length := by
  sorry

#check unique_determination_by_digit_sums

end unique_determination_by_digit_sums_l2035_203542


namespace inequalities_theorem_l2035_203590

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : a < 0) (h2 : b > 0) (h3 : a < b) (h4 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a < c / b) := by
  sorry

end inequalities_theorem_l2035_203590


namespace thirteen_students_in_line_l2035_203530

/-- The number of students in a line, given specific positions of Taehyung and Namjoon -/
def students_in_line (people_between_taehyung_and_namjoon : ℕ) (people_behind_namjoon : ℕ) : ℕ :=
  1 + people_between_taehyung_and_namjoon + 1 + people_behind_namjoon

/-- Theorem stating that there are 13 students in the line -/
theorem thirteen_students_in_line : 
  students_in_line 3 8 = 13 := by
  sorry

end thirteen_students_in_line_l2035_203530


namespace abs_negative_2023_l2035_203502

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l2035_203502


namespace chips_after_steps_chips_after_25_steps_l2035_203594

/-- Represents the state of trays with chips -/
def TrayState := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : TrayState :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Counts the number of true values in a list of booleans -/
def countTrueValues (l : List Bool) : Nat :=
  l.filter id |>.length

/-- The number of chips after n steps is equal to the number of 1s in the binary representation of n -/
theorem chips_after_steps (n : Nat) : 
  countTrueValues (toBinary n) = countTrueValues (toBinary n) := by sorry

/-- The number of chips after 25 steps is equal to the number of 1s in the binary representation of 25 -/
theorem chips_after_25_steps : 
  countTrueValues (toBinary 25) = 3 := by sorry

end chips_after_steps_chips_after_25_steps_l2035_203594


namespace sequence_properties_l2035_203567

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, S n = -n^2 + 7*n) →
  (∀ n : ℕ, n ≥ 2 → S n - S (n-1) = a n) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, n ≠ 3 ∧ n ≠ 4 → S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end sequence_properties_l2035_203567


namespace periodic_placement_exists_l2035_203519

/-- A function that maps integer coordinates to natural numbers -/
def f : ℤ × ℤ → ℕ := sorry

/-- Theorem stating the existence of a function satisfying the required properties -/
theorem periodic_placement_exists : 
  (∀ n : ℕ, ∃ x y : ℤ, f (x, y) = n) ∧ 
  (∀ a b c : ℤ, a ≠ 0 ∨ b ≠ 0 → c ≠ 0 → 
    ∃ k m : ℤ, ∀ x y : ℤ, a * x + b * y = c → 
      f (x + k, y + m) = f (x, y)) :=
by sorry

end periodic_placement_exists_l2035_203519


namespace bag_problem_l2035_203516

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the probability of drawing at least 1 white ball when drawing 2 balls -/
def prob_at_least_one_white : ℚ := 4/5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := sorry

/-- Calculates the probability of drawing exactly k white balls when drawing 2 balls -/
def prob_k_white (k : ℕ) : ℚ := sorry

/-- Calculates the mathematical expectation of the number of white balls drawn -/
def expectation : ℚ := sorry

theorem bag_problem :
  (1 - (choose (total_balls - num_white_balls) 2 : ℚ) / (choose total_balls 2 : ℚ) = prob_at_least_one_white) →
  (num_white_balls = 3 ∧ expectation = 1) :=
by sorry

end bag_problem_l2035_203516


namespace intersection_line_l2035_203504

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y - 4 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_l2035_203504


namespace divisibility_implies_gcd_greater_than_one_l2035_203534

theorem divisibility_implies_gcd_greater_than_one
  (a b x y : ℕ)
  (h : (a^2 + b^2) ∣ (a*x + b*y)) :
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 :=
sorry

end divisibility_implies_gcd_greater_than_one_l2035_203534


namespace speed_calculation_l2035_203570

-- Define the given conditions
def field_area : ℝ := 50
def travel_time_minutes : ℝ := 2

-- Define the theorem
theorem speed_calculation :
  let diagonal := Real.sqrt (2 * field_area)
  let speed_m_per_hour := diagonal / (travel_time_minutes / 60)
  speed_m_per_hour / 1000 = 0.3 := by
sorry

end speed_calculation_l2035_203570


namespace combined_tax_rate_l2035_203591

/-- Calculate the combined tax rate for two individuals given their incomes and tax rates -/
theorem combined_tax_rate
  (john_income : ℝ)
  (john_tax_rate : ℝ)
  (ingrid_income : ℝ)
  (ingrid_tax_rate : ℝ)
  (h1 : john_income = 56000)
  (h2 : john_tax_rate = 0.30)
  (h3 : ingrid_income = 72000)
  (h4 : ingrid_tax_rate = 0.40) :
  (john_income * john_tax_rate + ingrid_income * ingrid_tax_rate) / (john_income + ingrid_income) = 0.35625 := by
  sorry

end combined_tax_rate_l2035_203591


namespace other_number_difference_l2035_203524

theorem other_number_difference (x : ℕ) (h1 : x + 42 = 96) : x = 54 := by
  sorry

#check other_number_difference

end other_number_difference_l2035_203524


namespace square_root_problem_l2035_203584

theorem square_root_problem (a : ℝ) (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt x = 2*a - 1) (h3 : Real.sqrt x = -a + 3) : x = 25 := by
  sorry

end square_root_problem_l2035_203584


namespace correct_ticket_sales_l2035_203546

/-- A structure representing ticket sales for different movie genres --/
structure TicketSales where
  romance : ℕ
  horror : ℕ
  action : ℕ
  comedy : ℕ

/-- Definition of valid ticket sales based on the given conditions --/
def is_valid_ticket_sales (t : TicketSales) : Prop :=
  t.romance = 25 ∧
  t.horror = 3 * t.romance + 18 ∧
  t.action = 2 * t.romance ∧
  5 * t.comedy = 4 * t.horror

/-- Theorem stating the correct number of tickets sold for each genre --/
theorem correct_ticket_sales :
  ∃ (t : TicketSales), is_valid_ticket_sales t ∧
    t.horror = 93 ∧ t.action = 50 ∧ t.comedy = 74 := by
  sorry

end correct_ticket_sales_l2035_203546


namespace digit_700_of_3_11_is_7_l2035_203556

/-- The 700th digit past the decimal point in the decimal expansion of 3/11 -/
def digit_700_of_3_11 : ℕ :=
  -- Define the digit here
  7

/-- Theorem stating that the 700th digit past the decimal point
    in the decimal expansion of 3/11 is 7 -/
theorem digit_700_of_3_11_is_7 :
  digit_700_of_3_11 = 7 := by
  sorry

end digit_700_of_3_11_is_7_l2035_203556


namespace ice_cream_box_cost_l2035_203508

/-- Represents the cost of a box of ice cream bars -/
def box_cost : ℚ := sorry

/-- Number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- Number of friends -/
def num_friends : ℕ := 6

/-- Number of bars each friend wants to eat -/
def bars_per_friend : ℕ := 2

/-- Cost per person -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_cost :
  box_cost = 7.5 := by sorry

end ice_cream_box_cost_l2035_203508


namespace value_calculation_l2035_203557

/-- If 0.5% of a value equals 65 paise, then the value is 130 rupees -/
theorem value_calculation (a : ℝ) : (0.005 * a = 65 / 100) → a = 130 := by
  sorry

end value_calculation_l2035_203557


namespace dvd_sales_l2035_203569

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * (cd : ℝ) →
  dvd + cd = 273 →
  dvd = 168 := by
  sorry

end dvd_sales_l2035_203569


namespace adam_father_deposit_l2035_203564

/-- Calculates the total amount after a given period, including initial deposit and interest --/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given the specified conditions, the total amount after 2.5 years is $2400 --/
theorem adam_father_deposit :
  let initialDeposit : ℝ := 2000
  let interestRate : ℝ := 0.08
  let years : ℝ := 2.5
  totalAmount initialDeposit interestRate years = 2400 := by
  sorry

end adam_father_deposit_l2035_203564


namespace triathlete_average_rate_l2035_203500

/-- The average rate of a triathlete's round trip -/
theorem triathlete_average_rate 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (swimming_distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : total_distance = 6) 
  (h2 : running_distance = total_distance / 2) 
  (h3 : swimming_distance = total_distance / 2) 
  (h4 : running_speed = 10) 
  (h5 : swimming_speed = 6) : 
  (total_distance / ((running_distance / running_speed + swimming_distance / swimming_speed) * 60)) = 0.125 := by
  sorry

end triathlete_average_rate_l2035_203500


namespace tv_conditional_probability_l2035_203571

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end tv_conditional_probability_l2035_203571


namespace part_one_part_two_l2035_203532

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one (x : ℝ) : 
  p x 1 ∨ q x → 1 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (a > 0 ∧ (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)) → 
  1 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l2035_203532


namespace initial_mixture_volume_l2035_203521

/-- Given a mixture of milk and water, prove that the initial volume is 60 litres -/
theorem initial_mixture_volume
  (initial_ratio : ℚ) -- Initial ratio of milk to water
  (final_ratio : ℚ) -- Final ratio of milk to water
  (added_water : ℚ) -- Amount of water added to achieve final ratio
  (h1 : initial_ratio = 2 / 1) -- Initial ratio is 2:1
  (h2 : final_ratio = 1 / 2) -- Final ratio is 1:2
  (h3 : added_water = 60) -- 60 litres of water is added
  : ℚ :=
by
  sorry

#check initial_mixture_volume

end initial_mixture_volume_l2035_203521


namespace probability_five_heads_in_six_tosses_l2035_203536

def n : ℕ := 6  -- number of coin tosses
def k : ℕ := 5  -- number of heads we want to get
def p : ℚ := 1/2  -- probability of getting heads on a single toss (fair coin)

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of getting exactly k successes in n trials
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

-- The theorem to prove
theorem probability_five_heads_in_six_tosses :
  probability_k_successes n k p = 0.09375 := by
  sorry

end probability_five_heads_in_six_tosses_l2035_203536


namespace least_multiple_32_over_500_l2035_203589

theorem least_multiple_32_over_500 : ∃ (n : ℕ), n * 32 > 500 ∧ n * 32 = 512 ∧ ∀ (m : ℕ), m * 32 > 500 → m * 32 ≥ 512 := by
  sorry

end least_multiple_32_over_500_l2035_203589


namespace pure_imaginary_condition_l2035_203518

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ z : ℂ, z = a + 1 - a * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_condition_l2035_203518


namespace equation_solution_l2035_203563

theorem equation_solution :
  let f : ℂ → ℂ := λ x => (x - 2)^4 + (x - 6)^4
  ∃ (a b c d : ℂ),
    (f a = 16 ∧ f b = 16 ∧ f c = 16 ∧ f d = 16) ∧
    (a = 4 + Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (b = 4 - Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (c = 4 + Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    (d = 4 - Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    ∀ (x : ℂ), f x = 16 → (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by
  sorry

end equation_solution_l2035_203563


namespace candy_count_solution_l2035_203560

def is_valid_candy_count (x : ℕ) : Prop :=
  ∃ (brother_takes : ℕ),
    x % 4 = 0 ∧
    x % 2 = 0 ∧
    2 ≤ brother_takes ∧
    brother_takes ≤ 6 ∧
    (x / 4 * 3 / 3 * 2 - 40 - brother_takes = 10)

theorem candy_count_solution :
  ∀ x : ℕ, is_valid_candy_count x ↔ (x = 108 ∨ x = 112) :=
sorry

end candy_count_solution_l2035_203560


namespace triangle_angles_l2035_203527

theorem triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 108 →          -- One angle is 108°
  b = 2 * c →        -- One angle is twice the other
  (b = 48 ∧ c = 24)  -- The two smaller angles are 48° and 24°
  := by sorry

end triangle_angles_l2035_203527


namespace meters_equivalence_l2035_203537

-- Define the conversion rates
def meters_to_decimeters : ℝ := 10
def meters_to_centimeters : ℝ := 100

-- Define the theorem
theorem meters_equivalence : 
  7.34 = 7 + (3 / meters_to_decimeters) + (4 / meters_to_centimeters) := by
  sorry

end meters_equivalence_l2035_203537


namespace solve_equations_l2035_203528

theorem solve_equations :
  (∃ x : ℝ, 1 - 3 * (1 - x) = 2 * x ∧ x = 2) ∧
  (∃ x : ℝ, (3 * x + 1) / 2 - (4 * x - 2) / 5 = 1 ∧ x = 1 / 7) :=
by sorry

end solve_equations_l2035_203528


namespace difference_of_squares_l2035_203541

theorem difference_of_squares (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) := by
  sorry

end difference_of_squares_l2035_203541


namespace log_inequality_implies_base_inequality_l2035_203587

theorem log_inequality_implies_base_inequality (a b : ℝ) 
  (h1 : (Real.log 3 / Real.log a) > (Real.log 3 / Real.log b)) 
  (h2 : (Real.log 3 / Real.log b) > 0) : b > a ∧ a > 1 := by
  sorry

end log_inequality_implies_base_inequality_l2035_203587


namespace simplify_complex_fraction_l2035_203525

theorem simplify_complex_fraction (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) :
  1 - (1 / (1 + a^2 / (1 - a^2))) = a^2 := by
  sorry

end simplify_complex_fraction_l2035_203525


namespace margo_walk_distance_l2035_203533

/-- Calculates the total distance walked given the time to walk to a friend's house,
    the time to return, and the average walking rate for the entire trip. -/
def total_distance (time_to_friend : ℚ) (time_from_friend : ℚ) (avg_rate : ℚ) : ℚ :=
  let total_time : ℚ := time_to_friend + time_from_friend
  let total_time_hours : ℚ := total_time / 60
  avg_rate * total_time_hours

/-- Proves that given the specific conditions, the total distance walked is 1.5 miles. -/
theorem margo_walk_distance :
  let time_to_friend : ℚ := 15
  let time_from_friend : ℚ := 10
  let avg_rate : ℚ := 18/5  -- 3.6 as a rational number
  total_distance time_to_friend time_from_friend avg_rate = 3/2 := by
  sorry

#eval total_distance 15 10 (18/5)

end margo_walk_distance_l2035_203533


namespace no_solution_exists_l2035_203506

theorem no_solution_exists : ¬∃ (m n : ℤ), 
  m ≠ n ∧ 
  988 < m ∧ m < 1991 ∧ 
  988 < n ∧ n < 1991 ∧ 
  ∃ (a : ℤ), m * n + n = a ^ 2 ∧ 
  ∃ (b : ℤ), m * n + m = b ^ 2 := by
  sorry

end no_solution_exists_l2035_203506


namespace jackson_earnings_l2035_203596

def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400

def vacuum_hours : ℝ := 2 * 2
def dishes_hours : ℝ := 0.5
def bathroom_hours : ℝ := 0.5 * 3

def gbp_to_usd : ℝ := 1.35
def jpy_to_usd : ℝ := 0.009

theorem jackson_earnings : 
  (vacuum_hours * usd_per_hour) + 
  (dishes_hours * gbp_per_hour * gbp_to_usd) + 
  (bathroom_hours * jpy_per_hour * jpy_to_usd) = 27.425 := by
sorry

end jackson_earnings_l2035_203596


namespace distribute_6_3_l2035_203574

/-- The number of ways to distribute n items among k categories, 
    with each category receiving at least one item. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 10 ways to distribute 6 items among 3 categories, 
    with each category receiving at least one item. -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end distribute_6_3_l2035_203574


namespace building_height_from_shadows_l2035_203543

/-- Given a bamboo pole and a building with their respective shadows, 
    calculate the height of the building using similar triangles. -/
theorem building_height_from_shadows 
  (bamboo_height : ℝ) 
  (bamboo_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_bamboo_height : bamboo_height = 1.8)
  (h_bamboo_shadow : bamboo_shadow = 3)
  (h_building_shadow : building_shadow = 35)
  : (bamboo_height / bamboo_shadow) * building_shadow = 21 := by
  sorry


end building_height_from_shadows_l2035_203543


namespace sin_neg_five_pi_sixths_l2035_203515

theorem sin_neg_five_pi_sixths : Real.sin (-5 * π / 6) = -1 / 2 := by
  sorry

end sin_neg_five_pi_sixths_l2035_203515


namespace area_of_triangle_AGE_l2035_203523

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (0, 5)

-- Define point E on BC
def E : ℝ × ℝ := (5, 2)

-- G is on the diagonal BD
def G : ℝ × ℝ := sorry

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_AGE :
  triangleArea A G E = 43.25 := by sorry

end area_of_triangle_AGE_l2035_203523


namespace find_y_l2035_203545

-- Define the binary operation ⊕
def binary_op (a b c d : ℤ) : ℤ × ℤ := (a + d, b - c)

-- Theorem statement
theorem find_y : ∀ x y : ℤ, 
  binary_op 2 5 1 1 = binary_op x y 2 0 → y = 6 := by
  sorry

end find_y_l2035_203545


namespace diagonals_30_sided_polygon_l2035_203552

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This should output 405

end diagonals_30_sided_polygon_l2035_203552


namespace triangle_inequality_l2035_203565

/-- Triangle inequality proof -/
theorem triangle_inequality (a b c s R r : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 → r > 0 →
  s = (a + b + c) / 2 →
  a + b > c → b + c > a → c + a > b →
  (a / (s - a)) + (b / (s - b)) + (c / (s - c)) ≥ 3 * R / r := by
  sorry

end triangle_inequality_l2035_203565


namespace trig_identity_l2035_203531

theorem trig_identity (α : Real) (h : Real.tan α = -1/2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := by
  sorry

end trig_identity_l2035_203531


namespace range_of_c_l2035_203550

theorem range_of_c (c : ℝ) : 
  (∀ x > 0, c^2 * x^2 - (c * x + 1) * Real.log x + c * x ≥ 0) ↔ c ≥ 1 / Real.exp 1 := by
  sorry

end range_of_c_l2035_203550


namespace parabola_sum_l2035_203561

-- Define a quadratic function
def quadratic (p q r : ℝ) : ℝ → ℝ := λ x => p * x^2 + q * x + r

theorem parabola_sum (p q r : ℝ) :
  -- The vertex of the parabola is (3, -1)
  (∀ x, quadratic p q r x ≥ quadratic p q r 3) ∧
  quadratic p q r 3 = -1 ∧
  -- The parabola passes through the point (0, 8)
  quadratic p q r 0 = 8
  →
  p + q + r = 3 := by
sorry

end parabola_sum_l2035_203561


namespace pool_volume_calculation_l2035_203577

/-- Calculates the total volume of a pool given its draining parameters -/
theorem pool_volume_calculation 
  (drain_rate : ℝ) 
  (drain_time : ℝ) 
  (initial_capacity_percentage : ℝ) : 
  drain_rate * drain_time / initial_capacity_percentage = 90000 :=
by
  sorry

#check pool_volume_calculation 60 1200 0.8

end pool_volume_calculation_l2035_203577


namespace sum_of_coefficients_l2035_203509

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9 →
  a + b + c + d + e = 60 := by
sorry

end sum_of_coefficients_l2035_203509
