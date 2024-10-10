import Mathlib

namespace curve_symmetry_line_dot_product_l2588_258885

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem curve_symmetry_line_dot_product 
  (P Q : ℝ × ℝ) (m : ℝ) 
  (h_curve_P : curve P.1 P.2)
  (h_curve_Q : curve Q.1 Q.2)
  (h_symmetry : ∃ (x y : ℝ), symmetry_line m x y ∧ 
    (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2)
  (h_dot_product : dot_product_condition P.1 P.2 Q.1 Q.2) :
  m = -1 ∧ Q.2 = -Q.1 + 1 ∧ P.2 = -P.1 + 1 :=
sorry

end curve_symmetry_line_dot_product_l2588_258885


namespace total_marbles_l2588_258860

/-- The number of marbles of each color in a collection --/
structure MarbleCollection where
  red : ℝ
  blue : ℝ
  green : ℝ
  yellow : ℝ

/-- The conditions given in the problem --/
def satisfiesConditions (m : MarbleCollection) : Prop :=
  m.red = 1.3 * m.blue ∧
  m.green = 1.7 * m.red ∧
  m.yellow = m.blue + 40

/-- The theorem to be proved --/
theorem total_marbles (m : MarbleCollection) (h : satisfiesConditions m) :
  m.red + m.blue + m.green + m.yellow = 3.84615 * m.red + 40 := by
  sorry


end total_marbles_l2588_258860


namespace collinear_vectors_m_value_l2588_258850

theorem collinear_vectors_m_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, -3]
  ∀ m : ℝ, (∃ k : ℝ, k • (m • a + b) = 3 • a - b) → m = -3 := by
  sorry

end collinear_vectors_m_value_l2588_258850


namespace vertical_asymptote_at_five_l2588_258827

/-- The function f(x) = (x^2 + 3x + 4) / (x - 5) has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five (x : ℝ) :
  let f := fun (x : ℝ) => (x^2 + 3*x + 4) / (x - 5)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (abs (f (5 + δ)) > 1/δ) ∧ (abs (f (5 - δ)) > 1/δ) :=
by sorry

end vertical_asymptote_at_five_l2588_258827


namespace trigonometric_equations_l2588_258825

theorem trigonometric_equations (n m : ℤ) 
  (hn : -120 ≤ n ∧ n ≤ 120) (hm : -120 ≤ m ∧ m ≤ 120) :
  (Real.sin (n * π / 180) = Real.sin (580 * π / 180) → n = -40) ∧
  (Real.cos (m * π / 180) = Real.cos (300 * π / 180) → m = -60) := by
  sorry

end trigonometric_equations_l2588_258825


namespace larger_variance_greater_fluctuation_l2588_258887

-- Define a type for our data set
def DataSet := List ℝ

-- Define variance for a data set
def variance (data : DataSet) : ℝ := sorry

-- Define a measure of fluctuation for a data set
def fluctuation (data : DataSet) : ℝ := sorry

-- Theorem stating that larger variance implies greater fluctuation
theorem larger_variance_greater_fluctuation 
  (data1 data2 : DataSet) :
  variance data1 > variance data2 → fluctuation data1 > fluctuation data2 := by
  sorry

end larger_variance_greater_fluctuation_l2588_258887


namespace sector_area_given_arc_length_l2588_258867

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) : r * 2 = 4 → r^2 = 4 := by sorry

end sector_area_given_arc_length_l2588_258867


namespace max_product_constrained_sum_l2588_258868

theorem max_product_constrained_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x > 0) (h3 : y > 0) :
  x * y ≤ 400 ∧ ∃ (a b : ℝ), a + b = 40 ∧ a > 0 ∧ b > 0 ∧ a * b = 400 := by
  sorry

end max_product_constrained_sum_l2588_258868


namespace sum_of_multiples_is_even_l2588_258839

theorem sum_of_multiples_is_even (c d : ℤ) (hc : 6 ∣ c) (hd : 9 ∣ d) : Even (c + d) := by
  sorry

end sum_of_multiples_is_even_l2588_258839


namespace oplus_example_l2588_258863

/-- Definition of the ⊕ operation -/
def oplus (a b c : ℝ) (k : ℤ) : ℝ := b^2 - k * (a^2 * c)

/-- Theorem stating that ⊕(2, 5, 3, 3) = -11 -/
theorem oplus_example : oplus 2 5 3 3 = -11 := by
  sorry

end oplus_example_l2588_258863


namespace remainder_calculation_l2588_258861

theorem remainder_calculation (L S R : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 1620)
  (h3 : L = 6 * S + R) : 
  R = 90 := by
sorry

end remainder_calculation_l2588_258861


namespace triangle_area_l2588_258812

/-- Theorem: Area of a triangle with sides 8, 15, and 17 --/
theorem triangle_area (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  (1/2) * a * b = 60 := by
  sorry

end triangle_area_l2588_258812


namespace hyperbola_standard_equation_l2588_258833

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- Slope of the asymptote -/
  asymptote_slope : ℝ
  /-- Half of the focal length -/
  c : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  (∃ (x y : ℝ), x^2 / 36 - y^2 / 64 = 1) ∨ 
  (∃ (x y : ℝ), y^2 / 64 - x^2 / 36 = 1)

/-- Theorem stating the standard equation of a hyperbola given its asymptote slope and focal length -/
theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 4/3)
  (h_focal_length : h.c = 10) : 
  standard_equation h :=
sorry

end hyperbola_standard_equation_l2588_258833


namespace inequality_solution_set_l2588_258873

theorem inequality_solution_set (x : ℝ) :
  (4 * x^2 - 9 * x > 5) ↔ (x < -1/4 ∨ x > 5) := by sorry

end inequality_solution_set_l2588_258873


namespace two_true_propositions_l2588_258844

def p (x y : ℝ) : Prop := (x > |y|) → (x > y)

def q (x y : ℝ) : Prop := (x + y > 0) → (x^2 > y^2)

theorem two_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧
  ¬(¬(p x y) ∧ ¬(q x y)) ∧
  (p x y ∧ ¬(q x y)) ∧
  ¬(p x y ∧ q x y) :=
sorry

end two_true_propositions_l2588_258844


namespace product_repeating_decimal_and_seven_l2588_258828

theorem product_repeating_decimal_and_seven :
  let x : ℚ := 152 / 333
  x * 7 = 1064 / 333 := by sorry

end product_repeating_decimal_and_seven_l2588_258828


namespace adam_bought_26_books_l2588_258841

/-- The number of books Adam bought on his shopping trip -/
def books_bought (initial_books shelf_count books_per_shelf leftover_books : ℕ) : ℕ :=
  shelf_count * books_per_shelf + leftover_books - initial_books

/-- Theorem stating that Adam bought 26 books -/
theorem adam_bought_26_books : 
  books_bought 56 4 20 2 = 26 := by
  sorry

end adam_bought_26_books_l2588_258841


namespace volunteer_allocation_schemes_l2588_258847

theorem volunteer_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 5 → k = 3 → m = 2 → 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.factorial k) = 180 := by
  sorry

end volunteer_allocation_schemes_l2588_258847


namespace total_candies_in_store_l2588_258854

def chocolate_boxes : List Nat := [200, 320, 500, 500, 768, 768]
def candy_tubs : List Nat := [1380, 1150, 1150, 1720]

theorem total_candies_in_store : 
  (chocolate_boxes.sum + candy_tubs.sum) = 8456 := by
  sorry

end total_candies_in_store_l2588_258854


namespace only_48_satisfies_l2588_258898

/-- A function that returns the digits of a positive integer -/
def digits (n : ℕ+) : List ℕ :=
  sorry

/-- A function that checks if all elements in a list are between 1 and 9 (inclusive) -/
def all_between_1_and_9 (l : List ℕ) : Prop :=
  sorry

/-- The main theorem -/
theorem only_48_satisfies : ∃! (n : ℕ+),
  (n : ℕ) = (3/2 : ℚ) * (digits n).prod ∧
  all_between_1_and_9 (digits n) :=
by
  sorry

end only_48_satisfies_l2588_258898


namespace percentage_of_270_is_90_l2588_258832

theorem percentage_of_270_is_90 : 
  (90 : ℝ) / 270 * 100 = 33.33 := by sorry

end percentage_of_270_is_90_l2588_258832


namespace max_min_difference_circle_l2588_258816

theorem max_min_difference_circle (x y : ℝ) (h : x^2 - 4*x + y^2 + 3 = 0) :
  let f := fun (x y : ℝ) => x^2 + y^2
  ∃ (M m : ℝ), (∀ (a b : ℝ), a^2 - 4*a + b^2 + 3 = 0 → f a b ≤ M) ∧
               (∀ (a b : ℝ), a^2 - 4*a + b^2 + 3 = 0 → m ≤ f a b) ∧
               M - m = 8 :=
by sorry

end max_min_difference_circle_l2588_258816


namespace age_difference_eighteen_l2588_258811

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h_odd_tens : Odd tens)
  (h_odd_ones : Odd ones)
  (h_range_tens : tens ≤ 9)
  (h_range_ones : ones ≤ 9)

/-- The age as a natural number -/
def Age.toNat (a : Age) : Nat := 10 * a.tens + a.ones

theorem age_difference_eighteen :
  ∀ (alice bob : Age),
    alice.tens = bob.ones ∧ 
    alice.ones = bob.tens ∧
    (alice.toNat + 7 = 3 * (bob.toNat + 7)) →
    bob.toNat - alice.toNat = 18 := by
  sorry

#check age_difference_eighteen

end age_difference_eighteen_l2588_258811


namespace complex_equality_implies_b_value_l2588_258814

theorem complex_equality_implies_b_value (b : ℝ) : 
  let z : ℂ := (1 + b * I) / (2 + I)
  z.re = z.im → b = 3 := by
  sorry

end complex_equality_implies_b_value_l2588_258814


namespace binomial_60_3_l2588_258875

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l2588_258875


namespace metal_waste_l2588_258869

/-- Given a rectangle with sides a and b (a < b), calculate the total metal wasted
    after cutting out a maximum circular piece and then a maximum square piece from the circle. -/
theorem metal_waste (a b : ℝ) (h : 0 < a ∧ a < b) :
  let circle_area := Real.pi * (a / 2)^2
  let square_side := a / Real.sqrt 2
  let square_area := square_side^2
  ab - square_area = ab - a^2 / 2 := by sorry

end metal_waste_l2588_258869


namespace house_selling_price_l2588_258802

/-- Proves that the selling price of each house is $120,000 given the problem conditions -/
theorem house_selling_price (C S : ℝ) : 
  (C + 100000 = 1.5 * S - 60000) → -- Construction cost of certain house equals its selling price minus profit
  (C = S - 100000) →               -- Construction cost difference between certain house and others
  S = 120000 := by
sorry

end house_selling_price_l2588_258802


namespace min_difference_triangle_sides_l2588_258836

theorem min_difference_triangle_sides (a b c : ℕ) : 
  a + b + c = 2007 →
  a < b →
  b ≤ c →
  (∀ a' b' c' : ℕ, a' + b' + c' = 2007 → a' < b' → b' ≤ c' → b - a ≤ b' - a') →
  b - a = 1 :=
by sorry

end min_difference_triangle_sides_l2588_258836


namespace white_surface_area_fraction_l2588_258895

theorem white_surface_area_fraction (cube_edge : ℕ) (small_cube_edge : ℕ) 
  (total_cubes : ℕ) (white_cubes : ℕ) :
  cube_edge = 4 →
  small_cube_edge = 1 →
  total_cubes = 64 →
  white_cubes = 16 →
  (white_cubes : ℚ) / ((cube_edge ^ 2 * 6) : ℚ) = 1 / 6 := by
  sorry

end white_surface_area_fraction_l2588_258895


namespace megan_total_songs_l2588_258855

/-- Calculates the total number of songs bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Proves that the total number of songs bought is correct for Megan's scenario. -/
theorem megan_total_songs :
  total_songs 8 2 7 = 42 :=
by sorry

end megan_total_songs_l2588_258855


namespace simplify_expression_l2588_258893

theorem simplify_expression (a b : ℝ) : (18*a + 45*b) + (15*a + 36*b) - (12*a + 40*b) = 21*a + 41*b := by
  sorry

end simplify_expression_l2588_258893


namespace second_person_speed_l2588_258889

/-- Given two people moving in opposite directions, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem second_person_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 36)
  (h3 : speed1 = 6)
  (h4 : distance = time * (speed1 + speed2)) :
  speed2 = 3 :=
sorry

end second_person_speed_l2588_258889


namespace alcohol_fraction_in_mixture_l2588_258804

theorem alcohol_fraction_in_mixture (water_volume : ℚ) (alcohol_water_ratio : ℚ) :
  water_volume = 4/5 →
  alcohol_water_ratio = 3/4 →
  (1 - water_volume) = 3/5 :=
by
  sorry

end alcohol_fraction_in_mixture_l2588_258804


namespace profit_at_least_150_cents_l2588_258807

-- Define the buying and selling prices
def orange_buy_price : ℚ := 15 / 4
def orange_sell_price : ℚ := 35 / 7
def apple_buy_price : ℚ := 20 / 5
def apple_sell_price : ℚ := 50 / 8

-- Define the profit function
def profit (num_oranges num_apples : ℕ) : ℚ :=
  (orange_sell_price - orange_buy_price) * num_oranges +
  (apple_sell_price - apple_buy_price) * num_apples

-- Theorem statement
theorem profit_at_least_150_cents :
  profit 43 43 ≥ 150 := by sorry

end profit_at_least_150_cents_l2588_258807


namespace binomial_n_minus_two_l2588_258826

theorem binomial_n_minus_two (n : ℕ+) : Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_n_minus_two_l2588_258826


namespace sqrt_expression_equality_l2588_258823

theorem sqrt_expression_equality : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/6) * Real.sqrt 12 + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l2588_258823


namespace max_k_value_l2588_258896

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := (n^2 + 11*n) / 2

/-- Sequence a_n -/
def a (n : ℕ) : ℚ := n + 5

/-- Sequence b_n -/
def b (n : ℕ) : ℚ := 3*n + 2

/-- Sequence c_n -/
def c (n : ℕ) : ℚ := 6 / ((2*a n - 11) * (2*b n - 1))

/-- Sum of first n terms of sequence c_n -/
def T (n : ℕ) : ℚ := 1 - 1 / (2*n + 1)

theorem max_k_value (n : ℕ+) :
  ∃ (k : ℕ), k = 37 ∧ 
  (∀ (m : ℕ+), T m > (m : ℚ) / 57) ∧
  (∀ (l : ℕ), l > k → ∃ (m : ℕ+), T m ≤ (l : ℚ) / 57) :=
sorry

end max_k_value_l2588_258896


namespace bus_back_seat_capacity_l2588_258813

/-- Represents the seating capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  people_per_seat : ℕ
  total_capacity : ℕ

/-- Calculates the number of people who can sit at the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : ℕ :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people who can sit at the back seat of the given bus -/
theorem bus_back_seat_capacity :
  ∃ (bus : BusSeating),
    bus.left_seats = 15 ∧
    bus.right_seats = bus.left_seats - 3 ∧
    bus.people_per_seat = 3 ∧
    bus.total_capacity = 91 ∧
    back_seat_capacity bus = 10 := by
  sorry

end bus_back_seat_capacity_l2588_258813


namespace equation_solution_l2588_258806

theorem equation_solution (r : ℝ) : 
  (r^2 - 6*r + 8)/(r^2 - 9*r + 20) = (r^2 - 3*r - 10)/(r^2 - 2*r - 15) ↔ r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 :=
by sorry

end equation_solution_l2588_258806


namespace line_through_points_sum_m_b_l2588_258818

/-- Given a line passing through points (2,8) and (5,2) with equation y = mx + b, prove that m + b = 10 -/
theorem line_through_points_sum_m_b (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x = 2 ∧ y = 8) ∨ (x = 5 ∧ y = 2)) →
  m + b = 10 := by
  sorry

end line_through_points_sum_m_b_l2588_258818


namespace gcf_of_180_270_450_l2588_258851

theorem gcf_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by
  sorry

end gcf_of_180_270_450_l2588_258851


namespace fewer_buses_than_cars_l2588_258866

theorem fewer_buses_than_cars (ratio_buses_to_cars : ℚ) (num_cars : ℕ) : 
  ratio_buses_to_cars = 1 / 13 → num_cars = 65 → num_cars - (num_cars / 13 : ℕ) = 60 := by
  sorry

end fewer_buses_than_cars_l2588_258866


namespace seven_representations_l2588_258810

/-- An arithmetic expression using digits, operations, and parentheses -/
inductive ArithExpr
  | Digit (d : ℕ)
  | Add (e1 e2 : ArithExpr)
  | Sub (e1 e2 : ArithExpr)
  | Mul (e1 e2 : ArithExpr)
  | Div (e1 e2 : ArithExpr)

/-- Count the number of times a specific digit appears in an ArithExpr -/
def countDigit (e : ArithExpr) (d : ℕ) : ℕ := sorry

/-- Evaluate an ArithExpr to a rational number -/
def evaluate (e : ArithExpr) : ℚ := sorry

/-- Theorem: For each integer n from 1 to 10 inclusive, there exists an arithmetic
    expression using the digit 7 exactly four times, along with operation signs
    and parentheses, that evaluates to n. -/
theorem seven_representations :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 →
  ∃ e : ArithExpr, countDigit e 7 = 4 ∧ evaluate e = n := by sorry

end seven_representations_l2588_258810


namespace cauchy_schwarz_inequality_l2588_258872

theorem cauchy_schwarz_inequality (x y a b : ℝ) : 
  a * x + b * y ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) := by
  sorry

end cauchy_schwarz_inequality_l2588_258872


namespace quadratic_roots_sum_squares_l2588_258859

theorem quadratic_roots_sum_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    2 * x₁^2 + k * x₁ - 2 * k + 1 = 0 ∧
    2 * x₂^2 + k * x₂ - 2 * k + 1 = 0 ∧
    x₁^2 + x₂^2 = 29/4) → 
  k = 3 :=
by sorry


end quadratic_roots_sum_squares_l2588_258859


namespace f_composition_l2588_258821

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 5

-- State the theorem
theorem f_composition (x : ℝ) : f (3 * x - 7) = 9 * x - 16 := by
  sorry

end f_composition_l2588_258821


namespace equation_solution_l2588_258822

theorem equation_solution : 
  ∃ x : ℚ, x ≠ -2 ∧ (x^2 + 3*x + 4) / (x + 2) = x + 6 :=
by
  use -8/5
  sorry

#check equation_solution

end equation_solution_l2588_258822


namespace distance_to_origin_l2588_258865

theorem distance_to_origin (x y : ℝ) (h1 : y = 14) 
  (h2 : Real.sqrt ((x - 1)^2 + (y - 8)^2) = 8) (h3 : x > 1) :
  Real.sqrt (x^2 + y^2) = 15 := by
  sorry

end distance_to_origin_l2588_258865


namespace leftover_pie_share_l2588_258891

theorem leftover_pie_share (total_leftover : ℚ) (num_people : ℕ) : 
  total_leftover = 6/7 ∧ num_people = 3 → 
  total_leftover / num_people = 2/7 := by
  sorry

end leftover_pie_share_l2588_258891


namespace rahul_share_is_142_l2588_258862

/-- Calculates the share of payment for a worker given the total payment and the work rates of two workers --/
def calculate_share (total_payment : ℚ) (rahul_days : ℚ) (rajesh_days : ℚ) : ℚ :=
  let rahul_rate := 1 / rahul_days
  let rajesh_rate := 1 / rajesh_days
  let combined_rate := rahul_rate + rajesh_rate
  let rahul_share_ratio := rahul_rate / combined_rate
  total_payment * rahul_share_ratio

/-- Theorem stating that Rahul's share is 142 given the problem conditions --/
theorem rahul_share_is_142 :
  calculate_share 355 3 2 = 142 := by
  sorry

end rahul_share_is_142_l2588_258862


namespace special_function_inequality_l2588_258886

/-- A function satisfying the given differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  domain : ∀ x, x < 0 → f x ≠ 0
  ineq : ∀ x, x < 0 → 2 * f x + x * deriv f x > x^2

/-- The main theorem -/
theorem special_function_inequality (φ : SpecialFunction) :
  ∀ x, (x + 2016)^2 * φ.f (x + 2016) - 4 * φ.f (-2) > 0 ↔ x < -2018 :=
sorry

end special_function_inequality_l2588_258886


namespace class_selection_probabilities_l2588_258819

/-- Represents the total number of classes -/
def total_classes : ℕ := 10

/-- Represents the number of classes to be selected -/
def selected_classes : ℕ := 3

/-- Represents the class number we're interested in -/
def target_class : ℕ := 4

/-- Probability of the target class being drawn first -/
def prob_first : ℝ := sorry

/-- Probability of the target class being drawn second -/
def prob_second : ℝ := sorry

/-- Theorem stating the probabilities of the target class being drawn first and second -/
theorem class_selection_probabilities :
  prob_first = sorry ∧ prob_second = sorry :=
sorry

end class_selection_probabilities_l2588_258819


namespace smallest_x_floor_is_13_l2588_258843

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- Define the property for x
def is_valid_x (x : ℝ) : Prop :=
  x > 2 ∧ tan_deg x = tan_deg (x^2)

-- State the theorem
theorem smallest_x_floor_is_13 :
  ∃ x : ℝ, is_valid_x x ∧ 
  (∀ y : ℝ, is_valid_x y → x ≤ y) ∧
  ⌊x⌋ = 13 :=
sorry

end smallest_x_floor_is_13_l2588_258843


namespace parallel_lines_m_value_l2588_258838

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

-- Define the parallel relation
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), l₁ m x y ↔ l₂ m x y

-- Theorem statement
theorem parallel_lines_m_value (m : ℝ) :
  parallel m → m = 2 ∨ m = -3 :=
by sorry

end parallel_lines_m_value_l2588_258838


namespace no_self_composite_plus_1987_function_l2588_258849

theorem no_self_composite_plus_1987_function :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end no_self_composite_plus_1987_function_l2588_258849


namespace optimal_laundry_additions_l2588_258834

-- Define constants
def total_capacity : ℝ := 20
def clothes_weight : ℝ := 5
def initial_detergent_scoops : ℝ := 2
def scoop_weight : ℝ := 0.02
def optimal_ratio : ℝ := 0.004  -- 4 g per kg = 0.004 kg per kg

-- Define the problem
theorem optimal_laundry_additions 
  (h1 : total_capacity = 20)
  (h2 : clothes_weight = 5)
  (h3 : initial_detergent_scoops = 2)
  (h4 : scoop_weight = 0.02)
  (h5 : optimal_ratio = 0.004) :
  ∃ (additional_detergent additional_water : ℝ),
    -- The total weight matches the capacity
    clothes_weight + initial_detergent_scoops * scoop_weight + additional_detergent + additional_water = total_capacity ∧
    -- The ratio of total detergent to water is optimal
    (initial_detergent_scoops * scoop_weight + additional_detergent) / additional_water = optimal_ratio ∧
    -- The additional detergent is 0.02 kg
    additional_detergent = 0.02 ∧
    -- The additional water is 14.94 kg
    additional_water = 14.94 :=
by
  sorry


end optimal_laundry_additions_l2588_258834


namespace sum_of_squares_l2588_258829

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 86) : x^2 + y^2 = 404 := by
  sorry

end sum_of_squares_l2588_258829


namespace f_is_even_function_l2588_258857

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f (x : ℝ) : ℝ := x^2

theorem f_is_even_function : is_even_function f := by
  sorry

end f_is_even_function_l2588_258857


namespace candy_ratio_l2588_258831

theorem candy_ratio : 
  ∀ (emily_candies bob_candies jennifer_candies : ℕ),
    emily_candies = 6 →
    bob_candies = 4 →
    jennifer_candies = 3 * bob_candies →
    (jennifer_candies : ℚ) / emily_candies = 2 / 1 :=
by
  sorry

end candy_ratio_l2588_258831


namespace double_area_rectangle_l2588_258853

/-- The area of a rectangle with dimensions 50 cm × 160 cm is exactly double
    the area of a rectangle with dimensions 50 cm × 80 cm. -/
theorem double_area_rectangle : 
  ∀ (width height new_height : ℕ), 
    width = 50 → height = 80 → new_height = 160 →
    2 * (width * height) = width * new_height := by
  sorry

end double_area_rectangle_l2588_258853


namespace intercept_sum_l2588_258848

/-- The modulus of the congruence -/
def m : ℕ := 17

/-- The congruence relation -/
def congruence (x y : ℕ) : Prop :=
  (5 * x) % m = (3 * y + 2) % m

/-- Definition of x-intercept -/
def x_intercept (x₀ : ℕ) : Prop :=
  x₀ < m ∧ congruence x₀ 0

/-- Definition of y-intercept -/
def y_intercept (y₀ : ℕ) : Prop :=
  y₀ < m ∧ congruence 0 y₀

/-- The main theorem -/
theorem intercept_sum :
  ∀ x₀ y₀ : ℕ, x_intercept x₀ → y_intercept y₀ → x₀ + y₀ = 19 :=
by sorry

end intercept_sum_l2588_258848


namespace min_value_of_expression_l2588_258809

theorem min_value_of_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := ((a + b) / c)^4 + ((b + c) / d)^4 + ((c + d) / a)^4 + ((d + a) / b)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end min_value_of_expression_l2588_258809


namespace vegetables_problem_l2588_258815

theorem vegetables_problem (potatoes carrots onions green_beans : ℕ) :
  carrots = 6 * potatoes →
  onions = 2 * carrots →
  green_beans = onions / 3 →
  green_beans = 8 →
  potatoes = 2 := by
  sorry

end vegetables_problem_l2588_258815


namespace fifth_term_integer_probability_l2588_258878

def sequence_rule (prev : ℤ) (is_heads : Bool) : ℤ :=
  if is_heads then
    3 * prev + 1
  else if prev % 3 = 0 then
    prev / 3 - 1
  else
    prev - 2

def fourth_term_rule (third_term : ℤ) (third_term_was_heads : Bool) : ℤ :=
  sequence_rule third_term third_term_was_heads

def is_integer (x : ℚ) : Prop :=
  ∃ n : ℤ, x = n

theorem fifth_term_integer_probability :
  let first_term := 4
  let coin_probability := (1 : ℚ) / 2
  ∀ second_term_heads third_term_heads fifth_term_heads : Bool,
    is_integer (sequence_rule
      (fourth_term_rule
        (sequence_rule
          (sequence_rule first_term second_term_heads)
          third_term_heads)
        third_term_heads)
      fifth_term_heads) :=
by sorry

end fifth_term_integer_probability_l2588_258878


namespace product_of_x_and_z_l2588_258856

theorem product_of_x_and_z (x y z : ℕ+) 
  (hx : x = 4 * y) 
  (hz : z = 2 * x) 
  (hsum : x + y + z = 3 * y^2) : 
  (x : ℚ) * (z : ℚ) = 5408 / 9 := by
sorry

end product_of_x_and_z_l2588_258856


namespace a_10_value_l2588_258883

/-- Given a sequence {aₙ} where aₙ = (-1)ⁿ · 1/(2n+1), prove that a₁₀ = 1/21 -/
theorem a_10_value (a : ℕ → ℚ) (h : ∀ n, a n = (-1)^n / (2*n + 1)) : 
  a 10 = 1 / 21 := by
sorry

end a_10_value_l2588_258883


namespace no_solution_absolute_value_plus_seven_l2588_258880

theorem no_solution_absolute_value_plus_seven :
  (∀ x : ℝ, |x| + 7 ≠ 0) ∧
  (∃ x : ℝ, (x - 5)^2 = 0) ∧
  (∃ x : ℝ, Real.sqrt (x + 9) - 3 = 0) ∧
  (∃ x : ℝ, (x + 4)^(1/3) - 1 = 0) ∧
  (∃ x : ℝ, |x + 6| - 5 = 0) :=
by sorry

end no_solution_absolute_value_plus_seven_l2588_258880


namespace total_animals_seen_l2588_258876

/-- Represents the number of animals Erica saw on Saturday -/
def saturday_animals : ℕ := 3 + 2

/-- Represents the number of animals Erica saw on Sunday -/
def sunday_animals : ℕ := 2 + 5

/-- Represents the number of animals Erica saw on Monday -/
def monday_animals : ℕ := 5 + 3

/-- Theorem stating that the total number of animals Erica saw is 20 -/
theorem total_animals_seen : saturday_animals + sunday_animals + monday_animals = 20 := by
  sorry

end total_animals_seen_l2588_258876


namespace ellipse_condition_l2588_258817

/-- Represents the equation of a conic section -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Checks if a conic section is a non-degenerate ellipse -/
def isNonDegenerateEllipse (conic : ConicSection) (m : ℝ) : Prop :=
  conic.a > 0 ∧ conic.b > 0 ∧ conic.a * conic.b * m > 0

/-- The main theorem stating the condition for the given equation to be a non-degenerate ellipse -/
theorem ellipse_condition (m : ℝ) : 
  isNonDegenerateEllipse ⟨3, 2, 0, -6, -16, -m⟩ m ↔ m > -35 := by sorry

end ellipse_condition_l2588_258817


namespace xiaojun_school_time_l2588_258882

/-- Xiaojun's information -/
structure Student where
  weight : ℝ
  height : ℝ
  morning_routine_time : ℝ
  distance_to_school : ℝ
  walking_speed : ℝ
  time_to_school : ℝ

/-- Theorem: Given Xiaojun's walking speed and distance to school, prove that the time taken to get to school is 15 minutes -/
theorem xiaojun_school_time (xiaojun : Student)
  (h1 : xiaojun.walking_speed = 1.5)
  (h2 : xiaojun.distance_to_school = 1350)
  : xiaojun.time_to_school = 15 := by
  sorry


end xiaojun_school_time_l2588_258882


namespace exam_passing_probability_l2588_258884

def total_questions : ℕ := 10
def selected_questions : ℕ := 3
def questions_student_can_answer : ℕ := 6
def questions_to_pass : ℕ := 2

def probability_of_passing : ℚ :=
  (Nat.choose questions_student_can_answer selected_questions +
   Nat.choose questions_student_can_answer (selected_questions - 1) *
   Nat.choose (total_questions - questions_student_can_answer) 1) /
  Nat.choose total_questions selected_questions

theorem exam_passing_probability :
  probability_of_passing = 2 / 3 := by sorry

end exam_passing_probability_l2588_258884


namespace reciprocal_of_abs_negative_two_l2588_258881

theorem reciprocal_of_abs_negative_two : (|-2|)⁻¹ = (1/2 : ℝ) := by
  sorry

end reciprocal_of_abs_negative_two_l2588_258881


namespace school_population_l2588_258864

theorem school_population (blind : ℕ) (deaf : ℕ) (other : ℕ) : 
  deaf = 3 * blind → 
  other = 2 * blind → 
  deaf = 180 → 
  blind + deaf + other = 360 :=
by
  sorry

end school_population_l2588_258864


namespace distance_after_10_hours_l2588_258897

/-- The distance between two people walking in the same direction for a given time -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two people walking for 10 hours at 5.5 kmph and 7.5 kmph is 20 km -/
theorem distance_after_10_hours :
  let speed1 : ℝ := 5.5
  let speed2 : ℝ := 7.5
  let time : ℝ := 10
  distance_between speed1 speed2 time = 20 := by
  sorry

end distance_after_10_hours_l2588_258897


namespace max_value_and_constraint_optimization_l2588_258830

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

-- State the theorem
theorem max_value_and_constraint_optimization :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧ 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + 3*b^2 + 2*c^2 = 2 → 
    a*b + 2*b*c ≤ 1 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀^2 + 3*b₀^2 + 2*c₀^2 = 2 ∧ a₀*b₀ + 2*b₀*c₀ = 1) :=
by sorry


end max_value_and_constraint_optimization_l2588_258830


namespace condition_necessary_not_sufficient_l2588_258800

/-- Predicate defining when the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  m - 1 > 0 ∧ 3 - m > 0 ∧ m - 1 ≠ 3 - m

/-- The condition given in the problem statement -/
def condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → condition m) ∧
  ¬(∀ m : ℝ, condition m → is_ellipse m) :=
sorry

end condition_necessary_not_sufficient_l2588_258800


namespace prism_volume_l2588_258874

/-- The volume of a right rectangular prism with face areas 10, 14, and 35 square inches is 70 cubic inches. -/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 10) 
  (area2 : w * h = 14) 
  (area3 : l * h = 35) : 
  l * w * h = 70 := by
  sorry

end prism_volume_l2588_258874


namespace area_of_R_l2588_258888

-- Define the points and line segments
def A : ℝ × ℝ := (-36, 0)
def B : ℝ × ℝ := (36, 0)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (0, 30)

def AB : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = ((1 - t) • A.1 + t • B.1, 0)}
def CD : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (0, (1 - t) • C.2 + t • D.2)}

-- Define the region R
def R : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, -36 ≤ x ∧ x ≤ 36 ∧ -30 ≤ y ∧ y ≤ 30 ∧ p = (x/2, y/2)}

-- State the theorem
theorem area_of_R : MeasureTheory.volume R = 1080 := by sorry

end area_of_R_l2588_258888


namespace binomial_expansion_ratio_l2588_258808

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61/60 := by
sorry

end binomial_expansion_ratio_l2588_258808


namespace domain_shift_l2588_258824

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_shift (h : Set.Icc (-1 : ℝ) 1 = {x | ∃ y, f (y + 1) = x}) :
  {x | ∃ y, f y = x} = Set.Icc (-2 : ℝ) 0 := by sorry

end domain_shift_l2588_258824


namespace min_distance_after_nine_minutes_l2588_258890

/-- Represents the robot's position on a 2D grid -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the possible directions the robot can face -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a turn the robot can make -/
inductive Turn
  | Left
  | Right

/-- The distance the robot travels in one minute -/
def distancePerMinute : ℕ := 10

/-- The total number of minutes the robot moves -/
def totalMinutes : ℕ := 9

/-- A function to calculate the Manhattan distance between two positions -/
def manhattanDistance (p1 p2 : Position) : ℕ :=
  (Int.natAbs (p1.x - p2.x)) + (Int.natAbs (p1.y - p2.y))

/-- A function to simulate the robot's movement for one minute -/
def moveOneMinute (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.North => { x := pos.x, y := pos.y + distancePerMinute }
  | Direction.East => { x := pos.x + distancePerMinute, y := pos.y }
  | Direction.South => { x := pos.x, y := pos.y - distancePerMinute }
  | Direction.West => { x := pos.x - distancePerMinute, y := pos.y }

/-- The theorem stating that the minimum distance from the starting point after 9 minutes is 10 meters -/
theorem min_distance_after_nine_minutes :
  ∃ (finalPos : Position),
    manhattanDistance { x := 0, y := 0 } finalPos = distancePerMinute ∧
    (∀ (pos : Position),
      (∃ (moves : List Turn),
        moves.length = totalMinutes - 1 ∧
        -- First move is always East
        (let firstMove := moveOneMinute { x := 0, y := 0 } Direction.East
        -- Subsequent moves can involve turns
        let finalPosition := moves.foldl
          (fun acc turn =>
            let newDir := match turn with
              | Turn.Left => Direction.North  -- Simplified turn logic
              | Turn.Right => Direction.South
            moveOneMinute acc newDir)
          firstMove
        finalPosition = pos)) →
      manhattanDistance { x := 0, y := 0 } pos ≥ distancePerMinute) :=
sorry

end min_distance_after_nine_minutes_l2588_258890


namespace last_number_problem_l2588_258871

theorem last_number_problem (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : (b + c + d) / 3 = 5)
  (h3 : a + d = 11) :
  d = 4 := by
sorry

end last_number_problem_l2588_258871


namespace meaningful_fraction_range_l2588_258842

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 4)) ↔ x ≠ 4 := by sorry

end meaningful_fraction_range_l2588_258842


namespace complex_number_problem_l2588_258835

theorem complex_number_problem (z : ℂ) (h : 2 * z + Complex.abs z = 3 + 6 * Complex.I) :
  z = 3 * Complex.I ∧ 
  ∀ (b c : ℝ), (z ^ 2 + b * z + c = 0) → (b - c = -9) := by
  sorry

end complex_number_problem_l2588_258835


namespace lcm_gcd_problem_l2588_258879

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 3780)
  (h2 : Nat.gcd a b = 18)
  (h3 : a = 180) :
  b = 378 := by
  sorry

end lcm_gcd_problem_l2588_258879


namespace jewelry_store_restocking_l2588_258820

/-- A jewelry store restocking problem -/
theorem jewelry_store_restocking
  (necklace_capacity : ℕ)
  (current_necklaces : ℕ)
  (current_rings : ℕ)
  (bracelet_capacity : ℕ)
  (current_bracelets : ℕ)
  (necklace_cost : ℕ)
  (ring_cost : ℕ)
  (bracelet_cost : ℕ)
  (total_cost : ℕ)
  (h1 : necklace_capacity = 12)
  (h2 : current_necklaces = 5)
  (h3 : current_rings = 18)
  (h4 : bracelet_capacity = 15)
  (h5 : current_bracelets = 8)
  (h6 : necklace_cost = 4)
  (h7 : ring_cost = 10)
  (h8 : bracelet_cost = 5)
  (h9 : total_cost = 183) :
  ∃ (ring_capacity : ℕ), ring_capacity = 30 ∧
    (necklace_capacity - current_necklaces) * necklace_cost +
    (ring_capacity - current_rings) * ring_cost +
    (bracelet_capacity - current_bracelets) * bracelet_cost = total_cost :=
by sorry

end jewelry_store_restocking_l2588_258820


namespace quadratic_condition_l2588_258840

/-- For the equation (m-2)x^2 + 3mx + 1 = 0 to be a quadratic equation in x, m ≠ 2 must hold. -/
theorem quadratic_condition (m : ℝ) : 
  (∀ x, ∃ y, y = (m - 2) * x^2 + 3 * m * x + 1) → m ≠ 2 := by
  sorry

end quadratic_condition_l2588_258840


namespace geometric_series_sum_specific_geometric_series_l2588_258858

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a ≠ 0 → 
  |r| < 1 → 
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

theorem specific_geometric_series : 
  (∑' n, (1/4) * (1/2)^n) = 1/2 :=
sorry

end geometric_series_sum_specific_geometric_series_l2588_258858


namespace expression_value_l2588_258801

/-- The polynomial function p(x) = x^2 - x + 1 -/
def p (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem: If α is a root of p(p(p(p(x)))), then the given expression equals -1 -/
theorem expression_value (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 := by
  sorry

end expression_value_l2588_258801


namespace circle_tangent_condition_l2588_258877

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate for a circle being tangent to the x-axis at the origin -/
def is_tangent_at_origin (c : Circle) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + c.D * x + c.E * y + c.F = 0 → 
    (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0)

theorem circle_tangent_condition (c : Circle) :
  is_tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end circle_tangent_condition_l2588_258877


namespace exponent_rule_l2588_258852

theorem exponent_rule (x : ℝ) : 2 * x^2 * (3 * x^2) = 6 * x^4 := by
  sorry

end exponent_rule_l2588_258852


namespace sixteen_power_divided_by_eight_l2588_258805

theorem sixteen_power_divided_by_eight (n : ℕ) : 
  n = 16^2023 → (n / 8 : ℕ) = 2^8089 := by
  sorry

end sixteen_power_divided_by_eight_l2588_258805


namespace trig_identity_l2588_258870

theorem trig_identity (α : ℝ) :
  (Real.sin (2 * α) - Real.sin (3 * α) + Real.sin (4 * α)) /
  (Real.cos (2 * α) - Real.cos (3 * α) + Real.cos (4 * α)) =
  Real.tan (3 * α) := by
  sorry

end trig_identity_l2588_258870


namespace multiple_with_specific_remainders_l2588_258894

theorem multiple_with_specific_remainders (n : ℕ) : 
  (∃ k : ℕ, n = 23 * k) ∧ 
  (n % 1821 = 710) ∧ 
  (n % 24 = 13) ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 23 * k) ∧ (m % 1821 = 710) ∧ (m % 24 = 13))) ∧ 
  (n = 3024) → 
  23 = 23 := by sorry

end multiple_with_specific_remainders_l2588_258894


namespace x_minus_y_squared_times_x_plus_y_l2588_258899

theorem x_minus_y_squared_times_x_plus_y (x y : ℝ) (hx : x = 8) (hy : y = 3) :
  (x - y)^2 * (x + y) = 275 := by sorry

end x_minus_y_squared_times_x_plus_y_l2588_258899


namespace no_positive_solution_l2588_258846

theorem no_positive_solution :
  ¬ ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^4 + y^4 + z^4 = 13 ∧
    x^3 * y^3 * z + y^3 * z^3 * x + z^3 * x^3 * y = 6 * Real.sqrt 3 ∧
    x^3 * y * z + y^3 * z * x + z^3 * x * y = 5 * Real.sqrt 3 := by
  sorry

end no_positive_solution_l2588_258846


namespace min_product_of_tangent_line_to_unit_circle_l2588_258803

theorem min_product_of_tangent_line_to_unit_circle (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x^2 + y^2 = 1 ∧ x/a + y/b = 1) → a * b ≥ 2 := by
  sorry

end min_product_of_tangent_line_to_unit_circle_l2588_258803


namespace complex_magnitude_problem_l2588_258892

theorem complex_magnitude_problem (z : ℂ) (h : (Complex.I - 2) * z = 4 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l2588_258892


namespace child_growth_l2588_258845

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) :
  current_height - previous_height = 3 := by
  sorry

end child_growth_l2588_258845


namespace probability_is_one_third_l2588_258837

/-- Represents a ball with a label -/
structure Ball :=
  (label : ℕ)

/-- The set of all balls in the box -/
def box : Finset Ball := sorry

/-- The condition that the sum of labels on two balls is 5 -/
def sumIs5 (b1 b2 : Ball) : Prop :=
  b1.label + b2.label = 5

/-- The set of all possible pairs of balls -/
def allPairs : Finset (Ball × Ball) := sorry

/-- The set of favorable pairs (sum is 5) -/
def favorablePairs : Finset (Ball × Ball) := sorry

/-- The probability of drawing two balls with sum 5 -/
def probability : ℚ := (favorablePairs.card : ℚ) / (allPairs.card : ℚ)

theorem probability_is_one_third :
  probability = 1/3 := by sorry

end probability_is_one_third_l2588_258837
