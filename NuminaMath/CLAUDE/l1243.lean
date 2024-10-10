import Mathlib

namespace dave_train_books_l1243_124351

/-- The number of books about trains Dave bought -/
def num_train_books (num_animal_books num_space_books cost_per_book total_spent : ℕ) : ℕ :=
  (total_spent - (num_animal_books + num_space_books) * cost_per_book) / cost_per_book

theorem dave_train_books :
  num_train_books 8 6 6 102 = 3 :=
sorry

end dave_train_books_l1243_124351


namespace lcm_gcd_product_9_12_l1243_124374

theorem lcm_gcd_product_9_12 : Nat.lcm 9 12 * Nat.gcd 9 12 = 108 := by
  sorry

end lcm_gcd_product_9_12_l1243_124374


namespace sum_of_squares_lower_bound_range_of_a_l1243_124380

-- Part I
theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end sum_of_squares_lower_bound_range_of_a_l1243_124380


namespace function_properties_l1243_124349

-- Define the function f(x)
def f (d : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + d

-- State the theorem
theorem function_properties :
  ∃ (d : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≥ -4) ∧ 
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = -4) →
    d = 1 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≤ 23) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = 23) :=
by
  sorry


end function_properties_l1243_124349


namespace jeds_change_l1243_124394

/-- Given the conditions of Jed's board game purchase, prove the number of $5 bills received as change. -/
theorem jeds_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 6 → 
  game_cost = 15 → 
  payment = 100 → 
  change_bill = 5 → 
  (payment - num_games * game_cost) / change_bill = 2 := by
sorry

end jeds_change_l1243_124394


namespace rightmost_three_digits_of_3_to_1987_l1243_124344

theorem rightmost_three_digits_of_3_to_1987 : 3^1987 % 1000 = 187 := by sorry

end rightmost_three_digits_of_3_to_1987_l1243_124344


namespace exponential_function_inequality_range_l1243_124364

/-- The range of k for which the given inequality holds for all real x₁ and x₂ -/
theorem exponential_function_inequality_range :
  ∀ (k : ℝ), 
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → 
    |((Real.exp x₁) - (Real.exp x₂)) / (x₁ - x₂)| < |k| * ((Real.exp x₁) + (Real.exp x₂))) 
  ↔ 
  (k ≤ -1/2 ∨ k ≥ 1/2) :=
by sorry

end exponential_function_inequality_range_l1243_124364


namespace lioness_weight_l1243_124367

/-- The weight of a lioness given the weights of her cubs -/
theorem lioness_weight (L F M : ℝ) : 
  L = 6 * F →  -- The weight of the lioness is six times the weight of her female cub
  L = 4 * M →  -- The weight of the lioness is four times the weight of her male cub
  M - F = 14 → -- The difference between the weights of the male and female cub is 14 kg
  L = 168 :=   -- The weight of the lioness is 168 kg
by sorry

end lioness_weight_l1243_124367


namespace rectangular_prism_diagonals_l1243_124337

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of faces in a rectangular prism. -/
def faces_count : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges_count : ℕ := 12

/-- The number of diagonals in a rectangular prism. -/
def diagonals_count (rp : RectangularPrism) : ℕ := 16

/-- Theorem: A rectangular prism with different length, width, and height has exactly 16 diagonals. -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : 
  diagonals_count rp = 16 := by
  sorry

end rectangular_prism_diagonals_l1243_124337


namespace representatives_count_l1243_124363

/-- The number of ways to select representatives from a group with females and males -/
def select_representatives (num_females num_males num_representatives : ℕ) : ℕ :=
  Nat.choose num_females 1 * Nat.choose num_males 2 + 
  Nat.choose num_females 2 * Nat.choose num_males 1

/-- Theorem stating that selecting 3 representatives from 3 females and 4 males, 
    with at least one of each, results in 30 different ways -/
theorem representatives_count : select_representatives 3 4 3 = 30 := by
  sorry

#eval select_representatives 3 4 3

end representatives_count_l1243_124363


namespace julia_car_rental_cost_l1243_124324

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (mileageRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + mileageRate * miles

/-- Theorem stating that the total cost for Julia's car rental is $215 -/
theorem julia_car_rental_cost :
  carRentalCost 30 0.25 3 500 = 215 := by
  sorry

end julia_car_rental_cost_l1243_124324


namespace quadrilateral_perimeter_sum_l1243_124382

/-- A quadrilateral with vertices at (1,2), (4,6), (6,5), and (4,1) -/
def Quadrilateral : Set (ℝ × ℝ) :=
  {(1, 2), (4, 6), (6, 5), (4, 1)}

/-- The perimeter of the quadrilateral -/
noncomputable def perimeter : ℝ :=
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist (1, 2) (4, 6) + dist (4, 6) (6, 5) + dist (6, 5) (4, 1) + dist (4, 1) (1, 2)

/-- The theorem to be proved -/
theorem quadrilateral_perimeter_sum (c d : ℤ) :
  perimeter = c * Real.sqrt 5 + d * Real.sqrt 13 → c + d = 9 :=
by sorry

end quadrilateral_perimeter_sum_l1243_124382


namespace perimeter_ABCDEFG_l1243_124328

-- Define the points
variable (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := 
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem perimeter_ABCDEFG (h1 : is_equilateral A B C)
                          (h2 : is_equilateral A D E)
                          (h3 : is_equilateral E F G)
                          (h4 : is_midpoint D A C)
                          (h5 : is_midpoint G A E)
                          (h6 : is_midpoint F E G)
                          (h7 : dist A B = 6) :
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 25.5 := by
  sorry

end perimeter_ABCDEFG_l1243_124328


namespace tennis_players_count_l1243_124313

/-- Given a sports club with the following properties:
  * There are 42 total members
  * 20 members play badminton
  * 6 members play neither badminton nor tennis
  * 7 members play both badminton and tennis
  Prove that 23 members play tennis -/
theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 42)
  (h_badminton : badminton = 20)
  (h_neither : neither = 6)
  (h_both : both = 7) :
  ∃ tennis : ℕ, tennis = 23 ∧ 
  tennis = total - neither - (badminton - both) :=
by sorry

end tennis_players_count_l1243_124313


namespace shaded_area_of_carpet_l1243_124334

theorem shaded_area_of_carpet (S T : ℝ) : 
  12 / S = 4 →
  S / T = 4 →
  (8 * T^2 + S^2) = 27/2 := by
  sorry

end shaded_area_of_carpet_l1243_124334


namespace inequality_and_function_minimum_l1243_124305

-- Define the set A
def A (a : ℕ+) : Set ℝ := {x : ℝ | |x - 2| < a}

-- State the theorem
theorem inequality_and_function_minimum (a : ℕ+) 
  (h1 : (3/2 : ℝ) ∈ A a) 
  (h2 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end inequality_and_function_minimum_l1243_124305


namespace parabola_properties_l1243_124346

/-- Definition of the parabola -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Definition of the directrix -/
def directrix (x : ℝ) : Prop := x = -4

/-- Definition of a line passing through (-4, 0) with slope m -/
def line (m x y : ℝ) : Prop := y = m * (x + 4)

/-- Theorem stating the properties of the parabola and its intersecting lines -/
theorem parabola_properties :
  (∃ (x y : ℝ), directrix x ∧ y = 0) ∧ 
  (∀ (m : ℝ), (∃ (x y : ℝ), parabola x y ∧ line m x y) ↔ m ≤ 1/2) := by
  sorry

end parabola_properties_l1243_124346


namespace limit_expected_sides_l1243_124347

/-- The expected number of sides of a polygon after k cuts -/
def expected_sides (n : ℕ) (k : ℕ) : ℚ :=
  (n + 4 * k : ℚ) / (k + 1 : ℚ)

/-- Theorem: The limit of expected sides approaches 4 as k approaches infinity -/
theorem limit_expected_sides (n : ℕ) :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides n k - 4| < ε :=
sorry

end limit_expected_sides_l1243_124347


namespace polar_to_rectangular_conversion_l1243_124315

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 5 ∧ y = -5 * Real.sqrt 3 := by
  sorry

end polar_to_rectangular_conversion_l1243_124315


namespace cubic_function_properties_l1243_124338

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem stating the properties of the cubic function f -/
theorem cubic_function_properties :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧ 
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t : ℝ), 
      (t = 0 ∧ 2 * 0 + f a b 0 - 1 = 0) ∨
      (t = 1/4 ∧ 33 * (1/4) + 16 * (f a b (1/4)) - 16 = 0)) := by
  sorry

end cubic_function_properties_l1243_124338


namespace cube_sum_equals_one_l1243_124310

theorem cube_sum_equals_one (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = x * y * z) 
  (h2 : x + y + z = 1) : 
  x^3 + y^3 + z^3 = 1 := by
sorry

end cube_sum_equals_one_l1243_124310


namespace max_profit_year_l1243_124368

/-- Represents the financial model of the environmentally friendly building materials factory. -/
structure FactoryFinances where
  initialInvestment : ℕ
  firstYearOperatingCosts : ℕ
  annualOperatingCostsIncrease : ℕ
  annualRevenue : ℕ

/-- Calculates the net profit for a given year. -/
def netProfitAtYear (f : FactoryFinances) (year : ℕ) : ℤ :=
  (f.annualRevenue * year : ℤ) -
  (f.initialInvestment : ℤ) -
  (f.firstYearOperatingCosts * year : ℤ) -
  (f.annualOperatingCostsIncrease * (year * (year - 1) / 2) : ℤ)

/-- Theorem stating that the net profit reaches its maximum in the 10th year. -/
theorem max_profit_year (f : FactoryFinances)
  (h1 : f.initialInvestment = 720000)
  (h2 : f.firstYearOperatingCosts = 120000)
  (h3 : f.annualOperatingCostsIncrease = 40000)
  (h4 : f.annualRevenue = 500000) :
  ∀ y : ℕ, y ≠ 10 → netProfitAtYear f y ≤ netProfitAtYear f 10 :=
sorry

end max_profit_year_l1243_124368


namespace exponential_function_fixed_point_l1243_124308

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^(x-3) + 1 passes through the point (3, 2) -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 1
  f 3 = 2 := by sorry

end exponential_function_fixed_point_l1243_124308


namespace bridge_length_proof_l1243_124381

/-- Given a train of length 110 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds,
    prove that the length of the bridge is 265 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 265 := by
sorry

end bridge_length_proof_l1243_124381


namespace travel_time_proof_l1243_124345

/-- Given a person traveling at 20 km/hr for a distance of 160 km,
    prove that the time taken is 8 hours. -/
theorem travel_time_proof (speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : speed = 20)
    (h2 : distance = 160)
    (h3 : time * speed = distance) :
  time = 8 :=
by sorry

end travel_time_proof_l1243_124345


namespace even_iff_period_two_l1243_124312

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the condition f(1+x) = f(1-x)
def symmetry_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Define an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function with period 2
def has_period_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Theorem statement
theorem even_iff_period_two (f : ℝ → ℝ) (h : symmetry_condition f) :
  is_even f ↔ has_period_two f :=
sorry

end even_iff_period_two_l1243_124312


namespace final_price_in_euros_l1243_124304

-- Define the pin prices
def pin_prices : List ℝ := [23, 18, 20, 15, 25, 22, 19, 16, 24, 17]

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the exchange rate (USD to Euro)
def exchange_rate : ℝ := 0.85

-- Theorem statement
theorem final_price_in_euros :
  let original_price := pin_prices.sum
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + sales_tax_rate)
  let final_price := price_with_tax * exchange_rate
  ∃ ε > 0, |final_price - 155.28| < ε :=
sorry

end final_price_in_euros_l1243_124304


namespace double_inequality_l1243_124327

theorem double_inequality (a b : ℝ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end double_inequality_l1243_124327


namespace simplify_trig_expression_l1243_124330

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 4)) :
  Real.sqrt (1 - 2 * Real.sin (π + θ) * Real.sin ((3 * π) / 2 - θ)) = Real.cos θ - Real.sin θ := by
  sorry

end simplify_trig_expression_l1243_124330


namespace arrangement_exists_l1243_124323

/-- A type representing a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- A predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

/-- The main theorem statement -/
theorem arrangement_exists (S : Finset ℝ) (h : S.card = 100) :
  ∃ (f : Table), 
    (∀ x ∈ S, ∃ i j, f i j = x) ∧ 
    (∀ i j k l, adjacent i j k l → |f i j - f k l| ≠ 1) := by
  sorry

end arrangement_exists_l1243_124323


namespace min_value_quadratic_sum_l1243_124302

theorem min_value_quadratic_sum (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) : 
  (a + 1)^2 + 4*b^2 + 9*c^2 ≥ 144/49 := by
  sorry

end min_value_quadratic_sum_l1243_124302


namespace max_x_value_l1243_124360

theorem max_x_value (x y : ℝ) (h1 : x + y ≤ 1) (h2 : y + x + y ≤ 1) :
  x ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ ≤ 1 ∧ y₀ + x₀ + y₀ ≤ 1 ∧ x₀ = 2 := by
  sorry

end max_x_value_l1243_124360


namespace drums_filled_per_day_l1243_124311

/-- The number of drums filled per day given the total number of drums and days -/
def drums_per_day (total_drums : ℕ) (total_days : ℕ) : ℕ :=
  total_drums / total_days

/-- Theorem stating that 90 drums filled in 6 days results in 15 drums per day -/
theorem drums_filled_per_day :
  drums_per_day 90 6 = 15 := by
  sorry

end drums_filled_per_day_l1243_124311


namespace inverse_function_derivative_l1243_124373

-- Define the function f and its inverse g
variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (x₀ : ℝ) (y₀ : ℝ)

-- State the conditions
variable (hf : Differentiable ℝ f)
variable (hg : Differentiable ℝ g)
variable (hinverse : Function.LeftInverse g f ∧ Function.RightInverse g f)
variable (hderiv : (deriv f x₀) ≠ 0)
variable (hy : y₀ = f x₀)

-- State the theorem
theorem inverse_function_derivative :
  (deriv g y₀) = 1 / (deriv f x₀) := by sorry

end inverse_function_derivative_l1243_124373


namespace oatmeal_cookies_count_l1243_124306

def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def number_of_baggies : ℕ := 6

def total_cookies : ℕ := cookies_per_bag * number_of_baggies

theorem oatmeal_cookies_count :
  total_cookies - chocolate_chip_cookies = 41 :=
by sorry

end oatmeal_cookies_count_l1243_124306


namespace perpendicular_planes_from_lines_l1243_124342

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  perpendicular m α →
  parallel_lines m n →
  parallel_line_plane n β →
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_lines_l1243_124342


namespace boys_girls_ratio_l1243_124348

/-- Given a classroom with boys and girls, prove that the initial ratio of boys to girls is 1:1 --/
theorem boys_girls_ratio (total : ℕ) (left : ℕ) (B G : ℕ) : 
  total = 32 →  -- Total number of boys and girls initially
  left = 8 →  -- Number of girls who left
  B = 2 * (G - left) →  -- After girls left, there are twice as many boys as girls
  B + G = total →  -- Total is the sum of boys and girls
  B = G  -- The number of boys equals the number of girls, implying a 1:1 ratio
  := by sorry

end boys_girls_ratio_l1243_124348


namespace parabola_vertex_l1243_124391

/-- The parabola defined by y = -3(x-1)^2 - 2 has its vertex at (1, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * (x - 1)^2 - 2 → (1, -2) = (x, y) := by sorry

end parabola_vertex_l1243_124391


namespace cube_root_two_not_rational_plus_sqrt_l1243_124309

theorem cube_root_two_not_rational_plus_sqrt (a b c : ℚ) (hc : c > 0) :
  (a + b * Real.sqrt c) ^ 3 ≠ 2 :=
sorry

end cube_root_two_not_rational_plus_sqrt_l1243_124309


namespace R_value_when_S_is_5_l1243_124301

/-- Given that R = gS^2 - 6 and R = 15 when S = 3, prove that R = 157/3 when S = 5 -/
theorem R_value_when_S_is_5 (g : ℚ) :
  (∃ R, R = g * 3^2 - 6 ∧ R = 15) →
  g * 5^2 - 6 = 157 / 3 := by
sorry

end R_value_when_S_is_5_l1243_124301


namespace birch_tree_arrangement_probability_l1243_124370

def num_maple_trees : ℕ := 4
def num_oak_trees : ℕ := 5
def num_birch_trees : ℕ := 6

def total_trees : ℕ := num_maple_trees + num_oak_trees + num_birch_trees

def favorable_arrangements : ℕ := (Nat.choose 10 6) * (Nat.choose 9 4)
def total_arrangements : ℕ := Nat.factorial total_trees

theorem birch_tree_arrangement_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3003 := by
  sorry

end birch_tree_arrangement_probability_l1243_124370


namespace systematic_sample_first_product_l1243_124397

/-- Represents a systematic sample from a range of numbered products. -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  sample_interval : ℕ
  first_product : ℕ

/-- Creates a systematic sample given the total number of products and sample size. -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products,
    sample_size := sample_size,
    sample_interval := total_products / sample_size,
    first_product := 1 }

/-- Checks if a given product number is in the systematic sample. -/
def is_in_sample (s : SystematicSample) (product_number : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < s.sample_size ∧ product_number = s.first_product + k * s.sample_interval

/-- Theorem: In a systematic sample of size 5 from 80 products, 
    if product 42 is in the sample, then the first product's number is 10. -/
theorem systematic_sample_first_product :
  let s := create_systematic_sample 80 5
  is_in_sample s 42 → s.first_product = 10 := by
  sorry

end systematic_sample_first_product_l1243_124397


namespace S_formula_l1243_124396

def N (n : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem S_formula (n : ℕ) : S n = (4^n + 2) / 3 := by
  sorry

end S_formula_l1243_124396


namespace rainfall_sum_l1243_124352

theorem rainfall_sum (monday1 wednesday1 friday monday2 wednesday2 : ℝ)
  (h1 : monday1 = 0.17)
  (h2 : wednesday1 = 0.42)
  (h3 : friday = 0.08)
  (h4 : monday2 = 0.37)
  (h5 : wednesday2 = 0.51) :
  monday1 + wednesday1 + friday + monday2 + wednesday2 = 1.55 := by
sorry

end rainfall_sum_l1243_124352


namespace anthony_lunch_money_l1243_124390

theorem anthony_lunch_money (juice_cost cupcake_cost remaining_amount : ℕ) 
  (h1 : juice_cost = 27)
  (h2 : cupcake_cost = 40)
  (h3 : remaining_amount = 8) :
  juice_cost + cupcake_cost + remaining_amount = 75 := by
  sorry

end anthony_lunch_money_l1243_124390


namespace exactly_two_true_l1243_124372

-- Define the propositions
def proposition1 : Prop :=
  (∀ x, x^2 - 3*x + 2 = 0 → x = 2 ∨ x = 1) →
  (∀ x, x^2 - 3*x + 2 ≠ 0 → x ≠ 2 ∨ x ≠ 1)

def proposition2 : Prop :=
  (∀ x > 1, x^2 - 1 > 0) →
  (∃ x > 1, x^2 - 1 ≤ 0)

def proposition3 (p q : Prop) : Prop :=
  (¬p ∧ ¬q → ¬(p ∨ q)) ∧ ¬(¬(p ∨ q) → ¬p ∧ ¬q)

-- Theorem stating that exactly two propositions are true
theorem exactly_two_true :
  (¬proposition1 ∧ proposition2 ∧ proposition3 True False) ∨
  (¬proposition1 ∧ proposition2 ∧ proposition3 False True) ∨
  (proposition2 ∧ proposition3 True False ∧ proposition3 False True) :=
sorry

end exactly_two_true_l1243_124372


namespace max_surrounding_squares_l1243_124395

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Predicate to check if two squares are non-overlapping -/
def non_overlapping (s1 s2 : Square) : Prop :=
  sorry

/-- Function to count the number of non-overlapping squares around a central square -/
def count_surrounding_squares (central : Square) (surrounding : List Square) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of non-overlapping squares 
    that can be placed around a given square is 8 -/
theorem max_surrounding_squares (central : Square) (surrounding : List Square) :
  count_surrounding_squares central surrounding ≤ 8 :=
sorry

end max_surrounding_squares_l1243_124395


namespace continuous_function_zero_on_interval_l1243_124339

theorem continuous_function_zero_on_interval
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) :
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x = 0 := by
  sorry

end continuous_function_zero_on_interval_l1243_124339


namespace parallelogram_area_example_l1243_124366

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 25 15 = 375 := by
  sorry

end parallelogram_area_example_l1243_124366


namespace divisibility_by_nine_l1243_124383

theorem divisibility_by_nine (D E : Nat) : 
  D ≤ 9 → E ≤ 9 → (D * 100000 + 864000 + E * 100 + 72) % 9 = 0 →
  (D + E = 0 ∨ D + E = 9 ∨ D + E = 18) := by
  sorry

end divisibility_by_nine_l1243_124383


namespace polynomial_expansion_l1243_124365

theorem polynomial_expansion (x : ℝ) : 
  (5 * x - 3) * (2 * x^3 + 7 * x - 1) = 10 * x^4 - 6 * x^3 + 35 * x^2 - 26 * x + 3 := by
  sorry

end polynomial_expansion_l1243_124365


namespace solve_sqrt_equation_l1243_124331

theorem solve_sqrt_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end solve_sqrt_equation_l1243_124331


namespace yellow_stamp_price_is_two_l1243_124369

/-- Calculates the price per yellow stamp needed to reach a total sale amount --/
def price_per_yellow_stamp (red_count : ℕ) (red_price : ℚ) (blue_count : ℕ) (blue_price : ℚ) 
  (yellow_count : ℕ) (total_sale : ℚ) : ℚ :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining := total_sale - (red_earnings + blue_earnings)
  remaining / yellow_count

/-- Theorem stating that the price per yellow stamp is $2 given the problem conditions --/
theorem yellow_stamp_price_is_two :
  price_per_yellow_stamp 20 1.1 80 0.8 7 100 = 2 := by
  sorry

end yellow_stamp_price_is_two_l1243_124369


namespace polygon_sides_from_exterior_angle_l1243_124343

theorem polygon_sides_from_exterior_angle :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 :=
by
  sorry

end polygon_sides_from_exterior_angle_l1243_124343


namespace product_max_for_square_l1243_124378

/-- A quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The product of sums of opposite sides pairs -/
def productOfSums (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Theorem: The product of sums is maximum when the quadrilateral is a square -/
theorem product_max_for_square (q : CyclicQuadrilateral) :
  productOfSums q ≤ productOfSums { a := (q.a + q.b + q.c + q.d) / 4,
                                    b := (q.a + q.b + q.c + q.d) / 4,
                                    c := (q.a + q.b + q.c + q.d) / 4,
                                    d := (q.a + q.b + q.c + q.d) / 4,
                                    inscribed := sorry } := by
  sorry

end product_max_for_square_l1243_124378


namespace gcd_72_168_l1243_124314

theorem gcd_72_168 : Nat.gcd 72 168 = 24 := by
  sorry

end gcd_72_168_l1243_124314


namespace prime_power_difference_l1243_124362

theorem prime_power_difference (n : ℕ+) (p : ℕ) (k : ℕ) :
  (3 : ℕ) ^ n.val - (2 : ℕ) ^ n.val = p ^ k ∧ Nat.Prime p → Nat.Prime n.val :=
by sorry

end prime_power_difference_l1243_124362


namespace jean_gives_480_l1243_124332

/-- The amount Jean gives away to her grandchildren in a year -/
def total_amount_given (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proof that Jean gives away $480 to her grandchildren in a year -/
theorem jean_gives_480 :
  total_amount_given 3 2 80 = 480 :=
by sorry

end jean_gives_480_l1243_124332


namespace remaining_trees_correct_l1243_124300

/-- The number of dogwood trees remaining in a park after cutting some down. -/
def remaining_trees (part1 : ℝ) (part2 : ℝ) (cut : ℝ) : ℝ :=
  part1 + part2 - cut

/-- Theorem stating that the number of remaining trees is correct. -/
theorem remaining_trees_correct (part1 : ℝ) (part2 : ℝ) (cut : ℝ) :
  remaining_trees part1 part2 cut = part1 + part2 - cut :=
by sorry

end remaining_trees_correct_l1243_124300


namespace tuesday_occurs_five_times_in_august_l1243_124335

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := sorry

/-- August of year N -/
def august : Month := sorry

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

theorem tuesday_occurs_five_times_in_august 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Monday = 5)
  (h3 : august.days = 30) :
  countDayOccurrences august DayOfWeek.Tuesday = 5 := by sorry

end tuesday_occurs_five_times_in_august_l1243_124335


namespace circle_to_ellipse_transformation_l1243_124358

/-- A circle in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- An ellipse in the x'y'-plane -/
def Ellipse (x' y' : ℝ) : Prop := x'^2/16 + y'^2/4 = 1

/-- The scaling transformation -/
def ScalingTransformation (x' y' : ℝ) : ℝ × ℝ := (4*x', y')

theorem circle_to_ellipse_transformation :
  ∀ (x' y' : ℝ), 
  let (x, y) := ScalingTransformation x' y'
  Circle x y ↔ Ellipse x' y' := by
sorry

end circle_to_ellipse_transformation_l1243_124358


namespace f_derivative_l1243_124392

noncomputable def f (x : ℝ) : ℝ := 2 + x * Real.cos x

theorem f_derivative : 
  deriv f = λ x => Real.cos x - x * Real.sin x :=
sorry

end f_derivative_l1243_124392


namespace N_value_is_negative_twelve_point_five_l1243_124387

/-- Represents a grid with arithmetic sequences -/
structure ArithmeticGrid :=
  (row_first : ℚ)
  (col1_second : ℚ)
  (col1_third : ℚ)
  (col2_last : ℚ)
  (num_columns : ℕ)
  (num_rows : ℕ)

/-- Calculates the value of N in the arithmetic grid -/
def calculate_N (grid : ArithmeticGrid) : ℚ :=
  sorry

/-- Theorem stating that N equals -12.5 for the given grid -/
theorem N_value_is_negative_twelve_point_five :
  let grid : ArithmeticGrid := {
    row_first := 18,
    col1_second := 15,
    col1_third := 21,
    col2_last := -14,
    num_columns := 7,
    num_rows := 2
  }
  calculate_N grid = -12.5 := by sorry

end N_value_is_negative_twelve_point_five_l1243_124387


namespace quadratic_discriminant_l1243_124350

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x - 2/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := -2/5

theorem quadratic_discriminant : discriminant a b c = 876/25 := by
  sorry

end quadratic_discriminant_l1243_124350


namespace cards_satisfy_conditions_l1243_124377

def card1 : Finset Nat := {1, 4, 7}
def card2 : Finset Nat := {2, 3, 4}
def card3 : Finset Nat := {2, 5, 7}

theorem cards_satisfy_conditions : 
  (card1 ∩ card2).card = 1 ∧ 
  (card1 ∩ card3).card = 1 ∧ 
  (card2 ∩ card3).card = 1 := by
  sorry

end cards_satisfy_conditions_l1243_124377


namespace f_odd_and_decreasing_l1243_124319

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) :=
by
  sorry

end f_odd_and_decreasing_l1243_124319


namespace imaginary_part_of_z_l1243_124322

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I) / z = I) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l1243_124322


namespace min_square_sum_min_square_sum_achievable_l1243_124375

theorem min_square_sum (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 2 * x₃ = 50) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 1250 / 7 := by
  sorry

theorem min_square_sum_achievable : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  x₁ + 3 * x₂ + 2 * x₃ = 50 ∧ 
  x₁^2 + x₂^2 + x₃^2 = 1250 / 7 := by
  sorry

end min_square_sum_min_square_sum_achievable_l1243_124375


namespace roboducks_order_l1243_124354

theorem roboducks_order (shelves_percentage : ℚ) (storage_count : ℕ) : 
  shelves_percentage = 30 / 100 →
  storage_count = 140 →
  ∃ total : ℕ, total = 200 ∧ (1 - shelves_percentage) * total = storage_count :=
by
  sorry

end roboducks_order_l1243_124354


namespace hyperbola_parabola_intersection_range_l1243_124376

/-- The range of k for which a hyperbola and parabola have at most two intersections -/
theorem hyperbola_parabola_intersection_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 - y^2 + 1 = 0 ∧ y^2 = (k - 1) * x →
    (∃! p q : ℝ × ℝ, (p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∧
                     (q.1^2 - q.2^2 + 1 = 0 ∧ q.2^2 = (k - 1) * q.1) ∧
                     p ≠ q) ∨
    (∃! p : ℝ × ℝ, p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∨
    (∀ x y : ℝ, x^2 - y^2 + 1 ≠ 0 ∨ y^2 ≠ (k - 1) * x)) →
  -1 ≤ k ∧ k < 3 :=
by sorry

end hyperbola_parabola_intersection_range_l1243_124376


namespace smallest_block_size_l1243_124384

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are visible. -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Checks if the given dimensions satisfy the problem conditions. -/
def isValidBlock (d : BlockDimensions) : Prop :=
  invisibleCubes d = 300 ∧ d.length > 1 ∧ d.width > 1 ∧ d.height > 1

/-- Theorem stating that the smallest possible number of cubes is 462. -/
theorem smallest_block_size :
  ∃ (d : BlockDimensions), isValidBlock d ∧ totalCubes d = 462 ∧
  (∀ (d' : BlockDimensions), isValidBlock d' → totalCubes d' ≥ 462) :=
sorry

end smallest_block_size_l1243_124384


namespace circle_angle_distance_sum_l1243_124389

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the angle
def Angle : Type := Point → Point → Point → Prop

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the line segment
def LineSegment (p q : Point) : Point → Prop := sorry

-- State the theorem
theorem circle_angle_distance_sum
  (circle : Circle)
  (angle : Angle)
  (A B C : Point)
  (h1 : circle A ∧ circle B ∧ circle C)
  (h2 : angle A B C)
  (h3 : ∀ p, LineSegment A B p → distance C p = 8)
  (h4 : ∃ (d1 d2 : ℝ), d1 = d2 + 30 ∧
        (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2))) :
  ∃ (d1 d2 : ℝ), d1 + d2 = 34 ∧
    (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2)) :=
sorry

end circle_angle_distance_sum_l1243_124389


namespace order_of_numbers_l1243_124340

theorem order_of_numbers : 7^(3/10) > 0.3^7 ∧ 0.3^7 > Real.log 0.3 := by
  sorry

end order_of_numbers_l1243_124340


namespace simplify_complex_fraction_power_l1243_124333

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction_power :
  ((1 + i) / (1 - i)) ^ 1002 = -1 :=
by
  sorry

end simplify_complex_fraction_power_l1243_124333


namespace yahs_to_bahs_500_l1243_124388

/-- Represents the exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Represents the exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 10 / 6

/-- Converts yahs to bahs -/
def yahs_to_bahs (yahs : ℚ) : ℚ :=
  yahs * (1 / rah_to_yah_rate) * (1 / bah_to_rah_rate)

theorem yahs_to_bahs_500 :
  yahs_to_bahs 500 = 187.5 := by sorry

end yahs_to_bahs_500_l1243_124388


namespace problem_statements_l1243_124356

theorem problem_statements :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2))) :=
by sorry

end problem_statements_l1243_124356


namespace square_root_of_64_l1243_124393

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ↔ x = 8 ∨ x = -8 := by sorry

end square_root_of_64_l1243_124393


namespace product_of_difference_and_sum_of_squares_l1243_124316

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 21) : 
  a * b = 6 := by
sorry

end product_of_difference_and_sum_of_squares_l1243_124316


namespace fish_ratio_proof_l1243_124320

theorem fish_ratio_proof (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) (total_brought_home : ℕ)
  (h1 : ken_released = 3)
  (h2 : kendra_caught = 30)
  (h3 : (ken_caught - ken_released) + kendra_caught = total_brought_home)
  (h4 : total_brought_home = 87) :
  (ken_caught : ℚ) / kendra_caught = 19 / 10 := by
  sorry

end fish_ratio_proof_l1243_124320


namespace keyboard_warrior_estimate_l1243_124303

theorem keyboard_warrior_estimate (total_population : ℕ) (sample_size : ℕ) (favorable_count : ℕ) 
  (h1 : total_population = 9600)
  (h2 : sample_size = 50)
  (h3 : favorable_count = 15) :
  (total_population : ℚ) * (1 - favorable_count / sample_size) = 6720 := by
  sorry

end keyboard_warrior_estimate_l1243_124303


namespace smallest_sum_of_exponents_l1243_124355

theorem smallest_sum_of_exponents (m n : ℕ+) (h1 : m > n) 
  (h2 : 2012^(m.val) % 1000 = 2012^(n.val) % 1000) : 
  ∃ (k l : ℕ+), k.val + l.val = 104 ∧ k > l ∧ 
  2012^(k.val) % 1000 = 2012^(l.val) % 1000 ∧
  ∀ (p q : ℕ+), p > q → 2012^(p.val) % 1000 = 2012^(q.val) % 1000 → 
  p.val + q.val ≥ 104 :=
sorry

end smallest_sum_of_exponents_l1243_124355


namespace class_average_score_l1243_124357

theorem class_average_score
  (num_boys : ℕ)
  (num_girls : ℕ)
  (avg_score_boys : ℚ)
  (avg_score_girls : ℚ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 4)
  (h3 : avg_score_boys = 84)
  (h4 : avg_score_girls = 92) :
  (num_boys * avg_score_boys + num_girls * avg_score_girls) / (num_boys + num_girls) = 86 :=
by
  sorry

end class_average_score_l1243_124357


namespace pure_imaginary_solution_real_sum_solution_l1243_124336

def is_pure_imaginary (z : ℂ) : Prop := ∃ a : ℝ, z = Complex.I * a

theorem pure_imaginary_solution (z : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : Complex.abs (z - 1) = Complex.abs (z - 1 + Complex.I)) : 
  z = Complex.I ∨ z = -Complex.I :=
sorry

theorem real_sum_solution (z : ℂ) 
  (h1 : ∃ r : ℝ, z + 10 / z = r) 
  (h2 : 1 ≤ (z + 10 / z).re ∧ (z + 10 / z).re ≤ 6) :
  z = 1 + 3 * Complex.I ∨ z = 3 + Complex.I ∨ z = 3 - Complex.I :=
sorry

end pure_imaginary_solution_real_sum_solution_l1243_124336


namespace f_sixteen_value_l1243_124386

theorem f_sixteen_value (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x) = -2 * f x) 
  (h2 : f 1 = -3) : 
  f 16 = -48 := by
sorry

end f_sixteen_value_l1243_124386


namespace theresa_required_hours_l1243_124329

/-- The average number of hours Theresa needs to work per week over 4 weeks -/
def required_average : ℝ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The minimum total hours Theresa needs to work -/
def minimum_total_hours : ℝ := 50

/-- The hours Theresa worked in the first week -/
def first_week_hours : ℝ := 15

/-- The hours Theresa worked in the second week -/
def second_week_hours : ℝ := 8

/-- The number of remaining weeks -/
def remaining_weeks : ℕ := 2

theorem theresa_required_hours :
  let total_worked := first_week_hours + second_week_hours
  let remaining_hours := minimum_total_hours - total_worked
  (remaining_hours / remaining_weeks : ℝ) = 13.5 ∧
  remaining_hours ≥ required_average * remaining_weeks := by
  sorry

end theresa_required_hours_l1243_124329


namespace inverse_inequality_implies_reverse_l1243_124341

theorem inverse_inequality_implies_reverse (a b : ℝ) :
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end inverse_inequality_implies_reverse_l1243_124341


namespace medium_mall_sample_l1243_124353

def stratified_sample (total_sample : ℕ) (ratio : List ℕ) : List ℕ :=
  let total_ratio := ratio.sum
  ratio.map (λ r => (total_sample * r) / total_ratio)

theorem medium_mall_sample :
  let ratio := [2, 4, 9]
  let sample := stratified_sample 45 ratio
  sample[1] = 12 := by sorry

end medium_mall_sample_l1243_124353


namespace fraction_simplification_l1243_124325

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end fraction_simplification_l1243_124325


namespace triangle_equilateral_from_cos_product_l1243_124361

/-- A triangle is equilateral if all its angles are equal -/
def IsEquilateral (A B C : ℝ) : Prop := A = B ∧ B = C

/-- Given a triangle ABC, if cos(A-B)cos(B-C)cos(C-A)=1, then the triangle is equilateral -/
theorem triangle_equilateral_from_cos_product (A B C : ℝ) 
  (h : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  IsEquilateral A B C := by
  sorry


end triangle_equilateral_from_cos_product_l1243_124361


namespace helper_sequences_count_l1243_124359

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of student helpers possible in a week -/
def helper_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating that the number of different sequences of student helpers in a week is 3375 -/
theorem helper_sequences_count : helper_sequences = 3375 := by
  sorry

end helper_sequences_count_l1243_124359


namespace smallest_prime_divisor_of_sum_l1243_124307

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (4^11 + 6^13)) → 
  2 ∣ (4^11 + 6^13) ∧ 
  ∀ p : ℕ, Nat.Prime p → p ∣ (4^11 + 6^13) → p ≥ 2 :=
by sorry

end smallest_prime_divisor_of_sum_l1243_124307


namespace largest_average_is_17_multiples_l1243_124321

def upper_bound : ℕ := 100810

def average_of_multiples (n : ℕ) : ℚ :=
  let last_multiple := upper_bound - (upper_bound % n)
  (n + last_multiple) / 2

theorem largest_average_is_17_multiples :
  average_of_multiples 17 > average_of_multiples 11 ∧
  average_of_multiples 17 > average_of_multiples 13 ∧
  average_of_multiples 17 > average_of_multiples 19 :=
by sorry

end largest_average_is_17_multiples_l1243_124321


namespace ellen_chairs_count_l1243_124317

/-- The number of chairs Ellen bought at a garage sale -/
def num_chairs : ℕ := 180 / 15

/-- The cost of each chair in dollars -/
def chair_cost : ℕ := 15

/-- The total amount Ellen spent in dollars -/
def total_spent : ℕ := 180

theorem ellen_chairs_count :
  num_chairs = 12 ∧ chair_cost * num_chairs = total_spent :=
sorry

end ellen_chairs_count_l1243_124317


namespace triangle_ABC_c_value_l1243_124318

/-- Triangle ABC with vertices A(0, 4), B(3, 0), and C(c, 6) has area 7 and 0 < c < 3 -/
def triangle_ABC (c : ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (c, 6)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  (area = 7) ∧ (0 < c) ∧ (c < 3)

/-- If triangle ABC satisfies the given conditions, then c = 2 -/
theorem triangle_ABC_c_value :
  ∀ c : ℝ, triangle_ABC c → c = 2 := by
  sorry

end triangle_ABC_c_value_l1243_124318


namespace f_is_linear_l1243_124399

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x - 2 = 1/3 -/
def f (x : ℝ) : ℝ := x - 2

theorem f_is_linear : is_linear_equation f := by
  sorry

end f_is_linear_l1243_124399


namespace nth_group_sum_correct_l1243_124385

/-- The sum of the n-th group in the sequence of positive integers grouped as 1, 2+3, 4+5+6, 7+8+9+10, ... -/
def nth_group_sum (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- The first element of the n-th group -/
def first_element (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

/-- The last element of the n-th group -/
def last_element (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem nth_group_sum_correct (n : ℕ) (h : n > 0) :
  nth_group_sum n = (n * (first_element n + last_element n)) / 2 :=
by sorry

end nth_group_sum_correct_l1243_124385


namespace solve_equation_l1243_124371

theorem solve_equation : ∃ X : ℝ, 
  (((4 - 3.5 * (2 + 1/7 - (1 + 1/5))) / 0.16) / X = 
  ((3 + 2/7 - (3/14 / (1/6))) / (41 + 23/84 - (40 + 49/60))) ∧ X = 1) := by
  sorry

end solve_equation_l1243_124371


namespace negation_of_exists_lt_is_forall_ge_l1243_124326

theorem negation_of_exists_lt_is_forall_ge (p : Prop) : 
  (¬ (∃ x : ℝ, x^2 + 2*x < 0)) ↔ (∀ x : ℝ, x^2 + 2*x ≥ 0) :=
by sorry

end negation_of_exists_lt_is_forall_ge_l1243_124326


namespace house_height_difference_l1243_124379

/-- Given three houses with heights 80 feet, 70 feet, and 99 feet,
    prove that the difference between the average height and 80 feet is 3 feet. -/
theorem house_height_difference (h₁ h₂ h₃ : ℝ) 
  (h₁_height : h₁ = 80)
  (h₂_height : h₂ = 70)
  (h₃_height : h₃ = 99) :
  (h₁ + h₂ + h₃) / 3 - h₁ = 3 := by
  sorry

end house_height_difference_l1243_124379


namespace negative_sum_reciprocal_l1243_124398

theorem negative_sum_reciprocal (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  min (a + 1/b) (min (b + 1/c) (c + 1/a)) ≤ -2 := by
  sorry

end negative_sum_reciprocal_l1243_124398
