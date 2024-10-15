import Mathlib

namespace NUMINAMATH_CALUDE_town_budget_ratio_l3303_330382

theorem town_budget_ratio (total_budget education public_spaces : ℕ) 
  (h1 : total_budget = 32000000)
  (h2 : education = 12000000)
  (h3 : public_spaces = 4000000) :
  (total_budget - education - public_spaces) * 2 = total_budget :=
by sorry

end NUMINAMATH_CALUDE_town_budget_ratio_l3303_330382


namespace NUMINAMATH_CALUDE_triangle_max_area_l3303_330309

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    prove that the maximum area is 9√7 when a = 6 and √7 * b * cos(A) = 3 * a * sin(B) -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 6 →
  Real.sqrt 7 * b * Real.cos A = 3 * a * Real.sin B →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S ≤ 9 * Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3303_330309


namespace NUMINAMATH_CALUDE_product_of_max_min_elements_l3303_330337

def S : Set ℝ := {z | ∃ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 5 ∧ z = 5/x + y}

theorem product_of_max_min_elements (M N : ℝ) : 
  (∀ z ∈ S, z ≤ M) ∧ (M ∈ S) ∧ (∀ z ∈ S, N ≤ z) ∧ (N ∈ S) →
  M * N = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_max_min_elements_l3303_330337


namespace NUMINAMATH_CALUDE_rectangle_inside_circle_l3303_330325

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the unit circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  unit_circle p.1 p.2

-- Define a point inside the unit circle
def point_inside_circle (q : ℝ × ℝ) : Prop :=
  q.1^2 + q.2^2 < 1

-- Define the rectangle with diagonal pq and sides parallel to axes
def rectangle_with_diagonal (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | p.1 ≤ r.1 ∧ r.1 ≤ q.1 ∧ q.2 ≤ r.2 ∧ r.2 ≤ p.2 ∨
       q.1 ≤ r.1 ∧ r.1 ≤ p.1 ∧ p.2 ≤ r.2 ∧ r.2 ≤ q.2}

-- Theorem statement
theorem rectangle_inside_circle (p q : ℝ × ℝ) :
  point_on_circle p → point_inside_circle q →
  ∀ r ∈ rectangle_with_diagonal p q, r.1^2 + r.2^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_inside_circle_l3303_330325


namespace NUMINAMATH_CALUDE_tan_beta_value_l3303_330383

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (1 - Real.cos (2 * α)) = 1)
  (h2 : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3303_330383


namespace NUMINAMATH_CALUDE_children_tickets_sold_l3303_330348

theorem children_tickets_sold (adult_price child_price total_tickets total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100)
  (h5 : ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue) :
  ∃ (child_tickets : ℚ), child_tickets = 200 := by
  sorry

end NUMINAMATH_CALUDE_children_tickets_sold_l3303_330348


namespace NUMINAMATH_CALUDE_mean_temperature_is_zero_l3303_330371

def temperatures : List ℤ := [-3, -1, -6, 0, 4, 6]

theorem mean_temperature_is_zero : 
  (temperatures.sum : ℚ) / temperatures.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_zero_l3303_330371


namespace NUMINAMATH_CALUDE_rectangle_area_with_fixed_dimension_l3303_330314

theorem rectangle_area_with_fixed_dimension (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter is 200 cm
  (w = 30 ∨ l = 30) →      -- one dimension is fixed at 30 cm
  (l * w = 2100)           -- area is 2100 square cm
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_fixed_dimension_l3303_330314


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l3303_330339

/-- The number of seconds in one minute -/
def seconds_per_minute : ℝ := 60

/-- The number of minutes we want to convert to seconds -/
def minutes : ℝ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : minutes * seconds_per_minute = 750 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l3303_330339


namespace NUMINAMATH_CALUDE_base_prime_rep_441_l3303_330395

def base_prime_representation (n : ℕ) (primes : List ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 441 using primes 2, 3, 5, and 7 is 0202 -/
theorem base_prime_rep_441 : 
  base_prime_representation 441 [2, 3, 5, 7] = [0, 2, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_441_l3303_330395


namespace NUMINAMATH_CALUDE_equation_solutions_l3303_330374

def equation (x y : ℤ) : ℤ := 
  4*x^3 + 4*x^2*y - 15*x*y^2 - 18*y^3 - 12*x^2 + 6*x*y + 36*y^2 + 5*x - 10*y

def solution_set : Set (ℤ × ℤ) :=
  {p | p.1 = 1 ∧ p.2 = 1} ∪ {p | ∃ (y : ℕ), p.1 = 2*y ∧ p.2 = y}

theorem equation_solutions :
  ∀ (x y : ℤ), x > 0 ∧ y > 0 →
    (equation x y = 0 ↔ (x, y) ∈ solution_set) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3303_330374


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_l3303_330335

/-- Represents the admission fee structure and total collection -/
structure AdmissionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collected : ℕ

/-- Finds the ratio of adults to children closest to 1 given admission data -/
def closest_ratio_to_one (data : AdmissionData) : ℚ :=
  sorry

/-- The main theorem stating that the closest ratio to 1 is exactly 1 for the given data -/
theorem closest_ratio_is_one :
  let data : AdmissionData := {
    adult_fee := 30,
    child_fee := 15,
    total_collected := 2700
  }
  closest_ratio_to_one data = 1 := by sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_l3303_330335


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l3303_330342

def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_equation_solution :
  ∀ x : ℕ, binomial 15 (2*x+1) = binomial 15 (x+2) → x = 1 ∨ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l3303_330342


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l3303_330307

theorem stratified_sampling_proportion (total : ℕ) (males : ℕ) (females_selected : ℕ) :
  total = 220 →
  males = 60 →
  females_selected = 32 →
  (males / total : ℚ) = ((12 : ℕ) / (12 + females_selected) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l3303_330307


namespace NUMINAMATH_CALUDE_max_total_points_l3303_330318

/-- Represents the carnival game setup and Tiffany's current state -/
structure CarnivalGame where
  initial_money : ℕ := 3
  game_cost : ℕ := 1
  rings_per_game : ℕ := 5
  red_points : ℕ := 2
  green_points : ℕ := 3
  blue_points : ℕ := 5
  blue_success_rate : ℚ := 1/10
  time_limit : ℕ := 1
  games_played : ℕ := 2
  current_red : ℕ := 4
  current_green : ℕ := 5
  current_blue : ℕ := 1

/-- Calculates the maximum possible points for a single game -/
def max_points_per_game (game : CarnivalGame) : ℕ :=
  game.rings_per_game * game.blue_points

/-- Calculates the current total points -/
def current_total_points (game : CarnivalGame) : ℕ :=
  game.current_red * game.red_points +
  game.current_green * game.green_points +
  game.current_blue * game.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 53 -/
theorem max_total_points (game : CarnivalGame) :
  current_total_points game + max_points_per_game game = 53 :=
sorry

end NUMINAMATH_CALUDE_max_total_points_l3303_330318


namespace NUMINAMATH_CALUDE_grape_price_calculation_l3303_330370

theorem grape_price_calculation (G : ℚ) : 
  (11 * G + 7 * 50 = 1428) → G = 98 := by
  sorry

end NUMINAMATH_CALUDE_grape_price_calculation_l3303_330370


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l3303_330355

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧  -- Largest is 90
  (a + b + c + d) / 4 = 70  -- Average is 70
  → a ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l3303_330355


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l3303_330386

theorem crayons_in_drawer (initial_crayons : ℕ) :
  (initial_crayons + 3 = 10) → initial_crayons = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l3303_330386


namespace NUMINAMATH_CALUDE_cos_negative_1320_degrees_l3303_330356

theorem cos_negative_1320_degrees : Real.cos (-(1320 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1320_degrees_l3303_330356


namespace NUMINAMATH_CALUDE_complex_product_example_l3303_330330

-- Define the complex numbers
def z₁ : ℂ := 3 + 4 * Complex.I
def z₂ : ℂ := -2 - 3 * Complex.I

-- State the theorem
theorem complex_product_example : z₁ * z₂ = -18 - 17 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_example_l3303_330330


namespace NUMINAMATH_CALUDE_sample_size_of_500_selection_l3303_330364

/-- Represents a batch of CDs -/
structure CDBatch where
  size : ℕ

/-- Represents a sample of CDs -/
structure CDSample where
  size : ℕ
  source : CDBatch

/-- Defines a random selection of CDs from a batch -/
def randomSelection (batch : CDBatch) (n : ℕ) : CDSample :=
  { size := n
    source := batch }

/-- Theorem stating that the sample size of a random selection of 500 CDs is 500 -/
theorem sample_size_of_500_selection (batch : CDBatch) :
  (randomSelection batch 500).size = 500 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_of_500_selection_l3303_330364


namespace NUMINAMATH_CALUDE_fourteenth_root_unity_l3303_330306

theorem fourteenth_root_unity : ∃ n : ℤ, 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * Real.pi / 14)) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_unity_l3303_330306


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l3303_330352

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt (x^2 + y^2) + x/2 = x + y) → (min x y)/(max x y) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l3303_330352


namespace NUMINAMATH_CALUDE_g_composition_fixed_points_l3303_330389

def g (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem g_composition_fixed_points :
  {x : ℝ | g (g x) = g x} =
  {x : ℝ | x = 2 + Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 + Real.sqrt ((11 - 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 - 2*Real.sqrt 21)/2)} :=
by sorry

end NUMINAMATH_CALUDE_g_composition_fixed_points_l3303_330389


namespace NUMINAMATH_CALUDE_complex_number_solution_l3303_330378

theorem complex_number_solution (z : ℂ) : 
  Complex.abs z = Real.sqrt 13 ∧ 
  ∃ (k : ℝ), (2 + 3*I)*z*I = k*I → 
  z = 3 + 2*I ∨ z = -3 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l3303_330378


namespace NUMINAMATH_CALUDE_student_ability_theorem_l3303_330358

-- Define the function
def f (x : ℝ) : ℝ := -0.1 * x^2 + 2.6 * x + 43

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 30 }

theorem student_ability_theorem :
  (∀ x ∈ domain, x ≤ 13 → ∀ y ∈ domain, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ domain, x ≥ 13 → ∀ y ∈ domain, x ≤ y → f x ≥ f y) ∧
  f 10 = 59 ∧
  (∀ x ∈ domain, f x ≤ f 13) := by
  sorry

end NUMINAMATH_CALUDE_student_ability_theorem_l3303_330358


namespace NUMINAMATH_CALUDE_count_specific_coin_toss_sequences_l3303_330344

def coin_toss_sequences (n : ℕ) (th ht tt hh : ℕ) : ℕ :=
  Nat.choose 4 2 * Nat.choose 8 3 * Nat.choose 5 4 * Nat.choose 11 5

theorem count_specific_coin_toss_sequences :
  coin_toss_sequences 15 2 3 4 5 = 775360 := by
  sorry

end NUMINAMATH_CALUDE_count_specific_coin_toss_sequences_l3303_330344


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3303_330316

theorem simplify_sqrt_sum : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3303_330316


namespace NUMINAMATH_CALUDE_f_5_equals_142_l3303_330322

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- Theorem statement
theorem f_5_equals_142 :
  ∃ y : ℝ, f 2 y = 100 → f 5 y = 142 :=
by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_142_l3303_330322


namespace NUMINAMATH_CALUDE_derek_journey_l3303_330326

/-- Proves that given a journey where half the distance is traveled at 20 km/h 
    and the other half at 4 km/h, with a total travel time of 54 minutes, 
    the distance walked is 3.0 km. -/
theorem derek_journey (total_distance : ℝ) (total_time : ℝ) : 
  (total_distance / 2) / 20 + (total_distance / 2) / 4 = total_time ∧
  total_time = 54 / 60 →
  total_distance / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_derek_journey_l3303_330326


namespace NUMINAMATH_CALUDE_no_valid_polygon_pairs_l3303_330303

theorem no_valid_polygon_pairs : ¬∃ (r k : ℕ), 
  r > 2 ∧ k > 2 ∧ 
  (180 * r - 360) / (180 * k - 360) = 7 / 5 ∧
  ∃ (c : ℚ), c * r = k := by
  sorry

end NUMINAMATH_CALUDE_no_valid_polygon_pairs_l3303_330303


namespace NUMINAMATH_CALUDE_profit_share_b_is_1800_l3303_330399

/-- Represents the profit share calculation for a business partnership --/
def ProfitShare (investment_a investment_b investment_c : ℕ) (profit_diff_ac : ℕ) : ℕ :=
  let ratio_sum := (investment_a / 2000) + (investment_b / 2000) + (investment_c / 2000)
  let part_value := profit_diff_ac / ((investment_c / 2000) - (investment_a / 2000))
  (investment_b / 2000) * part_value

/-- Theorem stating that given the investments and profit difference, 
    the profit share of b is 1800 --/
theorem profit_share_b_is_1800 :
  ProfitShare 8000 10000 12000 720 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_b_is_1800_l3303_330399


namespace NUMINAMATH_CALUDE_definite_integral_quarter_circle_l3303_330388

theorem definite_integral_quarter_circle (f : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..(Real.sqrt 2), f x) = π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_definite_integral_quarter_circle_l3303_330388


namespace NUMINAMATH_CALUDE_car_speed_calculation_l3303_330396

/-- Calculates the car speed given train and car travel information -/
theorem car_speed_calculation (train_speed : ℝ) (train_time : ℝ) (remaining_distance : ℝ) (car_time : ℝ) :
  train_speed = 120 →
  train_time = 2 →
  remaining_distance = 2.4 →
  car_time = 3 →
  (train_speed * train_time + remaining_distance) / car_time = 80.8 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_calculation_l3303_330396


namespace NUMINAMATH_CALUDE_chris_age_l3303_330301

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 10 →
  c - 5 = 2 * a →
  b + 4 = 3/4 * (a + 4) →
  c = 283/15 :=
by sorry

end NUMINAMATH_CALUDE_chris_age_l3303_330301


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3303_330377

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define what it means for a point to be in the fourth quadrant
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- The point we want to prove is in the fourth quadrant
def point_to_check : Point := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : 
  is_in_fourth_quadrant point_to_check := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3303_330377


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3303_330392

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is equal to 64 + 48π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π := by
sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3303_330392


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l3303_330357

theorem largest_stamps_per_page (book1_stamps : ℕ) (book2_stamps : ℕ) :
  book1_stamps = 924 →
  book2_stamps = 1200 →
  ∃ (stamps_per_page : ℕ),
    stamps_per_page = Nat.gcd book1_stamps book2_stamps ∧
    stamps_per_page ≤ book1_stamps ∧
    stamps_per_page ≤ book2_stamps ∧
    ∀ (n : ℕ), n ∣ book1_stamps ∧ n ∣ book2_stamps → n ≤ stamps_per_page :=
by sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l3303_330357


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3303_330393

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2, 5, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3303_330393


namespace NUMINAMATH_CALUDE_hypotenuse_angle_is_45_degrees_l3303_330379

/-- A right triangle with perimeter 2 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq_two : a + b + c = 2
  pythagorean : a^2 + b^2 = c^2

/-- Point on the internal angle bisector of the right angle -/
structure BisectorPoint (t : RightTriangle) where
  distance_sqrt_two : ℝ
  is_sqrt_two : distance_sqrt_two = Real.sqrt 2

/-- The angle subtended by the hypotenuse from the bisector point -/
def hypotenuse_angle (t : RightTriangle) (p : BisectorPoint t) : ℝ := sorry

theorem hypotenuse_angle_is_45_degrees (t : RightTriangle) (p : BisectorPoint t) :
  hypotenuse_angle t p = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_angle_is_45_degrees_l3303_330379


namespace NUMINAMATH_CALUDE_rectangle_squares_sides_l3303_330313

/-- Represents the side lengths of squares in a rectangle divided into 6 squares. -/
structure SquareSides where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ
  s5 : ℝ
  s6 : ℝ

/-- Given a rectangle divided into 6 squares with specific conditions,
    proves that the side lengths of the squares are as calculated. -/
theorem rectangle_squares_sides (sides : SquareSides) 
    (h1 : sides.s1 = 18)
    (h2 : sides.s2 = 3) :
    sides.s3 = 15 ∧ 
    sides.s4 = 12 ∧ 
    sides.s5 = 12 ∧ 
    sides.s6 = 21 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_squares_sides_l3303_330313


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l3303_330380

def appetizer_cost : ℚ := 8
def entree_cost : ℚ := 20
def wine_cost : ℚ := 3
def dessert_cost : ℚ := 6
def discount_ratio : ℚ := (1/2)
def total_spent : ℚ := 38

def full_cost : ℚ := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
def discounted_cost : ℚ := appetizer_cost + entree_cost * (1 - discount_ratio) + 2 * wine_cost + dessert_cost
def tip_amount : ℚ := total_spent - discounted_cost

theorem tip_percentage_is_twenty_percent :
  (tip_amount / full_cost) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l3303_330380


namespace NUMINAMATH_CALUDE_line_through_parabola_focus_l3303_330373

/-- The focus of a parabola y² = 4x is the point (1, 0) -/
def focus_of_parabola : ℝ × ℝ := (1, 0)

/-- A line passing through a point (x, y) is represented by the equation ax - y + 1 = 0 -/
def line_passes_through (a : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 - p.2 + 1 = 0

theorem line_through_parabola_focus (a : ℝ) :
  line_passes_through a focus_of_parabola → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_focus_l3303_330373


namespace NUMINAMATH_CALUDE_value_of_a_l3303_330394

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3303_330394


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3303_330345

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ p x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ p x) := by
  sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 2 * x = 3

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ proposition x) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3303_330345


namespace NUMINAMATH_CALUDE_train_passing_time_l3303_330315

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 150 →
  train_speed_kmh = 36 →
  passing_time = 15 →
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3303_330315


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_at_least_one_true_l3303_330332

theorem not_p_or_q_false_implies_at_least_one_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_at_least_one_true_l3303_330332


namespace NUMINAMATH_CALUDE_divisible_by_120_l3303_330359

theorem divisible_by_120 (m : ℕ) : ∃ k : ℤ, (m ^ 5 : ℤ) - 5 * (m ^ 3) + 4 * m = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l3303_330359


namespace NUMINAMATH_CALUDE_workshop_wolf_prize_laureates_l3303_330361

theorem workshop_wolf_prize_laureates 
  (total_scientists : ℕ) 
  (both_wolf_and_nobel : ℕ) 
  (total_nobel : ℕ) 
  (h1 : total_scientists = 50)
  (h2 : both_wolf_and_nobel = 12)
  (h3 : total_nobel = 23)
  (h4 : ∃ (non_wolf_non_nobel : ℕ), 
        non_wolf_non_nobel + (non_wolf_non_nobel + 3) = total_scientists - both_wolf_and_nobel) :
  ∃ (wolf_laureates : ℕ), wolf_laureates = 31 ∧ 
    wolf_laureates + (total_scientists - wolf_laureates) = total_scientists :=
sorry

end NUMINAMATH_CALUDE_workshop_wolf_prize_laureates_l3303_330361


namespace NUMINAMATH_CALUDE_point_on_line_ratio_l3303_330324

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and a point P between B and C satisfying a ratio condition,
    prove that OP has the given value. -/
theorem point_on_line_ratio (a b c d k : ℝ) :
  let OA := a
  let OB := k * b
  let OC := c
  let OD := k * d
  ∀ P : ℝ, OB ≤ P ∧ P ≤ OC →
  (a - P) / (P - k * d) = k * (k * b - P) / (P - c) →
  P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_ratio_l3303_330324


namespace NUMINAMATH_CALUDE_stepa_multiplication_l3303_330302

theorem stepa_multiplication (sequence : Fin 5 → ℕ) 
  (h1 : ∀ i : Fin 4, sequence (i.succ) = (3 * sequence i) / 2)
  (h2 : sequence 4 = 81) :
  ∃ (a b : ℕ), a * b = sequence 3 ∧ a = 6 ∧ b = 9 :=
by sorry

end NUMINAMATH_CALUDE_stepa_multiplication_l3303_330302


namespace NUMINAMATH_CALUDE_parrot_silence_explanation_l3303_330300

-- Define the parrot type
structure Parrot where
  repeats_heard_words : Bool
  is_silent : Bool

-- Define the environment
structure Environment where
  words_spoken : Bool

-- Define the theorem
theorem parrot_silence_explanation (p : Parrot) (e : Environment) :
  p.repeats_heard_words ∧ p.is_silent →
  (¬e.words_spoken ∨ ¬p.repeats_heard_words) :=
by
  sorry

-- The negation of repeats_heard_words represents deafness

end NUMINAMATH_CALUDE_parrot_silence_explanation_l3303_330300


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l3303_330338

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 35 →
    2 * chickens + 4 * rabbits = 94 →
    chickens = 23 ∧ rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l3303_330338


namespace NUMINAMATH_CALUDE_systematic_sampling_questionnaire_B_l3303_330384

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_B 
  (total_pool : ℕ) 
  (sample_size : ℕ) 
  (first_selected : ℕ) 
  (questionnaire_B_start : ℕ) 
  (questionnaire_B_end : ℕ) : 
  total_pool = 960 → 
  sample_size = 32 → 
  first_selected = 9 → 
  questionnaire_B_start = 461 → 
  questionnaire_B_end = 761 → 
  (Finset.filter (fun n => 
    questionnaire_B_start ≤ (first_selected + (n - 1) * (total_pool / sample_size)) ∧ 
    (first_selected + (n - 1) * (total_pool / sample_size)) ≤ questionnaire_B_end
  ) (Finset.range (sample_size + 1))).card = 10 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_questionnaire_B_l3303_330384


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3303_330319

theorem quadratic_equation_proof (x₁ x₂ k : ℝ) : 
  (x₁^2 - 6*x₁ + k = 0) →
  (x₂^2 - 6*x₂ + k = 0) →
  (x₁^2 * x₂^2 - x₁ - x₂ = 115) →
  (k = -11 ∧ ((x₁ = 3 + 2*Real.sqrt 5 ∧ x₂ = 3 - 2*Real.sqrt 5) ∨ 
              (x₁ = 3 - 2*Real.sqrt 5 ∧ x₂ = 3 + 2*Real.sqrt 5))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3303_330319


namespace NUMINAMATH_CALUDE_total_timeout_time_total_timeout_is_185_minutes_l3303_330321

/-- Calculates the total time students spend in time-out given the number of time-outs for different offenses and the duration of each time-out. -/
theorem total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_timeouts / 3
  let total_timeouts := running_timeouts + food_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time students spend in time-out is 185 minutes under the given conditions. -/
theorem total_timeout_is_185_minutes : total_timeout_time 5 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_timeout_time_total_timeout_is_185_minutes_l3303_330321


namespace NUMINAMATH_CALUDE_flowerbed_fraction_is_one_eighth_l3303_330327

/-- Represents a rectangular park with flower beds -/
structure Park where
  /-- Length of the shorter parallel side of the trapezoidal area -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal area -/
  long_side : ℝ
  /-- Number of congruent isosceles right triangle flower beds -/
  num_flowerbeds : ℕ

/-- The fraction of the park occupied by flower beds -/
def flowerbed_fraction (p : Park) : ℝ :=
  -- Define the fraction calculation here
  sorry

/-- Theorem stating that for a park with specific dimensions, 
    the fraction of area occupied by flower beds is 1/8 -/
theorem flowerbed_fraction_is_one_eighth :
  ∀ (p : Park), 
  p.short_side = 30 ∧ 
  p.long_side = 50 ∧ 
  p.num_flowerbeds = 3 →
  flowerbed_fraction p = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_flowerbed_fraction_is_one_eighth_l3303_330327


namespace NUMINAMATH_CALUDE_area_of_square_on_XY_l3303_330367

-- Define the triangle XYZ
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2

-- Define the theorem
theorem area_of_square_on_XY (t : RightTriangle) 
  (sum_of_squares : t.XY^2 + t.YZ^2 + t.XZ^2 = 500) : 
  t.XY^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_on_XY_l3303_330367


namespace NUMINAMATH_CALUDE_tonys_monthly_rent_l3303_330333

/-- Calculates the monthly rent for a cottage given its room sizes and cost per square foot. -/
def calculate_monthly_rent (master_area : ℕ) (guest_bedroom_area : ℕ) (num_guest_bedrooms : ℕ) (other_areas : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  let total_area := master_area + (guest_bedroom_area * num_guest_bedrooms) + other_areas
  total_area * cost_per_sqft

/-- Theorem stating that Tony's monthly rent is $3000 given the specified conditions. -/
theorem tonys_monthly_rent : 
  calculate_monthly_rent 500 200 2 600 2 = 3000 := by
  sorry

#eval calculate_monthly_rent 500 200 2 600 2

end NUMINAMATH_CALUDE_tonys_monthly_rent_l3303_330333


namespace NUMINAMATH_CALUDE_reflection_sequence_exists_l3303_330391

/-- Definition of a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of a triangle using three points -/
structure Triangle :=
  (p1 : Point)
  (p2 : Point)
  (p3 : Point)

/-- Definition of a reflection line -/
inductive ReflectionLine
  | AB
  | BC
  | CA

/-- A sequence of reflections -/
def ReflectionSequence := List ReflectionLine

/-- Apply a single reflection to a point -/
def reflect (p : Point) (line : ReflectionLine) : Point :=
  match line with
  | ReflectionLine.AB => ⟨p.x, -p.y⟩
  | ReflectionLine.BC => ⟨3 - p.y, 3 - p.x⟩
  | ReflectionLine.CA => ⟨-p.x, p.y⟩

/-- Apply a sequence of reflections to a point -/
def applyReflections (p : Point) (seq : ReflectionSequence) : Point :=
  seq.foldl reflect p

/-- Apply a sequence of reflections to a triangle -/
def reflectTriangle (t : Triangle) (seq : ReflectionSequence) : Triangle :=
  ⟨applyReflections t.p1 seq, applyReflections t.p2 seq, applyReflections t.p3 seq⟩

/-- The original triangle -/
def originalTriangle : Triangle :=
  ⟨⟨0, 0⟩, ⟨0, 1⟩, ⟨2, 0⟩⟩

/-- The target triangle -/
def targetTriangle : Triangle :=
  ⟨⟨24, 36⟩, ⟨24, 37⟩, ⟨26, 36⟩⟩

theorem reflection_sequence_exists : ∃ (seq : ReflectionSequence), reflectTriangle originalTriangle seq = targetTriangle := by
  sorry

end NUMINAMATH_CALUDE_reflection_sequence_exists_l3303_330391


namespace NUMINAMATH_CALUDE_three_red_faces_count_total_cubes_count_l3303_330366

/-- Represents a rectangular solid composed of small cubes -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of corner cubes in a rectangular solid -/
def cornerCubes (solid : RectangularSolid) : ℕ :=
  8

/-- Theorem: In a 5 × 4 × 2 rectangular solid with painted outer surface,
    the number of small cubes with exactly 3 red faces is 8 -/
theorem three_red_faces_count :
  let solid : RectangularSolid := ⟨5, 4, 2⟩
  cornerCubes solid = 8 := by
  sorry

/-- Verifies that the total number of cubes is 40 -/
theorem total_cubes_count :
  let solid : RectangularSolid := ⟨5, 4, 2⟩
  solid.length * solid.width * solid.height = 40 := by
  sorry

end NUMINAMATH_CALUDE_three_red_faces_count_total_cubes_count_l3303_330366


namespace NUMINAMATH_CALUDE_brothers_age_ratio_l3303_330311

/-- Represents the ages of three brothers: Richard, David, and Scott -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the ages of the brothers after a given number of years -/
def agesAfterYears (ages : BrothersAges) (years : ℕ) : BrothersAges :=
  { david := ages.david + years
  , richard := ages.richard + years
  , scott := ages.scott + years }

/-- The theorem statement based on the given problem -/
theorem brothers_age_ratio : ∀ (ages : BrothersAges),
  ages.richard = ages.david + 6 →
  ages.david = ages.scott + 8 →
  ages.david = 14 →
  ∃ (k : ℕ), (agesAfterYears ages 8).richard = k * (agesAfterYears ages 8).scott →
  (agesAfterYears ages 8).richard / (agesAfterYears ages 8).scott = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_ratio_l3303_330311


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l3303_330346

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l3303_330346


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3303_330305

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  3 * a + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b))))

-- Theorem statement
theorem bowtie_equation_solution (y : ℝ) : bowtie 5 y = 20 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3303_330305


namespace NUMINAMATH_CALUDE_danny_soda_distribution_l3303_330341

theorem danny_soda_distribution (initial_bottles : ℝ) (drunk_percentage : ℝ) (remaining_percentage : ℝ) : 
  initial_bottles = 3 →
  drunk_percentage = 90 →
  remaining_percentage = 70 →
  let drunk_amount := (drunk_percentage / 100) * 1
  let remaining_amount := (remaining_percentage / 100) * 1
  let given_away := initial_bottles - (drunk_amount + remaining_amount)
  given_away / 2 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_danny_soda_distribution_l3303_330341


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l3303_330381

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l3303_330381


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3303_330353

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3303_330353


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l3303_330365

theorem arithmetic_progression_problem (a d : ℝ) : 
  -- The five numbers form a decreasing arithmetic progression
  (∀ i : Fin 5, (fun i => a - (2 - i) * d) i > (fun i => a - (2 - i.succ) * d) i.succ) →
  -- The sum of their cubes is zero
  ((a - 2*d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2*d)^3 = 0) →
  -- The sum of their fourth powers is 136
  ((a - 2*d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2*d)^4 = 136) →
  -- The smallest number is -2√2
  a - 2*d = -2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_progression_problem_l3303_330365


namespace NUMINAMATH_CALUDE_right_triangle_with_three_isosceles_l3303_330331

/-- A right-angled triangle that can be divided into three isosceles triangles has acute angles of 22.5° and 67.5°. -/
theorem right_triangle_with_three_isosceles (α β : Real) : 
  α + β = 90 → -- The sum of acute angles in a right triangle is 90°
  (∃ (γ : Real), γ = 90 ∧ 2*α + 2*α = γ) → -- One of the isosceles triangles has a right angle and two equal angles of 2α
  (α = 22.5 ∧ β = 67.5) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_three_isosceles_l3303_330331


namespace NUMINAMATH_CALUDE_base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l3303_330385

/-- Converts a base 7 number of the form 52x4 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The digit x in 52x4₇ makes the number divisible by 19 --/
theorem base7_divisible_by_19 :
  ∃ x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is unique --/
theorem unique_x_divisible_by_19 :
  ∃! x : ℕ, x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

/-- The digit x in 52x4₇ that makes the number divisible by 19 is 4 --/
theorem x_is_4 :
  ∃ x : ℕ, x = 4 ∧ x < 7 ∧ isDivisibleBy19 (base7ToDecimal x) :=
sorry

end NUMINAMATH_CALUDE_base7_divisible_by_19_unique_x_divisible_by_19_x_is_4_l3303_330385


namespace NUMINAMATH_CALUDE_classroom_to_total_ratio_is_one_to_four_l3303_330390

/-- Given a class of students with some on the playground and some in the classroom,
    prove that the ratio of students in the classroom to total students is 1:4. -/
theorem classroom_to_total_ratio_is_one_to_four
  (total_students : ℕ)
  (playground_students : ℕ)
  (classroom_students : ℕ)
  (playground_girls : ℕ)
  (h1 : total_students = 20)
  (h2 : total_students = playground_students + classroom_students)
  (h3 : playground_girls = 10)
  (h4 : playground_girls = (2 : ℚ) / 3 * playground_students) :
  (classroom_students : ℚ) / total_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_to_total_ratio_is_one_to_four_l3303_330390


namespace NUMINAMATH_CALUDE_unique_tangent_circle_l3303_330397

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Theorem: There exists exactly one circle of radius 4 that is tangent to two circles of radius 2
    which are tangent to each other, at their point of tangency -/
theorem unique_tangent_circle (c1 c2 : Circle) : 
  c1.radius = 2 → 
  c2.radius = 2 → 
  are_tangent c1 c2 → 
  ∃! c : Circle, c.radius = 4 ∧ are_tangent c c1 ∧ are_tangent c c2 :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_circle_l3303_330397


namespace NUMINAMATH_CALUDE_optimal_distribution_second_day_distribution_l3303_330340

/-- Represents a production line with its processing characteristics -/
structure ProductionLine where
  name : String
  process_time : ℝ → ℝ
  tonnage : ℝ

/-- The company with two production lines -/
structure Company where
  line_a : ProductionLine
  line_b : ProductionLine

/-- Defines the company with given production line characteristics -/
def our_company : Company :=
  { line_a := { name := "A", process_time := (λ a ↦ 4 * a + 1), tonnage := 0 },
    line_b := { name := "B", process_time := (λ b ↦ 2 * b + 3), tonnage := 0 } }

/-- Total raw materials allocated to both production lines -/
def total_raw_materials : ℝ := 5

/-- Theorem stating the optimal distribution of raw materials -/
theorem optimal_distribution (c : Company) (h : c = our_company) :
  ∃ (a b : ℝ),
    a + b = total_raw_materials ∧
    c.line_a.process_time a = c.line_b.process_time b ∧
    a = 2 ∧ b = 3 := by
  sorry

/-- Theorem stating the relationship between m and n for the second day -/
theorem second_day_distribution (m n : ℝ) (h : m + n = 6) :
  2 * m = n → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_optimal_distribution_second_day_distribution_l3303_330340


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l3303_330372

theorem fraction_multiplication_addition : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 6 : ℚ) = 13 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l3303_330372


namespace NUMINAMATH_CALUDE_company_bonus_problem_l3303_330376

/-- Represents the company bonus distribution problem -/
theorem company_bonus_problem (n : ℕ) 
  (h1 : 60 * n - 15 = 45 * n + 135) : 
  60 * n - 15 = 585 := by
  sorry

#check company_bonus_problem

end NUMINAMATH_CALUDE_company_bonus_problem_l3303_330376


namespace NUMINAMATH_CALUDE_square_sum_product_l3303_330347

theorem square_sum_product (a b : ℝ) (h1 : a + b = -3) (h2 : a * b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3303_330347


namespace NUMINAMATH_CALUDE_mean_temperature_l3303_330308

def temperatures : List ℤ := [-8, -5, -3, 0, 4, 2, 7]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -3/7 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l3303_330308


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3303_330398

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 / 3) :
  (a + 1)^2 + a * (1 - a) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3303_330398


namespace NUMINAMATH_CALUDE_borrowed_sum_proof_l3303_330317

/-- 
Given a principal P borrowed at 8% per annum simple interest for 8 years,
if the interest I is equal to P - 900, then P = 2500.
-/
theorem borrowed_sum_proof (P : ℝ) (I : ℝ) : 
  (I = P * 8 * 8 / 100) →   -- Simple interest formula
  (I = P - 900) →           -- Given condition
  P = 2500 := by
sorry

end NUMINAMATH_CALUDE_borrowed_sum_proof_l3303_330317


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_l3303_330349

theorem sqrt_eight_plus_sqrt_two : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_l3303_330349


namespace NUMINAMATH_CALUDE_function_always_positive_implies_a_less_than_one_l3303_330320

theorem function_always_positive_implies_a_less_than_one :
  (∀ x : ℝ, |x - 1| + |x - 2| - a > 0) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_implies_a_less_than_one_l3303_330320


namespace NUMINAMATH_CALUDE_marias_gum_l3303_330363

/-- Represents the number of pieces of gum Maria has -/
def total_gum (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ := initial + x + y

/-- Theorem stating the total number of pieces of gum Maria has -/
theorem marias_gum (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) :
  total_gum 58 x y = 58 + x + y := by sorry

end NUMINAMATH_CALUDE_marias_gum_l3303_330363


namespace NUMINAMATH_CALUDE_quadratic_incenter_theorem_l3303_330350

/-- A quadratic function that intersects the coordinate axes at three points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- A triangle formed by the intersection points of a quadratic function with the coordinate axes -/
structure IntersectionTriangle where
  quad : QuadraticFunction
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle -/
def incenter (t : IntersectionTriangle) : ℝ × ℝ := sorry

/-- The theorem statement -/
theorem quadratic_incenter_theorem (t : IntersectionTriangle) 
  (h1 : t.A.2 = 0 ∨ t.A.1 = 0)
  (h2 : t.B.2 = 0 ∨ t.B.1 = 0)
  (h3 : t.C.2 = 0 ∨ t.C.1 = 0)
  (h4 : t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.A ≠ t.C)
  (h5 : ∃ (x : ℝ), incenter t = (x, x)) :
  t.quad.a + t.quad.b + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_incenter_theorem_l3303_330350


namespace NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l3303_330312

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that when x = 1 for y = 3 and z = 2, then x = 8/9 when y = 9 and z = 4. -/
theorem inverse_square_direct_cube_relation (k : ℚ) :
  (1 : ℚ) = k * (2^3 : ℚ) / (3^2 : ℚ) →
  (8/9 : ℚ) = k * (4^3 : ℚ) / (9^2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l3303_330312


namespace NUMINAMATH_CALUDE_max_toads_in_two_ponds_l3303_330375

/-- Represents a pond with frogs and toads -/
structure Pond where
  frogRatio : ℕ
  toadRatio : ℕ

/-- The maximum number of toads given two ponds and a total number of frogs -/
def maxToads (pond1 pond2 : Pond) (totalFrogs : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of toads in the given scenario -/
theorem max_toads_in_two_ponds :
  let pond1 : Pond := { frogRatio := 3, toadRatio := 4 }
  let pond2 : Pond := { frogRatio := 5, toadRatio := 6 }
  let totalFrogs : ℕ := 36
  maxToads pond1 pond2 totalFrogs = 46 := by
  sorry

end NUMINAMATH_CALUDE_max_toads_in_two_ponds_l3303_330375


namespace NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2151_l3303_330354

/-- The count of positive integers less than 100,000 with at most two different digits -/
def count_numbers_with_at_most_two_digits : ℕ :=
  let max_number := 100000
  let single_digit_count := 9 * 5
  let two_digits_without_zero := 36 * (2^2 - 2 + 2^3 - 2 + 2^4 - 2 + 2^5 - 2)
  let two_digits_with_zero := 9 * (2^1 - 1 + 2^2 - 1 + 2^3 - 1 + 2^4 - 1)
  single_digit_count + two_digits_without_zero + two_digits_with_zero

theorem count_numbers_with_at_most_two_digits_is_2151 :
  count_numbers_with_at_most_two_digits = 2151 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2151_l3303_330354


namespace NUMINAMATH_CALUDE_equal_goldfish_after_six_months_l3303_330387

/-- Number of goldfish Brent has after n months -/
def brent_goldfish (n : ℕ) : ℕ := 2 * 4^n

/-- Number of goldfish Gretel has after n months -/
def gretel_goldfish (n : ℕ) : ℕ := 162 * 3^n

/-- The number of months it takes for Brent and Gretel to have the same number of goldfish -/
def months_to_equal_goldfish : ℕ := 6

/-- Theorem stating that after 'months_to_equal_goldfish' months, 
    Brent and Gretel have the same number of goldfish -/
theorem equal_goldfish_after_six_months : 
  brent_goldfish months_to_equal_goldfish = gretel_goldfish months_to_equal_goldfish :=
by sorry

end NUMINAMATH_CALUDE_equal_goldfish_after_six_months_l3303_330387


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l3303_330368

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/(2*y) = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/(2*w) = 1 → x + 4*y ≤ z + 4*w ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l3303_330368


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l3303_330329

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem smallest_integer_with_divisibility_condition :
  ∃ (n : ℕ) (i j : ℕ),
    n > 0 ∧
    i < j ∧
    j - i = 1 ∧
    j ≤ 30 ∧
    (∀ k : ℕ, k ≤ 30 → k ≠ i → k ≠ j → is_divisible n k) ∧
    ¬(is_divisible n i) ∧
    ¬(is_divisible n j) ∧
    (∀ m : ℕ, m > 0 →
      (∃ (x y : ℕ), x < y ∧ y - x = 1 ∧ y ≤ 30 ∧
        (∀ k : ℕ, k ≤ 30 → k ≠ x → k ≠ y → is_divisible m k) ∧
        ¬(is_divisible m x) ∧
        ¬(is_divisible m y)) →
      m ≥ n) ∧
    n = 2230928700 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l3303_330329


namespace NUMINAMATH_CALUDE_S_min_value_l3303_330334

/-- The function S defined on real numbers x and y -/
def S (x y : ℝ) : ℝ := 2 * x^2 - x*y + y^2 + 2*x + 3*y

/-- Theorem stating that S has a minimum value of -4 -/
theorem S_min_value :
  (∀ x y : ℝ, S x y ≥ -4) ∧ (∃ x y : ℝ, S x y = -4) :=
sorry

end NUMINAMATH_CALUDE_S_min_value_l3303_330334


namespace NUMINAMATH_CALUDE_garage_sale_theorem_l3303_330360

def garage_sale_problem (treadmill_price : ℝ) (chest_price : ℝ) (tv_price : ℝ) (total_sales : ℝ) : Prop :=
  treadmill_price = 100 ∧
  chest_price = treadmill_price / 2 ∧
  tv_price = 3 * treadmill_price ∧
  (treadmill_price + chest_price + tv_price) / total_sales = 0.75 ∧
  total_sales = 600

theorem garage_sale_theorem :
  ∃ (treadmill_price chest_price tv_price total_sales : ℝ),
    garage_sale_problem treadmill_price chest_price tv_price total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_garage_sale_theorem_l3303_330360


namespace NUMINAMATH_CALUDE_equation_solution_l3303_330351

theorem equation_solution : ∃ y : ℝ, (32 : ℝ) ^ (3 * y) = 8 ^ (2 * y + 1) ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3303_330351


namespace NUMINAMATH_CALUDE_dinner_slices_l3303_330304

/-- Represents the number of slices of pie served during different times of the day -/
structure PieSlices where
  lunch : ℕ
  total : ℕ

/-- Proves that the number of slices served during dinner is 5,
    given 7 slices were served during lunch and 12 slices in total -/
theorem dinner_slices (ps : PieSlices) 
  (h_lunch : ps.lunch = 7) 
  (h_total : ps.total = 12) : 
  ps.total - ps.lunch = 5 := by
  sorry

end NUMINAMATH_CALUDE_dinner_slices_l3303_330304


namespace NUMINAMATH_CALUDE_zebras_permutations_l3303_330336

theorem zebras_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_zebras_permutations_l3303_330336


namespace NUMINAMATH_CALUDE_bus_problem_l3303_330323

/-- Proof of the number of people who got off at the first bus stop -/
theorem bus_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_boarding : Nat) 
  (second_stop_boarding : Nat) (second_stop_departing : Nat) 
  (empty_seats_after_second : Nat) : 
  total_rows = 23 → 
  seats_per_row = 4 → 
  initial_boarding = 16 → 
  first_stop_boarding = 15 → 
  second_stop_boarding = 17 → 
  second_stop_departing = 10 → 
  empty_seats_after_second = 57 → 
  ∃ (first_stop_departing : Nat), 
    first_stop_departing = 3 ∧
    (total_rows * seats_per_row) - 
    (initial_boarding + first_stop_boarding + second_stop_boarding - 
     first_stop_departing - second_stop_departing) = 
    empty_seats_after_second :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l3303_330323


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l3303_330328

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β γ : Plane) (m l : Line) 
  (h1 : perp α γ)
  (h2 : intersect γ α = m)
  (h3 : intersect γ β = l)
  (h4 : perp_line l m) :
  perp_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l3303_330328


namespace NUMINAMATH_CALUDE_even_function_extension_l3303_330369

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x - x^4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l3303_330369


namespace NUMINAMATH_CALUDE_triangle_similarity_implies_pc_length_l3303_330343

/-- Triangle ABC with sides AB, BC, and CA -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Point P on the extension of BC -/
def P : Type := Unit

/-- The length of PC -/
def PC (t : Triangle) (p : P) : ℝ := sorry

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : Triangle) (p : P) : Prop := sorry

theorem triangle_similarity_implies_pc_length 
  (t : Triangle) 
  (p : P) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t p) : 
  PC t p = 1.5 := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_implies_pc_length_l3303_330343


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3303_330362

theorem rectangle_area_proof (large_square_side : ℝ) 
  (rectangle_length rectangle_width : ℝ) 
  (small_square_side : ℝ) :
  large_square_side = 4 →
  rectangle_length = 1 →
  rectangle_width = 4 →
  small_square_side = 2 →
  large_square_side^2 - (rectangle_length * rectangle_width + small_square_side^2) = 8 :=
by
  sorry

#check rectangle_area_proof

end NUMINAMATH_CALUDE_rectangle_area_proof_l3303_330362


namespace NUMINAMATH_CALUDE_salary_after_changes_l3303_330310

-- Define the initial salary
def initial_salary : ℝ := 2000

-- Define the raise percentage
def raise_percentage : ℝ := 0.20

-- Define the pay cut percentage
def pay_cut_percentage : ℝ := 0.20

-- Theorem to prove
theorem salary_after_changes (s : ℝ) (r : ℝ) (c : ℝ) 
  (h1 : s = initial_salary) 
  (h2 : r = raise_percentage) 
  (h3 : c = pay_cut_percentage) : 
  s * (1 + r) * (1 - c) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_changes_l3303_330310
