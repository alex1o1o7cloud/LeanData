import Mathlib

namespace ferris_wheel_seats_l2063_206345

-- Define the total number of people that can ride at once
def total_riders : ℕ := 4

-- Define the capacity of each seat
def seat_capacity : ℕ := 2

-- Define the number of seats on the Ferris wheel
def num_seats : ℕ := total_riders / seat_capacity

-- Theorem statement
theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end ferris_wheel_seats_l2063_206345


namespace greatest_three_digit_multiple_of_19_l2063_206316

theorem greatest_three_digit_multiple_of_19 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 19 ∣ n → n ≤ 988 :=
by
  sorry

end greatest_three_digit_multiple_of_19_l2063_206316


namespace absolute_value_equation_solution_product_l2063_206362

theorem absolute_value_equation_solution_product : 
  (∀ x : ℝ, |x - 5| + 4 = 7 → x = 8 ∨ x = 2) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ - 5| + 4 = 7 ∧ |x₂ - 5| + 4 = 7) ∧
  (∀ x₁ x₂ : ℝ, |x₁ - 5| + 4 = 7 → |x₂ - 5| + 4 = 7 → x₁ * x₂ = 16) := by
sorry

end absolute_value_equation_solution_product_l2063_206362


namespace fraction_equality_l2063_206366

theorem fraction_equality (a b y : ℝ) 
  (h1 : y = (a + 2*b) / a) 
  (h2 : a ≠ -2*b) 
  (h3 : a ≠ 0) : 
  (2*a + 2*b) / (a - 2*b) = (y + 1) / (3 - y) := by
  sorry

end fraction_equality_l2063_206366


namespace total_jelly_beans_l2063_206322

/-- The number of vanilla jelly beans -/
def vanilla : ℕ := 120

/-- The number of grape jelly beans -/
def grape : ℕ := 5 * vanilla + 50

/-- The number of strawberry jelly beans -/
def strawberry : ℕ := (2 * vanilla) / 3

/-- The total number of jelly beans -/
def total : ℕ := grape + vanilla + strawberry

/-- Theorem stating that the total number of jelly beans is 850 -/
theorem total_jelly_beans : total = 850 := by
  sorry

end total_jelly_beans_l2063_206322


namespace wendy_running_distance_l2063_206382

theorem wendy_running_distance (ran walked : ℝ) (h1 : ran = 19.833333333333332) 
  (h2 : walked = 9.166666666666666) : 
  ran - walked = 10.666666666666666 := by
  sorry

end wendy_running_distance_l2063_206382


namespace questions_to_write_l2063_206349

theorem questions_to_write 
  (total_mc : ℕ) (total_ps : ℕ) (total_tf : ℕ)
  (frac_mc : ℚ) (frac_ps : ℚ) (frac_tf : ℚ)
  (h1 : total_mc = 50)
  (h2 : total_ps = 30)
  (h3 : total_tf = 40)
  (h4 : frac_mc = 5/8)
  (h5 : frac_ps = 7/12)
  (h6 : frac_tf = 2/5) :
  ↑total_mc - ⌊frac_mc * total_mc⌋ + 
  ↑total_ps - ⌊frac_ps * total_ps⌋ + 
  ↑total_tf - ⌊frac_tf * total_tf⌋ = 56 := by
sorry

end questions_to_write_l2063_206349


namespace black_balloons_problem_l2063_206304

theorem black_balloons_problem (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end black_balloons_problem_l2063_206304


namespace custom_mult_example_l2063_206338

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (n / q)

/-- Theorem stating that 5/4 * 6/2 = 60 under the custom multiplication -/
theorem custom_mult_example : custom_mult 5 4 6 2 = 60 := by
  sorry

end custom_mult_example_l2063_206338


namespace expression_evaluation_l2063_206399

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 := by
  sorry

end expression_evaluation_l2063_206399


namespace numbers_satisfying_conditions_l2063_206387

def ends_with_196 (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 1000 * x + 196

def decreases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ n / k = n - 196

def satisfies_conditions (n : ℕ) : Prop :=
  ends_with_196 n ∧ decreases_by_integer_factor n

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {1196, 2196, 4196, 7196, 14196, 49196, 98196} :=
by sorry

end numbers_satisfying_conditions_l2063_206387


namespace semicircle_is_arc_l2063_206317

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define an arc
def Arc (c : Circle) := Set (ℝ × ℝ)

-- Define a semicircle
def Semicircle (c : Circle) := Arc c

-- Theorem: A semicircle is an arc
theorem semicircle_is_arc (c : Circle) : Semicircle c → Arc c := by
  sorry

end semicircle_is_arc_l2063_206317


namespace building_height_l2063_206340

theorem building_height :
  let standard_floor_height : ℝ := 3
  let taller_floor_height : ℝ := 3.5
  let num_standard_floors : ℕ := 18
  let num_taller_floors : ℕ := 2
  let total_floors : ℕ := num_standard_floors + num_taller_floors
  total_floors = 20 →
  (num_standard_floors : ℝ) * standard_floor_height + (num_taller_floors : ℝ) * taller_floor_height = 61 := by
  sorry

end building_height_l2063_206340


namespace equation_solution_l2063_206378

theorem equation_solution : ∃ x : ℝ, 64 + x * 12 / (180 / 3) = 65 ∧ x = 5 := by
  sorry

end equation_solution_l2063_206378


namespace scientific_notation_of_five_nm_l2063_206376

theorem scientific_notation_of_five_nm :
  ∃ (a : ℝ) (n : ℤ), 0.000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end scientific_notation_of_five_nm_l2063_206376


namespace simplify_sqrt_difference_l2063_206331

theorem simplify_sqrt_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 528 / Real.sqrt 32) - (Real.sqrt 297 / Real.sqrt 99)) - 2.318| < ε :=
by
  sorry

end simplify_sqrt_difference_l2063_206331


namespace problem_statements_l2063_206353

theorem problem_statements :
  (({0} : Set ℕ) ⊆ Set.univ) ∧
  (∀ (α : Type) (A B : Set α) (x : α), x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ (a b : ℝ), b^2 < a^2 ∧ ¬(a < b ∧ b < 0)) ∧
  (¬(∀ (x : ℤ), x^2 > 0) ↔ ∃ (x : ℤ), x^2 ≤ 0) :=
by sorry

end problem_statements_l2063_206353


namespace lewis_weekly_earnings_l2063_206320

/-- Lewis's earnings during the harvest -/
def total_earnings : ℕ := 1216

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 76

/-- Weekly earnings of Lewis during the harvest -/
def weekly_earnings : ℚ := total_earnings / harvest_duration

theorem lewis_weekly_earnings : weekly_earnings = 16 := by
  sorry

end lewis_weekly_earnings_l2063_206320


namespace four_digit_integer_problem_l2063_206375

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digit_sum (n : ℕ) : ℕ :=
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ :=
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_problem (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 18)
  (h3 : middle_digit_sum n = 11)
  (h4 : thousands_minus_units n = 1)
  (h5 : n % 11 = 0) :
  n = 4653 := by
sorry

end four_digit_integer_problem_l2063_206375


namespace cans_restocked_day2_is_1500_l2063_206324

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans restocked after the second day --/
def cans_restocked_day2 (fb : FoodBank) : ℕ :=
  fb.total_cans_given - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person + fb.day1_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that for the given scenario, the number of cans restocked after the second day is 1500 --/
theorem cans_restocked_day2_is_1500 (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.total_cans_given = 2500) :
  cans_restocked_day2 fb = 1500 := by
  sorry

end cans_restocked_day2_is_1500_l2063_206324


namespace stating_equation_is_quadratic_l2063_206371

/-- 
Theorem stating that when a = 3, the equation 3x^(a-1) - x = 5 is quadratic in x.
-/
theorem equation_is_quadratic (x : ℝ) : 
  let a : ℝ := 3
  let f : ℝ → ℝ := λ x => 3 * x^(a - 1) - x - 5
  ∃ (p q r : ℝ), f x = p * x^2 + q * x + r := by
  sorry

end stating_equation_is_quadratic_l2063_206371


namespace sufficient_but_not_necessary_condition_l2063_206305

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → x > a) ∧
  (∃ x : ℝ, x > a ∧ ¬(x^2 - 2*x - 3 < 0)) →
  a ≤ -1 := by
  sorry

end sufficient_but_not_necessary_condition_l2063_206305


namespace circle_probabilities_l2063_206312

/-- A type representing the 10 equally spaced points on a circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven | eight | nine | ten

/-- Function to check if two points form a diameter -/
def is_diameter (p1 p2 : CirclePoint) : Prop := sorry

/-- Function to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- Function to check if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : CirclePoint) : Prop := sorry

/-- The number of ways to choose n items from a set of 10 -/
def choose_10 (n : Nat) : Nat := sorry

theorem circle_probabilities :
  (∃ (num_diameters : Nat), 
    num_diameters / choose_10 2 = 1 / 9 ∧
    (∀ p1 p2 : CirclePoint, p1 ≠ p2 → 
      (Nat.card {pair | pair = (p1, p2) ∧ is_diameter p1 p2} = num_diameters))) ∧
  (∃ (num_right_triangles : Nat),
    num_right_triangles / choose_10 3 = 1 / 3 ∧
    (∀ p1 p2 p3 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
      (Nat.card {triple | triple = (p1, p2, p3) ∧ is_right_triangle p1 p2 p3} = num_right_triangles))) ∧
  (∃ (num_rectangles : Nat),
    num_rectangles / choose_10 4 = 1 / 21 ∧
    (∀ p1 p2 p3 p4 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 →
      (Nat.card {quad | quad = (p1, p2, p3, p4) ∧ is_rectangle p1 p2 p3 p4} = num_rectangles))) :=
by sorry

end circle_probabilities_l2063_206312


namespace parking_theorem_l2063_206334

/-- The number of ways to arrange n distinct objects in k positions --/
def arrange (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items --/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to park cars in a row with contiguous empty spaces --/
def parkingArrangements (total_spaces : ℕ) (cars : ℕ) : ℕ :=
  arrange cars cars * choose (cars + 1) 1

theorem parking_theorem :
  parkingArrangements 12 8 = arrange 8 8 * choose 9 1 := by sorry

end parking_theorem_l2063_206334


namespace total_cost_5_and_5_l2063_206380

/-- The cost of a single room in yuan -/
def single_room_cost : ℝ := sorry

/-- The cost of a double room in yuan -/
def double_room_cost : ℝ := sorry

/-- The total cost of 3 single rooms and 6 double rooms is 1020 yuan -/
axiom cost_equation_1 : 3 * single_room_cost + 6 * double_room_cost = 1020

/-- The total cost of 1 single room and 5 double rooms is 700 yuan -/
axiom cost_equation_2 : single_room_cost + 5 * double_room_cost = 700

/-- The theorem states that the total cost of 5 single rooms and 5 double rooms is 1100 yuan -/
theorem total_cost_5_and_5 : 5 * single_room_cost + 5 * double_room_cost = 1100 := by
  sorry

end total_cost_5_and_5_l2063_206380


namespace age_ratio_in_ten_years_l2063_206357

/-- Represents the age difference between Pete and Claire -/
structure AgeDifference where
  pete : ℕ
  claire : ℕ

/-- The conditions of the problem -/
def age_conditions (ad : AgeDifference) : Prop :=
  ∃ (x : ℕ),
    -- Claire's age 2 years ago
    ad.claire = x + 2 ∧
    -- Pete's age 2 years ago
    ad.pete = 3 * x + 2 ∧
    -- Four years ago condition
    3 * x - 4 = 4 * (x - 4)

/-- The theorem to be proved -/
theorem age_ratio_in_ten_years (ad : AgeDifference) :
  age_conditions ad →
  (ad.pete + 10) / (ad.claire + 10) = 2 := by
sorry

end age_ratio_in_ten_years_l2063_206357


namespace acid_dilution_l2063_206307

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end acid_dilution_l2063_206307


namespace binomial_prob_X_eq_one_l2063_206314

/-- A random variable X following a binomial distribution B(n, p) with given expectation and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 2
  var_eq : n * p * (1 - p) = 5 / 4

/-- The probability of X = 1 for the given binomial random variable -/
def prob_X_eq_one (X : BinomialRV) : ℝ :=
  X.n.choose 1 * X.p^1 * (1 - X.p)^(X.n - 1)

/-- Theorem stating that P(X=1) = 5/32 for the given binomial random variable -/
theorem binomial_prob_X_eq_one (X : BinomialRV) : prob_X_eq_one X = 5 / 32 := by
  sorry

end binomial_prob_X_eq_one_l2063_206314


namespace factorial_inequality_l2063_206386

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n.factorial ≤ ((n + 1) / 2 : ℝ) ^ n := by
  sorry

end factorial_inequality_l2063_206386


namespace multiple_of_99_sum_of_digits_l2063_206306

theorem multiple_of_99_sum_of_digits (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → 
  (100000 * A + 15000 + 100 * B + 94) % 99 = 0 →
  A + B = 8 := by
sorry

end multiple_of_99_sum_of_digits_l2063_206306


namespace ion_relationship_l2063_206303

/-- Represents an ion with atomic number and charge -/
structure Ion where
  atomic_number : ℕ
  charge : ℤ

/-- Two ions have the same electron shell structure -/
def same_electron_shell (x y : Ion) : Prop :=
  x.atomic_number + x.charge = y.atomic_number - y.charge

theorem ion_relationship {a b n m : ℕ} (X Y : Ion)
  (hX : X.atomic_number = a ∧ X.charge = -n)
  (hY : Y.atomic_number = b ∧ Y.charge = m)
  (h_same_shell : same_electron_shell X Y) :
  a + m = b - n := by
  sorry


end ion_relationship_l2063_206303


namespace sqrt_x_squared_y_simplification_l2063_206310

theorem sqrt_x_squared_y_simplification (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by sorry

end sqrt_x_squared_y_simplification_l2063_206310


namespace complementary_angles_difference_l2063_206396

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 4 / 5 →  -- ratio of angles is 4:5
  |a - b| = 10 :=  -- positive difference is 10°
by sorry

end complementary_angles_difference_l2063_206396


namespace calories_burned_walking_james_walking_calories_l2063_206335

/-- Calculates the calories burned per hour while walking based on dancing data -/
theorem calories_burned_walking (dancing_calories_per_hour : ℝ) 
  (dancing_sessions_per_day : ℕ) (dancing_hours_per_session : ℝ) 
  (dancing_days_per_week : ℕ) (total_calories_per_week : ℝ) : ℝ :=
  let dancing_calories_ratio := 2
  let dancing_hours_per_week := dancing_sessions_per_day * dancing_hours_per_session * dancing_days_per_week
  let walking_calories_per_hour := total_calories_per_week / dancing_hours_per_week / dancing_calories_ratio
  by
    -- Proof goes here
    sorry

/-- Verifies that James burns 300 calories per hour while walking -/
theorem james_walking_calories : 
  calories_burned_walking 600 2 0.5 4 2400 = 300 := by
  -- Proof goes here
  sorry

end calories_burned_walking_james_walking_calories_l2063_206335


namespace cube_rect_surface_area_ratio_l2063_206381

theorem cube_rect_surface_area_ratio (a b : ℝ) (h : a > 0) :
  2 * a^2 + 4 * a * b = 0.6 * (6 * a^2) → b = 0.6 * a := by
  sorry

end cube_rect_surface_area_ratio_l2063_206381


namespace max_abs_z_l2063_206351

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4*I) = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 3 + 4*I) = 2 ∧
  ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) = 2 → Complex.abs u ≤ Complex.abs w :=
sorry

end max_abs_z_l2063_206351


namespace sum_of_squares_of_roots_l2063_206333

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  a^2 + b^2 + c^2 = 175 := by
sorry

end sum_of_squares_of_roots_l2063_206333


namespace geometric_progression_ratio_l2063_206365

/-- Given three terms of a geometric progression, prove that the common ratio is 52/25 -/
theorem geometric_progression_ratio (x : ℝ) (h_x : x ≠ 0) :
  let a₁ : ℝ := x / 2
  let a₂ : ℝ := 2 * x - 3
  let a₃ : ℝ := 18 / x + 1
  (a₁ * a₃ = a₂^2) → (a₂ / a₁ = 52 / 25) := by
  sorry

end geometric_progression_ratio_l2063_206365


namespace a_100_eq_344934_l2063_206385

/-- Sequence defined by a(n) = a(n-1) + n^2 for n ≥ 1, with a(0) = 2009 -/
def a : ℕ → ℕ
  | 0 => 2009
  | n + 1 => a n + (n + 1)^2

/-- The 100th term of the sequence a is 344934 -/
theorem a_100_eq_344934 : a 100 = 344934 := by
  sorry

end a_100_eq_344934_l2063_206385


namespace simplify_fraction_l2063_206356

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := by
  sorry

end simplify_fraction_l2063_206356


namespace factor_expression_l2063_206390

theorem factor_expression (x : ℝ) : 84 * x^7 - 306 * x^13 = 6 * x^7 * (14 - 51 * x^6) := by
  sorry

end factor_expression_l2063_206390


namespace smallest_prime_perimeter_scalene_triangle_l2063_206370

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three natural numbers form a scalene triangle -/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A function that checks if three natural numbers satisfy the triangle inequality -/
def satisfiesTriangleInequality (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    a ≥ 11 ∧ b ≥ 11 ∧ c ≥ 11 ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    isScalene a b c ∧
    satisfiesTriangleInequality a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 41) ∧
    (∀ (x y z : ℕ), 
      x ≥ 11 ∧ y ≥ 11 ∧ z ≥ 11 →
      isPrime x ∧ isPrime y ∧ isPrime z →
      isScalene x y z →
      satisfiesTriangleInequality x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 41) := by
  sorry

end smallest_prime_perimeter_scalene_triangle_l2063_206370


namespace min_red_chips_l2063_206323

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ 72 :=
by sorry

end min_red_chips_l2063_206323


namespace right_triangle_sine_calculation_l2063_206330

theorem right_triangle_sine_calculation (D E F : ℝ) :
  0 < D ∧ D < π/2 →
  0 < E ∧ E < π/2 →
  0 < F ∧ F < π/2 →
  D + E + F = π →
  Real.sin D = 5/13 →
  Real.sin E = 1 →
  Real.sin F = 12/13 := by
sorry

end right_triangle_sine_calculation_l2063_206330


namespace log_equality_l2063_206384

theorem log_equality : Real.log 81 / Real.log 4 = Real.log 9 / Real.log 2 := by
  sorry

end log_equality_l2063_206384


namespace solution_sum_l2063_206361

theorem solution_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ * y₁ - x₁ = 180 ∧ y₁ + x₁ * y₁ = 208) ∧
  (x₂ * y₂ - x₂ = 180 ∧ y₂ + x₂ * y₂ = 208) ∧
  (x₁ ≠ x₂) →
  x₁ + 10 * y₁ + x₂ + 10 * y₂ = 317 := by
sorry

end solution_sum_l2063_206361


namespace divisibility_property_l2063_206368

theorem divisibility_property (n : ℕ) : 
  n > 0 ∧ n^2 ∣ 2^n + 1 ↔ n = 1 ∨ n = 3 := by sorry

end divisibility_property_l2063_206368


namespace system_solution_l2063_206377

theorem system_solution :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 := by
  sorry

end system_solution_l2063_206377


namespace computer_contract_probability_l2063_206308

theorem computer_contract_probability (p_hardware : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) :
  p_hardware = 3/4 →
  p_at_least_one = 5/6 →
  p_both = 0.31666666666666654 →
  1 - (p_at_least_one - p_hardware + p_both) = 0.6 :=
by sorry

end computer_contract_probability_l2063_206308


namespace cricket_average_l2063_206373

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (initial_average : ℕ) : 
  innings = 20 →
  next_runs = 200 →
  increase = 8 →
  (innings * initial_average + next_runs) / (innings + 1) = initial_average + increase →
  initial_average = 32 := by
  sorry

end cricket_average_l2063_206373


namespace negative_215_in_fourth_quadrant_l2063_206321

-- Define a function to convert degrees to the equivalent angle in the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem stating that -215° is in the fourth quadrant
theorem negative_215_in_fourth_quadrant :
  getQuadrant (-215) = 4 := by sorry

end negative_215_in_fourth_quadrant_l2063_206321


namespace swimmer_journey_l2063_206337

/-- Swimmer's journey problem -/
theorem swimmer_journey 
  (swimmer_speed : ℝ) 
  (current_speed : ℝ) 
  (distance_PQ : ℝ) 
  (distance_QR : ℝ) 
  (h1 : swimmer_speed = 1)
  (h2 : distance_PQ / (swimmer_speed + current_speed) + distance_QR / swimmer_speed = 3)
  (h3 : distance_QR / (swimmer_speed - current_speed) + distance_PQ / (swimmer_speed - current_speed) = 6)
  (h4 : (distance_PQ + distance_QR) / (swimmer_speed + current_speed) = 5/2)
  : (distance_QR + distance_PQ) / (swimmer_speed - current_speed) = 15/2 := by
  sorry

end swimmer_journey_l2063_206337


namespace rational_cube_sum_representation_l2063_206355

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end rational_cube_sum_representation_l2063_206355


namespace greatest_q_minus_r_l2063_206301

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 
    1043 = 23 * q + r ∧ 
    r > 0 ∧ 
    ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' > 0 → q' - r' ≤ q - r ∧ 
    q - r = 37 := by
  sorry

end greatest_q_minus_r_l2063_206301


namespace quadratic_equation_solution_l2063_206318

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2*x*(x+1) - 3*(x+1)
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3/2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l2063_206318


namespace yellow_red_block_difference_l2063_206389

/-- Given a toy bin with red, yellow, and blue blocks, prove the difference between yellow and red blocks -/
theorem yellow_red_block_difference 
  (red : ℕ) 
  (yellow : ℕ) 
  (blue : ℕ) 
  (h1 : red = 18) 
  (h2 : yellow > red) 
  (h3 : blue = red + 14) 
  (h4 : red + yellow + blue = 75) : 
  yellow - red = 7 := by
  sorry

end yellow_red_block_difference_l2063_206389


namespace range_of_a_when_union_is_reals_l2063_206393

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -4 + a < x ∧ x < 4 + a}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem statement
theorem range_of_a_when_union_is_reals :
  ∀ a : ℝ, (A a ∪ B = Set.univ) ↔ (1 < a ∧ a < 3) := by sorry

end range_of_a_when_union_is_reals_l2063_206393


namespace xy_equals_three_l2063_206309

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdist : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end xy_equals_three_l2063_206309


namespace min_even_integers_l2063_206369

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 46 →
  a + b + c + d + e + f = 64 →
  ∃ (x y z w u v : ℤ), 
    x + y = 28 ∧
    x + y + z + w = 46 ∧
    x + y + z + w + u + v = 64 ∧
    Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧ Odd u ∧ Odd v :=
by sorry

end min_even_integers_l2063_206369


namespace sum_of_twenty_terms_l2063_206359

/-- Given a sequence of non-zero terms {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₂₀ = 210 under the given conditions. -/
theorem sum_of_twenty_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →
  (∀ n, S n = (a n * a (n + 1)) / 2) →
  a 1 = 1 →
  S 20 = 210 := by
sorry

end sum_of_twenty_terms_l2063_206359


namespace greatest_common_factor_3465_10780_l2063_206394

theorem greatest_common_factor_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end greatest_common_factor_3465_10780_l2063_206394


namespace exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l2063_206313

-- Proposition ③
theorem exists_x0_exp_greater_than_square :
  ∃ x₀ : ℝ, ∀ x > x₀, (2 : ℝ) ^ x > x ^ 2 := by sorry

-- Proposition ⑤
theorem sum_of_roots_equals_five :
  let f₁ := fun x : ℝ => x + Real.log 2 * Real.log x / Real.log 10 - 5
  let f₂ := fun x : ℝ => x + (10 : ℝ) ^ x - 5
  ∀ x₁ x₂ : ℝ, f₁ x₁ = 0 → f₂ x₂ = 0 → x₁ + x₂ = 5 := by sorry

end exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l2063_206313


namespace chord_length_l2063_206328

-- Define the circle and points
variable (O A B C D : Point)
variable (r : ℝ)

-- Define the circle properties
def is_circle (O : Point) (r : ℝ) : Prop := sorry

-- Define diameter
def is_diameter (O A D : Point) : Prop := sorry

-- Define chord
def is_chord (O A B C : Point) : Prop := sorry

-- Define arc measure
def arc_measure (O C D : Point) : ℝ := sorry

-- Define angle measure
def angle_measure (A B O : Point) : ℝ := sorry

-- Define distance between points
def distance (P Q : Point) : ℝ := sorry

-- Theorem statement
theorem chord_length 
  (h_circle : is_circle O r)
  (h_diameter : is_diameter O A D)
  (h_chord : is_chord O A B C)
  (h_BO : distance B O = 7)
  (h_angle : angle_measure A B O = 45)
  (h_arc : arc_measure O C D = 90) :
  distance B C = 7 := by sorry

end chord_length_l2063_206328


namespace line_equation_through_points_l2063_206336

theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  (∃ t : ℝ, x = 1 - t ∧ y = -2 * t) :=
sorry

end line_equation_through_points_l2063_206336


namespace price_changes_l2063_206383

theorem price_changes (original_price : ℝ) : 
  let price_after_first_increase := original_price * 1.2
  let price_after_second_increase := price_after_first_increase + 5
  let price_after_first_decrease := price_after_second_increase * 0.8
  let final_price := price_after_first_decrease - 5
  final_price = 120 → original_price = 126.04 := by
sorry

#eval (121 / 0.96 : Float)

end price_changes_l2063_206383


namespace harry_lost_sea_creatures_l2063_206342

theorem harry_lost_sea_creatures (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34)
  (h2 : seashells = 21)
  (h3 : snails = 29)
  (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 := by
  sorry

end harry_lost_sea_creatures_l2063_206342


namespace ln_ratio_monotone_l2063_206398

open Real

theorem ln_ratio_monotone (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1) :
  (log a) / a < (log b) / b ∧ (log b) / b < (log c) / c :=
by sorry

end ln_ratio_monotone_l2063_206398


namespace quadratic_inequality_solution_l2063_206332

theorem quadratic_inequality_solution (x : ℝ) :
  2 * x^2 - 4 * x - 70 > 0 ∧ x ≠ -2 ∧ x ≠ 0 →
  x < -5 ∨ x > 7 :=
by sorry

end quadratic_inequality_solution_l2063_206332


namespace perpendicular_line_through_point_l2063_206329

/-- A line passing through a point and perpendicular to another line -/
structure PerpendicularLine where
  point : ℝ × ℝ
  other_line : ℝ → ℝ → ℝ → ℝ

/-- The equation of the perpendicular line -/
def perpendicular_line_equation (l : PerpendicularLine) : ℝ → ℝ → ℝ → ℝ :=
  fun x y c => 3 * x + 2 * y + c

theorem perpendicular_line_through_point (l : PerpendicularLine)
  (h1 : l.point = (-1, 2))
  (h2 : l.other_line = fun x y c => 2 * x - 3 * y + c) :
  perpendicular_line_equation l (-1) 2 (-1) = 0 ∧
  perpendicular_line_equation l = fun x y c => 3 * x + 2 * y - 1 :=
sorry

end perpendicular_line_through_point_l2063_206329


namespace quadratic_inequality_solution_set_l2063_206346

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5} ∪ {x : ℝ | x < -1} := by
sorry

end quadratic_inequality_solution_set_l2063_206346


namespace units_digit_of_expression_l2063_206315

theorem units_digit_of_expression : ∃ n : ℕ, 
  (13 + Real.sqrt 196)^13 + (13 + Real.sqrt 196)^71 = 10 * n :=
by sorry

end units_digit_of_expression_l2063_206315


namespace framed_painting_ratio_l2063_206311

theorem framed_painting_ratio : 
  let painting_size : ℝ := 20
  let frame_side (x : ℝ) := x
  let frame_top_bottom (x : ℝ) := 3 * x
  let framed_width (x : ℝ) := painting_size + 2 * frame_side x
  let framed_height (x : ℝ) := painting_size + 2 * frame_top_bottom x
  let frame_area (x : ℝ) := framed_width x * framed_height x - painting_size^2
  ∃ x : ℝ, 
    x > 0 ∧ 
    frame_area x = painting_size^2 ∧
    (min (framed_width x) (framed_height x)) / (max (framed_width x) (framed_height x)) = 4/7 :=
by sorry

end framed_painting_ratio_l2063_206311


namespace j_type_sequence_properties_l2063_206348

/-- Definition of a J_k type sequence -/
def is_J_k_type (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, (a (n + k))^2 = a n * a (n + 2*k)

theorem j_type_sequence_properties 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) :
  (is_J_k_type a 2 ∧ a 2 = 8 ∧ a 8 = 1 → 
    ∀ n : ℕ, a (2*n) = 2^(4-n)) ∧
  (is_J_k_type a 3 ∧ is_J_k_type a 4 → 
    ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n) :=
sorry

end j_type_sequence_properties_l2063_206348


namespace combined_swimming_distance_l2063_206326

/-- Given swimming distances for Jamir, Sarah, and Julien, prove their combined weekly distance --/
theorem combined_swimming_distance
  (julien_daily : ℕ)
  (sarah_daily : ℕ)
  (jamir_daily : ℕ)
  (days_in_week : ℕ)
  (h1 : julien_daily = 50)
  (h2 : sarah_daily = 2 * julien_daily)
  (h3 : jamir_daily = sarah_daily + 20)
  (h4 : days_in_week = 7) :
  julien_daily * days_in_week +
  sarah_daily * days_in_week +
  jamir_daily * days_in_week = 1890 := by
sorry

end combined_swimming_distance_l2063_206326


namespace geometric_sequence_ratio_l2063_206388

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = -1/2 * a n) →
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 := by
  sorry

end geometric_sequence_ratio_l2063_206388


namespace sphere_impulse_theorem_l2063_206360

/-- Represents a uniform sphere -/
structure UniformSphere where
  mass : ℝ
  radius : ℝ

/-- Represents the initial conditions and applied impulse -/
structure ImpulseConditions where
  sphere : UniformSphere
  impulse : ℝ
  beta : ℝ

/-- Theorem stating the final speed and condition for rolling without slipping -/
theorem sphere_impulse_theorem (conditions : ImpulseConditions) 
  (h1 : conditions.beta ≥ -1) 
  (h2 : conditions.beta ≤ 1) : 
  ∃ (v : ℝ), 
    v = (5 * conditions.impulse * conditions.beta) / (7 * conditions.sphere.mass) ∧
    (conditions.beta = 7/5 → 
      v * conditions.sphere.mass = conditions.impulse ∧ 
      v = conditions.sphere.radius * ((5 * conditions.impulse * conditions.beta) / 
        (7 * conditions.sphere.mass * conditions.sphere.radius))) := by
  sorry

end sphere_impulse_theorem_l2063_206360


namespace seating_arrangements_l2063_206379

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of people to be seated. -/
def totalPeople : ℕ := 10

/-- The number of people who cannot sit in three consecutive seats. -/
def cannotSitTogether : ℕ := 3

/-- The number of people who must sit together. -/
def mustSitTogether : ℕ := 2

/-- The number of seating arrangements where Alice, Bob, and Cindy sit together. -/
def arrangementsTogether : ℕ := factorial (totalPeople - cannotSitTogether + 1) * factorial cannotSitTogether

/-- The number of seating arrangements where Dave and Emma sit together. -/
def arrangementsPairTogether : ℕ := factorial (totalPeople - mustSitTogether + 1) * factorial mustSitTogether

/-- The number of seating arrangements where both conditions are met simultaneously. -/
def arrangementsOverlap : ℕ := factorial (totalPeople - cannotSitTogether - mustSitTogether + 2) * factorial cannotSitTogether * factorial mustSitTogether

/-- The total number of valid seating arrangements. -/
def validArrangements : ℕ := factorial totalPeople - (arrangementsTogether + arrangementsPairTogether - arrangementsOverlap)

theorem seating_arrangements : validArrangements = 3144960 := by sorry

end seating_arrangements_l2063_206379


namespace matrix_power_4_l2063_206367

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_power_4 : A^4 = !![0, -9; 9, -9] := by sorry

end matrix_power_4_l2063_206367


namespace blue_shirt_percentage_l2063_206327

/-- Proves that the percentage of students wearing blue shirts is 45% -/
theorem blue_shirt_percentage
  (total_students : ℕ)
  (red_shirt_percentage : ℚ)
  (green_shirt_percentage : ℚ)
  (other_colors_count : ℕ)
  (h1 : total_students = 600)
  (h2 : red_shirt_percentage = 23 / 100)
  (h3 : green_shirt_percentage = 15 / 100)
  (h4 : other_colors_count = 102)
  : (1 : ℚ) - (red_shirt_percentage + green_shirt_percentage + (other_colors_count : ℚ) / (total_students : ℚ)) = 45 / 100 := by
  sorry

#check blue_shirt_percentage

end blue_shirt_percentage_l2063_206327


namespace arithmetic_geometric_ratio_l2063_206391

/-- An arithmetic progression with a non-zero difference -/
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Consecutive terms of a geometric progression -/
def geometric_progression (x y z : ℝ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_progression a d)
  (h_geom : geometric_progression (a 10) (a 13) (a 19)) :
  (a 12) / (a 18) = 5 / 11 :=
sorry

end arithmetic_geometric_ratio_l2063_206391


namespace vacation_pictures_l2063_206302

theorem vacation_pictures (zoo museum beach deleted : ℕ) :
  zoo = 120 →
  museum = 75 →
  beach = 45 →
  deleted = 93 →
  zoo + museum + beach - deleted = 147 :=
by sorry

end vacation_pictures_l2063_206302


namespace inequality_equivalence_l2063_206319

theorem inequality_equivalence (x : ℝ) : 
  5 - 3 / (3 * x - 2) < 7 ↔ x < 1/6 := by sorry

end inequality_equivalence_l2063_206319


namespace bucket_weight_bucket_weight_proof_l2063_206392

/-- Given a bucket with the following properties:
  1. When three-quarters full of water, it weighs c kilograms (including the water).
  2. When one-third full of water, it weighs d kilograms (including the water).
  This theorem states that when the bucket is completely full of water, 
  its total weight is (8/5)c - (7/5)d kilograms. -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let three_quarters_full := c
  let one_third_full := d
  let full_weight := (8/5) * c - (7/5) * d
  full_weight

/-- Proof of the bucket_weight theorem -/
theorem bucket_weight_proof (c d : ℝ) : 
  bucket_weight c d = (8/5) * c - (7/5) * d :=
by sorry

end bucket_weight_bucket_weight_proof_l2063_206392


namespace f_negative_l2063_206300

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^2 - x + 1

-- Theorem statement
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^2 - x - 1 := by
sorry

end f_negative_l2063_206300


namespace largest_trick_number_l2063_206395

/-- The constant k representing the number 2017 -/
def k : ℕ := 2017

/-- A function that determines whether the card trick can be performed for a given number of cards -/
def canPerformTrick (n : ℕ) : Prop :=
  n ≤ k + 1 ∧ (n ≤ k → False)

/-- Theorem stating that 2018 is the largest number for which the trick can be performed -/
theorem largest_trick_number : ∀ n : ℕ, canPerformTrick n ↔ n = k + 1 :=
  sorry

end largest_trick_number_l2063_206395


namespace line_relationships_l2063_206347

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : 
  ¬ parallel c b := by sorry

end line_relationships_l2063_206347


namespace function_properties_l2063_206358

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Statement of the theorem
theorem function_properties :
  -- 1. The tangent line to y = f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. There are exactly 2 lines tangent to y = f(x) passing through (1, -1)
  (∃! a b : ℝ, a ≠ b ∧ 
    (∀ x, f x = (Real.log a + 1) * (x - a) + f a) ∧
    (Real.log a + 1) * (1 - a) + f a = -1 ∧
    (∀ x, f x = (Real.log b + 1) * (x - b) + f b) ∧
    (Real.log b + 1) * (1 - b) + f b = -1) ∧
  -- 3. f(x) has a local minimum and no local maximum
  (∃ c : ℝ, ∀ x, x > 0 → x ≠ c → f x > f c) ∧
  (¬ ∃ d : ℝ, ∀ x, x > 0 → x ≠ d → f x < f d) ∧
  -- 4. The equation f(x) = 1 does not have two distinct solutions
  ¬ (∃ x y : ℝ, x ≠ y ∧ f x = 1 ∧ f y = 1) :=
by sorry


end function_properties_l2063_206358


namespace necessary_but_not_sufficient_l2063_206339

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((a > 1) ∧ (b > 1)) → (a + b > 2)) ∧
  (∃ a b : ℝ, (a + b > 2) ∧ ¬((a > 1) ∧ (b > 1))) :=
by sorry

end necessary_but_not_sufficient_l2063_206339


namespace modulus_of_complex_product_l2063_206343

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (3 - 4 * Complex.I) ∧ Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end modulus_of_complex_product_l2063_206343


namespace even_function_property_l2063_206350

def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = -x * (1 - x) := by
sorry

end even_function_property_l2063_206350


namespace quarterback_throws_l2063_206344

/-- Proves that given the specified conditions, the quarterback stepped back to throw 80 times. -/
theorem quarterback_throws (p_no_throw : ℝ) (p_sack_given_no_throw : ℝ) (num_sacks : ℕ) :
  p_no_throw = 0.3 →
  p_sack_given_no_throw = 0.5 →
  num_sacks = 12 →
  ∃ (total_throws : ℕ), total_throws = 80 ∧ 
    (p_no_throw * p_sack_given_no_throw * total_throws : ℝ) = num_sacks := by
  sorry

#check quarterback_throws

end quarterback_throws_l2063_206344


namespace tan_negative_five_pi_sixths_l2063_206372

theorem tan_negative_five_pi_sixths : 
  Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_negative_five_pi_sixths_l2063_206372


namespace remainder_not_composite_l2063_206374

theorem remainder_not_composite (p : Nat) (h_prime : Nat.Prime p) (h_gt_30 : p > 30) :
  ¬(∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p % 30 = a * b) := by
  sorry

end remainder_not_composite_l2063_206374


namespace average_age_union_l2063_206364

-- Define the student groups and their properties
def StudentGroup := Type
variables (A B C : StudentGroup)

-- Define the number of students in each group
variables (a b c : ℕ)

-- Define the sum of ages in each group
variables (sumA sumB sumC : ℕ)

-- Define the average age function
def avgAge (sum : ℕ) (count : ℕ) : ℚ := sum / count

-- State the theorem
theorem average_age_union (h_disjoint : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_avgA : avgAge sumA a = 34)
  (h_avgB : avgAge sumB b = 25)
  (h_avgC : avgAge sumC c = 45)
  (h_avgAB : avgAge (sumA + sumB) (a + b) = 30)
  (h_avgAC : avgAge (sumA + sumC) (a + c) = 42)
  (h_avgBC : avgAge (sumB + sumC) (b + c) = 36) :
  avgAge (sumA + sumB + sumC) (a + b + c) = 33 := by
  sorry


end average_age_union_l2063_206364


namespace exists_m_for_even_f_l2063_206354

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx for some m ∈ ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_for_even_f : ∃ m : ℝ, IsEven (f m) := by
  sorry

end exists_m_for_even_f_l2063_206354


namespace ant_height_proof_l2063_206325

theorem ant_height_proof (rope_length : ℝ) (base_distance : ℝ) (shadow_rate : ℝ) (time : ℝ) 
  (h_rope : rope_length = 10)
  (h_base : base_distance = 6)
  (h_rate : shadow_rate = 0.3)
  (h_time : time = 5)
  (h_right_triangle : rope_length ^ 2 = base_distance ^ 2 + (rope_length ^ 2 - base_distance ^ 2))
  : ∃ (height : ℝ), 
    height = 2 ∧ 
    (shadow_rate * time) / base_distance = height / (rope_length ^ 2 - base_distance ^ 2).sqrt :=
by sorry

end ant_height_proof_l2063_206325


namespace infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l2063_206341

def IsFantastic (n : ℕ) : Prop :=
  ∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ n = ⌊a + 1/a + b + 1/b⌋

theorem infinite_primes_no_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ (p : ℕ) (k : ℕ), p ∈ S → k > 0 → ¬IsFantastic (k * p)) :=
sorry

theorem infinite_primes_with_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ p ∈ S, ∃ (k : ℕ), k > 0 ∧ IsFantastic (k * p)) :=
sorry

end infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l2063_206341


namespace hexagon_area_in_circle_l2063_206397

/-- The area of a regular hexagon inscribed in a circle with area 196π square units is 294√3 square units. -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 196 * Real.pi → hexagon_area = 294 * Real.sqrt 3 := by
  sorry

end hexagon_area_in_circle_l2063_206397


namespace equal_sprocket_production_l2063_206363

/-- Represents the production rates and times of two machines manufacturing sprockets -/
structure SprocketProduction where
  machine_a_rate : ℝ  -- Sprockets per hour for Machine A
  machine_b_rate : ℝ  -- Sprockets per hour for Machine B
  machine_b_time : ℝ  -- Time taken by Machine B in hours

/-- Theorem stating that both machines produce the same number of sprockets -/
theorem equal_sprocket_production (sp : SprocketProduction) 
  (h1 : sp.machine_a_rate = 4)  -- Machine A produces 4 sprockets per hour
  (h2 : sp.machine_b_rate = sp.machine_a_rate * 1.1)  -- Machine B is 10% faster
  (h3 : sp.machine_b_time * sp.machine_b_rate = (sp.machine_b_time + 10) * sp.machine_a_rate)  -- Total production is equal
  : sp.machine_a_rate * (sp.machine_b_time + 10) = 440 ∧ sp.machine_b_rate * sp.machine_b_time = 440 :=
by sorry

end equal_sprocket_production_l2063_206363


namespace range_of_a_l2063_206352

theorem range_of_a (x : ℝ) (a : ℝ) : 
  (∀ x, (0 < x ∧ x < a) → (|x - 2| < 3)) ∧ 
  (∃ x, |x - 2| < 3 ∧ ¬(0 < x ∧ x < a)) ∧
  (a > 0) →
  (0 < a ∧ a ≤ 5) :=
sorry

end range_of_a_l2063_206352
