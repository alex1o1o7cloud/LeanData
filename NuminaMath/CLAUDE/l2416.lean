import Mathlib

namespace NUMINAMATH_CALUDE_jeans_price_increase_l2416_241671

theorem jeans_price_increase (manufacturing_cost : ℝ) : 
  let retailer_price := manufacturing_cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.82 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l2416_241671


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2416_241604

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A decagon (10-sided polygon) has 35 diagonals -/
theorem decagon_diagonals :
  num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2416_241604


namespace NUMINAMATH_CALUDE_opposite_numbers_l2416_241603

theorem opposite_numbers : -4^2 = -((- 4)^2) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l2416_241603


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2416_241692

-- Define an isosceles triangle with one interior angle of 50°
structure IsoscelesTriangle :=
  (base_angle₁ : ℝ)
  (base_angle₂ : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle₁ = base_angle₂)
  (has_50_degree_angle : base_angle₁ = 50 ∨ base_angle₂ = 50 ∨ vertex_angle = 50)
  (angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180)

-- Theorem stating that the base angles are either 50° or 65°
theorem isosceles_triangle_base_angles (t : IsoscelesTriangle) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2416_241692


namespace NUMINAMATH_CALUDE_heathers_oranges_l2416_241690

/-- The total number of oranges Heather has after receiving more from Russell -/
def total_oranges (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Heather's total oranges is the sum of her initial oranges and those received from Russell -/
theorem heathers_oranges (initial : Float) (received : Float) :
  total_oranges initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_heathers_oranges_l2416_241690


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2416_241624

theorem sum_of_specific_numbers (a b : ℝ) 
  (ha_abs : |a| = 5)
  (hb_abs : |b| = 2)
  (ha_neg : a < 0)
  (hb_pos : b > 0) :
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2416_241624


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2416_241633

theorem floor_equation_solution (x : ℝ) : x - Int.floor (x / 2016) = 2016 ↔ x = 2017 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2416_241633


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2416_241685

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (invalid_votes : ℕ),
    invalid_votes = 100 →
    valid_votes = total_votes - invalid_votes →
    ∃ (loser_votes winner_votes : ℕ),
      loser_votes = (30 * valid_votes) / 100 ∧
      winner_votes = valid_votes - loser_votes ∧
      winner_votes = loser_votes + 5000 →
      total_votes = 12600 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2416_241685


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l2416_241630

theorem multiply_divide_sqrt (x y : ℝ) (hx : x = 1.4) (hx_neq_zero : x ≠ 0) :
  Real.sqrt ((x * y) / 5) = x → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l2416_241630


namespace NUMINAMATH_CALUDE_sum_of_units_digits_of_seven_powers_l2416_241606

def units_digit (n : ℕ) : ℕ := n % 10

def A (n : ℕ) : ℕ := units_digit (7^n)

theorem sum_of_units_digits_of_seven_powers : 
  (Finset.range 2013).sum A + A 2013 = 10067 := by sorry

end NUMINAMATH_CALUDE_sum_of_units_digits_of_seven_powers_l2416_241606


namespace NUMINAMATH_CALUDE_ten_steps_climb_ways_l2416_241650

/-- The number of ways to climb n steps, where each move is either climbing 1 step or 2 steps -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => climbStairs k + climbStairs (k + 1)

/-- Theorem stating that there are 89 ways to climb 10 steps -/
theorem ten_steps_climb_ways : climbStairs 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_steps_climb_ways_l2416_241650


namespace NUMINAMATH_CALUDE_chris_video_game_cost_l2416_241608

def video_game_cost (hourly_rate : ℕ) (hours_worked : ℕ) (candy_cost : ℕ) (leftover : ℕ) : ℕ :=
  hourly_rate * hours_worked - candy_cost - leftover

theorem chris_video_game_cost :
  video_game_cost 8 9 5 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chris_video_game_cost_l2416_241608


namespace NUMINAMATH_CALUDE_square_roots_of_625_l2416_241652

theorem square_roots_of_625 :
  (∃ x : ℝ, x > 0 ∧ x^2 = 625 ∧ x = 25) ∧
  (∀ x : ℝ, x^2 = 625 ↔ x = 25 ∨ x = -25) := by
  sorry

end NUMINAMATH_CALUDE_square_roots_of_625_l2416_241652


namespace NUMINAMATH_CALUDE_f_zero_equals_three_l2416_241673

-- Define the function f
noncomputable def f (t : ℝ) : ℝ :=
  let x := (t + 1) / 2
  (1 - x^2) / x^2

-- Theorem statement
theorem f_zero_equals_three :
  f 0 = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_equals_three_l2416_241673


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2416_241628

theorem add_preserves_inequality (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2416_241628


namespace NUMINAMATH_CALUDE_range_of_m_for_true_proposition_l2416_241602

theorem range_of_m_for_true_proposition (m : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) →
  m ≤ 1 ∧ ∀ y : ℝ, y < m → ∃ x : ℝ, 4^x - 2^(x + 1) + y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_true_proposition_l2416_241602


namespace NUMINAMATH_CALUDE_min_value_fraction_l2416_241641

theorem min_value_fraction (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 2) : 
  (∀ w x y z : ℝ, w > 0 → x > 0 → y > 0 → z > 0 → w + x + y + z = 2 → 
    (a + b + c) / (a * b * c * d) ≤ (w + x + y) / (w * x * y * z)) →
  (a + b + c) / (a * b * c * d) = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2416_241641


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2416_241697

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2416_241697


namespace NUMINAMATH_CALUDE_daily_profit_calculation_l2416_241600

theorem daily_profit_calculation (num_employees : ℕ) (employee_share : ℚ) (profit_share_percentage : ℚ) :
  num_employees = 9 →
  employee_share = 5 →
  profit_share_percentage = 9/10 →
  profit_share_percentage * ((num_employees : ℚ) * employee_share) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_daily_profit_calculation_l2416_241600


namespace NUMINAMATH_CALUDE_animals_to_shore_l2416_241699

theorem animals_to_shore (initial_sheep initial_cows initial_dogs : ℕ) 
  (drowned_sheep : ℕ) (h1 : initial_sheep = 20) (h2 : initial_cows = 10) 
  (h3 : initial_dogs = 14) (h4 : drowned_sheep = 3) 
  (h5 : 2 * drowned_sheep = initial_cows - (initial_cows - 2 * drowned_sheep)) :
  initial_sheep - drowned_sheep + (initial_cows - 2 * drowned_sheep) + initial_dogs = 35 := by
  sorry

end NUMINAMATH_CALUDE_animals_to_shore_l2416_241699


namespace NUMINAMATH_CALUDE_largest_three_digit_square_cube_l2416_241619

/-- The largest three-digit number that is both a perfect square and a perfect cube -/
def largest_square_cube : ℕ := 729

/-- A number is a three-digit number if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is a perfect square if there exists an integer whose square is that number -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is a perfect cube if there exists an integer whose cube is that number -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

theorem largest_three_digit_square_cube :
  is_three_digit largest_square_cube ∧
  is_perfect_square largest_square_cube ∧
  is_perfect_cube largest_square_cube ∧
  ∀ n : ℕ, is_three_digit n → is_perfect_square n → is_perfect_cube n → n ≤ largest_square_cube :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_cube_l2416_241619


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l2416_241620

/-- If a 35% reduction in a cost results in a decrease of $611, then the original cost was $1745.71 -/
theorem magazine_subscription_cost (C : ℝ) : (0.35 * C = 611) → C = 1745.71 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l2416_241620


namespace NUMINAMATH_CALUDE_number_problem_l2416_241657

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2416_241657


namespace NUMINAMATH_CALUDE_village_population_l2416_241691

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 23040 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 25600 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2416_241691


namespace NUMINAMATH_CALUDE_number_of_divisors_5005_l2416_241695

theorem number_of_divisors_5005 : Nat.card (Nat.divisors 5005) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_5005_l2416_241695


namespace NUMINAMATH_CALUDE_lottery_probability_l2416_241683

theorem lottery_probability (max_number : ℕ) 
  (prob_1_to_15 : ℚ) (prob_1_or_larger : ℚ) :
  max_number ≥ 15 →
  prob_1_to_15 = 1 / 3 →
  prob_1_or_larger = 2 / 3 →
  (∀ n : ℕ, n ≤ 15 → n ≥ 1) →
  (∀ n : ℕ, n ≤ max_number → n ≥ 1) →
  (probability_less_equal_15 : ℚ) →
  probability_less_equal_15 = prob_1_or_larger :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l2416_241683


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2416_241623

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 - 2*r₁ = 0 ∧ r₂^2 - 2*r₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2416_241623


namespace NUMINAMATH_CALUDE_sum_of_products_l2416_241696

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 57) :
  x*y + y*z + x*z = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2416_241696


namespace NUMINAMATH_CALUDE_person_age_is_54_l2416_241626

/-- Represents the age of a person and their eldest son, satisfying given conditions --/
structure AgeRelation where
  Y : ℕ  -- Current age of the person
  S : ℕ  -- Current age of the eldest son
  age_relation_past : Y - 9 = 5 * (S - 9)  -- Relation 9 years ago
  age_relation_present : Y = 3 * S         -- Current relation

/-- Theorem stating that given the conditions, the person's current age is 54 --/
theorem person_age_is_54 (ar : AgeRelation) : ar.Y = 54 := by
  sorry

end NUMINAMATH_CALUDE_person_age_is_54_l2416_241626


namespace NUMINAMATH_CALUDE_quadratic_properties_l2416_241698

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

-- Theorem statement
theorem quadratic_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (f a 0 = -2 → 
    (∃ x y : ℝ, (x = 1/2 ∨ x = -1/2) ∧ y = -9/4 ∧ 
    ∀ t : ℝ, f a t ≥ f a x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2416_241698


namespace NUMINAMATH_CALUDE_product_digit_sum_l2416_241625

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_digit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem product_digit_sum (n : ℕ) : 
  n ≥ 1 → digit_sum (5 * repeated_digit 5 n) ≥ 500 ↔ n ≥ 72 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2416_241625


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2416_241644

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with at least one box containing a ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 52 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2416_241644


namespace NUMINAMATH_CALUDE_cost_of_500_pieces_is_10_dollars_l2416_241607

/-- The cost of 500 pieces of gum in dollars -/
def cost_of_500_pieces : ℚ := 10

/-- The cost of 1 piece of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 10 dollars -/
theorem cost_of_500_pieces_is_10_dollars :
  cost_of_500_pieces = (500 * cost_per_piece : ℚ) / cents_per_dollar := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pieces_is_10_dollars_l2416_241607


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_with_equal_pairs_l2416_241694

theorem four_digit_perfect_square_with_equal_pairs : ∃ n : ℕ,
  (1000 ≤ n) ∧ (n ≤ 9999) ∧  -- four-digit number
  (∃ m : ℕ, n = m * m) ∧     -- perfect square
  (∃ a b : ℕ, 
    a < 10 ∧ b < 10 ∧        -- a and b are single digits
    n = 1000 * a + 100 * a + 10 * b + b) ∧  -- form aabb
  n = 7744 := by
sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_with_equal_pairs_l2416_241694


namespace NUMINAMATH_CALUDE_final_price_calculation_l2416_241647

/-- The markup percentage applied to the cost price -/
def markup : ℝ := 0.15

/-- The cost price of the computer table -/
def costPrice : ℝ := 5565.217391304348

/-- The final price paid by the customer -/
def finalPrice : ℝ := 6400

/-- Theorem stating that the final price is equal to the cost price plus the markup -/
theorem final_price_calculation :
  finalPrice = costPrice * (1 + markup) := by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l2416_241647


namespace NUMINAMATH_CALUDE_dave_apps_left_l2416_241618

/-- The number of apps Dave had left on his phone after adding and deleting apps. -/
def apps_left (initial_apps new_apps : ℕ) : ℕ :=
  initial_apps + new_apps - (new_apps + 1)

/-- Theorem stating that Dave had 14 apps left on his phone. -/
theorem dave_apps_left : apps_left 15 71 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l2416_241618


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_5_45_00_l2416_241645

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_45_00 :
  addSeconds { hours := 5, minutes := 45, seconds := 0 } 12345 =
  { hours := 9, minutes := 10, seconds := 45 } :=
sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_5_45_00_l2416_241645


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2416_241622

/-- For a quadratic equation px^2 - 16x + 5 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 64/5 -/
theorem quadratic_one_solution (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 16 * x + 5 = 0) ↔ p = 64/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2416_241622


namespace NUMINAMATH_CALUDE_smallest_n_for_square_and_cube_l2416_241610

theorem smallest_n_for_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 7 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 5 * x = y^2) → 
    (∃ (z : ℕ), 7 * x = z^3) → 
    x ≥ 245) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_and_cube_l2416_241610


namespace NUMINAMATH_CALUDE_slope_product_theorem_l2416_241639

theorem slope_product_theorem (m n : ℝ) (θ₁ θ₂ : ℝ) : 
  θ₁ = 3 * θ₂ →
  m = 9 * n →
  m ≠ 0 →
  m = Real.tan θ₁ →
  n = Real.tan θ₂ →
  m * n = 27 / 13 :=
by sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l2416_241639


namespace NUMINAMATH_CALUDE_integral_always_positive_l2416_241655

-- Define a continuous function f that is always positive
variable (f : ℝ → ℝ)
variable (hf : Continuous f)
variable (hfpos : ∀ x, f x > 0)

-- Define the integral bounds
variable (a b : ℝ)
variable (hab : a < b)

-- Theorem statement
theorem integral_always_positive :
  ∫ x in a..b, f x > 0 := by sorry

end NUMINAMATH_CALUDE_integral_always_positive_l2416_241655


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l2416_241687

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (x y : ℤ), 24 * x + 16 * y = n ∨ 24 * x + 16 * y < 0 ∨ 24 * x + 16 * y > n) ∧ 
  (∃ (a b : ℤ), 24 * a + 16 * b = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l2416_241687


namespace NUMINAMATH_CALUDE_percentage_difference_l2416_241648

theorem percentage_difference (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 1110 →
  z = 300 →
  (y - z) / z = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2416_241648


namespace NUMINAMATH_CALUDE_stratified_sampling_school_a_l2416_241613

theorem stratified_sampling_school_a (total_sample : ℕ) 
  (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) : 
  total_sample = 90 → 
  school_a = 3600 → 
  school_b = 5400 → 
  school_c = 1800 → 
  (school_a * total_sample) / (school_a + school_b + school_c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_a_l2416_241613


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2416_241676

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.sin (α + π/6) + 2 * Real.sin (α/2)^2 = 1 - Real.sqrt 2 / 2) : 
  Real.sin (α - π/6) = - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2416_241676


namespace NUMINAMATH_CALUDE_square_difference_equality_l2416_241614

theorem square_difference_equality : 1012^2 - 992^2 - 1008^2 + 996^2 = 16032 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2416_241614


namespace NUMINAMATH_CALUDE_range_of_a_l2416_241636

/-- A function that is monotonically increasing on an interval -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

/-- The definition of a hyperbola -/
def IsHyperbola (a : ℝ) : Prop :=
  2 * a^2 - 3 * a - 2 < 0

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (MonotonicallyIncreasing f 1 2) ∧
  (IsHyperbola a) →
  -1/2 < a ∧ a ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2416_241636


namespace NUMINAMATH_CALUDE_optimization_problem_l2416_241689

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 2 * x₀ * y₀ = 1/4) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 2 * x₁ * y₁ ≤ 1/4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 4 * x₀^2 + y₀^2 = 1/2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 4 * x₁^2 + y₁^2 ≥ 1/2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2 * Real.sqrt 2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 1/x₁ + 1/y₁ ≥ 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_optimization_problem_l2416_241689


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_fourth_power_equation_l2416_241679

theorem no_integer_solutions_for_fourth_power_equation :
  ¬ ∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_fourth_power_equation_l2416_241679


namespace NUMINAMATH_CALUDE_derangement_probability_five_l2416_241621

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of a derangement of n elements -/
def derangementProbability (n : ℕ) : ℚ :=
  (derangement n : ℚ) / (Nat.factorial n)

theorem derangement_probability_five :
  derangementProbability 5 = 11 / 30 := by sorry

end NUMINAMATH_CALUDE_derangement_probability_five_l2416_241621


namespace NUMINAMATH_CALUDE_probability_two_red_one_green_l2416_241663

def total_marbles : ℕ := 4 + 5 + 3 + 2

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_marbles 2 * Nat.choose green_marbles 1) / Nat.choose total_marbles marbles_drawn = 15 / 182 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_one_green_l2416_241663


namespace NUMINAMATH_CALUDE_certain_number_problem_l2416_241688

theorem certain_number_problem (N : ℚ) :
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 150 → N = 288 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2416_241688


namespace NUMINAMATH_CALUDE_total_results_l2416_241682

theorem total_results (average : ℝ) (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) :
  average = 24 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 228 →
  ∃ (n : ℕ), n = 25 ∧ (12 * first_12_avg + result_13 + 12 * last_12_avg) / n = average :=
by
  sorry


end NUMINAMATH_CALUDE_total_results_l2416_241682


namespace NUMINAMATH_CALUDE_equation_solution_l2416_241631

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x + 6) / (x - 3) = 5 / 2 ↔ x = 9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2416_241631


namespace NUMINAMATH_CALUDE_olly_owns_three_dogs_l2416_241656

/-- The number of shoes needed for each animal -/
def shoes_per_animal : ℕ := 4

/-- The total number of shoes needed -/
def total_shoes : ℕ := 24

/-- The number of cats Olly owns -/
def num_cats : ℕ := 2

/-- The number of ferrets Olly owns -/
def num_ferrets : ℕ := 1

/-- Calculates the number of dogs Olly owns -/
def num_dogs : ℕ :=
  (total_shoes - (num_cats + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem olly_owns_three_dogs : num_dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_olly_owns_three_dogs_l2416_241656


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2416_241601

theorem x_intercept_of_line (x y : ℝ) : 
  (5 * x - 2 * y - 10 = 0) → (y = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2416_241601


namespace NUMINAMATH_CALUDE_number_division_problem_l2416_241617

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → 
  (x - 14) / 10 = 4 → 
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2416_241617


namespace NUMINAMATH_CALUDE_keith_score_l2416_241653

theorem keith_score (keith larry danny : ℕ) 
  (larry_score : larry = 3 * keith)
  (danny_score : danny = larry + 5)
  (total_score : keith + larry + danny = 26) :
  keith = 3 := by
sorry

end NUMINAMATH_CALUDE_keith_score_l2416_241653


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2416_241672

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  (∃ q1 : ℤ, 4351 = d * q1 + 8) ∧ 
  (∃ r1 : ℤ, 5161 = d * r1 + 10) ∧ 
  (∀ n : ℕ, n > d → 
    (∃ q2 : ℤ, 4351 = n * q2 + 8) ∧ 
    (∃ r2 : ℤ, 5161 = n * r2 + 10) → n = d) → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2416_241672


namespace NUMINAMATH_CALUDE_batsman_highest_score_l2416_241686

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 140)
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = 
      ((total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score) ∧
    highest_score = 174 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l2416_241686


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2416_241611

open Real

theorem negation_of_existence_proposition :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 1 + sin x₀ = -x₀^2) ↔
  (∀ x : ℝ, x > 0 → 1 + sin x ≠ -x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2416_241611


namespace NUMINAMATH_CALUDE_spider_total_distance_l2416_241681

def spider_movement (start : ℤ) (first_move : ℤ) (second_move : ℤ) : ℕ :=
  (Int.natAbs (first_move - start)) + (Int.natAbs (second_move - first_move))

theorem spider_total_distance :
  spider_movement 3 (-4) 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_spider_total_distance_l2416_241681


namespace NUMINAMATH_CALUDE_distance_between_points_l2416_241674

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 3)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2416_241674


namespace NUMINAMATH_CALUDE_sum_c_d_eq_nine_l2416_241627

/-- A quadrilateral PQRS with specific vertex coordinates -/
structure Quadrilateral (c d : ℤ) :=
  (c_pos : c > 0)
  (d_pos : d > 0)
  (c_gt_d : c > d)

/-- The area of the quadrilateral PQRS -/
def area (q : Quadrilateral c d) : ℝ := 2 * ((c : ℝ)^2 - (d : ℝ)^2)

theorem sum_c_d_eq_nine {c d : ℤ} (q : Quadrilateral c d) (h : area q = 18) :
  c + d = 9 := by
  sorry

#check sum_c_d_eq_nine

end NUMINAMATH_CALUDE_sum_c_d_eq_nine_l2416_241627


namespace NUMINAMATH_CALUDE_museum_revenue_l2416_241646

def minutes_between (start_hour start_min end_hour end_min : ℕ) : ℕ :=
  (end_hour - start_hour) * 60 + end_min - start_min

def total_intervals (interval_length : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / interval_length

def total_people (people_per_interval intervals : ℕ) : ℕ :=
  people_per_interval * intervals

def student_tickets (total_people : ℕ) : ℕ :=
  total_people / 4

def regular_tickets (student_tickets : ℕ) : ℕ :=
  3 * student_tickets

def total_revenue (student_tickets regular_tickets : ℕ) (student_price regular_price : ℕ) : ℕ :=
  student_tickets * student_price + regular_tickets * regular_price

theorem museum_revenue : 
  let total_mins := minutes_between 9 0 17 55
  let intervals := total_intervals 5 total_mins
  let total_ppl := total_people 30 intervals
  let students := student_tickets total_ppl
  let regulars := regular_tickets students
  total_revenue students regulars 4 8 = 22456 := by
  sorry

end NUMINAMATH_CALUDE_museum_revenue_l2416_241646


namespace NUMINAMATH_CALUDE_diego_martha_can_ratio_l2416_241660

theorem diego_martha_can_ratio :
  let martha_cans : ℕ := 90
  let total_needed : ℕ := 150
  let more_needed : ℕ := 5
  let total_collected : ℕ := total_needed - more_needed
  let diego_cans : ℕ := total_collected - martha_cans
  (diego_cans : ℚ) / martha_cans = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_diego_martha_can_ratio_l2416_241660


namespace NUMINAMATH_CALUDE_M_remainder_500_l2416_241642

/-- The greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ := 8765432190

/-- M is a multiple of 9 -/
axiom M_multiple_of_9 : M % 9 = 0

/-- No two digits of M are the same -/
axiom M_unique_digits : ∀ i j, i ≠ j → (M / 10^i) % 10 ≠ (M / 10^j) % 10

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, n % 9 = 0 → (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) → n ≤ M

theorem M_remainder_500 : M % 500 = 190 := by
  sorry

end NUMINAMATH_CALUDE_M_remainder_500_l2416_241642


namespace NUMINAMATH_CALUDE_monkey_reaches_top_monkey_reaches_top_in_19_minutes_l2416_241675

def pole_height : ℕ := 10
def ascend_distance : ℕ := 2
def slip_distance : ℕ := 1

def monkey_position (minutes : ℕ) : ℕ :=
  let full_cycles := minutes / 2
  let remainder := minutes % 2
  if remainder = 0 then
    full_cycles * (ascend_distance - slip_distance)
  else
    full_cycles * (ascend_distance - slip_distance) + ascend_distance

theorem monkey_reaches_top :
  ∃ (minutes : ℕ), monkey_position minutes ≥ pole_height ∧
                   ∀ (m : ℕ), m < minutes → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

theorem monkey_reaches_top_in_19_minutes :
  monkey_position 19 ≥ pole_height ∧
  ∀ (m : ℕ), m < 19 → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_monkey_reaches_top_monkey_reaches_top_in_19_minutes_l2416_241675


namespace NUMINAMATH_CALUDE_sin_neg_pi_l2416_241615

theorem sin_neg_pi : Real.sin (-π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_pi_l2416_241615


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2416_241680

/-- Given a hyperbola with asymptote equations y = ± (1/3)x and one focus at (√10, 0),
    its standard equation is x²/9 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ y = k * x / 3 ∨ y = -k * x / 3) →  -- asymptote equations
  (∃ (c : ℝ), c^2 = 10 ∧ (c, 0) ∈ {p : ℝ × ℝ | p.1^2 / 9 - p.2^2 = 1}) →  -- focus condition
  (x^2 / 9 - y^2 = 1) :=  -- standard equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2416_241680


namespace NUMINAMATH_CALUDE_walk_distance_proof_l2416_241654

def walk_duration : ℝ := 5
def min_speed : ℝ := 3
def max_speed : ℝ := 4

def possible_distance (d : ℝ) : Prop :=
  ∃ (speed : ℝ), min_speed ≤ speed ∧ speed ≤ max_speed ∧ d = speed * walk_duration

theorem walk_distance_proof :
  possible_distance 19 ∧
  ¬ possible_distance 12 ∧
  ¬ possible_distance 14 ∧
  ¬ possible_distance 24 ∧
  ¬ possible_distance 35 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l2416_241654


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2416_241637

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (1 - a) * x > 1 - a ↔ x < 1) → a > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2416_241637


namespace NUMINAMATH_CALUDE_unique_zero_location_l2416_241666

def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem unique_zero_location (f : ℝ → ℝ) :
  has_unique_zero_in f 0 16 ∧
  has_unique_zero_in f 0 8 ∧
  has_unique_zero_in f 0 6 ∧
  has_unique_zero_in f 2 4 →
  ¬ ∃ x, 0 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_location_l2416_241666


namespace NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l2416_241665

/-- Calculates the profit per meter of cloth given the total length sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_length : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  (total_selling_price - total_length * cost_price_per_meter) / total_length

/-- Proves that for the given cloth sale, the profit per meter is 25 rupees. -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 85 8925 80 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l2416_241665


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_six_l2416_241649

-- Define the polynomials
def p (x : ℝ) : ℝ := -3*x^3 - 8*x^2 + 3*x + 2
def q (x : ℝ) : ℝ := -2*x^2 - 7*x - 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem statement
theorem coefficient_of_x_cubed_is_six :
  ∃ (a b c d : ℝ), product = fun x ↦ 6*x^3 + a*x^2 + b*x + c + d*x^4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_six_l2416_241649


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l2416_241667

theorem largest_gold_coins_distribution (total : ℕ) : 
  (∃ (k : ℕ), total = 13 * k + 3) →
  total < 150 →
  (∀ n : ℕ, (∃ (k : ℕ), n = 13 * k + 3) → n < 150 → n ≤ total) →
  total = 146 :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l2416_241667


namespace NUMINAMATH_CALUDE_total_red_balloons_l2416_241632

/-- The total number of red balloons Fred, Sam, and Dan have is 72. -/
theorem total_red_balloons : 
  let fred_balloons : ℕ := 10
  let sam_balloons : ℕ := 46
  let dan_balloons : ℕ := 16
  fred_balloons + sam_balloons + dan_balloons = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l2416_241632


namespace NUMINAMATH_CALUDE_boat_problem_l2416_241662

theorem boat_problem (boat1 boat2 boat3 boat4 boat5 : ℕ) 
  (h1 : boat1 = 2)
  (h2 : boat2 = 4)
  (h3 : boat3 = 3)
  (h4 : boat4 = 5)
  (h5 : boat5 = 6) :
  boat5 - (boat1 + boat2 + boat3 + boat4 + boat5) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_problem_l2416_241662


namespace NUMINAMATH_CALUDE_remaining_money_after_bike_purchase_l2416_241668

/-- Calculates the remaining money after buying a bike with quarters from jars --/
theorem remaining_money_after_bike_purchase (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℕ) : 
  num_jars = 5 → 
  quarters_per_jar = 160 → 
  bike_cost = 180 → 
  (num_jars * quarters_per_jar * 25 - bike_cost * 100) / 100 = 20 := by
  sorry

#check remaining_money_after_bike_purchase

end NUMINAMATH_CALUDE_remaining_money_after_bike_purchase_l2416_241668


namespace NUMINAMATH_CALUDE_no_inscribed_parallelepiped_l2416_241659

theorem no_inscribed_parallelepiped (π : ℝ) (h_π : π = Real.pi) :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x * y * z = 2 * π / 3 ∧
    x * y + y * z + z * x = π ∧
    x^2 + y^2 + z^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_parallelepiped_l2416_241659


namespace NUMINAMATH_CALUDE_soccer_balls_count_l2416_241629

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The cost of 2 footballs and some soccer balls in dollars -/
def first_set_cost : ℝ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℝ := 155

/-- The number of soccer balls in the second set -/
def soccer_balls_in_second_set : ℕ := 1

theorem soccer_balls_count : 
  2 * football_cost + soccer_balls_in_second_set * soccer_ball_cost = first_set_cost ∧
  3 * football_cost + soccer_ball_cost = second_set_cost →
  soccer_balls_in_second_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l2416_241629


namespace NUMINAMATH_CALUDE_numeria_base_l2416_241670

theorem numeria_base (s : ℕ) : s > 1 →
  (s^3 - 8*s^2 - 9*s + 1 = 0) →
  (2*s*(s - 4) = 0) →
  s = 4 := by
sorry

end NUMINAMATH_CALUDE_numeria_base_l2416_241670


namespace NUMINAMATH_CALUDE_sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l2416_241669

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be odd
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define a function that sums a list of integers
def SumList (list : List ℤ) : ℤ := list.foldl (· + ·) 0

-- Theorem 1: Sum of an even number of even integers is even
theorem sum_even_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 2: Sum of an even number of odd integers is even
theorem sum_even_number_of_odd_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsEven (SumList list) := by sorry

-- Theorem 3: Sum of an odd number of even integers is even
theorem sum_odd_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 4: Sum of an odd number of odd integers is odd
theorem sum_odd_number_of_odd_is_odd (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsOdd (SumList list) := by sorry

end NUMINAMATH_CALUDE_sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l2416_241669


namespace NUMINAMATH_CALUDE_books_read_l2416_241612

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (books_to_read : ℕ) (h1 : total_books = 22) (h2 : books_to_read = 10) :
  total_books - books_to_read = 12 := by
  sorry

#check books_read

end NUMINAMATH_CALUDE_books_read_l2416_241612


namespace NUMINAMATH_CALUDE_unique_solution_for_b_l2416_241658

def base_75_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (75 ^ i)) 0

theorem unique_solution_for_b : ∃! b : ℕ, 
  0 ≤ b ∧ b ≤ 19 ∧ 
  (base_75_to_decimal [9, 2, 4, 6, 1, 8, 7, 2, 5] - b) % 17 = 0 ∧
  b = 8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_b_l2416_241658


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2416_241609

theorem sum_of_fractions_equals_one 
  (a b c : ℝ) 
  (h : a * b * c = 1) : 
  (a / (a * b + a + 1)) + (b / (b * c + b + 1)) + (c / (c * a + c + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2416_241609


namespace NUMINAMATH_CALUDE_lower_class_students_l2416_241684

/-- Proves that in a school with 120 total students, where the lower class has 36 more students than the upper class, the number of students in the lower class is 78. -/
theorem lower_class_students (total : ℕ) (upper : ℕ) (lower : ℕ) 
  (h1 : total = 120)
  (h2 : upper + lower = total)
  (h3 : lower = upper + 36) :
  lower = 78 := by
  sorry

end NUMINAMATH_CALUDE_lower_class_students_l2416_241684


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2416_241661

def a : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i • a i = 0) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2416_241661


namespace NUMINAMATH_CALUDE_coprimality_preserving_polynomials_l2416_241635

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that a polynomial preserves coprimality -/
def PreservesCoprimality (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, Int.gcd a b = 1 → Int.gcd (P.eval a) (P.eval b) = 1

/-- Characterization of polynomials that preserve coprimality -/
theorem coprimality_preserving_polynomials :
  ∀ P : IntPolynomial,
  PreservesCoprimality P ↔
  (∃ n : ℕ, P = Polynomial.monomial n 1) ∨
  (∃ n : ℕ, P = Polynomial.monomial n (-1)) :=
sorry

end NUMINAMATH_CALUDE_coprimality_preserving_polynomials_l2416_241635


namespace NUMINAMATH_CALUDE_second_number_calculation_l2416_241678

theorem second_number_calculation (A B : ℕ) (h1 : A - B = 88) (h2 : A = 110) : B = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l2416_241678


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2416_241643

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 2 → 
  w = -x - u → 
  Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.mk 2 (-1) → 
  v + z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2416_241643


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2416_241651

/-- Represents the configuration of rectangles around a square -/
structure RectangleConfiguration where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem statement -/
theorem rectangle_ratio_theorem (config : RectangleConfiguration) :
  (config.inner_square_side > 0) →
  (config.rectangle_short_side > 0) →
  (config.rectangle_long_side > 0) →
  (config.inner_square_side + 2 * config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side + config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side / config.rectangle_short_side = 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2416_241651


namespace NUMINAMATH_CALUDE_winProbA_le_two_p_squared_l2416_241634

/-- A tennis game between players A and B where A wins a point with probability p ≤ 1/2 -/
structure TennisGame where
  /-- The probability of player A winning a point -/
  p : ℝ
  /-- The condition that p is at most 1/2 -/
  h_p_le_half : p ≤ 1/2

/-- The probability of player A winning the entire game -/
def winProbA (game : TennisGame) : ℝ :=
  sorry

/-- Theorem stating that the probability of A winning is at most 2p² -/
theorem winProbA_le_two_p_squared (game : TennisGame) :
  winProbA game ≤ 2 * game.p^2 := by
  sorry

end NUMINAMATH_CALUDE_winProbA_le_two_p_squared_l2416_241634


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2416_241605

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n-1)

theorem geometric_sequence_sum (a r : ℝ) (h1 : a < 0) :
  let seq := geometric_sequence a r
  (seq 2 * seq 4 + 2 * seq 3 * seq 5 + seq 4 * seq 6 = 36) →
  (seq 3 + seq 5 = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2416_241605


namespace NUMINAMATH_CALUDE_glasses_purchase_price_l2416_241616

/-- The purchase price of the glasses in yuan -/
def purchase_price : ℝ := 80

/-- The selling price after the initial increase -/
def increased_price (x : ℝ) : ℝ := 10 * x

/-- The selling price after applying the discount -/
def discounted_price (x : ℝ) : ℝ := 0.5 * increased_price x

/-- The profit made from selling the glasses -/
def profit (x : ℝ) : ℝ := discounted_price x - 20 - x

theorem glasses_purchase_price :
  profit purchase_price = 300 :=
sorry

end NUMINAMATH_CALUDE_glasses_purchase_price_l2416_241616


namespace NUMINAMATH_CALUDE_albert_running_laps_l2416_241640

theorem albert_running_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99)
  (h2 : track_length = 9)
  (h3 : laps_run = 6) :
  (total_distance / track_length) - laps_run = 5 :=
by
  sorry

#eval (99 / 9) - 6  -- This should output 5

end NUMINAMATH_CALUDE_albert_running_laps_l2416_241640


namespace NUMINAMATH_CALUDE_cos_36_degrees_l2416_241638

theorem cos_36_degrees : Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l2416_241638


namespace NUMINAMATH_CALUDE_rectangle_to_tetrahedron_sphere_area_l2416_241677

/-- A rectangle ABCD with sides AB and BC -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- A tetrahedron formed by folding a rectangle along its diagonal -/
structure Tetrahedron where
  base : Rectangle

/-- The surface area of the circumscribed sphere of a tetrahedron -/
def circumscribed_sphere_area (t : Tetrahedron) : ℝ := sorry

theorem rectangle_to_tetrahedron_sphere_area 
  (r : Rectangle) 
  (h1 : r.AB = 8) 
  (h2 : r.BC = 6) : 
  circumscribed_sphere_area (Tetrahedron.mk r) = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_tetrahedron_sphere_area_l2416_241677


namespace NUMINAMATH_CALUDE_complete_graph_inequality_l2416_241693

/-- Given n points on a plane with no three collinear and some connected by line segments,
    N_k denotes the number of complete graphs of k points. -/
def N (n k : ℕ) : ℕ := sorry

theorem complete_graph_inequality (n : ℕ) (h_n : n > 1) :
  ∀ k ∈ Finset.range (n - 1) \ {0, 1},
  N n k ≠ 0 →
  (N n (k + 1) : ℝ) / (N n k) ≥ 
    (1 : ℝ) / ((k^2 : ℝ) - 1) * ((k^2 : ℝ) * (N n k) / (N n (k + 1)) - n) := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_inequality_l2416_241693


namespace NUMINAMATH_CALUDE_marias_score_is_correct_score_difference_average_score_correct_l2416_241664

/-- Maria's score in a game, given that it was 50 points more than John's and their average was 112 -/
def marias_score : ℕ := 137

/-- John's score in the game -/
def johns_score : ℕ := marias_score - 50

/-- The average score of Maria and John -/
def average_score : ℕ := 112

theorem marias_score_is_correct : marias_score = 137 := by
  sorry

theorem score_difference : marias_score = johns_score + 50 := by
  sorry

theorem average_score_correct : (marias_score + johns_score) / 2 = average_score := by
  sorry

end NUMINAMATH_CALUDE_marias_score_is_correct_score_difference_average_score_correct_l2416_241664
