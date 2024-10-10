import Mathlib

namespace exchange_rate_proof_l2032_203202

-- Define the given quantities
def jack_pounds : ℝ := 42
def jack_euros : ℝ := 11
def jack_yen : ℝ := 3000
def pounds_per_euro : ℝ := 2
def total_yen : ℝ := 9400

-- Define the exchange rate we want to prove
def yen_per_pound : ℝ := 100

-- Theorem statement
theorem exchange_rate_proof :
  (jack_pounds + jack_euros * pounds_per_euro) * yen_per_pound + jack_yen = total_yen :=
by sorry

end exchange_rate_proof_l2032_203202


namespace conference_hall_tables_l2032_203230

/-- Represents the number of tables in the conference hall -/
def num_tables : ℕ := 16

/-- Represents the number of stools per table -/
def stools_per_table : ℕ := 8

/-- Represents the number of chairs per table -/
def chairs_per_table : ℕ := 4

/-- Represents the number of legs per stool -/
def legs_per_stool : ℕ := 3

/-- Represents the number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per table -/
def legs_per_table : ℕ := 4

/-- Represents the total number of legs for all furniture -/
def total_legs : ℕ := 704

theorem conference_hall_tables :
  num_tables * (stools_per_table * legs_per_stool + 
                chairs_per_table * legs_per_chair + 
                legs_per_table) = total_legs :=
by sorry

end conference_hall_tables_l2032_203230


namespace complex_division_equality_l2032_203266

/-- Given that i is the imaginary unit, prove that (2 + 4i) / (1 + i) = 3 + i -/
theorem complex_division_equality : (2 + 4 * I) / (1 + I) = 3 + I := by sorry

end complex_division_equality_l2032_203266


namespace ice_cream_cost_theorem_l2032_203272

/-- Ice cream shop prices and orders -/
structure IceCreamShop where
  chocolate_price : ℝ
  vanilla_price : ℝ
  strawberry_price : ℝ
  mint_price : ℝ
  waffle_cone_price : ℝ
  chocolate_chips_price : ℝ
  fudge_price : ℝ
  whipped_cream_price : ℝ

/-- Calculate the cost of Pierre's order -/
def pierre_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.chocolate_price + shop.mint_price + shop.waffle_cone_price + shop.chocolate_chips_price

/-- Calculate the cost of Pierre's mother's order -/
def mother_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.vanilla_price + shop.strawberry_price + shop.mint_price + 
  shop.waffle_cone_price + shop.fudge_price + shop.whipped_cream_price

/-- The total cost of both orders -/
def total_cost (shop : IceCreamShop) : ℝ :=
  pierre_order_cost shop + mother_order_cost shop

/-- Theorem stating that the total cost is $21.65 -/
theorem ice_cream_cost_theorem (shop : IceCreamShop) 
  (h1 : shop.chocolate_price = 2.50)
  (h2 : shop.vanilla_price = 2.00)
  (h3 : shop.strawberry_price = 2.25)
  (h4 : shop.mint_price = 2.20)
  (h5 : shop.waffle_cone_price = 1.50)
  (h6 : shop.chocolate_chips_price = 1.00)
  (h7 : shop.fudge_price = 1.25)
  (h8 : shop.whipped_cream_price = 0.75) :
  total_cost shop = 21.65 := by
  sorry

end ice_cream_cost_theorem_l2032_203272


namespace geometric_series_common_ratio_l2032_203227

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/4
  let a₂ : ℚ := 28/9
  let a₃ : ℚ := 112/27
  let r : ℚ := a₂ / a₁
  r = 16/9 := by sorry

end geometric_series_common_ratio_l2032_203227


namespace choir_arrangement_l2032_203269

theorem choir_arrangement (total_members : ℕ) (num_rows : ℕ) (h1 : total_members = 51) (h2 : num_rows = 4) :
  ∃ (row : ℕ), row ≤ num_rows ∧ 13 ≤ (total_members / num_rows + (if row ≤ total_members % num_rows then 1 else 0)) :=
by sorry

end choir_arrangement_l2032_203269


namespace room_width_calculation_l2032_203210

/-- Given a rectangular room with length 5.5 m, and a total paving cost of 12375 at a rate of 600 per sq. meter, the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (width : ℝ) : 
  length = 5.5 → 
  total_cost = 12375 → 
  cost_per_sqm = 600 → 
  width = total_cost / (length * cost_per_sqm) → 
  width = 3.75 := by
sorry

#eval (12375 : Float) / (5.5 * 600) -- Evaluates to 3.75

end room_width_calculation_l2032_203210


namespace solve_linear_equation_l2032_203289

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 := by
  sorry

end solve_linear_equation_l2032_203289


namespace double_negation_l2032_203293

theorem double_negation (x : ℝ) : -(-x) = x := by
  sorry

end double_negation_l2032_203293


namespace no_solution_implies_m_equals_six_l2032_203220

theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) → m = 6 :=
by sorry

end no_solution_implies_m_equals_six_l2032_203220


namespace margo_walk_distance_l2032_203259

/-- Calculates the total distance walked given the time and speed for each direction -/
def totalDistanceWalked (timeToFriend timeFromFriend : ℚ) (speedToFriend speedFromFriend : ℚ) : ℚ :=
  timeToFriend * speedToFriend + timeFromFriend * speedFromFriend

theorem margo_walk_distance :
  let timeToFriend : ℚ := 15 / 60
  let timeFromFriend : ℚ := 25 / 60
  let speedToFriend : ℚ := 5
  let speedFromFriend : ℚ := 3
  totalDistanceWalked timeToFriend timeFromFriend speedToFriend speedFromFriend = 5 / 2 := by
  sorry

#eval totalDistanceWalked (15/60) (25/60) 5 3

end margo_walk_distance_l2032_203259


namespace distinct_arrangements_of_basic_l2032_203261

theorem distinct_arrangements_of_basic (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end distinct_arrangements_of_basic_l2032_203261


namespace ratio_of_numbers_l2032_203280

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end ratio_of_numbers_l2032_203280


namespace real_part_of_inverse_one_minus_z_squared_l2032_203283

/-- For a complex number z = re^(iθ) where |z| = r ≠ 1 and r > 0, 
    the real part of 1 / (1 - z^2) is (1 - r^2 cos(2θ)) / (1 - 2r^2 cos(2θ) + r^4) -/
theorem real_part_of_inverse_one_minus_z_squared 
  (z : ℂ) (r θ : ℝ) (h1 : z = r * Complex.exp (θ * Complex.I)) 
  (h2 : Complex.abs z = r) (h3 : r ≠ 1) (h4 : r > 0) : 
  (1 / (1 - z^2)).re = (1 - r^2 * Real.cos (2 * θ)) / (1 - 2 * r^2 * Real.cos (2 * θ) + r^4) := by
  sorry

end real_part_of_inverse_one_minus_z_squared_l2032_203283


namespace range_of_a_l2032_203286

/-- A function f(x) = -x^2 + 2ax, where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

/-- The theorem stating the range of a given the conditions on f -/
theorem range_of_a (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → f a x < f a y) →
  (∀ x y, x ∈ Set.Icc 2 3 → y ∈ Set.Icc 2 3 → x < y → f a x > f a y) →
  1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l2032_203286


namespace sum_difference_squares_l2032_203204

theorem sum_difference_squares (x y : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 10) 
  (h3 : x - y = 19) : 
  (x + y)^2 - (x - y)^2 = -261 := by
  sorry

end sum_difference_squares_l2032_203204


namespace quadratic_solution_base_n_l2032_203238

/-- Given an integer n > 8, if n is a solution of x^2 - ax + b = 0 where a in base-n is 21,
    then b in base-n is 101. -/
theorem quadratic_solution_base_n (n : ℕ) (a b : ℕ) (h1 : n > 8) 
  (h2 : n^2 - a*n + b = 0) (h3 : a = 2*n + 1) : 
  b = n^2 + n := by
  sorry

end quadratic_solution_base_n_l2032_203238


namespace parabola_hyperbola_intersection_l2032_203279

/-- Given a parabola y² = 2px (p > 0) with focus F, if its directrix intersects 
    the hyperbola y²/3 - x² = 1 at points M and N, and MF is perpendicular to NF, 
    then p = 2√3. -/
theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y^2 = 2*p*x) →  -- equation of parabola
  (∀ x y, y^2/3 - x^2 = 1) →  -- equation of hyperbola
  (M.1 = -p/2 ∧ N.1 = -p/2) →  -- M and N are on the directrix
  (M.2^2/3 - M.1^2 = 1 ∧ N.2^2/3 - N.1^2 = 1) →  -- M and N are on the hyperbola
  ((M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0) →  -- MF ⊥ NF
  p = 2 * Real.sqrt 3 := by
    sorry

end parabola_hyperbola_intersection_l2032_203279


namespace counterexample_exists_l2032_203294

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 4)) := by
  sorry

end counterexample_exists_l2032_203294


namespace expand_expression_l2032_203251

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := by
  sorry

end expand_expression_l2032_203251


namespace extreme_points_condition_l2032_203234

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^3 + 2*a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 4*a*x + 1

-- Theorem statement
theorem extreme_points_condition (a x₁ x₂ : ℝ) : 
  (f_derivative a x₁ = 0) →  -- x₁ is an extreme point
  (f_derivative a x₂ = 0) →  -- x₂ is an extreme point
  (x₂ - x₁ = 2) →            -- Given condition
  (a^2 = 3) :=               -- Conclusion to prove
by sorry

end extreme_points_condition_l2032_203234


namespace max_product_constrained_sum_l2032_203284

theorem max_product_constrained_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  x * y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ a * b = 4 := by
  sorry

end max_product_constrained_sum_l2032_203284


namespace special_ten_digit_count_l2032_203274

/-- A natural number is special if all its digits are different or if changing one digit results in all digits being different. -/
def IsSpecial (n : ℕ) : Prop := sorry

/-- The count of 10-digit numbers. -/
def TenDigitCount : ℕ := 9000000000

/-- The count of special 10-digit numbers. -/
def SpecialTenDigitCount : ℕ := sorry

theorem special_ten_digit_count :
  SpecialTenDigitCount = 414 * Nat.factorial 9 := by sorry

end special_ten_digit_count_l2032_203274


namespace top_three_average_score_l2032_203250

theorem top_three_average_score (total_students : ℕ) (top_students : ℕ) 
  (class_average : ℝ) (score_difference : ℝ) : 
  total_students = 12 →
  top_students = 3 →
  class_average = 85 →
  score_difference = 8 →
  let other_students := total_students - top_students
  let top_average := (total_students * class_average - other_students * (class_average - score_difference)) / top_students
  top_average = 91 := by sorry

end top_three_average_score_l2032_203250


namespace farmer_seeds_total_l2032_203237

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_wednesday + seeds_thursday

theorem farmer_seeds_total :
  total_seeds = 22 :=
by sorry

end farmer_seeds_total_l2032_203237


namespace water_cup_fills_l2032_203224

theorem water_cup_fills (container_volume : ℚ) (cup_volume : ℚ) : 
  container_volume = 13/3 → cup_volume = 1/6 → 
  (container_volume / cup_volume : ℚ) = 26 := by
  sorry

end water_cup_fills_l2032_203224


namespace ball_hitting_ground_time_l2032_203242

theorem ball_hitting_ground_time : 
  let f (t : ℝ) := -4.9 * t^2 + 4.5 * t + 6
  ∃ t : ℝ, t > 0 ∧ f t = 0 ∧ t = 8121 / 4900 := by
  sorry

end ball_hitting_ground_time_l2032_203242


namespace cindys_envelopes_l2032_203295

/-- Cindy's envelope problem -/
theorem cindys_envelopes (initial_envelopes : ℕ) (friends : ℕ) (envelopes_per_friend : ℕ) :
  initial_envelopes = 37 →
  friends = 5 →
  envelopes_per_friend = 3 →
  initial_envelopes - friends * envelopes_per_friend = 22 := by
  sorry

#check cindys_envelopes

end cindys_envelopes_l2032_203295


namespace harkamal_fruit_purchase_l2032_203271

/-- The total amount paid by Harkamal for his fruit purchase --/
def total_amount : ℕ := by sorry

theorem harkamal_fruit_purchase :
  let grapes_quantity : ℕ := 8
  let grapes_price : ℕ := 80
  let mangoes_quantity : ℕ := 9
  let mangoes_price : ℕ := 55
  let apples_quantity : ℕ := 6
  let apples_price : ℕ := 120
  let oranges_quantity : ℕ := 4
  let oranges_price : ℕ := 75
  total_amount = grapes_quantity * grapes_price +
                 mangoes_quantity * mangoes_price +
                 apples_quantity * apples_price +
                 oranges_quantity * oranges_price :=
by sorry

end harkamal_fruit_purchase_l2032_203271


namespace complex_modulus_problem_l2032_203209

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) / (1 + Complex.I) →
  z.re = -2 →
  Complex.abs z = Real.sqrt 13 :=
sorry

end complex_modulus_problem_l2032_203209


namespace straight_insertion_sort_four_steps_l2032_203231

def initial_sequence : List Int := [7, 1, 3, 12, 8, 4, 9, 10]

def straight_insertion_sort (list : List Int) : List Int :=
  sorry

def first_four_steps (list : List Int) : List Int :=
  (straight_insertion_sort list).take 4

theorem straight_insertion_sort_four_steps :
  first_four_steps initial_sequence = [1, 3, 4, 7, 8, 12, 9, 10] :=
sorry

end straight_insertion_sort_four_steps_l2032_203231


namespace cheese_problem_l2032_203292

theorem cheese_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (cheese_first_night : ℚ) / total_rats = 2 * ((1 : ℚ) / total_rats) →
  (∃ (original_cheese : ℕ), original_cheese = cheese_first_night + 1) :=
by
  sorry

#check cheese_problem

end cheese_problem_l2032_203292


namespace number_exceeding_percentage_l2032_203207

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end number_exceeding_percentage_l2032_203207


namespace garden_perimeter_l2032_203257

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  8 * garden_length = 16 * 12 →
  2 * (garden_length + 8) = 64 := by
sorry

end garden_perimeter_l2032_203257


namespace complex_equation_imaginary_part_l2032_203288

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (1 + Complex.I) / (3 * Complex.I + z) = Complex.I →
  z.im = -4 := by
sorry

end complex_equation_imaginary_part_l2032_203288


namespace xyz_inequality_l2032_203205

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : x*y*z*(x + y + z) ≤ 1/3 := by
  sorry

end xyz_inequality_l2032_203205


namespace committee_formation_count_l2032_203298

/-- The number of ways to form a committee with specific requirements -/
theorem committee_formation_count : ∀ (n m k : ℕ),
  n ≥ m ∧ m ≥ k ∧ k ≥ 2 →
  (Nat.choose (n - 2) (k - 2) : ℕ) = Nat.choose n m →
  n = 12 ∧ m = 5 ∧ k = 3 →
  Nat.choose (n - 2) (k - 2) = 120 := by
  sorry

#check committee_formation_count

end committee_formation_count_l2032_203298


namespace total_apples_in_pile_l2032_203229

def initial_apples : ℕ := 8
def added_apples : ℕ := 5
def package_size : ℕ := 11

theorem total_apples_in_pile :
  initial_apples + added_apples = 13 := by
  sorry

end total_apples_in_pile_l2032_203229


namespace simplify_fraction_product_l2032_203208

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_product_l2032_203208


namespace problem_solution_l2032_203262

theorem problem_solution (m : ℚ) : 
  let f (x : ℚ) := 3 * x^3 - 1/x + 2
  let g (x : ℚ) := 2 * x^3 - 3*x + m
  let h (x : ℚ) := x^2
  (f 3 - g 3 + h 3 = 5) → m = 122/3 := by
sorry

end problem_solution_l2032_203262


namespace discriminant_of_quadratic_discriminant_of_specific_quadratic_l2032_203254

theorem discriminant_of_quadratic (a b c : ℝ) : 
  (a ≠ 0) → (b^2 - 4*a*c = (b^2 - 4*a*c)) := by sorry

theorem discriminant_of_specific_quadratic : 
  let a : ℝ := 4
  let b : ℝ := -9
  let c : ℝ := -15
  b^2 - 4*a*c = 321 := by sorry

end discriminant_of_quadratic_discriminant_of_specific_quadratic_l2032_203254


namespace train_length_specific_train_length_l2032_203249

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 144 km/hr crossing a point in 9.99920006399488 seconds has length approximately 399.97 meters -/
theorem specific_train_length : 
  ∃ (length : ℝ), abs (length - train_length 144 9.99920006399488) < 0.01 := by
  sorry

end train_length_specific_train_length_l2032_203249


namespace revenue_calculation_l2032_203228

/-- The revenue from a single sold-out performance for Steve's circus production -/
def revenue_per_performance : ℕ := sorry

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := 81000

/-- The production cost per performance for Steve's circus production -/
def production_cost_per_performance : ℕ := 7000

/-- The number of sold-out performances needed to break even -/
def performances_to_break_even : ℕ := 9

/-- Theorem stating that the revenue from a single sold-out performance is $16,000 -/
theorem revenue_calculation :
  revenue_per_performance = 16000 :=
by
  sorry

#check revenue_calculation

end revenue_calculation_l2032_203228


namespace cheese_calories_theorem_l2032_203270

/-- Calculates the remaining calories in a block of cheese -/
def remaining_calories (total_servings : ℕ) (calories_per_serving : ℕ) (eaten_servings : ℕ) : ℕ :=
  (total_servings - eaten_servings) * calories_per_serving

/-- Theorem: The remaining calories in a block of cheese with 16 servings, 
    where each serving contains 110 calories, and 5 servings have been eaten, 
    is equal to 1210 calories. -/
theorem cheese_calories_theorem : 
  remaining_calories 16 110 5 = 1210 := by
  sorry

end cheese_calories_theorem_l2032_203270


namespace sqrt_53_plus_20_sqrt_7_representation_l2032_203277

theorem sqrt_53_plus_20_sqrt_7_representation : 
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬ (∃ (k : ℕ), c = n^2 * k)) → 
    Real.sqrt (53 + 20 * Real.sqrt 7) = a + b * Real.sqrt c ∧ 
    a + b + c = 14 := by
  sorry

end sqrt_53_plus_20_sqrt_7_representation_l2032_203277


namespace base6_45_equals_29_l2032_203258

/-- Converts a base-6 number to decimal --/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The decimal representation of 45 in base 6 --/
def base6_45 : Nat := base6ToDecimal [5, 4]

theorem base6_45_equals_29 : base6_45 = 29 := by sorry

end base6_45_equals_29_l2032_203258


namespace jerry_candy_count_jerry_candy_count_proof_l2032_203299

theorem jerry_candy_count : ℕ → Prop :=
  fun total_candy : ℕ =>
    ∃ (candy_per_bag : ℕ),
      -- Total number of bags
      (9 : ℕ) * candy_per_bag = total_candy ∧
      -- Number of non-chocolate bags
      (9 - 2 - 3 : ℕ) * candy_per_bag = 28 ∧
      -- The result we want to prove
      total_candy = 63

-- The proof of the theorem
theorem jerry_candy_count_proof : jerry_candy_count 63 := by
  sorry

end jerry_candy_count_jerry_candy_count_proof_l2032_203299


namespace child_b_share_l2032_203215

theorem child_b_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_money = 900 → 
  ratio_a = 2 → 
  ratio_b = 3 → 
  ratio_c = 4 → 
  (ratio_b * total_money) / (ratio_a + ratio_b + ratio_c) = 300 := by
  sorry

end child_b_share_l2032_203215


namespace two_digit_number_problem_l2032_203264

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 * 10 + n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 * 10 + n.2 : ℚ) / 99

/-- The problem statement -/
theorem two_digit_number_problem (cd : TwoDigitNumber) :
  55 * (toRepeatingDecimal cd - toDecimal cd) = 1 → cd = (1, 8) := by
  sorry


end two_digit_number_problem_l2032_203264


namespace largest_reciprocal_l2032_203203

theorem largest_reciprocal (a b c d e : ℝ) 
  (ha : a = 1/4) 
  (hb : b = 3/7) 
  (hc : c = 0.25) 
  (hd : d = 7) 
  (he : e = 5000) : 
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by sorry

end largest_reciprocal_l2032_203203


namespace point_inside_circle_range_l2032_203244

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end point_inside_circle_range_l2032_203244


namespace temperature_conversion_l2032_203275

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by sorry

end temperature_conversion_l2032_203275


namespace triangle_area_l2032_203290

theorem triangle_area (A B C : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25) (h3 : Real.cos B = Real.sin A) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b + c) / 2 * r = 525 * Real.sqrt 2 / 2 :=
by sorry

end triangle_area_l2032_203290


namespace spending_difference_l2032_203285

/-- Represents the price of masks and the quantities purchased by Jiajia and Qiqi. -/
structure MaskPurchase where
  a : ℝ  -- Price of N95 mask in yuan
  b : ℝ  -- Price of regular medical mask in yuan
  jiajia_n95 : ℕ := 5  -- Number of N95 masks Jiajia bought
  jiajia_regular : ℕ := 2  -- Number of regular masks Jiajia bought
  qiqi_n95 : ℕ := 2  -- Number of N95 masks Qiqi bought
  qiqi_regular : ℕ := 5  -- Number of regular masks Qiqi bought

/-- The price difference between N95 and regular masks is 3 yuan. -/
def price_difference (m : MaskPurchase) : Prop :=
  m.a = m.b + 3

/-- The difference in spending between Jiajia and Qiqi is 9 yuan. -/
theorem spending_difference (m : MaskPurchase) 
  (h : price_difference m) : 
  (m.jiajia_n95 : ℝ) * m.a + (m.jiajia_regular : ℝ) * m.b - 
  ((m.qiqi_n95 : ℝ) * m.a + (m.qiqi_regular : ℝ) * m.b) = 9 := by
  sorry

end spending_difference_l2032_203285


namespace cyclic_quadrilateral_characterization_l2032_203233

/-- A quadrilateral is cyclic if and only if the sum of products of opposite angles equals π². -/
theorem cyclic_quadrilateral_characterization (α β γ δ : Real) 
  (h_angles : α + β + γ + δ = 2 * Real.pi) : 
  (α + γ = Real.pi ∧ β + δ = Real.pi) ↔ α * β + α * δ + γ * β + γ * δ = Real.pi ^ 2 := by
  sorry

end cyclic_quadrilateral_characterization_l2032_203233


namespace sum_of_reciprocals_inequality_l2032_203217

theorem sum_of_reciprocals_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1) + 1 / (5 * b^2 - 4 * b + 1) + 1 / (5 * c^2 - 4 * c + 1)) ≤ 1 / 4 := by
  sorry

end sum_of_reciprocals_inequality_l2032_203217


namespace sum_of_max_min_g_l2032_203232

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 2| + |x - 4| + |x - 6| - |2*x - 6|

-- Define the domain of x
def domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = 14 :=
sorry

end sum_of_max_min_g_l2032_203232


namespace farm_chickens_l2032_203291

/-- Represents the number of roosters initially on the farm. -/
def initial_roosters : ℕ := sorry

/-- Represents the number of hens initially on the farm. -/
def initial_hens : ℕ := 6 * initial_roosters

/-- Represents the number of roosters added to the farm. -/
def added_roosters : ℕ := 60

/-- Represents the number of hens added to the farm. -/
def added_hens : ℕ := 60

/-- Represents the total number of roosters after additions. -/
def final_roosters : ℕ := initial_roosters + added_roosters

/-- Represents the total number of hens after additions. -/
def final_hens : ℕ := initial_hens + added_hens

/-- States that after additions, the number of hens is 4 times the number of roosters. -/
axiom final_ratio : final_hens = 4 * final_roosters

/-- Represents the total number of chickens initially on the farm. -/
def total_chickens : ℕ := initial_roosters + initial_hens

/-- Proves that the total number of chickens initially on the farm was 630. -/
theorem farm_chickens : total_chickens = 630 := by sorry

end farm_chickens_l2032_203291


namespace tic_tac_toe_winning_probability_l2032_203282

/-- Represents a 3x3 tic-tac-toe board --/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The total number of cells on the board --/
def totalCells : Nat := 9

/-- The number of noughts on the board --/
def numNoughts : Nat := 3

/-- The number of crosses on the board --/
def numCrosses : Nat := 6

/-- The number of ways to arrange noughts on the board --/
def totalArrangements : Nat := Nat.choose totalCells numNoughts

/-- The number of winning positions for noughts --/
def winningPositions : Nat := 8

/-- The probability of noughts being in a winning position --/
def winningProbability : ℚ := winningPositions / totalArrangements

theorem tic_tac_toe_winning_probability :
  winningProbability = 2 / 21 := by sorry

end tic_tac_toe_winning_probability_l2032_203282


namespace expression_equals_negative_two_over_tan_l2032_203240

theorem expression_equals_negative_two_over_tan (α : Real) 
  (h : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  -2 / Real.tan α := by
  sorry

end expression_equals_negative_two_over_tan_l2032_203240


namespace cos_four_minus_sin_four_equals_cos_double_l2032_203287

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by sorry

end cos_four_minus_sin_four_equals_cos_double_l2032_203287


namespace value_between_seven_and_eight_l2032_203221

theorem value_between_seven_and_eight :
  7 < (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) ∧
  (3 * Real.sqrt 15 + 2 * Real.sqrt 5) * Real.sqrt (1/5) < 8 := by
  sorry

end value_between_seven_and_eight_l2032_203221


namespace max_value_interval_condition_l2032_203201

/-- The function f(x) = (1/3)x^3 - x has a maximum value on the interval (2m, 1-m) if and only if m ∈ [-1, -1/2). -/
theorem max_value_interval_condition (m : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Ioo (2*m) (1-m) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (2*m) (1-m) → 
      (1/3 * x^3 - x) ≥ (1/3 * y^3 - y))) ↔ 
  m ∈ Set.Icc (-1) (-1/2) := by
sorry

end max_value_interval_condition_l2032_203201


namespace wrong_to_correct_ratio_l2032_203239

theorem wrong_to_correct_ratio (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36)
  (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 := by
  sorry

end wrong_to_correct_ratio_l2032_203239


namespace betty_height_l2032_203213

theorem betty_height (dog_height : ℕ) (carter_height : ℕ) (betty_height_inches : ℕ) :
  dog_height = 24 →
  carter_height = 2 * dog_height →
  betty_height_inches = carter_height - 12 →
  betty_height_inches / 12 = 3 :=
by sorry

end betty_height_l2032_203213


namespace min_toothpicks_removal_l2032_203206

/-- Represents a triangular lattice structure made of toothpicks -/
structure TriangularLattice :=
  (toothpicks : ℕ)
  (triangles : ℕ)
  (horizontal_toothpicks : ℕ)

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (lattice : TriangularLattice) : ℕ :=
  lattice.horizontal_toothpicks

theorem min_toothpicks_removal (lattice : TriangularLattice) 
  (h1 : lattice.toothpicks = 40)
  (h2 : lattice.triangles > 40)
  (h3 : lattice.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove lattice = 15 := by
  sorry

end min_toothpicks_removal_l2032_203206


namespace soccer_league_games_times_each_team_plays_l2032_203253

/-- 
Proves that in a soccer league with 12 teams, where a total of 66 games are played, 
each team plays every other team exactly 2 times.
-/
theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  (n * (n - 1) * 2) / 2 = total_games :=
by sorry

/-- 
Proves that the number of times each team plays others is 2.
-/
theorem times_each_team_plays (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  ∃ x : ℕ, (n * (n - 1) * x) / 2 = total_games ∧ x = 2 :=
by sorry

end soccer_league_games_times_each_team_plays_l2032_203253


namespace leah_daily_earnings_l2032_203214

/-- Represents Leah's earnings over a period of time -/
structure Earnings where
  total : ℕ  -- Total earnings in dollars
  weeks : ℕ  -- Number of weeks worked
  daily : ℕ  -- Daily earnings in dollars

/-- Calculates the number of days in a given number of weeks -/
def daysInWeeks (weeks : ℕ) : ℕ :=
  7 * weeks

/-- Theorem: Leah's daily earnings are 60 dollars -/
theorem leah_daily_earnings (e : Earnings) (h1 : e.total = 1680) (h2 : e.weeks = 4) :
  e.daily = 60 := by
  sorry

end leah_daily_earnings_l2032_203214


namespace initial_marbles_l2032_203241

-- Define the variables
def marbles_given : ℕ := 14
def marbles_left : ℕ := 50

-- State the theorem
theorem initial_marbles : marbles_given + marbles_left = 64 := by
  sorry

end initial_marbles_l2032_203241


namespace prime_sum_equation_l2032_203256

theorem prime_sum_equation (a b n : ℕ) : 
  a < b ∧ 
  Nat.Prime a ∧ 
  Nat.Prime b ∧ 
  Odd n ∧ 
  a + b * n = 487 → 
  ((a = 2 ∧ b = 5 ∧ n = 97) ∨ (a = 2 ∧ b = 97 ∧ n = 5)) :=
by sorry

end prime_sum_equation_l2032_203256


namespace product_sum_of_three_numbers_l2032_203276

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 179)
  (sum_of_numbers : a + b + c = 21) :
  a*b + b*c + a*c = 131 := by
  sorry

end product_sum_of_three_numbers_l2032_203276


namespace movie_ticket_change_change_is_nine_l2032_203243

/-- The change received by two sisters after buying movie tickets -/
theorem movie_ticket_change (ticket_cost : ℕ) (money_brought : ℕ) : ℕ :=
  let num_sisters : ℕ := 2
  let total_cost : ℕ := num_sisters * ticket_cost
  money_brought - total_cost

/-- Proof that the change received is $9 -/
theorem change_is_nine :
  movie_ticket_change 8 25 = 9 := by
  sorry

end movie_ticket_change_change_is_nine_l2032_203243


namespace ramesh_profit_share_l2032_203297

/-- Calculates the share of profit for a partner in a business partnership -/
def calculateProfitShare (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  (investment2 * totalProfit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 -/
theorem ramesh_profit_share :
  calculateProfitShare 24000 40000 19000 = 11875 := by
  sorry

end ramesh_profit_share_l2032_203297


namespace boys_at_least_35_percent_l2032_203267

/-- Represents a child camp with 3-rooms and 4-rooms -/
structure ChildCamp where
  girls_3room : ℕ
  girls_4room : ℕ
  boys_3room : ℕ
  boys_4room : ℕ

/-- The proportion of boys in the camp -/
def boy_proportion (camp : ChildCamp) : ℚ :=
  (3 * camp.boys_3room + 4 * camp.boys_4room) / 
  (3 * camp.girls_3room + 4 * camp.girls_4room + 3 * camp.boys_3room + 4 * camp.boys_4room)

/-- Theorem stating that the proportion of boys is at least 35% -/
theorem boys_at_least_35_percent (camp : ChildCamp) 
  (h1 : 2 * (camp.girls_4room + camp.boys_4room) ≥ 
        camp.girls_3room + camp.girls_4room + camp.boys_3room + camp.boys_4room)
  (h2 : 3 * camp.girls_3room ≥ 8 * camp.girls_4room) :
  boy_proportion camp ≥ 7/20 := by
  sorry

end boys_at_least_35_percent_l2032_203267


namespace jack_bought_55_apples_l2032_203296

def apples_for_father : Nat := 10
def number_of_friends : Nat := 4
def apples_per_person : Nat := 9

theorem jack_bought_55_apples : 
  (apples_for_father + (number_of_friends + 1) * apples_per_person) = 55 := by
  sorry

end jack_bought_55_apples_l2032_203296


namespace total_ndfl_is_11050_l2032_203225

/-- Calculates the total NDFL (personal income tax) on income from securities --/
def calculate_ndfl (dividend_income : ℝ) (ofz_coupon_income : ℝ) (corporate_coupon_income : ℝ) 
  (shares_sold : ℕ) (sale_price_per_share : ℝ) (purchase_price_per_share : ℝ) 
  (dividend_tax_rate : ℝ) (corporate_coupon_tax_rate : ℝ) (capital_gains_tax_rate : ℝ) : ℝ :=
  let capital_gains := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let dividend_tax := dividend_income * dividend_tax_rate
  let corporate_coupon_tax := corporate_coupon_income * corporate_coupon_tax_rate
  let capital_gains_tax := capital_gains * capital_gains_tax_rate
  dividend_tax + corporate_coupon_tax + capital_gains_tax

/-- Theorem stating that the total NDFL on income from securities is 11,050 rubles --/
theorem total_ndfl_is_11050 :
  calculate_ndfl 50000 40000 30000 100 200 150 0.13 0.13 0.13 = 11050 := by
  sorry

end total_ndfl_is_11050_l2032_203225


namespace pear_percentage_difference_l2032_203268

/-- Proves that the percentage difference between canned and poached pears is 20% -/
theorem pear_percentage_difference (total pears_sold pears_canned pears_poached : ℕ) :
  total = 42 →
  pears_sold = 20 →
  pears_poached = pears_sold / 2 →
  total = pears_sold + pears_canned + pears_poached →
  (pears_canned - pears_poached : ℚ) / pears_poached * 100 = 20 := by
  sorry

end pear_percentage_difference_l2032_203268


namespace nested_g_equals_cos_fifteen_fourths_l2032_203236

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x / 2)

theorem nested_g_equals_cos_fifteen_fourths :
  0 < (1 : ℝ) / 2 ∧
  (∀ x : ℝ, 0 < x → 0 < g x) →
  g (g (g (g (g ((1 : ℝ) / 2) + 1) + 1) + 1) + 1) = Real.cos (15 / 4 * π / 180) :=
by sorry

end nested_g_equals_cos_fifteen_fourths_l2032_203236


namespace inequality_solution_existence_l2032_203226

theorem inequality_solution_existence (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end inequality_solution_existence_l2032_203226


namespace seeds_sown_count_l2032_203212

/-- The number of seeds that germinated -/
def seeds_germinated : ℕ := 970

/-- The frequency of normal seed germination -/
def germination_rate : ℚ := 97/100

/-- The total number of seeds sown -/
def total_seeds : ℕ := 1000

/-- Theorem stating that the total number of seeds sown is 1000 -/
theorem seeds_sown_count : 
  (seeds_germinated : ℚ) / germination_rate = total_seeds := by sorry

end seeds_sown_count_l2032_203212


namespace max_value_of_a_l2032_203252

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ (b : ℝ), (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) → 
              b ≤ -1) ∧
  (∃ (a : ℝ), a = -1 ∧ 
              (∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a)) :=
by sorry

end max_value_of_a_l2032_203252


namespace variance_scaling_l2032_203260

-- Define a function to calculate variance
def variance (data : List ℝ) : ℝ := sorry

-- Define a function to scale a list of real numbers
def scaleList (k : ℝ) (data : List ℝ) : List ℝ := sorry

theorem variance_scaling (data : List ℝ) (h : variance data = 0.01) :
  variance (scaleList 10 data) = 1 := by sorry

end variance_scaling_l2032_203260


namespace partner_a_profit_share_l2032_203235

/-- Calculates the share of profit for partner A in a business partnership --/
theorem partner_a_profit_share
  (initial_a initial_b : ℕ)
  (withdrawal_a addition_b : ℕ)
  (total_months : ℕ)
  (change_month : ℕ)
  (total_profit : ℕ)
  (h1 : initial_a = 2000)
  (h2 : initial_b = 4000)
  (h3 : withdrawal_a = 1000)
  (h4 : addition_b = 1000)
  (h5 : total_months = 12)
  (h6 : change_month = 8)
  (h7 : total_profit = 630) :
  let investment_months_a := initial_a * change_month + (initial_a - withdrawal_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + addition_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  let a_share := (investment_months_a * total_profit) / total_investment_months
  a_share = 175 := by sorry

end partner_a_profit_share_l2032_203235


namespace angle_ADB_is_270_degrees_l2032_203223

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry

def angleAIs45 (t : Triangle) : Prop := sorry

def angleBIs45 (t : Triangle) : Prop := sorry

-- Define the angle bisectors and their intersection
def angleBisectorA (t : Triangle) : Line := sorry

def angleBisectorB (t : Triangle) : Line := sorry

def D (t : Triangle) : Point := sorry

-- Define the measure of angle ADB
def measureAngleADB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_ADB_is_270_degrees (t : Triangle) :
  isRightTriangle t → angleAIs45 t → angleBIs45 t →
  measureAngleADB t = 270 :=
sorry

end angle_ADB_is_270_degrees_l2032_203223


namespace john_completion_time_l2032_203218

/-- The time it takes for John to complete the task alone -/
def john_time : ℝ := 20

/-- The time it takes for Jane to complete the task alone -/
def jane_time : ℝ := 10

/-- The total time they worked together -/
def total_time : ℝ := 10

/-- The time Jane worked before stopping -/
def jane_work_time : ℝ := 5

theorem john_completion_time :
  (jane_work_time * (1 / john_time + 1 / jane_time) + (total_time - jane_work_time) * (1 / john_time) = 1) →
  john_time = 20 := by
sorry

end john_completion_time_l2032_203218


namespace ellipse_properties_l2032_203219

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

-- Define the conditions
def is_valid_ellipse (e : Ellipse) : Prop :=
  ∃ c : ℝ, 
    e.a + c = Real.sqrt 2 + 1 ∧
    e.a = Real.sqrt 2 * c ∧
    e.a^2 = e.b^2 + c^2

-- Define the standard equation
def standard_equation (e : Ellipse) : Prop :=
  e.a^2 = 2 ∧ e.b^2 = 1

-- Define the line passing through the left focus
def line_through_focus (k : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, y t = k * (x t + 1)

-- Define the condition for the midpoint
def midpoint_on_line (x y : ℝ → ℝ) : Prop :=
  ∃ t₁ t₂, x ((t₁ + t₂)/2) + y ((t₁ + t₂)/2) = 0

-- The main theorem
theorem ellipse_properties (e : Ellipse) (h : is_valid_ellipse e) :
  standard_equation e ∧
  (∀ k x y, line_through_focus k x y → midpoint_on_line x y →
    (k = 0 ∨ k = 1/2)) :=
sorry

end ellipse_properties_l2032_203219


namespace possible_k_values_l2032_203245

theorem possible_k_values (a b k : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (b + 1 : ℚ) / a + (a + 1 : ℚ) / b = k →
  k = 3 ∨ k = 4 := by
sorry

end possible_k_values_l2032_203245


namespace power_of_power_l2032_203216

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2032_203216


namespace sum_equality_existence_l2032_203278

theorem sum_equality_existence (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_n : n > 3)
  (h_pos : ∀ i, a i > 0)
  (h_strict : ∀ i j, i < j → a i < a j)
  (h_upper : a (Fin.last n) ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i.succ + a j.succ = a k.succ + a l.succ ∧
    a i.succ + a j.succ = a m.succ :=
by sorry

end sum_equality_existence_l2032_203278


namespace two_designs_are_three_fifths_l2032_203255

/-- Represents a design with a shaded region --/
structure Design where
  shaded_fraction : Rat

/-- Checks if a given fraction is equal to 3/5 --/
def is_three_fifths (f : Rat) : Bool :=
  f = 3 / 5

/-- Counts the number of designs with shaded region equal to 3/5 --/
def count_three_fifths (designs : List Design) : Nat :=
  designs.filter (fun d => is_three_fifths d.shaded_fraction) |>.length

/-- The main theorem stating that exactly 2 out of 5 given designs have 3/5 shaded area --/
theorem two_designs_are_three_fifths :
  let designs : List Design := [
    ⟨3 / 8⟩,
    ⟨12 / 20⟩,
    ⟨2 / 3⟩,
    ⟨15 / 25⟩,
    ⟨4 / 8⟩
  ]
  count_three_fifths designs = 2 := by
  sorry


end two_designs_are_three_fifths_l2032_203255


namespace binomial_probability_one_third_l2032_203211

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_probability_one_third 
  (X : BinomialVariable) 
  (h_expectation : expectation X = 30)
  (h_variance : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end binomial_probability_one_third_l2032_203211


namespace paul_weekly_spending_l2032_203263

-- Define the given conditions
def lawn_money : ℕ := 3
def weed_eating_money : ℕ := 3
def weeks : ℕ := 2

-- Define the total money earned
def total_money : ℕ := lawn_money + weed_eating_money

-- Define the theorem to prove
theorem paul_weekly_spending :
  total_money / weeks = 3 := by
  sorry

end paul_weekly_spending_l2032_203263


namespace expression_simplification_l2032_203200

theorem expression_simplification :
  Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt 3 - 1| + Real.sqrt 3 = -13 / 4 + 2 * Real.sqrt 3 := by
  sorry

end expression_simplification_l2032_203200


namespace greatest_x_with_lcm_l2032_203247

theorem greatest_x_with_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 ∧ ∃ y : ℕ+, y = 105 ∧ Nat.lcm y (Nat.lcm 15 21) = 105 :=
by sorry

end greatest_x_with_lcm_l2032_203247


namespace expression_evaluation_l2032_203281

theorem expression_evaluation : 
  let f (x : ℚ) := (2 * x + 1) / (2 * x - 1)
  f 2 = 5 / 3 := by
sorry

end expression_evaluation_l2032_203281


namespace system_solutions_l2032_203248

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y^2 - 2 * y + 3 * x^2 = 0
def equation2 (x y : ℝ) : Prop := y^2 + x^2 * y + 2 * x = 0

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) := {(-1, 1), (-2 / Real.rpow 3 (1/3), -2 * Real.rpow 3 (1/3)), (0, 0)}

-- Theorem statement
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end system_solutions_l2032_203248


namespace geometric_sequence_common_ratio_l2032_203246

/-- For a geometric sequence with first term 1, if the first term, the sum of first two terms,
    and 5 form an arithmetic sequence, then the common ratio is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  (a 1 = 1) →  -- First term is 1
  (S 2 = a 1 + a 2) →  -- S_2 is the sum of first two terms
  (S 2 - a 1 = 5 - S 2) →  -- a_1, S_2, and 5 form an arithmetic sequence
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l2032_203246


namespace line_intercepts_l2032_203222

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-axis intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-axis intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The x-intercept and y-intercept of the line 3x - 2y - 6 = 0 are 2 and -3 respectively -/
theorem line_intercepts : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end line_intercepts_l2032_203222


namespace provisions_last_20_days_l2032_203265

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def daysAfterReinforcement (initialGarrison : ℕ) (initialDays : ℕ) (daysPassed : ℕ) (reinforcement : ℕ) : ℕ :=
  let remainingDays := initialDays - daysPassed
  let totalMen := initialGarrison + reinforcement
  (initialGarrison * remainingDays) / totalMen

/-- Theorem stating that given the initial conditions, the provisions will last 20 more days after reinforcement -/
theorem provisions_last_20_days :
  daysAfterReinforcement 2000 54 15 1900 = 20 := by
  sorry

#eval daysAfterReinforcement 2000 54 15 1900

end provisions_last_20_days_l2032_203265


namespace line_passes_through_quadrants_l2032_203273

theorem line_passes_through_quadrants (a b c p : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a + b) / c = p) (h5 : (b + c) / a = p) (h6 : (c + a) / b = p) :
  ∃ (x y : ℝ), x < 0 ∧ y = p * x + p ∧ y < 0 :=
sorry

end line_passes_through_quadrants_l2032_203273
