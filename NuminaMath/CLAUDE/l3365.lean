import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3365_336527

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5 ∧ 
  ¬(a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3365_336527


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3365_336558

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 2277 →
  a + b + c + d ≤ 84 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3365_336558


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3365_336566

theorem inequality_solution_set :
  let S : Set ℝ := {x | (3 - x) / (2 * x - 4) < 1}
  S = {x | x < 2 ∨ x > 7/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3365_336566


namespace NUMINAMATH_CALUDE_discontinuity_at_three_l3365_336579

/-- The function f(x) = 6 / (x-3)² is discontinuous at x = 3 -/
theorem discontinuity_at_three (f : ℝ → ℝ) (h : ∀ x ≠ 3, f x = 6 / (x - 3)^2) :
  ¬ ContinuousAt f 3 := by
  sorry

end NUMINAMATH_CALUDE_discontinuity_at_three_l3365_336579


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3365_336583

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : n > 0) (hodd : Odd n) :
  n ∣ 2^(n!) - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3365_336583


namespace NUMINAMATH_CALUDE_length_AB_l3365_336594

/-- A line passing through (2,0) with slope 2 -/
def line_l (x y : ℝ) : Prop := y = 2 * x - 4

/-- The curve y^2 - 4x = 0 -/
def curve (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A is on both the line and the curve -/
def point_A (x y : ℝ) : Prop := line_l x y ∧ curve x y

/-- Point B is on both the line and the curve, and is different from A -/
def point_B (x y : ℝ) : Prop := line_l x y ∧ curve x y ∧ (x, y) ≠ (1, -2)

/-- The main theorem: the length of AB is 3√5 -/
theorem length_AB :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_length_AB_l3365_336594


namespace NUMINAMATH_CALUDE_bumper_cars_cost_l3365_336511

/-- The cost of bumper cars given the costs of other attractions and ticket information -/
theorem bumper_cars_cost 
  (total_cost : ℕ → ℕ → ℕ)  -- Function to calculate total cost
  (current_tickets : ℕ)     -- Current number of tickets
  (additional_tickets : ℕ)  -- Additional tickets needed
  (ferris_wheel_cost : ℕ)   -- Cost of Ferris wheel
  (roller_coaster_cost : ℕ) -- Cost of roller coaster
  (h1 : current_tickets = 5)
  (h2 : additional_tickets = 8)
  (h3 : ferris_wheel_cost = 5)
  (h4 : roller_coaster_cost = 4)
  (h5 : ∀ x y, total_cost x y = x + y) -- Definition of total cost function
  : ∃ (bumper_cars_cost : ℕ), 
    total_cost current_tickets additional_tickets = 
    ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost ∧ 
    bumper_cars_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bumper_cars_cost_l3365_336511


namespace NUMINAMATH_CALUDE_bobs_muffin_cost_l3365_336547

/-- The cost of a single muffin for Bob -/
def muffin_cost (muffins_per_day : ℕ) (days_per_week : ℕ) (selling_price : ℚ) (weekly_profit : ℚ) : ℚ :=
  let total_muffins : ℕ := muffins_per_day * days_per_week
  let total_revenue : ℚ := (total_muffins : ℚ) * selling_price
  let total_cost : ℚ := total_revenue - weekly_profit
  total_cost / (total_muffins : ℚ)

/-- Theorem stating that Bob's muffin cost is $0.75 -/
theorem bobs_muffin_cost :
  muffin_cost 12 7 (3/2) 63 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_bobs_muffin_cost_l3365_336547


namespace NUMINAMATH_CALUDE_pythagorean_from_law_of_cosines_l3365_336504

/-- The law of cosines for a triangle with sides a, b, c and angle γ opposite side c -/
def lawOfCosines (a b c : ℝ) (γ : Real) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*Real.cos γ

/-- The Pythagorean theorem for a right triangle with sides a, b, c where c is the hypotenuse -/
def pythagoreanTheorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem stating that the Pythagorean theorem is a special case of the law of cosines -/
theorem pythagorean_from_law_of_cosines (a b c : ℝ) :
  lawOfCosines a b c (π/2) → pythagoreanTheorem a b c :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_from_law_of_cosines_l3365_336504


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3365_336560

theorem biased_coin_probability (p : ℝ) (n : ℕ) (h_p : p = 3/4) (h_n : n = 4) :
  1 - p^n = 175/256 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3365_336560


namespace NUMINAMATH_CALUDE_unique_sums_count_l3365_336510

def bag_x : Finset ℕ := {1, 4, 7}
def bag_y : Finset ℕ := {3, 5, 8}

def possible_sums : Finset ℕ := (bag_x.product bag_y).image (λ (x, y) => x + y)

theorem unique_sums_count : possible_sums.card = 7 := by sorry

end NUMINAMATH_CALUDE_unique_sums_count_l3365_336510


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l3365_336559

theorem walking_rate_ratio (usual_time new_time : ℝ) 
  (h1 : usual_time = 16)
  (h2 : new_time = usual_time - 4) :
  new_time / usual_time = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l3365_336559


namespace NUMINAMATH_CALUDE_prob_all_heads_or_five_plus_tails_is_one_eighth_l3365_336588

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The probability of getting heads on a single fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails on a single fair coin flip -/
def p_tails : ℚ := 1/2

/-- The probability of getting all heads or at least five tails when flipping six fair coins -/
def prob_all_heads_or_five_plus_tails : ℚ := 1/8

/-- Theorem stating that the probability of getting all heads or at least five tails 
    when flipping six fair coins is 1/8 -/
theorem prob_all_heads_or_five_plus_tails_is_one_eighth :
  prob_all_heads_or_five_plus_tails = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_heads_or_five_plus_tails_is_one_eighth_l3365_336588


namespace NUMINAMATH_CALUDE_alcohol_mixture_theorem_l3365_336584

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_theorem :
  let x_volume : ℝ := 300
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 100
  let y_concentration : ℝ := 0.30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = 0.15 := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_theorem_l3365_336584


namespace NUMINAMATH_CALUDE_exponent_calculation_l3365_336577

theorem exponent_calculation (a : ℝ) : a^3 * a * a^4 + (-3 * a^4)^2 = 10 * a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3365_336577


namespace NUMINAMATH_CALUDE_tangent_circle_problem_l3365_336539

theorem tangent_circle_problem (center_to_intersection : ℚ) (radius : ℚ) (center_to_line : ℚ) (x : ℚ) :
  center_to_intersection = 3/8 →
  radius = 3/16 →
  center_to_line = 1/2 →
  x = center_to_intersection + radius - center_to_line →
  x = 1/16 := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_problem_l3365_336539


namespace NUMINAMATH_CALUDE_telescope_purchase_sum_l3365_336571

/-- The sum of Joan and Karl's telescope purchases -/
def sum_of_purchases (joan_price karl_price : ℕ) : ℕ :=
  joan_price + karl_price

/-- Theorem stating the sum of Joan and Karl's telescope purchases -/
theorem telescope_purchase_sum :
  ∀ (joan_price karl_price : ℕ),
    joan_price = 158 →
    2 * joan_price = karl_price + 74 →
    sum_of_purchases joan_price karl_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_telescope_purchase_sum_l3365_336571


namespace NUMINAMATH_CALUDE_polynomial_equality_l3365_336586

theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 10 * x) = 
    (12 * x^5 + 6 * x^4 + 28 * x^3 + 30 * x^2 + 3 * x + 2)) →
  (∀ x, q x = -2 * x^6 + 12 * x^5 + 2 * x^4 + 23 * x^3 + 30 * x^2 - 7 * x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3365_336586


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3365_336557

theorem imaginary_part_of_z (z : ℂ) : (z - 2*I) * (2 - I) = 5 → z.im = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3365_336557


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3365_336543

/-- A convex polygon with the sum of all angles except two equal to 3240° has 22 sides. -/
theorem convex_polygon_sides (n : ℕ) (sum_except_two : ℝ) : 
  sum_except_two = 3240 → (∃ (a b : ℝ), 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 
    180 * (n - 2) = sum_except_two + a + b) → n = 22 :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l3365_336543


namespace NUMINAMATH_CALUDE_largest_number_l3365_336520

theorem largest_number (S : Finset ℕ) (h : S = {5, 8, 4, 3, 2}) : 
  Finset.max' S (by simp [h]) = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3365_336520


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l3365_336587

-- Part 1
theorem simplify_expression (a : ℝ) : -2*a^2 + 3 - (3*a^2 - 6*a + 1) + 3 = -5*a^2 + 6*a + 5 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  1/2*x - 2*(x - 1/3*y^2) + (-3/2*x + 1/3*y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l3365_336587


namespace NUMINAMATH_CALUDE_marbles_probability_l3365_336551

theorem marbles_probability (total : ℕ) (red : ℕ) (h1 : total = 48) (h2 : red = 12) :
  let p := (total - red) / total
  p * p = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_marbles_probability_l3365_336551


namespace NUMINAMATH_CALUDE_town_population_problem_l3365_336570

/-- The original population of the town -/
def original_population : ℕ := 1200

/-- The increase in population -/
def population_increase : ℕ := 1500

/-- The percentage decrease after the increase -/
def percentage_decrease : ℚ := 15 / 100

/-- The final difference in population compared to the original plus increase -/
def final_difference : ℕ := 45

theorem town_population_problem :
  let increased_population := original_population + population_increase
  let decreased_population := increased_population - (increased_population * percentage_decrease).floor
  decreased_population = original_population + population_increase - final_difference :=
by sorry

end NUMINAMATH_CALUDE_town_population_problem_l3365_336570


namespace NUMINAMATH_CALUDE_f_of_f_2_l3365_336581

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.log x / Real.log 2
  else if x ≥ 1 then 1 / x^2
  else 0  -- This case is added to make the function total

theorem f_of_f_2 : f (f 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_2_l3365_336581


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_l3365_336507

theorem integer_solutions_quadratic (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x y : ℤ, x^2 + p*x + q^4 = 0 ∧ y^2 + p*y + q^4 = 0 ∧ x ≠ y) ↔ 
  (p = 17 ∧ q = 2) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_l3365_336507


namespace NUMINAMATH_CALUDE_mistaken_divisor_l3365_336535

/-- Given a division with remainder 0, correct divisor 21, correct quotient 24,
    and a mistaken quotient of 42, prove that the mistaken divisor is 12. -/
theorem mistaken_divisor (dividend : ℕ) (mistaken_divisor : ℕ) : 
  dividend % 21 = 0 ∧ 
  dividend / 21 = 24 ∧ 
  dividend / mistaken_divisor = 42 →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l3365_336535


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l3365_336540

/-- Proves that the time taken to remove wallpaper from the first wall is 2 hours -/
theorem wallpaper_removal_time (total_walls : ℕ) (walls_removed : ℕ) (remaining_time : ℕ) :
  total_walls = 8 →
  walls_removed = 1 →
  remaining_time = 14 →
  remaining_time / (total_walls - walls_removed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_removal_time_l3365_336540


namespace NUMINAMATH_CALUDE_dress_price_difference_l3365_336596

theorem dress_price_difference (discounted_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discounted_price = 61.2 ∧ discount_rate = 0.15 ∧ increase_rate = 0.25 →
  (discounted_price / (1 - discount_rate) * (1 + increase_rate)) - (discounted_price / (1 - discount_rate)) = 4.5 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l3365_336596


namespace NUMINAMATH_CALUDE_floor_sum_example_l3365_336563

theorem floor_sum_example : ⌊(24.8 : ℝ)⌋ + ⌊(-24.8 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3365_336563


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l3365_336514

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l3365_336514


namespace NUMINAMATH_CALUDE_problem_statement_l3365_336572

theorem problem_statement (x : ℚ) (h : 5 * x - 8 = 15 * x - 2) : 5 * (x - 3) = -18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3365_336572


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l3365_336529

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents the given binary number 1011001₍₂₎ -/
def binary_number : List Bool := [true, false, false, true, true, false, true]

/-- The octal number we want to prove equality with -/
def octal_number : ℕ := 131

theorem binary_to_octal_conversion :
  binary_to_decimal binary_number = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l3365_336529


namespace NUMINAMATH_CALUDE_car_meeting_points_distance_prove_car_meeting_points_distance_l3365_336541

/-- Given two cars starting from points A and B, if they meet at a point 108 km from B, 
    then continue to each other's starting points and return, meeting again at a point 84 km from A, 
    the distance between their two meeting points is 48 km. -/
theorem car_meeting_points_distance : ℝ → Prop :=
  fun d =>
    let first_meeting := d - 108
    let second_meeting := 84
    first_meeting - second_meeting = 48

/-- Proof of the theorem -/
theorem prove_car_meeting_points_distance : ∃ d : ℝ, car_meeting_points_distance d :=
sorry

end NUMINAMATH_CALUDE_car_meeting_points_distance_prove_car_meeting_points_distance_l3365_336541


namespace NUMINAMATH_CALUDE_small_mold_radius_l3365_336534

theorem small_mold_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 2 → n = 64 → (2 / 3 * Real.pi * R^3) = (n * (2 / 3 * Real.pi * r^3)) → r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_small_mold_radius_l3365_336534


namespace NUMINAMATH_CALUDE_range_of_f_l3365_336545

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f : {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3365_336545


namespace NUMINAMATH_CALUDE_matrix_equation_l3365_336542

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_equation (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - 2 * d) / (c - 4 * b) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_l3365_336542


namespace NUMINAMATH_CALUDE_line_slope_equals_y_coord_l3365_336526

/-- Given a line passing through points (-1, -4) and (4, y), 
    if the slope of the line is equal to y, then y = 1. -/
theorem line_slope_equals_y_coord (y : ℝ) : 
  (y - (-4)) / (4 - (-1)) = y → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_equals_y_coord_l3365_336526


namespace NUMINAMATH_CALUDE_unique_five_digit_pair_l3365_336598

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Check if each digit of b is exactly 1 greater than the corresponding digit of a -/
def digitsOneGreater (a b : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → (b / 10^i) % 10 = (a / 10^i) % 10 + 1

/-- The main theorem -/
theorem unique_five_digit_pair : 
  ∀ a b : ℕ, 
    10000 ≤ a ∧ a < 100000 ∧
    10000 ≤ b ∧ b < 100000 ∧
    isPerfectSquare a ∧
    isPerfectSquare b ∧
    b - a = 11111 ∧
    digitsOneGreater a b →
    a = 13225 ∧ b = 24336 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_pair_l3365_336598


namespace NUMINAMATH_CALUDE_square_max_perimeter_l3365_336549

/-- A right-angled quadrilateral inscribed in a circle --/
structure InscribedRightQuadrilateral (r : ℝ) where
  x : ℝ
  y : ℝ
  right_angled : x^2 + y^2 = (2*r)^2
  inscribed : x > 0 ∧ y > 0

/-- The perimeter of an inscribed right-angled quadrilateral --/
def perimeter (r : ℝ) (q : InscribedRightQuadrilateral r) : ℝ :=
  2 * (q.x + q.y)

/-- The statement that the square has the largest perimeter --/
theorem square_max_perimeter (r : ℝ) (hr : r > 0) :
  ∀ q : InscribedRightQuadrilateral r,
    perimeter r q ≤ 4 * r * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_square_max_perimeter_l3365_336549


namespace NUMINAMATH_CALUDE_polynomial_properties_l3365_336567

def P (x : ℤ) : ℤ := x * (x + 1) * (x + 2)

theorem polynomial_properties :
  (∀ x : ℤ, ∃ k : ℤ, P x = 3 * k) ∧
  (∃ a b c d : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) ∧
  (∃ a b c : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_properties_l3365_336567


namespace NUMINAMATH_CALUDE_interest_difference_l3365_336515

theorem interest_difference (principal rate time : ℝ) 
  (h_principal : principal = 600)
  (h_rate : rate = 0.05)
  (h_time : time = 8) :
  principal - (principal * rate * time) = 360 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l3365_336515


namespace NUMINAMATH_CALUDE_points_three_units_from_origin_l3365_336517

theorem points_three_units_from_origin (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by
sorry

end NUMINAMATH_CALUDE_points_three_units_from_origin_l3365_336517


namespace NUMINAMATH_CALUDE_cuboid_vertices_sum_l3365_336519

theorem cuboid_vertices_sum (n : ℕ) (h : 6 * n + 12 * n = 216) : 8 * n = 96 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_vertices_sum_l3365_336519


namespace NUMINAMATH_CALUDE_points_on_line_implies_b_value_l3365_336509

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that if the given points lie on the same line, then b = -1/2. -/
theorem points_on_line_implies_b_value (b : ℝ) :
  collinear 6 (-10) (-b + 4) 3 (3*b + 6) 3 → b = -1/2 := by
  sorry

#check points_on_line_implies_b_value

end NUMINAMATH_CALUDE_points_on_line_implies_b_value_l3365_336509


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3365_336591

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_difference_magnitude :
  ∃ x : ℝ, parallel a (b x) ∧ 
    Real.sqrt ((a.1 - (b x).1)^2 + (a.2 - (b x).2)^2) = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3365_336591


namespace NUMINAMATH_CALUDE_average_book_width_l3365_336530

/-- The average width of 7 books with given widths is 4.5 cm -/
theorem average_book_width : 
  let book_widths : List ℝ := [5, 3/4, 1.5, 3, 12, 2, 7.5]
  (book_widths.sum / book_widths.length : ℝ) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l3365_336530


namespace NUMINAMATH_CALUDE_tp_supply_duration_l3365_336505

/-- Represents the toilet paper usage of a family member --/
structure TPUsage where
  weekdayTimes : ℕ
  weekdaySquares : ℕ
  weekendTimes : ℕ
  weekendSquares : ℕ

/-- Calculates the total squares used per week for a family member --/
def weeklyUsage (usage : TPUsage) : ℕ :=
  5 * usage.weekdayTimes * usage.weekdaySquares +
  2 * usage.weekendTimes * usage.weekendSquares

/-- Represents the family's toilet paper situation --/
structure TPFamily where
  bill : TPUsage
  wife : TPUsage
  kid : TPUsage
  kidCount : ℕ
  rollCount : ℕ
  squaresPerRoll : ℕ

/-- Calculates the total squares used per week for the entire family --/
def familyWeeklyUsage (family : TPFamily) : ℕ :=
  weeklyUsage family.bill +
  weeklyUsage family.wife +
  family.kidCount * weeklyUsage family.kid

/-- Calculates how many days the toilet paper supply will last --/
def supplyDuration (family : TPFamily) : ℕ :=
  let totalSquares := family.rollCount * family.squaresPerRoll
  let weeksSupply := totalSquares / familyWeeklyUsage family
  7 * weeksSupply

/-- The main theorem stating how long the toilet paper supply will last --/
theorem tp_supply_duration : 
  let family : TPFamily := {
    bill := { weekdayTimes := 3, weekdaySquares := 5, weekendTimes := 4, weekendSquares := 6 },
    wife := { weekdayTimes := 4, weekdaySquares := 8, weekendTimes := 5, weekendSquares := 10 },
    kid := { weekdayTimes := 5, weekdaySquares := 6, weekendTimes := 6, weekendSquares := 5 },
    kidCount := 2,
    rollCount := 1000,
    squaresPerRoll := 300
  }
  ∃ (d : ℕ), d ≥ 2615 ∧ d ≤ 2616 ∧ supplyDuration family = d :=
by sorry


end NUMINAMATH_CALUDE_tp_supply_duration_l3365_336505


namespace NUMINAMATH_CALUDE_divisors_of_720_l3365_336538

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l3365_336538


namespace NUMINAMATH_CALUDE_local_extremum_sum_l3365_336578

/-- A function f with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_sum (a b : ℝ) :
  f a b 1 = 10 ∧ f' a b 1 = 0 → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_sum_l3365_336578


namespace NUMINAMATH_CALUDE_a_value_is_two_l3365_336525

/-- Represents the chemical reaction 3A + B ⇌ aC + 2D -/
structure Reaction where
  a : ℕ

/-- Represents the reaction conditions -/
structure ReactionConditions where
  initial_A : ℝ
  initial_B : ℝ
  volume : ℝ
  time : ℝ
  final_C : ℝ
  rate_D : ℝ

/-- Determines the value of 'a' in the reaction equation -/
def determine_a (reaction : Reaction) (conditions : ReactionConditions) : ℕ :=
  sorry

/-- Theorem stating that the value of 'a' is 2 given the specified conditions -/
theorem a_value_is_two :
  ∀ (reaction : Reaction) (conditions : ReactionConditions),
    conditions.initial_A = 0.6 ∧
    conditions.initial_B = 0.5 ∧
    conditions.volume = 0.4 ∧
    conditions.time = 5 ∧
    conditions.final_C = 0.2 ∧
    conditions.rate_D = 0.1 →
    determine_a reaction conditions = 2 :=
  sorry

end NUMINAMATH_CALUDE_a_value_is_two_l3365_336525


namespace NUMINAMATH_CALUDE_rowing_problem_solution_l3365_336556

/-- Represents the problem of calculating the distance to a destination given rowing speeds and time. -/
def RowingProblem (stillWaterSpeed currentVelocity totalTime : ℝ) : Prop :=
  let downstreamSpeed := stillWaterSpeed + currentVelocity
  let upstreamSpeed := stillWaterSpeed - currentVelocity
  ∃ (distance : ℝ),
    distance > 0 ∧
    distance / downstreamSpeed + distance / upstreamSpeed = totalTime

/-- Theorem stating that given the specific conditions of the problem, the distance to the destination is 2.4 km. -/
theorem rowing_problem_solution :
  RowingProblem 5 1 1 →
  ∃ (distance : ℝ), distance = 2.4 := by
  sorry

#check rowing_problem_solution

end NUMINAMATH_CALUDE_rowing_problem_solution_l3365_336556


namespace NUMINAMATH_CALUDE_construction_labor_problem_l3365_336585

theorem construction_labor_problem (total_hired : ℕ) (operator_pay laborer_pay : ℚ) (total_payroll : ℚ) :
  total_hired = 35 →
  operator_pay = 140 →
  laborer_pay = 90 →
  total_payroll = 3950 →
  ∃ (operators laborers : ℕ),
    operators + laborers = total_hired ∧
    operators * operator_pay + laborers * laborer_pay = total_payroll ∧
    laborers = 19 := by
  sorry

end NUMINAMATH_CALUDE_construction_labor_problem_l3365_336585


namespace NUMINAMATH_CALUDE_unique_prime_with_14_divisors_l3365_336593

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The theorem stating that there is exactly one prime p such that p^2 + 23 has 14 positive divisors -/
theorem unique_prime_with_14_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ num_divisors (p^2 + 23) = 14 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_14_divisors_l3365_336593


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3365_336546

theorem opposite_of_negative_three : 
  ∃ x : ℤ, x + (-3) = 0 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3365_336546


namespace NUMINAMATH_CALUDE_subtracted_value_l3365_336574

theorem subtracted_value (x : ℝ) (h1 : (x - 5) / 7 = 7) : 
  ∃ y : ℝ, (x - y) / 10 = 4 ∧ y = 14 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3365_336574


namespace NUMINAMATH_CALUDE_fourth_number_11th_row_l3365_336592

/-- The last number in row i of the pattern -/
def lastNumber (i : ℕ) : ℕ := 5 * i

/-- The fourth number in row i of the pattern -/
def fourthNumber (i : ℕ) : ℕ := lastNumber i - 1

/-- Theorem: The fourth number in the 11th row is 54 -/
theorem fourth_number_11th_row :
  fourthNumber 11 = 54 := by sorry

end NUMINAMATH_CALUDE_fourth_number_11th_row_l3365_336592


namespace NUMINAMATH_CALUDE_fish_weight_l3365_336590

/-- Given a barrel of fish with the following properties:
  1. The total weight of the barrel and fish is 54 kg.
  2. After removing half of the fish, the total weight is 29 kg.
  This theorem proves that the initial weight of the fish alone is 50 kg. -/
theorem fish_weight (barrel_weight : ℝ) (fish_weight : ℝ) 
  (h1 : barrel_weight + fish_weight = 54)
  (h2 : barrel_weight + fish_weight / 2 = 29) :
  fish_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_fish_weight_l3365_336590


namespace NUMINAMATH_CALUDE_ed_hotel_stay_l3365_336548

/-- The number of hours Ed stayed in the hotel last night -/
def hours_stayed : ℕ := 6

/-- The cost per hour for staying at night -/
def night_cost_per_hour : ℚ := 3/2

/-- The cost per hour for staying in the morning -/
def morning_cost_per_hour : ℚ := 2

/-- Ed's initial money -/
def initial_money : ℕ := 80

/-- The number of hours Ed stayed in the morning -/
def morning_hours : ℕ := 4

/-- The amount of money Ed had left after paying for his stay -/
def money_left : ℕ := 63

theorem ed_hotel_stay :
  hours_stayed * night_cost_per_hour + 
  morning_hours * morning_cost_per_hour = 
  initial_money - money_left :=
by sorry

end NUMINAMATH_CALUDE_ed_hotel_stay_l3365_336548


namespace NUMINAMATH_CALUDE_complement_probability_l3365_336575

theorem complement_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_complement_probability_l3365_336575


namespace NUMINAMATH_CALUDE_tobias_daily_hours_l3365_336580

/-- Proves that Tobias plays 5 hours per day given the conditions of the problem -/
theorem tobias_daily_hours (nathan_daily_hours : ℕ) (nathan_days : ℕ) (tobias_days : ℕ) (total_hours : ℕ) :
  nathan_daily_hours = 3 →
  nathan_days = 14 →
  tobias_days = 7 →
  total_hours = 77 →
  ∃ (tobias_daily_hours : ℕ), 
    tobias_daily_hours * tobias_days + nathan_daily_hours * nathan_days = total_hours ∧
    tobias_daily_hours = 5 :=
by sorry

end NUMINAMATH_CALUDE_tobias_daily_hours_l3365_336580


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3365_336564

/-- Given a line passing through the points (2, -1) and (5, 2), 
    prove that the sum of its slope and y-intercept is -2. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 : ℝ) * m + b = -1 ∧ 
  (5 : ℝ) * m + b = 2 → 
  m + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3365_336564


namespace NUMINAMATH_CALUDE_inverse_computation_l3365_336582

-- Define the function g
def g : ℕ → ℕ
| 1 => 4
| 2 => 9
| 3 => 11
| 5 => 3
| 7 => 6
| 12 => 2
| _ => 0  -- for other inputs, we'll return 0

-- Assume g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define g_inv as the inverse of g
noncomputable def g_inv : ℕ → ℕ := Function.invFun g

-- State the theorem
theorem inverse_computation :
  g_inv ((g_inv 2 + g_inv 11) / g_inv 3) = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_computation_l3365_336582


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_tangent_ratio_l3365_336553

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid :=
  (A B C D : Point)

/-- Checks if a point lies on a given line segment -/
def pointOnSegment (P Q R : Point) : Prop :=
  sorry

/-- Checks if a line is tangent to a circle -/
def isTangent (P Q : Point) (circle : Circle) : Prop :=
  sorry

/-- Checks if a trapezoid is circumscribed around a circle -/
def isCircumscribed (trapezoid : IsoscelesTrapezoid) (circle : Circle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ :=
  sorry

theorem isosceles_trapezoid_tangent_ratio 
  (trapezoid : IsoscelesTrapezoid) 
  (circle : Circle) 
  (P Q R S : Point) :
  isCircumscribed trapezoid circle →
  isTangent P S circle →
  pointOnSegment P Q R →
  pointOnSegment P S R →
  distance P Q / distance Q R = distance R S / distance S R :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_tangent_ratio_l3365_336553


namespace NUMINAMATH_CALUDE_equation_solutions_l3365_336522

def equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 - x + 3*y - 7 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(3,1), (-1,1), (3,-1), (-3,-1), (-1,-5)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 3 1 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3365_336522


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3365_336550

/-- The shortest distance from a point on the parabola y = x^2 to the line x - y - 2 = 0 is 7√2/8 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  ∃ d : ℝ, d = 7 * Real.sqrt 2 / 8 ∧
    ∀ p ∈ parabola, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3365_336550


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l3365_336536

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing :
  (∀ x < -1, (f' x) < 0) ∧ (∀ x > 3, (f' x) < 0) :=
sorry

-- Theorem for minimum value on [-2, 2]
theorem f_min_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y ∧ f x = -7 :=
sorry

-- Theorem for maximum value on [-2, 2]
theorem f_max_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 20 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l3365_336536


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l3365_336569

/-- Represents the production rate and time for a machine. -/
structure Machine where
  rate : ℝ  -- Sprockets produced per hour
  time : ℝ  -- Hours to produce 2000 sprockets

/-- Given three machines A, B, and G with specific production relationships,
    prove that machine A produces 200/11 sprockets per hour. -/
theorem machine_a_production_rate 
  (a b g : Machine)
  (total_sprockets : ℝ)
  (h1 : total_sprockets = 2000)
  (h2 : a.time = g.time + 10)
  (h3 : b.time = g.time - 5)
  (h4 : g.rate = 1.1 * a.rate)
  (h5 : b.rate = 1.15 * a.rate)
  (h6 : a.rate * a.time = total_sprockets)
  (h7 : b.rate * b.time = total_sprockets)
  (h8 : g.rate * g.time = total_sprockets) :
  a.rate = 200 / 11 := by
  sorry

#eval (200 : ℚ) / 11

end NUMINAMATH_CALUDE_machine_a_production_rate_l3365_336569


namespace NUMINAMATH_CALUDE_sum_remainder_six_l3365_336528

theorem sum_remainder_six (m : ℤ) : (9 - m + (m + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_six_l3365_336528


namespace NUMINAMATH_CALUDE_music_students_l3365_336516

/-- Proves that the number of students taking music is 50 given the conditions of the problem -/
theorem music_students (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  ∃ music : ℕ, music = 50 ∧ total = music + art - both + neither :=
by sorry

end NUMINAMATH_CALUDE_music_students_l3365_336516


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_count_l3365_336599

/-- The number of ways to distribute n indistinguishable balls among k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
def five_balls_four_boxes : ℕ := distribute_balls 5 4

theorem five_balls_four_boxes_count : five_balls_four_boxes = 56 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_count_l3365_336599


namespace NUMINAMATH_CALUDE_min_chocolate_cookies_l3365_336555

theorem min_chocolate_cookies (chocolate_batch_size peanut_batch_size total_cookies : ℕ)
  (chocolate_ratio peanut_ratio : ℕ) :
  chocolate_batch_size = 5 →
  peanut_batch_size = 6 →
  chocolate_ratio = 3 →
  peanut_ratio = 2 →
  total_cookies = 94 →
  ∃ (chocolate_batches peanut_batches : ℕ),
    chocolate_batches * chocolate_batch_size + peanut_batches * peanut_batch_size = total_cookies ∧
    chocolate_batches * chocolate_batch_size * peanut_ratio = peanut_batches * peanut_batch_size * chocolate_ratio ∧
    chocolate_batches * chocolate_batch_size ≥ 60 ∧
    ∀ (c p : ℕ), c * chocolate_batch_size + p * peanut_batch_size = total_cookies →
      c * chocolate_batch_size * peanut_ratio = p * peanut_batch_size * chocolate_ratio →
      c * chocolate_batch_size ≥ chocolate_batches * chocolate_batch_size :=
by sorry

end NUMINAMATH_CALUDE_min_chocolate_cookies_l3365_336555


namespace NUMINAMATH_CALUDE_bandi_has_winning_strategy_l3365_336589

/-- Represents a player in the game -/
inductive Player
| Andi
| Bandi

/-- Represents a digit in the binary number -/
inductive Digit
| Zero
| One

/-- Represents a strategy for a player -/
def Strategy := List Digit → Digit

/-- The game state -/
structure GameState :=
(sequence : List Digit)
(turn : Player)
(moves_left : Nat)

/-- The result of the game -/
inductive GameResult
| AndiWin
| BandiWin

/-- Converts a list of digits to a natural number -/
def binary_to_nat (digits : List Digit) : Nat :=
  sorry

/-- Checks if a number is the sum of two squares -/
def is_sum_of_squares (n : Nat) : Prop :=
  sorry

/-- Plays the game given strategies for both players -/
def play_game (andi_strategy : Strategy) (bandi_strategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that Bandi has a winning strategy -/
theorem bandi_has_winning_strategy :
  ∃ (bandi_strategy : Strategy),
    ∀ (andi_strategy : Strategy),
      play_game andi_strategy bandi_strategy = GameResult.BandiWin :=
sorry

end NUMINAMATH_CALUDE_bandi_has_winning_strategy_l3365_336589


namespace NUMINAMATH_CALUDE_initial_boxes_count_l3365_336513

theorem initial_boxes_count (ali_boxes_per_circle ernie_boxes_per_circle ali_circles ernie_circles : ℕ) 
  (h1 : ali_boxes_per_circle = 8)
  (h2 : ernie_boxes_per_circle = 10)
  (h3 : ali_circles = 5)
  (h4 : ernie_circles = 4) :
  ali_boxes_per_circle * ali_circles + ernie_boxes_per_circle * ernie_circles = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_boxes_count_l3365_336513


namespace NUMINAMATH_CALUDE_total_crayons_l3365_336561

def number_of_boxes : ℕ := 7
def crayons_per_box : ℕ := 5

theorem total_crayons : number_of_boxes * crayons_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3365_336561


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3365_336503

theorem complex_equation_solution :
  ∀ (z : ℂ), (1 + Complex.I) * z = 2 + Complex.I → z = 3/2 - (1/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3365_336503


namespace NUMINAMATH_CALUDE_max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l3365_336531

/-- Represents a 3x3x3 cube --/
structure Cube :=
  (side_length : ℕ := 3)

/-- Represents a 3x1 strip --/
structure Strip :=
  (length : ℕ := 3)
  (width : ℕ := 1)

/-- Represents the configuration of strips on the cube --/
structure CubeConfiguration :=
  (cube : Cube)
  (total_strips : ℕ := 18)
  (bent_strips : ℕ)
  (flat_strips : ℕ)

/-- The theorem stating the maximum number of bent strips --/
theorem max_bent_strips (config : CubeConfiguration) : config.bent_strips ≤ 14 :=
by sorry

/-- The theorem stating that 14 bent strips is achievable --/
theorem achievable_14_bent_strips : ∃ (config : CubeConfiguration), config.bent_strips = 14 ∧ config.flat_strips = 4 :=
by sorry

/-- The main theorem combining the above results --/
theorem max_bent_strips_is_14 : 
  (∀ (config : CubeConfiguration), config.bent_strips ≤ 14) ∧
  (∃ (config : CubeConfiguration), config.bent_strips = 14) :=
by sorry

end NUMINAMATH_CALUDE_max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l3365_336531


namespace NUMINAMATH_CALUDE_negative_three_times_inequality_l3365_336533

theorem negative_three_times_inequality (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_inequality_l3365_336533


namespace NUMINAMATH_CALUDE_sum_of_possible_AB_values_l3365_336501

/-- Represents a 7-digit number in the form A568B72 -/
def SevenDigitNumber (A B : Nat) : Nat :=
  A * 1000000 + 568000 + B * 100 + 72

theorem sum_of_possible_AB_values :
  (∀ A B : Nat, A < 10 ∧ B < 10 →
    SevenDigitNumber A B % 9 = 0 →
    (A + B = 8 ∨ A + B = 17)) ∧
  (∃ A₁ B₁ A₂ B₂ : Nat,
    A₁ < 10 ∧ B₁ < 10 ∧ A₂ < 10 ∧ B₂ < 10 ∧
    SevenDigitNumber A₁ B₁ % 9 = 0 ∧
    SevenDigitNumber A₂ B₂ % 9 = 0 ∧
    A₁ + B₁ = 8 ∧ A₂ + B₂ = 17) :=
by sorry

#check sum_of_possible_AB_values

end NUMINAMATH_CALUDE_sum_of_possible_AB_values_l3365_336501


namespace NUMINAMATH_CALUDE_central_angle_is_45_degrees_l3365_336573

/-- Represents a circular dartboard divided into sectors -/
structure Dartboard where
  smallSectors : Nat
  largeSectors : Nat
  smallSectorProbability : ℝ

/-- Calculate the central angle of a smaller sector in degrees -/
def centralAngleSmallSector (d : Dartboard) : ℝ :=
  360 * d.smallSectorProbability

/-- Theorem: The central angle of a smaller sector is 45° for the given dartboard -/
theorem central_angle_is_45_degrees (d : Dartboard) 
  (h1 : d.smallSectors = 3)
  (h2 : d.largeSectors = 1)
  (h3 : d.smallSectorProbability = 1/8) :
  centralAngleSmallSector d = 45 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_is_45_degrees_l3365_336573


namespace NUMINAMATH_CALUDE_gift_wrapping_l3365_336506

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l3365_336506


namespace NUMINAMATH_CALUDE_households_using_neither_brand_l3365_336576

/-- Given information about household soap usage, prove the number of households using neither brand. -/
theorem households_using_neither_brand (total : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  only_A = 60 →
  both = 40 →
  (total - (only_A + 3 * both + both)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_households_using_neither_brand_l3365_336576


namespace NUMINAMATH_CALUDE_circle_through_points_l3365_336524

theorem circle_through_points : ∃ (A B C D : ℝ), 
  (A * 0^2 + B * 0^2 + C * 0 + D * 0 + 1 = 0) ∧ 
  (A * 4^2 + B * 0^2 + C * 4 + D * 0 + 1 = 0) ∧ 
  (A * (-1)^2 + B * 1^2 + C * (-1) + D * 1 + 1 = 0) ∧ 
  (A = 1 ∧ B = 1 ∧ C = -4 ∧ D = -6) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l3365_336524


namespace NUMINAMATH_CALUDE_function_characterization_l3365_336552

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem function_characterization :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3365_336552


namespace NUMINAMATH_CALUDE_sales_department_replacement_l3365_336521

/-- Represents the ages and work experience of employees in a sales department. -/
structure SalesDepartment where
  initialMenCount : ℕ
  initialAvgAge : ℝ
  initialAvgExperience : ℝ
  replacedMenAges : Fin 2 → ℕ
  womenAgeRanges : Fin 2 → Set ℕ
  newAvgAge : ℝ
  newAvgExperience : ℝ

/-- Theorem stating the average age of the two women and the change in work experience. -/
theorem sales_department_replacement
  (dept : SalesDepartment)
  (h_men_count : dept.initialMenCount = 8)
  (h_age_increase : dept.newAvgAge = dept.initialAvgAge + 2)
  (h_exp_change : dept.newAvgExperience = dept.initialAvgExperience + 1)
  (h_replaced_ages : dept.replacedMenAges 0 = 20 ∧ dept.replacedMenAges 1 = 24)
  (h_women_ages : dept.womenAgeRanges 0 = Set.Icc 26 30 ∧ dept.womenAgeRanges 1 = Set.Icc 32 36) :
  ∃ (w₁ w₂ : ℕ), w₁ ∈ dept.womenAgeRanges 0 ∧ w₂ ∈ dept.womenAgeRanges 1 ∧
  (w₁ + w₂) / 2 = 30 ∧
  (dept.initialMenCount * dept.newAvgExperience - dept.initialMenCount * dept.initialAvgExperience) = 8 := by
  sorry


end NUMINAMATH_CALUDE_sales_department_replacement_l3365_336521


namespace NUMINAMATH_CALUDE_f_sum_positive_l3365_336537

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For x₁, x₂ ∈ ℝ with x₁ + x₂ > 0, f(x₁) + f(x₂) > 0 -/
theorem f_sum_positive (x₁ x₂ : ℝ) (h : x₁ + x₂ > 0) : f x₁ + f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l3365_336537


namespace NUMINAMATH_CALUDE_magic_square_sum_l3365_336565

/-- Represents a 3x3 magic square with center 7 -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ := 7

/-- The magic sum of the square -/
def magicSum (s : MagicSquare) : ℤ := 22 + s.c

/-- Properties of the magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  s.a + s.y + s.d = magicSum s ∧
  s.c + s.y + s.b = magicSum s ∧
  s.x + s.y + s.a = magicSum s ∧
  s.c + s.y + s.x = magicSum s

theorem magic_square_sum (s : MagicSquare) (h : isMagicSquare s) :
  s.x + s.y + s.a + s.b + s.c + s.d = 68 := by
  sorry

#check magic_square_sum

end NUMINAMATH_CALUDE_magic_square_sum_l3365_336565


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_l3365_336518

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation
variable (line_not_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel 
  (α β : Plane) (m : Line) 
  (h1 : plane_parallel α β) 
  (h2 : line_parallel_plane m α) 
  (h3 : line_not_in_plane m β) : 
  line_parallel_plane m β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_l3365_336518


namespace NUMINAMATH_CALUDE_ratio_invariance_l3365_336554

theorem ratio_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x * y) / (y * x) = 1 → ∃ (r : ℝ), r ≠ 0 ∧ x / y = r :=
by sorry

end NUMINAMATH_CALUDE_ratio_invariance_l3365_336554


namespace NUMINAMATH_CALUDE_mailbox_distance_l3365_336562

/-- Represents Jeffrey's walking pattern and the total steps taken -/
structure WalkingPattern where
  forward_steps : ℕ
  backward_steps : ℕ
  total_steps : ℕ

/-- Calculates the effective distance covered given a walking pattern -/
def effectiveDistance (pattern : WalkingPattern) : ℕ :=
  let cycle := pattern.forward_steps + pattern.backward_steps
  let effective_steps_per_cycle := pattern.forward_steps - pattern.backward_steps
  (pattern.total_steps / cycle) * effective_steps_per_cycle

/-- Theorem: Given Jeffrey's walking pattern and total steps, the distance to the mailbox is 110 steps -/
theorem mailbox_distance (pattern : WalkingPattern) 
  (h1 : pattern.forward_steps = 3)
  (h2 : pattern.backward_steps = 2)
  (h3 : pattern.total_steps = 330) :
  effectiveDistance pattern = 110 := by
  sorry

end NUMINAMATH_CALUDE_mailbox_distance_l3365_336562


namespace NUMINAMATH_CALUDE_expected_remaining_people_l3365_336500

/-- The expected number of people remaining in a line of 100 people after a removal process -/
theorem expected_remaining_people (n : Nat) (h : n = 100) :
  let people := n
  let facing_right := n / 2
  let facing_left := n - facing_right
  let expected_remaining := (2^n : ℝ) / (Nat.choose n facing_right) - 1
  expected_remaining = (2^100 : ℝ) / (Nat.choose 100 50) - 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_remaining_people_l3365_336500


namespace NUMINAMATH_CALUDE_total_cookie_time_l3365_336512

/-- The time it takes to make black & white cookies -/
def cookie_making_time (batter_time baking_time cooling_time white_icing_time chocolate_icing_time : ℕ) : ℕ :=
  batter_time + baking_time + cooling_time + white_icing_time + chocolate_icing_time

/-- Theorem stating that the total time to make black & white cookies is 100 minutes -/
theorem total_cookie_time :
  ∃ (batter_time cooling_time : ℕ),
    batter_time = 10 ∧
    cooling_time = 15 ∧
    cookie_making_time batter_time 15 cooling_time 30 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cookie_time_l3365_336512


namespace NUMINAMATH_CALUDE_cosine_rule_triangle_l3365_336502

theorem cosine_rule_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  a = 3 ∧ b = 4 ∧ c = 6 → cos_B = 29 / 36 := by
  sorry

end NUMINAMATH_CALUDE_cosine_rule_triangle_l3365_336502


namespace NUMINAMATH_CALUDE_sum_of_squares_l3365_336523

theorem sum_of_squares (a b c d : ℝ) : 
  a + b = -3 →
  a * b + b * c + c * a = -4 →
  a * b * c + b * c * d + c * d * a + d * a * b = 14 →
  a * b * c * d = 30 →
  a^2 + b^2 + c^2 + d^2 = 141 / 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3365_336523


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3365_336544

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), (1052 + y) % 23 = 0 → y ≥ x) ∧ 
  (1052 + x) % 23 = 0 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3365_336544


namespace NUMINAMATH_CALUDE_first_share_interest_rate_l3365_336595

/-- Proves that the interest rate of the first type of share is 9% given the problem conditions --/
theorem first_share_interest_rate : 
  let total_investment : ℝ := 100000
  let second_share_rate : ℝ := 11
  let total_interest_rate : ℝ := 9.5
  let second_share_investment : ℝ := 25000
  let first_share_investment : ℝ := total_investment - second_share_investment
  let total_interest : ℝ := total_interest_rate / 100 * total_investment
  let second_share_interest : ℝ := second_share_rate / 100 * second_share_investment
  let first_share_interest : ℝ := total_interest - second_share_interest
  let first_share_rate : ℝ := first_share_interest / first_share_investment * 100
  first_share_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_first_share_interest_rate_l3365_336595


namespace NUMINAMATH_CALUDE_equation_real_solutions_l3365_336508

theorem equation_real_solutions :
  let f : ℝ → ℝ := λ x => 5*x/(x^2 + 2*x + 4) + 7*x/(x^2 - 7*x + 4) + 5/3
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_equation_real_solutions_l3365_336508


namespace NUMINAMATH_CALUDE_subtracted_value_l3365_336568

theorem subtracted_value (N : ℕ) (V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3365_336568


namespace NUMINAMATH_CALUDE_andrea_jim_age_sum_l3365_336532

theorem andrea_jim_age_sum : 
  ∀ (A J x y : ℕ),
  A = J + 29 →                   -- Andrea is 29 years older than Jim
  A - x + J - x = 47 →           -- Sum of their ages x years ago was 47
  J - y = 2 * (J - x) →          -- Jim's age y years ago was twice his age x years ago
  A = 3 * (J - y) →              -- Andrea's current age is three times Jim's age y years ago
  A + J = 79 :=                  -- The sum of their current ages is 79
by
  sorry

end NUMINAMATH_CALUDE_andrea_jim_age_sum_l3365_336532


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_6_l3365_336597

theorem nearest_integer_to_3_plus_sqrt2_to_6 :
  ∃ n : ℤ, ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - m| ∧ n = 3707 :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_6_l3365_336597
