import Mathlib

namespace NUMINAMATH_CALUDE_angle_sum_is_right_angle_l3473_347343

theorem angle_sum_is_right_angle (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_trig : (Real.cos α / Real.sin β) + (Real.cos β / Real.sin α) = 2) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_right_angle_l3473_347343


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3473_347327

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3473_347327


namespace NUMINAMATH_CALUDE_soda_packs_minimum_l3473_347395

def min_packs (total : ℕ) (pack_sizes : List ℕ) : ℕ :=
  sorry

theorem soda_packs_minimum :
  min_packs 120 [8, 15, 30] = 4 :=
sorry

end NUMINAMATH_CALUDE_soda_packs_minimum_l3473_347395


namespace NUMINAMATH_CALUDE_max_central_rectangle_area_l3473_347301

/-- Given a square of side length 23 divided into 9 rectangles, with 4 known areas,
    prove that the maximum area of the central rectangle is 180 -/
theorem max_central_rectangle_area :
  ∀ (a b c d e f : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a + b + c = 23 →
    d + e + f = 23 →
    a * d = 13 →
    b * f = 111 →
    c * e = 37 →
    a * f = 123 →
    b * e ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_max_central_rectangle_area_l3473_347301


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l3473_347396

def expand_polynomial (x : ℝ) := x * (x - 1) * (x + 1)^4

theorem x_squared_coefficient :
  ∃ (a b c d e : ℝ),
    expand_polynomial x = a*x^5 + b*x^4 + c*x^3 + 5*x^2 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l3473_347396


namespace NUMINAMATH_CALUDE_fifteenth_term_is_3_to_8_l3473_347331

def sequence_term (n : ℕ) : ℤ :=
  if n % 4 == 1 then (-3) ^ (n / 4 + 1)
  else if n % 4 == 3 then 3 ^ (n / 2)
  else 1

theorem fifteenth_term_is_3_to_8 :
  sequence_term 15 = 3^8 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_3_to_8_l3473_347331


namespace NUMINAMATH_CALUDE_train_crossing_time_l3473_347307

/-- Given a train and platform with specific dimensions and crossing time, 
    calculate the time it takes for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 450)
  (h3 : platform_crossing_time = 45)
  : ∃ (signal_pole_time : ℝ), 
    (signal_pole_time ≥ 17.9 ∧ signal_pole_time ≤ 18.1) := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3473_347307


namespace NUMINAMATH_CALUDE_green_to_blue_ratio_l3473_347330

/-- Represents the number of chairs of each color in a classroom --/
structure ClassroomChairs where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- The conditions of the classroom chair problem --/
def classroom_conditions (c : ClassroomChairs) : Prop :=
  c.blue = 10 ∧
  ∃ k : ℕ, c.green = k * c.blue ∧
  c.white = c.green + c.blue - 13 ∧
  c.blue + c.green + c.white = 67

/-- The theorem stating that the ratio of green to blue chairs is 3:1 --/
theorem green_to_blue_ratio (c : ClassroomChairs) 
  (h : classroom_conditions c) : c.green = 3 * c.blue :=
sorry

end NUMINAMATH_CALUDE_green_to_blue_ratio_l3473_347330


namespace NUMINAMATH_CALUDE_race_time_proof_l3473_347332

/-- In a 1000-meter race, runner A beats runner B by either 25 meters or 10 seconds. -/
theorem race_time_proof (v : ℝ) (t : ℝ) (h1 : v > 0) (h2 : t > 0) : 
  (1000 = v * t ∧ 975 = v * (t + 10)) → t = 400 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l3473_347332


namespace NUMINAMATH_CALUDE_adjacent_lateral_faces_angle_l3473_347352

/-- A regular quadrilateral pyramid is a pyramid with a square base and four congruent triangular faces. -/
structure RegularQuadrilateralPyramid where
  /-- The side length of the square base -/
  base_side : ℝ
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ

/-- The theorem states that if the lateral face of a regular quadrilateral pyramid
    forms a 45° angle with the base plane, then the angle between adjacent lateral faces is 120°. -/
theorem adjacent_lateral_faces_angle
  (pyramid : RegularQuadrilateralPyramid)
  (h : pyramid.lateral_base_angle = Real.pi / 4) :
  let adjacent_angle := Real.arccos (-1/3)
  adjacent_angle = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_lateral_faces_angle_l3473_347352


namespace NUMINAMATH_CALUDE_fertilizer_prices_l3473_347368

theorem fertilizer_prices (price_A price_B : ℝ)
  (h1 : price_A = price_B + 100)
  (h2 : 2 * price_A + price_B = 1700) :
  price_A = 600 ∧ price_B = 500 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_prices_l3473_347368


namespace NUMINAMATH_CALUDE_trees_survival_difference_l3473_347383

theorem trees_survival_difference (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 13)
  (h2 : dead_trees = 6) :
  initial_trees - dead_trees - dead_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_trees_survival_difference_l3473_347383


namespace NUMINAMATH_CALUDE_unique_satisfying_number_l3473_347356

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number satisfies the given condition -/
def satisfiesCondition (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n + 8802 > reverseDigits n

theorem unique_satisfying_number : 
  ∀ n : Nat, satisfiesCondition n ↔ n = 1099 :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_number_l3473_347356


namespace NUMINAMATH_CALUDE_tan_ratio_difference_l3473_347394

theorem tan_ratio_difference (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) - (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) - (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) - (Real.tan y / Real.tan x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_difference_l3473_347394


namespace NUMINAMATH_CALUDE_trig_problem_l3473_347399

theorem trig_problem (α β : Real) 
  (h1 : Real.sin (Real.pi - α) - 2 * Real.sin (Real.pi / 2 + α) = 0) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.sin α * Real.cos α + Real.sin α ^ 2 = 6 / 5 ∧ Real.tan β = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3473_347399


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l3473_347314

theorem waiter_tips_fraction (salary : ℚ) (h : salary > 0) :
  let tips := (5 / 2) * salary
  let total_income := salary + tips
  tips / total_income = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l3473_347314


namespace NUMINAMATH_CALUDE_root_equation_solution_l3473_347362

theorem root_equation_solution (p q r : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/p) * (M^(1/q))^(1/p) * ((M^(1/r))^(1/q))^(1/p))^p = M^(15/24)) :
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l3473_347362


namespace NUMINAMATH_CALUDE_cubic_function_min_value_l3473_347333

/-- Given a cubic function f(x) with a known maximum value on [-2, 2],
    prove that its minimum value on the same interval is -37. -/
theorem cubic_function_min_value (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2 * x^3 - 6 * x^2 + m) →
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x) →
  ∃ x ∈ Set.Icc (-2) 2, f x = -37 ∧ ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_min_value_l3473_347333


namespace NUMINAMATH_CALUDE_f_roots_l3473_347316

-- Define the function f
def f (x : ℝ) : ℝ := 
  let matrix := !![1, 1, 1; x, -1, 1; x^2, 2, 1]
  Matrix.det matrix

-- State the theorem
theorem f_roots : 
  {x : ℝ | f x = 0} = {-3/2, 1} := by sorry

end NUMINAMATH_CALUDE_f_roots_l3473_347316


namespace NUMINAMATH_CALUDE_emerald_puzzle_l3473_347345

theorem emerald_puzzle :
  ∃ n : ℕ,
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 6 ∧
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 6 → n ≤ m) ∧
    n % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emerald_puzzle_l3473_347345


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3473_347313

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3473_347313


namespace NUMINAMATH_CALUDE_sum_of_combinations_l3473_347364

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_combinations : 
  (∀ m n : ℕ, binomial m n + binomial (m - 1) n = binomial m (n + 1)) →
  (binomial 3 3 + binomial 4 3 + binomial 5 3 + binomial 6 3 + 
   binomial 7 3 + binomial 8 3 + binomial 9 3 + binomial 10 3 = 330) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l3473_347364


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3473_347337

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3473_347337


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3473_347359

theorem complex_equation_solution (Z : ℂ) : (2 + 4*I) / Z = 1 - I → Z = -1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3473_347359


namespace NUMINAMATH_CALUDE_cosine_value_from_ratio_l3473_347347

theorem cosine_value_from_ratio (α : Real) (h : (1 - Real.cos α) / Real.sin α = 3) :
  Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_ratio_l3473_347347


namespace NUMINAMATH_CALUDE_unique_prime_square_sum_l3473_347386

theorem unique_prime_square_sum (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_sum_l3473_347386


namespace NUMINAMATH_CALUDE_fraction_equality_l3473_347392

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3473_347392


namespace NUMINAMATH_CALUDE_area_ratio_is_one_twentyfifth_l3473_347366

/-- A square inscribed in a circle with a smaller square as described -/
structure InscribedSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  t : ℝ
  /-- The larger square is inscribed in the circle -/
  larger_inscribed : s = r * Real.sqrt 2
  /-- The smaller square has one side coinciding with the larger square -/
  coinciding_side : t ≤ s
  /-- Two vertices of the smaller square are on the circle -/
  smaller_on_circle : t * Real.sqrt ((s/2)^2 + (t/2)^2) = r * s

/-- The ratio of the areas of the smaller square to the larger square is 1/25 -/
theorem area_ratio_is_one_twentyfifth (sq : InscribedSquares) :
  (sq.t^2) / (sq.s^2) = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_twentyfifth_l3473_347366


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3473_347338

def z : ℂ := Complex.I^3 * (1 + Complex.I) * Complex.I

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3473_347338


namespace NUMINAMATH_CALUDE_mike_additional_money_needed_l3473_347360

def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_savings_percentage : ℝ := 0.40

theorem mike_additional_money_needed :
  let discounted_phone := phone_cost * (1 - phone_discount)
  let discounted_smartwatch := smartwatch_cost * (1 - smartwatch_discount)
  let total_before_tax := discounted_phone + discounted_smartwatch
  let total_with_tax := total_before_tax * (1 + sales_tax)
  let mike_savings := total_with_tax * mike_savings_percentage
  total_with_tax - mike_savings = 1023.99 := by sorry

end NUMINAMATH_CALUDE_mike_additional_money_needed_l3473_347360


namespace NUMINAMATH_CALUDE_toothpicks_required_l3473_347370

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 3000

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The total number of toothpicks if no sides were shared -/
def total_potential_toothpicks : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- The theorem stating the total number of toothpicks required -/
theorem toothpicks_required : 
  (total_potential_toothpicks - boundary_toothpicks) / 2 + boundary_toothpicks = 6761700 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_required_l3473_347370


namespace NUMINAMATH_CALUDE_octagon_side_length_l3473_347353

/-- The side length of a regular octagon formed from the same wire as a regular pentagon --/
theorem octagon_side_length (pentagon_side : ℝ) (h : pentagon_side = 16) : 
  let pentagon_perimeter := 5 * pentagon_side
  let octagon_side := pentagon_perimeter / 8
  octagon_side = 10 := by
sorry

end NUMINAMATH_CALUDE_octagon_side_length_l3473_347353


namespace NUMINAMATH_CALUDE_soccer_team_statistics_l3473_347324

theorem soccer_team_statistics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 10 →
  both_subjects = 6 →
  ∃ (statistics_players : ℕ),
    statistics_players = 23 ∧
    statistics_players + physics_players - both_subjects = total_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_statistics_l3473_347324


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_impossibility_l3473_347340

theorem chessboard_rearrangement_impossibility :
  ∀ (initial_placement final_placement : Fin 8 → Fin 8 → Bool),
  (∀ i j : Fin 8, (∃! k : Fin 8, initial_placement i k = true) ∧ 
                  (∃! k : Fin 8, initial_placement k j = true)) →
  (∀ i j : Fin 8, (∃! k : Fin 8, final_placement i k = true) ∧ 
                  (∃! k : Fin 8, final_placement k j = true)) →
  (∀ i j : Fin 8, initial_placement i j = true → 
    ∃ i' j' : Fin 8, final_placement i' j' = true ∧ 
    (i'.val + j'.val : ℕ) > (i.val + j.val)) →
  False :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_impossibility_l3473_347340


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3473_347303

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (∀ n, a (n + 1) > a n) →      -- increasing sequence
  a 2 = 2 →                     -- a_2 = 2
  a 4 - a 3 = 4 →               -- a_4 - a_3 = 4
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3473_347303


namespace NUMINAMATH_CALUDE_product_mod_25_l3473_347385

theorem product_mod_25 (m : ℕ) : 
  95 * 115 * 135 ≡ m [MOD 25] → 0 ≤ m → m < 25 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l3473_347385


namespace NUMINAMATH_CALUDE_hare_wolf_distance_l3473_347371

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

def relative_speed : ℝ := hare_speed - wolf_speed

theorem hare_wolf_distance :
  ∀ d : ℝ, d > 0 ∧ d < track_length / 2 →
  (d = distance_traveled relative_speed ∨ d = track_length - distance_traveled relative_speed) →
  d = 40 ∨ d = 60 := by sorry

end NUMINAMATH_CALUDE_hare_wolf_distance_l3473_347371


namespace NUMINAMATH_CALUDE_equal_quotient_remainder_divisible_by_seven_l3473_347389

theorem equal_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ (q : ℕ), n = 7 * q + q ∧ q < 7} = {8, 16, 24, 32, 40, 48} := by
  sorry

end NUMINAMATH_CALUDE_equal_quotient_remainder_divisible_by_seven_l3473_347389


namespace NUMINAMATH_CALUDE_hash_difference_l3473_347358

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l3473_347358


namespace NUMINAMATH_CALUDE_angle_C_value_l3473_347351

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- State the theorem
theorem angle_C_value (t : Triangle) 
  (h : t.a^2 + t.b^2 - t.c^2 + t.a * t.b = 0) : 
  t.angleC = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l3473_347351


namespace NUMINAMATH_CALUDE_x_values_l3473_347388

theorem x_values (x n : ℕ) (h1 : x = 2^n - 32) 
  (h2 : (Nat.factors x).card = 3) 
  (h3 : 3 ∈ Nat.factors x) : 
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_values_l3473_347388


namespace NUMINAMATH_CALUDE_tenth_valid_number_l3473_347350

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 13

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem tenth_valid_number : nth_valid_number 10 = 166 := sorry

end NUMINAMATH_CALUDE_tenth_valid_number_l3473_347350


namespace NUMINAMATH_CALUDE_inequality_proof_l3473_347310

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3473_347310


namespace NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_yoongi_hoseok_age_sum_proof_l3473_347378

/-- The sum of Yoongi's and Hoseok's ages is 26 years -/
theorem yoongi_hoseok_age_sum : ℕ → ℕ → ℕ → Prop :=
  fun yoongi_age hoseok_age aunt_age =>
    (aunt_age = yoongi_age + 23) →
    (yoongi_age = hoseok_age + 4) →
    (aunt_age = 38) →
    (yoongi_age + hoseok_age = 26)

/-- Proof of the theorem -/
theorem yoongi_hoseok_age_sum_proof :
  ∃ (yoongi_age hoseok_age aunt_age : ℕ),
    yoongi_hoseok_age_sum yoongi_age hoseok_age aunt_age :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_yoongi_hoseok_age_sum_proof_l3473_347378


namespace NUMINAMATH_CALUDE_equation_solution_l3473_347363

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (-7 + Real.sqrt 105) / 4) ∧ 
    (x₂ = (-7 - Real.sqrt 105) / 4) ∧ 
    (∀ x : ℝ, (4 * x^2 + 8 * x - 5 ≠ 0) → (2 * x - 1 ≠ 0) → 
      ((3 * x - 7) / (4 * x^2 + 8 * x - 5) = x / (2 * x - 1)) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3473_347363


namespace NUMINAMATH_CALUDE_series_solution_l3473_347373

-- Define the series
def S (x y : ℝ) : ℝ := 1 + 2*x*y + 3*(x*y)^2 + 4*(x*y)^3 + 5*(x*y)^4 + 6*(x*y)^5 + 7*(x*y)^6 + 8*(x*y)^7

-- State the theorem
theorem series_solution :
  ∃ (x y : ℝ), S x y = 16 ∧ x = 3/4 ∧ (y = 1 ∨ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_series_solution_l3473_347373


namespace NUMINAMATH_CALUDE_sticker_difference_l3473_347367

/-- Proves the difference in stickers received by Mandy and Justin -/
theorem sticker_difference (initial_stickers : ℕ) 
  (friends : ℕ) (stickers_per_friend : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 72 →
  friends = 3 →
  stickers_per_friend = 4 →
  remaining_stickers = 42 →
  ∃ (mandy_stickers justin_stickers : ℕ),
    mandy_stickers = friends * stickers_per_friend + 2 ∧
    justin_stickers < mandy_stickers ∧
    initial_stickers = remaining_stickers + friends * stickers_per_friend + mandy_stickers + justin_stickers ∧
    mandy_stickers - justin_stickers = 10 :=
by sorry

end NUMINAMATH_CALUDE_sticker_difference_l3473_347367


namespace NUMINAMATH_CALUDE_bottle_caps_remaining_l3473_347379

theorem bottle_caps_remaining (initial_caps : ℕ) (removed_caps : ℕ) :
  initial_caps = 16 → removed_caps = 6 → initial_caps - removed_caps = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_remaining_l3473_347379


namespace NUMINAMATH_CALUDE_problem_1_l3473_347369

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}
def C : Set ℝ := {0, 1/3, 1/5}

theorem problem_1 : ∀ a : ℝ, B a ⊆ A ↔ a ∈ C := by sorry

end NUMINAMATH_CALUDE_problem_1_l3473_347369


namespace NUMINAMATH_CALUDE_equation_solutions_l3473_347365

-- Define the equation
def equation (a x : ℝ) : Prop :=
  ((1 - x^2)^2 + 2*a^2 + 5*a)^7 - ((3*a + 2)*(1 - x^2) + 3)^7 = 
  5 - 2*a - (3*a + 2)*x^2 - 2*a^2 - (1 - x^2)^2

-- Define the interval
def in_interval (x : ℝ) : Prop :=
  -Real.sqrt 6 / 2 ≤ x ∧ x ≤ Real.sqrt 2

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ in_interval x₁ ∧ in_interval x₂ ∧ 
  equation a x₁ ∧ equation a x₂

-- State the theorem
theorem equation_solutions :
  ∀ a : ℝ, has_two_distinct_solutions a ↔ 
  (0.25 ≤ a ∧ a < 1) ∨ (-3.5 ≤ a ∧ a < -2) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3473_347365


namespace NUMINAMATH_CALUDE_smallest_k_value_l3473_347390

/-- Given positive integers a, b, c, d, e, and k satisfying the conditions,
    prove that the smallest possible value for k is 522. -/
theorem smallest_k_value (a b c d e k : ℕ+) 
  (eq1 : a + 2*b + 3*c + 4*d + 5*e = k)
  (eq2 : 5*a = 4*b)
  (eq3 : 4*b = 3*c)
  (eq4 : 3*c = 2*d)
  (eq5 : 2*d = e) :
  k ≥ 522 ∧ (∃ (a' b' c' d' e' : ℕ+), 
    a' + 2*b' + 3*c' + 4*d' + 5*e' = 522 ∧
    5*a' = 4*b' ∧ 4*b' = 3*c' ∧ 3*c' = 2*d' ∧ 2*d' = e') := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3473_347390


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3473_347344

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (s : ℝ) 
  (h1 : r = 150 * Real.sqrt 3) 
  (h2 : s = 150) : 
  ∃ (x : ℝ), x = 150 * (Real.sqrt 3 - 3) ∧ 
  (s + s + s + x)^2 = 3 * (2 * r)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3473_347344


namespace NUMINAMATH_CALUDE_mother_hubbard_children_l3473_347336

theorem mother_hubbard_children (total_bar : ℚ) (children : ℕ) : 
  total_bar = 1 →
  (total_bar - total_bar / 3) = (children * (total_bar / 12)) →
  children = 8 := by
  sorry

end NUMINAMATH_CALUDE_mother_hubbard_children_l3473_347336


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3473_347322

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = (4/3) * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 70°
  max a (max b c) = 70 := by
sorry


end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3473_347322


namespace NUMINAMATH_CALUDE_third_shiny_on_fifth_probability_l3473_347305

def total_pennies : ℕ := 10
def shiny_pennies : ℕ := 5
def dull_pennies : ℕ := 5
def draws : ℕ := 5

def probability_third_shiny_on_fifth : ℚ :=
  (Nat.choose 4 2 * Nat.choose 6 2) / Nat.choose total_pennies draws

theorem third_shiny_on_fifth_probability :
  probability_third_shiny_on_fifth = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_third_shiny_on_fifth_probability_l3473_347305


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_altitudes_collinear_l3473_347318

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

structure Quadrilateral : Type :=
  (A B C D : Point)

-- Define the properties
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

def is_altitude (P Q R S : Point) : Prop := sorry

-- Define the theorem
theorem circumscribed_quadrilateral_altitudes_collinear 
  (ABCD : Quadrilateral) (O : Point) (c : Circle) 
  (A₁ B₁ C₁ D₁ : Point) : 
  is_circumscribed ABCD c →
  c.center = O →
  is_altitude A O B A₁ →
  is_altitude B O A B₁ →
  is_altitude C O D C₁ →
  is_altitude D O C D₁ →
  ∃ (l : Set Point), A₁ ∈ l ∧ B₁ ∈ l ∧ C₁ ∈ l ∧ D₁ ∈ l ∧ 
    ∀ (P Q : Point), P ∈ l → Q ∈ l → ∃ (t : ℝ), Q.x = P.x + t * (Q.y - P.y) :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_altitudes_collinear_l3473_347318


namespace NUMINAMATH_CALUDE_triangle_area_l3473_347381

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b^2 + c^2 = a^2 - b*c →
  (a * b * Real.cos C) = -4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3473_347381


namespace NUMINAMATH_CALUDE_shopkeeper_change_l3473_347346

/-- Represents the change given by the shopkeeper -/
structure Change where
  total_bills : ℕ
  bill_value_1 : ℕ
  bill_value_2 : ℕ
  noodles_value : ℕ

/-- The problem statement -/
theorem shopkeeper_change (c : Change) (h1 : c.total_bills = 16)
    (h2 : c.bill_value_1 = 10) (h3 : c.bill_value_2 = 5) (h4 : c.noodles_value = 5)
    (h5 : 100 = c.noodles_value + c.bill_value_1 * x + c.bill_value_2 * (c.total_bills - x)) :
    x = 3 :=
  sorry

end NUMINAMATH_CALUDE_shopkeeper_change_l3473_347346


namespace NUMINAMATH_CALUDE_total_earnings_of_three_workers_l3473_347361

/-- The total earnings of three workers given their combined earnings -/
theorem total_earnings_of_three_workers
  (earnings_a : ℕ) (earnings_b : ℕ) (earnings_c : ℕ)
  (h1 : earnings_a + earnings_c = 400)
  (h2 : earnings_b + earnings_c = 300)
  (h3 : earnings_c = 100) :
  earnings_a + earnings_b + earnings_c = 600 :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_of_three_workers_l3473_347361


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l3473_347309

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l3473_347309


namespace NUMINAMATH_CALUDE_jill_has_six_peaches_l3473_347377

-- Define the number of peaches each person has
def steven_peaches : ℕ := 19
def jake_peaches : ℕ := steven_peaches - 18
def jill_peaches : ℕ := steven_peaches - 13

-- Theorem to prove
theorem jill_has_six_peaches : jill_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_jill_has_six_peaches_l3473_347377


namespace NUMINAMATH_CALUDE_number_difference_l3473_347334

theorem number_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) : 
  |x - y| = 7 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3473_347334


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3473_347323

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 24 - 16 * Complex.I ∧ Complex.abs w = Real.sqrt 52 →
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3473_347323


namespace NUMINAMATH_CALUDE_next_coincidence_after_lcm_robinsons_next_busy_day_l3473_347335

/-- Represents a periodic event --/
structure PeriodicEvent where
  period : ℕ

/-- Calculates the least common multiple (LCM) of a list of natural numbers --/
def lcmList (list : List ℕ) : ℕ :=
  list.foldl Nat.lcm 1

/-- Theorem: The next coincidence of periodic events occurs after their LCM --/
theorem next_coincidence_after_lcm (events : List PeriodicEvent) : 
  let periods := events.map (·.period)
  let nextCoincidence := lcmList periods
  ∀ t : ℕ, t < nextCoincidence → ¬ (∀ e ∈ events, t % e.period = 0) :=
by sorry

/-- Robinson Crusoe's activities --/
def robinsons_activities : List PeriodicEvent := [
  { period := 2 },  -- Water replenishment
  { period := 3 },  -- Fruit collection
  { period := 5 }   -- Hunting
]

/-- Theorem: Robinson's next busy day is 30 days after the current busy day --/
theorem robinsons_next_busy_day :
  lcmList (robinsons_activities.map (·.period)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_next_coincidence_after_lcm_robinsons_next_busy_day_l3473_347335


namespace NUMINAMATH_CALUDE_num_adults_on_trip_l3473_347398

def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def num_girls : ℕ := 7
def num_boys : ℕ := 10
def eggs_per_girl : ℕ := 1
def eggs_per_boy : ℕ := eggs_per_girl + 1

theorem num_adults_on_trip : 
  total_eggs - (num_girls * eggs_per_girl + num_boys * eggs_per_boy) = 3 * eggs_per_adult := by
  sorry

end NUMINAMATH_CALUDE_num_adults_on_trip_l3473_347398


namespace NUMINAMATH_CALUDE_no_such_polynomials_l3473_347319

/-- A polynomial is a perfect square if it's the square of another non-constant polynomial -/
def IsPerfectSquare (p : Polynomial ℝ) : Prop :=
  ∃ q : Polynomial ℝ, q.degree > 0 ∧ p = q^2

theorem no_such_polynomials :
  ¬∃ (f g : Polynomial ℝ),
    f.degree > 0 ∧ g.degree > 0 ∧
    ¬IsPerfectSquare f ∧
    ¬IsPerfectSquare g ∧
    IsPerfectSquare (f.comp g) ∧
    IsPerfectSquare (g.comp f) :=
by sorry

end NUMINAMATH_CALUDE_no_such_polynomials_l3473_347319


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3473_347342

theorem container_volume_ratio 
  (container1 container2 container3 : ℝ) 
  (h1 : 3/5 * container1 = 2/3 * container2) 
  (h2 : 2/3 * container2 - 1/2 * container3 = 1/2 * container3) 
  (h3 : container1 > 0) 
  (h4 : container2 > 0) 
  (h5 : container3 > 0) : 
  container2 / container3 = 2/3 := by
sorry


end NUMINAMATH_CALUDE_container_volume_ratio_l3473_347342


namespace NUMINAMATH_CALUDE_no_ten_goals_possible_l3473_347304

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the number of goals scored by each player -/
def GoalCount := Player → ℕ

/-- Represents the statements made by each player -/
def Statements := Player → Player → ℕ

/-- Checks if the statements are consistent with the goal count and the truth-lie condition -/
def ConsistentStatements (gc : GoalCount) (s : Statements) : Prop :=
  ∀ p : Player, (s p p = gc p ∧ s p (nextPlayer p) ≠ gc (nextPlayer p)) ∨
                (s p p ≠ gc p ∧ s p (nextPlayer p) = gc (nextPlayer p))
where
  nextPlayer : Player → Player
  | Player.Anton => Player.Ilya
  | Player.Ilya => Player.Sergey
  | Player.Sergey => Player.Anton

/-- The main theorem stating that it's impossible to have a total of 10 goals -/
theorem no_ten_goals_possible (gc : GoalCount) (s : Statements) :
  ConsistentStatements gc s → (gc Player.Anton + gc Player.Ilya + gc Player.Sergey ≠ 10) := by
  sorry

end NUMINAMATH_CALUDE_no_ten_goals_possible_l3473_347304


namespace NUMINAMATH_CALUDE_right_triangle_quadratic_roots_l3473_347321

theorem right_triangle_quadratic_roots (m : ℝ) : 
  let f := fun x : ℝ => x^2 - (2*m - 1)*x + 4*(m - 1)
  ∃ (a b : ℝ), 
    (f a = 0 ∧ f b = 0) ∧  -- BC and AC are roots of the quadratic equation
    (a ≠ b) ∧               -- Distinct roots
    (a > 0 ∧ b > 0) ∧       -- Positive lengths
    (a^2 + b^2 = 25) →      -- Pythagorean theorem (AB^2 = 5^2 = 25)
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_quadratic_roots_l3473_347321


namespace NUMINAMATH_CALUDE_no_regular_polygon_inscription_l3473_347375

-- Define an ellipse with unequal axes
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ
  axesUnequal : majorAxis ≠ minorAxis

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  moreThanFourSides : sides > 4

-- Define the concept of inscribing a polygon in an ellipse
def isInscribed (p : RegularPolygon) (e : Ellipse) : Prop :=
  sorry -- Definition of inscription

-- Theorem statement
theorem no_regular_polygon_inscription 
  (e : Ellipse) (p : RegularPolygon) : ¬ isInscribed p e := by
  sorry

#check no_regular_polygon_inscription

end NUMINAMATH_CALUDE_no_regular_polygon_inscription_l3473_347375


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3473_347380

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3473_347380


namespace NUMINAMATH_CALUDE_remainder_theorem_l3473_347311

theorem remainder_theorem : 4 * 6^24 + 3^48 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3473_347311


namespace NUMINAMATH_CALUDE_ratio_problem_l3473_347300

theorem ratio_problem (x y : ℤ) : 
  (y = 3 * x) → -- The two integers are in the ratio of 1 to 3
  (x + 10 = y) → -- Adding 10 to the smaller number makes them equal
  y = 15 := by -- The larger integer is 15
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3473_347300


namespace NUMINAMATH_CALUDE_fish_lives_12_years_l3473_347348

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by
  sorry

end NUMINAMATH_CALUDE_fish_lives_12_years_l3473_347348


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l3473_347320

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l3473_347320


namespace NUMINAMATH_CALUDE_simplify_expression_l3473_347325

theorem simplify_expression (y : ℝ) : 4*y + 5*y + 6*y + 2 = 15*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3473_347325


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l3473_347315

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_equals_two :
  bracket (bracket 60 30 90) (bracket 2 1 3) (bracket 10 5 15) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l3473_347315


namespace NUMINAMATH_CALUDE_mikes_total_work_hours_l3473_347341

/-- Calculates the total hours worked given a work schedule --/
def totalHoursWorked (hours_per_day1 hours_per_day2 hours_per_day3 : ℕ) 
                     (days1 days2 days3 : ℕ) : ℕ :=
  hours_per_day1 * days1 + hours_per_day2 * days2 + hours_per_day3 * days3

/-- Proves that Mike's total work hours is 93 --/
theorem mikes_total_work_hours :
  totalHoursWorked 3 4 5 5 7 10 = 93 := by
  sorry

#eval totalHoursWorked 3 4 5 5 7 10

end NUMINAMATH_CALUDE_mikes_total_work_hours_l3473_347341


namespace NUMINAMATH_CALUDE_equidistant_point_on_number_line_l3473_347302

/-- Given points A (-1) and B (5) on a number line, if point P is equidistant from A and B, then P represents the number 2. -/
theorem equidistant_point_on_number_line :
  let a : ℝ := -1
  let b : ℝ := 5
  ∀ p : ℝ, |p - a| = |p - b| → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_number_line_l3473_347302


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3473_347391

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3473_347391


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3473_347382

theorem greatest_whole_number_inequality (x : ℤ) : 
  (6 * x - 4 < 5 - 3 * x) → x ≤ 0 :=
by sorry

theorem zero_satisfies_inequality : 
  6 * 0 - 4 < 5 - 3 * 0 :=
by sorry

theorem no_positive_integer_satisfies_inequality (x : ℤ) :
  x > 0 → ¬(6 * x - 4 < 5 - 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3473_347382


namespace NUMINAMATH_CALUDE_triangle_problem_l3473_347306

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) :
  (t.a = 3 ∧ t.b = 5 ∧ t.B = 2 * π / 3 → Real.sin t.A = 3 * Real.sqrt 3 / 10) ∧
  (t.a = 3 ∧ t.b = 5 ∧ t.C = 2 * π / 3 → t.c = 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3473_347306


namespace NUMINAMATH_CALUDE_g_of_2_l3473_347328

/-- Given a function g(x) = px^8 + qx^4 + rx + 7 where g(-2) = -5,
    prove that g(2) = 2p(256) + 2q(16) + 19 -/
theorem g_of_2 (p q r : ℝ) (g : ℝ → ℝ) 
    (h1 : ∀ x, g x = p * x^8 + q * x^4 + r * x + 7)
    (h2 : g (-2) = -5) :
  g 2 = 2 * p * 256 + 2 * q * 16 + 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_l3473_347328


namespace NUMINAMATH_CALUDE_min_value_x_l3473_347372

theorem min_value_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : ∀ (a b : ℝ), a > 0 → b > 0 → 1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2)
  (h2 : ∀ (a b : ℝ), a > 0 → b > 0 → 4 * a + b * (1 - a) = 0) :
  x ≥ 1 ∧ ∀ (y : ℝ), y > 0 → y < 1 → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / a^2 + 16 / b^2 < 1 + y / 2 - y^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_l3473_347372


namespace NUMINAMATH_CALUDE_line_increase_percentage_l3473_347376

/-- Given that increasing the number of lines by 60 results in 240 lines,
    prove that the percentage increase is 100/3%. -/
theorem line_increase_percentage : ℝ → Prop :=
  fun original_lines =>
    (original_lines + 60 = 240) →
    ((60 / original_lines) * 100 = 100 / 3)

/-- Proof of the theorem -/
lemma prove_line_increase_percentage : ∃ x : ℝ, line_increase_percentage x := by
  sorry

end NUMINAMATH_CALUDE_line_increase_percentage_l3473_347376


namespace NUMINAMATH_CALUDE_vector_relations_l3473_347355

def a : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_relations :
  (parallel (a.1 + (b (-1)).1, a.2 + (b (-1)).2) c) ∧
  (perpendicular (a.1 + (b (3/2)).1, a.2 + (b (3/2)).2) c) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l3473_347355


namespace NUMINAMATH_CALUDE_toph_fish_count_l3473_347349

theorem toph_fish_count (total_people : ℕ) (average_fish : ℕ) (aang_fish : ℕ) (sokka_fish : ℕ) :
  total_people = 3 →
  average_fish = 8 →
  aang_fish = 7 →
  sokka_fish = 5 →
  average_fish * total_people - aang_fish - sokka_fish = 12 :=
by sorry

end NUMINAMATH_CALUDE_toph_fish_count_l3473_347349


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_f_values_l3473_347397

theorem infinite_primes_dividing_f_values
  (f : ℕ+ → ℕ+)
  (h_non_constant : ∃ a b : ℕ+, f a ≠ f b)
  (h_divides : ∀ a b : ℕ+, a ≠ b → (a - b) ∣ (f a - f b)) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ c : ℕ+, p ∣ f c} :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_f_values_l3473_347397


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3473_347317

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3473_347317


namespace NUMINAMATH_CALUDE_train_speed_l3473_347354

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 140)
  (h2 : platform_length = 260)
  (h3 : time_to_pass = 23.998080153587715) : 
  ∃ (speed : ℝ), abs (speed - 60.0048) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3473_347354


namespace NUMINAMATH_CALUDE_distance_between_projections_l3473_347308

/-- Given a point A(-1, 2, -3) in ℝ³, prove that the distance between its projection
    onto the yOz plane and its projection onto the x-axis is √14. -/
theorem distance_between_projections :
  let A : ℝ × ℝ × ℝ := (-1, 2, -3)
  let P₁ : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)  -- projection onto yOz plane
  let P₂ : ℝ × ℝ × ℝ := (A.1, 0, 0)        -- projection onto x-axis
  (P₁.1 - P₂.1)^2 + (P₁.2.1 - P₂.2.1)^2 + (P₁.2.2 - P₂.2.2)^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_projections_l3473_347308


namespace NUMINAMATH_CALUDE_solve_homework_problem_l3473_347312

def homework_problem (total_problems : ℕ) (completed_at_stop1 : ℕ) (completed_at_stop2 : ℕ) (completed_at_stop3 : ℕ) : Prop :=
  let completed_on_bus := completed_at_stop1 + completed_at_stop2 + completed_at_stop3
  let remaining_problems := total_problems - completed_on_bus
  remaining_problems = 3

theorem solve_homework_problem :
  homework_problem 9 2 3 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_homework_problem_l3473_347312


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3473_347357

/-- The equation of the tangent line to y = 2ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_at_origin (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.log (x + 1)
  let f' : ℝ → ℝ := λ x => 2 / (x + 1)
  let tangent_line : ℝ → ℝ := λ x => 2 * x
  (∀ x, HasDerivAt f (f' x) x) →
  HasDerivAt f 2 0 →
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - tangent_line x| ≤ ε * |x|
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3473_347357


namespace NUMINAMATH_CALUDE_finley_tickets_l3473_347387

theorem finley_tickets (total_tickets : ℕ) (ratio_jensen : ℕ) (ratio_finley : ℕ) : 
  total_tickets = 400 →
  ratio_jensen = 4 →
  ratio_finley = 11 →
  (3 * total_tickets / 4) * ratio_finley / (ratio_jensen + ratio_finley) = 220 := by
  sorry

end NUMINAMATH_CALUDE_finley_tickets_l3473_347387


namespace NUMINAMATH_CALUDE_number_of_planes_l3473_347329

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- Theorem: The number of commercial planes is 25 -/
theorem number_of_planes : 
  (total_wings / wings_per_plane : ℕ) = 25 := by sorry

end NUMINAMATH_CALUDE_number_of_planes_l3473_347329


namespace NUMINAMATH_CALUDE_equation_is_linear_l3473_347326

/-- A linear equation in one variable -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Check if an equation is a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The specific equation 3x = 2x -/
def f (x : ℝ) : ℝ := 3 * x - 2 * x

theorem equation_is_linear : is_linear_equation f := by sorry

end NUMINAMATH_CALUDE_equation_is_linear_l3473_347326


namespace NUMINAMATH_CALUDE_pencil_box_cost_l3473_347384

/-- The cost of Linda's purchases -/
def purchase_cost (notebook_price : ℝ) (notebook_quantity : ℕ) (pen_price : ℝ) (pencil_price : ℝ) : ℝ :=
  notebook_price * notebook_quantity + pen_price + pencil_price

/-- The theorem stating the cost of the box of pencils -/
theorem pencil_box_cost : 
  ∃ (pencil_price : ℝ),
    purchase_cost 1.20 3 1.70 pencil_price = 6.80 ∧ 
    pencil_price = 1.50 := by
  sorry

#check pencil_box_cost

end NUMINAMATH_CALUDE_pencil_box_cost_l3473_347384


namespace NUMINAMATH_CALUDE_number_plus_ten_l3473_347339

theorem number_plus_ten (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_ten_l3473_347339


namespace NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l3473_347393

theorem function_always_negative_iff_a_in_range :
  ∀ (a : ℝ), (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l3473_347393


namespace NUMINAMATH_CALUDE_complex_equation_product_l3473_347374

theorem complex_equation_product (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l3473_347374
