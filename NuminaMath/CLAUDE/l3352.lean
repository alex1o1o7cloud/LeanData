import Mathlib

namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l3352_335272

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + y) = 5 * x + y) :
  ∀ z : ℝ, 2 * x + y ≥ 9 ∧ (∃ x₀ y₀ : ℝ, 2 * x₀ + y₀ = 9 ∧ x₀ > 0 ∧ y₀ > 0 ∧ x₀ * (x₀ + y₀) = 5 * x₀ + y₀) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l3352_335272


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3352_335289

/-- Given a square with perimeter 12p, its area is 9p^2 -/
theorem square_area_from_perimeter (p : ℝ) :
  let perimeter : ℝ := 12 * p
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 9 * p ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3352_335289


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l3352_335246

/-- The number of seconds between each doubling of bacteria -/
def doubling_period : ℕ := 30

/-- The total time elapsed in seconds -/
def total_time : ℕ := 150

/-- The number of bacteria after the total time has elapsed -/
def final_bacteria_count : ℕ := 20480

/-- The number of doubling periods that have occurred -/
def num_doubling_periods : ℕ := total_time / doubling_period

theorem initial_bacteria_count : 
  ∃ (initial_count : ℕ), initial_count * (2^num_doubling_periods) = final_bacteria_count ∧ 
                          initial_count = 640 := by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l3352_335246


namespace NUMINAMATH_CALUDE_not_perfect_square_with_1234_divisors_l3352_335213

/-- A natural number with exactly 1234 divisors is not a perfect square. -/
theorem not_perfect_square_with_1234_divisors (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x | x ∣ n} ∧ d.card = 1234) → ¬∃ (m : ℕ), n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_with_1234_divisors_l3352_335213


namespace NUMINAMATH_CALUDE_tourist_distribution_l3352_335251

theorem tourist_distribution (total_tourists : ℕ) (h1 : total_tourists = 737) :
  ∃! (num_cars tourists_per_car : ℕ),
    num_cars * tourists_per_car = total_tourists ∧
    num_cars > 0 ∧
    tourists_per_car > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_l3352_335251


namespace NUMINAMATH_CALUDE_game_probability_l3352_335292

theorem game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 2/21)
  (h_outcomes : ∃ (p_win : ℚ), p_win + p_lose + p_tie = 1) :
  ∃ (p_win : ℚ), p_win = 10/21 := by
sorry

end NUMINAMATH_CALUDE_game_probability_l3352_335292


namespace NUMINAMATH_CALUDE_dolls_in_big_box_l3352_335204

/-- Given information about big and small boxes containing dolls, 
    prove that each big box contains 7 dolls. -/
theorem dolls_in_big_box 
  (num_big_boxes : ℕ) 
  (num_small_boxes : ℕ) 
  (dolls_per_small_box : ℕ) 
  (total_dolls : ℕ) 
  (h1 : num_big_boxes = 5)
  (h2 : num_small_boxes = 9)
  (h3 : dolls_per_small_box = 4)
  (h4 : total_dolls = 71) :
  ∃ (dolls_per_big_box : ℕ), 
    dolls_per_big_box * num_big_boxes + 
    dolls_per_small_box * num_small_boxes = total_dolls ∧ 
    dolls_per_big_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_dolls_in_big_box_l3352_335204


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l3352_335291

/-- The number of intersections created by n non-parallel straight streets -/
def intersections (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: 10 non-parallel straight streets create 45 intersections -/
theorem ten_streets_intersections :
  intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l3352_335291


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3352_335277

theorem quadratic_equation_solution :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x^2 - 6*x + 18 = 28 ↔ (x = a ∨ x = b)) →
  a ≥ b →
  a + 3*b = 12 - 2*Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3352_335277


namespace NUMINAMATH_CALUDE_total_books_l3352_335237

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3352_335237


namespace NUMINAMATH_CALUDE_sin_eq_cos_condition_l3352_335233

open Real

theorem sin_eq_cos_condition (α : ℝ) :
  (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α ∧
  ¬ (sin α = cos α → ∃ k : ℤ, α = π / 4 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_cos_condition_l3352_335233


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_empty_l3352_335266

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9 < 0}
def B : Set ℝ := {x | 2 ≤ x + 1 ∧ x + 1 ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_empty (m : ℝ) : 
  A ∩ C m = ∅ ↔ m ≤ -4 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_empty_l3352_335266


namespace NUMINAMATH_CALUDE_car_wash_earnings_l3352_335281

theorem car_wash_earnings (total : ℝ) (lisa : ℝ) (tommy : ℝ) : 
  total = 60 →
  lisa = total / 2 →
  tommy = lisa / 2 →
  lisa - tommy = 15 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l3352_335281


namespace NUMINAMATH_CALUDE_empty_solution_set_range_subset_solution_set_range_l3352_335264

/-- The solution set of the quadratic inequality mx² - (m+1)x + (m+1) ≥ 0 -/
def solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | m * x^2 - (m + 1) * x + (m + 1) ≥ 0}

/-- The range of m for which the solution set is empty -/
theorem empty_solution_set_range : 
  ∀ m : ℝ, solution_set m = ∅ ↔ m < -1 :=
sorry

/-- The range of m for which (1,+∞) is a subset of the solution set -/
theorem subset_solution_set_range : 
  ∀ m : ℝ, Set.Ioi 1 ⊆ solution_set m ↔ m ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_subset_solution_set_range_l3352_335264


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2023_l3352_335216

theorem units_digit_of_2_pow_2023 : 2^2023 % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2023_l3352_335216


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3352_335220

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 109 ∧
  (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 19 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3352_335220


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3352_335257

theorem inequalities_theorem (a b c : ℝ) 
  (ha : a < 0) 
  (hab : a < b) 
  (hb : b ≤ 0) 
  (hbc : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3352_335257


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3352_335286

/-- The size of the multiplication table (16 x 16) -/
def tableSize : ℕ := 16

/-- The number of odd numbers from 0 to 15 -/
def oddCount : ℕ := 8

/-- The total number of entries in the multiplication table -/
def totalEntries : ℕ := tableSize * tableSize

/-- The number of odd entries in the multiplication table -/
def oddEntries : ℕ := oddCount * oddCount

theorem multiplication_table_odd_fraction :
  (oddEntries : ℚ) / totalEntries = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3352_335286


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3352_335201

theorem imaginary_part_of_z (z : ℂ) : z = (2 + Complex.I) / (1 + Complex.I)^2 → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3352_335201


namespace NUMINAMATH_CALUDE_eliza_height_is_83_l3352_335241

/-- The height of Eliza given the heights of her siblings -/
def elizaHeight (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) 
  (sibling3_height : ℕ) (sibling4_height : ℕ) (sibling5_height : ℕ) : ℕ :=
  total_height - (sibling1_height + sibling2_height + sibling3_height + sibling4_height + sibling5_height)

theorem eliza_height_is_83 :
  let total_height := 435
  let sibling1_height := 66
  let sibling2_height := 66
  let sibling3_height := 60
  let sibling4_height := 75
  let sibling5_height := elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height 85 + 2
  elizaHeight total_height sibling1_height sibling2_height sibling3_height sibling4_height sibling5_height = 83 := by
  sorry

end NUMINAMATH_CALUDE_eliza_height_is_83_l3352_335241


namespace NUMINAMATH_CALUDE_existence_of_mn_l3352_335219

theorem existence_of_mn (k : ℕ+) : 
  (∃ m n : ℕ+, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_existence_of_mn_l3352_335219


namespace NUMINAMATH_CALUDE_f_eq_g_l3352_335244

/-- The given polynomial function f(x, y, z) -/
def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) +
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) +
  (x^2 - y^2) * (1 + y*z) * (1 + x*z)

/-- The factored form of the polynomial -/
def g (x y z : ℝ) : ℝ :=
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z)

/-- Theorem stating that f and g are equivalent for all real x, y, and z -/
theorem f_eq_g : ∀ x y z : ℝ, f x y z = g x y z := by
  sorry

end NUMINAMATH_CALUDE_f_eq_g_l3352_335244


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3352_335248

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (52 * x + 14) % 24 = 6 ∧
  ∀ (y : ℕ), y > 0 ∧ (52 * y + 14) % 24 = 6 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3352_335248


namespace NUMINAMATH_CALUDE_remainder_problem_l3352_335208

theorem remainder_problem (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3352_335208


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3352_335235

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3352_335235


namespace NUMINAMATH_CALUDE_tetrahedron_bug_return_probability_l3352_335230

/-- Probability of returning to the starting vertex after n steps in a regular tetrahedron -/
def return_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - return_probability n) / 3

/-- The probability of returning to the starting vertex after 8 steps is 547/2187 -/
theorem tetrahedron_bug_return_probability :
  return_probability 8 = 547 / 2187 := by
  sorry

#eval return_probability 8

end NUMINAMATH_CALUDE_tetrahedron_bug_return_probability_l3352_335230


namespace NUMINAMATH_CALUDE_sum_of_powers_and_reciprocals_is_integer_l3352_335218

theorem sum_of_powers_and_reciprocals_is_integer
  (x : ℝ)
  (h : ∃ (k : ℤ), x + 1 / x = k)
  (n : ℕ)
  : ∃ (m : ℤ), x^n + 1 / x^n = m :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_and_reciprocals_is_integer_l3352_335218


namespace NUMINAMATH_CALUDE_recreational_area_diameter_l3352_335229

/-- The diameter of the outer boundary of a circular recreational area -/
def outer_boundary_diameter (pond_diameter : ℝ) (flowerbed_width : ℝ) (jogging_path_width : ℝ) : ℝ :=
  pond_diameter + 2 * (flowerbed_width + jogging_path_width)

/-- Theorem: The diameter of the outer boundary of the circular recreational area is 64 feet -/
theorem recreational_area_diameter : 
  outer_boundary_diameter 20 10 12 = 64 := by sorry

end NUMINAMATH_CALUDE_recreational_area_diameter_l3352_335229


namespace NUMINAMATH_CALUDE_park_trees_l3352_335273

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: The park will have 77 walnut trees after planting -/
theorem park_trees : total_trees 33 44 = 77 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l3352_335273


namespace NUMINAMATH_CALUDE_distance_calculation_l3352_335249

/-- The distance between Xiao Ming's home and his grandmother's house -/
def distance_to_grandma : ℝ := 36

/-- Xiao Ming's speed in km/h -/
def xiao_ming_speed : ℝ := 12

/-- Father's speed in km/h -/
def father_speed : ℝ := 36

/-- Time Xiao Ming departs before his father in hours -/
def time_before_father : ℝ := 2.5

/-- Time father arrives after Xiao Ming in hours -/
def time_after_xiao_ming : ℝ := 0.5

theorem distance_calculation :
  ∃ (t : ℝ),
    t > 0 ∧
    distance_to_grandma = father_speed * t ∧
    distance_to_grandma = xiao_ming_speed * (t + time_before_father - time_after_xiao_ming) :=
by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l3352_335249


namespace NUMINAMATH_CALUDE_inequality_proof_l3352_335232

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3352_335232


namespace NUMINAMATH_CALUDE_prob_red_ball_specific_bag_l3352_335298

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem stating the probability of drawing a red ball from a specific bag -/
theorem prob_red_ball_specific_bag :
  let bag : ColoredBalls := { total := 9, red := 6, green := 3 }
  prob_red_ball bag = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_prob_red_ball_specific_bag_l3352_335298


namespace NUMINAMATH_CALUDE_complex_symmetry_l3352_335203

theorem complex_symmetry (z₁ z₂ : ℂ) : 
  (z₁ = 2 - 3*I) → (z₁ = -z₂) → (z₂ = -2 + 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_l3352_335203


namespace NUMINAMATH_CALUDE_solution_to_equation_l3352_335206

theorem solution_to_equation : 
  {x : ℝ | x = (1/x) + (-x)^2 + 3} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3352_335206


namespace NUMINAMATH_CALUDE_trig_identity_l3352_335271

theorem trig_identity (θ : ℝ) (h : Real.sin (π + θ) = 1/4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.sin (π/2 - θ)) / (Real.cos (θ + 2*π) * Real.cos (π + θ) + Real.cos (-θ)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3352_335271


namespace NUMINAMATH_CALUDE_jimmy_yellow_marbles_l3352_335285

theorem jimmy_yellow_marbles :
  ∀ (lorin_black jimmy_yellow alex_total : ℕ),
    lorin_black = 4 →
    alex_total = 19 →
    alex_total = 2 * lorin_black + (jimmy_yellow / 2) →
    jimmy_yellow = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_yellow_marbles_l3352_335285


namespace NUMINAMATH_CALUDE_probability_inconsistency_l3352_335212

-- Define the probability measure
variable (p : Set ℝ → ℝ)

-- Define events a and b
variable (a b : Set ℝ)

-- State the given probabilities
axiom pa : p a = 0.18
axiom pb : p b = 0.5
axiom pab : p (a ∩ b) = 0.36

-- Theorem to prove the inconsistency
theorem probability_inconsistency :
  ¬(0 ≤ p a ∧ p a ≤ 1 ∧
    0 ≤ p b ∧ p b ≤ 1 ∧
    0 ≤ p (a ∩ b) ∧ p (a ∩ b) ≤ 1 ∧
    p (a ∩ b) ≤ p a ∧ p (a ∩ b) ≤ p b) :=
by sorry

end NUMINAMATH_CALUDE_probability_inconsistency_l3352_335212


namespace NUMINAMATH_CALUDE_sale_discount_theorem_l3352_335283

/-- Calculates the final amount paid after applying a discount based on the purchase amount -/
def final_amount_paid (initial_amount : ℕ) (discount_per_hundred : ℕ) : ℕ :=
  initial_amount - (initial_amount / 100) * discount_per_hundred

/-- Theorem stating that for a $250 purchase with $10 off per $100 spent, the final amount is $230 -/
theorem sale_discount_theorem :
  final_amount_paid 250 10 = 230 := by
  sorry

end NUMINAMATH_CALUDE_sale_discount_theorem_l3352_335283


namespace NUMINAMATH_CALUDE_selina_shirts_sold_l3352_335255

/-- Calculates the number of shirts Selina sold given the conditions of the problem -/
def shirts_sold (pants_price shorts_price shirt_price : ℕ) 
  (pants_sold shorts_sold : ℕ) (bought_shirt_price : ℕ) 
  (bought_shirt_count : ℕ) (money_left : ℕ) : ℕ :=
  let total_before_buying := money_left + bought_shirt_price * bought_shirt_count
  let money_from_pants_shorts := pants_price * pants_sold + shorts_price * shorts_sold
  let money_from_shirts := total_before_buying - money_from_pants_shorts
  money_from_shirts / shirt_price

theorem selina_shirts_sold : 
  shirts_sold 5 3 4 3 5 10 2 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_selina_shirts_sold_l3352_335255


namespace NUMINAMATH_CALUDE_skating_time_l3352_335258

/-- Given a distance of 80 kilometers and a speed of 10 kilometers per hour,
    the time taken is 8 hours. -/
theorem skating_time (distance : ℝ) (speed : ℝ) (time : ℝ) 
    (h1 : distance = 80)
    (h2 : speed = 10)
    (h3 : time = distance / speed) : 
  time = 8 := by
sorry

end NUMINAMATH_CALUDE_skating_time_l3352_335258


namespace NUMINAMATH_CALUDE_sasha_can_get_123_l3352_335242

/-- Represents an arithmetic expression --/
inductive Expr
  | Num : Nat → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression --/
def eval : Expr → Int
  | Expr.Num n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each number from 1 to 5 exactly once --/
def usesAllNumbers : Expr → Bool := sorry

theorem sasha_can_get_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_sasha_can_get_123_l3352_335242


namespace NUMINAMATH_CALUDE_vivi_fabric_purchase_l3352_335279

/-- The total yards of fabric Vivi bought -/
def total_yards (checkered_cost plain_cost cost_per_yard : ℚ) : ℚ :=
  checkered_cost / cost_per_yard + plain_cost / cost_per_yard

/-- Proof that Vivi bought 16 yards of fabric -/
theorem vivi_fabric_purchase :
  total_yards 75 45 (7.5 : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_vivi_fabric_purchase_l3352_335279


namespace NUMINAMATH_CALUDE_salt_mixture_price_l3352_335293

theorem salt_mixture_price (initial_salt_weight : ℝ) (initial_salt_price : ℝ) 
  (new_salt_weight : ℝ) (selling_price : ℝ) (profit_percentage : ℝ) :
  initial_salt_weight = 40 ∧ 
  initial_salt_price = 0.35 ∧
  new_salt_weight = 5 ∧
  selling_price = 0.48 ∧
  profit_percentage = 0.2 →
  ∃ (new_salt_price : ℝ),
    new_salt_price = 0.80 ∧
    (initial_salt_weight * initial_salt_price + new_salt_weight * new_salt_price) * 
      (1 + profit_percentage) = 
    (initial_salt_weight + new_salt_weight) * selling_price :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_price_l3352_335293


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3352_335256

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of measures is 5:4
  abs (a - b) = 10 :=  -- positive difference is 10°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3352_335256


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_sarah_bottle_caps_l3352_335200

theorem sarah_bottle_caps : ℕ → Prop :=
  fun initial : ℕ =>
    (initial + 3 = 29) → initial = 26

/- Proof
theorem sarah_bottle_caps : ℕ → Prop :=
  fun initial : ℕ =>
    (initial + 3 = 29) → initial = 26 :=
by
  intro initial
  intro h
  -- Proof goes here
  sorry
-/

end NUMINAMATH_CALUDE_sarah_bottle_caps_sarah_bottle_caps_l3352_335200


namespace NUMINAMATH_CALUDE_economic_formula_solution_l3352_335243

theorem economic_formula_solution (p x : ℂ) :
  (3 * p - x = 15000) → (x = 9 + 225 * Complex.I) → (p = 5003 + 75 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_economic_formula_solution_l3352_335243


namespace NUMINAMATH_CALUDE_units_digit_2137_power_753_l3352_335260

def units_digit (n : ℕ) : ℕ := n % 10

def power_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit (units_digit base ^ exp)

def cycle_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case will never be reached

theorem units_digit_2137_power_753 :
  power_units_digit 2137 753 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_2137_power_753_l3352_335260


namespace NUMINAMATH_CALUDE_stating_arithmetic_sequence_iff_60_degree_l3352_335215

/-- A triangle with interior angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : A > 0 ∧ B > 0 ∧ C > 0

/-- The interior angles of a triangle form an arithmetic sequence. -/
def arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B ∨ t.A + t.B = 2 * t.C ∨ t.B + t.C = 2 * t.A

/-- One of the interior angles of a triangle is 60 degrees. -/
def has_60_degree (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

/-- 
Theorem stating that a triangle's interior angles form an arithmetic sequence 
if and only if one of its interior angles is 60 degrees.
-/
theorem arithmetic_sequence_iff_60_degree (t : Triangle) :
  arithmetic_sequence t ↔ has_60_degree t :=
sorry

end NUMINAMATH_CALUDE_stating_arithmetic_sequence_iff_60_degree_l3352_335215


namespace NUMINAMATH_CALUDE_triangle_inequality_with_additional_segment_l3352_335214

theorem triangle_inequality_with_additional_segment
  (a b c d : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_d_positive : d > 0)
  (a₁ : ℝ) (h_a₁ : a₁ = min a d)
  (b₁ : ℝ) (h_b₁ : b₁ = min b d)
  (c₁ : ℝ) (h_c₁ : c₁ = min c d) :
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_additional_segment_l3352_335214


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l3352_335227

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l3352_335227


namespace NUMINAMATH_CALUDE_facebook_bonus_percentage_l3352_335239

/-- Represents the Facebook employee bonus problem -/
theorem facebook_bonus_percentage (total_employees : ℕ) 
  (annual_earnings : ℝ) (non_mother_women : ℕ) (bonus_per_mother : ℝ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  non_mother_women = 1200 →
  bonus_per_mother = 1250 →
  (((total_employees * 2 / 3 - non_mother_women) * bonus_per_mother) / annual_earnings) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_facebook_bonus_percentage_l3352_335239


namespace NUMINAMATH_CALUDE_composite_for_n_greater_than_two_l3352_335276

def number_with_ones_and_seven (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

theorem composite_for_n_greater_than_two :
  ∀ n : ℕ, n > 2 → ¬(Nat.Prime (number_with_ones_and_seven n)) :=
sorry

end NUMINAMATH_CALUDE_composite_for_n_greater_than_two_l3352_335276


namespace NUMINAMATH_CALUDE_soy_sauce_bottle_ounces_l3352_335278

/-- Represents the number of ounces in one cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of cups of soy sauce required for the first recipe -/
def recipe1_cups : ℕ := 2

/-- Represents the number of cups of soy sauce required for the second recipe -/
def recipe2_cups : ℕ := 1

/-- Represents the number of cups of soy sauce required for the third recipe -/
def recipe3_cups : ℕ := 3

/-- Represents the number of bottles Stephanie needs to buy -/
def bottles_needed : ℕ := 3

/-- Theorem stating that one bottle of soy sauce contains 16 ounces -/
theorem soy_sauce_bottle_ounces : 
  (recipe1_cups + recipe2_cups + recipe3_cups) * ounces_per_cup / bottles_needed = 16 := by
  sorry

end NUMINAMATH_CALUDE_soy_sauce_bottle_ounces_l3352_335278


namespace NUMINAMATH_CALUDE_horner_method_f_at_3_f_at_3_equals_1_l3352_335207

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

/-- Coefficients of the polynomial in reverse order -/
def f_coeffs : List ℝ := [-1, 1, 0, -3, 4, 0.5]

theorem horner_method_f_at_3 :
  horner_eval f_coeffs 3 = f 3 := by
  sorry

theorem f_at_3_equals_1 :
  f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_at_3_f_at_3_equals_1_l3352_335207


namespace NUMINAMATH_CALUDE_negative_one_cubed_plus_squared_plus_one_l3352_335263

theorem negative_one_cubed_plus_squared_plus_one (x : ℤ) : 
  ((-1 : ℤ)^3) + ((-1 : ℤ)^2) + (-1 : ℤ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_cubed_plus_squared_plus_one_l3352_335263


namespace NUMINAMATH_CALUDE_quarter_circle_arcs_sum_limit_l3352_335299

/-- The sum of the lengths of quarter-circle arcs approaches πR/2 as n approaches infinity -/
theorem quarter_circle_arcs_sum_limit (R : ℝ) (h : R > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * R / (2 * n)) - π * R / 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_arcs_sum_limit_l3352_335299


namespace NUMINAMATH_CALUDE_f_min_implies_a_range_l3352_335290

/-- A function f with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 2

/-- The theorem stating that if f has a minimum value, then a is in [-3, 3] -/
theorem f_min_implies_a_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → a ∈ Set.Icc (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_f_min_implies_a_range_l3352_335290


namespace NUMINAMATH_CALUDE_function_transformation_l3352_335295

/-- Given a function f such that f(x-1) = x^2 + 4x - 5 for all x,
    prove that f(x) = x^2 + 6x for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = x^2 + 4*x - 5) : 
    ∀ x, f x = x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3352_335295


namespace NUMINAMATH_CALUDE_correct_calculation_l3352_335226

theorem correct_calculation (x y : ℝ) : 2 * x * y^2 - x * y^2 = x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3352_335226


namespace NUMINAMATH_CALUDE_alices_june_burger_spending_l3352_335294

/-- Calculate Alice's spending on burgers in June --/
def alices_burger_spending (
  days_in_june : Nat)
  (burgers_per_day : Nat)
  (burger_price : ℚ)
  (discount_days : Nat)
  (discount_percentage : ℚ)
  (free_burger_days : Nat)
  (coupon_count : Nat)
  (coupon_discount : ℚ) : ℚ :=
  let total_burgers := days_in_june * burgers_per_day
  let regular_cost := total_burgers * burger_price
  let discount_burgers := discount_days * burgers_per_day
  let discount_amount := discount_burgers * burger_price * discount_percentage
  let free_burgers := free_burger_days
  let free_burger_value := free_burgers * burger_price
  let coupon_savings := coupon_count * burger_price * coupon_discount
  regular_cost - discount_amount - free_burger_value - coupon_savings

/-- Theorem stating Alice's spending on burgers in June --/
theorem alices_june_burger_spending :
  alices_burger_spending 30 4 13 8 (1/10) 4 6 (1/2) = 1146.6 := by
  sorry

end NUMINAMATH_CALUDE_alices_june_burger_spending_l3352_335294


namespace NUMINAMATH_CALUDE_workers_completion_time_l3352_335259

theorem workers_completion_time (A B : ℝ) : 
  (A > 0) →  -- A's completion time is positive
  (B > 0) →  -- B's completion time is positive
  ((2/3) * B + B * (1 - (2*B)/(3*A)) = A*B/(A+B) + 2) →  -- Total time equation
  ((A*B)/(A+B) * (1/A) = (1/2) * (1 - (2*B)/(3*A))) →  -- A's work proportion equation
  (A = 6 ∧ B = 3) := by
  sorry

end NUMINAMATH_CALUDE_workers_completion_time_l3352_335259


namespace NUMINAMATH_CALUDE_remainder_x13_plus_1_div_x_minus_1_l3352_335234

theorem remainder_x13_plus_1_div_x_minus_1 (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_x13_plus_1_div_x_minus_1_l3352_335234


namespace NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l3352_335238

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (m n : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def parallel (a b : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the subset relation
def subset (a b : Set (ℝ × ℝ × ℝ)) : Prop := ∀ x, x ∈ a → x ∈ b

-- Define the angle between a line and a plane
def angle (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem B
theorem proposition_b (h1 : perpendicular m α) (h2 : parallel n α) :
  perpendicular m n := sorry

-- Theorem C
theorem proposition_c (h1 : parallel α β) (h2 : subset m α) :
  parallel m β := sorry

-- Theorem D
theorem proposition_d (h1 : parallel m n) (h2 : parallel α β) :
  angle m α = angle n β := sorry

end NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l3352_335238


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3352_335269

-- Define the given constants
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length : ℕ := 8
def roark_wire_length : ℕ := 2

-- Define Bonnie's prism
def bonnie_prism_volume : ℕ := (bonnie_wire_length / 2) ^ 3

-- Define Roark's unit prism
def roark_unit_prism_volume : ℕ := roark_wire_length ^ 3

-- Define the number of Roark's prisms
def roark_prism_count : ℕ := bonnie_prism_volume / roark_unit_prism_volume

-- Define the total wire lengths
def bonnie_total_wire : ℕ := bonnie_wire_pieces * bonnie_wire_length
def roark_total_wire : ℕ := roark_prism_count * (12 * roark_wire_length)

-- Theorem to prove
theorem wire_length_ratio :
  bonnie_total_wire / roark_total_wire = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3352_335269


namespace NUMINAMATH_CALUDE_flagpole_height_l3352_335223

/-- Represents the height and shadow length of an object -/
structure Object where
  height : ℝ
  shadowLength : ℝ

/-- Given two objects under similar conditions, their height-to-shadow ratios are equal -/
def similarConditions (obj1 obj2 : Object) : Prop :=
  obj1.height / obj1.shadowLength = obj2.height / obj2.shadowLength

theorem flagpole_height
  (flagpole : Object)
  (building : Object)
  (h_flagpole_shadow : flagpole.shadowLength = 45)
  (h_building_height : building.height = 24)
  (h_building_shadow : building.shadowLength = 60)
  (h_similar : similarConditions flagpole building) :
  flagpole.height = 18 := by
  sorry


end NUMINAMATH_CALUDE_flagpole_height_l3352_335223


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3352_335225

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3352_335225


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3352_335288

/-- A polynomial of degree 4 with coefficient b -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x - 1

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3352_335288


namespace NUMINAMATH_CALUDE_systematic_sampling_first_sample_first_sample_is_18_l3352_335274

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  interval : ℕ
  firstSample : ℕ
  eighteenthSample : ℕ

/-- Theorem stating the relationship between the first and eighteenth samples in systematic sampling -/
theorem systematic_sampling_first_sample
  (s : SystematicSampling)
  (h1 : s.population = 1000)
  (h2 : s.sampleSize = 40)
  (h3 : s.interval = s.population / s.sampleSize)
  (h4 : s.eighteenthSample = 443)
  (h5 : s.eighteenthSample = s.firstSample + 17 * s.interval) :
  s.firstSample = 18 := by
  sorry

/-- Main theorem proving the first sample number in the given scenario -/
theorem first_sample_is_18
  (population : ℕ)
  (sampleSize : ℕ)
  (eighteenthSample : ℕ)
  (h1 : population = 1000)
  (h2 : sampleSize = 40)
  (h3 : eighteenthSample = 443) :
  ∃ (s : SystematicSampling),
    s.population = population ∧
    s.sampleSize = sampleSize ∧
    s.interval = population / sampleSize ∧
    s.eighteenthSample = eighteenthSample ∧
    s.firstSample = 18 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_sample_first_sample_is_18_l3352_335274


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3352_335252

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3352_335252


namespace NUMINAMATH_CALUDE_kangaroo_hops_l3352_335231

/-- The distance covered in a single hop, given the remaining distance -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The sum of distances covered in n hops -/
def total_distance (n : ℕ) : ℚ :=
  (1 - (3/4)^n) / (1/4)

/-- The theorem stating that after 6 hops, the total distance covered is 3367/4096 -/
theorem kangaroo_hops : total_distance 6 = 3367 / 4096 := by sorry

end NUMINAMATH_CALUDE_kangaroo_hops_l3352_335231


namespace NUMINAMATH_CALUDE_street_length_l3352_335245

theorem street_length (forest_area : ℝ) (street_area : ℝ) (trees_per_sqm : ℝ) (total_trees : ℝ) :
  forest_area = 3 * street_area →
  trees_per_sqm = 4 →
  total_trees = 120000 →
  street_area = (100 : ℝ) ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_street_length_l3352_335245


namespace NUMINAMATH_CALUDE_product_bounds_l3352_335247

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l3352_335247


namespace NUMINAMATH_CALUDE_product_of_twelve_and_3460_l3352_335284

theorem product_of_twelve_and_3460 : ∃ x : ℕ, 12 * x = 173 * x ∧ x = 3460 → 12 * 3460 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_of_twelve_and_3460_l3352_335284


namespace NUMINAMATH_CALUDE_painting_theorem_l3352_335253

/-- The time required for two people to paint a room together, including a break -/
def paint_time (karl_time leo_time break_time : ℝ) : ℝ → Prop :=
  λ t : ℝ => (1 / karl_time + 1 / leo_time) * (t - break_time) = 1

theorem painting_theorem :
  ∃ t : ℝ, paint_time 6 8 0.5 t :=
by
  sorry

end NUMINAMATH_CALUDE_painting_theorem_l3352_335253


namespace NUMINAMATH_CALUDE_fred_book_purchase_l3352_335270

theorem fred_book_purchase (initial_amount remaining_amount cost_per_book : ℕ) 
  (h1 : initial_amount = 236)
  (h2 : remaining_amount = 14)
  (h3 : cost_per_book = 37) :
  (initial_amount - remaining_amount) / cost_per_book = 6 := by
  sorry

end NUMINAMATH_CALUDE_fred_book_purchase_l3352_335270


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3352_335282

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ n : ℕ, 2023^2024 - 2025 = 100*n + 4 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3352_335282


namespace NUMINAMATH_CALUDE_line_symmetry_l3352_335222

/-- Given two lines in the form y = mx + b, this function checks if they are symmetrical about the x-axis -/
def symmetrical_about_x_axis (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = -m2 ∧ b1 = -b2

/-- The original line y = 3x - 4 -/
def original_line (x : ℝ) : ℝ := 3 * x - 4

/-- The proposed symmetrical line y = -3x + 4 -/
def symmetrical_line (x : ℝ) : ℝ := -3 * x + 4

theorem line_symmetry :
  symmetrical_about_x_axis 3 (-4) (-3) 4 :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l3352_335222


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3352_335267

/-- A quadratic equation in x is an equation of the form ax^2 + bx + c = 0,
    where a, b, and c are constants and a ≠ 0. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem: f(x) = x^2 + 3x - 5 is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3352_335267


namespace NUMINAMATH_CALUDE_video_game_price_l3352_335240

def lawn_price : ℕ := 15
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5
def books_bought : ℕ := 60

theorem video_game_price :
  (lawn_price * lawns_mowed - book_price * books_bought) / video_games_wanted = 45 := by
  sorry

end NUMINAMATH_CALUDE_video_game_price_l3352_335240


namespace NUMINAMATH_CALUDE_average_of_ABC_l3352_335261

theorem average_of_ABC (A B C : ℝ) 
  (eq1 : 501 * C - 1002 * A = 2002)
  (eq2 : 501 * B + 2002 * A = 2505) :
  (A + B + C) / 3 = -A / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABC_l3352_335261


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3352_335280

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) :
  (P * (R + 5) * 9 / 100 = P * R * 9 / 100 + 1350) → P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3352_335280


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3352_335250

theorem polynomial_expansion (x : ℝ) : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = 
  -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3352_335250


namespace NUMINAMATH_CALUDE_some_number_value_l3352_335210

theorem some_number_value (x : ℝ) :
  64 + 5 * 12 / (180 / x) = 65 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3352_335210


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3352_335287

theorem inserted_numbers_sum (a b c : ℝ) : 
  (∃ r : ℝ, a = 3 * r ∧ b = 3 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ c = b + d ∧ 27 = c + d) →  -- Arithmetic progression condition
  a + b + c = 161 / 3 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3352_335287


namespace NUMINAMATH_CALUDE_x_value_l3352_335297

theorem x_value : ∃ x : ℝ, 
  ((x * (9^2)) / ((8^2) * (3^5)) = 0.16666666666666666) ∧ 
  (x = 5.333333333333333) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3352_335297


namespace NUMINAMATH_CALUDE_rosalina_total_gifts_l3352_335262

/-- The number of gifts Rosalina received from Emilio -/
def gifts_from_emilio : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def gifts_from_jorge : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def gifts_from_pedro : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := gifts_from_emilio + gifts_from_jorge + gifts_from_pedro

theorem rosalina_total_gifts : total_gifts = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_total_gifts_l3352_335262


namespace NUMINAMATH_CALUDE_probability_of_mathematics_letter_l3352_335236

theorem probability_of_mathematics_letter :
  let total_letters : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / total_letters
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_of_mathematics_letter_l3352_335236


namespace NUMINAMATH_CALUDE_max_value_tan_cos_l3352_335221

open Real

theorem max_value_tan_cos (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (max : Real), max = 2 * (Real.sqrt ((-9 + Real.sqrt 117) / 2))^3 / 
    Real.sqrt (1 - (Real.sqrt ((-9 + Real.sqrt 117) / 2))^2) ∧
  ∀ (x : Real), 0 < x ∧ x < π/2 → 
    tan (x/2) * (1 - cos x) ≤ max := by sorry

end NUMINAMATH_CALUDE_max_value_tan_cos_l3352_335221


namespace NUMINAMATH_CALUDE_function_equality_l3352_335211

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem function_equality (f : ℝ → ℝ) :
  (∀ x, f x^3 + f x ≤ x ∧ x ≤ f (x^3 + x)) →
  (∀ x, f x = Function.invFun g x) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_l3352_335211


namespace NUMINAMATH_CALUDE_path_area_calculation_l3352_335228

/-- Calculates the area of a path around a rectangular field -/
def pathArea (fieldLength fieldWidth pathWidth : ℝ) : ℝ :=
  (fieldLength + 2 * pathWidth) * (fieldWidth + 2 * pathWidth) - fieldLength * fieldWidth

/-- Theorem: The area of a 2.5m wide path around a 75m by 55m field is 675 sq m -/
theorem path_area_calculation :
  pathArea 75 55 2.5 = 675 := by sorry

end NUMINAMATH_CALUDE_path_area_calculation_l3352_335228


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l3352_335296

/-- Given an ellipse and a line, this theorem states the range of the y-intercept of the line for which
    there exist two distinct points on the ellipse symmetric about the line. -/
theorem ellipse_symmetric_points_range (x y : ℝ) (m : ℝ) : 
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- Point 1 on ellipse
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- Point 2 on ellipse
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧        -- Points are distinct
    (y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m)  -- Points symmetric about y = 4x + m
  ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l3352_335296


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_N_l3352_335209

/-- The number of vertices in the graph -/
def num_vertices : ℕ := 8

/-- The number of ordered pairs to examine for each permutation -/
def pairs_per_permutation : ℕ := num_vertices * (num_vertices - 1)

/-- The total number of examinations in Sophia's algorithm -/
def N : ℕ := (Nat.factorial num_vertices) * pairs_per_permutation

/-- The theorem stating that the largest power of two dividing N is 10 -/
theorem largest_power_of_two_dividing_N :
  ∃ k : ℕ, (2^10 : ℕ) ∣ N ∧ ¬(2^(k+1) : ℕ) ∣ N ∧ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_N_l3352_335209


namespace NUMINAMATH_CALUDE_intersection_point_l3352_335254

def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 4
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 18

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (49/23, 60/23) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3352_335254


namespace NUMINAMATH_CALUDE_marla_bags_per_trip_l3352_335265

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The number of shopping trips -/
def num_trips : ℕ := 300

/-- The amount of CO2 released by the canvas bag (in pounds) -/
def canvas_bag_co2 : ℕ := 600

/-- The amount of CO2 released by each plastic bag (in ounces) -/
def plastic_bag_co2 : ℕ := 4

/-- The number of plastic bags Marla uses per shopping trip -/
def bags_per_trip : ℕ := 8

theorem marla_bags_per_trip :
  (bags_per_trip * plastic_bag_co2 * num_trips) / ounces_per_pound = canvas_bag_co2 :=
sorry

end NUMINAMATH_CALUDE_marla_bags_per_trip_l3352_335265


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3352_335268

/-- Given a geometric sequence where the first term is 12 and the common ratio is 1/3,
    the 15th term is equal to 4/531441. -/
theorem fifteenth_term_of_geometric_sequence (a : ℕ → ℚ) :
  a 1 = 12 →
  (∀ n : ℕ, a (n + 1) = a n * (1/3)) →
  a 15 = 4/531441 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3352_335268


namespace NUMINAMATH_CALUDE_oblique_asymptote_of_f_l3352_335275

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 5) / (x + 4)

theorem oblique_asymptote_of_f :
  ∃ (a b : ℝ), a ≠ 0 ∧ (∀ ε > 0, ∃ M, ∀ x > M, |f x - (a * x + b)| < ε) ∧ a = 3 ∧ b = -4 :=
sorry

end NUMINAMATH_CALUDE_oblique_asymptote_of_f_l3352_335275


namespace NUMINAMATH_CALUDE_pens_count_in_second_set_l3352_335224

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and some pens in dollars -/
def first_set_cost : ℚ := 158/100

/-- The cost of 4 pencils and 5 pens in dollars -/
def second_set_cost : ℚ := 2

/-- The number of pens in the second set -/
def pens_in_second_set : ℕ := 5

/-- Theorem stating that given the conditions, the number of pens in the second set is 5 -/
theorem pens_count_in_second_set : 
  ∃ (pen_cost : ℚ) (pens_in_first_set : ℕ), 
    3 * pencil_cost + pens_in_first_set * pen_cost = first_set_cost ∧
    4 * pencil_cost + pens_in_second_set * pen_cost = second_set_cost :=
by
  sorry

#check pens_count_in_second_set

end NUMINAMATH_CALUDE_pens_count_in_second_set_l3352_335224


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3352_335205

/-- The equation of the tangent line to y = x^2 + x + 1/2 at (0, 1/2) is y = x + 1/2 -/
theorem tangent_line_at_origin (x : ℝ) :
  let f (x : ℝ) := x^2 + x + 1/2
  let f' (x : ℝ) := 2*x + 1
  let tangent_line (x : ℝ) := x + 1/2
  (∀ x, deriv f x = f' x) →
  (f 0 = 1/2) →
  (tangent_line 0 = 1/2) →
  (f' 0 = 1) →
  ∀ x, tangent_line x = f 0 + f' 0 * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3352_335205


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3352_335202

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ↔ 
  (k = 1 → ∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ∧ 
  (∃ k' : ℝ, k' ≠ 1 ∧ ∃ x y : ℝ, x - y + k' = 0 ∧ x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3352_335202


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l3352_335217

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 for m > 0 and n > 0 -/
def represents_parabola_ellipse_intersection (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧
  ∃ (x y : ℝ), 
    (m * x + n * y^2 = 0) ∧
    (m * x^2 + n * y^2 = 1) ∧
    (-1 < x ∧ x < 0)

/-- Theorem stating that the equations represent a parabola opening to the left intersecting an ellipse -/
theorem parabola_ellipse_intersection :
  ∀ (m n : ℝ), represents_parabola_ellipse_intersection m n →
  ∃ (x y : ℝ), 
    (y^2 = -m/n * x) ∧  -- Parabola equation
    (x^2/(1/m) + y^2/(1/n) = 1) ∧  -- Ellipse equation
    (-1 < x ∧ x < 0) :=
by sorry


end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l3352_335217
