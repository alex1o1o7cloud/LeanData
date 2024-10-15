import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l593_59351

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {-3, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l593_59351


namespace NUMINAMATH_CALUDE_total_worth_is_correct_l593_59345

-- Define the given conditions
def initial_packs : ℕ := 4
def new_packs : ℕ := 2
def price_per_pack : ℚ := 2.5
def discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.07

-- Define the function to calculate the total worth
def total_worth : ℚ :=
  let initial_cost := initial_packs * price_per_pack
  let new_cost := new_packs * price_per_pack
  let discount := new_cost * discount_rate
  let discounted_cost := new_cost - discount
  let tax := new_cost * tax_rate
  let total_new_cost := discounted_cost + tax
  initial_cost + total_new_cost

-- State the theorem
theorem total_worth_is_correct : total_worth = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_is_correct_l593_59345


namespace NUMINAMATH_CALUDE_solve_systems_of_equations_l593_59325

theorem solve_systems_of_equations :
  -- System 1
  (∃ (x y : ℝ), x - y = 3 ∧ 3*x - 8*y = 14 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ (x y : ℝ), 3*x + y = 1 ∧ 5*x - 2*y = 9 ∧ x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_solve_systems_of_equations_l593_59325


namespace NUMINAMATH_CALUDE_weight_of_person_a_l593_59360

/-- Given the average weights of different groups and the relationship between individuals' weights,
    prove that the weight of person A is 80 kg. -/
theorem weight_of_person_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 8 →
  (b + c + d + e) / 4 = 79 →
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_weight_of_person_a_l593_59360


namespace NUMINAMATH_CALUDE_number_between_24_and_28_l593_59327

def possibleNumbers : List ℕ := [20, 23, 26, 29]

theorem number_between_24_and_28 (n : ℕ) 
  (h1 : n > 24) 
  (h2 : n < 28) 
  (h3 : n ∈ possibleNumbers) : 
  n = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_between_24_and_28_l593_59327


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l593_59357

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_val : ℝ), max_val = 24 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((w - 2)^3 * (w + 2)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l593_59357


namespace NUMINAMATH_CALUDE_find_t_l593_59365

theorem find_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 95) (eq2 : s = 2 * t - 3) : t = 119 / 23 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l593_59365


namespace NUMINAMATH_CALUDE_max_value_theorem_l593_59330

theorem max_value_theorem (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ (max_x : ℝ), max_x = 1/2 ∧
  ∀ y, 0 < y ∧ y < 1 → x * (3 - 3 * x) ≤ max_x * (3 - 3 * max_x) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l593_59330


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l593_59300

theorem fraction_sum_simplification :
  2 / 520 + 23 / 40 = 301 / 520 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l593_59300


namespace NUMINAMATH_CALUDE_quadratic_solution_form_l593_59394

theorem quadratic_solution_form (x : ℝ) : 
  (5 * x^2 - 11 * x + 2 = 0) →
  ∃ (m n p : ℕ), 
    x = (m + Real.sqrt n) / p ∧ 
    m = 20 ∧ 
    n = 0 ∧ 
    p = 10 ∧
    m + n + p = 30 ∧
    Nat.gcd m (Nat.gcd n p) = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_form_l593_59394


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l593_59383

theorem stewart_farm_ratio : ∀ (num_sheep num_horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
  num_sheep = 24 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  num_horses * horse_food_per_day = total_horse_food →
  num_sheep * 7 = num_horses * 3 :=
by sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l593_59383


namespace NUMINAMATH_CALUDE_system_solution_l593_59350

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = a ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = a ∧ z = 0) ∨
   (x = 0 ∧ y = 0 ∧ z = a)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l593_59350


namespace NUMINAMATH_CALUDE_homework_completion_l593_59377

theorem homework_completion (total : ℝ) (h : total > 0) : 
  let monday := (3 / 5 : ℝ) * total
  let tuesday := (1 / 3 : ℝ) * (total - monday)
  let wednesday := total - monday - tuesday
  wednesday = (4 / 15 : ℝ) * total := by
  sorry

end NUMINAMATH_CALUDE_homework_completion_l593_59377


namespace NUMINAMATH_CALUDE_mike_remaining_nickels_l593_59361

/-- Given Mike's initial number of nickels and the number of nickels his dad borrowed,
    calculate the number of nickels Mike has left. -/
def nickels_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Mike has 12 nickels remaining after his dad's borrowing. -/
theorem mike_remaining_nickels :
  nickels_remaining 87 75 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_remaining_nickels_l593_59361


namespace NUMINAMATH_CALUDE_degree_of_g_l593_59347

/-- Given a polynomial f(x) = -7x^4 + 3x^3 + x - 5 and another polynomial g(x) such that 
    the degree of f(x) + g(x) is 2, prove that the degree of g(x) is 4. -/
theorem degree_of_g (f g : Polynomial ℝ) : 
  f = -7 * X^4 + 3 * X^3 + X - 5 →
  (f + g).degree = 2 →
  g.degree = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_g_l593_59347


namespace NUMINAMATH_CALUDE_abs_inequality_l593_59338

theorem abs_inequality (x : ℝ) : |5 - x| > 6 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 11 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_l593_59338


namespace NUMINAMATH_CALUDE_ratio_equality_l593_59356

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) : (a + b) / c = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l593_59356


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l593_59334

theorem min_value_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ m : ℝ, m = 4/9 ∧ ∀ a b c : ℝ, a + b + c = 1 → a^2 + b^2 + 4*c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l593_59334


namespace NUMINAMATH_CALUDE_josh_found_seven_marbles_l593_59336

/-- The number of marbles Josh had initially -/
def initial_marbles : ℕ := 21

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 28

/-- The number of marbles Josh found -/
def found_marbles : ℕ := current_marbles - initial_marbles

theorem josh_found_seven_marbles :
  found_marbles = 7 :=
by sorry

end NUMINAMATH_CALUDE_josh_found_seven_marbles_l593_59336


namespace NUMINAMATH_CALUDE_rectangle_dissection_theorem_l593_59323

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a triangle -/
structure Triangle

/-- Represents a pentagon -/
structure Pentagon

/-- Represents a set of shapes that can be rearranged -/
structure ShapeSet where
  triangles : Finset Triangle
  pentagon : Pentagon

theorem rectangle_dissection_theorem (initial : Rectangle) (final : Rectangle) 
  (h_initial : initial.width = 4 ∧ initial.height = 6)
  (h_final : final.width = 3 ∧ final.height = 8)
  (h_area_preservation : initial.width * initial.height = final.width * final.height) :
  ∃ (pieces : ShapeSet), 
    pieces.triangles.card = 2 ∧ 
    (∃ (arrangement : ShapeSet → Rectangle), arrangement pieces = final) :=
sorry

end NUMINAMATH_CALUDE_rectangle_dissection_theorem_l593_59323


namespace NUMINAMATH_CALUDE_xyz_inequality_l593_59335

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  Real.sqrt (1 + 8 * x) + Real.sqrt (1 + 8 * y) + Real.sqrt (1 + 8 * z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l593_59335


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l593_59311

theorem unique_solution_for_equation :
  ∀ x y : ℕ+,
    x > y →
    (x - y : ℕ+) ^ (x * y : ℕ) = x ^ (y : ℕ) * y ^ (x : ℕ) →
    x = 4 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l593_59311


namespace NUMINAMATH_CALUDE_carnival_game_cost_per_play_l593_59326

/-- Represents the carnival game scenario -/
structure CarnivalGame where
  budget : ℚ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  total_points_possible : ℕ

/-- Calculates the cost per play for the carnival game -/
def cost_per_play (game : CarnivalGame) : ℚ :=
  game.budget / game.games_played

/-- Theorem stating that the cost per play is $1.50 for the given scenario -/
theorem carnival_game_cost_per_play :
  let game : CarnivalGame := {
    budget := 3,
    red_points := 2,
    green_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    green_buckets_hit := 5,
    total_points_possible := 38
  }
  cost_per_play game = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_cost_per_play_l593_59326


namespace NUMINAMATH_CALUDE_triangle_side_mod_three_l593_59395

/-- 
Given two triangles with the same perimeter, where the first is equilateral
with integer side lengths and the second has integer side lengths with one
side of length 1 and another of length d, then d ≡ 1 (mod 3).
-/
theorem triangle_side_mod_three (a d : ℕ) : 
  (3 * a = 2 * d + 1) → (d % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_mod_three_l593_59395


namespace NUMINAMATH_CALUDE_square_root_domain_only_five_satisfies_l593_59315

theorem square_root_domain (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) ↔ x ≥ 4 :=
sorry

theorem only_five_satisfies : 
  (∃ y : ℝ, y^2 = 5 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 0 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 1 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 2 - 4) :=
sorry

end NUMINAMATH_CALUDE_square_root_domain_only_five_satisfies_l593_59315


namespace NUMINAMATH_CALUDE_profit_percentage_at_marked_price_l593_59316

theorem profit_percentage_at_marked_price 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0)
  (h3 : 0.8 * marked_price = 1.2 * cost_price) : 
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_at_marked_price_l593_59316


namespace NUMINAMATH_CALUDE_bottles_needed_to_fill_container_l593_59372

def craft_bottle_volume : ℕ := 150
def decorative_container_volume : ℕ := 2650

theorem bottles_needed_to_fill_container : 
  ∃ n : ℕ, n * craft_bottle_volume ≥ decorative_container_volume ∧ 
  ∀ m : ℕ, m * craft_bottle_volume ≥ decorative_container_volume → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_bottles_needed_to_fill_container_l593_59372


namespace NUMINAMATH_CALUDE_min_time_to_return_l593_59332

/-- Given a circular track and a person's walking pattern, calculate the minimum time to return to the starting point. -/
theorem min_time_to_return (track_length : ℝ) (speed : ℝ) (t1 t2 t3 : ℝ) : 
  track_length = 400 →
  speed = 6000 / 60 →
  t1 = 1 →
  t2 = 3 →
  t3 = 5 →
  (min_time : ℝ) * speed = track_length - ((t1 - t2 + t3) * speed) →
  min_time = 1 := by
  sorry

#check min_time_to_return

end NUMINAMATH_CALUDE_min_time_to_return_l593_59332


namespace NUMINAMATH_CALUDE_feathers_per_pound_is_300_l593_59348

/-- Represents the number of feathers in a goose -/
def goose_feathers : ℕ := 3600

/-- Represents the number of pillows that can be stuffed with one goose's feathers -/
def pillows_per_goose : ℕ := 6

/-- Represents the number of pounds of feathers needed for each pillow -/
def pounds_per_pillow : ℕ := 2

/-- Calculates the number of feathers in a pound of goose feathers -/
def feathers_per_pound : ℕ := goose_feathers / (pillows_per_goose * pounds_per_pillow)

theorem feathers_per_pound_is_300 : feathers_per_pound = 300 := by
  sorry

end NUMINAMATH_CALUDE_feathers_per_pound_is_300_l593_59348


namespace NUMINAMATH_CALUDE_rectangle_to_square_side_half_length_l593_59389

/-- Given a rectangle with dimensions 7 × 21 that is cut into two congruent shapes
    and rearranged into a square, half the length of a side of the resulting square
    is equal to 7√3/2. -/
theorem rectangle_to_square_side_half_length :
  let rectangle_length : ℝ := 21
  let rectangle_width : ℝ := 7
  let rectangle_area := rectangle_length * rectangle_width
  let square_side := Real.sqrt rectangle_area
  let y := square_side / 2
  y = 7 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_side_half_length_l593_59389


namespace NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l593_59384

/-- Given a sequence {a_n} where for any n ∈ ℕ*, the point P_n(n, a_n) lies on the line y = 2x + 1,
    prove that {a_n} is an arithmetic sequence with a common difference of 2. -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = 2 * n + 1) →
  ∃ (a₀ : ℝ), ∀ n : ℕ, a n = a₀ + 2 * n :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l593_59384


namespace NUMINAMATH_CALUDE_quadratic_root_inequality_l593_59303

theorem quadratic_root_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) 
  (h3 : a * 1^2 + b * 1 + c = 0) : -2 ≤ c / a ∧ c / a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_inequality_l593_59303


namespace NUMINAMATH_CALUDE_sphere_radius_is_6_l593_59314

/-- The radius of a sphere whose surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 12 cm. -/
def sphere_radius : ℝ := 6

/-- The height of the cylinder. -/
def cylinder_height : ℝ := 12

/-- The diameter of the cylinder. -/
def cylinder_diameter : ℝ := 12

/-- The theorem stating that the radius of the sphere is 6 cm. -/
theorem sphere_radius_is_6 :
  sphere_radius = 6 ∧
  4 * Real.pi * sphere_radius ^ 2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_is_6_l593_59314


namespace NUMINAMATH_CALUDE_garden_vegetables_l593_59321

theorem garden_vegetables (potatoes cucumbers peppers : ℕ) : 
  cucumbers = potatoes - 60 →
  peppers = 2 * cucumbers →
  potatoes + cucumbers + peppers = 768 →
  potatoes = 237 := by
sorry

end NUMINAMATH_CALUDE_garden_vegetables_l593_59321


namespace NUMINAMATH_CALUDE_complex_expression_equals_four_l593_59304

theorem complex_expression_equals_four :
  (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * π / 180) + (π - 2023)^0 + |-2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_four_l593_59304


namespace NUMINAMATH_CALUDE_sqrt_inequality_l593_59305

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l593_59305


namespace NUMINAMATH_CALUDE_souvenir_problem_l593_59396

/-- Represents the cost and selling prices of souvenirs -/
structure SouvenirPrices where
  costA : ℝ
  costB : ℝ
  sellingB : ℝ

/-- Represents the quantity and profit of souvenirs -/
structure SouvenirQuantities where
  totalQuantity : ℕ
  minQuantityA : ℕ

/-- Theorem stating the properties of the souvenir problem -/
theorem souvenir_problem 
  (prices : SouvenirPrices) 
  (quantities : SouvenirQuantities) 
  (h1 : prices.costA = prices.costB + 30)
  (h2 : 1000 / prices.costA = 400 / prices.costB)
  (h3 : quantities.totalQuantity = 200)
  (h4 : quantities.minQuantityA ≥ 60)
  (h5 : quantities.minQuantityA < quantities.totalQuantity - quantities.minQuantityA)
  (h6 : prices.sellingB = 30) :
  prices.costA = 50 ∧ 
  prices.costB = 20 ∧
  (∃ x : ℝ, x = 65 ∧ (x - prices.costA) * (400 - 5*x) = 1125) ∧
  (∃ y : ℝ, y = 2480 ∧ 
    y = (68 - prices.costA) * (400 - 5*68) + 
        (prices.sellingB - prices.costB) * (quantities.totalQuantity - (400 - 5*68))) :=
by sorry

end NUMINAMATH_CALUDE_souvenir_problem_l593_59396


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l593_59380

-- Define the sample space
def Ω : Type := Fin 3 → Fin 3

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A: sum of numbers drawn is 6
def A : Set Ω := {ω : Ω | ω 0 + ω 1 + ω 2 = 5}

-- Define event B: number 2 is drawn three times
def B : Set Ω := {ω : Ω | ∀ i, ω i = 1}

-- State the theorem
theorem conditional_probability_B_given_A :
  P B / P A = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l593_59380


namespace NUMINAMATH_CALUDE_special_polynomial_form_l593_59373

/-- A polynomial satisfying the given conditions -/
class SpecialPolynomial (P : ℝ → ℝ) where
  zero_condition : P 0 = 0
  functional_equation : ∀ x : ℝ, P x = (1/2) * (P (x + 1) + P (x - 1))

/-- Theorem stating that any polynomial satisfying the given conditions is of the form P(x) = ax -/
theorem special_polynomial_form {P : ℝ → ℝ} [SpecialPolynomial P] : 
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l593_59373


namespace NUMINAMATH_CALUDE_car_selling_price_l593_59363

/-- Calculates the selling price of a car given its purchase price, repair cost, and profit percentage. -/
theorem car_selling_price (purchase_price repair_cost : ℕ) (profit_percent : ℚ) :
  purchase_price = 42000 →
  repair_cost = 13000 →
  profit_percent = 17272727272727273 / 100000000000000000 →
  (purchase_price + repair_cost) * (1 + profit_percent) = 64500 := by
  sorry

end NUMINAMATH_CALUDE_car_selling_price_l593_59363


namespace NUMINAMATH_CALUDE_stadium_empty_seats_l593_59366

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: In a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats :
  empty_seats 92 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_stadium_empty_seats_l593_59366


namespace NUMINAMATH_CALUDE_exists_vector_not_in_span_l593_59397

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-4, -2)

/-- The statement to be proven -/
theorem exists_vector_not_in_span : ∃ d : ℝ × ℝ, ∀ k₁ k₂ : ℝ, d ≠ k₁ • b + k₂ • c := by
  sorry

end NUMINAMATH_CALUDE_exists_vector_not_in_span_l593_59397


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l593_59324

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x ≥ f' x₀) ∧ 
    y₀ = f x₀ ∧
    (∀ x : ℝ, 3*x - y₀ = 1 → f x = 3*x - 1) :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l593_59324


namespace NUMINAMATH_CALUDE_frenchHorn_trombone_difference_l593_59308

/-- The number of band members for each instrument and their relationships --/
structure BandComposition where
  flute : ℕ
  trumpet : ℕ
  trombone : ℕ
  drums : ℕ
  clarinet : ℕ
  frenchHorn : ℕ
  fluteCount : flute = 5
  trumpetCount : trumpet = 3 * flute
  tromboneCount : trombone = trumpet - 8
  drumsCount : drums = trombone + 11
  clarinetCount : clarinet = 2 * flute
  frenchHornMoreThanTrombone : frenchHorn > trombone
  totalSeats : flute + trumpet + trombone + drums + clarinet + frenchHorn = 65

/-- The theorem stating the difference between French horn and trombone players --/
theorem frenchHorn_trombone_difference (b : BandComposition) :
  b.frenchHorn - b.trombone = 3 := by
  sorry

end NUMINAMATH_CALUDE_frenchHorn_trombone_difference_l593_59308


namespace NUMINAMATH_CALUDE_peanut_box_count_l593_59364

/-- Given an initial quantity of peanuts in a box and an additional quantity added,
    compute the final quantity of peanuts in the box. -/
def final_peanut_count (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 4 initial peanuts and 6 added peanuts, 
    the final count is 10 peanuts. -/
theorem peanut_box_count : final_peanut_count 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanut_box_count_l593_59364


namespace NUMINAMATH_CALUDE_factor_expression_l593_59306

theorem factor_expression (x : ℝ) : 45 * x + 30 = 15 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l593_59306


namespace NUMINAMATH_CALUDE_unique_function_theorem_l593_59382

-- Define the function type
def IntFunction := ℤ → ℤ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : IntFunction) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

-- State the theorem
theorem unique_function_theorem :
  ∀ f : IntFunction, SatisfiesEquation f → ∀ n : ℤ, f n = 2 * n + 1007 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l593_59382


namespace NUMINAMATH_CALUDE_min_non_acute_angles_l593_59301

/-- A convex polygon with 1992 sides -/
structure ConvexPolygon1992 where
  sides : ℕ
  convex : Bool
  sides_eq : sides = 1992
  is_convex : convex = true

/-- The number of interior angles that are not acute in a polygon -/
def non_acute_angles (p : ConvexPolygon1992) : ℕ := sorry

/-- The theorem stating the minimum number of non-acute angles in a ConvexPolygon1992 -/
theorem min_non_acute_angles (p : ConvexPolygon1992) : 
  non_acute_angles p ≥ 1989 := by sorry

end NUMINAMATH_CALUDE_min_non_acute_angles_l593_59301


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l593_59333

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 2 ∧ 
    (f s = 0) ∧ 
    (∀ (x : ℝ), f x = 0 → x ≥ s) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l593_59333


namespace NUMINAMATH_CALUDE_optimal_production_consumption_theorem_l593_59371

/-- Represents a country's production capabilities and consumption --/
structure Country where
  eggplant_production : ℝ
  corn_production : ℝ
  consumption : ℝ × ℝ

/-- The global market for agricultural products --/
structure Market where
  price : ℝ

/-- Calculates the optimal production and consumption for two countries --/
def optimal_production_and_consumption (a b : Country) (m : Market) : (Country × Country) :=
  sorry

/-- Main theorem: Optimal production and consumption for countries A and B --/
theorem optimal_production_consumption_theorem (a b : Country) (m : Market) :
  a.eggplant_production = 10 ∧
  a.corn_production = 8 ∧
  b.eggplant_production = 18 ∧
  b.corn_production = 12 ∧
  m.price > 0 →
  let (a', b') := optimal_production_and_consumption a b m
  a'.consumption = (4, 4) ∧ b'.consumption = (9, 9) :=
sorry

end NUMINAMATH_CALUDE_optimal_production_consumption_theorem_l593_59371


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l593_59343

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 3
  f 2 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l593_59343


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l593_59391

def A : Set Int := {-1, 0, 1, 2}
def B : Set Int := {-2, 0, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l593_59391


namespace NUMINAMATH_CALUDE_seokgi_jumped_furthest_l593_59398

/-- Represents the jump distances of three people -/
structure JumpDistances where
  yooseung : ℚ
  shinyoung : ℚ
  seokgi : ℚ

/-- Given the jump distances, proves that Seokgi jumped the furthest -/
theorem seokgi_jumped_furthest (j : JumpDistances)
  (h1 : j.yooseung = 15/8)
  (h2 : j.shinyoung = 2)
  (h3 : j.seokgi = 17/8) :
  j.seokgi > j.yooseung ∧ j.seokgi > j.shinyoung :=
by sorry

end NUMINAMATH_CALUDE_seokgi_jumped_furthest_l593_59398


namespace NUMINAMATH_CALUDE_angle_C_measure_l593_59393

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_C_measure (abc : Triangle) (h1 : abc.A = 50) (h2 : abc.B = 60) : abc.C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l593_59393


namespace NUMINAMATH_CALUDE_sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l593_59378

-- Define a triangle with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (triangle_inequality₁ : a + b > c)
  (triangle_inequality₂ : b + c > a)
  (triangle_inequality₃ : c + a > b)

-- Theorem 1: A triangle with sides √a, √b, and √c always exists
theorem sqrt_triangle_exists (t : Triangle) : 
  ∃ (t' : Triangle), t'.a = Real.sqrt t.a ∧ t'.b = Real.sqrt t.b ∧ t'.c = Real.sqrt t.c :=
sorry

-- Theorem 2: A triangle with sides |a-b|+1, |b-c|+1, and |c-a|+1 always exists
theorem abs_diff_plus_one_triangle_exists (t : Triangle) :
  ∃ (t' : Triangle), t'.a = |t.a - t.b| + 1 ∧ t'.b = |t.b - t.c| + 1 ∧ t'.c = |t.c - t.a| + 1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l593_59378


namespace NUMINAMATH_CALUDE_intersection_points_count_l593_59374

/-- A line in the 2D plane --/
inductive Line
  | General (a b c : ℝ) : Line  -- ax + by + c = 0
  | Vertical (x : ℝ) : Line     -- x = k
  | Horizontal (y : ℝ) : Line   -- y = k

/-- Check if a point (x, y) is on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  match l with
  | General a b c => a * x + b * y + c = 0
  | Vertical k => x = k
  | Horizontal k => y = k

/-- The set of lines given in the problem --/
def problem_lines : List Line :=
  [Line.General 3 (-1) (-1), Line.General 1 2 (-5), Line.Vertical 3, Line.Horizontal 1]

/-- A point is an intersection point if it's contained in at least two distinct lines --/
def is_intersection_point (x y : ℝ) (lines : List Line) : Prop :=
  ∃ l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ l1.contains x y ∧ l2.contains x y

/-- The theorem to be proved --/
theorem intersection_points_count :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (∀ x y : ℝ, is_intersection_point x y problem_lines ↔ (x, y) = p1 ∨ (x, y) = p2) :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l593_59374


namespace NUMINAMATH_CALUDE_sum_of_first_eight_multiples_of_eleven_l593_59392

/-- The sum of the first n distinct positive integer multiples of m -/
def sum_of_multiples (n m : ℕ) : ℕ := 
  m * n * (n + 1) / 2

/-- Theorem: The sum of the first 8 distinct positive integer multiples of 11 is 396 -/
theorem sum_of_first_eight_multiples_of_eleven : 
  sum_of_multiples 8 11 = 396 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_eight_multiples_of_eleven_l593_59392


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l593_59388

theorem yellow_marbles_count (total red blue green yellow : ℕ) : 
  total = 110 →
  red = 8 →
  blue = 4 * red →
  green = 2 * blue →
  yellow = total - (red + blue + green) →
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l593_59388


namespace NUMINAMATH_CALUDE_quadratic_function_property_l593_59381

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : (fun x => a * x^2 + b * x + c) (-2) = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that for a quadratic function with given properties, (a+c)/b = 1/2 -/
theorem quadratic_function_property (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l593_59381


namespace NUMINAMATH_CALUDE_A_initial_investment_l593_59353

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_investment : ℝ := 16200

/-- Represents the number of months A's investment was active -/
def A_months : ℝ := 12

/-- Represents the number of months B's investment was active -/
def B_months : ℝ := 5

/-- Represents the ratio of A's profit share -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of B's profit share -/
def B_profit_ratio : ℝ := 3

theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_profit_ratio / B_profit_ratio →
  A_investment = 4500 := by
sorry

end NUMINAMATH_CALUDE_A_initial_investment_l593_59353


namespace NUMINAMATH_CALUDE_discount_difference_l593_59319

theorem discount_difference : 
  let original_bill : ℝ := 12000
  let single_discount_rate : ℝ := 0.35
  let first_successive_discount_rate : ℝ := 0.30
  let second_successive_discount_rate : ℝ := 0.06
  let single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discount_amount - single_discount_amount = 96 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l593_59319


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_nine_l593_59376

/-- The sum of digits in a number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The concatenation of numbers from 1 to n -/
def concatenateNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in the concatenation of numbers from 1 to 2015 is divisible by 9 -/
theorem sum_of_digits_divisible_by_nine :
  ∃ k : ℕ, sumOfDigits (concatenateNumbers 2015) = 9 * k := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_nine_l593_59376


namespace NUMINAMATH_CALUDE_drivers_distance_difference_l593_59390

/-- Calculates the difference in distance traveled between two drivers meeting on a highway --/
theorem drivers_distance_difference
  (initial_distance : ℝ)
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : initial_distance = 787)
  (h2 : speed_a = 90)
  (h3 : speed_b = 80)
  (h4 : delay = 1) :
  let remaining_distance := initial_distance - speed_a * delay
  let relative_speed := speed_a + speed_b
  let meeting_time := remaining_distance / relative_speed
  let distance_a := speed_a * (meeting_time + delay)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 131 := by sorry

end NUMINAMATH_CALUDE_drivers_distance_difference_l593_59390


namespace NUMINAMATH_CALUDE_smallest_third_altitude_l593_59312

/-- An isosceles triangle with integer altitudes -/
structure IsoscelesTriangle where
  -- The length of the equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The altitude to the equal sides
  altitude_to_equal_side : ℝ
  -- The altitude to the base
  altitude_to_base : ℝ
  -- Constraint: the triangle is isosceles
  isosceles : side > 0
  -- Constraint: altitudes are positive
  altitude_to_equal_side_pos : altitude_to_equal_side > 0
  altitude_to_base_pos : altitude_to_base > 0
  -- Constraint: altitudes are integers
  altitude_to_equal_side_int : ∃ n : ℤ, altitude_to_equal_side = n
  altitude_to_base_int : ∃ n : ℤ, altitude_to_base = n

/-- The theorem stating the smallest possible value for the third altitude -/
theorem smallest_third_altitude (t : IsoscelesTriangle) 
  (h1 : t.altitude_to_equal_side = 15)
  (h2 : t.altitude_to_base = 5) :
  ∃ h : ℝ, h ≥ 5 ∧ 
  (∀ h' : ℝ, (∃ n : ℤ, h' = n) → 
    (2 * t.side * t.base = t.altitude_to_equal_side * t.base + t.altitude_to_base * t.side) → 
    h' ≥ h) := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_l593_59312


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l593_59354

/-- The area of a circular sector with central angle 120° and radius √3 is π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 120 * π / 180  -- Convert 120° to radians
  let radius : ℝ := Real.sqrt 3
  let sector_area := (1 / 2) * angle * radius^2
  sector_area = π := by sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l593_59354


namespace NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_two_l593_59399

/-- Given a complex number z defined as z = 2/(1+i) + (1+i)^2, prove that its modulus |z| is equal to √2 -/
theorem modulus_of_z_is_sqrt_two : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 + Complex.I)^2
  ‖z‖ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_two_l593_59399


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l593_59386

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l593_59386


namespace NUMINAMATH_CALUDE_existence_of_finite_set_with_1993_unit_distance_neighbors_l593_59320

theorem existence_of_finite_set_with_1993_unit_distance_neighbors :
  ∃ (A : Set (ℝ × ℝ)), Set.Finite A ∧
    ∀ X ∈ A, ∃ (Y : Fin 1993 → ℝ × ℝ),
      (∀ i, Y i ∈ A) ∧
      (∀ i j, i ≠ j → Y i ≠ Y j) ∧
      (∀ i, dist X (Y i) = 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_finite_set_with_1993_unit_distance_neighbors_l593_59320


namespace NUMINAMATH_CALUDE_quadratic_sum_l593_59342

/-- A quadratic function f(x) = ax^2 + bx + c passing through (-2,0) and (4,0) with maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c (-2) = 0 →
  QuadraticFunction a b c 4 = 0 →
  (∀ x, QuadraticFunction a b c x ≤ 54) →
  (∃ x, QuadraticFunction a b c x = 54) →
  a + b + c = 54 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l593_59342


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l593_59302

theorem rectangular_field_perimeter (a b d A : ℝ) : 
  a = 2 * b →                 -- One side is twice as long as the other
  a * b = A →                 -- Area is A
  a^2 + b^2 = d^2 →           -- Pythagorean theorem for diagonal
  A = 240 →                   -- Area is 240 square meters
  d = 34 →                    -- Diagonal is 34 meters
  2 * (a + b) = 91.2 :=       -- Perimeter is 91.2 meters
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l593_59302


namespace NUMINAMATH_CALUDE_limit_of_S_is_infinity_l593_59340

def S (n : ℕ) : ℕ := (n + 1) * n / 2

theorem limit_of_S_is_infinity :
  ∀ M : ℝ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (S n : ℝ) > M :=
sorry

end NUMINAMATH_CALUDE_limit_of_S_is_infinity_l593_59340


namespace NUMINAMATH_CALUDE_length_of_diagonal_l593_59328

/-- Given two triangles AOC and BOD sharing a vertex O, with specified side lengths,
    prove that the length of AC is √1036/7 -/
theorem length_of_diagonal (AO BO CO DO BD : ℝ) (x : ℝ) 
    (h1 : AO = 3)
    (h2 : CO = 5)
    (h3 : BO = 7)
    (h4 : DO = 6)
    (h5 : BD = 11)
    (h6 : x = Real.sqrt (AO^2 + CO^2 - 2*AO*CO*(BO^2 + DO^2 - BD^2)/(2*BO*DO))) :
  x = Real.sqrt 1036 / 7 := by
  sorry

end NUMINAMATH_CALUDE_length_of_diagonal_l593_59328


namespace NUMINAMATH_CALUDE_circle_equation_and_line_intersection_l593_59317

/-- Represents a circle with center on the x-axis -/
structure CircleOnXAxis where
  center : ℤ
  radius : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def is_tangent (circle : CircleOnXAxis) (line : Line) : Prop :=
  (|line.a * circle.center + line.c| / Real.sqrt (line.a^2 + line.b^2)) = circle.radius

def intersects_circle (circle : CircleOnXAxis) (line : Line) : Prop :=
  ∃ x y : ℝ, line.a * x + line.b * y + line.c = 0 ∧
             (x - circle.center)^2 + y^2 = circle.radius^2

theorem circle_equation_and_line_intersection
  (circle : CircleOnXAxis)
  (tangent_line : Line)
  (h_radius : circle.radius = 5)
  (h_tangent : is_tangent circle tangent_line)
  (h_tangent_eq : tangent_line.a = 4 ∧ tangent_line.b = 3 ∧ tangent_line.c = -29) :
  (∃ equation : ℝ → ℝ → Prop, ∀ x y, equation x y ↔ (x - 1)^2 + y^2 = 25) ∧
  (∀ a : ℝ, a > 0 →
    let intersecting_line : Line := { a := a, b := -1, c := 5 }
    intersects_circle circle intersecting_line ↔ a > 5/12) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_line_intersection_l593_59317


namespace NUMINAMATH_CALUDE_speed_difference_l593_59379

/-- Proves that the difference between the car's and truck's average speeds is 18 km/h -/
theorem speed_difference (truck_distance : ℝ) (truck_time : ℝ) (car_time : ℝ) (distance_difference : ℝ)
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : distance_difference = 6.5)
  (h5 : (truck_distance + distance_difference) / car_time > truck_distance / truck_time) :
  (truck_distance + distance_difference) / car_time - truck_distance / truck_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l593_59379


namespace NUMINAMATH_CALUDE_count_grid_paths_l593_59375

/-- The number of paths from (0,0) to (m,n) in a grid with only right and up steps -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths from (0,0) to (m,n) is (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) :
  grid_paths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_count_grid_paths_l593_59375


namespace NUMINAMATH_CALUDE_parabola_max_distance_to_point_l593_59358

/-- The parabola C: y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line on the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The distance between a point and a line -/
def distance_point_to_line (pt : Point) (l : Line) : ℝ :=
  sorry

theorem parabola_max_distance_to_point 
  (C : Parabola) 
  (A : Point)
  (hA : A.x = -1/2 ∧ A.y = 1/2)
  (hAxis : A.x = -C.p/2)
  (M N : Point)
  (hM : M.y^2 = 2 * C.p * M.x)
  (hN : N.y^2 = 2 * C.p * N.x)
  (hMN : M.y * N.y < 0)
  (O : Point)
  (hO : O.x = 0 ∧ O.y = 0)
  (hDot : (M.x - O.x) * (N.x - O.x) + (M.y - O.y) * (N.y - O.y) = 3)
  : ∃ (l : Line), ∀ (l' : Line), 
    (∃ (P Q : Point), P.y^2 = 2 * C.p * P.x ∧ Q.y^2 = 2 * C.p * Q.x ∧ P.y * Q.y < 0 ∧ 
      P.y = l'.slope * P.x + l'.intercept ∧ Q.y = l'.slope * Q.x + l'.intercept) →
    distance_point_to_line A l ≥ distance_point_to_line A l' ∧
    distance_point_to_line A l = 5 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_max_distance_to_point_l593_59358


namespace NUMINAMATH_CALUDE_max_garden_area_l593_59344

theorem max_garden_area (L : ℝ) (h : L > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = L ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + 2*b = L → x*y ≥ a*b ∧
  x*y = L^2/8 :=
sorry

end NUMINAMATH_CALUDE_max_garden_area_l593_59344


namespace NUMINAMATH_CALUDE_prob_spade_seven_red_l593_59310

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of sevens in a standard deck -/
def NumSevens : ℕ := 4

/-- Number of red cards in a standard deck -/
def NumRed : ℕ := 26

/-- Probability of drawing a spade, then a 7, then a red card from a standard 52-card deck -/
theorem prob_spade_seven_red (deck : ℕ) (spades : ℕ) (sevens : ℕ) (red : ℕ) :
  deck = StandardDeck →
  spades = NumSpades →
  sevens = NumSevens →
  red = NumRed →
  (spades / deck) * (sevens / (deck - 1)) * (red / (deck - 2)) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_seven_red_l593_59310


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l593_59387

theorem twenty_is_eighty_percent_of_twentyfive : ∃ y : ℝ, y > 0 ∧ 20 / y = 80 / 100 → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l593_59387


namespace NUMINAMATH_CALUDE_chicken_feathers_l593_59341

theorem chicken_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : 
  initial_feathers = 5263 →
  cars_dodged = 23 →
  initial_feathers - (cars_dodged * 2) = 5217 := by
  sorry

end NUMINAMATH_CALUDE_chicken_feathers_l593_59341


namespace NUMINAMATH_CALUDE_triangle_could_be_isosceles_l593_59368

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  t.c^2 - t.a^2 + t.b^2 = (4*t.a*t.c - 2*t.b*t.c) * Real.cos t.A

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem statement
theorem triangle_could_be_isosceles (t : Triangle) 
  (h : satisfiesCondition t) : 
  ∃ (t' : Triangle), satisfiesCondition t' ∧ isIsosceles t' :=
sorry

end NUMINAMATH_CALUDE_triangle_could_be_isosceles_l593_59368


namespace NUMINAMATH_CALUDE_investment_growth_l593_59339

/-- Calculates the final amount after simple interest --/
def finalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 5 years is $350 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  finalAmount principal ((amount_after_2_years - principal) / (principal * 2)) 5 = 350 :=
by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l593_59339


namespace NUMINAMATH_CALUDE_al_wins_probability_l593_59369

/-- Represents the possible moves in Rock Paper Scissors -/
inductive Move
| Rock
| Paper
| Scissors

/-- The probability of Bob playing each move -/
def bobProbability : Move → ℚ
| Move.Rock => 1/3
| Move.Paper => 1/3
| Move.Scissors => 1/3

/-- Al's move is Rock -/
def alMove : Move := Move.Rock

/-- Determines if Al wins given Bob's move -/
def alWins (bobMove : Move) : Bool :=
  match bobMove with
  | Move.Scissors => true
  | _ => false

/-- The probability of Al winning -/
def probAlWins : ℚ := bobProbability Move.Scissors

theorem al_wins_probability :
  probAlWins = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_al_wins_probability_l593_59369


namespace NUMINAMATH_CALUDE_range_of_r_l593_59355

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l593_59355


namespace NUMINAMATH_CALUDE_absent_days_calculation_l593_59313

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days based on contract details -/
def absentDays (contract : ContractDetails) : ℚ :=
  (contract.totalDays * contract.paymentPerDay - contract.totalReceived) / (contract.paymentPerDay + contract.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 8 -/
theorem absent_days_calculation (contract : ContractDetails) 
  (h1 : contract.totalDays = 30)
  (h2 : contract.paymentPerDay = 25)
  (h3 : contract.finePerDay = 7.5)
  (h4 : contract.totalReceived = 490) :
  absentDays contract = 8 := by
  sorry

#eval absentDays { totalDays := 30, paymentPerDay := 25, finePerDay := 7.5, totalReceived := 490 }

end NUMINAMATH_CALUDE_absent_days_calculation_l593_59313


namespace NUMINAMATH_CALUDE_sin_330_degrees_l593_59337

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l593_59337


namespace NUMINAMATH_CALUDE_parallel_linear_functions_touch_theorem_l593_59329

/-- Two linear functions that are parallel but not parallel to the coordinate axes -/
structure ParallelLinearFunctions where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The condition that (f(x))^2 touches 20g(x) -/
def touches_condition_1 (f : ParallelLinearFunctions) : Prop :=
  ∃! x : ℝ, (f.a * x + f.b)^2 = 20 * (f.a * x + f.c)

/-- The condition that (g(x))^2 touches f(x)/A -/
def touches_condition_2 (f : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x : ℝ, (f.a * x + f.c)^2 = (f.a * x + f.b) / A

/-- The main theorem -/
theorem parallel_linear_functions_touch_theorem (f : ParallelLinearFunctions) :
  touches_condition_1 f → (touches_condition_2 f A ↔ A = -1/20) :=
sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_touch_theorem_l593_59329


namespace NUMINAMATH_CALUDE_candy_bar_cost_l593_59367

/-- The cost of a candy bar given initial and remaining amounts --/
theorem candy_bar_cost (initial : ℕ) (remaining : ℕ) (cost : ℕ) :
  initial = 4 →
  remaining = 3 →
  initial = remaining + cost →
  cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l593_59367


namespace NUMINAMATH_CALUDE_emus_per_pen_l593_59331

/-- Proves that the number of emus in each pen is 6 -/
theorem emus_per_pen (num_pens : ℕ) (eggs_per_week : ℕ) (h1 : num_pens = 4) (h2 : eggs_per_week = 84) : 
  (eggs_per_week / 7 * 2) / num_pens = 6 := by
  sorry

#check emus_per_pen

end NUMINAMATH_CALUDE_emus_per_pen_l593_59331


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l593_59318

theorem unique_congruence_in_range : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 7 ∧ n ≡ 12345 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l593_59318


namespace NUMINAMATH_CALUDE_f_min_at_three_l593_59349

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l593_59349


namespace NUMINAMATH_CALUDE_circle_m_range_l593_59309

-- Define the equation as a function of x, y, and m
def circle_equation (x y m : ℝ) : ℝ := x^2 + y^2 - 2*m*x + 2*m^2 + 2*m - 3

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y m = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ -3 < m ∧ m < 1/2 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l593_59309


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l593_59346

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (center : ℝ × ℝ), center = (x, y) ∧ r > 0

-- Define internal tangency of M to C₁
def internalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₁ (x + r) y

-- Define external tangency of M to C₂
def externalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₂ (x - r) y

-- Theorem statement
theorem trajectory_is_ellipse :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), internalTangent x y r ∧ externalTangent x y r) →
    x^2 / 16 + y^2 / 15 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l593_59346


namespace NUMINAMATH_CALUDE_unique_valid_result_exists_correct_answers_for_71_score_l593_59307

/-- Represents the score and correct answers for a math competition. -/
structure CompetitionResult where
  groupA_correct : Nat
  groupB_correct : Nat
  groupB_incorrect : Nat
  total_score : Int

/-- Checks if the CompetitionResult is valid according to the competition rules. -/
def is_valid_result (r : CompetitionResult) : Prop :=
  r.groupA_correct ≤ 5 ∧
  r.groupB_correct + r.groupB_incorrect ≤ 12 ∧
  r.total_score = 8 * r.groupA_correct + 5 * r.groupB_correct - 2 * r.groupB_incorrect

/-- Theorem stating that there is a unique valid result with a total score of 71 and 13 correct answers. -/
theorem unique_valid_result_exists :
  ∃! r : CompetitionResult,
    is_valid_result r ∧
    r.total_score = 71 ∧
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

/-- Theorem stating that any valid result with a total score of 71 must have 13 correct answers. -/
theorem correct_answers_for_71_score :
  ∀ r : CompetitionResult,
    is_valid_result r → r.total_score = 71 →
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_result_exists_correct_answers_for_71_score_l593_59307


namespace NUMINAMATH_CALUDE_pyramid_numbers_l593_59362

theorem pyramid_numbers (a b : ℕ) : 
  (42 = a * 6) → 
  (72 = 6 * b) → 
  (504 = 42 * 72) → 
  (a = 7 ∧ b = 12) := by
sorry

end NUMINAMATH_CALUDE_pyramid_numbers_l593_59362


namespace NUMINAMATH_CALUDE_sum_greater_than_two_l593_59322

theorem sum_greater_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_two_l593_59322


namespace NUMINAMATH_CALUDE_long_video_multiple_is_42_l593_59359

/-- Represents the video release schedule and durations for John's channel --/
structure VideoSchedule where
  short_videos_per_day : Nat
  long_videos_per_day : Nat
  short_video_duration : Nat
  days_per_week : Nat
  total_weekly_duration : Nat

/-- Calculates how many times longer the long video is compared to a short video --/
def long_video_multiple (schedule : VideoSchedule) : Nat :=
  let total_short_duration := schedule.short_videos_per_day * schedule.short_video_duration * schedule.days_per_week
  let long_video_duration := schedule.total_weekly_duration - total_short_duration
  long_video_duration / (schedule.long_videos_per_day * schedule.days_per_week * schedule.short_video_duration)

theorem long_video_multiple_is_42 (schedule : VideoSchedule) 
  (h1 : schedule.short_videos_per_day = 2)
  (h2 : schedule.long_videos_per_day = 1)
  (h3 : schedule.short_video_duration = 2)
  (h4 : schedule.days_per_week = 7)
  (h5 : schedule.total_weekly_duration = 112) :
  long_video_multiple schedule = 42 := by
  sorry

#eval long_video_multiple {
  short_videos_per_day := 2,
  long_videos_per_day := 1,
  short_video_duration := 2,
  days_per_week := 7,
  total_weekly_duration := 112
}

end NUMINAMATH_CALUDE_long_video_multiple_is_42_l593_59359


namespace NUMINAMATH_CALUDE_solve_for_B_l593_59385

theorem solve_for_B : ∃ B : ℚ, (3 * B - 5 = 23) ∧ (B = 28 / 3) := by sorry

end NUMINAMATH_CALUDE_solve_for_B_l593_59385


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l593_59352

theorem complex_in_second_quadrant (θ : Real) (h : θ ∈ Set.Ioo (3*Real.pi/4) (5*Real.pi/4)) :
  let z : ℂ := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l593_59352


namespace NUMINAMATH_CALUDE_particular_solution_correct_l593_59370

/-- The differential equation xy' = y - 1 -/
def diff_eq (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

/-- The general solution y = Cx + 1 -/
def general_solution (C : ℝ) (x : ℝ) : ℝ :=
  C * x + 1

/-- The particular solution y = 4x + 1 -/
def particular_solution (x : ℝ) : ℝ :=
  4 * x + 1

theorem particular_solution_correct :
  ∀ C : ℝ,
  (∀ x : ℝ, diff_eq x (general_solution C)) →
  general_solution C 1 = 5 →
  ∀ x : ℝ, general_solution C x = particular_solution x :=
by sorry

end NUMINAMATH_CALUDE_particular_solution_correct_l593_59370
