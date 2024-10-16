import Mathlib

namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3103_310342

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ q, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l3103_310342


namespace NUMINAMATH_CALUDE_M_N_disjoint_l3103_310339

def M : Set ℝ := {a | a > 1/2 ∧ a ≠ 1}
def N : Set ℝ := {a | 0 < a ∧ a ≤ 1/2}

theorem M_N_disjoint : M ∩ N = ∅ := by sorry

end NUMINAMATH_CALUDE_M_N_disjoint_l3103_310339


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l3103_310392

theorem henrys_cd_collection (country rock classical : ℕ) : 
  country = rock + 3 →
  rock = 2 * classical →
  country = 23 →
  classical = 10 := by
sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l3103_310392


namespace NUMINAMATH_CALUDE_cube_edge_length_proof_l3103_310305

-- Define the vessel dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15
def water_level_rise : ℝ := 3.3333333333333335

-- Define the cube's edge length
def cube_edge_length : ℝ := 10

-- Theorem statement
theorem cube_edge_length_proof :
  let vessel_base_area := vessel_length * vessel_width
  let water_volume_displaced := vessel_base_area * water_level_rise
  water_volume_displaced = cube_edge_length ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_proof_l3103_310305


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3103_310390

theorem cone_lateral_surface_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 120) :
  (θ / 360) * π * r^2 = 12 * π :=
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3103_310390


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l3103_310302

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (num_shelves : ℕ) 
  (h1 : initial_stock = 120)
  (h2 : books_sold = 39)
  (h3 : num_shelves = 9)
  (h4 : num_shelves > 0) :
  (initial_stock - books_sold) / num_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l3103_310302


namespace NUMINAMATH_CALUDE_no_digit_move_multiplier_l3103_310375

theorem no_digit_move_multiplier : ¬∃ (N : ℕ), 
  ∃ (d : ℕ) (M : ℕ) (k : ℕ),
    (N = d * 10^k + M) ∧ 
    (d ≥ 1) ∧ (d ≤ 9) ∧ 
    (10 * M + d = 5 * N ∨ 10 * M + d = 6 * N ∨ 10 * M + d = 8 * N) := by
  sorry

end NUMINAMATH_CALUDE_no_digit_move_multiplier_l3103_310375


namespace NUMINAMATH_CALUDE_min_value_for_four_digit_product_l3103_310324

theorem min_value_for_four_digit_product (n : ℕ) : 
  (341 * n ≥ 1000 ∧ ∀ m < n, 341 * m < 1000) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_for_four_digit_product_l3103_310324


namespace NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l3103_310346

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l3103_310346


namespace NUMINAMATH_CALUDE_sector_angle_l3103_310372

theorem sector_angle (r : ℝ) (θ : ℝ) (h1 : r * θ = 5) (h2 : r^2 * θ / 2 = 5) : θ = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3103_310372


namespace NUMINAMATH_CALUDE_honey_production_optimal_tax_revenue_optimal_l3103_310356

/-- The inverse demand function for honey -/
def inverse_demand (Q : ℝ) : ℝ := 310 - 3 * Q

/-- The production cost per jar of honey -/
def production_cost : ℝ := 10

/-- The profit function without tax -/
def profit (Q : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q

/-- The profit function with tax -/
def profit_with_tax (Q t : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q - t * Q

/-- The tax revenue function -/
def tax_revenue (Q t : ℝ) : ℝ := Q * t

theorem honey_production_optimal (Q : ℝ) :
  profit Q ≤ profit 50 := by sorry

theorem tax_revenue_optimal (t : ℝ) :
  tax_revenue ((310 - t) / 6) t ≤ tax_revenue ((310 - 150) / 6) 150 := by sorry

end NUMINAMATH_CALUDE_honey_production_optimal_tax_revenue_optimal_l3103_310356


namespace NUMINAMATH_CALUDE_simplify_expression_l3103_310338

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) + 3^(n+2)) / (3*(3^(n+3))) = 29/27 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3103_310338


namespace NUMINAMATH_CALUDE_unique_function_solution_l3103_310316

theorem unique_function_solution :
  ∃! f : ℕ → ℕ, ∀ x y : ℕ, x > 0 ∧ y > 0 →
    f x + y * (f (f x)) < x * (1 + f y) + 2021 ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3103_310316


namespace NUMINAMATH_CALUDE_find_a_value_l3103_310330

theorem find_a_value (x : ℝ) (a : ℝ) : 
  (2 * x - 3 = 5 * x - 2 * a) → (x = 1) → (a = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3103_310330


namespace NUMINAMATH_CALUDE_uncommon_card_ratio_l3103_310300

/-- Given a number of card packs, cards per pack, and total uncommon cards,
    prove that the ratio of uncommon cards to total cards per pack is 5:2 -/
theorem uncommon_card_ratio
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (total_uncommon : ℕ)
  (h1 : num_packs = 10)
  (h2 : cards_per_pack = 20)
  (h3 : total_uncommon = 50) :
  (total_uncommon : ℚ) / (num_packs * cards_per_pack : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_card_ratio_l3103_310300


namespace NUMINAMATH_CALUDE_evaluate_expression_l3103_310350

theorem evaluate_expression (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3103_310350


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l3103_310332

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
def DrawOutcome := Prod BallColor BallColor

/-- The set of all possible outcomes when drawing two balls -/
def SampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def ExactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def ExactlyTwoBlack : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = SampleSpace

theorem exactly_one_two_black_mutually_exclusive_not_complementary :
  MutuallyExclusive ExactlyOneBlack ExactlyTwoBlack ∧
  ¬Complementary ExactlyOneBlack ExactlyTwoBlack :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l3103_310332


namespace NUMINAMATH_CALUDE_quilt_sewing_percentage_l3103_310378

theorem quilt_sewing_percentage (total_squares : ℕ) (squares_left : ℕ) : 
  total_squares = 32 → squares_left = 24 → 
  (total_squares - squares_left : ℚ) / total_squares * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_quilt_sewing_percentage_l3103_310378


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3103_310347

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  9*a + 3*b + 6*c + d = -173 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3103_310347


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l3103_310358

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The value of sapphires in base 7 -/
def sapphires : List Nat := [2, 3, 5, 6]

/-- The value of silverware in base 7 -/
def silverware : List Nat := [0, 5, 6, 1]

/-- The value of spices in base 7 -/
def spices : List Nat := [0, 5, 2]

/-- The theorem stating the sum of the treasures in base 10 -/
theorem pirate_treasure_sum :
  base7ToBase10 sapphires + base7ToBase10 silverware + base7ToBase10 spices = 3131 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l3103_310358


namespace NUMINAMATH_CALUDE_inscribed_circles_diameter_l3103_310385

/-- A sequence of circles inscribed in a parabola -/
def InscribedCircles (ω : ℕ → Set (ℝ × ℝ)) : Prop :=
  ∀ n : ℕ, 
    -- Each circle is inscribed in the parabola y = x²
    (∀ (x y : ℝ), (x, y) ∈ ω n → y = x^2) ∧
    -- Each circle is tangent to the next one
    (∃ (x y : ℝ), (x, y) ∈ ω n ∧ (x, y) ∈ ω (n + 1)) ∧
    -- The first circle has diameter 1 and touches the parabola at (0,0)
    (n = 1 → (0, 0) ∈ ω 1 ∧ ∃ (x y : ℝ), (x, y) ∈ ω 1 ∧ x^2 + y^2 = 1/4)

/-- The diameter of a circle -/
def Diameter (ω : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of the nth circle is 2n - 1 -/
theorem inscribed_circles_diameter 
  (ω : ℕ → Set (ℝ × ℝ)) 
  (h : InscribedCircles ω) :
  ∀ n : ℕ, n > 0 → Diameter (ω n) = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_diameter_l3103_310385


namespace NUMINAMATH_CALUDE_star_equality_implies_y_value_l3103_310312

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that if (5, 0) ★ (2, -2) = (x, y) ★ (0, 3), then y = -5 -/
theorem star_equality_implies_y_value (x y : ℤ) :
  star (5, 0) (2, -2) = star (x, y) (0, 3) → y = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_y_value_l3103_310312


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3103_310388

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i^2 * (1 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3103_310388


namespace NUMINAMATH_CALUDE_binomial_16_12_l3103_310394

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_12_l3103_310394


namespace NUMINAMATH_CALUDE_base8_573_equals_379_l3103_310393

/-- Converts a base-8 number to base 10 --/
def base8_to_base10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

/-- The base-8 number 573₈ is equal to 379 in base 10 --/
theorem base8_573_equals_379 : base8_to_base10 5 7 3 = 379 := by
  sorry

end NUMINAMATH_CALUDE_base8_573_equals_379_l3103_310393


namespace NUMINAMATH_CALUDE_arc_length_sector_l3103_310357

/-- The length of an arc in a sector with radius 3 and central angle 120° is 2π -/
theorem arc_length_sector (r : ℝ) (θ : ℝ) : 
  r = 3 → θ = 120 → 2 * π * r * (θ / 360) = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_sector_l3103_310357


namespace NUMINAMATH_CALUDE_find_a_l3103_310337

def A : Set ℝ := {x | x^2 + 6*x < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a-2)*x - 2*a < 0}

theorem find_a : 
  A ∪ B a = {x : ℝ | -6 < x ∧ x < 5} → a = 5 := by
sorry

end NUMINAMATH_CALUDE_find_a_l3103_310337


namespace NUMINAMATH_CALUDE_remainder_of_sum_l3103_310317

theorem remainder_of_sum (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : (x + 3 * u * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l3103_310317


namespace NUMINAMATH_CALUDE_reciprocal_not_one_others_are_l3103_310379

theorem reciprocal_not_one_others_are (x : ℝ) (hx : x = -1) : 
  (x⁻¹ ≠ 1) ∧ (-x = 1) ∧ (|x| = 1) ∧ (x^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_not_one_others_are_l3103_310379


namespace NUMINAMATH_CALUDE_proportional_function_ratio_l3103_310303

theorem proportional_function_ratio (k a b : ℝ) : 
  k ≠ 0 →
  b ≠ 0 →
  3 = k * 1 →
  b = k * a →
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_proportional_function_ratio_l3103_310303


namespace NUMINAMATH_CALUDE_crazy_silly_school_unwatched_movies_l3103_310318

/-- Given a total number of movies and the number of watched movies,
    calculate the number of unwatched movies -/
def unwatched_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

/-- Theorem: In the 'crazy silly school' series, with 8 total movies
    and 4 watched movies, there are 4 unwatched movies -/
theorem crazy_silly_school_unwatched_movies :
  unwatched_movies 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_unwatched_movies_l3103_310318


namespace NUMINAMATH_CALUDE_triangular_trip_distance_l3103_310349

theorem triangular_trip_distance 
  (XY XZ YZ : ℝ) 
  (h1 : XY = 5000) 
  (h2 : XZ = 4000) 
  (h3 : YZ * YZ = XY * XY - XZ * XZ) : 
  XY + YZ + XZ = 12000 := by
sorry

end NUMINAMATH_CALUDE_triangular_trip_distance_l3103_310349


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_min_a_for_inequality_l3103_310397

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - 3
def g (x : ℝ) : ℝ := |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x > -2} :=
sorry

-- Theorem for the second part of the problem
theorem min_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, f x < g x + a) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_min_a_for_inequality_l3103_310397


namespace NUMINAMATH_CALUDE_partnership_capital_share_l3103_310361

theorem partnership_capital_share 
  (total_capital : ℝ) 
  (total_profit : ℝ) 
  (a_profit_share : ℝ) 
  (b_capital_share : ℝ) 
  (c_capital_share : ℝ) 
  (h1 : b_capital_share = (1 / 4 : ℝ) * total_capital) 
  (h2 : c_capital_share = (1 / 5 : ℝ) * total_capital) 
  (h3 : a_profit_share = (800 : ℝ)) 
  (h4 : total_profit = (2400 : ℝ)) 
  (h5 : a_profit_share / total_profit = (1 / 3 : ℝ)) :
  ∃ (a_capital_share : ℝ), 
    a_capital_share = (1 / 3 : ℝ) * total_capital ∧ 
    a_capital_share + b_capital_share + c_capital_share ≤ total_capital := by
  sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l3103_310361


namespace NUMINAMATH_CALUDE_round_trip_completion_l3103_310304

/-- Represents a round trip with equal outbound and inbound journeys -/
structure RoundTrip where
  total_distance : ℝ
  outbound_distance : ℝ
  inbound_distance : ℝ
  equal_journeys : outbound_distance = inbound_distance
  total_is_sum : total_distance = outbound_distance + inbound_distance

/-- Theorem stating that completing the outbound journey and 20% of the inbound journey
    results in completing 60% of the total trip -/
theorem round_trip_completion (trip : RoundTrip) :
  trip.outbound_distance + 0.2 * trip.inbound_distance = 0.6 * trip.total_distance := by
  sorry

end NUMINAMATH_CALUDE_round_trip_completion_l3103_310304


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_condition_l3103_310360

/-- The function f(x) = √x - a ln(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x - a * Real.log (x + 1)

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x) - a / (x + 1)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 → (fun x => x + Real.log 2) = fun x => f (-1) 1 + f_deriv (-1) 1 * (x - 1) := by sorry

theorem monotonicity_condition (a : ℝ) :
  a ≤ 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_condition_l3103_310360


namespace NUMINAMATH_CALUDE_triangle_problem_l3103_310381

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  (Real.cos C) / (Real.cos B) = (3 * a - c) / b →
  -- Part 1: Value of sin B
  Real.sin B = (2 * Real.sqrt 2) / 3 ∧
  -- Part 2: Area of triangle ABC when b = 4√2 and a = c
  (b = 4 * Real.sqrt 2 ∧ a = c →
    (1/2) * a * c * Real.sin B = 8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3103_310381


namespace NUMINAMATH_CALUDE_unique_solution_natural_equation_l3103_310323

theorem unique_solution_natural_equation :
  ∀ (a b x y : ℕ),
    x^(a + b) + y = x^a * y^b →
    (x = 2 ∧ y = 4) ∧ (∀ (x' y' : ℕ), x'^(a + b) + y' = x'^a * y'^b → x' = 2 ∧ y' = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_natural_equation_l3103_310323


namespace NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_9_l3103_310396

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (children : ℕ) : ℕ :=
  let adults := total_people - children
  let child_price := (total_revenue - adult_price * adults) / children
  child_price

theorem child_ticket_cost_is_9 :
  child_ticket_cost 16 24 258 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_9_l3103_310396


namespace NUMINAMATH_CALUDE_rods_to_furlongs_l3103_310386

/-- Conversion factor from furlongs to rods -/
def furlong_to_rods : ℕ := 50

/-- The number of rods we want to convert -/
def total_rods : ℕ := 1000

/-- The theorem states that 1000 rods is equal to 20 furlongs -/
theorem rods_to_furlongs : 
  (total_rods : ℚ) / furlong_to_rods = 20 := by sorry

end NUMINAMATH_CALUDE_rods_to_furlongs_l3103_310386


namespace NUMINAMATH_CALUDE_spinner_probability_l3103_310380

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_spinner_probability_l3103_310380


namespace NUMINAMATH_CALUDE_pumpkin_total_weight_l3103_310389

/-- The total weight of two pumpkins is 12.7 pounds, given their individual weights -/
theorem pumpkin_total_weight (weight1 weight2 : ℝ) 
  (h1 : weight1 = 4) 
  (h2 : weight2 = 8.7) : 
  weight1 + weight2 = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_total_weight_l3103_310389


namespace NUMINAMATH_CALUDE_square_field_area_l3103_310391

theorem square_field_area (diagonal : ℝ) (h : diagonal = 16) : 
  let side := diagonal / Real.sqrt 2
  let area := side ^ 2
  area = 128 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3103_310391


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3103_310354

theorem inequality_solution_set : 
  {x : ℝ | (x^2 + x^3 - 3*x^4) / (x + x^2 - 3*x^3) ≥ -1} = 
  {x : ℝ | x ∈ Set.Icc (-1) (((-1 - Real.sqrt 13) / 6 : ℝ)) ∪ 
           Set.Ioo (((-1 - Real.sqrt 13) / 6 : ℝ)) (((-1 + Real.sqrt 13) / 6 : ℝ)) ∪
           Set.Ioo (((-1 + Real.sqrt 13) / 6 : ℝ)) 0 ∪
           Set.Ioi 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3103_310354


namespace NUMINAMATH_CALUDE_product_of_symmetrical_complex_l3103_310387

/-- Two complex numbers are symmetrical about y = x if their real and imaginary parts are swapped -/
def symmetrical_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetrical_complex : ∀ z₁ z₂ : ℂ,
  symmetrical_about_y_eq_x z₁ z₂ →
  z₁ = 3 + 2*I →
  z₁ * z₂ = 13*I :=
sorry

end NUMINAMATH_CALUDE_product_of_symmetrical_complex_l3103_310387


namespace NUMINAMATH_CALUDE_probability_integer_log_l3103_310351

/-- The set S of powers of 3 from 1 to 18 -/
def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 18 ∧ n = 3^k}

/-- The condition for log_a b to be an integer -/
def is_integer_log (a b : ℕ) : Prop :=
  ∃ k : ℕ, a^k = b

/-- The number of valid pairs (a,b) where log_a b is an integer -/
def count_valid_pairs : ℕ := 40

/-- The total number of distinct pairs from S -/
def total_pairs : ℕ := 153

/-- The main theorem stating the probability -/
theorem probability_integer_log :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 40 / 153 := by sorry

end NUMINAMATH_CALUDE_probability_integer_log_l3103_310351


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3103_310369

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), 
    (∀ n : ℤ, n ∈ S ↔ (Real.sqrt n ≤ Real.sqrt (3 * n - 9) ∧ Real.sqrt (3 * n - 9) < Real.sqrt (n + 8))) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3103_310369


namespace NUMINAMATH_CALUDE_min_Q_value_l3103_310373

/-- The integer closest to a rational number -/
def closest_integer (m : ℤ) (k : ℤ) : ℤ := sorry

/-- The probability Q(k) as defined in the problem -/
def Q (k : ℤ) : ℚ := sorry

theorem min_Q_value :
  ∀ k : ℤ, k % 2 = 1 → 1 ≤ k → k ≤ 150 → Q k ≥ 37/75 := by sorry

end NUMINAMATH_CALUDE_min_Q_value_l3103_310373


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3103_310382

/-- Given an arithmetic sequence {aₙ}, prove that 3a₅ + a₇ = 20 when a₃ + a₈ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 3 + a 8 = 10) →                               -- given condition
  3 * a 5 + a 7 = 20 :=                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3103_310382


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3103_310376

/-- Represents the data from a single round of fish catching --/
structure RoundData where
  caught : Nat
  tagged : Nat

/-- Represents the data from the fish population study --/
structure FishStudy where
  round1 : RoundData
  round2 : RoundData
  round3 : RoundData

/-- The Lincoln-Petersen estimator function --/
def lincolnPetersen (c1 c2 r2 : Nat) : Nat :=
  (c1 * c2) / r2

/-- Theorem stating that the estimated fish population is 800 --/
theorem fish_population_estimate (study : FishStudy)
    (h1 : study.round1 = { caught := 30, tagged := 0 })
    (h2 : study.round2 = { caught := 80, tagged := 6 })
    (h3 : study.round3 = { caught := 100, tagged := 10 }) :
    lincolnPetersen study.round2.caught study.round3.caught study.round3.tagged = 800 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l3103_310376


namespace NUMINAMATH_CALUDE_option2_higher_expectation_l3103_310319

/-- Represents the number of red and white balls in the box -/
structure BallCount where
  red : ℕ
  white : ℕ

/-- Represents the two lottery options -/
inductive LotteryOption
  | Option1
  | Option2

/-- Calculates the expected value for Option 1 -/
def expectedValueOption1 (initial : BallCount) : ℚ :=
  sorry

/-- Calculates the expected value for Option 2 -/
def expectedValueOption2 (initial : BallCount) : ℚ :=
  sorry

/-- Theorem stating that Option 2 has a higher expected value -/
theorem option2_higher_expectation (initial : BallCount) :
  initial.red = 3 ∧ initial.white = 3 →
  expectedValueOption2 initial > expectedValueOption1 initial :=
sorry

end NUMINAMATH_CALUDE_option2_higher_expectation_l3103_310319


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3103_310334

/-- A regular hexagon with six segments drawn inside -/
structure SegmentedHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The lengths of the six segments -/
  segment_lengths : Fin 6 → ℝ
  /-- The segments are drawn sequentially with right angles between them -/
  segments_right_angled : Bool
  /-- The segments have lengths from 1 to 6 -/
  segment_lengths_valid : ∀ i, segment_lengths i = (i : ℝ) + 1

/-- The theorem stating that the side length of the hexagon is 15/2 -/
theorem hexagon_side_length (h : SegmentedHexagon) : h.side_length = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_side_length_l3103_310334


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3103_310398

theorem arithmetic_sequence_length :
  ∀ (a₁ l d : ℤ),
  a₁ = -48 →
  d = 6 →
  l = 78 →
  ∃ n : ℕ,
    n > 0 ∧
    l = a₁ + d * (n - 1) ∧
    n = 22 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3103_310398


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3103_310365

/-- Proves the relationship between y-coordinates of points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = -6 / (-2) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 3 →
  y₂ > y₁ ∧ y₁ > y₃ :=
by
  sorry

#check inverse_proportion_y_relationship

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3103_310365


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3103_310322

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3103_310322


namespace NUMINAMATH_CALUDE_astronaut_revolutions_l3103_310321

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of the three circles -/
structure CircleConfiguration where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  n : ℕ

/-- Defines the conditions of the problem -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  config.n > 2 ∧
  config.c₁.radius = config.n * config.c₃.radius ∧
  config.c₂.radius = 2 * config.c₃.radius

/-- Calculates the number of revolutions of c₃ relative to the ground -/
noncomputable def revolutions (config : CircleConfiguration) : ℝ :=
  config.n - 1

/-- The main theorem to be proved -/
theorem astronaut_revolutions 
  (config : CircleConfiguration) 
  (h : ValidConfiguration config) :
  revolutions config = config.n - 1 := by
  sorry

end NUMINAMATH_CALUDE_astronaut_revolutions_l3103_310321


namespace NUMINAMATH_CALUDE_tetrahedron_face_sum_squares_l3103_310335

/-- A tetrahedron with circumradius 1 and face triangles with sides a, b, and c -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the face triangle sides of a tetrahedron with circumradius 1 is equal to 8 -/
theorem tetrahedron_face_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_sum_squares_l3103_310335


namespace NUMINAMATH_CALUDE_predicted_y_value_l3103_310374

-- Define the linear regression equation
def linear_regression (x : ℝ) (a : ℝ) : ℝ := -0.7 * x + a

-- Define the mean values
def x_mean : ℝ := 1
def y_mean : ℝ := 0.3

-- Theorem statement
theorem predicted_y_value :
  ∃ (a : ℝ), 
    (linear_regression x_mean a = y_mean) ∧ 
    (linear_regression 2 a = -0.4) := by
  sorry

end NUMINAMATH_CALUDE_predicted_y_value_l3103_310374


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l3103_310325

theorem subset_implies_a_range (M N : Set ℝ) (a : ℝ) 
  (hM : M = {x : ℝ | x - 2 < 0})
  (hN : N = {x : ℝ | x < a})
  (hSubset : M ⊆ N) :
  a ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l3103_310325


namespace NUMINAMATH_CALUDE_pentatonic_scale_theorem_l3103_310301

/-- Calculates the length of the instrument for the nth note in the pentatonic scale,
    given the initial length and the number of alternations between subtracting
    and adding one-third. -/
def pentatonic_length (initial_length : ℚ) (n : ℕ) : ℚ :=
  initial_length * (2/3)^(n/2) * (4/3)^((n-1)/2)

theorem pentatonic_scale_theorem (a : ℚ) :
  pentatonic_length a 3 = 32 → a = 54 := by
  sorry

#check pentatonic_scale_theorem

end NUMINAMATH_CALUDE_pentatonic_scale_theorem_l3103_310301


namespace NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3103_310344

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3103_310344


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3103_310371

/-- The number of arrangements for a team of 3 players selected from 5 players (2 veterans and 3 new players) -/
def num_arrangements : ℕ :=
  let total_players : ℕ := 5
  let veteran_players : ℕ := 2
  let new_players : ℕ := 3
  let team_size : ℕ := 3
  -- Calculate the number of arrangements
  sorry

/-- Theorem stating that the number of valid arrangements is 48 -/
theorem valid_arrangements_count :
  num_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3103_310371


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3103_310311

/-- 
Given an arithmetic sequence starting with 3, 7, 11, ..., 
prove that its fifth term is 19.
-/
theorem fifth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℕ), 
  (a 0 = 3) → 
  (a 1 = 7) → 
  (a 2 = 11) → 
  (∀ n, a (n + 1) - a n = a 1 - a 0) → 
  a 4 = 19 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3103_310311


namespace NUMINAMATH_CALUDE_complex_division_l3103_310336

theorem complex_division (i : ℂ) : i^2 = -1 → (2 : ℂ) / (1 + i) = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l3103_310336


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l3103_310340

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ),
    (∀ x, f' x₀ ≤ f' x) ∧
    y₀ = f x₀ ∧
    (∀ x y, y = f x → 3*x - y - 2 = 0 ∨ 3*x - y - 2 > 0) ∧
    3*x₀ - y₀ - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l3103_310340


namespace NUMINAMATH_CALUDE_large_square_area_l3103_310306

/-- The area of a square formed by four congruent rectangles and a smaller square -/
theorem large_square_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 20) : 
  (x + y)^2 = 400 := by
  sorry

#check large_square_area

end NUMINAMATH_CALUDE_large_square_area_l3103_310306


namespace NUMINAMATH_CALUDE_tan_4125_degrees_l3103_310366

theorem tan_4125_degrees : Real.tan (4125 * π / 180) = -(2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_tan_4125_degrees_l3103_310366


namespace NUMINAMATH_CALUDE_dissimilarTerms_eq_distributionWays_l3103_310363

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := Nat.choose 13 3

/-- The number of ways to distribute 10 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ := Nat.choose 13 3

theorem dissimilarTerms_eq_distributionWays : dissimilarTerms = distributionWays := by
  sorry

end NUMINAMATH_CALUDE_dissimilarTerms_eq_distributionWays_l3103_310363


namespace NUMINAMATH_CALUDE_linear_regression_average_increase_l3103_310310

/- Define a linear regression model -/
def LinearRegression (x y : ℝ → ℝ) (a b : ℝ) :=
  ∀ t, y t = b * x t + a

/- Define the average increase in y when x increases by 1 unit -/
def AverageIncrease (x y : ℝ → ℝ) (b : ℝ) :=
  ∀ t, y (t + 1) - y t = b

/- Theorem: In a linear regression model, when x increases by 1 unit,
   y increases by b units on average -/
theorem linear_regression_average_increase
  (x y : ℝ → ℝ) (a b : ℝ)
  (h : LinearRegression x y a b) :
  AverageIncrease x y b :=
by sorry

end NUMINAMATH_CALUDE_linear_regression_average_increase_l3103_310310


namespace NUMINAMATH_CALUDE_plates_per_meal_l3103_310320

theorem plates_per_meal (guests : ℕ) (people : ℕ) (meals_per_day : ℕ) (days : ℕ) (total_plates : ℕ) :
  guests = 5 →
  people = guests + 1 →
  meals_per_day = 3 →
  days = 4 →
  total_plates = 144 →
  (total_plates / (people * meals_per_day * days) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_plates_per_meal_l3103_310320


namespace NUMINAMATH_CALUDE_trapezium_height_l3103_310331

theorem trapezium_height (a b area : ℝ) (ha : a > 0) (hb : b > 0) (harea : area > 0) :
  a = 4 → b = 5 → area = 27 →
  (area = (a + b) * (area / ((a + b) / 2)) / 2) →
  area / ((a + b) / 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l3103_310331


namespace NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3103_310315

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem factorial_30_trailing_zeros : 
  trailing_zeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3103_310315


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_9230_l3103_310314

/-- Calculates the sale in the fourth month given the sales for other months and the average sale. -/
def fourth_month_sale (first_month second_month third_month fifth_month sixth_month average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + second_month + third_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the fourth month is 9230 given the problem conditions. -/
theorem fourth_month_sale_is_9230 :
  fourth_month_sale 8435 8927 8855 8562 6991 8500 = 9230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_is_9230_l3103_310314


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l3103_310343

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_function_theorem (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : f a b c a = a^3)
  (h2 : f a b c b = b^3) :
  c = 16 := by sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l3103_310343


namespace NUMINAMATH_CALUDE_quadratic_sum_l3103_310355

/-- Given two quadratic equations with solution sets M and N, prove that p + q = 21 -/
theorem quadratic_sum (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x ∈ M, x^2 - p*x + 6 = 0) ∧ 
    (∀ x ∈ N, x^2 + 6*x - q = 0) ∧ 
    (M ∩ N = {2})) →
  p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3103_310355


namespace NUMINAMATH_CALUDE_equation_solution_set_l3103_310359

theorem equation_solution_set : ∃ (S : Set ℝ),
  S = {x : ℝ | Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                x ≥ 5 ∧ x ≤ 10} ∧
  ∀ x : ℝ, x ∈ S ↔ (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                    x ≥ 5 ∧ x ≤ 10) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3103_310359


namespace NUMINAMATH_CALUDE_expected_winnings_is_negative_half_dollar_l3103_310328

/-- Represents the sections of the spinner --/
inductive Section
  | Red
  | Blue
  | Green
  | Yellow

/-- Returns the probability of landing on a given section --/
def probability (s : Section) : ℚ :=
  match s with
  | Section.Red => 3/8
  | Section.Blue => 1/4
  | Section.Green => 1/4
  | Section.Yellow => 1/8

/-- Returns the winnings (in dollars) for a given section --/
def winnings (s : Section) : ℤ :=
  match s with
  | Section.Red => 2
  | Section.Blue => 4
  | Section.Green => -3
  | Section.Yellow => -6

/-- Calculates the expected winnings from spinning the spinner --/
def expectedWinnings : ℚ :=
  (probability Section.Red * winnings Section.Red) +
  (probability Section.Blue * winnings Section.Blue) +
  (probability Section.Green * winnings Section.Green) +
  (probability Section.Yellow * winnings Section.Yellow)

/-- Theorem stating that the expected winnings is -$0.50 --/
theorem expected_winnings_is_negative_half_dollar :
  expectedWinnings = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_negative_half_dollar_l3103_310328


namespace NUMINAMATH_CALUDE_a_3_equals_negative_8_l3103_310377

/-- The sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (x : ℝ) : ℝ := (x^2 + 3*x)*2^n - x + 1

/-- The n-th term of the geometric sequence -/
def a (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then S 1 x
  else S n x - S (n-1) x

/-- The common ratio of the geometric sequence -/
def q : ℝ := 2

/-- The value of x that satisfies the given condition -/
def x : ℝ := -1

theorem a_3_equals_negative_8 : a 3 x = -8 := by sorry

end NUMINAMATH_CALUDE_a_3_equals_negative_8_l3103_310377


namespace NUMINAMATH_CALUDE_abc_inequality_l3103_310384

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3103_310384


namespace NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l3103_310367

/-- Calculates the interest percentage given the conditions of an encyclopedia purchase --/
theorem encyclopedia_interest_percentage 
  (down_payment : ℚ)
  (total_cost : ℚ)
  (monthly_payment : ℚ)
  (num_monthly_payments : ℕ)
  (final_payment : ℚ)
  (h1 : down_payment = 300)
  (h2 : total_cost = 750)
  (h3 : monthly_payment = 57)
  (h4 : num_monthly_payments = 9)
  (h5 : final_payment = 21) :
  let total_paid := down_payment + (monthly_payment * num_monthly_payments) + final_payment
  let amount_borrowed := total_cost - down_payment
  let interest_paid := total_paid - total_cost
  interest_paid / amount_borrowed = 8533 / 10000 := by
sorry


end NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l3103_310367


namespace NUMINAMATH_CALUDE_max_marks_calculation_l3103_310352

theorem max_marks_calculation (passing_threshold : ℚ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  short_marks = 13 →
  ∃ max_marks : ℕ,
    max_marks = 750 ∧
    (scored_marks + short_marks : ℚ) / max_marks = passing_threshold :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l3103_310352


namespace NUMINAMATH_CALUDE_candy_theorem_l3103_310364

def candy_problem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) : ℕ :=
  bars_per_friend * num_friends + spare_bars

theorem candy_theorem (bars_per_friend : ℕ) (num_friends : ℕ) (spare_bars : ℕ) :
  candy_problem bars_per_friend num_friends spare_bars =
  bars_per_friend * num_friends + spare_bars :=
by
  sorry

#eval candy_problem 2 7 10

end NUMINAMATH_CALUDE_candy_theorem_l3103_310364


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3103_310313

theorem shaded_area_fraction (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let triangle_area := (1/2) * (s/2) * (s/2)
  let shaded_area := 2 * triangle_area
  shaded_area / square_area = (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3103_310313


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_max_value_achieved_l3103_310326

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49) 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 :=
by sorry

theorem max_value_achieved : 
  ∃ x, x ∈ Set.Icc (-49) 49 ∧ Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_max_value_achieved_l3103_310326


namespace NUMINAMATH_CALUDE_jenny_run_distance_l3103_310368

theorem jenny_run_distance (walked : Real) (ran_extra : Real) : 
  walked = 0.4 → ran_extra = 0.2 → walked + ran_extra = 0.6 := by
sorry

end NUMINAMATH_CALUDE_jenny_run_distance_l3103_310368


namespace NUMINAMATH_CALUDE_bread_slices_eaten_for_breakfast_l3103_310333

theorem bread_slices_eaten_for_breakfast 
  (total_slices : ℕ) 
  (lunch_slices : ℕ) 
  (remaining_slices : ℕ) 
  (h1 : total_slices = 12)
  (h2 : lunch_slices = 2)
  (h3 : remaining_slices = 6) :
  (total_slices - (remaining_slices + lunch_slices)) / total_slices = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_eaten_for_breakfast_l3103_310333


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3103_310353

/-- Given vectors a and b in R^2, with b = (-1, 2) and their sum (1, 3), 
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3103_310353


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3103_310362

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3103_310362


namespace NUMINAMATH_CALUDE_toys_between_l3103_310395

theorem toys_between (n : ℕ) (pos_a pos_b : ℕ) (h1 : n = 19) (h2 : pos_a = 9) (h3 : pos_b = 15) :
  pos_b - pos_a - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_between_l3103_310395


namespace NUMINAMATH_CALUDE_exam_time_ratio_l3103_310345

/-- Given an examination with the following parameters:
  * Total duration: 3 hours
  * Total number of questions: 200
  * Number of type A problems: 25
  * Time spent on type A problems: 40 minutes

  Prove that the ratio of time spent on type A problems to time spent on type B problems is 2:7. -/
theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- Total time in minutes
  let type_a_time : ℕ := 40     -- Time spent on type A problems
  let type_b_time : ℕ := total_time - type_a_time  -- Time spent on type B problems
  (type_a_time : ℚ) / (type_b_time : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l3103_310345


namespace NUMINAMATH_CALUDE_cube_sum_problem_l3103_310309

theorem cube_sum_problem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l3103_310309


namespace NUMINAMATH_CALUDE_vector_problem_l3103_310329

/-- The angle between two 2D vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Checks if two 2D vectors are collinear -/
def collinear (v w : ℝ × ℝ) : Prop := sorry

/-- Checks if two 2D vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop := sorry

theorem vector_problem (a b c : ℝ × ℝ) 
  (ha : a = (1, 2))
  (hb : b = (-2, 6))
  (hc : c = (-1, 3)) : 
  angle a b = π/4 ∧ 
  collinear b c ∧ 
  perpendicular a (a - c) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3103_310329


namespace NUMINAMATH_CALUDE_odd_score_probability_is_four_ninths_l3103_310370

/-- Represents the possible points on the dart board -/
inductive DartPoints
  | Three
  | Four

/-- Represents the regions on the dart board -/
structure DartRegion where
  isInner : Bool
  points : DartPoints

/-- The dart board configuration -/
def dartBoard : List DartRegion :=
  [
    { isInner := true,  points := DartPoints.Three },
    { isInner := true,  points := DartPoints.Four },
    { isInner := true,  points := DartPoints.Four },
    { isInner := false, points := DartPoints.Four },
    { isInner := false, points := DartPoints.Three },
    { isInner := false, points := DartPoints.Three }
  ]

/-- The probability of hitting each region -/
def regionProbability (region : DartRegion) : ℚ :=
  if region.isInner then 1 / 21 else 2 / 21

/-- The probability of getting an odd score with two dart throws -/
def oddScoreProbability : ℚ := sorry

/-- Theorem stating that the probability of getting an odd score is 4/9 -/
theorem odd_score_probability_is_four_ninths :
  oddScoreProbability = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_odd_score_probability_is_four_ninths_l3103_310370


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3103_310308

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = (5 * total_homes) / 8)
  (h2 : collapsing = (11 * termite_ridden) / 16) :
  (termite_ridden - collapsing) = (25 * total_homes) / 128 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3103_310308


namespace NUMINAMATH_CALUDE_optimal_usage_duration_l3103_310383

/-- Represents the cost structure and usage of a car -/
structure CarCost where
  initialCost : ℕ := 150000
  annualExpenses : ℕ := 15000
  firstYearMaintenance : ℕ := 3000
  maintenanceIncrease : ℕ := 3000

/-- Calculates the average annual cost of using the car for a given number of years -/
def averageAnnualCost (c : CarCost) (years : ℕ) : ℚ :=
  let totalCost : ℚ := c.initialCost + c.annualExpenses * years + (years * (c.firstYearMaintenance + c.maintenanceIncrease * years) / 2)
  totalCost / years

/-- States that 10 years is the optimal duration to use the car -/
theorem optimal_usage_duration (c : CarCost) :
  ∀ n : ℕ, n ≠ 0 → n ≠ 10 → averageAnnualCost c 10 ≤ averageAnnualCost c n :=
sorry

end NUMINAMATH_CALUDE_optimal_usage_duration_l3103_310383


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3103_310327

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3103_310327


namespace NUMINAMATH_CALUDE_common_volume_theorem_l3103_310341

/-- Represents a triangular pyramid with a point O on the segment connecting
    the vertex with the intersection point of base medians -/
structure TriangularPyramid where
  volume : ℝ
  ratio : ℝ

/-- Calculates the volume of the common part of the original pyramid
    and its symmetric counterpart with respect to point O -/
noncomputable def commonVolume (pyramid : TriangularPyramid) : ℝ :=
  if pyramid.ratio = 1 then 2 * pyramid.volume / 9
  else if pyramid.ratio = 3 then pyramid.volume / 2
  else if pyramid.ratio = 2 then 110 * pyramid.volume / 243
  else if pyramid.ratio = 4 then 12 * pyramid.volume / 25
  else 0  -- undefined for other ratios

theorem common_volume_theorem (pyramid : TriangularPyramid) :
  (pyramid.ratio = 1 → commonVolume pyramid = 2 * pyramid.volume / 9) ∧
  (pyramid.ratio = 3 → commonVolume pyramid = pyramid.volume / 2) ∧
  (pyramid.ratio = 2 → commonVolume pyramid = 110 * pyramid.volume / 243) ∧
  (pyramid.ratio = 4 → commonVolume pyramid = 12 * pyramid.volume / 25) :=
by sorry

end NUMINAMATH_CALUDE_common_volume_theorem_l3103_310341


namespace NUMINAMATH_CALUDE_aria_apple_purchase_l3103_310348

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Aria needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Aria should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem aria_apple_purchase : apples_to_buy = 14 := by
  sorry

end NUMINAMATH_CALUDE_aria_apple_purchase_l3103_310348


namespace NUMINAMATH_CALUDE_stratified_sample_problem_l3103_310307

/-- Given a stratified sample from a high school where:
  * The sample size is 55 students
  * 10 students are from the first year
  * 25 students are from the second year
  * There are 400 students in the third year
Prove that the total number of students in the first and second years combined is 700. -/
theorem stratified_sample_problem (sample_size : ℕ) (first_year_sample : ℕ) (second_year_sample : ℕ) (third_year_total : ℕ) :
  sample_size = 55 →
  first_year_sample = 10 →
  second_year_sample = 25 →
  third_year_total = 400 →
  ∃ (first_and_second_total : ℕ),
    first_and_second_total = 700 ∧
    (first_year_sample + second_year_sample : ℚ) / sample_size = first_and_second_total / (first_and_second_total + third_year_total) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_problem_l3103_310307


namespace NUMINAMATH_CALUDE_total_pencils_l3103_310399

-- Define the number of pencils in each set
def pencils_set_a : ℕ := 10
def pencils_set_b : ℕ := 20
def pencils_set_c : ℕ := 30

-- Define the number of friends who bought each set
def friends_set_a : ℕ := 3
def friends_set_b : ℕ := 2
def friends_set_c : ℕ := 2

-- Define Chloe's purchase
def chloe_sets : ℕ := 1

-- Theorem statement
theorem total_pencils :
  (friends_set_a * pencils_set_a + 
   friends_set_b * pencils_set_b + 
   friends_set_c * pencils_set_c) +
  (chloe_sets * (pencils_set_a + pencils_set_b + pencils_set_c)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3103_310399
