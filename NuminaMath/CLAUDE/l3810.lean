import Mathlib

namespace NUMINAMATH_CALUDE_eight_coin_flips_sequences_l3810_381048

/-- The number of distinct sequences for n coin flips -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of distinct sequences for 8 coin flips is 256 -/
theorem eight_coin_flips_sequences : coin_sequences 8 = 256 := by
  sorry

end NUMINAMATH_CALUDE_eight_coin_flips_sequences_l3810_381048


namespace NUMINAMATH_CALUDE_inequality_proof_l3810_381012

theorem inequality_proof (a b c x y z k : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : a + x = k ∧ b + y = k ∧ c + z = k) : 
  a * x + b * y + c * z < k^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3810_381012


namespace NUMINAMATH_CALUDE_square_figure_perimeter_l3810_381042

/-- A figure composed of four identical squares -/
structure SquareFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 100 cm² -/
  area_eq : 4 * side_length ^ 2 = 100

/-- The perimeter of the square figure -/
def perimeter (fig : SquareFigure) : ℝ := 10 * fig.side_length

/-- Theorem stating that the perimeter of the square figure is 50 cm -/
theorem square_figure_perimeter (fig : SquareFigure) : perimeter fig = 50 := by
  sorry


end NUMINAMATH_CALUDE_square_figure_perimeter_l3810_381042


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3810_381028

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 19 →
  neither = 2 →
  ∃ (both : ℕ), both = 8 ∧
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3810_381028


namespace NUMINAMATH_CALUDE_motorcyclist_speed_l3810_381095

theorem motorcyclist_speed 
  (hiker_speed : ℝ)
  (time_to_stop : ℝ)
  (time_to_catch_up : ℝ)
  (h1 : hiker_speed = 6)
  (h2 : time_to_stop = 0.2)
  (h3 : time_to_catch_up = 0.8) :
  ∃ (motorcyclist_speed : ℝ),
    motorcyclist_speed * time_to_stop = 
    hiker_speed * (time_to_stop + time_to_catch_up) ∧
    motorcyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_speed_l3810_381095


namespace NUMINAMATH_CALUDE_exposed_sides_is_30_l3810_381063

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons -/
structure PolygonArrangement :=
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (octagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the arrangement -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.hexagon.sides +
  arrangement.heptagon.sides +
  arrangement.octagon.sides +
  arrangement.nonagon.sides -
  12 -- Subtracting the shared sides

/-- The specific arrangement described in the problem -/
def specificArrangement : PolygonArrangement :=
  { triangle := ⟨3⟩
  , square := ⟨4⟩
  , pentagon := ⟨5⟩
  , hexagon := ⟨6⟩
  , heptagon := ⟨7⟩
  , octagon := ⟨8⟩
  , nonagon := ⟨9⟩ }

/-- Theorem stating that the number of exposed sides in the specific arrangement is 30 -/
theorem exposed_sides_is_30 : exposedSides specificArrangement = 30 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_is_30_l3810_381063


namespace NUMINAMATH_CALUDE_difference_of_101st_terms_l3810_381058

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem difference_of_101st_terms : 
  let X := arithmetic_sequence 40 12
  let Y := arithmetic_sequence 40 (-8)
  |X 101 - Y 101| = 2000 := by
sorry

end NUMINAMATH_CALUDE_difference_of_101st_terms_l3810_381058


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3810_381088

/-- Given a hyperbola and a parabola, if the asymptote of the hyperbola
    intersects the parabola at only one point, then the eccentricity
    of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℝ, ∀ x y : ℝ,
    (x^2/a^2 - y^2/b^2 = 1 → y = k*x) ∧
    (x^2 = y - 1 → y = k*x) →
    (∀ z : ℝ, z ≠ x → x^2 = z - 1 → y ≠ k*z)) →
  let c := Real.sqrt (a^2 + b^2)
  c/a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3810_381088


namespace NUMINAMATH_CALUDE_coin_counting_fee_percentage_l3810_381076

def coinValue (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

def totalValue (quarters dimes nickels pennies : ℕ) : ℚ :=
  quarters * coinValue "quarter" + 
  dimes * coinValue "dime" + 
  nickels * coinValue "nickel" + 
  pennies * coinValue "penny"

theorem coin_counting_fee_percentage 
  (quarters dimes nickels pennies : ℕ) 
  (amountAfterFee : ℚ) : 
  quarters = 76 → 
  dimes = 85 → 
  nickels = 20 → 
  pennies = 150 → 
  amountAfterFee = 27 → 
  (totalValue quarters dimes nickels pennies - amountAfterFee) / 
  (totalValue quarters dimes nickels pennies) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_counting_fee_percentage_l3810_381076


namespace NUMINAMATH_CALUDE_parabola_sum_l3810_381078

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 10) has p + q + r = 8 1/3 -/
theorem parabola_sum (p q r : ℝ) : 
  (∀ x y : ℝ, y = p * x^2 + q * x + r) →  -- Equation of the parabola
  (∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 7) →  -- Vertex form with (3, 7)
  (10 : ℝ) = p * 0^2 + q * 0 + r →  -- Point (0, 10) on the parabola
  p + q + r = 8 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l3810_381078


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l3810_381044

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * Real.pi / 180) * Real.sin (36 * Real.pi / 180) * 
  Real.sin (54 * Real.pi / 180) * Real.sin (72 * Real.pi / 180) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l3810_381044


namespace NUMINAMATH_CALUDE_investment_options_count_l3810_381026

/-- The number of ways to distribute 3 distinct projects among 5 cities, 
    with no more than 2 projects per city. -/
def investmentOptions : ℕ := 120

/-- The number of cities available for investment. -/
def numCities : ℕ := 5

/-- The number of projects to be distributed. -/
def numProjects : ℕ := 3

/-- The maximum number of projects allowed in a single city. -/
def maxProjectsPerCity : ℕ := 2

theorem investment_options_count :
  investmentOptions = 
    (numCities.factorial / (numCities - numProjects).factorial) +
    (numCities.choose 1) * (numProjects.choose 2) * ((numCities - 1).choose 1) :=
by sorry

end NUMINAMATH_CALUDE_investment_options_count_l3810_381026


namespace NUMINAMATH_CALUDE_x_value_proof_l3810_381096

theorem x_value_proof (x : ℚ) (h : (1/2 : ℚ) - (1/4 : ℚ) + (1/8 : ℚ) = 8/x) : x = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3810_381096


namespace NUMINAMATH_CALUDE_domestic_tourists_scientific_notation_l3810_381047

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem domestic_tourists_scientific_notation :
  toScientificNotation 274000000 =
    ScientificNotation.mk 2.74 8 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_domestic_tourists_scientific_notation_l3810_381047


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3810_381039

theorem multiplication_addition_equality : 45 * 52 + 78 * 45 = 5850 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3810_381039


namespace NUMINAMATH_CALUDE_rectangle_width_l3810_381053

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its width is 12 cm. -/
theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 50 ∧ length = 13 ∧ perimeter = 2 * length + 2 * width → width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3810_381053


namespace NUMINAMATH_CALUDE_charity_fundraising_l3810_381003

theorem charity_fundraising 
  (total_amount : ℝ) 
  (sponsor_contribution : ℝ) 
  (number_of_people : ℕ) :
  total_amount = 2400 →
  sponsor_contribution = 300 →
  number_of_people = 8 →
  (total_amount - sponsor_contribution) / number_of_people = 262.5 := by
sorry

end NUMINAMATH_CALUDE_charity_fundraising_l3810_381003


namespace NUMINAMATH_CALUDE_power_of_power_l3810_381084

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3810_381084


namespace NUMINAMATH_CALUDE_line_equivalence_l3810_381051

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form --/
theorem line_equivalence (x y : ℝ) : 
  4 * (x + 2) - 3 * (y - 8) = 0 ↔ y = (4/3) * x + 32/3 := by
  sorry

end NUMINAMATH_CALUDE_line_equivalence_l3810_381051


namespace NUMINAMATH_CALUDE_dispatch_plans_eq_180_l3810_381023

/-- Represents the number of male officials -/
def num_males : ℕ := 5

/-- Represents the number of female officials -/
def num_females : ℕ := 3

/-- Represents the total number of officials -/
def total_officials : ℕ := num_males + num_females

/-- Represents the minimum number of officials in each group -/
def min_group_size : ℕ := 3

/-- Calculates the number of ways to divide officials into two groups -/
def dispatch_plans : ℕ := sorry

/-- Theorem stating that the number of dispatch plans is 180 -/
theorem dispatch_plans_eq_180 : dispatch_plans = 180 := by sorry

end NUMINAMATH_CALUDE_dispatch_plans_eq_180_l3810_381023


namespace NUMINAMATH_CALUDE_prime_representation_l3810_381006

theorem prime_representation (p : ℕ) (hp : p.Prime) (hp_gt_2 : p > 2) :
  (p % 8 = 1 → ∃ x y : ℤ, ↑p = x^2 + 16 * y^2) ∧
  (p % 8 = 5 → ∃ x y : ℤ, ↑p = 4 * x^2 + 4 * x * y + 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_l3810_381006


namespace NUMINAMATH_CALUDE_subset_X_l3810_381036

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_subset_X_l3810_381036


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3810_381043

/-- For a quadratic equation x^2 + x = k with two distinct real roots, k > -1/4 --/
theorem quadratic_equation_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x = k ∧ y^2 + y = k) → k > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3810_381043


namespace NUMINAMATH_CALUDE_pet_center_cats_l3810_381033

theorem pet_center_cats (initial_dogs : ℕ) (adopted_dogs : ℕ) (new_cats : ℕ) (final_total : ℕ) :
  initial_dogs = 36 →
  adopted_dogs = 20 →
  new_cats = 12 →
  final_total = 57 →
  ∃ initial_cats : ℕ,
    initial_cats = 29 ∧
    final_total = (initial_dogs - adopted_dogs) + (initial_cats + new_cats) :=
by sorry

end NUMINAMATH_CALUDE_pet_center_cats_l3810_381033


namespace NUMINAMATH_CALUDE_subsets_of_three_element_set_l3810_381089

theorem subsets_of_three_element_set : 
  Finset.card (Finset.powerset {1, 2, 3}) = 8 := by sorry

end NUMINAMATH_CALUDE_subsets_of_three_element_set_l3810_381089


namespace NUMINAMATH_CALUDE_inequality_proof_l3810_381022

theorem inequality_proof (a b c : ℝ) (M N P : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1)
  (hM : M = 2^a) (hN : N = 5^(-b)) (hP : P = Real.log c) :
  P < N ∧ N < M := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3810_381022


namespace NUMINAMATH_CALUDE_nested_fourth_root_l3810_381001

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l3810_381001


namespace NUMINAMATH_CALUDE_initial_honey_amount_honey_jar_problem_l3810_381069

/-- The amount of honey remaining after each extraction -/
def honey_remaining (initial_honey : ℝ) (num_extractions : ℕ) : ℝ :=
  initial_honey * (0.8 ^ num_extractions)

/-- Theorem stating the initial amount of honey given the final amount and number of extractions -/
theorem initial_honey_amount 
  (final_honey : ℝ) 
  (num_extractions : ℕ) 
  (h_final : honey_remaining initial_honey num_extractions = final_honey) : 
  initial_honey = final_honey / (0.8 ^ num_extractions) :=
by sorry

/-- The solution to the honey jar problem -/
theorem honey_jar_problem : 
  ∃ (initial_honey : ℝ), 
    honey_remaining initial_honey 4 = 512 ∧ 
    initial_honey = 1250 :=
by sorry

end NUMINAMATH_CALUDE_initial_honey_amount_honey_jar_problem_l3810_381069


namespace NUMINAMATH_CALUDE_base_8_to_10_conversion_l3810_381010

-- Define the base 8 number as a list of digits
def base_8_number : List Nat := [2, 4, 6]

-- Define the conversion function from base 8 to base 10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_conversion :
  base_8_to_10 base_8_number = 166 := by sorry

end NUMINAMATH_CALUDE_base_8_to_10_conversion_l3810_381010


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3810_381015

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α + π/4) = 12/13)
  (h2 : π/4 < α) 
  (h3 : α < 3*π/4) : 
  Real.cos α = 7*Real.sqrt 2/26 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3810_381015


namespace NUMINAMATH_CALUDE_margarets_mean_score_l3810_381098

def scores : List ℝ := [82, 85, 89, 91, 95, 97]

theorem margarets_mean_score (cyprians_mean : ℝ) (h1 : cyprians_mean = 88) :
  let total_sum := scores.sum
  let cyprians_sum := 3 * cyprians_mean
  let margarets_sum := total_sum - cyprians_sum
  margarets_sum / 3 = 91 + 2/3 := by sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l3810_381098


namespace NUMINAMATH_CALUDE_train_speed_l3810_381000

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 16) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3810_381000


namespace NUMINAMATH_CALUDE_power_division_equals_729_l3810_381062

theorem power_division_equals_729 : (3 : ℤ)^15 / (27 : ℤ)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l3810_381062


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l3810_381074

theorem smallest_value_in_range (x : ℝ) (h1 : -1 < x) (h2 : x < 0) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt (x^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l3810_381074


namespace NUMINAMATH_CALUDE_equation_solution_existence_l3810_381083

theorem equation_solution_existence (n : ℤ) : 
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x*y - y*z - z*x = n) → 
  (∃ a b : ℤ, a^2 + b^2 - a*b = n) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l3810_381083


namespace NUMINAMATH_CALUDE_equation_solution_existence_l3810_381052

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ (3 < a ∧ a < 27) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l3810_381052


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3810_381034

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 7 + (5 : ℚ) / 9 = (31 : ℚ) / 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3810_381034


namespace NUMINAMATH_CALUDE_equation_solution_l3810_381050

theorem equation_solution (x : ℝ) : 
  (x^2 - 2*x - 8 = -(x + 4)*(x - 1)) ↔ (x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3810_381050


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l3810_381055

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos 2α = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l3810_381055


namespace NUMINAMATH_CALUDE_triangle_side_relation_l3810_381029

theorem triangle_side_relation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l3810_381029


namespace NUMINAMATH_CALUDE_balloon_count_l3810_381002

/-- The number of gold balloons -/
def gold_balloons : ℕ := sorry

/-- The number of silver balloons -/
def silver_balloons : ℕ := sorry

/-- The number of black balloons -/
def black_balloons : ℕ := 150

theorem balloon_count : 
  (silver_balloons = 2 * gold_balloons) ∧ 
  (gold_balloons + silver_balloons + black_balloons = 573) → 
  gold_balloons = 141 :=
by sorry

end NUMINAMATH_CALUDE_balloon_count_l3810_381002


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3810_381016

/-- The time taken for a power boat to travel downstream from dock A to dock B,
    given the conditions of the river journey problem. -/
theorem power_boat_travel_time
  (r : ℝ) -- speed of the river current
  (p : ℝ) -- relative speed of the power boat with respect to the river
  (h1 : r > 0) -- river speed is positive
  (h2 : p > r) -- power boat speed is greater than river speed
  : ∃ t : ℝ,
    t > 0 ∧
    t = (12 * r) / (6 * p - r) ∧
    (p + r) * t + (p - r) * (12 - t) = 12 * r :=
by sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l3810_381016


namespace NUMINAMATH_CALUDE_f_minimum_value_g_range_condition_l3810_381040

noncomputable section

def f (x : ℝ) := 2 * x * Real.log x

def g (a x : ℝ) := -x^2 + a*x - 3

theorem f_minimum_value :
  ∃ (m : ℝ), m = 2 / Real.exp 1 ∧ ∀ x > 0, f x ≥ m :=
sorry

theorem g_range_condition (a : ℝ) :
  (∃ x > 0, f x ≤ g a x) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_g_range_condition_l3810_381040


namespace NUMINAMATH_CALUDE_gasoline_added_l3810_381066

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 29.999999999999996 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 4.499999999999999 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_added_l3810_381066


namespace NUMINAMATH_CALUDE_cos_alpha_for_given_point_l3810_381072

/-- If the terminal side of angle α passes through the point (√3/2, 1/2), then cos α = √3/2 -/
theorem cos_alpha_for_given_point (α : Real) :
  (∃ (r : Real), r * (Real.sqrt 3 / 2) = Real.cos α ∧ r * (1 / 2) = Real.sin α) →
  Real.cos α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_given_point_l3810_381072


namespace NUMINAMATH_CALUDE_triangle_problem_l3810_381075

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- arithmetic progression
  (7 * Real.sin A = 3 * Real.sin C) →
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4) →
  -- Conclusions
  (Real.cos B = 11/14) ∧ (b = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3810_381075


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_k_value_when_x_is_two_l3810_381079

/-- The quadratic equation x^2 - kx + k - 1 = 0 -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + k - 1

theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 :=
sorry

theorem k_value_when_x_is_two :
  ∃ k : ℝ, quadratic k 2 = 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_k_value_when_x_is_two_l3810_381079


namespace NUMINAMATH_CALUDE_exponent_rules_l3810_381087

theorem exponent_rules :
  (∀ x : ℝ, x^5 * x^2 = x^7) ∧
  (∀ m : ℝ, (m^2)^4 = m^8) ∧
  (∀ x y : ℝ, (-2*x*y^2)^3 = -8*x^3*y^6) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l3810_381087


namespace NUMINAMATH_CALUDE_projection_orthogonal_vectors_l3810_381030

/-- Given two orthogonal vectors a and b in ℝ², prove that if the projection of (4, -2) onto a
    is (1/2, 1), then the projection of (4, -2) onto b is (7/2, -3). -/
theorem projection_orthogonal_vectors (a b : ℝ × ℝ) : 
  a.1 * b.1 + a.2 * b.2 = 0 →  -- a and b are orthogonal
  (∃ k : ℝ, k • a = (1/2, 1) ∧ k * (a.1 * 4 + a.2 * (-2)) = a.1^2 + a.2^2) →  -- proj_a (4, -2) = (1/2, 1)
  (∃ m : ℝ, m • b = (7/2, -3) ∧ m * (b.1 * 4 + b.2 * (-2)) = b.1^2 + b.2^2)  -- proj_b (4, -2) = (7/2, -3)
  := by sorry

end NUMINAMATH_CALUDE_projection_orthogonal_vectors_l3810_381030


namespace NUMINAMATH_CALUDE_walking_speeds_l3810_381009

/-- The speeds of two people walking on a highway -/
theorem walking_speeds (x y : ℝ) : 
  (30 * x - 30 * y = 300) →  -- If both walk eastward for 30 minutes, A catches up with B
  (2 * x + 2 * y = 300) →    -- If they walk towards each other, they meet after 2 minutes
  (x = 80 ∧ y = 70) :=        -- Then A's speed is 80 m/min and B's speed is 70 m/min
by
  sorry

end NUMINAMATH_CALUDE_walking_speeds_l3810_381009


namespace NUMINAMATH_CALUDE_conic_single_point_implies_d_eq_11_l3810_381014

/-- A conic section represented by the equation 2x^2 + y^2 + 4x - 6y + d = 0 -/
def conic (d : ℝ) (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 4 * x - 6 * y + d = 0

/-- The conic degenerates to a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, conic d p.1 p.2

/-- If the conic degenerates to a single point, then d = 11 -/
theorem conic_single_point_implies_d_eq_11 :
  ∀ d : ℝ, is_single_point d → d = 11 := by sorry

end NUMINAMATH_CALUDE_conic_single_point_implies_d_eq_11_l3810_381014


namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l3810_381035

theorem line_segments_in_proportion : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l3810_381035


namespace NUMINAMATH_CALUDE_michelle_initial_crayons_l3810_381031

/-- Given that Janet has 2 crayons and the sum of Michelle's initial crayons
    and Janet's crayons is 4, prove that Michelle initially has 2 crayons. -/
theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_crayons_l3810_381031


namespace NUMINAMATH_CALUDE_inequality_proof_l3810_381017

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3810_381017


namespace NUMINAMATH_CALUDE_f_positive_iff_f_inequality_iff_l3810_381004

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the first part of the problem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the second part of the problem
theorem f_inequality_iff (a : ℝ) : 
  (∀ x : ℝ, f x + 3 * |x + 2| ≥ |a - 1|) ↔ -4 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_inequality_iff_l3810_381004


namespace NUMINAMATH_CALUDE_rainfall_difference_l3810_381090

/-- The number of Mondays -/
def numMondays : ℕ := 7

/-- The number of Tuesdays -/
def numTuesdays : ℕ := 9

/-- The amount of rain on each Monday in centimeters -/
def rainPerMonday : ℝ := 1.5

/-- The amount of rain on each Tuesday in centimeters -/
def rainPerTuesday : ℝ := 2.5

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (numTuesdays : ℝ) * rainPerTuesday - (numMondays : ℝ) * rainPerMonday = 12 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l3810_381090


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_possible_values_complete_l3810_381093

-- Define the set of possible values for |a + b + c|
def PossibleValues : Set ℝ := {1, 2, 3}

-- Main theorem
theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 →
  Complex.abs (a + b + c) ∈ PossibleValues := by
  sorry

-- Completeness of the set of possible values
theorem possible_values_complete (x : ℝ) :
  x ∈ PossibleValues →
  ∃ (a b c : ℂ), Complex.abs a = 1 ∧
                  Complex.abs b = 1 ∧
                  Complex.abs c = 1 ∧
                  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 ∧
                  Complex.abs (a + b + c) = x := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_possible_values_complete_l3810_381093


namespace NUMINAMATH_CALUDE_distance_to_origin_l3810_381038

theorem distance_to_origin (z : ℂ) (h : z = 1 - 2*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3810_381038


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3810_381099

def a (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
             (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧ 
             k = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3810_381099


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_specific_semi_circle_perimeter_l3810_381046

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2r -/
theorem semi_circle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.6 cm is approximately 33.93 cm -/
theorem specific_semi_circle_perimeter :
  let r : ℝ := 6.6
  let perimeter := π * r + 2 * r
  ∃ (approx : ℝ), abs (perimeter - approx) < 0.005 ∧ approx = 33.93 :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_specific_semi_circle_perimeter_l3810_381046


namespace NUMINAMATH_CALUDE_series_sum_minus_eight_l3810_381013

theorem series_sum_minus_eight : 
  (5/3 + 13/9 + 41/27 + 125/81 + 379/243 + 1145/729) - 8 = 950/729 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_minus_eight_l3810_381013


namespace NUMINAMATH_CALUDE_contrapositive_of_zero_product_l3810_381007

theorem contrapositive_of_zero_product (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) →
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_of_zero_product_l3810_381007


namespace NUMINAMATH_CALUDE_flour_calculation_sugar_calculation_l3810_381061

-- Define the original recipe quantities
def original_flour : ℚ := 27/4  -- 6 3/4 cups of flour
def original_sugar : ℚ := 5/2   -- 2 1/2 cups of sugar

-- Define the fraction of the recipe we want to make
def recipe_fraction : ℚ := 1/3

-- Theorem for flour calculation
theorem flour_calculation :
  recipe_fraction * original_flour = 9/4 := by sorry

-- Theorem for sugar calculation
theorem sugar_calculation :
  recipe_fraction * original_sugar = 5/6 := by sorry

end NUMINAMATH_CALUDE_flour_calculation_sugar_calculation_l3810_381061


namespace NUMINAMATH_CALUDE_problem_statement_l3810_381019

theorem problem_statement : (481 + 426)^2 - 4 * 481 * 426 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3810_381019


namespace NUMINAMATH_CALUDE_biscuits_butter_cookies_difference_l3810_381027

-- Define the number of cookies baked in the morning and afternoon
def morning_butter_cookies : ℕ := 20
def morning_biscuits : ℕ := 40
def afternoon_butter_cookies : ℕ := 10
def afternoon_biscuits : ℕ := 20

-- Define the total number of each type of cookie
def total_butter_cookies : ℕ := morning_butter_cookies + afternoon_butter_cookies
def total_biscuits : ℕ := morning_biscuits + afternoon_biscuits

-- Theorem statement
theorem biscuits_butter_cookies_difference :
  total_biscuits - total_butter_cookies = 30 := by
  sorry

end NUMINAMATH_CALUDE_biscuits_butter_cookies_difference_l3810_381027


namespace NUMINAMATH_CALUDE_monotonic_shift_l3810_381054

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being monotonic on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- State the theorem
theorem monotonic_shift (a b : ℝ) (h : MonotonicOn f a b) :
  MonotonicOn (fun x => f (x + 3)) (a - 3) (b - 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_shift_l3810_381054


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3810_381082

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The general term of an arithmetic sequence. -/
def arithmetic_general_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * (a 2 - a 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 6 = 8)
  (h_sum2 : a 3 + a 4 = 3) :
  ∀ n : ℕ, arithmetic_general_term a n = 5 * n - 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3810_381082


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l3810_381081

/-- Two non-collinear vectors in a vector space -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  not_collinear : ∃ (a b : ℝ), a • e₁ + b • e₂ ≠ 0

/-- Three collinear points in a vector space -/
structure CollinearPoints (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V
  collinear : ∃ (t : ℝ), C - A = t • (B - A)

/-- Theorem: If e₁ and e₂ are non-collinear vectors, AB = 2e₁ + me₂, BC = e₁ + 3e₂,
    and points A, B, C are collinear, then m = 6 -/
theorem collinear_points_m_value
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (ncv : NonCollinearVectors V)
  (cp : CollinearPoints V)
  (h₁ : cp.B - cp.A = 2 • ncv.e₁ + m • ncv.e₂)
  (h₂ : cp.C - cp.B = ncv.e₁ + 3 • ncv.e₂)
  : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l3810_381081


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_l3810_381091

theorem binomial_coefficient_third_term (x : ℝ) : 
  Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_l3810_381091


namespace NUMINAMATH_CALUDE_zeroPointThreeBarSix_eq_elevenThirties_l3810_381060

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLessThanOne : repeating < 1

/-- The value of a repeating decimal as a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (1 - (1/10)^(d.repeating.den))

/-- 0.3̄6 as a RepeatingDecimal -/
def zeroPointThreeBarSix : RepeatingDecimal :=
  { nonRepeating := 3/10
    repeating := 6/10
    repeatingLessThanOne := by sorry }

theorem zeroPointThreeBarSix_eq_elevenThirties : 
  zeroPointThreeBarSix.toRational = 11/30 := by sorry

end NUMINAMATH_CALUDE_zeroPointThreeBarSix_eq_elevenThirties_l3810_381060


namespace NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3810_381070

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3810_381070


namespace NUMINAMATH_CALUDE_y_derivative_l3810_381085

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + 
  (1 / 3) * (3 * x - 1) / (3 * x^2 - 2 * x + 1)

theorem y_derivative (x : ℝ) :
  deriv y x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l3810_381085


namespace NUMINAMATH_CALUDE_coopers_age_l3810_381049

theorem coopers_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (maria_older : maria = dante + 1) :
  cooper = 6 := by
  sorry

end NUMINAMATH_CALUDE_coopers_age_l3810_381049


namespace NUMINAMATH_CALUDE_base_conversion_equivalence_l3810_381041

theorem base_conversion_equivalence :
  ∀ (C B : ℕ),
    C < 9 →
    B < 6 →
    9 * C + B = 6 * B + C →
    C = 0 ∧ B = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_equivalence_l3810_381041


namespace NUMINAMATH_CALUDE_antifreeze_concentration_proof_l3810_381077

-- Define the constants
def total_volume : ℝ := 55
def pure_antifreeze_volume : ℝ := 6.11
def other_mixture_concentration : ℝ := 0.1

-- Define the theorem
theorem antifreeze_concentration_proof :
  let other_mixture_volume : ℝ := total_volume - pure_antifreeze_volume
  let total_pure_antifreeze : ℝ := pure_antifreeze_volume + other_mixture_concentration * other_mixture_volume
  let final_concentration : ℝ := total_pure_antifreeze / total_volume
  ∃ ε > 0, |final_concentration - 0.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_antifreeze_concentration_proof_l3810_381077


namespace NUMINAMATH_CALUDE_binomial_and_power_evaluation_l3810_381008

theorem binomial_and_power_evaluation : 
  (Nat.choose 12 6 = 924) ∧ ((1 + 1 : ℕ)^12 = 4096) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_power_evaluation_l3810_381008


namespace NUMINAMATH_CALUDE_universally_energetic_characterization_no_specific_energetic_triplets_l3810_381005

/-- A triplet (a, b, c) is n-energetic if it satisfies the given conditions --/
def isNEnergetic (a b c n : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Nat.gcd a (Nat.gcd b c) = 1 ∧ (a^n + b^n + c^n) % (a + b + c) = 0

/-- A triplet (a, b, c) is universally energetic if it is n-energetic for all n ≥ 1 --/
def isUniversallyEnergetic (a b c : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → isNEnergetic a b c n

/-- The set of all universally energetic triplets --/
def universallyEnergeticTriplets : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ isUniversallyEnergetic t.1 t.2.1 t.2.2}

theorem universally_energetic_characterization :
    universallyEnergeticTriplets = {(1, 1, 1), (1, 1, 4)} := by sorry

theorem no_specific_energetic_triplets :
    ∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 →
      (isNEnergetic a b c 2004 ∧ isNEnergetic a b c 2005 ∧ ¬isNEnergetic a b c 2007) → False := by sorry

end NUMINAMATH_CALUDE_universally_energetic_characterization_no_specific_energetic_triplets_l3810_381005


namespace NUMINAMATH_CALUDE_part_time_employees_count_l3810_381057

def total_employees : ℕ := 65134
def full_time_employees : ℕ := 63093

theorem part_time_employees_count : total_employees - full_time_employees = 2041 := by
  sorry

end NUMINAMATH_CALUDE_part_time_employees_count_l3810_381057


namespace NUMINAMATH_CALUDE_distance_representation_l3810_381011

theorem distance_representation (a : ℝ) : 
  |a + 1| = |a - (-1)| := by sorry

-- The statement proves that |a + 1| is equal to the distance between a and -1,
-- which represents the distance between points A and C on the number line.

end NUMINAMATH_CALUDE_distance_representation_l3810_381011


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3810_381073

theorem geometric_sequence_minimum_value (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  (∀ x : ℝ, (x - 2) * Real.exp x ≥ b) →  -- b is the minimum value of (x-2)e^x
  a * c = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3810_381073


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3810_381071

theorem triangle_perimeter : 
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 4)
  let C : ℝ × ℝ := (2, 4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist A B + dist B C + dist C A
  perimeter = 12 + 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3810_381071


namespace NUMINAMATH_CALUDE_calculation_proof_l3810_381067

theorem calculation_proof : (30 / (7 + 2 - 3)) * 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3810_381067


namespace NUMINAMATH_CALUDE_cuboid_distance_theorem_l3810_381025

/-- Given a cuboid with edges a, b, and c, and a vertex P, 
    the distance m from P to the plane passing through the vertices adjacent to P 
    satisfies the equation: 1/m² = 1/a² + 1/b² + 1/c² -/
theorem cuboid_distance_theorem (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : m > 0) :
  (1 / m^2) = (1 / a^2) + (1 / b^2) + (1 / c^2) :=
sorry

end NUMINAMATH_CALUDE_cuboid_distance_theorem_l3810_381025


namespace NUMINAMATH_CALUDE_cone_base_radius_l3810_381086

/-- A cone with surface area 3π whose lateral surface unfolds into a semicircle has base radius 1. -/
theorem cone_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem)
  3 * π * r^2 = 3 * π → -- surface area condition
  π * (2 * r) = 2 * π * r → -- lateral surface unfolds into semicircle condition
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3810_381086


namespace NUMINAMATH_CALUDE_points_per_treasure_l3810_381018

/-- Calculates the points per treasure in Tiffany's video game. -/
theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) : 
  treasures_level1 = 3 → treasures_level2 = 5 → total_score = 48 →
  total_score / (treasures_level1 + treasures_level2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l3810_381018


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l3810_381056

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) 
  (hright : a^2 + c^2 = b^2) : 
  c = Real.sqrt (b^2 - a^2) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l3810_381056


namespace NUMINAMATH_CALUDE_subset_range_m_l3810_381037

theorem subset_range_m (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  B ⊆ A → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_range_m_l3810_381037


namespace NUMINAMATH_CALUDE_area_two_quarter_circles_l3810_381065

/-- The area of a figure formed by two 90° sectors of a circle with radius 10 -/
theorem area_two_quarter_circles (r : ℝ) (h : r = 10) : 
  2 * (π * r^2 / 4) = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_area_two_quarter_circles_l3810_381065


namespace NUMINAMATH_CALUDE_square_fraction_count_l3810_381024

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ k : ℤ, n / (25 - n) = k^2 ∧ 25 - n ≠ 0) ∧ 
    S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3810_381024


namespace NUMINAMATH_CALUDE_water_balloon_problem_l3810_381080

theorem water_balloon_problem (janice randy cynthia : ℕ) : 
  cynthia = 4 * randy →
  randy = janice / 2 →
  cynthia + randy = janice + 12 →
  janice = 8 := by
sorry

end NUMINAMATH_CALUDE_water_balloon_problem_l3810_381080


namespace NUMINAMATH_CALUDE_club_officer_selection_ways_l3810_381032

def club_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_alice_bob : ℕ := 28 * 27 * 26 * 25
def ways_with_alice_bob : ℕ := 4 * 3 * 28 * 27

theorem club_officer_selection_ways :
  (ways_without_alice_bob + ways_with_alice_bob) = 500472 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_ways_l3810_381032


namespace NUMINAMATH_CALUDE_rhombus_dot_product_l3810_381021

/-- A rhombus OABC in a Cartesian coordinate system -/
structure Rhombus where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem rhombus_dot_product (r : Rhombus) : 
  r.O = (0, 0) → 
  r.A = (1, 1) → 
  dot_product (vec r.O r.A) (vec r.O r.C) = 1 → 
  dot_product (vec r.A r.B) (vec r.A r.C) = 1 := by
  sorry

#check rhombus_dot_product

end NUMINAMATH_CALUDE_rhombus_dot_product_l3810_381021


namespace NUMINAMATH_CALUDE_inequality_proof_l3810_381092

theorem inequality_proof (x : ℝ) : 
  (|(7 - x) / 4| < 3) ∧ (x ≥ 0) → (0 ≤ x ∧ x < 19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3810_381092


namespace NUMINAMATH_CALUDE_aftershave_dilution_l3810_381068

/-- Proves that adding 10 ounces of water to a 12-ounce bottle of 60% alcohol solution, 
    then removing 4 ounces of the mixture, results in a 40% alcohol solution. -/
theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (removed_amount : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧ 
  water_added = 10 ∧ 
  removed_amount = 4 ∧ 
  final_concentration = 0.4 →
  let initial_alcohol := initial_volume * initial_concentration
  let total_volume := initial_volume + water_added
  let final_volume := total_volume - removed_amount
  initial_alcohol / final_volume = final_concentration :=
by
  sorry

#check aftershave_dilution

end NUMINAMATH_CALUDE_aftershave_dilution_l3810_381068


namespace NUMINAMATH_CALUDE_jeans_pricing_l3810_381064

theorem jeans_pricing (cost : ℝ) (cost_positive : cost > 0) :
  let retailer_price := cost * (1 + 0.4)
  let customer_price := retailer_price * (1 + 0.15)
  (customer_price - cost) / cost = 0.61 := by
  sorry

end NUMINAMATH_CALUDE_jeans_pricing_l3810_381064


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l3810_381020

theorem angle_terminal_side_value (a : ℝ) (h : a > 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  let sinα := y / r
  let cosα := x / r
  2 * sinα + cosα = -19 / 13 := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l3810_381020


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3810_381059

/-- Given a quadratic function y = x^2 + 2mx + 2 with symmetry axis x = 2, prove that m = -2 -/
theorem quadratic_symmetry_axis (m : ℝ) : 
  (∀ x, x^2 + 2*m*x + 2 = (x-2)^2 + (2^2 + 2*m*2 + 2)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3810_381059


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3810_381097

-- Define set A
def A : Set ℝ := {x : ℝ | x * Real.sqrt (x^2 - 4) ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x - 1| + |x + 1| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {-2} ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3810_381097


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3810_381045

/-- Given a quadratic function y = ax^2 + bx - 1 where a ≠ 0 and 
    the graph passes through the point (1, 1), 
    prove that the value of 1 - a - b is equal to -1 -/
theorem quadratic_function_property (a b : ℝ) : 
  a ≠ 0 → 
  (1 : ℝ) = a * (1 : ℝ)^2 + b * (1 : ℝ) - 1 → 
  1 - a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3810_381045


namespace NUMINAMATH_CALUDE_dormitory_problem_l3810_381094

theorem dormitory_problem (rooms : ℕ) (students : ℕ) : 
  (students % 4 = 19) ∧ 
  (0 < students - 6 * (rooms - 1)) ∧ 
  (students - 6 * (rooms - 1) < 6) →
  ((rooms = 10 ∧ students = 59) ∨ 
   (rooms = 11 ∧ students = 63) ∨ 
   (rooms = 12 ∧ students = 67)) :=
by sorry

end NUMINAMATH_CALUDE_dormitory_problem_l3810_381094
