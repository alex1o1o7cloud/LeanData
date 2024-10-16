import Mathlib

namespace NUMINAMATH_CALUDE_permutations_of_red_l3923_392399

-- Define the number of letters in 'red'
def n : ℕ := 3

-- Theorem to prove
theorem permutations_of_red : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_red_l3923_392399


namespace NUMINAMATH_CALUDE_common_roots_product_square_l3923_392397

-- Define the two cubic equations
def cubic1 (x A : ℝ) : ℝ := x^3 + A*x + 20
def cubic2 (x B : ℝ) : ℝ := x^3 + B*x^2 + 80

-- Define the property of having two common roots
def has_two_common_roots (A B : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ 
    cubic1 p A = 0 ∧ cubic1 q A = 0 ∧
    cubic2 p B = 0 ∧ cubic2 q B = 0

-- Theorem statement
theorem common_roots_product_square (A B : ℝ) 
  (h : has_two_common_roots A B) :
  ∃ (p q : ℝ), p ≠ q ∧ 
    cubic1 p A = 0 ∧ cubic1 q A = 0 ∧
    cubic2 p B = 0 ∧ cubic2 q B = 0 ∧
    (p*q)^2 = 16 * Real.sqrt 100 :=
sorry

end NUMINAMATH_CALUDE_common_roots_product_square_l3923_392397


namespace NUMINAMATH_CALUDE_min_planes_for_300_parts_l3923_392379

def q (n : ℕ) : ℕ := (n^3 + 5*n + 6) / 6

theorem min_planes_for_300_parts : 
  (∀ m : ℕ, m < 13 → q m < 300) ∧ q 13 ≥ 300 := by sorry

end NUMINAMATH_CALUDE_min_planes_for_300_parts_l3923_392379


namespace NUMINAMATH_CALUDE_cube_sum_equality_l3923_392354

theorem cube_sum_equality (h : 2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) :
  3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l3923_392354


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l3923_392323

/-- Proves that given the conditions of the coconut grove problem, 
    when x = 8, the yield Y of each (x - 4) tree is 180 nuts per year. -/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : 
  x = 8 →
  ((x + 4) * 60 + x * 120 + (x - 4) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

#check coconut_grove_yield

end NUMINAMATH_CALUDE_coconut_grove_yield_l3923_392323


namespace NUMINAMATH_CALUDE_child_tickets_sold_l3923_392336

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ)
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l3923_392336


namespace NUMINAMATH_CALUDE_sufficient_condition_l3923_392350

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement that a_4 and a_12 are roots of x^2 + 3x = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 + 3 * a 4 = 0 ∧ a 12 ^ 2 + 3 * a 12 = 0

/-- The theorem stating that the conditions are sufficient for a_8 = ±1 -/
theorem sufficient_condition (a : ℕ → ℝ) :
  geometric_sequence a → roots_condition a → (a 8 = 1 ∨ a 8 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_l3923_392350


namespace NUMINAMATH_CALUDE_negative_twenty_one_div_three_l3923_392313

theorem negative_twenty_one_div_three : -21 / 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_twenty_one_div_three_l3923_392313


namespace NUMINAMATH_CALUDE_no_consecutive_squares_equal_consecutive_fourth_powers_l3923_392301

theorem no_consecutive_squares_equal_consecutive_fourth_powers :
  ¬ ∃ (m n : ℕ), m^2 + (m+1)^2 = n^4 + (n+1)^4 := by
sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_equal_consecutive_fourth_powers_l3923_392301


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_lcm_l3923_392335

theorem gcd_bound_for_special_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) → 
  (10^6 ≤ b ∧ b < 10^7) → 
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) → 
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_lcm_l3923_392335


namespace NUMINAMATH_CALUDE_percent_of_percent_l3923_392337

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3923_392337


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3923_392348

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k - 3 = 0 ∧ (∀ y : ℝ, y^2 + k - 3 = 0 → y = x)) ↔ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3923_392348


namespace NUMINAMATH_CALUDE_jane_age_problem_l3923_392320

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Define what it means for a number to be a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 1) ∧ is_perfect_cube (x + 1) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_jane_age_problem_l3923_392320


namespace NUMINAMATH_CALUDE_midpoint_property_l3923_392384

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x + 2y = 39 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (12, 9) → Q = (4, 6) → R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  3 * R.1 + 2 * R.2 = 39 := by
  sorry

#check midpoint_property

end NUMINAMATH_CALUDE_midpoint_property_l3923_392384


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3923_392353

theorem reciprocal_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3923_392353


namespace NUMINAMATH_CALUDE_f_properties_l3923_392333

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3923_392333


namespace NUMINAMATH_CALUDE_cheese_arrangement_count_l3923_392356

/-- Represents a cheese flavor -/
inductive Flavor
| Paprika
| BearsGarlic

/-- Represents a cheese slice -/
structure CheeseSlice :=
  (flavor : Flavor)

/-- Represents a box of cheese slices -/
structure CheeseBox :=
  (slices : List CheeseSlice)

/-- Represents an arrangement of cheese slices in two boxes -/
structure CheeseArrangement :=
  (box1 : CheeseBox)
  (box2 : CheeseBox)

/-- Checks if two arrangements are equivalent under rotation -/
def areEquivalentUnderRotation (arr1 arr2 : CheeseArrangement) : Prop :=
  sorry

/-- Counts the number of distinct arrangements -/
def countDistinctArrangements (arrangements : List CheeseArrangement) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem cheese_arrangement_count :
  let totalSlices := 16
  let paprikaSlices := 8
  let bearsGarlicSlices := 8
  let allArrangements := sorry -- List of all possible arrangements
  countDistinctArrangements allArrangements = 234 :=
sorry

end NUMINAMATH_CALUDE_cheese_arrangement_count_l3923_392356


namespace NUMINAMATH_CALUDE_pencil_distribution_l3923_392307

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) (pencils_per_student : ℕ) :
  total_pencils = 125 →
  num_students = 25 →
  pencils_per_student * num_students = total_pencils →
  pencils_per_student = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3923_392307


namespace NUMINAMATH_CALUDE_dons_pizza_consumption_l3923_392387

/-- Don's pizza consumption problem -/
theorem dons_pizza_consumption (darias_consumption : ℝ) (total_consumption : ℝ) 
  (h1 : darias_consumption = 2.5 * (total_consumption - darias_consumption))
  (h2 : total_consumption = 280) : 
  total_consumption - darias_consumption = 80 := by
  sorry

end NUMINAMATH_CALUDE_dons_pizza_consumption_l3923_392387


namespace NUMINAMATH_CALUDE_original_laborers_l3923_392381

/-- Given a piece of work that can be completed by x laborers in 15 days,
    if 5 laborers are absent and the remaining laborers complete the work in 20 days,
    then x = 20. -/
theorem original_laborers (x : ℕ) 
  (h1 : x * 15 = (x - 5) * 20) : x = 20 := by
  sorry

#check original_laborers

end NUMINAMATH_CALUDE_original_laborers_l3923_392381


namespace NUMINAMATH_CALUDE_inequality_proof_l3923_392305

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3923_392305


namespace NUMINAMATH_CALUDE_fraction_calculation_l3923_392395

theorem fraction_calculation : (3 / 4 + 2 + 1 / 3) / (1 + 1 / 2) = 37 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3923_392395


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3923_392303

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  X^4 + 3 * X^3 = (X^2 - 3 * X + 2) * q + r ∧ 
  r = 36 * X - 32 ∧ 
  r.degree < (X^2 - 3 * X + 2).degree := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3923_392303


namespace NUMINAMATH_CALUDE_distance_to_y_axis_reflection_distance_specific_point_l3923_392302

/-- The distance between a point and its reflection over the y-axis --/
theorem distance_to_y_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - (-x))^2 + (y - y)^2) = 2 * |x| :=
sorry

/-- The distance between (2, -4) and its reflection over the y-axis is 4 --/
theorem distance_specific_point : 
  Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_reflection_distance_specific_point_l3923_392302


namespace NUMINAMATH_CALUDE_larger_number_proof_l3923_392317

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 31) :
  max x y = 17 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3923_392317


namespace NUMINAMATH_CALUDE_kelly_textbook_weight_difference_l3923_392324

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem kelly_textbook_weight_difference :
  let chemistry_weight : ℚ := 7125 / 1000
  let geometry_weight : ℚ := 625 / 1000
  chemistry_weight - geometry_weight = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_kelly_textbook_weight_difference_l3923_392324


namespace NUMINAMATH_CALUDE_vector_problem_solution_l3923_392332

def vector_problem (a b : ℝ × ℝ) (m : ℝ) : Prop :=
  let norm_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b : ℝ := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
  norm_a = 3 ∧
  norm_b = 2 ∧
  dot_product a b = norm_a * norm_b * (-1/2) ∧
  dot_product (a.1 + m * b.1, a.2 + m * b.2) a = 0

theorem vector_problem_solution (a b : ℝ × ℝ) (m : ℝ) 
  (h : vector_problem a b m) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_solution_l3923_392332


namespace NUMINAMATH_CALUDE_square_perimeter_from_quadratic_root_l3923_392390

theorem square_perimeter_from_quadratic_root : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ - 10) = 0 ∧ 
  (x₂ - 1) * (x₂ - 10) = 0 ∧ 
  x₁ ≠ x₂ ∧
  (max x₁ x₂)^2 = 100 ∧
  4 * (max x₁ x₂) = 40 :=
by sorry


end NUMINAMATH_CALUDE_square_perimeter_from_quadratic_root_l3923_392390


namespace NUMINAMATH_CALUDE_tennis_tournament_balls_l3923_392362

theorem tennis_tournament_balls (total_balls : ℕ) (balls_per_can : ℕ) : 
  total_balls = 225 →
  balls_per_can = 3 →
  (8 + 4 + 2 + 1 : ℕ) * (total_balls / balls_per_can / (8 + 4 + 2 + 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_balls_l3923_392362


namespace NUMINAMATH_CALUDE_lisa_weight_l3923_392358

theorem lisa_weight (amy lisa : ℝ) 
  (h1 : amy + lisa = 240)
  (h2 : lisa - amy = lisa / 3) : 
  lisa = 144 := by sorry

end NUMINAMATH_CALUDE_lisa_weight_l3923_392358


namespace NUMINAMATH_CALUDE_consecutive_squares_theorem_l3923_392314

theorem consecutive_squares_theorem :
  (∀ x : ℤ, ¬∃ y : ℤ, 3 * x^2 + 2 = y^2) ∧
  (∀ x : ℤ, ¬∃ y : ℤ, 6 * x^2 + 6 * x + 19 = y^2) ∧
  (∃ x : ℤ, ∃ y : ℤ, 11 * x^2 + 110 = y^2) ∧
  (∃ y : ℤ, 11 * 23^2 + 110 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_theorem_l3923_392314


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l3923_392365

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-value when x = -3 -/
def y₁ : ℝ := f (-3)

/-- y₂ is the y-value when x = 4 -/
def y₂ : ℝ := f 4

/-- Theorem: For the linear function f(x) = 2x + 1, y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l3923_392365


namespace NUMINAMATH_CALUDE_cookie_ratio_l3923_392329

def cookie_problem (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) : Prop :=
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  price_per_cookie = 2 ∧
  total_money = 648 ∧
  price_per_cookie * (clementine_cookies + jake_cookies + tory_cookies) = total_money ∧
  tory_cookies * 2 = clementine_cookies + jake_cookies

theorem cookie_ratio (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) :
  cookie_problem clementine_cookies jake_cookies tory_cookies price_per_cookie total_money →
  tory_cookies * 2 = clementine_cookies + jake_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3923_392329


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3923_392396

/-- Given an initial solution of 5 liters containing 40% alcohol,
    after adding 2 liters of water and 1 liter of pure alcohol,
    the resulting mixture contains 37.5% alcohol. -/
theorem alcohol_mixture_percentage :
  let initial_volume : ℝ := 5
  let initial_alcohol_percentage : ℝ := 40 / 100
  let water_added : ℝ := 2
  let pure_alcohol_added : ℝ := 1
  let final_volume : ℝ := initial_volume + water_added + pure_alcohol_added
  let final_alcohol_volume : ℝ := initial_volume * initial_alcohol_percentage + pure_alcohol_added
  final_alcohol_volume / final_volume = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3923_392396


namespace NUMINAMATH_CALUDE_sad_children_count_l3923_392334

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : total = 60 := by sorry
  have h2 : happy = 30 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of sad children is 10
  exact total - happy - neither

end NUMINAMATH_CALUDE_sad_children_count_l3923_392334


namespace NUMINAMATH_CALUDE_meat_division_l3923_392368

theorem meat_division (pot1_weight pot2_weight total_meat : ℕ) 
  (h1 : pot1_weight = 645)
  (h2 : pot2_weight = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1_weight + meat1 = pot2_weight + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

#check meat_division

end NUMINAMATH_CALUDE_meat_division_l3923_392368


namespace NUMINAMATH_CALUDE_race_runners_count_l3923_392364

theorem race_runners_count :
  ∀ (total_runners : ℕ) (ammar_position : ℕ) (julia_position : ℕ),
  ammar_position > 0 →
  julia_position > ammar_position →
  ammar_position - 1 = (total_runners - ammar_position) / 2 →
  julia_position = ammar_position + 10 →
  julia_position - 1 = 2 * (total_runners - julia_position) →
  total_runners = 31 :=
by
  sorry

#check race_runners_count

end NUMINAMATH_CALUDE_race_runners_count_l3923_392364


namespace NUMINAMATH_CALUDE_marbles_distribution_l3923_392340

theorem marbles_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 5504 →
  num_friends = 64 →
  marbles_per_friend = total_marbles / num_friends →
  marbles_per_friend = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3923_392340


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_equals_one_eighth_l3923_392316

theorem sin_cos_pi_12_equals_one_eighth :
  1/2 * Real.sin (π/12) * Real.cos (π/12) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_equals_one_eighth_l3923_392316


namespace NUMINAMATH_CALUDE_basketball_league_games_l3923_392312

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3923_392312


namespace NUMINAMATH_CALUDE_total_surveys_completed_l3923_392370

def regular_rate : ℚ := 10
def cellphone_rate : ℚ := regular_rate * (1 + 30 / 100)
def cellphone_surveys : ℕ := 60
def total_earnings : ℚ := 1180

theorem total_surveys_completed :
  ∃ (regular_surveys : ℕ),
    (regular_surveys : ℚ) * regular_rate + 
    (cellphone_surveys : ℚ) * cellphone_rate = total_earnings ∧
    regular_surveys + cellphone_surveys = 100 :=
by sorry

end NUMINAMATH_CALUDE_total_surveys_completed_l3923_392370


namespace NUMINAMATH_CALUDE_ceiling_of_3_7_l3923_392331

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_3_7_l3923_392331


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l3923_392326

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

theorem oak_trees_after_planting :
  total_oak_trees 5 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l3923_392326


namespace NUMINAMATH_CALUDE_rearrange_3008_eq_6_l3923_392360

/-- The number of different four-digit numbers that can be formed by rearranging the digits in 3008 -/
def rearrange_3008 : ℕ :=
  let digits : List ℕ := [3, 0, 0, 8]
  let total_permutations := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_permutations := 
    (Nat.factorial 3 / Nat.factorial 2) +  -- starting with 3
    (Nat.factorial 3 / Nat.factorial 2)    -- starting with 8
  valid_permutations

theorem rearrange_3008_eq_6 : rearrange_3008 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rearrange_3008_eq_6_l3923_392360


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_recursive_formula_initial_condition_l3923_392319

/-- Arithmetic progression with first term 13.5 and common difference 4.2 -/
def arithmetic_progression (n : ℕ) : ℝ :=
  13.5 + (n - 1 : ℝ) * 4.2

/-- The nth term of the arithmetic progression -/
def nth_term (n : ℕ) : ℝ :=
  4.2 * n + 9.3

theorem arithmetic_progression_formula (n : ℕ) :
  arithmetic_progression n = nth_term n := by sorry

theorem recursive_formula (n : ℕ) (h : n > 0) :
  arithmetic_progression (n + 1) = arithmetic_progression n + 4.2 := by sorry

theorem initial_condition :
  arithmetic_progression 1 = 13.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_recursive_formula_initial_condition_l3923_392319


namespace NUMINAMATH_CALUDE_files_remaining_l3923_392349

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l3923_392349


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l3923_392386

open Real

theorem solution_set_implies_m_range (m : ℝ) :
  (∀ x : ℝ, |x - 3| - 2 - (-|x + 1| + 4) ≥ m + 1) →
  m ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l3923_392386


namespace NUMINAMATH_CALUDE_sqrt_256_equals_2_to_n_l3923_392346

theorem sqrt_256_equals_2_to_n (n : ℕ) : (256 : ℝ)^(1/2) = 2^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_256_equals_2_to_n_l3923_392346


namespace NUMINAMATH_CALUDE_red_faced_cubes_l3923_392321

theorem red_faced_cubes (n : ℕ) (h : n = 4) : 
  (n ^ 3) - (8 + 12 * (n - 2) + (n - 2) ^ 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_red_faced_cubes_l3923_392321


namespace NUMINAMATH_CALUDE_tomatoes_on_tuesday_eq_2500_l3923_392352

/-- Calculates the amount of tomatoes ready for sale on Tuesday given the initial shipment,
    sales, rotting, and new shipment. -/
def tomatoesOnTuesday (initialShipment sales rotted : ℕ) : ℕ :=
  let remainingAfterSales := initialShipment - sales
  let remainingAfterRotting := remainingAfterSales - rotted
  let newShipment := 2 * initialShipment
  remainingAfterRotting + newShipment

/-- Theorem stating that given the specific conditions, the amount of tomatoes
    ready for sale on Tuesday is 2500 kg. -/
theorem tomatoes_on_tuesday_eq_2500 :
  tomatoesOnTuesday 1000 300 200 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_on_tuesday_eq_2500_l3923_392352


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3923_392392

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3923_392392


namespace NUMINAMATH_CALUDE_polynomial_difference_simplification_l3923_392308

/-- The difference of two polynomials is equal to a simplified polynomial. -/
theorem polynomial_difference_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x^2 + 7) - 
  (x^6 + 4 * x^5 + 2 * x^4 - x^3 + x^2 + 8) = 
  x^6 - x^5 - x^4 + 6 * x^3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_simplification_l3923_392308


namespace NUMINAMATH_CALUDE_mary_sugar_amount_l3923_392310

/-- The amount of sugar required by the recipe in cups -/
def total_sugar : ℕ := 14

/-- The amount of sugar Mary still needs to add in cups -/
def sugar_to_add : ℕ := 12

/-- The amount of sugar Mary has already put in -/
def sugar_already_added : ℕ := total_sugar - sugar_to_add

theorem mary_sugar_amount : sugar_already_added = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_amount_l3923_392310


namespace NUMINAMATH_CALUDE_sum_of_digits_cd_l3923_392355

/-- c is an integer made up of a sequence of 2023 sixes -/
def c : ℕ := (6 : ℕ) * ((10 ^ 2023 - 1) / 9)

/-- d is an integer made up of a sequence of 2023 ones -/
def d : ℕ := (10 ^ 2023 - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits in cd is 12133 -/
theorem sum_of_digits_cd : sum_of_digits (c * d) = 12133 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_cd_l3923_392355


namespace NUMINAMATH_CALUDE_fourth_number_in_row_15_l3923_392309

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

theorem fourth_number_in_row_15 : pascal_triangle 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_row_15_l3923_392309


namespace NUMINAMATH_CALUDE_max_similar_triangle_lines_l3923_392338

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is outside a triangle --/
def IsOutside (P : Point) (T : Triangle) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (L : Line) (P : Point) : Prop := sorry

/-- Predicate to check if a line cuts off a similar triangle --/
def CutsSimilarTriangle (L : Line) (T : Triangle) : Prop := sorry

/-- The main theorem --/
theorem max_similar_triangle_lines 
  (T : Triangle) (P : Point) (h : IsOutside P T) :
  ∃ (S : Finset Line), 
    (∀ L ∈ S, PassesThrough L P ∧ CutsSimilarTriangle L T) ∧ 
    S.card = 6 ∧
    (∀ S' : Finset Line, 
      (∀ L ∈ S', PassesThrough L P ∧ CutsSimilarTriangle L T) → 
      S'.card ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_similar_triangle_lines_l3923_392338


namespace NUMINAMATH_CALUDE_dunkers_lineup_count_l3923_392325

theorem dunkers_lineup_count (n : ℕ) (k : ℕ) (a : ℕ) (z : ℕ) : 
  n = 15 → k = 5 → a ≠ z → a ≤ n → z ≤ n →
  (Nat.choose (n - 2) (k - 1) * 2 + Nat.choose (n - 2) k) = 2717 :=
by sorry

end NUMINAMATH_CALUDE_dunkers_lineup_count_l3923_392325


namespace NUMINAMATH_CALUDE_jerry_debt_payment_l3923_392357

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℕ) (remaining_debt : ℕ) (payment_two_months_ago : ℕ) 
  (h1 : total_debt = 50)
  (h2 : remaining_debt = 23)
  (h3 : payment_two_months_ago = 12)
  (h4 : total_debt > remaining_debt)
  (h5 : total_debt - remaining_debt > payment_two_months_ago) :
  ∃ (payment_last_month : ℕ), 
    payment_last_month - payment_two_months_ago = 3 ∧
    payment_last_month > payment_two_months_ago ∧
    payment_last_month + payment_two_months_ago = total_debt - remaining_debt :=
by
  sorry


end NUMINAMATH_CALUDE_jerry_debt_payment_l3923_392357


namespace NUMINAMATH_CALUDE_range_of_a_l3923_392343

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h_odd : is_odd f) 
  (h_domain : ∀ x ∈ Set.Ioo (-1) 1, f x ≠ 0) 
  (h_ineq : ∀ a : ℝ, f (1 - a) + f (2 * a - 1) < 0) :
  Set.Ioo 0 1 = {a : ℝ | 0 < a ∧ a < 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3923_392343


namespace NUMINAMATH_CALUDE_inequalities_hold_l3923_392391

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * b ≤ 1 / 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3923_392391


namespace NUMINAMATH_CALUDE_trig_roots_equation_l3923_392345

theorem trig_roots_equation (θ : ℝ) (a : ℝ) :
  (∀ x, x^2 - a*x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ) →
  Real.cos (θ - 3*Real.pi/2) + Real.sin (3*Real.pi/2 + θ) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_roots_equation_l3923_392345


namespace NUMINAMATH_CALUDE_min_value_expression_l3923_392389

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  ((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) ≥ 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3923_392389


namespace NUMINAMATH_CALUDE_function_identity_l3923_392361

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l3923_392361


namespace NUMINAMATH_CALUDE_reflection_segment_length_C_l3923_392366

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_segment_length (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from C(4, 3) to its reflection C' over the x-axis is 6 --/
theorem reflection_segment_length_C : reflection_segment_length 4 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_segment_length_C_l3923_392366


namespace NUMINAMATH_CALUDE_square_root_problem_l3923_392311

theorem square_root_problem (x y : ℝ) 
  (h1 : (x + 7) = 9) 
  (h2 : (2 * x - y - 13) = -8) : 
  Real.sqrt (5 * x - 6 * y) = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_problem_l3923_392311


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3923_392322

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + x + y^2 = 3*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3923_392322


namespace NUMINAMATH_CALUDE_valid_solutions_l3923_392376

/-- Defines a function that checks if a triple of digits forms a valid solution --/
def is_valid_solution (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  ∃ (k : ℤ), k * (10*a + b + 10*b + c + 10*c + a) = 100*a + 10*b + c + a + b + c

/-- The main theorem stating the valid solutions --/
theorem valid_solutions :
  ∀ a b c : ℕ,
    is_valid_solution a b c ↔
      (a = 5 ∧ b = 1 ∧ c = 6) ∨
      (a = 9 ∧ b = 1 ∧ c = 2) ∨
      (a = 6 ∧ b = 4 ∧ c = 5) ∨
      (a = 3 ∧ b = 7 ∧ c = 8) ∨
      (a = 5 ∧ b = 7 ∧ c = 6) ∨
      (a = 7 ∧ b = 7 ∧ c = 4) ∨
      (a = 9 ∧ b = 7 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_solutions_l3923_392376


namespace NUMINAMATH_CALUDE_divisibility_transitivity_l3923_392398

theorem divisibility_transitivity (m n k : ℕ+) 
  (h1 : m ^ n.val ∣ n ^ m.val) 
  (h2 : n ^ k.val ∣ k ^ n.val) : 
  m ^ k.val ∣ k ^ m.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_transitivity_l3923_392398


namespace NUMINAMATH_CALUDE_least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l3923_392318

theorem least_pennies_count (n : ℕ) : n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 → n ≥ 11 :=
by sorry

theorem eleven_satisfies_conditions : 11 % 5 = 1 ∧ 11 % 3 = 2 :=
by sorry

theorem least_pennies_is_eleven : ∃ (n : ℕ), n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 ∧ ∀ m : ℕ, (m > 0 ∧ m % 5 = 1 ∧ m % 3 = 2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l3923_392318


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3923_392339

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  base_angle₁ : ℕ
  base_angle₂ : ℕ
  vertex_angle : ℕ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : t.base_angle₁ = 50 ∨ t.vertex_angle = 50) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3923_392339


namespace NUMINAMATH_CALUDE_second_number_in_lcm_l3923_392378

theorem second_number_in_lcm (x : ℕ) : 
  Nat.lcm (Nat.lcm 24 x) 42 = 504 → x = 3 := by sorry

end NUMINAMATH_CALUDE_second_number_in_lcm_l3923_392378


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l3923_392363

theorem roots_product_plus_one (a b : ℝ) : 
  a^2 + 2*a - 2023 = 0 → 
  b^2 + 2*b - 2023 = 0 → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l3923_392363


namespace NUMINAMATH_CALUDE_simplify_fraction_l3923_392304

theorem simplify_fraction : (36 : ℚ) / 54 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3923_392304


namespace NUMINAMATH_CALUDE_school_travel_time_l3923_392367

/-- If a boy reaches school t minutes early when walking at 1.2 times his usual speed,
    his usual time to reach school is 6t minutes. -/
theorem school_travel_time (t : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0) 
    (h3 : usual_speed * usual_time = 1.2 * usual_speed * (usual_time - t)) : 
  usual_time = 6 * t := by
sorry

end NUMINAMATH_CALUDE_school_travel_time_l3923_392367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3923_392394

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean_2_6 : (a 2 + a 6) / 2 = 5)
  (h_mean_3_7 : (a 3 + a 7) / 2 = 7) :
  ∃ f : ℕ → ℝ, (∀ n, a n = f n) ∧ (∀ n, f n = 2 * n - 3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3923_392394


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_unique_l3923_392373

theorem quadratic_equation_magnitude_unique :
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_unique_l3923_392373


namespace NUMINAMATH_CALUDE_monkeys_for_48_bananas_l3923_392347

/-- Given that 8 monkeys can eat 8 bananas in some time, 
    this function calculates the number of monkeys needed to eat 48 bananas in 48 minutes -/
def monkeys_needed (initial_monkeys : ℕ) (initial_bananas : ℕ) (target_bananas : ℕ) : ℕ :=
  initial_monkeys * (target_bananas / initial_bananas)

/-- Theorem stating that 48 monkeys are needed to eat 48 bananas in 48 minutes -/
theorem monkeys_for_48_bananas : monkeys_needed 8 8 48 = 48 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_for_48_bananas_l3923_392347


namespace NUMINAMATH_CALUDE_alice_added_nineteen_plates_l3923_392385

/-- The number of plates Alice added before the tower fell -/
def additional_plates (initial : ℕ) (second_addition : ℕ) (total : ℕ) : ℕ :=
  total - (initial + second_addition)

/-- Theorem stating that Alice added 19 more plates before the tower fell -/
theorem alice_added_nineteen_plates : 
  additional_plates 27 37 83 = 19 := by
  sorry

end NUMINAMATH_CALUDE_alice_added_nineteen_plates_l3923_392385


namespace NUMINAMATH_CALUDE_twenty_three_percent_of_200_is_46_l3923_392315

theorem twenty_three_percent_of_200_is_46 : ∃ x : ℝ, (23 / 100) * x = 46 ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_percent_of_200_is_46_l3923_392315


namespace NUMINAMATH_CALUDE_odd_function_property_l3923_392344

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Property of the function f as given in the problem -/
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_prop : HasProperty f) :
  f (-4) > f (-6) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3923_392344


namespace NUMINAMATH_CALUDE_odd_number_induction_l3923_392328

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1)
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n ≥ 1 → Odd n → P n :=
sorry

end NUMINAMATH_CALUDE_odd_number_induction_l3923_392328


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l3923_392380

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ :=
  (n / 10000) * 3125 + ((n / 1000) % 10) * 625 + ((n / 100) % 10) * 125 + ((n / 10) % 10) * 25 + (n % 10) * 5

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base_conversion_subtraction :
  base5ToBase10 52143 - base8ToBase10 4310 = 1175 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l3923_392380


namespace NUMINAMATH_CALUDE_y_squared_eq_three_x_squared_plus_one_l3923_392371

/-- Sequence x defined recursively -/
def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

/-- Sequence y defined recursively -/
def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

/-- Main theorem: For all natural numbers n, y(n)² = 3x(n)² + 1 -/
theorem y_squared_eq_three_x_squared_plus_one (n : ℕ) : (y n)^2 = 3*(x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_eq_three_x_squared_plus_one_l3923_392371


namespace NUMINAMATH_CALUDE_least_possible_radios_l3923_392393

theorem least_possible_radios (n d : ℕ) (h1 : d > 0) : 
  (d + 8 * n - 16 - d = 72) → (∃ (m : ℕ), m ≥ n ∧ m ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_least_possible_radios_l3923_392393


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3923_392374

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x - 1 = 0) : 2*x^2 - 2*x + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3923_392374


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3923_392342

/-- The sum of the solutions to the quadratic equation x² - 6x - 8 = 2x + 18 is 8 -/
theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 8 - (2*x + 18)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3923_392342


namespace NUMINAMATH_CALUDE_four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l3923_392369

/-- A function that checks if it's possible to form a triangle with given number of lolly sticks -/
def can_form_triangle (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a + b + c = n ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that it's impossible to form a triangle with 4 lolly sticks -/
theorem four_sticks_impossible : ¬ can_form_triangle 4 :=
sorry

/-- Theorem stating that it's possible to form triangles with 3, 5, 6, and 7 lolly sticks -/
theorem other_sticks_possible :
  can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7 :=
sorry

/-- Main theorem combining the above results -/
theorem lolly_stick_triangle :
  ¬ can_form_triangle 4 ∧
  (can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7) :=
sorry

end NUMINAMATH_CALUDE_four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l3923_392369


namespace NUMINAMATH_CALUDE_exactly_two_approve_probability_l3923_392382

def approval_rate : ℝ := 0.6

def num_voters : ℕ := 4

def num_approving : ℕ := 2

def probability_exactly_two_approve (p : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approve_probability :
  probability_exactly_two_approve approval_rate num_voters num_approving = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_approve_probability_l3923_392382


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_incenters_form_rectangle_l3923_392306

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the Euclidean plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Predicate to check if a quadrilateral is inscribed in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Predicate to check if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

theorem inscribed_quadrilateral_incenters_form_rectangle 
  (ABCD : Quadrilateral) (c : Circle) :
  is_inscribed ABCD c →
  let I_A := incenter ABCD.B ABCD.C ABCD.D
  let I_B := incenter ABCD.C ABCD.D ABCD.A
  let I_C := incenter ABCD.D ABCD.A ABCD.B
  let I_D := incenter ABCD.A ABCD.B ABCD.C
  is_rectangle (Quadrilateral.mk I_A I_B I_C I_D) :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_incenters_form_rectangle_l3923_392306


namespace NUMINAMATH_CALUDE_park_distance_l3923_392372

theorem park_distance (d : ℝ) 
  (alice_false : ¬(d ≥ 8))
  (bob_false : ¬(d ≤ 7))
  (charlie_false : ¬(d ≤ 6)) :
  7 < d ∧ d < 8 := by
sorry

end NUMINAMATH_CALUDE_park_distance_l3923_392372


namespace NUMINAMATH_CALUDE_spinner_probability_l3923_392300

/-- Represents an equilateral triangle dissected by its altitudes -/
structure DissectedTriangle where
  regions : ℕ
  shaded_regions : ℕ

/-- The probability of a spinner landing in a shaded region -/
def landing_probability (t : DissectedTriangle) : ℚ :=
  t.shaded_regions / t.regions

/-- Theorem stating the probability of landing in a shaded region -/
theorem spinner_probability (t : DissectedTriangle) 
  (h1 : t.regions = 6)
  (h2 : t.shaded_regions = 3) : 
  landing_probability t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3923_392300


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l3923_392330

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (last_group : ℕ)
  (first_group_purchases : ℕ) (second_group_purchases : ℕ) (total_sales : ℕ) :
  total_customers = first_group + second_group + last_group →
  first_group = 4 →
  second_group = 12 →
  last_group = 4 →
  first_group_purchases = 2 →
  second_group_purchases = 1 →
  total_sales = 36 →
  ∃ (last_group_purchases : ℕ),
    total_sales = first_group * first_group_purchases +
                  second_group * second_group_purchases +
                  last_group * last_group_purchases ∧
    last_group_purchases = 4 :=
by sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l3923_392330


namespace NUMINAMATH_CALUDE_fraction_equality_l3923_392375

theorem fraction_equality (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^4 / n^5) * (n^4 / m^3) = m / n := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3923_392375


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3923_392351

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) 
  (h_monic : MonicQuarticPolynomial p)
  (h_neg_two : p (-2) = -4)
  (h_one : p 1 = -1)
  (h_three : p 3 = -9)
  (h_five : p 5 = -25) :
  p 0 = -30 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3923_392351


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3923_392327

/-- Given a rectangular metallic sheet with length 48 meters, 
    from which squares of side 8 meters are cut from each corner to form a box,
    if the resulting box has a volume of 5632 cubic meters,
    then the width of the original sheet is 38 meters. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ), 
    w > 0 →
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5632 →
    w = 38 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l3923_392327


namespace NUMINAMATH_CALUDE_stating_sum_of_digits_special_product_l3923_392388

/-- 
Represents the product of numbers of the form (10^k - 1) where k is a power of 2 up to 2^n.
-/
def specialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (10^(2^i) - 1)) 9

/-- 
Represents the sum of digits of a natural number in decimal notation.
-/
def sumOfDigits (m : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the sum of digits of the special product is equal to 9 · 2^n.
-/
theorem sum_of_digits_special_product (n : ℕ) : 
  sumOfDigits (specialProduct n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_stating_sum_of_digits_special_product_l3923_392388


namespace NUMINAMATH_CALUDE_infinitely_many_factorizable_numbers_l3923_392341

theorem infinitely_many_factorizable_numbers :
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧
    ∃ a b : ℕ, 
      (n^3 + 4*n + 505 : ℤ) = (a * b : ℤ) ∧
      a > n.sqrt ∧
      b > n.sqrt :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_factorizable_numbers_l3923_392341


namespace NUMINAMATH_CALUDE_carlos_july_reading_l3923_392383

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- The total number of books Carlos needed to read -/
def total_books : ℕ := 100

/-- The number of books Carlos read in July -/
def july_books : ℕ := total_books - (june_books + august_books)

theorem carlos_july_reading :
  july_books = 28 := by sorry

end NUMINAMATH_CALUDE_carlos_july_reading_l3923_392383


namespace NUMINAMATH_CALUDE_paths_count_is_40_l3923_392377

/-- Represents the arrangement of letters and numerals --/
structure Arrangement where
  centralA : Unit
  adjacentM : Fin 4
  adjacentC : Fin 4 → Fin 3
  adjacent1 : Unit
  adjacent0 : Fin 2

/-- Counts the number of paths to spell AMC10 in the given arrangement --/
def countPaths (arr : Arrangement) : ℕ :=
  let pathsFromM (m : Fin 4) := arr.adjacentC m * 1 * 2
  (pathsFromM 0 + pathsFromM 1 + pathsFromM 2 + pathsFromM 3)

/-- The theorem stating that the number of paths is 40 --/
theorem paths_count_is_40 (arr : Arrangement) : countPaths arr = 40 := by
  sorry

#check paths_count_is_40

end NUMINAMATH_CALUDE_paths_count_is_40_l3923_392377


namespace NUMINAMATH_CALUDE_teal_survey_l3923_392359

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_green = 90)
  (h3 : both = 40)
  (h4 : neither = 20) :
  ∃ more_blue : ℕ, more_blue = 80 ∧ 
    total = more_green + more_blue - both + neither :=
by sorry

end NUMINAMATH_CALUDE_teal_survey_l3923_392359
