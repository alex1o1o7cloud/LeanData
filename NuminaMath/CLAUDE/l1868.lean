import Mathlib

namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1868_186822

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1868_186822


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1868_186863

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1868_186863


namespace NUMINAMATH_CALUDE_fourth_root_of_207360000_l1868_186888

theorem fourth_root_of_207360000 : (207360000 : ℝ) ^ (1/4 : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_207360000_l1868_186888


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l1868_186853

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) ≥ 2 :=
by sorry

theorem min_value_equality :
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + (1 / ((1 + 0) * (1 + 0) * (1 + 0))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l1868_186853


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1868_186880

theorem sphere_volume_from_surface_area (S : ℝ) (h : S = 36 * Real.pi) :
  (4 / 3 : ℝ) * Real.pi * ((S / (4 * Real.pi)) ^ (3 / 2 : ℝ)) = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1868_186880


namespace NUMINAMATH_CALUDE_floor_length_is_20_l1868_186831

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- Theorem stating the length of the floor under given conditions. -/
theorem floor_length_is_20 (floor : RectangularFloor)
  (h1 : floor.length = 3 * floor.breadth)
  (h2 : floor.paintingCost = 400)
  (h3 : floor.paintingRate = 3)
  (h4 : floor.paintingCost / floor.paintingRate = floor.length * floor.breadth) :
  floor.length = 20 := by
  sorry


end NUMINAMATH_CALUDE_floor_length_is_20_l1868_186831


namespace NUMINAMATH_CALUDE_x₄_x₁_diff_l1868_186891

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = -f (200 - x)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- The x-intercepts are in increasing order
axiom x_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

-- The difference between x₃ and x₂
axiom x₃_x₂_diff : x₃ - x₂ = 200

-- The vertex of g is on the graph of f
axiom vertex_on_f : ∃ x, g x = f x ∧ ∀ y, g y ≤ g x

-- Theorem to prove
theorem x₄_x₁_diff : x₄ - x₁ = 1000 + 800 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_x₄_x₁_diff_l1868_186891


namespace NUMINAMATH_CALUDE_pool_capacity_pool_capacity_is_2000_liters_l1868_186859

theorem pool_capacity (water_loss_per_jump : ℝ) (cleaning_threshold : ℝ) (jumps_before_cleaning : ℕ) : ℝ :=
  let total_water_loss := water_loss_per_jump * jumps_before_cleaning
  let water_loss_percentage := 1 - cleaning_threshold
  total_water_loss / water_loss_percentage

#check pool_capacity 0.4 0.8 1000 = 2000

theorem pool_capacity_is_2000_liters :
  pool_capacity 0.4 0.8 1000 = 2000 := by sorry

end NUMINAMATH_CALUDE_pool_capacity_pool_capacity_is_2000_liters_l1868_186859


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1868_186836

theorem product_remainder_mod_five : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1868_186836


namespace NUMINAMATH_CALUDE_near_integer_intervals_l1868_186826

-- Definition of "near-integer interval"
def near_integer_interval (T : ℝ) : Set ℝ :=
  {x | ∃ (m n : ℤ), m < T ∧ T < n ∧ x ∈ Set.Ioo (↑m : ℝ) (↑n : ℝ) ∧
    ∀ (k : ℤ), k ≤ m ∨ n ≤ k}

-- Theorem statement
theorem near_integer_intervals :
  (near_integer_interval (Real.sqrt 5) = Set.Ioo 2 3) ∧
  (near_integer_interval (-Real.sqrt 10) = Set.Ioo (-4) (-3)) ∧
  (∀ (x y : ℝ), y = Real.sqrt (x - 2023) + Real.sqrt (2023 - x) →
    near_integer_interval (Real.sqrt (x + y)) = Set.Ioo 44 45) :=
by sorry

end NUMINAMATH_CALUDE_near_integer_intervals_l1868_186826


namespace NUMINAMATH_CALUDE_absolute_value_equation_one_root_l1868_186816

theorem absolute_value_equation_one_root :
  ∃! x : ℝ, (abs x - 4 / x = 3 * abs x / x) ∧ (x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_one_root_l1868_186816


namespace NUMINAMATH_CALUDE_terminal_side_in_first_or_third_quadrant_l1868_186882

-- Define the angle α as a function of k
def α (k : ℤ) : Real := k * 180 + 45

-- Define a function to determine the quadrant of an angle
def inFirstOrThirdQuadrant (angle : Real) : Prop :=
  (0 < angle % 360 ∧ angle % 360 < 90) ∨ 
  (180 < angle % 360 ∧ angle % 360 < 270)

-- Theorem statement
theorem terminal_side_in_first_or_third_quadrant (k : ℤ) :
  inFirstOrThirdQuadrant (α k) := by sorry

end NUMINAMATH_CALUDE_terminal_side_in_first_or_third_quadrant_l1868_186882


namespace NUMINAMATH_CALUDE_larger_number_problem_l1868_186870

theorem larger_number_problem (x y : ℤ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1868_186870


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1868_186881

theorem quadratic_roots_difference_squared : 
  ∀ Θ θ : ℝ, 
  (Θ^2 - 3*Θ + 1 = 0) → 
  (θ^2 - 3*θ + 1 = 0) → 
  (Θ ≠ θ) → 
  (Θ - θ)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1868_186881


namespace NUMINAMATH_CALUDE_factorization_identity_l1868_186873

theorem factorization_identity (m : ℝ) : m^2 + 3*m = m*(m + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l1868_186873


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l1868_186801

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 336 →
  total_cost = 42 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l1868_186801


namespace NUMINAMATH_CALUDE_sequence_formulas_l1868_186814

/-- Given a geometric sequence {a_n} and another sequence {b_n}, prove the formulas for a_n and b_n -/
theorem sequence_formulas (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence condition
  (a 1 = 4) →  -- initial condition for a_n
  (2 * a 2 + a 3 = 60) →  -- additional condition for a_n
  (∀ n, b (n + 1) = b n + a n) →  -- recurrence relation for b_n
  (b 1 = a 2) →  -- initial condition for b_n
  (b 1 > 0) →  -- positivity condition for b_1
  (∀ n, a n = 4 * 3^(n - 1)) ∧ 
  (∀ n, b n = 2 * 3^n + 10) := by
sorry


end NUMINAMATH_CALUDE_sequence_formulas_l1868_186814


namespace NUMINAMATH_CALUDE_second_batch_weight_is_100_l1868_186819

-- Define the initial stock
def initial_stock : ℝ := 400

-- Define the percentage of decaf in initial stock
def initial_decaf_percent : ℝ := 0.20

-- Define the percentage of decaf in the second batch
def second_batch_decaf_percent : ℝ := 0.50

-- Define the final percentage of decaf in total stock
def final_decaf_percent : ℝ := 0.26

-- Define the weight of the second batch as a variable
variable (second_batch_weight : ℝ)

-- Theorem statement
theorem second_batch_weight_is_100 :
  (initial_stock * initial_decaf_percent + second_batch_weight * second_batch_decaf_percent) / 
  (initial_stock + second_batch_weight) = final_decaf_percent →
  second_batch_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_batch_weight_is_100_l1868_186819


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_l1868_186811

/-- The equation has infinitely many solutions if and only if c = 5 -/
theorem infinite_solutions_iff_c_eq_five :
  (∃ c : ℝ, ∀ x : ℝ, 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_l1868_186811


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1868_186858

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {9}) ∧ (a = -3) ∧ (A a ∪ B a = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1868_186858


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_31_17_l1868_186890

theorem smallest_fraction_greater_than_31_17 :
  ∀ a b : ℤ, b < 17 → (a : ℚ) / b > 31 / 17 → 11 / 6 ≤ (a : ℚ) / b :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_31_17_l1868_186890


namespace NUMINAMATH_CALUDE_specific_seating_arrangements_l1868_186848

/-- Represents the seating arrangement in a theater -/
structure TheaterSeating where
  front_row : ℕ
  back_row : ℕ
  unusable_middle_seats : ℕ

/-- Calculates the number of ways to seat two people in the theater -/
def seating_arrangements (theater : TheaterSeating) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem specific_seating_arrangements :
  let theater : TheaterSeating := {
    front_row := 10,
    back_row := 11,
    unusable_middle_seats := 3
  }
  seating_arrangements theater = 276 := by
  sorry

end NUMINAMATH_CALUDE_specific_seating_arrangements_l1868_186848


namespace NUMINAMATH_CALUDE_three_sequence_non_decreasing_indices_l1868_186838

theorem three_sequence_non_decreasing_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end NUMINAMATH_CALUDE_three_sequence_non_decreasing_indices_l1868_186838


namespace NUMINAMATH_CALUDE_bingley_has_four_bracelets_l1868_186860

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingley_remaining_bracelets : ℕ :=
  let bingley_initial : ℕ := 5
  let kelly_initial : ℕ := 16
  let kelly_gives : ℕ := kelly_initial / 4 / 3
  let bingley_after_receiving : ℕ := bingley_initial + kelly_gives
  let bingley_gives : ℕ := bingley_after_receiving / 3
  bingley_after_receiving - bingley_gives

/-- Theorem stating that Bingley has 4 bracelets remaining -/
theorem bingley_has_four_bracelets : bingley_remaining_bracelets = 4 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_four_bracelets_l1868_186860


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l1868_186885

theorem solve_percentage_equation : ∃ x : ℝ, 0.65 * x = 0.20 * 487.50 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l1868_186885


namespace NUMINAMATH_CALUDE_inequality_range_l1868_186884

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) → 
  -1/2 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1868_186884


namespace NUMINAMATH_CALUDE_transformed_roots_equation_l1868_186841

/-- Given that a, b, c, and d are the solutions of x^4 + 2x^3 - 5 = 0,
    prove that abc/d, abd/c, acd/b, and bcd/a are the solutions of the same equation. -/
theorem transformed_roots_equation (a b c d : ℂ) : 
  (a^4 + 2*a^3 - 5 = 0) ∧ 
  (b^4 + 2*b^3 - 5 = 0) ∧ 
  (c^4 + 2*c^3 - 5 = 0) ∧ 
  (d^4 + 2*d^3 - 5 = 0) →
  ((a*b*c/d)^4 + 2*(a*b*c/d)^3 - 5 = 0) ∧
  ((a*b*d/c)^4 + 2*(a*b*d/c)^3 - 5 = 0) ∧
  ((a*c*d/b)^4 + 2*(a*c*d/b)^3 - 5 = 0) ∧
  ((b*c*d/a)^4 + 2*(b*c*d/a)^3 - 5 = 0) := by
  sorry


end NUMINAMATH_CALUDE_transformed_roots_equation_l1868_186841


namespace NUMINAMATH_CALUDE_remainder_19_pow_60_mod_7_l1868_186832

theorem remainder_19_pow_60_mod_7 : 19^60 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_19_pow_60_mod_7_l1868_186832


namespace NUMINAMATH_CALUDE_exactly_four_intersections_l1868_186800

-- Define the graphs
def graph1 (B : ℝ) (x y : ℝ) : Prop := y = B * x^2
def graph2 (x y : ℝ) : Prop := y^2 + 2 * x^2 = 5 + 6 * y

-- Define an intersection point
def is_intersection (B : ℝ) (x y : ℝ) : Prop :=
  graph1 B x y ∧ graph2 x y

-- Theorem statement
theorem exactly_four_intersections (B : ℝ) (h : B > 0) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    is_intersection B x₁ y₁ ∧
    is_intersection B x₂ y₂ ∧
    is_intersection B x₃ y₃ ∧
    is_intersection B x₄ y₄ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧
    (x₁ ≠ x₄ ∨ y₁ ≠ y₄) ∧
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₂ ≠ x₄ ∨ y₂ ≠ y₄) ∧
    (x₃ ≠ x₄ ∨ y₃ ≠ y₄) ∧
    ∀ (x y : ℝ), is_intersection B x y →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨
       (x = x₃ ∧ y = y₃) ∨ (x = x₄ ∧ y = y₄)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_four_intersections_l1868_186800


namespace NUMINAMATH_CALUDE_tournament_handshakes_correct_l1868_186850

/-- The number of handshakes in a tennis tournament with 3 teams of 2 players each --/
def tournament_handshakes : ℕ := 12

/-- The number of teams in the tournament --/
def num_teams : ℕ := 3

/-- The number of players per team --/
def players_per_team : ℕ := 2

/-- The total number of players in the tournament --/
def total_players : ℕ := num_teams * players_per_team

/-- The number of handshakes per player --/
def handshakes_per_player : ℕ := total_players - 2

theorem tournament_handshakes_correct :
  tournament_handshakes = (total_players * handshakes_per_player) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tournament_handshakes_correct_l1868_186850


namespace NUMINAMATH_CALUDE_sqrt_221_between_15_and_16_l1868_186867

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_between_15_and_16_l1868_186867


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1868_186802

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 24 * π → π * r^2 = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1868_186802


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1868_186829

-- Problem 1
theorem inequality_one (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by sorry

-- Problem 2
theorem inequality_two (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) :
  |1 - a * b| > |a - b| := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1868_186829


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1868_186837

/-- 
Given a geometric series with first term a and common ratio r,
prove that if the sum of the series is 24 and the sum of terms
with odd powers of r is 10, then r = 5/7.
-/
theorem geometric_series_ratio (a r : ℝ) : 
  (∑' n, a * r^n) = 24 →
  (∑' n, a * r^(2*n+1)) = 10 →
  r = 5/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1868_186837


namespace NUMINAMATH_CALUDE_cat_bird_hunting_l1868_186877

theorem cat_bird_hunting (day_catch : ℕ) (night_catch : ℕ) : 
  day_catch = 8 → night_catch = 2 * day_catch → day_catch + night_catch = 24 := by
  sorry

end NUMINAMATH_CALUDE_cat_bird_hunting_l1868_186877


namespace NUMINAMATH_CALUDE_subtracted_number_l1868_186862

theorem subtracted_number (x : ℝ) : x = 7 → 4 * 5.0 - x = 13 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1868_186862


namespace NUMINAMATH_CALUDE_probability_specific_pair_from_six_l1868_186876

/-- The probability of selecting a specific pair when choosing 2 from 6 -/
theorem probability_specific_pair_from_six (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (1 : ℚ) / (n.choose k) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_pair_from_six_l1868_186876


namespace NUMINAMATH_CALUDE_room_width_calculation_l1868_186824

/-- Proves that given a rectangular room with a length of 5.5 meters, where the cost of paving the floor at a rate of 800 Rs/m² is 17,600 Rs, the width of the room is 4 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 17600 →
  cost_per_sqm = 800 →
  total_cost / cost_per_sqm / length = 4 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1868_186824


namespace NUMINAMATH_CALUDE_mollys_age_l1868_186815

/-- Given the ratio of Sandy's age to Molly's age and Sandy's future age, 
    prove Molly's current age -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l1868_186815


namespace NUMINAMATH_CALUDE_negation_equivalence_l1868_186849

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1868_186849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1868_186892

/-- Given an arithmetic sequence with sum S_n = a n^2, prove a_5/d = 9/2 -/
theorem arithmetic_sequence_ratio (a : ℝ) (d : ℝ) (S : ℕ → ℝ) (a_seq : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, S n = a * n^2) →
  (∀ n : ℕ, a_seq (n + 1) - a_seq n = d) →
  (∀ n : ℕ, S n = (n * (a_seq 1 + a_seq n)) / 2) →
  a_seq 5 / d = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1868_186892


namespace NUMINAMATH_CALUDE_jackies_tree_climbing_ratio_l1868_186844

/-- Given the following conditions about Jackie's tree climbing:
  - Jackie climbed 4 trees in total
  - The first tree is 1000 feet tall
  - Two trees are of equal height
  - The fourth tree is 200 feet taller than the first tree
  - The average height of all trees is 800 feet

  Prove that the ratio of the height of the two equal trees to the height of the first tree is 1:2.
-/
theorem jackies_tree_climbing_ratio :
  ∀ (h₁ h₂ h₄ : ℝ),
  h₁ = 1000 →
  h₄ = h₁ + 200 →
  (h₁ + 2 * h₂ + h₄) / 4 = 800 →
  h₂ / h₁ = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_jackies_tree_climbing_ratio_l1868_186844


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l1868_186871

theorem average_cost_before_gratuity
  (num_individuals : ℕ)
  (total_bill_with_gratuity : ℚ)
  (gratuity_rate : ℚ)
  (h1 : num_individuals = 9)
  (h2 : total_bill_with_gratuity = 756)
  (h3 : gratuity_rate = 1/5) :
  (total_bill_with_gratuity / (1 + gratuity_rate)) / num_individuals = 70 := by
sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l1868_186871


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1868_186887

theorem sin_cos_identity : 
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) - 
  Real.sin (40 * π / 180) * Real.cos (70 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1868_186887


namespace NUMINAMATH_CALUDE_james_training_hours_l1868_186861

/-- James' Olympic training schedule and yearly hours --/
theorem james_training_hours :
  (sessions_per_day : ℕ) →
  (hours_per_session : ℕ) →
  (training_days_per_week : ℕ) →
  (weeks_per_year : ℕ) →
  sessions_per_day = 2 →
  hours_per_session = 4 →
  training_days_per_week = 5 →
  weeks_per_year = 52 →
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year = 2080 :=
by sorry

end NUMINAMATH_CALUDE_james_training_hours_l1868_186861


namespace NUMINAMATH_CALUDE_initial_oranges_count_l1868_186883

/-- Proves that the initial number of oranges in a bin was 50, given the described changes and final count. -/
theorem initial_oranges_count (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 40)
  (h2 : added = 24)
  (h3 : final = 34)
  (h4 : initial - thrown_away + added = final) : initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l1868_186883


namespace NUMINAMATH_CALUDE_max_marks_proof_l1868_186878

/-- Given a passing threshold, actual score, and shortfall, calculates the maximum possible marks -/
def calculate_max_marks (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) : ℚ :=
  (actual_score + shortfall : ℚ) / passing_threshold

/-- Proves that the maximum marks is 617.5 given the problem conditions -/
theorem max_marks_proof (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) 
    (h1 : passing_threshold = 0.4)
    (h2 : actual_score = 212)
    (h3 : shortfall = 35) :
  calculate_max_marks passing_threshold actual_score shortfall = 617.5 := by
  sorry

#eval calculate_max_marks 0.4 212 35

end NUMINAMATH_CALUDE_max_marks_proof_l1868_186878


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l1868_186825

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (a b : ℕ), n = 12 * a + 6 * b

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l1868_186825


namespace NUMINAMATH_CALUDE_concatenated_numbers_problem_l1868_186899

theorem concatenated_numbers_problem : 
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x < 1000 ∧ 
    100 ≤ y ∧ y < 1000 ∧ 
    1000 * x + y = 7 * x * y ∧
    x = 143 ∧ y = 143 := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_problem_l1868_186899


namespace NUMINAMATH_CALUDE_cubic_inequality_l1868_186817

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1868_186817


namespace NUMINAMATH_CALUDE_item_sale_ratio_l1868_186897

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l1868_186897


namespace NUMINAMATH_CALUDE_square_root_sum_fractions_l1868_186804

theorem square_root_sum_fractions : 
  Real.sqrt (1/25 + 1/36 + 1/49) = Real.sqrt 7778 / 297 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_fractions_l1868_186804


namespace NUMINAMATH_CALUDE_five_and_half_hours_in_seconds_l1868_186854

/-- Converts hours to seconds -/
def hours_to_seconds (hours : ℝ) : ℝ :=
  hours * 60 * 60

/-- Theorem: 5.5 hours is equal to 19800 seconds -/
theorem five_and_half_hours_in_seconds : 
  hours_to_seconds 5.5 = 19800 := by sorry

end NUMINAMATH_CALUDE_five_and_half_hours_in_seconds_l1868_186854


namespace NUMINAMATH_CALUDE_correct_calculation_l1868_186874

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1868_186874


namespace NUMINAMATH_CALUDE_catch_up_equation_l1868_186851

/-- The number of days it takes for a good horse to catch up with a slow horse -/
def catch_up_days (good_horse_speed slow_horse_speed : ℕ) (head_start : ℕ) : ℕ → Prop :=
  λ x => good_horse_speed * x = slow_horse_speed * x + slow_horse_speed * head_start

/-- Theorem stating the equation for the number of days it takes for the good horse to catch up -/
theorem catch_up_equation :
  let good_horse_speed := 240
  let slow_horse_speed := 150
  let head_start := 12
  ∃ x : ℕ, catch_up_days good_horse_speed slow_horse_speed head_start x :=
by
  sorry

end NUMINAMATH_CALUDE_catch_up_equation_l1868_186851


namespace NUMINAMATH_CALUDE_circus_acrobats_l1868_186875

/-- Represents the number of acrobats in the circus show -/
def acrobats : ℕ := 11

/-- Represents the number of elephants in the circus show -/
def elephants : ℕ := 4

/-- Represents the number of clowns in the circus show -/
def clowns : ℕ := 10

/-- The total number of legs in the circus show -/
def total_legs : ℕ := 58

/-- The total number of heads in the circus show -/
def total_heads : ℕ := 25

/-- Theorem stating that the number of acrobats is 11 given the conditions of the circus show -/
theorem circus_acrobats :
  (2 * acrobats + 4 * elephants + 2 * clowns = total_legs) ∧
  (acrobats + elephants + clowns = total_heads) ∧
  (acrobats = 11) := by
  sorry

end NUMINAMATH_CALUDE_circus_acrobats_l1868_186875


namespace NUMINAMATH_CALUDE_rhombus_area_l1868_186869

/-- A rhombus with diagonals of lengths 10 and 30 has an area of 150 -/
theorem rhombus_area (d₁ d₂ area : ℝ) (h₁ : d₁ = 10) (h₂ : d₂ = 30) 
    (h₃ : area = (d₁ * d₂) / 2) : area = 150 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1868_186869


namespace NUMINAMATH_CALUDE_range_of_m_l1868_186807

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the property that ¬p is sufficient but not necessary for ¬q
def neg_p_sufficient_not_necessary_for_neg_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(q x m) ∧ p x

-- Theorem statement
theorem range_of_m :
  ∀ m, neg_p_sufficient_not_necessary_for_neg_q m ↔ -3 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1868_186807


namespace NUMINAMATH_CALUDE_lemon_pie_angle_l1868_186845

theorem lemon_pie_angle (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h_total : total = 40)
  (h_chocolate : chocolate = 15)
  (h_apple : apple = 10)
  (h_blueberry : blueberry = 5)
  (h_remaining : total - (chocolate + apple + blueberry) = 2 * (total - (chocolate + apple + blueberry)) / 2) :
  (((total - (chocolate + apple + blueberry)) / 2) : ℚ) / total * 360 = 45 := by
sorry

end NUMINAMATH_CALUDE_lemon_pie_angle_l1868_186845


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1868_186812

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1868_186812


namespace NUMINAMATH_CALUDE_lewis_total_earnings_l1868_186821

/-- Lewis's weekly earnings in dollars -/
def weekly_earnings : ℕ := 92

/-- Number of weeks Lewis works during the harvest -/
def weeks_worked : ℕ := 5

/-- Theorem stating Lewis's total earnings during the harvest -/
theorem lewis_total_earnings : weekly_earnings * weeks_worked = 460 := by
  sorry

end NUMINAMATH_CALUDE_lewis_total_earnings_l1868_186821


namespace NUMINAMATH_CALUDE_divide_by_seven_l1868_186823

theorem divide_by_seven (x : ℚ) (h : x = 5/2) : x / 7 = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_seven_l1868_186823


namespace NUMINAMATH_CALUDE_ratio_m_n_l1868_186833

theorem ratio_m_n (m n : ℕ) (h1 : m > n) (h2 : ¬(n ∣ m)) 
  (h3 : m % n = (m + n) % (m - n)) : m / n = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_m_n_l1868_186833


namespace NUMINAMATH_CALUDE_susan_drinks_eight_l1868_186857

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℚ := 2

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℚ := 2 * paul_bottles + 3

/-- The number of juice bottles Susan drinks per day -/
def susan_bottles : ℚ := 1.5 * donald_bottles - 2.5

/-- Theorem stating that Susan drinks 8 bottles of juice per day -/
theorem susan_drinks_eight : susan_bottles = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_drinks_eight_l1868_186857


namespace NUMINAMATH_CALUDE_point_coordinates_l1868_186803

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If a point M is in the first quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 2, then its coordinates are (2, 3) -/
theorem point_coordinates (M : Point) 
  (h1 : inFirstQuadrant M) 
  (h2 : distToXAxis M = 3) 
  (h3 : distToYAxis M = 2) : 
  M.x = 2 ∧ M.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1868_186803


namespace NUMINAMATH_CALUDE_right_triangle_leg_identity_l1868_186813

theorem right_triangle_leg_identity (a b : ℝ) : 2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_identity_l1868_186813


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1868_186852

theorem square_difference_divided_by_nine : (108^2 - 99^2) / 9 = 207 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1868_186852


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1868_186809

/-- Given an ellipse with equation x²/16 + y²/9 = 1, the distance between its foci is 2√7. -/
theorem ellipse_foci_distance :
  ∀ (F₁ F₂ : ℝ × ℝ),
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 4 * (4 + 3)) →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1868_186809


namespace NUMINAMATH_CALUDE_min_C_for_inequality_l1868_186846

/-- The minimum value of C that satisfies the given inequality for all x and any α, β where |α| ≤ 1 and |β| ≤ 1 -/
theorem min_C_for_inequality : 
  (∃ (C : ℝ), ∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) ∧ 
  (∀ (C : ℝ), (∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) → C ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_C_for_inequality_l1868_186846


namespace NUMINAMATH_CALUDE_no_x_squared_term_l1868_186855

theorem no_x_squared_term (m : ℚ) : 
  (∀ x, (x + 1) * (x^2 + 5*m*x + 3) = x^3 + (3 + 5*m)*x + 3) → m = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l1868_186855


namespace NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l1868_186810

/-- The number of cards Janessa keeps for herself given the initial conditions -/
def janessas_kept_cards (initial_cards : ℕ) (father_cards : ℕ) (ebay_cards : ℕ) (bad_cards : ℕ) (given_cards : ℕ) : ℕ :=
  initial_cards + father_cards + ebay_cards - bad_cards - given_cards

/-- Theorem stating that Janessa keeps 20 cards for herself -/
theorem janessa_keeps_twenty_cards :
  janessas_kept_cards 4 13 36 4 29 = 20 := by sorry

end NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l1868_186810


namespace NUMINAMATH_CALUDE_bank_deposit_years_l1868_186843

/-- Proves that the number of years for the second bank deposit is 5 given the problem conditions. -/
theorem bank_deposit_years (principal : ℚ) (rate : ℚ) (years1 : ℚ) (interest_diff : ℚ) 
  (h1 : principal = 640)
  (h2 : rate = 15 / 100)
  (h3 : years1 = 7 / 2)
  (h4 : interest_diff = 144) :
  ∃ (years2 : ℚ), 
    principal * rate * years2 - principal * rate * years1 = interest_diff ∧ 
    years2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_bank_deposit_years_l1868_186843


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l1868_186805

-- Define the room dimensions and total cost
def roomLength : Real := 6.5
def roomWidth : Real := 2.75
def totalCost : Real := 10725

-- Define the theorem
theorem paving_rate_calculation :
  let area := roomLength * roomWidth
  let ratePerSqMetre := totalCost / area
  ratePerSqMetre = 600 := by
  sorry


end NUMINAMATH_CALUDE_paving_rate_calculation_l1868_186805


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1868_186827

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1868_186827


namespace NUMINAMATH_CALUDE_square_area_ratio_l1868_186898

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((4*y)^2) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1868_186898


namespace NUMINAMATH_CALUDE_fifth_term_value_l1868_186842

theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n^2 + 3 * n - 1) :
  a 5 = 21 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1868_186842


namespace NUMINAMATH_CALUDE_ball_collection_theorem_l1868_186889

theorem ball_collection_theorem (r b y : ℕ) : 
  b + y = 9 →
  r + y = 5 →
  r + b = 6 →
  r + b + y = 10 := by
sorry

end NUMINAMATH_CALUDE_ball_collection_theorem_l1868_186889


namespace NUMINAMATH_CALUDE_isosceles_triangle_50_largest_angle_l1868_186864

/-- An isosceles triangle with one angle opposite an equal side measuring 50 degrees -/
structure IsoscelesTriangle50 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- Assertion that the angle opposite an equal side is 50 degrees -/
  h_angle_50 : angle_opposite_equal_side = 50

/-- 
Theorem: In an isosceles triangle where one of the angles opposite an equal side 
measures 50°, the largest angle measures 80°.
-/
theorem isosceles_triangle_50_largest_angle 
  (t : IsoscelesTriangle50) : t.largest_angle = 80 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_50_largest_angle_l1868_186864


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l1868_186830

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l1868_186830


namespace NUMINAMATH_CALUDE_brownies_in_container_l1868_186847

/-- Represents the problem of calculating the fraction of remaining brownies in the container --/
theorem brownies_in_container (batches : ℕ) (brownies_per_batch : ℕ) 
  (bake_sale_fraction : ℚ) (given_out : ℕ) : 
  batches = 10 →
  brownies_per_batch = 20 →
  bake_sale_fraction = 3/4 →
  given_out = 20 →
  let total_brownies := batches * brownies_per_batch
  let bake_sale_brownies := (bake_sale_fraction * brownies_per_batch) * batches
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies
  let remaining_after_given_out := remaining_after_bake_sale - given_out
  (remaining_after_given_out : ℚ) / remaining_after_bake_sale = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_in_container_l1868_186847


namespace NUMINAMATH_CALUDE_puppies_ratio_l1868_186806

/-- Puppies problem -/
theorem puppies_ratio (total : ℕ) (kept : ℕ) (price : ℕ) (stud_fee : ℕ) (profit : ℕ) :
  total = 8 →
  kept = 1 →
  price = 600 →
  stud_fee = 300 →
  profit = 1500 →
  (total - kept - (profit + stud_fee) / price : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_ratio_l1868_186806


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1868_186872

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  sorry

/-- The specific linear function y = 2x + 1 -/
def f : LinearFunction :=
  { slope := 2, yIntercept := 1 }

theorem linear_function_not_in_fourth_quadrant :
  ¬ passesThrough f Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1868_186872


namespace NUMINAMATH_CALUDE_old_man_gold_coins_l1868_186835

theorem old_man_gold_coins (x y : ℕ) (h : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_old_man_gold_coins_l1868_186835


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1868_186856

theorem expand_and_simplify (x : ℝ) : (2*x - 1)^2 - x*(4*x - 1) = -3*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1868_186856


namespace NUMINAMATH_CALUDE_max_advancing_teams_for_specific_tournament_l1868_186808

/-- Represents a football tournament with specified rules --/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum number of teams that can advance in the tournament --/
def max_advancing_teams (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of advancing teams for the specific tournament --/
theorem max_advancing_teams_for_specific_tournament :
  let tournament : FootballTournament := {
    num_teams := 7,
    min_points_to_advance := 12,
    points_for_win := 3,
    points_for_draw := 1,
    points_for_loss := 0
  }
  max_advancing_teams tournament = 5 := by sorry

end NUMINAMATH_CALUDE_max_advancing_teams_for_specific_tournament_l1868_186808


namespace NUMINAMATH_CALUDE_system_solution_l1868_186895

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2 - 5*x*y + 6*y^2 = 0
def equation2 (x y : ℝ) : Prop := x^2 + y^2 + x - 11*y - 2 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-2/5, -1/5), (4, 2), (-3/5, -1/5), (3, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1868_186895


namespace NUMINAMATH_CALUDE_max_time_between_happy_moments_l1868_186818

/-- A happy moment on a 24-hour digital clock --/
structure HappyMoment where
  hours : Fin 24
  minutes : Fin 60
  is_happy : (hours = 6 * minutes) ∨ (minutes = 6 * hours)

/-- The time difference between two happy moments in minutes --/
def time_difference (h1 h2 : HappyMoment) : ℕ :=
  let total_minutes1 := h1.hours * 60 + h1.minutes
  let total_minutes2 := h2.hours * 60 + h2.minutes
  if total_minutes2 ≥ total_minutes1 then
    total_minutes2 - total_minutes1
  else
    (24 * 60) - (total_minutes1 - total_minutes2)

/-- Theorem stating the maximum time difference between consecutive happy moments --/
theorem max_time_between_happy_moments :
  ∃ (max : ℕ), max = 361 ∧
  ∀ (h1 h2 : HappyMoment), time_difference h1 h2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_time_between_happy_moments_l1868_186818


namespace NUMINAMATH_CALUDE_sum_pqrs_equals_32_1_l1868_186868

theorem sum_pqrs_equals_32_1 
  (p q r s : ℝ)
  (hp : p = 2)
  (hpq : p * q = 20)
  (hpqr : p * q * r = 202)
  (hpqrs : p * q * r * s = 2020) :
  p + q + r + s = 32.1 := by
sorry

end NUMINAMATH_CALUDE_sum_pqrs_equals_32_1_l1868_186868


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1868_186896

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 10*x - 4*y + 6 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 6 - 10*h + 4*k) ∧ h + k = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1868_186896


namespace NUMINAMATH_CALUDE_factor_and_multiple_of_thirteen_l1868_186865

theorem factor_and_multiple_of_thirteen (n : ℕ) : 
  (∃ k : ℕ, 13 = n * k) ∧ (∃ m : ℕ, n = 13 * m) → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_factor_and_multiple_of_thirteen_l1868_186865


namespace NUMINAMATH_CALUDE_complex_magnitude_l1868_186866

theorem complex_magnitude (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1868_186866


namespace NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l1868_186894

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := 115

/-- The cost of a t-shirt in dollars -/
def tshirt_cost : ℕ := 25

/-- The number of t-shirts sold during the game -/
def tshirts_sold : ℕ := 113

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := 78

theorem jersey_tshirt_cost_difference : jersey_cost - tshirt_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l1868_186894


namespace NUMINAMATH_CALUDE_coverable_polyhedron_exists_l1868_186840

/-- A polyhedron that can be covered by a square and an equilateral triangle -/
structure CoverablePolyhedron where
  /-- Side length of the square -/
  s : ℝ
  /-- Side length of the equilateral triangle -/
  t : ℝ
  /-- The perimeters of the square and triangle are equal -/
  h_perimeter : 4 * s = 3 * t
  /-- The polyhedron exists and can be covered -/
  h_exists : Prop

/-- Theorem stating that there exists a polyhedron that can be covered by a square and an equilateral triangle with equal perimeters -/
theorem coverable_polyhedron_exists : ∃ (p : CoverablePolyhedron), p.h_exists := by
  sorry

end NUMINAMATH_CALUDE_coverable_polyhedron_exists_l1868_186840


namespace NUMINAMATH_CALUDE_unique_solution_system_l1868_186893

/-- Given positive real numbers a, b, c, prove that the unique solution to the system of equations:
    1. x + y + z = a + b + c
    2. 4xyz - (a²x + b²y + c²z) = abc
    is x = (b+c)/2, y = (c+a)/2, z = (a+b)/2, where x, y, z are positive real numbers. -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_system_l1868_186893


namespace NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1868_186828

theorem closest_to_sqrt_difference : 
  let diff := Real.sqrt 101 - Real.sqrt 99
  let options := [0.10, 0.12, 0.14, 0.16, 0.18]
  ∀ x ∈ options, x ≠ 0.10 → |diff - 0.10| < |diff - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1868_186828


namespace NUMINAMATH_CALUDE_solve_bath_towels_problem_l1868_186839

def bath_towels_problem (kylie_towels husband_towels : ℕ) 
  (towels_per_load loads : ℕ) : Prop :=
  let total_towels := towels_per_load * loads
  let daughters_towels := total_towels - (kylie_towels + husband_towels)
  daughters_towels = 6

theorem solve_bath_towels_problem : 
  bath_towels_problem 3 3 4 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_bath_towels_problem_l1868_186839


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l1868_186886

theorem tan_thirty_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l1868_186886


namespace NUMINAMATH_CALUDE_second_prize_proportion_l1868_186879

theorem second_prize_proportion (total winners : ℕ) 
  (first second third : ℕ) 
  (h1 : first + second + third = winners)
  (h2 : (first + second : ℚ) / winners = 3 / 4)
  (h3 : (second + third : ℚ) / winners = 2 / 3) :
  (second : ℚ) / winners = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_second_prize_proportion_l1868_186879


namespace NUMINAMATH_CALUDE_solution_characterization_l1868_186820

/-- A function satisfying the given differential equation for all real x and positive integers n -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (n : ℕ), n > 0 → Differentiable ℝ f ∧ 
    deriv f x = (f (x + n) - f x) / n

/-- The main theorem stating that any function satisfying the differential equation
    is of the form f(x) = ax + b for some real constants a and b -/
theorem solution_characterization (f : ℝ → ℝ) :
  SatisfiesDiffEq f → ∃ (a b : ℝ), ∀ x, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l1868_186820


namespace NUMINAMATH_CALUDE_animus_tower_workers_l1868_186834

theorem animus_tower_workers (beavers spiders : ℕ) 
  (h1 : beavers = 318) 
  (h2 : spiders = 544) : 
  beavers + spiders = 862 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_workers_l1868_186834
