import Mathlib

namespace complex_equation_solution_l2459_245983

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l2459_245983


namespace prime_between_squares_l2459_245902

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ n : ℕ, n^2 = p - 9 ∧ (n+1)^2 = p + 8 := by
  sorry

end prime_between_squares_l2459_245902


namespace product_of_difference_and_sum_of_squares_l2459_245924

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end product_of_difference_and_sum_of_squares_l2459_245924


namespace root_exists_in_interval_l2459_245915

def f (x : ℝ) := x^3 + 3*x - 3

theorem root_exists_in_interval : ∃ x ∈ Set.Icc 0 1, f x = 0 := by
  sorry

end root_exists_in_interval_l2459_245915


namespace simplify_sqrt_expression_l2459_245982

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l2459_245982


namespace range_of_f_l2459_245997

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {π/4, Real.arctan 2} := by sorry

end range_of_f_l2459_245997


namespace cryptarithmetic_puzzle_l2459_245931

theorem cryptarithmetic_puzzle (A B C : ℕ) : 
  A + B + C = 10 →
  B + A + 1 = 10 →
  A + 1 = 3 →
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
  C = 1 := by
sorry

end cryptarithmetic_puzzle_l2459_245931


namespace perfect_square_factors_of_4410_l2459_245911

/-- Given that 4410 = 2 × 3² × 5 × 7², this function counts the number of positive integer factors of 4410 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization := [(2, 1), (3, 2), (5, 1), (7, 2)]
  sorry

/-- The theorem states that the number of positive integer factors of 4410 that are perfect squares is 4. -/
theorem perfect_square_factors_of_4410 : count_perfect_square_factors = 4 := by
  sorry

end perfect_square_factors_of_4410_l2459_245911


namespace no_solution_exists_l2459_245922

theorem no_solution_exists : ∀ n : ℤ, n^2022 - 2*n^2021 + 3*n^2019 ≠ 2020 := by
  sorry

end no_solution_exists_l2459_245922


namespace not_perfect_square_l2459_245993

theorem not_perfect_square (n : ℕ) (d : ℕ) (h : d > 0) (h' : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), x^2 = n^2 + d := by
sorry

end not_perfect_square_l2459_245993


namespace relationship_abcd_l2459_245979

theorem relationship_abcd (a b c d : ℝ) 
  (h : (a + 2*b) / (b + 2*c) = (c + 2*d) / (d + 2*a)) :
  b = 2*a ∨ a + b + c + d = 0 := by
sorry

end relationship_abcd_l2459_245979


namespace factorization_of_4x_cubed_minus_16x_l2459_245996

theorem factorization_of_4x_cubed_minus_16x (x : ℝ) :
  4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := by
  sorry

end factorization_of_4x_cubed_minus_16x_l2459_245996


namespace pedestrian_cyclist_speeds_l2459_245984

/-- Proves that given the conditions of the problem, the pedestrian's speed is 5 km/h and the cyclist's speed is 11 km/h -/
theorem pedestrian_cyclist_speeds :
  ∀ (v₁ v₂ : ℝ),
    (27 : ℝ) > 0 →  -- Distance from A to B is 27 km
    (12 / 5 * v₁ - v₂ = 1) →  -- After 1 hour of cyclist's travel, they were 1 km behind the pedestrian
    (27 - 17 / 5 * v₁ = 2 * (27 - 2 * v₂)) →  -- After 2 hours of cyclist's travel, the cyclist had half the distance to B remaining compared to the pedestrian
    v₁ = 5 ∧ v₂ = 11 := by
  sorry

#check pedestrian_cyclist_speeds

end pedestrian_cyclist_speeds_l2459_245984


namespace quadratic_sets_equal_or_disjoint_l2459_245968

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of f(2n) where n is an integer -/
def M (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n)}

/-- The set of f(2n+1) where n is an integer -/
def N (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n + 1)}

/-- Theorem: For any quadratic function, M and N are either equal or disjoint -/
theorem quadratic_sets_equal_or_disjoint (a b c : ℝ) :
  let f := QuadraticFunction a b c
  (M f = N f) ∨ (M f ∩ N f = ∅) := by
  sorry

end quadratic_sets_equal_or_disjoint_l2459_245968


namespace can_repair_propeller_l2459_245941

/-- Represents the cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- Represents the cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- Represents the discount threshold in tugriks -/
def discount_threshold : ℕ := 250

/-- Represents the discount rate as a percentage -/
def discount_rate : ℚ := 20 / 100

/-- Represents Karlson's budget in tugriks -/
def budget : ℕ := 360

/-- Calculates the discounted price of an item -/
def apply_discount (price : ℕ) : ℚ :=
  (1 - discount_rate) * price

/-- Theorem stating that Karlson can repair his propeller with his budget -/
theorem can_repair_propeller : ∃ (first_purchase second_purchase : ℕ),
  first_purchase ≥ discount_threshold ∧
  first_purchase + second_purchase ≤ budget ∧
  first_purchase = 2 * blade_cost + 2 * screw_cost ∧
  second_purchase = apply_discount blade_cost :=
sorry

end can_repair_propeller_l2459_245941


namespace B_pow_101_eq_B_l2459_245950

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_pow_101_eq_B : B^101 = B := by sorry

end B_pow_101_eq_B_l2459_245950


namespace sum_six_terms_eq_neg_24_l2459_245936

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term : a 1 = 1
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d
  d_nonzero : d ≠ 0
  geometric_subseq : (a 3 / a 2) = (a 6 / a 3)

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem sum_six_terms_eq_neg_24 (seq : ArithmeticSequence) :
  sum_n_terms seq 6 = -24 := by
  sorry

end sum_six_terms_eq_neg_24_l2459_245936


namespace fraction_to_decimal_l2459_245978

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l2459_245978


namespace point_on_line_l2459_245914

/-- Given a line passing through (0,10) and (-8,0), this theorem proves that 
    the x-coordinate of a point on this line with y-coordinate -6 is -64/5 -/
theorem point_on_line (x : ℚ) : 
  (∀ t : ℚ, t * (-8) = x ∧ t * (-10) + 10 = -6) → x = -64/5 := by
  sorry

end point_on_line_l2459_245914


namespace middle_number_calculation_l2459_245961

theorem middle_number_calculation (n : ℕ) (total_avg first_avg last_avg : ℚ) : 
  n = 11 →
  total_avg = 9.9 →
  first_avg = 10.5 →
  last_avg = 11.4 →
  ∃ (middle : ℚ), 
    middle = 22.5 ∧
    n * total_avg = (n / 2 : ℚ) * first_avg + (n / 2 : ℚ) * last_avg - middle :=
by
  sorry

end middle_number_calculation_l2459_245961


namespace hazel_received_six_l2459_245916

/-- The number of shirts Hazel received -/
def hazel_shirts : ℕ := sorry

/-- The number of shirts Razel received -/
def razel_shirts : ℕ := sorry

/-- The total number of shirts Hazel and Razel have -/
def total_shirts : ℕ := 18

/-- Razel received twice the number of shirts as Hazel -/
axiom razel_twice_hazel : razel_shirts = 2 * hazel_shirts

/-- The total number of shirts is the sum of Hazel's and Razel's shirts -/
axiom total_is_sum : total_shirts = hazel_shirts + razel_shirts

/-- Theorem: Hazel received 6 shirts -/
theorem hazel_received_six : hazel_shirts = 6 := by sorry

end hazel_received_six_l2459_245916


namespace angle_half_in_third_quadrant_l2459_245980

/-- Given an angle α in the second quadrant with |cos(α/2)| = -cos(α/2),
    prove that α/2 is in the third quadrant. -/
theorem angle_half_in_third_quadrant (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (|Real.cos (α/2)| = -Real.cos (α/2)) →  -- |cos(α/2)| = -cos(α/2)
  (π < α/2 ∧ α/2 < 3*π/2) :=  -- α/2 is in the third quadrant
by sorry

end angle_half_in_third_quadrant_l2459_245980


namespace davidsons_class_as_l2459_245907

/-- Proves that given the conditions of the problem, 12 students in Mr. Davidson's class received an 'A' -/
theorem davidsons_class_as (carter_total : ℕ) (carter_as : ℕ) (davidson_total : ℕ) :
  carter_total = 20 →
  carter_as = 8 →
  davidson_total = 30 →
  ∃ davidson_as : ℕ,
    davidson_as * carter_total = carter_as * davidson_total ∧
    davidson_as = 12 :=
by sorry

end davidsons_class_as_l2459_245907


namespace max_roses_for_680_l2459_245966

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 5.3
  , dozen := 36
  , twoDozen := 50 }

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 317 := by
  sorry

end max_roses_for_680_l2459_245966


namespace min_players_on_team_l2459_245935

theorem min_players_on_team (total_score : ℕ) (min_score max_score : ℕ) : 
  total_score = 100 →
  min_score = 7 →
  max_score = 23 →
  (∃ (num_players : ℕ), 
    num_players ≥ 1 ∧
    (∀ (player_scores : List ℕ), 
      player_scores.length = num_players →
      (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) →
      player_scores.sum = total_score) ∧
    (∀ (n : ℕ), n < num_players →
      ¬∃ (player_scores : List ℕ),
        player_scores.length = n ∧
        (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) ∧
        player_scores.sum = total_score)) →
  (∃ (num_players : ℕ), num_players = 12) :=
by
  sorry

end min_players_on_team_l2459_245935


namespace some_ounce_glass_size_l2459_245927

/-- Given the following conditions:
  - Claudia has 122 ounces of water
  - She fills six 5-ounce glasses and four 8-ounce glasses
  - She can fill 15 glasses of the some-ounce size with the remaining water
  Prove that the size of the some-ounce glasses is 4 ounces. -/
theorem some_ounce_glass_size (total_water : ℕ) (five_ounce_count : ℕ) (eight_ounce_count : ℕ) (some_ounce_count : ℕ)
  (h1 : total_water = 122)
  (h2 : five_ounce_count = 6)
  (h3 : eight_ounce_count = 4)
  (h4 : some_ounce_count = 15)
  (h5 : total_water = 5 * five_ounce_count + 8 * eight_ounce_count + some_ounce_count * (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count) :
  (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count = 4 := by
  sorry

end some_ounce_glass_size_l2459_245927


namespace parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l2459_245987

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line passing through P(-2, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 2) + 2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the distance from a point to the focus
def distToFocus (x y : ℝ) : ℝ := y + 1

theorem parabola_line_intersection (k : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    x₁ ≠ x₂ := by sorry

theorem midpoint_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line (-1) x₁ y₁ ∧ line (-1) x₂ y₂ ∧
    x₁ + x₂ = -4 ∧ y₁ + y₂ = 4 := by sorry

theorem min_distance_product :
  ∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ ≥ 9/2 := by sorry

-- Main theorems to prove
theorem parabola_line_intersection_properties :
  -- 1) When P(-2, 2) is the midpoint of AB, the equation of line AB is x + y = 0
  (∀ (x y : ℝ), line (-1) x y ↔ x + y = 0) ∧
  -- 2) The minimum value of |AF|•|BF| is 9/2
  (∃ (k : ℝ),
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      parabola x₁ y₁ → parabola x₂ y₂ →
      line k x₁ y₁ → line k x₂ y₂ →
      distToFocus x₁ y₁ * distToFocus x₂ y₂ = 9/2) := by sorry

end parabola_line_intersection_midpoint_line_equation_min_distance_product_parabola_line_intersection_properties_l2459_245987


namespace roden_fish_purchase_cost_l2459_245944

/-- Calculate the total cost of Roden's fish purchase -/
theorem roden_fish_purchase_cost : 
  let goldfish_cost : ℕ := 15 * 3
  let blue_fish_cost : ℕ := 7 * 6
  let neon_tetra_cost : ℕ := 10 * 2
  let angelfish_cost : ℕ := 5 * 8
  let total_cost : ℕ := goldfish_cost + blue_fish_cost + neon_tetra_cost + angelfish_cost
  total_cost = 147 := by
  sorry

end roden_fish_purchase_cost_l2459_245944


namespace sin_three_pi_half_plus_alpha_l2459_245965

/-- If the terminal side of angle α passes through point P(-5,-12), then sin(3π/2 + α) = 5/13 -/
theorem sin_three_pi_half_plus_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = -5 ∧ r * (Real.sin α) = -12) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
sorry

end sin_three_pi_half_plus_alpha_l2459_245965


namespace union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l2459_245906

open Set Real

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (ℂR A) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

-- Theorems for A ∩ C in different cases
theorem intersect_A_C_case1 (a : ℝ) (h : a ≤ 1) : A ∩ C a = ∅ := by sorry

theorem intersect_A_C_case2 (a : ℝ) (h : 1 < a ∧ a ≤ 7) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < a} := by sorry

theorem intersect_A_C_case3 (a : ℝ) (h : 7 < a) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < 7} := by sorry

end union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l2459_245906


namespace diana_charge_amount_l2459_245934

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem diana_charge_amount :
  ∃ (P : ℝ),
    (P > 0) ∧
    (P < 80.25) ∧
    (P + simple_interest P 0.07 1 = 80.25) ∧
    (abs (P - 75) < 0.01) := by
  sorry

end diana_charge_amount_l2459_245934


namespace pension_calculation_l2459_245973

/-- Given a pension system where:
  * The annual pension is proportional to the square root of years served
  * Serving 'a' additional years increases the pension by 'p' dollars
  * Serving 'b' additional years (b ≠ a) increases the pension by 'q' dollars
This theorem proves that the annual pension can be expressed in terms of a, b, p, and q. -/
theorem pension_calculation (a b p q : ℝ) (h_ab : a ≠ b) :
  ∃ (x y k : ℝ),
    x = k * Real.sqrt y ∧
    x + p = k * Real.sqrt (y + a) ∧
    x + q = k * Real.sqrt (y + b) →
    x = (a * q^2 - b * p^2) / (2 * (b * p - a * q)) :=
sorry

end pension_calculation_l2459_245973


namespace fractional_equation_solution_l2459_245956

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ -1) ∧ (6 / (x + 1) = (x + 5) / (x * (x + 1))) ∧ x = 1 :=
by
  sorry

end fractional_equation_solution_l2459_245956


namespace y_intercept_of_given_line_l2459_245948

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

/-- The given line has slope 3 and passes through the point (4, 0) -/
def given_line : Line :=
  { slope := 3, point := (4, 0) }

theorem y_intercept_of_given_line :
  y_intercept given_line = -12 := by
  sorry

end y_intercept_of_given_line_l2459_245948


namespace problem_solution_l2459_245909

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 8) (h2 : x = 2) : y = 3/2 := by
  sorry

end problem_solution_l2459_245909


namespace variety_show_theorem_l2459_245946

/-- Represents the number of acts in the variety show -/
def num_acts : ℕ := 7

/-- Represents the number of acts with adjacency restrictions -/
def num_restricted_acts : ℕ := 3

/-- Represents the number of acts without adjacency restrictions -/
def num_unrestricted_acts : ℕ := num_acts - num_restricted_acts

/-- Represents the number of spaces available for restricted acts -/
def num_spaces : ℕ := num_unrestricted_acts + 1

/-- The number of ways to arrange the variety show program -/
def variety_show_arrangements : ℕ :=
  (num_spaces.choose num_restricted_acts) * 
  (Nat.factorial num_restricted_acts) * 
  (Nat.factorial num_unrestricted_acts)

theorem variety_show_theorem : 
  variety_show_arrangements = 1440 := by
  sorry

end variety_show_theorem_l2459_245946


namespace tangent_line_implies_a_b_values_l2459_245913

noncomputable section

def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

def tangent_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

theorem tangent_line_implies_a_b_values (a b : ℝ) :
  (∀ x, tangent_line x (f a b x)) →
  (tangent_line 1 (f a b 1)) →
  (a = 1 ∧ b = 1) := by sorry

end

end tangent_line_implies_a_b_values_l2459_245913


namespace bottom_row_bricks_count_l2459_245908

/-- Represents a brick wall with a triangular pattern -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ
  h_rows : rows > 0
  h_pattern : total_bricks = (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- The specific brick wall in the problem -/
def problem_wall : BrickWall where
  rows := 5
  total_bricks := 200
  bottom_row_bricks := 42
  h_rows := by norm_num
  h_pattern := by norm_num

theorem bottom_row_bricks_count (wall : BrickWall) 
  (h_rows : wall.rows = 5) 
  (h_total : wall.total_bricks = 200) : 
  wall.bottom_row_bricks = 42 := by
  sorry

#check bottom_row_bricks_count

end bottom_row_bricks_count_l2459_245908


namespace fraction_simplification_l2459_245985

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : 
  (2 / (1 - x)) - ((2 * x) / (1 - x)) = 2 := by
sorry

end fraction_simplification_l2459_245985


namespace fraction_inequality_l2459_245930

theorem fraction_inequality (a b c d : ℕ+) 
  (h1 : a + c ≤ 1982)
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  1 - (a : ℚ) / b - (c : ℚ) / d > 1 / (1983 ^ 3) := by
  sorry

end fraction_inequality_l2459_245930


namespace equation_solution_l2459_245917

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1/9 ∧ x₂ = 1/18) ∧ 
  (∀ x : ℚ, (101*x^2 - 18*x + 1)^2 - 121*x^2*(101*x^2 - 18*x + 1) + 2020*x^4 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l2459_245917


namespace digit_sum_divisibility_27_l2459_245932

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_divisibility_27 : 
  ∃ n : ℕ, (sum_of_digits n % 27 = 0) ∧ (n % 27 ≠ 0) := by sorry

end digit_sum_divisibility_27_l2459_245932


namespace line_y_intercept_l2459_245976

/-- A line in the xy-plane is defined by its slope and a point it passes through. 
    This theorem proves that for a line with slope 2 passing through (498, 998), 
    the y-intercept is 2. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 2 ∧ x = 498 ∧ y = 998 ∧ y = m * x + b → b = 2 :=
by sorry

end line_y_intercept_l2459_245976


namespace kenny_must_do_at_least_three_on_thursday_l2459_245986

/-- Represents the number of jumping jacks done on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def weekTotal (w : WeeklyJumpingJacks) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

theorem kenny_must_do_at_least_three_on_thursday 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (someDay : ℕ) :
  lastWeek = 324 →
  thisWeek.sunday = 34 →
  thisWeek.monday = 20 →
  thisWeek.tuesday = 0 →
  thisWeek.wednesday = 123 →
  thisWeek.saturday = 61 →
  (thisWeek.thursday = someDay ∨ thisWeek.friday = someDay) →
  someDay = 23 →
  weekTotal thisWeek > lastWeek →
  thisWeek.thursday ≥ 3 :=
by sorry

end kenny_must_do_at_least_three_on_thursday_l2459_245986


namespace rectangular_playground_area_l2459_245957

theorem rectangular_playground_area : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  2 * (length + width) = 84 →
  length = 3 * width →
  length * width = 330.75 := by
sorry

end rectangular_playground_area_l2459_245957


namespace sets_are_equal_l2459_245967

def M : Set ℝ := {y | ∃ x, y = x^2 + 3}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 3)}

theorem sets_are_equal : M = N := by sorry

end sets_are_equal_l2459_245967


namespace different_winning_scores_l2459_245904

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runners_per_team : Nat
  /-- The total number of runners -/
  total_runners : Nat
  /-- The sum of all positions -/
  total_sum : Nat
  /-- The lowest possible winning score -/
  min_winning_score : Nat
  /-- The highest possible winning score -/
  max_winning_score : Nat
  /-- Assertion that there are two teams -/
  two_teams : total_runners = 2 * runners_per_team
  /-- Assertion that the total sum is correct -/
  sum_correct : total_sum = (total_runners * (total_runners + 1)) / 2
  /-- Assertion that the minimum winning score is correct -/
  min_score_correct : min_winning_score = (runners_per_team * (runners_per_team + 1)) / 2
  /-- Assertion that the maximum winning score is less than half the total sum -/
  max_score_correct : max_winning_score = (total_sum / 2) - 1

/-- The main theorem stating the number of different winning scores -/
theorem different_winning_scores (meet : CrossCountryMeet) (h : meet.runners_per_team = 5) :
  (meet.max_winning_score - meet.min_winning_score + 1) = 13 := by
  sorry

end different_winning_scores_l2459_245904


namespace min_value_of_f_l2459_245925

/-- The quadratic function f(x) = 2x^2 - 16x + 22 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

/-- Theorem: The minimum value of f(x) = 2x^2 - 16x + 22 is -10 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -10 ∧ ∃ x₀ : ℝ, f x₀ = -10 :=
by sorry

end min_value_of_f_l2459_245925


namespace intersection_A_B_l2459_245905

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 4) * (x - 1) < 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by sorry

end intersection_A_B_l2459_245905


namespace prob_odd_sum_given_even_product_l2459_245975

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 3 / 8

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 5 / 8

/-- The number of ways to get an odd sum with one odd die -/
def odd_sum_one_odd : ℕ := num_dice * 3 * 5^4

/-- The number of ways to get an odd sum with three odd dice -/
def odd_sum_three_odd : ℕ := (num_dice.choose 3) * 3^3 * 5^2

/-- The number of ways to get an odd sum with all odd dice -/
def odd_sum_all_odd : ℕ := 3^5

/-- The total number of favorable outcomes (odd sum) -/
def total_favorable : ℕ := odd_sum_one_odd + odd_sum_three_odd + odd_sum_all_odd

/-- The total number of possible outcomes where the product is even -/
def total_possible : ℕ := 8^5 - (3/8)^5 * 8^5

/-- The probability of getting an odd sum given that the product is even -/
theorem prob_odd_sum_given_even_product :
  (total_favorable : ℚ) / total_possible =
  (5 * 3 * 5^4 + 10 * 27 * 25 + 243) / (8^5 - (3/8)^5 * 8^5) :=
by sorry

end prob_odd_sum_given_even_product_l2459_245975


namespace diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l2459_245901

structure Polygon where
  sides : ℕ
  interior_angle : ℝ
  diagonal_angle : ℝ

def is_equilateral_triangle (p : Polygon) : Prop :=
  p.sides = 3 ∧ p.interior_angle = 60

def is_regular_hexagon (p : Polygon) : Prop :=
  p.sides = 6 ∧ p.interior_angle = 120

theorem diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon (p : Polygon) :
  p.diagonal_angle = 60 → is_equilateral_triangle p ∨ is_regular_hexagon p :=
by sorry

end diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l2459_245901


namespace solve_coloring_book_problem_l2459_245981

def coloring_book_problem (book1 book2 book3 colored : ℕ) : Prop :=
  let total := book1 + book2 + book3
  total - colored = 53

theorem solve_coloring_book_problem :
  coloring_book_problem 35 45 40 67 := by
  sorry

end solve_coloring_book_problem_l2459_245981


namespace smallest_n_square_and_cube_l2459_245964

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (a : ℕ), 5 * n = a ^ 2) ∧ 
   (∃ (b : ℕ), 3 * n = b ^ 3)) → 
  (∀ (m : ℕ), m > 0 → 
   (∃ (a : ℕ), 5 * m = a ^ 2) → 
   (∃ (b : ℕ), 3 * m = b ^ 3) → 
   m ≥ 1125) ∧
  (∃ (a b : ℕ), 5 * 1125 = a ^ 2 ∧ 3 * 1125 = b ^ 3) :=
by sorry

end smallest_n_square_and_cube_l2459_245964


namespace area_APRQ_is_6_25_l2459_245977

/-- A rectangle with points P, Q, and R located on its sides. -/
structure RectangleWithPoints where
  /-- The area of rectangle ABCD -/
  area : ℝ
  /-- Point P is located one-fourth the length of side AD from vertex A -/
  p_location : ℝ
  /-- Point Q is located one-fourth the length of side CD from vertex C -/
  q_location : ℝ
  /-- Point R is located one-fourth the length of side BC from vertex B -/
  r_location : ℝ

/-- The area of quadrilateral APRQ in a rectangle with given properties -/
def area_APRQ (rect : RectangleWithPoints) : ℝ := sorry

/-- Theorem stating that the area of APRQ is 6.25 square meters -/
theorem area_APRQ_is_6_25 (rect : RectangleWithPoints) 
  (h1 : rect.area = 100)
  (h2 : rect.p_location = 1/4)
  (h3 : rect.q_location = 1/4)
  (h4 : rect.r_location = 1/4) : 
  area_APRQ rect = 6.25 := by sorry

end area_APRQ_is_6_25_l2459_245977


namespace lcm_problem_l2459_245918

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm m 45 = 180) : m = 72 := by
  sorry

end lcm_problem_l2459_245918


namespace f_intersects_y_axis_l2459_245969

-- Define the function f(x) = 4x - 4
def f (x : ℝ) : ℝ := 4 * x - 4

-- Theorem: f intersects the y-axis at (0, -4)
theorem f_intersects_y_axis :
  f 0 = -4 := by sorry

end f_intersects_y_axis_l2459_245969


namespace sandra_coffee_cups_l2459_245920

/-- Given that Sandra and Marcie took a total of 8 cups of coffee, 
    and Marcie took 2 cups, prove that Sandra took 6 cups of coffee. -/
theorem sandra_coffee_cups (total : ℕ) (marcie : ℕ) (sandra : ℕ) 
  (h1 : total = 8) 
  (h2 : marcie = 2) 
  (h3 : sandra + marcie = total) : 
  sandra = 6 := by
  sorry

end sandra_coffee_cups_l2459_245920


namespace factorization_proof_l2459_245998

theorem factorization_proof (x y : ℝ) : 9*y - 25*x^2*y = y*(3+5*x)*(3-5*x) := by
  sorry

end factorization_proof_l2459_245998


namespace johnny_earnings_l2459_245912

def calculate_earnings (hourly_wage : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) 
  (overtime_rate : ℝ) (tax_rate : ℝ) (insurance_rate : ℝ) : ℝ :=
  let regular_pay := hourly_wage * regular_hours
  let overtime_pay := hourly_wage * overtime_rate * overtime_hours
  let total_earnings := regular_pay + overtime_pay
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  total_earnings - tax_deduction - insurance_deduction

theorem johnny_earnings :
  let hourly_wage : ℝ := 8.25
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := 7
  let overtime_rate : ℝ := 1.5
  let tax_rate : ℝ := 0.08
  let insurance_rate : ℝ := 0.05
  abs (calculate_earnings hourly_wage regular_hours overtime_hours overtime_rate tax_rate insurance_rate - 362.47) < 0.01 :=
by sorry

end johnny_earnings_l2459_245912


namespace double_sum_reciprocal_product_l2459_245971

/-- The double sum of 1/(mn(m+n+2)) from m=1 to infinity and n=1 to infinity equals -π²/6 -/
theorem double_sum_reciprocal_product : 
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = -π^2 / 6 := by sorry

end double_sum_reciprocal_product_l2459_245971


namespace complement_union_theorem_l2459_245972

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,3,4,5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {1,2,3,6} := by sorry

end complement_union_theorem_l2459_245972


namespace range_intersection_l2459_245903

theorem range_intersection (x : ℝ) : 
  (x^2 - 7*x + 10 ≤ 0) ∧ ((x - 3)*(x + 1) ≤ 0) ↔ 2 ≤ x ∧ x ≤ 3 := by
  sorry

end range_intersection_l2459_245903


namespace calculate_product_l2459_245970

theorem calculate_product : 150 * 22.5 * (1.5^2) * 10 = 75937.5 := by
  sorry

end calculate_product_l2459_245970


namespace cycle_price_proof_l2459_245953

/-- Proves that a cycle sold at a 12% loss for Rs. 1232 had an original price of Rs. 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1232)
  (h2 : loss_percentage = 12) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

#check cycle_price_proof

end cycle_price_proof_l2459_245953


namespace notebook_cost_l2459_245947

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total : notebook_cost + pencil_cost = 2.40)
  (difference : notebook_cost = pencil_cost + 2) :
  notebook_cost = 2.20 := by
  sorry

end notebook_cost_l2459_245947


namespace smallest_with_2023_divisors_l2459_245952

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n can be written as m * 6^k where 6 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * 6^k ∧ ¬(6 ∣ m)

theorem smallest_with_2023_divisors :
  ∃ (m k : ℕ),
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    has_form (m * 6^k) m k ∧
    num_divisors (m * 6^k) = 2023 ∧
    m + k = 59055 :=
sorry

end smallest_with_2023_divisors_l2459_245952


namespace max_value_quadratic_l2459_245958

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 53 ∧ ∀ (s : ℝ), -3 * s^2 + 24 * s + 5 ≤ M :=
by sorry

end max_value_quadratic_l2459_245958


namespace sum_of_perpendiculars_eq_twice_side_l2459_245954

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- A point inside a square -/
structure PointInSquare (s : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 ≤ x ∧ x ≤ s
  y_bound : 0 ≤ y ∧ y ≤ s

/-- The sum of perpendiculars from a point to the sides of a square -/
def sumOfPerpendiculars (s : ℝ) (p : PointInSquare s) : ℝ :=
  p.x + (s - p.x) + p.y + (s - p.y)

/-- Theorem: The sum of perpendiculars from any point inside a square to its sides
    is equal to twice the side length of the square -/
theorem sum_of_perpendiculars_eq_twice_side {s : ℝ} (sq : Square s) (p : PointInSquare s) :
  sumOfPerpendiculars s p = 2 * s := by
  sorry


end sum_of_perpendiculars_eq_twice_side_l2459_245954


namespace ten_times_a_l2459_245995

theorem ten_times_a (a : ℝ) (h : a = 6) : 10 * a = 60 := by
  sorry

end ten_times_a_l2459_245995


namespace executive_board_selection_l2459_245943

theorem executive_board_selection (n m : ℕ) (h1 : n = 12) (h2 : m = 5) :
  Nat.choose n m = 792 := by
  sorry

end executive_board_selection_l2459_245943


namespace instantaneous_velocity_at_one_l2459_245999

def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

theorem instantaneous_velocity_at_one :
  (deriv h) 1 = 0.2 := by
  sorry

end instantaneous_velocity_at_one_l2459_245999


namespace count_tree_frogs_l2459_245974

theorem count_tree_frogs (total_frogs poison_frogs wood_frogs : ℕ) 
  (h1 : total_frogs = 78)
  (h2 : poison_frogs = 10)
  (h3 : wood_frogs = 13)
  (h4 : ∃ tree_frogs : ℕ, total_frogs = tree_frogs + poison_frogs + wood_frogs) :
  ∃ tree_frogs : ℕ, tree_frogs = 55 ∧ total_frogs = tree_frogs + poison_frogs + wood_frogs :=
by
  sorry

end count_tree_frogs_l2459_245974


namespace sum_of_fourth_powers_inequality_l2459_245991

theorem sum_of_fourth_powers_inequality (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 9 = 4*(x + y + z)) :
  x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) ≥ 8*(x^3 + y^3 + z^3) + 27 ∧
  (x^4 + y^4 + z^4 + 16*(x^2 + y^2 + z^2) = 8*(x^3 + y^3 + z^3) + 27 ↔ 
   (x = 1 ∨ x = 3) ∧ (y = 1 ∨ y = 3) ∧ (z = 1 ∨ z = 3)) :=
by sorry

end sum_of_fourth_powers_inequality_l2459_245991


namespace variable_equals_one_l2459_245940

/-- The operator  applied to a real number x -/
def box_operator (x : ℝ) : ℝ := x * (2 - x)

/-- Theorem stating that if y + 1 = (y + 1), then y = 1 -/
theorem variable_equals_one (y : ℝ) (h : y + 1 = box_operator (y + 1)) : y = 1 := by
  sorry

end variable_equals_one_l2459_245940


namespace science_fiction_total_pages_l2459_245963

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each science fiction book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages : total_pages = 3824 := by
  sorry

end science_fiction_total_pages_l2459_245963


namespace inequality_solution_upper_bound_l2459_245942

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem inequality_solution (x : ℝ) : f x < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

-- Part II
theorem upper_bound (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x ≤ 5/6 := by sorry

end inequality_solution_upper_bound_l2459_245942


namespace triangle_angle_max_l2459_245921

theorem triangle_angle_max (c : ℝ) (X Y Z : ℝ) : 
  0 < X ∧ 0 < Y ∧ 0 < Z →  -- angles are positive
  X + Y + Z = 180 →  -- angle sum in a triangle
  Z ≤ Y ∧ Y ≤ X →  -- given order of angles
  c * X = 6 * Z →  -- given relation between X and Z
  Z ≤ 36 :=  -- maximum value of Z
by sorry

end triangle_angle_max_l2459_245921


namespace binomial_coefficient_1500_2_l2459_245989

theorem binomial_coefficient_1500_2 : Nat.choose 1500 2 = 1124250 := by
  sorry

end binomial_coefficient_1500_2_l2459_245989


namespace billy_ice_cubes_l2459_245992

/-- Calculates the total number of ice cubes that can be made given the tray capacity and number of trays. -/
def total_ice_cubes (tray_capacity : ℕ) (num_trays : ℕ) : ℕ :=
  tray_capacity * num_trays

/-- Proves that with a tray capacity of 48 ice cubes and 24 trays, the total number of ice cubes is 1152. -/
theorem billy_ice_cubes : total_ice_cubes 48 24 = 1152 := by
  sorry

end billy_ice_cubes_l2459_245992


namespace rectangle_area_l2459_245900

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end rectangle_area_l2459_245900


namespace necessary_but_not_sufficient_l2459_245994

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end necessary_but_not_sufficient_l2459_245994


namespace largest_non_formable_is_correct_l2459_245949

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- Predicate to check if an amount can be formed using the given coin denominations -/
def is_formable (n : ℕ) (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-formable amount in Limonia -/
def largest_non_formable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem stating that the largest non-formable amount is correct -/
theorem largest_non_formable_is_correct (n : ℕ) :
  (∀ m : ℕ, m > largest_non_formable n → is_formable n m) ∧
  ¬is_formable n (largest_non_formable n) :=
sorry

end largest_non_formable_is_correct_l2459_245949


namespace selling_price_ratio_l2459_245926

/-- Given an item with cost price c, prove that the ratio of selling prices y:x is 25:16,
    where x results in a 20% loss and y results in a 25% profit. -/
theorem selling_price_ratio (c x y : ℝ) 
  (loss : x = 0.8 * c)   -- 20% loss condition
  (profit : y = 1.25 * c) -- 25% profit condition
  : y / x = 25 / 16 := by
  sorry

end selling_price_ratio_l2459_245926


namespace equation_solution_l2459_245939

def solution_set : Set (ℤ × ℤ) := {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)}

theorem equation_solution (x y : ℤ) : 
  (x + y ≠ 0) → ((x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ solution_set) := by
sorry

end equation_solution_l2459_245939


namespace eds_initial_money_l2459_245988

def night_rate : ℚ := 1.5
def morning_rate : ℚ := 2
def night_hours : ℕ := 6
def morning_hours : ℕ := 4
def remaining_money : ℚ := 63

theorem eds_initial_money :
  night_rate * night_hours + morning_rate * morning_hours + remaining_money = 80 := by
  sorry

end eds_initial_money_l2459_245988


namespace team_a_two_projects_probability_l2459_245960

/-- The number of ways to distribute n identical objects into k distinct boxes,
    where each box must contain at least one object. -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The probability of team A contracting exactly two projects out of five projects
    distributed among four teams, where each team must contract at least one project. -/
theorem team_a_two_projects_probability :
  let total_distributions := stars_and_bars 5 4
  let favorable_distributions := stars_and_bars 3 3
  (favorable_distributions : ℚ) / total_distributions = 1 / 4 := by
  sorry

end team_a_two_projects_probability_l2459_245960


namespace last_erased_numbers_l2459_245919

-- Define a function to count prime factors
def count_prime_factors (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem last_erased_numbers :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 →
    (count_prime_factors n = 6 ↔ n = 64 ∨ n = 96) ∧
    (count_prime_factors n ≤ 6) :=
by sorry

end last_erased_numbers_l2459_245919


namespace second_car_speed_l2459_245938

/-- Proves that the speed of the second car is 70 km/h given the conditions of the problem -/
theorem second_car_speed (initial_distance : ℝ) (first_car_speed : ℝ) (time : ℝ) :
  initial_distance = 60 →
  first_car_speed = 90 →
  time = 3 →
  ∃ (second_car_speed : ℝ),
    second_car_speed * time + initial_distance = first_car_speed * time ∧
    second_car_speed = 70 :=
by
  sorry


end second_car_speed_l2459_245938


namespace polynomial_simplification_l2459_245937

theorem polynomial_simplification (q : ℝ) :
  (5 * q^3 - 7 * q + 8) + (3 - 9 * q^2 + 3 * q) = 5 * q^3 - 9 * q^2 - 4 * q + 11 := by
  sorry

end polynomial_simplification_l2459_245937


namespace geometric_sequence_increasing_condition_l2459_245928

/-- A geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_sequence_increasing_condition (a₁ q : ℝ) :
  ¬(((q > 1) → IncreasingSequence (GeometricSequence a₁ q)) ∧
    (IncreasingSequence (GeometricSequence a₁ q) → (q > 1))) :=
sorry

end geometric_sequence_increasing_condition_l2459_245928


namespace inverse_relation_values_l2459_245962

/-- Represents the constant product of two inversely related quantities -/
def k : ℝ := 800 * 0.5

/-- Represents the relationship between inversely related quantities a and b -/
def inverse_relation (a b : ℝ) : Prop := a * b = k

theorem inverse_relation_values (a₁ a₂ : ℝ) (h₁ : inverse_relation 800 0.5) :
  (inverse_relation 1600 0.250) ∧ (inverse_relation 400 1.000) := by
  sorry

#check inverse_relation_values

end inverse_relation_values_l2459_245962


namespace complex_equation_solution_l2459_245945

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2*I) = 3 + 2*I → z = -1/5 + 8/5*I :=
by sorry

end complex_equation_solution_l2459_245945


namespace shift_hours_is_eight_l2459_245910

/-- Calculates the number of hours in each person's shift given the following conditions:
  * 20 people are hired
  * Each person makes on average 20 shirts per day
  * Employees are paid $12 an hour plus $5 per shirt
  * Shirts are sold for $35 each
  * Nonemployee expenses are $1000 a day
  * The company makes $9080 in profits per day
-/
def calculateShiftHours (
  numEmployees : ℕ)
  (shirtsPerPerson : ℕ)
  (hourlyWage : ℕ)
  (perShirtBonus : ℕ)
  (shirtPrice : ℕ)
  (nonEmployeeExpenses : ℕ)
  (dailyProfit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the shift hours calculated by the function is 8 -/
theorem shift_hours_is_eight :
  calculateShiftHours 20 20 12 5 35 1000 9080 = 8 := by
  sorry

end shift_hours_is_eight_l2459_245910


namespace equalize_volume_l2459_245959

-- Define the volumes in milliliters
def transparent_volume : ℚ := 12400
def opaque_volume : ℚ := 7600

-- Define the conversion factor from milliliters to liters
def ml_to_l : ℚ := 1000

-- Define the function to calculate the volume to be transferred
def volume_to_transfer : ℚ :=
  (transparent_volume - opaque_volume) / 2

-- Theorem statement
theorem equalize_volume :
  volume_to_transfer = 2400 ∧
  volume_to_transfer / ml_to_l = (12 / 5 : ℚ) := by
  sorry

end equalize_volume_l2459_245959


namespace jeremy_watermelon_consumption_l2459_245955

/-- The number of watermelons Jeremy eats per week, given the total number of watermelons,
    the number of weeks they last, and the number given away each week. -/
def watermelons_eaten_per_week (total : ℕ) (weeks : ℕ) (given_away_per_week : ℕ) : ℕ :=
  (total - weeks * given_away_per_week) / weeks

/-- Theorem stating that given 30 watermelons lasting 6 weeks, 
    with 2 given away each week, Jeremy eats 3 watermelons per week. -/
theorem jeremy_watermelon_consumption :
  watermelons_eaten_per_week 30 6 2 = 3 := by
  sorry

end jeremy_watermelon_consumption_l2459_245955


namespace specific_ellipse_sum_l2459_245923

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of the semi-major axis
  b : ℝ  -- length of the semi-minor axis

/-- The sum of center coordinates and axis lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (3, -5), horizontal semi-major axis 6, and vertical semi-minor axis 2, the sum h + k + a + b equals 6 -/
theorem specific_ellipse_sum :
  ∃ (e : Ellipse), e.h = 3 ∧ e.k = -5 ∧ e.a = 6 ∧ e.b = 2 ∧ ellipse_sum e = 6 := by
  sorry

end specific_ellipse_sum_l2459_245923


namespace pet_store_kittens_l2459_245933

/-- The total number of kittens after receiving more -/
def total_kittens (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: If a pet store initially has 6 kittens and receives 3 more, 
    the total number of kittens will be 9 -/
theorem pet_store_kittens : total_kittens 6 3 = 9 := by
  sorry

end pet_store_kittens_l2459_245933


namespace investment_principal_calculation_l2459_245990

/-- Proves that given an investment with a monthly interest payment of $228 and a simple annual interest rate of 9%, the principal amount of the investment is $30,400. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 228 →
  annual_rate = 0.09 →
  ∃ principal : ℝ, principal = 30400 ∧ monthly_interest = principal * (annual_rate / 12) :=
by sorry

end investment_principal_calculation_l2459_245990


namespace negation_equivalence_l2459_245951

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end negation_equivalence_l2459_245951


namespace triangle_side_bounds_l2459_245929

theorem triangle_side_bounds (k : ℕ) (a b c : ℕ) 
  (h1 : a + b + c = k) 
  (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (2 - (k - 2*(k/2)) ≤ a ∧ a ≤ k/3) ∧
  ((k+4)/4 ≤ b ∧ b ≤ (k-1)/2) ∧
  ((k+2)/3 ≤ c ∧ c ≤ (k-1)/2) := by
  sorry

end triangle_side_bounds_l2459_245929
