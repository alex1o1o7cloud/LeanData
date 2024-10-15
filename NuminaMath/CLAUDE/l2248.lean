import Mathlib

namespace NUMINAMATH_CALUDE_dishwasher_manager_ratio_l2248_224811

/-- The hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions of the wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.22 ∧
  w.manager = 8.50 ∧
  w.manager = w.chef + 3.315

/-- The theorem stating the ratio of dishwasher to manager wages -/
theorem dishwasher_manager_ratio (w : Wages) :
  wage_conditions w → w.dishwasher / w.manager = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_dishwasher_manager_ratio_l2248_224811


namespace NUMINAMATH_CALUDE_food_additives_budget_percentage_l2248_224897

theorem food_additives_budget_percentage :
  ∀ (total_degrees : ℝ) 
    (microphotonics_percent : ℝ) 
    (home_electronics_percent : ℝ) 
    (genetically_modified_microorganisms_percent : ℝ) 
    (industrial_lubricants_percent : ℝ) 
    (basic_astrophysics_degrees : ℝ),
  total_degrees = 360 →
  microphotonics_percent = 14 →
  home_electronics_percent = 19 →
  genetically_modified_microorganisms_percent = 24 →
  industrial_lubricants_percent = 8 →
  basic_astrophysics_degrees = 90 →
  ∃ (food_additives_percent : ℝ),
    food_additives_percent = 10 ∧
    microphotonics_percent + home_electronics_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent +
    (basic_astrophysics_degrees / total_degrees * 100) + food_additives_percent = 100 :=
by sorry

end NUMINAMATH_CALUDE_food_additives_budget_percentage_l2248_224897


namespace NUMINAMATH_CALUDE_vector_sum_squared_norms_l2248_224804

theorem vector_sum_squared_norms (a b : ℝ × ℝ) :
  let m : ℝ × ℝ := (4, 10)  -- midpoint
  (∀ (x : ℝ) (y : ℝ), m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) →  -- midpoint condition
  (a.1 * b.1 + a.2 * b.2 = 12) →  -- dot product condition
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 440 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_squared_norms_l2248_224804


namespace NUMINAMATH_CALUDE_product_zero_l2248_224823

theorem product_zero (a b c : ℝ) : 
  (a^2 + b^2 = 1 ∧ a + b = 1 → a * b = 0) ∧
  (a^3 + b^3 + c^3 = 1 ∧ a^2 + b^2 + c^2 = 1 ∧ a + b + c = 1 → a * b * c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2248_224823


namespace NUMINAMATH_CALUDE_count_80_in_scores_l2248_224848

def scores : List ℕ := [80, 90, 80, 80, 100, 70]

theorem count_80_in_scores : (scores.filter (· = 80)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_80_in_scores_l2248_224848


namespace NUMINAMATH_CALUDE_product_equals_zero_l2248_224827

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2248_224827


namespace NUMINAMATH_CALUDE_x_convergence_bound_l2248_224864

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem x_convergence_bound :
  ∃ m : ℕ, (∀ k < m, x k > 4 + 1 / 2^20) ∧
           (x m ≤ 4 + 1 / 2^20) ∧
           (81 ≤ m) ∧ (m ≤ 242) :=
by sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l2248_224864


namespace NUMINAMATH_CALUDE_naples_pizza_weight_l2248_224895

/-- The total weight of pizza eaten by Rachel and Bella in Naples -/
def total_pizza_weight (rachel_pizza : ℕ) (rachel_mushrooms : ℕ) (rachel_olives : ℕ)
                       (bella_pizza : ℕ) (bella_cheese : ℕ) (bella_onions : ℕ) : ℕ :=
  (rachel_pizza + rachel_mushrooms + rachel_olives) + (bella_pizza + bella_cheese + bella_onions)

/-- Theorem stating the total weight of pizza eaten by Rachel and Bella in Naples -/
theorem naples_pizza_weight :
  total_pizza_weight 598 100 50 354 75 55 = 1232 := by
  sorry

end NUMINAMATH_CALUDE_naples_pizza_weight_l2248_224895


namespace NUMINAMATH_CALUDE_danny_soda_consumption_l2248_224814

theorem danny_soda_consumption (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →
  (1 - x / 100) + (0.3 + 0.3) = 0.7 →
  x = 90 := by
sorry

end NUMINAMATH_CALUDE_danny_soda_consumption_l2248_224814


namespace NUMINAMATH_CALUDE_circle_radius_l2248_224830

theorem circle_radius (x y : ℝ) (h : x + 2*y = 100*Real.pi) : 
  x = Real.pi * 8^2 ∧ y = 2 * Real.pi * 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2248_224830


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2248_224856

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2248_224856


namespace NUMINAMATH_CALUDE_roller_coaster_cost_proof_l2248_224891

/-- The cost of a ride on the Ferris wheel in tickets -/
def ferris_wheel_cost : ℚ := 2

/-- The discount in tickets for going on multiple rides -/
def multiple_ride_discount : ℚ := 1

/-- The value of the newspaper coupon in tickets -/
def newspaper_coupon : ℚ := 1

/-- The total number of tickets Zach needed to buy for both rides -/
def total_tickets_bought : ℚ := 7

/-- The cost of a ride on the roller coaster in tickets -/
def roller_coaster_cost : ℚ := 7

theorem roller_coaster_cost_proof :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = total_tickets_bought :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_proof_l2248_224891


namespace NUMINAMATH_CALUDE_factorization_proof_l2248_224862

theorem factorization_proof (x : ℝ) : 4*x^3 - 8*x^2 + 4*x = 4*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2248_224862


namespace NUMINAMATH_CALUDE_least_period_is_twelve_l2248_224853

/-- A function satisfying the given condition -/
def SatisfiesCondition (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) + g (x - 2) = g x

/-- The period of a function -/
def IsPeriod (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g (x + p) = g x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (g : ℝ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ IsPeriod g q ∧ ∀ p, 0 < p ∧ p < q → ¬IsPeriod g p

theorem least_period_is_twelve :
  ∀ g : ℝ → ℝ, SatisfiesCondition g → IsLeastPositivePeriod g 12 := by
  sorry

end NUMINAMATH_CALUDE_least_period_is_twelve_l2248_224853


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2248_224839

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2248_224839


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l2248_224831

/-- In a rectangle with a line cutting it, if the area of the resulting quadrilateral
    is 40% of the total area, then the height of this quadrilateral is 0.8 times
    the length of the rectangle. -/
theorem rectangle_cut_theorem (L W x : ℝ) : 
  L > 0 → W > 0 → x > 0 →
  x * W / 2 = 0.4 * L * W →
  x = 0.8 * L := by
sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l2248_224831


namespace NUMINAMATH_CALUDE_polynomial_piecewise_function_l2248_224849

theorem polynomial_piecewise_function 
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = 0)
  (hg : ∀ x, g x = -x)
  (hh : ∀ x, h x = -x + 2) :
  ∀ x, |f x| - |g x| + h x = 
    if x < -1 then -1
    else if x ≤ 0 then 2
    else -2*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_piecewise_function_l2248_224849


namespace NUMINAMATH_CALUDE_max_prize_winners_l2248_224815

/-- Represents a tournament with given number of players and point thresholds. -/
structure Tournament :=
  (num_players : ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)
  (prize_threshold : ℕ)

/-- Calculates the total number of games in a round-robin tournament. -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total points available in the tournament. -/
def total_points (t : Tournament) : ℕ :=
  total_games t.num_players * t.win_points

/-- Theorem stating the maximum number of prize winners in the specific tournament. -/
theorem max_prize_winners (t : Tournament) 
  (h1 : t.num_players = 15)
  (h2 : t.win_points = 2)
  (h3 : t.draw_points = 1)
  (h4 : t.loss_points = 0)
  (h5 : t.prize_threshold = 20) :
  ∃ (max_winners : ℕ), max_winners = 9 ∧ 
  (∀ (n : ℕ), n > max_winners → 
    n * t.prize_threshold > total_points t) :=
sorry

end NUMINAMATH_CALUDE_max_prize_winners_l2248_224815


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2248_224882

theorem inequality_system_solution :
  let S : Set ℝ := {x | 5 * x - 2 < 3 * (x + 1) ∧ (3 * x - 2) / 3 ≥ x + (x - 2) / 2}
  S = {x | x ≤ 2/3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2248_224882


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_permutations_l2248_224843

/-- Represents the number of legs a centipede has -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the constraint that for each leg, the sock must be put on before the shoe -/
def sock_before_shoe_constraint (leg : ℕ) : Prop :=
  leg ≤ num_legs ∧ ∃ (sock_pos shoe_pos : ℕ), sock_pos < shoe_pos

/-- The main theorem stating the number of valid permutations -/
theorem centipede_sock_shoe_permutations :
  (Nat.factorial total_items) / (2^num_legs) =
  (Nat.factorial 20) / (2^10) :=
sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_permutations_l2248_224843


namespace NUMINAMATH_CALUDE_chocolate_bars_bought_correct_number_of_bars_l2248_224800

def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def calories_in_lollipop : ℕ := 190
def total_sugar : ℕ := 177

theorem chocolate_bars_bought : ℕ :=
  (total_sugar - sugar_in_lollipop) / sugar_per_chocolate_bar

theorem correct_number_of_bars : chocolate_bars_bought = 14 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_bought_correct_number_of_bars_l2248_224800


namespace NUMINAMATH_CALUDE_least_consecutive_bigness_l2248_224884

def bigness (a b c : ℕ) : ℕ := a * b * c + 2 * (a * b + b * c + a * c) + 4 * (a + b + c)

def has_integer_sides (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ bigness a b c = n

theorem least_consecutive_bigness :
  (∀ k < 55, ¬(has_integer_sides k ∧ has_integer_sides (k + 1))) ∧
  (has_integer_sides 55 ∧ has_integer_sides 56) :=
sorry

end NUMINAMATH_CALUDE_least_consecutive_bigness_l2248_224884


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l2248_224880

-- Define the given circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0
def line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  (∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → result_circle x y) ∧
  (∃ h k : ℝ, line h k ∧ ∀ x y : ℝ, result_circle x y ↔ (x - h)^2 + (y - k)^2 = ((h^2 + k^2 - 3) / 2)) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l2248_224880


namespace NUMINAMATH_CALUDE_largest_inscribed_square_area_l2248_224842

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 10

/-- The rhombus formed by two identical equilateral triangles -/
structure Rhombus where
  side : ℝ
  is_formed_by_triangles : side = triangle_side

/-- The largest square inscribed in the rhombus -/
def largest_inscribed_square (r : Rhombus) : ℝ := sorry

/-- Theorem stating that the area of the largest inscribed square is 50 -/
theorem largest_inscribed_square_area (r : Rhombus) :
  (largest_inscribed_square r) ^ 2 = 50 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_area_l2248_224842


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l2248_224857

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l2248_224857


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l2248_224869

/-- The volume of a right triangular prism with base edge length 2 and height 3 is √3. -/
theorem right_triangular_prism_volume : 
  let base_edge : ℝ := 2
  let height : ℝ := 3
  let base_area : ℝ := Real.sqrt 3 / 4 * base_edge ^ 2
  let volume : ℝ := 1 / 3 * base_area * height
  volume = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l2248_224869


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2248_224866

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℚ := 3

/-- Length of the rectangular prism in yards -/
def length_yd : ℚ := 1

/-- Width of the rectangular prism in yards -/
def width_yd : ℚ := 2

/-- Height of the rectangular prism in yards -/
def height_yd : ℚ := 3

/-- Volume of a rectangular prism in cubic feet -/
def volume_cubic_feet (l w h : ℚ) : ℚ := l * w * h * (yards_to_feet ^ 3)

/-- Theorem stating that the volume of the given rectangular prism is 162 cubic feet -/
theorem rectangular_prism_volume :
  volume_cubic_feet length_yd width_yd height_yd = 162 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2248_224866


namespace NUMINAMATH_CALUDE_solution_equivalence_l2248_224858

theorem solution_equivalence (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (x^2 + 3*x + 1/(x-1) = a + 1/(x-1) ↔ x^2 + 3*x = a)) ∧
  (a = 4 → ∃ x : ℝ, x^2 + 3*x = a ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2248_224858


namespace NUMINAMATH_CALUDE_not_perfect_squares_l2248_224801

theorem not_perfect_squares : 
  ¬(∃ n : ℕ, n^2 = 12345678) ∧ 
  ¬(∃ n : ℕ, n^2 = 987654) ∧ 
  ¬(∃ n : ℕ, n^2 = 1234560) ∧ 
  ¬(∃ n : ℕ, n^2 = 98765445) := by
sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l2248_224801


namespace NUMINAMATH_CALUDE_cube_cross_section_area_l2248_224878

/-- The area of a cross-section in a cube -/
theorem cube_cross_section_area (a : ℝ) (h : a > 0) :
  let cube_edge := a
  let face_diagonal := a * Real.sqrt 2
  let space_diagonal := a * Real.sqrt 3
  let cross_section_area := (face_diagonal * space_diagonal) / 2
  cross_section_area = (a^2 * Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_cross_section_area_l2248_224878


namespace NUMINAMATH_CALUDE_area_of_isosceles_right_triangle_l2248_224890

/-- Given a square ABCD and an isosceles right triangle CMN where:
  - The area of square ABCD is 4 square inches
  - MN = NC
  - x is the length of BN
Prove that the area of triangle CMN is (2 - 2x + 0.5x^2)√2 square inches -/
theorem area_of_isosceles_right_triangle (x : ℝ) :
  let abcd_area : ℝ := 4
  let cmn_is_isosceles_right : Prop := true
  let mn_eq_nc : Prop := true
  let bn_length : ℝ := x
  let cmn_area : ℝ := (2 - 2*x + 0.5*x^2) * Real.sqrt 2
  abcd_area = 4 → cmn_is_isosceles_right → mn_eq_nc → cmn_area = (2 - 2*x + 0.5*x^2) * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_area_of_isosceles_right_triangle_l2248_224890


namespace NUMINAMATH_CALUDE_percentOutsideC_eq_61_11_l2248_224816

def gradeScale : List (Char × (Int × Int)) := [
  ('A', (94, 100)),
  ('B', (86, 93)),
  ('C', (76, 85)),
  ('D', (65, 75)),
  ('F', (0, 64))
]

def scores : List Int := [98, 73, 55, 100, 76, 93, 88, 72, 77, 65, 82, 79, 68, 85, 91, 56, 81, 89]

def isOutsideC (score : Int) : Bool :=
  score < 76 || score > 85

def countOutsideC : Nat :=
  scores.filter isOutsideC |>.length

theorem percentOutsideC_eq_61_11 :
  (countOutsideC : ℚ) / scores.length * 100 = 61.11 := by
  sorry

end NUMINAMATH_CALUDE_percentOutsideC_eq_61_11_l2248_224816


namespace NUMINAMATH_CALUDE_sin_graph_transform_l2248_224852

/-- Given a function f : ℝ → ℝ where f x = sin x for all x ∈ ℝ,
    prove that shifting its graph left by π/3 and then halving the x-coordinates
    results in the function g where g x = sin(2x + π/3) for all x ∈ ℝ. -/
theorem sin_graph_transform (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) 
  (h₂ : ∀ x, g x = f (2*x + π/3)) : ∀ x, g x = Real.sin (2*x + π/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transform_l2248_224852


namespace NUMINAMATH_CALUDE_sum_is_odd_l2248_224881

theorem sum_is_odd : Odd (2^1990 + 3^1990 + 7^1990 + 9^1990) := by
  sorry

end NUMINAMATH_CALUDE_sum_is_odd_l2248_224881


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_2718_l2248_224847

/-- Calculate the cost of whitewashing a room's walls --/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ) 
                     (doorHeight doorWidth : ℝ)
                     (windowHeight windowWidth : ℝ)
                     (numWindows : ℕ)
                     (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * numWindows
  (wallArea - doorArea - windowArea) * costPerSquareFoot

/-- Theorem stating the cost of whitewashing for the given room --/
theorem whitewashing_cost_is_2718 :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_2718_l2248_224847


namespace NUMINAMATH_CALUDE_probability_of_condition_l2248_224870

-- Define the bounds for x and y
def x_lower : ℝ := 0
def x_upper : ℝ := 4
def y_lower : ℝ := 0
def y_upper : ℝ := 7

-- Define the condition
def condition (x y : ℝ) : Prop := x + y ≤ 5

-- Define the probability function
def probability : ℝ := sorry

-- Theorem statement
theorem probability_of_condition : probability = 3/7 := by sorry

end NUMINAMATH_CALUDE_probability_of_condition_l2248_224870


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2248_224885

theorem ratio_of_percentages (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.3 * N)
  (hP : P ≠ 0) : 
  M / R = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2248_224885


namespace NUMINAMATH_CALUDE_sum_of_units_digits_of_seven_powers_l2248_224899

def units_digit (n : ℕ) : ℕ := n % 10

def A (n : ℕ) : ℕ := units_digit (7^n)

theorem sum_of_units_digits_of_seven_powers : 
  (Finset.range 2013).sum A + A 2013 = 10067 := by sorry

end NUMINAMATH_CALUDE_sum_of_units_digits_of_seven_powers_l2248_224899


namespace NUMINAMATH_CALUDE_piggy_bank_dime_difference_l2248_224883

theorem piggy_bank_dime_difference :
  ∀ (a b c d : ℕ),
  a + b + c + d = 150 →
  5 * a + 10 * b + 25 * c + 50 * d = 1500 →
  (∃ (b_max b_min : ℕ),
    (∀ (a' b' c' d' : ℕ),
      a' + b' + c' + d' = 150 →
      5 * a' + 10 * b' + 25 * c' + 50 * d' = 1500 →
      b' ≤ b_max ∧ b' ≥ b_min) ∧
    b_max - b_min = 150) :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_dime_difference_l2248_224883


namespace NUMINAMATH_CALUDE_bruce_bags_l2248_224874

/-- The number of bags Bruce can buy with his change after purchasing crayons, books, and calculators. -/
def bags_bruce_can_buy (crayons_packs : ℕ) (crayons_price : ℕ) 
                       (books : ℕ) (books_price : ℕ)
                       (calculators : ℕ) (calculators_price : ℕ)
                       (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayons_price + books * books_price + calculators * calculators_price
  let change := initial_money - total_spent
  change / bag_price

theorem bruce_bags : 
  bags_bruce_can_buy 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bags_l2248_224874


namespace NUMINAMATH_CALUDE_special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l2248_224826

/-- A quadrilateral with specific side lengths and angle properties -/
structure SpecialQuadrilateral where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- Angle properties
  B_obtuse : ℝ
  C_obtuse : ℝ
  sin_C : ℝ
  cos_B : ℝ
  -- Conditions
  AB_eq : AB = 6
  BC_eq : BC = 8
  CD_eq : CD = 15
  B_obtuse_cond : B_obtuse > π / 2
  C_obtuse_cond : C_obtuse > π / 2
  sin_C_eq : sin_C = 4 / 5
  cos_B_eq : cos_B = -4 / 5

/-- The length of side AD in the special quadrilateral is 30 -/
theorem special_quadrilateral_AD_length (q : SpecialQuadrilateral) : ℝ :=
  30

/-- The main theorem stating that for any special quadrilateral, 
    the length of side AD is 30 -/
theorem special_quadrilateral_AD_length_is_30 (q : SpecialQuadrilateral) :
  special_quadrilateral_AD_length q = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l2248_224826


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l2248_224865

theorem equation_satisfies_condition (x y z : ℤ) :
  x = y ∧ y = z + 1 → x^2 - x*y + y^2 - y*z + z^2 - z*x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l2248_224865


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2248_224817

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 - 2*r₁ = 0 ∧ r₂^2 - 2*r₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2248_224817


namespace NUMINAMATH_CALUDE_drama_club_pets_l2248_224887

theorem drama_club_pets (S : Finset ℕ) (R G : Finset ℕ) : 
  Finset.card S = 50 → 
  (∀ s ∈ S, s ∈ R ∨ s ∈ G) → 
  Finset.card R = 35 → 
  Finset.card G = 40 → 
  Finset.card (R ∩ G) = 25 := by
sorry

end NUMINAMATH_CALUDE_drama_club_pets_l2248_224887


namespace NUMINAMATH_CALUDE_triangles_on_ABC_l2248_224828

/-- The number of triangles that can be formed with marked points on the sides of a triangle -/
def num_triangles (points_AB points_BC points_AC : ℕ) : ℕ :=
  let total_points := points_AB + points_BC + points_AC
  let total_combinations := (total_points.choose 3)
  let invalid_combinations := (points_AB.choose 3) + (points_BC.choose 3) + (points_AC.choose 3)
  total_combinations - invalid_combinations

/-- Theorem stating the number of triangles formed with marked points on triangle ABC -/
theorem triangles_on_ABC : num_triangles 12 9 10 = 4071 := by
  sorry

end NUMINAMATH_CALUDE_triangles_on_ABC_l2248_224828


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_exists_l2248_224888

theorem min_value_of_expression (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem equality_condition_exists : ∃ x > 1, x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_exists_l2248_224888


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2248_224859

theorem arithmetic_sequence_difference (n : ℕ) (sum : ℝ) (min max : ℝ) : 
  n = 150 →
  sum = 9000 →
  min = 20 →
  max = 90 →
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L' := avg - (79 * d)
  let G' := avg + (79 * d)
  G' - L' = 6660 / 149 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2248_224859


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2248_224810

def S : Finset ℕ := {1, 2, 3, 4}

def f (k x y z : ℕ) : ℕ := k * x^y - z

theorem max_value_of_expression :
  ∃ (k x y z : ℕ), k ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    f k x y z = 127 ∧
    ∀ (k' x' y' z' : ℕ), k' ∈ S → x' ∈ S → y' ∈ S → z' ∈ S →
      f k' x' y' z' ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2248_224810


namespace NUMINAMATH_CALUDE_like_terms_exponents_l2248_224834

/-- Given two algebraic expressions that are like terms, prove the values of their exponents. -/
theorem like_terms_exponents (a b : ℤ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 2 * x^a * y^2 = k * (-3 * x^3 * y^(b+3))) → 
  (a = 3 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l2248_224834


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2248_224886

theorem max_product_sum_300 : 
  ∀ a b : ℤ, a + b = 300 → a * b ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2248_224886


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2248_224846

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point P(1, -3),
    prove that the line L2 with equation 2x + y + 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 3 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y + 1 = 0
  let P : ℝ × ℝ := (1, -3)
  (L2 P.1 P.2) ∧                           -- L2 passes through P
  (∀ (x1 y1 x2 y2 : ℝ),
    L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 →        -- L1 and L2 are perpendicular
    L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * ((x2 - x1) / (y2 - y1)) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2248_224846


namespace NUMINAMATH_CALUDE_ballCombinations_2005_l2248_224898

/-- The number of ways to choose n balls from red, green, and yellow balls
    such that the number of red balls is even or the number of green balls is odd. -/
def ballCombinations (n : ℕ) : ℕ := sorry

/-- The main theorem stating the number of combinations for 2005 balls. -/
theorem ballCombinations_2005 : ballCombinations 2005 = Nat.choose 2007 2 - Nat.choose 1004 2 := by
  sorry

end NUMINAMATH_CALUDE_ballCombinations_2005_l2248_224898


namespace NUMINAMATH_CALUDE_kim_payroll_time_l2248_224889

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  num_employees : ℕ
  total_routine_time : ℕ

/-- Calculates the time spent per employee on payroll records -/
def payroll_time_per_employee (routine : MorningRoutine) : ℕ :=
  let total_status_update_time := routine.status_update_time_per_employee * routine.num_employees
  let remaining_time := routine.total_routine_time - (routine.coffee_time + total_status_update_time)
  remaining_time / routine.num_employees

/-- Theorem stating that Kim spends 3 minutes per employee updating payroll records -/
theorem kim_payroll_time (kim_routine : MorningRoutine) 
  (h1 : kim_routine.coffee_time = 5)
  (h2 : kim_routine.status_update_time_per_employee = 2)
  (h3 : kim_routine.num_employees = 9)
  (h4 : kim_routine.total_routine_time = 50) :
  payroll_time_per_employee kim_routine = 3 := by
  sorry

#eval payroll_time_per_employee { coffee_time := 5, status_update_time_per_employee := 2, num_employees := 9, total_routine_time := 50 }

end NUMINAMATH_CALUDE_kim_payroll_time_l2248_224889


namespace NUMINAMATH_CALUDE_xy_value_l2248_224821

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2248_224821


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2248_224854

/-- The number of students in the class -/
def num_students : ℕ := 11

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 4

/-- The number of possible sequences of crabapple recipients in a week -/
def possible_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  possible_sequences = 14641 :=
sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2248_224854


namespace NUMINAMATH_CALUDE_function_composition_equality_l2248_224813

/-- Given real numbers a, b, c, d, and functions f and h, 
    prove that f(h(x)) = h(f(x)) for all x if and only if a = c or b = d -/
theorem function_composition_equality 
  (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b) 
  (hh : ∀ x, h x = c * x + d) : 
  (∀ x, f (h x) = h (f x)) ↔ (a = c ∨ b = d) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2248_224813


namespace NUMINAMATH_CALUDE_table_tennis_equipment_theorem_l2248_224845

/-- Represents the price of table tennis equipment and store discounts -/
structure TableTennisEquipment where
  racket_price : ℝ
  ball_price : ℝ
  store_a_discount : ℝ
  store_b_free_balls : ℕ

/-- Proves that given the conditions, the prices are correct and Store A is more cost-effective -/
theorem table_tennis_equipment_theorem (e : TableTennisEquipment)
  (h1 : 2 * e.racket_price + 3 * e.ball_price = 75)
  (h2 : 3 * e.racket_price + 2 * e.ball_price = 100)
  (h3 : e.store_a_discount = 0.1)
  (h4 : e.store_b_free_balls = 10) :
  e.racket_price = 30 ∧
  e.ball_price = 5 ∧
  (1 - e.store_a_discount) * (20 * e.racket_price + 30 * e.ball_price) <
  20 * e.racket_price + (30 - e.store_b_free_balls) * e.ball_price := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_equipment_theorem_l2248_224845


namespace NUMINAMATH_CALUDE_distance_45N_90long_diff_l2248_224829

/-- The spherical distance between two points on Earth --/
def spherical_distance (R : ℝ) (lat1 lat2 long1 long2 : ℝ) : ℝ := sorry

/-- Theorem: The spherical distance between two points at 45°N with 90° longitude difference --/
theorem distance_45N_90long_diff (R : ℝ) :
  spherical_distance R (π/4) (π/4) (π/9) (11*π/18) = π*R/3 := by sorry

end NUMINAMATH_CALUDE_distance_45N_90long_diff_l2248_224829


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2248_224873

theorem simplify_and_evaluate : 
  let a : ℝ := -2
  let b : ℝ := 1
  3 * a + 2 * (a - 1/2 * b^2) - (a - 2 * b^2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2248_224873


namespace NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l2248_224822

theorem smallest_divisor_for_perfect_square (n : ℕ) (h : n = 2880) :
  ∃ (d : ℕ), d > 0 ∧ d.min = 5 ∧ (∃ (k : ℕ), n / d = k * k) ∧
  ∀ (x : ℕ), 0 < x ∧ x < d → ¬∃ (m : ℕ), n / x = m * m :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l2248_224822


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2248_224838

/-- A line l passes through point A(t,0) and is tangent to the curve y = x^2 with an angle of inclination of 45° -/
theorem tangent_line_intersection (t : ℝ) : 
  (∃ (m : ℝ), 
    -- The line passes through (t, 0)
    (t - m) * (m^2 - 0) = (1 - 0) * (0 - t) ∧ 
    -- The line is tangent to y = x^2 at (m, m^2)
    2 * m = 1 ∧ 
    -- The angle of inclination is 45°
    (m^2 - 0) / (m - t) = 1) → 
  t = 1/4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2248_224838


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l2248_224868

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

def sum_five_consecutive_even (k : ℕ) : ℕ := 5 * (2 * k + 2)

theorem smallest_of_five_consecutive_even : 
  ∃ k : ℕ, sum_five_consecutive_even k = sum_first_n_even 25 ∧ 
  2 * k + 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l2248_224868


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_increase_l2248_224875

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem bicycle_helmet_cost_increase 
  (bicycle_cost : ℝ) 
  (helmet_cost : ℝ) 
  (bicycle_increase_percent : ℝ) 
  (helmet_increase_percent : ℝ) 
  (h1 : bicycle_cost = 150)
  (h2 : helmet_cost = 50)
  (h3 : bicycle_increase_percent = 10)
  (h4 : helmet_increase_percent = 20) : 
  ((bicycle_cost * (1 + bicycle_increase_percent / 100) + 
    helmet_cost * (1 + helmet_increase_percent / 100)) - 
   (bicycle_cost + helmet_cost)) / (bicycle_cost + helmet_cost) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_increase_l2248_224875


namespace NUMINAMATH_CALUDE_savings_distribution_child_receives_1680_l2248_224893

/-- Calculates the amount each child receives from a couple's savings --/
theorem savings_distribution (husband_weekly : ℕ) (wife_weekly : ℕ) 
  (months : ℕ) (weeks_per_month : ℕ) (num_children : ℕ) : ℕ :=
  let total_savings := (husband_weekly + wife_weekly) * weeks_per_month * months
  let half_savings := total_savings / 2
  half_savings / num_children

/-- Proves that each child receives $1680 given the specific conditions --/
theorem child_receives_1680 : 
  savings_distribution 335 225 6 4 4 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_savings_distribution_child_receives_1680_l2248_224893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triple_sums_l2248_224892

/-- Given an arithmetic sequence {a_n}, the sequence formed by sums of consecutive triples
    (a_1 + a_2 + a_3), (a_4 + a_5 + a_6), (a_7 + a_8 + a_9), ... is also an arithmetic sequence. -/
theorem arithmetic_sequence_triple_sums (a : ℕ → ℝ) (d : ℝ) 
    (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d' : ℝ, ∀ n, (a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) + d' = 
    (a (3*(n+1) + 1) + a (3*(n+1) + 2) + a (3*(n+1) + 3)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triple_sums_l2248_224892


namespace NUMINAMATH_CALUDE_function_inequality_l2248_224867

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_decreasing : is_decreasing_on f 0 1) :
  f (3/2) < f (1/4) ∧ f (1/4) < f (-1/4) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2248_224867


namespace NUMINAMATH_CALUDE_stock_bond_relationship_l2248_224863

/-- Represents the investment portfolio of Matthew -/
structure Portfolio where
  expensive_stock_value : ℝ  -- Value of the more expensive stock per share
  expensive_stock_shares : ℕ  -- Number of shares of the more expensive stock
  cheap_stock_shares : ℕ     -- Number of shares of the cheaper stock
  bond_face_value : ℝ        -- Face value of the bond
  bond_coupon_rate : ℝ       -- Annual coupon rate of the bond
  bond_discount : ℝ          -- Discount rate at which the bond was purchased
  total_assets : ℝ           -- Total value of assets in stocks and bond
  bond_market_value : ℝ      -- Current market value of the bond

/-- Theorem stating the relationship between the more expensive stock value and the bond market value -/
theorem stock_bond_relationship (p : Portfolio) 
  (h1 : p.expensive_stock_shares = 14)
  (h2 : p.cheap_stock_shares = 26)
  (h3 : p.bond_face_value = 1000)
  (h4 : p.bond_coupon_rate = 0.06)
  (h5 : p.bond_discount = 0.03)
  (h6 : p.total_assets = 2106)
  (h7 : p.expensive_stock_value * p.expensive_stock_shares + 
        (p.expensive_stock_value / 2) * p.cheap_stock_shares + 
        p.bond_market_value = p.total_assets) :
  p.bond_market_value = 2106 - 27 * p.expensive_stock_value := by
  sorry

end NUMINAMATH_CALUDE_stock_bond_relationship_l2248_224863


namespace NUMINAMATH_CALUDE_inequality_proof_l2248_224877

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2248_224877


namespace NUMINAMATH_CALUDE_existence_of_solution_l2248_224820

theorem existence_of_solution (p : Nat) (hp : Nat.Prime p) (hodd : Odd p) :
  ∃ (x y z t : Nat), x^2 + y^2 + z^2 = t * p ∧ t < p ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2248_224820


namespace NUMINAMATH_CALUDE_inequality_solution_quadratic_inequality_l2248_224876

-- Part 1
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic (-4) ∪ Set.Ici (1/2))

theorem inequality_solution :
  ∀ x : ℝ, (9 / (x + 4) ≤ 2) ↔ solution_set x := by sorry

-- Part 2
def valid_k (k : ℝ) : Prop :=
  k ∈ (Set.Iio (-Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2))

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → valid_k k := by sorry

end NUMINAMATH_CALUDE_inequality_solution_quadratic_inequality_l2248_224876


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2248_224812

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 8

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  -- Assuming f(1) < 0 and f(2) > 0
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2248_224812


namespace NUMINAMATH_CALUDE_locus_and_circle_existence_l2248_224806

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 18
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2

-- Define the locus of the center of circle M
def locus (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the circle centered at the origin
def origin_circle (x y : ℝ) : Prop := x^2 + y^2 = 8 / 3

-- Define the tangency and orthogonality conditions
def tangent_intersects_locus (t m n : ℝ × ℝ) : Prop :=
  locus m.1 m.2 ∧ locus n.1 n.2 ∧ 
  (∃ k b : ℝ, t.2 = k * t.1 + b ∧ origin_circle t.1 t.2)

def orthogonal (o m n : ℝ × ℝ) : Prop :=
  (m.1 - o.1) * (n.1 - o.1) + (m.2 - o.2) * (n.2 - o.2) = 0

-- Main theorem
theorem locus_and_circle_existence :
  (∀ x y : ℝ, C₁ x y ∨ C₂ x y → 
    ∃ m : ℝ × ℝ, locus m.1 m.2 ∧
    (∀ t : ℝ × ℝ, origin_circle t.1 t.2 → 
      ∃ n : ℝ × ℝ, tangent_intersects_locus t m n ∧ 
        orthogonal (0, 0) m n)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_circle_existence_l2248_224806


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2248_224802

theorem consecutive_pages_sum (x : ℕ) : x > 0 ∧ x + (x + 1) = 137 → x + 1 = 69 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2248_224802


namespace NUMINAMATH_CALUDE_men_in_first_group_l2248_224837

/-- The number of days taken by the first group to complete the job -/
def days_group1 : ℕ := 15

/-- The number of men in the second group -/
def men_group2 : ℕ := 25

/-- The number of days taken by the second group to complete the job -/
def days_group2 : ℕ := 24

/-- The amount of work done is the product of the number of workers and the number of days they work -/
def work_done (men : ℕ) (days : ℕ) : ℕ := men * days

/-- The theorem stating that the number of men in the first group is 40 -/
theorem men_in_first_group : 
  ∃ (men_group1 : ℕ), 
    men_group1 = 40 ∧ 
    work_done men_group1 days_group1 = work_done men_group2 days_group2 :=
by sorry

end NUMINAMATH_CALUDE_men_in_first_group_l2248_224837


namespace NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l2248_224809

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Finds the number of moles of a specific compound in a list of compounds --/
def findMoles (compounds : List Compound) (name : String) : ℚ :=
  match compounds.find? (fun c => c.name = name) with
  | some compound => compound.moles
  | none => 0

/-- The chemical reaction --/
def reaction : Reaction :=
  { reactants := [
      { name := "NH4Cl", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NH4OH", moles := 1 },
      { name := "NaCl", moles := 1 }
    ]
  }

theorem ammonium_hydroxide_formation :
  findMoles reaction.products "NH4OH" = 1 :=
by sorry

end NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l2248_224809


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2248_224803

/-- Proves that if the average weight of 6 persons increases by 1.5 kg when a person
    weighing 65 kg is replaced by a new person, then the weight of the new person is 74 kg. -/
theorem weight_of_new_person
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (old_weight : ℝ)
  (new_weight : ℝ)
  (h1 : num_persons = 6)
  (h2 : avg_increase = 1.5)
  (h3 : old_weight = 65)
  (h4 : new_weight = num_persons * avg_increase + old_weight) :
  new_weight = 74 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2248_224803


namespace NUMINAMATH_CALUDE_cindy_marbles_l2248_224871

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 :=
by sorry

end NUMINAMATH_CALUDE_cindy_marbles_l2248_224871


namespace NUMINAMATH_CALUDE_cakes_served_during_lunch_l2248_224836

/-- Given that a restaurant served some cakes for lunch and dinner, with a total of 15 cakes served today and 9 cakes served during dinner, prove that 6 cakes were served during lunch. -/
theorem cakes_served_during_lunch (total_cakes dinner_cakes lunch_cakes : ℕ) 
  (h1 : total_cakes = 15)
  (h2 : dinner_cakes = 9)
  (h3 : total_cakes = lunch_cakes + dinner_cakes) :
  lunch_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_during_lunch_l2248_224836


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2248_224818

theorem sum_of_specific_numbers (a b : ℝ) 
  (ha_abs : |a| = 5)
  (hb_abs : |b| = 2)
  (ha_neg : a < 0)
  (hb_pos : b > 0) :
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2248_224818


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l2248_224841

/-- Given a triangle with rational coordinates, its circumcenter has rational coordinates. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l2248_224841


namespace NUMINAMATH_CALUDE_magic_square_sum_l2248_224844

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_sum : a + 27 + b = sum
  row2_sum : 15 + c + d = sum
  row3_sum : 30 + e + 18 = sum
  col1_sum : 30 + 15 + a = sum
  col2_sum : e + c + 27 = sum
  col3_sum : 18 + d + b = sum
  diag1_sum : 30 + c + b = sum
  diag2_sum : 18 + c + a = sum

/-- Theorem: In a 3x3 magic square with the given known numbers, 
    if the sums of all rows, columns, and diagonals are equal, then d + e = 108 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 108 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l2248_224844


namespace NUMINAMATH_CALUDE_candidate_failing_marks_l2248_224861

def max_marks : ℕ := 500
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℕ := 180

theorem candidate_failing_marks :
  (max_marks * passing_percentage).floor - candidate_marks = 45 := by
  sorry

end NUMINAMATH_CALUDE_candidate_failing_marks_l2248_224861


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l2248_224808

theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ (a b : ℝ), 
    -- a and b are the roots of the quadratic equation
    a^2 - 12*a + k = 0 ∧ 
    b^2 - 12*b + k = 0 ∧ 
    -- a and b are equal (isosceles triangle)
    a = b ∧ 
    -- triangle inequality
    3 + a > b ∧ 3 + b > a ∧ a + b > 3 ∧
    -- one side is 3
    3 > 0) → 
  k = 36 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l2248_224808


namespace NUMINAMATH_CALUDE_exchange_impossibility_l2248_224833

theorem exchange_impossibility : 
  ¬∃ (x y z : ℕ), x + y + z = 10 ∧ x + 3*y + 5*z = 25 :=
sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l2248_224833


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l2248_224825

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → b ≤ d) ∧
  a = 20 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l2248_224825


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2248_224850

theorem sum_with_radical_conjugate :
  let a := 15 - Real.sqrt 500
  let radical_conjugate := 15 + Real.sqrt 500
  a + radical_conjugate = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2248_224850


namespace NUMINAMATH_CALUDE_ophelia_age_l2248_224835

/-- Given the following conditions:
  1. In 10 years, Ophelia will be thrice as old as Lennon.
  2. In 10 years, Mike will be twice the age difference between Ophelia and Lennon.
  3. Lennon is currently 8 years old.
  4. Mike is currently 5 years older than Lennon.
Prove that Ophelia's current age is 44 years. -/
theorem ophelia_age (lennon_age : ℕ) (mike_age : ℕ) (ophelia_age : ℕ) :
  lennon_age = 8 →
  mike_age = lennon_age + 5 →
  ophelia_age + 10 = 3 * (lennon_age + 10) →
  mike_age + 10 = 2 * ((ophelia_age + 10) - (lennon_age + 10)) →
  ophelia_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_ophelia_age_l2248_224835


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2248_224860

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 217 ∧ ∀ x, 3 * x^2 - 18 * x + 244 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2248_224860


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2248_224807

/-- The quadratic equation 3x^2 - 4x + 1 = 0 has two distinct real roots -/
theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 - 4 * x₁ + 1 = 0 ∧ 3 * x₂^2 - 4 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2248_224807


namespace NUMINAMATH_CALUDE_price_of_first_oil_l2248_224896

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem price_of_first_oil :
  -- Define the volumes of oils
  let volume_first : ℝ := 10
  let volume_second : ℝ := 5
  -- Define the price of the second oil
  let price_second : ℝ := 68
  -- Define the price of the mixture
  let price_mixture : ℝ := 56
  -- Define the total volume
  let volume_total : ℝ := volume_first + volume_second
  -- The equation that represents the mixing of oils
  ∀ price_first : ℝ,
    volume_first * price_first + volume_second * price_second =
    volume_total * price_mixture →
    -- Prove that the price of the first oil is 50
    price_first = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_of_first_oil_l2248_224896


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_l2248_224832

theorem negation_of_forall_greater_than_one (x : ℝ) : 
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_l2248_224832


namespace NUMINAMATH_CALUDE_max_leftover_fruits_l2248_224872

theorem max_leftover_fruits (A G : ℕ) : 
  (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) ∧ 
  ∃ (A₀ G₀ : ℕ), A₀ % 7 = 6 ∧ G₀ % 7 = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_fruits_l2248_224872


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2248_224851

/-- Given a parabola and a circle with specific properties, 
    prove the distance from the focus of the parabola to its directrix -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (h_p_pos : p > 0)
  (h_intersect : ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧
    A.2^2 = 2*p*A.1 ∧ 
    B.2^2 = 2*p*B.1 ∧
    A.1^2 + (A.2 - 1)^2 = 1 ∧
    B.1^2 + (B.2 - 1)^2 = 1)
  (h_distance : ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧
    A.2^2 = 2*p*A.1 ∧ 
    B.2^2 = 2*p*B.1 ∧
    A.1^2 + (A.2 - 1)^2 = 1 ∧
    B.1^2 + (B.2 - 1)^2 = 1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4/3) :
  p = Real.sqrt 2 / 6 := by
    sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2248_224851


namespace NUMINAMATH_CALUDE_cao_required_proof_l2248_224840

/-- Represents the balanced chemical equation for the reaction between Calcium oxide and Water to form Calcium hydroxide -/
structure BalancedEquation where
  cao : ℕ
  h2o : ℕ
  caoh2 : ℕ
  balanced : cao = h2o ∧ cao = caoh2

/-- Calculates the required amount of Calcium oxide given the amounts of Water and Calcium hydroxide -/
def calcCaORequired (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) : ℕ :=
  if water = hydroxide then water else 0

theorem cao_required_proof (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) 
  (h1 : water = 3) 
  (h2 : hydroxide = 3) : 
  calcCaORequired water hydroxide eq = 3 := by
  sorry

end NUMINAMATH_CALUDE_cao_required_proof_l2248_224840


namespace NUMINAMATH_CALUDE_nancy_water_intake_percentage_l2248_224855

/-- Given Nancy's daily water intake and body weight, calculate the percentage of her body weight she drinks in water. -/
theorem nancy_water_intake_percentage 
  (daily_water_intake : ℝ) 
  (body_weight : ℝ) 
  (h1 : daily_water_intake = 54) 
  (h2 : body_weight = 90) : 
  (daily_water_intake / body_weight) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_nancy_water_intake_percentage_l2248_224855


namespace NUMINAMATH_CALUDE_jake_initial_balloons_l2248_224805

/-- The number of balloons Jake initially brought to the park -/
def jake_initial : ℕ := 2

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional : ℕ := 3

theorem jake_initial_balloons :
  jake_initial = 2 :=
by
  have h1 : allan_balloons = jake_initial + jake_additional + 1 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_jake_initial_balloons_l2248_224805


namespace NUMINAMATH_CALUDE_point_slope_equation_l2248_224879

/-- Given a line with slope 3 passing through the point (-1, -2),
    prove that its point-slope form equation is y + 2 = 3(x + 1) -/
theorem point_slope_equation (x y : ℝ) :
  let slope : ℝ := 3
  let point : ℝ × ℝ := (-1, -2)
  (y - point.2 = slope * (x - point.1)) ↔ (y + 2 = 3 * (x + 1)) := by
sorry

end NUMINAMATH_CALUDE_point_slope_equation_l2248_224879


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_in_range_l2248_224819

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_iff_m_in_range (m : ℝ) :
  (A m ∩ B m).Nonempty ↔ 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_in_range_l2248_224819


namespace NUMINAMATH_CALUDE_largest_non_expressible_l2248_224824

def is_non_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_expressible (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k > 0 ∧ is_non_prime m ∧ n = 47 * k + m

theorem largest_non_expressible : 
  (∀ n > 90, is_expressible n) ∧ 
  ¬is_expressible 90 ∧
  (∀ n < 90, ¬is_expressible n → n < 90) :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l2248_224824


namespace NUMINAMATH_CALUDE_equation_is_parabola_l2248_224894

-- Define the equation
def equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2)

-- Theorem statement
theorem equation_is_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, equation x y ↔ y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l2248_224894
