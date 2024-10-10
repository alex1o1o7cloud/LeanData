import Mathlib

namespace monotonicity_condition_equiv_a_range_l701_70152

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

/-- Theorem stating the equivalence between the monotonicity condition and the range of a -/
theorem monotonicity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ a ∈ Set.Icc (-3) (-2) :=
by sorry

end monotonicity_condition_equiv_a_range_l701_70152


namespace vectors_orthogonal_l701_70185

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 (x : ℝ) : Fin 2 → ℝ := ![x, -3]

theorem vectors_orthogonal :
  let x : ℝ := 15/2
  (vector1 0 * vector2 x 0 + vector1 1 * vector2 x 1 = 0) := by sorry

end vectors_orthogonal_l701_70185


namespace factors_of_180_l701_70180

/-- The number of positive factors of 180 is 18 -/
theorem factors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end factors_of_180_l701_70180


namespace unique_solution_l701_70153

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x^2 - 5 * x + 6 ≠ 0 :=
sorry

end unique_solution_l701_70153


namespace regression_properties_l701_70101

def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

def empirical_regression (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_properties :
  let avg_sales := (List.sum sales_volumes) / (List.length sales_volumes)
  let slope := -4
  let a := 106
  (avg_sales = 80) ∧
  (∀ x₁ x₂, empirical_regression x₂ a - empirical_regression x₁ a = slope * (x₂ - x₁)) ∧
  (empirical_regression 10 a = 66) := by
  sorry

end regression_properties_l701_70101


namespace function_composition_result_l701_70175

theorem function_composition_result (a b : ℝ) :
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 3) →
  a + b = 11 / 3 :=
by sorry

end function_composition_result_l701_70175


namespace quadratic_vertex_form_h_l701_70141

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex form
def vertex_form (n h k : ℝ) (x : ℝ) : ℝ := n * (x - h)^2 + k

-- Theorem statement
theorem quadratic_vertex_form_h (a b c : ℝ) :
  (∃ n k : ℝ, ∀ x : ℝ, 4 * f a b c x = vertex_form n 3 k x) →
  (∀ x : ℝ, f a b c x = 3 * (x - 3)^2 + 6) :=
by sorry

end quadratic_vertex_form_h_l701_70141


namespace fibCoeff_symmetry_l701_70181

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Fibonacci coefficient -/
def fibCoeff (n k : ℕ) : ℚ :=
  if k ≤ n then
    (List.range k).foldl (λ acc i => acc * fib (n - i)) 1 /
    (List.range k).foldl (λ acc i => acc * fib (k - i)) 1
  else 0

/-- Symmetry property of Fibonacci coefficients -/
theorem fibCoeff_symmetry (n k : ℕ) (h : k ≤ n) :
  fibCoeff n k = fibCoeff n (n - k) := by
  sorry

end fibCoeff_symmetry_l701_70181


namespace heptagon_side_sum_l701_70106

/-- Represents a polygon with 7 vertices --/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- Calculates the area of a polygon --/
def area (p : Heptagon) : ℝ := sorry

/-- Calculates the distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem heptagon_side_sum (p : Heptagon) :
  area p = 120 ∧
  distance p.A p.B = 10 ∧
  distance p.B p.C = 15 ∧
  distance p.G p.A = 7 →
  distance p.D p.E + distance p.E p.F = 11.75 := by
  sorry

end heptagon_side_sum_l701_70106


namespace center_coordinates_sum_l701_70172

/-- The sum of the coordinates of the center of the circle given by x^2 + y^2 = 6x - 8y + 18 is -1 -/
theorem center_coordinates_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 8*y + 18) → (∃ a b : ℝ, (x - a)^2 + (y - b)^2 = (x^2 + y^2 - 6*x + 8*y - 18) ∧ a + b = -1) :=
by sorry

end center_coordinates_sum_l701_70172


namespace number_accuracy_l701_70189

-- Define a function to represent the accuracy of a number
def accuracy_place (n : ℝ) : ℕ :=
  sorry

-- Define the number in scientific notation
def number : ℝ := 2.3 * (10 ^ 4)

-- Theorem stating that the number is accurate to the thousands place
theorem number_accuracy :
  accuracy_place number = 3 :=
sorry

end number_accuracy_l701_70189


namespace triangular_number_difference_l701_70199

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2009th triangular numbers is 2010 -/
theorem triangular_number_difference : triangularNumber 2010 - triangularNumber 2009 = 2010 := by
  sorry

end triangular_number_difference_l701_70199


namespace quadratic_no_real_roots_l701_70137

theorem quadratic_no_real_roots : ¬ ∃ (x : ℝ), x^2 + x + 2 = 0 := by
  sorry

end quadratic_no_real_roots_l701_70137


namespace circle_tangent_and_point_condition_l701_70158

-- Define the given points and lines
def point_A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that the center of C is on line l
def center_on_line_l (C : Circle) : Prop :=
  C.center.2 = line_l C.center.1

-- Define the condition that the center of C is on y = x - 1
def center_on_diagonal (C : Circle) : Prop :=
  C.center.2 = C.center.1 - 1

-- Define the tangent line
def is_tangent_line (k b : ℝ) (C : Circle) : Prop :=
  let (cx, cy) := C.center
  (k * cx - cy + b)^2 = (k^2 + 1) * C.radius^2

-- Define the condition |MA| = 2|MO|
def condition_MA_MO (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx^2 + (my - 3)^2) = 4 * (mx^2 + my^2)

-- Main theorem
theorem circle_tangent_and_point_condition (C : Circle) :
  C.radius = 1 →
  center_on_line_l C →
  (center_on_diagonal C →
    (∃ k b, is_tangent_line k b C ∧ (k = 0 ∨ (k = -3/4 ∧ b = 3)))) ∧
  (∃ M, condition_MA_MO M → 
    C.center.1 ≥ 0 ∧ C.center.1 ≤ 12/5) :=
sorry

end circle_tangent_and_point_condition_l701_70158


namespace equation_solution_l701_70183

theorem equation_solution : 
  ∃ (n : ℚ), (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4) ∧ (n = -2/3) := by
  sorry

end equation_solution_l701_70183


namespace tire_pricing_ratio_l701_70162

/-- Represents the daily tire production capacity --/
def daily_production : ℕ := 1000

/-- Represents the daily tire demand --/
def daily_demand : ℕ := 1200

/-- Represents the production cost of each tire in cents --/
def production_cost : ℕ := 25000

/-- Represents the weekly loss in cents due to limited production capacity --/
def weekly_loss : ℕ := 17500000

/-- Represents the ratio of selling price to production cost --/
def selling_price_ratio : ℚ := 3/2

theorem tire_pricing_ratio :
  daily_production = 1000 →
  daily_demand = 1200 →
  production_cost = 25000 →
  weekly_loss = 17500000 →
  selling_price_ratio = 3/2 := by sorry

end tire_pricing_ratio_l701_70162


namespace area_of_folded_rectangle_l701_70111

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A B C D : Point)

/-- Represents the folded configuration -/
structure FoldedConfig :=
  (rect : Rectangle)
  (E F B' C' : Point)

/-- The main theorem -/
theorem area_of_folded_rectangle 
  (config : FoldedConfig) 
  (h1 : config.rect.A.x < config.E.x) -- E is on AB
  (h2 : config.rect.C.x > config.F.x) -- F is on CD
  (h3 : config.E.x - config.rect.B.x < config.rect.C.x - config.F.x) -- BE < CF
  (h4 : config.C'.y = config.rect.A.y) -- C' is on AD
  (h5 : (config.B'.x - config.rect.A.x) * (config.C'.y - config.E.y) = 
        (config.C'.x - config.rect.A.x) * (config.B'.y - config.E.y)) -- ∠AB'C' ≅ ∠B'EA
  (h6 : Real.sqrt ((config.B'.x - config.rect.A.x)^2 + (config.B'.y - config.rect.A.y)^2) = 7) -- AB' = 7
  (h7 : config.E.x - config.rect.B.x = 17) -- BE = 17
  : (config.rect.B.x - config.rect.A.x) * (config.rect.C.y - config.rect.A.y) = 
    (1372 + 833 * Real.sqrt 2) / 6 := by
  sorry

end area_of_folded_rectangle_l701_70111


namespace puzzle_solution_l701_70176

/-- Represents a chip in the puzzle -/
def Chip := Fin 25

/-- Represents the arrangement of chips -/
def Arrangement := Fin 25 → Chip

/-- The initial arrangement of chips -/
def initial_arrangement : Arrangement := sorry

/-- The target arrangement of chips (in order) -/
def target_arrangement : Arrangement := sorry

/-- Represents a swap of two chips -/
def Swap := Chip × Chip

/-- Applies a swap to an arrangement -/
def apply_swap (a : Arrangement) (s : Swap) : Arrangement := sorry

/-- A sequence of swaps -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to an arrangement -/
def apply_swap_sequence (a : Arrangement) (ss : SwapSequence) : Arrangement := sorry

/-- The optimal swap sequence to solve the puzzle -/
def optimal_swap_sequence : SwapSequence := sorry

theorem puzzle_solution :
  apply_swap_sequence initial_arrangement optimal_swap_sequence = target_arrangement ∧
  optimal_swap_sequence.length = 19 := by sorry

end puzzle_solution_l701_70176


namespace bill_omelet_time_l701_70131

/-- Represents the time Bill spends on preparing and cooking omelets -/
def total_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (cheese_grate_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
  pepper_chop_time * num_peppers +
  onion_chop_time * num_onions +
  cheese_grate_time * num_omelets +
  omelet_cook_time * num_omelets

/-- Theorem stating that Bill spends 50 minutes preparing and cooking omelets -/
theorem bill_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end bill_omelet_time_l701_70131


namespace tangent_circles_theorem_l701_70140

/-- Given two circles with centers E and F tangent to segment BD and semicircles with diameters AB, BC, and AC,
    where r1, r2, and r are the radii of semicircles with diameters AB, BC, and AC respectively,
    and l1 and l2 are the radii of circles with centers E and F respectively. -/
theorem tangent_circles_theorem 
  (r1 r2 r l1 l2 : ℝ) 
  (h_r : r = r1 + r2) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ l1 > 0 ∧ l2 > 0) :
  (∃ (distance_E_to_AC : ℝ), distance_E_to_AC = Real.sqrt ((r1 + l1)^2 - (r1 - l1)^2)) ∧ 
  l1 = (r1 * r2) / (r1 + r2) := by
  sorry

end tangent_circles_theorem_l701_70140


namespace car_wash_group_composition_l701_70142

theorem car_wash_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (2 * total : ℚ) / 5 →    -- Initially 40% of the group are girls
  ((girls : ℚ) - 2) / total = 3 / 10 →   -- After changes, 30% of the group are girls
  girls = 8 := by
sorry

end car_wash_group_composition_l701_70142


namespace alien_gems_count_l701_70123

/-- Converts a number from base 6 to base 10 --/
def base6To10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- The number of gems the alien has --/
def alienGems : ℕ := base6To10 2 5 6

theorem alien_gems_count : alienGems = 108 := by
  sorry

end alien_gems_count_l701_70123


namespace stratified_sampling_theorem_l701_70107

/-- Represents the stratified sampling problem --/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  elderly_count : ℕ
  middle_aged_count : ℕ
  young_count : ℕ

/-- Calculates the sample size for a specific group --/
def group_sample_size (s : StratifiedSample) (group_count : ℕ) : ℕ :=
  (group_count * s.sample_size) / s.total_population

/-- Theorem statement for the stratified sampling problem --/
theorem stratified_sampling_theorem (s : StratifiedSample)
  (h1 : s.total_population = s.elderly_count + s.middle_aged_count + s.young_count)
  (h2 : s.total_population = 162)
  (h3 : s.sample_size = 36)
  (h4 : s.elderly_count = 27)
  (h5 : s.middle_aged_count = 54)
  (h6 : s.young_count = 81) :
  group_sample_size s s.elderly_count = 6 ∧
  group_sample_size s s.middle_aged_count = 12 ∧
  group_sample_size s s.young_count = 18 := by
  sorry


end stratified_sampling_theorem_l701_70107


namespace pump_time_correct_l701_70122

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 6

/-- The time it takes to fill the tank with both pump and leak -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

/-- Theorem stating that the pump time is correct given the conditions -/
theorem pump_time_correct : 
  (1 / pump_time - 1 / leak_empty_time) = 1 / fill_time_with_leak := by sorry

end pump_time_correct_l701_70122


namespace smallest_positive_integer_congruence_l701_70159

theorem smallest_positive_integer_congruence : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (5 * x ≡ 18 [MOD 33]) ∧ 
  (x ≡ 4 [MOD 7]) ∧ 
  (∀ (y : ℕ), y > 0 → (5 * y ≡ 18 [MOD 33]) → (y ≡ 4 [MOD 7]) → x ≤ y) ∧
  x = 10 :=
by sorry

end smallest_positive_integer_congruence_l701_70159


namespace toy_position_l701_70186

theorem toy_position (total_toys : ℕ) (position_from_right : ℕ) (position_from_left : ℕ) :
  total_toys = 19 →
  position_from_right = 8 →
  position_from_left = total_toys - (position_from_right - 1) →
  position_from_left = 12 :=
by
  sorry

end toy_position_l701_70186


namespace sqrt_3_times_sqrt_6_minus_2_bounds_l701_70133

theorem sqrt_3_times_sqrt_6_minus_2_bounds : 2 < Real.sqrt 3 * Real.sqrt 6 - 2 ∧ Real.sqrt 3 * Real.sqrt 6 - 2 < 3 := by
  sorry

end sqrt_3_times_sqrt_6_minus_2_bounds_l701_70133


namespace books_not_sold_percentage_l701_70195

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  percentage_not_sold = 66 := by sorry

end books_not_sold_percentage_l701_70195


namespace equation_solution_l701_70163

theorem equation_solution (x : ℝ) : (x^2 - 1) / (x + 1) = 0 ↔ x = 1 := by
  sorry

end equation_solution_l701_70163


namespace inequality_proof_l701_70168

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l701_70168


namespace shirt_sale_price_l701_70182

/-- Given a shirt with a cost price, profit margin, and discount percentage,
    calculate the final sale price. -/
def final_sale_price (cost_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that for a shirt with a cost price of $20, a profit margin of 30%,
    and a discount of 50%, the final sale price is $13. -/
theorem shirt_sale_price :
  final_sale_price 20 0.3 0.5 = 13 := by
sorry

end shirt_sale_price_l701_70182


namespace largest_number_in_block_l701_70149

/-- Represents a 2x3 block of numbers in a 10-column table -/
structure NumberBlock where
  first_number : ℕ
  deriving Repr

/-- The sum of numbers in a 2x3 block -/
def block_sum (block : NumberBlock) : ℕ :=
  6 * block.first_number + 36

theorem largest_number_in_block (block : NumberBlock) 
  (h1 : block.first_number ≥ 1)
  (h2 : block.first_number + 12 ≤ 100)
  (h3 : block_sum block = 480) :
  (block.first_number + 12 = 86) :=
sorry

end largest_number_in_block_l701_70149


namespace fraction_subtraction_l701_70104

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_l701_70104


namespace circle_tangent_sum_l701_70127

def circle_radius_sum : ℝ := 14

theorem circle_tangent_sum (C : ℝ × ℝ) (r : ℝ) :
  (C.1 = r ∧ C.2 = r) →  -- Circle center C is at (r, r)
  ((C.1 - 5)^2 + C.2^2 = (r + 2)^2) →  -- External tangency condition
  (∃ (r1 r2 : ℝ), r1 + r2 = circle_radius_sum ∧ 
    ((C.1 = r1 ∧ C.2 = r1) ∨ (C.1 = r2 ∧ C.2 = r2))) :=
by sorry

end circle_tangent_sum_l701_70127


namespace identify_counterfeit_pile_l701_70154

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : Nat)
  (hasRealCoin : Bool)

/-- Represents the result of weighing two sets of coins -/
inductive WeighResult
  | Equal
  | Unequal

/-- Function to weigh two sets of coins -/
def weigh (pile1 : CoinPile) (pile2 : CoinPile) (count : Nat) : WeighResult :=
  sorry

/-- Theorem stating that it's possible to identify the all-counterfeit pile -/
theorem identify_counterfeit_pile 
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)
  (h1 : pile1.count = 15)
  (h2 : pile2.count = 19)
  (h3 : pile3.count = 25)
  (h4 : pile1.hasRealCoin ∨ pile2.hasRealCoin ∨ pile3.hasRealCoin)
  (h5 : ¬(pile1.hasRealCoin ∧ pile2.hasRealCoin) ∧ 
        ¬(pile1.hasRealCoin ∧ pile3.hasRealCoin) ∧ 
        ¬(pile2.hasRealCoin ∧ pile3.hasRealCoin)) :
  ∃ (p : CoinPile), p ∈ [pile1, pile2, pile3] ∧ ¬p.hasRealCoin :=
sorry

end identify_counterfeit_pile_l701_70154


namespace hospital_transfer_l701_70164

theorem hospital_transfer (x : ℝ) (x_pos : x > 0) : 
  let wing_a := x
  let wing_b := 2 * x
  let wing_c := 3 * x
  let occupied_a := (1/3) * wing_a
  let occupied_b := (1/2) * wing_b
  let occupied_c := (1/4) * wing_c
  let max_capacity_b := (3/4) * wing_b
  let max_capacity_c := (5/6) * wing_c
  occupied_a + occupied_b ≤ max_capacity_b →
  (occupied_a + occupied_b) / wing_b = 2/3 ∧ occupied_c / wing_c = 1/4 :=
by sorry

end hospital_transfer_l701_70164


namespace initial_shells_count_l701_70105

/-- The number of shells Ed found at the beach -/
def ed_shells : ℕ := 13

/-- The number of shells Jacob found at the beach -/
def jacob_shells : ℕ := ed_shells + 2

/-- The total number of shells after collecting -/
def total_shells : ℕ := 30

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := total_shells - (ed_shells + jacob_shells)

theorem initial_shells_count : initial_shells = 2 := by
  sorry

end initial_shells_count_l701_70105


namespace unique_a_value_l701_70156

def A (a : ℝ) : Set ℝ := {a - 2, 2 * a^2 + 5 * a, 12}

theorem unique_a_value : ∀ a : ℝ, -3 ∈ A a ↔ a = -3/2 := by sorry

end unique_a_value_l701_70156


namespace sum_256_64_base_8_l701_70174

def to_base_8 (n : ℕ) : ℕ := sorry

theorem sum_256_64_base_8 : 
  to_base_8 (256 + 64) = 500 := by sorry

end sum_256_64_base_8_l701_70174


namespace second_derivative_zero_l701_70139

open Real

/-- Given a differentiable function f and a point x₀ such that 
    the limit of (f(x₀) - f(x₀ + 2Δx)) / Δx as Δx approaches 0 is 2,
    prove that the second derivative of f at x₀ is 0. -/
theorem second_derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f x₀ - f (x₀ + 2*Δx)) / Δx) - 2| < ε) :
  deriv (deriv f) x₀ = 0 := by
  sorry

end second_derivative_zero_l701_70139


namespace cubic_root_function_l701_70129

theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 := by
sorry

end cubic_root_function_l701_70129


namespace rectangle_perimeter_l701_70165

/-- Theorem: For a rectangle with length L and width W, if L/W = 5/2 and L * W = 4000, 
    then the perimeter 2L + 2W = 280. -/
theorem rectangle_perimeter (L W : ℝ) 
    (h1 : L / W = 5 / 2) 
    (h2 : L * W = 4000) : 
  2 * L + 2 * W = 280 := by
  sorry

end rectangle_perimeter_l701_70165


namespace shortest_major_axis_ellipse_l701_70160

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the equation of the ellipse we want to prove
def target_ellipse (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- State the theorem
theorem shortest_major_axis_ellipse :
  ∃ (M : ℝ × ℝ), 
    line_l M.1 M.2 ∧ 
    (∀ (E : ℝ × ℝ → Prop), 
      (∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
        (∀ (x y : ℝ), E (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
        E M ∧ 
        (∀ (x y : ℝ), E (x, y) → 
          Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) + 
          Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a)) →
      (∀ (x y : ℝ), E (x, y) → target_ellipse x y)) :=
sorry

end shortest_major_axis_ellipse_l701_70160


namespace probability_heart_joker_value_l701_70114

/-- A deck of cards with 54 cards total, including 13 hearts and 2 jokers -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (jokers : Nat)
  (h_total : total = 54)
  (h_hearts : hearts = 13)
  (h_jokers : jokers = 2)

/-- The probability of drawing a heart first and a joker second from the deck -/
def probability_heart_joker (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total * d.jokers / (d.total - 1)

/-- Theorem stating the probability of drawing a heart first and a joker second -/
theorem probability_heart_joker_value (d : Deck) :
  probability_heart_joker d = 13 / 1419 := by
  sorry

#eval (13 : ℚ) / 1419

end probability_heart_joker_value_l701_70114


namespace remove_parentheses_first_step_l701_70143

/-- Represents the steps in solving a linear equation -/
inductive SolvingStep
  | RemoveParentheses
  | EliminateDenominator
  | MoveTerms
  | CombineTerms

/-- Represents a linear equation -/
structure LinearEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- The given equation: 2x + 3(2x - 1) = 16 - (x + 1) -/
def givenEquation : LinearEquation :=
  { lhs := λ x ↦ 2*x + 3*(2*x - 1)
    rhs := λ x ↦ 16 - (x + 1) }

/-- The first step in solving the given linear equation -/
def firstSolvingStep (eq : LinearEquation) : SolvingStep := sorry

/-- Theorem stating that removing parentheses is the first step for the given equation -/
theorem remove_parentheses_first_step :
  firstSolvingStep givenEquation = SolvingStep.RemoveParentheses := sorry

end remove_parentheses_first_step_l701_70143


namespace project_completion_time_l701_70150

theorem project_completion_time (team_a_time team_b_time team_c_time total_time : ℝ) 
  (h1 : team_a_time = 10)
  (h2 : team_b_time = 15)
  (h3 : team_c_time = 20)
  (h4 : total_time = 6) :
  (1 - (1 / team_b_time + 1 / team_c_time) * total_time) / (1 / team_a_time) = 3 := by
  sorry

#check project_completion_time

end project_completion_time_l701_70150


namespace pet_store_siamese_cats_l701_70117

/-- The number of Siamese cats initially in the pet store. -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store. -/
def initial_house_cats : ℕ := 5

/-- The number of cats sold during the sale. -/
def cats_sold : ℕ := 10

/-- The number of cats left after the sale. -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is correct. -/
theorem pet_store_siamese_cats :
  initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
by sorry

end pet_store_siamese_cats_l701_70117


namespace problem_1_problem_2_problem_3_problem_4_l701_70177

-- Problem 1
theorem problem_1 : -7 - |(-9)| - (-11) - 3 = -8 := by sorry

-- Problem 2
theorem problem_2 : 5.6 + (-0.9) + 4.4 + (-8.1) = 1 := by sorry

-- Problem 3
theorem problem_3 : (-1/6 : ℚ) + (1/3 : ℚ) + (-1/12 : ℚ) = 1/12 := by sorry

-- Problem 4
theorem problem_4 : (2/5 : ℚ) - |(-1.5 : ℚ)| - (2.25 : ℚ) - (-2.75 : ℚ) = -0.6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l701_70177


namespace frank_bags_theorem_l701_70121

/-- Given that Frank has a total number of candy pieces and puts an equal number of pieces in each bag, 
    calculate the number of bags used. -/
def bags_used (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem stating that Frank used 2 bags given the problem conditions -/
theorem frank_bags_theorem (total_candy : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : 
  bags_used total_candy candy_per_bag = 2 := by
  sorry

end frank_bags_theorem_l701_70121


namespace five_balls_four_boxes_l701_70138

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end five_balls_four_boxes_l701_70138


namespace adult_ticket_cost_adult_ticket_cost_is_seven_l701_70126

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let adult_ticket_cost := (total_revenue - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost

#check adult_ticket_cost 4 900 5100 400 = 7

theorem adult_ticket_cost_is_seven :
  adult_ticket_cost 4 900 5100 400 = 7 := by
  sorry

end adult_ticket_cost_adult_ticket_cost_is_seven_l701_70126


namespace triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l701_70147

/-- A triangle with sides in arithmetic progression including its semiperimeter -/
structure TriangleWithArithmeticSides where
  /-- The common difference of the arithmetic progression -/
  d : ℝ
  /-- The middle term of the arithmetic progression -/
  a : ℝ
  /-- Ensures that the sides are positive -/
  d_pos : 0 < d
  a_pos : 0 < a
  /-- Ensures that the triangle inequality holds -/
  triangle_ineq : 2 * d < a

theorem triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff 
  (t : TriangleWithArithmeticSides) : 
  /- The triangle is right-angled -/
  (3 * t.a / 4) ^ 2 + (4 * t.a / 4) ^ 2 = (5 * t.a / 4) ^ 2 ∧ 
  /- The common difference equals the inradius -/
  t.d = (3 * t.a / 4 + 4 * t.a / 4 - 5 * t.a / 4) / 2 := by
  sorry

end triangle_with_arithmetic_sides_is_right_angled_and_inradius_equals_diff_l701_70147


namespace impossible_valid_arrangement_l701_70124

/-- Represents the colors of chips -/
inductive Color
| Blue
| Red
| Green

/-- Represents a circular arrangement of chips -/
def CircularArrangement := List Color

/-- Represents a swap operation -/
inductive SwapOperation
| BlueRed
| BlueGreen

/-- Initial arrangement of chips -/
def initial_arrangement : CircularArrangement :=
  (List.replicate 40 Color.Blue) ++ (List.replicate 30 Color.Red) ++ (List.replicate 20 Color.Green)

/-- Checks if an arrangement has no adjacent chips of the same color -/
def is_valid_arrangement (arr : CircularArrangement) : Bool :=
  sorry

/-- Applies a swap operation to an arrangement -/
def apply_swap (arr : CircularArrangement) (op : SwapOperation) : CircularArrangement :=
  sorry

/-- Theorem stating that it's impossible to achieve a valid arrangement -/
theorem impossible_valid_arrangement :
  ∀ (ops : List SwapOperation),
    let final_arrangement := ops.foldl apply_swap initial_arrangement
    ¬ (is_valid_arrangement final_arrangement) :=
  sorry

end impossible_valid_arrangement_l701_70124


namespace sum_le_one_plus_product_l701_70169

theorem sum_le_one_plus_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≤ 1 + a * b :=
by sorry

end sum_le_one_plus_product_l701_70169


namespace geometric_sequence_sum_l701_70179

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_3 : a 3 = 4) (h_6 : a 6 = 1/2) : a 4 + a 5 = 3 := by
  sorry

end geometric_sequence_sum_l701_70179


namespace mikes_initial_cards_l701_70196

theorem mikes_initial_cards (initial_cards current_cards cards_sold : ℕ) :
  current_cards = 74 →
  cards_sold = 13 →
  initial_cards = current_cards + cards_sold →
  initial_cards = 87 := by
sorry

end mikes_initial_cards_l701_70196


namespace all_choose_same_house_probability_l701_70197

/-- The probability that all 3 persons choose the same house when there are 3 houses
    and each person independently chooses a house with equal probability. -/
theorem all_choose_same_house_probability :
  let num_houses : ℕ := 3
  let num_persons : ℕ := 3
  let prob_choose_house : ℚ := 1 / 3
  (num_houses * (prob_choose_house ^ num_persons)) = 1 / 9 :=
by sorry

end all_choose_same_house_probability_l701_70197


namespace diamond_equation_solution_l701_70134

def diamond (a b : ℤ) : ℤ := 2 * a + b

theorem diamond_equation_solution :
  ∃ y : ℤ, diamond 4 (diamond 3 y) = 17 ∧ y = 3 :=
by
  sorry

end diamond_equation_solution_l701_70134


namespace cantaloupes_left_total_l701_70198

/-- The total number of cantaloupes left after each person's changes -/
def total_cantaloupes_left (fred_initial fred_eaten tim_initial tim_lost susan_initial susan_given nancy_initial nancy_traded : ℕ) : ℕ :=
  (fred_initial - fred_eaten) + (tim_initial - tim_lost) + (susan_initial - susan_given) + (nancy_initial - nancy_traded)

/-- Theorem stating the total number of cantaloupes left is 138 -/
theorem cantaloupes_left_total :
  total_cantaloupes_left 38 4 44 7 57 10 25 5 = 138 := by
  sorry

end cantaloupes_left_total_l701_70198


namespace joe_team_wins_l701_70178

/-- Represents the number of points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- Represents the number of draws Joe's team had -/
def joe_team_draws : ℕ := 3

/-- Represents the number of wins the first-place team had -/
def first_place_wins : ℕ := 2

/-- Represents the number of ties the first-place team had -/
def first_place_ties : ℕ := 2

/-- Represents the point difference between the first-place team and Joe's team -/
def point_difference : ℕ := 2

/-- Theorem stating that Joe's team won exactly one game -/
theorem joe_team_wins : ℕ := by
  sorry

end joe_team_wins_l701_70178


namespace system_of_equations_solution_l701_70146

theorem system_of_equations_solution :
  ∃! (x y z u : ℤ),
    x + y + z = 15 ∧
    x + y + u = 16 ∧
    x + z + u = 18 ∧
    y + z + u = 20 ∧
    x = 3 ∧ y = 5 ∧ z = 7 ∧ u = 8 := by
  sorry

end system_of_equations_solution_l701_70146


namespace equal_split_contribution_l701_70115

def earnings : List ℝ := [18, 22, 30, 38, 45]

theorem equal_split_contribution (total : ℝ) (equal_share : ℝ) :
  total = earnings.sum →
  equal_share = total / 5 →
  45 - equal_share = 14.4 := by
  sorry

end equal_split_contribution_l701_70115


namespace work_multiple_l701_70118

/-- If a person can complete one unit of work in 5 days, and takes 15 days to complete 
    a certain amount of the same type of work, then the amount of work completed in 15 days 
    is 3 times the original unit of work. -/
theorem work_multiple (original_days : ℕ) (new_days : ℕ) (work_multiple : ℚ) :
  original_days = 5 →
  new_days = 15 →
  work_multiple = (new_days : ℚ) / (original_days : ℚ) →
  work_multiple = 3 := by
sorry

end work_multiple_l701_70118


namespace line_passes_through_fixed_point_l701_70108

/-- A line that always passes through a fixed point regardless of the parameter m -/
def line (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
sorry

end line_passes_through_fixed_point_l701_70108


namespace hero_qin_equivalence_l701_70113

theorem hero_qin_equivalence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c)) =
  Real.sqrt ((1 / 4) * (a^2 * b^2 - ((a^2 + b^2 + c^2) / 2)^2)) :=
by sorry

end hero_qin_equivalence_l701_70113


namespace solve_linear_equation_l701_70151

theorem solve_linear_equation (x : ℝ) : (3 * x - 8 = -2 * x + 17) → x = 5 := by
  sorry

end solve_linear_equation_l701_70151


namespace ceiling_floor_expression_l701_70136

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end ceiling_floor_expression_l701_70136


namespace at_least_one_solution_l701_70110

-- Define the polynomials
variable (P S T : ℂ → ℂ)

-- Define the properties of the polynomials
axiom P_degree : ∃ (a b c : ℂ), ∀ z, P z = z^3 + a*z^2 + b*z + 4
axiom S_degree : ∃ (a b c d : ℂ), ∀ z, S z = z^4 + a*z^3 + b*z^2 + c*z + 5
axiom T_degree : ∃ (a b c d e f g : ℂ), ∀ z, T z = z^7 + a*z^6 + b*z^5 + c*z^4 + d*z^3 + e*z^2 + f*z + 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, P z * S z = T z :=
sorry

end at_least_one_solution_l701_70110


namespace matrix_equality_implies_fraction_l701_70119

/-- Given two 2x2 matrices A and B, where A is [[2, 5], [3, 7]] and B is [[a, b], [c, d]],
    if AB = BA and 5b ≠ c, then (a - d) / (c - 5b) = 6c / (5a + 22c) -/
theorem matrix_equality_implies_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (5 * b ≠ c) → 
  (a - d) / (c - 5 * b) = 6 * c / (5 * a + 22 * c) := by
  sorry

end matrix_equality_implies_fraction_l701_70119


namespace range_of_f_l701_70128

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 1 5, f x = y ∧
  ∀ x ∈ Set.Icc 1 5, f x ∈ Set.Icc 1 10 :=
by sorry

end range_of_f_l701_70128


namespace prev_geng_yin_year_is_1950_l701_70188

/-- The number of Heavenly Stems in the Ganzhi system -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches in the Ganzhi system -/
def earthly_branches : ℕ := 12

/-- The year we know to be a Geng-Yin year -/
def known_geng_yin_year : ℕ := 2010

/-- The function to calculate the previous Geng-Yin year -/
def prev_geng_yin_year (current_year : ℕ) : ℕ :=
  current_year - Nat.lcm heavenly_stems earthly_branches

theorem prev_geng_yin_year_is_1950 :
  prev_geng_yin_year known_geng_yin_year = 1950 := by
  sorry

#eval prev_geng_yin_year known_geng_yin_year

end prev_geng_yin_year_is_1950_l701_70188


namespace coeff_x6_eq_30_implies_a_eq_2_l701_70193

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 -/
def coeff_x6 (a : ℚ) : ℚ := (binomial 10 3 : ℚ) - a * (binomial 10 2 : ℚ)

/-- Theorem: If the coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 is 30, then a = 2 -/
theorem coeff_x6_eq_30_implies_a_eq_2 :
  coeff_x6 2 = 30 :=
by sorry

end coeff_x6_eq_30_implies_a_eq_2_l701_70193


namespace discount_sales_income_increase_l701_70190

/-- Proves that a 10% discount with 15% increase in sales volume results in 3.5% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.15) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + sales_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.035 := by
sorry

end discount_sales_income_increase_l701_70190


namespace optimal_solution_l701_70192

/-- Represents a container with its size and count -/
structure Container where
  size : Nat
  count : Nat

/-- Calculates the total volume of water from a list of containers -/
def totalVolume (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.size * c.count) 0

/-- Calculates the total number of trips for a list of containers -/
def totalTrips (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.count) 0

/-- Theorem stating that the given solution is optimal -/
theorem optimal_solution (initialVolume timeLimit : Nat) : 
  let targetVolume : Nat := 823
  let containers : List Container := [
    { size := 8, count := 18 },
    { size := 2, count := 1 },
    { size := 5, count := 1 }
  ]
  (initialVolume = 676) →
  (timeLimit = 45) →
  (totalVolume containers + initialVolume ≥ targetVolume) ∧
  (totalTrips containers ≤ timeLimit) ∧
  (∀ (otherContainers : List Container),
    (totalVolume otherContainers + initialVolume ≥ targetVolume) →
    (totalTrips otherContainers ≤ timeLimit) →
    (totalTrips containers ≤ totalTrips otherContainers)) :=
by sorry


end optimal_solution_l701_70192


namespace count_juggling_sequences_l701_70103

/-- The number of juggling sequences of length n with exactly 1 ball -/
def jugglingSequences (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The number of juggling sequences of length n with exactly 1 ball is 2^n - 1 -/
theorem count_juggling_sequences (n : ℕ) : 
  jugglingSequences n = 2^n - 1 := by
  sorry

end count_juggling_sequences_l701_70103


namespace simplify_fraction_l701_70145

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) :
  (x^2 / (x - 2)) - (4 / (x - 2)) = x + 2 := by
  sorry

end simplify_fraction_l701_70145


namespace total_flowers_sold_l701_70166

/-- Represents the number of flowers in a bouquet -/
def bouquet_size : ℕ := 12

/-- Represents the total number of bouquets sold -/
def total_bouquets : ℕ := 20

/-- Represents the number of rose bouquets sold -/
def rose_bouquets : ℕ := 10

/-- Represents the number of daisy bouquets sold -/
def daisy_bouquets : ℕ := 10

/-- Theorem stating that the total number of flowers sold is 240 -/
theorem total_flowers_sold : 
  bouquet_size * rose_bouquets + bouquet_size * daisy_bouquets = 240 :=
by sorry

end total_flowers_sold_l701_70166


namespace sum_of_consecutive_odd_primes_l701_70148

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

theorem sum_of_consecutive_odd_primes (p q : ℕ) (h : ConsecutivePrimes p q) (hp_odd : Odd p) (hq_odd : Odd q) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p + q = 2 * a * b :=
sorry

end sum_of_consecutive_odd_primes_l701_70148


namespace sum_of_baby_ages_theorem_l701_70194

/-- Calculates the sum of ages of baby animals in 5 years -/
def sum_of_baby_ages_in_5_years (lioness_age : ℕ) : ℕ :=
  let hyena_age := lioness_age / 2
  let baby_lioness_age := lioness_age / 2
  let baby_hyena_age := hyena_age / 2
  (baby_lioness_age + 5) + (baby_hyena_age + 5)

/-- Theorem stating that the sum of ages of baby animals in 5 years is 19 -/
theorem sum_of_baby_ages_theorem :
  sum_of_baby_ages_in_5_years 12 = 19 := by
  sorry

end sum_of_baby_ages_theorem_l701_70194


namespace three_number_sum_l701_70173

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by sorry

end three_number_sum_l701_70173


namespace complex_calculation_l701_70102

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a + 4*b = 17 - 6*I := by sorry

end complex_calculation_l701_70102


namespace factorial_product_not_perfect_power_l701_70112

-- Define the factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the product of factorials from 1 to n
def factorial_product (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define a function to check if a number is a perfect power greater than 1
def is_perfect_power (n : ℕ) : Prop :=
  ∃ (base exponent : ℕ), base > 1 ∧ exponent > 1 ∧ base ^ exponent = n

-- State the theorem
theorem factorial_product_not_perfect_power :
  ¬ (is_perfect_power (factorial_product 2022)) :=
sorry

end factorial_product_not_perfect_power_l701_70112


namespace inequality_proof_l701_70157

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end inequality_proof_l701_70157


namespace absolute_value_equation_solutions_l701_70155

theorem absolute_value_equation_solutions :
  let S : Set ℝ := {x | |x + 1| * |x - 2| * |x + 3| * |x - 4| = |x - 1| * |x + 2| * |x - 3| * |x + 4|}
  S = {0, Real.sqrt 7, -Real.sqrt 7, 
       Real.sqrt ((13 + Real.sqrt 73) / 2), -Real.sqrt ((13 + Real.sqrt 73) / 2),
       Real.sqrt ((13 - Real.sqrt 73) / 2), -Real.sqrt ((13 - Real.sqrt 73) / 2)} := by
  sorry


end absolute_value_equation_solutions_l701_70155


namespace max_work_hours_l701_70144

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  regularHours : ℕ := 20
  regularRate : ℚ := 8
  overtimeRate : ℚ := 10
  maxEarnings : ℚ := 760

/-- Calculates the total hours worked given regular and overtime hours --/
def totalHours (regular : ℕ) (overtime : ℕ) : ℕ :=
  regular + overtime

/-- Calculates the total earnings given regular and overtime hours --/
def totalEarnings (schedule : WorkSchedule) (overtime : ℕ) : ℚ :=
  (schedule.regularHours : ℚ) * schedule.regularRate + (overtime : ℚ) * schedule.overtimeRate

/-- Theorem: The maximum number of hours Mary can work in a week is 80 --/
theorem max_work_hours (schedule : WorkSchedule) : 
  ∃ (overtime : ℕ), 
    totalHours schedule.regularHours overtime = 80 ∧ 
    totalEarnings schedule overtime ≤ schedule.maxEarnings ∧
    ∀ (h : ℕ), totalEarnings schedule h ≤ schedule.maxEarnings → 
      totalHours schedule.regularHours h ≤ 80 :=
by
  sorry

end max_work_hours_l701_70144


namespace decimal_93_to_binary_l701_70184

def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_93_to_binary :
  decimalToBinary 93 = [1, 0, 1, 1, 1, 0, 1] := by
  sorry

end decimal_93_to_binary_l701_70184


namespace smallest_sum_of_sequence_l701_70120

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C : ℚ) / B = 7 / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' : ℚ) / B' = 7 / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end smallest_sum_of_sequence_l701_70120


namespace zombies_less_than_threshold_days_l701_70167

/-- The number of zombies in the mall today -/
def current_zombies : ℕ := 480

/-- The threshold number of zombies -/
def threshold : ℕ := 50

/-- The function that calculates the number of zombies n days ago -/
def zombies_n_days_ago (n : ℕ) : ℚ :=
  current_zombies / (2 ^ n : ℚ)

/-- The theorem stating that 4 days ago is when there were less than 50 zombies -/
theorem zombies_less_than_threshold_days : 
  (∃ (n : ℕ), zombies_n_days_ago n < threshold) ∧ 
  (∀ (m : ℕ), m < 4 → zombies_n_days_ago m ≥ threshold) ∧
  zombies_n_days_ago 4 < threshold :=
sorry

end zombies_less_than_threshold_days_l701_70167


namespace regular_polygon_properties_l701_70161

theorem regular_polygon_properties :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
  n > 2 →
  interior_angle - exterior_angle = 90 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = n * interior_angle →
  (n - 2) * 180 = 1080 ∧ n = 8 := by
sorry

end regular_polygon_properties_l701_70161


namespace dividend_calculation_l701_70135

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 := by
sorry

end dividend_calculation_l701_70135


namespace elizabeth_stickers_l701_70109

/-- Represents the number of stickers Elizabeth placed on each water bottle. -/
def stickers_per_bottle (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (total_stickers : ℕ) : ℕ :=
  total_stickers / (initial_bottles - lost_bottles - stolen_bottles)

/-- Theorem: Elizabeth placed 3 stickers on each remaining water bottle. -/
theorem elizabeth_stickers :
  stickers_per_bottle 10 2 1 21 = 3 := by
  sorry

end elizabeth_stickers_l701_70109


namespace sum_of_angles_three_triangles_l701_70132

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that the sum of angles in a triangle is 180°
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define three non-overlapping triangles
variable (A B C : Triangle)

-- Assume each triangle is valid
variable (hA : is_valid_triangle A)
variable (hB : is_valid_triangle B)
variable (hC : is_valid_triangle C)

-- Theorem: The sum of all angles in the three triangles is 540°
theorem sum_of_angles_three_triangles :
  A.angle1 + A.angle2 + A.angle3 +
  B.angle1 + B.angle2 + B.angle3 +
  C.angle1 + C.angle2 + C.angle3 = 540 :=
by sorry

end sum_of_angles_three_triangles_l701_70132


namespace min_n_for_constant_term_l701_70171

theorem min_n_for_constant_term (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), (n.choose k) * (-1)^k * x^(n - 8*k) = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬(∃ (k : ℕ), (m.choose k) * (-1)^k * x^(m - 8*k) = 1)) ∧
  n = 8 :=
sorry

end min_n_for_constant_term_l701_70171


namespace dice_arithmetic_progression_probability_l701_70191

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def is_arithmetic_progression (nums : Finset ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i ∈ nums, ∃ k : ℕ, i = a + k * d

def favorable_outcomes : Finset (Finset ℕ) :=
  {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}

theorem dice_arithmetic_progression_probability :
  (Finset.card favorable_outcomes) / (faces_per_die ^ num_dice : ℚ) = 1 / 432 := by
  sorry

end dice_arithmetic_progression_probability_l701_70191


namespace fry_costs_60_cents_l701_70125

-- Define the costs in cents
def burger_cost : ℕ := 80
def soda_cost : ℕ := 60

-- Define the total costs of Alice's and Bill's purchases in cents
def alice_total : ℕ := 420
def bill_total : ℕ := 340

-- Define the function to calculate the cost of a fry
def fry_cost : ℕ :=
  alice_total - 3 * burger_cost - 2 * soda_cost

-- Theorem to prove
theorem fry_costs_60_cents :
  fry_cost = 60 ∧
  2 * burger_cost + soda_cost + 2 * fry_cost = bill_total :=
by sorry

end fry_costs_60_cents_l701_70125


namespace gcd_lcm_power_equation_l701_70130

/-- Given positive integers m and n, if m^(gcd m n) = n^(lcm m n), then m = 1 and n = 1 -/
theorem gcd_lcm_power_equation (m n : ℕ+) :
  m ^ (Nat.gcd m.val n.val) = n ^ (Nat.lcm m.val n.val) → m = 1 ∧ n = 1 := by
  sorry

end gcd_lcm_power_equation_l701_70130


namespace total_insects_l701_70170

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (stones : ℕ) (ants_per_stone : ℕ) 
  (bees : ℕ) (flowers : ℕ) : 
  leaves = 345 → 
  ladybugs_per_leaf = 267 → 
  stones = 178 → 
  ants_per_stone = 423 → 
  bees = 498 → 
  flowers = 6 → 
  leaves * ladybugs_per_leaf + stones * ants_per_stone + bees = 167967 := by
  sorry

end total_insects_l701_70170


namespace line_segment_has_measurable_length_l701_70187

-- Define the characteristics of geometric objects
structure GeometricObject where
  has_endpoints : Bool
  is_infinite : Bool

-- Define specific geometric objects
def line : GeometricObject :=
  { has_endpoints := false, is_infinite := true }

def ray : GeometricObject :=
  { has_endpoints := true, is_infinite := true }

def line_segment : GeometricObject :=
  { has_endpoints := true, is_infinite := false }

-- Define a property for having measurable length
def has_measurable_length (obj : GeometricObject) : Prop :=
  obj.has_endpoints ∧ ¬obj.is_infinite

-- Theorem statement
theorem line_segment_has_measurable_length :
  has_measurable_length line_segment ∧
  ¬has_measurable_length line ∧
  ¬has_measurable_length ray :=
sorry

end line_segment_has_measurable_length_l701_70187


namespace solve_equation_l701_70116

theorem solve_equation (x : ℝ) (h : 5 - 5 / x = 4 + 4 / x) : x = 9 := by
  sorry

end solve_equation_l701_70116


namespace line_proof_l701_70100

-- Define the lines
def line1 (x y : ℝ) : Prop := 4 * x + 2 * y + 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 2 * y + 9 = 0
def line3 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def result_line (x y : ℝ) : Prop := 4 * x - 2 * y + 11 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular
      ((4 : ℝ) / 2) -- Slope of result_line
      ((-1 : ℝ) / 2) -- Slope of line3
  := by sorry

end line_proof_l701_70100
