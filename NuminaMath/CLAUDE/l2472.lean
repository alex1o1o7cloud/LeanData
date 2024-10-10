import Mathlib

namespace intersection_parallel_line_l2472_247225

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i j : ℝ) :
  (∃ x y, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g * x + h * y + i = 0) →                                -- Third line
  (∃ k l m : ℝ, k ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      k * (g * x + h * y) + l * x + m * y + j = 0)) →      -- Parallel condition
  (∃ p q r : ℝ, p ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      p * x + q * y + r = 0) ∧                             -- Line through intersection
    ∃ s, p = s * g ∧ q = s * h) →                          -- Parallel to third line
  ∃ t, 2 * x + y = t                                       -- Resulting equation
  := by sorry

end intersection_parallel_line_l2472_247225


namespace associated_equation_part1_associated_equation_part2_associated_equation_part3_l2472_247246

-- Part 1
theorem associated_equation_part1 (x : ℝ) : 
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) → 
  (x - (3 * x + 1) = -5) :=
sorry

-- Part 2
theorem associated_equation_part2 (x : ℤ) : 
  (x - 1/4 < 1 ∧ 4 + 2 * x > -7 * x + 5) → 
  (x - 1 = 0) :=
sorry

-- Part 3
theorem associated_equation_part3 (m : ℝ) : 
  (∀ x : ℝ, (x < 2 * x - m ∧ x - 2 ≤ m) → 
  (2 * x - 1 = x + 2 ∨ 3 + x = 2 * (x + 1/2))) → 
  (1 ≤ m ∧ m < 2) :=
sorry

end associated_equation_part1_associated_equation_part2_associated_equation_part3_l2472_247246


namespace shaded_area_percentage_l2472_247210

theorem shaded_area_percentage (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 16 →
  shaded_squares = 8 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 50 := by
  sorry

end shaded_area_percentage_l2472_247210


namespace triangle_area_problem_l2472_247275

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * a * b * Real.sin C

theorem triangle_area_problem (a b c : ℝ) (A B C : ℝ) 
  (h1 : c^2 = (a-b)^2 + 6)
  (h2 : C = π/3) :
  triangle_area a b c A B C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_problem_l2472_247275


namespace divisors_sum_8_implies_one_zero_l2472_247299

def has_three_smallest_distinct_divisors_sum_8 (A : ℕ+) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ+, 
    d₁ < d₂ ∧ d₂ < d₃ ∧
    d₁ ∣ A ∧ d₂ ∣ A ∧ d₃ ∣ A ∧
    d₁.val + d₂.val + d₃.val = 8 ∧
    ∀ d : ℕ+, d ∣ A → d ≤ d₃ → d = d₁ ∨ d = d₂ ∨ d = d₃

def ends_with_one_zero (A : ℕ+) : Prop :=
  ∃ k : ℕ, A.val = 10 * k ∧ k % 10 ≠ 0

theorem divisors_sum_8_implies_one_zero (A : ℕ+) :
  has_three_smallest_distinct_divisors_sum_8 A → ends_with_one_zero A :=
sorry

end divisors_sum_8_implies_one_zero_l2472_247299


namespace parabola_kite_sum_l2472_247256

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

theorem parabola_kite_sum (k : Kite) : 
  k.p1.a > 0 ∧ k.p2.a < 0 ∧  -- Ensure parabolas open in opposite directions
  k.p1.b = -4 ∧ k.p2.b = 6 ∧  -- Specific y-intercepts
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    k.p1.a * x^2 + k.p1.b = 0 ∧   -- x-intercepts of first parabola
    k.p2.a * x^2 + k.p2.b = 0 ∧   -- x-intercepts of second parabola
    k.p1.a * y^2 + k.p1.b = k.p2.a * y^2 + k.p2.b) ∧  -- Intersection point
  (1/2 * (2 * (k.p2.b - k.p1.b)) * (2 * Real.sqrt (k.p2.b / (-k.p2.a))) = 24) →  -- Area of kite
  k.p1.a + (-k.p2.a) = 125/72 := by
sorry

end parabola_kite_sum_l2472_247256


namespace lines_intersect_at_point_l2472_247233

/-- Represents a 2D point --/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line parameterization --/
structure LineParam where
  base : Point2D
  direction : Point2D

/-- First line parameterization --/
def line1 : LineParam := {
  base := { x := 2, y := 3 },
  direction := { x := 1, y := -4 }
}

/-- Second line parameterization --/
def line2 : LineParam := {
  base := { x := 4, y := -6 },
  direction := { x := 5, y := 3 }
}

/-- The intersection point of the two lines --/
def intersection_point : Point2D := {
  x := 185 / 23,
  y := 21 / 23
}

/-- Theorem stating that the given point is the intersection of the two lines --/
theorem lines_intersect_at_point :
  ∃ (t u : ℚ),
    (line1.base.x + t * line1.direction.x = intersection_point.x) ∧
    (line1.base.y + t * line1.direction.y = intersection_point.y) ∧
    (line2.base.x + u * line2.direction.x = intersection_point.x) ∧
    (line2.base.y + u * line2.direction.y = intersection_point.y) := by
  sorry

end lines_intersect_at_point_l2472_247233


namespace driptown_rainfall_2011_l2472_247242

/-- The total rainfall in Driptown in 2011 -/
def total_rainfall_2011 (avg_2010 avg_increase : ℝ) : ℝ :=
  12 * (avg_2010 + avg_increase)

/-- Theorem: The total rainfall in Driptown in 2011 was 468 mm -/
theorem driptown_rainfall_2011 :
  total_rainfall_2011 37.2 1.8 = 468 := by
  sorry

end driptown_rainfall_2011_l2472_247242


namespace iodine_mixture_theorem_l2472_247279

-- Define the given constants
def solution1_percentage : ℝ := 40
def solution2_volume : ℝ := 4.5
def final_mixture_volume : ℝ := 6
def final_mixture_percentage : ℝ := 50

-- Define the unknown percentage of the second solution
def solution2_percentage : ℝ := 26.67

-- Theorem statement
theorem iodine_mixture_theorem :
  solution1_percentage / 100 * solution2_volume + 
  solution2_percentage / 100 * solution2_volume = 
  final_mixture_percentage / 100 * final_mixture_volume := by
  sorry

end iodine_mixture_theorem_l2472_247279


namespace new_drive_free_space_l2472_247257

/-- Calculates the free space on a new external drive after file operations and transfer. -/
theorem new_drive_free_space 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (deleted_size : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_drive_size = 20) :
  new_drive_size - (initial_used - deleted_size + new_files_size) = 10 := by
  sorry

#check new_drive_free_space

end new_drive_free_space_l2472_247257


namespace price_reduction_equation_correct_option_is_c_l2472_247285

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- The equation correctly represents the price reduction scenario -/
def correct_equation (pr : PriceReduction) : Prop :=
  pr.initial_price * (1 - pr.reduction_percentage)^2 = pr.final_price

/-- Theorem stating that the equation correctly represents the given scenario -/
theorem price_reduction_equation :
  ∀ (pr : PriceReduction),
    pr.initial_price = 150 →
    pr.final_price = 96 →
    correct_equation pr :=
by
  sorry

/-- The correct option is C -/
theorem correct_option_is_c : 
  ∃ (pr : PriceReduction),
    pr.initial_price = 150 ∧
    pr.final_price = 96 ∧
    correct_equation pr :=
by
  sorry

end price_reduction_equation_correct_option_is_c_l2472_247285


namespace sum_positive_implies_at_least_one_positive_l2472_247231

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → (a > 0 ∨ b > 0) := by
  sorry

end sum_positive_implies_at_least_one_positive_l2472_247231


namespace parallel_vectors_k_value_l2472_247201

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (3, 4)
  ∀ k : ℝ, are_parallel (a.1 - b.1, a.2 - b.2) (2 * a.1 + k * b.1, 2 * a.2 + k * b.2) →
    k = -2 := by
  sorry

end parallel_vectors_k_value_l2472_247201


namespace repair_shop_earnings_l2472_247295

/-- Calculates the total earnings for a repair shop after applying discounts and taxes. -/
def totalEarnings (
  phoneRepairCost : ℚ)
  (laptopRepairCost : ℚ)
  (computerRepairCost : ℚ)
  (tabletRepairCost : ℚ)
  (smartwatchRepairCost : ℚ)
  (phoneRepairs : ℕ)
  (laptopRepairs : ℕ)
  (computerRepairs : ℕ)
  (tabletRepairs : ℕ)
  (smartwatchRepairs : ℕ)
  (computerRepairDiscount : ℚ)
  (salesTaxRate : ℚ) : ℚ :=
  sorry

theorem repair_shop_earnings :
  totalEarnings 11 15 18 12 8 9 5 4 6 8 (1/10) (1/20) = 393.54 := by
  sorry

end repair_shop_earnings_l2472_247295


namespace ceiling_floor_difference_l2472_247219

theorem ceiling_floor_difference : ⌈(15 / 8) * (-34 / 4)⌉ - ⌊(15 / 8) * ⌊-34 / 4⌋⌋ = 2 := by sorry

end ceiling_floor_difference_l2472_247219


namespace complement_of_union_l2472_247252

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end complement_of_union_l2472_247252


namespace total_jewelry_is_83_l2472_247280

/-- Represents the initial jewelry counts and purchase rules --/
structure JewelryInventory where
  initial_necklaces : ℕ
  initial_earrings : ℕ
  initial_bracelets : ℕ
  initial_rings : ℕ
  store_a_necklaces : ℕ
  store_a_bracelets : ℕ
  store_b_necklaces : ℕ

/-- Calculates the total number of jewelry pieces after all additions --/
def totalJewelryPieces (inventory : JewelryInventory) : ℕ :=
  let store_a_earrings := (2 * inventory.initial_earrings) / 3
  let store_b_rings := 2 * inventory.initial_rings
  let mother_gift_earrings := store_a_earrings / 5
  
  inventory.initial_necklaces + inventory.initial_earrings + inventory.initial_bracelets + inventory.initial_rings +
  inventory.store_a_necklaces + store_a_earrings + inventory.store_a_bracelets +
  inventory.store_b_necklaces + store_b_rings +
  mother_gift_earrings

/-- Theorem stating that the total jewelry pieces is 83 given the specific inventory --/
theorem total_jewelry_is_83 :
  totalJewelryPieces ⟨10, 15, 5, 8, 10, 3, 4⟩ = 83 := by
  sorry

end total_jewelry_is_83_l2472_247280


namespace james_club_night_cost_l2472_247207

def club_night_cost (entry_fee : ℚ) (friend_rounds : ℕ) (friends : ℕ) 
  (self_drinks : ℕ) (cocktail_price : ℚ) (non_alcoholic_price : ℚ) 
  (cocktail_discount : ℚ) (cocktails_bought : ℕ) (burger_price : ℚ) 
  (fries_price : ℚ) (food_tip_rate : ℚ) (drink_tip_rate : ℚ) : ℚ :=
  let total_drinks := friend_rounds * friends + self_drinks
  let non_alcoholic_drinks := total_drinks - cocktails_bought
  let cocktail_cost := cocktails_bought * cocktail_price
  let discounted_cocktail_cost := 
    if cocktails_bought ≥ 3 then cocktail_cost * (1 - cocktail_discount) else cocktail_cost
  let non_alcoholic_cost := non_alcoholic_drinks * non_alcoholic_price
  let food_cost := burger_price + fries_price
  let food_tip := food_cost * food_tip_rate
  let drink_tip := (cocktail_cost + non_alcoholic_cost) * drink_tip_rate
  entry_fee + discounted_cocktail_cost + non_alcoholic_cost + food_cost + food_tip + drink_tip

theorem james_club_night_cost :
  club_night_cost 30 3 10 8 10 5 0.2 7 20 8 0.2 0.15 = 308.35 := by
  sorry

end james_club_night_cost_l2472_247207


namespace second_bush_pink_roses_l2472_247277

def rose_problem (red : ℕ) (yellow : ℕ) (orange : ℕ) (total_picked : ℕ) : ℕ :=
  let red_picked := red / 2
  let yellow_picked := yellow / 4
  let orange_picked := orange / 4
  let pink_picked := total_picked - red_picked - yellow_picked - orange_picked
  2 * pink_picked

theorem second_bush_pink_roses :
  rose_problem 12 20 8 22 = 18 := by
  sorry

end second_bush_pink_roses_l2472_247277


namespace brookes_cows_solution_l2472_247272

/-- Represents the problem of determining the number of cows Brooke has --/
def brookes_cows (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) (total_earnings : ℚ) : Prop :=
  milk_price = 3 ∧
  butter_conversion = 2 ∧
  butter_price = 3/2 ∧
  milk_per_cow = 4 ∧
  num_customers = 6 ∧
  milk_per_customer = 6 ∧
  total_earnings = 144 ∧
  ∃ (num_cows : ℕ), 
    (↑num_cows : ℚ) * milk_per_cow = 
      (↑num_customers * milk_per_customer) + 
      ((total_earnings - (↑num_customers * milk_per_customer * milk_price)) / butter_price * (1 / butter_conversion))

theorem brookes_cows_solution :
  ∀ (milk_price butter_conversion butter_price milk_per_cow : ℚ)
    (num_customers : ℕ) (milk_per_customer total_earnings : ℚ),
  brookes_cows milk_price butter_conversion butter_price milk_per_cow num_customers milk_per_customer total_earnings →
  ∃ (num_cows : ℕ), num_cows = 12 :=
by sorry

end brookes_cows_solution_l2472_247272


namespace min_sum_squares_l2472_247206

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
sorry

end min_sum_squares_l2472_247206


namespace right_triangle_sets_l2472_247273

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_sets :
  ¬(is_right_triangle 6 7 8) ∧
  ¬(is_right_triangle 1 (Real.sqrt 2) 5) ∧
  is_right_triangle 6 8 10 ∧
  ¬(is_right_triangle (Real.sqrt 5) (2 * Real.sqrt 3) (Real.sqrt 15)) :=
by sorry

end right_triangle_sets_l2472_247273


namespace largest_prime_divisor_of_2100210021_base_3_l2472_247241

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_2100210021_base_3 :
  let n := base_3_to_decimal [1, 2, 0, 0, 1, 2, 0, 0, 1, 2]
  ∃ (p : Nat), is_prime p ∧ p ∣ n ∧ ∀ (q : Nat), is_prime q → q ∣ n → q ≤ p ∧ p = 46501 :=
sorry

end largest_prime_divisor_of_2100210021_base_3_l2472_247241


namespace expression_evaluation_l2472_247220

theorem expression_evaluation :
  let x : ℝ := 0
  let expr := (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2))
  expr = -1 := by sorry

end expression_evaluation_l2472_247220


namespace chessboard_game_stone_range_min_le_max_stones_l2472_247237

/-- A game on an n × n chessboard with k stones -/
def ChessboardGame (n : ℕ) (k : ℕ) : Prop :=
  n > 0 ∧ k ≥ 2 * n^2 - 2 * n ∧ k ≤ 3 * n^2 - 4 * n

/-- The theorem stating the range of stones for the game -/
theorem chessboard_game_stone_range (n : ℕ) :
  n > 0 → ∀ k, ChessboardGame n k ↔ 2 * n^2 - 2 * n ≤ k ∧ k ≤ 3 * n^2 - 4 * n :=
by sorry

/-- The minimum number of stones for the game -/
def min_stones (n : ℕ) : ℕ := 2 * n^2 - 2 * n

/-- The maximum number of stones for the game -/
def max_stones (n : ℕ) : ℕ := 3 * n^2 - 4 * n

/-- Theorem stating that the minimum number of stones is always less than or equal to the maximum -/
theorem min_le_max_stones (n : ℕ) : n > 0 → min_stones n ≤ max_stones n :=
by sorry

end chessboard_game_stone_range_min_le_max_stones_l2472_247237


namespace problem_statement_l2472_247240

theorem problem_statement :
  (∀ a b c d : ℝ, a^6 + b^6 + c^6 + d^6 - 6*a*b*c*d ≥ -2) ∧
  (∀ k : ℕ, k % 2 = 1 → k ≥ 5 →
    ∃ M_k : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k*a*b*c*d ≥ M_k) :=
by sorry

end problem_statement_l2472_247240


namespace reaction_weight_equality_l2472_247212

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Bromine in g/mol -/
def Br_weight : ℝ := 79.904

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Molecular weight of CaBr2 in g/mol -/
def CaBr2_weight : ℝ := Ca_weight + 2 * Br_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of Ca(OH)2 in g/mol -/
def CaOH2_weight : ℝ := Ca_weight + 2 * (O_weight + H_weight)

/-- Molecular weight of HBr in g/mol -/
def HBr_weight : ℝ := H_weight + Br_weight

/-- Theorem stating that the molecular weight of reactants equals the molecular weight of products
    and is equal to 235.918 g/mol -/
theorem reaction_weight_equality :
  CaBr2_weight + 2 * H2O_weight = CaOH2_weight + 2 * HBr_weight ∧
  CaBr2_weight + 2 * H2O_weight = 235.918 :=
by sorry

end reaction_weight_equality_l2472_247212


namespace complex_fraction_calculation_l2472_247224

theorem complex_fraction_calculation : 
  (1 / (7 / 9) - (3 / 5) / 7) * (11 / (6 + 3 / 5)) / (4 / 13) - 2.4 = 4.1 := by
  sorry

end complex_fraction_calculation_l2472_247224


namespace complex_magnitude_l2472_247263

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l2472_247263


namespace find_alpha_l2472_247278

theorem find_alpha (α β : ℝ) (h1 : α + β = 11) (h2 : α * β = 24) (h3 : α > β) : α = 8 := by
  sorry

end find_alpha_l2472_247278


namespace pizza_slices_per_person_l2472_247248

theorem pizza_slices_per_person 
  (coworkers : ℕ) 
  (pizzas : ℕ) 
  (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8)
  : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end pizza_slices_per_person_l2472_247248


namespace floor_equation_solution_l2472_247205

def solution_set : Set ℚ := {16/23, 17/23, 18/23, 19/23, 20/23, 21/23, 22/23, 1}

theorem floor_equation_solution (x : ℚ) :
  (⌊(20 : ℚ) * x + 23⌋ = 20 + 23 * x) ↔ x ∈ solution_set := by
  sorry

end floor_equation_solution_l2472_247205


namespace car_cost_calculation_l2472_247262

/-- The total cost of a car given an initial payment, monthly payment, and number of months -/
theorem car_cost_calculation (initial_payment monthly_payment num_months : ℕ) :
  initial_payment = 5400 →
  monthly_payment = 420 →
  num_months = 19 →
  initial_payment + monthly_payment * num_months = 13380 := by
  sorry

end car_cost_calculation_l2472_247262


namespace broken_marbles_count_l2472_247244

/-- The number of marbles in the first set -/
def set1_total : ℕ := 50

/-- The percentage of broken marbles in the first set -/
def set1_broken_percent : ℚ := 1/10

/-- The number of marbles in the second set -/
def set2_total : ℕ := 60

/-- The percentage of broken marbles in the second set -/
def set2_broken_percent : ℚ := 1/5

/-- The total number of broken marbles in both sets -/
def total_broken_marbles : ℕ := 17

theorem broken_marbles_count : 
  ⌊(set1_total : ℚ) * set1_broken_percent⌋ + ⌊(set2_total : ℚ) * set2_broken_percent⌋ = total_broken_marbles :=
by sorry

end broken_marbles_count_l2472_247244


namespace equality_of_cyclic_powers_l2472_247236

theorem equality_of_cyclic_powers (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq1 : x^y = y^z) (h_eq2 : y^z = z^x) : x = y ∧ y = z :=
sorry

end equality_of_cyclic_powers_l2472_247236


namespace sum_of_digits_of_B_l2472_247298

def digit_sum (n : ℕ) : ℕ := sorry

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_of_B : digit_sum B = 7 := by sorry

end sum_of_digits_of_B_l2472_247298


namespace split_investment_average_rate_l2472_247286

/-- The average interest rate for a split investment --/
theorem split_investment_average_rate (total_investment : ℝ) 
  (rate1 rate2 : ℝ) (fee : ℝ) : 
  total_investment > 0 →
  rate1 > 0 →
  rate2 > 0 →
  rate1 < rate2 →
  fee > 0 →
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_investment ∧
    rate1 * (total_investment - x) - fee = rate2 * x →
  (rate2 * x + (rate1 * (total_investment - x) - fee)) / total_investment = 0.05133 :=
by sorry

end split_investment_average_rate_l2472_247286


namespace school_cinema_visit_payment_l2472_247270

/-- Represents the ticket pricing structure and student count for a cinema visit -/
structure CinemaVisit where
  individual_price : ℝ  -- Price of an individual ticket
  group_price : ℝ       -- Price of a group ticket for 10 people
  student_discount : ℝ  -- Discount rate for students (as a decimal)
  student_count : ℕ     -- Number of students

/-- Calculates the minimum amount to be paid for a school cinema visit -/
def minimum_payment (cv : CinemaVisit) : ℝ :=
  let group_size := 10
  let full_groups := cv.student_count / group_size
  let total_group_price := (full_groups * cv.group_price) * (1 - cv.student_discount)
  total_group_price

/-- The theorem stating the minimum payment for the given scenario -/
theorem school_cinema_visit_payment :
  let cv : CinemaVisit := {
    individual_price := 6,
    group_price := 40,
    student_discount := 0.1,
    student_count := 1258
  }
  minimum_payment cv = 4536 := by sorry

end school_cinema_visit_payment_l2472_247270


namespace min_races_correct_l2472_247203

/-- Represents a race strategy to find the top 3 fastest horses -/
structure RaceStrategy where
  numRaces : ℕ
  ensuresTop3 : Bool

/-- The minimum number of races needed to find the top 3 fastest horses -/
def minRaces : ℕ := 7

/-- The total number of horses -/
def totalHorses : ℕ := 25

/-- The maximum number of horses that can race together -/
def maxHorsesPerRace : ℕ := 5

/-- Predicate to check if a race strategy is valid -/
def isValidStrategy (s : RaceStrategy) : Prop :=
  s.numRaces ≥ minRaces ∧ s.ensuresTop3

/-- Theorem stating that the minimum number of races is correct -/
theorem min_races_correct :
  ∀ s : RaceStrategy,
    isValidStrategy s →
    s.numRaces ≥ minRaces :=
sorry

end min_races_correct_l2472_247203


namespace multiplication_puzzle_l2472_247282

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem multiplication_puzzle (a b : ℕ) (ha : is_digit a) (hb : is_digit b) 
  (h_mult : (30 + a) * (10 * b + 4) = 156) : a + b = 9 := by
  sorry

end multiplication_puzzle_l2472_247282


namespace f_inequality_solution_l2472_247213

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 < x ∧ x < 2

-- Define the solution set
def solution_set (a : ℝ) : Prop := (-2 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)

-- State the theorem
theorem f_inequality_solution :
  ∀ a : ℝ, domain a → domain (a^2 - 2) →
  (f a + f (a^2 - 2) < 0 ↔ solution_set a) :=
sorry

end f_inequality_solution_l2472_247213


namespace intersection_of_A_and_B_l2472_247271

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2472_247271


namespace henry_kombucha_consumption_l2472_247247

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle in dollars -/
def bottle_cost : ℚ := 3

/-- The cash refund for each bottle in dollars -/
def bottle_refund : ℚ := 1/10

/-- The number of bottles Henry can buy with his yearly refund -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem henry_kombucha_consumption :
  bottles_per_month * bottle_refund * months_in_year = bottles_from_refund * bottle_cost :=
sorry

end henry_kombucha_consumption_l2472_247247


namespace x_value_implies_y_value_l2472_247243

theorem x_value_implies_y_value :
  let x := (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48
  x^3 - 2*x^2 + Real.sin (2*Real.pi*x) - Real.cos (Real.pi*x) = 10 := by
  sorry

end x_value_implies_y_value_l2472_247243


namespace squirrel_travel_time_l2472_247254

-- Define the speed of the squirrel in miles per hour
def speed : ℝ := 4

-- Define the distance to be traveled in miles
def distance : ℝ := 1

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem squirrel_travel_time :
  (distance / speed) * minutes_per_hour = 15 := by
  sorry

end squirrel_travel_time_l2472_247254


namespace intersection_slopes_sum_l2472_247297

/-- Given a line y = 2x - 3 and a parabola y² = 4x intersecting at points A and B,
    with O as the origin and k₁, k₂ as the slopes of OA and OB respectively,
    prove that the sum of the reciprocals of the slopes 1/k₁ + 1/k₂ = 1/2 -/
theorem intersection_slopes_sum (A B : ℝ × ℝ) (k₁ k₂ : ℝ) : 
  (A.2 = 2 * A.1 - 3) →
  (B.2 = 2 * B.1 - 3) →
  (A.2^2 = 4 * A.1) →
  (B.2^2 = 4 * B.1) →
  (k₁ = A.2 / A.1) →
  (k₂ = B.2 / B.1) →
  (1 / k₁ + 1 / k₂ = 1 / 2) := by
  sorry

end intersection_slopes_sum_l2472_247297


namespace benny_seashells_l2472_247259

theorem benny_seashells (initial_seashells given_away remaining : ℕ) : 
  initial_seashells = 66 → given_away = 52 → remaining = 14 →
  initial_seashells - given_away = remaining := by sorry

end benny_seashells_l2472_247259


namespace distinct_sums_theorem_l2472_247287

theorem distinct_sums_theorem (k n : ℕ) (a b c : Fin n → ℝ) :
  k ≥ 3 →
  n > Nat.choose k 3 →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j) →
  ∃ S : Finset ℝ,
    S.card ≥ k + 1 ∧
    (∀ i : Fin n, (a i + b i) ∈ S ∧ (a i + c i) ∈ S ∧ (b i + c i) ∈ S) :=
by sorry

end distinct_sums_theorem_l2472_247287


namespace simplify_and_evaluate_l2472_247202

theorem simplify_and_evaluate :
  ∀ x : ℤ, -2 < x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 →
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) ∧
  (x = 2 → x^2 / (x - 1) = 4) :=
by sorry

end simplify_and_evaluate_l2472_247202


namespace banana_jar_candy_count_l2472_247293

/-- Given three jars of candy with specific relationships, prove the number of candy pieces in the banana jar. -/
theorem banana_jar_candy_count (peanut_butter grape banana : ℕ) 
  (h1 : peanut_butter = 4 * grape)
  (h2 : grape = banana + 5)
  (h3 : peanut_butter = 192) : 
  banana = 43 := by
  sorry

end banana_jar_candy_count_l2472_247293


namespace min_value_sin_2x_minus_pi_4_l2472_247238

theorem min_value_sin_2x_minus_pi_4 :
  ∃ (min : ℝ), min = -Real.sqrt 2 / 2 ∧
  ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (2 * x - Real.pi / 4) ≥ min :=
sorry

end min_value_sin_2x_minus_pi_4_l2472_247238


namespace zorg_vamp_and_not_wook_l2472_247234

-- Define the types
variable (U : Type) -- Universe set
variable (Zorg Xyon Wook Vamp : Set U)

-- Define the conditions
variable (h1 : Zorg ⊆ Xyon)
variable (h2 : Wook ⊆ Xyon)
variable (h3 : Vamp ⊆ Zorg)
variable (h4 : Wook ⊆ Vamp)
variable (h5 : Zorg ∩ Wook = ∅)

-- Theorem to prove
theorem zorg_vamp_and_not_wook : 
  Zorg ⊆ Vamp ∧ Zorg ∩ Wook = ∅ :=
by sorry

end zorg_vamp_and_not_wook_l2472_247234


namespace notebook_cost_l2472_247239

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : notebooks_per_student > 2)
  (h4 : cost_per_notebook > notebooks_per_student)
  (h5 : buyers * notebooks_per_student * cost_per_notebook = 2601) :
  cost_per_notebook = 289 := by
  sorry

end notebook_cost_l2472_247239


namespace max_cos_a_l2472_247221

theorem max_cos_a (a b : Real) (h : Real.cos (a + b) = Real.cos a - Real.cos b) :
  ∃ (max_cos_a : Real), max_cos_a = 1 ∧ ∀ x, Real.cos x ≤ max_cos_a :=
by sorry

end max_cos_a_l2472_247221


namespace walking_distance_l2472_247261

/-- Given a person who walks at two speeds, prove that the actual distance traveled at the slower speed is 50 km. -/
theorem walking_distance (slow_speed fast_speed extra_distance : ℝ) 
  (h1 : slow_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : extra_distance = 20)
  (h4 : slow_speed > 0)
  (h5 : fast_speed > slow_speed) :
  ∃ (actual_distance : ℝ),
    actual_distance / slow_speed = (actual_distance + extra_distance) / fast_speed ∧
    actual_distance = 50 := by
  sorry


end walking_distance_l2472_247261


namespace problem_solution_l2472_247208

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A(a)
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

theorem problem_solution :
  (B = Set.Ici 3) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici 2) := by sorry

end problem_solution_l2472_247208


namespace condition_relationship_l2472_247269

theorem condition_relationship (x : ℝ) :
  (∀ x, 2 * Real.sqrt 2 ≤ (x^2 + 2) / x ∧ (x^2 + 2) / x ≤ 3 → Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2) ∧
  (∃ x, Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2 ∧ (2 * Real.sqrt 2 > (x^2 + 2) / x ∨ (x^2 + 2) / x > 3)) :=
by sorry

end condition_relationship_l2472_247269


namespace ball_draw_probability_l2472_247217

theorem ball_draw_probability (n : ℕ) : 
  (200 ≤ n) ∧ (n ≤ 1000) ∧ 
  (∃ k : ℕ, n = k^2) ∧
  (∃ x y : ℕ, x + y = n ∧ (x - y)^2 = n) →
  (∃ l : List ℕ, l.length = 17 ∧ n ∈ l) :=
sorry

end ball_draw_probability_l2472_247217


namespace dining_bill_share_l2472_247230

theorem dining_bill_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) :
  total_bill = 139 ∧ num_people = 7 ∧ tip_percentage = 1/10 →
  (total_bill + total_bill * tip_percentage) / num_people = 2184/100 := by
  sorry

end dining_bill_share_l2472_247230


namespace smallest_two_digit_prime_with_composite_reverse_l2472_247267

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- The main theorem -/
theorem smallest_two_digit_prime_with_composite_reverse :
  ∀ p : ℕ, is_two_digit p → Nat.Prime p →
    (∀ q : ℕ, is_two_digit q → Nat.Prime q → q < p →
      Nat.Prime (reverse_digits q)) →
    ¬Nat.Prime (reverse_digits p) →
    p = 19 := by
  sorry

end smallest_two_digit_prime_with_composite_reverse_l2472_247267


namespace parallelepiped_diagonals_edges_squares_sum_equal_l2472_247200

/-- A parallelepiped with side lengths a, b, and c. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The sum of squares of the lengths of the four diagonals of a parallelepiped. -/
def sum_squares_diagonals (p : Parallelepiped) : ℝ :=
  4 * (p.a^2 + p.b^2 + p.c^2)

/-- The sum of squares of the lengths of the twelve edges of a parallelepiped. -/
def sum_squares_edges (p : Parallelepiped) : ℝ :=
  4 * p.a^2 + 4 * p.b^2 + 4 * p.c^2

/-- 
Theorem: The sum of the squares of the lengths of the four diagonals 
of a parallelepiped is equal to the sum of the squares of the lengths of its twelve edges.
-/
theorem parallelepiped_diagonals_edges_squares_sum_equal (p : Parallelepiped) :
  sum_squares_diagonals p = sum_squares_edges p := by
  sorry

end parallelepiped_diagonals_edges_squares_sum_equal_l2472_247200


namespace ice_cream_box_problem_l2472_247283

/-- The number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- The cost of a box of ice cream bars in dollars -/
def box_cost : ℚ := 15/2

/-- The number of friends -/
def num_friends : ℕ := 6

/-- The number of bars each friend wants -/
def bars_per_friend : ℕ := 2

/-- The cost per person in dollars -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_problem :
  bars_per_box = (num_friends * bars_per_friend) / ((num_friends * cost_per_person) / box_cost) :=
by sorry

end ice_cream_box_problem_l2472_247283


namespace lesser_number_l2472_247253

theorem lesser_number (a b : ℝ) (h1 : a + b = 55) (h2 : a - b = 7) : min a b = 24 := by
  sorry

end lesser_number_l2472_247253


namespace nina_reading_homework_multiplier_l2472_247266

-- Define the given conditions
def ruby_math_homework : ℕ := 6
def ruby_reading_homework : ℕ := 2
def nina_total_homework : ℕ := 48
def nina_math_homework_multiplier : ℕ := 4

-- Define Nina's math homework
def nina_math_homework : ℕ := ruby_math_homework * (nina_math_homework_multiplier + 1)

-- Define Nina's reading homework
def nina_reading_homework : ℕ := nina_total_homework - nina_math_homework

-- Theorem to prove
theorem nina_reading_homework_multiplier :
  nina_reading_homework / ruby_reading_homework = 9 := by
  sorry


end nina_reading_homework_multiplier_l2472_247266


namespace least_divisible_by_1_to_9_halved_l2472_247258

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ k : ℕ, a ≤ k → k ≤ b → k ∣ n

theorem least_divisible_by_1_to_9_halved :
  ∃ l : ℕ, (∀ m : ℕ, is_divisible_by_range m 1 9 → l ≤ m) ∧
           is_divisible_by_range l 1 9 ∧
           l / 2 = 1260 := by
  sorry

end least_divisible_by_1_to_9_halved_l2472_247258


namespace mango_rate_problem_l2472_247291

/-- Calculates the rate per kg of mangoes given the total amount paid, grape weight, grape rate, and mango weight -/
def mango_rate (total_paid : ℕ) (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_weight * grape_rate) / mango_weight

theorem mango_rate_problem :
  mango_rate 1376 14 54 10 = 62 := by
  sorry

end mango_rate_problem_l2472_247291


namespace intersection_x_coordinate_l2472_247276

/-- Given two distinct real numbers k and b, prove that the x-coordinate of the 
    intersection point of the lines y = kx + b and y = bx + k is 1. -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) : 
  ∃ x : ℝ, x = 1 ∧ kx + b = bx + k := by
  sorry

end intersection_x_coordinate_l2472_247276


namespace auditorium_seats_l2472_247265

theorem auditorium_seats (initial_seats : ℕ) (final_seats : ℕ) (seat_increase : ℕ) : 
  initial_seats = 320 →
  final_seats = 420 →
  seat_increase = 4 →
  ∃ (initial_rows : ℕ),
    initial_rows > 0 ∧
    initial_seats % initial_rows = 0 ∧
    (initial_seats / initial_rows + seat_increase) * (initial_rows + 1) = final_seats ∧
    initial_rows + 1 = 21 := by
  sorry

end auditorium_seats_l2472_247265


namespace added_number_proof_l2472_247250

theorem added_number_proof (x : ℝ) : 
  (((2 * (62.5 + x)) / 5) - 5 = 22) → x = 5 := by
  sorry

end added_number_proof_l2472_247250


namespace expression_evaluation_l2472_247215

theorem expression_evaluation : 
  1 / (2 - Real.sqrt 3) - Real.pi ^ 0 - 2 * Real.cos (30 * π / 180) = 1 := by
  sorry

end expression_evaluation_l2472_247215


namespace sugar_profit_problem_l2472_247222

/-- A merchant sells sugar with two different profit percentages --/
theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_known_profit : ℝ) (sugar_at_unknown_profit : ℝ)
  (known_profit_percentage : ℝ) (overall_profit_percentage : ℝ) (unknown_profit_percentage : ℝ)
  (h1 : total_sugar = 1000)
  (h2 : sugar_at_known_profit = 400)
  (h3 : sugar_at_unknown_profit = 600)
  (h4 : known_profit_percentage = 8)
  (h5 : overall_profit_percentage = 14)
  (h6 : total_sugar = sugar_at_known_profit + sugar_at_unknown_profit)
  (h7 : sugar_at_known_profit * (known_profit_percentage / 100) +
        sugar_at_unknown_profit * (unknown_profit_percentage / 100) =
        total_sugar * (overall_profit_percentage / 100)) :
  unknown_profit_percentage = 18 := by
sorry

end sugar_profit_problem_l2472_247222


namespace inequality_always_positive_l2472_247216

theorem inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end inequality_always_positive_l2472_247216


namespace square_area_ratio_l2472_247284

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l2472_247284


namespace geometric_sequence_problem_l2472_247264

/-- 
A geometric sequence is defined by its first term and common ratio.
This theorem proves that for a geometric sequence where the third term is 18
and the sixth term is 162, the first term is 2 and the common ratio is 3.
-/
theorem geometric_sequence_problem (a r : ℝ) : 
  a * r^2 = 18 → a * r^5 = 162 → a = 2 ∧ r = 3 := by
sorry

end geometric_sequence_problem_l2472_247264


namespace rectangular_solid_volume_l2472_247268

-- Define the rectangular solid
structure RectangularSolid where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the theorem
theorem rectangular_solid_volume 
  (r : RectangularSolid) 
  (h1 : r.x * r.y = 15) 
  (h2 : r.y * r.z = 10) 
  (h3 : r.x * r.z = 6) 
  (h4 : r.x = 3 * r.y) : 
  r.x * r.y * r.z = 6 * Real.sqrt 5 := by
  sorry


end rectangular_solid_volume_l2472_247268


namespace box_capacity_l2472_247274

-- Define the volumes and capacities
def small_box_volume : ℝ := 24
def small_box_paperclips : ℕ := 60
def large_box_volume : ℝ := 72
def large_box_staples : ℕ := 90

-- Define the theorem
theorem box_capacity :
  ∃ (large_box_paperclips large_box_mixed_staples : ℕ),
    large_box_paperclips = 90 ∧ 
    large_box_mixed_staples = 45 ∧
    (large_box_paperclips : ℝ) / (large_box_volume / 2) = (small_box_paperclips : ℝ) / small_box_volume ∧
    (large_box_mixed_staples : ℝ) / (large_box_volume / 2) = (large_box_staples : ℝ) / large_box_volume :=
by
  sorry

end box_capacity_l2472_247274


namespace lemming_average_distance_l2472_247228

/-- Given a square with side length 12, prove that a point moving 7 units along
    the diagonal from a corner and then 4 units perpendicular to the diagonal
    results in an average distance of 6 units to each side of the square. -/
theorem lemming_average_distance (square_side : ℝ) (diag_move : ℝ) (perp_move : ℝ) :
  square_side = 12 →
  diag_move = 7 →
  perp_move = 4 →
  let diag := square_side * Real.sqrt 2
  let diag_ratio := diag_move / diag
  let x := diag_ratio * square_side
  let y := diag_ratio * square_side
  let final_x := x + perp_move
  let final_y := y
  let dist_left := final_x
  let dist_bottom := final_y
  let dist_right := square_side - final_x
  let dist_top := square_side - final_y
  let avg_dist := (dist_left + dist_bottom + dist_right + dist_top) / 4
  avg_dist = 6 := by
sorry

end lemming_average_distance_l2472_247228


namespace max_tiles_theorem_l2472_247211

/-- Represents a rhombic tile with side length 1 and angles 60° and 120° -/
structure RhombicTile :=
  (side_length : ℝ := 1)
  (angle1 : ℝ := 60)
  (angle2 : ℝ := 120)

/-- Represents an equilateral triangle with side length n -/
structure EquilateralTriangle :=
  (side_length : ℕ)

/-- Calculates the maximum number of rhombic tiles that can fit in an equilateral triangle -/
def max_tiles_in_triangle (triangle : EquilateralTriangle) : ℕ :=
  (triangle.side_length^2 - triangle.side_length) / 2

/-- Theorem: The maximum number of rhombic tiles in an equilateral triangle is (n^2 - n) / 2 -/
theorem max_tiles_theorem (n : ℕ) (triangle : EquilateralTriangle) (tile : RhombicTile) :
  triangle.side_length = n →
  tile.side_length = 1 →
  tile.angle1 = 60 →
  tile.angle2 = 120 →
  max_tiles_in_triangle triangle = (n^2 - n) / 2 := by
  sorry

end max_tiles_theorem_l2472_247211


namespace inequality_solution_set_l2472_247223

theorem inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - (a + 1/a + 1)*x + a + 1/a < 0} = {x : ℝ | 1 < x ∧ x < a + 1/a} := by
  sorry

end inequality_solution_set_l2472_247223


namespace unique_divisiblity_solution_l2472_247289

theorem unique_divisiblity_solution : 
  ∀ a m n : ℕ+, 
    (a^n.val + 1 ∣ (a+1)^m.val) → 
    (a = 2 ∧ m = 2 ∧ n = 3) :=
by sorry

end unique_divisiblity_solution_l2472_247289


namespace win_sector_area_l2472_247209

/-- Given a circular spinner with radius 8 cm and a probability of winning of 1/4,
    the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_sector_area : ℝ) : 
  radius = 8 →
  win_probability = 1 / 4 →
  win_sector_area = win_probability * π * radius^2 →
  win_sector_area = 16 * π := by
  sorry

end win_sector_area_l2472_247209


namespace complex_number_in_fourth_quadrant_l2472_247290

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l2472_247290


namespace ruble_payment_combinations_l2472_247218

theorem ruble_payment_combinations : 
  ∃! n : ℕ, n = (Finset.filter (λ (x : ℕ × ℕ) => 3 * x.1 + 5 * x.2 = 78) (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 6 :=
by sorry

end ruble_payment_combinations_l2472_247218


namespace dales_peppers_theorem_l2472_247294

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem dales_peppers_theorem : total_peppers = 5.666666666666667 := by
  sorry

end dales_peppers_theorem_l2472_247294


namespace arithmetic_calculations_l2472_247251

theorem arithmetic_calculations :
  ((-2 : ℝ) + |3| + (-6) + |7| = 2) ∧
  (3.7 + (-1.3) + (-6.7) + 2.3 = -2) := by
  sorry

end arithmetic_calculations_l2472_247251


namespace alice_bob_meet_l2472_247281

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- Bob's extra skip every second turn -/
def bob_extra : ℕ := 1

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 36

/-- Function to calculate position on the circle after a number of moves -/
def position (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

/-- Alice's position after a given number of turns -/
def alice_position (turns : ℕ) : ℕ :=
  position n (alice_move * turns)

/-- Bob's position after a given number of turns -/
def bob_position (turns : ℕ) : ℕ :=
  position n (n * turns - bob_move * turns - bob_extra * (turns / 2))

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet : alice_position meeting_turns = bob_position meeting_turns := by
  sorry


end alice_bob_meet_l2472_247281


namespace no_natural_square_difference_2018_l2472_247204

theorem no_natural_square_difference_2018 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2018 := by
  sorry

end no_natural_square_difference_2018_l2472_247204


namespace hyperbola_asymptotes_l2472_247255

/-- Given a hyperbola with equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    if the eccentricity is √3, then the equation of its asymptotes is x ± √2y = 0 -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2 - 2 * y^2 = 0) :=
sorry

end hyperbola_asymptotes_l2472_247255


namespace charging_bull_rounds_in_hour_l2472_247229

/-- The time in seconds for the racing magic to complete one round -/
def racing_magic_time : ℕ := 60

/-- The time in minutes it takes for both to meet at the starting point for the second time -/
def meeting_time : ℕ := 6

/-- The number of rounds the racing magic completes in the meeting time -/
def racing_magic_rounds : ℕ := meeting_time

/-- The number of rounds the charging bull completes in the meeting time -/
def charging_bull_rounds : ℕ := racing_magic_rounds + 1

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of rounds the charging bull makes in an hour -/
def charging_bull_hourly_rounds : ℕ := 
  (charging_bull_rounds * minutes_per_hour) / meeting_time

theorem charging_bull_rounds_in_hour : 
  charging_bull_hourly_rounds = 70 := by sorry

end charging_bull_rounds_in_hour_l2472_247229


namespace total_rectangles_is_eighteen_l2472_247296

/-- Represents a rectangle in the figure -/
structure Rectangle where
  size : Nat

/-- Represents the figure composed of rectangles -/
structure Figure where
  big_rectangle : Rectangle
  small_rectangles : Finset Rectangle
  middle_rectangles : Finset Rectangle

/-- Counts the total number of rectangles in the figure -/
def count_rectangles (f : Figure) : Nat :=
  1 + f.small_rectangles.card + f.middle_rectangles.card

/-- The theorem stating that the total number of rectangles is 18 -/
theorem total_rectangles_is_eighteen (f : Figure) 
  (h1 : f.big_rectangle.size = 1)
  (h2 : f.small_rectangles.card = 6)
  (h3 : f.middle_rectangles.card = 11) : 
  count_rectangles f = 18 := by
  sorry

#check total_rectangles_is_eighteen

end total_rectangles_is_eighteen_l2472_247296


namespace fraction_multiplication_l2472_247235

theorem fraction_multiplication : (1 / 2) * (3 / 5) * (7 / 11) * (4 / 13) = 84 / 1430 := by
  sorry

end fraction_multiplication_l2472_247235


namespace product_of_divisors_60_l2472_247292

-- Define the number we're working with
def n : ℕ := 60

-- Define a function to get all divisors of a natural number
def divisors (m : ℕ) : Finset ℕ :=
  sorry

-- Define the product of all divisors
def product_of_divisors (m : ℕ) : ℕ :=
  (divisors m).prod id

-- Theorem statement
theorem product_of_divisors_60 :
  product_of_divisors n = 46656000000000 := by
  sorry

end product_of_divisors_60_l2472_247292


namespace kati_age_l2472_247226

/-- Represents a person's age and birthday information -/
structure Person where
  age : ℕ
  birthdays : ℕ

/-- Represents the family members -/
structure Family where
  kati : Person
  brother : Person
  grandfather : Person

/-- The conditions of the problem -/
def problem_conditions (f : Family) : Prop :=
  f.kati.age = f.grandfather.birthdays ∧
  f.kati.age + f.brother.age + f.grandfather.age = 111 ∧
  f.kati.age > f.brother.age ∧
  f.kati.age - f.brother.age < 4 ∧
  f.grandfather.age = 4 * f.grandfather.birthdays + (f.grandfather.age % 4)

/-- The theorem to prove -/
theorem kati_age (f : Family) : 
  problem_conditions f → f.kati.age = 18 :=
by
  sorry

end kati_age_l2472_247226


namespace binary_addition_multiplication_l2472_247214

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_addition_multiplication : 
  let b1 := [true, true, true, true, true]
  let b2 := [true, true, true, true, true, true, true, true]
  let b3 := [false, true]
  (binary_to_decimal b1 + binary_to_decimal b2) * binary_to_decimal b3 = 572 := by
  sorry

end binary_addition_multiplication_l2472_247214


namespace space_needle_height_is_184_l2472_247249

-- Define the heights of the towers
def cn_tower_height : ℝ := 553
def height_difference : ℝ := 369

-- Define the height of the Space Needle
def space_needle_height : ℝ := cn_tower_height - height_difference

-- Theorem to prove
theorem space_needle_height_is_184 : space_needle_height = 184 := by
  sorry

end space_needle_height_is_184_l2472_247249


namespace polynomial_remainder_theorem_l2472_247232

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^3 - 5*x^2 + 3*x - 7) % (x - 3) = -16 := by
  sorry

end polynomial_remainder_theorem_l2472_247232


namespace jim_future_age_l2472_247288

/-- Tom's current age -/
def tom_current_age : ℕ := 37

/-- Tom's age 7 years ago -/
def tom_age_7_years_ago : ℕ := tom_current_age - 7

/-- Jim's age 7 years ago -/
def jim_age_7_years_ago : ℕ := tom_age_7_years_ago / 2 + 5

/-- Jim's current age -/
def jim_current_age : ℕ := jim_age_7_years_ago + 7

/-- Jim's age in 2 years -/
def jim_age_in_2_years : ℕ := jim_current_age + 2

theorem jim_future_age :
  tom_current_age = 37 →
  tom_age_7_years_ago = 30 →
  jim_age_7_years_ago = 20 →
  jim_current_age = 27 →
  jim_age_in_2_years = 29 := by
  sorry

end jim_future_age_l2472_247288


namespace celias_budget_savings_percentage_l2472_247227

/-- Celia's budget problem --/
theorem celias_budget_savings_percentage :
  let weeks : ℕ := 4
  let food_budget_per_week : ℕ := 100
  let rent : ℕ := 1500
  let streaming_cost : ℕ := 30
  let phone_cost : ℕ := 50
  let savings : ℕ := 198
  let total_spending : ℕ := weeks * food_budget_per_week + rent + streaming_cost + phone_cost
  let savings_percentage : ℚ := (savings : ℚ) / (total_spending : ℚ) * 100
  savings_percentage = 10 := by sorry

end celias_budget_savings_percentage_l2472_247227


namespace photo_collection_l2472_247260

theorem photo_collection (total : ℕ) (tim_less : ℕ) (paul_more : ℕ) : 
  total = 152 →
  tim_less = 100 →
  paul_more = 10 →
  ∃ (tim paul tom : ℕ), 
    tim = total - tim_less ∧
    paul = tim + paul_more ∧
    tom = total - (tim + paul) ∧
    tom = 38 := by
  sorry

end photo_collection_l2472_247260


namespace divisibility_condition_l2472_247245

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end divisibility_condition_l2472_247245
