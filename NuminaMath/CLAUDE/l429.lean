import Mathlib

namespace area_of_large_square_l429_42968

/-- Given three squares with side lengths a, b, and c, prove that the area of the largest square is 100 --/
theorem area_of_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32)
  (h2 : 4*a = 4*c + 16) : 
  a^2 = 100 := by
  sorry

end area_of_large_square_l429_42968


namespace function_properties_l429_42966

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∃ y, -3 < y ∧ y ≤ 0 ∧ f a b c (-1) = y ∧ f a b c 1 = y ∧ f a b c 2 = y) →
  a = -2 ∧ b = -1 ∧ -1 < c ∧ c ≤ 2 := by
  sorry

end function_properties_l429_42966


namespace equation_solution_l429_42938

theorem equation_solution :
  ∃! x : ℚ, 2 * x + 3 = 500 - (4 * x + 5 * x) + 7 ∧ x = 504 / 11 := by
  sorry

end equation_solution_l429_42938


namespace exists_valid_expression_l429_42980

def Expression := List (Fin 4 → ℕ)

def applyOps (nums : Fin 4 → ℕ) (ops : Fin 3 → Char) : ℕ :=
  let e1 := match ops 0 with
    | '+' => nums 0 + nums 1
    | '-' => nums 0 - nums 1
    | '×' => nums 0 * nums 1
    | _ => 0
  let e2 := match ops 1 with
    | '+' => e1 + nums 2
    | '-' => e1 - nums 2
    | '×' => e1 * nums 2
    | _ => 0
  match ops 2 with
    | '+' => e2 + nums 3
    | '-' => e2 - nums 3
    | '×' => e2 * nums 3
    | _ => 0

def isValidOps (ops : Fin 3 → Char) : Prop :=
  (ops 0 = '+' ∨ ops 0 = '-' ∨ ops 0 = '×') ∧
  (ops 1 = '+' ∨ ops 1 = '-' ∨ ops 1 = '×') ∧
  (ops 2 = '+' ∨ ops 2 = '-' ∨ ops 2 = '×') ∧
  (ops 0 ≠ ops 1) ∧ (ops 1 ≠ ops 2) ∧ (ops 0 ≠ ops 2)

theorem exists_valid_expression : ∃ (ops : Fin 3 → Char),
  isValidOps ops ∧ applyOps (λ i => [5, 4, 6, 3][i]) ops = 19 := by
  sorry

end exists_valid_expression_l429_42980


namespace discount_percentage_proof_l429_42933

/-- Prove that the discount percentage is 10% given the conditions of the sale --/
theorem discount_percentage_proof (actual_sp cost_price : ℝ) (profit_rate : ℝ) : 
  actual_sp = 21000 ∧ 
  cost_price = 17500 ∧ 
  profit_rate = 0.08 → 
  (actual_sp - (cost_price * (1 + profit_rate))) / actual_sp = 0.1 := by
sorry

end discount_percentage_proof_l429_42933


namespace recycling_program_earnings_l429_42961

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Theorem stating that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings : 
  total_money_earned 5 5 5 7 = 125 := by
  sorry

#eval total_money_earned 5 5 5 7

end recycling_program_earnings_l429_42961


namespace count_integers_with_repeated_digits_is_140_l429_42969

/-- A function that counts the number of positive three-digit integers 
    between 500 and 999 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let range_start := 500
  let range_end := 999
  let digits := 3
  -- Count of integers where last two digits are the same
  let case1 := 10 * (range_end.div 100 - range_start.div 100 + 1)
  -- Count of integers where first two digits are the same (and different from third)
  let case2 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  -- Count of integers where first and third digits are the same (and different from second)
  let case3 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  case1 + case2 + case3

/-- Theorem stating that the count of integers with repeated digits is 140 -/
theorem count_integers_with_repeated_digits_is_140 :
  count_integers_with_repeated_digits = 140 := by
  sorry

end count_integers_with_repeated_digits_is_140_l429_42969


namespace school_distribution_l429_42954

theorem school_distribution (a b : ℝ) : 
  a + b = 100 →
  0.3 * a + 0.4 * b = 34 →
  a = 60 :=
by sorry

end school_distribution_l429_42954


namespace negation_of_tangent_equality_l429_42909

theorem negation_of_tangent_equality (x : ℝ) :
  (¬ ∀ x : ℝ, Real.tan (-x) = Real.tan x) ↔ (∃ x : ℝ, Real.tan (-x) ≠ Real.tan x) := by sorry

end negation_of_tangent_equality_l429_42909


namespace armands_guessing_game_l429_42992

theorem armands_guessing_game (x : ℤ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end armands_guessing_game_l429_42992


namespace arithmetic_sequence_seventh_term_l429_42995

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_eq1 : a 2 + a 4 + a 5 = a 3 + a 6)
  (h_eq2 : a 9 + a 10 = 3) :
  a 7 = 1 := by
sorry

end arithmetic_sequence_seventh_term_l429_42995


namespace tetrahedron_inequality_l429_42965

theorem tetrahedron_inequality (a b c d h_a h_b h_c h_d V : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hV : V > 0) 
  (h1 : V = (1/3) * a * h_a) 
  (h2 : V = (1/3) * b * h_b) 
  (h3 : V = (1/3) * c * h_c) 
  (h4 : V = (1/3) * d * h_d) : 
  (a + b + c + d) * (h_a + h_b + h_c + h_d) ≥ 48 * V := by
  sorry

end tetrahedron_inequality_l429_42965


namespace linked_rings_height_l429_42974

/-- Represents the properties of a sequence of linked rings -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  diameterDecrease : ℝ
  bottomOutsideDiameter : ℝ

/-- Calculates the total height of the linked rings -/
def totalHeight (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total height of the linked rings with given properties is 273 cm -/
theorem linked_rings_height :
  let rings : LinkedRings := {
    thickness := 2,
    topOutsideDiameter := 20,
    diameterDecrease := 0.5,
    bottomOutsideDiameter := 10
  }
  totalHeight rings = 273 := by sorry

end linked_rings_height_l429_42974


namespace integral_f_zero_to_one_l429_42957

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem integral_f_zero_to_one :
  ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end integral_f_zero_to_one_l429_42957


namespace prime_power_sum_l429_42925

theorem prime_power_sum (p a n : ℕ) : 
  Prime p → 
  a > 0 → 
  n > 0 → 
  2^p + 3^p = a^n → 
  n = 1 := by
sorry

end prime_power_sum_l429_42925


namespace total_seashells_l429_42953

theorem total_seashells (day1 day2 day3 : ℕ) 
  (h1 : day1 = 27) 
  (h2 : day2 = 46) 
  (h3 : day3 = 19) : 
  day1 + day2 + day3 = 92 := by
  sorry

end total_seashells_l429_42953


namespace rational_root_of_polynomial_l429_42976

def f (x : ℚ) : ℚ := 3 * x^3 - 7 * x^2 - 8 * x + 4

theorem rational_root_of_polynomial :
  ∀ x : ℚ, f x = 0 ↔ x = 1/3 := by sorry

end rational_root_of_polynomial_l429_42976


namespace cube_paint_puzzle_l429_42971

theorem cube_paint_puzzle (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → 
  n = 2 :=
by sorry

end cube_paint_puzzle_l429_42971


namespace competitive_exam_candidates_l429_42984

theorem competitive_exam_candidates (candidates : ℕ) : 
  (candidates * 8 / 100 : ℚ) + 220 = (candidates * 12 / 100 : ℚ) →
  candidates = 5500 := by
sorry

end competitive_exam_candidates_l429_42984


namespace unique_positive_solution_l429_42914

/-- The polynomial function f(x) = x^8 + 5x^7 + 10x^6 + 1728x^5 - 1380x^4 -/
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 1728*x^5 - 1380*x^4

/-- The statement that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by sorry

end unique_positive_solution_l429_42914


namespace tangent_line_at_negative_one_l429_42935

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x - y + 3 = 0

-- Theorem statement
theorem tangent_line_at_negative_one :
  tangent_line x₀ (f x₀) ∧
  ∀ x : ℝ, tangent_line x (f x₀ + f' x₀ * (x - x₀)) :=
sorry

end tangent_line_at_negative_one_l429_42935


namespace smallest_special_number_l429_42949

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_below (n k : ℕ) : Prop := ∀ p : ℕ, p < k → is_prime p → n % p ≠ 0

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n < 4091 → 
  (¬ is_prime n ∧ ¬ is_perfect_square n ∧ has_no_prime_factor_below n 60) → False :=
sorry

#check smallest_special_number

end smallest_special_number_l429_42949


namespace difference_of_squares_l429_42944

theorem difference_of_squares (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end difference_of_squares_l429_42944


namespace max_matches_theorem_l429_42902

/-- The maximum number of matches that cannot form a triangle with any two sides differing by at least 10 matches -/
def max_matches : ℕ := 62

/-- A function that checks if three numbers can form a triangle -/
def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if any two sides differ by at least 10 -/
def sides_differ_by_10 (a b c : ℕ) : Prop :=
  (a ≥ b + 10 ∨ b ≥ a + 10) ∧ (b ≥ c + 10 ∨ c ≥ b + 10) ∧ (c ≥ a + 10 ∨ a ≥ c + 10)

theorem max_matches_theorem :
  ∀ n : ℕ, n > max_matches →
    ∃ a b c : ℕ, a + b + c = n ∧ is_triangle a b c ∧ sides_differ_by_10 a b c :=
sorry

end max_matches_theorem_l429_42902


namespace parallelogram_fourth_vertex_l429_42947

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Check if four points form a valid parallelogram -/
def isValidParallelogram (p : Parallelogram) : Prop :=
  (p.v1.x + p.v3.x = p.v2.x + p.v4.x) ∧ 
  (p.v1.y + p.v3.y = p.v2.y + p.v4.y)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = Point.mk (-1) 0)
  (h2 : p.v2 = Point.mk 3 0)
  (h3 : p.v3 = Point.mk 1 (-5)) :
  isValidParallelogram p →
  (p.v4 = Point.mk 5 (-5) ∨ p.v4 = Point.mk (-3) (-5) ∨ p.v4 = Point.mk 1 5) :=
by sorry


end parallelogram_fourth_vertex_l429_42947


namespace consecutive_even_numbers_sum_l429_42917

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧  -- four consecutive even numbers
    a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 = 344) →        -- sum of squares is 344
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=       -- sum of the numbers is 36
by
  sorry

end consecutive_even_numbers_sum_l429_42917


namespace cos_double_angle_on_graph_l429_42993

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -3x
def terminal_side_on_graph (α : Real) : Prop :=
  ∃ x : Real, Real.tan α = -3 ∧ x ≠ 0

-- State the theorem
theorem cos_double_angle_on_graph (α : Real) 
  (h : terminal_side_on_graph α) : Real.cos (2 * α) = -4/5 := by
  sorry

end cos_double_angle_on_graph_l429_42993


namespace intersection_A_B_l429_42994

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 - 1 > 0}

theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l429_42994


namespace abs_five_implies_plus_minus_five_l429_42982

theorem abs_five_implies_plus_minus_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end abs_five_implies_plus_minus_five_l429_42982


namespace f_5_solution_set_l429_42921

def f (x : ℝ) : ℝ := x^2 + 12*x + 30

def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

theorem f_5_solution_set :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 - (6 : ℝ)^(1/32) ∨ x = -6 + (6 : ℝ)^(1/32) := by
  sorry

end f_5_solution_set_l429_42921


namespace f_composed_eq_6_has_three_solutions_l429_42900

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 7

-- Define the composite function f(f(x))
noncomputable def f_composed (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem f_composed_eq_6_has_three_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ f_composed x = 6 :=
sorry

end f_composed_eq_6_has_three_solutions_l429_42900


namespace shaded_area_calculation_l429_42987

/-- Given a square banner with side length 12 feet, one large shaded square
    with side length S, and twelve smaller congruent shaded squares with
    side length T, where 12:S = S:T = 4, the total shaded area is 15.75 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S = 12 / 4 →
  T = S / 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end shaded_area_calculation_l429_42987


namespace quadratic_equation_problem_l429_42991

theorem quadratic_equation_problem (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x * y = 15) →
  m = 3 * n →
  m + n = 180 := by
sorry


end quadratic_equation_problem_l429_42991


namespace adrian_water_needed_l429_42934

/-- Represents the recipe ratios and amount of orange juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_juice_ratio : ℚ
  orange_juice_cups : ℚ

/-- Calculates the amount of water needed for the punch recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_juice_ratio * r.orange_juice_cups

/-- Theorem stating that Adrian needs 60 cups of water --/
theorem adrian_water_needed :
  let recipe := Recipe.mk 5 3 4
  water_needed recipe = 60 := by
  sorry

end adrian_water_needed_l429_42934


namespace problem_solution_l429_42904

def problem (m : ℝ) : Prop :=
  let a : Fin 2 → ℝ := ![m + 2, 1]
  let b : Fin 2 → ℝ := ![1, -2*m]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- a ⊥ b condition
  ‖(a 0 + b 0, a 1 + b 1)‖ = Real.sqrt 34

theorem problem_solution :
  ∃ m : ℝ, problem m := by sorry

end problem_solution_l429_42904


namespace sales_price_calculation_l429_42937

theorem sales_price_calculation (C S : ℝ) : 
  (1.20 * C = 24) →  -- Gross profit is $24
  (S = C + 1.20 * C) →  -- Sales price is cost plus gross profit
  (S = 44) :=  -- Prove that sales price is $44
by
  sorry

end sales_price_calculation_l429_42937


namespace trishas_walk_l429_42964

/-- Proves that given a total distance and two equal segments, the remaining distance is as expected. -/
theorem trishas_walk (total : ℝ) (segment : ℝ) (h1 : total = 0.8888888888888888) 
  (h2 : segment = 0.1111111111111111) : 
  total - 2 * segment = 0.6666666666666666 := by
  sorry

end trishas_walk_l429_42964


namespace chord_length_on_circle_l429_42929

/-- The length of the chord intercepted by y=x on (x-0)^2+(y-2)^2=4 is 2√2 -/
theorem chord_length_on_circle (x y : ℝ) : 
  (x - 0)^2 + (y - 2)^2 = 4 → y = x → 
  ∃ (a b : ℝ), (a - 0)^2 + (b - 2)^2 = 4 ∧ b = a ∧ 
  Real.sqrt ((a - x)^2 + (b - y)^2) = 2 * Real.sqrt 2 :=
sorry

end chord_length_on_circle_l429_42929


namespace repairs_count_l429_42990

/-- Represents the mechanic shop scenario --/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Calculates the number of repairs given the shop's data --/
def calculate_repairs (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - (shop.oil_change_price * shop.oil_changes + shop.car_wash_price * shop.car_washes)) / shop.repair_price

/-- Theorem stating that given the specific conditions, the number of repairs is 10 --/
theorem repairs_count (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.repair_price = 30)
  (h3 : shop.car_wash_price = 5)
  (h4 : shop.oil_changes = 5)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  calculate_repairs shop = 10 := by
  sorry

#eval calculate_repairs { 
  oil_change_price := 20, 
  repair_price := 30, 
  car_wash_price := 5, 
  oil_changes := 5, 
  car_washes := 15, 
  total_earnings := 475 
}

end repairs_count_l429_42990


namespace computer_preference_ratio_l429_42955

theorem computer_preference_ratio (total : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  (total - (mac_preference + no_preference)) = mac_preference :=
by sorry

end computer_preference_ratio_l429_42955


namespace highest_power_of_three_dividing_N_l429_42913

def N : ℕ := sorry

theorem highest_power_of_three_dividing_N : 
  (∃ m : ℕ, N = 3^3 * m) ∧ (∀ k > 3, ¬∃ m : ℕ, N = 3^k * m) := by sorry

end highest_power_of_three_dividing_N_l429_42913


namespace exists_valid_marking_l429_42985

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a marking of squares on the chessboard -/
def BoardMarking := Position → Bool

/-- Calculates the minimum number of rook moves between two positions given a board marking -/
def minRookMoves (start finish : Position) (marking : BoardMarking) : ℕ :=
  sorry

/-- Theorem stating the existence of a board marking satisfying the given conditions -/
theorem exists_valid_marking : 
  ∃ (marking : BoardMarking),
    (minRookMoves ⟨0, 0⟩ ⟨2, 3⟩ marking = 3) ∧ 
    (minRookMoves ⟨2, 3⟩ ⟨7, 7⟩ marking = 2) ∧
    (minRookMoves ⟨0, 0⟩ ⟨7, 7⟩ marking = 4) :=
  sorry

end exists_valid_marking_l429_42985


namespace vase_capacity_l429_42919

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations roses vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Theorem: Given 7 carnations, 47 roses, and 9 vases, each vase can hold 6 flowers -/
theorem vase_capacity :
  flowers_per_vase 7 47 9 = 6 := by
sorry

end vase_capacity_l429_42919


namespace extended_quadrilateral_area_l429_42936

/-- A quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  /-- Side length EF -/
  ef : ℝ
  /-- Side length FG -/
  fg : ℝ
  /-- Side length GH -/
  gh : ℝ
  /-- Side length HE -/
  he : ℝ
  /-- Area of EFGH -/
  area : ℝ
  /-- Extension ratio for EF -/
  ef_ratio : ℝ
  /-- Extension ratio for FG -/
  fg_ratio : ℝ
  /-- Extension ratio for GH -/
  gh_ratio : ℝ
  /-- Extension ratio for HE -/
  he_ratio : ℝ

/-- The area of the extended quadrilateral E'F'G'H' -/
def extended_area (q : ExtendedQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of E'F'G'H' given specific conditions -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral)
  (h1 : q.ef = 5)
  (h2 : q.fg = 6)
  (h3 : q.gh = 7)
  (h4 : q.he = 8)
  (h5 : q.area = 12)
  (h6 : q.ef_ratio = 2)
  (h7 : q.fg_ratio = 3/2)
  (h8 : q.gh_ratio = 4/3)
  (h9 : q.he_ratio = 5/4) :
  extended_area q = 84 := by sorry

end extended_quadrilateral_area_l429_42936


namespace inequality_proof_l429_42970

def M : Set ℝ := {x | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end inequality_proof_l429_42970


namespace number_problem_l429_42975

theorem number_problem : ∃ x : ℝ, x * 0.007 = 0.0063 ∧ x = 0.9 := by
  sorry

end number_problem_l429_42975


namespace parking_arrangements_count_l429_42908

-- Define the number of parking spaces
def num_spaces : ℕ := 7

-- Define the number of trucks
def num_trucks : ℕ := 2

-- Define the number of buses
def num_buses : ℕ := 2

-- Define a function to calculate the number of parking arrangements
def num_parking_arrangements (spaces : ℕ) (trucks : ℕ) (buses : ℕ) : ℕ :=
  (spaces.choose trucks) * ((spaces - trucks).choose buses) * (trucks.factorial) * (buses.factorial)

-- Theorem statement
theorem parking_arrangements_count :
  num_parking_arrangements num_spaces num_trucks num_buses = 840 := by
  sorry


end parking_arrangements_count_l429_42908


namespace quadratic_equation_solution_l429_42952

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end quadratic_equation_solution_l429_42952


namespace linear_increasing_positive_slope_l429_42907

def f (k : ℝ) (x : ℝ) : ℝ := k * x - 100

theorem linear_increasing_positive_slope (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y, x < y → f k x < f k y) → k > 0 := by
  sorry

end linear_increasing_positive_slope_l429_42907


namespace y1_greater_y2_l429_42903

/-- A line in the 2D plane represented by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem y1_greater_y2 (l : Line) (p1 p2 : Point) :
  l.m = -1 →
  l.b = 1 →
  p1.x = -2 →
  p2.x = 3 →
  p1.liesOn l →
  p2.liesOn l →
  p1.y > p2.y := by
  sorry

end y1_greater_y2_l429_42903


namespace not_sufficient_for_parallelogram_l429_42928

/-- A quadrilateral with vertices A, B, C, and D -/
structure Quadrilateral (V : Type*) :=
  (A B C D : V)

/-- Parallelism relation between line segments -/
def Parallel {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Equality of line segments -/
def SegmentEqual {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Definition of a parallelogram -/
def IsParallelogram {V : Type*} (quad : Quadrilateral V) : Prop := sorry

/-- The main theorem: AB parallel to CD and AD = BC does not imply ABCD is a parallelogram -/
theorem not_sufficient_for_parallelogram {V : Type*} (quad : Quadrilateral V) :
  Parallel (quad.A, quad.B) (quad.C, quad.D) →
  SegmentEqual (quad.A, quad.D) (quad.B, quad.C) →
  ¬ (IsParallelogram quad) := by
  sorry

end not_sufficient_for_parallelogram_l429_42928


namespace initial_members_family_e_l429_42988

/-- The number of families in Indira Nagar -/
def num_families : ℕ := 6

/-- The initial number of members in family a -/
def family_a : ℕ := 7

/-- The initial number of members in family b -/
def family_b : ℕ := 8

/-- The initial number of members in family c -/
def family_c : ℕ := 10

/-- The initial number of members in family d -/
def family_d : ℕ := 13

/-- The initial number of members in family f -/
def family_f : ℕ := 10

/-- The number of members that left each family -/
def members_left : ℕ := 1

/-- The average number of members in each family after some left -/
def new_average : ℕ := 8

/-- The initial number of members in family e -/
def family_e : ℕ := 6

theorem initial_members_family_e :
  family_a + family_b + family_c + family_d + family_e + family_f - 
  (num_families * members_left) = num_families * new_average := by
  sorry

end initial_members_family_e_l429_42988


namespace quadratic_intersection_count_l429_42930

/-- The quadratic function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The number of intersection points between f and the coordinate axes -/
def intersection_count : ℕ := 2

theorem quadratic_intersection_count :
  (∃! x, f x = 0) ∧ (f 0 ≠ 0) → intersection_count = 2 :=
by sorry

end quadratic_intersection_count_l429_42930


namespace proportion_problem_l429_42924

/-- Given four real numbers a, b, c, d in proportion, where a = 2, b = 3, and d = 6, prove that c = 4. -/
theorem proportion_problem (a b c d : ℝ) 
  (h_prop : a / b = c / d) 
  (h_a : a = 2) 
  (h_b : b = 3) 
  (h_d : d = 6) : 
  c = 4 := by
  sorry

end proportion_problem_l429_42924


namespace inequality_region_l429_42951

theorem inequality_region (x y : ℝ) : 
  Real.sqrt (x * y) ≥ x - 2 * y ↔ 
  ((x ≥ 0 ∧ y ≥ 0 ∧ y ≥ x / 2) ∨ 
   (x ≤ 0 ∧ y ≤ 0 ∧ y ≥ x / 2) ∨ 
   (x = 0 ∧ y ≥ 0) ∨ 
   (x ≥ 0 ∧ y = 0)) := by
  sorry

end inequality_region_l429_42951


namespace solve_equation_l429_42972

theorem solve_equation : ∃ x : ℚ, 25 * x = 675 ∧ x = 27 := by
  sorry

end solve_equation_l429_42972


namespace sum_of_variables_l429_42918

theorem sum_of_variables (a b c d : ℚ) 
  (h1 : 2*a + 3 = 2*b + 5)
  (h2 : 2*b + 5 = 2*c + 7)
  (h3 : 2*c + 7 = 2*d + 9)
  (h4 : 2*d + 9 = 2*(a + b + c + d) + 13) :
  a + b + c + d = -14/3 := by
sorry

end sum_of_variables_l429_42918


namespace power_function_through_2_4_l429_42941

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_2_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 4) : 
  f 3 = 9 := by
  sorry

end power_function_through_2_4_l429_42941


namespace data_transmission_time_l429_42967

-- Define the number of packets
def num_packets : ℕ := 100

-- Define the number of bytes per packet
def bytes_per_packet : ℕ := 256

-- Define the transmission rate in bytes per second
def transmission_rate : ℕ := 200

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Theorem to prove
theorem data_transmission_time :
  (num_packets * bytes_per_packet) / transmission_rate / seconds_per_minute = 2 := by
  sorry


end data_transmission_time_l429_42967


namespace hyperbola_equation_l429_42931

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  -- The equation of the hyperbola in the form (y²/a² - x²/b² = 1)
  a : ℝ
  b : ℝ
  -- The hyperbola passes through the point (2, -2)
  passes_through : a^2 * 4 - b^2 * 4 = a^2 * b^2
  -- The hyperbola has asymptotes y = ± (√2/2)x
  asymptotes : a / b = Real.sqrt 2 / 2

/-- The equation of the hyperbola is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : h.a^2 = 2 ∧ h.b^2 = 4 :=
  sorry

end hyperbola_equation_l429_42931


namespace isosceles_triangle_n_count_l429_42915

/-- The number of valid positive integer values for n in the isosceles triangle problem -/
def valid_n_count : ℕ := 7

/-- Checks if a given n satisfies the triangle inequality and angle conditions -/
def is_valid_n (n : ℕ) : Prop :=
  let ab := n + 10
  let bc := 4 * n + 2
  (ab + ab > bc) ∧ 
  (ab + bc > ab) ∧ 
  (bc + ab > ab) ∧
  (bc < ab)  -- This ensures ∠A > ∠B > ∠C in the isosceles triangle

theorem isosceles_triangle_n_count :
  (∃ (S : Finset ℕ), S.card = valid_n_count ∧ 
    (∀ n, n ∈ S ↔ (n > 0 ∧ is_valid_n n)) ∧
    (∀ n, n ∉ S → (n = 0 ∨ ¬is_valid_n n))) := by
  sorry

end isosceles_triangle_n_count_l429_42915


namespace quadratic_inequality_solution_l429_42942

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end quadratic_inequality_solution_l429_42942


namespace triangle_minimum_shortest_side_l429_42939

theorem triangle_minimum_shortest_side :
  ∀ a b : ℕ,
  a < b ∧ b < 3 * a →  -- Condition for unequal sides
  a + b + 3 * a = 120 →  -- Total number of matches
  a ≥ 18 →  -- Minimum value of shortest side
  ∃ (a₀ : ℕ), a₀ = 18 ∧ 
    ∃ (b₀ : ℕ), a₀ < b₀ ∧ b₀ < 3 * a₀ ∧ 
    a₀ + b₀ + 3 * a₀ = 120 :=
by
  sorry

end triangle_minimum_shortest_side_l429_42939


namespace worker_payment_l429_42920

theorem worker_payment (total_days : ℕ) (days_not_worked : ℕ) (return_amount : ℕ) 
  (h1 : total_days = 30)
  (h2 : days_not_worked = 24)
  (h3 : return_amount = 25)
  : ∃ x : ℕ, 
    (total_days - days_not_worked) * x = days_not_worked * return_amount ∧ 
    x = 100 := by
  sorry

end worker_payment_l429_42920


namespace josh_marbles_problem_l429_42948

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 16

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 8

/-- The initial number of marbles Josh had -/
def initial_marbles : ℕ := 4

theorem josh_marbles_problem :
  marbles_lost = marbles_found + 8 :=
by sorry

end josh_marbles_problem_l429_42948


namespace first_three_digits_of_large_number_l429_42986

-- Define the expression
def large_number : ℝ := (10^100 + 1)^(5/3)

-- Define a function to extract the first three decimal digits
def first_three_decimal_digits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- State the theorem
theorem first_three_digits_of_large_number :
  first_three_decimal_digits large_number = (6, 6, 6) := by sorry

end first_three_digits_of_large_number_l429_42986


namespace absolute_value_inequality_solution_l429_42916

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by
  sorry

end absolute_value_inequality_solution_l429_42916


namespace extra_interest_proof_l429_42958

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

def investment_amount : ℝ := 7000
def high_rate : ℝ := 0.18
def low_rate : ℝ := 0.12
def investment_time : ℝ := 2

theorem extra_interest_proof :
  simple_interest investment_amount high_rate investment_time -
  simple_interest investment_amount low_rate investment_time = 840 := by
  sorry

end extra_interest_proof_l429_42958


namespace equal_area_and_perimeter_l429_42973

-- Define the quadrilaterals
def quadrilateralA : List (ℝ × ℝ) := [(0,0), (3,0), (3,2), (0,3)]
def quadrilateralB : List (ℝ × ℝ) := [(0,0), (3,0), (3,3), (0,2)]

-- Function to calculate area of a quadrilateral
def area (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Function to calculate perimeter of a quadrilateral
def perimeter (quad : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the areas and perimeters are equal
theorem equal_area_and_perimeter :
  area quadrilateralA = area quadrilateralB ∧
  perimeter quadrilateralA = perimeter quadrilateralB := by
  sorry

end equal_area_and_perimeter_l429_42973


namespace two_digit_powers_of_three_l429_42932

theorem two_digit_powers_of_three : 
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ 
  (∃ (s : Finset ℕ), (∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ s.card = 2) := by
  sorry

end two_digit_powers_of_three_l429_42932


namespace base_b_square_l429_42927

theorem base_b_square (b : ℕ) (h : b > 1) :
  (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 → b = 7 :=
by sorry

end base_b_square_l429_42927


namespace factorization_of_x2y_minus_4y_l429_42945

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end factorization_of_x2y_minus_4y_l429_42945


namespace field_length_calculation_l429_42946

/-- Given a rectangular field wrapped with tape, calculate its length. -/
theorem field_length_calculation (total_tape : ℕ) (width : ℕ) (leftover_tape : ℕ) :
  total_tape = 250 →
  width = 20 →
  leftover_tape = 90 →
  2 * (width + (total_tape - leftover_tape) / 2) = total_tape - leftover_tape →
  (total_tape - leftover_tape) / 2 - width = 60 := by
  sorry

end field_length_calculation_l429_42946


namespace max_distance_circle_to_line_l429_42962

/-- The maximum distance from the center of the circle x² + y² = 4 to the line mx + (5-2m)y - 2 = 0, where m ∈ ℝ, is 2√5/5. -/
theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | m*x + (5 - 2*m)*y - 2 = 0}
  ∀ m : ℝ, (⨆ p ∈ line m, dist (0, 0) p) = 2 * Real.sqrt 5 / 5 :=
by sorry

end max_distance_circle_to_line_l429_42962


namespace min_sum_of_squares_l429_42978

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 64 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
by sorry

end min_sum_of_squares_l429_42978


namespace expected_value_of_coin_flips_l429_42905

def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50
def dollar : ℚ := 100

def coin_flip_probability : ℚ := 1/2

theorem expected_value_of_coin_flips :
  coin_flip_probability * (penny + nickel + dime + quarter + half_dollar + dollar) = 95.5 := by
  sorry

end expected_value_of_coin_flips_l429_42905


namespace walnut_weight_in_mixture_l429_42999

/-- Given a mixture of nuts with a specific ratio and total weight, 
    calculate the weight of walnuts -/
theorem walnut_weight_in_mixture 
  (ratio_almonds : ℕ) 
  (ratio_walnuts : ℕ) 
  (ratio_peanuts : ℕ) 
  (ratio_cashews : ℕ) 
  (total_weight : ℕ) 
  (h1 : ratio_almonds = 5) 
  (h2 : ratio_walnuts = 3) 
  (h3 : ratio_peanuts = 2) 
  (h4 : ratio_cashews = 4) 
  (h5 : total_weight = 420) : 
  (ratio_walnuts * total_weight) / (ratio_almonds + ratio_walnuts + ratio_peanuts + ratio_cashews) = 90 := by
  sorry


end walnut_weight_in_mixture_l429_42999


namespace bus_total_capacity_l429_42912

/-- Represents the seating capacity of a bus with specified seat arrangements. -/
def bus_capacity (left_seats : ℕ) (right_seats_difference : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seats_difference
  let left_capacity := left_seats * people_per_seat
  let right_capacity := right_seats * people_per_seat
  left_capacity + right_capacity + back_seat_capacity

/-- Theorem stating the total seating capacity of the bus under given conditions. -/
theorem bus_total_capacity : bus_capacity 15 3 3 8 = 89 := by
  sorry

end bus_total_capacity_l429_42912


namespace fraction_of_one_third_is_one_fifth_l429_42989

theorem fraction_of_one_third_is_one_fifth : (1 : ℚ) / 5 / ((1 : ℚ) / 3) = 3 / 5 := by
  sorry

end fraction_of_one_third_is_one_fifth_l429_42989


namespace circle_equation_l429_42926

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def isTangent (c : Circle) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

-- Main theorem
theorem circle_equation (C : Circle) 
  (h1 : isTangent C line1)
  (h2 : isTangent C line2)
  (h3 : line3 C.center.1 C.center.2) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 2 ↔ ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) :=
sorry

end circle_equation_l429_42926


namespace compute_expression_l429_42922

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end compute_expression_l429_42922


namespace alex_ate_six_ounces_l429_42963

/-- The amount of jelly beans Alex ate -/
def jelly_beans_eaten (initial : ℕ) (num_piles : ℕ) (weight_per_pile : ℕ) : ℕ :=
  initial - (num_piles * weight_per_pile)

/-- Theorem stating that Alex ate 6 ounces of jelly beans -/
theorem alex_ate_six_ounces : 
  jelly_beans_eaten 36 3 10 = 6 := by
  sorry

end alex_ate_six_ounces_l429_42963


namespace chess_tournament_games_l429_42979

/-- Calculate the number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 10

/-- Theorem: In a chess tournament with 10 players, where each player plays twice 
    with every other player, the total number of games played is 180. -/
theorem chess_tournament_games : 
  2 * tournament_games num_players = 180 := by
sorry

end chess_tournament_games_l429_42979


namespace strawberry_harvest_l429_42940

/-- Proves that the number of strawberries harvested from each plant is 14 --/
theorem strawberry_harvest (
  strawberry_plants : ℕ)
  (tomato_plants : ℕ)
  (tomatoes_per_plant : ℕ)
  (fruits_per_basket : ℕ)
  (strawberry_basket_price : ℕ)
  (tomato_basket_price : ℕ)
  (total_revenue : ℕ)
  (h1 : strawberry_plants = 5)
  (h2 : tomato_plants = 7)
  (h3 : tomatoes_per_plant = 16)
  (h4 : fruits_per_basket = 7)
  (h5 : strawberry_basket_price = 9)
  (h6 : tomato_basket_price = 6)
  (h7 : total_revenue = 186) :
  (total_revenue - tomato_basket_price * (tomato_plants * tomatoes_per_plant / fruits_per_basket)) / strawberry_basket_price * fruits_per_basket / strawberry_plants = 14 :=
by sorry

end strawberry_harvest_l429_42940


namespace perfect_square_natural_number_l429_42983

theorem perfect_square_natural_number (n : ℕ) :
  (∃ k : ℕ, n^2 + 5*n + 13 = k^2) → n = 4 := by
  sorry

end perfect_square_natural_number_l429_42983


namespace wayne_blocks_problem_l429_42910

theorem wayne_blocks_problem (initial_blocks : ℕ) (father_blocks : ℕ) : 
  initial_blocks = 9 →
  father_blocks = 6 →
  (3 * (initial_blocks + father_blocks)) - (initial_blocks + father_blocks) = 30 :=
by sorry

end wayne_blocks_problem_l429_42910


namespace prove_average_speed_l429_42950

-- Define the distances traveled on each day
def distance_day1 : ℝ := 160
def distance_day2 : ℝ := 280

-- Define the time difference between the two trips
def time_difference : ℝ := 3

-- Define the average speed
def average_speed : ℝ := 40

-- Theorem statement
theorem prove_average_speed :
  (distance_day2 / average_speed) - (distance_day1 / average_speed) = time_difference :=
by
  sorry

end prove_average_speed_l429_42950


namespace option1_cheaper_at_30_l429_42997

/-- Represents the cost calculation for two shopping options -/
def shopping_options (x : ℕ) : Prop :=
  let shoe_price : ℕ := 200
  let sock_price : ℕ := 40
  let num_shoes : ℕ := 20
  let option1_cost : ℕ := sock_price * x + num_shoes * shoe_price
  let option2_cost : ℕ := (sock_price * x * 9 + num_shoes * shoe_price * 9) / 10
  x > num_shoes ∧ option1_cost < option2_cost

/-- Theorem stating that Option 1 is cheaper when buying 30 pairs of socks -/
theorem option1_cheaper_at_30 : shopping_options 30 := by
  sorry

#check option1_cheaper_at_30

end option1_cheaper_at_30_l429_42997


namespace perpendicular_planes_parallel_skew_lines_parallel_planes_l429_42923

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem for proposition ②
theorem perpendicular_planes_parallel
  (n : Line) (α β : Plane)
  (h1 : perpendicular n α)
  (h2 : perpendicular n β) :
  parallel α β :=
sorry

-- Theorem for proposition ⑤
theorem skew_lines_parallel_planes
  (m n : Line) (α β : Plane)
  (h1 : skew m n)
  (h2 : contains α n)
  (h3 : line_parallel n β)
  (h4 : contains β m)
  (h5 : line_parallel m α) :
  parallel α β :=
sorry

end perpendicular_planes_parallel_skew_lines_parallel_planes_l429_42923


namespace slope_of_line_l429_42960

theorem slope_of_line (x y : ℝ) :
  4 * y = -6 * x + 12 → (y - 3 = -3/2 * (x - 0)) := by
  sorry

end slope_of_line_l429_42960


namespace direction_vector_b_value_l429_42911

/-- Given a line passing through points (-6, 0) and (-3, 3) with direction vector (3, b), prove b = 3 -/
theorem direction_vector_b_value (b : ℝ) : b = 3 :=
  by
  -- Define the two points on the line
  let p1 : Fin 2 → ℝ := ![- 6, 0]
  let p2 : Fin 2 → ℝ := ![- 3, 3]
  
  -- Define the direction vector of the line
  let dir : Fin 2 → ℝ := ![3, b]
  
  -- Assert that the direction vector is parallel to the vector between the two points
  have h : ∃ (k : ℝ), k ≠ 0 ∧ (λ i => p2 i - p1 i) = (λ i => k * dir i) := by sorry
  
  sorry

end direction_vector_b_value_l429_42911


namespace not_divisible_by_169_l429_42901

theorem not_divisible_by_169 (x : ℤ) : ¬(169 ∣ (x^2 + 5*x + 16)) := by
  sorry

end not_divisible_by_169_l429_42901


namespace kickball_difference_l429_42998

theorem kickball_difference (wednesday : ℕ) (total : ℕ) : 
  wednesday = 37 →
  total = 65 →
  wednesday - (total - wednesday) = 9 := by
sorry

end kickball_difference_l429_42998


namespace line_intersects_plane_l429_42906

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships between points, lines, and planes
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_intersects_plane (l : Line) (α : Plane) :
  (∃ p q : Point, on_line p l ∧ on_line q l ∧ in_plane p α ∧ ¬in_plane q α) →
  intersects l α :=
sorry

end line_intersects_plane_l429_42906


namespace expression_value_l429_42943

theorem expression_value (a b : ℤ) (h1 : a = -4) (h2 : b = 3) :
  -2*a - b^3 + 2*a*b = -43 := by
  sorry

end expression_value_l429_42943


namespace integral_curves_of_differential_equation_l429_42959

/-- The differential equation -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  6 * x * dx - 6 * y * dy = 2 * x^2 * y * dy - 3 * x * y^2 * dx

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + 3)^3 / (2 + y^2) = C

/-- Theorem stating that the integral curves of the given differential equation
    are described by the integral_curve equation -/
theorem integral_curves_of_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
  (∀ (dx dy : ℝ), differential_equation x y dx dy) →
  ∃ (C : ℝ), integral_curve x y C :=
sorry

end integral_curves_of_differential_equation_l429_42959


namespace exp_7pi_i_div_3_rectangular_form_l429_42956

theorem exp_7pi_i_div_3_rectangular_form :
  Complex.exp (7 * Real.pi * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end exp_7pi_i_div_3_rectangular_form_l429_42956


namespace modulus_of_w_l429_42981

theorem modulus_of_w (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end modulus_of_w_l429_42981


namespace max_distinct_distance_selection_l429_42977

/-- A regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

/-- The distance between two vertices of a regular polygon -/
def distance (p : RegularPolygon) (i j : Fin p.sides) : ℝ := sorry

/-- A selection of vertices from a regular polygon -/
def VertexSelection (p : RegularPolygon) := Fin p.sides → Bool

/-- The number of vertices in a selection -/
def selectionSize (p : RegularPolygon) (s : VertexSelection p) : ℕ := sorry

/-- Whether all distances between selected vertices are distinct -/
def distinctDistances (p : RegularPolygon) (s : VertexSelection p) : Prop := sorry

theorem max_distinct_distance_selection (p : RegularPolygon) 
  (h : p.sides = 21) :
  (∃ (s : VertexSelection p), selectionSize p s = 5 ∧ distinctDistances p s) ∧
  (∀ (s : VertexSelection p), selectionSize p s > 5 → ¬ distinctDistances p s) :=
sorry

end max_distinct_distance_selection_l429_42977


namespace quadratic_other_x_intercept_l429_42996

/-- Given a quadratic function f(x) with vertex (5,10) and one x-intercept at (1,0),
    the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = 10) :
  ∃ x, f x = 0 ∧ x = 9 :=
sorry

end quadratic_other_x_intercept_l429_42996
