import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l3963_396311

theorem quadratic_form_k_value :
  ∃ (a h k : ℚ), ∀ x, x^2 - 5*x = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l3963_396311


namespace NUMINAMATH_CALUDE_triangle_altitude_l3963_396352

theorem triangle_altitude (base : ℝ) (square_side : ℝ) (altitude : ℝ) : 
  base = 6 →
  square_side = 6 →
  (1/2) * base * altitude = square_side * square_side →
  altitude = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3963_396352


namespace NUMINAMATH_CALUDE_simplify_expression_l3963_396378

theorem simplify_expression : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3963_396378


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l3963_396387

-- Define the triangles
def Triangle := Fin 3 → ℝ × ℝ

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the side length between two points of a triangle
def side_length (t : Triangle) (i k : Fin 3) : ℝ := sorry

-- Define if a triangle is obtuse-angled
def is_obtuse (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_area_comparison 
  (A B : Triangle) 
  (h_sides : ∀ (i k : Fin 3), side_length A i k ≥ side_length B i k) 
  (h_not_obtuse : ¬ is_obtuse A) : 
  area A ≥ area B := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l3963_396387


namespace NUMINAMATH_CALUDE_pentagon_sum_l3963_396318

/-- Pentagon with specific properties -/
structure Pentagon where
  u : ℤ
  v : ℤ
  h1 : 1 ≤ v
  h2 : v < u
  A : ℝ × ℝ := (u, v)
  B : ℝ × ℝ := (v, u)
  C : ℝ × ℝ := (-v, u)
  D : ℝ × ℝ := (-u, v)
  E : ℝ × ℝ := (-u, -v)
  h3 : (D.1 - E.1) * (A.1 - E.1) + (D.2 - E.2) * (A.2 - E.2) = 0  -- ∠DEA = 90°
  h4 : (u^2 : ℝ) + v^2 = 500  -- Area of pentagon ABCDE is 500

/-- Theorem stating the sum of u and v -/
theorem pentagon_sum (p : Pentagon) : p.u + p.v = 20 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l3963_396318


namespace NUMINAMATH_CALUDE_arccos_cos_eq_three_halves_x_implies_x_zero_l3963_396313

theorem arccos_cos_eq_three_halves_x_implies_x_zero 
  (x : ℝ) 
  (h1 : -π ≤ x ∧ x ≤ π) 
  (h2 : Real.arccos (Real.cos x) = (3 * x) / 2) : 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_three_halves_x_implies_x_zero_l3963_396313


namespace NUMINAMATH_CALUDE_jims_cousin_money_l3963_396353

-- Define the costs of items
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8

-- Define the number of items ordered
def num_cheeseburgers : ℕ := 2
def num_milkshakes : ℕ := 2
def num_cheese_fries : ℕ := 1

-- Define Jim's contribution
def jim_money : ℚ := 20

-- Define the percentage of combined money spent
def percentage_spent : ℚ := 80 / 100

-- Theorem to prove
theorem jims_cousin_money :
  let total_cost := num_cheeseburgers * cheeseburger_cost + 
                    num_milkshakes * milkshake_cost + 
                    num_cheese_fries * cheese_fries_cost
  let total_money := total_cost / percentage_spent
  let cousin_money := total_money - jim_money
  cousin_money = 10 := by sorry

end NUMINAMATH_CALUDE_jims_cousin_money_l3963_396353


namespace NUMINAMATH_CALUDE_percentage_increase_l3963_396393

theorem percentage_increase (original_earnings new_earnings : ℝ) :
  original_earnings = 60 →
  new_earnings = 68 →
  (new_earnings - original_earnings) / original_earnings * 100 = (68 - 60) / 60 * 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3963_396393


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3963_396362

/-- Proves that the cost price of an article is 540 given the specified conditions -/
theorem cost_price_calculation (marked_up_price : ℝ → ℝ) (discounted_price : ℝ → ℝ) :
  (∀ x, marked_up_price x = x * 1.15) →
  (∀ x, discounted_price x = x * (1 - 0.2608695652173913)) →
  discounted_price (marked_up_price 540) = 459 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3963_396362


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3963_396351

/-- Given that c + √d and its radical conjugate have a sum of 0 and a product of 9, prove that c + d = -9 -/
theorem radical_conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = 0) ∧ 
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 9) → 
  c + d = -9 := by sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3963_396351


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3963_396396

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3963_396396


namespace NUMINAMATH_CALUDE_division_simplification_l3963_396300

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by sorry

end NUMINAMATH_CALUDE_division_simplification_l3963_396300


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3963_396348

theorem exponential_equation_solution : ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^4 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3963_396348


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l3963_396395

theorem movie_length_after_cut (final_length cut_length : ℕ) (h1 : final_length = 57) (h2 : cut_length = 3) :
  final_length + cut_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l3963_396395


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3963_396306

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3963_396306


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l3963_396312

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l3963_396312


namespace NUMINAMATH_CALUDE_new_year_markup_percentage_l3963_396322

/-- Proves that given specific markups and profit, the New Year season markup is 25% -/
theorem new_year_markup_percentage
  (initial_markup : ℝ)
  (february_discount : ℝ)
  (final_profit : ℝ)
  (h1 : initial_markup = 0.20)
  (h2 : february_discount = 0.10)
  (h3 : final_profit = 0.35)
  : ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
sorry

end NUMINAMATH_CALUDE_new_year_markup_percentage_l3963_396322


namespace NUMINAMATH_CALUDE_angle_value_l3963_396346

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  is_geometric_sequence a →
  (∀ x : ℝ, x^2 - 2*x*Real.sin α - Real.sqrt 3*Real.sin α = 0 ↔ (x = a 1 ∨ x = a 8)) →
  (a 1 + a 8)^2 = 2*a 3*a 6 + 6 →
  0 < α ∧ α < Real.pi/2 →
  α = Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_angle_value_l3963_396346


namespace NUMINAMATH_CALUDE_sin_squared_plus_cos_squared_equals_one_l3963_396381

-- Define a point on a unit circle
def PointOnUnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the relationship between x, y, and θ on the unit circle
def UnitCirclePoint (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.cos θ ∧ y = Real.sin θ

-- Theorem statement
theorem sin_squared_plus_cos_squared_equals_one (θ : ℝ) :
  ∃ x y : ℝ, UnitCirclePoint θ x y → (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_plus_cos_squared_equals_one_l3963_396381


namespace NUMINAMATH_CALUDE_pears_picked_total_l3963_396304

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

theorem pears_picked_total :
  total_pears = 5 := by sorry

end NUMINAMATH_CALUDE_pears_picked_total_l3963_396304


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l3963_396374

/-- The sum of positive factors of a natural number -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l3963_396374


namespace NUMINAMATH_CALUDE_bird_eggs_problem_l3963_396370

theorem bird_eggs_problem (total_eggs : ℕ) 
  (eggs_per_nest_tree1 : ℕ) (nests_in_tree1 : ℕ) 
  (eggs_in_front_yard : ℕ) (eggs_in_tree2 : ℕ) : 
  total_eggs = 17 →
  eggs_per_nest_tree1 = 5 →
  nests_in_tree1 = 2 →
  eggs_in_front_yard = 4 →
  total_eggs = nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_front_yard + eggs_in_tree2 →
  eggs_in_tree2 = 3 := by
sorry

end NUMINAMATH_CALUDE_bird_eggs_problem_l3963_396370


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3963_396342

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 210 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3963_396342


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3963_396325

theorem rectangle_area_increase (x y : ℝ) :
  let original_area := 1
  let new_length := 1 + x / 100
  let new_width := 1 + y / 100
  let new_area := new_length * new_width
  let area_increase_percentage := (new_area - original_area) / original_area * 100
  area_increase_percentage = x + y + (x * y / 100) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3963_396325


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l3963_396359

/-- Given a parabola y = ax^2 + bx, prove that after reflecting about the y-axis
    and translating one parabola 4 units right and the other 4 units left,
    the sum of the resulting parabolas' equations is y = 2ax^2 - 8b. -/
theorem parabola_reflection_translation_sum (a b : ℝ) :
  let f (x : ℝ) := a * x^2 + b * (x - 4)
  let g (x : ℝ) := a * x^2 - b * (x + 4)
  ∀ x, (f + g) x = 2 * a * x^2 - 8 * b :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l3963_396359


namespace NUMINAMATH_CALUDE_chloe_david_distance_difference_l3963_396323

-- Define the speeds and time
def chloe_speed : ℝ := 18
def david_speed : ℝ := 15
def bike_time : ℝ := 5

-- Define the theorem
theorem chloe_david_distance_difference :
  chloe_speed * bike_time - david_speed * bike_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_chloe_david_distance_difference_l3963_396323


namespace NUMINAMATH_CALUDE_product_of_sines_l3963_396382

theorem product_of_sines : 
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) * 
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_l3963_396382


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l3963_396335

theorem smallest_advantageous_discount : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (1 - m / 100 : ℝ) ≥ (1 - 0.15)^2 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.10)^3 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.25) * (1 - 0.05)) ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.10)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.25) * (1 - 0.05) ∧
  n = 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l3963_396335


namespace NUMINAMATH_CALUDE_courier_cost_formula_l3963_396324

def courier_cost (P : ℕ+) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem courier_cost_formula (P : ℕ+) :
  courier_cost P = if P ≤ 2 then 15 else 15 + 5 * (P - 2) :=
by sorry

end NUMINAMATH_CALUDE_courier_cost_formula_l3963_396324


namespace NUMINAMATH_CALUDE_exists_player_in_win_range_l3963_396372

/-- Represents a chess tournament with 2n+1 players -/
structure ChessTournament (n : ℕ) where
  /-- The number of games won by lower-rated players -/
  k : ℕ
  /-- Player ratings, assumed to be unique -/
  ratings : Fin (2*n+1) → ℕ
  ratings_unique : ∀ i j, i ≠ j → ratings i ≠ ratings j

/-- The number of wins for each player -/
def wins (t : ChessTournament n) : Fin (2*n+1) → ℕ :=
  sorry

theorem exists_player_in_win_range (n : ℕ) (t : ChessTournament n) :
  ∃ p : Fin (2*n+1), 
    (n : ℝ) - Real.sqrt (2 * t.k) ≤ wins t p ∧ 
    wins t p ≤ (n : ℝ) + Real.sqrt (2 * t.k) :=
  sorry

end NUMINAMATH_CALUDE_exists_player_in_win_range_l3963_396372


namespace NUMINAMATH_CALUDE_circle_and_line_tangency_l3963_396319

-- Define the line l
def line (x y a : ℝ) : Prop := Real.sqrt 3 * x - y - a = 0

-- Define the circle C in polar form
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the circle C in Cartesian form
def circle_cartesian (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_and_line_tangency :
  -- Part I: Equivalence of polar and Cartesian forms of circle C
  (∀ x y ρ θ : ℝ, circle_polar ρ θ ↔ circle_cartesian x y) ∧
  -- Part II: Tangency condition
  (∀ a : ℝ, (∃ x y : ℝ, line x y a ∧ circle_cartesian x y ∧
    (∀ x' y' : ℝ, line x' y' a ∧ circle_cartesian x' y' → x = x' ∧ y = y'))
    ↔ (a = -3 ∨ a = 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_tangency_l3963_396319


namespace NUMINAMATH_CALUDE_monotonicity_condition_max_value_condition_l3963_396309

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^3 + 9 * x

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ m ≥ 3 :=
sorry

-- Part 2: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x ∈ Set.Icc 1 2, f m x ≤ 4) ∧
           (∃ x ∈ Set.Icc 1 2, f m x = 4) ∧
           m = -2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_max_value_condition_l3963_396309


namespace NUMINAMATH_CALUDE_expression_evaluation_l3963_396376

theorem expression_evaluation : 2 * (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 94 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3963_396376


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l3963_396368

theorem product_remainder_mod_17 : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l3963_396368


namespace NUMINAMATH_CALUDE_wheel_circumference_proof_l3963_396321

/-- The circumference of the front wheel -/
def front_wheel_circumference : ℝ := 24

/-- The circumference of the rear wheel -/
def rear_wheel_circumference : ℝ := 18

/-- The distance traveled -/
def distance : ℝ := 360

theorem wheel_circumference_proof :
  (distance / front_wheel_circumference = distance / rear_wheel_circumference + 4) ∧
  (distance / (front_wheel_circumference - 3) = distance / (rear_wheel_circumference - 3) + 6) →
  (front_wheel_circumference = 24 ∧ rear_wheel_circumference = 18) :=
by sorry

end NUMINAMATH_CALUDE_wheel_circumference_proof_l3963_396321


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l3963_396373

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (x : ℕ) : ℕ :=
  let r := x % 10
  if r < 5 then x - r else x + (10 - r)

def kate_sum (n : ℕ) : ℕ :=
  List.range n |> List.map (λ x => round_to_nearest_ten (x + 1)) |> List.sum

theorem sum_difference_theorem :
  jo_sum 60 - kate_sum 60 = 1530 := by sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l3963_396373


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3963_396341

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, false, true]                     -- 1011₂
  let product := [true, true, true, true, false, false, true, false, false, false, true]  -- 10001001111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

#eval binary_to_nat [true, false, true, true, false, true, true]  -- Should output 109
#eval binary_to_nat [true, true, false, true]  -- Should output 11
#eval binary_to_nat [true, true, true, true, false, false, true, false, false, false, true]  -- Should output 1103

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3963_396341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3963_396343

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- definition of arithmetic sequence
  : a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3963_396343


namespace NUMINAMATH_CALUDE_ruble_bill_combination_l3963_396316

theorem ruble_bill_combination : ∃ x y z : ℕ, x + y + z = 11 ∧ x + 3 * y + 5 * z = 25 := by
  sorry

end NUMINAMATH_CALUDE_ruble_bill_combination_l3963_396316


namespace NUMINAMATH_CALUDE_investment_calculation_l3963_396385

/-- Given a total investment split between a savings account and mutual funds,
    where the investment in mutual funds is 6 times the investment in the savings account,
    calculate the total investment in mutual funds. -/
theorem investment_calculation (total : ℝ) (savings : ℝ) (mutual_funds : ℝ)
    (h1 : total = 320000)
    (h2 : mutual_funds = 6 * savings)
    (h3 : total = savings + mutual_funds) :
  mutual_funds = 274285.74 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l3963_396385


namespace NUMINAMATH_CALUDE_recipe_calculation_l3963_396329

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the required amount of an ingredient based on the ratio and the amount of sugar used -/
def calculateAmount (ratio : RecipeRatio) (sugarAmount : ℚ) (partRatio : ℚ) : ℚ :=
  (sugarAmount / ratio.sugar) * partRatio

/-- Proves that given a recipe with a butter:flour:sugar ratio of 1:6:4 and using 10 cups of sugar,
    the required amounts of butter and flour are 2.5 cups and 15 cups, respectively -/
theorem recipe_calculation (ratio : RecipeRatio) (sugarAmount : ℚ) :
  ratio.butter = 1 → ratio.flour = 6 → ratio.sugar = 4 → sugarAmount = 10 →
  calculateAmount ratio sugarAmount ratio.butter = 5/2 ∧
  calculateAmount ratio sugarAmount ratio.flour = 15 :=
by sorry

end NUMINAMATH_CALUDE_recipe_calculation_l3963_396329


namespace NUMINAMATH_CALUDE_smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l3963_396331

-- Define the original angle
def original_angle : Int := -2010

-- Define a function to find the smallest positive equivalent angle
def smallest_positive_equivalent (angle : Int) : Int :=
  angle % 360

-- Define a function to find the largest negative equivalent angle
def largest_negative_equivalent (angle : Int) : Int :=
  (angle % 360) - 360

-- Define a function to find equivalent angles within a range
def equivalent_angles_in_range (angle : Int) (lower : Int) (upper : Int) : List Int :=
  let base_angle := angle % 360
  List.filter (fun x => lower ≤ x ∧ x < upper)
    [base_angle - 720, base_angle - 360, base_angle, base_angle + 360]

-- Theorem statements
theorem smallest_positive_angle_correct :
  smallest_positive_equivalent original_angle = 150 := by sorry

theorem largest_negative_angle_correct :
  largest_negative_equivalent original_angle = -210 := by sorry

theorem equivalent_angles_in_range_correct :
  equivalent_angles_in_range original_angle (-720) 720 = [-570, -210, 150, 510] := by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l3963_396331


namespace NUMINAMATH_CALUDE_abs_g_zero_equals_70_l3963_396328

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧
  (|g 1| = 10) ∧ (|g 3| = 10) ∧ (|g 4| = 10) ∧
  (|g 6| = 10) ∧ (|g 8| = 10) ∧ (|g 9| = 10)

/-- Theorem: If g satisfies the condition, then |g(0)| = 70 -/
theorem abs_g_zero_equals_70 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 0| = 70 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_zero_equals_70_l3963_396328


namespace NUMINAMATH_CALUDE_product_evaluation_l3963_396361

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3963_396361


namespace NUMINAMATH_CALUDE_inequality_proof_l3963_396383

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a / b + b / c + c / a + b / a + a / c + c / b + 6 ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3963_396383


namespace NUMINAMATH_CALUDE_laura_charge_account_l3963_396327

/-- Represents the simple interest calculation for a charge account -/
def simple_interest_charge_account (principal : ℝ) (interest_rate : ℝ) (time : ℝ) (total_owed : ℝ) : Prop :=
  total_owed = principal + (principal * interest_rate * time)

theorem laura_charge_account :
  ∀ (principal : ℝ),
    simple_interest_charge_account principal 0.05 1 36.75 →
    principal = 35 := by
  sorry

end NUMINAMATH_CALUDE_laura_charge_account_l3963_396327


namespace NUMINAMATH_CALUDE_probability_closer_to_point1_l3963_396349

/-- The rectangular region from which point P is selected -/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The area of the rectangular region -/
def RectangleArea : ℝ := 6

/-- The point (1,1) -/
def Point1 : ℝ × ℝ := (1, 1)

/-- The point (4,1) -/
def Point2 : ℝ × ℝ := (4, 1)

/-- The region where points are closer to (1,1) than to (4,1) -/
def CloserRegion : Set (ℝ × ℝ) :=
  {p ∈ Rectangle | dist p Point1 < dist p Point2}

/-- The area of the region closer to (1,1) -/
def CloserRegionArea : ℝ := 5

/-- The probability of a randomly selected point being closer to (1,1) than to (4,1) -/
theorem probability_closer_to_point1 :
  CloserRegionArea / RectangleArea = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_closer_to_point1_l3963_396349


namespace NUMINAMATH_CALUDE_ian_money_left_l3963_396379

/-- Calculates the amount of money Ian has left after spending half of his earnings from online surveys. -/
def money_left (hourly_rate : ℝ) (hours_worked : ℝ) (spending_ratio : ℝ) : ℝ :=
  let total_earnings := hourly_rate * hours_worked
  let amount_spent := total_earnings * spending_ratio
  total_earnings - amount_spent

/-- Proves that Ian has $72 left after working 8 hours at $18 per hour and spending half of his earnings. -/
theorem ian_money_left :
  money_left 18 8 0.5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l3963_396379


namespace NUMINAMATH_CALUDE_even_decreasing_compare_l3963_396366

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a decreasing function on negative reals
def DecreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

-- Theorem statement
theorem even_decreasing_compare (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_even : EvenFunction f) 
  (h_decreasing : DecreasingOnNegative f) 
  (h_abs : |x₁| < |x₂|) : 
  f x₁ - f x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_compare_l3963_396366


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3963_396330

theorem inequality_system_solutions (m : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   3 - 2 * (x : ℝ) ≥ 0 ∧ (x : ℝ) ≥ m ∧
   3 - 2 * (y : ℝ) ≥ 0 ∧ (y : ℝ) ≥ m ∧
   (∀ z : ℤ, z ≠ x ∧ z ≠ y → ¬(3 - 2 * (z : ℝ) ≥ 0 ∧ (z : ℝ) ≥ m))) →
  -1 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3963_396330


namespace NUMINAMATH_CALUDE_vincent_laundry_week_l3963_396367

def loads_wednesday : ℕ := 6

def loads_thursday (w : ℕ) : ℕ := 2 * w

def loads_friday (t : ℕ) : ℕ := t / 2

def loads_saturday (w : ℕ) : ℕ := w / 3

def total_loads (w t f s : ℕ) : ℕ := w + t + f + s

theorem vincent_laundry_week :
  total_loads loads_wednesday 
              (loads_thursday loads_wednesday)
              (loads_friday (loads_thursday loads_wednesday))
              (loads_saturday loads_wednesday) = 26 := by
  sorry

end NUMINAMATH_CALUDE_vincent_laundry_week_l3963_396367


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3963_396339

theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) → m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3963_396339


namespace NUMINAMATH_CALUDE_leopard_arrangement_l3963_396375

theorem leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * Nat.factorial 2 * Nat.factorial (n - 3) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_l3963_396375


namespace NUMINAMATH_CALUDE_least_value_quadratic_l3963_396371

theorem least_value_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 6) → y ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l3963_396371


namespace NUMINAMATH_CALUDE_solar_panel_height_P_l3963_396315

/-- Regular hexagon with side length 10 and pillars at vertices -/
structure SolarPanelSupport where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at L, M, and N
  height_L : ℝ
  height_M : ℝ
  height_N : ℝ

/-- The height of the pillar at P in the solar panel support system -/
def height_P (s : SolarPanelSupport) : ℝ := sorry

/-- Theorem stating the height of pillar P given specific conditions -/
theorem solar_panel_height_P (s : SolarPanelSupport) 
  (h_side : s.side_length = 10)
  (h_L : s.height_L = 15)
  (h_M : s.height_M = 12)
  (h_N : s.height_N = 13) : 
  height_P s = 22 := by sorry

end NUMINAMATH_CALUDE_solar_panel_height_P_l3963_396315


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3963_396305

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := 2 * 19 * factorial 19 * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 :
  ⌊M / 100⌋ = 499 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3963_396305


namespace NUMINAMATH_CALUDE_album_collection_problem_l3963_396388

/-- The number of albums in either Andrew's or John's collection, but not both -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ) : ℕ :=
  (andrew_total - shared) + john_exclusive

theorem album_collection_problem (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ)
  (h1 : shared = 12)
  (h2 : andrew_total = 20)
  (h3 : john_exclusive = 8) :
  exclusive_albums shared andrew_total john_exclusive = 16 := by
  sorry

end NUMINAMATH_CALUDE_album_collection_problem_l3963_396388


namespace NUMINAMATH_CALUDE_difference_of_squares_l3963_396380

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3963_396380


namespace NUMINAMATH_CALUDE_probability_two_red_one_green_l3963_396377

def red_shoes : ℕ := 6
def green_shoes : ℕ := 8
def blue_shoes : ℕ := 5
def yellow_shoes : ℕ := 3

def total_shoes : ℕ := red_shoes + green_shoes + blue_shoes + yellow_shoes

def draw_count : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_shoes 2 * Nat.choose green_shoes 1) / Nat.choose total_shoes draw_count = 6 / 77 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_one_green_l3963_396377


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_abs_l3963_396399

theorem quadratic_roots_sum_abs (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x - 6 = 0 ∧ y^2 + p*y - 6 = 0 ∧ |x| + |y| = 5) → 
  (p = 1 ∨ p = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_abs_l3963_396399


namespace NUMINAMATH_CALUDE_program_result_l3963_396336

/-- The smallest positive integer n for which n² + 4n ≥ 10000 -/
def smallest_n : ℕ := 99

/-- The function that computes x given n -/
def x (n : ℕ) : ℕ := 3 + 2 * n

/-- The function that computes S given n -/
def S (n : ℕ) : ℕ := n^2 + 4*n

theorem program_result :
  (∀ m : ℕ, m < smallest_n → S m < 10000) ∧
  S smallest_n ≥ 10000 ∧
  x smallest_n = 201 := by sorry

end NUMINAMATH_CALUDE_program_result_l3963_396336


namespace NUMINAMATH_CALUDE_solution_difference_l3963_396356

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem solution_difference (x y : ℝ) :
  (floor x : ℝ) + frac y = 3.7 →
  frac x + (floor y : ℝ) = 8.2 →
  |x - y| = 5.5 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3963_396356


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3963_396344

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 3/5 of 15 bananas are worth as much as 9 oranges. -/
theorem banana_orange_equivalence (banana_value : ℚ) :
  (3/4 : ℚ) * 12 * banana_value = 9 →
  (3/5 : ℚ) * 15 * banana_value = 9 :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3963_396344


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3963_396338

/-- A positive geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q ≠ 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  a 3 = 3 ∧
  ∀ n ≥ 2, a (n + 1) = 2 * a n + 3 * a (n - 1)

/-- The sum of the first 5 terms of the sequence -/
def S5 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3 + a 4 + a 5

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : S5 a = 121 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3963_396338


namespace NUMINAMATH_CALUDE_max_eel_coverage_l3963_396301

/-- An eel is a polyomino formed by a path of unit squares which makes two turns in opposite directions -/
def Eel : Type := Unit

/-- A configuration of non-overlapping eels on a grid -/
def EelConfiguration (n : ℕ) : Type := Unit

/-- The area covered by a configuration of eels -/
def coveredArea (n : ℕ) (config : EelConfiguration n) : ℕ := sorry

theorem max_eel_coverage :
  ∃ (config : EelConfiguration 1000),
    coveredArea 1000 config = 999998 ∧
    ∀ (other_config : EelConfiguration 1000),
      coveredArea 1000 other_config ≤ 999998 := by sorry

end NUMINAMATH_CALUDE_max_eel_coverage_l3963_396301


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3963_396397

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3963_396397


namespace NUMINAMATH_CALUDE_min_value_expression_l3963_396394

theorem min_value_expression (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (min : ℝ), min = 36 ∧ ∀ (x' y' z' : ℝ), x' - 2*y' + 2*z' = 5 → 
    (x' + 5)^2 + (y' - 1)^2 + (z' + 3)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3963_396394


namespace NUMINAMATH_CALUDE_students_without_A_l3963_396307

theorem students_without_A (total : ℕ) (chem : ℕ) (phys : ℕ) (both : ℕ) : 
  total = 40 → chem = 10 → phys = 18 → both = 6 →
  total - (chem + phys - both) = 18 := by sorry

end NUMINAMATH_CALUDE_students_without_A_l3963_396307


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l3963_396320

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- Theorem for part 1
theorem intersection_complement_when_a_three :
  A ∩ (Set.univ \ B 3) = Set.Icc 3 4 := by sorry

-- Theorem for part 2
theorem a_greater_than_four_when_A_subset_B (a : ℝ) :
  A ⊆ B a → a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_three_a_greater_than_four_when_A_subset_B_l3963_396320


namespace NUMINAMATH_CALUDE_decreasing_interval_of_even_f_l3963_396364

def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + (k - 1) * x + 3

theorem decreasing_interval_of_even_f (k : ℝ) :
  (∀ x, f k x = f k (-x)) →
  ∀ x > 0, ∀ y > x, f k y < f k x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_even_f_l3963_396364


namespace NUMINAMATH_CALUDE_comic_books_sale_proof_l3963_396314

/-- The number of comic books sold by Scott and Sam -/
def comic_books_sold (initial_total remaining : ℕ) : ℕ :=
  initial_total - remaining

theorem comic_books_sale_proof :
  comic_books_sold 90 25 = 65 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_sale_proof_l3963_396314


namespace NUMINAMATH_CALUDE_poem_line_increase_l3963_396333

theorem poem_line_increase (initial_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) (months : ℕ) : 
  initial_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  initial_lines + months * lines_per_month = target_lines →
  months = 22 := by
sorry

end NUMINAMATH_CALUDE_poem_line_increase_l3963_396333


namespace NUMINAMATH_CALUDE_sphere_dimensions_l3963_396363

-- Define the hole dimensions
def hole_diameter : ℝ := 12
def hole_depth : ℝ := 2

-- Define the sphere
def sphere_radius : ℝ := 10

-- Theorem statement
theorem sphere_dimensions (r : ℝ) (h : r = sphere_radius) :
  -- The radius satisfies the Pythagorean theorem for the right triangle formed
  (r - hole_depth) ^ 2 + (hole_diameter / 2) ^ 2 = r ^ 2 ∧
  -- The surface area of the sphere is 400π
  4 * Real.pi * r ^ 2 = 400 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_dimensions_l3963_396363


namespace NUMINAMATH_CALUDE_solve_system_and_find_perimeter_l3963_396365

/-- Given a system of equations, prove the values of a and b, and the perimeter of an isosceles triangle with these side lengths. -/
theorem solve_system_and_find_perimeter :
  ∃ (a b : ℝ),
    (4 * a - 3 * b = 22) ∧
    (2 * a + b = 16) ∧
    (a = 7) ∧
    (b = 2) ∧
    (2 * max a b + min a b = 16) := by
  sorry


end NUMINAMATH_CALUDE_solve_system_and_find_perimeter_l3963_396365


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3963_396317

theorem complex_arithmetic_evaluation : 2 - 2 * (2 - 2 * (2 - 2 * (4 - 2))) = -10 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3963_396317


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3963_396391

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - a 8 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3963_396391


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l3963_396386

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l3963_396386


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l3963_396384

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens : ℕ) : ℕ :=
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  total_birds 200 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l3963_396384


namespace NUMINAMATH_CALUDE_second_candidate_votes_l3963_396369

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) :
  total_votes = 1200 →
  first_candidate_percentage = 60 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 480 :=
by sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l3963_396369


namespace NUMINAMATH_CALUDE_odd_decreasing_function_properties_l3963_396303

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x < y → f x > f y)

-- Define the theorem
theorem odd_decreasing_function_properties
  (a b : ℝ)
  (h_sum_neg : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 := by
sorry


end NUMINAMATH_CALUDE_odd_decreasing_function_properties_l3963_396303


namespace NUMINAMATH_CALUDE_probability_of_strong_l3963_396350

def word_train : Finset Char := {'T', 'R', 'A', 'I', 'N'}
def word_shield : Finset Char := {'S', 'H', 'I', 'E', 'L', 'D'}
def word_grow : Finset Char := {'G', 'R', 'O', 'W'}
def word_strong : Finset Char := {'S', 'T', 'R', 'O', 'N', 'G'}

def prob_train : ℚ := 1 / (word_train.card.choose 3)
def prob_shield : ℚ := 3 / (word_shield.card.choose 4)
def prob_grow : ℚ := 1 / (word_grow.card.choose 2)

theorem probability_of_strong :
  prob_train * prob_shield * prob_grow = 1 / 300 :=
sorry

end NUMINAMATH_CALUDE_probability_of_strong_l3963_396350


namespace NUMINAMATH_CALUDE_network_coloring_l3963_396345

/-- A node in the network --/
structure Node where
  lines : Finset (Fin 10)

/-- A network of lines on a plane --/
structure Network where
  nodes : Finset Node
  adjacent : Node → Node → Prop

/-- A coloring of the network --/
def Coloring (n : Network) := Node → Fin 15

/-- A valid coloring of the network --/
def ValidColoring (n : Network) (c : Coloring n) : Prop :=
  ∀ (node1 node2 : Node), n.adjacent node1 node2 → c node1 ≠ c node2

/-- The main theorem: any network can be colored with at most 15 colors --/
theorem network_coloring (n : Network) : ∃ (c : Coloring n), ValidColoring n c := by
  sorry

end NUMINAMATH_CALUDE_network_coloring_l3963_396345


namespace NUMINAMATH_CALUDE_swim_time_ratio_l3963_396398

/-- Proves that the ratio of time taken to swim upstream to time taken to swim downstream is 2:1 -/
theorem swim_time_ratio (swim_speed : ℝ) (stream_speed : ℝ) 
  (h1 : swim_speed = 1.5) (h2 : stream_speed = 0.5) : 
  (swim_speed - stream_speed) / (swim_speed + stream_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l3963_396398


namespace NUMINAMATH_CALUDE_square_difference_l3963_396392

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3963_396392


namespace NUMINAMATH_CALUDE_log_equation_proof_l3963_396358

theorem log_equation_proof : -2 * Real.log 10 / Real.log 5 - Real.log 0.25 / Real.log 5 + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l3963_396358


namespace NUMINAMATH_CALUDE_joan_payment_amount_l3963_396332

-- Define the costs and change as constants
def cat_toy_cost : ℚ := 877 / 100
def cage_cost : ℚ := 1097 / 100
def change_received : ℚ := 26 / 100

-- Define the theorem
theorem joan_payment_amount :
  cat_toy_cost + cage_cost + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_amount_l3963_396332


namespace NUMINAMATH_CALUDE_share_distribution_l3963_396354

theorem share_distribution (a b c d : ℝ) : 
  a + b + c + d = 1200 →
  a = (3/5) * (b + c + d) →
  b = (2/3) * (a + c + d) →
  c = (4/7) * (a + b + d) →
  a = 247.5 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l3963_396354


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3963_396310

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3963_396310


namespace NUMINAMATH_CALUDE_bathroom_tiles_count_l3963_396355

-- Define the bathroom dimensions in feet
def bathroom_length : ℝ := 10
def bathroom_width : ℝ := 6

-- Define the tile side length in inches
def tile_side : ℝ := 6

-- Define the conversion factor from feet to inches
def inches_per_foot : ℝ := 12

theorem bathroom_tiles_count :
  (bathroom_length * inches_per_foot) * (bathroom_width * inches_per_foot) / (tile_side * tile_side) = 240 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_tiles_count_l3963_396355


namespace NUMINAMATH_CALUDE_divides_power_plus_one_l3963_396389

theorem divides_power_plus_one (n : ℕ) : (3 ^ (n + 1)) ∣ (2 ^ (3 ^ n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_power_plus_one_l3963_396389


namespace NUMINAMATH_CALUDE_equal_milk_water_ratio_l3963_396357

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The ratio of two quantities -/
def ratio (a b : ℚ) : ℚ := a / b

/-- Mixture p with milk to water ratio 5:4 -/
def mixture_p : Mixture := { milk := 5, water := 4 }

/-- Mixture q with milk to water ratio 2:7 -/
def mixture_q : Mixture := { milk := 2, water := 7 }

/-- Combine two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r : ℚ) : Mixture :=
  { milk := m1.milk * r + m2.milk,
    water := m1.water * r + m2.water }

/-- Theorem stating that mixing p and q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q (5/1)
  ratio result.milk result.water = 1 := by sorry

end NUMINAMATH_CALUDE_equal_milk_water_ratio_l3963_396357


namespace NUMINAMATH_CALUDE_problem_statement_l3963_396334

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) (α : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasPeriod f 5)
    (h_f_neg_three : f (-3) = 1)
    (h_tan_α : Real.tan α = 2) :
    f (20 * Real.sin α * Real.cos α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3963_396334


namespace NUMINAMATH_CALUDE_add_negative_two_and_two_equals_zero_l3963_396308

theorem add_negative_two_and_two_equals_zero : (-2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_two_and_two_equals_zero_l3963_396308


namespace NUMINAMATH_CALUDE_exists_x_sqrt_x_squared_neq_x_l3963_396390

theorem exists_x_sqrt_x_squared_neq_x : ∃ x : ℝ, Real.sqrt (x^2) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_exists_x_sqrt_x_squared_neq_x_l3963_396390


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3963_396302

theorem power_fraction_simplification : (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3963_396302


namespace NUMINAMATH_CALUDE_division_remainder_l3963_396326

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 760 → divisor = 36 → quotient = 21 → 
  dividend = divisor * quotient + remainder → remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3963_396326


namespace NUMINAMATH_CALUDE_long_sleeve_shirts_to_wash_l3963_396360

theorem long_sleeve_shirts_to_wash :
  ∀ (total_shirts short_sleeve_shirts long_sleeve_shirts shirts_washed shirts_not_washed : ℕ),
    total_shirts = short_sleeve_shirts + long_sleeve_shirts →
    shirts_washed = 29 →
    shirts_not_washed = 1 →
    short_sleeve_shirts = 9 →
    long_sleeve_shirts = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_long_sleeve_shirts_to_wash_l3963_396360


namespace NUMINAMATH_CALUDE_student_distribution_proof_l3963_396337

def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

theorem student_distribution_proof : 
  distribute_students 24 3 = 475 := by sorry

end NUMINAMATH_CALUDE_student_distribution_proof_l3963_396337


namespace NUMINAMATH_CALUDE_f_sum_equals_half_l3963_396340

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f (x - 2)

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > -2 ∧ x < 0 → f x = -2^x

theorem f_sum_equals_half (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_4 f)
  (h_condition : f_condition f) :
  f 1 + f 4 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_f_sum_equals_half_l3963_396340


namespace NUMINAMATH_CALUDE_tomato_weight_l3963_396347

/-- Calculates the weight of a tomato based on grocery shopping information. -/
theorem tomato_weight (meat_price meat_weight buns_price lettuce_price pickle_price pickle_discount tomato_price_per_pound paid change : ℝ) :
  meat_price = 3.5 →
  meat_weight = 2 →
  buns_price = 1.5 →
  lettuce_price = 1 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  tomato_price_per_pound = 2 →
  paid = 20 →
  change = 6 →
  (paid - change - (meat_price * meat_weight + buns_price + lettuce_price + (pickle_price - pickle_discount))) / tomato_price_per_pound = 1.5 := by
sorry

end NUMINAMATH_CALUDE_tomato_weight_l3963_396347
