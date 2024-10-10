import Mathlib

namespace possible_set_A_l3715_371567

-- Define the set B
def B : Set ℝ := {x | x ≥ 0}

-- Define the theorem
theorem possible_set_A (A : Set ℝ) (h1 : A ∩ B = A) : 
  ∃ A', A' = {1, 2} ∧ A' ∩ B = A' :=
sorry

end possible_set_A_l3715_371567


namespace arccos_minus_one_equals_pi_l3715_371595

theorem arccos_minus_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end arccos_minus_one_equals_pi_l3715_371595


namespace tim_took_25_rulers_l3715_371518

/-- The number of rulers Tim took from the drawer -/
def rulers_taken (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Tim took 25 rulers from the drawer -/
theorem tim_took_25_rulers :
  let initial_rulers : ℕ := 46
  let remaining_rulers : ℕ := 21
  rulers_taken initial_rulers remaining_rulers = 25 := by
  sorry

end tim_took_25_rulers_l3715_371518


namespace card_shuffle_bound_l3715_371554

theorem card_shuffle_bound (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end card_shuffle_bound_l3715_371554


namespace largest_initial_number_l3715_371591

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end largest_initial_number_l3715_371591


namespace third_month_sale_l3715_371558

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sales : List ℕ := [6400, 7000, 7200, 6500, 5100]

theorem third_month_sale :
  (num_months * average_sale - sales.sum) = 6800 :=
sorry

end third_month_sale_l3715_371558


namespace patrick_caught_eight_l3715_371568

/-- The number of fish caught by each person -/
structure FishCaught where
  patrick : ℕ
  angus : ℕ
  ollie : ℕ

/-- The conditions of the fishing problem -/
def fishing_conditions (fc : FishCaught) : Prop :=
  fc.angus = fc.patrick + 4 ∧
  fc.ollie = fc.angus - 7 ∧
  fc.ollie = 5

/-- Theorem: Given the fishing conditions, Patrick caught 8 fish -/
theorem patrick_caught_eight (fc : FishCaught) 
  (h : fishing_conditions fc) : fc.patrick = 8 := by
  sorry

end patrick_caught_eight_l3715_371568


namespace evaluate_expression_l3715_371587

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4 * 7) = 10 := by
  sorry

end evaluate_expression_l3715_371587


namespace cloth_sale_proof_l3715_371509

/-- Given a trader selling cloth with a profit of 55 per meter and a total profit of 2200,
    prove that the number of meters sold is 40. -/
theorem cloth_sale_proof (profit_per_meter : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_meter = 55) (h2 : total_profit = 2200) : 
    total_profit / profit_per_meter = 40 := by
  sorry

end cloth_sale_proof_l3715_371509


namespace equation_solutions_l3715_371585

-- Define the equation
def equation (x y : ℝ) : Prop :=
  (36 / Real.sqrt (abs x)) + (9 / Real.sqrt (abs y)) = 
  42 - 9 * (if x < 0 then Complex.I * Real.sqrt (abs x) else Real.sqrt x) - 
  (if y < 0 then Complex.I * Real.sqrt (abs y) else Real.sqrt y)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(4, 9), (-4, 873 + 504 * Real.sqrt 3), (-4, 873 - 504 * Real.sqrt 3), 
   ((62 + 14 * Real.sqrt 13) / 9, -9), ((62 - 14 * Real.sqrt 13) / 9, -9)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions :=
sorry

end equation_solutions_l3715_371585


namespace alex_sandwiches_l3715_371512

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) : ℕ :=
  (num_meats.choose 2) * num_cheeses

/-- Theorem stating the number of sandwiches Alex can make -/
theorem alex_sandwiches :
  num_sandwiches 8 7 = 196 :=
by sorry

end alex_sandwiches_l3715_371512


namespace inscribed_sphere_volume_l3715_371536

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 10
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (500 / 3) * π := by
sorry


end inscribed_sphere_volume_l3715_371536


namespace pattern_1010_is_BCDA_l3715_371589

/-- Represents the four vertices of a square -/
inductive Vertex
| A
| B
| C
| D

/-- Represents a square configuration -/
def Square := List Vertex

/-- The initial square configuration -/
def initial_square : Square := [Vertex.A, Vertex.B, Vertex.C, Vertex.D]

/-- Performs a 90-degree counterclockwise rotation on a square -/
def rotate (s : Square) : Square := 
  match s with
  | [a, b, c, d] => [b, c, d, a]
  | _ => s

/-- Reflects a square over its horizontal line of symmetry -/
def reflect (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, c, b, a]
  | _ => s

/-- Applies the alternating pattern of rotation and reflection n times -/
def apply_pattern (s : Square) (n : Nat) : Square :=
  match n with
  | 0 => s
  | n + 1 => if n % 2 == 0 then rotate (apply_pattern s n) else reflect (apply_pattern s n)

theorem pattern_1010_is_BCDA : 
  apply_pattern initial_square 1010 = [Vertex.B, Vertex.C, Vertex.D, Vertex.A] := by
  sorry

end pattern_1010_is_BCDA_l3715_371589


namespace min_weighings_required_l3715_371549

/-- Represents a 4x4 grid of coins -/
def CoinGrid := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if two positions are adjacent in the grid -/
def adjacent (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- A valid coin grid satisfying the problem conditions -/
def valid_coin_grid (g : CoinGrid) : Prop :=
  ∃ p q : Fin 4 × Fin 4,
    adjacent p q ∧
    g p.1 p.2 = 9 ∧ g q.1 q.2 = 9 ∧
    ∀ r : Fin 4 × Fin 4, (r ≠ p ∧ r ≠ q) → g r.1 r.2 = 10

/-- A weighing selects a subset of coins and returns their total weight -/
def Weighing := Set (Fin 4 × Fin 4) → ℕ

/-- The theorem stating the minimum number of weighings required -/
theorem min_weighings_required (g : CoinGrid) (h : valid_coin_grid g) :
  ∃ (w₁ w₂ w₃ : Weighing),
    (∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
      (∀ S : Set (Fin 4 × Fin 4), w₁ S = w₁ S → w₂ S = w₂ S → w₃ S = w₃ S) →
      g₁ = g₂) ∧
    (∀ w₁' w₂' : Weighing,
      ¬∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
        (∀ S : Set (Fin 4 × Fin 4), w₁' S = w₁' S → w₂' S = w₂' S) →
        g₁ = g₂) :=
by
  sorry

end min_weighings_required_l3715_371549


namespace value_of_x_l3715_371593

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end value_of_x_l3715_371593


namespace last_two_nonzero_digits_80_factorial_l3715_371503

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- The number of factors of 10 in n! -/
def factorsOfTen (n : ℕ) : ℕ := sorry

theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 52 := by sorry

end last_two_nonzero_digits_80_factorial_l3715_371503


namespace al_sandwich_options_l3715_371507

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether roast beef is available. -/
def roast_beef_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a turkey/Swiss cheese combination. -/
def no_turkey_swiss : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a rye bread/roast beef combination. -/
def no_rye_roast_beef : Prop := True

/-- The number of different sandwiches Al could order. -/
def num_al_sandwiches : ℕ := num_breads * num_meats * num_cheeses - 5 - 6

theorem al_sandwich_options :
  num_breads = 5 →
  num_meats = 7 →
  num_cheeses = 6 →
  turkey_available →
  roast_beef_available →
  swiss_cheese_available →
  rye_bread_available →
  no_turkey_swiss →
  no_rye_roast_beef →
  num_al_sandwiches = 199 := by
  sorry

#eval num_al_sandwiches -- This should output 199

end al_sandwich_options_l3715_371507


namespace quadratic_inequality_solution_set_l3715_371541

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end quadratic_inequality_solution_set_l3715_371541


namespace complex_numbers_satisfying_conditions_l3715_371544

theorem complex_numbers_satisfying_conditions :
  ∀ z : ℂ,
    (∃ t : ℝ, z + 10 / z = t ∧ 1 < t ∧ t ≤ 6) ∧
    (∃ a b : ℤ, z = ↑a + ↑b * I) →
    z = 1 + 3 * I ∨ z = 1 - 3 * I ∨ z = 3 + I ∨ z = 3 - I :=
by sorry

end complex_numbers_satisfying_conditions_l3715_371544


namespace min_distance_curve_to_line_l3715_371525

/-- Given a > 0 and b = -1/2 * a^2 + 3 * ln(a), and a point Q(m, n) on the line y = 2x + 1/2,
    the minimum value of (a-m)^2 + (b-n)^2 is 9/5 -/
theorem min_distance_curve_to_line (a b m n : ℝ) (ha : a > 0) 
  (hb : b = -1/2 * a^2 + 3 * Real.log a) (hq : n = 2 * m + 1/2) :
  ∃ (min_val : ℝ), min_val = 9/5 ∧ 
  ∀ (x y : ℝ), (y = -1/2 * x^2 + 3 * Real.log x) → 
  (a - m)^2 + (b - n)^2 ≤ (x - m)^2 + (y - n)^2 :=
sorry

end min_distance_curve_to_line_l3715_371525


namespace decimal_sum_to_fraction_l3715_371580

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 1466 / 6250 := by
  sorry

end decimal_sum_to_fraction_l3715_371580


namespace g_range_g_range_achieves_bounds_l3715_371598

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 - 2*Real.pi * Real.arccos (x/3) + (Real.arccos (x/3))^2 + 
  (Real.pi^2/4) * (x^2 - 9*x + 27)

theorem g_range : 
  ∀ y ∈ Set.range g, -3*(Real.pi^2/4) ≤ y ∧ y ≤ 33*(Real.pi^2/4) :=
by sorry

theorem g_range_achieves_bounds : 
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
  g x₁ = -3*(Real.pi^2/4) ∧ g x₂ = 33*(Real.pi^2/4) :=
by sorry

end g_range_g_range_achieves_bounds_l3715_371598


namespace max_area_at_120_l3715_371506

/-- Represents a rectangular cow pasture -/
structure Pasture where
  fence_length : ℝ
  barn_length : ℝ

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn -/
def pasture_area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fence_length - 2 * x)

/-- Theorem stating that the maximum area occurs when the side parallel to the barn is 120 feet -/
theorem max_area_at_120 (p : Pasture) (h1 : p.fence_length = 240) (h2 : p.barn_length = 500) :
  ∃ (max_x : ℝ), (∀ (x : ℝ), pasture_area p x ≤ pasture_area p max_x) ∧ p.fence_length - 2 * max_x = 120 := by
  sorry


end max_area_at_120_l3715_371506


namespace dilution_correct_l3715_371523

/-- The amount of pure alcohol needed to dilute iodine tincture -/
def alcohol_amount : ℝ := 2275

/-- The initial amount of iodine tincture in grams -/
def initial_tincture : ℝ := 350

/-- The initial iodine content as a percentage -/
def initial_content : ℝ := 15

/-- The desired iodine content as a percentage -/
def desired_content : ℝ := 2

/-- Theorem stating that adding the calculated amount of alcohol results in the desired iodine content -/
theorem dilution_correct : 
  (initial_tincture * initial_content) / (initial_tincture + alcohol_amount) = desired_content := by
  sorry

end dilution_correct_l3715_371523


namespace parametric_represents_curve_l3715_371565

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the parametric equations
def parametric_x (t : ℝ) : ℝ := t
def parametric_y (t : ℝ) : ℝ := t^2

-- Theorem statement
theorem parametric_represents_curve :
  ∀ (t : ℝ), curve (parametric_x t) = parametric_y t :=
sorry

end parametric_represents_curve_l3715_371565


namespace opposite_face_of_one_is_three_l3715_371546

/-- Represents a face of a cube --/
inductive CubeFace
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents a net of a cube --/
structure CubeNet where
  faces : Finset CubeFace
  valid : faces.card = 6

/-- Represents a folded cube --/
structure FoldedCube where
  net : CubeNet
  topFace : CubeFace
  bottomFace : CubeFace
  oppositeFaces : CubeFace → CubeFace

/-- Theorem stating that in a cube formed by folding a net with faces numbered 1 to 6,
    where face 1 becomes the top face, the face opposite to face 1 is face 3 --/
theorem opposite_face_of_one_is_three (c : FoldedCube) 
    (h1 : c.topFace = CubeFace.One) :
  c.oppositeFaces CubeFace.One = CubeFace.Three :=
sorry

end opposite_face_of_one_is_three_l3715_371546


namespace banana_tree_problem_l3715_371563

/-- The number of bananas initially on the tree -/
def initial_bananas : ℕ := 1180

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left : ℕ := 500

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 170

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 3 * bananas_eaten

theorem banana_tree_problem :
  initial_bananas = bananas_left + bananas_eaten + bananas_in_basket :=
by sorry

end banana_tree_problem_l3715_371563


namespace uncles_gift_amount_l3715_371535

def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def shorts_cost : ℕ := 8
def money_left : ℕ := 14

theorem uncles_gift_amount : 
  jerseys_cost + basketball_cost + shorts_cost + money_left = 50 := by
  sorry

end uncles_gift_amount_l3715_371535


namespace janes_number_l3715_371560

theorem janes_number (x : ℝ) : 5 * (2 * x + 15) = 175 → x = 10 := by
  sorry

end janes_number_l3715_371560


namespace f_neg_one_eq_zero_l3715_371533

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

end f_neg_one_eq_zero_l3715_371533


namespace number_of_baskets_l3715_371584

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : 
  total_apples / apples_per_basket = 37 := by sorry

end number_of_baskets_l3715_371584


namespace linear_function_property_l3715_371521

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) 
  (hg : g 8 - g 4 = 16) : 
  g 16 - g 4 = 48 := by
sorry

end linear_function_property_l3715_371521


namespace monomial_is_algebraic_expression_l3715_371597

-- Define what an algebraic expression is
def AlgebraicExpression (α : Type*) := α → ℝ

-- Define what a monomial is
def Monomial (α : Type*) := AlgebraicExpression α

-- Theorem: Every monomial is an algebraic expression
theorem monomial_is_algebraic_expression {α : Type*} :
  ∀ (m : Monomial α), ∃ (a : AlgebraicExpression α), m = a :=
sorry

end monomial_is_algebraic_expression_l3715_371597


namespace inequality_always_true_l3715_371562

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  (a + c > b + d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a * c > b * d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a / c > b / d) :=
by sorry

end inequality_always_true_l3715_371562


namespace complex_square_imaginary_part_l3715_371504

theorem complex_square_imaginary_part : 
  ∃ (a b : ℝ), (1 + Complex.I)^2 = (a : ℂ) + (b : ℂ) * Complex.I → b = 2 := by
  sorry

end complex_square_imaginary_part_l3715_371504


namespace investment_earnings_l3715_371576

/-- Calculates the earnings from a stock investment --/
def calculate_earnings (investment : ℕ) (dividend_rate : ℕ) (market_price : ℕ) (face_value : ℕ) : ℕ :=
  let shares := investment / market_price
  let total_face_value := shares * face_value
  (dividend_rate * total_face_value) / 100

/-- Theorem stating that the given investment yields the expected earnings --/
theorem investment_earnings : 
  calculate_earnings 5760 1623 64 100 = 146070 := by
  sorry

end investment_earnings_l3715_371576


namespace octahedron_cube_volume_ratio_l3715_371555

/-- The ratio of the volume of a regular octahedron formed by joining the centers of adjoining faces
    of a cube to the volume of the cube, when the cube has a side length of 2 units. -/
theorem octahedron_cube_volume_ratio : 
  let cube_side : ℝ := 2
  let cube_volume : ℝ := cube_side ^ 3
  let octahedron_side : ℝ := Real.sqrt 2
  let octahedron_volume : ℝ := (octahedron_side ^ 3 * Real.sqrt 2) / 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry


end octahedron_cube_volume_ratio_l3715_371555


namespace direct_variation_problem_l3715_371582

/-- A function representing direct variation --/
def direct_variation (k : ℝ) (x : ℝ) : ℝ := k * x

theorem direct_variation_problem (k : ℝ) :
  (direct_variation k 2.5 = 10) →
  (direct_variation k (-5) = -20) := by
  sorry

#check direct_variation_problem

end direct_variation_problem_l3715_371582


namespace equation_solution_l3715_371556

theorem equation_solution : 
  ∃ y : ℝ, (2 / y + (3 / y) / (6 / y) = 1.2) ∧ y = 20 / 7 := by
  sorry

end equation_solution_l3715_371556


namespace novels_on_ends_l3715_371557

theorem novels_on_ends (total_books : ℕ) (novels : ℕ) (other_books : ℕ) 
  (h1 : total_books = 5)
  (h2 : novels = 2)
  (h3 : other_books = 3)
  (h4 : total_books = novels + other_books) :
  (other_books.factorial * novels.factorial) = 12 :=
by sorry

end novels_on_ends_l3715_371557


namespace curve_intersection_minimum_a_l3715_371516

theorem curve_intersection_minimum_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) →
  a ≥ Real.exp 2 / 4 :=
by sorry

end curve_intersection_minimum_a_l3715_371516


namespace fox_initial_money_l3715_371552

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 4

/-- The toll paid after each crossing -/
def toll : ℕ := 50

/-- The initial toll paid before the first crossing -/
def initial_toll : ℕ := 10

/-- The function that calculates Fox's money after each crossing -/
def money_after_crossing (initial_money : ℕ) (crossing : ℕ) : ℤ :=
  (2^crossing) * (initial_money - initial_toll) - 
  (2^crossing - 1) * toll - 
  initial_toll

/-- The theorem stating that Fox started with 56 coins -/
theorem fox_initial_money : 
  ∃ (initial_money : ℕ), 
    initial_money = 56 ∧ 
    money_after_crossing initial_money num_crossings = 0 :=
  sorry

end fox_initial_money_l3715_371552


namespace white_most_likely_probabilities_game_is_fair_l3715_371527

/-- Represents the colors of ping-pong balls in the box -/
inductive Color
  | White
  | Yellow
  | Red

/-- The total number of balls in the box -/
def totalBalls : ℕ := 6

/-- The number of balls of each color -/
def numBalls (c : Color) : ℕ :=
  match c with
  | Color.White => 3
  | Color.Yellow => 2
  | Color.Red => 1

/-- The probability of picking a ball of a given color -/
def prob (c : Color) : ℚ :=
  (numBalls c : ℚ) / totalBalls

/-- Theorem stating that white is the most likely color to be picked -/
theorem white_most_likely :
  ∀ c : Color, c ≠ Color.White → prob Color.White > prob c := by sorry

/-- Theorem stating the probabilities for each color -/
theorem probabilities :
  prob Color.White = 1/2 ∧ prob Color.Yellow = 1/3 ∧ prob Color.Red = 1/6 := by sorry

/-- Theorem stating that the game is fair -/
theorem game_is_fair :
  prob Color.White = 1 - prob Color.White := by sorry

end white_most_likely_probabilities_game_is_fair_l3715_371527


namespace max_y_over_x_l3715_371543

theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ M := by
  sorry

end max_y_over_x_l3715_371543


namespace heart_ratio_two_four_four_two_l3715_371532

def heart (n m : ℕ) : ℕ := n^2 * m^3

theorem heart_ratio_two_four_four_two :
  (heart 2 4) / (heart 4 2) = 2 := by sorry

end heart_ratio_two_four_four_two_l3715_371532


namespace stratified_sample_theorem_l3715_371548

/-- Calculates the number of employees to be drawn from a department in a stratified sampling method. -/
def stratified_sample_size (total_employees : ℕ) (sample_size : ℕ) (department_size : ℕ) : ℕ :=
  (department_size * sample_size) / total_employees

/-- Theorem stating that for a company with 240 employees and a sample size of 20,
    the number of employees to be drawn from a department with 60 employees is 5. -/
theorem stratified_sample_theorem :
  stratified_sample_size 240 20 60 = 5 := by
  sorry

end stratified_sample_theorem_l3715_371548


namespace complex_equation_solution_l3715_371531

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (w : ℂ) :
  w - 1 = (1 + w) * i → w = i := by
  sorry

end complex_equation_solution_l3715_371531


namespace bryden_receive_amount_l3715_371581

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The percentage the collector offers for state quarters -/
def collector_offer_percent : ℕ := 2500

/-- Calculate the amount Bryden will receive from the collector -/
def bryden_receive : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

/-- Theorem stating that Bryden will receive $43.75 from the collector -/
theorem bryden_receive_amount :
  bryden_receive = 43.75 := by sorry

end bryden_receive_amount_l3715_371581


namespace triangle_formation_l3715_371599

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three real numbers can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 1 1.5 3 ∧
  ¬ can_form_triangle 3 4 8 :=
sorry

end triangle_formation_l3715_371599


namespace min_attempts_correct_l3715_371511

/-- Represents the minimum number of attempts to make a lamp work given a set of batteries. -/
def min_attempts (total : ℕ) (good : ℕ) (bad : ℕ) : ℕ :=
  if total = 2 * good - 1 then good else good - 1

theorem min_attempts_correct (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) n = n + 1) ∧
  (min_attempts (2 * n) n n = n) :=
by sorry

#check min_attempts_correct

end min_attempts_correct_l3715_371511


namespace max_k_inequality_k_max_is_tight_l3715_371501

theorem max_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∀ k : ℝ, k ≤ 174960 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ k * a * b * c * d^3 :=
by sorry

theorem k_max_is_tight :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) = 174960 * a * b * c * d^3 :=
by sorry

end max_k_inequality_k_max_is_tight_l3715_371501


namespace adjacent_fractions_property_l3715_371515

-- Define the type for our rational numbers
def IrreducibleRational := {q : ℚ // q > 0 ∧ Irreducible q ∧ q.num * q.den < 1988}

-- Define the property of being adjacent in the sequence
def Adjacent (q1 q2 : IrreducibleRational) : Prop :=
  q1.val < q2.val ∧ ∀ q : IrreducibleRational, q.val ≤ q1.val ∨ q2.val ≤ q.val

-- State the theorem
theorem adjacent_fractions_property (q1 q2 : IrreducibleRational) 
  (h : Adjacent q1 q2) : 
  q1.val.den * q2.val.num - q1.val.num * q2.val.den = 1 :=
sorry

end adjacent_fractions_property_l3715_371515


namespace max_value_of_product_sum_l3715_371537

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : 0 ≤ w) (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z)
  (sum_condition : w + x + y + z = 200) :
  w * x + w * y + y * z + z * x ≤ 10000 :=
by sorry

end max_value_of_product_sum_l3715_371537


namespace sin_2alpha_value_l3715_371534

theorem sin_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (π / 4 - α))
  (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (2 * α) = -7/8 := by
sorry

end sin_2alpha_value_l3715_371534


namespace factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l3715_371596

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem: 101! · 102! is a perfect square -/
theorem factorial_101_102_is_perfect_square :
  is_perfect_square (factorial 101 * factorial 102) := by sorry

/-- Theorem: 100! · 101! is not a perfect square -/
theorem factorial_100_101_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 101) := by sorry

/-- Theorem: 100! · 102! is not a perfect square -/
theorem factorial_100_102_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 102) := by sorry

/-- Theorem: 101! · 103! is not a perfect square -/
theorem factorial_101_103_not_perfect_square :
  ¬ is_perfect_square (factorial 101 * factorial 103) := by sorry

/-- Theorem: 102! · 103! is not a perfect square -/
theorem factorial_102_103_not_perfect_square :
  ¬ is_perfect_square (factorial 102 * factorial 103) := by sorry

end factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l3715_371596


namespace a_10_value_l3715_371569

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 5 ∧ a 7 = -7 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In the given arithmetic sequence, a_10 = -16 -/
theorem a_10_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 10 = -16 := by
  sorry

end a_10_value_l3715_371569


namespace airport_passenger_ratio_l3715_371588

/-- Proves that the ratio of passengers using Miami Airport to those using Logan Airport is 4:1 -/
theorem airport_passenger_ratio :
  let total_passengers : ℝ := 38.3 * 1000000
  let kennedy_passengers : ℝ := total_passengers / 3
  let miami_passengers : ℝ := kennedy_passengers / 2
  let logan_passengers : ℝ := 1.5958333333333332 * 1000000
  miami_passengers / logan_passengers = 4 := by
  sorry

end airport_passenger_ratio_l3715_371588


namespace line_segment_intersection_l3715_371583

/-- Given a line ax + y + 2 = 0 and points P(-2, 1) and Q(3, 2), 
    if the line intersects with the line segment PQ, 
    then a ≤ -4/3 or a ≥ 3/2 -/
theorem line_segment_intersection (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0 ∧ 
    ((x = -2 ∧ y = 1) ∨ 
     (x = 3 ∧ y = 2) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
       x = -2 + 5*t ∧ 
       y = 1 + t))) → 
  (a ≤ -4/3 ∨ a ≥ 3/2) :=
by sorry

end line_segment_intersection_l3715_371583


namespace theater_lost_revenue_l3715_371564

/-- Calculates the lost revenue for a movie theater given its capacity, ticket price, and actual tickets sold. -/
theorem theater_lost_revenue (capacity : ℕ) (ticket_price : ℚ) (tickets_sold : ℕ) :
  capacity = 50 →
  ticket_price = 8 →
  tickets_sold = 24 →
  (capacity : ℚ) * ticket_price - (tickets_sold : ℚ) * ticket_price = 208 := by
  sorry

end theater_lost_revenue_l3715_371564


namespace team_a_finishes_faster_l3715_371529

/-- Proves that Team A finishes 3 hours faster than Team R given the specified conditions --/
theorem team_a_finishes_faster (course_distance : ℝ) (team_r_speed : ℝ) (speed_difference : ℝ) :
  course_distance = 300 →
  team_r_speed = 20 →
  speed_difference = 5 →
  let team_a_speed := team_r_speed + speed_difference
  let team_r_time := course_distance / team_r_speed
  let team_a_time := course_distance / team_a_speed
  team_r_time - team_a_time = 3 := by
  sorry

end team_a_finishes_faster_l3715_371529


namespace sin_cos_identity_l3715_371508

theorem sin_cos_identity :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + 
  Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 1 :=
by sorry

end sin_cos_identity_l3715_371508


namespace units_digit_of_expression_l3715_371573

-- Define a function to get the unit's place digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_expression : unitsDigit ((3^34 * 7^21) + 5^17) = 8 := by
  sorry

end units_digit_of_expression_l3715_371573


namespace smallest_n_congruence_l3715_371550

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(13 * m.val) % 8 = 567 % 8) ∧ 
  (13 * n.val) % 8 = 567 % 8 → 
  n = 3 := by sorry

end smallest_n_congruence_l3715_371550


namespace waiter_tables_l3715_371547

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) 
  (h1 : initial_customers = 22)
  (h2 : left_customers = 14)
  (h3 : people_per_table = 4) :
  (initial_customers - left_customers) / people_per_table = 2 := by
  sorry

end waiter_tables_l3715_371547


namespace percent_of_double_is_eighteen_l3715_371522

theorem percent_of_double_is_eighteen (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * (2 * y) = 18) : y = 30 := by
  sorry

end percent_of_double_is_eighteen_l3715_371522


namespace marble_average_l3715_371579

/-- Given the conditions about the average numbers of marbles of different colors,
    prove that the average number of all three colors is 30. -/
theorem marble_average (R Y B : ℕ) : 
  (R + Y : ℚ) / 2 = 26.5 →
  (B + Y : ℚ) / 2 = 34.5 →
  (R + B : ℚ) / 2 = 29 →
  (R + Y + B : ℚ) / 3 = 30 := by
  sorry

end marble_average_l3715_371579


namespace complex_equation_implies_ab_eight_l3715_371561

theorem complex_equation_implies_ab_eight (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * (3 + i) = 10 + 10 * i →
  a * b = 8 := by
  sorry

end complex_equation_implies_ab_eight_l3715_371561


namespace characterize_valid_k_l3715_371566

/-- A coloring of the complete graph on n vertices using k colors -/
def GraphColoring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Property: for any k vertices, all edges between them have different colors -/
def HasUniqueColors (n : ℕ) (k : ℕ) (coloring : GraphColoring n k) : Prop :=
  ∀ (vertices : Finset (Fin n)), vertices.card = k →
    (∀ (i j : Fin n), i ∈ vertices → j ∈ vertices → i ≠ j →
      ∀ (x y : Fin n), x ∈ vertices → y ∈ vertices → x ≠ y → (x, y) ≠ (i, j) →
        coloring i j ≠ coloring x y)

/-- The set of valid k values for a 10-vertex graph -/
def ValidK : Set ℕ := {k | k ≥ 5 ∧ k ≤ 10}

/-- Main theorem: characterization of valid k for a 10-vertex graph -/
theorem characterize_valid_k :
  ∀ k, k ∈ ValidK ↔ ∃ (coloring : GraphColoring 10 k), HasUniqueColors 10 k coloring :=
sorry

end characterize_valid_k_l3715_371566


namespace unique_line_existence_l3715_371578

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def line_passes_through (a b : ℚ) (x y : ℚ) : Prop :=
  x / a + y / b = 1

theorem unique_line_existence :
  ∃! (a b : ℚ), 
    (∃ n : ℕ, a = n ∧ is_prime n ∧ n < 10) ∧ 
    (∃ m : ℕ, b = m ∧ is_even m) ∧ 
    line_passes_through a b 5 4 :=
sorry

end unique_line_existence_l3715_371578


namespace parabola_focus_directrix_distance_l3715_371528

/-- For a parabola with equation x^2 = (1/2)y, the distance from its focus to its directrix is 1/4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = (1/2) * y → 
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ 
     focus_y = 1/8 ∧ 
     directrix_y = -1/8 ∧
     focus_y - directrix_y = 1/4) := by
  sorry

end parabola_focus_directrix_distance_l3715_371528


namespace remainder_sum_l3715_371574

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 58) 
  (hb : b % 90 = 84) : 
  (a + b) % 30 = 22 := by
sorry

end remainder_sum_l3715_371574


namespace barbed_wire_cost_l3715_371577

theorem barbed_wire_cost (field_area : ℝ) (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) : 
  field_area = 3136 ∧ 
  wire_cost_per_meter = 1.4 ∧ 
  gate_width = 1 ∧ 
  num_gates = 2 → 
  (Real.sqrt field_area * 4 - (gate_width * num_gates)) * wire_cost_per_meter = 310.8 := by
  sorry

end barbed_wire_cost_l3715_371577


namespace soda_price_l3715_371539

/-- The cost of a burger in cents -/
def burger_cost : ℚ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℚ := sorry

/-- Uri's purchase: 3 burgers and 1 soda for 360 cents -/
axiom uri_purchase : 3 * burger_cost + soda_cost = 360

/-- Gen's purchase: 1 burger and 3 sodas for 330 cents -/
axiom gen_purchase : burger_cost + 3 * soda_cost = 330

theorem soda_price : soda_cost = 78.75 := by sorry

end soda_price_l3715_371539


namespace shooting_game_cost_l3715_371575

theorem shooting_game_cost (jen_plays : ℕ) (russel_rides : ℕ) (carousel_cost : ℕ) (total_tickets : ℕ) :
  jen_plays = 2 →
  russel_rides = 3 →
  carousel_cost = 3 →
  total_tickets = 19 →
  ∃ (shooting_cost : ℕ), jen_plays * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ shooting_cost = 5 := by
  sorry

end shooting_game_cost_l3715_371575


namespace max_polygon_size_no_parallel_sides_l3715_371592

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ
  -- Assuming angle is in radians and normalized to [0, 2π)

/-- The number of points marked on the circle -/
def num_points : ℕ := 2012

/-- The set of all points on the circle -/
def circle_points : Finset CirclePoint :=
  sorry

/-- Predicate to check if two line segments are parallel -/
def are_parallel (p1 p2 p3 p4 : CirclePoint) : Prop :=
  sorry

/-- Predicate to check if a set of points forms a convex polygon -/
def is_convex_polygon (points : Finset CirclePoint) : Prop :=
  sorry

/-- The main theorem -/
theorem max_polygon_size_no_parallel_sides :
  ∃ (points : Finset CirclePoint),
    points.card = 1509 ∧
    is_convex_polygon points ∧
    (∀ (p1 p2 p3 p4 : CirclePoint),
      p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
      p1 ≠ p2 → p3 ≠ p4 → ¬(are_parallel p1 p2 p3 p4)) ∧
    (∀ (larger_set : Finset CirclePoint),
      larger_set.card > 1509 →
      is_convex_polygon larger_set →
      (∃ (q1 q2 q3 q4 : CirclePoint),
        q1 ∈ larger_set ∧ q2 ∈ larger_set ∧ q3 ∈ larger_set ∧ q4 ∈ larger_set ∧
        q1 ≠ q2 ∧ q3 ≠ q4 ∧ are_parallel q1 q2 q3 q4)) :=
by sorry


end max_polygon_size_no_parallel_sides_l3715_371592


namespace complement_of_hit_at_least_once_l3715_371571

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcomes of two shots -/
def TwoShots := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once in two shots -/
def HitAtLeastOnce (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Hit ∨ shots.2 = ShotOutcome.Hit

/-- The event of missing the target both times -/
def MissBothTimes (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Miss ∧ shots.2 = ShotOutcome.Miss

/-- Theorem stating that MissBothTimes is the complement of HitAtLeastOnce -/
theorem complement_of_hit_at_least_once :
  ∀ (shots : TwoShots), ¬(HitAtLeastOnce shots) ↔ MissBothTimes shots :=
sorry


end complement_of_hit_at_least_once_l3715_371571


namespace fruit_salad_cherries_l3715_371519

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 390 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  c = 119 := by
sorry

end fruit_salad_cherries_l3715_371519


namespace subset_condition_intersection_condition_l3715_371502

open Set Real

-- Define set A
def A : Set ℝ := {x : ℝ | |x + 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m → m ≤ -5 := by sorry

-- Theorem for part 2
theorem intersection_condition (m n : ℝ) : A ∩ B m = Ioo (-1) n → m = -1 ∧ n = 1 := by sorry

end subset_condition_intersection_condition_l3715_371502


namespace opposites_sum_to_zero_l3715_371553

theorem opposites_sum_to_zero (a b : ℝ) (h : a = -b) : a + b = 0 := by
  sorry

end opposites_sum_to_zero_l3715_371553


namespace min_value_of_product_l3715_371524

theorem min_value_of_product (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  (a - b) * (b - c) * (c - d) * (d - a) ≥ -1/8 := by
  sorry

end min_value_of_product_l3715_371524


namespace box_bottles_count_l3715_371538

-- Define the number of items in a dozen
def dozen : ℕ := 12

-- Define the number of water bottles
def water_bottles : ℕ := 2 * dozen

-- Define the number of additional apple bottles
def additional_apple_bottles : ℕ := dozen / 2

-- Define the total number of apple bottles
def apple_bottles : ℕ := water_bottles + additional_apple_bottles

-- Define the total number of bottles
def total_bottles : ℕ := water_bottles + apple_bottles

-- Theorem statement
theorem box_bottles_count : total_bottles = 54 := by
  sorry

end box_bottles_count_l3715_371538


namespace defective_probability_l3715_371586

/-- Represents a box of components -/
structure Box where
  total : ℕ
  defective : ℕ

/-- The probability of selecting a box -/
def boxProb : ℚ := 1 / 2

/-- The probability of selecting a defective component from a given box -/
def defectiveProb (box : Box) : ℚ := box.defective / box.total

/-- The two boxes of components -/
def box1 : Box := ⟨10, 2⟩
def box2 : Box := ⟨20, 3⟩

/-- The main theorem stating the probability of selecting a defective component -/
theorem defective_probability : 
  boxProb * defectiveProb box1 + boxProb * defectiveProb box2 = 7 / 40 := by
  sorry

end defective_probability_l3715_371586


namespace polynomial_factorization_l3715_371517

theorem polynomial_factorization (a x : ℝ) : a * x^2 - a * x - 2 * a = a * (x - 2) * (x + 1) := by
  sorry

end polynomial_factorization_l3715_371517


namespace smallest_area_of_2020th_square_l3715_371572

theorem smallest_area_of_2020th_square :
  ∀ (n : ℕ),
  (∃ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1) →
  (∀ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1 → a ≥ 112225) :=
by sorry

end smallest_area_of_2020th_square_l3715_371572


namespace m_range_for_monotonic_function_l3715_371513

-- Define a monotonically increasing function on ℝ
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem m_range_for_monotonic_function (f : ℝ → ℝ) (m : ℝ) 
  (h1 : MonotonicIncreasing f) (h2 : f (m^2) > f (-m)) : 
  m ∈ Set.Ioi 0 ∪ Set.Iic (-1) :=
sorry

end m_range_for_monotonic_function_l3715_371513


namespace number_difference_l3715_371526

theorem number_difference (a b : ℕ) : 
  a + b = 34800 → 
  b % 25 = 0 → 
  b = 25 * a → 
  b - a = 32112 := by
sorry

end number_difference_l3715_371526


namespace answer_key_combinations_l3715_371520

/-- Represents the number of possible answers for a true-false question -/
def true_false_options : ℕ := 2

/-- Represents the number of possible answers for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions in the quiz -/
def num_true_false : ℕ := 3

/-- Represents the number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 3

/-- Calculates the number of ways to arrange true-false answers where all answers cannot be the same -/
def true_false_combinations : ℕ := true_false_options ^ num_true_false - 2

/-- Calculates the number of ways to arrange multiple-choice answers -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- Theorem stating that the total number of ways to create an answer key is 384 -/
theorem answer_key_combinations : 
  true_false_combinations * multiple_choice_combinations = 384 := by
  sorry

end answer_key_combinations_l3715_371520


namespace indigo_restaurant_reviews_l3715_371570

theorem indigo_restaurant_reviews :
  let five_star : ℕ := 6
  let four_star : ℕ := 7
  let three_star : ℕ := 4
  let two_star : ℕ := 1
  let average_rating : ℚ := 4
  let total_reviews := five_star + four_star + three_star + two_star
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  (total_stars : ℚ) / total_reviews = average_rating →
  total_reviews = 18 := by
sorry

end indigo_restaurant_reviews_l3715_371570


namespace largest_power_of_two_dividing_3_512_minus_1_l3715_371510

theorem largest_power_of_two_dividing_3_512_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) ∧
  (∀ (n : ℕ), (2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) → n = 11) :=
sorry

end largest_power_of_two_dividing_3_512_minus_1_l3715_371510


namespace expression_values_l3715_371542

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 :=
by sorry

end expression_values_l3715_371542


namespace room_tiling_theorem_l3715_371530

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications -/
def tiles_needed (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width) + 4 * border_width * border_width
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let large_tiles := (inner_area + 8) / 9  -- Ceiling division
  border_tiles + large_tiles

/-- The theorem stating that 80 tiles are needed for the given room specifications -/
theorem room_tiling_theorem : tiles_needed 18 14 2 = 80 := by
  sorry

end room_tiling_theorem_l3715_371530


namespace ticket_sales_l3715_371590

theorem ticket_sales (adult_price children_price total_amount adult_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : children_price = 2)
  (h3 : total_amount = 275)
  (h4 : adult_tickets = 35) :
  ∃ children_tickets : ℕ, adult_tickets + children_tickets = 85 ∧ 
    adult_price * adult_tickets + children_price * children_tickets = total_amount :=
by sorry

end ticket_sales_l3715_371590


namespace nick_sold_fewer_bottles_l3715_371505

/-- Proves that Nick sold 6 fewer bottles of soda than Remy in the morning -/
theorem nick_sold_fewer_bottles (remy_morning : ℕ) (price : ℚ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  price = 1/2 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (nick_morning : ℕ), remy_morning - nick_morning = 6 :=
by
  sorry

end nick_sold_fewer_bottles_l3715_371505


namespace spadesuit_calculation_l3715_371540

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem spadesuit_calculation : spadesuit 3 (spadesuit 5 (spadesuit 7 10)) = 1 := by
  sorry

end spadesuit_calculation_l3715_371540


namespace percent_relation_l3715_371514

theorem percent_relation (a b c : ℝ) (x : ℝ) 
  (h1 : c = 0.20 * a) 
  (h2 : b = 2.00 * a) 
  (h3 : c = (x / 100) * b) : 
  x = 10 := by
sorry

end percent_relation_l3715_371514


namespace chicken_egg_production_l3715_371545

theorem chicken_egg_production 
  (num_chickens : ℕ) 
  (price_per_dozen : ℚ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) :
  num_chickens = 8 →
  price_per_dozen = 5 →
  total_revenue = 280 →
  num_weeks = 4 →
  (total_revenue / price_per_dozen * 12) / (num_weeks * 7) / num_chickens = 3 :=
by sorry

end chicken_egg_production_l3715_371545


namespace no_solutions_cubic_equation_l3715_371594

theorem no_solutions_cubic_equation :
  (∀ x y : ℕ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) ∧
  (∀ x y : ℤ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) :=
by sorry

end no_solutions_cubic_equation_l3715_371594


namespace wilson_payment_is_17_10_l3715_371559

/-- Calculates the total payment for Wilson's fast-food order --/
def wilsonPayment (hamburgerPrice fryPrice colaPrice sundaePrice couponDiscount loyaltyDiscount : ℚ) : ℚ :=
  let subtotal := 2 * hamburgerPrice + 3 * colaPrice + fryPrice + sundaePrice
  let afterCoupon := subtotal - couponDiscount
  afterCoupon * (1 - loyaltyDiscount)

/-- Theorem stating that Wilson's payment is $17.10 --/
theorem wilson_payment_is_17_10 :
  wilsonPayment 5 3 2 4 4 (1/10) = 171/10 := by
  sorry

end wilson_payment_is_17_10_l3715_371559


namespace cannot_cover_naturals_with_disjoint_sets_l3715_371500

def S (α : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * α⌋}

theorem cannot_cover_naturals_with_disjoint_sets :
  ∀ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 →
  ¬(Disjoint (S α) (S β) ∧ Disjoint (S α) (S γ) ∧ Disjoint (S β) (S γ) ∧
    (S α ∪ S β ∪ S γ) = Set.univ) :=
sorry

end cannot_cover_naturals_with_disjoint_sets_l3715_371500


namespace pattern_result_l3715_371551

-- Define the pattern function
def pattern (a b : ℕ) : ℕ := sorry

-- Define the given operations
axiom op1 : pattern 3 7 = 27
axiom op2 : pattern 4 5 = 32
axiom op3 : pattern 5 8 = 60
axiom op4 : pattern 6 7 = 72
axiom op5 : pattern 7 8 = 98

-- Theorem to prove
theorem pattern_result : pattern 2 3 = 26 := by sorry

end pattern_result_l3715_371551
