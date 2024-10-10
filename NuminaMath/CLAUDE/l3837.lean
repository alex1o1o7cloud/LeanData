import Mathlib

namespace parabola_chord_length_l3837_383773

/-- Parabola struct representing y^2 = ax --/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Line struct representing y = m(x - h) + k --/
structure Line where
  m : ℝ
  h : ℝ
  k : ℝ
  eq : ∀ x y : ℝ, y = m * (x - h) + k

/-- The length of the chord AB formed by intersecting a parabola with a line --/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_chord_length :
  let p : Parabola := { a := 3, eq := sorry }
  let f : ℝ × ℝ := (3/4, 0)
  let l : Line := { m := Real.sqrt 3 / 3, h := 3/4, k := 0, eq := sorry }
  chordLength p l = 12 := by sorry

end parabola_chord_length_l3837_383773


namespace optimal_journey_solution_l3837_383795

/-- Represents the problem setup for the journey from M to N --/
structure JourneySetup where
  total_distance : ℝ
  walking_speed : ℝ
  cycling_speed : ℝ

/-- Represents the optimal solution for the journey --/
structure OptimalSolution where
  c_departure_time : ℝ
  walking_distance : ℝ
  cycling_distance : ℝ

/-- Theorem stating the optimal solution for the journey --/
theorem optimal_journey_solution (setup : JourneySetup) 
  (h1 : setup.total_distance = 15)
  (h2 : setup.walking_speed = 6)
  (h3 : setup.cycling_speed = 15) :
  ∃ (sol : OptimalSolution), 
    sol.c_departure_time = 3 / 11 ∧
    sol.walking_distance = 60 / 11 ∧
    sol.cycling_distance = 105 / 11 ∧
    (sol.walking_distance / setup.walking_speed + 
     sol.cycling_distance / setup.cycling_speed = 
     setup.total_distance / setup.cycling_speed + 
     sol.walking_distance / setup.walking_speed) ∧
    ∀ (other : OptimalSolution), 
      (other.walking_distance / setup.walking_speed + 
       other.cycling_distance / setup.cycling_speed ≥
       sol.walking_distance / setup.walking_speed + 
       sol.cycling_distance / setup.cycling_speed) :=
by sorry


end optimal_journey_solution_l3837_383795


namespace traveler_money_problem_l3837_383789

/-- Represents the amount of money a traveler has at the start of each day -/
def money_at_day (initial_money : ℚ) : ℕ → ℚ
  | 0 => initial_money
  | n + 1 => (money_at_day initial_money n / 2) - 1

theorem traveler_money_problem (initial_money : ℚ) :
  (money_at_day initial_money 0 > 0) ∧
  (money_at_day initial_money 1 > 0) ∧
  (money_at_day initial_money 2 > 0) ∧
  (money_at_day initial_money 3 = 0) →
  initial_money = 14 := by
sorry

end traveler_money_problem_l3837_383789


namespace option_C_equals_nine_l3837_383771

theorem option_C_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end option_C_equals_nine_l3837_383771


namespace icosahedron_edge_probability_l3837_383711

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Nat)
  (edges : Nat)
  (is_regular : vertices = 12 ∧ edges = 30)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  i.edges / (i.vertices.choose 2)

/-- Theorem stating that the probability of randomly selecting two vertices 
    of a regular icosahedron that form an edge is 5/11 -/
theorem icosahedron_edge_probability :
  ∀ i : Icosahedron, edge_probability i = 5 / 11 := by
  sorry

end icosahedron_edge_probability_l3837_383711


namespace square_triangle_area_equality_l3837_383717

theorem square_triangle_area_equality (x : ℝ) (h : x > 0) :
  let square_area := x^2
  let triangle_base := x
  let triangle_altitude := 2 * x
  let triangle_area := (1 / 2) * triangle_base * triangle_altitude
  square_area = triangle_area := by
  sorry

end square_triangle_area_equality_l3837_383717


namespace brownie_pieces_count_l3837_383732

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end brownie_pieces_count_l3837_383732


namespace prince_total_spent_prince_total_spent_proof_l3837_383747

-- Define the total number of CDs
def total_cds : ℕ := 200

-- Define the percentage of CDs that cost $10
def percentage_expensive : ℚ := 40 / 100

-- Define the cost of expensive CDs
def cost_expensive : ℕ := 10

-- Define the cost of cheap CDs
def cost_cheap : ℕ := 5

-- Define the fraction of expensive CDs Prince bought
def fraction_bought : ℚ := 1 / 2

-- Theorem to prove
theorem prince_total_spent (total_cds : ℕ) (percentage_expensive : ℚ) 
  (cost_expensive cost_cheap : ℕ) (fraction_bought : ℚ) : ℕ :=
  -- The total amount Prince spent on CDs
  1000

-- Proof of the theorem
theorem prince_total_spent_proof :
  prince_total_spent total_cds percentage_expensive cost_expensive cost_cheap fraction_bought = 1000 := by
  sorry

end prince_total_spent_prince_total_spent_proof_l3837_383747


namespace series_sum_210_l3837_383779

def series_sum (n : ℕ) : ℤ :=
  let groups := n / 3
  let last_term := 3 * (groups - 1)
  (groups : ℤ) * last_term / 2

theorem series_sum_210 :
  series_sum 210 = 7245 := by
  sorry

end series_sum_210_l3837_383779


namespace expand_expression_l3837_383793

-- Statement of the theorem
theorem expand_expression (x : ℝ) : (x + 3) * (6 * x - 12) = 6 * x^2 + 6 * x - 36 := by
  sorry

end expand_expression_l3837_383793


namespace horner_method_v₂_l3837_383760

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ × ℤ × ℤ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  (v₀, v₁, v₂)

/-- The polynomial f(x) = 208 + 9x² + 6x⁴ + x⁶ -/
def f (x : ℤ) : ℤ := 208 + 9*x^2 + 6*x^4 + x^6

theorem horner_method_v₂ : 
  (horner 208 0 9 0 6 0 1 (-4)).2.2 = 22 := by sorry

end horner_method_v₂_l3837_383760


namespace difference_of_squares_l3837_383750

theorem difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by sorry

end difference_of_squares_l3837_383750


namespace interior_triangle_area_l3837_383796

theorem interior_triangle_area (a b c : ℝ) (ha : a = 64) (hb : b = 225) (hc : c = 289)
  (h_right_triangle : a + b = c) : (1/2) * Real.sqrt a * Real.sqrt b = 60 := by
  sorry

end interior_triangle_area_l3837_383796


namespace max_monthly_profit_l3837_383770

/-- Represents the monthly profit function for Xiao Ming's eye-protecting desk lamp business. -/
def monthly_profit (x : ℝ) : ℝ := -10 * x^2 + 700 * x - 10000

/-- Represents the monthly sales volume function. -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

/-- The cost price of each lamp. -/
def cost_price : ℝ := 20

/-- The maximum allowed profit percentage. -/
def max_profit_percentage : ℝ := 0.6

/-- Theorem stating the maximum monthly profit and the corresponding selling price. -/
theorem max_monthly_profit :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    max_profit = 2160 ∧
    optimal_price = 32 ∧
    (∀ x : ℝ, cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) →
      monthly_profit x ≤ max_profit) ∧
    monthly_profit optimal_price = max_profit :=
  sorry

/-- Lemma: The monthly profit function is correctly defined based on the given conditions. -/
lemma profit_function_correct :
  ∀ x : ℝ, monthly_profit x = (x - cost_price) * sales_volume x :=
  sorry

/-- Lemma: The selling price is within the specified range. -/
lemma selling_price_range :
  ∀ x : ℝ, monthly_profit x > 0 → cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) :=
  sorry

end max_monthly_profit_l3837_383770


namespace weight_of_other_new_member_l3837_383751

/-- Given the initial and final average weights of a group, the number of initial members,
    and the weight of one new member, calculate the weight of the other new member. -/
theorem weight_of_other_new_member
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_of_one_new_member : ℝ)
  (h1 : initial_average = 48)
  (h2 : final_average = 51)
  (h3 : initial_members = 23)
  (h4 : weight_of_one_new_member = 78) :
  (initial_members + 2) * final_average - initial_members * initial_average - weight_of_one_new_member = 93 :=
by sorry

end weight_of_other_new_member_l3837_383751


namespace solve_dancers_earnings_l3837_383757

def dancers_earnings (total : ℚ) (d1 d2 d3 d4 : ℚ) : Prop :=
  d1 + d2 + d3 + d4 = total ∧
  d2 = d1 - 16 ∧
  d3 = d1 + d2 - 24 ∧
  d4 = d1 + d3

theorem solve_dancers_earnings :
  ∃ d1 d2 d3 d4 : ℚ,
    dancers_earnings 280 d1 d2 d3 d4 ∧
    d1 = 53 + 5/7 ∧
    d2 = 37 + 5/7 ∧
    d3 = 67 + 3/7 ∧
    d4 = 121 + 1/7 :=
by sorry

end solve_dancers_earnings_l3837_383757


namespace min_l_trominos_count_l3837_383710

/-- Represents a tile type -/
inductive TileType
| LTromino
| STetromino

/-- Represents the grid -/
def Grid := Fin 2020 × Fin 2021

/-- A tiling is a function that assigns a tile type to each grid position -/
def Tiling := Grid → Option TileType

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def is_valid_tiling (t : Tiling) : Prop := sorry

/-- Counts the number of L-Trominos in a tiling -/
def count_l_trominos (t : Tiling) : Nat := sorry

/-- Theorem: The minimum number of L-Trominos in a valid tiling is 1010 -/
theorem min_l_trominos_count :
  ∃ (t : Tiling), is_valid_tiling t ∧
    ∀ (t' : Tiling), is_valid_tiling t' →
      count_l_trominos t ≤ count_l_trominos t' ∧
      count_l_trominos t = 1010 :=
sorry

end min_l_trominos_count_l3837_383710


namespace line_through_points_equation_l3837_383762

-- Define a line by two points
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define the equation of a line in the form ax + by + c = 0
def LineEquation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem statement
theorem line_through_points_equation :
  Line (3, 0) (0, 2) = LineEquation 2 3 (-6) := by sorry

end line_through_points_equation_l3837_383762


namespace problem_solution_l3837_383737

theorem problem_solution :
  (∀ x : ℝ, |x + 2| + |6 - x| ≥ 8) ∧
  (∃ x : ℝ, |x + 2| + |6 - x| = 8) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 → 7 * a + 4 * b ≥ 9 / 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 ∧ 7 * a + 4 * b = 9 / 4) :=
by sorry

end problem_solution_l3837_383737


namespace page_lines_increase_percentage_l3837_383765

theorem page_lines_increase_percentage : 
  ∀ (original_lines : ℕ), 
  original_lines + 200 = 350 → 
  (200 : ℝ) / original_lines * 100 = 400 / 3 := by
sorry

end page_lines_increase_percentage_l3837_383765


namespace john_cycling_distance_l3837_383755

def base_eight_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem john_cycling_distance : base_eight_to_decimal 6375 = 3325 := by
  sorry

end john_cycling_distance_l3837_383755


namespace inequality_solution_l3837_383721

theorem inequality_solution (x y : ℝ) : 
  2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by sorry

end inequality_solution_l3837_383721


namespace parabola_directrix_l3837_383772

/-- The parabola defined by y = 8x^2 + 2 has a directrix y = 63/32 -/
theorem parabola_directrix : ∀ (x y : ℝ), y = 8 * x^2 + 2 → 
  ∃ (f d : ℝ), f = -d ∧ f - d = 1/16 ∧ d = -1/32 ∧ 
  (∀ (p : ℝ × ℝ), p.2 = 8 * p.1^2 + 2 → 
    (p.1^2 + (p.2 - (f + 2))^2 = (p.2 - (d + 2))^2)) ∧
  63/32 = d + 2 := by
  sorry


end parabola_directrix_l3837_383772


namespace arithmetic_progression_y_range_l3837_383706

theorem arithmetic_progression_y_range (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    Real.log r = Real.log 2 - Real.log (Real.sin x - 1/3) ∧ 
    Real.log (Real.sin x - 1/3) = Real.log 2 - Real.log (1 - y)) →
  (∃ y_min : ℝ, y_min = 7/9 ∧ y ≥ y_min) ∧ 
  (∀ y_max : ℝ, y < y_max) :=
by sorry

end arithmetic_progression_y_range_l3837_383706


namespace production_days_calculation_l3837_383705

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of past n days
  (h2 : ((n * 50 + 95 : ℝ) / (n + 1) = 55)) : -- New average including today
  n = 8 := by
sorry

end production_days_calculation_l3837_383705


namespace quadratic_roots_relation_l3837_383718

theorem quadratic_roots_relation : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 4 = 0) → 
  (x₂^2 - 2*x₂ - 4 = 0) → 
  x₁ ≠ x₂ →
  (x₁ + x₂) / (x₁ * x₂) = -1/2 := by
  sorry

end quadratic_roots_relation_l3837_383718


namespace f_properties_l3837_383742

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a)^2 * (x + b) * Real.exp x

-- Define the first derivative of f
def f' (a b x : ℝ) : ℝ := Real.exp x * (x - a) * (x^2 + (3 - a + b) * x + 2 * b - a * b - a)

-- State the theorem
theorem f_properties (a b : ℝ) :
  (∀ x, f' a b x ≤ f' a b a) →
  ((a = 0 → b < 0) ∧
   (∃ x₄, (x₄ = a + 2 * Real.sqrt 6 ∨ x₄ = a - 2 * Real.sqrt 6) ∧ b = -a - 3) ∨
   (∃ x₄, x₄ = a + (1 + Real.sqrt 13) / 2 ∧ b = -a - (7 + Real.sqrt 13) / 2) ∨
   (∃ x₄, x₄ = a + (1 - Real.sqrt 13) / 2 ∧ b = -a - (7 - Real.sqrt 13) / 2)) :=
by sorry

end

end f_properties_l3837_383742


namespace eight_teams_satisfy_conditions_l3837_383749

/-- The number of days in the tournament -/
def tournament_days : ℕ := 7

/-- The number of games scheduled per day -/
def games_per_day : ℕ := 4

/-- The total number of games in the tournament -/
def total_games : ℕ := tournament_days * games_per_day

/-- Function to calculate the number of games for a given number of teams -/
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 8 teams satisfy the tournament conditions -/
theorem eight_teams_satisfy_conditions : 
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = total_games ∧ n = 8 :=
sorry

end eight_teams_satisfy_conditions_l3837_383749


namespace michaels_crayons_value_l3837_383728

/-- The value of crayons Michael will have after the purchase -/
def total_value (initial_packs : ℕ) (additional_packs : ℕ) (price_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs : ℚ) * price_per_pack

/-- Proof that Michael's crayons will be worth $15 after the purchase -/
theorem michaels_crayons_value :
  total_value 4 2 (5/2) = 15 := by
  sorry

end michaels_crayons_value_l3837_383728


namespace complex_parts_of_1_plus_sqrt3_i_l3837_383712

theorem complex_parts_of_1_plus_sqrt3_i : 
  let z : ℂ := Complex.I * (1 + Real.sqrt 3)
  (z.re = 0) ∧ (z.im = 1 + Real.sqrt 3) := by sorry

end complex_parts_of_1_plus_sqrt3_i_l3837_383712


namespace summer_lecture_team_selection_probability_l3837_383715

/-- Represents the probability of a teacher being selected for the summer lecture team -/
def selection_probability (total : ℕ) (eliminated : ℕ) (team_size : ℕ) : ℚ :=
  team_size / (total - eliminated)

theorem summer_lecture_team_selection_probability :
  selection_probability 118 6 16 = 1 / 7 := by
  sorry

end summer_lecture_team_selection_probability_l3837_383715


namespace lucy_balance_l3837_383714

/-- Calculates the final balance after a deposit and withdrawal --/
def final_balance (initial : ℕ) (deposit : ℕ) (withdrawal : ℕ) : ℕ :=
  initial + deposit - withdrawal

/-- Proves that Lucy's final balance is $76 --/
theorem lucy_balance : final_balance 65 15 4 = 76 := by
  sorry

end lucy_balance_l3837_383714


namespace max_savings_63_l3837_383790

/-- Represents the price of a pastry package -/
structure PastryPrice where
  quantity : Nat
  price : Nat

/-- Represents the discount options for a type of pastry -/
structure PastryDiscount where
  regular_price : Nat
  discounts : List PastryPrice

/-- Calculates the minimum cost for a given quantity using available discounts -/
def min_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Calculates the cost without any discounts -/
def regular_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Doughnut discount options -/
def doughnut_discount : PastryDiscount :=
  { regular_price := 8,
    discounts := [
      { quantity := 12, price := 8 },
      { quantity := 24, price := 14 },
      { quantity := 48, price := 26 }
    ] }

/-- Croissant discount options -/
def croissant_discount : PastryDiscount :=
  { regular_price := 10,
    discounts := [
      { quantity := 12, price := 10 },
      { quantity := 36, price := 28 },
      { quantity := 60, price := 45 }
    ] }

/-- Muffin discount options -/
def muffin_discount : PastryDiscount :=
  { regular_price := 6,
    discounts := [
      { quantity := 12, price := 6 },
      { quantity := 24, price := 11 },
      { quantity := 72, price := 30 }
    ] }

theorem max_savings_63 :
  let doughnut_qty := 20 * 12
  let croissant_qty := 15 * 12
  let muffin_qty := 18 * 12
  let total_discounted := min_cost doughnut_discount doughnut_qty +
                          min_cost croissant_discount croissant_qty +
                          min_cost muffin_discount muffin_qty
  let total_regular := regular_cost doughnut_discount doughnut_qty +
                       regular_cost croissant_discount croissant_qty +
                       regular_cost muffin_discount muffin_qty
  total_regular - total_discounted = 63 :=
by sorry

end max_savings_63_l3837_383790


namespace joan_seashells_l3837_383778

/-- The number of seashells Joan has after receiving some from Sam -/
def total_seashells (original : ℕ) (received : ℕ) : ℕ :=
  original + received

/-- Theorem: If Joan found 70 seashells and Sam gave her 27 seashells, 
    then Joan now has 97 seashells -/
theorem joan_seashells : total_seashells 70 27 = 97 := by
  sorry

end joan_seashells_l3837_383778


namespace washer_dryer_cost_difference_l3837_383775

theorem washer_dryer_cost_difference :
  ∀ (washer_cost dryer_cost : ℝ),
    dryer_cost = 490 →
    washer_cost > dryer_cost →
    washer_cost + dryer_cost = 1200 →
    washer_cost - dryer_cost = 220 :=
by
  sorry

end washer_dryer_cost_difference_l3837_383775


namespace line_intersects_ellipse_l3837_383723

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m^2 ≥ 1/20}

/-- Theorem stating the condition for the line to intersect the ellipse -/
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ possible_slopes := by
  sorry

end line_intersects_ellipse_l3837_383723


namespace f_is_even_l3837_383744

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^4

theorem f_is_even : is_even_function f := by
  sorry

end f_is_even_l3837_383744


namespace tony_water_consumption_l3837_383785

theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago - 0.04 * two_days_ago) :
  two_days_ago = 50 := by
  sorry

end tony_water_consumption_l3837_383785


namespace geometric_sequence_property_l3837_383769

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

/-- The property that 2a_2 + a_3 = a_4 for a geometric sequence -/
def property1 (a q : ℝ) : Prop :=
  2 * (geometric_sequence a q 2) + (geometric_sequence a q 3) = geometric_sequence a q 4

/-- The property that (a_2 + 1)(a_3 + 1) = a_5 - 1 for a geometric sequence -/
def property2 (a q : ℝ) : Prop :=
  (geometric_sequence a q 2 + 1) * (geometric_sequence a q 3 + 1) = geometric_sequence a q 5 - 1

/-- Theorem stating that for a geometric sequence satisfying both properties, a_1 ≠ 2 -/
theorem geometric_sequence_property (a q : ℝ) (h1 : property1 a q) (h2 : property2 a q) : a ≠ 2 := by
  sorry

end geometric_sequence_property_l3837_383769


namespace round_and_convert_0_000359_l3837_383754

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- Rounds a real number to a given number of significant figures -/
def round_to_sig_figs (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem round_and_convert_0_000359 :
  let rounded := round_to_sig_figs 0.000359 2
  let scientific := to_scientific_notation rounded
  scientific.coefficient = 3.6 ∧ scientific.exponent = -4 := by
  sorry

end round_and_convert_0_000359_l3837_383754


namespace min_total_cost_l3837_383733

/-- Represents the number of rooms of each type -/
structure RoomAllocation where
  triple : ℕ
  double : ℕ
  single : ℕ

/-- Calculates the total cost for a given room allocation -/
def totalCost (a : RoomAllocation) : ℕ :=
  300 * a.triple + 300 * a.double + 200 * a.single

/-- Checks if a room allocation is valid for the given constraints -/
def isValidAllocation (a : RoomAllocation) : Prop :=
  a.triple + a.double + a.single = 20 ∧
  3 * a.triple + 2 * a.double + a.single = 50

/-- Theorem: The minimum total cost for the given constraints is 5500 yuan -/
theorem min_total_cost :
  ∃ (a : RoomAllocation), isValidAllocation a ∧
    totalCost a = 5500 ∧
    ∀ (b : RoomAllocation), isValidAllocation b → totalCost a ≤ totalCost b :=
  sorry

end min_total_cost_l3837_383733


namespace line_inclination_angle_l3837_383781

/-- The inclination angle of a line is the angle it makes with the positive x-axis. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := -1, c := 1 }
  inclination_angle l.a l.b l.c = 45 * π / 180 := by sorry

end line_inclination_angle_l3837_383781


namespace drum_oil_capacity_l3837_383716

theorem drum_oil_capacity (c : ℝ) (h1 : c > 0) : 
  let drum_x_capacity := c
  let drum_x_oil := (1 / 2 : ℝ) * drum_x_capacity
  let drum_y_capacity := 2 * drum_x_capacity
  let drum_y_oil := (1 / 3 : ℝ) * drum_y_capacity
  let final_oil := drum_y_oil + drum_x_oil
  final_oil / drum_y_capacity = 7 / 12
  := by sorry

end drum_oil_capacity_l3837_383716


namespace four_char_word_count_l3837_383724

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (excluding 'Y') -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of four-character words formed by arranging two consonants and two vowels
    in the order consonant-vowel-vowel-consonant -/
def word_count : ℕ := consonant_count * vowel_count * vowel_count * consonant_count

theorem four_char_word_count : word_count = 11025 := by
  sorry

end four_char_word_count_l3837_383724


namespace min_value_expression_min_value_achievable_l3837_383700

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) ≥ 216 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) = 216 :=
by sorry

end min_value_expression_min_value_achievable_l3837_383700


namespace x_range_l3837_383734

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 
  -1 ≤ x ∧ x < 5/4 := by
  sorry

end x_range_l3837_383734


namespace eight_digit_numbers_with_consecutive_digits_l3837_383743

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Total number of 8-digit numbers with digits 1 or 2 -/
def total_numbers : ℕ := 2^8

/-- Number of 8-digit numbers with no consecutive same digits -/
def numbers_without_consecutive : ℕ := 2 * fib 7

theorem eight_digit_numbers_with_consecutive_digits : 
  total_numbers - numbers_without_consecutive = 230 := by
  sorry

end eight_digit_numbers_with_consecutive_digits_l3837_383743


namespace angle_ABC_measure_l3837_383752

theorem angle_ABC_measure :
  ∀ (ABC ABD CBD : ℝ),
  CBD = 90 →
  ABC + ABD + CBD = 180 →
  ABD = 60 →
  ABC = 30 := by
sorry

end angle_ABC_measure_l3837_383752


namespace total_amount_l3837_383745

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℚ
  y : ℚ
  z : ℚ

/-- The problem setup -/
def problem_setup (s : ShareDistribution) : Prop :=
  s.y = 18 ∧ s.y = 0.45 * s.x ∧ s.z = 0.3 * s.x

/-- The theorem statement -/
theorem total_amount (s : ShareDistribution) : 
  problem_setup s → s.x + s.y + s.z = 70 := by
  sorry

end total_amount_l3837_383745


namespace complex_power_abs_l3837_383768

theorem complex_power_abs : Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 3) ^ 6 = 4096 := by
  sorry

end complex_power_abs_l3837_383768


namespace circle_diameter_problem_l3837_383767

/-- Given two circles A and B where A is inside B, proves that the diameter of A
    satisfies the given conditions. -/
theorem circle_diameter_problem (center_distance : ℝ) (diameter_B : ℝ) :
  center_distance = 5 →
  diameter_B = 20 →
  let radius_B := diameter_B / 2
  let area_B := π * radius_B ^ 2
  ∃ (radius_A : ℝ),
    π * radius_A ^ 2 * 6 = area_B ∧
    (2 * radius_A : ℝ) = 2 * Real.sqrt (50 / 3) := by
  sorry

end circle_diameter_problem_l3837_383767


namespace carla_bug_collection_l3837_383788

theorem carla_bug_collection (leaves : ℕ) (days : ℕ) (items_per_day : ℕ) 
  (h1 : leaves = 30)
  (h2 : days = 10)
  (h3 : items_per_day = 5) :
  let total_items := days * items_per_day
  let bugs := total_items - leaves
  bugs = 20 := by sorry

end carla_bug_collection_l3837_383788


namespace expression_value_l3837_383741

theorem expression_value (a : ℝ) (h : a ≠ 0) : (20 * a^5) * (8 * a^4) * (1 / (4 * a^3)^3) = 2.5 := by
  sorry

end expression_value_l3837_383741


namespace extreme_values_and_range_l3837_383739

/-- The function f(x) with parameters a, b, and c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (∀ x ∈ Set.Icc 0 3, f a b c x < c^2) →
  (a = -3 ∧ b = 4 ∧ c < -1 ∨ c > 9) :=
sorry

end extreme_values_and_range_l3837_383739


namespace pure_imaginary_solutions_l3837_383791

def polynomial (x : ℂ) : ℂ := x^4 - 3*x^3 + 5*x^2 - 27*x - 36

theorem pure_imaginary_solutions :
  ∃ (k : ℝ), k > 0 ∧ 
  polynomial (k * Complex.I) = 0 ∧
  polynomial (-k * Complex.I) = 0 ∧
  ∀ (z : ℂ), polynomial z = 0 → z.re = 0 → z = k * Complex.I ∨ z = -k * Complex.I :=
by sorry

end pure_imaginary_solutions_l3837_383791


namespace budget_remainder_l3837_383782

-- Define the given conditions
def weekly_budget : ℝ := 80
def fried_chicken_cost : ℝ := 12
def beef_pounds : ℝ := 4.5
def beef_price_per_pound : ℝ := 3
def soup_cans : ℕ := 3
def soup_cost_per_can : ℝ := 2
def milk_original_price : ℝ := 4
def milk_discount_percentage : ℝ := 0.1

-- Define the theorem
theorem budget_remainder : 
  let beef_cost := beef_pounds * beef_price_per_pound
  let soup_cost := (soup_cans - 1) * soup_cost_per_can
  let milk_cost := milk_original_price * (1 - milk_discount_percentage)
  let total_cost := fried_chicken_cost + beef_cost + soup_cost + milk_cost
  weekly_budget - total_cost = 46.90 := by
  sorry

end budget_remainder_l3837_383782


namespace negative_reals_inequality_l3837_383720

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (Real.sqrt (a / (b + c)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (b / (c + a)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (c / (a + b)) + 1 / Real.sqrt 2) ^ 2 ≥ 6 := by
  sorry

end negative_reals_inequality_l3837_383720


namespace min_editors_at_conference_l3837_383784

theorem min_editors_at_conference (total : Nat) (writers : Nat) (x : Nat) :
  total = 100 →
  writers = 45 →
  x ≤ 18 →
  total = writers + (55 + x) - x + 2 * x →
  55 + x ≥ 73 :=
by
  sorry

end min_editors_at_conference_l3837_383784


namespace y_minimum_value_l3837_383777

/-- The function y in terms of x, a, b, and k -/
def y (x a b k : ℝ) : ℝ := 3 * (x - a)^2 + (x - b)^2 + k * x

/-- The derivative of y with respect to x -/
def y_deriv (x a b k : ℝ) : ℝ := 8 * x - 6 * a - 2 * b + k

/-- The second derivative of y with respect to x -/
def y_second_deriv : ℝ := 8

theorem y_minimum_value (a b k : ℝ) :
  ∃ x : ℝ, y_deriv x a b k = 0 ∧
           y_second_deriv > 0 ∧
           x = (6 * a + 2 * b - k) / 8 :=
sorry

end y_minimum_value_l3837_383777


namespace min_value_expression_l3837_383763

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x) / (x + 2 * y) + y / x ≥ 3 / 2 := by
  sorry

end min_value_expression_l3837_383763


namespace min_value_expression_min_value_attainable_l3837_383725

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

theorem min_value_attainable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) = 24 :=
by
  sorry

end min_value_expression_min_value_attainable_l3837_383725


namespace charity_event_probability_l3837_383764

/-- The number of students participating in the charity event -/
def num_students : ℕ := 4

/-- The number of days students can choose from (Saturday and Sunday) -/
def num_days : ℕ := 2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := num_days ^ num_students

/-- The number of outcomes where students participate on both days -/
def both_days_outcomes : ℕ := total_outcomes - num_days

/-- The probability of students participating on both days -/
def probability_both_days : ℚ := both_days_outcomes / total_outcomes

theorem charity_event_probability : probability_both_days = 7/8 := by
  sorry

end charity_event_probability_l3837_383764


namespace lisa_phone_expenses_l3837_383783

/-- Calculate the total cost of Lisa's phone and related expenses after three years -/
theorem lisa_phone_expenses :
  let iphone_cost : ℝ := 1000
  let monthly_contract : ℝ := 200
  let case_cost : ℝ := 0.2 * iphone_cost
  let headphones_cost : ℝ := 0.5 * case_cost
  let charger_cost : ℝ := 60
  let warranty_cost : ℝ := 150
  let discount_rate : ℝ := 0.1
  let years : ℝ := 3

  let discounted_case_cost : ℝ := case_cost * (1 - discount_rate)
  let discounted_headphones_cost : ℝ := headphones_cost * (1 - discount_rate)
  let total_contract_cost : ℝ := monthly_contract * 12 * years

  let total_cost : ℝ := iphone_cost + total_contract_cost + discounted_case_cost + 
                        discounted_headphones_cost + charger_cost + warranty_cost

  total_cost = 8680 := by sorry

end lisa_phone_expenses_l3837_383783


namespace negation_of_existence_l3837_383707

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end negation_of_existence_l3837_383707


namespace sector_area_l3837_383798

/-- The area of a circular sector with radius 2 cm and central angle 120° is 4π/3 cm² -/
theorem sector_area (r : ℝ) (θ_deg : ℝ) (A : ℝ) : 
  r = 2 → θ_deg = 120 → A = (1/2) * r^2 * (θ_deg * π / 180) → A = (4/3) * π := by
  sorry

end sector_area_l3837_383798


namespace unique_positive_solution_l3837_383702

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l3837_383702


namespace integer_solutions_equation_l3837_383787

theorem integer_solutions_equation :
  ∀ x y : ℤ, 2*x^2 + 8*y^2 = 17*x*y - 423 ↔ (x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19) := by
  sorry

end integer_solutions_equation_l3837_383787


namespace exists_same_type_quadratic_surd_with_three_l3837_383759

/-- Two square roots are of the same type of quadratic surd if one can be expressed as a rational multiple of the other. -/
def same_type_quadratic_surd (x y : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * y ∨ y = q * x

theorem exists_same_type_quadratic_surd_with_three :
  ∃ (a : ℕ), a > 0 ∧ same_type_quadratic_surd (Real.sqrt a) (Real.sqrt 3) ∧ a = 12 := by
  sorry

end exists_same_type_quadratic_surd_with_three_l3837_383759


namespace tom_total_money_l3837_383735

/-- Tom's initial amount of money in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def car_wash_earnings : ℕ := 86

/-- Tom's total money after washing cars -/
def total_money : ℕ := initial_amount + car_wash_earnings

theorem tom_total_money :
  total_money = 160 := by sorry

end tom_total_money_l3837_383735


namespace badminton_tournament_l3837_383799

theorem badminton_tournament (n : ℕ) : 
  (∃ (x : ℕ), 
    (5 * n * (5 * n - 1)) / 2 = 7 * x ∧ 
    4 * x = (2 * n * (2 * n - 1)) / 2 + 2 * n * 3 * n) → 
  n = 3 :=
by
  sorry


end badminton_tournament_l3837_383799


namespace max_value_of_d_l3837_383729

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (prod_sum_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ 5 + (5 * Real.sqrt 34) / 3 := by
sorry

end max_value_of_d_l3837_383729


namespace sugar_sacks_weight_difference_l3837_383722

theorem sugar_sacks_weight_difference (x y : ℝ) : 
  x + y = 40 →
  x - 1 = 0.6 * (y + 1) →
  |x - y| = 8 := by
sorry

end sugar_sacks_weight_difference_l3837_383722


namespace square_plus_reciprocal_square_l3837_383726

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 3) : x^2 + (1/x^2) = 7 := by
  sorry

end square_plus_reciprocal_square_l3837_383726


namespace subtract_negative_real_l3837_383736

theorem subtract_negative_real : 3.7 - (-1.45) = 5.15 := by
  sorry

end subtract_negative_real_l3837_383736


namespace min_union_cardinality_l3837_383766

theorem min_union_cardinality (A B : Finset ℕ) (hA : A.card = 30) (hB : B.card = 20) :
  35 ≤ (A ∪ B).card := by sorry

end min_union_cardinality_l3837_383766


namespace smallest_b_for_inequality_l3837_383727

theorem smallest_b_for_inequality (b : ℕ) : (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ↔ b = 9 := by
  sorry

end smallest_b_for_inequality_l3837_383727


namespace negative_x_power_seven_divided_by_negative_x_l3837_383709

theorem negative_x_power_seven_divided_by_negative_x (x : ℝ) :
  ((-x)^7) / (-x) = x^6 := by sorry

end negative_x_power_seven_divided_by_negative_x_l3837_383709


namespace jack_and_jill_probability_l3837_383794

/-- The probability of selecting both Jack and Jill when choosing 2 workers at random -/
def probability : ℚ := 1/6

/-- The number of other workers besides Jack and Jill -/
def other_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem jack_and_jill_probability :
  (1 : ℚ) / (total_workers.choose 2) = probability → other_workers = 2 := by
  sorry

end jack_and_jill_probability_l3837_383794


namespace problem_solution_l3837_383701

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 1

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6*a^2 * log x + 2*b + 1

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_solution (a : ℝ) (ha : a > 0) :
  ∃ b : ℝ,
    (∃ x : ℝ, x > 0 ∧ f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x) ∧
    b = (5/2)*a^2 - 3*a^2 * log a ∧
    ∀ b' : ℝ, b' ≤ (3/2) * Real.exp ((2:ℝ)/3) ∧
    (a ≥ Real.sqrt 3 - 1 →
      ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
        (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8) :=
by sorry

end problem_solution_l3837_383701


namespace ship_distance_theorem_l3837_383730

/-- Represents the ship's position relative to Island X -/
structure ShipPosition where
  angle : ℝ  -- angle in radians for circular motion
  distance : ℝ -- distance from Island X

/-- Represents the ship's path -/
inductive ShipPath
  | Circle (t : ℝ) -- t represents time spent on circular path
  | StraightLine (t : ℝ) -- t represents time spent on straight line

/-- Function to calculate the ship's distance from Island X -/
def shipDistance (r : ℝ) (path : ShipPath) : ℝ :=
  match path with
  | ShipPath.Circle _ => r
  | ShipPath.StraightLine t => r + t

theorem ship_distance_theorem (r : ℝ) (h : r > 0) :
  ∃ (t₁ t₂ : ℝ), t₁ > 0 ∧ t₂ > 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ t₁ → shipDistance r (ShipPath.Circle t) = r) ∧
    (∀ t, t > t₁ ∧ t ≤ t₁ + t₂ → shipDistance r (ShipPath.StraightLine (t - t₁)) > r ∧
      (shipDistance r (ShipPath.StraightLine (t - t₁)) - r) = t - t₁) :=
  sorry

end ship_distance_theorem_l3837_383730


namespace response_rate_percentage_l3837_383792

theorem response_rate_percentage 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 240) 
  (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
  sorry

end response_rate_percentage_l3837_383792


namespace max_take_home_pay_l3837_383776

/-- The take-home pay function for a given income x (in thousands of dollars) -/
def takehomePay (x : ℝ) : ℝ := 1000 * x - 20 * x^2

/-- The income that maximizes take-home pay -/
def maxTakeHomeIncome : ℝ := 25

theorem max_take_home_pay :
  ∀ x : ℝ, takehomePay x ≤ takehomePay maxTakeHomeIncome :=
sorry

end max_take_home_pay_l3837_383776


namespace modulus_of_9_minus_40i_l3837_383708

theorem modulus_of_9_minus_40i : Complex.abs (9 - 40*I) = 41 := by sorry

end modulus_of_9_minus_40i_l3837_383708


namespace greatest_power_of_ten_dividing_twenty_factorial_l3837_383758

theorem greatest_power_of_ten_dividing_twenty_factorial : 
  (∃ m : ℕ, (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) → 
  (∃ m : ℕ, m = 4 ∧ (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) :=
by sorry

end greatest_power_of_ten_dividing_twenty_factorial_l3837_383758


namespace five_people_arrangement_with_restriction_l3837_383756

/-- The number of ways to arrange n people in a line. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where two specific people are always adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a line where two specific people are not adjacent. -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement_with_restriction :
  nonAdjacentArrangements 5 = 72 := by
  sorry

end five_people_arrangement_with_restriction_l3837_383756


namespace eccentricity_difference_l3837_383704

/-- Given an ellipse and a hyperbola sharing the same foci, prove that
    the difference of their eccentricities is √2 under certain conditions. -/
theorem eccentricity_difference (a b m n : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | x^2/a^2 + y^2/b^2 = 1}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/m^2 - y^2/n^2 = 1}
  let c := Real.sqrt (a^2 - b^2)
  let f := Real.sqrt (m^2 + n^2)
  let e₁ := c / a
  let e₂ := f / m
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∃ P ∈ C₁ ∩ C₂, P.1 > 0 ∧ P.2 > 0 ∧ 
  c = f ∧ 
  dist P F₁ = dist P F₂ ∧
  dist P F₁ = dist F₁ F₂ →
  e₂ - e₁ = Real.sqrt 2 :=
sorry


end eccentricity_difference_l3837_383704


namespace percentage_problem_l3837_383774

theorem percentage_problem (P : ℝ) : P * 300 - 70 = 20 → P = 0.3 := by
  sorry

end percentage_problem_l3837_383774


namespace indexCardsPerStudentIs10_l3837_383748

/-- Calculates the number of index cards each student receives given the following conditions:
  * Carl teaches 6 periods a day
  * Each class has 30 students
  * A 50 pack of index cards costs $3
  * Carl spent $108 on index cards
-/
def indexCardsPerStudent (periods : Nat) (studentsPerClass : Nat) (cardsPerPack : Nat) 
  (costPerPack : Nat) (totalSpent : Nat) : Nat :=
  let totalPacks := totalSpent / costPerPack
  let totalCards := totalPacks * cardsPerPack
  let totalStudents := periods * studentsPerClass
  totalCards / totalStudents

theorem indexCardsPerStudentIs10 : 
  indexCardsPerStudent 6 30 50 3 108 = 10 := by
  sorry

end indexCardsPerStudentIs10_l3837_383748


namespace number_of_hens_l3837_383786

/-- Represents the number of hens and cows a man has. -/
structure Animals where
  hens : ℕ
  cows : ℕ

/-- The total number of heads for the given animals. -/
def totalHeads (a : Animals) : ℕ := a.hens + a.cows

/-- The total number of feet for the given animals. -/
def totalFeet (a : Animals) : ℕ := 2 * a.hens + 4 * a.cows

/-- Theorem stating that given the conditions, the number of hens is 24. -/
theorem number_of_hens : 
  ∃ (a : Animals), totalHeads a = 48 ∧ totalFeet a = 144 ∧ a.hens = 24 := by
  sorry

end number_of_hens_l3837_383786


namespace coefficient_x_squared_in_expansion_l3837_383719

theorem coefficient_x_squared_in_expansion :
  let n : ℕ := 6
  let a : ℤ := 1
  let b : ℤ := -3
  (Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) * a^(n - k) * b^k * (if k = 2 then 1 else 0))) = 135 :=
by sorry

end coefficient_x_squared_in_expansion_l3837_383719


namespace ski_down_time_l3837_383738

-- Define the lift ride time
def lift_time : ℕ := 15

-- Define the number of round trips in 2 hours
def round_trips : ℕ := 6

-- Define the total time for 6 round trips in minutes
def total_time : ℕ := 2 * 60

-- Theorem: The time to ski down the mountain is 20 minutes
theorem ski_down_time : 
  (total_time / round_trips) - lift_time = 20 :=
sorry

end ski_down_time_l3837_383738


namespace ellipse_major_axis_length_l3837_383753

/-- An ellipse with foci at (7, 15) and (53, 65) that is tangent to the y-axis has a major axis of length 68. -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
    F₁ = (7, 15) →
    F₂ = (53, 65) →
    (∃ (y : ℝ), (0, y) ∈ E) →
    (∀ (P : ℝ × ℝ), P ∈ E ↔ 
      ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
      ∀ (Q : ℝ × ℝ), dist Q F₁ + dist Q F₂ ≤ k) →
    ∃ (a : ℝ), a = 68 ∧ 
      ∀ (P : ℝ × ℝ), P ∈ E → dist P F₁ + dist P F₂ = a :=
by
  sorry


end ellipse_major_axis_length_l3837_383753


namespace ellipse_condition_l3837_383703

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 / (k - 1) + y^2 / (5 - k) = 1 → is_ellipse k) ∧
  ¬(∀ k : ℝ, 1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end ellipse_condition_l3837_383703


namespace work_completion_proof_l3837_383780

/-- The number of days it takes a to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days it takes b to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days b worked alone to complete the remaining work -/
def b_remaining_days : ℕ := 23

/-- The number of days a worked before leaving -/
def days_a_worked : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  let combined_rate := a_rate + b_rate
  combined_rate * days_a_worked + b_rate * b_remaining_days = total_work :=
by sorry

end work_completion_proof_l3837_383780


namespace ninety_mile_fare_l3837_383797

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  baseFare : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.baseFare + tf.ratePerMile * distance

theorem ninety_mile_fare :
  ∃ (tf : TaxiFare),
    tf.baseFare = 30 ∧
    totalFare tf 60 = 150 ∧
    totalFare tf 90 = 210 := by
  sorry

end ninety_mile_fare_l3837_383797


namespace problem_solution_l3837_383713

theorem problem_solution (m n : ℝ) (h : 2 * m - n = 3) :
  (∀ x : ℝ, |x| + |n + 3| ≥ 9 → x ≤ -3 ∨ x ≥ 3) ∧
  (∃ min : ℝ, min = 3 ∧ ∀ x y : ℝ, 2 * x - y = 3 →
    |5/3 * x - 1/3 * y| + |1/3 * x - 2/3 * y| ≥ min) :=
by sorry

end problem_solution_l3837_383713


namespace line_passes_through_fixed_point_l3837_383746

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 1) * (-2) - 1 + (2 * m - 1) = 0 := by sorry

end line_passes_through_fixed_point_l3837_383746


namespace inverse_of_five_mod_221_l3837_383740

theorem inverse_of_five_mod_221 : ∃! x : ℕ, x ∈ Finset.range 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end inverse_of_five_mod_221_l3837_383740


namespace quadratic_completion_l3837_383761

theorem quadratic_completion (x : ℝ) : ∃ (a b : ℝ), x^2 - 6*x + 5 = 0 ↔ (x + a)^2 = b ∧ b = 4 := by
  sorry

end quadratic_completion_l3837_383761


namespace mandys_data_plan_charge_l3837_383731

/-- The normal monthly charge for Mandy's data plan -/
def normal_charge : ℝ := 30

/-- The total amount Mandy paid for 6 months -/
def total_paid : ℝ := 175

/-- The extra fee charged in the fourth month -/
def extra_fee : ℝ := 15

theorem mandys_data_plan_charge :
  (normal_charge / 3) +  -- First month (promotional rate)
  (normal_charge + extra_fee) +  -- Fourth month (with extra fee)
  (4 * normal_charge) =  -- Other four months
  total_paid := by sorry

end mandys_data_plan_charge_l3837_383731
