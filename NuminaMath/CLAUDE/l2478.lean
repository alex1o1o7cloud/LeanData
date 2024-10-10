import Mathlib

namespace perpendicular_line_equation_l2478_247803

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (given_line : Line) (point : Point) : 
  given_line.slope = 2/3 ∧ given_line.intercept = -2 ∧ point.x = 4 ∧ point.y = 2 →
  ∃ (result_line : Line), 
    result_line.slope = -3/2 ∧ 
    result_line.intercept = 8 ∧
    pointOnLine point result_line ∧
    perpendicular given_line result_line :=
by sorry

end perpendicular_line_equation_l2478_247803


namespace overall_loss_percentage_is_about_2_09_percent_l2478_247871

/-- Represents an appliance with its cost price and profit/loss percentage -/
structure Appliance where
  costPrice : ℕ
  profitLossPercentage : ℤ

/-- Calculates the selling price of an appliance -/
def sellingPrice (a : Appliance) : ℚ :=
  a.costPrice * (1 + a.profitLossPercentage / 100)

/-- The list of appliances with their cost prices and profit/loss percentages -/
def appliances : List Appliance := [
  ⟨15000, -5⟩,
  ⟨8000, 10⟩,
  ⟨12000, -8⟩,
  ⟨10000, 15⟩,
  ⟨5000, 7⟩,
  ⟨20000, -12⟩
]

/-- The total cost price of all appliances -/
def totalCostPrice : ℕ := (appliances.map (·.costPrice)).sum

/-- The total selling price of all appliances -/
def totalSellingPrice : ℚ := (appliances.map sellingPrice).sum

/-- The overall loss percentage -/
def overallLossPercentage : ℚ :=
  (totalCostPrice - totalSellingPrice) / totalCostPrice * 100

/-- Theorem stating that the overall loss percentage is approximately 2.09% -/
theorem overall_loss_percentage_is_about_2_09_percent :
  abs (overallLossPercentage - 2.09) < 0.01 := by sorry

end overall_loss_percentage_is_about_2_09_percent_l2478_247871


namespace car_race_bet_l2478_247864

theorem car_race_bet (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  ∃ w : ℝ, w = 8 / 3 ∧ 
    karen_speed * (w / tom_speed - karen_delay) = w + winning_margin :=
by sorry

end car_race_bet_l2478_247864


namespace even_function_implies_a_equals_one_l2478_247800

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+1)(x-a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x - a)

/-- If f(x) = (x+1)(x-a) is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry


end even_function_implies_a_equals_one_l2478_247800


namespace roots_ratio_implies_k_value_l2478_247840

theorem roots_ratio_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
    r ≠ 0 → s ≠ 0 →
    r^2 + 8*r + k = 0 →
    s^2 + 8*s + k = 0 →
    r / s = 3 →
    k = 12 := by
sorry

end roots_ratio_implies_k_value_l2478_247840


namespace tennis_tournament_player_count_l2478_247854

/-- Represents a valid number of players in a tennis tournament with 2 vs 2 matches -/
def ValidPlayerCount (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 8 * k + 1

/-- Each player plays against every other player exactly once -/
def EachPlayerPlaysAllOthers (n : ℕ) : Prop :=
  (n - 1) % 2 = 0

/-- The total number of games is an integer -/
def TotalGamesInteger (n : ℕ) : Prop :=
  (n * (n - 1)) % 8 = 0

/-- Main theorem: Characterization of valid player counts in the tennis tournament -/
theorem tennis_tournament_player_count (n : ℕ) :
  (EachPlayerPlaysAllOthers n ∧ TotalGamesInteger n) ↔ ValidPlayerCount n :=
sorry

end tennis_tournament_player_count_l2478_247854


namespace henrys_cd_collection_l2478_247896

theorem henrys_cd_collection :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 :=
by
  sorry

end henrys_cd_collection_l2478_247896


namespace white_ball_count_l2478_247821

theorem white_ball_count : ∃ (x y : ℕ), 
  x < y ∧ 
  y < 2 * x ∧ 
  2 * x + 3 * y = 60 ∧ 
  x = 9 ∧ 
  y = 14 := by
  sorry

end white_ball_count_l2478_247821


namespace original_tree_count_l2478_247895

/-- The number of leaves each tree drops during fall. -/
def leaves_per_tree : ℕ := 100

/-- The total number of fallen leaves. -/
def total_fallen_leaves : ℕ := 1400

/-- The current number of trees is twice the original plan. -/
def current_trees_twice_original (original : ℕ) : Prop :=
  2 * original = total_fallen_leaves / leaves_per_tree

/-- Theorem stating the original number of trees the town council intended to plant. -/
theorem original_tree_count : ∃ (original : ℕ), 
  current_trees_twice_original original ∧ original = 7 :=
by
  sorry


end original_tree_count_l2478_247895


namespace triangle_perimeter_21_l2478_247872

-- Define the triangle
def Triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter_21 :
  ∀ c : ℝ,
  Triangle 10 3 c →
  (Perimeter 10 3 c = 18 ∨ Perimeter 10 3 c = 19 ∨ Perimeter 10 3 c = 20 ∨ Perimeter 10 3 c = 21) →
  Perimeter 10 3 c = 21 :=
by
  sorry

#check triangle_perimeter_21

end triangle_perimeter_21_l2478_247872


namespace problem_statement_l2478_247823

theorem problem_statement (x y : ℝ) (m n : ℤ) 
  (h : x > 0) (h' : y > 0) 
  (eq : x^m * y * 4*y^n / (4*x^6*y^4) = 1) : 
  m - n = 3 := by
sorry

end problem_statement_l2478_247823


namespace hyperbola_focal_length_specific_hyperbola_focal_length_l2478_247860

/-- The focal length of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let equation := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

/-- The focal length of the hyperbola x^2/4 - y^2/9 = 1 is 2√13 -/
theorem specific_hyperbola_focal_length :
  let equation := fun (x y : ℝ) => x^2 / 4 - y^2 / 9 = 1
  let focal_length := 2 * Real.sqrt (4 + 9)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

end hyperbola_focal_length_specific_hyperbola_focal_length_l2478_247860


namespace shark_difference_l2478_247889

theorem shark_difference (cape_may_sharks daytona_beach_sharks : ℕ) 
  (h1 : cape_may_sharks = 32) 
  (h2 : daytona_beach_sharks = 12) : 
  cape_may_sharks - 2 * daytona_beach_sharks = 8 := by
  sorry

end shark_difference_l2478_247889


namespace fish_count_proof_l2478_247891

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken released -/
def kens_released : ℕ := 3

/-- The number of fish Ken brought home -/
def kens_brought_home : ℕ := kens_catch - kens_released

/-- The number of fish Kendra brought home (same as caught) -/
def kendras_brought_home : ℕ := kendras_catch

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := kens_brought_home + kendras_brought_home

theorem fish_count_proof : total_brought_home = 87 := by
  sorry

end fish_count_proof_l2478_247891


namespace quadratic_inequality_l2478_247811

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x > 44 ↔ x < -4 ∨ x > 11 := by
  sorry

end quadratic_inequality_l2478_247811


namespace third_root_of_cubic_l2478_247842

theorem third_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (2*d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 76/11) :=
by sorry

end third_root_of_cubic_l2478_247842


namespace average_hamburgers_is_nine_l2478_247866

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 63

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem to prove
theorem average_hamburgers_is_nine : average_hamburgers = 9 := by
  sorry

end average_hamburgers_is_nine_l2478_247866


namespace star_symmetric_zero_l2478_247853

/-- Define the binary operation ⋆ for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For any real numbers x and y, (x-y)² ⋆ (y-x)² = 0 -/
theorem star_symmetric_zero (x y : ℝ) : star ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end star_symmetric_zero_l2478_247853


namespace thermostat_problem_l2478_247809

theorem thermostat_problem (initial_temp : ℝ) (final_temp : ℝ) (x : ℝ) 
  (h1 : initial_temp = 40)
  (h2 : final_temp = 59) : 
  (((initial_temp * 2 - 30) * 0.7) + x = final_temp) → x = 24 := by
  sorry

end thermostat_problem_l2478_247809


namespace journey_speed_calculation_l2478_247847

/-- Proves that given a journey of 1.5 km, if a person arrives 7 minutes late when traveling
    at speed v km/hr, and arrives 8 minutes early when traveling at 6 km/hr, then v = 10 km/hr. -/
theorem journey_speed_calculation (v : ℝ) : 
  (∃ t : ℝ, 
    1.5 = v * (t - 7/60) ∧ 
    1.5 = 6 * (t - 8/60)) → 
  v = 10 := by
sorry

end journey_speed_calculation_l2478_247847


namespace drama_club_organization_l2478_247838

theorem drama_club_organization (participants : ℕ) (girls : ℕ) (boys : ℕ) : 
  participants = girls + boys →
  girls > (85 * participants) / 100 →
  boys ≥ 2 →
  participants ≥ 14 :=
by
  sorry

end drama_club_organization_l2478_247838


namespace savings_account_percentage_l2478_247858

theorem savings_account_percentage (initial_amount : ℝ) (P : ℝ) : 
  initial_amount > 0 →
  (initial_amount + initial_amount * P / 100) * 0.8 = initial_amount →
  P = 25 := by
sorry

end savings_account_percentage_l2478_247858


namespace probability_of_convex_pentagon_l2478_247820

def num_points : ℕ := 7
def num_chords_selected : ℕ := 5

def total_chords (n : ℕ) : ℕ := n.choose 2

def ways_to_select_chords (total : ℕ) (k : ℕ) : ℕ := total.choose k

def convex_pentagons (n : ℕ) : ℕ := n.choose 5

theorem probability_of_convex_pentagon :
  (convex_pentagons num_points : ℚ) / (ways_to_select_chords (total_chords num_points) num_chords_selected) = 1 / 969 :=
sorry

end probability_of_convex_pentagon_l2478_247820


namespace factor_theorem_cubic_l2478_247843

theorem factor_theorem_cubic (a : ℚ) :
  (∀ x, x^3 + 2*x^2 + a*x + 20 = 0 → x = 3) →
  a = -65/3 := by
sorry

end factor_theorem_cubic_l2478_247843


namespace pizza_toppings_l2478_247801

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 24)
  (h_pep : pepperoni_slices = 15)
  (h_mush : mushroom_slices = 16)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 7 := by
  sorry

end pizza_toppings_l2478_247801


namespace sum_of_squares_and_products_l2478_247826

theorem sum_of_squares_and_products (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 58) (h5 : a*b + b*c + c*a = 32) :
  a + b + c = Real.sqrt 122 := by
  sorry

end sum_of_squares_and_products_l2478_247826


namespace exists_small_triangle_area_l2478_247884

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a point to be within the given bounds
def withinBounds (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define the condition for three points to be non-collinear
def nonCollinear (p q r : LatticePoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

-- Calculate the area of a triangle formed by three points
def triangleArea (p q r : LatticePoint) : ℚ :=
  let a := (q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)
  (abs a : ℚ) / 2

-- Main theorem
theorem exists_small_triangle_area 
  (P : Fin 6 → LatticePoint)
  (h_bounds : ∀ i, withinBounds (P i))
  (h_noncollinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (P i) (P j) (P k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (P i) (P j) (P k) ≤ 2 := by
  sorry

end exists_small_triangle_area_l2478_247884


namespace no_integer_solution_l2478_247835

theorem no_integer_solution (n : ℕ+) : ¬ (∃ k : ℤ, (n.val^2 + 1 : ℤ) = k * ((Int.floor (Real.sqrt n.val))^2 + 2)) := by
  sorry

end no_integer_solution_l2478_247835


namespace rational_equation_sum_l2478_247808

theorem rational_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 →
    (B * x - 11) / (x^2 - 7*x + 10) = A / (x - 2) + 3 / (x - 5)) →
  A + B = 5 := by
sorry

end rational_equation_sum_l2478_247808


namespace expansion_coefficients_l2478_247876

/-- The coefficient of x^n in the expansion of (1 + x^5 + x^7)^20 -/
def coeff (n : ℕ) : ℕ :=
  (Finset.range 21).sum (fun k =>
    (Finset.range (21 - k)).sum (fun m =>
      if 5 * k + 7 * m == n && k + m ≤ 20
      then Nat.choose 20 k * Nat.choose (20 - k) m
      else 0))

theorem expansion_coefficients :
  coeff 17 = 3420 ∧ coeff 18 = 0 := by sorry

end expansion_coefficients_l2478_247876


namespace canoe_water_removal_rate_l2478_247806

theorem canoe_water_removal_rate 
  (distance : ℝ) 
  (paddling_speed : ℝ) 
  (water_intake_rate : ℝ) 
  (sinking_threshold : ℝ) 
  (h1 : distance = 2) 
  (h2 : paddling_speed = 3) 
  (h3 : water_intake_rate = 8) 
  (h4 : sinking_threshold = 40) : 
  ∃ (min_removal_rate : ℝ), 
    min_removal_rate = 7 ∧ 
    ∀ (removal_rate : ℝ), 
      removal_rate ≥ min_removal_rate → 
      (water_intake_rate - removal_rate) * (distance / paddling_speed * 60) ≤ sinking_threshold :=
by sorry

end canoe_water_removal_rate_l2478_247806


namespace expression_simplification_and_evaluation_l2478_247870

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = (2 + x) / (2 - x) ∧
  (2 + 0) / (2 - 0) = 1 := by
  sorry

end expression_simplification_and_evaluation_l2478_247870


namespace onion_price_per_pound_l2478_247899

/-- Represents the price and quantity of ingredients --/
structure Ingredient where
  name : String
  quantity : ℝ
  price_per_unit : ℝ

/-- Represents the ratatouille recipe --/
def Recipe : List Ingredient := [
  ⟨"eggplants", 5, 2⟩,
  ⟨"zucchini", 4, 2⟩,
  ⟨"tomatoes", 4, 3.5⟩,
  ⟨"basil", 1, 5⟩  -- Price adjusted for 1 pound
]

def onion_quantity : ℝ := 3
def quart_yield : ℕ := 4
def price_per_quart : ℝ := 10

/-- Calculates the total cost of ingredients excluding onions --/
def total_cost_without_onions : ℝ :=
  Recipe.map (fun i => i.quantity * i.price_per_unit) |>.sum

/-- Calculates the target total cost --/
def target_total_cost : ℝ := quart_yield * price_per_quart

/-- Theorem: The price per pound of onions is $1.00 --/
theorem onion_price_per_pound :
  (target_total_cost - total_cost_without_onions) / onion_quantity = 1 := by
  sorry

end onion_price_per_pound_l2478_247899


namespace goldfinch_percentage_l2478_247883

/-- The number of goldfinches -/
def goldfinches : ℕ := 6

/-- The number of sparrows -/
def sparrows : ℕ := 9

/-- The number of grackles -/
def grackles : ℕ := 5

/-- The total number of birds -/
def total_birds : ℕ := goldfinches + sparrows + grackles

/-- The fraction of goldfinches -/
def goldfinch_fraction : ℚ := goldfinches / total_birds

/-- Theorem: The percentage of goldfinches is 30% -/
theorem goldfinch_percentage :
  goldfinch_fraction * 100 = 30 := by sorry

end goldfinch_percentage_l2478_247883


namespace g_2022_l2478_247827

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    g(x - y) = 2022 * (g x + g y) - 2021 * x * y for all real x and y,
    prove that g(2022) = 2043231 -/
theorem g_2022 (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g (x - y) = 2022 * (g x + g y) - 2021 * x * y) : 
  g 2022 = 2043231 := by
  sorry

end g_2022_l2478_247827


namespace negation_false_implies_proposition_true_l2478_247815

theorem negation_false_implies_proposition_true (P : Prop) : 
  ¬(¬P) → P :=
sorry

end negation_false_implies_proposition_true_l2478_247815


namespace apartment_counts_equation_l2478_247802

/-- Represents the number of apartments of each type in a building -/
structure ApartmentCounts where
  studio : ℝ
  twoPerson : ℝ
  threePerson : ℝ
  fourPerson : ℝ
  fivePerson : ℝ

/-- The apartment complex configuration -/
structure ApartmentComplex where
  buildingCount : ℕ
  maxOccupancy : ℕ
  occupancyRate : ℝ
  studioCapacity : ℝ
  twoPersonCapacity : ℝ
  threePersonCapacity : ℝ
  fourPersonCapacity : ℝ
  fivePersonCapacity : ℝ

/-- Theorem stating the equation for apartment counts given the complex configuration -/
theorem apartment_counts_equation (complex : ApartmentComplex) 
    (counts : ApartmentCounts) : 
    complex.buildingCount = 8 ∧ 
    complex.maxOccupancy = 3000 ∧ 
    complex.occupancyRate = 0.9 ∧
    complex.studioCapacity = 0.95 ∧
    complex.twoPersonCapacity = 0.85 ∧
    complex.threePersonCapacity = 0.8 ∧
    complex.fourPersonCapacity = 0.75 ∧
    complex.fivePersonCapacity = 0.65 →
    0.11875 * counts.studio + 0.2125 * counts.twoPerson + 
    0.3 * counts.threePerson + 0.375 * counts.fourPerson + 
    0.40625 * counts.fivePerson = 337.5 := by
  sorry

end apartment_counts_equation_l2478_247802


namespace rectangle_center_sum_l2478_247892

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- Rectangle is in the first quadrant
  rect.A.1 ≥ 0 ∧ rect.A.2 ≥ 0 ∧
  rect.B.1 ≥ 0 ∧ rect.B.2 ≥ 0 ∧
  rect.C.1 ≥ 0 ∧ rect.C.2 ≥ 0 ∧
  rect.D.1 ≥ 0 ∧ rect.D.2 ≥ 0 ∧
  -- Points on the lines
  (2 : ℝ) ∈ Set.Icc rect.D.1 rect.A.1 ∧
  (6 : ℝ) ∈ Set.Icc rect.C.1 rect.B.1 ∧
  (10 : ℝ) ∈ Set.Icc rect.A.1 rect.B.1 ∧
  (18 : ℝ) ∈ Set.Icc rect.C.1 rect.D.1 ∧
  -- Ratio of AB to BC is 2:1
  2 * (rect.B.1 - rect.C.1) = rect.B.1 - rect.A.1

-- Theorem statement
theorem rectangle_center_sum (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  (rect.A.1 + rect.C.1) / 2 + (rect.A.2 + rect.C.2) / 2 = 10 := by
  sorry

end rectangle_center_sum_l2478_247892


namespace least_number_with_remainders_l2478_247888

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 34 = 4 ∧ 
  n % 5 = 4 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 5 = 4 → n ≤ m :=
by sorry

end least_number_with_remainders_l2478_247888


namespace factor_x12_minus_4096_l2478_247875

theorem factor_x12_minus_4096 (x : ℝ) :
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factor_x12_minus_4096_l2478_247875


namespace complex_sum_equals_i_l2478_247824

theorem complex_sum_equals_i : Complex.I ^ 2 = -1 → (1 : ℂ) + Complex.I + Complex.I ^ 2 = Complex.I := by
  sorry

end complex_sum_equals_i_l2478_247824


namespace gcd_lcm_product_24_36_l2478_247898

theorem gcd_lcm_product_24_36 : 
  (Nat.gcd 24 36) * (Nat.lcm 24 36) = 864 := by
  sorry

end gcd_lcm_product_24_36_l2478_247898


namespace count_integers_satisfying_inequality_l2478_247830

/-- The number of integers satisfying (x - 2)^2 ≤ 4 is 5 -/
theorem count_integers_satisfying_inequality : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end count_integers_satisfying_inequality_l2478_247830


namespace simplest_quadratic_radical_l2478_247893

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a structure for quadratic radicals
structure QuadraticRadical where
  coefficient : ℚ
  radicand : ℕ

-- Define a function to determine if a QuadraticRadical is in its simplest form
def is_simplest_form (qr : QuadraticRadical) : Prop :=
  qr.coefficient ≠ 0 ∧ 
  ¬(is_perfect_square qr.radicand) ∧ 
  is_prime qr.radicand

-- Define the given options
def option_A : QuadraticRadical := ⟨1, 2⟩ -- We represent √(2/3) as √2 / √3
def option_B : QuadraticRadical := ⟨2, 2⟩
def option_C : QuadraticRadical := ⟨1, 24⟩
def option_D : QuadraticRadical := ⟨1, 81⟩

-- Theorem statement
theorem simplest_quadratic_radical :
  is_simplest_form option_B ∧
  ¬(is_simplest_form option_A) ∧
  ¬(is_simplest_form option_C) ∧
  ¬(is_simplest_form option_D) :=
sorry

end simplest_quadratic_radical_l2478_247893


namespace garden_perimeter_is_60_l2478_247833

/-- A rectangular garden with given diagonal and area -/
structure RectangularGarden where
  width : ℝ
  height : ℝ
  diagonal_sq : width^2 + height^2 = 26^2
  area : width * height = 120

/-- The perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.height)

/-- Theorem: The perimeter of the given rectangular garden is 60 meters -/
theorem garden_perimeter_is_60 (g : RectangularGarden) : perimeter g = 60 := by
  sorry

end garden_perimeter_is_60_l2478_247833


namespace inequalities_solution_sets_l2478_247856

def inequality1 (x : ℝ) : Prop := x^2 + 3*x + 2 ≤ 0

def inequality2 (x : ℝ) : Prop := -3*x^2 + 2*x + 2 < 0

def solution_set1 : Set ℝ := {x | -2 ≤ x ∧ x ≤ -1}

def solution_set2 : Set ℝ := {x | x < (1 - Real.sqrt 7) / 3 ∨ x > (1 + Real.sqrt 7) / 3}

theorem inequalities_solution_sets :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) := by sorry

end inequalities_solution_sets_l2478_247856


namespace prime_pair_divisibility_l2478_247832

theorem prime_pair_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q → (p * q ∣ p^p + q^q + 1) ↔ ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end prime_pair_divisibility_l2478_247832


namespace geometric_sequence_a5_l2478_247807

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 10 →
  a 5 = Real.sqrt 10 := by
sorry

end geometric_sequence_a5_l2478_247807


namespace n_must_be_even_l2478_247857

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n_must_be_even (n : ℕ) 
  (h1 : n > 0)
  (h2 : sum_of_digits n = 2014)
  (h3 : sum_of_digits (5 * n) = 1007) :
  Even n := by
  sorry

end n_must_be_even_l2478_247857


namespace min_coefficient_value_l2478_247863

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≤ 15 →
  b ≤ 15 →
  a * b = 30 →
  box = a^2 + b^2 →
  61 ≤ box :=
by sorry

end min_coefficient_value_l2478_247863


namespace arithmetic_sequence_ratio_l2478_247829

/-- An arithmetic sequence with a_5 = 5a_3 has S_9/S_5 = 9 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 5 = 5 * a 3 →  -- given condition
  S 9 / S 5 = 9 := by
sorry

end arithmetic_sequence_ratio_l2478_247829


namespace right_triangle_squares_area_l2478_247885

theorem right_triangle_squares_area (XY YZ XZ : ℝ) :
  XY = 5 →
  XZ = 13 →
  XY^2 + YZ^2 = XZ^2 →
  XY^2 + YZ^2 = 169 :=
by
  sorry

end right_triangle_squares_area_l2478_247885


namespace gary_initial_stickers_l2478_247836

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left -/
def stickers_left : ℕ := 31

/-- The initial number of stickers Gary had -/
def initial_stickers : ℕ := stickers_to_lucy + stickers_to_alex + stickers_left

theorem gary_initial_stickers :
  initial_stickers = 99 :=
by sorry

end gary_initial_stickers_l2478_247836


namespace remainder_of_55_power_55_plus_55_mod_56_l2478_247887

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by sorry

end remainder_of_55_power_55_plus_55_mod_56_l2478_247887


namespace tshirt_price_correct_l2478_247825

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.5

/-- The total number of T-shirts purchased -/
def total_shirts : ℕ := 12

/-- The total cost of the purchase -/
def total_cost : ℝ := 120

/-- The cost of a group of three T-shirts (two at regular price, one at $1) -/
def group_cost (price : ℝ) : ℝ := 2 * price + 1

/-- The number of groups of three T-shirts -/
def num_groups : ℕ := total_shirts / 3

theorem tshirt_price_correct :
  group_cost regular_price * num_groups = total_cost ∧
  regular_price > 0 := by
  sorry

end tshirt_price_correct_l2478_247825


namespace cylinder_volume_change_l2478_247834

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an original volume of 20 cubic feet,
  if its radius is tripled and its height is quadrupled,
  then its new volume will be 720 cubic feet.
-/
theorem cylinder_volume_change (r h : ℝ) :
  (π * r^2 * h = 20) →  -- Original volume is 20 cubic feet
  (π * (3*r)^2 * (4*h) = 720) :=  -- New volume is 720 cubic feet
by sorry

end cylinder_volume_change_l2478_247834


namespace ant_path_problem_l2478_247828

/-- Represents the ant's path in the rectangle -/
structure AntPath where
  rectangle_width : ℝ
  rectangle_height : ℝ
  start_point : ℝ
  path_angle : ℝ

/-- The problem statement -/
theorem ant_path_problem (path : AntPath) :
  path.rectangle_width = 150 ∧
  path.rectangle_height = 18 ∧
  path.path_angle = π / 4 ∧
  path.start_point ≥ 0 ∧
  path.start_point ≤ path.rectangle_height ∧
  (∃ (n : ℕ), 
    path.start_point + n * path.rectangle_height - 2 * n * path.start_point = path.rectangle_width / 2) →
  min path.start_point (path.rectangle_height - path.start_point) = 3 := by
  sorry

end ant_path_problem_l2478_247828


namespace complement_of_B_in_U_l2478_247818

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {2, 5}

-- Theorem statement
theorem complement_of_B_in_U :
  U \ B = {1, 3, 4} := by sorry

end complement_of_B_in_U_l2478_247818


namespace cats_remaining_after_missions_l2478_247869

/-- The number of cats remaining on Tatoosh Island after two relocation missions -/
def cats_remaining (initial : ℕ) (first_relocation : ℕ) : ℕ :=
  let after_first := initial - first_relocation
  let second_relocation := after_first / 2
  after_first - second_relocation

/-- Theorem stating that 600 cats remain on the island after the relocation missions -/
theorem cats_remaining_after_missions :
  cats_remaining 1800 600 = 600 := by
  sorry

end cats_remaining_after_missions_l2478_247869


namespace distance_between_circle_centers_l2478_247852

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem distance_between_circle_centers (DE DF EF : ℝ) (h_DE : DE = 16) (h_DF : DF = 17) (h_EF : EF = 15) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let DI := Real.sqrt (((s - DE) ^ 2) + (r ^ 2))
  let DE' := 3 * DI
  DE' - DI = 10 * Real.sqrt 30 := by
  sorry

end distance_between_circle_centers_l2478_247852


namespace range_of_m_l2478_247855

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, ((x - 4) / 3)^2 > 4 → x^2 - 2*x + 1 - m^2 > 0) →
  (∃ x, ((x - 4) / 3)^2 > 4 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m ≥ 9 :=
sorry

end range_of_m_l2478_247855


namespace points_on_line_y_relation_l2478_247861

/-- Given two points A(1, y₁) and B(-1, y₂) on the line y = -3x + 2, 
    prove that y₁ < y₂ -/
theorem points_on_line_y_relation (y₁ y₂ : ℝ) : 
  (1 : ℝ) > (-1 : ℝ) → -- x₁ > x₂
  y₁ = -3 * (1 : ℝ) + 2 → -- Point A satisfies the line equation
  y₂ = -3 * (-1 : ℝ) + 2 → -- Point B satisfies the line equation
  y₁ < y₂ := by
sorry

end points_on_line_y_relation_l2478_247861


namespace factor_expression_l2478_247804

theorem factor_expression (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end factor_expression_l2478_247804


namespace integral_x_squared_l2478_247882

theorem integral_x_squared : ∫ x in (-1)..1, x^2 = 2/3 := by sorry

end integral_x_squared_l2478_247882


namespace remaining_gift_cards_value_l2478_247873

/-- Represents the types of gift cards --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents a gift card with its type and value --/
structure GiftCard where
  type : GiftCardType
  value : Nat

def initial_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 },
  { type := GiftCardType.Amazon, value := 1000 }
]

def sent_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 }
]

theorem remaining_gift_cards_value : 
  (List.sum (initial_gift_cards.map (λ g => g.value)) - 
   List.sum (sent_gift_cards.map (λ g => g.value))) = 4250 := by
  sorry

end remaining_gift_cards_value_l2478_247873


namespace distribution_problem_l2478_247831

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups,
    where two specific objects cannot be in the same group -/
def distributeWithRestriction (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 114 ways to distribute 5 distinct objects
    into 3 non-empty groups, where two specific objects cannot be in the same group -/
theorem distribution_problem : distributeWithRestriction 5 3 = 114 := by sorry

end distribution_problem_l2478_247831


namespace land_development_profit_l2478_247850

theorem land_development_profit (cost_per_acre : ℝ) (sale_price_per_acre : ℝ) (profit : ℝ) (acres : ℝ) : 
  cost_per_acre = 70 →
  sale_price_per_acre = 200 →
  profit = 6000 →
  sale_price_per_acre * (acres / 2) - cost_per_acre * acres = profit →
  acres = 200 := by
sorry

end land_development_profit_l2478_247850


namespace find_m_l2478_247862

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

theorem find_m : 
  ∃ (m : ℝ), (U \ A m = {1, 2}) → m = -3 := by
  sorry

end find_m_l2478_247862


namespace hyperbola_parabola_intersection_l2478_247817

/-- The value of p for which the left focus of the hyperbola 
    x²/3 - 16y²/p² = 1 (p > 0) lies on the latus rectum of 
    the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2/3 - 16*y^2/p^2 = 1) → 
  (∃ x y : ℝ, y^2 = 2*p*x) → 
  (∃ x : ℝ, x^2/3 - 16*0^2/p^2 = 1 ∧ 0^2 = 2*p*x) → 
  p = 4 := by sorry

end hyperbola_parabola_intersection_l2478_247817


namespace complex_equation_solution_l2478_247819

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end complex_equation_solution_l2478_247819


namespace advertising_department_size_l2478_247880

/-- Proves that given a company with 1000 total employees, using stratified sampling
    to draw 80 employees, if 4 employees are sampled from the advertising department,
    then the number of employees in the advertising department is 50. -/
theorem advertising_department_size
  (total_employees : ℕ)
  (sample_size : ℕ)
  (sampled_from_advertising : ℕ)
  (h_total : total_employees = 1000)
  (h_sample : sample_size = 80)
  (h_ad_sample : sampled_from_advertising = 4)
  : (sampled_from_advertising : ℚ) / sample_size * total_employees = 50 := by
  sorry

end advertising_department_size_l2478_247880


namespace least_sum_of_four_primes_l2478_247839

-- Define a function that checks if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function that represents the sum of 4 different primes greater than 10
def sumOfFourPrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  a > 10 ∧ b > 10 ∧ c > 10 ∧ d > 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Theorem statement
theorem least_sum_of_four_primes :
  ∀ n : ℕ, (∃ a b c d : ℕ, sumOfFourPrimes a b c d ∧ a + b + c + d = n) →
  n ≥ 60 :=
sorry

end least_sum_of_four_primes_l2478_247839


namespace cube_number_placement_impossible_l2478_247841

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → ℕ)

/-- Predicate to check if two vertices are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop := sorry

/-- The theorem stating the impossibility of the number placement on a cube -/
theorem cube_number_placement_impossible :
  ¬ ∃ (c : Cube),
    (∀ i : Fin 8, 1 ≤ c.vertices i ∧ c.vertices i ≤ 220) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, adjacent i j → ∃ (d : ℕ), d > 1 ∧ d ∣ c.vertices i ∧ d ∣ c.vertices j) ∧
    (∀ i j : Fin 8, ¬adjacent i j → ∀ (d : ℕ), d > 1 → ¬(d ∣ c.vertices i ∧ d ∣ c.vertices j)) :=
sorry

end cube_number_placement_impossible_l2478_247841


namespace set_operations_and_inclusion_l2478_247865

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x - a - 1 < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem set_operations_and_inclusion (a : ℝ) : 
  (Set.compl A ∪ Set.compl B = {x | x ≤ 3 ∨ x ≥ 6}) ∧
  (B ⊆ C a ↔ a ≥ 8) := by sorry

end set_operations_and_inclusion_l2478_247865


namespace trains_meet_time_trains_meet_time_approx_l2478_247867

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
theorem trains_meet_time (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := initial_distance + length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  total_distance / relative_speed

/-- The time for two trains to meet is approximately 6.69 seconds. -/
theorem trains_meet_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |trains_meet_time 90 95 250 64 92 - 6.69| < ε :=
sorry

end trains_meet_time_trains_meet_time_approx_l2478_247867


namespace polynomial_simplification_l2478_247868

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 3 * x^4 - 5 * x^3 + 2 * x^2 - 10 * x + 8) + 
  (-3 * x^5 - x^4 + 4 * x^3 - 2 * x^2 + 15 * x - 12) = 
  -x^5 + 2 * x^4 - x^3 + 5 * x - 4 := by
sorry

end polynomial_simplification_l2478_247868


namespace system_solution_l2478_247886

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 = -x + 3*y + z ∧ 
  y^2 + z^2 = x + 3*y - z ∧ 
  x^2 + z^2 = 2*x + 2*y - z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end system_solution_l2478_247886


namespace cubic_inequality_l2478_247890

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end cubic_inequality_l2478_247890


namespace min_values_ab_and_a_plus_2b_l2478_247877

theorem min_values_ab_and_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 8 ∧ a + 2 * b = 9 := by
sorry

end min_values_ab_and_a_plus_2b_l2478_247877


namespace percentage_relation_l2478_247805

/-- Given three real numbers A, B, and C, where A is 6% of C and 20% of B,
    prove that B is 30% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.30 * C := by
  sorry

end percentage_relation_l2478_247805


namespace partial_fraction_A_value_l2478_247874

-- Define the polynomial in the denominator
def p (x : ℝ) : ℝ := x^4 - 2*x^3 - 29*x^2 + 70*x + 120

-- Define the partial fraction decomposition
def partial_fraction (x A B C D : ℝ) : Prop :=
  1 / p x = A / (x + 4) + B / (x - 2) + C / (x - 2)^2 + D / (x - 3)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction x A B C D) → A = -1/252 :=
by
  sorry

end partial_fraction_A_value_l2478_247874


namespace no_solution_implies_a_zero_l2478_247849

/-- A system of equations with no solutions implies a = 0 -/
theorem no_solution_implies_a_zero 
  (h : ∀ (x y : ℝ), (y^2 = x^2 + a*x + b ∧ x^2 = y^2 + a*y + b) → False) :
  a = 0 :=
by sorry

end no_solution_implies_a_zero_l2478_247849


namespace problem_pyramid_volume_l2478_247816

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  height : ℝ
  base1_sides : Fin 3 → ℝ
  base2_perimeter : ℝ

/-- Calculates the volume of a truncated triangular pyramid -/
def volume (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The specific truncated pyramid from the problem -/
def problem_pyramid : TruncatedPyramid :=
  { height := 10
  , base1_sides := ![27, 29, 52]
  , base2_perimeter := 72 }

/-- Theorem stating that the volume of the problem pyramid is 1900 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = 1900 := by sorry

end problem_pyramid_volume_l2478_247816


namespace max_volume_cylinder_in_sphere_l2478_247822

noncomputable section

theorem max_volume_cylinder_in_sphere (R : ℝ) (h r : ℝ → ℝ) :
  (∀ t, 4 * R^2 = 4 * (r t)^2 + (h t)^2) →
  (∀ t, (r t) ≥ 0 ∧ (h t) ≥ 0) →
  (∃ t₀, ∀ t, π * (r t)^2 * (h t) ≤ π * (r t₀)^2 * (h t₀)) →
  h t₀ = 2 * R / Real.sqrt 3 ∧ r t₀ = R * Real.sqrt (2/3) :=
by sorry

end

end max_volume_cylinder_in_sphere_l2478_247822


namespace possible_value_less_than_five_l2478_247859

theorem possible_value_less_than_five : ∃ x : ℝ, x < 5 ∧ x = 0 := by
  sorry

end possible_value_less_than_five_l2478_247859


namespace lukes_final_balance_l2478_247881

/-- Calculates Luke's final balance after six months of financial activities --/
def lukesFinalBalance (initialAmount : ℝ) (februarySpendingRate : ℝ) 
  (marchSpending marchIncome : ℝ) (monthlyPiggyBankRate : ℝ) : ℝ :=
  let afterFebruary := initialAmount * (1 - februarySpendingRate)
  let afterMarch := afterFebruary - marchSpending + marchIncome
  let afterApril := afterMarch * (1 - monthlyPiggyBankRate)
  let afterMay := afterApril * (1 - monthlyPiggyBankRate)
  let afterJune := afterMay * (1 - monthlyPiggyBankRate)
  afterJune

/-- Theorem stating Luke's final balance after six months --/
theorem lukes_final_balance :
  lukesFinalBalance 48 0.3 11 21 0.1 = 31.79 := by
  sorry

end lukes_final_balance_l2478_247881


namespace complement_A_intersect_B_l2478_247897

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | -2 < x ∧ x < 1} := by sorry

end complement_A_intersect_B_l2478_247897


namespace distance_between_points_l2478_247894

theorem distance_between_points (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let angleABC := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  AB = 20 ∧ BC = 30 ∧ angleABC = 2 * Real.pi / 3 →
  AC = 10 * Real.sqrt 19 := by
  sorry

end distance_between_points_l2478_247894


namespace running_is_experimental_l2478_247810

/-- Represents an investigation method -/
inductive InvestigationMethod
  | Experimental
  | NonExperimental

/-- Represents the characteristics of an investigation -/
structure Investigation where
  description : String
  quantitative : Bool
  directlyMeasurable : Bool
  controlledSetting : Bool

/-- Determines if an investigation is suitable for the experimental method -/
def isSuitableForExperiment (i : Investigation) : InvestigationMethod :=
  if i.quantitative && i.directlyMeasurable && i.controlledSetting then
    InvestigationMethod.Experimental
  else
    InvestigationMethod.NonExperimental

/-- The investigation of running distance in 10 seconds -/
def runningInvestigation : Investigation where
  description := "How many meters you can run in 10 seconds"
  quantitative := true
  directlyMeasurable := true
  controlledSetting := true

/-- Theorem stating that the running investigation is suitable for the experimental method -/
theorem running_is_experimental :
  isSuitableForExperiment runningInvestigation = InvestigationMethod.Experimental := by
  sorry


end running_is_experimental_l2478_247810


namespace circle_equation_l2478_247812

/-- A circle with center on the line y = x passing through (-1, 1) and (1, 3) has the equation (x-1)^2 + (y-1)^2 = 4 -/
theorem circle_equation : ∀ (a : ℝ),
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a + 1)^2 + (a - 1)^2) →
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a - 1)^2 + (a - 3)^2) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4) :=
by sorry

end circle_equation_l2478_247812


namespace midpoint_of_complex_line_segment_l2478_247844

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 9*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 - 2*I := by sorry

end midpoint_of_complex_line_segment_l2478_247844


namespace circle_radius_l2478_247878

/-- Given a circle with equation x^2 + y^2 - 4x - 2y - 5 = 0, its radius is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 10 :=
by sorry

end circle_radius_l2478_247878


namespace sin_double_angle_tangent_two_l2478_247846

theorem sin_double_angle_tangent_two (α : Real) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end sin_double_angle_tangent_two_l2478_247846


namespace floor_inequality_iff_equal_l2478_247848

theorem floor_inequality_iff_equal (m n : ℕ+) :
  (∀ α β : ℝ, ⌊(m + n : ℝ) * α⌋ + ⌊(m + n : ℝ) * β⌋ ≥ ⌊(m : ℝ) * α⌋ + ⌊(m : ℝ) * β⌋ + ⌊(n : ℝ) * (α + β)⌋) ↔
  m = n :=
by sorry

end floor_inequality_iff_equal_l2478_247848


namespace calculator_sum_theorem_l2478_247837

/-- The number of participants in the game --/
def num_participants : ℕ := 47

/-- The initial value of calculator A --/
def initial_A : ℤ := 2

/-- The initial value of calculator B --/
def initial_B : ℕ := 0

/-- The initial value of calculator C --/
def initial_C : ℤ := -1

/-- The initial value of calculator D --/
def initial_D : ℕ := 3

/-- The final value of calculator A after all participants have processed it --/
def final_A : ℤ := -initial_A

/-- The final value of calculator B after all participants have processed it --/
def final_B : ℕ := initial_B

/-- The final value of calculator C after all participants have processed it --/
def final_C : ℤ := -initial_C

/-- The final value of calculator D after all participants have processed it --/
noncomputable def final_D : ℕ := initial_D ^ (3 ^ num_participants)

/-- The theorem stating that the sum of the final calculator values equals 3^(3^47) - 3 --/
theorem calculator_sum_theorem :
  final_A + final_B + final_C + final_D = 3^(3^47) - 3 := by
  sorry


end calculator_sum_theorem_l2478_247837


namespace first_day_sale_l2478_247814

theorem first_day_sale (total_days : ℕ) (average_sale : ℕ) (known_days_sales : List ℕ) :
  total_days = 5 →
  average_sale = 625 →
  known_days_sales = [927, 855, 230, 562] →
  (total_days * average_sale) - known_days_sales.sum = 551 := by
  sorry

end first_day_sale_l2478_247814


namespace sum_angles_S_and_R_l2478_247845

-- Define the circle and points
variable (circle : Type) (E F R G H : circle)

-- Define the measure of an arc
variable (arc_measure : circle → circle → ℝ)

-- Define the measure of an angle
variable (angle_measure : circle → ℝ)

-- State the theorem
theorem sum_angles_S_and_R (h1 : arc_measure F R = 60)
                           (h2 : arc_measure R G = 48) :
  angle_measure S + angle_measure R = 54 := by
  sorry

end sum_angles_S_and_R_l2478_247845


namespace maggies_age_l2478_247851

theorem maggies_age (kate_age sue_age maggie_age : ℕ) 
  (total_age : kate_age + sue_age + maggie_age = 48)
  (kate : kate_age = 19)
  (sue : sue_age = 12) :
  maggie_age = 17 := by
  sorry

end maggies_age_l2478_247851


namespace carA_distance_at_2016th_meeting_l2478_247813

/-- Represents a car with its current speed and direction -/
structure Car where
  speed : ℝ
  direction : Bool

/-- Represents the state of the system at any given time -/
structure State where
  carA : Car
  carB : Car
  positionA : ℝ
  positionB : ℝ
  meetingCount : ℕ
  distanceTraveledA : ℝ

/-- The distance between points A and B -/
def distance : ℝ := 900

/-- Function to update the state after each meeting -/
def updateState (s : State) : State :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the total distance traveled by Car A at the 2016th meeting -/
theorem carA_distance_at_2016th_meeting :
  ∃ (finalState : State),
    finalState.meetingCount = 2016 ∧
    finalState.distanceTraveledA = 1813900 :=
by
  sorry

end carA_distance_at_2016th_meeting_l2478_247813


namespace isosceles_right_triangle_quotient_isosceles_right_triangle_max_quotient_l2478_247879

/-- For an isosceles right triangle with legs of length a, 
    the value of 2a / √(a^2 + a^2) is equal to √2 -/
theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  2 * a / Real.sqrt (a^2 + a^2) = Real.sqrt 2 := by
  sorry

/-- The maximum quotient (a + b) / c for an isosceles right triangle 
    with legs of length a is √2 -/
theorem isosceles_right_triangle_max_quotient (a : ℝ) (h : a > 0) :
  (a + a) / Real.sqrt (2 * a^2) = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_quotient_isosceles_right_triangle_max_quotient_l2478_247879
