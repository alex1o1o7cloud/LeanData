import Mathlib

namespace daughter_normal_probability_l2590_259064

-- Define the inheritance types
inductive InheritanceType
| XLinked
| Autosomal

-- Define the phenotypes
inductive Phenotype
| Normal
| Affected

-- Define the genotypes
structure Genotype where
  hemophilia : Bool  -- true if carrier
  phenylketonuria : Bool  -- true if carrier

-- Define the parents
structure Parents where
  mother : Genotype
  father : Genotype

-- Define the conditions
def conditions (parents : Parents) : Prop :=
  (InheritanceType.XLinked = InheritanceType.XLinked) ∧  -- Hemophilia is X-linked
  (InheritanceType.Autosomal = InheritanceType.Autosomal) ∧  -- Phenylketonuria is autosomal
  (parents.mother.hemophilia = true) ∧  -- Mother is carrier for hemophilia
  (parents.father.hemophilia = false) ∧  -- Father is not affected by hemophilia
  (parents.mother.phenylketonuria = true) ∧  -- Mother is carrier for phenylketonuria
  (parents.father.phenylketonuria = true)  -- Father is carrier for phenylketonuria

-- Define the probability of a daughter being phenotypically normal
def prob_normal_daughter (parents : Parents) : ℚ :=
  3 / 4

-- The theorem to prove
theorem daughter_normal_probability (parents : Parents) :
  conditions parents → prob_normal_daughter parents = 3 / 4 :=
by sorry

end daughter_normal_probability_l2590_259064


namespace percent_change_condition_l2590_259034

theorem percent_change_condition (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hN : N > 0) (hq_bound : q < 50) :
  N * (1 + 3 * p / 100) * (1 - 2 * q / 100) > N ↔ p > 100 * q / (147 - 3 * q) :=
sorry

end percent_change_condition_l2590_259034


namespace rooks_arrangement_count_l2590_259081

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of squares threatened by a rook (excluding its own square) -/
def squaresThreatened : ℕ := 14

/-- The number of ways to arrange two rooks on a chessboard such that they cannot capture each other -/
def rooksArrangements : ℕ := chessboardSquares * (chessboardSquares - squaresThreatened - 1)

theorem rooks_arrangement_count :
  rooksArrangements = 3136 := by sorry

end rooks_arrangement_count_l2590_259081


namespace book_club_unique_books_book_club_unique_books_eq_61_l2590_259038

theorem book_club_unique_books : ℕ :=
  let tony_books : ℕ := 23
  let dean_books : ℕ := 20
  let breanna_books : ℕ := 30
  let piper_books : ℕ := 26
  let asher_books : ℕ := 25
  let tony_dean_shared : ℕ := 5
  let breanna_piper_asher_shared : ℕ := 7
  let dean_piper_shared : ℕ := 6
  let dean_piper_tony_shared : ℕ := 3
  let asher_breanna_tony_shared : ℕ := 8
  let all_shared : ℕ := 2
  let breanna_piper_shared : ℕ := 9
  let breanna_piper_dean_shared : ℕ := 4
  let breanna_piper_asher_shared : ℕ := 2

  let total_books : ℕ := tony_books + dean_books + breanna_books + piper_books + asher_books
  let overlaps : ℕ := tony_dean_shared + 
                      2 * breanna_piper_asher_shared + 
                      2 * dean_piper_tony_shared + 
                      (dean_piper_shared - dean_piper_tony_shared) + 
                      2 * (asher_breanna_tony_shared - all_shared) + 
                      4 * all_shared + 
                      (breanna_piper_shared - breanna_piper_dean_shared - breanna_piper_asher_shared) + 
                      2 * breanna_piper_dean_shared + 
                      breanna_piper_asher_shared

  total_books - overlaps
  
theorem book_club_unique_books_eq_61 : book_club_unique_books = 61 := by
  sorry

end book_club_unique_books_book_club_unique_books_eq_61_l2590_259038


namespace van_tire_usage_l2590_259001

/-- Represents the number of miles each tire is used in a van with a tire rotation system -/
def miles_per_tire (total_miles : ℕ) (total_tires : ℕ) (simultaneous_tires : ℕ) : ℕ :=
  (total_miles * simultaneous_tires) / total_tires

/-- Theorem stating that in a van with 6 tires, where 4 are used simultaneously,
    traveling 40,000 miles results in each tire being used for approximately 26,667 miles -/
theorem van_tire_usage :
  miles_per_tire 40000 6 4 = 26667 := by
  sorry

end van_tire_usage_l2590_259001


namespace simplify_sqrt_expression_l2590_259032

theorem simplify_sqrt_expression :
  (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 4 * Real.sqrt (1/2) = 4 * Real.sqrt 3 + Real.sqrt 2 :=
by sorry

end simplify_sqrt_expression_l2590_259032


namespace nth_equation_holds_l2590_259053

theorem nth_equation_holds (n : ℕ) :
  (n : ℚ) / (n + 2) * (1 - 1 / (n + 1)) = n^2 / ((n + 1) * (n + 2)) := by
  sorry

end nth_equation_holds_l2590_259053


namespace lucky_larry_problem_l2590_259028

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  a - b - c - d + e = a - (b - (c - (d + e))) →
  e / 2 = 2 := by
  sorry

end lucky_larry_problem_l2590_259028


namespace optimal_tax_and_revenue_correct_l2590_259045

/-- Market model with linear supply and demand functions -/
structure MarketModel where
  -- Supply function coefficients
  supply_slope : ℝ
  supply_intercept : ℝ
  -- Demand function coefficient (slope)
  demand_slope : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax rate
  tax_rate : ℝ
  -- Consumer price after tax
  consumer_price : ℝ

/-- Calculate the optimal tax rate and maximum tax revenue -/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  -- Placeholder for the actual calculation
  (60, 8640)

/-- Theorem stating the optimal tax rate and maximum tax revenue -/
theorem optimal_tax_and_revenue_correct (model : MarketModel) :
  model.supply_slope = 6 ∧
  model.supply_intercept = -312 ∧
  model.demand_slope = -4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax_rate = 30 ∧
  model.consumer_price = 118 →
  optimal_tax_and_revenue model = (60, 8640) := by
  sorry

end optimal_tax_and_revenue_correct_l2590_259045


namespace sqrt_product_equality_l2590_259069

theorem sqrt_product_equality : 3 * Real.sqrt 2 * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2590_259069


namespace combinatorial_identity_l2590_259092

theorem combinatorial_identity 
  (n k m : ℕ) 
  (h1 : 1 ≤ k) 
  (h2 : k < m) 
  (h3 : m ≤ n) : 
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose k i * Nat.choose n (m - i))) = 
  Nat.choose (n + k) m := by
  sorry

end combinatorial_identity_l2590_259092


namespace complement_P_intersect_Q_l2590_259089

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_P_intersect_Q : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end complement_P_intersect_Q_l2590_259089


namespace dinner_bill_split_l2590_259083

theorem dinner_bill_split (total_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (tax_percent : ℝ) :
  total_bill = 425 →
  num_people = 15 →
  tip_percent = 0.18 →
  tax_percent = 0.08 →
  (total_bill * (1 + tip_percent + tax_percent)) / num_people = 35.70 := by
  sorry

end dinner_bill_split_l2590_259083


namespace triangle_inequality_l2590_259003

theorem triangle_inequality (R r p : ℝ) (a b c m_a m_b m_c : ℝ) 
  (h1 : R * r = a * b * c / (4 * p))
  (h2 : a * b * c ≤ 8 * p^3)
  (h3 : p^2 ≤ (m_a^2 + m_b^2 + m_c^2) / 4)
  (h4 : m_a^2 + m_b^2 + m_c^2 ≤ 27 * R^2 / 4) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 := by
  sorry

end triangle_inequality_l2590_259003


namespace even_quadratic_function_range_l2590_259018

/-- A quadratic function that is even -/
def EvenQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * x^2 + c

theorem even_quadratic_function_range
  (f : ℝ → ℝ)
  (hf : EvenQuadraticFunction f)
  (h1 : 1 ≤ f 1 ∧ f 1 ≤ 2)
  (h2 : 3 ≤ f 2 ∧ f 2 ≤ 4) :
  14/3 ≤ f 3 ∧ f 3 ≤ 9 := by
sorry

end even_quadratic_function_range_l2590_259018


namespace line_intersection_range_l2590_259029

/-- Given a line y = 2x + (3-a) intersecting the x-axis between points (3,0) and (4,0) inclusive, 
    the range of values for a is 9 ≤ a ≤ 11. -/
theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 4 ∧ 0 = 2*x + (3-a)) → 
  (9 ≤ a ∧ a ≤ 11) := by
sorry

end line_intersection_range_l2590_259029


namespace first_day_is_thursday_l2590_259060

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Function to determine the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Theorem stating that in a month with 31 days, 5 Saturdays, and 4 Sundays, 
    the first day is Thursday -/
theorem first_day_is_thursday :
  ∀ (m : Month), m.days = 31 → m.saturdays = 5 → m.sundays = 4 →
  firstDayOfMonth m = DayOfWeek.Thursday :=
  sorry

end first_day_is_thursday_l2590_259060


namespace triangle_area_implies_angle_l2590_259059

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = (a^2 + b^2 - c^2) / 4, then the measure of angle C is π/4. -/
theorem triangle_area_implies_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / 4 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :
  Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = π/4 := by
  sorry

end triangle_area_implies_angle_l2590_259059


namespace count_unique_polygonal_chains_l2590_259031

/-- The number of unique closed 2n-segment polygonal chains on an n x n grid -/
def uniquePolygonalChains (n : ℕ) : ℕ :=
  (n.factorial * (n - 1).factorial) / 2

/-- Theorem stating the number of unique closed 2n-segment polygonal chains
    that can be drawn on an n x n grid, passing through all horizontal and
    vertical lines exactly once -/
theorem count_unique_polygonal_chains (n : ℕ) (h : n > 0) :
  uniquePolygonalChains n = (n.factorial * (n - 1).factorial) / 2 := by
  sorry

end count_unique_polygonal_chains_l2590_259031


namespace erica_ice_cream_weeks_l2590_259078

/-- The number of weeks Erica buys ice cream -/
def ice_cream_weeks (orange_creamsicle_price : ℚ) 
                    (ice_cream_sandwich_price : ℚ)
                    (nutty_buddy_price : ℚ)
                    (total_spent : ℚ) : ℚ :=
  let weekly_spending := 3 * orange_creamsicle_price + 
                         2 * ice_cream_sandwich_price + 
                         2 * nutty_buddy_price
  total_spent / weekly_spending

/-- Theorem stating that Erica buys ice cream for 6 weeks -/
theorem erica_ice_cream_weeks : 
  ice_cream_weeks 2 1.5 3 90 = 6 := by
  sorry

end erica_ice_cream_weeks_l2590_259078


namespace beach_problem_l2590_259057

/-- The number of people originally in the first row of the beach. -/
def first_row : ℕ := sorry

/-- The number of people originally in the second row of the beach. -/
def second_row : ℕ := 20

/-- The number of people in the third row of the beach. -/
def third_row : ℕ := 18

/-- The number of people who left the first row to wade in the water. -/
def left_first_row : ℕ := 3

/-- The number of people who left the second row to wade in the water. -/
def left_second_row : ℕ := 5

/-- The total number of people left relaxing on the beach. -/
def total_remaining : ℕ := 54

theorem beach_problem :
  first_row = 24 :=
by
  sorry

end beach_problem_l2590_259057


namespace flower_garden_area_proof_l2590_259040

/-- The area of a circular flower garden -/
def flower_garden_area (radius : ℝ) (pi : ℝ) : ℝ :=
  pi * radius ^ 2

/-- Proof that the area of a circular flower garden with radius 0.6 meters is 1.08 square meters, given that π is assumed to be 3 -/
theorem flower_garden_area_proof :
  let radius : ℝ := 0.6
  let pi : ℝ := 3
  flower_garden_area radius pi = 1.08 := by
  sorry

end flower_garden_area_proof_l2590_259040


namespace unique_solution_at_85_l2590_259075

/-- Represents the American High School Mathematics Examination (AHSME) -/
structure AHSME where
  total_questions : Nat
  score_formula : (correct : Nat) → (wrong : Nat) → Int

/-- Defines the specific AHSME instance -/
def ahsme : AHSME :=
  { total_questions := 30
  , score_formula := λ c w => 30 + 4 * c - w }

/-- Theorem stating the uniqueness of the solution for a score of 85 -/
theorem unique_solution_at_85 (exam : AHSME := ahsme) :
  ∃! (c w : Nat), c + w ≤ exam.total_questions ∧
                  exam.score_formula c w = 85 ∧
                  (∀ s, 85 > s → s > 85 - 4 →
                    ¬∃! (c' w' : Nat), c' + w' ≤ exam.total_questions ∧
                                      exam.score_formula c' w' = s) :=
by sorry

end unique_solution_at_85_l2590_259075


namespace always_two_real_roots_integer_roots_condition_l2590_259044

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - a*x + (a - 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (a : ℝ) :
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x = 0 ∧ quadratic_equation a y = 0 :=
sorry

-- Theorem 2: When roots are integers and one is twice the other, a = 3
theorem integer_roots_condition (a : ℝ) :
  (∃ x y : ℤ, x ≠ y ∧ quadratic_equation a (x : ℝ) = 0 ∧ quadratic_equation a (y : ℝ) = 0 ∧ y = 2*x) →
  a = 3 :=
sorry

end always_two_real_roots_integer_roots_condition_l2590_259044


namespace evaluate_expression_l2590_259067

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end evaluate_expression_l2590_259067


namespace binary_110101_to_base7_l2590_259004

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-7 representation (as a list of digits). -/
def nat_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_110101_to_base7 :
  nat_to_base7 (binary_to_nat [true, false, true, false, true, true]) = [1, 0, 4] := by
  sorry

end binary_110101_to_base7_l2590_259004


namespace tabithas_initial_money_l2590_259098

theorem tabithas_initial_money :
  ∀ (initial : ℚ) 
    (given_to_mom : ℚ) 
    (num_items : ℕ) 
    (item_cost : ℚ) 
    (money_left : ℚ),
  given_to_mom = 8 →
  num_items = 5 →
  item_cost = 1/2 →
  money_left = 6 →
  initial - given_to_mom = 2 * ((initial - given_to_mom) / 2 - num_items * item_cost - money_left) →
  initial = 25 := by
sorry

end tabithas_initial_money_l2590_259098


namespace cheryl_leftover_material_l2590_259054

theorem cheryl_leftover_material :
  let material_type1 : ℚ := 2/9
  let material_type2 : ℚ := 1/8
  let total_bought : ℚ := material_type1 + material_type2
  let material_used : ℚ := 0.125
  let material_leftover : ℚ := total_bought - material_used
  material_leftover = 2/9 := by
  sorry

end cheryl_leftover_material_l2590_259054


namespace anika_age_l2590_259030

/-- Given the ages of Ben, Clara, and Anika, prove that Anika is 15 years old. -/
theorem anika_age (ben_age clara_age anika_age : ℕ) 
  (h1 : clara_age = ben_age + 5)
  (h2 : anika_age = clara_age - 10)
  (h3 : ben_age = 20) : 
  anika_age = 15 := by
  sorry

end anika_age_l2590_259030


namespace square_circles_intersection_area_l2590_259070

/-- The area of intersection between a square and four circles --/
theorem square_circles_intersection_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 3
  let square_area : ℝ := square_side ^ 2
  let circle_sector_area : ℝ := π * circle_radius ^ 2 / 4
  let total_sector_area : ℝ := 4 * circle_sector_area
  let triangle_area : ℝ := (square_side / 2 - circle_radius) ^ 2 / 2
  let total_triangle_area : ℝ := 4 * triangle_area
  let shaded_area : ℝ := square_area - (total_sector_area + total_triangle_area)
  shaded_area = 64 - 9 * π - 18 :=
by sorry

end square_circles_intersection_area_l2590_259070


namespace intersection_of_lines_l2590_259036

/-- Given two lines m and n that intersect at (2, 7), 
    where m has equation y = 2x + 3 and n has equation y = kx + 1,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = k*x + 1 → x = 2 ∧ y = 7) → 
  k = 3 := by
  sorry

end intersection_of_lines_l2590_259036


namespace sum_of_cubes_of_roots_l2590_259009

theorem sum_of_cubes_of_roots (r s t : ℝ) : 
  (r - (27 : ℝ)^(1/3 : ℝ)) * (r - (64 : ℝ)^(1/3 : ℝ)) * (r - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (s - (27 : ℝ)^(1/3 : ℝ)) * (s - (64 : ℝ)^(1/3 : ℝ)) * (s - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (t - (27 : ℝ)^(1/3 : ℝ)) * (t - (64 : ℝ)^(1/3 : ℝ)) * (t - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 214.5 := by
sorry

end sum_of_cubes_of_roots_l2590_259009


namespace thirty_blocks_placeable_l2590_259012

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a rectangular block -/
structure Block :=
  (length : Nat)
  (width : Nat)

/-- Calculates the number of blocks that can be placed on the modified chessboard -/
def countPlaceableBlocks (board : ModifiedChessboard) (block : Block) : Nat :=
  sorry

/-- Theorem stating that 30 blocks can be placed on the modified 8x8 chessboard -/
theorem thirty_blocks_placeable :
  ∀ (board : ModifiedChessboard) (block : Block),
    board.size = 8 ∧ 
    board.cornersRemoved = 2 ∧ 
    block.length = 2 ∧ 
    block.width = 1 →
    countPlaceableBlocks board block = 30 :=
  sorry

end thirty_blocks_placeable_l2590_259012


namespace fraction_value_at_two_l2590_259049

theorem fraction_value_at_two :
  let f (x : ℝ) := (x^10 + 20*x^5 + 100) / (x^5 + 10)
  f 2 = 42 := by
  sorry

end fraction_value_at_two_l2590_259049


namespace quadratic_expression_values_l2590_259033

theorem quadratic_expression_values (a c : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + c = 10 → x = 1) →
  (∀ x : ℝ, a * x^2 + x + c = 8 → x = -1) :=
by sorry

end quadratic_expression_values_l2590_259033


namespace range_of_a_l2590_259065

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 5 else a^x + 2*a + 2

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, y ≥ 3 → ∃ x : ℝ, f a x = y) ∧ 
  (∀ x : ℝ, f a x ≥ 3) →
  a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi 1 :=
sorry

end range_of_a_l2590_259065


namespace unique_modular_congruence_l2590_259050

theorem unique_modular_congruence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -215 ≡ n [ZMOD 23] ∧ n = 15 := by
  sorry

end unique_modular_congruence_l2590_259050


namespace rectangle_perimeter_l2590_259058

theorem rectangle_perimeter (length width : ℕ+) : 
  length * width = 24 → 2 * (length + width) ≠ 36 := by
  sorry

end rectangle_perimeter_l2590_259058


namespace b_four_lt_b_seven_l2590_259010

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (b α n + 1 / α (n + 1))

theorem b_four_lt_b_seven (α : ℕ → ℕ) : b α 4 < b α 7 := by
  sorry

end b_four_lt_b_seven_l2590_259010


namespace distance_covered_l2590_259002

theorem distance_covered (walk_speed run_speed : ℝ) (total_time : ℝ) (h1 : walk_speed = 4)
    (h2 : run_speed = 8) (h3 : total_time = 0.75) : ℝ :=
  let total_distance := 8
  let half_distance := total_distance / 2
  let walk_time := half_distance / walk_speed
  let run_time := half_distance / run_speed
  have time_equation : walk_time + run_time = total_time := by sorry
  have distance_equation : total_distance = walk_speed * walk_time + run_speed * run_time := by sorry
  total_distance

#check distance_covered

end distance_covered_l2590_259002


namespace three_solutions_iff_a_values_l2590_259000

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0

def equation2 (x y a : ℝ) : Prop :=
  (x + 3)^2 + (y - 5)^2 = a

-- Define the solution set
def solution_set (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation1 p.1 p.2 ∧ equation2 p.1 p.2 a}

-- Theorem statement
theorem three_solutions_iff_a_values (a : ℝ) :
  (solution_set a).ncard = 3 ↔ (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
sorry

end three_solutions_iff_a_values_l2590_259000


namespace sqrt_sum_greater_than_sqrt_of_sum_l2590_259074

theorem sqrt_sum_greater_than_sqrt_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end sqrt_sum_greater_than_sqrt_of_sum_l2590_259074


namespace prob_two_red_two_blue_correct_l2590_259048

/-- The probability of selecting 2 red and 2 blue marbles from a bag -/
def probability_two_red_two_blue : ℚ :=
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let selected_marbles : ℕ := 4
  616 / 1615

/-- Theorem stating that the probability of selecting 2 red and 2 blue marbles
    from a bag with 12 red and 8 blue marbles, when 4 marbles are selected
    at random without replacement, is equal to 616/1615 -/
theorem prob_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 := by
  sorry

end prob_two_red_two_blue_correct_l2590_259048


namespace pen_count_difference_l2590_259090

theorem pen_count_difference (red : ℕ) (black : ℕ) (blue : ℕ) : 
  red = 8 →
  black = red + 10 →
  red + black + blue = 41 →
  blue > red →
  blue - red = 7 := by
sorry

end pen_count_difference_l2590_259090


namespace odd_prime_gcd_sum_and_fraction_l2590_259084

theorem odd_prime_gcd_sum_and_fraction (p a b : ℕ) : 
  Nat.Prime p → p % 2 = 1 → Nat.Coprime a b → 
  Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := by
sorry

end odd_prime_gcd_sum_and_fraction_l2590_259084


namespace fraction_inequality_l2590_259086

theorem fraction_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end fraction_inequality_l2590_259086


namespace power_difference_evaluation_l2590_259087

theorem power_difference_evaluation : (3^3)^4 - (4^4)^3 = -16245775 := by
  sorry

end power_difference_evaluation_l2590_259087


namespace teacher_age_l2590_259094

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (student_age : ℕ) (final_avg : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : student_age = 11)
  (h4 : final_avg = 11) :
  (n : ℚ) * initial_avg - student_age + (n - 1 : ℚ) * final_avg - ((n - 1 : ℚ) * initial_avg - student_age) = 30 := by
  sorry

end teacher_age_l2590_259094


namespace possible_values_of_a_l2590_259042

theorem possible_values_of_a (x y a : ℝ) :
  (|3 * y - 18| + |a * x - y| = 0) →
  (x > 0) →
  (∃ n : ℕ, x = 2 * n) →
  (x ≤ y) →
  (a = 3 ∨ a = 3/2 ∨ a = 1) :=
by sorry

end possible_values_of_a_l2590_259042


namespace lowest_power_x4_l2590_259051

theorem lowest_power_x4 (x : ℝ) : 
  let A : ℝ := 1/3
  let B : ℝ := -1/9
  let C : ℝ := 5/81
  let f : ℝ → ℝ := λ x => (1 + A*x + B*x^2 + C*x^3)^3 - (1 + x)
  ∃ (D E F G H I : ℝ), f x = D*x^4 + E*x^5 + F*x^6 + G*x^7 + H*x^8 + I*x^9 ∧ D ≠ 0 := by
  sorry

end lowest_power_x4_l2590_259051


namespace find_n_l2590_259068

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10^35)) ∧ n = 35 := by
  sorry

end find_n_l2590_259068


namespace solve_for_Q_l2590_259035

theorem solve_for_Q : ∃ Q : ℝ, (Q ^ 4).sqrt = 32 * (64 ^ (1/6)) → Q = 8 := by
  sorry

end solve_for_Q_l2590_259035


namespace consecutive_points_segment_length_l2590_259088

/-- Given 5 consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points condition
  (bc_eq_3cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (ab_eq_5 : b - a = 5) -- ab = 5
  (ac_eq_11 : c - a = 11) -- ac = 11
  (ae_eq_21 : e - a = 21) -- ae = 21
  : e - d = 8 := by -- de = 8
  sorry

end consecutive_points_segment_length_l2590_259088


namespace dilation_problem_l2590_259080

/-- Dilation of a complex number -/
def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-1 + 2*I) 4 (3 + 4*I) = 15 + 10*I := by
  sorry

end dilation_problem_l2590_259080


namespace arctan_equation_equivalence_l2590_259014

theorem arctan_equation_equivalence (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^5) = π / 6 →
  x^6 - Real.sqrt 3 * x^5 - Real.sqrt 3 * x - 1 = 0 := by
  sorry

end arctan_equation_equivalence_l2590_259014


namespace sum_product_quadratic_l2590_259055

theorem sum_product_quadratic (S P x y : ℝ) :
  x + y = S ∧ x * y = P →
  ∃ t : ℝ, t ^ 2 - S * t + P = 0 ∧ (t = x ∨ t = y) :=
by sorry

end sum_product_quadratic_l2590_259055


namespace g_value_at_3_l2590_259021

def g (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^2 - 3 * x + 6

theorem g_value_at_3 (h : g (-3) = 2) : g 3 = -20 := by
  sorry

end g_value_at_3_l2590_259021


namespace angle_supplement_in_parallel_lines_l2590_259041

-- Define the structure for our parallel lines and transversal system
structure ParallelLinesSystem where
  -- The smallest angle created by the transversal with line m
  smallest_angle : ℝ
  -- The angle between the transversal and line n on the same side
  other_angle : ℝ

-- Define our theorem
theorem angle_supplement_in_parallel_lines 
  (system : ParallelLinesSystem) 
  (h1 : system.smallest_angle = 40)
  (h2 : system.other_angle = 70) :
  180 - system.other_angle = 110 :=
by
  sorry

#check angle_supplement_in_parallel_lines

end angle_supplement_in_parallel_lines_l2590_259041


namespace mathematics_letter_probability_l2590_259022

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in 'MATHEMATICS' -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from the alphabet that appears in 'MATHEMATICS' -/
def probability : ℚ := unique_letters / alphabet_size

theorem mathematics_letter_probability : probability = 4 / 13 := by
  sorry

end mathematics_letter_probability_l2590_259022


namespace function_characterization_l2590_259047

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, is_perfect_square (f (f a - b) + b * f (2 * a))

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def solution_type_1 (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, is_even n → f n = 0) ∧
  (∀ n : ℤ, ¬is_even n → is_perfect_square (f n))

def solution_type_2 (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = n * n

theorem function_characterization (f : ℤ → ℤ) :
  satisfies_condition f → solution_type_1 f ∨ solution_type_2 f :=
sorry

end function_characterization_l2590_259047


namespace complex_collinear_solution_l2590_259099

def collinear (a b c : ℂ) : Prop :=
  ∃ t : ℝ, b - a = t • (c - a) ∨ c - a = t • (b - a)

theorem complex_collinear_solution (z : ℂ) :
  collinear 1 Complex.I z ∧ Complex.abs z = 5 →
  z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I :=
by sorry

end complex_collinear_solution_l2590_259099


namespace area_to_paint_is_132_l2590_259024

/-- The area to be painted on a wall, given its dimensions and the dimensions of an area that doesn't need painting. -/
def areaToPaint (wallHeight wallLength paintingWidth paintingHeight : ℕ) : ℕ :=
  wallHeight * wallLength - paintingWidth * paintingHeight

/-- Theorem stating that the area to be painted is 132 square feet for the given dimensions. -/
theorem area_to_paint_is_132 :
  areaToPaint 10 15 3 6 = 132 := by
  sorry

end area_to_paint_is_132_l2590_259024


namespace greatest_number_with_odd_factors_under_150_l2590_259097

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_with_odd_factors_under_150 :
  ∃ n : ℕ, n < 150 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 150 → has_odd_number_of_factors m → m ≤ n :=
by
  sorry

end greatest_number_with_odd_factors_under_150_l2590_259097


namespace square_sum_of_sqrt3_plus_minus_2_l2590_259077

theorem square_sum_of_sqrt3_plus_minus_2 :
  let a : ℝ := Real.sqrt 3 + 2
  let b : ℝ := Real.sqrt 3 - 2
  a^2 + b^2 = 14 := by
sorry

end square_sum_of_sqrt3_plus_minus_2_l2590_259077


namespace survey_solution_l2590_259061

def survey_problem (mac_preference : ℕ) (no_preference : ℕ) : Prop :=
  let both_preference : ℕ := mac_preference / 3
  let total_students : ℕ := mac_preference + both_preference + no_preference
  (mac_preference = 60) ∧ (no_preference = 90) → (total_students = 170)

theorem survey_solution : survey_problem 60 90 := by
  sorry

end survey_solution_l2590_259061


namespace mary_has_seven_balloons_l2590_259095

-- Define the number of balloons for each person
def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def total_balloons : ℕ := 18

-- Define Mary's balloons as the difference between total and the sum of Fred's and Sam's
def mary_balloons : ℕ := total_balloons - (fred_balloons + sam_balloons)

-- Theorem to prove
theorem mary_has_seven_balloons : mary_balloons = 7 := by
  sorry

end mary_has_seven_balloons_l2590_259095


namespace basketball_team_cutoff_l2590_259072

theorem basketball_team_cutoff (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 9)
  (h2 : boys = 14)
  (h3 : called_back = 2) :
  girls + boys - called_back = 21 := by
  sorry

end basketball_team_cutoff_l2590_259072


namespace propositions_true_l2590_259056

theorem propositions_true :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → (a - c) / c > (b - c) / b) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) :=
by sorry

end propositions_true_l2590_259056


namespace two_green_marbles_probability_l2590_259052

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem two_green_marbles_probability
  (red : ℕ) (green : ℕ) (white : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_white : white = 12)
  : (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 21 :=
by sorry

end two_green_marbles_probability_l2590_259052


namespace ice_cream_cost_l2590_259079

/-- Given the following conditions:
    - 16 chapatis, each costing Rs. 6
    - 5 plates of rice, each costing Rs. 45
    - 7 plates of mixed vegetable, each costing Rs. 70
    - 6 ice-cream cups
    - Total amount paid: Rs. 931
    Prove that the cost of each ice-cream cup is Rs. 20. -/
theorem ice_cream_cost (chapati_count : ℕ) (chapati_cost : ℕ)
                       (rice_count : ℕ) (rice_cost : ℕ)
                       (veg_count : ℕ) (veg_cost : ℕ)
                       (ice_cream_count : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  chapati_cost = 6 →
  rice_count = 5 →
  rice_cost = 45 →
  veg_count = 7 →
  veg_cost = 70 →
  ice_cream_count = 6 →
  total_paid = 931 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + veg_count * veg_cost)) / ice_cream_count = 20 :=
by sorry

end ice_cream_cost_l2590_259079


namespace events_mutually_exclusive_but_not_complementary_l2590_259091

structure Ball :=
  (color : String)

def Bag := List Ball

def draw (bag : Bag) : Ball × Bag :=
  match bag with
  | [] => ⟨Ball.mk "empty", []⟩
  | (b::bs) => ⟨b, bs⟩

def Event := Bag → Prop

def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ∀ bag, ¬(e1 bag ∧ e2 bag)

def complementary (e1 e2 : Event) : Prop :=
  ∀ bag, e1 bag ↔ ¬(e2 bag)

def initialBag : Bag :=
  [Ball.mk "red", Ball.mk "blue", Ball.mk "black", Ball.mk "white"]

def ADrawsWhite : Event :=
  λ bag => (draw bag).1.color = "white"

def BDrawsWhite : Event :=
  λ bag => let (_, remainingBag) := draw bag
           (draw remainingBag).1.color = "white"

theorem events_mutually_exclusive_but_not_complementary :
  mutuallyExclusive ADrawsWhite BDrawsWhite ∧
  ¬(complementary ADrawsWhite BDrawsWhite) :=
sorry

end events_mutually_exclusive_but_not_complementary_l2590_259091


namespace num_sequences_l2590_259073

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 4

/-- The minimum number of times each element appears -/
def min_appearances : ℕ := 3

/-- Theorem stating the number of possible sequences -/
theorem num_sequences (h : min_appearances ≥ sequence_length) :
  num_elements ^ sequence_length = 625 := by sorry

end num_sequences_l2590_259073


namespace opposite_of_2023_l2590_259076

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by sorry

end opposite_of_2023_l2590_259076


namespace tank_capacity_is_640_l2590_259019

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 640

/-- The time in hours it takes to empty the tank with only the outlet pipe open -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe adds water in liters per minute -/
def inlet_rate : ℝ := 4

/-- The time in hours it takes to empty the tank with both pipes open -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 liters given the conditions -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by sorry

end tank_capacity_is_640_l2590_259019


namespace smallest_divisible_term_l2590_259037

/-- An integer sequence satisfying the given recurrence relation -/
def IntegerSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)

/-- The property that 2008 divides a_2007 -/
def DivisibilityCondition (a : ℕ → ℤ) : Prop :=
  2008 ∣ a 2007

/-- The main theorem statement -/
theorem smallest_divisible_term
  (a : ℕ → ℤ)
  (h_seq : IntegerSequence a)
  (h_div : DivisibilityCondition a) :
  (∀ n : ℕ, 2 ≤ n ∧ n < 501 → ¬(2008 ∣ a n)) ∧
  (2008 ∣ a 501) := by
  sorry

end smallest_divisible_term_l2590_259037


namespace arithmetic_sequence_sum_l2590_259016

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end arithmetic_sequence_sum_l2590_259016


namespace jelly_bean_problem_l2590_259071

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial condition: 3 times as many blueberry as cherry
  b - 15 = 4 * (c - 15) →      -- Condition after eating 15 of each
  b = 135 :=                   -- Conclusion: original number of blueberry jelly beans
by sorry

end jelly_bean_problem_l2590_259071


namespace tunnel_length_l2590_259063

/-- Calculates the length of a tunnel given train and travel parameters -/
theorem tunnel_length
  (train_length : Real)
  (train_speed : Real)
  (exit_time : Real)
  (h1 : train_length = 1.5)
  (h2 : train_speed = 45)
  (h3 : exit_time = 4 / 60) :
  train_speed * exit_time - train_length = 1.5 := by
  sorry

end tunnel_length_l2590_259063


namespace area_inequality_l2590_259008

/-- A convex n-gon with circumscribed and inscribed circles -/
class ConvexNGon (n : ℕ) where
  /-- The area of the n-gon -/
  area : ℝ
  /-- The area of the circumscribed circle -/
  circumArea : ℝ
  /-- The area of the inscribed circle -/
  inscribedArea : ℝ
  /-- The n-gon is convex -/
  convex : Prop
  /-- The n-gon has a circumscribed circle -/
  hasCircumscribed : Prop
  /-- The n-gon has an inscribed circle -/
  hasInscribed : Prop

/-- Theorem: For a convex n-gon with circumscribed and inscribed circles,
    twice the area of the n-gon is less than the sum of the areas of the circumscribed and inscribed circles -/
theorem area_inequality {n : ℕ} (ngon : ConvexNGon n) :
  2 * ngon.area < ngon.circumArea + ngon.inscribedArea :=
sorry

end area_inequality_l2590_259008


namespace bella_age_is_five_l2590_259017

/-- Bella's age in years -/
def bella_age : ℕ := sorry

/-- Bella's brother's age in years -/
def brother_age : ℕ := sorry

/-- Theorem stating Bella's age given the conditions -/
theorem bella_age_is_five :
  (brother_age = bella_age + 9) →  -- Brother is 9 years older
  (bella_age + brother_age = 19) →  -- Ages add up to 19
  bella_age = 5 := by sorry

end bella_age_is_five_l2590_259017


namespace constant_product_l2590_259096

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Define a point on the right branch of the hyperbola
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

-- Define a line passing through a point
def line_through (x₀ y₀ x y : ℝ) : Prop := ∃ (m b : ℝ), y - y₀ = m * (x - x₀) ∧ y = m*x + b

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop := x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

-- Main theorem
theorem constant_product (x₀ y₀ x_A y_A x_B y_B : ℝ) :
  right_branch x₀ y₀ →
  asymptote x_A y_A ∧ asymptote x_B y_B →
  line_through x₀ y₀ x_A y_A ∧ line_through x₀ y₀ x_B y_B →
  is_midpoint x₀ y₀ x_A y_A x_B y_B →
  (x_A^2 + y_A^2) * (x_B^2 + y_B^2) = 25 :=
sorry

end constant_product_l2590_259096


namespace hotpot_expenditure_theorem_l2590_259011

/-- Represents the expenditure of three people on hotpot base materials. -/
structure HotpotExpenditure where
  a : ℕ  -- number of brands
  m : ℕ  -- price of clear soup flavor
  n : ℕ  -- price of mushroom soup flavor
  spicy_price : ℕ := 25  -- price of spicy flavor

/-- Conditions for the hotpot expenditure problem -/
def valid_expenditure (h : HotpotExpenditure) : Prop :=
  h.a * (h.spicy_price + h.m + h.n) = 1900 ∧
  33 ≤ h.m ∧ h.m < h.n ∧ h.n ≤ 37

/-- The maximum amount Xiao Li could have spent on clear soup and mushroom soup flavors -/
def max_non_spicy_expenditure (h : HotpotExpenditure) : ℕ :=
  700 - h.spicy_price

/-- The main theorem stating the maximum amount Xiao Li could have spent on non-spicy flavors -/
theorem hotpot_expenditure_theorem (h : HotpotExpenditure) 
  (h_valid : valid_expenditure h) : 
  max_non_spicy_expenditure h = 675 := by
  sorry

end hotpot_expenditure_theorem_l2590_259011


namespace probability_of_two_in_three_elevenths_l2590_259062

/-- The decimal representation of 3/11 as a sequence of digits -/
def decimalRep : ℕ → Fin 10
  | 0 => 2
  | 1 => 7
  | n + 2 => decimalRep n

/-- The period of the decimal representation of 3/11 -/
def period : ℕ := 2

/-- Count of digit 2 in one period of the decimal representation -/
def countOfTwo : ℕ := 1

theorem probability_of_two_in_three_elevenths :
  (countOfTwo : ℚ) / (period : ℚ) = 1 / 2 := by sorry

end probability_of_two_in_three_elevenths_l2590_259062


namespace locus_equation_l2590_259006

-- Define the focus point F
def F : ℝ × ℝ := (2, 0)

-- Define the directrix line l: x + 3 = 0
def l (x : ℝ) : Prop := x + 3 = 0

-- Define the distance condition for point M
def distance_condition (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_l := |x + 3|
  dist_to_F + 1 = dist_to_l

-- State the theorem
theorem locus_equation :
  ∀ M : ℝ × ℝ, distance_condition M ↔ M.2^2 = 8 * M.1 :=
sorry

end locus_equation_l2590_259006


namespace distinct_numbers_probability_l2590_259015

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 5

/-- The probability of rolling five standard, six-sided dice and getting five distinct numbers -/
def probabilityDistinctNumbers : ℚ := 5 / 54

theorem distinct_numbers_probability :
  (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice = probabilityDistinctNumbers :=
sorry

end distinct_numbers_probability_l2590_259015


namespace arctan_special_angle_combination_l2590_259082

/-- Proves that arctan(tan 75° - 3tan 15° + tan 45°) = 30° --/
theorem arctan_special_angle_combination :
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180) + Real.tan (45 * π / 180)) = 30 * π / 180 := by
  sorry

#check arctan_special_angle_combination

end arctan_special_angle_combination_l2590_259082


namespace smallest_n_divisible_by_50_and_294_l2590_259007

theorem smallest_n_divisible_by_50_and_294 :
  ∃ (n : ℕ), n > 0 ∧ 50 ∣ n^2 ∧ 294 ∣ n^3 ∧
  ∀ (m : ℕ), m > 0 ∧ 50 ∣ m^2 ∧ 294 ∣ m^3 → n ≤ m :=
by
  use 210
  sorry

end smallest_n_divisible_by_50_and_294_l2590_259007


namespace rotten_oranges_percentage_l2590_259039

/-- Proves that the percentage of rotten oranges is 15% given the conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℝ)
  (good_fruits_percentage : ℝ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 6)
  (h4 : good_fruits_percentage = 88.6)
  : (100 - (good_fruits_percentage * (total_oranges + total_bananas) / total_oranges -
     rotten_bananas_percentage * total_bananas / total_oranges)) = 15 :=
by sorry

end rotten_oranges_percentage_l2590_259039


namespace intersection_of_A_and_B_l2590_259013

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ -1 < x ∧ x < 3 := by sorry

end intersection_of_A_and_B_l2590_259013


namespace box_volume_increase_l2590_259085

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 5670, surface area is 2534, and sum of edges is 252,
    then increasing each dimension by 1 results in a volume of 7001 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5670)
  (hs : 2 * (l * w + w * h + h * l) = 2534)
  (he : 4 * (l + w + h) = 252) :
  (l + 1) * (w + 1) * (h + 1) = 7001 := by
  sorry

end box_volume_increase_l2590_259085


namespace picklminster_to_quickville_distance_l2590_259027

/-- The distance between Picklminster and Quickville satisfies the given conditions -/
theorem picklminster_to_quickville_distance :
  ∃ (d : ℝ) (vA vB vC vD : ℝ),
    d > 0 ∧ vA > 0 ∧ vB > 0 ∧ vC > 0 ∧ vD > 0 ∧
    120 * vC = vA * (d - 120) ∧
    140 * vD = vA * (d - 140) ∧
    126 * vB = vC * (d - 126) ∧
    vB = vD ∧
    d = 210 :=
by
  sorry

#check picklminster_to_quickville_distance

end picklminster_to_quickville_distance_l2590_259027


namespace initial_red_marbles_l2590_259020

theorem initial_red_marbles (r g : ℕ) : 
  (r : ℚ) / g = 5 / 3 →
  ((r - 15) : ℚ) / (g + 18) = 1 / 2 →
  r = 34 :=
by sorry

end initial_red_marbles_l2590_259020


namespace system_equation_result_l2590_259043

theorem system_equation_result (a b A B C : ℝ) (x : ℝ) 
  (h1 : a * Real.sin x + b * Real.cos x = 0)
  (h2 : A * Real.sin (2 * x) + B * Real.cos (2 * x) = C)
  (h3 : a ≠ 0) :
  2 * a * b * A + (b^2 - a^2) * B + (a^2 + b^2) * C = 0 :=
by sorry

end system_equation_result_l2590_259043


namespace compare_sqrt_sums_l2590_259023

theorem compare_sqrt_sums (a : ℝ) (h : a > 0) :
  Real.sqrt a + Real.sqrt (a + 3) < Real.sqrt (a + 1) + Real.sqrt (a + 2) := by
sorry

end compare_sqrt_sums_l2590_259023


namespace a_10_value_l2590_259046

def sequence_property (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem a_10_value (a : ℕ+ → ℤ) (h1 : sequence_property a) (h2 : a 2 = -6) :
  a 10 = -30 := by
  sorry

end a_10_value_l2590_259046


namespace correct_average_l2590_259026

-- Define the number of elements in the set
def n : ℕ := 20

-- Define the initial incorrect average
def incorrect_avg : ℚ := 25.6

-- Define the three pairs of incorrect and correct numbers
def num1 : (ℚ × ℚ) := (57.5, 78.5)
def num2 : (ℚ × ℚ) := (25.25, 35.25)
def num3 : (ℚ × ℚ) := (24.25, 47.5)

-- Define the correct average
def correct_avg : ℚ := 28.3125

-- Theorem statement
theorem correct_average : 
  let incorrect_sum := n * incorrect_avg
  let diff1 := num1.2 - num1.1
  let diff2 := num2.2 - num2.1
  let diff3 := num3.2 - num3.1
  let correct_sum := incorrect_sum + diff1 + diff2 + diff3
  correct_sum / n = correct_avg := by sorry

end correct_average_l2590_259026


namespace admission_difference_l2590_259005

/-- Represents the admission plan for a university -/
structure AdmissionPlan where
  firstTier : ℕ
  secondTier : ℕ
  thirdTier : ℕ
  ratio_condition : firstTier * 5 = secondTier * 2 ∧ firstTier * 3 = thirdTier * 2

/-- Theorem stating the difference between second-tier and first-tier admissions -/
theorem admission_difference (plan : AdmissionPlan) (h : plan.thirdTier = 1500) :
  plan.secondTier - plan.firstTier = 1500 := by
  sorry

#check admission_difference

end admission_difference_l2590_259005


namespace first_group_size_correct_l2590_259025

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 78

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end first_group_size_correct_l2590_259025


namespace inscribed_sphere_radius_l2590_259066

/-- A right cone with a sphere inscribed inside it. -/
structure InscribedSphere where
  /-- The base radius of the cone in cm. -/
  base_radius : ℝ
  /-- The height of the cone in cm. -/
  cone_height : ℝ
  /-- The radius of the inscribed sphere in cm. -/
  sphere_radius : ℝ
  /-- The base radius is 9 cm. -/
  base_radius_eq : base_radius = 9
  /-- The cone height is 27 cm. -/
  cone_height_eq : cone_height = 27
  /-- The sphere is inscribed in the cone. -/
  inscribed : sphere_radius ≤ base_radius ∧ sphere_radius ≤ cone_height

/-- The radius of the inscribed sphere is 3√10 - 3 cm. -/
theorem inscribed_sphere_radius (s : InscribedSphere) : 
  s.sphere_radius = 3 * Real.sqrt 10 - 3 := by
  sorry

#check inscribed_sphere_radius

end inscribed_sphere_radius_l2590_259066


namespace sequence_relation_l2590_259093

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : y n ^ 2 = 3 * x n ^ 2 + 1 := by
  sorry

end sequence_relation_l2590_259093
