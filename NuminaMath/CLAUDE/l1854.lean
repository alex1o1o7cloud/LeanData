import Mathlib

namespace min_sum_squared_distances_l1854_185440

/-- Given points A, B, C, D, and E on a line in that order, with specified distances between them,
    this theorem states that the minimum sum of squared distances from these points to any point P
    on the same line is 66. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_AB : B - A = 1)
  (h_BC : C - B = 2)
  (h_CD : D - C = 3)
  (h_DE : E - D = 4)
  (h_P : A ≤ P ∧ P ≤ E) :
  66 ≤ (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 :=
sorry

end min_sum_squared_distances_l1854_185440


namespace coconut_cost_is_fifty_cents_l1854_185465

/-- Represents the cost per coconut on Rohan's farm -/
def coconut_cost (farm_size : ℕ) (trees_per_sqm : ℕ) (coconuts_per_tree : ℕ) 
  (harvest_interval : ℕ) (months : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_trees := farm_size * trees_per_sqm
  let total_coconuts := total_trees * coconuts_per_tree
  let harvests := months / harvest_interval
  let total_harvested := total_coconuts * harvests
  total_earnings / total_harvested

/-- Proves that the cost per coconut on Rohan's farm is $0.50 -/
theorem coconut_cost_is_fifty_cents :
  coconut_cost 20 2 6 3 6 240 = 1/2 := by
  sorry

end coconut_cost_is_fifty_cents_l1854_185465


namespace complex_circle_range_l1854_185493

theorem complex_circle_range (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (Complex.abs (z - Complex.mk 3 4) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end complex_circle_range_l1854_185493


namespace tan_negative_405_degrees_l1854_185468

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end tan_negative_405_degrees_l1854_185468


namespace simplify_sqrt_expression_l1854_185462

/-- Simplification of a complex expression involving square roots and exponents -/
theorem simplify_sqrt_expression :
  let x := Real.sqrt 3
  (x - 1) ^ (1 - Real.sqrt 2) / (x + 1) ^ (1 + Real.sqrt 2) = 4 - 2 * x :=
by sorry

end simplify_sqrt_expression_l1854_185462


namespace sin_inequality_l1854_185488

theorem sin_inequality (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  f x ≤ |f (π / 6)| := by
sorry

end sin_inequality_l1854_185488


namespace total_cost_is_54_44_l1854_185464

-- Define the quantities and prices
def book_quantity : ℕ := 1
def book_price : ℚ := 16
def binder_quantity : ℕ := 3
def binder_price : ℚ := 2
def notebook_quantity : ℕ := 6
def notebook_price : ℚ := 1
def pen_quantity : ℕ := 4
def pen_price : ℚ := 1/2
def calculator_quantity : ℕ := 2
def calculator_price : ℚ := 12

-- Define discount and tax rates
def discount_rate : ℚ := 1/10
def tax_rate : ℚ := 7/100

-- Define the total cost function
def total_cost : ℚ :=
  let book_cost := book_quantity * book_price
  let binder_cost := binder_quantity * binder_price
  let notebook_cost := notebook_quantity * notebook_price
  let pen_cost := pen_quantity * pen_price
  let calculator_cost := calculator_quantity * calculator_price
  
  let discounted_book_cost := book_cost * (1 - discount_rate)
  let discounted_binder_cost := binder_cost * (1 - discount_rate)
  
  let subtotal := discounted_book_cost + discounted_binder_cost + notebook_cost + pen_cost + calculator_cost
  let tax := (notebook_cost + pen_cost + calculator_cost) * tax_rate
  
  subtotal + tax

-- Theorem statement
theorem total_cost_is_54_44 : total_cost = 5444 / 100 := by
  sorry

end total_cost_is_54_44_l1854_185464


namespace quadratic_no_real_roots_l1854_185448

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end quadratic_no_real_roots_l1854_185448


namespace difference_of_squares_l1854_185439

theorem difference_of_squares : 72^2 - 54^2 = 2268 := by sorry

end difference_of_squares_l1854_185439


namespace stationery_cost_theorem_l1854_185474

/-- The cost of stationery items -/
structure StationeryCost where
  pencil : ℝ  -- Cost of one pencil
  pen : ℝ     -- Cost of one pen
  eraser : ℝ  -- Cost of one eraser

/-- Given conditions on stationery costs -/
def stationery_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil + 3 * c.pen + c.eraser = 5.40 ∧
  2 * c.pencil + 2 * c.pen + 2 * c.eraser = 4.60

/-- Theorem stating the cost of 1 pencil, 2 pens, and 3 erasers -/
theorem stationery_cost_theorem (c : StationeryCost) 
  (h : stationery_conditions c) : 
  c.pencil + 2 * c.pen + 3 * c.eraser = 4.60 := by
  sorry

end stationery_cost_theorem_l1854_185474


namespace max_value_implies_a_l1854_185424

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 :=
sorry

end max_value_implies_a_l1854_185424


namespace stating_modified_mindmaster_secret_codes_l1854_185454

/-- The number of different colors available in the modified Mindmaster game -/
def num_colors : ℕ := 6

/-- The number of slots to be filled in the modified Mindmaster game -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the modified Mindmaster game -/
def num_secret_codes : ℕ := num_colors ^ num_slots

/-- 
Theorem stating that the number of possible secret codes in the modified Mindmaster game is 7776,
given 6 colors, 5 slots, allowing color repetition, and no empty slots.
-/
theorem modified_mindmaster_secret_codes : num_secret_codes = 7776 := by
  sorry

end stating_modified_mindmaster_secret_codes_l1854_185454


namespace power_fraction_equality_l1854_185478

theorem power_fraction_equality : (27 ^ 20) / (81 ^ 10) = 3 ^ 20 := by sorry

end power_fraction_equality_l1854_185478


namespace apples_bought_correct_l1854_185444

/-- Represents the number of apples Mary bought -/
def apples_bought : ℕ := 6

/-- Represents the number of apples Mary ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of trees planted per apple eaten -/
def trees_per_apple : ℕ := 2

/-- Theorem stating that the number of apples Mary bought is correct -/
theorem apples_bought_correct : 
  apples_bought = apples_eaten + apples_eaten * trees_per_apple :=
by sorry

end apples_bought_correct_l1854_185444


namespace earl_initial_ascent_l1854_185482

def building_height : ℕ := 20

def initial_floor : ℕ := 1

theorem earl_initial_ascent (x : ℕ) : 
  x + 5 = building_height - 9 → x = 6 := by
  sorry

end earl_initial_ascent_l1854_185482


namespace pears_equivalent_to_24_bananas_is_12_l1854_185412

/-- The number of pears equivalent in cost to 24 bananas -/
def pears_equivalent_to_24_bananas (banana_apple_ratio : ℚ) (apple_pear_ratio : ℚ) : ℚ :=
  24 * banana_apple_ratio * apple_pear_ratio

theorem pears_equivalent_to_24_bananas_is_12 :
  pears_equivalent_to_24_bananas (3/4) (6/9) = 12 := by
  sorry

end pears_equivalent_to_24_bananas_is_12_l1854_185412


namespace horner_method_v3_equals_20_l1854_185445

def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 + 3) * x - 2
  (v2 * x + 1) * x - 1

theorem horner_method_v3_equals_20 :
  horner_v3 f 2 = 20 :=
by sorry

end horner_method_v3_equals_20_l1854_185445


namespace octagonal_pyramid_volume_l1854_185480

/-- The volume of a regular octagonal pyramid with given dimensions -/
theorem octagonal_pyramid_volume :
  ∀ (base_side_length equilateral_face_side_length : ℝ),
    base_side_length = 5 →
    equilateral_face_side_length = 10 →
    ∃ (volume : ℝ),
      volume = (250 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 3 ∧
      volume = (1 / 3) * (2 * (1 + Real.sqrt 2) * base_side_length^2) * 
               ((equilateral_face_side_length * Real.sqrt 3) / 2) :=
by sorry

end octagonal_pyramid_volume_l1854_185480


namespace rectangle_and_triangle_l1854_185400

/-- Given a rectangle ABCD and an isosceles right triangle DCE, prove that DE = 4√3 -/
theorem rectangle_and_triangle (AB AD DC DE : ℝ) : 
  AB = 6 →
  AD = 8 →
  DC = DE →
  AB * AD = 2 * (1/2 * DC * DE) →
  DE = 4 * Real.sqrt 3 := by
  sorry

end rectangle_and_triangle_l1854_185400


namespace pencil_count_l1854_185471

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := 30

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := mitchell_pencils - (mitchell_pencils * 20 / 100)

/-- The number of pencils Elizabeth has -/
def elizabeth_pencils : ℕ := 2 * antonio_pencils

/-- The total number of pencils Mitchell, Antonio, and Elizabeth have together -/
def total_pencils : ℕ := mitchell_pencils + antonio_pencils + elizabeth_pencils

theorem pencil_count : total_pencils = 102 := by
  sorry

end pencil_count_l1854_185471


namespace book_arrangement_proof_l1854_185467

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrange_books (total : ℕ) (math_copies : ℕ) (physics_copies : ℕ) : ℕ :=
  factorial total / (factorial math_copies * factorial physics_copies)

theorem book_arrangement_proof :
  arrange_books 7 3 2 = 420 := by
  sorry

end book_arrangement_proof_l1854_185467


namespace total_match_sequences_l1854_185427

/-- Represents the number of players in each team -/
def n : ℕ := 7

/-- Calculates the number of possible match sequences for one team winning -/
def sequences_for_one_team_winning : ℕ := Nat.choose (2 * n - 1) (n - 1)

/-- Theorem stating the total number of possible match sequences -/
theorem total_match_sequences : 2 * sequences_for_one_team_winning = 3432 := by
  sorry

end total_match_sequences_l1854_185427


namespace distance_after_two_hours_l1854_185487

/-- The distance between two people walking in opposite directions for a given time -/
def distance_apart (maya_speed : ℚ) (lucas_speed : ℚ) (time : ℚ) : ℚ :=
  maya_speed * time + lucas_speed * time

/-- Theorem stating the distance apart after 2 hours -/
theorem distance_after_two_hours :
  let maya_speed : ℚ := 1 / 20 -- miles per minute
  let lucas_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- 2 hours in minutes
  distance_apart maya_speed lucas_speed time = 15 := by
  sorry

#eval distance_apart (1/20) (3/40) 120

end distance_after_two_hours_l1854_185487


namespace product_of_number_and_sum_of_digits_l1854_185441

theorem product_of_number_and_sum_of_digits : 
  let n : ℕ := 26
  let tens : ℕ := n / 10
  let units : ℕ := n % 10
  units = tens + 4 →
  n * (tens + units) = 208 :=
by
  sorry

end product_of_number_and_sum_of_digits_l1854_185441


namespace basketball_cards_per_box_l1854_185429

theorem basketball_cards_per_box : 
  ∀ (basketball_cards_per_box : ℕ),
    (4 * basketball_cards_per_box + 5 * 8 = 58 + 22) → 
    basketball_cards_per_box = 10 := by
  sorry

end basketball_cards_per_box_l1854_185429


namespace simplify_part1_simplify_part2_l1854_185411

-- Part 1
theorem simplify_part1 (x : ℝ) (h1 : 1 ≤ x) (h2 : x < 4) :
  Real.sqrt (1 - 2*x + x^2) + Real.sqrt (x^2 - 8*x + 16) = 3 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : 2 - x ≥ 0) :
  (Real.sqrt (2 - x))^2 - Real.sqrt (x^2 - 6*x + 9) = -1 := by sorry

end simplify_part1_simplify_part2_l1854_185411


namespace modulo_23_equivalence_l1854_185498

theorem modulo_23_equivalence (n : ℤ) : 0 ≤ n ∧ n < 23 ∧ -207 ≡ n [ZMOD 23] → n = 0 := by
  sorry

end modulo_23_equivalence_l1854_185498


namespace binomial_13_8_l1854_185436

theorem binomial_13_8 (h1 : Nat.choose 14 7 = 3432) 
                      (h2 : Nat.choose 14 8 = 3003) 
                      (h3 : Nat.choose 12 7 = 792) : 
  Nat.choose 13 8 = 1287 := by
  sorry

end binomial_13_8_l1854_185436


namespace first_grade_enrollment_l1854_185494

theorem first_grade_enrollment :
  ∃ (n : ℕ),
    200 ≤ n ∧ n ≤ 300 ∧
    ∃ (r : ℕ), n = 25 * r + 10 ∧
    ∃ (l : ℕ), n = 30 * l - 15 ∧
    n = 285 := by
  sorry

end first_grade_enrollment_l1854_185494


namespace money_distribution_l1854_185497

theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1000 →
  a + c = 300 →
  b + c = 200 →
  d + e = 350 →
  a + d = 250 →
  b + e = 150 →
  a + b + c = 400 →
  (a = 200 ∧ b = 100 ∧ c = 100 ∧ d = 50 ∧ e = 300) :=
by sorry

end money_distribution_l1854_185497


namespace average_first_six_l1854_185420

theorem average_first_six (total_count : Nat) (total_avg : ℝ) (last_six_avg : ℝ) (sixth_num : ℝ) :
  total_count = 11 →
  total_avg = 10.7 →
  last_six_avg = 11.4 →
  sixth_num = 13.700000000000017 →
  (6 * ((total_count : ℝ) * total_avg - 6 * last_six_avg + sixth_num)) / 6 = 10.5 := by
  sorry

#check average_first_six

end average_first_six_l1854_185420


namespace A_intersect_B_l1854_185423

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {2, 3} := by
  sorry

end A_intersect_B_l1854_185423


namespace marinara_stains_l1854_185443

theorem marinara_stains (grass_time : ℕ) (marinara_time : ℕ) (grass_count : ℕ) (total_time : ℕ) :
  grass_time = 4 →
  marinara_time = 7 →
  grass_count = 3 →
  total_time = 19 →
  (total_time - grass_time * grass_count) / marinara_time = 1 :=
by sorry

end marinara_stains_l1854_185443


namespace base_nine_proof_l1854_185425

theorem base_nine_proof (b : ℕ) : 
  (∃ (n : ℕ), n = 144 ∧ 
    n = (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)) →
  b = 9 :=
by sorry

end base_nine_proof_l1854_185425


namespace hyperbola_equation_l1854_185404

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c^2 = 7 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop := ∃ a b : ℝ, a^2 - b^2 = 1 ∧ (P.1^2/a^2) - (P.2^2/b^2) = 1

-- Define perpendicularity of PF₁ and PF₂
def perpendicular (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the product condition
def product_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Theorem statement
theorem hyperbola_equation (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  on_hyperbola P →
  perpendicular P F₁ F₂ →
  product_condition P F₁ F₂ →
  ∃ x y : ℝ, P = (x, y) ∧ x^2/6 - y^2 = 1 :=
sorry

end hyperbola_equation_l1854_185404


namespace profit_maximizing_price_l1854_185459

/-- Represents the profit maximization problem for a product -/
structure ProfitProblem where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialQuantity : ℝ
  priceElasticity : ℝ

/-- Calculates the profit for a given price increase -/
def profit (problem : ProfitProblem) (priceIncrease : ℝ) : ℝ :=
  let newPrice := problem.initialSellingPrice + priceIncrease
  let newQuantity := problem.initialQuantity - problem.priceElasticity * priceIncrease
  (newPrice - problem.initialPurchasePrice) * newQuantity

/-- Theorem stating that the profit-maximizing price is 95 yuan -/
theorem profit_maximizing_price (problem : ProfitProblem) 
  (h1 : problem.initialPurchasePrice = 80)
  (h2 : problem.initialSellingPrice = 90)
  (h3 : problem.initialQuantity = 400)
  (h4 : problem.priceElasticity = 20) :
  ∃ (maxProfit : ℝ), ∀ (price : ℝ), 
    profit problem (price - problem.initialSellingPrice) ≤ maxProfit ∧
    profit problem (95 - problem.initialSellingPrice) = maxProfit :=
sorry

end profit_maximizing_price_l1854_185459


namespace new_ratio_is_one_to_two_l1854_185430

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Calculates the new ratio after adding new boarders -/
def new_ratio (initial : Ratio) (new_boarders : ℕ) : Ratio :=
  { boarders := initial.boarders + new_boarders,
    day_students := initial.day_students }

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.boarders r.day_students
  { boarders := r.boarders / gcd,
    day_students := r.day_students / gcd }

theorem new_ratio_is_one_to_two :
  let initial_ratio : Ratio := { boarders := 330, day_students := 792 }
  let new_boarders : ℕ := 66
  let final_ratio := simplify_ratio (new_ratio initial_ratio new_boarders)
  final_ratio.boarders = 1 ∧ final_ratio.day_students = 2 := by
  sorry


end new_ratio_is_one_to_two_l1854_185430


namespace sufficient_not_necessary_negation_l1854_185495

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h1 : ¬p → q)  -- ¬p is sufficient for q
  (h2 : ¬(q → ¬p)) -- ¬p is not necessary for q
  : (¬q → p) ∧ ¬(p → ¬q) := by sorry

end sufficient_not_necessary_negation_l1854_185495


namespace solution_set_l1854_185461

theorem solution_set (x : ℝ) : (x^2 - 3*x > 8 ∧ |x| > 2) ↔ x < -2 ∨ x > 4 := by
  sorry

end solution_set_l1854_185461


namespace omelets_per_person_l1854_185408

/-- Given 3 dozen eggs, 4 eggs per omelet, and 3 people, prove that each person gets 3 omelets when all eggs are used. -/
theorem omelets_per_person (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 3 * 12 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by sorry

end omelets_per_person_l1854_185408


namespace red_pens_count_l1854_185476

/-- The number of red pens in Maria's desk drawer. -/
def red_pens : ℕ := sorry

/-- The number of black pens in Maria's desk drawer. -/
def black_pens : ℕ := red_pens + 10

/-- The number of blue pens in Maria's desk drawer. -/
def blue_pens : ℕ := red_pens + 7

/-- The total number of pens in Maria's desk drawer. -/
def total_pens : ℕ := 41

theorem red_pens_count : red_pens = 8 := by
  sorry

end red_pens_count_l1854_185476


namespace rectangle_area_diagonal_relation_l1854_185490

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal^2 ∧ k = 10 / 29 := by
sorry

end rectangle_area_diagonal_relation_l1854_185490


namespace probability_of_even_sum_l1854_185416

def number_of_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_is_even (a b : ℕ) : Prop := is_even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by
  sorry

end probability_of_even_sum_l1854_185416


namespace cassidy_poster_addition_l1854_185414

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy has currently -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will have after this summer -/
def future_posters : ℕ := 2 * posters_two_years_ago

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := future_posters - current_posters

theorem cassidy_poster_addition : posters_to_add = 6 := by
  sorry

end cassidy_poster_addition_l1854_185414


namespace water_fraction_after_three_replacements_l1854_185426

/-- Represents the fraction of water remaining in a radiator after repeated partial replacements with antifreeze. -/
def waterFractionAfterReplacements (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  ((initialVolume - replacementVolume) / initialVolume) ^ numReplacements

/-- Theorem stating that after three replacements in a 20-quart radiator, 
    the fraction of water remaining is 27/64. -/
theorem water_fraction_after_three_replacements :
  waterFractionAfterReplacements 20 5 3 = 27 / 64 := by
  sorry

#eval waterFractionAfterReplacements 20 5 3

end water_fraction_after_three_replacements_l1854_185426


namespace complementary_angles_difference_l1854_185485

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 → x = 5 * y → |x - y| = 60 := by
  sorry

end complementary_angles_difference_l1854_185485


namespace pretzels_john_ate_l1854_185479

/-- Given a bowl of pretzels and information about how many pretzels three people ate,
    prove how many pretzels John ate. -/
theorem pretzels_john_ate (total : ℕ) (john alan marcus : ℕ) 
    (h1 : total = 95)
    (h2 : alan = john - 9)
    (h3 : marcus = john + 12)
    (h4 : marcus = 40) :
    john = 28 := by sorry

end pretzels_john_ate_l1854_185479


namespace modulus_of_complex_fraction_l1854_185435

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 * i / (1 - i)) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l1854_185435


namespace multiples_difference_cubed_zero_l1854_185477

theorem multiples_difference_cubed_zero : 
  let a := (Finset.filter (fun x => x % 12 = 0 ∧ x > 0) (Finset.range 60)).card
  let b := (Finset.filter (fun x => x % 4 = 0 ∧ x % 3 = 0 ∧ x > 0) (Finset.range 60)).card
  (a - b)^3 = 0 := by
sorry

end multiples_difference_cubed_zero_l1854_185477


namespace shaded_fraction_of_rectangle_l1854_185405

theorem shaded_fraction_of_rectangle (length width : ℕ) (h1 : length = 15) (h2 : width = 24) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 3
  (shaded_area : ℚ) / total_area = 1 / 9 := by
sorry

end shaded_fraction_of_rectangle_l1854_185405


namespace simplify_sqrt_450_l1854_185402

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l1854_185402


namespace characterize_f_l1854_185432

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f n ≠ 1 ∧ f n + f (n + 1) = f (n + 2) + f (n + 3) - 168

theorem characterize_f (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ (c d a : ℕ), (∀ n, f (2 * n) = c + n * d) ∧
                 (∀ n, f (2 * n + 1) = (168 - d) * n + a - c) ∧
                 c > 1 ∧
                 a > c + 1 :=
sorry

end characterize_f_l1854_185432


namespace min_black_cells_l1854_185410

/-- Represents a board configuration -/
def Board := Fin 2007 → Fin 2007 → Bool

/-- Checks if three cells form an L-trinome -/
def is_L_trinome (b : Board) (i j k : Fin 2007 × Fin 2007) : Prop :=
  sorry

/-- Checks if a board configuration is valid -/
def is_valid_configuration (b : Board) : Prop :=
  ∀ i j k, is_L_trinome b i j k → ¬(b i.1 i.2 ∧ b j.1 j.2 ∧ b k.1 k.2)

/-- Counts the number of black cells in a board configuration -/
def count_black_cells (b : Board) : Nat :=
  sorry

/-- The main theorem -/
theorem min_black_cells :
  ∃ (b : Board),
    is_valid_configuration b ∧
    count_black_cells b = (2007^2 / 3 : Nat) ∧
    ∀ (b' : Board),
      (∀ i j, b i j → b' i j) →
      count_black_cells b' > count_black_cells b →
      ¬is_valid_configuration b' :=
sorry

end min_black_cells_l1854_185410


namespace tangent_points_constant_sum_l1854_185460

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- Checks if a line through two points is tangent to the parabola -/
def isTangent (p1 p2 : Point) : Prop :=
  p2 ∈ Parabola ∧ (∃ k : ℝ, p1.y - p2.y = k * (p1.x - p2.x) ∧ k = p2.x / 2)

theorem tangent_points_constant_sum (a : ℝ) :
  ∀ A B : Point,
  isTangent (Point.mk a (-2)) A ∧
  isTangent (Point.mk a (-2)) B ∧
  A ≠ B →
  A.x * B.x + A.y * B.y = -4 := by
  sorry

end tangent_points_constant_sum_l1854_185460


namespace girls_in_math_class_l1854_185473

theorem girls_in_math_class
  (boy_girl_ratio : ℚ)
  (math_science_ratio : ℚ)
  (science_lit_ratio : ℚ)
  (total_students : ℕ)
  (h1 : boy_girl_ratio = 5 / 8)
  (h2 : math_science_ratio = 7 / 4)
  (h3 : science_lit_ratio = 3 / 5)
  (h4 : total_students = 720) :
  ∃ (girls_math : ℕ), girls_math = 176 :=
by sorry

end girls_in_math_class_l1854_185473


namespace solve_system_l1854_185466

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end solve_system_l1854_185466


namespace factorial_fraction_equals_four_l1854_185489

theorem factorial_fraction_equals_four :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 4 := by
  sorry

end factorial_fraction_equals_four_l1854_185489


namespace min_sequence_length_l1854_185469

def S : Finset Nat := {1, 2, 3, 4}

def isValidPermutation (perm : List Nat) : Prop :=
  perm.length = 4 ∧ perm.toFinset = S ∧ perm.getLast? ≠ some 1

def containsAllValidPermutations (seq : List Nat) : Prop :=
  ∀ perm : List Nat, isValidPermutation perm →
    ∃ i₁ i₂ i₃ i₄ : Nat,
      i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧
      i₄ ≤ seq.length ∧
      seq.get? i₁ = some (perm.get! 0) ∧
      seq.get? i₂ = some (perm.get! 1) ∧
      seq.get? i₃ = some (perm.get! 2) ∧
      seq.get? i₄ = some (perm.get! 3)

theorem min_sequence_length :
  ∃ seq : List Nat, seq.length = 11 ∧ containsAllValidPermutations seq ∧
  ∀ seq' : List Nat, seq'.length < 11 → ¬containsAllValidPermutations seq' :=
sorry

end min_sequence_length_l1854_185469


namespace percent_of_y_l1854_185491

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end percent_of_y_l1854_185491


namespace additional_money_needed_mrs_smith_purchase_l1854_185446

theorem additional_money_needed (initial_amount : ℝ) 
  (additional_fraction : ℝ) (discount_percentage : ℝ) : ℝ :=
  let total_before_discount := initial_amount * (1 + additional_fraction)
  let discounted_amount := total_before_discount * (1 - discount_percentage / 100)
  discounted_amount - initial_amount

theorem mrs_smith_purchase : 
  additional_money_needed 500 (2/5) 15 = 95 := by
  sorry

end additional_money_needed_mrs_smith_purchase_l1854_185446


namespace product_mod_six_l1854_185458

theorem product_mod_six : 2017 * 2018 * 2019 * 2020 ≡ 0 [ZMOD 6] := by
  sorry

end product_mod_six_l1854_185458


namespace sister_glue_sticks_l1854_185438

theorem sister_glue_sticks (total : ℕ) (emily : ℕ) (sister : ℕ) : 
  total = 13 → emily = 6 → sister = total - emily → sister = 7 := by
  sorry

end sister_glue_sticks_l1854_185438


namespace max_sum_of_xy_l1854_185449

theorem max_sum_of_xy (x y : ℕ+) : 
  (x * y : ℕ) - (x + y : ℕ) = Nat.gcd x y + Nat.lcm x y → 
  (∃ (c : ℕ), ∀ (a b : ℕ+), 
    (a * b : ℕ) - (a + b : ℕ) = Nat.gcd a b + Nat.lcm a b → 
    (a + b : ℕ) ≤ c) ∧ 
  (x + y : ℕ) ≤ 10 :=
sorry

end max_sum_of_xy_l1854_185449


namespace window_side_length_main_theorem_l1854_185409

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio_height_to_width : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  border_width : ℝ
  side_length : ℝ
  pane_arrangement : side_length = 3 * pane.width + 4 * border_width

/-- Theorem: The side length of the square window is 24 inches -/
theorem window_side_length (w : SquareWindow) 
  (h1 : w.border_width = 3) : w.side_length = 24 := by
  sorry

/-- Main theorem combining all conditions -/
theorem main_theorem : ∃ (w : SquareWindow), 
  w.border_width = 3 ∧ w.side_length = 24 := by
  sorry

end window_side_length_main_theorem_l1854_185409


namespace scientific_notation_of_19_4_billion_l1854_185452

theorem scientific_notation_of_19_4_billion :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 19.4 * (10 ^ 9) = a * (10 ^ n) ∧ a = 1.94 ∧ n = 10 := by
  sorry

end scientific_notation_of_19_4_billion_l1854_185452


namespace john_coffee_consumption_l1854_185433

/-- Represents the number of fluid ounces in a gallon -/
def gallonToOunces : ℚ := 128

/-- Represents the number of fluid ounces in a standard cup -/
def cupToOunces : ℚ := 8

/-- Represents the number of days between John's coffee purchases -/
def purchaseInterval : ℚ := 4

/-- Represents the fraction of a gallon John buys each time -/
def purchaseAmount : ℚ := 1/2

/-- Theorem stating that John drinks 2 cups of coffee per day -/
theorem john_coffee_consumption :
  let cupsPerPurchase := purchaseAmount * gallonToOunces / cupToOunces
  cupsPerPurchase / purchaseInterval = 2 := by sorry

end john_coffee_consumption_l1854_185433


namespace total_remaining_is_13589_08_l1854_185499

/-- Represents the daily sales and ingredient cost data for Du Chin's meat pie business --/
structure DailyData where
  pies_sold : ℕ
  sales : ℚ
  ingredient_cost : ℚ
  remaining : ℚ

/-- Calculates the daily data for Du Chin's meat pie business over a week --/
def calculate_week_data : List DailyData :=
  let monday_data : DailyData := {
    pies_sold := 200,
    sales := 4000,
    ingredient_cost := 2400,
    remaining := 1600
  }
  let tuesday_data : DailyData := {
    pies_sold := 220,
    sales := 4400,
    ingredient_cost := 2640,
    remaining := 1760
  }
  let wednesday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let thursday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let friday_data : DailyData := {
    pies_sold := 240,
    sales := 4800,
    ingredient_cost := 2494.80,
    remaining := 2305.20
  }
  let saturday_data : DailyData := {
    pies_sold := 221,
    sales := 4420,
    ingredient_cost := 2370.06,
    remaining := 2049.94
  }
  let sunday_data : DailyData := {
    pies_sold := 232,
    sales := 4640,
    ingredient_cost := 2370.06,
    remaining := 2269.94
  }
  [monday_data, tuesday_data, wednesday_data, thursday_data, friday_data, saturday_data, sunday_data]

/-- Calculates the total remaining money for the week --/
def total_remaining (week_data : List DailyData) : ℚ :=
  week_data.foldl (fun acc day => acc + day.remaining) 0

/-- Theorem stating that the total remaining money for the week is $13589.08 --/
theorem total_remaining_is_13589_08 :
  total_remaining (calculate_week_data) = 13589.08 := by
  sorry


end total_remaining_is_13589_08_l1854_185499


namespace N_subset_M_l1854_185472

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 4}

theorem N_subset_M : N ⊆ M := by sorry

end N_subset_M_l1854_185472


namespace probability_four_different_socks_l1854_185442

/-- The number of pairs of socks in the bag -/
def num_pairs : ℕ := 5

/-- The number of socks drawn in each sample -/
def sample_size : ℕ := 4

/-- The probability of drawing 4 different socks in the first draw -/
def p1 : ℚ := 8 / 21

/-- The probability of drawing exactly one pair and two different socks in the first draw -/
def p2 : ℚ := 4 / 7

/-- The probability of drawing 2 different socks in the next draw, given that we already have 3 different socks and one pair discarded -/
def p3 : ℚ := 4 / 15

/-- The theorem stating the probability of ending up with 4 socks of different colors -/
theorem probability_four_different_socks : 
  p1 + p2 * p3 = 8 / 15 := by sorry

end probability_four_different_socks_l1854_185442


namespace power_sum_equation_l1854_185406

/-- Given two real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_equation (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 123 := by
  sorry

end power_sum_equation_l1854_185406


namespace replacement_stove_cost_l1854_185470

/-- The cost of a replacement stove and wall repair, given specific conditions. -/
theorem replacement_stove_cost (stove_cost wall_cost : ℚ) : 
  wall_cost = (1 : ℚ) / 6 * stove_cost →
  stove_cost + wall_cost = 1400 →
  stove_cost = 1200 := by
sorry

end replacement_stove_cost_l1854_185470


namespace tens_digit_of_3_to_405_l1854_185419

theorem tens_digit_of_3_to_405 : ∃ n : ℕ, 3^405 ≡ 40 + n [ZMOD 100] :=
sorry

end tens_digit_of_3_to_405_l1854_185419


namespace binomial_square_expansion_l1854_185451

theorem binomial_square_expansion (x : ℝ) : (1 - x)^2 = 1 - 2*x + x^2 := by
  sorry

end binomial_square_expansion_l1854_185451


namespace incorrect_bracket_expansion_l1854_185455

theorem incorrect_bracket_expansion : ∀ x : ℝ, 3 * x^2 - 3 * (x + 6) ≠ 3 * x^2 - 3 * x - 6 := by
  sorry

end incorrect_bracket_expansion_l1854_185455


namespace student_selection_l1854_185407

theorem student_selection (male_count : Nat) (female_count : Nat) :
  male_count = 5 →
  female_count = 4 →
  (Nat.choose (male_count + female_count) 3 -
   Nat.choose male_count 3 -
   Nat.choose female_count 3) = 70 := by
  sorry

end student_selection_l1854_185407


namespace quiz_passing_requirement_l1854_185496

theorem quiz_passing_requirement (total_questions : ℕ) 
  (chemistry_questions biology_questions physics_questions : ℕ)
  (chemistry_correct_percent biology_correct_percent physics_correct_percent : ℚ)
  (passing_grade : ℚ) :
  total_questions = 100 →
  chemistry_questions = 20 →
  biology_questions = 40 →
  physics_questions = 40 →
  chemistry_correct_percent = 80 / 100 →
  biology_correct_percent = 50 / 100 →
  physics_correct_percent = 55 / 100 →
  passing_grade = 65 / 100 →
  (passing_grade * total_questions : ℚ).ceil - 
  (chemistry_correct_percent * chemistry_questions +
   biology_correct_percent * biology_questions +
   physics_correct_percent * physics_questions : ℚ).floor = 7 := by
  sorry

end quiz_passing_requirement_l1854_185496


namespace unique_score_theorem_l1854_185418

/-- Represents a score in the mathematics competition. -/
structure Score where
  value : ℕ
  correct : ℕ
  wrong : ℕ
  total_questions : ℕ
  h1 : value = 5 * correct - 2 * wrong
  h2 : correct + wrong ≤ total_questions

/-- The unique score over 70 that allows determination of correct answers. -/
def unique_determinable_score : ℕ := 71

theorem unique_score_theorem (s : Score) (h_total : s.total_questions = 25) 
    (h_over_70 : s.value > 70) : 
  (∃! c w, s.correct = c ∧ s.wrong = w) ↔ s.value = unique_determinable_score :=
sorry

end unique_score_theorem_l1854_185418


namespace no_45_degree_rectangle_with_odd_intersections_l1854_185486

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a rectangle on a grid --/
structure GridRectangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint
  D : GridPoint

/-- Checks if a point is on a grid line --/
def isOnGridLine (p : GridPoint) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if a line segment intersects the grid at a 45° angle --/
def intersectsAt45Degrees (p1 p2 : GridPoint) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p2.x - p1.x = k ∧ p2.y - p1.y = k

/-- Counts the number of grid lines intersected by a line segment --/
noncomputable def gridLinesIntersected (p1 p2 : GridPoint) : ℕ :=
  sorry

/-- Main theorem: No rectangle exists with the given properties --/
theorem no_45_degree_rectangle_with_odd_intersections :
  ¬ ∃ (rect : GridRectangle),
    (¬ isOnGridLine rect.A) ∧ (¬ isOnGridLine rect.B) ∧ 
    (¬ isOnGridLine rect.C) ∧ (¬ isOnGridLine rect.D) ∧
    (intersectsAt45Degrees rect.A rect.B) ∧ 
    (intersectsAt45Degrees rect.B rect.C) ∧
    (intersectsAt45Degrees rect.C rect.D) ∧ 
    (intersectsAt45Degrees rect.D rect.A) ∧
    (Odd (gridLinesIntersected rect.A rect.B)) ∧
    (Odd (gridLinesIntersected rect.B rect.C)) ∧
    (Odd (gridLinesIntersected rect.C rect.D)) ∧
    (Odd (gridLinesIntersected rect.D rect.A)) :=
by
  sorry

end no_45_degree_rectangle_with_odd_intersections_l1854_185486


namespace parallelogram_properties_independence_l1854_185457

/-- A parallelogram with potentially equal sides and/or right angles -/
structure Parallelogram where
  has_equal_sides : Bool
  has_right_angles : Bool

/-- Theorem: There exist parallelograms with equal sides but not right angles, 
    and parallelograms with right angles but not equal sides -/
theorem parallelogram_properties_independence :
  ∃ (p q : Parallelogram), 
    (p.has_equal_sides ∧ ¬p.has_right_angles) ∧
    (q.has_right_angles ∧ ¬q.has_equal_sides) :=
by
  sorry


end parallelogram_properties_independence_l1854_185457


namespace time_for_accidents_l1854_185453

-- Define the frequency of car collisions and big crashes
def collision_frequency : ℕ := 10  -- seconds
def crash_frequency : ℕ := 20  -- seconds

-- Define the total number of accidents
def total_accidents : ℕ := 36

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Theorem to prove
theorem time_for_accidents : 
  ∃ (minutes : ℕ), 
    (seconds_per_minute / collision_frequency + seconds_per_minute / crash_frequency) * minutes = total_accidents ∧
    minutes = 4 :=
by sorry

end time_for_accidents_l1854_185453


namespace greatest_divisor_four_consecutive_integers_l1854_185431

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → 12 ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧
  (∀ (m : ℕ), m > 12 → ∃ (l : ℕ), l > 0 ∧ ¬(m ∣ (l * (l + 1) * (l + 2) * (l + 3)))) := by
  sorry

end greatest_divisor_four_consecutive_integers_l1854_185431


namespace polynomial_expansion_l1854_185413

theorem polynomial_expansion (x : ℝ) :
  (7 * x^2 + 3) * (5 * x^3 + 4 * x + 1) = 35 * x^5 + 43 * x^3 + 7 * x^2 + 12 * x + 3 := by
  sorry

end polynomial_expansion_l1854_185413


namespace largest_selected_is_57_l1854_185421

/-- Represents the systematic sampling of students. -/
structure StudentSampling where
  total_students : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the sample interval based on the first two selected numbers. -/
def sample_interval (s : StudentSampling) : Nat :=
  s.second_selected - s.first_selected

/-- Calculates the number of selected students. -/
def num_selected (s : StudentSampling) : Nat :=
  s.total_students / sample_interval s

/-- Calculates the largest selected number. -/
def largest_selected (s : StudentSampling) : Nat :=
  s.first_selected + (sample_interval s) * (num_selected s - 1)

/-- Theorem stating that the largest selected number is 57 for the given conditions. -/
theorem largest_selected_is_57 (s : StudentSampling) 
    (h1 : s.total_students = 60)
    (h2 : s.first_selected = 3)
    (h3 : s.second_selected = 9) : 
  largest_selected s = 57 := by
  sorry

#eval largest_selected { total_students := 60, first_selected := 3, second_selected := 9 }

end largest_selected_is_57_l1854_185421


namespace jerry_makes_two_trips_l1854_185492

def jerry_trips (carry_capacity : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + carry_capacity - 1) / carry_capacity

theorem jerry_makes_two_trips (carry_capacity : ℕ) (total_trays : ℕ) :
  carry_capacity = 8 → total_trays = 16 → jerry_trips carry_capacity total_trays = 2 := by
  sorry

end jerry_makes_two_trips_l1854_185492


namespace bread_cost_is_1_1_l1854_185456

/-- The cost of each bread given the conditions of the problem -/
def bread_cost (total_breads : ℕ) (num_people : ℕ) (compensation : ℚ) : ℚ :=
  (compensation * 2 * num_people) / total_breads

/-- Theorem stating that the cost of each bread is 1.1 yuan -/
theorem bread_cost_is_1_1 :
  bread_cost 12 3 (22/10) = 11/10 := by
  sorry

end bread_cost_is_1_1_l1854_185456


namespace toms_trip_speed_l1854_185447

/-- Proves that given the conditions of Tom's trip, his speed during the first part was 20 mph -/
theorem toms_trip_speed : 
  ∀ (v : ℝ),
  (v > 0) →
  (50 / v + 1 > 0) →
  (100 / (50 / v + 1) = 28.571428571428573) →
  v = 20 := by
  sorry

end toms_trip_speed_l1854_185447


namespace roosevelt_bonus_points_l1854_185484

/-- Represents the points scored by Roosevelt High School in each game and the bonus points received --/
structure RooseveltPoints where
  first_game : ℕ
  second_game : ℕ
  third_game : ℕ
  bonus : ℕ

/-- Represents the total points scored by Greendale High School --/
def greendale_points : ℕ := 130

/-- Calculates the total points scored by Roosevelt High School before bonus --/
def roosevelt_total (p : RooseveltPoints) : ℕ :=
  p.first_game + p.second_game + p.third_game

/-- Theorem stating the bonus points received by Roosevelt High School --/
theorem roosevelt_bonus_points :
  ∀ p : RooseveltPoints,
  p.first_game = 30 →
  p.second_game = p.first_game / 2 →
  p.third_game = p.second_game * 3 →
  greendale_points = roosevelt_total p + p.bonus →
  p.bonus = 40 := by
  sorry

end roosevelt_bonus_points_l1854_185484


namespace smallest_area_squared_l1854_185437

/-- A regular hexagon ABCDEF with side length 10 inscribed in a circle ω -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 10)

/-- Points X, Y, Z on minor arcs AB, CD, EF respectively -/
structure TriangleXYZ (h : RegularHexagon) :=
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (Z : ℝ × ℝ)
  (X_on_AB : True)  -- Placeholder for the condition that X is on minor arc AB
  (Y_on_CD : True)  -- Placeholder for the condition that Y is on minor arc CD
  (Z_on_EF : True)  -- Placeholder for the condition that Z is on minor arc EF

/-- The area of triangle XYZ -/
def triangle_area (h : RegularHexagon) (t : TriangleXYZ h) : ℝ :=
  sorry  -- Definition of triangle area

/-- The theorem stating the smallest possible area squared -/
theorem smallest_area_squared (h : RegularHexagon) :
  ∃ (t : TriangleXYZ h), ∀ (t' : TriangleXYZ h), (triangle_area h t)^2 ≤ (triangle_area h t')^2 ∧ (triangle_area h t)^2 = 7500 :=
sorry

end smallest_area_squared_l1854_185437


namespace matrix_multiplication_result_l1854_185401

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by sorry

end matrix_multiplication_result_l1854_185401


namespace abs_negative_eight_l1854_185403

theorem abs_negative_eight : |(-8 : ℤ)| = 8 := by sorry

end abs_negative_eight_l1854_185403


namespace number_wall_value_l1854_185422

/-- Represents a simplified Number Wall with four bottom values and a top value --/
structure NumberWall where
  bottom_left : ℕ
  bottom_mid_left : ℕ
  bottom_mid_right : ℕ
  bottom_right : ℕ
  top : ℕ

/-- The Number Wall is valid if it follows the construction rules --/
def is_valid_number_wall (w : NumberWall) : Prop :=
  ∃ (mid_left mid_right : ℕ),
    w.bottom_left + w.bottom_mid_left = mid_left ∧
    w.bottom_mid_left + w.bottom_mid_right = mid_right ∧
    w.bottom_mid_right + w.bottom_right = w.top - mid_left ∧
    mid_left + mid_right = w.top

theorem number_wall_value (w : NumberWall) 
    (h : is_valid_number_wall w)
    (h1 : w.bottom_mid_left = 6)
    (h2 : w.bottom_mid_right = 10)
    (h3 : w.bottom_right = 9)
    (h4 : w.top = 64) :
  w.bottom_left = 7 := by
  sorry

end number_wall_value_l1854_185422


namespace simplify_expression_l1854_185434

theorem simplify_expression (x : ℝ) : x * (4 * x^2 - 3) - 6 * (x^2 - 3*x + 8) = 4 * x^3 - 6 * x^2 + 15 * x - 48 := by
  sorry

end simplify_expression_l1854_185434


namespace leo_statement_true_only_on_tuesday_l1854_185450

-- Define the days of the week
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

-- Define Leo's lying pattern
def lies_on_day (d : Day) : Prop :=
  match d with
  | Day.monday => True
  | Day.tuesday => True
  | Day.wednesday => True
  | _ => False

-- Define the 'yesterday' and 'tomorrow' functions
def yesterday (d : Day) : Day :=
  match d with
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday
  | Day.sunday => Day.saturday

def tomorrow (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

-- Define Leo's statement
def leo_statement (d : Day) : Prop :=
  lies_on_day (yesterday d) ∧ lies_on_day (tomorrow d)

-- Theorem: Leo's statement is true only on Tuesday
theorem leo_statement_true_only_on_tuesday :
  ∀ (d : Day), leo_statement d ↔ d = Day.tuesday :=
by sorry

end leo_statement_true_only_on_tuesday_l1854_185450


namespace point_B_coordinates_l1854_185417

def point_A : ℝ × ℝ := (-1, -5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  let vector_AB : ℝ × ℝ := (3 * vector_a.1, 3 * vector_a.2)
  let point_B : ℝ × ℝ := (point_A.1 + vector_AB.1, point_A.2 + vector_AB.2)
  point_B = (5, 4) := by sorry

end point_B_coordinates_l1854_185417


namespace total_pamphlets_printed_prove_total_pamphlets_l1854_185483

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
theorem total_pamphlets_printed (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) 
  (mike_additional_hours : ℕ) (leo_speed_multiplier : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_reduced_speed := mike_initial_speed / 3
  let mike_additional_pamphlets := mike_reduced_speed * mike_additional_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * leo_speed_multiplier
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_additional_pamphlets + leo_pamphlets

/-- Proves that Mike and Leo print 9400 pamphlets in total -/
theorem prove_total_pamphlets : total_pamphlets_printed 600 9 2 2 = 9400 := by
  sorry

end total_pamphlets_printed_prove_total_pamphlets_l1854_185483


namespace partial_fraction_decomposition_l1854_185475

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
  (6 * x + 3) / (x^2 - 8 * x - 48) = (75 / 16) / (x - 12) + (21 / 16) / (x + 4) := by
  sorry

end partial_fraction_decomposition_l1854_185475


namespace problem_paths_l1854_185415

/-- Represents the number of ways to reach a specific arrow type -/
structure ArrowPaths where
  count : Nat
  arrows : Nat

/-- The modified hexagonal lattice structure -/
structure HexLattice where
  redPaths : ArrowPaths
  bluePaths : ArrowPaths
  greenPaths : ArrowPaths
  endPaths : Nat

/-- The specific hexagonal lattice in the problem -/
def problemLattice : HexLattice :=
  { redPaths := { count := 1, arrows := 1 }
    bluePaths := { count := 3, arrows := 2 }
    greenPaths := { count := 6, arrows := 2 }
    endPaths := 4 }

/-- Calculates the total number of paths in the lattice -/
def totalPaths (lattice : HexLattice) : Nat :=
  lattice.redPaths.count *
  lattice.bluePaths.count * lattice.bluePaths.arrows *
  lattice.greenPaths.count * lattice.greenPaths.arrows *
  lattice.endPaths

theorem problem_paths :
  totalPaths problemLattice = 288 := by sorry

end problem_paths_l1854_185415


namespace smallest_angle_in_ratio_triangle_l1854_185463

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = 6/5 * a) →
  (c = 7/5 * a) →
  a = 50 :=
by sorry

end smallest_angle_in_ratio_triangle_l1854_185463


namespace mountain_climb_speeds_l1854_185481

theorem mountain_climb_speeds (V₁ V₂ V k m n : ℝ) 
  (hpos : V₁ > 0 ∧ V₂ > 0 ∧ V > 0 ∧ k > 0 ∧ m > 0 ∧ n > 0)
  (hV₂ : V₂ = k * V₁)
  (hVm : V = m * V₁)
  (hVn : V = n * V₂) : 
  m = 2 * k / (1 + k) ∧ m + n = 2 := by
  sorry

end mountain_climb_speeds_l1854_185481


namespace f_minimum_value_l1854_185428

def f (x : ℝ) : ℝ := |x - 1| + |x - 2| - |x - 3|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ -1) ∧ (∃ x : ℝ, f x = -1) :=
sorry

end f_minimum_value_l1854_185428
