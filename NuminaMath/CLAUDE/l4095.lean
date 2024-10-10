import Mathlib

namespace mikes_lawn_mowing_earnings_l4095_409559

/-- Proves that Mike's total earnings from mowing lawns is $101 given the conditions --/
theorem mikes_lawn_mowing_earnings : 
  ∀ (total_earnings : ℕ) 
    (mower_blades_cost : ℕ) 
    (num_games : ℕ) 
    (game_cost : ℕ),
  mower_blades_cost = 47 →
  num_games = 9 →
  game_cost = 6 →
  total_earnings = mower_blades_cost + num_games * game_cost →
  total_earnings = 101 := by
sorry

end mikes_lawn_mowing_earnings_l4095_409559


namespace quadratic_is_perfect_square_l4095_409572

theorem quadratic_is_perfect_square (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + a = (b*x + c)^2) → a = 16 :=
by sorry

end quadratic_is_perfect_square_l4095_409572


namespace bench_arrangements_l4095_409515

theorem bench_arrangements (n : Nat) (h : n = 9) : Nat.factorial n = 362880 := by
  sorry

end bench_arrangements_l4095_409515


namespace polynomial_simplification_l4095_409596

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) =
  -2 * y^3 + y^2 + 10 * y + 3 := by
  sorry

end polynomial_simplification_l4095_409596


namespace small_triangle_perimeter_l4095_409565

/-- Given a triangle and three trapezoids formed by cuts parallel to its sides,
    this theorem proves the perimeter of the resulting small triangle. -/
theorem small_triangle_perimeter
  (original_perimeter : ℝ)
  (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
  (h1 : original_perimeter = 11)
  (h2 : trapezoid1_perimeter = 5)
  (h3 : trapezoid2_perimeter = 7)
  (h4 : trapezoid3_perimeter = 9) :
  trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter - original_perimeter = 10 :=
by sorry

end small_triangle_perimeter_l4095_409565


namespace johns_investment_l4095_409518

theorem johns_investment (total_investment : ℝ) (rate_a rate_b : ℝ) (investment_a : ℝ) (final_amount : ℝ) :
  total_investment = 1500 →
  rate_a = 0.04 →
  rate_b = 0.06 →
  investment_a = 750 →
  final_amount = 1575 →
  investment_a * (1 + rate_a) + (total_investment - investment_a) * (1 + rate_b) = final_amount :=
by sorry

end johns_investment_l4095_409518


namespace quadratic_function_property_l4095_409501

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 1 = f 3 ∧ f 1 > f 4) → (a < 0 ∧ 4 * a + b = 0) := by
  sorry

end quadratic_function_property_l4095_409501


namespace no_simultaneous_roots_one_and_neg_one_l4095_409505

theorem no_simultaneous_roots_one_and_neg_one :
  ¬ ∃ (a b : ℝ), (1 : ℝ)^3 + a * (1 : ℝ)^2 + b = 0 ∧ (-1 : ℝ)^3 + a * (-1 : ℝ)^2 + b = 0 :=
by sorry

end no_simultaneous_roots_one_and_neg_one_l4095_409505


namespace train_length_l4095_409586

/-- The length of a train given specific conditions. -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 200 := by
sorry

end train_length_l4095_409586


namespace cos_120_degrees_l4095_409581

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1 / 2 := by
  sorry

end cos_120_degrees_l4095_409581


namespace calculate_expression_l4095_409561

theorem calculate_expression : -(-1) + 3^2 / (1 - 4) * 2 = -5 := by
  sorry

end calculate_expression_l4095_409561


namespace binomial_expansion_97_cubed_l4095_409571

theorem binomial_expansion_97_cubed : 97^3 + 3*(97^2) + 3*97 + 1 = 941192 := by
  sorry

end binomial_expansion_97_cubed_l4095_409571


namespace jessica_remaining_money_l4095_409527

/-- Given Jessica's initial amount and spending, calculate the remaining amount -/
theorem jessica_remaining_money (initial : ℚ) (spent : ℚ) (remaining : ℚ) : 
  initial = 11.73 ∧ spent = 10.22 ∧ remaining = initial - spent → remaining = 1.51 := by
  sorry

end jessica_remaining_money_l4095_409527


namespace sum_of_multiples_l4095_409592

theorem sum_of_multiples (n : ℕ) (h : n = 13) : n + n + 2 * n + 4 * n = 104 := by
  sorry

end sum_of_multiples_l4095_409592


namespace minimum_width_for_garden_l4095_409577

-- Define the garden width as a real number
variable (w : ℝ)

-- Define the conditions of the problem
def garden_length (w : ℝ) : ℝ := w + 10
def garden_area (w : ℝ) : ℝ := w * garden_length w
def area_constraint (w : ℝ) : Prop := garden_area w ≥ 150

-- Theorem statement
theorem minimum_width_for_garden :
  (∀ x : ℝ, x > 0 → area_constraint x → x ≥ 10) ∧ area_constraint 10 :=
sorry

end minimum_width_for_garden_l4095_409577


namespace binomial_coefficient_19_13_l4095_409516

theorem binomial_coefficient_19_13 (h1 : Nat.choose 18 11 = 31824)
                                   (h2 : Nat.choose 18 12 = 18564)
                                   (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end binomial_coefficient_19_13_l4095_409516


namespace circular_field_diameter_l4095_409576

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter 
  (cost_per_meter : ℝ) 
  (total_cost : ℝ) 
  (h_cost : cost_per_meter = 3) 
  (h_total : total_cost = 395.84067435231395) : 
  ∃ (diameter : ℝ), abs (diameter - 42) < 0.00001 := by
  sorry

end circular_field_diameter_l4095_409576


namespace curve_not_parabola_l4095_409549

-- Define the curve equation
def curve_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x^2 + y^2 = 1

-- Define what it means for a curve to be a parabola
-- (This is a simplified definition for the purpose of this statement)
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem curve_not_parabola :
  ∀ k : ℝ, ¬(is_parabola (curve_equation k)) :=
sorry

end curve_not_parabola_l4095_409549


namespace mod_equivalence_l4095_409511

theorem mod_equivalence (n : ℕ) (h1 : n < 41) (h2 : (5 * n) % 41 = 1) :
  (((2 ^ n) ^ 3) - 3) % 41 = 6 := by
  sorry

end mod_equivalence_l4095_409511


namespace division_problem_l4095_409509

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5/2) :
  z / x = 2/15 := by
sorry

end division_problem_l4095_409509


namespace diamond_15_25_l4095_409539

-- Define the ⋄ operation
noncomputable def diamond (x y : ℝ) : ℝ := 
  sorry

-- Define the properties of the ⋄ operation
axiom diamond_prop1 (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  diamond (x * y) y = x * (diamond y y)

axiom diamond_prop2 (x : ℝ) (hx : x > 0) : 
  diamond (diamond x 1) x = diamond x 1

axiom diamond_one : diamond 1 1 = 1

-- State the theorem to be proved
theorem diamond_15_25 : diamond 15 25 = 375 := by
  sorry

end diamond_15_25_l4095_409539


namespace cubic_expansion_property_l4095_409574

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end cubic_expansion_property_l4095_409574


namespace sum_of_distinct_prime_factors_l4095_409528

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → p ∣ (7^7 - 7^4) ↔ p ∈ s) ∧
  (s.sum id = 24) := by
sorry

end sum_of_distinct_prime_factors_l4095_409528


namespace fraction_transformation_l4095_409526

theorem fraction_transformation (x y : ℝ) : 
  -(x - y) / (x + y) = (-x + y) / (x + y) :=
by sorry

end fraction_transformation_l4095_409526


namespace trivia_team_tryouts_l4095_409591

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  not_picked + groups * students_per_group = 65 :=
by sorry

end trivia_team_tryouts_l4095_409591


namespace banana_permutations_count_l4095_409552

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_count :
  banana_permutations = 60 := by
  sorry

end banana_permutations_count_l4095_409552


namespace petes_number_l4095_409521

theorem petes_number (x : ℚ) : 3 * (x + 15) - 5 = 125 → x = 85 / 3 := by
  sorry

end petes_number_l4095_409521


namespace multiply_98_squared_l4095_409579

theorem multiply_98_squared : 98 * 98 = 9604 := by
  sorry

end multiply_98_squared_l4095_409579


namespace congruence_problem_l4095_409587

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end congruence_problem_l4095_409587


namespace log_expression_eval_l4095_409599

-- Define lg as base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the fourth root
noncomputable def fourthRoot (x : ℝ) := Real.rpow x (1/4)

theorem log_expression_eval :
  Real.log (fourthRoot 27 / 3) / Real.log 3 + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) = 15/4 := by
  sorry

end log_expression_eval_l4095_409599


namespace solar_usage_exponential_growth_l4095_409530

/-- Represents the percentage of households using solar energy -/
def SolarUsage : ℕ → ℝ
  | 2000 => 6
  | 2010 => 12
  | 2015 => 24
  | 2020 => 48
  | _ => 0  -- For years not specified, we return 0

/-- Checks if the growth is exponential between two time points -/
def IsExponentialGrowth (t₁ t₂ : ℕ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ SolarUsage t₂ = SolarUsage t₁ * r^(t₂ - t₁)

/-- Theorem stating that the solar usage growth is exponential -/
theorem solar_usage_exponential_growth :
  IsExponentialGrowth 2000 2010 ∧
  IsExponentialGrowth 2010 2015 ∧
  IsExponentialGrowth 2015 2020 :=
sorry

end solar_usage_exponential_growth_l4095_409530


namespace power_division_rule_l4095_409570

theorem power_division_rule (x : ℝ) : x^4 / x = x^3 := by sorry

end power_division_rule_l4095_409570


namespace kolya_is_collection_agency_l4095_409569

-- Define the actors in the scenario
structure Person :=
  (name : String)

-- Define the book lending scenario
structure BookLendingScenario :=
  (lender : Person)
  (borrower : Person)
  (collector : Person)
  (books_lent : ℕ)
  (return_promised : Bool)
  (books_returned : Bool)
  (collector_fee : ℕ)

-- Define the characteristics of a collection agency
structure CollectionAgency :=
  (collects_items : Bool)
  (acts_on_behalf : Bool)
  (receives_fee : Bool)

-- Define Kolya's role in the scenario
def kolya_role (scenario : BookLendingScenario) : CollectionAgency :=
  { collects_items := true
  , acts_on_behalf := true
  , receives_fee := scenario.collector_fee > 0 }

-- Theorem statement
theorem kolya_is_collection_agency (scenario : BookLendingScenario) : 
  kolya_role scenario = CollectionAgency.mk true true true :=
sorry

end kolya_is_collection_agency_l4095_409569


namespace complement_intersection_equal_set_l4095_409558

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_equal_set : (U \ M) ∩ N = {-3, -4} := by
  sorry

end complement_intersection_equal_set_l4095_409558


namespace problem_one_problem_two_problem_three_l4095_409588

-- Problem 1
theorem problem_one (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 := by sorry

-- Problem 2
theorem problem_two (α : Real) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by sorry

-- Problem 3
theorem problem_three (α : Real) (h1 : Real.sin α + Real.cos α = 1/2) (h2 : 0 < α) (h3 : α < π) :
  Real.sin α * Real.cos α = -3/8 := by sorry

end problem_one_problem_two_problem_three_l4095_409588


namespace fifth_term_product_l4095_409566

/-- Given an arithmetic sequence a and a geometric sequence b with specified initial terms,
    prove that the product of their 5th terms is 80. -/
theorem fifth_term_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 = 1 → b 1 = 1 → 
  a 2 = 2 → b 2 = 2 → 
  a 5 * b 5 = 80 := by
sorry


end fifth_term_product_l4095_409566


namespace school_play_scenes_l4095_409553

theorem school_play_scenes (Tom Ben Sam Nick Chris : ℕ) : 
  Tom = 8 ∧ Chris = 5 ∧ 
  Ben > Chris ∧ Ben < Tom ∧
  Sam > Chris ∧ Sam < Tom ∧
  Nick > Chris ∧ Nick < Tom ∧
  (∀ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) ∧
  (∀ pair : ℕ × ℕ, pair.1 ≠ pair.2 → pair.1 ≤ 5 ∧ pair.2 ≤ 5 → 
    ∃ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) →
  (Tom + Ben + Sam + Nick + Chris) / 2 = 16 := by
sorry

end school_play_scenes_l4095_409553


namespace female_students_count_l4095_409542

theorem female_students_count (total : ℕ) (sample : ℕ) (female_diff : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_female_diff : female_diff = 20)
  (h_ratio : (sample - female_diff) / (sample + female_diff) = 9 / 11) :
  ∃ F : ℕ, F = 720 ∧ F + (total - F) = total := by
  sorry

end female_students_count_l4095_409542


namespace original_sales_tax_percentage_l4095_409514

theorem original_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) :
  market_price = 8400 ∧ 
  new_tax_rate = 10 / 3 ∧ 
  savings = 14 → 
  ∃ original_tax_rate : ℝ,
    original_tax_rate = 3.5 ∧
    market_price * (original_tax_rate / 100) = 
      market_price * (new_tax_rate / 100) + savings :=
by sorry

end original_sales_tax_percentage_l4095_409514


namespace mod_eight_equivalence_l4095_409510

theorem mod_eight_equivalence (m : ℕ) : 
  13^7 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end mod_eight_equivalence_l4095_409510


namespace dividend_calculation_l4095_409537

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 20 * quotient →
  divisor = 10 * remainder →
  remainder = 100 →
  divisor * quotient + remainder = 50100 :=
by
  sorry

end dividend_calculation_l4095_409537


namespace statement_equivalence_l4095_409548

theorem statement_equivalence (x y : ℝ) : 
  ((abs y < abs x) ↔ (x^2 > y^2)) ∧ 
  ((x^3 - y^3 = 0) ↔ (x - y = 0)) ∧ 
  ((x^3 - y^3 ≠ 0) ↔ (x - y ≠ 0)) ∧ 
  ¬((x^2 - y^2 ≠ 0 ∧ x^3 - y^3 ≠ 0) ↔ (x^2 - y^2 ≠ 0 ∨ x^3 - y^3 ≠ 0)) := by
  sorry

end statement_equivalence_l4095_409548


namespace brand_x_pen_price_l4095_409507

/-- The price of a brand X pen given the total number of pens, total cost, number of brand X pens, and price of brand Y pens. -/
theorem brand_x_pen_price
  (total_pens : ℕ)
  (total_cost : ℚ)
  (brand_x_count : ℕ)
  (brand_y_price : ℚ)
  (h1 : total_pens = 12)
  (h2 : total_cost = 42)
  (h3 : brand_x_count = 6)
  (h4 : brand_y_price = 2.2)
  : (total_cost - (total_pens - brand_x_count) * brand_y_price) / brand_x_count = 4.8 := by
  sorry

#check brand_x_pen_price

end brand_x_pen_price_l4095_409507


namespace union_p_complement_q_l4095_409506

-- Define the set P
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 4*x - 3)}

-- Define the set Q
def Q : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem union_p_complement_q :
  P ∪ (Set.univ \ Q) = Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end union_p_complement_q_l4095_409506


namespace product_pure_imaginary_solution_l4095_409536

theorem product_pure_imaginary_solution (x : ℝ) : 
  (∃ y : ℝ, (x + 2 * Complex.I) * ((x + 1) + 2 * Complex.I) * ((x + 2) + 2 * Complex.I) = y * Complex.I) ↔ 
  x = 1 :=
by sorry

end product_pure_imaginary_solution_l4095_409536


namespace bisected_polyhedron_edges_l4095_409544

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the new polyhedron after bisection -/
structure BisectedPolyhedron where
  original : ConvexPolyhedron
  planes : ℕ

/-- Calculate the number of edges in the bisected polyhedron -/
def edges_after_bisection (T : BisectedPolyhedron) : ℕ :=
  T.original.edges + 2 * T.original.edges

/-- Theorem stating the number of edges in the bisected polyhedron -/
theorem bisected_polyhedron_edges 
  (P : ConvexPolyhedron) 
  (h_vertices : P.vertices = 0)  -- placeholder for actual number of vertices
  (h_edges : P.edges = 150)
  (T : BisectedPolyhedron)
  (h_T : T.original = P)
  (h_planes : T.planes = P.vertices)
  : edges_after_bisection T = 450 := by
  sorry

#check bisected_polyhedron_edges

end bisected_polyhedron_edges_l4095_409544


namespace log_relationship_l4095_409583

theorem log_relationship (a b : ℝ) : 
  a = (Real.log 125) / (Real.log 4) → 
  b = (Real.log 32) / (Real.log 5) → 
  a = (3 * b) / 10 := by
sorry

end log_relationship_l4095_409583


namespace smallest_positive_multiple_of_48_l4095_409594

theorem smallest_positive_multiple_of_48 :
  ∀ n : ℕ, n > 0 → 48 ∣ n → n ≥ 48 :=
by
  sorry

end smallest_positive_multiple_of_48_l4095_409594


namespace cryptarithm_solution_l4095_409534

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Fin 10

/-- Check if a mapping is valid for the cryptarithm -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m 'Г' ≠ 0 ∧
  m 'О' ≠ 0 ∧
  m 'В' ≠ 0 ∧
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧
  (m 'Г' * 1000 + m 'О' * 100 + m 'Р' * 10 + m 'А') +
  (m 'О' * 10000 + m 'Г' * 1000 + m 'О' * 100 + m 'Н' * 10 + m 'Ь') =
  (m 'В' * 100000 + m 'У' * 10000 + m 'Л' * 1000 + m 'К' * 100 + m 'А' * 10 + m 'Н')

theorem cryptarithm_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧
    m 'Г' = 6 ∧ m 'О' = 9 ∧ m 'Р' = 4 ∧ m 'А' = 7 ∧
    m 'Н' = 2 ∧ m 'Ь' = 5 ∧
    m 'В' = 1 ∧ m 'У' = 0 ∧ m 'Л' = 3 ∧ m 'К' = 8 :=
by sorry

end cryptarithm_solution_l4095_409534


namespace championship_probability_l4095_409517

def is_win_for_A (n : ℕ) : Bool :=
  n ≤ 5

def count_wins_A (numbers : List ℕ) : ℕ :=
  (numbers.filter is_win_for_A).length

def estimate_probability (wins : ℕ) (total : ℕ) : ℚ :=
  ↑wins / ↑total

def generated_numbers : List ℕ := [1, 9, 2, 9, 0, 7, 9, 6, 6, 9, 2, 5, 2, 7, 1, 9, 3, 2, 8, 1, 2, 6, 7, 3, 9, 3, 1, 2, 7, 5, 5, 6, 4, 8, 8, 7, 3, 0, 1, 1, 3, 5, 3, 7, 9, 8, 9, 4, 3, 1]

theorem championship_probability :
  estimate_probability (count_wins_A generated_numbers) generated_numbers.length = 13/20 := by
  sorry

end championship_probability_l4095_409517


namespace paint_more_expensive_than_wallpaper_l4095_409557

/-- Proves that a can of paint costs more than a roll of wallpaper given specific purchase scenarios -/
theorem paint_more_expensive_than_wallpaper 
  (wallpaper_cost movie_ticket_cost paint_cost : ℝ) 
  (wallpaper_cost_positive : 0 < wallpaper_cost)
  (paint_cost_positive : 0 < paint_cost)
  (movie_ticket_cost_positive : 0 < movie_ticket_cost)
  (equal_spending : 4 * wallpaper_cost + 4 * paint_cost = 7 * wallpaper_cost + 2 * paint_cost + movie_ticket_cost) :
  paint_cost > wallpaper_cost := by
  sorry


end paint_more_expensive_than_wallpaper_l4095_409557


namespace simplify_expression_1_simplify_expression_2_l4095_409546

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end simplify_expression_1_simplify_expression_2_l4095_409546


namespace parallel_vectors_imply_x_value_l4095_409504

/-- Given two vectors a and b in R², prove that if 2a is parallel to b, then the x-coordinate of b is -4. -/
theorem parallel_vectors_imply_x_value (a b : ℝ × ℝ) : 
  a = (2, 3) → 
  b.2 = -6 → 
  ∃ (k : ℝ), (2 * a.1, 2 * a.2) = (k * b.1, k * b.2) → 
  b.1 = -4 := by
  sorry

end parallel_vectors_imply_x_value_l4095_409504


namespace remainder_theorem_l4095_409543

theorem remainder_theorem : 
  (86592 : ℤ) % 8 = 0 ∧ (8741 : ℤ) % 13 = 5 := by
  sorry

end remainder_theorem_l4095_409543


namespace y_coordinate_of_first_point_l4095_409590

/-- Given a line with equation x = 2y + 5 passing through points (m, n) and (m + 5, n + 2.5),
    prove that the y-coordinate of the first point (n) is equal to (m - 5)/2. -/
theorem y_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := by
  sorry

end y_coordinate_of_first_point_l4095_409590


namespace intersection_theorem_l4095_409551

def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℝ := {x | x + 1 < 0}

theorem intersection_theorem :
  A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by sorry

end intersection_theorem_l4095_409551


namespace biology_marks_calculation_l4095_409535

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem biology_marks_calculation : 
  (english_marks + math_marks + physics_marks + chemistry_marks + 
   (average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks))) 
  / total_subjects = average_marks :=
by sorry

end biology_marks_calculation_l4095_409535


namespace escalator_walking_speed_l4095_409532

/-- Proves that given an escalator moving at 12 ft/sec with a length of 160 feet,
    if a person covers the entire length in 8 seconds,
    then the person's walking speed on the escalator is 8 ft/sec. -/
theorem escalator_walking_speed
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 160)
  (h3 : time_taken = 8)
  (h4 : escalator_length = (person_speed + escalator_speed) * time_taken) :
  person_speed = 8 := by
  sorry

#check escalator_walking_speed

end escalator_walking_speed_l4095_409532


namespace production_average_l4095_409568

/-- Proves that given the conditions in the problem, n = 1 --/
theorem production_average (n : ℕ) : 
  (n * 50 + 60) / (n + 1) = 55 → n = 1 := by
  sorry


end production_average_l4095_409568


namespace tan_period_l4095_409519

/-- The period of tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  (fun x => Real.tan ((3 : ℝ) * x / 4)) = (fun x => Real.tan ((3 : ℝ) * (x + 4 * Real.pi / 3) / 4)) :=
by sorry

end tan_period_l4095_409519


namespace cubic_function_coefficient_l4095_409538

/-- Given a function f(x) = ax^3 - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem cubic_function_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end cubic_function_coefficient_l4095_409538


namespace number_of_payment_ways_l4095_409545

/-- Represents the number of ways to pay 16 rubles using 10-ruble, 2-ruble, and 1-ruble coins. -/
def payment_ways : ℕ := 13

/-- Represents the total amount to be paid in rubles. -/
def total_amount : ℕ := 16

/-- Represents the value of a 10-ruble coin. -/
def ten_ruble : ℕ := 10

/-- Represents the value of a 2-ruble coin. -/
def two_ruble : ℕ := 2

/-- Represents the value of a 1-ruble coin. -/
def one_ruble : ℕ := 1

/-- Represents the minimum number of coins of each type available. -/
def min_coins : ℕ := 21

/-- Theorem stating that the number of ways to pay 16 rubles is 13. -/
theorem number_of_payment_ways :
  payment_ways = (Finset.filter
    (fun n : ℕ × ℕ × ℕ => n.1 * ten_ruble + n.2.1 * two_ruble + n.2.2 * one_ruble = total_amount)
    (Finset.product (Finset.range (min_coins + 1))
      (Finset.product (Finset.range (min_coins + 1)) (Finset.range (min_coins + 1))))).card :=
by sorry

end number_of_payment_ways_l4095_409545


namespace function_symmetry_implies_a_value_l4095_409564

theorem function_symmetry_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → a = 4 := by
  sorry

end function_symmetry_implies_a_value_l4095_409564


namespace stack_height_three_pipes_l4095_409554

/-- The height of a stack of identical cylindrical pipes -/
def stack_height (num_pipes : ℕ) (pipe_diameter : ℝ) : ℝ :=
  (num_pipes : ℝ) * pipe_diameter

/-- Theorem: The height of a stack of three identical cylindrical pipes with a diameter of 12 cm is 36 cm -/
theorem stack_height_three_pipes :
  stack_height 3 12 = 36 := by
  sorry

end stack_height_three_pipes_l4095_409554


namespace science_team_selection_l4095_409541

def number_of_boys : ℕ := 7
def number_of_girls : ℕ := 9
def boys_in_team : ℕ := 2
def girls_in_team : ℕ := 3

theorem science_team_selection :
  (number_of_boys.choose boys_in_team) * (number_of_girls.choose girls_in_team) = 1764 := by
  sorry

end science_team_selection_l4095_409541


namespace complex_fraction_simplification_l4095_409512

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * i
  let z₂ : ℂ := 4 - 7 * i
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end complex_fraction_simplification_l4095_409512


namespace B_power_15_minus_3_power_14_l4095_409529

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end B_power_15_minus_3_power_14_l4095_409529


namespace abc_inequality_l4095_409523

theorem abc_inequality (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) > 
  a^(α + 2) * (b + c - a) + b^(α + 2) * (a - b + c) + c^(α + 2) * (a + b - c) := by
  sorry

end abc_inequality_l4095_409523


namespace adele_age_fraction_l4095_409533

/-- Given the ages of Jackson, Mandy, and Adele, prove that Adele's age is 3/4 of Jackson's age. -/
theorem adele_age_fraction (jackson_age mandy_age adele_age : ℕ) : 
  jackson_age = 20 →
  mandy_age = jackson_age + 10 →
  (jackson_age + 10) + (mandy_age + 10) + (adele_age + 10) = 95 →
  ∃ f : ℚ, adele_age = f * jackson_age ∧ f = 3/4 := by
  sorry

end adele_age_fraction_l4095_409533


namespace solution_of_system_l4095_409567

theorem solution_of_system (x y : ℝ) : 
  (1 / Real.sqrt (1 + 2 * x^2) + 1 / Real.sqrt (1 + 2 * y^2) = 2 / Real.sqrt (1 + 2 * x * y)) ∧
  (Real.sqrt (x * (1 - 2 * x)) + Real.sqrt (y * (1 - 2 * y)) = 2 / 9) →
  (x = y) ∧ 
  ((x = 1 / 4 + Real.sqrt 73 / 36) ∨ (x = 1 / 4 - Real.sqrt 73 / 36)) :=
by sorry

end solution_of_system_l4095_409567


namespace number_of_factors_24_l4095_409540

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem number_of_factors_24 : (factors 24).card = 8 := by
  sorry

end number_of_factors_24_l4095_409540


namespace outdoor_scouts_hike_l4095_409595

theorem outdoor_scouts_hike (cars taxis vans buses : ℕ) 
  (people_per_car people_per_taxi people_per_van people_per_bus : ℕ) :
  cars = 5 →
  taxis = 8 →
  vans = 3 →
  buses = 2 →
  people_per_car = 4 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  people_per_bus = 20 →
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van + buses * people_per_bus = 123 :=
by
  sorry

#check outdoor_scouts_hike

end outdoor_scouts_hike_l4095_409595


namespace sara_movie_rental_cost_l4095_409597

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let known_expenses : ℚ := theater_total + bought_movie_price
  let rental_cost : ℚ := total_spent - known_expenses
  rental_cost = 159/100

theorem sara_movie_rental_cost :
  movie_spending (1062/100) 2 (1395/100) (3678/100) :=
sorry

end sara_movie_rental_cost_l4095_409597


namespace sin_product_equality_l4095_409598

theorem sin_product_equality : 
  Real.sin (3 * π / 180) * Real.sin (39 * π / 180) * Real.sin (63 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end sin_product_equality_l4095_409598


namespace matrix_equation_proof_l4095_409582

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![3, -8; 4, -11]
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![4/13, -31/13; 5/13, -42/13]
  P * A = B := by sorry

end matrix_equation_proof_l4095_409582


namespace production_days_calculation_l4095_409593

theorem production_days_calculation (n : ℕ) : 
  (n * 50 + 90) / (n + 1) = 52 → n = 19 := by sorry

end production_days_calculation_l4095_409593


namespace ascending_order_proof_l4095_409550

theorem ascending_order_proof :
  222^2 < 2^(2^(2^2)) ∧
  2^(2^(2^2)) < 22^(2^2) ∧
  22^(2^2) < 22^22 ∧
  22^22 < 2^222 ∧
  2^222 < 2^(22^2) ∧
  2^(22^2) < 2^(2^22) := by
  sorry

end ascending_order_proof_l4095_409550


namespace trigonometric_calculations_l4095_409502

open Real

theorem trigonometric_calculations :
  (sin (-60 * π / 180) = -Real.sqrt 3 / 2) ∧
  (cos (-45 * π / 180) = Real.sqrt 2 / 2) ∧
  (tan (-945 * π / 180) = -1) := by
  sorry

-- Definitions and properties used in the proof
axiom sine_odd (x : ℝ) : sin (-x) = -sin x
axiom cosine_even (x : ℝ) : cos (-x) = cos x
axiom tangent_odd (x : ℝ) : tan (-x) = -tan x
axiom sin_60 : sin (60 * π / 180) = Real.sqrt 3 / 2
axiom cos_45 : cos (45 * π / 180) = Real.sqrt 2 / 2
axiom tan_45 : tan (45 * π / 180) = 1
axiom tan_period (x : ℝ) (k : ℤ) : tan (x + k * π) = tan x

end trigonometric_calculations_l4095_409502


namespace ed_limpet_shells_l4095_409531

/-- The number of limpet shells Ed found -/
def L : ℕ := sorry

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := 2

/-- The number of oyster shells Ed found -/
def ed_oyster_shells : ℕ := 2

/-- The number of conch shells Ed found -/
def ed_conch_shells : ℕ := 4

/-- The total number of shells Ed found -/
def ed_total_shells : ℕ := L + ed_oyster_shells + ed_conch_shells

/-- The total number of shells Jacob found -/
def jacob_total_shells : ℕ := ed_total_shells + 2

/-- The total number of shells in the final collection -/
def total_shells : ℕ := 30

theorem ed_limpet_shells :
  initial_shells + ed_total_shells + jacob_total_shells = total_shells ∧ L = 7 := by
  sorry

end ed_limpet_shells_l4095_409531


namespace conditional_prob_B_given_A_l4095_409560

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
structure FairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  fair_heads : prob_heads = 1/2
  fair_tails : prob_tails = 1/2

/-- Event A: "the first appearance of heads" when a coin is tossed twice -/
def event_A (c : FairCoin) : ℝ := c.prob_heads

/-- Event B: "the second appearance of tails" -/
def event_B (c : FairCoin) : ℝ := c.prob_tails

/-- The probability of both events A and B occurring -/
def prob_AB (c : FairCoin) : ℝ := c.prob_heads * c.prob_tails

/-- Theorem: The conditional probability P(B|A) is 1/2 -/
theorem conditional_prob_B_given_A (c : FairCoin) : 
  prob_AB c / event_A c = 1/2 := by
  sorry


end conditional_prob_B_given_A_l4095_409560


namespace cos_2theta_minus_7pi_over_2_l4095_409555

theorem cos_2theta_minus_7pi_over_2 (θ : ℝ) (h : Real.sin θ + Real.cos θ = -Real.sqrt 5 / 3) :
  Real.cos (2 * θ - 7 * Real.pi / 2) = 4 / 9 := by
  sorry

end cos_2theta_minus_7pi_over_2_l4095_409555


namespace sqrt_square_eq_abs_l4095_409575

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end sqrt_square_eq_abs_l4095_409575


namespace even_function_coefficient_l4095_409522

theorem even_function_coefficient (a : ℝ) :
  (∀ x : ℝ, (fun x => x^2 + a*x + 1) x = (fun x => x^2 + a*x + 1) (-x)) →
  a = 0 :=
by sorry

end even_function_coefficient_l4095_409522


namespace factorization_x4_minus_81_complete_factorization_l4095_409508

theorem factorization_x4_minus_81 (x : ℝ) :
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by sorry

theorem complete_factorization (p q r : ℝ → ℝ) :
  (∀ x, x^4 - 81 = p x * q x * r x) →
  (∀ x, p x = x - 3 ∨ p x = x + 3 ∨ p x = x^2 + 9) →
  (∀ x, q x = x - 3 ∨ q x = x + 3 ∨ q x = x^2 + 9) →
  (∀ x, r x = x - 3 ∨ r x = x + 3 ∨ r x = x^2 + 9) →
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ x, p x * q x * r x = (x - 3) * (x + 3) * (x^2 + 9)) :=
by sorry

end factorization_x4_minus_81_complete_factorization_l4095_409508


namespace camp_cedar_counselor_ratio_l4095_409556

theorem camp_cedar_counselor_ratio : 
  ∀ (num_boys : ℕ) (num_girls : ℕ) (num_counselors : ℕ),
    num_boys = 40 →
    num_girls = 3 * num_boys →
    num_counselors = 20 →
    (num_boys + num_girls) / num_counselors = 8 :=
by
  sorry

end camp_cedar_counselor_ratio_l4095_409556


namespace egg_grouping_l4095_409520

theorem egg_grouping (total_eggs : ℕ) (eggs_per_group : ℕ) (groups : ℕ) : 
  total_eggs = 8 → eggs_per_group = 2 → groups = total_eggs / eggs_per_group → groups = 4 := by
  sorry

end egg_grouping_l4095_409520


namespace monotone_decreasing_iff_b_positive_l4095_409589

/-- The function f(x) = (ax + b) / x is monotonically decreasing on (0, +∞) if and only if b > 0 -/
theorem monotone_decreasing_iff_b_positive (a b : ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (a * x + b) / x > (a * y + b) / y) ↔ b > 0 := by
  sorry

end monotone_decreasing_iff_b_positive_l4095_409589


namespace expensive_coat_savings_l4095_409525

/-- Represents a coat with its cost and lifespan. -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period. -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period + coat.lifespan - 1) / coat.lifespan * coat.cost

/-- Proves that buying the more expensive coat saves $120 over 30 years. -/
theorem expensive_coat_savings :
  let expensiveCoat : Coat := { cost := 300, lifespan := 15 }
  let cheapCoat : Coat := { cost := 120, lifespan := 5 }
  let period : ℕ := 30
  totalCost cheapCoat period - totalCost expensiveCoat period = 120 := by
  sorry


end expensive_coat_savings_l4095_409525


namespace circle_line_distance_range_l4095_409573

theorem circle_line_distance_range (b : ℝ) :
  (∃! (p q : ℝ × ℝ), 
    ((p.1 - 1)^2 + (p.2 - 1)^2 = 4) ∧
    ((q.1 - 1)^2 + (q.2 - 1)^2 = 4) ∧
    (p ≠ q) ∧
    (|p.2 - (p.1 + b)| / Real.sqrt 2 = 1) ∧
    (|q.2 - (q.1 + b)| / Real.sqrt 2 = 1)) →
  b ∈ Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end circle_line_distance_range_l4095_409573


namespace second_square_width_is_seven_l4095_409503

/-- Represents the dimensions of a rectangular piece of fabric -/
structure Fabric where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular piece of fabric -/
def area (f : Fabric) : ℝ := f.length * f.width

/-- Represents the three pieces of fabric and the desired flag dimensions -/
structure FlagProblem where
  fabric1 : Fabric
  fabric2 : Fabric
  fabric3 : Fabric
  flagLength : ℝ
  flagHeight : ℝ

/-- The flag problem with given dimensions -/
def bobbysProblem : FlagProblem :=
  { fabric1 := { length := 8, width := 5 }
  , fabric2 := { length := 10, width := 7 }  -- We'll prove this width
  , fabric3 := { length := 5, width := 5 }
  , flagLength := 15
  , flagHeight := 9
  }

theorem second_square_width_is_seven :
  let p := bobbysProblem
  area p.fabric1 + area p.fabric2 + area p.fabric3 = p.flagLength * p.flagHeight ∧
  p.fabric2.width = 7 := by
  sorry


end second_square_width_is_seven_l4095_409503


namespace equation_solution_l4095_409585

theorem equation_solution : 
  ∃ (x : ℚ), (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 :=
by sorry

end equation_solution_l4095_409585


namespace jason_lost_three_balloons_l4095_409513

/-- The number of violet balloons Jason lost -/
def lost_balloons (initial current : ℕ) : ℕ := initial - current

/-- Proof that Jason lost 3 violet balloons -/
theorem jason_lost_three_balloons :
  let initial_violet : ℕ := 7
  let current_violet : ℕ := 4
  lost_balloons initial_violet current_violet = 3 := by
  sorry

end jason_lost_three_balloons_l4095_409513


namespace sandy_paint_area_l4095_409584

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length window_width window_height : ℝ) : ℝ :=
  wall_height * wall_length - window_width * window_height

/-- Theorem stating the area Sandy needs to paint -/
theorem sandy_paint_area :
  area_to_paint 9 12 2 4 = 100 := by
  sorry

end sandy_paint_area_l4095_409584


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4095_409562

/-- An isosceles triangle with side lengths 2, 2, and 5 has a perimeter of 9 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 2 ∧ b = 2 ∧ c = 5 ∧  -- Two sides are 2, one side is 5
      (a = b ∨ b = c ∨ a = c) ∧  -- Triangle is isosceles
      a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
      perimeter = a + b + c ∧  -- Definition of perimeter
      perimeter = 9  -- The perimeter is 9

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4095_409562


namespace cubic_root_b_value_l4095_409578

theorem cubic_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) →
  b = -37/2 := by
sorry

end cubic_root_b_value_l4095_409578


namespace chord_length_polar_l4095_409547

theorem chord_length_polar (ρ θ : ℝ) : 
  ρ = 4 * Real.sin θ → θ = π / 4 → ρ = 2 * Real.sqrt 2 := by
  sorry

end chord_length_polar_l4095_409547


namespace fraction_equality_l4095_409580

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) →
  Q / P = 8 / 3 := by
  sorry

end fraction_equality_l4095_409580


namespace hyperbola_focus_smaller_x_l4095_409563

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 20)^2 / 15^2 = 1

-- Define the focus with smaller x-coordinate
def focus_smaller_x : ℝ × ℝ := (-11.55, 20)

-- Theorem statement
theorem hyperbola_focus_smaller_x :
  ∃ (f : ℝ × ℝ), 
    (∀ x y, hyperbola x y → (x - 5)^2 + (y - 20)^2 ≥ (f.1 - 5)^2 + (f.2 - 20)^2) ∧
    (∀ x y, hyperbola x y → x ≤ 5 → x ≥ f.1) ∧
    f = focus_smaller_x :=
sorry

end hyperbola_focus_smaller_x_l4095_409563


namespace system_solution_l4095_409524

theorem system_solution (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 2*b)
  (eq3 : a + b = 9 - 5*c) :
  3*a + 3*b + 3*c = 18/17 := by
sorry

end system_solution_l4095_409524


namespace quadratic_equation_solution_l4095_409500

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 4 * x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 / 2 ∧
              x₂ = -1 - Real.sqrt 6 / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_equation_solution_l4095_409500
