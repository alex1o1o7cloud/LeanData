import Mathlib

namespace largest_four_digit_sum_19_l1683_168398

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 8920 :=
by sorry

end largest_four_digit_sum_19_l1683_168398


namespace transport_cost_is_162_50_l1683_168359

/-- Calculates the transport cost for a refrigerator purchase given the following conditions:
  * purchase_price: The price Ramesh paid after discount
  * discount_rate: The discount rate on the labelled price
  * installation_cost: The cost of installation
  * profit_rate: The desired profit rate if no discount was offered
  * selling_price: The price to sell at to achieve the desired profit rate
-/
def calculate_transport_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (installation_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit := labelled_price * profit_rate
  let calculated_selling_price := labelled_price + profit
  selling_price - calculated_selling_price - installation_cost

/-- Theorem stating that given the specific conditions of Ramesh's refrigerator purchase,
    the transport cost is 162.50 rupees. -/
theorem transport_cost_is_162_50 :
  calculate_transport_cost 12500 0.20 250 0.10 17600 = 162.50 := by
  sorry

end transport_cost_is_162_50_l1683_168359


namespace carbon_atoms_in_compound_l1683_168326

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem carbon_atoms_in_compound :
  ∀ (c : Compound),
    c.hydrogen = 6 →
    c.oxygen = 1 →
    molecularWeight c 12 1 16 = 58 →
    c.carbon = 3 := by
  sorry

end carbon_atoms_in_compound_l1683_168326


namespace f_composition_equality_l1683_168371

noncomputable def f (x : ℝ) : ℝ :=
  if x > 3 then Real.exp x else Real.log (x + 1)

theorem f_composition_equality : f (f (f 1)) = Real.log (Real.log (Real.log 2 + 1) + 1) := by
  sorry

end f_composition_equality_l1683_168371


namespace spherical_coordinate_equivalence_l1683_168390

/-- Given a point in spherical coordinates, find its equivalent representation in standard spherical coordinates. -/
theorem spherical_coordinate_equivalence :
  ∀ (ρ θ φ : ℝ),
  ρ > 0 →
  (∃ (k : ℤ), θ = 3 * π / 8 + 2 * π * k) →
  (∃ (m : ℤ), φ = 9 * π / 5 + 2 * π * m) →
  ∃ (θ' φ' : ℝ),
    0 ≤ θ' ∧ θ' < 2 * π ∧
    0 ≤ φ' ∧ φ' ≤ π ∧
    (ρ, θ', φ') = (4, 11 * π / 8, π / 5) :=
by sorry


end spherical_coordinate_equivalence_l1683_168390


namespace regular_polygon_perimeter_l1683_168378

/-- A regular polygon with side length 6 units and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter (n : ℕ) (s : ℝ) (E : ℝ) : 
  n > 0 → 
  s = 6 → 
  E = 90 → 
  E = 360 / n → 
  n * s = 24 := by
sorry

end regular_polygon_perimeter_l1683_168378


namespace circle_C_and_min_chord_length_l1683_168306

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 5

-- Define the lines
def line_1 (x y : ℝ) : Prop := y = 2*x - 21
def line_2 (x y : ℝ) : Prop := y = 2*x - 11
def center_line (x y : ℝ) : Prop := x + y = 8

-- Define the intersecting line
def line_l (x y a : ℝ) : Prop := 2*x + a*y + 6*a = a*x + 14

-- Theorem statement
theorem circle_C_and_min_chord_length :
  ∃ (x₀ y₀ : ℝ),
    -- Center of C lies on the center line
    center_line x₀ y₀ ∧
    -- C is tangent to line_1 and line_2
    (∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ line_1 x₁ y₁) ∧
    (∃ (x₂ y₂ : ℝ), circle_C x₂ y₂ ∧ line_2 x₂ y₂) ∧
    -- The equation of circle C
    (∀ (x y : ℝ), circle_C x y ↔ (x - 8)^2 + y^2 = 5) ∧
    -- Minimum length of chord MN
    (∀ (a : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
        line_l x₁ y₁ a ∧ line_l x₂ y₂ a) →
      ∃ (m n : ℝ), m ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧ m = 12 ∧ n^2 = m) :=
sorry

end circle_C_and_min_chord_length_l1683_168306


namespace sqrt_inequality_l1683_168358

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < 1) :
  let f : ℝ → ℝ := fun x ↦ Real.sqrt x
  f a < f b ∧ f b < f (1/b) ∧ f (1/b) < f (1/a) := by
  sorry

end sqrt_inequality_l1683_168358


namespace cubic_inequality_l1683_168385

theorem cubic_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : a^3 < a^2 * b := by
  sorry

end cubic_inequality_l1683_168385


namespace investment_problem_l1683_168302

theorem investment_problem (x y T : ℝ) : 
  x + y = T →
  y = 800 →
  0.1 * x - 0.08 * y = 56 →
  T = 2000 := by
sorry

end investment_problem_l1683_168302


namespace water_evaporation_period_l1683_168389

theorem water_evaporation_period (initial_amount : Real) (daily_rate : Real) (evaporation_percentage : Real) : 
  initial_amount > 0 → 
  daily_rate > 0 → 
  evaporation_percentage > 0 → 
  evaporation_percentage < 100 →
  initial_amount = 40 →
  daily_rate = 0.01 →
  evaporation_percentage = 0.5 →
  (initial_amount * evaporation_percentage / 100) / daily_rate = 20 := by
sorry

end water_evaporation_period_l1683_168389


namespace quadrilateral_Q₁PNF_is_cyclic_l1683_168303

/-- Two circles with points on them and their intersections -/
structure TwoCirclesConfig where
  /-- The first circle -/
  circle1 : Set (ℝ × ℝ)
  /-- The second circle -/
  circle2 : Set (ℝ × ℝ)
  /-- Point Q₁, an intersection of the two circles -/
  Q₁ : ℝ × ℝ
  /-- Point Q₂, another intersection of the two circles -/
  Q₂ : ℝ × ℝ
  /-- Point A on the first circle -/
  A : ℝ × ℝ
  /-- Point B on the first circle -/
  B : ℝ × ℝ
  /-- Point C, where AQ₂ intersects circle2 again -/
  C : ℝ × ℝ
  /-- Point F on arc Q₁Q₂ of circle1, inside circle2 -/
  F : ℝ × ℝ
  /-- Point P, intersection of AF and BQ₁ -/
  P : ℝ × ℝ
  /-- Point N, where PC intersects circle2 again -/
  N : ℝ × ℝ

  /-- Q₁ and Q₂ are on both circles -/
  h1 : Q₁ ∈ circle1 ∧ Q₁ ∈ circle2
  h2 : Q₂ ∈ circle1 ∧ Q₂ ∈ circle2
  /-- A and B are on circle1 -/
  h3 : A ∈ circle1
  h4 : B ∈ circle1
  /-- C is on circle2 -/
  h5 : C ∈ circle2
  /-- F is on arc Q₁Q₂ of circle1, inside circle2 -/
  h6 : F ∈ circle1
  /-- N is on circle2 -/
  h7 : N ∈ circle2

/-- The main theorem: Quadrilateral Q₁PNF is cyclic -/
theorem quadrilateral_Q₁PNF_is_cyclic (config : TwoCirclesConfig) :
  ∃ (circle : Set (ℝ × ℝ)), config.Q₁ ∈ circle ∧ config.P ∈ circle ∧ config.N ∈ circle ∧ config.F ∈ circle :=
sorry

end quadrilateral_Q₁PNF_is_cyclic_l1683_168303


namespace final_shell_count_l1683_168396

def calculate_final_shells (initial : ℕ) 
  (vacation1_day1to3 : ℕ) (vacation1_day4 : ℕ) (vacation1_lost : ℕ)
  (vacation2_day1to2 : ℕ) (vacation2_day3 : ℕ) (vacation2_given : ℕ)
  (vacation3_day1 : ℕ) (vacation3_day2 : ℕ) (vacation3_day3to4 : ℕ) (vacation3_misplaced : ℕ) : ℕ :=
  initial + 
  (vacation1_day1to3 * 3 + vacation1_day4 - vacation1_lost) +
  (vacation2_day1to2 * 2 + vacation2_day3 - vacation2_given) +
  (vacation3_day1 + vacation3_day2 + vacation3_day3to4 * 2 - vacation3_misplaced)

theorem final_shell_count :
  calculate_final_shells 20 5 6 4 4 7 3 8 4 3 5 = 62 := by
  sorry

end final_shell_count_l1683_168396


namespace all_props_true_l1683_168368

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x * y = 0) → (x = 0 ∨ y = 0)

-- Define the inverse proposition
def inverse_prop (x y : ℝ) : Prop := (x = 0 ∨ y = 0) → (x * y = 0)

-- Define the negation proposition
def negation_prop (x y : ℝ) : Prop := (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)

-- Define the contrapositive proposition
def contrapositive_prop (x y : ℝ) : Prop := (x ≠ 0 ∧ y ≠ 0) → (x * y ≠ 0)

-- Theorem stating that all three derived propositions are true
theorem all_props_true : 
  (∀ x y : ℝ, inverse_prop x y) ∧ 
  (∀ x y : ℝ, negation_prop x y) ∧ 
  (∀ x y : ℝ, contrapositive_prop x y) :=
sorry

end all_props_true_l1683_168368


namespace quadratic_root_problem_l1683_168397

theorem quadratic_root_problem (b : ℝ) :
  ((-2 : ℝ)^2 + b * (-2) = 0) → (0^2 + b * 0 = 0) :=
by
  sorry

end quadratic_root_problem_l1683_168397


namespace expression_equality_l1683_168377

theorem expression_equality : 
  (-3^2 ≠ -2^3) ∧ 
  ((-3)^2 ≠ (-2)^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end expression_equality_l1683_168377


namespace alcohol_dilution_l1683_168329

/-- Proves that adding 3 litres of water to a 20-litre mixture containing 20% alcohol
    results in a new mixture with 17.391304347826086% alcohol. -/
theorem alcohol_dilution (original_volume : ℝ) (original_alcohol_percentage : ℝ) 
    (added_water : ℝ) (new_alcohol_percentage : ℝ) : 
    original_volume = 20 →
    original_alcohol_percentage = 0.20 →
    added_water = 3 →
    new_alcohol_percentage = 0.17391304347826086 →
    (original_volume * original_alcohol_percentage) / (original_volume + added_water) = new_alcohol_percentage :=
by sorry

end alcohol_dilution_l1683_168329


namespace vertex_locus_is_parabola_l1683_168328

theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  let vertex (t : ℝ) := (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)
  ∃ f : ℝ → ℝ, (∀ x, f x = -a * x^2 + c) ∧
    (∀ t, (vertex t).2 = f (vertex t).1) :=
by sorry

end vertex_locus_is_parabola_l1683_168328


namespace unique_congruence_solution_l1683_168327

theorem unique_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -998 [ZMOD 5] ∧ n = 2 := by
  sorry

end unique_congruence_solution_l1683_168327


namespace sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l1683_168309

def sum_1_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_1_to_1000_equals_500500 :
  sum_1_to_n 1000 = 500500 :=
by sorry

theorem sum_forward_equals_sum_backward (n : ℕ) :
  (List.range n).sum = (List.range n).reverse.sum :=
by sorry

#check sum_1_to_1000_equals_500500
#check sum_forward_equals_sum_backward

end sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l1683_168309


namespace ln_plus_const_increasing_l1683_168394

theorem ln_plus_const_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => Real.log x + 2) :=
sorry

end ln_plus_const_increasing_l1683_168394


namespace last_four_average_l1683_168361

theorem last_four_average (numbers : Fin 7 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13)
  (h2 : numbers 4 + numbers 5 + numbers 6 = 55)
  (h3 : numbers 3 ^ 2 = numbers 6)
  (h4 : numbers 6 = 25) :
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 := by
sorry

end last_four_average_l1683_168361


namespace reciprocal_product_equals_19901_l1683_168357

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n / (1 + (n + 1) * a n * a (n + 1))

-- State the theorem
theorem reciprocal_product_equals_19901 :
  1 / (a 190 * a 200) = 19901 := by
  sorry

end reciprocal_product_equals_19901_l1683_168357


namespace loan_principal_calculation_l1683_168301

/-- Proves that given a loan with 4% annual simple interest over 8 years,
    if the interest is Rs. 306 less than the principal,
    then the principal must be Rs. 450. -/
theorem loan_principal_calculation (P : ℚ) : 
  (P * (4 : ℚ) * (8 : ℚ) / (100 : ℚ) = P - (306 : ℚ)) → P = (450 : ℚ) := by
  sorry

end loan_principal_calculation_l1683_168301


namespace system_of_equations_solution_l1683_168304

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 7 * y = 5) ∧ (x = 62 / 3) ∧ (y = 17) := by
  sorry

end system_of_equations_solution_l1683_168304


namespace math_quiz_items_l1683_168317

theorem math_quiz_items (score_percentage : ℝ) (mistakes : ℕ) (total_items : ℕ) : 
  score_percentage = 0.80 → 
  mistakes = 5 → 
  (total_items - mistakes : ℝ) / total_items = score_percentage → 
  total_items = 25 := by
sorry

end math_quiz_items_l1683_168317


namespace unique_divisible_digit_l1683_168393

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 26372 * 100 + A * 10 + 21

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧ 
    (∃ k₁ k₂ k₃ : ℕ, 
      number_with_A A = 2 * k₁ ∧
      number_with_A A = 3 * k₂ ∧
      number_with_A A = 4 * k₃) :=
sorry

end unique_divisible_digit_l1683_168393


namespace product_of_fraction_parts_l1683_168300

/-- Represents a repeating decimal with a 4-digit repeating sequence -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

/-- The fraction representation of 0.0012 (repeating) -/
def fraction : ℚ := RepeatingDecimal 0 0 1 2

theorem product_of_fraction_parts : ∃ (n d : ℕ), fraction = n / d ∧ Nat.gcd n d = 1 ∧ n * d = 13332 := by
  sorry

end product_of_fraction_parts_l1683_168300


namespace quadratic_factorization_l1683_168356

theorem quadratic_factorization (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 := by
  sorry

end quadratic_factorization_l1683_168356


namespace systematic_sampling_removal_l1683_168338

theorem systematic_sampling_removal (total : Nat) (sample_size : Nat) (h : total = 162 ∧ sample_size = 16) :
  total % sample_size = 2 := by
  sorry

end systematic_sampling_removal_l1683_168338


namespace donnas_truck_weight_l1683_168312

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let produce_weight := 2 * soda_weight
  empty_truck_weight + soda_weight + dryers_weight + produce_weight = 24000 := by
  sorry

end donnas_truck_weight_l1683_168312


namespace specific_grid_toothpicks_l1683_168391

/-- Represents a rectangular grid of toothpicks with reinforcements -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  horizontalReinforcementInterval : ℕ
  verticalReinforcementInterval : ℕ

/-- Calculates the total number of toothpicks in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalLines := grid.height + 1
  let verticalLines := grid.width + 1
  let baseHorizontal := horizontalLines * grid.width
  let baseVertical := verticalLines * grid.height
  let reinforcedHorizontal := (horizontalLines / grid.horizontalReinforcementInterval) * grid.width
  let reinforcedVertical := (verticalLines / grid.verticalReinforcementInterval) * grid.height
  baseHorizontal + baseVertical + reinforcedHorizontal + reinforcedVertical

/-- Theorem stating that the specific grid configuration results in 990 toothpicks -/
theorem specific_grid_toothpicks :
  totalToothpicks { height := 25, width := 15, horizontalReinforcementInterval := 5, verticalReinforcementInterval := 3 } = 990 := by
  sorry

end specific_grid_toothpicks_l1683_168391


namespace bowling_team_score_l1683_168305

theorem bowling_team_score (total_score : ℕ) (bowler1 bowler2 bowler3 : ℕ) : 
  total_score = 810 →
  bowler1 = bowler2 / 3 →
  bowler2 = 3 * bowler3 →
  bowler1 + bowler2 + bowler3 = total_score →
  bowler3 = 162 := by
sorry

end bowling_team_score_l1683_168305


namespace simplify_and_evaluate_l1683_168307

theorem simplify_and_evaluate (a : ℝ) (h : a = -2) :
  (1 - 1 / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = 2/3 := by
  sorry

end simplify_and_evaluate_l1683_168307


namespace complex_fraction_equals_i_l1683_168335

theorem complex_fraction_equals_i : (1 + 5*I) / (5 - I) = I := by
  sorry

end complex_fraction_equals_i_l1683_168335


namespace ellipse_properties_l1683_168320

/-- Prove that an ellipse with given properties has specific semi-major and semi-minor axes -/
theorem ellipse_properties (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c > 0 ∧ c^2 = m^2 - n^2) →  -- Ellipse property: c^2 = a^2 - b^2
  (2 : ℝ) = m - (m^2 - n^2).sqrt →  -- Right focus at (2, 0)
  (1 / 2 : ℝ) = (m^2 - n^2).sqrt / m →  -- Eccentricity is 1/2
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end ellipse_properties_l1683_168320


namespace job_interviews_comprehensive_l1683_168330

/-- Represents a scenario that may or may not require comprehensive investigation. -/
inductive Scenario
| AirQuality
| VisionStatus
| JobInterviews
| FishCount

/-- Determines if a scenario requires comprehensive investigation. -/
def requiresComprehensiveInvestigation (s : Scenario) : Prop :=
  match s with
  | Scenario.JobInterviews => True
  | _ => False

/-- Theorem stating that job interviews is the only scenario requiring comprehensive investigation. -/
theorem job_interviews_comprehensive :
  ∀ s : Scenario, requiresComprehensiveInvestigation s ↔ s = Scenario.JobInterviews :=
by sorry

end job_interviews_comprehensive_l1683_168330


namespace q_gt_one_neither_sufficient_nor_necessary_l1683_168381

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem q_gt_one_neither_sufficient_nor_necessary :
  ∃ (a₁ b₁ : ℕ → ℝ) (q₁ q₂ : ℝ),
    GeometricSequence a₁ q₁ ∧ q₁ > 1 ∧ ¬IncreasingSequence a₁ ∧
    GeometricSequence b₁ q₂ ∧ q₂ ≤ 1 ∧ IncreasingSequence b₁ :=
  sorry

end q_gt_one_neither_sufficient_nor_necessary_l1683_168381


namespace solve_grocery_store_problem_l1683_168384

def grocery_store_problem (regular_soda : ℕ) (diet_soda : ℕ) (total_bottles : ℕ) : Prop :=
  let lite_soda : ℕ := total_bottles - (regular_soda + diet_soda)
  lite_soda = 27

theorem solve_grocery_store_problem :
  grocery_store_problem 57 26 110 := by
  sorry

end solve_grocery_store_problem_l1683_168384


namespace positive_integer_solutions_count_l1683_168374

theorem positive_integer_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 1001) (Finset.product (Finset.range 1002) (Finset.range 1002))).card = 167 :=
by sorry

end positive_integer_solutions_count_l1683_168374


namespace molar_mass_not_unique_l1683_168375

/-- Represents a solution with a solute -/
structure Solution :=
  (mass_fraction : ℝ)
  (mass : ℝ)

/-- Represents the result of mixing two solutions and evaporating water -/
structure MixedSolution :=
  (solution1 : Solution)
  (solution2 : Solution)
  (evaporated_water : ℝ)
  (final_molarity : ℝ)

/-- Function to calculate molar mass given additional information -/
noncomputable def calculate_molar_mass (mixed : MixedSolution) (additional_info : ℝ) : ℝ :=
  sorry

/-- Theorem stating that molar mass cannot be uniquely determined without additional information -/
theorem molar_mass_not_unique (mixed : MixedSolution) :
  ∃ (info1 info2 : ℝ), info1 ≠ info2 ∧ 
  calculate_molar_mass mixed info1 ≠ calculate_molar_mass mixed info2 :=
sorry

end molar_mass_not_unique_l1683_168375


namespace exam_maximum_marks_l1683_168324

theorem exam_maximum_marks :
  ∀ (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) (failing_margin : ℕ),
    passing_percentage = 1/4 →
    student_score = 185 →
    failing_margin = 25 →
    (passing_percentage * total_marks : ℚ) = (student_score + failing_margin) →
    total_marks = 840 := by
  sorry

end exam_maximum_marks_l1683_168324


namespace student_hall_ratio_l1683_168315

theorem student_hall_ratio : 
  let general_hall : ℕ := 30
  let total_students : ℕ := 144
  let math_hall (biology_hall : ℕ) : ℚ := (3/5) * (general_hall + biology_hall)
  ∃ biology_hall : ℕ, 
    (general_hall : ℚ) + biology_hall + math_hall biology_hall = total_students ∧
    biology_hall / general_hall = 2 := by
  sorry

end student_hall_ratio_l1683_168315


namespace stating_price_reduction_achieves_target_profit_l1683_168388

/-- Represents the price reduction problem for a product in a shopping mall. -/
structure PriceReductionProblem where
  initialSales : ℕ        -- Initial average daily sales
  initialProfit : ℕ       -- Initial profit per unit
  salesIncrease : ℕ       -- Sales increase per yuan of price reduction
  targetProfit : ℕ        -- Target daily profit
  priceReduction : ℕ      -- Price reduction per unit

/-- 
Theorem stating that the given price reduction achieves the target profit 
for the specified problem parameters.
-/
theorem price_reduction_achieves_target_profit 
  (p : PriceReductionProblem)
  (h1 : p.initialSales = 30)
  (h2 : p.initialProfit = 50)
  (h3 : p.salesIncrease = 2)
  (h4 : p.targetProfit = 2000)
  (h5 : p.priceReduction = 25) :
  (p.initialProfit - p.priceReduction) * (p.initialSales + p.salesIncrease * p.priceReduction) = p.targetProfit :=
by sorry

end stating_price_reduction_achieves_target_profit_l1683_168388


namespace salt_water_ratio_l1683_168387

theorem salt_water_ratio (salt : ℕ) (water : ℕ) :
  salt = 1 ∧ water = 10 →
  (salt : ℚ) / (salt + water : ℚ) = 1 / 11 :=
by sorry

end salt_water_ratio_l1683_168387


namespace line_vertical_shift_specific_line_shift_l1683_168343

/-- Given a line y = mx + b, moving it down by k units results in y = mx + (b - k) -/
theorem line_vertical_shift (m b k : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let shifted_line := fun (x : ℝ) => m * x + (b - k)
  (∀ x, shifted_line x = original_line x - k) :=
by sorry

/-- Moving the line y = 3x down 2 units results in y = 3x - 2 -/
theorem specific_line_shift :
  let original_line := fun (x : ℝ) => 3 * x
  let shifted_line := fun (x : ℝ) => 3 * x - 2
  (∀ x, shifted_line x = original_line x - 2) :=
by sorry

end line_vertical_shift_specific_line_shift_l1683_168343


namespace vasya_upward_run_time_l1683_168344

/-- Represents the speed and time properties of Vasya's escalator run -/
structure EscalatorRun where
  -- Vasya's speed going down (in units per minute)
  speed_down : ℝ
  -- Vasya's speed going up (in units per minute)
  speed_up : ℝ
  -- Escalator's speed (in units per minute)
  escalator_speed : ℝ
  -- Time for stationary run (in minutes)
  time_stationary : ℝ
  -- Time for downward moving escalator run (in minutes)
  time_down : ℝ
  -- Constraint: Vasya runs down twice as fast as he runs up
  speed_constraint : speed_down = 2 * speed_up
  -- Constraint: Stationary run takes 6 minutes
  stationary_constraint : time_stationary = 6
  -- Constraint: Downward moving escalator run takes 13.5 minutes
  down_constraint : time_down = 13.5

/-- Theorem stating the time for Vasya's upward moving escalator run -/
theorem vasya_upward_run_time (run : EscalatorRun) :
  let time_up := (1 / (run.speed_down - run.escalator_speed) + 1 / (run.speed_up + run.escalator_speed)) * 60
  time_up = 324 := by
  sorry

end vasya_upward_run_time_l1683_168344


namespace leak_empty_time_l1683_168362

def tank_capacity : ℝ := 1
def fill_time_no_leak : ℝ := 3
def empty_time_leak : ℝ := 12

theorem leak_empty_time :
  let fill_rate : ℝ := tank_capacity / fill_time_no_leak
  let leak_rate : ℝ := tank_capacity / empty_time_leak
  tank_capacity / leak_rate = empty_time_leak := by
sorry

end leak_empty_time_l1683_168362


namespace opposite_of_sqrt_seven_l1683_168366

-- Define the opposite function
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_sqrt_seven :
  opposite (Real.sqrt 7) = -(Real.sqrt 7) := by
  sorry

end opposite_of_sqrt_seven_l1683_168366


namespace substitution_method_simplification_l1683_168376

theorem substitution_method_simplification (x y : ℝ) :
  (4 * x - 3 * y = -1) ∧ (5 * x + y = 13) →
  y = 13 - 5 * x := by
sorry

end substitution_method_simplification_l1683_168376


namespace cone_volume_l1683_168319

/-- The volume of a cone with slant height 5 and base radius 3 is 12π -/
theorem cone_volume (s h r : ℝ) (hs : s = 5) (hr : r = 3) 
  (height_eq : h^2 + r^2 = s^2) : 
  (1/3 : ℝ) * π * r^2 * h = 12 * π := by
  sorry

end cone_volume_l1683_168319


namespace average_age_of_new_joiners_l1683_168353

/-- Given a group of people going for a picnic, prove the average age of new joiners -/
theorem average_age_of_new_joiners
  (initial_count : ℕ)
  (initial_avg_age : ℝ)
  (new_count : ℕ)
  (new_total_avg_age : ℝ)
  (h1 : initial_count = 12)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 12)
  (h4 : new_total_avg_age = 15.5) :
  let total_count := initial_count + new_count
  let new_joiners_avg_age := (total_count * new_total_avg_age - initial_count * initial_avg_age) / new_count
  new_joiners_avg_age = 15 := by
sorry

end average_age_of_new_joiners_l1683_168353


namespace symmetry_axis_of_sine_l1683_168383

theorem symmetry_axis_of_sine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (1/2 * x + π/3)
  f (π/3 + (x - π/3)) = f (π/3 - (x - π/3)) := by
  sorry

end symmetry_axis_of_sine_l1683_168383


namespace triangle_inequality_with_heights_l1683_168325

theorem triangle_inequality_with_heights 
  (a b c h_a h_b h_c t : ℝ) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) 
  (heights_def : h_a * a = h_b * b ∧ h_b * b = h_c * c) 
  (t_bound : t ≥ (1 : ℝ) / 2) : 
  (t * a + h_a) + (t * b + h_b) > t * c + h_c ∧ 
  (t * b + h_b) + (t * c + h_c) > t * a + h_a ∧ 
  (t * c + h_c) + (t * a + h_a) > t * b + h_b :=
sorry

end triangle_inequality_with_heights_l1683_168325


namespace limit_sequence_is_zero_l1683_168311

/-- The limit of the sequence (n - (n^5 - 5)^(1/3)) * n * sqrt(n) as n approaches infinity is 0. -/
theorem limit_sequence_is_zero :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n : ℝ) - ((n : ℝ)^5 - 5)^(1/3)) * n * (n : ℝ).sqrt| < ε :=
by sorry

end limit_sequence_is_zero_l1683_168311


namespace min_value_of_expression_l1683_168334

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (4 * x / (x + 3 * y)) + (3 * y / x) ≥ 3 := by
  sorry

end min_value_of_expression_l1683_168334


namespace quadratic_inequality_solution_set_l1683_168342

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 5*x - 6 > 0} = {x | x < -1 ∨ x > 6} := by sorry

end quadratic_inequality_solution_set_l1683_168342


namespace sqrt_calculations_l1683_168367

theorem sqrt_calculations :
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x * y) →
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x / y) →
  (∃ (sqrt10 sqrt2 sqrt15 sqrt3 sqrt5 sqrt27 sqrt12 sqrt_third : ℝ),
    sqrt10 > 0 ∧ sqrt10 * sqrt10 = 10 ∧
    sqrt2 > 0 ∧ sqrt2 * sqrt2 = 2 ∧
    sqrt15 > 0 ∧ sqrt15 * sqrt15 = 15 ∧
    sqrt3 > 0 ∧ sqrt3 * sqrt3 = 3 ∧
    sqrt5 > 0 ∧ sqrt5 * sqrt5 = 5 ∧
    sqrt27 > 0 ∧ sqrt27 * sqrt27 = 27 ∧
    sqrt12 > 0 ∧ sqrt12 * sqrt12 = 12 ∧
    sqrt_third > 0 ∧ sqrt_third * sqrt_third = 1/3 ∧
    sqrt10 * sqrt2 + sqrt15 / sqrt3 = 3 * sqrt5 ∧
    sqrt27 - (sqrt12 - sqrt_third) = 4/3 * sqrt3) :=
by sorry

end sqrt_calculations_l1683_168367


namespace doll_collection_increase_l1683_168331

theorem doll_collection_increase (initial_count : ℕ) : 
  (initial_count : ℚ) * (1 + 1/4) = initial_count + 2 → 
  initial_count + 2 = 10 := by
  sorry

end doll_collection_increase_l1683_168331


namespace max_profit_profit_range_l1683_168350

/-- Represents the store's pricing and sales model -/
structure Store where
  costPrice : ℝ
  maxProfitPercent : ℝ
  k : ℝ
  b : ℝ

/-- Calculates the profit given a selling price -/
def profit (s : Store) (x : ℝ) : ℝ :=
  (x - s.costPrice) * (s.k * x + s.b)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit (s : Store) 
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∃ (maxProfit sellPrice : ℝ),
      maxProfit = 891 ∧
      sellPrice = 87 ∧
      ∀ x, s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
        profit s x ≤ maxProfit := by
  sorry

/-- Theorem stating the selling price range for profit ≥ 500 -/
theorem profit_range (s : Store)
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∀ x, profit s x ≥ 500 ∧ s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
      70 ≤ x ∧ x ≤ 87 := by
  sorry

end max_profit_profit_range_l1683_168350


namespace tethered_dog_area_l1683_168332

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  let outside_area := (rope_length^2 * (5/6) + 2 * (rope_length - side_length)^2 * (1/6)) * π
  outside_area = (49/6) * π :=
by sorry

end tethered_dog_area_l1683_168332


namespace prob_at_least_one_junior_l1683_168364

/-- The probability of selecting at least one junior when randomly choosing 4 people from a group of 8 seniors and 4 juniors -/
theorem prob_at_least_one_junior (total : ℕ) (seniors : ℕ) (juniors : ℕ) (selected : ℕ) : 
  total = seniors + juniors →
  seniors = 8 →
  juniors = 4 →
  selected = 4 →
  (1 - (seniors.choose selected : ℚ) / (total.choose selected : ℚ)) = 85 / 99 := by
  sorry

end prob_at_least_one_junior_l1683_168364


namespace distinct_cubes_modulo_prime_l1683_168314

theorem distinct_cubes_modulo_prime (a b c p : ℤ) : 
  Prime p → 
  p = a * b + b * c + a * c → 
  a ≠ b → b ≠ c → a ≠ c → 
  (a^3 % p ≠ b^3 % p) ∧ (b^3 % p ≠ c^3 % p) ∧ (a^3 % p ≠ c^3 % p) :=
by sorry

end distinct_cubes_modulo_prime_l1683_168314


namespace fraction_equals_zero_l1683_168339

theorem fraction_equals_zero (x : ℝ) (h : (x - 3) / x = 0) : x = 3 := by
  sorry

end fraction_equals_zero_l1683_168339


namespace egyptian_pi_approximation_l1683_168321

theorem egyptian_pi_approximation (d : ℝ) (h : d > 0) :
  (π * d^2 / 4 = (8 * d / 9)^2) → π = 256 / 81 := by
  sorry

end egyptian_pi_approximation_l1683_168321


namespace four_digit_number_problem_l1683_168373

theorem four_digit_number_problem :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (n / 1000 = 2) ∧  -- thousand's place is 2
    (((n % 1000) * 10 + 2) = 2 * n + 66) ∧  -- condition for moving 2 to unit's place
    n = 2508 :=
by sorry

end four_digit_number_problem_l1683_168373


namespace geometric_sequence_statements_l1683_168340

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 1: One term of a geometric sequence can be 0. -/
def Statement1 : Prop :=
  ∃ (a : ℕ → ℝ) (n : ℕ), IsGeometricSequence a ∧ a n = 0

/-- Statement 2: The common ratio of a geometric sequence can take any real value. -/
def Statement2 : Prop :=
  ∀ r : ℝ, ∃ a : ℕ → ℝ, IsGeometricSequence a ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 3: If b² = ac, then a, b, c form a geometric sequence. -/
def Statement3 : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → ∃ r : ℝ, r ≠ 0 ∧ b = r * a ∧ c = r * b

/-- Statement 4: If a constant sequence is a geometric sequence, then its common ratio is 1. -/
def Statement4 : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n m : ℕ, a n = a m) → IsGeometricSequence a → ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_statements :
  ¬Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ Statement4 := by sorry

end geometric_sequence_statements_l1683_168340


namespace decimal_density_between_half_and_seven_tenths_l1683_168372

theorem decimal_density_between_half_and_seven_tenths :
  ∃ (x y : ℚ), 0.5 < x ∧ x < y ∧ y < 0.7 :=
sorry

end decimal_density_between_half_and_seven_tenths_l1683_168372


namespace first_player_wins_or_draws_l1683_168316

/-- Represents a game where two players take turns picking bills from a sequence. -/
structure BillGame where
  n : ℕ
  bills : List ℕ
  turn : ℕ

/-- Represents a move in the game, either taking from the left or right end. -/
inductive Move
  | Left
  | Right

/-- Represents the result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Defines an optimal strategy for the first player. -/
def optimalStrategy : BillGame → Move
  | _ => sorry

/-- Simulates the game with both players following the optimal strategy. -/
def playGame : BillGame → GameResult
  | _ => sorry

/-- Theorem stating that the first player can always ensure a win or draw. -/
theorem first_player_wins_or_draws (n : ℕ) :
  ∀ (game : BillGame),
    game.n = n ∧
    game.bills = List.range (2*n) ∧
    game.turn = 0 →
    playGame game ≠ GameResult.SecondPlayerWins :=
  sorry

end first_player_wins_or_draws_l1683_168316


namespace complex_subtraction_l1683_168345

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end complex_subtraction_l1683_168345


namespace percentage_calculation_l1683_168346

-- Define constants
def rupees_to_paise : ℝ → ℝ := (· * 100)

-- Theorem statement
theorem percentage_calculation (x : ℝ) : 
  (x / 100) * rupees_to_paise 160 = 80 → x = 0.5 := by
  sorry

end percentage_calculation_l1683_168346


namespace cherry_pie_problem_l1683_168308

/-- The number of single cherries in one pound of cherries -/
def cherries_per_pound (total_cherries : ℕ) (total_pounds : ℕ) : ℕ :=
  total_cherries / total_pounds

/-- The number of cherries that can be pitted in a given time -/
def cherries_pitted (time_minutes : ℕ) (cherries_per_10_min : ℕ) : ℕ :=
  (time_minutes / 10) * cherries_per_10_min

theorem cherry_pie_problem (pounds_needed : ℕ) (pitting_time_hours : ℕ) (cherries_per_10_min : ℕ) :
  pounds_needed = 3 →
  pitting_time_hours = 2 →
  cherries_per_10_min = 20 →
  cherries_per_pound (cherries_pitted (pitting_time_hours * 60) cherries_per_10_min) pounds_needed = 80 := by
  sorry

end cherry_pie_problem_l1683_168308


namespace minimum_average_for_remaining_semesters_l1683_168347

def required_average : ℝ := 85
def num_semesters : ℕ := 5
def first_three_scores : List ℝ := [84, 88, 80]

theorem minimum_average_for_remaining_semesters :
  let total_required := required_average * num_semesters
  let current_total := first_three_scores.sum
  let remaining_semesters := num_semesters - first_three_scores.length
  let remaining_required := total_required - current_total
  (remaining_required / remaining_semesters : ℝ) = 86.5 := by
sorry

end minimum_average_for_remaining_semesters_l1683_168347


namespace sufficient_not_necessary_condition_l1683_168310

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) :=
sorry

end sufficient_not_necessary_condition_l1683_168310


namespace tree_height_difference_l1683_168399

-- Define the heights of the trees
def pine_height : ℚ := 49/4
def maple_height : ℚ := 75/4

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 13/2 :=
by sorry

end tree_height_difference_l1683_168399


namespace range_of_g_l1683_168360

def g (x : ℝ) : ℝ := -x^2 + 3*x - 3

theorem range_of_g :
  ∀ y ∈ Set.range (fun (x : ℝ) => g x), -31 ≤ y ∧ y ≤ -3/4 ∧
  ∃ x₁ x₂ : ℝ, -4 ≤ x₁ ∧ x₁ ≤ 4 ∧ -4 ≤ x₂ ∧ x₂ ≤ 4 ∧ g x₁ = -31 ∧ g x₂ = -3/4 :=
by sorry

end range_of_g_l1683_168360


namespace point_on_circle_l1683_168370

/-- 
Given a line ax + by - 1 = 0 that is tangent to the circle x² + y² = 1,
prove that the point P(a, b) lies on the circle.
-/
theorem point_on_circle (a b : ℝ) 
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 = 1 := by
  sorry


end point_on_circle_l1683_168370


namespace washing_machines_removed_count_l1683_168336

/-- Represents the number of washing machines removed from a shipping container --/
def washing_machines_removed (crates boxes_per_crate machines_per_box machines_removed_per_box : ℕ) : ℕ :=
  crates * boxes_per_crate * machines_removed_per_box

/-- Theorem stating the number of washing machines removed from the shipping container --/
theorem washing_machines_removed_count : 
  washing_machines_removed 10 6 4 1 = 60 := by
  sorry

#eval washing_machines_removed 10 6 4 1

end washing_machines_removed_count_l1683_168336


namespace wage_increase_percentage_l1683_168313

theorem wage_increase_percentage (initial_wage : ℝ) (final_wage : ℝ) : 
  initial_wage = 10 →
  final_wage = 9 →
  final_wage = 0.75 * (initial_wage * (1 + x/100)) →
  x = 20 :=
by
  sorry

end wage_increase_percentage_l1683_168313


namespace sufficiency_not_necessity_l1683_168322

theorem sufficiency_not_necessity (a b : ℝ) :
  (a < b ∧ b < 0 → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(a < b ∧ b < 0) :=
by sorry

end sufficiency_not_necessity_l1683_168322


namespace circle_center_distance_l1683_168349

/-- The distance between the center of the circle x²+y²=4x+6y+3 and the point (8,4) is √37 -/
theorem circle_center_distance : ∃ (h k : ℝ),
  (∀ x y : ℝ, x^2 + y^2 = 4*x + 6*y + 3 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 3)) ∧
  Real.sqrt ((8 - h)^2 + (4 - k)^2) = Real.sqrt 37 := by
  sorry

end circle_center_distance_l1683_168349


namespace acclimation_time_is_one_year_l1683_168379

/-- Represents the time spent on different phases of PhD study -/
structure PhDTime where
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Conditions for John's PhD timeline -/
def johnPhDConditions (t : PhDTime) : Prop :=
  t.basics = 2 ∧
  t.research = 1.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.acclimation + t.basics + t.research + t.dissertation = 7

/-- Theorem stating that under the given conditions, the acclimation time is 1 year -/
theorem acclimation_time_is_one_year (t : PhDTime) 
  (h : johnPhDConditions t) : t.acclimation = 1 := by
  sorry


end acclimation_time_is_one_year_l1683_168379


namespace sixth_term_of_arithmetic_sequence_l1683_168355

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem sixth_term_of_arithmetic_sequence :
  let a₁ := 2
  let d := 3
  arithmetic_sequence a₁ d 6 = 17 := by
sorry

end sixth_term_of_arithmetic_sequence_l1683_168355


namespace arithmetic_progression_product_divisible_l1683_168392

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

/-- The product of all elements in a list -/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem arithmetic_progression_product_divisible (a : ℕ) :
  (listProduct (arithmeticProgression a 11 10)) % (Nat.factorial 10) = 0 := by
  sorry

end arithmetic_progression_product_divisible_l1683_168392


namespace rates_sum_of_squares_l1683_168323

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of activities for Ed -/
def ed_sum (r : Rates) : ℕ := 3 * r.biking + 2 * r.jogging + 3 * r.swimming

/-- The sum of activities for Sue -/
def sue_sum (r : Rates) : ℕ := 5 * r.biking + 3 * r.jogging + 2 * r.swimming

/-- The sum of squares of the rates -/
def sum_of_squares (r : Rates) : ℕ := r.biking^2 + r.jogging^2 + r.swimming^2

theorem rates_sum_of_squares : 
  ∃ r : Rates, ed_sum r = 82 ∧ sue_sum r = 99 ∧ sum_of_squares r = 314 := by
  sorry

end rates_sum_of_squares_l1683_168323


namespace empty_tank_weight_is_80_l1683_168337

/-- The weight of an empty water tank --/
def empty_tank_weight (tank_capacity : ℝ) (fill_percentage : ℝ) (water_weight : ℝ) (filled_weight : ℝ) : ℝ :=
  filled_weight - (tank_capacity * fill_percentage * water_weight)

/-- Theorem stating the weight of the empty tank --/
theorem empty_tank_weight_is_80 :
  empty_tank_weight 200 0.80 8 1360 = 80 := by
  sorry

end empty_tank_weight_is_80_l1683_168337


namespace negative_integer_square_plus_self_l1683_168351

theorem negative_integer_square_plus_self (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end negative_integer_square_plus_self_l1683_168351


namespace jeff_wins_three_matches_l1683_168369

/-- Calculates the number of matches won given play time in hours, 
    minutes per point, and points needed per match. -/
def matches_won (play_time_hours : ℕ) (minutes_per_point : ℕ) (points_per_match : ℕ) : ℕ :=
  (play_time_hours * 60) / (minutes_per_point * points_per_match)

/-- Theorem stating that playing for 2 hours, scoring a point every 5 minutes, 
    and needing 8 points to win a match results in winning 3 matches. -/
theorem jeff_wins_three_matches : 
  matches_won 2 5 8 = 3 := by
  sorry

end jeff_wins_three_matches_l1683_168369


namespace congruence_problem_l1683_168348

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + y) % (4^3) = 2^3 % (4^3))
  (h3 : (6 + y) % (6^3) = 2^3 % (6^3)) :
  y % 24 = 6 := by
  sorry

end congruence_problem_l1683_168348


namespace smallest_gcd_qr_l1683_168318

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1155) :
  ∃ (m : ℕ+), (∀ (n : ℕ+), Nat.gcd q r ≥ n → n ≤ m) ∧ Nat.gcd q r ≥ m :=
by sorry

end smallest_gcd_qr_l1683_168318


namespace sequence_sum_formula_l1683_168380

def sequence_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (List.range n).map a |>.sum

theorem sequence_sum_formula (a : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 - a n * (sequence_sum a n - 1) - a n = 0) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by
  sorry

end sequence_sum_formula_l1683_168380


namespace complex_fraction_equality_l1683_168352

theorem complex_fraction_equality : (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end complex_fraction_equality_l1683_168352


namespace negative_square_cubed_l1683_168365

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l1683_168365


namespace ten_gentlemen_hat_probability_l1683_168363

/-- The harmonic number H_n is defined as the sum of reciprocals of the first n positive integers. -/
def harmonicNumber (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The probability that n gentlemen each receive their own hat when distributed randomly. -/
def hatProbability (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (Finset.range (n - 1)).prod (fun i => harmonicNumber (i + 2) / (i + 2 : ℚ))

/-- Theorem stating the probability that 10 gentlemen each receive their own hat. -/
theorem ten_gentlemen_hat_probability :
  ∃ (p : ℚ), hatProbability 10 = p ∧ 0.000515 < p ∧ p < 0.000517 := by
  sorry


end ten_gentlemen_hat_probability_l1683_168363


namespace quadratic_polynomial_equality_l1683_168395

theorem quadratic_polynomial_equality 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) 
  (h_equality : ∀ x, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) :
  ∃ (a b c : ℝ), (a = 1 ∧ b = 5 ∧ c = 1) ∧ (∀ x, f x = x^2 + 5 * x + 1) := by
  sorry

end quadratic_polynomial_equality_l1683_168395


namespace distinct_elements_condition_l1683_168386

theorem distinct_elements_condition (x : ℝ) : 
  ({1, x, x^2 - x} : Set ℝ).ncard = 3 ↔ 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 :=
sorry

end distinct_elements_condition_l1683_168386


namespace brad_age_is_13_l1683_168382

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

def average_age : ℕ := (shara_age + jaymee_age) / 2

def brad_age : ℕ := average_age - 3

theorem brad_age_is_13 : brad_age = 13 := by
  sorry

end brad_age_is_13_l1683_168382


namespace jakes_weight_l1683_168333

theorem jakes_weight (jake sister brother : ℝ) : 
  (0.8 * jake = 2 * sister) →
  (jake + sister = 168) →
  (brother = 1.25 * (jake + sister)) →
  (jake + sister + brother = 221) →
  jake = 120 := by
sorry

end jakes_weight_l1683_168333


namespace casino_money_theorem_l1683_168354

/-- The amount of money on table A -/
def table_a : ℕ := 40

/-- The amount of money on table C -/
def table_c : ℕ := table_a + 20

/-- The amount of money on table B -/
def table_b : ℕ := 2 * table_c

/-- The total amount of money on all tables -/
def total_money : ℕ := table_a + table_b + table_c

theorem casino_money_theorem : total_money = 220 := by
  sorry

end casino_money_theorem_l1683_168354


namespace unique_solution_for_equation_l1683_168341

theorem unique_solution_for_equation : 
  ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end unique_solution_for_equation_l1683_168341
