import Mathlib

namespace emperor_strategy_exists_l2531_253138

/-- Represents the nature of a wizard -/
inductive WizardNature
| Good
| Evil

/-- Represents a wizard -/
structure Wizard where
  nature : WizardNature

/-- Represents the Emperor's knowledge about a wizard -/
inductive WizardKnowledge
| Unknown
| KnownGood
| KnownEvil

/-- Represents the state of the festival -/
structure FestivalState where
  wizards : Finset Wizard
  knowledge : Wizard → WizardKnowledge

/-- Represents a strategy for the Emperor -/
structure EmperorStrategy where
  askQuestion : FestivalState → Wizard → Prop
  expelWizard : FestivalState → Option Wizard

/-- The main theorem -/
theorem emperor_strategy_exists :
  ∃ (strategy : EmperorStrategy),
    ∀ (initial_state : FestivalState),
      initial_state.wizards.card = 2015 →
      ∃ (final_state : FestivalState),
        (∀ w ∈ final_state.wizards, w.nature = WizardNature.Good) ∧
        (∃! w, w ∉ final_state.wizards ∧ w.nature = WizardNature.Good) :=
by sorry

end emperor_strategy_exists_l2531_253138


namespace weight_measurement_l2531_253163

def weight_set : List Nat := [2, 5, 15]

def heaviest_weight (weights : List Nat) : Nat :=
  weights.sum

def different_weights (weights : List Nat) : Finset Nat :=
  sorry

theorem weight_measurement (weights : List Nat := weight_set) :
  (heaviest_weight weights = 22) ∧
  (different_weights weights).card = 9 := by
  sorry

end weight_measurement_l2531_253163


namespace arithmetic_geometric_mean_sum_of_squares_l2531_253108

theorem arithmetic_geometric_mean_sum_of_squares (a b : ℝ) :
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 110 → a^2 + b^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_sum_of_squares_l2531_253108


namespace sculpture_height_proof_l2531_253136

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 8

/-- The total height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.5

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height := by
  sorry

end sculpture_height_proof_l2531_253136


namespace hyperbola_distance_property_l2531_253145

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance_property (P : ℝ × ℝ) :
  is_on_hyperbola P.1 P.2 →
  distance P F1 = 12 →
  (distance P F2 = 2 ∨ distance P F2 = 22) :=
sorry

end hyperbola_distance_property_l2531_253145


namespace f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l2531_253157

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a) * abs x + b

-- Part 1: Prove that when a = 2 and b = 3, the only zero of f is x = -1
theorem f_zero_at_negative_one :
  ∃! x : ℝ, f 2 3 x = 0 ∧ x = -1 := by sorry

-- Part 2: Prove that when b = -2, f(x) < 0 for all x ∈ [-1, 1] if and only if a > -1
theorem f_negative_iff_a_greater_than_negative_one :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a (-2) x < 0) ↔ a > -1 := by sorry

end f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l2531_253157


namespace church_members_difference_church_members_proof_l2531_253192

theorem church_members_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_members adult_percentage children_difference =>
    total_members = 120 →
    adult_percentage = 40 →
    let adult_count := total_members * adult_percentage / 100
    let children_count := total_members - adult_count
    children_count - adult_count = children_difference

-- The proof of the theorem
theorem church_members_proof : church_members_difference 120 40 24 := by
  sorry

end church_members_difference_church_members_proof_l2531_253192


namespace power_of_product_l2531_253121

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_of_product_l2531_253121


namespace lcm_hcf_relation_l2531_253141

theorem lcm_hcf_relation (a b : ℕ) (ha : a = 210) (hlcm : Nat.lcm a b = 2310) :
  Nat.gcd a b = a * b / Nat.lcm a b :=
by sorry

end lcm_hcf_relation_l2531_253141


namespace smallest_prime_divisor_of_sum_l2531_253197

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2) →
  p 2 (3^19 + 11^13) ∧ 
  ∀ q : ℕ, p q (3^19 + 11^13) → q ≥ 2 :=
by sorry

end smallest_prime_divisor_of_sum_l2531_253197


namespace jeff_fish_problem_l2531_253165

/-- The problem of finding the maximum mass of a single fish caught by Jeff. -/
theorem jeff_fish_problem (n : ℕ) (min_mass : ℝ) (first_three_mass : ℝ) :
  n = 21 ∧
  min_mass = 0.2 ∧
  first_three_mass = 1.5 ∧
  (∀ fish : ℕ, fish ≤ n → ∃ (mass : ℝ), mass ≥ min_mass) ∧
  (first_three_mass / 3 = (first_three_mass + (n - 3) * min_mass) / n) →
  ∃ (max_mass : ℝ), max_mass = 5.6 ∧ 
    ∀ (fish_mass : ℝ), (∃ (fish : ℕ), fish ≤ n ∧ fish_mass ≥ min_mass) → fish_mass ≤ max_mass :=
by sorry

end jeff_fish_problem_l2531_253165


namespace egg_container_problem_l2531_253140

theorem egg_container_problem (num_containers : ℕ) 
  (front_pos back_pos left_pos right_pos : ℕ) :
  num_containers = 28 →
  front_pos + back_pos = 34 →
  left_pos + right_pos = 5 →
  (num_containers * ((front_pos + back_pos - 1) * (left_pos + right_pos - 1))) = 3696 :=
by sorry

end egg_container_problem_l2531_253140


namespace flower_pot_cost_difference_flower_pot_cost_difference_proof_l2531_253106

theorem flower_pot_cost_difference 
  (num_pots : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (cost_difference : ℚ) : Prop :=
  num_pots = 6 ∧ 
  total_cost = 33/4 ∧ 
  largest_pot_cost = 13/8 ∧
  (∀ i : ℕ, i < num_pots - 1 → 
    (largest_pot_cost - i * cost_difference) > 
    (largest_pot_cost - (i + 1) * cost_difference)) ∧
  total_cost = (num_pots : ℚ) / 2 * 
    (2 * largest_pot_cost - (num_pots - 1 : ℚ) * cost_difference) →
  cost_difference = 1/10

theorem flower_pot_cost_difference_proof : 
  flower_pot_cost_difference 6 (33/4) (13/8) (1/10) :=
sorry

end flower_pot_cost_difference_flower_pot_cost_difference_proof_l2531_253106


namespace division_problem_l2531_253154

theorem division_problem (L S q : ℕ) : 
  L - S = 2415 → 
  L = 2520 → 
  L = S * q + 15 → 
  q = 23 := by sorry

end division_problem_l2531_253154


namespace polynomial_product_l2531_253177

-- Define the polynomials
def P (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x
def Q (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3

-- State the theorem
theorem polynomial_product :
  ∀ x : ℝ, P x * Q x = 4 * x^7 - 2 * x^6 - 6 * x^5 + 9 * x^4 := by
  sorry

end polynomial_product_l2531_253177


namespace triangle_side_length_l2531_253105

/-- Given a right-angled triangle XYZ where angle XZY is 30° and XZ = 12, prove XY = 12√3 -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →  -- right-angled triangle
  Real.cos (Real.arccos ((Y.1 - Z.1) * (X.1 - Z.1) + (Y.2 - Z.2) * (X.2 - Z.2)) / 
    (Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) * Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2))) = 1/2 →  -- angle XZY is 30°
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 144 →  -- XZ = 12
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 432  -- XY = 12√3
  := by sorry

end triangle_side_length_l2531_253105


namespace superbloom_probability_l2531_253155

def campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def sherbert : Finset Char := {'S', 'H', 'E', 'R', 'B', 'E', 'R', 'T'}
def globe : Finset Char := {'G', 'L', 'O', 'B', 'E'}
def superbloom : Finset Char := {'S', 'U', 'P', 'E', 'R', 'B', 'L', 'O', 'O', 'M'}

def probability_campus : ℚ := 1 / (campus.card.choose 3)
def probability_sherbert : ℚ := 9 / (sherbert.card.choose 5)
def probability_globe : ℚ := 1

theorem superbloom_probability :
  probability_campus * probability_sherbert * probability_globe = 9 / 1120 := by
  sorry

end superbloom_probability_l2531_253155


namespace solve_score_problem_l2531_253118

def score_problem (s1 s3 s4 : ℕ) (avg : ℚ) : Prop :=
  s1 ≤ 100 ∧ s3 ≤ 100 ∧ s4 ≤ 100 ∧
  s1 = 65 ∧ s3 = 82 ∧ s4 = 85 ∧
  avg = 75 ∧
  ∃ (s2 : ℕ), s2 ≤ 100 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg ∧ s2 = 68

theorem solve_score_problem (s1 s3 s4 : ℕ) (avg : ℚ) 
  (h : score_problem s1 s3 s4 avg) : 
  ∃ (s2 : ℕ), s2 = 68 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg :=
by sorry

end solve_score_problem_l2531_253118


namespace not_divisible_sum_of_not_divisible_product_plus_one_l2531_253160

theorem not_divisible_sum_of_not_divisible_product_plus_one (n : ℕ) 
  (h : ∀ (a b : ℕ), ¬(n ∣ 2^a * 3^b + 1)) :
  ∀ (c d : ℕ), ¬(n ∣ 2^c + 3^d) := by
  sorry

end not_divisible_sum_of_not_divisible_product_plus_one_l2531_253160


namespace big_boxes_count_l2531_253134

/-- The number of big boxes given the conditions of the problem -/
def number_of_big_boxes (small_boxes_per_big_box : ℕ) 
                        (candles_per_small_box : ℕ) 
                        (total_candles : ℕ) : ℕ :=
  total_candles / (small_boxes_per_big_box * candles_per_small_box)

theorem big_boxes_count :
  number_of_big_boxes 4 40 8000 = 50 := by
  sorry

#eval number_of_big_boxes 4 40 8000

end big_boxes_count_l2531_253134


namespace pell_equation_solution_form_l2531_253146

/-- Pell's equation solution type -/
structure PellSolution (d : ℕ) :=
  (x : ℕ)
  (y : ℕ)
  (eq : x^2 - d * y^2 = 1)

/-- Fundamental solution to Pell's equation -/
def fundamental_solution (d : ℕ) : PellSolution d := sorry

/-- Any solution to Pell's equation -/
def any_solution (d : ℕ) : PellSolution d := sorry

/-- Square-free natural number -/
def is_square_free (d : ℕ) : Prop := sorry

theorem pell_equation_solution_form 
  (d : ℕ) 
  (h_square_free : is_square_free d) 
  (x₁ y₁ : ℕ) 
  (h_fund : fundamental_solution d = ⟨x₁, y₁, sorry⟩) 
  (xₙ yₙ : ℕ) 
  (h_any : any_solution d = ⟨xₙ, yₙ, sorry⟩) :
  ∃ (n : ℕ), (xₙ : ℝ) + yₙ * Real.sqrt d = ((x₁ : ℝ) + y₁ * Real.sqrt d) ^ n :=
sorry

end pell_equation_solution_form_l2531_253146


namespace sufficient_not_necessary_condition_l2531_253190

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b < a ∧ a < 0 → 1/b > 1/a) ∧
  ¬(∀ a b : ℝ, 1/b > 1/a → b < a ∧ a < 0) := by
  sorry

end sufficient_not_necessary_condition_l2531_253190


namespace pipe_stack_height_l2531_253185

theorem pipe_stack_height (d : ℝ) (h : ℝ) :
  d = 12 →
  h = 2 * d + d * Real.sqrt 3 →
  h = 24 + 12 * Real.sqrt 3 := by
  sorry

end pipe_stack_height_l2531_253185


namespace chalkboard_area_l2531_253166

/-- A rectangular chalkboard with a width of 3 feet and a length that is 2 times its width has an area of 18 square feet. -/
theorem chalkboard_area : 
  ∀ (width length area : ℝ), 
  width = 3 → 
  length = 2 * width → 
  area = width * length → 
  area = 18 :=
by
  sorry

end chalkboard_area_l2531_253166


namespace at_least_one_passes_l2531_253181

/-- Represents the number of questions in the exam pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions A can answer correctly -/
def a_correct : ℕ := 6

/-- Represents the number of questions B can answer correctly -/
def b_correct : ℕ := 8

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the minimum number of correct answers needed to pass -/
def pass_threshold : ℕ := 2

/-- Calculates the probability of an event -/
def prob (favorable : ℕ) (total : ℕ) : ℚ := (favorable : ℚ) / (total : ℚ)

/-- Calculates the probability of person A passing the exam -/
def prob_a_pass : ℚ := 
  prob (Nat.choose a_correct 2 * Nat.choose (total_questions - a_correct) 1 + 
        Nat.choose a_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Calculates the probability of person B passing the exam -/
def prob_b_pass : ℚ := 
  prob (Nat.choose b_correct 2 * Nat.choose (total_questions - b_correct) 1 + 
        Nat.choose b_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Theorem: The probability that at least one person passes the exam is 44/45 -/
theorem at_least_one_passes : 
  1 - (1 - prob_a_pass) * (1 - prob_b_pass) = 44 / 45 := by sorry

end at_least_one_passes_l2531_253181


namespace inequality_relations_l2531_253189

theorem inequality_relations (a d e : ℝ) 
  (h1 : a < 0) (h2 : a < d) (h3 : d < e) : 
  (a * d < d * e) ∧ 
  (a * e < d * e) ∧ 
  (a + d < d + e) ∧ 
  (e / a < 1) := by
  sorry

end inequality_relations_l2531_253189


namespace total_customers_is_43_l2531_253112

/-- Represents a table with a number of women and men -/
structure Table where
  women : ℕ
  men : ℕ

/-- Calculates the total number of customers at a table -/
def Table.total (t : Table) : ℕ := t.women + t.men

/-- Represents the waiter's situation -/
structure WaiterSituation where
  table1 : Table
  table2 : Table
  table3 : Table
  table4 : Table
  table5 : Table
  table6 : Table
  walkIn : Table
  table3Left : ℕ
  table4Joined : Table

/-- The initial situation of the waiter -/
def initialSituation : WaiterSituation where
  table1 := { women := 2, men := 4 }
  table2 := { women := 4, men := 3 }
  table3 := { women := 3, men := 5 }
  table4 := { women := 5, men := 2 }
  table5 := { women := 2, men := 1 }
  table6 := { women := 1, men := 2 }
  walkIn := { women := 4, men := 4 }
  table3Left := 2
  table4Joined := { women := 1, men := 2 }

/-- Calculates the total number of customers served by the waiter -/
def totalCustomersServed (s : WaiterSituation) : ℕ :=
  s.table1.total +
  s.table2.total +
  (s.table3.total - s.table3Left) +
  (s.table4.total + s.table4Joined.total) +
  s.table5.total +
  s.table6.total +
  s.walkIn.total

/-- Theorem stating that the total number of customers served is 43 -/
theorem total_customers_is_43 : totalCustomersServed initialSituation = 43 := by
  sorry

end total_customers_is_43_l2531_253112


namespace area_of_B_l2531_253196

-- Define the set A
def A : Set (ℝ × ℝ) := {p | p.1 + p.2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the transformation function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Define the set B
def B : Set (ℝ × ℝ) := f '' A

-- State the theorem
theorem area_of_B : MeasureTheory.volume B = 1 := by sorry

end area_of_B_l2531_253196


namespace diameter_endpoint_coordinates_l2531_253194

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- The other endpoint of a diameter given the circle and one endpoint --/
def otherDiameterEndpoint (c : Circle) (p : Point) : Point :=
  (2 * c.center.1 - p.1, 2 * c.center.2 - p.2)

theorem diameter_endpoint_coordinates :
  let c : Circle := { center := (3, 5) }
  let p : Point := (0, 1)
  otherDiameterEndpoint c p = (6, 9) := by
  sorry

#check diameter_endpoint_coordinates

end diameter_endpoint_coordinates_l2531_253194


namespace binomial_coefficient_7_choose_5_l2531_253114

theorem binomial_coefficient_7_choose_5 : Nat.choose 7 5 = 21 := by
  sorry

end binomial_coefficient_7_choose_5_l2531_253114


namespace subtract_negative_three_l2531_253149

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end subtract_negative_three_l2531_253149


namespace cubic_roots_inequality_l2531_253104

theorem cubic_roots_inequality (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 ∧
    a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧
    a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) :
  b * c < 3 * a * d := by
sorry

end cubic_roots_inequality_l2531_253104


namespace fourth_month_sale_is_7230_l2531_253110

/-- Represents the sales data for a grocer over 6 months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the fourth month given the sales data -/
def fourthMonthSale (sales : GrocerSales) : ℕ :=
  sales.average * 6 - (sales.month1 + sales.month2 + sales.month3 + sales.month5 + sales.month6)

/-- Theorem stating that the fourth month sale is 7230 given the provided sales data -/
theorem fourth_month_sale_is_7230 (sales : GrocerSales) 
  (h1 : sales.month1 = 6435)
  (h2 : sales.month2 = 6927)
  (h3 : sales.month3 = 6855)
  (h5 : sales.month5 = 6562)
  (h6 : sales.month6 = 6791)
  (ha : sales.average = 6800) :
  fourthMonthSale sales = 7230 := by
  sorry

#eval fourthMonthSale {
  month1 := 6435,
  month2 := 6927,
  month3 := 6855,
  month5 := 6562,
  month6 := 6791,
  average := 6800
}

end fourth_month_sale_is_7230_l2531_253110


namespace repeating_decimal_to_fraction_l2531_253151

/-- The repeating decimal 0.5656... is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction : 
  (∑' n, (56 : ℚ) / (100 ^ (n + 1))) = 56 / 99 := by
  sorry

end repeating_decimal_to_fraction_l2531_253151


namespace cos_72_minus_cos_144_eq_zero_l2531_253162

theorem cos_72_minus_cos_144_eq_zero : 
  Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 0 := by
  sorry

end cos_72_minus_cos_144_eq_zero_l2531_253162


namespace geometric_sequence_sum_l2531_253152

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 455/1365 := by
sorry

end geometric_sequence_sum_l2531_253152


namespace P_on_x_axis_P_distance_to_y_axis_l2531_253116

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2 * a - 1, a + 3)

-- Condition 1: P lies on the x-axis
theorem P_on_x_axis (a : ℝ) :
  P a = (-7, 0) ↔ (P a).2 = 0 :=
sorry

-- Condition 2: Distance from P to y-axis is 5
theorem P_distance_to_y_axis (a : ℝ) :
  (abs (P a).1 = 5) ↔ (P a = (-5, 1) ∨ P a = (5, 6)) :=
sorry

end P_on_x_axis_P_distance_to_y_axis_l2531_253116


namespace exam_sections_percentage_l2531_253132

theorem exam_sections_percentage :
  let total_candidates : ℕ := 1200
  let all_sections_percent : ℚ := 5 / 100
  let no_sections_percent : ℚ := 5 / 100
  let one_section_percent : ℚ := 25 / 100
  let four_sections_percent : ℚ := 20 / 100
  let three_sections_count : ℕ := 300
  
  ∃ (two_sections_percent : ℚ),
    two_sections_percent = 20 / 100 ∧
    (all_sections_percent + no_sections_percent + one_section_percent + 
     four_sections_percent + two_sections_percent + 
     (three_sections_count : ℚ) / total_candidates) = 1 :=
by sorry

end exam_sections_percentage_l2531_253132


namespace geometric_sequence_problem_l2531_253126

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {aₙ} where a₂ = 4 and a₆a₇ = 16a₉, prove that a₅ = ±32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_a2 : a 2 = 4)
    (h_a6a7 : a 6 * a 7 = 16 * a 9) : 
  a 5 = 32 ∨ a 5 = -32 := by
  sorry

end geometric_sequence_problem_l2531_253126


namespace at_least_three_equal_l2531_253150

theorem at_least_three_equal (a b c d : ℕ) 
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := by
sorry

end at_least_three_equal_l2531_253150


namespace partial_fraction_decomposition_l2531_253128

/-- The partial fraction decomposition equation holds for the given values of A, B, and C -/
theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x ≠ 1 → x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = A / x + (B * x + C) / (x^2 - 1) ∧
    A = 7 ∧ B = -9 ∧ C = 5 := by
  sorry

end partial_fraction_decomposition_l2531_253128


namespace add_8035_seconds_to_8am_l2531_253129

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time (8:00:00 AM) -/
def startTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 8035

/-- The expected end time (10:13:35) -/
def endTime : Time :=
  { hours := 10, minutes := 13, seconds := 35 }

theorem add_8035_seconds_to_8am :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end add_8035_seconds_to_8am_l2531_253129


namespace f_values_l2531_253167

/-- 
Represents the number of permutations a₁, ..., aₙ of the set {1, 2, ..., n} 
such that |aᵢ - aᵢ₊₁| ≠ 1 for all i = 1, 2, ..., n-1.
-/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem stating the values of f for n from 2 to 6 -/
theorem f_values : 
  f 2 = 0 ∧ f 3 = 0 ∧ f 4 = 2 ∧ f 5 = 14 ∧ f 6 = 90 := by sorry

end f_values_l2531_253167


namespace jennys_coins_value_l2531_253193

/-- Represents the value of Jenny's coins in cents -/
def coin_value (n : ℕ) : ℚ :=
  300 - 5 * n

/-- Represents the value of Jenny's coins in cents if nickels and dimes were swapped -/
def swapped_value (n : ℕ) : ℚ :=
  150 + 5 * n

/-- The number of nickels Jenny has -/
def number_of_nickels : ℕ :=
  27

theorem jennys_coins_value :
  coin_value number_of_nickels = 165 ∧
  swapped_value number_of_nickels = coin_value number_of_nickels + 120 :=
sorry

end jennys_coins_value_l2531_253193


namespace quadratic_roots_property_l2531_253179

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 13/3 := by
sorry

end quadratic_roots_property_l2531_253179


namespace identity_proof_l2531_253169

theorem identity_proof (a b c x y z : ℝ) : 
  (a*x + b*y + c*z)^2 + (b*x + c*y + a*z)^2 + (c*x + a*y + b*z)^2 = 
  (c*x + b*y + a*z)^2 + (b*x + a*y + c*z)^2 + (a*x + c*y + b*z)^2 := by
  sorry

end identity_proof_l2531_253169


namespace power_nine_mod_hundred_l2531_253123

theorem power_nine_mod_hundred : 9^2050 % 100 = 1 := by
  sorry

end power_nine_mod_hundred_l2531_253123


namespace vins_school_distance_l2531_253117

/-- The distance Vins rides to school -/
def distance_to_school : ℝ := sorry

/-- The distance Vins rides back home -/
def distance_back_home : ℝ := 7

/-- The number of round trips Vins made this week -/
def number_of_trips : ℕ := 5

/-- The total distance Vins rode this week -/
def total_distance : ℝ := 65

theorem vins_school_distance :
  distance_to_school = 6 :=
by
  sorry

end vins_school_distance_l2531_253117


namespace count_non_degenerate_triangles_l2531_253176

/-- The number of points in the figure -/
def total_points : ℕ := 16

/-- The number of collinear points on the base of the triangle -/
def base_points : ℕ := 5

/-- The number of collinear points on the semicircle -/
def semicircle_points : ℕ := 5

/-- The number of non-collinear points -/
def other_points : ℕ := total_points - base_points - semicircle_points

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of non-degenerate triangles -/
def non_degenerate_triangles : ℕ := 
  choose total_points 3 - 2 * choose base_points 3

theorem count_non_degenerate_triangles : 
  non_degenerate_triangles = 540 := by sorry

end count_non_degenerate_triangles_l2531_253176


namespace min_cubes_satisfy_conditions_num_cubes_is_minimum_l2531_253142

/-- Represents the number of cubes in each identical box -/
def num_cubes : ℕ := 1344

/-- Represents the side length of the outer square in the first girl's frame -/
def frame_outer : ℕ := 50

/-- Represents the side length of the inner square in the first girl's frame -/
def frame_inner : ℕ := 34

/-- Represents the side length of the second girl's square -/
def square_second : ℕ := 62

/-- Represents the side length of the third girl's square -/
def square_third : ℕ := 72

/-- Theorem stating that the given number of cubes satisfies all conditions -/
theorem min_cubes_satisfy_conditions :
  (frame_outer^2 - frame_inner^2 = num_cubes) ∧
  (square_second^2 = num_cubes) ∧
  (square_third^2 + 4 = num_cubes) := by
  sorry

/-- Theorem stating that the given number of cubes is the minimum possible -/
theorem num_cubes_is_minimum (n : ℕ) :
  (n < num_cubes) →
  ¬((frame_outer^2 - frame_inner^2 = n) ∧
    (∃ m : ℕ, m^2 = n) ∧
    (∃ k : ℕ, k^2 + 4 = n)) := by
  sorry

end min_cubes_satisfy_conditions_num_cubes_is_minimum_l2531_253142


namespace total_payment_for_bikes_l2531_253115

/-- The payment for painting a bike -/
def paint_fee : ℕ := 5

/-- The additional payment for selling a bike compared to painting it -/
def sell_bonus : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment for selling and painting one bike -/
def payment_per_bike : ℕ := paint_fee + (paint_fee + sell_bonus)

/-- Theorem stating the total payment for selling and painting 8 bikes -/
theorem total_payment_for_bikes : num_bikes * payment_per_bike = 144 := by
  sorry

end total_payment_for_bikes_l2531_253115


namespace nested_fraction_value_l2531_253172

theorem nested_fraction_value : 
  (1 : ℚ) / (1 + 1 / (1 + 1 / 2)) = 3 / 5 := by
  sorry

end nested_fraction_value_l2531_253172


namespace cube_root_equation_l2531_253127

theorem cube_root_equation : ∃ A : ℝ, 32 * A * A * A = 42592 ∧ A = 11 := by
  sorry

end cube_root_equation_l2531_253127


namespace brainiac_survey_l2531_253191

theorem brainiac_survey (R M : ℕ) : 
  R = 2 * M →                   -- Twice as many like rebus as math
  18 ≤ M ∧ 18 ≤ R →             -- 18 like both rebus and math
  20 ≤ M →                      -- 20 like math but not rebus
  R + M - 18 + 4 = 100          -- Total surveyed is 100
  := by sorry

end brainiac_survey_l2531_253191


namespace pencil_price_l2531_253147

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℕ) (pen_price : ℕ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 630 →
  pen_price = 16 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end pencil_price_l2531_253147


namespace right_triangle_side_length_l2531_253170

theorem right_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 17 → a = 15 →
  c^2 = a^2 + b^2 →
  b = 8 := by
sorry

end right_triangle_side_length_l2531_253170


namespace discount_savings_l2531_253174

theorem discount_savings (original_price : ℝ) (discount_rate : ℝ) (num_contributors : ℕ) 
  (discounted_price : ℝ) (individual_savings : ℝ) : 
  original_price > 0 → 
  discount_rate = 0.2 → 
  num_contributors = 3 → 
  discounted_price = 48 → 
  discounted_price = original_price * (1 - discount_rate) → 
  individual_savings = (original_price - discounted_price) / num_contributors → 
  individual_savings = 4 := by
sorry

end discount_savings_l2531_253174


namespace area_ratio_is_three_thirtyseconds_l2531_253199

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  -- Side lengths of triangle PQR
  pq : ℝ
  qr : ℝ
  rp : ℝ
  -- Ratios for points X, Y, Z
  x : ℝ
  y : ℝ
  z : ℝ
  -- Conditions
  pq_eq : pq = 7
  qr_eq : qr = 24
  rp_eq : rp = 25
  x_pos : x > 0
  y_pos : y > 0
  z_pos : z > 0
  sum_eq : x + y + z = 3/4
  sum_sq_eq : x^2 + y^2 + z^2 = 3/8

/-- The ratio of areas of triangle XYZ to triangle PQR -/
def areaRatio (t : TriangleWithPoints) : ℚ :=
  3/32

/-- Theorem stating that the area ratio is 3/32 -/
theorem area_ratio_is_three_thirtyseconds (t : TriangleWithPoints) :
  areaRatio t = 3/32 := by
  sorry

end area_ratio_is_three_thirtyseconds_l2531_253199


namespace inequality_proof_l2531_253153

theorem inequality_proof (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hca : c + d ≤ a) (hcb : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
sorry

end inequality_proof_l2531_253153


namespace average_speed_to_first_summit_l2531_253100

/-- Proves that the average speed to the first summit is equal to the overall average speed
    given the total journey time, overall average speed, and time to first summit. -/
theorem average_speed_to_first_summit
  (total_time : ℝ)
  (overall_avg_speed : ℝ)
  (time_to_first_summit : ℝ)
  (h_total_time : total_time = 8)
  (h_overall_avg_speed : overall_avg_speed = 3)
  (h_time_to_first_summit : time_to_first_summit = 3) :
  (overall_avg_speed * time_to_first_summit) / time_to_first_summit = overall_avg_speed :=
by sorry

#check average_speed_to_first_summit

end average_speed_to_first_summit_l2531_253100


namespace integer_power_sum_l2531_253135

theorem integer_power_sum (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end integer_power_sum_l2531_253135


namespace first_number_problem_l2531_253107

theorem first_number_problem (x : ℤ) : x + 7314 = 3362 + 13500 → x = 9548 := by
  sorry

end first_number_problem_l2531_253107


namespace equation_solutions_l2531_253188

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10)
  ∀ x : ℝ, f x = 0 ↔ x = -2 + 2*Real.sqrt 14 ∨ x = -2 - 2*Real.sqrt 14 ∨ 
                    x = (7 + Real.sqrt 89) / 2 ∨ x = (7 - Real.sqrt 89) / 2 := by
  sorry

end equation_solutions_l2531_253188


namespace range_of_f_l2531_253164

noncomputable def f (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end range_of_f_l2531_253164


namespace increase_by_percentage_l2531_253143

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) :
  initial = 240 →
  percentage = 20 →
  result = initial * (1 + percentage / 100) →
  result = 288 := by
  sorry

end increase_by_percentage_l2531_253143


namespace complex_number_quadrant_l2531_253111

theorem complex_number_quadrant : 
  let z : ℂ := (1/2 + (Real.sqrt 3 / 2) * Complex.I)^2
  z.re < 0 ∧ z.im > 0 := by sorry

end complex_number_quadrant_l2531_253111


namespace marked_price_calculation_l2531_253173

theorem marked_price_calculation (total_cost : ℚ) (discount_rate : ℚ) : 
  total_cost = 50 → discount_rate = 1/10 → 
  ∃ (marked_price : ℚ), marked_price = 250/9 ∧ 
  2 * (marked_price * (1 - discount_rate)) = total_cost :=
by sorry

end marked_price_calculation_l2531_253173


namespace omega_range_l2531_253120

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  let symmetry_axis (k : ℤ) : ℝ := (3*π/4 + k*π) / ω
  (∀ k : ℤ, symmetry_axis k ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.Icc (3/8) (7/12) ∪ Set.Icc (7/8) (11/12) :=
by sorry

end omega_range_l2531_253120


namespace new_person_weight_l2531_253148

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : 
  initial_count = 8 → 
  weight_increase = 2.5 → 
  replaced_weight = 55 → 
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end new_person_weight_l2531_253148


namespace min_value_of_expression_l2531_253133

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y ≤ 1) :
  x^4 + y^4 - x^2*y - x*y^2 ≥ -1/8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≤ 1 ∧ a^4 + b^4 - a^2*b - a*b^2 = -1/8 :=
sorry

end min_value_of_expression_l2531_253133


namespace triangle_properties_l2531_253175

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.C = Real.pi / 3 ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_properties_l2531_253175


namespace original_triangle_area_l2531_253125

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_side = 5 * side) → 
  new_area = 125 → 
  original_area = 5 :=
by
  sorry

end original_triangle_area_l2531_253125


namespace two_plus_three_eq_eight_is_proposition_l2531_253137

-- Define what a proposition is
def is_proposition (s : String) : Prop := ∃ (b : Bool), (s = "true" ∨ s = "false")

-- State the theorem
theorem two_plus_three_eq_eight_is_proposition :
  is_proposition "2+3=8" :=
sorry

end two_plus_three_eq_eight_is_proposition_l2531_253137


namespace school_community_count_l2531_253156

theorem school_community_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 187 := by
  sorry

end school_community_count_l2531_253156


namespace twelfth_number_is_seven_l2531_253198

/-- A circular arrangement of 20 numbers -/
def CircularArrangement := Fin 20 → ℕ

/-- The property that the sum of any six consecutive numbers is 24 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 20, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 24

theorem twelfth_number_is_seven
  (arr : CircularArrangement)
  (h_sum : SumProperty arr)
  (h_first : arr 0 = 1) :
  arr 11 = 7 := by
  sorry

end twelfth_number_is_seven_l2531_253198


namespace new_average_age_l2531_253186

theorem new_average_age (n : ℕ) (original_avg : ℚ) (new_person_age : ℕ) : 
  n = 8 → original_avg = 14 → new_person_age = 32 → 
  (n * original_avg + new_person_age : ℚ) / (n + 1) = 16 := by
  sorry

end new_average_age_l2531_253186


namespace jerry_reading_pages_l2531_253124

theorem jerry_reading_pages : ∀ (total_pages pages_read_saturday pages_remaining : ℕ),
  total_pages = 93 →
  pages_read_saturday = 30 →
  pages_remaining = 43 →
  total_pages - pages_read_saturday - pages_remaining = 20 :=
by
  sorry

end jerry_reading_pages_l2531_253124


namespace estate_distribution_l2531_253131

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) : 
  -- Daughter and son together receive half the estate
  (∃ x : ℕ, 5 * x = E / 2) →
  -- Wife receives three times as much as the son
  (∃ y : ℕ, y = 6 * x) →
  -- First cook receives $800
  (∃ z₁ : ℕ, z₁ = 800) →
  -- Second cook receives $1200
  (∃ z₂ : ℕ, z₂ = 1200) →
  -- Total estate equals sum of all shares
  (E = 11 * x + 2000) →
  -- The estate value is $20000
  E = 20000 := by
sorry

end estate_distribution_l2531_253131


namespace median_in_interval_75_79_l2531_253178

structure ScoreInterval :=
  (lower upper : ℕ)
  (count : ℕ)

def total_students : ℕ := 100

def score_distribution : List ScoreInterval :=
  [⟨85, 89, 18⟩, ⟨80, 84, 15⟩, ⟨75, 79, 20⟩, ⟨70, 74, 25⟩, ⟨65, 69, 12⟩, ⟨60, 64, 10⟩]

def cumulative_count (n : ℕ) : ℕ :=
  (score_distribution.take n).foldl (λ acc interval => acc + interval.count) 0

theorem median_in_interval_75_79 :
  ∃ k, k ∈ [75, 76, 77, 78, 79] ∧
    cumulative_count 2 < total_students / 2 ∧
    total_students / 2 ≤ cumulative_count 3 :=
  sorry

end median_in_interval_75_79_l2531_253178


namespace sale_total_cost_l2531_253144

/-- Calculates the total cost of ice cream and juice during a sale. -/
def calculate_total_cost (original_ice_cream_price : ℚ) 
                         (ice_cream_discount : ℚ) 
                         (juice_price : ℚ) 
                         (juice_cans_per_price : ℕ) 
                         (ice_cream_tubs : ℕ) 
                         (juice_cans : ℕ) : ℚ :=
  let sale_ice_cream_price := original_ice_cream_price - ice_cream_discount
  let ice_cream_cost := sale_ice_cream_price * ice_cream_tubs
  let juice_cost := (juice_price / juice_cans_per_price) * juice_cans
  ice_cream_cost + juice_cost

/-- Theorem stating that the total cost is $24 for the given conditions. -/
theorem sale_total_cost : 
  calculate_total_cost 12 2 2 5 2 10 = 24 := by
  sorry

end sale_total_cost_l2531_253144


namespace sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l2531_253168

theorem sufficient_condition_for_existence (m : ℝ) :
  (m ≤ 4) → (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) :=
by sorry

theorem necessary_condition_for_existence (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) → (m ≤ 4) :=
by sorry

theorem not_necessary_condition (m : ℝ) :
  ∃ m₀ : ℝ, m₀ ≤ 4 ∧ m₀ ≠ 4 ∧ (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m₀ ≥ 0) :=
by sorry

end sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l2531_253168


namespace fish_difference_l2531_253139

/-- Proves that Matthias has 15 fewer fish than Kenneth given the conditions in the problem -/
theorem fish_difference (micah_fish : ℕ) (total_fish : ℕ) : 
  micah_fish = 7 →
  total_fish = 34 →
  let kenneth_fish := 3 * micah_fish
  let matthias_fish := total_fish - micah_fish - kenneth_fish
  kenneth_fish - matthias_fish = 15 := by
sorry


end fish_difference_l2531_253139


namespace spinner_probability_l2531_253182

theorem spinner_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end spinner_probability_l2531_253182


namespace marble_remainder_l2531_253113

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end marble_remainder_l2531_253113


namespace min_sum_distances_to_four_points_l2531_253161

/-- The minimum sum of distances from a point to four fixed points -/
theorem min_sum_distances_to_four_points :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, -1)
  let C : ℝ × ℝ := (0, 3)
  let D : ℝ × ℝ := (-1, 3)
  ∀ P : ℝ × ℝ,
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) +
    Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) +
    Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) ≥
    3 * Real.sqrt 2 + 2 * Real.sqrt 5 :=
by
  sorry

end min_sum_distances_to_four_points_l2531_253161


namespace parallel_condition_l2531_253180

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

/-- The lines l₁ and l₂ parameterized by m -/
def l1 (m : ℝ) : Line := ⟨1, 2*m, -1⟩
def l2 (m : ℝ) : Line := ⟨3*m+1, -m, -1⟩

/-- The statement to be proved -/
theorem parallel_condition :
  (∀ m : ℝ, are_parallel (l1 m) (l2 m) → m = -1/2 ∨ m = 0) ∧
  (∃ m : ℝ, m ≠ -1/2 ∧ m ≠ 0 ∧ ¬are_parallel (l1 m) (l2 m)) :=
sorry

end parallel_condition_l2531_253180


namespace intersection_point_y_coordinate_l2531_253171

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := (x + 3)^2 + 3

-- Theorem statement
theorem intersection_point_y_coordinate :
  shifted_function 0 = 12 :=
by sorry

end intersection_point_y_coordinate_l2531_253171


namespace orange_harvest_per_day_l2531_253159

/-- Given a consistent daily harvest of oranges over 6 days resulting in 498 sacks,
    prove that the daily harvest is 83 sacks. -/
theorem orange_harvest_per_day :
  ∀ (daily_harvest : ℕ),
  daily_harvest * 6 = 498 →
  daily_harvest = 83 := by
  sorry

end orange_harvest_per_day_l2531_253159


namespace max_value_of_expression_l2531_253102

theorem max_value_of_expression (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (x + y + a)^2 / (x^2 + y^2 + a^2) ≤ 3 := by
  sorry

end max_value_of_expression_l2531_253102


namespace dot_product_om_on_l2531_253119

/-- Given two points M and N on the line x + y - 2 = 0, where M(1,1) and |MN| = √2,
    prove that the dot product of OM and ON equals 2 -/
theorem dot_product_om_on (N : ℝ × ℝ) : 
  N.1 + N.2 = 2 →  -- N is on the line x + y - 2 = 0
  (N.1 - 1)^2 + (N.2 - 1)^2 = 2 →  -- |MN| = √2
  (1 * N.1 + 1 * N.2 : ℝ) = 2 := by  -- OM · ON = 2
sorry

end dot_product_om_on_l2531_253119


namespace parallel_vectors_x_value_l2531_253109

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 4)
  let b : ℝ × ℝ := (4, x)
  parallel a b → x = 4 ∨ x = -4 :=
by
  sorry

end parallel_vectors_x_value_l2531_253109


namespace halfway_fraction_l2531_253183

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/6) :
  (a + b) / 2 = 5/24 := by
  sorry

end halfway_fraction_l2531_253183


namespace expression_equality_l2531_253158

theorem expression_equality (x : ℝ) (h : x > 0) : 
  (∃! e : ℕ, e = 1) ∧ 
  (6^x * x^3 = 3^x * x^3 + 3^x * x^3) ∧ 
  ((3*x)^(3*x) ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  (3^x * x^6 ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  ((6*x)^x ≠ 3^x * x^3 + 3^x * x^3) :=
sorry

end expression_equality_l2531_253158


namespace plane_through_three_points_l2531_253195

/-- A plane passing through three points in 3D space. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane. -/
def Point3D.liesOn (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The three given points. -/
def P₀ : Point3D := ⟨2, -1, 2⟩
def P₁ : Point3D := ⟨4, 3, 0⟩
def P₂ : Point3D := ⟨5, 2, 1⟩

/-- The plane equation we want to prove. -/
def targetPlane : Plane3D := ⟨1, -2, -3, 2⟩

theorem plane_through_three_points :
  P₀.liesOn targetPlane ∧ P₁.liesOn targetPlane ∧ P₂.liesOn targetPlane := by
  sorry


end plane_through_three_points_l2531_253195


namespace basketball_baseball_volume_ratio_l2531_253103

theorem basketball_baseball_volume_ratio : 
  ∀ (r R : ℝ), r > 0 → R = 4 * r → 
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
sorry

end basketball_baseball_volume_ratio_l2531_253103


namespace total_distance_traveled_l2531_253101

theorem total_distance_traveled (XY XZ : ℝ) (h1 : XY = 4500) (h2 : XZ = 4000) : 
  XY + Real.sqrt (XY^2 - XZ^2) + XZ = 10562 := by
sorry

end total_distance_traveled_l2531_253101


namespace art_gallery_total_pieces_l2531_253122

theorem art_gallery_total_pieces : 
  ∀ (D S : ℕ),
  (2 : ℚ) / 5 * D + (3 : ℚ) / 7 * S = (D + S) * (2 : ℚ) / 5 →
  (1 : ℚ) / 5 * D + (2 : ℚ) / 7 * S = 1500 →
  (2 : ℚ) / 5 * D = 600 →
  D + S = 5700 :=
by
  sorry

end art_gallery_total_pieces_l2531_253122


namespace polar_to_cartesian_circle_l2531_253184

/-- The polar equation ρ = cos(π/4 - θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = Real.cos (π/4 - θ) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end polar_to_cartesian_circle_l2531_253184


namespace matrix_equation_solution_l2531_253187

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem matrix_equation_solution :
  ∃! (p q r : ℝ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -9 ∧ q = 0 ∧ r = 54 := by
  sorry

end matrix_equation_solution_l2531_253187


namespace intersection_points_theorem_l2531_253130

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 5)
def g (x : ℝ) : ℝ := 2 * f x
def h (x : ℝ) : ℝ := f (-x) + 2

-- Define the number of intersection points
def a : ℕ := 2  -- number of intersection points between y=f(x) and y=g(x)
def b : ℕ := 1  -- number of intersection points between y=f(x) and y=h(x)

-- Theorem statement
theorem intersection_points_theorem : 10 * a + b = 21 := by sorry

end intersection_points_theorem_l2531_253130
