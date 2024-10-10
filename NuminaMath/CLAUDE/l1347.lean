import Mathlib

namespace complex_fraction_simplification_l1347_134777

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 7 * Complex.I
  let z₂ : ℂ := 2 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -90 / 53 := by sorry

end complex_fraction_simplification_l1347_134777


namespace total_marks_difference_l1347_134706

theorem total_marks_difference (P C M : ℝ) 
  (h1 : P + C + M > P) 
  (h2 : (C + M) / 2 = 75) : 
  P + C + M - P = 150 := by
sorry

end total_marks_difference_l1347_134706


namespace equation_solution_l1347_134776

theorem equation_solution :
  ∃ (x : ℚ), (x + 36) / 3 = (7 - 2*x) / 6 ∧ x = -65 / 4 := by
  sorry

end equation_solution_l1347_134776


namespace complex_magnitude_problem_l1347_134715

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end complex_magnitude_problem_l1347_134715


namespace lunch_cost_before_tip_l1347_134766

/-- Given a 20% tip and a total spending of $60.6, prove that the original cost of the lunch before the tip was $50.5. -/
theorem lunch_cost_before_tip (tip_percentage : Real) (total_spent : Real) (lunch_cost : Real) : 
  tip_percentage = 0.20 →
  total_spent = 60.6 →
  lunch_cost * (1 + tip_percentage) = total_spent →
  lunch_cost = 50.5 := by
sorry

end lunch_cost_before_tip_l1347_134766


namespace basketball_shot_probability_l1347_134713

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) →
  b ∈ (Set.Ioo 0 1) →
  c ∈ (Set.Ioo 0 1) →
  a + b + c = 1 →
  3*a + 2*b = 2 →
  ∀ x y : ℝ, x ∈ (Set.Ioo 0 1) → y ∈ (Set.Ioo 0 1) → x + y < 1 → x * y ≤ a * b →
  a * b ≤ 1/6 :=
by sorry

end basketball_shot_probability_l1347_134713


namespace right_triangle_hypotenuse_l1347_134765

/-- A right triangle with perimeter 60 and area 48 has a hypotenuse of length 28.4 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  a + b + c = 60 ∧  -- perimeter is 60
  (1/2) * a * b = 48 ∧  -- area is 48
  c = 28.4 :=  -- hypotenuse is 28.4
by sorry

end right_triangle_hypotenuse_l1347_134765


namespace modulus_of_z_equals_sqrt_two_l1347_134760

theorem modulus_of_z_equals_sqrt_two :
  let z : ℂ := (Complex.I + 1) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_equals_sqrt_two_l1347_134760


namespace fraction_sum_l1347_134757

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 14 = (33 : ℚ) / 56 := by
  sorry

end fraction_sum_l1347_134757


namespace square_side_length_l1347_134722

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side ^ 2 = area → side = 1 / 3 := by
  sorry

end square_side_length_l1347_134722


namespace intersection_M_N_l1347_134710

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end intersection_M_N_l1347_134710


namespace moles_of_Cu_CN_2_formed_l1347_134749

/-- Represents a chemical species in a reaction -/
inductive Species
| HCN
| CuSO4
| Cu_CN_2
| H2SO4

/-- Represents the coefficients of a balanced chemical equation -/
structure BalancedEquation :=
(reactants : Species → ℕ)
(products : Species → ℕ)

/-- Represents the available moles of each species -/
structure AvailableMoles :=
(moles : Species → ℝ)

def reaction : BalancedEquation :=
{ reactants := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
, products := λ s => match s with
  | Species.Cu_CN_2 => 1
  | Species.H2SO4 => 1
  | _ => 0
}

def available : AvailableMoles :=
{ moles := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
}

/-- Calculates the moles of product formed based on the limiting reactant -/
def moles_of_product (eq : BalancedEquation) (avail : AvailableMoles) (product : Species) : ℝ :=
sorry

theorem moles_of_Cu_CN_2_formed :
  moles_of_product reaction available Species.Cu_CN_2 = 1 :=
sorry

end moles_of_Cu_CN_2_formed_l1347_134749


namespace expand_product_l1347_134730

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l1347_134730


namespace goldfish_red_balls_l1347_134795

/-- Given a fish tank with goldfish and platyfish, prove the number of red balls each goldfish plays with -/
theorem goldfish_red_balls 
  (total_balls : ℕ) 
  (num_goldfish : ℕ) 
  (num_platyfish : ℕ) 
  (white_balls_per_platyfish : ℕ) 
  (h1 : total_balls = 80) 
  (h2 : num_goldfish = 3) 
  (h3 : num_platyfish = 10) 
  (h4 : white_balls_per_platyfish = 5) : 
  (total_balls - num_platyfish * white_balls_per_platyfish) / num_goldfish = 10 := by
  sorry

end goldfish_red_balls_l1347_134795


namespace midpoint_trajectory_l1347_134773

/-- The trajectory of the midpoint of a chord on a circle -/
theorem midpoint_trajectory (k x y : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Line equation
    (k * x₁ - y₁ + 1 = 0) ∧ (k * x₂ - y₂ + 1 = 0) ∧
    -- Circle equation
    (x₁^2 + y₁^2 = 1) ∧ (x₂^2 + y₂^2 = 1) ∧
    -- (x, y) is the midpoint of (x₁, y₁) and (x₂, y₂)
    (x = (x₁ + x₂) / 2) ∧ (y = (y₁ + y₂) / 2)) →
  x^2 + y^2 - y = 0 :=
by sorry

end midpoint_trajectory_l1347_134773


namespace proposition_ranges_l1347_134768

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, -3 < x ∧ x < 1 → x^2 + 4*x + 9 - m > 0

def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*m*x + 1 < 0

theorem proposition_ranges (m : ℝ) :
  (prop_p m ↔ m < 5) ∧
  (prop_p m ≠ prop_q m ↔ m ≤ 1 ∨ m ≥ 5) :=
sorry

end proposition_ranges_l1347_134768


namespace pure_imaginary_complex_number_l1347_134759

theorem pure_imaginary_complex_number (a : ℝ) :
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end pure_imaginary_complex_number_l1347_134759


namespace least_candies_eleven_candies_maria_candies_l1347_134711

theorem least_candies (c : ℕ) : c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 → c ≥ 11 :=
by sorry

theorem eleven_candies : 11 % 3 = 2 ∧ 11 % 4 = 3 ∧ 11 % 6 = 5 :=
by sorry

theorem maria_candies : ∃ (c : ℕ), c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 ∧ c = 11 :=
by sorry

end least_candies_eleven_candies_maria_candies_l1347_134711


namespace sphere_surface_area_l1347_134798

theorem sphere_surface_area (r₁ r₂ d R : ℝ) : 
  r₁ > 0 → r₂ > 0 → d > 0 → R > 0 →
  r₁^2 * π = 9 * π →
  r₂^2 * π = 16 * π →
  d = 1 →
  R^2 = r₂^2 + (R - d)^2 →
  R^2 = r₁^2 + R^2 →
  4 * π * R^2 = 100 * π := by
sorry

end sphere_surface_area_l1347_134798


namespace floor_a_equals_four_l1347_134781

theorem floor_a_equals_four (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) :
  let a := Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)
  ⌊a⌋ = 4 := by sorry

end floor_a_equals_four_l1347_134781


namespace cupcake_production_difference_l1347_134783

def cupcake_difference (betty_rate : ℕ) (dora_rate : ℕ) (total_time : ℕ) (break_time : ℕ) : ℕ :=
  (dora_rate * total_time) - (betty_rate * (total_time - break_time))

theorem cupcake_production_difference :
  cupcake_difference 10 8 5 2 = 10 := by
  sorry

end cupcake_production_difference_l1347_134783


namespace sum_of_ages_at_milestone_l1347_134786

-- Define the ages of Hans, Josiah, and Julia
def hans_age : ℕ := 15
def josiah_age : ℕ := 3 * hans_age
def julia_age : ℕ := hans_age - 5

-- Define Julia's age when Hans was born
def julia_age_at_hans_birth : ℕ := julia_age / 2

-- Define Josiah's age when Julia was half her current age
def josiah_age_at_milestone : ℕ := josiah_age - hans_age - julia_age_at_hans_birth

-- Theorem statement
theorem sum_of_ages_at_milestone : 
  josiah_age_at_milestone + julia_age_at_hans_birth + 0 = 30 := by
  sorry

end sum_of_ages_at_milestone_l1347_134786


namespace anthony_percentage_more_than_mabel_l1347_134758

theorem anthony_percentage_more_than_mabel :
  ∀ (mabel anthony cal jade : ℕ),
    mabel = 90 →
    cal = (2 * anthony) / 3 →
    jade = cal + 18 →
    jade = 84 →
    (anthony : ℚ) / mabel = 11 / 10 :=
by
  sorry

end anthony_percentage_more_than_mabel_l1347_134758


namespace final_values_l1347_134774

def program_execution (a b : Int) : Int × Int :=
  let a' := a + b
  let b' := a' - b
  (a', b')

theorem final_values : program_execution 1 3 = (4, 1) := by
  sorry

end final_values_l1347_134774


namespace smallest_prime_factor_of_175_l1347_134755

theorem smallest_prime_factor_of_175 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 175 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 175 → p ≤ q :=
by sorry

end smallest_prime_factor_of_175_l1347_134755


namespace insulated_cups_problem_l1347_134709

-- Define the cost prices and quantities
def cost_A : ℝ := 110
def cost_B : ℝ := 88
def quantity_A : ℕ := 30
def quantity_B : ℕ := 50

-- Define the selling prices
def sell_A : ℝ := 160
def sell_B : ℝ := 140

-- Define the total number of cups and profit
def total_cups : ℕ := 80
def total_profit : ℝ := 4100

-- Theorem statement
theorem insulated_cups_problem :
  -- Condition 1: 4 type A cups cost the same as 5 type B cups
  4 * cost_A = 5 * cost_B ∧
  -- Condition 2: 3 type A cups cost $154 more than 2 type B cups
  3 * cost_A = 2 * cost_B + 154 ∧
  -- Condition 3: Total cups purchased is 80
  quantity_A + quantity_B = total_cups ∧
  -- Condition 4: Profit calculation
  (sell_A - cost_A) * quantity_A + (sell_B - cost_B) * quantity_B = total_profit :=
by
  sorry


end insulated_cups_problem_l1347_134709


namespace erik_money_left_l1347_134727

-- Define the problem parameters
def initial_money : ℚ := 86
def bread_price : ℚ := 3
def juice_price : ℚ := 6
def eggs_price : ℚ := 4
def chocolate_price : ℚ := 2
def apples_price : ℚ := 1.25
def grapes_price : ℚ := 2.50

def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def eggs_quantity : ℕ := 2
def chocolate_quantity : ℕ := 5
def apples_quantity : ℚ := 4
def grapes_quantity : ℚ := 1.5

def bread_eggs_discount : ℚ := 0.10
def other_items_discount : ℚ := 0.05
def sales_tax_rate : ℚ := 0.06

-- Define the theorem
theorem erik_money_left : 
  let total_cost := bread_price * bread_quantity + juice_price * juice_quantity + 
                    eggs_price * eggs_quantity + chocolate_price * chocolate_quantity + 
                    apples_price * apples_quantity + grapes_price * grapes_quantity
  let bread_eggs_cost := bread_price * bread_quantity + eggs_price * eggs_quantity
  let other_items_cost := total_cost - bread_eggs_cost
  let discounted_cost := total_cost - (bread_eggs_cost * bread_eggs_discount) - 
                         (other_items_cost * other_items_discount)
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  initial_money - final_cost = 32.78 := by
  sorry


end erik_money_left_l1347_134727


namespace additional_hovering_time_l1347_134735

/-- Represents the hovering time of a plane in different time zones over two days. -/
structure PlaneHoveringTime where
  mountain_day1 : ℕ
  central_day1 : ℕ
  eastern_day1 : ℕ
  mountain_day2 : ℕ
  central_day2 : ℕ
  eastern_day2 : ℕ

/-- Theorem stating that given the conditions of the problem, the additional hovering time
    in each time zone on the second day is 5 hours. -/
theorem additional_hovering_time
  (h : PlaneHoveringTime)
  (h_mountain_day1 : h.mountain_day1 = 3)
  (h_central_day1 : h.central_day1 = 4)
  (h_eastern_day1 : h.eastern_day1 = 2)
  (h_total_time : h.mountain_day1 + h.central_day1 + h.eastern_day1 +
                  h.mountain_day2 + h.central_day2 + h.eastern_day2 = 24)
  (h_equal_additional : h.mountain_day2 = h.central_day2 ∧ h.central_day2 = h.eastern_day2) :
  h.mountain_day2 = 5 ∧ h.central_day2 = 5 ∧ h.eastern_day2 = 5 :=
by
  sorry

end additional_hovering_time_l1347_134735


namespace intersection_point_coordinates_l1347_134742

theorem intersection_point_coordinates
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let line1 := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let line2 := {(x, y) : ℝ × ℝ | b * x + c * y = a}
  let line3 := {(x, y) : ℝ × ℝ | y = 2 * x}
  (∀ (p q : ℝ × ℝ), p ∈ line1 ∧ q ∈ line2 → (p.1 - q.1) * (p.2 - q.2) = -1) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3 ∧ P = (-3/5, -6/5)) :=
by sorry

end intersection_point_coordinates_l1347_134742


namespace complex_modulus_equality_l1347_134754

theorem complex_modulus_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : x + 3 * i = 2 + y * i) : Complex.abs (x + y * i) = Real.sqrt 13 := by
  sorry

end complex_modulus_equality_l1347_134754


namespace function_has_two_zeros_l1347_134787

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem function_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end function_has_two_zeros_l1347_134787


namespace additional_amount_for_free_shipping_l1347_134790

/-- The cost of the first book -/
def book1_cost : ℚ := 13

/-- The cost of the second book -/
def book2_cost : ℚ := 15

/-- The cost of the third and fourth books -/
def book34_cost : ℚ := 10

/-- The discount rate applied to the first two books -/
def discount_rate : ℚ := 1/4

/-- The free shipping threshold -/
def free_shipping_threshold : ℚ := 50

/-- Calculate the discounted price of a book -/
def apply_discount (price : ℚ) : ℚ :=
  price * (1 - discount_rate)

/-- Calculate the total cost of all four books with discounts applied -/
def total_cost : ℚ :=
  apply_discount book1_cost + apply_discount book2_cost + 2 * book34_cost

/-- The theorem stating the additional amount needed for free shipping -/
theorem additional_amount_for_free_shipping :
  free_shipping_threshold - total_cost = 9 := by sorry

end additional_amount_for_free_shipping_l1347_134790


namespace sum_of_squares_of_roots_l1347_134793

theorem sum_of_squares_of_roots (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → a^2 + b^2 = 48 :=
by
  sorry

end sum_of_squares_of_roots_l1347_134793


namespace thirtieth_term_of_sequence_l1347_134731

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 := by
  sorry

end thirtieth_term_of_sequence_l1347_134731


namespace no_feasible_distribution_no_feasible_distribution_proof_l1347_134712

/-- Represents a cricket player with their initial average runs and desired increase --/
structure Player where
  initialAvg : ℕ
  desiredIncrease : ℕ

/-- Theorem stating that no feasible distribution exists for the given problem --/
theorem no_feasible_distribution 
  (playerA : Player) 
  (playerB : Player) 
  (playerC : Player) 
  (totalRunsLimit : ℕ) : Prop :=
  playerA.initialAvg = 32 ∧ 
  playerA.desiredIncrease = 4 ∧
  playerB.initialAvg = 45 ∧ 
  playerB.desiredIncrease = 5 ∧
  playerC.initialAvg = 55 ∧ 
  playerC.desiredIncrease = 6 ∧
  totalRunsLimit = 250 →
  ¬∃ (runsA runsB runsC : ℕ),
    (runsA + runsB + runsC ≤ totalRunsLimit) ∧
    ((playerA.initialAvg * 10 + runsA) / 11 ≥ playerA.initialAvg + playerA.desiredIncrease) ∧
    ((playerB.initialAvg * 10 + runsB) / 11 ≥ playerB.initialAvg + playerB.desiredIncrease) ∧
    ((playerC.initialAvg * 10 + runsC) / 11 ≥ playerC.initialAvg + playerC.desiredIncrease)

/-- The proof of the theorem --/
theorem no_feasible_distribution_proof : no_feasible_distribution 
  { initialAvg := 32, desiredIncrease := 4 }
  { initialAvg := 45, desiredIncrease := 5 }
  { initialAvg := 55, desiredIncrease := 6 }
  250 := by
  sorry

end no_feasible_distribution_no_feasible_distribution_proof_l1347_134712


namespace folded_area_ratio_paper_folding_problem_l1347_134772

/-- Represents a rectangular piece of paper. -/
structure Paper where
  length : ℝ
  width : ℝ
  area : ℝ
  widthIsSquareRootTwo : width = Real.sqrt 2 * length
  areaIsLengthTimesWidth : area = length * width

/-- Represents the paper after folding. -/
structure FoldedPaper where
  original : Paper
  foldedArea : ℝ

/-- The ratio of the folded area to the original area is (16 - √6) / 16. -/
theorem folded_area_ratio (p : Paper) (fp : FoldedPaper) 
    (h : fp.original = p) : 
    fp.foldedArea / p.area = (16 - Real.sqrt 6) / 16 := by
  sorry

/-- Main theorem stating the result of the problem. -/
theorem paper_folding_problem :
  ∃ (p : Paper) (fp : FoldedPaper), 
    fp.original = p ∧ fp.foldedArea / p.area = (16 - Real.sqrt 6) / 16 := by
  sorry

end folded_area_ratio_paper_folding_problem_l1347_134772


namespace coprime_20172019_l1347_134733

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end coprime_20172019_l1347_134733


namespace remainder_three_power_2010_mod_8_l1347_134723

theorem remainder_three_power_2010_mod_8 : 3^2010 % 8 = 1 := by
  sorry

end remainder_three_power_2010_mod_8_l1347_134723


namespace p_sufficient_not_necessary_for_q_l1347_134743

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x - 1| < 2 → x^2 - 5*x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - 5*x - 6 < 0 ∧ ¬(|x - 1| < 2)) := by
  sorry

end p_sufficient_not_necessary_for_q_l1347_134743


namespace project_over_budget_proof_l1347_134725

/-- Calculates the amount a project is over budget given the total budget, 
    number of months, months passed, and actual expenditure. -/
def project_over_budget (total_budget : ℚ) (num_months : ℕ) 
                        (months_passed : ℕ) (actual_expenditure : ℚ) : ℚ :=
  actual_expenditure - (total_budget / num_months) * months_passed

/-- Proves that given the specific conditions of the problem, 
    the project is over budget by $280. -/
theorem project_over_budget_proof : 
  project_over_budget 12600 12 6 6580 = 280 := by
  sorry

#eval project_over_budget 12600 12 6 6580

end project_over_budget_proof_l1347_134725


namespace polynomial_sum_simplification_l1347_134746

theorem polynomial_sum_simplification :
  let p₁ : Polynomial ℚ := 2 * X^5 - 3 * X^3 + 5 * X^2 - 4 * X + 6
  let p₂ : Polynomial ℚ := -X^5 + 4 * X^4 - 2 * X^3 - X^2 + 3 * X - 8
  p₁ + p₂ = X^5 + 4 * X^4 - 5 * X^3 + 4 * X^2 - X - 2 := by
  sorry

end polynomial_sum_simplification_l1347_134746


namespace penny_fountain_problem_l1347_134751

theorem penny_fountain_problem (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  rachelle + gretchen + rocky = 300 :=
by sorry

end penny_fountain_problem_l1347_134751


namespace cyclic_quadrilateral_properties_l1347_134717

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  R : ℝ  -- circumradius
  a : ℝ  -- side length
  b : ℝ  -- side length
  c : ℝ  -- side length
  d : ℝ  -- side length
  S : ℝ  -- area
  positive_R : R > 0
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  positive_S : S > 0

-- Define what it means for a cyclic quadrilateral to be a square
def is_square (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_properties (q : CyclicQuadrilateral) :
  (16 * q.R^2 * q.S^2 = (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)) ∧
  (q.R * q.S * Real.sqrt 2 ≥ (q.a * q.b * q.c * q.d)^(3/4)) ∧
  (q.R * q.S * Real.sqrt 2 = (q.a * q.b * q.c * q.d)^(3/4) ↔ is_square q) :=
sorry

end cyclic_quadrilateral_properties_l1347_134717


namespace square_dissection_interior_rectangle_l1347_134726

-- Define a rectangle type
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

-- Define the square dissection
def SquareDissection (n : ℕ) (rectangles : Finset Rectangle) : Prop :=
  n > 1 ∧
  rectangles.card = n ∧
  (∀ r ∈ rectangles, r.x ≥ 0 ∧ r.y ≥ 0 ∧ r.x + r.width ≤ 1 ∧ r.y + r.height ≤ 1) ∧
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
    ∃ r ∈ rectangles, r.x < x ∧ x < r.x + r.width ∧ r.y < y ∧ y < r.y + r.height)

-- Define an interior rectangle
def InteriorRectangle (r : Rectangle) : Prop :=
  r.x > 0 ∧ r.y > 0 ∧ r.x + r.width < 1 ∧ r.y + r.height < 1

-- The theorem to be proved
theorem square_dissection_interior_rectangle
  (n : ℕ) (rectangles : Finset Rectangle) (h : SquareDissection n rectangles) :
  ∃ r ∈ rectangles, InteriorRectangle r := by
  sorry

end square_dissection_interior_rectangle_l1347_134726


namespace student_selection_permutation_l1347_134796

theorem student_selection_permutation :
  (Nat.factorial 6) / (Nat.factorial 4) = 30 := by
  sorry

end student_selection_permutation_l1347_134796


namespace animal_distance_calculation_l1347_134770

/-- Calculates the total distance covered by a fox, rabbit, and deer running at their maximum speeds for 120 minutes. -/
theorem animal_distance_calculation :
  let fox_speed : ℝ := 50  -- km/h
  let rabbit_speed : ℝ := 60  -- km/h
  let deer_speed : ℝ := 80  -- km/h
  let time_hours : ℝ := 120 / 60  -- Convert 120 minutes to hours
  let fox_distance := fox_speed * time_hours
  let rabbit_distance := rabbit_speed * time_hours
  let deer_distance := deer_speed * time_hours
  let total_distance := fox_distance + rabbit_distance + deer_distance
  total_distance = 380  -- km
  := by sorry

end animal_distance_calculation_l1347_134770


namespace jenny_distance_difference_l1347_134794

theorem jenny_distance_difference : 
  ∀ (run_distance walk_distance : ℝ),
    run_distance = 0.6 →
    walk_distance = 0.4 →
    run_distance - walk_distance = 0.2 :=
by
  sorry

end jenny_distance_difference_l1347_134794


namespace news_watching_probability_l1347_134719

/-- Represents a survey conducted in a town -/
structure TownSurvey where
  total_population : ℕ
  sample_size : ℕ
  news_watchers : ℕ

/-- Calculates the probability of a random person watching the news based on survey results -/
def probability_watch_news (survey : TownSurvey) : ℚ :=
  survey.news_watchers / survey.sample_size

/-- Theorem stating the probability of watching news for the given survey -/
theorem news_watching_probability (survey : TownSurvey) 
  (h1 : survey.total_population = 100000)
  (h2 : survey.sample_size = 2000)
  (h3 : survey.news_watchers = 250) :
  probability_watch_news survey = 1/8 := by
  sorry

end news_watching_probability_l1347_134719


namespace carla_water_calculation_l1347_134700

/-- The amount of water Carla needs to bring for her animals -/
def water_needed (pig_count : ℕ) (horse_count : ℕ) (pig_water : ℕ) (chicken_tank : ℕ) : ℕ :=
  let pig_total := pig_count * pig_water
  let horse_total := horse_count * (2 * pig_water)
  pig_total + horse_total + chicken_tank

/-- Theorem stating the total amount of water Carla needs -/
theorem carla_water_calculation :
  water_needed 8 10 3 30 = 114 := by
  sorry

end carla_water_calculation_l1347_134700


namespace jung_mi_number_problem_l1347_134729

theorem jung_mi_number_problem :
  ∃ x : ℚ, (-4/5) * (x + (-2/3)) = -1/2 ∧ x = 31/24 := by
  sorry

end jung_mi_number_problem_l1347_134729


namespace complex_modulus_theorem_l1347_134707

theorem complex_modulus_theorem : Complex.abs (-6 + (9/4) * Complex.I) = (Real.sqrt 657) / 4 := by
  sorry

end complex_modulus_theorem_l1347_134707


namespace fraction_problem_l1347_134714

theorem fraction_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 6) :
  d / a = 1 / 15 := by
  sorry

end fraction_problem_l1347_134714


namespace investment_interest_proof_l1347_134701

/-- Calculates the total interest earned on an investment with compound interest. -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on a $2000 investment at 8% annual interest
    compounded annually for 5 years is approximately $938.656. -/
theorem investment_interest_proof :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.656) < 0.001 := by
  sorry

#eval totalInterestEarned 2000 0.08 5

end investment_interest_proof_l1347_134701


namespace roots_properties_l1347_134708

theorem roots_properties (x₁ x₂ : ℝ) (h : x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) : 
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end roots_properties_l1347_134708


namespace water_depth_multiple_l1347_134780

/-- Given Dean's height and the water depth, prove that the multiple of Dean's height
    representing the water depth is 10. -/
theorem water_depth_multiple (dean_height water_depth : ℝ) 
  (h1 : dean_height = 6)
  (h2 : water_depth = 60) :
  water_depth / dean_height = 10 := by
  sorry

end water_depth_multiple_l1347_134780


namespace yellow_marble_probability_l1347_134752

/-- Represents a bag of marbles with counts for different colors -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the probability of drawing a yellow marble as the last marble
    given the contents of bags A, B, C, and D and the described drawing process -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black
  let totalB := bagB.yellow + bagB.blue
  let totalC := bagC.yellow + bagC.blue
  let totalD := bagD.yellow + bagD.blue
  
  let probWhiteA := bagA.white / totalA
  let probBlackA := bagA.black / totalA
  let probYellowB := bagB.yellow / totalB
  let probBlueC := bagC.blue / totalC
  let probYellowC := bagC.yellow / totalC
  let probYellowD := bagD.yellow / totalD
  
  probWhiteA * probYellowB + 
  probBlackA * probBlueC * probYellowD + 
  probBlackA * probYellowC

theorem yellow_marble_probability :
  let bagA : Bag := { white := 5, black := 6 }
  let bagB : Bag := { yellow := 8, blue := 6 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 1, blue := 4 }
  yellowProbability bagA bagB bagC bagD = 136 / 275 := by
  sorry

end yellow_marble_probability_l1347_134752


namespace coefficient_of_x_squared_l1347_134785

def expression (x : ℝ) : ℝ := 6 * (x - 2 * x^3) - 5 * (2 * x^2 - 3 * x^3 + 2 * x^4) + 3 * (3 * x^2 - 2 * x^6)

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ), 
    (∀ x, expression x = a * x + (-1) * x^2 + c * x^3 + d * x^4 + e * x^6 + f) :=
by
  sorry

end coefficient_of_x_squared_l1347_134785


namespace sum_of_solutions_eq_eight_l1347_134789

theorem sum_of_solutions_eq_eight : 
  ∃ (x y : ℝ), x * (x - 8) = 7 ∧ y * (y - 8) = 7 ∧ x + y = 8 := by
  sorry

end sum_of_solutions_eq_eight_l1347_134789


namespace packs_per_box_l1347_134799

/-- Given that Jenny sold 24.0 boxes of Trefoils and 192 packs in total,
    prove that there are 8 packs in each box. -/
theorem packs_per_box (boxes : ℝ) (total_packs : ℕ) 
    (h1 : boxes = 24.0) 
    (h2 : total_packs = 192) : 
  (total_packs : ℝ) / boxes = 8 := by
  sorry

end packs_per_box_l1347_134799


namespace coin_flip_probability_l1347_134737

theorem coin_flip_probability : 
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- minimum number of heads we're interested in
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := (Finset.range (n - k + 1)).sum (λ i => Nat.choose n (k + i))
  (favorable_outcomes : ℚ) / total_outcomes = 1/2 := by
  sorry

end coin_flip_probability_l1347_134737


namespace power_of_two_divides_factorial_l1347_134779

theorem power_of_two_divides_factorial (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
sorry

end power_of_two_divides_factorial_l1347_134779


namespace cyclic_sum_inequality_l1347_134745

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) / Real.sqrt ((x^2 + x*z + z^2) * (y^2 + y*z + z^2)) +
  (y * z) / Real.sqrt ((y^2 + y*x + x^2) * (z^2 + z*x + x^2)) +
  (z * x) / Real.sqrt ((z^2 + z*y + y^2) * (x^2 + x*y + y^2)) ≥ 1 := by
sorry

end cyclic_sum_inequality_l1347_134745


namespace odd_integers_equality_l1347_134761

theorem odd_integers_equality (a b : ℕ) (ha : Odd a) (hb : Odd b) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (h_div : (2 * a * b + 1) ∣ (a^2 + b^2 + 1)) : a = b := by
  sorry

end odd_integers_equality_l1347_134761


namespace base_7_digits_of_1234_l1347_134739

theorem base_7_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end base_7_digits_of_1234_l1347_134739


namespace largest_box_volume_l1347_134797

/-- The volume of the largest rectangular parallelopiped that can be enclosed in a cylindrical container with a hemispherical lid. -/
theorem largest_box_volume (total_height radius : ℝ) (h_total_height : total_height = 60) (h_radius : radius = 30) :
  let cylinder_height : ℝ := total_height - radius
  let box_base_side : ℝ := 2 * radius
  let box_height : ℝ := cylinder_height
  let box_volume : ℝ := box_base_side^2 * box_height
  box_volume = 108000 := by
  sorry

#check largest_box_volume

end largest_box_volume_l1347_134797


namespace absolute_difference_equals_one_l1347_134791

theorem absolute_difference_equals_one (x y : ℝ) :
  |x| - |y| = 1 ↔
  ((y = x - 1 ∧ x ≥ 1) ∨
   (y = 1 - x ∧ x ≥ 1) ∨
   (y = -x - 1 ∧ x ≤ -1) ∨
   (y = x + 1 ∧ x ≤ -1)) :=
by sorry

end absolute_difference_equals_one_l1347_134791


namespace line_parallel_perpendicular_implies_planes_perpendicular_l1347_134703

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → planes_perpendicular α β :=
by sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l1347_134703


namespace tiffany_treasures_l1347_134782

theorem tiffany_treasures (points_per_treasure : ℕ) (first_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 6 →
  first_level_treasures = 3 →
  total_score = 48 →
  (total_score - points_per_treasure * first_level_treasures) / points_per_treasure = 5 :=
by sorry

end tiffany_treasures_l1347_134782


namespace stone_fall_time_exists_stone_fall_time_approx_l1347_134705

theorem stone_fall_time_exists : ∃ s : ℝ, s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0 := by
  sorry

theorem stone_fall_time_approx (s : ℝ) (hs : s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0) : 
  ∃ ε > 0, |s - 3.82| < ε := by
  sorry

end stone_fall_time_exists_stone_fall_time_approx_l1347_134705


namespace vector_equality_l1347_134724

/-- Given two vectors in ℝ², prove that if their sum and difference have equal magnitudes, 
    then the second component of the second vector must be 3/2. -/
theorem vector_equality (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) 
    (ha : a = (1, 2)) (hb : b.1 = -3) : b.2 = 3/2 := by
  sorry

end vector_equality_l1347_134724


namespace two_digit_numbers_divisibility_l1347_134764

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 2019

/-- The set of possible digits -/
def digits : Set ℕ := {d | 0 ≤ d ∧ d ≤ 9}

/-- A function that counts the number of n-digit numbers made of 2 different digits -/
noncomputable def count_two_digit_numbers (n : ℕ) : ℕ :=
  sorry

/-- The highest power of 3 that divides a natural number -/
noncomputable def highest_power_of_three (m : ℕ) : ℕ :=
  sorry

theorem two_digit_numbers_divisibility :
  highest_power_of_three (count_two_digit_numbers n) = 5 := by
  sorry

end two_digit_numbers_divisibility_l1347_134764


namespace partner_investment_duration_l1347_134740

/-- Given two partners P and Q with investments and profits, calculate Q's investment duration -/
theorem partner_investment_duration
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (p_duration : ℕ)
  (h_investment : investment_ratio_p = 7 ∧ investment_ratio_q = 5)
  (h_profit : profit_ratio_p = 7 ∧ profit_ratio_q = 14)
  (h_p_duration : p_duration = 5) :
  ∃ q_duration : ℕ,
    q_duration = 14 ∧
    (investment_ratio_p * p_duration) / (investment_ratio_q * q_duration) =
    profit_ratio_p / profit_ratio_q :=
by sorry

end partner_investment_duration_l1347_134740


namespace math_test_questions_l1347_134721

/-- Proves that the total number of questions in a math test is 60 -/
theorem math_test_questions : ∃ N : ℕ,
  (N : ℚ) * (80 : ℚ) / 100 + 35 - N / 2 = N - 7 ∧
  N = 60 := by
  sorry

end math_test_questions_l1347_134721


namespace quadratic_solution_sum_l1347_134741

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 34 → 
  x = Real.sqrt a - b → 
  a + b = 64 := by
sorry

end quadratic_solution_sum_l1347_134741


namespace remaining_money_after_expenses_l1347_134771

def rent : ℝ := 1200
def salary : ℝ := 5000

theorem remaining_money_after_expenses :
  let food_and_travel := 2 * rent
  let shared_rent := rent / 2
  let total_expenses := food_and_travel + shared_rent
  salary - total_expenses = 2000 := by sorry

end remaining_money_after_expenses_l1347_134771


namespace f_2009_equals_one_l1347_134748

-- Define the function f
axiom f : ℝ → ℝ

-- Define the conditions
axiom func_prop : ∀ x y : ℝ, f (x * y) = f x * f y
axiom f0_nonzero : f 0 ≠ 0

-- State the theorem
theorem f_2009_equals_one : f 2009 = 1 := by
  sorry

end f_2009_equals_one_l1347_134748


namespace printer_price_ratio_printer_price_ratio_proof_l1347_134718

/-- The ratio of the printer price to the total price of enhanced computer and printer -/
theorem printer_price_ratio : ℚ :=
let basic_computer_price : ℕ := 2000
let basic_total_price : ℕ := 2500
let price_difference : ℕ := 500
let printer_price : ℕ := basic_total_price - basic_computer_price
let enhanced_computer_price : ℕ := basic_computer_price + price_difference
let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
1 / 6

theorem printer_price_ratio_proof :
  let basic_computer_price : ℕ := 2000
  let basic_total_price : ℕ := 2500
  let price_difference : ℕ := 500
  let printer_price : ℕ := basic_total_price - basic_computer_price
  let enhanced_computer_price : ℕ := basic_computer_price + price_difference
  let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
  (printer_price : ℚ) / enhanced_total_price = 1 / 6 := by
  sorry

end printer_price_ratio_printer_price_ratio_proof_l1347_134718


namespace s₂_is_zero_l1347_134720

-- Define the polynomial division operation
def poly_div (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ := sorry

-- Define p₁(x) and s₁
def p₁_and_s₁ : (ℝ → ℝ) × ℝ := poly_div (λ x => x^6) (λ x => x - 1/2)

def p₁ : ℝ → ℝ := (p₁_and_s₁.1)
def s₁ : ℝ := (p₁_and_s₁.2)

-- Define p₂(x) and s₂
def p₂_and_s₂ : (ℝ → ℝ) × ℝ := poly_div p₁ (λ x => x - 1/2)

def p₂ : ℝ → ℝ := (p₂_and_s₂.1)
def s₂ : ℝ := (p₂_and_s₂.2)

-- The theorem to prove
theorem s₂_is_zero : s₂ = 0 := by sorry

end s₂_is_zero_l1347_134720


namespace distance_is_134_div_7_l1347_134732

/-- The distance from a point to a plane defined by three points -/
def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The points given in the problem -/
def M₀ : ℝ × ℝ × ℝ := (-13, -8, 16)
def M₁ : ℝ × ℝ × ℝ := (1, 2, 0)
def M₂ : ℝ × ℝ × ℝ := (3, 0, -3)
def M₃ : ℝ × ℝ × ℝ := (5, 2, 6)

/-- The theorem stating that the distance is equal to 134/7 -/
theorem distance_is_134_div_7 : distance_point_to_plane M₀ M₁ M₂ M₃ = 134 / 7 := by sorry

end distance_is_134_div_7_l1347_134732


namespace sum_of_odd_coefficients_l1347_134769

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₃ + a₅ = 32 := by
sorry

end sum_of_odd_coefficients_l1347_134769


namespace victors_total_money_l1347_134704

/-- Victor's initial money in dollars -/
def initial_money : ℕ := 10

/-- Victor's allowance in dollars -/
def allowance : ℕ := 8

/-- Theorem: Victor's total money is $18 -/
theorem victors_total_money : initial_money + allowance = 18 := by
  sorry

end victors_total_money_l1347_134704


namespace marks_score_ratio_l1347_134747

theorem marks_score_ratio (highest_score range marks_score : ℕ) : 
  highest_score = 98 →
  range = 75 →
  marks_score = 46 →
  marks_score % (highest_score - range) = 0 →
  marks_score / (highest_score - range) = 2 :=
by sorry

end marks_score_ratio_l1347_134747


namespace mean_temperature_is_84_l1347_134738

def temperatures : List ℝ := [82, 84, 83, 85, 86]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length) = 84 := by
  sorry

end mean_temperature_is_84_l1347_134738


namespace sheila_hourly_wage_l1347_134716

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's actual work schedule -/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 360 }

/-- Theorem: Sheila's hourly wage is $10 -/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 10 := by
  sorry

end sheila_hourly_wage_l1347_134716


namespace cubic_factorization_l1347_134736

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end cubic_factorization_l1347_134736


namespace g_lower_bound_l1347_134763

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem g_lower_bound : ∀ x > 0, g x > 4/3 := by
  sorry

end g_lower_bound_l1347_134763


namespace isabel_candy_count_l1347_134788

/-- The total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (from_friend : ℕ) (from_cousin : ℕ) : ℕ :=
  initial + from_friend + from_cousin

/-- Theorem stating the total number of candy pieces Isabel has -/
theorem isabel_candy_count :
  ∀ x : ℕ, total_candy 216 137 x = 353 + x :=
by sorry

end isabel_candy_count_l1347_134788


namespace lemonade_scaling_l1347_134775

/-- Lemonade recipe and scaling -/
theorem lemonade_scaling (lemons : ℕ) (sugar : ℚ) :
  (30 : ℚ) / 40 = lemons / 10 →
  (2 : ℚ) / 5 = sugar / 10 →
  lemons = 8 ∧ sugar = 4 := by
  sorry

#check lemonade_scaling

end lemonade_scaling_l1347_134775


namespace solve_equation_l1347_134762

-- Define the operation "*"
def star (a b : ℝ) : ℝ := 2 * a - b

-- Theorem statement
theorem solve_equation (x : ℝ) (h : star x (star 2 1) = 3) : x = 3 := by
  sorry

end solve_equation_l1347_134762


namespace calculate_expression_l1347_134792

theorem calculate_expression : 18 * 35 + 45 * 18 - 18 * 10 = 1260 := by
  sorry

end calculate_expression_l1347_134792


namespace unique_satisfying_function_l1347_134702

/-- The property that a function f: ℕ → ℕ must satisfy -/
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (f n) + f (f (f n)) = 3 * n

/-- Theorem stating that the identity function is the only function satisfying the property -/
theorem unique_satisfying_function :
  ∀ f : ℕ → ℕ, SatisfiesProperty f → f = id := by
  sorry

end unique_satisfying_function_l1347_134702


namespace derivative_difference_bound_l1347_134784

variable (f : ℝ → ℝ) (M : ℝ)

theorem derivative_difference_bound
  (h_diff : Differentiable ℝ f)
  (h_pos : M > 0)
  (h_bound : ∀ x t : ℝ, |f (x + t) - 2 * f x + f (x - t)| ≤ M * t^2) :
  ∀ x t : ℝ, |deriv f (x + t) - deriv f x| ≤ M * |t| :=
by sorry

end derivative_difference_bound_l1347_134784


namespace range_of_a_for_union_equality_intersection_A_B_union_A_B_l1347_134767

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem range_of_a_for_union_equality :
  ∀ a : ℝ, (A ∪ C a = C a) ↔ (2 ≤ a ∧ a < 3) :=
by sorry

-- Additional theorems for intersection and union of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} :=
by sorry

theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} :=
by sorry

end range_of_a_for_union_equality_intersection_A_B_union_A_B_l1347_134767


namespace triangle_ratio_l1347_134778

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) - b*sin(B) = 4c*sin(C) and cos(A) = -1/4, then b/c = 6 -/
theorem triangle_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
sorry

end triangle_ratio_l1347_134778


namespace calf_grazing_area_increase_calf_grazing_area_increase_value_l1347_134756

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m -/
theorem calf_grazing_area_increase : ℝ :=
  let initial_length : ℝ := 12
  let final_length : ℝ := 25
  let initial_area := Real.pi * initial_length ^ 2
  let final_area := Real.pi * final_length ^ 2
  final_area - initial_area

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m is 481π m² -/
theorem calf_grazing_area_increase_value : 
  calf_grazing_area_increase = 481 * Real.pi := by sorry

end calf_grazing_area_increase_calf_grazing_area_increase_value_l1347_134756


namespace total_distance_eight_points_circle_l1347_134750

/-- The total distance traveled by 8 points on a circle visiting non-adjacent points -/
theorem total_distance_eight_points_circle (r : ℝ) (h : r = 40) :
  let n := 8
  let distance_two_apart := r * Real.sqrt 2
  let distance_three_apart := r * Real.sqrt (2 + Real.sqrt 2)
  let distance_four_apart := 2 * r
  let single_point_distance := 4 * distance_two_apart + 2 * distance_three_apart + distance_four_apart
  n * single_point_distance = 1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640 :=
by sorry

end total_distance_eight_points_circle_l1347_134750


namespace sin_m_eq_cos_810_l1347_134744

theorem sin_m_eq_cos_810 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.sin (m * π / 180) = Real.cos (810 * π / 180)) :
  m = 0 ∨ m = 180 := by
  sorry

end sin_m_eq_cos_810_l1347_134744


namespace p_plus_q_equals_42_l1347_134734

theorem p_plus_q_equals_42 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 4 → P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4)) →
  P + Q = 42 := by
  sorry

end p_plus_q_equals_42_l1347_134734


namespace randy_tower_blocks_l1347_134728

/-- 
Given:
- Randy has 90 blocks in total
- He uses 89 blocks to build a house
- He uses some blocks to build a tower
- He used 26 more blocks for the house than for the tower

Prove that Randy used 63 blocks to build the tower.
-/
theorem randy_tower_blocks : 
  ∀ (total house tower : ℕ),
  total = 90 →
  house = 89 →
  house = tower + 26 →
  tower = 63 := by
sorry

end randy_tower_blocks_l1347_134728


namespace centroid_of_V_l1347_134753

-- Define the region V
def V : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a region
def centroid (S : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem centroid_of_V :
  centroid V = (0, 2.31) := by
  sorry

end centroid_of_V_l1347_134753
