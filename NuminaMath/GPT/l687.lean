import Mathlib

namespace minimum_length_intersection_l687_68712

def length (a b : ℝ) : ℝ := b - a

def M (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2/3 }
def N (n : ℝ) : Set ℝ := { x | n - 1/2 ≤ x ∧ x ≤ n }

def IntervalSet : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem minimum_length_intersection (m n : ℝ) (hM : M m ⊆ IntervalSet) (hN : N n ⊆ IntervalSet) :
  length (max m (n - 1/2)) (min (m + 2/3) n) = 1/6 :=
by
  sorry

end minimum_length_intersection_l687_68712


namespace smallest_non_six_digit_palindrome_l687_68719

-- Definition of a four-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

-- Definition of a six-digit number
def is_six_digit (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000

-- Definition of a non-palindrome
def not_palindrome (n : ℕ) : Prop :=
  ¬ is_palindrome n

-- Find the smallest four-digit palindrome whose product with 103 is not a six-digit palindrome
theorem smallest_non_six_digit_palindrome :
  ∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ is_palindrome n ∧ not_palindrome (103 * n)
  ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ is_palindrome m ∧ not_palindrome (103 * m) → n ≤ m) :=
  sorry

end smallest_non_six_digit_palindrome_l687_68719


namespace find_number_l687_68736

theorem find_number (x : ℝ) (h : 5 * 1.6 - (2 * 1.4) / x = 4) : x = 0.7 :=
by
  sorry

end find_number_l687_68736


namespace circle_area_l687_68760

theorem circle_area :
  let circle := {p : ℝ × ℝ | (p.fst - 8) ^ 2 + p.snd ^ 2 = 64}
  let line := {p : ℝ × ℝ | p.snd = 10 - p.fst}
  ∃ area : ℝ, 
    (area = 8 * Real.pi) ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle → p.snd ≥ 0 → p ∈ line → p.snd ≥ 10 - p.fst →
  sorry := sorry

end circle_area_l687_68760


namespace multiply_identity_l687_68746

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity_l687_68746


namespace total_length_proof_l687_68739

noncomputable def total_length_climbed (keaton_ladder_height : ℕ) (keaton_times : ℕ) (shortening : ℕ) (reece_times : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - shortening
  let keaton_total := keaton_ladder_height * keaton_times
  let reece_total := reece_ladder_height * reece_times
  (keaton_total + reece_total) * 100

theorem total_length_proof :
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end total_length_proof_l687_68739


namespace angle_B_plus_angle_D_105_l687_68752

theorem angle_B_plus_angle_D_105
(angle_A : ℝ) (angle_AFG angle_AGF : ℝ)
(h1 : angle_A = 30)
(h2 : angle_AFG = angle_AGF)
: angle_B + angle_D = 105 := sorry

end angle_B_plus_angle_D_105_l687_68752


namespace base_conversion_l687_68718

theorem base_conversion {b : ℕ} (h : 5 * 6 + 2 = b * b + b + 1) : b = 5 :=
by
  -- Begin omitted steps to solve the proof
  sorry

end base_conversion_l687_68718


namespace find_WZ_length_l687_68723

noncomputable def WZ_length (XY YZ XZ WX : ℝ) (theta : ℝ) : ℝ :=
  Real.sqrt ((WX^2 + XZ^2 - 2 * WX * XZ * (-1 / 2)))

-- Define the problem within the context of the provided lengths and condition
theorem find_WZ_length :
  WZ_length 3 5 7 8.5 (-1 / 2) = Real.sqrt 180.75 :=
by 
  -- This "by sorry" is used to indicate the proof is omitted
  sorry

end find_WZ_length_l687_68723


namespace system_of_equations_has_two_solutions_l687_68777

theorem system_of_equations_has_two_solutions :
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  xy + yz = 63 ∧ 
  xz + yz = 23 :=
sorry

end system_of_equations_has_two_solutions_l687_68777


namespace Trishul_investment_percentage_l687_68737

-- Definitions from the conditions
def Vishal_invested (T : ℝ) : ℝ := 1.10 * T
def total_investment (T : ℝ) (V : ℝ) : ℝ := T + V + 2000

-- Problem statement
theorem Trishul_investment_percentage (T : ℝ) (V : ℝ) (H1 : V = Vishal_invested T) (H2 : total_investment T V = 5780) :
  ((2000 - T) / 2000) * 100 = 10 :=
sorry

end Trishul_investment_percentage_l687_68737


namespace lunch_people_count_l687_68702

theorem lunch_people_count
  (C : ℝ)   -- total lunch cost including gratuity
  (G : ℝ)   -- gratuity rate
  (P : ℝ)   -- average price per person excluding gratuity
  (n : ℕ)   -- number of people
  (h1 : C = 207.0)  -- condition: total cost with gratuity
  (h2 : G = 0.15)   -- condition: gratuity rate of 15%
  (h3 : P = 12.0)   -- condition: average price per person
  (h4 : C = (1 + G) * n * P) -- condition: total cost with gratuity is (1 + gratuity rate) * number of people * average price per person
  : n = 15 :=       -- conclusion: number of people
sorry

end lunch_people_count_l687_68702


namespace blue_pens_count_l687_68776

variable (redPenCost bluePenCost totalCost totalPens : ℕ)
variable (numRedPens numBluePens : ℕ)

-- Conditions
axiom PriceOfRedPen : redPenCost = 5
axiom PriceOfBluePen : bluePenCost = 7
axiom TotalCost : totalCost = 102
axiom TotalPens : totalPens = 16
axiom PenCount : numRedPens + numBluePens = totalPens
axiom CostEquation : redPenCost * numRedPens + bluePenCost * numBluePens = totalCost

theorem blue_pens_count : numBluePens = 11 :=
by
  sorry

end blue_pens_count_l687_68776


namespace side_length_of_square_l687_68708

theorem side_length_of_square 
  (x : ℝ) 
  (h₁ : 4 * x = 2 * (x * x)) :
  x = 2 :=
by 
  sorry

end side_length_of_square_l687_68708


namespace parabola_b_value_l687_68768

variable {q : ℝ}

theorem parabola_b_value (a b c : ℝ) (h_a : a = -3 / q)
  (h_eq : ∀ x : ℝ, (a * x^2 + b * x + c) = a * (x - q)^2 + q)
  (h_intercept : (a * 0^2 + b * 0 + c) = -2 * q)
  (h_q_nonzero : q ≠ 0) :
  b = 6 / q := 
sorry

end parabola_b_value_l687_68768


namespace slowerPainterDuration_l687_68704

def slowerPainterStartTime : ℝ := 14 -- 2:00 PM in 24-hour format
def fasterPainterStartTime : ℝ := slowerPainterStartTime + 3 -- 3 hours later
def finishTime : ℝ := 24.6 -- 0.6 hours past midnight

theorem slowerPainterDuration :
  finishTime - slowerPainterStartTime = 10.6 :=
by
  sorry

end slowerPainterDuration_l687_68704


namespace polynomial_not_product_of_single_var_l687_68750

theorem polynomial_not_product_of_single_var :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ), 
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = (x^200) * (y^200) + 1) := sorry

end polynomial_not_product_of_single_var_l687_68750


namespace amount_spent_per_trip_l687_68715

def trips_per_month := 4
def months_per_year := 12
def initial_amount := 200
def final_amount := 104

def total_amount_spent := initial_amount - final_amount
def total_trips := trips_per_month * months_per_year

theorem amount_spent_per_trip :
  (total_amount_spent / total_trips) = 2 := 
by 
  sorry

end amount_spent_per_trip_l687_68715


namespace movement_left_3m_l687_68717

-- Define the condition
def movement_right_1m : ℝ := 1

-- Define the theorem stating that movement to the left by 3m should be denoted as -3
theorem movement_left_3m : movement_right_1m * (-3) = -3 :=
by
  sorry

end movement_left_3m_l687_68717


namespace MatthewSharedWithTwoFriends_l687_68773

theorem MatthewSharedWithTwoFriends
  (crackers : ℕ)
  (cakes : ℕ)
  (cakes_per_person : ℕ)
  (persons : ℕ)
  (H1 : crackers = 29)
  (H2 : cakes = 30)
  (H3 : cakes_per_person = 15)
  (H4 : persons * cakes_per_person = cakes) :
  persons = 2 := by
  sorry

end MatthewSharedWithTwoFriends_l687_68773


namespace modulus_of_complex_l687_68764

-- Define the conditions
variables {x y : ℝ}
def i := Complex.I

-- State the conditions of the problem
def condition1 : 1 + x * i = (2 - y) - 3 * i :=
by sorry

-- State the hypothesis and the goal
theorem modulus_of_complex (h : 1 + x * i = (2 - y) - 3 * i) : Complex.abs (x + y * i) = Real.sqrt 10 :=
sorry

end modulus_of_complex_l687_68764


namespace sufficient_but_not_necessary_condition_for_square_l687_68771

theorem sufficient_but_not_necessary_condition_for_square (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ (¬(x^2 > 4 → x > 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_square_l687_68771


namespace total_votes_400_l687_68793

theorem total_votes_400 
    (V : ℝ)
    (h1 : ∃ (c1_votes c2_votes : ℝ), c1_votes = 0.70 * V ∧ c2_votes = 0.30 * V)
    (h2 : ∃ (majority : ℝ), majority = 160)
    (h3 : ∀ (c1_votes c2_votes majority : ℝ), c1_votes - c2_votes = majority) : V = 400 :=
by 
  sorry

end total_votes_400_l687_68793


namespace range_of_a_l687_68745

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) :
    (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
    a ≥ 18 := sorry

end range_of_a_l687_68745


namespace div2_implies_div2_of_either_l687_68748

theorem div2_implies_div2_of_either (a b : ℕ) (h : 2 ∣ a * b) : (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end div2_implies_div2_of_either_l687_68748


namespace proof_completion_l687_68766

namespace MathProof

def p : ℕ := 10 * 7

def r : ℕ := p - 3

def q : ℚ := (3 / 5) * r

theorem proof_completion : q = 40.2 := by
  sorry

end MathProof

end proof_completion_l687_68766


namespace behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l687_68798

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 5 * x ^ 2 + 4

theorem behavior_of_g_as_x_approaches_infinity_and_negative_infinity :
  (∀ ε > 0, ∃ M > 0, ∀ x > M, g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x < -N, g x > ε) :=
by
  sorry

end behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l687_68798


namespace josette_paid_correct_amount_l687_68703

-- Define the number of small and large bottles
def num_small_bottles : ℕ := 3
def num_large_bottles : ℕ := 2

-- Define the cost of each type of bottle
def cost_per_small_bottle : ℝ := 1.50
def cost_per_large_bottle : ℝ := 2.40

-- Define the total number of bottles purchased
def total_bottles : ℕ := num_small_bottles + num_large_bottles

-- Define the discount rate applicable when purchasing 5 or more bottles
def discount_rate : ℝ := 0.10

-- Calculate the initial total cost before any discount
def total_cost_before_discount : ℝ :=
  (num_small_bottles * cost_per_small_bottle) + 
  (num_large_bottles * cost_per_large_bottle)

-- Calculate the discount amount if applicable
def discount_amount : ℝ :=
  if total_bottles >= 5 then
    discount_rate * total_cost_before_discount
  else
    0

-- Calculate the final amount Josette paid after applying any discount
def final_amount_paid : ℝ :=
  total_cost_before_discount - discount_amount

-- Prove that the final amount paid is €8.37
theorem josette_paid_correct_amount :
  final_amount_paid = 8.37 :=
by
  sorry

end josette_paid_correct_amount_l687_68703


namespace radishes_per_row_l687_68733

theorem radishes_per_row 
  (bean_seedlings : ℕ) (beans_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (rows_per_bed : ℕ) (plant_beds : ℕ)
  (h1 : bean_seedlings = 64) (h2 : beans_per_row = 8)
  (h3 : pumpkin_seeds = 84) (h4 : pumpkins_per_row = 7)
  (h5 : radishes = 48) (h6 : rows_per_bed = 2) (h7 : plant_beds = 14) : 
  (radishes / ((plant_beds * rows_per_bed) - (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row))) = 6 := 
by sorry

end radishes_per_row_l687_68733


namespace problem_solution_l687_68761

lemma factor_def (m n : ℕ) : n ∣ m ↔ ∃ k, m = n * k := by sorry

def is_true_A : Prop := 4 ∣ 24
def is_true_B : Prop := 19 ∣ 209 ∧ ¬ (19 ∣ 63)
def is_true_C : Prop := ¬ (30 ∣ 90) ∧ ¬ (30 ∣ 65)
def is_true_D : Prop := 11 ∣ 33 ∧ ¬ (11 ∣ 77)
def is_true_E : Prop := 9 ∣ 180

theorem problem_solution : (is_true_A ∧ is_true_B ∧ is_true_E) ∧ ¬(is_true_C) ∧ ¬(is_true_D) :=
  by sorry

end problem_solution_l687_68761


namespace total_cost_proof_l687_68725

def F : ℝ := 20.50
def R : ℝ := 61.50
def M : ℝ := 1476

def total_cost (mangos : ℝ) (rice : ℝ) (flour : ℝ) : ℝ :=
  (M * mangos) + (R * rice) + (F * flour)

theorem total_cost_proof:
  total_cost 4 3 5 = 6191 := by
  sorry

end total_cost_proof_l687_68725


namespace extreme_points_range_l687_68728

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end extreme_points_range_l687_68728


namespace men_wages_l687_68729

-- Conditions
variable (M W B : ℝ)
variable (h1 : 15 * M = W)
variable (h2 : W = 12 * B)
variable (h3 : 15 * M + W + B = 432)

-- Statement to prove
theorem men_wages : 15 * M = 144 :=
by
  sorry

end men_wages_l687_68729


namespace combined_girls_avg_l687_68724

variables (A a B b : ℕ) -- Number of boys and girls at Adams and Baker respectively.
variables (avgBoysAdams avgGirlsAdams avgAdams avgBoysBaker avgGirlsBaker avgBaker : ℚ)

-- Conditions
def avgAdamsBoys := 72
def avgAdamsGirls := 78
def avgAdamsCombined := 75
def avgBakerBoys := 84
def avgBakerGirls := 91
def avgBakerCombined := 85
def combinedAvgBoys := 80

-- Equations derived from the problem statement
def equations : Prop :=
  (72 * A + 78 * a) / (A + a) = 75 ∧
  (84 * B + 91 * b) / (B + b) = 85 ∧
  (72 * A + 84 * B) / (A + B) = 80

-- The goal is to show the combined average score of girls
def combinedAvgGirls := 85

theorem combined_girls_avg (h : equations A a B b):
  (78 * (6 * b / 7) + 91 * b) / ((6 * b / 7) + b) = 85 := by
  sorry

end combined_girls_avg_l687_68724


namespace original_price_of_shoes_l687_68794

theorem original_price_of_shoes (x : ℝ) (h : 1/4 * x = 18) : x = 72 := by
  sorry

end original_price_of_shoes_l687_68794


namespace subset_implies_value_l687_68775

theorem subset_implies_value (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 3, 2*m-1}) (hB : B = {3, m}) (hSub : B ⊆ A) : 
  m = -1 ∨ m = 1 := by
  sorry

end subset_implies_value_l687_68775


namespace fruit_fly_cell_division_l687_68734

/-- Genetic properties of fruit flies:
  1. Fruit flies have 2N = 8 chromosomes.
  2. Alleles A/a and B/b are inherited independently.
  3. Genotype AaBb is given.
  4. This genotype undergoes cell division without chromosomal variation.

Prove that:
Cells with a genetic composition of AAaaBBbb contain 8 or 16 chromosomes.
-/
theorem fruit_fly_cell_division (genotype : ℕ → ℕ) (A a B b : ℕ) :
  genotype 2 = 8 ∧
  (A + a + B + b = 8) ∧
  (genotype 0 = 2 * 4) →
  (genotype 1 = 8 ∨ genotype 1 = 16) :=
by
  sorry

end fruit_fly_cell_division_l687_68734


namespace CarmenBrushLengthIsCorrect_l687_68784

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l687_68784


namespace roof_shingles_area_l687_68751

-- Definitions based on given conditions
def base_main_roof : ℝ := 20.5
def height_main_roof : ℝ := 25
def upper_base_porch : ℝ := 2.5
def lower_base_porch : ℝ := 4.5
def height_porch : ℝ := 3
def num_gables_main_roof : ℕ := 2
def num_trapezoids_porch : ℕ := 4

-- Proof problem statement
theorem roof_shingles_area : 
  2 * (1 / 2 * base_main_roof * height_main_roof) +
  4 * (1 / 2 * (upper_base_porch + lower_base_porch) * height_porch) = 554.5 :=
by sorry

end roof_shingles_area_l687_68751


namespace sum_of_numbers_is_twenty_l687_68716

-- Given conditions
variables {a b c : ℝ}

-- Prove that the sum of a, b, and c is 20 given the conditions
theorem sum_of_numbers_is_twenty (h1 : a^2 + b^2 + c^2 = 138) (h2 : ab + bc + ca = 131) :
  a + b + c = 20 :=
by
  sorry

end sum_of_numbers_is_twenty_l687_68716


namespace triangle_area_16_l687_68792

theorem triangle_area_16 : 
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 8)
  let base := (B.1 - A.1)
  let height := (C.2 - A.2)
  (base * height) / 2 = 16 := by
  sorry

end triangle_area_16_l687_68792


namespace calculate_expression_l687_68720

theorem calculate_expression : 1 + (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 2 - Real.sqrt 3) = 1 :=
by
  sorry

end calculate_expression_l687_68720


namespace lineup_count_l687_68796

-- Define five distinct people
inductive Person 
| youngest : Person 
| oldest : Person 
| person1 : Person 
| person2 : Person 
| person3 : Person 

-- Define the total number of people
def numberOfPeople : ℕ := 5

-- Define a function to calculate the number of ways to line up five people with constraints
def lineupWays : ℕ := 3 * 4 * 3 * 2 * 1

-- State the theorem
theorem lineup_count (h₁ : numberOfPeople = 5) (h₂ : ¬ ∃ (p : Person), p = Person.youngest ∨ p = Person.oldest → p = Person.youngest) :
  lineupWays = 72 :=
by
  sorry

end lineup_count_l687_68796


namespace mary_shirts_left_l687_68701

theorem mary_shirts_left :
  let blue_shirts := 35
  let brown_shirts := 48
  let red_shirts := 27
  let yellow_shirts := 36
  let green_shirts := 18
  let blue_given_away := 4 / 5 * blue_shirts
  let brown_given_away := 5 / 6 * brown_shirts
  let red_given_away := 2 / 3 * red_shirts
  let yellow_given_away := 3 / 4 * yellow_shirts
  let green_given_away := 1 / 3 * green_shirts
  let blue_left := blue_shirts - blue_given_away
  let brown_left := brown_shirts - brown_given_away
  let red_left := red_shirts - red_given_away
  let yellow_left := yellow_shirts - yellow_given_away
  let green_left := green_shirts - green_given_away
  blue_left + brown_left + red_left + yellow_left + green_left = 45 := by
  sorry

end mary_shirts_left_l687_68701


namespace algebraic_identity_neg_exponents_l687_68782

theorem algebraic_identity_neg_exponents (x y z : ℂ) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end algebraic_identity_neg_exponents_l687_68782


namespace men_in_first_group_l687_68772

theorem men_in_first_group (M : ℕ) 
  (h1 : (M * 25 : ℝ) = (15 * 26.666666666666668 : ℝ)) : 
  M = 16 := 
by 
  sorry

end men_in_first_group_l687_68772


namespace problem_statement_l687_68741

theorem problem_statement (a b : ℝ) (h1 : 1 / a + 1 / b = Real.sqrt 5) (h2 : a ≠ b) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 :=
by
  sorry

end problem_statement_l687_68741


namespace consecutive_odd_product_l687_68731

theorem consecutive_odd_product (n : ℤ) :
  (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by sorry

end consecutive_odd_product_l687_68731


namespace one_third_times_seven_times_nine_l687_68713

theorem one_third_times_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_times_seven_times_nine_l687_68713


namespace L_shape_perimeter_correct_l687_68711

-- Define the dimensions of the rectangles
def rect_height : ℕ := 3
def rect_width : ℕ := 4

-- Define the combined shape and perimeter calculation
def L_shape_perimeter (h w : ℕ) : ℕ := (2 * w) + (2 * h)

theorem L_shape_perimeter_correct : 
  L_shape_perimeter rect_height rect_width = 14 := 
  sorry

end L_shape_perimeter_correct_l687_68711


namespace hannah_books_per_stocking_l687_68749

theorem hannah_books_per_stocking
  (candy_canes_per_stocking : ℕ)
  (beanie_babies_per_stocking : ℕ)
  (num_kids : ℕ)
  (total_stuffers : ℕ)
  (books_per_stocking : ℕ) :
  candy_canes_per_stocking = 4 →
  beanie_babies_per_stocking = 2 →
  num_kids = 3 →
  total_stuffers = 21 →
  books_per_stocking = (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids →
  books_per_stocking = 1 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  simp at h5
  sorry

end hannah_books_per_stocking_l687_68749


namespace probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l687_68786

noncomputable def diameter := 19 -- mm
noncomputable def side_length := 50 -- mm, side length of each square
noncomputable def total_area := side_length^2 -- 2500 mm^2 for each square
noncomputable def coin_radius := diameter / 2 -- 9.5 mm

theorem probability_completely_inside_square : 
  (side_length - 2 * coin_radius)^2 / total_area = 961 / 2500 :=
by sorry

theorem probability_partial_one_edge :
  4 * ((side_length - 2 * coin_radius) * coin_radius) / total_area = 1178 / 2500 :=
by sorry

theorem probability_partial_two_edges_not_vertex :
  (4 * ((diameter)^2 - (coin_radius^2 * Real.pi / 4))) / total_area = (4 * 290.12) / 2500 :=
by sorry

theorem probability_vertex :
  4 * (coin_radius^2 * Real.pi / 4) / total_area = 4 * 70.88 / 2500 :=
by sorry

end probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l687_68786


namespace clock_angle_150_at_5pm_l687_68722

theorem clock_angle_150_at_5pm :
  (∀ t : ℕ, (t = 5) ↔ (∀ θ : ℝ, θ = 150 → θ = (30 * t))) := sorry

end clock_angle_150_at_5pm_l687_68722


namespace x_finishes_in_nine_days_l687_68774

-- Definitions based on the conditions
def x_work_rate : ℚ := 1 / 24
def y_work_rate : ℚ := 1 / 16
def y_days_worked : ℚ := 10
def y_work_done : ℚ := y_work_rate * y_days_worked
def remaining_work : ℚ := 1 - y_work_done
def x_days_to_finish : ℚ := remaining_work / x_work_rate

-- Statement to be proven
theorem x_finishes_in_nine_days : x_days_to_finish = 9 := 
by
  -- Skipping actual proof steps as instructed
  sorry

end x_finishes_in_nine_days_l687_68774


namespace number_of_games_between_men_and_women_l687_68755

theorem number_of_games_between_men_and_women
    (W M : ℕ)
    (hW : W * (W - 1) / 2 = 72)
    (hM : M * (M - 1) / 2 = 288) :
  M * W = 288 :=
by
  sorry

end number_of_games_between_men_and_women_l687_68755


namespace cans_collected_on_first_day_l687_68753

-- Declare the main theorem
theorem cans_collected_on_first_day 
  (x : ℕ) -- Number of cans collected on the first day
  (total_cans : x + (x + 5) + (x + 10) + (x + 15) + (x + 20) = 150) :
  x = 20 :=
sorry

end cans_collected_on_first_day_l687_68753


namespace num_packages_l687_68757

-- Defining the given conditions
def packages_count_per_package := 6
def total_tshirts := 426

-- The statement to be proved
theorem num_packages : (total_tshirts / packages_count_per_package) = 71 :=
by sorry

end num_packages_l687_68757


namespace total_clients_correct_l687_68790

-- Define the number of each type of cars and total cars
def num_cars : ℕ := 12
def num_sedans : ℕ := 4
def num_coupes : ℕ := 4
def num_suvs : ℕ := 4

-- Define the number of selections per car and total selections required
def selections_per_car : ℕ := 3

-- Define the number of clients per type of car
def num_clients_who_like_sedans : ℕ := (num_sedans * selections_per_car) / 2
def num_clients_who_like_coupes : ℕ := (num_coupes * selections_per_car) / 2
def num_clients_who_like_suvs : ℕ := (num_suvs * selections_per_car) / 2

-- Compute total number of clients
def total_clients : ℕ := num_clients_who_like_sedans + num_clients_who_like_coupes + num_clients_who_like_suvs

-- Prove that the total number of clients is 18
theorem total_clients_correct : total_clients = 18 := by
  sorry

end total_clients_correct_l687_68790


namespace solve_for_k_l687_68763

theorem solve_for_k (t s k : ℝ) :
  (∀ t s : ℝ, (∃ t s : ℝ, (⟨1, 4⟩ : ℝ × ℝ) + t • ⟨5, -3⟩ = ⟨0, 1⟩ + s • ⟨-2, k⟩) → false) ↔ k = 6 / 5 :=
by
  sorry

end solve_for_k_l687_68763


namespace number_of_positive_divisors_of_60_l687_68780

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l687_68780


namespace total_pieces_in_10_row_triangle_l687_68735

open Nat

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_in_10_row_triangle : 
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  unit_rods + connectors = 231 :=
by
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  show unit_rods + connectors = 231
  sorry

end total_pieces_in_10_row_triangle_l687_68735


namespace solving_equation_l687_68740

theorem solving_equation (x : ℝ) : 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := 
by
  sorry

end solving_equation_l687_68740


namespace positive_difference_of_numbers_l687_68789

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l687_68789


namespace hannah_strawberries_l687_68787

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end hannah_strawberries_l687_68787


namespace wire_length_before_cutting_l687_68700

theorem wire_length_before_cutting (L S : ℝ) (h1 : S = 40) (h2 : S = 2 / 5 * L) : L + S = 140 :=
by
  sorry

end wire_length_before_cutting_l687_68700


namespace part1_part2_l687_68705

-- Define the function f
def f (x m : ℝ) : ℝ := abs (x + m) + abs (2 * x - 1)

-- First part of the problem
theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 := 
by sorry

-- Second part of the problem
theorem part2 (m : ℝ) : 
  (∀ x, 3 / 4 ≤ x → x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 := 
by sorry

end part1_part2_l687_68705


namespace perfect_square_trinomial_k_l687_68744

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m + 1)^2) ∨
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m - 1)^2) ↔
  k = 14 ∨ k = -14 :=
sorry

end perfect_square_trinomial_k_l687_68744


namespace simplify_trig_expression_l687_68707

open Real

theorem simplify_trig_expression (α : ℝ) : 
  sin (2 * π - α)^2 + (cos (π + α) * cos (π - α)) + 1 = 2 := 
by 
  sorry

end simplify_trig_expression_l687_68707


namespace solution_l687_68765

-- Define the equations and their solution sets
def eq1 (x p : ℝ) : Prop := x^2 - p * x + 6 = 0
def eq2 (x q : ℝ) : Prop := x^2 + 6 * x - q = 0

-- Define the condition that the solution sets intersect at {2}
def intersect_at_2 (p q : ℝ) : Prop :=
  eq1 2 p ∧ eq2 2 q

-- The main theorem stating the value of p + q given the conditions
theorem solution (p q : ℝ) (h : intersect_at_2 p q) : p + q = 21 :=
by
  sorry

end solution_l687_68765


namespace volume_of_rectangular_prism_l687_68779

theorem volume_of_rectangular_prism
  (l w h : ℝ)
  (Hlw : l * w = 10)
  (Hwh : w * h = 15)
  (Hlh : l * h = 6) : l * w * h = 30 := 
by
  sorry

end volume_of_rectangular_prism_l687_68779


namespace range_of_quadratic_function_l687_68783

variable (x : ℝ)
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem range_of_quadratic_function :
  (∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = quadratic_function x) ↔ (1 ≤ y ∧ y ≤ 5)) :=
by
  sorry

end range_of_quadratic_function_l687_68783


namespace gcd_m_n_is_one_l687_68714

-- Definitions of m and n
def m : ℕ := 101^2 + 203^2 + 307^2
def n : ℕ := 100^2 + 202^2 + 308^2

-- The main theorem stating the gcd of m and n
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l687_68714


namespace fg_of_2_eq_513_l687_68754

def f (x : ℤ) : ℤ := x^3 + 1
def g (x : ℤ) : ℤ := 3*x + 2

theorem fg_of_2_eq_513 : f (g 2) = 513 := by
  sorry

end fg_of_2_eq_513_l687_68754


namespace range_of_m_l687_68795

noncomputable def is_quadratic (m : ℝ) : Prop := (m^2 - 4) ≠ 0

theorem range_of_m (m : ℝ) : is_quadratic m → m ≠ 2 ∧ m ≠ -2 :=
by sorry

end range_of_m_l687_68795


namespace complex_num_z_imaginary_square_l687_68799

theorem complex_num_z_imaginary_square (z : ℂ) (h1 : z.im ≠ 0) (h2 : z.re = 0) (h3 : ((z + 1) ^ 2).re = 0) :
  z = Complex.I ∨ z = -Complex.I :=
by
  sorry

end complex_num_z_imaginary_square_l687_68799


namespace product_of_two_numbers_ratio_l687_68778

theorem product_of_two_numbers_ratio (x y : ℝ)
  (h1 : x - y ≠ 0)
  (h2 : x + y = 4 * (x - y))
  (h3 : x * y = 18 * (x - y)) :
  x * y = 86.4 :=
by
  sorry

end product_of_two_numbers_ratio_l687_68778


namespace pyramid_side_length_l687_68788

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l687_68788


namespace graph_quadrant_l687_68770

theorem graph_quadrant (x y : ℝ) : 
  y = 3 * x - 4 → ¬ ((x < 0) ∧ (y > 0)) :=
by
  intro h
  sorry

end graph_quadrant_l687_68770


namespace valid_m_values_l687_68767

theorem valid_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) → m < 1 :=
by
  sorry

end valid_m_values_l687_68767


namespace sum_of_squares_l687_68727

theorem sum_of_squares (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 1) (h2 : b^2 + b * c + c^2 = 3) (h3 : c^2 + c * a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := 
sorry

end sum_of_squares_l687_68727


namespace pos_sum_inequality_l687_68738

theorem pos_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := 
sorry

end pos_sum_inequality_l687_68738


namespace initial_typists_count_l687_68797

theorem initial_typists_count 
  (typists_rate : ℕ → ℕ)
  (letters_in_20min : ℕ)
  (total_typists : ℕ)
  (letters_in_1hour : ℕ)
  (initial_typists : ℕ) 
  (h1 : letters_in_20min = 38)
  (h2 : letters_in_1hour = 171)
  (h3 : total_typists = 30)
  (h4 : ∀ t, 3 * (typists_rate t) = letters_in_1hour / total_typists)
  (h5 : ∀ t, typists_rate t = letters_in_20min / t) 
  : initial_typists = 20 := 
sorry

end initial_typists_count_l687_68797


namespace price_rollback_is_correct_l687_68726

-- Define the conditions
def liters_today : ℕ := 10
def cost_per_liter_today : ℝ := 1.4
def liters_friday : ℕ := 25
def total_liters : ℕ := 35
def total_cost : ℝ := 39

-- Define the price rollback calculation
noncomputable def price_rollback : ℝ :=
  (cost_per_liter_today - (total_cost - (liters_today * cost_per_liter_today)) / liters_friday)

-- The theorem stating the rollback per liter is $0.4
theorem price_rollback_is_correct : price_rollback = 0.4 := by
  sorry

end price_rollback_is_correct_l687_68726


namespace system_solution_xz_y2_l687_68769

theorem system_solution_xz_y2 (x y z : ℝ) (k : ℝ)
  (h : (x + 2 * k * y + 4 * z = 0) ∧
       (4 * x + k * y - 3 * z = 0) ∧
       (3 * x + 5 * y - 2 * z = 0) ∧
       x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k = 95 / 12) :
  (x * z) / (y ^ 2) = 10 :=
by sorry

end system_solution_xz_y2_l687_68769


namespace range_of_a_l687_68759

open Set

theorem range_of_a (a x : ℝ) (p : ℝ → Prop) (q : ℝ → ℝ → Prop)
    (hp : p x → |x - a| > 3)
    (hq : q x a → (x + 1) * (2 * x - 1) ≥ 0)
    (hsuff : ∀ x, ¬p x → q x a) :
    {a | ∀ x, (¬ (|x - a| > 3) → (x + 1) * (2 * x - 1) ≥ 0) → (( a ≤ -4) ∨ (a ≥ 7 / 2))} :=
by
  sorry

end range_of_a_l687_68759


namespace similar_triangles_perimeter_l687_68785

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end similar_triangles_perimeter_l687_68785


namespace sine_sum_square_greater_l687_68743

variable {α β : Real} (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1)

theorem sine_sum_square_greater (α β : Real) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
  (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : 
  Real.sin (α + β) ^ 2 > Real.sin α ^ 2 + Real.sin β ^ 2 :=
sorry

end sine_sum_square_greater_l687_68743


namespace max_value_fn_l687_68710

theorem max_value_fn : ∀ x : ℝ, y = 1 / (|x| + 2) → 
  ∃ y : ℝ, y = 1 / 2 ∧ ∀ x : ℝ, 1 / (|x| + 2) ≤ y :=
sorry

end max_value_fn_l687_68710


namespace total_games_High_School_Nine_l687_68732

-- Define the constants and assumptions.
def num_teams := 9
def games_against_non_league := 6

-- Calculation of the number of games within the league.
def games_within_league := (num_teams * (num_teams - 1) / 2) * 2

-- Calculation of the number of games against non-league teams.
def games_non_league := num_teams * games_against_non_league

-- The total number of games.
def total_games := games_within_league + games_non_league

-- The statement to prove.
theorem total_games_High_School_Nine : total_games = 126 := 
by
  -- You do not need to provide the proof.
  sorry

end total_games_High_School_Nine_l687_68732


namespace longer_diagonal_rhombus_l687_68762

theorem longer_diagonal_rhombus (a b d1 d2 : ℝ) 
  (h1 : a = 35) 
  (h2 : d1 = 42) : 
  d2 = 56 := 
by 
  sorry

end longer_diagonal_rhombus_l687_68762


namespace cyclic_trapezoid_radii_relation_l687_68756

variables (A B C D O : Type)
variables (AD BC : Type)
variables (r1 r2 r3 r4 : ℝ)

-- Conditions
def cyclic_trapezoid (A B C D: Type) (AD BC: Type): Prop := sorry
def intersection (A B C D O : Type): Prop := sorry
def radius_incircle (triangle : Type) (radius : ℝ): Prop := sorry

theorem cyclic_trapezoid_radii_relation
  (h1: cyclic_trapezoid A B C D AD BC)
  (h2: intersection A B C D O)
  (hr1: radius_incircle AOD r1)
  (hr2: radius_incircle AOB r2)
  (hr3: radius_incircle BOC r3)
  (hr4: radius_incircle COD r4):
  (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4) :=
sorry

end cyclic_trapezoid_radii_relation_l687_68756


namespace ratio_of_means_l687_68747

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end ratio_of_means_l687_68747


namespace malesWithCollegeDegreesOnly_l687_68706

-- Define the parameters given in the problem
def totalEmployees : ℕ := 180
def totalFemales : ℕ := 110
def employeesWithAdvancedDegrees : ℕ := 90
def employeesWithCollegeDegreesOnly : ℕ := totalEmployees - employeesWithAdvancedDegrees
def femalesWithAdvancedDegrees : ℕ := 55

-- Define the question as a theorem
theorem malesWithCollegeDegreesOnly : 
  totalEmployees = 180 →
  totalFemales = 110 →
  employeesWithAdvancedDegrees = 90 →
  employeesWithCollegeDegreesOnly = 90 →
  femalesWithAdvancedDegrees = 55 →
  ∃ (malesWithCollegeDegreesOnly : ℕ), 
    malesWithCollegeDegreesOnly = 35 := 
by
  intros
  sorry

end malesWithCollegeDegreesOnly_l687_68706


namespace probability_neither_orange_nor_white_l687_68721

/-- Define the problem conditions. -/
def num_orange_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 6

/-- Define the total number of balls. -/
def total_balls : ℕ := num_orange_balls + num_black_balls + num_white_balls

/-- Define the probability of picking a black ball (neither orange nor white). -/
noncomputable def probability_black_ball : ℚ := num_black_balls / total_balls

/-- The main statement to be proved: The probability is 1/3. -/
theorem probability_neither_orange_nor_white : probability_black_ball = 1 / 3 :=
sorry

end probability_neither_orange_nor_white_l687_68721


namespace total_pounds_of_peppers_l687_68791

-- Definitions and conditions
def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335

-- Theorem statement
theorem total_pounds_of_peppers : green_peppers + red_peppers = 5.666666666666667 :=
by
  sorry

end total_pounds_of_peppers_l687_68791


namespace tangent_line_at_zero_decreasing_intervals_l687_68742

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem tangent_line_at_zero :
  let t : ℝ × ℝ := (0, f 0)
  (∀ x : ℝ, (9 * x - f x - 2 = 0) → t.snd = -2) := by
  sorry

theorem decreasing_intervals :
  ∀ x : ℝ, (-3 * x^2 + 6 * x + 9 < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end tangent_line_at_zero_decreasing_intervals_l687_68742


namespace density_of_second_part_l687_68781

theorem density_of_second_part (V m : ℝ) (h1 : ∀ V m : ℝ, V_1 = 0.3 * V) 
  (h2 : ∀ V m : ℝ, m_1 = 0.6 * m) 
  (rho1 : ρ₁ = 7800) : 
  ∃ ρ₂, ρ₂ = 2229 :=
by sorry

end density_of_second_part_l687_68781


namespace smallest_positive_period_and_monotonic_increase_max_min_in_interval_l687_68709

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∃ a b : ℝ, (k * π - π / 3 ≤ a ∧ a ≤ x) ∧ (x ≤ b ∧ b ≤ k * π + π / 6) → f x = 1) := sorry

theorem max_min_in_interval :
  (∀ x : ℝ, (-π / 4 ≤ x ∧ x ≤ π / 6) → (1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3)) := sorry

end smallest_positive_period_and_monotonic_increase_max_min_in_interval_l687_68709


namespace commission_percentage_l687_68730

-- Define the conditions
def cost_of_item := 18.0
def observed_price := 27.0
def profit_percentage := 0.20
def desired_selling_price := cost_of_item + profit_percentage * cost_of_item
def commission_amount := observed_price - desired_selling_price

-- Prove the commission percentage taken by the online store
theorem commission_percentage : (commission_amount / desired_selling_price) * 100 = 25 :=
by
  -- Here the proof would normally be implemented
  sorry

end commission_percentage_l687_68730


namespace troll_ratio_l687_68758

theorem troll_ratio 
  (B : ℕ)
  (h1 : 6 + B + (1 / 2 : ℚ) * B = 33) : 
  B / 6 = 3 :=
by
  sorry

end troll_ratio_l687_68758
