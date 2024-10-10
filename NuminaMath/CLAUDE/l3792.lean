import Mathlib

namespace equation_solution_l3792_379268

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end equation_solution_l3792_379268


namespace count_true_propositions_l3792_379253

theorem count_true_propositions : 
  (∀ (x : ℝ), x^2 + 1 > 0) ∧ 
  (∃ (x : ℤ), x^3 < 1) ∧ 
  (∀ (x : ℚ), x^2 ≠ 2) ∧ 
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

#check count_true_propositions

end count_true_propositions_l3792_379253


namespace range_of_a_l3792_379225

def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

theorem range_of_a (a : ℝ) : (M a ⊆ A) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end range_of_a_l3792_379225


namespace algebraic_expression_equality_l3792_379202

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = -2) : 
  9 - 2*x + 4*y = 13 := by
  sorry

end algebraic_expression_equality_l3792_379202


namespace total_revenue_calculation_l3792_379251

def ticket_revenue (student_price adult_price child_price senior_price : ℚ)
                   (group_discount : ℚ)
                   (separate_students separate_adults separate_children separate_seniors : ℕ)
                   (group_students group_adults group_children group_seniors : ℕ) : ℚ :=
  let separate_revenue := student_price * separate_students +
                          adult_price * separate_adults +
                          child_price * separate_children +
                          senior_price * separate_seniors
  let group_subtotal := student_price * group_students +
                        adult_price * group_adults +
                        child_price * group_children +
                        senior_price * group_seniors
  let group_size := group_students + group_adults + group_children + group_seniors
  let group_revenue := if group_size > 10 then group_subtotal * (1 - group_discount) else group_subtotal
  separate_revenue + group_revenue

theorem total_revenue_calculation :
  ticket_revenue 6 8 4 7 (1/10)
                 20 12 15 10
                 5 8 10 9 = 523.3 := by
  sorry

end total_revenue_calculation_l3792_379251


namespace smallest_digit_for_divisibility_l3792_379289

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_with_d (d : ℕ) : ℕ := 437000 + d * 1000 + 3

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧ 
    (∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (number_with_d d')) ∧
    d = 1 := by
  sorry

end smallest_digit_for_divisibility_l3792_379289


namespace relay_team_selection_l3792_379266

/-- The number of ways to select and arrange 4 sprinters out of 6 for a 4×100m relay, 
    given that one sprinter cannot run the first leg and another cannot run the fourth leg. -/
theorem relay_team_selection (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) -     -- Total arrangements without restrictions
  2 * ((n - 1).factorial / (n - k).factorial) +  -- Subtracting arrangements with A or B in wrong position
  ((n - 2).factorial / (n - k).factorial) -- Adding back arrangements with both A and B in wrong positions
  = 252 := by sorry

end relay_team_selection_l3792_379266


namespace lcm_of_ratio_and_hcf_l3792_379271

/-- Given two positive integers with a ratio of 4:5 and HCF of 4, their LCM is 80 -/
theorem lcm_of_ratio_and_hcf (a b : ℕ+) (h_ratio : a.val * 5 = b.val * 4) 
  (h_hcf : Nat.gcd a.val b.val = 4) : Nat.lcm a.val b.val = 80 := by
  sorry

end lcm_of_ratio_and_hcf_l3792_379271


namespace square_perimeter_l3792_379291

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 520 → perimeter = 8 * Real.sqrt 65 := by
  sorry

end square_perimeter_l3792_379291


namespace chocolate_milk_probability_l3792_379293

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem chocolate_milk_probability :
  binomial_probability 7 3 (2/3) = 280/2187 := by
  sorry

end chocolate_milk_probability_l3792_379293


namespace quadratic_equation_solutions_l3792_379224

theorem quadratic_equation_solutions (a b m : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧ ∀ x : ℝ, a * (x + m)^2 + b = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 0 ∧ ∀ x : ℝ, a * (x - m + 2)^2 + b = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end quadratic_equation_solutions_l3792_379224


namespace boat_travel_difference_l3792_379237

/-- Represents the difference in distance traveled downstream vs upstream for a boat -/
def boat_distance_difference (a b : ℝ) : ℝ :=
  let downstream_speed := a + b
  let upstream_speed := a - b
  let downstream_distance := 3 * downstream_speed
  let upstream_distance := 2 * upstream_speed
  downstream_distance - upstream_distance

/-- Theorem stating the difference in distance traveled by the boat -/
theorem boat_travel_difference (a b : ℝ) (h : a > b) :
  boat_distance_difference a b = a + 5*b := by
  sorry

#check boat_travel_difference

end boat_travel_difference_l3792_379237


namespace equal_savings_after_25_weeks_l3792_379287

/-- Proves that saving 7 dollars per week results in equal savings after 25 weeks --/
theorem equal_savings_after_25_weeks 
  (your_initial_savings : ℕ := 160)
  (friend_initial_savings : ℕ := 210)
  (friend_weekly_savings : ℕ := 5)
  (weeks : ℕ := 25)
  (your_weekly_savings : ℕ := 7) : 
  your_initial_savings + weeks * your_weekly_savings = 
  friend_initial_savings + weeks * friend_weekly_savings := by
sorry

end equal_savings_after_25_weeks_l3792_379287


namespace sum_proper_divisors_540_l3792_379203

theorem sum_proper_divisors_540 : 
  (Finset.filter (λ x => x < 540 ∧ 540 % x = 0) (Finset.range 540)).sum id = 1140 := by
  sorry

end sum_proper_divisors_540_l3792_379203


namespace probability_theorem_l3792_379260

def num_male_students : ℕ := 5
def num_female_students : ℕ := 2
def num_representatives : ℕ := 3

def probability_B_or_C_given_A (total_students : ℕ) (remaining_selections : ℕ) : ℚ :=
  let favorable_outcomes := (remaining_selections * (total_students - 3)) + 1
  let total_outcomes := Nat.choose (total_students - 1) remaining_selections
  favorable_outcomes / total_outcomes

theorem probability_theorem :
  probability_B_or_C_given_A (num_male_students + num_female_students) (num_representatives - 1) = 3/5 :=
by sorry

end probability_theorem_l3792_379260


namespace colton_sticker_distribution_l3792_379284

/-- Proves that Colton gave 4 stickers to each of his 3 friends --/
theorem colton_sticker_distribution :
  ∀ (initial_stickers : ℕ) 
    (friends : ℕ) 
    (remaining_stickers : ℕ) 
    (stickers_to_friend : ℕ),
  initial_stickers = 72 →
  friends = 3 →
  remaining_stickers = 42 →
  initial_stickers = 
    remaining_stickers + 
    (friends * stickers_to_friend) + 
    (friends * stickers_to_friend + 2) + 
    (friends * stickers_to_friend - 8) →
  stickers_to_friend = 4 := by
sorry

end colton_sticker_distribution_l3792_379284


namespace water_formation_l3792_379286

-- Define the molecules and their quantities
def HCl_moles : ℕ := 1
def NaHCO3_moles : ℕ := 1

-- Define the reaction equation
def reaction_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the function to calculate water moles produced
def water_moles_produced (hcl : ℕ) (nahco3 : ℕ) : ℕ :=
  min hcl nahco3

-- Theorem statement
theorem water_formation :
  water_moles_produced HCl_moles NaHCO3_moles = 1 :=
by
  sorry

end water_formation_l3792_379286


namespace jimmy_pens_purchase_l3792_379208

theorem jimmy_pens_purchase (pen_cost : ℕ) (notebook_cost : ℕ) (folder_cost : ℕ)
  (notebooks_bought : ℕ) (folders_bought : ℕ) (paid : ℕ) (change : ℕ) :
  pen_cost = 1 →
  notebook_cost = 3 →
  folder_cost = 5 →
  notebooks_bought = 4 →
  folders_bought = 2 →
  paid = 50 →
  change = 25 →
  (paid - change - (notebooks_bought * notebook_cost + folders_bought * folder_cost)) / pen_cost = 3 :=
by sorry

end jimmy_pens_purchase_l3792_379208


namespace number_difference_l3792_379250

theorem number_difference (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 := by
  sorry

end number_difference_l3792_379250


namespace additional_black_balls_probability_l3792_379295

/-- Represents the contents of a bag of colored balls -/
structure BagContents where
  white : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the probability of drawing a black ball from the bag -/
def probBlack (bag : BagContents) : ℚ :=
  bag.black / (bag.white + bag.black + bag.red)

/-- The initial contents of the bag -/
def initialBag : BagContents :=
  { white := 2, black := 3, red := 5 }

/-- The number of additional black balls added -/
def additionalBlackBalls : ℕ := 18

/-- The final contents of the bag after adding black balls -/
def finalBag : BagContents :=
  { white := initialBag.white,
    black := initialBag.black + additionalBlackBalls,
    red := initialBag.red }

theorem additional_black_balls_probability :
  probBlack finalBag = 3/4 := by sorry

end additional_black_balls_probability_l3792_379295


namespace polynomial_form_theorem_l3792_379269

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial P satisfies for all real a, b, c -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 → 
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form_theorem (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry


end polynomial_form_theorem_l3792_379269


namespace raise_doubles_earnings_l3792_379239

/-- Calculates the new weekly earnings after a percentage raise -/
def new_earnings (initial_earnings : ℕ) (percentage_raise : ℕ) : ℕ :=
  initial_earnings + initial_earnings * percentage_raise / 100

/-- Proves that a 100% raise on $40 results in $80 weekly earnings -/
theorem raise_doubles_earnings : new_earnings 40 100 = 80 := by
  sorry

end raise_doubles_earnings_l3792_379239


namespace root_product_l3792_379235

theorem root_product (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^7 - b*x - c = 0) → 
  b * c = 11830 := by
sorry

end root_product_l3792_379235


namespace quotient_of_A_and_B_l3792_379246

/-- Given A and B as defined, prove that A / B = 31 -/
theorem quotient_of_A_and_B : 
  let A := 8 * 10 + 13 * 1
  let B := 30 - 9 - 9 - 9
  A / B = 31 := by
sorry

end quotient_of_A_and_B_l3792_379246


namespace ice_cream_cost_theorem_l3792_379236

def calculate_ice_cream_cost (chapati_count : ℕ) (chapati_price : ℚ)
                             (rice_count : ℕ) (rice_price : ℚ)
                             (veg_count : ℕ) (veg_price : ℚ)
                             (soup_count : ℕ) (soup_price : ℚ)
                             (dessert_count : ℕ) (dessert_price : ℚ)
                             (drink_count : ℕ) (drink_price : ℚ)
                             (discount_rate : ℚ) (tax_rate : ℚ)
                             (ice_cream_count : ℕ) (total_paid : ℚ) : ℚ :=
  let chapati_total := chapati_count * chapati_price
  let rice_total := rice_count * rice_price
  let veg_total := veg_count * veg_price
  let soup_total := soup_count * soup_price
  let dessert_total := dessert_count * dessert_price
  let drink_total := drink_count * drink_price * (1 - discount_rate)
  let subtotal := chapati_total + rice_total + veg_total + soup_total + dessert_total + drink_total
  let total_with_tax := subtotal * (1 + tax_rate)
  let ice_cream_total := total_paid - total_with_tax
  ice_cream_total / ice_cream_count

theorem ice_cream_cost_theorem :
  let chapati_count : ℕ := 16
  let chapati_price : ℚ := 6
  let rice_count : ℕ := 5
  let rice_price : ℚ := 45
  let veg_count : ℕ := 7
  let veg_price : ℚ := 70
  let soup_count : ℕ := 4
  let soup_price : ℚ := 30
  let dessert_count : ℕ := 3
  let dessert_price : ℚ := 85
  let drink_count : ℕ := 2
  let drink_price : ℚ := 50
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.18
  let ice_cream_count : ℕ := 6
  let total_paid : ℚ := 2159
  abs (calculate_ice_cream_cost chapati_count chapati_price rice_count rice_price
                                veg_count veg_price soup_count soup_price
                                dessert_count dessert_price drink_count drink_price
                                discount_rate tax_rate ice_cream_count total_paid - 108.89) < 0.01 := by
  sorry

end ice_cream_cost_theorem_l3792_379236


namespace function_defined_range_l3792_379264

open Real

theorem function_defined_range (a : ℝ) :
  (∀ x ∈ Set.Iic 1, (1 + 2^x + 4^x * a) / 3 > 0) ↔ a > -3/4 := by
  sorry

end function_defined_range_l3792_379264


namespace gauss_family_mean_age_is_correct_l3792_379263

/-- The mean age of the Gauss family children -/
def gauss_family_mean_age : ℚ :=
  let ages : List ℕ := [7, 7, 8, 14, 12, 15, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating that the mean age of the Gauss family children is 79/7 -/
theorem gauss_family_mean_age_is_correct : gauss_family_mean_age = 79 / 7 := by
  sorry

end gauss_family_mean_age_is_correct_l3792_379263


namespace absolute_value_inequality_l3792_379227

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by
sorry

end absolute_value_inequality_l3792_379227


namespace tan_seven_pi_sixths_l3792_379222

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_seven_pi_sixths_l3792_379222


namespace max_sum_of_distances_l3792_379254

/-- Triangle ABC is a right triangle with ∠ABC = 90°, AC = 10, AB = 8, BC = 6 -/
structure RightTriangle where
  AC : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : AC^2 = AB^2 + BC^2
  AC_eq : AC = 10
  AB_eq : AB = 8
  BC_eq : BC = 6

/-- Point on triangle A'B'C'' -/
structure PointOnTriangleABC' where
  x : ℝ
  y : ℝ
  on_triangle : 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1

/-- Sum of distances from a point to the sides of triangle ABC -/
def sum_of_distances (t : RightTriangle) (p : PointOnTriangleABC') : ℝ :=
  p.x * t.AB + p.y * t.BC + 1

/-- Maximum sum of distances theorem -/
theorem max_sum_of_distances (t : RightTriangle) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (p : PointOnTriangleABC'), sum_of_distances t p ≤ max :=
sorry

end max_sum_of_distances_l3792_379254


namespace complementary_angle_theorem_l3792_379267

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary_angles (α β : ℝ) : Prop := α + β = 90

/-- Given complementary angles α and β where α = 40°, prove that β = 50° -/
theorem complementary_angle_theorem (α β : ℝ) 
  (h1 : complementary_angles α β) (h2 : α = 40) : β = 50 := by
  sorry

end complementary_angle_theorem_l3792_379267


namespace multiplication_and_division_problem_l3792_379276

theorem multiplication_and_division_problem : (-12 * 3) + ((-15 * -5) / 3) = -11 := by
  sorry

end multiplication_and_division_problem_l3792_379276


namespace total_buttons_is_117_l3792_379206

/-- The number of buttons Jack uses for all shirts -/
def total_buttons : ℕ :=
  let jack_kids : ℕ := 3
  let jack_shirts_per_kid : ℕ := 3
  let jack_buttons_per_shirt : ℕ := 7
  let neighbor_kids : ℕ := 2
  let neighbor_shirts_per_kid : ℕ := 3
  let neighbor_buttons_per_shirt : ℕ := 9
  
  let jack_total_shirts := jack_kids * jack_shirts_per_kid
  let jack_total_buttons := jack_total_shirts * jack_buttons_per_shirt
  
  let neighbor_total_shirts := neighbor_kids * neighbor_shirts_per_kid
  let neighbor_total_buttons := neighbor_total_shirts * neighbor_buttons_per_shirt
  
  jack_total_buttons + neighbor_total_buttons

/-- Theorem stating that the total number of buttons Jack uses is 117 -/
theorem total_buttons_is_117 : total_buttons = 117 := by
  sorry

end total_buttons_is_117_l3792_379206


namespace sales_growth_rate_is_ten_percent_l3792_379207

/-- The average monthly growth rate of total sales from February to April -/
def average_monthly_growth_rate : ℝ := 0.1

/-- Total sales in February (in yuan) -/
def february_sales : ℝ := 240000

/-- Total sales in April (in yuan) -/
def april_sales : ℝ := 290400

/-- Number of months between February and April -/
def months_between : ℕ := 2

theorem sales_growth_rate_is_ten_percent :
  april_sales = february_sales * (1 + average_monthly_growth_rate) ^ months_between := by
  sorry

end sales_growth_rate_is_ten_percent_l3792_379207


namespace assistant_productivity_increase_l3792_379232

theorem assistant_productivity_increase 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (output_increase_factor : ℝ) 
  (hours_decrease_factor : ℝ) 
  (h1 : output_increase_factor = 1.8) 
  (h2 : hours_decrease_factor = 0.9) :
  (output_increase_factor * base_output) / (hours_decrease_factor * base_hours) / 
  (base_output / base_hours) - 1 = 1 := by
sorry

end assistant_productivity_increase_l3792_379232


namespace factorization_proof_l3792_379231

theorem factorization_proof (x : ℝ) : (x^2 + 4)^2 - 16*x^2 = (x + 2)^2 * (x - 2)^2 := by
  sorry

end factorization_proof_l3792_379231


namespace chess_tournament_players_l3792_379248

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 56) :
  ∃ (n : ℕ), n > 0 ∧ total_games = n * (n - 1) ∧ n = 14 := by
  sorry

end chess_tournament_players_l3792_379248


namespace club_members_after_four_years_l3792_379201

def club_members (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 3 * club_members k - 10

theorem club_members_after_four_years :
  club_members 4 = 1220 := by
  sorry

end club_members_after_four_years_l3792_379201


namespace probability_5_odd_in_8_rolls_l3792_379281

def roll_die_8_times : ℕ := 8
def fair_die_sides : ℕ := 6
def odd_outcomes : ℕ := 3
def target_odd_rolls : ℕ := 5

theorem probability_5_odd_in_8_rolls : 
  (Nat.choose roll_die_8_times target_odd_rolls * (odd_outcomes ^ target_odd_rolls) * ((fair_die_sides - odd_outcomes) ^ (roll_die_8_times - target_odd_rolls))) / (fair_die_sides ^ roll_die_8_times) = 7 / 32 :=
sorry

end probability_5_odd_in_8_rolls_l3792_379281


namespace complex_equation_solution_l3792_379265

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 1 - 3 * Complex.I → z = 6 + Complex.I := by
  sorry

end complex_equation_solution_l3792_379265


namespace solve_exponential_equation_l3792_379278

theorem solve_exponential_equation (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 :=
by
  sorry

end solve_exponential_equation_l3792_379278


namespace maricela_production_l3792_379288

/-- Represents the orange production and juice sale scenario of the Morales sisters. -/
structure OrangeGroveSale where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  alba_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℚ
  total_revenue : ℚ

/-- Calculates the number of oranges Maricela's trees must produce per tree. -/
def maricela_oranges_per_tree (sale : OrangeGroveSale) : ℚ :=
  sorry

/-- Theorem stating that given the conditions, Maricela's trees must produce 500 oranges per tree. -/
theorem maricela_production (sale : OrangeGroveSale) 
  (h1 : sale.trees_per_sister = 110)
  (h2 : sale.gabriela_oranges_per_tree = 600)
  (h3 : sale.alba_oranges_per_tree = 400)
  (h4 : sale.oranges_per_cup = 3)
  (h5 : sale.price_per_cup = 4)
  (h6 : sale.total_revenue = 220000) :
  maricela_oranges_per_tree sale = 500 := by
  sorry

end maricela_production_l3792_379288


namespace newspaper_weeks_l3792_379210

/-- The cost of a weekday newspaper --/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper --/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week --/
def weekday_papers : ℕ := 3

/-- The total amount spent on newspapers --/
def total_spent : ℚ := 28

/-- The number of weeks Hillary buys newspapers --/
def weeks_buying : ℕ := 8

theorem newspaper_weeks : 
  (weekday_papers * weekday_cost + sunday_cost) * weeks_buying = total_spent := by
  sorry

end newspaper_weeks_l3792_379210


namespace f_g_derivatives_neg_l3792_379228

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_neg_pos : ∀ x : ℝ, x > 0 → deriv g (-x) > 0

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (h : x < 0) :
  deriv f x > 0 ∧ deriv g (-x) < 0 := by sorry

end f_g_derivatives_neg_l3792_379228


namespace smallest_max_sum_l3792_379298

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_eq : p + q + r + s + t = 3015) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∃ (min_N : ℕ), 
    (∀ (p' q' r' s' t' : ℕ+), 
      p' + q' + r' + s' + t' = 3015 → 
      max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) ≥ min_N) ∧
    N = min_N ∧
    min_N = 1508 :=
by
  sorry

end smallest_max_sum_l3792_379298


namespace car_speed_l3792_379274

/-- Proves that given the conditions, the speed of the car is 50 miles per hour -/
theorem car_speed (gasoline_consumption : Real) (tank_capacity : Real) (travel_time : Real) (gasoline_used_fraction : Real) :
  gasoline_consumption = 1 / 30 →
  tank_capacity = 10 →
  travel_time = 5 →
  gasoline_used_fraction = 0.8333333333333334 →
  (tank_capacity * gasoline_used_fraction) / (travel_time * gasoline_consumption) = 50 := by
  sorry

end car_speed_l3792_379274


namespace factory_non_defective_percentage_l3792_379275

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  { production_percentage := 0.25, defective_percentage := 0.02 },
  { production_percentage := 0.35, defective_percentage := 0.04 },
  { production_percentage := 0.40, defective_percentage := 0.05 }
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating that the percentage of non-defective products is 96.1% -/
theorem factory_non_defective_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

end factory_non_defective_percentage_l3792_379275


namespace problem_statement_l3792_379241

theorem problem_statement (x y : ℝ) 
  (h1 : (x - y) / (x + y) = 9)
  (h2 : x * y / (x + y) = -60) :
  (x + y) + (x - y) + x * y = -150 := by sorry

end problem_statement_l3792_379241


namespace least_n_for_length_50_l3792_379282

-- Define the points A_n on the x-axis
def A (n : ℕ) : ℝ × ℝ := (0, 0)  -- We only need A_0 for the statement

-- Define the points B_n on y = x^2
def B (n : ℕ) : ℝ × ℝ := sorry

-- Define the property that A_{n-1}B_nA_n is an equilateral triangle
def is_equilateral_triangle (n : ℕ) : Prop := sorry

-- Define the length of A_0A_n
def length_A0An (n : ℕ) : ℝ := sorry

-- The main theorem
theorem least_n_for_length_50 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → length_A0An m < 50) ∧ length_A0An n ≥ 50 ∧ n = 5 := by
  sorry

end least_n_for_length_50_l3792_379282


namespace parallelepiped_dimensions_l3792_379216

theorem parallelepiped_dimensions (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 :=
by sorry

end parallelepiped_dimensions_l3792_379216


namespace quadratic_root_relation_l3792_379299

theorem quadratic_root_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ = 4 * x₁ ∧ x₁^2 + a*x₁ + 2*a = 0 ∧ x₂^2 + a*x₂ + 2*a = 0) → 
  a = 25/2 := by
sorry

end quadratic_root_relation_l3792_379299


namespace star_two_three_solve_equation_l3792_379262

-- Define the new operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_three : star 2 3 = 16 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : star (-2) x = -2 + x → x = 6/5 := by sorry

end star_two_three_solve_equation_l3792_379262


namespace consecutive_page_numbers_l3792_379214

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20460 → n + (n + 1) = 285 := by
  sorry

end consecutive_page_numbers_l3792_379214


namespace condition_necessary_not_sufficient_l3792_379218

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The condition that a_n^2 = a_(n-1) * a_(n+1) for n ≥ 2 -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n ^ 2 = a (n - 1) * a (n + 1)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  ¬(∀ a : Sequence, Condition a → IsGeometric a) :=
by sorry

end condition_necessary_not_sufficient_l3792_379218


namespace function_bound_l3792_379277

theorem function_bound (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1)
  (h2 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1) :
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end function_bound_l3792_379277


namespace sum_of_dimensions_l3792_379240

-- Define the dimensions of the rectangular box
variable (X Y Z : ℝ)

-- Define the surface areas of the faces
def surfaceArea1 : ℝ := 18
def surfaceArea2 : ℝ := 18
def surfaceArea3 : ℝ := 36
def surfaceArea4 : ℝ := 36
def surfaceArea5 : ℝ := 54
def surfaceArea6 : ℝ := 54

-- State the theorem
theorem sum_of_dimensions (h1 : X * Y = surfaceArea1)
                          (h2 : X * Z = surfaceArea5)
                          (h3 : Y * Z = surfaceArea3)
                          (h4 : X > 0) (h5 : Y > 0) (h6 : Z > 0) :
  X + Y + Z = 11 := by
  sorry

end sum_of_dimensions_l3792_379240


namespace number_of_divisors_of_36_l3792_379243

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end number_of_divisors_of_36_l3792_379243


namespace max_true_statements_l3792_379212

theorem max_true_statements (y : ℝ) : 
  let statement1 := 0 < y^3 ∧ y^3 < 1
  let statement2 := y^3 > 1
  let statement3 := -1 < y ∧ y < 0
  let statement4 := 0 < y ∧ y < 1
  let statement5 := 0 < y^2 - y^3 ∧ y^2 - y^3 < 1
  (∃ y : ℝ, (statement1 ∧ statement4 ∧ statement5)) ∧
  (∀ y : ℝ, ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement4) ∧
            ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement5) ∧
            ¬(statement1 ∧ statement2 ∧ statement4 ∧ statement5) ∧
            ¬(statement1 ∧ statement3 ∧ statement4 ∧ statement5) ∧
            ¬(statement2 ∧ statement3 ∧ statement4 ∧ statement5)) :=
by
  sorry

end max_true_statements_l3792_379212


namespace sequence_growth_l3792_379229

theorem sequence_growth (a : ℕ → ℤ) 
  (h1 : a 1 > a 0) 
  (h2 : a 1 > 0) 
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 := by
  sorry

end sequence_growth_l3792_379229


namespace principal_amount_calculation_l3792_379200

/-- Calculate the principal amount given the difference between compound and simple interest -/
theorem principal_amount_calculation (interest_rate : ℝ) (compounding_frequency : ℕ) (time : ℝ) (interest_difference : ℝ) :
  interest_rate = 0.10 →
  compounding_frequency = 2 →
  time = 1 →
  interest_difference = 3.9999999999999147 →
  ∃ (principal : ℝ), 
    (principal * ((1 + interest_rate / compounding_frequency) ^ (compounding_frequency * time) - 1) - 
     principal * interest_rate * time = interest_difference) ∧
    (abs (principal - 1600) < 1) :=
by sorry

end principal_amount_calculation_l3792_379200


namespace calculate_expression_l3792_379226

theorem calculate_expression : (3.6 * 0.5) / 0.2 = 9 := by
  sorry

end calculate_expression_l3792_379226


namespace max_value_log_product_l3792_379279

/-- The maximum value of lg a · lg c given the conditions -/
theorem max_value_log_product (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (∃ (x : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ x) ∧
  (∀ (y : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ y → 16/3 ≤ y) :=
sorry

end max_value_log_product_l3792_379279


namespace max_expression_value_l3792_379233

def expression (a b c d : ℕ) : ℕ := c * a^b + d

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 196 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({0, 1, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 3, 4} : Set ℕ) →
      w ∈ ({0, 1, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 196 :=
by
  sorry

#check max_expression_value

end max_expression_value_l3792_379233


namespace john_needs_thirteen_more_l3792_379249

def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def previous_weekend_earnings : ℕ := 20
def pogo_stick_cost : ℕ := 60

theorem john_needs_thirteen_more : 
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings) = 13 := by
  sorry

end john_needs_thirteen_more_l3792_379249


namespace tg_2alpha_l3792_379256

theorem tg_2alpha (α : Real) 
  (h1 : Real.cos (α - Real.pi/2) = 0.2) 
  (h2 : Real.pi/2 < α ∧ α < Real.pi) : 
  Real.tan (2*α) = -4 * Real.sqrt 6 / 23 := by
sorry

end tg_2alpha_l3792_379256


namespace f_is_smallest_f_is_minimal_l3792_379244

/-- 
For a given integer n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

/-- 
Theorem: For integers n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
theorem f_is_smallest (n : ℕ) (h : n ≥ 4) : 
  ∀ (m : ℕ+), ∀ (S : Finset ℕ), 
    S.card = f n → 
    (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) → 
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by
  sorry

/-- 
Corollary: There is no smaller integer than f(n) that satisfies 
the conditions for all n ≥ 4.
-/
theorem f_is_minimal (n : ℕ) (h : n ≥ 4) :
  ∀ g : ℕ → ℕ, (∀ k ≥ 4, g k < f k) → 
    ∃ (m : ℕ+) (S : Finset ℕ), 
      S.card = g n ∧
      (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → 
        ¬(Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) :=
by
  sorry

end f_is_smallest_f_is_minimal_l3792_379244


namespace min_swaps_at_most_five_l3792_379290

/-- Represents a 4026-digit number composed of ones and twos -/
structure NumberConfig :=
  (ones_count : Nat)
  (twos_count : Nat)
  (total_digits : Nat)
  (h1 : ones_count = 2013)
  (h2 : twos_count = 2013)
  (h3 : total_digits = 4026)
  (h4 : ones_count + twos_count = total_digits)

/-- Represents the state of the number after some swaps -/
structure NumberState :=
  (config : NumberConfig)
  (ones_in_odd : Nat)
  (h : ones_in_odd ≤ config.ones_count)

/-- Checks if a NumberState is divisible by 11 -/
def isDivisibleBy11 (state : NumberState) : Prop :=
  (state.config.total_digits - 2 * state.ones_in_odd) % 11 = 0

/-- The minimum number of swaps required to make the number divisible by 11 -/
def minSwapsToDiv11 (state : NumberState) : Nat :=
  min (state.ones_in_odd % 11) ((11 - state.ones_in_odd % 11) % 11)

/-- The main theorem stating that the minimum number of swaps is at most 5 -/
theorem min_swaps_at_most_five (state : NumberState) : minSwapsToDiv11 state ≤ 5 := by
  sorry


end min_swaps_at_most_five_l3792_379290


namespace existence_of_function_with_properties_l3792_379219

theorem existence_of_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧ 
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧ 
  (∃ z : ℝ, f z < f 0) ∧
  (let g : ℝ → ℝ := fun x ↦ (x - 1)^2;
   (∀ x : ℝ, g (1 + x) = g (1 - x)) ∧ 
   (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → g y ≤ g x) ∧ 
   (∃ z : ℝ, g z < g 0)) :=
by sorry

end existence_of_function_with_properties_l3792_379219


namespace remaining_problems_l3792_379255

theorem remaining_problems (total : ℕ) (first_20min : ℕ) (second_20min : ℕ) 
  (h1 : total = 75)
  (h2 : first_20min = 10)
  (h3 : second_20min = 2 * first_20min) :
  total - (first_20min + second_20min) = 45 := by
  sorry

end remaining_problems_l3792_379255


namespace triangle_sine_inequality_l3792_379220

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = 180) :
  (3 : ℝ) * Real.sqrt 3 / 2 ≥ Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≥ -2 := by
  sorry

end triangle_sine_inequality_l3792_379220


namespace independence_test_confidence_l3792_379213

/-- The critical value for the independence test -/
def critical_value : ℝ := 5.024

/-- The confidence level for "X and Y are related" given k > critical_value -/
def confidence_level (k : ℝ) : ℝ := 97.5

/-- Theorem stating that when k > critical_value, the confidence level is 97.5% -/
theorem independence_test_confidence (k : ℝ) (h : k > critical_value) :
  confidence_level k = 97.5 := by sorry

end independence_test_confidence_l3792_379213


namespace yellow_marbles_count_l3792_379280

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end yellow_marbles_count_l3792_379280


namespace pens_to_pencils_ratio_l3792_379285

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total : Nat
  pencils : Nat
  eraser : Nat
  pens : Nat

/-- Theorem stating the ratio of pens to pencils in Tommy's pencil case -/
theorem pens_to_pencils_ratio (case : PencilCase) 
  (h_total : case.total = 13)
  (h_pencils : case.pencils = 4)
  (h_eraser : case.eraser = 1)
  (h_sum : case.total = case.pencils + case.pens + case.eraser)
  (h_multiple : ∃ k : Nat, case.pens = k * case.pencils) :
  case.pens / case.pencils = 2 := by
  sorry

end pens_to_pencils_ratio_l3792_379285


namespace budi_can_win_l3792_379258

/-- The set of numbers from which players choose -/
def S : Finset ℕ := Finset.range 30

/-- The total number of balls in the game -/
def totalBalls : ℕ := 2015

/-- Astri's chosen numbers -/
structure AstriChoice where
  a : ℕ
  b : ℕ
  a_in_S : a ∈ S
  b_in_S : b ∈ S
  a_ne_b : a ≠ b

/-- Budi's chosen numbers -/
structure BudiChoice (ac : AstriChoice) where
  c : ℕ
  d : ℕ
  c_in_S : c ∈ S
  d_in_S : d ∈ S
  c_ne_d : c ≠ d
  c_ne_a : c ≠ ac.a
  c_ne_b : c ≠ ac.b
  d_ne_a : d ≠ ac.a
  d_ne_b : d ≠ ac.b

/-- The game state -/
structure GameState where
  ballsLeft : ℕ
  astriTurn : Bool

/-- A winning strategy for Budi -/
def isWinningStrategy (ac : AstriChoice) (bc : BudiChoice ac) (strategy : GameState → ℕ) : Prop :=
  ∀ (gs : GameState), 
    (gs.astriTurn ∧ gs.ballsLeft < ac.a ∧ gs.ballsLeft < ac.b) ∨
    (¬gs.astriTurn ∧ 
      ((strategy gs = bc.c ∧ gs.ballsLeft ≥ bc.c) ∨ 
       (strategy gs = bc.d ∧ gs.ballsLeft ≥ bc.d)))

/-- The main theorem -/
theorem budi_can_win : 
  ∀ (ac : AstriChoice), ∃ (bc : BudiChoice ac) (strategy : GameState → ℕ), 
    isWinningStrategy ac bc strategy :=
sorry

end budi_can_win_l3792_379258


namespace parallelogram_height_l3792_379223

-- Define the parallelogram
def parallelogram_area : ℝ := 72
def parallelogram_base : ℝ := 12

-- Theorem to prove
theorem parallelogram_height :
  parallelogram_area / parallelogram_base = 6 :=
by
  sorry


end parallelogram_height_l3792_379223


namespace power_of_two_expression_l3792_379215

theorem power_of_two_expression : 2^4 * 2^2 + 2^4 / 2^2 = 68 := by
  sorry

end power_of_two_expression_l3792_379215


namespace smallest_difference_l3792_379297

def Digits : Finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

def valid_pair (a b : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (a / 10^i) % 10) {0,1,2,3} ∪ Finset.image (λ i => (b / 10^i) % 10) {0,1,2,3})) = 8)

theorem smallest_difference :
  ∃ (a b : ℕ), valid_pair a b ∧
    (a > b) ∧
    (a - b = 247) ∧
    (∀ (c d : ℕ), valid_pair c d → c > d → c - d ≥ 247) :=
sorry

end smallest_difference_l3792_379297


namespace chocolate_bars_l3792_379259

theorem chocolate_bars (cost_per_bar : ℕ) (remaining_bars : ℕ) (revenue : ℕ) :
  cost_per_bar = 4 →
  remaining_bars = 3 →
  revenue = 20 →
  ∃ total_bars : ℕ, total_bars = 8 ∧ cost_per_bar * (total_bars - remaining_bars) = revenue :=
by
  sorry

end chocolate_bars_l3792_379259


namespace garden_length_l3792_379294

/-- A rectangular garden with given perimeter and breadth has a specific length. -/
theorem garden_length (perimeter breadth : ℝ) (h_perimeter : perimeter = 600) (h_breadth : breadth = 150) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 150 := by
sorry

end garden_length_l3792_379294


namespace batsman_innings_properties_l3792_379242

/-- Represents a cricket batsman's innings statistics -/
structure BatsmanInnings where
  total_runs : ℕ
  total_balls : ℕ
  singles : ℕ
  doubles : ℕ

/-- Calculates the percentage of runs scored by running between wickets -/
def runs_by_running_percentage (innings : BatsmanInnings) : ℚ :=
  ((innings.singles + 2 * innings.doubles : ℚ) / innings.total_runs) * 100

/-- Calculates the strike rate of the batsman -/
def strike_rate (innings : BatsmanInnings) : ℚ :=
  (innings.total_runs : ℚ) / innings.total_balls * 100

/-- Theorem stating the properties of the given batsman's innings -/
theorem batsman_innings_properties :
  let innings : BatsmanInnings := {
    total_runs := 180,
    total_balls := 120,
    singles := 35,
    doubles := 15
  }
  runs_by_running_percentage innings = 36.11 ∧
  strike_rate innings = 150 := by
  sorry

end batsman_innings_properties_l3792_379242


namespace log_8_40_l3792_379234

theorem log_8_40 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 40 / Real.log 8 = 1 + c / (3 * a) := by
  sorry

end log_8_40_l3792_379234


namespace smallest_divisible_number_l3792_379230

theorem smallest_divisible_number (n : ℕ) : 
  n = 1008 → 
  (1020 - 12 = n) → 
  (∃ k : ℕ, 36 * k = n) → 
  (∃ k : ℕ, 48 * k = n) → 
  (∃ k : ℕ, 56 * k = n) → 
  ∀ m : ℕ, m ∣ 1008 ∧ m ∣ 36 ∧ m ∣ 48 ∧ m ∣ 56 → m ≤ n :=
by sorry

#check smallest_divisible_number

end smallest_divisible_number_l3792_379230


namespace triangle_inequality_theorem_l3792_379209

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  (t.a / (2 * (t.b + t.c))) + (t.b / (2 * (t.c + t.a))) + (t.c / (2 * (t.a + t.b))) ≥ 3/4 := by
  sorry

end triangle_inequality_theorem_l3792_379209


namespace coefficient_x_cubed_in_expansion_l3792_379261

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℚ := 1
  let b : ℚ := 2
  let r : ℕ := 3
  let binomial_coeff := Nat.choose n r
  binomial_coeff * b^r = 80 := by
  sorry

end coefficient_x_cubed_in_expansion_l3792_379261


namespace experts_win_probability_l3792_379283

/-- The probability of Experts winning a single round -/
def p_win : ℝ := 0.6

/-- The probability of Experts losing a single round -/
def p_lose : ℝ := 1 - p_win

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Viewers -/
def viewers_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that Experts will win the game from the current position -/
def experts_win_prob : ℝ :=
  p_win ^ 3 + 3 * p_win ^ 3 * p_lose

theorem experts_win_probability :
  experts_win_prob = 0.4752 :=
sorry

end experts_win_probability_l3792_379283


namespace copy_machines_total_output_l3792_379272

/-- 
Given two copy machines with constant rates:
- Machine 1 makes 35 copies per minute
- Machine 2 makes 65 copies per minute

Prove that they make 3000 copies together in 30 minutes.
-/
theorem copy_machines_total_output : 
  let machine1_rate : ℕ := 35
  let machine2_rate : ℕ := 65
  let time_in_minutes : ℕ := 30
  (machine1_rate * time_in_minutes) + (machine2_rate * time_in_minutes) = 3000 := by
  sorry

end copy_machines_total_output_l3792_379272


namespace hat_price_theorem_l3792_379211

theorem hat_price_theorem (final_price : ℚ) 
  (h1 : final_price = 8)
  (h2 : ∃ original_price : ℚ, 
    final_price = original_price * (1/5) * (1 + 1/5)) : 
  ∃ original_price : ℚ, original_price = 100/3 ∧ 
    final_price = original_price * (1/5) * (1 + 1/5) := by
  sorry

end hat_price_theorem_l3792_379211


namespace line_parallel_to_plane_parallel_to_line_set_l3792_379205

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define parallelism between a line and a plane
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define a set of parallel lines within a plane
def parallel_lines_in_plane (p : Plane3D) : Set Line3D :=
  sorry

-- Define parallelism between two lines
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- The theorem to be proved
theorem line_parallel_to_plane_parallel_to_line_set 
  (a : Line3D) (α : Plane3D) 
  (h : line_parallel_to_plane a α) :
  ∃ (S : Set Line3D), S ⊆ parallel_lines_in_plane α ∧ 
    ∀ l ∈ S, lines_parallel a l :=
  sorry

end line_parallel_to_plane_parallel_to_line_set_l3792_379205


namespace peter_walk_time_l3792_379217

/-- Calculates the remaining time to walk given total distance, walking speed, and distance already walked -/
def remaining_walk_time (total_distance : ℝ) (walking_speed : ℝ) (distance_walked : ℝ) : ℝ :=
  (total_distance - distance_walked) * walking_speed

theorem peter_walk_time :
  let total_distance : ℝ := 2.5
  let walking_speed : ℝ := 20
  let distance_walked : ℝ := 1
  remaining_walk_time total_distance walking_speed distance_walked = 30 := by
sorry

end peter_walk_time_l3792_379217


namespace total_people_on_bus_l3792_379292

/-- The number of people initially on the bus -/
def initial_people : ℕ := 4

/-- The number of people who got on the bus at the stop -/
def people_who_got_on : ℕ := 13

/-- Theorem stating the total number of people on the bus after the stop -/
theorem total_people_on_bus : initial_people + people_who_got_on = 17 := by
  sorry

end total_people_on_bus_l3792_379292


namespace no_descending_nat_function_exists_descending_int_function_l3792_379238

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function exists when the range is ℕ
theorem no_descending_nat_function :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) :=
sorry

-- Theorem 2: Such a function exists when the range is ℤ
theorem exists_descending_int_function :
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) :=
sorry

end no_descending_nat_function_exists_descending_int_function_l3792_379238


namespace solve_for_b_l3792_379247

theorem solve_for_b (a b c d m : ℝ) (h1 : a ≠ b) (h2 : m = (c * a * d * b) / (a - b)) :
  b = (m * a) / (c * a * d + m) := by
sorry

end solve_for_b_l3792_379247


namespace factorial_8_divisors_l3792_379270

theorem factorial_8_divisors : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by sorry

end factorial_8_divisors_l3792_379270


namespace vovochka_candy_theorem_l3792_379252

/-- Given a total number of candies and classmates, calculates the maximum number
    of candies Vovochka can keep while satisfying the distribution condition. -/
def max_candies_for_vovochka (total_candies : ℕ) (num_classmates : ℕ) : ℕ :=
  total_candies - (num_classmates - 1) * 7 + 4

/-- Checks if the candy distribution satisfies the condition that
    any 16 classmates have at least 100 candies. -/
def satisfies_condition (candies_kept : ℕ) (total_candies : ℕ) (num_classmates : ℕ) : Prop :=
  ∀ (group : Finset (Fin num_classmates)),
    group.card = 16 →
    (total_candies - candies_kept) * 16 / num_classmates ≥ 100

theorem vovochka_candy_theorem (total_candies num_classmates : ℕ)
    (h1 : total_candies = 200)
    (h2 : num_classmates = 25) :
    let max_candies := max_candies_for_vovochka total_candies num_classmates
    satisfies_condition max_candies total_candies num_classmates ∧
    ∀ k, k > max_candies →
      ¬satisfies_condition k total_candies num_classmates :=
  sorry

end vovochka_candy_theorem_l3792_379252


namespace cuboid_third_edge_length_l3792_379204

/-- Given a cuboid with two known edge lengths and its volume, 
    calculate the length of the third edge. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) :
  edge1 = 2 →
  edge2 = 5 →
  volume = 30 →
  volume = edge1 * edge2 * third_edge →
  third_edge = 3 := by
  sorry

end cuboid_third_edge_length_l3792_379204


namespace square_minus_five_equals_two_l3792_379257

theorem square_minus_five_equals_two : ∃ x : ℤ, x - 5 = 2 := by
  use 7
  sorry

end square_minus_five_equals_two_l3792_379257


namespace mikes_age_l3792_379245

theorem mikes_age (mike anna : ℝ) 
  (h1 : mike = 3 * anna - 20)
  (h2 : mike + anna = 70) : 
  mike = 47.5 := by
sorry

end mikes_age_l3792_379245


namespace quadratic_two_distinct_roots_l3792_379296

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l3792_379296


namespace solution_exists_l3792_379221

theorem solution_exists : ∃ x : ℝ, 0.05 < x ∧ x < 0.051 :=
by
  use 0.0505
  sorry

#check solution_exists

end solution_exists_l3792_379221


namespace simplify_expression_l3792_379273

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l3792_379273
