import Mathlib

namespace revenue_difference_is_400_l4062_406216

/-- Represents the revenue difference between making elephant and giraffe statues -/
def revenue_difference (total_jade : ℕ) (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_price : ℕ) : ℕ :=
  let elephant_jade := 2 * giraffe_jade
  let num_giraffes := total_jade / giraffe_jade
  let num_elephants := total_jade / elephant_jade
  let giraffe_revenue := num_giraffes * giraffe_price
  let elephant_revenue := num_elephants * elephant_price
  elephant_revenue - giraffe_revenue

/-- Proves that the revenue difference is $400 for the given conditions -/
theorem revenue_difference_is_400 :
  revenue_difference 1920 120 150 350 = 400 := by
  sorry

end revenue_difference_is_400_l4062_406216


namespace relation_xyz_l4062_406267

theorem relation_xyz (x y z t : ℝ) (h : x / Real.sin t = y / Real.sin (2 * t) ∧ 
                                        x / Real.sin t = z / Real.sin (3 * t)) : 
  x^2 - y^2 + x*z = 0 := by
sorry

end relation_xyz_l4062_406267


namespace unique_modulus_for_xy_plus_one_implies_x_plus_y_l4062_406260

theorem unique_modulus_for_xy_plus_one_implies_x_plus_y (n : ℕ+) : 
  (∀ x y : ℤ, (x * y + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 := by
  sorry

end unique_modulus_for_xy_plus_one_implies_x_plus_y_l4062_406260


namespace lizas_paycheck_amount_l4062_406219

/-- Calculates the amount of Liza's paycheck given her initial balance, expenses, and final balance -/
def calculate_paycheck (initial_balance rent electricity internet phone final_balance : ℕ) : ℕ :=
  final_balance + rent + electricity + internet + phone - initial_balance

/-- Theorem stating that Liza's paycheck is $1563 given the provided financial information -/
theorem lizas_paycheck_amount :
  calculate_paycheck 800 450 117 100 70 1563 = 1563 := by
  sorry

end lizas_paycheck_amount_l4062_406219


namespace commission_percentage_is_21_875_l4062_406233

/-- Calculates the commission percentage for a sale with given rates and total amount -/
def commission_percentage (rate_below_500 : ℚ) (rate_above_500 : ℚ) (total_amount : ℚ) : ℚ :=
  let commission_below_500 := min total_amount 500 * rate_below_500
  let commission_above_500 := max (total_amount - 500) 0 * rate_above_500
  let total_commission := commission_below_500 + commission_above_500
  (total_commission / total_amount) * 100

/-- Theorem stating that the commission percentage for the given problem is 21.875% -/
theorem commission_percentage_is_21_875 :
  commission_percentage (20 / 100) (25 / 100) 800 = 21875 / 1000 := by sorry

end commission_percentage_is_21_875_l4062_406233


namespace y1_less_than_y2_l4062_406200

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_less_than_y2 (y₁ y₂ : ℝ) :
  quadratic_function (-1) = y₁ →
  quadratic_function 4 = y₂ →
  y₁ < y₂ := by
  sorry

end y1_less_than_y2_l4062_406200


namespace boat_travel_distance_l4062_406297

/-- Prove that the distance between two destinations is 40 km given the specified conditions. -/
theorem boat_travel_distance 
  (boatsman_speed : ℝ) 
  (river_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : boatsman_speed = 7)
  (h2 : river_speed = 3)
  (h3 : time_difference = 6)
  (h4 : (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference = 
        2 * river_speed * boatsman_speed * (boatsman_speed - river_speed)) :
  (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference / 
  (2 * river_speed) = 40 := by
  sorry

end boat_travel_distance_l4062_406297


namespace probability_factor_90_less_than_8_l4062_406270

def positive_factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x > 0 ∧ n % x = 0) (Finset.range (n + 1))

def factors_less_than (n k : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x < k) (positive_factors n)

theorem probability_factor_90_less_than_8 :
  (Finset.card (factors_less_than 90 8) : ℚ) / (Finset.card (positive_factors 90) : ℚ) = 5 / 12 := by
  sorry

end probability_factor_90_less_than_8_l4062_406270


namespace unique_solution_for_x_l4062_406227

theorem unique_solution_for_x : ∃! x : ℝ, 
  (x ≠ 2) ∧ 
  ((x^3 - 8) / (x - 2) = 3 * x^2) ∧ 
  (x = -1) := by
sorry

end unique_solution_for_x_l4062_406227


namespace benches_around_circular_track_l4062_406238

/-- The radius of the circular walking track in feet -/
def radius : ℝ := 15

/-- The spacing between benches in feet -/
def bench_spacing : ℝ := 3

/-- The number of benches needed for the circular track -/
def num_benches : ℕ := 31

/-- Theorem stating that the number of benches needed is approximately 31 -/
theorem benches_around_circular_track :
  Int.floor ((2 * Real.pi * radius) / bench_spacing) = num_benches := by
  sorry

end benches_around_circular_track_l4062_406238


namespace x_range_for_negative_f_l4062_406205

def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0 → x ∈ Set.Ioo 1 2) :=
sorry

end x_range_for_negative_f_l4062_406205


namespace condition_sufficient_not_necessary_l4062_406276

theorem condition_sufficient_not_necessary : 
  (∃ (S T : Set ℝ), 
    (S = {x : ℝ | x - 1 > 0}) ∧ 
    (T = {x : ℝ | x^2 - 1 > 0}) ∧ 
    (S ⊂ T) ∧ 
    (∃ x, x ∈ T ∧ x ∉ S)) := by sorry

end condition_sufficient_not_necessary_l4062_406276


namespace trigonometric_identity_l4062_406225

open Real

theorem trigonometric_identity (α β : ℝ) 
  (h1 : sin (π - α) - 2 * sin ((π / 2) + α) = 0) 
  (h2 : tan (α + β) = -1) : 
  (sin α * cos α + sin α ^ 2 = 6 / 5) ∧ 
  (tan β = 3) := by
  sorry

end trigonometric_identity_l4062_406225


namespace legs_more_than_twice_heads_l4062_406259

-- Define the group of animals
structure AnimalGroup where
  donkeys : ℕ
  pigs : ℕ

-- Define the properties of the group
def AnimalGroup.heads (g : AnimalGroup) : ℕ := g.donkeys + g.pigs
def AnimalGroup.legs (g : AnimalGroup) : ℕ := 4 * g.donkeys + 4 * g.pigs

-- Theorem statement
theorem legs_more_than_twice_heads (g : AnimalGroup) (h : g.donkeys = 8) :
  g.legs ≥ 2 * g.heads + 16 := by
  sorry

end legs_more_than_twice_heads_l4062_406259


namespace toy_cost_calculation_l4062_406240

/-- Represents the initial weekly cost price of a toy in Rupees -/
def initial_cost : ℝ := 1300

/-- Number of toys sold -/
def num_toys : ℕ := 18

/-- Discount rate applied to the toys -/
def discount_rate : ℝ := 0.1

/-- Total revenue from the sale in Rupees -/
def total_revenue : ℝ := 27300

theorem toy_cost_calculation :
  initial_cost * num_toys * (1 - discount_rate) = total_revenue - 3 * initial_cost := by
  sorry

#check toy_cost_calculation

end toy_cost_calculation_l4062_406240


namespace max_dot_product_CA_CB_l4062_406299

/-- Given planar vectors OA, OB, and OC satisfying certain conditions,
    the maximum value of CA · CB is 3. -/
theorem max_dot_product_CA_CB (OA OB OC : ℝ × ℝ) : 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- OA · OB = 0
  (OA.1^2 + OA.2^2 = 1) →            -- |OA| = 1
  (OC.1^2 + OC.2^2 = 1) →            -- |OC| = 1
  (OB.1^2 + OB.2^2 = 3) →            -- |OB| = √3
  (∃ (CA CB : ℝ × ℝ), 
    CA = (OA.1 - OC.1, OA.2 - OC.2) ∧ 
    CB = (OB.1 - OC.1, OB.2 - OC.2) ∧
    ∀ (CA' CB' : ℝ × ℝ), 
      CA' = (OA.1 - OC.1, OA.2 - OC.2) → 
      CB' = (OB.1 - OC.1, OB.2 - OC.2) →
      CA'.1 * CB'.1 + CA'.2 * CB'.2 ≤ 3) :=
by sorry

end max_dot_product_CA_CB_l4062_406299


namespace sqrt_3_times_612_times_3_and_half_l4062_406202

theorem sqrt_3_times_612_times_3_and_half : Real.sqrt 3 * 612 * (3 + 3/2) = 3 := by
  sorry

end sqrt_3_times_612_times_3_and_half_l4062_406202


namespace pollywogs_disappearance_l4062_406268

/-- The number of pollywogs that mature into toads and leave the pond per day -/
def maturation_rate : ℕ := 50

/-- The number of pollywogs Melvin catches per day for the first 20 days -/
def melvin_catch_rate : ℕ := 10

/-- The number of days Melvin catches pollywogs -/
def melvin_catch_days : ℕ := 20

/-- The total number of days it took for all pollywogs to disappear -/
def total_days : ℕ := 44

/-- The initial number of pollywogs in the pond -/
def initial_pollywogs : ℕ := 2400

theorem pollywogs_disappearance :
  initial_pollywogs = 
    (maturation_rate + melvin_catch_rate) * melvin_catch_days + 
    maturation_rate * (total_days - melvin_catch_days) := by
  sorry

end pollywogs_disappearance_l4062_406268


namespace radio_range_is_125_l4062_406262

/-- The range of radios for two teams traveling in opposite directions --/
def radio_range (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The radio range for the given scenario is 125 miles --/
theorem radio_range_is_125 :
  radio_range 20 30 2.5 = 125 := by
  sorry

end radio_range_is_125_l4062_406262


namespace binomial_variance_problem_l4062_406283

/-- A function representing the expectation of a binomial distribution -/
def expectation_binomial (n : ℕ) (p : ℝ) : ℝ := n * p

/-- A function representing the variance of a binomial distribution -/
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (p : ℝ) 
  (hX : expectation_binomial 3 p = 1) 
  (hY : expectation_binomial 4 p = 4 * p) :
  variance_binomial 4 p = 8/9 := by
sorry

end binomial_variance_problem_l4062_406283


namespace set_A_properties_l4062_406273

/-- Property P: For any i, j (1 ≤ i ≤ j ≤ n), at least one of aᵢaⱼ and aⱼ/aᵢ belongs to A -/
def property_P (A : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ A → y ∈ A → x ≤ y → (x * y ∈ A ∨ y / x ∈ A)

theorem set_A_properties {n : ℕ} (A : Set ℝ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_A : A = {x | ∃ i, i ∈ Finset.range n ∧ x = a i})
  (h_sorted : ∀ i j, i < j → j < n → a i < a j)
  (h_P : property_P A) :
  (a 0 = 1) ∧ 
  ((Finset.range n).sum a / (Finset.range n).sum (λ i => (a i)⁻¹) = a (n - 1)) ∧
  (n = 5 → ∃ r : ℝ, ∀ i, i < 4 → a (i + 1) = r * a i) :=
by sorry

end set_A_properties_l4062_406273


namespace scale_drawing_conversion_l4062_406264

/-- Given a scale where 1 inch represents 1000 feet, 
    a line segment of 3.6 inches represents 3600 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (drawing_length : ℝ) :
  scale = 1000 →
  drawing_length = 3.6 →
  drawing_length * scale = 3600 := by
  sorry

end scale_drawing_conversion_l4062_406264


namespace sum_of_angles_in_divided_hexagon_l4062_406279

-- Define a hexagon divided into two quadrilaterals
structure DividedHexagon where
  quad1 : Finset ℕ
  quad2 : Finset ℕ
  h1 : quad1.card = 4
  h2 : quad2.card = 4
  h3 : quad1 ∩ quad2 = ∅
  h4 : quad1 ∪ quad2 = Finset.range 8

-- Define the sum of angles in a quadrilateral (in degrees)
def quadrilateralAngleSum : ℕ := 360

-- Theorem statement
theorem sum_of_angles_in_divided_hexagon (h : DividedHexagon) :
  (h.quad1.sum (λ i => quadrilateralAngleSum / 4)) +
  (h.quad2.sum (λ i => quadrilateralAngleSum / 4)) = 720 := by
  sorry

end sum_of_angles_in_divided_hexagon_l4062_406279


namespace mary_nickels_l4062_406274

theorem mary_nickels (initial_nickels : ℕ) (dad_gave_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_gave_nickels = 5) : 
  initial_nickels + dad_gave_nickels = 12 := by
sorry

end mary_nickels_l4062_406274


namespace angle_expression_equality_l4062_406242

theorem angle_expression_equality (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin (2*π + θ) - Real.sin (π/2 - θ) = Real.sqrt 5 / 2 := by
  sorry

end angle_expression_equality_l4062_406242


namespace captain_age_proof_l4062_406229

def cricket_team_problem (team_size : ℕ) (team_avg_age : ℕ) (age_diff : ℕ) (remaining_avg_diff : ℕ) : Prop :=
  let captain_age : ℕ := 26
  let keeper_age : ℕ := captain_age + age_diff
  let total_age : ℕ := team_size * team_avg_age
  let remaining_players : ℕ := team_size - 2
  let remaining_avg : ℕ := team_avg_age - remaining_avg_diff
  total_age = captain_age + keeper_age + remaining_players * remaining_avg

theorem captain_age_proof :
  cricket_team_problem 11 23 3 1 := by
  sorry

end captain_age_proof_l4062_406229


namespace simple_interest_rate_problem_l4062_406214

/-- Given the principal, amount, time, and formulas for simple interest and amount,
    prove that the rate percent is 5%. -/
theorem simple_interest_rate_problem (P A : ℕ) (T : ℕ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 4) :
  ∃ R : ℚ,
    R = 5 ∧
    A = P + P * R * (T : ℚ) / 100 :=
by sorry

end simple_interest_rate_problem_l4062_406214


namespace complex_sum_of_parts_l4062_406257

theorem complex_sum_of_parts (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) :
  z.re + z.im = 1 := by sorry

end complex_sum_of_parts_l4062_406257


namespace solve_bank_problem_l4062_406248

def bank_problem (initial_balance : ℚ) : Prop :=
  let tripled_balance := initial_balance * 3
  let balance_after_withdrawal := tripled_balance - 250
  balance_after_withdrawal = 950

theorem solve_bank_problem :
  ∃ (initial_balance : ℚ), bank_problem initial_balance ∧ initial_balance = 400 :=
by
  sorry

end solve_bank_problem_l4062_406248


namespace cost_prices_calculation_l4062_406269

/-- Represents the cost price of an item -/
structure CostPrice where
  value : ℝ
  positive : value > 0

/-- Represents the selling price of an item -/
structure SellingPrice where
  value : ℝ
  positive : value > 0

/-- Calculates the selling price given a cost price and a percentage change -/
def calculateSellingPrice (cp : CostPrice) (percentageChange : ℝ) : SellingPrice :=
  { value := cp.value * (1 + percentageChange),
    positive := sorry }

/-- Determines if two real numbers are approximately equal within a small tolerance -/
def approximatelyEqual (x y : ℝ) : Prop :=
  |x - y| < 0.01

theorem cost_prices_calculation
  (diningSet : CostPrice)
  (chandelier : CostPrice)
  (sofaSet : CostPrice)
  (diningSetSelling : SellingPrice)
  (chandelierSelling : SellingPrice)
  (sofaSetSelling : SellingPrice) :
  (diningSetSelling = calculateSellingPrice diningSet (-0.18)) →
  (calculateSellingPrice diningSet 0.15).value = diningSetSelling.value + 2500 →
  (chandelierSelling = calculateSellingPrice chandelier 0.20) →
  (calculateSellingPrice chandelier (-0.20)).value = chandelierSelling.value - 3000 →
  (sofaSetSelling = calculateSellingPrice sofaSet (-0.10)) →
  (calculateSellingPrice sofaSet 0.25).value = sofaSetSelling.value + 4000 →
  approximatelyEqual diningSet.value 7576 ∧
  chandelier.value = 7500 ∧
  approximatelyEqual sofaSet.value 11429 := by
  sorry

#check cost_prices_calculation

end cost_prices_calculation_l4062_406269


namespace periodic_function_property_l4062_406207

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers,
    if f(2011) = 5, then f(2012) = 3 -/
theorem periodic_function_property (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2011 = 5) → (f 2012 = 3) := by sorry

end periodic_function_property_l4062_406207


namespace max_value_theorem_l4062_406258

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) :
  (Real.sqrt (x + 64) + 2 * Real.sqrt (25 - x) + Real.sqrt x) ≤ Real.sqrt 328 ∧
  ∃ x₀, 0 ≤ x₀ ∧ x₀ ≤ 25 ∧ Real.sqrt (x₀ + 64) + 2 * Real.sqrt (25 - x₀) + Real.sqrt x₀ = Real.sqrt 328 :=
by sorry

end max_value_theorem_l4062_406258


namespace students_guinea_pigs_difference_l4062_406222

theorem students_guinea_pigs_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 22 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 95 :=
by sorry

end students_guinea_pigs_difference_l4062_406222


namespace greatest_common_divisor_of_arithmetic_sum_l4062_406292

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of the first 12 terms of an arithmetic sequence with first term a and common difference d -/
def sum_12_terms (a d : ℤ) : ℤ := arithmetic_sum 12 a d

theorem greatest_common_divisor_of_arithmetic_sum :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a d : ℕ), (sum_12_terms a d).natAbs % k = 0) ∧
  (∀ (m : ℕ), m > k → ∃ (a d : ℕ), (sum_12_terms a d).natAbs % m ≠ 0) ∧
  k = 6 :=
sorry

end greatest_common_divisor_of_arithmetic_sum_l4062_406292


namespace steve_union_dues_l4062_406230

/-- Calculate the amount lost to local union dues given gross salary, tax rate, healthcare rate, and take-home pay -/
def union_dues (gross_salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (take_home_pay : ℝ) : ℝ :=
  gross_salary - (tax_rate * gross_salary) - (healthcare_rate * gross_salary) - take_home_pay

/-- Theorem: Given Steve's financial information, prove that he loses $800 to local union dues -/
theorem steve_union_dues :
  union_dues 40000 0.20 0.10 27200 = 800 := by
  sorry

end steve_union_dues_l4062_406230


namespace victor_books_left_l4062_406204

def book_count (initial bought gifted donated : ℕ) : ℕ :=
  initial + bought - gifted - donated

theorem victor_books_left : book_count 25 12 7 15 = 15 := by
  sorry

end victor_books_left_l4062_406204


namespace equation_solution_l4062_406289

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 3 * x = 500 - (4 * x + 5 * x) + 20) ∧ (x = 520 / 14) := by
  sorry

end equation_solution_l4062_406289


namespace unique_N_for_210_terms_l4062_406295

/-- The number of terms in the expansion of (a+b+c+d+1)^n that contain all four variables
    a, b, c, and d, each to some positive power -/
def numTermsWithAllVars (n : ℕ) : ℕ := Nat.choose n 4

theorem unique_N_for_210_terms :
  ∃! N : ℕ, N > 0 ∧ numTermsWithAllVars N = 210 := by sorry

end unique_N_for_210_terms_l4062_406295


namespace equipment_prices_l4062_406203

theorem equipment_prices (price_A price_B : ℝ) 
  (h1 : price_A = price_B + 25)
  (h2 : 2000 / price_A = 2 * (750 / price_B)) :
  price_A = 100 ∧ price_B = 75 := by
  sorry

end equipment_prices_l4062_406203


namespace intersection_A_B_l4062_406206

def A : Set ℝ := {-3, -1, 1, 2}
def B : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l4062_406206


namespace min_perimeter_triangle_l4062_406245

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℕ)

-- Define the condition for integer side lengths and no two sides being equal
def validTriangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the semiperimeter
def semiperimeter (t : Triangle) : ℚ :=
  (t.a + t.b + t.c) / 2

-- Define the inradius
def inradius (t : Triangle) : ℚ :=
  let s := semiperimeter t
  (s - t.a) * (s - t.b) * (s - t.c) / s

-- Define the excircle radii
def excircleRadius (t : Triangle) (side : ℕ) : ℚ :=
  let s := semiperimeter t
  let r := inradius t
  r * s / (s - side)

-- Define the tangency conditions
def tangencyConditions (t : Triangle) : Prop :=
  let r := inradius t
  let rA := excircleRadius t t.a
  let rB := excircleRadius t t.b
  let rC := excircleRadius t t.c
  r + rA = rB ∧ r + rA = rC

-- Theorem statement
theorem min_perimeter_triangle (t : Triangle) :
  validTriangle t → tangencyConditions t → t.a + t.b + t.c ≥ 12 :=
by sorry

end min_perimeter_triangle_l4062_406245


namespace triangle_side_length_l4062_406217

/-- Given a triangle XYZ with sides x, y, and z, where y = 7, z = 6, and cos(Y - Z) = 47/64,
    prove that x = √63.75 -/
theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 6 →
  Real.cos (Y - Z) = 47 / 64 →
  x = Real.sqrt 63.75 := by
  sorry

end triangle_side_length_l4062_406217


namespace mean_proportional_81_100_l4062_406244

theorem mean_proportional_81_100 : 
  Real.sqrt (81 * 100) = 90 := by sorry

end mean_proportional_81_100_l4062_406244


namespace arithmetic_sequence_sum_l4062_406213

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end arithmetic_sequence_sum_l4062_406213


namespace right_triangle_cos_a_l4062_406239

theorem right_triangle_cos_a (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) : 
  B = π/2 → 3 * Real.tan A = 4 * Real.sin A → Real.cos A = 3/4 := by
  sorry

end right_triangle_cos_a_l4062_406239


namespace quadratic_inequality_l4062_406210

theorem quadratic_inequality (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end quadratic_inequality_l4062_406210


namespace triangle_area_proof_l4062_406211

open Real

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  -- a * tan(B) = 2b * sin(A)
  a * tan B = 2 * b * sin A →
  -- b = √3
  b = sqrt 3 →
  -- A = 5π/12
  A = 5 * π / 12 →
  -- The area of triangle ABC is (3 + √3) / 4
  (1 / 2) * b * c * sin A = (3 + sqrt 3) / 4 := by
  sorry

end triangle_area_proof_l4062_406211


namespace vote_combinations_l4062_406201

/-- The number of ways to select k items from n distinct items with replacement,
    where order doesn't matter (combinations with repetition) -/
def combinations_with_repetition (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

/-- Theorem: There are 6 ways to select 2 items from 3 items with replacement,
    where order doesn't matter -/
theorem vote_combinations : combinations_with_repetition 3 2 = 6 := by
  sorry

end vote_combinations_l4062_406201


namespace cloth_profit_per_meter_l4062_406282

/-- Calculates the profit per meter of cloth given the total selling price, 
    number of meters sold, and cost price per meter. -/
def profit_per_meter (selling_price total_meters cost_price_per_meter : ℕ) : ℕ :=
  ((selling_price - (cost_price_per_meter * total_meters)) / total_meters)

/-- Proves that the profit per meter of cloth is 15 rupees given the specified conditions. -/
theorem cloth_profit_per_meter :
  profit_per_meter 8500 85 85 = 15 := by
  sorry


end cloth_profit_per_meter_l4062_406282


namespace restaurant_budget_allocation_l4062_406294

/-- Given a restaurant's budget allocation, prove that the fraction of
    remaining budget spent on food and beverages is 1/4. -/
theorem restaurant_budget_allocation (B : ℝ) (B_pos : B > 0) :
  let rent : ℝ := (1 / 4) * B
  let remaining : ℝ := B - rent
  let food_and_beverages : ℝ := 0.1875 * B
  food_and_beverages / remaining = 1 / 4 := by
sorry

end restaurant_budget_allocation_l4062_406294


namespace bowling_team_average_weight_l4062_406293

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 78) :
  (original_players * (original_players + 2) * new_average_weight - 
   (original_players * new_player1_weight + original_players * new_player2_weight)) / 
  (original_players * original_players) = 76 :=
by sorry

end bowling_team_average_weight_l4062_406293


namespace inverse_proportionality_l4062_406218

/-- Proves that given α is inversely proportional to β, and α = 4 when β = 12, then α = -16 when β = -3 -/
theorem inverse_proportionality (α β : ℝ → ℝ) (k : ℝ) : 
  (∀ x, α x * β x = k) →  -- α is inversely proportional to β
  (α 12 = 4) →            -- α = 4 when β = 12
  (β 12 = 12) →           -- ensuring β 12 is indeed 12
  (β (-3) = -3) →         -- ensuring β (-3) is indeed -3
  (α (-3) = -16) :=       -- α = -16 when β = -3
by
  sorry


end inverse_proportionality_l4062_406218


namespace p_range_l4062_406223

/-- The function p(x) defined for x ≥ 0 -/
def p (x : ℝ) : ℝ := x^4 + 8*x^2 + 16

/-- The range of p(x) is [16, ∞) -/
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 16 := by sorry

end p_range_l4062_406223


namespace remaining_length_is_24_l4062_406220

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  sides : List ℝ
  perpendicular : Bool

/-- Function to calculate the total length of remaining segments after removal -/
def remainingLength (figure : PerpendicularFigure) (removedSides : ℕ) : ℝ :=
  sorry

/-- Theorem stating the total length of remaining segments is 24 units -/
theorem remaining_length_is_24 (figure : PerpendicularFigure) 
  (h1 : figure.sides = [10, 3, 8, 1, 1, 5]) 
  (h2 : figure.perpendicular = true) 
  (h3 : removedSides = 6) : 
  remainingLength figure removedSides = 24 :=
sorry

end remaining_length_is_24_l4062_406220


namespace prime_square_mod_180_l4062_406277

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (s : Finset ℕ), (∀ x ∈ s, x < 180) ∧ (Finset.card s = 2) ∧
  (∀ q : ℕ, Nat.Prime q → q > 5 → (q^2 % 180) ∈ s) :=
sorry

end prime_square_mod_180_l4062_406277


namespace pirate_treasure_distribution_l4062_406287

/-- The number of coins in the final round of distribution -/
def x : ℕ := sorry

/-- The sum of coins Pete gives himself in each round -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution :
  -- Paul ends up with x coins
  -- Pete ends up with 5x coins
  -- Pete's coins follow the pattern 1 + 2 + 3 + ... + x
  -- The total number of coins is 54
  x + 5 * x = 54 ∧ petes_coins x = 5 * x := by sorry

end pirate_treasure_distribution_l4062_406287


namespace total_games_won_l4062_406243

-- Define the number of games won by Betsy
def betsy_games : ℕ := 5

-- Define Helen's games in terms of Betsy's
def helen_games : ℕ := 2 * betsy_games

-- Define Susan's games in terms of Betsy's
def susan_games : ℕ := 3 * betsy_games

-- Theorem to prove the total number of games won
theorem total_games_won : betsy_games + helen_games + susan_games = 30 := by
  sorry

end total_games_won_l4062_406243


namespace twelfth_day_is_monday_l4062_406271

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific conditions -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount ∈ [28, 29, 30, 31]

/-- Function to determine the day of week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end twelfth_day_is_monday_l4062_406271


namespace monochromatic_rectangle_exists_l4062_406272

/-- A color type representing red, green, or blue -/
inductive Color
  | Red
  | Green
  | Blue

/-- A type representing a 4 x 82 grid where each point is colored -/
def ColoredGrid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (grid : ColoredGrid) (i j p q : Nat) : Prop :=
  i < j ∧ p < q ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨i, by sorry⟩ ⟨q, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨p, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨q, by sorry⟩

/-- The main theorem stating that any 4 x 82 grid colored with three colors
    contains a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (grid : ColoredGrid) :
  ∃ i j p q, isMonochromaticRectangle grid i j p q :=
sorry

end monochromatic_rectangle_exists_l4062_406272


namespace teenager_age_problem_l4062_406237

theorem teenager_age_problem (a b : ℕ) (h1 : a > b) (h2 : a^2 - b^2 = 4*(a + b)) (h3 : a + b = 8*(a - b)) : a = 18 := by
  sorry

end teenager_age_problem_l4062_406237


namespace smallest_geometric_sequence_number_l4062_406247

/-- A function that checks if a three-digit number's digits form a geometric sequence -/
def is_geometric_sequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c

/-- A function that checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_geometric_sequence_number :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 →
    is_geometric_sequence n ∧ has_distinct_digits n →
    124 ≤ n :=
by sorry

end smallest_geometric_sequence_number_l4062_406247


namespace complement_of_M_l4062_406208

def M : Set ℝ := {x | (1 + x) / (1 - x) > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x | x ≤ -1 ∨ x ≥ 1} := by sorry

end complement_of_M_l4062_406208


namespace heather_start_time_l4062_406231

/-- Proves that Heather started her journey 24 minutes after Stacy given the problem conditions -/
theorem heather_start_time (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 10 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 3.4545454545454546 →
  (total_distance - heather_distance) / stacy_speed - heather_distance / heather_speed = 0.4 :=
by sorry

end heather_start_time_l4062_406231


namespace three_same_one_different_probability_l4062_406296

/-- The probability of a child being born a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The number of possible combinations for having three children of one sex and one of the opposite sex -/
def num_combinations : ℕ := 8

/-- The probability of having three children of one sex and one of the opposite sex in a family of four children -/
theorem three_same_one_different_probability :
  (child_probability ^ num_children) * num_combinations = 1 / 2 := by
  sorry

end three_same_one_different_probability_l4062_406296


namespace circle_P_radius_l4062_406281

-- Define the circles and points
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0
def point_M : ℝ × ℝ := (-1, 0)

-- Define the curve τ
def curve_τ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 1) ∧ k > 0

-- Define the theorem
theorem circle_P_radius : 
  ∃ (x_P y_P r : ℝ),
  -- Circle P passes through M
  (x_P + 1)^2 + y_P^2 = r^2 ∧
  -- Circle P is internally tangent to N
  ∃ (x_N y_N : ℝ), circle_N x_N y_N ∧ ((x_P - x_N)^2 + (y_P - y_N)^2 = (4 - r)^2) ∧
  -- Center of P is on curve τ
  curve_τ x_P y_P ∧
  -- Line l is tangent to P and intersects τ
  ∃ (k x_A y_A x_B y_B : ℝ),
    line_l k x_A y_A ∧ line_l k x_B y_B ∧
    curve_τ x_A y_A ∧ curve_τ x_B y_B ∧
    -- Q is midpoint of AB with abscissa -4/13
    (x_A + x_B)/2 = -4/13 ∧
  -- One possible radius of P is 6/5
  r = 6/5 :=
sorry

end circle_P_radius_l4062_406281


namespace smallest_with_four_odd_eight_even_divisors_l4062_406254

/-- Count of positive odd integer divisors of n -/
def oddDivisorCount (n : ℕ+) : ℕ := sorry

/-- Count of positive even integer divisors of n -/
def evenDivisorCount (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having exactly four positive odd integer divisors and eight positive even integer divisors -/
def hasFourOddEightEvenDivisors (n : ℕ+) : Prop :=
  oddDivisorCount n = 4 ∧ evenDivisorCount n = 8

theorem smallest_with_four_odd_eight_even_divisors :
  ∃ (n : ℕ+), hasFourOddEightEvenDivisors n ∧
  ∀ (m : ℕ+), hasFourOddEightEvenDivisors m → n ≤ m :=
by
  use 60
  sorry

end smallest_with_four_odd_eight_even_divisors_l4062_406254


namespace ted_work_time_l4062_406298

theorem ted_work_time (julie_rate ted_rate : ℚ) (julie_finish_time : ℚ) : 
  julie_rate = 1/10 →
  ted_rate = 1/8 →
  julie_finish_time = 999999999999999799 / 1000000000000000000 →
  ∃ t : ℚ, t = 4 ∧ (julie_rate + ted_rate) * t + julie_rate * julie_finish_time = 1 :=
by sorry

end ted_work_time_l4062_406298


namespace license_plate_theorem_l4062_406285

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let digit_size : ℕ := 10
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 2

  let choose_repeated_letter := alphabet_size
  let choose_distinct_letters := Nat.choose (alphabet_size - 1) 2
  let place_repeated_letter := Nat.choose letter_positions 2
  let arrange_nonrepeated_letters := 2

  let letter_combinations := 
    choose_repeated_letter * choose_distinct_letters * place_repeated_letter * arrange_nonrepeated_letters

  let digit_combinations := digit_size ^ digit_positions

  letter_combinations * digit_combinations

theorem license_plate_theorem : license_plate_combinations = 936000 := by
  sorry

end license_plate_theorem_l4062_406285


namespace subtract_fractions_l4062_406224

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end subtract_fractions_l4062_406224


namespace fruit_purchase_total_l4062_406235

/-- The total amount paid for a fruit purchase given the quantity and rate per kg for two types of fruits -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that the total amount paid for 8 kg of grapes at 70 per kg and 9 kg of mangoes at 45 per kg is 965 -/
theorem fruit_purchase_total :
  total_amount_paid 8 70 9 45 = 965 := by
  sorry

end fruit_purchase_total_l4062_406235


namespace sin_minus_cos_value_l4062_406253

theorem sin_minus_cos_value (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin θ + Real.cos θ = 5 / 4) : 
  Real.sin θ - Real.cos θ = Real.sqrt 7 / 4 := by
  sorry

end sin_minus_cos_value_l4062_406253


namespace vovochka_max_candies_l4062_406251

/-- Represents the problem of distributing candies among classmates -/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep -/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates - 1) * (cd.min_group_candies / cd.min_group_size)

/-- Theorem stating the maximum number of candies Vovochka can keep -/
theorem vovochka_max_candies :
  let cd : CandyDistribution := {
    total_candies := 200,
    num_classmates := 25,
    min_group_size := 16,
    min_group_candies := 100
  }
  max_candies_kept cd = 37 := by
  sorry

end vovochka_max_candies_l4062_406251


namespace complement_of_A_in_U_l4062_406266

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A (domain of f(x))
def A : Set ℝ := {x | x ≥ Real.exp 1}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = Set.Ioo 0 (Real.exp 1) := by sorry

end complement_of_A_in_U_l4062_406266


namespace line_AB_equation_min_area_and_B_coords_l4062_406286

-- Define the line l: y = 4x
def line_l (x y : ℝ) : Prop := y = 4 * x

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define that point A is in the first quadrant and lies on line l
def point_A_condition (A : ℝ × ℝ) : Prop :=
  A.1 > 0 ∧ A.2 > 0 ∧ line_l A.1 A.2

-- Define that line PA intersects the positive half of the x-axis at point B
def point_B_condition (A B : ℝ × ℝ) : Prop :=
  B.2 = 0 ∧ B.1 > 0 ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  B.1 = t * A.1 + (1 - t) * point_P.1 ∧
  B.2 = t * A.2 + (1 - t) * point_P.2

-- Theorem for part (1)
theorem line_AB_equation (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) 
  (h3 : (A.2 - point_P.2) * (B.1 - point_P.1) = -(A.1 - point_P.1) * (B.2 - point_P.2)) :
  ∃ k c : ℝ, k = -3/2 ∧ c = 13 ∧ ∀ x y : ℝ, y = k * x + c ↔ 3 * x + 2 * y - 26 = 0 :=
sorry

-- Theorem for part (2)
theorem min_area_and_B_coords (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) :
  ∃ S_min : ℝ, S_min = 40 ∧
  (∀ A' B' : ℝ × ℝ, point_A_condition A' → point_B_condition A' B' →
    1/2 * A'.1 * B'.2 - 1/2 * A'.2 * B'.1 ≥ S_min) ∧
  (∃ A_min B_min : ℝ × ℝ, 
    point_A_condition A_min ∧ 
    point_B_condition A_min B_min ∧
    1/2 * A_min.1 * B_min.2 - 1/2 * A_min.2 * B_min.1 = S_min ∧
    B_min = (10, 0)) :=
sorry

end line_AB_equation_min_area_and_B_coords_l4062_406286


namespace quadratic_equation_from_means_l4062_406265

theorem quadratic_equation_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → α * β = 144 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = α ∨ x = β) := by
sorry

end quadratic_equation_from_means_l4062_406265


namespace roof_area_is_400_l4062_406284

/-- Represents a rectangular roof with given properties -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  length_is_triple_width : length = 3 * width
  length_width_difference : length - width = 30

/-- The area of a rectangular roof -/
def roof_area (roof : RectangularRoof) : ℝ :=
  roof.length * roof.width

/-- Theorem stating that a roof with the given properties has an area of 400 square feet -/
theorem roof_area_is_400 (roof : RectangularRoof) : roof_area roof = 400 := by
  sorry

end roof_area_is_400_l4062_406284


namespace cubic_root_sum_l4062_406278

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - 7*α^2 + 11*α - 13 = 0) →
  (β^3 - 7*β^2 + 11*β - 13 = 0) →
  (γ^3 - 7*γ^2 + 11*γ - 13 = 0) →
  (α*β/γ + β*γ/α + γ*α/β = -61/13) :=
by sorry

end cubic_root_sum_l4062_406278


namespace sum_of_cubes_divisible_by_nine_l4062_406212

theorem sum_of_cubes_divisible_by_nine (n : ℤ) : 
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k := by
sorry

end sum_of_cubes_divisible_by_nine_l4062_406212


namespace area_enclosed_by_four_circles_l4062_406228

/-- The area of the figure enclosed by four identical circles inscribed in a larger circle -/
theorem area_enclosed_by_four_circles (R : ℝ) : 
  ∃ (area : ℝ), 
    area = R^2 * (4 - π) * (3 - 2 * Real.sqrt 2) ∧
    (∀ (r : ℝ), 
      r = R * (Real.sqrt 2 - 1) →
      area = 4 * r^2 - π * r^2 ∧
      (∃ (O₁ O₂ O₃ O₄ : ℝ × ℝ),
        -- Four circles with centers O₁, O₂, O₃, O₄ and radius r
        -- Each touching two others and the larger circle with radius R
        True)) :=
by sorry

end area_enclosed_by_four_circles_l4062_406228


namespace average_salary_non_officers_l4062_406275

/-- Proof of the average salary of non-officers in an office --/
theorem average_salary_non_officers
  (total_avg : ℝ)
  (officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_officer_avg : officer_avg = 430)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 465) :
  let non_officer_avg := (((officer_count + non_officer_count) * total_avg) - (officer_count * officer_avg)) / non_officer_count
  non_officer_avg = 110 := by
sorry


end average_salary_non_officers_l4062_406275


namespace maggie_goldfish_fraction_l4062_406280

theorem maggie_goldfish_fraction (total : ℕ) (caught_fraction : ℚ) (remaining : ℕ) :
  total = 100 →
  caught_fraction = 3 / 5 →
  remaining = 20 →
  (total : ℚ) / 2 = (caught_fraction * ((caught_fraction * (total : ℚ) + remaining) / caught_fraction) + remaining) / caught_fraction :=
by sorry

end maggie_goldfish_fraction_l4062_406280


namespace square_side_length_from_circle_area_l4062_406250

/-- Given a square from which a circle is described, if the area of the circle is 78.53981633974483 square inches, then the side length of the square is 10 inches. -/
theorem square_side_length_from_circle_area (circle_area : ℝ) (square_side : ℝ) : 
  circle_area = 78.53981633974483 →
  circle_area = Real.pi * (square_side / 2)^2 →
  square_side = 10 := by
  sorry

end square_side_length_from_circle_area_l4062_406250


namespace inequality_proof_l4062_406290

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  3 - Real.sqrt 3 + x^2 / y + y^2 / z + z^2 / x ≥ (x + y + z)^2 := by
  sorry

end inequality_proof_l4062_406290


namespace sum_last_three_coefficients_eq_21_l4062_406226

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function that calculates the sum of the last three coefficients
def sum_last_three_coefficients (a : ℝ) : ℝ :=
  binomial 8 0 * 1 + binomial 8 1 * (-1) + binomial 8 2 * 1

-- Theorem statement
theorem sum_last_three_coefficients_eq_21 :
  ∀ a : ℝ, sum_last_three_coefficients a = 21 := by sorry

end sum_last_three_coefficients_eq_21_l4062_406226


namespace quadratic_no_real_roots_l4062_406236

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 3 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l4062_406236


namespace sum_of_tens_equal_hundred_to_ten_l4062_406291

theorem sum_of_tens_equal_hundred_to_ten (n : ℕ) : 
  (n * 10 = 100^10) → (n = 10^19) := by
  sorry

end sum_of_tens_equal_hundred_to_ten_l4062_406291


namespace midpoint_locus_l4062_406263

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

/-- The point M is the midpoint of OA -/
def is_midpoint (x y : ℝ) : Prop := ∃ (ax ay : ℝ), circle_equation ax ay ∧ x = ax/2 ∧ y = ay/2

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem midpoint_locus : ∀ (x y : ℝ), is_midpoint x y → locus_equation x y :=
by sorry

end midpoint_locus_l4062_406263


namespace toms_dog_age_l4062_406246

/-- Given the ages of Tom's pets, prove the age of his dog. -/
theorem toms_dog_age (cat_age : ℕ) (rabbit_age : ℕ) (dog_age : ℕ)
  (h1 : cat_age = 8)
  (h2 : rabbit_age = cat_age / 2)
  (h3 : dog_age = rabbit_age * 3) :
  dog_age = 12 :=
by sorry

end toms_dog_age_l4062_406246


namespace f_2023_equals_2_l4062_406261

theorem f_2023_equals_2 (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2)) :
  f 2023 = 2 := by
  sorry

end f_2023_equals_2_l4062_406261


namespace abc_ratio_theorem_l4062_406221

theorem abc_ratio_theorem (a b c : ℚ) 
  (h : (|a|/a) + (|b|/b) + (|c|/c) = 1) : 
  a * b * c / |a * b * c| = -1 := by
sorry

end abc_ratio_theorem_l4062_406221


namespace M_intersect_N_eq_M_l4062_406288

/-- Set M defined as {x | x^2 - x < 0} -/
def M : Set ℝ := {x | x^2 - x < 0}

/-- Set N defined as {x | |x| < 2} -/
def N : Set ℝ := {x | |x| < 2}

/-- Theorem stating that the intersection of M and N equals M -/
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l4062_406288


namespace tangent_line_slope_l4062_406252

/-- The slope of a line tangent to a circle --/
theorem tangent_line_slope (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x - 2) + 2 → x^2 + y^2 - 2*x - 2*y = 0 → 
   ∃! x y : ℝ, y = k * (x - 2) + 2 ∧ x^2 + y^2 - 2*x - 2*y = 0) → 
  k = -1 := by
sorry

end tangent_line_slope_l4062_406252


namespace g_of_seven_l4062_406241

/-- Given a function g(x) = (2x + 3) / (4x - 5), prove that g(7) = 17/23 -/
theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x, g x = (2 * x + 3) / (4 * x - 5)) : 
  g 7 = 17 / 23 := by
  sorry

end g_of_seven_l4062_406241


namespace min_abc_value_l4062_406232

/-- Given prime numbers a, b, c where a^5 divides (b^2 - c) and b + c is a perfect square,
    the minimum value of abc is 1958. -/
theorem min_abc_value (a b c : ℕ) : Prime a → Prime b → Prime c →
  (a^5 ∣ (b^2 - c)) → ∃ (n : ℕ), b + c = n^2 → (∀ x y z : ℕ, Prime x → Prime y → Prime z →
  (x^5 ∣ (y^2 - z)) → ∃ (m : ℕ), y + z = m^2 → x*y*z ≥ a*b*c) → a*b*c = 1958 := by
  sorry

end min_abc_value_l4062_406232


namespace particle_hit_probability_l4062_406249

/-- Probability of hitting (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

/-- The particle starts at (5,5) -/
def start_pos : ℕ × ℕ := (5, 5)

/-- The probability of hitting (0,0) is m/3^n -/
def hit_prob : ℚ := 1 / 3^5

theorem particle_hit_probability :
  P start_pos.1 start_pos.2 = hit_prob :=
sorry

end particle_hit_probability_l4062_406249


namespace valid_medium_triangle_counts_l4062_406234

/-- Represents the side length of the original equilateral triangle -/
def originalSideLength : ℕ := 10

/-- Represents the side length of the smallest equilateral triangles -/
def smallestSideLength : ℕ := 1

/-- Represents the side length of the medium equilateral triangles -/
def mediumSideLength : ℕ := 2

/-- Represents the total number of shapes (triangles and parallelograms) -/
def totalShapes : ℕ := 25

/-- Predicate to check if a number is a valid count of medium triangles -/
def isValidMediumTriangleCount (m : ℕ) : Prop :=
  m % 2 = 1 ∧ 5 ≤ m ∧ m ≤ 25

/-- The set of all valid counts of medium triangles -/
def validMediumTriangleCounts : Set ℕ :=
  {m | isValidMediumTriangleCount m}

/-- Theorem stating the properties of valid medium triangle counts -/
theorem valid_medium_triangle_counts :
  validMediumTriangleCounts = {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25} :=
sorry

end valid_medium_triangle_counts_l4062_406234


namespace smallest_solution_of_equation_l4062_406209

theorem smallest_solution_of_equation (x : ℝ) :
  x = (5 - Real.sqrt 241) / 6 →
  (3 * x / (x - 3) + (3 * x^2 - 36) / x = 14) ∧
  ∀ y : ℝ, (3 * y / (y - 3) + (3 * y^2 - 36) / y = 14) → y ≥ x :=
by sorry

end smallest_solution_of_equation_l4062_406209


namespace compare_x_powers_l4062_406256

theorem compare_x_powers (x : ℝ) (h : 0 < x ∧ x < 1) : x^2 < Real.sqrt x ∧ Real.sqrt x < x ∧ x < 1/x := by
  sorry

end compare_x_powers_l4062_406256


namespace intersection_of_A_and_B_l4062_406215

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def B : Set ℤ := {x | ∃ k : ℕ, x = 2 * k + 1 ∧ k < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end intersection_of_A_and_B_l4062_406215


namespace A_intersect_B_l4062_406255

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by
  sorry

end A_intersect_B_l4062_406255
