import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_numbers_square_sum_l527_52782

theorem consecutive_numbers_square_sum (a b c : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 27) → a^2 + b^2 + c^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_square_sum_l527_52782


namespace NUMINAMATH_CALUDE_inequality_solution_set_l527_52789

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + a) > 0 ↔
    (a < -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a > -1 ∧ (x < -a ∨ x > 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l527_52789


namespace NUMINAMATH_CALUDE_solution_set_empty_l527_52726

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Define the constants a and b
def a : ℝ := 2
def b : ℝ := -3

-- State the theorem
theorem solution_set_empty :
  ∀ x : ℝ, f a (a * x + b) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_empty_l527_52726


namespace NUMINAMATH_CALUDE_ratio_sum_to_y_l527_52733

theorem ratio_sum_to_y (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) :
  (x + y) / y = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_y_l527_52733


namespace NUMINAMATH_CALUDE_orange_juice_price_l527_52745

def initial_money : ℕ := 86
def bread_price : ℕ := 3
def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def money_left : ℕ := 59

theorem orange_juice_price :
  ∃ (juice_price : ℕ),
    initial_money - (bread_price * bread_quantity + juice_price * juice_quantity) = money_left ∧
    juice_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_price_l527_52745


namespace NUMINAMATH_CALUDE_coat_price_reduction_l527_52795

theorem coat_price_reduction (original_price reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction = 250) :
  (reduction / original_price) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l527_52795


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l527_52705

theorem min_value_quadratic_form (a b : ℝ) (h : 4 ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 9) :
  2 ≤ a^2 - a*b + b^2 ∧ ∃ (x y : ℝ), 4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧ x^2 - x*y + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l527_52705


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l527_52707

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l527_52707


namespace NUMINAMATH_CALUDE_jie_is_tallest_l527_52742

-- Define a type for the people
inductive Person : Type
  | Igor : Person
  | Jie : Person
  | Faye : Person
  | Goa : Person
  | Han : Person

-- Define a relation for "taller than"
def taller_than : Person → Person → Prop := sorry

-- Define the conditions
axiom igor_shorter_jie : taller_than Person.Jie Person.Igor
axiom faye_taller_goa : taller_than Person.Faye Person.Goa
axiom jie_taller_faye : taller_than Person.Jie Person.Faye
axiom han_shorter_goa : taller_than Person.Goa Person.Han

-- Define what it means to be the tallest
def is_tallest (p : Person) : Prop :=
  ∀ q : Person, p ≠ q → taller_than p q

-- State the theorem
theorem jie_is_tallest : is_tallest Person.Jie := by
  sorry

end NUMINAMATH_CALUDE_jie_is_tallest_l527_52742


namespace NUMINAMATH_CALUDE_complex_expression_equality_l527_52731

theorem complex_expression_equality : 
  ∀ (z₁ z₂ : ℂ), 
    z₁ = 2 - I → 
    z₂ = -I → 
    z₁ / z₂ + Complex.abs z₂ = 2 + 2*I := by
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l527_52731


namespace NUMINAMATH_CALUDE_bowtie_equation_l527_52777

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1)))))

-- State the theorem
theorem bowtie_equation (h : ℝ) :
  bowtie 5 h = 8 → h = 9 - Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_l527_52777


namespace NUMINAMATH_CALUDE_exam_score_calculation_l527_52735

/-- Proves that the number of marks awarded for a correct answer is 4 in the given exam scenario -/
theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_score : ℕ) 
  (h1 : total_questions = 150)
  (h2 : correct_answers = 120)
  (h3 : total_score = 420)
  (h4 : ∀ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score → x = 4) :
  ∃ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l527_52735


namespace NUMINAMATH_CALUDE_infinite_capacitor_chain_effective_capacitance_l527_52764

/-- Given an infinitely long chain of capacitors, each with capacitance C,
    the effective capacitance Ce between any two adjacent points
    is equal to ((1 + √3) * C) / 2. -/
theorem infinite_capacitor_chain_effective_capacitance (C : ℝ) (Ce : ℝ) 
  (h1 : C > 0) -- Capacitance is always positive
  (h2 : Ce = C + Ce / (2 + Ce / C)) -- Relationship derived from the infinite chain
  : Ce = ((1 + Real.sqrt 3) * C) / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_capacitor_chain_effective_capacitance_l527_52764


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l527_52722

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l527_52722


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l527_52766

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  (z₁.re = 2 ∧ z₁.im = 1) →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l527_52766


namespace NUMINAMATH_CALUDE_prob_more_ones_than_eights_l527_52760

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling more 1's than 8's when rolling five fair eight-sided dice -/
def probMoreOnesThanEights : ℚ := 14026 / 32768

/-- Theorem stating that the probability of rolling more 1's than 8's is correct -/
theorem prob_more_ones_than_eights :
  let totalOutcomes : ℕ := numSides ^ numDice
  let probEqualOnesAndEights : ℚ := 4716 / totalOutcomes
  probMoreOnesThanEights = (1 - probEqualOnesAndEights) / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_eights_l527_52760


namespace NUMINAMATH_CALUDE_N_subset_M_l527_52768

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x - 2 = 0}

theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l527_52768


namespace NUMINAMATH_CALUDE_model_evaluation_criteria_l527_52771

-- Define the concept of a model
def Model : Type := ℝ → ℝ

-- Define the concept of residuals
def Residuals (m : Model) (data : Set (ℝ × ℝ)) : Set ℝ := sorry

-- Define the concept of residual plot distribution
def EvenlyDistributedInHorizontalBand (r : Set ℝ) : Prop := sorry

-- Define the sum of squared residuals
def SumSquaredResiduals (r : Set ℝ) : ℝ := sorry

-- Define the concept of model appropriateness
def ModelAppropriate (m : Model) (data : Set (ℝ × ℝ)) : Prop := 
  EvenlyDistributedInHorizontalBand (Residuals m data)

-- Define the concept of better fitting model
def BetterFittingModel (m1 m2 : Model) (data : Set (ℝ × ℝ)) : Prop :=
  SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)

-- Theorem statement
theorem model_evaluation_criteria 
  (m : Model) (data : Set (ℝ × ℝ)) (m1 m2 : Model) :
  (ModelAppropriate m data ↔ 
    EvenlyDistributedInHorizontalBand (Residuals m data)) ∧
  (BetterFittingModel m1 m2 data ↔ 
    SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)) :=
by sorry

end NUMINAMATH_CALUDE_model_evaluation_criteria_l527_52771


namespace NUMINAMATH_CALUDE_sandro_children_l527_52725

/-- The number of sons Sandro has -/
def num_sons : ℕ := 3

/-- The ratio of daughters to sons -/
def daughter_son_ratio : ℕ := 6

/-- The number of daughters Sandro has -/
def num_daughters : ℕ := daughter_son_ratio * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := num_daughters + num_sons

theorem sandro_children : total_children = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l527_52725


namespace NUMINAMATH_CALUDE_atomic_weight_Ba_value_l527_52772

/-- The atomic weight of Fluorine (F) -/
def atomic_weight_F : ℝ := 19

/-- The molecular weight of the compound BaF₂ -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of F atoms in the compound -/
def num_F_atoms : ℕ := 2

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_BaF2 - num_F_atoms * atomic_weight_F

theorem atomic_weight_Ba_value : atomic_weight_Ba = 137 := by sorry

end NUMINAMATH_CALUDE_atomic_weight_Ba_value_l527_52772


namespace NUMINAMATH_CALUDE_texas_migration_l527_52739

/-- The number of people moving to Texas in four days -/
def people_moving : ℕ := 3600

/-- The number of days -/
def num_days : ℕ := 4

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def avg_people_per_hour : ℚ :=
  people_moving / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem texas_migration :
  round_to_nearest avg_people_per_hour = 38 := by
  sorry

end NUMINAMATH_CALUDE_texas_migration_l527_52739


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l527_52787

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l527_52787


namespace NUMINAMATH_CALUDE_first_part_speed_l527_52767

/-- Represents a train journey with two parts -/
structure TrainJourney where
  x : ℝ  -- distance of first part in km
  V : ℝ  -- speed of first part in kmph

/-- Theorem stating the speed of the first part of the journey -/
theorem first_part_speed (j : TrainJourney) (h1 : j.x > 0) 
    (h2 : (j.x / j.V) + (2 * j.x / 20) = (3 * j.x) / 24) : j.V = 40 := by
  sorry

#check first_part_speed

end NUMINAMATH_CALUDE_first_part_speed_l527_52767


namespace NUMINAMATH_CALUDE_age_difference_is_28_l527_52721

/-- The age difference between a man and his son -/
def ageDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

/-- Prove that the age difference between a man and his son is 28 years -/
theorem age_difference_is_28 :
  ∃ (sonAge manAge : ℕ),
    sonAge = 26 ∧
    manAge + 2 = 2 * (sonAge + 2) ∧
    ageDifference sonAge manAge = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_28_l527_52721


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l527_52774

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income from tips for a waitress -/
def fractionFromTips (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: If tips are 11/4 of salary, then 11/15 of income is from tips -/
theorem tips_fraction_of_income 
  (income : WaitressIncome) 
  (h : income.tips = (11 / 4) * income.salary) : 
  fractionFromTips income = 11 / 15 := by
  sorry

#check tips_fraction_of_income

end NUMINAMATH_CALUDE_tips_fraction_of_income_l527_52774


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l527_52719

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ x₁^2 - 6*x₁ + 8 = 0 ∧ x₂^2 - 6*x₂ + 8 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l527_52719


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l527_52793

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 11 * p - 20 = 0) → 
  (3 * q^2 + 11 * q - 20 = 0) → 
  (5 * p - 4) * (3 * q - 2) = -89/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l527_52793


namespace NUMINAMATH_CALUDE_bill_selling_price_l527_52706

theorem bill_selling_price (original_purchase_price : ℝ) : 
  let original_selling_price := 1.1 * original_purchase_price
  let new_selling_price := 1.17 * original_purchase_price
  new_selling_price = original_selling_price + 28 →
  original_selling_price = 440 := by
sorry

end NUMINAMATH_CALUDE_bill_selling_price_l527_52706


namespace NUMINAMATH_CALUDE_combined_length_legs_arms_l527_52701

/-- Calculates the combined length of legs and arms for two people given their heights and body proportions -/
theorem combined_length_legs_arms 
  (aisha_height : ℝ) 
  (benjamin_height : ℝ) 
  (aisha_legs_ratio : ℝ) 
  (aisha_arms_ratio : ℝ) 
  (benjamin_legs_ratio : ℝ) 
  (benjamin_arms_ratio : ℝ) 
  (h1 : aisha_height = 174) 
  (h2 : benjamin_height = 190) 
  (h3 : aisha_legs_ratio = 1/3) 
  (h4 : aisha_arms_ratio = 1/6) 
  (h5 : benjamin_legs_ratio = 3/7) 
  (h6 : benjamin_arms_ratio = 1/4) : 
  (aisha_legs_ratio * aisha_height + aisha_arms_ratio * aisha_height + 
   benjamin_legs_ratio * benjamin_height + benjamin_arms_ratio * benjamin_height) = 215.93 := by
  sorry

end NUMINAMATH_CALUDE_combined_length_legs_arms_l527_52701


namespace NUMINAMATH_CALUDE_rogers_retirement_experience_l527_52779

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- The theorem to be proved -/
theorem rogers_retirement_experience (e : Experience) :
  satisfies_conditions e → e.roger + 8 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_rogers_retirement_experience_l527_52779


namespace NUMINAMATH_CALUDE_cube_volume_problem_l527_52788

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →  -- Ensuring the side length is positive
  (a + 2) * a * (a - 2) = a^3 - 8 → 
  a^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l527_52788


namespace NUMINAMATH_CALUDE_decimal_period_equals_number_period_l527_52754

/-- The length of the repeating period in the decimal representation of a fraction -/
def decimal_period_length (n p : ℕ) : ℕ := sorry

/-- The length of the period of a number in decimal representation -/
def number_period_length (p : ℕ) : ℕ := sorry

/-- Theorem stating that for a natural number n and a prime number p, 
    where n ≤ p - 1, the length of the repeating period in the decimal 
    representation of n/p is equal to the length of the period of p -/
theorem decimal_period_equals_number_period (n p : ℕ) 
  (h_prime : Nat.Prime p) (h_n_le_p_minus_one : n ≤ p - 1) : 
  decimal_period_length n p = number_period_length p := by
  sorry

end NUMINAMATH_CALUDE_decimal_period_equals_number_period_l527_52754


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l527_52773

/-- Proves that Danny found 1 more bottle cap than he threw away. -/
theorem danny_bottle_caps (found : ℕ) (thrown_away : ℕ) (current : ℕ)
  (h1 : found = 36)
  (h2 : thrown_away = 35)
  (h3 : current = 22)
  : found - thrown_away = 1 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l527_52773


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l527_52757

theorem square_sum_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a - 1/b - 1/(a+b) = 0) : (b/a + a/b)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l527_52757


namespace NUMINAMATH_CALUDE_congruence_mod_nine_l527_52753

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_congruence_mod_nine_l527_52753


namespace NUMINAMATH_CALUDE_truck_mileage_l527_52769

/-- Given a truck that travels 240 miles on 5 gallons of gas, 
    prove that it can travel 336 miles on 7 gallons of gas. -/
theorem truck_mileage (miles_on_five : ℝ) (gallons_five : ℝ) (gallons_seven : ℝ) 
  (h1 : miles_on_five = 240)
  (h2 : gallons_five = 5)
  (h3 : gallons_seven = 7) :
  (miles_on_five / gallons_five) * gallons_seven = 336 := by
sorry

end NUMINAMATH_CALUDE_truck_mileage_l527_52769


namespace NUMINAMATH_CALUDE_odd_expressions_l527_52723

theorem odd_expressions (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5*p^2 + 2*q^2) ∧ Odd (p^2 + p*q + q^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_expressions_l527_52723


namespace NUMINAMATH_CALUDE_bottom_sphere_radius_l527_52718

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents three stacked spheres in a cone -/
structure StackedSpheres where
  cone : Cone
  bottomSphere : Sphere
  middleSphere : Sphere
  topSphere : Sphere

/-- The condition for the spheres to fit in the cone -/
def spheresFitInCone (s : StackedSpheres) : Prop :=
  s.bottomSphere.radius + s.middleSphere.radius + s.topSphere.radius ≤ s.cone.height

/-- The theorem stating the radius of the bottom sphere -/
theorem bottom_sphere_radius (s : StackedSpheres) 
  (h1 : s.cone.baseRadius = 8)
  (h2 : s.cone.height = 18)
  (h3 : s.middleSphere.radius = 2 * s.bottomSphere.radius)
  (h4 : s.topSphere.radius = 3 * s.bottomSphere.radius)
  (h5 : spheresFitInCone s) :
  s.bottomSphere.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottom_sphere_radius_l527_52718


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l527_52747

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  a < Real.sqrt 7 ∧ Real.sqrt 7 < b ∧ b = a + 1 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l527_52747


namespace NUMINAMATH_CALUDE_base_conversion_314_to_1242_l527_52729

/-- Converts a natural number from base 10 to base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 --/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_314_to_1242 :
  toBase6 314 = [1, 2, 4, 2] ∧ fromBase6 [1, 2, 4, 2] = 314 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_314_to_1242_l527_52729


namespace NUMINAMATH_CALUDE_train_length_l527_52720

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 235 →
  train_speed * crossing_time - bridge_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l527_52720


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l527_52775

theorem sqrt_product_equals_two : Real.sqrt 20 * Real.sqrt (1/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l527_52775


namespace NUMINAMATH_CALUDE_periodic_and_zeros_l527_52791

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_and_zeros (f : ℝ → ℝ) (a : ℝ) :
  (a ≠ 0 ∧ ∀ x, f (x + a) = -f x) →
  IsPeriodic f (2 * a) ∧
  (IsOdd f → (∀ x, f (x + 1) = -f x) →
    ∃ (zeros : Finset ℝ), zeros.card ≥ 4035 ∧
      (∀ x ∈ zeros, -2017 ≤ x ∧ x ≤ 2017 ∧ f x = 0)) :=
by sorry


end NUMINAMATH_CALUDE_periodic_and_zeros_l527_52791


namespace NUMINAMATH_CALUDE_students_in_band_or_sports_l527_52770

theorem students_in_band_or_sports
  (total : ℕ)
  (band : ℕ)
  (sports : ℕ)
  (both : ℕ)
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : both = 60) :
  band + sports - both = 225 :=
by sorry

end NUMINAMATH_CALUDE_students_in_band_or_sports_l527_52770


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l527_52759

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l527_52759


namespace NUMINAMATH_CALUDE_no_extreme_value_at_negative_one_increasing_function_p_range_l527_52713

def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*p*x + 1

theorem no_extreme_value_at_negative_one (p : ℝ) :
  ¬∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε → f p x ≤ f p (-1) ∨ f p x ≥ f p (-1) :=
sorry

theorem increasing_function_p_range :
  ∀ (p : ℝ), (∀ (x y : ℝ), -1 < x ∧ x < y → f p x < f p y) ↔ 0 ≤ p ∧ p ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_value_at_negative_one_increasing_function_p_range_l527_52713


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l527_52700

/-- Given a geometric sequence {a_n} where a_3 = 6 and the sum of the first three terms S_3 = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 6 →                     -- Third term is 6
  a 1 + a 2 + a 3 = 18 →        -- Sum of first three terms is 18
  q = 1 ∨ q = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l527_52700


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l527_52786

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1) * f (2 * n)}

/-- The smallest natural number M such that for any f ∈ S and any n ∈ ℕ, f(n) < M -/
def M : ℕ := 10

theorem smallest_upper_bound : 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ m < M, ∃ f ∈ S, ∃ n, f n ≥ m) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l527_52786


namespace NUMINAMATH_CALUDE_inequality_proof_l527_52702

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b + 1/a > a + 1/b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l527_52702


namespace NUMINAMATH_CALUDE_line_equation_from_intercept_and_slope_l527_52737

/-- A line with x-intercept a and slope m -/
structure Line where
  a : ℝ  -- x-intercept
  m : ℝ  -- slope

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with x-intercept 2 and slope 1, its equation is x - y - 2 = 0 -/
theorem line_equation_from_intercept_and_slope :
  ∀ (L : Line), L.a = 2 ∧ L.m = 1 →
  ∃ (eq : LineEquation), eq.a = 1 ∧ eq.b = -1 ∧ eq.c = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intercept_and_slope_l527_52737


namespace NUMINAMATH_CALUDE_integer_multiplication_result_l527_52785

theorem integer_multiplication_result (x : ℤ) : 
  (10 * x = 64 ∨ 10 * x = 32 ∨ 10 * x = 12 ∨ 10 * x = 25 ∨ 10 * x = 30) → 10 * x = 30 := by
sorry

end NUMINAMATH_CALUDE_integer_multiplication_result_l527_52785


namespace NUMINAMATH_CALUDE_bill_sunday_miles_bill_sunday_miles_proof_l527_52743

theorem bill_sunday_miles : ℕ → ℕ → ℕ → Prop :=
  fun bill_saturday bill_sunday julia_sunday =>
    (bill_sunday = bill_saturday + 4) →
    (julia_sunday = 2 * bill_sunday) →
    (bill_saturday + bill_sunday + julia_sunday = 28) →
    bill_sunday = 8

-- The proof would go here, but we'll skip it as requested
theorem bill_sunday_miles_proof : ∃ (bill_saturday bill_sunday julia_sunday : ℕ),
  bill_sunday_miles bill_saturday bill_sunday julia_sunday :=
sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_bill_sunday_miles_proof_l527_52743


namespace NUMINAMATH_CALUDE_same_birthday_probability_l527_52746

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The probability of two classmates having their birthdays on the same day -/
def birthdayProbability : ℚ := 1 / daysInYear

theorem same_birthday_probability :
  birthdayProbability = 1 / daysInYear := by
  sorry

end NUMINAMATH_CALUDE_same_birthday_probability_l527_52746


namespace NUMINAMATH_CALUDE_range_of_a_l527_52794

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- Define the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l527_52794


namespace NUMINAMATH_CALUDE_planting_area_is_2x_l527_52783

/-- Represents the area of the planting region in a rectangular garden with an internal path. -/
def planting_area (x : ℝ) : ℝ :=
  let garden_length : ℝ := x + 2
  let garden_width : ℝ := 4
  let path_width : ℝ := 1
  let planting_length : ℝ := garden_length - 2 * path_width
  let planting_width : ℝ := garden_width - 2 * path_width
  planting_length * planting_width

/-- Theorem stating that the planting area is equal to 2x square meters. -/
theorem planting_area_is_2x (x : ℝ) : planting_area x = 2 * x := by
  sorry


end NUMINAMATH_CALUDE_planting_area_is_2x_l527_52783


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l527_52799

/-- In a cyclic quadrilateral ABCD where ∠A : ∠B : ∠C = 1 : 2 : 3, ∠D = 90° -/
theorem cyclic_quadrilateral_angle (A B C D : Real) (h1 : A + C = 180) (h2 : B + D = 180)
  (h3 : A / B = 1 / 2) (h4 : B / C = 2 / 3) : D = 90 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l527_52799


namespace NUMINAMATH_CALUDE_binary_1101011_equals_base5_412_l527_52781

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem binary_1101011_equals_base5_412 : 
  decimal_to_base5 (binary_to_decimal [true, true, false, true, false, true, true]) = [4, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_base5_412_l527_52781


namespace NUMINAMATH_CALUDE_choir_arrangement_l527_52763

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 14 = 0) ↔ n ≥ 6930 ∧ ∀ m : ℕ, m < n → (m % 9 ≠ 0 ∨ m % 10 ≠ 0 ∨ m % 11 ≠ 0 ∨ m % 14 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l527_52763


namespace NUMINAMATH_CALUDE_square_value_l527_52797

theorem square_value (square : ℚ) : 44 * 25 = square * 100 → square = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l527_52797


namespace NUMINAMATH_CALUDE_system_of_equations_l527_52717

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 8 * x - 5 * y = c)
  (eq2 : 12 * y - 18 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -16 / 27 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l527_52717


namespace NUMINAMATH_CALUDE_sum_mod_nine_l527_52716

theorem sum_mod_nine : (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l527_52716


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l527_52778

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use -11
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l527_52778


namespace NUMINAMATH_CALUDE_chess_tournament_games_l527_52765

/-- The number of games played in a round-robin chess tournament. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 21 participants, where each participant
    plays exactly one game with each of the remaining participants, 
    the total number of games played is 210. -/
theorem chess_tournament_games :
  num_games 21 = 210 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l527_52765


namespace NUMINAMATH_CALUDE_min_disks_for_vincent_l527_52711

/-- Represents the number of disks required to store files -/
def MinDisks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_09 : ℕ) (files_075 : ℕ) (files_05 : ℕ) : ℕ :=
  sorry

theorem min_disks_for_vincent : 
  MinDisks 40 2 5 15 20 = 18 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_vincent_l527_52711


namespace NUMINAMATH_CALUDE_jack_classics_books_l527_52715

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- Theorem: The total number of books in Jack's classics section is 198 -/
theorem jack_classics_books : num_authors * books_per_author = 198 := by
  sorry

end NUMINAMATH_CALUDE_jack_classics_books_l527_52715


namespace NUMINAMATH_CALUDE_probability_sum_five_l527_52740

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 4

theorem probability_sum_five (dice_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice_outcomes = 36 →
  favorable_outcomes = 4 →
  (favorable_outcomes : ℚ) / dice_outcomes = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l527_52740


namespace NUMINAMATH_CALUDE_estimate_value_l527_52732

theorem estimate_value : 
  3 < (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) ∧ 
  (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) < 4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l527_52732


namespace NUMINAMATH_CALUDE_paving_job_units_l527_52734

theorem paving_job_units (worker1_rate worker2_rate reduced_efficiency total_time : ℝ) 
  (h1 : worker1_rate = 1 / 8)
  (h2 : worker2_rate = 1 / 12)
  (h3 : reduced_efficiency = 8)
  (h4 : total_time = 6) :
  let combined_rate := worker1_rate + worker2_rate - reduced_efficiency / total_time
  total_time * combined_rate = 192 := by
sorry

end NUMINAMATH_CALUDE_paving_job_units_l527_52734


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l527_52756

theorem fraction_sum_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l527_52756


namespace NUMINAMATH_CALUDE_unique_hair_color_assignment_l527_52755

/-- Represents the three people in the problem -/
inductive Person : Type
  | Belokurov : Person
  | Chernov : Person
  | Ryzhov : Person

/-- Represents the three hair colors in the problem -/
inductive HairColor : Type
  | Blond : HairColor
  | Brunette : HairColor
  | RedHaired : HairColor

/-- Represents the assignment of hair colors to people -/
def hairColorAssignment : Person → HairColor
  | Person.Belokurov => HairColor.RedHaired
  | Person.Chernov => HairColor.Blond
  | Person.Ryzhov => HairColor.Brunette

/-- Condition: No person has a hair color matching their surname -/
def noMatchingSurname (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Blond ∧
  assignment Person.Chernov ≠ HairColor.Brunette ∧
  assignment Person.Ryzhov ≠ HairColor.RedHaired

/-- Condition: The brunette is not Belokurov -/
def brunetteNotBelokurov (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Brunette

/-- Condition: All three hair colors are represented -/
def allColorsRepresented (assignment : Person → HairColor) : Prop :=
  (∃ p, assignment p = HairColor.Blond) ∧
  (∃ p, assignment p = HairColor.Brunette) ∧
  (∃ p, assignment p = HairColor.RedHaired)

/-- Main theorem: The given hair color assignment is the only one satisfying all conditions -/
theorem unique_hair_color_assignment :
  ∀ (assignment : Person → HairColor),
    noMatchingSurname assignment ∧
    brunetteNotBelokurov assignment ∧
    allColorsRepresented assignment →
    assignment = hairColorAssignment :=
by sorry

end NUMINAMATH_CALUDE_unique_hair_color_assignment_l527_52755


namespace NUMINAMATH_CALUDE_x_equals_4n_l527_52792

/-- Given that x is 3 times larger than n, and 2n + 3 is some percentage of 25, prove that x = 4n -/
theorem x_equals_4n (n x : ℝ) (p : ℝ) 
  (h1 : x = n + 3 * n) 
  (h2 : 2 * n + 3 = p / 100 * 25) : 
  x = 4 * n := by
sorry

end NUMINAMATH_CALUDE_x_equals_4n_l527_52792


namespace NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l527_52741

theorem solution_set_of_quadratic_inequality :
  ∀ x : ℝ, 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l527_52741


namespace NUMINAMATH_CALUDE_stratified_sampling_l527_52750

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The stratified sampling theorem -/
theorem stratified_sampling
  (pop : Population)
  (sample : Sample)
  (h1 : pop.elderly = 27)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (h4 : sample.elderly = 6)
  (h5 : sample.middleAged / pop.middleAged = sample.elderly / pop.elderly)
  (h6 : sample.young / pop.young = sample.elderly / pop.elderly) :
  sample.elderly + sample.middleAged + sample.young = 36 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l527_52750


namespace NUMINAMATH_CALUDE_interest_difference_l527_52730

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the difference between the principal and the simple interest is 1260 -/
theorem interest_difference :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℝ := 4
  principal - simple_interest principal rate time = 1260 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l527_52730


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l527_52728

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same height and radius has a volume of 24π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * h = 24 * π := by
  sorry

#check cone_volume_from_cylinder_volume

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l527_52728


namespace NUMINAMATH_CALUDE_intersecting_triangles_circumcircle_containment_l527_52727

/-- A triangle in a plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- Two triangles intersect if they have a common point -/
def intersect (t1 t2 : Triangle) : Prop :=
  sorry

/-- A point is inside or on a circle -/
def inside_or_on_circle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem intersecting_triangles_circumcircle_containment 
  (t1 t2 : Triangle) (h : intersect t1 t2) :
  ∃ (i : Fin 3), inside_or_on_circle (t1.vertices i) (circumcircle t2) ∨
                 inside_or_on_circle (t2.vertices i) (circumcircle t1) :=
sorry

end NUMINAMATH_CALUDE_intersecting_triangles_circumcircle_containment_l527_52727


namespace NUMINAMATH_CALUDE_a_range_l527_52762

theorem a_range (P : ∀ x > 0, x + 4 / x ≥ a) 
                (q : ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) : 
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_a_range_l527_52762


namespace NUMINAMATH_CALUDE_g_composition_equals_514_l527_52752

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_514 : g (g (g 1)) = 514 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_514_l527_52752


namespace NUMINAMATH_CALUDE_circle_tangency_line_intersection_l527_52790

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Part I
theorem circle_tangency (m : ℝ) :
  (∃ x y : ℝ, circle_C m x y ∧ unit_circle x y) →
  (∀ x y : ℝ, circle_C m x y → ¬(unit_circle x y)) →
  m = 9 :=
sorry

-- Part II
theorem line_intersection (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C m x₁ y₁ ∧ circle_C m x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 14) →
  m = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_line_intersection_l527_52790


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l527_52703

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l527_52703


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l527_52748

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l527_52748


namespace NUMINAMATH_CALUDE_derivative_at_one_l527_52736

/-- Given f(x) = 2x³ + x² - 5, prove that f'(1) = 8 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + x^2 - 5) : 
  deriv f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l527_52736


namespace NUMINAMATH_CALUDE_jerichos_money_l527_52798

theorem jerichos_money (x : ℕ) : 
  x - (14 + 7) = 9 → 2 * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_jerichos_money_l527_52798


namespace NUMINAMATH_CALUDE_output_for_15_l527_52744

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by sorry

end NUMINAMATH_CALUDE_output_for_15_l527_52744


namespace NUMINAMATH_CALUDE_arithmetic_mean_characterization_l527_52712

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- φ(n) is Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- One of n, τ(n), or φ(n) is the arithmetic mean of the other two -/
def is_arithmetic_mean (n : ℕ+) : Prop :=
  (n : ℚ) = (tau n + phi n) / 2 ∨
  (tau n : ℚ) = (n + phi n) / 2 ∨
  (phi n : ℚ) = (n + tau n) / 2

theorem arithmetic_mean_characterization (n : ℕ+) :
  is_arithmetic_mean n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_characterization_l527_52712


namespace NUMINAMATH_CALUDE_min_value_theorem_l527_52784

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a / (4 * b) + 1 / a ≥ 2 ∧
  (a / (4 * b) + 1 / a = 2 ↔ a = 2/3 ∧ b = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l527_52784


namespace NUMINAMATH_CALUDE_det_4523_equals_2_l527_52796

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det_4523_equals_2 : det2x2 4 5 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_det_4523_equals_2_l527_52796


namespace NUMINAMATH_CALUDE_cubic_root_sum_l527_52776

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 12 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 12 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 12 * r - 7 = 0) →
  (p + q + r = 5/3) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -35/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l527_52776


namespace NUMINAMATH_CALUDE_wheat_cost_per_acre_l527_52780

theorem wheat_cost_per_acre 
  (total_land : ℕ)
  (wheat_land : ℕ)
  (corn_cost_per_acre : ℕ)
  (total_capital : ℕ)
  (h1 : total_land = 4500)
  (h2 : wheat_land = 3400)
  (h3 : corn_cost_per_acre = 42)
  (h4 : total_capital = 165200) :
  ∃ (wheat_cost_per_acre : ℕ),
    wheat_cost_per_acre * wheat_land + 
    corn_cost_per_acre * (total_land - wheat_land) = 
    total_capital ∧ 
    wheat_cost_per_acre = 35 := by
  sorry

end NUMINAMATH_CALUDE_wheat_cost_per_acre_l527_52780


namespace NUMINAMATH_CALUDE_power_steering_count_l527_52761

theorem power_steering_count (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_windows = 25)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_windows - both) = 45 :=
by sorry

end NUMINAMATH_CALUDE_power_steering_count_l527_52761


namespace NUMINAMATH_CALUDE_min_yellow_surface_fraction_l527_52751

-- Define the cube dimensions
def large_cube_edge : ℕ := 4
def small_cube_edge : ℕ := 1

-- Define the number of cubes
def total_cubes : ℕ := 64
def blue_cubes : ℕ := 48
def yellow_cubes : ℕ := 16

-- Define the surface area of the large cube
def large_cube_surface_area : ℕ := 6 * large_cube_edge * large_cube_edge

-- Define the minimum number of yellow cubes that must be on the surface
def min_yellow_surface_cubes : ℕ := yellow_cubes - 1

-- Theorem statement
theorem min_yellow_surface_fraction :
  (min_yellow_surface_cubes : ℚ) / large_cube_surface_area = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_yellow_surface_fraction_l527_52751


namespace NUMINAMATH_CALUDE_contingency_table_confidence_level_l527_52710

/-- Represents a 2x2 contingency table -/
structure ContingencyTable :=
  (data : Matrix (Fin 2) (Fin 2) ℕ)

/-- Calculates the k^2 value for a contingency table -/
def calculate_k_squared (table : ContingencyTable) : ℝ :=
  sorry

/-- Determines the confidence level based on the k^2 value -/
def confidence_level (k_squared : ℝ) : ℝ :=
  sorry

theorem contingency_table_confidence_level :
  ∀ (table : ContingencyTable),
  calculate_k_squared table = 4.013 →
  confidence_level (calculate_k_squared table) = 0.99 :=
sorry

end NUMINAMATH_CALUDE_contingency_table_confidence_level_l527_52710


namespace NUMINAMATH_CALUDE_complex_multiplication_imaginary_zero_l527_52724

theorem complex_multiplication_imaginary_zero (a : ℝ) :
  (Complex.I * (a + Complex.I) + (1 : ℂ) * (a + Complex.I)).im = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_imaginary_zero_l527_52724


namespace NUMINAMATH_CALUDE_digit_150_of_3_over_11_l527_52714

theorem digit_150_of_3_over_11 : ∃ (d : ℕ), d = 7 ∧ 
  (∀ (n : ℕ), n ≥ 1 → n ≤ 150 → 
    (3 * 10^n) % 11 = (d * 10^(150 - n)) % 11) := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_3_over_11_l527_52714


namespace NUMINAMATH_CALUDE_length_of_AB_l527_52704

-- Define the curves (M) and (N)
def curve_M (x y : ℝ) : Prop := x - y = 1

def curve_N (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_M A.1 A.2 ∧ curve_N A.1 A.2 ∧
  curve_M B.1 B.2 ∧ curve_N B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l527_52704


namespace NUMINAMATH_CALUDE_kelsey_travel_time_l527_52708

theorem kelsey_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 400)
  (h2 : speed1 = 25)
  (h3 : speed2 = 40) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_travel_time_l527_52708


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l527_52749

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) :=
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) + (a (i+5)) + (a (i+6)) < 0)

/-- The maximum length of a valid sequence is 10 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ) (n : ℕ), n = 10 ∧ ValidSequence (λ i => if i < n then a i else 0)) ∧
  (∀ (a : ℕ → ℤ) (n : ℕ), n > 10 → ¬ValidSequence (λ i => if i < n then a i else 0)) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l527_52749


namespace NUMINAMATH_CALUDE_parallel_resistances_solutions_l527_52738

theorem parallel_resistances_solutions : 
  ∀ x y z : ℕ+, 
    (1 : ℚ) / z = 1 / x + 1 / y → 
    ((x = 3 ∧ y = 6 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 4 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 12 ∧ z = 3) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_resistances_solutions_l527_52738


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l527_52758

theorem polynomial_factor_sum (m n : ℚ) : 
  (∀ y, my^2 + n*y + 2 = (y + 1)*(y + 2)) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l527_52758


namespace NUMINAMATH_CALUDE_cone_roll_ratio_sum_l527_52709

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  r_pos : r > 0
  h_pos : h > 0

/-- Checks if a number is not a multiple of any prime squared -/
def not_multiple_of_prime_squared (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- Main theorem -/
theorem cone_roll_ratio_sum (cone : RightCircularCone) 
    (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0)
    (h_ratio : cone.h / cone.r = m * Real.sqrt n)
    (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 40 * Real.pi * cone.r)
    (h_not_multiple : not_multiple_of_prime_squared n) :
    m + n = 136 := by
  sorry

end NUMINAMATH_CALUDE_cone_roll_ratio_sum_l527_52709
