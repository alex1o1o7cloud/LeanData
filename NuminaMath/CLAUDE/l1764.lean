import Mathlib

namespace NUMINAMATH_CALUDE_trajectory_and_slope_product_l1764_176499

-- Define the points and the trajectory
def A : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (1, 2)
def P : ℝ × ℝ := (0, -2)

def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1 ∧ p.2 ≠ 0}

-- Define the conditions
structure Triangle (A B C : ℝ × ℝ) : Prop where
  b_on_x_axis : B.2 = 0
  equal_sides : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2
  midpoint_on_y : B.1 + C.1 = 0

-- Define the theorem
theorem trajectory_and_slope_product 
  (B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hC : C ∈ Γ) 
  (l : Set (ℝ × ℝ)) 
  (hl : P ∈ l) 
  (M N : ℝ × ℝ) 
  (hM : M ∈ l ∩ Γ) 
  (hN : N ∈ l ∩ Γ) 
  (hMN : M ≠ N) :
  -- Part I: C satisfies the equation of Γ
  C.2 ^ 2 = 4 * C.1 ∧ 
  -- Part II: Product of slopes is constant
  (M.2 - Q.2) / (M.1 - Q.1) * (N.2 - Q.2) / (N.1 - Q.1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_slope_product_l1764_176499


namespace NUMINAMATH_CALUDE_units_digit_17_power_2024_l1764_176476

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sequence of units digits for powers of a number -/
def unitsDigitSequence (base : ℕ) : ℕ → ℕ
  | 0 => unitsDigit base
  | n + 1 => unitsDigit (base * unitsDigitSequence base n)

theorem units_digit_17_power_2024 :
  unitsDigit (17^2024) = 1 :=
sorry

end NUMINAMATH_CALUDE_units_digit_17_power_2024_l1764_176476


namespace NUMINAMATH_CALUDE_percentage_problem_l1764_176455

theorem percentage_problem (P : ℝ) : (P / 100) * 600 = (50 / 100) * 720 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1764_176455


namespace NUMINAMATH_CALUDE_daycare_count_l1764_176421

/-- The real number of toddlers in the daycare -/
def real_count (bill_count playground_count new_count double_counted missed : ℕ) : ℕ :=
  bill_count - double_counted + missed - playground_count + new_count

/-- Theorem stating the real number of toddlers given the conditions -/
theorem daycare_count : real_count 28 6 4 9 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_daycare_count_l1764_176421


namespace NUMINAMATH_CALUDE_complex_percentage_calculation_l1764_176473

theorem complex_percentage_calculation : 
  let a := 0.15 * 50
  let b := 0.25 * 75
  let c := -0.10 * 120
  let sum := a + b + c
  let d := -0.05 * 150
  2.5 * d - (1/3) * sum = -23.5 := by
sorry

end NUMINAMATH_CALUDE_complex_percentage_calculation_l1764_176473


namespace NUMINAMATH_CALUDE_symmetry_point_x_axis_l1764_176490

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p q : Point3D) : Prop :=
  q.x = p.x ∧ q.y = -p.y ∧ q.z = -p.z

theorem symmetry_point_x_axis :
  let M : Point3D := ⟨1, 2, 3⟩
  let N : Point3D := ⟨1, -2, -3⟩
  symmetricToXAxis M N := by sorry

end NUMINAMATH_CALUDE_symmetry_point_x_axis_l1764_176490


namespace NUMINAMATH_CALUDE_final_amount_theorem_l1764_176416

def initial_amount : ℚ := 1499.9999999999998

def remaining_after_clothes (initial : ℚ) : ℚ := initial - (1/3 * initial)

def remaining_after_food (after_clothes : ℚ) : ℚ := after_clothes - (1/5 * after_clothes)

def remaining_after_travel (after_food : ℚ) : ℚ := after_food - (1/4 * after_food)

theorem final_amount_theorem :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 600 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_theorem_l1764_176416


namespace NUMINAMATH_CALUDE_walters_age_l1764_176481

theorem walters_age (walter_age_1994 : ℝ) (grandmother_age_1994 : ℝ) : 
  walter_age_1994 = grandmother_age_1994 / 3 →
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = 3750 →
  walter_age_1994 + 6 = 65.5 := by
sorry

end NUMINAMATH_CALUDE_walters_age_l1764_176481


namespace NUMINAMATH_CALUDE_trigonometric_function_property_l1764_176491

theorem trigonometric_function_property (f : ℝ → ℝ) :
  (∀ α, f (Real.cos α) = Real.sin α) → f 1 = 0 ∧ f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_function_property_l1764_176491


namespace NUMINAMATH_CALUDE_min_value_theorem_l1764_176426

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 4) :
  9/4 ≤ a + 2*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 2/b₀ = 4 ∧ a₀ + 2*b₀ = 9/4 :=
by sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l1764_176426


namespace NUMINAMATH_CALUDE_distinct_combinations_l1764_176410

def num_shirts : ℕ := 8
def num_ties : ℕ := 7
def num_jackets : ℕ := 3

theorem distinct_combinations : num_shirts * num_ties * num_jackets = 168 := by
  sorry

end NUMINAMATH_CALUDE_distinct_combinations_l1764_176410


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1764_176453

/-- An arithmetic expression using threes and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Count the number of threes in an expression -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- There exists an expression using fewer than ten threes that evaluates to 100 -/
theorem hundred_with_fewer_threes : ∃ e : Expr, countThrees e < 10 ∧ eval e = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1764_176453


namespace NUMINAMATH_CALUDE_total_legs_is_43_l1764_176456

/-- Represents the number of legs for different types of passengers -/
structure LegCount where
  cat : Nat
  human : Nat
  oneLeggedCaptain : Nat

/-- Calculates the total number of legs given the number of heads and cats -/
def totalLegs (totalHeads : Nat) (catCount : Nat) (legCount : LegCount) : Nat :=
  let humanCount := totalHeads - catCount
  let regularHumanCount := humanCount - 1 -- Subtract the one-legged captain
  catCount * legCount.cat + regularHumanCount * legCount.human + legCount.oneLeggedCaptain

/-- Theorem stating that given the conditions, the total number of legs is 43 -/
theorem total_legs_is_43 (totalHeads : Nat) (catCount : Nat) (legCount : LegCount)
    (h1 : totalHeads = 15)
    (h2 : catCount = 7)
    (h3 : legCount.cat = 4)
    (h4 : legCount.human = 2)
    (h5 : legCount.oneLeggedCaptain = 1) :
    totalLegs totalHeads catCount legCount = 43 := by
  sorry


end NUMINAMATH_CALUDE_total_legs_is_43_l1764_176456


namespace NUMINAMATH_CALUDE_angle_bisectors_rational_l1764_176441

/-- Given a triangle with sides a = 84, b = 125, and c = 169, 
    the lengths of all angle bisectors are rational numbers -/
theorem angle_bisectors_rational (a b c : ℚ) (h1 : a = 84) (h2 : b = 125) (h3 : c = 169) :
  ∃ (fa fb fc : ℚ), 
    (fa = 2 * b * c / (b + c) * (((b^2 + c^2 - a^2) / (2 * b * c) + 1) / 2).sqrt) ∧
    (fb = 2 * a * c / (a + c) * (((a^2 + c^2 - b^2) / (2 * a * c) + 1) / 2).sqrt) ∧
    (fc = 2 * a * b / (a + b) * (((a^2 + b^2 - c^2) / (2 * a * b) + 1) / 2).sqrt) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisectors_rational_l1764_176441


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1764_176469

def g (x : ℝ) : ℝ := 3 * x - 5

theorem g_composition_of_three : g (g (g 3)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1764_176469


namespace NUMINAMATH_CALUDE_circle_symmetry_l1764_176411

/-- Given two circles and a line of symmetry, prove that the parameter 'a' in the first circle's equation must equal 2 for the circles to be symmetrical. -/
theorem circle_symmetry (x y : ℝ) (a : ℝ) : 
  (∀ x y, x^2 + y^2 - a*x + 2*y + 1 = 0) →  -- First circle equation
  (∀ x y, x^2 + y^2 = 1) →                  -- Second circle equation
  (∀ x y, x - y = 1) →                      -- Line of symmetry
  a = 2 := by
sorry


end NUMINAMATH_CALUDE_circle_symmetry_l1764_176411


namespace NUMINAMATH_CALUDE_set_difference_N_M_l1764_176420

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_N_M_l1764_176420


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1764_176485

theorem units_digit_of_expression (k : ℕ) (h : k = 2012^2 + 2^2012) :
  (k^3 + 2^(k+1)) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1764_176485


namespace NUMINAMATH_CALUDE_expected_baby_hawks_l1764_176450

/-- The number of kettles being tracked -/
def num_kettles : ℕ := 6

/-- The average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- The number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- The percentage of babies lost -/
def loss_percentage : ℚ := 1/4

/-- The expected number of baby hawks this season -/
def expected_babies : ℕ := 270

theorem expected_baby_hawks :
  expected_babies = num_kettles * pregnancies_per_kettle * babies_per_pregnancy - 
    (num_kettles * pregnancies_per_kettle * babies_per_pregnancy * loss_percentage).floor := by
  sorry

end NUMINAMATH_CALUDE_expected_baby_hawks_l1764_176450


namespace NUMINAMATH_CALUDE_percentage_relation_l1764_176495

theorem percentage_relation (A B C x y : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A = B * (1 + x / 100) →
  A = C * (1 - y / 100) →
  A = 120 →
  B = 100 →
  C = 150 →
  x = 20 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1764_176495


namespace NUMINAMATH_CALUDE_multiples_of_four_l1764_176424

theorem multiples_of_four (n : ℕ) : n = 20 ↔ (
  (∃ (m : List ℕ), 
    m.length = 24 ∧ 
    (∀ x ∈ m, x % 4 = 0) ∧
    (∀ x ∈ m, n ≤ x ∧ x ≤ 112) ∧
    (∀ y, n ≤ y ∧ y ≤ 112 ∧ y % 4 = 0 → y ∈ m)
  )
) := by sorry

end NUMINAMATH_CALUDE_multiples_of_four_l1764_176424


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l1764_176430

theorem mom_tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) : 
  total_tshirts = 51 → tshirts_per_package = 3 → total_tshirts / tshirts_per_package = 17 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l1764_176430


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l1764_176494

theorem geometric_progression_ratio_equation 
  (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (hgp : ∃ a : ℝ, a ≠ 0 ∧ 
    y * (z + x) = r * (x * (y + z)) ∧ 
    z * (x + y) = r * (y * (z + x))) : 
  r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l1764_176494


namespace NUMINAMATH_CALUDE_problem_statements_l1764_176467

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab ≤ 1 → 1/a + 1/b ≥ 2) ∧
  (a + b = 4 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/x + 9/y ≤ 1/a + 9/b ∧ 1/a + 9/b = 4) ∧
  (a^2 + b^2 = 4 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 4 → x*y ≤ a*b ∧ a*b = 2) ∧
  ¬(2*a + b = 1 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y = 1 → x*y ≤ a*b ∧ a*b = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1764_176467


namespace NUMINAMATH_CALUDE_roots_ratio_implies_k_value_l1764_176458

theorem roots_ratio_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
    r ≠ 0 → s ≠ 0 →
    r^2 + 8*r + k = 0 →
    s^2 + 8*s + k = 0 →
    r / s = 3 →
    k = 12 := by
sorry

end NUMINAMATH_CALUDE_roots_ratio_implies_k_value_l1764_176458


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1764_176471

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 5)) ↔ x ≠ -5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1764_176471


namespace NUMINAMATH_CALUDE_polynomial_identity_l1764_176480

theorem polynomial_identity (x : ℝ) :
  (5 * x^3 - 32 * x^2 + 75 * x - 71 = 
   5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) + (-9)) ∧
  (∀ (a b c d : ℝ), 
    (∀ x : ℝ, 5 * x^3 - 32 * x^2 + 75 * x - 71 = 
      a * (x - 2)^3 + b * (x - 2)^2 + c * (x - 2) + d) →
    a = 5 ∧ b = -2 ∧ c = 7 ∧ d = -9) := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1764_176480


namespace NUMINAMATH_CALUDE_class_overall_score_l1764_176482

/-- Calculates the overall score for a class based on four aspects --/
def calculate_overall_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the overall score for the given class is 84 --/
theorem class_overall_score :
  calculate_overall_score 85 90 80 75 = 84 := by
  sorry

#eval calculate_overall_score 85 90 80 75

end NUMINAMATH_CALUDE_class_overall_score_l1764_176482


namespace NUMINAMATH_CALUDE_radio_price_calculation_l1764_176474

def original_price : ℕ := 5000
def final_amount : ℕ := 2468

def discount_tier1 : ℚ := 0.02
def discount_tier2 : ℚ := 0.05
def discount_tier3 : ℚ := 0.10

def tax_tier1 : ℚ := 0.04
def tax_tier2 : ℚ := 0.07
def tax_tier3 : ℚ := 0.09

def discount_threshold1 : ℕ := 2000
def discount_threshold2 : ℕ := 4000

def tax_threshold1 : ℕ := 2500
def tax_threshold2 : ℕ := 4500

def reduced_price : ℕ := 2423

theorem radio_price_calculation :
  let discount := min discount_threshold1 reduced_price * discount_tier1 +
                  min (discount_threshold2 - discount_threshold1) (max (reduced_price - discount_threshold1) 0) * discount_tier2 +
                  max (reduced_price - discount_threshold2) 0 * discount_tier3
  let tax := min tax_threshold1 reduced_price * tax_tier1 +
             min (tax_threshold2 - tax_threshold1) (max (reduced_price - tax_threshold1) 0) * tax_tier2 +
             max (reduced_price - tax_threshold2) 0 * tax_tier3
  reduced_price - discount + tax = final_amount := by
  sorry

end NUMINAMATH_CALUDE_radio_price_calculation_l1764_176474


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l1764_176428

theorem sine_cosine_identity :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l1764_176428


namespace NUMINAMATH_CALUDE_no_integer_solution_l1764_176438

theorem no_integer_solution (n : ℕ+) : ¬ (∃ k : ℤ, (n.val^2 + 1 : ℤ) = k * ((Int.floor (Real.sqrt n.val))^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1764_176438


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l1764_176444

/-- The cost of Linda's TV purchase, given her original savings and the fraction spent on furniture. -/
theorem lindas_tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 1200 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l1764_176444


namespace NUMINAMATH_CALUDE_problem_statement_l1764_176451

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1764_176451


namespace NUMINAMATH_CALUDE_total_animals_l1764_176461

theorem total_animals (a b c : ℕ) (ha : a = 6) (hb : b = 8) (hc : c = 4) :
  a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l1764_176461


namespace NUMINAMATH_CALUDE_sum_of_six_smallest_multiples_of_12_l1764_176445

theorem sum_of_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (λ i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_smallest_multiples_of_12_l1764_176445


namespace NUMINAMATH_CALUDE_additional_distance_with_speed_increase_l1764_176454

/-- Calculates the additional distance traveled when increasing speed for a given initial distance and speeds. -/
theorem additional_distance_with_speed_increase 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed > 0)
  (h2 : faster_speed > actual_speed)
  (h3 : actual_distance > 0)
  : let time := actual_distance / actual_speed
    let faster_distance := faster_speed * time
    faster_distance - actual_distance = 20 :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_with_speed_increase_l1764_176454


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1764_176446

/-- Proves that given a simple interest of 100, an interest rate of 5% per annum,
    and a time period of 4 years, the principal sum is 500. -/
theorem simple_interest_problem (interest : ℕ) (rate : ℕ) (time : ℕ) (principal : ℕ) : 
  interest = 100 → rate = 5 → time = 4 → 
  interest = principal * rate * time / 100 →
  principal = 500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1764_176446


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1764_176434

theorem sum_of_fractions : (1 : ℚ) / 6 + 5 / 12 = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1764_176434


namespace NUMINAMATH_CALUDE_correct_diagnosis_l1764_176435

structure Doctor where
  name : String
  statements : List String

structure Patient where
  diagnosis : List String

def homeopath : Doctor :=
  { name := "Homeopath"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient smokes too much"
    , "The patient has a tropical fever"
    ]
  }

def therapist : Doctor :=
  { name := "Therapist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient doesn't eat well"
    , "The patient suffers from high blood pressure"
    ]
  }

def ophthalmologist : Doctor :=
  { name := "Ophthalmologist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient is near-sighted"
    , "The patient has no signs of retinal detachment"
    ]
  }

def correct_statements : List (Doctor × Nat) :=
  [ (homeopath, 1)
  , (therapist, 0)
  , (ophthalmologist, 0)
  ]

theorem correct_diagnosis (doctors : List Doctor) 
  (correct : List (Doctor × Nat)) : 
  ∃ (p : Patient), 
    p.diagnosis = 
      [ "I have a strong astigmatism"
      , "I smoke too much"
      , "I am not eating well enough!"
      , "I do not have tropical fever"
      ] :=
  sorry

end NUMINAMATH_CALUDE_correct_diagnosis_l1764_176435


namespace NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l1764_176479

theorem largest_multiple_six_negation_greater_than_neg_150 :
  ∀ n : ℤ, (∃ k : ℤ, n = 6 * k) → -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l1764_176479


namespace NUMINAMATH_CALUDE_range_of_f_l1764_176418

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1764_176418


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1764_176431

theorem smallest_integer_solution (x : ℤ) : 
  (3 - 5 * x > 24) ↔ (x ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1764_176431


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l1764_176468

theorem max_value_quadratic_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), x^2 - a*x + a ≥ 0) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - a*x + a ≥ 0) → a ≤ 4) ∧
  (∀ (x : ℝ), x^2 - 4*x + 4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l1764_176468


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1764_176436

/-- The volume of a sphere inscribed in a cube with edge length 4 is 32π/3 -/
theorem inscribed_sphere_volume (cube_edge : ℝ) (sphere_volume : ℝ) :
  cube_edge = 4 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (32 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1764_176436


namespace NUMINAMATH_CALUDE_student_council_distribution_l1764_176462

/-- The number of ways to distribute n indistinguishable items among k distinguishable bins,
    with each bin containing at least 1 item. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 252 ways to distribute 11 positions among 6 classes
    with at least 1 position per class. -/
theorem student_council_distribution : distribute_with_minimum 11 6 = 252 := by
  sorry

end NUMINAMATH_CALUDE_student_council_distribution_l1764_176462


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1764_176457

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 9 ∧ b = 4 ∧
      (a + a > b) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 22
      
#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1764_176457


namespace NUMINAMATH_CALUDE_relay_team_permutations_l1764_176483

theorem relay_team_permutations (n : ℕ) (k : ℕ) :
  n = 5 → k = 3 → Nat.factorial k = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l1764_176483


namespace NUMINAMATH_CALUDE_max_triangle_area_l1764_176484

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 6*x

-- Define points A and B on the parabola
def pointA (x₁ y₁ : ℝ) : Prop := parabola x₁ y₁
def pointB (x₂ y₂ : ℝ) : Prop := parabola x₂ y₂

-- Define the conditions
def conditions (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the perpendicular bisector intersection with x-axis
def pointC (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle ABC
def triangleArea (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (hA : pointA x₁ y₁) 
  (hB : pointB x₂ y₂) 
  (hC : conditions x₁ x₂) :
  ∃ (max_area : ℝ), 
    (∀ (x₁' y₁' x₂' y₂' : ℝ), 
      pointA x₁' y₁' → pointB x₂' y₂' → conditions x₁' x₂' →
      triangleArea x₁' y₁' x₂' y₂' ≤ max_area) ∧
    max_area = (14/3) * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1764_176484


namespace NUMINAMATH_CALUDE_line_through_points_l1764_176412

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 9⟩
  let p2 : Point := ⟨2, -3⟩
  let p3 : Point := ⟨5, 3⟩
  let p4 : Point := ⟨6, 6⟩
  let p5 : Point := ⟨3, 0⟩
  let p6 : Point := ⟨0, -9⟩
  let p7 : Point := ⟨4, 1⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p7 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1764_176412


namespace NUMINAMATH_CALUDE_infinite_solutions_for_c_l1764_176464

theorem infinite_solutions_for_c (x : ℕ) (hx : x > 1) :
  ∃ y z : ℕ,
    y > 0 ∧ z > 0 ∧
    (x^2 - (x^2 - 2)) * (y^2 - (x^2 - 2)) = z^2 - (x^2 - 2) ∧
    (x^2 + (x^2 - 2)) * (y^2 - (x^2 - 2)) = z^2 - (x^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_c_l1764_176464


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1764_176475

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 216 →
  volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) →
  volume = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1764_176475


namespace NUMINAMATH_CALUDE_binomial_7_2_l1764_176406

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l1764_176406


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1764_176488

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  m : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_intersection_theorem (E : Ellipse) (L : IntersectingLine) : 
  (E.a^2 - E.b^2 = 1) →  -- Focal length is 2
  (1 / E.a^2 + (9/4) / E.b^2 = 1) →  -- Ellipse passes through (1, 3/2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Intersection points exist
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    y₁ = 3/2 * x₁ + L.m ∧
    y₂ = 3/2 * x₂ + L.m) →
  (∃ (k₁ k₂ : ℝ),  -- Slope ratio condition
    k₁ / k₂ = 2 ∧
    k₁ = y₂ / (x₂ + 2) ∧
    k₂ = y₁ / (x₁ - 2)) →
  L.m = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1764_176488


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_118_l1764_176409

theorem alpha_plus_beta_equals_118 :
  ∀ α β : ℝ,
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 63*x - 3969)) →
  α + β = 118 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_118_l1764_176409


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1764_176448

theorem consecutive_numbers_sum (a : ℤ) : 
  (a + (a + 1) + (a + 2) = 184) ∧
  (a + (a + 1) + (a + 3) = 201) ∧
  (a + (a + 2) + (a + 3) = 212) ∧
  ((a + 1) + (a + 2) + (a + 3) = 226) →
  (a + 3 = 70) := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1764_176448


namespace NUMINAMATH_CALUDE_ursula_change_l1764_176497

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change (hot_dog_price : ℚ) (salad_price : ℚ) 
  (num_hot_dogs : ℕ) (num_salads : ℕ) (bill_value : ℚ) (num_bills : ℕ) :
  hot_dog_price = 3/2 →
  salad_price = 5/2 →
  num_hot_dogs = 5 →
  num_salads = 3 →
  bill_value = 10 →
  num_bills = 2 →
  (num_bills * bill_value) - (num_hot_dogs * hot_dog_price + num_salads * salad_price) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ursula_change_l1764_176497


namespace NUMINAMATH_CALUDE_problem_statement_l1764_176403

theorem problem_statement (x y : ℝ) (m n : ℤ) 
  (h : x > 0) (h' : y > 0) 
  (eq : x^m * y * 4*y^n / (4*x^6*y^4) = 1) : 
  m - n = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1764_176403


namespace NUMINAMATH_CALUDE_illuminated_area_of_cube_l1764_176459

/-- The area of the illuminated part of a cube's surface when illuminated by a cylindrical beam -/
theorem illuminated_area_of_cube (a ρ : ℝ) (h_a : a = Real.sqrt (2 + Real.sqrt 3)) (h_ρ : ρ = Real.sqrt 2) :
  let S := ρ^2 * Real.sqrt 3 * (Real.pi - 6 * Real.arccos (a / (ρ * Real.sqrt 2)) + 
           6 * (a / (ρ * Real.sqrt 2)) * Real.sqrt (1 - (a / (ρ * Real.sqrt 2))^2))
  S = Real.sqrt 3 * (Real.pi + 3) :=
sorry

end NUMINAMATH_CALUDE_illuminated_area_of_cube_l1764_176459


namespace NUMINAMATH_CALUDE_prism_15_edges_l1764_176477

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Theorem: A prism with 15 edges has 7 faces and 10 vertices. -/
theorem prism_15_edges (p : Prism) (h : p.edges = 15) : p.faces = 7 ∧ p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_prism_15_edges_l1764_176477


namespace NUMINAMATH_CALUDE_factorial_350_trailing_zeros_l1764_176449

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_350_trailing_zeros :
  trailing_zeros 350 = 86 := by
  sorry

end NUMINAMATH_CALUDE_factorial_350_trailing_zeros_l1764_176449


namespace NUMINAMATH_CALUDE_two_thousandth_digit_sum_l1764_176460

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 2000 ∧
  seq.head? = some 3 ∧
  ∀ i, i < 1999 → (seq.get? i).isSome ∧ (seq.get? (i+1)).isSome →
    (17 ∣ (seq.get! i * 10 + seq.get! (i+1))) ∨ (23 ∣ (seq.get! i * 10 + seq.get! (i+1)))

theorem two_thousandth_digit_sum (seq : List Nat) (a b : Nat) :
  is_valid_sequence seq →
  (seq.get? 1999 = some a ∨ seq.get? 1999 = some b) →
  a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_thousandth_digit_sum_l1764_176460


namespace NUMINAMATH_CALUDE_golden_ratio_bounds_l1764_176413

theorem golden_ratio_bounds : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_bounds_l1764_176413


namespace NUMINAMATH_CALUDE_triangle_side_integral_difference_l1764_176429

def triangle_side_difference (x : ℤ) : Prop :=
  x > 2 ∧ x < 18

theorem triangle_side_integral_difference :
  (∃ x_max x_min : ℤ, 
    (∀ x : ℤ, triangle_side_difference x → x ≤ x_max ∧ x ≥ x_min) ∧
    triangle_side_difference x_max ∧
    triangle_side_difference x_min ∧
    x_max - x_min = 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_integral_difference_l1764_176429


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1764_176487

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.25 * x + 80) / (x + 200) = 0.31 →
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1764_176487


namespace NUMINAMATH_CALUDE_circle_symmetry_l1764_176440

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ∧ symmetry_line x y → symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1764_176440


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l1764_176414

theorem arc_length_of_sector (θ : Real) (r : Real) (L : Real) : 
  θ = 120 → r = 3/2 → L = θ / 360 * (2 * Real.pi * r) → L = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l1764_176414


namespace NUMINAMATH_CALUDE_initial_cards_equals_sum_l1764_176423

/-- The number of Pokemon cards Jason had initially --/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away --/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left --/
def cards_left : ℕ := 4

/-- Theorem stating that the initial number of cards equals the sum of cards given away and cards left --/
theorem initial_cards_equals_sum : initial_cards = cards_given_away + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_equals_sum_l1764_176423


namespace NUMINAMATH_CALUDE_max_volume_cylinder_in_sphere_l1764_176402

noncomputable section

theorem max_volume_cylinder_in_sphere (R : ℝ) (h r : ℝ → ℝ) :
  (∀ t, 4 * R^2 = 4 * (r t)^2 + (h t)^2) →
  (∀ t, (r t) ≥ 0 ∧ (h t) ≥ 0) →
  (∃ t₀, ∀ t, π * (r t)^2 * (h t) ≤ π * (r t₀)^2 * (h t₀)) →
  h t₀ = 2 * R / Real.sqrt 3 ∧ r t₀ = R * Real.sqrt (2/3) :=
by sorry

end

end NUMINAMATH_CALUDE_max_volume_cylinder_in_sphere_l1764_176402


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1764_176472

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ k m : ℕ, 1 < k ∧ k < n ∧ n = k * m

theorem units_digit_of_product :
  isComposite 9 ∧ isComposite 10 ∧ isComposite 12 →
  unitsDigit (9 * 10 * 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1764_176472


namespace NUMINAMATH_CALUDE_journey_distance_l1764_176425

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 15 →
  speed1 = 21 →
  speed2 = 24 →
  ∃ (distance : ℝ),
    distance / 2 / speed1 + distance / 2 / speed2 = total_time ∧
    distance = 336 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1764_176425


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1764_176492

/-- Two planar vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given planar vectors a and b, if they are parallel, then x = -3/2 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-2, 3)
  are_parallel a b → x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1764_176492


namespace NUMINAMATH_CALUDE_edward_additional_spending_l1764_176463

def edward_spending (initial_amount spent_first final_amount : ℕ) : ℕ :=
  initial_amount - spent_first - final_amount

theorem edward_additional_spending :
  edward_spending 34 9 17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_additional_spending_l1764_176463


namespace NUMINAMATH_CALUDE_max_cake_pieces_l1764_176404

/-- Represents the dimensions of a rectangular cake -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Checks if the given dimensions satisfy the required condition -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  (d.m - 2) * (d.n - 2) = (d.m * d.n) / 2

/-- Calculates the total number of cake pieces -/
def totalPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n

/-- Theorem stating the maximum number of cake pieces possible -/
theorem max_cake_pieces :
  ∃ (d : CakeDimensions), satisfiesCondition d ∧ 
    (∀ (d' : CakeDimensions), satisfiesCondition d' → totalPieces d' ≤ totalPieces d) ∧
    totalPieces d = 60 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l1764_176404


namespace NUMINAMATH_CALUDE_triangle_side_c_l1764_176486

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_c (t : Triangle) 
  (h1 : t.a = 5) 
  (h2 : t.b = 7) 
  (h3 : t.B = 60 * π / 180) : 
  t.c = 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_c_l1764_176486


namespace NUMINAMATH_CALUDE_sample_size_l1764_176489

theorem sample_size (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  2 * f₆ = f₁ →
  3 * f₆ = f₂ →
  4 * f₆ = f₃ →
  6 * f₆ = f₄ →
  4 * f₆ = f₅ →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_l1764_176489


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l1764_176427

/-- The number of ways to arrange volunteers among events --/
def arrangeVolunteers (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to arrange volunteers among events, excluding one event --/
def arrangeVolunteersExcludeOne (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

/-- The number of ways to arrange volunteers to only one event --/
def arrangeVolunteersToOne (n : ℕ) (k : ℕ) : ℕ := k

theorem volunteer_arrangement_count :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 3  -- number of events
  arrangeVolunteers n k - k * arrangeVolunteersExcludeOne n (k-1) + arrangeVolunteersToOne n k = 150 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l1764_176427


namespace NUMINAMATH_CALUDE_smallest_n_for_doughnuts_l1764_176470

theorem smallest_n_for_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (13 * m - 1) % 9 = 0 → m ≥ n) ∧
  (13 * n - 1) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_doughnuts_l1764_176470


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l1764_176437

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an original volume of 20 cubic feet,
  if its radius is tripled and its height is quadrupled,
  then its new volume will be 720 cubic feet.
-/
theorem cylinder_volume_change (r h : ℝ) :
  (π * r^2 * h = 20) →  -- Original volume is 20 cubic feet
  (π * (3*r)^2 * (4*h) = 720) :=  -- New volume is 720 cubic feet
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l1764_176437


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_solution_is_185_l1764_176447

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem solution_is_185 : ∃ (n : ℕ), n = 185 ∧ n < 200 ∧ Nat.gcd n 30 = 5 ∧
  ∀ (m : ℕ), m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_solution_is_185_l1764_176447


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1764_176465

theorem sufficient_not_necessary (a b : ℝ) : 
  (a * b ≥ 2 → a^2 + b^2 ≥ 4) ∧ 
  ∃ a b : ℝ, a^2 + b^2 ≥ 4 ∧ a * b < 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1764_176465


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1764_176401

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a r n = 29524/59049 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1764_176401


namespace NUMINAMATH_CALUDE_fold_theorem_l1764_176405

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line on a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Defines a fold on graph paper -/
def Fold (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (foldLine : Line),
    -- The fold line is perpendicular to the line connecting p1 and p2
    foldLine.slope * ((p2.x - p1.x) / (p2.y - p1.y)) = -1 ∧
    -- The midpoint of p1 and p2 is on the fold line
    (p1.y + p2.y) / 2 = foldLine.slope * ((p1.x + p2.x) / 2) + foldLine.yIntercept ∧
    -- The midpoint of p3 and p4 is on the fold line
    (p3.y + p4.y) / 2 = foldLine.slope * ((p3.x + p4.x) / 2) + foldLine.yIntercept ∧
    -- The line connecting p3 and p4 is perpendicular to the fold line
    foldLine.slope * ((p4.x - p3.x) / (p4.y - p3.y)) = -1

/-- The main theorem to prove -/
theorem fold_theorem (m n : ℝ) :
  Fold ⟨0, 3⟩ ⟨5, 0⟩ ⟨8, 5⟩ ⟨m, n⟩ → m + n = 10.3 := by
  sorry

end NUMINAMATH_CALUDE_fold_theorem_l1764_176405


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1764_176408

-- Define the sample space
def SampleSpace := Finset (Fin 4 × Fin 4)

-- Define the event of selecting exactly one girl
def exactlyOneGirl (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∨ p.2 > 1)).card = 1

-- Define the event of selecting exactly two girls
def exactlyTwoGirls (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∧ p.2 > 1)).card = 2

-- State the theorem
theorem mutually_exclusive_but_not_complementary :
  (∀ s : SampleSpace, ¬(exactlyOneGirl s ∧ exactlyTwoGirls s)) ∧
  (∃ s : SampleSpace, ¬(exactlyOneGirl s ∨ exactlyTwoGirls s)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1764_176408


namespace NUMINAMATH_CALUDE_arccos_of_neg_one_eq_pi_l1764_176419

theorem arccos_of_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_neg_one_eq_pi_l1764_176419


namespace NUMINAMATH_CALUDE_problem_solution_l1764_176415

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := (1/3) * x^3 - x
def g (x : ℝ) := 33 * f x + 3 * x

-- Define the sequence bₙ
def b (n : ℕ) : ℝ := g n ^ (1 / g (n + 1))

-- Theorem statement
theorem problem_solution :
  -- f(x) reaches its maximum value 2/3 when x = -1
  (f (-1) = 2/3 ∧ ∀ x, f x ≤ 2/3) ∧
  -- The graph of y = f(x+1) is symmetrical about the point (-1, 0)
  (∀ x, f (x + 1) = -f (-x - 1)) →
  -- 1. f(x) = (1/3)x³ - x is implied by the above conditions
  (∀ x, f x = (1/3) * x^3 - x) ∧
  -- 2. When x > 0, [1 + 1/g(x)]^g(x) < e
  (∀ x > 0, (1 + 1 / g x) ^ (g x) < Real.exp 1) ∧
  -- 3. The sequence bₙ has only one equal pair: b₂ = b₈
  (∀ n m : ℕ, n ≠ m → b n = b m ↔ (n = 2 ∧ m = 8) ∨ (n = 8 ∧ m = 2)) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l1764_176415


namespace NUMINAMATH_CALUDE_least_value_quadratic_l1764_176417

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l1764_176417


namespace NUMINAMATH_CALUDE_max_pieces_is_sixteen_l1764_176400

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 16

/-- The size of a small cake piece in inches -/
def small_piece_size : ℕ := 4

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_sixteen : max_pieces = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_sixteen_l1764_176400


namespace NUMINAMATH_CALUDE_sequence_bounded_l1764_176422

/-- A sequence of non-negative real numbers satisfying certain conditions is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) 
  (hc : c > 2)
  (ha_nonneg : ∀ n, a n ≥ 0)
  (h1 : ∀ m n : ℕ, a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k + 1 : ℝ)^c)) :
  ∃ M : ℝ, ∀ n : ℕ, a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l1764_176422


namespace NUMINAMATH_CALUDE_algebrist_great_probability_l1764_176442

def algebrist : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'I', 'S', 'T'}
def great : Finset Char := {'G', 'R', 'E', 'A', 'T'}

theorem algebrist_great_probability :
  Finset.card (algebrist ∩ great) / Finset.card algebrist = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_algebrist_great_probability_l1764_176442


namespace NUMINAMATH_CALUDE_only_pi_smaller_than_neg_three_l1764_176496

theorem only_pi_smaller_than_neg_three : 
  (-Real.sqrt 2 > -3) ∧ (1 > -3) ∧ (0 > -3) ∧ (-Real.pi < -3) := by
  sorry

end NUMINAMATH_CALUDE_only_pi_smaller_than_neg_three_l1764_176496


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l1764_176452

theorem continued_fraction_equality : 
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l1764_176452


namespace NUMINAMATH_CALUDE_gary_initial_stickers_l1764_176439

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left -/
def stickers_left : ℕ := 31

/-- The initial number of stickers Gary had -/
def initial_stickers : ℕ := stickers_to_lucy + stickers_to_alex + stickers_left

theorem gary_initial_stickers :
  initial_stickers = 99 :=
by sorry

end NUMINAMATH_CALUDE_gary_initial_stickers_l1764_176439


namespace NUMINAMATH_CALUDE_jeremy_dosage_l1764_176498

/-- Represents the duration of Jeremy's medication course in weeks -/
def duration : ℕ := 2

/-- Represents the number of pills Jeremy takes in total -/
def total_pills : ℕ := 112

/-- Represents the dosage of each pill in milligrams -/
def pill_dosage : ℕ := 500

/-- Represents the interval between doses in hours -/
def dose_interval : ℕ := 6

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the total milligrams of medication taken over the entire course -/
def total_mg : ℕ := total_pills * pill_dosage

/-- Calculates the number of doses taken per day -/
def doses_per_day : ℕ := hours_per_day / dose_interval

/-- Calculates the total number of doses taken over the entire course -/
def total_doses : ℕ := duration * 7 * doses_per_day

/-- Theorem stating that Jeremy takes 1000 mg every 6 hours -/
theorem jeremy_dosage : total_mg / total_doses = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_dosage_l1764_176498


namespace NUMINAMATH_CALUDE_range_of_a_l1764_176432

theorem range_of_a (x₁ x₂ m a : ℝ) : 
  (∀ m ∈ Set.Icc (-1 : ℝ) 1, x₁^2 - m*x₁ - 2 = 0 ∧ x₂^2 - m*x₂ - 2 = 0) →
  (∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ |x₁ - x₂|) →
  (¬∃ x, a*x^2 + 2*x - 1 > 0) →
  (∃ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 < |x₁ - x₂|) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1764_176432


namespace NUMINAMATH_CALUDE_puppies_per_cage_l1764_176433

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : num_cages = 9) 
  : (initial_puppies - sold_puppies) / num_cages = 9 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l1764_176433


namespace NUMINAMATH_CALUDE_phone_call_probability_l1764_176478

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.2

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering a phone call within the first four rings -/
def p_within_four_rings : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability : p_within_four_rings = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_probability_l1764_176478


namespace NUMINAMATH_CALUDE_derivatives_at_zero_l1764_176407

open Function Real

/-- A function f satisfying the given conditions -/
def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → f (1 / n) = n^2 / (n^2 + 1)

/-- The theorem statement -/
theorem derivatives_at_zero
  (f : ℝ → ℝ)
  (h_smooth : ContDiff ℝ ⊤ f)
  (h_cond : f_condition f) :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv^[2] f 0 = -2 ∧
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by sorry

end NUMINAMATH_CALUDE_derivatives_at_zero_l1764_176407


namespace NUMINAMATH_CALUDE_cake_slices_problem_l1764_176466

theorem cake_slices_problem (num_cakes : ℕ) (price_per_slice donation1 donation2 total_raised : ℚ) :
  num_cakes = 10 →
  price_per_slice = 1 →
  donation1 = 1/2 →
  donation2 = 1/4 →
  total_raised = 140 →
  ∃ (slices_per_cake : ℕ), 
    slices_per_cake = 8 ∧
    (num_cakes * slices_per_cake : ℚ) * (price_per_slice + donation1 + donation2) = total_raised :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_problem_l1764_176466


namespace NUMINAMATH_CALUDE_factors_lcm_gcd_of_24_60_180_l1764_176443

def numbers : List Nat := [24, 60, 180]

theorem factors_lcm_gcd_of_24_60_180 :
  (∃ (common_factors : List Nat), common_factors.length = 6 ∧ 
    ∀ n ∈ common_factors, ∀ m ∈ numbers, n ∣ m) ∧
  Nat.lcm 24 (Nat.lcm 60 180) = 180 ∧
  Nat.gcd 24 (Nat.gcd 60 180) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_lcm_gcd_of_24_60_180_l1764_176443


namespace NUMINAMATH_CALUDE_coprime_20172019_l1764_176493

theorem coprime_20172019 : 
  (Nat.gcd 20172019 20172017 = 1) ∧ 
  (Nat.gcd 20172019 20172018 = 1) ∧ 
  (Nat.gcd 20172019 20172020 = 1) ∧ 
  (Nat.gcd 20172019 20172021 = 1) := by
  sorry

end NUMINAMATH_CALUDE_coprime_20172019_l1764_176493
