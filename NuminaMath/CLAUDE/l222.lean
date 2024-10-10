import Mathlib

namespace no_real_solution_log_equation_l222_22240

theorem no_real_solution_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 7*x + 10)) ∧ 
             (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 7*x + 10 > 0) :=
by sorry

end no_real_solution_log_equation_l222_22240


namespace inequality_proof_l222_22231

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) : 
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end inequality_proof_l222_22231


namespace shooting_scores_mode_and_variance_l222_22238

def scores : List ℕ := [8, 9, 9, 10, 10, 7, 8, 9, 10, 10]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ := 
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ := 
  let μ := mean l
  (l.map (λ x => ((x : ℚ) - μ) ^ 2)).sum / l.length

theorem shooting_scores_mode_and_variance :
  mode scores = 10 ∧ variance scores = 1 := by sorry

end shooting_scores_mode_and_variance_l222_22238


namespace eighth_term_value_l222_22244

/-- An arithmetic sequence with 30 terms, first term 5, and last term 80 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  let d := (80 - 5) / 29
  5 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighthTerm : ℚ := arithmeticSequence 8

theorem eighth_term_value : eighthTerm = 670 / 29 := by sorry

end eighth_term_value_l222_22244


namespace no_matching_units_digits_l222_22258

theorem no_matching_units_digits :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by sorry

end no_matching_units_digits_l222_22258


namespace caleb_dandelion_friends_l222_22262

/-- The number of friends Caleb shared dandelion puffs with -/
def num_friends (total : ℕ) (mom sister grandma dog friend : ℕ) : ℕ :=
  (total - (mom + sister + grandma + dog)) / friend

/-- Theorem stating the number of friends Caleb shared dandelion puffs with -/
theorem caleb_dandelion_friends :
  num_friends 40 3 3 5 2 9 = 3 := by
  sorry

end caleb_dandelion_friends_l222_22262


namespace jason_retirement_age_l222_22259

/-- Jason's career in the military -/
def military_career (joining_age : ℕ) (years_to_chief : ℕ) (additional_years : ℕ) : Prop :=
  let years_to_master_chief : ℕ := years_to_chief + (years_to_chief / 4)
  let total_years : ℕ := years_to_chief + years_to_master_chief + additional_years
  let retirement_age : ℕ := joining_age + total_years
  retirement_age = 46

theorem jason_retirement_age :
  military_career 18 8 10 :=
by
  sorry

end jason_retirement_age_l222_22259


namespace inequality_equivalence_l222_22233

theorem inequality_equivalence (x : ℝ) : 
  (|(x^2 - 9) / 3| < 3) ↔ (-Real.sqrt 18 < x ∧ x < Real.sqrt 18) := by
  sorry

end inequality_equivalence_l222_22233


namespace no_natural_number_with_sum_of_squared_divisors_perfect_square_l222_22237

theorem no_natural_number_with_sum_of_squared_divisors_perfect_square :
  ¬ ∃ (n : ℕ), ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ), 
    (∀ d : ℕ, d ∣ n → d ≥ d₅) ∧ 
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    ∃ (m : ℕ), d₁^2 + d₂^2 + d₃^2 + d₄^2 + d₅^2 = m^2 :=
by
  sorry


end no_natural_number_with_sum_of_squared_divisors_perfect_square_l222_22237


namespace chord_intersection_angle_l222_22275

theorem chord_intersection_angle (θ : Real) : 
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  (∃ (x y : Real), 
    x * Real.sin θ + y * Real.cos θ - 1 = 0 ∧
    (x - 1)^2 + (y - Real.cos θ)^2 = 1/4 ∧
    ∃ (x' y' : Real), 
      x' * Real.sin θ + y' * Real.cos θ - 1 = 0 ∧
      (x' - 1)^2 + (y' - Real.cos θ)^2 = 1/4 ∧
      (x - x')^2 + (y - y')^2 = 3/4) →
  θ = Real.pi / 6 := by
sorry

end chord_intersection_angle_l222_22275


namespace tangent_lines_to_parabola_l222_22208

/-- The curve function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Point B -/
def B : ℝ × ℝ := (3, 5)

/-- Tangent line equation type -/
structure TangentLine where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a * x + b * y + c = 0

/-- Theorem: The equations of the lines that pass through B and are tangent to f are 2x - y - 1 = 0 and 10x - y - 25 = 0 -/
theorem tangent_lines_to_parabola :
  ∃ (l₁ l₂ : TangentLine),
    (l₁.equation 3 5 ∧ l₂.equation 3 5) ∧
    (∀ x y, y = f x → (l₁.equation x y ∨ l₂.equation x y) → 
      ∃ ε > 0, ∀ h ∈ Set.Ioo (x - ε) (x + ε), h ≠ x → f h > (l₁.a * h + l₁.c) ∧ f h > (l₂.a * h + l₂.c)) ∧
    l₁.equation = fun x y => 2 * x - y - 1 = 0 ∧
    l₂.equation = fun x y => 10 * x - y - 25 = 0 :=
by sorry

end tangent_lines_to_parabola_l222_22208


namespace no_more_permutations_than_value_l222_22230

theorem no_more_permutations_than_value (b n : ℕ) : b > 1 → n > 1 → 
  let r := (Nat.log b n).succ
  let digits := Nat.digits b n
  (List.permutations digits).length ≤ n := by
  sorry

end no_more_permutations_than_value_l222_22230


namespace stadium_length_in_feet_l222_22251

/-- Proves that the length of a 61-yard stadium is 183 feet. -/
theorem stadium_length_in_feet :
  let stadium_length_yards : ℕ := 61
  let yards_to_feet_conversion : ℕ := 3
  stadium_length_yards * yards_to_feet_conversion = 183 :=
by sorry

end stadium_length_in_feet_l222_22251


namespace quadratic_roots_property_l222_22205

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 1994*m + 7 = 0 → 
  n^2 + 1994*n + 7 = 0 → 
  (m^2 + 1993*m + 6) * (n^2 + 1995*n + 8) = 1986 := by
sorry

end quadratic_roots_property_l222_22205


namespace smallest_non_factor_product_of_60_l222_22297

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_60 (a b : ℕ) :
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 60 →
  is_factor b 60 →
  ¬ is_factor (a * b) 60 →
  ∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → is_factor c 60 → is_factor d 60 → ¬ is_factor (c * d) 60 → a * b ≤ c * d :=
by sorry

end smallest_non_factor_product_of_60_l222_22297


namespace problem_statement_l222_22261

theorem problem_statement (a b c : ℝ) : 
  a^2 + b^2 + c^2 + 4 ≤ a*b + 3*b + 2*c → 200*a + 9*b + c = 219 := by
  sorry

end problem_statement_l222_22261


namespace sqrt_equation_solution_l222_22223

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt (4 * x - 5)) = Real.sqrt 10 → x = 13.5 := by
  sorry

end sqrt_equation_solution_l222_22223


namespace sum_fifth_powers_divisible_by_30_l222_22283

theorem sum_fifth_powers_divisible_by_30 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h : 30 ∣ (Finset.univ.sum (λ i => a i))) : 
  30 ∣ (Finset.univ.sum (λ i => (a i)^5)) :=
sorry

end sum_fifth_powers_divisible_by_30_l222_22283


namespace three_digit_number_proof_l222_22235

theorem three_digit_number_proof :
  ∃! (a b c : ℕ),
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a + b + c = 16 ∧
    100 * b + 10 * a + c = 100 * a + 10 * b + c - 360 ∧
    100 * a + 10 * c + b = 100 * a + 10 * b + c + 54 ∧
    100 * a + 10 * b + c = 628 :=
by sorry

end three_digit_number_proof_l222_22235


namespace triangle_angle_calculation_l222_22298

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 60 → B = 2 * C → A + B + C = 180 → B = 80 := by sorry

end triangle_angle_calculation_l222_22298


namespace restaurant_bill_split_l222_22280

-- Define the meal costs and discounts
def sarah_meal : ℝ := 20
def mary_meal : ℝ := 22
def tuan_meal : ℝ := 18
def michael_meal : ℝ := 24
def linda_meal : ℝ := 16
def sarah_coupon : ℝ := 4
def student_discount : ℝ := 0.1
def sales_tax : ℝ := 0.08
def tip_percentage : ℝ := 0.15
def num_people : ℕ := 5

-- Define the theorem
theorem restaurant_bill_split (
  sarah_meal mary_meal tuan_meal michael_meal linda_meal : ℝ)
  (sarah_coupon student_discount sales_tax tip_percentage : ℝ)
  (num_people : ℕ) :
  let total_before_discount := sarah_meal + mary_meal + tuan_meal + michael_meal + linda_meal
  let sarah_discounted := sarah_meal - sarah_coupon
  let tuan_discounted := tuan_meal * (1 - student_discount)
  let linda_discounted := linda_meal * (1 - student_discount)
  let total_after_discount := sarah_discounted + mary_meal + tuan_discounted + michael_meal + linda_discounted
  let tax_amount := total_after_discount * sales_tax
  let tip_amount := total_before_discount * tip_percentage
  let final_bill := total_after_discount + tax_amount + tip_amount
  let individual_contribution := final_bill / num_people
  individual_contribution = 23 :=
by
  sorry


end restaurant_bill_split_l222_22280


namespace quadratic_inequality_l222_22218

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  solution_set : ∀ x : ℝ, (x < -2 ∨ x > 4) ↔ (a * x^2 + b * x + c > 0)

/-- The main theorem stating the inequality for specific x values -/
theorem quadratic_inequality (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * (-1)^2 + f.b * (-1) + f.c ∧
  f.a * (-1)^2 + f.b * (-1) + f.c < f.a * 5^2 + f.b * 5 + f.c :=
sorry

end quadratic_inequality_l222_22218


namespace cos_half_angle_l222_22207

theorem cos_half_angle (θ : ℝ) (h1 : |Real.cos θ| = (1 : ℝ) / 5) 
  (h2 : (7 : ℝ) * Real.pi / 2 < θ) (h3 : θ < 4 * Real.pi) : 
  Real.cos (θ / 2) = Real.sqrt 15 / 5 := by
  sorry

end cos_half_angle_l222_22207


namespace inequality_proof_l222_22270

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  (1 / (a^4 + 3)) + (1 / (b^4 + 3)) + (1 / (c^4 + 3)) ≥ 3/4 := by
  sorry

end inequality_proof_l222_22270


namespace hyperbola_equation_l222_22228

theorem hyperbola_equation (x y : ℝ) :
  (∀ t : ℝ, y = (2/3) * x ∨ y = -(2/3) * x) →  -- asymptotes condition
  (∃ x₀ y₀ : ℝ, x₀ = 3 ∧ y₀ = 4 ∧ (y₀^2 / 12 - x₀^2 / 27 = 1)) →  -- point condition
  (y^2 / 12 - x^2 / 27 = 1) :=  -- equation of the hyperbola
by sorry

end hyperbola_equation_l222_22228


namespace odd_function_negative_domain_l222_22291

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 := by
  sorry

end odd_function_negative_domain_l222_22291


namespace roses_cut_l222_22222

theorem roses_cut (initial_roses vase_roses garden_roses : ℕ) 
  (h1 : initial_roses = 7)
  (h2 : vase_roses = 20)
  (h3 : garden_roses = 59) :
  vase_roses - initial_roses = 13 := by
  sorry

end roses_cut_l222_22222


namespace max_min_difference_l222_22241

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (heq : x + 2 * y = 4) :
  ∃ (max min : ℝ), 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → |2 * z - w| / (|z| + |w|) ≤ max) ∧
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → min ≤ |2 * z - w| / (|z| + |w|)) ∧
    max - min = 5 :=
sorry

end max_min_difference_l222_22241


namespace ball_piles_problem_l222_22210

theorem ball_piles_problem (x y z a : ℕ) : 
  x + y + z = 2012 →
  y - a = 17 →
  x - a = 2 * (z - a) →
  z = 665 := by
sorry

end ball_piles_problem_l222_22210


namespace regression_change_l222_22219

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The change in the dependent variable when the independent variable increases by one unit -/
def change_in_y (reg : LinearRegression) : ℝ :=
  reg.intercept - reg.slope * (reg.intercept + 1) - (reg.intercept - reg.slope * reg.intercept)

theorem regression_change (reg : LinearRegression) 
  (h : reg.intercept = 2 ∧ reg.slope = 3) : 
  change_in_y reg = -3 := by
  sorry

#eval change_in_y { intercept := 2, slope := 3 }

end regression_change_l222_22219


namespace sara_grew_four_onions_l222_22221

/-- The number of onions grown by Sara, given the total number of onions and the numbers grown by Sally and Fred. -/
def saras_onions (total : ℕ) (sallys : ℕ) (freds : ℕ) : ℕ :=
  total - sallys - freds

/-- Theorem stating that Sara grew 4 onions given the conditions of the problem. -/
theorem sara_grew_four_onions :
  let total := 18
  let sallys := 5
  let freds := 9
  saras_onions total sallys freds = 4 := by
  sorry

end sara_grew_four_onions_l222_22221


namespace quadratic_solutions_l222_22269

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

-- State the theorem
theorem quadratic_solutions (b : ℝ) :
  (∀ x, f b x = x^2 + b*x - 5) →  -- Definition of f
  (-b/(2:ℝ) = 2) →               -- Axis of symmetry condition
  (∀ x, f b x = 2*x - 13 ↔ x = 2 ∨ x = 4) :=
by sorry


end quadratic_solutions_l222_22269


namespace remaining_flavors_to_try_l222_22211

def ice_cream_flavors (total : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : Prop :=
  tried_two_years_ago = total / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  tried_two_years_ago + tried_last_year + 25 = total

theorem remaining_flavors_to_try
  (total : ℕ)
  (tried_two_years_ago : ℕ)
  (tried_last_year : ℕ)
  (h : ice_cream_flavors total tried_two_years_ago tried_last_year)
  (h_total : total = 100) :
  25 = total - (tried_two_years_ago + tried_last_year) :=
by
  sorry

end remaining_flavors_to_try_l222_22211


namespace quadratic_equation_problem_l222_22217

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end quadratic_equation_problem_l222_22217


namespace inscribed_quadrilateral_theorem_l222_22242

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop := sorry

/-- Function to get the radius of the circumscribed circle of a quadrilateral -/
def circumscribed_radius (q : Quadrilateral) : ℝ := sorry

/-- Predicate to check if a point is inside a quadrilateral -/
def is_inside (P : Point) (q : Quadrilateral) : Prop := sorry

/-- Function to divide a quadrilateral into four parts given an internal point -/
def divide_quadrilateral (q : Quadrilateral) (P : Point) : 
  (Quadrilateral × Quadrilateral × Quadrilateral × Quadrilateral) := sorry

theorem inscribed_quadrilateral_theorem 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h_inscribed : is_inscribed ABCD) 
  (h_inside : is_inside P ABCD) :
  let (APB, BPC, CPD, APD) := divide_quadrilateral ABCD P
  (is_inscribed APB ∧ is_inscribed BPC ∧ is_inscribed CPD) →
  (circumscribed_radius APB = circumscribed_radius BPC) →
  (circumscribed_radius BPC = circumscribed_radius CPD) →
  (is_inscribed APD ∧ circumscribed_radius APD = circumscribed_radius APB) :=
by sorry

end inscribed_quadrilateral_theorem_l222_22242


namespace bin_drawing_probability_l222_22288

def bin_probability : ℚ :=
  let total_balls : ℕ := 20
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let drawn_balls : ℕ := 4
  let favorable_outcomes : ℕ := (Nat.choose black_balls 2) * (Nat.choose white_balls 2)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem bin_drawing_probability : bin_probability = 135 / 323 := by
  sorry

end bin_drawing_probability_l222_22288


namespace cube_side_ratio_l222_22285

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 36 → a / b = 6 := by
  sorry

end cube_side_ratio_l222_22285


namespace selected_student_in_range_l222_22234

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (n : ℕ) : ℕ :=
  firstSelected + (n - 1) * (totalStudents / sampleSize)

/-- Theorem: The selected student number in the range 33 to 48 is 39 -/
theorem selected_student_in_range (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) :
  totalStudents = 800 →
  sampleSize = 50 →
  firstSelected = 7 →
  ∃ n : ℕ, systematicSample totalStudents sampleSize firstSelected n ∈ Set.Icc 33 48 ∧
           systematicSample totalStudents sampleSize firstSelected n = 39 :=
by
  sorry


end selected_student_in_range_l222_22234


namespace items_after_price_drop_l222_22296

/-- Calculates the number of items that can be purchased after a price drop -/
theorem items_after_price_drop (original_price : ℚ) (original_quantity : ℕ) (new_price : ℚ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check items_after_price_drop (4 : ℚ) 15 (3 : ℚ)

end items_after_price_drop_l222_22296


namespace function_value_at_two_l222_22216

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ≠ 0,
    prove that f(2) = -1 -/
theorem function_value_at_two (f : ℝ → ℝ) 
    (h : ∀ (x : ℝ), x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
    f 2 = -1 := by
  sorry

end function_value_at_two_l222_22216


namespace power_of_fraction_l222_22246

theorem power_of_fraction :
  (5 : ℚ) / 6 ^ 4 = 625 / 1296 := by
  sorry

end power_of_fraction_l222_22246


namespace inequality_proof_l222_22253

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l222_22253


namespace polynomial_simplification_l222_22257

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 2*x^9 - x^6) =
  15*x^13 - 19*x^12 - 6*x^11 + 6*x^10 - 4*x^9 - 3*x^7 + 2*x^6 := by
  sorry

end polynomial_simplification_l222_22257


namespace correct_number_of_guesses_l222_22215

/-- The number of valid guesses for three prizes with given digits -/
def number_of_valid_guesses : ℕ :=
  let digits : List ℕ := [2, 2, 2, 4, 4, 4, 4]
  let min_price : ℕ := 1
  let max_price : ℕ := 9999
  420

/-- Theorem stating that the number of valid guesses is 420 -/
theorem correct_number_of_guesses :
  number_of_valid_guesses = 420 := by sorry

end correct_number_of_guesses_l222_22215


namespace max_min_f_on_interval_l222_22277

def f (x : ℝ) := x^3 - 12*x

theorem max_min_f_on_interval :
  let a := -3
  let b := 5
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 65 ∧ min = -16 := by
  sorry

end max_min_f_on_interval_l222_22277


namespace reciprocal_of_negative_one_third_l222_22273

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (∀ z : ℚ, x * z = 1 → z = y) := by
  sorry

end reciprocal_of_negative_one_third_l222_22273


namespace range_of_a_l222_22299

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 3, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the set of a values that satisfy the conditions
def A : Set ℝ := {a | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem statement
theorem range_of_a : A = Set.Icc (-1) 1 ∪ Set.Ioi 3 := by sorry

end range_of_a_l222_22299


namespace g_2010_equals_one_l222_22204

/-- A function satisfying the given properties -/
def g_function (g : ℝ → ℝ) : Prop :=
  (∀ x > 0, g x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → g (x - y) = (g (x * y) + 1) ^ (1/3)) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x - y = x * y ∧ x * y = 2010)

/-- The main theorem stating that g(2010) = 1 -/
theorem g_2010_equals_one (g : ℝ → ℝ) (h : g_function g) : g 2010 = 1 := by
  sorry

end g_2010_equals_one_l222_22204


namespace roberts_chocolates_l222_22206

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : nickel_chocolates = 2 → difference = 7 → nickel_chocolates + difference = 9 := by
  sorry

end roberts_chocolates_l222_22206


namespace product_upper_bound_l222_22290

theorem product_upper_bound (x y z t : ℝ) 
  (h_order : x ≤ y ∧ y ≤ z ∧ z ≤ t) 
  (h_sum : x*y + x*z + x*t + y*z + y*t + z*t = 1) : 
  x*t < 1/3 ∧ ∀ C, (∀ a b c d, a ≤ b ∧ b ≤ c ∧ c ≤ d → 
    a*b + a*c + a*d + b*c + b*d + c*d = 1 → a*d < C) → 1/3 ≤ C :=
by sorry

end product_upper_bound_l222_22290


namespace min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l222_22209

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 1 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_3_plus_2sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2*Real.sqrt 2 :=
by sorry

end min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l222_22209


namespace expression_simplification_l222_22286

theorem expression_simplification :
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt (1/2) - |-1/2| = 1 + (Real.sqrt 2) / 2 := by
  sorry

end expression_simplification_l222_22286


namespace det_A_eq_48_l222_22272

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end det_A_eq_48_l222_22272


namespace equation_solutions_count_l222_22267

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), 
    (∀ θ ∈ s, 0 < θ ∧ θ ≤ π ∧ 4 - 2 * Real.sin θ + 3 * Real.cos (2 * θ) = 0) ∧
    s.card = 4 :=
by sorry

end equation_solutions_count_l222_22267


namespace parallelogram_area_error_l222_22220

/-- Calculates the percentage error in the area of a parallelogram given measurement errors -/
theorem parallelogram_area_error (x y : ℝ) (z : Real) (hx : x > 0) (hy : y > 0) (hz : 0 < z ∧ z < pi) :
  let actual_area := x * y * Real.sin z
  let measured_area := (1.05 * x) * (1.07 * y) * Real.sin z
  (measured_area - actual_area) / actual_area * 100 = 12.35 := by
sorry


end parallelogram_area_error_l222_22220


namespace range_of_c_l222_22245

def A (c : ℝ) := {x : ℝ | |x - 1| < c}
def B := {x : ℝ | |x - 3| > 4}

theorem range_of_c :
  ∀ c : ℝ, (A c ∩ B = ∅) ↔ c ≤ 2 :=
sorry

end range_of_c_l222_22245


namespace vending_machine_drinks_l222_22295

def arcade_problem (num_machines : ℕ) (sections_per_machine : ℕ) (drinks_left : ℕ) (drinks_dispensed : ℕ) : Prop :=
  let drinks_per_section : ℕ := drinks_left + drinks_dispensed
  let drinks_per_machine : ℕ := drinks_per_section * sections_per_machine
  let total_drinks : ℕ := drinks_per_machine * num_machines
  total_drinks = 840

theorem vending_machine_drinks :
  arcade_problem 28 6 3 2 := by
  sorry

end vending_machine_drinks_l222_22295


namespace probability_odd_divisor_15_factorial_l222_22282

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc (p, e) => acc * (e + 1)) 1

def count_odd_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  (factors.filter (fun (p, _) => p ≠ 2)).foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem probability_odd_divisor_15_factorial :
  let f15 := factorial 15
  let factors := prime_factorization f15
  let total_divisors := count_divisors factors
  let odd_divisors := count_odd_divisors factors
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by sorry

end probability_odd_divisor_15_factorial_l222_22282


namespace fast_area_scientific_notation_l222_22200

/-- The area of the reflecting surface of the FAST radio telescope in square meters -/
def fast_area : ℝ := 250000

/-- Scientific notation representation of the FAST area -/
def fast_area_scientific : ℝ := 2.5 * (10 ^ 5)

/-- Theorem stating that the FAST area is equal to its scientific notation representation -/
theorem fast_area_scientific_notation : fast_area = fast_area_scientific := by
  sorry

end fast_area_scientific_notation_l222_22200


namespace circumcircle_area_of_triangle_l222_22201

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the area of its circumcircle is π/2 under certain conditions. -/
theorem circumcircle_area_of_triangle (a b c : Real) (S : Real) :
  a = 1 →
  4 * S = b^2 + c^2 - 1 →
  (∃ A B C : Real, 
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    S = (1/2) * b * c * Real.sin A) →
  (∃ R : Real, R > 0 ∧ π * R^2 = π/2) :=
by sorry

end circumcircle_area_of_triangle_l222_22201


namespace factorization_of_4x_squared_minus_16_l222_22271

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_4x_squared_minus_16_l222_22271


namespace z_in_terms_of_x_and_y_l222_22239

theorem z_in_terms_of_x_and_y (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y ≠ 2*x) 
  (h : 1/x - 2/y = 1/z) : z = x*y/(y - 2*x) := by
  sorry

end z_in_terms_of_x_and_y_l222_22239


namespace discount_calculation_l222_22243

/-- Calculates the discounted price for a purchase with a percentage discount on amounts over a threshold --/
def discountedPrice (itemCount : ℕ) (itemPrice : ℚ) (discountPercentage : ℚ) (discountThreshold : ℚ) : ℚ :=
  let totalPrice := itemCount * itemPrice
  let amountOverThreshold := max (totalPrice - discountThreshold) 0
  let discountAmount := discountPercentage * amountOverThreshold
  totalPrice - discountAmount

/-- Proves that for a purchase of 7 items at $200 each, with a 10% discount on amounts over $1000, the final cost is $1360 --/
theorem discount_calculation :
  discountedPrice 7 200 0.1 1000 = 1360 := by
  sorry

end discount_calculation_l222_22243


namespace P_bounds_l222_22214

/-- A convex n-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Transformation that replaces triangles ABC and ACD with ABD and BCD -/
def transformation (n : ℕ) (g : ConvexNGon n) : ConvexNGon n := sorry

/-- P(n) is the minimum number of transformations required to convert any partition into any other partition -/
def P (n : ℕ) : ℕ := sorry

/-- Main theorem about bounds on P(n) -/
theorem P_bounds (n : ℕ) (g : ConvexNGon n) :
  P n ≥ n - 3 ∧
  P n ≤ 2*n - 7 ∧
  (n ≥ 13 → P n ≤ 2*n - 10) :=
sorry

end P_bounds_l222_22214


namespace article_sale_loss_percentage_l222_22292

theorem article_sale_loss_percentage 
  (cost : ℝ) 
  (original_price : ℝ) 
  (discounted_price : ℝ) 
  (h1 : original_price = cost * 1.35)
  (h2 : discounted_price = original_price * (2/3)) :
  (cost - discounted_price) / cost * 100 = 10 := by
sorry

end article_sale_loss_percentage_l222_22292


namespace no_primes_divisible_by_56_l222_22263

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0 ∨ m = 1

theorem no_primes_divisible_by_56 :
  ∀ p : ℕ, is_prime p → p % 56 ≠ 0 :=
by
  sorry

end no_primes_divisible_by_56_l222_22263


namespace certain_number_plus_two_l222_22229

theorem certain_number_plus_two (x : ℝ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end certain_number_plus_two_l222_22229


namespace chromosome_stability_processes_l222_22294

-- Define the type for physiological processes
inductive PhysiologicalProcess
  | Mitosis
  | Amitosis
  | Meiosis
  | Fertilization

-- Define the set of all physiological processes
def allProcesses : Set PhysiologicalProcess :=
  {PhysiologicalProcess.Mitosis, PhysiologicalProcess.Amitosis, 
   PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization}

-- Define the property of maintaining chromosome stability and continuity
def maintainsChromosomeStability (p : PhysiologicalProcess) : Prop :=
  match p with
  | PhysiologicalProcess.Meiosis => true
  | PhysiologicalProcess.Fertilization => true
  | _ => false

-- Theorem: The set of processes that maintain chromosome stability
--          is equal to {Meiosis, Fertilization}
theorem chromosome_stability_processes :
  {p ∈ allProcesses | maintainsChromosomeStability p} = 
  {PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization} :=
by
  sorry


end chromosome_stability_processes_l222_22294


namespace star_polygon_angles_l222_22279

/-- Given a star polygon where the sum of five angles is 500°, 
    prove that the sum of the other five angles is 140°. -/
theorem star_polygon_angles (p q r s t A B C D E : ℝ) 
  (h1 : p + q + r + s + t = 500) 
  (h2 : A + B + C + D + E = x) : x = 140 := by
  sorry

end star_polygon_angles_l222_22279


namespace no_x_squared_term_l222_22289

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 2) * (x^2 - 5*a*x + 1) = x^3 + (-9*a)*x + 2) → a = 2/5 := by
sorry

end no_x_squared_term_l222_22289


namespace center_of_mass_position_l222_22266

/-- A system of disks with specific properties -/
structure DiskSystem where
  -- The ratio of radii of two adjacent disks
  ratio : ℝ
  -- The radius of the largest disk
  largest_radius : ℝ
  -- Assertion that the ratio is 1/2
  ratio_is_half : ratio = 1/2
  -- Assertion that the largest radius is 2 meters
  largest_radius_is_two : largest_radius = 2

/-- The center of mass of the disk system -/
noncomputable def center_of_mass (ds : DiskSystem) : ℝ := sorry

/-- Theorem stating that the center of mass is at 6/7 meters from the largest disk's center -/
theorem center_of_mass_position (ds : DiskSystem) : 
  center_of_mass ds = 6/7 := by sorry

end center_of_mass_position_l222_22266


namespace square_equals_self_l222_22256

theorem square_equals_self (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end square_equals_self_l222_22256


namespace parent_son_age_ratio_l222_22226

/-- The ratio of a parent's age to their son's age -/
def age_ratio (parent_age : ℕ) (son_age : ℕ) : ℚ :=
  parent_age / son_age

theorem parent_son_age_ratio :
  let parent_age : ℕ := 35
  let son_age : ℕ := 7
  age_ratio parent_age son_age = 5 := by
  sorry

end parent_son_age_ratio_l222_22226


namespace fraction_begins_with_0_239_l222_22274

/-- The infinite decimal a --/
def a : ℝ := 0.1234567891011

/-- The infinite decimal b --/
def b : ℝ := 0.51504948

/-- Theorem stating that the fraction a/b begins with 0.239 --/
theorem fraction_begins_with_0_239 (h1 : 0.515 < b) (h2 : b < 0.516) :
  0.239 * b ≤ a ∧ a < 0.24 * b := by sorry

end fraction_begins_with_0_239_l222_22274


namespace integral_curves_satisfy_differential_equation_l222_22268

/-- The differential equation in terms of x, y, dx, and dy -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  x * dx + y * dy + (x * dy - y * dx) / (x^2 + y^2) = 0

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + y^2) / 2 - y * Real.arctan (x / y) = C

/-- Theorem stating that the integral_curve satisfies the differential_equation -/
theorem integral_curves_satisfy_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
    x^2 + y^2 > 0 →
    integral_curve x y C →
    ∃ (dx dy : ℝ), differential_equation x y dx dy :=
sorry

end integral_curves_satisfy_differential_equation_l222_22268


namespace initial_breads_l222_22284

/-- The number of thieves -/
def num_thieves : ℕ := 8

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 5

/-- The function representing how many breads remain after each thief -/
def breads_after_thief (n : ℕ) (b : ℚ) : ℚ :=
  if n = 0 then b else (1/2) * (breads_after_thief (n-1) b) - (1/2)

/-- The theorem stating the initial number of breads -/
theorem initial_breads :
  ∃ (b : ℚ), breads_after_thief num_thieves b = remaining_breads ∧ b = 1535 := by
  sorry

end initial_breads_l222_22284


namespace count_not_divisible_by_5_and_7_l222_22250

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem count_not_divisible_by_5_and_7 :
  count_not_divisible 499 5 7 = 343 := by sorry

end count_not_divisible_by_5_and_7_l222_22250


namespace water_amount_l222_22287

/-- Represents the recipe ratios and quantities -/
structure Recipe where
  water : ℝ
  sugar : ℝ
  cranberry : ℝ
  water_sugar_ratio : water = 5 * sugar
  sugar_cranberry_ratio : sugar = 3 * cranberry
  cranberry_amount : cranberry = 4

/-- Proves that the amount of water needed is 60 cups -/
theorem water_amount (r : Recipe) : r.water = 60 := by
  sorry

end water_amount_l222_22287


namespace divisibility_by_101_l222_22236

theorem divisibility_by_101 : ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ (201300 + 10 * x + y) % 101 = 0 := by
  sorry

end divisibility_by_101_l222_22236


namespace fractional_equation_solution_l222_22203

theorem fractional_equation_solution :
  ∃ (x : ℚ), (1 - x) / (2 - x) - 1 = (3 * x - 4) / (x - 2) ∧ x = 5 / 3 :=
by sorry

end fractional_equation_solution_l222_22203


namespace complex_expression_equals_negative_two_l222_22249

theorem complex_expression_equals_negative_two :
  (2023 * Real.pi) ^ 0 + (-1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) = -2 := by
  sorry

end complex_expression_equals_negative_two_l222_22249


namespace parabola_y_intercepts_l222_22232

theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 6 * y + 1
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ f y₁ = 0 ∧ f y₂ = 0 ∧ ∀ y, f y = 0 → y = y₁ ∨ y = y₂ :=
by sorry

end parabola_y_intercepts_l222_22232


namespace matrix_power_eight_l222_22248

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_eight :
  A^8 = !![16, 0; 0, 16] := by sorry

end matrix_power_eight_l222_22248


namespace prob_second_unqualified_given_first_is_one_fifth_l222_22252

/-- A box containing disinfectant bottles -/
structure DisinfectantBox where
  total : ℕ
  qualified : ℕ
  unqualified : ℕ

/-- The probability of drawing an unqualified bottle for the second time,
    given that an unqualified bottle was drawn for the first time -/
def prob_second_unqualified_given_first (box : DisinfectantBox) : ℚ :=
  (box.unqualified - 1 : ℚ) / (box.total - 1)

/-- The main theorem -/
theorem prob_second_unqualified_given_first_is_one_fifth
  (box : DisinfectantBox)
  (h_total : box.total = 6)
  (h_qualified : box.qualified = 4)
  (h_unqualified : box.unqualified = 2) :
  prob_second_unqualified_given_first box = 1/5 :=
sorry

end prob_second_unqualified_given_first_is_one_fifth_l222_22252


namespace three_digit_number_problem_l222_22225

theorem three_digit_number_problem :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (x : ℚ) - (x : ℚ) / 10 = 201.6 :=
by
  sorry

end three_digit_number_problem_l222_22225


namespace seeds_per_row_in_top_bed_l222_22213

theorem seeds_per_row_in_top_bed (
  top_beds : Nat
  ) (bottom_beds : Nat)
  (rows_per_top_bed : Nat)
  (rows_per_bottom_bed : Nat)
  (seeds_per_row_bottom : Nat)
  (total_seeds : Nat)
  (h1 : top_beds = 2)
  (h2 : bottom_beds = 2)
  (h3 : rows_per_top_bed = 4)
  (h4 : rows_per_bottom_bed = 3)
  (h5 : seeds_per_row_bottom = 20)
  (h6 : total_seeds = 320) :
  (total_seeds - (bottom_beds * rows_per_bottom_bed * seeds_per_row_bottom)) / (top_beds * rows_per_top_bed) = 25 := by
  sorry

end seeds_per_row_in_top_bed_l222_22213


namespace x_range_proof_l222_22247

theorem x_range_proof (x : ℝ) : 
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → 1/(Real.sin θ)^2 + 4/(Real.cos θ)^2 ≥ |2*x - 1|) 
  ↔ -4 ≤ x ∧ x ≤ 5 := by
sorry

end x_range_proof_l222_22247


namespace smallest_positive_integer_ending_in_9_divisible_by_26_l222_22293

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_26 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 26 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 26 = 0 → m ≥ n :=
by sorry

end smallest_positive_integer_ending_in_9_divisible_by_26_l222_22293


namespace distance_to_y_axis_l222_22281

/-- The distance from point P(x, -5) to the y-axis is 10 units, given that the distance
    from P to the x-axis is half the distance from P to the y-axis. -/
theorem distance_to_y_axis (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  let dist_to_x_axis := |P.2|
  let dist_to_y_axis := |P.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis → dist_to_y_axis = 10 :=
by
  sorry

end distance_to_y_axis_l222_22281


namespace rectangular_field_width_l222_22255

/-- Proves that a rectangular field with length 7/5 times its width and perimeter 432 meters has a width of 90 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 432 →
  perimeter = 2 * length + 2 * width →
  width = 90 := by
sorry

end rectangular_field_width_l222_22255


namespace fourth_root_equation_solution_l222_22227

theorem fourth_root_equation_solution (x : ℝ) :
  (x^3)^(1/4) = 81 * 81^(1/16) → x = 243 * 9^(1/3) := by
  sorry

end fourth_root_equation_solution_l222_22227


namespace wire_cutting_l222_22224

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 30 →
  difference = 2 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 14 := by
sorry

end wire_cutting_l222_22224


namespace video_game_players_l222_22202

theorem video_game_players (lives_per_player : ℕ) (total_lives : ℕ) (h1 : lives_per_player = 8) (h2 : total_lives = 64) :
  total_lives / lives_per_player = 8 :=
by sorry

end video_game_players_l222_22202


namespace solution_set_f_gt_2_min_a_for_full_solution_set_l222_22212

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

-- Theorem for part (II)
theorem min_a_for_full_solution_set (a : ℝ) :
  (∀ x, f x ≤ (3/2)*a^2 - a) ↔ a ≥ 5/3 :=
sorry

end solution_set_f_gt_2_min_a_for_full_solution_set_l222_22212


namespace min_bailing_rate_calculation_l222_22260

-- Define the problem parameters
def distance_to_shore : Real := 2 -- miles
def water_intake_rate : Real := 15 -- gallons per minute
def max_water_capacity : Real := 60 -- gallons
def rowing_speed : Real := 3 -- miles per hour

-- Define the theorem
theorem min_bailing_rate_calculation :
  let time_to_shore := distance_to_shore / rowing_speed * 60 -- Convert to minutes
  let total_water_intake := water_intake_rate * time_to_shore
  let water_to_bail := total_water_intake - max_water_capacity
  let min_bailing_rate := water_to_bail / time_to_shore
  min_bailing_rate = 13.5 := by
  sorry


end min_bailing_rate_calculation_l222_22260


namespace fraction_divisibility_l222_22276

theorem fraction_divisibility (a b n : ℕ) (hodd : Odd n) 
  (hnum : n ∣ (a^n + b^n)) (hden : n ∣ (a + b)) : 
  n ∣ ((a^n + b^n) / (a + b)) := by
  sorry

end fraction_divisibility_l222_22276


namespace lemonade_stand_profit_l222_22254

/-- Calculate the profit from a lemonade stand -/
theorem lemonade_stand_profit 
  (price_per_cup : ℕ) 
  (cups_sold : ℕ) 
  (lemon_cost sugar_cost cup_cost : ℕ) : 
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost) = 66 :=
by
  sorry

#check lemonade_stand_profit 4 21 10 5 3

end lemonade_stand_profit_l222_22254


namespace sqrt_equation_solution_l222_22265

theorem sqrt_equation_solution (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt (4 + x) + 3 * Real.sqrt (4 - x) = 5 * Real.sqrt 6 →
  x = Real.sqrt 43 / 9 := by
  sorry

end sqrt_equation_solution_l222_22265


namespace unique_number_with_two_perfect_square_increments_l222_22264

theorem unique_number_with_two_perfect_square_increments : 
  ∃! n : ℕ, n > 1000 ∧ 
    ∃ a b : ℕ, (n + 79 = a^2) ∧ (n + 204 = b^2) ∧ 
    n = 3765 :=
by sorry

end unique_number_with_two_perfect_square_increments_l222_22264


namespace suitcase_electronics_weight_l222_22278

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 5 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 2) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 9) = 10 / 4) -- New ratio after removing 9 pounds of clothes
  : E = 9 := by
  sorry

end suitcase_electronics_weight_l222_22278
