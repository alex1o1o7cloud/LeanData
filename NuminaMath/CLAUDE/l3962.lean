import Mathlib

namespace inequality_solution_set_l3962_396279

-- Define the solution set for the inequality
def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then { x | -1 < x ∧ x < 1 / (m + 3) }
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then { x | 1 / (m + 3) < x ∧ x < -1 }
  else if m = -3 then { x | x > -1 }
  else { x | x < -1 ∨ x > 1 / (m + 3) }

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  { x : ℝ | (m + 3) * x - 1 > 0 } = solution_set m := by
  sorry

end inequality_solution_set_l3962_396279


namespace alphametic_puzzle_impossibility_l3962_396266

theorem alphametic_puzzle_impossibility : ¬ ∃ (f : Char → Nat),
  (∀ x y : Char, x ≠ y → f x ≠ f y) ∧
  (∀ x : Char, x ∈ ['K', 'O', 'T', 'U', 'C', 'E', 'N', 'W', 'Y'] → f x ∈ Set.range (Fin.val : Fin 9 → Nat)) ∧
  (f 'K' * f 'O' * f 'T' = f 'U' * f 'C' * f 'E' * f 'N' * f 'W' * f 'Y') :=
by sorry


end alphametic_puzzle_impossibility_l3962_396266


namespace certain_number_problem_l3962_396225

theorem certain_number_problem : ∃! x : ℕ+, 220030 = (x + 445) * (2 * (x - 445)) + 30 := by
  sorry

end certain_number_problem_l3962_396225


namespace rahul_savings_l3962_396214

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℚ) : 
  (1/3 : ℚ) * nsc = (1/2 : ℚ) * ppf →
  nsc + ppf = 180000 →
  ppf = 72000 := by
sorry

end rahul_savings_l3962_396214


namespace solution_set_inequality_l3962_396255

theorem solution_set_inequality (x : ℝ) :
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) :=
by sorry

end solution_set_inequality_l3962_396255


namespace determine_fifth_subject_marks_l3962_396235

/-- Given the marks of a student in 4 subjects and the average marks of 5 subjects,
    this theorem proves that the marks in the fifth subject can be uniquely determined. -/
theorem determine_fifth_subject_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 70)
  (h2 : mathematics = 63)
  (h3 : chemistry = 63)
  (h4 : biology = 65)
  (h5 : average = 68.2)
  : ∃! physics : ℕ,
    (english + mathematics + physics + chemistry + biology : ℚ) / 5 = average :=
by sorry

end determine_fifth_subject_marks_l3962_396235


namespace tan_alpha_one_third_implies_cos_2alpha_over_expression_l3962_396209

theorem tan_alpha_one_third_implies_cos_2alpha_over_expression (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos (2*α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8/15 := by
  sorry

end tan_alpha_one_third_implies_cos_2alpha_over_expression_l3962_396209


namespace alternating_exponent_inequality_l3962_396215

theorem alternating_exponent_inequality (n : ℕ) (h : n ≥ 1) :
  2^(3^n) > 3^(2^(n-1)) := by
  sorry

end alternating_exponent_inequality_l3962_396215


namespace science_fiction_books_l3962_396284

theorem science_fiction_books (pages_per_book : ℕ) (total_pages : ℕ) (h1 : pages_per_book = 478) (h2 : total_pages = 3824) :
  total_pages / pages_per_book = 8 := by
  sorry

end science_fiction_books_l3962_396284


namespace equation_roots_iff_q_condition_l3962_396245

/-- The equation x^4 + qx^3 + 2x^2 + qx + 4 = 0 has at least two distinct negative real roots
    if and only if q ≤ 3/√2 -/
theorem equation_roots_iff_q_condition (q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 4 = 0 ∧
    x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 4 = 0) ↔
  q ≤ 3 / Real.sqrt 2 :=
sorry

end equation_roots_iff_q_condition_l3962_396245


namespace sqrt_product_property_l3962_396216

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_property_l3962_396216


namespace optimal_production_value_l3962_396289

/-- Represents the production plan for products A and B -/
structure ProductionPlan where
  a : ℝ  -- Amount of product A in kg
  b : ℝ  -- Amount of product B in kg

/-- Calculates the total value of a production plan -/
def totalValue (plan : ProductionPlan) : ℝ :=
  600 * plan.a + 400 * plan.b

/-- Checks if a production plan is feasible given the raw material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  4 * plan.a + 2 * plan.b ≤ 100 ∧  -- Raw material A constraint
  2 * plan.a + 3 * plan.b ≤ 120    -- Raw material B constraint

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { a := 7.5, b := 35 }

theorem optimal_production_value :
  (∀ plan : ProductionPlan, isFeasible plan → totalValue plan ≤ totalValue optimalPlan) ∧
  isFeasible optimalPlan ∧
  totalValue optimalPlan = 18500 := by
  sorry

end optimal_production_value_l3962_396289


namespace age_sum_after_ten_years_l3962_396202

theorem age_sum_after_ten_years 
  (kareem_age : ℕ) 
  (son_age : ℕ) 
  (h1 : kareem_age = 42) 
  (h2 : son_age = 14) 
  (h3 : kareem_age = 3 * son_age) : 
  (kareem_age + 10) + (son_age + 10) = 76 := by
sorry

end age_sum_after_ten_years_l3962_396202


namespace apple_juice_percentage_is_40_percent_l3962_396258

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  apples : Nat
  appleJuice : Nat
  bananas : Nat
  bananaJuice : Nat

/-- Calculates the percentage of apple juice in a blend -/
def appleJuicePercentage (yield : JuiceYield) : Rat :=
  let appleJuicePerFruit := yield.appleJuice / yield.apples
  let bananaJuicePerFruit := yield.bananaJuice / yield.bananas
  let totalJuice := appleJuicePerFruit + bananaJuicePerFruit
  appleJuicePerFruit / totalJuice

/-- Theorem: The percentage of apple juice in the blend is 40% -/
theorem apple_juice_percentage_is_40_percent (yield : JuiceYield) 
    (h1 : yield.apples = 5)
    (h2 : yield.appleJuice = 10)
    (h3 : yield.bananas = 4)
    (h4 : yield.bananaJuice = 12) : 
  appleJuicePercentage yield = 2/5 := by
  sorry

#eval (2 : Rat) / 5

end apple_juice_percentage_is_40_percent_l3962_396258


namespace arrangements_count_l3962_396282

def num_tour_groups : ℕ := 4
def num_scenic_spots : ℕ := 4

/-- The number of arrangements for tour groups choosing scenic spots -/
def num_arrangements : ℕ :=
  (num_tour_groups.choose 2) * (num_scenic_spots * (num_scenic_spots - 1) * (num_scenic_spots - 2))

theorem arrangements_count :
  num_arrangements = 144 := by sorry

end arrangements_count_l3962_396282


namespace complement_of_A_l3962_396224

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A :
  (U \ A) = {2, 4, 5} := by sorry

end complement_of_A_l3962_396224


namespace remainder_of_2367905_div_5_l3962_396276

theorem remainder_of_2367905_div_5 : 2367905 % 5 = 0 := by
  sorry

end remainder_of_2367905_div_5_l3962_396276


namespace S_five_three_l3962_396292

-- Define the operation ∘
def S (a b : ℕ) : ℕ := 4 * a + 3 * b

-- Theorem statement
theorem S_five_three : S 5 3 = 29 := by
  sorry

end S_five_three_l3962_396292


namespace expression_evaluation_l3962_396231

theorem expression_evaluation : 5 + 15 / 3 - 2^2 * 4 = -6 := by
  sorry

end expression_evaluation_l3962_396231


namespace height_comparison_l3962_396267

theorem height_comparison (ashis_height babji_height : ℝ) 
  (h : babji_height = ashis_height * (1 - 0.2)) :
  (ashis_height - babji_height) / babji_height = 0.25 := by
sorry

end height_comparison_l3962_396267


namespace complex_number_properties_l3962_396281

theorem complex_number_properties (z : ℂ) (h : z = -1/2 + Complex.I * (Real.sqrt 3 / 2)) : 
  z^3 = 1 ∧ z^2 + z + 1 = 0 := by sorry

end complex_number_properties_l3962_396281


namespace problem_solution_l3962_396237

theorem problem_solution : (-1)^2022 + |(-2)^3 + (-3)^2| - (-1/4 + 1/6) * (-24) = 0 := by
  sorry

end problem_solution_l3962_396237


namespace sine_cosine_parity_l3962_396206

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sine_cosine_parity (sine cosine : ℝ → ℝ) 
  (h1 : ∀ x, sine (-x) = -(sine x)) 
  (h2 : ∀ x, cosine (-x) = cosine x) : 
  is_odd_function sine ∧ is_even_function cosine := by
  sorry

end sine_cosine_parity_l3962_396206


namespace power_product_sum_equality_l3962_396295

theorem power_product_sum_equality : (3^5 * 6^3) + 3^3 = 52515 := by
  sorry

end power_product_sum_equality_l3962_396295


namespace susie_piggy_bank_total_l3962_396253

/-- Calculates the total amount in Susie's piggy bank after two years -/
def piggy_bank_total (initial_amount : ℝ) (first_year_addition : ℝ) (second_year_addition : ℝ) (interest_rate : ℝ) : ℝ :=
  let first_year_total := (initial_amount + initial_amount * first_year_addition) * (1 + interest_rate)
  let second_year_total := (first_year_total + first_year_total * second_year_addition) * (1 + interest_rate)
  second_year_total

/-- Theorem stating that Susie's piggy bank total after two years is $343.98 -/
theorem susie_piggy_bank_total :
  piggy_bank_total 200 0.2 0.3 0.05 = 343.98 := by
  sorry

end susie_piggy_bank_total_l3962_396253


namespace A_3_1_equals_13_l3962_396277

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1_equals_13 : A 3 1 = 13 := by
  sorry

end A_3_1_equals_13_l3962_396277


namespace sqrt_fraction_equals_sixteen_l3962_396222

theorem sqrt_fraction_equals_sixteen :
  let eight : ℕ := 2^3
  let four : ℕ := 2^2
  ∀ x : ℝ, x = (((eight^10 + four^10) : ℝ) / (eight^4 + four^11 : ℝ))^(1/2) → x = 16 := by
sorry

end sqrt_fraction_equals_sixteen_l3962_396222


namespace sports_meeting_formation_l3962_396217

/-- The number of performers in the initial formation -/
def initial_performers : ℕ := sorry

/-- The number of performers after adding 16 -/
def after_addition : ℕ := initial_performers + 16

/-- The number of performers after 15 leave -/
def after_leaving : ℕ := after_addition - 15

theorem sports_meeting_formation :
  (∃ n : ℕ, initial_performers = 8 * n) ∧ 
  (∃ m : ℕ, after_addition = m * m) ∧
  (∃ k : ℕ, after_leaving = k * k) →
  initial_performers = 48 := by sorry

end sports_meeting_formation_l3962_396217


namespace inequality_proof_l3962_396207

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end inequality_proof_l3962_396207


namespace f_composition_l3962_396274

def f (x : ℝ) := 2 * x + 1

theorem f_composition (x : ℝ) : f (2 * x - 1) = 4 * x - 1 := by
  sorry

end f_composition_l3962_396274


namespace P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l3962_396226

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Theorem 1
theorem P_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem 2
theorem P_parallel_to_y_axis (a : ℝ) :
  P a = (4, 8) ↔ (P a).1 = 4 :=
sorry

-- Theorem 3
theorem P_second_quadrant_equidistant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2022 = 2021 :=
sorry

end P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l3962_396226


namespace product_evaluation_l3962_396268

theorem product_evaluation (n : ℤ) (h : n = 3) : 
  (n - 4) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = -5040 := by
  sorry

end product_evaluation_l3962_396268


namespace log_inequality_l3962_396264

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x / (2 + x) := by
  sorry

end log_inequality_l3962_396264


namespace quadratic_factorization_l3962_396293

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x, 25 * x^2 - 115 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -55 := by
sorry

end quadratic_factorization_l3962_396293


namespace no_linear_term_condition_l3962_396212

/-- For a polynomial (x-m)(x-n), the condition for it to not contain a linear term in x is m + n = 0. -/
theorem no_linear_term_condition (x m n : ℝ) : 
  (∀ (a b c : ℝ), (x - m) * (x - n) = a * x^2 + c → b = 0) ↔ m + n = 0 :=
by sorry

end no_linear_term_condition_l3962_396212


namespace correct_arrangements_l3962_396251

/-- The number of people in the row -/
def n : ℕ := 8

/-- The number of special people (A, B, C, D, E) -/
def k : ℕ := 5

/-- Function to calculate the number of arrangements -/
def count_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  count_arrangements n k = 11520 := by sorry

end correct_arrangements_l3962_396251


namespace yearly_increase_fraction_l3962_396263

theorem yearly_increase_fraction (initial_value final_value : ℝ) (f : ℝ) 
    (h1 : initial_value = 51200)
    (h2 : final_value = 64800)
    (h3 : initial_value * (1 + f)^2 = final_value) :
  f = 0.125 := by
  sorry

end yearly_increase_fraction_l3962_396263


namespace sum_other_y_coordinates_specific_parallelogram_l3962_396201

/-- A parallelogram with two opposite corners given -/
structure Parallelogram where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The sum of y-coordinates of the other two vertices of the parallelogram -/
def sumOtherYCoordinates (p : Parallelogram) : ℝ :=
  (p.corner1.2 + p.corner2.2)

theorem sum_other_y_coordinates_specific_parallelogram :
  let p := Parallelogram.mk (2, 15) (8, -6)
  sumOtherYCoordinates p = 9 := by
  sorry

#check sum_other_y_coordinates_specific_parallelogram

end sum_other_y_coordinates_specific_parallelogram_l3962_396201


namespace expression_evaluation_l3962_396265

theorem expression_evaluation : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0 := by
  sorry

end expression_evaluation_l3962_396265


namespace net_growth_rate_calculation_l3962_396271

/-- Given birth and death rates per certain number of people and an initial population,
    calculate the net growth rate as a percentage. -/
theorem net_growth_rate_calculation 
  (birth_rate death_rate : ℕ) 
  (initial_population : ℕ) 
  (birth_rate_val : birth_rate = 32)
  (death_rate_val : death_rate = 11)
  (initial_population_val : initial_population = 1000) :
  (birth_rate - death_rate : ℝ) / initial_population * 100 = 2.1 := by
  sorry

end net_growth_rate_calculation_l3962_396271


namespace geometric_sequence_seventh_term_l3962_396241

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given a geometric sequence {aₙ} satisfying a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
  sorry


end geometric_sequence_seventh_term_l3962_396241


namespace sibling_ages_l3962_396275

theorem sibling_ages (sister_age brother_age : ℕ) : 
  (brother_age - 2 = 2 * (sister_age - 2)) →
  (brother_age - 8 = 5 * (sister_age - 8)) →
  (sister_age = 10 ∧ brother_age = 18) :=
by sorry

end sibling_ages_l3962_396275


namespace circle_and_line_properties_l3962_396254

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure Line where
  k : ℝ
  b : ℝ

-- Define the problem conditions
def circle_conditions (C : Circle) : Prop :=
  let (x, y) := C.center
  x > 0 ∧ y > 0 ∧  -- Center is in the first quadrant
  3 * x = y ∧      -- Center lies on the line 3x - y = 0
  C.radius = y ∧   -- Circle is tangent to x-axis
  (2 * Real.sqrt 7) ^ 2 = 4 * (C.radius ^ 2 - x ^ 2)  -- Chord length condition

def line_intersects_circle (l : Line) (C : Circle) : Prop :=
  ∃ (x y : ℝ), l.k * x - y - 2 * l.k + 5 = 0 ∧
                (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2

-- Theorem statement
theorem circle_and_line_properties :
  ∀ (C : Circle) (l : Line),
    circle_conditions C →
    line_intersects_circle l C →
    (∀ (x y : ℝ), (x - 1) ^ 2 + (y - 3) ^ 2 = 9 ↔ (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2) ∧
    (∃ (k : ℝ), l.k = k ∧ l.b = 5 - 2 * k) ∧
    (∃ (l_shortest : Line), 
      l_shortest.k = -1/2 ∧ 
      l_shortest.b = 6 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (d d' : ℝ), 
          d = (abs (l_shortest.k * C.center.1 - C.center.2 + l_shortest.b)) / Real.sqrt (l_shortest.k ^ 2 + 1) ∧
          d' = (abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1) ∧
          d < d') ∧
    (∃ (shortest_chord : ℝ), shortest_chord = 4 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (chord : ℝ), 
          chord = 2 * Real.sqrt (C.radius ^ 2 - ((abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1)) ^ 2) ∧
          chord > shortest_chord) :=
sorry

end circle_and_line_properties_l3962_396254


namespace tim_has_33_books_l3962_396290

/-- The number of books Tim has, given the initial conditions -/
def tims_books (benny_initial : ℕ) (sandy_received : ℕ) (total : ℕ) : ℕ :=
  total - (benny_initial - sandy_received)

/-- Theorem stating that Tim has 33 books under the given conditions -/
theorem tim_has_33_books :
  tims_books 24 10 47 = 33 := by
  sorry

end tim_has_33_books_l3962_396290


namespace stewart_farm_ratio_l3962_396208

/-- Represents the Stewart farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  total_horse_food : ℕ
  food_per_horse : ℕ

/-- Calculates the number of horses on the farm -/
def num_horses (f : Farm) : ℕ := f.total_horse_food / f.food_per_horse

/-- Calculates the ratio of sheep to horses as a pair of natural numbers -/
def sheep_to_horse_ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.sheep (num_horses f)
  (f.sheep / gcd, num_horses f / gcd)

/-- Theorem stating that for the given farm conditions, the sheep to horse ratio is 2:7 -/
theorem stewart_farm_ratio :
  let f : Farm := ⟨16, 12880, 230⟩
  sheep_to_horse_ratio f = (2, 7) := by
  sorry

end stewart_farm_ratio_l3962_396208


namespace base9_to_base10_conversion_l3962_396219

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The base-9 representation of the number -/
def base9Number : List Nat := [7, 4, 8, 2]

theorem base9_to_base10_conversion :
  base9ToBase10 base9Number = 2149 := by
  sorry

end base9_to_base10_conversion_l3962_396219


namespace probability_matches_given_l3962_396230

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def pens_bought : ℕ := 2

def probability_no_defective (total : ℕ) (defective : ℕ) (bought : ℕ) : ℚ :=
  (Nat.choose (total - defective) bought : ℚ) / (Nat.choose total bought : ℚ)

theorem probability_matches_given :
  probability_no_defective total_pens defective_pens pens_bought = 5 / 14 :=
by sorry

end probability_matches_given_l3962_396230


namespace gilbert_cricket_ratio_l3962_396280

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time the temperature is 90°F -/
def temp_90_fraction : ℚ := 4/5

/-- The total number of crickets eaten over the entire period -/
def total_crickets : ℕ := 72

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 8

theorem gilbert_cricket_ratio :
  crickets_100 / crickets_90 = 2 :=
sorry

end gilbert_cricket_ratio_l3962_396280


namespace binomial_20_19_l3962_396227

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_20_19_l3962_396227


namespace fixed_point_parabola_l3962_396210

theorem fixed_point_parabola :
  ∀ (k : ℝ), 3 * (5 : ℝ)^2 + k * 5 - 5 * k = 75 := by
  sorry

end fixed_point_parabola_l3962_396210


namespace tulip_arrangement_l3962_396256

/-- The number of red tulips needed for the smile -/
def smile_tulips : ℕ := 18

/-- The number of yellow tulips for the background is 9 times the number of red tulips in the smile -/
def background_tulips : ℕ := 9 * smile_tulips

/-- The total number of tulips needed -/
def total_tulips : ℕ := 196

/-- The number of red tulips needed for each eye -/
def eye_tulips : ℕ := 8

theorem tulip_arrangement : 
  2 * eye_tulips + smile_tulips + background_tulips = total_tulips :=
sorry

end tulip_arrangement_l3962_396256


namespace NaNO3_formed_l3962_396244

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def balancedReaction : Reaction :=
  { reactants := [
      { name := "NH4NO3", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NaNO3", moles := 1 },
      { name := "NH3", moles := 1 },
      { name := "H2O", moles := 1 }
    ]
  }

def initialNH4NO3 : Compound :=
  { name := "NH4NO3", moles := 3 }

def initialNaOH : Compound :=
  { name := "NaOH", moles := 3 }

/-- Calculates the moles of a product formed in a reaction -/
def molesFormed (reaction : Reaction) (initialReactants : List Compound) (product : String) : ℝ :=
  sorry

theorem NaNO3_formed :
  molesFormed balancedReaction [initialNH4NO3, initialNaOH] "NaNO3" = 3 := by
  sorry

end NaNO3_formed_l3962_396244


namespace number_operations_l3962_396203

theorem number_operations (n y : ℝ) : ((2 * n + y) / 2) - n = y / 2 := by
  sorry

end number_operations_l3962_396203


namespace term_206_of_specific_sequence_l3962_396223

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem term_206_of_specific_sequence :
  let a₁ := 10
  let a₂ := -10
  let r := a₂ / a₁
  geometric_sequence a₁ r 206 = -10 := by sorry

end term_206_of_specific_sequence_l3962_396223


namespace pocket_money_problem_l3962_396240

/-- Pocket money problem -/
theorem pocket_money_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 2300 →
  (a + b) / 2 = 3000 →
  (b + c) / 2 = 2100 →
  (c + d) / 2 = 2750 →
  a = b + 800 →
  d = 3900 := by
sorry

end pocket_money_problem_l3962_396240


namespace divisible_by_three_or_six_percentage_l3962_396260

theorem divisible_by_three_or_six_percentage (n : Nat) : 
  n = 200 → 
  (((Finset.filter (fun x => x % 3 = 0 ∨ x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / n) * 100 = 33 := by
  sorry

end divisible_by_three_or_six_percentage_l3962_396260


namespace cow_count_l3962_396261

/-- Given a group of cows and hens, prove that the number of cows is 4 when the total number of legs
    is 8 more than twice the number of heads. -/
theorem cow_count (cows hens : ℕ) : 
  (4 * cows + 2 * hens = 2 * (cows + hens) + 8) → cows = 4 := by
  sorry

end cow_count_l3962_396261


namespace missing_carton_dimension_l3962_396250

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the carton with one unknown dimension -/
def carton (x : ℝ) : BoxDimensions :=
  { length := 25, width := x, height := 60 }

/-- Represents the soap box dimensions -/
def soapBox : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- The maximum number of soap boxes that can fit in the carton -/
def maxSoapBoxes : ℕ := 300

theorem missing_carton_dimension :
  ∃ x : ℝ, boxVolume (carton x) = (maxSoapBoxes : ℝ) * boxVolume soapBox ∧ x = 48 := by
  sorry

end missing_carton_dimension_l3962_396250


namespace two_digit_numbers_with_gcd_lcm_l3962_396218

theorem two_digit_numbers_with_gcd_lcm (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 →
  Nat.gcd a b = 8 →
  Nat.lcm a b = 96 →
  a + b = 56 := by
sorry

end two_digit_numbers_with_gcd_lcm_l3962_396218


namespace quadratic_equation_solution_l3962_396204

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end quadratic_equation_solution_l3962_396204


namespace complex_modulus_3_plus_2i_l3962_396287

theorem complex_modulus_3_plus_2i : 
  Complex.abs (3 + 2 * Complex.I) = Real.sqrt 13 := by sorry

end complex_modulus_3_plus_2i_l3962_396287


namespace angle_in_second_quadrant_l3962_396270

/-- Given an angle α in the second quadrant with P(x,4) on its terminal side and cos α = (1/5)x,
    prove that x = -3 and tan α = -4/3 -/
theorem angle_in_second_quadrant (α : Real) (x : Real) 
    (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
    (h2 : x < 0) -- P(x,4) is on the terminal side in the second quadrant
    (h3 : Real.cos α = (1/5) * x) -- Given condition
    : x = -3 ∧ Real.tan α = -4/3 := by
  sorry


end angle_in_second_quadrant_l3962_396270


namespace intersection_M_N_l3962_396213

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end intersection_M_N_l3962_396213


namespace quadratic_root_relation_l3962_396288

theorem quadratic_root_relation (b c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁^2 = -x₂) →
  (b^3 - 3*b*c - c^2 - c = 0) :=
by sorry

end quadratic_root_relation_l3962_396288


namespace cube_with_corners_removed_faces_l3962_396200

-- Define the properties of the cube
def cube_side_length : ℝ := 3
def small_cube_side_length : ℝ := 1
def initial_faces : ℕ := 6
def corners_in_cube : ℕ := 8
def new_faces_per_corner : ℕ := 3

-- Theorem statement
theorem cube_with_corners_removed_faces :
  initial_faces + corners_in_cube * new_faces_per_corner = 30 := by
  sorry

end cube_with_corners_removed_faces_l3962_396200


namespace min_value_inequality_l3962_396229

theorem min_value_inequality (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  1 / (x + 1) + 1 / y ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end min_value_inequality_l3962_396229


namespace hyperbola_eccentricity_l3962_396291

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the eccentricity of the hyperbola is √(17)/3 -/
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (F : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : b = 1)
  (h4 : F = (c, 0))
  (h5 : B = (0, 1))
  (h6 : A.1^2 / a^2 - A.2^2 / b^2 = 1)  -- A is on the hyperbola
  (h7 : A.1^2 = 4 * A.2)                -- A is on the parabola
  (h8 : (A.1 - B.1, A.2 - B.2) = 3 * (F.1 - A.1, F.2 - A.2))  -- BA = 3AF
  : c / a = Real.sqrt 17 / 3 := by
  sorry

end hyperbola_eccentricity_l3962_396291


namespace exists_m_between_alpha_beta_l3962_396238

theorem exists_m_between_alpha_beta (α β : ℝ) (h1 : 0 ≤ α) (h2 : α < β) (h3 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := by
  sorry

end exists_m_between_alpha_beta_l3962_396238


namespace product_and_reciprocal_relation_sum_l3962_396285

theorem product_and_reciprocal_relation_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b = 16 ∧ 1 / a = 3 / b → a + b = 16 * Real.sqrt 3 / 3 := by
  sorry

end product_and_reciprocal_relation_sum_l3962_396285


namespace polynomial_transformation_l3962_396236

theorem polynomial_transformation (x y : ℝ) : 
  x^3 - 6*x^2 + 11*x - 6 = 0 → 
  y = x + 1/x → 
  x^2*(y^2 + y - 6) = 0 := by
sorry

end polynomial_transformation_l3962_396236


namespace same_color_probability_l3962_396205

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles) / 
  Nat.choose total_marbles drawn_marbles = 8 / 399 := by
  sorry

end same_color_probability_l3962_396205


namespace adjacent_knights_probability_l3962_396298

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 5

/-- The probability that at least two of the chosen knights are sitting next to each other -/
def P : ℚ := 141505 / 142506

/-- Theorem stating the probability of adjacent chosen knights -/
theorem adjacent_knights_probability :
  (1 : ℚ) - (Nat.choose (total_knights - chosen_knights - (chosen_knights - 1)) (chosen_knights - 1) : ℚ) / 
  (Nat.choose total_knights chosen_knights : ℚ) = P := by sorry

end adjacent_knights_probability_l3962_396298


namespace tangent_line_at_2_range_of_m_for_three_roots_l3962_396278

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (A * x + B * y + C = 0) ↔ x = 2) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end tangent_line_at_2_range_of_m_for_three_roots_l3962_396278


namespace symposium_pair_selection_l3962_396220

theorem symposium_pair_selection (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 2) :
  Nat.choose n k = 435 := by
  sorry

end symposium_pair_selection_l3962_396220


namespace min_value_reciprocal_sum_l3962_396247

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end min_value_reciprocal_sum_l3962_396247


namespace percentage_of_number_l3962_396273

theorem percentage_of_number (n : ℝ) : n * 0.001 = 0.24 → n = 240 := by
  sorry

end percentage_of_number_l3962_396273


namespace largest_reciprocal_l3962_396243

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/4 → b = 5/3 → c = -1/6 → d = 7 → e = 3 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l3962_396243


namespace f_properties_l3962_396249

noncomputable section

def f (x : ℝ) : ℝ := Real.log x - (x - 1)^2 / 2

def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem f_properties :
  (∀ x y, 0 < x ∧ x < y ∧ y < phi → f x < f y) ∧
  (∀ x, 1 < x → f x < x - 1) ∧
  (∀ k, (∃ x₀, 1 < x₀ ∧ ∀ x, 1 < x ∧ x < x₀ → k * (x - 1) < f x) → k < 1) :=
sorry

end

end f_properties_l3962_396249


namespace zero_not_in_N_star_l3962_396246

-- Define the set of natural numbers
def N : Set ℕ := {n : ℕ | n > 0}

-- Define the set of positive integers (N*)
def N_star : Set ℕ := N

-- Define the set of rational numbers
def Q : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Theorem statement
theorem zero_not_in_N_star : 0 ∉ N_star := by
  sorry

end zero_not_in_N_star_l3962_396246


namespace hyperbola_equation_l3962_396211

/-- Given a hyperbola with the equation (x^2/a^2) - (y^2/b^2) = 1, where a > 0 and b > 0,
    if the eccentricity is 2 and the distance from the right focus to one of the asymptotes is √3,
    then the equation of the hyperbola is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a →  -- eccentricity is 2
  b = Real.sqrt 3 →  -- distance from right focus to asymptote is √3
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 := by
  sorry

end hyperbola_equation_l3962_396211


namespace pasture_rent_is_870_l3962_396283

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total rent for a pasture given rental information and a known payment --/
def calculate_total_rent (a b c : RentalInfo) (b_payment : ℕ) : ℕ :=
  let total_horse_months := a.horses * a.months + b.horses * b.months + c.horses * c.months
  let cost_per_horse_month := b_payment / (b.horses * b.months)
  cost_per_horse_month * total_horse_months

/-- Theorem stating that the total rent for the pasture is 870 --/
theorem pasture_rent_is_870 (a b c : RentalInfo) (h1 : a.horses = 12) (h2 : a.months = 8)
    (h3 : b.horses = 16) (h4 : b.months = 9) (h5 : c.horses = 18) (h6 : c.months = 6)
    (h7 : calculate_total_rent a b c 360 = 870) : 
  calculate_total_rent a b c 360 = 870 := by
  sorry

end pasture_rent_is_870_l3962_396283


namespace second_bag_popped_kernels_l3962_396259

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_popped_kernels 
  (bag1 : PopcornBag)
  (bag2 : PopcornBag)
  (bag3 : PopcornBag)
  (h1 : bag1.total = 75)
  (h2 : bag1.popped = 60)
  (h3 : bag2.total = 50)
  (h4 : bag3.total = 100)
  (h5 : bag3.popped = 82)
  (h6 : (popPercentage bag1 + popPercentage bag2 + popPercentage bag3) / 3 = 82) :
  bag2.popped = 42 := by
  sorry

#eval PopcornBag.popped { total := 50, popped := 42 }

end second_bag_popped_kernels_l3962_396259


namespace gcd_powers_of_two_l3962_396228

theorem gcd_powers_of_two : Nat.gcd (2^115 - 1) (2^105 - 1) = 2^10 - 1 := by
  sorry

end gcd_powers_of_two_l3962_396228


namespace factorial_quotient_l3962_396233

theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_quotient_l3962_396233


namespace solution_x_equals_three_l3962_396242

theorem solution_x_equals_three : ∃ (f : ℝ → ℝ), f 3 = 0 ∧ (∀ x, f x = 0 → x = 3) :=
by
  -- Proof goes here
  sorry

end solution_x_equals_three_l3962_396242


namespace exactly_two_sets_l3962_396239

/-- A structure representing a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ+
  length : ℕ+

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length : ℕ) * (2 * (s.start : ℕ) + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 256 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 256

theorem exactly_two_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end exactly_two_sets_l3962_396239


namespace solution_set_equivalence_l3962_396262

/-- Given that the solution set of ax^2 + bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≤ -3 or x ≥ -2} -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) →
  (∀ x, x^2 - b*x - a ≥ 0 ↔ x ≤ -3 ∨ x ≥ -2) :=
by sorry

end solution_set_equivalence_l3962_396262


namespace systematic_sampling_correct_l3962_396257

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  start : ℕ

/-- Generates the sequence of selected student numbers -/
def generate_sequence (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (fun i => s.start + i * (s.total_students / s.sample_size))

/-- Theorem: The systematic sampling of 6 students from 60 results in the correct sequence -/
theorem systematic_sampling_correct : 
  let s : SystematicSampling := ⟨60, 6, 6⟩
  generate_sequence s = [6, 16, 26, 36, 46, 56] := by
  sorry

#eval generate_sequence ⟨60, 6, 6⟩

end systematic_sampling_correct_l3962_396257


namespace price_increase_consumption_reduction_l3962_396296

/-- Theorem: If the price of a commodity increases by 25%, a person must reduce their consumption by 20% to maintain the same expenditure. -/
theorem price_increase_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := P * 1.25
  let new_consumption := C * 0.8
  new_price * new_consumption = P * C := by
  sorry

end price_increase_consumption_reduction_l3962_396296


namespace shelter_cats_l3962_396234

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 →
  cats / (dogs + 12) = 15 / 11 →
  cats = 45 := by
sorry

end shelter_cats_l3962_396234


namespace set_product_theorem_l3962_396286

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | 0 ≤ y}

-- Define the operation ×
def setProduct (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Theorem statement
theorem set_product_theorem :
  setProduct A B = {x | -1 ≤ x ∧ x < 0 ∨ 1 < x} :=
by sorry

end set_product_theorem_l3962_396286


namespace inscribed_circle_radius_l3962_396294

/-- Given a quarter circle sector with radius 5, the radius of the inscribed circle
    tangent to both radii and the arc is 5√2 - 5. -/
theorem inscribed_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  r * (1 + Real.sqrt 2) = 5 → 
  r = 5 * Real.sqrt 2 - 5 := by sorry

end inscribed_circle_radius_l3962_396294


namespace min_colors_needed_l3962_396248

/-- Represents a color assignment for hats and ribbons --/
structure ColorAssignment (n : ℕ) where
  hatColors : Fin n → Fin n
  ribbonColors : Fin n → Fin n → Fin n

/-- A valid color assignment satisfies the problem constraints --/
def isValidColorAssignment (n : ℕ) (ca : ColorAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors i) ∧
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors j) ∧
  (∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → ca.ribbonColors i j ≠ ca.ribbonColors i k)

/-- The main theorem: n colors are sufficient and necessary --/
theorem min_colors_needed (n : ℕ) (h : n ≥ 2) :
  (∃ ca : ColorAssignment n, isValidColorAssignment n ca) ∧
  (∀ m : ℕ, m < n → ¬∃ ca : ColorAssignment m, isValidColorAssignment m ca) :=
sorry

end min_colors_needed_l3962_396248


namespace power_sum_simplification_l3962_396269

theorem power_sum_simplification :
  (-1)^2006 - (-1)^2007 + 1^2008 + 1^2009 - 1^2010 = 3 := by
  sorry

end power_sum_simplification_l3962_396269


namespace root_product_equality_l3962_396232

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry


end root_product_equality_l3962_396232


namespace vampire_daily_victims_l3962_396272

-- Define the vampire's weekly blood requirement in gallons
def weekly_blood_requirement : ℚ := 7

-- Define the amount of blood sucked per person in pints
def blood_per_person : ℚ := 2

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Define the number of pints in a gallon
def pints_per_gallon : ℕ := 8

-- Theorem: The vampire needs to suck blood from 4 people per day
theorem vampire_daily_victims : 
  (weekly_blood_requirement / days_per_week * pints_per_gallon) / blood_per_person = 4 := by
  sorry


end vampire_daily_victims_l3962_396272


namespace monotonicity_condition_solution_set_l3962_396299

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 2*a

-- Theorem for monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (f a)) ↔ (a ≥ -1/2 ∨ a ≤ -5/2) := by sorry

-- Theorem for solution set of f(x) < 0
theorem solution_set (a : ℝ) :
  {x : ℝ | f a x < 0} = 
    if a = -1/2 then ∅ 
    else if a < -1/2 then Set.Ioo 1 (-2*a)
    else Set.Ioo (-2*a) 1 := by sorry

end monotonicity_condition_solution_set_l3962_396299


namespace dawns_lemonade_price_l3962_396297

/-- The price of Dawn's lemonade in cents -/
def dawns_price : ℕ := sorry

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The difference in earnings between Bea and Dawn in cents -/
def earnings_difference : ℕ := 26

theorem dawns_lemonade_price :
  dawns_price = 28 ∧
  bea_glasses * bea_price = dawn_glasses * dawns_price + earnings_difference :=
sorry

end dawns_lemonade_price_l3962_396297


namespace square_ratio_proof_l3962_396221

theorem square_ratio_proof (area_ratio : Rat) (a b c : ℕ) : 
  area_ratio = 50 / 98 →
  (a : Rat) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  (a : Rat) / c = 5 / 7 →
  a + b + c = 12 :=
by sorry

end square_ratio_proof_l3962_396221


namespace exponent_division_l3962_396252

theorem exponent_division (a : ℝ) : a^12 / a^6 = a^6 := by
  sorry

end exponent_division_l3962_396252
