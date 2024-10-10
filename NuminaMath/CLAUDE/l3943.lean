import Mathlib

namespace divisibility_implies_fraction_simplification_l3943_394367

theorem divisibility_implies_fraction_simplification (a b c : ℕ) :
  (100 * a + 10 * b + c) % 7 = 0 →
  ((10 * b + c + 16 * a) % 7 = 0 ∧ (10 * b + c - 61 * a) % 7 = 0) := by
  sorry

end divisibility_implies_fraction_simplification_l3943_394367


namespace katherines_bananas_l3943_394348

theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = 21 →
  total = apples + pears + bananas →
  bananas = 5 := by
sorry

end katherines_bananas_l3943_394348


namespace rationalize_denominator_l3943_394336

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end rationalize_denominator_l3943_394336


namespace fraction_division_simplification_l3943_394351

theorem fraction_division_simplification :
  (3 / 4) / (5 / 8) = 6 / 5 := by sorry

end fraction_division_simplification_l3943_394351


namespace segment_AB_length_l3943_394327

-- Define the points on the number line
def point_A : ℝ := -5
def point_B : ℝ := 2

-- Define the length of the segment
def segment_length (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem segment_AB_length :
  segment_length point_A point_B = 7 := by
  sorry

end segment_AB_length_l3943_394327


namespace no_cube_root_exists_l3943_394326

theorem no_cube_root_exists (n : ℤ) : ¬ ∃ k : ℤ, k^3 = 3*n^2 + 3*n + 7 := by
  sorry

end no_cube_root_exists_l3943_394326


namespace probability_two_girls_l3943_394370

/-- The probability of choosing two girls from a class with given composition -/
theorem probability_two_girls (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = girls + boys) 
  (h2 : total = 8) 
  (h3 : girls = 5) 
  (h4 : boys = 3) : 
  (Nat.choose girls 2 : ℚ) / (Nat.choose total 2) = 5 / 14 := by
  sorry

end probability_two_girls_l3943_394370


namespace crayon_factory_output_l3943_394314

/-- Calculates the number of boxes filled per hour in a crayon factory --/
def boxes_per_hour (num_colors : ℕ) (crayons_per_color_per_box : ℕ) (total_crayons_in_4_hours : ℕ) : ℕ :=
  let crayons_per_hour := total_crayons_in_4_hours / 4
  let crayons_per_box := num_colors * crayons_per_color_per_box
  crayons_per_hour / crayons_per_box

/-- Theorem stating that under given conditions, the factory fills 5 boxes per hour --/
theorem crayon_factory_output : 
  boxes_per_hour 4 2 160 = 5 := by
  sorry

end crayon_factory_output_l3943_394314


namespace system_solution_ratio_l3943_394343

theorem system_solution_ratio (a b x y : ℝ) (h1 : b ≠ 0) (h2 : 4*x - y = a) (h3 : 5*y - 20*x = b) : a / b = -1 / 5 := by
  sorry

end system_solution_ratio_l3943_394343


namespace quadrilateral_reconstruction_l3943_394346

/-- Given a quadrilateral ABCD with extended sides, prove that A can be expressed
    as a linear combination of A'', B'', C'', D'' with specific coefficients. -/
theorem quadrilateral_reconstruction
  (A B C D A'' B'' C'' D'' : ℝ × ℝ) -- Points in 2D space
  (h1 : A'' - A = 2 * (B - A))      -- AA'' = 2AB
  (h2 : B'' - B = 3 * (C - B))      -- BB'' = 3BC
  (h3 : C'' - C = 2 * (D - C))      -- CC'' = 2CD
  (h4 : D'' - D = 2 * (A - D)) :    -- DD'' = 2DA
  A = (1/6 : ℝ) • A'' + (1/9 : ℝ) • B'' + (1/9 : ℝ) • C'' + (1/18 : ℝ) • D'' := by
  sorry


end quadrilateral_reconstruction_l3943_394346


namespace sum_of_squared_coefficients_l3943_394347

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 4*x^2 + 3)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (20^2) + (0^2) + (15^2) = 750 :=
by sorry

end sum_of_squared_coefficients_l3943_394347


namespace jamie_flyer_earnings_l3943_394321

/-- Calculates Jamie's earnings from delivering flyers --/
def jamies_earnings (hourly_rate : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) (total_weeks : ℕ) : ℕ :=
  hourly_rate * days_per_week * hours_per_day * total_weeks

/-- Proves that Jamie's earnings after 6 weeks will be $360 --/
theorem jamie_flyer_earnings :
  jamies_earnings 10 2 3 6 = 360 := by
  sorry

#eval jamies_earnings 10 2 3 6

end jamie_flyer_earnings_l3943_394321


namespace zeros_after_decimal_for_40_pow_40_l3943_394337

/-- The number of zeros immediately following the decimal point in 1/(40^40) -/
def zeros_after_decimal (n : ℕ) : ℕ :=
  let base := 40
  let exponent := 40
  let denominator := base ^ exponent
  -- The actual computation of zeros is not implemented here
  sorry

/-- Theorem stating that the number of zeros after the decimal point in 1/(40^40) is 76 -/
theorem zeros_after_decimal_for_40_pow_40 : zeros_after_decimal 40 = 76 := by
  sorry

end zeros_after_decimal_for_40_pow_40_l3943_394337


namespace thomas_work_hours_l3943_394316

theorem thomas_work_hours 
  (total_hours : ℕ)
  (rebecca_hours : ℕ)
  (h1 : total_hours = 157)
  (h2 : rebecca_hours = 56) :
  ∃ (thomas_hours : ℕ),
    thomas_hours = 37 ∧
    ∃ (toby_hours : ℕ),
      toby_hours = 2 * thomas_hours - 10 ∧
      rebecca_hours = toby_hours - 8 ∧
      total_hours = thomas_hours + toby_hours + rebecca_hours :=
by sorry

end thomas_work_hours_l3943_394316


namespace diophantine_equation_solutions_l3943_394318

theorem diophantine_equation_solutions :
  ∀ m n k : ℕ, 2 * m + 3 * n = k^2 ↔
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5) :=
by sorry

end diophantine_equation_solutions_l3943_394318


namespace seven_balls_four_boxes_l3943_394386

/-- The number of ways to distribute n identical balls into k distinct boxes, leaving no box empty -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 20 ways to distribute 7 identical balls into 4 distinct boxes with no empty box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 20 := by
  sorry

end seven_balls_four_boxes_l3943_394386


namespace fraction_to_decimal_l3943_394306

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := by
  sorry

end fraction_to_decimal_l3943_394306


namespace successive_discounts_equivalence_l3943_394322

theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 ∧ 
  equivalent_discount = 0.235 →
  original_price * (1 - first_discount) * (1 - second_discount) = 
  original_price * (1 - equivalent_discount) := by
  sorry

end successive_discounts_equivalence_l3943_394322


namespace union_of_sets_l3943_394349

theorem union_of_sets : 
  let A : Set Nat := {1, 2, 4}
  let B : Set Nat := {2, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end union_of_sets_l3943_394349


namespace johnny_distance_when_met_l3943_394390

/-- The distance between Q and Y in km -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in km/hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in km/hour -/
def johnny_rate : ℝ := 4

/-- The time difference between Matthew's and Johnny's start in hours -/
def time_difference : ℝ := 1

/-- The distance Johnny walked when they met -/
def johnny_distance : ℝ := 24

theorem johnny_distance_when_met :
  let t := (total_distance - matthew_rate * time_difference) / (matthew_rate + johnny_rate)
  johnny_distance = johnny_rate * t :=
by sorry

end johnny_distance_when_met_l3943_394390


namespace choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l3943_394365

def num_boys : ℕ := 4
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls

-- Question 1
theorem choose_five_representatives : Nat.choose total_people 5 = 126 := by sorry

-- Question 2
theorem choose_five_with_specific_girl :
  (Nat.choose num_boys 2) * (Nat.choose (num_girls - 1) 2) = 36 := by sorry

-- Question 3
theorem choose_five_with_at_least_two_boys :
  (Nat.choose num_boys 2) * (Nat.choose num_girls 3) +
  (Nat.choose num_boys 3) * (Nat.choose num_girls 2) +
  (Nat.choose num_boys 4) * (Nat.choose num_girls 1) = 105 := by sorry

-- Question 4
theorem divide_into_three_groups :
  (Nat.choose total_people 4) * (Nat.choose (total_people - 4) 3) = 1260 := by sorry

end choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l3943_394365


namespace fish_tank_leakage_rate_l3943_394375

/-- Proves that the rate of leakage is 1.5 ounces per hour given the problem conditions -/
theorem fish_tank_leakage_rate 
  (bucket_capacity : ℝ) 
  (leakage_duration : ℝ) 
  (h1 : bucket_capacity = 36) 
  (h2 : leakage_duration = 12) 
  (h3 : bucket_capacity = 2 * (leakage_duration * leakage_rate)) : 
  leakage_rate = 1.5 := by
  sorry

#check fish_tank_leakage_rate

end fish_tank_leakage_rate_l3943_394375


namespace equation_solution_l3943_394380

theorem equation_solution : 
  ∃ x : ℝ, (45 * x) + (625 / 25) - (300 * 4) = 2950 + 1500 / (75 * 2) ∧ x = 4135 / 45 := by
  sorry

end equation_solution_l3943_394380


namespace basketball_scores_second_half_total_l3943_394324

/-- Represents the score of a team in a quarter -/
structure QuarterScore :=
  (score : ℕ)

/-- Represents the scores of a team for all four quarters -/
structure GameScore :=
  (q1 : QuarterScore)
  (q2 : QuarterScore)
  (q3 : QuarterScore)
  (q4 : QuarterScore)

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℚ), b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ (diff : ℤ), b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem statement -/
theorem basketball_scores_second_half_total
  (eagles : GameScore)
  (lions : GameScore)
  (h1 : eagles.q1.score = lions.q1.score)
  (h2 : eagles.q1.score + eagles.q2.score = lions.q1.score + lions.q2.score)
  (h3 : isGeometricSequence eagles.q1.score eagles.q2.score eagles.q3.score eagles.q4.score)
  (h4 : isArithmeticSequence lions.q1.score lions.q2.score lions.q3.score lions.q4.score)
  (h5 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score = 
        lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score + 1)
  (h6 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score ≤ 100)
  (h7 : lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score ≤ 100) :
  eagles.q3.score + eagles.q4.score + lions.q3.score + lions.q4.score = 109 :=
sorry

end basketball_scores_second_half_total_l3943_394324


namespace series_sum_equals_one_fourth_l3943_394329

/-- The sum of the infinite series ∑(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1))] is equal to 1/4 -/
theorem series_sum_equals_one_fourth :
  let a : ℕ → ℝ := λ n => (3 : ℝ)^n / (1 + (3 : ℝ)^n + (3 : ℝ)^(n+1) + (3 : ℝ)^(2*n+1))
  ∑' n, a n = 1/4 := by sorry

end series_sum_equals_one_fourth_l3943_394329


namespace total_cleaning_time_is_180_l3943_394384

/-- The total time Matt and Alex spend cleaning their cars -/
def total_cleaning_time (matt_outside : ℕ) : ℕ :=
  let matt_inside := matt_outside / 4
  let matt_total := matt_outside + matt_inside
  let alex_outside := matt_outside / 2
  let alex_inside := matt_inside * 2
  let alex_total := alex_outside + alex_inside
  matt_total + alex_total

/-- Theorem stating that the total cleaning time is 180 minutes -/
theorem total_cleaning_time_is_180 :
  total_cleaning_time 80 = 180 := by sorry

end total_cleaning_time_is_180_l3943_394384


namespace other_root_of_quadratic_l3943_394369

theorem other_root_of_quadratic (m : ℚ) :
  (3 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} →
  (-5/9 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} :=
by sorry

end other_root_of_quadratic_l3943_394369


namespace square_difference_l3943_394363

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l3943_394363


namespace distribute_five_to_three_l3943_394379

/-- The number of ways to distribute n students to k universities, 
    with each university admitting at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 students to 3 universities results in 150 different methods -/
theorem distribute_five_to_three : distribute_students 5 3 = 150 := by
  sorry

end distribute_five_to_three_l3943_394379


namespace car_speed_proof_l3943_394335

/-- Proves that given the conditions of the car journey, the initial speed must be 75 km/hr -/
theorem car_speed_proof (v : ℝ) : 
  v > 0 →
  (320 / (160 / v + 160 / 80) = 77.4193548387097) →
  v = 75 := by
sorry

end car_speed_proof_l3943_394335


namespace power_of_negative_one_2010_l3943_394385

theorem power_of_negative_one_2010 : ∃ x : ℕ, ((-1 : ℤ) ^ 2010 : ℤ) = x ∧ ∀ y : ℕ, y ≥ x :=
by sorry

end power_of_negative_one_2010_l3943_394385


namespace digit_equation_solution_l3943_394307

/-- Represents a base-ten digit -/
def Digit := Fin 10

/-- Checks if three digits are all different -/
def all_different (d1 d2 d3 : Digit) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (tens ones : Digit) : Nat :=
  10 * tens.val + ones.val

/-- Converts a digit to a three-digit number with all digits the same -/
def to_three_digit (d : Digit) : Nat :=
  111 * d.val

theorem digit_equation_solution :
  ∃ (V E A : Digit),
    all_different V E A ∧
    (to_two_digit V E) * (to_two_digit A E) = to_three_digit A ∧
    E.val + A.val + A.val + V.val = 26 := by
  sorry

end digit_equation_solution_l3943_394307


namespace arithmetic_square_root_of_16_l3943_394389

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_of_16_l3943_394389


namespace savings_proof_l3943_394350

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that savings are 4000 given the conditions -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 20000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end savings_proof_l3943_394350


namespace incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l3943_394392

/-- The ratio of the incircle radius to the circumcircle radius of a right triangle is at most √2 - 1 -/
theorem incircle_circumcircle_ratio_bound (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (a + b - c) / c ≤ Real.sqrt 2 - 1 :=
sorry

/-- The upper bound √2 - 1 is achievable for the ratio of incircle to circumcircle radius in a right triangle -/
theorem incircle_circumcircle_ratio_bound_tight :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b - c) / c = Real.sqrt 2 - 1 :=
sorry

end incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l3943_394392


namespace max_value_constrained_sum_l3943_394342

theorem max_value_constrained_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → 2*x + y + 2*z ≤ 2*a + b + 2*c) →
  2*a + b + 2*c = 3 :=
by sorry

end max_value_constrained_sum_l3943_394342


namespace solution_set_of_equation_l3943_394378

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end solution_set_of_equation_l3943_394378


namespace toucans_joined_l3943_394352

theorem toucans_joined (initial final joined : ℕ) : 
  initial = 2 → final = 3 → joined = final - initial :=
by sorry

end toucans_joined_l3943_394352


namespace erased_number_proof_l3943_394374

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 2 →
  (↑n * (↑n + 1) / 2 - 3) - x = (454 / 9 : ℚ) * (↑n - 1) →
  x = 107 :=
sorry

end erased_number_proof_l3943_394374


namespace inequality_solution_set_l3943_394396

/-- A point in the second quadrant has a negative x-coordinate and positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The solution set of the inequality (2-m)x + 2 > m with respect to x -/
def solution_set (m : ℝ) : Set ℝ := {x | (2 - m) * x + 2 > m}

theorem inequality_solution_set (m : ℝ) :
  second_quadrant (3 - m) 1 → solution_set m = {x | x < -1} := by sorry

end inequality_solution_set_l3943_394396


namespace binomial_prob_two_to_four_out_of_five_l3943_394317

/-- The probability of getting 2, 3, or 4 successes in 5 trials with probability 0.5 each -/
theorem binomial_prob_two_to_four_out_of_five (n : Nat) (p : Real) (X : Nat → Real) :
  n = 5 →
  p = 0.5 →
  (∀ k, X k = Nat.choose n k * p^k * (1 - p)^(n - k)) →
  X 2 + X 3 + X 4 = 25/32 :=
by sorry

end binomial_prob_two_to_four_out_of_five_l3943_394317


namespace complex_number_modulus_l3943_394313

theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end complex_number_modulus_l3943_394313


namespace bus_stop_time_l3943_394300

/-- Given a bus with speeds excluding and including stoppages, calculate the stop time per hour -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50)
  (h2 : speed_with_stops = 43) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 8.4 := by
  sorry

end bus_stop_time_l3943_394300


namespace roots_of_quadratic_l3943_394376

theorem roots_of_quadratic (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = 6 ∧ x₁ * x₂ = -7 := by
  sorry

end roots_of_quadratic_l3943_394376


namespace water_mixture_adjustment_l3943_394328

theorem water_mixture_adjustment (initial_volume : ℝ) (initial_water_percentage : ℝ) 
  (initial_acid_percentage : ℝ) (water_to_add : ℝ) (final_water_percentage : ℝ) 
  (final_acid_percentage : ℝ) : 
  initial_volume = 300 →
  initial_water_percentage = 0.60 →
  initial_acid_percentage = 0.40 →
  water_to_add = 100 →
  final_water_percentage = 0.70 →
  final_acid_percentage = 0.30 →
  (initial_volume * initial_water_percentage + water_to_add) / (initial_volume + water_to_add) = final_water_percentage ∧
  (initial_volume * initial_acid_percentage) / (initial_volume + water_to_add) = final_acid_percentage :=
by sorry

end water_mixture_adjustment_l3943_394328


namespace trig_expression_equals_sqrt_three_l3943_394355

theorem trig_expression_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_expression_equals_sqrt_three_l3943_394355


namespace chromatic_number_iff_k_constructible_l3943_394338

/-- A graph is k-constructible if it can be built up from K_k by repeatedly adding a new vertex
    and joining it to a k-clique in the existing graph. -/
def is_k_constructible (G : SimpleGraph V) (k : ℕ) : Prop :=
  sorry

theorem chromatic_number_iff_k_constructible (G : SimpleGraph V) (k : ℕ) :
  G.chromaticNumber ≥ k ↔ ∃ H : SimpleGraph V, H ≤ G ∧ is_k_constructible H k :=
sorry

end chromatic_number_iff_k_constructible_l3943_394338


namespace three_X_four_equals_ten_l3943_394308

-- Define the operation X
def X (a b : ℤ) : ℤ := 2*b + 5*a - a^2 - b

-- Theorem statement
theorem three_X_four_equals_ten : X 3 4 = 10 := by
  sorry

end three_X_four_equals_ten_l3943_394308


namespace vector_sum_l3943_394382

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_sum : vector_a + vector_b = (1, -2) := by sorry

end vector_sum_l3943_394382


namespace factor_expression_l3943_394360

theorem factor_expression (b : ℝ) : 180 * b^2 + 36 * b = 36 * b * (5 * b + 1) := by
  sorry

end factor_expression_l3943_394360


namespace min_magnitude_linear_combination_l3943_394341

/-- Given vectors a and b in ℝ², prove that the minimum magnitude of their linear combination c = xa + yb is √3, under specific conditions. -/
theorem min_magnitude_linear_combination (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : a • b = 1/2) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → 
  ‖x • a + y • b‖ ≥ min :=
sorry

end min_magnitude_linear_combination_l3943_394341


namespace power_division_equality_l3943_394358

theorem power_division_equality : 3^12 / 27^2 = 729 := by sorry

end power_division_equality_l3943_394358


namespace hypotenuse_product_squared_l3943_394356

/-- Right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagorean : side1^2 + side2^2 = hypotenuse^2

/-- The problem statement -/
theorem hypotenuse_product_squared
  (T₁ T₂ : RightTriangle)
  (h_area₁ : T₁.area = 2)
  (h_area₂ : T₂.area = 3)
  (h_side_congruent : T₁.side1 = T₂.side1)
  (h_side_double : T₁.side2 = 2 * T₂.side2) :
  (T₁.hypotenuse * T₂.hypotenuse)^2 = 325 := by
  sorry

end hypotenuse_product_squared_l3943_394356


namespace cube_volume_ratio_l3943_394371

-- Define the edge lengths
def cube1_edge_inches : ℝ := 4
def cube2_edge_feet : ℝ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem cube_volume_ratio :
  (cube1_edge_inches ^ 3) / ((cube2_edge_feet * feet_to_inches) ^ 3) = 1 / 216 := by
  sorry

end cube_volume_ratio_l3943_394371


namespace high_precision_census_suitability_l3943_394315

/-- Represents different types of surveys --/
inductive SurveyType
  | DestructiveTesting
  | WideScopePopulation
  | HighPrecisionRequired
  | LargeAudienceSampling

/-- Represents different survey methods --/
inductive SurveyMethod
  | Census
  | Sampling

/-- Defines the characteristics of a survey --/
structure Survey where
  type : SurveyType
  method : SurveyMethod

/-- Defines the suitability of a survey method for a given survey type --/
def is_suitable (s : Survey) : Prop :=
  match s.type, s.method with
  | SurveyType.HighPrecisionRequired, SurveyMethod.Census => true
  | SurveyType.DestructiveTesting, SurveyMethod.Sampling => true
  | SurveyType.WideScopePopulation, SurveyMethod.Sampling => true
  | SurveyType.LargeAudienceSampling, SurveyMethod.Sampling => true
  | _, _ => false

/-- Theorem: A survey requiring high precision is most suitable for a census method --/
theorem high_precision_census_suitability :
  ∀ (s : Survey), s.type = SurveyType.HighPrecisionRequired → 
  is_suitable { type := s.type, method := SurveyMethod.Census } = true :=
by
  sorry


end high_precision_census_suitability_l3943_394315


namespace factor_x_squared_minus_64_l3943_394330

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l3943_394330


namespace tangent_point_x_coordinate_l3943_394368

/-- Given a circle and a point on its tangent, prove the x-coordinate of the point. -/
theorem tangent_point_x_coordinate 
  (a : ℝ) -- x-coordinate of point P
  (h1 : (a + 2)^2 + 16 = ((2 : ℝ) * Real.sqrt 3)^2 + 4) -- P is on the tangent and tangent length is 2√3
  : a = -2 := by
  sorry

end tangent_point_x_coordinate_l3943_394368


namespace problem_solution_l3943_394397

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / Real.sqrt 7 := by
  sorry

end problem_solution_l3943_394397


namespace parities_of_E_10_11_12_l3943_394387

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => 2 * E (n + 2) + E n

theorem parities_of_E_10_11_12 :
  Even (E 10) ∧ Odd (E 11) ∧ Odd (E 12) := by
  sorry

end parities_of_E_10_11_12_l3943_394387


namespace cube_monotone_l3943_394301

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_monotone_l3943_394301


namespace rationalize_denominator_l3943_394373

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 3) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -3 ∧
    B = -2 ∧
    C = 0 ∧
    D = 1 ∧
    E = 30 ∧
    F = 12 := by
  sorry

end rationalize_denominator_l3943_394373


namespace jason_total_cards_l3943_394357

/-- The number of Pokemon cards Jason has after receiving new ones from Alyssa -/
def total_cards (initial_cards new_cards : ℕ) : ℕ :=
  initial_cards + new_cards

/-- Theorem stating that Jason's total cards is 900 given the initial and new card counts -/
theorem jason_total_cards :
  total_cards 676 224 = 900 := by
  sorry

end jason_total_cards_l3943_394357


namespace stratified_sampling_total_l3943_394325

/-- Calculates the total number of students sampled using stratified sampling -/
def totalSampleSize (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ) : ℕ :=
  let totalStudents := firstGradeTotal + secondGradeTotal + thirdGradeTotal
  (firstGradeSample * totalStudents) / firstGradeTotal

theorem stratified_sampling_total (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ)
    (h1 : firstGradeTotal = 600)
    (h2 : secondGradeTotal = 500)
    (h3 : thirdGradeTotal = 400)
    (h4 : firstGradeSample = 30) :
    totalSampleSize firstGradeTotal secondGradeTotal thirdGradeTotal firstGradeSample = 75 := by
  sorry

#eval totalSampleSize 600 500 400 30

end stratified_sampling_total_l3943_394325


namespace perpendicular_vectors_l3943_394333

theorem perpendicular_vectors (m : ℝ) : 
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (m, 1)
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → m = 7 := by
sorry

end perpendicular_vectors_l3943_394333


namespace angle_a1fb1_is_right_angle_l3943_394391

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Theorem: Angle A1FB1 is 90 degrees in a parabola -/
theorem angle_a1fb1_is_right_angle (parab : Parabola) 
  (focus : Point) 
  (directrix : ℝ) 
  (line : Line) 
  (a b : Point) 
  (a1 b1 : Point) :
  focus.x = parab.p / 2 →
  focus.y = 0 →
  directrix = -parab.p / 2 →
  parab.equation a.x a.y →
  parab.equation b.x b.y →
  line.p1 = focus →
  (line.p2 = a ∨ line.p2 = b) →
  a1.x = directrix →
  b1.x = directrix →
  a1.y = a.y →
  b1.y = b.y →
  -- The conclusion: ∠A1FB1 = 90°
  ∃ (angle : ℝ), angle = Real.pi / 2 :=
by
  sorry

end angle_a1fb1_is_right_angle_l3943_394391


namespace walking_problem_solution_l3943_394377

/-- Two people walking on a line between points A and B -/
def WalkingProblem (distance_AB : ℝ) : Prop :=
  ∃ (first_meeting second_meeting : ℝ),
    first_meeting = 5 ∧
    second_meeting = distance_AB - 4 ∧
    2 * distance_AB = first_meeting + second_meeting

theorem walking_problem_solution :
  WalkingProblem 11 := by sorry

end walking_problem_solution_l3943_394377


namespace sqrt_one_third_same_type_as_2sqrt3_l3943_394381

-- Define a function to check if a number is of the same type as 2√3
def isSameTypeAs2Sqrt3 (x : ℝ) : Prop :=
  ∃ (a : ℝ), x = a * Real.sqrt 3

-- Theorem statement
theorem sqrt_one_third_same_type_as_2sqrt3 :
  isSameTypeAs2Sqrt3 (Real.sqrt (1/3)) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 8) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 18) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 9) :=
by sorry

end sqrt_one_third_same_type_as_2sqrt3_l3943_394381


namespace five_circles_theorem_l3943_394303

/-- A circle in a plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A point in a plane -/
structure Point where
  -- We don't need to define the internal structure of a point for this problem

/-- Predicate to check if a point is on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is common to a list of circles -/
def CommonPoint (p : Point) (circles : List Circle) : Prop :=
  ∀ c ∈ circles, PointOnCircle p c

theorem five_circles_theorem (circles : List Circle) :
  circles.length = 5 →
  (∀ (subset : List Circle), subset.length = 4 ∧ subset ⊆ circles →
    ∃ (p : Point), CommonPoint p subset) →
  ∃ (p : Point), CommonPoint p circles := by
  sorry

end five_circles_theorem_l3943_394303


namespace estimate_comparison_l3943_394304

theorem estimate_comparison (x y a b : ℝ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : a > b) 
  (h4 : b > 0) : 
  (x + a) - (y - b) > x - y := by
  sorry

end estimate_comparison_l3943_394304


namespace inequality_transformation_l3943_394309

theorem inequality_transformation (a : ℝ) : 
  (∀ x, (1 - a) * x > 2 ↔ x < 2 / (1 - a)) → a > 1 :=
by sorry

end inequality_transformation_l3943_394309


namespace perpendicular_vectors_x_value_l3943_394339

def vector_a : ℝ × ℝ := (-5, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_x_value :
  perpendicular vector_a (vector_b x) → x = 10 := by
  sorry

end perpendicular_vectors_x_value_l3943_394339


namespace real_part_of_reciprocal_l3943_394383

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (z - Complex.I)).re = z.re / (2 - 2 * z.im) := by
  sorry

end real_part_of_reciprocal_l3943_394383


namespace rals_age_l3943_394399

/-- Given that Ral's age is twice Suri's age and Suri's age plus 3 years equals 16 years,
    prove that Ral's current age is 26 years. -/
theorem rals_age (suri_age : ℕ) (ral_age : ℕ) : 
  ral_age = 2 * suri_age → 
  suri_age + 3 = 16 → 
  ral_age = 26 :=
by
  sorry

end rals_age_l3943_394399


namespace point_M_coordinates_l3943_394364

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point M on the left half of x-axis
def point_M (a : ℝ) : Prop := a < 0

-- Define the tangent line from M to circle O
def tangent_line (a x y : ℝ) : Prop := 
  ∃ (t : ℝ), x = a * (1 - t^2) / (1 + t^2) ∧ y = 2 * a * t / (1 + t^2)

-- Define points A, B, and C
def point_A (a x y : ℝ) : Prop := circle_O x y ∧ tangent_line a x y
def point_B (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y
def point_C (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y ∧ ¬(point_B a x y)

-- Define the condition AB = BC
def equal_segments (a : ℝ) : Prop := 
  ∀ (xa ya xb yb xc yc : ℝ), 
    point_A a xa ya → point_B a xb yb → point_C a xc yc →
    (xa - xb)^2 + (ya - yb)^2 = (xb - xc)^2 + (yb - yc)^2

-- Theorem statement
theorem point_M_coordinates : 
  ∀ (a : ℝ), point_M a → equal_segments a → a = -4 :=
sorry

end point_M_coordinates_l3943_394364


namespace travel_time_ratio_l3943_394359

def time_NY_to_SF : ℝ := 24
def layover_time : ℝ := 16
def total_time : ℝ := 58

def time_NO_to_NY : ℝ := total_time - layover_time - time_NY_to_SF

theorem travel_time_ratio : time_NO_to_NY / time_NY_to_SF = 3 / 4 := by
  sorry

end travel_time_ratio_l3943_394359


namespace sum_faces_edges_vertices_l3943_394340

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where

/-- The number of faces in a rectangular prism. -/
def faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism. -/
def vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26. -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  faces rp + edges rp + vertices rp = 26 := by
  sorry

end sum_faces_edges_vertices_l3943_394340


namespace problem_solution_l3943_394320

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + 2

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2*x + k/x

theorem problem_solution :
  -- Part 1: Average rate of change
  (f 2 - f 0) / 2 = 2 ∧
  -- Part 2: Parallel tangent lines
  (∃ k : ℝ, (deriv f 1 = deriv (g k) 1) → k = 1) ∧
  -- Part 3: Tangent line equation
  (∃ a b : ℝ, (∀ x : ℝ, a*x + b = 10*x - 14) ∧
              f 2 = a*2 + b ∧
              deriv f 2 = a) :=
by sorry

end problem_solution_l3943_394320


namespace inequality_comparison_l3943_394372

theorem inequality_comparison : 
  (¬ (0 < -1/2)) ∧ 
  (¬ (4/5 < -6/7)) ∧ 
  (9/8 > 8/9) ∧ 
  (¬ (-4 > -3)) :=
by sorry

end inequality_comparison_l3943_394372


namespace solution_set_of_inequality_l3943_394395

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
by sorry

end solution_set_of_inequality_l3943_394395


namespace birthday_ratio_l3943_394361

def peters_candles : ℕ := 10
def ruperts_candles : ℕ := 35

def age_ratio (x y : ℕ) : ℚ := (x : ℚ) / (y : ℚ)

theorem birthday_ratio : 
  age_ratio ruperts_candles peters_candles = 7 / 2 := by
  sorry

end birthday_ratio_l3943_394361


namespace unique_integer_satisfying_equation_l3943_394393

theorem unique_integer_satisfying_equation :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 20200 ∧
  1 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 200⌉ := by
  sorry

end unique_integer_satisfying_equation_l3943_394393


namespace max_sum_cubes_l3943_394394

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (max : ℝ), max = 5 * Real.sqrt 5 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ max ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = max :=
by sorry

end max_sum_cubes_l3943_394394


namespace electric_blankets_sold_l3943_394388

/-- Represents the number of electric blankets sold -/
def electric_blankets : ℕ := sorry

/-- Represents the number of hot-water bottles sold -/
def hot_water_bottles : ℕ := sorry

/-- Represents the number of thermometers sold -/
def thermometers : ℕ := sorry

/-- The price of a thermometer in dollars -/
def thermometer_price : ℕ := 2

/-- The price of a hot-water bottle in dollars -/
def hot_water_bottle_price : ℕ := 6

/-- The price of an electric blanket in dollars -/
def electric_blanket_price : ℕ := 10

/-- The total sales for all items in dollars -/
def total_sales : ℕ := 1800

theorem electric_blankets_sold :
  (thermometer_price * thermometers + 
   hot_water_bottle_price * hot_water_bottles + 
   electric_blanket_price * electric_blankets = total_sales) ∧
  (thermometers = 7 * hot_water_bottles) ∧
  (hot_water_bottles = 2 * electric_blankets) →
  electric_blankets = 36 := by sorry

end electric_blankets_sold_l3943_394388


namespace complex_number_imaginary_part_l3943_394323

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 - Complex.I)
  Complex.im z = 4 → a = -7 := by
  sorry

end complex_number_imaginary_part_l3943_394323


namespace bus_driver_hours_l3943_394302

-- Define constants
def regular_rate : ℝ := 15
def regular_hours : ℝ := 40
def overtime_rate_factor : ℝ := 1.75
def total_compensation : ℝ := 976

-- Define functions
def overtime_rate : ℝ := regular_rate * overtime_rate_factor

def total_hours (overtime_hours : ℝ) : ℝ :=
  regular_hours + overtime_hours

def compensation (overtime_hours : ℝ) : ℝ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

-- Theorem to prove
theorem bus_driver_hours :
  ∃ (overtime_hours : ℝ),
    compensation overtime_hours = total_compensation ∧
    total_hours overtime_hours = 54 := by
  sorry

end bus_driver_hours_l3943_394302


namespace complex_modulus_problem_l3943_394305

theorem complex_modulus_problem (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : a + 2 * i = 2 - b * i) :
  Complex.abs (a + b * i) = 2 * Real.sqrt 2 := by
sorry

end complex_modulus_problem_l3943_394305


namespace systematic_sampling_removal_l3943_394332

theorem systematic_sampling_removal (total_students sample_size : ℕ) 
  (h1 : total_students = 1252)
  (h2 : sample_size = 50) :
  ∃ (removed : ℕ), 
    removed = 2 ∧ 
    (total_students - removed) % sample_size = 0 := by
  sorry

end systematic_sampling_removal_l3943_394332


namespace second_derivative_parametric_function_l3943_394345

noncomputable def x (t : ℝ) : ℝ := Real.cosh t

noncomputable def y (t : ℝ) : ℝ := (Real.sinh t) ^ (2/3)

theorem second_derivative_parametric_function (t : ℝ) :
  let x_t' := Real.sinh t
  let y_t' := (2 * Real.cosh t) / (3 * (Real.sinh t)^(1/3))
  let y_x' := y_t' / x_t'
  let y_x'_t' := -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 3)
  (y_x'_t' / x_t') = -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 4) :=
by sorry

end second_derivative_parametric_function_l3943_394345


namespace area_of_triangle_l3943_394310

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- State that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- State that F₁PF₂ forms a right angle
axiom right_angle_condition : right_angle left_focus P right_focus

-- The theorem to prove
theorem area_of_triangle : 
  ∃ (S : ℝ), S = 16 ∧ S = (1/2) * ‖P - left_focus‖ * ‖P - right_focus‖ :=
sorry

end area_of_triangle_l3943_394310


namespace seating_probability_l3943_394362

-- Define the number of boys in the class
def num_boys : ℕ := 9

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the derangement function for 4 elements
def derangement_4 : ℕ := 9

-- Define the probability we want to prove
def target_probability : ℚ := 1 / 32

-- Theorem statement
theorem seating_probability :
  (choose num_boys 3 * choose (num_boys - 3) 2 * derangement_4) / (Nat.factorial num_boys) = target_probability := by
  sorry

end seating_probability_l3943_394362


namespace min_product_of_three_min_product_is_neg_480_l3943_394334

def S : Finset Int := {-10, -7, -3, 1, 4, 6, 8}

theorem min_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z :=
by
  sorry

theorem min_product_is_neg_480 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = -480 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z →
   a * b * c ≤ x * y * z) :=
by
  sorry

end min_product_of_three_min_product_is_neg_480_l3943_394334


namespace three_f_value_l3943_394366

axiom f : ℝ → ℝ
axiom f_def : ∀ x > 0, f (3 * x) = 3 / (3 + x)

theorem three_f_value : ∀ x > 0, 3 * f x = 27 / (9 + x) := by sorry

end three_f_value_l3943_394366


namespace dollar_three_neg_one_l3943_394398

def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y - 5

theorem dollar_three_neg_one : dollar 3 (-1) = -5 := by
  sorry

end dollar_three_neg_one_l3943_394398


namespace imaginary_part_of_z_plus_reciprocal_l3943_394344

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 - I) :
  (z + z⁻¹).im = -1/2 := by sorry

end imaginary_part_of_z_plus_reciprocal_l3943_394344


namespace paint_contribution_is_360_l3943_394319

/-- Calculates the contribution of each person for the paint cost --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ)
  (jason_wall_area : ℚ) (jason_coats : ℕ) (jeremy_wall_area : ℚ) (jeremy_coats : ℕ) : ℚ :=
  let total_area := jason_wall_area * jason_coats + jeremy_wall_area * jeremy_coats
  let gallons_needed := (total_area / coverage_per_gallon).ceil
  let total_cost := gallons_needed * paint_cost_per_gallon
  total_cost / 2

/-- Theorem stating that each person's contribution is $360 --/
theorem paint_contribution_is_360 :
  calculate_contribution 45 400 1025 3 1575 2 = 360 := by
  sorry

end paint_contribution_is_360_l3943_394319


namespace train_bridge_crossing_time_l3943_394331

/-- Proves that a train of given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l3943_394331


namespace algorithm_output_l3943_394311

theorem algorithm_output (x : ℤ) (y z : ℕ) : 
  x = -3 → 
  y = Int.natAbs x → 
  z = 2^y - y → 
  z = 5 := by sorry

end algorithm_output_l3943_394311


namespace minimum_loads_is_nineteen_l3943_394353

/-- Represents the capacity of the washing machine -/
structure MachineCapacity where
  shirts : ℕ
  sweaters : ℕ
  socks : ℕ

/-- Represents the number of clothes to be washed -/
structure ClothesCount where
  white_shirts : ℕ
  colored_shirts : ℕ
  white_sweaters : ℕ
  colored_sweaters : ℕ
  white_socks : ℕ
  colored_socks : ℕ

/-- Calculates the number of loads required for a given type of clothing -/
def loadsForClothingType (clothes : ℕ) (capacity : ℕ) : ℕ :=
  (clothes + capacity - 1) / capacity

/-- Calculates the total number of loads required -/
def totalLoads (capacity : MachineCapacity) (clothes : ClothesCount) : ℕ :=
  let white_loads := max (loadsForClothingType clothes.white_shirts capacity.shirts)
                         (max (loadsForClothingType clothes.white_sweaters capacity.sweaters)
                              (loadsForClothingType clothes.white_socks capacity.socks))
  let colored_loads := max (loadsForClothingType clothes.colored_shirts capacity.shirts)
                           (max (loadsForClothingType clothes.colored_sweaters capacity.sweaters)
                                (loadsForClothingType clothes.colored_socks capacity.socks))
  white_loads + colored_loads

/-- Theorem: The minimum number of loads required is 19 -/
theorem minimum_loads_is_nineteen (capacity : MachineCapacity) (clothes : ClothesCount) :
  capacity.shirts = 3 ∧ capacity.sweaters = 2 ∧ capacity.socks = 4 ∧
  clothes.white_shirts = 9 ∧ clothes.colored_shirts = 12 ∧
  clothes.white_sweaters = 18 ∧ clothes.colored_sweaters = 20 ∧
  clothes.white_socks = 16 ∧ clothes.colored_socks = 24 →
  totalLoads capacity clothes = 19 := by
  sorry


end minimum_loads_is_nineteen_l3943_394353


namespace bisection_method_approximation_l3943_394312

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem bisection_method_approximation 
  (h_continuous : Continuous f)
  (h1 : f 0.64 < 0)
  (h2 : f 0.72 > 0)
  (h3 : f 0.68 < 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0.68 0.72) ∧ f x = 0 ∧ |x - 0.7| < 0.1 := by
  sorry


end bisection_method_approximation_l3943_394312


namespace triangle_angle_solution_l3943_394354

theorem triangle_angle_solution (a b c : ℝ) (h1 : a = 40)
  (h2 : b = 3 * y) (h3 : c = y + 10) (h4 : a + b + c = 180) : y = 32.5 := by
  sorry

end triangle_angle_solution_l3943_394354
