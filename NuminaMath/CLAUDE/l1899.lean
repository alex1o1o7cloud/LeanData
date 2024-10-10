import Mathlib

namespace valid_sequences_count_l1899_189989

/-- Represents a binary sequence with no consecutive 1s -/
inductive ValidSequence : Nat → Type
  | zero : ValidSequence 0
  | one : ValidSequence 1
  | appendZero : ValidSequence n → ValidSequence (n + 1)
  | appendOneZero : ValidSequence n → ValidSequence (n + 2)

/-- Counts the number of valid sequences of length n or less -/
def countValidSequences (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => countValidSequences (n+1) + countValidSequences n

theorem valid_sequences_count :
  countValidSequences 11 = 233 := by sorry

end valid_sequences_count_l1899_189989


namespace wood_measurement_l1899_189907

theorem wood_measurement (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ rope / 2 = x + 1) → 
  (1/2 : ℝ) * (x + 4.5) = x - 1 :=
by sorry

end wood_measurement_l1899_189907


namespace megan_songs_count_l1899_189922

/-- The number of songs Megan bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating the total number of songs Megan bought -/
theorem megan_songs_count :
  total_songs 2 8 7 = 70 := by
  sorry

end megan_songs_count_l1899_189922


namespace units_digit_of_m_squared_plus_3_to_m_l1899_189966

def m : ℕ := 2023^2 + 3^2023

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 5 := by
  sorry

end units_digit_of_m_squared_plus_3_to_m_l1899_189966


namespace factor_implies_m_equals_one_l1899_189990

theorem factor_implies_m_equals_one (m : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 42 = (x + 6) * k) →
  m = 1 := by
sorry

end factor_implies_m_equals_one_l1899_189990


namespace waste_recovery_analysis_l1899_189992

structure WasteData where
  m : ℕ
  a : ℝ
  freq1 : ℝ
  freq2 : ℝ
  freq5 : ℝ

def WasteAnalysis (data : WasteData) : Prop :=
  data.m > 0 ∧
  0.20 ≤ data.a ∧ data.a ≤ 0.30 ∧
  data.freq1 + data.freq2 + data.a + data.freq5 = 1 ∧
  data.freq1 = 0.05 ∧
  data.freq2 = 0.10 ∧
  data.freq5 = 0.15

theorem waste_recovery_analysis (data : WasteData) 
  (h : WasteAnalysis data) : 
  data.m = 20 ∧ 
  (∃ (median : ℝ), 4 ≤ median ∧ median < 5) ∧
  (∃ (avg : ℝ), avg ≥ 3) :=
sorry

end waste_recovery_analysis_l1899_189992


namespace total_people_all_tribes_l1899_189978

/-- Represents a tribe with cannoneers, women, and men -/
structure Tribe where
  cannoneers : ℕ
  women : ℕ
  men : ℕ

/-- Calculates the total number of people in a tribe -/
def total_people (t : Tribe) : ℕ := t.cannoneers + t.women + t.men

/-- Represents the conditions for Tribe A -/
def tribe_a : Tribe :=
  { cannoneers := 63,
    women := 2 * 63,
    men := 2 * (2 * 63) }

/-- Represents the conditions for Tribe B -/
def tribe_b : Tribe :=
  { cannoneers := 45,
    women := 45 / 3,
    men := 3 * (45 / 3) }

/-- Represents the conditions for Tribe C -/
def tribe_c : Tribe :=
  { cannoneers := 108,
    women := 108 / 2,
    men := 108 / 2 }

theorem total_people_all_tribes : 
  total_people tribe_a + total_people tribe_b + total_people tribe_c = 834 := by
  sorry

end total_people_all_tribes_l1899_189978


namespace min_value_sum_reciprocals_l1899_189950

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_sum_reciprocals (a m n : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (hmn : m * n > 0) (h_line : -2 * m - n + 1 = 0) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_sum_reciprocals_l1899_189950


namespace jacks_recycling_l1899_189988

/-- Proves the number of cans Jack recycled given the deposit amounts and quantities of other items --/
theorem jacks_recycling
  (bottle_deposit : ℚ)
  (can_deposit : ℚ)
  (glass_deposit : ℚ)
  (num_bottles : ℕ)
  (num_glass : ℕ)
  (total_earnings : ℚ)
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : glass_deposit = 15 / 100)
  (h4 : num_bottles = 80)
  (h5 : num_glass = 50)
  (h6 : total_earnings = 25) :
  (total_earnings - (num_bottles * bottle_deposit + num_glass * glass_deposit)) / can_deposit = 190 := by
  sorry

end jacks_recycling_l1899_189988


namespace cost_type_B_calculation_l1899_189906

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased. -/
def cost_type_B (total_books : ℕ) (price_A : ℕ) (price_B : ℕ) (x : ℕ) : ℕ :=
  price_B * (total_books - x)

/-- Theorem stating that the cost of purchasing type B books is 8(100-x) yuan -/
theorem cost_type_B_calculation (x : ℕ) (h : x ≤ 100) :
  cost_type_B 100 10 8 x = 8 * (100 - x) := by
  sorry

end cost_type_B_calculation_l1899_189906


namespace ellipse_a_plus_k_l1899_189991

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  focusA : Point
  focusB : Point
  passingPoint : Point

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

theorem ellipse_a_plus_k (e : Ellipse) :
  e.focusA = ⟨0, 1⟩ →
  e.focusB = ⟨0, -3⟩ →
  e.passingPoint = ⟨5, 0⟩ →
  e.a > 0 →
  e.b > 0 →
  satisfiesEllipseEquation e e.passingPoint →
  e.a + e.k = (Real.sqrt 26 + Real.sqrt 34 - 2) / 2 := by
  sorry

end ellipse_a_plus_k_l1899_189991


namespace pizzeria_sales_l1899_189971

theorem pizzeria_sales (small_price large_price total_sales small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_sales = 40)
  (h4 : small_count = 8) : 
  ∃ large_count : ℕ, 
    large_count = 3 ∧ 
    small_price * small_count + large_price * large_count = total_sales :=
by
  sorry

end pizzeria_sales_l1899_189971


namespace bee_population_theorem_bee_problem_solution_l1899_189995

/-- Represents the daily change in bee population -/
def daily_change (hatch_rate : ℕ) (loss_rate : ℕ) : ℤ :=
  hatch_rate - loss_rate

/-- Calculates the final bee population after a given number of days -/
def final_population (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_change hatch_rate loss_rate

/-- Theorem stating the relationship between initial population, hatch rate, loss rate, and final population -/
theorem bee_population_theorem (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) (final : ℕ) :
  final_population initial hatch_rate loss_rate days = final ↔ loss_rate = 899 :=
by
  sorry

#eval final_population 12500 3000 899 7  -- Should evaluate to 27201

/-- Main theorem proving the specific case in the problem -/
theorem bee_problem_solution :
  final_population 12500 3000 899 7 = 27201 :=
by
  sorry

end bee_population_theorem_bee_problem_solution_l1899_189995


namespace play_admission_receipts_l1899_189959

/-- Calculates the total admission receipts for a play -/
def totalAdmissionReceipts (totalPeople : ℕ) (adultPrice childPrice : ℕ) (children : ℕ) : ℕ :=
  let adults := totalPeople - children
  adults * adultPrice + children * childPrice

/-- Theorem: The total admission receipts for the play is $960 -/
theorem play_admission_receipts :
  totalAdmissionReceipts 610 2 1 260 = 960 := by
  sorry

end play_admission_receipts_l1899_189959


namespace highest_numbered_street_l1899_189911

/-- Represents the length of Apple Street in meters -/
def street_length : ℕ := 15000

/-- Represents the distance between intersections in meters -/
def intersection_distance : ℕ := 500

/-- Calculates the number of numbered intersecting streets -/
def numbered_intersections : ℕ :=
  (street_length / intersection_distance) - 2

/-- Proves that the highest-numbered street is the 28th Street -/
theorem highest_numbered_street :
  numbered_intersections = 28 := by
  sorry

end highest_numbered_street_l1899_189911


namespace playground_girls_l1899_189927

theorem playground_girls (total_children : ℕ) (boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
  sorry

end playground_girls_l1899_189927


namespace inequality_proof_l1899_189948

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2) 
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end inequality_proof_l1899_189948


namespace rectangle_to_square_l1899_189938

/-- A rectangle can be cut into three parts to form a square --/
theorem rectangle_to_square :
  ∃ (a b c : ℕ × ℕ),
    -- The original rectangle is 25 × 4
    25 * 4 = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- The three parts can form a square
    ∃ (s : ℕ), s * s = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- There are exactly three parts
    a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry


end rectangle_to_square_l1899_189938


namespace fourth_month_sales_l1899_189931

def sales_problem (m1 m2 m3 m5 m6 average : ℕ) : Prop :=
  ∃ m4 : ℕ, (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem fourth_month_sales :
  sales_problem 6435 6927 6855 6562 7391 6900 →
  ∃ m4 : ℕ, m4 = 7230 ∧ (6435 + 6927 + 6855 + m4 + 6562 + 7391) / 6 = 6900 :=
by sorry

end fourth_month_sales_l1899_189931


namespace equation_system_solution_l1899_189934

theorem equation_system_solution (x z : ℝ) 
  (eq1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (eq2 : 3 * x + z + 4 = 0) :
  z^2 + 20 * z - 14 = 0 := by
  sorry

end equation_system_solution_l1899_189934


namespace march_1900_rainfall_average_l1899_189962

/-- The average rainfall per minute given total rainfall and number of days -/
def average_rainfall_per_minute (total_rainfall : ℚ) (days : ℕ) : ℚ :=
  total_rainfall / (days * 24 * 60)

/-- Theorem stating that 620 inches of rainfall over 15 days results in an average of 31/1080 inches per minute -/
theorem march_1900_rainfall_average : 
  average_rainfall_per_minute 620 15 = 31 / 1080 := by
  sorry

end march_1900_rainfall_average_l1899_189962


namespace f_2015_equals_negative_5_l1899_189942

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2015_equals_negative_5
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : is_periodic f 4)
  (h3 : f 1 = 5) :
  f 2015 = -5 := by
  sorry

end f_2015_equals_negative_5_l1899_189942


namespace arithmetic_mean_of_special_set_l1899_189984

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate n 1 ++ [1 + 1 / n]
  (set.sum / set.length : ℚ) = 1 + 1 / (n * (n + 1)) := by sorry

end arithmetic_mean_of_special_set_l1899_189984


namespace fairview_soccer_contest_l1899_189910

/-- Calculates the number of penalty kicks in a soccer team contest --/
def penalty_kicks (total_players : ℕ) (initial_goalies : ℕ) (absent_players : ℕ) (absent_goalies : ℕ) : ℕ :=
  let remaining_players := total_players - absent_players
  let remaining_goalies := initial_goalies - absent_goalies
  remaining_goalies * (remaining_players - 1)

/-- Theorem stating the number of penalty kicks for the Fairview College Soccer Team contest --/
theorem fairview_soccer_contest : 
  penalty_kicks 25 4 2 1 = 66 := by
  sorry

end fairview_soccer_contest_l1899_189910


namespace square_of_1017_l1899_189920

theorem square_of_1017 : (1017 : ℕ)^2 = 1034289 := by
  sorry

end square_of_1017_l1899_189920


namespace initial_visual_range_proof_l1899_189917

/-- The initial visual range without the telescope -/
def initial_range : ℝ := 50

/-- The visual range with the telescope -/
def telescope_range : ℝ := 150

/-- The percentage increase in visual range -/
def percentage_increase : ℝ := 200

theorem initial_visual_range_proof :
  initial_range = telescope_range / (1 + percentage_increase / 100) :=
by sorry

end initial_visual_range_proof_l1899_189917


namespace sum_of_polynomials_l1899_189900

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials :
  let f : ℝ → ℝ := λ x => -4 * x^2 + 2 * x - 5
  let g : ℝ → ℝ := λ x => -6 * x^2 + 4 * x - 9
  let h : ℝ → ℝ := λ x => 6 * x^2 + 6 * x + 2
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
sorry

end sum_of_polynomials_l1899_189900


namespace larger_number_proof_l1899_189997

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365) 
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by
  sorry

end larger_number_proof_l1899_189997


namespace trajectory_is_parabola_l1899_189932

noncomputable section

-- Define the * operation
def ast (x₁ x₂ : ℝ) : ℝ := (x₁ + x₂)^2 - (x₁ - x₂)^2

-- Define the point P
def P (x a : ℝ) : ℝ × ℝ := (x, Real.sqrt (ast x a))

-- Theorem statement
theorem trajectory_is_parabola (a : ℝ) (h₁ : a > 0) :
  ∃ k c : ℝ, ∀ x : ℝ, x ≥ 0 → (P x a).2^2 = k * (P x a).1 + c :=
sorry

end trajectory_is_parabola_l1899_189932


namespace hot_dogs_remainder_l1899_189954

theorem hot_dogs_remainder : 25197621 % 4 = 1 := by sorry

end hot_dogs_remainder_l1899_189954


namespace min_length_shared_side_l1899_189965

/-- Given two triangles ABC and DBC sharing side BC, with AB = 8, AC = 15, DC = 10, and BD = 25,
    the minimum possible integer length of BC is 15. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → 
  BC > AC - AB → BC > BD - DC → 
  BC ≥ 15 ∧ ∀ n : ℕ, n < 15 → ¬(BC = n) :=
by sorry

end min_length_shared_side_l1899_189965


namespace absolute_value_equation_solution_l1899_189963

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l1899_189963


namespace problem_statement_l1899_189994

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) 
  (h2 : a ≤ 13) 
  (h3 : (51 ^ 2016 - a) % 13 = 0) : 
  a = 1 := by
sorry

end problem_statement_l1899_189994


namespace equilateral_triangle_area_perimeter_ratio_l1899_189928

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l1899_189928


namespace sum_of_coefficients_l1899_189979

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x * (1 - 2*x)^4 = a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₃ + a₄ + a₅ = 0 := by
sorry

end sum_of_coefficients_l1899_189979


namespace arccos_sin_one_point_five_l1899_189929

theorem arccos_sin_one_point_five (π : Real) :
  π = 3.14159265358979323846 →
  Real.arccos (Real.sin 1.5) = 0.0708 := by
  sorry

end arccos_sin_one_point_five_l1899_189929


namespace solution_value_l1899_189980

theorem solution_value (a b : ℝ) (h : a^2 + b^2 - 4*a - 6*b + 13 = 0) : 
  (a - b)^2023 = -1 := by sorry

end solution_value_l1899_189980


namespace train_speed_l1899_189937

/-- Proves that a train with given parameters has a speed of 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 100 →
  crossing_time = 30 →
  total_length = 275 →
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by
  sorry

#check train_speed

end train_speed_l1899_189937


namespace shopkeeper_profit_percentage_l1899_189939

/-- Represents a faulty meter with its weight and associated profit percentage -/
structure FaultyMeter where
  weight : ℕ
  profit_percentage : ℚ

/-- Calculates the weighted profit for a meter given its profit percentage and sales volume ratio -/
def weighted_profit (meter : FaultyMeter) (sales_ratio : ℚ) (total_ratio : ℚ) : ℚ :=
  meter.profit_percentage * (sales_ratio / total_ratio)

/-- Theorem stating that the overall profit percentage is 11.6% given the conditions -/
theorem shopkeeper_profit_percentage 
  (meter1 : FaultyMeter)
  (meter2 : FaultyMeter)
  (meter3 : FaultyMeter)
  (h1 : meter1.weight = 900)
  (h2 : meter2.weight = 850)
  (h3 : meter3.weight = 950)
  (h4 : meter1.profit_percentage = 1/10)
  (h5 : meter2.profit_percentage = 12/100)
  (h6 : meter3.profit_percentage = 15/100)
  (sales_ratio1 : ℚ)
  (sales_ratio2 : ℚ)
  (sales_ratio3 : ℚ)
  (h7 : sales_ratio1 = 5)
  (h8 : sales_ratio2 = 3)
  (h9 : sales_ratio3 = 2) :
  weighted_profit meter1 sales_ratio1 (sales_ratio1 + sales_ratio2 + sales_ratio3) +
  weighted_profit meter2 sales_ratio2 (sales_ratio1 + sales_ratio2 + sales_ratio3) +
  weighted_profit meter3 sales_ratio3 (sales_ratio1 + sales_ratio2 + sales_ratio3) =
  116/1000 := by
  sorry

end shopkeeper_profit_percentage_l1899_189939


namespace product_equals_243_l1899_189901

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_equals_243_l1899_189901


namespace modular_exponentiation_difference_l1899_189925

theorem modular_exponentiation_difference (n : ℕ) :
  (45^2011 - 23^2011) % 7 = 5 := by sorry

end modular_exponentiation_difference_l1899_189925


namespace consecutive_products_not_3000000_l1899_189958

theorem consecutive_products_not_3000000 :
  ∀ n : ℕ, (n - 1) * n + n * (n + 1) + (n - 1) * (n + 1) ≠ 3000000 := by
  sorry

end consecutive_products_not_3000000_l1899_189958


namespace schedule_theorem_l1899_189957

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of subjects to be scheduled -/
def num_subjects : ℕ := 4

/-- Calculates the number of ways to schedule subjects -/
def schedule_ways : ℕ := Nat.choose num_periods num_subjects * Nat.factorial num_subjects

/-- Theorem stating that the number of ways to schedule 4 subjects in 7 periods
    with no consecutive subjects is 840 -/
theorem schedule_theorem : schedule_ways = 840 := by
  sorry

end schedule_theorem_l1899_189957


namespace abs_neg_gt_neg_implies_positive_l1899_189952

theorem abs_neg_gt_neg_implies_positive (a : ℝ) : |(-a)| > -a → a > 0 := by
  sorry

end abs_neg_gt_neg_implies_positive_l1899_189952


namespace right_to_left_equiv_standard_not_equiv_l1899_189941

/-- Evaluates an expression in a right-to-left order -/
noncomputable def evaluateRightToLeft (a b c d : ℝ) : ℝ :=
  a / (b - c - d)

/-- Standard algebraic evaluation -/
noncomputable def evaluateStandard (a b c d : ℝ) : ℝ :=
  a / b - c + d

/-- Theorem stating the equivalence of right-to-left evaluation and the correct standard algebraic form -/
theorem right_to_left_equiv (a b c d : ℝ) :
  evaluateRightToLeft a b c d = a / (b - c - d) :=
by sorry

/-- Theorem stating that the standard algebraic evaluation is not equivalent to the right-to-left evaluation -/
theorem standard_not_equiv (a b c d : ℝ) :
  evaluateStandard a b c d ≠ evaluateRightToLeft a b c d :=
by sorry

end right_to_left_equiv_standard_not_equiv_l1899_189941


namespace apex_to_center_distance_for_specific_pyramid_l1899_189908

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ

/-- The distance between the apex and the center of the circumsphere of the frustum -/
noncomputable def apex_to_center_distance (p : CutPyramid) : ℝ :=
  sorry

/-- Theorem stating the relationship between the pyramid's properties and the apex-to-center distance -/
theorem apex_to_center_distance_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 6
  }
  apex_to_center_distance p = 5 * (36 ^ (1/3 : ℝ)) / 2 := by
  sorry

end apex_to_center_distance_for_specific_pyramid_l1899_189908


namespace partner_A_share_is_8160_l1899_189921

/-- Calculates the share of profit for partner A in a business partnership --/
def partner_A_share (total_profit : ℚ) (A_investment : ℚ) (B_investment : ℚ) (management_fee_percent : ℚ) : ℚ :=
  let management_fee := total_profit * management_fee_percent / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := A_investment + B_investment
  let A_proportion := A_investment / total_investment
  management_fee + (remaining_profit * A_proportion)

/-- Theorem stating that partner A's share is 8160 Rs under given conditions --/
theorem partner_A_share_is_8160 :
  partner_A_share 9600 5000 1000 10 = 8160 := by
  sorry

end partner_A_share_is_8160_l1899_189921


namespace special_rectangle_dimensions_and_perimeter_l1899_189943

/-- A rectangle with integer sides where the area equals twice the perimeter -/
structure SpecialRectangle where
  a : ℕ
  b : ℕ
  h1 : a ≠ b
  h2 : a * b = 2 * (2 * a + 2 * b)

theorem special_rectangle_dimensions_and_perimeter (rect : SpecialRectangle) :
  (rect.a = 12 ∧ rect.b = 6) ∨ (rect.a = 6 ∧ rect.b = 12) ∧
  2 * (rect.a + rect.b) = 36 := by
  sorry

#check special_rectangle_dimensions_and_perimeter

end special_rectangle_dimensions_and_perimeter_l1899_189943


namespace equation_implies_fraction_value_l1899_189972

theorem equation_implies_fraction_value (a b : ℝ) :
  a^2 + b^2 - 4*a - 2*b + 5 = 0 →
  (Real.sqrt a + b) / (2 * Real.sqrt a + b + 1) = 1/2 := by
sorry

end equation_implies_fraction_value_l1899_189972


namespace solve_for_y_l1899_189967

theorem solve_for_y (x y : ℚ) (h1 : x = 103) (h2 : x^3 * y - 2 * x^2 * y + x * y = 103030) : y = 10 / 103 := by
  sorry

end solve_for_y_l1899_189967


namespace max_value_of_f_l1899_189974

-- Define the function f on [1, 4]
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem max_value_of_f :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Icc 1 4, f x = x^2 - 4*x + 5) →  -- f(x) = x^2 - 4x + 5 for x ∈ [1, 4]
  (∃ c ∈ Set.Icc (-4) (-1), ∀ x ∈ Set.Icc (-4) (-1), f x ≤ f c) →  -- maximum exists on [-4, -1]
  (∀ x ∈ Set.Icc (-4) (-1), f x ≤ -1) ∧  -- maximum value is at most -1
  (∃ x ∈ Set.Icc (-4) (-1), f x = -1)  -- maximum value -1 is achieved
  := by sorry

end max_value_of_f_l1899_189974


namespace complex_fraction_equals_i_l1899_189993

/-- Given that z = (a^2 - 1) + (a - 1)i is a purely imaginary number and a is real,
    prove that (a^2 + i) / (1 + ai) = i -/
theorem complex_fraction_equals_i (a : ℝ) (h : (a^2 - 1 : ℂ) + (a - 1)*I = (0 : ℂ) + I * ((a - 1 : ℝ) : ℂ)) :
  (a^2 + I) / (1 + a*I) = I := by
  sorry

end complex_fraction_equals_i_l1899_189993


namespace multiple_solutions_exist_four_wheelers_not_unique_l1899_189960

/-- Represents the number of wheels on a vehicle -/
inductive WheelCount
  | two
  | four

/-- Represents the parking lot with 2 wheelers and 4 wheelers -/
structure ParkingLot where
  twoWheelers : ℕ
  fourWheelers : ℕ

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (lot : ParkingLot) : ℕ :=
  2 * lot.twoWheelers + 4 * lot.fourWheelers

/-- Theorem stating that multiple solutions exist for a given total wheel count -/
theorem multiple_solutions_exist (totalWheelCount : ℕ) :
  ∃ (lot1 lot2 : ParkingLot), lot1 ≠ lot2 ∧ totalWheels lot1 = totalWheelCount ∧ totalWheels lot2 = totalWheelCount :=
sorry

/-- Theorem stating that the number of 4 wheelers cannot be uniquely determined -/
theorem four_wheelers_not_unique (totalWheelCount : ℕ) :
  ¬∃! (fourWheelerCount : ℕ), ∃ (twoWheelerCount : ℕ), totalWheels {twoWheelers := twoWheelerCount, fourWheelers := fourWheelerCount} = totalWheelCount :=
sorry

end multiple_solutions_exist_four_wheelers_not_unique_l1899_189960


namespace gaokao_probability_l1899_189953

/-- The probability of choosing both Physics and History in the Gaokao exam -/
theorem gaokao_probability (p_physics_not_history p_history_not_physics : ℝ) 
  (h1 : p_physics_not_history = 0.5)
  (h2 : p_history_not_physics = 0.3) :
  1 - p_physics_not_history - p_history_not_physics = 0.2 := by sorry

end gaokao_probability_l1899_189953


namespace pascal_triangle_fifth_number_l1899_189916

theorem pascal_triangle_fifth_number : 
  let row := List.cons 1 (List.cons 15 (List.replicate 3 0))  -- represents the start of the row
  let fifth_number := Nat.choose 15 4  -- represents ₁₅C₄
  fifth_number = 1365 := by
  sorry

end pascal_triangle_fifth_number_l1899_189916


namespace intersection_of_M_and_N_l1899_189977

open Set

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_of_M_and_N_l1899_189977


namespace profit_sharing_l1899_189913

/-- Profit sharing in a partnership --/
theorem profit_sharing
  (tom_investment jerry_investment : ℝ)
  (total_profit : ℝ)
  (tom_extra : ℝ)
  (h1 : tom_investment = 700)
  (h2 : jerry_investment = 300)
  (h3 : total_profit = 3000)
  (h4 : tom_extra = 800) :
  ∃ (equal_portion : ℝ),
    equal_portion = 1000 ∧
    (equal_portion / 2 + (tom_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion)) =
    (equal_portion / 2 + (jerry_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion) + tom_extra) :=
by sorry

end profit_sharing_l1899_189913


namespace sphere_radii_ratio_l1899_189970

/-- Given four spheres arranged such that each sphere touches three others and a plane,
    with two spheres having radius R and two spheres having radius r,
    prove that the ratio of the larger radius to the smaller radius is 2 + √3. -/
theorem sphere_radii_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0)
  (h3 : R^2 + r^2 = 4*R*r) : R/r = 2 + Real.sqrt 3 ∨ r/R = 2 + Real.sqrt 3 := by
  sorry

end sphere_radii_ratio_l1899_189970


namespace x_intercept_of_line_x_intercept_specific_line_l1899_189936

/-- Given two points on a line, calculate its x-intercept -/
theorem x_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = (m * x₁ - y₁) / m :=
by sorry

/-- The x-intercept of a line passing through (10, 3) and (-12, -8) is 4 -/
theorem x_intercept_specific_line :
  let x₁ : ℝ := 10
  let y₁ : ℝ := 3
  let x₂ : ℝ := -12
  let y₂ : ℝ := -8
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = 4 :=
by sorry

end x_intercept_of_line_x_intercept_specific_line_l1899_189936


namespace searchlight_probability_l1899_189905

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- The number of degrees in a full circle -/
def degrees_in_circle : ℝ := 360

/-- The minimum number of seconds the man should stay in the dark -/
def min_dark_seconds : ℝ := 5

/-- The probability of a man staying in the dark for at least 5 seconds
    when a searchlight makes 2 revolutions per minute -/
theorem searchlight_probability : 
  (degrees_in_circle - (min_dark_seconds / (seconds_per_minute / revolutions_per_minute)) * degrees_in_circle) / degrees_in_circle = 5 / 6 := by
  sorry

end searchlight_probability_l1899_189905


namespace solution_set_quadratic_inequality_l1899_189944

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end solution_set_quadratic_inequality_l1899_189944


namespace least_non_lucky_multiple_of_11_l1899_189969

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % (sumOfDigits n) = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (isMultipleOf11 132) ∧
  ¬(isLuckyInteger 132) ∧
  ∀ n : ℕ, n > 0 ∧ n < 132 ∧ (isMultipleOf11 n) → (isLuckyInteger n) := by
  sorry

end least_non_lucky_multiple_of_11_l1899_189969


namespace largest_three_digit_congruence_l1899_189986

theorem largest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    40 * n ≡ 140 [MOD 320] ∧
    ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 40 * m ≡ 140 [MOD 320]) → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_three_digit_congruence_l1899_189986


namespace goose_egg_hatch_fraction_l1899_189918

theorem goose_egg_hatch_fraction (eggs : ℕ) (hatched : ℕ) 
  (h1 : hatched ≤ eggs) 
  (h2 : (4 : ℚ) / 5 * ((2 : ℚ) / 5 * hatched) = 120) : 
  (hatched : ℚ) / eggs = 1 := by
  sorry

end goose_egg_hatch_fraction_l1899_189918


namespace chads_birthday_money_l1899_189968

/-- Chad's savings problem -/
theorem chads_birthday_money (
  savings_rate : ℝ)
  (other_earnings : ℝ)
  (total_savings : ℝ)
  (birthday_money : ℝ) :
  savings_rate = 0.4 →
  other_earnings = 900 →
  total_savings = 460 →
  savings_rate * (other_earnings + birthday_money) = total_savings →
  birthday_money = 250 := by
  sorry

end chads_birthday_money_l1899_189968


namespace sum_11_is_negative_11_l1899_189935

/-- An arithmetic sequence with its sum of terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  first_term : a 1 = -11
  sum_condition : S 10 / 10 - S 8 / 8 = 2

/-- The sum of the first 11 terms in the given arithmetic sequence is -11 -/
theorem sum_11_is_negative_11 (seq : ArithmeticSequence) : seq.S 11 = -11 := by
  sorry

end sum_11_is_negative_11_l1899_189935


namespace sugar_left_in_grams_l1899_189955

/-- The amount of sugar Pamela bought in ounces -/
def sugar_bought : ℝ := 9.8

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℝ := 28.35

/-- Theorem stating the amount of sugar Pamela has left in grams -/
theorem sugar_left_in_grams : 
  (sugar_bought - sugar_spilled) * oz_to_g = 130.41 := by
  sorry

end sugar_left_in_grams_l1899_189955


namespace distinct_primes_in_product_l1899_189945

theorem distinct_primes_in_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (85 * 87 * 88 * 90) % p = 0 → p ∈ s) ∧ 
  Finset.card s = 6 := by
  sorry

end distinct_primes_in_product_l1899_189945


namespace students_taking_statistics_l1899_189961

theorem students_taking_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (history_or_statistics : ℕ) 
  (history_not_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 29) :
  ∃ (statistics : ℕ), statistics = 30 :=
by sorry

end students_taking_statistics_l1899_189961


namespace iggy_running_time_l1899_189973

/-- Represents the daily running distances in miles -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Represents the pace in minutes per mile -/
def pace : Nat := 10

/-- Calculates the total running time in hours -/
def total_running_hours (miles : List Nat) (pace : Nat) : Nat :=
  (miles.sum * pace) / 60

/-- Theorem: Iggy's total running time from Monday to Friday is 4 hours -/
theorem iggy_running_time :
  total_running_hours daily_miles pace = 4 := by
  sorry

#eval total_running_hours daily_miles pace

end iggy_running_time_l1899_189973


namespace exists_valid_number_l1899_189976

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, (n / 10^i) % 10 ≠ 0)

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem exists_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ (n + reverse_number n) % 101 = 0 :=
sorry

end exists_valid_number_l1899_189976


namespace cube_volume_l1899_189923

/-- Given a cube with side perimeter 32 cm, its volume is 512 cubic cm. -/
theorem cube_volume (side_perimeter : ℝ) (h : side_perimeter = 32) : 
  (side_perimeter / 4)^3 = 512 := by
  sorry

end cube_volume_l1899_189923


namespace mickey_vs_twice_minnie_l1899_189919

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per week -/
def mickey_horses_per_week : ℕ := 98

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := mickey_horses_per_week / days_in_week

theorem mickey_vs_twice_minnie :
  2 * minnie_horses_per_day - mickey_horses_per_day = 6 :=
sorry

end mickey_vs_twice_minnie_l1899_189919


namespace inequality_bound_l1899_189926

theorem inequality_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) + 
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) < 2 := by
  sorry

end inequality_bound_l1899_189926


namespace line_tangent_to_circle_l1899_189987

/-- A line given by parametric equations is tangent to a circle. -/
theorem line_tangent_to_circle (α : Real) (h1 : α > π / 2) :
  (∃ t : Real, ∀ φ : Real,
    let x_line := t * Real.cos α
    let y_line := t * Real.sin α
    let x_circle := 4 + 2 * Real.cos φ
    let y_circle := 2 * Real.sin φ
    (x_line - x_circle)^2 + (y_line - y_circle)^2 = 4) →
  α = 5 * π / 6 := by
sorry

end line_tangent_to_circle_l1899_189987


namespace parabola_shift_theorem_l1899_189975

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 1
  y = 2*x^2 → y = shifted.a * (x + 4)^2 + 1 :=
by sorry

end parabola_shift_theorem_l1899_189975


namespace arithmetic_to_geometric_progression_l1899_189983

theorem arithmetic_to_geometric_progression 
  (x y z : ℝ) 
  (h1 : y^2 - x*y = z^2 - y^2) : 
  z^2 = y * (2*y - x) := by
sorry

end arithmetic_to_geometric_progression_l1899_189983


namespace canoe_downstream_speed_l1899_189999

/-- Represents the speed of a canoe in different conditions -/
structure CanoeSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a canoe given its speed in still water and upstream -/
def downstreamSpeed (c : CanoeSpeed) : ℝ :=
  2 * c.stillWater - c.upstream

/-- Theorem stating that for a canoe with 12.5 km/hr speed in still water and 9 km/hr upstream speed, 
    the downstream speed is 16 km/hr -/
theorem canoe_downstream_speed :
  let c : CanoeSpeed := { stillWater := 12.5, upstream := 9 }
  downstreamSpeed c = 16 := by
  sorry


end canoe_downstream_speed_l1899_189999


namespace modular_difference_in_range_l1899_189981

def problem (a b : ℤ) : Prop :=
  a % 36 = 22 ∧ b % 36 = 85

def valid_range (n : ℤ) : Prop :=
  120 ≤ n ∧ n ≤ 161

theorem modular_difference_in_range (a b : ℤ) (h : problem a b) :
  ∃! n : ℤ, valid_range n ∧ (a - b) % 36 = n % 36 ∧ n = 153 := by sorry

end modular_difference_in_range_l1899_189981


namespace f_monotone_decreasing_l1899_189947

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem f_monotone_decreasing :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → f x1 > f x2 := by
  sorry

end f_monotone_decreasing_l1899_189947


namespace initial_garlic_cloves_l1899_189998

/-- 
Given that Maria used 86 cloves of garlic for roast chicken and has 7 cloves left,
prove that she initially stored 93 cloves of garlic.
-/
theorem initial_garlic_cloves (used : ℕ) (left : ℕ) (h1 : used = 86) (h2 : left = 7) :
  used + left = 93 := by
  sorry

end initial_garlic_cloves_l1899_189998


namespace range_of_x_given_integer_part_l1899_189930

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the theorem
theorem range_of_x_given_integer_part (x : ℝ) :
  integerPart ((1 - 3*x) / 2) = -1 → 1/3 < x ∧ x ≤ 1 :=
by sorry

end range_of_x_given_integer_part_l1899_189930


namespace total_cost_of_promotional_items_l1899_189982

/-- The cost of a calendar in dollars -/
def calendar_cost : ℚ := 3/4

/-- The cost of a date book in dollars -/
def date_book_cost : ℚ := 1/2

/-- The number of calendars ordered -/
def calendars_ordered : ℕ := 300

/-- The number of date books ordered -/
def date_books_ordered : ℕ := 200

/-- The total number of items ordered -/
def total_items : ℕ := 500

/-- Theorem stating the total cost of promotional items -/
theorem total_cost_of_promotional_items :
  calendars_ordered * calendar_cost + date_books_ordered * date_book_cost = 325/1 :=
by sorry

end total_cost_of_promotional_items_l1899_189982


namespace fraction_calculation_l1899_189909

theorem fraction_calculation : 
  (8 / 4 * 9 / 3 * 20 / 5) / (10 / 5 * 12 / 4 * 15 / 3) = 4 / 5 := by
  sorry

end fraction_calculation_l1899_189909


namespace optimal_newspaper_sales_l1899_189915

/-- Represents the daily newspaper sales data --/
structure NewspaperSalesData where
  costPrice : ℝ
  sellingPrice : ℝ
  returnPrice : ℝ
  highSalesDays : ℕ
  highSalesAmount : ℕ
  lowSalesDays : ℕ
  lowSalesAmount : ℕ

/-- Calculates the monthly profit based on the number of copies purchased daily --/
def monthlyProfit (data : NewspaperSalesData) (dailyPurchase : ℕ) : ℝ :=
  let soldProfit := data.sellingPrice - data.costPrice
  let returnLoss := data.costPrice - data.returnPrice
  let totalSold := data.highSalesDays * (min dailyPurchase data.highSalesAmount) +
                   data.lowSalesDays * (min dailyPurchase data.lowSalesAmount)
  let totalReturned := (data.highSalesDays + data.lowSalesDays) * dailyPurchase - totalSold
  soldProfit * totalSold - returnLoss * totalReturned

/-- Theorem stating the optimal daily purchase and maximum monthly profit --/
theorem optimal_newspaper_sales (data : NewspaperSalesData)
  (h1 : data.costPrice = 0.12)
  (h2 : data.sellingPrice = 0.20)
  (h3 : data.returnPrice = 0.04)
  (h4 : data.highSalesDays = 20)
  (h5 : data.highSalesAmount = 400)
  (h6 : data.lowSalesDays = 10)
  (h7 : data.lowSalesAmount = 250) :
  (∀ x : ℕ, monthlyProfit data x ≤ monthlyProfit data 400) ∧
  monthlyProfit data 400 = 840 := by
  sorry


end optimal_newspaper_sales_l1899_189915


namespace water_in_altered_solution_l1899_189933

/-- Represents the ratios of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the new ratio after altering the original ratio -/
def alter_ratio (original : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * original.bleach,
    detergent := original.detergent,
    water := 2 * original.water }

/-- Theorem: Given the conditions, the altered solution contains 150 liters of water -/
theorem water_in_altered_solution :
  let original_ratio : SolutionRatio := ⟨2, 40, 100⟩
  let altered_ratio := alter_ratio original_ratio
  let detergent_volume : ℚ := 60
  (detergent_volume * altered_ratio.water) / altered_ratio.detergent = 150 := by
  sorry

end water_in_altered_solution_l1899_189933


namespace domain_exclusion_sum_l1899_189985

theorem domain_exclusion_sum (A B : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ x = A ∨ x = B) → A + B = 3 := by
sorry

end domain_exclusion_sum_l1899_189985


namespace largest_product_l1899_189946

def S : Finset Int := {-4, -3, -1, 5, 6, 7}

def isConsecutive (a b : Int) : Prop := b = a + 1 ∨ a = b + 1

def fourDistinctElements (a b c d : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def twoConsecutive (a b c d : Int) : Prop :=
  isConsecutive a b ∨ isConsecutive a c ∨ isConsecutive a d ∨
  isConsecutive b c ∨ isConsecutive b d ∨ isConsecutive c d

theorem largest_product :
  ∀ a b c d : Int,
    fourDistinctElements a b c d →
    twoConsecutive a b c d →
    a * b * c * d ≤ -210 :=
by sorry

end largest_product_l1899_189946


namespace grocer_coffee_percentage_l1899_189903

/-- Calculates the percentage of decaffeinated coffee in a grocer's stock -/
theorem grocer_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60)
  : (initial_stock * initial_decaf_percent / 100 + additional_stock * additional_decaf_percent / 100) /
    (initial_stock + additional_stock) * 100 = 28 := by
  sorry

end grocer_coffee_percentage_l1899_189903


namespace quadratic_integer_roots_l1899_189912

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end quadratic_integer_roots_l1899_189912


namespace smallest_number_divisible_by_all_l1899_189902

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 3147 → ¬(is_divisible_by_all n) ∧ is_divisible_by_all 3147 := by
  sorry

end smallest_number_divisible_by_all_l1899_189902


namespace multiply_63_57_l1899_189904

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l1899_189904


namespace leg_length_in_special_right_isosceles_triangle_l1899_189956

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0

/-- Theorem: In a 45-45-90 triangle with hypotenuse 12√2, the length of a leg is 12 -/
theorem leg_length_in_special_right_isosceles_triangle 
  (triangle : RightIsoscelesTriangle) 
  (h : triangle.hypotenuse = 12 * Real.sqrt 2) : 
  triangle.hypotenuse / Real.sqrt 2 = 12 := by
  sorry

#check leg_length_in_special_right_isosceles_triangle

end leg_length_in_special_right_isosceles_triangle_l1899_189956


namespace y_value_l1899_189924

theorem y_value (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 := by
  sorry

end y_value_l1899_189924


namespace midpoint_trajectory_l1899_189996

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (a b : ℝ), a^2 + b^2 = 25 ∧ (x - a)^2 + (y - b)^2 = 4) →
  x^2 + y^2 = 9 :=
by sorry

end midpoint_trajectory_l1899_189996


namespace max_third_side_length_l1899_189940

theorem max_third_side_length (a b x : ℕ) (ha : a = 28) (hb : b = 47) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → x ≤ 74 :=
sorry

end max_third_side_length_l1899_189940


namespace complex_addition_multiplication_l1899_189951

theorem complex_addition_multiplication : 
  let z₁ : ℂ := 2 + 6 * I
  let z₂ : ℂ := 5 - 3 * I
  3 * (z₁ + z₂) = 21 + 9 * I :=
by sorry

end complex_addition_multiplication_l1899_189951


namespace inflection_point_and_concavity_l1899_189964

-- Define the function f(x) = x³ - 3x² + 5
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the first derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem inflection_point_and_concavity :
  -- The inflection point occurs at x = 1
  (∃ (ε : ℝ), ε > 0 ∧ 
    (∀ x ∈ Set.Ioo (1 - ε) 1, f'' x < 0) ∧
    (∀ x ∈ Set.Ioo 1 (1 + ε), f'' x > 0)) ∧
  -- f(1) = 3
  f 1 = 3 ∧
  -- The function is concave down for x < 1
  (∀ x < 1, f'' x < 0) ∧
  -- The function is concave up for x > 1
  (∀ x > 1, f'' x > 0) :=
by sorry

end inflection_point_and_concavity_l1899_189964


namespace polynomial_derivative_sum_l1899_189914

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end polynomial_derivative_sum_l1899_189914


namespace ship_grain_calculation_l1899_189949

/-- The amount of grain spilled from a ship, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining on the ship, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 := by sorry

end ship_grain_calculation_l1899_189949
