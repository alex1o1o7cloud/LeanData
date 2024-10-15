import Mathlib

namespace NUMINAMATH_CALUDE_sector_area_l3241_324171

/-- The area of a circular sector with central angle 54° and radius 20 cm is 60π cm² -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 54 * π / 180 → r = 20 → (1/2) * r^2 * θ = 60 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3241_324171


namespace NUMINAMATH_CALUDE_debate_team_boys_l3241_324101

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) (boys : ℕ) : 
  girls = 4 → 
  groups = 8 → 
  group_size = 4 → 
  total = groups * group_size → 
  boys = total - girls → 
  boys = 28 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l3241_324101


namespace NUMINAMATH_CALUDE_roberta_listening_time_l3241_324178

/-- The number of days it takes Roberta to listen to her entire record collection -/
def listen_time (initial_records : ℕ) (gift_records : ℕ) (bought_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + bought_records) * days_per_record

theorem roberta_listening_time :
  listen_time 8 12 30 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_roberta_listening_time_l3241_324178


namespace NUMINAMATH_CALUDE_base7_321_equals_base10_162_l3241_324138

def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_321_equals_base10_162 :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_base7_321_equals_base10_162_l3241_324138


namespace NUMINAMATH_CALUDE_no_unique_solution_l3241_324123

/-- 
Theorem: The system of equations 4(3x + 4y) = 48 and kx + 12y = 30 
does not have a unique solution if and only if k = -9.
-/
theorem no_unique_solution (k : ℝ) : 
  (∀ x y : ℝ, 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) → 
  (¬∃! (x y : ℝ), 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) ↔ 
  k = -9 :=
sorry


end NUMINAMATH_CALUDE_no_unique_solution_l3241_324123


namespace NUMINAMATH_CALUDE_smallest_period_scaled_l3241_324155

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 30) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 3) = f (x / 3)) ∧
  ∀ b' : ℝ, 0 < b' ∧ (∀ x, f ((x - b') / 3) = f (x / 3)) → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_l3241_324155


namespace NUMINAMATH_CALUDE_min_n_for_sum_greater_than_1020_l3241_324103

def sequence_term (n : ℕ) : ℕ := 2^n - 1

def sequence_sum (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_sum_greater_than_1020 :
  (∀ k < 10, sequence_sum k ≤ 1020) ∧ (sequence_sum 10 > 1020) := by sorry

end NUMINAMATH_CALUDE_min_n_for_sum_greater_than_1020_l3241_324103


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3241_324158

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
axiom p : ℂ
axiom q : ℂ
axiom r : ℂ
axiom s : ℂ

-- Axiom that p, q, r, s are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0
axiom root_s : f s = 0

-- The theorem to prove
theorem root_sum_reciprocals :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = -10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3241_324158


namespace NUMINAMATH_CALUDE_exp_addition_property_l3241_324197

open Real

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end NUMINAMATH_CALUDE_exp_addition_property_l3241_324197


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3241_324134

theorem inequality_equivalence (x : ℝ) : 
  (3 / (5 - 3 * x) > 1) ↔ (2 / 3 < x ∧ x < 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3241_324134


namespace NUMINAMATH_CALUDE_function_property_l3241_324194

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3241_324194


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3241_324196

/-- 
Given:
- Sandy attempts 30 sums
- Sandy obtains 45 marks in total
- Sandy got 21 sums correct
- Sandy loses 2 marks for each incorrect sum

Prove that Sandy gets 3 marks for each correct sum
-/
theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_sums = 21)
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3241_324196


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l3241_324141

theorem lawnmower_value_drop (initial_price : ℝ) (first_drop_percent : ℝ) (final_value : ℝ) :
  initial_price = 100 →
  first_drop_percent = 25 →
  final_value = 60 →
  let value_after_six_months := initial_price * (1 - first_drop_percent / 100)
  let drop_over_next_year := value_after_six_months - final_value
  let drop_percent_next_year := (drop_over_next_year / value_after_six_months) * 100
  drop_percent_next_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_value_drop_l3241_324141


namespace NUMINAMATH_CALUDE_gary_egg_collection_l3241_324190

/-- Represents the egg-laying rates of the initial chickens -/
def initial_rates : List Nat := [6, 5, 7, 4]

/-- Calculates the number of surviving chickens after two years -/
def surviving_chickens (initial : Nat) (growth_factor : Nat) (mortality_rate : Rat) : Nat :=
  Nat.floor ((initial * growth_factor : Rat) * (1 - mortality_rate))

/-- Calculates the average egg-laying rate -/
def average_rate (rates : List Nat) : Rat :=
  (rates.sum : Rat) / rates.length

/-- Calculates the total eggs per week -/
def total_eggs_per_week (chickens : Nat) (avg_rate : Rat) : Nat :=
  Nat.floor (7 * (chickens : Rat) * avg_rate)

/-- Theorem stating the number of eggs Gary collects per week -/
theorem gary_egg_collection :
  total_eggs_per_week
    (surviving_chickens 4 8 (1/5))
    (average_rate initial_rates) = 959 := by
  sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l3241_324190


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_l3241_324174

/-- Parallelogram ABCD with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (4, 4)
    B := (-2, -2)
    C := (-8, -2)
    D := (0, 4) }

/-- Function to calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Function to calculate the area of the part of the parallelogram below the x-axis -/
def areaBelowXAxis (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the probability of a point not being above the x-axis -/
theorem probability_not_above_x_axis (p : Parallelogram) :
  p = ABCD →
  (areaBelowXAxis p) / (area p) = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_not_above_x_axis_l3241_324174


namespace NUMINAMATH_CALUDE_decimal_addition_l3241_324136

theorem decimal_addition : 1 + 0.01 + 0.0001 = 1.0101 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l3241_324136


namespace NUMINAMATH_CALUDE_sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l3241_324167

/-- Represents the daily sales model for a specialty store -/
structure SalesModel where
  x : ℝ  -- Selling price per item in yuan
  y : ℝ  -- Daily sales volume in items
  W : ℝ  -- Daily total profit in yuan
  h1 : 16 ≤ x ∧ x ≤ 48  -- Price constraints
  h2 : y = -10 * x + 560  -- Relationship between y and x
  h3 : W = (x - 16) * y  -- Definition of total profit

/-- The daily sales volume is a linear function of the selling price -/
theorem sales_volume_linear (model : SalesModel) :
  ∃ a b : ℝ, model.y = a * model.x + b :=
sorry

/-- The daily total profit is a quadratic function of the selling price -/
theorem profit_quadratic (model : SalesModel) :
  ∃ a b c : ℝ, model.W = a * model.x^2 + b * model.x + c :=
sorry

/-- The maximum daily profit occurs when the selling price is 36 yuan and equals 4000 yuan -/
theorem max_profit (model : SalesModel) :
  (∀ x : ℝ, 16 ≤ x ∧ x ≤ 48 → model.W ≤ 4000) ∧
  (∃ model' : SalesModel, model'.x = 36 ∧ model'.W = 4000) :=
sorry

/-- There exists a selling price that ensures a daily profit of 3000 yuan while minimizing inventory -/
theorem profit_3000_min_inventory (model : SalesModel) :
  ∃ x : ℝ, 16 ≤ x ∧ x ≤ 48 ∧
  (∃ model' : SalesModel, model'.x = x ∧ model'.W = 3000) ∧
  (∀ x' : ℝ, 16 ≤ x' ∧ x' ≤ 48 →
    (∃ model'' : SalesModel, model''.x = x' ∧ model''.W = 3000) →
    x ≤ x') :=
sorry

end NUMINAMATH_CALUDE_sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l3241_324167


namespace NUMINAMATH_CALUDE_complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l3241_324120

/-- The set A defined as {x | -1 ≤ x < 4} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}

/-- The set B defined as {x | m ≤ x ≤ m+2} for a real number m -/
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Part 1: The complement of A ∩ B when m = 3 -/
theorem complement_A_inter_B_when_m_3 :
  (A ∩ B 3)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

/-- Part 2: Characterization of m when A ∩ B is empty -/
theorem A_inter_B_empty_iff (m : ℝ) :
  A ∩ B m = ∅ ↔ m < -3 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l3241_324120


namespace NUMINAMATH_CALUDE_sum_of_roots_l3241_324182

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 5) (hb : b * (b - 4) = 5) (hab : a ≠ b) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3241_324182


namespace NUMINAMATH_CALUDE_articles_count_l3241_324177

/-- 
Given:
- The selling price is double the cost price
- The cost price of X articles equals the selling price of 25 articles
Prove that X = 50
-/
theorem articles_count (cost_price selling_price : ℝ) (X : ℕ) 
  (h1 : selling_price = 2 * cost_price) 
  (h2 : X * cost_price = 25 * selling_price) : 
  X = 50 := by
  sorry

end NUMINAMATH_CALUDE_articles_count_l3241_324177


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3241_324170

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℚ)
  (subset_count : Nat)
  (subset_average : ℚ)
  (h1 : total_count = 50)
  (h2 : total_average = 76)
  (h3 : subset_count = 40)
  (h4 : subset_average = 80)
  (h5 : subset_count < total_count) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 60 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3241_324170


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3241_324185

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 5 * x + 4 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3241_324185


namespace NUMINAMATH_CALUDE_students_taking_neither_l3241_324125

theorem students_taking_neither (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75) 
  (h2 : chem = 40) 
  (h3 : bio = 35) 
  (h4 : both = 25) : 
  total - (chem + bio - both) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_neither_l3241_324125


namespace NUMINAMATH_CALUDE_hyperbola_and_line_theorem_l3241_324147

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - (4 * y^2 / 33) = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define point P
def P : ℝ × ℝ := (7, 12)

-- Define the asymptotes
def asymptote_positive (x y : ℝ) : Prop := y = (Real.sqrt 33 / 2) * x
def asymptote_negative (x y : ℝ) : Prop := y = -(Real.sqrt 33 / 2) * x

-- Define line l
def line_l (x y t : ℝ) : Prop := y = x + t

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_and_line_theorem :
  -- Hyperbola C passes through P
  hyperbola_C P.1 P.2 →
  -- There exist points A and B on C and l
  ∃ (A B : ℝ × ℝ) (t : ℝ),
    hyperbola_C A.1 A.2 ∧ 
    hyperbola_C B.1 B.2 ∧
    line_l A.1 A.2 t ∧
    line_l B.1 B.2 t ∧
    -- A and B are perpendicular from the origin
    perpendicular A.1 A.2 B.1 B.2 →
  -- Then the equation of line l is y = x ± √(66/29)
  t = Real.sqrt (66 / 29) ∨ t = -Real.sqrt (66 / 29) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_theorem_l3241_324147


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3241_324131

/-- The speed of a boat in still water, given its downstream speed and the current speed -/
theorem boat_speed_in_still_water
  (downstream_speed : ℝ) -- Speed of the boat downstream
  (current_speed : ℝ)    -- Speed of the current
  (h1 : downstream_speed = 36) -- Given downstream speed
  (h2 : current_speed = 6)     -- Given current speed
  : downstream_speed - current_speed = 30 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3241_324131


namespace NUMINAMATH_CALUDE_root_expression_value_l3241_324164

theorem root_expression_value (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2022 = 0) 
  (h₂ : x₂^2 - x₂ - 2022 = 0) : 
  x₁^3 - 2022*x₁ + x₂^2 = 4045 := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l3241_324164


namespace NUMINAMATH_CALUDE_geometric_progression_sum_not_end_20_l3241_324102

/-- Given a, b, c form a geometric progression, prove that a^3 + b^3 + c^3 - 3abc cannot end with 20 -/
theorem geometric_progression_sum_not_end_20 
  (a b c : ℤ) 
  (h_geom : ∃ (q : ℚ), b = a * q ∧ c = b * q) : 
  ¬ (∃ (k : ℤ), a^3 + b^3 + c^3 - 3*a*b*c = 100*k + 20) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_not_end_20_l3241_324102


namespace NUMINAMATH_CALUDE_prob_at_least_one_correct_l3241_324124

/-- The probability of subscribing to at least one of two newspapers -/
def prob_at_least_one (p1 p2 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2)

theorem prob_at_least_one_correct (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  prob_at_least_one p1 p2 = 1 - (1 - p1) * (1 - p2) := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_correct_l3241_324124


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3241_324104

/-- The repeating decimal 0.6̄3 as a real number -/
def repeating_decimal : ℚ := 19/30

/-- Theorem stating that the repeating decimal 0.6̄3 is equal to 19/30 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = 19/30 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3241_324104


namespace NUMINAMATH_CALUDE_horner_method_v3_l3241_324161

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_v3 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v3 :
  horner_v3 1 2 1 (-1) 3 (-5) 5 = 179 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3241_324161


namespace NUMINAMATH_CALUDE_min_difference_of_extreme_points_l3241_324162

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 1/x - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x + 2*a * log x

theorem min_difference_of_extreme_points (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ∈ Set.Icc 0 1 → 
  (∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≥ min (g a x₁) (g a x₂)) →
  g a x₁ - g a x₂ ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_extreme_points_l3241_324162


namespace NUMINAMATH_CALUDE_triangle_area_l3241_324109

theorem triangle_area (a b : ℝ) (θ : Real) (h1 : a = 30) (h2 : b = 24) (h3 : θ = π/3) :
  (1/2) * a * b * Real.sin θ = 180 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3241_324109


namespace NUMINAMATH_CALUDE_netflix_series_seasons_l3241_324152

theorem netflix_series_seasons (episodes_per_season : ℕ) (episodes_remaining : ℕ) : 
  episodes_per_season = 20 →
  episodes_remaining = 160 →
  (∃ (total_episodes : ℕ), 
    total_episodes * (1 / 3 : ℚ) = total_episodes - episodes_remaining ∧
    total_episodes / episodes_per_season = 12) :=
by sorry

end NUMINAMATH_CALUDE_netflix_series_seasons_l3241_324152


namespace NUMINAMATH_CALUDE_b_investment_value_l3241_324143

/-- Calculates the investment of partner B in a partnership business --/
def calculate_b_investment (a_investment b_investment c_investment total_profit a_profit : ℚ) : Prop :=
  let total_investment := a_investment + b_investment + c_investment
  (a_investment / total_investment = a_profit / total_profit) ∧
  b_investment = 13650

/-- Theorem stating B's investment given the problem conditions --/
theorem b_investment_value :
  calculate_b_investment 6300 13650 10500 12500 3750 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_value_l3241_324143


namespace NUMINAMATH_CALUDE_partition_twelve_possible_partition_twentytwo_impossible_l3241_324115

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def valid_partition (s : Set ℕ) (n : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    partition.length = n ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ s ∧ pair.2 ∈ s) ∧
    (∀ x : ℕ, x ∈ s → ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (x = pair.1 ∨ x = pair.2)) ∧
    (∀ (pair1 pair2 : ℕ × ℕ), pair1 ∈ partition → pair2 ∈ partition → pair1 ≠ pair2 →
      is_prime (pair1.1 + pair1.2) ∧
      is_prime (pair2.1 + pair2.2) ∧
      pair1.1 + pair1.2 ≠ pair2.1 + pair2.2)

theorem partition_twelve_possible : 
  valid_partition (Finset.range 12).toSet 6 := sorry

theorem partition_twentytwo_impossible : 
  ¬ valid_partition (Finset.range 22).toSet 11 := sorry

end NUMINAMATH_CALUDE_partition_twelve_possible_partition_twentytwo_impossible_l3241_324115


namespace NUMINAMATH_CALUDE_constant_value_proof_l3241_324111

theorem constant_value_proof (t : ℝ) (constant : ℝ) : 
  let x := 1 - 3 * t
  let y := constant * t - 3
  (t = 0.8 → x = y) → constant = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l3241_324111


namespace NUMINAMATH_CALUDE_competition_probabilities_l3241_324181

/-- Represents the type of question in the competition -/
inductive QuestionType
| MultipleChoice
| TrueFalse

/-- Represents a question in the competition -/
structure Question where
  id : Nat
  type : QuestionType

/-- Represents the competition setup -/
structure Competition where
  questions : Finset Question
  numMultipleChoice : Nat
  numTrueFalse : Nat

/-- Represents a draw outcome for two participants -/
structure DrawOutcome where
  questionA : Question
  questionB : Question

/-- The probability of A drawing a multiple-choice question and B drawing a true/false question -/
def probAMultipleBTrue (c : Competition) : ℚ :=
  sorry

/-- The probability of at least one of A and B drawing a multiple-choice question -/
def probAtLeastOneMultiple (c : Competition) : ℚ :=
  sorry

/-- The main theorem stating the probabilities for the given competition setup -/
theorem competition_probabilities (c : Competition) 
  (h1 : c.questions.card = 4)
  (h2 : c.numMultipleChoice = 2)
  (h3 : c.numTrueFalse = 2) :
  probAMultipleBTrue c = 1/3 ∧ probAtLeastOneMultiple c = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_competition_probabilities_l3241_324181


namespace NUMINAMATH_CALUDE_paperclips_exceed_200_l3241_324198

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem paperclips_exceed_200 : ∀ k : ℕ, paperclips k ≤ 200 ↔ k < 7 := by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_200_l3241_324198


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3241_324132

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x ∧ f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3241_324132


namespace NUMINAMATH_CALUDE_dog_age_difference_l3241_324163

/-- The age difference between the 1st and 2nd fastest dogs -/
def age_difference (d1 d2 d3 d4 d5 : ℕ) : ℕ := d1 - d2

theorem dog_age_difference :
  ∀ d1 d2 d3 d4 d5 : ℕ,
  (d1 + d5) / 2 = 18 →  -- Average age of 1st and 5th dogs
  d1 = 10 →             -- Age of 1st dog
  d2 = d1 - 2 →         -- Age of 2nd dog
  d3 = d2 + 4 →         -- Age of 3rd dog
  d4 * 2 = d3 →         -- Age of 4th dog
  d5 = d4 + 20 →        -- Age of 5th dog
  age_difference d1 d2 = 2 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l3241_324163


namespace NUMINAMATH_CALUDE_negation_equivalence_l3241_324199

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3241_324199


namespace NUMINAMATH_CALUDE_distinct_cube_models_count_l3241_324145

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of colors available -/
def available_colors : ℕ := 8

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of distinct models of cubes with differently colored vertices -/
def distinct_cube_models : ℕ := Nat.factorial available_colors / cube_rotations

theorem distinct_cube_models_count :
  distinct_cube_models = 1680 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_models_count_l3241_324145


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l3241_324179

/-- A line in the plane defined by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Predicate to check if a line passes through a quadrant -/
def passes_through (l : Line) (q : Quadrant) : Prop := sorry

/-- Theorem stating that under given conditions, the line passes through specific quadrants -/
theorem line_passes_through_quadrants (l : Line) 
  (h1 : l.A * l.C < 0) (h2 : l.B * l.C < 0) : 
  passes_through l Quadrant.first ∧ 
  passes_through l Quadrant.second ∧ 
  passes_through l Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l3241_324179


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3241_324117

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3241_324117


namespace NUMINAMATH_CALUDE_rahim_book_purchase_l3241_324133

/-- The amount Rahim paid for books from the first shop -/
def amount_first_shop (books_first_shop : ℕ) (books_second_shop : ℕ) (price_second_shop : ℚ) (average_price : ℚ) : ℚ :=
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_second_shop

/-- Theorem stating the amount Rahim paid for books from the first shop -/
theorem rahim_book_purchase :
  amount_first_shop 65 50 920 (18088695652173913 / 1000000000000000) = 1160 := by
  sorry

end NUMINAMATH_CALUDE_rahim_book_purchase_l3241_324133


namespace NUMINAMATH_CALUDE_prism_faces_l3241_324151

/-- Represents a prism with n-sided polygonal bases -/
structure Prism where
  n : ℕ
  vertices : ℕ := 2 * n
  edges : ℕ := 3 * n
  faces : ℕ := n + 2

/-- Theorem: A prism with 40 as the sum of its vertices and edges has 10 faces -/
theorem prism_faces (p : Prism) (h : p.vertices + p.edges = 40) : p.faces = 10 := by
  sorry


end NUMINAMATH_CALUDE_prism_faces_l3241_324151


namespace NUMINAMATH_CALUDE_telecom_plans_l3241_324165

/-- Represents the monthly fee for Plan A given the call duration -/
def plan_a_fee (x : ℝ) : ℝ := 0.4 * x + 50

/-- Represents the monthly fee for Plan B given the call duration -/
def plan_b_fee (x : ℝ) : ℝ := 0.6 * x

theorem telecom_plans :
  (∀ x : ℝ, plan_a_fee x = 0.4 * x + 50) ∧
  (∀ x : ℝ, plan_b_fee x = 0.6 * x) ∧
  (plan_a_fee 300 < plan_b_fee 300) ∧
  (∃ x : ℝ, x = 250 ∧ plan_a_fee x = plan_b_fee x) :=
sorry

end NUMINAMATH_CALUDE_telecom_plans_l3241_324165


namespace NUMINAMATH_CALUDE_magic_square_a_plus_b_l3241_324150

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (w y a b z : ℕ)
  (magic_sum : ℕ)
  (top_row : 19 + w + 23 = magic_sum)
  (middle_row : 22 + y + a = magic_sum)
  (bottom_row : b + 18 + z = magic_sum)
  (left_column : 19 + 22 + b = magic_sum)
  (middle_column : w + y + 18 = magic_sum)
  (right_column : 23 + a + z = magic_sum)
  (main_diagonal : 19 + y + z = magic_sum)
  (secondary_diagonal : 23 + y + b = magic_sum)

/-- The sum of a and b in the magic square is 23 -/
theorem magic_square_a_plus_b (ms : MagicSquare) : ms.a + ms.b = 23 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_a_plus_b_l3241_324150


namespace NUMINAMATH_CALUDE_power_calculation_l3241_324169

theorem power_calculation : (9^4 * 3^10) / 27^7 = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3241_324169


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3241_324184

theorem complex_equation_sum (a b : ℝ) (h : (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I)) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3241_324184


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3241_324192

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : y + 9 * x = x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ b + 9 * a = a * b → x + y ≤ a + b ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3241_324192


namespace NUMINAMATH_CALUDE_profit_share_difference_theorem_l3241_324153

/-- Represents an investor's contribution to the business --/
structure Investor where
  investment : ℕ
  duration : ℕ

/-- Calculates the difference in profit shares between two investors --/
def profit_share_difference (suresh rohan sudhir : Investor) (total_profit : ℕ) : ℕ :=
  let total_investment_months := suresh.investment * suresh.duration + 
                                 rohan.investment * rohan.duration + 
                                 sudhir.investment * sudhir.duration
  let rohan_share := (rohan.investment * rohan.duration * total_profit) / total_investment_months
  let sudhir_share := (sudhir.investment * sudhir.duration * total_profit) / total_investment_months
  rohan_share - sudhir_share

/-- Theorem stating the difference in profit shares --/
theorem profit_share_difference_theorem (suresh rohan sudhir : Investor) (total_profit : ℕ) :
  suresh.investment = 18000 ∧ suresh.duration = 12 ∧
  rohan.investment = 12000 ∧ rohan.duration = 9 ∧
  sudhir.investment = 9000 ∧ sudhir.duration = 8 ∧
  total_profit = 3795 →
  profit_share_difference suresh rohan sudhir total_profit = 345 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_theorem_l3241_324153


namespace NUMINAMATH_CALUDE_units_digit_sum_base_8_l3241_324186

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ :=
  n % base

/-- Addition in a given base -/
def baseAddition (a b base : ℕ) : ℕ :=
  (a + b) % base^2

theorem units_digit_sum_base_8 :
  unitsDigit (baseAddition 35 47 8) 8 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base_8_l3241_324186


namespace NUMINAMATH_CALUDE_infinitely_many_composite_sums_l3241_324146

theorem infinitely_many_composite_sums : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k n : ℕ), ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + (f k)^4 = x * y :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_sums_l3241_324146


namespace NUMINAMATH_CALUDE_square_area_on_parabola_prove_square_area_l3241_324140

theorem square_area_on_parabola : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola
      x₁^2 + 4*x₁ + 3 = 6 ∧
      x₂^2 + 4*x₂ + 3 = 6 ∧
      -- The side length is the distance between x-coordinates
      (x₂ - x₁)^2 = area ∧
      -- The area is 28
      area = 28

theorem prove_square_area : square_area_on_parabola 28 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_prove_square_area_l3241_324140


namespace NUMINAMATH_CALUDE_easter_egg_ratio_l3241_324180

def total_eggs : ℕ := 63
def hannah_eggs : ℕ := 42

theorem easter_egg_ratio :
  let helen_eggs := total_eggs - hannah_eggs
  (hannah_eggs : ℚ) / helen_eggs = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_easter_egg_ratio_l3241_324180


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3241_324193

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) : 
  Real.sin (2 * α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3241_324193


namespace NUMINAMATH_CALUDE_subset_collection_m_eq_seven_l3241_324166

/-- A structure representing a collection of 3-element subsets of {1, ..., n} -/
structure SubsetCollection (n : ℕ) where
  m : ℕ
  subsets : Fin m → Finset (Fin n)
  m_gt_one : m > 1
  three_elements : ∀ i, (subsets i).card = 3
  unique_pairs : ∀ {x y : Fin n}, x ≠ y → ∃! i, {x, y} ⊆ subsets i
  one_common : ∀ {i j : Fin m}, i ≠ j → ∃! x, x ∈ subsets i ∩ subsets j

/-- The main theorem stating that for any valid SubsetCollection, m = 7 -/
theorem subset_collection_m_eq_seven {n : ℕ} (sc : SubsetCollection n) : sc.m = 7 :=
sorry

end NUMINAMATH_CALUDE_subset_collection_m_eq_seven_l3241_324166


namespace NUMINAMATH_CALUDE_sasha_took_right_triangle_l3241_324100

-- Define the triangle types
inductive TriangleType
  | Acute
  | Right
  | Obtuse

-- Define a function to check if two triangles can form the third
def canFormThird (t1 t2 t3 : TriangleType) : Prop :=
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) ∧
  ((t1 = TriangleType.Acute ∧ t2 = TriangleType.Obtuse) ∨
   (t1 = TriangleType.Obtuse ∧ t2 = TriangleType.Acute)) ∧
  t3 = TriangleType.Right

-- Theorem statement
theorem sasha_took_right_triangle (t1 t2 t3 : TriangleType) :
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) →
  canFormThird t1 t2 t3 →
  t3 = TriangleType.Right :=
by sorry

end NUMINAMATH_CALUDE_sasha_took_right_triangle_l3241_324100


namespace NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l3241_324142

theorem xy_positive_sufficient_not_necessary (x y : ℝ) :
  (x * y > 0 → |x + y| = |x| + |y|) ∧
  ¬(∀ x y : ℝ, |x + y| = |x| + |y| → x * y > 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l3241_324142


namespace NUMINAMATH_CALUDE_tile_arrangements_l3241_324108

def num_red_tiles : ℕ := 1
def num_blue_tiles : ℕ := 2
def num_green_tiles : ℕ := 2
def num_yellow_tiles : ℕ := 4

def total_tiles : ℕ := num_red_tiles + num_blue_tiles + num_green_tiles + num_yellow_tiles

theorem tile_arrangements :
  (total_tiles.factorial) / (num_red_tiles.factorial * num_blue_tiles.factorial * num_green_tiles.factorial * num_yellow_tiles.factorial) = 3780 :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangements_l3241_324108


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l3241_324135

variable (A B C G : ℝ × ℝ)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property (h_centroid : is_centroid G A B C)
  (h_condition : distance_squared G A + 2 * distance_squared G B + 3 * distance_squared G C = 123) :
  distance_squared A B + distance_squared A C + distance_squared B C = 246 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l3241_324135


namespace NUMINAMATH_CALUDE_logical_equivalence_l3241_324195

theorem logical_equivalence (P Q : Prop) :
  (¬P → ¬Q) ↔ (Q → P) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3241_324195


namespace NUMINAMATH_CALUDE_pecan_pies_count_l3241_324112

/-- The number of pecan pies baked by Mrs. Hilt -/
def pecan_pies : ℝ := 16

/-- The number of apple pies baked by Mrs. Hilt -/
def apple_pies : ℝ := 14

/-- The factor by which the total number of pies needs to be increased -/
def increase_factor : ℝ := 5

/-- The total number of pies needed -/
def total_pies_needed : ℝ := 150

/-- Theorem stating that the number of pecan pies is correct given the conditions -/
theorem pecan_pies_count : 
  increase_factor * (pecan_pies + apple_pies) = total_pies_needed := by
  sorry

end NUMINAMATH_CALUDE_pecan_pies_count_l3241_324112


namespace NUMINAMATH_CALUDE_arctan_sum_greater_than_pi_half_l3241_324113

theorem arctan_sum_greater_than_pi_half (a b : ℝ) : 
  a = 2/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b > π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_greater_than_pi_half_l3241_324113


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l3241_324110

theorem fourth_rectangle_area (total_area : ℝ) (area1 area2 area3 : ℝ) :
  total_area = 168 ∧ 
  area1 = 33 ∧ 
  area2 = 45 ∧ 
  area3 = 20 →
  total_area - (area1 + area2 + area3) = 70 :=
by sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l3241_324110


namespace NUMINAMATH_CALUDE_pencil_distribution_remainder_l3241_324122

theorem pencil_distribution_remainder : 25197629 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_remainder_l3241_324122


namespace NUMINAMATH_CALUDE_correct_operation_l3241_324129

theorem correct_operation (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3241_324129


namespace NUMINAMATH_CALUDE_solve_for_a_l3241_324139

theorem solve_for_a : ∃ a : ℝ, 
  (2 : ℝ) - a * (1 : ℝ) = -1 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3241_324139


namespace NUMINAMATH_CALUDE_factor_iff_t_eq_neg_six_or_one_l3241_324127

/-- The polynomial in question -/
def f (x : ℝ) : ℝ := 4 * x^2 + 20 * x - 24

/-- Theorem stating that x - t is a factor of f(x) if and only if t is -6 or 1 -/
theorem factor_iff_t_eq_neg_six_or_one :
  ∀ t : ℝ, (∃ g : ℝ → ℝ, ∀ x, f x = (x - t) * g x) ↔ (t = -6 ∨ t = 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_iff_t_eq_neg_six_or_one_l3241_324127


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3241_324175

theorem cube_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3241_324175


namespace NUMINAMATH_CALUDE_parabola_one_y_intercept_l3241_324118

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define what a y-intercept is
def is_y_intercept (y : ℝ) : Prop := f 0 = y

-- Theorem: The parabola has exactly one y-intercept
theorem parabola_one_y_intercept :
  ∃! y : ℝ, is_y_intercept y :=
sorry

end NUMINAMATH_CALUDE_parabola_one_y_intercept_l3241_324118


namespace NUMINAMATH_CALUDE_rachel_bought_three_tables_l3241_324119

/-- Represents the number of minutes spent on each piece of furniture -/
def time_per_furniture : ℕ := 4

/-- Represents the total number of chairs bought -/
def num_chairs : ℕ := 7

/-- Represents the total time spent assembling all furniture -/
def total_time : ℕ := 40

/-- Calculates the number of tables bought -/
def num_tables : ℕ :=
  (total_time - time_per_furniture * num_chairs) / time_per_furniture

theorem rachel_bought_three_tables :
  num_tables = 3 :=
sorry

end NUMINAMATH_CALUDE_rachel_bought_three_tables_l3241_324119


namespace NUMINAMATH_CALUDE_total_cost_of_kept_shirts_l3241_324114

def all_shirts : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def returned_shirts : List ℕ := [20, 25, 30, 22, 23, 29]

theorem total_cost_of_kept_shirts :
  (all_shirts.sum - returned_shirts.sum) = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_kept_shirts_l3241_324114


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3241_324157

/-- Given a parabola y^2 = 4x and a line passing through its focus intersecting 
    the parabola at points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 6, 
    prove that the length |PQ| = 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- P is on the parabola
  y₂^2 = 4*x₂ →  -- Q is on the parabola
  x₁ + x₂ = 6 →  -- Given condition
  (∃ t : ℝ, t*x₁ + (1-t)*1 = 0 ∧ t*y₁ = 0) →  -- Line PQ passes through focus (1,0)
  (∃ s : ℝ, s*x₂ + (1-s)*1 = 0 ∧ s*y₂ = 0) →  -- Line PQ passes through focus (1,0)
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 8 :=  -- |PQ| = 8
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3241_324157


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3241_324148

/-- 
Given a rectangular field with one side of length 20 feet and a perimeter 
(excluding that side) of 85 feet, the area of the field is 650 square feet.
-/
theorem rectangular_field_area : 
  ∀ (length width : ℝ), 
    length = 20 →
    2 * width + length = 85 →
    length * width = 650 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3241_324148


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3241_324149

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3241_324149


namespace NUMINAMATH_CALUDE_nine_digit_multiplier_problem_l3241_324137

theorem nine_digit_multiplier_problem : 
  ∃! (N : ℕ), 
    (100000000 ≤ N ∧ N ≤ 999999999) ∧ 
    (N * 123456789) % 1000000000 = 987654321 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_multiplier_problem_l3241_324137


namespace NUMINAMATH_CALUDE_range_of_a_given_points_on_opposite_sides_l3241_324189

/-- Given points M(1, -a) and N(a, 1) are on opposite sides of the line 2x-3y+1=0,
    prove that the range of the real number a is -1 < a < 1. -/
theorem range_of_a_given_points_on_opposite_sides (a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (1, -a) ∧ 
    N = (a, 1) ∧ 
    (2 * M.1 - 3 * M.2 + 1) * (2 * N.1 - 3 * N.2 + 1) < 0) →
  -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_points_on_opposite_sides_l3241_324189


namespace NUMINAMATH_CALUDE_min_value_product_squares_l3241_324106

theorem min_value_product_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 3) :
  x^2 * y^2 * z^2 ≥ 1/64 ∧ ∃ (a : ℝ), a > 0 ∧ a^2 * a^2 * a^2 = 1/64 ∧ 1/a + 1/a + 1/a = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_squares_l3241_324106


namespace NUMINAMATH_CALUDE_polynomial_root_product_l3241_324159

theorem polynomial_root_product (k : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁^4 - 18*x₁^3 + k*x₁^2 + 200*x₁ - 1984 = 0) ∧
    (x₂^4 - 18*x₂^3 + k*x₂^2 + 200*x₂ - 1984 = 0) ∧
    (x₃^4 - 18*x₃^3 + k*x₃^2 + 200*x₃ - 1984 = 0) ∧
    (x₄^4 - 18*x₄^3 + k*x₄^2 + 200*x₄ - 1984 = 0) ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ 
     x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32)) →
  k = 86 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l3241_324159


namespace NUMINAMATH_CALUDE_complex_solutions_count_l3241_324105

/-- The equation (z^3 - 1) / (z^2 + z - 6) = 0 has exactly 3 complex solutions. -/
theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 + z - 6) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 + z - 6) = 0 → z ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l3241_324105


namespace NUMINAMATH_CALUDE_chicken_count_l3241_324126

theorem chicken_count (C : ℚ) 
  (roosters : ℚ → ℚ) (hens : ℚ → ℚ) (laying_hens : ℚ → ℚ) (non_laying : ℚ) :
  roosters C = (1 / 4) * C →
  hens C = (3 / 4) * C →
  laying_hens C = (3 / 4) * hens C →
  roosters C + (hens C - laying_hens C) = 35 →
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l3241_324126


namespace NUMINAMATH_CALUDE_joans_books_l3241_324121

theorem joans_books (tom_books : ℕ) (total_books : ℕ) (h1 : tom_books = 38) (h2 : total_books = 48) :
  total_books - tom_books = 10 := by
sorry

end NUMINAMATH_CALUDE_joans_books_l3241_324121


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l3241_324173

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l3241_324173


namespace NUMINAMATH_CALUDE_pure_imaginary_second_quadrant_l3241_324160

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number if and only if m = 3
theorem pure_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3 := by
  sorry

-- Theorem 2: z is in the second quadrant if and only if -1 < m < 3
theorem second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_second_quadrant_l3241_324160


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3241_324128

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -182 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3241_324128


namespace NUMINAMATH_CALUDE_cube_root_nine_thirty_two_squared_l3241_324176

theorem cube_root_nine_thirty_two_squared :
  (((9 : ℝ) / 32) ^ (1/3 : ℝ)) ^ 2 = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_cube_root_nine_thirty_two_squared_l3241_324176


namespace NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l3241_324188

theorem sqrt_eight_and_one_ninth (x : ℝ) : x = Real.sqrt (8 + 1/9) → x = Real.sqrt 73 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l3241_324188


namespace NUMINAMATH_CALUDE_weekly_commute_cost_l3241_324154

-- Define the parameters
def workDays : ℕ := 5
def carToll : ℚ := 12.5
def motorcycleToll : ℚ := 7
def milesPerGallon : ℚ := 35
def commuteDistance : ℚ := 14
def gasPrice : ℚ := 3.75
def carTrips : ℕ := 3
def motorcycleTrips : ℕ := 2

-- Define the theorem
theorem weekly_commute_cost :
  let carTollCost := carToll * carTrips
  let motorcycleTollCost := motorcycleToll * motorcycleTrips
  let totalDistance := commuteDistance * 2 * workDays
  let totalGasUsed := totalDistance / milesPerGallon
  let gasCost := totalGasUsed * gasPrice
  let totalCost := carTollCost + motorcycleTollCost + gasCost
  totalCost = 59 := by sorry

end NUMINAMATH_CALUDE_weekly_commute_cost_l3241_324154


namespace NUMINAMATH_CALUDE_third_sample_is_51_l3241_324107

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : Nat
  numGroups : Nat
  firstSample : Nat

/-- Calculates the sample for a given group in a systematic sampling -/
def getSample (s : SystematicSampling) (group : Nat) : Nat :=
  s.firstSample + (group - 1) * (s.totalItems / s.numGroups)

/-- Theorem: In a systematic sampling of 400 items into 20 groups, 
    if the first sample is 11, then the third sample will be 51 -/
theorem third_sample_is_51 (s : SystematicSampling) 
  (h1 : s.totalItems = 400) 
  (h2 : s.numGroups = 20) 
  (h3 : s.firstSample = 11) : 
  getSample s 3 = 51 := by
  sorry

/-- Example setup for the given problem -/
def exampleSampling : SystematicSampling := {
  totalItems := 400
  numGroups := 20
  firstSample := 11
}

#eval getSample exampleSampling 3

end NUMINAMATH_CALUDE_third_sample_is_51_l3241_324107


namespace NUMINAMATH_CALUDE_composite_number_l3241_324116

theorem composite_number (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 6 * 2^(2^(4*n)) + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l3241_324116


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l3241_324130

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 7

/-- A large triangle is made up of this many smaller triangles -/
def triangles_per_large : ℕ := 4

/-- The number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors num_corners -- All corners different
  let total_corner_combinations := corner_same + corner_two_same + corner_all_diff
  total_corner_combinations * num_colors -- Multiply by center triangle color choices

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 588 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l3241_324130


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l3241_324183

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a > 1 ∧ a ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l3241_324183


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l3241_324172

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → m * x^2 - m * x - 1 < -m + 5) ↔ 
  m > 0 ∧ m < 6/7 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l3241_324172


namespace NUMINAMATH_CALUDE_bisection_method_root_existence_l3241_324168

theorem bisection_method_root_existence
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_cont : ContinuousOn f (Set.Icc a b))
  (h_sign : f a * f b < 0)
  (h_a_neg : f a < 0)
  (h_b_pos : f b > 0)
  (h_mid_pos : f ((a + b) / 2) > 0) :
  ∃ x ∈ Set.Ioo a ((a + b) / 2), f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_root_existence_l3241_324168


namespace NUMINAMATH_CALUDE_convention_handshakes_l3241_324187

/-- The number of handshakes in a convention with multiple companies -/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 5 companies, each having 4 representatives,
    where every person shakes hands once with every person except those
    from their own company, the total number of handshakes is 160. -/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l3241_324187


namespace NUMINAMATH_CALUDE_parabola_tangent_values_l3241_324156

/-- A parabola tangent to a line -/
structure ParabolaTangentToLine where
  /-- Coefficient of x^2 term in the parabola equation -/
  a : ℝ
  /-- Coefficient of x term in the parabola equation -/
  b : ℝ
  /-- The parabola y = ax^2 + bx is tangent to the line y = 2x + 4 -/
  is_tangent : ∃ (x : ℝ), a * x^2 + b * x = 2 * x + 4
  /-- The x-coordinate of the point of tangency is 1 -/
  tangent_point : ∃ (y : ℝ), a * 1^2 + b * 1 = 2 * 1 + 4 ∧ a * 1^2 + b * 1 = y

/-- The values of a and b for the parabola tangent to the line -/
theorem parabola_tangent_values (p : ParabolaTangentToLine) : p.a = 4/3 ∧ p.b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_values_l3241_324156


namespace NUMINAMATH_CALUDE_max_towns_is_four_l3241_324191

/-- Represents the type of connection between two towns -/
inductive Connection
  | Air
  | Bus
  | Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents the network of towns and their connections -/
structure TownNetwork where
  towns : Finset Town
  connections : Town → Town → Option Connection

/-- Checks if the given network satisfies all conditions -/
def satisfiesConditions (network : TownNetwork) : Prop :=
  -- Condition 1: Each pair of towns is directly linked by just one of air, bus, or train
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! c : Connection, network.connections t1 t2 = some c) ∧
  -- Condition 2: At least one pair is linked by each type of connection
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Air) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Bus) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Train) ∧
  -- Condition 3: No town has all three types of connections
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ t1 t2 t3 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ t3 ∈ network.towns ∧
      network.connections t t1 = some Connection.Air ∧
      network.connections t t2 = some Connection.Bus ∧
      network.connections t t3 = some Connection.Train)) ∧
  -- Condition 4: No three towns have all connections of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 → t2 ≠ t3 → t1 ≠ t3 →
    ¬(network.connections t1 t2 = network.connections t2 t3 ∧
      network.connections t2 t3 = network.connections t1 t3))

/-- The main theorem stating that the maximum number of towns satisfying the conditions is 4 -/
theorem max_towns_is_four :
  (∃ (network : TownNetwork), satisfiesConditions network ∧ network.towns.card = 4) ∧
  (∀ (network : TownNetwork), satisfiesConditions network → network.towns.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_towns_is_four_l3241_324191


namespace NUMINAMATH_CALUDE_solve_system_l3241_324144

theorem solve_system (x y : ℚ) (eq1 : 2 * x - 3 * y = 15) (eq2 : x + 2 * y = 8) : x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3241_324144
