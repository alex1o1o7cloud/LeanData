import Mathlib

namespace sequence_terms_l3861_386143

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sequence_terms : a 3 = 5 ∧ a 5 = 9 := by sorry

end sequence_terms_l3861_386143


namespace calculate_income_l3861_386187

/-- Given a person's income and expenditure ratio, and their savings, calculate their income. -/
theorem calculate_income (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →  -- income to expenditure ratio is 5:4
  income - expenditure = savings → -- savings definition
  savings = 4000 → -- given savings amount
  income = 20000 := by
  sorry

end calculate_income_l3861_386187


namespace purely_imaginary_trajectory_l3861_386128

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ y

theorem purely_imaginary_trajectory (x y : ℝ) :
  is_purely_imaginary ((x^2 + y^2 - 4 : ℝ) + (x - y) * I) ↔ trajectory x y :=
sorry

end purely_imaginary_trajectory_l3861_386128


namespace negative_five_greater_than_negative_seven_l3861_386142

theorem negative_five_greater_than_negative_seven : -5 > -7 := by
  sorry

end negative_five_greater_than_negative_seven_l3861_386142


namespace car_speed_adjustment_l3861_386114

theorem car_speed_adjustment (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) :
  distance = 324 →
  original_time = 6 →
  time_factor = 3 / 2 →
  (distance / (original_time * time_factor)) = 36 := by
  sorry

end car_speed_adjustment_l3861_386114


namespace expression_simplification_l3861_386105

theorem expression_simplification : 
  ((0.2 * 0.4 - (0.3 / 0.5)) + ((0.6 * 0.8 + (0.1 / 0.2)) - (0.9 * (0.3 - 0.2 * 0.4)))^2) * (1 - (0.4^2 / (0.2 * 0.8))) = 0 := by
  sorry

end expression_simplification_l3861_386105


namespace man_speed_calculation_man_speed_specific_case_l3861_386146

/-- Calculates the speed of a man walking in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- The speed of a man walking in the same direction as a train, given specific conditions. -/
theorem man_speed_specific_case : 
  ∃ (ε : Real), ε > 0 ∧ 
  |man_speed_calculation 100 63 5.999520038396929 - 0.831946| < ε :=
sorry

end man_speed_calculation_man_speed_specific_case_l3861_386146


namespace toy_store_inventory_l3861_386192

/-- Calculates the final number of games in a toy store's inventory --/
theorem toy_store_inventory (initial : ℕ) (sold : ℕ) (received : ℕ) :
  initial = 95 →
  sold = 68 →
  received = 47 →
  initial - sold + received = 74 := by
  sorry

end toy_store_inventory_l3861_386192


namespace banana_permutations_count_l3861_386119

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of times 'A' appears in "BANANA" -/
def count_A : ℕ := 3

/-- The number of times 'N' appears in "BANANA" -/
def count_N : ℕ := 2

/-- Theorem stating that the number of distinct permutations of the letters in "BANANA" is 60 -/
theorem banana_permutations_count : 
  banana_permutations = (Nat.factorial total_letters) / ((Nat.factorial count_A) * (Nat.factorial count_N)) :=
by sorry

end banana_permutations_count_l3861_386119


namespace largest_divisible_sum_fourth_powers_l3861_386110

/-- A set of n prime numbers greater than 10 -/
def PrimeSet (n : ℕ) := { S : Finset ℕ | S.card = n ∧ ∀ p ∈ S, Nat.Prime p ∧ p > 10 }

/-- The sum of fourth powers of elements in a finite set -/
def SumFourthPowers (S : Finset ℕ) : ℕ := S.sum (λ x => x^4)

/-- The main theorem statement -/
theorem largest_divisible_sum_fourth_powers :
  ∀ n > 240, ∃ S ∈ PrimeSet n, ¬ (n ∣ SumFourthPowers S) ∧
  ∀ m ≤ 240, ∀ T ∈ PrimeSet m, m ∣ SumFourthPowers T :=
sorry

end largest_divisible_sum_fourth_powers_l3861_386110


namespace hyperbola_eccentricity_l3861_386100

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote x - 2y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by sorry

end hyperbola_eccentricity_l3861_386100


namespace stock_price_increase_l3861_386134

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 8) 
  (h2 : closing_price = 9) : 
  (closing_price - opening_price) / opening_price * 100 = 12.5 := by
  sorry

end stock_price_increase_l3861_386134


namespace sum_of_reciprocal_relations_l3861_386172

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end sum_of_reciprocal_relations_l3861_386172


namespace square_root_of_25_l3861_386111

theorem square_root_of_25 : Real.sqrt 25 = 5 ∨ Real.sqrt 25 = -5 := by
  sorry

end square_root_of_25_l3861_386111


namespace lamp_distribution_and_profit_l3861_386141

/-- Represents the types of lamps --/
inductive LampType
| A
| B

/-- Represents the purchase price of a lamp --/
def purchasePrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 40
  | LampType.B => 65

/-- Represents the selling price of a lamp --/
def sellingPrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 60
  | LampType.B => 100

/-- Represents the profit from selling a lamp --/
def profit (t : LampType) : ℕ := sellingPrice t - purchasePrice t

/-- The total number of lamps --/
def totalLamps : ℕ := 50

/-- The total purchase cost --/
def totalPurchaseCost : ℕ := 2500

/-- The minimum total profit --/
def minTotalProfit : ℕ := 1400

theorem lamp_distribution_and_profit :
  (∃ (x y : ℕ),
    x + y = totalLamps ∧
    x * purchasePrice LampType.A + y * purchasePrice LampType.B = totalPurchaseCost ∧
    x = 30 ∧ y = 20) ∧
  (∃ (m : ℕ),
    m * profit LampType.B + (totalLamps - m) * profit LampType.A ≥ minTotalProfit ∧
    m ≥ 27 ∧
    ∀ (n : ℕ), n * profit LampType.B + (totalLamps - n) * profit LampType.A ≥ minTotalProfit → n ≥ m) :=
by sorry

end lamp_distribution_and_profit_l3861_386141


namespace water_evaporation_per_day_l3861_386135

/-- Proves that given a bowl with 10 ounces of water, where 0.04% of the original amount
    evaporates over 50 days, the amount of water evaporated each day is 0.0008 ounces. -/
theorem water_evaporation_per_day
  (initial_water : Real)
  (evaporation_period : Nat)
  (evaporation_percentage : Real)
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.04)
  : (initial_water * evaporation_percentage / 100) / evaporation_period = 0.0008 := by
  sorry

#check water_evaporation_per_day

end water_evaporation_per_day_l3861_386135


namespace first_terrific_tuesday_l3861_386129

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The fiscal year starts on Tuesday, February 1 -/
def fiscalYearStart : Date := { day := 1, month := 2 }

/-- The day of the week for the fiscal year start -/
def fiscalYearStartDay : DayOfWeek := DayOfWeek.Tuesday

/-- Function to determine if a given date is a Terrific Tuesday -/
def isTerrificTuesday (d : Date) : Prop := sorry

/-- The first Terrific Tuesday after the fiscal year starts -/
def firstTerrificTuesday : Date := { day := 29, month := 3 }

/-- Theorem stating that the first Terrific Tuesday after the fiscal year starts is March 29 -/
theorem first_terrific_tuesday :
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ d : Date, d.month < firstTerrificTuesday.month ∨ 
    (d.month = firstTerrificTuesday.month ∧ d.day < firstTerrificTuesday.day) → 
    ¬isTerrificTuesday d) :=
by sorry

end first_terrific_tuesday_l3861_386129


namespace radio_selling_price_l3861_386173

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def sellingPrice (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Theorem stating that a radio purchased for Rs 490 with a 5% loss has a selling price of Rs 465.5. -/
theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 := by
  sorry

#eval sellingPrice 490 5

end radio_selling_price_l3861_386173


namespace removed_number_theorem_l3861_386177

theorem removed_number_theorem (n : ℕ) (m : ℕ) :
  m ≤ n →
  (n * (n + 1) / 2 - m) / (n - 1) = 163/4 →
  m = 61 := by
sorry

end removed_number_theorem_l3861_386177


namespace solution_and_minimum_value_l3861_386165

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem solution_and_minimum_value :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry

end solution_and_minimum_value_l3861_386165


namespace arithmetic_mean_geq_product_l3861_386155

theorem arithmetic_mean_geq_product (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end arithmetic_mean_geq_product_l3861_386155


namespace line_intersects_x_axis_l3861_386171

/-- The line equation 5y - 3x = 15 intersects the x-axis at the point (-5, 0). -/
theorem line_intersects_x_axis :
  ∃ (x y : ℝ), 5 * y - 3 * x = 15 ∧ y = 0 ∧ x = -5 := by
  sorry

end line_intersects_x_axis_l3861_386171


namespace horner_method_f_at_3_l3861_386169

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^3 + 3x^2 - x + 1 -/
def f : List ℝ := [1, -2, 3, 0, -1, 1]

/-- Theorem: Horner's method applied to f(x) at x = 3 yields v₃ = 24 -/
theorem horner_method_f_at_3 :
  horner f 3 = 24 := by
  sorry

#eval horner f 3

end horner_method_f_at_3_l3861_386169


namespace village_population_equation_l3861_386196

/-- The initial population of a village in Sri Lanka -/
def initial_population : ℕ := 4500

/-- The fraction of people who survived the bombardment -/
def survival_rate : ℚ := 9/10

/-- The fraction of people who remained in the village after some left due to fear -/
def remaining_rate : ℚ := 4/5

/-- The final population of the village -/
def final_population : ℕ := 3240

/-- Theorem stating that the initial population satisfies the given conditions -/
theorem village_population_equation :
  ↑initial_population * (survival_rate * remaining_rate) = final_population := by
  sorry

end village_population_equation_l3861_386196


namespace tetrahedron_triangles_l3861_386103

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles : distinct_triangles = 4 := by
  sorry

end tetrahedron_triangles_l3861_386103


namespace fixed_point_exponential_function_l3861_386121

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end fixed_point_exponential_function_l3861_386121


namespace inscribed_sphere_pyramid_volume_l3861_386170

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- Side length of the base of the pyramid -/
  a : ℝ
  /-- The sphere touches the base and all lateral faces -/
  sphere_touches_all_faces : True
  /-- The sphere divides the height in a 4:5 ratio from the apex -/
  height_ratio : True

/-- Volume of the pyramid -/
noncomputable def pyramid_volume (p : InscribedSpherePyramid) : ℝ :=
  2 * p.a^3 / 5

/-- Theorem stating the volume of the pyramid -/
theorem inscribed_sphere_pyramid_volume (p : InscribedSpherePyramid) :
  pyramid_volume p = 2 * p.a^3 / 5 := by
  sorry

end inscribed_sphere_pyramid_volume_l3861_386170


namespace salary_unspent_fraction_l3861_386152

theorem salary_unspent_fraction (salary : ℝ) (salary_positive : salary > 0) :
  let first_week_spent := (1 / 4 : ℝ) * salary
  let each_other_week_spent := (1 / 5 : ℝ) * salary
  let total_spent := first_week_spent + 3 * each_other_week_spent
  (salary - total_spent) / salary = 3 / 20 := by
  sorry

end salary_unspent_fraction_l3861_386152


namespace solution_set_of_equation_l3861_386109

theorem solution_set_of_equation (x y : ℝ) : 
  (Real.sqrt (3 * x - 1) + abs (2 * y + 2) = 0) ↔ (x = 1/3 ∧ y = -1) :=
sorry

end solution_set_of_equation_l3861_386109


namespace gcf_lcm_product_8_12_l3861_386133

theorem gcf_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end gcf_lcm_product_8_12_l3861_386133


namespace building_height_calculation_l3861_386123

/-- Given a flagstaff and a building casting shadows under the same sun angle, 
    calculate the height of the building. -/
theorem building_height_calculation 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_flagstaff : flagstaff_height = 17.5) 
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  (flagstaff_height * building_shadow) / flagstaff_shadow = 12.5 := by
  sorry

end building_height_calculation_l3861_386123


namespace part_one_part_two_l3861_386145

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0)
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, q x ∧ p x a) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end part_one_part_two_l3861_386145


namespace parabola_focus_focus_of_specific_parabola_l3861_386124

/-- The focus of a parabola y = ax^2 + c is at (0, 1/(4a) + c) -/
theorem parabola_focus (a c : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + c)
  ∀ x y : ℝ, y = a * x^2 + c → (x - f.1)^2 + (y - f.2)^2 = (y - c + 1/(4*a))^2 :=
by sorry

/-- The focus of the parabola y = 9x^2 - 5 is at (0, -179/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, -179/36)
  ∀ x y : ℝ, y = 9 * x^2 - 5 → (x - f.1)^2 + (y - f.2)^2 = (y + 5 + 1/36)^2 :=
by sorry

end parabola_focus_focus_of_specific_parabola_l3861_386124


namespace average_sleep_is_eight_l3861_386136

def monday_sleep : ℕ := 8
def tuesday_sleep : ℕ := 7
def wednesday_sleep : ℕ := 8
def thursday_sleep : ℕ := 10
def friday_sleep : ℕ := 7

def total_days : ℕ := 5

def total_sleep : ℕ := monday_sleep + tuesday_sleep + wednesday_sleep + thursday_sleep + friday_sleep

theorem average_sleep_is_eight :
  (total_sleep : ℚ) / total_days = 8 := by sorry

end average_sleep_is_eight_l3861_386136


namespace number_of_divisors_of_2002_l3861_386144

theorem number_of_divisors_of_2002 : ∃ (d : ℕ → ℕ), d 2002 = 16 := by
  sorry

end number_of_divisors_of_2002_l3861_386144


namespace proposition_q_false_l3861_386198

theorem proposition_q_false (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬((¬p) ∧ q)) : 
  ¬q := by
  sorry

end proposition_q_false_l3861_386198


namespace count_hyperbola_integer_tangent_points_l3861_386125

/-- The number of points on the hyperbola y = 2013/x where the tangent line
    intersects both coordinate axes at integer points -/
def hyperbola_integer_tangent_points : ℕ := 48

/-- The hyperbola equation y = 2013/x -/
def hyperbola (x y : ℝ) : Prop := y = 2013 / x

/-- Predicate for a point (x, y) on the hyperbola having a tangent line
    that intersects both axes at integer coordinates -/
def has_integer_intercepts (x y : ℝ) : Prop :=
  hyperbola x y ∧
  ∃ (x_int y_int : ℤ),
    (x_int ≠ 0 ∧ y_int ≠ 0) ∧
    (y - 2013 / x = -(2013 / x^2) * (x_int - x)) ∧
    (0 = -(2013 / x^2) * x_int + 2 * 2013 / x)

theorem count_hyperbola_integer_tangent_points :
  (∑' p : {p : ℝ × ℝ // has_integer_intercepts p.1 p.2}, 1) =
    hyperbola_integer_tangent_points :=
sorry

end count_hyperbola_integer_tangent_points_l3861_386125


namespace triangular_array_sum_l3861_386102

theorem triangular_array_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 3003 → (N / 10 + N % 10) = 14 := by
  sorry

end triangular_array_sum_l3861_386102


namespace valid_arrangements_count_l3861_386194

/-- Represents a seating arrangement in the minibus -/
def SeatingArrangement := Fin 6 → Fin 6

/-- Checks if a seating arrangement is valid (no sibling sits directly in front) -/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  ∀ i j : Fin 3, arr i ≠ arr (i + 3)

/-- The total number of valid seating arrangements -/
def total_valid_arrangements : ℕ := sorry

/-- Theorem stating that the number of valid seating arrangements is 12 -/
theorem valid_arrangements_count : total_valid_arrangements = 12 := by sorry

end valid_arrangements_count_l3861_386194


namespace quadratic_shift_sum_l3861_386174

/-- Given a quadratic function f(x) = 3x^2 + 5x + 9, when shifted 6 units to the left,
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 191. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3*x^2 + 5*x + 9) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a*x^2 + b*x + c) →
  a + b + c = 191 := by
sorry

end quadratic_shift_sum_l3861_386174


namespace max_boxes_fit_l3861_386181

def large_box_length : ℕ := 8
def large_box_width : ℕ := 7
def large_box_height : ℕ := 6

def small_box_length : ℕ := 4
def small_box_width : ℕ := 7
def small_box_height : ℕ := 6

def cm_per_meter : ℕ := 100

theorem max_boxes_fit (large_box_volume small_box_volume max_boxes : ℕ) : 
  large_box_volume = (large_box_length * cm_per_meter) * (large_box_width * cm_per_meter) * (large_box_height * cm_per_meter) →
  small_box_volume = small_box_length * small_box_width * small_box_height →
  max_boxes = large_box_volume / small_box_volume →
  max_boxes = 2000000 := by
  sorry

#check max_boxes_fit

end max_boxes_fit_l3861_386181


namespace extra_page_number_l3861_386112

/-- Given a book with 77 pages, if one page number is included three times
    instead of once, resulting in a sum of 3028, then the page number
    that was added extra times is 25. -/
theorem extra_page_number :
  let n : ℕ := 77
  let correct_sum := n * (n + 1) / 2
  let incorrect_sum := 3028
  ∃ k : ℕ, k ≤ n ∧ correct_sum + 2 * k = incorrect_sum ∧ k = 25 := by
  sorry

#check extra_page_number

end extra_page_number_l3861_386112


namespace product_in_N_not_in_M_l3861_386131

def M : Set ℤ := {x | ∃ m : ℤ, x = 3*m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 2}

theorem product_in_N_not_in_M (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) :
  (x * y) ∈ N ∧ (x * y) ∉ M := by
  sorry

end product_in_N_not_in_M_l3861_386131


namespace arithmetic_sequence_sum_l3861_386140

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 3 → d = 2 → last = 25 →
    last = a + (n - 1) * d →
    (n : ℝ) / 2 * (a + last) = 168 := by
  sorry

end arithmetic_sequence_sum_l3861_386140


namespace complement_of_M_in_U_l3861_386193

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define the set M
def M : Set Nat := {1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {2, 3} := by sorry

end complement_of_M_in_U_l3861_386193


namespace union_of_A_and_B_l3861_386101

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -1} := by
  sorry

end union_of_A_and_B_l3861_386101


namespace counterexample_exists_l3861_386120

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ a * c ≤ b * c := by
  sorry

end counterexample_exists_l3861_386120


namespace eventually_one_first_l3861_386104

/-- Represents a permutation of integers from 1 to 1993 -/
def Permutation := Fin 1993 → Fin 1993

/-- The reversal operation on a permutation -/
def reverseOperation (p : Permutation) : Permutation :=
  sorry

/-- Predicate to check if 1 is the first element in the permutation -/
def isOneFirst (p : Permutation) : Prop :=
  p 0 = 0

/-- Main theorem: The reversal operation will eventually make 1 the first element -/
theorem eventually_one_first (p : Permutation) : 
  ∃ n : ℕ, isOneFirst (n.iterate reverseOperation p) :=
sorry

end eventually_one_first_l3861_386104


namespace sam_win_probability_proof_l3861_386178

/-- The probability of hitting the target with one shot -/
def hit_probability : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_probability : ℚ := 3/5

/-- Sam wins when the total number of shots (including the last successful one) is odd -/
axiom sam_wins_on_odd : True

/-- The probability that Sam wins the game -/
def sam_win_probability : ℚ := 5/8

theorem sam_win_probability_proof : 
  sam_win_probability = hit_probability + miss_probability * miss_probability * sam_win_probability :=
sorry

end sam_win_probability_proof_l3861_386178


namespace consecutive_integers_sum_l3861_386122

theorem consecutive_integers_sum (n : ℕ) (h1 : n > 0) 
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070) :
  n + 5 = 347 := by
  sorry

end consecutive_integers_sum_l3861_386122


namespace election_win_condition_l3861_386106

theorem election_win_condition 
  (total_students : ℕ) 
  (boy_percentage : ℚ) 
  (girl_percentage : ℚ) 
  (male_vote_percentage : ℚ) 
  (h1 : total_students = 200)
  (h2 : boy_percentage = 3/5)
  (h3 : girl_percentage = 2/5)
  (h4 : boy_percentage + girl_percentage = 1)
  (h5 : male_vote_percentage = 27/40)
  : ∃ (female_vote_percentage : ℚ),
    female_vote_percentage ≥ 1/4 ∧
    (boy_percentage * male_vote_percentage + girl_percentage * female_vote_percentage) * total_students > total_students / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (boy_percentage * male_vote_percentage + girl_percentage * x) * total_students ≤ total_students / 2 :=
by sorry

end election_win_condition_l3861_386106


namespace negative_fraction_comparison_l3861_386160

theorem negative_fraction_comparison : -1/2 < -1/3 := by
  sorry

end negative_fraction_comparison_l3861_386160


namespace simple_interest_problem_l3861_386183

/-- Proves that for a principal of 800 at simple interest, if increasing the
    interest rate by 5% results in 400 more interest, then the time period is 10 years. -/
theorem simple_interest_problem (r : ℝ) (t : ℝ) :
  (800 * r * t / 100) + 400 = 800 * (r + 5) * t / 100 →
  t = 10 :=
by sorry

end simple_interest_problem_l3861_386183


namespace min_value_parabola_vectors_l3861_386127

/-- Given a parabola y² = 2px where p > 0, prove that the minimum value of 
    |⃗OA + ⃗OB|² - |⃗AB|² for any two distinct points A and B on the parabola is -4p² -/
theorem min_value_parabola_vectors (p : ℝ) (hp : p > 0) :
  ∃ (min : ℝ), min = -4 * p^2 ∧
  ∀ (A B : ℝ × ℝ), A ≠ B →
  (A.2)^2 = 2 * p * A.1 →
  (B.2)^2 = 2 * p * B.1 →
  (A.1 + B.1)^2 + (A.2 + B.2)^2 - ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min :=
by sorry


end min_value_parabola_vectors_l3861_386127


namespace trig_identities_l3861_386108

/-- Proof of trigonometric identities -/
theorem trig_identities :
  (Real.cos (π / 3) + Real.sin (π / 4) - Real.tan (π / 4) = (-1 + Real.sqrt 2) / 2) ∧
  (6 * (Real.tan (π / 6))^2 - Real.sqrt 3 * Real.sin (π / 3) - 2 * Real.cos (π / 4) = 1 / 2 - Real.sqrt 2) :=
by sorry

end trig_identities_l3861_386108


namespace circle_C_equation_l3861_386166

/-- A circle C in the xy-plane -/
structure CircleC where
  /-- x-coordinate of a point on the circle -/
  x : ℝ → ℝ
  /-- y-coordinate of a point on the circle -/
  y : ℝ → ℝ
  /-- The parameter θ ranges over all real numbers -/
  θ : ℝ
  /-- x-coordinate is defined as 2 + 2cos(θ) -/
  x_eq : x θ = 2 + 2 * Real.cos θ
  /-- y-coordinate is defined as 2sin(θ) -/
  y_eq : y θ = 2 * Real.sin θ

/-- The standard equation of circle C -/
def standard_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- Theorem stating that the parametric equations of CircleC satisfy its standard equation -/
theorem circle_C_equation (c : CircleC) :
  ∀ θ, standard_equation c (c.x θ) (c.y θ) := by
  sorry

end circle_C_equation_l3861_386166


namespace polygon_with_20_diagonals_has_8_sides_l3861_386182

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 diagonals has 8 sides -/
theorem polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end polygon_with_20_diagonals_has_8_sides_l3861_386182


namespace initial_principal_is_500_l3861_386167

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial principal must be $500 -/
theorem initial_principal_is_500 :
  ∃ (rate : ℝ),
    simpleInterest 500 rate 2 = 590 ∧
    simpleInterest 500 rate 7 = 815 :=
by
  sorry

#check initial_principal_is_500

end initial_principal_is_500_l3861_386167


namespace tory_sold_seven_guns_l3861_386164

/-- The number of toy guns Tory sold -/
def tory_guns : ℕ := sorry

/-- The price of each toy phone Bert sold -/
def bert_phone_price : ℕ := 18

/-- The number of toy phones Bert sold -/
def bert_phones : ℕ := 8

/-- The price of each toy gun Tory sold -/
def tory_gun_price : ℕ := 20

/-- The difference in earnings between Bert and Tory -/
def earning_difference : ℕ := 4

theorem tory_sold_seven_guns :
  tory_guns = 7 :=
by
  sorry

end tory_sold_seven_guns_l3861_386164


namespace g_shifted_l3861_386118

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_shifted (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 := by
  sorry

end g_shifted_l3861_386118


namespace min_value_theorem_l3861_386189

-- Define the function f(x) = ax^2 - 4x + c
def f (a c x : ℝ) : ℝ := a * x^2 - 4 * x + c

-- State the theorem
theorem min_value_theorem (a c : ℝ) (h1 : a > 0) 
  (h2 : Set.range (f a c) = Set.Ici 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), (1 / (c - 1)) + (9 / a) ≥ m :=
sorry

end min_value_theorem_l3861_386189


namespace area_of_triangle_AEB_l3861_386151

-- Define the points
variable (A B C D E F G : Euclidean_plane)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Euclidean_plane) : Prop := sorry

-- Define the lengths
def length (P Q : Euclidean_plane) : ℝ := sorry

-- Define a point being on a line segment
def on_segment (P Q R : Euclidean_plane) : Prop := sorry

-- Define line intersection
def intersect (P Q R S : Euclidean_plane) : Euclidean_plane := sorry

-- Define triangle area
def triangle_area (P Q R : Euclidean_plane) : ℝ := sorry

theorem area_of_triangle_AEB 
  (h_rect : is_rectangle A B C D)
  (h_AB : length A B = 10)
  (h_BC : length B C = 4)
  (h_F_on_CD : on_segment C D F)
  (h_G_on_CD : on_segment C D G)
  (h_DF : length D F = 2)
  (h_GC : length G C = 3)
  (h_E : E = intersect A F B G) :
  triangle_area A E B = 40 := by sorry

end area_of_triangle_AEB_l3861_386151


namespace yellow_tiles_count_l3861_386126

theorem yellow_tiles_count (total : ℕ) (purple : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : purple = 6)
  (h3 : white = 7)
  : ∃ (yellow : ℕ), 
    yellow + (yellow + 1) + purple + white = total ∧ 
    yellow = 3 := by
  sorry

end yellow_tiles_count_l3861_386126


namespace tangent_and_normal_equations_l3861_386186

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 3)

/-- Tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 2*x - 7*y + 19 = 0

/-- Normal line equation -/
def normal_line (x y : ℝ) : Prop := 7*x + 2*y - 13 = 0

theorem tangent_and_normal_equations :
  curve point.1 point.2 →
  (∀ x y, tangent_line x y ↔ 
    (y - point.2) = (2/7) * (x - point.1)) ∧
  (∀ x y, normal_line x y ↔ 
    (y - point.2) = (-7/2) * (x - point.1)) := by
  sorry

end tangent_and_normal_equations_l3861_386186


namespace same_remainder_mod_ten_l3861_386197

theorem same_remainder_mod_ten (a b c : ℕ) 
  (h : ∃ r : ℕ, (2*a + b) % 10 = r ∧ (2*b + c) % 10 = r ∧ (2*c + a) % 10 = r) :
  ∃ s : ℕ, a % 10 = s ∧ b % 10 = s ∧ c % 10 = s := by
  sorry

end same_remainder_mod_ten_l3861_386197


namespace number_ratio_problem_l3861_386157

theorem number_ratio_problem (x y z : ℝ) : 
  x = 18 →  -- The smallest number is 18
  y = 4 * x →  -- The second number is 4 times the first
  ∃ k : ℝ, z = k * y →  -- The third number is some multiple of the second
  (x + y + z) / 3 = 78 →  -- Their average is 78
  z / y = 2 :=  -- The ratio of the third to the second is 2
by sorry

end number_ratio_problem_l3861_386157


namespace recycling_program_earnings_l3861_386179

/-- Represents the referral program structure and earnings --/
structure ReferralProgram where
  initial_signup_bonus : ℚ
  first_tier_referral_bonus : ℚ
  second_tier_referral_bonus : ℚ
  friend_signup_bonus : ℚ
  friend_referral_bonus : ℚ
  first_day_referrals : ℕ
  first_day_friends_referrals : ℕ
  week_end_friends_referrals : ℕ
  third_day_referrals : ℕ
  fourth_day_friends_referrals : ℕ

/-- Calculates the total earnings for Katrina and her friends --/
def total_earnings (program : ReferralProgram) : ℚ :=
  sorry

/-- The recycling program referral structure --/
def recycling_program : ReferralProgram := {
  initial_signup_bonus := 5,
  first_tier_referral_bonus := 8,
  second_tier_referral_bonus := 3/2,
  friend_signup_bonus := 5,
  friend_referral_bonus := 2,
  first_day_referrals := 5,
  first_day_friends_referrals := 3,
  week_end_friends_referrals := 2,
  third_day_referrals := 2,
  fourth_day_friends_referrals := 1
}

/-- Theorem stating that the total earnings for Katrina and her friends is $190.50 --/
theorem recycling_program_earnings :
  total_earnings recycling_program = 381/2 := by
  sorry

end recycling_program_earnings_l3861_386179


namespace kingsley_pants_per_day_l3861_386137

/-- Represents the number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- Represents the amount of fabric used for one shirt in yards -/
def fabric_per_shirt : ℕ := 2

/-- Represents the amount of fabric used for one pair of pants in yards -/
def fabric_per_pants : ℕ := 5

/-- Represents the total amount of fabric needed every 3 days in yards -/
def total_fabric_3days : ℕ := 93

/-- Theorem stating that Kingsley makes 5 pairs of pants per day given the conditions -/
theorem kingsley_pants_per_day :
  ∃ (p : ℕ), 
    p * fabric_per_pants + jenson_shirts_per_day * fabric_per_shirt = total_fabric_3days / 3 ∧
    p = 5 := by
  sorry

end kingsley_pants_per_day_l3861_386137


namespace negation_of_proposition_l3861_386147

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l3861_386147


namespace girls_left_auditorium_l3861_386176

theorem girls_left_auditorium (initial_boys : ℕ) (initial_girls : ℕ) (remaining_students : ℕ) : 
  initial_boys = 24 →
  initial_girls = 14 →
  remaining_students = 30 →
  ∃ (left_girls : ℕ), left_girls = 4 ∧ 
    ∃ (left_boys : ℕ), left_boys = left_girls ∧
    initial_boys + initial_girls - (left_boys + left_girls) = remaining_students :=
by sorry

end girls_left_auditorium_l3861_386176


namespace alcohol_concentration_is_40_percent_l3861_386175

-- Define the ratios of water to alcohol in solutions A and B
def waterToAlcoholRatioA : Rat := 4 / 1
def waterToAlcoholRatioB : Rat := 2 / 3

-- Define the amount of each solution mixed (assuming 1 unit each)
def amountA : Rat := 1
def amountB : Rat := 1

-- Define the function to calculate the alcohol concentration in the mixed solution
def alcoholConcentration (ratioA ratioB amountA amountB : Rat) : Rat :=
  let waterA := amountA * (ratioA / (ratioA + 1))
  let alcoholA := amountA * (1 / (ratioA + 1))
  let waterB := amountB * (ratioB / (ratioB + 1))
  let alcoholB := amountB * (1 / (ratioB + 1))
  let totalAlcohol := alcoholA + alcoholB
  let totalMixture := waterA + alcoholA + waterB + alcoholB
  totalAlcohol / totalMixture

-- Theorem statement
theorem alcohol_concentration_is_40_percent :
  alcoholConcentration waterToAlcoholRatioA waterToAlcoholRatioB amountA amountB = 2/5 := by
  sorry


end alcohol_concentration_is_40_percent_l3861_386175


namespace first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l3861_386163

-- Define the set of integer solutions
def IntegerSolutions : Set ℤ := {0, 1, 2}

-- Theorem for the first inequality
theorem first_inequality_solution (x : ℝ) :
  3 * (2 * x + 2) > 4 * x - 1 + 7 ↔ x > -3/2 := by sorry

-- Theorem for the system of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x + 1 > 0 ∧ x ≤ (x - 2) / 3 + 2) ↔ (-1 < x ∧ x ≤ 2) := by sorry

-- Theorem for integer solutions
theorem integer_solutions_correct :
  ∀ (n : ℤ), n ∈ IntegerSolutions ↔ (n + 1 > 0 ∧ n ≤ (n - 2) / 3 + 2) := by sorry

end first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l3861_386163


namespace profit_minimum_at_radius_one_l3861_386117

noncomputable def profit_function (r : ℝ) : ℝ :=
  0.2 * (4/3) * Real.pi * r^3 - 0.8 * Real.pi * r^2

theorem profit_minimum_at_radius_one :
  ∀ r : ℝ, 0 < r → r ≤ 6 →
  profit_function r ≥ profit_function 1 :=
sorry

end profit_minimum_at_radius_one_l3861_386117


namespace fourth_mile_relation_l3861_386154

/-- Represents the relationship between distance and time for a mile -/
structure MileData where
  n : ℕ      -- The mile number
  time : ℝ    -- Time taken to cover the mile (in hours)
  distance : ℝ -- Distance covered (in miles)

/-- The constant k in the inverse relationship -/
def k : ℝ := 2

/-- The inverse relationship between distance and time for a given mile -/
def inverse_relation (md : MileData) : Prop :=
  md.distance = k / md.time

/-- Theorem stating the relationship for the 2nd and 4th miles -/
theorem fourth_mile_relation 
  (mile2 : MileData) 
  (mile4 : MileData) 
  (h1 : mile2.n = 2) 
  (h2 : mile2.time = 2) 
  (h3 : mile2.distance = 1) 
  (h4 : mile4.n = 4) 
  (h5 : inverse_relation mile2) 
  (h6 : inverse_relation mile4) : 
  mile4.time = 4 ∧ mile4.distance = 0.5 := by
  sorry


end fourth_mile_relation_l3861_386154


namespace root_existence_condition_l3861_386199

theorem root_existence_condition (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, a * x + 3 = 0) ↔ (a ≤ -3/2 ∨ a ≥ 3) := by
  sorry

end root_existence_condition_l3861_386199


namespace slope_of_right_triangle_l3861_386113

/-- Given a right triangle ABC in the x-y plane where:
  * ∠B = 90°
  * AC = 225
  * AB = 180
  Prove that the slope of line segment AC is 4/3 -/
theorem slope_of_right_triangle (A B C : ℝ × ℝ) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 180^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 225^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 - (B.1 - A.1)^2 - (B.2 - A.2)^2 →
  (C.2 - A.2) / (C.1 - A.1) = 4 / 3 :=
by sorry

end slope_of_right_triangle_l3861_386113


namespace quadratic_coefficient_l3861_386161

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 3)
  (point : f.eval 0 = 7) : 
  f.a = 1 := by
  sorry

end quadratic_coefficient_l3861_386161


namespace rice_weight_l3861_386180

/-- Given rice divided equally into 4 containers, with 70 ounces in each container,
    and 1 pound equaling 16 ounces, the total amount of rice is 17.5 pounds. -/
theorem rice_weight (containers : Nat) (ounces_per_container : Nat) (ounces_per_pound : Nat)
    (h1 : containers = 4)
    (h2 : ounces_per_container = 70)
    (h3 : ounces_per_pound = 16) :
    (containers * ounces_per_container : Rat) / ounces_per_pound = 17.5 := by
  sorry

end rice_weight_l3861_386180


namespace germination_probability_l3861_386185

/-- The germination rate of the seeds -/
def germination_rate : ℝ := 0.7

/-- The number of seeds -/
def total_seeds : ℕ := 3

/-- The number of seeds we want to germinate -/
def target_germination : ℕ := 2

/-- The probability of exactly 2 out of 3 seeds germinating -/
def probability_2_out_of_3 : ℝ := 
  (Nat.choose total_seeds target_germination : ℝ) * 
  germination_rate ^ target_germination * 
  (1 - germination_rate) ^ (total_seeds - target_germination)

theorem germination_probability : 
  probability_2_out_of_3 = 0.441 := by sorry

end germination_probability_l3861_386185


namespace stating_distribution_schemes_count_l3861_386156

/-- Represents the number of schools --/
def num_schools : ℕ := 5

/-- Represents the number of computers --/
def num_computers : ℕ := 6

/-- Represents the number of schools that must receive at least 2 computers --/
def num_special_schools : ℕ := 2

/-- Represents the minimum number of computers each special school must receive --/
def min_computers_per_special_school : ℕ := 2

/-- 
Calculates the number of ways to distribute computers to schools 
under the given constraints
--/
def distribution_schemes : ℕ := sorry

/-- 
Theorem stating that the number of distribution schemes is 15
--/
theorem distribution_schemes_count : distribution_schemes = 15 := by sorry

end stating_distribution_schemes_count_l3861_386156


namespace largest_n_divisible_by_seven_l3861_386191

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 ∧ 
  (∃ k : ℤ, 6 * (n - 3)^5 - n^2 + 16*n - 36 = 7 * k) →
  n ≤ 99996 :=
by sorry

end largest_n_divisible_by_seven_l3861_386191


namespace exists_number_with_digit_sum_property_l3861_386139

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of digits of (n + 18) 
    is equal to the sum of digits of n minus 18 -/
theorem exists_number_with_digit_sum_property : 
  ∃ n : ℕ, sumOfDigits (n + 18) = sumOfDigits n - 18 := by sorry

end exists_number_with_digit_sum_property_l3861_386139


namespace parabola_coefficient_l3861_386195

/-- A parabola with equation x = ay² + by + c, vertex at (3, -1), and passing through (7, 3) has a = 1/4 -/
theorem parabola_coefficient (a b c : ℝ) : 
  (∀ y : ℝ, 3 = a * (-1)^2 + b * (-1) + c) →  -- vertex condition
  (∀ y : ℝ, 7 = a * 3^2 + b * 3 + c) →        -- point condition
  a = (1 : ℝ) / 4 := by sorry

end parabola_coefficient_l3861_386195


namespace banana_arrangements_l3861_386149

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters) / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60 :=
by sorry

end banana_arrangements_l3861_386149


namespace all_students_visiting_one_student_visiting_l3861_386168

-- Define the probabilities of each student visiting
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Theorem for the probability of all three students visiting
theorem all_students_visiting : 
  prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Theorem for the probability of exactly one student visiting
theorem one_student_visiting : 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C = 9/20 := by
  sorry

end all_students_visiting_one_student_visiting_l3861_386168


namespace julia_basketball_success_rate_increase_l3861_386148

theorem julia_basketball_success_rate_increase :
  let initial_success : ℕ := 3
  let initial_attempts : ℕ := 8
  let subsequent_success : ℕ := 12
  let subsequent_attempts : ℕ := 16
  let total_success := initial_success + subsequent_success
  let total_attempts := initial_attempts + subsequent_attempts
  let initial_rate := initial_success / initial_attempts
  let final_rate := total_success / total_attempts
  final_rate - initial_rate = 1/4 := by sorry

end julia_basketball_success_rate_increase_l3861_386148


namespace smallest_number_l3861_386132

theorem smallest_number (a b c d : ℝ) (ha : a = 1/2) (hb : b = Real.sqrt 3) (hc : c = 0) (hd : d = -2) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end smallest_number_l3861_386132


namespace tom_stamp_collection_tom_final_collection_l3861_386184

theorem tom_stamp_collection (tom_initial : ℕ) (mike_gift : ℕ) : ℕ :=
  let harry_gift := 2 * mike_gift + 10
  let sarah_gift := 3 * mike_gift - 5
  let total_gifts := mike_gift + harry_gift + sarah_gift
  tom_initial + total_gifts

theorem tom_final_collection :
  tom_stamp_collection 3000 17 = 3107 := by
  sorry

end tom_stamp_collection_tom_final_collection_l3861_386184


namespace three_digit_numbers_from_five_l3861_386190

/-- The number of ways to create a three-digit number using five different single-digit numbers -/
def three_digit_combinations (n : ℕ) (r : ℕ) : ℕ :=
  (n.factorial) / ((r.factorial) * ((n - r).factorial))

/-- The number of permutations of r items -/
def permutations (r : ℕ) : ℕ := r.factorial

theorem three_digit_numbers_from_five : 
  three_digit_combinations 5 3 * permutations 3 = 60 := by
  sorry

end three_digit_numbers_from_five_l3861_386190


namespace min_cuts_correct_l3861_386138

/-- The minimum number of cuts required to transform a square into 100 20-gons -/
def min_cuts : ℕ := 1699

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a 20-gon -/
def twenty_gon_sides : ℕ := 20

/-- The number of 20-gons we want to obtain -/
def target_polygons : ℕ := 100

/-- The maximum increase in the number of sides per cut -/
def max_side_increase : ℕ := 4

/-- The total number of sides in the final configuration -/
def total_final_sides : ℕ := target_polygons * twenty_gon_sides

theorem min_cuts_correct :
  min_cuts = (total_final_sides - square_sides) / max_side_increase + 
             (target_polygons - 1) :=
by sorry

end min_cuts_correct_l3861_386138


namespace triangle_ratio_l3861_386150

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + 2*b - 3*c) / (Real.sin A + 2*Real.sin B - 3*Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l3861_386150


namespace circle_line_intersection_max_k_l3861_386162

theorem circle_line_intersection_max_k : 
  ∃ (k_max : ℝ),
    k_max = 4/3 ∧
    ∀ (k : ℝ),
      (∃ (x₀ y₀ : ℝ),
        y₀ = k * x₀ - 2 ∧
        ∃ (x y : ℝ),
          (x - 4)^2 + y^2 = 1 ∧
          (x - x₀)^2 + (y - y₀)^2 ≤ 1) →
      k ≤ k_max :=
by sorry

end circle_line_intersection_max_k_l3861_386162


namespace marias_green_beans_l3861_386116

/-- Given Maria's vegetable cutting preferences and the number of potatoes,
    calculate the number of green beans she needs to cut. -/
theorem marias_green_beans (potatoes : ℕ) : potatoes = 2 → 8 = (potatoes * 6 * 2) / 3 := by
  sorry

end marias_green_beans_l3861_386116


namespace shifted_sine_function_proof_l3861_386188

open Real

theorem shifted_sine_function_proof (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) : 
  (∃ x₁ x₂ : ℝ, |sin (2*x₁) - sin (2*(x₂ - φ))| = 2 ∧ 
   (∀ y₁ y₂ : ℝ, |sin (2*y₁) - sin (2*(y₂ - φ))| = 2 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
   |x₁ - x₂| = π/3) →
  φ = π/6 := by
sorry

end shifted_sine_function_proof_l3861_386188


namespace cube_condition_l3861_386130

theorem cube_condition (n : ℤ) : 
  (∃ k : ℤ, 6 * n + 2 = k ^ 3) ↔ 
  (∃ m : ℤ, n = 36 * m ^ 3 + 36 * m ^ 2 + 12 * m + 1) := by
sorry

end cube_condition_l3861_386130


namespace Q_space_diagonals_l3861_386159

-- Define the structure of our polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

-- Define our specific polyhedron Q
def Q : Polyhedron := {
  vertices := 30,
  edges := 72,
  faces := 38,
  triangular_faces := 20,
  quadrilateral_faces := 18
}

-- Function to calculate the number of space diagonals
def space_diagonals (p : Polyhedron) : ℕ :=
  let total_pairs := p.vertices.choose 2
  let face_diagonals := 2 * p.quadrilateral_faces
  total_pairs - p.edges - face_diagonals

-- Theorem statement
theorem Q_space_diagonals : space_diagonals Q = 327 := by
  sorry


end Q_space_diagonals_l3861_386159


namespace total_baseball_fans_l3861_386153

theorem total_baseball_fans (yankees mets redsox : ℕ) : 
  yankees * 2 = mets * 3 →
  mets * 5 = redsox * 4 →
  mets = 88 →
  yankees + mets + redsox = 330 :=
by sorry

end total_baseball_fans_l3861_386153


namespace cube_root_equation_solution_l3861_386158

theorem cube_root_equation_solution (Q P : ℝ) 
  (h1 : (13 * Q + 6 * P + 1) ^ (1/3) - (13 * Q - 6 * P - 1) ^ (1/3) = 2 ^ (1/3))
  (h2 : Q > 0) : 
  Q = 7 := by
sorry

end cube_root_equation_solution_l3861_386158


namespace min_ratio_of_circles_l3861_386107

/-- The locus M of point A -/
def locus_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ y ≠ 0

/-- Point B -/
def B : ℝ × ℝ := (-1, 0)

/-- Point C -/
def C : ℝ × ℝ := (1, 0)

/-- The area of the inscribed circle of triangle PBC -/
noncomputable def S₁ (P : ℝ × ℝ) : ℝ := sorry

/-- The area of the circumscribed circle of triangle PBC -/
noncomputable def S₂ (P : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_ratio_of_circles :
  ∀ P : ℝ × ℝ, locus_M P.1 P.2 → S₂ P / S₁ P ≥ 4 ∧ ∃ Q : ℝ × ℝ, locus_M Q.1 Q.2 ∧ S₂ Q / S₁ Q = 4 := by
  sorry

end min_ratio_of_circles_l3861_386107


namespace population_decrease_rate_l3861_386115

theorem population_decrease_rate (initial_population : ℕ) (population_after_2_years : ℕ) 
  (h1 : initial_population = 30000)
  (h2 : population_after_2_years = 19200) :
  ∃ (r : ℝ), r = 0.2 ∧ (1 - r)^2 * initial_population = population_after_2_years :=
by sorry

end population_decrease_rate_l3861_386115
