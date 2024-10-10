import Mathlib

namespace ham_to_pepperoni_ratio_l21_2122

/-- Represents the number of pieces of each type of meat on a pizza -/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Represents the properties of the pizza -/
structure Pizza where
  toppings : PizzaToppings
  slices : ℕ
  meat_per_slice : ℕ

/-- The ratio of ham to pepperoni is 2:1 given the specified conditions -/
theorem ham_to_pepperoni_ratio (pizza : Pizza) : 
  pizza.toppings.pepperoni = 30 ∧ 
  pizza.toppings.sausage = pizza.toppings.pepperoni + 12 ∧
  pizza.slices = 6 ∧
  pizza.meat_per_slice = 22 →
  pizza.toppings.ham = 2 * pizza.toppings.pepperoni := by
  sorry

#check ham_to_pepperoni_ratio

end ham_to_pepperoni_ratio_l21_2122


namespace widescreen_tv_horizontal_length_l21_2111

theorem widescreen_tv_horizontal_length :
  ∀ (h w d : ℝ),
  h > 0 ∧ w > 0 ∧ d > 0 →
  w / h = 16 / 9 →
  h^2 + w^2 = d^2 →
  d = 40 →
  w = (640 * Real.sqrt 337) / 337 := by
sorry

end widescreen_tv_horizontal_length_l21_2111


namespace collinear_points_sum_l21_2139

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), p2 = (t1 • p1 + (1 - t1) • p3) ∧ p3 = (t2 • p1 + (1 - t2) • p2)

/-- If the points (1, a, b), (a, 2, b), and (a, b, 3) are collinear in 3-space, then a + b = 4. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, a, b) (a, 2, b) (a, b, 3) → a + b = 4 := by
  sorry

end collinear_points_sum_l21_2139


namespace max_crate_weight_l21_2151

/-- Proves that the maximum weight each crate can hold is 20 kg given the problem conditions --/
theorem max_crate_weight (num_crates : ℕ) (nail_bags : ℕ) (hammer_bags : ℕ) (plank_bags : ℕ)
  (nail_weight : ℝ) (hammer_weight : ℝ) (plank_weight : ℝ) (left_out_weight : ℝ) :
  num_crates = 15 →
  nail_bags = 4 →
  hammer_bags = 12 →
  plank_bags = 10 →
  nail_weight = 5 →
  hammer_weight = 5 →
  plank_weight = 30 →
  left_out_weight = 80 →
  (nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight - left_out_weight) / num_crates = 20 := by
  sorry

#check max_crate_weight

end max_crate_weight_l21_2151


namespace perpendicular_lines_intersection_l21_2144

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℚ) : Prop := a * x + b * y + c = 0

/-- Given two perpendicular lines and their intersection point, prove p - m - n = 4 -/
theorem perpendicular_lines_intersection (m n p : ℚ) : 
  perpendicular (-2/m) (3/2) →
  point_on_line 2 p 2 m (-1) →
  point_on_line 2 p 3 (-2) n →
  p - m - n = 4 := by sorry

end perpendicular_lines_intersection_l21_2144


namespace expression_equals_forty_l21_2172

theorem expression_equals_forty : (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := by
  sorry

end expression_equals_forty_l21_2172


namespace smallest_prime_factor_in_C_l21_2174

def C : Set Nat := {47, 49, 51, 53, 55}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → (Nat.minFac n ≤ Nat.minFac m)) ∧
  n = 51 := by
  sorry

end smallest_prime_factor_in_C_l21_2174


namespace trig_expression_simplification_l21_2196

/-- Simplification of a trigonometric expression -/
theorem trig_expression_simplification :
  let expr := (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
               Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
              Real.cos (10 * π / 180)
  ∃ (k : ℝ), expr = (2 * Real.cos (40 * π / 180)) / Real.cos (10 * π / 180) * k :=
by
  sorry


end trig_expression_simplification_l21_2196


namespace vector_addition_scalar_multiplication_l21_2160

def vec_a : ℝ × ℝ := (2, 3)
def vec_b : ℝ × ℝ := (-1, 5)

theorem vector_addition_scalar_multiplication :
  vec_a + 3 • vec_b = (-1, 18) := by sorry

end vector_addition_scalar_multiplication_l21_2160


namespace polynomial_division_quotient_l21_2106

theorem polynomial_division_quotient (z : ℝ) : 
  ((5/4 : ℝ) * z^4 - (23/16 : ℝ) * z^3 + (129/64 : ℝ) * z^2 - (353/256 : ℝ) * z + 949/1024) * (4 * z + 1) = 
  5 * z^5 - 3 * z^4 + 4 * z^3 - 7 * z^2 + 9 * z - 3 := by
  sorry

end polynomial_division_quotient_l21_2106


namespace locus_is_parabola_l21_2186

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus of points P
def locus : Set (ℝ × ℝ) := {P : ℝ × ℝ | ∃ B ∈ l, dist P M = dist P B}

-- Theorem statement
theorem locus_is_parabola : 
  ∃ a b c : ℝ, locus = {P : ℝ × ℝ | P.2 = a * P.1^2 + b * P.1 + c} := by
  sorry

end locus_is_parabola_l21_2186


namespace workshop_average_salary_l21_2128

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) : 
  total_workers = 12 → 
  num_technicians = 7 → 
  avg_salary_technicians = 12000 → 
  avg_salary_rest = 6000 → 
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 9500 := by
sorry

end workshop_average_salary_l21_2128


namespace marble_difference_l21_2163

theorem marble_difference (drew_initial marcus_initial : ℕ) : 
  drew_initial - marcus_initial = 24 ∧ 
  drew_initial - 12 = 25 ∧ 
  marcus_initial + 12 = 25 :=
by
  sorry

#check marble_difference

end marble_difference_l21_2163


namespace prob_queens_or_aces_l21_2175

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

def prob_two_queens : ℚ := (num_queens * (num_queens - 1)) / (standard_deck * (standard_deck - 1))
def prob_one_ace : ℚ := 2 * (num_aces * (standard_deck - num_aces)) / (standard_deck * (standard_deck - 1))
def prob_two_aces : ℚ := (num_aces * (num_aces - 1)) / (standard_deck * (standard_deck - 1))

theorem prob_queens_or_aces :
  prob_two_queens + prob_one_ace + prob_two_aces = 2 / 13 :=
sorry

end prob_queens_or_aces_l21_2175


namespace square_circle_union_area_l21_2149

/-- The area of the union of a square and a circle, where the square has side length 8 and the circle has radius 12 and is centered at one of the square's vertices. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 108 * π :=
by sorry

end square_circle_union_area_l21_2149


namespace website_development_time_ratio_l21_2101

/-- The time Katherine takes to develop a website -/
def katherine_time : ℕ := 20

/-- The number of websites Naomi developed -/
def naomi_websites : ℕ := 30

/-- The total time Naomi took to develop all websites -/
def naomi_total_time : ℕ := 750

/-- The ratio of the time Naomi takes to the time Katherine takes to develop a website -/
def time_ratio : ℚ := (naomi_total_time / naomi_websites : ℚ) / katherine_time

theorem website_development_time_ratio :
  time_ratio = 5/4 := by sorry

end website_development_time_ratio_l21_2101


namespace no_valid_m_exists_l21_2155

theorem no_valid_m_exists : ¬ ∃ (m : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 1 2 →
    x₁ + m > x₂^2 - m*x₂ + m^2/2 + 2*m - 3) ∧
  (Set.Ioo 1 2 = {x | x^2 - m*x + m^2/2 + 2*m - 3 < m^2/2 + 1}) :=
by sorry

end no_valid_m_exists_l21_2155


namespace arithmetic_progression_first_term_l21_2127

theorem arithmetic_progression_first_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n + 5) →  -- Common difference is 5
    a 21 = 103 →                 -- 21st term is 103
    a 1 = 3 :=                   -- First term is 3
by
  sorry

end arithmetic_progression_first_term_l21_2127


namespace fraction_of_120_l21_2180

theorem fraction_of_120 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 120 = 5 / 3 := by
  sorry

end fraction_of_120_l21_2180


namespace inscribed_circle_area_l21_2102

/-- Given an equilateral triangle with a point inside at distances 1, 2, and 4 inches from its sides,
    the area of the inscribed circle is 49π/9 square inches. -/
theorem inscribed_circle_area (s : ℝ) (h : s > 0) : 
  let triangle_area := (7 * s) / 2
  let inscribed_circle_radius := triangle_area / ((3 * s) / 2)
  (π * inscribed_circle_radius ^ 2) = 49 * π / 9 := by
  sorry

end inscribed_circle_area_l21_2102


namespace pure_imaginary_condition_l21_2119

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * ((a - Complex.I) * (1 + Complex.I))).re = 0 → a = -1 := by
  sorry

end pure_imaginary_condition_l21_2119


namespace probability_of_same_number_l21_2129

/-- The upper bound for the selected numbers -/
def upper_bound : ℕ := 500

/-- Billy's number is a multiple of this value -/
def billy_multiple : ℕ := 20

/-- Bobbi's number is a multiple of this value -/
def bobbi_multiple : ℕ := 30

/-- The probability of Billy and Bobbi selecting the same number -/
def same_number_probability : ℚ := 1 / 50

/-- Theorem stating the probability of Billy and Bobbi selecting the same number -/
theorem probability_of_same_number :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upper_bound ∧ b₂ < upper_bound ∧
   b₁ % billy_multiple = 0 ∧ b₂ % bobbi_multiple = 0) →
  same_number_probability = 1 / 50 :=
by sorry

end probability_of_same_number_l21_2129


namespace bowling_ball_weight_l21_2182

/-- Given that four identical canoes weigh the same as nine identical bowling balls,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℝ) (ball_weight : ℝ) : 
  canoe_weight = 36 →  -- One canoe weighs 36 pounds
  4 * canoe_weight = 9 * ball_weight →  -- Four canoes weigh the same as nine bowling balls
  ball_weight = 16 :=  -- One bowling ball weighs 16 pounds
by
  sorry

#check bowling_ball_weight

end bowling_ball_weight_l21_2182


namespace caroline_lassis_l21_2110

/-- Given that Caroline can make 11 lassis with 2 mangoes, 
    prove that she can make 55 lassis with 10 mangoes. -/
theorem caroline_lassis (lassis_per_two_mangoes : ℕ) (mangoes : ℕ) :
  lassis_per_two_mangoes = 11 ∧ mangoes = 10 →
  (lassis_per_two_mangoes : ℚ) / 2 * mangoes = 55 := by
  sorry

end caroline_lassis_l21_2110


namespace number_2008_row_l21_2158

theorem number_2008_row : ∃ (n : ℕ), n = 45 ∧ 
  (n - 1)^2 < 2008 ∧ 2008 ≤ n^2 ∧ 
  (∀ (k : ℕ), k < n → k^2 < 2008) :=
by sorry

end number_2008_row_l21_2158


namespace sqrt_fraction_equality_specific_sqrt_equality_l21_2164

theorem sqrt_fraction_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n+1) : ℝ) := by sorry

theorem specific_sqrt_equality :
  Real.sqrt (101/100 + 1/121) = 1 + 1/110 := by sorry

end sqrt_fraction_equality_specific_sqrt_equality_l21_2164


namespace cost_of_seven_sandwiches_five_sodas_l21_2195

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.1

def total_cost (num_sandwiches num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if total_items > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem cost_of_seven_sandwiches_five_sodas :
  total_cost 7 5 = 38.7 := by
  sorry

end cost_of_seven_sandwiches_five_sodas_l21_2195


namespace price_increase_l21_2109

theorem price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  increase_percentage > 0 →
  (1 + increase_percentage) * (1 + increase_percentage) = 1 + 0.44 →
  increase_percentage = 0.2 := by
sorry

end price_increase_l21_2109


namespace harry_travel_time_ratio_l21_2185

/-- Given Harry's travel times, prove the ratio of walking time to bus journey time -/
theorem harry_travel_time_ratio :
  let total_time : ℕ := 60
  let bus_time_elapsed : ℕ := 15
  let bus_time_remaining : ℕ := 25
  let bus_time_total : ℕ := bus_time_elapsed + bus_time_remaining
  let walking_time : ℕ := total_time - bus_time_total
  walking_time / bus_time_total = 1 / 2 := by sorry

end harry_travel_time_ratio_l21_2185


namespace sugar_in_recipe_l21_2166

/-- Given a cake recipe and Mary's baking progress, calculate the amount of sugar required. -/
theorem sugar_in_recipe (total_flour sugar remaining_flour : ℕ) : 
  total_flour = 10 →
  remaining_flour = total_flour - 7 →
  remaining_flour = sugar + 1 →
  sugar = 2 := by sorry

end sugar_in_recipe_l21_2166


namespace factorization_1_factorization_2_l21_2178

-- First expression
theorem factorization_1 (x y : ℝ) : -x^2 + 12*x*y - 36*y^2 = -(x - 6*y)^2 := by
  sorry

-- Second expression
theorem factorization_2 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) := by
  sorry

end factorization_1_factorization_2_l21_2178


namespace min_value_3x_plus_y_l21_2162

theorem min_value_3x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 2 / (x + 4) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ -8 + 20 * Real.sqrt 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    2 / (x₀ + 4) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = -8 + 20 * Real.sqrt 2 :=
by sorry

end min_value_3x_plus_y_l21_2162


namespace expression_value_l21_2181

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) : 15*y - 5*x + 6 = -14 := by
  sorry

end expression_value_l21_2181


namespace infinitely_many_n_exist_l21_2131

-- Define the s operation on sets of integers
def s (F : Set ℤ) : Set ℤ :=
  {a : ℤ | (a ∈ F ∧ a - 1 ∉ F) ∨ (a ∉ F ∧ a - 1 ∈ F)}

-- Define the n-fold application of s
def s_power (F : Set ℤ) : ℕ → Set ℤ
  | 0 => F
  | n + 1 => s (s_power F n)

theorem infinitely_many_n_exist (F : Set ℤ) (h_finite : Set.Finite F) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, s_power F n = F ∪ {a + n | a ∈ F} := by
  sorry

end infinitely_many_n_exist_l21_2131


namespace twentieth_term_of_sequence_l21_2132

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence with first term -6 and common difference 5 -/
theorem twentieth_term_of_sequence :
  arithmeticSequence (-6) 5 20 = 89 := by
  sorry

end twentieth_term_of_sequence_l21_2132


namespace smallest_number_divisible_by_28_remainder_4_mod_15_l21_2133

theorem smallest_number_divisible_by_28_remainder_4_mod_15 :
  ∃ n : ℕ, (n % 28 = 0) ∧ (n % 15 = 4) ∧ 
  (∀ m : ℕ, m < n → (m % 28 ≠ 0 ∨ m % 15 ≠ 4)) ∧ 
  n = 364 := by
sorry

end smallest_number_divisible_by_28_remainder_4_mod_15_l21_2133


namespace monotone_decreasing_function_positivity_l21_2112

theorem monotone_decreasing_function_positivity 
  (f : ℝ → ℝ) 
  (h_monotone : ∀ x y, x < y → f x > f y) 
  (h_inequality : ∀ x, f x / (deriv f x) + x < 1) : 
  ∀ x, f x > 0 := by
sorry

end monotone_decreasing_function_positivity_l21_2112


namespace initial_men_count_l21_2140

/-- Given provisions that last 15 days for an initial group of men and 12.5 days when 200 more men join,
    prove that the initial number of men is 1000. -/
theorem initial_men_count (M : ℕ) (P : ℝ) : 
  (P / (15 * M) = P / (12.5 * (M + 200))) → M = 1000 := by
  sorry

end initial_men_count_l21_2140


namespace min_gennadies_for_festival_l21_2141

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  counts.borises - 1 - (counts.alexanders + counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given counts -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end min_gennadies_for_festival_l21_2141


namespace family_ages_solution_l21_2126

def family_ages (w h s d : ℕ) : Prop :=
  -- Woman's age reversed equals husband's age
  w = 10 * (h % 10) + (h / 10) ∧
  -- Husband is older than woman
  h > w ∧
  -- Difference between ages is one-eleventh of their sum
  h - w = (h + w) / 11 ∧
  -- Son's age is difference between parents' ages
  s = h - w ∧
  -- Daughter's age is average of all ages
  d = (w + h + s) / 3 ∧
  -- Sum of digits of each age is the same
  (w % 10 + w / 10) = (h % 10 + h / 10) ∧
  (w % 10 + w / 10) = s ∧
  (w % 10 + w / 10) = (d % 10 + d / 10)

theorem family_ages_solution :
  ∃ (w h s d : ℕ), family_ages w h s d ∧ w = 45 ∧ h = 54 ∧ s = 9 ∧ d = 36 :=
by sorry

end family_ages_solution_l21_2126


namespace no_solution_exists_l21_2107

theorem no_solution_exists : ¬∃ (x a z b : ℕ), 
  (0 < x) ∧ (x < 10) ∧ 
  (0 < a) ∧ (a < 10) ∧ 
  (0 < z) ∧ (z < 10) ∧ 
  (0 < b) ∧ (b < 10) ∧ 
  (4 * x = a) ∧ 
  (4 * z = b) ∧ 
  (x^2 + a^2 = z^2 + b^2) ∧ 
  ((x + a)^3 > (z + b)^3) :=
sorry

end no_solution_exists_l21_2107


namespace value_of_a_minus_b_l21_2191

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 2020 * a + 2024 * b = 2040)
  (eq2 : 2022 * a + 2026 * b = 2044) : 
  a - b = 1002 := by
sorry

end value_of_a_minus_b_l21_2191


namespace leap_day_2024_is_thursday_l21_2118

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to calculate the number of days between two dates -/
def daysBetween (date1 date2 : Date) : ℕ :=
  sorry

/-- Function to determine the day of the week given a starting day and number of days passed -/
def getDayOfWeek (startDay : DayOfWeek) (daysPassed : ℕ) : DayOfWeek :=
  sorry

theorem leap_day_2024_is_thursday :
  let leap_day_1996 : Date := ⟨1996, 2, 29⟩
  let leap_day_2024 : Date := ⟨2024, 2, 29⟩
  let days_between := daysBetween leap_day_1996 leap_day_2024
  getDayOfWeek DayOfWeek.Thursday days_between = DayOfWeek.Thursday :=
sorry

end leap_day_2024_is_thursday_l21_2118


namespace right_triangle_legs_l21_2152

theorem right_triangle_legs (a Δ : ℝ) (ha : a > 0) (hΔ : Δ > 0) :
  ∃ x y : ℝ,
    x > 0 ∧ y > 0 ∧
    x^2 + y^2 = a^2 ∧
    x * y / 2 = Δ ∧
    x = (Real.sqrt (a^2 + 4*Δ) + Real.sqrt (a^2 - 4*Δ)) / 2 ∧
    y = (Real.sqrt (a^2 + 4*Δ) - Real.sqrt (a^2 - 4*Δ)) / 2 := by
  sorry

end right_triangle_legs_l21_2152


namespace tan_fifteen_ratio_equals_sqrt_three_l21_2167

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_ratio_equals_sqrt_three_l21_2167


namespace a_positive_if_f_decreasing_l21_2123

/-- A function that represents a(x³ - x) --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

/-- The theorem stating that if f is decreasing on (-√3/3, √3/3), then a > 0 --/
theorem a_positive_if_f_decreasing (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -Real.sqrt 3 / 3 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 3 / 3 → f a x₁ > f a x₂) →
  a > 0 := by
  sorry


end a_positive_if_f_decreasing_l21_2123


namespace product_remainder_mod_five_l21_2194

theorem product_remainder_mod_five : (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end product_remainder_mod_five_l21_2194


namespace train_length_l21_2146

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 65 →
  man_speed = 7 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.5 := by
  sorry


end train_length_l21_2146


namespace sqrt_meaningful_iff_x_geq_5_l21_2116

theorem sqrt_meaningful_iff_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) ↔ x ≥ 5 := by
sorry

end sqrt_meaningful_iff_x_geq_5_l21_2116


namespace intersection_P_complement_Q_equals_one_two_l21_2153

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Finset Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_P_complement_Q_equals_one_two :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_P_complement_Q_equals_one_two_l21_2153


namespace rod_system_equilibrium_l21_2168

/-- Represents the equilibrium state of a rod system -/
structure RodSystem where
  l : Real          -- Length of the rod in meters
  m₂ : Real         -- Mass of the rod in kg
  s : Real          -- Distance of left thread attachment from right end in meters
  m₁ : Real         -- Mass of the load in kg

/-- Checks if the rod system is in equilibrium -/
def is_equilibrium (sys : RodSystem) : Prop :=
  sys.m₁ * sys.s = sys.m₂ * (sys.l / 2)

/-- Theorem stating the equilibrium condition for the given rod system -/
theorem rod_system_equilibrium :
  ∀ (sys : RodSystem),
    sys.l = 0.5 ∧ 
    sys.m₂ = 2 ∧ 
    sys.s = 0.1 ∧ 
    sys.m₁ = 5 →
    is_equilibrium sys := by
  sorry

end rod_system_equilibrium_l21_2168


namespace max_perimeter_constrained_quadrilateral_l21_2173

/-- A convex quadrilateral with specific side and diagonal constraints -/
structure ConstrainedQuadrilateral where
  -- Two sides are equal to 1
  side1 : ℝ
  side2 : ℝ
  side1_eq_one : side1 = 1
  side2_eq_one : side2 = 1
  -- Other sides and diagonals are not greater than 1
  side3 : ℝ
  side4 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  side3_le_one : side3 ≤ 1
  side4_le_one : side4 ≤ 1
  diagonal1_le_one : diagonal1 ≤ 1
  diagonal2_le_one : diagonal2 ≤ 1
  -- Convexity condition (simplified for this problem)
  is_convex : diagonal1 + diagonal2 > side1 + side3

/-- The maximum perimeter of a constrained quadrilateral -/
theorem max_perimeter_constrained_quadrilateral (q : ConstrainedQuadrilateral) :
  q.side1 + q.side2 + q.side3 + q.side4 ≤ 2 + 4 * Real.sin (15 * π / 180) := by
  sorry

end max_perimeter_constrained_quadrilateral_l21_2173


namespace complement_of_angle_A_l21_2147

-- Define the angle A
def angle_A : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 54 := by
  sorry

end complement_of_angle_A_l21_2147


namespace highest_page_number_l21_2197

/-- Represents the count of available digits --/
def DigitCount := Fin 10 → ℕ

/-- The set of digits where all digits except 5 are unlimited --/
def unlimitedExceptFive (d : DigitCount) : Prop :=
  ∀ i : Fin 10, i.val ≠ 5 → d i = 0 ∧ d 5 = 18

/-- Counts the occurrences of a digit in a natural number --/
def countDigit (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in all numbers up to n --/
def totalDigitCount (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem highest_page_number (d : DigitCount) (h : unlimitedExceptFive d) :
  ∀ n : ℕ, n > 99 → totalDigitCount 5 n > 18 :=
sorry

end highest_page_number_l21_2197


namespace odd_function_properties_and_inequality_l21_2143

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem odd_function_properties_and_inequality (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →
  (a = 1 ∧ ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ t : ℝ, (∀ x : ℝ, f a (x - t) + f a (x^2 - t^2) ≥ 0) → t = -1/2) := by
  sorry

end odd_function_properties_and_inequality_l21_2143


namespace max_sum_of_goods_l21_2156

theorem max_sum_of_goods (a b : ℕ) : 
  a > 0 → b > 0 → 5 * a + 19 * b = 213 → (∀ x y : ℕ, x > 0 → y > 0 → 5 * x + 19 * y = 213 → a + b ≥ x + y) → a + b = 37 := by
  sorry

end max_sum_of_goods_l21_2156


namespace passengers_off_north_carolina_l21_2171

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  offTexas : ℕ
  onTexas : ℕ
  onNorthCarolina : ℕ
  crew : ℕ
  landedVirginia : ℕ

/-- Calculates the number of passengers who got off in North Carolina --/
def passengersOffNorthCarolina (fp : FlightPassengers) : ℕ :=
  fp.initial - fp.offTexas + fp.onTexas - (fp.landedVirginia - fp.crew - fp.onNorthCarolina)

/-- Theorem stating that 47 passengers got off in North Carolina --/
theorem passengers_off_north_carolina :
  let fp : FlightPassengers := {
    initial := 124,
    offTexas := 58,
    onTexas := 24,
    onNorthCarolina := 14,
    crew := 10,
    landedVirginia := 67
  }
  passengersOffNorthCarolina fp = 47 := by
  sorry


end passengers_off_north_carolina_l21_2171


namespace steven_peach_count_l21_2145

/-- 
Given that:
- Jake has 84 more apples than Steven
- Jake has 10 fewer peaches than Steven
- Steven has 52 apples
- Jake has 3 peaches

Prove that Steven has 13 peaches.
-/
theorem steven_peach_count (jake_apple_diff : ℕ) (jake_peach_diff : ℕ) 
  (steven_apples : ℕ) (jake_peaches : ℕ) 
  (h1 : jake_apple_diff = 84)
  (h2 : jake_peach_diff = 10)
  (h3 : steven_apples = 52)
  (h4 : jake_peaches = 3) : 
  jake_peaches + jake_peach_diff = 13 := by
  sorry

end steven_peach_count_l21_2145


namespace tony_books_count_l21_2177

/-- The number of books Tony read -/
def tony_books : ℕ := 23

/-- The number of books Dean read -/
def dean_books : ℕ := 12

/-- The number of books Breanna read -/
def breanna_books : ℕ := 17

/-- The number of books Tony and Dean both read -/
def tony_dean_overlap : ℕ := 3

/-- The number of books all three read -/
def all_overlap : ℕ := 1

/-- The total number of different books read by all three -/
def total_different_books : ℕ := 47

theorem tony_books_count :
  tony_books + dean_books - tony_dean_overlap + breanna_books - all_overlap = total_different_books :=
by sorry

end tony_books_count_l21_2177


namespace triangular_weight_is_60_l21_2183

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- Theorem stating that the weight of a triangular weight is 60 grams -/
theorem triangular_weight_is_60 :
  (1 * round_weight + 1 * triangular_weight = 3 * round_weight) ∧
  (4 * round_weight + 1 * triangular_weight = 1 * triangular_weight + 1 * round_weight + rectangular_weight) →
  triangular_weight = 60 := by
  sorry

end triangular_weight_is_60_l21_2183


namespace inequality_solution_l21_2190

theorem inequality_solution (x : ℝ) : 
  (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((-3 - Real.sqrt 41) / -8) ∪ 
   Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪
   Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪
   Set.Ioi 0) ∧
  (x ≠ 0) ∧ (x ≠ ((-3 - Real.sqrt 41) / -8)) ∧ (x ≠ ((-3 + Real.sqrt 41) / -8)) :=
by sorry

end inequality_solution_l21_2190


namespace correct_units_l21_2193

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define the containers
structure Container where
  name : String
  volume : ℕ
  unit : VolumeUnit

-- Define the given containers
def orangeJuiceCup : Container :=
  { name := "Cup of orange juice", volume := 500, unit := VolumeUnit.Milliliter }

def waterBottle : Container :=
  { name := "Water bottle", volume := 3, unit := VolumeUnit.Liter }

-- Theorem to prove
theorem correct_units :
  (orangeJuiceCup.unit = VolumeUnit.Milliliter) ∧
  (waterBottle.unit = VolumeUnit.Liter) :=
by sorry

end correct_units_l21_2193


namespace even_function_range_theorem_l21_2187

-- Define an even function f on ℝ
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_range_theorem (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_even : isEvenFunction f) 
  (h_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  {x : ℝ | x^2 * f x - 4 * f 2 < x^2 - 4} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

end even_function_range_theorem_l21_2187


namespace complement_A_intersect_B_l21_2150

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

-- Define the universal set U
def U : Set ℕ := A ∪ B

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4, 5} := by sorry

end complement_A_intersect_B_l21_2150


namespace right_triangle_sin_c_l21_2130

theorem right_triangle_sin_c (A B C : ℝ) (h_right : A = 90) (h_sin_b : Real.sin B = 3/5) :
  Real.sin C = 4/5 := by
  sorry

end right_triangle_sin_c_l21_2130


namespace gwen_race_results_l21_2136

/-- Represents the race details --/
structure RaceDetails where
  jogging_time : ℕ
  jogging_elevation : ℕ
  jogging_ratio : ℕ
  walking_ratio : ℕ

/-- Calculates the walking time based on race details --/
def walking_time (race : RaceDetails) : ℕ :=
  (race.jogging_time / race.jogging_ratio) * race.walking_ratio

/-- Calculates the total elevation gain based on race details --/
def total_elevation_gain (race : RaceDetails) : ℕ :=
  (race.jogging_elevation * (race.jogging_time + walking_time race)) / race.jogging_time

/-- Theorem stating the walking time and total elevation gain for Gwen's race --/
theorem gwen_race_results (race : RaceDetails) 
  (h1 : race.jogging_time = 15)
  (h2 : race.jogging_elevation = 500)
  (h3 : race.jogging_ratio = 5)
  (h4 : race.walking_ratio = 3) :
  walking_time race = 9 ∧ total_elevation_gain race = 800 := by
  sorry


end gwen_race_results_l21_2136


namespace package_contains_100_masks_l21_2188

/-- The number of masks in a package used by a family -/
def number_of_masks (family_members : ℕ) (days_per_mask : ℕ) (total_days : ℕ) : ℕ :=
  family_members * (total_days / days_per_mask)

/-- Theorem: The package contains 100 masks -/
theorem package_contains_100_masks :
  number_of_masks 5 4 80 = 100 := by
  sorry

end package_contains_100_masks_l21_2188


namespace hamburger_cost_l21_2170

/-- Proves that the cost of each hamburger is $4 given the initial amount,
    the number of items purchased, the cost of milkshakes, and the remaining amount. -/
theorem hamburger_cost (initial_amount : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ)
                        (milkshake_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  milkshake_cost = 3 →
  remaining_amount = 70 →
  ∃ (hamburger_cost : ℕ),
    initial_amount = num_hamburgers * hamburger_cost + num_milkshakes * milkshake_cost + remaining_amount ∧
    hamburger_cost = 4 :=
by sorry

end hamburger_cost_l21_2170


namespace solve_y_l21_2124

theorem solve_y (x y : ℝ) (h1 : x^2 = y - 3) (h2 : x = 7) : 
  y = 52 ∧ y ≥ 10 := by
sorry

end solve_y_l21_2124


namespace larger_number_with_given_hcf_and_lcm_factors_l21_2169

theorem larger_number_with_given_hcf_and_lcm_factors : 
  ∀ (a b : ℕ+), 
    (Nat.gcd a b = 47) → 
    (∃ (k : ℕ+), Nat.lcm a b = k * 47 * 7^2 * 11 * 13 * 17^3) →
    (a ≥ b) →
    a = 123800939 := by
  sorry

end larger_number_with_given_hcf_and_lcm_factors_l21_2169


namespace inequality_statements_l21_2189

theorem inequality_statements :
  (∃ (a b : ℝ) (c : ℝ), c < 0 ∧ a < b ∧ c * a > c * b) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (2 * a * b) / (a + b) < Real.sqrt (a * b)) ∧
  (∀ (k : ℝ), k > 0 → ∀ (a b : ℝ), a > 0 → b > 0 → a * b = k → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x * y = k → a + b ≤ x + y)) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a^2 + b^2) / 2 < (a + b)^2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a + b)^2 ≥ a^2 + b^2) := by
  sorry

end inequality_statements_l21_2189


namespace tricycle_count_l21_2115

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) :
  total_children = 10 →
  total_wheels = 26 →
  ∃ (walking bicycles tricycles : ℕ),
    walking + bicycles + tricycles = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 6 :=
by sorry

end tricycle_count_l21_2115


namespace min_ties_for_twelve_pairs_l21_2104

/-- Represents the minimum number of ties needed to guarantee a certain number of pairs -/
def min_ties_for_pairs (num_pairs : ℕ) : ℕ :=
  5 + 2 * (num_pairs - 1)

/-- Theorem stating the minimum number of ties needed for 12 pairs -/
theorem min_ties_for_twelve_pairs :
  min_ties_for_pairs 12 = 27 :=
by sorry

end min_ties_for_twelve_pairs_l21_2104


namespace magnitude_of_p_l21_2134

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem magnitude_of_p (a b p : ℝ × ℝ) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hab : a.1 * b.1 + a.2 * b.2 = -1/2) 
  (hpa : p.1 * a.1 + p.2 * a.2 = 1/2) 
  (hpb : p.1 * b.1 + p.2 * b.2 = 1/2) : 
  p.1^2 + p.2^2 = 1 := by
  sorry

end magnitude_of_p_l21_2134


namespace sin_cos_sum_equals_one_l21_2199

theorem sin_cos_sum_equals_one : 
  Real.sin (65 * π / 180) * Real.sin (115 * π / 180) + 
  Real.cos (65 * π / 180) * Real.sin (25 * π / 180) = 1 := by
  sorry

end sin_cos_sum_equals_one_l21_2199


namespace rectangle_area_l21_2103

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l21_2103


namespace solution_set_min_value_l21_2108

-- Define the function f
def f (x : ℝ) := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ -7 ∨ x ≥ 5/3 :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9/2 :=
sorry

end solution_set_min_value_l21_2108


namespace candy_price_increase_l21_2165

theorem candy_price_increase (W : ℝ) (P : ℝ) (h1 : W > 0) (h2 : P > 0) :
  let new_weight := 0.6 * W
  let old_price_per_unit := P / W
  let new_price_per_unit := P / new_weight
  (new_price_per_unit - old_price_per_unit) / old_price_per_unit * 100 = (5/3 - 1) * 100 :=
by sorry

end candy_price_increase_l21_2165


namespace ellipse_sum_bound_l21_2176

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, is_on_ellipse x y → -13 ≤ x + y ∧ x + y ≤ 13 := by
  sorry

end ellipse_sum_bound_l21_2176


namespace modulo_evaluation_l21_2159

theorem modulo_evaluation : (203 * 19 - 22 * 8 + 6) % 17 = 12 := by
  sorry

end modulo_evaluation_l21_2159


namespace base9_arithmetic_l21_2137

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Addition in base 9 --/
def add_base9 (a b : Base9) : Base9 := sorry

/-- Subtraction in base 9 --/
def sub_base9 (a b : Base9) : Base9 := sorry

/-- Conversion from decimal to base 9 --/
def to_base9 (n : ℕ) : Base9 := sorry

theorem base9_arithmetic :
  sub_base9 (add_base9 (to_base9 374) (to_base9 625)) (to_base9 261) = to_base9 738 := by sorry

end base9_arithmetic_l21_2137


namespace f_2x_eq_3_l21_2198

/-- A function that is constant 3 for all real inputs -/
def f : ℝ → ℝ := fun x ↦ 3

/-- Theorem: f(2x) = 3 given that f(x) = 3 for all real x -/
theorem f_2x_eq_3 : ∀ x : ℝ, f (2 * x) = 3 := by
  sorry

end f_2x_eq_3_l21_2198


namespace hit_at_least_once_complement_of_miss_all_l21_2192

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ∃ i, ω i = true

-- Define the event of not hitting the target at all
def miss_all (ω : Ω) : Prop :=
  ∀ i, ω i = false

-- Theorem statement
theorem hit_at_least_once_complement_of_miss_all :
  ∀ ω : Ω, hit_at_least_once ω ↔ ¬(miss_all ω) :=
sorry

end hit_at_least_once_complement_of_miss_all_l21_2192


namespace expression_simplification_l21_2138

theorem expression_simplification (x y : ℚ) 
  (hx : x = -3/8) (hy : y = 4) : 
  (x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 2*x*(x - y) = 3 := by
  sorry

end expression_simplification_l21_2138


namespace three_inequalities_l21_2114

theorem three_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x + y) * (y + z) * (z + x) ≥ 8 * x * y * z) ∧
  (x^2 + y^2 + z^2 ≥ x*y + y*z + z*x) ∧
  (x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3)) := by
  sorry

end three_inequalities_l21_2114


namespace max_value_theorem_l21_2161

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 4 + 9 * p.y^2 / 4 = 1

def condition (p q : Point) : Prop :=
  p.x * q.x + 9 * p.y * q.y = -2

def expression (p q : Point) : ℝ :=
  |2 * p.x + 3 * p.y - 3| + |2 * q.x + 3 * q.y - 3|

theorem max_value_theorem (p q : Point) 
  (h1 : p ≠ q) 
  (h2 : ellipse p) 
  (h3 : ellipse q) 
  (h4 : condition p q) : 
  ∃ (max : ℝ), max = 6 + 2 * Real.sqrt 5 ∧ 
    ∀ (p' q' : Point), 
      p' ≠ q' → 
      ellipse p' → 
      ellipse q' → 
      condition p' q' → 
      expression p' q' ≤ max :=
sorry

end max_value_theorem_l21_2161


namespace field_ratio_proof_l21_2142

theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → 
  width = 13.5 → 
  (2 * width) / length = 9 / 8 := by
sorry

end field_ratio_proof_l21_2142


namespace circle_properties_l21_2105

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 5*x - 6*y + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x + 6*y - 6 = 0

-- Define points A and B
def pointA : ℝ × ℝ := (1, 0)
def pointB : ℝ × ℝ := (0, 1)

-- Define the chord length on x-axis
def chordLength : ℝ := 6

-- Theorem statement
theorem circle_properties :
  (∀ (x y : ℝ), circle1 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∀ (x y : ℝ), circle2 x y → ((x = pointA.1 ∧ y = pointA.2) ∨ (x = pointB.1 ∧ y = pointB.2))) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle1 x1 0 ∧ circle1 x2 0 ∧ x2 - x1 = chordLength) ∧
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ circle2 x1 0 ∧ circle2 x2 0 ∧ x2 - x1 = chordLength) :=
sorry

end circle_properties_l21_2105


namespace num_divisors_30_is_8_l21_2100

/-- The number of positive divisors of 30 -/
def num_divisors_30 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 30 is 8 -/
theorem num_divisors_30_is_8 : num_divisors_30 = 8 := by sorry

end num_divisors_30_is_8_l21_2100


namespace function_sum_negative_l21_2120

theorem function_sum_negative
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (x + 2) = -f (-x + 2))
  (h_increasing : ∀ x y, x > 2 → y > 2 → x < y → f x < f y)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_product : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ < 0 := by
sorry

end function_sum_negative_l21_2120


namespace product_of_polynomials_l21_2157

theorem product_of_polynomials (p q : ℚ) : 
  (∀ d : ℚ, (8 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 9) = 32 * d^4 - 68 * d^3 + 5 * d^2 + 23 * d - 36) →
  p + q = 3/4 := by
sorry

end product_of_polynomials_l21_2157


namespace age_difference_l21_2184

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 5 →
  sachin_age * 12 = rahul_age * 5 →
  rahul_age - sachin_age = 7 := by
sorry

end age_difference_l21_2184


namespace circle_reflection_y_axis_l21_2148

/-- Given a circle with equation (x+2)^2 + y^2 = 5, 
    its reflection about the y-axis has the equation (x-2)^2 + y^2 = 5 -/
theorem circle_reflection_y_axis (x y : ℝ) :
  ((x + 2)^2 + y^2 = 5) → 
  ∃ (x' y' : ℝ), ((x' - 2)^2 + y'^2 = 5 ∧ x' = -x ∧ y' = y) :=
sorry

end circle_reflection_y_axis_l21_2148


namespace no_rotation_matrix_exists_zero_matrix_is_answer_l21_2113

theorem no_rotation_matrix_exists : ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]] := by
  sorry

theorem zero_matrix_is_answer : 
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ), (0 : Matrix (Fin 2) (Fin 2) ℝ) * A ≠ ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) ∧
  (¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) :=
by
  sorry

end no_rotation_matrix_exists_zero_matrix_is_answer_l21_2113


namespace sphere_radius_ratio_l21_2117

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 450 * Real.pi) 
  (h2 : V_small = 0.25 * V_large) : 
  (V_small / V_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end sphere_radius_ratio_l21_2117


namespace orange_juice_ratio_l21_2179

-- Define the given quantities
def servings : Nat := 280
def serving_size : Nat := 6  -- in ounces
def concentrate_cans : Nat := 35
def concentrate_can_size : Nat := 12  -- in ounces

-- Define the theorem
theorem orange_juice_ratio :
  let total_juice := servings * serving_size
  let total_concentrate := concentrate_cans * concentrate_can_size
  let water_needed := total_juice - total_concentrate
  let water_cans := water_needed / concentrate_can_size
  (water_cans : Int) / (concentrate_cans : Int) = 3 / 1 := by
  sorry

end orange_juice_ratio_l21_2179


namespace star_neg_two_three_l21_2135

/-- The "star" operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b^2 + a

/-- Theorem: The result of (-2)☆3 is -20 -/
theorem star_neg_two_three : star (-2) 3 = -20 := by
  sorry

end star_neg_two_three_l21_2135


namespace field_trip_adults_l21_2154

/-- Field trip problem -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 12 →
  num_vans = 3 →
  (num_vans * van_capacity - num_students : ℕ) = 3 := by
  sorry

end field_trip_adults_l21_2154


namespace sphere_diameter_sum_l21_2121

theorem sphere_diameter_sum (r : ℝ) (d : ℝ) (a b : ℕ) : 
  r = 6 →
  d = 2 * (3 * (4 / 3 * π * r^3))^(1/3) →
  d = a * (b : ℝ)^(1/3) →
  b > 0 →
  ∀ k : ℕ, k > 1 → k^3 ∣ b → k = 1 →
  a + b = 15 := by
  sorry

end sphere_diameter_sum_l21_2121


namespace sufficient_not_necessary_l21_2125

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end sufficient_not_necessary_l21_2125
