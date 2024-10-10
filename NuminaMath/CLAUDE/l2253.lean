import Mathlib

namespace parallel_vectors_trig_expression_l2253_225311

/-- Given two vectors a and b in R², prove that if they are parallel,
    then a specific trigonometric expression involving their components equals 1/3. -/
theorem parallel_vectors_trig_expression (α : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin α)
  let b : ℝ × ℝ := (2, Real.cos α)
  (∃ (k : ℝ), a = k • b) →
  (Real.cos α - Real.sin α) / (2 * Real.cos (-α) - Real.sin α) = 1/3 := by
  sorry

end parallel_vectors_trig_expression_l2253_225311


namespace intersection_of_A_and_B_l2253_225344

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l2253_225344


namespace sum_of_coefficients_l2253_225353

theorem sum_of_coefficients (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) →
  a + b + c + d + e + f + g + h + j + k = 92 :=
by sorry

end sum_of_coefficients_l2253_225353


namespace trucks_sold_l2253_225303

theorem trucks_sold (total : ℕ) (car_truck_diff : ℕ) (h1 : total = 69) (h2 : car_truck_diff = 27) :
  ∃ trucks : ℕ, trucks * 2 + car_truck_diff = total ∧ trucks = 21 :=
by sorry

end trucks_sold_l2253_225303


namespace four_good_numbers_l2253_225341

/-- A real number k is a "good number" if the equation (x^2 - 1)(kx^2 - 6x - 8) = 0 
    has exactly three distinct real roots. -/
def is_good_number (k : ℝ) : Prop :=
  ∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    ∀ x : ℝ, (x^2 - 1) * (k * x^2 - 6 * x - 8) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃

/-- There are exactly 4 "good numbers". -/
theorem four_good_numbers : ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ k : ℝ, k ∈ s ↔ is_good_number k :=
  sorry

end four_good_numbers_l2253_225341


namespace sqrt_xy_plus_3_l2253_225384

theorem sqrt_xy_plus_3 (x y : ℝ) (h : y = Real.sqrt (1 - 4*x) + Real.sqrt (4*x - 1) + 4) :
  Real.sqrt (x*y + 3) = 2 := by
  sorry

end sqrt_xy_plus_3_l2253_225384


namespace zeros_of_f_range_of_b_l2253_225342

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Part 1
theorem zeros_of_f (b : ℝ) :
  f b 0 = f b 4 → (∃ x : ℝ, f b x = 0) ∧ (∀ x : ℝ, f b x = 0 → x = 3 ∨ x = 1) :=
sorry

-- Part 2
theorem range_of_b :
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ f b x = 0 ∧ f b y = 0) → b > 4 :=
sorry

end zeros_of_f_range_of_b_l2253_225342


namespace average_temperature_l2253_225335

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday given the conditions -/
theorem average_temperature (t w th : ℝ) : 
  (t + w + th + 33) / 4 = 46 →
  (41 + t + w + th) / 4 = 48 := by
sorry

end average_temperature_l2253_225335


namespace revenue_change_l2253_225346

theorem revenue_change (revenue_1995 : ℝ) : 
  let revenue_1996 := revenue_1995 * 1.2
  let revenue_1997 := revenue_1996 * 0.8
  (revenue_1995 - revenue_1997) / revenue_1995 * 100 = 4 := by
  sorry

end revenue_change_l2253_225346


namespace triangle_FIL_area_l2253_225387

-- Define the triangle and squares
structure Triangle :=
  (F I L : ℝ × ℝ)

structure Square :=
  (area : ℝ)

-- Define the problem setup
def triangle_FIL : Triangle := sorry
def square_GQOP : Square := ⟨10⟩
def square_HJNO : Square := ⟨90⟩
def square_RKMN : Square := ⟨40⟩

-- Function to check if squares are on triangle sides
def squares_on_triangle_sides (t : Triangle) (s1 s2 s3 : Square) : Prop := sorry

-- Function to calculate triangle area
def triangle_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_FIL_area :
  squares_on_triangle_sides triangle_FIL square_GQOP square_HJNO square_RKMN →
  triangle_area triangle_FIL = 220.5 := by
  sorry

end triangle_FIL_area_l2253_225387


namespace new_person_weight_is_77_l2253_225310

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (total_persons : ℕ) (average_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + total_persons * average_weight_increase

/-- Theorem stating that the weight of the new person is 77 kg given the problem conditions -/
theorem new_person_weight_is_77 :
  weight_of_new_person 8 1.5 65 = 77 := by
  sorry

#eval weight_of_new_person 8 1.5 65

end new_person_weight_is_77_l2253_225310


namespace tan_double_angle_l2253_225316

theorem tan_double_angle (α : ℝ) (h : Real.sin α - 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = -4/3 := by
  sorry

end tan_double_angle_l2253_225316


namespace magnitude_BD_l2253_225322

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

def parallelogram_ABCD (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

theorem magnitude_BD (D : ℂ) (h : parallelogram_ABCD A B C) : 
  Complex.abs (D - B) = Real.sqrt 13 := by
  sorry

end magnitude_BD_l2253_225322


namespace inequality_proof_l2253_225325

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end inequality_proof_l2253_225325


namespace inverse_sum_equals_negative_six_l2253_225377

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by sorry

end inverse_sum_equals_negative_six_l2253_225377


namespace constant_function_solution_l2253_225372

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

theorem constant_function_solution
  (f : ℝ → ℝ)
  (hf : FunctionalEquation f)
  (hnz : ∃ x, f x ≠ 0) :
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k := by
  sorry

end constant_function_solution_l2253_225372


namespace total_participants_is_280_l2253_225357

/-- The number of students who participated in at least one competition -/
def total_participants (math physics chem math_physics math_chem phys_chem all_three : ℕ) : ℕ :=
  math + physics + chem - math_physics - math_chem - phys_chem + all_three

/-- Theorem stating that the total number of participants is 280 given the conditions -/
theorem total_participants_is_280 :
  total_participants 203 179 165 143 116 97 89 = 280 := by
  sorry

#eval total_participants 203 179 165 143 116 97 89

end total_participants_is_280_l2253_225357


namespace lives_per_player_l2253_225354

/-- Given 8 friends playing a video game with a total of 64 lives,
    prove that each friend has 8 lives. -/
theorem lives_per_player (num_friends : ℕ) (total_lives : ℕ) :
  num_friends = 8 →
  total_lives = 64 →
  total_lives / num_friends = 8 :=
by sorry

end lives_per_player_l2253_225354


namespace base_conversion_problem_l2253_225393

theorem base_conversion_problem : ∃ (n A B : ℕ), 
  n > 0 ∧
  n = 8 * A + B ∧
  n = 6 * B + A ∧
  A < 8 ∧
  B < 6 ∧
  n = 47 := by
  sorry

end base_conversion_problem_l2253_225393


namespace cube_ratio_equals_64_l2253_225376

theorem cube_ratio_equals_64 : (88888 / 22222)^3 = 64 := by
  have h : 88888 / 22222 = 4 := by sorry
  sorry

end cube_ratio_equals_64_l2253_225376


namespace sarah_remaining_pages_l2253_225304

/-- Given the initial number of problems, the number of completed problems,
    and the number of problems per page, calculates the number of remaining pages. -/
def remaining_pages (initial_problems : ℕ) (completed_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (initial_problems - completed_problems) / problems_per_page

/-- Proves that Sarah has 5 pages of problems left to do. -/
theorem sarah_remaining_pages :
  remaining_pages 60 20 8 = 5 := by
  sorry

#eval remaining_pages 60 20 8

end sarah_remaining_pages_l2253_225304


namespace ones_digit_of_9_to_53_l2253_225382

theorem ones_digit_of_9_to_53 : Nat.mod (9^53) 10 = 9 := by
  sorry

end ones_digit_of_9_to_53_l2253_225382


namespace roll_distribution_probability_l2253_225308

def total_rolls : ℕ := 9
def rolls_per_type : ℕ := 3
def num_guests : ℕ := 3

def total_arrangements : ℕ := (total_rolls.factorial) / ((rolls_per_type.factorial) ^ 3)

def favorable_outcomes : ℕ := (rolls_per_type.factorial) ^ num_guests

def probability : ℚ := favorable_outcomes / total_arrangements

theorem roll_distribution_probability :
  probability = 9 / 70 := by sorry

end roll_distribution_probability_l2253_225308


namespace platform_length_calculation_platform_length_proof_l2253_225371

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length

/-- Proves that the platform length is approximately 190.08 m given the specified conditions --/
theorem platform_length_proof 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmph = 56) 
  (h3 : crossing_time = 18) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (platform_length_calculation train_length train_speed_kmph crossing_time - 190.08) < ε :=
by
  sorry

end platform_length_calculation_platform_length_proof_l2253_225371


namespace female_workers_count_l2253_225313

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of female workers based on the given workforce data --/
def calculate_female_workers (data : WorkforceData) : ℕ :=
  sorry

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_female_workers data = 15 := by
  sorry

end female_workers_count_l2253_225313


namespace tangent_line_slope_l2253_225364

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x

/-- A point on the curve where the tangent line passes through -/
noncomputable def x₀ : ℝ := 1

/-- The slope of the tangent line -/
noncomputable def k : ℝ := f' x₀

theorem tangent_line_slope : k = Real.exp 1 := by sorry

end tangent_line_slope_l2253_225364


namespace handshake_count_l2253_225362

theorem handshake_count (n : ℕ) (couples : ℕ) (extra_exemptions : ℕ) : 
  n = 2 * couples → 
  n ≥ 2 →
  extra_exemptions ≤ n - 2 →
  (n * (n - 2) - extra_exemptions) / 2 = 57 :=
by
  sorry

#check handshake_count 12 6 2

end handshake_count_l2253_225362


namespace f_properties_l2253_225390

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, x2 > x1 → f x2 > f x1) ∧
  (∀ x t : ℝ, x ∈ Set.Icc 1 2 → (f (x - t) + f (x^2 - t^2) ≥ 0 ↔ t ∈ Set.Icc (-2) 1)) :=
sorry

end f_properties_l2253_225390


namespace range_of_f_l2253_225358

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc 3 6 := by
  sorry

end range_of_f_l2253_225358


namespace div_three_sevenths_by_four_l2253_225383

theorem div_three_sevenths_by_four :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by
  sorry

end div_three_sevenths_by_four_l2253_225383


namespace parallel_vectors_sum_l2253_225361

/-- Given two parallel vectors a and b in R², prove that 3a + 2b equals (-1, -2) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  (a.1 * b.2 = a.2 * b.1) →  -- Parallel condition
  (3 * a.1 + 2 * b.1 = -1 ∧ 3 * a.2 + 2 * b.2 = -2) :=
by sorry

end parallel_vectors_sum_l2253_225361


namespace hyperbola_parabola_intersection_l2253_225318

/-- Given a hyperbola and a parabola with specific properties, prove the value of p -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1) →  -- Hyperbola equation
  (∀ x y : ℝ, x = 2 * p * y^2) →         -- Parabola equation
  (∃ e : ℝ, e = (4 : ℝ) / 2 ∧            -- Eccentricity of hyperbola
    (∀ y : ℝ, e = 2 * p * y^2)) →        -- Focus of parabola at (e, 0)
  p = 1 / 16 := by
sorry

end hyperbola_parabola_intersection_l2253_225318


namespace partial_fraction_decomposition_l2253_225319

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (8 * x + 1) / ((x - 4) * (x - 2)^2) =
  (33 / 4) / (x - 4) + (-19 / 4) / (x - 2) + (-17 / 2) / (x - 2)^2 := by
  sorry

end partial_fraction_decomposition_l2253_225319


namespace rhombus_area_l2253_225363

/-- The area of a rhombus with sides of length 3 cm and one internal angle of 45 degrees is (9√2)/2 square centimeters. -/
theorem rhombus_area (s : ℝ) (angle : ℝ) (h1 : s = 3) (h2 : angle = 45 * π / 180) :
  let area := s * s * Real.sin angle
  area = 9 * Real.sqrt 2 / 2 := by sorry

end rhombus_area_l2253_225363


namespace fifty_numbers_with_negative_products_l2253_225329

theorem fifty_numbers_with_negative_products (total : Nat) (neg_products : Nat) 
  (h1 : total = 50) (h2 : neg_products = 500) : 
  ∃ (m n p : Nat), m + n + p = total ∧ m * p = neg_products ∧ n = 5 := by
  sorry

end fifty_numbers_with_negative_products_l2253_225329


namespace initially_calculated_average_weight_l2253_225317

/-- Given a class of boys, prove that the initially calculated average weight
    is correct based on the given conditions. -/
theorem initially_calculated_average_weight
  (num_boys : ℕ)
  (correct_avg_weight : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : correct_avg_weight = 58.6)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 60) :
  let correct_total_weight := correct_avg_weight * num_boys
  let initial_total_weight := correct_total_weight - (correct_weight - misread_weight)
  let initial_avg_weight := initial_total_weight / num_boys
  initial_avg_weight = 58.4 := by
sorry

end initially_calculated_average_weight_l2253_225317


namespace lunch_cost_calculation_l2253_225374

/-- Calculates the total cost of lunches for a field trip --/
theorem lunch_cost_calculation (total_lunches : ℕ) 
  (vegetarian_lunches : ℕ) (gluten_free_lunches : ℕ) (both_veg_gf : ℕ)
  (regular_cost : ℕ) (special_cost : ℕ) (both_cost : ℕ) : 
  total_lunches = 44 ∧ 
  vegetarian_lunches = 10 ∧ 
  gluten_free_lunches = 5 ∧ 
  both_veg_gf = 2 ∧
  regular_cost = 7 ∧ 
  special_cost = 8 ∧ 
  both_cost = 9 → 
  (both_veg_gf * both_cost + 
   (vegetarian_lunches - both_veg_gf) * special_cost + 
   (gluten_free_lunches - both_veg_gf) * special_cost + 
   (total_lunches - vegetarian_lunches - gluten_free_lunches + both_veg_gf) * regular_cost) = 323 := by
  sorry


end lunch_cost_calculation_l2253_225374


namespace sqrt_equality_implies_m_and_n_l2253_225332

theorem sqrt_equality_implies_m_and_n (m n : ℝ) :
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt m - Real.sqrt n →
  m = 3 ∧ n = 2 := by
sorry

end sqrt_equality_implies_m_and_n_l2253_225332


namespace fred_basketball_games_l2253_225334

def games_this_year : ℕ := 36
def total_games : ℕ := 47

def games_last_year : ℕ := total_games - games_this_year

theorem fred_basketball_games : games_last_year = 11 := by
  sorry

end fred_basketball_games_l2253_225334


namespace julia_short_amount_l2253_225392

def rock_price : ℝ := 5
def pop_price : ℝ := 10
def dance_price : ℝ := 3
def country_price : ℝ := 7
def discount_rate : ℝ := 0.1
def julia_money : ℝ := 75

def rock_quantity : ℕ := 3
def pop_quantity : ℕ := 4
def dance_quantity : ℕ := 2
def country_quantity : ℕ := 4

def discount_threshold : ℕ := 3

def genre_cost (price : ℝ) (quantity : ℕ) : ℝ := price * quantity

def apply_discount (cost : ℝ) (quantity : ℕ) : ℝ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := apply_discount (genre_cost rock_price rock_quantity) rock_quantity
  let pop_cost := apply_discount (genre_cost pop_price pop_quantity) pop_quantity
  let dance_cost := apply_discount (genre_cost dance_price dance_quantity) dance_quantity
  let country_cost := apply_discount (genre_cost country_price country_quantity) country_quantity
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  total_cost - julia_money = 7.2 := by sorry

end julia_short_amount_l2253_225392


namespace largest_k_inequality_l2253_225398

theorem largest_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ k : ℝ, k > 174960 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) < k * a * b * c * d^3) ∧
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ 174960 * a * b * c * d^3) :=
by sorry

end largest_k_inequality_l2253_225398


namespace greatest_four_digit_number_l2253_225315

theorem greatest_four_digit_number (n : ℕ) : n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k₁ k₂ : ℕ, n = 11 * k₁ + 2 ∧ n = 7 * k₂ + 4 → n ≤ 9973 :=
by sorry

end greatest_four_digit_number_l2253_225315


namespace fraction_comparison_l2253_225381

theorem fraction_comparison (m : ℕ) (h : m = 23^1973) :
  (23^1873 + 1) / (23^1974 + 1) > (23^1974 + 1) / (23^1975 + 1) := by
sorry

end fraction_comparison_l2253_225381


namespace problem_solution_l2253_225333

def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

theorem problem_solution (a b : ℝ) :
  (B ⊆ A a b → a = -1) ∧
  (A a b ∩ B ≠ ∅ → a^2 - b^2 + 2*a = -1) := by
  sorry

end problem_solution_l2253_225333


namespace existence_of_property_P_one_third_l2253_225367

-- Define the property P(m) for a function f on an interval
def has_property_P (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  ∃ x₀ ∈ D, f x₀ = f (x₀ + m) ∧ x₀ + m ∈ D

-- Theorem statement
theorem existence_of_property_P_one_third
  (f : ℝ → ℝ) (h_cont : Continuous f) (h_eq : f 0 = f 2) :
  ∃ x₀ ∈ Set.Icc 0 (5/3), f x₀ = f (x₀ + 1/3) :=
by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end existence_of_property_P_one_third_l2253_225367


namespace min_value_a_plus_b_l2253_225312

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 1) (hb : b > 2) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ y > 2 ∧ x * y = 2 * x + y → a + b ≤ x + y) ∧ a + b = 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_a_plus_b_l2253_225312


namespace orange_juice_price_l2253_225320

/-- The cost of a glass of orange juice -/
def orange_juice_cost : ℚ := 85/100

/-- The cost of a bagel -/
def bagel_cost : ℚ := 95/100

/-- The cost of a sandwich -/
def sandwich_cost : ℚ := 465/100

/-- The cost of milk -/
def milk_cost : ℚ := 115/100

/-- The additional amount spent on lunch compared to breakfast -/
def lunch_breakfast_difference : ℚ := 4

theorem orange_juice_price : 
  bagel_cost + orange_juice_cost + lunch_breakfast_difference = sandwich_cost + milk_cost := by
  sorry

end orange_juice_price_l2253_225320


namespace slope_range_l2253_225368

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory of point P
def trajectory_P (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for points A and B
def condition_AB (xA yA xB yB : ℝ) : Prop :=
  xA * xB + yA * yB = -4 ∧
  4 * Real.sqrt 6 ≤ Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ∧
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ≤ 4 * Real.sqrt 30

-- Theorem statement
theorem slope_range (k : ℝ) :
  (∃ m xA yA xB yB,
    circle_F 1 0 ∧
    trajectory_P xA yA ∧ trajectory_P xB yB ∧
    line_l k m xA yA ∧ line_l k m xB yB ∧
    condition_AB xA yA xB yB ∧
    xA > 0 ∧ xB > 0 ∧ xA ≠ xB) →
  (k ∈ Set.Icc (-1) (-1/2) ∨ k ∈ Set.Icc (1/2) 1) :=
sorry

end slope_range_l2253_225368


namespace increasing_function_condition_l2253_225330

/-- The function f(x) = x^2 + a/x is increasing on (1, +∞) when 0 < a < 2 -/
theorem increasing_function_condition (a : ℝ) :
  (0 < a ∧ a < 2) →
  ∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (¬ ∀ a, (∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
    (∀ x y, 1 < x ∧ x < y → f x < f y)) → (0 < a ∧ a < 2)) :=
by sorry

end increasing_function_condition_l2253_225330


namespace max_sum_given_constraints_l2253_225349

/-- The maximum value of x+y given x^2 + y^2 = 100 and xy = 36 is 2√43 -/
theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 := by
  sorry

end max_sum_given_constraints_l2253_225349


namespace tangent_circles_theorem_l2253_225365

/-- Two circles in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point : Type := ℝ × ℝ

/-- Predicate to check if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on the common tangent of two circles -/
def on_common_tangent (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if the common tangent is perpendicular to the line joining the centers -/
def perpendicular_to_center_line (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is tangent to another circle -/
def is_tangent_to (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem tangent_circles_theorem 
  (c1 c2 : Circle) 
  (p : Point) 
  (h1 : are_tangent c1 c2)
  (h2 : on_common_tangent p c1 c2)
  (h3 : perpendicular_to_center_line p c1 c2) :
  ∃! (s1 s2 : Circle), 
    s1 ≠ s2 ∧ 
    is_tangent_to s1 c1 ∧ 
    is_tangent_to s1 c2 ∧ 
    point_on_circle p s1 ∧
    is_tangent_to s2 c1 ∧ 
    is_tangent_to s2 c2 ∧ 
    point_on_circle p s2 :=
  sorry

end tangent_circles_theorem_l2253_225365


namespace decimal_sum_l2253_225305

theorem decimal_sum : 0.5 + 0.035 + 0.0041 = 0.5391 := by
  sorry

end decimal_sum_l2253_225305


namespace opposite_numbers_quotient_l2253_225355

theorem opposite_numbers_quotient (p q : ℝ) (h1 : p ≠ 0) (h2 : p + q = 0) : |q| / p = -1 := by
  sorry

end opposite_numbers_quotient_l2253_225355


namespace intersection_of_A_and_B_l2253_225375

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 4 < x ∧ x < 7} := by
  sorry

end intersection_of_A_and_B_l2253_225375


namespace fraction_meaningful_iff_not_negative_one_l2253_225388

theorem fraction_meaningful_iff_not_negative_one (x : ℝ) :
  (∃ y : ℝ, y = (x - 1) / (x + 1)) ↔ x ≠ -1 := by
  sorry

end fraction_meaningful_iff_not_negative_one_l2253_225388


namespace root_product_one_l2253_225360

theorem root_product_one (b c : ℝ) (hb : b > 0) (hc : c > 0) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 + 2*b*x₁ + c = 0) ∧ 
    (x₂^2 + 2*b*x₂ + c = 0) ∧ 
    (x₃^2 + 2*c*x₃ + b = 0) ∧ 
    (x₄^2 + 2*c*x₄ + b = 0) ∧ 
    (x₁ * x₂ * x₃ * x₄ = 1)) → 
  b = 1 ∧ c = 1 :=
by sorry

end root_product_one_l2253_225360


namespace weight_problem_l2253_225340

/-- Given three weights A, B, and C, prove that their average weights satisfy certain conditions -/
theorem weight_problem (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)  -- The average weight of A, B, and C is 45 kg
  (h2 : (B + C) / 2 = 43)      -- The average weight of B and C is 43 kg
  (h3 : B = 31)                -- The weight of B is 31 kg
  : (A + B) / 2 = 40 :=        -- The average weight of A and B is 40 kg
by sorry

end weight_problem_l2253_225340


namespace sphere_radius_l2253_225351

theorem sphere_radius (d h : ℝ) (h1 : d = 30) (h2 : h = 10) : 
  ∃ r : ℝ, r^2 = d^2 / 4 + h^2 ∧ r = 5 * Real.sqrt 13 :=
sorry

end sphere_radius_l2253_225351


namespace sum_of_squares_and_square_of_sum_l2253_225343

theorem sum_of_squares_and_square_of_sum : (3 + 5 + 7)^2 + (3^2 + 5^2 + 7^2) = 308 := by
  sorry

end sum_of_squares_and_square_of_sum_l2253_225343


namespace mersenne_divisibility_l2253_225350

theorem mersenne_divisibility (n : ℕ+) :
  (∃ m : ℕ+, (2^n.val - 1) ∣ (m.val^2 + 81)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end mersenne_divisibility_l2253_225350


namespace pen_pencil_cost_total_cost_is_13_l2253_225379

/-- The total cost of a pen and a pencil, where the pen costs $9 more than the pencil and the pencil costs $2. -/
theorem pen_pencil_cost : ℕ → ℕ → ℕ
  | pencil_cost, pen_extra_cost =>
    let pen_cost := pencil_cost + pen_extra_cost
    pencil_cost + pen_cost

/-- Proof that the total cost of a pen and a pencil is $13, given the conditions. -/
theorem total_cost_is_13 : pen_pencil_cost 2 9 = 13 := by
  sorry

end pen_pencil_cost_total_cost_is_13_l2253_225379


namespace wood_piece_weight_relation_l2253_225369

/-- Represents a square piece of wood -/
structure WoodPiece where
  sideLength : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square wood pieces -/
theorem wood_piece_weight_relation 
  (piece1 piece2 : WoodPiece)
  (h1 : piece1.sideLength = 4)
  (h2 : piece1.weight = 16)
  (h3 : piece2.sideLength = 6)
  : piece2.weight = 36 := by
  sorry

end wood_piece_weight_relation_l2253_225369


namespace michael_and_anna_ages_l2253_225397

theorem michael_and_anna_ages :
  ∀ (michael anna : ℕ),
  michael = anna + 8 →
  michael + 12 = 3 * (anna - 6) →
  michael + anna = 46 :=
by sorry

end michael_and_anna_ages_l2253_225397


namespace cylinder_lateral_area_l2253_225386

/-- Given a cylinder with a rectangular front view of area 6,
    prove that its lateral area is 6π. -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h
  let lateral_area := π * d * h
  lateral_area = 6 * π := by
  sorry

end cylinder_lateral_area_l2253_225386


namespace cubic_sum_minus_product_l2253_225309

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 13) 
  (product_sum_condition : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 637 := by
  sorry

end cubic_sum_minus_product_l2253_225309


namespace sum_of_max_min_on_interval_l2253_225370

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem sum_of_max_min_on_interval :
  let a : ℝ := 0
  let b : ℝ := 3
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max + min = -10) :=
by sorry

end sum_of_max_min_on_interval_l2253_225370


namespace factor_implies_root_l2253_225366

theorem factor_implies_root (a : ℝ) : 
  (∀ t : ℝ, (2*t + 1) ∣ (4*t^2 + 12*t + a)) → a = 5 := by
  sorry

end factor_implies_root_l2253_225366


namespace least_positive_integer_congruence_l2253_225323

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7351 : ℤ) ≡ 3071 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 7351 : ℤ) ≡ 3071 [ZMOD 17] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l2253_225323


namespace cloth_cost_per_metre_l2253_225324

theorem cloth_cost_per_metre (total_length : Real) (total_cost : Real) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 425.50) :
  total_cost / total_length = 46 := by
  sorry

end cloth_cost_per_metre_l2253_225324


namespace wendy_recycling_l2253_225347

/-- Given that Wendy earns 5 points per bag recycled, had 11 bags in total, 
    and earned 45 points, prove that she did not recycle 2 bags. -/
theorem wendy_recycling (points_per_bag : ℕ) (total_bags : ℕ) (total_points : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 11)
  (h3 : total_points = 45) :
  total_bags - (total_points / points_per_bag) = 2 := by
  sorry


end wendy_recycling_l2253_225347


namespace nanometers_to_meters_l2253_225359

-- Define the conversion factors
def nanometer_to_millimeter : ℝ := 1e-6
def millimeter_to_meter : ℝ := 1e-3

-- Define the given length in nanometers
def length_in_nanometers : ℝ := 3e10

-- State the theorem
theorem nanometers_to_meters :
  length_in_nanometers * nanometer_to_millimeter * millimeter_to_meter = 30 := by
  sorry

end nanometers_to_meters_l2253_225359


namespace regular_polygon_with_108_degree_interior_angles_l2253_225394

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 108) → 
  n = 5 := by
  sorry

end regular_polygon_with_108_degree_interior_angles_l2253_225394


namespace amy_total_score_l2253_225378

/-- Calculates the total score for Amy's video game performance --/
def total_score (treasure_points enemy_points : ℕ)
                (level1_treasures level1_enemies : ℕ)
                (level2_enemies : ℕ) : ℕ :=
  let level1_score := treasure_points * level1_treasures + enemy_points * level1_enemies
  let level2_score := enemy_points * level2_enemies * 2
  level1_score + level2_score

/-- Theorem stating that Amy's total score is 154 points --/
theorem amy_total_score :
  total_score 4 10 6 3 5 = 154 :=
by sorry

end amy_total_score_l2253_225378


namespace divisible_by_2000_arrangement_l2253_225339

theorem divisible_by_2000_arrangement (nums : Vector ℕ 23) :
  ∃ (arrangement : List (Sum (Prod ℕ ℕ) ℕ)),
    (arrangement.foldl (λ acc x => match x with
      | Sum.inl (a, b) => acc * (a * b)
      | Sum.inr a => acc + a
    ) 0) % 2000 = 0 :=
sorry

end divisible_by_2000_arrangement_l2253_225339


namespace cistern_problem_l2253_225328

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that for a cistern with given dimensions, the wet surface area is 88 square meters -/
theorem cistern_problem : 
  cistern_wet_surface_area 12 4 1.25 = 88 := by
  sorry

#eval cistern_wet_surface_area 12 4 1.25

end cistern_problem_l2253_225328


namespace larry_cards_larry_cards_proof_l2253_225336

/-- If Larry initially has 67 cards and Dennis takes 9 cards away, 
    then Larry will have 58 cards remaining. -/
theorem larry_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards cards_taken remaining_cards =>
    initial_cards = 67 ∧ 
    cards_taken = 9 ∧ 
    remaining_cards = initial_cards - cards_taken →
    remaining_cards = 58

-- The proof would go here
theorem larry_cards_proof : larry_cards 67 9 58 := by
  sorry

end larry_cards_larry_cards_proof_l2253_225336


namespace exponent_division_l2253_225345

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a^m / a^n = a^(m - n) := by
  sorry

end exponent_division_l2253_225345


namespace lindas_coins_value_l2253_225348

theorem lindas_coins_value :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 150 →
  5 * n + 10 * d + 25 * q = 500 :=
by
  sorry

end lindas_coins_value_l2253_225348


namespace total_weight_is_675_l2253_225326

/-- The total weight Tom is moving with, given his weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight_moved (tom_weight : ℝ) (hand_weight_multiplier : ℝ) (vest_weight_multiplier : ℝ) : ℝ :=
  tom_weight + (vest_weight_multiplier * tom_weight) + (2 * hand_weight_multiplier * tom_weight)

/-- Theorem stating that the total weight Tom is moving with is 675 kg -/
theorem total_weight_is_675 :
  total_weight_moved 150 1.5 0.5 = 675 := by
  sorry

end total_weight_is_675_l2253_225326


namespace price_difference_shirt_sweater_l2253_225389

theorem price_difference_shirt_sweater : 
  ∀ (shirt_price sweater_price : ℝ),
    shirt_price = 36.46 →
    shirt_price < sweater_price →
    shirt_price + sweater_price = 80.34 →
    sweater_price - shirt_price = 7.42 := by
sorry

end price_difference_shirt_sweater_l2253_225389


namespace function_value_at_cos_15_deg_l2253_225331

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 1

theorem function_value_at_cos_15_deg :
  f (Real.cos (15 * π / 180)) = -(Real.sqrt 3 / 2) - 1 :=
by sorry

end function_value_at_cos_15_deg_l2253_225331


namespace classroom_handshakes_l2253_225314

theorem classroom_handshakes (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) 
  (h3 : 2 * m * n - m - n = 252) : m * n = 72 := by
  sorry

end classroom_handshakes_l2253_225314


namespace exists_long_period_in_range_l2253_225321

/-- The length of the period of the decimal expansion of 1/n -/
def period_length (n : ℕ) : ℕ := sorry

theorem exists_long_period_in_range :
  ∀ (start : ℕ), 
  (10^99 ≤ start) →
  ∃ (n : ℕ), 
    (start ≤ n) ∧ 
    (n < start + 100000) ∧ 
    (period_length n > 2011) := by
  sorry

end exists_long_period_in_range_l2253_225321


namespace quadratic_sum_l2253_225302

/-- Given a quadratic equation 100x^2 + 80x - 144 = 0, rewritten as (dx + e)^2 = f,
    where d, e, and f are integers and d > 0, prove that d + e + f = 174 -/
theorem quadratic_sum (d e f : ℤ) : 
  d > 0 → 
  (∀ x, 100 * x^2 + 80 * x - 144 = 0 ↔ (d * x + e)^2 = f) →
  d + e + f = 174 :=
by sorry

end quadratic_sum_l2253_225302


namespace division_problem_l2253_225385

theorem division_problem : ∃ (n : ℕ), n = 12401 ∧ n / 163 = 76 ∧ n % 163 = 13 := by
  sorry

end division_problem_l2253_225385


namespace coffee_shop_total_sales_l2253_225306

/-- Calculates the total money made by a coffee shop given the number of coffee and tea orders and their respective prices. -/
def coffee_shop_sales (coffee_orders : ℕ) (coffee_price : ℕ) (tea_orders : ℕ) (tea_price : ℕ) : ℕ :=
  coffee_orders * coffee_price + tea_orders * tea_price

/-- Theorem stating that the coffee shop made $67 given the specified orders and prices. -/
theorem coffee_shop_total_sales :
  coffee_shop_sales 7 5 8 4 = 67 := by
  sorry

end coffee_shop_total_sales_l2253_225306


namespace average_problem_k_problem_point_problem_quadratic_problem_l2253_225395

-- Question 1
theorem average_problem (p q r t : ℝ) :
  (p + q + r) / 3 = 12 ∧ (p + q + r + t + 2*t) / 5 = 15 → t = 13 := by sorry

-- Question 2
theorem k_problem (k s : ℝ) :
  k^4 + 1/k^4 = 14 ∧ s = k^2 + 1/k^2 → s = 4 := by sorry

-- Question 3
theorem point_problem (a b s : ℝ) :
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (11, 7)
  let P : ℝ × ℝ := (a, b)
  P.1 = (1 * N.1 + s * M.1) / (1 + s) ∧
  P.2 = (1 * N.2 + s * M.2) / (1 + s) ∧
  s = 4 → a = 3 := by sorry

-- Question 4
theorem quadratic_problem (a c : ℝ) :
  a = 3 ∧ (∃ x : ℝ, a * x^2 + 12 * x + c = 0 ∧
    ∀ y : ℝ, y ≠ x → a * y^2 + 12 * y + c ≠ 0) → c = 12 := by sorry

end average_problem_k_problem_point_problem_quadratic_problem_l2253_225395


namespace product_divisible_by_seven_l2253_225391

theorem product_divisible_by_seven (A B : ℕ+) 
  (hA : Nat.Prime A.val)
  (hB : Nat.Prime B.val)
  (hAminusB : Nat.Prime (A.val - B.val))
  (hAplusB : Nat.Prime (A.val + B.val)) :
  7 ∣ (A.val * B.val * (A.val - B.val) * (A.val + B.val)) := by
sorry

end product_divisible_by_seven_l2253_225391


namespace line_tangent_to_ellipse_l2253_225300

/-- The line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 if and only if m^2 = 1/3 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → (∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2)) ↔
  m^2 = 1/3 := by
sorry


end line_tangent_to_ellipse_l2253_225300


namespace intersection_points_count_l2253_225338

/-- The number of points with positive x-coordinates that lie on at least two of the graphs
    y = log₂x, y = 1/log₂x, y = -log₂x, and y = -1/log₂x -/
theorem intersection_points_count : ℕ := by
  sorry

#check intersection_points_count

end intersection_points_count_l2253_225338


namespace initial_maple_trees_l2253_225337

/-- The number of maple trees in the park after planting -/
def total_trees : ℕ := 64

/-- The number of maple trees planted today -/
def planted_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_maple_trees : initial_trees = 53 := by
  sorry

end initial_maple_trees_l2253_225337


namespace division_of_fractions_l2253_225396

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end division_of_fractions_l2253_225396


namespace circle_equation_from_diameter_l2253_225380

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_from_diameter (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 0)
  let N : ℝ × ℝ := (0, 4)
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
      radius^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ∧
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_equation_from_diameter_l2253_225380


namespace only_81_satisfies_l2253_225373

/-- A function that returns true if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that swaps the digits of a two-digit number -/
def swapDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The main theorem -/
theorem only_81_satisfies : ∃! n : ℕ, isTwoDigit n ∧ (swapDigits n)^2 = 4 * n :=
  sorry

end only_81_satisfies_l2253_225373


namespace remainder_problem_l2253_225301

theorem remainder_problem (n m p : ℕ) 
  (hn : n % 4 = 3)
  (hm : m % 7 = 5)
  (hp : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 := by
  sorry

end remainder_problem_l2253_225301


namespace hyperbola_asymptote_l2253_225352

/-- Theorem: For a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has equation y = 3x, then b = 3. -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∃ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 3 * x ∧ x^2 - y^2 / b^2 = 1) →
  b = 3 := by
  sorry

end hyperbola_asymptote_l2253_225352


namespace correlation_coefficient_relationship_l2253_225307

-- Define the data points
def x_data : List ℝ := [1, 2, 3, 4, 5]
def y_data : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U_data : List ℝ := [1, 2, 3, 4, 5]
def V_data : List ℝ := [12.7, 10.2, 7, 3.6, 1]

-- Define linear correlation coefficient
def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define r₁ and r₂
def r₁ : ℝ := linear_correlation_coefficient x_data y_data
def r₂ : ℝ := linear_correlation_coefficient U_data V_data

-- Theorem to prove
theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end correlation_coefficient_relationship_l2253_225307


namespace quadratic_inequality_coefficients_l2253_225399

theorem quadratic_inequality_coefficients 
  (a b : ℝ) 
  (h1 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | 5 - x > 7 * |x + 1|})
  (h2 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | a * x^2 + b * x - 2 > 0}) :
  a = -4 ∧ b = -9 := by
sorry

end quadratic_inequality_coefficients_l2253_225399


namespace soccer_campers_l2253_225356

theorem soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : football = 32) : 
  total - (basketball + football) = 32 := by
  sorry

end soccer_campers_l2253_225356


namespace redskins_win_streak_probability_l2253_225327

/-- The probability of arranging wins and losses in exactly three winning streaks -/
theorem redskins_win_streak_probability 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (h1 : total_games = wins + losses)
  (h2 : wins = 10)
  (h3 : losses = 6) :
  (Nat.choose 9 2 * Nat.choose 7 3 : ℚ) / Nat.choose total_games losses = 45 / 286 := by
sorry

end redskins_win_streak_probability_l2253_225327
