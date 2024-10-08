import Mathlib

namespace miniature_tower_height_l189_189682

-- Definitions of conditions
def actual_tower_height := 60
def actual_dome_volume := 200000 -- in liters
def miniature_dome_volume := 0.4 -- in liters

-- Goal: Prove the height of the miniature tower
theorem miniature_tower_height
  (actual_tower_height: ℝ)
  (actual_dome_volume: ℝ)
  (miniature_dome_volume: ℝ) : 
  actual_tower_height = 60 ∧ actual_dome_volume = 200000 ∧ miniature_dome_volume = 0.4 →
  (actual_tower_height / ( (actual_dome_volume / miniature_dome_volume)^(1/3) )) = 1.2 :=
by
  sorry

end miniature_tower_height_l189_189682


namespace claire_earning_l189_189388

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l189_189388


namespace common_ratio_l189_189424

theorem common_ratio (a_3 S_3 : ℝ) (q : ℝ) 
  (h1 : a_3 = 3 / 2) 
  (h2 : S_3 = 9 / 2)
  (h3 : S_3 = (1 + q + q^2) * a_3 / q^2) :
  q = 1 ∨ q = -1 / 2 := 
by 
  sorry

end common_ratio_l189_189424


namespace find_positive_square_root_l189_189214

theorem find_positive_square_root (x : ℝ) (h_pos : x > 0) (h_eq : x^2 = 625) : x = 25 :=
sorry

end find_positive_square_root_l189_189214


namespace solve_for_x_l189_189961

theorem solve_for_x (x : ℝ) (h : (x^2 + 4*x - 5)^0 = 1) : x^2 - 5*x + 5 = 1 → x = 4 := 
by
  intro h2
  have : ∀ x, (x^2 + 4*x - 5 = 0) ↔ false := sorry
  exact sorry

end solve_for_x_l189_189961


namespace solve_n_minus_m_l189_189224

theorem solve_n_minus_m :
  ∃ m n, 
    (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m < 1000 ∧ 
    (n ≡ 4 [MOD 7]) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
    n - m = 903 :=
by
  sorry

end solve_n_minus_m_l189_189224


namespace probability_of_neither_tamil_nor_english_l189_189877

-- Definitions based on the conditions
def TotalPopulation := 1500
def SpeakTamil := 800
def SpeakEnglish := 650
def SpeakTamilAndEnglish := 250

-- Use Inclusion-Exclusion Principle
def SpeakTamilOrEnglish : ℕ := SpeakTamil + SpeakEnglish - SpeakTamilAndEnglish

-- Number of people who speak neither Tamil nor English
def SpeakNeitherTamilNorEnglish : ℕ := TotalPopulation - SpeakTamilOrEnglish

-- The probability calculation
def Probability := (SpeakNeitherTamilNorEnglish : ℚ) / (TotalPopulation : ℚ)

-- Theorem to prove
theorem probability_of_neither_tamil_nor_english : Probability = (1/5 : ℚ) :=
sorry

end probability_of_neither_tamil_nor_english_l189_189877


namespace number_of_people_needed_to_lift_car_l189_189964

-- Define the conditions as Lean definitions
def twice_as_many_people_to_lift_truck (C T : ℕ) : Prop :=
  T = 2 * C

def people_needed_for_cars_and_trucks (C T total_people : ℕ) : Prop :=
  60 = 6 * C + 3 * T

-- Define the theorem statement using the conditions
theorem number_of_people_needed_to_lift_car :
  ∃ C, (∃ T, twice_as_many_people_to_lift_truck C T) ∧ people_needed_for_cars_and_trucks C T 60 ∧ C = 5 :=
sorry

end number_of_people_needed_to_lift_car_l189_189964


namespace price_decrease_percentage_l189_189967

theorem price_decrease_percentage (original_price : ℝ) :
  let first_sale_price := (4/5) * original_price
  let second_sale_price := (1/2) * original_price
  let decrease := first_sale_price - second_sale_price
  let percentage_decrease := (decrease / first_sale_price) * 100
  percentage_decrease = 37.5 := by
  sorry

end price_decrease_percentage_l189_189967


namespace paths_mat8_l189_189514

-- Define variables
def grid := [
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"]
]

def is_adjacent (x1 y1 x2 y2 : Nat): Bool :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def count_paths (grid: List (List String)): Nat :=
  -- implementation to count number of paths
  4 * 4 * 2

theorem paths_mat8 (grid: List (List String)): count_paths grid = 32 := by
  sorry

end paths_mat8_l189_189514


namespace new_price_after_increase_l189_189876

def original_price : ℝ := 220
def percentage_increase : ℝ := 0.15

def new_price (original_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_price + (original_price * percentage_increase)

theorem new_price_after_increase : new_price original_price percentage_increase = 253 := 
by
  sorry

end new_price_after_increase_l189_189876


namespace sufficient_but_not_necessary_condition_l189_189536

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 1 ∧ y = 1 → x + y = 2) ∧ (¬(x + y = 2 → x = 1 ∧ y = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l189_189536


namespace rate_per_square_meter_l189_189036

-- Define the conditions
def length (L : ℝ) := L = 8
def width (W : ℝ) := W = 4.75
def total_cost (C : ℝ) := C = 34200
def area (A : ℝ) (L W : ℝ) := A = L * W
def rate (R C A : ℝ) := R = C / A

-- The theorem to prove
theorem rate_per_square_meter (L W C A R : ℝ) 
  (hL : length L) (hW : width W) (hC : total_cost C) (hA : area A L W) : 
  rate R C A :=
by
  -- By the conditions, length is 8, width is 4.75, and total cost is 34200.
  simp [length, width, total_cost, area, rate] at hL hW hC hA ⊢
  -- It remains to calculate the rate and use conditions
  have hA : A = L * W := hA
  rw [hL, hW] at hA
  have hA' : A = 8 * 4.75 := by simp [hA]
  rw [hA']
  simp [rate]
  sorry -- The detailed proof is omitted.

end rate_per_square_meter_l189_189036


namespace ratio_x_2y_l189_189050

theorem ratio_x_2y (x y : ℤ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : x / (2 * y) = 3 / 2 :=
sorry

end ratio_x_2y_l189_189050


namespace geometric_sequence_a4_value_l189_189532

variable {α : Type} [LinearOrderedField α]

noncomputable def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, n < m → ∃ r : α, 0 < r ∧ a m = a n * r^(m - n)

theorem geometric_sequence_a4_value (a : ℕ → α)
  (pos : ∀ n, 0 < a n)
  (geo_seq : is_geometric_sequence a)
  (h : a 1 * a 7 = 36) :
  a 4 = 6 :=
by 
  sorry

end geometric_sequence_a4_value_l189_189532


namespace total_rainfall_in_2004_l189_189156

noncomputable def average_monthly_rainfall_2003 : ℝ := 35.0
noncomputable def average_monthly_rainfall_2004 : ℝ := average_monthly_rainfall_2003 + 4.0
noncomputable def total_rainfall_2004 : ℝ := 
  let regular_months := 11 * average_monthly_rainfall_2004
  let daily_rainfall_feb := average_monthly_rainfall_2004 / 30
  let feb_rain := daily_rainfall_feb * 29 
  regular_months + feb_rain

theorem total_rainfall_in_2004 : total_rainfall_2004 = 466.7 := by
  sorry

end total_rainfall_in_2004_l189_189156


namespace inequality_solution_l189_189844

theorem inequality_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b + c = 1) : (1 / (b * c + a + 1 / a) + 1 / (c * a + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31) :=
by sorry

end inequality_solution_l189_189844


namespace percentage_x_equals_twenty_percent_of_487_50_is_65_l189_189828

theorem percentage_x_equals_twenty_percent_of_487_50_is_65
    (x : ℝ)
    (hx : x = 150)
    (y : ℝ)
    (hy : y = 487.50) :
    (∃ (P : ℝ), P * x = 0.20 * y ∧ P * 100 = 65) :=
by
  sorry

end percentage_x_equals_twenty_percent_of_487_50_is_65_l189_189828


namespace hyperbola_equation_l189_189744

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_y_axis (x : ℝ) : Prop := x = 0
def focal_distance (d : ℝ) : Prop := d = 4
def point_on_hyperbola (x y : ℝ) : Prop := x = 1 ∧ y = -Real.sqrt 3

-- Final statement to prove
theorem hyperbola_equation :
  (center_at_origin 0 0) ∧
  (focus_on_y_axis 0) ∧
  (focal_distance 4) ∧
  (point_on_hyperbola 1 (-Real.sqrt 3))
  → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = Real.sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l189_189744


namespace bookA_net_change_bookB_net_change_bookC_net_change_l189_189172

-- Define the price adjustments for Book A
def bookA_initial_price := 100.0
def bookA_after_first_adjustment := bookA_initial_price * (1 - 0.5)
def bookA_after_second_adjustment := bookA_after_first_adjustment * (1 + 0.6)
def bookA_final_price := bookA_after_second_adjustment * (1 + 0.1)
def bookA_net_percentage_change := (bookA_final_price - bookA_initial_price) / bookA_initial_price * 100

-- Define the price adjustments for Book B
def bookB_initial_price := 100.0
def bookB_after_first_adjustment := bookB_initial_price * (1 + 0.2)
def bookB_after_second_adjustment := bookB_after_first_adjustment * (1 - 0.3)
def bookB_final_price := bookB_after_second_adjustment * (1 + 0.25)
def bookB_net_percentage_change := (bookB_final_price - bookB_initial_price) / bookB_initial_price * 100

-- Define the price adjustments for Book C
def bookC_initial_price := 100.0
def bookC_after_first_adjustment := bookC_initial_price * (1 + 0.4)
def bookC_after_second_adjustment := bookC_after_first_adjustment * (1 - 0.1)
def bookC_final_price := bookC_after_second_adjustment * (1 - 0.05)
def bookC_net_percentage_change := (bookC_final_price - bookC_initial_price) / bookC_initial_price * 100

-- Statements to prove the net percentage changes
theorem bookA_net_change : bookA_net_percentage_change = -12 := by
  sorry

theorem bookB_net_change : bookB_net_percentage_change = 5 := by
  sorry

theorem bookC_net_change : bookC_net_percentage_change = 19.7 := by
  sorry

end bookA_net_change_bookB_net_change_bookC_net_change_l189_189172


namespace goods_train_speed_l189_189734

-- Define the given constants
def train_length : ℕ := 370 -- in meters
def platform_length : ℕ := 150 -- in meters
def crossing_time : ℕ := 26 -- in seconds
def conversion_factor : ℕ := 36 / 10 -- conversion from m/s to km/hr

-- Define the total distance covered
def total_distance : ℕ := train_length + platform_length -- in meters

-- Define the speed of the train in m/s
def speed_m_per_s : ℕ := total_distance / crossing_time

-- Define the speed of the train in km/hr
def speed_km_per_hr : ℕ := speed_m_per_s * conversion_factor

-- The proof problem statement
theorem goods_train_speed : speed_km_per_hr = 72 := 
by 
  -- Placeholder for the proof
  sorry

end goods_train_speed_l189_189734


namespace find_f_neg5_l189_189633

-- Define the function f and the constants a, b, and c
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

-- State the main theorem we want to prove
theorem find_f_neg5 (a b c : ℝ) (h : f 5 a b c = 9) : f (-5) a b c = 1 :=
by
  sorry

end find_f_neg5_l189_189633


namespace students_like_both_l189_189453

theorem students_like_both (total_students French_fries_likers burger_likers neither_likers : ℕ)
(H1 : total_students = 25)
(H2 : French_fries_likers = 15)
(H3 : burger_likers = 10)
(H4 : neither_likers = 6)
: (French_fries_likers + burger_likers + neither_likers - total_students) = 12 :=
by sorry

end students_like_both_l189_189453


namespace probability_of_specific_choice_l189_189447

-- Define the sets of subjects
inductive Subject
| Chinese
| Mathematics
| ForeignLanguage
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

-- Define the conditions of the examination mode "3+1+2"
def threeSubjects := [Subject.Chinese, Subject.Mathematics, Subject.ForeignLanguage]
def oneSubject := [Subject.Physics, Subject.History]
def twoSubjects := [Subject.PoliticalScience, Subject.Geography, Subject.Chemistry, Subject.Biology]

-- Calculate the total number of ways to choose one subject from Physics or History and two subjects from PoliticalScience, Geography, Chemistry, and Biology
def totalWays : Nat := 2 * Nat.choose 4 2  -- 2 choices for "1" part, and C(4, 2) ways for "2" part

-- Calculate the probability that a candidate will choose Political Science, History, and Geography
def favorableOutcome := 1  -- Only one specific combination counts

theorem probability_of_specific_choice :
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  (specific_combination : ℚ) / total_ways = 1 / 12 :=
by
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  show (specific_combination : ℚ) / total_ways = 1 / 12
  sorry

end probability_of_specific_choice_l189_189447


namespace find_tangent_parallel_to_x_axis_l189_189505

theorem find_tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), y = x^2 - 3 * x ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) := 
by
  sorry

end find_tangent_parallel_to_x_axis_l189_189505


namespace Ram_Shyam_weight_ratio_l189_189026

theorem Ram_Shyam_weight_ratio :
  ∃ (R S : ℝ), 
    (1.10 * R + 1.21 * S = 82.8) ∧ 
    (1.15 * (R + S) = 82.8) ∧ 
    (R / S = 1.20) :=
by {
  sorry
}

end Ram_Shyam_weight_ratio_l189_189026


namespace percentage_defective_l189_189538

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end percentage_defective_l189_189538


namespace yeast_population_at_1_20_pm_l189_189813

def yeast_population (initial : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  initial * rate^time

theorem yeast_population_at_1_20_pm : 
  yeast_population 50 3 4 = 4050 :=
by
  -- Proof goes here
  sorry

end yeast_population_at_1_20_pm_l189_189813


namespace distance_from_origin_l189_189312

noncomputable def point_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin (x y : ℝ) (h₁ : abs y = 15) (h₂ : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h₃ : x > 2) :
  point_distance x y = Real.sqrt (334 + 4 * Real.sqrt 105) :=
by
  sorry

end distance_from_origin_l189_189312


namespace street_lights_per_side_l189_189798

theorem street_lights_per_side
  (neighborhoods : ℕ)
  (roads_per_neighborhood : ℕ)
  (total_street_lights : ℕ)
  (total_neighborhoods : neighborhoods = 10)
  (roads_in_each_neighborhood : roads_per_neighborhood = 4)
  (street_lights_in_town : total_street_lights = 20000) :
  (total_street_lights / (neighborhoods * roads_per_neighborhood * 2) = 250) :=
by
  sorry

end street_lights_per_side_l189_189798


namespace isosceles_triangle_perimeter_correct_l189_189108

noncomputable def isosceles_triangle_perimeter (x y : ℝ) : ℝ :=
  if x = y then 2 * x + y else if (2 * x > y ∧ y > 2 * x - y) ∨ (2 * y > x ∧ x > 2 * y - x) then 2 * y + x else 0

theorem isosceles_triangle_perimeter_correct (x y : ℝ) (h : |x - 5| + (y - 8)^2 = 0) :
  isosceles_triangle_perimeter x y = 18 ∨ isosceles_triangle_perimeter x y = 21 := by
sorry

end isosceles_triangle_perimeter_correct_l189_189108


namespace land_for_crop_production_l189_189237

-- Conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def cattle_rearing : ℕ := 40

-- Proof statement defining the goal
theorem land_for_crop_production : 
  total_land - (house_and_machinery + future_expansion + cattle_rearing) = 70 := 
by
  sorry

end land_for_crop_production_l189_189237


namespace find_a_plus_b_l189_189330

theorem find_a_plus_b 
  (a b : ℝ)
  (f : ℝ → ℝ) 
  (f_def : ∀ x, f x = x^3 + 3 * x^2 + 6 * x + 14)
  (cond_a : f a = 1) 
  (cond_b : f b = 19) :
  a + b = -2 :=
sorry

end find_a_plus_b_l189_189330


namespace johns_yearly_grass_cutting_cost_l189_189021

-- Definitions of the conditions
def initial_height : ℝ := 2.0
def growth_rate : ℝ := 0.5
def cutting_height : ℝ := 4.0
def cost_per_cut : ℝ := 100.0
def months_per_year : ℝ := 12.0

-- Formulate the statement
theorem johns_yearly_grass_cutting_cost :
  let months_to_grow : ℝ := (cutting_height - initial_height) / growth_rate
  let cuts_per_year : ℝ := months_per_year / months_to_grow
  let total_cost_per_year : ℝ := cuts_per_year * cost_per_cut
  total_cost_per_year = 300.0 :=
by
  sorry

end johns_yearly_grass_cutting_cost_l189_189021


namespace num_large_posters_l189_189139

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l189_189139


namespace sin_eleven_pi_over_three_l189_189467

theorem sin_eleven_pi_over_three : Real.sin (11 * Real.pi / 3) = -((Real.sqrt 3) / 2) :=
by
  -- Conversion factor between radians and degrees
  -- periodicity of sine function: sin theta = sin (theta + n * 360 degrees) for any integer n
  -- the sine function is odd: sin (-theta) = -sin theta
  -- sin 60 degrees = sqrt(3)/2
  sorry

end sin_eleven_pi_over_three_l189_189467


namespace compare_expr_l189_189381

theorem compare_expr (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) :=
sorry

end compare_expr_l189_189381


namespace return_trip_time_l189_189181

theorem return_trip_time 
  (d p w : ℝ) 
  (h1 : d = 90 * (p - w))
  (h2 : ∀ t, t = d / p → d / (p + w) = t - 15) : 
  d / (p + w) = 64 :=
by
  sorry

end return_trip_time_l189_189181


namespace nina_total_spent_l189_189976

open Real

def toy_price : ℝ := 10
def toy_count : ℝ := 3
def toy_discount : ℝ := 0.15

def card_price : ℝ := 5
def card_count : ℝ := 2
def card_discount : ℝ := 0.10

def shirt_price : ℝ := 6
def shirt_count : ℝ := 5
def shirt_discount : ℝ := 0.20

def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_price (price : ℝ) (count : ℝ) (discount : ℝ) : ℝ :=
  count * price * (1 - discount)

noncomputable def total_cost_before_tax : ℝ := 
  discounted_price toy_price toy_count toy_discount +
  discounted_price card_price card_count card_discount +
  discounted_price shirt_price shirt_count shirt_discount

noncomputable def total_cost_after_tax : ℝ :=
  total_cost_before_tax * (1 + sales_tax_rate)

theorem nina_total_spent : total_cost_after_tax = 62.60 :=
by
  sorry

end nina_total_spent_l189_189976


namespace syllogism_correct_l189_189945

theorem syllogism_correct 
  (natnum : ℕ → Prop) 
  (intnum : ℤ → Prop) 
  (is_natnum  : natnum 4) 
  (natnum_to_intnum : ∀ n, natnum n → intnum n) : intnum 4 :=
by
  sorry

end syllogism_correct_l189_189945


namespace equation_solutions_equivalence_l189_189842

theorem equation_solutions_equivalence {n k : ℕ} (hn : 1 < n) (hk : 1 < k) (hnk : n > k) :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^(n - k)) :=
by
  sorry

end equation_solutions_equivalence_l189_189842


namespace minimal_value_of_function_l189_189203

theorem minimal_value_of_function (x : ℝ) (hx : x > 1 / 2) :
  (x = 1 → (x^2 + 1) / x = 2) ∧
  (∀ y, (∀ z, z > 1 / 2 → y ≤ (z^2 + 1) / z) → y = 2) :=
by {
  sorry
}

end minimal_value_of_function_l189_189203


namespace numerology_eq_l189_189648

theorem numerology_eq : 2222 - 222 + 22 - 2 = 2020 :=
by
  sorry

end numerology_eq_l189_189648


namespace only_exprC_cannot_be_calculated_with_square_of_binomial_l189_189299

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial_l189_189299


namespace aerith_seat_l189_189320

-- Let the seats be numbered 1 through 8
-- Assigned seats for Aerith, Bob, Chebyshev, Descartes, Euler, Fermat, Gauss, and Hilbert
variables (a b c d e f g h : ℕ)

-- Define the conditions described in the problem
axiom Bob_assigned : b = 1
axiom Chebyshev_assigned : c = g + 2
axiom Descartes_assigned : d = f - 1
axiom Euler_assigned : e = h - 4
axiom Fermat_assigned : f = d + 5
axiom Gauss_assigned : g = e + 1
axiom Hilbert_assigned : h = a - 3

-- Provide the proof statement to find whose seat Aerith sits
theorem aerith_seat : a = c := sorry

end aerith_seat_l189_189320


namespace largest_number_value_l189_189008

theorem largest_number_value (x : ℕ) (h : 7 * x - 3 * x = 40) : 7 * x = 70 :=
by
  sorry

end largest_number_value_l189_189008


namespace distance_symmetric_line_eq_l189_189105

noncomputable def distance_from_point_to_line : ℝ :=
  let x0 := 2
  let y0 := -1
  let A := 2
  let B := 3
  let C := 0
  (|A * x0 + B * y0 + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_symmetric_line_eq : distance_from_point_to_line = 1 / (Real.sqrt 13) := by
  sorry

end distance_symmetric_line_eq_l189_189105


namespace no_odd_multiples_between_1500_and_3000_l189_189999

theorem no_odd_multiples_between_1500_and_3000 :
  ∀ n : ℤ, 1500 ≤ n → n ≤ 3000 → (18 ∣ n) → (24 ∣ n) → (36 ∣ n) → ¬(n % 2 = 1) :=
by
  -- The proof steps would go here, but we skip them according to the instructions.
  sorry

end no_odd_multiples_between_1500_and_3000_l189_189999


namespace johns_weekly_earnings_increase_l189_189286

def combined_percentage_increase (initial final : ℕ) : ℕ :=
  ((final - initial) * 100) / initial

theorem johns_weekly_earnings_increase :
  combined_percentage_increase 40 60 = 50 :=
by
  sorry

end johns_weekly_earnings_increase_l189_189286


namespace part_1_part_2a_part_2b_l189_189060

namespace InequalityProofs

-- Definitions extracted from the problem
def quadratic_function (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

-- Lean statement for Part 1
theorem part_1 (m : ℝ) : (∀ x : ℝ, quadratic_function m x ≥ -2) ↔ m ∈ Set.Ici (1 / 3) :=
sorry

-- Lean statement for Part 2, breaking into separate theorems for different ranges of m
theorem part_2a (m : ℝ) (h : m < -1) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic (-1 / m) ∪ Set.Ici 1)) :=
sorry

theorem part_2b (m : ℝ) (h : -1 < m ∧ m < 0) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic 1 ∪ Set.Ici (-1 / m))) :=
sorry

end InequalityProofs

end part_1_part_2a_part_2b_l189_189060


namespace points_on_curve_is_parabola_l189_189348

theorem points_on_curve_is_parabola (X Y : ℝ) (h : Real.sqrt X + Real.sqrt Y = 1) :
  ∃ a b c : ℝ, Y = a * X^2 + b * X + c :=
sorry

end points_on_curve_is_parabola_l189_189348


namespace square_area_increase_l189_189265

theorem square_area_increase (s : ℕ) (h : (s = 5) ∨ (s = 10) ∨ (s = 15)) :
  (1.35^2 - 1) * 100 = 82.25 :=
by
  sorry

end square_area_increase_l189_189265


namespace total_chairs_l189_189220

-- Define the conditions as constants
def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6
def dining_room_chairs : ℕ := 8
def outdoor_patio_chairs : ℕ := 12

-- State the goal to prove
theorem total_chairs : 
  living_room_chairs + kitchen_chairs + dining_room_chairs + outdoor_patio_chairs = 29 := 
by
  -- The proof is not required as per instructions
  sorry

end total_chairs_l189_189220


namespace solve_for_x_l189_189702

theorem solve_for_x (x : ℝ) : 9 * x^2 - 4 = 0 → (x = 2/3 ∨ x = -2/3) :=
by
  sorry

end solve_for_x_l189_189702


namespace union_of_sets_l189_189955

def A (x : ℤ) : Set ℤ := {x^2, 2*x - 1, -4}
def B (x : ℤ) : Set ℤ := {x - 5, 1 - x, 9}

theorem union_of_sets (x : ℤ) (hx : x = -3) (h_inter : A x ∩ B x = {9}) :
  A x ∪ B x = {-8, -4, 4, -7, 9} :=
by
  sorry

end union_of_sets_l189_189955


namespace scientific_notation_to_decimal_l189_189670

theorem scientific_notation_to_decimal :
  5.2 * 10^(-5) = 0.000052 :=
sorry

end scientific_notation_to_decimal_l189_189670


namespace infinite_coprime_pairs_divisibility_l189_189741

theorem infinite_coprime_pairs_divisibility :
  ∃ (S : ℕ → ℕ × ℕ), (∀ n, Nat.gcd (S n).1 (S n).2 = 1 ∧ (S n).1 ∣ (S n).2^2 - 5 ∧ (S n).2 ∣ (S n).1^2 - 5) ∧
  Function.Injective S :=
sorry

end infinite_coprime_pairs_divisibility_l189_189741


namespace smallest_sector_angle_l189_189630

-- Definitions and conditions identified in step a.

def a1 (d : ℕ) : ℕ := (48 - 14 * d) / 2

-- Proof statement
theorem smallest_sector_angle : ∀ d : ℕ, d ≥ 0 → d ≤ 3 → 15 * (a1 d + (a1 d + 14 * d)) = 720 → (a1 d = 3) :=
by
  sorry

end smallest_sector_angle_l189_189630


namespace num_males_in_group_l189_189325

-- Definitions based on the given conditions
def num_females (f : ℕ) : Prop := f = 16
def num_males_choose_malt (m_malt : ℕ) : Prop := m_malt = 6
def num_females_choose_malt (f_malt : ℕ) : Prop := f_malt = 8
def num_choose_malt (m_malt f_malt n_malt : ℕ) : Prop := n_malt = m_malt + f_malt
def num_choose_coke (c : ℕ) (n_malt : ℕ) : Prop := n_malt = 2 * c
def total_cheerleaders (t : ℕ) (n_malt c : ℕ) : Prop := t = n_malt + c
def num_males (m f t : ℕ) : Prop := m = t - f

theorem num_males_in_group
  (f m_malt f_malt n_malt c t m : ℕ)
  (hf : num_females f)
  (hmm : num_males_choose_malt m_malt)
  (hfm : num_females_choose_malt f_malt)
  (hmalt : num_choose_malt m_malt f_malt n_malt)
  (hc : num_choose_coke c n_malt)
  (ht : total_cheerleaders t n_malt c)
  (hm : num_males m f t) :
  m = 5 := 
sorry

end num_males_in_group_l189_189325


namespace josanna_next_test_score_l189_189135

theorem josanna_next_test_score :
  let scores := [75, 85, 65, 95, 70]
  let current_sum := scores.sum
  let current_average := current_sum / scores.length
  let desired_average := current_average + 10
  let new_test_count := scores.length + 1
  let desired_sum := desired_average * new_test_count
  let required_score := desired_sum - current_sum
  required_score = 138 :=
by
  sorry

end josanna_next_test_score_l189_189135


namespace lawn_unmowed_fraction_l189_189383

noncomputable def rate_mary : ℚ := 1 / 6
noncomputable def rate_tom : ℚ := 1 / 3

theorem lawn_unmowed_fraction :
  (1 : ℚ) - ((1 * rate_tom) + (2 * (rate_mary + rate_tom))) = 1 / 6 :=
by
  -- This part will be the actual proof which we are skipping
  sorry

end lawn_unmowed_fraction_l189_189383


namespace houses_with_garage_l189_189680

theorem houses_with_garage (P GP N : ℕ) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
    (total_houses : P + GP - GP + N = 65) : 
    P + 65 - P - GP + GP - N = 50 :=
by
  sorry

end houses_with_garage_l189_189680


namespace equilateral_triangle_AB_length_l189_189423

noncomputable def Q := 2
noncomputable def R := 3
noncomputable def S := 4

theorem equilateral_triangle_AB_length :
  ∀ (AB BC CA : ℝ), 
  AB = BC ∧ BC = CA ∧ (∃ P : ℝ × ℝ, (Q = 2) ∧ (R = 3) ∧ (S = 4)) →
  AB = 6 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_AB_length_l189_189423


namespace complete_the_square_l189_189572

theorem complete_the_square (x : ℝ) : x^2 - 2 * x - 1 = 0 -> (x - 1)^2 = 2 := by
  sorry

end complete_the_square_l189_189572


namespace maximize_binom_term_l189_189282

theorem maximize_binom_term :
  ∃ k, k ∈ Finset.range (207) ∧
  (∀ m ∈ Finset.range (207), (Nat.choose 206 k * (Real.sqrt 5)^k) ≥ (Nat.choose 206 m * (Real.sqrt 5)^m)) ∧ k = 143 :=
sorry

end maximize_binom_term_l189_189282


namespace intersection_A_B_l189_189927

def A := {x : ℝ | 2 < x ∧ x < 4}
def B := {x : ℝ | (x-1) * (x-3) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end intersection_A_B_l189_189927


namespace ratio_nine_years_ago_correct_l189_189289

-- Conditions
def C : ℕ := 24
def G : ℕ := C / 2

-- Question and expected answer
def ratio_nine_years_ago : ℕ := (C - 9) / (G - 9)

theorem ratio_nine_years_ago_correct : ratio_nine_years_ago = 5 := by
  sorry

end ratio_nine_years_ago_correct_l189_189289


namespace non_zero_const_c_l189_189834

theorem non_zero_const_c (a b c x1 x2 : ℝ) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) 
(h3 : (a - 1) * x1 ^ 2 + b * x1 + c = 0) 
(h4 : (a - 1) * x2 ^ 2 + b * x2 + c = 0)
(h5 : x1 * x2 = -1) 
(h6 : x1 ≠ x2) 
(h7 : x1 * x2 < 0): c ≠ 0 :=
sorry

end non_zero_const_c_l189_189834


namespace total_goals_correct_l189_189415

-- Define the number of goals scored by each team in each period
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := (1 / 2) * kickers_first_period_goals
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- Define the total goals scored by both teams
def total_goals : ℕ :=
  kickers_first_period_goals + 
  kickers_second_period_goals + 
  spiders_first_period_goals + 
  spiders_second_period_goals

-- State the theorem to be proved
theorem total_goals_correct : total_goals = 15 := by
  sorry

end total_goals_correct_l189_189415


namespace martha_saves_half_daily_allowance_l189_189096

theorem martha_saves_half_daily_allowance {f : ℚ} (h₁ : 12 > 0) (h₂ : (6 : ℚ) * 12 * f + (3 : ℚ) = 39) : f = 1 / 2 :=
by
  sorry

end martha_saves_half_daily_allowance_l189_189096


namespace pipe_fill_time_l189_189401

theorem pipe_fill_time (T : ℝ) 
  (h1 : ∃ T : ℝ, 0 < T) 
  (h2 : T + (1/2) > 0) 
  (h3 : ∃ leak_rate : ℝ, leak_rate = 1/10) 
  (h4 : ∃ pipe_rate : ℝ, pipe_rate = 1/T) 
  (h5 : ∃ effective_rate : ℝ, effective_rate = pipe_rate - leak_rate) 
  (h6 : effective_rate = 1 / (T + 1/2))  : 
  T = Real.sqrt 5 :=
  sorry

end pipe_fill_time_l189_189401


namespace find_slope_of_parallel_line_l189_189134

-- Define the condition that line1 is parallel to line2.
def lines_parallel (k : ℝ) : Prop :=
  k = -3

-- The theorem that proves the condition given.
theorem find_slope_of_parallel_line (k : ℝ) (h : lines_parallel k) : k = -3 :=
by
  exact h

end find_slope_of_parallel_line_l189_189134


namespace tetrahedron_volume_le_one_l189_189711

open Real

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := A
  let (x1, y1, z1) := B
  let (x2, y2, z2) := C
  let (x3, y3, z3) := D
  abs ((x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) -
       (x2 - x0) * ((y1 - y0) * (z3 - z0) - (y3 - y0) * (z1 - z0)) +
       (x3 - x0) * ((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0))) / 6

theorem tetrahedron_volume_le_one (A B C D : ℝ × ℝ × ℝ)
  (h1 : dist A B ≤ 2) (h2 : dist A C ≤ 2) (h3 : dist A D ≤ 2)
  (h4 : dist B C ≤ 2) (h5 : dist B D ≤ 2) (h6 : dist C D ≤ 2) :
  volume_tetrahedron A B C D ≤ 1 := by
  sorry

end tetrahedron_volume_le_one_l189_189711


namespace f_neg_l189_189027

variable (f : ℝ → ℝ)

-- Given condition that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The form of f for x ≥ 0
def f_pos (x : ℝ) (h : 0 ≤ x) : f x = -x^2 + 2 * x := sorry

-- Objective to prove f(x) for x < 0
theorem f_neg {x : ℝ} (h : x < 0) (hf_odd : odd_function f) (hf_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x) : f x = x^2 + 2 * x := 
by 
  sorry

end f_neg_l189_189027


namespace increased_numerator_value_l189_189853

theorem increased_numerator_value (x y a : ℝ) (h1 : x / y = 2 / 5) (h2 : (x + a) / (2 * y) = 1 / 3) (h3 : x + y = 5.25) : a = 1 :=
by
  -- skipped proof: sorry
  sorry

end increased_numerator_value_l189_189853


namespace problem_solution_l189_189263

theorem problem_solution (m : ℝ) (h : (m - 2023)^2 + (2024 - m)^2 = 2025) :
  (m - 2023) * (2024 - m) = -1012 :=
sorry

end problem_solution_l189_189263


namespace not_both_perfect_squares_l189_189137

theorem not_both_perfect_squares (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ a b : ℕ, (n+1) * 2^n = a^2 ∧ (n+3) * 2^(n + 2) = b^2) :=
sorry

end not_both_perfect_squares_l189_189137


namespace find_c_for_Q_l189_189074

noncomputable def Q (c : ℚ) (x : ℚ) : ℚ := x^3 + 3*x^2 + c*x + 8

theorem find_c_for_Q (c : ℚ) : 
  (Q c 3 = 0) ↔ (c = -62 / 3) := by
  sorry

end find_c_for_Q_l189_189074


namespace unique_f_l189_189619

def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 10^10 }

noncomputable def f : ℕ → ℕ := sorry

axiom f_cond (x : ℕ) (hx : x ∈ S) :
  f (x + 1) % (10^10) = (f (f x) + 1) % (10^10)

axiom f_boundary :
  f (10^10 + 1) % (10^10) = f 1

theorem unique_f (x : ℕ) (hx : x ∈ S) :
  f x % (10^10) = x % (10^10) :=
sorry

end unique_f_l189_189619


namespace twelve_sided_figure_area_is_13_cm2_l189_189240

def twelve_sided_figure_area_cm2 : ℝ :=
  let unit_square := 1
  let full_squares := 9
  let triangle_pairs := 4
  full_squares * unit_square + triangle_pairs * unit_square

theorem twelve_sided_figure_area_is_13_cm2 :
  twelve_sided_figure_area_cm2 = 13 := 
by
  sorry

end twelve_sided_figure_area_is_13_cm2_l189_189240


namespace solution_l189_189778

theorem solution (x y : ℝ) (h₁ : x + 3 * y = -1) (h₂ : x - 3 * y = 5) : x^2 - 9 * y^2 = -5 := 
by
  sorry

end solution_l189_189778


namespace bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l189_189337

variables {a b c : ℝ}
-- Given conditions from Vieta's formulas for the polynomial x^3 - 20x^2 + 22
axiom vieta1 : a + b + c = 20
axiom vieta2 : a * b + b * c + c * a = 0
axiom vieta3 : a * b * c = -22

theorem bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3 (a b c : ℝ)
  (h1 : a + b + c = 20)
  (h2 : a * b + b * c + c * a = 0)
  (h3 : a * b * c = -22) :
  (b * c / a^2) + (a * c / b^2) + (a * b / c^2) = 3 := 
  sorry

end bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l189_189337


namespace common_difference_and_first_three_terms_l189_189776

-- Given condition that for any n, the sum of the first n terms of an arithmetic progression is equal to 5n^2.
def arithmetic_sum_property (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 5 * n ^ 2

-- Define the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n-1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a1 d n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d)/2

-- Conditions and prove that common difference d is 10 and the first three terms are 5, 15, and 25
theorem common_difference_and_first_three_terms :
  (∃ (a1 d : ℕ), arithmetic_sum_property (sum_first_n_terms a1 d) ∧ d = 10 ∧ nth_term a1 d 1 = 5 ∧ nth_term a1 d 2 = 15 ∧ nth_term a1 d 3  = 25) :=
sorry

end common_difference_and_first_three_terms_l189_189776


namespace find_g1_l189_189170

variables {f g : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_g1 (hf : odd_function f)
                (hg : even_function g)
                (h1 : f (-1) + g 1 = 2)
                (h2 : f 1 + g (-1) = 4) :
                g 1 = 3 :=
sorry

end find_g1_l189_189170


namespace probability_math_majors_consecutive_l189_189899

theorem probability_math_majors_consecutive :
  (5 / 12) * (4 / 11) * (3 / 10) * (2 / 9) * (1 / 8) * 12 = 1 / 66 :=
by
  sorry

end probability_math_majors_consecutive_l189_189899


namespace find_A_l189_189231

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B - 2

theorem find_A (A : ℝ) : spadesuit A 7 = 40 ↔ A = 21 / 4 :=
by
  sorry

end find_A_l189_189231


namespace sum_partition_ominous_years_l189_189211

def is_ominous (n : ℕ) : Prop :=
  n = 1 ∨ Nat.Prime n

theorem sum_partition_ominous_years :
  ∀ n : ℕ, (¬ ∃ (A B : Finset ℕ), A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id ∧ A.card = B.card)) ↔ is_ominous n := 
sorry

end sum_partition_ominous_years_l189_189211


namespace rohan_house_rent_percentage_l189_189307

variable (salary savings food entertainment conveyance : ℕ)
variable (spend_on_house : ℚ)

-- Given conditions
axiom h1 : salary = 5000
axiom h2 : savings = 1000
axiom h3 : food = 40
axiom h4 : entertainment = 10
axiom h5 : conveyance = 10

-- Define savings percentage
def savings_percentage (salary savings : ℕ) : ℚ := (savings : ℚ) / salary * 100

-- Define percentage equation
def total_percentage (food entertainment conveyance spend_on_house savings_percentage : ℚ) : ℚ :=
  food + spend_on_house + entertainment + conveyance + savings_percentage

-- Prove that house rent percentage is 20%
theorem rohan_house_rent_percentage : 
  food = 40 → entertainment = 10 → conveyance = 10 → salary = 5000 → savings = 1000 → 
  total_percentage 40 10 10 spend_on_house (savings_percentage 5000 1000) = 100 →
  spend_on_house = 20 := by
  intros
  sorry

end rohan_house_rent_percentage_l189_189307


namespace max_min_x2_minus_xy_plus_y2_l189_189193

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2_l189_189193


namespace fraction_of_cookies_l189_189984

-- Given conditions
variables 
  (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3)

-- Proof statement
theorem fraction_of_cookies (Millie_cookies Mike_cookies Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3) : 
  (Frank_cookies / Mike_cookies : ℚ) = 1 / 4 :=
by
  sorry

end fraction_of_cookies_l189_189984


namespace difference_of_fractions_l189_189473

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h₁ : a = 7000) (h₂ : b = 1/10) :
  (a * b - a * (0.1 / 100)) = 693 :=
by 
  sorry

end difference_of_fractions_l189_189473


namespace sum_greater_than_two_l189_189671

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l189_189671


namespace possible_values_of_x_l189_189695

theorem possible_values_of_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) 
    (h1 : x + 1 / z = 15) (h2 : z + 1 / x = 9 / 20) :
    x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by
  sorry

end possible_values_of_x_l189_189695


namespace length_of_segment_XY_l189_189722

noncomputable def rectangle_length (A B C D : ℝ) (BX DY : ℝ) : ℝ :=
  2 * BX + DY

theorem length_of_segment_XY (A B C D : ℝ) (BX DY : ℝ) (h1 : C = 2 * B) (h2 : BX = 4) (h3 : DY = 10) :
  rectangle_length A B C D BX DY = 13 :=
by
  rw [rectangle_length, h2, h3]
  sorry

end length_of_segment_XY_l189_189722


namespace smaller_circle_radius_l189_189367

theorem smaller_circle_radius
  (radius_largest : ℝ)
  (h1 : radius_largest = 10)
  (aligned_circles : ℝ)
  (h2 : 4 * aligned_circles = 2 * radius_largest) :
  aligned_circles / 2 = 2.5 :=
by
  sorry

end smaller_circle_radius_l189_189367


namespace arithmetic_sqrt_of_nine_l189_189079

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l189_189079


namespace sandwich_cost_is_5_l189_189912

-- We define the variables and conditions first
def total_people := 4
def sandwiches := 4
def fruit_salads := 4
def sodas := 8
def snack_bags := 3

def fruit_salad_cost_per_unit := 3
def soda_cost_per_unit := 2
def snack_bag_cost_per_unit := 4
def total_cost := 60

-- We now define the calculations based on the given conditions
def total_fruit_salad_cost := fruit_salads * fruit_salad_cost_per_unit
def total_soda_cost := sodas * soda_cost_per_unit
def total_snack_bag_cost := snack_bags * snack_bag_cost_per_unit
def other_items_cost := total_fruit_salad_cost + total_soda_cost + total_snack_bag_cost
def remaining_budget := total_cost - other_items_cost
def sandwich_cost := remaining_budget / sandwiches

-- The final proof problem statement in Lean 4
theorem sandwich_cost_is_5 : sandwich_cost = 5 := by
  sorry

end sandwich_cost_is_5_l189_189912


namespace remaining_three_digit_numbers_l189_189528

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid_number (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧ B ≠ 0 ∧ n = 100 * A + 10 * B + A

def count_valid_three_digit_numbers : ℕ :=
  let total_numbers := 900
  let invalid_numbers := 10 * 9
  total_numbers - invalid_numbers

theorem remaining_three_digit_numbers : count_valid_three_digit_numbers = 810 := by
  sorry

end remaining_three_digit_numbers_l189_189528


namespace avg_xy_36_l189_189275

-- Given condition: The average of the numbers 2, 6, 10, x, and y is 18
def avg_condition (x y : ℝ) : Prop :=
  (2 + 6 + 10 + x + y) / 5 = 18

-- Goal: To prove that the average of x and y is 36
theorem avg_xy_36 (x y : ℝ) (h : avg_condition x y) : (x + y) / 2 = 36 :=
by
  sorry

end avg_xy_36_l189_189275


namespace maxwell_walking_speed_l189_189094

-- Define Maxwell's walking speed
def Maxwell_speed (v : ℕ) : Prop :=
  ∀ t1 t2 : ℕ, t1 = 10 → t2 = 9 →
  ∀ d1 d2 : ℕ, d1 = 10 * v → d2 = 6 * t2 →
  ∀ d_total : ℕ, d_total = 94 →
  d1 + d2 = d_total

theorem maxwell_walking_speed : Maxwell_speed 4 :=
by
  sorry

end maxwell_walking_speed_l189_189094


namespace cubic_coefficient_relationship_l189_189460

theorem cubic_coefficient_relationship (a b c p q r : ℝ)
    (h1 : ∀ s1 s2 s3: ℝ, s1 + s2 + s3 = -a ∧ s1 * s2 + s2 * s3 + s3 * s1 = b ∧ s1 * s2 * s3 = -c)
    (h2 : ∀ s1 s2 s3: ℝ, s1^2 + s2^2 + s3^2 = -p ∧ s1^2 * s2^2 + s2^2 * s3^2 + s3^2 * s1^2 = q ∧ s1^2 * s2^2 * s3^2 = r) :
    p = a^2 - 2 * b ∧ q = b^2 + 2 * a * c ∧ r = c^2 :=
by
  sorry

end cubic_coefficient_relationship_l189_189460


namespace greatest_visible_unit_cubes_from_single_point_l189_189866

-- Define the size of the cube
def cube_size : ℕ := 9

-- The total number of unit cubes in the 9x9x9 cube
def total_unit_cubes (n : ℕ) : ℕ := n^3

-- The greatest number of unit cubes visible from a single point
def visible_unit_cubes (n : ℕ) : ℕ := 3 * n^2 - 3 * (n - 1) + 1

-- The given cube size is 9
def given_cube_size : ℕ := cube_size

-- The correct answer for the greatest number of visible unit cubes from a single point
def correct_visible_cubes : ℕ := 220

-- Theorem stating the visibility calculation for a 9x9x9 cube
theorem greatest_visible_unit_cubes_from_single_point :
  visible_unit_cubes cube_size = correct_visible_cubes := by
  sorry

end greatest_visible_unit_cubes_from_single_point_l189_189866


namespace unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l189_189252

theorem unique_solution_x_ln3_plus_x_ln4_eq_x_ln5 :
  ∃! x : ℝ, 0 < x ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) := sorry

end unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l189_189252


namespace inequality_k_distance_comparison_l189_189118

theorem inequality_k (k : ℝ) (x : ℝ) : 
  -3 < k ∧ k ≤ 0 → 2 * k * x^2 + k * x - 3/8 < 0 := sorry

theorem distance_comparison (a b : ℝ) (hab : a ≠ b) : 
  (abs ((a^2 + b^2) / 2 - (a + b)^2 / 4) > abs (a * b - (a + b)^2 / 4)) := sorry

end inequality_k_distance_comparison_l189_189118


namespace product_of_integers_l189_189883

theorem product_of_integers (a b : ℚ) (h1 : a / b = 12) (h2 : a + b = 144) :
  a * b = 248832 / 169 := 
sorry

end product_of_integers_l189_189883


namespace length_PQ_eq_b_l189_189128

open Real

variables {a b : ℝ} (h : a > b) (p : ℝ × ℝ) (h₁ : (p.fst / a) ^ 2 + (p.snd / b) ^ 2 = 1)
variables (F₁ F₂ : ℝ × ℝ) (P Q : ℝ × ℝ)
variable (Q_on_segment : Q.1 = (F₁.1 + F₂.1) / 2)
variable (equal_inradii : inradius (triangle P Q F₁) = inradius (triangle P Q F₂))

theorem length_PQ_eq_b : dist P Q = b :=
by
  sorry

end length_PQ_eq_b_l189_189128


namespace tim_total_spent_l189_189608

variable (lunch_cost : ℝ)
variable (tip_percentage : ℝ)
variable (total_spent : ℝ)

theorem tim_total_spent (h_lunch_cost : lunch_cost = 60.80)
                        (h_tip_percentage : tip_percentage = 0.20)
                        (h_total_spent : total_spent = lunch_cost + (tip_percentage * lunch_cost)) :
                        total_spent = 72.96 :=
sorry

end tim_total_spent_l189_189608


namespace factorize_problem_1_factorize_problem_2_l189_189835

-- Problem 1 Statement
theorem factorize_problem_1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := 
sorry

-- Problem 2 Statement
theorem factorize_problem_2 (x y : ℝ) : (x - y)^2 + 4 * (x * y) = (x + y)^2 := 
sorry

end factorize_problem_1_factorize_problem_2_l189_189835


namespace price_of_feed_corn_l189_189705

theorem price_of_feed_corn :
  ∀ (num_sheep : ℕ) (num_cows : ℕ) (grass_per_cow : ℕ) (grass_per_sheep : ℕ)
    (feed_corn_duration_cow : ℕ) (feed_corn_duration_sheep : ℕ)
    (total_grass : ℕ) (total_expenditure : ℕ) (months_in_year : ℕ),
  num_sheep = 8 →
  num_cows = 5 →
  grass_per_cow = 2 →
  grass_per_sheep = 1 →
  feed_corn_duration_cow = 1 →
  feed_corn_duration_sheep = 2 →
  total_grass = 144 →
  total_expenditure = 360 →
  months_in_year = 12 →
  ((total_expenditure : ℝ) / (((num_cows * feed_corn_duration_cow * 4) + (num_sheep * (4 / feed_corn_duration_sheep))) : ℝ)) = 10 :=
by
  intros
  sorry

end price_of_feed_corn_l189_189705


namespace fraction_value_l189_189089

theorem fraction_value :
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 :=
by
  -- simplified proof omitted
  sorry

end fraction_value_l189_189089


namespace solve_inequality_l189_189690

theorem solve_inequality (x : ℝ) : 1 + 2 * (x - 1) ≤ 3 → x ≤ 2 :=
by
  sorry

end solve_inequality_l189_189690


namespace questionnaires_drawn_from_D_l189_189862

theorem questionnaires_drawn_from_D (a1 a2 a3 a4 total sample_b sample_total sample_d : ℕ)
  (h1 : a2 - a1 = a3 - a2)
  (h2 : a3 - a2 = a4 - a3)
  (h3 : a1 + a2 + a3 + a4 = total)
  (h4 : total = 1000)
  (h5 : sample_b = 30)
  (h6 : a2 = 200)
  (h7 : sample_total = 150)
  (h8 : sample_d * total = sample_total * a4) :
  sample_d = 60 :=
by sorry

end questionnaires_drawn_from_D_l189_189862


namespace largest_n_l189_189617

-- Define the condition that n, x, y, z are positive integers
def conditions (n x y z : ℕ) := (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < n) 

-- Formulate the main theorem
theorem largest_n (x y z : ℕ) : 
  conditions 8 x y z →
  8^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 10 :=
by 
  sorry

end largest_n_l189_189617


namespace mira_jogs_hours_each_morning_l189_189723

theorem mira_jogs_hours_each_morning 
  (h : ℝ) -- number of hours Mira jogs each morning
  (speed : ℝ) -- Mira's jogging speed in miles per hour
  (days : ℝ) -- number of days Mira jogs
  (total_distance : ℝ) -- total distance Mira jogs

  (H1 : speed = 5) 
  (H2 : days = 5) 
  (H3 : total_distance = 50) 
  (H4 : total_distance = speed * h * days) :

  h = 2 :=
by
  sorry

end mira_jogs_hours_each_morning_l189_189723


namespace find_initial_books_l189_189625

/-- The number of books the class initially obtained from the library --/
def initial_books : ℕ := sorry

/-- The number of books added later --/
def books_added_later : ℕ := 23

/-- The total number of books the class has --/
def total_books : ℕ := 77

theorem find_initial_books : initial_books + books_added_later = total_books → initial_books = 54 :=
by
  intros h
  sorry

end find_initial_books_l189_189625


namespace option_D_correct_l189_189643

variables (Line : Type) (Plane : Type)
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (perpendicular_planes : Plane → Plane → Prop)

theorem option_D_correct (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicular_planes α β :=
sorry

end option_D_correct_l189_189643


namespace train_passing_time_l189_189146

theorem train_passing_time 
  (length_of_train : ℕ) 
  (length_of_platform : ℕ) 
  (time_to_pass_pole : ℕ) 
  (speed_of_train : ℕ) 
  (combined_length : ℕ) 
  (time_to_pass_platform : ℕ) 
  (h1 : length_of_train = 240) 
  (h2 : length_of_platform = 650)
  (h3 : time_to_pass_pole = 24)
  (h4 : speed_of_train = length_of_train / time_to_pass_pole)
  (h5 : combined_length = length_of_train + length_of_platform)
  (h6 : time_to_pass_platform = combined_length / speed_of_train) : 
  time_to_pass_platform = 89 :=
sorry

end train_passing_time_l189_189146


namespace parity_of_f_min_value_of_f_min_value_of_f_l189_189477

open Real

def f (a x : ℝ) := x^2 + abs (x - a) + 1

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f 0 x = f 0 (-x)) ∧ (∀ x : ℝ, f a x ≠ f a (-x) ∧ f a x ≠ -f a x) ↔ a = 0 :=
by sorry

theorem min_value_of_f (a : ℝ) (h : a ≤ -1/2) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a (-1/2) :=
by sorry

theorem min_value_of_f' (a : ℝ) (h : -1/2 < a) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a a :=
by sorry

end parity_of_f_min_value_of_f_min_value_of_f_l189_189477


namespace quotient_when_divided_by_5_l189_189041

theorem quotient_when_divided_by_5 (N : ℤ) (k : ℤ) (Q : ℤ) 
  (h1 : N = 5 * Q) 
  (h2 : N % 4 = 2) : 
  Q = 2 := 
sorry

end quotient_when_divided_by_5_l189_189041


namespace average_consecutive_from_c_l189_189449

variable (a : ℕ) (c : ℕ)

-- Condition: c is the average of seven consecutive integers starting from a
axiom h1 : c = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7

-- Target statement: Prove the average of seven consecutive integers starting from c is a + 6
theorem average_consecutive_from_c : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 6 :=
by
  sorry

end average_consecutive_from_c_l189_189449


namespace quadratic_points_relation_l189_189526

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end quadratic_points_relation_l189_189526


namespace part1_part2_l189_189884

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (b : ℝ) := 0.5 * x^2 - b * x
noncomputable def h (x : ℝ) (b : ℝ) := f x + g x b

theorem part1 (b : ℝ) :
  (∃ (tangent_point : ℝ),
    tangent_point = 1 ∧
    deriv f tangent_point = 1 ∧
    f tangent_point = 0 ∧
    ∃ (y_tangent : ℝ → ℝ), (∀ (x : ℝ), y_tangent x = x - 1) ∧
    ∃ (tangent_for_g : ℝ), (∀ (x : ℝ), y_tangent x = g x b)
  ) → false :=
sorry 

theorem part2 (b : ℝ) :
  ¬ (∀ (x : ℝ) (hx : 0 < x), deriv (h x) b = 0 → deriv (h x) b < 0) →
  2 < b :=
sorry

end part1_part2_l189_189884


namespace rationalize_sqrt_5_div_18_l189_189387

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l189_189387


namespace calculate_distance_l189_189488

def velocity (t : ℝ) : ℝ := 3 * t^2 + t

theorem calculate_distance : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := 
by
  sorry

end calculate_distance_l189_189488


namespace smallest_positive_integer_l189_189497

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l189_189497


namespace bricks_for_wall_l189_189188

theorem bricks_for_wall
  (wall_length : ℕ) (wall_height : ℕ) (wall_width : ℕ)
  (brick_length : ℕ) (brick_height : ℕ) (brick_width : ℕ)
  (L_eq : wall_length = 600) (H_eq : wall_height = 400) (W_eq : wall_width = 2050)
  (l_eq : brick_length = 30) (h_eq : brick_height = 12) (w_eq : brick_width = 10)
  : (wall_length * wall_height * wall_width) / (brick_length * brick_height * brick_width) = 136667 :=
by
  sorry

end bricks_for_wall_l189_189188


namespace train_crossing_time_correct_l189_189521

noncomputable def train_crossing_time (speed_kmph : ℕ) (length_m : ℕ) (train_dir_opposite : Bool) : ℕ :=
  if train_dir_opposite then
    let speed_mps := speed_kmph * 1000 / 3600
    let relative_speed := speed_mps + speed_mps
    let total_distance := length_m + length_m
    total_distance / relative_speed
  else 0

theorem train_crossing_time_correct :
  train_crossing_time 54 120 true = 8 :=
by
  sorry

end train_crossing_time_correct_l189_189521


namespace find_number_l189_189694

-- Assume the necessary definitions and conditions
variable (x : ℝ)

-- Sixty-five percent of the number is 21 less than four-fifths of the number
def condition := 0.65 * x = 0.8 * x - 21

-- Final proof goal: We need to prove that the number x is 140
theorem find_number (h : condition x) : x = 140 := by
  sorry

end find_number_l189_189694


namespace father_l189_189700

-- Define the variables
variables (F S : ℕ)

-- Define the conditions
def condition1 : Prop := F = 4 * S
def condition2 : Prop := F + 20 = 2 * (S + 20)
def condition3 : Prop := S = 10

-- Statement of the problem
theorem father's_age (h1 : condition1 F S) (h2 : condition2 F S) (h3 : condition3 S) : F = 40 :=
by sorry

end father_l189_189700


namespace max_price_of_most_expensive_product_l189_189737

noncomputable def greatest_possible_price
  (num_products : ℕ)
  (avg_price : ℕ)
  (min_price : ℕ)
  (mid_price : ℕ)
  (higher_price_count : ℕ)
  (total_retail_price : ℕ)
  (least_expensive_total_price : ℕ)
  (remaining_price : ℕ)
  (less_expensive_total_price : ℕ) : ℕ :=
  total_retail_price - least_expensive_total_price - less_expensive_total_price

theorem max_price_of_most_expensive_product :
  greatest_possible_price 20 1200 400 1000 10 (20 * 1200) (10 * 400) (20 * 1200 - 10 * 400) (9 * 1000) = 11000 :=
by
  sorry

end max_price_of_most_expensive_product_l189_189737


namespace find_m_probability_l189_189506

theorem find_m_probability (m : ℝ) (ξ : ℕ → ℝ) :
  (ξ 1 = m * (2/3)) ∧ (ξ 2 = m * (2/3)^2) ∧ (ξ 3 = m * (2/3)^3) ∧ 
  (ξ 1 + ξ 2 + ξ 3 = 1) → 
  m = 27 / 38 := 
sorry

end find_m_probability_l189_189506


namespace area_of_triangle_AMN_l189_189786

theorem area_of_triangle_AMN
  (α : ℝ) -- Angle at vertex A
  (S : ℝ) -- Area of triangle ABC
  (area_AMN_eq : ∀ (α : ℝ) (S : ℝ), ∃ (area_AMN : ℝ), area_AMN = S * (Real.cos α)^2) :
  ∃ area_AMN, area_AMN = S * (Real.cos α)^2 := by
  sorry

end area_of_triangle_AMN_l189_189786


namespace minimum_n_divisible_20_l189_189600

theorem minimum_n_divisible_20 :
  ∃ (n : ℕ), (∀ (l : List ℕ), l.length = n → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0) ∧ 
  (∀ m, m < n → ¬(∀ (l : List ℕ), l.length = m → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0)) := 
⟨9, 
  by sorry, 
  by sorry⟩

end minimum_n_divisible_20_l189_189600


namespace rectangle_k_value_l189_189889

theorem rectangle_k_value (a k : ℝ) (h1 : k > 0) (h2 : 2 * (3 * a + a) = k) (h3 : 3 * a^2 = k) : k = 64 / 3 :=
by
  sorry

end rectangle_k_value_l189_189889


namespace roots_magnitude_order_l189_189278

theorem roots_magnitude_order (m : ℝ) (a b c d : ℝ)
  (h1 : m > 0)
  (h2 : a ^ 2 - m * a - 1 = 0)
  (h3 : b ^ 2 - m * b - 1 = 0)
  (h4 : c ^ 2 + m * c - 1 = 0)
  (h5 : d ^ 2 + m * d - 1 = 0)
  (ha_pos : a > 0) (hb_neg : b < 0)
  (hc_pos : c > 0) (hd_neg : d < 0) :
  |a| > |c| ∧ |c| > |b| ∧ |b| > |d| :=
sorry

end roots_magnitude_order_l189_189278


namespace initial_children_on_bus_l189_189426

-- Define the conditions
variables (x : ℕ)

-- Define the problem statement
theorem initial_children_on_bus (h : x + 7 = 25) : x = 18 :=
sorry

end initial_children_on_bus_l189_189426


namespace geometric_sequence_sum_l189_189732

open Nat

noncomputable def geometric_sum (a q n : ℕ) : ℕ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (S : ℕ → ℕ) (q a₁ : ℕ)
  (h_q: q = 2)
  (h_S5: S 5 = 1)
  (h_S: ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) :
  S 10 = 33 :=
by
  sorry

end geometric_sequence_sum_l189_189732


namespace greatest_integer_x_l189_189885

theorem greatest_integer_x (x : ℤ) : (5 - 4 * x > 17) → x ≤ -4 :=
by
  sorry

end greatest_integer_x_l189_189885


namespace Mina_age_is_10_l189_189838

-- Define the conditions as Lean definitions
variable (S : ℕ)

def Minho_age := 3 * S
def Mina_age := 2 * S - 2

-- State the main problem as a theorem
theorem Mina_age_is_10 (h_sum : S + Minho_age S + Mina_age S = 34) : Mina_age S = 10 :=
by
  sorry

end Mina_age_is_10_l189_189838


namespace inequality_proof_l189_189086

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  a^a * b^b + a^b * b^a ≤ 1 :=
  sorry

end inequality_proof_l189_189086


namespace tangent_line_sin_at_pi_l189_189047

theorem tangent_line_sin_at_pi :
  ∀ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin x) → ∀ x y, (x, y) = (Real.pi, 0) → 
    ∃ (m : ℝ) (b : ℝ), (∀ x, y = m * x + b) ∧ (m = -1) ∧ (b = Real.pi) :=
by
  sorry

end tangent_line_sin_at_pi_l189_189047


namespace trains_meet_480_km_away_l189_189019

-- Define the conditions
def bombay_express_speed : ℕ := 60 -- speed in km/h
def rajdhani_express_speed : ℕ := 80 -- speed in km/h
def bombay_express_start_time : ℕ := 1430 -- 14:30 in 24-hour format
def rajdhani_express_start_time : ℕ := 1630 -- 16:30 in 24-hour format

-- Define the function to calculate the meeting point distance
noncomputable def meeting_distance (bombay_speed rajdhani_speed : ℕ) (bombay_start rajdhani_start : ℕ) : ℕ :=
  let t := 6 -- time taken for Rajdhani to catch up in hours, derived from the solution
  rajdhani_speed * t

-- The statement we need to prove:
theorem trains_meet_480_km_away :
  meeting_distance bombay_express_speed rajdhani_express_speed bombay_express_start_time rajdhani_express_start_time = 480 := by
  sorry

end trains_meet_480_km_away_l189_189019


namespace trapezium_division_l189_189512

theorem trapezium_division (h : ℝ) (m n : ℕ) (h_pos : 0 < h) 
  (areas_equal : 4 / (3 * ↑m) = 7 / (6 * ↑n)) :
  m + n = 15 := by
  sorry

end trapezium_division_l189_189512


namespace gcd_polynomials_l189_189807

noncomputable def b : ℤ := sorry -- since b is given as an odd multiple of 997

theorem gcd_polynomials (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 :=
sorry

end gcd_polynomials_l189_189807


namespace find_c_l189_189087

/-- Seven unit squares are arranged in a row in the coordinate plane, 
with the lower left corner of the first square at the origin. 
A line extending from (c,0) to (4,4) divides the entire region 
into two regions of equal area. What is the value of c?
-/
theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 7 ∧ y = (4 / (4 - c)) * (x - c)) : c = 2.25 :=
sorry

end find_c_l189_189087


namespace angle_B_is_30_degrees_l189_189914

variable (a b : ℝ)
variable (A B : ℝ)

axiom a_value : a = 2 * Real.sqrt 3
axiom b_value : b = Real.sqrt 6
axiom A_value : A = Real.pi / 4

theorem angle_B_is_30_degrees (h1 : a = 2 * Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = Real.pi / 4) : B = Real.pi / 6 :=
  sorry

end angle_B_is_30_degrees_l189_189914


namespace optimal_position_station_l189_189298

-- Definitions for the conditions
def num_buildings := 5
def building_workers (k : ℕ) : ℕ := if k ≤ 5 then k else 0
def distance_between_buildings := 50

-- Function to calculate the total walking distance
noncomputable def total_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

-- Theorem statement
theorem optimal_position_station :
  ∃ x : ℝ, (∀ y : ℝ, total_distance x ≤ total_distance y) ∧ x = 150 :=
by
  sorry

end optimal_position_station_l189_189298


namespace lilly_fish_l189_189606

-- Define the conditions
def total_fish : ℕ := 18
def rosy_fish : ℕ := 8

-- Statement: Prove that Lilly has 10 fish
theorem lilly_fish (h1 : total_fish = 18) (h2 : rosy_fish = 8) :
  total_fish - rosy_fish = 10 :=
by sorry

end lilly_fish_l189_189606


namespace smallest_composite_no_prime_factors_lt_20_l189_189335

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n : ℕ, (n > 1 ∧ ¬ Prime n ∧ (∀ p : ℕ, Prime p → p < 20 → p ∣ n → False)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_lt_20_l189_189335


namespace evaluate_expression_l189_189989

theorem evaluate_expression : 3 + (-3)^2 = 12 := by
  sorry

end evaluate_expression_l189_189989


namespace vec_subtraction_l189_189294

-- Definitions
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Condition: a is parallel to b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Main theorem
theorem vec_subtraction (m : ℝ) (h : are_parallel a (b m)) :
  2 • a - b m = (4, -8) :=
sorry

end vec_subtraction_l189_189294


namespace find_S_9_l189_189035

-- Conditions
def aₙ (n : ℕ) : ℕ := sorry  -- arithmetic sequence

def Sₙ (n : ℕ) : ℕ := sorry  -- sum of the first n terms of the sequence

axiom condition_1 : 2 * aₙ 8 = 6 + aₙ 11

-- Proof goal
theorem find_S_9 : Sₙ 9 = 54 :=
sorry

end find_S_9_l189_189035


namespace initially_calculated_average_height_l189_189077

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l189_189077


namespace complement_intersection_l189_189290

open Set -- Open the Set namespace

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2})
variable (B : Set ℝ := {x | x ≤ -1 ∨ x > 2})

theorem complement_intersection :
  (U \ B) ∩ A = {x | x = 0 ∨ x = 1 ∨ x = 2} :=
by
  sorry -- Proof not required as per the instructions

end complement_intersection_l189_189290


namespace trip_early_movie_savings_l189_189925

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l189_189925


namespace carrots_weight_l189_189344

-- Let the weight of the carrots be denoted by C (in kg).
variables (C : ℕ)

-- Conditions:
-- The merchant installed 13 kg of zucchini and 8 kg of broccoli.
-- He sold only half of the total, which amounted to 18 kg, so the total weight was 36 kg.
def conditions := (C + 13 + 8 = 36)

-- Prove that the weight of the carrots installed is 15 kg.
theorem carrots_weight (H : C + 13 + 8 = 36) : C = 15 :=
by {
  sorry -- proof to be filled in
}

end carrots_weight_l189_189344


namespace makeup_palette_cost_l189_189962

variable (lipstick_cost : ℝ := 2.5)
variable (num_lipsticks : ℕ := 4)
variable (hair_color_cost : ℝ := 4)
variable (num_boxes_hair_color : ℕ := 3)
variable (total_cost : ℝ := 67)
variable (num_palettes : ℕ := 3)

theorem makeup_palette_cost :
  (total_cost - (num_lipsticks * lipstick_cost + num_boxes_hair_color * hair_color_cost)) / num_palettes = 15 := 
by
  sorry

end makeup_palette_cost_l189_189962


namespace Sandy_fingernails_reach_world_record_in_20_years_l189_189701

-- Definitions for the conditions of the problem
def world_record_len : ℝ := 26
def current_len : ℝ := 2
def growth_rate : ℝ := 0.1

-- Proof goal
theorem Sandy_fingernails_reach_world_record_in_20_years :
  (world_record_len - current_len) / growth_rate / 12 = 20 :=
by
  sorry

end Sandy_fingernails_reach_world_record_in_20_years_l189_189701


namespace cards_problem_l189_189285

-- Define the conditions and goal
theorem cards_problem 
    (L R : ℕ) 
    (h1 : L + 6 = 3 * (R - 6))
    (h2 : R + 2 = 2 * (L - 2)) : 
    L = 66 := 
by 
  -- proof goes here
  sorry

end cards_problem_l189_189285


namespace population_hypothetical_town_l189_189284

theorem population_hypothetical_town :
  ∃ (a b c : ℕ), a^2 + 150 = b^2 + 1 ∧ b^2 + 1 + 150 = c^2 ∧ a^2 = 5476 :=
by {
  sorry
}

end population_hypothetical_town_l189_189284


namespace kim_earrings_l189_189937

-- Define the number of pairs of earrings on the first day E as a variable
variable (E : ℕ)

-- Define the total number of gumballs Kim receives based on the earrings she brings each day
def total_gumballs_received (E : ℕ) : ℕ :=
  9 * E + 9 * 2 * E + 9 * (2 * E - 1)

-- Define the total number of gumballs Kim eats in 42 days
def total_gumballs_eaten : ℕ :=
  3 * 42

-- Define the statement to be proved
theorem kim_earrings : 
  total_gumballs_received E = total_gumballs_eaten + 9 → E = 3 :=
by sorry

end kim_earrings_l189_189937


namespace cube_strictly_increasing_l189_189364

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l189_189364


namespace both_false_of_not_or_l189_189196

-- Define propositions p and q
variables (p q : Prop)

-- The condition given: ¬(p ∨ q)
theorem both_false_of_not_or (h : ¬(p ∨ q)) : ¬ p ∧ ¬ q :=
by {
  sorry
}

end both_false_of_not_or_l189_189196


namespace geometric_sequence_a9_value_l189_189389

theorem geometric_sequence_a9_value {a : ℕ → ℝ} (q a1 : ℝ) 
  (h_geom : ∀ n, a n = a1 * q ^ n)
  (h_a3 : a 3 = 2)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (h_sum : S 12 = 4 * S 6) : a 9 = 2 := 
by 
  sorry

end geometric_sequence_a9_value_l189_189389


namespace combined_work_time_l189_189992

theorem combined_work_time (W : ℝ) (A B C : ℝ) (ha : A = W / 12) (hb : B = W / 18) (hc : C = W / 9) : 
  1 / (A + B + C) = 4 := 
by sorry

end combined_work_time_l189_189992


namespace polynomial_expansion_correct_l189_189614

def polynomial1 (x : ℝ) := 3 * x^2 - 4 * x + 3
def polynomial2 (x : ℝ) := -2 * x^2 + 3 * x - 4

theorem polynomial_expansion_correct {x : ℝ} :
  (polynomial1 x) * (polynomial2 x) = -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 :=
by
  sorry

end polynomial_expansion_correct_l189_189614


namespace evaluate_expression_l189_189829

theorem evaluate_expression (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 2 * x ^ y + 5 * y ^ x - z ^ 2 = 42 :=
by
  sorry

end evaluate_expression_l189_189829


namespace digit_150_of_1_over_13_is_3_l189_189933

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l189_189933


namespace part_I_part_II_l189_189160

variables {x a : ℝ} (p : Prop) (q : Prop)

-- Proposition p
def prop_p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

-- Proposition q
def prop_q (x : ℝ) : Prop := (x^2 - 2*x - 8 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Part (I)
theorem part_I (a : ℝ) (h : a = 1) : (prop_p x a) → (prop_q x) → (2 < x ∧ x < 4) :=
by
  sorry

-- Part (II)
theorem part_II (a : ℝ) : ¬(∃ x, prop_p x a) → ¬(∃ x, prop_q x) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_part_II_l189_189160


namespace evaluate_abs_expression_l189_189547

noncomputable def approx_pi : ℝ := 3.14159 -- Defining the approximate value of pi

theorem evaluate_abs_expression : |5 * approx_pi - 16| = 0.29205 :=
by
  sorry -- Proof is skipped, as per instructions

end evaluate_abs_expression_l189_189547


namespace raja_journey_distance_l189_189117

theorem raja_journey_distance
  (T : ℝ) (D : ℝ)
  (H1 : T = 10)
  (H2 : ∀ t1 t2, t1 = D / 42 ∧ t2 = D / 48 → T = t1 + t2) :
  D = 224 :=
by
  sorry

end raja_journey_distance_l189_189117


namespace katie_ds_games_l189_189623

theorem katie_ds_games (new_friends_games old_friends_games total_friends_games katie_games : ℕ) 
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_friends_games = 141)
  (h4 : total_friends_games = new_friends_games + old_friends_games + katie_games) :
  katie_games = 0 :=
by
  sorry

end katie_ds_games_l189_189623


namespace sugar_solution_l189_189546

theorem sugar_solution (V x : ℝ) (h1 : V > 0) (h2 : 0.1 * (V - x) + 0.5 * x = 0.2 * V) : x / V = 1 / 4 :=
by sorry

end sugar_solution_l189_189546


namespace union_A_B_l189_189714

def A : Set ℝ := {x | ∃ y : ℝ, y = Real.log x}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : (A ∪ B) = Set.univ :=
by
  sorry

end union_A_B_l189_189714


namespace roberto_outfits_l189_189107

theorem roberto_outfits (trousers shirts jackets : ℕ) (restricted_shirt restricted_jacket : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_restricted_shirt : restricted_shirt = 1) 
  (h_restricted_jacket : restricted_jacket = 1) : 
  ((trousers * shirts * jackets) - (restricted_shirt * restricted_jacket * trousers) = 115) := 
  by 
    sorry

end roberto_outfits_l189_189107


namespace hour_division_convenience_dozen_division_convenience_l189_189679

theorem hour_division_convenience :
  ∃ (a b c d e f g h i j : ℕ), 
  60 = 2 * a ∧
  60 = 3 * b ∧
  60 = 4 * c ∧
  60 = 5 * d ∧
  60 = 6 * e ∧
  60 = 10 * f ∧
  60 = 12 * g ∧
  60 = 15 * h ∧
  60 = 20 * i ∧
  60 = 30 * j := by
  -- to be filled with a proof later
  sorry

theorem dozen_division_convenience :
  ∃ (a b c d : ℕ),
  12 = 2 * a ∧
  12 = 3 * b ∧
  12 = 4 * c ∧
  12 = 6 * d := by
  -- to be filled with a proof later
  sorry

end hour_division_convenience_dozen_division_convenience_l189_189679


namespace bryan_samples_l189_189772

noncomputable def initial_samples_per_shelf : ℕ := 128
noncomputable def shelves : ℕ := 13
noncomputable def samples_removed_per_shelf : ℕ := 2
noncomputable def remaining_samples_per_shelf := initial_samples_per_shelf - samples_removed_per_shelf
noncomputable def total_remaining_samples := remaining_samples_per_shelf * shelves

theorem bryan_samples : total_remaining_samples = 1638 := 
by 
  sorry

end bryan_samples_l189_189772


namespace evaluate_expression_l189_189178

theorem evaluate_expression (a : ℝ) : (a^7 + a^7 + a^7 - a^7) = a^8 :=
by
  sorry

end evaluate_expression_l189_189178


namespace size_of_angle_C_l189_189315

theorem size_of_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 5) 
  (h2 : b + c = 2 * a) 
  (h3 : 3 * Real.sin A = 5 * Real.sin B) : 
  C = 2 * Real.pi / 3 := 
sorry

end size_of_angle_C_l189_189315


namespace triangle_inequality_l189_189208

theorem triangle_inequality (a b c : ℝ) (α : ℝ) 
  (h_triangle_sides : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  (2 * b * c * Real.cos α) / (b + c) < (b + c - a) ∧ (b + c - a) < (2 * b * c) / a := 
sorry

end triangle_inequality_l189_189208


namespace min_max_values_in_interval_l189_189024

def func (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

theorem min_max_values_in_interval :
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≥ -1/3) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = -1/3) ∧
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≤ 9/8) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = 9/8) :=
by
  sorry

end min_max_values_in_interval_l189_189024


namespace quadrant_classification_l189_189158

theorem quadrant_classification :
  ∀ (x y : ℝ), (4 * x - 3 * y = 24) → (|x| = |y|) → 
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  intros x y h_line h_eqdist
  sorry

end quadrant_classification_l189_189158


namespace heather_shared_blocks_l189_189804

-- Define the initial number of blocks Heather starts with
def initial_blocks : ℕ := 86

-- Define the final number of blocks Heather ends up with
def final_blocks : ℕ := 45

-- Define the number of blocks Heather shared
def blocks_shared (initial final : ℕ) : ℕ := initial - final

-- Prove that the number of blocks Heather shared is 41
theorem heather_shared_blocks : blocks_shared initial_blocks final_blocks = 41 := by
  -- Proof steps will be added here
  sorry

end heather_shared_blocks_l189_189804


namespace katie_miles_l189_189326

theorem katie_miles (x : ℕ) (h1 : ∀ y, y = 3 * x → y ≤ 240) (h2 : x + 3 * x = 240) : x = 60 :=
sorry

end katie_miles_l189_189326


namespace parallel_lines_m_value_l189_189954

noncomputable def m_value_parallel (m : ℝ) : Prop :=
  (m-1) / 2 = 1 / -3

theorem parallel_lines_m_value :
  ∀ (m : ℝ), (m_value_parallel m) → m = 1 / 3 :=
by
  intro m
  intro h
  sorry

end parallel_lines_m_value_l189_189954


namespace calculate_final_price_l189_189262

noncomputable def final_price (j_init p_init : ℝ) (j_inc p_inc : ℝ) (tax discount : ℝ) (j_quantity p_quantity : ℕ) : ℝ :=
  let j_new := j_init + j_inc
  let p_new := p_init * (1 + p_inc)
  let total_price := (j_new * j_quantity) + (p_new * p_quantity)
  let tax_amount := total_price * tax
  let price_with_tax := total_price + tax_amount
  let final_price := if j_quantity > 1 ∧ p_quantity >= 3 then price_with_tax * (1 - discount) else price_with_tax
  final_price

theorem calculate_final_price :
  final_price 30 100 10 (0.20) (0.07) (0.10) 2 5 = 654.84 :=
by
  sorry

end calculate_final_price_l189_189262


namespace freezer_temp_is_correct_l189_189257

def freezer_temp (temp: ℤ) := temp

theorem freezer_temp_is_correct (temp: ℤ)
  (freezer_below_zero: temp = -18): freezer_temp temp = -18 := 
by
  -- since freezer_below_zero state that temperature is -18
  exact freezer_below_zero

end freezer_temp_is_correct_l189_189257


namespace two_x_plus_y_eq_12_l189_189372

-- Variables representing the prime numbers x and y
variables {x y : ℕ}

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Prime n
def lcm_eq (a b c : ℕ) : Prop := Nat.lcm a b = c

-- The theorem statement
theorem two_x_plus_y_eq_12 (h1 : lcm_eq x y 10) (h2 : is_prime x) (h3 : is_prime y) (h4 : x > y) :
    2 * x + y = 12 :=
sorry

end two_x_plus_y_eq_12_l189_189372


namespace max_min_value_l189_189485

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x - 2)

theorem max_min_value (M m : ℝ) (hM : M = f 3) (hm : m = f 4) : (m * m) / M = 8 / 3 := by
  sorry

end max_min_value_l189_189485


namespace exists_non_deg_triangle_in_sets_l189_189559

-- Definitions used directly from conditions in a)
def non_deg_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem statement
theorem exists_non_deg_triangle_in_sets (S : Fin 100 → Set ℕ) (h_disjoint : ∀ i j : Fin 100, i ≠ j → Disjoint (S i) (S j))
  (h_union : (⋃ i, S i) = {x | 1 ≤ x ∧ x ≤ 400}) :
  ∃ i : Fin 100, ∃ a b c : ℕ, a ∈ S i ∧ b ∈ S i ∧ c ∈ S i ∧ non_deg_triangle a b c := sorry

end exists_non_deg_triangle_in_sets_l189_189559


namespace johnny_ran_4_times_l189_189644

-- Block length is 200 meters
def block_length : ℕ := 200

-- Distance run by Johnny is Johnny's running times times the block length
def johnny_distance (J : ℕ) : ℕ := J * block_length

-- Distance run by Mickey is half of Johnny's running times times the block length
def mickey_distance (J : ℕ) : ℕ := (J / 2) * block_length

-- Average distance run by Johnny and Mickey is 600 meters
def average_distance_condition (J : ℕ) : Prop :=
  ((johnny_distance J + mickey_distance J) / 2) = 600

-- We are to prove that Johnny ran 4 times based on the condition
theorem johnny_ran_4_times (J : ℕ) (h : average_distance_condition J) : J = 4 :=
sorry

end johnny_ran_4_times_l189_189644


namespace num_houses_with_digit_7_in_range_l189_189216

-- Define the condition for a number to contain a digit 7
def contains_digit_7 (n : Nat) : Prop :=
  (n / 10 = 7) || (n % 10 = 7)

-- The main theorem
theorem num_houses_with_digit_7_in_range (h : Nat) (H1 : 1 ≤ h ∧ h ≤ 70) : 
  ∃! n, 1 ≤ n ∧ n ≤ 70 ∧ contains_digit_7 n :=
sorry

end num_houses_with_digit_7_in_range_l189_189216


namespace two_digit_number_is_91_l189_189042

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end two_digit_number_is_91_l189_189042


namespace correct_order_option_C_l189_189819

def length_unit_ordered (order : List String) : Prop :=
  order = ["kilometer", "meter", "centimeter", "millimeter"]

def option_A := ["kilometer", "meter", "millimeter", "centimeter"]
def option_B := ["meter", "kilometer", "centimeter", "millimeter"]
def option_C := ["kilometer", "meter", "centimeter", "millimeter"]

theorem correct_order_option_C : length_unit_ordered option_C := by
  sorry

end correct_order_option_C_l189_189819


namespace B_starts_after_A_l189_189727

theorem B_starts_after_A :
  ∀ (A_walk_speed B_cycle_speed dist_from_start t : ℝ), 
    A_walk_speed = 10 →
    B_cycle_speed = 20 →
    dist_from_start = 80 →
    B_cycle_speed * (dist_from_start - A_walk_speed * t) / A_walk_speed = t →
    t = 4 :=
by 
  intros A_walk_speed B_cycle_speed dist_from_start t hA_speed hB_speed hdist heq;
  sorry

end B_starts_after_A_l189_189727


namespace smallest_n_l189_189817

theorem smallest_n (n : ℕ) : 
  (n > 0 ∧ ((n^2 + n + 1)^2 > 1999) ∧ ∀ m : ℕ, (m > 0 ∧ (m^2 + m + 1)^2 > 1999) → m ≥ n) → n = 7 :=
sorry

end smallest_n_l189_189817


namespace church_full_capacity_l189_189235

theorem church_full_capacity
  (chairs_per_row : ℕ)
  (rows : ℕ)
  (people_per_chair : ℕ)
  (h1 : chairs_per_row = 6)
  (h2 : rows = 20)
  (h3 : people_per_chair = 5) :
  (chairs_per_row * rows * people_per_chair) = 600 := by
  sorry

end church_full_capacity_l189_189235


namespace ancient_chinese_poem_l189_189454

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) := by
  sorry

end ancient_chinese_poem_l189_189454


namespace factorize_expression_l189_189911

theorem factorize_expression (a : ℝ) : a^3 - 4 * a^2 + 4 * a = a * (a - 2)^2 := 
by
  sorry

end factorize_expression_l189_189911


namespace double_root_conditions_l189_189857

theorem double_root_conditions (k : ℝ) :
  (∃ x, (k - 1)/(x^2 - 1) - 1/(x - 1) = k/(x + 1) ∧ (∀ ε > 0, (∃ δ > 0, (∀ y, |y - x| < δ → (k - 1)/(y^2 - 1) - 1/(y - 1) = k/(y + 1)))))
  → k = 3 ∨ k = 1/3 :=
sorry

end double_root_conditions_l189_189857


namespace drivers_distance_difference_l189_189349

noncomputable def total_distance_driven (initial_distance : ℕ) (speed_A : ℕ) (speed_B : ℕ) (start_delay : ℕ) : ℕ := sorry

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (start_delay : ℕ)
  (correct_difference : ℕ)
  (h_initial : initial_distance = 1025)
  (h_speed_A : speed_A = 90)
  (h_speed_B : speed_B = 80)
  (h_start_delay : start_delay = 1)
  (h_correct_difference : correct_difference = 145) :
  total_distance_driven initial_distance speed_A speed_B start_delay = correct_difference :=
sorry

end drivers_distance_difference_l189_189349


namespace interest_difference_l189_189071

-- Conditions
def principal : ℕ := 350
def rate : ℕ := 4
def time : ℕ := 8

-- Question rewritten as a statement to prove
theorem interest_difference :
  let SI := (principal * rate * time) / 100 
  let difference := principal - SI
  difference = 238 := by
  sorry

end interest_difference_l189_189071


namespace calculate_total_payment_l189_189898

theorem calculate_total_payment
(adult_price : ℕ := 30)
(teen_price : ℕ := 20)
(child_price : ℕ := 15)
(num_adults : ℕ := 4)
(num_teenagers : ℕ := 4)
(num_children : ℕ := 2)
(num_activities : ℕ := 5)
(has_coupon : Bool := true)
(soda_price : ℕ := 5)
(num_sodas : ℕ := 5)

(total_admission_before_discount : ℕ := 
  num_adults * adult_price + num_teenagers * teen_price + num_children * child_price)
(discount_on_activities : ℕ := if num_activities >= 7 then 15 else if num_activities >= 5 then 10 else if num_activities >= 3 then 5 else 0)
(admission_after_activity_discount : ℕ := 
  total_admission_before_discount - total_admission_before_discount * discount_on_activities / 100)
(additional_discount : ℕ := if has_coupon then 5 else 0)
(admission_after_all_discounts : ℕ := 
  admission_after_activity_discount - admission_after_activity_discount * additional_discount / 100)

(total_cost : ℕ := admission_after_all_discounts + num_sodas * soda_price) :
total_cost = 22165 := 
sorry

end calculate_total_payment_l189_189898


namespace four_digit_perfect_square_l189_189686

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l189_189686


namespace profit_percentage_is_twenty_percent_l189_189801

def selling_price : ℕ := 900
def profit : ℕ := 150
def cost_price : ℕ := selling_price - profit
def profit_percentage : ℕ := (profit * 100) / cost_price

theorem profit_percentage_is_twenty_percent : profit_percentage = 20 := by
  sorry

end profit_percentage_is_twenty_percent_l189_189801


namespace solution_set_of_inequality_l189_189498

theorem solution_set_of_inequality (x : ℝ) : (|x - 3| < 1) → (2 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_inequality_l189_189498


namespace ratio_of_rooms_l189_189433

def rooms_in_danielle_apartment : Nat := 6
def rooms_in_heidi_apartment : Nat := 3 * rooms_in_danielle_apartment
def rooms_in_grant_apartment : Nat := 2

theorem ratio_of_rooms :
  (rooms_in_grant_apartment : ℚ) / (rooms_in_heidi_apartment : ℚ) = 1 / 9 := 
by 
  sorry

end ratio_of_rooms_l189_189433


namespace odd_number_divisibility_l189_189124

theorem odd_number_divisibility (a : ℤ) (h : a % 2 = 1) : ∃ (k : ℤ), a^4 + 9 * (9 - 2 * a^2) = 16 * k :=
by
  sorry

end odd_number_divisibility_l189_189124


namespace problem_statement_l189_189620

theorem problem_statement (A B : ℤ) (h1 : A * B = 15) (h2 : -7 * B - 8 * A = -94) : AB + A = 20 := by
  sorry

end problem_statement_l189_189620


namespace probability_at_least_one_defective_item_l189_189963

def total_products : ℕ := 10
def defective_items : ℕ := 3
def selected_items : ℕ := 3
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_least_one_defective_item :
    let total_combinations := comb total_products selected_items
    let non_defective_combinations := comb (total_products - defective_items) selected_items
    let opposite_probability := (non_defective_combinations : ℚ) / (total_combinations : ℚ)
    let probability := 1 - opposite_probability
    probability = 17 / 24 :=
by
  sorry

end probability_at_least_one_defective_item_l189_189963


namespace tomatoes_picked_today_l189_189125

theorem tomatoes_picked_today (initial yesterday_picked left_after_yesterday today_picked : ℕ)
  (h1 : initial = 160)
  (h2 : yesterday_picked = 56)
  (h3 : left_after_yesterday = 104)
  (h4 : initial - yesterday_picked = left_after_yesterday) :
  today_picked = 56 :=
by
  sorry

end tomatoes_picked_today_l189_189125


namespace triangle_area_l189_189446

-- Define the vertices of the triangle
def point_A : (ℝ × ℝ) := (0, 0)
def point_B : (ℝ × ℝ) := (8, -3)
def point_C : (ℝ × ℝ) := (4, 7)

-- Function to compute the area of a triangle given its vertices
def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Conjecture the area of triangle ABC is 30.0 square units
theorem triangle_area : area_of_triangle point_A point_B point_C = 30.0 := by
  sorry

end triangle_area_l189_189446


namespace solve_for_x_l189_189461

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (x + 2) / 5) ^ (1 / 4)

theorem solve_for_x : 
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -404 / 201 := 
by {
  sorry
}

end solve_for_x_l189_189461


namespace sum_of_cubes_three_consecutive_divisible_by_three_l189_189376

theorem sum_of_cubes_three_consecutive_divisible_by_three (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 3 = 0 := 
by 
  sorry

end sum_of_cubes_three_consecutive_divisible_by_three_l189_189376


namespace contractor_realized_work_done_after_20_days_l189_189815

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l189_189815


namespace equal_real_roots_value_of_m_l189_189280

theorem equal_real_roots_value_of_m (m : ℝ) (h : (x^2 - 4*x + m = 0)) 
  (discriminant_zero : (16 - 4*m) = 0) : m = 4 :=
sorry

end equal_real_roots_value_of_m_l189_189280


namespace ads_minutes_l189_189789

-- Definitions and conditions
def videos_per_day : Nat := 2
def minutes_per_video : Nat := 7
def total_time_on_youtube : Nat := 17

-- The theorem to prove
theorem ads_minutes : (total_time_on_youtube - (videos_per_day * minutes_per_video)) = 3 :=
by
  sorry

end ads_minutes_l189_189789


namespace range_of_a_l189_189213

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 1 then x^2 - x + 3 else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ -47 / 16 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l189_189213


namespace cos_135_eq_neg_inv_sqrt_2_l189_189960

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l189_189960


namespace find_number_l189_189411

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end find_number_l189_189411


namespace harmonic_mean_2_3_6_l189_189470

def harmonic_mean (a b c : ℕ) : ℚ := 3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  sorry

end harmonic_mean_2_3_6_l189_189470


namespace probability_problem_l189_189609

def ang_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def ben_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def jasmin_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]

def boxes : Fin 6 := sorry  -- represents 6 empty boxes
def white_restriction (box : Fin 6) : Prop := box ≠ 0  -- white block can't be in the first box

def probability_at_least_one_box_three_same_color : ℚ := 1 / 72  -- The given probability

theorem probability_problem (p q : ℕ) 
  (hpq_coprime : Nat.gcd p q = 1) 
  (hprob_eq : probability_at_least_one_box_three_same_color = p / q) :
  p + q = 73 :=
sorry

end probability_problem_l189_189609


namespace Cody_initial_money_l189_189223

-- Define the conditions
def initial_money (x : ℕ) : Prop :=
  x + 9 - 19 = 35

-- Define the theorem we need to prove
theorem Cody_initial_money : initial_money 45 :=
by
  -- Add a placeholder for the proof
  sorry

end Cody_initial_money_l189_189223


namespace abs_sum_eq_3_given_condition_l189_189025

theorem abs_sum_eq_3_given_condition (m n p : ℤ)
  (h : |m - n|^3 + |p - m|^5 = 1) :
  |p - m| + |m - n| + 2 * |n - p| = 3 :=
sorry

end abs_sum_eq_3_given_condition_l189_189025


namespace edward_money_left_l189_189272

noncomputable def toy_cost : ℝ := 0.95

noncomputable def toy_quantity : ℕ := 4

noncomputable def toy_discount : ℝ := 0.15

noncomputable def race_track_cost : ℝ := 6.00

noncomputable def race_track_tax : ℝ := 0.08

noncomputable def initial_amount : ℝ := 17.80

noncomputable def total_toy_cost_before_discount : ℝ := toy_quantity * toy_cost

noncomputable def discount_amount : ℝ := toy_discount * total_toy_cost_before_discount

noncomputable def total_toy_cost_after_discount : ℝ := total_toy_cost_before_discount - discount_amount

noncomputable def race_track_tax_amount : ℝ := race_track_tax * race_track_cost

noncomputable def total_race_track_cost_after_tax : ℝ := race_track_cost + race_track_tax_amount

noncomputable def total_amount_spent : ℝ := total_toy_cost_after_discount + total_race_track_cost_after_tax

noncomputable def money_left : ℝ := initial_amount - total_amount_spent

theorem edward_money_left : money_left = 8.09 := by
  -- proof goes here
  sorry

end edward_money_left_l189_189272


namespace hyperbola_asymptote_ratio_l189_189347

theorem hyperbola_asymptote_ratio
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))
  (h₃ : ∀ m n: ℝ, m * n = -1 → ∃ θ: ℝ, θ = 90* (π / 180)): 
  a / b = 1 := 
sorry

end hyperbola_asymptote_ratio_l189_189347


namespace alexis_dresses_l189_189068

-- Definitions based on the conditions
def isabella_total : ℕ := 13
def alexis_total : ℕ := 3 * isabella_total
def alexis_pants : ℕ := 21

-- Theorem statement
theorem alexis_dresses : alexis_total - alexis_pants = 18 := by
  sorry

end alexis_dresses_l189_189068


namespace choir_members_count_l189_189112

theorem choir_members_count : ∃ n : ℕ, n = 226 ∧ 
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (200 < n ∧ n < 300) :=
by
  sorry

end choir_members_count_l189_189112


namespace solution_to_inequality_l189_189940

theorem solution_to_inequality : 
  ∀ x : ℝ, (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 :=
by
  intro x
  sorry

end solution_to_inequality_l189_189940


namespace vampires_after_two_nights_l189_189341

-- Define the initial conditions and calculations
def initial_vampires : ℕ := 2
def transformation_rate : ℕ := 5
def first_night_vampires : ℕ := initial_vampires * transformation_rate + initial_vampires
def second_night_vampires : ℕ := first_night_vampires * transformation_rate + first_night_vampires

-- Prove that the number of vampires after two nights is 72
theorem vampires_after_two_nights : second_night_vampires = 72 :=
by sorry

end vampires_after_two_nights_l189_189341


namespace sophia_finished_more_pages_l189_189916

noncomputable def length_of_book : ℝ := 89.99999999999999

noncomputable def total_pages : ℕ := 90  -- Considering the practical purpose

noncomputable def finished_pages : ℕ := total_pages * 2 / 3

noncomputable def remaining_pages : ℕ := total_pages - finished_pages

theorem sophia_finished_more_pages :
  finished_pages - remaining_pages = 30 := 
  by
    -- Use sorry here as placeholder for the proof
    sorry

end sophia_finished_more_pages_l189_189916


namespace savings_for_23_students_is_30_yuan_l189_189832

-- Define the number of students
def number_of_students : ℕ := 23

-- Define the price per ticket in yuan
def price_per_ticket : ℕ := 10

-- Define the discount rate for the group ticket
def discount_rate : ℝ := 0.8

-- Define the group size that is eligible for the discount
def group_size_discount : ℕ := 25

-- Define the cost without ticket discount
def cost_without_discount : ℕ := number_of_students * price_per_ticket

-- Define the cost with the group ticket discount
def cost_with_discount : ℝ := price_per_ticket * discount_rate * group_size_discount

-- Define the expected amount saved by using the group discount
def expected_savings : ℝ := cost_without_discount - cost_with_discount

-- Theorem statement that the expected_savings is 30 yuan
theorem savings_for_23_students_is_30_yuan :
  expected_savings = 30 := 
sorry

end savings_for_23_students_is_30_yuan_l189_189832


namespace decrease_in_silver_coins_l189_189167

theorem decrease_in_silver_coins
  (a : ℕ) (h₁ : 2 * a = 3 * (50 - a))
  (h₂ : a + (50 - a) = 50) :
  (5 * (50 - a) - 3 * a = 10) :=
by
sorry

end decrease_in_silver_coins_l189_189167


namespace pictures_at_the_museum_l189_189517

theorem pictures_at_the_museum (M : ℕ) (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ)
    (h1 : zoo_pics = 15) (h2 : deleted_pics = 31) (h3 : remaining_pics = 2) (h4 : zoo_pics + M = deleted_pics + remaining_pics) :
    M = 18 := 
sorry

end pictures_at_the_museum_l189_189517


namespace function_is_quadratic_l189_189001

-- Definitions for the conditions
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0) ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

-- The function to be proved as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- The theorem statement: f must be a quadratic function
theorem function_is_quadratic : is_quadratic_function f :=
  sorry

end function_is_quadratic_l189_189001


namespace sum_of_four_numbers_eq_zero_l189_189369

theorem sum_of_four_numbers_eq_zero
  (x y s t : ℝ)
  (h₀ : x ≠ y)
  (h₁ : x ≠ s)
  (h₂ : x ≠ t)
  (h₃ : y ≠ s)
  (h₄ : y ≠ t)
  (h₅ : s ≠ t)
  (h_eq : (x + s) / (x + t) = (y + t) / (y + s)) :
  x + y + s + t = 0 := by
sorry

end sum_of_four_numbers_eq_zero_l189_189369


namespace point_A_on_x_axis_l189_189502

def point_A : ℝ × ℝ := (-2, 0)

theorem point_A_on_x_axis : point_A.snd = 0 :=
by
  unfold point_A
  sorry

end point_A_on_x_axis_l189_189502


namespace bananas_per_monkey_l189_189852

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l189_189852


namespace complement_P_relative_to_U_l189_189494

variable (U : Set ℝ) (P : Set ℝ)

theorem complement_P_relative_to_U (hU : U = Set.univ) (hP : P = {x : ℝ | x < 1}) : 
  U \ P = {x : ℝ | x ≥ 1} := by
  sorry

end complement_P_relative_to_U_l189_189494


namespace min_value_of_quadratic_l189_189638

theorem min_value_of_quadratic (x : ℝ) : ∃ y, y = x^2 + 14*x + 20 ∧ ∀ z, z = x^2 + 14*x + 20 → z ≥ -29 :=
by
  sorry

end min_value_of_quadratic_l189_189638


namespace female_managers_count_l189_189742

-- Definitions based on conditions
def total_employees : Nat := 250
def female_employees : Nat := 90
def total_managers : Nat := 40
def male_associates : Nat := 160

-- Statement to prove
theorem female_managers_count : (total_managers = 40) :=
by
  sorry

end female_managers_count_l189_189742


namespace oliver_used_fraction_l189_189783

variable (x : ℚ)

/--
Oliver had 135 stickers. He used a fraction x of his stickers, gave 2/5 of the remaining to his friend, and kept the remaining 54 stickers. Prove that he used 1/3 of his stickers.
-/
theorem oliver_used_fraction (h : 135 - (135 * x) - (2 / 5) * (135 - 135 * x) = 54) : 
  x = 1 / 3 := 
sorry

end oliver_used_fraction_l189_189783


namespace cevian_sum_equals_two_l189_189587

-- Definitions based on conditions
variables {A B C D E F O : Type*}
variables (AD BE CF : ℝ) (R : ℝ)
variables (circumcenter_O : O = circumcenter A B C)
variables (intersect_AD_O : AD = abs ((line A D).proj O))
variables (intersect_BE_O : BE = abs ((line B E).proj O))
variables (intersect_CF_O : CF = abs ((line C F).proj O))

-- Prove the main statement
theorem cevian_sum_equals_two (h : circumcenter_O ∧ intersect_AD_O ∧ intersect_BE_O ∧ intersect_CF_O) :
  1 / AD + 1 / BE + 1 / CF = 2 / R :=
sorry

end cevian_sum_equals_two_l189_189587


namespace real_solutions_x4_plus_3_minus_x4_eq_82_l189_189392

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end real_solutions_x4_plus_3_minus_x4_eq_82_l189_189392


namespace scientific_notation_l189_189924

theorem scientific_notation (h : 0.000000007 = 7 * 10^(-9)) : 0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_l189_189924


namespace vertex_difference_l189_189570

theorem vertex_difference (n m : ℝ) : 
  ∀ x : ℝ, (∀ x, -x^2 + 2*x + n = -((x - m)^2) + 1) → m - n = 1 := 
by 
  sorry

end vertex_difference_l189_189570


namespace arithmetic_sequence_problem_l189_189296

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 + a 4 = 15)
  (h2 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2)
  (h_positive : ∀ n, 0 < a n) :
  a 10 = 19 :=
sorry

end arithmetic_sequence_problem_l189_189296


namespace jason_total_spending_l189_189393

def cost_of_shorts : ℝ := 14.28
def cost_of_jacket : ℝ := 4.74
def total_spent : ℝ := 19.02

theorem jason_total_spending : cost_of_shorts + cost_of_jacket = total_spent :=
by
  sorry

end jason_total_spending_l189_189393


namespace quadratic_inequality_solution_l189_189122

theorem quadratic_inequality_solution (k : ℝ) :
  (-1 < k ∧ k < 7) ↔ ∀ x : ℝ, x^2 - (k - 5) * x - k + 8 > 0 :=
by
  sorry

end quadratic_inequality_solution_l189_189122


namespace common_difference_of_arithmetic_sequence_l189_189509

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def S_n (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (a1 d : ℕ) (h1 : a_n a1 d 3 = 8) (h2 : S_n a1 d 6 = 54) : d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l189_189509


namespace r_fourth_power_sum_l189_189266

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l189_189266


namespace pqr_value_l189_189859

theorem pqr_value (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h1 : p + q + r = 30) 
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + (420 : ℚ) / (p * q * r) = 1) : 
  p * q * r = 1800 := 
sorry

end pqr_value_l189_189859


namespace probability_correct_l189_189154

def elenaNameLength : Nat := 5
def markNameLength : Nat := 4
def juliaNameLength : Nat := 5
def totalCards : Nat := elenaNameLength + markNameLength + juliaNameLength

-- Without replacement, drawing three cards from 14 cards randomly
def probabilityThreeDifferentSources : ℚ := 
  (elenaNameLength / totalCards) * (markNameLength / (totalCards - 1)) * (juliaNameLength / (totalCards - 2))

def totalPermutations : Nat := 6  -- EMJ, EJM, MEJ, MJE, JEM, JME

def requiredProbability : ℚ := totalPermutations * probabilityThreeDifferentSources

theorem probability_correct :
  requiredProbability = 25 / 91 := by
  sorry

end probability_correct_l189_189154


namespace no_real_roots_of_ffx_eq_ninex_l189_189595

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ :=
  x^2 * Real.log (4*(a+1)/a) / Real.log 2 +
  2 * x * Real.log (2 * a / (a + 1)) / Real.log 2 +
  Real.log ((a + 1)^2 / (4 * a^2)) / Real.log 2

theorem no_real_roots_of_ffx_eq_ninex (a : ℝ) (h_pos : ∀ x, 1 ≤ x → f a x > 0) :
  ¬ ∃ x, 1 ≤ x ∧ f a (f a x) = 9 * x :=
  sorry

end no_real_roots_of_ffx_eq_ninex_l189_189595


namespace tangent_of_curve_at_point_l189_189959

def curve (x : ℝ) : ℝ := x^3 - 4 * x

def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

theorem tangent_of_curve_at_point : 
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ tangent_line x y) :=
sorry

end tangent_of_curve_at_point_l189_189959


namespace sufficient_not_necessary_l189_189626

def M : Set ℤ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

theorem sufficient_not_necessary (a : ℤ) :
  (a = 1 → N a ⊆ M) ∧ (N a ⊆ M → a = 1) = false :=
by 
  sorry

end sufficient_not_necessary_l189_189626


namespace largest_k_rooks_l189_189994

noncomputable def rooks_max_k (board_size : ℕ) : ℕ := 
  if board_size = 10 then 16 else 0

theorem largest_k_rooks {k : ℕ} (h : 0 ≤ k ∧ k ≤ 100) :
  k ≤ rooks_max_k 10 := 
sorry

end largest_k_rooks_l189_189994


namespace find_ab_l189_189618

-- Define the conditions and the goal
theorem find_ab (a b : ℝ) (h1 : a^2 + b^2 = 26) (h2 : a + b = 7) : ab = 23 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end find_ab_l189_189618


namespace solve_system_of_equations_l189_189996

def solution_set : Set (ℝ × ℝ) := {(0, 0), (-1, 1), (-2 / (3^(1/3)), -2 * (3^(1/3)))}

theorem solve_system_of_equations (x y : ℝ) :
  (x * y^2 - 2 * y + 3 * x^2 = 0 ∧ y^2 + x^2 * y + 2 * x = 0) ↔ (x, y) ∈ solution_set := sorry

end solve_system_of_equations_l189_189996


namespace liar_and_truth_tellers_l189_189400

-- Define the characters and their nature (truth-teller or liar)
inductive Character : Type
| Kikimora
| Leshy
| Vodyanoy

def always_truthful (c : Character) : Prop := sorry
def always_lying (c : Character) : Prop := sorry

axiom kikimora_statement : always_lying Character.Kikimora
axiom leshy_statement : ∃ l₁ l₂ : Character, l₁ ≠ l₂ ∧ always_lying l₁ ∧ always_lying l₂
axiom vodyanoy_statement : true -- Vodyanoy's silence

-- Proof that Kikimora and Vodyanoy are liars and Leshy is truthful
theorem liar_and_truth_tellers :
  always_lying Character.Kikimora ∧
  always_lying Character.Vodyanoy ∧
  always_truthful Character.Leshy := sorry

end liar_and_truth_tellers_l189_189400


namespace distance_ratio_l189_189011

variables (dw dr : ℝ)

theorem distance_ratio (h1 : 4 * (dw / 4) + 8 * (dr / 8) = 8)
  (h2 : dw + dr = 8)
  (h3 : (dw / 4) + (dr / 8) = 1.5) :
  dw / dr = 1 :=
by
  sorry

end distance_ratio_l189_189011


namespace increase_in_license_plates_l189_189814

/-- The number of old license plates and new license plates in MiraVille. -/
def old_license_plates : ℕ := 26^2 * 10^3
def new_license_plates : ℕ := 26^2 * 10^4

/-- The ratio of the number of new license plates to the number of old license plates is 10. -/
theorem increase_in_license_plates : new_license_plates / old_license_plates = 10 := by
  unfold old_license_plates new_license_plates
  sorry

end increase_in_license_plates_l189_189814


namespace find_center_radius_sum_l189_189417

theorem find_center_radius_sum :
    let x := x
    let y := y
    let a := 2
    let b := 3
    let r := 2 * Real.sqrt 6
    (x^2 - 4 * x + y^2 - 6 * y = 11) →
    (a + b + r = 5 + 2 * Real.sqrt 6) :=
by
  intros x y a b r
  sorry

end find_center_radius_sum_l189_189417


namespace smallest_positive_period_1_smallest_positive_period_2_l189_189639

-- To prove the smallest positive period T for f(x) = |sin x| + |cos x| is π/2
theorem smallest_positive_period_1 : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x : ℝ, (abs (Real.sin (x + T)) + abs (Real.cos (x + T)) = abs (Real.sin x) + abs (Real.cos x))  := sorry

-- To prove the smallest positive period T for f(x) = tan (2x/3) is 3π/2
theorem smallest_positive_period_2 : ∃ T > 0, T = 3 * Real.pi / 2 ∧ ∀ x : ℝ, (Real.tan ((2 * x) / 3 + T) = Real.tan ((2 * x) / 3)) := sorry

end smallest_positive_period_1_smallest_positive_period_2_l189_189639


namespace find_x_l189_189198

theorem find_x
    (x : ℝ)
    (l : ℝ := 4 * x)
    (w : ℝ := x + 8)
    (area_eq_twice_perimeter : l * w = 2 * (2 * l + 2 * w)) :
    x = 2 :=
by
  sorry

end find_x_l189_189198


namespace common_difference_zero_l189_189159

theorem common_difference_zero (a b c : ℕ) 
  (h_seq : ∃ d : ℕ, a = b + d ∧ b = c + d)
  (h_eq : (c - b) / a + (a - c) / b + (b - a) / c = 0) : 
  ∀ d : ℕ, d = 0 :=
by sorry

end common_difference_zero_l189_189159


namespace length_of_garden_l189_189856

variables (w l : ℕ)

-- Definitions based on the problem conditions
def length_twice_width := l = 2 * w
def perimeter_eq_900 := 2 * l + 2 * w = 900

-- The statement to be proved
theorem length_of_garden (h1 : length_twice_width w l) (h2 : perimeter_eq_900 w l) : l = 300 :=
sorry

end length_of_garden_l189_189856


namespace arithmetic_sequence_sum_l189_189952

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 4) (h_a101 : a 101 = 36) : 
  a 9 + a 52 + a 95 = 60 :=
sorry

end arithmetic_sequence_sum_l189_189952


namespace sum_of_sequence_l189_189032

def a (n : ℕ) : ℕ := 2 * n + 1 + 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_of_sequence (n : ℕ) : S n = n^2 + 2 * n + 2^(n + 1) - 2 := 
by 
  sorry

end sum_of_sequence_l189_189032


namespace find_angle_B_in_right_triangle_l189_189078

theorem find_angle_B_in_right_triangle (A B C : ℝ) (hC : C = 90) (hA : A = 35) :
  B = 55 :=
by
  -- Assuming A, B, and C represent the three angles of a triangle ABC
  -- where C = 90 degrees and A = 35 degrees, we need to prove B = 55 degrees.
  sorry

end find_angle_B_in_right_triangle_l189_189078


namespace triangle_area_example_l189_189249

-- Define the right triangle DEF with angle at D being 45 degrees and DE = 8 units
noncomputable def area_of_45_45_90_triangle (DE : ℝ) (angle_d : ℝ) (h_angle : angle_d = 45) (h_DE : DE = 8) : ℝ :=
  1 / 2 * DE * DE

-- State the theorem to prove the area
theorem triangle_area_example {DE : ℝ} {angle_d : ℝ} (h_angle : angle_d = 45) (h_DE : DE = 8) :
  area_of_45_45_90_triangle DE angle_d h_angle h_DE = 32 := 
sorry

end triangle_area_example_l189_189249


namespace rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l189_189566

noncomputable def surface_area_rhombic_dodecahedron (a : ℝ) : ℝ :=
  6 * (a ^ 2) * Real.sqrt 2

noncomputable def volume_rhombic_dodecahedron (a : ℝ) : ℝ :=
  2 * (a ^ 3)

theorem rhombic_dodecahedron_surface_area (a : ℝ) :
  surface_area_rhombic_dodecahedron a = 6 * (a ^ 2) * Real.sqrt 2 :=
by
  sorry

theorem rhombic_dodecahedron_volume (a : ℝ) :
  volume_rhombic_dodecahedron a = 2 * (a ^ 3) :=
by
  sorry

end rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l189_189566


namespace intersection_A_B_intersection_CA_B_intersection_CA_CB_l189_189355

-- Set definitions
def A := {x : ℝ | -5 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | x < -2 ∨ x > 4}
def C_A := {x : ℝ | x < -5 ∨ x > 3}  -- Complement of A
def C_B := {x : ℝ | -2 ≤ x ∧ x ≤ 4}  -- Complement of B

-- Lean statements proving the intersections
theorem intersection_A_B : {x : ℝ | -5 ≤ x ∧ x ≤ 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | -5 ≤ x ∧ x < -2} :=
by sorry

theorem intersection_CA_B : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | x < -5 ∨ x > 4} :=
by sorry

theorem intersection_CA_CB : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 4} = {x : ℝ | 3 < x ∧ x ≤ 4} :=
by sorry

end intersection_A_B_intersection_CA_B_intersection_CA_CB_l189_189355


namespace sequence_sum_a5_a6_l189_189336

-- Given sequence partial sum definition
def partial_sum (n : ℕ) : ℕ := n^3

-- Definition of sequence term a_n
def a (n : ℕ) : ℕ := partial_sum n - partial_sum (n - 1)

-- Main theorem to prove a_5 + a_6 = 152
theorem sequence_sum_a5_a6 : a 5 + a 6 = 152 :=
by
  sorry

end sequence_sum_a5_a6_l189_189336


namespace problem_1_part_1_problem_1_part_2_l189_189496

-- Define the function f
def f (x a : ℝ) := |x - a| + 3 * x

-- The first problem statement - Part (Ⅰ)
theorem problem_1_part_1 (x : ℝ) : { x | x ≥ 3 ∨ x ≤ -1 } = { x | f x 1 ≥ 3 * x + 2 } :=
by {
  sorry
}

-- The second problem statement - Part (Ⅱ)
theorem problem_1_part_2 : { x | x ≤ -1 } = { x | f x 2 ≤ 0 } :=
by {
  sorry
}

end problem_1_part_1_problem_1_part_2_l189_189496


namespace range_of_m_l189_189717

noncomputable def proposition_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 + m * x + 1 ≥ 0

noncomputable def proposition_q (m : ℝ) : Prop :=
∀ x : ℝ, (8 * x + 4 * (m - 1)) ≥ 0

def conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) : 
  conditions m → ( -2 ≤ m ∧ m < 1 ) ∨ m > 2 :=
by
  intros h
  sorry

end range_of_m_l189_189717


namespace triangle_area_l189_189321

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) (h4 : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 336 := 
by 
  rw [h1, h2]
  sorry

end triangle_area_l189_189321


namespace log_domain_l189_189457

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2

theorem log_domain :
  ∀ x : ℝ, (∃ y : ℝ, f y = Real.log (x - 1) / Real.log 2) ↔ x ∈ Set.Ioi 1 :=
by {
  sorry
}

end log_domain_l189_189457


namespace year_2013_is_not_lucky_l189_189585

-- Definitions based on conditions
def last_two_digits (year : ℕ) : ℕ := year % 100

def is_valid_date (month : ℕ) (day : ℕ) (year : ℕ) : Prop :=
  month * day = last_two_digits year

def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month : ℕ) (day : ℕ), month <= 12 ∧ day <= 12 ∧ is_valid_date month day year

-- The main statement to prove
theorem year_2013_is_not_lucky : ¬ is_lucky_year 2013 :=
by {
  sorry
}

end year_2013_is_not_lucky_l189_189585


namespace stratified_sampling_expected_females_l189_189468

noncomputable def sample_size := 14
noncomputable def total_athletes := 44 + 33
noncomputable def female_athletes := 33
noncomputable def stratified_sample := (female_athletes * sample_size) / total_athletes

theorem stratified_sampling_expected_females :
  stratified_sample = 6 :=
by
  sorry

end stratified_sampling_expected_females_l189_189468


namespace modulus_of_z_l189_189794

-- Define the complex number z
def z : ℂ := -5 + 12 * Complex.I

-- Define a theorem stating the modulus of z is 13
theorem modulus_of_z : Complex.abs z = 13 :=
by
  -- This will be the place to provide proof steps
  sorry

end modulus_of_z_l189_189794


namespace votes_for_veggies_l189_189691

theorem votes_for_veggies (T M V : ℕ) (hT : T = 672) (hM : M = 335) (hV : V = T - M) : V = 337 := 
by
  rw [hT, hM] at hV
  simp at hV
  exact hV

end votes_for_veggies_l189_189691


namespace enrico_earnings_l189_189279

def roosterPrice (weight: ℕ) : ℝ :=
  if weight < 20 then weight * 0.80
  else if weight ≤ 35 then weight * 0.65
  else weight * 0.50

theorem enrico_earnings :
  roosterPrice 15 + roosterPrice 30 + roosterPrice 40 + roosterPrice 50 = 76.50 := 
by
  sorry

end enrico_earnings_l189_189279


namespace interior_angle_sum_of_regular_polygon_l189_189555

theorem interior_angle_sum_of_regular_polygon (h: ∀ θ, θ = 45) :
  ∃ s, s = 1080 := by
  sorry

end interior_angle_sum_of_regular_polygon_l189_189555


namespace proposition_D_l189_189016

-- Definitions extracted from the conditions
variables {a b : ℝ} (c d : ℝ)

-- Proposition D to be proven
theorem proposition_D (ha : a < b) (hb : b < 0) : a^2 > b^2 := sorry

end proposition_D_l189_189016


namespace ratio_of_x_to_y_l189_189956

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 := 
by
  sorry

end ratio_of_x_to_y_l189_189956


namespace gcd_442872_312750_l189_189692

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 :=
by
  sorry

end gcd_442872_312750_l189_189692


namespace distance_to_destination_l189_189965

theorem distance_to_destination 
  (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  speed * time = 500 :=
by
  rw [h_speed, h_time]
  -- This simplifies to 100 * 5 = 500
  norm_num

end distance_to_destination_l189_189965


namespace circle_condition_l189_189271

theorem circle_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + 5 * k = 0) ↔ k < 1 := 
sorry

end circle_condition_l189_189271


namespace certain_number_l189_189212

theorem certain_number (n q1 q2: ℕ) (h1 : 49 = n * q1 + 4) (h2 : 66 = n * q2 + 6): n = 15 :=
sorry

end certain_number_l189_189212


namespace a_n_formula_S_n_formula_T_n_formula_l189_189938

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence (3 ^ n)
noncomputable def T (n : ℕ) : ℕ := 3^(n + 1) - 3

theorem a_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → a_sequence n = 2 * n :=
sorry

theorem S_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → S n = n * (n + 1) :=
sorry

theorem T_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → T n = 3^(n + 1) - 3 :=
sorry

end a_n_formula_S_n_formula_T_n_formula_l189_189938


namespace gcd_660_924_l189_189199

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end gcd_660_924_l189_189199


namespace wine_age_proof_l189_189221

-- Definitions based on conditions
def Age_Carlo_Rosi : ℕ := 40
def Age_Twin_Valley : ℕ := Age_Carlo_Rosi / 4
def Age_Franzia : ℕ := 3 * Age_Carlo_Rosi

-- We'll use a definition to represent the total age of the three brands of wine.
def Total_Age : ℕ := Age_Franzia + Age_Carlo_Rosi + Age_Twin_Valley

-- Statement to be proven
theorem wine_age_proof : Total_Age = 170 :=
by {
  sorry -- Proof goes here
}

end wine_age_proof_l189_189221


namespace triangle_inequality_not_true_l189_189997

theorem triangle_inequality_not_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : ¬ (b + c > 2 * a) :=
by {
  -- assume (b + c > 2 * a)
  -- we need to reach a contradiction
  sorry
}

end triangle_inequality_not_true_l189_189997


namespace perfect_square_digits_l189_189654

theorem perfect_square_digits (x y : ℕ) (h_ne_zero : x ≠ 0) (h_perfect_square : ∀ n: ℕ, n ≥ 1 → ∃ k: ℕ, (10^(n + 2) * x + 10^(n + 1) * 6 + 10 * y + 4) = k^2) :
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0) :=
sorry

end perfect_square_digits_l189_189654


namespace find_w_l189_189870

variables {x y z w : ℝ}

theorem find_w (h : (1 / x) + (1 / y) + (1 / z) = 1 / w) :
  w = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end find_w_l189_189870


namespace maximum_cards_without_equal_pair_sums_l189_189153

def max_cards_no_equal_sum_pairs : ℕ :=
  let card_points := {x : ℕ | 1 ≤ x ∧ x ≤ 13}
  6

theorem maximum_cards_without_equal_pair_sums (deck : Finset ℕ) (h_deck : deck = {x : ℕ | 1 ≤ x ∧ x ≤ 13}) :
  ∃ S ⊆ deck, S.card = 6 ∧ ∀ {a b c d : ℕ}, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a + b = c + d → a = c ∧ b = d ∨ a = d ∧ b = c := 
sorry

end maximum_cards_without_equal_pair_sums_l189_189153


namespace total_distance_is_105_km_l189_189802

-- Define the boat's speed in still water
def boat_speed_still_water : ℝ := 50

-- Define the current speeds for each hour
def current_speed_first_hour : ℝ := 10
def current_speed_second_hour : ℝ := 20
def current_speed_third_hour : ℝ := 15

-- Calculate the effective speeds for each hour
def effective_speed_first_hour := boat_speed_still_water - current_speed_first_hour
def effective_speed_second_hour := boat_speed_still_water - current_speed_second_hour
def effective_speed_third_hour := boat_speed_still_water - current_speed_third_hour

-- Calculate the distance traveled in each hour
def distance_first_hour := effective_speed_first_hour * 1
def distance_second_hour := effective_speed_second_hour * 1
def distance_third_hour := effective_speed_third_hour * 1

-- Define the total distance
def total_distance_traveled := distance_first_hour + distance_second_hour + distance_third_hour

-- Prove that the total distance traveled is 105 km
theorem total_distance_is_105_km : total_distance_traveled = 105 := by
  sorry

end total_distance_is_105_km_l189_189802


namespace inverse_of_g_compose_three_l189_189847

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Assuming g(x) is defined only for x in {1, 2, 3, 4, 5}

noncomputable def g_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 1 => 3
  | 5 => 4
  | 2 => 5
  | _ => 0  -- Assuming g_inv(y) is defined only for y in {1, 3, 1, 5, 2}

theorem inverse_of_g_compose_three : g_inv (g_inv (g_inv 3)) = 4 := by
  sorry

end inverse_of_g_compose_three_l189_189847


namespace power_of_thousand_l189_189677

-- Define the notion of googol
def googol := 10^100

-- Prove that 1000^100 is equal to googol^3
theorem power_of_thousand : (1000 ^ 100) = googol^3 := by
  -- proof step to be filled here
  sorry

end power_of_thousand_l189_189677


namespace units_digit_of_square_l189_189629

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := 
by 
  sorry

end units_digit_of_square_l189_189629


namespace routeY_is_quicker_l189_189685

noncomputable def timeRouteX : ℝ := 
  8 / 40 

noncomputable def timeRouteY1 : ℝ := 
  6.5 / 50 

noncomputable def timeRouteY2 : ℝ := 
  0.5 / 10

noncomputable def timeRouteY : ℝ := 
  timeRouteY1 + timeRouteY2  

noncomputable def timeDifference : ℝ := 
  (timeRouteX - timeRouteY) * 60 

theorem routeY_is_quicker : 
  timeDifference = 1.2 :=
by
  sorry

end routeY_is_quicker_l189_189685


namespace algebraic_identity_l189_189014

theorem algebraic_identity (theta : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < theta) (h2 : theta < π) (h3 : x + x⁻¹ = 2 * Real.cos theta) : 
  x^n + (x⁻¹)^n = 2 * Real.cos (n * theta) :=
by
  sorry

end algebraic_identity_l189_189014


namespace line_perp_to_plane_contains_line_implies_perp_l189_189303

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables (contains : Plane → Line → Prop) (perp : Line → Line → Prop) (perp_plane : Line → Plane → Prop)

-- Given: 
-- m and n are two different lines
-- α is a plane
-- m ⊥ α (m is perpendicular to the plane α)
-- n ⊂ α (n is contained in the plane α)
-- Prove: m ⊥ n
theorem line_perp_to_plane_contains_line_implies_perp (hm : perp_plane m α) (hn : contains α n) : perp m n :=
sorry

end line_perp_to_plane_contains_line_implies_perp_l189_189303


namespace prove_b_minus_a_l189_189054

noncomputable def point := (ℝ × ℝ)

def rotate90 (p : point) (c : point) : point :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (p : point) : point :=
  let (x, y) := p
  (y, x)

def transformed_point (a b : ℝ) : point :=
  reflect_y_eq_x (rotate90 (a, b) (2, 6))

theorem prove_b_minus_a (a b : ℝ) (h1 : transformed_point a b = (-7, 4)) : b - a = 15 :=
by
  sorry

end prove_b_minus_a_l189_189054


namespace sum_of_k_l189_189463

theorem sum_of_k : ∃ (k_vals : List ℕ), 
  (∀ k ∈ k_vals, ∃ α β : ℤ, α + β = k ∧ α * β = -20) 
  ∧ k_vals.sum = 29 :=
by 
  sorry

end sum_of_k_l189_189463


namespace certain_amount_l189_189541

theorem certain_amount (x : ℝ) (A : ℝ) (h1: x = 900) (h2: 0.25 * x = 0.15 * 1600 - A) : A = 15 :=
by
  sorry

end certain_amount_l189_189541


namespace twentieth_fisherman_caught_l189_189408

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l189_189408


namespace first_worker_time_l189_189267

def productivity (x y z : ℝ) : Prop :=
  x + y + z = 20 ∧
  (20 / x) > 3 ∧
  (20 / x) + (60 / (y + z)) = 8

theorem first_worker_time (x y z : ℝ) (h : productivity x y z) : 
  (80 / x) = 16 :=
  sorry

end first_worker_time_l189_189267


namespace set_union_example_l189_189755

theorem set_union_example (x : ℕ) (M N : Set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) :
  M ∪ N = {0, 1, 2} := by
  sorry

end set_union_example_l189_189755


namespace points_on_circle_l189_189917

theorem points_on_circle (t : ℝ) : ∃ x y : ℝ, x = Real.cos t ∧ y = Real.sin t ∧ x^2 + y^2 = 1 :=
by
  sorry

end points_on_circle_l189_189917


namespace smartphone_customers_l189_189613

theorem smartphone_customers (k : ℝ) (p1 p2 c1 c2 : ℝ)
  (h₁ : p1 * c1 = k)
  (h₂ : 20 = p1)
  (h₃ : 200 = c1)
  (h₄ : 400 = c2) :
  p2 * c2 = k  → p2 = 10 :=
by
  sorry

end smartphone_customers_l189_189613


namespace both_solve_correctly_l189_189097

-- Define the probabilities of making an error for individuals A and B
variables (a b : ℝ)

-- Assuming a and b are probabilities, they must lie in the interval [0, 1]
axiom a_prob : 0 ≤ a ∧ a ≤ 1
axiom b_prob : 0 ≤ b ∧ b ≤ 1

-- Define the event that both individuals solve the problem correctly
theorem both_solve_correctly : (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by
  sorry

end both_solve_correctly_l189_189097


namespace tangent_parallel_l189_189987

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P₀ : ℝ × ℝ) :
  (∃ x : ℝ, (P₀ = (x, f x) ∧ deriv f x = 4)) 
  ↔ (P₀ = (1, 0) ∨ P₀ = (-1, -4)) :=
by 
  sorry

end tangent_parallel_l189_189987


namespace unique_solution_of_system_of_equations_l189_189102
open Set

variable {α : Type*} (A B X : Set α)

theorem unique_solution_of_system_of_equations :
  (X ∩ (A ∪ B) = X) ∧
  (A ∩ (B ∪ X) = A) ∧
  (B ∩ (A ∪ X) = B) ∧
  (X ∩ A ∩ B = ∅) →
  (X = (A \ B) ∪ (B \ A)) :=
by
  sorry

end unique_solution_of_system_of_equations_l189_189102


namespace greatest_divisor_of_arithmetic_sequence_sum_l189_189435

theorem greatest_divisor_of_arithmetic_sequence_sum (x c : ℕ) (hx : x > 0) (hc : c > 0) :
  ∃ k, (∀ (S : ℕ), S = 6 * (2 * x + 11 * c) → k ∣ S) ∧ k = 6 :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l189_189435


namespace remainder_of_3_pow_2023_mod_7_l189_189809

theorem remainder_of_3_pow_2023_mod_7 : (3^2023) % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l189_189809


namespace triangle_area_l189_189748

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 180 := 
by 
  -- proof is skipped with sorry
  sorry

end triangle_area_l189_189748


namespace y_intercept_of_line_is_minus_one_l189_189243

theorem y_intercept_of_line_is_minus_one : 
  (∀ x y : ℝ, y = 2 * x - 1 → y = -1) :=
by
  sorry

end y_intercept_of_line_is_minus_one_l189_189243


namespace twenty_two_percent_of_three_hundred_l189_189773

theorem twenty_two_percent_of_three_hundred : 
  (22 / 100) * 300 = 66 :=
by
  sorry

end twenty_two_percent_of_three_hundred_l189_189773


namespace pounds_lost_per_month_l189_189413

variable (starting_weight : ℕ) (ending_weight : ℕ) (months_in_year : ℕ) 

theorem pounds_lost_per_month
    (h_start : starting_weight = 250)
    (h_end : ending_weight = 154)
    (h_months : months_in_year = 12) :
    (starting_weight - ending_weight) / months_in_year = 8 := 
sorry

end pounds_lost_per_month_l189_189413


namespace kibble_recommendation_difference_l189_189430

theorem kibble_recommendation_difference :
  (0.2 * 1000 : ℝ) < (0.3 * 1000) ∧ ((0.3 * 1000) - (0.2 * 1000)) = 100 :=
by
  sorry

end kibble_recommendation_difference_l189_189430


namespace students_per_minibus_calculation_l189_189684

-- Define the conditions
variables (vans minibusses total_students students_per_van : ℕ)
variables (students_per_minibus : ℕ)

-- Define the given conditions based on the problem
axiom six_vans : vans = 6
axiom four_minibusses : minibusses = 4
axiom ten_students_per_van : students_per_van = 10
axiom total_students_are_156 : total_students = 156

-- Define the problem statement in Lean
theorem students_per_minibus_calculation
  (h1 : vans = 6)
  (h2 : minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : total_students = 156) :
  students_per_minibus = 24 :=
sorry

end students_per_minibus_calculation_l189_189684


namespace sum_of_constants_eq_zero_l189_189093

theorem sum_of_constants_eq_zero (A B C D E : ℝ) :
  (∀ (x : ℝ), (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
              A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 :=
by
  sorry

end sum_of_constants_eq_zero_l189_189093


namespace arithmetic_seq_40th_term_l189_189432

theorem arithmetic_seq_40th_term (a₁ d : ℕ) (n : ℕ) (h1 : a₁ = 3) (h2 : d = 4) (h3 : n = 40) : 
  a₁ + (n - 1) * d = 159 :=
by
  sorry

end arithmetic_seq_40th_term_l189_189432


namespace intersection_A_B_l189_189287

open Set

noncomputable def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

noncomputable def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 3} :=
by {
  sorry
}

end intersection_A_B_l189_189287


namespace zebras_total_games_l189_189245

theorem zebras_total_games 
  (x y : ℝ)
  (h1 : x = 0.40 * y)
  (h2 : (x + 8) / (y + 11) = 0.55) 
  : y + 11 = 24 :=
sorry

end zebras_total_games_l189_189245


namespace students_between_hoseok_and_minyoung_l189_189230

def num_students : Nat := 13
def hoseok_position_from_right : Nat := 9
def minyoung_position_from_left : Nat := 8

theorem students_between_hoseok_and_minyoung
    (n : Nat)
    (h : n = num_students)
    (p_h : n - hoseok_position_from_right + 1 = 5)
    (p_m : minyoung_position_from_left = 8):
    ∃ k : Nat, k = 2 :=
by
  sorry

end students_between_hoseok_and_minyoung_l189_189230


namespace coconut_grove_nut_yield_l189_189759

/--
In a coconut grove, the trees produce nuts based on some given conditions. Prove that the number of nuts produced by (x + 4) trees per year is 720 when x is 8. The conditions are:

1. (x + 4) trees yield a certain number of nuts per year.
2. x trees yield 120 nuts per year.
3. (x - 4) trees yield 180 nuts per year.
4. The average yield per year per tree is 100.
5. x is 8.
-/

theorem coconut_grove_nut_yield (x : ℕ) (y z w: ℕ) (h₁ : x = 8) (h₂ : y = 120) (h₃ : z = 180) (h₄ : w = 100) :
  ((x + 4) * w) - (x * y + (x - 4) * z) = 720 := 
by
  sorry

end coconut_grove_nut_yield_l189_189759


namespace cars_meet_time_l189_189524

-- Define the initial conditions as Lean definitions
def distance_car1 (t : ℝ) : ℝ := 15 * t
def distance_car2 (t : ℝ) : ℝ := 20 * t
def total_distance : ℝ := 105

-- Define the proposition we want to prove
theorem cars_meet_time : ∃ (t : ℝ), distance_car1 t + distance_car2 t = total_distance ∧ t = 3 :=
by
  sorry

end cars_meet_time_l189_189524


namespace retirement_total_l189_189792

/-- A company retirement plan allows an employee to retire when their age plus years of employment total a specific number.
A female employee was hired in 1990 on her 32nd birthday. She could first be eligible to retire under this provision in 2009. -/
def required_total_age_years_of_employment : ℕ :=
  let hire_year := 1990
  let retirement_year := 2009
  let age_when_hired := 32
  let years_of_employment := retirement_year - hire_year
  let age_at_retirement := age_when_hired + years_of_employment
  age_at_retirement + years_of_employment

theorem retirement_total :
  required_total_age_years_of_employment = 70 :=
by
  sorry

end retirement_total_l189_189792


namespace parts_of_cut_square_l189_189229

theorem parts_of_cut_square (folds_to_one_by_one : ℕ) : folds_to_one_by_one = 9 :=
  sorry

end parts_of_cut_square_l189_189229


namespace avg_marks_l189_189233

theorem avg_marks (P C M B E H G : ℝ) 
  (h1 : C = P + 75)
  (h2 : M = P + 105)
  (h3 : B = P - 15)
  (h4 : E = P - 25)
  (h5 : H = P - 25)
  (h6 : G = P - 25)
  (h7 : P + C + M + B + E + H + G = P + 520) :
  (M + B + H + G) / 4 = 82 :=
by 
  sorry

end avg_marks_l189_189233


namespace power_of_five_trailing_zeros_l189_189779

theorem power_of_five_trailing_zeros (n : ℕ) (h : n = 1968) : 
  ∃ k : ℕ, 5^n = 10^k ∧ k ≥ 1968 := 
by 
  sorry

end power_of_five_trailing_zeros_l189_189779


namespace instantaneous_speed_at_4_l189_189822

def motion_equation (t : ℝ) : ℝ := t^2 - 2 * t + 5

theorem instantaneous_speed_at_4 :
  (deriv motion_equation 4) = 6 :=
by
  sorry

end instantaneous_speed_at_4_l189_189822


namespace determine_digit_z_l189_189580

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l189_189580


namespace Jorge_age_in_2005_l189_189636

theorem Jorge_age_in_2005
  (age_Simon_2010 : ℕ)
  (age_difference : ℕ)
  (age_of_Simon_2010 : age_Simon_2010 = 45)
  (age_difference_Simon_Jorge : age_difference = 24)
  (age_Simon_2005 : ℕ := age_Simon_2010 - 5)
  (age_Jorge_2005 : ℕ := age_Simon_2005 - age_difference) :
  age_Jorge_2005 = 16 := by
  sorry

end Jorge_age_in_2005_l189_189636


namespace sum_of_first_and_third_l189_189605

theorem sum_of_first_and_third :
  ∀ (A B C : ℕ),
  A + B + C = 330 →
  A = 2 * B →
  C = A / 3 →
  B = 90 →
  A + C = 240 :=
by
  intros A B C h1 h2 h3 h4
  sorry

end sum_of_first_and_third_l189_189605


namespace lassis_from_mangoes_l189_189681

theorem lassis_from_mangoes (L M : ℕ) (h : 2 * L = 11 * M) : 12 * L = 66 :=
by sorry

end lassis_from_mangoes_l189_189681


namespace ab_sum_l189_189150

theorem ab_sum (A B C D : Nat) (h_digits: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_mult : A * (10 * C + D) = 1001 + 100 * A + 10 * B + A) : A + B = 1 := 
  sorry

end ab_sum_l189_189150


namespace bowling_tournament_orders_l189_189277

theorem bowling_tournament_orders :
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  total_orders = 32 :=
by
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  show total_orders = 32
  sorry

end bowling_tournament_orders_l189_189277


namespace oliver_shirts_problem_l189_189242

-- Defining the quantities of short sleeve shirts, long sleeve shirts, and washed shirts.
def shortSleeveShirts := 39
def longSleeveShirts  := 47
def shirtsWashed := 20

-- Stating the problem formally.
theorem oliver_shirts_problem :
  shortSleeveShirts + longSleeveShirts - shirtsWashed = 66 :=
by
  -- Proof goes here.
  sorry

end oliver_shirts_problem_l189_189242


namespace haleigh_cats_l189_189934

open Nat

def total_pairs := 14
def dog_leggings := 4
def legging_per_animal := 1

theorem haleigh_cats : ∀ (dogs cats : ℕ), 
  dogs = 4 → 
  total_pairs = dogs * legging_per_animal + cats * legging_per_animal → 
  cats = 10 :=
by
  intros dogs cats h1 h2
  sorry

end haleigh_cats_l189_189934


namespace solve_quadratic_eqn_l189_189043

theorem solve_quadratic_eqn : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_quadratic_eqn_l189_189043


namespace sum_of_first_70_odd_integers_l189_189174

theorem sum_of_first_70_odd_integers : 
  let sum_even := 70 * (70 + 1)
  let sum_odd := 70 ^ 2
  let diff := sum_even - sum_odd
  diff = 70 → sum_odd = 4900 :=
by
  intros
  sorry

end sum_of_first_70_odd_integers_l189_189174


namespace new_trailer_homes_added_l189_189018

theorem new_trailer_homes_added (n : ℕ) (h1 : (20 * 20 + 2 * n)/(20 + n) = 14) : n = 10 :=
by
  sorry

end new_trailer_homes_added_l189_189018


namespace Jacqueline_gave_Jane_l189_189295

def total_fruits (plums guavas apples : ℕ) : ℕ :=
  plums + guavas + apples

def fruits_given_to_Jane (initial left : ℕ) : ℕ :=
  initial - left

theorem Jacqueline_gave_Jane :
  let plums := 16
  let guavas := 18
  let apples := 21
  let left := 15
  let initial := total_fruits plums guavas apples
  fruits_given_to_Jane initial left = 40 :=
by
  sorry

end Jacqueline_gave_Jane_l189_189295


namespace color_5x5_grid_excluding_two_corners_l189_189607

-- Define the total number of ways to color a 5x5 grid with each row and column having exactly one colored cell
def total_ways : Nat := 120

-- Define the number of ways to color a 5x5 grid excluding one specific corner cell such that each row and each column has exactly one colored cell
def ways_excluding_one_corner : Nat := 96

-- Prove the number of ways to color the grid excluding two specific corner cells is 78
theorem color_5x5_grid_excluding_two_corners : total_ways - (ways_excluding_one_corner + ways_excluding_one_corner - 6) = 78 := by
  -- We state our given conditions directly as definitions
  -- Now we state our theorem explicitly and use the correct answer we derived
  sorry

end color_5x5_grid_excluding_two_corners_l189_189607


namespace exists_trinomial_with_exponents_three_l189_189085

theorem exists_trinomial_with_exponents_three (x y : ℝ) :
  ∃ (a b c : ℝ) (t1 t2 t3 : ℕ × ℕ), 
  t1.1 + t1.2 = 3 ∧ t2.1 + t2.2 = 3 ∧ t3.1 + t3.2 = 3 ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (a * x ^ t1.1 * y ^ t1.2 + b * x ^ t2.1 * y ^ t2.2 + c * x ^ t3.1 * y ^ t3.2 ≠ 0) := sorry

end exists_trinomial_with_exponents_three_l189_189085


namespace find_a_of_inequality_solution_set_l189_189259

theorem find_a_of_inequality_solution_set :
  (∃ (a : ℝ), (∀ (x : ℝ), |2*x - a| + a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ a = 1) :=
by sorry

end find_a_of_inequality_solution_set_l189_189259


namespace smallest_n_condition_l189_189751

-- Define the conditions
def condition1 (x : ℤ) : Prop := 2 * x - 3 ≡ 0 [ZMOD 13]
def condition2 (y : ℤ) : Prop := 3 * y + 4 ≡ 0 [ZMOD 13]

-- Problem statement: finding n such that the expression is a multiple of 13
theorem smallest_n_condition (x y : ℤ) (n : ℤ) :
  condition1 x → condition2 y → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 13] → n = 1 := 
by
  sorry

end smallest_n_condition_l189_189751


namespace average_infected_per_round_is_nine_l189_189664

theorem average_infected_per_round_is_nine (x : ℝ) :
  1 + x + x * (1 + x) = 100 → x = 9 :=
by {
  sorry
}

end average_infected_per_round_is_nine_l189_189664


namespace missing_bricks_is_26_l189_189621

-- Define the number of bricks per row and the number of rows
def bricks_per_row : Nat := 10
def number_of_rows : Nat := 6

-- Calculate the total number of bricks for a fully completed wall
def total_bricks_full_wall : Nat := bricks_per_row * number_of_rows

-- Assume the number of bricks currently present
def bricks_currently_present : Nat := total_bricks_full_wall - 26

-- Define a function that calculates the number of missing bricks
def number_of_missing_bricks (total_bricks : Nat) (bricks_present : Nat) : Nat :=
  total_bricks - bricks_present

-- Prove that the number of missing bricks is 26
theorem missing_bricks_is_26 : 
  number_of_missing_bricks total_bricks_full_wall bricks_currently_present = 26 :=
by
  sorry

end missing_bricks_is_26_l189_189621


namespace min_square_side_length_l189_189519

theorem min_square_side_length 
  (table_length : ℕ) (table_breadth : ℕ) (cube_side : ℕ) (num_tables : ℕ)
  (cond1 : table_length = 12)
  (cond2 : table_breadth = 16)
  (cond3 : cube_side = 4)
  (cond4 : num_tables = 4) :
  (2 * table_length + 2 * table_breadth) = 56 := 
by
  sorry

end min_square_side_length_l189_189519


namespace number_of_red_notes_each_row_l189_189292

-- Definitions for the conditions
variable (R : ℕ) -- Number of red notes in each row
variable (total_notes : ℕ := 100) -- Total number of notes

-- Derived quantities
def total_red_notes := 5 * R
def total_blue_notes := 2 * total_red_notes + 10

-- Statement of the theorem
theorem number_of_red_notes_each_row 
  (h : total_red_notes + total_blue_notes = total_notes) : 
  R = 6 :=
by
  sorry

end number_of_red_notes_each_row_l189_189292


namespace max_value_l189_189316

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ {a b}, a < b → f a < f b

theorem max_value (f : ℝ → ℝ) (x y : ℝ)
  (h_odd : is_odd f)
  (h_increasing : is_increasing f)
  (h_eq : f (x^2 - 2 * x) + f y = 0) :
  2 * x + y ≤ 4 :=
sorry

end max_value_l189_189316


namespace prove_a_lt_one_l189_189227

/-- Given the function f defined as -2 * ln x + 1 / 2 * (x^2 + 1) - a * x,
    where a > 0, if f(x) ≥ 0 holds in the interval (1, ∞)
    and f(x) = 0 has a unique solution, then a < 1. -/
theorem prove_a_lt_one (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = -2 * Real.log x + 1 / 2 * (x^2 + 1) - a * x)
    (h2 : a > 0)
    (h3 : ∀ x, x > 1 → f x ≥ 0)
    (h4 : ∃! x, f x = 0) : 
    a < 1 :=
by
  sorry

end prove_a_lt_one_l189_189227


namespace original_average_rent_l189_189915

theorem original_average_rent
    (A : ℝ) -- original average rent per person
    (h1 : 4 * A + 200 = 3400) -- condition derived from the rent problem
    : A = 800 := 
sorry

end original_average_rent_l189_189915


namespace obtuse_triangle_iff_distinct_real_roots_l189_189588

theorem obtuse_triangle_iff_distinct_real_roots
  (A B C : ℝ)
  (h_triangle : 2 * A + B = Real.pi)
  (h_isosceles : A = C) :
  (B > Real.pi / 2) ↔ (B^2 - 4 * A * C > 0) :=
sorry

end obtuse_triangle_iff_distinct_real_roots_l189_189588


namespace solve_for_x_l189_189048

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x (x : ℝ) : 2 * f x - 10 = f (x - 2) ↔ x = 3 :=
by
  sorry

end solve_for_x_l189_189048


namespace simplify_expr1_simplify_expr2_l189_189061

variable {a b : ℝ}

theorem simplify_expr1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 :=
by
  sorry

theorem simplify_expr2 : 2 * (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = 10 * a - 3 * a ^ 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l189_189061


namespace smallest_sum_of_squares_l189_189165

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 231) :
  x^2 + y^2 ≥ 281 :=
sorry

end smallest_sum_of_squares_l189_189165


namespace frood_game_least_n_l189_189377

theorem frood_game_least_n (n : ℕ) (h : n > 0) (drop_score : ℕ := n * (n + 1) / 2) (eat_score : ℕ := 15 * n) 
  : drop_score > eat_score ↔ n ≥ 30 :=
by
  sorry

end frood_game_least_n_l189_189377


namespace xyz_line_segments_total_length_l189_189631

noncomputable def total_length_XYZ : ℝ :=
  let length_X := 2 * Real.sqrt 2
  let length_Y := 2 + 2 * Real.sqrt 2
  let length_Z := 2 + Real.sqrt 2
  length_X + length_Y + length_Z

theorem xyz_line_segments_total_length : total_length_XYZ = 4 + 5 * Real.sqrt 2 := 
  sorry

end xyz_line_segments_total_length_l189_189631


namespace james_out_of_pocket_cost_l189_189982

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l189_189982


namespace find_value_of_m_l189_189354

noncomputable def m : ℤ := -2

theorem find_value_of_m (m : ℤ) :
  (m-2) ≠ 0 ∧ (m^2 - 3 = 1) → m = -2 :=
by
  intros h
  sorry

end find_value_of_m_l189_189354


namespace arithmetic_sequence_sum_l189_189537

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 2 + a 6 = 37) : 
  a 1 + a 3 + a 5 + a 7 = 74 :=
  sorry

end arithmetic_sequence_sum_l189_189537


namespace number_of_special_divisors_l189_189360

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end number_of_special_divisors_l189_189360


namespace a_c3_b3_equiv_zero_l189_189720

-- Definitions based on conditions
def cubic_eq_has_geom_progression_roots (a b c : ℝ) :=
  ∃ d q : ℝ, d ≠ 0 ∧ q ≠ 0 ∧ d + d * q + d * q^2 = -a ∧
    d^2 * q * (1 + q + q^2) = b ∧
    d^3 * q^3 = -c

-- Main theorem to prove
theorem a_c3_b3_equiv_zero (a b c : ℝ) :
  cubic_eq_has_geom_progression_roots a b c → a^3 * c - b^3 = 0 :=
by
  sorry

end a_c3_b3_equiv_zero_l189_189720


namespace determine_a_l189_189130

theorem determine_a
  (a b : ℝ)
  (P1 P2 : ℝ × ℝ)
  (direction_vector : ℝ × ℝ)
  (h1 : P1 = (-3, 4))
  (h2 : P2 = (4, -1))
  (h3 : direction_vector = (4 - (-3), -1 - 4))
  (h4 : b = a / 2)
  (h5 : direction_vector = (7, -5)) :
  a = -10 :=
sorry

end determine_a_l189_189130


namespace problem_solution_l189_189634

theorem problem_solution (a b c : ℝ) (h : (a / (36 - a)) + (b / (45 - b)) + (c / (54 - c)) = 8) :
    (4 / (36 - a)) + (5 / (45 - b)) + (6 / (54 - c)) = 11 / 9 := 
by
  sorry

end problem_solution_l189_189634


namespace geometric_sequence_common_ratio_l189_189404

theorem geometric_sequence_common_ratio
  (q : ℝ) (a_n : ℕ → ℝ)
  (h_inc : ∀ n, a_n (n + 1) = q * a_n n ∧ q > 1)
  (h_a2 : a_n 2 = 2)
  (h_a4_a3 : a_n 4 - a_n 3 = 4) : 
  q = 2 :=
sorry

end geometric_sequence_common_ratio_l189_189404


namespace perfect_squares_perfect_square_plus_one_l189_189149

theorem perfect_squares : (∃ n : ℕ, 2^n + 3 = (x : ℕ)^2) ↔ n = 0 ∨ n = 3 :=
by
  sorry

theorem perfect_square_plus_one : (∃ n : ℕ, 2^n + 1 = (x : ℕ)^2) ↔ n = 3 :=
by
  sorry

end perfect_squares_perfect_square_plus_one_l189_189149


namespace grocer_sales_l189_189002

theorem grocer_sales 
  (s1 s2 s3 s4 s5 s6 s7 s8 sales : ℝ)
  (h_sales_1 : s1 = 5420)
  (h_sales_2 : s2 = 5660)
  (h_sales_3 : s3 = 6200)
  (h_sales_4 : s4 = 6350)
  (h_sales_5 : s5 = 6500)
  (h_sales_6 : s6 = 6780)
  (h_sales_7 : s7 = 7000)
  (h_sales_8 : s8 = 7200)
  (h_avg : (5420 + 5660 + 6200 + 6350 + 6500 + 6780 + 7000 + 7200 + 2 * sales) / 10 = 6600) :
  sales = 9445 := 
  by 
  sorry

end grocer_sales_l189_189002


namespace stream_current_rate_l189_189148

theorem stream_current_rate (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (3 * r + w) + 2 = 18 / (3 * r - w)) : w = 3 :=
  sorry

end stream_current_rate_l189_189148


namespace product_of_values_l189_189825

-- Given definitions: N as a real number and R as a real constant
variables (N R : ℝ)

-- Condition
def condition : Prop := N - 5 / N = R

-- The proof statement
theorem product_of_values (h : condition N R) : ∀ (N1 N2 : ℝ), ((N1 - 5 / N1 = R) ∧ (N2 - 5 / N2 = R)) → (N1 * N2 = -5) :=
by sorry

end product_of_values_l189_189825


namespace find_z_l189_189780

open Complex

theorem find_z (z : ℂ) : (1 + 2*I) * z = 3 - I → z = (1/5) - (7/5)*I :=
by
  intro h
  sorry

end find_z_l189_189780


namespace smallest_integer_in_set_of_seven_l189_189361

theorem smallest_integer_in_set_of_seven (n : ℤ) (h : n + 6 < 3 * (n + 3)) : n = -1 :=
sorry

end smallest_integer_in_set_of_seven_l189_189361


namespace positive_expression_l189_189106

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 * (b + c) + a * (b^2 + c^2 - b * c) > 0 :=
by sorry

end positive_expression_l189_189106


namespace sum_possible_values_l189_189833

theorem sum_possible_values (M : ℝ) (h : M * (M - 6) = -5) : ∀ x ∈ {M | M * (M - 6) = -5}, x + (-x) = 6 :=
by sorry

end sum_possible_values_l189_189833


namespace complement_union_l189_189101

open Set

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement relative to U
def complement (A B : Set ℕ) : Set ℕ := { x ∈ B | x ∉ A }

-- The theorem we need to prove
theorem complement_union :
  complement (M ∪ N) U = {4} :=
by
  sorry

end complement_union_l189_189101


namespace part1_part2_l189_189561

-- Problem statement (1)
theorem part1 (a : ℝ) (h : a = -3) :
  (∀ x : ℝ, (x^2 + a * x + 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  { x : ℝ // (x^2 + a * x + 2) ≥ 1 - x^2 } = { x : ℝ // x ≤ 1 / 2 ∨ x ≥ 1 } :=
sorry

-- Problem statement (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x + 2) + x^2 + 1 = 2 * x^2 + a * x + 3) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ (2 * x^2 + a * x + 3) = 0) →
  -5 < a ∧ a < -2 * Real.sqrt 6 :=
sorry

end part1_part2_l189_189561


namespace limit_expression_l189_189144

theorem limit_expression :
  (∀ (n : ℕ), ∃ l : ℝ, 
    ∀ ε > 0, ∃ N : ℕ, n > N → 
      abs (( (↑(n) + 1)^3 - (↑(n) - 1)^3) / ((↑(n) + 1)^2 + (↑(n) - 1)^2) - l) < ε) 
  → l = 3 :=
sorry

end limit_expression_l189_189144


namespace line_through_point_with_slope_l189_189022

theorem line_through_point_with_slope (x y : ℝ) (h : y - 2 = -3 * (x - 1)) : 3 * x + y - 5 = 0 :=
sorry

example : 3 * 1 + 2 - 5 = 0 := by sorry

end line_through_point_with_slope_l189_189022


namespace largest_sum_is_7_over_12_l189_189206

-- Define the five sums
def sum1 : ℚ := 1/3 + 1/4
def sum2 : ℚ := 1/3 + 1/5
def sum3 : ℚ := 1/3 + 1/6
def sum4 : ℚ := 1/3 + 1/9
def sum5 : ℚ := 1/3 + 1/8

-- Define the problem statement
theorem largest_sum_is_7_over_12 : 
  max (max (max sum1 sum2) (max sum3 sum4)) sum5 = 7/12 := 
by
  sorry

end largest_sum_is_7_over_12_l189_189206


namespace part_1_part_3_500_units_part_3_1000_units_l189_189012

/-- Define the pricing function P as per the given conditions -/
def P (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x <= 550 then 62 - 0.02 * x
  else 51

/-- Verify that ordering 550 units results in a per-unit price of 51 yuan -/
theorem part_1 : P 550 = 51 := sorry

/-- Compute profit for given order quantities -/
def profit (x : ℕ) : ℝ :=
  x * (P x - 40)

/-- Verify that an order of 500 units results in a profit of 6000 yuan -/
theorem part_3_500_units : profit 500 = 6000 := sorry

/-- Verify that an order of 1000 units results in a profit of 11000 yuan -/
theorem part_3_1000_units : profit 1000 = 11000 := sorry

end part_1_part_3_500_units_part_3_1000_units_l189_189012


namespace new_tv_cost_l189_189168

/-
Mark bought his first TV which was 24 inches wide and 16 inches tall. It cost $672.
His new TV is 48 inches wide and 32 inches tall.
The first TV was $1 more expensive per square inch compared to his newest TV.
Prove that the cost of his new TV is $1152.
-/

theorem new_tv_cost :
  let width_first_tv := 24
  let height_first_tv := 16
  let cost_first_tv := 672
  let width_new_tv := 48
  let height_new_tv := 32
  let discount_per_square_inch := 1
  let area_first_tv := width_first_tv * height_first_tv
  let cost_per_square_inch_first_tv := cost_first_tv / area_first_tv
  let cost_per_square_inch_new_tv := cost_per_square_inch_first_tv - discount_per_square_inch
  let area_new_tv := width_new_tv * height_new_tv
  let cost_new_tv := cost_per_square_inch_new_tv * area_new_tv
  cost_new_tv = 1152 := by
  sorry

end new_tv_cost_l189_189168


namespace gym_monthly_cost_l189_189005

theorem gym_monthly_cost (down_payment total_cost total_months : ℕ) (h_down_payment : down_payment = 50) (h_total_cost : total_cost = 482) (h_total_months : total_months = 36) : 
  (total_cost - down_payment) / total_months = 12 := by 
  sorry

end gym_monthly_cost_l189_189005


namespace math_problem_solution_l189_189480

theorem math_problem_solution : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^y - 1 = y^x ∧ 2*x^y = y^x + 5 ∧ x = 2 ∧ y = 2 :=
by {
  sorry
}

end math_problem_solution_l189_189480


namespace inequality_solution_l189_189088

theorem inequality_solution (x : ℝ) (h : x ≠ -5) : 
  (x^2 - 25) / (x + 5) < 0 ↔ x ∈ Set.union (Set.Iio (-5)) (Set.Ioo (-5) 5) := 
by
  sorry

end inequality_solution_l189_189088


namespace range_of_m_l189_189194

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x-3| + |x+4| ≥ |2*m-1|) ↔ (-3 ≤ m ∧ m ≤ 4) := by
  sorry

end range_of_m_l189_189194


namespace population_size_in_15th_year_l189_189972

theorem population_size_in_15th_year
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * Real.logb 2 (x + 1))
  (h2 : y 1 = 100) :
  y 15 = 400 :=
by
  sorry

end population_size_in_15th_year_l189_189972


namespace like_terms_ratio_l189_189210

theorem like_terms_ratio (m n : ℕ) (h₁ : m - 2 = 2) (h₂ : 3 = 2 * n - 1) : m / n = 2 := 
by
  sorry

end like_terms_ratio_l189_189210


namespace find_angle_A_l189_189865

theorem find_angle_A (a b c A B C : ℝ)
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) :
  A = Real.pi / 6 :=
sorry

end find_angle_A_l189_189865


namespace distinct_x_intercepts_l189_189704

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (x - 5) * (x ^ 2 + 3 * x + 2) = 0) ∧ s.card = 3 :=
by {
  sorry
}

end distinct_x_intercepts_l189_189704


namespace minimum_number_of_kings_maximum_number_of_non_attacking_kings_l189_189419

-- Definitions for the chessboard and king placement problem

-- Problem (a): Minimum number of kings covering the board
def minimum_kings_covering_board (board_size : Nat) : Nat :=
  sorry

theorem minimum_number_of_kings (h : 6 = board_size) :
  minimum_kings_covering_board 6 = 4 := 
  sorry

-- Problem (b): Maximum number of non-attacking kings
def maximum_non_attacking_kings (board_size : Nat) : Nat :=
  sorry

theorem maximum_number_of_non_attacking_kings (h : 6 = board_size) :
  maximum_non_attacking_kings 6 = 9 :=
  sorry

end minimum_number_of_kings_maximum_number_of_non_attacking_kings_l189_189419


namespace original_number_is_two_thirds_l189_189675

theorem original_number_is_two_thirds (x : ℚ) (h : 1 + (1 / x) = 5 / 2) : x = 2 / 3 :=
by
  sorry

end original_number_is_two_thirds_l189_189675


namespace number_of_pairs_satisfying_equation_l189_189010

theorem number_of_pairs_satisfying_equation :
  ∃ n : ℕ, n = 4998 ∧ (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → (x, y) ≠ (0, 0)) ∧
  (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → ((x + 6 * y) = (3 * 5) ^ a ∧ (x + y) = (3 ^ (50 - a) * 5 ^ (50 - b)) ∨
        (x + 6 * y) = -(3 * 5) ^ a ∧ (x + y) = -(3 ^ (50 - a) * 5 ^ (50 - b)) → (a + b = 50))) :=
sorry

end number_of_pairs_satisfying_equation_l189_189010


namespace part1_part2_l189_189425

theorem part1 (a m n : ℕ) (ha : a > 1) (hdiv : a^m + 1 ∣ a^n + 1) : n ∣ m :=
sorry

theorem part2 (a b m n : ℕ) (ha : a > 1) (coprime_ab : Nat.gcd a b = 1) (hdiv : a^m + b^m ∣ a^n + b^n) : n ∣ m :=
sorry

end part1_part2_l189_189425


namespace juice_m_smoothie_l189_189067

/-- 
24 oz of juice p and 25 oz of juice v are mixed to make smoothies m and y. 
The ratio of p to v in smoothie m is 4 to 1 and that in y is 1 to 5. 
Prove that the amount of juice p in the smoothie m is 20 oz.
-/
theorem juice_m_smoothie (P_m P_y V_m V_y : ℕ)
  (h1 : P_m + P_y = 24)
  (h2 : V_m + V_y = 25)
  (h3 : 4 * V_m = P_m)
  (h4 : V_y = 5 * P_y) :
  P_m = 20 :=
sorry

end juice_m_smoothie_l189_189067


namespace plane_equation_l189_189436

theorem plane_equation 
  (s t : ℝ)
  (x y z : ℝ)
  (parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ)
  (plane_eq : ℝ × ℝ × ℝ → Prop) :
  parametric_plane s t = (2 + 2 * s - t, 1 + 2 * s, 4 - 3 * s + t) →
  plane_eq (x, y, z) ↔ 2 * x - 5 * y + 2 * z - 7 = 0 :=
by
  sorry

end plane_equation_l189_189436


namespace hypotenuse_length_l189_189637

-- Define the properties of the right-angled triangle
variables (α β γ : ℝ) (a b c : ℝ)
-- Right-angled triangle condition
axiom right_angled_triangle : α = 30 ∧ β = 60 ∧ γ = 90 → c = 2 * a

-- Given side opposite 30° angle is 6 cm
axiom side_opposite_30_is_6cm : a = 6

-- Proof that hypotenuse is 12 cm
theorem hypotenuse_length : c = 12 :=
by 
  sorry

end hypotenuse_length_l189_189637


namespace total_stamps_received_l189_189431

theorem total_stamps_received
  (initial_stamps : ℕ)
  (final_stamps : ℕ)
  (received_stamps : ℕ)
  (h_initial : initial_stamps = 34)
  (h_final : final_stamps = 61)
  (h_received : received_stamps = final_stamps - initial_stamps) :
  received_stamps = 27 :=
by 
  sorry

end total_stamps_received_l189_189431


namespace solve_for_y_l189_189114

theorem solve_for_y (x y : ℝ) (h₁ : x - y = 16) (h₂ : x + y = 4) : y = -6 := 
by 
  sorry

end solve_for_y_l189_189114


namespace percentage_decrease_hours_worked_l189_189173

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l189_189173


namespace museum_college_students_income_l189_189906

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l189_189906


namespace balloons_given_by_mom_l189_189069

def num_balloons_initial : ℕ := 26
def num_balloons_total : ℕ := 60

theorem balloons_given_by_mom :
  (num_balloons_total - num_balloons_initial) = 34 := 
by
  sorry

end balloons_given_by_mom_l189_189069


namespace part1_part2_l189_189070

-- Define set A
def A : Set ℝ := {x | 3 < x ∧ x < 6}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set complement in ℝ
def CR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- First part of the problem
theorem part1 :
  (A ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (CR A ∪ CR B = {x | x ≤ 3 ∨ x ≥ 6}) :=
sorry

-- Define set C depending on a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Second part of the problem
theorem part2 (a : ℝ) (h : B ∪ C a = B) :
  a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5) :=
sorry

end part1_part2_l189_189070


namespace expression_B_between_2_and_3_l189_189176

variable (a b : ℝ)
variable (h : 3 * a = 5 * b)

theorem expression_B_between_2_and_3 : 2 < (|a + b| / b) ∧ (|a + b| / b) < 3 :=
by sorry

end expression_B_between_2_and_3_l189_189176


namespace train_still_there_when_susan_arrives_l189_189064

-- Define the conditions and primary question
def time_between_1_and_2 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 60

def train_arrival := {t : ℝ // time_between_1_and_2 t}
def susan_arrival := {t : ℝ // time_between_1_and_2 t}

def train_present (train : train_arrival) (susan : susan_arrival) : Prop :=
  susan.val ≥ train.val ∧ susan.val ≤ (train.val + 30)

-- Define the probability calculation
noncomputable def probability_train_present : ℝ :=
  (30 * 30 + (30 * (60 - 30) * 2) / 2) / (60 * 60)

theorem train_still_there_when_susan_arrives :
  probability_train_present = 1 / 2 :=
sorry

end train_still_there_when_susan_arrives_l189_189064


namespace unique_function_satisfying_condition_l189_189824

theorem unique_function_satisfying_condition (k : ℕ) (hk : 0 < k) :
  ∀ f : ℕ → ℕ, (∀ m n : ℕ, 0 < m → 0 < n → f m + f n ∣ (m + n) ^ k) →
  ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
by
  sorry

end unique_function_satisfying_condition_l189_189824


namespace mom_prepared_pieces_l189_189261

-- Define the conditions
def jane_pieces : ℕ := 4
def total_eaters : ℕ := 3

-- Define the hypothesis that each of the eaters ate an equal number of pieces
def each_ate_equal (pieces : ℕ) : Prop := pieces = jane_pieces

-- The number of pieces Jane's mom prepared
theorem mom_prepared_pieces : total_eaters * jane_pieces = 12 :=
by
  -- Placeholder for actual proof
  sorry

end mom_prepared_pieces_l189_189261


namespace arithmetic_seq_fraction_l189_189205

theorem arithmetic_seq_fraction (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h2 : a 1 + a 10 = a 9) 
  (d_ne_zero : d ≠ 0) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / a 10 = 27 / 8 := 
sorry

end arithmetic_seq_fraction_l189_189205


namespace part_1_solution_set_part_2_a_range_l189_189571

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l189_189571


namespace cannot_buy_same_number_of_notebooks_l189_189023

theorem cannot_buy_same_number_of_notebooks
  (price_softcover : ℝ)
  (price_hardcover : ℝ)
  (notebooks_ming : ℝ)
  (notebooks_li : ℝ)
  (h1 : price_softcover = 12)
  (h2 : price_hardcover = 21)
  (h3 : price_hardcover = price_softcover + 1.2) :
  notebooks_ming = 12 / price_softcover ∧
  notebooks_li = 21 / price_hardcover →
  ¬ (notebooks_ming = notebooks_li) :=
by
  sorry

end cannot_buy_same_number_of_notebooks_l189_189023


namespace adam_money_ratio_l189_189878

theorem adam_money_ratio 
  (initial_dollars: ℕ) 
  (spent_dollars: ℕ) 
  (remaining_dollars: ℕ := initial_dollars - spent_dollars) 
  (ratio_numerator: ℕ := remaining_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (ratio_denominator: ℕ := spent_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (h_initial: initial_dollars = 91) 
  (h_spent: spent_dollars = 21) 
  (h_gcd: Nat.gcd (initial_dollars - spent_dollars) spent_dollars = 7) :
  ratio_numerator = 10 ∧ ratio_denominator = 3 := by
  sorry

end adam_money_ratio_l189_189878


namespace white_area_of_sign_l189_189152

theorem white_area_of_sign : 
  let total_area := 6 * 18
  let F_area := 2 * (4 * 1) + 6 * 1
  let O_area := 2 * (6 * 1) + 2 * (4 * 1)
  let D_area := 6 * 1 + 4 * 1 + 4 * 1
  let total_black_area := F_area + O_area + O_area + D_area
  total_area - total_black_area = 40 :=
by
  sorry

end white_area_of_sign_l189_189152


namespace meal_cost_l189_189155

theorem meal_cost (M : ℝ) (h1 : 3 * M + 15 = 45) : M = 10 :=
by
  sorry

end meal_cost_l189_189155


namespace total_distance_l189_189565

/--
A man completes a journey in 30 hours. He travels the first half of the journey at the rate of 20 km/hr and 
the second half at the rate of 10 km/hr. Prove that the total journey is 400 km.
-/
theorem total_distance (D : ℝ) (h : D / 40 + D / 20 = 30) :
  D = 400 :=
sorry

end total_distance_l189_189565


namespace pencils_to_sell_for_profit_l189_189787

theorem pencils_to_sell_for_profit 
    (total_pencils : ℕ) 
    (buy_price sell_price : ℝ) 
    (desired_profit : ℝ) 
    (h_total_pencils : total_pencils = 2000) 
    (h_buy_price : buy_price = 0.15) 
    (h_sell_price : sell_price = 0.30) 
    (h_desired_profit : desired_profit = 150) :
    total_pencils * buy_price + desired_profit = total_pencils * sell_price → total_pencils = 1500 :=
by
    sorry

end pencils_to_sell_for_profit_l189_189787


namespace area_of_sector_l189_189073

noncomputable def circleAreaAboveXAxisAndRightOfLine : ℝ :=
  let radius := 10
  let area_of_circle := Real.pi * radius^2
  area_of_circle / 4

theorem area_of_sector :
  circleAreaAboveXAxisAndRightOfLine = 25 * Real.pi := sorry

end area_of_sector_l189_189073


namespace sqrt_difference_l189_189610

theorem sqrt_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
sorry

end sqrt_difference_l189_189610


namespace find_cost_price_l189_189622

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l189_189622


namespace constant_COG_of_mercury_column_l189_189793

theorem constant_COG_of_mercury_column (L : ℝ) (A : ℝ) (beta_g : ℝ) (beta_m : ℝ) (alpha_g : ℝ) (x : ℝ) :
  L = 1 ∧ A = 1e-4 ∧ beta_g = 1 / 38700 ∧ beta_m = 1 / 5550 ∧ alpha_g = beta_g / 3 ∧
  x = (2 / (3 * 38700)) / ((1 / 5550) - (2 / 116100)) →
  x = 0.106 :=
by
  sorry

end constant_COG_of_mercury_column_l189_189793


namespace find_a2_l189_189119

variable (a : ℕ → ℝ) (d : ℝ)

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a n + d
axiom common_diff : d = 2
axiom geometric_mean : (a 4) ^ 2 = (a 5) * (a 2)

theorem find_a2 : a 2 = -8 := 
by 
  sorry

end find_a2_l189_189119


namespace laura_annual_income_l189_189729

variable (p : ℝ) -- percentage p
variable (A T : ℝ) -- annual income A and total income tax T

def tax1 : ℝ := 0.01 * p * 35000
def tax2 : ℝ := 0.01 * (p + 3) * (A - 35000)
def tax3 : ℝ := 0.01 * (p + 5) * (A - 55000)

theorem laura_annual_income (h_cond1 : A > 55000)
  (h_tax : T = 350 * p + 600 + 0.01 * (p + 5) * (A - 55000))
  (h_paid_tax : T = (0.01 * (p + 0.45)) * A):
  A = 75000 := by
  sorry

end laura_annual_income_l189_189729


namespace number_of_paths_l189_189099

theorem number_of_paths (n m : ℕ) (h : m ≤ n) : 
  ∃ paths : ℕ, paths = Nat.choose n m := 
sorry

end number_of_paths_l189_189099


namespace fewer_columns_after_rearrangement_l189_189663

theorem fewer_columns_after_rearrangement : 
  ∀ (T R R' C C' fewer_columns : ℕ),
    T = 30 → 
    R = 5 → 
    R' = R + 4 →
    C * R = T →
    C' * R' = T →
    fewer_columns = C - C' →
    fewer_columns = 3 :=
by
  intros T R R' C C' fewer_columns hT hR hR' hCR hC'R' hfewer_columns
  -- sorry to skip the proof part
  sorry

end fewer_columns_after_rearrangement_l189_189663


namespace garden_bed_length_l189_189539

theorem garden_bed_length (total_area : ℕ) (garden_area : ℕ) (width : ℕ) (n : ℕ)
  (total_area_eq : total_area = 42)
  (garden_area_eq : garden_area = 9)
  (num_gardens_eq : n = 2)
  (width_eq : width = 3)
  (lhs_eq : lhs = total_area - n * garden_area)
  (area_to_length_eq : length = lhs / width) :
  length = 8 := by
  sorry

end garden_bed_length_l189_189539


namespace sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l189_189373

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) : (x + 1) * (x - 2) > 0 → x > 2 :=
by sorry

theorem converse_not_true 
  (x : ℝ) : x > 2 → (x + 1) * (x - 2) > 0 :=
by sorry

theorem cond_x_gt_2_iff_sufficient_not_necessary 
  (x : ℝ) : (x > 2 → (x + 1) * (x - 2) > 0) ∧ 
            ((x + 1) * (x - 2) > 0 → x > 2) :=
by sorry

end sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l189_189373


namespace total_puppies_adopted_l189_189543

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l189_189543


namespace part1_f_inequality_part2_a_range_l189_189900

open Real

-- Proof Problem 1
theorem part1_f_inequality (x : ℝ) : 
    (|x - 1| + |x + 1| ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5) :=
sorry

-- Proof Problem 2
theorem part2_a_range (a : ℝ) : 
    (∀ x : ℝ, |x - 1| + |x - a| ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end part1_f_inequality_part2_a_range_l189_189900


namespace sample_size_is_correct_l189_189590

-- Define the school and selection conditions
def total_classes := 40
def students_per_class := 50

-- Given condition
def selected_students := 150

-- Theorem statement
theorem sample_size_is_correct : selected_students = 150 := 
by 
  sorry

end sample_size_is_correct_l189_189590


namespace negation_of_proposition_l189_189642

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬ ∃ x : ℝ, x^2 + 1 < 0) :=
by
  sorry

end negation_of_proposition_l189_189642


namespace PersonX_job_completed_time_l189_189363

-- Definitions for conditions
def Dan_job_time := 15 -- hours
def PersonX_job_time (x : ℝ) := x -- hours
def Dan_work_time := 3 -- hours
def PersonX_remaining_work_time := 8 -- hours

-- Given Dan's and Person X's work time, prove Person X's job completion time
theorem PersonX_job_completed_time (x : ℝ) (h1 : Dan_job_time > 0)
    (h2 : PersonX_job_time x > 0)
    (h3 : Dan_work_time > 0)
    (h4 : PersonX_remaining_work_time * (1 - Dan_work_time / Dan_job_time) = 1 / x * 8) :
    x = 10 :=
  sorry

end PersonX_job_completed_time_l189_189363


namespace line_intercepts_and_slope_l189_189090

theorem line_intercepts_and_slope :
  ∀ (x y : ℝ), (4 * x - 5 * y - 20 = 0) → 
  ∃ (x_intercept : ℝ) (y_intercept : ℝ) (slope : ℝ), 
    x_intercept = 5 ∧ y_intercept = -4 ∧ slope = 4 / 5 :=
by
  sorry

end line_intercepts_and_slope_l189_189090


namespace math_problem_l189_189953

theorem math_problem :
  |(-3 : ℝ)| - Real.sqrt 8 - (1/2 : ℝ)⁻¹ + 2 * Real.cos (Real.pi / 4) = 1 - Real.sqrt 2 :=
by
  sorry

end math_problem_l189_189953


namespace polynomial_square_l189_189416

theorem polynomial_square (x : ℝ) : x^4 + 2*x^3 - 2*x^2 - 4*x - 5 = y^2 → x = 3 ∨ x = -3 := by
  sorry

end polynomial_square_l189_189416


namespace number_of_cows_is_six_l189_189882

variable (C H : Nat) -- C for cows and H for chickens

-- Number of legs is 12 more than twice the number of heads.
def cows_count_condition : Prop :=
  4 * C + 2 * H = 2 * (C + H) + 12

theorem number_of_cows_is_six (h : cows_count_condition C H) : C = 6 :=
sorry

end number_of_cows_is_six_l189_189882


namespace surface_area_of_interior_of_box_l189_189501

-- Definitions from conditions in a)
def length : ℕ := 25
def width : ℕ := 40
def cut_side : ℕ := 4

-- The proof statement we need to prove, using the correct answer from b)
theorem surface_area_of_interior_of_box : 
  (length - 2 * cut_side) * (width - 2 * cut_side) + 2 * (cut_side * (length + width - 2 * cut_side)) = 936 :=
by
  sorry

end surface_area_of_interior_of_box_l189_189501


namespace worker_efficiency_l189_189405

theorem worker_efficiency (Wq : ℝ) (x : ℝ) : 
  (1.4 * (1 / x) = 1 / (1.4 * x)) → 
  (14 * (1 / x + 1 / (1.4 * x)) = 1) → 
  x = 24 :=
by
  sorry

end worker_efficiency_l189_189405


namespace min_sqrt_diff_l189_189076

theorem min_sqrt_diff (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧
    ∀ a b : ℕ, (a ≤ b) → (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) → 
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y) ≤ (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) := 
by 
  -- Proof to be filled in
  sorry

end min_sqrt_diff_l189_189076


namespace sin_eq_sqrt3_div_2_l189_189875

open Real

theorem sin_eq_sqrt3_div_2 (theta : ℝ) :
  sin theta = (sqrt 3) / 2 ↔ (∃ k : ℤ, theta = π/3 + 2*k*π ∨ theta = 2*π/3 + 2*k*π) :=
by
  sorry

end sin_eq_sqrt3_div_2_l189_189875


namespace transformed_sum_of_coordinates_l189_189556

theorem transformed_sum_of_coordinates (g : ℝ → ℝ) (h : g 8 = 5) :
  let x := 8 / 3
  let y := 14 / 9
  3 * y = g (3 * x) / 3 + 3 ∧ (x + y = 38 / 9) :=
by
  sorry

end transformed_sum_of_coordinates_l189_189556


namespace brian_stones_l189_189175

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l189_189175


namespace hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l189_189513

theorem hundredth_odd_positive_integer_equals_199 : (2 * 100 - 1 = 199) :=
by {
  sorry
}

theorem even_integer_following_199_equals_200 : (199 + 1 = 200) :=
by {
  sorry
}

end hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l189_189513


namespace odd_number_representation_l189_189573

theorem odd_number_representation (n : ℤ) : 
  (∃ m : ℤ, 2 * m + 1 = 2 * n + 3) ∧ (¬ ∃ m : ℤ, 2 * m + 1 = 4 * n - 1) :=
by
  -- Proof steps would go here
  sorry

end odd_number_representation_l189_189573


namespace binomial_coefficient_times_two_l189_189948

theorem binomial_coefficient_times_two : 2 * Nat.choose 8 5 = 112 := 
by 
  -- The proof is omitted here
  sorry

end binomial_coefficient_times_two_l189_189948


namespace pencils_per_person_l189_189399

theorem pencils_per_person (x : ℕ) (h : 3 * x = 24) : x = 8 :=
by
  -- sorry we are skipping the actual proof
  sorry

end pencils_per_person_l189_189399


namespace reciprocal_of_neg_two_l189_189487

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l189_189487


namespace age_of_female_employee_when_hired_l189_189270

-- Defining the conditions
def hired_year : ℕ := 1989
def retirement_year : ℕ := 2008
def sum_age_employment : ℕ := 70

-- Given the conditions we found that years of employment (Y):
def years_of_employment : ℕ := retirement_year - hired_year -- 19

-- Defining the age when hired (A)
def age_when_hired : ℕ := sum_age_employment - years_of_employment -- 51

-- Now we need to prove
theorem age_of_female_employee_when_hired : age_when_hired = 51 :=
by
  -- Here should be the proof steps, but we use sorry for now
  sorry

end age_of_female_employee_when_hired_l189_189270


namespace triangle_right_angle_and_m_values_l189_189481

open Real

-- Definitions and conditions
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y + 6 = 0
def line_AC (x y : ℝ) : Prop := 2 * x + 3 * y - 22 = 0
def line_BC (x y m : ℝ) : Prop := 3 * x + 4 * y - m = 0

-- Prove the shape and value of m when the height from BC is 1
theorem triangle_right_angle_and_m_values :
  (∃ (x y : ℝ), line_AB x y ∧ line_AC x y ∧ line_AB x y ∧ (-3/2) ≠ (2/3)) ∧
  (∀ x y, line_AB x y → line_AC x y → 3 * x + 4 * y - 25 = 0 ∨ 3 * x + 4 * y - 35 = 0) := 
sorry

end triangle_right_angle_and_m_values_l189_189481


namespace find_a_b_sum_l189_189260

-- Conditions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f_prime (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem find_a_b_sum (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a + b = 7 := 
sorry

end find_a_b_sum_l189_189260


namespace parallel_lines_coplanar_l189_189624

axiom Plane : Type
axiom Point : Type
axiom Line : Type

axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

axiom α : Plane
axiom β : Plane

axiom in_plane (p : Point) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop
axiom parallel_line (l1 l2 : Line) : Prop
axiom line_through (P Q : Point) : Line
axiom coplanar (P Q R S : Point) : Prop

-- Conditions
axiom A_in_α : in_plane A α
axiom C_in_α : in_plane C α
axiom B_in_β : in_plane B β
axiom D_in_β : in_plane D β
axiom α_parallel_β : parallel_plane α β

-- Statement
theorem parallel_lines_coplanar :
  parallel_line (line_through A C) (line_through B D) ↔ coplanar A B C D :=
sorry

end parallel_lines_coplanar_l189_189624


namespace ratio_Sydney_to_Sherry_l189_189936

variable (Randolph_age Sydney_age Sherry_age : ℕ)

-- Conditions
axiom Randolph_older_than_Sydney : Randolph_age = Sydney_age + 5
axiom Sherry_age_is_25 : Sherry_age = 25
axiom Randolph_age_is_55 : Randolph_age = 55

-- Theorem to prove
theorem ratio_Sydney_to_Sherry : (Sydney_age : ℝ) / (Sherry_age : ℝ) = 2 := by
  sorry

end ratio_Sydney_to_Sherry_l189_189936


namespace range_of_a_l189_189127

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, (2 * a - 1 ≤ x ∧ x ≤ a + 3)) →
  (-1 ≤ a ∧ a ≤ 0) :=
by
  -- Prove the theorem
  sorry

end range_of_a_l189_189127


namespace taxi_ride_cost_l189_189059

-- Define the base fare
def base_fare : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the distance traveled
def distance : ℝ := 8.00

-- Define the total cost function
def total_cost (base : ℝ) (per_mile : ℝ) (miles : ℝ) : ℝ :=
  base + (per_mile * miles)

-- The statement to prove: the total cost of an 8-mile taxi ride
theorem taxi_ride_cost : total_cost base_fare cost_per_mile distance = 4.40 :=
by
sorry

end taxi_ride_cost_l189_189059


namespace equality_holds_iff_l189_189596

theorem equality_holds_iff (k t x y z : ℤ) (h_arith_prog : x + z = 2 * y) :
  (k * y^3 = x^3 + z^3) ↔ (k = 2 * (3 * t^2 + 1)) := by
  sorry

end equality_holds_iff_l189_189596


namespace quadratic_root_k_eq_one_l189_189083

theorem quadratic_root_k_eq_one
  (k : ℝ)
  (h₀ : (k + 3) ≠ 0)
  (h₁ : ∃ x : ℝ, (x = 0) ∧ ((k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0)) :
  k = 1 :=
by
  sorry

end quadratic_root_k_eq_one_l189_189083


namespace y_is_80_percent_less_than_x_l189_189764

theorem y_is_80_percent_less_than_x (x y : ℝ) (h : x = 5 * y) : ((x - y) / x) * 100 = 80 :=
by sorry

end y_is_80_percent_less_than_x_l189_189764


namespace first_number_less_than_twice_second_l189_189322

theorem first_number_less_than_twice_second (x y z : ℕ) : 
  x + y = 50 ∧ y = 19 ∧ x = 2 * y - z → z = 7 :=
by sorry

end first_number_less_than_twice_second_l189_189322


namespace part_I_part_II_l189_189192

noncomputable def a (n : Nat) : Nat := sorry

def is_odd (n : Nat) : Prop := n % 2 = 1

theorem part_I
  (h : a 1 = 19) :
  a 2014 = 98 := by
  sorry

theorem part_II
  (h1: ∀ n : Nat, is_odd (a n))
  (h2: ∀ n m : Nat, a n = a m) -- constant sequence
  (h3: ∀ n : Nat, a n > 1) :
  ∃ k : Nat, a k = 5 := by
  sorry


end part_I_part_II_l189_189192


namespace can_construct_length_one_l189_189331

noncomputable def possible_to_construct_length_one_by_folding (n : ℕ) : Prop :=
  ∃ k ≤ 10, ∃ (segment_constructed : ℝ), segment_constructed = 1

theorem can_construct_length_one : possible_to_construct_length_one_by_folding 2016 :=
by sorry

end can_construct_length_one_l189_189331


namespace problem_I_solution_set_problem_II_range_a_l189_189707

-- Problem (I)
-- Given f(x) = |x-1|, g(x) = 2|x+1|, and a=1, prove that the inequality f(x) - g(x) > 1 has the solution set (-1, -1/3)
theorem problem_I_solution_set (x: ℝ) : abs (x - 1) - 2 * abs (x + 1) > 1 ↔ -1 < x ∧ x < -1 / 3 := 
by sorry

-- Problem (II)
-- Given f(x) = |x-1|, g(x) = 2|x+a|, prove that if 2f(x) + g(x) ≤ (a + 1)^2 has a solution for x,
-- then a ∈ (-∞, -3] ∪ [1, ∞)
theorem problem_II_range_a (a x: ℝ) (h : ∃ x, 2 * abs (x - 1) + 2 * abs (x + a) ≤ (a + 1) ^ 2) : 
  a ≤ -3 ∨ a ≥ 1 := 
by sorry

end problem_I_solution_set_problem_II_range_a_l189_189707


namespace log_ride_cost_l189_189291

noncomputable def cost_of_log_ride (ferris_wheel : ℕ) (roller_coaster : ℕ) (initial_tickets : ℕ) (additional_tickets : ℕ) : ℕ :=
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  total_needed - total_known

theorem log_ride_cost :
  cost_of_log_ride 6 5 2 16 = 7 :=
by
  -- specify the values for ferris_wheel, roller_coaster, initial_tickets, additional_tickets
  let ferris_wheel := 6
  let roller_coaster := 5
  let initial_tickets := 2
  let additional_tickets := 16
  -- calculate the cost of the log ride
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  let log_ride := total_needed - total_known
  -- assert that the cost of the log ride is 7
  have : log_ride = 7 := by
    -- use arithmetic to justify the answer
    sorry
  exact this

end log_ride_cost_l189_189291


namespace inscribed_circle_radius_eq_four_l189_189283

theorem inscribed_circle_radius_eq_four
  (A p s r : ℝ)
  (hA : A = 2 * p)
  (hp : p = 2 * s)
  (hArea : A = r * s) :
  r = 4 :=
by
  -- Proof would go here.
  sorry

end inscribed_circle_radius_eq_four_l189_189283


namespace area_of_EFGH_l189_189869

-- Define the dimensions of the smaller rectangles
def smaller_rectangle_short_side : ℕ := 7
def smaller_rectangle_long_side : ℕ := 2 * smaller_rectangle_short_side

-- Define the configuration of rectangles
def width_EFGH : ℕ := 2 * smaller_rectangle_short_side
def length_EFGH : ℕ := smaller_rectangle_long_side

-- Prove that the area of rectangle EFGH is 196 square feet
theorem area_of_EFGH : width_EFGH * length_EFGH = 196 := by
  sorry

end area_of_EFGH_l189_189869


namespace arithmetic_sequence_term_number_l189_189443

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end arithmetic_sequence_term_number_l189_189443


namespace Hilt_payment_l189_189558

def total_cost : ℝ := 2.05
def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10

theorem Hilt_payment (n : ℕ) (h : n_n = n ∧ n_d = n) 
  (h_nickel : ℝ := n * nickel_value)
  (h_dime : ℝ := n * dime_value): 
  (n * nickel_value + n * dime_value = total_cost) 
  →  n = 14 :=
by {
  sorry
}

end Hilt_payment_l189_189558


namespace simplify_expression_l189_189319

theorem simplify_expression (a : ℝ) (h : a ≠ -1) : a - 1 + 1 / (a + 1) = a^2 / (a + 1) :=
  sorry

end simplify_expression_l189_189319


namespace find_vertex_D_l189_189872

noncomputable def quadrilateral_vertices : Prop :=
  let A : (ℤ × ℤ) := (-1, -2)
  let B : (ℤ × ℤ) := (3, 1)
  let C : (ℤ × ℤ) := (0, 2)
  A ≠ B ∧ A ≠ C ∧ B ≠ C

theorem find_vertex_D (A B C D : ℤ × ℤ) (h_quad : quadrilateral_vertices) :
    (A = (-1, -2)) →
    (B = (3, 1)) →
    (C = (0, 2)) →
    (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) →
    D = (-4, -1) :=
by
  sorry

end find_vertex_D_l189_189872


namespace total_volume_correct_l189_189676

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2812

-- Define the target volume
def total_volume_of_water : ℕ := 11248

-- The theorem to be proved
theorem total_volume_correct : volume_of_hemisphere * number_of_hemispheres = total_volume_of_water :=
by
  sorry

end total_volume_correct_l189_189676


namespace alternating_binomial_sum_l189_189455

open BigOperators Finset

theorem alternating_binomial_sum :
  ∑ k in range 34, (-1 : ℤ)^k * (Nat.choose 99 (3 * k)) = -1 := by
  sorry

end alternating_binomial_sum_l189_189455


namespace books_sold_l189_189082

theorem books_sold (initial_books remaining_books sold_books : ℕ):
  initial_books = 33 → 
  remaining_books = 7 → 
  sold_books = initial_books - remaining_books → 
  sold_books = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end books_sold_l189_189082


namespace hyperbola_eccentricity_l189_189466

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_angle : b / a = Real.sqrt 3 / 3) :
    let e := Real.sqrt (1 + (b / a)^2)
    e = 2 * Real.sqrt 3 / 3 := 
sorry

end hyperbola_eccentricity_l189_189466


namespace alice_bob_meeting_point_l189_189191

def meet_same_point (turns : ℕ) : Prop :=
  ∃ n : ℕ, turns = 2 * n ∧ 18 ∣ (7 * n - (7 * n + n))

theorem alice_bob_meeting_point :
  meet_same_point 36 :=
by
  sorry

end alice_bob_meeting_point_l189_189191


namespace Matt_jumped_for_10_minutes_l189_189806

def Matt_skips_per_second : ℕ := 3

def total_skips : ℕ := 1800

def minutes_jumped (m : ℕ) : Prop :=
  m * (Matt_skips_per_second * 60) = total_skips

theorem Matt_jumped_for_10_minutes : minutes_jumped 10 :=
by
  sorry

end Matt_jumped_for_10_minutes_l189_189806


namespace number_of_ordered_pairs_l189_189366

noncomputable def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k

theorem number_of_ordered_pairs :
  (∃ (n : ℕ), n = 29 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ x ≤ 2020 ∧ y ≤ 2020 →
    is_power_of_prime (3 * x^2 + 10 * x * y + 3 * y^2) → n = 29) :=
by
  sorry

end number_of_ordered_pairs_l189_189366


namespace initial_sugar_weight_l189_189640

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l189_189640


namespace sin_neg_pi_over_three_l189_189268

theorem sin_neg_pi_over_three : Real.sin (-Real.pi / 3) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_pi_over_three_l189_189268


namespace smallest_t_eq_3_over_4_l189_189743

theorem smallest_t_eq_3_over_4 (t : ℝ) :
  (∀ t : ℝ,
    (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2 → t >= (3 / 4)) ∧
  (∃ t₀ : ℝ, (16 * t₀^3 - 49 * t₀^2 + 35 * t₀ - 6) / (4 * t₀ - 3) + 7 * t₀ = 8 * t₀ - 2 ∧ t₀ = (3 / 4)) :=
sorry

end smallest_t_eq_3_over_4_l189_189743


namespace notebook_problem_l189_189000

theorem notebook_problem :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 2 * x + 5 * y + 6 * z = 62 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x = 14 :=
by
  sorry

end notebook_problem_l189_189000


namespace largest_divisor_of_n5_minus_n_l189_189515

theorem largest_divisor_of_n5_minus_n (n : ℤ) : 
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n^5 - n)) ∧ d = 30 :=
sorry

end largest_divisor_of_n5_minus_n_l189_189515


namespace gcd_45_81_63_l189_189293

theorem gcd_45_81_63 : Nat.gcd 45 (Nat.gcd 81 63) = 9 := 
sorry

end gcd_45_81_63_l189_189293


namespace carol_total_points_l189_189456

/-- Conditions -/
def first_round_points : ℤ := 17
def second_round_points : ℤ := 6
def last_round_points : ℤ := -16

/-- Proof problem statement -/
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l189_189456


namespace no_solution_5x_plus_2_eq_17y_l189_189800

theorem no_solution_5x_plus_2_eq_17y :
  ¬∃ (x y : ℕ), 5^x + 2 = 17^y :=
sorry

end no_solution_5x_plus_2_eq_17y_l189_189800


namespace net_investment_change_l189_189652

variable (I : ℝ)

def first_year_increase (I : ℝ) : ℝ := I * 1.75
def second_year_decrease (W : ℝ) : ℝ := W * 0.70

theorem net_investment_change : 
  let I' := first_year_increase 100 
  let I'' := second_year_decrease I' 
  I'' - 100 = 22.50 :=
by
  sorry

end net_investment_change_l189_189652


namespace tan_addition_example_l189_189592

theorem tan_addition_example (x : ℝ) (h : Real.tan x = 1/3) : 
  Real.tan (x + π/3) = 2 + 5 * Real.sqrt 3 / 3 := 
by 
  sorry

end tan_addition_example_l189_189592


namespace complement_intersection_l189_189601

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {1, 2, 3})
variable (hB : B = {2, 3, 4})

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4} :=
by
  sorry

end complement_intersection_l189_189601


namespace min_value_l189_189318

variable {α : Type*} [LinearOrderedField α]

-- Define a geometric sequence with strictly positive terms
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ (q : α), q > 0 ∧ ∀ n, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → α) (S : ℕ → α)
variables (h_geom : is_geometric_sequence a)
variables (h_pos : ∀ n, a n > 0)
variables (h_a23 : a 2 * a 6 = 4) (h_a3 : a 3 = 1)

-- Sum of the first n terms of a geometric sequence
def sum_first_n (a : ℕ → α) (n : ℕ) : α :=
  if n = 0 then 0
  else a 0 * ((1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0)))

-- Statement of the theorem
theorem min_value (a : ℕ → α) (S : ℕ → α) 
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a23 : a 2 * a 6 = 4)
  (h_a3 : a 3 = 1)
  (h_Sn : ∀ n, S n = sum_first_n a n) :
  ∃ n, n = 3 ∧ (S n + 9 / 4) ^ 2 / (2 * a n) = 8 :=
sorry

end min_value_l189_189318


namespace problem_statement_l189_189880

-- Definitions from the problem conditions
variable (r : ℝ) (A B C : ℝ)

-- Problem condition that A, B are endpoints of the diameter of the circle
-- Defining the length AB being the diameter -> length AB = 2r
def AB := 2 * r

-- Condition that ABC is inscribed in a circle and AB is the diameter implies the angle ACB = 90°
-- Using Thales' theorem we know that A, B, C satisfy certain geometric properties in a right triangle
-- AC and BC are the other two sides with H right angle at C.

-- Proving the target equation
theorem problem_statement (h : C ≠ A ∧ C ≠ B) : (AC + BC)^2 ≤ 8 * r^2 := 
sorry


end problem_statement_l189_189880


namespace line_segment_endpoint_l189_189594

theorem line_segment_endpoint (x : ℝ) (h1 : (x - 3)^2 + 36 = 289) (h2 : x < 0) : x = 3 - Real.sqrt 253 :=
sorry

end line_segment_endpoint_l189_189594


namespace trapezoid_ratio_l189_189248

-- Define the isosceles trapezoid properties and the point inside it
noncomputable def isosceles_trapezoid (r s : ℝ) (hr : r > s) (triangle_areas : List ℝ) : Prop :=
  triangle_areas = [2, 3, 4, 5]

-- Define the problem statement
theorem trapezoid_ratio (r s : ℝ) (hr : r > s) (areas : List ℝ) (hareas : isosceles_trapezoid r s hr areas) :
  r / s = 2 + Real.sqrt 2 := sorry

end trapezoid_ratio_l189_189248


namespace staircase_toothpicks_l189_189459

theorem staircase_toothpicks :
  ∀ (T : ℕ → ℕ), 
  (T 4 = 28) →
  (∀ n : ℕ, T (n + 1) = T n + (12 + 3 * (n - 3))) →
  T 6 - T 4 = 33 :=
by
  intros T T4_step H_increase
  -- proof goes here
  sorry

end staircase_toothpicks_l189_189459


namespace airplane_speed_l189_189810

noncomputable def distance : ℝ := 378.6   -- Distance in km
noncomputable def time : ℝ := 693.5       -- Time in seconds

noncomputable def altitude : ℝ := 10      -- Altitude in km
noncomputable def earth_radius : ℝ := 6370 -- Earth's radius in km

noncomputable def speed : ℝ := distance / time * 3600  -- Speed in km/h
noncomputable def adjusted_speed : ℝ := speed * (earth_radius + altitude) / earth_radius

noncomputable def min_distance : ℝ := 378.6 - 0.03     -- Minimum possible distance in km
noncomputable def max_distance : ℝ := 378.6 + 0.03     -- Maximum possible distance in km
noncomputable def min_time : ℝ := 693.5 - 1.5          -- Minimum possible time in s
noncomputable def max_time : ℝ := 693.5 + 1.5          -- Maximum possible time in s

noncomputable def max_speed : ℝ := max_distance / min_time * 3600 -- Max speed with uncertainty
noncomputable def min_speed : ℝ := min_distance / max_time * 3600 -- Min speed with uncertainty

theorem airplane_speed :
  1960 < adjusted_speed ∧ adjusted_speed < 1970 :=
by
  sorry

end airplane_speed_l189_189810


namespace determine_c_l189_189103

-- Define the points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (0, 4)

-- Define the direction vector calculation
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Define the target direction vector form
def target_direction_vector (c : ℝ) : ℝ × ℝ := (3, c)

-- Theorem stating that the calculated direction vector equals the target direction vector when c = 3
theorem determine_c : direction_vector = target_direction_vector 3 :=
by
  -- Proof omitted
  sorry

end determine_c_l189_189103


namespace miles_to_burger_restaurant_l189_189932

-- Definitions and conditions
def miles_per_gallon : ℕ := 19
def gallons_of_gas : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_friend_house : ℕ := 4
def miles_to_home : ℕ := 11
def total_gas_distance := miles_per_gallon * gallons_of_gas
def total_known_distances := miles_to_school + miles_to_softball_park + miles_to_friend_house + miles_to_home

-- Problem statement to prove
theorem miles_to_burger_restaurant :
  ∃ (miles_to_burger_restaurant : ℕ), 
  total_gas_distance = total_known_distances + miles_to_burger_restaurant ∧ miles_to_burger_restaurant = 2 := 
by
  sorry

end miles_to_burger_restaurant_l189_189932


namespace solution_l189_189871

theorem solution (x : ℝ) : (x = -2/5) → (x < x^3 ∧ x^3 < x^2) :=
by
  intro h
  rw [h]
  -- sorry to skip the proof
  sorry

end solution_l189_189871


namespace sum_roots_x_squared_minus_5x_plus_6_eq_5_l189_189576

noncomputable def sum_of_roots (a b c : Real) : Real :=
  -b / a

theorem sum_roots_x_squared_minus_5x_plus_6_eq_5 :
  sum_of_roots 1 (-5) 6 = 5 := by
  sorry

end sum_roots_x_squared_minus_5x_plus_6_eq_5_l189_189576


namespace a_2_correct_l189_189653

noncomputable def a_2_value (a a1 a2 a3 : ℝ) : Prop :=
∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3

theorem a_2_correct (a a1 a2 a3 : ℝ) (h : a_2_value a a1 a2 a3) : a2 = 6 :=
sorry

end a_2_correct_l189_189653


namespace find_n_l189_189334

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n % 9 = 4897 % 9 ∧ n = 1 :=
by
  use 1
  sorry

end find_n_l189_189334


namespace exist_odd_distinct_integers_l189_189356

theorem exist_odd_distinct_integers (n : ℕ) (h1 : n % 2 = 1) (h2 : n > 3) (h3 : n % 3 ≠ 0) : 
  ∃ a b c : ℕ, a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  3 / (n : ℚ) = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) :=
sorry

end exist_odd_distinct_integers_l189_189356


namespace total_monthly_bill_working_from_home_l189_189861

-- Definitions based on conditions
def original_bill : ℝ := 60
def increase_rate : ℝ := 0.45
def additional_internet_cost : ℝ := 25
def additional_cloud_cost : ℝ := 15

-- The theorem to prove
theorem total_monthly_bill_working_from_home : 
  original_bill * (1 + increase_rate) + additional_internet_cost + additional_cloud_cost = 127 := by
  sorry

end total_monthly_bill_working_from_home_l189_189861


namespace a_runs_4_times_faster_than_b_l189_189495

theorem a_runs_4_times_faster_than_b (v_A v_B : ℝ) (k : ℝ) 
    (h1 : v_A = k * v_B) 
    (h2 : 92 / v_A = 23 / v_B) : 
    k = 4 := 
sorry

end a_runs_4_times_faster_than_b_l189_189495


namespace B_cycling_speed_l189_189747

theorem B_cycling_speed (v : ℝ) : 
  (∀ (t : ℝ), 10 * t + 30 = B_start_distance) ∧ 
  (B_start_distance = 60) ∧ 
  (t = 3) →
  v = 20 :=
sorry

end B_cycling_speed_l189_189747


namespace arc_length_l189_189863

theorem arc_length (r α : ℝ) (h1 : r = 3) (h2 : α = π / 3) : r * α = π :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is the step where actual simplification and calculation will happen

end arc_length_l189_189863


namespace find_functions_l189_189410

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_functions
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_1 : ∀ x : ℝ, |x| ≤ 1 → |f a b c x| ≤ 1)
  (h_2 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ ∀ x : ℝ, |x| ≤ 1 → |f' a b x₀| ≥ |f' a b x| )
  (K : ℝ)
  (h_3 : ∃ x₀ : ℝ, |x₀| ≤ 1 ∧ |f' a b x₀| = K) :
  (f a b c = fun x ↦ 2 * x^2 - 1) ∨ (f a b c = fun x ↦ -2 * x^2 + 1) ∧ K = 4 := 
sorry

end find_functions_l189_189410


namespace range_of_m_for_inequality_l189_189973

theorem range_of_m_for_inequality (x y m : ℝ) :
  (∀ x y : ℝ, 3*x^2 + y^2 ≥ m * x * (x + y)) ↔ (-6 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_for_inequality_l189_189973


namespace rackets_packed_l189_189831

theorem rackets_packed (total_cartons : ℕ) (cartons_3 : ℕ) (cartons_2 : ℕ) 
  (h1 : total_cartons = 38) 
  (h2 : cartons_3 = 24) 
  (h3 : cartons_2 = total_cartons - cartons_3) :
  3 * cartons_3 + 2 * cartons_2 = 100 := 
by
  -- The proof is omitted
  sorry

end rackets_packed_l189_189831


namespace price_difference_l189_189830

def P := ℝ

def Coupon_A_savings (P : ℝ) := 0.20 * P
def Coupon_B_savings : ℝ := 40
def Coupon_C_savings (P : ℝ) := 0.30 * (P - 120) + 20

def Coupon_A_geq_Coupon_B (P : ℝ) := Coupon_A_savings P ≥ Coupon_B_savings
def Coupon_A_geq_Coupon_C (P : ℝ) := Coupon_A_savings P ≥ Coupon_C_savings P

noncomputable def x : ℝ := 200
noncomputable def y : ℝ := 300

theorem price_difference (P : ℝ) (h1 : P > 120)
  (h2 : Coupon_A_geq_Coupon_B P)
  (h3 : Coupon_A_geq_Coupon_C P) :
  y - x = 100 := by
  sorry

end price_difference_l189_189830


namespace copies_made_in_half_hour_l189_189698

theorem copies_made_in_half_hour
  (rate1 rate2 : ℕ)  -- rates of the two copy machines
  (time : ℕ)         -- time considered
  (h_rate1 : rate1 = 40)  -- the first machine's rate
  (h_rate2 : rate2 = 55)  -- the second machine's rate
  (h_time : time = 30)    -- time in minutes
  : (rate1 * time + rate2 * time = 2850) := 
sorry

end copies_made_in_half_hour_l189_189698


namespace find_mn_solutions_l189_189346

theorem find_mn_solutions :
  ∀ (m n : ℤ), m^5 - n^5 = 16 * m * n →
  (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
by
  sorry

end find_mn_solutions_l189_189346


namespace shopkeeper_profit_percentage_l189_189084

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ := 100) 
  (loss_due_to_theft_percent : ℝ := 30) 
  (overall_loss_percent : ℝ := 23) 
  (remaining_goods_value : ℝ := 70) 
  (overall_loss_value : ℝ := 23) 
  (selling_price : ℝ := 77) 
  (profit_percentage : ℝ) 
  (h1 : remaining_goods_value = cost_price * (1 - loss_due_to_theft_percent / 100)) 
  (h2 : overall_loss_value = cost_price * (overall_loss_percent / 100)) 
  (h3 : selling_price = cost_price - overall_loss_value) 
  (h4 : remaining_goods_value + remaining_goods_value * profit_percentage / 100 = selling_price) :
  profit_percentage = 10 := 
by 
  sorry

end shopkeeper_profit_percentage_l189_189084


namespace value_of_a3_l189_189921

theorem value_of_a3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (a : ℝ) (h₀ : (1 + x) * (a - x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) 
(h₁ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) : 
a₃ = -5 :=
sorry

end value_of_a3_l189_189921


namespace systematic_sampling_40th_number_l189_189718

open Nat

theorem systematic_sampling_40th_number (N n : ℕ) (sample_size_eq : n = 50) (total_students_eq : N = 1000) (k_def : k = N / n) (first_number : ℕ) (first_number_eq : first_number = 15) : 
  first_number + k * 39 = 795 := by
  sorry

end systematic_sampling_40th_number_l189_189718


namespace exist_three_primes_sum_to_30_l189_189110

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def less_than_twenty (n : ℕ) : Prop := n < 20

theorem exist_three_primes_sum_to_30 : 
  ∃ A B C : ℕ, is_prime A ∧ is_prime B ∧ is_prime C ∧ 
  less_than_twenty A ∧ less_than_twenty B ∧ less_than_twenty C ∧ 
  A + B + C = 30 :=
by 
  -- assume A = 2, prime and less than 20
  -- find B, C such that B and C are primes less than 20 and A + B + C = 30
  sorry

end exist_three_primes_sum_to_30_l189_189110


namespace problem_statement_l189_189995

open Complex

theorem problem_statement (x y : ℂ) (h : (x + y) / (x - y) - (3 * (x - y)) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := 
by 
  sorry

end problem_statement_l189_189995


namespace days_for_B_l189_189750

theorem days_for_B
  (x : ℝ)
  (hA : 15 ≠ 0)
  (h_nonzero_fraction : 0.5833333333333334 ≠ 0)
  (hfraction : 0 <  0.5833333333333334 ∧ 0.5833333333333334 < 1)
  (h_fraction_work_left : 5 * (1 / 15 + 1 / x) = 0.5833333333333334) :
  x = 20 := by
  sorry

end days_for_B_l189_189750


namespace sum_slope_y_intercept_l189_189749

theorem sum_slope_y_intercept (A B C F : ℝ × ℝ) (midpoint_A_C : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) 
  (coords_A : A = (0, 6)) (coords_B : B = (0, 0)) (coords_C : C = (8, 0)) :
  let slope : ℝ := (F.2 - B.2) / (F.1 - B.1)
  let y_intercept : ℝ := B.2
  slope + y_intercept = 3 / 4 := by
{
  -- proof steps
  sorry
}

end sum_slope_y_intercept_l189_189749


namespace correct_statement_a_l189_189368

theorem correct_statement_a (x y : ℝ) (h : x + y < 0) : x^2 - y > x :=
sorry

end correct_statement_a_l189_189368


namespace tangent_line_eq_l189_189891

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 + x + 1) (h_point : x = 1 ∧ y = 3) : 
  y = 4 * x - 1 := 
sorry

end tangent_line_eq_l189_189891


namespace even_three_digit_numbers_count_l189_189805

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end even_three_digit_numbers_count_l189_189805


namespace conversion_problems_l189_189374

def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 2 + 10 * decimal_to_binary (n / 2)

def largest_two_digit_octal : ℕ := 77

theorem conversion_problems :
  decimal_to_binary 111 = 1101111 ∧ (7 * 8 + 7) = 63 :=
by
  sorry

end conversion_problems_l189_189374


namespace transmission_time_l189_189339

theorem transmission_time :
  let regular_blocks := 70
  let large_blocks := 30
  let chunks_per_regular_block := 800
  let chunks_per_large_block := 1600
  let channel_rate := 200
  let total_chunks := (regular_blocks * chunks_per_regular_block) + (large_blocks * chunks_per_large_block)
  let total_time_seconds := total_chunks / channel_rate
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 8.67 := 
by 
  sorry

end transmission_time_l189_189339


namespace probability_same_color_correct_l189_189966

-- Defining the contents of Bag A and Bag B
def bagA : List (String × ℕ) := [("white", 1), ("red", 2), ("black", 3)]
def bagB : List (String × ℕ) := [("white", 2), ("red", 3), ("black", 1)]

-- The probability calculation
noncomputable def probability_same_color (bagA bagB : List (String × ℕ)) : ℚ :=
  let p_white := (1 / 6 : ℚ) * (1 / 3 : ℚ)
  let p_red := (1 / 3 : ℚ) * (1 / 2 : ℚ)
  let p_black := (1 / 2 : ℚ) * (1 / 6 : ℚ)
  p_white + p_red + p_black

-- Proof problem statement
theorem probability_same_color_correct :
  probability_same_color bagA bagB = 11 / 36 := 
by 
  sorry

end probability_same_color_correct_l189_189966


namespace BRAIN_7225_cycle_line_number_l189_189132

def BRAIN_cycle : Nat := 5
def _7225_cycle : Nat := 4

theorem BRAIN_7225_cycle_line_number : Nat.lcm BRAIN_cycle _7225_cycle = 20 :=
by
  sorry

end BRAIN_7225_cycle_line_number_l189_189132


namespace yulia_max_candies_l189_189402

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies_l189_189402


namespace cubic_product_of_roots_l189_189599

theorem cubic_product_of_roots (k : ℝ) :
  (∃ a b c : ℝ, a + b + c = 2 ∧ ab + bc + ca = 1 ∧ abc = -k ∧ -k = (max (max a b) c - min (min a b) c)^2) ↔ k = -2 :=
by
  sorry

end cubic_product_of_roots_l189_189599


namespace sphere_volume_l189_189979

theorem sphere_volume (h : 4 * π * r^2 = 256 * π) : (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l189_189979


namespace meetings_percentage_l189_189357

theorem meetings_percentage
  (workday_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_factor : ℕ)
  (third_meeting_factor : ℕ)
  (total_minutes : ℕ)
  (total_meeting_minutes : ℕ) :
  workday_hours = 9 →
  first_meeting_minutes = 30 →
  second_meeting_factor = 2 →
  third_meeting_factor = 3 →
  total_minutes = workday_hours * 60 →
  total_meeting_minutes = first_meeting_minutes + second_meeting_factor * first_meeting_minutes + third_meeting_factor * first_meeting_minutes →
  (total_meeting_minutes : ℚ) / (total_minutes : ℚ) * 100 = 33.33 :=
by
  sorry

end meetings_percentage_l189_189357


namespace simplify_cosine_tangent_product_of_cosines_l189_189907

-- Problem 1
theorem simplify_cosine_tangent :
  Real.cos 40 * (1 + Real.sqrt 3 * Real.tan 10) = 1 :=
sorry

-- Problem 2
theorem product_of_cosines :
  (Real.cos (2 * Real.pi / 7)) * (Real.cos (4 * Real.pi / 7)) * (Real.cos (6 * Real.pi / 7)) = 1 / 8 :=
sorry

end simplify_cosine_tangent_product_of_cosines_l189_189907


namespace ellen_smoothie_l189_189113

theorem ellen_smoothie :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries_used := total_ingredients - (yogurt + orange_juice)
  strawberries_used = 0.2 := by
  sorry

end ellen_smoothie_l189_189113


namespace min_value_reciprocal_sum_l189_189058

theorem min_value_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ c d : ℝ, 0 < c ∧ 0 < d ∧ c + d = 2 → (1/c + 1/d) ≥ m := 
sorry

end min_value_reciprocal_sum_l189_189058


namespace problem1_expr_eval_l189_189238

theorem problem1_expr_eval : 
  (1:ℤ) - (1:ℤ)^(2022:ℕ) - (3 * (2/3:ℚ)^2 - (8/3:ℚ) / ((-2)^3:ℤ)) = -8/3 :=
by
  sorry

end problem1_expr_eval_l189_189238


namespace no_coprime_odd_numbers_for_6_8_10_l189_189765

theorem no_coprime_odd_numbers_for_6_8_10 :
  ∀ (m n : ℤ), m > n ∧ n > 0 ∧ (m.gcd n = 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1) →
    (1 / 2 : ℚ) * (m^2 - n^2) ≠ 6 ∨ (m * n) ≠ 8 ∨ (1 / 2 : ℚ) * (m^2 + n^2) ≠ 10 :=
by
  sorry

end no_coprime_odd_numbers_for_6_8_10_l189_189765


namespace original_amount_l189_189031

theorem original_amount (x : ℝ) (h : 0.25 * x = 200) : x = 800 := 
by
  sorry

end original_amount_l189_189031


namespace smallest_number_among_options_l189_189441

noncomputable def binary_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 111111 => 63
  | _ => 0

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 210 => 2 * 6^2 + 1 * 6
  | _ => 0

noncomputable def base_nine_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 85 => 8 * 9 + 5
  | _ => 0

theorem smallest_number_among_options :
  min 75 (min (binary_to_decimal 111111) (min (base_six_to_decimal 210) (base_nine_to_decimal 85))) = binary_to_decimal 111111 :=
by 
  sorry

end smallest_number_among_options_l189_189441


namespace min_rain_fourth_day_l189_189540

def rain_overflow_problem : Prop :=
    let holding_capacity := 6 * 12 -- in inches
    let drainage_per_day := 3 -- in inches
    let rainfall_day1 := 10 -- in inches
    let rainfall_day2 := 2 * rainfall_day1 -- 20 inches
    let rainfall_day3 := 1.5 * rainfall_day2 -- 30 inches
    let total_rain_three_days := rainfall_day1 + rainfall_day2 + rainfall_day3 -- 60 inches
    let total_drainage_three_days := 3 * drainage_per_day -- 9 inches
    let remaining_capacity := holding_capacity - (total_rain_three_days - total_drainage_three_days) -- 21 inches
    (remaining_capacity = 21)

theorem min_rain_fourth_day : rain_overflow_problem := sorry

end min_rain_fourth_day_l189_189540


namespace cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l189_189015

-- Define the production cost function
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Define the profit function
def profit (n : ℕ) : ℤ := 90 * n - 4000 - 50 * n

-- 1. Prove that the cost for producing 1000 pairs of shoes is 54,000 yuan
theorem cost_of_1000_pairs : production_cost 1000 = 54000 := 
by sorry

-- 2. Prove that if the production cost is 48,000 yuan, then 880 pairs of shoes were produced
theorem pairs_for_48000_yuan (n : ℕ) (h : production_cost n = 48000) : n = 880 := 
by sorry

-- 3. Prove that at least 100 pairs of shoes must be produced each day to avoid a loss
theorem minimum_pairs_to_avoid_loss (n : ℕ) : profit n ≥ 0 ↔ n ≥ 100 := 
by sorry

end cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l189_189015


namespace problem_statement_l189_189894

theorem problem_statement (x y : ℝ) (M N P : ℝ) 
  (hM_def : M = 2 * x + y)
  (hN_def : N = 2 * x - y)
  (hP_def : P = x * y)
  (hM : M = 4)
  (hN : N = 2) : P = 1.5 :=
by
  sorry

end problem_statement_l189_189894


namespace orchard_trees_l189_189518

theorem orchard_trees (x p : ℕ) (h : x + p = 480) (h2 : p = 3 * x) : x = 120 ∧ p = 360 :=
by
  sorry

end orchard_trees_l189_189518


namespace cotangent_identity_l189_189757

noncomputable def cotangent (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cotangent_identity (x : ℝ) (i : ℂ) (n : ℕ) (k : ℕ) (h : (0 < k) ∧ (k < n)) :
  ((x + i) / (x - i))^n = 1 → x = cotangent (k * Real.pi / n) := 
sorry

end cotangent_identity_l189_189757


namespace sets_relationship_l189_189909

variables {U : Type*} (A B C : Set U)

theorem sets_relationship (h1 : A ∩ B = C) (h2 : B ∩ C = A) : A = C ∧ ∃ B, A ⊆ B := by
  sorry

end sets_relationship_l189_189909


namespace possible_values_l189_189993

theorem possible_values (m n : ℕ) (h1 : 10 ≥ m) (h2 : m > n) (h3 : n ≥ 4) (h4 : (m - n) ^ 2 = m + n) :
    (m, n) = (10, 6) :=
sorry

end possible_values_l189_189993


namespace cost_price_percentage_l189_189603

/-- The cost price (CP) as a percentage of the marked price (MP) given 
that the discount is 18% and the gain percent is 28.125%. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : CP / MP = 0.64) : 
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l189_189603


namespace max_gcd_expression_l189_189951

theorem max_gcd_expression (n : ℕ) (h1 : n > 0) (h2 : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 :=
by
  sorry

end max_gcd_expression_l189_189951


namespace trigonometric_identity_l189_189116

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := 
sorry

end trigonometric_identity_l189_189116


namespace tickets_difference_is_cost_l189_189338

def tickets_won : ℝ := 48.5
def yoyo_cost : ℝ := 11.7
def tickets_left (w : ℝ) (c : ℝ) : ℝ := w - c
def difference (w : ℝ) (l : ℝ) : ℝ := w - l

theorem tickets_difference_is_cost :
  difference tickets_won (tickets_left tickets_won yoyo_cost) = yoyo_cost :=
by
  -- Proof will be written here
  sorry

end tickets_difference_is_cost_l189_189338


namespace spadesuit_problem_l189_189378

def spadesuit (x y : ℝ) := (x + y) * (x - y)

theorem spadesuit_problem : spadesuit 5 (spadesuit 3 2) = 0 := by
  sorry

end spadesuit_problem_l189_189378


namespace solve_for_n_l189_189201

theorem solve_for_n (n : ℝ) : 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 → n = 62.5 :=
by
  intros h
  sorry

end solve_for_n_l189_189201


namespace probability_odd_product_l189_189327

theorem probability_odd_product :
  let box1 := [1, 2, 3, 4]
  let box2 := [1, 2, 3, 4]
  let total_outcomes := 4 * 4
  let favorable_outcomes := [(1,1), (1,3), (3,1), (3,3)]
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_l189_189327


namespace average_t_value_is_15_l189_189029

noncomputable def average_of_distinct_t_values (t_vals : List ℤ) : ℤ :=
t_vals.sum / t_vals.length

theorem average_t_value_is_15 :
  average_of_distinct_t_values [8, 14, 18, 20] = 15 :=
by
  sorry

end average_t_value_is_15_l189_189029


namespace no_real_solutions_l189_189484

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) = 4 * x^2 + 5 :=
by
  sorry

end no_real_solutions_l189_189484


namespace general_term_formula_l189_189712

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.sqrt n)

theorem general_term_formula :
  ∀ (n : ℕ), a n = 1 / Real.sqrt n :=
by
  intros
  rfl

end general_term_formula_l189_189712


namespace cathy_wins_probability_l189_189492

theorem cathy_wins_probability : 
  -- Definitions of the problem conditions
  let p_win := (1 : ℚ) / 6
  let p_not_win := (5 : ℚ) / 6
  -- The probability that Cathy wins
  (p_not_win ^ 2 * p_win) / (1 - p_not_win ^ 3) = 25 / 91 :=
by
  sorry

end cathy_wins_probability_l189_189492


namespace degree_le_of_lt_eventually_l189_189306

open Polynomial

theorem degree_le_of_lt_eventually {P Q : Polynomial ℝ} (h_exists : ∃ N : ℝ, ∀ x : ℝ, x > N → P.eval x < Q.eval x) :
  P.degree ≤ Q.degree :=
sorry

end degree_le_of_lt_eventually_l189_189306


namespace determine_n_for_square_l189_189189

theorem determine_n_for_square (n : ℕ) : (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 :=
by
-- The proof will be included here, but for now, we just provide the structure
sorry

end determine_n_for_square_l189_189189


namespace find_k_value_l189_189100

theorem find_k_value (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 2) (h4 : Z = K^3)
  (h5 : ∃ n : ℤ, Z = n^6) : K = 16 :=
sorry

end find_k_value_l189_189100


namespace A_inter_CUB_eq_l189_189274

noncomputable def U := Set.univ (ℝ)

noncomputable def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }

noncomputable def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x + 1 }

noncomputable def C_U (s : Set ℝ) := { x : ℝ | x ∉ s }

noncomputable def A_inter_CUB := A ∩ C_U B

theorem A_inter_CUB_eq : A_inter_CUB = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
  by sorry

end A_inter_CUB_eq_l189_189274


namespace barbara_wins_l189_189490

theorem barbara_wins (n : ℕ) (h : n = 15) (num_winning_sequences : ℕ) :
  num_winning_sequences = 8320 :=
sorry

end barbara_wins_l189_189490


namespace negation_equiv_l189_189905

open Nat

theorem negation_equiv (P : Prop) :
  (¬ (∃ n : ℕ, (n! * n!) > (2^n))) ↔ (∀ n : ℕ, (n! * n!) ≤ (2^n)) :=
by
  sorry

end negation_equiv_l189_189905


namespace least_integer_gt_sqrt_450_l189_189478

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l189_189478


namespace sports_parade_children_l189_189790

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l189_189790


namespace S_17_33_50_sum_l189_189109

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    - (n / 2)
  else
    (n + 1) / 2

theorem S_17_33_50_sum : S 17 + S 33 + S 50 = 1 :=
by
  sorry

end S_17_33_50_sum_l189_189109


namespace sin_theta_value_l189_189503

theorem sin_theta_value {θ : ℝ} (h₁ : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 :=
by
  sorry

end sin_theta_value_l189_189503


namespace no_unhappy_days_l189_189350

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l189_189350


namespace volleyball_team_selection_l189_189724

/-- A set representing players on the volleyball team -/
def players : Finset String := {
  "Missy", "Lauren", "Liz", -- triplets
  "Anna", "Mia",           -- twins
  "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10" -- other players
}

/-- The triplets -/
def triplets : Finset String := {"Missy", "Lauren", "Liz"}

/-- The twins -/
def twins : Finset String := {"Anna", "Mia"}

/-- The number of ways to choose 7 starters given the restrictions -/
theorem volleyball_team_selection : 
  let total_ways := (players.card.choose 7)
  let select_3_triplets := (players \ triplets).card.choose 4
  let select_2_twins := (players \ twins).card.choose 5
  let select_all_restriction := (players \ (triplets ∪ twins)).card.choose 2
  total_ways - select_3_triplets - select_2_twins + select_all_restriction = 9778 := by
  sorry

end volleyball_team_selection_l189_189724


namespace total_amount_invested_l189_189550

theorem total_amount_invested (x y total : ℝ) (h1 : 0.10 * x - 0.08 * y = 83) (h2 : y = 650) : total = 2000 :=
sorry

end total_amount_invested_l189_189550


namespace syrup_cost_per_week_l189_189923

theorem syrup_cost_per_week (gallons_per_week : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) 
  (h1 : gallons_per_week = 180) 
  (h2 : gallons_per_box = 30) 
  (h3 : cost_per_box = 40) : 
  (gallons_per_week / gallons_per_box) * cost_per_box = 240 := 
by
  sorry

end syrup_cost_per_week_l189_189923


namespace min_sum_of_consecutive_natural_numbers_l189_189034

theorem min_sum_of_consecutive_natural_numbers (a b c : ℕ) 
  (h1 : a + 1 = b)
  (h2 : a + 2 = c)
  (h3 : a % 9 = 0)
  (h4 : b % 8 = 0)
  (h5 : c % 7 = 0) :
  a + b + c = 1488 :=
sorry

end min_sum_of_consecutive_natural_numbers_l189_189034


namespace town_population_original_l189_189179

noncomputable def original_population (n : ℕ) : Prop :=
  let increased_population := n + 1500
  let decreased_population := (85 / 100 : ℚ) * increased_population
  decreased_population = n + 1455

theorem town_population_original : ∃ n : ℕ, original_population n ∧ n = 1200 :=
by
  sorry

end town_population_original_l189_189179


namespace area_of_fig_eq_2_l189_189311

noncomputable def area_of_fig : ℝ :=
  - ∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem area_of_fig_eq_2 : area_of_fig = 2 :=
by
  sorry

end area_of_fig_eq_2_l189_189311


namespace fraction_of_clerical_staff_is_one_third_l189_189678

-- Defining the conditions
variables (employees clerical_f clerical employees_reduced employees_remaining : ℝ)

def company_conditions (employees clerical_f clerical employees_reduced employees_remaining : ℝ) : Prop :=
  employees = 3600 ∧
  clerical = 3600 * clerical_f ∧
  employees_reduced = clerical * (2 / 3) ∧
  employees_remaining = employees - clerical * (1 / 3) ∧
  employees_reduced = 0.25 * employees_remaining

-- The statement to prove the fraction of clerical employees given the conditions
theorem fraction_of_clerical_staff_is_one_third
  (hc : company_conditions employees clerical_f clerical employees_reduced employees_remaining) :
  clerical_f = 1 / 3 :=
sorry

end fraction_of_clerical_staff_is_one_third_l189_189678


namespace courtyard_width_l189_189658

theorem courtyard_width 
  (length_of_courtyard : ℝ) 
  (num_paving_stones : ℕ) 
  (length_of_stone width_of_stone : ℝ) 
  (total_area_stone : ℝ) 
  (W : ℝ) : 
  length_of_courtyard = 40 →
  num_paving_stones = 132 →
  length_of_stone = 2.5 →
  width_of_stone = 2 →
  total_area_stone = 660 →
  40 * W = 660 →
  W = 16.5 :=
by
  intros
  sorry

end courtyard_width_l189_189658


namespace initial_scooter_value_l189_189781

theorem initial_scooter_value (V : ℝ) 
    (h : (9 / 16) * V = 22500) : V = 40000 :=
sorry

end initial_scooter_value_l189_189781


namespace solve_for_x_l189_189142

theorem solve_for_x (x : ℝ) (h : (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4)) : x = 257 / 16 := by
  sorry

end solve_for_x_l189_189142


namespace find_m_and_e_l189_189696

theorem find_m_and_e (m e : ℕ) (hm : 0 < m) (he : e < 10) 
(h1 : 4 * m^2 + m + e = 346) 
(h2 : 4 * m^2 + m + 6 = 442 + 7 * e) : 
  m + e = 22 := by
  sorry

end find_m_and_e_l189_189696


namespace no_nonconstant_poly_prime_for_all_l189_189504

open Polynomial

theorem no_nonconstant_poly_prime_for_all (f : Polynomial ℤ) (h : ∀ n : ℕ, Prime (f.eval (n : ℤ))) :
  ∃ c : ℤ, f = Polynomial.C c :=
sorry

end no_nonconstant_poly_prime_for_all_l189_189504


namespace a_equals_5_l189_189300

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end a_equals_5_l189_189300


namespace solution_set_of_inequality_minimum_value_2a_plus_b_l189_189038

noncomputable def f (x : ℝ) : ℝ := x + 1 + |3 - x|

theorem solution_set_of_inequality :
  {x : ℝ | x ≥ -1 ∧ f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem minimum_value_2a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 8 * a * b = a + 2 * b) :
  2 * a + b = 9 / 8 :=
by
  sorry

end solution_set_of_inequality_minimum_value_2a_plus_b_l189_189038


namespace color_of_217th_marble_l189_189666

-- Definitions of conditions
def total_marbles := 240
def pattern_length := 15
def red_marbles := 6
def blue_marbles := 5
def green_marbles := 4
def position := 217

-- Lean 4 statement
theorem color_of_217th_marble :
  (position % pattern_length ≤ red_marbles) :=
by sorry

end color_of_217th_marble_l189_189666


namespace latest_time_temperature_84_l189_189881

noncomputable def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem latest_time_temperature_84 :
  ∃ t_max : ℝ, temperature t_max = 84 ∧ ∀ t : ℝ, temperature t = 84 → t ≤ t_max ∧ t_max = 11 :=
by
  sorry

end latest_time_temperature_84_l189_189881


namespace train_usual_time_l189_189520

theorem train_usual_time (T : ℝ) (h1 : T > 0) : 
  (4 / 5 : ℝ) * (T + 1/2) = T :=
by 
  sorry

end train_usual_time_l189_189520


namespace solve_fractional_eq_l189_189632

theorem solve_fractional_eq {x : ℚ} : (3 / (x - 1)) = (1 / x) ↔ x = -1/2 :=
by sorry

end solve_fractional_eq_l189_189632


namespace inequality_proof_l189_189746

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l189_189746


namespace question_mark_value_l189_189709

theorem question_mark_value :
  ∀ (x : ℕ), ( ( (5568: ℝ) / (x: ℝ) )^(1/3: ℝ) + ( (72: ℝ) * (2: ℝ) )^(1/2: ℝ) = (256: ℝ)^(1/2: ℝ) ) → x = 87 :=
by
  intro x
  intro h
  sorry

end question_mark_value_l189_189709


namespace smallest_brownie_pan_size_l189_189795

theorem smallest_brownie_pan_size :
  ∃ s : ℕ, (s - 2) ^ 2 = 4 * s - 4 ∧ ∀ t : ℕ, (t - 2) ^ 2 = 4 * t - 4 → s <= t :=
by
  sorry

end smallest_brownie_pan_size_l189_189795


namespace john_new_earnings_after_raise_l189_189228

-- Definition of original earnings and raise percentage
def original_earnings : ℝ := 50
def raise_percentage : ℝ := 0.50

-- Calculate raise amount and new earnings after raise
def raise_amount : ℝ := raise_percentage * original_earnings
def new_earnings : ℝ := original_earnings + raise_amount

-- Math proof problem: Prove new earnings after raise equals $75
theorem john_new_earnings_after_raise : new_earnings = 75 := by
  sorry

end john_new_earnings_after_raise_l189_189228


namespace algebra_expression_l189_189301

theorem algebra_expression (a b : ℝ) (h : a = b + 1) : 3 + 2 * a - 2 * b = 5 :=
sorry

end algebra_expression_l189_189301


namespace simplify_fraction_l189_189890

theorem simplify_fraction:
  ((1/2 - 1/3) / (3/7 + 1/9)) * (1/4) = 21/272 :=
by
  sorry

end simplify_fraction_l189_189890


namespace part1_part2_l189_189143

theorem part1 (u v w : ℤ) (h_uv : gcd u v = 1) (h_vw : gcd v w = 1) (h_wu : gcd w u = 1) 
: gcd (u * v + v * w + w * u) (u * v * w) = 1 :=
sorry

theorem part2 (u v w : ℤ) (b := u * v + v * w + w * u) (c := u * v * w) (h : gcd b c = 1) 
: gcd u v = 1 ∧ gcd v w = 1 ∧ gcd w u = 1 :=
sorry

end part1_part2_l189_189143


namespace market_value_of_stock_l189_189493

def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.10 * face_value
def yield : ℝ := 0.08

theorem market_value_of_stock : (dividend_per_share / yield) = 125 := by
  -- Proof not required
  sorry

end market_value_of_stock_l189_189493


namespace max_area_of_triangle_l189_189713

theorem max_area_of_triangle :
  ∀ (O O' : EuclideanSpace ℝ (Fin 2)) (M : EuclideanSpace ℝ (Fin 2)),
  dist O O' = 2014 →
  dist O M = 1 ∨ dist O' M = 1 →
  ∃ (A : ℝ), A = 1007 :=
by
  intros O O' M h₁ h₂
  sorry

end max_area_of_triangle_l189_189713


namespace symmetric_line_eq_l189_189796

theorem symmetric_line_eq : ∀ (x y : ℝ), (x - 2*y - 1 = 0) ↔ (2*x - y + 1 = 0) :=
by sorry

end symmetric_line_eq_l189_189796


namespace factor_quadratic_l189_189533

theorem factor_quadratic (y : ℝ) : 9 * y ^ 2 - 30 * y + 25 = (3 * y - 5) ^ 2 := by
  sorry

end factor_quadratic_l189_189533


namespace roots_eq_two_iff_a_gt_neg1_l189_189651

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l189_189651


namespace percentage_return_on_investment_l189_189362

theorem percentage_return_on_investment (dividend_rate : ℝ) (face_value : ℝ) (purchase_price : ℝ) (return_percentage : ℝ) :
  dividend_rate = 0.125 → face_value = 40 → purchase_price = 20 → return_percentage = 25 :=
by
  intros h1 h2 h3
  sorry

end percentage_return_on_investment_l189_189362


namespace complementary_not_supplementary_l189_189910

theorem complementary_not_supplementary (α β : ℝ) (h₁ : α + β = 90) (h₂ : α + β ≠ 180) : (α + β = 180) = false :=
by 
  sorry

end complementary_not_supplementary_l189_189910


namespace fraction_remaining_distance_l189_189660

theorem fraction_remaining_distance
  (total_distance : ℕ)
  (first_stop_fraction : ℚ)
  (remaining_distance_after_second_stop : ℕ)
  (fraction_between_stops : ℚ) :
  total_distance = 280 →
  first_stop_fraction = 1/2 →
  remaining_distance_after_second_stop = 105 →
  (fraction_between_stops * (total_distance - (first_stop_fraction * total_distance)) + remaining_distance_after_second_stop = (total_distance - (first_stop_fraction * total_distance))) →
  fraction_between_stops = 1/4 :=
by
  sorry

end fraction_remaining_distance_l189_189660


namespace distance_between_first_and_last_student_l189_189544

theorem distance_between_first_and_last_student 
  (n : ℕ) (d : ℕ)
  (students : n = 30) 
  (distance_between_students : d = 3) : 
  n - 1 * d = 87 := 
by
  sorry

end distance_between_first_and_last_student_l189_189544


namespace find_p_l189_189190

open Real

variable (A : ℝ × ℝ)
variable (p : ℝ) (hp : p > 0)

-- Conditions
def on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop := A.snd^2 = 2 * p * A.fst
def dist_focus (A : ℝ × ℝ) (p : ℝ) : Prop := sqrt ((A.fst - p / 2)^2 + A.snd^2) = 12
def dist_y_axis (A : ℝ × ℝ) : Prop := abs (A.fst) = 9

-- Theorem to prove
theorem find_p (h1 : on_parabola A p) (h2 : dist_focus A p) (h3 : dist_y_axis A) : p = 6 :=
sorry

end find_p_l189_189190


namespace remaining_soup_feeds_20_adults_l189_189716

theorem remaining_soup_feeds_20_adults (cans_of_soup : ℕ) (feed_4_adults : ℕ) (feed_7_children : ℕ) (initial_cans : ℕ) (children_fed : ℕ)
    (h1 : feed_4_adults = 4)
    (h2 : feed_7_children = 7)
    (h3 : initial_cans = 8)
    (h4 : children_fed = 21) : 
    (initial_cans - (children_fed / feed_7_children)) * feed_4_adults = 20 :=
by
  sorry

end remaining_soup_feeds_20_adults_l189_189716


namespace compound_ratio_is_one_fourteenth_l189_189886

theorem compound_ratio_is_one_fourteenth :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) = 1 / 14 :=
by sorry

end compound_ratio_is_one_fourteenth_l189_189886


namespace g_42_value_l189_189577

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n : ℕ) (hn : n > 0) : g (n + 1) > g n
axiom g_multiplicative (m n : ℕ) (hm : m > 0) (hn : n > 0) : g (m * n) = g m * g n
axiom g_property_iii (m n : ℕ) (hm : m > 0) (hn : n > 0) : (m ≠ n ∧ m^n = n^m) → (g m = n ∨ g n = m)

theorem g_42_value : g 42 = 4410 :=
by
  sorry

end g_42_value_l189_189577


namespace frank_spent_per_week_l189_189452

theorem frank_spent_per_week (mowing_dollars : ℕ) (weed_eating_dollars : ℕ) (weeks : ℕ) 
    (total_dollars := mowing_dollars + weed_eating_dollars) 
    (spending_rate := total_dollars / weeks) :
    mowing_dollars = 5 → weed_eating_dollars = 58 → weeks = 9 → spending_rate = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end frank_spent_per_week_l189_189452


namespace two_trains_crossing_time_l189_189195

theorem two_trains_crossing_time
  (length_train: ℝ) (time_telegraph_post_first: ℝ) (time_telegraph_post_second: ℝ)
  (length_train_eq: length_train = 120) 
  (time_telegraph_post_first_eq: time_telegraph_post_first = 10) 
  (time_telegraph_post_second_eq: time_telegraph_post_second = 15) :
  (2 * length_train) / (length_train / time_telegraph_post_first + length_train / time_telegraph_post_second) = 12 :=
by
  sorry

end two_trains_crossing_time_l189_189195


namespace hydroflow_rate_30_minutes_l189_189837

def hydroflow_pumped (rate_per_hour: ℕ) (minutes: ℕ) : ℕ :=
  let hours := minutes / 60
  rate_per_hour * hours

theorem hydroflow_rate_30_minutes : 
  hydroflow_pumped 500 30 = 250 :=
by 
  -- place the proof here
  sorry

end hydroflow_rate_30_minutes_l189_189837


namespace leak_emptying_time_l189_189508

theorem leak_emptying_time (fill_rate_no_leak : ℝ) (combined_rate_with_leak : ℝ) (L : ℝ) :
  fill_rate_no_leak = 1/10 →
  combined_rate_with_leak = 1/12 →
  fill_rate_no_leak - L = combined_rate_with_leak →
  1 / L = 60 :=
by
  intros h1 h2 h3
  sorry

end leak_emptying_time_l189_189508


namespace xyz_value_l189_189017

-- Define the real numbers x, y, and z
variables (x y z : ℝ)

-- Condition 1
def condition1 := (x + y + z) * (x * y + x * z + y * z) = 49

-- Condition 2
def condition2 := x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19

-- Main theorem statement
theorem xyz_value (h1 : condition1 x y z) (h2 : condition2 x y z) : x * y * z = 10 :=
sorry

end xyz_value_l189_189017


namespace count_3_digit_integers_with_product_36_l189_189407

theorem count_3_digit_integers_with_product_36 : 
  ∃ n, n = 21 ∧ 
         (∀ d1 d2 d3 : ℕ, 
           1 ≤ d1 ∧ d1 ≤ 9 ∧ 
           1 ≤ d2 ∧ d2 ≤ 9 ∧ 
           1 ≤ d3 ∧ d3 ≤ 9 ∧
           d1 * d2 * d3 = 36 → 
           (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0)) := sorry

end count_3_digit_integers_with_product_36_l189_189407


namespace remainder_p_q_add_42_l189_189667

def p (k : ℤ) : ℤ := 98 * k + 84
def q (m : ℤ) : ℤ := 126 * m + 117

theorem remainder_p_q_add_42 (k m : ℤ) : 
  (p k + q m) % 42 = 33 := by
  sorry

end remainder_p_q_add_42_l189_189667


namespace combination_count_l189_189534

-- Definitions from conditions
def packagingPapers : Nat := 10
def ribbons : Nat := 4
def stickers : Nat := 5

-- Proof problem statement
theorem combination_count : packagingPapers * ribbons * stickers = 200 := 
by
  sorry

end combination_count_l189_189534


namespace total_money_before_spending_l189_189030

-- Define the amounts for each friend
variables (J P Q A: ℝ)

-- Define the conditions from the problem
def condition1 := P = 2 * J
def condition2 := Q = P + 20
def condition3 := A = 1.15 * Q
def condition4 := J + P + Q + A = 1211
def cost_of_item : ℝ := 1200

-- The total amount before buying the item
theorem total_money_before_spending (J P Q A : ℝ)
  (h1 : condition1 J P)
  (h2 : condition2 P Q)
  (h3 : condition3 Q A)
  (h4 : condition4 J P Q A) : 
  J + P + Q + A - cost_of_item = 11 :=
by
  sorry

end total_money_before_spending_l189_189030


namespace point_on_angle_bisector_l189_189500

theorem point_on_angle_bisector (a : ℝ) 
  (h : (2 : ℝ) * a + (3 : ℝ) = a) : a = -3 :=
sorry

end point_on_angle_bisector_l189_189500


namespace auston_height_l189_189057

noncomputable def auston_height_in_meters (height_in_inches : ℝ) : ℝ :=
  let height_in_cm := height_in_inches * 2.54
  height_in_cm / 100

theorem auston_height : auston_height_in_meters 65 = 1.65 :=
by
  sorry

end auston_height_l189_189057


namespace similar_triangle_perimeter_l189_189560

/-
  Given an isosceles triangle with two equal sides of 18 inches and a base of 12 inches, 
  and a similar triangle with the shortest side of 30 inches, 
  prove that the perimeter of the similar triangle is 120 inches.
-/

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_perimeter
  (a b c : ℕ) (a' b' c' : ℕ) (h1 : is_isosceles a b c)
  (h2 : a = 12) (h3 : b = 18) (h4 : c = 18)
  (h5 : a' = 30) (h6 : a' * 18 = a * b')
  (h7 : a' * 18 = a * c') :
  a' + b' + c' = 120 :=
by {
  sorry
}

end similar_triangle_perimeter_l189_189560


namespace perfect_square_solutions_l189_189281

theorem perfect_square_solutions (a b : ℕ) (ha : a > b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hA : ∃ k : ℕ, a^2 + 4 * b + 1 = k^2) (hB : ∃ l : ℕ, b^2 + 4 * a + 1 = l^2) :
  a = 8 ∧ b = 4 ∧ (a^2 + 4 * b + 1 = (a+1)^2) ∧ (b^2 + 4 * a + 1 = (b + 3)^2) :=
by
  sorry

end perfect_square_solutions_l189_189281


namespace solve_system_l189_189006

variable {x y z : ℝ}

theorem solve_system :
  (y + z = 16 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 13 - 4 * z) →
  2 * x + 2 * y + 2 * z = 11 / 3 :=
by
  intros h1 h2 h3
  -- proof skips, to be completed
  sorry

end solve_system_l189_189006


namespace base9_subtraction_l189_189739

theorem base9_subtraction (a b : Nat) (h1 : a = 256) (h2 : b = 143) : 
  (a - b) = 113 := 
sorry

end base9_subtraction_l189_189739


namespace simplify_complex_fraction_l189_189867

open Complex

theorem simplify_complex_fraction :
  (⟨2, 2⟩ : ℂ) / (⟨-3, 4⟩ : ℂ) = (⟨-14 / 25, -14 / 25⟩ : ℂ) :=
by
  sorry

end simplify_complex_fraction_l189_189867


namespace cost_of_pencils_and_pens_l189_189841

theorem cost_of_pencils_and_pens (p q : ℝ) 
  (h₁ : 3 * p + 2 * q = 3.60) 
  (h₂ : 2 * p + 3 * q = 3.15) : 
  3 * p + 3 * q = 4.05 :=
sorry

end cost_of_pencils_and_pens_l189_189841


namespace num_first_and_second_year_students_total_l189_189579

-- Definitions based on conditions
def num_sampled_students : ℕ := 55
def num_first_year_students_sampled : ℕ := 10
def num_second_year_students_sampled : ℕ := 25
def num_third_year_students_total : ℕ := 400

-- Given that 20 students from the third year are sampled
def num_third_year_students_sampled := num_sampled_students - num_first_year_students_sampled - num_second_year_students_sampled

-- Proportion equality condition
theorem num_first_and_second_year_students_total (x : ℕ) :
  20 / 55 = 400 / (x + num_third_year_students_total) →
  x = 700 :=
by
  sorry

end num_first_and_second_year_students_total_l189_189579


namespace sum_digits_in_possibilities_l189_189439

noncomputable def sum_of_digits (a b c d : ℕ) : ℕ :=
  a + b + c + d

theorem sum_digits_in_possibilities :
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (sum_of_digits a b c d = 10 ∨ sum_of_digits a b c d = 18 ∨ sum_of_digits a b c d = 19) := sorry

end sum_digits_in_possibilities_l189_189439


namespace smaller_angle_clock_8_10_l189_189562

/-- The measure of the smaller angle formed by the hour and minute hands of a clock at 8:10 p.m. is 175 degrees. -/
theorem smaller_angle_clock_8_10 : 
  let full_circle := 360
  let hour_increment := 30
  let hour_angle_8 := 8 * hour_increment
  let minute_angle_increment := 6
  let hour_hand_adjustment := 10 * (hour_increment / 60)
  let hour_hand_position := hour_angle_8 + hour_hand_adjustment
  let minute_hand_position := 10 * minute_angle_increment
  let angle_difference := if hour_hand_position > minute_hand_position 
                          then hour_hand_position - minute_hand_position 
                          else minute_hand_position - hour_hand_position  
  let smaller_angle := if 2 * angle_difference > full_circle 
                       then full_circle - angle_difference 
                       else angle_difference
  smaller_angle = 175 :=
by 
  sorry

end smaller_angle_clock_8_10_l189_189562


namespace general_term_formula_l189_189850

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 3 * n ^ 2 - 2 * n) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ a 1 = S 1 → 
  ∀ n, a n = 6 * n - 5 := 
by
  sorry

end general_term_formula_l189_189850


namespace transform_cos_function_l189_189554

theorem transform_cos_function :
  ∀ x : ℝ, 2 * Real.cos (x + π / 3) =
           2 * Real.cos (2 * (x - π / 12) + π / 6) := 
sorry

end transform_cos_function_l189_189554


namespace factor_expression_l189_189649

theorem factor_expression (y : ℝ) : 
  (16 * y ^ 6 + 36 * y ^ 4 - 9) - (4 * y ^ 6 - 6 * y ^ 4 - 9) = 6 * y ^ 4 * (2 * y ^ 2 + 7) := 
by sorry

end factor_expression_l189_189649


namespace percentage_difference_l189_189858

theorem percentage_difference (x : ℝ) (h1 : 0.38 * 80 = 30.4) (h2 : 30.4 - (x / 100) * 160 = 11.2) :
    x = 12 :=
by
  sorry

end percentage_difference_l189_189858


namespace ellipse_foci_distance_l189_189342

-- Definitions based on the problem conditions
def ellipse_eq (x y : ℝ) :=
  Real.sqrt (((x - 4)^2) + ((y - 5)^2)) + Real.sqrt (((x + 6)^2) + ((y + 9)^2)) = 22

def focus1 : (ℝ × ℝ) := (4, -5)
def focus2 : (ℝ × ℝ) := (-6, 9)

-- Statement of the problem
noncomputable def distance_between_foci : ℝ :=
  Real.sqrt (((focus1.1 + 6)^2) + ((focus1.2 - 9)^2))

-- Proof statement
theorem ellipse_foci_distance : distance_between_foci = 2 * Real.sqrt 74 := by
  sorry

end ellipse_foci_distance_l189_189342


namespace squirrel_divides_acorns_l189_189775

theorem squirrel_divides_acorns (total_acorns parts_per_month remaining_acorns month_acorns winter_months spring_acorns : ℕ)
  (h1 : total_acorns = 210)
  (h2 : parts_per_month = 3)
  (h3 : winter_months = 3)
  (h4 : remaining_acorns = 60)
  (h5 : month_acorns = total_acorns / winter_months)
  (h6 : spring_acorns = 30)
  (h7 : month_acorns - remaining_acorns = spring_acorns / parts_per_month) :
  parts_per_month = 3 :=
by
  sorry

end squirrel_divides_acorns_l189_189775


namespace no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l189_189615

variables {a b : ℝ} (spells : list (ℝ × ℝ)) (infinite_spells : ℕ → ℝ × ℝ)

-- Condition: 0 < a < b
def valid_spell (spell : ℝ × ℝ) : Prop := 0 < spell.1 ∧ spell.1 < spell.2

-- Question a: Finite set of spells, prove that no spell set exists such that the second wizard can guarantee a win.
theorem no_finite_spells_guarantee_second_wizard_win :
  (∀ spell ∈ spells, valid_spell spell) →
  ¬(∃ (strategy : ℕ → ℝ × ℝ), ∀ n, valid_spell (strategy n) ∧ ∃ k, n < k ∧ valid_spell (strategy k)) :=
sorry

-- Question b: Infinite set of spells, prove that there exists a spell set such that the second wizard can guarantee a win.
theorem exists_infinite_spells_guarantee_second_wizard_win :
  (∀ n, valid_spell (infinite_spells n)) →
  ∃ (strategy : ℕ → ℝ × ℝ), ∀ n, ∃ k, n < k ∧ valid_spell (strategy k) :=
sorry

end no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l189_189615


namespace fg_2_eq_9_l189_189358

def f (x: ℝ) := x^2
def g (x: ℝ) := -4 * x + 5

theorem fg_2_eq_9 : f (g 2) = 9 :=
by
  sorry

end fg_2_eq_9_l189_189358


namespace num_partitions_of_staircase_l189_189163

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end num_partitions_of_staircase_l189_189163


namespace negation_exists_l189_189715

theorem negation_exists:
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by
  sorry

end negation_exists_l189_189715


namespace sector_central_angle_l189_189930

theorem sector_central_angle (r l : ℝ) (α : ℝ) 
  (h1 : l + 2 * r = 12) 
  (h2 : 1 / 2 * l * r = 8) : 
  α = 1 ∨ α = 4 :=
by
  sorry

end sector_central_angle_l189_189930


namespace eight_digit_number_div_by_9_l189_189578

theorem eight_digit_number_div_by_9 (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 9)
  (h : (8 + 5 + 4 + n + 5 + 2 + 6 + 8) % 9 = 0) : n = 7 :=
by
  sorry

end eight_digit_number_div_by_9_l189_189578


namespace price_jemma_sells_each_frame_is_5_l189_189888

noncomputable def jemma_price_per_frame : ℝ :=
  let num_frames_jemma := 400
  let num_frames_dorothy := num_frames_jemma / 2
  let total_income := 2500
  let P_jemma := total_income / (num_frames_jemma + num_frames_dorothy / 2)
  P_jemma

theorem price_jemma_sells_each_frame_is_5 :
  jemma_price_per_frame = 5 := by
  sorry

end price_jemma_sells_each_frame_is_5_l189_189888


namespace prime_factors_of_product_l189_189448

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  -- Assume we have a function that returns a list of prime factors of n
  sorry

def num_distinct_primes (n : ℕ) : ℕ :=
  (prime_factors n).toFinset.card

theorem prime_factors_of_product :
  num_distinct_primes (85 * 87 * 91 * 94) = 8 :=
by
  have prod_factorizations : 85 = 5 * 17 ∧ 87 = 3 * 29 ∧ 91 = 7 * 13 ∧ 94 = 2 * 47 := 
    by sorry -- each factorization step
  sorry

end prime_factors_of_product_l189_189448


namespace total_amount_spent_l189_189946

variable (you friend : ℝ)

theorem total_amount_spent (h1 : friend = you + 3) (h2 : friend = 7) : 
  you + friend = 11 :=
by
  sorry

end total_amount_spent_l189_189946


namespace difference_in_soda_bottles_l189_189450

-- Define the given conditions
def regular_soda_bottles : ℕ := 81
def diet_soda_bottles : ℕ := 60

-- Define the difference in the number of bottles
def difference_bottles : ℕ := regular_soda_bottles - diet_soda_bottles

-- The theorem we want to prove
theorem difference_in_soda_bottles : difference_bottles = 21 := by
  sorry

end difference_in_soda_bottles_l189_189450


namespace tape_mounting_cost_correct_l189_189462

-- Define the given conditions as Lean definitions
def os_overhead_cost : ℝ := 1.07
def cost_per_millisecond : ℝ := 0.023
def total_cost : ℝ := 40.92
def runtime_seconds : ℝ := 1.5

-- Define the required target cost for mounting a data tape
def cost_of_data_tape : ℝ := 5.35

-- Prove that the cost of mounting a data tape is correct given the conditions
theorem tape_mounting_cost_correct :
  let computer_time_cost := cost_per_millisecond * (runtime_seconds * 1000)
  let total_cost_computed := os_overhead_cost + computer_time_cost
  cost_of_data_tape = total_cost - total_cost_computed := by
{
  sorry
}

end tape_mounting_cost_correct_l189_189462


namespace solve_for_x_l189_189308

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem solve_for_x (x : ℝ) :
  let a := vec 1 2
  let b := vec x 1
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 - 2 * b.1, 2 * a.2 - 2 * b.2)
  (u.1 * v.2 = u.2 * v.1) → x = 1 / 2 := by
  sorry

end solve_for_x_l189_189308


namespace solve_f_eq_x_l189_189491

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_domain : ∀ (x : ℝ), 0 ≤ x ∧ x < 1 → 1 ≤ f_inv x ∧ f_inv x < 2
axiom f_inv_range : ∀ (x : ℝ), 2 < x ∧ x ≤ 4 → 0 ≤ f_inv x ∧ f_inv x < 1
-- Assumption that f is invertible on [0, 3]
axiom f_inv_exists : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, f y = x

theorem solve_f_eq_x : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = x → x = 2 :=
by
  sorry

end solve_f_eq_x_l189_189491


namespace number_of_carbon_atoms_l189_189823

/-- A proof to determine the number of carbon atoms in a compound given specific conditions
-/
theorem number_of_carbon_atoms
  (H_atoms : ℕ) (O_atoms : ℕ) (C_weight : ℕ) (H_weight : ℕ) (O_weight : ℕ) (Molecular_weight : ℕ) :
  H_atoms = 6 →
  O_atoms = 1 →
  C_weight = 12 →
  H_weight = 1 →
  O_weight = 16 →
  Molecular_weight = 58 →
  (Molecular_weight - (H_atoms * H_weight + O_atoms * O_weight)) / C_weight = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_carbon_atoms_l189_189823


namespace scout_weekend_earnings_l189_189591

-- Definitions for conditions
def base_pay_per_hour : ℝ := 10.00
def tip_saturday : ℝ := 5.00
def tip_sunday_low : ℝ := 3.00
def tip_sunday_high : ℝ := 7.00
def transportation_cost_per_delivery : ℝ := 1.00
def hours_worked_saturday : ℝ := 6
def deliveries_saturday : ℝ := 5
def hours_worked_sunday : ℝ := 8
def deliveries_sunday : ℝ := 10
def deliveries_sunday_low_tip : ℝ := 5
def deliveries_sunday_high_tip : ℝ := 5
def holiday_multiplier : ℝ := 2

-- Calculation of total earnings for the weekend after transportation costs
theorem scout_weekend_earnings : 
  let base_pay_saturday := hours_worked_saturday * base_pay_per_hour
  let tips_saturday := deliveries_saturday * tip_saturday
  let transportation_costs_saturday := deliveries_saturday * transportation_cost_per_delivery
  let total_earnings_saturday := base_pay_saturday + tips_saturday - transportation_costs_saturday

  let base_pay_sunday := hours_worked_sunday * base_pay_per_hour * holiday_multiplier
  let tips_sunday := deliveries_sunday_low_tip * tip_sunday_low + deliveries_sunday_high_tip * tip_sunday_high
  let transportation_costs_sunday := deliveries_sunday * transportation_cost_per_delivery
  let total_earnings_sunday := base_pay_sunday + tips_sunday - transportation_costs_sunday

  let total_earnings_weekend := total_earnings_saturday + total_earnings_sunday

  total_earnings_weekend = 280.00 :=
by
  -- Add detailed proof here
  sorry

end scout_weekend_earnings_l189_189591


namespace find_three_digit_number_l189_189851

-- Definitions of digit constraints and the number representation
def is_three_digit_number (N : ℕ) (a b c : ℕ) : Prop :=
  N = 100 * a + 10 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9

-- Definition of the problem condition
def sum_of_digits_condition (N : ℕ) (a b c : ℕ) : Prop :=
  a + b + c = N / 11

-- Lean theorem statement
theorem find_three_digit_number (N a b c : ℕ) :
  is_three_digit_number N a b c ∧ sum_of_digits_condition N a b c → N = 198 :=
by
  sorry

end find_three_digit_number_l189_189851


namespace total_pies_sold_l189_189968

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l189_189968


namespace proportion_of_fathers_with_full_time_jobs_l189_189980

theorem proportion_of_fathers_with_full_time_jobs
  (P : ℕ) -- Total number of parents surveyed
  (mothers_proportion : ℝ := 0.4) -- Proportion of mothers in the survey
  (mothers_ftj_proportion : ℝ := 0.9) -- Proportion of mothers with full-time jobs
  (parents_no_ftj_proportion : ℝ := 0.19) -- Proportion of parents without full-time jobs
  (hfathers : ℝ := 0.6) -- Proportion of fathers in the survey
  (hfathers_ftj_proportion : ℝ) -- Proportion of fathers with full-time jobs
  : hfathers_ftj_proportion = 0.75 := 
by 
  sorry

end proportion_of_fathers_with_full_time_jobs_l189_189980


namespace counterexample_exists_l189_189839

theorem counterexample_exists : 
  ∃ (m : ℤ), (∃ (k1 : ℤ), m = 2 * k1) ∧ ¬(∃ (k2 : ℤ), m = 4 * k2) := 
sorry

end counterexample_exists_l189_189839


namespace jimmy_change_l189_189157

def cost_of_pens (num_pens : ℕ) (cost_per_pen : ℕ): ℕ := num_pens * cost_per_pen
def cost_of_notebooks (num_notebooks : ℕ) (cost_per_notebook : ℕ): ℕ := num_notebooks * cost_per_notebook
def cost_of_folders (num_folders : ℕ) (cost_per_folder : ℕ): ℕ := num_folders * cost_per_folder

def total_cost : ℕ :=
  cost_of_pens 3 1 + cost_of_notebooks 4 3 + cost_of_folders 2 5

def paid_amount : ℕ := 50

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end jimmy_change_l189_189157


namespace prepaid_card_cost_correct_l189_189251

noncomputable def prepaid_phone_card_cost
    (cost_per_minute : ℝ) (call_minutes : ℝ) (remaining_credit : ℝ) : ℝ :=
  remaining_credit + (call_minutes * cost_per_minute)

theorem prepaid_card_cost_correct :
  let cost_per_minute := 0.16
  let call_minutes := 22
  let remaining_credit := 26.48
  prepaid_phone_card_cost cost_per_minute call_minutes remaining_credit = 30.00 := by
  sorry

end prepaid_card_cost_correct_l189_189251


namespace age_difference_is_40_l189_189563

-- Define the ages of the daughter and the mother
variables (D M : ℕ)

-- Conditions
-- 1. The mother's age is the digits of the daughter's age reversed
def mother_age_is_reversed_daughter_age : Prop :=
  M = 10 * D + D

-- 2. In thirteen years, the mother will be twice as old as the daughter
def mother_twice_as_old_in_thirteen_years : Prop :=
  M + 13 = 2 * (D + 13)

-- The theorem: The difference in their current ages is 40
theorem age_difference_is_40
  (h1 : mother_age_is_reversed_daughter_age D M)
  (h2 : mother_twice_as_old_in_thirteen_years D M) :
  M - D = 40 :=
sorry

end age_difference_is_40_l189_189563


namespace intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l189_189703

noncomputable def h (a x : ℝ) : ℝ := a * x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def f (a x : ℝ) : ℝ := h a x + 3 * x * g x
noncomputable def F (a x : ℝ) : ℝ := (a - (1/3)) * x^3 + (1/2) * x^2 * g a - h a x - 1

theorem intervals_of_monotonicity (a : ℝ) (ha : f a 1 = -1) :
  ((a = 0) → (∀ x : ℝ, (0 < x ∧ x < Real.exp (-1) → f 0 x < f 0 x + 3 * x * g x)) ∧
    (Real.exp (-1) < x ∧ 0 < x → f 0 x + 3 * x * g x > f 0 x)) := sorry

theorem m_in_terms_of_x0 (a x0 m : ℝ) (ha : a > Real.exp (10 / 3))
  (tangent_line : ∀ y, y - ( -(1 / 3) * x0^3 + (1 / 2) * x0^2 * g a) = 
    (-(x0^2) + x0 * g a) * (x - x0)) :
  m = (2 / 3) * x0^3 - (1 + (1 / 2) * g a) * x0^2 + x0 * g a := sorry

theorem at_least_two_tangents (a m : ℝ) (ha : a > Real.exp (10 / 3))
  (at_least_two : ∃ x0 y, x0 ≠ y ∧ F a x0 = m ∧ F a y = m) :
  m = 4 / 3 := sorry

end intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l189_189703


namespace smallest_number_diminished_by_8_divisible_by_9_6_12_18_l189_189903

theorem smallest_number_diminished_by_8_divisible_by_9_6_12_18 :
  ∃ x : ℕ, (x - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 ∧ ∀ y : ℕ, (y - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 → x ≤ y → x = 44 :=
by
  sorry

end smallest_number_diminished_by_8_divisible_by_9_6_12_18_l189_189903


namespace inequality_solution_exists_l189_189769

theorem inequality_solution_exists (x m : ℝ) (h1: 1 < x) (h2: x ≤ 2) (h3: x > m) : m < 2 :=
sorry

end inequality_solution_exists_l189_189769


namespace sum_greater_l189_189864

theorem sum_greater {a b c d : ℝ} (h1 : b + Real.sin a > d + Real.sin c) (h2 : a + Real.sin b > c + Real.sin d) : a + b > c + d := by
  sorry

end sum_greater_l189_189864


namespace one_and_one_third_of_what_number_is_45_l189_189044

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l189_189044


namespace fran_speed_calculation_l189_189656

theorem fran_speed_calculation:
  let Joann_speed := 15
  let Joann_time := 5
  let Fran_time := 4
  let Fran_speed := (Joann_speed * Joann_time) / Fran_time
  Fran_speed = 18.75 := by
  sorry

end fran_speed_calculation_l189_189656


namespace jim_profit_percentage_l189_189053

theorem jim_profit_percentage (S C : ℝ) (H1 : S = 670) (H2 : C = 536) :
  ((S - C) / C) * 100 = 25 :=
by
  sorry

end jim_profit_percentage_l189_189053


namespace find_integer_pairs_l189_189028

theorem find_integer_pairs (x y : ℤ) (h : x^3 - y^3 = 2 * x * y + 8) : 
  (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) := 
by {
  sorry
}

end find_integer_pairs_l189_189028


namespace not_difference_of_squares_10_l189_189529

theorem not_difference_of_squares_10 (a b : ℤ) : a^2 - b^2 ≠ 10 :=
sorry

end not_difference_of_squares_10_l189_189529


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l189_189589

-- Define the conditions for each problem explicitly
def cond1 : Prop := ∃ (A B C : Type), -- "A" can only be in the middle or on the sides (positions are constrainted)
  True -- (specific arrangements are abstracted here)

def cond2 : Prop := ∃ (A B C : Type), -- male students must be grouped together
  True

def cond3 : Prop := ∃ (A B C : Type), -- male students cannot be grouped together
  True

def cond4 : Prop := ∃ (A B C : Type), -- the order of "A", "B", "C" from left to right remains unchanged
  True

def cond5 : Prop := ∃ (A B C : Type), -- "A" is not on the far left and "B" is not on the far right
  True

def cond6 : Prop := ∃ (A B C D : Type), -- One more female student, males and females are not next to each other
  True

def cond7 : Prop := ∃ (A B C : Type), -- arranged in two rows, with 3 people in the front row and 2 in the back row
  True

def cond8 : Prop := ∃ (A B C : Type), -- there must be 1 person between "A" and "B"
  True

-- Prove each condition results in the specified number of arrangements

theorem problem1 : cond1 → True := by
  -- Problem (1) is to show 72 arrangements given conditions
  sorry

theorem problem2 : cond2 → True := by
  -- Problem (2) is to show 36 arrangements given conditions
  sorry

theorem problem3 : cond3 → True := by
  -- Problem (3) is to show 12 arrangements given conditions
  sorry

theorem problem4 : cond4 → True := by
  -- Problem (4) is to show 20 arrangements given conditions
  sorry

theorem problem5 : cond5 → True := by
  -- Problem (5) is to show 78 arrangements given conditions
  sorry

theorem problem6 : cond6 → True := by
  -- Problem (6) is to show 144 arrangements given conditions
  sorry

theorem problem7 : cond7 → True := by
  -- Problem (7) is to show 120 arrangements given conditions
  sorry

theorem problem8 : cond8 → True := by
  -- Problem (8) is to show 36 arrangements given conditions
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l189_189589


namespace trigonometric_identity_l189_189584

variable (α β : Real) 

theorem trigonometric_identity (h₁ : Real.tan (α + β) = 1) 
                              (h₂ : Real.tan (α - β) = 2) 
                              : (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := 
by 
  sorry

end trigonometric_identity_l189_189584


namespace student_marks_l189_189754

theorem student_marks (M P C X : ℕ) 
  (h1 : M + P = 60)
  (h2 : C = P + X)
  (h3 : M + C = 80) : X = 20 :=
by sorry

end student_marks_l189_189754


namespace smallest_N_circular_table_l189_189840

theorem smallest_N_circular_table (N chairs : ℕ) (circular_seating : N < chairs) :
  (∀ new_person_reserved : ℕ, new_person_reserved < chairs →
    (∃ i : ℕ, (i < N) ∧ (new_person_reserved = (i + 1) % chairs ∨ 
                           new_person_reserved = (i - 1) % chairs))) ↔ N = 18 := by
sorry

end smallest_N_circular_table_l189_189840


namespace or_false_iff_not_p_l189_189351

theorem or_false_iff_not_p (p q : Prop) : (p ∨ q → false) ↔ ¬p :=
by sorry

end or_false_iff_not_p_l189_189351


namespace circles_tangent_radii_product_eq_l189_189991

/-- Given two circles that pass through a fixed point \(M(x_1, y_1)\)
    and are tangent to both the x-axis and y-axis, with radii \(r_1\) and \(r_2\),
    prove that \(r_1 r_2 = x_1^2 + y_1^2\). -/
theorem circles_tangent_radii_product_eq (x1 y1 r1 r2 : ℝ)
  (h1 : (∃ (a : ℝ), ∃ (circle1 : ℝ → ℝ → ℝ), ∀ x y, circle1 x y = (x - a)^2 + (y - a)^2 - r1^2)
    ∧ (∃ (b : ℝ), ∃ (circle2 : ℝ → ℝ → ℝ), ∀ x y, circle2 x y = (x - b)^2 + (y - b)^2 - r2^2))
  (hm1 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r1)^2 + (p.snd - r1)^2 = r1^2 })
  (hm2 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r2)^2 + (p.snd - r2)^2 = r2^2 }) :
  r1 * r2 = x1^2 + y1^2 := sorry

end circles_tangent_radii_product_eq_l189_189991


namespace shoe_length_increase_l189_189981

noncomputable def shoeSizeLength (l : ℕ → ℝ) (size : ℕ) : ℝ :=
  if size = 15 then 9.25
  else if size = 17 then 1.3 * l 8
  else l size

theorem shoe_length_increase :
  (forall l : ℕ → ℝ,
    (shoeSizeLength l 15 = 9.25) ∧
    (shoeSizeLength l 17 = 1.3 * (shoeSizeLength l 8)) ∧
    (forall n, shoeSizeLength l (n + 1) = shoeSizeLength l n + 0.25)
  ) :=
  sorry

end shoe_length_increase_l189_189981


namespace sum_of_roots_of_quadratic_l189_189697

theorem sum_of_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (Polynomial.eval x1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) ∧ 
                 (Polynomial.eval x2 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) -> 
                 x1 + x2 = 3 := 
by
  intro x1 x2
  intro H
  sorry

end sum_of_roots_of_quadratic_l189_189697


namespace infinitely_many_triples_l189_189929

theorem infinitely_many_triples (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∀ k : ℕ, 
  ∃ (x y z : ℕ), 
    x = 2^(k * m * n + 1) ∧ 
    y = 2^(n + n * k * (m * n + 1)) ∧ 
    z = 2^(m + m * k * (m * n + 1)) ∧ 
    x^(m * n + 1) = y^m + z^n := 
by 
  intros k
  use 2^(k * m * n + 1), 2^(n + n * k * (m * n + 1)), 2^(m + m * k * (m * n + 1))
  simp
  sorry

end infinitely_many_triples_l189_189929


namespace volume_of_cube_l189_189752

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l189_189752


namespace evaluate_expression_l189_189479

theorem evaluate_expression : 40 + 5 * 12 / (180 / 3) = 41 :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l189_189479


namespace length_of_P1P2_segment_l189_189104

theorem length_of_P1P2_segment (x : ℝ) (h₀ : 0 < x ∧ x < π / 2) (h₁ : 6 * Real.cos x = 9 * Real.tan x) :
  Real.sin x = 1 / 2 :=
by
  sorry

end length_of_P1P2_segment_l189_189104


namespace remaining_perimeter_l189_189414

-- Definitions based on conditions
noncomputable def GH : ℝ := 2
noncomputable def HI : ℝ := 2
noncomputable def GI : ℝ := Real.sqrt (GH^2 + HI^2)
noncomputable def side_JKL : ℝ := 5
noncomputable def JI : ℝ := side_JKL - GH
noncomputable def IK : ℝ := side_JKL - HI
noncomputable def JK : ℝ := side_JKL

-- Problem statement in Lean 4
theorem remaining_perimeter :
  JI + IK + JK = 11 :=
by
  sorry

end remaining_perimeter_l189_189414


namespace part1_subsets_m_0_part2_range_m_l189_189310

namespace MathProof

variables {α : Type*} {m : ℝ}

def A := {x : ℝ | x^2 + 5 * x - 6 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 3 = 0}
def subsets (A : Set ℝ) := {s : Set ℝ | s ⊆ A}

theorem part1_subsets_m_0 :
  subsets (A ∪ B 0) = {∅, {-6}, {1}, {-3}, {-6,1}, {-6,-3}, {1,-3}, {-6,1,-3}} :=
sorry

theorem part2_range_m (h : ∀ x, x ∈ B m → x ∈ A) : m ≤ -2 :=
sorry

end MathProof

end part1_subsets_m_0_part2_range_m_l189_189310


namespace trigonometric_identity_l189_189919

theorem trigonometric_identity :
  Real.sin (17 * Real.pi / 180) * Real.sin (223 * Real.pi / 180) + 
  Real.sin (253 * Real.pi / 180) * Real.sin (313 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l189_189919


namespace soccer_tournament_solution_l189_189066

-- Define the statement of the problem
theorem soccer_tournament_solution (k : ℕ) (n m : ℕ) (h1 : k ≥ 1) (h2 : n = (k+1)^2) (h3 : m = k*(k+1) / 2)
  (h4 : n > m) : 
  ∃ k : ℕ, n = (k + 1) ^ 2 ∧ m = k * (k + 1) / 2 ∧ k ≥ 1 := 
sorry

end soccer_tournament_solution_l189_189066


namespace round_time_of_A_l189_189037

theorem round_time_of_A (T_a T_b : ℝ) 
  (h1 : 4 * T_b = 5 * T_a) 
  (h2 : 4 * T_b = 4 * T_a + 10) : T_a = 10 :=
by
  sorry

end round_time_of_A_l189_189037


namespace symmetric_inverse_sum_l189_189525

theorem symmetric_inverse_sum {f g : ℝ → ℝ} (h₁ : ∀ x, f (-x - 2) = -f (x)) (h₂ : ∀ y, g (f y) = y) (h₃ : ∀ y, f (g y) = y) (x₁ x₂ : ℝ) (h₄ : x₁ + x₂ = 0) : 
  g x₁ + g x₂ = -2 :=
by
  sorry

end symmetric_inverse_sum_l189_189525


namespace paint_needed_270_statues_l189_189406

theorem paint_needed_270_statues:
  let height_large := 12
  let paint_large := 2
  let height_small := 3
  let num_statues := 270
  let ratio_height := (height_small : ℝ) / (height_large : ℝ)
  let ratio_area := ratio_height ^ 2
  let paint_small := paint_large * ratio_area
  let total_paint := num_statues * paint_small
  total_paint = 33.75 := by
  sorry

end paint_needed_270_statues_l189_189406


namespace algebraic_expression_is_200_l189_189569

-- Define the condition
def satisfies_ratio (x : ℕ) : Prop :=
  x / 10 = 20

-- The proof problem statement
theorem algebraic_expression_is_200 : ∃ x : ℕ, satisfies_ratio x ∧ x = 200 :=
by
  -- Providing the necessary proof infrastructure
  use 200
  -- Assuming the proof is correct
  sorry


end algebraic_expression_is_200_l189_189569


namespace map_point_to_result_l189_189427

def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem map_point_to_result :
  f 2 0 = (2, 2) :=
by
  unfold f
  simp

end map_point_to_result_l189_189427


namespace total_cost_all_children_l189_189232

-- Defining the constants and conditions
def regular_tuition : ℕ := 45
def early_bird_discount : ℕ := 15
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10
def weekend_class_extra_cost : ℕ := 20
def multi_instrument_discount : ℕ := 10

def Ali_cost : ℕ := regular_tuition - early_bird_discount
def Matt_cost : ℕ := regular_tuition - first_sibling_discount
def Jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount
def Sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount

-- Proof statement
theorem total_cost_all_children : Ali_cost + Matt_cost + Jane_cost + Sarah_cost = 150 := by
  sorry

end total_cost_all_children_l189_189232


namespace factor_roots_l189_189421

theorem factor_roots (t : ℝ) : (x - t) ∣ (8 * x^2 + 18 * x - 5) ↔ (t = 1 / 4 ∨ t = -5) :=
by
  sorry

end factor_roots_l189_189421


namespace inequality_solution_l189_189422

theorem inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 5) →
  ((x * x - 4 * x - 5) / (x * x + 3 * x + 2) < 0 ↔ (x ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∨ x ∈ Set.Ioo (-1:ℝ) (5:ℝ))) :=
by
  sorry

end inequality_solution_l189_189422


namespace students_at_table_l189_189920

def numStudents (candies : ℕ) (first_last : ℕ) (st_len : ℕ) : Prop :=
  candies - 1 = st_len * first_last

theorem students_at_table 
  (candies : ℕ)
  (first_last : ℕ)
  (st_len : ℕ)
  (h1 : candies = 120) 
  (h2 : first_last = 1) :
  (st_len = 7 ∨ st_len = 17) :=
by
  sorry

end students_at_table_l189_189920


namespace p_scale_measurement_l189_189386

theorem p_scale_measurement (a b P S : ℝ) (h1 : 30 = 6 * a + b) (h2 : 60 = 24 * a + b) (h3 : 100 = a * P + b) : P = 48 :=
by
  sorry

end p_scale_measurement_l189_189386


namespace smallest_possible_area_of_2020th_square_l189_189314

theorem smallest_possible_area_of_2020th_square :
  ∃ A : ℕ, (∃ n : ℕ, n * n = 2019 + A) ∧ A ≠ 1 ∧
  ∀ A' : ℕ, A' > 0 ∧ (∃ n : ℕ, n * n = 2019 + A') ∧ A' ≠ 1 → A ≤ A' :=
by
  sorry

end smallest_possible_area_of_2020th_square_l189_189314


namespace find_A_l189_189926

theorem find_A (A : ℕ) (h : 10 * A + 2 - 23 = 549) : A = 5 :=
by sorry

end find_A_l189_189926


namespace minimize_sum_of_squares_l189_189051

open Real

-- Assume x, y are positive real numbers and x + y = s
variables {x y s : ℝ}
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = s)

theorem minimize_sum_of_squares :
  (x = y) ∧ (2 * x * x = s * s / 2) → (x = s / 2 ∧ y = s / 2 ∧ x^2 + y^2 = s^2 / 2) :=
by
  sorry

end minimize_sum_of_squares_l189_189051


namespace find_x_l189_189234

theorem find_x (x : ℚ) (h : 2 / 5 = (4 / 3) / x) : x = 10 / 3 :=
by
sorry

end find_x_l189_189234


namespace value_of_x_minus_y_l189_189549

theorem value_of_x_minus_y (x y a : ℝ) (h₁ : x + y > 0) (h₂ : a < 0) (h₃ : a * y > 0) : x - y > 0 :=
sorry

end value_of_x_minus_y_l189_189549


namespace average_age_is_35_l189_189848

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l189_189848


namespace gcd_polynomial_correct_l189_189846

noncomputable def gcd_polynomial (b : ℤ) := 5 * b^3 + b^2 + 8 * b + 38

theorem gcd_polynomial_correct (b : ℤ) (h : 342 ∣ b) : Int.gcd (gcd_polynomial b) b = 38 := by
  sorry

end gcd_polynomial_correct_l189_189846


namespace base_9_perfect_square_b_l189_189483

theorem base_9_perfect_square_b (b : ℕ) (a : ℕ) 
  (h0 : 0 < b) (h1 : b < 9) (h2 : a < 9) : 
  ∃ n, n^2 ≡ 729 * b + 81 * a + 54 [MOD 81] :=
sorry

end base_9_perfect_square_b_l189_189483


namespace number_of_ways_to_divide_friends_l189_189166

theorem number_of_ways_to_divide_friends :
  let friends := 8
  let teams := 4
  (teams ^ friends) = 65536 := by
  sorry

end number_of_ways_to_divide_friends_l189_189166


namespace domain_of_function_l189_189896

def function_undefined_at (x : ℝ) : Prop :=
  ∃ y : ℝ, y = (x - 3) / (x - 2)

theorem domain_of_function (x : ℝ) : ¬(x = 2) ↔ function_undefined_at x :=
sorry

end domain_of_function_l189_189896


namespace proof_expr_l189_189020

theorem proof_expr (a b c : ℤ) (h1 : a - b = 3) (h2 : b - c = 2) : (a - c)^2 + 3 * a + 1 - 3 * c = 41 := by {
  sorry
}

end proof_expr_l189_189020


namespace output_for_input_8_is_8_over_65_l189_189826

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l189_189826


namespace circle_center_sum_l189_189548

theorem circle_center_sum (x y : ℝ) (hx : (x, y) = (3, -4)) :
  (x + y) = -1 :=
by {
  -- We are given that the center of the circle is (3, -4)
  sorry -- Proof is omitted
}

end circle_center_sum_l189_189548


namespace order_of_magnitudes_l189_189843

variable (x : ℝ)
variable (a : ℝ)

theorem order_of_magnitudes (h1 : x < 0) (h2 : a = 2 * x) : x^2 < a * x ∧ a * x < a^2 := 
by
  sorry

end order_of_magnitudes_l189_189843


namespace some_number_value_l189_189045

theorem some_number_value (a : ℕ) (some_number : ℕ) (h_a : a = 105)
  (h_eq : a ^ 3 = some_number * 25 * 35 * 63) : some_number = 7 := by
  sorry

end some_number_value_l189_189045


namespace unit_vector_perpendicular_to_a_l189_189398

-- Definitions of a vector and the properties of unit and perpendicular vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def is_unit_vector (v : Vector2D) : Prop :=
  v.x ^ 2 + v.y ^ 2 = 1

def is_perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

-- Given vector a
def a : Vector2D := ⟨3, 4⟩

-- Coordinates of the unit vector that is perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ (b : Vector2D), is_unit_vector b ∧ is_perpendicular a b ∧
  (b = ⟨-4 / 5, 3 / 5⟩ ∨ b = ⟨4 / 5, -3 / 5⟩) :=
sorry

end unit_vector_perpendicular_to_a_l189_189398


namespace solution_set_inequality_range_of_t_l189_189777

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solution_set_inequality :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f (x - t) ≤ x - 2) ↔ 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
sorry

end solution_set_inequality_range_of_t_l189_189777


namespace find_x_minus_y_l189_189575

theorem find_x_minus_y (x y z : ℤ) (h₁ : x - y - z = 7) (h₂ : x - y + z = 15) : x - y = 11 := by
  sorry

end find_x_minus_y_l189_189575


namespace total_donation_correct_l189_189302

-- Define the donations to each orphanage
def first_orphanage_donation : ℝ := 175.00
def second_orphanage_donation : ℝ := 225.00
def third_orphanage_donation : ℝ := 250.00

-- State the total donation
def total_donation : ℝ := 650.00

-- The theorem statement to be proved
theorem total_donation_correct :
  first_orphanage_donation + second_orphanage_donation + third_orphanage_donation = total_donation :=
by
  sorry

end total_donation_correct_l189_189302


namespace prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l189_189939

-- Problem 1: Proof of sin(C - B) = 1 given the trigonometric identity
theorem prove_sin_c_minus_b_eq_one
  (A B C : ℝ)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  : Real.sin (C - B) = 1 := 
sorry

-- Problem 2: Proof of CD/BC given the ratios AB:AD:AC and the trigonometric identity
theorem prove_cd_div_bc_eq
  (A B C : ℝ)
  (AB AD AC BC CD : ℝ)
  (h_ratio : AB / AD = Real.sqrt 3 / Real.sqrt 2)
  (h_ratio_2 : AB / AC = Real.sqrt 3 / 1)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  (h_D_on_BC : True) -- Placeholder for D lies on BC condition
  : CD / BC = (Real.sqrt 5 - 1) / 2 := 
sorry

end prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l189_189939


namespace total_wheels_in_garage_l189_189145

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l189_189145


namespace arithmetic_sequence_sum_l189_189055

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (a_1 : ℚ) (d : ℚ) (m : ℕ) 
    (ha1 : a_1 = 2) 
    (ha2 : a 2 + a 8 = 24)
    (ham : 2 * a m = 24) 
    (h_sum : ∀ n, S n = (n * (2 * a_1 + (n - 1) * d)) / 2) 
    (h_an : ∀ n, a n = a_1 + (n - 1) * d) : 
    S (2 * m) = 265 / 2 :=
by
    sorry

end arithmetic_sequence_sum_l189_189055


namespace smallest_n_for_sum_or_difference_divisible_l189_189721

theorem smallest_n_for_sum_or_difference_divisible (n : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card = n → (∃ (x y : ℤ) (h₁ : x ≠ y), ((x + y) % 1991 = 0) ∨ ((x - y) % 1991 = 0))) ↔ n = 997 :=
sorry

end smallest_n_for_sum_or_difference_divisible_l189_189721


namespace division_of_product_l189_189689

theorem division_of_product :
  (1.6 * 0.5) / 1 = 0.8 :=
sorry

end division_of_product_l189_189689


namespace problem_solution_l189_189444

noncomputable def problem_statement : Prop :=
  8 * (Real.cos (25 * Real.pi / 180)) ^ 2 - Real.tan (40 * Real.pi / 180) - 4 = Real.sqrt 3

theorem problem_solution : problem_statement :=
by
sorry

end problem_solution_l189_189444


namespace jack_handing_in_amount_l189_189808

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l189_189808


namespace living_room_size_l189_189256

theorem living_room_size :
  let length := 16
  let width := 10
  let total_rooms := 6
  let total_area := length * width
  let unit_size := total_area / total_rooms
  let living_room_size := 3 * unit_size
  living_room_size = 80 := by
    sorry

end living_room_size_l189_189256


namespace other_root_eq_l189_189756

theorem other_root_eq (b : ℝ) : (∀ x, x^2 + b * x - 2 = 0 → (x = 1 ∨ x = -2)) :=
by
  intro x hx
  have : x = 1 ∨ x = -2 := sorry
  exact this

end other_root_eq_l189_189756


namespace largest_possible_a_l189_189818

theorem largest_possible_a :
  ∀ (a b c d : ℕ), a < 3 * b ∧ b < 4 * c ∧ c < 5 * d ∧ d < 80 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d → a ≤ 4724 := by
  sorry

end largest_possible_a_l189_189818


namespace mike_travel_distance_l189_189394

theorem mike_travel_distance
  (mike_start : ℝ := 2.50)
  (mike_per_mile : ℝ := 0.25)
  (annie_start : ℝ := 2.50)
  (annie_toll : ℝ := 5.00)
  (annie_per_mile : ℝ := 0.25)
  (annie_miles : ℝ := 14)
  (mike_cost : ℝ)
  (annie_cost : ℝ) :
  mike_cost = annie_cost → mike_cost = mike_start + mike_per_mile * 34 := by
  sorry

end mike_travel_distance_l189_189394


namespace A_eq_B_l189_189438

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l189_189438


namespace pizza_slices_l189_189854

theorem pizza_slices (S L : ℕ) (h1 : S + L = 36) (h2 : L = 2 * S) :
  (8 * S + 12 * L) = 384 :=
by
  sorry

end pizza_slices_l189_189854


namespace triangle_relations_l189_189568

theorem triangle_relations (A B C_1 C_2 C_3 : ℝ)
  (h1 : B > A)
  (h2 : C_2 > C_1 ∧ C_2 > C_3)
  (h3 : A + C_1 = 90) 
  (h4 : C_2 = 90)
  (h5 : B + C_3 = 90) :
  C_1 - C_3 = B - A :=
sorry

end triangle_relations_l189_189568


namespace jack_age_difference_l189_189531

def beckett_age : ℕ := 12
def olaf_age : ℕ := beckett_age + 3
def shannen_age : ℕ := olaf_age - 2
def total_age : ℕ := 71
def jack_age : ℕ := total_age - (beckett_age + olaf_age + shannen_age)
def difference := jack_age - 2 * shannen_age

theorem jack_age_difference :
  difference = 5 :=
by
  -- Math proof goes here
  sorry

end jack_age_difference_l189_189531


namespace factorize_expression_l189_189365

theorem factorize_expression (a b x y : ℝ) : 
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = ab * (x - y)^2 * (a * x - a * y - b) :=
by
  sorry

end factorize_expression_l189_189365


namespace sequence_recurrence_l189_189218

theorem sequence_recurrence (a : ℕ → ℝ) (h₀ : a 1 = 1) (h : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  intro n hn
  exact sorry

end sequence_recurrence_l189_189218


namespace goldfish_remaining_to_catch_l189_189276

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l189_189276


namespace number_of_possible_ordered_pairs_l189_189669

theorem number_of_possible_ordered_pairs (n : ℕ) (f m : ℕ) 
  (cond1 : n = 6) 
  (cond2 : f ≥ 0) 
  (cond3 : m ≥ 0) 
  (cond4 : f + m ≤ 12) 
  : ∃ s : Finset (ℕ × ℕ), s.card = 6 := 
by 
  sorry

end number_of_possible_ordered_pairs_l189_189669


namespace arithmetic_sequence_example_l189_189970

theorem arithmetic_sequence_example (a : ℕ → ℝ) (h : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) (h₁ : a 1 + a 19 = 10) : a 10 = 5 :=
by
  sorry

end arithmetic_sequence_example_l189_189970


namespace sum_seven_consecutive_integers_l189_189458

theorem sum_seven_consecutive_integers (m : ℕ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 :=
by
  -- Sorry to skip the actual proof steps.
  sorry

end sum_seven_consecutive_integers_l189_189458


namespace Tim_total_score_l189_189897

/-- Given the following conditions:
1. A single line is worth 1000 points.
2. A tetris is worth 8 times a single line.
3. If a single line and a tetris are made consecutively, the score of the tetris doubles.
4. If two tetrises are scored back to back, an additional 5000-point bonus is awarded.
5. If a player scores a single, double and triple line consecutively, a 3000-point bonus is awarded.
6. Tim scored 6 singles, 4 tetrises, 2 doubles, and 1 triple during his game.
7. He made a single line and a tetris consecutively once, scored 2 tetrises back to back, 
   and scored a single, double and triple consecutively.
Prove that Tim’s total score is 54000 points.
-/
theorem Tim_total_score :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6 * single_points
  let tetrises := 4 * tetris_points
  let base_score := singles + tetrises
  let consecutive_tetris_bonus := tetris_points
  let back_to_back_tetris_bonus := 5000
  let consecutive_lines_bonus := 3000
  let total_score := base_score + consecutive_tetris_bonus + back_to_back_tetris_bonus + consecutive_lines_bonus
  total_score = 54000 := by
  sorry

end Tim_total_score_l189_189897


namespace mark_spending_l189_189092

theorem mark_spending (initial_money : ℕ) (first_store_half : ℕ) (first_store_additional : ℕ) 
                      (second_store_third : ℕ) (remaining_money : ℕ) (total_spent : ℕ) : 
  initial_money = 180 ∧ 
  first_store_half = 90 ∧ 
  first_store_additional = 14 ∧ 
  total_spent = first_store_half + first_store_additional ∧
  remaining_money = initial_money - total_spent ∧
  second_store_third = 60 ∧ 
  remaining_money - second_store_third = 16 ∧ 
  initial_money - (total_spent + second_store_third + 16) = 0 → 
  remaining_money - second_store_third = 16 :=
by
  intro h
  sorry

end mark_spending_l189_189092


namespace solve_fraction_eq_l189_189207

theorem solve_fraction_eq (x : ℝ) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6) ↔ 
  (x = 7 ∨ x = -2) := 
by
  sorry

end solve_fraction_eq_l189_189207


namespace trajectory_of_M_l189_189708

theorem trajectory_of_M
  (x y : ℝ)
  (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
sorry

end trajectory_of_M_l189_189708


namespace sum_first_five_terms_l189_189766

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 d : ℝ, ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the specific condition a_5 + a_8 - a_10 = 2
def specific_condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 8 - a 10 = 2

-- Define the sum of the first five terms S₅
def S5 (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 

-- The statement to be proved
theorem sum_first_five_terms (a : ℕ → ℝ) (h₁ : arithmetic_sequence a) (h₂ : specific_condition a) : 
  S5 a = 10 :=
sorry

end sum_first_five_terms_l189_189766


namespace cannot_contain_2003_0_l189_189582

noncomputable def point_not_on_line (m b : ℝ) (h : m * b < 0) : Prop :=
  ∀ y : ℝ, ¬(0 = 2003 * m + b)

-- Prove that if m and b are real numbers and mb < 0, the line y = mx + b
-- cannot contain the point (2003, 0).
theorem cannot_contain_2003_0 (m b : ℝ) (h : m * b < 0) : point_not_on_line m b h :=
by
  sorry

end cannot_contain_2003_0_l189_189582


namespace quadratic_inequality_solution_l189_189340

theorem quadratic_inequality_solution :
  (∀ x : ℝ, x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3) → -9 * x^2 + 6 * x + 1 < 0) ∧
  (∀ x : ℝ, -9 * x^2 + 6 * x + 1 < 0 → x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3)) :=
by
  sorry

end quadratic_inequality_solution_l189_189340


namespace total_travel_distance_l189_189007

noncomputable def total_distance_traveled (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 - DF^2)
  DE + EF + DF

theorem total_travel_distance
  (DE DF : ℝ)
  (hDE : DE = 4500)
  (hDF : DF = 4000)
  : total_distance_traveled DE DF = 10560.992 :=
by
  rw [hDE, hDF]
  unfold total_distance_traveled
  norm_num
  sorry

end total_travel_distance_l189_189007


namespace parking_lot_total_spaces_l189_189892

-- Given conditions
def section1_spaces := 320
def section2_spaces := 440
def section3_spaces := section2_spaces - 200
def total_spaces := section1_spaces + section2_spaces + section3_spaces

-- Problem statement to be proved
theorem parking_lot_total_spaces : total_spaces = 1000 :=
by
  sorry

end parking_lot_total_spaces_l189_189892


namespace find_N_l189_189445

theorem find_N (N : ℕ) :
  ((5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N) → N = 1240 :=
by
  sorry

end find_N_l189_189445


namespace polynomial_function_correct_l189_189475

theorem polynomial_function_correct :
  ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f (x^2 + 1) = x^4 + 5 * x^2 + 3) →
  ∀ (x : ℝ), f (x^2 - 1) = x^4 + x^2 - 3 :=
by
  sorry

end polynomial_function_correct_l189_189475


namespace john_hourly_rate_with_bonus_l189_189056

theorem john_hourly_rate_with_bonus:
  ∀ (daily_wage : ℝ) (work_hours : ℕ) (bonus : ℝ) (extra_hours : ℕ),
    daily_wage = 80 →
    work_hours = 8 →
    bonus = 20 →
    extra_hours = 2 →
    (daily_wage + bonus) / (work_hours + extra_hours) = 10 :=
by
  intros daily_wage work_hours bonus extra_hours
  intros h1 h2 h3 h4
  -- sorry: the proof is omitted
  sorry

end john_hourly_rate_with_bonus_l189_189056


namespace cost_per_item_l189_189735

theorem cost_per_item (total_profit : ℝ) (total_customers : ℕ) (purchase_percentage : ℝ) (pays_advertising : ℝ)
    (H1: total_profit = 1000)
    (H2: total_customers = 100)
    (H3: purchase_percentage = 0.80)
    (H4: pays_advertising = 1000)
    : (total_profit / (total_customers * purchase_percentage)) = 12.50 :=
by
  sorry

end cost_per_item_l189_189735


namespace tom_age_ratio_l189_189204

theorem tom_age_ratio (T : ℕ) (h1 : T = 3 * (3 : ℕ)) (h2 : T - 5 = 3 * ((T / 3) - 10)) : T / 5 = 9 := 
by
  sorry

end tom_age_ratio_l189_189204


namespace probability_largest_ball_is_six_l189_189040

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_largest_ball_is_six : 
  (choose 6 4 : ℝ) / (choose 10 4 : ℝ) = (15 : ℝ) / (210 : ℝ) :=
by
  sorry

end probability_largest_ball_is_six_l189_189040


namespace intersection_eq_l189_189305

namespace SetIntersection

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Prove the intersection of A and B is {1, 2}
theorem intersection_eq : A ∩ B = {1, 2} :=
by
  sorry

end SetIntersection

end intersection_eq_l189_189305


namespace quadratic_conversion_l189_189370

theorem quadratic_conversion (x : ℝ) :
  (2*x - 1)^2 = (x + 1)*(3*x + 4) →
  ∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a*x^2 + b*x + c = 0 :=
by simp [pow_two, mul_add, add_mul, mul_comm]; sorry

end quadratic_conversion_l189_189370


namespace find_x_l189_189738

theorem find_x (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (2, x))
  (hc : c = (1, -2))
  (h_perpendicular : (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2)) = 0) :
  x = 1 / 2 :=
sorry

end find_x_l189_189738


namespace log_8_4000_l189_189957

theorem log_8_4000 : ∃ (n : ℤ), 8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 4000 ∧ 4000 < 4096 ∧ n = 4 :=
by
  sorry

end log_8_4000_l189_189957


namespace number_of_club_members_l189_189574

theorem number_of_club_members
  (num_committee : ℕ)
  (pair_of_committees_has_unique_member : ∀ (c1 c2 : Fin num_committee), c1 ≠ c2 → ∃! m : ℕ, c1 ≠ c2 ∧ c2 ≠ c1 ∧ m = m)
  (members_belong_to_two_committees : ∀ m : ℕ, ∃ (c1 c2 : Fin num_committee), c1 ≠ c2 ∧ m = m)
  : num_committee = 5 → ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end number_of_club_members_l189_189574


namespace measure_angle_C_and_area_l189_189215

noncomputable def triangleProblem (a b c A B C : ℝ) :=
  (a + b = 5) ∧ (c = Real.sqrt 7) ∧ (4 * Real.sin ((A + B) / 2)^2 - Real.cos (2 * C) = 7 / 2)

theorem measure_angle_C_and_area (a b c A B C : ℝ) (h: triangleProblem a b c A B C) :
  C = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  sorry

end measure_angle_C_and_area_l189_189215


namespace fraction_equivalence_l189_189352

theorem fraction_equivalence (a b : ℝ) (h : ((1 / a) + (1 / b)) / ((1 / a) - (1 / b)) = 2020) : (a + b) / (a - b) = 2020 :=
sorry

end fraction_equivalence_l189_189352


namespace triangle_circle_square_value_l189_189770

theorem triangle_circle_square_value (Δ : ℝ) (bigcirc : ℝ) (square : ℝ) 
  (h1 : 2 * Δ + 3 * bigcirc + square = 45)
  (h2 : Δ + 5 * bigcirc + 2 * square = 58)
  (h3 : 3 * Δ + bigcirc + 3 * square = 62) :
  Δ + 2 * bigcirc + square = 35 :=
sorry

end triangle_circle_square_value_l189_189770


namespace final_bicycle_price_is_225_l189_189699

noncomputable def final_selling_price (cp_A : ℝ) (profit_A : ℝ) (profit_B : ℝ) : ℝ :=
  let sp_B := cp_A * (1 + profit_A / 100)
  let sp_C := sp_B * (1 + profit_B / 100)
  sp_C

theorem final_bicycle_price_is_225 :
  final_selling_price 114.94 35 45 = 224.99505 :=
by
  sorry

end final_bicycle_price_is_225_l189_189699


namespace periodic_minus_decimal_is_correct_l189_189217

-- Definitions based on conditions

def periodic_63_as_fraction : ℚ := 63 / 99
def decimal_63_as_fraction : ℚ := 63 / 100
def difference : ℚ := periodic_63_as_fraction - decimal_63_as_fraction

-- Lean 4 statement to prove the mathematically equivalent proof problem
theorem periodic_minus_decimal_is_correct :
  difference = 7 / 1100 :=
by
  sorry

end periodic_minus_decimal_is_correct_l189_189217


namespace missing_fraction_correct_l189_189895

theorem missing_fraction_correct : 
  (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-2 / 15) + (3 / 5) = 0.13333333333333333 :=
by sorry

end missing_fraction_correct_l189_189895


namespace problem_1_problem_2_l189_189693

-- Definitions according to the conditions
def f (x a : ℝ) := |2 * x + a| + |x - 2|

-- The first part of the problem: Proof when a = -4, solve f(x) >= 6
theorem problem_1 (x : ℝ) : 
  f x (-4) ≥ 6 ↔ x ≤ 0 ∨ x ≥ 4 := by
  sorry

-- The second part of the problem: Prove the range of a for inequality f(x) >= 3a^2 - |2 - x|
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 * a^2 - |2 - x|) ↔ (-1 ≤ a ∧ a ≤ 4 / 3) := by
  sorry

end problem_1_problem_2_l189_189693


namespace inequality_proof_l189_189187

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b + c = 1) :
  a * (1 + b - c) ^ (1 / 3) + b * (1 + c - a) ^ (1 / 3) + c * (1 + a - b) ^ (1 / 3) ≤ 1 := 
by
  sorry

end inequality_proof_l189_189187


namespace monotonically_increasing_interval_l189_189725

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / 2 → (∀ y : ℝ, y < x → f y < f x) :=
by
  intro x h
  intro y hy
  sorry

end monotonically_increasing_interval_l189_189725


namespace triangle_area_is_24_l189_189126

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (0, 6)
def C : point := (8, 10)

def triangle_area (A B C : point) : ℝ := 
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_24 : triangle_area A B C = 24 :=
by
  -- Insert proof here
  sorry

end triangle_area_is_24_l189_189126


namespace probability_circle_l189_189465

theorem probability_circle (total_figures triangles circles squares : ℕ)
  (h_total : total_figures = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3) :
  circles / total_figures = 3 / 10 :=
by
  sorry

end probability_circle_l189_189465


namespace smaller_number_l189_189391

theorem smaller_number (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_l189_189391


namespace product_sum_l189_189297

theorem product_sum (y x z: ℕ) 
  (h1: 2014 + y = 2015 + x) 
  (h2: 2015 + x = 2016 + z) 
  (h3: y * x * z = 504): 
  y * x + x * z = 128 := 
by 
  sorry

end product_sum_l189_189297


namespace f_decreasing_on_negative_interval_and_min_value_l189_189409

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ x0, f x0 = m

-- Given the conditions
variables (condition1 : even_function f)
          (condition2 : increasing_on_interval f 3 7)
          (condition3 : minimum_value f 2)

-- Prove that f is decreasing on [-7,-3] and minimum value is 2
theorem f_decreasing_on_negative_interval_and_min_value :
  ∀ x y, -7 ≤ x → x ≤ y → y ≤ -3 → f y ≤ f x ∧ minimum_value f 2 :=
sorry

end f_decreasing_on_negative_interval_and_min_value_l189_189409


namespace domain_of_function_l189_189384

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x * (x - 1) ≥ 0) ↔ (x = 0 ∨ x ≥ 1) :=
by sorry

end domain_of_function_l189_189384


namespace solution_of_phi_l189_189442

theorem solution_of_phi 
    (φ : ℝ) 
    (H : ∃ k : ℤ, 2 * (π / 6) + φ = k * π) :
    φ = - (π / 3) := 
sorry

end solution_of_phi_l189_189442


namespace find_X_l189_189990

variable (X : ℝ)  -- Threshold income level for the lower tax rate
variable (I : ℝ)  -- Income of the citizen
variable (T : ℝ)  -- Total tax amount

-- Conditions
def income : Prop := I = 50000
def tax_amount : Prop := T = 8000
def tax_formula : Prop := T = 0.15 * X + 0.20 * (I - X)

theorem find_X (h1 : income I) (h2 : tax_amount T) (h3 : tax_formula T I X) : X = 40000 :=
by
  sorry

end find_X_l189_189990


namespace number_of_distinct_intersections_l189_189123

/-- The problem is to prove that the number of distinct intersection points
in the xy-plane for the graphs of the given equations is exactly 4. -/
theorem number_of_distinct_intersections :
  ∃ (S : Finset (ℝ × ℝ)), 
  (∀ p : ℝ × ℝ, p ∈ S ↔
    ((p.1 + p.2 = 7 ∨ 2 * p.1 - 3 * p.2 + 1 = 0) ∧
     (p.1 - p.2 - 2 = 0 ∨ 3 * p.1 + 2 * p.2 - 10 = 0))) ∧
  S.card = 4 :=
sorry

end number_of_distinct_intersections_l189_189123


namespace tan_x_value_l189_189197

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_x_value:
  (∀ x : ℝ, deriv f x = 2 * f x) → (∀ x : ℝ, f x = Real.sin x - Real.cos x) → (∀ x : ℝ, Real.tan x = 3) := 
by
  intros h_deriv h_f
  sorry

end tan_x_value_l189_189197


namespace cello_viola_pairs_are_70_l189_189147

-- Given conditions
def cellos : ℕ := 800
def violas : ℕ := 600
def pair_probability : ℝ := 0.00014583333333333335

-- Theorem statement translating the mathematical problem
theorem cello_viola_pairs_are_70 (n : ℕ) (h1 : cellos = 800) (h2 : violas = 600) (h3 : pair_probability = 0.00014583333333333335) :
  n = 70 :=
sorry

end cello_viola_pairs_are_70_l189_189147


namespace paul_and_lisa_total_dollars_l189_189710

def total_dollars_of_paul_and_lisa (paul_dol : ℚ) (lisa_dol : ℚ) : ℚ :=
  paul_dol + lisa_dol

theorem paul_and_lisa_total_dollars (paul_dol := (5 / 6 : ℚ)) (lisa_dol := (2 / 5 : ℚ)) :
  total_dollars_of_paul_and_lisa paul_dol lisa_dol = (123 / 100 : ℚ) :=
by
  sorry

end paul_and_lisa_total_dollars_l189_189710


namespace least_possible_value_f_1998_l189_189403

theorem least_possible_value_f_1998 
  (f : ℕ → ℕ)
  (h : ∀ m n, f (n^2 * f m) = m * (f n)^2) : 
  f 1998 = 120 :=
sorry

end least_possible_value_f_1998_l189_189403


namespace number_of_possible_values_for_a_l189_189075

theorem number_of_possible_values_for_a 
  (a b c d : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a > b) (h6 : b > c) (h7 : c > d)
  (h8 : a + b + c + d = 2004)
  (h9 : a^2 - b^2 - c^2 + d^2 = 1004) : 
  ∃ n : ℕ, n = 500 :=
  sorry

end number_of_possible_values_for_a_l189_189075


namespace students_not_opt_for_math_l189_189046

theorem students_not_opt_for_math (total_students S E both_subjects M : ℕ) 
    (h1 : total_students = 40) 
    (h2 : S = 15) 
    (h3 : E = 2) 
    (h4 : both_subjects = 7) 
    (h5 : total_students - both_subjects = M + S - E) : M = 20 := 
  by
  sorry

end students_not_opt_for_math_l189_189046


namespace max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l189_189095

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2 * b|

theorem max_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x, f x a b ≤ a + 2 * b :=
by sorry

theorem min_value_of_a2_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 :=
by sorry

theorem min_value_of_a2_4b2_equals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  ∃ a b, a = 1 / 2 ∧ b = 1 / 4 ∧ (a^2 + 4 * b^2 = 1 / 2) :=
by sorry

end max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l189_189095


namespace jerry_current_average_l189_189969

-- Definitions for Jerry's first 3 tests average and conditions
variable (A : ℝ)

-- Condition details
def total_score_of_first_3_tests := 3 * A
def new_desired_average := A + 2
def total_score_needed := (A + 2) * 4
def score_on_fourth_test := 93

theorem jerry_current_average :
  (total_score_needed A = total_score_of_first_3_tests A + score_on_fourth_test) → A = 85 :=
by
  sorry

end jerry_current_average_l189_189969


namespace tan_sub_pi_div_four_eq_neg_seven_f_range_l189_189706

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof for the first part
theorem tan_sub_pi_div_four_eq_neg_seven (x : ℝ) (h : 3 / 4 * Real.cos x + Real.sin x = 0) :
  Real.tan (x - Real.pi / 4) = -7 := sorry

noncomputable def f (x : ℝ) : ℝ := 
  2 * ((a x).fst + (b x).fst) * (b x).fst + 2 * ((a x).snd + (b x).snd) * (b x).snd

-- Proof for the second part
theorem f_range (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  1 / 2 < f x ∧ f x < 3 / 2 + Real.sqrt 2 := sorry

end tan_sub_pi_div_four_eq_neg_seven_f_range_l189_189706


namespace small_panda_bears_count_l189_189740

theorem small_panda_bears_count :
  ∃ (S : ℕ), ∃ (B : ℕ),
    B = 5 ∧ 7 * (25 * S + 40 * B) = 2100 ∧ S = 4 :=
by
  exists 4
  exists 5
  repeat { sorry }

end small_panda_bears_count_l189_189740


namespace find_y_l189_189627

theorem find_y 
  (y : ℝ) 
  (h1 : (y^2 - 11 * y + 24) / (y - 3) + (2 * y^2 + 7 * y - 18) / (2 * y - 3) = -10)
  (h2 : y ≠ 3)
  (h3 : y ≠ 3 / 2) : 
  y = -4 := 
sorry

end find_y_l189_189627


namespace correct_sum_104th_parenthesis_l189_189397

noncomputable def sum_104th_parenthesis : ℕ := sorry

theorem correct_sum_104th_parenthesis :
  sum_104th_parenthesis = 2072 := 
by 
  sorry

end correct_sum_104th_parenthesis_l189_189397


namespace solve_for_a_l189_189874

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l189_189874


namespace cos_960_eq_neg_half_l189_189868

theorem cos_960_eq_neg_half (cos : ℝ → ℝ) (h1 : ∀ x, cos (x + 360) = cos x) 
  (h_even : ∀ x, cos (-x) = cos x) (h_cos120 : cos 120 = - cos 60)
  (h_cos60 : cos 60 = 1 / 2) : cos 960 = -(1 / 2) := by
  sorry

end cos_960_eq_neg_half_l189_189868


namespace caffeine_in_cup_l189_189943

-- Definitions based on the conditions
def caffeine_goal : ℕ := 200
def excess_caffeine : ℕ := 40
def total_cups : ℕ := 3

-- The statement proving that the amount of caffeine in a cup is 80 mg given the conditions.
theorem caffeine_in_cup : (3 * (80 : ℕ)) = (caffeine_goal + excess_caffeine) := by
  -- Plug in the value and simplify
  simp [caffeine_goal, excess_caffeine]

end caffeine_in_cup_l189_189943


namespace volume_to_surface_area_ratio_l189_189887

theorem volume_to_surface_area_ratio (base_layer: ℕ) (top_layer: ℕ) (unit_cube_volume: ℕ) (unit_cube_faces_exposed_base: ℕ) (unit_cube_faces_exposed_top: ℕ) 
  (V : ℕ := base_layer * top_layer * unit_cube_volume) 
  (S : ℕ := base_layer * unit_cube_faces_exposed_base + top_layer * unit_cube_faces_exposed_top) 
  (ratio := V / S) : ratio = 1 / 2 :=
by
  -- Base Layer: 4 cubes, 3 faces exposed per cube
  have base_layer_faces : ℕ := 4 * 3
  -- Top Layer: 4 cubes, 1 face exposed per cube
  have top_layer_faces : ℕ := 4 * 1
  -- Total volume is 8
  have V : ℕ := 4 * 2
  -- Total surface area is 16
  have S : ℕ := base_layer_faces + top_layer_faces
  -- Volume to surface area ratio computation
  have ratio : ℕ := V / S
  sorry

end volume_to_surface_area_ratio_l189_189887


namespace equivalent_expression_l189_189177

theorem equivalent_expression (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h1 : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by
  sorry

end equivalent_expression_l189_189177


namespace carol_additional_cupcakes_l189_189412

-- Define the initial number of cupcakes Carol made
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes Carol sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes Carol wanted to have
def total_cupcakes : ℕ := 49

-- Calculate the number of cupcakes Carol had left after selling
def remaining_cupcakes : ℕ := initial_cupcakes - sold_cupcakes

-- The number of additional cupcakes Carol made can be defined and proved as follows:
theorem carol_additional_cupcakes : initial_cupcakes - sold_cupcakes + 28 = total_cupcakes :=
by
  -- left side: initial_cupcakes (30) - sold_cupcakes (9) + additional_cupcakes (28) = total_cupcakes (49)
  sorry

end carol_additional_cupcakes_l189_189412


namespace negation_of_universal_proposition_l189_189986

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 3 * x + 2 > 0)) ↔ (∃ x : ℝ, x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l189_189986


namespace polygon_diagonals_l189_189390

theorem polygon_diagonals (n : ℕ) (h : n - 3 ≤ 6) : n = 9 :=
by sorry

end polygon_diagonals_l189_189390


namespace binomial_coeff_equal_l189_189774

theorem binomial_coeff_equal (n : ℕ) (h₁ : 6 ≤ n) (h₂ : (n.choose 5) * 3^5 = (n.choose 6) * 3^6) :
  n = 7 := sorry

end binomial_coeff_equal_l189_189774


namespace nickels_count_l189_189033

theorem nickels_count (N Q : ℕ) 
  (h_eq : N = Q) 
  (h_total_value : 5 * N + 25 * Q = 1200) :
  N = 40 := 
by 
  sorry

end nickels_count_l189_189033


namespace distance_A_C_15_l189_189184

noncomputable def distance_from_A_to_C : ℝ := 
  let AB := 6
  let AC := AB + (3 * AB) / 2
  AC

theorem distance_A_C_15 (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 24) (h5 : D - B = 3 * (B - A)) 
  (h6 : C = (B + D) / 2) :
  distance_from_A_to_C = 15 :=
by sorry

end distance_A_C_15_l189_189184


namespace baked_goods_not_eaten_l189_189482

theorem baked_goods_not_eaten : 
  let cookies_initial := 200
  let brownies_initial := 150
  let cupcakes_initial := 100
  
  let cookies_after_wife := cookies_initial - 0.30 * cookies_initial
  let brownies_after_wife := brownies_initial - 0.20 * brownies_initial
  let cupcakes_after_wife := cupcakes_initial / 2
  
  let cookies_after_daughter := cookies_after_wife - 40
  let brownies_after_daughter := brownies_after_wife - 0.15 * brownies_after_wife
  
  let cookies_after_friend := cookies_after_daughter - (cookies_after_daughter / 4)
  let brownies_after_friend := brownies_after_daughter - 0.10 * brownies_after_daughter
  let cupcakes_after_friend := cupcakes_after_wife - 10
  
  let cookies_after_other_friend := cookies_after_friend - 0.05 * cookies_after_friend
  let brownies_after_other_friend := brownies_after_friend - 0.05 * brownies_after_friend
  let cupcakes_after_other_friend := cupcakes_after_friend - 5
  
  let cookies_after_javier := cookies_after_other_friend / 2
  let brownies_after_javier := brownies_after_other_friend / 2
  let cupcakes_after_javier := cupcakes_after_other_friend / 2
  
  let total_remaining := cookies_after_javier + brownies_after_javier + cupcakes_after_javier
  total_remaining = 98 := by
{
  sorry
}

end baked_goods_not_eaten_l189_189482


namespace value_of_x_plus_4_l189_189219

theorem value_of_x_plus_4 (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 :=
by
  sorry

end value_of_x_plus_4_l189_189219


namespace probability_of_reaching_3_1_without_2_0_in_8_steps_l189_189978

theorem probability_of_reaching_3_1_without_2_0_in_8_steps :
  let n_total := 1680
  let invalid := 30
  let total := n_total - invalid
  let q := total / 4^8
  let gcd := Nat.gcd total 65536
  let m := total / gcd
  let n := 65536 / gcd
  (m + n = 11197) :=
by
  sorry

end probability_of_reaching_3_1_without_2_0_in_8_steps_l189_189978


namespace paper_cups_calculation_l189_189803

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l189_189803


namespace total_collection_value_l189_189183

theorem total_collection_value (total_stickers : ℕ) (partial_stickers : ℕ) (partial_value : ℕ)
  (same_value : ∀ (stickers : ℕ), stickers = total_stickers → stickers * partial_value / partial_stickers = stickers * (partial_value / partial_stickers)):
  partial_value = 24 ∧ partial_stickers = 6 ∧ total_stickers = 18 → total_stickers * (partial_value / partial_stickers) = 72 :=
by {
  sorry
}

end total_collection_value_l189_189183


namespace remaining_mushroom_pieces_l189_189661

theorem remaining_mushroom_pieces 
  (mushrooms : ℕ) 
  (pieces_per_mushroom : ℕ) 
  (pieces_used_by_kenny : ℕ) 
  (pieces_used_by_karla : ℕ) 
  (mushrooms_cut : mushrooms = 22) 
  (pieces_per_mushroom_def : pieces_per_mushroom = 4) 
  (kenny_pieces_def : pieces_used_by_kenny = 38) 
  (karla_pieces_def : pieces_used_by_karla = 42) : 
  (mushrooms * pieces_per_mushroom - (pieces_used_by_kenny + pieces_used_by_karla)) = 8 := 
by 
  sorry

end remaining_mushroom_pieces_l189_189661


namespace geometric_progression_common_ratio_l189_189343

theorem geometric_progression_common_ratio :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) :=
by
  sorry

end geometric_progression_common_ratio_l189_189343


namespace judy_pencil_cost_l189_189552

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end judy_pencil_cost_l189_189552


namespace difference_between_scores_l189_189849

variable (H F : ℕ)
variable (h_hajar_score : H = 24)
variable (h_sum_scores : H + F = 69)
variable (h_farah_higher : F > H)

theorem difference_between_scores : F - H = 21 := by
  sorry

end difference_between_scores_l189_189849


namespace situation1_correct_situation2_correct_situation3_correct_l189_189535

noncomputable def situation1 : Nat :=
  let choices_for_A := 4
  let remaining_perm := Nat.factorial 6
  choices_for_A * remaining_perm

theorem situation1_correct : situation1 = 2880 := by
  sorry

noncomputable def situation2 : Nat :=
  let permutations_A_B := Nat.factorial 2
  let remaining_perm := Nat.factorial 5
  permutations_A_B * remaining_perm

theorem situation2_correct : situation2 = 240 := by
  sorry

noncomputable def situation3 : Nat :=
  let perm_boys := Nat.factorial 3
  let perm_girls := Nat.factorial 4
  perm_boys * perm_girls

theorem situation3_correct : situation3 = 144 := by
  sorry

end situation1_correct_situation2_correct_situation3_correct_l189_189535


namespace lucas_min_deliveries_l189_189730

theorem lucas_min_deliveries (cost_of_scooter earnings_per_delivery fuel_cost_per_delivery parking_fee_per_delivery : ℕ)
  (cost_eq : cost_of_scooter = 3000)
  (earnings_eq : earnings_per_delivery = 12)
  (fuel_cost_eq : fuel_cost_per_delivery = 4)
  (parking_fee_eq : parking_fee_per_delivery = 1) :
  ∃ d : ℕ, 7 * d ≥ cost_of_scooter ∧ d = 429 := by
  sorry

end lucas_min_deliveries_l189_189730


namespace multiple_of_27_l189_189499

theorem multiple_of_27 (x y z : ℤ) 
  (h1 : (2 * x + 5 * y + 11 * z) = 4 * (x + y + z)) 
  (h2 : (2 * x + 20 * y + 110 * z) = 6 * (2 * x + 5 * y + 11 * z)) :
  ∃ k : ℤ, x + y + z = 27 * k :=
by
  sorry

end multiple_of_27_l189_189499


namespace incorrect_statement_C_l189_189511

theorem incorrect_statement_C :
  (∀ (b h : ℝ), b > 0 → h > 0 → 2 * (b * h) = (2 * b) * h) ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → 2 * (π * r^2 * h) = π * r^2 * (2 * h)) ∧
  (∀ (a : ℝ), a > 0 → 4 * (a^3) ≠ (2 * a)^3) ∧
  (∀ (a b : ℚ), b ≠ 0 → a / (2 * b) ≠ (a / 2) / b) ∧
  (∀ (x : ℝ), x < 0 → 2 * x < x) :=
by
  sorry

end incorrect_statement_C_l189_189511


namespace cube_volume_from_surface_area_l189_189998

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l189_189998


namespace problem1_correct_problem2_correct_l189_189523

-- Definition for Problem 1
def problem1 (a b c d : ℚ) : ℚ :=
  (a - b + c) * d

-- Statement for Problem 1
theorem problem1_correct : problem1 (1/6) (5/7) (2/3) (-42) = -5 :=
by
  sorry

-- Definitions for Problem 2
def problem2 (a b c d : ℚ) : ℚ :=
  (-a^2 + b^2 * c - d^2 / |d|)

-- Statement for Problem 2
theorem problem2_correct : problem2 (-2) (-3) (-2/3) 4 = -14 :=
by
  sorry

end problem1_correct_problem2_correct_l189_189523


namespace sum_infinite_geometric_series_l189_189602

theorem sum_infinite_geometric_series :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  (a / (1 - r) = (3 : ℚ) / 8) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  sorry

end sum_infinite_geometric_series_l189_189602


namespace min_people_liking_both_l189_189129

theorem min_people_liking_both (total : ℕ) (Beethoven : ℕ) (Chopin : ℕ) 
    (total_eq : total = 150) (Beethoven_eq : Beethoven = 120) (Chopin_eq : Chopin = 95) : 
    ∃ (both : ℕ), both = 65 := 
by 
  have H := Beethoven + Chopin - total
  sorry

end min_people_liking_both_l189_189129


namespace multiple_of_persons_l189_189983

variable (Persons Work : ℕ) (Rate : ℚ)

def work_rate (P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D
def multiple_work_rate (m P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D

theorem multiple_of_persons
  (P : ℕ) (W : ℕ)
  (h1 : work_rate P W 12 = W / 12)
  (h2 : multiple_work_rate 1 P (W / 2) 3 = (W / 6)) :
  m = 2 :=
by sorry

end multiple_of_persons_l189_189983


namespace compute_expression_l189_189812

theorem compute_expression : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 :=
by
  sorry

end compute_expression_l189_189812


namespace mary_prevents_pat_l189_189486

noncomputable def smallest_initial_integer (N: ℕ) : Prop :=
  N > 2017 ∧ 
  ∀ x, ∃ n: ℕ, 
  (x = N + n * 2018 → x % 2018 ≠ 0 ∧
   (2017 * x + 2) % 2018 ≠ 0 ∧
   (2017 * x + 2021) % 2018 ≠ 0)

theorem mary_prevents_pat (N : ℕ) : smallest_initial_integer N → N = 2022 :=
sorry

end mary_prevents_pat_l189_189486


namespace find_a_l189_189091

def point_of_tangency (x0 y0 a : ℝ) : Prop :=
  (x0 - y0 - 1 = 0) ∧ (y0 = a * x0^2) ∧ (2 * a * x0 = 1)

theorem find_a (x0 y0 a : ℝ) (h : point_of_tangency x0 y0 a) : a = 1/4 :=
by
  sorry

end find_a_l189_189091


namespace sqrt_diff_approx_l189_189171

theorem sqrt_diff_approx : abs ((Real.sqrt 122) - (Real.sqrt 120) - 0.15) < 0.01 := 
sorry

end sqrt_diff_approx_l189_189171


namespace solve_for_w_l189_189557

theorem solve_for_w (w : ℝ) : (2 : ℝ)^(2 * w) = (8 : ℝ)^(w - 4) → w = 12 := by
  sorry

end solve_for_w_l189_189557


namespace bug_total_distance_l189_189332

/-- 
A bug starts at position 3 on a number line. It crawls to -4, then to 7, and finally to 1.
The total distance the bug crawls is 24 units.
-/
theorem bug_total_distance : 
  let start := 3
  let first_stop := -4
  let second_stop := 7
  let final_position := 1
  let distance := abs (first_stop - start) + abs (second_stop - first_stop) + abs (final_position - second_stop)
  distance = 24 := 
by
  sorry

end bug_total_distance_l189_189332


namespace fraction_painted_red_l189_189947

theorem fraction_painted_red :
  let matilda_section := (1:ℚ) / 2 -- Matilda's half section
  let ellie_section := (1:ℚ) / 2    -- Ellie's half section
  let matilda_painted := matilda_section / 2 -- Matilda's painted fraction
  let ellie_painted := ellie_section / 3    -- Ellie's painted fraction
  (matilda_painted + ellie_painted) = 5 / 12 := 
by
  sorry

end fraction_painted_red_l189_189947


namespace total_bones_in_graveyard_l189_189516

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l189_189516


namespace problem_D_l189_189616

variable (f : ℕ → ℝ)

-- Function condition: If f(k) ≥ k^2, then f(k+1) ≥ (k+1)^2
axiom f_property (k : ℕ) (hk : f k ≥ k^2) : f (k + 1) ≥ (k + 1)^2

theorem problem_D (hf4 : f 4 ≥ 25) : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_D_l189_189616


namespace total_students_l189_189121

-- Define the conditions
variables (S : ℕ) -- total number of students
variable (h1 : (3/5 : ℚ) * S + (1/5 : ℚ) * S + 10 = S)

-- State the theorem
theorem total_students (HS : S = 50) : 3 / 5 * S + 1 / 5 * S + 10 = S := by
  -- Here we declare the proof is to be filled in later.
  sorry

end total_students_l189_189121


namespace smallest_n_divisible_31_l189_189785

theorem smallest_n_divisible_31 (n : ℕ) : 31 ∣ (5 ^ n + n) → n = 30 :=
by
  sorry

end smallest_n_divisible_31_l189_189785


namespace remainder_mul_mod_l189_189471

theorem remainder_mul_mod (a b n : ℕ) (h₁ : a ≡ 3 [MOD n]) (h₂ : b ≡ 150 [MOD n]) (n_eq : n = 400) : 
  (a * b) % n = 50 :=
by 
  sorry

end remainder_mul_mod_l189_189471


namespace symmetry_implies_condition_l189_189451

open Function

variable {R : Type*} [Field R]
variables (p q r s : R)

theorem symmetry_implies_condition
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0) 
  (h_symmetry : ∀ x y : R, y = (p * x + q) / (r * x - s) → 
                          -x = (p * (-y) + q) / (r * (-y) - s)) :
  r + s = 0 := 
sorry

end symmetry_implies_condition_l189_189451


namespace sum_coordinates_B_l189_189901

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B (x : ℝ) : (ℝ × ℝ) := (x, 4)

theorem sum_coordinates_B 
  (x : ℝ) 
  (h_slope : (4 - 0)/(x - 0) = 3/4) : x + 4 = 28 / 3 := by
sorry

end sum_coordinates_B_l189_189901


namespace oranges_to_apples_ratio_l189_189182

theorem oranges_to_apples_ratio :
  ∀ (total_fruits : ℕ) (weight_oranges : ℕ) (weight_apples : ℕ),
  total_fruits = 12 →
  weight_oranges = 10 →
  weight_apples = total_fruits - weight_oranges →
  weight_oranges / weight_apples = 5 :=
by
  intros total_fruits weight_oranges weight_apples h1 h2 h3
  sorry

end oranges_to_apples_ratio_l189_189182


namespace part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l189_189913

-- Definitions for the sequences
def first_row (n : ℕ) : ℤ := (-3) ^ n
def second_row (n : ℕ) : ℤ := (-3) ^ n - 3
def third_row (n : ℕ) : ℤ := -((-3) ^ n) - 1

-- Statement for part 1
theorem part1_fifth_numbers:
  first_row 5 = -243 ∧ second_row 5 = -246 ∧ third_row 5 = 242 := sorry

-- Statement for part 2
theorem part2_three_adjacent_sum :
  ∃ n : ℕ, first_row (n-1) + first_row n + first_row (n+1) = -1701 ∧
           first_row (n-1) = -243 ∧ first_row n = 729 ∧ first_row (n+1) = -2187 := sorry

-- Statement for part 3
def sum_nth (n : ℕ) : ℤ := first_row n + second_row n + third_row n
theorem part3_difference_largest_smallest (n : ℕ) (m : ℤ) (hn : sum_nth n = m) :
  (∃ diff, (n % 2 = 1 → diff = -2 * m - 6) ∧ (n % 2 = 0 → diff = 2 * m + 9)) := sorry

end part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l189_189913


namespace find_set_of_x_l189_189241

noncomputable def exponential_inequality_solution (x : ℝ) : Prop :=
  1 < Real.exp x ∧ Real.exp x < 2

theorem find_set_of_x (x : ℝ) :
  exponential_inequality_solution x ↔ 0 < x ∧ x < Real.log 2 :=
by
  sorry

end find_set_of_x_l189_189241


namespace beetle_distance_l189_189164

theorem beetle_distance :
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  dist1 + dist2 = 20 :=
by
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  show dist1 + dist2 = 20
  sorry

end beetle_distance_l189_189164


namespace correct_value_calculation_l189_189530

theorem correct_value_calculation (x : ℤ) (h : 2 * (x + 6) = 28) : 6 * x = 48 :=
by
  -- Proof steps would be here
  sorry

end correct_value_calculation_l189_189530


namespace each_cow_gives_5_liters_per_day_l189_189753

-- Define conditions
def cows : ℕ := 52
def weekly_milk : ℕ := 1820
def days_in_week : ℕ := 7

-- Define daily_milk as the daily milk production
def daily_milk := weekly_milk / days_in_week

-- Define milk_per_cow as the amount of milk each cow produces per day
def milk_per_cow := daily_milk / cows

-- Statement to prove
theorem each_cow_gives_5_liters_per_day : milk_per_cow = 5 :=
by
  -- This is where you would normally fill in the proof steps
  sorry

end each_cow_gives_5_liters_per_day_l189_189753


namespace restaurant_tip_difference_l189_189974

theorem restaurant_tip_difference
  (a b : ℝ)
  (h1 : 0.15 * a = 3)
  (h2 : 0.25 * b = 3)
  : a - b = 8 := 
sorry

end restaurant_tip_difference_l189_189974


namespace grains_in_gray_parts_l189_189635

theorem grains_in_gray_parts (total1 total2 shared : ℕ) (h1 : total1 = 87) (h2 : total2 = 110) (h_shared : shared = 68) :
  (total1 - shared) + (total2 - shared) = 61 :=
by sorry

end grains_in_gray_parts_l189_189635


namespace simplify_expression_correct_l189_189236

def simplify_expression : ℚ :=
  15 * (7 / 10) * (1 / 9)

theorem simplify_expression_correct : simplify_expression = 7 / 6 :=
by
  unfold simplify_expression
  sorry

end simplify_expression_correct_l189_189236


namespace lines_parallel_l189_189395

theorem lines_parallel (m : ℝ) : 
  (m = 2 ↔ ∀ x y : ℝ, (2 * x - m * y - 1 = 0) ∧ ((m - 1) * x - y + 1 = 0) → 
  (∃ k : ℝ, (2 * x - m * y - 1 = k * ((m - 1) * x - y + 1)))) :=
by sorry

end lines_parallel_l189_189395


namespace find_smaller_number_l189_189052

def smaller_number (x y : ℕ) : ℕ :=
  if x < y then x else y

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 64) (h2 : a = b + 12) : smaller_number a b = 26 :=
by
  sorry

end find_smaller_number_l189_189052


namespace contractor_absent_days_l189_189180

theorem contractor_absent_days
    (total_days : ℤ) (work_rate : ℤ) (fine_rate : ℤ) (total_amount : ℤ)
    (x y : ℤ)
    (h1 : total_days = 30)
    (h2 : work_rate = 25)
    (h3 : fine_rate = 75) -- fine_rate here is multiplied by 10 to avoid decimals
    (h4 : total_amount = 4250) -- total_amount multiplied by 10 for the same reason
    (h5 : x + y = total_days)
    (h6 : work_rate * x - fine_rate * y = total_amount) :
  y = 10 := 
by
  -- Here, we would provide the proof steps.
  sorry

end contractor_absent_days_l189_189180


namespace tangent_line_b_value_l189_189222

theorem tangent_line_b_value (a k b : ℝ) 
  (h_curve : ∀ x, x^3 + a * x + 1 = 3 ↔ x = 2)
  (h_derivative : k = 3 * 2^2 - 3)
  (h_tangent : 3 = k * 2 + b) : b = -15 :=
sorry

end tangent_line_b_value_l189_189222


namespace correct_option_D_l189_189328

theorem correct_option_D (a : ℝ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end correct_option_D_l189_189328


namespace problem_I_problem_II_l189_189688

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (1 - a) * x
noncomputable def h (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - Real.log x)

theorem problem_I (a : ℝ) (ha : a > 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → deriv (f a) x > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < a → deriv (f a) x < 0) ∧
  (∀ x : ℝ, a < x → deriv (f a) x > 0) :=
sorry

theorem problem_II (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ f a x₀ ≥ g a x₀) ↔ a ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
sorry

end problem_I_problem_II_l189_189688


namespace inequality_proof_l189_189768

open Real

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (hSum : x + y + z = 1) :
  x * y / sqrt (x * y + y * z) + y * z / sqrt (y * z + z * x) + z * x / sqrt (z * x + x * y) ≤ sqrt 2 / 2 := 
sorry

end inequality_proof_l189_189768


namespace additional_discount_percentage_l189_189611

theorem additional_discount_percentage
  (MSRP : ℝ)
  (p : ℝ)
  (d : ℝ)
  (sale_price : ℝ)
  (H1 : MSRP = 45.0)
  (H2 : p = 0.30)
  (H3 : d = MSRP - (p * MSRP))
  (H4 : d = 31.50)
  (H5 : sale_price = 25.20) :
  sale_price = d - (0.20 * d) :=
by
  sorry

end additional_discount_percentage_l189_189611


namespace time_to_eat_cereal_l189_189761

noncomputable def MrFatRate : ℝ := 1 / 40
noncomputable def MrThinRate : ℝ := 1 / 15
noncomputable def CombinedRate : ℝ := MrFatRate + MrThinRate
noncomputable def CerealAmount : ℝ := 4
noncomputable def TimeToFinish : ℝ := CerealAmount / CombinedRate
noncomputable def expected_time : ℝ := 96

theorem time_to_eat_cereal :
  TimeToFinish = expected_time :=
by
  sorry

end time_to_eat_cereal_l189_189761


namespace infinite_sum_fraction_equals_quarter_l189_189662

theorem infinite_sum_fraction_equals_quarter :
  (∑' n : ℕ, (3 ^ n) / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1))) = 1 / 4 :=
by
  -- With the given conditions, we need to prove the above statement
  -- The conditions have been used to express the problem in Lean
  sorry

end infinite_sum_fraction_equals_quarter_l189_189662


namespace homer_total_points_l189_189845

noncomputable def first_try_points : ℕ := 400
noncomputable def second_try_points : ℕ := first_try_points - 70
noncomputable def third_try_points : ℕ := 2 * second_try_points
noncomputable def total_points : ℕ := first_try_points + second_try_points + third_try_points

theorem homer_total_points : total_points = 1390 :=
by
  -- Using the definitions above, we need to show that total_points = 1390
  sorry

end homer_total_points_l189_189845


namespace train_speed_l189_189931

theorem train_speed (length_train length_platform : ℝ) (time : ℝ) 
  (h_length_train : length_train = 170.0416) 
  (h_length_platform : length_platform = 350) 
  (h_time : time = 26) : 
  (length_train + length_platform) / time * 3.6 = 72 :=
by 
  sorry

end train_speed_l189_189931


namespace greatest_possible_median_l189_189345

theorem greatest_possible_median {k m r s t : ℕ} 
  (h_mean : (k + m + r + s + t) / 5 = 18) 
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) 
  (h_t : t = 40) :
  r = 23 := sorry

end greatest_possible_median_l189_189345


namespace smallest_sum_of_sequence_l189_189728

theorem smallest_sum_of_sequence {
  A B C D k : ℕ
} (h1 : 2 * B = A + C)
  (h2 : D - C = (C - B) ^ 2)
  (h3 : 4 * B = 3 * C)
  (h4 : B = 3 * k)
  (h5 : C = 4 * k)
  (h6 : A = 2 * k)
  (h7 : D = 4 * k + k ^ 2) :
  A + B + C + D = 14 :=
by
  sorry

end smallest_sum_of_sequence_l189_189728


namespace function_machine_output_is_38_l189_189440

def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  if multiplied > 40 then
    multiplied - 7
  else
    multiplied + 10

theorem function_machine_output_is_38 :
  function_machine 15 = 38 :=
by
   sorry

end function_machine_output_is_38_l189_189440


namespace sum_of_sequence_l189_189836

-- Definitions based on conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 2^(a n) + n
def S (n : ℕ) := (Finset.range n).sum (λ i => b (i + 1))

-- The theorem assertion / problem statement
theorem sum_of_sequence (n : ℕ) : 
  S n = (2 * (4^n - 1)) / 3 + n * (n + 1) / 2 := 
sorry

end sum_of_sequence_l189_189836


namespace amount_spent_on_drink_l189_189672

-- Definitions based on conditions provided
def initialAmount : ℝ := 9
def remainingAmount : ℝ := 6
def additionalSpending : ℝ := 1.25

-- Theorem to prove the amount spent on the drink
theorem amount_spent_on_drink : 
  initialAmount - remainingAmount - additionalSpending = 1.75 := 
by 
  sorry

end amount_spent_on_drink_l189_189672


namespace difference_in_surface_area_l189_189209

-- Defining the initial conditions
def original_length : ℝ := 6
def original_width : ℝ := 5
def original_height : ℝ := 4
def cube_side : ℝ := 2

-- Define the surface area calculation for a rectangular solid
def surface_area_rectangular_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

-- Define the surface area of the cube
def surface_area_cube (a : ℝ) : ℝ :=
  6 * a * a

-- Define the removed face areas when cube is extracted
def exposed_faces_area (a : ℝ) : ℝ :=
  2 * (a * a)

-- Define the problem statement in Lean
theorem difference_in_surface_area :
  surface_area_rectangular_prism original_length original_width original_height
  - (surface_area_rectangular_prism original_length original_width original_height - surface_area_cube cube_side + exposed_faces_area cube_side) = 12 :=
by
  sorry

end difference_in_surface_area_l189_189209


namespace find_a_l189_189186

noncomputable def base25_num : ℕ := 3 * 25^7 + 1 * 25^6 + 4 * 25^5 + 2 * 25^4 + 6 * 25^3 + 5 * 25^2 + 2 * 25^1 + 3 * 25^0

theorem find_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a ≤ 14) : ((base25_num - a) % 12 = 0) → a = 2 := 
sorry

end find_a_l189_189186


namespace largest_p_q_sum_l189_189797

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end largest_p_q_sum_l189_189797


namespace total_meals_per_week_l189_189665

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l189_189665


namespace john_school_year_hours_l189_189115

noncomputable def requiredHoursPerWeek (summerHoursPerWeek : ℕ) (summerWeeks : ℕ) 
                                       (summerEarnings : ℕ) (schoolWeeks : ℕ) 
                                       (schoolEarnings : ℕ) : ℕ :=
    schoolEarnings * summerHoursPerWeek * summerWeeks / (summerEarnings * schoolWeeks)

theorem john_school_year_hours :
  ∀ (summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings : ℕ),
    summerHoursPerWeek = 40 →
    summerWeeks = 10 →
    summerEarnings = 4000 →
    schoolWeeks = 50 →
    schoolEarnings = 4000 →
    requiredHoursPerWeek summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings = 8 :=
by
  intros
  sorry

end john_school_year_hours_l189_189115


namespace range_of_m_min_value_a2_2b2_3c2_l189_189226

theorem range_of_m (x m : ℝ) (h : ∀ x : ℝ, abs (x + 3) + abs (x + m) ≥ 2 * m) : m ≤ 1 :=
sorry

theorem min_value_a2_2b2_3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  ∃ (a b c : ℝ), a = 6/11 ∧ b = 3/11 ∧ c = 2/11 ∧ a^2 + 2 * b^2 + 3 * c^2 = 6/11 :=
sorry

end range_of_m_min_value_a2_2b2_3c2_l189_189226


namespace exists_pow_two_sub_one_divisible_by_odd_l189_189065

theorem exists_pow_two_sub_one_divisible_by_odd {a : ℕ} (h_odd : a % 2 = 1) 
  : ∃ b : ℕ, (2^b - 1) % a = 0 :=
sorry

end exists_pow_two_sub_one_divisible_by_odd_l189_189065


namespace taxi_fare_l189_189788

theorem taxi_fare (fare : ℕ → ℝ) (distance : ℕ) :
  (∀ d, d > 10 → fare d = 20 + (d - 10) * (140 / 70)) →
  fare 80 = 160 →
  fare 100 = 200 :=
by
  intros h_fare h_fare_80
  show fare 100 = 200
  sorry

end taxi_fare_l189_189788


namespace solve_inequality_l189_189647

open Set

theorem solve_inequality (x : ℝ) :
  { x | (x^2 - 9) / (x^2 - 16) > 0 } = (Iio (-4)) ∪ (Ioi 4) :=
by
  sorry

end solve_inequality_l189_189647


namespace chairs_in_fifth_row_l189_189935

theorem chairs_in_fifth_row : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 14 ∧ 
    a 2 = 23 ∧ 
    a 3 = 32 ∧ 
    a 4 = 41 ∧ 
    a 6 = 59 ∧ 
    (∀ n, a (n + 1) = a n + 9) → 
  a 5 = 50 :=
by
  sorry

end chairs_in_fifth_row_l189_189935


namespace find_x_l189_189472

theorem find_x (x : ℤ) (A : Set ℤ) (B : Set ℤ) (hA : A = {1, 4, x}) (hB : B = {1, 2 * x, x ^ 2}) (hinter : A ∩ B = {4, 1}) : x = -2 :=
sorry

end find_x_l189_189472


namespace max_x_add_inv_x_l189_189269

variable (x : ℝ) (y : Fin 2022 → ℝ)

-- Conditions
def sum_condition : Prop := x + (Finset.univ.sum y) = 2024
def reciprocal_sum_condition : Prop := (1/x) + (Finset.univ.sum (λ i => 1 / (y i))) = 2024

-- The statement we need to prove
theorem max_x_add_inv_x (h_sum : sum_condition x y) (h_rec_sum : reciprocal_sum_condition x y) : 
  x + (1/x) ≤ 2 := by
  sorry

end max_x_add_inv_x_l189_189269


namespace perimeter_right_triangle_l189_189246

-- Given conditions
def area : ℝ := 200
def b : ℝ := 20

-- Mathematical problem
theorem perimeter_right_triangle :
  ∀ (x c : ℝ), 
  (1 / 2) * b * x = area →
  c^2 = x^2 + b^2 →
  x + b + c = 40 + 20 * Real.sqrt 2 := 
  by
  sorry

end perimeter_right_triangle_l189_189246


namespace range_of_a_l189_189879

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) + f (1 - 2 * a) < 0

-- Theorem statement
theorem range_of_a (h_decreasing : decreasing_on f (Set.Ioo (-1) 1))
                   (h_odd : odd_function f)
                   (h_condition : condition f a) :
  0 < a ∧ a < 2 / 3 :=
sorry

end range_of_a_l189_189879


namespace complement_intersection_l189_189254

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 3 ≤ x}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A ∩ B) ↔ (0 ≤ x ∧ x < 3) :=
by {
  sorry
}

end complement_intersection_l189_189254


namespace difference_of_digits_is_three_l189_189855

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem difference_of_digits_is_three :
  ∀ n : ℕ, n = 63 → tens_digit n + ones_digit n = 9 → tens_digit n - ones_digit n = 3 :=
by
  intros n h1 h2
  sorry

end difference_of_digits_is_three_l189_189855


namespace max_rabbits_with_traits_l189_189628

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l189_189628


namespace servings_in_one_week_l189_189581

theorem servings_in_one_week (daily_servings : ℕ) (days_in_week : ℕ) (total_servings : ℕ)
  (h1 : daily_servings = 3)
  (h2 : days_in_week = 7)
  (h3 : total_servings = daily_servings * days_in_week) :
  total_servings = 21 := by
  sorry

end servings_in_one_week_l189_189581


namespace find_fx_l189_189820

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2 * x) : ∀ x, f x = x^2 - 1 :=
by
  intro x
  sorry

end find_fx_l189_189820


namespace equation_conditions_l189_189687

theorem equation_conditions (m n : ℤ) (h1 : m ≠ 1) (h2 : n = 1) :
  ∃ x : ℤ, (m - 1) * x = 3 ↔ m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end equation_conditions_l189_189687


namespace sum_of_altitudes_l189_189567

theorem sum_of_altitudes (x y : ℝ) (h : 12 * x + 5 * y = 60) :
  let a := (if y = 0 then x else 0)
  let b := (if x = 0 then y else 0)
  let c := (60 / (Real.sqrt (12^2 + 5^2)))
  a + b + c = 281 / 13 :=
sorry

end sum_of_altitudes_l189_189567


namespace problem_1_problem_2_problem_3_problem_4_l189_189612

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by
  sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by
  sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by
  sorry

-- Problem 4
theorem problem_4 : (-19 + 15 / 16) * 8 = -159 + 1 / 2 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l189_189612


namespace stratified_sampling_grade10_l189_189985

theorem stratified_sampling_grade10
  (total_students : ℕ)
  (grade10_students : ℕ)
  (grade11_students : ℕ)
  (grade12_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 700)
  (h2 : grade10_students = 300)
  (h3 : grade11_students = 200)
  (h4 : grade12_students = 200)
  (h5 : sample_size = 35)
  : (grade10_students * sample_size / total_students) = 15 := 
sorry

end stratified_sampling_grade10_l189_189985


namespace number_exceeds_its_part_by_20_l189_189816

theorem number_exceeds_its_part_by_20 (x : ℝ) (h : x = (3/8) * x + 20) : x = 32 :=
sorry

end number_exceeds_its_part_by_20_l189_189816


namespace correct_operation_is_multiplication_by_3_l189_189151

theorem correct_operation_is_multiplication_by_3
  (x : ℝ)
  (percentage_error : ℝ)
  (correct_result : ℝ := 3 * x)
  (incorrect_result : ℝ := x / 5)
  (error_percentage : ℝ := (correct_result - incorrect_result) / correct_result * 100) :
  percentage_error = 93.33333333333333 → correct_result / x = 3 :=
by
  intro h
  sorry

end correct_operation_is_multiplication_by_3_l189_189151


namespace volume_to_surface_area_ratio_l189_189811

-- Define the shape as described in the problem
structure Shape :=
(center_cube : ℕ)  -- Center cube
(surrounding_cubes : ℕ)  -- Surrounding cubes
(unit_volume : ℕ)  -- Volume of each unit cube
(unit_face_area : ℕ)  -- Surface area of each face of the unit cube

-- Conditions and definitions
def is_special_shape (s : Shape) : Prop :=
  s.center_cube = 1 ∧ s.surrounding_cubes = 7 ∧ s.unit_volume = 1 ∧ s.unit_face_area = 1

-- Theorem statement
theorem volume_to_surface_area_ratio (s : Shape) (h : is_special_shape s) : (s.center_cube + s.surrounding_cubes) * s.unit_volume / (s.surrounding_cubes * 5 * s.unit_face_area) = 8 / 35 :=
by
  sorry

end volume_to_surface_area_ratio_l189_189811


namespace M_inter_P_eq_l189_189655

-- Define the sets M and P
def M : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 4 * x + y = 6 }
def P : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 3 * x + 2 * y = 7 }

-- Prove that the intersection of M and P is {(1, 2)}
theorem M_inter_P_eq : M ∩ P = { (1, 2) } := 
by 
sorry

end M_inter_P_eq_l189_189655


namespace fraction_of_cream_in_cup1_l189_189944

/-
Problem statement:
Sarah places five ounces of coffee into an eight-ounce cup (Cup 1) and five ounces of cream into a second cup (Cup 2).
After pouring half the coffee from Cup 1 to Cup 2, one ounce of cream is added to Cup 2.
After stirring Cup 2 thoroughly, Sarah then pours half the liquid in Cup 2 back into Cup 1.
Prove that the fraction of the liquid in Cup 1 that is now cream is 4/9.
-/

theorem fraction_of_cream_in_cup1
  (initial_coffee_cup1 : ℝ)
  (initial_cream_cup2 : ℝ)
  (half_initial_coffee : ℝ)
  (added_cream : ℝ)
  (total_mixture : ℝ)
  (half_mixture : ℝ)
  (coffee_fraction : ℝ)
  (cream_fraction : ℝ)
  (coffee_transferred_back : ℝ)
  (cream_transferred_back : ℝ)
  (total_coffee_in_cup1 : ℝ)
  (total_cream_in_cup1 : ℝ)
  (total_liquid_in_cup1 : ℝ)
  :
  initial_coffee_cup1 = 5 →
  initial_cream_cup2 = 5 →
  half_initial_coffee = initial_coffee_cup1 / 2 →
  added_cream = 1 →
  total_mixture = initial_cream_cup2 + half_initial_coffee + added_cream →
  half_mixture = total_mixture / 2 →
  coffee_fraction = half_initial_coffee / total_mixture →
  cream_fraction = (total_mixture - half_initial_coffee) / total_mixture →
  coffee_transferred_back = half_mixture * coffee_fraction →
  cream_transferred_back = half_mixture * cream_fraction →
  total_coffee_in_cup1 = initial_coffee_cup1 - half_initial_coffee + coffee_transferred_back →
  total_cream_in_cup1 = cream_transferred_back →
  total_liquid_in_cup1 = total_coffee_in_cup1 + total_cream_in_cup1 →
  total_cream_in_cup1 / total_liquid_in_cup1 = 4 / 9 :=
by {
  sorry
}

end fraction_of_cream_in_cup1_l189_189944


namespace problem_proof_l189_189598

variable (A B C a b c : ℝ)
variable (ABC_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (sides_opposite : a = (b * sin A / sin B) ∧ b = (a * sin B / sin A))
variable (cos_eq : b + b * cos A = a * cos B)

theorem problem_proof :
  (A = 2 * B ∧ (π / 6 < B ∧ B < π / 4) ∧ a^2 = b^2 + b * c) :=
  sorry

end problem_proof_l189_189598


namespace inverse_of_217_mod_397_l189_189429

theorem inverse_of_217_mod_397 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ 217 * a % 397 = 1 :=
sorry

end inverse_of_217_mod_397_l189_189429


namespace tangent_points_sum_constant_l189_189782

theorem tangent_points_sum_constant 
  (a : ℝ) (x1 y1 x2 y2 : ℝ)
  (hC1 : x1^2 = 4 * y1)
  (hC2 : x2^2 = 4 * y2)
  (hT1 : y1 - (-2) = (1/2)*x1*(x1 - a))
  (hT2 : y2 - (-2) = (1/2)*x2*(x2 - a)) :
  x1 * x2 + y1 * y2 = -4 :=
sorry

end tangent_points_sum_constant_l189_189782


namespace find_fraction_eq_l189_189942

theorem find_fraction_eq 
  {x : ℚ} 
  (h : x / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 7 / 15 :=
by
  sorry

end find_fraction_eq_l189_189942


namespace johnny_savings_l189_189111

variable (S : ℤ) -- The savings in September.

theorem johnny_savings :
  (S + 49 + 46 - 58 = 67) → (S = 30) :=
by
  intro h
  sorry

end johnny_savings_l189_189111


namespace cost_of_article_l189_189767

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l189_189767


namespace total_distance_of_journey_l189_189169

-- Definitions corresponding to conditions in the problem
def electric_distance : ℝ := 30 -- The first 30 miles were in electric mode
def gasoline_consumption_rate : ℝ := 0.03 -- Gallons per mile for gasoline mode
def average_mileage : ℝ := 50 -- Miles per gallon for the entire trip

-- Final goal: proving the total distance is 90 miles
theorem total_distance_of_journey (d : ℝ) :
  (d / (gasoline_consumption_rate * (d - electric_distance)) = average_mileage) → d = 90 :=
by
  sorry

end total_distance_of_journey_l189_189169


namespace find_incorrect_value_l189_189469

variable (k b : ℝ)

-- Linear function definition
def linear_function (x : ℝ) : ℝ := k * x + b

-- Given points
theorem find_incorrect_value (h₁ : linear_function k b (-1) = 3)
                             (h₂ : linear_function k b 0 = 2)
                             (h₃ : linear_function k b 1 = 1)
                             (h₄ : linear_function k b 2 = 0)
                             (h₅ : linear_function k b 3 = -2) :
                             (∃ x y, linear_function k b x ≠ y) := by
  sorry

end find_incorrect_value_l189_189469


namespace endpoint_of_parallel_segment_l189_189745

theorem endpoint_of_parallel_segment (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hA : A = (2, 1)) (h_parallel : B.snd = A.snd) (h_length : abs (B.fst - A.fst) = 5) :
  B = (7, 1) ∨ B = (-3, 1) :=
by
  -- Proof goes here
  sorry

end endpoint_of_parallel_segment_l189_189745


namespace tv_horizontal_length_l189_189784

noncomputable def rectangleTvLengthRatio (l h : ℝ) : Prop :=
  l / h = 16 / 9

noncomputable def rectangleTvDiagonal (l h d : ℝ) : Prop :=
  l^2 + h^2 = d^2

theorem tv_horizontal_length
  (h : ℝ)
  (h_positive : h > 0)
  (d : ℝ)
  (h_ratio : rectangleTvLengthRatio l h)
  (h_diagonal : rectangleTvDiagonal l h d)
  (h_diagonal_value : d = 36) :
  l = 56.27 :=
by
  sorry

end tv_horizontal_length_l189_189784


namespace sandy_net_amount_spent_l189_189371

def amount_spent_shorts : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def amount_received_return : ℝ := 7.43

theorem sandy_net_amount_spent :
  amount_spent_shorts + amount_spent_shirt - amount_received_return = 18.70 :=
by
  sorry

end sandy_net_amount_spent_l189_189371


namespace sum_of_factors_30_l189_189304

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end sum_of_factors_30_l189_189304


namespace proof_equiv_l189_189645

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem proof_equiv (x : ℝ) : f (g x) - g (f x) = 6 * x ^ 2 - 12 * x + 9 := by
  sorry

end proof_equiv_l189_189645


namespace value_of_expression_eq_34_l189_189597

theorem value_of_expression_eq_34 : (2 - 6 + 10 - 14 + 18 - 22 + 26 - 30 + 34 - 38 + 42 - 46 + 50 - 54 + 58 - 62 + 66 - 70 + 70) = 34 :=
by
  sorry

end value_of_expression_eq_34_l189_189597


namespace outfit_choices_l189_189553

theorem outfit_choices:
  let shirts := 8
  let pants := 8
  let hats := 8
  -- Each has 8 different colors
  -- No repetition of color within type of clothing
  -- Refuse to wear same color shirt and pants
  (shirts * pants * hats) - (shirts * hats) = 448 := 
sorry

end outfit_choices_l189_189553


namespace calculate_B_l189_189072
open Real

theorem calculate_B 
  (A B : ℝ) 
  (a b : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sin_relation : sin B = (b * sin A) / a) : 
  (B = π / 3 ∨ B = 2 * π / 3) :=
sorry

end calculate_B_l189_189072


namespace boat_speed_in_still_water_eq_16_l189_189564

theorem boat_speed_in_still_water_eq_16 (stream_rate : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) (V_b : ℝ) 
(h1 : stream_rate = 5) (h2 : time_downstream = 6) (h3 : distance_downstream = 126) : 
  V_b = 16 :=
by sorry

end boat_speed_in_still_water_eq_16_l189_189564


namespace average_income_of_other_40_customers_l189_189418

theorem average_income_of_other_40_customers
    (avg_income_50 : ℝ)
    (num_50 : ℕ)
    (avg_income_10 : ℝ)
    (num_10 : ℕ)
    (total_num : ℕ)
    (remaining_num : ℕ)
    (total_income_50 : ℝ)
    (total_income_10 : ℝ)
    (total_income_40 : ℝ)
    (avg_income_40 : ℝ) 
    (hyp_avg_income_50 : avg_income_50 = 45000)
    (hyp_num_50 : num_50 = 50)
    (hyp_avg_income_10 : avg_income_10 = 55000)
    (hyp_num_10 : num_10 = 10)
    (hyp_total_num : total_num = 50)
    (hyp_remaining_num : remaining_num = 40)
    (hyp_total_income_50 : total_income_50 = 2250000)
    (hyp_total_income_10 : total_income_10 = 550000)
    (hyp_total_income_40 : total_income_40 = 1700000)
    (hyp_avg_income_40 : avg_income_40 = total_income_40 / remaining_num) :
  avg_income_40 = 42500 :=
  by
    sorry

end average_income_of_other_40_customers_l189_189418


namespace ratio_comparison_l189_189977

theorem ratio_comparison (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_lt_n : m < n) :
  (m + 3) / (n + 3) > m / n :=
sorry

end ratio_comparison_l189_189977


namespace force_of_water_on_lock_wall_l189_189827

noncomputable def force_on_the_wall (l h γ g : ℝ) : ℝ :=
  γ * g * l * (h^2 / 2)

theorem force_of_water_on_lock_wall :
  force_on_the_wall 20 5 1000 9.81 = 2.45 * 10^6 := by
  sorry

end force_of_water_on_lock_wall_l189_189827


namespace unique_sequence_l189_189489

theorem unique_sequence (n : ℕ) (h : 1 < n)
  (x : Fin (n-1) → ℕ)
  (h_pos : ∀ i, 0 < x i)
  (h_incr : ∀ i j, i < j → x i < x j)
  (h_symm : ∀ i : Fin (n-1), x i + x ⟨n - 2 - i.val, sorry⟩ = 2 * n)
  (h_sum : ∀ i j : Fin (n-1), x i + x j < 2 * n → ∃ k : Fin (n-1), x i + x j = x k) :
  ∀ i : Fin (n-1), x i = 2 * (i + 1) :=
by
  sorry

end unique_sequence_l189_189489


namespace inverse_sum_is_minus_two_l189_189719

variable (f : ℝ → ℝ)
variable (h_injective : Function.Injective f)
variable (h_surjective : Function.Surjective f)
variable (h_eq : ∀ x : ℝ, f (x + 1) + f (-x - 3) = 2)

theorem inverse_sum_is_minus_two (x : ℝ) : f⁻¹ (2009 - x) + f⁻¹ (x - 2007) = -2 := 
  sorry

end inverse_sum_is_minus_two_l189_189719


namespace tricycles_count_l189_189323

theorem tricycles_count (B T : ℕ) (hB : B = 50) (hW : 2 * B + 3 * T = 160) : T = 20 :=
by
  sorry

end tricycles_count_l189_189323


namespace jack_total_yen_l189_189760

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end jack_total_yen_l189_189760


namespace imaginary_part_of_complex_l189_189133

theorem imaginary_part_of_complex : ∀ z : ℂ, z = i^2 * (1 + i) → z.im = -1 :=
by
  intro z
  intro h
  sorry

end imaginary_part_of_complex_l189_189133


namespace inradius_inequality_l189_189333

theorem inradius_inequality
  (r r_A r_B r_C : ℝ) 
  (h_inscribed_circle: r > 0) 
  (h_tangent_circles_A: r_A > 0) 
  (h_tangent_circles_B: r_B > 0) 
  (h_tangent_circles_C: r_C > 0)
  : r ≤ r_A + r_B + r_C :=
  sorry

end inradius_inequality_l189_189333


namespace red_suit_top_card_probability_l189_189329

theorem red_suit_top_card_probability :
  let num_cards := 104
  let num_red_suits := 4
  let cards_per_suit := 26
  let num_red_cards := num_red_suits * cards_per_suit
  let top_card_is_red_probability := num_red_cards / num_cards
  top_card_is_red_probability = 1 := by
  sorry

end red_suit_top_card_probability_l189_189329


namespace same_side_interior_not_complementary_l189_189799

-- Defining the concept of same-side interior angles and complementary angles
def same_side_interior (α β : ℝ) : Prop := 
  α + β = 180 

def complementary (α β : ℝ) : Prop :=
  α + β = 90

-- To state the proposition that should be proven false
theorem same_side_interior_not_complementary (α β : ℝ) (h : same_side_interior α β) : ¬ complementary α β :=
by
  -- We state the observable contradiction here, and since the proof is not required we use sorry
  sorry

end same_side_interior_not_complementary_l189_189799


namespace probability_of_4_vertices_in_plane_l189_189545

-- Definition of the problem conditions
def vertices_of_cube : Nat := 8
def selecting_vertices : Nat := 4

-- Combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 4 vertices from the 8 vertices of a cube
def total_ways : Nat := combination vertices_of_cube selecting_vertices

-- Number of favorable ways that these 4 vertices lie in the same plane
def favorable_ways : Nat := 12

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

-- The ultimate proof problem
theorem probability_of_4_vertices_in_plane :
  probability = 6 / 35 :=
by
  -- Here, the proof steps would go to verify that our setup correctly leads to the given probability.
  sorry

end probability_of_4_vertices_in_plane_l189_189545


namespace polar_to_rectangular_l189_189763

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end polar_to_rectangular_l189_189763


namespace round_balloons_burst_l189_189003

theorem round_balloons_burst :
  let round_balloons := 5 * 20
  let long_balloons := 4 * 30
  let total_balloons := round_balloons + long_balloons
  let balloons_left := 215
  ((total_balloons - balloons_left) = 5) :=
by 
  sorry

end round_balloons_burst_l189_189003


namespace number_picked_by_person_announcing_average_5_l189_189657

-- Definition of given propositions and assumptions
def numbers_picked (b : Fin 6 → ℕ) (average : Fin 6 → ℕ) :=
  (b 4 = 15) ∧
  (average 4 = 8) ∧
  (average 1 = 5) ∧
  (b 2 + b 4 = 16) ∧
  (b 0 + b 2 = 10) ∧
  (b 4 + b 0 = 12)

-- Prove that given the conditions, the number picked by the person announcing an average of 5 is 7
theorem number_picked_by_person_announcing_average_5 (b : Fin 6 → ℕ) (average : Fin 6 → ℕ)
  (h : numbers_picked b average) : b 2 = 7 :=
  sorry

end number_picked_by_person_announcing_average_5_l189_189657


namespace find_larger_number_l189_189385

theorem find_larger_number
  (x y : ℝ)
  (h1 : y = 2 * x + 3)
  (h2 : x + y = 27)
  : y = 19 :=
by
  sorry

end find_larger_number_l189_189385


namespace simplify_fraction_l189_189140

theorem simplify_fraction : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := 
by {
  sorry
}

end simplify_fraction_l189_189140


namespace parabola_equation_l189_189941

theorem parabola_equation
  (axis_of_symmetry : ∀ x y : ℝ, x = 1)
  (focus : ∀ x y : ℝ, x = -1 ∧ y = 0) :
  ∀ y x : ℝ, y^2 = -4*x := 
sorry

end parabola_equation_l189_189941


namespace find_son_age_l189_189904

variable {S F : ℕ}

theorem find_son_age (h1 : F = S + 35) (h2 : F + 2 = 2 * (S + 2)) : S = 33 :=
sorry

end find_son_age_l189_189904


namespace pairing_probability_l189_189136

variable {students : Fin 28} (Alex Jamie : Fin 28)

theorem pairing_probability (h1 : ∀ (i j : Fin 28), i ≠ j) :
  ∃ p : ℚ, p = 1 / 27 ∧ 
  (∃ (A_J_pairs : Finset (Fin 28) × Finset (Fin 28)),
  A_J_pairs.1 = {Alex} ∧ A_J_pairs.2 = {Jamie}) -> p = 1 / 27
:= sorry

end pairing_probability_l189_189136


namespace tangent_line_through_P_l189_189375

theorem tangent_line_through_P (x y : ℝ) :
  (∃ l : ℝ, l = 3*x - 4*y + 5) ∨ (x = 1) :=
by
  sorry

end tangent_line_through_P_l189_189375


namespace rectangle_similarity_l189_189683

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def is_congruent (A B : Rectangle) : Prop :=
  A.length = B.length ∧ A.width = B.width

def is_similar (A B : Rectangle) : Prop :=
  A.length / A.width = B.length / B.width

theorem rectangle_similarity (A B : Rectangle)
  (h1 : ∀ P, is_congruent P A → ∃ Q, is_similar Q B)
  : ∀ P, is_congruent P B → ∃ Q, is_similar Q A :=
by sorry

end rectangle_similarity_l189_189683


namespace each_niece_gets_13_l189_189141

-- Define the conditions
def total_sandwiches : ℕ := 143
def number_of_nieces : ℕ := 11

-- Prove that each niece can get 13 ice cream sandwiches
theorem each_niece_gets_13 : total_sandwiches / number_of_nieces = 13 :=
by
  -- Proof omitted
  sorry

end each_niece_gets_13_l189_189141


namespace xy_difference_l189_189762

noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

theorem xy_difference : x^2 * y - x * y^2 = 4 := by
  sorry

end xy_difference_l189_189762


namespace simplify_fraction_mul_l189_189420

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : 405 = 27 * a) (h2 : 1215 = 27 * b) (h3 : a / d = 1) (h4 : b / d = 3) : (a / d) * (27 : ℕ) = 9 :=
by
  sorry

end simplify_fraction_mul_l189_189420


namespace number_of_dolls_of_jane_l189_189791

-- Given conditions
def total_dolls (J D : ℕ) := J + D = 32
def jill_has_more (J D : ℕ) := D = J + 6

-- Statement to prove
theorem number_of_dolls_of_jane (J D : ℕ) (h1 : total_dolls J D) (h2 : jill_has_more J D) : J = 13 :=
by
  sorry

end number_of_dolls_of_jane_l189_189791


namespace subtraction_correct_l189_189731

theorem subtraction_correct : 900000009000 - 123456789123 = 776543220777 :=
by
  -- Placeholder proof to ensure it compiles
  sorry

end subtraction_correct_l189_189731


namespace Alice_min_speed_l189_189771

theorem Alice_min_speed (d : ℝ) (v_bob : ℝ) (delta_t : ℝ) (v_alice : ℝ) :
  d = 180 ∧ v_bob = 40 ∧ delta_t = 0.5 ∧ 0 < v_alice ∧ v_alice * (d / v_bob - delta_t) ≥ d →
  v_alice > 45 :=
by
  sorry

end Alice_min_speed_l189_189771


namespace joe_avg_speed_l189_189317

noncomputable def total_distance : ℝ :=
  420 + 250 + 120 + 65

noncomputable def total_time : ℝ :=
  (420 / 60) + (250 / 50) + (120 / 40) + (65 / 70)

noncomputable def avg_speed : ℝ :=
  total_distance / total_time

theorem joe_avg_speed : avg_speed = 53.67 := by
  sorry

end joe_avg_speed_l189_189317


namespace cos_neg_79_pi_over_6_l189_189324

theorem cos_neg_79_pi_over_6 : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_neg_79_pi_over_6_l189_189324


namespace right_triangle_legs_l189_189674

theorem right_triangle_legs (a b c : ℝ) 
  (h : ℝ) 
  (h_h : h = 12) 
  (h_perimeter : a + b + c = 60) 
  (h1 : a^2 + b^2 = c^2) 
  (h_altitude : h = a * b / c) :
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l189_189674


namespace derivative_at_zero_l189_189309

def f (x : ℝ) : ℝ := x^3

theorem derivative_at_zero : deriv f 0 = 0 :=
by
  sorry

end derivative_at_zero_l189_189309


namespace bowling_average_decrease_l189_189120

theorem bowling_average_decrease
    (initial_average : ℝ) (wickets_last_match : ℝ) (runs_last_match : ℝ)
    (average_decrease : ℝ) (W : ℝ)
    (H_initial : initial_average = 12.4)
    (H_wickets_last_match : wickets_last_match = 6)
    (H_runs_last_match : runs_last_match = 26)
    (H_average_decrease : average_decrease = 0.4) :
    W = 115 :=
by
  sorry

end bowling_average_decrease_l189_189120


namespace find_m_l189_189593

theorem find_m (a0 a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ)
  (h1 : (1 + m) * x ^ 6 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5 + a6 * x ^ 6)
  (h2 : a1 - a2 + a3 - a4 + a5 - a6 = -63)
  (h3 : a0 = 1) :
  m = 3 ∨ m = -1 :=
by
  sorry

end find_m_l189_189593


namespace frog_climb_time_l189_189359

-- Definitions related to the problem
def well_depth : ℕ := 12
def climb_per_cycle : ℕ := 3
def slip_per_cycle : ℕ := 1
def effective_climb_per_cycle : ℕ := climb_per_cycle - slip_per_cycle

-- Time taken for each activity
def time_to_climb : ℕ := 10 -- given as t
def time_to_slip : ℕ := time_to_climb / 3
def total_time_per_cycle : ℕ := time_to_climb + time_to_slip

-- Condition specifying the observed frog position at a certain time
def observed_time : ℕ := 17 -- minutes since 8:00
def observed_position : ℕ := 9 -- meters climbed since it's 3 meters from the top of the well (well_depth - 3)

-- The main theorem stating the total time taken to climb to the top of the well
theorem frog_climb_time : 
  ∃ (k : ℕ), k * effective_climb_per_cycle + climb_per_cycle = well_depth ∧ k * total_time_per_cycle + time_to_climb = 22 := 
sorry

end frog_climb_time_l189_189359


namespace number_of_large_balls_l189_189161

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l189_189161


namespace angle_ABC_is_83_l189_189396

-- Definitions of angles and the quadrilateral
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angleBAC angleCAD angleACD : ℝ)
variables (AB AD AC : ℝ)

-- Conditions as hypotheses
axiom h1 : angleBAC = 60
axiom h2 : angleCAD = 60
axiom h3 : AB + AD = AC
axiom h4 : angleACD = 23

-- The theorem to prove
theorem angle_ABC_is_83 (h1 : angleBAC = 60) (h2 : angleCAD = 60) (h3 : AB + AD = AC) (h4 : angleACD = 23) : 
  ∃ angleABC : ℝ, angleABC = 83 :=
sorry

end angle_ABC_is_83_l189_189396


namespace fraction_of_students_getting_A_l189_189264

theorem fraction_of_students_getting_A
    (frac_B : ℚ := 1/2)
    (frac_C : ℚ := 1/8)
    (frac_D : ℚ := 1/12)
    (frac_F : ℚ := 1/24)
    (passing_grade_frac: ℚ := 0.875) :
    (1 - (frac_B + frac_C + frac_D + frac_F) = 1/8) :=
by
  sorry

end fraction_of_students_getting_A_l189_189264


namespace percentage_lower_grades_have_cars_l189_189542

-- Definitions for the conditions
def n_seniors : ℕ := 300
def p_car : ℚ := 0.50
def n_lower : ℕ := 900
def p_total : ℚ := 0.20

-- Definition for the number of students who have cars in the lower grades
def n_cars_lower : ℚ := 
  let total_students := n_seniors + n_lower
  let total_cars := p_total * total_students
  total_cars - (p_car * n_seniors)

-- Prove the percentage of freshmen, sophomores, and juniors who have cars
theorem percentage_lower_grades_have_cars : 
  (n_cars_lower / n_lower) * 100 = 10 := 
by sorry

end percentage_lower_grades_have_cars_l189_189542


namespace alina_sent_fewer_messages_l189_189659

-- Definitions based on conditions
def messages_lucia_day1 : Nat := 120
def messages_lucia_day2 : Nat := 1 / 3 * messages_lucia_day1
def messages_lucia_day3 : Nat := messages_lucia_day1
def messages_total : Nat := 680

-- Def statement for Alina's messages on the first day, which we need to find as 100
def messages_alina_day1 : Nat := 100

-- Condition checks
def condition_alina_day2 : Prop := 2 * messages_alina_day1 = 2 * 100
def condition_alina_day3 : Prop := messages_alina_day1 = 100
def condition_total_messages : Prop := 
  messages_alina_day1 + messages_lucia_day1 +
  2 * messages_alina_day1 + messages_lucia_day2 +
  messages_alina_day1 + messages_lucia_day1 = messages_total

-- Theorem statement
theorem alina_sent_fewer_messages :
  messages_lucia_day1 - messages_alina_day1 = 20 :=
by
  -- Ensure the conditions hold
  have h1 : messages_alina_day1 = 100 := by sorry
  have h2 : condition_alina_day2 := by sorry
  have h3 : condition_alina_day3 := by sorry
  have h4 : condition_total_messages := by sorry
  -- Prove the theorem
  sorry

end alina_sent_fewer_messages_l189_189659


namespace sum_of_numbers_l189_189013

theorem sum_of_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000)
(h_eq : 100 * x + y = 7 * x * y) : x + y = 18 :=
sorry

end sum_of_numbers_l189_189013


namespace total_wet_surface_area_is_correct_l189_189062

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct_l189_189062


namespace mean_home_runs_per_game_l189_189873

variable (home_runs : Nat) (games_played : Nat)

def total_home_runs : Nat := 
  (5 * 4) + (6 * 5) + (4 * 7) + (3 * 9) + (2 * 11)

def total_games_played : Nat :=
  (5 * 5) + (6 * 6) + (4 * 8) + (3 * 10) + (2 * 12)

theorem mean_home_runs_per_game :
  (total_home_runs : ℚ) / total_games_played = 127 / 147 :=
  by 
    sorry

end mean_home_runs_per_game_l189_189873


namespace option_D_is_greater_than_reciprocal_l189_189971

theorem option_D_is_greater_than_reciprocal:
  ∀ (x : ℚ), (x = 2) → x > 1/x := by
  intro x
  intro hx
  rw [hx]
  norm_num

end option_D_is_greater_than_reciprocal_l189_189971


namespace find_real_number_a_l189_189668

theorem find_real_number_a (a : ℝ) (h : (a^2 - 3*a + 2 = 0)) (h' : (a - 2) ≠ 0) : a = 1 :=
sorry

end find_real_number_a_l189_189668


namespace round_robin_highest_score_l189_189353

theorem round_robin_highest_score
  (n : ℕ) (hn : n = 16)
  (teams : Fin n → ℕ)
  (games_played : Fin n → Fin n → ℕ)
  (draws : Fin n → Fin n → ℕ)
  (win_points : ℕ := 2)
  (draw_points : ℕ := 1)
  (total_games : ℕ := (n * (n - 1)) / 2) :
  ¬ (∃ max_score : ℕ, ∀ i : Fin n, teams i ≤ max_score ∧ max_score < 16) :=
by sorry

end round_robin_highest_score_l189_189353


namespace finite_solutions_l189_189063

variable (a b : ℕ) (h1 : a ≠ b)

theorem finite_solutions (a b : ℕ) (h1 : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
  (x * y + z * w = a) ∧ (x * z + y * w = b) →
  (x, y, z, w) ∈ S :=
sorry

end finite_solutions_l189_189063


namespace stuffed_animal_tickets_correct_l189_189726

-- Define the total tickets spent
def total_tickets : ℕ := 14

-- Define the tickets spent on the hat
def hat_tickets : ℕ := 2

-- Define the tickets spent on the yoyo
def yoyo_tickets : ℕ := 2

-- Define the tickets spent on the stuffed animal
def stuffed_animal_tickets : ℕ := total_tickets - (hat_tickets + yoyo_tickets)

-- The theorem we want to prove.
theorem stuffed_animal_tickets_correct :
  stuffed_animal_tickets = 10 :=
by
  sorry

end stuffed_animal_tickets_correct_l189_189726


namespace melissa_total_time_l189_189039

-- Definitions based on the conditions in the problem
def time_replace_buckle : Nat := 5
def time_even_heel : Nat := 10
def time_fix_straps : Nat := 7
def time_reattach_soles : Nat := 12
def pairs_of_shoes : Nat := 8

-- Translation of the mathematically equivalent proof problem
theorem melissa_total_time : 
  (time_replace_buckle + time_even_heel + time_fix_straps + time_reattach_soles) * 16 = 544 :=
by
  sorry

end melissa_total_time_l189_189039


namespace original_ratio_l189_189225

theorem original_ratio (F J : ℚ) (hJ : J = 180) (h_ratio : (F + 45) / J = 3 / 2) : F / J = 5 / 4 :=
by
  sorry

end original_ratio_l189_189225


namespace geometric_sequence_general_term_and_sum_l189_189239

theorem geometric_sequence_general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₁ : ∀ n, a n = 2 ^ n)
  (h₂ : ∀ n, b n = 2 * n - 1)
  : (∀ n, T n = 6 + (2 * n - 3) * 2 ^ (n + 1)) :=
by {
  sorry
}

end geometric_sequence_general_term_and_sum_l189_189239


namespace right_triangle_third_side_l189_189382

theorem right_triangle_third_side (a b : ℕ) (c : ℝ) (h₁: a = 3) (h₂: b = 4) (h₃: ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2)) ∨ (c^2 + b^2 = a^2)):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end right_triangle_third_side_l189_189382


namespace cards_received_at_home_l189_189258

-- Definitions based on the conditions
def initial_cards := 403
def total_cards := 690

-- The theorem to prove the number of cards received at home
theorem cards_received_at_home : total_cards - initial_cards = 287 :=
by
  -- Proof goes here, but we use sorry as a placeholder.
  sorry

end cards_received_at_home_l189_189258


namespace lcm_18_20_l189_189200

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l189_189200


namespace part_I_part_II_l189_189380

section problem_1

def f (x : ℝ) (a : ℝ) := |x - 3| - |x + a|

theorem part_I (x : ℝ) (hx : f x 2 < 1) : 0 < x :=
by
  sorry

theorem part_II (a : ℝ) (h : ∀ (x : ℝ), f x a ≤ 2 * a) : 3 ≤ a :=
by
  sorry

end problem_1

end part_I_part_II_l189_189380


namespace ab_greater_than_a_plus_b_l189_189313

theorem ab_greater_than_a_plus_b (a b : ℝ) (h₁ : a ≥ 2) (h₂ : b > 2) : a * b > a + b :=
  sorry

end ab_greater_than_a_plus_b_l189_189313


namespace required_moles_H2SO4_l189_189428

-- Definitions for the problem
def moles_NaCl := 2
def moles_H2SO4_needed := 2
def moles_HCl_produced := 2
def moles_NaHSO4_produced := 2

-- Condition representing stoichiometry of the reaction
axiom reaction_stoichiometry : ∀ (moles_NaCl moles_H2SO4 moles_HCl moles_NaHSO4 : ℕ), 
  moles_NaCl = moles_HCl ∧ moles_HCl = moles_NaHSO4 → moles_NaCl = moles_H2SO4

-- Proof statement we want to establish
theorem required_moles_H2SO4 : 
  ∃ (moles_H2SO4 : ℕ), moles_H2SO4 = 2 ∧ ∀ (moles_NaCl : ℕ), moles_NaCl = 2 → moles_H2SO4_needed = 2 := by
  sorry

end required_moles_H2SO4_l189_189428


namespace who_had_second_value_card_in_first_game_l189_189202

variable (A B C : ℕ)
variable (x y z : ℕ)
variable (points_A points_B points_C : ℕ)

-- Provided conditions
variable (h1 : x < y ∧ y < z)
variable (h2 : points_A = 20)
variable (h3 : points_B = 10)
variable (h4 : points_C = 9)
variable (number_of_games : ℕ)
variable (h5 : number_of_games = 3)
variable (h6 : A + B + C = 39)  -- This corresponds to points_A + points_B + points_C = 39.
variable (h7 : ∃ x y z, x + y + z = 13 ∧ x < y ∧ y < z)
variable (h8 : B = z)

-- Question/Proof to establish
theorem who_had_second_value_card_in_first_game :
  ∃ p : ℕ, p = C :=
sorry

end who_had_second_value_card_in_first_game_l189_189202


namespace base_four_30121_eq_793_l189_189586

-- Definition to convert a base-four (radix 4) number 30121_4 to its base-ten equivalent
def base_four_to_base_ten (d4 d3 d2 d1 d0 : ℕ) : ℕ :=
  d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

theorem base_four_30121_eq_793 : base_four_to_base_ten 3 0 1 2 1 = 793 := 
by
  sorry

end base_four_30121_eq_793_l189_189586


namespace line_perpendicular_value_of_a_l189_189247

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l189_189247


namespace find_expression_for_f_l189_189131

theorem find_expression_for_f (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 6 * x) :
  ∀ x, f x = x^2 + 8 * x + 7 :=
by
  sorry

end find_expression_for_f_l189_189131


namespace trajectory_of_centroid_l189_189604

def foci (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (0, 1) ∧ F2 = (0, -1)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 3) + (P.2^2 / 4) = 1

def centroid_eq (G : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, on_ellipse P ∧ 
  foci (0, 1) (0, -1) ∧ 
  G = (P.1 / 3, (1 + -1 + P.2) / 3)

theorem trajectory_of_centroid :
  ∀ G : ℝ × ℝ, (centroid_eq G → 3 * G.1^2 + (9 * G.2^2) / 4 = 1 ∧ G.1 ≠ 0) :=
by 
  intros G h
  sorry

end trajectory_of_centroid_l189_189604


namespace dogwood_trees_after_work_l189_189551

theorem dogwood_trees_after_work 
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_part3 : ℝ)
  (trees_cut : ℝ) (trees_planted : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0) (h3 : trees_part3 = 6.0)
  (h_cut : trees_cut = 7.0) (h_planted : trees_planted = 3.0) :
  trees_part1 + trees_part2 + trees_part3 - trees_cut + trees_planted = 11.0 :=
by
  sorry

end dogwood_trees_after_work_l189_189551


namespace math_problem_l189_189004

noncomputable def canA_red_balls := 3
noncomputable def canA_black_balls := 4
noncomputable def canB_red_balls := 2
noncomputable def canB_black_balls := 3

noncomputable def prob_event_A := canA_red_balls / (canA_red_balls + canA_black_balls) -- P(A)
noncomputable def prob_event_B := 
  (canA_red_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls + 1) / (6) +
  (canA_black_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls) / (6) -- P(B)

theorem math_problem : 
  (prob_event_A = 3 / 7) ∧ 
  (prob_event_B = 17 / 42) ∧
  (¬ (prob_event_A * prob_event_B = (3 / 7) * (17 / 42))) ∧
  ((prob_event_A * (canB_red_balls + 1) / 6) / prob_event_A = 1 / 2) := by
  repeat { sorry }

end math_problem_l189_189004


namespace Jina_mascots_total_l189_189244

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l189_189244


namespace geometric_sum_l189_189673

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 6) (h3 : a 3 = -18) :
  a 1 + a 2 + a 3 + a 4 = 40 :=
sorry

end geometric_sum_l189_189673


namespace expression_value_l189_189009

theorem expression_value (y : ℤ) (h : y = 5) : (y^2 - y - 12) / (y - 4) = 8 :=
by
  rw[h]
  sorry

end expression_value_l189_189009


namespace solve_inequality_system_l189_189273

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l189_189273


namespace probability_to_form_computers_l189_189821

def letters_in_campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def letters_in_threads : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def letters_in_glow : Finset Char := {'G', 'L', 'O', 'W'}
def letters_in_computers : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

noncomputable def probability_campus : ℚ := 1 / Nat.choose 6 3
noncomputable def probability_threads : ℚ := 1 / Nat.choose 7 5
noncomputable def probability_glow : ℚ := 1 / (Nat.choose 4 2 / Nat.choose 3 1)

noncomputable def overall_probability : ℚ :=
  probability_campus * probability_threads * probability_glow

theorem probability_to_form_computers :
  overall_probability = 1 / 840 := by
  sorry

end probability_to_form_computers_l189_189821


namespace mean_of_remaining_four_numbers_l189_189736

theorem mean_of_remaining_four_numbers 
  (a b c d max_num : ℝ) 
  (h1 : max_num = 105) 
  (h2 : (a + b + c + d + max_num) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.75 :=
by
  sorry

end mean_of_remaining_four_numbers_l189_189736


namespace usual_time_to_reach_school_l189_189646

variable (R T : ℝ)
variable (h : T * R = (T - 4) * (7/6 * R))

theorem usual_time_to_reach_school (h : T * R = (T - 4) * (7/6 * R)) : T = 28 := by
  sorry

end usual_time_to_reach_school_l189_189646


namespace larger_of_two_numbers_l189_189507

theorem larger_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 8) : max x y = 29 :=
by
  sorry

end larger_of_two_numbers_l189_189507


namespace cone_lateral_surface_area_l189_189510

theorem cone_lateral_surface_area (r V: ℝ) (h : ℝ) (l : ℝ) (L: ℝ):
  r = 3 →
  V = 12 * Real.pi →
  V = (1 / 3) * Real.pi * r^2 * h →
  l = Real.sqrt (r^2 + h^2) →
  L = Real.pi * r * l →
  L = 15 * Real.pi :=
by
  intros hr hv hV hl hL
  rw [hr, hv] at hV
  sorry

end cone_lateral_surface_area_l189_189510


namespace g_g_g_g_3_l189_189860

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l189_189860


namespace halfway_between_one_eighth_and_one_third_l189_189434

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 + 1 / 3) / 2 = 11 / 48 :=
by
  -- Skipping the proof here
  sorry

end halfway_between_one_eighth_and_one_third_l189_189434


namespace probability_odd_number_die_l189_189758

theorem probability_odd_number_die :
  let total_outcomes := 6
  let favorable_outcomes := 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_number_die_l189_189758


namespace point_P_in_Quadrant_II_l189_189080

noncomputable def α : ℝ := (5 * Real.pi) / 8

theorem point_P_in_Quadrant_II : (Real.sin α > 0) ∧ (Real.tan α < 0) := sorry

end point_P_in_Quadrant_II_l189_189080


namespace sarah_toy_cars_l189_189908

theorem sarah_toy_cars (initial_money toy_car_cost scarf_cost beanie_cost remaining_money: ℕ) 
  (h_initial: initial_money = 53) 
  (h_toy_car_cost: toy_car_cost = 11) 
  (h_scarf_cost: scarf_cost = 10) 
  (h_beanie_cost: beanie_cost = 14) 
  (h_remaining: remaining_money = 7) : 
  (initial_money - remaining_money - scarf_cost - beanie_cost) / toy_car_cost = 2 := 
by 
  sorry

end sarah_toy_cars_l189_189908


namespace max_value_of_sum_l189_189958

theorem max_value_of_sum 
  (a b c : ℝ) 
  (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  a + b + c ≤ Real.sqrt 11 := 
by 
  sorry

end max_value_of_sum_l189_189958


namespace function_no_extrema_k_equals_one_l189_189464

theorem function_no_extrema_k_equals_one (k : ℝ) (h : ∀ x : ℝ, ¬ ∃ m, (k - 1) * x^2 - 4 * x + 5 - k = m) : k = 1 :=
sorry

end function_no_extrema_k_equals_one_l189_189464


namespace train_length_l189_189138

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℕ) 
  (h1 : speed_kmh = 180)
  (h2 : time_s = 18)
  (h3 : 1 = 1000 / 3600) :
  length_m = (speed_kmh * 1000 / 3600) * time_s :=
by
  sorry

end train_length_l189_189138


namespace smallest_resolvable_debt_l189_189733

theorem smallest_resolvable_debt (p g : ℤ) : 
  ∃ p g : ℤ, (500 * p + 350 * g = 50) ∧ ∀ D > 0, (∃ p g : ℤ, 500 * p + 350 * g = D) → 50 ≤ D :=
by {
  sorry
}

end smallest_resolvable_debt_l189_189733


namespace find_common_ratio_and_difference_l189_189474

theorem find_common_ratio_and_difference (q d : ℤ) 
  (h1 : q^3 = 1 + 7 * d) 
  (h2 : 1 + q + q^2 + q^3 = 1 + 7 * d + 21) : 
  (q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18) :=
by
  sorry

end find_common_ratio_and_difference_l189_189474


namespace kimberly_store_visits_l189_189583

def peanuts_per_visit : ℕ := 7
def total_peanuts : ℕ := 21

def visits : ℕ := total_peanuts / peanuts_per_visit

theorem kimberly_store_visits : visits = 3 :=
by
  sorry

end kimberly_store_visits_l189_189583


namespace factorial_expression_simplification_l189_189522

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l189_189522


namespace determine_a_l189_189250

lemma even_exponent (a : ℤ) : (a^2 - 4*a) % 2 = 0 :=
sorry

lemma decreasing_function (a : ℤ) : a^2 - 4*a < 0 :=
sorry

theorem determine_a (a : ℤ) (h1 : (a^2 - 4*a) % 2 = 0) (h2 : a^2 - 4*a < 0) : a = 2 :=
sorry

end determine_a_l189_189250


namespace quadratic_has_exactly_one_solution_l189_189379

theorem quadratic_has_exactly_one_solution (k : ℚ) :
  (3 * x^2 - 8 * x + k = 0) → ((-8)^2 - 4 * 3 * k = 0) → k = 16 / 3 :=
by
  sorry

end quadratic_has_exactly_one_solution_l189_189379


namespace sid_fraction_left_l189_189922

noncomputable def fraction_left (original total_spent remaining additional : ℝ) : ℝ :=
  (remaining - additional) / original

theorem sid_fraction_left 
  (original : ℝ := 48) 
  (spent_computer : ℝ := 12) 
  (spent_snacks : ℝ := 8) 
  (remaining : ℝ := 28) 
  (additional : ℝ := 4) :
  fraction_left original (spent_computer + spent_snacks) remaining additional = 1 / 2 :=
by
  sorry

end sid_fraction_left_l189_189922


namespace find_list_price_l189_189988

theorem find_list_price (P : ℝ) (h1 : 0.873 * P = 61.11) : P = 61.11 / 0.873 :=
by
  sorry

end find_list_price_l189_189988


namespace range_of_a_l189_189255

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 
  a * x + 1 - 4 * a 
else 
  x ^ 2 - 3 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → 
  a ∈ (Set.Ioi (2/3) ∪ Set.Iic 0) :=
sorry

end range_of_a_l189_189255


namespace time_2517_hours_from_now_l189_189949

-- Define the initial time and the function to calculate time after certain hours on a 12-hour clock
def current_time := 3
def hours := 2517

noncomputable def final_time_mod_12 (current_time : ℕ) (hours : ℕ) : ℕ :=
  (current_time + (hours % 12)) % 12

theorem time_2517_hours_from_now :
  final_time_mod_12 current_time hours = 12 :=
by
  sorry

end time_2517_hours_from_now_l189_189949


namespace ratio_problem_l189_189253

/-
  Given the ratio A : B : C = 3 : 2 : 5, we need to prove that 
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19.
-/

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 :=
by sorry

end ratio_problem_l189_189253


namespace sqrt_15_minus_1_range_l189_189641

theorem sqrt_15_minus_1_range (h : 9 < 15 ∧ 15 < 16) : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := 
  sorry

end sqrt_15_minus_1_range_l189_189641


namespace domain_f_l189_189185

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / ((x^2) - 4)

theorem domain_f : {x : ℝ | 0 ≤ x ∧ x ≠ 2} = {x | 0 ≤ x ∧ x < 2} ∪ {x | x > 2} :=
by sorry

end domain_f_l189_189185


namespace james_muffins_l189_189928

theorem james_muffins (arthur_muffins : ℕ) (times : ℕ) (james_muffins : ℕ) 
  (h1 : arthur_muffins = 115) 
  (h2 : times = 12) 
  (h3 : james_muffins = arthur_muffins * times) : 
  james_muffins = 1380 := 
by 
  sorry

end james_muffins_l189_189928


namespace certain_number_l189_189950

theorem certain_number (x y : ℕ) (h₁ : x = 14) (h₂ : 2^x - 2^(x - 2) = 3 * 2^y) : y = 12 :=
  by
  sorry

end certain_number_l189_189950


namespace sum_ge_3_implies_one_ge_2_l189_189162

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end sum_ge_3_implies_one_ge_2_l189_189162


namespace sqrt_sqrt4_of_decimal_l189_189098

theorem sqrt_sqrt4_of_decimal (h : 0.000625 = 625 / (10 ^ 6)) :
  Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 625) / 1000)) = 0.4 :=
by
  sorry

end sqrt_sqrt4_of_decimal_l189_189098


namespace total_cost_of_office_supplies_l189_189918

-- Define the conditions
def cost_of_pencil : ℝ := 0.5
def cost_of_folder : ℝ := 0.9
def count_of_pencils : ℕ := 24
def count_of_folders : ℕ := 20

-- Define the theorem to prove
theorem total_cost_of_office_supplies
  (cop : ℝ := cost_of_pencil)
  (cof : ℝ := cost_of_folder)
  (ncp : ℕ := count_of_pencils)
  (ncg : ℕ := count_of_folders) :
  cop * ncp + cof * ncg = 30 :=
sorry

end total_cost_of_office_supplies_l189_189918


namespace pencils_given_l189_189049

theorem pencils_given (pencils_original pencils_left pencils_given : ℕ)
  (h1 : pencils_original = 142)
  (h2 : pencils_left = 111)
  (h3 : pencils_given = pencils_original - pencils_left) :
  pencils_given = 31 :=
by
  sorry

end pencils_given_l189_189049


namespace any_positive_integer_can_be_expressed_l189_189527

theorem any_positive_integer_can_be_expressed 
  (N : ℕ) (hN : 0 < N) : 
  ∃ (p q u v : ℤ), N = p * q + u * v ∧ (u - v = 2 * (p - q)) := 
sorry

end any_positive_integer_can_be_expressed_l189_189527


namespace find_unique_n_k_l189_189975

theorem find_unique_n_k (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
    (n+1)^n = 2 * n^k + 3 * n + 1 ↔ (n = 3 ∧ k = 3) := by
  sorry

end find_unique_n_k_l189_189975


namespace mobius_trip_proof_l189_189893

noncomputable def mobius_trip_time : ℝ :=
  let speed_no_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let distance_total := 257
  let distance_typical := 120
  let distance_light := distance_total - distance_typical
  let time_typical := distance_typical / speed_typical_load
  let time_light := distance_light / speed_light_load
  let time_return := distance_total / speed_no_load
  let rest_first := (20 + 25 + 35) / 60.0
  let rest_second := (45 + 30) / 60.0
  time_typical + time_light + time_return + rest_first + rest_second

theorem mobius_trip_proof : mobius_trip_time = 44.6783 :=
  by sorry

end mobius_trip_proof_l189_189893


namespace sin_alpha_plus_2beta_l189_189437

theorem sin_alpha_plus_2beta
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosalpha_plus_beta : Real.cos (α + β) = -5 / 13)
  (h sinbeta : Real.sin β = 3 / 5) :
  Real.sin (α + 2 * β) = 33 / 65 :=
  sorry

end sin_alpha_plus_2beta_l189_189437


namespace ticket_sales_amount_theater_collected_50_dollars_l189_189081

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l189_189081


namespace fraction_burritos_given_away_l189_189902

noncomputable def total_burritos_bought : Nat := 3 * 20
noncomputable def burritos_eaten : Nat := 3 * 10
noncomputable def burritos_left : Nat := 10
noncomputable def burritos_before_eating : Nat := burritos_eaten + burritos_left
noncomputable def burritos_given_away : Nat := total_burritos_bought - burritos_before_eating

theorem fraction_burritos_given_away : (burritos_given_away : ℚ) / total_burritos_bought = 1 / 3 := by
  sorry

end fraction_burritos_given_away_l189_189902


namespace curve_left_of_line_l189_189288

theorem curve_left_of_line (x y : ℝ) : x^3 + 2*y^2 = 8 → x ≤ 2 := 
sorry

end curve_left_of_line_l189_189288


namespace henry_earnings_correct_l189_189476

-- Define constants for the amounts earned per task
def earn_per_lawn : Nat := 5
def earn_per_leaves : Nat := 10
def earn_per_driveway : Nat := 15

-- Define constants for the number of tasks he actually managed to do
def lawns_mowed : Nat := 5
def leaves_raked : Nat := 3
def driveways_shoveled : Nat := 2

-- Define the expected total earnings calculation
def expected_earnings : Nat :=
  (lawns_mowed * earn_per_lawn) +
  (leaves_raked * earn_per_leaves) +
  (driveways_shoveled * earn_per_driveway)

-- State the theorem that the total earnings are 85 dollars.
theorem henry_earnings_correct : expected_earnings = 85 :=
by
  sorry

end henry_earnings_correct_l189_189476


namespace solve_quadratic_l189_189650

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, 
  (-6) * x1^2 + 11 * x1 - 3 = 0 ∧ (-6) * x2^2 + 11 * x2 - 3 = 0 ∧ x1 = 1.5 ∧ x2 = 1 / 3 :=
by
  sorry

end solve_quadratic_l189_189650
