import Mathlib

namespace value_of_m_over_q_l762_76211

-- Definitions for the given conditions
variables (n m p q : ℤ) 

-- Main theorem statement
theorem value_of_m_over_q (h1 : m = 10 * n) (h2 : p = 2 * n) (h3 : p = q / 5) :
  m / q = 1 :=
sorry

end value_of_m_over_q_l762_76211


namespace probability_of_winning_quiz_l762_76297

theorem probability_of_winning_quiz :
  let n := 4 -- number of questions
  let choices := 3 -- number of choices per question
  let probability_correct := 1 / choices -- probability of answering correctly
  let probability_incorrect := 1 - probability_correct -- probability of answering incorrectly
  let probability_all_correct := probability_correct^n -- probability of getting all questions correct
  let probability_exactly_three_correct := 4 * probability_correct^3 * probability_incorrect -- probability of getting exactly 3 questions correct
  probability_all_correct + probability_exactly_three_correct = 1 / 9 :=
by
  sorry

end probability_of_winning_quiz_l762_76297


namespace fraction_of_sum_l762_76229

theorem fraction_of_sum (n S : ℕ) 
  (h1 : S = (n-1) * ((n:ℚ) / 3))
  (h2 : n > 0) : 
  (n:ℚ) / (S + n) = 3 / (n + 2) := 
by 
  sorry

end fraction_of_sum_l762_76229


namespace excircle_opposite_side_b_l762_76284

-- Definition of the terms and assumptions
variables {a b c : ℝ} -- sides of the triangle
variables {r r1 : ℝ}  -- radii of the circles

-- Given conditions
def touches_side_c_and_extensions_of_a_b (r : ℝ) (a b c : ℝ) : Prop :=
  r = (a + b + c) / 2

-- The goal to be proved
theorem excircle_opposite_side_b (a b c : ℝ) (r1 : ℝ) (h1 : touches_side_c_and_extensions_of_a_b r a b c) :
  r1 = (a + c - b) / 2 := 
by
  sorry

end excircle_opposite_side_b_l762_76284


namespace sufficient_condition_m_ge_4_range_of_x_for_m5_l762_76281

variable (x m : ℝ)

-- Problem (1)
theorem sufficient_condition_m_ge_4 (h : m > 0)
  (hpq : ∀ x, ((x + 2) * (x - 6) ≤ 0) → (2 - m ≤ x ∧ x ≤ 2 + m)) : m ≥ 4 := by
  sorry

-- Problem (2)
theorem range_of_x_for_m5 (h : m = 5)
  (hp_or_q : ∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) )
  (hp_and_not_q : ∀ x, ¬(((x + 2) * (x - 6) ≤ 0) ∧ (-3 ≤ x ∧ x ≤ 7))):
  ∀ x, x ∈ Set.Ico (-3) (-2) ∨ x ∈ Set.Ioc (6) (7) := by
  sorry

end sufficient_condition_m_ge_4_range_of_x_for_m5_l762_76281


namespace geometric_sequence_eighth_term_l762_76286

theorem geometric_sequence_eighth_term (a r : ℝ) (h1 : a * r ^ 3 = 12) (h2 : a * r ^ 11 = 3) : 
  a * r ^ 7 = 6 * Real.sqrt 2 :=
sorry

end geometric_sequence_eighth_term_l762_76286


namespace cost_per_book_eq_three_l762_76249

-- Let T be the total amount spent, B be the number of books, and C be the cost per book
variables (T B C : ℕ)
-- Conditions: Edward spent $6 (T = 6) to buy 2 books (B = 2)
-- Each book costs the same amount (C = T / B)
axiom total_amount : T = 6
axiom number_of_books : B = 2

-- We need to prove that each book cost $3
theorem cost_per_book_eq_three (h1 : T = 6) (h2 : B = 2) : (T / B) = 3 := by
  sorry

end cost_per_book_eq_three_l762_76249


namespace trapezoid_area_l762_76220

theorem trapezoid_area (x : ℝ) (y : ℝ) :
  (∀ x, y = x + 1) →
  (∀ y, y = 12) →
  (∀ y, y = 7) →
  (∀ x, x = 0) →
  ∃ area,
  area = (1/2) * (6 + 11) * 5 ∧ area = 42.5 :=
by {
  sorry
}

end trapezoid_area_l762_76220


namespace composite_solid_volume_l762_76207

theorem composite_solid_volume :
  let V_prism := 2 * 2 * 1
  let V_cylinder := Real.pi * 1^2 * 3
  let V_overlap := Real.pi / 2
  V_prism + V_cylinder - V_overlap = 4 + 5 * Real.pi / 2 :=
by
  sorry

end composite_solid_volume_l762_76207


namespace work_completion_days_l762_76208

theorem work_completion_days (A B : ℕ) (h1 : A = 2 * B) (h2 : 6 * (A + B) = 18) : B = 1 → 18 = 18 :=
by
  sorry

end work_completion_days_l762_76208


namespace longer_diagonal_length_l762_76261

-- Conditions
def rhombus_side_length := 65
def shorter_diagonal_length := 72

-- Prove that the length of the longer diagonal is 108
theorem longer_diagonal_length : 
  (2 * (Real.sqrt ((rhombus_side_length: ℝ)^2 - (shorter_diagonal_length / 2)^2))) = 108 := 
by 
  sorry

end longer_diagonal_length_l762_76261


namespace cost_price_of_toy_l762_76221

-- Define the conditions
def sold_toys := 18
def selling_price := 23100
def gain_toys := 3

-- Define the cost price of one toy 
noncomputable def C := 1100

-- Lean 4 statement to prove the cost price
theorem cost_price_of_toy (C : ℝ) (sold_toys selling_price gain_toys : ℕ) (h1 : selling_price = (sold_toys + gain_toys) * C) : 
  C = 1100 := 
by
  sorry


end cost_price_of_toy_l762_76221


namespace maximum_area_l762_76278

-- Define necessary variables and conditions
variables (x y : ℝ)
variable (A : ℝ)
variable (peri : ℝ := 30)

-- Provide the premise that defines the perimeter condition
axiom perimeter_condition : 2 * x + 2 * y = peri

-- Define y in terms of x based on the perimeter condition
def y_in_terms_of_x (x : ℝ) : ℝ := 15 - x

-- Define the area of the rectangle in terms of x
def area (x : ℝ) : ℝ := x * (y_in_terms_of_x x)

-- The statement that needs to be proved
theorem maximum_area : A = 56.25 :=
by sorry

end maximum_area_l762_76278


namespace vehicle_value_last_year_l762_76298

theorem vehicle_value_last_year (value_this_year : ℝ) (ratio : ℝ) (value_this_year_cond : value_this_year = 16000) (ratio_cond : ratio = 0.8) :
  ∃ (value_last_year : ℝ), value_this_year = ratio * value_last_year ∧ value_last_year = 20000 :=
by
  use 20000
  sorry

end vehicle_value_last_year_l762_76298


namespace total_apples_l762_76213

def green_apples : ℕ := 2
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

theorem total_apples : green_apples + red_apples + yellow_apples = 19 :=
by
  -- Placeholder for the proof
  sorry

end total_apples_l762_76213


namespace speed_of_man_is_approx_4_99_l762_76280

noncomputable def train_length : ℝ := 110  -- meters
noncomputable def train_speed : ℝ := 50  -- km/h
noncomputable def time_to_pass_man : ℝ := 7.2  -- seconds

def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def relative_speed_kmph : ℝ :=
  mps_to_kmph (relative_speed train_length time_to_pass_man)

noncomputable def speed_of_man (relative_speed_kmph : ℝ) (train_speed : ℝ) : ℝ :=
  relative_speed_kmph - train_speed

theorem speed_of_man_is_approx_4_99 :
  abs (speed_of_man relative_speed_kmph train_speed - 4.99) < 0.01 :=
by
  sorry

end speed_of_man_is_approx_4_99_l762_76280


namespace sufficient_not_necessary_condition_not_necessary_condition_l762_76296

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  a^2 + b^2 = 1 → (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) → ¬(a^2 + b^2 = 1) :=
by
  sorry

end sufficient_not_necessary_condition_not_necessary_condition_l762_76296


namespace wine_count_l762_76256

theorem wine_count (S B total W : ℕ) (hS : S = 22) (hB : B = 17) (htotal : S - B + W = total) (htotal_val : total = 31) : W = 26 :=
by
  sorry

end wine_count_l762_76256


namespace quadrilateral_diagonal_length_l762_76215

theorem quadrilateral_diagonal_length (D A₁ A₂ : ℝ) (hA₁ : A₁ = 9) (hA₂ : A₂ = 6) (Area : ℝ) (hArea : Area = 165) :
  (1/2) * D * (A₁ + A₂) = Area → D = 22 :=
by
  -- Use the given conditions and solve to obtain D = 22
  intros
  sorry

end quadrilateral_diagonal_length_l762_76215


namespace no_linear_term_in_product_l762_76291

theorem no_linear_term_in_product (a : ℝ) (h : ∀ x : ℝ, (x + 4) * (x + a) - x^2 - 4 * a = 0) : a = -4 :=
sorry

end no_linear_term_in_product_l762_76291


namespace part_a_part_b_l762_76276

variable {p q n : ℕ}

-- Conditions
def coprime (a b : ℕ) : Prop := gcd a b = 1
def differ_by_more_than_one (p q : ℕ) : Prop := (q > p + 1) ∨ (p > q + 1)

-- Part (a): Prove there exists a natural number n such that p + n and q + n are not coprime
theorem part_a (coprime_pq : coprime p q) (diff : differ_by_more_than_one p q) : 
  ∃ n : ℕ, ¬ coprime (p + n) (q + n) :=
sorry

-- Part (b): Prove the smallest such n is 41 for p = 2 and q = 2023
theorem part_b (h : p = 2) (h1 : q = 2023) : 
  ∃ n : ℕ, (n = 41) ∧ (¬ coprime (2 + n) (2023 + n)) :=
sorry

end part_a_part_b_l762_76276


namespace artist_paints_total_exposed_surface_area_l762_76235

def num_cubes : Nat := 18
def edge_length : Nat := 1

-- Define the configuration of cubes
def bottom_layer_grid : Nat := 9 -- Number of cubes in the 3x3 grid (bottom layer)
def top_layer_cross : Nat := 9 -- Number of cubes in the cross shape (top layer)

-- Exposed surfaces in bottom layer
def bottom_layer_exposed_surfaces : Nat :=
  let top_surfaces := 9 -- 9 top surfaces for 9 cubes
  let corner_cube_sides := 4 * 3 -- 4 corners, 3 exposed sides each
  let edge_cube_sides := 4 * 2 -- 4 edge (non-corner) cubes, 2 exposed sides each
  top_surfaces + corner_cube_sides + edge_cube_sides

-- Exposed surfaces in top layer
def top_layer_exposed_surfaces : Nat :=
  let top_surfaces := 5 -- 5 top surfaces for 5 cubes in the cross
  let side_surfaces_of_cross_arms := 4 * 3 -- 4 arms, 3 exposed sides each
  top_surfaces + side_surfaces_of_cross_arms

-- Total exposed surface area
def total_exposed_surface_area : Nat :=
  bottom_layer_exposed_surfaces + top_layer_exposed_surfaces

-- Problem statement
theorem artist_paints_total_exposed_surface_area :
  total_exposed_surface_area = 46 := by
    sorry

end artist_paints_total_exposed_surface_area_l762_76235


namespace minnie_lucy_time_difference_is_66_minutes_l762_76292

noncomputable def minnie_time_uphill : ℚ := 12 / 6
noncomputable def minnie_time_downhill : ℚ := 18 / 25
noncomputable def minnie_time_flat : ℚ := 15 / 15

noncomputable def minnie_total_time : ℚ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat

noncomputable def lucy_time_flat : ℚ := 15 / 25
noncomputable def lucy_time_uphill : ℚ := 12 / 8
noncomputable def lucy_time_downhill : ℚ := 18 / 35

noncomputable def lucy_total_time : ℚ := lucy_time_flat + lucy_time_uphill + lucy_time_downhill

-- Convert hours to minutes
noncomputable def minnie_total_time_minutes : ℚ := minnie_total_time * 60
noncomputable def lucy_total_time_minutes : ℚ := lucy_total_time * 60

-- Difference in minutes
noncomputable def time_difference : ℚ := minnie_total_time_minutes - lucy_total_time_minutes

theorem minnie_lucy_time_difference_is_66_minutes : time_difference = 66 := by
  sorry

end minnie_lucy_time_difference_is_66_minutes_l762_76292


namespace smallest_class_size_l762_76271

theorem smallest_class_size (n : ℕ) 
  (eight_students_scored_120 : 8 * 120 ≤ n * 92)
  (three_students_scored_115 : 3 * 115 ≤ n * 92)
  (min_score_70 : 70 * n ≤ n * 92)
  (mean_score_92 : (8 * 120 + 3 * 115 + 70 * (n - 11)) / n = 92) :
  n = 25 :=
by
  sorry

end smallest_class_size_l762_76271


namespace sum_of_first_41_terms_is_94_l762_76202

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end sum_of_first_41_terms_is_94_l762_76202


namespace three_layer_carpet_area_l762_76279

-- Define the dimensions of the carpets and the hall
structure Carpet := (width : ℕ) (height : ℕ)

def principal_carpet : Carpet := ⟨6, 8⟩
def caretaker_carpet : Carpet := ⟨6, 6⟩
def parent_committee_carpet : Carpet := ⟨5, 7⟩
def hall : Carpet := ⟨10, 10⟩

-- Define the area function
def area (c : Carpet) : ℕ := c.width * c.height

-- Prove the area of the part of the hall covered by all three carpets
theorem three_layer_carpet_area : area ⟨3, 2⟩ = 6 :=
by
  sorry

end three_layer_carpet_area_l762_76279


namespace percentage_problem_l762_76226

theorem percentage_problem
    (x : ℕ) (h1 : (x:ℝ) / 100 * 20 = 8) :
    x = 40 :=
by
    sorry

end percentage_problem_l762_76226


namespace negation_of_p_l762_76218

-- Define the proposition p
def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- State the theorem: the negation of proposition p
theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by 
  sorry

end negation_of_p_l762_76218


namespace problem_statement_l762_76241

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then 4^x
  else if (-1 < x ∧ x < 0) then -4^(-x)
  else if (-2 < x ∧ x < -1) then -4^(x + 2)
  else if (1 < x ∧ x < 2) then 4^(x - 2)
  else 0

theorem problem_statement :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end problem_statement_l762_76241


namespace square_perimeter_l762_76269

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l762_76269


namespace corrected_mean_35_25_l762_76244

theorem corrected_mean_35_25 (n : ℕ) (mean : ℚ) (x_wrong x_correct : ℚ) :
  n = 20 → mean = 36 → x_wrong = 40 → x_correct = 25 → 
  ( (mean * n - x_wrong + x_correct) / n = 35.25) :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_35_25_l762_76244


namespace train_speed_is_117_l762_76223

noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_length / crossing_time
  (relative_speed - man_speed_mps) * 3.6

theorem train_speed_is_117 :
  train_speed 300 9 3 = 117 :=
by
  -- We leave the proof as sorry since only the statement is needed
  sorry

end train_speed_is_117_l762_76223


namespace sushi_downstream_distance_l762_76233

variable (sushi_speed : ℕ)
variable (stream_speed : ℕ := 12)
variable (upstream_distance : ℕ := 27)
variable (upstream_time : ℕ := 9)
variable (downstream_time : ℕ := 9)

theorem sushi_downstream_distance (h : upstream_distance = (sushi_speed - stream_speed) * upstream_time) : 
  ∃ (D_d : ℕ), D_d = (sushi_speed + stream_speed) * downstream_time ∧ D_d = 243 :=
by {
  -- We assume the given condition for upstream_distance
  sorry
}

end sushi_downstream_distance_l762_76233


namespace total_cost_l762_76237

-- Define the given conditions.
def coffee_pounds : ℕ := 4
def coffee_cost_per_pound : ℝ := 2.50
def milk_gallons : ℕ := 2
def milk_cost_per_gallon : ℝ := 3.50

-- The total cost Jasmine will pay is $17.00
theorem total_cost : coffee_pounds * coffee_cost_per_pound + milk_gallons * milk_cost_per_gallon = 17.00 := by
  sorry

end total_cost_l762_76237


namespace sheila_hourly_wage_is_correct_l762_76250

-- Definitions based on conditions
def works_hours_per_day_mwf : ℕ := 8
def works_days_mwf : ℕ := 3
def works_hours_per_day_tt : ℕ := 6
def works_days_tt : ℕ := 2
def weekly_earnings : ℕ := 216

-- Total calculated hours based on the problem conditions
def total_weekly_hours : ℕ := (works_hours_per_day_mwf * works_days_mwf) + (works_hours_per_day_tt * works_days_tt)

-- Target wage per hour
def wage_per_hour : ℕ := weekly_earnings / total_weekly_hours

-- The theorem stating the proof problem
theorem sheila_hourly_wage_is_correct : wage_per_hour = 6 := by
  sorry

end sheila_hourly_wage_is_correct_l762_76250


namespace find_n_l762_76274

noncomputable def arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℕ) (n d : ℕ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 5 = 12)
  (h4 : a n = 25) : 
  n = 13 := 
sorry

end find_n_l762_76274


namespace inequality_solution_real_l762_76260

theorem inequality_solution_real (x : ℝ) :
  (x + 1) * (2 - x) < 4 ↔ true :=
by
  sorry

end inequality_solution_real_l762_76260


namespace pythagorean_triple_transformation_l762_76204

theorem pythagorean_triple_transformation
  (a b c α β γ s p q r : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : α^2 + β^2 - γ^2 = 2)
  (h₃ : s = a * α + b * β - c * γ)
  (h₄ : p = a - α * s)
  (h₅ : q = b - β * s)
  (h₆ : r = c - γ * s) :
  p^2 + q^2 = r^2 :=
by
  sorry

end pythagorean_triple_transformation_l762_76204


namespace classroom_students_count_l762_76216

-- Definitions of given conditions
def total_students : ℕ := 1260

def aud_students : ℕ := (7 * total_students) / 18

def non_aud_students : ℕ := total_students - aud_students

def classroom_students : ℕ := (6 * non_aud_students) / 11

-- Theorem statement
theorem classroom_students_count : classroom_students = 420 := by
  sorry

end classroom_students_count_l762_76216


namespace john_sublets_to_3_people_l762_76268

def monthly_income (n : ℕ) : ℕ := 400 * n
def monthly_cost : ℕ := 900
def annual_profit (n : ℕ) : ℕ := 12 * (monthly_income n - monthly_cost)

theorem john_sublets_to_3_people
  (h1 : forall n : ℕ, monthly_income n - monthly_cost > 0)
  (h2 : annual_profit 3 = 3600) :
  3 = 3 := by
  sorry

end john_sublets_to_3_people_l762_76268


namespace geometric_sequence_problem_l762_76259

theorem geometric_sequence_problem (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 2 * a 5 = -32)
  (h2 : a 3 + a 4 = 4)
  (hq : ∃ (k : ℤ), q = k) :
  a 9 = -256 := 
sorry

end geometric_sequence_problem_l762_76259


namespace Jeff_total_ounces_of_peanut_butter_l762_76232

theorem Jeff_total_ounces_of_peanut_butter
    (jars : ℕ)
    (equal_count : ℕ)
    (total_jars : jars = 9)
    (j16 : equal_count = 3) 
    (j28 : equal_count = 3)
    (j40 : equal_count = 3) :
    (3 * 16 + 3 * 28 + 3 * 40 = 252) :=
by
  sorry

end Jeff_total_ounces_of_peanut_butter_l762_76232


namespace move_line_upwards_l762_76290

theorem move_line_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by
  intro h
  sorry

end move_line_upwards_l762_76290


namespace value_of_expression_l762_76267

theorem value_of_expression (x : ℝ) : 
  let a := 2000 * x + 2001
  let b := 2000 * x + 2002
  let c := 2000 * x + 2003
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end value_of_expression_l762_76267


namespace range_of_a_l762_76262

variable (A B : Set ℝ) (a : ℝ)

def setA : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def setB : Set ℝ := {x | (2^(1 - x) + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

theorem range_of_a :
  A ⊆ B ↔ (-4 ≤ a) ∧ (a ≤ -1) :=
by
  sorry

end range_of_a_l762_76262


namespace provisions_last_for_girls_l762_76273

theorem provisions_last_for_girls (P : ℝ) (G : ℝ) (h1 : P / (50 * G) = P / (250 * (G + 20))) : G = 25 := 
by
  sorry

end provisions_last_for_girls_l762_76273


namespace rachel_study_time_l762_76251

-- Define the conditions
def pages_math := 2
def pages_reading := 3
def pages_biology := 10
def pages_history := 4
def pages_physics := 5
def pages_chemistry := 8

def total_pages := pages_math + pages_reading + pages_biology + pages_history + pages_physics + pages_chemistry

def percent_study_time_biology := 30
def percent_study_time_reading := 30

-- State the theorem
theorem rachel_study_time :
  percent_study_time_biology = 30 ∧ 
  percent_study_time_reading = 30 →
  (100 - (percent_study_time_biology + percent_study_time_reading)) = 40 :=
by
  sorry

end rachel_study_time_l762_76251


namespace fraction_of_teeth_removed_l762_76222

theorem fraction_of_teeth_removed
  (total_teeth : ℕ)
  (initial_teeth : ℕ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (second_removed : ℕ)
  (third_removed : ℕ)
  (fourth_removed : ℕ)
  (total_removed : ℕ)
  (first_removed : ℕ)
  (fraction_first_removed : ℚ) :
  total_teeth = 32 →
  initial_teeth = 32 →
  second_fraction = 3 / 8 →
  third_fraction = 1 / 2 →
  second_removed = 12 →
  third_removed = 16 →
  fourth_removed = 4 →
  total_removed = 40 →
  first_removed + second_removed + third_removed + fourth_removed = total_removed →
  first_removed = 8 →
  fraction_first_removed = first_removed / initial_teeth →
  fraction_first_removed = 1 / 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end fraction_of_teeth_removed_l762_76222


namespace prod_sum_reciprocal_bounds_l762_76217

-- Define the product of the sum of three positive numbers and the sum of their reciprocals.
theorem prod_sum_reciprocal_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end prod_sum_reciprocal_bounds_l762_76217


namespace common_ratio_of_geometric_sequence_l762_76289

variable (a₁ q : ℝ)

def geometric_sequence (n : ℕ) := a₁ * q^n

theorem common_ratio_of_geometric_sequence
  (h_sum : geometric_sequence a₁ q 0 + geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 3 * a₁) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_sequence_l762_76289


namespace prob_four_children_at_least_one_boy_one_girl_l762_76254

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end prob_four_children_at_least_one_boy_one_girl_l762_76254


namespace order_of_a_b_c_l762_76231

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem order_of_a_b_c : b < a ∧ a < c :=
by
  sorry

end order_of_a_b_c_l762_76231


namespace quadratic_complete_square_l762_76209

open Real

theorem quadratic_complete_square (d e : ℝ) :
  (∀ x, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  intros h
  have h_eq := h 12
  sorry

end quadratic_complete_square_l762_76209


namespace find_x_value_l762_76236

theorem find_x_value (x : ℝ) 
  (h₁ : 1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) :
  x = (-1 + Real.sqrt 97) / 6 :=
sorry

end find_x_value_l762_76236


namespace simplify_expression_l762_76245

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x * (x - 4) = 2 * x^2 + 4 := by
  sorry

end simplify_expression_l762_76245


namespace inverse_sum_l762_76293

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2

theorem inverse_sum :
  let f_inv_2 := (1 + Real.sqrt 3)
  let f_inv_1 := 2
  let f_inv_4 := -1
  f_inv_2 + f_inv_1 + f_inv_4 = 2 + Real.sqrt 3 :=
by
  sorry

end inverse_sum_l762_76293


namespace minimal_degree_of_g_l762_76219

noncomputable def g_degree_minimal (f g h : Polynomial ℝ) (deg_f : ℕ) (deg_h : ℕ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h) : Prop :=
  Polynomial.degree f = deg_f ∧ Polynomial.degree h = deg_h → Polynomial.degree g = 12

theorem minimal_degree_of_g (f g h : Polynomial ℝ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h)
    (deg_f : Polynomial.degree f = 5) (deg_h : Polynomial.degree h = 12) :
    Polynomial.degree g = 12 := by
  sorry

end minimal_degree_of_g_l762_76219


namespace mushrooms_safe_to_eat_l762_76230

theorem mushrooms_safe_to_eat (S : ℕ) (Total_mushrooms Poisonous_mushrooms Uncertain_mushrooms : ℕ)
  (h1: Total_mushrooms = 32)
  (h2: Poisonous_mushrooms = 2 * S)
  (h3: Uncertain_mushrooms = 5)
  (h4: S + Poisonous_mushrooms + Uncertain_mushrooms = Total_mushrooms) :
  S = 9 :=
sorry

end mushrooms_safe_to_eat_l762_76230


namespace inequality_not_always_true_l762_76272

theorem inequality_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬(∀ a > 0, ∀ b > 0, (2 / ((1 / a) + (1 / b)) ≥ Real.sqrt (a * b))) :=
sorry

end inequality_not_always_true_l762_76272


namespace find_a_of_tangent_area_l762_76285

theorem find_a_of_tangent_area (a : ℝ) (h : a > 0) (h_area : (a^3 / 4) = 2) : a = 2 :=
by
  -- Proof is omitted as it's not required.
  sorry

end find_a_of_tangent_area_l762_76285


namespace axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l762_76238

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt 3 * Real.sin (Real.pi - x) + 5 * Real.sin (Real.pi / 2 + x) + 5

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi / 3 + k * Real.pi) :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi - Real.pi / 6) = 5 :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x) - 8

theorem g_max_value :
  ∀ x : ℝ, g x ≤ 2 :=
sorry

theorem g_increasing_intervals :
  ∀ k : ℤ, -Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≤ g (x + 1) :=
sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≥ g (x + 1) :=
sorry

end axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l762_76238


namespace min_val_of_q_l762_76282

theorem min_val_of_q (p q : ℕ) (h1 : 72 / 487 < p / q) (h2 : p / q < 18 / 121) : 
  ∃ p q : ℕ, (72 / 487 < p / q) ∧ (p / q < 18 / 121) ∧ q = 27 :=
sorry

end min_val_of_q_l762_76282


namespace max_a_value_l762_76258

def f (a x : ℝ) : ℝ := x^3 - a*x^2 + (a^2 - 2)*x + 1

theorem max_a_value (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ f a m ≤ 0) → a ≤ 1 :=
by
  intro h
  sorry

end max_a_value_l762_76258


namespace find_b_l762_76287

noncomputable def point (x y : Float) : Float × Float := (x, y)

def line_y_eq_b_plus_x (b x : Float) : Float := b + x

def intersects_y_axis (b : Float) : Float × Float := (0, b)

def intersects_x_axis (b : Float) : Float × Float := (-b, 0)

def intersects_x_eq_5 (b : Float) : Float × Float := (5, b + 5)

def area_triangle_qrs (b : Float) : Float :=
  0.5 * (5 + b) * (b + 5)

def area_triangle_qop (b : Float) : Float :=
  0.5 * b * b

theorem find_b (b : Float) (h : b > 0) (h_area_ratio : area_triangle_qrs b / area_triangle_qop b = 4 / 9) : b = 5 :=
by
  sorry

end find_b_l762_76287


namespace mike_spent_total_l762_76266

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l762_76266


namespace non_rent_extra_expenses_is_3000_l762_76225

-- Define the constants
def cost_parts : ℕ := 800
def markup : ℝ := 1.4
def num_computers : ℕ := 60
def rent : ℕ := 5000
def profit : ℕ := 11200

-- Calculate the selling price per computer
def selling_price : ℝ := cost_parts * markup

-- Calculate the total revenue from selling 60 computers
def total_revenue : ℝ := selling_price * num_computers

-- Calculate the total cost of components for 60 computers
def total_cost_components : ℕ := cost_parts * num_computers

-- Calculate the total expenses
def total_expenses : ℝ := total_revenue - profit

-- Define the non-rent extra expenses
def non_rent_extra_expenses : ℝ := total_expenses - rent - total_cost_components

-- Prove that the non-rent extra expenses equal to $3000
theorem non_rent_extra_expenses_is_3000 : non_rent_extra_expenses = 3000 := sorry

end non_rent_extra_expenses_is_3000_l762_76225


namespace sum_arithmetic_sequence_n_ge_52_l762_76255

theorem sum_arithmetic_sequence_n_ge_52 (n : ℕ) : 
  (∃ k, k = n) → 22 - 3 * (n - 1) = 22 - 3 * (n - 1) ∧ n ∈ { k | 3 ≤ k ∧ k ≤ 13 } :=
by
  sorry

end sum_arithmetic_sequence_n_ge_52_l762_76255


namespace min_abs_value_sum_l762_76224

theorem min_abs_value_sum (x : ℚ) : (min (|x - 1| + |x + 3|) = 4) :=
sorry

end min_abs_value_sum_l762_76224


namespace find_m_l762_76263

theorem find_m (C D m : ℤ) (h1 : C = D + m) (h2 : C - 1 = 6 * (D - 1)) (h3 : C = D^3) : m = 0 :=
by sorry

end find_m_l762_76263


namespace cosine_square_plus_alpha_sine_l762_76228

variable (α : ℝ)

theorem cosine_square_plus_alpha_sine (h1 : 0 ≤ α) (h2 : α ≤ Real.pi / 2) : 
  Real.cos α * Real.cos α + α * Real.sin α ≥ 1 :=
sorry

end cosine_square_plus_alpha_sine_l762_76228


namespace digging_project_depth_l762_76257

theorem digging_project_depth : 
  ∀ (P : ℕ) (D : ℝ), 
  (12 * P) * (25 * 30 * D) / 12 = (12 * P) * (75 * 20 * 50) / 12 → 
  D = 100 :=
by
  intros P D h
  sorry

end digging_project_depth_l762_76257


namespace weight_of_7th_person_l762_76246

-- Defining the constants and conditions
def num_people_initial : ℕ := 6
def avg_weight_initial : ℝ := 152
def num_people_total : ℕ := 7
def avg_weight_total : ℝ := 151

-- Calculating the total weights from the given average weights
def total_weight_initial := num_people_initial * avg_weight_initial
def total_weight_total := num_people_total * avg_weight_total

-- Theorem stating the weight of the 7th person
theorem weight_of_7th_person : total_weight_total - total_weight_initial = 145 := 
sorry

end weight_of_7th_person_l762_76246


namespace matrix_determinant_zero_l762_76264

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det (Matrix.of ![![1, a + b, b + c], ![1, a + 2 * b, b + 2 * c], ![1, a + 3 * b, b + 3 * c]]) = 0 := 
by
  sorry

end matrix_determinant_zero_l762_76264


namespace baker_sold_more_pastries_l762_76295

theorem baker_sold_more_pastries {cakes_made pastries_made pastries_sold cakes_sold : ℕ}
    (h1 : cakes_made = 105)
    (h2 : pastries_made = 275)
    (h3 : pastries_sold = 214)
    (h4 : cakes_sold = 163) :
    pastries_sold - cakes_sold = 51 := by
  sorry

end baker_sold_more_pastries_l762_76295


namespace min_distance_AB_tangent_line_circle_l762_76242

theorem min_distance_AB_tangent_line_circle 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h_tangent : a^2 + b^2 = 1) :
  ∃ A B : ℝ × ℝ, (A = (0, 1/b) ∧ B = (2/a, 0)) ∧ dist A B = 3 :=
by
  sorry

end min_distance_AB_tangent_line_circle_l762_76242


namespace truck_wheels_l762_76299

theorem truck_wheels (t x : ℝ) (wheels_front : ℕ) (wheels_other : ℕ) :
  (t = 1.50 + 1.50 * (x - 2)) → (t = 6) → (wheels_front = 2) → (wheels_other = 4) → x = 5 → 
  (wheels_front + wheels_other * (x - 1) = 18) :=
by
  intros h1 h2 h3 h4 h5
  rw [h5] at *
  sorry

end truck_wheels_l762_76299


namespace find_value_l762_76203

variables (a b c d : ℝ)

theorem find_value
  (h1 : a - b = 3)
  (h2 : c + d = 2) :
  (a + c) - (b - d) = 5 :=
by sorry

end find_value_l762_76203


namespace decreased_value_of_expression_l762_76252

theorem decreased_value_of_expression (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  (x' * y' * z'^2) = 0.1296 * (x * y * z^2) :=
by
  sorry

end decreased_value_of_expression_l762_76252


namespace cube_sum_l762_76206

theorem cube_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end cube_sum_l762_76206


namespace find_square_number_divisible_by_three_between_90_and_150_l762_76205

theorem find_square_number_divisible_by_three_between_90_and_150 :
  ∃ x : ℕ, 90 < x ∧ x < 150 ∧ ∃ y : ℕ, x = y * y ∧ 3 ∣ x ∧ x = 144 := 
by 
  sorry

end find_square_number_divisible_by_three_between_90_and_150_l762_76205


namespace more_valley_than_humpy_l762_76248

def is_humpy (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 > d4 ∧ d4 > d5

def is_valley (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 < d5

def starts_with_5 (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  d1 = 5

theorem more_valley_than_humpy :
  (∃ m, starts_with_5 m ∧ is_humpy m) → (∃ n, starts_with_5 n ∧ is_valley n) ∧ 
  (∀ x, starts_with_5 x → is_humpy x → ∃ y, starts_with_5 y ∧ is_valley y ∧ y ≠ x) :=
by sorry

end more_valley_than_humpy_l762_76248


namespace Carlton_button_up_shirts_l762_76214

/-- 
Given that the number of sweater vests V is twice the number of button-up shirts S, 
and the total number of unique outfits (each combination of a sweater vest and a button-up shirt) is 18, 
prove that the number of button-up shirts S is 3. 
-/
theorem Carlton_button_up_shirts (V S : ℕ) (h1 : V = 2 * S) (h2 : V * S = 18) : S = 3 := by
  sorry

end Carlton_button_up_shirts_l762_76214


namespace domain_of_f_f_is_monotonically_increasing_l762_76201

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 8) + 3

theorem domain_of_f :
  ∀ x, (x ≠ 5 * π / 16 + k * π / 2) := sorry

theorem f_is_monotonically_increasing :
  ∀ x, (π / 16 < x ∧ x < 3 * π / 16 → f x < f (x + ε)) := sorry

end domain_of_f_f_is_monotonically_increasing_l762_76201


namespace two_pow_65537_mod_19_l762_76247

theorem two_pow_65537_mod_19 : (2 ^ 65537) % 19 = 2 := by
  -- We will use Fermat's Little Theorem and given conditions.
  sorry

end two_pow_65537_mod_19_l762_76247


namespace female_democrats_count_l762_76288

-- Define the parameters and conditions
variables (F M D_f D_m D_total : ℕ)
variables (h1 : F + M = 840)
variables (h2 : D_total = 1/3 * (F + M))
variables (h3 : D_f = 1/2 * F)
variables (h4 : D_m = 1/4 * M)
variables (h5 : D_total = D_f + D_m)

-- State the theorem
theorem female_democrats_count : D_f = 140 :=
by
  sorry

end female_democrats_count_l762_76288


namespace profit_without_discount_l762_76270

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 0.05
noncomputable def profit_with_discount_percentage : ℝ := 0.387
noncomputable def selling_price_with_discount : ℝ := cost_price * (1 + profit_with_discount_percentage)

noncomputable def profit_without_discount_percentage : ℝ :=
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percentage)
  ((selling_price_without_discount - cost_price) / cost_price) * 100

theorem profit_without_discount :
  profit_without_discount_percentage = 45.635 := by
  sorry

end profit_without_discount_l762_76270


namespace probability_space_diagonal_l762_76210

theorem probability_space_diagonal : 
  let vertices := 8
  let space_diagonals := 4
  let total_pairs := Nat.choose vertices 2
  4 / total_pairs = 1 / 7 :=
by
  sorry

end probability_space_diagonal_l762_76210


namespace consecutive_integers_product_l762_76283

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l762_76283


namespace max_possible_percent_error_in_garden_area_l762_76243

open Real

theorem max_possible_percent_error_in_garden_area :
  ∃ (error_max : ℝ), error_max = 21 :=
by
  -- Given conditions
  let accurate_diameter := 30
  let max_error_percent := 10

  -- Defining lower and upper bounds for the diameter
  let lower_diameter := accurate_diameter - accurate_diameter * (max_error_percent / 100)
  let upper_diameter := accurate_diameter + accurate_diameter * (max_error_percent / 100)

  -- Calculating the exact and potential extreme areas
  let exact_area := π * (accurate_diameter / 2) ^ 2
  let lower_area := π * (lower_diameter / 2) ^ 2
  let upper_area := π * (upper_diameter / 2) ^ 2

  -- Calculating the percent errors
  let lower_error_percent := ((exact_area - lower_area) / exact_area) * 100
  let upper_error_percent := ((upper_area - exact_area) / exact_area) * 100

  -- We need to show the maximum error is 21%
  use upper_error_percent -- which should be 21% according to the problem statement
  sorry -- proof goes here

end max_possible_percent_error_in_garden_area_l762_76243


namespace kelly_single_shot_decrease_l762_76253

def kelly_salary_decrease (s : ℝ) : ℝ :=
  let first_cut := s * 0.92
  let second_cut := first_cut * 0.86
  let third_cut := second_cut * 0.82
  third_cut

theorem kelly_single_shot_decrease :
  let original_salary := 1.0 -- Assume original salary is 1 for percentage calculation
  let final_salary := kelly_salary_decrease original_salary
  (100 : ℝ) - (final_salary * 100) = 34.8056 :=
by
  sorry

end kelly_single_shot_decrease_l762_76253


namespace spinner_prob_l762_76275

theorem spinner_prob:
  let sections := 4
  let prob := 1 / sections
  let prob_not_e := 1 - prob
  (prob_not_e * prob_not_e) = 9 / 16 :=
by
  sorry

end spinner_prob_l762_76275


namespace calculate_c_from_law_of_cosines_l762_76294

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem calculate_c_from_law_of_cosines 
  (a b c : ℝ) (B : ℝ)
  (ha : a = 8) (hb : b = 7) (hB : B = Real.pi / 3) : 
  (c = 3) ∨ (c = 5) :=
sorry

end calculate_c_from_law_of_cosines_l762_76294


namespace find_amount_l762_76234

theorem find_amount (x : ℝ) (A : ℝ) (h1 : 0.65 * x = 0.20 * A) (h2 : x = 230) : A = 747.5 := by
  sorry

end find_amount_l762_76234


namespace part1_l762_76277

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l762_76277


namespace molecular_weight_of_1_mole_l762_76200

theorem molecular_weight_of_1_mole (m : ℝ) (w : ℝ) (h : 7 * m = 420) : m = 60 :=
by
  sorry

end molecular_weight_of_1_mole_l762_76200


namespace int_to_fourth_power_l762_76239

theorem int_to_fourth_power:
  3^4 * 9^8 = 243^4 :=
by 
  sorry

end int_to_fourth_power_l762_76239


namespace expressions_equality_l762_76240

-- Assumptions that expressions (1) and (2) are well-defined (denominators are non-zero)
variable {a b c m n p : ℝ}
variable (h1 : m ≠ 0)
variable (h2 : bp + cn ≠ 0)
variable (h3 : n ≠ 0)
variable (h4 : ap + cm ≠ 0)

-- Main theorem statement
theorem expressions_equality
  (hS : (a / m) + (bc + np) / (bp + cn) = 0) :
  (b / n) + (ac + mp) / (ap + cm) = 0 :=
  sorry

end expressions_equality_l762_76240


namespace johnson_potatoes_left_l762_76265

theorem johnson_potatoes_left :
  ∀ (initial gina tom anne remaining : Nat),
  initial = 300 →
  gina = 69 →
  tom = 2 * gina →
  anne = tom / 3 →
  remaining = initial - (gina + tom + anne) →
  remaining = 47 := by
sorry

end johnson_potatoes_left_l762_76265


namespace roller_skate_wheels_l762_76227

theorem roller_skate_wheels (number_of_people : ℕ)
  (feet_per_person : ℕ)
  (skates_per_foot : ℕ)
  (wheels_per_skate : ℕ)
  (h_people : number_of_people = 40)
  (h_feet : feet_per_person = 2)
  (h_skates : skates_per_foot = 1)
  (h_wheels : wheels_per_skate = 4)
  : (number_of_people * feet_per_person * skates_per_foot * wheels_per_skate) = 320 := 
by
  sorry

end roller_skate_wheels_l762_76227


namespace total_screens_sold_l762_76212

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l762_76212
