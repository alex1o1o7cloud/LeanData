import Mathlib

namespace lampshire_parade_group_max_members_l213_21381

theorem lampshire_parade_group_max_members 
  (n : ℕ) 
  (h1 : 30 * n % 31 = 7)
  (h2 : 30 * n % 17 = 0)
  (h3 : 30 * n < 1500) :
  30 * n = 1020 :=
sorry

end lampshire_parade_group_max_members_l213_21381


namespace skipping_ropes_l213_21356

theorem skipping_ropes (length1 length2 : ℕ) (h1 : length1 = 18) (h2 : length2 = 24) :
  ∃ (max_length : ℕ) (num_ropes : ℕ),
    max_length = Nat.gcd length1 length2 ∧
    max_length = 6 ∧
    num_ropes = length1 / max_length + length2 / max_length ∧
    num_ropes = 7 :=
by
  have max_length : ℕ := Nat.gcd length1 length2
  have num_ropes : ℕ := length1 / max_length + length2 / max_length
  use max_length, num_ropes
  sorry

end skipping_ropes_l213_21356


namespace intersection_A_B_l213_21369

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l213_21369


namespace animals_on_farm_l213_21346

theorem animals_on_farm (cows : ℕ) (sheep : ℕ) (pigs : ℕ) 
  (h1 : cows = 12) 
  (h2 : sheep = 2 * cows) 
  (h3 : pigs = 3 * sheep) : 
  cows + sheep + pigs = 108 := 
by
  sorry

end animals_on_farm_l213_21346


namespace min_abs_sum_l213_21331

theorem min_abs_sum (a b c : ℝ) (h₁ : a + b + c = -2) (h₂ : a * b * c = -4) :
  ∃ (m : ℝ), m = min (abs a + abs b + abs c) 6 :=
sorry

end min_abs_sum_l213_21331


namespace fish_left_in_sea_l213_21370

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end fish_left_in_sea_l213_21370


namespace find_a4_l213_21394

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

axiom hyp1 : is_arithmetic_sequence a d
axiom hyp2 : a 5 = 9
axiom hyp3 : a 7 + a 8 = 28

-- Goal
theorem find_a4 : a 4 = 7 :=
by
  sorry

end find_a4_l213_21394


namespace expression_range_l213_21310

theorem expression_range (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2)
  + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ∧ 
  (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ≤ 8 :=
sorry

end expression_range_l213_21310


namespace pizza_consumption_order_l213_21375

theorem pizza_consumption_order :
  let e := 1/6
  let s := 1/4
  let n := 1/3
  let o := 1/8
  let j := 1 - e - s - n - o
  (n > s) ∧ (s > e) ∧ (e = j) ∧ (j > o) :=
by
  sorry

end pizza_consumption_order_l213_21375


namespace anya_hair_growth_l213_21393

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l213_21393


namespace annual_growth_rate_l213_21341

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l213_21341


namespace decreasing_power_function_l213_21358

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem decreasing_power_function (k : ℝ) : 
  (∀ x : ℝ, 0 < x → (f k x) ≤ 0) ↔ k < 0 ∧ k ≠ 0 := sorry

end decreasing_power_function_l213_21358


namespace original_area_area_after_translation_l213_21365

-- Defining vectors v, w, and t
def v : ℝ × ℝ := (6, -4)
def w : ℝ × ℝ := (-8, 3)
def t : ℝ × ℝ := (3, 2)

-- Function to compute the determinant of two vectors in R^2
def det (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

-- The area of a parallelogram is the absolute value of the determinant
def parallelogram_area (v w : ℝ × ℝ) : ℝ := |det v w|

-- Proving the original area is 14
theorem original_area : parallelogram_area v w = 14 := by
  sorry

-- Proving the area remains the same after translation
theorem area_after_translation : parallelogram_area v w = parallelogram_area (v.1 + t.1, v.2 + t.2) (w.1 + t.1, w.2 + t.2) := by
  sorry

end original_area_area_after_translation_l213_21365


namespace incorrect_vertex_is_false_l213_21395

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 2)^2 + 1

-- Define the incorrect hypothesis: Vertex at (-2, 1)
def incorrect_vertex (x y : ℝ) : Prop := (x, y) = (-2, 1)

-- Proposition to prove that the vertex is not at (-2, 1)
theorem incorrect_vertex_is_false : ¬ ∃ x y, (x, y) = (-2, 1) ∧ parabola x = y :=
by
  sorry

end incorrect_vertex_is_false_l213_21395


namespace largest_integer_x_divisible_l213_21324

theorem largest_integer_x_divisible (x : ℤ) : 
  (∃ x : ℤ, (x^2 + 3 * x + 8) % (x - 2) = 0 ∧ x ≤ 1) → x = 1 :=
sorry

end largest_integer_x_divisible_l213_21324


namespace find_k_l213_21300

def equation (k : ℝ) (x : ℝ) : Prop := 2 * x^2 + 3 * x - k = 0

theorem find_k (k : ℝ) (h : equation k 7) : k = 119 :=
by
  sorry

end find_k_l213_21300


namespace unit_trip_to_expo_l213_21354

theorem unit_trip_to_expo (n : ℕ) (cost : ℕ) (total_cost : ℕ) :
  (n ≤ 30 → cost = 120) ∧ 
  (n > 30 → cost = 120 - 2 * (n - 30) ∧ cost ≥ 90) →
  (total_cost = 4000) →
  (total_cost = n * cost) →
  n = 40 :=
by
  sorry

end unit_trip_to_expo_l213_21354


namespace dividend_rate_of_stock_l213_21388

variable (MarketPrice : ℝ) (YieldPercent : ℝ) (DividendPercent : ℝ)
variable (NominalValue : ℝ) (AnnualDividend : ℝ)

def stock_dividend_rate_condition (YieldPercent MarketPrice NominalValue DividendPercent : ℝ) 
  (AnnualDividend : ℝ) : Prop :=
  YieldPercent = 20 ∧ MarketPrice = 125 ∧ DividendPercent = 0.25 ∧ NominalValue = 100 ∧
  AnnualDividend = (YieldPercent / 100) * MarketPrice

theorem dividend_rate_of_stock :
  stock_dividend_rate_condition 20 125 100 0.25 25 → (DividendPercent * NominalValue) = 25 :=
by 
  sorry

end dividend_rate_of_stock_l213_21388


namespace f_99_eq_1_l213_21352

-- Define an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- The conditions to be satisfied by the function f
variables (f : ℝ → ℝ)
variable (h_even : is_even_function f)
variable (h_f1 : f 1 = 1)
variable (h_period : ∀ x, f (x + 4) = f x)

-- Prove that f(99) = 1
theorem f_99_eq_1 : f 99 = 1 :=
by
  sorry

end f_99_eq_1_l213_21352


namespace fourth_intersection_point_l213_21363

noncomputable def fourth_point_of_intersection : Prop :=
  let hyperbola (x y : ℝ) := x * y = 1
  let circle (x y : ℝ) := (x - 1)^2 + (y + 1)^2 = 10
  let known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/2, 2)]
  let fourth_point := (-1/6, -6)
  (hyperbola 3 (1/3)) ∧ (hyperbola (-4) (-1/4)) ∧ (hyperbola (1/2) 2) ∧
  (circle 3 (1/3)) ∧ (circle (-4) (-1/4)) ∧ (circle (1/2) 2) ∧ 
  (hyperbola (-1/6) (-6)) ∧ (circle (-1/6) (-6)) ∧ 
  ∀ (x y : ℝ), (hyperbola x y) → (circle x y) → ((x, y) = fourth_point ∨ (x, y) ∈ known_points)
  
theorem fourth_intersection_point :
  fourth_point_of_intersection :=
sorry

end fourth_intersection_point_l213_21363


namespace total_games_in_season_is_correct_l213_21347

-- Definitions based on given conditions
def games_per_month : ℕ := 7
def season_months : ℕ := 2

-- The theorem to prove
theorem total_games_in_season_is_correct : 
  (games_per_month * season_months = 14) :=
by
  sorry

end total_games_in_season_is_correct_l213_21347


namespace cyclic_quadrilateral_sides_equal_l213_21303

theorem cyclic_quadrilateral_sides_equal
  (A B C D P : ℝ) -- Points represented as reals for simplicity
  (AB CD BC AD : ℝ) -- Lengths of sides AB, CD, BC, AD
  (a b c d e θ : ℝ) -- Various lengths and angle as given in the solution
  (h1 : a + e = b + c + d)
  (h2 : (1 / 2) * a * e * Real.sin θ = (1 / 2) * b * e * Real.sin θ + (1 / 2) * c * d * Real.sin θ) :
  c = e ∨ d = e := sorry

end cyclic_quadrilateral_sides_equal_l213_21303


namespace geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l213_21351

variable (a b c : ℝ)

theorem geometric_implies_b_squared_eq_ac
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∃ r : ℝ, b = r * a ∧ c = r * b) :
  b^2 = a * c :=
by
  sorry

theorem not_geometric_if_all_zero 
  (hz : a = 0 ∧ b = 0 ∧ c = 0) : 
  ¬(∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

theorem sufficient_but_not_necessary_condition :
  (∃ r : ℝ, b = r * a ∧ c = r * b → b^2 = a * c) ∧ ¬(b^2 = a * c → ∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

end geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l213_21351


namespace number_of_authors_l213_21302

/-- Define the number of books each author has and the total number of books. -/
def books_per_author : ℕ := 33
def total_books : ℕ := 198

/-- Main theorem stating that the number of authors Jack has is derived by dividing total books by the number of books per author. -/
theorem number_of_authors (n : ℕ) (h : total_books = n * books_per_author) : n = 6 := by
  sorry

end number_of_authors_l213_21302


namespace f_value_third_quadrant_l213_21304

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3 * Real.pi / 2 + α))

theorem f_value_third_quadrant (α : ℝ) (h1 : (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)) (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end f_value_third_quadrant_l213_21304


namespace apples_used_l213_21387

theorem apples_used (initial_apples remaining_apples : ℕ) (h_initial : initial_apples = 40) (h_remaining : remaining_apples = 39) : initial_apples - remaining_apples = 1 := 
by
  sorry

end apples_used_l213_21387


namespace correct_propositions_l213_21380

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def symmetry_about_points (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x, f (x + k) = f (x - k)

theorem correct_propositions (h1: is_odd_function f) (h2 : ∀ x, f (x + 1) = f (x -1)) :
  period_2 f ∧ (∀ k : ℤ, symmetry_about_points f k) :=
by
  sorry

end correct_propositions_l213_21380


namespace baskets_of_peaches_l213_21362

theorem baskets_of_peaches (n : ℕ) :
  (∀ x : ℕ, (n * 2 = 14) → (n = x)) := by
  sorry

end baskets_of_peaches_l213_21362


namespace children_getting_on_bus_l213_21325

theorem children_getting_on_bus (a b c: ℕ) (ha : a = 64) (hb : b = 78) (hc : c = b - a) : c = 14 :=
by
  sorry

end children_getting_on_bus_l213_21325


namespace period_f_2pi_max_value_f_exists_max_f_l213_21361

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem period_f_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem max_value_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 := by
  sorry

-- Optional: Existence of the maximum value.
theorem exists_max_f : ∃ x : ℝ, f x = Real.sin 1 + 1 := by
  sorry

end period_f_2pi_max_value_f_exists_max_f_l213_21361


namespace roots_real_l213_21333

variable {x p q k : ℝ}
variable {x1 x2 : ℝ}

theorem roots_real 
  (h1 : x^2 + p * x + q = 0) 
  (h2 : p = -(x1 + x2)) 
  (h3 : q = x1 * x2) 
  (h4 : x1 ≠ x2) 
  (h5 :  x1^2 - 2*x1*x2 + x2^2 + 4*q = 0):
  (∃ y1 y2, y1 = k * x1 + (1 / k) * x2 ∧ y2 = k * x2 + (1 / k) * x1 ∧ 
    (y1^2 + (k + 1/k) * p * y1 + (p^2 + q * ((k - 1/k)^2)) = 0) ∧ 
    (y2^2 + (k + 1/k) * p * y2 + (p^2 + q * ((k - 1/k)^2)) = 0)) → 
  (∃ z1 z2, z1 = k * x1 ∧ z2 = 1/k * x2 ∧ 
    (z1^2 - y1 * z1 + q = 0) ∧ 
    (z2^2 - y2 * z2 + q = 0)) :=
sorry

end roots_real_l213_21333


namespace ella_incorrect_answers_l213_21311

theorem ella_incorrect_answers
  (marion_score : ℕ)
  (ella_score : ℕ)
  (total_items : ℕ)
  (h1 : marion_score = 24)
  (h2 : marion_score = (ella_score / 2) + 6)
  (h3 : total_items = 40) : 
  total_items - ella_score = 4 :=
by
  sorry

end ella_incorrect_answers_l213_21311


namespace simplify_expression_l213_21342

theorem simplify_expression (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) * (x - 2) + (x - 2) * (2 * x^2 - 3 * x + 9) - (4 * x - 7) * (x - 2) * (x - 3) 
  = x^3 + x^2 + 12 * x - 36 := 
by
  sorry

end simplify_expression_l213_21342


namespace determinant_matrix_A_l213_21330

open Matrix

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2], ![1, 3, 4], ![0, -1, 1]]

theorem determinant_matrix_A :
  det matrix_A = 33 :=
by
  sorry

end determinant_matrix_A_l213_21330


namespace find_multiple_l213_21399

-- Define the conditions
def ReetaPencils : ℕ := 20
def TotalPencils : ℕ := 64

-- Define the question and proof statement
theorem find_multiple (AnikaPencils : ℕ) (M : ℕ) :
  AnikaPencils = ReetaPencils * M + 4 →
  AnikaPencils + ReetaPencils = TotalPencils →
  M = 2 :=
by
  intros hAnika hTotal
  -- Skip the proof
  sorry

end find_multiple_l213_21399


namespace range_of_a_l213_21368

variable {x a : ℝ}

theorem range_of_a (h1 : x > 1) (h2 : a ≤ x + 1 / (x - 1)) : a ≤ 3 :=
sorry

end range_of_a_l213_21368


namespace original_speed_of_Person_A_l213_21383

variable (v_A v_B : ℝ)

-- Define the conditions
def condition1 : Prop := v_B = 2 * v_A
def condition2 : Prop := v_A + 10 = 4 * (v_B - 5)

-- Define the theorem to prove
theorem original_speed_of_Person_A (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_B) : v_A = 18 := 
by
  sorry

end original_speed_of_Person_A_l213_21383


namespace angle_B_magnitude_value_of_b_l213_21306
open Real

theorem angle_B_magnitude (B : ℝ) (h : 2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) :
  B = π / 3 ∨ B = 2 * π / 3 := sorry

theorem value_of_b (a B S : ℝ) (hB : B = π / 3) (ha : a = 6) (hS : S = 6 * sqrt 3) :
  let c := 4
  let b := 2 * sqrt 7
  let half_angle_B := 1 / 2 * a * c * sin B
  half_angle_B = S :=
by
  sorry

end angle_B_magnitude_value_of_b_l213_21306


namespace apples_shared_l213_21373

-- Definitions and conditions based on problem statement
def initial_apples : ℕ := 89
def remaining_apples : ℕ := 84

-- The goal to prove that Ruth shared 5 apples with Peter
theorem apples_shared : initial_apples - remaining_apples = 5 := by
  sorry

end apples_shared_l213_21373


namespace largest_constant_inequality_l213_21360

theorem largest_constant_inequality (C : ℝ) (h : ∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) : 
  C ≤ 2 / Real.sqrt 3 :=
sorry

end largest_constant_inequality_l213_21360


namespace common_ratio_geometric_series_l213_21307

theorem common_ratio_geometric_series 
  (a : ℚ) (b : ℚ) (r : ℚ)
  (h_a : a = 4 / 5)
  (h_b : b = -5 / 12)
  (h_r : r = b / a) :
  r = -25 / 48 :=
by sorry

end common_ratio_geometric_series_l213_21307


namespace sample_size_correct_l213_21318

def total_students (freshmen sophomores juniors : ℕ) : ℕ :=
  freshmen + sophomores + juniors

def sample_size (total : ℕ) (prob : ℝ) : ℝ :=
  total * prob

theorem sample_size_correct (f : ℕ) (s : ℕ) (j : ℕ) (p : ℝ) (h_f : f = 400) (h_s : s = 320) (h_j : j = 280) (h_p : p = 0.2) :
  sample_size (total_students f s j) p = 200 :=
by
  sorry

end sample_size_correct_l213_21318


namespace students_before_Yoongi_l213_21389

theorem students_before_Yoongi (total_students : ℕ) (students_after_Yoongi : ℕ) 
  (condition1 : total_students = 20) (condition2 : students_after_Yoongi = 11) :
  total_students - students_after_Yoongi - 1 = 8 :=
by 
  sorry

end students_before_Yoongi_l213_21389


namespace ellen_lost_legos_l213_21329

theorem ellen_lost_legos (L_initial L_final : ℕ) (h1 : L_initial = 2080) (h2 : L_final = 2063) : L_initial - L_final = 17 := by
  sorry

end ellen_lost_legos_l213_21329


namespace balls_into_boxes_l213_21366

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l213_21366


namespace distance_dormitory_to_city_l213_21377

variable (D : ℝ)
variable (c : ℝ := 12)
variable (f := (1/5) * D)
variable (b := (2/3) * D)

theorem distance_dormitory_to_city (h : f + b + c = D) : D = 90 := by
  sorry

end distance_dormitory_to_city_l213_21377


namespace option_c_correct_l213_21340

theorem option_c_correct (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end option_c_correct_l213_21340


namespace square_area_dimensions_l213_21392

theorem square_area_dimensions (x : ℝ) (n : ℝ) : 
  (x^2 + (x + 12)^2 = 2120) → 
  (n = x + 12) → 
  (x = 26) → 
  (n = 38) := 
by
  sorry

end square_area_dimensions_l213_21392


namespace find_N_l213_21390

theorem find_N (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) :
  (x + y) / 3 = 1.222222222222222 := 
by
  -- We state the conditions.
  -- Lean will check whether these assumptions are consistent 
  sorry

end find_N_l213_21390


namespace two_pow_2001_mod_127_l213_21379

theorem two_pow_2001_mod_127 : (2^2001) % 127 = 64 := 
by
  sorry

end two_pow_2001_mod_127_l213_21379


namespace number_is_more_than_sum_l213_21337

theorem number_is_more_than_sum : 20.2 + 33.8 - 5.1 = 48.9 :=
by
  sorry

end number_is_more_than_sum_l213_21337


namespace measure_of_angle_A_l213_21348

-- Defining the measures of angles
def angle_B : ℝ := 50
def angle_C : ℝ := 40
def angle_D : ℝ := 30

-- Prove that measure of angle A is 120 degrees given the conditions
theorem measure_of_angle_A (B C D : ℝ) (hB : B = angle_B) (hC : C = angle_C) (hD : D = angle_D) : B + C + D + 60 = 180 -> 180 - (B + C + D + 60) = 120 :=
by sorry

end measure_of_angle_A_l213_21348


namespace bees_second_day_l213_21343

-- Define the number of bees on the first day
def bees_on_first_day : ℕ := 144 

-- Define the multiplier for the second day
def multiplier : ℕ := 3

-- Define the number of bees on the second day
def bees_on_second_day : ℕ := bees_on_first_day * multiplier

-- Theorem stating the number of bees seen on the second day
theorem bees_second_day : bees_on_second_day = 432 := by
  -- Proof is pending.
  sorry

end bees_second_day_l213_21343


namespace yen_per_pound_l213_21308

theorem yen_per_pound 
  (pounds_initial : ℕ) 
  (euros : ℕ) 
  (yen_initial : ℕ) 
  (pounds_per_euro : ℕ) 
  (yen_total : ℕ) 
  (hp : pounds_initial = 42) 
  (he : euros = 11) 
  (hy : yen_initial = 3000) 
  (hpe : pounds_per_euro = 2) 
  (hy_total : yen_total = 9400) 
  : (yen_total - yen_initial) / (pounds_initial + euros * pounds_per_euro) = 100 := 
by
  sorry

end yen_per_pound_l213_21308


namespace smallest_N_l213_21309

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l213_21309


namespace solve_for_a_minus_b_l213_21312

theorem solve_for_a_minus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 7) (h3 : |a + b| = a + b) : a - b = -2 := 
sorry

end solve_for_a_minus_b_l213_21312


namespace area_conversion_correct_l213_21349

-- Define the legs of the right triangle
def leg1 : ℕ := 60
def leg2 : ℕ := 80

-- Define the conversion factor
def square_feet_in_square_yard : ℕ := 9

-- Calculate the area of the triangle in square feet
def area_in_square_feet : ℕ := (leg1 * leg2) / 2

-- Calculate the area of the triangle in square yards
def area_in_square_yards : ℚ := area_in_square_feet / square_feet_in_square_yard

-- The theorem stating the problem
theorem area_conversion_correct : area_in_square_yards = 266 + 2 / 3 := by
  sorry

end area_conversion_correct_l213_21349


namespace rectangle_area_l213_21372

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l213_21372


namespace value_of_f_at_2_l213_21322

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem value_of_f_at_2 : f 2 = -2 := 
by 
  sorry

end value_of_f_at_2_l213_21322


namespace find_divisor_l213_21305

/-- Given a dividend of 15698, a quotient of 89, and a remainder of 14, find the divisor. -/
theorem find_divisor :
  ∃ D : ℕ, 15698 = 89 * D + 14 ∧ D = 176 :=
by
  sorry

end find_divisor_l213_21305


namespace complex_expression_simplification_l213_21371

-- Given: i is the imaginary unit
def i := Complex.I

-- Prove that the expression simplifies to -1
theorem complex_expression_simplification : (i^3 * (i + 1)) / (i - 1) = -1 := by
  -- We are skipping the proof and adding sorry for now
  sorry

end complex_expression_simplification_l213_21371


namespace range_of_p_l213_21332

def p (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_p : Set.Ici 9 = { y | ∃ x ≥ 0, p x = y } :=
by
  -- We skip the proof to only provide the statement as requested.
  sorry

end range_of_p_l213_21332


namespace no_third_degree_polynomial_exists_l213_21353

theorem no_third_degree_polynomial_exists (a b c d : ℤ) (h : a ≠ 0) :
  ¬(p 15 = 3 ∧ p 21 = 12 ∧ p = λ x => a * x ^ 3 + b * x ^ 2 + c * x + d) :=
sorry

end no_third_degree_polynomial_exists_l213_21353


namespace calc_expression_l213_21328

theorem calc_expression : (113^2 - 104^2) / 9 = 217 := by
  sorry

end calc_expression_l213_21328


namespace maximum_value_of_f_l213_21355

noncomputable def f (a x : ℝ) : ℝ := (1 + x) ^ a - a * x

theorem maximum_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  ∃ x : ℝ, x > -1 ∧ ∀ y : ℝ, y > -1 → f a y ≤ f a x ∧ f a x = 1 :=
by {
  sorry
}

end maximum_value_of_f_l213_21355


namespace find_d_from_sine_wave_conditions_l213_21396

theorem find_d_from_sine_wave_conditions (a b d : ℝ) (h1 : d + a = 4) (h2 : d - a = -2) : d = 1 :=
by {
  sorry
}

end find_d_from_sine_wave_conditions_l213_21396


namespace hyperbola_through_focus_and_asymptotes_l213_21313

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

def asymptotes_holds (x y : ℝ) : Prop :=
  (x + y = 0) ∨ (x - y = 0)

theorem hyperbola_through_focus_and_asymptotes :
  hyperbola parabola_focus.1 parabola_focus.2 ∧ asymptotes_holds parabola_focus.1 parabola_focus.2 :=
sorry

end hyperbola_through_focus_and_asymptotes_l213_21313


namespace average_speed_monday_to_wednesday_l213_21350

theorem average_speed_monday_to_wednesday :
  ∃ x : ℝ, (∀ (total_hours total_distance thursday_friday_distance : ℝ),
    total_hours = 2 * 5 ∧
    thursday_friday_distance = 9 * 2 * 2 ∧
    total_distance = 108 ∧
    total_distance - thursday_friday_distance = x * (2 * 3))
    → x = 12 :=
sorry

end average_speed_monday_to_wednesday_l213_21350


namespace identity_map_a_plus_b_l213_21384

theorem identity_map_a_plus_b (a b : ℝ) (h : ∀ x ∈ ({-1, b / a, 1} : Set ℝ), x ∈ ({a, b, b - a} : Set ℝ)) : a + b = -1 ∨ a + b = 1 :=
by
  sorry

end identity_map_a_plus_b_l213_21384


namespace least_positive_x_l213_21335

variable (a b : ℝ)

noncomputable def tan_inv (x : ℝ) : ℝ := Real.arctan x

theorem least_positive_x (x k : ℝ) 
  (h1 : Real.tan x = a / b)
  (h2 : Real.tan (2 * x) = b / (a + b))
  (h3 : Real.tan (3 * x) = (a - b) / (a + b))
  (h4 : x = tan_inv k)
  : k = 13 / 9 := sorry

end least_positive_x_l213_21335


namespace range_of_a_l213_21323

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x = 3 ∧ 3 * x - (a * x + 1) / 2 < 4 * x / 3) → a > 3 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  sorry

end range_of_a_l213_21323


namespace intersection_point_sum_l213_21320

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

axiom h2 : h 2 = 2
axiom j2 : j 2 = 2
axiom h4 : h 4 = 6
axiom j4 : j 4 = 6
axiom h6 : h 6 = 12
axiom j6 : j 6 = 12
axiom h8 : h 8 = 12
axiom j8 : j 8 = 12

theorem intersection_point_sum :
  (∃ x, h (x + 2) = j (2 * x)) →
  (h (2 + 2) = j (2 * 2) ∨ h (4 + 2) = j (2 * 4)) →
  (h (4) = 6 ∧ j (4) = 6 ∧ h 6 = 12 ∧ j 8 = 12) →
  (∃ x, (x = 2 ∧ (x + h (x + 2) = 8) ∨ x = 4 ∧ (x + h (x + 2) = 16))) :=
by
  sorry

end intersection_point_sum_l213_21320


namespace capsule_cost_difference_l213_21319

theorem capsule_cost_difference :
  let cost_per_capsule_r := 6.25 / 250
  let cost_per_capsule_t := 3.00 / 100
  cost_per_capsule_t - cost_per_capsule_r = 0.005 := by
  sorry

end capsule_cost_difference_l213_21319


namespace train_crosses_signal_post_in_40_seconds_l213_21339

noncomputable def time_to_cross_signal_post : Nat := 40

theorem train_crosses_signal_post_in_40_seconds
  (train_length : Nat) -- Length of the train in meters
  (bridge_length_km : Nat) -- Length of the bridge in kilometers
  (bridge_cross_time_min : Nat) -- Time to cross the bridge in minutes
  (constant_speed : Prop) -- Assumption that the speed is constant
  (h1 : train_length = 600) -- Train is 600 meters long
  (h2 : bridge_length_km = 9) -- Bridge is 9 kilometers long
  (h3 : bridge_cross_time_min = 10) -- Time to cross the bridge is 10 minutes
  (h4 : constant_speed) -- The train's speed is constant
  : time_to_cross_signal_post = 40 :=
sorry

end train_crosses_signal_post_in_40_seconds_l213_21339


namespace difference_of_two_numbers_l213_21391

theorem difference_of_two_numbers (a b : ℕ) (h₀ : a + b = 25800) (h₁ : b = 12 * a) (h₂ : b % 10 = 0) (h₃ : b / 10 = a) : b - a = 21824 :=
by 
  -- sorry to skip the proof
  sorry

end difference_of_two_numbers_l213_21391


namespace range_of_x_l213_21386

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Theorem statement
theorem range_of_x (x : ℝ) : (¬ q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  sorry

end range_of_x_l213_21386


namespace rearrangements_of_abcde_l213_21317

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 == 'a' ∧ c2 == 'b') ∨ 
  (c1 == 'b' ∧ c1 == 'a') ∨ 
  (c1 == 'b' ∧ c2 == 'c') ∨ 
  (c1 == 'c' ∧ c2 == 'b') ∨ 
  (c1 == 'c' ∧ c2 == 'd') ∨ 
  (c1 == 'd' ∧ c2 == 'c') ∨ 
  (c1 == 'd' ∧ c2 == 'e') ∨ 
  (c1 == 'e' ∧ c2 == 'd')

def is_valid_rearrangement (lst : List Char) : Bool :=
  match lst with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => 
    ¬is_adjacent c1 c2 ∧ is_valid_rearrangement (c2 :: rest)

def count_valid_rearrangements (chars : List Char) : Nat :=
  chars.permutations.filter is_valid_rearrangement |>.length

theorem rearrangements_of_abcde : count_valid_rearrangements ['a', 'b', 'c', 'd', 'e'] = 8 := 
by
  sorry

end rearrangements_of_abcde_l213_21317


namespace combined_weight_of_three_parcels_l213_21314

theorem combined_weight_of_three_parcels (x y z : ℕ)
  (h1 : x + y = 112) (h2 : y + z = 146) (h3 : z + x = 132) :
  x + y + z = 195 :=
by
  sorry

end combined_weight_of_three_parcels_l213_21314


namespace smallest_value_of_3a_plus_2_l213_21316

variable (a : ℝ)

theorem smallest_value_of_3a_plus_2 (h : 5 * a^2 + 7 * a + 2 = 1) : 3 * a + 2 = -1 :=
sorry

end smallest_value_of_3a_plus_2_l213_21316


namespace range_of_a_l213_21374

variable (a : ℝ)

def discriminant (a : ℝ) : ℝ := 4 * a ^ 2 - 12

theorem range_of_a
  (h : discriminant a > 0) :
  a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end range_of_a_l213_21374


namespace arithmetic_series_sum_l213_21338

theorem arithmetic_series_sum :
  let a1 : ℚ := 22
  let d : ℚ := 3 / 7
  let an : ℚ := 73
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  S = 5700 := by
  sorry

end arithmetic_series_sum_l213_21338


namespace julia_money_left_l213_21385

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l213_21385


namespace maximal_possible_degree_difference_l213_21334

theorem maximal_possible_degree_difference (n_vertices : ℕ) (n_edges : ℕ) (disjoint_edge_pairs : ℕ) 
    (h1 : n_vertices = 30) (h2 : n_edges = 105) (h3 : disjoint_edge_pairs = 4822) : 
    ∃ (max_diff : ℕ), max_diff = 22 :=
by
  sorry

end maximal_possible_degree_difference_l213_21334


namespace parabola_unique_solution_l213_21327

theorem parabola_unique_solution (b c : ℝ) :
  (∀ x y : ℝ, (x, y) = (-2, -8) ∨ (x, y) = (4, 28) ∨ (x, y) = (1, 4) →
    (y = x^2 + b * x + c)) →
  b = 4 ∧ c = -1 :=
by
  intro h
  have h₁ := h (-2) (-8) (Or.inl rfl)
  have h₂ := h 4 28 (Or.inr (Or.inl rfl))
  have h₃ := h 1 4 (Or.inr (Or.inr rfl))
  sorry

end parabola_unique_solution_l213_21327


namespace fraction_zero_solution_l213_21301

theorem fraction_zero_solution (x : ℝ) (h1 : x - 5 = 0) (h2 : 4 * x^2 - 1 ≠ 0) : x = 5 :=
by {
  sorry -- The proof
}

end fraction_zero_solution_l213_21301


namespace rectangle_fraction_l213_21344

noncomputable def side_of_square : ℝ := Real.sqrt 900
noncomputable def radius_of_circle : ℝ := side_of_square
noncomputable def area_of_rectangle : ℝ := 120
noncomputable def breadth_of_rectangle : ℝ := 10
noncomputable def length_of_rectangle : ℝ := area_of_rectangle / breadth_of_rectangle
noncomputable def fraction : ℝ := length_of_rectangle / radius_of_circle

theorem rectangle_fraction :
  (length_of_rectangle / radius_of_circle) = (2 / 5) :=
by
  sorry

end rectangle_fraction_l213_21344


namespace multiple_of_3_l213_21326

theorem multiple_of_3 (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 3 ∣ n :=
sorry

end multiple_of_3_l213_21326


namespace work_time_B_l213_21397

theorem work_time_B (A_efficiency : ℕ) (B_efficiency : ℕ) (days_together : ℕ) (total_work : ℕ) :
  (A_efficiency = 2 * B_efficiency) →
  (days_together = 5) →
  (total_work = (A_efficiency + B_efficiency) * days_together) →
  (total_work / B_efficiency = 15) :=
by
  intros
  sorry

end work_time_B_l213_21397


namespace find_x_l213_21378

theorem find_x (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 :=
by
  sorry

end find_x_l213_21378


namespace compute_value_l213_21367

open Nat Real

theorem compute_value (A B : ℝ × ℝ) (hA : A = (15, 10)) (hB : B = (-5, 6)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (x y : ℝ), C = (x, y) ∧ 2 * x - 4 * y = -22 := by
  sorry

end compute_value_l213_21367


namespace find_set_of_points_B_l213_21398

noncomputable def is_incenter (A B C I : Point) : Prop :=
  -- define the incenter condition
  sorry

noncomputable def angle_less_than (A B C : Point) (α : ℝ) : Prop :=
  -- define the condition that all angles of triangle ABC are less than α
  sorry

theorem find_set_of_points_B (A I : Point) (α : ℝ) (hα1 : 60 < α) (hα2 : α < 90) :
  ∃ B : Point, ∃ C : Point,
    is_incenter A B C I ∧ angle_less_than A B C α :=
by
  -- The proof will go here
  sorry

end find_set_of_points_B_l213_21398


namespace tan_45_deg_l213_21345

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l213_21345


namespace max_knights_is_seven_l213_21364

-- Definitions of conditions
def students : ℕ := 11
def total_statements : ℕ := students * (students - 1)
def liar_statements : ℕ := 56

-- Definition translating the problem statement
theorem max_knights_is_seven : ∃ (k li : ℕ), 
  (k + li = students) ∧ 
  (k * li = liar_statements) ∧ 
  (k = 7) := 
by
  sorry

end max_knights_is_seven_l213_21364


namespace crayons_per_box_l213_21321

theorem crayons_per_box (total_crayons : ℝ) (total_boxes : ℝ) (h1 : total_crayons = 7.0) (h2 : total_boxes = 1.4) : total_crayons / total_boxes = 5 :=
by
  sorry

end crayons_per_box_l213_21321


namespace solve_exponential_equation_l213_21357

theorem solve_exponential_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_exponential_equation_l213_21357


namespace gcd_153_119_eq_17_l213_21376

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_eq_17_l213_21376


namespace factorization_correctness_l213_21359

theorem factorization_correctness :
  (∀ x : ℝ, (x + 1) * (x - 1) = x^2 - 1 → false) ∧
  (∀ x : ℝ, x^2 - 4 * x + 4 = x * (x - 4) + 4 → false) ∧
  (∀ x : ℝ, (x + 3) * (x - 4) = x^2 - x - 12 → false) ∧
  (∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correctness_l213_21359


namespace product_of_points_l213_21336

def f (n : ℕ) : ℕ :=
  if n % 6 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls := [5, 6, 1, 2, 3]
def betty_rolls := [6, 1, 1, 2, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldl (fun acc n => acc + f n) 0

theorem product_of_points :
  total_points allie_rolls * total_points betty_rolls = 169 :=
by
  sorry

end product_of_points_l213_21336


namespace message_hours_needed_l213_21315

-- Define the sequence and the condition
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem message_hours_needed : ∃ n : ℕ, S n > 55 ∧ n = 5 := by
  sorry

end message_hours_needed_l213_21315


namespace tank_capacity_l213_21382

variable (C : ℝ)

theorem tank_capacity (h : (3/4) * C + 9 = (7/8) * C) : C = 72 :=
by
  sorry

end tank_capacity_l213_21382
