import Mathlib

namespace difference_in_perimeter_is_50_cm_l702_70210

-- Define the lengths of the four ribbons
def ribbon_lengths (x : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, x + 25, x + 50, x + 75)

-- Define the perimeter of the first shape
def perimeter_first_shape (x : ℕ) : ℕ :=
  2 * x + 230

-- Define the perimeter of the second shape
def perimeter_second_shape (x : ℕ) : ℕ :=
  2 * x + 280

-- Define the main theorem that the difference in perimeter is 50 cm
theorem difference_in_perimeter_is_50_cm (x : ℕ) :
  perimeter_second_shape x - perimeter_first_shape x = 50 := by
  sorry

end difference_in_perimeter_is_50_cm_l702_70210


namespace problem_statement_l702_70245

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) :=
  ∀ x y, x ≤ y → f x ≤ f y

noncomputable def isOddFunction (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_statement (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) (a3 : ℝ):
  isMonotonicIncreasing f →
  isOddFunction f →
  isArithmeticSeq a →
  a 3 = a3 →
  a3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  -- proof will go here
  sorry

end problem_statement_l702_70245


namespace value_of_expression_l702_70202

-- Definitions of the variables x and y along with their assigned values
def x : ℕ := 20
def y : ℕ := 8

-- The theorem that asserts the value of (x - y) * (x + y) equals 336
theorem value_of_expression : (x - y) * (x + y) = 336 := by 
  -- Skipping proof
  sorry

end value_of_expression_l702_70202


namespace square_root_value_l702_70226

-- Define the problem conditions
def x : ℝ := 5

-- Prove the solution
theorem square_root_value : (Real.sqrt (x - 3)) = Real.sqrt 2 :=
by
  -- Proof steps skipped
  sorry

end square_root_value_l702_70226


namespace first_diamond_second_spade_prob_l702_70218

/--
Given a standard deck of 52 cards, there are 13 cards of each suit.
What is the probability that the first card dealt is a diamond (♦) 
and the second card dealt is a spade (♠)?
-/
theorem first_diamond_second_spade_prob : 
  let total_cards := 52
  let diamonds := 13
  let spades := 13
  let first_diamond_prob := diamonds / total_cards
  let second_spade_prob_after_diamond := spades / (total_cards - 1)
  let combined_prob := first_diamond_prob * second_spade_prob_after_diamond
  combined_prob = 13 / 204 := 
by
  sorry

end first_diamond_second_spade_prob_l702_70218


namespace sweet_apples_percentage_is_75_l702_70290

noncomputable def percentage_sweet_apples 
  (price_sweet : ℝ) 
  (price_sour : ℝ) 
  (total_apples : ℕ) 
  (total_earnings : ℝ) 
  (percentage_sweet_expr : ℝ) :=
  price_sweet * percentage_sweet_expr + price_sour * (total_apples - percentage_sweet_expr) = total_earnings

theorem sweet_apples_percentage_is_75 :
  percentage_sweet_apples 0.5 0.1 100 40 75 :=
by
  unfold percentage_sweet_apples
  sorry

end sweet_apples_percentage_is_75_l702_70290


namespace conquering_Loulan_necessary_for_returning_home_l702_70250

theorem conquering_Loulan_necessary_for_returning_home : 
  ∀ (P Q : Prop), (¬ Q → ¬ P) → (P → Q) :=
by sorry

end conquering_Loulan_necessary_for_returning_home_l702_70250


namespace algorithm_outputs_min_value_l702_70275

theorem algorithm_outputs_min_value (a b c d : ℕ) :
  let m := a;
  let m := if b < m then b else m;
  let m := if c < m then c else m;
  let m := if d < m then d else m;
  m = min (min (min a b) c) d :=
by
  sorry

end algorithm_outputs_min_value_l702_70275


namespace sum_geometric_series_l702_70276

noncomputable def S (r : ℝ) : ℝ :=
  12 / (1 - r)

theorem sum_geometric_series (a : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : S a * S (-a) = 2016) :
  S a + S (-a) = 336 :=
by
  sorry

end sum_geometric_series_l702_70276


namespace original_class_strength_l702_70240

variable (x : ℕ)

/-- The average age of an adult class is 40 years.
  18 new students with an average age of 32 years join the class, 
  therefore decreasing the average by 4 years.
  Find the original strength of the class.
-/
theorem original_class_strength (h1 : 40 * x + 18 * 32 = (x + 18) * 36) : x = 18 := 
by sorry

end original_class_strength_l702_70240


namespace xiaoming_xiaoqiang_common_visit_l702_70219

-- Define the initial visit dates and subsequent visit intervals
def xiaoming_initial_visit : ℕ := 3 -- The first Wednesday of January
def xiaoming_interval : ℕ := 4

def xiaoqiang_initial_visit : ℕ := 4 -- The first Thursday of January
def xiaoqiang_interval : ℕ := 3

-- Prove that the only common visit date is January 7
theorem xiaoming_xiaoqiang_common_visit : 
  ∃! d, (d < 32) ∧ ∃ n m, d = xiaoming_initial_visit + n * xiaoming_interval ∧ d = xiaoqiang_initial_visit + m * xiaoqiang_interval :=
  sorry

end xiaoming_xiaoqiang_common_visit_l702_70219


namespace factorize_poly1_factorize_poly2_l702_70277

-- Define y substitution for first problem
def poly1_y := fun (x : ℝ) => x^2 + 2*x
-- Define y substitution for second problem
def poly2_y := fun (x : ℝ) => x^2 - 4*x

-- Define the given polynomial expressions 
def poly1 := fun (x : ℝ) => (x^2 + 2*x)*(x^2 + 2*x + 2) + 1
def poly2 := fun (x : ℝ) => (x^2 - 4*x)*(x^2 - 4*x + 8) + 16

theorem factorize_poly1 (x : ℝ) : poly1 x = (x + 1) ^ 4 := sorry

theorem factorize_poly2 (x : ℝ) : poly2 x = (x - 2) ^ 4 := sorry

end factorize_poly1_factorize_poly2_l702_70277


namespace third_side_length_l702_70249

theorem third_side_length (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < x) (h₄ : x < 11) : x = 6 :=
sorry

end third_side_length_l702_70249


namespace distinct_factors_count_l702_70291

-- Given conditions
def eight_squared : ℕ := 8^2
def nine_cubed : ℕ := 9^3
def seven_fifth : ℕ := 7^5
def number : ℕ := eight_squared * nine_cubed * seven_fifth

-- Proving the number of natural-number factors of the given number
theorem distinct_factors_count : 
  (number.factors.count 1 = 294) := sorry

end distinct_factors_count_l702_70291


namespace cos_of_angle_through_point_l702_70209

-- Define the point P and the angle α
def P : ℝ × ℝ := (4, 3)
def α : ℝ := sorry  -- α is an angle such that its terminal side passes through P

-- Define the squared distance from the origin to the point P
noncomputable def distance_squared : ℝ := P.1^2 + P.2^2

-- Define cos α
noncomputable def cosα : ℝ := P.1 / (Real.sqrt distance_squared)

-- State the theorem
theorem cos_of_angle_through_point : cosα = 4 / 5 := 
by sorry

end cos_of_angle_through_point_l702_70209


namespace coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l702_70258

theorem coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45 :
  let general_term (r : ℕ) := (Nat.choose 10 r) * (x^(10 - 3 * r)/2)
  ∃ r : ℕ, (general_term r) = 2 ∧ (Nat.choose 10 r) = 45 :=
by
  sorry

end coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l702_70258


namespace ratio_H_G_l702_70244

theorem ratio_H_G (G H : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (G / (x + 3) + H / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x^3 + x^2 - 15 * x))) :
    H / G = 64 :=
sorry

end ratio_H_G_l702_70244


namespace number_of_remaining_grandchildren_l702_70233

-- Defining the given values and conditions
def total_amount : ℕ := 124600
def half_amount : ℕ := total_amount / 2
def amount_per_remaining_grandchild : ℕ := 6230

-- Defining the goal to prove the number of remaining grandchildren
theorem number_of_remaining_grandchildren : (half_amount / amount_per_remaining_grandchild) = 10 := by
  sorry

end number_of_remaining_grandchildren_l702_70233


namespace two_lines_perpendicular_to_same_plane_are_parallel_l702_70238

variables {Plane Line : Type} 
variables (perp : Line → Plane → Prop) (parallel : Line → Line → Prop)

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane) (ha : perp a α) (hb : perp b α) : parallel a b :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l702_70238


namespace base8_subtraction_correct_l702_70261

theorem base8_subtraction_correct : (453 - 326 : ℕ) = 125 :=
by sorry

end base8_subtraction_correct_l702_70261


namespace find_m_l702_70295

def vector_collinear {α : Type*} [Field α] (a b : α × α) : Prop :=
  ∃ k : α, b = (k * (a.1), k * (a.2))

theorem find_m (m : ℝ) : 
  let a := (2, 3)
  let b := (-1, 2)
  vector_collinear (2 * m - 4, 3 * m + 8) (4, -1) → m = -2 :=
by
  intros
  sorry

end find_m_l702_70295


namespace range_of_a_l702_70217

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x - (Real.cos x)^2 ≤ 3) : -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l702_70217


namespace find_m_if_polynomial_is_perfect_square_l702_70284

theorem find_m_if_polynomial_is_perfect_square (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = x^2 + m * x + 4) → (m = 4 ∨ m = -4) :=
sorry

end find_m_if_polynomial_is_perfect_square_l702_70284


namespace son_present_age_l702_70293

variable (S F : ℕ)

-- Given conditions
def father_age := F = S + 34
def future_age_rel := F + 2 = 2 * (S + 2)

-- Theorem to prove the son's current age
theorem son_present_age (h₁ : father_age S F) (h₂ : future_age_rel S F) : S = 32 := by
  sorry

end son_present_age_l702_70293


namespace minimum_value_expression_l702_70248

theorem minimum_value_expression (F M N : ℝ × ℝ) (x y : ℝ) (a : ℝ) (k : ℝ) :
  (y ^ 2 = 16 * x ∧ F = (4, 0) ∧ l = (k * (x - 4), y) ∧ (M = (x₁, y₁) ∧ N = (x₂, y₂)) ∧
  0 ≤ x₁ ∧ y₁ ^ 2 = 16 * x₁ ∧ 0 ≤ x₂ ∧ y₂ ^ 2 = 16 * x₂) →
  (abs (dist F N) / 9 - 4 / abs (dist F M) ≥ 1 / 3) :=
sorry -- proof will be provided

end minimum_value_expression_l702_70248


namespace min_f_l702_70246

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x then (x + 1) * Real.log x
else 2 * x + 3

noncomputable def f' (x : ℝ) : ℝ :=
if 0 < x then Real.log x + (x + 1) / x
else 2

theorem min_f'_for_x_pos : ∃ (c : ℝ), c = 2 ∧ ∀ x > 0, f' x ≥ c := 
  sorry

end min_f_l702_70246


namespace packs_needed_is_six_l702_70299

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l702_70299


namespace actual_average_height_l702_70241

theorem actual_average_height (average_height : ℝ) (num_students : ℕ)
  (incorrect_heights actual_heights : Fin 3 → ℝ)
  (h_avg : average_height = 165)
  (h_num : num_students = 50)
  (h_incorrect : incorrect_heights 0 = 150 ∧ incorrect_heights 1 = 175 ∧ incorrect_heights 2 = 190)
  (h_actual : actual_heights 0 = 135 ∧ actual_heights 1 = 170 ∧ actual_heights 2 = 185) :
  (average_height * num_students 
   - (incorrect_heights 0 + incorrect_heights 1 + incorrect_heights 2) 
   + (actual_heights 0 + actual_heights 1 + actual_heights 2))
   / num_students = 164.5 :=
by
  -- proof steps here
  sorry

end actual_average_height_l702_70241


namespace max_height_reached_l702_70201

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 120 * t + 36

theorem max_height_reached :
  ∃ t : ℝ, h t = 216 ∧ t = 3 :=
sorry

end max_height_reached_l702_70201


namespace part1_l702_70283

theorem part1 (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) : 2 * x^2 + y^2 > x^2 + x * y := 
sorry

end part1_l702_70283


namespace inconsistent_equation_system_l702_70223

variables {a x c : ℝ}

theorem inconsistent_equation_system (h1 : (a + x) / 2 = 110) (h2 : (x + c) / 2 = 170) (h3 : a - c = 120) : false :=
by
  sorry

end inconsistent_equation_system_l702_70223


namespace guilty_prob_l702_70215

-- Defining suspects
inductive Suspect
| A
| B
| C

open Suspect

-- Constants for the problem
def looks_alike (x y : Suspect) : Prop :=
(x = A ∧ y = B) ∨ (x = B ∧ y = A)

def timid (x : Suspect) : Prop :=
x = A ∨ x = B

def bold (x : Suspect) : Prop :=
x = C

def alibi_dover (x : Suspect) : Prop :=
x = A ∨ x = B

def needs_accomplice (x : Suspect) : Prop :=
timid x

def works_alone (x : Suspect) : Prop :=
bold x

def in_bar_during_robbery (x : Suspect) : Prop :=
x = A ∨ x = B

-- Theorem to be proved
theorem guilty_prob :
  ∃ x : Suspect, (x = B) ∧ ∀ y : Suspect, y ≠ B → 
    ((y = A ∧ timid y ∧ needs_accomplice y ∧ in_bar_during_robbery y) ∨
    (y = C ∧ bold y ∧ works_alone y)) :=
by
  sorry

end guilty_prob_l702_70215


namespace correct_choice_is_C_l702_70252

def first_quadrant_positive_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def right_angle_is_axial (θ : ℝ) : Prop :=
  θ = 90

def obtuse_angle_second_quadrant (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def terminal_side_initial_side_same (θ : ℝ) : Prop :=
  θ = 0 ∨ θ = 360

theorem correct_choice_is_C : obtuse_angle_second_quadrant 120 :=
by
  sorry

end correct_choice_is_C_l702_70252


namespace cost_price_600_l702_70270

variable (CP SP : ℝ)

theorem cost_price_600 
  (h1 : SP = 1.08 * CP) 
  (h2 : SP = 648) : 
  CP = 600 := 
by
  sorry

end cost_price_600_l702_70270


namespace rectangle_area_l702_70287

theorem rectangle_area (l w : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 120) : l * w = 800 :=
by
  -- proof to be filled in
  sorry

end rectangle_area_l702_70287


namespace people_who_cannot_do_either_l702_70216

def people_total : ℕ := 120
def can_dance : ℕ := 88
def can_write_calligraphy : ℕ := 32
def can_do_both : ℕ := 18

theorem people_who_cannot_do_either : 
  people_total - (can_dance + can_write_calligraphy - can_do_both) = 18 := 
by
  sorry

end people_who_cannot_do_either_l702_70216


namespace hyperbola_standard_equation_l702_70236

def ellipse_equation (x y : ℝ) : Prop :=
  (y^2) / 16 + (x^2) / 12 = 1

def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2) / 2 - (x^2) / 2 = 1

def passes_through_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3

theorem hyperbola_standard_equation (x y : ℝ) (hx : passes_through_point x y)
  (ellipse_foci_shared : ∀ x y : ℝ, ellipse_equation x y → ellipse_equation x y)
  : hyperbola_equation x y := 
sorry

end hyperbola_standard_equation_l702_70236


namespace sum_of_fractions_equals_l702_70228

theorem sum_of_fractions_equals :
  (1 / 15 + 2 / 25 + 3 / 35 + 4 / 45 : ℚ) = 0.32127 :=
  sorry

end sum_of_fractions_equals_l702_70228


namespace chairs_per_row_l702_70285

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) 
  (h_total_chairs : total_chairs = 432) (h_num_rows : num_rows = 27) : 
  total_chairs / num_rows = 16 :=
by
  sorry

end chairs_per_row_l702_70285


namespace correct_log_values_l702_70234

theorem correct_log_values (a b c : ℝ)
                          (log_027 : ℝ) (log_21 : ℝ) (log_1_5 : ℝ) (log_2_8 : ℝ)
                          (log_3 : ℝ) (log_5 : ℝ) (log_6 : ℝ) (log_7 : ℝ)
                          (log_8 : ℝ) (log_9 : ℝ) (log_14 : ℝ) :
  (log_3 = 2 * a - b) →
  (log_5 = a + c) →
  (log_6 = 1 + a - b - c) →
  (log_7 = 2 * (b + c)) →
  (log_9 = 4 * a - 2 * b) →
  (log_1_5 = 3 * a - b + c) →
  (log_14 = 1 - c + 2 * b) →
  (log_1_5 = 3 * a - b + c - 1) ∧ (log_7 = 2 * b + c) := sorry

end correct_log_values_l702_70234


namespace triple_overlap_area_correct_l702_70232

-- Define the dimensions of the auditorium and carpets
def auditorium_dim : ℕ × ℕ := (10, 10)
def carpet1_dim : ℕ × ℕ := (6, 8)
def carpet2_dim : ℕ × ℕ := (6, 6)
def carpet3_dim : ℕ × ℕ := (5, 7)

-- The coordinates and dimensions of the overlap regions are derived based on the given positions
-- Here we assume derivations as described in the solution steps without recalculating them

-- Overlap area of the second and third carpets
def overlap23 : ℕ × ℕ := (5, 3)

-- Intersection of this overlap with the first carpet
def overlap_all : ℕ × ℕ := (2, 3)

-- Calculate the area of the region where all three carpets overlap
def triple_overlap_area : ℕ :=
  (overlap_all.1 * overlap_all.2)

theorem triple_overlap_area_correct :
  triple_overlap_area = 6 := by
  -- Expected result should be 6 square meters
  sorry

end triple_overlap_area_correct_l702_70232


namespace distance_between_parallel_lines_l702_70269

theorem distance_between_parallel_lines : 
  ∀ (x y : ℝ), 
  (3 * x - 4 * y - 3 = 0) ∧ (6 * x - 8 * y + 5 = 0) → 
  ∃ d : ℝ, d = 11 / 10 :=
by
  sorry

end distance_between_parallel_lines_l702_70269


namespace solve_quadratic_l702_70213

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l702_70213


namespace range_of_a_l702_70274

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, x^2 - a * x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
by sorry

end range_of_a_l702_70274


namespace market_value_of_10_percent_yielding_8_percent_stock_l702_70268

/-- 
Given:
1. The stock yields 8%.
2. It is a 10% stock, meaning the annual dividend per share is 10% of the face value.
3. Assume the face value of the stock is $100.

Prove:
The market value of the stock is $125.
-/
theorem market_value_of_10_percent_yielding_8_percent_stock
    (annual_dividend_per_share : ℝ)
    (face_value : ℝ)
    (dividend_yield : ℝ)
    (market_value_per_share : ℝ) 
    (h1 : face_value = 100)
    (h2 : annual_dividend_per_share = 0.10 * face_value)
    (h3 : dividend_yield = 8) :
    market_value_per_share = 125 := 
by
  /-
  Here, the following conditions are already given:
  1. face_value = 100
  2. annual_dividend_per_share = 0.10 * 100 = 10
  3. dividend_yield = 8
  
  We need to prove: market_value_per_share = 125
  -/
  sorry

end market_value_of_10_percent_yielding_8_percent_stock_l702_70268


namespace combined_area_of_three_walls_l702_70222

theorem combined_area_of_three_walls (A : ℝ) :
  (A - 2 * 30 - 3 * 45 = 180) → (A = 375) :=
by
  intro h
  sorry

end combined_area_of_three_walls_l702_70222


namespace sale_in_second_month_l702_70243

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (total_months : ℕ)
  (h_sale1 : sale1 = 5420)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500)
  (h_sale6 : sale6 = 6470)
  (h_average_sale : average_sale = 6100)
  (h_total_months : total_months = 6) :
  ∃ sale2 : ℕ, sale2 = 5660 := 
by
  sorry

end sale_in_second_month_l702_70243


namespace find_k_value_l702_70206

theorem find_k_value (k : ℝ) (h₁ : ∀ x, k * x^2 - 5 * x - 12 = 0 → (x = 3 ∨ x = -4 / 3)) : k = 3 :=
sorry

end find_k_value_l702_70206


namespace greatest_possible_perimeter_l702_70220

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l702_70220


namespace initial_files_count_l702_70254

theorem initial_files_count (deleted_files folders files_per_folder total_files initial_files : ℕ)
    (h1 : deleted_files = 21)
    (h2 : folders = 9)
    (h3 : files_per_folder = 8)
    (h4 : total_files = folders * files_per_folder)
    (h5 : initial_files = total_files + deleted_files) :
    initial_files = 93 :=
by
  sorry

end initial_files_count_l702_70254


namespace danny_marks_in_math_l702_70229

theorem danny_marks_in_math
  (english_marks : ℕ := 76)
  (physics_marks : ℕ := 82)
  (chemistry_marks : ℕ := 67)
  (biology_marks : ℕ := 75)
  (average_marks : ℕ := 73)
  (num_subjects : ℕ := 5) :
  ∃ (math_marks : ℕ), math_marks = 65 :=
by
  let total_marks := average_marks * num_subjects
  let other_subjects_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  have math_marks := total_marks - other_subjects_marks
  use math_marks
  sorry

end danny_marks_in_math_l702_70229


namespace car_price_difference_l702_70282

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end car_price_difference_l702_70282


namespace k_range_l702_70267

noncomputable def range_of_k (k : ℝ): Prop :=
  ∀ x : ℤ, (x - 2) * (x + 1) > 0 ∧ (2 * x + 5) * (x + k) < 0 → x = -2

theorem k_range:
  (∃ k : ℝ, range_of_k k) ↔ -3 ≤ k ∧ k < 2 :=
by
  sorry

end k_range_l702_70267


namespace teammates_score_is_correct_l702_70292

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end teammates_score_is_correct_l702_70292


namespace plates_difference_l702_70266

noncomputable def num_pots_angela : ℕ := 20
noncomputable def num_plates_angela (P : ℕ) := P
noncomputable def num_cutlery_angela (P : ℕ) := P / 2
noncomputable def num_pots_sharon : ℕ := 10
noncomputable def num_plates_sharon (P : ℕ) := 3 * P - 20
noncomputable def num_cutlery_sharon (P : ℕ) := P
noncomputable def total_kitchen_supplies_sharon (P : ℕ) := 
  num_pots_sharon + num_plates_sharon P + num_cutlery_sharon P

theorem plates_difference (P : ℕ) 
  (hP: num_plates_angela P > 3 * num_pots_angela) 
  (h_supplies: total_kitchen_supplies_sharon P = 254) :
  P - 3 * num_pots_angela = 6 := 
sorry

end plates_difference_l702_70266


namespace number_of_triangles_l702_70211

/-!
# Problem Statement
Given a square with 20 interior points connected such that the lines do not intersect and divide the square into triangles,
prove that the number of triangles formed is 42.
-/

theorem number_of_triangles (V E F : ℕ) (hV : V = 24) (hE : E = (3 * F + 1) / 2) (hF : V - E + F = 2) :
  (F - 1) = 42 :=
by
  sorry

end number_of_triangles_l702_70211


namespace sum_arith_seq_l702_70255

theorem sum_arith_seq (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h₁ : ∀ n, S n = n * a 1 + (n * (n - 1)) * d / 2)
    (h₂ : S 10 = S 20)
    (h₃ : d > 0) :
    a 10 + a 22 > 0 := 
sorry

end sum_arith_seq_l702_70255


namespace primes_p_p2_p4_l702_70204

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem primes_p_p2_p4 (p : ℕ) (hp : is_prime p) (hp2 : is_prime (p + 2)) (hp4 : is_prime (p + 4)) :
  p = 3 :=
sorry

end primes_p_p2_p4_l702_70204


namespace xy_value_l702_70286

theorem xy_value (x y : ℝ) (h1 : (x + y) / 3 = 1.222222222222222) : x + y = 3.666666666666666 :=
by
  sorry

end xy_value_l702_70286


namespace max_k_value_l702_70264

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k_value
    (h : ∀ (x : ℝ), 1 < x → f x > k * (x - 1)) :
    k = 3 := sorry

end max_k_value_l702_70264


namespace k_value_five_l702_70247

theorem k_value_five (a b k : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a^2 + b^2) / (a * b - 1) = k) : k = 5 := 
sorry

end k_value_five_l702_70247


namespace total_interest_is_68_l702_70298

-- Definitions of the initial conditions
def amount_2_percent : ℝ := 600
def amount_4_percent : ℝ := amount_2_percent + 800
def interest_rate_2_percent : ℝ := 0.02
def interest_rate_4_percent : ℝ := 0.04
def invested_total_1 : ℝ := amount_2_percent
def invested_total_2 : ℝ := amount_4_percent

-- The total interest calculation
def interest_2_percent : ℝ := invested_total_1 * interest_rate_2_percent
def interest_4_percent : ℝ := invested_total_2 * interest_rate_4_percent

-- Claim: The total interest earned is $68
theorem total_interest_is_68 : interest_2_percent + interest_4_percent = 68 := by
  sorry

end total_interest_is_68_l702_70298


namespace solve_for_x_l702_70237

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l702_70237


namespace first_sequence_correct_second_sequence_correct_l702_70259

theorem first_sequence_correct (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 = 12) (h2 : a2 = a1 + 4) (h3 : a3 = a2 + 4) (h4 : a4 = a3 + 4) (h5 : a5 = a4 + 4) :
  a4 = 24 ∧ a5 = 28 :=
by sorry

theorem second_sequence_correct (b1 b2 b3 b4 b5 : ℕ) (h1 : b1 = 2) (h2 : b2 = b1 * 2) (h3 : b3 = b2 * 2) (h4 : b4 = b3 * 2) (h5 : b5 = b4 * 2) :
  b4 = 16 ∧ b5 = 32 :=
by sorry

end first_sequence_correct_second_sequence_correct_l702_70259


namespace num_five_letter_words_correct_l702_70265

-- Define the number of letters in the alphabet
def num_letters : ℕ := 26

-- Define the number of vowels
def num_vowels : ℕ := 5

-- Define a function that calculates the number of valid five-letter words
def num_five_letter_words : ℕ :=
  num_letters * num_vowels * num_letters * num_letters

-- The theorem statement we need to prove
theorem num_five_letter_words_correct : num_five_letter_words = 87700 :=
by
  -- The proof is omitted; it should equate the calculated value to 87700
  sorry

end num_five_letter_words_correct_l702_70265


namespace sammy_offer_l702_70214

-- Declaring the given constants and assumptions
def peggy_records : ℕ := 200
def bryan_interested_records : ℕ := 100
def bryan_uninterested_records : ℕ := 100
def bryan_interested_offer : ℕ := 6
def bryan_uninterested_offer : ℕ := 1
def sammy_offer_diff : ℕ := 100

-- The problem to be proved
theorem sammy_offer:
    ∃ S : ℝ, 
    (200 * S) - 
    (bryan_interested_records * bryan_interested_offer +
    bryan_uninterested_records * bryan_uninterested_offer) = sammy_offer_diff → 
    S = 4 :=
sorry

end sammy_offer_l702_70214


namespace kelseys_sister_is_3_years_older_l702_70263

-- Define the necessary conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := 2021 - 50
def age_difference (a b : ℕ) : ℕ := a - b

-- State the theorem to prove
theorem kelseys_sister_is_3_years_older :
  age_difference kelsey_birth_year sister_birth_year = 3 :=
by
  -- Skipping the proof steps as only the statement is needed
  sorry

end kelseys_sister_is_3_years_older_l702_70263


namespace intersection_A_B_l702_70231

-- Define sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { 2, 3, 4, 5 }

-- State the theorem about the intersection A ∩ B
theorem intersection_A_B : A ∩ B = { 2, 3 } :=
by
  sorry

end intersection_A_B_l702_70231


namespace problem_statement_l702_70208

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem problem_statement : ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 := by
  sorry

end problem_statement_l702_70208


namespace positive_n_of_single_solution_l702_70279

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l702_70279


namespace passenger_gets_ticket_l702_70271

variables (p1 p2 p3 p4 p5 p6 : ℝ)

-- Conditions:
axiom h_sum_eq_one : p1 + p2 + p3 = 1
axiom h_p1_nonneg : 0 ≤ p1
axiom h_p2_nonneg : 0 ≤ p2
axiom h_p3_nonneg : 0 ≤ p3
axiom h_p4_nonneg : 0 ≤ p4
axiom h_p4_le_one : p4 ≤ 1
axiom h_p5_nonneg : 0 ≤ p5
axiom h_p5_le_one : p5 ≤ 1
axiom h_p6_nonneg : 0 ≤ p6
axiom h_p6_le_one : p6 ≤ 1

-- Theorem:
theorem passenger_gets_ticket :
  (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) = (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) :=
by sorry

end passenger_gets_ticket_l702_70271


namespace expected_value_coin_flip_l702_70257

def probability_heads : ℚ := 2 / 3
def probability_tails : ℚ := 1 / 3
def gain_heads : ℤ := 5
def loss_tails : ℤ := -9

theorem expected_value_coin_flip : (2 / 3 : ℚ) * 5 + (1 / 3 : ℚ) * (-9) = 1 / 3 :=
by sorry

end expected_value_coin_flip_l702_70257


namespace division_proof_l702_70296

def dividend : ℕ := 144
def inner_divisor_num : ℕ := 12
def inner_divisor_denom : ℕ := 2
def final_divisor : ℕ := inner_divisor_num / inner_divisor_denom
def expected_result : ℕ := 24

theorem division_proof : (dividend / final_divisor) = expected_result := by
  sorry

end division_proof_l702_70296


namespace product_of_possible_x_l702_70242

theorem product_of_possible_x : 
  (∀ x : ℚ, abs ((18 / x) + 4) = 3 → x = -18 ∨ x = -18 / 7) → 
  ((-18) * (-18 / 7) = 324 / 7) :=
by
  sorry

end product_of_possible_x_l702_70242


namespace largest_abs_val_among_2_3_neg3_neg4_l702_70262

def abs_val (a : Int) : Nat := a.natAbs

theorem largest_abs_val_among_2_3_neg3_neg4 : 
  ∀ (x : Int), x ∈ [2, 3, -3, -4] → abs_val x ≤ abs_val (-4) := by
  sorry

end largest_abs_val_among_2_3_neg3_neg4_l702_70262


namespace option_C_incorrect_l702_70297

structure Line := (point1 point2 : ℝ × ℝ × ℝ)
structure Plane := (point : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ)

variables (m n : Line) (α β : Plane)

def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def lines_parallel (l1 l2 : Line) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry
def planes_parallel (p1 p2 : Plane) : Prop := sorry

theorem option_C_incorrect 
  (h1 : line_in_plane m α)
  (h2 : line_parallel_to_plane n α)
  (h3 : lines_parallel m n) :
  false :=
sorry

end option_C_incorrect_l702_70297


namespace distance_between_red_lights_l702_70227

def position_of_nth_red (n : ℕ) : ℕ :=
  7 * (n - 1) / 3 + n

def in_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_red_lights :
  in_feet ((position_of_nth_red 30 - position_of_nth_red 5) * 8) = 41 :=
by
  sorry

end distance_between_red_lights_l702_70227


namespace maximum_value_of_func_l702_70294

noncomputable def func (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

def domain_x (x : ℝ) : Prop := (1/3 : ℝ) ≤ x ∧ x ≤ (2/5 : ℝ)
def domain_y (y : ℝ) : Prop := (1/2 : ℝ) ≤ y ∧ y ≤ (5/8 : ℝ)

theorem maximum_value_of_func :
  ∀ (x y : ℝ), domain_x x → domain_y y → func x y ≤ (20 / 21 : ℝ) ∧ 
  (∃ (x y : ℝ), domain_x x ∧ domain_y y ∧ func x y = (20 / 21 : ℝ)) :=
by sorry

end maximum_value_of_func_l702_70294


namespace Terrence_earns_l702_70273

theorem Terrence_earns :
  ∀ (J T E : ℝ), J + T + E = 90 ∧ J = T + 5 ∧ E = 25 → T = 30 :=
by
  intro J T E
  intro h
  obtain ⟨h₁, h₂, h₃⟩ := h
  sorry -- proof steps go here

end Terrence_earns_l702_70273


namespace slope_of_line_through_focus_of_parabola_l702_70212

theorem slope_of_line_through_focus_of_parabola
  (C : (x y : ℝ) → y^2 = 4 * x)
  (F : (ℝ × ℝ) := (1, 0))
  (A B : (ℝ × ℝ))
  (l : ℝ → ℝ)
  (intersects : (x : ℝ) → (l x) ^ 2 = 4 * x)
  (passes_through_focus : l 1 = 0)
  (distance_condition : ∀ (d1 d2 : ℝ), d1 = 4 * d2 → dist F A = d1 ∧ dist F B = d2) :
  ∃ k : ℝ, (∀ (x : ℝ), l x = k * (x - 1)) ∧ (k = 4 / 3 ∨ k = -4 / 3) :=
by
  sorry

end slope_of_line_through_focus_of_parabola_l702_70212


namespace man_speed_still_water_l702_70288

noncomputable def speed_in_still_water (U D : ℝ) : ℝ := (U + D) / 2

theorem man_speed_still_water :
  let U := 45
  let D := 55
  speed_in_still_water U D = 50 := by
  sorry

end man_speed_still_water_l702_70288


namespace tessa_still_owes_greg_l702_70225

def initial_debt : ℝ := 40
def first_repayment : ℝ := 0.25 * initial_debt
def debt_after_first_repayment : ℝ := initial_debt - first_repayment
def second_borrowing : ℝ := 25
def debt_after_second_borrowing : ℝ := debt_after_first_repayment + second_borrowing
def second_repayment : ℝ := 0.5 * debt_after_second_borrowing
def debt_after_second_repayment : ℝ := debt_after_second_borrowing - second_repayment
def third_borrowing : ℝ := 30
def debt_after_third_borrowing : ℝ := debt_after_second_repayment + third_borrowing
def third_repayment : ℝ := 0.1 * debt_after_third_borrowing
def final_debt : ℝ := debt_after_third_borrowing - third_repayment

theorem tessa_still_owes_greg : final_debt = 51.75 := by
  sorry

end tessa_still_owes_greg_l702_70225


namespace cyclic_inequality_l702_70281

variables {a b c : ℝ}

theorem cyclic_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (ab / (a + b + 2 * c) + bc / (b + c + 2 * a) + ca / (c + a + 2 * b)) ≤ (a + b + c) / 4 :=
sorry

end cyclic_inequality_l702_70281


namespace polynomial_m_n_values_l702_70239

theorem polynomial_m_n_values :
  ∀ (m n : ℝ), ((x - 1) * (x + m) = x^2 - n * x - 6) → (m = 6 ∧ n = -5) := 
by
  intros m n h
  sorry

end polynomial_m_n_values_l702_70239


namespace influenza_probability_l702_70235

theorem influenza_probability :
  let flu_rate_A := 0.06
  let flu_rate_B := 0.05
  let flu_rate_C := 0.04
  let population_ratio_A := 6
  let population_ratio_B := 5
  let population_ratio_C := 4
  (population_ratio_A * flu_rate_A + population_ratio_B * flu_rate_B + population_ratio_C * flu_rate_C) / 
  (population_ratio_A + population_ratio_B + population_ratio_C) = 77 / 1500 :=
by
  sorry

end influenza_probability_l702_70235


namespace roots_theorem_l702_70289

-- Definitions and Conditions
def root1 (a b p : ℝ) : Prop := 
  a + b = -p ∧ a * b = 1

def root2 (b c q : ℝ) : Prop := 
  b + c = -q ∧ b * c = 2

-- The theorem to prove
theorem roots_theorem (a b c p q : ℝ) (h1 : root1 a b p) (h2 : root2 b c q) : 
  (b - a) * (b - c) = p * q - 6 :=
sorry

end roots_theorem_l702_70289


namespace max_MB_value_l702_70260

open Real

-- Define the conditions of the problem
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : sqrt 6 / 3 = sqrt (1 - b^2 / a^2))

-- Define the point M and the vertex B on the ellipse
variables (M : ℝ × ℝ) (hM : (M.1)^2 / (a)^2 + (M.2)^2 / (b)^2 = 1)
def B : ℝ × ℝ := (0, -b)

-- The task is to prove the maximum value of |MB| given the conditions
theorem max_MB_value : ∃ (maxMB : ℝ), maxMB = (3 * sqrt 2 / 2) * b :=
sorry

end max_MB_value_l702_70260


namespace maximum_xyz_l702_70207

theorem maximum_xyz {x y z : ℝ} (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) 
  (h : (x * y) + z = (x + z) * (y + z)) : xyz ≤ (1 / 27) :=
by
  sorry

end maximum_xyz_l702_70207


namespace min_value_abs_diff_l702_70278

-- Definitions of conditions
def is_in_interval (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 4

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  (b^2 - a^2 = 2) ∧ (c^2 - b^2 = 2)

-- Main statement
theorem min_value_abs_diff (x y z : ℝ)
  (h1 : is_in_interval x)
  (h2 : is_in_interval y)
  (h3 : is_in_interval z)
  (h4 : is_arithmetic_progression x y z) :
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_value_abs_diff_l702_70278


namespace find_absolute_difference_l702_70221

def condition_avg_sum (m n : ℝ) : Prop :=
  m + n + 5 + 6 + 4 = 25

def condition_variance (m n : ℝ) : Prop :=
  (m - 5) ^ 2 + (n - 5) ^ 2 = 8

theorem find_absolute_difference (m n : ℝ) (h1 : condition_avg_sum m n) (h2 : condition_variance m n) : |m - n| = 4 :=
sorry

end find_absolute_difference_l702_70221


namespace num_blue_balls_l702_70251

theorem num_blue_balls (total_balls blue_balls : ℕ) 
  (prob_all_blue : ℚ)
  (h_total : total_balls = 12)
  (h_prob : prob_all_blue = 1 / 55)
  (h_prob_eq : (blue_balls / 12) * ((blue_balls - 1) / 11) * ((blue_balls - 2) / 10) = prob_all_blue) :
  blue_balls = 4 :=
by
  -- Placeholder for proof
  sorry

end num_blue_balls_l702_70251


namespace garden_roller_length_l702_70205

noncomputable def length_of_garden_roller (d : ℝ) (A : ℝ) (revolutions : ℕ) (π : ℝ) : ℝ :=
  let r := d / 2
  let area_in_one_revolution := A / revolutions
  let L := area_in_one_revolution / (2 * π * r)
  L

theorem garden_roller_length :
  length_of_garden_roller 1.2 37.714285714285715 5 (22 / 7) = 2 := by
  sorry

end garden_roller_length_l702_70205


namespace simplify_expression_l702_70272

variable (x y : ℝ)

theorem simplify_expression (h : x ≠ y ∧ x ≠ -y) : 
  ((1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x) :=
by sorry

end simplify_expression_l702_70272


namespace tagged_fish_in_second_catch_l702_70256

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ := 3200) 
  (initial_tagged : ℕ := 80) 
  (second_catch : ℕ := 80) 
  (T : ℕ) 
  (h : (T : ℚ) / second_catch = initial_tagged / total_fish) :
  T = 2 :=
by 
  sorry

end tagged_fish_in_second_catch_l702_70256


namespace negation_of_at_most_one_obtuse_l702_70253

-- Defining a predicate to express the concept of an obtuse angle
def is_obtuse (θ : ℝ) : Prop := θ > 90

-- Defining a triangle with three interior angles α, β, and γ
structure Triangle :=
  (α β γ : ℝ)
  (sum_angles : α + β + γ = 180)

-- Defining the condition that "At most, only one interior angle of a triangle is obtuse"
def at_most_one_obtuse (T : Triangle) : Prop :=
  (is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ)

-- The theorem we want to prove: Negation of "At most one obtuse angle" is "At least two obtuse angles"
theorem negation_of_at_most_one_obtuse (T : Triangle) :
  ¬ at_most_one_obtuse T ↔ (is_obtuse T.α ∧ is_obtuse T.β) ∨ (is_obtuse T.α ∧ is_obtuse T.γ) ∨ (is_obtuse T.β ∧ is_obtuse T.γ) := by
  sorry

end negation_of_at_most_one_obtuse_l702_70253


namespace y1_lt_y2_of_linear_graph_l702_70224

/-- In the plane rectangular coordinate system xOy, if points A(2, y1) and B(5, y2) 
    lie on the graph of a linear function y = x + b (where b is a constant), then y1 < y2. -/
theorem y1_lt_y2_of_linear_graph (y1 y2 b : ℝ) (hA : y1 = 2 + b) (hB : y2 = 5 + b) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_graph_l702_70224


namespace moving_circle_trajectory_is_ellipse_l702_70230

noncomputable def trajectory_of_center (x y : ℝ) : Prop :=
  let ellipse_eq := x^2 / 4 + y^2 / 3 = 1 
  ellipse_eq ∧ x ≠ -2

theorem moving_circle_trajectory_is_ellipse
  (M_1 M_2 center : ℝ × ℝ)
  (r1 r2 R : ℝ)
  (h1 : M_1 = (-1, 0))
  (h2 : M_2 = (1, 0))
  (h3 : r1 = 1)
  (h4 : r2 = 3)
  (h5 : (center.1 + 1)^2 + center.2^2 = (1 + R)^2)
  (h6 : (center.1 - 1)^2 + center.2^2 = (3 - R)^2) :
  trajectory_of_center center.1 center.2 :=
by sorry

end moving_circle_trajectory_is_ellipse_l702_70230


namespace age_ratio_l702_70280

-- Definitions of the ages based on the given conditions.
def Rachel_age : ℕ := 12  -- Rachel's age
def Father_age_when_Rachel_25 : ℕ := 60

-- Defining Mother, Father, Grandfather ages based on given conditions.
def Grandfather_age (R : ℕ) (F : ℕ) : ℕ := 2 * (F - 5)
def Father_age (R : ℕ) : ℕ := Father_age_when_Rachel_25 - (25 - R)

-- Proving the ratio of Grandfather's age to Rachel's age is 7:1
theorem age_ratio (R : ℕ) (F : ℕ) (G : ℕ) :
  R = Rachel_age →
  F = Father_age R →
  G = Grandfather_age R F →
  G / R = 7 := by
  exact sorry

end age_ratio_l702_70280


namespace triangle_inscribed_angle_l702_70200

theorem triangle_inscribed_angle 
  (y : ℝ)
  (arc_PQ arc_QR arc_RP : ℝ)
  (h1 : arc_PQ = 2 * y + 40)
  (h2 : arc_QR = 3 * y + 15)
  (h3 : arc_RP = 4 * y - 40)
  (h4 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_P : ℝ, angle_P = 64.995 := 
by 
  sorry

end triangle_inscribed_angle_l702_70200


namespace parts_processed_per_hour_l702_70203

theorem parts_processed_per_hour (x : ℕ) (y : ℕ) (h1 : y = x + 10) (h2 : 150 / y = 120 / x) :
  x = 40 ∧ y = 50 :=
by {
  sorry
}

end parts_processed_per_hour_l702_70203
