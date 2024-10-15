import Mathlib

namespace NUMINAMATH_GPT_find_c_l631_63164

theorem find_c (a b c : ℝ) (h_line : 4 * a - 3 * b + c = 0) 
  (h_min : (a - 1)^2 + (b - 1)^2 = 4) : c = 9 ∨ c = -11 := 
    sorry

end NUMINAMATH_GPT_find_c_l631_63164


namespace NUMINAMATH_GPT_sheet_width_l631_63113

theorem sheet_width (L : ℕ) (w : ℕ) (A_typist : ℚ) 
  (L_length : L = 30)
  (A_typist_percentage : A_typist = 0.64) 
  (width_used : ∀ w, w > 0 → (w - 4) * (24 : ℕ) = A_typist * w * 30) : 
  w = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sheet_width_l631_63113


namespace NUMINAMATH_GPT_cloth_sold_l631_63117

theorem cloth_sold (total_sell_price : ℤ) (loss_per_meter : ℤ) (cost_price_per_meter : ℤ) (x : ℤ) 
    (h1 : total_sell_price = 18000) 
    (h2 : loss_per_meter = 5) 
    (h3 : cost_price_per_meter = 50) 
    (h4 : (cost_price_per_meter - loss_per_meter) * x = total_sell_price) : 
    x = 400 :=
by
  sorry

end NUMINAMATH_GPT_cloth_sold_l631_63117


namespace NUMINAMATH_GPT_band_formation_max_l631_63120

-- Define the conditions provided in the problem
theorem band_formation_max (m r x : ℕ) (h1 : m = r * x + 5)
  (h2 : (r - 3) * (x + 2) = m) (h3 : m < 100) :
  m = 70 :=
sorry

end NUMINAMATH_GPT_band_formation_max_l631_63120


namespace NUMINAMATH_GPT_solve_quadratic_eq_l631_63189

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 * x = 2 ↔ (x = 2 + Real.sqrt 6) ∨ (x = 2 - Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l631_63189


namespace NUMINAMATH_GPT_radian_to_degree_conversion_l631_63145

theorem radian_to_degree_conversion
: (π : ℝ) = 180 → ((-23 / 12) * π) = -345 :=
by
  sorry

end NUMINAMATH_GPT_radian_to_degree_conversion_l631_63145


namespace NUMINAMATH_GPT_base_number_is_three_l631_63171

theorem base_number_is_three (some_number : ℝ) (y : ℕ) (h1 : 9^y = some_number^14) (h2 : y = 7) : some_number = 3 :=
by { sorry }

end NUMINAMATH_GPT_base_number_is_three_l631_63171


namespace NUMINAMATH_GPT_total_cost_of_two_books_l631_63139

theorem total_cost_of_two_books (C1 C2 total_cost: ℝ) :
  C1 = 262.5 →
  0.85 * C1 = 1.19 * C2 →
  total_cost = C1 + C2 →
  total_cost = 450 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_cost_of_two_books_l631_63139


namespace NUMINAMATH_GPT_sum_of_multiples_of_6_and_9_is_multiple_of_3_l631_63115

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 
  (x y : ℤ) (hx : ∃ m : ℤ, x = 6 * m) (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_multiples_of_6_and_9_is_multiple_of_3_l631_63115


namespace NUMINAMATH_GPT_min_circles_l631_63102

noncomputable def segments_intersecting_circles (N : ℕ) : Prop :=
  ∀ seg : (ℝ × ℝ) × ℝ, (seg.fst.fst ≥ 0 ∧ seg.fst.fst + seg.snd ≤ 100 ∧ seg.fst.snd ≥ 0 ∧ seg.fst.snd ≤ 100 ∧ seg.snd = 10) →
    ∃ c : ℝ × ℝ, (dist c seg.fst < 1 ∧ c.fst ≥ 0 ∧ c.fst ≤ 100 ∧ c.snd ≥ 0 ∧ c.snd ≤ 100) 

theorem min_circles (N : ℕ) (h : segments_intersecting_circles N) : N ≥ 400 :=
sorry

end NUMINAMATH_GPT_min_circles_l631_63102


namespace NUMINAMATH_GPT_find_ab_l631_63100

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_ab_l631_63100


namespace NUMINAMATH_GPT_yevgeniy_age_2014_l631_63183

theorem yevgeniy_age_2014 (birth_year : ℕ) (h1 : birth_year = 1900 + (birth_year % 100))
  (h2 : 2011 - birth_year = (birth_year / 1000) + ((birth_year % 1000) / 100) + ((birth_year % 100) / 10) + (birth_year % 10)) :
  2014 - birth_year = 23 :=
by
  sorry

end NUMINAMATH_GPT_yevgeniy_age_2014_l631_63183


namespace NUMINAMATH_GPT_part1_part2_l631_63161

-- Part (1)
theorem part1 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hx : x > 0) (hy : y < 0) : x + y = -4 :=
sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hxy : x < y) : x - y = -10 ∨ x - y = -4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l631_63161


namespace NUMINAMATH_GPT_sum_of_solutions_l631_63160

theorem sum_of_solutions (x : ℝ) (hx : x + 36 / x = 12) : x = 6 ∨ x = -6 := sorry

end NUMINAMATH_GPT_sum_of_solutions_l631_63160


namespace NUMINAMATH_GPT_ellen_painted_17_lilies_l631_63155

theorem ellen_painted_17_lilies :
  (∃ n : ℕ, n * 5 + 10 * 7 + 6 * 3 + 20 * 2 = 213) → 
    ∃ n : ℕ, n = 17 := 
by sorry

end NUMINAMATH_GPT_ellen_painted_17_lilies_l631_63155


namespace NUMINAMATH_GPT_cubic_difference_l631_63156

theorem cubic_difference (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) : 
  235 < x^2 ∧ x^2 < 240 := 
sorry

end NUMINAMATH_GPT_cubic_difference_l631_63156


namespace NUMINAMATH_GPT_substitution_result_l631_63123

-- Conditions
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := x - 2 * y = 6

-- The statement to be proven
theorem substitution_result (x y : ℝ) (h1 : eq1 x y) : (x - 4 * x + 6 = 6) :=
by sorry

end NUMINAMATH_GPT_substitution_result_l631_63123


namespace NUMINAMATH_GPT_max_m_n_value_l631_63178

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end NUMINAMATH_GPT_max_m_n_value_l631_63178


namespace NUMINAMATH_GPT_determine_k_l631_63142

theorem determine_k (a b c k : ℤ) (h1 : c = -a - b) 
  (h2 : 60 < 6 * (8 * a + b) ∧ 6 * (8 * a + b) < 70)
  (h3 : 80 < 7 * (9 * a + b) ∧ 7 * (9 * a + b) < 90)
  (h4 : 2000 * k < (50^2 * a + 50 * b + c) ∧ (50^2 * a + 50 * b + c) < 2000 * (k + 1)) :
  k = 1 :=
  sorry

end NUMINAMATH_GPT_determine_k_l631_63142


namespace NUMINAMATH_GPT_gas_volume_ranking_l631_63188

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end NUMINAMATH_GPT_gas_volume_ranking_l631_63188


namespace NUMINAMATH_GPT_translation_of_segment_l631_63173

structure Point where
  x : ℝ
  y : ℝ

variables (A B A' : Point)

def translation_vector (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y }

def translate (P Q : Point) : Point :=
  { x := P.x + Q.x,
    y := P.y + Q.y }

theorem translation_of_segment (hA : A = {x := -2, y := 0})
                                (hB : B = {x := 0, y := 3})
                                (hA' : A' = {x := 2, y := 1}) :
  translate B (translation_vector A A') = {x := 4, y := 4} := by
  sorry

end NUMINAMATH_GPT_translation_of_segment_l631_63173


namespace NUMINAMATH_GPT_domain_of_f_l631_63106

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.sqrt (-x^2 + x + 2)

theorem domain_of_f :
  {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l631_63106


namespace NUMINAMATH_GPT_find_a8_l631_63191

noncomputable def a (n : ℕ) : ℤ := sorry

noncomputable def b (n : ℕ) : ℤ := a (n + 1) - a n

theorem find_a8 :
  (a 1 = 3) ∧
  (∀ n : ℕ, b n = b 1 + n * 2) ∧
  (b 3 = -2) ∧
  (b 10 = 12) →
  a 8 = 3 :=
by sorry

end NUMINAMATH_GPT_find_a8_l631_63191


namespace NUMINAMATH_GPT_percentage_discount_l631_63170

theorem percentage_discount (original_price sale_price : ℝ) (h1 : original_price = 25) (h2 : sale_price = 18.75) : 
  100 * (original_price - sale_price) / original_price = 25 := 
by
  -- Begin Proof
  sorry

end NUMINAMATH_GPT_percentage_discount_l631_63170


namespace NUMINAMATH_GPT_quadratic_form_h_l631_63186

theorem quadratic_form_h (x h : ℝ) (a k : ℝ) (h₀ : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_form_h_l631_63186


namespace NUMINAMATH_GPT_function_behavior_on_intervals_l631_63158

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem function_behavior_on_intervals :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < deriv f x) ∧
  (∀ x : ℝ, Real.exp 1 < x ∧ x < 10 → deriv f x < 0) := sorry

end NUMINAMATH_GPT_function_behavior_on_intervals_l631_63158


namespace NUMINAMATH_GPT_cubic_no_maximum_value_l631_63125

theorem cubic_no_maximum_value (x : ℝ) : ¬ ∃ M, ∀ x : ℝ, 3 * x^2 + 6 * x^3 + 27 * x + 100 ≤ M := 
by
  sorry

end NUMINAMATH_GPT_cubic_no_maximum_value_l631_63125


namespace NUMINAMATH_GPT_secretary_longest_time_l631_63128

def ratio_times (x : ℕ) : Prop := 
  let t1 := 2 * x
  let t2 := 3 * x
  let t3 := 5 * x
  (t1 + t2 + t3 = 110) ∧ (t3 = 55)

theorem secretary_longest_time :
  ∃ x : ℕ, ratio_times x :=
sorry

end NUMINAMATH_GPT_secretary_longest_time_l631_63128


namespace NUMINAMATH_GPT_infinite_solutions_l631_63140

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def eq2 (x y : ℝ) : Prop := 6 * x - 8 * y = 2

-- State that there are an unlimited number of solutions
theorem infinite_solutions : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧
  ∀ y : ℝ, ∃ x : ℝ, eq1 x y :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_l631_63140


namespace NUMINAMATH_GPT_soccer_boys_percentage_l631_63132

theorem soccer_boys_percentage (total_students boys total_playing_soccer girls_not_playing_soccer : ℕ)
  (h_total_students : total_students = 500)
  (h_boys : boys = 350)
  (h_total_playing_soccer : total_playing_soccer = 250)
  (h_girls_not_playing_soccer : girls_not_playing_soccer = 115) :
  (boys - (total_students - total_playing_soccer) / total_playing_soccer * 100) = 86 :=
by
  sorry

end NUMINAMATH_GPT_soccer_boys_percentage_l631_63132


namespace NUMINAMATH_GPT_A_time_240m_race_l631_63116

theorem A_time_240m_race (t : ℕ) :
  (∀ t, (240 / t) = (184 / t) * (t + 7) ∧ 240 = 184 + ((184 * 7) / t)) → t = 23 :=
by
  sorry

end NUMINAMATH_GPT_A_time_240m_race_l631_63116


namespace NUMINAMATH_GPT_final_inventory_is_correct_l631_63165

def initial_inventory : ℕ := 4500
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_remaining_week : ℕ := 50
def supplier_delivery : ℕ := 650

def bottles_sold_first_two_days : ℕ := bottles_sold_monday + bottles_sold_tuesday
def days_remaining : ℕ := 5
def bottles_sold_remaining_week : ℕ := days_remaining * bottles_sold_per_day_remaining_week
def total_bottles_sold_week : ℕ := bottles_sold_first_two_days + bottles_sold_remaining_week
def remaining_inventory : ℕ := initial_inventory - total_bottles_sold_week
def final_inventory : ℕ := remaining_inventory + supplier_delivery

theorem final_inventory_is_correct :
  final_inventory = 1555 :=
by
  sorry

end NUMINAMATH_GPT_final_inventory_is_correct_l631_63165


namespace NUMINAMATH_GPT_smaller_number_is_three_l631_63175

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_three_l631_63175


namespace NUMINAMATH_GPT_trigonometric_identity_l631_63110

theorem trigonometric_identity (α : Real) (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l631_63110


namespace NUMINAMATH_GPT_model_tower_height_l631_63185

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_cond : real_height = 80) (vol_cond : real_volume = 200000) (model_vol_cond : model_volume = 0.2) : 
  ∃ h : ℝ, h = 0.8 :=
by sorry

end NUMINAMATH_GPT_model_tower_height_l631_63185


namespace NUMINAMATH_GPT_december_28_is_saturday_l631_63152

def days_per_week := 7

def thanksgiving_day : Nat := 28

def november_length : Nat := 30

def december_28_day_of_week : Nat :=
  (thanksgiving_day % days_per_week + november_length + 28 - thanksgiving_day) % days_per_week

theorem december_28_is_saturday :
  (december_28_day_of_week = 6) :=
by
  sorry

end NUMINAMATH_GPT_december_28_is_saturday_l631_63152


namespace NUMINAMATH_GPT_largest_divisor_of_n4_minus_n_l631_63131

theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : ¬(Prime n) ∧ n ≠ 1) : 6 ∣ (n^4 - n) :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_minus_n_l631_63131


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_perpendicular_l631_63177

theorem sufficient_but_not_necessary_perpendicular (a : ℝ) :
  (∃ a' : ℝ, a' = -1 ∧ (a' = -1 → (0 : ℝ) ≠ 3 * a' - 1)) ∨
  (∃ a' : ℝ, a' ≠ -1 ∧ (a' ≠ -1 → (0 : ℝ) ≠ 3 * a' - 1)) →
  (3 * a' - 1) * (a' - 3) = -1 := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_perpendicular_l631_63177


namespace NUMINAMATH_GPT_necessarily_positive_y_plus_xsq_l631_63146

theorem necessarily_positive_y_plus_xsq {x y z : ℝ} 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  y + x^2 > 0 :=
sorry

end NUMINAMATH_GPT_necessarily_positive_y_plus_xsq_l631_63146


namespace NUMINAMATH_GPT_sum_of_roots_of_polynomial_l631_63172

theorem sum_of_roots_of_polynomial (a b c : ℝ) (h : 3*a^3 - 7*a^2 + 6*a = 0) : 
    (∀ x, 3*x^2 - 7*x + 6 = 0 → x = a ∨ x = b ∨ x = c) →
    (∀ (x : ℝ), (x = a ∨ x = b ∨ x = c → 3*x^3 - 7*x^2 + 6*x = 0)) → 
    a + b + c = 7 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_polynomial_l631_63172


namespace NUMINAMATH_GPT_probability_correct_l631_63174

structure SockDrawSetup where
  total_socks : ℕ
  color_pairs : ℕ
  socks_per_color : ℕ
  draw_size : ℕ

noncomputable def probability_one_pair (S : SockDrawSetup) : ℚ :=
  let total_combinations := Nat.choose S.total_socks S.draw_size
  let favorable_combinations := (Nat.choose S.color_pairs 3) * (Nat.choose 3 1) * 2 * 2
  favorable_combinations / total_combinations

theorem probability_correct (S : SockDrawSetup) (h1 : S.total_socks = 12) (h2 : S.color_pairs = 6) (h3 : S.socks_per_color = 2) (h4 : S.draw_size = 6) :
  probability_one_pair S = 20 / 77 :=
by
  apply sorry

end NUMINAMATH_GPT_probability_correct_l631_63174


namespace NUMINAMATH_GPT_express_set_A_l631_63130

def A := {x : ℤ | -1 < abs (x - 1) ∧ abs (x - 1) < 2}

theorem express_set_A : A = {0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_express_set_A_l631_63130


namespace NUMINAMATH_GPT_option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l631_63121

section

variable (π : Real) (x : Real)

-- Definition of a fraction in this context
def is_fraction (num denom : Real) : Prop := denom ≠ 0

-- Proving each given option is a fraction
theorem option_a_is_fraction : is_fraction 1 π := 
sorry

theorem option_b_is_fraction : is_fraction x 3 :=
sorry

theorem option_c_is_fraction : is_fraction 2 5 :=
sorry

theorem option_d_is_fraction : is_fraction 1 (x - 1) :=
sorry

end

end NUMINAMATH_GPT_option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l631_63121


namespace NUMINAMATH_GPT_adam_bought_dog_food_packages_l631_63196

-- Define the constants and conditions
def num_cat_food_packages : ℕ := 9
def cans_per_cat_food_package : ℕ := 10
def cans_per_dog_food_package : ℕ := 5
def additional_cat_food_cans : ℕ := 55

-- Define the variable for dog food packages and our equation
def num_dog_food_packages (d : ℕ) : Prop :=
  (num_cat_food_packages * cans_per_cat_food_package) = (d * cans_per_dog_food_package + additional_cat_food_cans)

-- The theorem statement representing the proof problem
theorem adam_bought_dog_food_packages : ∃ d : ℕ, num_dog_food_packages d ∧ d = 7 :=
sorry

end NUMINAMATH_GPT_adam_bought_dog_food_packages_l631_63196


namespace NUMINAMATH_GPT_number_of_people_in_group_l631_63184

variable (T L : ℕ)

theorem number_of_people_in_group
  (h1 : 90 + L = T)
  (h2 : (L : ℚ) / T = 0.4) :
  T = 150 := by
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l631_63184


namespace NUMINAMATH_GPT_simplify_expression_l631_63182

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l631_63182


namespace NUMINAMATH_GPT_find_x_l631_63194

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (4, -6, x)
def dot_product : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ
  | (a1, a2, a3), (b1, b2, b3) => a1 * b1 + a2 * b2 + a3 * b3

theorem find_x (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -26 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l631_63194


namespace NUMINAMATH_GPT_male_female_ratio_l631_63101

-- Definitions and constants
variable (M F : ℕ) -- Number of male and female members respectively
variable (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) -- Average ticket sales condition

-- Statement of the theorem
theorem male_female_ratio (M F : ℕ) (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) : M / F = 1 / 2 :=
sorry

end NUMINAMATH_GPT_male_female_ratio_l631_63101


namespace NUMINAMATH_GPT_blocks_left_l631_63114

def blocks_initial := 78
def blocks_used := 19

theorem blocks_left : blocks_initial - blocks_used = 59 :=
by
  -- Solution is not required here, so we add a sorry placeholder.
  sorry

end NUMINAMATH_GPT_blocks_left_l631_63114


namespace NUMINAMATH_GPT_A_minus_one_not_prime_l631_63193

theorem A_minus_one_not_prime (n : ℕ) (h : 0 < n) (m : ℕ) (h1 : 10^(m-1) < 14^n) (h2 : 14^n < 10^m) :
  ¬ (Nat.Prime (2^n * 10^m + 14^n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_A_minus_one_not_prime_l631_63193


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l631_63197

theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2) ↔ (∃ m1 m2 : ℝ, (m1 = -1/(4 : ℝ)) ∧ (m2 = (4 : ℝ)) ∧ (m1 * m2 = -1)) :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l631_63197


namespace NUMINAMATH_GPT_parallel_line_slope_l631_63167

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, y = (1 / 2) * x + b) → 
  (∃ a : ℝ, 3 * x - 6 * y = a) → 
  ∃ k : ℝ, k = 1 / 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_parallel_line_slope_l631_63167


namespace NUMINAMATH_GPT_problem_statement_l631_63147

noncomputable def probability_different_colors : ℚ :=
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red)

theorem problem_statement :
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red) = 56 / 121 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l631_63147


namespace NUMINAMATH_GPT_find_k_l631_63111

def vector := (ℝ × ℝ)

def a : vector := (3, 1)
def b : vector := (1, 3)
def c (k : ℝ) : vector := (k, 2)

def subtract (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) (h : dot_product (subtract a (c k)) b = 0) : k = 0 := by
  sorry

end NUMINAMATH_GPT_find_k_l631_63111


namespace NUMINAMATH_GPT_sequence_a4_l631_63181

theorem sequence_a4 :
  (∀ n : ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    (a 1 = 1) →
    (∀ n > 0, a (n + 1) = (1 / 2) * a n + 1 / (2 ^ n)) →
    a 4 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_l631_63181


namespace NUMINAMATH_GPT_total_prize_money_l631_63148

theorem total_prize_money (P1 P2 P3 : ℕ) (d : ℕ) (total : ℕ) 
(h1 : P1 = 2000) (h2 : d = 400) (h3 : P2 = P1 - d) (h4 : P3 = P2 - d) 
(h5 : total = P1 + P2 + P3) : total = 4800 :=
sorry

end NUMINAMATH_GPT_total_prize_money_l631_63148


namespace NUMINAMATH_GPT_max_value_of_x_l631_63108

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem max_value_of_x 
  (x : ℤ) 
  (h : log_base (1 / 4 : ℝ) (2 * x + 1) < log_base (1 / 2 : ℝ) (x - 1)) : x ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_l631_63108


namespace NUMINAMATH_GPT_students_not_making_cut_l631_63176

theorem students_not_making_cut :
  let girls := 39
  let boys := 4
  let called_back := 26
  let total := girls + boys
  total - called_back = 17 :=
by
  -- add the proof here
  sorry

end NUMINAMATH_GPT_students_not_making_cut_l631_63176


namespace NUMINAMATH_GPT_cube_cut_edges_l631_63127

theorem cube_cut_edges (original_edges new_edges_per_vertex vertices : ℕ) (h1 : original_edges = 12) (h2 : new_edges_per_vertex = 6) (h3 : vertices = 8) :
  original_edges + new_edges_per_vertex * vertices = 60 :=
by
  sorry

end NUMINAMATH_GPT_cube_cut_edges_l631_63127


namespace NUMINAMATH_GPT_exists_distinct_positive_integers_l631_63166

theorem exists_distinct_positive_integers (n : ℕ) (h : 0 < n) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end NUMINAMATH_GPT_exists_distinct_positive_integers_l631_63166


namespace NUMINAMATH_GPT_simplify_fraction_l631_63134

theorem simplify_fraction :
  (5 : ℚ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
sorry

end NUMINAMATH_GPT_simplify_fraction_l631_63134


namespace NUMINAMATH_GPT_triangle_statements_l631_63141

-- Definitions of internal angles and sides of the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Statement A: If ABC is an acute triangle, then sin A > cos B
lemma statement_A (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  Real.sin A > Real.cos B := 
sorry

-- Statement B: If A > B, then sin A > sin B
lemma statement_B (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_AB : A > B) : 
  Real.sin A > Real.sin B := 
sorry

-- Statement C: If ABC is a non-right triangle, then tan A + tan B + tan C = tan A * tan B * tan C
lemma statement_C (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2) : 
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

-- Statement D: If a cos A = b cos B, then triangle ABC must be isosceles
lemma statement_D (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  ¬(A = B) ∧ ¬(B = C) := 
sorry

-- Theorem to combine all statements
theorem triangle_statements (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)
  (h_AB : A > B)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  Real.sin A > Real.cos B ∧ Real.sin A > Real.sin B ∧ 
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) ∧ 
  ¬(A = B) ∧ ¬(B = C) := 
by
  exact ⟨statement_A A B C a b c h_triangle h_acute, statement_B A B C a b c h_triangle h_AB, statement_C A B C a b c h_triangle h_non_right, statement_D A B C a b c h_triangle h_cos⟩

end NUMINAMATH_GPT_triangle_statements_l631_63141


namespace NUMINAMATH_GPT_student_first_subject_percentage_l631_63199

variable (P : ℝ)

theorem student_first_subject_percentage 
  (H1 : 80 = 80)
  (H2 : 75 = 75)
  (H3 : (P + 80 + 75) / 3 = 75) :
  P = 70 :=
by
  sorry

end NUMINAMATH_GPT_student_first_subject_percentage_l631_63199


namespace NUMINAMATH_GPT_find_XY_length_l631_63118

variables (a b c : ℝ) -- sides of triangle ABC
variables (s : ℝ) -- semi-perimeter s = (a + b + c) / 2

-- Definition of similar triangles and perimeter condition
noncomputable def XY_length
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ) 
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) : ℝ :=
  s * a / (b + c) -- by the given solution

-- The theorem statement
theorem find_XY_length
  (a b c : ℝ) (s : ℝ) -- given sides and semi-perimeter
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ)
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) :
  XY = s * a / (b + c) :=
sorry -- proof


end NUMINAMATH_GPT_find_XY_length_l631_63118


namespace NUMINAMATH_GPT_find_f2a_eq_zero_l631_63137

variable {α : Type} [LinearOrderedField α]

-- Definitions for the function f and its inverse
variable (f : α → α)
variable (finv : α → α)

-- Given conditions
variable (a : α)
variable (h_nonzero : a ≠ 0)
variable (h_inverse1 : ∀ x : α, finv (x + a) = f (x + a)⁻¹)
variable (h_inverse2 : ∀ x : α, f (x) = finv⁻¹ x)
variable (h_fa : f a = a)

-- Statement to be proved in Lean
theorem find_f2a_eq_zero : f (2 * a) = 0 :=
sorry

end NUMINAMATH_GPT_find_f2a_eq_zero_l631_63137


namespace NUMINAMATH_GPT_inequality_solution_set_l631_63169

noncomputable def solution_set := { x : ℝ | 0 < x ∧ x < 2 }

theorem inequality_solution_set : 
  { x : ℝ | (4 / x > |x|) } = solution_set :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l631_63169


namespace NUMINAMATH_GPT_num_ways_to_distribute_7_balls_in_4_boxes_l631_63153

def num_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  -- Implement the function to calculate the number of ways here, but we'll keep it as a placeholder for now.
  sorry

theorem num_ways_to_distribute_7_balls_in_4_boxes : 
  num_ways_to_distribute_balls 7 4 = 3 := 
sorry

end NUMINAMATH_GPT_num_ways_to_distribute_7_balls_in_4_boxes_l631_63153


namespace NUMINAMATH_GPT_false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l631_63195

-- Proposition A
theorem false_proposition_A (a b c : ℝ) (hac : a > b) (hca : b > 0) : ac * c^2 = b * c^2 :=
  sorry

-- Proposition B
theorem false_proposition_B (a b : ℝ) (hab : a < b) : (1/a) < (1/b) :=
  sorry

-- Proposition C
theorem true_proposition_C (a b : ℝ) (hab : a > b) (hba : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
  sorry

-- Proposition D
theorem true_proposition_D (a b : ℝ) (hba : a > |b|) : a^2 > b^2 :=
  sorry

end NUMINAMATH_GPT_false_proposition_A_false_proposition_B_true_proposition_C_true_proposition_D_l631_63195


namespace NUMINAMATH_GPT_factorization_correct_l631_63187

-- Define the expression
def expression (a b : ℝ) : ℝ := 3 * a^2 - 3 * b^2

-- Define the factorized form of the expression
def factorized (a b : ℝ) : ℝ := 3 * (a + b) * (a - b)

-- The main statement we need to prove
theorem factorization_correct (a b : ℝ) : expression a b = factorized a b :=
by 
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_factorization_correct_l631_63187


namespace NUMINAMATH_GPT_find_parabola_eq_find_range_of_b_l631_63154

-- Problem 1: Finding the equation of the parabola
theorem find_parabola_eq (p : ℝ) (h1 : p > 0) (x1 x2 y1 y2 : ℝ) 
  (A : (x1 + 4) * 2 = 2 * p * y1) (C : (x2 + 4) * 2 = 2 * p * y2)
  (h3 : x1^2 = 2 * p * y1) (h4 : x2^2 = 2 * p * y2) 
  (h5 : y2 = 4 * y1) :
  x1^2 = 4 * y1 :=
sorry

-- Problem 2: Finding the range of b
theorem find_range_of_b (k : ℝ) (h : k > 0 ∨ k < -4) : 
  ∃ b : ℝ, b = 2 * (k + 1)^2 ∧ b > 2 :=
sorry

end NUMINAMATH_GPT_find_parabola_eq_find_range_of_b_l631_63154


namespace NUMINAMATH_GPT_find_square_number_divisible_by_six_l631_63179

theorem find_square_number_divisible_by_six :
  ∃ x : ℕ, (∃ k : ℕ, x = k^2) ∧ x % 6 = 0 ∧ 24 < x ∧ x < 150 ∧ (x = 36 ∨ x = 144) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_square_number_divisible_by_six_l631_63179


namespace NUMINAMATH_GPT_area_of_rectangle_R_l631_63151

-- Define the side lengths of the squares and rectangles involved
def larger_square_side := 4
def smaller_square_side := 2
def rectangle_side1 := 1
def rectangle_side2 := 4

-- The areas of these shapes
def area_larger_square := larger_square_side * larger_square_side
def area_smaller_square := smaller_square_side * smaller_square_side
def area_first_rectangle := rectangle_side1 * rectangle_side2

-- Define the sum of all possible values for the area of rectangle R
def area_remaining := area_larger_square - (area_smaller_square + area_first_rectangle)

theorem area_of_rectangle_R : area_remaining = 8 := sorry

end NUMINAMATH_GPT_area_of_rectangle_R_l631_63151


namespace NUMINAMATH_GPT_spending_difference_l631_63135

-- Define the cost of the candy bar
def candy_bar_cost : ℕ := 6

-- Define the cost of the chocolate
def chocolate_cost : ℕ := 3

-- Prove the difference between candy_bar_cost and chocolate_cost
theorem spending_difference : candy_bar_cost - chocolate_cost = 3 :=
by
    sorry

end NUMINAMATH_GPT_spending_difference_l631_63135


namespace NUMINAMATH_GPT_probability_one_hits_l631_63105

theorem probability_one_hits (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.6) :
  (P_A * (1 - P_B) + (1 - P_A) * P_B) = 0.48 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_hits_l631_63105


namespace NUMINAMATH_GPT_alan_more_wings_per_minute_to_beat_record_l631_63109

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end NUMINAMATH_GPT_alan_more_wings_per_minute_to_beat_record_l631_63109


namespace NUMINAMATH_GPT_equilateral_triangle_ratio_is_correct_l631_63192

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_ratio_is_correct_l631_63192


namespace NUMINAMATH_GPT_mixed_operations_with_decimals_false_l631_63112

-- Definitions and conditions
def operations_same_level_with_decimals : Prop :=
  ∀ (a b c : ℝ), a + b - c = (a + b) - c

def calculate_left_to_right_with_decimals : Prop :=
  ∀ (a b c : ℝ), (a - b + c) = a - b + c ∧ (a + b - c) = a + b - c

-- Proposition we're proving
theorem mixed_operations_with_decimals_false :
  ¬ ∀ (a b c : ℝ), (a + b - c) ≠ (a - b + c) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mixed_operations_with_decimals_false_l631_63112


namespace NUMINAMATH_GPT_length_of_AD_l631_63168

theorem length_of_AD 
  (A B C D : Type) 
  (vertex_angle_equal: ∀ {a b c d : Type}, a = A →
    ∀ (AB AC AD : ℝ), (AB = 24) → (AC = 54) → (AD = 36)) 
  (right_triangles : ∀ {a b : Type}, a = A → ∀ {AB AC : ℝ}, (AB > 0) → (AC > 0) → (AB ^ 2 + AC ^ 2 = AD ^ 2)) :
  ∃ (AD : ℝ), AD = 36 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AD_l631_63168


namespace NUMINAMATH_GPT_plane_split_into_8_regions_l631_63180

-- Define the conditions as separate lines in the plane.
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := x = y

-- Define a theorem stating that these lines together split the plane into 8 regions.
theorem plane_split_into_8_regions :
  (∀ (x y : ℝ), line1 x y ∨ line2 x y ∨ line3 x y) →
  -- The plane is split into exactly 8 regions by these lines
  ∃ (regions : ℕ), regions = 8 :=
sorry

end NUMINAMATH_GPT_plane_split_into_8_regions_l631_63180


namespace NUMINAMATH_GPT_wendy_total_gas_to_add_l631_63162

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end NUMINAMATH_GPT_wendy_total_gas_to_add_l631_63162


namespace NUMINAMATH_GPT_average_length_one_third_of_strings_l631_63129

theorem average_length_one_third_of_strings (average_six_strings : ℕ → ℕ → ℕ)
    (average_four_strings : ℕ → ℕ → ℕ)
    (total_length : ℕ → ℕ → ℕ)
    (n m : ℕ) :
    (n = 6) →
    (m = 4) →
    (average_six_strings 80 n = 480) →
    (average_four_strings 85 m = 340) →
    (total_length 2 70 = 140) →
    70 = (480 - 340) / 2 :=
by
  intros h_n h_m avg_six avg_four total_len
  sorry

end NUMINAMATH_GPT_average_length_one_third_of_strings_l631_63129


namespace NUMINAMATH_GPT_set_A_is_correct_l631_63124

open Complex

def A : Set ℤ := {x | ∃ n : ℕ, n > 0 ∧ x = (I ^ n + (-I) ^ n).re}

theorem set_A_is_correct : A = {-2, 0, 2} :=
sorry

end NUMINAMATH_GPT_set_A_is_correct_l631_63124


namespace NUMINAMATH_GPT_tickets_total_l631_63122

theorem tickets_total (x y : ℕ) 
  (h1 : 12 * x + 8 * y = 3320)
  (h2 : y = x + 190) : 
  x + y = 370 :=
by
  sorry

end NUMINAMATH_GPT_tickets_total_l631_63122


namespace NUMINAMATH_GPT_geom_seq_sum_first_four_terms_l631_63107

noncomputable def sum_first_n_terms_geom (a₁ q: ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_first_four_terms
  (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (h₂ : a₁ * q^3 = 27) :
  sum_first_n_terms_geom a₁ q 4 = 40 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_first_four_terms_l631_63107


namespace NUMINAMATH_GPT_fraction_start_with_9_end_with_0_is_1_over_72_l631_63136

-- Definition of valid 8-digit telephone number
def valid_phone_number (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  2 ≤ d.val ∧ d.val ≤ 9 ∧ n.val ≤ 8

-- Definition of phone numbers that start with 9 and end with 0
def starts_with_9_ends_with_0 (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  d.val = 9 ∧ n.val = 0

-- The total number of valid 8-digit phone numbers
noncomputable def total_valid_numbers : ℕ :=
  8 * (10 ^ 6) * 9

-- The number of valid phone numbers that start with 9 and end with 0
noncomputable def valid_start_with_9_end_with_0 : ℕ :=
  10 ^ 6

-- The target fraction
noncomputable def target_fraction : ℚ :=
  valid_start_with_9_end_with_0 / total_valid_numbers

-- Main theorem
theorem fraction_start_with_9_end_with_0_is_1_over_72 :
  target_fraction = (1 / 72 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_start_with_9_end_with_0_is_1_over_72_l631_63136


namespace NUMINAMATH_GPT_tangent_line_parallel_x_axis_coordinates_l631_63144

theorem tangent_line_parallel_x_axis_coordinates :
  (∃ P : ℝ × ℝ, P = (1, -2) ∨ P = (-1, 2)) ↔
  (∃ x y : ℝ, y = x^3 - 3 * x ∧ ∃ y', y' = 3 * x^2 - 3 ∧ y' = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_x_axis_coordinates_l631_63144


namespace NUMINAMATH_GPT_problem_statement_l631_63119

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f a b α β 2007 = 5) :
  f a b α β 2008 = 3 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l631_63119


namespace NUMINAMATH_GPT_cost_of_six_hotdogs_and_seven_burgers_l631_63103

theorem cost_of_six_hotdogs_and_seven_burgers :
  ∀ (h b : ℝ), 4 * h + 5 * b = 3.75 → 5 * h + 3 * b = 3.45 → 6 * h + 7 * b = 5.43 :=
by
  intros h b h_eqn b_eqn
  sorry

end NUMINAMATH_GPT_cost_of_six_hotdogs_and_seven_burgers_l631_63103


namespace NUMINAMATH_GPT_remainder_500th_in_T_l631_63143

def sequence_T (n : ℕ) : ℕ := sorry -- Assume a definition for the sequence T where n represents the position and the sequence contains numbers having exactly 9 ones in their binary representation.

theorem remainder_500th_in_T :
  (sequence_T 500) % 500 = 191 := 
sorry

end NUMINAMATH_GPT_remainder_500th_in_T_l631_63143


namespace NUMINAMATH_GPT_final_value_of_A_l631_63104

theorem final_value_of_A : 
  ∀ (A : Int), 
    (A = 20) → 
    (A = -A + 10) → 
    A = -10 :=
by
  intros A h1 h2
  sorry

end NUMINAMATH_GPT_final_value_of_A_l631_63104


namespace NUMINAMATH_GPT_hyperbola_correct_eqn_l631_63198

open Real

def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_correct_eqn (e c a b x y : ℝ)
  (h_eccentricity : e = 2)
  (h_foci_distance : c = 4)
  (h_major_axis_half_length : a = 2)
  (h_minor_axis_half_length_square : b^2 = c^2 - a^2) :
  hyperbola_eqn x y :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_correct_eqn_l631_63198


namespace NUMINAMATH_GPT_units_digit_of_m_squared_plus_3_to_the_m_l631_63163

theorem units_digit_of_m_squared_plus_3_to_the_m (m : ℕ) (h : m = 2010^2 + 2^2010) : 
  (m^2 + 3^m) % 10 = 7 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_units_digit_of_m_squared_plus_3_to_the_m_l631_63163


namespace NUMINAMATH_GPT_solve_quadratic_equation_l631_63150

theorem solve_quadratic_equation (x : ℝ) :
    2 * x * (x - 5) = 3 * (5 - x) ↔ (x = 5 ∨ x = -3/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l631_63150


namespace NUMINAMATH_GPT_ratio_a7_b7_l631_63133

variable {α : Type*}
variables {a_n b_n : ℕ → α} [AddGroup α] [Field α]
variables {S_n T_n : ℕ → α}

-- Define the sum of the first n terms for sequences a_n and b_n
def sum_of_first_terms_a (n : ℕ) := S_n n = (n * (a_n n + a_n (n-1))) / 2
def sum_of_first_terms_b (n : ℕ) := T_n n = (n * (b_n n + b_n (n-1))) / 2

-- Given condition about the ratio of sums
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (3 * n - 2) / (2 * n + 1)

-- The statement to be proved
theorem ratio_a7_b7 : (a_n 7 / b_n 7) = (37 / 27) := sorry

end NUMINAMATH_GPT_ratio_a7_b7_l631_63133


namespace NUMINAMATH_GPT_g_2_eq_8_l631_63138

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_2_eq_8 : g 2 = 8 := 
by 
  unfold g
  unfold f_inv
  sorry

end NUMINAMATH_GPT_g_2_eq_8_l631_63138


namespace NUMINAMATH_GPT_molecular_weight_correct_l631_63190

-- Define atomic weights of elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_D : ℝ := 2.01

-- Define the number of each type of atom in the compound
def num_Ba : ℕ := 2
def num_O : ℕ := 3
def num_H : ℕ := 4
def num_D : ℕ := 1

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  (num_Ba * atomic_weight_Ba) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_D * atomic_weight_D)

-- Theorem stating the molecular weight is 328.71 g/mol
theorem molecular_weight_correct :
  molecular_weight = 328.71 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l631_63190


namespace NUMINAMATH_GPT_polynomial_expansion_sum_l631_63149

theorem polynomial_expansion_sum (a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a_6 + a_5 + a_4 + a_3 + a_2 + a_1 + a = 64 :=
by
  -- Proof is not needed, placeholder here.
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_l631_63149


namespace NUMINAMATH_GPT_volume_of_larger_cube_is_343_l631_63159

-- We will define the conditions first
def smaller_cube_side_length : ℤ := 1
def number_of_smaller_cubes : ℤ := 343
def volume_small_cube (l : ℤ) : ℤ := l^3
def diff_surface_area (l L : ℤ) : ℤ := (number_of_smaller_cubes * 6 * l^2) - (6 * L^2)

-- Main statement to prove the volume of the larger cube
theorem volume_of_larger_cube_is_343 :
  ∃ L, volume_small_cube smaller_cube_side_length * number_of_smaller_cubes = L^3 ∧
        diff_surface_area smaller_cube_side_length L = 1764 ∧
        volume_small_cube L = 343 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_larger_cube_is_343_l631_63159


namespace NUMINAMATH_GPT_chair_cost_l631_63157

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end NUMINAMATH_GPT_chair_cost_l631_63157


namespace NUMINAMATH_GPT_total_money_l631_63126

-- Conditions
def mark_amount : ℚ := 5 / 6
def carolyn_amount : ℚ := 2 / 5

-- Combine both amounts and state the theorem to be proved
theorem total_money : mark_amount + carolyn_amount = 1.233 := by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_total_money_l631_63126
