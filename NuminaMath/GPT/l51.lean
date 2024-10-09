import Mathlib

namespace triangle_base_l51_5131

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l51_5131


namespace isosceles_triangle_three_times_ce_l51_5172

/-!
# Problem statement
In the isosceles triangle \( ABC \) with \( \overline{AC} = \overline{BC} \), 
\( D \) is the foot of the altitude through \( C \) and \( M \) is 
the midpoint of segment \( CD \). The line \( BM \) intersects \( AC \) 
at \( E \). Prove that \( AC \) is three times as long as \( CE \).
-/

-- Definition of isosceles triangle and related points
variables {A B C D E M : Type} 

-- Assume necessary conditions
variables (triangle_isosceles : A = B)
variables (D_foot : true) -- Placeholder, replace with proper definition if needed
variables (M_midpoint : true) -- Placeholder, replace with proper definition if needed
variables (BM_intersects_AC : true) -- Placeholder, replace with proper definition if needed

-- Main statement to prove
theorem isosceles_triangle_three_times_ce (h1 : A = B)
    (h2 : true) (h3 : true) (h4 : true) : 
    AC = 3 * CE :=
by
  sorry

end isosceles_triangle_three_times_ce_l51_5172


namespace cookie_calories_l51_5133

theorem cookie_calories 
  (burger_calories : ℕ)
  (carrot_stick_calories : ℕ)
  (num_carrot_sticks : ℕ)
  (total_lunch_calories : ℕ) :
  burger_calories = 400 ∧ 
  carrot_stick_calories = 20 ∧ 
  num_carrot_sticks = 5 ∧ 
  total_lunch_calories = 750 →
  (total_lunch_calories - (burger_calories + num_carrot_sticks * carrot_stick_calories) = 250) :=
by sorry

end cookie_calories_l51_5133


namespace Marty_combinations_l51_5136

theorem Marty_combinations :
  let colors := 5
  let methods := 4
  let patterns := 3
  colors * methods * patterns = 60 :=
by
  sorry

end Marty_combinations_l51_5136


namespace product_of_square_roots_of_nine_l51_5119

theorem product_of_square_roots_of_nine (a b : ℝ) (ha : a^2 = 9) (hb : b^2 = 9) : a * b = -9 :=
sorry

end product_of_square_roots_of_nine_l51_5119


namespace g_50_zero_l51_5177

noncomputable def g : ℕ → ℝ → ℝ
| 0, x     => x + |x - 50| - |x + 50|
| (n+1), x => |g n x| - 2

theorem g_50_zero :
  ∃! x : ℝ, g 50 x = 0 :=
sorry

end g_50_zero_l51_5177


namespace spider_has_eight_legs_l51_5162

-- Define the number of legs a human has
def human_legs : ℕ := 2

-- Define the number of legs for a spider, based on the given condition
def spider_legs : ℕ := 2 * (2 * human_legs)

-- The theorem to be proven, that the spider has 8 legs
theorem spider_has_eight_legs : spider_legs = 8 :=
by
  sorry

end spider_has_eight_legs_l51_5162


namespace john_sixth_quiz_score_l51_5113

noncomputable def sixth_quiz_score_needed : ℤ :=
  let scores := [86, 91, 88, 84, 97]
  let desired_average := 95
  let number_of_quizzes := 6
  let total_score_needed := number_of_quizzes * desired_average
  let total_score_so_far := scores.sum
  total_score_needed - total_score_so_far

theorem john_sixth_quiz_score :
  sixth_quiz_score_needed = 124 := 
by
  sorry

end john_sixth_quiz_score_l51_5113


namespace triangle_area_l51_5190

theorem triangle_area 
  (DE EL EF : ℝ)
  (hDE : DE = 14)
  (hEL : EL = 9)
  (hEF : EF = 17)
  (DL : ℝ)
  (hDL : DE^2 = DL^2 + EL^2)
  (hDL_val : DL = Real.sqrt 115):
  (1/2) * EF * DL = 17 * Real.sqrt 115 / 2 :=
by
  -- Sorry, as the proof is not required.
  sorry

end triangle_area_l51_5190


namespace coin_flip_probability_l51_5104

theorem coin_flip_probability (P : ℕ → ℕ → ℚ) (n : ℕ) :
  (∀ k, P k 0 = 1/2) →
  (∀ k, P k 1 = 1/2) →
  (∀ k m, P k m = 1/2) →
  n = 3 →
  P 0 0 * P 1 1 * P 2 1 = 1/8 :=
by
  intros h0 h1 h_indep hn
  sorry

end coin_flip_probability_l51_5104


namespace num_four_digit_integers_with_3_and_6_l51_5151

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l51_5151


namespace find_grazing_months_l51_5194

def oxen_months_A := 10 * 7
def oxen_months_B := 12 * 5
def total_rent := 175
def rent_C := 45

def proportion_equation (x : ℕ) : Prop :=
  45 / 175 = (15 * x) / (oxen_months_A + oxen_months_B + 15 * x)

theorem find_grazing_months (x : ℕ) (h : proportion_equation x) : x = 3 :=
by
  -- We will need to involve some calculations leading to x = 3
  sorry

end find_grazing_months_l51_5194


namespace sphere_radius_eq_3_l51_5184

theorem sphere_radius_eq_3 (r : ℝ) (h : (4/3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_eq_3_l51_5184


namespace avg_seven_consecutive_integers_l51_5176

variable (c d : ℕ)
variable (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_seven_consecutive_integers (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 :=
sorry

end avg_seven_consecutive_integers_l51_5176


namespace leila_armchairs_l51_5130

theorem leila_armchairs :
  ∀ {sofa_price armchair_price coffee_table_price total_invoice armchairs : ℕ},
  sofa_price = 1250 →
  armchair_price = 425 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  1 * sofa_price + armchairs * armchair_price + 1 * coffee_table_price = total_invoice →
  armchairs = 2 :=
by
  intros sofa_price armchair_price coffee_table_price total_invoice armchairs
  intros h1 h2 h3 h4 h_eq
  sorry

end leila_armchairs_l51_5130


namespace perfect_square_tens_digits_l51_5140

theorem perfect_square_tens_digits
  (a b : ℕ)
  (is_square_a : ∃ k : ℕ, a = k * k)
  (is_square_b : ∃ k : ℕ, b = k * k)
  (units_digit_a : a % 10 = 1)
  (tens_digit_a : ∃ x : ℕ, a / 10 % 10 = x)
  (units_digit_b : b % 10 = 6)
  (tens_digit_b : ∃ y : ℕ, b / 10 % 10 = y) :
  ∃ x y : ℕ, (a / 10 % 10 = x) ∧ (b / 10 % 10 = y) ∧ (x % 2 = 0) ∧ (y % 2 = 1) :=
sorry

end perfect_square_tens_digits_l51_5140


namespace total_books_l51_5107

-- Definitions based on the conditions
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52
def AlexBooks : ℕ := 65

-- Theorem to be proven
theorem total_books : TimBooks + SamBooks + AlexBooks = 161 := by
  sorry

end total_books_l51_5107


namespace line_circle_interaction_l51_5103

theorem line_circle_interaction (a : ℝ) :
  let r := 10
  let d := |a| / 5
  let intersects := -50 < a ∧ a < 50 
  let tangent := a = 50 ∨ a = -50 
  let separate := a < -50 ∨ a > 50 
  (d < r ↔ intersects) ∧ (d = r ↔ tangent) ∧ (d > r ↔ separate) :=
by sorry

end line_circle_interaction_l51_5103


namespace sum_of_first_4_terms_l51_5160

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_first_4_terms (a r : ℝ) 
  (h1 : a * (1 + r + r^2) = 13) (h2 : a * (1 + r + r^2 + r^3 + r^4) = 121) : 
  a * (1 + r + r^2 + r^3) = 27.857 :=
by
  sorry

end sum_of_first_4_terms_l51_5160


namespace line_ellipse_tangent_l51_5164

theorem line_ellipse_tangent (m : ℝ) (h : ∃ x y : ℝ, y = 2 * m * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) :
  m^2 = 3 / 16 :=
sorry

end line_ellipse_tangent_l51_5164


namespace books_read_l51_5126

-- Definitions
def total_books : ℕ := 13
def unread_books : ℕ := 4

-- Theorem
theorem books_read : total_books - unread_books = 9 :=
by
  sorry

end books_read_l51_5126


namespace no_three_distinct_positive_perfect_squares_sum_to_100_l51_5154

theorem no_three_distinct_positive_perfect_squares_sum_to_100 :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ (m n p : ℕ), a = m^2 ∧ b = n^2 ∧ c = p^2) ∧ a + b + c = 100 :=
by
  sorry

end no_three_distinct_positive_perfect_squares_sum_to_100_l51_5154


namespace inequality_proof_l51_5161

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : x1 > 0) (hx2 : x2 > 0) (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hx1y1_pos : x1 * y1 - z1^2 > 0) (hx2y2_pos : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 
    1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
by
  sorry

end inequality_proof_l51_5161


namespace original_data_props_l51_5100

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {new_x : Fin n → ℝ} 

noncomputable def average (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => data i)) / n

noncomputable def variance (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (data i - average data) ^ 2)) / n

-- Conditions
def condition1 (x new_x : Fin n → ℝ) (h : ∀ i, new_x i = x i - 80) : Prop := true

def condition2 (new_x : Fin n → ℝ) : Prop :=
  average new_x = 1.2

def condition3 (new_x : Fin n → ℝ) : Prop :=
  variance new_x = 4.4

theorem original_data_props (h : ∀ i, new_x i = x i - 80)
  (h_avg : average new_x = 1.2) 
  (h_var : variance new_x = 4.4) :
  average x = 81.2 ∧ variance x = 4.4 :=
sorry

end original_data_props_l51_5100


namespace philip_oranges_count_l51_5129

def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def betty_bill_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * betty_bill_oranges
def seeds_planted := frank_oranges * 2
def orange_trees := seeds_planted
def oranges_per_tree : ℕ := 5
def oranges_for_philip := orange_trees * oranges_per_tree

theorem philip_oranges_count : oranges_for_philip = 810 := by sorry

end philip_oranges_count_l51_5129


namespace laura_owes_amount_l51_5185

def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1
def interest (P R T : ℝ) := P * R * T
def totalAmountOwed (P I : ℝ) := P + I

theorem laura_owes_amount : totalAmountOwed principal (interest principal rate time) = 36.75 :=
by
  sorry

end laura_owes_amount_l51_5185


namespace teacher_age_l51_5166

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_total : ℕ) 
  (h1 : avg_age_students = 14) (h2 : num_students = 50) (h3 : avg_age_with_teacher = 15) (h4 : num_total = 51) :
  ∃ (teacher_age : ℕ), teacher_age = 65 :=
by sorry

end teacher_age_l51_5166


namespace find_D_l51_5144

-- Definitions from conditions
def is_different (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- The proof problem
theorem find_D (A B C D : ℕ) (h_diff: is_different A B C D) (h_eq : 700 + 10 * A + 5 + 100 * B + 70 + C = 100 * D + 38) : D = 9 :=
sorry

end find_D_l51_5144


namespace tangent_line_at_M_l51_5109

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

theorem tangent_line_at_M :
  let M : ℝ × ℝ := (2, 0)
  ∃ (m n : ℝ), n = f m ∧ m = 4 ∧ n = -2 * Real.exp 4 ∧
    ∀ (x y : ℝ), y = -Real.exp 4 * (x - 2) →
    M.2 = y :=
by
  sorry

end tangent_line_at_M_l51_5109


namespace square_area_multiplier_l51_5192

theorem square_area_multiplier 
  (perimeter_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (perimeter_square_eq : perimeter_square = 800) 
  (length_rectangle_eq : length_rectangle = 125) 
  (width_rectangle_eq : width_rectangle = 64)
  : (perimeter_square / 4) ^ 2 / (length_rectangle * width_rectangle) = 5 := 
by
  sorry

end square_area_multiplier_l51_5192


namespace inner_rectangle_length_is_4_l51_5150

-- Define the conditions
def inner_rectangle_width : ℝ := 2
def shaded_region_width : ℝ := 2

-- Define the lengths and areas of the respective regions
def inner_rectangle_length (x : ℝ) : ℝ := x
def second_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 4, 6)
def largest_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 8, 10)

def inner_rectangle_area (x : ℝ) : ℝ := inner_rectangle_length x * inner_rectangle_width
def second_rectangle_area (x : ℝ) : ℝ := (second_rectangle_dimensions x).1 * (second_rectangle_dimensions x).2
def largest_rectangle_area (x : ℝ) : ℝ := (largest_rectangle_dimensions x).1 * (largest_rectangle_dimensions x).2

def first_shaded_region_area (x : ℝ) : ℝ := second_rectangle_area x - inner_rectangle_area x
def second_shaded_region_area (x : ℝ) : ℝ := largest_rectangle_area x - second_rectangle_area x

-- Define the arithmetic progression condition
def arithmetic_progression (x : ℝ) : Prop :=
  (first_shaded_region_area x - inner_rectangle_area x) = (second_shaded_region_area x - first_shaded_region_area x)

-- State the theorem
theorem inner_rectangle_length_is_4 :
  ∃ x : ℝ, arithmetic_progression x ∧ inner_rectangle_length x = 4 := 
by
  use 4
  -- Proof goes here
  sorry

end inner_rectangle_length_is_4_l51_5150


namespace basketball_rim_height_l51_5120

theorem basketball_rim_height
    (height_in_inches : ℕ)
    (reach_in_inches : ℕ)
    (jump_in_inches : ℕ)
    (above_rim_in_inches : ℕ) :
    height_in_inches = 72
    → reach_in_inches = 22
    → jump_in_inches = 32
    → above_rim_in_inches = 6
    → (height_in_inches + reach_in_inches + jump_in_inches - above_rim_in_inches) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end basketball_rim_height_l51_5120


namespace find_abc_l51_5137

open Real

theorem find_abc {a b c : ℝ}
  (h1 : b + c = 16)
  (h2 : c + a = 17)
  (h3 : a + b = 18) :
  a * b * c = 606.375 :=
sorry

end find_abc_l51_5137


namespace reggie_loses_by_21_points_l51_5182

-- Define the points for each type of shot.
def layup_points := 1
def free_throw_points := 2
def three_pointer_points := 3
def half_court_points := 5

-- Define Reggie's shot counts.
def reggie_layups := 4
def reggie_free_throws := 3
def reggie_three_pointers := 2
def reggie_half_court_shots := 1

-- Define Reggie's brother's shot counts.
def brother_layups := 3
def brother_free_throws := 2
def brother_three_pointers := 5
def brother_half_court_shots := 4

-- Calculate Reggie's total points.
def reggie_total_points :=
  reggie_layups * layup_points +
  reggie_free_throws * free_throw_points +
  reggie_three_pointers * three_pointer_points +
  reggie_half_court_shots * half_court_points

-- Calculate Reggie's brother's total points.
def brother_total_points :=
  brother_layups * layup_points +
  brother_free_throws * free_throw_points +
  brother_three_pointers * three_pointer_points +
  brother_half_court_shots * half_court_points

-- Calculate the difference in points.
def point_difference := brother_total_points - reggie_total_points

-- Prove that the difference in points Reggie lost by is 21.
theorem reggie_loses_by_21_points : point_difference = 21 := by
  sorry

end reggie_loses_by_21_points_l51_5182


namespace sum_of_prime_factors_1320_l51_5118

theorem sum_of_prime_factors_1320 : 
  let smallest_prime := 2
  let largest_prime := 11
  smallest_prime + largest_prime = 13 :=
by
  sorry

end sum_of_prime_factors_1320_l51_5118


namespace D_is_largest_l51_5127

def D := (2008 / 2007) + (2008 / 2009)
def E := (2008 / 2009) + (2010 / 2009)
def F := (2009 / 2008) + (2009 / 2010) - (1 / 2009)

theorem D_is_largest : D > E ∧ D > F := by
  sorry

end D_is_largest_l51_5127


namespace digits_difference_l51_5199

theorem digits_difference (X Y : ℕ) (h : 10 * X + Y - (10 * Y + X) = 90) : X - Y = 10 :=
by
  sorry

end digits_difference_l51_5199


namespace rosa_called_pages_sum_l51_5102

theorem rosa_called_pages_sum :
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  sorry  -- proof will be done here

end rosa_called_pages_sum_l51_5102


namespace find_largest_number_l51_5148

theorem find_largest_number 
  (a b c : ℕ) 
  (h1 : a + b = 16) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) : 
  c = 19 := 
sorry

end find_largest_number_l51_5148


namespace domain_of_composite_function_l51_5134

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 → f x = f x) →
  (∀ (x : ℝ), -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → f (x^2) = f (x^2)) :=
by
  sorry

end domain_of_composite_function_l51_5134


namespace length_of_rectangle_l51_5123

-- Definitions based on conditions:
def side_length_square : ℝ := 4
def width_rectangle : ℝ := 8
def area_square (side : ℝ) : ℝ := side * side
def area_rectangle (width length : ℝ) : ℝ := width * length

-- The goal is to prove the length of the rectangle
theorem length_of_rectangle :
  (area_square side_length_square) = (area_rectangle width_rectangle 2) :=
by
  sorry

end length_of_rectangle_l51_5123


namespace find_divisor_l51_5186

theorem find_divisor (x : ℝ) (h : 740 / x - 175 = 10) : x = 4 := by
  sorry

end find_divisor_l51_5186


namespace amount_received_by_Sam_l51_5167

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem amount_received_by_Sam 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hP : P = 12000) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  final_amount P r n t = 12607.50 :=
by
  sorry

end amount_received_by_Sam_l51_5167


namespace min_airlines_needed_l51_5183

theorem min_airlines_needed 
  (towns : Finset ℕ) 
  (h_towns : towns.card = 21)
  (flights : Π (a : Finset ℕ), a.card = 5 → Finset (Finset ℕ))
  (h_flight : ∀ {a : Finset ℕ} (ha : a.card = 5), (flights a ha).card = 10):
  ∃ (n : ℕ), n = 21 :=
sorry

end min_airlines_needed_l51_5183


namespace number_of_students_on_wednesday_l51_5108

-- Define the problem conditions
variables (W T : ℕ)

-- Define the given conditions
def condition1 : Prop := T = W - 9
def condition2 : Prop := W + T = 65

-- Define the theorem to prove
theorem number_of_students_on_wednesday (h1 : condition1 W T) (h2 : condition2 W T) : W = 37 :=
by
  sorry

end number_of_students_on_wednesday_l51_5108


namespace arithmetic_sequence_general_formula_l51_5135

noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 3

theorem arithmetic_sequence_general_formula
    (a : ℕ → ℤ)
    (h1 : (a 2 + a 6) / 2 = 5)
    (h2 : (a 3 + a 7) / 2 = 7) :
  arithmetic_sequence a :=
by
  sorry

end arithmetic_sequence_general_formula_l51_5135


namespace arithmetic_sequence_tenth_term_l51_5158

theorem arithmetic_sequence_tenth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 6 * d = 13) :
  a + 9 * d = 19 := 
sorry

end arithmetic_sequence_tenth_term_l51_5158


namespace intersection_of_sets_l51_5188

noncomputable def set_A := {x : ℝ | Real.log x ≥ 0}
noncomputable def set_B := {x : ℝ | x^2 < 9}

theorem intersection_of_sets :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by {
  sorry
}

end intersection_of_sets_l51_5188


namespace simplify_expression_l51_5170

theorem simplify_expression :
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 125 / 13 :=
by
  sorry

end simplify_expression_l51_5170


namespace field_area_l51_5155

-- Define a rectangular field
structure RectangularField where
  length : ℕ
  width : ℕ
  fencing : ℕ := 2 * width + length
  
-- Given conditions
def field_conditions (L W F : ℕ) : Prop :=
  L = 30 ∧ 2 * W + L = F

-- Theorem stating the required proof
theorem field_area : ∀ (L W F : ℕ), field_conditions L W F → F = 84 → (L * W) = 810 :=
by
  intros L W F h1 h2
  sorry

end field_area_l51_5155


namespace find_p_l51_5111

variables (m n p : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x = y / 3 - 2 / 5

theorem find_p
  (h1 : line_equation m n)
  (h2 : line_equation (m + p) (n + 9)) :
  p = 3 :=
by
  sorry

end find_p_l51_5111


namespace num_ways_to_assign_grades_l51_5121

theorem num_ways_to_assign_grades : (4 ^ 12) = 16777216 := by
  sorry

end num_ways_to_assign_grades_l51_5121


namespace moles_required_to_form_2_moles_H2O_l51_5101

def moles_of_NH4NO3_needed (moles_of_H2O : ℕ) : ℕ := moles_of_H2O

theorem moles_required_to_form_2_moles_H2O :
  moles_of_NH4NO3_needed 2 = 2 := 
by 
  -- From the balanced equation 1 mole of NH4NO3 produces 1 mole of H2O
  -- Therefore, 2 moles of NH4NO3 are needed to produce 2 moles of H2O
  sorry

end moles_required_to_form_2_moles_H2O_l51_5101


namespace prism_volume_l51_5153

theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) : a * b * c = 1470 := by
  sorry

end prism_volume_l51_5153


namespace cos_equality_l51_5143

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem cos_equality : ∃ n : ℝ, (0 ≤ n ∧ n ≤ 180) ∧ Real.cos (degrees_to_radians n) = Real.cos (degrees_to_radians 317) :=
by
  use 43
  simp [degrees_to_radians, Real.cos]
  sorry

end cos_equality_l51_5143


namespace fraction_equals_decimal_l51_5139

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l51_5139


namespace trapezoid_rectangle_ratio_l51_5142

noncomputable def area_ratio (a1 a2 r : ℝ) : ℝ := 
  if a2 = 0 then 0 else a1 / a2

theorem trapezoid_rectangle_ratio 
  (radius : ℝ) (AD BC : ℝ)
  (trapezoid_area rectangle_area : ℝ) :
  radius = 13 →
  AD = 10 →
  BC = 24 →
  area_ratio trapezoid_area rectangle_area = 1 / 2 ∨
  area_ratio trapezoid_area rectangle_area = 289 / 338 :=
  sorry

end trapezoid_rectangle_ratio_l51_5142


namespace find_sum_of_A_and_B_l51_5173

theorem find_sum_of_A_and_B :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ B = A - 2 ∧ A = 5 + 3 ∧ A + B = 14 :=
by
  sorry

end find_sum_of_A_and_B_l51_5173


namespace bobby_initial_blocks_l51_5138

variable (b : ℕ)

theorem bobby_initial_blocks
  (h : b + 6 = 8) : b = 2 := by
  sorry

end bobby_initial_blocks_l51_5138


namespace dividend_is_5336_l51_5157

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) : 
  D * Q + R = 5336 := 
by sorry

end dividend_is_5336_l51_5157


namespace inequality_always_holds_l51_5191

theorem inequality_always_holds (x b : ℝ) (h : ∀ x : ℝ, x^2 + b * x + b > 0) : 0 < b ∧ b < 4 :=
sorry

end inequality_always_holds_l51_5191


namespace value_of_p_l51_5116

-- Let us assume the conditions given, and the existence of positive values p and q such that p + q = 1,
-- and the second term and fourth term of the polynomial expansion (x + y)^10 are equal when x = p and y = q.

theorem value_of_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h_sum : p + q = 1) (h_eq_terms : 10 * p ^ 9 * q = 120 * p ^ 7 * q ^ 3) :
    p = Real.sqrt (12 / 13) :=
    by sorry

end value_of_p_l51_5116


namespace bus_driver_total_hours_l51_5146

def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_compensation : ℝ := 976
def max_regular_hours : ℝ := 40

theorem bus_driver_total_hours :
  ∃ (hours_worked : ℝ), 
  (hours_worked = max_regular_hours + (total_compensation - (regular_rate * max_regular_hours)) / overtime_rate) ∧
  hours_worked = 52 :=
by
  sorry

end bus_driver_total_hours_l51_5146


namespace value_of_function_at_2_l51_5117

theorem value_of_function_at_2 (q : ℝ → ℝ) : q 2 = 5 :=
by
  -- Condition: The point (2, 5) lies on the graph of q
  have point_on_graph : q 2 = 5 := sorry
  exact point_on_graph

end value_of_function_at_2_l51_5117


namespace ratio_pow_eq_l51_5156

variable (a b c d e f p q r : ℝ)
variable (n : ℕ)
variable (h : a / b = c / d)
variable (h1 : a / b = e / f)
variable (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)

theorem ratio_pow_eq
  (h : a / b = c / d)
  (h1 : a / b = e / f)
  (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)
  (n_ne_zero : n ≠ 0):
  (a / b) ^ n = (p * a ^ n + q * c ^ n + r * e ^ n) / (p * b ^ n + q * d ^ n + r * f ^ n) :=
by
  sorry

end ratio_pow_eq_l51_5156


namespace value_of_x_plus_y_squared_l51_5180

variable (x y : ℝ)

def condition1 : Prop := x * (x + y) = 40
def condition2 : Prop := y * (x + y) = 90
def condition3 : Prop := x - y = 5

theorem value_of_x_plus_y_squared (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : (x + y) ^ 2 = 130 :=
by
  sorry

end value_of_x_plus_y_squared_l51_5180


namespace find_a_and_b_l51_5159

noncomputable def find_ab (a b : ℝ) : Prop :=
  (3 - 2 * a + b = 0) ∧
  (27 + 6 * a + b = 0)

theorem find_a_and_b :
  ∃ (a b : ℝ), (find_ab a b) ∧ (a = -3) ∧ (b = -9) :=
by
  sorry

end find_a_and_b_l51_5159


namespace sum_of_a_b_l51_5105

theorem sum_of_a_b (a b : ℝ) (h₁ : a^3 - 3 * a^2 + 5 * a = 1) (h₂ : b^3 - 3 * b^2 + 5 * b = 5) : a + b = 2 :=
sorry

end sum_of_a_b_l51_5105


namespace people_came_in_first_hour_l51_5181
-- Import the entirety of the necessary library

-- Lean 4 statement for the given problem
theorem people_came_in_first_hour (X : ℕ) (net_change_first_hour : ℕ) (net_change_second_hour : ℕ) (people_after_2_hours : ℕ) : 
    (net_change_first_hour = X - 27) → 
    (net_change_second_hour = 18 - 9) →
    (people_after_2_hours = 76) → 
    (X - 27 + 9 = 76) → 
    X = 94 :=
by 
    intros h1 h2 h3 h4 
    sorry -- Proof is not required by instructions

end people_came_in_first_hour_l51_5181


namespace largest_positive_x_l51_5147

def largest_positive_solution : ℝ := 1

theorem largest_positive_x 
  (x : ℝ) 
  (h : (2 * x^3 - x^2 - x + 1) ^ (1 + 1 / (2 * x + 1)) = 1) : 
  x ≤ largest_positive_solution := 
sorry

end largest_positive_x_l51_5147


namespace total_foreign_objects_l51_5132

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end total_foreign_objects_l51_5132


namespace sin_alpha_expression_l51_5125

theorem sin_alpha_expression (α : ℝ) 
  (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 := 
sorry

end sin_alpha_expression_l51_5125


namespace months_b_after_a_started_business_l51_5115

theorem months_b_after_a_started_business
  (A_initial : ℝ)
  (B_initial : ℝ)
  (profit_ratio : ℝ)
  (A_investment_time : ℕ)
  (B_investment_time : ℕ)
  (investment_ratio : A_initial * A_investment_time / (B_initial * B_investment_time) = profit_ratio) :
  B_investment_time = 6 :=
by
  -- Given:
  -- A_initial = 3500
  -- B_initial = 10500
  -- profit_ratio = 2 / 3
  -- A_investment_time = 12 months
  -- B_investment_time = 12 - x months
  -- We need to prove that x = 6 months such that investment ratio matches profit ratio.
  sorry

end months_b_after_a_started_business_l51_5115


namespace simplify_fraction_l51_5178

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l51_5178


namespace intersecting_circles_range_l51_5175

theorem intersecting_circles_range {k : ℝ} (a b : ℝ) :
  (-36 : ℝ) ≤ k ∧ k ≤ 104 →
  (∃ (x y : ℝ), (x^2 + y^2 - 4 - 12 * x + 6 * y) = 0 ∧ (x^2 + y^2 = k + 4 * x + 12 * y)) →
  b - a = (140 : ℝ) :=
by
  intro hk hab
  sorry

end intersecting_circles_range_l51_5175


namespace lisa_need_add_pure_juice_l51_5187

theorem lisa_need_add_pure_juice
  (x : ℝ) 
  (total_volume : ℝ := 2)
  (initial_pure_juice_fraction : ℝ := 0.10)
  (desired_pure_juice_fraction : ℝ := 0.25) 
  (added_pure_juice : ℝ := x) 
  (initial_pure_juice_amount : ℝ := total_volume * initial_pure_juice_fraction)
  (final_pure_juice_amount : ℝ := initial_pure_juice_amount + added_pure_juice)
  (final_volume : ℝ := total_volume + added_pure_juice) :
  (final_pure_juice_amount / final_volume) = desired_pure_juice_fraction → x = 0.4 :=
by
  intro h
  sorry

end lisa_need_add_pure_juice_l51_5187


namespace bags_filled_on_saturday_l51_5145

-- Definitions of the conditions
def bags_sat (S : ℕ) := S
def bags_sun := 4
def cans_per_bag := 9
def total_cans := 63

-- The statement to prove
theorem bags_filled_on_saturday (S : ℕ) 
  (h : total_cans = (bags_sat S + bags_sun) * cans_per_bag) : 
  S = 3 :=
by sorry

end bags_filled_on_saturday_l51_5145


namespace roger_total_miles_l51_5174

def morning_miles : ℕ := 2
def evening_multiplicative_factor : ℕ := 5
def evening_miles := evening_multiplicative_factor * morning_miles
def third_session_subtract : ℕ := 1
def third_session_miles := (2 * morning_miles) - third_session_subtract
def total_miles := morning_miles + evening_miles + third_session_miles

theorem roger_total_miles : total_miles = 15 := by
  sorry

end roger_total_miles_l51_5174


namespace value_of_a_plus_b_l51_5165

variables (a b : ℝ)

theorem value_of_a_plus_b (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : a + b = 12 := 
by
  sorry

end value_of_a_plus_b_l51_5165


namespace sum_first_50_natural_numbers_l51_5110

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Prove that the sum of the first 50 natural numbers is 1275
theorem sum_first_50_natural_numbers : sum_natural 50 = 1275 := 
by
  -- Skipping proof details
  sorry

end sum_first_50_natural_numbers_l51_5110


namespace original_fish_count_l51_5198

def initial_fish_count (fish_taken_out : ℕ) (current_fish : ℕ) : ℕ :=
  fish_taken_out + current_fish

theorem original_fish_count :
  initial_fish_count 16 3 = 19 :=
by
  sorry

end original_fish_count_l51_5198


namespace cubic_inequality_solution_l51_5124

theorem cubic_inequality_solution (x : ℝ) (h : 0 ≤ x) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ 16 < x := 
by 
  sorry

end cubic_inequality_solution_l51_5124


namespace units_digit_of_a_l51_5195

theorem units_digit_of_a (a : ℕ) (ha : (∃ b : ℕ, 1 ≤ b ∧ b ≤ 9 ∧ (a*a / 10^1) % 10 = b)) : 
  ((a % 10 = 4) ∨ (a % 10 = 6)) :=
sorry

end units_digit_of_a_l51_5195


namespace Vishal_investment_percentage_more_than_Trishul_l51_5114

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end Vishal_investment_percentage_more_than_Trishul_l51_5114


namespace product_of_areas_eq_square_of_volume_l51_5128

-- define the dimensions of the prism
variables (x y z : ℝ)

-- define the areas of the faces as conditions
def top_area := x * y
def back_area := y * z
def lateral_face_area := z * x

-- define the product of the areas of the top, back, and one lateral face
def product_of_areas := (top_area x y) * (back_area y z) * (lateral_face_area z x)

-- define the volume of the prism
def volume := x * y * z

-- theorem to prove: product of areas equals square of the volume
theorem product_of_areas_eq_square_of_volume 
  (ht: top_area x y = x * y)
  (hb: back_area y z = y * z)
  (hl: lateral_face_area z x = z * x) :
  product_of_areas x y z = (volume x y z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l51_5128


namespace proof_stops_with_two_pizzas_l51_5193

/-- The number of stops with orders of two pizzas. -/
def stops_with_two_pizzas : ℕ := 2

theorem proof_stops_with_two_pizzas
  (total_pizzas : ℕ)
  (single_stops : ℕ)
  (two_pizza_stops : ℕ)
  (average_time : ℕ)
  (total_time : ℕ)
  (h1 : total_pizzas = 12)
  (h2 : two_pizza_stops * 2 + single_stops = total_pizzas)
  (h3 : total_time = 40)
  (h4 : average_time = 4)
  (h5 : two_pizza_stops + single_stops = total_time / average_time) :
  two_pizza_stops = stops_with_two_pizzas := 
sorry

end proof_stops_with_two_pizzas_l51_5193


namespace min_value_inequality_l51_5168

open Real

theorem min_value_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 9) :
  ( (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ) ≥ 9 :=
sorry

end min_value_inequality_l51_5168


namespace money_spent_l51_5112

def initial_money (Henry : Type) : ℤ := 11
def birthday_money (Henry : Type) : ℤ := 18
def final_money (Henry : Type) : ℤ := 19

theorem money_spent (Henry : Type) : (initial_money Henry + birthday_money Henry - final_money Henry = 10) := 
by sorry

end money_spent_l51_5112


namespace juan_marbles_eq_64_l51_5189

def connie_marbles : ℕ := 39
def juan_extra_marbles : ℕ := 25

theorem juan_marbles_eq_64 : (connie_marbles + juan_extra_marbles) = 64 :=
by
  -- definition and conditions handled above
  sorry

end juan_marbles_eq_64_l51_5189


namespace relationship_between_a_b_c_l51_5122

theorem relationship_between_a_b_c (a b c : ℕ) (h1 : a = 2^40) (h2 : b = 3^32) (h3 : c = 4^24) : a < c ∧ c < b := by
  -- Definitions as per conditions
  have ha : a = 32^8 := by sorry
  have hb : b = 81^8 := by sorry
  have hc : c = 64^8 := by sorry
  -- Comparisons involving the bases
  have h : 32 < 64 := by sorry
  have h' : 64 < 81 := by sorry
  -- Resultant comparison
  exact ⟨by sorry, by sorry⟩

end relationship_between_a_b_c_l51_5122


namespace find_k_value_l51_5149

theorem find_k_value
  (x y k : ℝ)
  (h1 : 4 * x + 3 * y = 1)
  (h2 : k * x + (k - 1) * y = 3)
  (h3 : x = y) :
  k = 11 :=
  sorry

end find_k_value_l51_5149


namespace series_sum_equals_seven_ninths_l51_5106

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l51_5106


namespace stratified_sampling_by_edu_stage_is_reasonable_l51_5179

variable (visionConditions : String → Type) -- visionConditions for different sampling methods
variable (primaryVision : Type) -- vision condition for primary school
variable (juniorVision : Type) -- vision condition for junior high school
variable (seniorVision : Type) -- vision condition for senior high school
variable (insignificantDiffGender : Prop) -- insignificant differences between boys and girls

-- Given conditions
variable (sigDiffEduStage : Prop) -- significant differences between educational stages

-- Stating the theorem
theorem stratified_sampling_by_edu_stage_is_reasonable (h1 : sigDiffEduStage) (h2 : insignificantDiffGender) : 
  visionConditions "Stratified_sampling_by_educational_stage" = visionConditions C :=
sorry

end stratified_sampling_by_edu_stage_is_reasonable_l51_5179


namespace no_quadratic_polynomials_f_g_l51_5197

theorem no_quadratic_polynomials_f_g (f g : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h, ∀ x, g x = d * x^2 + e * x + h) : 
  ¬ (∀ x, f (g x) = x^4 - 3 * x^3 + 3 * x^2 - x) :=
by
  sorry

end no_quadratic_polynomials_f_g_l51_5197


namespace triangle_area_l51_5163

theorem triangle_area : 
  let p1 := (3, 2)
  let p2 := (3, -4)
  let p3 := (12, 2)
  let height := |2 - (-4)|
  let base := |12 - 3|
  let area := (1 / 2) * base * height
  area = 27 := sorry

end triangle_area_l51_5163


namespace total_amount_l51_5171

-- Conditions as given definitions
def ratio_a : Nat := 2
def ratio_b : Nat := 3
def ratio_c : Nat := 4
def share_b : Nat := 1500

-- The final statement
theorem total_amount (parts_b := 3) (one_part := share_b / parts_b) :
  (2 * one_part) + (3 * one_part) + (4 * one_part) = 4500 :=
by
  sorry

end total_amount_l51_5171


namespace tank_length_is_25_l51_5152

noncomputable def cost_to_paise (cost_in_rupees : ℕ) : ℕ :=
  cost_in_rupees * 100

noncomputable def total_area_plastered (total_cost_in_paise : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  total_cost_in_paise / cost_per_sq_m

noncomputable def length_of_tank (width height cost_in_rupees rate : ℕ) : ℕ :=
  let total_cost_in_paise := cost_to_paise cost_in_rupees
  let total_area := total_area_plastered total_cost_in_paise rate
  let area_eq := total_area = (2 * (height * width) + 2 * (6 * height) + (height * width))
  let simplified_eq := total_area - 144 = 24 * height
  (total_area - 144) / 24

theorem tank_length_is_25 (width height cost_in_rupees rate : ℕ) : 
  width = 12 → height = 6 → cost_in_rupees = 186 → rate = 25 → length_of_tank width height cost_in_rupees rate = 25 :=
  by
    intros hwidth hheight hcost hrate
    unfold length_of_tank
    rw [hwidth, hheight, hcost, hrate]
    simp
    sorry

end tank_length_is_25_l51_5152


namespace necessary_condition_for_acute_angle_l51_5196

-- Defining vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (x - 3, 2)
def vec_b : ℝ × ℝ := (1, 1)

-- Condition for the dot product to be positive
def dot_product_positive (x : ℝ) : Prop :=
  let (ax1, ax2) := vec_a x
  let (bx1, bx2) := vec_b
  ax1 * bx1 + ax2 * bx2 > 0

-- Statement for necessary condition
theorem necessary_condition_for_acute_angle (x : ℝ) :
  (dot_product_positive x) → (1 < x) :=
sorry

end necessary_condition_for_acute_angle_l51_5196


namespace persons_in_first_group_l51_5141

-- Define the given conditions
def first_group_work_done (P : ℕ) : ℕ := P * 12 * 10
def second_group_work_done : ℕ := 30 * 26 * 6

-- Define the proof problem statement
theorem persons_in_first_group (P : ℕ) (h : first_group_work_done P = second_group_work_done) : P = 39 :=
by
  unfold first_group_work_done second_group_work_done at h
  sorry

end persons_in_first_group_l51_5141


namespace machine_work_rate_l51_5169

theorem machine_work_rate (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -6 ∧ x ≠ -1) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end machine_work_rate_l51_5169
