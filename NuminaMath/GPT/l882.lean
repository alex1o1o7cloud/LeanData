import Mathlib

namespace cube_cut_edges_l882_88211

theorem cube_cut_edges (original_edges new_edges_per_vertex vertices : ℕ) (h1 : original_edges = 12) (h2 : new_edges_per_vertex = 6) (h3 : vertices = 8) :
  original_edges + new_edges_per_vertex * vertices = 60 :=
by
  sorry

end cube_cut_edges_l882_88211


namespace snakes_in_cage_l882_88262

theorem snakes_in_cage (snakes_hiding : Nat) (snakes_not_hiding : Nat) (total_snakes : Nat) 
  (h : snakes_hiding = 64) (nh : snakes_not_hiding = 31) : 
  total_snakes = snakes_hiding + snakes_not_hiding := by
  sorry

end snakes_in_cage_l882_88262


namespace min_circles_l882_88206

noncomputable def segments_intersecting_circles (N : ℕ) : Prop :=
  ∀ seg : (ℝ × ℝ) × ℝ, (seg.fst.fst ≥ 0 ∧ seg.fst.fst + seg.snd ≤ 100 ∧ seg.fst.snd ≥ 0 ∧ seg.fst.snd ≤ 100 ∧ seg.snd = 10) →
    ∃ c : ℝ × ℝ, (dist c seg.fst < 1 ∧ c.fst ≥ 0 ∧ c.fst ≤ 100 ∧ c.snd ≥ 0 ∧ c.snd ≤ 100) 

theorem min_circles (N : ℕ) (h : segments_intersecting_circles N) : N ≥ 400 :=
sorry

end min_circles_l882_88206


namespace simpsons_paradox_example_l882_88268

theorem simpsons_paradox_example :
  ∃ n1 n2 a1 a2 b1 b2,
    n1 = 10 ∧ a1 = 3 ∧ b1 = 2 ∧
    n2 = 90 ∧ a2 = 45 ∧ b2 = 488 ∧
    ((a1 : ℝ) / n1 > (b1 : ℝ) / n1) ∧
    ((a2 : ℝ) / n2 > (b2 : ℝ) / n2) ∧
    ((a1 + a2 : ℝ) / (n1 + n2) < (b1 + b2 : ℝ) / (n1 + n2)) :=
by
  use 10, 90, 3, 45, 2, 488
  simp
  sorry

end simpsons_paradox_example_l882_88268


namespace range_of_f_l882_88255

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (f x ≥ (Real.pi / 2 - Real.arctan 2) ∧ f x ≤ (Real.pi / 2 + Real.arctan 2)) :=
by
  sorry

end range_of_f_l882_88255


namespace functional_equation_solution_l882_88240

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (f x + f y)) = f x + y) : ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l882_88240


namespace g_2_eq_8_l882_88209

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_2_eq_8 : g 2 = 8 := 
by 
  unfold g
  unfold f_inv
  sorry

end g_2_eq_8_l882_88209


namespace percent_increase_second_half_century_l882_88291

variable (P : ℝ) -- Initial population
variable (x : ℝ) -- Percentage increase in the second half of the century

noncomputable def population_first_half_century := 3 * P
noncomputable def population_end_century := P + 11 * P

theorem percent_increase_second_half_century :
  3 * P + (x / 100) * (3 * P) = 12 * P → x = 300 :=
by
  intro h
  sorry

end percent_increase_second_half_century_l882_88291


namespace solve_abs_eqn_l882_88273

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) :=
by
  sorry

end solve_abs_eqn_l882_88273


namespace max_candy_received_l882_88241

theorem max_candy_received (students : ℕ) (candies : ℕ) (min_candy_per_student : ℕ) 
    (h_students : students = 40) (h_candies : candies = 200) (h_min_candy : min_candy_per_student = 2) :
    ∃ max_candy : ℕ, max_candy = 122 := by
  sorry

end max_candy_received_l882_88241


namespace range_of_a_l882_88235

variables {f : ℝ → ℝ} (a : ℝ)

-- Even function definition
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Monotonically increasing on (-∞, 0)
def mono_increasing_on_neg (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

-- Problem statement
theorem range_of_a
  (h_even : even_function f)
  (h_mono_neg : mono_increasing_on_neg f)
  (h_inequality : f (2 ^ |a - 1|) > f 4) :
  -1 < a ∧ a < 3 :=
sorry

end range_of_a_l882_88235


namespace airplane_seats_l882_88220

theorem airplane_seats (s : ℝ)
  (h1 : 0.30 * s = 0.30 * s)
  (h2 : (3 / 5) * s = (3 / 5) * s)
  (h3 : 36 + 0.30 * s + (3 / 5) * s = s) : s = 360 :=
by
  sorry

end airplane_seats_l882_88220


namespace abs_linear_combination_l882_88259

theorem abs_linear_combination (a b : ℝ) :
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) →
  (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0) :=
by {
  sorry
}

end abs_linear_combination_l882_88259


namespace absolute_value_of_neg_five_l882_88203

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l882_88203


namespace smallest_digit_to_make_divisible_by_9_l882_88248

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l882_88248


namespace range_of_m_l882_88289

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (m / (2*x - 1) + 3 = 0) ∧ (x > 0)) ↔ (m < 3 ∧ m ≠ 0) :=
by
  sorry

end range_of_m_l882_88289


namespace amount_deducted_from_third_l882_88287

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end amount_deducted_from_third_l882_88287


namespace smallest_disk_cover_count_l882_88216

theorem smallest_disk_cover_count (D : ℝ) (r : ℝ) (n : ℕ) 
  (hD : D = 1) (hr : r = 1 / 2) : n = 7 :=
by
  sorry

end smallest_disk_cover_count_l882_88216


namespace Jake_weight_l882_88252

variables (J S : ℝ)

theorem Jake_weight (h1 : 0.8 * J = 2 * S) (h2 : J + S = 168) : J = 120 :=
  sorry

end Jake_weight_l882_88252


namespace labourer_savings_l882_88232

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end labourer_savings_l882_88232


namespace regular_vs_diet_sodas_l882_88231

theorem regular_vs_diet_sodas :
  let regular_cola := 67
  let regular_lemon := 45
  let regular_orange := 23
  let diet_cola := 9
  let diet_lemon := 32
  let diet_orange := 12
  let regular_sodas := regular_cola + regular_lemon + regular_orange
  let diet_sodas := diet_cola + diet_lemon + diet_orange
  regular_sodas - diet_sodas = 82 := sorry

end regular_vs_diet_sodas_l882_88231


namespace solve_for_x_l882_88256

variable (x : ℝ)

theorem solve_for_x (h : 5 * x - 3 = 17) : x = 4 := sorry

end solve_for_x_l882_88256


namespace book_arrangement_count_l882_88296

-- Define the conditions
def total_books : ℕ := 6
def identical_books : ℕ := 3
def different_books : ℕ := total_books - identical_books

-- Prove the number of arrangements
theorem book_arrangement_count : (total_books.factorial / identical_books.factorial) = 120 := by
  sorry

end book_arrangement_count_l882_88296


namespace lines_through_origin_l882_88253

theorem lines_through_origin (n : ℕ) (h : 0 < n) :
    ∃ S : Finset (ℤ × ℤ), 
    (∀ xy : ℤ × ℤ, xy ∈ S ↔ (0 ≤ xy.1 ∧ xy.1 ≤ n ∧ 0 ≤ xy.2 ∧ xy.2 ≤ n ∧ Int.gcd xy.1 xy.2 = 1)) ∧
    S.card ≥ n^2 / 4 := 
sorry

end lines_through_origin_l882_88253


namespace solve_for_y_l882_88263

theorem solve_for_y (x y : ℝ) (h : (x + y)^5 - x^5 + y = 0) : y = 0 :=
sorry

end solve_for_y_l882_88263


namespace radius_of_circle_l882_88242

-- Define the given circle equation as a condition
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

theorem radius_of_circle : ∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, r = 3 :=
by
  sorry

end radius_of_circle_l882_88242


namespace max_value_of_quadratic_l882_88221

theorem max_value_of_quadratic :
  ∀ (x : ℝ), ∃ y : ℝ, y = -3 * x^2 + 18 ∧
  (∀ x' : ℝ, -3 * x'^2 + 18 ≤ y) := by
  sorry

end max_value_of_quadratic_l882_88221


namespace simplify_abs_expression_l882_88245

theorem simplify_abs_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := 
by
  sorry

end simplify_abs_expression_l882_88245


namespace polynomial_factorization_l882_88250

noncomputable def factorize_polynomial (a b : ℝ) : ℝ :=
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3

theorem polynomial_factorization (a b : ℝ) : 
  factorize_polynomial a b = -3 * a * b * (a - b)^2 := 
by
  sorry

end polynomial_factorization_l882_88250


namespace sale_price_correct_l882_88237

variable (x : ℝ)

-- Conditions
def decreased_price (x : ℝ) : ℝ :=
  0.9 * x

def final_sale_price (decreased_price : ℝ) : ℝ :=
  0.7 * decreased_price

-- Proof statement
theorem sale_price_correct : final_sale_price (decreased_price x) = 0.63 * x := by
  sorry

end sale_price_correct_l882_88237


namespace find_x_l882_88236

theorem find_x (a x : ℝ) (ha : 1 < a) (hx : 0 < x)
  (h : (3 * x)^(Real.log 3 / Real.log a) - (4 * x)^(Real.log 4 / Real.log a) = 0) : 
  x = 1 / 4 := 
by 
  sorry

end find_x_l882_88236


namespace Frank_has_four_one_dollar_bills_l882_88279

noncomputable def Frank_one_dollar_bills : ℕ :=
  let total_money := 4 * 5 + 2 * 10 + 20 -- Money from five, ten, and twenty dollar bills
  let peanuts_cost := 10 - 4 -- Cost of peanuts (given $10 and received $4 in change)
  let one_dollar_bills_value := 54 - total_money -- Total money Frank has - money from large bills
  (one_dollar_bills_value : ℕ)

theorem Frank_has_four_one_dollar_bills 
   (five_dollar_bills : ℕ := 4) 
   (ten_dollar_bills : ℕ := 2)
   (twenty_dollar_bills : ℕ := 1)
   (peanut_price : ℚ := 3)
   (change : ℕ := 4)
   (total_money : ℕ := 50)
   (total_money_incl_change : ℚ := 54):
   Frank_one_dollar_bills = 4 := by
  sorry

end Frank_has_four_one_dollar_bills_l882_88279


namespace ball_maximum_height_l882_88244
-- Import necessary libraries

-- Define the height function
def ball_height (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

-- Proposition asserting that the maximum height of the ball is 145 meters
theorem ball_maximum_height : ∃ t : ℝ, ball_height t = 145 :=
  sorry

end ball_maximum_height_l882_88244


namespace simplify_expression_to_fraction_l882_88277

theorem simplify_expression_to_fraction : 
  (1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5)) = 1/60 :=
by 
  have h1 : 1 / (1/2)^2 = 4 := by sorry
  have h2 : 1 / (1/2)^3 = 8 := by sorry
  have h3 : 1 / (1/2)^4 = 16 := by sorry
  have h4 : 1 / (1/2)^5 = 32 := by sorry
  have h5 : 4 + 8 + 16 + 32 = 60 := by sorry
  have h6 : 1 / 60 = 1/60 := by sorry
  sorry

end simplify_expression_to_fraction_l882_88277


namespace remainder_sum_of_numbers_l882_88272

theorem remainder_sum_of_numbers :
  ((123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7) = 5 :=
by
  sorry

end remainder_sum_of_numbers_l882_88272


namespace sum_of_three_consecutive_odd_integers_l882_88239

-- Define the variables and conditions
variables (a : ℤ) (h1 : (a + (a + 4) = 100))

-- Define the statement that needs to be proved
theorem sum_of_three_consecutive_odd_integers (ha : a = 48) : a + (a + 2) + (a + 4) = 150 := by
  sorry

end sum_of_three_consecutive_odd_integers_l882_88239


namespace hike_distance_l882_88275

theorem hike_distance :
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  stream_to_meadow = 0.4 :=
by
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  show stream_to_meadow = 0.4
  sorry

end hike_distance_l882_88275


namespace inscribed_triangle_area_is_12_l882_88258

noncomputable def area_of_triangle_in_inscribed_circle 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) : 
  ℝ := 
1 / 2 * (2 * (4 / 2)) * (3 * (4 / 2))

theorem inscribed_triangle_area_is_12 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) :
  area_of_triangle_in_inscribed_circle a b c h_ratio h_radius h_inscribed = 12 :=
sorry

end inscribed_triangle_area_is_12_l882_88258


namespace spending_difference_l882_88207

-- Define the cost of the candy bar
def candy_bar_cost : ℕ := 6

-- Define the cost of the chocolate
def chocolate_cost : ℕ := 3

-- Prove the difference between candy_bar_cost and chocolate_cost
theorem spending_difference : candy_bar_cost - chocolate_cost = 3 :=
by
    sorry

end spending_difference_l882_88207


namespace dot_product_theorem_l882_88281

open Real

namespace VectorProof

-- Define the vectors m and n
def m := (2, 5)
def n (t : ℝ) := (-5, t)

-- Define the condition that m is perpendicular to n
def perpendicular (t : ℝ) : Prop := (2 * -5) + (5 * t) = 0

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the vectors m+n and m-2n
def vector_add (t : ℝ) : ℝ × ℝ := (m.1 + (n t).1, m.2 + (n t).2)
def vector_sub (t : ℝ) : ℝ × ℝ := (m.1 - 2 * (n t).1, m.2 - 2 * (n t).2)

-- The theorem to prove
theorem dot_product_theorem : ∀ (t : ℝ), perpendicular t → dot_product (vector_add t) (vector_sub t) = -29 :=
by
  intros t ht
  sorry

end VectorProof

end dot_product_theorem_l882_88281


namespace jack_helped_hours_l882_88251

-- Definitions based on the problem's conditions
def sam_rate : ℕ := 6  -- Sam assembles 6 widgets per hour
def tony_rate : ℕ := 2  -- Tony assembles 2 widgets per hour
def jack_rate : ℕ := sam_rate  -- Jack assembles at the same rate as Sam
def total_widgets : ℕ := 68  -- The total number of widgets assembled by all three

-- Statement to prove
theorem jack_helped_hours : 
  ∃ h : ℕ, (sam_rate * h) + (tony_rate * h) + (jack_rate * h) = total_widgets ∧ h = 4 := 
  by
  -- The proof is not necessary; we only need the statement
  sorry

end jack_helped_hours_l882_88251


namespace male_female_ratio_l882_88205

-- Definitions and constants
variable (M F : ℕ) -- Number of male and female members respectively
variable (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) -- Average ticket sales condition

-- Statement of the theorem
theorem male_female_ratio (M F : ℕ) (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) : M / F = 1 / 2 :=
sorry

end male_female_ratio_l882_88205


namespace proof_problem_l882_88278

noncomputable def problem_statement (m : ℕ) : Prop :=
  ∀ pairs : List (ℕ × ℕ),
  (∀ (x y : ℕ), (x, y) ∈ pairs ↔ x^2 - 3 * y^2 + 2 = 16 * m ∧ 2 * y ≤ x - 1) →
  pairs.length % 2 = 0 ∨ pairs.length = 0

theorem proof_problem (m : ℕ) (hm : m > 0) : problem_statement m :=
by
  sorry

end proof_problem_l882_88278


namespace division_remainder_is_7_l882_88295

theorem division_remainder_is_7 (d q D r : ℕ) (hd : d = 21) (hq : q = 14) (hD : D = 301) (h_eq : D = d * q + r) : r = 7 :=
by
  sorry

end division_remainder_is_7_l882_88295


namespace point_below_line_l882_88228

theorem point_below_line {a : ℝ} (h : 2 * a - 3 < 3) : a < 3 :=
by {
  sorry
}

end point_below_line_l882_88228


namespace raj_earns_more_l882_88214

theorem raj_earns_more :
  let cost_per_sqft := 2
  let raj_length := 30
  let raj_width := 50
  let lena_length := 40
  let lena_width := 35
  let raj_area := raj_length * raj_width
  let lena_area := lena_length * lena_width
  let raj_earnings := raj_area * cost_per_sqft
  let lena_earnings := lena_area * cost_per_sqft
  raj_earnings - lena_earnings = 200 :=
by
  sorry

end raj_earns_more_l882_88214


namespace number_of_ways_to_choose_one_top_and_one_bottom_l882_88223

theorem number_of_ways_to_choose_one_top_and_one_bottom :
  let number_of_hoodies := 5
  let number_of_sweatshirts := 4
  let number_of_jeans := 3
  let number_of_slacks := 5
  let total_tops := number_of_hoodies + number_of_sweatshirts
  let total_bottoms := number_of_jeans + number_of_slacks
  total_tops * total_bottoms = 72 := 
by
  sorry

end number_of_ways_to_choose_one_top_and_one_bottom_l882_88223


namespace equidistant_point_quadrants_l882_88222

theorem equidistant_point_quadrants :
  ∀ (x y : ℝ), 3 * x + 5 * y = 15 → (|x| = |y| → (x > 0 → y > 0 ∧ x = y ∧ y = x) ∧ (x < 0 → y > 0 ∧ x = -y ∧ -x = y)) := 
by
  sorry

end equidistant_point_quadrants_l882_88222


namespace one_plane_halves_rect_prism_l882_88270

theorem one_plane_halves_rect_prism :
  ∀ (T : Type) (a b c : ℝ)
  (x y z : ℝ) 
  (black_prisms_volume white_prisms_volume : ℝ),
  (black_prisms_volume = (x * y * z + x * (b - y) * (c - z) + (a - x) * y * (c - z) + (a - x) * (b - y) * z)) ∧
  (white_prisms_volume = ((a - x) * (b - y) * (c - z) + (a - x) * y * z + x * (b - y) * z + x * y * (c - z))) ∧
  (black_prisms_volume = white_prisms_volume) →
  (x = a / 2 ∨ y = b / 2 ∨ z = c / 2) :=
by
  sorry

end one_plane_halves_rect_prism_l882_88270


namespace sum_geq_three_implies_one_geq_two_l882_88247

theorem sum_geq_three_implies_one_geq_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by { sorry }

end sum_geq_three_implies_one_geq_two_l882_88247


namespace toys_lost_l882_88226

theorem toys_lost (initial_toys found_in_closet total_after_finding : ℕ) 
  (h1 : initial_toys = 40) 
  (h2 : found_in_closet = 9) 
  (h3 : total_after_finding = 43) : 
  initial_toys - (total_after_finding - found_in_closet) = 9 :=
by 
  sorry

end toys_lost_l882_88226


namespace compute_fraction_power_l882_88238

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l882_88238


namespace shaded_areas_sum_l882_88282

theorem shaded_areas_sum (triangle_area : ℕ) (parts : ℕ)
  (h1 : triangle_area = 18)
  (h2 : parts = 9) :
  3 * (triangle_area / parts) = 6 :=
by
  sorry

end shaded_areas_sum_l882_88282


namespace three_digit_number_with_units5_and_hundreds3_divisible_by_9_l882_88264

theorem three_digit_number_with_units5_and_hundreds3_divisible_by_9 :
  ∃ n : ℕ, ∃ x : ℕ, n = 305 + 10 * x ∧ (n % 9) = 0 ∧ n = 315 := by
sorry

end three_digit_number_with_units5_and_hundreds3_divisible_by_9_l882_88264


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l882_88266

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l882_88266


namespace sum_of_largest_smallest_angles_l882_88261

noncomputable section

def sides_ratio (a b c : ℝ) : Prop := a / 5 = b / 7 ∧ b / 7 = c / 8

theorem sum_of_largest_smallest_angles (a b c : ℝ) (θA θB θC : ℝ) 
  (h1 : sides_ratio a b c) 
  (h2 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos θC)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * Real.cos θA)
  (h4 : c^2 + a^2 - b^2 = 2 * c * a * Real.cos θB)
  (h5 : θA + θB + θC = 180) :
  θA + θC = 120 :=
sorry

end sum_of_largest_smallest_angles_l882_88261


namespace not_sufficient_nor_necessary_condition_l882_88267

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_increasing_for_nonpositive (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ 0 → y ≤ 0 → x < y → f x < f y

theorem not_sufficient_nor_necessary_condition
  {f : ℝ → ℝ}
  (hf_even : is_even_function f)
  (hf_incr : is_increasing_for_nonpositive f)
  (x : ℝ) :
  (6/5 < x ∧ x < 2) → ¬((1 < x ∧ x < 7/4) ↔ (f (Real.log (2 * x - 2) / Real.log 2) > f (Real.log (2 / 3) / Real.log (1 / 2)))) :=
sorry

end not_sufficient_nor_necessary_condition_l882_88267


namespace find_positive_k_l882_88246

noncomputable def polynomial_with_equal_roots (k: ℚ) : Prop := 
  ∃ a b : ℚ, a ≠ b ∧ 2 * a + b = -3 ∧ 2 * a * b + a^2 = -50 ∧ k = -2 * a^2 * b

theorem find_positive_k : ∃ k : ℚ, polynomial_with_equal_roots k ∧ 0 < k ∧ k = 950 / 27 :=
by
  sorry

end find_positive_k_l882_88246


namespace range_of_m_l882_88274

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < |m - 2|) ↔ m < 0 ∨ m > 4 := 
sorry

end range_of_m_l882_88274


namespace determine_BD_l882_88227

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC CD DA : ℝ)
variables (BD : ℝ)

-- Setting up the conditions:
axiom AB_eq_5 : AB = 5
axiom BC_eq_17 : BC = 17
axiom CD_eq_5 : CD = 5
axiom DA_eq_9 : DA = 9
axiom BD_is_integer : ∃ (n : ℤ), BD = n

theorem determine_BD : BD = 13 :=
by
  sorry

end determine_BD_l882_88227


namespace monikaTotalSpending_l882_88288

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end monikaTotalSpending_l882_88288


namespace middle_part_division_l882_88254

theorem middle_part_division 
  (x : ℝ) 
  (x_pos : x > 0) 
  (H : x + (1 / 4) * x + (1 / 8) * x = 96) :
  (1 / 4) * x = 17 + 21 / 44 :=
by
  sorry

end middle_part_division_l882_88254


namespace percentage_decrease_l882_88260

theorem percentage_decrease (original_price new_price decrease: ℝ) (h₁: original_price = 2400) (h₂: new_price = 1200) (h₃: decrease = original_price - new_price): 
  decrease / original_price * 100 = 50 :=
by
  rw [h₁, h₂] at h₃ -- Update the decrease according to given prices
  sorry -- Left as a placeholder for the actual proof

end percentage_decrease_l882_88260


namespace complex_root_of_unity_prod_l882_88229

theorem complex_root_of_unity_prod (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 :=
by
  sorry

end complex_root_of_unity_prod_l882_88229


namespace math_problem_l882_88213

theorem math_problem :
  (-1:ℤ) ^ 2023 - |(-3:ℤ)| + ((-1/3:ℚ) ^ (-2:ℤ)) + ((Real.pi - 3.14)^0) = 6 := 
by 
  sorry

end math_problem_l882_88213


namespace ticket_cost_difference_l882_88224

theorem ticket_cost_difference
  (num_adults : ℕ) (num_children : ℕ)
  (cost_adult_ticket : ℕ) (cost_child_ticket : ℕ)
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : cost_adult_ticket = 11)
  (h4 : cost_child_ticket = 7) :
  num_adults * cost_adult_ticket - num_children * cost_child_ticket = 50 := 
by
  sorry

end ticket_cost_difference_l882_88224


namespace northton_time_capsule_depth_l882_88285

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end northton_time_capsule_depth_l882_88285


namespace total_marks_by_category_l882_88294

theorem total_marks_by_category 
  (num_candidates_A : ℕ) (num_candidates_B : ℕ) (num_candidates_C : ℕ)
  (avg_marks_A : ℕ) (avg_marks_B : ℕ) (avg_marks_C : ℕ) 
  (hA : num_candidates_A = 30) (hB : num_candidates_B = 25) (hC : num_candidates_C = 25)
  (h_avg_A : avg_marks_A = 35) (h_avg_B : avg_marks_B = 42) (h_avg_C : avg_marks_C = 46) :
  (num_candidates_A * avg_marks_A = 1050) ∧
  (num_candidates_B * avg_marks_B = 1050) ∧
  (num_candidates_C * avg_marks_C = 1150) := 
by
  sorry

end total_marks_by_category_l882_88294


namespace cannot_achieve_1970_minuses_l882_88298

theorem cannot_achieve_1970_minuses :
  ∃ (x y : ℕ), x ≤ 100 ∧ y ≤ 100 ∧ (x - 50) * (y - 50) = 1515 → false :=
by
  sorry

end cannot_achieve_1970_minuses_l882_88298


namespace peggy_records_l882_88217

theorem peggy_records (R : ℕ) (h : 4 * R - (3 * R + R / 2) = 100) : R = 200 :=
sorry

end peggy_records_l882_88217


namespace water_leaving_rate_l882_88293

-- Definitions: Volume of water and time taken
def volume_of_water : ℕ := 300
def time_taken : ℕ := 25

-- Theorem statement: Rate of water leaving the tank
theorem water_leaving_rate : (volume_of_water / time_taken) = 12 := 
by sorry

end water_leaving_rate_l882_88293


namespace land_area_of_each_section_l882_88286

theorem land_area_of_each_section (n : ℕ) (total_area : ℕ) (h1 : n = 3) (h2 : total_area = 7305) :
  total_area / n = 2435 :=
by {
  sorry
}

end land_area_of_each_section_l882_88286


namespace total_limes_picked_l882_88292

def Fred_limes : ℕ := 36
def Alyssa_limes : ℕ := 32
def Nancy_limes : ℕ := 35
def David_limes : ℕ := 42
def Eileen_limes : ℕ := 50

theorem total_limes_picked :
  Fred_limes + Alyssa_limes + Nancy_limes + David_limes + Eileen_limes = 195 :=
by
  sorry

end total_limes_picked_l882_88292


namespace min_value_a_plus_one_over_a_minus_one_l882_88299

theorem min_value_a_plus_one_over_a_minus_one (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ (a = 2 → a + 1 / (a - 1) = 3) :=
by
  -- Translate the mathematical proof problem into a Lean 4 theorem statement.
  sorry

end min_value_a_plus_one_over_a_minus_one_l882_88299


namespace negation_equivalence_l882_88219

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by
  sorry

end negation_equivalence_l882_88219


namespace blocks_left_l882_88200

def blocks_initial := 78
def blocks_used := 19

theorem blocks_left : blocks_initial - blocks_used = 59 :=
by
  -- Solution is not required here, so we add a sorry placeholder.
  sorry

end blocks_left_l882_88200


namespace total_kids_receive_macarons_l882_88234

theorem total_kids_receive_macarons :
  let mitch_good := 18
  let joshua := 26 -- 20 + 6
  let joshua_good := joshua - 3
  let miles := joshua * 2
  let miles_good := miles
  let renz := (3 * miles) / 4 - 1
  let renz_good := renz - 4
  let leah_good := 35 - 5
  let total_good := mitch_good + joshua_good + miles_good + renz_good + leah_good 
  let kids_with_3_macarons := 10
  let macaron_per_3 := kids_with_3_macarons * 3
  let remaining_macarons := total_good - macaron_per_3
  let kids_with_2_macarons := remaining_macarons / 2
  kids_with_3_macarons + kids_with_2_macarons = 73 :=
by 
  sorry

end total_kids_receive_macarons_l882_88234


namespace lines_parallel_iff_l882_88257

theorem lines_parallel_iff (a : ℝ) : (∀ x y : ℝ, x + 2*a*y - 1 = 0 ∧ (2*a - 1)*x - a*y - 1 = 0 → x = 1 ∧ x = -1 ∨ ∃ (slope : ℝ), slope = - (1 / (2 * a)) ∧ slope = (2 * a - 1) / a) ↔ (a = 0 ∨ a = 1/4) :=
by
  sorry

end lines_parallel_iff_l882_88257


namespace equal_candies_l882_88290

theorem equal_candies
  (sweet_math_per_box : ℕ := 12)
  (geometry_nuts_per_box : ℕ := 15)
  (sweet_math_boxes : ℕ := 5)
  (geometry_nuts_boxes : ℕ := 4) :
  sweet_math_boxes * sweet_math_per_box = geometry_nuts_boxes * geometry_nuts_per_box := 
  by
  sorry

end equal_candies_l882_88290


namespace secretary_longest_time_l882_88212

def ratio_times (x : ℕ) : Prop := 
  let t1 := 2 * x
  let t2 := 3 * x
  let t3 := 5 * x
  (t1 + t2 + t3 = 110) ∧ (t3 = 55)

theorem secretary_longest_time :
  ∃ x : ℕ, ratio_times x :=
sorry

end secretary_longest_time_l882_88212


namespace calculate_expression_l882_88271

theorem calculate_expression : (50 - (5020 - 520) + (5020 - (520 - 50))) = 100 := 
by
  sorry

end calculate_expression_l882_88271


namespace derivative_u_l882_88249

noncomputable def u (x : ℝ) : ℝ :=
  let z := Real.sin x
  let y := x^2
  Real.exp (z - 2 * y)

theorem derivative_u (x : ℝ) :
  deriv u x = Real.exp (Real.sin x - 2 * x^2) * (Real.cos x - 4 * x) :=
by
  sorry

end derivative_u_l882_88249


namespace faith_earnings_correct_l882_88283

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l882_88283


namespace problem_l882_88280

variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Define the acute triangle and constructions
axiom acute_triangle (ABC : Type) : Prop
axiom circumcircle (ABC : Type) (A1 B1 C1 : Type) : Prop
axiom extended_angle_bisectors (ABC : Type) (A0 B0 C0 : Type) : Prop

-- Define the points according to the problem statement
axiom intersections_A0 (ABC : Type) (A0 : Type) : Prop
axiom intersections_B0 (ABC : Type) (B0 : Type) : Prop
axiom intersections_C0 (ABC : Type) (C0 : Type) : Prop

-- Define the areas of triangles and hexagon
axiom area_triangle_A0B0C0 (ABC : Type) (A0 B0 C0 : Type) : ℝ
axiom area_hexagon_AC1B_A1CB1 (ABC : Type) (A1 B1 C1 : Type) : ℝ
axiom area_triangle_ABC (ABC : Type) : ℝ

-- Problem: Prove the area relationships
theorem problem
  (ABC: Type)
  (h1 : acute_triangle ABC)
  (h2 : circumcircle ABC A1 B1 C1)
  (h3 : extended_angle_bisectors ABC A0 B0 C0)
  (h4 : intersections_A0 ABC A0)
  (h5 : intersections_B0 ABC B0)
  (h6 : intersections_C0 ABC C0):
  area_triangle_A0B0C0 ABC A0 B0 C0 = 2 * area_hexagon_AC1B_A1CB1 ABC A1 B1 C1 ∧
  area_triangle_A0B0C0 ABC A0 B0 C0 ≥ 4 * area_triangle_ABC ABC :=
sorry

end problem_l882_88280


namespace horner_value_x_neg2_l882_88243

noncomputable def horner (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 0.3) * x + 2

theorem horner_value_x_neg2 : horner (-2) = -40 :=
by
  sorry

end horner_value_x_neg2_l882_88243


namespace correct_calculation_l882_88297

-- Definitions of the conditions
def condition1 : Prop := 3 + Real.sqrt 3 ≠ 3 * Real.sqrt 3
def condition2 : Prop := 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3
def condition3 : Prop := 2 * Real.sqrt 3 - Real.sqrt 3 ≠ 2
def condition4 : Prop := Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5

-- Proposition using the conditions to state the correct calculation
theorem correct_calculation (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 :=
by
  exact h2

end correct_calculation_l882_88297


namespace find_matrix_N_l882_88284

def matrix2x2 := ℚ × ℚ × ℚ × ℚ

def apply_matrix (M : matrix2x2) (v : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b, c, d) := M;
  let (x, y) := v;
  (a * x + b * y, c * x + d * y)

theorem find_matrix_N : ∃ (N : matrix2x2), 
  apply_matrix N (3, 1) = (5, -1) ∧ 
  apply_matrix N (1, -2) = (0, 6) ∧ 
  N = (10/7, 5/7, 4/7, -19/7) :=
by {
  sorry
}

end find_matrix_N_l882_88284


namespace solution_set_l882_88233

noncomputable def truncated_interval (x : ℝ) (n : ℤ) : Prop :=
n ≤ x ∧ x < n + 1

theorem solution_set (x : ℝ) (hx : ∃ n : ℤ, n > 0 ∧ truncated_interval x n) :
  2 ≤ x ∧ x < 8 :=
sorry

end solution_set_l882_88233


namespace scalene_triangle_process_l882_88225

theorem scalene_triangle_process (a b c : ℝ) 
  (h1: a > 0) (h2: b > 0) (h3: c > 0) 
  (h4: a + b > c) (h5: b + c > a) (h6: a + c > b) : 
  ¬(∃ k : ℝ, (k > 0) ∧ 
    ((k * a = a + b - c) ∧ 
     (k * b = b + c - a) ∧ 
     (k * c = a + c - b))) ∧ 
  (∀ n: ℕ, n > 0 → (a + b - c)^n + (b + c - a)^n + (a + c - b)^n < 1) :=
by
  sorry

end scalene_triangle_process_l882_88225


namespace general_formula_for_sequence_l882_88265

def sequence_terms (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1))

def seq_conditions (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
a 1 = 1 / 2 ∧ (∀ n : ℕ, n > 0 → S n = n^2 * a n)

theorem general_formula_for_sequence :
  ∃ a S : ℕ → ℚ, seq_conditions a S ∧ sequence_terms a := by
  sorry

end general_formula_for_sequence_l882_88265


namespace total_cost_of_two_books_l882_88210

theorem total_cost_of_two_books (C1 C2 total_cost: ℝ) :
  C1 = 262.5 →
  0.85 * C1 = 1.19 * C2 →
  total_cost = C1 + C2 →
  total_cost = 450 :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_two_books_l882_88210


namespace geom_seq_sum_first_four_terms_l882_88208

noncomputable def sum_first_n_terms_geom (a₁ q: ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_first_four_terms
  (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (h₂ : a₁ * q^3 = 27) :
  sum_first_n_terms_geom a₁ q 4 = 40 :=
by
  sorry

end geom_seq_sum_first_four_terms_l882_88208


namespace find_f2a_eq_zero_l882_88215

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

end find_f2a_eq_zero_l882_88215


namespace solve_quadratic_equation_l882_88202

theorem solve_quadratic_equation (x : ℝ) :
    2 * x * (x - 5) = 3 * (5 - x) ↔ (x = 5 ∨ x = -3/2) :=
by
  sorry

end solve_quadratic_equation_l882_88202


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l882_88276

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l882_88276


namespace total_prize_money_l882_88204

theorem total_prize_money (P1 P2 P3 : ℕ) (d : ℕ) (total : ℕ) 
(h1 : P1 = 2000) (h2 : d = 400) (h3 : P2 = P1 - d) (h4 : P3 = P2 - d) 
(h5 : total = P1 + P2 + P3) : total = 4800 :=
sorry

end total_prize_money_l882_88204


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l882_88201

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 
  (x y : ℤ) (hx : ∃ m : ℤ, x = 6 * m) (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := 
by 
  sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l882_88201


namespace smaller_angle_at_9_15_l882_88269

theorem smaller_angle_at_9_15 (h_degree : ℝ) (m_degree : ℝ) (smaller_angle : ℝ) :
  (h_degree = 277.5) → (m_degree = 90) → (smaller_angle = 172.5) :=
by
  sorry

end smaller_angle_at_9_15_l882_88269


namespace songs_downloaded_later_l882_88218

-- Definition that each song has a size of 5 MB
def song_size : ℕ := 5

-- Definition that the new songs will occupy 140 MB of memory space
def total_new_song_memory : ℕ := 140

-- Prove that the number of songs Kira downloaded later on that day is 28
theorem songs_downloaded_later (x : ℕ) (h : song_size * x = total_new_song_memory) : x = 28 :=
by
  sorry

end songs_downloaded_later_l882_88218


namespace smallest_number_of_pencils_l882_88230

theorem smallest_number_of_pencils
  (P : ℕ)
  (h5 : P % 5 = 2)
  (h9 : P % 9 = 2)
  (h11 : P % 11 = 2)
  (hP_gt2 : P > 2) :
  P = 497 :=
by
  sorry

end smallest_number_of_pencils_l882_88230
