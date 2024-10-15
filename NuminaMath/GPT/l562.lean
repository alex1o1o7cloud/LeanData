import Mathlib

namespace NUMINAMATH_GPT_no_intersection_points_l562_56236

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- The main theorem to prove the number of intersection points is zero
theorem no_intersection_points : ∀ x : ℝ, f x ≠ g x := by
  intro x
  sorry -- Proof goes here

end NUMINAMATH_GPT_no_intersection_points_l562_56236


namespace NUMINAMATH_GPT_total_dreams_correct_l562_56250

def dreams_per_day : Nat := 4
def days_in_year : Nat := 365
def current_year_dreams : Nat := dreams_per_day * days_in_year
def last_year_dreams : Nat := 2 * current_year_dreams
def total_dreams : Nat := current_year_dreams + last_year_dreams

theorem total_dreams_correct : total_dreams = 4380 :=
by
  -- prime verification needed here
  sorry

end NUMINAMATH_GPT_total_dreams_correct_l562_56250


namespace NUMINAMATH_GPT_quadratic_inequality_l562_56267

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l562_56267


namespace NUMINAMATH_GPT_triangle_problem_proof_l562_56252

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
variables (h2 : c = Real.sqrt 7)
variables (area : ℝ := 3 * Real.sqrt 3 / 2)

-- Prove angle C = π / 3 and perimeter of triangle
theorem triangle_problem_proof 
(h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
(h2 : c = Real.sqrt 7)
(area_condition : (1 / 2) * a * b * (Real.sin C) = area) :
  (C = Real.pi / 3) ∧ (a + b + c = 5 + Real.sqrt 7) := 
by
  sorry

end NUMINAMATH_GPT_triangle_problem_proof_l562_56252


namespace NUMINAMATH_GPT_probability_equivalence_l562_56283

-- Definitions for the conditions:
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3

-- Function to return the probability of selecting a genuine product on the second draw, given first is defective
def probability_genuine_given_defective : ℚ := 
  (defective_products / total_products) * (genuine_products / (total_products - 1))

-- The theorem we need to prove:
theorem probability_equivalence :
  probability_genuine_given_defective = 2 / 3 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_probability_equivalence_l562_56283


namespace NUMINAMATH_GPT_mult_469158_9999_l562_56208

theorem mult_469158_9999 : 469158 * 9999 = 4691176842 := 
by sorry

end NUMINAMATH_GPT_mult_469158_9999_l562_56208


namespace NUMINAMATH_GPT_chocolate_cookies_initial_count_l562_56274

theorem chocolate_cookies_initial_count
  (andy_ate : ℕ) (brother : ℕ) (friends_each : ℕ) (num_friends : ℕ)
  (team_members : ℕ) (first_share : ℕ) (common_diff : ℕ)
  (last_member_share : ℕ) (total_sum_team : ℕ)
  (total_cookies : ℕ) :
  andy_ate = 4 →
  brother = 6 →
  friends_each = 2 →
  num_friends = 3 →
  team_members = 10 →
  first_share = 2 →
  common_diff = 2 →
  last_member_share = first_share + (team_members - 1) * common_diff →
  total_sum_team = team_members / 2 * (first_share + last_member_share) →
  total_cookies = andy_ate + brother + (friends_each * num_friends) + total_sum_team →
  total_cookies = 126 :=
by
  intros ha hb hf hn ht hf1 hc hl hs ht
  sorry

end NUMINAMATH_GPT_chocolate_cookies_initial_count_l562_56274


namespace NUMINAMATH_GPT_value_of_number_l562_56211

theorem value_of_number (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0) 
  (h2 : ∀ n m : ℝ, (n + 5) * (m - 5) = 0 → n^2 + m^2 ≥ 25) 
  (h3 : number^2 + y^2 = 25) : number = -5 :=
sorry

end NUMINAMATH_GPT_value_of_number_l562_56211


namespace NUMINAMATH_GPT_complement_intersection_l562_56238

section SetTheory

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 3}) (hB : B = {2, 3, 4}) : 
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end SetTheory

end NUMINAMATH_GPT_complement_intersection_l562_56238


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l562_56226

theorem sufficient_but_not_necessary {a b : ℝ} (h : a > b ∧ b > 0) : 
  a^2 > b^2 ∧ (¬ (a^2 > b^2 → a > b ∧ b > 0)) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l562_56226


namespace NUMINAMATH_GPT_quadratic_inequality_l562_56244

theorem quadratic_inequality (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) : ∀ x : ℝ, c * x^2 - b * x + a > c * x - b := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l562_56244


namespace NUMINAMATH_GPT_quadratic_has_solution_l562_56230

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end NUMINAMATH_GPT_quadratic_has_solution_l562_56230


namespace NUMINAMATH_GPT_distinct_ordered_pairs_eq_49_l562_56246

theorem distinct_ordered_pairs_eq_49 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 49) (hy : 1 ≤ y ∧ y ≤ 49) (h_eq : x + y = 50) :
  ∃ xs : List (ℕ × ℕ), (∀ p ∈ xs, p.1 + p.2 = 50 ∧ 1 ≤ p.1 ∧ p.1 ≤ 49 ∧ 1 ≤ p.2 ∧ p.2 ≤ 49) ∧ xs.length = 49 :=
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_eq_49_l562_56246


namespace NUMINAMATH_GPT_ratio_of_areas_is_correct_l562_56263

-- Definition of the lengths of the sides of the triangles
def triangle_XYZ_sides := (7, 24, 25)
def triangle_PQR_sides := (9, 40, 41)

-- Definition of the areas of the right triangles
def area_triangle_XYZ := (7 * 24) / 2
def area_triangle_PQR := (9 * 40) / 2

-- The ratio of the areas of the triangles
def ratio_of_areas := area_triangle_XYZ / area_triangle_PQR

-- The expected answer
def expected_ratio := 7 / 15

-- The theorem proving that ratio_of_areas is equal to expected_ratio
theorem ratio_of_areas_is_correct :
  ratio_of_areas = expected_ratio := by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_ratio_of_areas_is_correct_l562_56263


namespace NUMINAMATH_GPT_number_is_seven_l562_56289

-- We will define the problem conditions and assert the answer
theorem number_is_seven (x : ℤ) (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by 
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_number_is_seven_l562_56289


namespace NUMINAMATH_GPT_geometric_sequence_a_div_n_sum_first_n_terms_l562_56257

variable {a : ℕ → ℝ} -- sequence a_n
variable {S : ℕ → ℝ} -- sum of first n terms S_n

axiom S_recurrence {n : ℕ} (hn : n > 0) : 
  S (n + 1) = S n + (n + 1) / (3 * n) * a n

axiom a_1 : a 1 = 1

theorem geometric_sequence_a_div_n :
  ∃ (r : ℝ), ∀ {n : ℕ} (hn : n > 0), (a n / n) = r^n := 
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = (9 / 4) - ((9 / 4) + (3 * n / 2)) * (1 / 3) ^ n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a_div_n_sum_first_n_terms_l562_56257


namespace NUMINAMATH_GPT_box_mass_calculation_l562_56273

variable (h₁ w₁ l₁ : ℝ) (m₁ : ℝ)
variable (h₂ w₂ l₂ density₁ density₂ : ℝ)

theorem box_mass_calculation
  (h₁_eq : h₁ = 3)
  (w₁_eq : w₁ = 4)
  (l₁_eq : l₁ = 6)
  (m₁_eq : m₁ = 72)
  (h₂_eq : h₂ = 1.5 * h₁)
  (w₂_eq : w₂ = 2.5 * w₁)
  (l₂_eq : l₂ = l₁)
  (density₂_eq : density₂ = 2 * density₁)
  (density₁_eq : density₁ = m₁ / (h₁ * w₁ * l₁)) :
  h₂ * w₂ * l₂ * density₂ = 540 := by
  sorry

end NUMINAMATH_GPT_box_mass_calculation_l562_56273


namespace NUMINAMATH_GPT_find_a_l562_56282

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ := 2 * 3^n + a
noncomputable def a_1 (a : ℝ) : ℝ := S_n 1 a
noncomputable def a_2 (a : ℝ) : ℝ := S_n 2 a - S_n 1 a
noncomputable def a_3 (a : ℝ) : ℝ := S_n 3 a - S_n 2 a

theorem find_a (a : ℝ) : a_1 a * a_3 a = (a_2 a)^2 → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l562_56282


namespace NUMINAMATH_GPT_product_nonzero_except_cases_l562_56212

theorem product_nonzero_except_cases (n : ℤ) (h : n ≠ 5 ∧ n ≠ 17 ∧ n ≠ 257) : 
  (n - 5) * (n - 17) * (n - 257) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_product_nonzero_except_cases_l562_56212


namespace NUMINAMATH_GPT_ratio_Laura_to_Ken_is_2_to_1_l562_56286

def Don_paint_tiles_per_minute : ℕ := 3

def Ken_paint_tiles_per_minute : ℕ := Don_paint_tiles_per_minute + 2

def multiple : ℕ := sorry -- Needs to be introduced, not directly from the solution steps

def Laura_paint_tiles_per_minute : ℕ := multiple * Ken_paint_tiles_per_minute

def Kim_paint_tiles_per_minute : ℕ := Laura_paint_tiles_per_minute - 3

def total_tiles_in_15_minutes : ℕ := 375

def total_tiles_per_minute : ℕ := total_tiles_in_15_minutes / 15

def total_tiles_equation : Prop :=
  Don_paint_tiles_per_minute + Ken_paint_tiles_per_minute + Laura_paint_tiles_per_minute + Kim_paint_tiles_per_minute = total_tiles_per_minute

theorem ratio_Laura_to_Ken_is_2_to_1 :
  (total_tiles_equation → Laura_paint_tiles_per_minute / Ken_paint_tiles_per_minute = 2) := sorry

end NUMINAMATH_GPT_ratio_Laura_to_Ken_is_2_to_1_l562_56286


namespace NUMINAMATH_GPT_find_white_balls_l562_56220

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 4

-- Define the number of white balls
def white_balls : ℕ := 12

theorem find_white_balls (x : ℕ) (h1 : (red_balls : ℚ) / (red_balls + x) = prob_red) : x = white_balls :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_white_balls_l562_56220


namespace NUMINAMATH_GPT_b_completion_days_l562_56242

theorem b_completion_days (x : ℝ) :
  (7 * (1 / 24 + 1 / x + 1 / 40) + 4 * (1 / 24 + 1 / x) = 1) → x = 26.25 := 
by 
  sorry

end NUMINAMATH_GPT_b_completion_days_l562_56242


namespace NUMINAMATH_GPT_mean_weight_participants_l562_56224

def weights_120s := [123, 125]
def weights_130s := [130, 132, 133, 135, 137, 138]
def weights_140s := [141, 145, 145, 149, 149]
def weights_150s := [150, 152, 153, 155, 158]
def weights_160s := [164, 167, 167, 169]

def total_weights := weights_120s ++ weights_130s ++ weights_140s ++ weights_150s ++ weights_160s

def total_sum : ℕ := total_weights.sum
def total_count : ℕ := total_weights.length

theorem mean_weight_participants : (total_sum : ℚ) / total_count = 3217 / 22 := by
  sorry -- Proof goes here, but we're skipping it

end NUMINAMATH_GPT_mean_weight_participants_l562_56224


namespace NUMINAMATH_GPT_band_song_arrangements_l562_56200

theorem band_song_arrangements (n : ℕ) (t : ℕ) (r : ℕ) 
  (h1 : n = 8) (h2 : t = 3) (h3 : r = 5) : 
  ∃ (ways : ℕ), ways = 14400 := by
  sorry

end NUMINAMATH_GPT_band_song_arrangements_l562_56200


namespace NUMINAMATH_GPT_sum_of_odd_coefficients_in_binomial_expansion_l562_56213

theorem sum_of_odd_coefficients_in_binomial_expansion :
  let a_0 := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  (a_1 + a_3 + a_5 + a_7 + a_9) = 512 := by
  sorry

end NUMINAMATH_GPT_sum_of_odd_coefficients_in_binomial_expansion_l562_56213


namespace NUMINAMATH_GPT_savings_difference_correct_l562_56271

noncomputable def savings_1989_dick : ℝ := 5000
noncomputable def savings_1989_jane : ℝ := 5000

noncomputable def savings_1990_dick : ℝ := savings_1989_dick + 0.10 * savings_1989_dick
noncomputable def savings_1990_jane : ℝ := savings_1989_jane - 0.05 * savings_1989_jane

noncomputable def savings_1991_dick : ℝ := savings_1990_dick + 0.07 * savings_1990_dick
noncomputable def savings_1991_jane : ℝ := savings_1990_jane + 0.08 * savings_1990_jane

noncomputable def savings_1992_dick : ℝ := savings_1991_dick - 0.12 * savings_1991_dick
noncomputable def savings_1992_jane : ℝ := savings_1991_jane + 0.15 * savings_1991_jane

noncomputable def total_savings_dick : ℝ :=
savings_1989_dick + savings_1990_dick + savings_1991_dick + savings_1992_dick

noncomputable def total_savings_jane : ℝ :=
savings_1989_jane + savings_1990_jane + savings_1991_jane + savings_1992_jane

noncomputable def difference_of_savings : ℝ :=
total_savings_dick - total_savings_jane

theorem savings_difference_correct :
  difference_of_savings = 784.30 :=
by sorry

end NUMINAMATH_GPT_savings_difference_correct_l562_56271


namespace NUMINAMATH_GPT_vector_dot_product_problem_l562_56218

theorem vector_dot_product_problem :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-1, 3)
  let C : ℝ × ℝ := (2, 1)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  let dot_prod := AB.1 * (2 * AC.1 + BC.1) + AB.2 * (2 * AC.2 + BC.2)
  dot_prod = -14 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_problem_l562_56218


namespace NUMINAMATH_GPT_required_jogging_speed_l562_56239

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end NUMINAMATH_GPT_required_jogging_speed_l562_56239


namespace NUMINAMATH_GPT_smallest_n_with_units_digit_and_reorder_l562_56297

theorem smallest_n_with_units_digit_and_reorder :
  ∃ n : ℕ, (∃ a : ℕ, n = 10 * a + 6) ∧ (∃ m : ℕ, 6 * 10^m + a = 4 * n) ∧ n = 153846 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_with_units_digit_and_reorder_l562_56297


namespace NUMINAMATH_GPT_houses_before_boom_l562_56231

theorem houses_before_boom (current_houses built_during_boom houses_before : ℕ) 
  (h1 : current_houses = 2000)
  (h2 : built_during_boom = 574)
  (h3 : current_houses = houses_before + built_during_boom) : 
  houses_before = 1426 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_houses_before_boom_l562_56231


namespace NUMINAMATH_GPT_total_snakes_among_pet_owners_l562_56229

theorem total_snakes_among_pet_owners :
  let owns_only_snakes := 15
  let owns_cats_and_snakes := 7
  let owns_dogs_and_snakes := 10
  let owns_birds_and_snakes := 2
  let owns_snakes_and_hamsters := 3
  let owns_cats_dogs_and_snakes := 4
  let owns_cats_snakes_and_hamsters := 2
  let owns_all_categories := 1
  owns_only_snakes + owns_cats_and_snakes + owns_dogs_and_snakes + owns_birds_and_snakes + owns_snakes_and_hamsters + owns_cats_dogs_and_snakes + owns_cats_snakes_and_hamsters + owns_all_categories = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_snakes_among_pet_owners_l562_56229


namespace NUMINAMATH_GPT_problem_solution_l562_56259

open Function

-- Definitions of the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 2⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨4, 1⟩
def D : Point := ⟨-2, 4⟩

-- Definitions of vectors
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Definitions of conditions
def AB := vec A B
def AD := vec A D
def DC := vec D C

-- Definitions of dot product to check orthogonality
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

-- Lean statement to prove the conditions
theorem problem_solution :
  AB ≠ ⟨-4, 2⟩ ∧
  dot AB AD = 0 ∧
  AB.y * DC.x = AB.x * DC.y ∧
  ((AB.y * DC.x = AB.x * DC.y) ∧ (dot AB AD = 0) → 
  (∃ a b : ℝ, a ≠ b ∧ (a = 0 ∨ b = 0) ∧ AB = ⟨a, -a⟩  ∧ DC = ⟨3 * a, -3 * a⟩)) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem_solution_l562_56259


namespace NUMINAMATH_GPT_series_fraction_simplify_l562_56293

theorem series_fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_series_fraction_simplify_l562_56293


namespace NUMINAMATH_GPT_min_distance_feasible_region_line_l562_56237

def point (x y : ℝ) : Type := ℝ × ℝ 

theorem min_distance_feasible_region_line :
  ∃ (M N : ℝ × ℝ),
    (2 * M.1 + M.2 - 4 >= 0) ∧
    (M.1 - M.2 - 2 <= 0) ∧
    (M.2 - 3 <= 0) ∧
    (N.2 = -2 * N.1 + 2) ∧
    (dist M N = (2 * Real.sqrt 5)/5) :=
by 
  sorry

end NUMINAMATH_GPT_min_distance_feasible_region_line_l562_56237


namespace NUMINAMATH_GPT_find_monthly_salary_l562_56291

-- Definitions based on the conditions
def initial_saving_rate : ℝ := 0.25
def initial_expense_rate : ℝ := 1 - initial_saving_rate
def expense_increase_rate : ℝ := 1.25
def final_saving : ℝ := 300

-- Theorem: Prove the man's monthly salary
theorem find_monthly_salary (S : ℝ) (h1 : initial_saving_rate = 0.25)
  (h2 : initial_expense_rate = 0.75) (h3 : expense_increase_rate = 1.25)
  (h4 : final_saving = 300) : S = 4800 :=
by
  sorry

end NUMINAMATH_GPT_find_monthly_salary_l562_56291


namespace NUMINAMATH_GPT_hexagon_planting_schemes_l562_56292

theorem hexagon_planting_schemes (n m : ℕ) (h : n = 4 ∧ m = 6) : 
  ∃ k, k = 732 := 
by sorry

end NUMINAMATH_GPT_hexagon_planting_schemes_l562_56292


namespace NUMINAMATH_GPT_find_value_of_2a_minus_b_l562_56266

def A : Set ℝ := {x | x < 1 ∨ x > 5}
def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

theorem find_value_of_2a_minus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = {x | 5 < x ∧ x ≤ 6}) : 2 * a - b = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_2a_minus_b_l562_56266


namespace NUMINAMATH_GPT_gym_class_students_correct_l562_56279

noncomputable def check_gym_class_studens :=
  let P1 := 15
  let P2 := 5
  let P3 := 12.5
  let P4 := 9.166666666666666
  let P5 := 8.333333333333334
  P1 = P2 + 10 ∧
  P2 = 2 * P3 - 20 ∧
  P3 = P4 + P5 - 5 ∧
  P4 = (1 / 2) * P5 + 5

theorem gym_class_students_correct : check_gym_class_studens := by
  simp [check_gym_class_studens]
  sorry

end NUMINAMATH_GPT_gym_class_students_correct_l562_56279


namespace NUMINAMATH_GPT_exponent_multiplication_l562_56201

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l562_56201


namespace NUMINAMATH_GPT_seashells_in_jar_at_end_of_month_l562_56232

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end NUMINAMATH_GPT_seashells_in_jar_at_end_of_month_l562_56232


namespace NUMINAMATH_GPT_sin_alpha_value_l562_56227

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end NUMINAMATH_GPT_sin_alpha_value_l562_56227


namespace NUMINAMATH_GPT_perimeter_of_region_proof_l562_56269

noncomputable def perimeter_of_region (total_area : ℕ) (num_squares : ℕ) (arrangement : String) : ℕ :=
  if total_area = 512 ∧ num_squares = 8 ∧ arrangement = "vertical rectangle" then 160 else 0

theorem perimeter_of_region_proof :
  perimeter_of_region 512 8 "vertical rectangle" = 160 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_region_proof_l562_56269


namespace NUMINAMATH_GPT_distance_between_cities_l562_56215

noncomputable def distance_A_to_B : ℕ := 180
noncomputable def distance_B_to_A : ℕ := 150
noncomputable def total_distance : ℕ := distance_A_to_B + distance_B_to_A

theorem distance_between_cities : total_distance = 330 := by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l562_56215


namespace NUMINAMATH_GPT_problem1_problem2_l562_56268

noncomputable def triangle_boscos_condition (a b c A B : ℝ) : Prop :=
  b * Real.cos A = (2 * c + a) * Real.cos (Real.pi - B)

noncomputable def triangle_area (a b c : ℝ) (S : ℝ) : Prop :=
  S = (1 / 2) * a * c * Real.sin (2 * Real.pi / 3)

noncomputable def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = b + a + c

theorem problem1 (a b c A : ℝ) (h : triangle_boscos_condition a b c A (2 * Real.pi / 3)) : 
  ∃ B : ℝ, B = 2 * Real.pi / 3 :=
by
  sorry

theorem problem2 (a c : ℝ) (b : ℝ := 4) (area : ℝ := Real.sqrt 3) (P : ℝ) (h : triangle_area a b c area) (h_perim : triangle_perimeter a b c P) :
  ∃ x : ℝ, x = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l562_56268


namespace NUMINAMATH_GPT_matt_worked_more_on_wednesday_l562_56294

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end NUMINAMATH_GPT_matt_worked_more_on_wednesday_l562_56294


namespace NUMINAMATH_GPT_speed_of_train_l562_56270

-- Conditions
def length_of_train : ℝ := 100
def time_to_cross : ℝ := 12

-- Question and answer
theorem speed_of_train : length_of_train / time_to_cross = 8.33 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_train_l562_56270


namespace NUMINAMATH_GPT_least_possible_value_of_expression_l562_56258

noncomputable def min_expression_value (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023

theorem least_possible_value_of_expression :
  ∃ x : ℝ, min_expression_value x = 2022 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_of_expression_l562_56258


namespace NUMINAMATH_GPT_fraction_of_eggs_hatched_l562_56280

variable (x : ℚ)
variable (survived_first_month_fraction : ℚ := 3/4)
variable (survived_first_year_fraction : ℚ := 2/5)
variable (geese_survived : ℕ := 100)
variable (total_eggs : ℕ := 500)

theorem fraction_of_eggs_hatched :
  (x * survived_first_month_fraction * survived_first_year_fraction * total_eggs : ℚ) = geese_survived → x = 2/3 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_fraction_of_eggs_hatched_l562_56280


namespace NUMINAMATH_GPT_intersection_is_equilateral_triangle_l562_56288

noncomputable def circle_eq (x y : ℝ) := x^2 + (y - 1)^2 = 1
noncomputable def ellipse_eq (x y : ℝ) := 9*x^2 + (y + 1)^2 = 9

theorem intersection_is_equilateral_triangle :
  ∀ A B C : ℝ × ℝ, circle_eq A.1 A.2 ∧ ellipse_eq A.1 A.2 ∧
                 circle_eq B.1 B.2 ∧ ellipse_eq B.1 B.2 ∧
                 circle_eq C.1 C.2 ∧ ellipse_eq C.1 C.2 → 
                 (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_equilateral_triangle_l562_56288


namespace NUMINAMATH_GPT_pastries_count_l562_56235

def C : ℕ := 19
def P : ℕ := C + 112

theorem pastries_count : P = 131 := by
  -- P = 19 + 112
  -- P = 131
  sorry

end NUMINAMATH_GPT_pastries_count_l562_56235


namespace NUMINAMATH_GPT_cube_surface_area_l562_56247

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) :
  V = 729 ∧ V = s^3 ∧ A = 6 * s^2 → A = 486 := by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l562_56247


namespace NUMINAMATH_GPT_total_questions_to_review_is_1750_l562_56265

-- Define the relevant conditions
def num_classes := 5
def students_per_class := 35
def questions_per_exam := 10

-- The total number of questions to be reviewed by Professor Oscar
def total_questions : Nat := num_classes * students_per_class * questions_per_exam

-- The theorem stating the equivalent proof problem
theorem total_questions_to_review_is_1750 : total_questions = 1750 := by
  -- proof steps are skipped here 
  sorry

end NUMINAMATH_GPT_total_questions_to_review_is_1750_l562_56265


namespace NUMINAMATH_GPT_smallest_n_l562_56234

theorem smallest_n (n : ℕ) : 634 * n ≡ 1275 * n [MOD 30] ↔ n = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l562_56234


namespace NUMINAMATH_GPT_mean_of_all_students_is_79_l562_56295

def mean_score_all_students (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) : ℕ :=
  (36 * s + 75 * s) / ((2/5 * s) + s)

theorem mean_of_all_students_is_79 (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) (hF : F = 90) (hS : S = 75) : 
  mean_score_all_students F S f s hf = 79 := by
  sorry

end NUMINAMATH_GPT_mean_of_all_students_is_79_l562_56295


namespace NUMINAMATH_GPT_bob_needs_50_percent_improvement_l562_56240

def bob_time_in_seconds : ℕ := 640
def sister_time_in_seconds : ℕ := 320
def percentage_improvement_needed (bob_time sister_time : ℕ) : ℚ :=
  ((bob_time - sister_time) / bob_time : ℚ) * 100

theorem bob_needs_50_percent_improvement :
  percentage_improvement_needed bob_time_in_seconds sister_time_in_seconds = 50 := by
  sorry

end NUMINAMATH_GPT_bob_needs_50_percent_improvement_l562_56240


namespace NUMINAMATH_GPT_shirts_sold_l562_56253

theorem shirts_sold (pants shorts shirts jackets credit_remaining : ℕ) 
  (price_shirt1 price_shirt2 price_pants : ℕ) 
  (discount tax : ℝ) :
  (pants = 3) →
  (shorts = 5) →
  (jackets = 2) →
  (price_shirt1 = 10) →
  (price_shirt2 = 12) →
  (price_pants = 15) →
  (discount = 0.10) →
  (tax = 0.05) →
  (credit_remaining = 25) →
  (store_credit : ℕ) →
  (store_credit = pants * 5 + shorts * 3 + jackets * 7 + shirts * 4) →
  (total_cost : ℝ) →
  (total_cost = (price_shirt1 + price_shirt2 + price_pants) * (1 - discount) * (1 + tax)) →
  (total_store_credit_used : ℝ) →
  (total_store_credit_used = total_cost - credit_remaining) →
  (initial_credit : ℝ) →
  (initial_credit = total_store_credit_used + (pants * 5 + shorts * 3 + jackets * 7)) →
  shirts = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_shirts_sold_l562_56253


namespace NUMINAMATH_GPT_equalized_distance_l562_56290

noncomputable def wall_width : ℝ := 320 -- wall width in centimeters
noncomputable def poster_count : ℕ := 6 -- number of posters
noncomputable def poster_width : ℝ := 30 -- width of each poster in centimeters
noncomputable def equal_distance : ℝ := 20 -- equal distance in centimeters to be proven

theorem equalized_distance :
  let total_posters_width := poster_count * poster_width
  let remaining_space := wall_width - total_posters_width
  let number_of_spaces := poster_count + 1
  remaining_space / number_of_spaces = equal_distance :=
by {
  sorry
}

end NUMINAMATH_GPT_equalized_distance_l562_56290


namespace NUMINAMATH_GPT_joanne_total_weekly_earnings_l562_56262

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end NUMINAMATH_GPT_joanne_total_weekly_earnings_l562_56262


namespace NUMINAMATH_GPT_number_of_persons_in_first_group_eq_39_l562_56206

theorem number_of_persons_in_first_group_eq_39 :
  ∀ (P : ℕ),
    (P * 12 * 5 = 15 * 26 * 6) →
    P = 39 :=
by
  intros P h
  have h1 : P = (15 * 26 * 6) / (12 * 5) := sorry
  simp at h1
  exact h1

end NUMINAMATH_GPT_number_of_persons_in_first_group_eq_39_l562_56206


namespace NUMINAMATH_GPT_problem_1_problem_2_l562_56254

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

theorem problem_1 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 :=
by
  sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + 2 * b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l562_56254


namespace NUMINAMATH_GPT_smallest_divisor_subtracted_l562_56260

theorem smallest_divisor_subtracted (a b d : ℕ) (h1: a = 899830) (h2: b = 6) (h3: a - b = 899824) (h4 : 6 < d) 
(h5 : d ∣ (a - b)) : d = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisor_subtracted_l562_56260


namespace NUMINAMATH_GPT_barrels_in_one_ton_l562_56245

-- Definitions (conditions)
def barrel_weight : ℕ := 10 -- in kilograms
def ton_in_kilograms : ℕ := 1000

-- Theorem Statement
theorem barrels_in_one_ton : ton_in_kilograms / barrel_weight = 100 :=
by
  sorry

end NUMINAMATH_GPT_barrels_in_one_ton_l562_56245


namespace NUMINAMATH_GPT_average_visitors_per_day_l562_56222

theorem average_visitors_per_day:
  (∃ (Sundays OtherDays: ℕ) (visitors_per_sunday visitors_per_other_day: ℕ),
    Sundays = 4 ∧
    OtherDays = 26 ∧
    visitors_per_sunday = 600 ∧
    visitors_per_other_day = 240 ∧
    (Sundays + OtherDays = 30) ∧
    (Sundays * visitors_per_sunday + OtherDays * visitors_per_other_day) / 30 = 288) :=
sorry

end NUMINAMATH_GPT_average_visitors_per_day_l562_56222


namespace NUMINAMATH_GPT_cost_of_video_game_console_l562_56214

-- Define the problem conditions
def earnings_Mar_to_Aug : ℕ := 460
def hours_Mar_to_Aug : ℕ := 23
def earnings_per_hour : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug
def hours_Sep_to_Feb : ℕ := 8
def cost_car_fix : ℕ := 340
def additional_hours_needed : ℕ := 16

-- Proof that the cost of the video game console is $600
theorem cost_of_video_game_console :
  let initial_earnings := earnings_Mar_to_Aug
  let earnings_from_Sep_to_Feb := hours_Sep_to_Feb * earnings_per_hour
  let total_earnings_before_expenses := initial_earnings + earnings_from_Sep_to_Feb
  let current_savings := total_earnings_before_expenses - cost_car_fix
  let earnings_after_additional_work := additional_hours_needed * earnings_per_hour
  let total_savings := current_savings + earnings_after_additional_work
  total_savings = 600 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_video_game_console_l562_56214


namespace NUMINAMATH_GPT_swap_instruments_readings_change_l562_56296

def U0 : ℝ := 45
def R : ℝ := 50
def r : ℝ := 20

theorem swap_instruments_readings_change :
  let I_total := U0 / (R / 2 + r)
  let U1 := I_total * r
  let I1 := I_total / 2
  let I2 := U0 / R
  let I := U0 / (R + r)
  let U2 := I * r
  let ΔI := I2 - I1
  let ΔU := U1 - U2
  ΔI = 0.4 ∧ ΔU = 7.14 :=
by
  sorry

end NUMINAMATH_GPT_swap_instruments_readings_change_l562_56296


namespace NUMINAMATH_GPT_kim_shirts_left_l562_56264

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_kim_shirts_left_l562_56264


namespace NUMINAMATH_GPT_volume_inequality_find_min_k_l562_56298

noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

noncomputable def cylinder_volume (R h : ℝ) : ℝ :=
    let r := (R * h) / Real.sqrt (R^2 + h^2)
    Real.pi * r^2 * h

noncomputable def k_value (R h : ℝ) : ℝ := (R^2 + h^2) / (3 * h^2)

theorem volume_inequality (R h : ℝ) (h_pos : R > 0 ∧ h > 0) : 
    cone_volume R h ≠ cylinder_volume R h := by sorry

theorem find_min_k (R h : ℝ) (h_pos : R > 0 ∧ h > 0) (k : ℝ) :
    cone_volume R h = k * cylinder_volume R h → k = (R^2 + h^2) / (3 * h^2) := by sorry

end NUMINAMATH_GPT_volume_inequality_find_min_k_l562_56298


namespace NUMINAMATH_GPT_joey_pills_l562_56284

-- Definitions for the initial conditions
def TypeA_initial := 2
def TypeA_increment := 1

def TypeB_initial := 3
def TypeB_increment := 2

def TypeC_initial := 4
def TypeC_increment := 3

def days := 42

-- Function to calculate the sum of an arithmetic series
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- The theorem to be proved
theorem joey_pills :
  arithmetic_sum TypeA_initial TypeA_increment days = 945 ∧
  arithmetic_sum TypeB_initial TypeB_increment days = 1848 ∧
  arithmetic_sum TypeC_initial TypeC_increment days = 2751 :=
by sorry

end NUMINAMATH_GPT_joey_pills_l562_56284


namespace NUMINAMATH_GPT_triangle_inequality_l562_56241

theorem triangle_inequality (a b c : ℝ) (h1 : b + c > a) (h2 : c + a > b) (h3 : a + b > c) :
  ab + bc + ca ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (ab + bc + ca) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l562_56241


namespace NUMINAMATH_GPT_perpendicular_line_directional_vector_l562_56216

theorem perpendicular_line_directional_vector
  (l1 : ℝ → ℝ → Prop)
  (l2 : ℝ → ℝ → Prop)
  (perpendicular : ∀ x y, l1 x y ↔ l2 y (-x))
  (l2_eq : ∀ x y, l2 x y ↔ 2 * x + 5 * y = 1) :
  ∃ d1 d2, (d1, d2) = (5, -2) ∧ (d1 * 2 + d2 * 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_directional_vector_l562_56216


namespace NUMINAMATH_GPT_geometric_sequence_a7_l562_56210

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n - 1)

theorem geometric_sequence_a7 
  (a1 q : ℝ)
  (a1_neq_zero : a1 ≠ 0)
  (a9_eq_256 : a_n a1 q 9 = 256)
  (a1_a3_eq_4 : a_n a1 q 1 * a_n a1 q 3 = 4) :
  a_n a1 q 7 = 64 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l562_56210


namespace NUMINAMATH_GPT_mean_weight_of_soccer_team_l562_56299

-- Define the weights as per the conditions
def weights : List ℕ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

-- Define the total weight
def total_weight : ℕ := 64 + 68 + 71 + 73 + 76 + 76 + 77 + 78 + 80 + 82 + 85 + 87 + 89 + 89

-- Define the number of players
def number_of_players : ℕ := 14

-- Calculate the mean weight
noncomputable def mean_weight : ℚ := total_weight / number_of_players

-- The proof problem statement
theorem mean_weight_of_soccer_team : mean_weight = 75.357 := by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_mean_weight_of_soccer_team_l562_56299


namespace NUMINAMATH_GPT_solve_for_x_l562_56205

theorem solve_for_x (x : ℝ) (h : |2000 * x + 2000| = 20 * 2000) : x = 19 ∨ x = -21 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l562_56205


namespace NUMINAMATH_GPT_unique_solution_l562_56203

theorem unique_solution (m n : ℕ) (h1 : n^4 ∣ 2 * m^5 - 1) (h2 : m^4 ∣ 2 * n^5 + 1) : m = 1 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l562_56203


namespace NUMINAMATH_GPT_polynomial_simplification_l562_56287

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2)*(5*x^12 + 3*x^11 + 2*x^10 - x^9) = 15*x^13 - x^12 - 7*x^10 + 2*x^9 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_simplification_l562_56287


namespace NUMINAMATH_GPT_tylenol_interval_l562_56276

/-- Mark takes 2 Tylenol tablets of 500 mg each at certain intervals for 12 hours, and he ends up taking 3 grams of Tylenol in total. Prove that the interval in hours at which he takes the tablets is 2.4 hours. -/
theorem tylenol_interval 
    (total_dose_grams : ℝ)
    (tablet_mg : ℝ)
    (hours : ℝ)
    (tablets_taken_each_time : ℝ) 
    (total_tablets : ℝ) 
    (interval_hours : ℝ) :
    total_dose_grams = 3 → 
    tablet_mg = 500 → 
    hours = 12 → 
    tablets_taken_each_time = 2 → 
    total_tablets = (total_dose_grams * 1000) / tablet_mg → 
    interval_hours = hours / (total_tablets / tablets_taken_each_time - 1) → 
    interval_hours = 2.4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tylenol_interval_l562_56276


namespace NUMINAMATH_GPT_evaluate_expression_l562_56248

theorem evaluate_expression : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l562_56248


namespace NUMINAMATH_GPT_num_of_winnable_players_l562_56219

noncomputable def num_players := 2 ^ 2013

def can_win_if (x y : Nat) : Prop := x ≤ y + 3

def single_elimination_tournament (players : Nat) : Nat :=
  -- Function simulating the single elimination based on the specified can_win_if condition
  -- Assuming the given conditions and returning the number of winnable players directly
  6038

theorem num_of_winnable_players : single_elimination_tournament num_players = 6038 :=
  sorry

end NUMINAMATH_GPT_num_of_winnable_players_l562_56219


namespace NUMINAMATH_GPT_slope_of_line_l562_56249

theorem slope_of_line (x y : ℝ) :
  (∀ (x y : ℝ), (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l562_56249


namespace NUMINAMATH_GPT_oldest_son_park_visits_l562_56285

theorem oldest_son_park_visits 
    (season_pass_cost : ℕ)
    (cost_per_trip : ℕ)
    (youngest_son_trips : ℕ) 
    (remaining_value : ℕ)
    (oldest_son_trips : ℕ) : 
    season_pass_cost = 100 →
    cost_per_trip = 4 →
    youngest_son_trips = 15 →
    remaining_value = season_pass_cost - youngest_son_trips * cost_per_trip →
    oldest_son_trips = remaining_value / cost_per_trip →
    oldest_son_trips = 10 := 
by sorry

end NUMINAMATH_GPT_oldest_son_park_visits_l562_56285


namespace NUMINAMATH_GPT_area_of_R3_l562_56204

theorem area_of_R3 (r1 r2 r3 : ℝ) (h1: r1^2 = 25) 
                   (h2: r2 = (2/3) * r1) (h3: r3 = (2/3) * r2) :
                   r3^2 = 400 / 81 := 
by
  sorry

end NUMINAMATH_GPT_area_of_R3_l562_56204


namespace NUMINAMATH_GPT_hallway_length_l562_56278

theorem hallway_length (s t d : ℝ) (h1 : 3 * s * t = 12) (h2 : s * t = d - 12) : d = 16 :=
sorry

end NUMINAMATH_GPT_hallway_length_l562_56278


namespace NUMINAMATH_GPT_oliver_bumper_cars_proof_l562_56233

def rides_of_bumper_cars (total_tickets : ℕ) (tickets_per_ride : ℕ) (rides_ferris_wheel : ℕ) : ℕ :=
  (total_tickets - rides_ferris_wheel * tickets_per_ride) / tickets_per_ride

def oliver_bumper_car_rides : Prop :=
  rides_of_bumper_cars 30 3 7 = 3

theorem oliver_bumper_cars_proof : oliver_bumper_car_rides :=
by
  sorry

end NUMINAMATH_GPT_oliver_bumper_cars_proof_l562_56233


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l562_56202

noncomputable def problem1 : Real :=
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2)

noncomputable def problem2 : Real :=
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6)

theorem problem1_solution : problem1 = 0 := by
  sorry

theorem problem2_solution : problem2 = 6 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l562_56202


namespace NUMINAMATH_GPT_total_cost_of_pets_l562_56243

theorem total_cost_of_pets 
  (num_puppies num_kittens num_parakeets : ℕ)
  (cost_parakeet cost_puppy cost_kitten : ℕ)
  (h1 : num_puppies = 2)
  (h2 : num_kittens = 2)
  (h3 : num_parakeets = 3)
  (h4 : cost_parakeet = 10)
  (h5 : cost_puppy = 3 * cost_parakeet)
  (h6 : cost_kitten = 2 * cost_parakeet) : 
  num_puppies * cost_puppy + num_kittens * cost_kitten + num_parakeets * cost_parakeet = 130 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_pets_l562_56243


namespace NUMINAMATH_GPT_notebooks_left_l562_56261

theorem notebooks_left (bundles : ℕ) (notebooks_per_bundle : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  bundles = 5 ∧ notebooks_per_bundle = 25 ∧ groups = 8 ∧ students_per_group = 13 →
  bundles * notebooks_per_bundle - groups * students_per_group = 21 := 
by sorry

end NUMINAMATH_GPT_notebooks_left_l562_56261


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l562_56281

-- Statement part (1)
theorem problem_part1 : ( (2 / 3) - (1 / 4) - (1 / 6) ) * 24 = 6 :=
sorry

-- Statement part (2)
theorem problem_part2 : (-2)^3 + (-9 + (-3)^2 * (1 / 3)) = -14 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l562_56281


namespace NUMINAMATH_GPT_integer_solutions_of_linear_diophantine_eq_l562_56228

theorem integer_solutions_of_linear_diophantine_eq 
  (a b c : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (x₀ y₀ : ℤ)
  (h_particular_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, (a * x + b * y = c) → ∃ (k : ℤ), (x = x₀ + k * b) ∧ (y = y₀ - k * a) := 
by
  sorry

end NUMINAMATH_GPT_integer_solutions_of_linear_diophantine_eq_l562_56228


namespace NUMINAMATH_GPT_problem_solution_l562_56217

theorem problem_solution (a b c : ℝ) (h1 : a^2 - b^2 = 5) (h2 : a * b = 2) (h3 : a^2 + b^2 + c^2 = 8) : 
  a^4 + b^4 + c^4 = 38 :=
sorry

end NUMINAMATH_GPT_problem_solution_l562_56217


namespace NUMINAMATH_GPT_choose_integers_l562_56207

def smallest_prime_divisor (n : ℕ) : ℕ := sorry
def number_of_divisors (n : ℕ) : ℕ := sorry

theorem choose_integers :
  ∃ (a : ℕ → ℕ), (∀ i, i < 2022 → a i < a (i + 1)) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 2022 →
    number_of_divisors (a (k + 1) - a k - 1) > 2023^k ∧
    smallest_prime_divisor (a (k + 1) - a k) > 2023^k
  ) :=
sorry

end NUMINAMATH_GPT_choose_integers_l562_56207


namespace NUMINAMATH_GPT_original_faculty_members_l562_56251

theorem original_faculty_members (reduced_faculty : ℕ) (percentage : ℝ) : 
  reduced_faculty = 195 → percentage = 0.80 → 
  (∃ (original_faculty : ℕ), (original_faculty : ℝ) = reduced_faculty / percentage ∧ original_faculty = 244) :=
by
  sorry

end NUMINAMATH_GPT_original_faculty_members_l562_56251


namespace NUMINAMATH_GPT_disjoint_polynomial_sets_l562_56209

theorem disjoint_polynomial_sets (A B : ℤ) : 
  ∃ C : ℤ, ∀ x1 x2 : ℤ, x1^2 + A * x1 + B ≠ 2 * x2^2 + 2 * x2 + C :=
by
  sorry

end NUMINAMATH_GPT_disjoint_polynomial_sets_l562_56209


namespace NUMINAMATH_GPT_snowboard_price_after_discounts_l562_56225

theorem snowboard_price_after_discounts
  (original_price : ℝ) (friday_discount_rate : ℝ) (monday_discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (price_after_all_adjustments : ℝ) :
  original_price = 200 →
  friday_discount_rate = 0.40 →
  monday_discount_rate = 0.20 →
  sales_tax_rate = 0.05 →
  price_after_all_adjustments = 100.80 :=
by
  intros
  sorry

end NUMINAMATH_GPT_snowboard_price_after_discounts_l562_56225


namespace NUMINAMATH_GPT_longest_side_of_triangle_l562_56272

theorem longest_side_of_triangle (a d : ℕ) (h1 : d = 2) (h2 : a - d > 0) (h3 : a + d > 0)
    (h_angle : ∃ C : ℝ, C = 120) 
    (h_arith_seq : ∃ (b c : ℕ), b = a - d ∧ c = a ∧ b + 2 * d = c + d) : 
    a + d = 7 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l562_56272


namespace NUMINAMATH_GPT_fresh_grapes_water_percentage_l562_56221

/--
Given:
- Fresh grapes contain a certain percentage (P%) of water by weight.
- Dried grapes contain 25% water by weight.
- The weight of dry grapes obtained from 200 kg of fresh grapes is 66.67 kg.

Prove:
- The percentage of water (P) in fresh grapes is 75%.
-/
theorem fresh_grapes_water_percentage
  (P : ℝ) (H1 : ∃ P, P / 100 * 200 = 0.75 * 66.67) :
  P = 75 :=
sorry

end NUMINAMATH_GPT_fresh_grapes_water_percentage_l562_56221


namespace NUMINAMATH_GPT_max_correct_answers_l562_56223

theorem max_correct_answers :
  ∃ (c w b : ℕ), c + w + b = 25 ∧ 4 * c - 3 * w = 57 ∧ c = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_correct_answers_l562_56223


namespace NUMINAMATH_GPT_number_of_men_l562_56277

variable (M : ℕ)

-- Define the first condition: M men reaping 80 hectares in 24 days.
def first_work_rate (M : ℕ) : ℚ := (80 : ℚ) / (M * 24)

-- Define the second condition: 36 men reaping 360 hectares in 30 days.
def second_work_rate : ℚ := (360 : ℚ) / (36 * 30)

-- Lean 4 statement: Prove the equivalence given conditions.
theorem number_of_men (h : first_work_rate M = second_work_rate) : M = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_l562_56277


namespace NUMINAMATH_GPT_milly_needs_flamingoes_l562_56255

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end NUMINAMATH_GPT_milly_needs_flamingoes_l562_56255


namespace NUMINAMATH_GPT_uncle_bruce_dough_weight_l562_56275

-- Definitions based on the conditions
variable {TotalChocolate : ℕ} (h1 : TotalChocolate = 13)
variable {ChocolateLeftOver : ℕ} (h2 : ChocolateLeftOver = 4)
variable {ChocolatePercentage : ℝ} (h3 : ChocolatePercentage = 0.2) 
variable {WeightOfDough : ℝ}

-- Target statement expressing the final question and answer
theorem uncle_bruce_dough_weight 
  (h1 : TotalChocolate = 13) 
  (h2 : ChocolateLeftOver = 4) 
  (h3 : ChocolatePercentage = 0.2) : 
  WeightOfDough = 36 := by
  sorry

end NUMINAMATH_GPT_uncle_bruce_dough_weight_l562_56275


namespace NUMINAMATH_GPT_colorable_graph_l562_56256

variable (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]

/-- Each city has at least one road leading out of it -/
def has_one_road (v : V) : Prop := ∃ w : V, E v w

/-- No city is connected by roads to all other cities -/
def not_connected_to_all (v : V) : Prop := ¬ ∀ w : V, E v w ↔ w ≠ v

/-- A set of cities D is dominating if every city not in D is connected by a road to at least one city in D -/
def is_dominating_set (D : Finset V) : Prop :=
  ∀ v : V, v ∉ D → ∃ d ∈ D, E v d

noncomputable def dominating_set_min_card (k : ℕ) : Prop :=
  ∀ D : Finset V, is_dominating_set V E D → D.card ≥ k

/-- Prove that the graph can be colored using 2001 - k colors such that no two adjacent vertices share the same color -/
theorem colorable_graph (k : ℕ) (hk : dominating_set_min_card V E k) :
    ∃ (colors : V → Fin (2001 - k)), ∀ v w : V, E v w → colors v ≠ colors w := 
by 
  sorry

end NUMINAMATH_GPT_colorable_graph_l562_56256
