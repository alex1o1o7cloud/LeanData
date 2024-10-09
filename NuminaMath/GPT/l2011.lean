import Mathlib

namespace triangle_area_is_17_point_5_l2011_201131

-- Define the points A, B, and C as tuples of coordinates
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (7, 2)
def C : (ℝ × ℝ) := (4, 9)

-- Function to calculate the area of a triangle given its vertices
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- The theorem statement asserting the area of the triangle is 17.5 square units
theorem triangle_area_is_17_point_5 :
  area_of_triangle A B C = 17.5 :=
by
  sorry -- Proof is omitted

end triangle_area_is_17_point_5_l2011_201131


namespace original_expenditure_l2011_201176

theorem original_expenditure (initial_students new_students : ℕ) (increment_expense : ℝ) (decrement_avg_expense : ℝ) (original_avg_expense : ℝ) (new_avg_expense : ℝ) 
  (total_initial_expense original_expenditure : ℝ)
  (h1 : initial_students = 35) 
  (h2 : new_students = 7) 
  (h3 : increment_expense = 42)
  (h4 : decrement_avg_expense = 1)
  (h5 : new_avg_expense = original_avg_expense - decrement_avg_expense)
  (h6 : total_initial_expense = initial_students * original_avg_expense)
  (h7 : original_expenditure = total_initial_expense)
  (h8 : 42 * new_avg_expense - original_students * original_avg_expense = increment_expense) :
  original_expenditure = 420 := 
by
  sorry

end original_expenditure_l2011_201176


namespace cookies_in_the_fridge_l2011_201144

-- Define the conditions
def total_baked : ℕ := 256
def tim_cookies : ℕ := 15
def mike_cookies : ℕ := 23
def anna_cookies : ℕ := 2 * tim_cookies

-- Define the proof problem
theorem cookies_in_the_fridge : (total_baked - (tim_cookies + mike_cookies + anna_cookies)) = 188 :=
by
  -- insert proof here
  sorry

end cookies_in_the_fridge_l2011_201144


namespace problem_1_problem_2_l2011_201114

-- Define the given function
def f (x : ℝ) := |x - 1|

-- Problem 1: Prove if f(x) + f(1 - x) ≥ a always holds, then a ≤ 1
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f x + f (1 - x) ≥ a) → a ≤ 1 :=
  sorry

-- Problem 2: Prove if a + 2b = 8, then f(a)^2 + f(b)^2 ≥ 5
theorem problem_2 (a b : ℝ) : 
  (a + 2 * b = 8) → (f a)^2 + (f b)^2 ≥ 5 :=
  sorry

end problem_1_problem_2_l2011_201114


namespace volunteer_distribution_l2011_201190

theorem volunteer_distribution :
  let students := 5
  let projects := 4
  let combinations := Nat.choose students 2
  let permutations := Nat.factorial projects
  combinations * permutations = 240 := 
by
  sorry

end volunteer_distribution_l2011_201190


namespace minimum_perimeter_triangle_MAF_is_11_l2011_201187

-- Define point, parabola, and focus
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the specific points in the problem
def A : Point := ⟨5, 3⟩

-- Parabola with the form y^2 = 4x has the focus at (1, 0)
def F : Point := ⟨1, 0⟩

-- Minimum perimeter problem for ΔMAF
noncomputable def minimum_perimeter_triangle_MAF (M : Point) : ℝ :=
  (dist (M.x, M.y) (A.x, A.y)) + (dist (M.x, M.y) (F.x, F.y))

-- The goal is to show the minimum value of the perimeter is 11
theorem minimum_perimeter_triangle_MAF_is_11 (M : Point) 
  (hM_parabola : M.y^2 = 4 * M.x) 
  (hM_not_AF : M.x ≠ (5 + (3 * ((M.y - 0) / (M.x - 1))) )) : 
  ∃ M, minimum_perimeter_triangle_MAF M = 11 :=
sorry

end minimum_perimeter_triangle_MAF_is_11_l2011_201187


namespace product_of_numbers_l2011_201171

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := sorry

end product_of_numbers_l2011_201171


namespace triangle_area_l2011_201175

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) 
  (h_c : c = 2) (h_C : C = π / 3)
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l2011_201175


namespace physics_marks_l2011_201150

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 180) 
  (h2 : P + M = 180) 
  (h3 : P + C = 140) : 
  P = 140 := 
by 
  sorry

end physics_marks_l2011_201150


namespace quotient_of_x6_plus_8_by_x_minus_1_l2011_201168

theorem quotient_of_x6_plus_8_by_x_minus_1 :
  ∀ (x : ℝ), x ≠ 1 →
  (∃ Q : ℝ → ℝ, x^6 + 8 = (x - 1) * Q x + 9 ∧ Q x = x^5 + x^4 + x^3 + x^2 + x + 1) := 
  by
    intros x hx
    sorry

end quotient_of_x6_plus_8_by_x_minus_1_l2011_201168


namespace find_a_l2011_201106

noncomputable def a := 1/2

theorem find_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 1 - a^2 = 3/4) : a = 1/2 :=
sorry

end find_a_l2011_201106


namespace prime_cond_l2011_201143

theorem prime_cond (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1) : 
  (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) → (p = 2 ∧ q = 5 ∧ n = 2) :=
  sorry

end prime_cond_l2011_201143


namespace factorization_correct_l2011_201193

theorem factorization_correct (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  -- The actual proof will be written here.
  sorry

end factorization_correct_l2011_201193


namespace M1_M2_product_l2011_201148

theorem M1_M2_product (M_1 M_2 : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 →
  (42 * x - 51) / (x^2 - 5 * x + 6) = (M_1 / (x - 2)) + (M_2 / (x - 3))) →
  M_1 * M_2 = -2981.25 :=
by
  intros h
  sorry

end M1_M2_product_l2011_201148


namespace factorize_expression_l2011_201183

variable (x y : ℝ)

theorem factorize_expression : 9 * x^2 * y - y = y * (3 * x + 1) * (3 * x - 1) := 
by
  sorry

end factorize_expression_l2011_201183


namespace men_in_first_group_l2011_201113

theorem men_in_first_group (M : ℕ) (h1 : (M * 15) = (M + 0) * 15) (h2 : (15 * 36) = 540) : M = 36 :=
by
  -- Proof would go here
  sorry

end men_in_first_group_l2011_201113


namespace find_positive_solution_l2011_201115

-- Defining the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions from the problem statement
def condition1 : Prop := x * y + 3 * x + 4 * y + 10 = 30
def condition2 : Prop := y * z + 4 * y + 2 * z + 8 = 6
def condition3 : Prop := x * z + 4 * x + 3 * z + 12 = 30

-- The theorem that states the positive solution for x is 3
theorem find_positive_solution (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 x z) : x = 3 :=
by {
  sorry
}

end find_positive_solution_l2011_201115


namespace bruised_more_than_wormy_l2011_201155

noncomputable def total_apples : ℕ := 85
noncomputable def fifth_of_apples (n : ℕ) : ℕ := n / 5
noncomputable def apples_left_to_eat_raw : ℕ := 42

noncomputable def wormy_apples : ℕ := fifth_of_apples total_apples
noncomputable def total_non_raw_eatable_apples : ℕ := total_apples - apples_left_to_eat_raw
noncomputable def bruised_apples : ℕ := total_non_raw_eatable_apples - wormy_apples

theorem bruised_more_than_wormy :
  bruised_apples - wormy_apples = 43 - 17 :=
by sorry

end bruised_more_than_wormy_l2011_201155


namespace find_number_l2011_201132

theorem find_number : ∃ x : ℝ, (x / 6 * 12 = 10) ∧ x = 5 :=
by
 sorry

end find_number_l2011_201132


namespace max_a_value_l2011_201197

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) :
  a ≤ 2924 :=
by sorry

end max_a_value_l2011_201197


namespace geometric_series_solution_l2011_201100

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l2011_201100


namespace total_pears_after_giving_away_l2011_201152

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17
def carlos_pears : ℕ := 25
def pears_given_away_per_person : ℕ := 5

theorem total_pears_after_giving_away :
  (alyssa_pears + nancy_pears + carlos_pears) - (3 * pears_given_away_per_person) = 69 :=
by
  sorry

end total_pears_after_giving_away_l2011_201152


namespace prove_logical_proposition_l2011_201166

theorem prove_logical_proposition (p q : Prop) (hp : p) (hq : ¬q) : (¬p ∨ ¬q) :=
by
  sorry

end prove_logical_proposition_l2011_201166


namespace calculate_y_position_l2011_201102

/--
Given a number line with equally spaced markings, if eight steps are taken from \( 0 \) to \( 32 \),
then the position \( y \) after five steps can be calculated.
-/
theorem calculate_y_position : 
    ∃ y : ℕ, (∀ (step length : ℕ), (8 * step = 32) ∧ (y = 5 * length) → y = 20) :=
by
  -- Provide initial definitions based on the conditions
  let step := 4
  let length := 4
  use (5 * length)
  sorry

end calculate_y_position_l2011_201102


namespace scientific_notation_correct_l2011_201147

-- Define the problem conditions
def original_number : ℝ := 6175700

-- Define the expected output in scientific notation
def scientific_notation_representation (x : ℝ) : Prop :=
  x = 6.1757 * 10^6

-- The theorem to prove
theorem scientific_notation_correct : scientific_notation_representation original_number :=
by sorry

end scientific_notation_correct_l2011_201147


namespace smallest_number_is_1013_l2011_201177

def smallest_number_divisible (n : ℕ) : Prop :=
  n - 5 % Nat.lcm 12 (Nat.lcm 16 (Nat.lcm 18 (Nat.lcm 21 28))) = 0

theorem smallest_number_is_1013 : smallest_number_divisible 1013 :=
by
  sorry

end smallest_number_is_1013_l2011_201177


namespace sin_minus_cos_eq_sqrt2_l2011_201194

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l2011_201194


namespace average_score_group2_l2011_201167

-- Total number of students
def total_students : ℕ := 50

-- Overall average score
def overall_average_score : ℝ := 92

-- Number of students from 1 to 30
def group1_students : ℕ := 30

-- Average score of students from 1 to 30
def group1_average_score : ℝ := 90

-- Total number of students - group1_students = 50 - 30 = 20
def group2_students : ℕ := total_students - group1_students

-- Lean 4 statement to prove the average score of students with student numbers 31 to 50 is 95
theorem average_score_group2 :
  (overall_average_score * total_students = group1_average_score * group1_students + x * group2_students) →
  x = 95 :=
sorry

end average_score_group2_l2011_201167


namespace percentage_of_trout_is_correct_l2011_201163

-- Define the conditions
def video_game_cost := 60
def last_weekend_earnings := 35
def earnings_per_trout := 5
def earnings_per_bluegill := 4
def total_fish_caught := 5
def additional_savings_needed := 2

-- Define the total amount needed to buy the game
def total_required_savings := video_game_cost - additional_savings_needed

-- Define the amount earned this Sunday
def earnings_this_sunday := total_required_savings - last_weekend_earnings

-- Define the number of trout and blue-gill caught thisSunday
def num_trout := 3
def num_bluegill := 2    -- Derived from the conditions

-- Theorem: given the conditions, prove that the percentage of trout is 60%
theorem percentage_of_trout_is_correct :
  (num_trout + num_bluegill = total_fish_caught) ∧
  (earnings_per_trout * num_trout + earnings_per_bluegill * num_bluegill = earnings_this_sunday) →
  100 * num_trout / total_fish_caught = 60 := 
by
  sorry

end percentage_of_trout_is_correct_l2011_201163


namespace ratio_of_c_d_l2011_201135

theorem ratio_of_c_d 
  (x y c d : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c)
  (h2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 :=
by 
  sorry

end ratio_of_c_d_l2011_201135


namespace johns_age_l2011_201103

-- Define the variables and conditions
def age_problem (j d : ℕ) : Prop :=
j = d - 34 ∧ j + d = 84

-- State the theorem to prove that John's age is 25
theorem johns_age : ∃ (j d : ℕ), age_problem j d ∧ j = 25 :=
by {
  sorry
}

end johns_age_l2011_201103


namespace circles_positional_relationship_l2011_201139

theorem circles_positional_relationship :
  ∃ R r : ℝ, (R * r = 2 ∧ R + r = 3) ∧ 3 = R + r → "externally tangent" = "externally tangent" :=
by
  sorry

end circles_positional_relationship_l2011_201139


namespace volume_of_56_ounces_is_24_cubic_inches_l2011_201118

-- Given information as premises
def directlyProportional (V W : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ V = k * W

-- The specific conditions in the problem
def initial_volume := 48   -- in cubic inches
def initial_weight := 112  -- in ounces
def target_weight := 56    -- in ounces
def target_volume := 24    -- in cubic inches (the value we need to prove)

-- The theorem statement 
theorem volume_of_56_ounces_is_24_cubic_inches
  (h1 : directlyProportional initial_volume initial_weight)
  (h2 : directlyProportional target_volume target_weight)
  (h3 : target_weight = 56)
  (h4 : initial_volume = 48)
  (h5 : initial_weight = 112) :
  target_volume = 24 :=
sorry -- Proof not required as per instructions

end volume_of_56_ounces_is_24_cubic_inches_l2011_201118


namespace molecular_weight_l2011_201119

theorem molecular_weight :
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  (2 * H_weight + 1 * Br_weight + 3 * O_weight + 1 * C_weight + 1 * N_weight + 2 * S_weight) = 220.065 :=
by
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  sorry

end molecular_weight_l2011_201119


namespace initial_number_of_trees_l2011_201198

theorem initial_number_of_trees (trees_removed remaining_trees initial_trees : ℕ) 
  (h1 : trees_removed = 4) 
  (h2 : remaining_trees = 2) 
  (h3 : remaining_trees + trees_removed = initial_trees) : 
  initial_trees = 6 :=
by
  sorry

end initial_number_of_trees_l2011_201198


namespace repeating_decimal_eq_fraction_l2011_201182

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l2011_201182


namespace ceil_sqrt_200_eq_15_l2011_201156

theorem ceil_sqrt_200_eq_15 : ⌈Real.sqrt 200⌉ = 15 := 
sorry

end ceil_sqrt_200_eq_15_l2011_201156


namespace line_intersects_ellipse_two_points_l2011_201133

theorem line_intersects_ellipse_two_points (k b : ℝ) : 
  (-2 < b) ∧ (b < 2) ↔ ∀ x y : ℝ, (y = k * x + b) ↔ (x ^ 2 / 9 + y ^ 2 / 4 = 1) → true :=
sorry

end line_intersects_ellipse_two_points_l2011_201133


namespace surface_area_of_each_smaller_cube_l2011_201140

theorem surface_area_of_each_smaller_cube
  (L : ℝ) (l : ℝ)
  (h1 : 6 * L^2 = 600)
  (h2 : 125 * l^3 = L^3) :
  6 * l^2 = 24 := by
  sorry

end surface_area_of_each_smaller_cube_l2011_201140


namespace total_surface_area_first_rectangular_parallelepiped_equals_22_l2011_201173

theorem total_surface_area_first_rectangular_parallelepiped_equals_22
  (x y z : ℝ)
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) :
  2 * (x * y + x * z + y * z) = 22 := sorry

end total_surface_area_first_rectangular_parallelepiped_equals_22_l2011_201173


namespace fraction_of_income_to_taxes_l2011_201117

noncomputable def joe_income : ℕ := 2120
noncomputable def joe_taxes : ℕ := 848

theorem fraction_of_income_to_taxes : (joe_taxes / gcd joe_taxes joe_income) / (joe_income / gcd joe_taxes joe_income) = 106 / 265 := sorry

end fraction_of_income_to_taxes_l2011_201117


namespace fraction_multiplication_l2011_201191

theorem fraction_multiplication :
  (2 / 3) * (3 / 8) = (1 / 4) :=
sorry

end fraction_multiplication_l2011_201191


namespace money_needed_to_finish_collection_l2011_201142

-- Define the conditions
def initial_action_figures : ℕ := 9
def total_action_figures_needed : ℕ := 27
def cost_per_action_figure : ℕ := 12

-- Define the goal
theorem money_needed_to_finish_collection 
  (initial : ℕ) (total_needed : ℕ) (cost_per : ℕ) 
  (h1 : initial = initial_action_figures)
  (h2 : total_needed = total_action_figures_needed)
  (h3 : cost_per = cost_per_action_figure) :
  ((total_needed - initial) * cost_per = 216) := 
by
  sorry

end money_needed_to_finish_collection_l2011_201142


namespace trig_identity_solution_l2011_201174

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Statement of the theorem
theorem trig_identity_solution (x : ℝ) (k : ℤ) (hcos : Real.cos x ≠ 0) (hsin : Real.sin x ≠ 0) :
  (Real.sin x) ^ 2 * tan x + (Real.cos x) ^ 2 * cot x + 2 * Real.sin x * Real.cos x = (4 * Real.sqrt 3) / 3 →
  ∃ k : ℤ, x = (-1) ^ k * (Real.pi / 6) + (Real.pi / 2) :=
sorry

end trig_identity_solution_l2011_201174


namespace verify_exact_countries_attended_l2011_201110

theorem verify_exact_countries_attended :
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  (attended_countries = 68) :=
by
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  have : attended_countries = 68 := rfl
  exact this

end verify_exact_countries_attended_l2011_201110


namespace sqrt_221_range_l2011_201141

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_range_l2011_201141


namespace who_plays_piano_l2011_201196

theorem who_plays_piano 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (hA : A = True)
  (hB : B = False)
  (hC : A = False)
  (only_one_true : (A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)) : B = True := 
sorry

end who_plays_piano_l2011_201196


namespace smallest_number_among_given_l2011_201136

theorem smallest_number_among_given :
  ∀ (a b c d : ℚ), a = -2 → b = -5/2 → c = 0 → d = 1/5 →
  (min (min (min a b) c) d) = b :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_among_given_l2011_201136


namespace calculate_x_l2011_201116

theorem calculate_x :
  let a := 3
  let b := 5
  let c := 2
  let d := 4
  let term1 := (a ^ 2) * b * 0.47 * 1442
  let term2 := c * d * 0.36 * 1412
  (term1 - term2) + 63 = 26544.74 := by
  sorry

end calculate_x_l2011_201116


namespace find_v_l2011_201158

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![0, 1]]

def v : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![3], ![1]]

def target : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![15], ![5]]

theorem find_v :
  let B2 := B * B
  let B3 := B2 * B
  let B4 := B3 * B
  (B4 + B3 + B2 + B + (1 : Matrix (Fin 2) (Fin 2) ℚ)) * v = target :=
by
  sorry

end find_v_l2011_201158


namespace parabola_at_point_has_value_zero_l2011_201146

theorem parabola_at_point_has_value_zero (a m : ℝ) :
  (x ^ 2 + (a + 1) * x + a) = 0 -> m = 0 :=
by
  -- We know the parabola passes through the point (-1, m)
  sorry

end parabola_at_point_has_value_zero_l2011_201146


namespace tea_customers_count_l2011_201124

theorem tea_customers_count :
  ∃ T : ℕ, 7 * 5 + T * 4 = 67 ∧ T = 8 :=
by
  sorry

end tea_customers_count_l2011_201124


namespace sin_cos_ratio_value_sin_cos_expression_value_l2011_201125

variable (α : ℝ)

-- Given condition
def tan_alpha_eq_3 := Real.tan α = 3

-- Goal (1)
theorem sin_cos_ratio_value 
  (h : tan_alpha_eq_3 α) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 4 / 5 := 
  sorry

-- Goal (2)
theorem sin_cos_expression_value
  (h : tan_alpha_eq_3 α) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 15 := 
  sorry

end sin_cos_ratio_value_sin_cos_expression_value_l2011_201125


namespace max_value_of_z_l2011_201151

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y - 5 ≥ 0) (h2 : x - 2 * y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∃ x y, x + y = 9 :=
by {
  sorry
}

end max_value_of_z_l2011_201151


namespace inequality_iff_positive_l2011_201138

variable (x y : ℝ)

theorem inequality_iff_positive :
  x + y > abs (x - y) ↔ x > 0 ∧ y > 0 :=
sorry

end inequality_iff_positive_l2011_201138


namespace find_C_l2011_201178

theorem find_C (A B C : ℕ) (h0 : 3 * A - A = 10) (h1 : B + A = 12) (h2 : C - B = 6) (h3 : A ≠ B) (h4 : B ≠ C) (h5 : C ≠ A) 
: C = 13 :=
sorry

end find_C_l2011_201178


namespace original_number_of_men_l2011_201179

theorem original_number_of_men (M : ℕ) : 
  (∀ t : ℕ, (t = 8) -> (8:ℕ) * M = 8 * 10 / (M - 3) ) -> ( M = 12 ) :=
by sorry

end original_number_of_men_l2011_201179


namespace speed_of_train_l2011_201120

-- Define the given conditions
def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def time_to_cross_bridge : ℝ := 60

-- Define the speed conversion factor
def m_per_s_to_km_per_h : ℝ := 3.6

-- Prove that the speed of the train is 18 km/h
theorem speed_of_train :
  (length_of_bridge + length_of_train) / time_to_cross_bridge * m_per_s_to_km_per_h = 18 :=
by
  sorry

end speed_of_train_l2011_201120


namespace Gake_needs_fewer_boards_than_Tom_l2011_201105

noncomputable def Tom_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (3 * width_char + 2 * 6) / width_board

noncomputable def Gake_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (4 * width_char + 3 * 1) / width_board

theorem Gake_needs_fewer_boards_than_Tom :
  Gake_boards_needed < Tom_boards_needed :=
by
  -- Here you will put the actual proof steps
  sorry

end Gake_needs_fewer_boards_than_Tom_l2011_201105


namespace no_integer_n_such_that_squares_l2011_201161

theorem no_integer_n_such_that_squares :
  ¬ ∃ n : ℤ, (∃ k1 : ℤ, 10 * n - 1 = k1 ^ 2) ∧
             (∃ k2 : ℤ, 13 * n - 1 = k2 ^ 2) ∧
             (∃ k3 : ℤ, 85 * n - 1 = k3 ^ 2) := 
by sorry

end no_integer_n_such_that_squares_l2011_201161


namespace geo_seq_sum_S4_l2011_201130

noncomputable def geom_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geo_seq_sum_S4 {a : ℝ} {q : ℝ} (h1 : a * q^2 - a = 15) (h2 : a * q - a = 5) :
  geom_seq_sum a q 4 = 75 :=
by
  sorry

end geo_seq_sum_S4_l2011_201130


namespace simplify_fraction_rationalize_denominator_l2011_201184

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end simplify_fraction_rationalize_denominator_l2011_201184


namespace fred_money_last_week_l2011_201111

-- Definitions for the conditions in the problem
variables {f j : ℕ} (current_fred : ℕ) (current_jason : ℕ) (last_week_jason : ℕ)
variable (earning : ℕ)

-- Conditions
axiom Fred_current_money : current_fred = 115
axiom Jason_current_money : current_jason = 44
axiom Jason_last_week_money : last_week_jason = 40
axiom Earning_amount : earning = 4

-- Theorem statement: prove Fred's money last week
theorem fred_money_last_week (current_fred last_week_jason current_jason earning : ℕ)
  (Fred_current_money : current_fred = 115)
  (Jason_current_money : current_jason = 44)
  (Jason_last_week_money : last_week_jason = 40)
  (Earning_amount : earning = 4)
  : current_fred - earning = 111 :=
sorry

end fred_money_last_week_l2011_201111


namespace max_halls_visited_l2011_201122

theorem max_halls_visited (side_len large_tri small_tri: ℕ) 
  (h1 : side_len = 100)
  (h2 : large_tri = 100)
  (h3 : small_tri = 10)
  (div : large_tri = (side_len / small_tri) ^ 2) :
  ∃ m : ℕ, m = 91 → m ≤ large_tri - 9 := 
sorry

end max_halls_visited_l2011_201122


namespace C_converges_l2011_201162

noncomputable def behavior_of_C (e R r : ℝ) (n : ℕ) : ℝ := e * (n^2) / (R + n * (r^2))

theorem C_converges (e R r : ℝ) (h₁ : 0 < r) : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |behavior_of_C e R r n - e / r^2| < ε := 
sorry

end C_converges_l2011_201162


namespace abs_eq_neg_imp_nonpos_l2011_201112

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l2011_201112


namespace third_month_sale_l2011_201189

theorem third_month_sale (s3 : ℝ)
  (s1 s2 s4 s5 s6 : ℝ)
  (h1 : s1 = 2435)
  (h2 : s2 = 2920)
  (h4 : s4 = 3230)
  (h5 : s5 = 2560)
  (h6 : s6 = 1000)
  (average : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 2500) :
  s3 = 2855 := 
by sorry

end third_month_sale_l2011_201189


namespace pencils_placed_by_dan_l2011_201169

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l2011_201169


namespace x_gt_neg2_is_necessary_for_prod_lt_0_l2011_201134

theorem x_gt_neg2_is_necessary_for_prod_lt_0 (x : Real) :
  (x > -2) ↔ (((x + 2) * (x - 3)) < 0) → (x > -2) :=
by
  sorry

end x_gt_neg2_is_necessary_for_prod_lt_0_l2011_201134


namespace g_g1_eq_43_l2011_201195

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem g_g1_eq_43 : g (g 1) = 43 :=
by
  sorry

end g_g1_eq_43_l2011_201195


namespace binary_operation_l2011_201127

-- Definitions of the binary numbers.
def a : ℕ := 0b10110      -- 10110_2 in base 10
def b : ℕ := 0b10100      -- 10100_2 in base 10
def c : ℕ := 0b10         -- 10_2 in base 10
def result : ℕ := 0b11011100 -- 11011100_2 in base 10

-- The theorem to be proven
theorem binary_operation : (a * b) / c = result := by
  -- Placeholder for the proof
  sorry

end binary_operation_l2011_201127


namespace solve_fraction_eq_l2011_201188

theorem solve_fraction_eq (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_eq_l2011_201188


namespace non_vegan_gluten_cupcakes_eq_28_l2011_201170

def total_cupcakes : ℕ := 80
def gluten_free_cupcakes : ℕ := total_cupcakes / 2
def vegan_cupcakes : ℕ := 24
def vegan_gluten_free_cupcakes : ℕ := vegan_cupcakes / 2
def non_vegan_cupcakes : ℕ := total_cupcakes - vegan_cupcakes
def gluten_cupcakes : ℕ := total_cupcakes - gluten_free_cupcakes
def non_vegan_gluten_cupcakes : ℕ := gluten_cupcakes - vegan_gluten_free_cupcakes

theorem non_vegan_gluten_cupcakes_eq_28 :
  non_vegan_gluten_cupcakes = 28 := by
  sorry

end non_vegan_gluten_cupcakes_eq_28_l2011_201170


namespace inequality_proof_l2011_201101

noncomputable def problem_statement (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : Prop :=
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z))) ≤ 
  ((x + y + z) / 3) ^ (5 / 8)

-- The statement below is what needs to be proven.
theorem inequality_proof (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : problem_statement x y z positive_x positive_y positive_z condition :=
sorry

end inequality_proof_l2011_201101


namespace robert_balls_l2011_201192

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l2011_201192


namespace initial_walking_speed_l2011_201181

open Real

theorem initial_walking_speed :
  ∃ (v : ℝ), (∀ (d : ℝ), d = 9.999999999999998 →
  (∀ (lateness_time : ℝ), lateness_time = 10 / 60 →
  ((d / v) - (d / 15) = lateness_time + lateness_time)) → v = 11.25) :=
by
  sorry

end initial_walking_speed_l2011_201181


namespace area_of_triangle_l2011_201107

theorem area_of_triangle (m : ℝ) 
  (h : ∀ x y : ℝ, ((m + 3) * x + y = 3 * m - 4) → 
                  (7 * x + (5 - m) * y - 8 ≠ 0)
  ) : ((m = -2) → (1/2) * 2 * 2 = 2) := 
by {
  sorry
}

end area_of_triangle_l2011_201107


namespace product_bases_l2011_201185

def base2_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 2 + (d.toNat - '0'.toNat)) 0

def base3_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 3 + (d.toNat - '0'.toNat)) 0

def base4_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 4 + (d.toNat - '0'.toNat)) 0

theorem product_bases :
  base2_to_nat "1101" * base3_to_nat "202" * base4_to_nat "22" = 2600 :=
by
  sorry

end product_bases_l2011_201185


namespace empty_pipe_time_l2011_201121

theorem empty_pipe_time (R1 R2 : ℚ) (t1 t2 t_total : ℕ) (h1 : t1 = 60) (h2 : t_total = 180) (H1 : R1 = 1 / t1) (H2 : R1 - R2 = 1 / t_total) :
  1 / R2 = 90 :=
by
  sorry

end empty_pipe_time_l2011_201121


namespace area_of_annulus_l2011_201186

-- Define the conditions
def concentric_circles (r s : ℝ) (h : r > s) (x : ℝ) := 
  r^2 = s^2 + x^2

-- State the theorem
theorem area_of_annulus (r s x : ℝ) (h : r > s) (h₁ : concentric_circles r s h x) :
  π * x^2 = π * r^2 - π * s^2 :=
by 
  rw [concentric_circles] at h₁
  sorry

end area_of_annulus_l2011_201186


namespace people_to_right_of_taehyung_l2011_201129

-- Given conditions
def total_people : Nat := 11
def people_to_left_of_taehyung : Nat := 5

-- Question and proof: How many people are standing to Taehyung's right?
theorem people_to_right_of_taehyung : total_people - people_to_left_of_taehyung - 1 = 4 :=
by
  sorry

end people_to_right_of_taehyung_l2011_201129


namespace cafeteria_apples_pies_l2011_201199

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end cafeteria_apples_pies_l2011_201199


namespace cube_difference_positive_l2011_201137

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l2011_201137


namespace sticker_distribution_ways_l2011_201165

theorem sticker_distribution_ways : 
  ∃ ways : ℕ, ways = Nat.choose (9) (4) ∧ ways = 126 :=
by
  sorry

end sticker_distribution_ways_l2011_201165


namespace problem_part_1_problem_part_2_l2011_201104

theorem problem_part_1 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) : 
  1 - p_A ^ 3 = 19 / 27 :=
by sorry

theorem problem_part_2 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) 
  (h1 : 3 * (p_A ^ 2) * (1 - p_A) = 4 / 9)
  (h2 : 3 * p_B * ((1 - p_B) ^ 2) = 9 / 64) : 
  (4 / 9) * (9 / 64) = 1 / 16 :=
by sorry

end problem_part_1_problem_part_2_l2011_201104


namespace problem1_problem2_problem3_problem4_l2011_201126

-- Problem 1
theorem problem1 : (-3 / 8) + ((-5 / 8) * (-6)) = 27 / 8 :=
by sorry

-- Problem 2
theorem problem2 : 12 + (7 * (-3)) - (18 / (-3)) = -3 :=
by sorry

-- Problem 3
theorem problem3 : -((2:ℤ)^2) - (4 / 7) * (2:ℚ) - (-((3:ℤ)^2:ℤ) : ℤ) = -99 / 7 :=
by sorry

-- Problem 4
theorem problem4 : -(((-1) ^ 2020 : ℤ)) + ((6 : ℚ) / (-(2 : ℤ) ^ 3)) * (-1 / 3) = -3 / 4 :=
by sorry

end problem1_problem2_problem3_problem4_l2011_201126


namespace mod_remainder_l2011_201109

open Int

theorem mod_remainder (n : ℤ) : 
  (1125 * 1127 * n) % 12 = 3 ↔ n % 12 = 1 :=
by
  sorry

end mod_remainder_l2011_201109


namespace bus_driver_total_compensation_l2011_201159

-- Definitions of conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours : ℝ := 65
def total_compensation : ℝ := (regular_rate * regular_hours) + (overtime_rate * (total_hours - regular_hours))

-- Theorem stating the total compensation
theorem bus_driver_total_compensation : total_compensation = 1340 :=
by
  sorry

end bus_driver_total_compensation_l2011_201159


namespace quadratic_roots_always_implies_l2011_201157

variable {k x1 x2 : ℝ}

theorem quadratic_roots_always_implies (h1 : k^2 > 16) 
  (h2 : x1 + x2 = -k)
  (h3 : x1 * x2 = 4) : x1^2 + x2^2 > 8 :=
by
  sorry

end quadratic_roots_always_implies_l2011_201157


namespace minimum_S_l2011_201153

theorem minimum_S (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  S = (a + 1/a)^2 + (b + 1/b)^2 → S ≥ 8 :=
by
  sorry

end minimum_S_l2011_201153


namespace solution_set_inequality_l2011_201123

theorem solution_set_inequality (x : ℝ) : 3 * x - 2 > x → x > 1 := by
  sorry

end solution_set_inequality_l2011_201123


namespace abs_f_sub_lt_abs_l2011_201145

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem abs_f_sub_lt_abs (a b : ℝ) (h : a ≠ b) : 
  |f a - f b| < |a - b| := 
by
  sorry

end abs_f_sub_lt_abs_l2011_201145


namespace cone_height_l2011_201149

theorem cone_height (V : ℝ) (h r : ℝ) (π : ℝ) (h_eq_r : h = r) (volume_eq : V = 12288 * π) (V_def : V = (1/3) * π * r^3) : h = 36 := 
by
  sorry

end cone_height_l2011_201149


namespace shopkeeper_loss_percent_l2011_201172

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_percent : ℝ)
  (loss_percent : ℝ)
  (remaining_value_percent : ℝ)
  (profit_percent_10 : profit_percent = 0.10)
  (loss_percent_70 : loss_percent = 0.70)
  (initial_value_100 : initial_value = 100)
  (remaining_value_percent_30 : remaining_value_percent = 0.30)
  (selling_price : ℝ := initial_value * (1 + profit_percent))
  (remaining_value : ℝ := initial_value * remaining_value_percent)
  (remaining_selling_price : ℝ := remaining_value * (1 + profit_percent))
  (loss_value : ℝ := initial_value - remaining_selling_price)
  (shopkeeper_loss_percent : ℝ := loss_value / initial_value * 100) : 
  shopkeeper_loss_percent = 67 :=
sorry

end shopkeeper_loss_percent_l2011_201172


namespace birds_are_crows_l2011_201160

theorem birds_are_crows (total_birds pigeons crows sparrows parrots non_pigeons: ℕ)
    (h1: pigeons = 20)
    (h2: crows = 40)
    (h3: sparrows = 15)
    (h4: parrots = total_birds - pigeons - crows - sparrows)
    (h5: total_birds = pigeons + crows + sparrows + parrots)
    (h6: non_pigeons = total_birds - pigeons) :
    (crows * 100 / non_pigeons = 50) :=
by sorry

end birds_are_crows_l2011_201160


namespace estimate_shaded_area_l2011_201180

theorem estimate_shaded_area 
  (side_length : ℝ)
  (points_total : ℕ)
  (points_shaded : ℕ)
  (area_shaded_estimation : ℝ) :
  side_length = 6 →
  points_total = 800 →
  points_shaded = 200 →
  area_shaded_estimation = (36 * (200 / 800)) →
  area_shaded_estimation = 9 :=
by
  intros h_side_length h_points_total h_points_shaded h_area_shaded_estimation
  rw [h_side_length, h_points_total, h_points_shaded] at *
  norm_num at h_area_shaded_estimation
  exact h_area_shaded_estimation

end estimate_shaded_area_l2011_201180


namespace equation_of_line_l2011_201128

theorem equation_of_line (x y : ℝ) 
  (l1 : 4 * x + y + 6 = 0) 
  (l2 : 3 * x - 5 * y - 6 = 0) 
  (midpoint_origin : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * x₁ + y₁ + 6 = 0) ∧ 
    (3 * x₂ - 5 * y₂ - 6 = 0) ∧ 
    (x₁ + x₂ = 0) ∧ 
    (y₁ + y₂ = 0)) : 
  7 * x + 4 * y = 0 :=
sorry

end equation_of_line_l2011_201128


namespace b_remaining_work_days_l2011_201154

-- Definitions of the conditions
def together_work (a b: ℕ) := a + b = 12
def alone_work (a: ℕ) := a = 20
def c_work (c: ℕ) := c = 30
def initial_work_days := 5

-- Question to prove:
theorem b_remaining_work_days (a b c : ℕ) (h1 : together_work a b) (h2 : alone_work a) (h3 : c_work c) : 
  let b_rate := 1 / 30 
  let remaining_work := 25 / 60
  let work_to_days := remaining_work / b_rate
  work_to_days = 12.5 := 
sorry

end b_remaining_work_days_l2011_201154


namespace problem1_problem2_l2011_201164

open Set

variable (a : Real)

-- Problem 1: Prove the intersection M ∩ (C_R N) equals the given set
theorem problem1 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | 3 ≤ x ∧ x ≤ 5 }
  let C_RN := { x : ℝ | x < 3 ∨ 5 < x }
  M ∩ C_RN = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

-- Problem 2: Prove the range of values for a such that M ∪ N = M
theorem problem2 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | a+1 ≤ x ∧ x ≤ 2*a+1 }
  (M ∪ N = M) → a ≤ 2 :=
by
  sorry

end problem1_problem2_l2011_201164


namespace initial_oranges_l2011_201108

theorem initial_oranges (O : ℕ) (h1 : (1 / 4 : ℚ) * (1 / 2 : ℚ) * O = 39) (h2 : (1 / 8 : ℚ) * (1 / 2 : ℚ) * O = 4 + 78 - (1 / 4 : ℚ) * (1 / 2 : ℚ) * O) :
  O = 96 :=
by
  sorry

end initial_oranges_l2011_201108
