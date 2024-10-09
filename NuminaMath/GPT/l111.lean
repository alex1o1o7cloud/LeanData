import Mathlib

namespace length_of_la_l111_11138

variables {A b c l_a: ℝ}
variables (S_ABC S_ACA' S_ABA': ℝ)

axiom area_of_ABC: S_ABC = (1 / 2) * b * c * Real.sin A
axiom area_of_ACA: S_ACA' = (1 / 2) * b * l_a * Real.sin (A / 2)
axiom area_of_ABA: S_ABA' = (1 / 2) * c * l_a * Real.sin (A / 2)
axiom sin_double_angle: Real.sin A = 2 * Real.sin (A / 2) * Real.cos (A / 2)

theorem length_of_la :
  l_a = (2 * b * c * Real.cos (A / 2)) / (b + c) :=
sorry

end length_of_la_l111_11138


namespace total_shoes_l111_11125

variables (people : ℕ) (shoes_per_person : ℕ)

-- There are 10 people
axiom h1 : people = 10
-- Each person has 2 shoes
axiom h2 : shoes_per_person = 2

-- The total number of shoes kept outside the library is 10 * 2 = 20
theorem total_shoes (people shoes_per_person : ℕ) (h1 : people = 10) (h2 : shoes_per_person = 2) : people * shoes_per_person = 20 :=
by sorry

end total_shoes_l111_11125


namespace sum_divides_product_iff_l111_11153

theorem sum_divides_product_iff (n : ℕ) : 
  (n*(n+1)/2) ∣ n! ↔ ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ a * b = n + 1 ∧ a ≤ n ∧ b ≤ n :=
sorry

end sum_divides_product_iff_l111_11153


namespace area_of_diamond_l111_11106

theorem area_of_diamond (x y : ℝ) : (|x / 2| + |y / 2| = 1) → 
∃ (area : ℝ), area = 8 :=
by sorry

end area_of_diamond_l111_11106


namespace speed_of_freight_train_l111_11186

-- Definitions based on the conditions
def distance := 390  -- The towns are 390 km apart
def express_speed := 80  -- The express train travels at 80 km per hr
def travel_time := 3  -- They pass one another 3 hr later

-- The freight train travels 30 km per hr slower than the express train
def freight_speed := express_speed - 30

-- The statement that we aim to prove:
theorem speed_of_freight_train : freight_speed = 50 := 
by 
  sorry

end speed_of_freight_train_l111_11186


namespace perpendicular_lines_l111_11147

theorem perpendicular_lines {a : ℝ} :
  a*(a-1) + (1-a)*(2*a+3) = 0 → (a = 1 ∨ a = -3) := 
by
  intro h
  sorry

end perpendicular_lines_l111_11147


namespace sum_of_non_solutions_l111_11130

theorem sum_of_non_solutions (A B C x : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 32)) = 4 * ((x + C) * (x + 8))) :
  (x = -B ∨ x = -8) → x ≠ -B → -B ≠ -8 → x ≠ -8 → x + 8 + B = 0 := 
sorry

end sum_of_non_solutions_l111_11130


namespace prove_product_of_b_l111_11166

noncomputable def g (x b : ℝ) := b / (5 * x - 7)

noncomputable def g_inv (y b : ℝ) := (b + 7 * y) / (5 * y)

theorem prove_product_of_b (b1 b2 : ℝ) (h1 : g 3 b1 = g_inv (b1 + 2) b1) (h2 : g 3 b2 = g_inv (b2 + 2) b2) :
  b1 * b2 = -22.39 := by
  sorry

end prove_product_of_b_l111_11166


namespace num_4_digit_odd_distinct_l111_11113

theorem num_4_digit_odd_distinct : 
  ∃ n : ℕ, n = 5 * 4 * 3 * 2 :=
sorry

end num_4_digit_odd_distinct_l111_11113


namespace paths_via_checkpoint_l111_11151

/-- Define the grid configuration -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Calculate the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  n.choose k

/-- Define points A, B, C -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 4⟩
def C : Point := ⟨3, 2⟩

/-- Calculate number of paths from A to C -/
def paths_A_to_C : ℕ :=
  binomial (3 + 2) 2

/-- Calculate number of paths from C to B -/
def paths_C_to_B : ℕ :=
  binomial (2 + 2) 2

/-- Calculate total number of paths from A to B via C -/
def total_paths_A_to_B_via_C : ℕ :=
  (paths_A_to_C * paths_C_to_B)

theorem paths_via_checkpoint :
  total_paths_A_to_B_via_C = 60 :=
by
  -- The proof is skipped as per the instruction
  sorry

end paths_via_checkpoint_l111_11151


namespace algebra_expression_value_l111_11144

theorem algebra_expression_value (a : ℝ) (h : a^2 - 4 * a - 6 = 0) : a^2 - 4 * a + 3 = 9 :=
by
  sorry

end algebra_expression_value_l111_11144


namespace calculate_percentage_l111_11121

/-- A candidate got a certain percentage of the votes polled and he lost to his rival by 2000 votes.
There were 10,000.000000000002 votes cast. What percentage of the votes did the candidate get? --/

def candidate_vote_percentage (P : ℝ) (total_votes : ℝ) (rival_margin : ℝ) : Prop :=
  (P / 100 * total_votes = total_votes - rival_margin) → P = 80

theorem calculate_percentage:
  candidate_vote_percentage P 10000.000000000002 2000 := 
by 
  sorry

end calculate_percentage_l111_11121


namespace company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l111_11191

-- Define the conditions about the fishing company's boat purchase and expenses
def initial_purchase_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def expense_increment : ℕ := 40000
def annual_income : ℕ := 500000

-- Prove that the company starts to make a profit in the third year
theorem company_starts_to_make_profit_in_third_year : 
  ∃ (year : ℕ), year = 3 ∧ 
  annual_income * year > initial_purchase_cost + first_year_expenses + (expense_increment * (year - 1) * year / 2) :=
sorry

-- Prove that the first option is more cost-effective
theorem first_option_more_cost_effective : 
  (annual_income * 3 - (initial_purchase_cost + first_year_expenses + expense_increment * (3 - 1) * 3 / 2) + 260000) > 
  (annual_income * 5 - (initial_purchase_cost + first_year_expenses + expense_increment * (5 - 1) * 5 / 2) + 80000) :=
sorry

end company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l111_11191


namespace number_of_teams_in_league_l111_11167

theorem number_of_teams_in_league (n : ℕ) :
  (6 * n * (n - 1)) / 2 = 396 ↔ n = 12 :=
by
  sorry

end number_of_teams_in_league_l111_11167


namespace find_number_of_students_l111_11115

theorem find_number_of_students (N : ℕ) (T : ℕ) (hN : N ≠ 0) (hT : T = 80 * N) 
  (h_avg_excluded : (T - 200) / (N - 5) = 90) : N = 25 :=
by
  sorry

end find_number_of_students_l111_11115


namespace hens_count_l111_11169

theorem hens_count (H R : ℕ) (h₁ : H = 9 * R - 5) (h₂ : H + R = 75) : H = 67 :=
by {
  sorry
}

end hens_count_l111_11169


namespace hemisphere_surface_area_l111_11133

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l111_11133


namespace ham_block_cut_mass_distribution_l111_11149

theorem ham_block_cut_mass_distribution
  (length width height : ℝ) (mass : ℝ)
  (parallelogram_side1 parallelogram_side2 : ℝ)
  (condition1 : length = 12) 
  (condition2 : width = 12) 
  (condition3 : height = 35)
  (condition4 : mass = 5)
  (condition5 : parallelogram_side1 = 15) 
  (condition6 : parallelogram_side2 = 20) :
  ∃ (mass_piece1 mass_piece2 : ℝ),
    mass_piece1 = 1.7857 ∧ mass_piece2 = 3.2143 :=
by
  sorry

end ham_block_cut_mass_distribution_l111_11149


namespace fraction_inhabitable_l111_11172

-- Define the constants based on the given conditions
def fraction_water : ℚ := 3 / 5
def fraction_inhabitable_land : ℚ := 3 / 4

-- Define the theorem to prove that the fraction of Earth's surface that is inhabitable is 3/10
theorem fraction_inhabitable (w h : ℚ) (hw : w = fraction_water) (hh : h = fraction_inhabitable_land) : 
  (1 - w) * h = 3 / 10 :=
by
  sorry

end fraction_inhabitable_l111_11172


namespace value_of_4_and_2_l111_11160

noncomputable def custom_and (a b : ℕ) : ℕ :=
  ((a + b) * (a - b)) ^ 2

theorem value_of_4_and_2 : custom_and 4 2 = 144 :=
  sorry

end value_of_4_and_2_l111_11160


namespace stripe_area_l111_11184

-- Definitions based on conditions
def diameter : ℝ := 40
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- The statement we want to prove
theorem stripe_area (π : ℝ) : 
  (revolutions * π * diameter * stripe_width) = 480 * π :=
by
  sorry

end stripe_area_l111_11184


namespace circle_diameter_l111_11145

theorem circle_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) (h1 : A = 64 * π) (h2 : A = π * r^2) (h3 : d = 2 * r) :
  d = 16 :=
by
  sorry

end circle_diameter_l111_11145


namespace length_CD_l111_11196

-- Given data
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (AB BC : ℝ)

noncomputable def triangle_ABC : Prop :=
  AB = 5 ∧ BC = 7 ∧ ∃ (angle_ABC : ℝ), angle_ABC = 90

-- The target condition to prove
theorem length_CD {CD : ℝ} (h : triangle_ABC AB BC) : CD = 7 :=
by {
  -- proof would be here
  sorry
}

end length_CD_l111_11196


namespace max_isosceles_tris_2017_gon_l111_11142

theorem max_isosceles_tris_2017_gon :
  ∀ (n : ℕ), n = 2017 →
  ∃ (t : ℕ), (∃ (d : ℕ), d = 2014 ∧ 2015 = (n - 2)) →
  t = 2010 :=
by
  sorry

end max_isosceles_tris_2017_gon_l111_11142


namespace permutations_eq_factorial_l111_11175

theorem permutations_eq_factorial (n : ℕ) : 
  (∃ Pn : ℕ, Pn = n!) := 
sorry

end permutations_eq_factorial_l111_11175


namespace complement_A_A_inter_complement_B_l111_11183

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem complement_A : compl A = {x | x ≤ 1 ∨ 4 ≤ x} :=
by sorry

theorem A_inter_complement_B : A ∩ compl B = {x | 3 < x ∧ x < 4} :=
by sorry

end complement_A_A_inter_complement_B_l111_11183


namespace angle_C_in_triangle_ABC_l111_11185

noncomputable def find_angle_C (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : Prop :=
  C = Real.pi / 6

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : find_angle_C A B C h1 h2 h3 :=
by
  -- proof omitted
  sorry

end angle_C_in_triangle_ABC_l111_11185


namespace triangle_area_l111_11143

-- Define the conditions and problem
def BC : ℝ := 10
def height_from_A : ℝ := 12
def AC : ℝ := 13

-- State the main theorem
theorem triangle_area (BC height_from_A AC : ℝ) (hBC : BC = 10) (hheight : height_from_A = 12) (hAC : AC = 13) : 
  (1/2 * BC * height_from_A) = 60 :=
by 
  -- Insert the proof
  sorry

end triangle_area_l111_11143


namespace shaded_area_l111_11126

-- Define the problem in Lean
theorem shaded_area (area_large_square area_small_square : ℝ) (H_large_square : area_large_square = 10) (H_small_square : area_small_square = 4) (diagonals_contain : True) : 
  (area_large_square - area_small_square) / 4 = 1.5 :=
by
  sorry -- proof not required

end shaded_area_l111_11126


namespace sum_of_coordinates_l111_11120

-- Define the given conditions as hypotheses
def isThreeUnitsFromLine (x y : ℝ) : Prop := y = 18 ∨ y = 12
def isTenUnitsFromPoint (x y : ℝ) : Prop := (x - 5)^2 + (y - 15)^2 = 100

-- We aim to prove the sum of the coordinates of the points satisfying these conditions
theorem sum_of_coordinates (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : isThreeUnitsFromLine x1 y1 ∧ isTenUnitsFromPoint x1 y1)
  (h2 : isThreeUnitsFromLine x2 y2 ∧ isTenUnitsFromPoint x2 y2)
  (h3 : isThreeUnitsFromLine x3 y3 ∧ isTenUnitsFromPoint x3 y3)
  (h4 : isThreeUnitsFromLine x4 y4 ∧ isTenUnitsFromPoint x4 y4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 50 :=
  sorry

end sum_of_coordinates_l111_11120


namespace smallest_digits_to_append_l111_11197

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l111_11197


namespace arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l111_11116

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) :
  ∀ n : ℕ, a n = 5 - 2 * n :=
by
  sorry

theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) (h_sum : ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2) :
  S 2 = 4 :=
by
  sorry

end arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l111_11116


namespace range_of_a_l111_11140

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < a + 2 → x ≤ 2) ↔ a ≤ 0 := by
  sorry

end range_of_a_l111_11140


namespace arithmetic_sequence_sum_l111_11135

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 1)
  (h2 : a 2 + a 3 = 9) :
  a 4 + a 5 = 17 :=
sorry

end arithmetic_sequence_sum_l111_11135


namespace fourth_guard_run_distance_l111_11108

-- Define the rectangle's dimensions
def length : ℝ := 300
def width : ℝ := 200

-- Define the perimeter of the rectangle
def perimeter : ℝ := 2 * (length + width)

-- Given the sum of the distances run by three guards
def sum_of_three_guards : ℝ := 850

-- The fourth guard's distance is what we need to prove
def fourth_guard_distance := perimeter - sum_of_three_guards

-- The proof goal: we need to show that the fourth guard's distance is 150 meters
theorem fourth_guard_run_distance : fourth_guard_distance = 150 := by
  sorry  -- This placeholder means that the proof is omitted

end fourth_guard_run_distance_l111_11108


namespace probability_MAME_on_top_l111_11163

theorem probability_MAME_on_top : 
  let num_sections := 8
  let favorable_outcome := 1
  (favorable_outcome : ℝ) / (num_sections : ℝ) = 1 / 8 :=
by 
  sorry

end probability_MAME_on_top_l111_11163


namespace product_of_numbers_l111_11128

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x * y = 375 :=
sorry

end product_of_numbers_l111_11128


namespace Ahmed_total_distance_traveled_l111_11157

/--
Ahmed stops one-quarter of the way to the store.
He continues for 12 km to reach the store.
Prove that the total distance Ahmed travels is 16 km.
-/
theorem Ahmed_total_distance_traveled
  (D : ℝ) (h1 : D > 0)  -- D is the total distance to the store, assumed to be positive
  (h_stop : D / 4 + 12 = D) : D = 16 := 
sorry

end Ahmed_total_distance_traveled_l111_11157


namespace candidate_X_votes_l111_11124

theorem candidate_X_votes (Z : ℕ) (Y : ℕ) (X : ℕ) (hZ : Z = 25000) 
                          (hY : Y = Z - (2 / 5) * Z) 
                          (hX : X = Y + (1 / 2) * Y) : 
                          X = 22500 :=
by
  sorry

end candidate_X_votes_l111_11124


namespace X_investment_l111_11148

theorem X_investment (P : ℝ) 
  (Y_investment : ℝ := 42000)
  (Z_investment : ℝ := 48000)
  (Z_joins_at : ℝ := 4)
  (total_profit : ℝ := 14300)
  (Z_share : ℝ := 4160) :
  (P * 12 / (P * 12 + Y_investment * 12 + Z_investment * (12 - Z_joins_at))) * total_profit = Z_share → P = 35700 :=
by
  sorry

end X_investment_l111_11148


namespace weight_of_172_is_around_60_316_l111_11100

noncomputable def weight_prediction (x : ℝ) : ℝ := 0.849 * x - 85.712

theorem weight_of_172_is_around_60_316 :
  ∀ (x : ℝ), x = 172 → abs (weight_prediction x - 60.316) < 1 :=
by
  sorry

end weight_of_172_is_around_60_316_l111_11100


namespace cook_one_potato_l111_11122

theorem cook_one_potato (total_potatoes cooked_potatoes remaining_potatoes remaining_time : ℕ) 
  (h1 : total_potatoes = 15) 
  (h2 : cooked_potatoes = 6) 
  (h3 : remaining_time = 72)
  (h4 : remaining_potatoes = total_potatoes - cooked_potatoes) :
  (remaining_time / remaining_potatoes) = 8 :=
by
  sorry

end cook_one_potato_l111_11122


namespace product_of_consecutive_integers_l111_11182

theorem product_of_consecutive_integers (n : ℤ) :
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) :=
by
  sorry

end product_of_consecutive_integers_l111_11182


namespace warehouse_capacity_l111_11102

theorem warehouse_capacity (total_bins : ℕ) (bins_20_tons : ℕ) (bins_15_tons : ℕ)
    (total_capacity : ℕ) (h1 : total_bins = 30) (h2 : bins_20_tons = 12) 
    (h3 : bins_15_tons = total_bins - bins_20_tons) 
    (h4 : total_capacity = (bins_20_tons * 20) + (bins_15_tons * 15)) : 
    total_capacity = 510 :=
by {
  sorry
}

end warehouse_capacity_l111_11102


namespace problem_l111_11152

noncomputable def p : Prop :=
  ∀ x : ℝ, (0 < x) → Real.exp x > 1 + x

def q (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + 2 = -(f x + 2)) → ∀ x : ℝ, f (-x) = f x - 4

theorem problem (f : ℝ → ℝ) : p ∨ q f :=
  sorry

end problem_l111_11152


namespace discount_price_equation_correct_l111_11192

def original_price := 200
def final_price := 148
variable (a : ℝ) -- assuming a is a real number representing the percentage discount

theorem discount_price_equation_correct :
  original_price * (1 - a / 100) ^ 2 = final_price :=
sorry

end discount_price_equation_correct_l111_11192


namespace number_leaves_remainder_3_l111_11162

theorem number_leaves_remainder_3 (n : ℕ) (h1 : 1680 % 9 = 0) (h2 : 1680 = n * 9) : 1680 % 1677 = 3 := by
  sorry

end number_leaves_remainder_3_l111_11162


namespace prime_number_condition_l111_11127

theorem prime_number_condition (n : ℕ) (h1 : n ≥ 2) :
  (∀ d : ℕ, d ∣ n → d > 1 → d^2 + n ∣ n^2 + d) → Prime n :=
sorry

end prime_number_condition_l111_11127


namespace find_y_l111_11104

theorem find_y (x y : ℤ) (h1 : x^2 - 2 * x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end find_y_l111_11104


namespace reduced_price_l111_11114

theorem reduced_price (P Q : ℝ) (h : P ≠ 0) (h₁ : 900 = Q * P) (h₂ : 900 = (Q + 6) * (0.90 * P)) : 0.90 * P = 15 :=
by 
  sorry

end reduced_price_l111_11114


namespace part1_part2_l111_11165

def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem part1 (m n : ℕ) (h : divides m n) : divides (fibonacci m) (fibonacci n) :=
sorry

theorem part2 (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) :=
sorry

end part1_part2_l111_11165


namespace isabella_hair_length_after_haircut_cm_l111_11123

theorem isabella_hair_length_after_haircut_cm :
  let initial_length_in : ℝ := 18  -- initial length in inches
  let growth_rate_in_per_week : ℝ := 0.5  -- growth rate in inches per week
  let weeks : ℝ := 4  -- time in weeks
  let hair_trimmed_in : ℝ := 2.25  -- length of hair trimmed in inches
  let cm_per_inch : ℝ := 2.54  -- conversion factor from inches to centimeters
  let final_length_in := initial_length_in + growth_rate_in_per_week * weeks - hair_trimmed_in  -- final length in inches
  let final_length_cm := final_length_in * cm_per_inch  -- final length in centimeters
  final_length_cm = 45.085 := by
  sorry

end isabella_hair_length_after_haircut_cm_l111_11123


namespace soda_quantity_difference_l111_11161

noncomputable def bottles_of_diet_soda := 19
noncomputable def bottles_of_regular_soda := 60
noncomputable def bottles_of_cherry_soda := 35
noncomputable def bottles_of_orange_soda := 45

theorem soda_quantity_difference : 
  (max bottles_of_regular_soda (max bottles_of_diet_soda 
    (max bottles_of_cherry_soda bottles_of_orange_soda)) 
  - min bottles_of_regular_soda (min bottles_of_diet_soda 
    (min bottles_of_cherry_soda bottles_of_orange_soda))) = 41 := 
by
  sorry

end soda_quantity_difference_l111_11161


namespace potions_needed_l111_11178

-- Definitions
def galleons_to_knuts (galleons : Int) : Int := galleons * 17 * 23
def sickles_to_knuts (sickles : Int) : Int := sickles * 23

-- Conditions from the problem
def cost_of_owl_in_knuts : Int := galleons_to_knuts 2 + sickles_to_knuts 1 + 5
def knuts_per_potion : Int := 9

-- Prove the number of potions needed is 90
theorem potions_needed : cost_of_owl_in_knuts / knuts_per_potion = 90 := by
  sorry

end potions_needed_l111_11178


namespace same_color_combination_sum_l111_11173

theorem same_color_combination_sum (m n : ℕ) (coprime_mn : Nat.gcd m n = 1)
  (prob_together : ∀ (total_candies : ℕ), total_candies = 20 →
    let terry_red := Nat.choose 8 2;
    let total_cases := Nat.choose total_candies 2;
    let prob_terry_red := terry_red / total_cases;
    
    let mary_red_given_terry := Nat.choose 6 2;
    let reduced_total_cases := Nat.choose 18 2;
    let prob_mary_red_given_terry := mary_red_given_terry / reduced_total_cases;
    
    let both_red := prob_terry_red * prob_mary_red_given_terry;
    
    let terry_blue := Nat.choose 12 2;
    let prob_terry_blue := terry_blue / total_cases;
    
    let mary_blue_given_terry := Nat.choose 10 2;
    let prob_mary_blue_given_terry := mary_blue_given_terry / reduced_total_cases;
    
    let both_blue := prob_terry_blue * prob_mary_blue_given_terry;
    
    let mixed_red_blue := Nat.choose 8 1 * Nat.choose 12 1;
    let prob_mixed_red_blue := mixed_red_blue / total_cases;
    let both_mixed := prob_mixed_red_blue;
    
    let prob_same_combination := both_red + both_blue + both_mixed;
    
    prob_same_combination = m / n
  ) :
  m + n = 5714 :=
by
  sorry

end same_color_combination_sum_l111_11173


namespace total_amount_shared_l111_11177

theorem total_amount_shared (A B C : ℕ) (h1 : 3 * B = 5 * A) (h2 : B = 25) (h3 : 5 * C = 8 * B) : A + B + C = 80 := by
  sorry

end total_amount_shared_l111_11177


namespace max_tickets_l111_11187

-- Define the conditions
def ticket_cost (n : ℕ) : ℝ :=
  if n ≤ 6 then 15 * n
  else 13.5 * n

-- Define the main theorem
theorem max_tickets (budget : ℝ) : (∀ n : ℕ, ticket_cost n ≤ budget) → budget = 120 → n ≤ 8 :=
  by
  sorry

end max_tickets_l111_11187


namespace distance_from_rachel_to_nicholas_l111_11170

def distance (speed time : ℝ) := speed * time

theorem distance_from_rachel_to_nicholas :
  distance 2 5 = 10 :=
by
  -- Proof goes here
  sorry

end distance_from_rachel_to_nicholas_l111_11170


namespace find_cosine_of_angle_subtraction_l111_11164

variable (α : ℝ)
variable (h : Real.sin ((Real.pi / 6) - α) = 1 / 3)

theorem find_cosine_of_angle_subtraction :
  Real.cos ((2 * Real.pi / 3) - α) = -1 / 3 :=
by
  exact sorry

end find_cosine_of_angle_subtraction_l111_11164


namespace calvin_gym_duration_l111_11179

theorem calvin_gym_duration (initial_weight loss_per_month final_weight : ℕ) (h1 : initial_weight = 250)
    (h2 : loss_per_month = 8) (h3 : final_weight = 154) : 
    (initial_weight - final_weight) / loss_per_month = 12 :=
by 
  sorry

end calvin_gym_duration_l111_11179


namespace archers_in_golden_l111_11174

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l111_11174


namespace correct_statement_l111_11155

theorem correct_statement (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) : (a ≥ 0 ∧ b ≥ 0) :=
by
  exact ⟨h_a, h_b⟩

end correct_statement_l111_11155


namespace probability_shadedRegion_l111_11107

noncomputable def triangleVertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (0, 5), (5, 0))

noncomputable def totalArea : ℝ :=
  12.5

noncomputable def shadedArea : ℝ :=
  4.5

theorem probability_shadedRegion (x y : ℝ) :
  let p := (x, y)
  let condition := x + y <= 3
  let totalArea := 12.5
  let shadedArea := 4.5
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 5}) →
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 + p.2 ≤ 3}) →
  (shadedArea / totalArea) = 9/25 :=
by
  sorry

end probability_shadedRegion_l111_11107


namespace round_robin_odd_game_count_l111_11132

theorem round_robin_odd_game_count (n : ℕ) (h17 : n = 17) :
  ∃ p : ℕ, p < n ∧ (p % 2 = 0) :=
by {
  sorry
}

end round_robin_odd_game_count_l111_11132


namespace sin2a_minus_cos2a_half_l111_11101

theorem sin2a_minus_cos2a_half (a : ℝ) (h : Real.tan (a - Real.pi / 4) = 1 / 2) :
  Real.sin (2 * a) - Real.cos a ^ 2 = 1 / 2 := 
sorry

end sin2a_minus_cos2a_half_l111_11101


namespace problem1_problem2_problem3_problem4_l111_11181

-- (1) Prove (1 + sqrt 3) * (2 - sqrt 3) = -1 + sqrt 3
theorem problem1 : (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 :=
by sorry

-- (2) Prove (sqrt 36 * sqrt 12) / sqrt 3 = 12
theorem problem2 : (Real.sqrt 36 * Real.sqrt 12) / Real.sqrt 3 = 12 :=
by sorry

-- (3) Prove sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4
theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = (5 * Real.sqrt 2) / 4 :=
by sorry

-- (4) Prove (3 * sqrt 18 + (1 / 5) * sqrt 50 - 4 * sqrt (1 / 2)) / sqrt 32 = 2
theorem problem4 : (3 * Real.sqrt 18 + (1 / 5) * Real.sqrt 50 - 4 * Real.sqrt (1 / 2)) / Real.sqrt 32 = 2 :=
by sorry

end problem1_problem2_problem3_problem4_l111_11181


namespace simplify_expression_l111_11158

theorem simplify_expression : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 :=
by sorry

end simplify_expression_l111_11158


namespace part1_solution_part2_solution_l111_11171

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) * x + abs (x - 2) * (x - a)

theorem part1_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a < 0} = {x : ℝ | x < 1} :=
by
  sorry

theorem part2_solution (x : ℝ) (hx : x < 1) :
  {a : ℝ | f x a < 0} = {a : ℝ | 1 ≤ a} :=
by
  sorry

end part1_solution_part2_solution_l111_11171


namespace arithmetic_expression_evaluation_l111_11112

theorem arithmetic_expression_evaluation :
  12 / 4 - 3 - 6 + 3 * 5 = 9 :=
by
  sorry

end arithmetic_expression_evaluation_l111_11112


namespace largest_A_l111_11103

def F (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem largest_A :
  ∃ n₁ n₂ n₃ n₄ n₅ n₆ : ℕ,
  (0 < n₁ ∧ 0 < n₂ ∧ 0 < n₃ ∧ 0 < n₄ ∧ 0 < n₅ ∧ 0 < n₆) ∧
  ∀ a, (1 ≤ a ∧ a ≤ 53590) -> 
    (F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) :=
sorry

end largest_A_l111_11103


namespace inequality_holds_for_all_x_iff_a_in_range_l111_11119

theorem inequality_holds_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ (-4 < a ∧ a < -1) :=
by
  sorry

end inequality_holds_for_all_x_iff_a_in_range_l111_11119


namespace investment_Y_l111_11188

theorem investment_Y
  (X_investment : ℝ)
  (Y_investment : ℝ)
  (Z_investment : ℝ)
  (X_months : ℝ)
  (Y_months : ℝ)
  (Z_months : ℝ)
  (total_profit : ℝ)
  (Z_profit_share : ℝ)
  (h1 : X_investment = 36000)
  (h2 : Z_investment = 48000)
  (h3 : X_months = 12)
  (h4 : Y_months = 12)
  (h5 : Z_months = 8)
  (h6 : total_profit = 13970)
  (h7 : Z_profit_share = 4064) :
  Y_investment = 75000 := by
  -- Proof omitted
  sorry

end investment_Y_l111_11188


namespace cube_angle_diagonals_l111_11109

theorem cube_angle_diagonals (q : ℝ) (h : q = 60) : 
  ∃ (d : String), d = "space diagonals" :=
by
  sorry

end cube_angle_diagonals_l111_11109


namespace sum_of_interior_diagonals_l111_11195

theorem sum_of_interior_diagonals (a b c : ℝ)
  (h₁ : 2 * (a * b + b * c + c * a) = 166)
  (h₂ : a + b + c = 16) :
  4 * Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) = 12 * Real.sqrt 10 :=
by
  sorry

end sum_of_interior_diagonals_l111_11195


namespace parabola_intersection_points_l111_11110

theorem parabola_intersection_points :
  (∃ (x y : ℝ), y = 4 * x ^ 2 + 3 * x - 7 ∧ y = 2 * x ^ 2 - 5)
  ↔ ((-2, 3) = (x, y) ∨ (1/2, -4.5) = (x, y)) :=
by
   -- To be proved (proof omitted)
   sorry

end parabola_intersection_points_l111_11110


namespace bottle_cap_count_l111_11131

theorem bottle_cap_count (price_per_cap total_cost : ℕ) (h_price : price_per_cap = 2) (h_total : total_cost = 12) : total_cost / price_per_cap = 6 :=
by
  sorry

end bottle_cap_count_l111_11131


namespace max_value_neg_a_inv_l111_11194

theorem max_value_neg_a_inv (a : ℝ) (h : a < 0) : a + (1 / a) ≤ -2 := 
by
  sorry

end max_value_neg_a_inv_l111_11194


namespace SufficientCondition_l111_11199

theorem SufficientCondition :
  ∀ x y z : ℤ, x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  intros x y z h
  cases h with
  | intro h1 h2 =>
  sorry

end SufficientCondition_l111_11199


namespace solve_equation_l111_11134

noncomputable def is_solution (x : ℝ) : Prop :=
  (x / (2 * Real.sqrt 2) + (5 * Real.sqrt 2) / 2) * Real.sqrt (x^3 - 64 * x + 200) = x^2 + 6 * x - 40

noncomputable def conditions (x : ℝ) : Prop :=
  (x^3 - 64 * x + 200) ≥ 0 ∧ x ≥ 4

theorem solve_equation :
  (∀ x, is_solution x → conditions x) = (x = 6 ∨ x = 1 + Real.sqrt 13) :=
by sorry

end solve_equation_l111_11134


namespace opposite_of_pi_is_neg_pi_l111_11146

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l111_11146


namespace no_integer_solutions_for_system_l111_11189

theorem no_integer_solutions_for_system :
  ∀ (y z : ℤ),
    (2 * y^2 - 2 * y * z - z^2 = 15) ∧ 
    (6 * y * z + 2 * z^2 = 60) ∧ 
    (y^2 + 8 * z^2 = 90) 
    → False :=
by 
  intro y z
  simp
  sorry

end no_integer_solutions_for_system_l111_11189


namespace new_lamp_height_is_correct_l111_11159

-- Define the height of the old lamp
def old_lamp_height : ℝ := 1

-- Define the additional height of the new lamp
def additional_height : ℝ := 1.33

-- Proof statement
theorem new_lamp_height_is_correct :
  old_lamp_height + additional_height = 2.33 :=
sorry

end new_lamp_height_is_correct_l111_11159


namespace find_surface_area_of_sphere_l111_11168

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 := a * b = 6
def condition2 := b * c = 2
def condition3 := a * c = 3
def vertices_on_sphere := true  -- Assuming vertices on tensor sphere condition for mathematical completion

theorem find_surface_area_of_sphere
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c)
  (h4 : vertices_on_sphere) :
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 14 * Real.pi :=
  sorry

end find_surface_area_of_sphere_l111_11168


namespace cost_of_white_car_l111_11180

variable (W : ℝ)
variable (red_cars white_cars : ℕ)
variable (rent_red rent_white : ℝ)
variable (rented_hours : ℝ)
variable (total_earnings : ℝ)

theorem cost_of_white_car 
  (h1 : red_cars = 3)
  (h2 : white_cars = 2) 
  (h3 : rent_red = 3)
  (h4 : rented_hours = 3)
  (h5 : total_earnings = 2340) :
  2 * W * (rented_hours * 60) + 3 * rent_red * (rented_hours * 60) = total_earnings → 
  W = 2 :=
by 
  sorry

end cost_of_white_car_l111_11180


namespace distance_between_intersections_is_sqrt3_l111_11193

noncomputable def intersection_distance : ℝ :=
  let C1_polar := (θ : ℝ) → θ = (2 * Real.pi / 3)
  let C2_standard := (x y : ℝ) → (x + Real.sqrt 3)^2 + (y + 2)^2 = 1
  let C3 := (θ : ℝ) → θ = (Real.pi / 3) 
  let C3_cartesian := (x y : ℝ) → y = Real.sqrt 3 * x
  let center := (-Real.sqrt 3, -2)
  let dist_to_C3 := abs (-3 + 2) / 2
  2 * Real.sqrt (1 - (dist_to_C3)^2)

theorem distance_between_intersections_is_sqrt3:
  intersection_distance = Real.sqrt 3 := by
  sorry

end distance_between_intersections_is_sqrt3_l111_11193


namespace range_of_x_function_l111_11136

open Real

theorem range_of_x_function : 
  ∀ x : ℝ, (x + 1 >= 0) ∧ (x - 3 ≠ 0) ↔ (x >= -1) ∧ (x ≠ 3) := 
by 
  sorry 

end range_of_x_function_l111_11136


namespace no_perfect_square_with_one_digit_appending_l111_11154

def append_digit (n : Nat) (d : Fin 10) : Nat :=
  n * 10 + d.val

theorem no_perfect_square_with_one_digit_appending :
  ∀ n : Nat, (∃ k : Nat, k * k = n) → 
  (¬ (∃ d1 : Fin 10, ∃ k : Nat, k * k = append_digit n d1.val) ∧
   ¬ (∃ d2 : Fin 10, ∃ d3 : Fin 10, ∃ k : Nat, k * k = d2.val * 10 ^ (Nat.digits 10 n).length + n * 10 + d3.val)) :=
by sorry

end no_perfect_square_with_one_digit_appending_l111_11154


namespace sqrt_multiplication_l111_11105

theorem sqrt_multiplication :
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 :=
by
  -- statement follows
  sorry

end sqrt_multiplication_l111_11105


namespace heidi_and_liam_paint_in_15_minutes_l111_11139

-- Definitions
def Heidi_rate : ℚ := 1 / 60
def Liam_rate : ℚ := 1 / 90
def combined_rate : ℚ := Heidi_rate + Liam_rate
def painting_time : ℚ := 15

-- Theorem to Prove
theorem heidi_and_liam_paint_in_15_minutes : painting_time * combined_rate = 5 / 12 := by
  sorry

end heidi_and_liam_paint_in_15_minutes_l111_11139


namespace solve_for_x_l111_11150

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h_eq : y = 1 / (3 * x^2 + 2 * x + 1)) : x = 0 ∨ x = -2 / 3 :=
by
  sorry

end solve_for_x_l111_11150


namespace symmetry_of_transformed_graphs_l111_11129

variable (f : ℝ → ℝ)

theorem symmetry_of_transformed_graphs :
  (∀ x, f x = f (-x)) → (∀ x, f (1 + x) = f (1 - x)) :=
by
  intro h_symmetry
  intro x
  sorry

end symmetry_of_transformed_graphs_l111_11129


namespace gcd_198_286_l111_11117

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l111_11117


namespace sixth_grade_percentage_combined_l111_11141

def maplewood_percentages := [10, 20, 15, 15, 10, 15, 15]
def brookside_percentages := [16, 14, 13, 12, 12, 18, 15]

def maplewood_students := 150
def brookside_students := 180

def sixth_grade_maplewood := maplewood_students * (maplewood_percentages.get! 6) / 100
def sixth_grade_brookside := brookside_students * (brookside_percentages.get! 6) / 100

def total_students := maplewood_students + brookside_students
def total_sixth_graders := sixth_grade_maplewood + sixth_grade_brookside

def sixth_grade_percentage := total_sixth_graders / total_students * 100

theorem sixth_grade_percentage_combined : sixth_grade_percentage = 15 := by 
  sorry

end sixth_grade_percentage_combined_l111_11141


namespace range_of_a_l111_11198

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (x + 1 - a) >= 0 → x ≠ 1) ↔ (1 < a ∧ a < 2) := 
sorry

end range_of_a_l111_11198


namespace mean_home_runs_l111_11118

theorem mean_home_runs :
  let players6 := 5
  let players8 := 6
  let players10 := 4
  let home_runs6 := players6 * 6
  let home_runs8 := players8 * 8
  let home_runs10 := players10 * 10
  let total_home_runs := home_runs6 + home_runs8 + home_runs10
  let total_players := players6 + players8 + players10
  total_home_runs / total_players = 118 / 15 :=
by
  sorry

end mean_home_runs_l111_11118


namespace derivative_at_0_l111_11176

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_at_0 : deriv f 0 = 1 := by
  -- Proof goes here
  sorry

end derivative_at_0_l111_11176


namespace arctan_sum_l111_11111

theorem arctan_sum (θ₁ θ₂ : ℝ) (h₁ : θ₁ = Real.arctan (1/2))
                              (h₂ : θ₂ = Real.arctan 2) :
  θ₁ + θ₂ = Real.pi / 2 :=
by
  have : θ₁ + θ₂ + Real.pi / 2 = Real.pi := sorry
  linarith

end arctan_sum_l111_11111


namespace combined_age_l111_11137

-- Define the conditions given in the problem
def Hezekiah_age : Nat := 4
def Ryanne_age := Hezekiah_age + 7

-- The statement to prove
theorem combined_age : Ryanne_age + Hezekiah_age = 15 :=
by
  -- we would provide the proof here, but for now we'll skip it with 'sorry'
  sorry

end combined_age_l111_11137


namespace age_of_teacher_l111_11190

/-- Given that the average age of 23 students is 22 years, and the average age increases
by 1 year when the teacher's age is included, prove that the teacher's age is 46 years. -/
theorem age_of_teacher (n : ℕ) (s_avg : ℕ) (new_avg : ℕ) (teacher_age : ℕ) :
  n = 23 →
  s_avg = 22 →
  new_avg = s_avg + 1 →
  teacher_age = new_avg * (n + 1) - s_avg * n →
  teacher_age = 46 :=
by
  intros h_n h_s_avg h_new_avg h_teacher_age
  sorry

end age_of_teacher_l111_11190


namespace sum_of_integers_ways_l111_11156

theorem sum_of_integers_ways (n : ℕ) (h : n > 0) : 
  ∃ ways : ℕ, ways = 2^(n-1) := sorry

end sum_of_integers_ways_l111_11156
