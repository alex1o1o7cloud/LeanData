import Mathlib

namespace tan_angle_sum_l151_151836

variable (α β : ℝ)

theorem tan_angle_sum (h1 : Real.tan (α - Real.pi / 6) = 3 / 7)
                      (h2 : Real.tan (Real.pi / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
by
  sorry

end tan_angle_sum_l151_151836


namespace number_of_students_playing_sports_students_not_playing_any_sports_l151_151191

def total_students : ℕ := 40
def students_basketball : ℕ := 15
def students_cricket : ℕ := 20
def students_baseball : ℕ := 12
def students_basketball_and_cricket : ℕ := 5
def students_cricket_and_baseball : ℕ := 7
def students_basketball_and_baseball : ℕ := 3
def students_all_three : ℕ := 2

theorem number_of_students_playing_sports (total students_b students_c students_ba students_bc students_cba students_bba students_bcb : ℕ) :
  students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb = 32 :=
begin
  have total := students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb,
  have heq : total = 32,
  sorry
end

theorem students_not_playing_any_sports (total students_b students_c students_ba students_bc students_cba students_bba students_bcb : ℕ) :
  total - (students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb) = 8 :=
begin
  have students_playing := students_b + students_c + students_ba - students_bc - students_cba - students_bba + students_bcb,
  have heq : total - students_playing = 8,
  sorry
end

end number_of_students_playing_sports_students_not_playing_any_sports_l151_151191


namespace second_term_of_geometric_series_l151_151382

theorem second_term_of_geometric_series (a r S: ℝ) (h_r : r = 1/4) (h_S : S = 40) (h_geom_sum : S = a / (1 - r)) : a * r = 7.5 :=
by
  sorry

end second_term_of_geometric_series_l151_151382


namespace probability_playing_one_instrument_l151_151580

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_instruments : ℚ := 1 / 5
noncomputable def number_playing_two_or_more : ℕ := 32

theorem probability_playing_one_instrument :
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  (number_playing_exactly_one / total_people) = 1 / 6.25 :=
by 
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  have key : (number_playing_exactly_one / total_people) = 1 / 6.25 := sorry
  exact key

end probability_playing_one_instrument_l151_151580


namespace complement_union_l151_151847

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 4}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {2, 4}) (hB : B = {1, 4}) :
  (U \ (A ∪ B)) = {3} :=
by
  simp [hU, hA, hB]
  sorry

end complement_union_l151_151847


namespace square_diff_l151_151438

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l151_151438


namespace grasshoppers_total_l151_151199

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l151_151199


namespace find_n_modulo_conditions_l151_151827

theorem find_n_modulo_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n % 7 = -3137 % 7 ∧ (n = 1 ∨ n = 8) := sorry

end find_n_modulo_conditions_l151_151827


namespace slope_of_bisecting_line_l151_151387

theorem slope_of_bisecting_line (m n : ℕ) (hmn : Int.gcd m n = 1) : 
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  -- Define conditions for line through origin (x = 0, y = 0) bisecting the parallelogram
  let b := 135 / 19
  let slope := (90 + b) / 20
  -- The slope must be equal to 369/76 (m = 369, n = 76)
  m = 369 ∧ n = 76 → m + n = 445 := by
  intro m n hmn
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  let b := 135 / 19
  let slope := (90 + b) / 20
  sorry

end slope_of_bisecting_line_l151_151387


namespace necessary_but_not_sufficient_l151_151851

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a - b > 0 → a^2 - b^2 > 0) ∧ ¬(a^2 - b^2 > 0 → a - b > 0) := by
sorry

end necessary_but_not_sufficient_l151_151851


namespace gwen_spent_zero_l151_151685

theorem gwen_spent_zero 
  (m : ℕ) 
  (d : ℕ) 
  (S : ℕ) 
  (h1 : m = 8) 
  (h2 : d = 5)
  (h3 : (m - S) = (d - S) + 3) : 
  S = 0 :=
by
  sorry

end gwen_spent_zero_l151_151685


namespace sum_xyz_eq_11sqrt5_l151_151322

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l151_151322


namespace average_boxes_per_day_by_third_day_l151_151491

theorem average_boxes_per_day_by_third_day (day1 day2 day3_part1 day3_part2 : ℕ) :
  day1 = 318 →
  day2 = 312 →
  day3_part1 = 180 →
  day3_part2 = 162 →
  ((day1 + day2 + (day3_part1 + day3_part2)) / 3) = 324 :=
by
  intros h1 h2 h3 h4
  sorry

end average_boxes_per_day_by_third_day_l151_151491


namespace calculate_sum_l151_151276

open Real

theorem calculate_sum :
  (-1: ℝ) ^ 2023 + (1/2) ^ (-2: ℝ) + 3 * tan (pi / 6) - (3 - pi) ^ 0 + |sqrt 3 - 2| = 4 :=
by
  sorry

end calculate_sum_l151_151276


namespace ab_leq_one_fraction_inequality_l151_151162

-- Part 1: Prove that ab ≤ 1
theorem ab_leq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) : a * b ≤ 1 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

-- Part 2: Prove that (1/a^3 - 1/b^3) > 3 * (1/a - 1/b) given b > a
theorem fraction_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) (h4 : b > a) :
  1/(a^3) - 1/(b^3) > 3 * (1/a - 1/b) :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end ab_leq_one_fraction_inequality_l151_151162


namespace zeros_of_derivative_interior_l151_151965

noncomputable def polynomial
def is_jordan_curve (L : set ℂ) : Prop :=
  is_compact L ∧ is_connected L ∧ ∀ z ∈ L, ∃ U, is_open U ∧ z ∈ U ∧ ∀ y ∈ U ∩ L, y ≠ z

theorem zeros_of_derivative_interior (P : polynomial ℂ) 
  (L : set ℂ) (hL : L = {z : ℂ | ∥P.eval z∥ = 1} ∧ is_jordan_curve L) :
  ∀ z₀ ∈ L, P.derivative.eval z₀ = 0 → ∃ ε > 0, ∀ y, dist z₀ y < ε → (∥P.eval y∥ < 1) :=
sorry

end zeros_of_derivative_interior_l151_151965


namespace smallest_integer_solution_l151_151238

theorem smallest_integer_solution (x : ℝ) :
  x^4 - 40 * x^2 + 324 = 0 → x = -4 :=
begin
  sorry
end

end smallest_integer_solution_l151_151238


namespace ones_digit_of_73_pow_351_l151_151246

-- Definition of the problem in Lean 4
theorem ones_digit_of_73_pow_351 : (73 ^ 351) % 10 = 7 := by
  sorry

end ones_digit_of_73_pow_351_l151_151246


namespace sum_of_interiors_l151_151217

theorem sum_of_interiors (n : ℕ) (h : 180 * (n - 2) = 1620) : 180 * ((n + 3) - 2) = 2160 :=
by sorry

end sum_of_interiors_l151_151217


namespace eq1_solution_eq2_no_solution_l151_151060

theorem eq1_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2) :
  (2/x + 1/(x*(x-2)) = 5/(2*x)) ↔ x = 4 :=
by sorry

theorem eq2_no_solution (x : ℝ) (h : x ≠ 2) :
  (5*x - 4)/ (x - 2) = (4*x + 10) / (3*x - 6) - 1 ↔ false :=
by sorry

end eq1_solution_eq2_no_solution_l151_151060


namespace farm_total_amount_90000_l151_151337

-- Defining the conditions
def apples_produce (mangoes: ℕ) : ℕ := 2 * mangoes
def oranges_produce (mangoes: ℕ) : ℕ := mangoes + 200

-- Defining the total produce of all fruits
def total_produce (mangoes: ℕ) : ℕ := apples_produce mangoes + mangoes + oranges_produce mangoes

-- Defining the price per kg
def price_per_kg : ℕ := 50

-- Defining the total amount from selling all fruits
noncomputable def total_amount (mangoes: ℕ) : ℕ := total_produce mangoes * price_per_kg

-- Proving that the total amount he got in that season is $90,000
theorem farm_total_amount_90000 : total_amount 400 = 90000 := by
  sorry

end farm_total_amount_90000_l151_151337


namespace distinct_rational_numbers_l151_151678

theorem distinct_rational_numbers (m : ℚ) :
  abs m < 100 ∧ (∃ x : ℤ, 4 * x^2 + m * x + 15 = 0) → 
  ∃ n : ℕ, n = 48 :=
sorry

end distinct_rational_numbers_l151_151678


namespace joan_gemstones_l151_151033

def number_of_gemstones (M : ℕ) (M_y : ℕ) (G_y : ℕ) : ℕ :=
  G_y

theorem joan_gemstones
  (M : ℕ)
  (h1 : M = 48)
  (M_y : ℕ)
  (h2 : M_y = M - 6)
  (G_y : ℕ)
  (h3 : G_y = M_y / 2) : 
  number_of_gemstones M M_y G_y = 21 := 
by 
  rw [number_of_gemstones, h1, h2, h3]
  rfl


end joan_gemstones_l151_151033


namespace original_investment_amount_l151_151656

-- Definitions
def annual_interest_rate : ℝ := 0.04
def investment_period_years : ℝ := 0.25
def final_amount : ℝ := 10204

-- Statement to prove
theorem original_investment_amount :
  let P := final_amount / (1 + annual_interest_rate * investment_period_years)
  P = 10104 :=
by
  -- Placeholder for the proof
  sorry

end original_investment_amount_l151_151656


namespace four_digit_numbers_using_0_and_9_l151_151868

theorem four_digit_numbers_using_0_and_9 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d, d ∈ Nat.digits 10 n → (d = 0 ∨ d = 9)} = {9000, 9009, 9090, 9099, 9900, 9909, 9990, 9999} :=
by
  sorry

end four_digit_numbers_using_0_and_9_l151_151868


namespace smallest_number_of_integers_l151_151651

theorem smallest_number_of_integers (a b n : ℕ) 
  (h_avg_original : 89 * n = 73 * a + 111 * b) 
  (h_group_sum : a + b = n)
  (h_ratio : 8 * a = 11 * b) : 
  n = 19 :=
sorry

end smallest_number_of_integers_l151_151651


namespace units_digit_of_modifiedLucas_379_l151_151815

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
| 0     := 3
| 1     := 1
| (n+2) := 2 * modifiedLucas (n + 1) + modifiedLucas n

-- Define the function to get the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- State the main theorem
theorem units_digit_of_modifiedLucas_379 : units_digit (modifiedLucas 379) = 3 :=
sorry

end units_digit_of_modifiedLucas_379_l151_151815


namespace negation_correct_l151_151076

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct_l151_151076


namespace sin_alpha_correct_complicated_expression_correct_l151_151842

open Real

variables {α : ℝ}
noncomputable def sin_alpha : ℝ := 4/5
noncomputable def cos_alpha : ℝ := -3/5

theorem sin_alpha_correct :
  α.vertex_at_origin ∧ α.initial_side_non_negative_x_axis ∧ α.terminal_side_intersects_unit_circle (-3/5) (4/5) →
  sin α = 4/5 :=
sorry

theorem complicated_expression_correct :
  α.vertex_at_origin ∧ α.initial_side_non_negative_x_axis ∧ α.terminal_side_intersects_unit_circle (-3/5) (4/5) →
  (sin (2*α) + cos (2*α) + 1) / (1 + tan α) = 6/5 :=
sorry

end sin_alpha_correct_complicated_expression_correct_l151_151842


namespace calculate_total_calories_l151_151115

-- Definition of variables and conditions
def total_calories (C : ℝ) : Prop :=
  let FDA_recommended_intake := 25
  let consumed_calories := FDA_recommended_intake + 5
  (3 / 4) * C = consumed_calories

-- Theorem statement
theorem calculate_total_calories : ∃ C : ℝ, total_calories C ∧ C = 40 :=
by
  sorry  -- Proof will be provided here

end calculate_total_calories_l151_151115


namespace min_value_of_fraction_l151_151335

theorem min_value_of_fraction 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Real.sqrt 3) = Real.sqrt (3 ^ a * 3 ^ (2 * b))) : 
  ∃ (min : ℝ), min = (2 / a + 1 / b) ∧ min = 8 :=
by
  -- proof will be skipped using sorry
  sorry

end min_value_of_fraction_l151_151335


namespace find_gain_percent_l151_151778

-- Definitions based on the conditions
def CP : ℕ := 900
def SP : ℕ := 1170

-- Calculation of gain
def Gain := SP - CP

-- Calculation of gain percent
def GainPercent := (Gain * 100) / CP

-- The theorem to prove the gain percent is 30%
theorem find_gain_percent : GainPercent = 30 := 
by
  sorry -- Proof to be filled in.

end find_gain_percent_l151_151778


namespace find_sphere_volume_l151_151326

noncomputable def sphere_volume (d: ℝ) (V: ℝ) : Prop := d = 3 * (16 / 9) * V

theorem find_sphere_volume :
  sphere_volume (2 / 3) (1 / 6) :=
by
  sorry

end find_sphere_volume_l151_151326


namespace discount_problem_l151_151646

theorem discount_problem (n : ℕ) : 
  (∀ x : ℝ, 0 < x → (1 - n / 100 : ℝ) * x < min (0.72 * x) (min (0.6724 * x) (0.681472 * x))) ↔ n ≥ 33 :=
by
  sorry

end discount_problem_l151_151646


namespace count_reflectional_symmetry_l151_151711

def tetrominoes : List String := ["I", "O", "T", "S", "Z", "L", "J"]

def has_reflectional_symmetry (tetromino : String) : Bool :=
  match tetromino with
  | "I" => true
  | "O" => true
  | "T" => true
  | "S" => false
  | "Z" => false
  | "L" => false
  | "J" => false
  | _   => false

theorem count_reflectional_symmetry : 
  (tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end count_reflectional_symmetry_l151_151711


namespace propositions_alpha_and_beta_true_l151_151208

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

def strictly_increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def strictly_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x > f y

def alpha (f : ℝ → ℝ) : Prop :=
∀ x, ∃ g h : ℝ → ℝ, even_function g ∧ odd_function h ∧ f x = g x + h x

def beta (f : ℝ → ℝ) : Prop :=
∀ x, strictly_increasing_function f → ∃ p q : ℝ → ℝ, 
  strictly_increasing_function p ∧ strictly_decreasing_function q ∧ f x = p x + q x

theorem propositions_alpha_and_beta_true (f : ℝ → ℝ) :
  alpha f ∧ beta f :=
by
  sorry

end propositions_alpha_and_beta_true_l151_151208


namespace floor_sqrt_225_l151_151289

theorem floor_sqrt_225 : Int.floor (Real.sqrt 225) = 15 := by
  sorry

end floor_sqrt_225_l151_151289


namespace train_crosses_bridge_in_30_seconds_l151_151120

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l151_151120


namespace function_is_zero_l151_151332

-- Define the condition that for any three points A, B, and C forming an equilateral triangle,
-- the sum of their function values is zero.
def has_equilateral_property (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ), dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 → 
  f A + f B + f C = 0

-- Define the theorem that states that a function with the equilateral property is identically zero.
theorem function_is_zero {f : ℝ × ℝ → ℝ} (h : has_equilateral_property f) : 
  ∀ (x : ℝ × ℝ), f x = 0 := 
by
  sorry

end function_is_zero_l151_151332


namespace solve_g_eq_5_l151_151730

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_g_eq_5 : {x : ℝ | g x = 5} = {-3/4, 20/3} :=
by
  sorry

end solve_g_eq_5_l151_151730


namespace find_pairs_l151_151150

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l151_151150


namespace equation_has_two_distinct_real_roots_l151_151968

open Real

theorem equation_has_two_distinct_real_roots (m : ℝ) :
  (∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 16 ∧ 0 < x2 ∧ x2 < 16 ∧ x1 ≠ x2 ∧ exp (m * x1) = x1^2 ∧ exp (m * x2) = x2^2) ↔
  (log 2 / 2 < m ∧ m < 2 / exp 1) :=
by sorry

end equation_has_two_distinct_real_roots_l151_151968


namespace fraction_product_eq_l151_151819
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end fraction_product_eq_l151_151819


namespace time_for_A_to_complete_race_l151_151860

theorem time_for_A_to_complete_race
  (V_A V_B : ℝ) (T_A : ℝ)
  (h1 : V_B = 975 / T_A) (h2 : V_B = 2.5) :
  T_A = 390 :=
by
  sorry

end time_for_A_to_complete_race_l151_151860


namespace colombian_coffee_amount_l151_151457

theorem colombian_coffee_amount 
  (C B : ℝ) 
  (h1 : C + B = 100)
  (h2 : 8.75 * C + 3.75 * B = 635) :
  C = 52 := 
sorry

end colombian_coffee_amount_l151_151457


namespace andrew_donates_160_to_homeless_shelter_l151_151928

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l151_151928


namespace distribute_balls_into_boxes_l151_151012

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l151_151012


namespace current_rate_l151_151634

variable (c : ℝ)

def still_water_speed : ℝ := 3.6

axiom rowing_time_ratio (c : ℝ) : (2 : ℝ) * (still_water_speed - c) = still_water_speed + c

theorem current_rate : c = 1.2 :=
by
  sorry

end current_rate_l151_151634


namespace grasshoppers_total_l151_151200

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end grasshoppers_total_l151_151200


namespace stacy_paper_shortage_l151_151607

theorem stacy_paper_shortage:
  let bought_sheets : ℕ := 240 + 320
  let daily_mwf : ℕ := 60
  let daily_tt : ℕ := 100
  -- Calculate sheets used in a week
  let used_one_week : ℕ := (daily_mwf * 3) + (daily_tt * 2)
  -- Calculate sheets used in two weeks
  let used_two_weeks : ℕ := used_one_week * 2
  -- Remaining sheets at the end of two weeks
  let remaining_sheets : Int := bought_sheets - used_two_weeks
  remaining_sheets = -200 :=
by sorry

end stacy_paper_shortage_l151_151607


namespace range_of_m_l151_151487

theorem range_of_m (m : ℝ) : 
  (m - 1 < 0 ∧ 4 * m - 3 > 0) → (3 / 4 < m ∧ m < 1) := 
by
  sorry

end range_of_m_l151_151487


namespace sin_squared_identity_l151_151275

theorem sin_squared_identity :
  1 - 2 * (Real.sin (105 * Real.pi / 180))^2 = - (Real.sqrt 3) / 2 :=
by sorry

end sin_squared_identity_l151_151275


namespace total_employees_l151_151858

variable (E : ℕ)
variable (employees_prefer_X employees_prefer_Y number_of_prefers : ℕ)
variable (X_percentage Y_percentage : ℝ)

-- Conditions based on the problem
axiom prefer_X : X_percentage = 0.60
axiom prefer_Y : Y_percentage = 0.40
axiom max_preference_relocation : number_of_prefers = 140

-- Defining the total number of employees who prefer city X or Y and get relocated accordingly:
axiom equation : X_percentage * E + Y_percentage * E = number_of_prefers

-- The theorem we are proving
theorem total_employees : E = 140 :=
by
  -- Proof placeholder
  sorry

end total_employees_l151_151858


namespace convert_fraction_to_decimal_l151_151135

theorem convert_fraction_to_decimal : (3 / 40 : ℝ) = 0.075 := 
by
  sorry

end convert_fraction_to_decimal_l151_151135


namespace find_pos_real_nums_l151_151944

theorem find_pos_real_nums (x y z a b c : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z):
  (x + y + z = a + b + c) ∧ (4 * x * y * z = a^2 * x + b^2 * y + c^2 * z + a * b * c) →
  (a = y + z - x ∧ b = z + x - y ∧ c = x + y - z) :=
by
  sorry

end find_pos_real_nums_l151_151944


namespace problem_solution_inf_problem_solution_prime_l151_151341

-- Definitions based on the given conditions and problem statement
def is_solution_inf (m : ℕ) : Prop := 3^m ∣ 2^(3^m) + 1

def is_solution_prime (n : ℕ) : Prop := n.Prime ∧ n ∣ 2^n + 1

-- Lean statement for the math proof problem
theorem problem_solution_inf : ∀ m : ℕ, m ≥ 0 → is_solution_inf m := sorry

theorem problem_solution_prime : ∀ n : ℕ, n.Prime → is_solution_prime n → n = 3 := sorry

end problem_solution_inf_problem_solution_prime_l151_151341


namespace functionG_has_inverse_l151_151709

noncomputable def functionG : ℝ → ℝ := -- function G described in the problem.
sorry

-- Define the horizontal line test
def horizontal_line_test (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃! x : ℝ, f x = y

theorem functionG_has_inverse : horizontal_line_test functionG :=
sorry

end functionG_has_inverse_l151_151709


namespace carrots_thrown_out_l151_151672

variable (x : ℕ)

theorem carrots_thrown_out :
  let initial_carrots := 23
  let picked_later := 47
  let total_carrots := 60
  initial_carrots - x + picked_later = total_carrots → x = 10 :=
by
  intros
  sorry

end carrots_thrown_out_l151_151672


namespace correct_answers_proof_l151_151581

variable (n p q s c : ℕ)
variable (total_questions points_per_correct penalty_per_wrong total_score correct_answers : ℕ)

def num_questions := 20
def points_correct := 5
def penalty_wrong := 1
def total_points := 76

theorem correct_answers_proof :
  (total_questions * points_per_correct - (total_questions - correct_answers) * penalty_wrong) = total_points →
  correct_answers = 16 :=
by {
  sorry
}

end correct_answers_proof_l151_151581


namespace initial_amount_of_money_l151_151088

-- Define the costs and purchased quantities
def cost_tshirt : ℕ := 8
def cost_keychain_set : ℕ := 2
def cost_bag : ℕ := 10
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

-- Define derived quantities
def sets_of_keychains_bought : ℕ := keychains_bought / 3

-- Define the total costs
def total_cost_tshirts : ℕ := tshirts_bought * cost_tshirt
def total_cost_bags : ℕ := bags_bought * cost_bag
def total_cost_keychains : ℕ := sets_of_keychains_bought * cost_keychain_set

-- Define the initial amount of money
def total_initial_amount : ℕ := total_cost_tshirts + total_cost_bags + total_cost_keychains

-- The theorem proving the initial amount Timothy had
theorem initial_amount_of_money : total_initial_amount = 50 := by
  -- The proof is not required, so we use sorry to skip it
  sorry

end initial_amount_of_money_l151_151088


namespace simplify_expression_l151_151605

theorem simplify_expression (tan_60 cot_60 : ℝ) (h1 : tan_60 = Real.sqrt 3) (h2 : cot_60 = 1 / Real.sqrt 3) :
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 31 / 3 :=
by
  -- proof will go here
  sorry

end simplify_expression_l151_151605


namespace min_value_x_plus_y_l151_151405

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + y ≥ 4 :=
by
  sorry

end min_value_x_plus_y_l151_151405


namespace lowest_degree_is_4_l151_151506

noncomputable def lowest_degree_polynomial (P : ℝ → ℝ) : ℕ :=
  if ∃ b : ℤ, (∀ coeff ∈ (P.coefficients), coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ (P.coefficients), coeff = (b : ℝ))
  then Polynomial.natDegree P
  else 0

theorem lowest_degree_is_4 : ∀ (P : Polynomial ℝ), 
  (∃ b : ℤ, (∀ coeff ∈ P.coefficients, coeff < (b : ℝ) ∨ coeff > (b : ℝ)) ∧ (¬ ∃ coeff ∈ P.coefficients, coeff = (b : ℝ)))
  → lowest_degree_polynomial P = 4 :=
by
  sorry

end lowest_degree_is_4_l151_151506


namespace find_two_fractions_sum_eq_86_over_111_l151_151947

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l151_151947


namespace min_sum_squares_l151_151251

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l151_151251


namespace second_integer_value_l151_151224

theorem second_integer_value (n : ℚ) (h : (n - 1) + (n + 1) + (n + 2) = 175) : n = 57 + 2 / 3 :=
by
  sorry

end second_integer_value_l151_151224


namespace base_eight_to_base_ten_l151_151233

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l151_151233


namespace solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l151_151401

-- Define the function f(x) based on the given conditions
def f (x k : ℝ) : ℝ := abs (x ^ 2 - 1) + x ^ 2 + k * x

-- Statement 1
theorem solve_f_zero_k_eq_2 :
  (∀ x : ℝ, f x 2 = 0 ↔ x = - (1 + Real.sqrt 3) / 2 ∨ x = -1 / 2) :=
sorry

-- Statement 2
theorem find_k_range_has_two_zeros (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < 2) :
  (∃ k : ℝ, f α k = 0 ∧ f β k = 0) ↔ - 7 / 2 < k ∧ k < -1 :=
sorry

-- Statement 3
theorem sum_of_reciprocals (α β : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
    (hα : f α (-1/α) = 0) (hβ : ∃ k : ℝ, f β k = 0) :
  (1 / α + 1 / β < 4) :=
sorry

end solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l151_151401


namespace problem_l151_151852

def f (x : ℝ) : ℝ := sorry -- We assume f is defined as per the given condition but do not provide an implementation.

theorem problem (h : ∀ x : ℝ, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry -- The proof is omitted

end problem_l151_151852


namespace intersection_points_l151_151003

def equation1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def equation2 (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) := sorry

end intersection_points_l151_151003


namespace constant_term_in_first_equation_l151_151687

/-- Given the system of equations:
  1. 5x + y = C
  2. x + 3y = 1
  3. 3x + 2y = 10
  Prove that the constant term C is 19.
-/
theorem constant_term_in_first_equation
  (x y C : ℝ)
  (h1 : 5 * x + y = C)
  (h2 : x + 3 * y = 1)
  (h3 : 3 * x + 2 * y = 10) :
  C = 19 :=
by
  sorry

end constant_term_in_first_equation_l151_151687


namespace binary_to_decimal_110011_l151_151137

theorem binary_to_decimal_110011 : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_110011_l151_151137


namespace minimum_k_exists_l151_151467

theorem minimum_k_exists (k : ℕ) (h : k > 0) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ k = 6 :=
sorry

end minimum_k_exists_l151_151467


namespace squared_difference_l151_151433

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l151_151433


namespace inscribed_circle_quadrilateral_l151_151790

theorem inscribed_circle_quadrilateral
  (AB CD BC AD AC BD E : ℝ)
  (r1 r2 r3 r4 : ℝ)
  (h1 : BC = AD)
  (h2 : AB + CD = BC + AD)
  (h3 : ∃ E, ∃ AC BD, AC * BD = E∧ AC > 0 ∧ BD > 0)
  (h_r1 : r1 > 0)
  (h_r2 : r2 > 0)
  (h_r3 : r3 > 0)
  (h_r4 : r4 > 0):
  1 / r1 + 1 / r3 = 1 / r2 + 1 / r4 := 
by
  sorry

end inscribed_circle_quadrilateral_l151_151790


namespace parabola_slope_l151_151973

theorem parabola_slope (p k : ℝ) (h1 : p > 0)
  (h_focus_distance : (p / 2) * (3^(1/2)) / (3 + 1^(1/2))^(1/2) = 3^(1/2))
  (h_AF_FB : exists A B : ℝ × ℝ, (A.1 = 2 - p / 2 ∧ 2 * (B.1 - 2) = 2)
    ∧ (A.2 = p - p / 2 ∧ A.2 = -2 * B.2)) :
  abs k = 2 * (2^(1/2)) :=
sorry

end parabola_slope_l151_151973


namespace digit_B_divisible_by_9_l151_151070

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l151_151070


namespace find_pairs_l151_151149

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end find_pairs_l151_151149


namespace conformal_2z_conformal_z_minus_2_squared_l151_151518

-- For the function w = 2z
theorem conformal_2z :
  ∀ z : ℂ, true :=
by
  intro z
  sorry

-- For the function w = (z-2)^2
theorem conformal_z_minus_2_squared :
  ∀ z : ℂ, z ≠ 2 → true :=
by
  intro z h
  sorry

end conformal_2z_conformal_z_minus_2_squared_l151_151518


namespace largest_possible_value_l151_151727

variable (a b : ℝ)

theorem largest_possible_value (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) :
  2 * a + b ≤ 5 :=
sorry

end largest_possible_value_l151_151727


namespace Eunji_has_most_marbles_l151_151336

-- Declare constants for each person's marbles
def Minyoung_marbles : ℕ := 4
def Yujeong_marbles : ℕ := 2
def Eunji_marbles : ℕ := Minyoung_marbles + 1

-- Theorem: Eunji has the most marbles
theorem Eunji_has_most_marbles :
  Eunji_marbles > Minyoung_marbles ∧ Eunji_marbles > Yujeong_marbles :=
by
  sorry

end Eunji_has_most_marbles_l151_151336


namespace ordered_pair_exists_l151_151557

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l151_151557


namespace incorrect_inequality_exists_l151_151104

theorem incorrect_inequality_exists :
  ∃ (x y : ℝ), x < y ∧ x^2 ≥ y^2 :=
by {
  sorry
}

end incorrect_inequality_exists_l151_151104


namespace group_D_forms_a_definite_set_l151_151125

theorem group_D_forms_a_definite_set : 
  ∃ (S : Set ℝ), S = { x : ℝ | x = 1 ∨ x = -1 } :=
by
  sorry

end group_D_forms_a_definite_set_l151_151125


namespace max_xy_l151_151714

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  xy ≤ 2 :=
by
  sorry

end max_xy_l151_151714


namespace three_digit_numbers_eq_11_sum_squares_l151_151774

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l151_151774


namespace second_class_students_l151_151216

-- Define the conditions
variables (x : ℕ)
variable (sum_marks_first_class : ℕ := 35 * 40)
variable (sum_marks_second_class : ℕ := x * 60)
variable (total_students : ℕ := 35 + x)
variable (total_marks_all_students : ℕ := total_students * 5125 / 100)

-- The theorem to prove
theorem second_class_students : 
  1400 + (x * 60) = (35 + x) * 5125 / 100 →
  x = 45 :=
by
  sorry

end second_class_students_l151_151216


namespace corresponding_angles_equal_l151_151587

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l151_151587


namespace part_I_part_II_l151_151693

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

end part_I_part_II_l151_151693


namespace inscribed_rectangle_area_l151_151657

theorem inscribed_rectangle_area (h a b x : ℝ) (ha_gt_b : a > b) :
  ∃ A : ℝ, A = (b * x / h) * (h - x) :=
by
  sorry

end inscribed_rectangle_area_l151_151657


namespace range_of_m_l151_151176

def p (m : ℝ) : Prop := ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0)
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1 ∨ ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0))
  ∧ (¬ (∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0) → ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1)) ↔
  (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2) :=
  sorry

end range_of_m_l151_151176


namespace percentage_of_360_is_120_l151_151240

theorem percentage_of_360_is_120 (part whole : ℝ) (h1 : part = 120) (h2 : whole = 360) : 
  ((part / whole) * 100 = 33.33) :=
by
  sorry

end percentage_of_360_is_120_l151_151240


namespace chord_ratio_l151_151090

theorem chord_ratio (A B C D P : Type) (AP BP CP DP : ℝ)
  (h1 : AP = 4) (h2 : CP = 9)
  (h3 : AP * BP = CP * DP) : BP / DP = 9 / 4 := 
by 
  sorry

end chord_ratio_l151_151090


namespace mark_charged_more_hours_l151_151108

variable {p k m : ℕ}

theorem mark_charged_more_hours (h1 : p + k + m = 216)
                                (h2 : p = 2 * k)
                                (h3 : p = m / 3) :
                                m - k = 120 :=
sorry

end mark_charged_more_hours_l151_151108


namespace connie_correct_answer_l151_151133

theorem connie_correct_answer 
  (x : ℝ) 
  (h1 : 2 * x = 80) 
  (correct_ans : ℝ := x / 3) :
  correct_ans = 40 / 3 :=
by
  sorry

end connie_correct_answer_l151_151133


namespace problem_1_problem_2_problem_3_problem_4_l151_151541

theorem problem_1 : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 :=
by sorry

theorem problem_2 : Real.sqrt (2 / 3) / Real.sqrt (8 / 27) = (3 / 2) :=
by sorry

theorem problem_3 : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = (10 * Real.sqrt 2 - 3 * Real.sqrt 3) :=
by sorry

theorem problem_4 : (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1 / 8) - Real.sqrt 24) = (Real.sqrt 2 / 4) + 3 * Real.sqrt 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l151_151541


namespace proof_base_5_conversion_and_addition_l151_151175

-- Define the given numbers in decimal (base 10)
def n₁ := 45
def n₂ := 25

-- Base 5 conversion function and proofs of correctness
def to_base_5 (n : ℕ) : ℕ := sorry
def from_base_5 (n : ℕ) : ℕ := sorry

-- Converted values to base 5
def a₅ : ℕ := to_base_5 n₁
def b₅ : ℕ := to_base_5 n₂

-- Sum in base 5
def c₅ : ℕ := a₅ + b₅  -- addition in base 5

-- Convert the final sum back to decimal base 10
def d₁₀ : ℕ := from_base_5 c₅

theorem proof_base_5_conversion_and_addition :
  d₁₀ = 65 ∧ to_base_5 65 = 230 :=
by sorry

end proof_base_5_conversion_and_addition_l151_151175


namespace perpendicular_tangents_sum_x1_x2_gt_4_l151_151837

noncomputable def f (x : ℝ) : ℝ := (1 / 6) * x^3 - (1 / 2) * x^2 + (1 / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def F (x : ℝ) : ℝ := (1 / 2) * x^2 - x - 2 * Real.log x

theorem perpendicular_tangents (a : ℝ) (b : ℝ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 1 / 3) (h₃ : c = 0) :
  let f' x := (1 / 2) * x^2 - x
  let g' x := 2 / x
  f' 1 * g' 1 = -1 :=
by sorry

theorem sum_x1_x2_gt_4 (x1 x2 : ℝ) (h₁ : 0 < x1 ∧ x1 < 4) (h₂ : 0 < x2 ∧ x2 < 4) (h₃ : x1 ≠ x2) (h₄ : F x1 = F x2) :
  x1 + x2 > 4 :=
by sorry

end perpendicular_tangents_sum_x1_x2_gt_4_l151_151837


namespace rectangular_area_length_width_l151_151378

open Nat

theorem rectangular_area_length_width (lengthInMeters widthInMeters : ℕ) (h1 : lengthInMeters = 500) (h2 : widthInMeters = 60) :
  (lengthInMeters * widthInMeters = 30000) ∧ ((lengthInMeters * widthInMeters) / 10000 = 3) :=
by
  sorry

end rectangular_area_length_width_l151_151378


namespace simple_interest_time_period_l151_151536

variable (SI P R T : ℝ)

theorem simple_interest_time_period (h₁ : SI = 4016.25) (h₂ : P = 8925) (h₃ : R = 9) :
  (P * R * T) / 100 = SI ↔ T = 5 := by
  sorry

end simple_interest_time_period_l151_151536


namespace find_f79_l151_151073

noncomputable def f : ℝ → ℝ :=
  sorry

axiom condition1 : ∀ x y : ℝ, f (x * y) = x * f y
axiom condition2 : f 1 = 25

theorem find_f79 : f 79 = 1975 :=
by
  sorry

end find_f79_l151_151073


namespace negation_of_proposition_l151_151768

theorem negation_of_proposition (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (∃ x y z : ℝ, (x < 0) ∧ (y < 0) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y)) →
  ¬(∀ x y z : ℝ, (x < 0 ∨ y < 0 ∨ z < 0) → (x ≠ y → x ≠ z → y ≠ z → (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (z = a ∨ z = b ∨ z = c))) :=
sorry

end negation_of_proposition_l151_151768


namespace remainder_2023_div_73_l151_151095

theorem remainder_2023_div_73 : 2023 % 73 = 52 := 
by
  -- Proof goes here
  sorry

end remainder_2023_div_73_l151_151095


namespace find_difference_of_squares_l151_151427

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l151_151427


namespace octahedron_vertices_sum_l151_151395

noncomputable def octahedron_faces_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem octahedron_vertices_sum (a b c d e f : ℕ) 
  (h : 8 * (octahedron_faces_sum a b c d e f) = 440) : 
  octahedron_faces_sum a b c d e f = 147 :=
by
  sorry

end octahedron_vertices_sum_l151_151395


namespace smallest_x_exists_l151_151940

theorem smallest_x_exists {M : ℤ} (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ x : ℕ, 2520 * x = M^3 ∧ x = 3675 := 
by {
  sorry
}

end smallest_x_exists_l151_151940


namespace find_n_l151_151785

theorem find_n : ∃ n : ℕ, n < 200 ∧ ∃ k : ℕ, n^2 + (n + 1)^2 = k^2 ∧ (n = 3 ∨ n = 20 ∨ n = 119) := 
by
  sorry

end find_n_l151_151785


namespace squared_difference_l151_151431

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l151_151431


namespace average_income_l151_151777

theorem average_income (income1 income2 income3 income4 income5 : ℝ)
    (h1 : income1 = 600) (h2 : income2 = 250) (h3 : income3 = 450) (h4 : income4 = 400) (h5 : income5 = 800) :
    (income1 + income2 + income3 + income4 + income5) / 5 = 500 := by
    sorry

end average_income_l151_151777


namespace three_digit_numbers_satisfying_condition_l151_151776

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l151_151776


namespace number_of_red_cars_l151_151760

theorem number_of_red_cars (B R : ℕ) (h1 : R / B = 3 / 8) (h2 : B = 70) : R = 26 :=
by
  sorry

end number_of_red_cars_l151_151760


namespace sufficient_but_not_necessary_condition_l151_151488

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 2*x < 0) → (|x - 2| < 2) ∧ ¬(|x - 2| < 2) → (x^2 - 2*x < 0 ↔ |x-2| < 2) :=
sorry

end sufficient_but_not_necessary_condition_l151_151488


namespace parabola_properties_l151_151066

theorem parabola_properties :
  ∀ x : ℝ, (x - 3)^2 + 5 = (x-3)^2 + 5 ∧ 
  (x - 3)^2 + 5 > 0 ∧ 
  (∃ h : ℝ, h = 3 ∧ ∀ x1 x2 : ℝ, (x1 - h)^2 <= (x2 - h)^2) ∧ 
  (∃ h k : ℝ, h = 3 ∧ k = 5) := 
by 
  sorry

end parabola_properties_l151_151066


namespace stations_visited_l151_151809

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited_l151_151809


namespace value_of_business_l151_151905

variable (V : ℝ)
variable (h1 : (2 / 3) * V = S)
variable (h2 : (3 / 4) * S = 75000)

theorem value_of_business (h1 : (2 / 3) * V = S) (h2 : (3 / 4) * S = 75000) : V = 150000 :=
sorry

end value_of_business_l151_151905


namespace arithmetic_mean_is_one_l151_151485

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 :=
by
  sorry

end arithmetic_mean_is_one_l151_151485


namespace measure_of_y_l151_151828

variables (A B C D : Point) (y : ℝ)
-- Given conditions
def angle_ABC := 120
def angle_BAD := 30
def angle_BDA := 21
def angle_ABD := 180 - angle_ABC

-- Theorem to prove
theorem measure_of_y :
  angle_BAD + angle_ABD + angle_BDA + y = 180 → y = 69 :=
by
  sorry

end measure_of_y_l151_151828


namespace part_a_part_b_l151_151370

-- Assuming existence of function S satisfying certain properties
variable (S : Type → Type → Type → ℝ)

-- Part (a)
theorem part_a (A B C : Type) : 
  S A B C = -S B A C ∧ S A B C = S B C A :=
sorry

-- Part (b)
theorem part_b (A B C D : Type) : 
  S A B C = S D A B + S D B C + S D C A :=
sorry

end part_a_part_b_l151_151370


namespace friends_meet_probability_l151_151624

noncomputable def probability_of_meeting :=
  let duration_total := 60 -- Total duration from 14:00 to 15:00 in minutes
  let duration_meeting := 30 -- Duration they can meet from 14:00 to 14:30 in minutes
  duration_meeting / duration_total

theorem friends_meet_probability : probability_of_meeting = 1 / 2 := by
  sorry

end friends_meet_probability_l151_151624


namespace jelly_bean_ratio_l151_151620

theorem jelly_bean_ratio (total_jelly_beans : ℕ) (coconut_flavored_jelly_beans : ℕ) (quarter_red_jelly_beans : ℕ) 
  (H_total : total_jelly_beans = 4000)
  (H_coconut : coconut_flavored_jelly_beans = 750)
  (H_quarter_red : quarter_red_jelly_beans = 1 / 4 * (4 * coconut_flavored_jelly_beans)) : 
  (quarter_red_jelly_beans * 4) / total_jelly_beans = 3 / 4 :=
by
  sorry

end jelly_bean_ratio_l151_151620


namespace no_solution_for_m_l151_151960

theorem no_solution_for_m (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (m : ℕ) (h3 : (ab)^2015 = (a^2 + b^2)^m) : false := 
sorry

end no_solution_for_m_l151_151960


namespace ratio_expression_l151_151017

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 1 ∧ B / C = 1 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 :=
by sorry

end ratio_expression_l151_151017


namespace statement_true_when_b_le_a_div_5_l151_151573

theorem statement_true_when_b_le_a_div_5
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₀ : ∀ x : ℝ, f x = 5 * x + 3)
  (h₁ : ∀ x : ℝ, |f x + 7| < a ↔ |x + 2| < b)
  (h₂ : 0 < a)
  (h₃ : 0 < b) :
  b ≤ a / 5 :=
by
  sorry

end statement_true_when_b_le_a_div_5_l151_151573


namespace chipmunks_initial_count_l151_151271

variable (C : ℕ) (total : ℕ) (morning_beavers : ℕ) (afternoon_beavers : ℕ) (decrease_chipmunks : ℕ)

axiom chipmunks_count : morning_beavers = 20 
axiom beavers_double : afternoon_beavers = 2 * morning_beavers
axiom decrease_chipmunks_initial : decrease_chipmunks = 10
axiom total_animals : total = 130

theorem chipmunks_initial_count : 
  20 + C + (2 * 20) + (C - 10) = 130 → C = 40 :=
by
  intros h
  sorry

end chipmunks_initial_count_l151_151271


namespace train_length_l151_151380

theorem train_length (L V : ℝ) (h1 : V = L / 15) (h2 : V = (L + 100) / 40) : L = 60 := by
  sorry

end train_length_l151_151380


namespace ferris_wheel_seats_l151_151642

theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) (h1 : total_people = 16) (h2 : people_per_seat = 4) : (total_people / people_per_seat) = 4 := by
  sorry

end ferris_wheel_seats_l151_151642


namespace number_of_C_animals_l151_151633

-- Define the conditions
def A : ℕ := 45
def B : ℕ := 32
def C : ℕ := 5

-- Define the theorem that we need to prove
theorem number_of_C_animals : B + C = A - 8 :=
by
  -- placeholder to complete the proof (not part of the problem's requirement)
  sorry

end number_of_C_animals_l151_151633


namespace shaded_region_area_is_15_l151_151862

noncomputable def area_of_shaded_region : ℝ :=
  let radius := 1
  let area_of_one_circle := Real.pi * (radius ^ 2)
  4 * area_of_one_circle + 3 * (4 - area_of_one_circle)

theorem shaded_region_area_is_15 : 
  abs (area_of_shaded_region - 15) < 1 :=
by
  exact sorry

end shaded_region_area_is_15_l151_151862


namespace work_together_days_l151_151372

theorem work_together_days (A_rate B_rate : ℝ) (x B_alone_days : ℝ)
  (hA : A_rate = 1 / 5)
  (hB : B_rate = 1 / 15)
  (h_total_work : (A_rate + B_rate) * x + B_rate * B_alone_days = 1) :
  x = 2 :=
by
  -- Set up the equation based on given rates and solving for x.
  sorry

end work_together_days_l151_151372


namespace expected_value_consecutive_red_draws_l151_151752

/-- The expected value of the total number of draws until two consecutive red balls are drawn,
given a bag of four differently colored balls drawn independently with replacement. --/
theorem expected_value_consecutive_red_draws :
  let ζ := -- define the total number of draws until two consecutive red draws here
  ζ = 2 -- because the variables E, ζ, and condition are not provided explicitly as definitions
 )
  -- some expectation mechanism and probability involved directly similar in the proof definition
  sorry
  :=
  sorry
where E is the expected value defined in steps cases, we need justification on letter precise defined.
  )
  20 :=
sorry

end expected_value_consecutive_red_draws_l151_151752


namespace initial_children_on_bus_l151_151228

-- Definitions based on conditions
variable (x : ℕ) -- number of children who got off the bus
variable (y : ℕ) -- initial number of children on the bus
variable (after_exchange : ℕ := 30) -- number of children on the bus after exchange
variable (got_on : ℕ := 82) -- number of children who got on the bus
variable (extra_on : ℕ := 2) -- extra children who got on compared to got off

-- Problem translated to Lean 4 statement
theorem initial_children_on_bus (h : got_on = x + extra_on) (hx : y + got_on - x = after_exchange) : y = 28 :=
by
  sorry

end initial_children_on_bus_l151_151228


namespace actual_time_between_two_and_three_l151_151367

theorem actual_time_between_two_and_three (x y : ℕ) 
  (h1 : 2 ≤ x ∧ x < 3)
  (h2 : 60 * y + x = 60 * x + y - 55) : 
  x = 2 ∧ y = 5 + 5 / 11 := 
sorry

end actual_time_between_two_and_three_l151_151367


namespace find_other_cat_weight_l151_151810

variable (cat1 cat2 dog : ℕ)

def weight_of_other_cat (cat1 cat2 dog : ℕ) : Prop :=
  cat1 = 7 ∧
  dog = 34 ∧
  dog = 2 * (cat1 + cat2) ∧
  cat2 = 10

theorem find_other_cat_weight (cat1 : ℕ) (cat2 : ℕ) (dog : ℕ) :
  weight_of_other_cat cat1 cat2 dog := by
  sorry

end find_other_cat_weight_l151_151810


namespace complement_union_eq_l151_151160

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_union_eq :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  A = {1, 3, 5, 7} →
  B = {2, 4, 5} →
  U \ (A ∪ B) = {6, 8} :=
by
  intros hU hA hB
  -- Proof goes here
  sorry

end complement_union_eq_l151_151160


namespace difference_of_cubes_l151_151021

theorem difference_of_cubes (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) : m^3 - n^3 = 1387 :=
by
  sorry

end difference_of_cubes_l151_151021


namespace andrew_total_homeless_shelter_donation_l151_151926

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l151_151926


namespace profit_percent_is_20_l151_151107

variable (C S : ℝ)

-- Definition from condition: The cost price of 60 articles is equal to the selling price of 50 articles
def condition : Prop := 60 * C = 50 * S

-- Definition of profit percent to be proven as 20%
def profit_percent_correct : Prop := ((S - C) / C) * 100 = 20

theorem profit_percent_is_20 (h : condition C S) : profit_percent_correct C S :=
sorry

end profit_percent_is_20_l151_151107


namespace probability_of_roll_6_after_E_l151_151287

/- Darryl has a six-sided die with faces 1, 2, 3, 4, 5, 6.
   The die is weighted so that one face comes up with probability 1/2,
   and the other five faces have equal probability.
   Darryl does not know which side is weighted, but each face is equally likely to be the weighted one.
   Darryl rolls the die 5 times and gets a 1, 2, 3, 4, and 5 in some unspecified order. -/

def probability_of_next_roll_getting_6 : ℚ :=
  let p_weighted := (1 / 2 : ℚ)
  let p_unweighted := (1 / 10 : ℚ)
  let p_w6_given_E := (1 / 26 : ℚ)
  let p_not_w6_given_E := (25 / 26 : ℚ)
  p_w6_given_E * p_weighted + p_not_w6_given_E * p_unweighted

theorem probability_of_roll_6_after_E : probability_of_next_roll_getting_6 = 3 / 26 := sorry

end probability_of_roll_6_after_E_l151_151287


namespace fill_tank_time_l151_151639

-- Define the rates at which the pipes fill the tank
noncomputable def rate_A := (1:ℝ)/50
noncomputable def rate_B := (1:ℝ)/75

-- Define the combined rate of both pipes
noncomputable def combined_rate := rate_A + rate_B

-- Define the time to fill the tank at the combined rate
noncomputable def time_to_fill := 1 / combined_rate

-- The theorem that states the time taken to fill the tank is 30 hours
theorem fill_tank_time : time_to_fill = 30 := sorry

end fill_tank_time_l151_151639


namespace greatest_integer_le_x_squared_div_50_l151_151050

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end greatest_integer_le_x_squared_div_50_l151_151050


namespace find_number_of_persons_l151_151127

-- Definitions of the given conditions
def total_amount : ℕ := 42900
def amount_per_person : ℕ := 1950

-- The statement to prove
theorem find_number_of_persons (n : ℕ) (h : total_amount = n * amount_per_person) : n = 22 :=
sorry

end find_number_of_persons_l151_151127


namespace train_speed_in_m_per_s_l151_151779

theorem train_speed_in_m_per_s (speed_kmph : ℕ) (h : speed_kmph = 162) :
  (speed_kmph * 1000) / 3600 = 45 :=
by {
  sorry
}

end train_speed_in_m_per_s_l151_151779


namespace cone_bead_path_l151_151530

theorem cone_bead_path (r h : ℝ) (h_sqrt : h / r = 3 * Real.sqrt 11) : 3 + 11 = 14 := by
  sorry

end cone_bead_path_l151_151530


namespace product_of_primes_impossible_l151_151202

theorem product_of_primes_impossible (q : ℕ) (hq1 : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬ ∀ i ∈ Finset.range (q-1), ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ (i^2 + i + q = p1 * p2) :=
sorry

end product_of_primes_impossible_l151_151202


namespace sum_of_remainders_mod_8_l151_151629

theorem sum_of_remainders_mod_8 
  (x y z w : ℕ)
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 :=
by
  sorry

end sum_of_remainders_mod_8_l151_151629


namespace exists_irreducible_fractions_sum_to_86_over_111_l151_151957

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l151_151957


namespace solving_linear_equations_count_l151_151343

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end solving_linear_equations_count_l151_151343


namespace steve_pencils_left_l151_151747

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l151_151747


namespace coefficient_of_x7_in_expansion_eq_15_l151_151611

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := n.choose k

theorem coefficient_of_x7_in_expansion_eq_15 (a : ℝ) (hbinom : binomial 10 3 * (-a) ^ 3 = 15) : a = -1 / 2 := by
  sorry

end coefficient_of_x7_in_expansion_eq_15_l151_151611


namespace length_of_adjacent_side_l151_151698

variable (a b : ℝ)

theorem length_of_adjacent_side (area : ℝ) (side : ℝ) :
  area = 6 * a^3 + 9 * a^2 - 3 * a * b →
  side = 3 * a →
  (area / side = 2 * a^2 + 3 * a - b) :=
by
  intro h_area
  intro h_side
  sorry

end length_of_adjacent_side_l151_151698


namespace ralph_fewer_pictures_l151_151477

-- Define the number of wild animal pictures Ralph and Derrick have.
def ralph_pictures : ℕ := 26
def derrick_pictures : ℕ := 34

-- The main theorem stating that Ralph has 8 fewer pictures than Derrick.
theorem ralph_fewer_pictures : derrick_pictures - ralph_pictures = 8 := by
  -- The proof is omitted, denoted by 'sorry'.
  sorry

end ralph_fewer_pictures_l151_151477


namespace find_abcd_from_N_l151_151353

theorem find_abcd_from_N (N : ℕ) (hN1 : N ≥ 10000) (hN2 : N < 100000)
  (hN3 : N % 100000 = (N ^ 2) % 100000) : (N / 10) / 10 / 10 / 10 = 2999 := by
  sorry

end find_abcd_from_N_l151_151353


namespace umar_age_is_ten_l151_151265

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l151_151265


namespace lily_spent_amount_l151_151346

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l151_151346


namespace man_speed_approx_l151_151787

noncomputable def speed_of_man : ℝ :=
  let L := 700    -- Length of the train in meters
  let u := 63 / 3.6  -- Speed of the train in meters per second (converted)
  let t := 41.9966402687785 -- Time taken to cross the man in seconds
  let v := (u * t - L) / t  -- Speed of the man
  v

-- The main theorem to prove that the speed of the man is approximately 0.834 m/s.
theorem man_speed_approx : abs (speed_of_man - 0.834) < 1e-3 :=
by
  -- Simplification and exact calculations will be handled by the Lean prover or could be manually done.
  sorry

end man_speed_approx_l151_151787


namespace squared_difference_l151_151432

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l151_151432


namespace find_x_values_l151_151945

noncomputable def tan_inv := Real.arctan (Real.sqrt 3 / 2)

theorem find_x_values (x : ℝ) :
  (-Real.pi < x ∧ x ≤ Real.pi) ∧ (2 * Real.tan x - Real.sqrt 3 = 0) ↔
  (x = tan_inv ∨ x = tan_inv - Real.pi) :=
by
  sorry

end find_x_values_l151_151945


namespace f_alpha_l151_151694

variables (α : Real) (x : Real)

noncomputable def f (x : Real) : Real := 
  (Real.cos (Real.pi + x) * Real.sin (2 * Real.pi - x)) / Real.cos (Real.pi - x)

lemma sin_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) : 
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

lemma tan_alpha {α : Real} (hsin : Real.sin α = 2 * Real.sqrt 2 / 3) (hcos : Real.cos α = 1 / 3) :
  Real.tan α = 2 * Real.sqrt 2 :=
sorry

theorem f_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) :
  f α = -2 * Real.sqrt 2 / 3 :=
sorry

end f_alpha_l151_151694


namespace seafood_noodles_l151_151114

theorem seafood_noodles (total_plates lobster_rolls spicy_hot_noodles : ℕ)
  (h_total : total_plates = 55)
  (h_lobster : lobster_rolls = 25)
  (h_spicy : spicy_hot_noodles = 14) :
  total_plates - (lobster_rolls + spicy_hot_noodles) = 16 :=
by
  sorry

end seafood_noodles_l151_151114


namespace recurrence_solution_proof_l151_151061

noncomputable def recurrence_relation (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 2, a n = 5 * a (n - 1) - 6 * a (n - 2) + n + 2) ∧
  a 0 = 27 / 4 ∧
  a 1 = 49 / 4

noncomputable def solution (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = 3 * 2^n + 3^n + n / 2 + 11 / 4

theorem recurrence_solution_proof : ∃ a : ℕ → ℚ, recurrence_relation a ∧ solution a :=
by { sorry }

end recurrence_solution_proof_l151_151061


namespace nonnegative_integer_with_divisors_is_multiple_of_6_l151_151390

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end nonnegative_integer_with_divisors_is_multiple_of_6_l151_151390


namespace michael_birth_year_l151_151483

theorem michael_birth_year (first_imo_year : ℕ) (annual_event : ∀ n : ℕ, n > 0 → (first_imo_year + n) ≥ first_imo_year) 
  (michael_age_at_10th_imo : ℕ) (imo_count : ℕ) 
  (H1 : first_imo_year = 1959) (H2 : imo_count = 10) (H3 : michael_age_at_10th_imo = 15) : 
  (first_imo_year + imo_count - 1 - michael_age_at_10th_imo = 1953) := 
by 
  sorry

end michael_birth_year_l151_151483


namespace three_digit_numbers_satisfying_condition_l151_151775

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l151_151775


namespace wario_missed_field_goals_wide_right_l151_151091

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end wario_missed_field_goals_wide_right_l151_151091


namespace pet_store_animals_left_l151_151797

def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15

def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

def birds_left : Nat := initial_birds - birds_sold
def puppies_left : Nat := initial_puppies - puppies_adopted
def cats_left : Nat := initial_cats
def spiders_left : Nat := initial_spiders - spiders_loose

def total_animals_left : Nat := birds_left + puppies_left + cats_left + spiders_left

theorem pet_store_animals_left : total_animals_left = 25 :=
by
  sorry

end pet_store_animals_left_l151_151797


namespace denominator_expression_l151_151316

theorem denominator_expression (x y a b E : ℝ)
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / E = 3)
  (h3 : a / b = 4.5) : E = 3 * b - y :=
sorry

end denominator_expression_l151_151316


namespace interval_is_correct_l151_151623

def total_population : ℕ := 2000
def sample_size : ℕ := 40
def interval_between_segments (N : ℕ) (n : ℕ) : ℕ := N / n

theorem interval_is_correct : interval_between_segments total_population sample_size = 50 :=
by
  sorry

end interval_is_correct_l151_151623


namespace beads_used_total_l151_151725

theorem beads_used_total :
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
  bracelets_wednesday * beads_per_bracelet + 
  earrings_wednesday * beads_per_earring = 325 := by
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  calc
    (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
    bracelets_wednesday * beads_per_bracelet + 
    earrings_wednesday * beads_per_earring
    = (10 + 2) * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 12 * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 240 + 50 + 35 : by rfl
    ... = 325 : by rfl

end beads_used_total_l151_151725


namespace find_x_value_l151_151824

noncomputable def floor_plus_2x_eq_33 (x : ℝ) : Prop :=
  ∃ n : ℤ, ⌊x⌋ = n ∧ n + 2 * x = 33 ∧  (0 : ℝ) ≤ x - n ∧ x - n < 1

theorem find_x_value : ∀ x : ℝ, floor_plus_2x_eq_33 x → x = 11 :=
by
  intro x
  intro h
  -- Proof skipped, included as 'sorry' to compile successfully.
  sorry

end find_x_value_l151_151824


namespace quadrilateral_is_parallelogram_l151_151075

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : (a - c) ^ 2 + (b - d) ^ 2 = 0) : 
  -- The theorem states that if lengths a, b, c, d of a quadrilateral satisfy the given equation,
  -- then the quadrilateral must be a parallelogram.
  a = c ∧ b = d :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l151_151075


namespace min_sum_squares_l151_151252

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l151_151252


namespace spending_on_hydrangeas_l151_151347

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l151_151347


namespace width_of_wide_flags_l151_151543

def total_fabric : ℕ := 1000
def leftover_fabric : ℕ := 294
def num_square_flags : ℕ := 16
def square_flag_area : ℕ := 16
def num_tall_flags : ℕ := 10
def tall_flag_area : ℕ := 15
def num_wide_flags : ℕ := 20
def wide_flag_height : ℕ := 3

theorem width_of_wide_flags :
  (total_fabric - leftover_fabric - (num_square_flags * square_flag_area + num_tall_flags * tall_flag_area)) / num_wide_flags / wide_flag_height = 5 :=
by
  sorry

end width_of_wide_flags_l151_151543


namespace inequality_abc_l151_151164

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) :
  (a / (b * c + 1)) + (b / (a * c + 1)) + (c / (a * b + 1)) ≤ 2 := by
  sorry

end inequality_abc_l151_151164


namespace sqrt_difference_l151_151101

theorem sqrt_difference:
  sqrt (49 + 81) - sqrt (36 - 9) = sqrt 130 - 3 * sqrt 3 :=
by
  sorry

end sqrt_difference_l151_151101


namespace limit_of_sequence_l151_151761

noncomputable def sequence := ℕ → ℝ
variables {a : sequence}

def a1 : ℝ := π / 4

def a_n (n : ℕ) (prev : ℝ) : ℝ :=
  ∫ x in 0..(1/2), (cos (π * x) + prev) * cos (π * x)

theorem limit_of_sequence :
  (∃ L, Tendsto (λ n, a n) atTop (𝓝 L) ∧ L = π / (4 * (π - 1))) :=
by
  sorry

end limit_of_sequence_l151_151761


namespace probability_of_chosen_figure_is_circle_l151_151264

-- Define the total number of figures and number of circles.
def total_figures : ℕ := 12
def number_of_circles : ℕ := 5

-- Define the probability calculation.
def probability_of_circle (total : ℕ) (circles : ℕ) : ℚ := circles / total

-- State the theorem using the defined conditions.
theorem probability_of_chosen_figure_is_circle : 
  probability_of_circle total_figures number_of_circles = 5 / 12 :=
by
  sorry  -- Placeholder for the actual proof.

end probability_of_chosen_figure_is_circle_l151_151264


namespace solve_absolute_value_equation_l151_151222

theorem solve_absolute_value_equation (x : ℝ) : x^2 - 3 * |x| - 4 = 0 ↔ x = 4 ∨ x = -4 :=
by
  sorry

end solve_absolute_value_equation_l151_151222


namespace area_of_rectangular_garden_l151_151074

-- Definition of conditions
def width : ℕ := 14
def length : ℕ := 3 * width

-- Statement for proof of the area of the rectangular garden
theorem area_of_rectangular_garden :
  length * width = 588 := 
by
  sorry

end area_of_rectangular_garden_l151_151074


namespace ratio_owners_on_horse_l151_151645

-- Definitions based on the given conditions.
def number_of_horses : Nat := 12
def number_of_owners : Nat := 12
def total_legs_walking_on_ground : Nat := 60
def owner_leg_count : Nat := 2
def horse_leg_count : Nat := 4
def total_owners_leg_horse_count : Nat := owner_leg_count + horse_leg_count

-- Prove the ratio of the number of owners on their horses' back to the total number of owners is 1:6
theorem ratio_owners_on_horse (R W : Nat) 
  (h1 : R + W = number_of_owners)
  (h2 : total_owners_leg_horse_count * W = total_legs_walking_on_ground) :
  R = 2 → W = 10 → (R : Nat)/(number_of_owners : Nat) = (1 : Nat)/(6 : Nat) := 
sorry

end ratio_owners_on_horse_l151_151645


namespace matrix_scalars_exist_l151_151333

namespace MatrixProof

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, -1]]

theorem matrix_scalars_exist :
  ∃ r s : ℝ, B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ r = 0 ∧ s = 64 := by
  sorry

end MatrixProof

end matrix_scalars_exist_l151_151333


namespace percent_increase_second_half_century_l151_151997

variable (P : ℝ) -- Initial population
variable (x : ℝ) -- Percentage increase in the second half of the century

noncomputable def population_first_half_century := 3 * P
noncomputable def population_end_century := P + 11 * P

theorem percent_increase_second_half_century :
  3 * P + (x / 100) * (3 * P) = 12 * P → x = 300 :=
by
  intro h
  sorry

end percent_increase_second_half_century_l151_151997


namespace parity_of_f_monotonicity_of_f_9_l151_151970

-- Condition: f(x) = x + k / x with k ≠ 0
variable (k : ℝ) (hkn0 : k ≠ 0)
noncomputable def f (x : ℝ) : ℝ := x + k / x

-- 1. Prove the parity of the function is odd
theorem parity_of_f : ∀ x : ℝ, f k (-x) = -f k x := by
  sorry

-- Given condition: f(3) = 6, we derive k = 9
def k_9 : ℝ := 9
noncomputable def f_9 (x : ℝ) : ℝ := x + k_9 / x

-- 2. Prove the monotonicity of the function y = f(x) in the interval (-∞, -3]
theorem monotonicity_of_f_9 : ∀ (x1 x2 : ℝ), x1 < x2 → x1 ≤ -3 → x2 ≤ -3 → f_9 x1 < f_9 x2 := by
  sorry

end parity_of_f_monotonicity_of_f_9_l151_151970


namespace number_of_students_to_bring_donuts_l151_151398

theorem number_of_students_to_bring_donuts (students_brownies students_cookies students_donuts : ℕ) :
  (students_brownies * 12 * 2) + (students_cookies * 24 * 2) + (students_donuts * 12 * 2) = 2040 →
  students_brownies = 30 →
  students_cookies = 20 →
  students_donuts = 15 :=
by
  -- Proof skipped
  sorry

end number_of_students_to_bring_donuts_l151_151398


namespace math_problem_proof_l151_151281

theorem math_problem_proof :
    24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 :=
by
  -- Proof omitted
  sorry

end math_problem_proof_l151_151281


namespace sufficient_but_not_necessary_to_increasing_l151_151894

theorem sufficient_but_not_necessary_to_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → (x^2 - 2*a*x) ≤ (y^2 - 2*a*y)) ↔ (a ≤ 1) := sorry

end sufficient_but_not_necessary_to_increasing_l151_151894


namespace johns_age_l151_151397

variable (J : ℕ)

theorem johns_age :
  J - 5 = (1 / 2) * (J + 8) → J = 18 := by
    sorry

end johns_age_l151_151397


namespace smallest_three_digit_number_l151_151013

theorem smallest_three_digit_number (digits : Finset ℕ) (h_digits : digits = {0, 3, 5, 6}) : 
  ∃ n, n = 305 ∧ ∀ m, (m ∈ digits) → (m ≠ 0) → (m < 305) → false :=
by
  sorry

end smallest_three_digit_number_l151_151013


namespace lagrange_intermediate_value_l151_151890

open Set

variable {a b : ℝ} (f : ℝ → ℝ)

-- Ensure that a < b for the interval [a, b]
axiom hab : a < b

-- Assume f is differentiable on [a, b]
axiom differentiable_on_I : DifferentiableOn ℝ f (Icc a b)

theorem lagrange_intermediate_value :
  ∃ (x0 : ℝ), x0 ∈ Ioo a b ∧ (deriv f x0) = (f a - f b) / (a - b) :=
sorry

end lagrange_intermediate_value_l151_151890


namespace estimate_fish_in_pond_l151_151048

theorem estimate_fish_in_pond
  (n m k : ℕ)
  (h_pr: k = 200)
  (h_cr: k = 8)
  (h_m: n = 200):
  n / (m / k) = 5000 := sorry

end estimate_fish_in_pond_l151_151048


namespace common_root_cubic_polynomials_l151_151249

open Real

theorem common_root_cubic_polynomials (a b c : ℝ)
  (h1 : ∃ α : ℝ, α^3 - a * α^2 + b = 0 ∧ α^3 - b * α^2 + c = 0)
  (h2 : ∃ β : ℝ, β^3 - b * β^2 + c = 0 ∧ β^3 - c * β^2 + a = 0)
  (h3 : ∃ γ : ℝ, γ^3 - c * γ^2 + a = 0 ∧ γ^3 - a * γ^2 + b = 0)
  : a = b ∧ b = c :=
sorry

end common_root_cubic_polynomials_l151_151249


namespace count_integer_values_l151_151317

theorem count_integer_values (π : Real) (hπ : Real.pi = π):
  ∃ n : ℕ, n = 27 ∧ ∀ x : ℤ, |(x:Real)| < 4 * π + 1 ↔ -13 ≤ x ∧ x ≤ 13 :=
by sorry

end count_integer_values_l151_151317


namespace no_extrema_1_1_l151_151676

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 3 * x

theorem no_extrema_1_1 : ∀ x : ℝ, (x > -1) ∧ (x < 1) → ¬ (∃ c : ℝ, c ∈ Set.Ioo (-1) (1) ∧ (∀ y ∈ Set.Ioo (-1) (1), f y ≤ f c ∨ f c ≤ f y)) :=
by
  sorry

end no_extrema_1_1_l151_151676


namespace tangent_BD_circumcircle_ADZ_l151_151123

noncomputable section

-- Define the setup
variable {A B C D E X Y Z : Point} -- Points in the plane
variable {circumcircle : Circle} -- Circumcircle of triangle ABC

-- Define the conditions
variables (h₁ : ∠A = 90)
           (h₂ : isTangentAt A circumcircle line_BC D)
           (h₃ : reflection_line_BC A = E)
           (h₄ : foot_perpendicular A line_BE = X)
           (h₅ : midpoint AX = Y)
           (h₆ : line_BY_meets_circumcircle_at_B_again Z)

-- The statement to be proved
theorem tangent_BD_circumcircle_ADZ
  (h₁ : ∠A = 90)
  (h₂ : isTangentAt A circumcircle line_BC D)
  (h₃ : reflection_line_BC A = E)
  (h₄ : foot_perpendicular A line_BE = X)
  (h₅ : midpoint AX = Y)
  (h₆ : line_BY_meets_circumcircle_at_B_again Z) :
  isTangentAt D (circumcircle_triangle AD Z) line_BD :=
sorry

end tangent_BD_circumcircle_ADZ_l151_151123


namespace count_three_digit_distinct_under_800_l151_151979

-- Definitions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 800
def distinct_digits (n : ℕ) : Prop := (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) 

-- Theorem
theorem count_three_digit_distinct_under_800 : ∃ k : ℕ, k = 504 ∧ ∀ n : ℕ, is_three_digit n → distinct_digits n → n < 800 :=
by 
  exists 504
  sorry

end count_three_digit_distinct_under_800_l151_151979


namespace sculpture_cost_in_inr_l151_151474

def convert_currency (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) : ℕ := 
  (n_cost / n_to_b_rate) * b_to_i_rate

theorem sculpture_cost_in_inr (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) :
  n_cost = 360 → 
  n_to_b_rate = 18 → 
  b_to_i_rate = 20 →
  convert_currency n_cost n_to_b_rate b_to_i_rate = 400 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- turns 360 / 18 * 20 = 400
  sorry

end sculpture_cost_in_inr_l151_151474


namespace smallest_possible_perimeter_l151_151916

open Real

theorem smallest_possible_perimeter
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 + b^2 = 2016) :
  a + b + 2^3 * 3 * sqrt 14 = 48 + 2^3 * 3 * sqrt 14 :=
sorry

end smallest_possible_perimeter_l151_151916


namespace part1_A_inter_B_and_union_A_B_part2_range_of_a_l151_151044

open Set

-- Define sets A and B under given conditions
def set_A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 + a < 0 }

theorem part1_A_inter_B_and_union_A_B :
  set_B (-4) ∩ set_A = {x | (1/2 : ℝ) ≤ x ∧ x < 2} ∧
  set_B (-4) ∪ set_A = {x | -2 < x ∧ x ≤ 3} := sorry

theorem part2_range_of_a :
  {a : ℝ | (∅ ∩ set_A = (∅ : Set ℝ)) ∧ (∀ x, x ∈ (set_B a) → x ∈ (set_Aᶜ))} = {a | a ≥ -1/4} := sorry

end part1_A_inter_B_and_union_A_B_part2_range_of_a_l151_151044


namespace tapB_fill_in_20_l151_151879

-- Conditions definitions
def tapA_rate (A: ℝ) : Prop := A = 3 -- Tap A fills 3 liters per minute
def total_volume (V: ℝ) : Prop := V = 36 -- Total bucket volume is 36 liters
def together_fill_time (t: ℝ) : Prop := t = 10 -- Both taps fill the bucket in 10 minutes

-- Tap B's rate can be derived from these conditions
def tapB_rate (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : Prop := V - (A * t) = B * t

-- The final question we need to prove
theorem tapB_fill_in_20 (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : 
  tapA_rate A → total_volume V → together_fill_time t → tapB_rate B A V t → B * 20 = 12 := by
  sorry

end tapB_fill_in_20_l151_151879


namespace garden_area_maximal_l151_151835

/-- Given a garden with sides 20 meters, 16 meters, 12 meters, and 10 meters, 
    prove that the area is approximately 194.4 square meters. -/
theorem garden_area_maximal (a b c d : ℝ) (h1 : a = 20) (h2 : b = 16) (h3 : c = 12) (h4 : d = 10) :
    ∃ A : ℝ, abs (A - 194.4) < 0.1 :=
by
  sorry

end garden_area_maximal_l151_151835


namespace anya_takes_home_balloons_l151_151358

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end anya_takes_home_balloons_l151_151358


namespace cashier_window_open_probability_l151_151658

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l151_151658


namespace remainder_of_power_division_l151_151362

theorem remainder_of_power_division :
  (2^222 + 222) % (2^111 + 2^56 + 1) = 218 :=
by sorry

end remainder_of_power_division_l151_151362


namespace trajectory_center_of_C_number_of_lines_l_l151_151838

noncomputable def trajectory_equation : Prop :=
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_count : Prop :=
  ∀ (k m : ℤ), 
  ∃ (num_lines : ℕ), 
  (∀ (x : ℝ), (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 48 = 0 → num_lines = 9 ∨ num_lines = 0) ∧
  (∀ (x : ℝ), (3 - k^2) * x^2 - 2 * k * m * x - m^2 - 12 = 0 → num_lines = 9 ∨ num_lines = 0)

theorem trajectory_center_of_C :
  trajectory_equation :=
sorry

theorem number_of_lines_l :
  line_count :=
sorry

end trajectory_center_of_C_number_of_lines_l_l151_151838


namespace jenga_blocks_before_jess_turn_l151_151454

theorem jenga_blocks_before_jess_turn
  (total_blocks : ℕ := 54)
  (players : ℕ := 5)
  (rounds : ℕ := 5)
  (father_turn_blocks : ℕ := 1)
  (original_blocks := total_blocks - (players * rounds + father_turn_blocks))
  (jess_turn_blocks : ℕ := total_blocks - original_blocks):
  jess_turn_blocks = 28 := by
begin
  sorry
end

end jenga_blocks_before_jess_turn_l151_151454


namespace tangent_at_point_l151_151972

theorem tangent_at_point (a b : ℝ) :
  (∀ x : ℝ, (x^3 - x^2 - a * x + b) = 2 * x + 1) →
  (a + b = -1) :=
by
  intro tangent_condition
  sorry

end tangent_at_point_l151_151972


namespace three_digit_max_l151_151897

theorem three_digit_max (n : ℕ) : 
  n % 9 = 1 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 <= n ∧ n <= 999 → n = 793 :=
by
  sorry

end three_digit_max_l151_151897


namespace sequence_conjecture_l151_151403

theorem sequence_conjecture (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (a n + 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / n := by
  sorry

end sequence_conjecture_l151_151403


namespace num_of_arithmetic_sequences_l151_151849

-- Define the set of digits {1, 2, ..., 15}
def digits := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define an arithmetic sequence condition 
def is_arithmetic_sequence (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

-- Define the count of valid sequences with a specific difference
def count_arithmetic_sequences_with_difference (d : ℕ) : ℕ :=
  if d = 1 then 13
  else if d = 5 then 6
  else 0

-- Define the total count of valid sequences
def total_arithmetic_sequences : ℕ :=
  count_arithmetic_sequences_with_difference 1 +
  count_arithmetic_sequences_with_difference 5

-- The final statement to prove
theorem num_of_arithmetic_sequences : total_arithmetic_sequences = 19 := 
  sorry

end num_of_arithmetic_sequences_l151_151849


namespace min_distance_squared_l151_151254

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l151_151254


namespace sqrt_square_eq_self_sqrt_784_square_l151_151282

theorem sqrt_square_eq_self (n : ℕ) (h : n ≥ 0) : (Real.sqrt n) ^ 2 = n :=
by
  sorry

theorem sqrt_784_square : (Real.sqrt 784) ^ 2 = 784 :=
by
  exact sqrt_square_eq_self 784 (Nat.zero_le 784)

end sqrt_square_eq_self_sqrt_784_square_l151_151282


namespace largest_angle_in_hexagon_l151_151065

theorem largest_angle_in_hexagon :
  ∀ (x : ℝ), (2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) →
  5 * x = 1200 / 7 :=
by
  intros x h
  sorry

end largest_angle_in_hexagon_l151_151065


namespace point_in_fourth_quadrant_l151_151996

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l151_151996


namespace find_initial_nickels_l151_151739

variable (initial_nickels current_nickels borrowed_nickels : ℕ)

def initial_nickels_equation (initial_nickels current_nickels borrowed_nickels : ℕ) : Prop :=
  initial_nickels - borrowed_nickels = current_nickels

theorem find_initial_nickels (h : initial_nickels_equation initial_nickels current_nickels borrowed_nickels) 
                             (h_current : current_nickels = 11) 
                             (h_borrowed : borrowed_nickels = 20) : 
                             initial_nickels = 31 :=
by
  sorry

end find_initial_nickels_l151_151739


namespace probability_457_more_ones_than_sixes_l151_151185

noncomputable def probability_of_more_ones_than_sixes : ℚ :=
  let total_ways := 6^6 in
  let same_number_ways :=
    (4^6) + (Nat.choose 6 1 * Nat.choose 5 1 * 4^4) + (Nat.choose 6 2 * Nat.choose 4 2 * 4^2) + (Nat.choose 6 3 * Nat.choose 3 3) in
  let desired_probability := (1/2) * (1 - (same_number_ways / total_ways)) in
  let simplified_probability := 8355 / 23328 in
  simplified_probability

theorem probability_457_more_ones_than_sixes :
  probability_of_more_ones_than_sixes = 8355 / 23328 := sorry

end probability_457_more_ones_than_sixes_l151_151185


namespace total_cost_correct_l151_151723

-- Define the costs for each repair
def engine_labor_cost := 75 * 16
def engine_part_cost := 1200
def brake_labor_cost := 85 * 10
def brake_part_cost := 800
def tire_labor_cost := 50 * 4
def tire_part_cost := 600

-- Calculate the total costs
def engine_total_cost := engine_labor_cost + engine_part_cost
def brake_total_cost := brake_labor_cost + brake_part_cost
def tire_total_cost := tire_labor_cost + tire_part_cost

-- Calculate the total combined cost
def total_combined_cost := engine_total_cost + brake_total_cost + tire_total_cost

-- The theorem to prove
theorem total_cost_correct : total_combined_cost = 4850 := by
  sorry

end total_cost_correct_l151_151723


namespace solve_inequality_solution_set_l151_151762

def solution_set (x : ℝ) : Prop := -x^2 + 5 * x > 6

theorem solve_inequality_solution_set :
  { x : ℝ | solution_set x } = { x : ℝ | 2 < x ∧ x < 3 } :=
sorry

end solve_inequality_solution_set_l151_151762


namespace esther_walks_975_yards_l151_151594

def miles_to_feet (miles : ℕ) : ℕ := miles * 5280
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

variable (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) (esther_yards : ℕ)
variable (h_lionel : lionel_miles = 4)
variable (h_niklaus : niklaus_feet = 1287)
variable (h_total : total_feet = 25332)
variable (h_esther : esther_yards = 975)

theorem esther_walks_975_yards :
  let lionel_distance_in_feet := miles_to_feet lionel_miles
  let combined_distance := lionel_distance_in_feet + niklaus_feet
  let esther_distance_in_feet := total_feet - combined_distance
  feet_to_yards esther_distance_in_feet = esther_yards := by {
    sorry
  }

end esther_walks_975_yards_l151_151594


namespace find_n_in_sequence_l151_151415

theorem find_n_in_sequence (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = 2 * a n) 
    (h3 : S n = 126) 
    (h4 : S n = 2^(n+1) - 2) : 
  n = 6 :=
sorry

end find_n_in_sequence_l151_151415


namespace number_of_players_tournament_l151_151480

theorem number_of_players_tournament (n : ℕ) : 
  (2 * n * (n - 1) = 272) → n = 17 :=
by
  sorry

end number_of_players_tournament_l151_151480


namespace fran_ate_15_green_macaroons_l151_151299

variable (total_red total_green initial_remaining green_macaroons_eaten : ℕ)

-- Conditions as definitions
def initial_red_macaroons := 50
def initial_green_macaroons := 40
def total_macaroons := 90
def remaining_macaroons := 45

-- Total eaten macaroons
def total_eaten_macaroons (G : ℕ) := G + 2 * G

-- The proof statement
theorem fran_ate_15_green_macaroons
  (h1 : total_red = initial_red_macaroons)
  (h2 : total_green = initial_green_macaroons)
  (h3 : initial_remaining = remaining_macaroons)
  (h4 : total_macaroons = initial_red_macaroons + initial_green_macaroons)
  (h5 : initial_remaining = total_macaroons - total_eaten_macaroons green_macaroons_eaten):
  green_macaroons_eaten = 15 :=
  by
  sorry

end fran_ate_15_green_macaroons_l151_151299


namespace students_with_uncool_parents_but_cool_siblings_l151_151448

-- The total number of students in the classroom
def total_students : ℕ := 40

-- The number of students with cool dads
def students_with_cool_dads : ℕ := 18

-- The number of students with cool moms
def students_with_cool_moms : ℕ := 22

-- The number of students with both cool dads and cool moms
def students_with_both_cool_parents : ℕ := 10

-- The number of students with cool siblings
def students_with_cool_siblings : ℕ := 8

-- The theorem we want to prove
theorem students_with_uncool_parents_but_cool_siblings
  (h1 : total_students = 40)
  (h2 : students_with_cool_dads = 18)
  (h3 : students_with_cool_moms = 22)
  (h4 : students_with_both_cool_parents = 10)
  (h5 : students_with_cool_siblings = 8) :
  8 = (students_with_cool_siblings) :=
sorry

end students_with_uncool_parents_but_cool_siblings_l151_151448


namespace expr_is_irreducible_fraction_l151_151820

def a : ℚ := 3 / 2015
def b : ℚ := 11 / 2016

noncomputable def expr : ℚ := 
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a

theorem expr_is_irreducible_fraction : expr = 11 / 112 := by
  sorry

end expr_is_irreducible_fraction_l151_151820


namespace theater_ticket_cost_l151_151921

theorem theater_ticket_cost
  (num_persons : ℕ) 
  (num_children : ℕ) 
  (num_adults : ℕ)
  (children_ticket_cost : ℕ)
  (total_receipts_cents : ℕ)
  (A : ℕ) :
  num_persons = 280 →
  num_children = 80 →
  children_ticket_cost = 25 →
  total_receipts_cents = 14000 →
  num_adults = num_persons - num_children →
  200 * A + (num_children * children_ticket_cost) = total_receipts_cents →
  A = 60 :=
by
  intros h_num_persons h_num_children h_children_ticket_cost h_total_receipts_cents h_num_adults h_eqn
  sorry

end theater_ticket_cost_l151_151921


namespace exp_log_pb_eq_log_ba_l151_151577

noncomputable def log_b (b a : ℝ) := Real.log a / Real.log b

theorem exp_log_pb_eq_log_ba (a b p : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : p = log_b b (log_b b a) / log_b b a) :
  a^p = log_b b a :=
by
  sorry

end exp_log_pb_eq_log_ba_l151_151577


namespace milk_price_same_after_reductions_l151_151244

theorem milk_price_same_after_reductions (x : ℝ) (h1 : 0 < x) :
  (x - 0.4 * x) = ((x - 0.2 * x) - 0.25 * (x - 0.2 * x)) :=
by
  sorry

end milk_price_same_after_reductions_l151_151244


namespace find_EQ_length_l151_151089

theorem find_EQ_length (a b c d : ℕ) (parallel : Prop) (circle_tangent : Prop) :
  a = 105 ∧ b = 45 ∧ c = 21 ∧ d = 80 ∧ parallel ∧ circle_tangent → (∃ x : ℚ, x = 336 / 5) :=
by
  sorry

end find_EQ_length_l151_151089


namespace surface_area_circumscribed_sphere_l151_151062

-- Define the problem
theorem surface_area_circumscribed_sphere (a b c : ℝ)
    (h1 : a^2 + b^2 = 3)
    (h2 : b^2 + c^2 = 5)
    (h3 : c^2 + a^2 = 4) : 
    4 * Real.pi * (a^2 + b^2 + c^2) / 4 = 6 * Real.pi :=
by
  -- The proof is omitted
  sorry

end surface_area_circumscribed_sphere_l151_151062


namespace johny_journey_distance_l151_151034

def south_distance : ℕ := 40
def east_distance : ℕ := south_distance + 20
def north_distance : ℕ := 2 * east_distance
def total_distance : ℕ := south_distance + east_distance + north_distance

theorem johny_journey_distance :
  total_distance = 220 := by
  sorry

end johny_journey_distance_l151_151034


namespace find_multiple_of_ron_l151_151874

variable (R_d R_g R_n m : ℕ)

def rodney_can_lift_146 : Prop := R_d = 146
def combined_weight_239 : Prop := R_d + R_g + R_n = 239
def rodney_twice_as_roger : Prop := R_d = 2 * R_g
def roger_seven_less_than_multiple_of_ron : Prop := R_g = m * R_n - 7

theorem find_multiple_of_ron (h1 : rodney_can_lift_146 R_d) 
                             (h2 : combined_weight_239 R_d R_g R_n) 
                             (h3 : rodney_twice_as_roger R_d R_g) 
                             (h4 : roger_seven_less_than_multiple_of_ron R_g R_n m) 
                             : m = 4 :=
by 
    sorry

end find_multiple_of_ron_l151_151874


namespace sum_of_ages_l151_151113

variable (S F : ℕ)

-- Conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F + 6 = 2 * (S + 6)

-- Theorem Statement
theorem sum_of_ages (h1 : condition1 S F) (h2 : condition2 S F) : S + 6 + (F + 6) = 36 := by
  sorry

end sum_of_ages_l151_151113


namespace percentage_books_not_sold_is_60_percent_l151_151458

def initial_stock : ℕ := 700
def sold_monday : ℕ := 50
def sold_tuesday : ℕ := 82
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40

def total_sold : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
def books_not_sold : ℕ := initial_stock - total_sold
def percentage_not_sold : ℚ := (books_not_sold * 100) / initial_stock

theorem percentage_books_not_sold_is_60_percent : percentage_not_sold = 60 := by
  sorry

end percentage_books_not_sold_is_60_percent_l151_151458


namespace find_difference_of_squares_l151_151426

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l151_151426


namespace largest_prime_factor_9801_l151_151553

/-- Definition to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Definition for the largest prime factor of a number -/
def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else Classical.find (Nat.exists_greatest_prime_factor n h)

/-- Condition: 241 and 41 are prime factors of 9801 -/
def prime_factors_9801 : ∀ (p : ℕ), p ∣ 9801 → is_prime p → (p = 41 ∨ p = 241) :=
λ p hdiv hprime, by {
  obtain ⟨a, ha⟩ := Nat.exists_mul_of_dvd hdiv,
  have h : 9801 = 41 * 241 := rfl,
  have ph31 : is_prime 41 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have ph241 : is_prime 241 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have h_9801 : 9801 = 41 * 241, by refl,
  rw [ha, h_9801] at *,
  cases ha with ha l,
  { exact Or.inl ha },
  { exact Or.inr ha },
}

/-- Statement: the largest prime factor of 9801 is 241 -/
theorem largest_prime_factor_9801 : largest_prime_factor 9801 = 241 :=
by {
  rw largest_prime_factor,
  have h1 : 9801 = 41 * 241 := rfl,
  exact Classical.find_spec {
    exists := 241, 
    h := _,
    obtain ⟨m, hm⟩ := Nat.exists_dvd_of_not_prime,
  sorry,
}

end largest_prime_factor_9801_l151_151553


namespace Anya_took_home_balloons_l151_151359

theorem Anya_took_home_balloons :
  ∃ (balloons_per_color : ℕ), 
  ∃ (yellow_balloons_home : ℕ), 
  (672 = 4 * balloons_per_color) ∧ 
  (yellow_balloons_home = balloons_per_color / 2) ∧ 
  (yellow_balloons_home = 84) :=
begin
  sorry
end

end Anya_took_home_balloons_l151_151359


namespace complex_props_hold_l151_151298

theorem complex_props_hold (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ (a^2 = a*b → a = b) :=
by
  sorry

end complex_props_hold_l151_151298


namespace cashier_opens_probability_l151_151663

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l151_151663


namespace usage_difference_correct_l151_151632

def computerUsageLastWeek : ℕ := 91

def computerUsageThisWeek : ℕ :=
  let first4days := 4 * 8
  let last3days := 3 * 10
  first4days + last3days

def computerUsageFollowingWeek : ℕ :=
  let weekdays := 5 * (5 + 3)
  let weekends := 2 * 12
  weekdays + weekends

def differenceThisWeek : ℕ := computerUsageLastWeek - computerUsageThisWeek
def differenceFollowingWeek : ℕ := computerUsageLastWeek - computerUsageFollowingWeek

theorem usage_difference_correct :
  differenceThisWeek = 29 ∧ differenceFollowingWeek = 27 := by
  sorry

end usage_difference_correct_l151_151632


namespace trebled_resultant_is_correct_l151_151653

-- Let's define the initial number and the transformations
def initial_number := 17
def doubled (n : ℕ) := n * 2
def added_five (n : ℕ) := n + 5
def trebled (n : ℕ) := n * 3

-- Finally, we state the problem to prove
theorem trebled_resultant_is_correct : 
  trebled (added_five (doubled initial_number)) = 117 :=
by
  -- Here we just print sorry which means the proof is expected but not provided yet.
  sorry

end trebled_resultant_is_correct_l151_151653


namespace smallest_positive_integer_y_l151_151096

theorem smallest_positive_integer_y
  (y : ℕ)
  (h_pos : 0 < y)
  (h_ineq : y^3 > 80) :
  y = 5 :=
sorry

end smallest_positive_integer_y_l151_151096


namespace log_relationship_l151_151699

open Real

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then log x / log 2 else f (-x)

theorem log_relationship :
  let a := f (-3)
  let b := f (1 / 4)
  let c := f 2
  a > c ∧ c > b :=
by
  let a := f (-3)
  let b := f (1 / 4)
  let c := f 2
  sorry

end log_relationship_l151_151699


namespace two_mathematicians_contemporaries_l151_151502

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l151_151502


namespace sugar_percentage_l151_151052

theorem sugar_percentage (S : ℝ) (P : ℝ) : 
  (3 / 4 * S * 0.10 + (1 / 4) * S * P / 100 = S * 0.20) → 
  P = 50 := 
by 
  intro h
  sorry

end sugar_percentage_l151_151052


namespace value_of_f_at_6_l151_151166

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom odd_function (x : R) : f (-x) = -f x
axiom periodicity (x : R) : f (x + 2) = -f x

-- Theorem to prove
theorem value_of_f_at_6 : f 6 = 0 := by sorry

end value_of_f_at_6_l151_151166


namespace max_x1_squared_plus_x2_squared_l151_151784

theorem max_x1_squared_plus_x2_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = k - 2)
  (h2 : x₁ * x₂ = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) :
  x₁ ^ 2 + x₂ ^ 2 ≤ 18 :=
sorry

end max_x1_squared_plus_x2_squared_l151_151784


namespace exam_time_ratio_l151_151366

-- Lean statements to define the problem conditions and goal
theorem exam_time_ratio (x M : ℝ) (h1 : x > 0) (h2 : M = x / 18) : 
  (5 * x / 6 + 2 * M) / (x / 6 - 2 * M) = 17 := by
  sorry

end exam_time_ratio_l151_151366


namespace min_value_l151_151839

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 1/(a-1) + 4/(b-1) ≥ 4 :=
by
  sorry

end min_value_l151_151839


namespace steps_probability_to_point_3_3_l151_151481

theorem steps_probability_to_point_3_3 : 
  let a := 35
  let b := 4096
  a + b = 4131 :=
by {
  sorry
}

end steps_probability_to_point_3_3_l151_151481


namespace sum_of_g_of_nine_values_l151_151589

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (y : ℝ) : ℝ := 3 * y - 4

theorem sum_of_g_of_nine_values : (g 9) = 19 := by
  sorry

end sum_of_g_of_nine_values_l151_151589


namespace correct_assignment_statement_l151_151243

theorem correct_assignment_statement (a b : ℕ) : 
  (2 = a → False) ∧ 
  (a = a + 1 → True) ∧ 
  (a * b = 2 → False) ∧ 
  (a + 1 = a → False) :=
by {
  sorry
}

end correct_assignment_statement_l151_151243


namespace sandbox_length_l151_151531

theorem sandbox_length (width : ℕ) (area : ℕ) (h_width : width = 146) (h_area : area = 45552) : ∃ length : ℕ, length = 312 :=
by {
  sorry
}

end sandbox_length_l151_151531


namespace geometric_sequence_a6_l151_151030

variable {a : ℕ → ℝ} (h_geo : ∀ n, a (n+1) / a n = a (n+2) / a (n+1))

theorem geometric_sequence_a6 (h5 : a 5 = 2) (h7 : a 7 = 8) : a 6 = 4 ∨ a 6 = -4 :=
by
  sorry

end geometric_sequence_a6_l151_151030


namespace hcf_of_two_numbers_l151_151578

-- Definitions based on conditions
def LCM (x y : ℕ) : ℕ := sorry  -- Assume some definition of LCM
def HCF (x y : ℕ) : ℕ := sorry  -- Assume some definition of HCF

-- Given conditions
axiom cond1 (x y : ℕ) : LCM x y = 600
axiom cond2 (x y : ℕ) : x * y = 18000

-- Statement to prove
theorem hcf_of_two_numbers (x y : ℕ) (h1 : LCM x y = 600) (h2 : x * y = 18000) : HCF x y = 30 :=
by {
  -- Proof omitted, hence we use sorry
  sorry
}

end hcf_of_two_numbers_l151_151578


namespace zero_unique_multiple_prime_l151_151103

-- Condition: let n be a number
def n : Int := sorry

-- Condition: let p be any prime number
def is_prime (p : Int) : Prop := sorry  -- Predicate definition for prime number

-- Proof problem statement
theorem zero_unique_multiple_prime (n : Int) :
  (∀ p : Int, is_prime p → (∃ k : Int, n * p = k * p)) ↔ (n = 0) := by
  sorry

end zero_unique_multiple_prime_l151_151103


namespace trig_identity_solutions_l151_151112

open Real

theorem trig_identity_solutions (x : ℝ) (k n : ℤ) :
  (4 * sin x * cos (π / 2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3 * π / 2 - x) * cos (π + x) = 1) ↔ 
  (∃ k : ℤ, x = arctan (1 / 3) + π * k) ∨ (∃ n : ℤ, x = π / 4 + π * n) := 
sorry

end trig_identity_solutions_l151_151112


namespace max_value_ineq_l151_151041

theorem max_value_ineq (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_value_ineq_l151_151041


namespace no_solution_system_l151_151906

theorem no_solution_system (a : ℝ) :
  (∀ (x : ℝ), (a ≠ 0 → (a^2 * x + 2 * a) / (a * x - 2 + a^2) < 0 ∨ ax + a ≤ 5/4)) ∧ 
  (a = 0 → ¬ ∃ (x : ℝ), (a^2 * x + 2 * a) / (a * x - 2 + a^2) ≥ 0 ∧ ax + a > 5/4) ↔ 
  a ∈ Set.Iic (-1/2) ∪ {0} :=
by sorry

end no_solution_system_l151_151906


namespace fill_bathtub_with_drain_open_l151_151523

theorem fill_bathtub_with_drain_open :
  let fill_rate := 1 / 10
  let drain_rate := 1 / 12
  let net_fill_rate := fill_rate - drain_rate
  fill_rate = 1 / 10 ∧ drain_rate = 1 / 12 → 1 / net_fill_rate = 60 :=
by
  intros
  sorry

end fill_bathtub_with_drain_open_l151_151523


namespace length_PQ_calc_l151_151650

noncomputable def length_PQ 
  (F : ℝ × ℝ) 
  (P Q : ℝ × ℝ) 
  (hF : F = (1, 0)) 
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1) 
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1) 
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1) 
  (hx1x2 : P.1 + Q.1 = 9) : ℝ :=
|P.1 - Q.1|

theorem length_PQ_calc : ∀ F P Q
  (hF : F = (1, 0))
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1)
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1)
  (hx1x2 : P.1 + Q.1 = 9),
  length_PQ F P Q hF hP_on_parabola hQ_on_parabola hLine_through_focus hx1x2 = 11 := 
by
  sorry

end length_PQ_calc_l151_151650


namespace ordered_pair_exists_l151_151558

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l151_151558


namespace darry_full_ladder_climbs_l151_151138

-- Definitions and conditions
def full_ladder_steps : ℕ := 11
def smaller_ladder_steps : ℕ := 6
def smaller_ladder_climbs : ℕ := 7
def total_steps_climbed_today : ℕ := 152

-- Question: How many times did Darry climb his full ladder?
theorem darry_full_ladder_climbs (x : ℕ) 
  (H : 11 * x + smaller_ladder_steps * 7 = total_steps_climbed_today) : 
  x = 10 := by
  -- proof steps omitted, so we write
  sorry

end darry_full_ladder_climbs_l151_151138


namespace prove_m_add_n_l151_151016

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end prove_m_add_n_l151_151016


namespace probability_is_correct_l151_151666

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l151_151666


namespace top_card_probability_spades_or_clubs_l151_151919

-- Definitions
def total_cards : ℕ := 52
def suits : ℕ := 4
def ranks : ℕ := 13
def spades_cards : ℕ := ranks
def clubs_cards : ℕ := ranks
def favorable_outcomes : ℕ := spades_cards + clubs_cards

-- Probability calculation statement
theorem top_card_probability_spades_or_clubs :
  (favorable_outcomes : ℚ) / (total_cards : ℚ) = 1 / 2 :=
  sorry

end top_card_probability_spades_or_clubs_l151_151919


namespace find_two_fractions_sum_eq_86_over_111_l151_151949

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l151_151949


namespace apple_juice_production_l151_151064

noncomputable def apple_usage 
  (total_apples : ℝ) 
  (mixed_percentage : ℝ) 
  (juice_percentage : ℝ) 
  (sold_fresh_percentage : ℝ) : ℝ := 
  let mixed_apples := total_apples * mixed_percentage / 100
  let remainder_apples := total_apples - mixed_apples
  let juice_apples := remainder_apples * juice_percentage / 100
  juice_apples

theorem apple_juice_production :
  apple_usage 6 20 60 40 = 2.9 := 
by
  sorry

end apple_juice_production_l151_151064


namespace smallest_multiple_of_8_and_9_l151_151239

theorem smallest_multiple_of_8_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 9 = 0) ∧ (∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 9 = 0) → n ≤ m) ∧ n = 72 :=
by
  sorry

end smallest_multiple_of_8_and_9_l151_151239


namespace not_rain_probability_l151_151354

-- Define the probability of rain tomorrow
def prob_rain : ℚ := 3 / 10

-- Define the complementary probability (probability that it will not rain tomorrow)
def prob_no_rain : ℚ := 1 - prob_rain

-- Statement to prove: probability that it will not rain tomorrow equals 7/10 
theorem not_rain_probability : prob_no_rain = 7 / 10 := 
by sorry

end not_rain_probability_l151_151354


namespace jackson_vacuuming_time_l151_151453

-- Definitions based on the conditions
def hourly_wage : ℕ := 5
def washing_dishes_time : ℝ := 0.5
def cleaning_bathroom_time : ℝ := 3 * washing_dishes_time
def total_earnings : ℝ := 30

-- The total time spent on chores
def total_chore_time (V : ℝ) : ℝ :=
  2 * V + washing_dishes_time + cleaning_bathroom_time

-- The main theorem that needs to be proven
theorem jackson_vacuuming_time :
  ∃ V : ℝ, hourly_wage * total_chore_time V = total_earnings ∧ V = 2 :=
by
  sorry

end jackson_vacuuming_time_l151_151453


namespace point_in_fourth_quadrant_l151_151994

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l151_151994


namespace find_normal_price_l151_151237

theorem find_normal_price (P : ℝ) (S : ℝ) (d1 d2 d3 : ℝ) : 
  (P * (1 - d1) * (1 - d2) * (1 - d3) = S) → S = 144 → d1 = 0.12 → d2 = 0.22 → d3 = 0.15 → P = 246.81 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_normal_price_l151_151237


namespace expression_simplifies_to_36_l151_151985

theorem expression_simplifies_to_36 (x : ℝ) : (x + 1)^2 + 2 * (x + 1) * (5 - x) + (5 - x)^2 = 36 :=
by
  sorry

end expression_simplifies_to_36_l151_151985


namespace shirt_cost_l151_151106

theorem shirt_cost (J S : ℕ) 
  (h₁ : 3 * J + 2 * S = 69) 
  (h₂ : 2 * J + 3 * S = 61) :
  S = 9 :=
by 
  sorry

end shirt_cost_l151_151106


namespace walnut_price_l151_151625

theorem walnut_price {total_weight total_value walnut_price hazelnut_price : ℕ} 
  (h1 : total_weight = 55)
  (h2 : total_value = 1978)
  (h3 : walnut_price > hazelnut_price)
  (h4 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ hazelnut_price = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9)
  (h5 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ b = a - 1) : 
  walnut_price = 43 := 
sorry

end walnut_price_l151_151625


namespace variance_scaled_l151_151764

theorem variance_scaled (s1 : ℝ) (c : ℝ) (h1 : s1 = 3) (h2 : c = 3) :
  s1 * (c^2) = 27 :=
by
  rw [h1, h2]
  norm_num

end variance_scaled_l151_151764


namespace point_in_fourth_quadrant_l151_151993

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l151_151993


namespace consecutive_odd_integers_l151_151856

theorem consecutive_odd_integers (n : ℕ) (h1 : n > 0) (h2 : (1 : ℚ) / n * ((n : ℚ) * 154) = 154) : n = 10 :=
sorry

end consecutive_odd_integers_l151_151856


namespace calculate_expression_l151_151129

theorem calculate_expression : 
  -3^2 + Real.sqrt ((-2)^4) - (-27)^(1/3 : ℝ) = -2 := 
by
  sorry

end calculate_expression_l151_151129


namespace no_representation_of_216p3_l151_151307

theorem no_representation_of_216p3 (p : ℕ) (hp_prime : Nat.Prime p)
  (hp_form : ∃ m : ℤ, p = 4 * m + 1) : ¬ ∃ x y z : ℤ, 216 * (p ^ 3) = x^2 + y^2 + z^9 := by
  sorry

end no_representation_of_216p3_l151_151307


namespace exists_irreducible_fractions_sum_to_86_over_111_l151_151958

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l151_151958


namespace chalk_pieces_original_l151_151943

theorem chalk_pieces_original (
  siblings : ℕ := 3,
  friends : ℕ := 3,
  pieces_lost : ℕ := 2,
  pieces_added : ℕ := 12,
  pieces_needed_per_person : ℕ := 3
) : 
  ∃ (original_pieces : ℕ), 
    (original_pieces - pieces_lost + pieces_added) = 
    ((1 + siblings + friends) * pieces_needed_per_person) := 
  sorry

end chalk_pieces_original_l151_151943


namespace sum_of_distinct_products_of_6_23H_508_3G4_l151_151892

theorem sum_of_distinct_products_of_6_23H_508_3G4 (G H : ℕ) : 
  (G < 10) → (H < 10) →
  (623 * 1000 + H * 100 + 508 * 10 + 3 * 10 + G * 1 + 4) % 72 = 0 →
  (if G = 0 then 0 + if G = 4 then 4 else 0 else 0) = 4 :=
by
  intros
  sorry

end sum_of_distinct_products_of_6_23H_508_3G4_l151_151892


namespace find_difference_of_squares_l151_151428

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l151_151428


namespace total_steps_correct_l151_151599

/-- Definition of the initial number of steps on the first day --/
def steps_first_day : Nat := 200 + 300

/-- Definition of the number of steps on the second day --/
def steps_second_day : Nat := (3 / 2) * steps_first_day -- 1.5 is expressed as 3/2

/-- Definition of the number of steps on the third day --/
def steps_third_day : Nat := 2 * steps_second_day

/-- The total number of steps Eliana walked during the three days --/
def total_steps : Nat := steps_first_day + steps_second_day + steps_third_day

theorem total_steps_correct : total_steps = 2750 :=
  by
  -- provide the proof here
  sorry

end total_steps_correct_l151_151599


namespace sin_cos_identity_l151_151408

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end sin_cos_identity_l151_151408


namespace negation_of_real_root_proposition_l151_151873

theorem negation_of_real_root_proposition :
  (¬ ∃ m : ℝ, ∃ (x : ℝ), x^2 + m * x + 1 = 0) ↔ (∀ m : ℝ, ∀ (x : ℝ), x^2 + m * x + 1 ≠ 0) :=
by
  sorry

end negation_of_real_root_proposition_l151_151873


namespace profit_difference_l151_151261

variable (P : ℕ) -- P is the total profit
variable (r1 r2 : ℚ) -- r1 and r2 are the parts of the ratio for X and Y, respectively

noncomputable def X_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r1 / (r1 + r2)) * P

noncomputable def Y_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r2 / (r1 + r2)) * P

theorem profit_difference (P : ℕ) (r1 r2 : ℚ) (hP : P = 800) (hr1 : r1 = 1/2) (hr2 : r2 = 1/3) :
  X_share P r1 r2 - Y_share P r1 r2 = 160 := by
  sorry

end profit_difference_l151_151261


namespace necessary_and_sufficient_l151_151690

variable (α β : ℝ)
variable (p : Prop := α > β)
variable (q : Prop := α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α)

theorem necessary_and_sufficient : (p ↔ q) :=
by
  sorry

end necessary_and_sufficient_l151_151690


namespace original_cost_price_40_l151_151789

theorem original_cost_price_40
  (selling_price : ℝ)
  (decrease_rate : ℝ)
  (profit_increase_rate : ℝ)
  (new_selling_price := selling_price)
  (original_cost_price : ℝ)
  (new_cost_price := (1 - decrease_rate) * original_cost_price)
  (original_profit_margin := (selling_price - original_cost_price) / original_cost_price)
  (new_profit_margin := (new_selling_price - new_cost_price) / new_cost_price)
  (profit_margin_increase := profit_increase_rate)
  (h1 : selling_price = 48)
  (h2 : decrease_rate = 0.04)
  (h3 : profit_increase_rate = 0.05)
  (h4 : new_profit_margin = original_profit_margin + profit_margin_increase) :
  original_cost_price = 40 := 
by 
  sorry

end original_cost_price_40_l151_151789


namespace xiaochun_age_l151_151086

theorem xiaochun_age
  (x y : ℕ)
  (h1 : x = y - 18)
  (h2 : 2 * (x + 3) = y + 3) :
  x = 15 :=
sorry

end xiaochun_age_l151_151086


namespace smallest_k_l151_151179

theorem smallest_k (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * max m n + min m n - 1 ∧ 
  (∀ (persons : Finset ℕ),
    persons.card ≥ k →
    (∃ (acquainted : Finset (ℕ × ℕ)), acquainted.card = m ∧ 
      (∀ (x y : ℕ), (x, y) ∈ acquainted → (x ∈ persons ∧ y ∈ persons))) ∨
    (∃ (unacquainted : Finset (ℕ × ℕ)), unacquainted.card = n ∧ 
      (∀ (x y : ℕ), (x, y) ∈ unacquainted → (x ∈ persons ∧ y ∈ persons ∧ x ≠ y)))) :=
sorry

end smallest_k_l151_151179


namespace sum_positive_implies_at_least_one_positive_l151_151446

variables {a b : ℝ}

theorem sum_positive_implies_at_least_one_positive (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l151_151446


namespace constant_term_expansion_l151_151194

theorem constant_term_expansion (r : Nat) (h : 12 - 3 * r = 0) :
  (Nat.choose 6 r) * 2^r = 240 :=
sorry

end constant_term_expansion_l151_151194


namespace part1_area_quadrilateral_part2_maximized_line_equation_l151_151177

noncomputable def area_MA_NB (α : ℝ) : ℝ :=
  (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)

theorem part1_area_quadrilateral (α : ℝ) :
  area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2) :=
by sorry

theorem part2_maximized_line_equation :
  ∃ α : ℝ, area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)
    ∧ (Real.tan α = -1 / 2) ∧ (∀ x : ℝ, x = -1 / 2 * y + Real.sqrt 5 / 2) :=
by sorry

end part1_area_quadrilateral_part2_maximized_line_equation_l151_151177


namespace product_M1_M2_l151_151040

theorem product_M1_M2 :
  (∃ M1 M2 : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    (45 * x - 36) / (x^2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) ∧
    M1 * M2 = -222.75) :=
sorry

end product_M1_M2_l151_151040


namespace milk_left_l151_151045

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left_l151_151045


namespace king_plan_feasibility_l151_151914

-- Create a predicate for the feasibility of the king's plan
def feasible (n : ℕ) : Prop :=
  (n = 6 ∧ true) ∨ (n = 2004 ∧ false)

theorem king_plan_feasibility :
  ∀ n : ℕ, feasible n :=
by
  intro n
  sorry

end king_plan_feasibility_l151_151914


namespace Karl_selects_five_crayons_l151_151492

theorem Karl_selects_five_crayons : ∃ (k : ℕ), k = 3003 ∧ (finset.card (finset.powerset_len 5 (finset.range 15))).nat_abs = k :=
by
  -- existence proof of k = 3003 and showing that k equals the combination count
  sorry

end Karl_selects_five_crayons_l151_151492


namespace draw_white_ball_is_impossible_l151_151889

-- Definitions based on the conditions
def redBalls : Nat := 2
def blackBalls : Nat := 6
def totalBalls : Nat := redBalls + blackBalls

-- Definition for the white ball drawing event
def whiteBallDraw (redBalls blackBalls : Nat) : Prop :=
  ∀ (n : Nat), n ≠ 0 → n ≤ redBalls + blackBalls → false

-- Theorem to prove the event is impossible
theorem draw_white_ball_is_impossible : whiteBallDraw redBalls blackBalls :=
  by
  sorry

end draw_white_ball_is_impossible_l151_151889


namespace probability_of_receiving_1_l151_151990

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end probability_of_receiving_1_l151_151990


namespace total_pizzas_two_days_l151_151977

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end total_pizzas_two_days_l151_151977


namespace find_k_l151_151189

theorem find_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 :=
sorry

end find_k_l151_151189


namespace number_of_ways_to_distribute_balls_l151_151006

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l151_151006


namespace people_going_to_zoo_l151_151746

theorem people_going_to_zoo (buses people_per_bus total_people : ℕ) 
  (h1 : buses = 3) 
  (h2 : people_per_bus = 73) 
  (h3 : total_people = buses * people_per_bus) : 
  total_people = 219 := by
  rw [h1, h2] at h3
  exact h3

end people_going_to_zoo_l151_151746


namespace equation_of_line_l151_151682

-- Define the points P and Q
def P : (ℝ × ℝ) := (3, 2)
def Q : (ℝ × ℝ) := (4, 7)

-- Prove that the equation of the line passing through points P and Q is 5x - y - 13 = 0
theorem equation_of_line : ∃ (A B C : ℝ), A = 5 ∧ B = -1 ∧ C = -13 ∧
  ∀ x y : ℝ, (y - 2) / (7 - 2) = (x - 3) / (4 - 3) → 5 * x - y - 13 = 0 :=
by
  sorry

end equation_of_line_l151_151682


namespace marble_cut_in_third_week_l151_151800

def percentage_cut_third_week := 
  let initial_weight : ℝ := 250 
  let final_weight : ℝ := 105
  let percent_cut_first_week : ℝ := 0.30
  let percent_cut_second_week : ℝ := 0.20
  let weight_after_first_week := initial_weight * (1 - percent_cut_first_week)
  let weight_after_second_week := weight_after_first_week * (1 - percent_cut_second_week)
  (weight_after_second_week - final_weight) / weight_after_second_week * 100 = 25

theorem marble_cut_in_third_week :
  percentage_cut_third_week = true :=
by
  sorry

end marble_cut_in_third_week_l151_151800


namespace value_of_m_l151_151833

theorem value_of_m :
  ∃ m : ℕ, 3 * 4 * 5 * m = fact 8 ∧ m = 672 :=
by
  use 672
  split
  sorry

end value_of_m_l151_151833


namespace positive_difference_l151_151754

theorem positive_difference (y : ℤ) (h : (46 + y) / 2 = 52) : |y - 46| = 12 := by
  sorry

end positive_difference_l151_151754


namespace solid_brick_height_l151_151503

theorem solid_brick_height (n c base_perimeter height : ℕ) 
  (h1 : n = 42) 
  (h2 : c = 1) 
  (h3 : base_perimeter = 18)
  (h4 : n % base_area = 0)
  (h5 : 2 * (length + width) = base_perimeter)
  (h6 : base_area * height = n) : 
  height = 3 :=
by sorry

end solid_brick_height_l151_151503


namespace coin_change_count_ways_l151_151464

theorem coin_change_count_ways :
  ∃ n : ℕ, (∀ q h : ℕ, (25 * q + 50 * h = 1500) ∧ q > 0 ∧ h > 0 → (1 ≤ h ∧ h < 30)) ∧ n = 29 :=
  sorry

end coin_change_count_ways_l151_151464


namespace min_value_of_fraction_l151_151572

theorem min_value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 3) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_of_fraction_l151_151572


namespace man_finishes_work_in_100_days_l151_151255

variable (M W : ℝ)
variable (H1 : 10 * M * 6 + 15 * W * 6 = 1)
variable (H2 : W * 225 = 1)

theorem man_finishes_work_in_100_days (M W : ℝ) (H1 : 10 * M * 6 + 15 * W * 6 = 1) (H2 : W * 225 = 1) : M = 1 / 100 :=
by
  sorry

end man_finishes_work_in_100_days_l151_151255


namespace more_soccer_balls_than_basketballs_l151_151496

theorem more_soccer_balls_than_basketballs :
  let soccer_boxes := 8
  let basketball_boxes := 5
  let balls_per_box := 12
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end more_soccer_balls_than_basketballs_l151_151496


namespace value_of_a_over_b_l151_151279

def elements : List ℤ := [-5, -3, -1, 2, 4]

def maxProduct (l : List ℤ) : ℤ :=
  l.product $ prod for (x, y) in l.allPairs if x ≠ y and x * y

def minQuotient (l : List ℤ) : Rat :=
  l.allPairs $ min (x / y) for (x, y) in l.allPairs if x ≠ y and y ≠ 0

theorem value_of_a_over_b :
  let a := maxProduct elements
  let b := minQuotient elements
  a = 15 → b = -4 → a / b = -4
by
  intro a h_ma b h_mb
  have ha : a = 15 := h_ma
  have hb : b = -4 := h_mb
  rw [ha, hb]
  norm_num [ha, hb]
  sorry

end value_of_a_over_b_l151_151279


namespace penguins_count_l151_151226

theorem penguins_count (fish_total penguins_fed penguins_require : ℕ) (h1 : fish_total = 68) (h2 : penguins_fed = 19) (h3 : penguins_require = 17) : penguins_fed + penguins_require = 36 :=
by
  sorry

end penguins_count_l151_151226


namespace find_petra_age_l151_151082

namespace MathProof
  -- Definitions of the given conditions
  variables (P M : ℕ)
  axiom sum_of_ages : P + M = 47
  axiom mother_age_relation : M = 2 * P + 14
  axiom mother_actual_age : M = 36

  -- The proof goal which we need to fill later
  theorem find_petra_age : P = 11 :=
  by
    -- Using the axioms we have
    sorry -- Proof steps, which you don't need to fill according to the instructions
end MathProof

end find_petra_age_l151_151082


namespace price_reduction_correct_eqn_l151_151525

theorem price_reduction_correct_eqn (x : ℝ) :
  120 * (1 - x)^2 = 85 :=
sorry

end price_reduction_correct_eqn_l151_151525


namespace smallest_angle_in_icosagon_l151_151887

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l151_151887


namespace original_pencils_l151_151621

-- Define the conditions given in the problem
variable (total_pencils_now : ℕ) [DecidableEq ℕ] (pencils_by_Mike : ℕ)

-- State the problem to prove
theorem original_pencils (h1 : total_pencils_now = 71) (h2 : pencils_by_Mike = 30) : total_pencils_now - pencils_by_Mike = 41 := by
  sorry

end original_pencils_l151_151621


namespace smallest_nonneg_integer_l151_151939

theorem smallest_nonneg_integer (n : ℕ) (h : 0 ≤ n ∧ n < 53) :
  50 * n ≡ 47 [MOD 53] → n = 2 :=
by
  sorry

end smallest_nonneg_integer_l151_151939


namespace first_quadrant_solution_l151_151204

theorem first_quadrant_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ 0 < x ∧ 0 < y) ↔ -1 < c ∧ c < 3 / 2 :=
by
  sorry

end first_quadrant_solution_l151_151204


namespace find_x_in_terms_of_y_l151_151018

theorem find_x_in_terms_of_y 
(h₁ : x ≠ 0) 
(h₂ : x ≠ 3) 
(h₃ : y ≠ 0) 
(h₄ : y ≠ 5) 
(h_eq : 3 / x + 2 / y = 1 / 3) : 
x = 9 * y / (y - 6) :=
by
  sorry

end find_x_in_terms_of_y_l151_151018


namespace find_fractions_l151_151952

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l151_151952


namespace find_y_given_x_inverse_square_l151_151368

theorem find_y_given_x_inverse_square (x y : ℚ) : 
  (∀ k, (3 * y = k / x^2) ∧ (3 * 5 = k / 2^2)) → (x = 6) → y = 5 / 9 :=
by
  sorry

end find_y_given_x_inverse_square_l151_151368


namespace ellipse_a_value_l151_151404

theorem ellipse_a_value
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1)
  (e : ℝ)
  (h3 : e = 2 / 3)
  : a = 3 :=
by
  sorry

end ellipse_a_value_l151_151404


namespace clock_angle_3_to_7_l151_151791

theorem clock_angle_3_to_7 : 
  let number_of_rays := 12
  let total_degrees := 360
  let degree_per_ray := total_degrees / number_of_rays
  let angle_3_to_7 := 4 * degree_per_ray
  angle_3_to_7 = 120 :=
by
  sorry

end clock_angle_3_to_7_l151_151791


namespace andrew_total_homeless_shelter_donation_l151_151927

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l151_151927


namespace base_conversion_subtraction_l151_151290

namespace BaseConversion

def base9_to_base10 (n : ℕ) : ℕ :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def base6_to_base10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 6 * 6^0

theorem base_conversion_subtraction : (base9_to_base10 324) - (base6_to_base10 156) = 193 := by
  sorry

end BaseConversion

end base_conversion_subtraction_l151_151290


namespace smallest_solution_l151_151232

theorem smallest_solution (x : ℕ) (h1 : 6 * x ≡ 17 [MOD 31]) (h2 : x ≡ 3 [MOD 7]) : x = 24 := 
by 
  sorry

end smallest_solution_l151_151232


namespace tan_theta_sub_9pi_l151_151409

theorem tan_theta_sub_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -1 / 2) : 
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 :=
by
  sorry

end tan_theta_sub_9pi_l151_151409


namespace Horner_method_eval_l151_151504

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

theorem Horner_method_eval :
  let x := 10
  let v₀ := 3
  let v₁ := v₀ * x + 2
  v₁ = 32 :=
by
  let x := 10
  let v₀ := 3
  let v₁ := v₀ * x + 2
  show v₁ = 32
  sorry

end Horner_method_eval_l151_151504


namespace product_of_divisor_and_dividend_l151_151171

theorem product_of_divisor_and_dividend (d D : ℕ) (q : ℕ := 6) (r : ℕ := 3) 
  (h₁ : D = d + 78) 
  (h₂ : D = d * q + r) : 
  D * d = 1395 :=
by 
  sorry

end product_of_divisor_and_dividend_l151_151171


namespace max_cookies_andy_could_have_eaten_l151_151622

theorem max_cookies_andy_could_have_eaten (x k : ℕ) (hk : k > 0) 
  (h_total : x + k * x + 2 * x = 36) : x ≤ 9 :=
by
  -- Using the conditions to construct the proof (which is not required based on the instructions)
  sorry

end max_cookies_andy_could_have_eaten_l151_151622


namespace ratio_children_to_adults_l151_151930

variable (f m c : ℕ)

-- Conditions
def average_age_female (f : ℕ) := 35
def average_age_male (m : ℕ) := 30
def average_age_child (c : ℕ) := 10
def overall_average_age (f m c : ℕ) := 25

-- Total age sums based on given conditions
def total_age_sum_female (f : ℕ) := 35 * f
def total_age_sum_male (m : ℕ) := 30 * m
def total_age_sum_child (c : ℕ) := 10 * c

-- Total sum and average conditions
def total_age_sum (f m c : ℕ) := total_age_sum_female f + total_age_sum_male m + total_age_sum_child c
def total_members (f m c : ℕ) := f + m + c

theorem ratio_children_to_adults (f m c : ℕ) (h : (total_age_sum f m c) / (total_members f m c) = 25) :
  (c : ℚ) / (f + m) = 2 / 3 := sorry

end ratio_children_to_adults_l151_151930


namespace castor_chess_players_l151_151737

theorem castor_chess_players : 
  let total_players := 40 in
  let never_lost_to_ai := total_players / 4 in
  let lost_to_ai := total_players - never_lost_to_ai in
  lost_to_ai = 30 :=
by
  let total_players := 40
  let never_lost_to_ai := total_players / 4
  let lost_to_ai := total_players - never_lost_to_ai
  show lost_to_ai = 30
  exact sorry

end castor_chess_players_l151_151737


namespace find_row_with_sum_2013_squared_l151_151732

-- Define the sum of the numbers in the nth row
def sum_of_row (n : ℕ) : ℕ := (2 * n - 1)^2

theorem find_row_with_sum_2013_squared : (∃ n : ℕ, sum_of_row n = 2013^2) ∧ (sum_of_row 1007 = 2013^2) :=
by
  sorry

end find_row_with_sum_2013_squared_l151_151732


namespace inequality_proof_l151_151440

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1 / a < 1 / b) :=
sorry

end inequality_proof_l151_151440


namespace range_of_a_l151_151002

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 1) → a ≤ 0 :=
by
  sorry

end range_of_a_l151_151002


namespace brad_red_balloons_l151_151673

theorem brad_red_balloons (total balloons green : ℕ) (h1 : total = 17) (h2 : green = 9) : total - green = 8 := 
by {
  sorry
}

end brad_red_balloons_l151_151673


namespace emberly_walks_miles_in_march_l151_151144

theorem emberly_walks_miles_in_march (days_not_walked : ℕ) (total_days_in_march : ℕ) (miles_per_hour : ℕ) (hours_per_walk : ℕ) :
  total_days_in_march = 31 → 
  days_not_walked = 4 → 
  hours_per_walk = 1 → 
  miles_per_hour = 4 → 
  let days_walked := total_days_in_march - days_not_walked in
  let total_hours_walked := days_walked * hours_per_walk in
  let total_miles_walked := total_hours_walked * miles_per_hour in
  total_miles_walked = 108 :=
by {
  intros h1 h2 h3 h4,
  have days_walked_def : days_walked = 31 - 4 := by rw [h1, h2],
  have hours_walked_def : total_hours_walked = (31 - 4) * 1 := by rw [days_walked_def, h3],
  have miles_walked_def : total_miles_walked = (27) * 4 := by rw [hours_walked_def, h4],
  have final_result : total_miles_walked = 108 := by rw miles_walked_def,
  exact final_result,
}

end emberly_walks_miles_in_march_l151_151144


namespace sequence_sum_a1_a3_l151_151895

theorem sequence_sum_a1_a3 (S : ℕ → ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → S n + S (n - 1) = 2 * n - 1) 
  (h2 : S 2 = 3) : 
  a 1 + a 3 = -1 := by
  sorry

end sequence_sum_a1_a3_l151_151895


namespace max_prime_factors_of_c_l151_151482

-- Definitions of conditions
variables (c d : ℕ)
variable (prime_factor_count : ℕ → ℕ)
variable (gcd : ℕ → ℕ → ℕ)
variable (lcm : ℕ → ℕ → ℕ)

-- Conditions
axiom gcd_condition : prime_factor_count (gcd c d) = 11
axiom lcm_condition : prime_factor_count (lcm c d) = 44
axiom fewer_prime_factors : prime_factor_count c < prime_factor_count d

-- Proof statement
theorem max_prime_factors_of_c : prime_factor_count c ≤ 27 := 
sorry

end max_prime_factors_of_c_l151_151482


namespace lunch_break_duration_l151_151053

/-- Paula and her two helpers start at 7:00 AM and paint 60% of a house together,
    finishing at 5:00 PM. The next day, only the helpers paint and manage to
    paint 30% of another house, finishing at 3:00 PM. On the third day, Paula
    paints alone and paints the remaining 40% of the house, finishing at 4:00 PM.
    Prove that the length of their lunch break each day is 1 hour (60 minutes). -/
theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (0 < L) ∧ 
    (L < 10) ∧
    (∃ (p h : ℝ), 
       (10 - L) * (p + h) = 0.6 ∧
       (8 - L) * h = 0.3 ∧
       (9 - L) * p = 0.4) ∧  
    L = 1 :=
by
  sorry

end lunch_break_duration_l151_151053


namespace sum_of_seven_consecutive_integers_l151_151608

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l151_151608


namespace total_golf_balls_l151_151755

theorem total_golf_balls :
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  dan + gus + chris = 132 :=
by
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  sorry

end total_golf_balls_l151_151755


namespace exists_quadratic_satisfying_conditions_l151_151941

theorem exists_quadratic_satisfying_conditions :
  ∃ (a b c : ℝ), 
  (a - b + c = 0) ∧
  (∀ x : ℝ, x ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧ 
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
  sorry

end exists_quadratic_satisfying_conditions_l151_151941


namespace partition_exists_iff_l151_151296

theorem partition_exists_iff (k : ℕ) :
  (∃ (A B : Finset ℕ), A ∪ B = Finset.range (1990 + k + 1) ∧ A ∩ B = ∅ ∧ 
  (A.sum id + 1990 * A.card = B.sum id + 1990 * B.card)) ↔ 
  (k % 4 = 3 ∨ (k % 4 = 0 ∧ k ≥ 92)) :=
by
  sorry

end partition_exists_iff_l151_151296


namespace range_quadratic_function_l151_151221

theorem range_quadratic_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = x^2 - 2 * x + 5 ↔ y ∈ Set.Ici 4 :=
by 
  sorry

end range_quadratic_function_l151_151221


namespace remainder_of_144_div_k_l151_151831

theorem remainder_of_144_div_k
  (k : ℕ)
  (h1 : 0 < k)
  (h2 : 120 % k^2 = 12) :
  144 % k = 0 :=
by
  sorry

end remainder_of_144_div_k_l151_151831


namespace lg_sum_geometric_seq_l151_151450

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem lg_sum_geometric_seq (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 5 * a 8 = 1) :
  Real.log (a 4) + Real.log (a 6) = 0 := 
sorry

end lg_sum_geometric_seq_l151_151450


namespace probability_of_receiving_one_l151_151991

noncomputable def probability_received_one : ℝ :=
let P_A := 0.5 in
let P_not_A := 0.5 in
let P_B_given_A := 0.9 in
let P_not_B_given_A := 0.1 in
let P_B_given_not_A := 0.05 in
let P_not_B_given_not_A := 0.95 in
let P_B := P_A * P_B_given_A + P_not_A * P_B_given_not_A in
1 - P_B

theorem probability_of_receiving_one :
  probability_received_one = 0.525 :=
by
  -- P_A = 0.5
  -- P_not_A = 0.5
  -- P_B_given_A = 0.9
  -- P_not_B_given_A = 0.1
  -- P_B_given_not_A = 0.05
  -- P_not_B_given_not_A = 0.95
  -- P_B = 0.5 * 0.9 + 0.5 * 0.05
  -- P_B = 0.45 + 0.025
  -- P_B = 0.475
  -- P_not_B = 1 - P_B
  -- P_not_B = 1 - 0.475
  -- P_not_B = 0.525
  sorry

end probability_of_receiving_one_l151_151991


namespace distribute_balls_into_boxes_l151_151010

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l151_151010


namespace lowest_degree_polynomial_is_4_l151_151508

noncomputable def lowest_degree_polynomial : ℕ :=
  let exists_polynomial_of_degree_four_with_conditions := ∃ (P : Polynomial ℤ), P.degree = 4 ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  let no_polynomial_of_degree_less_than_four_with_conditions := ∀ (d < 4), ¬∃ (P : Polynomial ℤ), P.degree = d ∧
    ∃ b : ℤ, ∀ coeff ∈ P.coeffs, (coeff < b ∨ coeff > b) ∧ coeff ≠ b
  if h₁ : exists_polynomial_of_degree_four_with_conditions ∧ no_polynomial_of_degree_less_than_four_with_conditions then 4 else 0

theorem lowest_degree_polynomial_is_4 :
  lowest_degree_polynomial = 4 :=
by
  sorry

end lowest_degree_polynomial_is_4_l151_151508


namespace lower_denomination_cost_l151_151924

-- Conditions
def total_stamps : ℕ := 20
def total_cost_cents : ℕ := 706
def high_denomination_stamps : ℕ := 18
def high_denomination_cost : ℕ := 37
def low_denomination_stamps : ℕ := total_stamps - high_denomination_stamps

-- Theorem proving the cost of the lower denomination stamp.
theorem lower_denomination_cost :
  ∃ (x : ℕ), (high_denomination_stamps * high_denomination_cost) + (low_denomination_stamps * x) = total_cost_cents
  ∧ x = 20 :=
by
  use 20
  sorry

end lower_denomination_cost_l151_151924


namespace shinyoung_initial_candies_l151_151055

theorem shinyoung_initial_candies : 
  ∀ (C : ℕ), 
    (C / 2) - ((C / 6) + 5) = 5 → 
    C = 30 := by
  intros C h
  sorry

end shinyoung_initial_candies_l151_151055


namespace find_common_difference_l151_151763

variable {α : Type*} [LinearOrderedField α]

-- Define the properties of the arithmetic sequence
def arithmetic_sum (a1 d : α) (n : ℕ) : α := n * a1 + (n * (n - 1) * d) / 2

variables (a1 d : α) -- First term and common difference of the arithmetic sequence (to be found)
variable (S : ℕ → α) -- Sum of the first n terms of the arithmetic sequence

-- Conditions given in the problem
axiom sum_3_eq_6 : S 3 = 6
axiom term_3_eq_4 : a1 + 2 * d = 4

-- The question translated into a theorem statement that the common difference is 2
theorem find_common_difference : d = 2 :=
by
  sorry

end find_common_difference_l151_151763


namespace face_value_of_share_l151_151259

-- Let FV be the face value of each share.
-- Given conditions:
-- Dividend rate is 9%
-- Market value of each share is Rs. 42
-- Desired interest rate is 12%

theorem face_value_of_share (market_value : ℝ) (dividend_rate : ℝ) (interest_rate : ℝ) (FV : ℝ) :
  market_value = 42 ∧ dividend_rate = 0.09 ∧ interest_rate = 0.12 →
  0.09 * FV = 0.12 * market_value →
  FV = 56 :=
by
  sorry

end face_value_of_share_l151_151259


namespace intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l151_151641

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem intervals_of_increase_decrease_a_neg1 : 
  ∀ x : ℝ, quadratic_function (-1) x = x^2 - 2 * x + 3 → 
  (∀ x ≥ 1, quadratic_function (-1) x ≥ quadratic_function (-1) 1) ∧ 
  (∀ x ≤ 1, quadratic_function (-1) x ≤ quadratic_function (-1) 1) :=
  sorry

theorem max_min_values_a_neg2 :
  ∃ min : ℝ, min = -1 ∧ (∀ x : ℝ, quadratic_function (-2) x ≥ min) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y > x → quadratic_function (-2) y > quadratic_function (-2) x) :=
  sorry

theorem no_a_for_monotonic_function : 
  ∀ a : ℝ, ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≤ quadratic_function a y) ∧ ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≥ quadratic_function a y) :=
  sorry

end intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l151_151641


namespace umar_age_is_ten_l151_151266

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l151_151266


namespace castor_chess_players_l151_151736

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end castor_chess_players_l151_151736


namespace benny_money_l151_151933

-- Conditions
def cost_per_apple (cost : ℕ) := cost = 4
def apples_needed (apples : ℕ) := apples = 5 * 18

-- The proof problem
theorem benny_money (cost : ℕ) (apples : ℕ) (total_money : ℕ) :
  cost_per_apple cost → apples_needed apples → total_money = apples * cost → total_money = 360 :=
by
  intros h_cost h_apples h_total
  rw [h_cost, h_apples] at h_total
  exact h_total

end benny_money_l151_151933


namespace anne_initial_sweettarts_l151_151270

variable (x : ℕ)
variable (num_friends : ℕ := 3)
variable (sweettarts_per_friend : ℕ := 5)
variable (total_sweettarts_given : ℕ := num_friends * sweettarts_per_friend)

theorem anne_initial_sweettarts 
  (h1 : ∀ person, person < num_friends → sweettarts_per_friend = 5)
  (h2 : total_sweettarts_given = 15) : 
  total_sweettarts_given = 15 := 
by 
  sorry

end anne_initial_sweettarts_l151_151270


namespace simplify_fraction_l151_151744

theorem simplify_fraction (a b : ℕ) (h : a = 150) (hb : b = 450) : a / b = 1 / 3 := by
  sorry

end simplify_fraction_l151_151744


namespace trigonometric_identity_simplification_l151_151741

theorem trigonometric_identity_simplification :
  (Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + Real.cos (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1) :=
by sorry

end trigonometric_identity_simplification_l151_151741


namespace three_digit_integer_condition_l151_151151

theorem three_digit_integer_condition (n a b c : ℕ) (hn : 100 ≤ n ∧ n < 1000)
  (hdigits : n = 100 * a + 10 * b + c)
  (hdadigits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (fact_condition : 2 * n / 3 = a.factorial * b.factorial * c.factorial) :
  n = 432 := sorry

end three_digit_integer_condition_l151_151151


namespace ratio_of_couch_to_table_l151_151722

theorem ratio_of_couch_to_table
    (C T X : ℝ)
    (h1 : T = 3 * C)
    (h2 : X = 300)
    (h3 : C + T + X = 380) :
  X / T = 5 := 
by 
  sorry

end ratio_of_couch_to_table_l151_151722


namespace find_N_l151_151902

def f (N : ℕ) : ℕ :=
  if N % 2 = 0 then 5 * N else 3 * N + 2

theorem find_N (N : ℕ) :
  f (f (f (f (f N)))) = 542 ↔ N = 112500 := by
  sorry

end find_N_l151_151902


namespace evenFunctionExists_l151_151613

-- Definitions based on conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def passesThroughPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = p.2

-- Example function
def exampleEvenFunction (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

-- Points to pass through
def givenPoints : List (ℝ × ℝ) := [(-1, 0), (0.5, 2.5), (3, 0)]

-- Theorem to be proven
theorem evenFunctionExists : 
  isEvenFunction exampleEvenFunction ∧ passesThroughPoints exampleEvenFunction givenPoints :=
by
  sorry

end evenFunctionExists_l151_151613


namespace find_original_number_l151_151036

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l151_151036


namespace perfect_square_trinomial_l151_151853

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, a^2 = 1 ∧ b^2 = 1 ∧ x^2 + m * x * y + y^2 = (a * x + b * y)^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l151_151853


namespace solution_to_fractional_equation_l151_151355

theorem solution_to_fractional_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 1) :
  (x / (x - 3) = (x + 1) / (x - 1)) ↔ (x = -3) :=
by
  sorry

end solution_to_fractional_equation_l151_151355


namespace johns_donation_l151_151576

theorem johns_donation (A : ℝ) (T : ℝ) (J : ℝ) (h1 : A + 0.5 * A = 75) (h2 : T = 3 * A) 
                       (h3 : (T + J) / 4 = 75) : J = 150 := by
  sorry

end johns_donation_l151_151576


namespace log_fraction_identity_l151_151439

theorem log_fraction_identity (a b : ℝ) (h2 : Real.log 2 = a) (h3 : Real.log 3 = b) :
  (Real.log 12 / Real.log 15) = (2 * a + b) / (1 - a + b) := 
  sorry

end log_fraction_identity_l151_151439


namespace square_diff_l151_151434

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l151_151434


namespace max_product_min_quotient_l151_151278

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end max_product_min_quotient_l151_151278


namespace average_age_of_9_l151_151215

theorem average_age_of_9 : 
  ∀ (avg_20 avg_5 age_15 : ℝ),
  avg_20 = 15 →
  avg_5 = 14 →
  age_15 = 86 →
  (9 * (69/9)) = 7.67 :=
by
  intros avg_20 avg_5 age_15 avg_20_val avg_5_val age_15_val
  -- The proof is skipped
  sorry

end average_age_of_9_l151_151215


namespace rope_length_l151_151277

theorem rope_length (x : ℝ) 
  (h : 10^2 + (x - 4)^2 = x^2) : 
  x = 14.5 :=
sorry

end rope_length_l151_151277


namespace find_angle_A_find_b_c_l151_151975
open Real

-- Part I: Proving angle A
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h₁ : (a + b + c) * (b + c - a) = 3 * b * c) :
  A = π / 3 :=
by sorry

-- Part II: Proving values of b and c given a=2 and area of triangle ABC is √3
theorem find_b_c (A B C : ℝ) (a b c : ℝ) (h₁ : a = 2) (h₂ : (1 / 2) * b * c * (sin (π / 3)) = sqrt 3) :
  b = 2 ∧ c = 2 :=
by sorry

end find_angle_A_find_b_c_l151_151975


namespace smallest_angle_in_convex_20_gon_seq_l151_151882

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l151_151882


namespace no_real_solutions_eqn_l151_151181

theorem no_real_solutions_eqn : ∀ x : ℝ, (2 * x - 4 * x + 7)^2 + 1 ≠ -|x^2 - 1| :=
by
  intro x
  sorry

end no_real_solutions_eqn_l151_151181


namespace trajectory_eq_l151_151520

theorem trajectory_eq (M : Type) [MetricSpace M] : 
  (∀ (r x y : ℝ), (x + 2)^2 + y^2 = (r + 1)^2 ∧ |x - 1| = 1 → y^2 = -8 * x) :=
by sorry

end trajectory_eq_l151_151520


namespace initial_cupcakes_baked_l151_151720

variable (toddAte := 21)       -- Todd ate 21 cupcakes.
variable (packages := 6)       -- She could make 6 packages.
variable (cupcakesPerPackage := 3) -- Each package contains 3 cupcakes.
variable (cupcakesLeft := packages * cupcakesPerPackage) -- Cupcakes left after Todd ate some.

theorem initial_cupcakes_baked : cupcakesLeft + toddAte = 39 :=
by
  -- Proof placeholder
  sorry

end initial_cupcakes_baked_l151_151720


namespace range_of_a_l151_151406

variables (m a x y : ℝ)

def p (m a : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0

def ellipse (m x y : ℝ) : Prop := (x^2)/(m-1) + (y^2)/(2-m) = 1

def q (m : ℝ) (x y : ℝ) : Prop := ellipse m x y ∧ 1 < m ∧ m < 3/2

theorem range_of_a :
  (∃ m, p m a → (∀ x y, q m x y)) → (1/3 ≤ a ∧ a ≤ 3/8) :=
sorry

end range_of_a_l151_151406


namespace total_respondents_l151_151733

theorem total_respondents (X Y : ℕ) 
  (hX : X = 60) 
  (hRatio : 3 * Y = X) : 
  X + Y = 80 := 
by
  sorry

end total_respondents_l151_151733


namespace solution_pairs_l151_151677

open Int

theorem solution_pairs (a b : ℝ) (h : ∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int) :=
by sorry

end solution_pairs_l151_151677


namespace largest_number_is_B_l151_151364
open Real

noncomputable def A := 0.989
noncomputable def B := 0.998
noncomputable def C := 0.899
noncomputable def D := 0.9899
noncomputable def E := 0.8999

theorem largest_number_is_B :
  B = max (max (max (max A B) C) D) E :=
by
  sorry

end largest_number_is_B_l151_151364


namespace Trevor_tip_l151_151767

variable (Uber Lyft Taxi : ℕ)
variable (TotalCost : ℕ)

theorem Trevor_tip 
  (h1 : Uber = Lyft + 3) 
  (h2 : Lyft = Taxi + 4) 
  (h3 : Uber = 22) 
  (h4 : TotalCost = 18)
  (h5 : Taxi = 15) :
  (TotalCost - Taxi) * 100 / Taxi = 20 := by
  sorry

end Trevor_tip_l151_151767


namespace seq_bounded_l151_151546

def digit_product (n : ℕ) : ℕ :=
  n.digits 10 |>.prod

def a_seq (a : ℕ → ℕ) (m : ℕ) : Prop :=
  a 0 = m ∧ (∀ n, a (n + 1) = a n + digit_product (a n))

theorem seq_bounded (m : ℕ) : ∃ B, ∀ n, a_seq a m → a n < B :=
by sorry

end seq_bounded_l151_151546


namespace at_least_eight_composites_l151_151303

theorem at_least_eight_composites (n : ℕ) (h : n > 1000) :
  ∃ (comps : Finset ℕ), 
    comps.card ≥ 8 ∧ 
    (∀ x ∈ comps, ¬Prime x) ∧ 
    (∀ k, k < 12 → n + k ∈ comps ∨ Prime (n + k)) :=
by
  sorry

end at_least_eight_composites_l151_151303


namespace decreasing_function_range_of_a_l151_151561

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem decreasing_function_range_of_a :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1/7 ≤ a ∧ a < 1/3) :=
by
  sorry

end decreasing_function_range_of_a_l151_151561


namespace turtle_distance_in_six_minutes_l151_151122

theorem turtle_distance_in_six_minutes 
  (observers : ℕ)
  (time_interval : ℕ)
  (distance_seen : ℕ)
  (total_time : ℕ)
  (total_distance : ℕ)
  (observation_per_minute : ∀ t ≤ total_time, ∃ n : ℕ, n ≤ observers ∧ (∃ interval : ℕ, interval ≤ time_interval ∧ distance_seen = 1)) :
  total_distance = 10 :=
sorry

end turtle_distance_in_six_minutes_l151_151122


namespace compare_abc_l151_151696

noncomputable def a := Real.sqrt 0.3
noncomputable def b := Real.sqrt 0.4
noncomputable def c := Real.log 0.6 / Real.log 3

theorem compare_abc : c < a ∧ a < b :=
by
  -- Proof goes here
  sorry

end compare_abc_l151_151696


namespace chord_bisected_by_point_of_ellipse_l151_151170

theorem chord_bisected_by_point_of_ellipse 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1)
  (bisecting_point : ∃ x y : ℝ, x = 4 ∧ y = 2) :
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -8 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
   sorry

end chord_bisected_by_point_of_ellipse_l151_151170


namespace tile_in_center_l151_151056

-- Define the coloring pattern of the grid
inductive Color
| A | B | C

-- Predicates for grid, tile placement, and colors
def Grid := Fin 5 × Fin 5

def is_1x3_tile (t : Grid × Grid × Grid) : Prop :=
  -- Ensure each tuple t represents three cells that form a $1 \times 3$ tile
  sorry

def is_tiling (g : Grid → Option Color) : Prop :=
  -- Ensure the entire grid is correctly tiled with the given tiles and within the coloring pattern
  sorry

def center : Grid := (Fin.mk 2 (by decide), Fin.mk 2 (by decide))

-- The theorem statement
theorem tile_in_center (g : Grid → Option Color) : is_tiling g → 
  (∃! tile : Grid, g tile = some Color.B) :=
sorry

end tile_in_center_l151_151056


namespace find_two_irreducible_fractions_l151_151953

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l151_151953


namespace problem_1_solution_problem_2_solution_l151_151447

variables (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_ball : ℕ)

def probability_of_red_or_black_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (red_balls + black_balls : ℚ) / total_balls

def probability_of_at_least_one_red_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (((red_balls * (total_balls - red_balls)) + ((red_balls * (red_balls - 1)) / 2)) : ℚ)
  / ((total_balls * (total_balls - 1) / 2) : ℚ)

theorem problem_1_solution :
  probability_of_red_or_black_ball 12 5 4 2 1 = 3 / 4 :=
by
  sorry

theorem problem_2_solution :
  probability_of_at_least_one_red_ball 12 5 4 2 1 = 15 / 22 :=
by
  sorry

end problem_1_solution_problem_2_solution_l151_151447


namespace possible_N_l151_151039

/-- 
  Let N be an integer with N ≥ 3, and let a₀, a₁, ..., a_(N-1) be pairwise distinct reals such that 
  aᵢ ≥ a_(2i mod N) for all i. Prove that N must be a power of 2.
-/
theorem possible_N (N : ℕ) (hN : N ≥ 3) (a : Fin N → ℝ) (h_distinct: Function.Injective a) 
  (h_condition : ∀ i : Fin N, a i ≥ a (⟨(2 * i) % N, sorry⟩)) 
  : ∃ k : ℕ, N = 2^k := 
sorry

end possible_N_l151_151039


namespace value_of_a_minus_b_l151_151184

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 :=
sorry

end value_of_a_minus_b_l151_151184


namespace find_x_y_l151_151442

theorem find_x_y (x y : ℤ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 :=
by
  sorry

end find_x_y_l151_151442


namespace emily_points_l151_151145

theorem emily_points (r1 r2 r3 r4 r5 m4 m5 l : ℤ)
  (h1 : r1 = 16)
  (h2 : r2 = 33)
  (h3 : r3 = 21)
  (h4 : r4 = 10)
  (h5 : r5 = 4)
  (hm4 : m4 = 2)
  (hm5 : m5 = 3)
  (hl : l = 48) :
  r1 + r2 + r3 + r4 * m4 + r5 * m5 - l = 54 := by
  sorry

end emily_points_l151_151145


namespace cristina_running_pace_l151_151046

theorem cristina_running_pace
  (nicky_pace : ℝ) (nicky_headstart : ℝ) (time_nicky_run : ℝ) 
  (distance_nicky_run : ℝ) (time_cristina_catch : ℝ) :
  (nicky_pace = 3) →
  (nicky_headstart = 12) →
  (time_nicky_run = 30) →
  (distance_nicky_run = nicky_pace * time_nicky_run) →
  (time_cristina_catch = time_nicky_run - nicky_headstart) →
  (cristina_pace : ℝ) →
  (cristina_pace = distance_nicky_run / time_cristina_catch) →
  cristina_pace = 5 :=
by
  sorry

end cristina_running_pace_l151_151046


namespace solution_l151_151964

-- Definitions for perpendicular and parallel relations
def perpendicular (a b : Type) : Prop := sorry -- Abstraction for perpendicularity
def parallel (a b : Type) : Prop := sorry -- Abstraction for parallelism

-- Here we define x, y, z as variables
variables {x y : Type} {z : Type}

-- Conditions for Case 2
def case2_lines_plane (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Conditions for Case 3
def case3_planes_line (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Theorem statement combining both cases
theorem solution : case2_lines_plane x y z ∧ case3_planes_line x y z := 
sorry

end solution_l151_151964


namespace abe_age_equation_l151_151081

theorem abe_age_equation (a : ℕ) (x : ℕ) (h1 : a = 19) (h2 : a + (a - x) = 31) : x = 7 :=
by
  sorry

end abe_age_equation_l151_151081


namespace cosine_angle_is_zero_l151_151352

-- Define the structure of an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  angle_60_deg : Prop

-- Define the structure of a parallelogram built from 6 equilateral triangles
structure Parallelogram where
  composed_of_6_equilateral_triangles : Prop
  folds_into_hexahedral_shape : Prop

-- Define the angle and its cosine computation between two specific directions in the folded hexahedral shape
def cosine_of_angle_between_AB_and_CD (parallelogram : Parallelogram) : ℝ := sorry

-- The condition that needs to be proved
axiom parallelogram_conditions : Parallelogram
axiom cosine_angle_proof : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0

-- Final proof statement
theorem cosine_angle_is_zero : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0 :=
cosine_angle_proof

end cosine_angle_is_zero_l151_151352


namespace generic_packages_needed_eq_2_l151_151272

-- Define parameters
def tees_per_generic_package : ℕ := 12
def tees_per_aero_package : ℕ := 2
def members_foursome : ℕ := 4
def tees_needed_per_member : ℕ := 20
def aero_packages_purchased : ℕ := 28

-- Calculate total tees needed and total tees obtained from aero packages
def total_tees_needed : ℕ := members_foursome * tees_needed_per_member
def aero_tees_obtained : ℕ := aero_packages_purchased * tees_per_aero_package
def generic_tees_needed : ℕ := total_tees_needed - aero_tees_obtained

-- Prove the number of generic packages needed is 2
theorem generic_packages_needed_eq_2 : 
  generic_tees_needed / tees_per_generic_package = 2 :=
  sorry

end generic_packages_needed_eq_2_l151_151272


namespace integer_not_in_range_l151_151463

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l151_151463


namespace umar_age_is_10_l151_151267

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l151_151267


namespace not_odd_iff_exists_ne_l151_151590

open Function

variable {f : ℝ → ℝ}

theorem not_odd_iff_exists_ne : (∃ x : ℝ, f (-x) ≠ -f x) ↔ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end not_odd_iff_exists_ne_l151_151590


namespace determine_y_l151_151320

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end determine_y_l151_151320


namespace base_eight_to_base_ten_l151_151234

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l151_151234


namespace train_speed_kmph_l151_151805

/-- Given that the length of the train is 200 meters and it crosses a pole in 9 seconds,
the speed of the train in km/hr is 80. -/
theorem train_speed_kmph (length : ℝ) (time : ℝ) (length_eq : length = 200) (time_eq : time = 9) : 
  (length / time) * (3600 / 1000) = 80 :=
by
  sorry

end train_speed_kmph_l151_151805


namespace kanul_initial_amount_l151_151330

noncomputable def initial_amount : ℝ :=
  (5000 : ℝ) + 200 + 1200 + (11058.82 : ℝ) * 0.15 + 3000

theorem kanul_initial_amount (X : ℝ) 
  (raw_materials : ℝ := 5000) 
  (machinery : ℝ := 200) 
  (employee_wages : ℝ := 1200) 
  (maintenance_cost : ℝ := 0.15 * X)
  (remaining_balance : ℝ := 3000) 
  (expenses : ℝ := raw_materials + machinery + employee_wages + maintenance_cost) 
  (total_expenses : ℝ := expenses + remaining_balance) :
  X = total_expenses :=
by sorry

end kanul_initial_amount_l151_151330


namespace compare_abc_l151_151982

theorem compare_abc :
  let a := Real.log 17
  let b := 3
  let c := Real.exp (Real.sqrt 2)
  a < b ∧ b < c :=
by
  sorry

end compare_abc_l151_151982


namespace speed_of_A_is_3_l151_151967

theorem speed_of_A_is_3:
  (∃ x : ℝ, 3 * x + 3 * (x + 2) = 24) → x = 3 :=
by
  sorry

end speed_of_A_is_3_l151_151967


namespace arithmetic_sequence_minimization_l151_151998

theorem arithmetic_sequence_minimization (a b : ℕ) (h_range : 1 ≤ a ∧ b ≤ 17) (h_seq : a + b = 18) (h_min : ∀ x y, (1 ≤ x ∧ y ≤ 17 ∧ x + y = 18) → (1 / x + 25 / y) ≥ (1 / a + 25 / b)) : ∃ n : ℕ, n = 9 :=
by
  -- We'd usually follow by proving the conditions and defining the sequence correctly.
  -- Definitions and steps leading to finding n = 9 will be elaborated here.
  -- This placeholder is to satisfy the requirement only.
  sorry

end arithmetic_sequence_minimization_l151_151998


namespace rational_solutions_product_l151_151083

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem rational_solutions_product :
  ∀ c : ℕ, (c > 0) → (is_perfect_square (49 - 12 * c)) → (∃ a b : ℕ, a = 4 ∧ b = 2 ∧ a * b = 8) :=
by sorry

end rational_solutions_product_l151_151083


namespace system_of_equations_solutions_l151_151416

theorem system_of_equations_solutions (x y a b : ℝ) 
  (h1 : 2 * x + y = b) 
  (h2 : x - b * y = a) 
  (hx : x = 1)
  (hy : y = 0) : a - b = -1 :=
by 
  sorry

end system_of_equations_solutions_l151_151416


namespace compare_f_values_max_f_value_l151_151969

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem compare_f_values :
  f (Real.pi / 4) > f (Real.pi / 6) :=
sorry

theorem max_f_value :
  ∃ x : ℝ, f x = 3 :=
sorry

end compare_f_values_max_f_value_l151_151969


namespace sequence_starting_point_l151_151766

theorem sequence_starting_point
  (n : ℕ) 
  (k : ℕ) 
  (h₁ : n * 9 ≤ 100000)
  (h₂ : k = 11110)
  (h₃ : 9 * (n + k - 1) = 99999) : 
  9 * n = 88890 :=
by 
  sorry

end sequence_starting_point_l151_151766


namespace sqrt_difference_l151_151102

theorem sqrt_difference:
  sqrt (49 + 81) - sqrt (36 - 9) = sqrt 130 - 3 * sqrt 3 :=
by
  sorry

end sqrt_difference_l151_151102


namespace sin_theta_plus_2cos_theta_eq_zero_l151_151410

theorem sin_theta_plus_2cos_theta_eq_zero (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (1 + Real.sin (2 * θ)) / (Real.cos θ)^2 = 1 :=
  sorry

end sin_theta_plus_2cos_theta_eq_zero_l151_151410


namespace number_of_men_in_first_group_l151_151373

-- Define the conditions
def condition1 (M : ℕ) : Prop := M * 80 = 20 * 40

-- State the main theorem to be proved
theorem number_of_men_in_first_group (M : ℕ) (h : condition1 M) : M = 10 := by
  sorry

end number_of_men_in_first_group_l151_151373


namespace initial_black_pens_correct_l151_151375

-- Define the conditions
def initial_blue_pens : ℕ := 9
def removed_blue_pens : ℕ := 4
def remaining_blue_pens : ℕ := initial_blue_pens - removed_blue_pens

def initial_red_pens : ℕ := 6
def removed_red_pens : ℕ := 0
def remaining_red_pens : ℕ := initial_red_pens - removed_red_pens

def total_remaining_pens : ℕ := 25
def removed_black_pens : ℕ := 7

-- Assume B is the initial number of black pens
def B : ℕ := 21

-- Prove the initial number of black pens condition
theorem initial_black_pens_correct : 
  (initial_blue_pens + B + initial_red_pens) - (removed_blue_pens + removed_black_pens) = total_remaining_pens :=
by 
  have h1 : initial_blue_pens - removed_blue_pens = remaining_blue_pens := rfl
  have h2 : initial_red_pens - removed_red_pens = remaining_red_pens := rfl
  have h3 : remaining_blue_pens + (B - removed_black_pens) + remaining_red_pens = total_remaining_pens := sorry
  exact h3

end initial_black_pens_correct_l151_151375


namespace complement_union_eq_l151_151178

variable (U : Set ℝ) (M N : Set ℝ)

noncomputable def complement_union (U M N : Set ℝ) : Set ℝ :=
  U \ (M ∪ N)

theorem complement_union_eq :
  U = Set.univ → 
  M = {x | |x| < 1} → 
  N = {y | ∃ x, y = 2^x} → 
  complement_union U M N = {x | x ≤ -1} :=
by
  intros hU hM hN
  unfold complement_union
  sorry

end complement_union_eq_l151_151178


namespace domain_of_function_l151_151350

theorem domain_of_function :
  {x : ℝ | 3 - x > 0 ∧ x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end domain_of_function_l151_151350


namespace each_friend_gets_four_pieces_l151_151863

noncomputable def pieces_per_friend : ℕ :=
  let oranges := 80
  let pieces_per_orange := 10
  let friends := 200
  (oranges * pieces_per_orange) / friends

theorem each_friend_gets_four_pieces :
  pieces_per_friend = 4 :=
by
  sorry

end each_friend_gets_four_pieces_l151_151863


namespace Jill_tax_on_clothing_l151_151735

theorem Jill_tax_on_clothing 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ) (total_spent : ℝ) (tax_clothing : ℝ) 
  (tax_other_rate : ℝ) (total_tax_rate : ℝ) 
  (h_clothing : spent_clothing = 0.5 * total_spent) 
  (h_food : spent_food = 0.2 * total_spent) 
  (h_other : spent_other = 0.3 * total_spent) 
  (h_other_tax : tax_other_rate = 0.1) 
  (h_total_tax : total_tax_rate = 0.055) 
  (h_total_spent : total_spent = 100):
  (tax_clothing * spent_clothing + tax_other_rate * spent_other) = total_tax_rate * total_spent → 
  tax_clothing = 0.05 :=
by
  sorry

end Jill_tax_on_clothing_l151_151735


namespace quad_common_root_l151_151154

theorem quad_common_root (a b c d : ℝ) :
  (∃ α : ℝ, α^2 + a * α + b = 0 ∧ α^2 + c * α + d = 0) ↔ (a * d - b * c) * (c - a) = (b - d)^2 ∧ (a ≠ c) := 
sorry

end quad_common_root_l151_151154


namespace total_passengers_landed_l151_151331

theorem total_passengers_landed (on_time late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) : 
    on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l151_151331


namespace amy_total_soups_l151_151269

def chicken_soup := 6
def tomato_soup := 3
def vegetable_soup := 4
def clam_chowder := 2
def french_onion_soup := 1
def minestrone_soup := 5

theorem amy_total_soups : (chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup) = 21 := by
  sorry

end amy_total_soups_l151_151269


namespace khalil_total_payment_l151_151206

def cost_dog := 60
def cost_cat := 40
def cost_parrot := 70
def cost_rabbit := 50

def num_dogs := 25
def num_cats := 45
def num_parrots := 15
def num_rabbits := 10

def total_cost := num_dogs * cost_dog + num_cats * cost_cat + num_parrots * cost_parrot + num_rabbits * cost_rabbit

theorem khalil_total_payment : total_cost = 4850 := by
  sorry

end khalil_total_payment_l151_151206


namespace tan_sum_l151_151907

open Real

theorem tan_sum 
  (α β γ θ φ : ℝ)
  (h1 : tan θ = (sin α * cos γ - sin β * sin γ) / (cos α * cos γ - cos β * sin γ))
  (h2 : tan φ = (sin α * sin γ - sin β * cos γ) / (cos α * sin γ - cos β * cos γ)) : 
  tan (θ + φ) = tan (α + β) :=
by
  sorry

end tan_sum_l151_151907


namespace fraction_is_half_l151_151900

variable (N : ℕ) (F : ℚ)

theorem fraction_is_half (h1 : N = 90) (h2 : 3 + F * (1/3) * (1/5) * N = (1/15) * N) : F = 1/2 :=
by
  sorry

end fraction_is_half_l151_151900


namespace car_drive_distance_l151_151671

-- Define the conditions as constants
def driving_speed : ℕ := 8 -- miles per hour
def driving_hours_before_cool : ℕ := 5 -- hours of constant driving
def cooling_hours : ℕ := 1 -- hours needed for cooling down
def total_time : ℕ := 13 -- hours available

-- Define the calculation for distance driven in cycles
def distance_per_cycle : ℕ := driving_speed * driving_hours_before_cool

-- Calculate the duration of one complete cycle
def cycle_duration : ℕ := driving_hours_before_cool + cooling_hours

-- Theorem statement: the car can drive 88 miles in 13 hours
theorem car_drive_distance : distance_per_cycle * (total_time / cycle_duration) + driving_speed * (total_time % cycle_duration) = 88 :=
by
  sorry

end car_drive_distance_l151_151671


namespace inequality_proof_l151_151043

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b < 2) : 
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ 
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ a < 1) := 
by 
  sorry

end inequality_proof_l151_151043


namespace numbers_whose_triples_plus_1_are_primes_l151_151231

def is_prime (n : ℕ) : Prop := Nat.Prime n

def in_prime_range (n : ℕ) : Prop := 
  is_prime n ∧ 70 ≤ n ∧ n ≤ 110

def transformed_by_3_and_1 (x : ℕ) : ℕ := 3 * x + 1

theorem numbers_whose_triples_plus_1_are_primes :
  { x : ℕ | in_prime_range (transformed_by_3_and_1 x) } = {24, 26, 32, 34, 36} :=
by
  sorry

end numbers_whose_triples_plus_1_are_primes_l151_151231


namespace cos_A_eq_l151_151311

variable (A : Real) (A_interior_angle_tri_ABC : A > π / 2 ∧ A < π) (tan_A_eq_neg_two : Real.tan A = -2)

theorem cos_A_eq : Real.cos A = - (Real.sqrt 5) / 5 := by
  sorry

end cos_A_eq_l151_151311


namespace temperature_on_Friday_l151_151880

-- Definitions of the temperatures on the days
variables {M T W Th F : ℝ}

-- Conditions given in the problem
def avg_temp_mon_thu (M T W Th : ℝ) : Prop := (M + T + W + Th) / 4 = 48
def avg_temp_tue_fri (T W Th F : ℝ) : Prop := (T + W + Th + F) / 4 = 46
def temp_mon (M : ℝ) : Prop := M = 44

-- Statement to prove
theorem temperature_on_Friday (h1 : avg_temp_mon_thu M T W Th)
                               (h2 : avg_temp_tue_fri T W Th F)
                               (h3 : temp_mon M) : F = 36 :=
sorry

end temperature_on_Friday_l151_151880


namespace probability_window_opens_correct_l151_151664

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l151_151664


namespace steve_pencils_left_l151_151748

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l151_151748


namespace initial_money_l151_151360

theorem initial_money (cost_of_candy_bar : ℕ) (change_received : ℕ) (initial_money : ℕ) 
  (h_cost : cost_of_candy_bar = 45) (h_change : change_received = 5) :
  initial_money = cost_of_candy_bar + change_received :=
by
  -- here is the place for the proof which is not needed
  sorry

end initial_money_l151_151360


namespace lowest_degree_required_l151_151507

noncomputable def smallest_degree_poly (b : ℤ) : ℕ :=
  if h : ∃ P : Polynomial ℝ, (∀ x, (P.eval x ≠ b)) ∧
    (∃ y, (P.eval y > b)) ∧ (∃ z, (P.eval z < b)) 
  then Nat.find h 
  else 0

theorem lowest_degree_required :
  ∃ b : ℤ, smallest_degree_poly b = 4 :=
by
  -- b is some integer that fits the described conditions
  use 0
  sorry

end lowest_degree_required_l151_151507


namespace find_A_coords_find_AC_equation_l151_151313

theorem find_A_coords
  (B : ℝ × ℝ) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ A : ℝ × ℝ, A = (-2, 2) :=
by
  sorry

theorem find_AC_equation
  (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ k b : ℝ, ∀ x y, y = k * x + b ↔ 3 * x - 4 * y + 14 = 0 :=
by
  sorry

end find_A_coords_find_AC_equation_l151_151313


namespace domain_f_x_plus_2_l151_151971

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Define the given condition: the domain of y = f(2x - 3) is [-2, 3]
def domain_f_2x_minus_3 : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 3}

-- Express this condition formally
axiom domain_f_2x_minus_3_axiom :
  ∀ (x : ℝ), (x ∈ domain_f_2x_minus_3) → (2 * x - 3 ∈ Set.Icc (-7 : ℝ) 3)

-- Prove the desired result: the domain of y = f(x + 2) is [-9, 1]
theorem domain_f_x_plus_2 :
  ∀ (x : ℝ), (x ∈ Set.Icc (-9 : ℝ) 1) ↔ ((x + 2) ∈ Set.Icc (-7 : ℝ) 3) :=
sorry

end domain_f_x_plus_2_l151_151971


namespace solution_set_of_inequality_l151_151565

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_def : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2 * x) :
  {x : ℝ | f (x + 2) < 3} = {x : ℝ | -5 < x ∧ x < 1} :=
by sorry

end solution_set_of_inequality_l151_151565


namespace cashier_window_open_probability_l151_151659

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l151_151659


namespace sqrt_subtraction_l151_151100

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l151_151100


namespace maximum_rectangle_area_l151_151532

-- Define the perimeter condition
def perimeter (rectangle : ℝ × ℝ) : ℝ :=
  2 * rectangle.fst + 2 * rectangle.snd

-- Define the area function
def area (rectangle : ℝ × ℝ) : ℝ :=
  rectangle.fst * rectangle.snd

-- Define the question statement in terms of Lean
theorem maximum_rectangle_area (length_width : ℝ × ℝ) (h : perimeter length_width = 32) : 
  area length_width ≤ 64 :=
sorry

end maximum_rectangle_area_l151_151532


namespace ratio_of_silver_to_gold_l151_151356

-- Definitions for balloon counts
def gold_balloons : Nat := 141
def black_balloons : Nat := 150
def total_balloons : Nat := 573

-- Define the number of silver balloons S
noncomputable def silver_balloons : Nat :=
  total_balloons - gold_balloons - black_balloons

-- The goal is to prove the ratio of silver to gold balloons is 2
theorem ratio_of_silver_to_gold :
  (silver_balloons / gold_balloons) = 2 := by
  sorry

end ratio_of_silver_to_gold_l151_151356


namespace total_pizzas_made_l151_151976

theorem total_pizzas_made (hc1 : Heather made 4 * Craig made on day1)
                          (hc2 : Heather made on day2 = Craig made on day2 - 20)
                          (hc3 : Craig made on day1 = 40)
                          (hc4 : Craig made on day2 = Craig made on day1 + 60) :
   Heather made on day1 + Craig made on day1 + Heather made on day2 + Craig made on day2 = 380 := 
by
  sorry

end total_pizzas_made_l151_151976


namespace max_distinct_values_is_two_l151_151686

-- Definitions of non-negative numbers and conditions
variable (a b c d : ℝ)
variable (ha : 0 ≤ a)
variable (hb : 0 ≤ b)
variable (hc : 0 ≤ c)
variable (hd : 0 ≤ d)
variable (h1 : Real.sqrt (a + b) + Real.sqrt (c + d) = Real.sqrt (a + c) + Real.sqrt (b + d))
variable (h2 : Real.sqrt (a + c) + Real.sqrt (b + d) = Real.sqrt (a + d) + Real.sqrt (b + c))

-- Theorem stating that the maximum number of distinct values among a, b, c, d is 2.
theorem max_distinct_values_is_two : 
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ (u = a ∨ u = b ∨ u = c ∨ u = d) ∧ (v = a ∨ v = b ∨ v = c ∨ v = d) ∧ 
  ∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x = y ∨ x = u ∨ x = v :=
sorry

end max_distinct_values_is_two_l151_151686


namespace eval_expression_at_a_l151_151363

theorem eval_expression_at_a (a : ℝ) (h : a = 1 / 2) : (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end eval_expression_at_a_l151_151363


namespace largest_number_is_D_l151_151631

noncomputable def A : ℝ := 15467 + 3 / 5791
noncomputable def B : ℝ := 15467 - 3 / 5791
noncomputable def C : ℝ := 15467 * (3 / 5791)
noncomputable def D : ℝ := 15467 / (3 / 5791)
noncomputable def E : ℝ := 15467.5791

theorem largest_number_is_D :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_is_D_l151_151631


namespace staff_member_pays_l151_151647

noncomputable def calculate_final_price (d : ℝ) : ℝ :=
  let discounted_price := 0.55 * d
  let staff_discounted_price := 0.33 * d
  let final_price := staff_discounted_price + 0.08 * staff_discounted_price
  final_price

theorem staff_member_pays (d : ℝ) : calculate_final_price d = 0.3564 * d :=
by
  unfold calculate_final_price
  sorry

end staff_member_pays_l151_151647


namespace num_zeros_in_interval_l151_151490

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + 7

theorem num_zeros_in_interval : 
    (∃ (a b : ℝ), a < b ∧ a = 0 ∧ b = 2 ∧
     (∀ x, f x = 0 → (0 < x ∧ x < 2)) ∧
     (∃! x, (0 < x ∧ x < 2) ∧ f x = 0)) :=
by
    sorry

end num_zeros_in_interval_l151_151490


namespace number_of_ways_to_distribute_balls_l151_151005

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l151_151005


namespace distinct_real_roots_l151_151559

def otimes (a b : ℝ) : ℝ := b^2 - a * b

theorem distinct_real_roots (m x : ℝ) :
  otimes (m - 2) x = m -> ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x^2 - (m - 2) * x - m = 0) := by
  sorry

end distinct_real_roots_l151_151559


namespace program_count_l151_151534

noncomputable def choose_programs : ℕ :=
  let courses := {'English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Biology}.to_finset in
  let math_courses := {'Algebra, 'Geometry}.to_finset in
  let english := 'English in
  let non_english_courses := courses.erase english in
  let total_choices := non_english_courses.card.choose 4 in
  let invalid_choices :=
    (finset.singleton (non_english_courses).choose 4).card + -- No math courses
    (math_courses.card.choose 1 * (non_english_courses.card - 1).choose 3) -- Exactly one math course
  in
  total_choices - invalid_choices

theorem program_count : choose_programs = 6 :=
  by
    -- Calculation details hidden
    rw [choose_programs] 
    sorry

end program_count_l151_151534


namespace terminal_side_quadrant_l151_151850

theorem terminal_side_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) :
  ∃ k : ℤ, (k % 2 = 0 ∧ (k * Real.pi + Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + Real.pi / 2)) ∨
           (k % 2 = 1 ∧ (k * Real.pi + 3 * Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + 5 * Real.pi / 4)) := sorry

end terminal_side_quadrant_l151_151850


namespace runner_injury_point_l151_151917

theorem runner_injury_point
  (v d : ℝ)
  (h1 : 2 * (40 - d) / v = d / v + 11)
  (h2 : 2 * (40 - d) / v = 22) :
  d = 20 := 
by
  sorry

end runner_injury_point_l151_151917


namespace original_price_increased_by_total_percent_l151_151195

noncomputable def percent_increase_sequence (P : ℝ) : ℝ :=
  let step1 := P * 1.15
  let step2 := step1 * 1.40
  let step3 := step2 * 1.20
  let step4 := step3 * 0.90
  let step5 := step4 * 1.25
  (step5 - P) / P * 100

theorem original_price_increased_by_total_percent (P : ℝ) : percent_increase_sequence P = 117.35 :=
by
  -- Sorry is used here for simplicity, but the automated proof will involve calculating the exact percentage increase step-by-step.
  sorry

end original_price_increased_by_total_percent_l151_151195


namespace balls_in_boxes_l151_151009

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l151_151009


namespace sum_of_M_l151_151078

theorem sum_of_M (x y z w M : ℕ) (hxw : w = x + y + z) (hM : M = x * y * z * w) (hM_cond : M = 12 * (x + y + z + w)) :
  ∃ sum_M, sum_M = 2208 :=
by 
  sorry

end sum_of_M_l151_151078


namespace quadratic_root_a_value_l151_151713

theorem quadratic_root_a_value (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ x^2 + (3 / 2) * a * x - a^2 = 0) → (a = 1 ∨ a = -4) := 
by
  intro h
  sorry

end quadratic_root_a_value_l151_151713


namespace find_k_value_l151_151734

theorem find_k_value 
  (A B C k : ℤ)
  (hA : A = -3)
  (hB : B = -5)
  (hC : C = 6)
  (hSum : A + B + C + k = -A - B - C - k) : 
  k = 2 :=
sorry

end find_k_value_l151_151734


namespace salt_mixture_problem_l151_151710

theorem salt_mixture_problem :
  ∃ (m : ℝ), 0.20 = (150 + 0.05 * m) / (600 + m) :=
by
  sorry

end salt_mixture_problem_l151_151710


namespace multiples_of_seven_with_units_digit_three_l151_151417

theorem multiples_of_seven_with_units_digit_three :
  ∃ n : ℕ, n = 2 ∧ ∀ k : ℕ, (k < 150 ∧ k % 7 = 0 ∧ k % 10 = 3) ↔ (k = 63 ∨ k = 133) := by
  sorry

end multiples_of_seven_with_units_digit_three_l151_151417


namespace necessary_but_not_sufficient_l151_151250

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (x > 1) ∨ (x ≤ -1) := 
by 
  sorry

end necessary_but_not_sufficient_l151_151250


namespace inequality_solution_inequality_proof_l151_151703

def f (x: ℝ) := |x - 5|

theorem inequality_solution : {x : ℝ | f x + f (x + 2) ≤ 3} = {x | 5 / 2 ≤ x ∧ x ≤ 11 / 2} :=
sorry

theorem inequality_proof (a x : ℝ) (h : a < 0) : f (a * x) - f (5 * a) ≥ a * f x :=
sorry

end inequality_solution_inequality_proof_l151_151703


namespace g_zero_not_in_range_l151_151461

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else ⌊2 / (x + 3)⌋

theorem g_zero_not_in_range :
  ¬ ∃ x : ℝ, x ≠ -3 ∧ g x = 0 := 
sorry

end g_zero_not_in_range_l151_151461


namespace shorts_more_than_checkered_l151_151988

noncomputable def total_students : ℕ := 81

noncomputable def striped_shirts : ℕ := (2 * total_students) / 3

noncomputable def checkered_shirts : ℕ := total_students - striped_shirts

noncomputable def shorts : ℕ := striped_shirts - 8

theorem shorts_more_than_checkered :
  shorts - checkered_shirts = 19 :=
by
  sorry

end shorts_more_than_checkered_l151_151988


namespace unique_intersection_value_l151_151959

theorem unique_intersection_value :
  (∀ (x y : ℝ), y = x^2 → y = 4 * x + k) → (k = -4) := 
by
  sorry

end unique_intersection_value_l151_151959


namespace man_l151_151794

noncomputable def speed_in_still_water (current_speed_kmph : ℝ) (distance_m : ℝ) (time_seconds : ℝ) : ℝ :=
   let current_speed_mps := current_speed_kmph * 1000 / 3600
   let downstream_speed_mps := distance_m / time_seconds
   let still_water_speed_mps := downstream_speed_mps - current_speed_mps
   let still_water_speed_kmph := still_water_speed_mps * 3600 / 1000
   still_water_speed_kmph

theorem man's_speed_in_still_water :
  speed_in_still_water 6 100 14.998800095992323 = 18 := by
  sorry

end man_l151_151794


namespace smallest_sum_of_three_l151_151813

open Finset

-- Define the set of numbers
def my_set : Finset ℤ := {10, 2, -4, 15, -7}

-- Statement of the problem: Prove the smallest sum of any three different numbers from the set is -9
theorem smallest_sum_of_three :
  ∃ (a b c : ℤ), a ∈ my_set ∧ b ∈ my_set ∧ c ∈ my_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end smallest_sum_of_three_l151_151813


namespace octal_subtraction_l151_151935

theorem octal_subtraction : (53 - 27 : ℕ) = 24 :=
by sorry

end octal_subtraction_l151_151935


namespace length_of_floor_is_10_l151_151118

variable (L : ℝ) -- Declare the variable representing the length of the floor

-- Conditions as definitions
def width_of_floor := 8
def strip_width := 2
def area_of_rug := 24
def rug_length := L - 2 * strip_width
def rug_width := width_of_floor - 2 * strip_width

-- Math proof problem statement
theorem length_of_floor_is_10
  (h1 : rug_length * rug_width = area_of_rug)
  (h2 : width_of_floor = 8)
  (h3 : strip_width = 2) :
  L = 10 :=
by
  -- Placeholder for the actual proof
  sorry

end length_of_floor_is_10_l151_151118


namespace total_games_played_is_53_l151_151866

theorem total_games_played_is_53 :
  ∃ (ken_wins dave_wins jerry_wins larry_wins total_ties total_games_played : ℕ),
  jerry_wins = 7 ∧
  dave_wins = jerry_wins + 3 ∧
  ken_wins = dave_wins + 5 ∧
  larry_wins = 2 * jerry_wins ∧
  5 ≤ ken_wins ∧ 5 ≤ dave_wins ∧ 5 ≤ jerry_wins ∧ 5 ≤ larry_wins ∧
  total_ties = jerry_wins ∧
  total_games_played = ken_wins + dave_wins + jerry_wins + larry_wins + total_ties ∧
  total_games_played = 53 :=
by
  sorry

end total_games_played_is_53_l151_151866


namespace cupcakes_left_l151_151210

theorem cupcakes_left (initial_cupcakes : ℕ)
  (students_delmont : ℕ) (ms_delmont : ℕ)
  (students_donnelly : ℕ) (mrs_donnelly : ℕ)
  (school_nurse : ℕ) (school_principal : ℕ) (school_custodians : ℕ)
  (favorite_teachers : ℕ) (cupcakes_per_favorite_teacher : ℕ)
  (other_classmates : ℕ) :
  initial_cupcakes = 80 →
  students_delmont = 18 → ms_delmont = 1 →
  students_donnelly = 16 → mrs_donnelly = 1 →
  school_nurse = 1 → school_principal = 1 → school_custodians = 3 →
  favorite_teachers = 5 → cupcakes_per_favorite_teacher = 2 → 
  other_classmates = 10 →
  initial_cupcakes - (students_delmont + ms_delmont +
                      students_donnelly + mrs_donnelly +
                      school_nurse + school_principal + school_custodians +
                      favorite_teachers * cupcakes_per_favorite_teacher +
                      other_classmates) = 19 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end cupcakes_left_l151_151210


namespace same_color_probability_l151_151418

variables (red blue green total : ℕ)
variables (select_ways same_color_ways : ℕ)

-- Given conditions
def conditions (red blue green total select_ways same_color_ways : ℕ) : Prop :=
  red = 7 ∧ blue = 5 ∧ green = 3 ∧ total = red + blue + green ∧
  select_ways = total.choose 3 ∧
  same_color_ways = red.choose 3 + blue.choose 3 + green.choose 3

-- The probability that all three plates are the same color equals 46/455
theorem same_color_probability :
  conditions red blue green total select_ways same_color_ways → 
  (same_color_ways : ℚ) / select_ways = 46 / 455 :=
by
  intros h
  cases h with h_red h_rest
  cases h_rest with h_blue h_rest
  cases h_rest with h_green h_rest
  cases h_rest with h_total h_rest
  cases h_rest with h_select h_fav
  rw [h_red, h_blue, h_green, h_total, h_select, h_fav]
  sorry

end same_color_probability_l151_151418


namespace fourth_person_height_l151_151110

variable (H : ℝ)
variable (height1 height2 height3 height4 : ℝ)

theorem fourth_person_height
  (h1 : height1 = H)
  (h2 : height2 = H + 2)
  (h3 : height3 = H + 4)
  (h4 : height4 = H + 10)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 78) :
  height4 = 84 :=
by
  sorry

end fourth_person_height_l151_151110


namespace fraction_passengers_from_asia_l151_151020

theorem fraction_passengers_from_asia (P : ℕ)
  (hP : P = 108)
  (frac_NA : ℚ) (frac_EU : ℚ) (frac_AF : ℚ)
  (Other_continents : ℕ)
  (h_frac_NA : frac_NA = 1/12)
  (h_frac_EU : frac_EU = 1/4)
  (h_frac_AF : frac_AF = 1/9)
  (h_Other_continents : Other_continents = 42) :
  (P * (1 - (frac_NA + frac_EU + frac_AF)) - Other_continents) / P = 1/6 :=
by
  sorry

end fraction_passengers_from_asia_l151_151020


namespace solve_fraction_equation_l151_151783

-- Defining the function f
def f (x : ℝ) : ℝ := x + 4

-- Statement of the problem
theorem solve_fraction_equation (x : ℝ) :
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ↔ x = 2 / 5 := by
  sorry

end solve_fraction_equation_l151_151783


namespace find_ordered_pair_l151_151555

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l151_151555


namespace exhibition_admission_fees_ratio_l151_151484

theorem exhibition_admission_fees_ratio
  (a c : ℕ)
  (h1 : 30 * a + 15 * c = 2925)
  (h2 : a % 5 = 0)
  (h3 : c % 5 = 0) :
  (a / 5 = c / 5) :=
by
  sorry

end exhibition_admission_fees_ratio_l151_151484


namespace sum_xyz_eq_11sqrt5_l151_151321

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l151_151321


namespace possible_values_of_b_l151_151602

theorem possible_values_of_b (r s : ℝ) (t t' : ℝ)
  (hp : ∀ x, x^3 + a * x + b = 0 → (x = r ∨ x = s ∨ x = t))
  (hq : ∀ x, x^3 + a * x + b + 240 = 0 → (x = r + 4 ∨ x = s - 3 ∨ x = t'))
  (h_sum_p : r + s + t = 0)
  (h_sum_q : (r + 4) + (s - 3) + t' = 0)
  (ha_p : a = r * s + r * t + s * t)
  (ha_q : a = (r + 4) * (s - 3) + (r + 4) * (t' - 1) + (s - 3) * (t' - 1))
  (ht'_def : t' = t - 1)
  : b = -330 ∨ b = 90 :=
by
  sorry

end possible_values_of_b_l151_151602


namespace minimum_ceiling_height_l151_151798

def is_multiple_of_0_1 (h : ℝ) : Prop := ∃ (k : ℤ), h = k / 10

def football_field_illuminated (h : ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 80 →
  (x^2 + y^2 ≤ h^2) ∨ ((x - 100)^2 + y^2 ≤ h^2) ∨
  (x^2 + (y - 80)^2 ≤ h^2) ∨ ((x - 100)^2 + (y - 80)^2 ≤ h^2)

theorem minimum_ceiling_height :
  ∃ (h : ℝ), football_field_illuminated h ∧ is_multiple_of_0_1 h ∧ h = 32.1 :=
sorry

end minimum_ceiling_height_l151_151798


namespace sin_plus_5cos_l151_151407

theorem sin_plus_5cos (x : Real) (h : cos x - 5 * sin x = 2) :
  sin x + 5 * cos x = -676 / 211 :=
sorry

end sin_plus_5cos_l151_151407


namespace corresponding_angles_equal_l151_151586

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l151_151586


namespace parabola_and_length_ef_l151_151000

theorem parabola_and_length_ef :
  ∃ a b : ℝ, (∀ x : ℝ, (x + 1) * (x - 3) = 0 → a * x^2 + b * x + 3 = 0) ∧ 
            (∀ x : ℝ, -a * x^2 + b * x + 3 = 7 / 4 → 
              ∃ x1 x2 : ℝ, x1 = -1 / 2 ∧ x2 = 5 / 2 ∧ abs (x2 - x1) = 3) := 
sorry

end parabola_and_length_ef_l151_151000


namespace consecutive_integer_sets_sum_100_l151_151218

theorem consecutive_integer_sets_sum_100 :
  ∃ s : Finset (Finset ℕ), 
    (∀ seq ∈ s, (∀ x ∈ seq, x > 0) ∧ (seq.sum id = 100)) ∧
    (s.card = 2) :=
sorry

end consecutive_integer_sets_sum_100_l151_151218


namespace weighted_average_correct_l151_151814

-- Define the marks
def english_marks : ℝ := 76
def mathematics_marks : ℝ := 65
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 85

-- Define the weightages
def english_weightage : ℝ := 0.20
def mathematics_weightage : ℝ := 0.25
def physics_weightage : ℝ := 0.25
def chemistry_weightage : ℝ := 0.15
def biology_weightage : ℝ := 0.15

-- Define the weighted sum calculation
def weighted_sum : ℝ :=
  english_marks * english_weightage + 
  mathematics_marks * mathematics_weightage + 
  physics_marks * physics_weightage + 
  chemistry_marks * chemistry_weightage + 
  biology_marks * biology_weightage

-- Define the theorem statement: the weighted average marks
theorem weighted_average_correct : weighted_sum = 74.75 :=
by
  sorry

end weighted_average_correct_l151_151814


namespace find_packs_size_l151_151300

theorem find_packs_size (y : ℕ) :
  (24 - 2 * y) * (36 + 4 * y) = 864 → y = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end find_packs_size_l151_151300


namespace division_correct_multiplication_correct_l151_151141

theorem division_correct : 400 / 5 = 80 := by
  sorry

theorem multiplication_correct : 230 * 3 = 690 := by
  sorry

end division_correct_multiplication_correct_l151_151141


namespace sandy_distance_l151_151604

theorem sandy_distance :
  ∃ d : ℝ, d = 18 * (1000 / 3600) * 99.9920006399488 := sorry

end sandy_distance_l151_151604


namespace solve_for_lambda_l151_151708

def vector_dot_product : (ℤ × ℤ) → (ℤ × ℤ) → ℤ
| (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

theorem solve_for_lambda
  (a : ℤ × ℤ) (b : ℤ × ℤ) (lambda : ℤ)
  (h1 : a = (3, -2))
  (h2 : b = (1, 2))
  (h3 : vector_dot_product (a.1 + lambda * b.1, a.2 + lambda * b.2) a = 0) :
  lambda = 13 :=
sorry

end solve_for_lambda_l151_151708


namespace value_of_a_l151_151854

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a_l151_151854


namespace eq1_eq2_eq3_eq4_l151_151349

theorem eq1 : ∀ x : ℝ, x = 6 → 3 * x - 8 = x + 4 := by
  intros x hx
  rw [hx]
  sorry

theorem eq2 : ∀ x : ℝ, x = -2 → 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) := by
  intros x hx
  rw [hx]
  sorry

theorem eq3 : ∀ x : ℝ, x = -20 → (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 := by
  intros x hx
  rw [hx]
  sorry

theorem eq4 : ∀ y : ℝ, y = -1 → (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  intros y hy
  rw [hy]
  sorry

end eq1_eq2_eq3_eq4_l151_151349


namespace find_finleys_age_l151_151738

-- Definitions for given problem
def rogers_age (J A : ℕ) := (J + A) / 2
def alex_age (F : ℕ) := 3 * (F + 10) - 5

-- Given conditions
def jills_age : ℕ := 20
def in_15_years_age_difference (R J F : ℕ) := R + 15 - (J + 15) = F - 30
def rogers_age_twice_jill_plus_five (J : ℕ) := 2 * J + 5

-- Theorem stating the problem assertion
theorem find_finleys_age (F : ℕ) :
  rogers_age jills_age (alex_age F) = rogers_age_twice_jill_plus_five jills_age ∧ 
  in_15_years_age_difference (rogers_age jills_age (alex_age F)) jills_age F →
  F = 15 :=
by
  sorry

end find_finleys_age_l151_151738


namespace monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l151_151560

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Monotonicity of f(x)
theorem monotonic_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

-- 2. f(x) is odd implies a = 1
theorem odd_function_implies_a_eq_1 (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

-- 3. Find max m such that f(x) ≥ m / 2^x for all x ∈ [2, 3]
theorem find_max_m (h : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f 1 x ≥ m / 2^x) : m ≤ 12/5 := sorry

end monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l151_151560


namespace minimum_stamps_l151_151540

theorem minimum_stamps (c f : ℕ) (h : 3 * c + 4 * f = 50) : c + f = 13 :=
sorry

end minimum_stamps_l151_151540


namespace arithmetic_sequence_a3_is_8_l151_151717

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove a3 = 8 given a1 = 4 and d = 2
theorem arithmetic_sequence_a3_is_8 (a1 d : ℕ) (h1 : a1 = 4) (h2 : d = 2) : arithmetic_sequence a1 d 3 = 8 :=
by
  sorry -- Proof not required as per instruction

end arithmetic_sequence_a3_is_8_l151_151717


namespace initial_investment_l151_151187

noncomputable def doubling_period (r : ℝ) : ℝ := 70 / r
noncomputable def investment_after_doubling (P : ℝ) (n : ℝ) : ℝ := P * (2 ^ n)

theorem initial_investment (total_amount : ℝ) (years : ℝ) (rate : ℝ) (initial : ℝ) :
  rate = 8 → total_amount = 28000 → years = 18 → 
  initial = total_amount / (2 ^ (years / (doubling_period rate))) :=
by
  intros hrate htotal hyears
  simp [doubling_period, investment_after_doubling] at *
  rw [hrate, htotal, hyears]
  norm_num
  sorry

end initial_investment_l151_151187


namespace combinatorics_sum_l151_151441

theorem combinatorics_sum :
  (Nat.choose 20 6 + Nat.choose 20 5 = 62016) :=
by
  sorry

end combinatorics_sum_l151_151441


namespace max_xy_of_conditions_l151_151465

noncomputable def max_xy : ℝ := 37.5

theorem max_xy_of_conditions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 10 * x + 15 * y = 150) (h4 : x^2 + y^2 ≤ 100) :
  xy ≤ max_xy :=
by sorry

end max_xy_of_conditions_l151_151465


namespace probability_window_opens_correct_l151_151665

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l151_151665


namespace remaining_payment_l151_151781
noncomputable def total_cost (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / percentage

noncomputable def remaining_amount (deposit : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost - deposit

theorem remaining_payment (deposit : ℝ) (percentage : ℝ) (total_cost : ℝ) (remaining_amount : ℝ) :
  deposit = 140 → percentage = 0.1 → total_cost = deposit / percentage → remaining_amount = total_cost - deposit → remaining_amount = 1260 :=
by
  intros
  sorry

end remaining_payment_l151_151781


namespace num_wide_right_misses_l151_151092

-- Define the conditions
variables (totalFieldGoals : ℕ) (missFraction : ℚ) (wideRightFraction : ℚ)

-- Given conditions
def totalFieldGoals := 60
def missFraction := 1 / 4
def wideRightFraction := 20 / 100

-- Calculate the actual number of field goals
def missedFieldGoals := totalFieldGoals * missFraction
def wideRightMisses := missedFieldGoals * wideRightFraction

-- Theorem to be proved
theorem num_wide_right_misses : wideRightMisses = 3 := sorry

end num_wide_right_misses_l151_151092


namespace persons_in_office_l151_151989

theorem persons_in_office
  (P : ℕ)
  (h1 : (P - (1/7 : ℚ)*P) = (6/7 : ℚ)*P)
  (h2 : (16.66666666666667/100 : ℚ) = 1/6) :
  P = 35 :=
sorry

end persons_in_office_l151_151989


namespace bag_contains_n_black_balls_l151_151788

theorem bag_contains_n_black_balls (n : ℕ) : (5 / (n + 5) = 1 / 3) → n = 10 := by
  sorry

end bag_contains_n_black_balls_l151_151788


namespace stratified_sample_over_30_l151_151526

-- Define the total number of employees and conditions
def total_employees : ℕ := 49
def employees_over_30 : ℕ := 14
def employees_30_or_younger : ℕ := 35
def sample_size : ℕ := 7

-- State the proportion and the final required count
def proportion_over_30 (total : ℕ) (over_30 : ℕ) : ℚ := (over_30 : ℚ) / (total : ℚ)
def required_count (proportion : ℚ) (sample : ℕ) : ℚ := proportion * (sample : ℚ)

theorem stratified_sample_over_30 :
  required_count (proportion_over_30 total_employees employees_over_30) sample_size = 2 := 
by sorry

end stratified_sample_over_30_l151_151526


namespace trapezoid_perimeter_l151_151881

theorem trapezoid_perimeter (a b : ℝ) (h : ∃ c : ℝ, a * b = c^2) :
  ∃ K : ℝ, K = 2 * (a + b + Real.sqrt (a * b)) :=
by
  sorry

end trapezoid_perimeter_l151_151881


namespace solution_proof_l151_151660

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l151_151660


namespace tetrahedron_ratio_l151_151381

theorem tetrahedron_ratio (a b c d : ℝ) (h₁ : a^2 = b^2 + c^2) (h₂ : b^2 = a^2 + d^2) (h₃ : c^2 = a^2 + b^2) : 
  a / d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end tetrahedron_ratio_l151_151381


namespace value_of_a_l151_151963

theorem value_of_a (x y z a : ℤ) (k : ℤ) 
  (h1 : x = 4 * k) (h2 : y = 6 * k) (h3 : z = 10 * k) 
  (hy_eq : y^2 = 40 * a - 20) 
  (ha_int : ∃ m : ℤ, a = m) : a = 1 := 
  sorry

end value_of_a_l151_151963


namespace necessary_condition_transitivity_l151_151521

theorem necessary_condition_transitivity (A B C : Prop) 
  (hAB : A → B) (hBC : B → C) : A → C := 
by
  intro ha
  apply hBC
  apply hAB
  exact ha

-- sorry


end necessary_condition_transitivity_l151_151521


namespace find_hcf_l151_151891

-- Defining the conditions given in the problem
def hcf_of_two_numbers_is_H (A B H : ℕ) : Prop := Nat.gcd A B = H
def lcm_of_A_B (A B : ℕ) (H : ℕ) : Prop := Nat.lcm A B = H * 21 * 23
def larger_number_is_460 (A : ℕ) : Prop := A = 460

-- The propositional goal to prove that H = 20 given the above conditions
theorem find_hcf (A B H : ℕ) (hcf_cond : hcf_of_two_numbers_is_H A B H)
  (lcm_cond : lcm_of_A_B A B H) (larger_cond : larger_number_is_460 A) : H = 20 :=
sorry

end find_hcf_l151_151891


namespace find_unit_prices_minimal_cost_l151_151992

-- Definitions for part 1
def unitPrices (x y : ℕ) : Prop :=
  20 * x + 30 * y = 2920 ∧ x - y = 11 

-- Definitions for part 2
def costFunction (m : ℕ) : ℕ :=
  52 * m + 48 * (40 - m)

def additionalPurchase (m : ℕ) : Prop :=
  m ≥ 40 / 3

-- Statement for unit prices proof
theorem find_unit_prices (x y : ℕ) (h1 : 20 * x + 30 * y = 2920) (h2 : x - y = 11) : x = 65 ∧ y = 54 := 
  sorry

-- Statement for minimal cost proof
theorem minimal_cost (m : ℕ) (x y : ℕ) 
  (hx : 20 * x + 30 * y = 2920) 
  (hy : x - y = 11)
  (hx_65 : x = 65)
  (hy_54 : y = 54)
  (hm : m ≥ 40 / 3) : 
  costFunction m = 1976 ∧ m = 14 :=
  sorry

end find_unit_prices_minimal_cost_l151_151992


namespace solve_equation_l151_151059

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (x + 1) / (x - 1) = 1 / (x - 2) + 1 → x = 3 := by
  sorry

end solve_equation_l151_151059


namespace exists_root_in_interval_l151_151205

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem exists_root_in_interval :
  f 1 < 0 → f 1.5 > 0 → f 1.25 < 0 → ∃ x ∈ (Set.Ioo 1.25 1.5), f x = 0 :=
by
  intros h1 h2 h3
  sorry

end exists_root_in_interval_l151_151205


namespace inequality_proof_l151_151728

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a * b * c := 
by {
  sorry
}

end inequality_proof_l151_151728


namespace base_eight_to_base_ten_l151_151236

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l151_151236


namespace measure_of_angle_C_sin_A_plus_sin_B_l151_151452

-- Problem 1
theorem measure_of_angle_C (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) : C = Real.pi / 3 := 
sorry

-- Problem 2
theorem sin_A_plus_sin_B (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) (h3 : c = 2 * Real.sqrt 3) : Real.sin A + Real.sin B = 3 / 2 := 
sorry

end measure_of_angle_C_sin_A_plus_sin_B_l151_151452


namespace largest_n_satisfying_condition_l151_151683

theorem largest_n_satisfying_condition
  (exists_n_elements: ∀ (n : ℕ) (S : Finset ℕ), S.card = n → 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ¬ (x ∣ y)) → 
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → (∃ (w : ℕ), w ∈ S ∧ w ∣ (x + y))) → 
    n ≤ 6) : 
  ∃ n (S : Finset ℕ), n = 6 ∧ S.card = n ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ¬ (x ∣ y)) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → 
      ∃ (w : ℕ), w ∈ S ∧ w ∣ (x + y)) :=
sorry

end largest_n_satisfying_condition_l151_151683


namespace line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l151_151826

-- Definitions for the first condition
def P : ℝ × ℝ := (3, 2)
def passes_through_P (l : ℝ → ℝ) := l P.1 = P.2
def equal_intercepts (l : ℝ → ℝ) := ∃ a : ℝ, l a = 0 ∧ l (-a) = 0

-- Equation 1: Line passing through P with equal intercepts
theorem line_through_P_with_equal_intercepts :
  (∃ l : ℝ → ℝ, passes_through_P l ∧ equal_intercepts l ∧ 
   (∀ x y : ℝ, l x = y ↔ (2 * x - 3 * y = 0) ∨ (x + y - 5 = 0))) :=
sorry

-- Definitions for the second condition
def A : ℝ × ℝ := (-1, -3)
def passes_through_A (l : ℝ → ℝ) := l A.1 = A.2
def inclination_90 (l : ℝ → ℝ) := ∀ x : ℝ, l x = l 0

-- Equation 2: Line passing through A with inclination 90°
theorem line_through_A_with_inclination_90 :
  (∃ l : ℝ → ℝ, passes_through_A l ∧ inclination_90 l ∧ 
   (∀ x y : ℝ, l x = y ↔ (x + 1 = 0))) :=
sorry

end line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l151_151826


namespace angles_of_triangle_arith_seq_l151_151314

theorem angles_of_triangle_arith_seq (A B C a b c : ℝ) (h1 : A + B + C = 180) (h2 : A = B - (B - C)) (h3 : (1 / a + 1 / c) / 2 = 1 / b) : 
  A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end angles_of_triangle_arith_seq_l151_151314


namespace find_number_l151_151379

-- Define the necessary variables and constants
variables (N : ℝ) (h1 : (5 / 4) * N = (4 / 5) * N + 18)

-- State the problem as a theorem to be proved
theorem find_number : N = 40 :=
by
  sorry

end find_number_l151_151379


namespace find_two_irreducible_fractions_l151_151954

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l151_151954


namespace tapA_turned_off_time_l151_151111

noncomputable def tapA_rate := 1 / 45
noncomputable def tapB_rate := 1 / 40
noncomputable def tapB_fill_time := 23

theorem tapA_turned_off_time :
  ∃ t : ℕ, t * (tapA_rate + tapB_rate) + tapB_fill_time * tapB_rate = 1 ∧ t = 9 :=
by
  sorry

end tapA_turned_off_time_l151_151111


namespace certain_number_105_l151_151986

theorem certain_number_105 (a x : ℕ) (h0 : a = 105) (h1 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end certain_number_105_l151_151986


namespace proof_l151_151028

noncomputable def line_standard_form (t : ℝ) : Prop :=
  let (x, y) := (t + 3, 3 - t)
  x + y = 6

noncomputable def circle_standard_form (θ : ℝ) : Prop :=
  let (x, y) := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  x^2 + (y - 2)^2 = 4

noncomputable def distance_center_to_line (x1 y1 : ℝ) : ℝ :=
  let (a, b, c) := (1, 1, -6)
  let num := abs (a * x1 + b * y1 + c)
  let denom := Real.sqrt (a^2 + b^2)
  num / denom

theorem proof : 
  (∀ t, line_standard_form t) ∧ 
  (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → circle_standard_form θ) ∧ 
  distance_center_to_line 0 2 = 2 * Real.sqrt 2 :=
by
  sorry

end proof_l151_151028


namespace calculate_expression_l151_151674

theorem calculate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + (1 / 6)) = 57 :=
by
  sorry

end calculate_expression_l151_151674


namespace find_rolls_of_toilet_paper_l151_151538

theorem find_rolls_of_toilet_paper (visits : ℕ) (squares_per_visit : ℕ) (squares_per_roll : ℕ) (days : ℕ)
  (h_visits : visits = 3)
  (h_squares_per_visit : squares_per_visit = 5)
  (h_squares_per_roll : squares_per_roll = 300)
  (h_days : days = 20000) : (visits * squares_per_visit * days) / squares_per_roll = 1000 :=
by
  sorry

end find_rolls_of_toilet_paper_l151_151538


namespace total_notebooks_distributed_l151_151301

/-- Define the parameters for children in Class A and Class B and the conditions given. -/
def ClassAChildren : ℕ := 64
def ClassBChildren : ℕ := 13

/-- Define the conditions as per the problem -/
def notebooksPerChildInClassA (A : ℕ) : ℕ := A / 8
def notebooksPerChildInClassB (A : ℕ) : ℕ := 2 * A
def totalChildrenClasses (A B : ℕ) : ℕ := A + B
def totalChildrenCondition (A : ℕ) : ℕ := 6 * A / 5

/-- Theorem to state the number of notebooks distributed between the two classes -/
theorem total_notebooks_distributed (A : ℕ) (B : ℕ) (H : A = 64) (H1 : B = 13) : 
  (A * (A / 8) + B * (2 * A)) = 2176 := by
  -- Conditions from the problem
  have conditionA : A = 64 := H
  have conditionB : B = 13 := H1
  have classA_notebooks : ℕ := (notebooksPerChildInClassA A) * A
  have classB_notebooks : ℕ := (notebooksPerChildInClassB A) * B
  have total_notebooks : ℕ := classA_notebooks + classB_notebooks
  -- Proof that total notebooks equals 2176
  sorry

end total_notebooks_distributed_l151_151301


namespace intersection_complement_correct_l151_151846

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A based on the condition given
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 3}

-- Define set B based on the condition given
def B : Set ℝ := {x | x > 3}

-- Define the complement of set B in the universal set U
def compl_B : Set ℝ := {x | x ≤ 3}

-- Define the expected result of A ∩ compl_B
def expected_result : Set ℝ := {x | x ≤ -3} ∪ {3}

-- State the theorem to be proven
theorem intersection_complement_correct :
  (A ∩ compl_B) = expected_result :=
sorry

end intersection_complement_correct_l151_151846


namespace arithmetic_sequence_sum_mod_l151_151142

theorem arithmetic_sequence_sum_mod (a d l k S n : ℕ) 
  (h_seq_start : a = 3)
  (h_common_difference : d = 5)
  (h_last_term : l = 103)
  (h_sum_formula : S = (k * (3 + 103)) / 2)
  (h_term_count : k = 21)
  (h_mod_condition : 1113 % 17 = n)
  (h_range_condition : 0 ≤ n ∧ n < 17) : 
  n = 8 :=
by
  sorry

end arithmetic_sequence_sum_mod_l151_151142


namespace eq_infinite_solutions_function_satisfies_identity_l151_151209

-- First Part: Proving the equation has infinitely many positive integer solutions
theorem eq_infinite_solutions : ∃ (x y z t : ℕ), ∀ n : ℕ, x^2 + 2 * y^2 = z^2 + 2 * t^2 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 := 
sorry

-- Second Part: Finding and proving the function f
def f (n : ℕ) : ℕ := n

theorem function_satisfies_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f n^2 + 2 * f m^2) = n^2 + 2 * m^2) : ∀ k : ℕ, f k = k :=
sorry

end eq_infinite_solutions_function_satisfies_identity_l151_151209


namespace simplify_expr_l151_151743

theorem simplify_expr : 2 - 2 / (1 + Real.sqrt 2) - 2 / (1 - Real.sqrt 2) = -2 := by
  sorry

end simplify_expr_l151_151743


namespace determine_correct_path_l151_151087

variable (A B C : Type)
variable (truthful : A → Prop)
variable (whimsical : A → Prop)
variable (answers : A → Prop)
variable (path_correct : A → Prop)

-- Conditions
axiom two_truthful_one_whimsical (x y z : A) : (truthful x ∧ truthful y ∧ whimsical z) ∨ 
                                                (truthful x ∧ truthful z ∧ whimsical y) ∨ 
                                                (truthful y ∧ truthful z ∧ whimsical x)

axiom traveler_aware : ∀ x y : A, truthful x → ¬ truthful y
axiom siblings : A → B → C → Prop
axiom ask_sibling : A → B → C → Prop

-- Conditions formalized
axiom ask_about_truthfulness (x y : A) : answers x → (truthful y ↔ ¬truthful y)

theorem determine_correct_path (x y z : A) :
  (truthful x ∧ ¬truthful y ∧ path_correct x) ∨
  (¬truthful x ∧ truthful y ∧ path_correct y) ∨
  (¬truthful x ∧ ¬truthful y ∧ truthful z ∧ path_correct z) :=
sorry

end determine_correct_path_l151_151087


namespace area_of_rhombus_l151_151093

-- Defining the conditions
def diagonal1 : ℝ := 20
def diagonal2 : ℝ := 30

-- Proving the area of the rhombus
theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  (d1 * d2 / 2) = 300 := by
  sorry

end area_of_rhombus_l151_151093


namespace zero_not_in_range_of_g_l151_151462

def g (x : ℝ) : ℤ :=
  if x > -3 then (Real.ceil (2 / (x + 3)))
  else if x < -3 then (Real.floor (2 / (x + 3)))
  else 0 -- g(x) is not defined at x = -3, hence this is a placeholder

noncomputable def range_g : Set ℤ := {n | ∃ x : ℝ, g x = n}

theorem zero_not_in_range_of_g : 0 ∉ range_g :=
by
  intros h,
  exact sorry

end zero_not_in_range_of_g_l151_151462


namespace one_plane_halves_rect_prism_l151_151799

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

end one_plane_halves_rect_prism_l151_151799


namespace bus_passengers_final_count_l151_151213

theorem bus_passengers_final_count :
  let initial_passengers := 15
  let changes := [(3, -6), (-2, 4), (-7, 2), (3, -5)]
  let apply_change (acc : Int) (change : Int × Int) : Int :=
    acc + change.1 + change.2
  initial_passengers + changes.foldl apply_change 0 = 7 :=
by
  intros
  sorry

end bus_passengers_final_count_l151_151213


namespace cube_volume_l151_151872

theorem cube_volume (a : ℕ) (h : a^3 - ((a - 2) * a * (a + 2)) = 16) : a^3 = 64 := by
  sorry

end cube_volume_l151_151872


namespace quadratic_coefficients_l151_151759

theorem quadratic_coefficients (x : ℝ) : 
  let a := 3
  let b := -5
  let c := 1
  3 * x^2 + 1 = 5 * x → a * x^2 + b * x + c = 0 := by
sorry

end quadratic_coefficients_l151_151759


namespace difference_red_white_l151_151670

/-
Allie picked 100 wildflowers. The categories of flowers are given as below:
- 13 of the flowers were yellow and white
- 17 of the flowers were red and yellow
- 14 of the flowers were red and white
- 16 of the flowers were blue and yellow
- 9 of the flowers were blue and white
- 8 of the flowers were red, blue, and yellow
- 6 of the flowers were red, white, and blue

The goal is to define the number of flowers containing red and white, and
prove that the difference between the number of flowers containing red and 
those containing white is 3.
-/

def total_flowers : ℕ := 100
def yellow_and_white : ℕ := 13
def red_and_yellow : ℕ := 17
def red_and_white : ℕ := 14
def blue_and_yellow : ℕ := 16
def blue_and_white : ℕ := 9
def red_blue_and_yellow : ℕ := 8
def red_white_and_blue : ℕ := 6

def flowers_with_red : ℕ := red_and_yellow + red_and_white + red_blue_and_yellow + red_white_and_blue
def flowers_with_white : ℕ := yellow_and_white + red_and_white + blue_and_white + red_white_and_blue

theorem difference_red_white : flowers_with_red - flowers_with_white = 3 := by
  rw [flowers_with_red, flowers_with_white]
  sorry

end difference_red_white_l151_151670


namespace original_number_l151_151038

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l151_151038


namespace total_students_l151_151859

theorem total_students (ratio_boys_girls : ℕ) (girls : ℕ) (boys : ℕ) (total_students : ℕ)
  (h1 : ratio_boys_girls = 2)     -- The simplified ratio of boys to girls
  (h2 : girls = 200)              -- There are 200 girls
  (h3 : boys = ratio_boys_girls * girls) -- Number of boys is ratio * number of girls
  (h4 : total_students = boys + girls)   -- Total number of students is the sum of boys and girls
  : total_students = 600 :=             -- Prove that the total number of students is 600
sorry

end total_students_l151_151859


namespace shift_graph_to_right_l151_151229

theorem shift_graph_to_right (x : ℝ) : 
  4 * Real.cos (2 * x + π / 4) = 4 * Real.cos (2 * (x - π / 8) + π / 4) :=
by 
  -- sketch of the intended proof without actual steps for clarity
  sorry

end shift_graph_to_right_l151_151229


namespace joe_prob_at_least_two_diff_fruits_l151_151864

noncomputable def probabilityAtLeastTwoDifferentFruits : ℝ :=
  1 - (4 * ((1 / 4) ^ 3))

theorem joe_prob_at_least_two_diff_fruits :
  probabilityAtLeastTwoDifferentFruits = 15 / 16 :=
by
  sorry

end joe_prob_at_least_two_diff_fruits_l151_151864


namespace find_number_l151_151295

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 56) : x = 140 := 
by {
  -- The proof would be written here,
  -- but it is indicated to skip it using "sorry"
  sorry
}

end find_number_l151_151295


namespace enjoyable_gameplay_time_l151_151328

def total_gameplay_time_base : ℝ := 150
def enjoyable_fraction_base : ℝ := 0.30
def total_gameplay_time_expansion : ℝ := 50
def load_screen_fraction_expansion : ℝ := 0.25
def inventory_management_fraction_expansion : ℝ := 0.25
def mod_skip_fraction : ℝ := 0.15

def enjoyable_time_base : ℝ := total_gameplay_time_base * enjoyable_fraction_base
def not_load_screen_time_expansion : ℝ := total_gameplay_time_expansion * (1 - load_screen_fraction_expansion)
def not_inventory_management_time_expansion : ℝ := not_load_screen_time_expansion * (1 - inventory_management_fraction_expansion)

def tedious_time_base : ℝ := total_gameplay_time_base * (1 - enjoyable_fraction_base)
def tedious_time_expansion : ℝ := total_gameplay_time_expansion - not_inventory_management_time_expansion
def total_tedious_time : ℝ := tedious_time_base + tedious_time_expansion

def time_skipped_by_mod : ℝ := total_tedious_time * mod_skip_fraction

def total_enjoyable_time : ℝ := enjoyable_time_base + not_inventory_management_time_expansion + time_skipped_by_mod

theorem enjoyable_gameplay_time :
  total_enjoyable_time = 92.16 :=     by     simp [total_enjoyable_time, enjoyable_time_base, not_inventory_management_time_expansion, time_skipped_by_mod]; sorry

end enjoyable_gameplay_time_l151_151328


namespace find_fractions_l151_151951

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l151_151951


namespace hypotenuse_length_l151_151915

theorem hypotenuse_length (a b : ℕ) (h : a = 9 ∧ b = 12) : ∃ c : ℕ, c = 15 ∧ a * a + b * b = c * c :=
by
  sorry

end hypotenuse_length_l151_151915


namespace find_n_l151_151152

theorem find_n (n : ℤ) (hn_range : -150 < n ∧ n < 150) (h_tan : Real.tan (n * Real.pi / 180) = Real.tan (286 * Real.pi / 180)) : 
  n = -74 :=
sorry

end find_n_l151_151152


namespace lowest_degree_is_4_l151_151511

noncomputable def lowest_degree_polynomial (P : Polynomial ℤ) (b : ℤ) : Prop :=
  ∃ (b : ℤ), 
    let A_P := P.support in
    (∀ (a ∈ A_P), a < b ∨ a > b) ∧ 
    (¬(b ∈ A_P)) ∧
    (∃ (a1 a2 : ℤ), a1 ∈ A_P ∧ a2 ∈ A_P ∧ a1 < b ∧ a2 > b)

theorem lowest_degree_is_4 :
  ∀ P : Polynomial ℤ, 
    let b := lowest_degree_polynomial P in
    b P 4 :=
sorry

end lowest_degree_is_4_l151_151511


namespace perpendicular_line_equation_l151_151550

theorem perpendicular_line_equation :
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 → x - 3 * y + 4 = 0 →
  ∃ (l : ℝ) (m : ℝ), m = 4 / 3 ∧ y = m * x + l → y = 4 / 3 * x + 1 / 9) 
  ∧ (∀ (x y : ℝ), 3 * x + 4 * y - 7 = 0 → -3 / 4 * 4 / 3 = -1) :=
by 
  sorry

end perpendicular_line_equation_l151_151550


namespace largest_divisor_of_three_consecutive_even_integers_is_sixteen_l151_151460

theorem largest_divisor_of_three_consecutive_even_integers_is_sixteen (n : ℕ) :
  ∃ d : ℕ, d = 16 ∧ 16 ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4)) :=
by
  sorry

end largest_divisor_of_three_consecutive_even_integers_is_sixteen_l151_151460


namespace prob_of_three_digit_divisible_by_3_l151_151567

/-- Define the exponents and the given condition --/
def a : ℕ := 5
def b : ℕ := 2
def c : ℕ := 3
def d : ℕ := 1

def condition : Prop := (2^a) * (3^b) * (5^c) * (7^d) = 252000

/-- The probability that a randomly chosen three-digit number formed by any 3 of a, b, c, d 
    is divisible by 3 and less than 250 is 1/4 --/
theorem prob_of_three_digit_divisible_by_3 :
  condition →
  ((sorry : ℝ) = 1/4) := sorry

end prob_of_three_digit_divisible_by_3_l151_151567


namespace minimum_days_to_owe_double_l151_151201

/-- Kim borrows $100$ dollars from Sam with a simple interest rate of $10\%$ per day.
    There's a one-time borrowing fee of $10$ dollars that is added to the debt immediately.
    We need to prove that the least integer number of days after which Kim will owe 
    Sam at least twice as much as she borrowed is 9 days.
-/
theorem minimum_days_to_owe_double :
  ∀ (x : ℕ), 100 + 10 + 10 * x ≥ 200 → x ≥ 9 :=
by
  intros x h
  sorry

end minimum_days_to_owe_double_l151_151201


namespace factor_difference_of_squares_l151_151821

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4 * x) * (7 + 4 * x) :=
by
  sorry

end factor_difference_of_squares_l151_151821


namespace x_cubed_inverse_cubed_l151_151183

theorem x_cubed_inverse_cubed (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 :=
by sorry

end x_cubed_inverse_cubed_l151_151183


namespace train_cross_time_l151_151121

-- Define the given conditions
def length_of_train : ℕ := 110
def length_of_bridge : ℕ := 265
def speed_kmh : ℕ := 45

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the total distance the train needs to travel
def total_distance : ℕ := length_of_train + length_of_bridge

-- Calculate the time it takes to cross the bridge
noncomputable def time_to_cross : ℝ := total_distance / speed_ms

-- State the theorem
theorem train_cross_time : time_to_cross = 30 := by sorry

end train_cross_time_l151_151121


namespace approximate_number_of_fish_in_pond_l151_151026

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, N = 800 ∧
  (40 : ℕ) / N = (2 : ℕ) / (40 : ℕ) := 
sorry

end approximate_number_of_fish_in_pond_l151_151026


namespace tan_sub_pi_over_4_l151_151983

variables (α : ℝ)
axiom tan_alpha : Real.tan α = 1 / 6

theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = -5 / 7 := by
  sorry

end tan_sub_pi_over_4_l151_151983


namespace arithmetic_expression_proof_l151_151823

theorem arithmetic_expression_proof : 4 * 6 * 8 + 18 / 3 ^ 2 = 194 := by
  sorry

end arithmetic_expression_proof_l151_151823


namespace corresponding_angles_equal_l151_151585

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l151_151585


namespace simplify_expression_l151_151479

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end simplify_expression_l151_151479


namespace range_of_a_l151_151563

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) → a ≤ -1 :=
by
  sorry

end range_of_a_l151_151563


namespace bivalid_positions_count_l151_151529

/-- 
A position of the hands of a (12-hour, analog) clock is called valid if it occurs in the course of a day.
A position of the hands is called bivalid if it is valid and, in addition, the position formed by interchanging the hour and minute hands is valid.
-/
def is_valid (h m : ℕ) : Prop := 
  0 ≤ h ∧ h < 360 ∧ 
  0 ≤ m ∧ m < 360

def satisfies_conditions (h m : Int) (a b : Int) : Prop :=
  m = 12 * h - 360 * a ∧ h = 12 * m - 360 * b

def is_bivalid (h m : ℕ) : Prop := 
  ∃ (a b : Int), satisfies_conditions (h : Int) (m : Int) a b ∧ satisfies_conditions (m : Int) (h : Int) b a

theorem bivalid_positions_count : 
  ∃ (n : ℕ), n = 143 ∧ 
  ∀ (h m : ℕ), is_bivalid h m → n = 143 :=
sorry

end bivalid_positions_count_l151_151529


namespace polynomial_simplification_l151_151134

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := 
by
  sorry

end polynomial_simplification_l151_151134


namespace remainder_sum_is_74_l151_151542

-- Defining the values from the given conditions
def num1 : ℕ := 1234567
def num2 : ℕ := 890123
def divisor : ℕ := 256

-- We state the theorem to capture the main problem
theorem remainder_sum_is_74 : (num1 + num2) % divisor = 74 := 
sorry

end remainder_sum_is_74_l151_151542


namespace base_eight_to_base_ten_l151_151235

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l151_151235


namespace middle_number_in_8th_row_l151_151047

-- Define a function that describes the number on the far right of the nth row.
def far_right_number (n : ℕ) : ℕ := n^2

-- Define a function that calculates the number of elements in the nth row.
def row_length (n : ℕ) : ℕ := 2 * n - 1

-- Define the middle number in the nth row.
def middle_number (n : ℕ) : ℕ := 
  let mid_index := (row_length n + 1) / 2
  far_right_number (n - 1) + mid_index

-- Statement to prove the middle number in the 8th row is 57
theorem middle_number_in_8th_row : middle_number 8 = 57 :=
by
  -- Placeholder for proof
  sorry

end middle_number_in_8th_row_l151_151047


namespace ellipse_sum_l151_151203

theorem ellipse_sum (F1 F2 : ℝ × ℝ) (h k a b : ℝ) 
  (hf1 : F1 = (0, 0)) (hf2 : F2 = (6, 0))
  (h_eqn : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 10) :
  h + k + a + b = 12 :=
by
  sorry

end ellipse_sum_l151_151203


namespace jemma_grasshoppers_l151_151197

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l151_151197


namespace corresponding_angles_equal_l151_151584

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l151_151584


namespace prod_of_extrema_l151_151305

noncomputable def f (x k : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem prod_of_extrema (k : ℝ) (h : ∀ x : ℝ, f x k ≥ 0 ∧ f x k ≤ 1 + (k - 1) / 3) :
  (∀ x : ℝ, f x k ≤ (k + 2) / 3) ∧ (∀ x : ℝ, f x k ≥ 1) → 
  (∃ φ ψ : ℝ, φ = 1 ∧ ψ = (k + 2) / 3 ∧ ∀ x y : ℝ, f x k = φ → f y k = ψ) → 
  (∃ φ ψ : ℝ, φ * ψ = (k + 2) / 3) :=
sorry

end prod_of_extrema_l151_151305


namespace simplify_expression_l151_151742

variable (x y : ℝ)

theorem simplify_expression : 3 * y + 5 * y + 6 * y + 2 * x + 4 * x = 14 * y + 6 * x :=
by
  sorry

end simplify_expression_l151_151742


namespace find_b_l151_151770

-- Define the lines and the condition of parallelism
def line1 := ∀ (x y b : ℝ), 4 * y + 8 * b = 16 * x
def line2 := ∀ (x y b : ℝ), y - 2 = (b - 3) * x
def are_parallel (m1 m2 : ℝ) := m1 = m2

-- Translate the problem to a Lean statement
theorem find_b (b : ℝ) : (∀ x y, 4 * y + 8 * b = 16 * x) → (∀ x y, y - 2 = (b - 3) * x) → b = 7 :=
by
  sorry

end find_b_l151_151770


namespace avg_weekly_income_500_l151_151263

theorem avg_weekly_income_500 :
  let base_salary := 350
  let income_past_5_weeks := [406, 413, 420, 436, 495]
  let commission_next_2_weeks_avg := 315
  let total_income_past_5_weeks := income_past_5_weeks.sum
  let total_base_salary_next_2_weeks := base_salary * 2
  let total_commission_next_2_weeks := commission_next_2_weeks_avg * 2
  let total_income := total_income_past_5_weeks + total_base_salary_next_2_weeks + total_commission_next_2_weeks
  let avg_weekly_income := total_income / 7
  avg_weekly_income = 500 := by
{
  sorry
}

end avg_weekly_income_500_l151_151263


namespace min_value_sq_sum_l151_151729

theorem min_value_sq_sum (x1 x2 : ℝ) (h : x1 * x2 = 2013) : (x1 + x2)^2 ≥ 8052 :=
by
  sorry

end min_value_sq_sum_l151_151729


namespace sum_of_fractions_l151_151274

-- Definitions of parameters and conditions
variables {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- The statement of the proof problem
theorem sum_of_fractions (hx : x ≠ 0) (hy : y ≠ 0) : 
  (3 / x) + (2 / y) = (3 * y + 2 * x) / (x * y) :=
sorry

end sum_of_fractions_l151_151274


namespace solve_for_x_l151_151898

theorem solve_for_x (x : ℝ) (h : 3375 = (1 / 4) * x + 144) : x = 12924 :=
by
  sorry

end solve_for_x_l151_151898


namespace equation_result_l151_151901

theorem equation_result : 
  ∀ (n : ℝ), n = 5.0 → (4 * n + 7 * n) = 55.0 :=
by
  intro n h
  rw [h]
  norm_num

end equation_result_l151_151901


namespace math_problem_l151_151063

noncomputable def parametric_equation_line (x y t : ℝ) : Prop :=
  x = 1 + (1/2) * t ∧ y = -5 + (Real.sqrt 3 / 2) * t

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

noncomputable def line_disjoint_circle (sqrt3 x y d : ℝ) : Prop :=
  sqrt3 = Real.sqrt 3 ∧ x = 0 ∧ y = 4 ∧ d = (9 + sqrt3) / 2 ∧ d > 4

theorem math_problem 
  (t θ x y ρ sqrt3 d : ℝ) :
  parametric_equation_line x y t ∧
  polar_equation_circle ρ θ ∧
  line_disjoint_circle sqrt3 x y d :=
by
  sorry

end math_problem_l151_151063


namespace number_of_people_got_on_train_l151_151084

theorem number_of_people_got_on_train (initial_people : ℕ) (people_got_off : ℕ) (final_people : ℕ) (x : ℕ) 
  (h_initial : initial_people = 78) 
  (h_got_off : people_got_off = 27) 
  (h_final : final_people = 63) 
  (h_eq : final_people = initial_people - people_got_off + x) : x = 12 :=
by 
  sorry

end number_of_people_got_on_train_l151_151084


namespace sqrt_subtraction_l151_151097

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l151_151097


namespace locus_eq_closed_curve_l151_151812

noncomputable def locus_of_points (d : ℝ) : set (ℝ × ℝ) :=
  let A := (d, 0)
  let B := (0, d)
  let O := (0, 0)
  let segment_AB := { P | 0 ≤ P.1 ∧ P.1 ≤ d ∧ P.2 = d - P.1 }
  let arc_third_quadrant := { Q | Q.1^2 + Q.2^2 = (d/2)^2 ∧ Q.1 ≤ 0 ∧ Q.2 ≤ 0 }
  let parabola_second_quadrant := { X | (X.1^2 - X.2) = (d^2 / 4) }
  let parabola_fourth_quadrant := { Y | (Y.1^2 + Y.2) = (d^2 / 4) }
  in segment_AB ∪ arc_third_quadrant ∪ parabola_second_quadrant ∪ parabola_fourth_quadrant

theorem locus_eq_closed_curve (d : ℝ) :
  ∃ (closed_curve : set (ℝ × ℝ)), closed_curve = locus_of_points d :=
by
  sorry

end locus_eq_closed_curve_l151_151812


namespace binom_9_6_l151_151675

theorem binom_9_6 : nat.choose 9 6 = 84 := by
  sorry

end binom_9_6_l151_151675


namespace spending_on_hydrangeas_l151_151348

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l151_151348


namespace three_digit_numbers_eq_11_sum_squares_l151_151773

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l151_151773


namespace buy_tshirts_l151_151220

theorem buy_tshirts
  (P T : ℕ)
  (h1 : 3 * P + 6 * T = 1500)
  (h2 : P + 12 * T = 1500)
  (budget : ℕ)
  (budget_eq : budget = 800) :
  (budget / T) = 8 := by
  sorry

end buy_tshirts_l151_151220


namespace smallest_sum_of_20_consecutive_integers_is_triangular_l151_151618

theorem smallest_sum_of_20_consecutive_integers_is_triangular : 
  ∃ n, let S := 10 * (2 * n + 19) in S = 190 ∧ ∃ m, S = m * (m + 1) / 2 :=
by sorry

end smallest_sum_of_20_consecutive_integers_is_triangular_l151_151618


namespace steve_pencils_left_l151_151749

-- Conditions
def initial_pencils : Nat := 24
def pencils_given_to_Lauren : Nat := 6
def extra_pencils_given_to_Matt : Nat := 3

-- Question: How many pencils does Steve have left?
theorem steve_pencils_left :
  initial_pencils - (pencils_given_to_Lauren + (pencils_given_to_Lauren + extra_pencils_given_to_Matt)) = 9 := by
  -- You need to provide a proof here
  sorry

end steve_pencils_left_l151_151749


namespace mathematicians_contemporaries_probability_l151_151500

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l151_151500


namespace part1_part2_l151_151001

def f (x : ℝ) := |x + 2|

theorem part1 (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7/3 < x ∧ x < -1 :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  (∀ x, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l151_151001


namespace min_value_f_l151_151564

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (4 / (1 - 2 * x))

theorem min_value_f : ∃ (x : ℝ), (0 < x ∧ x < 1 / 2) ∧ f x = 6 + 4 * sqrt 2 := by
  sorry

end min_value_f_l151_151564


namespace arithmetic_operations_correct_l151_151476

theorem arithmetic_operations_correct :
  (3 + (3 / 3) = (77 / 7) - 7) :=
by
  sorry

end arithmetic_operations_correct_l151_151476


namespace intersection_of_sets_l151_151570

open Set

theorem intersection_of_sets (p q : ℝ) :
  (M = {x : ℝ | x^2 - 5 * x < 0}) →
  (M = {x : ℝ | 0 < x ∧ x < 5}) →
  (N = {x : ℝ | p < x ∧ x < 6}) →
  (M ∩ N = {x : ℝ | 2 < x ∧ x < q}) →
  p + q = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end intersection_of_sets_l151_151570


namespace quarter_sector_area_l151_151626

theorem quarter_sector_area (d : ℝ) (h : d = 10) : (π * (d / 2)^2) / 4 = 6.25 * π :=
by 
  sorry

end quarter_sector_area_l151_151626


namespace perimeter_of_cube_face_is_28_l151_151615

-- Define the volume of the cube
def volume_of_cube : ℝ := 343

-- Define the side length of the cube based on the volume
def side_length_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the perimeter of one face of the cube
def perimeter_of_one_face (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: Prove the perimeter of one face of the cube is 28 cm given the volume is 343 cm³
theorem perimeter_of_cube_face_is_28 : 
  perimeter_of_one_face side_length_of_cube = 28 := 
by
  sorry

end perimeter_of_cube_face_is_28_l151_151615


namespace ratio_Polly_Willy_l151_151248

theorem ratio_Polly_Willy (P S W : ℝ) (h1 : P / S = 4 / 5) (h2 : S / W = 5 / 2) :
  P / W = 2 :=
by sorry

end ratio_Polly_Willy_l151_151248


namespace solve_cubic_equation_l151_151681

theorem solve_cubic_equation (x : ℚ) : (∃ x : ℚ, ∛(5 - x) = -5/3 ∧ x = 260/27) :=
by
  sorry

end solve_cubic_equation_l151_151681


namespace problem1_simplification_problem2_simplification_l151_151284

theorem problem1_simplification : (3 / Real.sqrt 3 - (Real.sqrt 3) ^ 2 - Real.sqrt 27 + (abs (Real.sqrt 3 - 2))) = -1 - 3 * Real.sqrt 3 :=
  by
    sorry

theorem problem2_simplification (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  ((x + 2) / (x ^ 2 - 2 * x) - (x - 1) / (x ^ 2 - 4 * x + 4)) / ((x - 4) / x) = 1 / (x - 2) ^ 2 :=
  by
    sorry

end problem1_simplification_problem2_simplification_l151_151284


namespace basketball_club_boys_l151_151644

theorem basketball_club_boys (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : B + (1 / 3) * G = 18) : B = 12 := 
by
  sorry

end basketball_club_boys_l151_151644


namespace final_pens_count_l151_151640

-- Define the initial number of pens and subsequent operations
def initial_pens : ℕ := 7
def pens_after_mike (initial : ℕ) : ℕ := initial + 22
def pens_after_cindy (pens : ℕ) : ℕ := pens * 2
def pens_after_sharon (pens : ℕ) : ℕ := pens - 19

-- Prove that the final number of pens is 39
theorem final_pens_count : pens_after_sharon (pens_after_cindy (pens_after_mike initial_pens)) = 39 := 
sorry

end final_pens_count_l151_151640


namespace cylinder_ratio_l151_151306

theorem cylinder_ratio
  (V : ℝ) (r h : ℝ)
  (h_volume : π * r^2 * h = V)
  (h_surface_area : 2 * π * r * h = 2 * (V / r)) :
  h / r = 2 :=
sorry

end cylinder_ratio_l151_151306


namespace find_two_fractions_sum_eq_86_over_111_l151_151948

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l151_151948


namespace value_of_m_l151_151832

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end value_of_m_l151_151832


namespace solve_equation_l151_151606

theorem solve_equation (x : ℝ) (h : x ≠ 3) (hx : x + 2 = 4 / (x - 3)) : 
    x = (1 + Real.sqrt 41) / 2 ∨ x = (1 - Real.sqrt 41) / 2 := by
sorry

end solve_equation_l151_151606


namespace solve_fraction_equation_l151_151396

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 5 * x - 14) - 1 / (x^2 - 15 * x - 18) = 0) →
  x = 2 ∨ x = -9 ∨ x = 6 ∨ x = -3 :=
sorry

end solve_fraction_equation_l151_151396


namespace triangle_base_and_height_l151_151609

theorem triangle_base_and_height (h b : ℕ) (A : ℕ) (hb : b = h - 4) (hA : A = 96) 
  (hArea : A = (1 / 2) * b * h) : (b = 12 ∧ h = 16) :=
by
  sorry

end triangle_base_and_height_l151_151609


namespace mixtilinear_incircle_radius_l151_151498
open Real

variable (AB BC AC : ℝ)
variable (r_A : ℝ)

def triangle_conditions : Prop :=
  AB = 65 ∧ BC = 33 ∧ AC = 56

theorem mixtilinear_incircle_radius 
  (h : triangle_conditions AB BC AC)
  : r_A = 12.89 := 
sorry

end mixtilinear_incircle_radius_l151_151498


namespace symmetric_slope_angle_l151_151845

theorem symmetric_slope_angle (α₁ : ℝ)
  (hα₁ : 0 ≤ α₁ ∧ α₁ < Real.pi) :
  ∃ α₂ : ℝ, (α₁ < Real.pi / 2 → α₂ = Real.pi - α₁) ∧
            (α₁ = Real.pi / 2 → α₂ = 0) :=
sorry

end symmetric_slope_angle_l151_151845


namespace john_annual_profit_l151_151459

namespace JohnProfit

def number_of_people_subletting := 3
def rent_per_person_per_month := 400
def john_rent_per_month := 900
def months_in_year := 12

theorem john_annual_profit 
  (h1 : number_of_people_subletting = 3)
  (h2 : rent_per_person_per_month = 400)
  (h3 : john_rent_per_month = 900)
  (h4 : months_in_year = 12) : 
  (number_of_people_subletting * rent_per_person_per_month - john_rent_per_month) * months_in_year = 3600 :=
by
  sorry

end JohnProfit

end john_annual_profit_l151_151459


namespace problem1_problem2_problem3_problem4_l151_151936

theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := 
by
  sorry

theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := 
by 
  sorry

theorem problem3 : ( (1/2) + (1/3) - (1/6) ) / (-1/18) = -12 := 
by 
  sorry

theorem problem4 : - 1 ^ 4 - (1 + 0.5) * (1/3) * (-4) ^ 2 = -33 / 32 := 
by 
  sorry


end problem1_problem2_problem3_problem4_l151_151936


namespace meetings_percent_l151_151588

/-- Define the lengths of the meetings and total workday in minutes -/
def first_meeting : ℕ := 40
def second_meeting : ℕ := 80
def second_meeting_overlap : ℕ := 10
def third_meeting : ℕ := 30
def workday_minutes : ℕ := 8 * 60

/-- Define the effective duration of the second meeting -/
def effective_second_meeting : ℕ := second_meeting - second_meeting_overlap

/-- Define the total time spent in meetings -/
def total_meeting_time : ℕ := first_meeting + effective_second_meeting + third_meeting

/-- Define the percentage of the workday spent in meetings -/
noncomputable def percent_meeting_time : ℚ := (total_meeting_time * 100 : ℕ) / workday_minutes

/-- Theorem: Given Laura's workday and meeting durations, prove that the percent of her workday spent in meetings is approximately 29.17%. -/
theorem meetings_percent {epsilon : ℚ} (h : epsilon = 0.01) : abs (percent_meeting_time - 29.17) < epsilon :=
sorry

end meetings_percent_l151_151588


namespace num_ordered_quadruples_l151_151466

theorem num_ordered_quadruples (n : ℕ) :
  ∃ (count : ℕ), count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) ∧
  (∀ (k1 k2 k3 k4 : ℕ), k1 ≤ n ∧ k2 ≤ n ∧ k3 ≤ n ∧ k4 ≤ n → 
    ((k1 + k3) / 2 = (k2 + k4) / 2) → 
    count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3)) :=
by sorry

end num_ordered_quadruples_l151_151466


namespace smallest_x_l151_151918

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_x (x a : ℕ) (h1 : a = 100 * x + 4950)
  (h2 : digitSum a = 50) :
  x = 99950 :=
by sorry

end smallest_x_l151_151918


namespace bean_lands_outside_inscribed_circle_l151_151582

theorem bean_lands_outside_inscribed_circle :
  let a := 8
  let b := 15
  let c := 17  -- hypotenuse computed as sqrt(a^2 + b^2)
  let area_triangle := (1 / 2) * a * b
  let s := (a + b + c) / 2  -- semiperimeter
  let r := area_triangle / s -- radius of the inscribed circle
  let area_incircle := π * r^2
  let probability_outside := 1 - area_incircle / area_triangle
  probability_outside = 1 - (3 * π) / 20 := 
by
  sorry

end bean_lands_outside_inscribed_circle_l151_151582


namespace least_distinct_values_l151_151793

theorem least_distinct_values (lst : List ℕ) (h_len : lst.length = 2023) (h_mode : ∃ m, (∀ n ≠ m, lst.count n < lst.count m) ∧ lst.count m = 13) : ∃ x, x = 169 :=
by
  sorry

end least_distinct_values_l151_151793


namespace girls_ran_9_miles_l151_151999

def boys_laps : ℕ := 34
def additional_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6

def girls_laps : ℕ := boys_laps + additional_laps
def girls_miles : ℚ := girls_laps * lap_distance

theorem girls_ran_9_miles : girls_miles = 9 := by
  sorry

end girls_ran_9_miles_l151_151999


namespace sequence_general_formula_l151_151293

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 + n * 5 

theorem sequence_general_formula (n : ℕ) : n > 0 → sequence_term n = 5 * n - 2 :=
by 
  sorry

end sequence_general_formula_l151_151293


namespace problem_statement_l151_151042

noncomputable def floor_T (u v w x : ℝ) : ℤ :=
  ⌊u + v + w + x⌋

theorem problem_statement (u v w x : ℝ) (T : ℝ) (h₁: u^2 + v^2 = 3005) (h₂: w^2 + x^2 = 3005) (h₃: u * w = 1729) (h₄: v * x = 1729) :
  floor_T u v w x = 155 :=
by
  sorry

end problem_statement_l151_151042


namespace cashier_opens_probability_l151_151662

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l151_151662


namespace arithmetic_sequence_sum_l151_151312

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ),
    (∀ (n : ℕ), a_n n = 1 + (n - 1) * d) →  -- first condition
    d ≠ 0 →  -- second condition
    (∀ (n : ℕ), S_n n = n / 2 * (2 * 1 + (n - 1) * d)) →  -- third condition
    (1 * (1 + 4 * d) = (1 + d) ^ 2) →  -- fourth condition
    S_n 8 = 64 :=  -- conclusion
by {
  sorry
}

end arithmetic_sequence_sum_l151_151312


namespace find_constant_l151_151025

theorem find_constant (t : ℝ) (constant : ℝ) :
  (x = constant - 3 * t) → (y = 2 * t - 3) → (t = 0.8) → (x = y) → constant = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end find_constant_l151_151025


namespace emberly_total_miles_l151_151143

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end emberly_total_miles_l151_151143


namespace min_value_abs_plus_one_l151_151394

theorem min_value_abs_plus_one : ∃ x : ℝ, |x| + 1 = 1 :=
by
  use 0
  sorry

end min_value_abs_plus_one_l151_151394


namespace area_of_triangle_AGE_l151_151049

open Set
open Real

-- Define the problem conditions
def square_side_length : ℝ := 5
def point_B (s : ℝ) : EuclideanSpace ℝ 2 := ⟨0, 0⟩
def point_C (s : ℝ) : EuclideanSpace ℝ 2 := ⟨s, 0⟩
def point_D (s : ℝ) : EuclideanSpace ℝ 2 := ⟨s, s⟩
def point_A (s : ℝ) : EuclideanSpace ℝ 2 := ⟨0, s⟩
def point_E : EuclideanSpace ℝ 2 := ⟨2, 0⟩

-- intersection on circumscribed circle of triangle ABE and diagonal BD at G
def point_G : Type := { G : EuclideanSpace ℝ 2 // ∃ (x : ℝ), G = (x, 5 - x) }

-- Calculate the area of triangle AGE
noncomputable def area_triangle_AGE (A E G: EuclideanSpace ℝ 2) : ℝ :=
  0.5 * abs ((A.1 * (E.2 - G.2)) + (E.1 * (G.2 - A.2)) + (G.1 * (A.2 - E.2)))


-- The statement to prove
theorem area_of_triangle_AGE :
  ∀ (s : ℝ), s = 5 →
  ∃ G : point_G,
  area_triangle_AGE (point_A s) point_E (G : EuclideanSpace ℝ 2) = 54.5 :=
by
  intro s hs
  rw hs
  -- Proceed to prove, omitted as it's not required.
  sorry

end area_of_triangle_AGE_l151_151049


namespace find_fractions_l151_151950

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l151_151950


namespace bowling_team_score_ratio_l151_151648

theorem bowling_team_score_ratio :
  ∀ (F S T : ℕ),
  F + S + T = 810 →
  F = (1 / 3 : ℚ) * S →
  T = 162 →
  S / T = 3 := 
by
  intros F S T h1 h2 h3
  sorry

end bowling_team_score_ratio_l151_151648


namespace price_restoration_percentage_l151_151077

noncomputable def original_price := 100
def reduced_price (P : ℝ) := 0.8 * P
def restored_price (P : ℝ) (x : ℝ) := P = x * reduced_price P

theorem price_restoration_percentage (P : ℝ) (x : ℝ) (h : restored_price P x) : x = 1.25 :=
by
  sorry

end price_restoration_percentage_l151_151077


namespace residue_mod_13_l151_151392

theorem residue_mod_13 : 
  (156 % 13 = 0) ∧ (52 % 13 = 0) ∧ (182 % 13 = 0) ∧ (26 % 13 = 0) →
  (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 :=
by
  intros h
  sorry

end residue_mod_13_l151_151392


namespace number_of_ways_to_distribute_balls_l151_151004

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l151_151004


namespace find_a_for_square_of_binomial_l151_151291

theorem find_a_for_square_of_binomial (a : ℝ) :
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) ↔ a = 9 := 
sorry

end find_a_for_square_of_binomial_l151_151291


namespace pairs_nat_eq_l151_151147

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l151_151147


namespace student_marks_l151_151192

theorem student_marks 
    (correct: ℕ) 
    (attempted: ℕ) 
    (marks_per_correct: ℕ) 
    (marks_per_incorrect: ℤ) 
    (correct_answers: correct = 27)
    (attempted_questions: attempted = 70)
    (marks_per_correct_condition: marks_per_correct = 3)
    (marks_per_incorrect_condition: marks_per_incorrect = -1): 
    (correct * marks_per_correct + (attempted - correct) * marks_per_incorrect) = 38 :=
by
    sorry

end student_marks_l151_151192


namespace sale_in_fifth_month_l151_151258

theorem sale_in_fifth_month (s1 s2 s3 s4 s5 s6 : ℤ) (avg_sale : ℤ) (h1 : s1 = 6435) (h2 : s2 = 6927)
  (h3 : s3 = 6855) (h4 : s4 = 7230) (h6 : s6 = 7391) (h_avg_sale : avg_sale = 6900) :
    (s1 + s2 + s3 + s4 + s5 + s6) / 6 = avg_sale → s5 = 6562 :=
by
  sorry

end sale_in_fifth_month_l151_151258


namespace no_linear_term_l151_151190

theorem no_linear_term (m : ℝ) (x : ℝ) : 
  (x + m) * (x + 3) - (x * x + 3 * m) = 0 → m = -3 :=
by
  sorry

end no_linear_term_l151_151190


namespace hyperbola_asymptotes_l151_151068

theorem hyperbola_asymptotes (x y : ℝ) : x^2 - 4 * y^2 = -1 → (x = 2 * y) ∨ (x = -2 * y) := 
by
  intro h
  sorry

end hyperbola_asymptotes_l151_151068


namespace andrew_donates_160_to_homeless_shelter_l151_151929

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l151_151929


namespace sum_of_three_numbers_l151_151241

theorem sum_of_three_numbers (x y z : ℝ) (h1 : x + y = 31) (h2 : y + z = 41) (h3 : z + x = 55) :
  x + y + z = 63.5 :=
by
  sorry

end sum_of_three_numbers_l151_151241


namespace tangent_line_at_1_l151_151888

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line at x=1
def slope_at_1 : ℝ := f' 1

-- Define the tangent line equation at x=1
def tangent_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem that the tangent line to f at x=1 is 2x - y + 1 = 0
theorem tangent_line_at_1 :
  tangent_line 1 (f 1) :=
by
  sorry

end tangent_line_at_1_l151_151888


namespace bricks_needed_for_courtyard_l151_151527

noncomputable def total_bricks_required (courtyard_length courtyard_width : ℝ)
  (brick_length_cm brick_width_cm : ℝ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_length := brick_length_cm / 100
  let brick_width := brick_width_cm / 100
  let brick_area := brick_length * brick_width
  courtyard_area / brick_area

theorem bricks_needed_for_courtyard :
  total_bricks_required 35 24 15 8 = 70000 := by
  sorry

end bricks_needed_for_courtyard_l151_151527


namespace percentage_temporary_workers_l151_151449

-- Definitions based on the given conditions
def total_workers : ℕ := 100
def percentage_technicians : ℝ := 0.9
def percentage_non_technicians : ℝ := 0.1
def percentage_permanent_technicians : ℝ := 0.9
def percentage_permanent_non_technicians : ℝ := 0.1

-- Statement to prove that the percentage of temporary workers is 18%
theorem percentage_temporary_workers :
  100 * (1 - (percentage_permanent_technicians * percentage_technicians +
              percentage_permanent_non_technicians * percentage_non_technicians)) = 18 :=
by sorry

end percentage_temporary_workers_l151_151449


namespace problem_l151_151422

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l151_151422


namespace distance_between_planes_l151_151946

-- Define the plane equations
def plane1 (x y z : ℝ) : Prop := 3 * x + y - z + 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 2 * y - 2 * z + 7 = 0

-- Define a point on the first plane
def point_on_plane1 : ℝ × ℝ × ℝ := (0, 0, 3)

-- Compute the distance between a point and a plane
def point_to_plane_distance (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  abs (6 * x + 2 * y - 2 * z + 7) / real.sqrt (6^2 + 2^2 + (-2)^2)

-- Prove the distance between the planes
theorem distance_between_planes : point_to_plane_distance point_on_plane1 = 1 / (2 * real.sqrt 11) := sorry

end distance_between_planes_l151_151946


namespace decimal_89_to_binary_l151_151136

def decimal_to_binary (n : ℕ) : ℕ := sorry

theorem decimal_89_to_binary :
  decimal_to_binary 89 = 1011001 :=
sorry

end decimal_89_to_binary_l151_151136


namespace syllogistic_reasoning_l151_151630

theorem syllogistic_reasoning (a b c : Prop) (h1 : b → c) (h2 : a → b) : a → c :=
by sorry

end syllogistic_reasoning_l151_151630


namespace average_of_primes_less_than_twenty_l151_151094

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sum_primes : ℕ := 77
def count_primes : ℕ := 8
def average_primes : ℚ := 77 / 8

theorem average_of_primes_less_than_twenty : (primes_less_than_twenty.sum / count_primes : ℚ) = 9.625 := by
  sorry

end average_of_primes_less_than_twenty_l151_151094


namespace value_of_x_for_fn_inv_eq_l151_151544

def f (x : ℝ) : ℝ := 4 * x - 9
def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem value_of_x_for_fn_inv_eq (x : ℝ) : f(x) = f_inv(x) → x = 3 :=
by
  sorry

end value_of_x_for_fn_inv_eq_l151_151544


namespace percentage_of_students_passed_l151_151193

def total_students : ℕ := 740
def failed_students : ℕ := 481
def passed_students : ℕ := total_students - failed_students
def pass_percentage : ℚ := (passed_students / total_students) * 100

theorem percentage_of_students_passed : pass_percentage = 35 := by
  sorry

end percentage_of_students_passed_l151_151193


namespace slope_of_line_l151_151547

noncomputable def line_equation (x y : ℝ) : Prop := 4 * y + 2 * x = 10

theorem slope_of_line (x y : ℝ) (h : line_equation x y) : -1 / 2 = -1 / 2 :=
by
  sorry

end slope_of_line_l151_151547


namespace triangle_largest_angle_l151_151027

theorem triangle_largest_angle (x : ℝ) (AB : ℝ) (AC : ℝ) (BC : ℝ) (h1 : AB = x + 5) 
                               (h2 : AC = 2 * x + 3) (h3 : BC = x + 10)
                               (h_angle_A_largest : BC > AB ∧ BC > AC)
                               (triangle_inequality_1 : AB + AC > BC)
                               (triangle_inequality_2 : AB + BC > AC)
                               (triangle_inequality_3 : AC + BC > AB) :
  1 < x ∧ x < 7 ∧ 6 = 6 := 
by {
  sorry
}

end triangle_largest_angle_l151_151027


namespace sum_of_digits_of_smallest_divisible_is_6_l151_151867

noncomputable def smallest_divisible (n : ℕ) : ℕ :=
Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_divisible_is_6 : sum_of_digits (smallest_divisible 7) = 6 := 
by
  simp [smallest_divisible, sum_of_digits]
  sorry

end sum_of_digits_of_smallest_divisible_is_6_l151_151867


namespace smallest_angle_in_icosagon_l151_151886

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l151_151886


namespace ellipse_properties_l151_151309

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def slopes_condition (x1 y1 x2 y2 : ℝ) (k_ab k_oa k_ob : ℝ) : Prop :=
  (k_ab^2 = k_oa * k_ob)

variables {x y : ℝ}

theorem ellipse_properties :
  (ellipse x y 2 1) ∧ -- Given ellipse equation
  (∃ (x1 y1 x2 y2 k_ab k_oa k_ob : ℝ), slopes_condition x1 y1 x2 y2 k_ab k_oa k_ob) →
  (∃ (OA OB : ℝ), OA^2 + OB^2 = 5) ∧ -- Prove sum of squares is constant
  (∃ (m : ℝ), (m = 1 → ∃ (line_eq : ℝ → ℝ), ∀ x, line_eq x = (1 / 2) * x + m)) -- Maximum area of triangle AOB

:= sorry

end ellipse_properties_l151_151309


namespace smallest_number_l151_151126

-- Definitions of the numbers in their respective bases
def num1 := 5 * 9^0 + 8 * 9^1 -- 85_9
def num2 := 0 * 6^0 + 1 * 6^1 + 2 * 6^2 -- 210_6
def num3 := 0 * 4^0 + 0 * 4^1 + 0 * 4^2 + 1 * 4^3 -- 1000_4
def num4 := 1 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 -- 111111_2

-- Assert that num4 is the smallest
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 :=
by 
  sorry

end smallest_number_l151_151126


namespace prove_q_l151_151186

theorem prove_q 
  (p q : ℝ)
  (h : (∀ x, (x + 3) * (x + p) = x^2 + q * x + 12)) : 
  q = 7 :=
sorry

end prove_q_l151_151186


namespace find_x_l151_151691

def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
def vector_b : ℝ × ℝ := (-3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular (vector_a x + vector_b) vector_b) : 
  x = -7 / 2 :=
  sorry

end find_x_l151_151691


namespace bridget_initial_skittles_l151_151539

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end bridget_initial_skittles_l151_151539


namespace lambda_range_l151_151569

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range_l151_151569


namespace find_a_of_ellipse_foci_l151_151188

theorem find_a_of_ellipse_foci (a : ℝ) :
  (∀ x y : ℝ, a^2 * x^2 - (a / 2) * y^2 = 1) →
  (a^2 - (2 / a) = 4) →
  a = (1 - Real.sqrt 5) / 4 :=
by 
  intros h1 h2
  sorry

end find_a_of_ellipse_foci_l151_151188


namespace zeros_of_f_l151_151619

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- State the theorem about its roots
theorem zeros_of_f : ∃ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end zeros_of_f_l151_151619


namespace Ivan_defeats_Koschei_l151_151978

-- Definitions of the springs and conditions based on the problem
section

variable (S: ℕ → Prop)  -- S(n) means the water from spring n
variable (deadly: ℕ → Prop)  -- deadly(n) if water from nth spring is deadly

-- Conditions
axiom accessibility (n: ℕ): (1 ≤ n ∧ n ≤ 9 → ∀ i: ℕ, S i)
axiom koschei_access: S 10
axiom lethality (n: ℕ): (S n → deadly n)
axiom neutralize (i j: ℕ): (1 ≤ i ∧ i < j ∧ j ≤ 9 → ∃ k: ℕ, S k ∧ k > j → ¬deadly i)

-- Statement to prove
theorem Ivan_defeats_Koschei:
  ∃ i: ℕ, (1 ≤ i ∧ i ≤ 9) → (S 10 → ¬deadly i) ∧ (S 0 ∧ (S 10 → deadly 0)) :=
sorry

end

end Ivan_defeats_Koschei_l151_151978


namespace smallest_k_l151_151399

theorem smallest_k (M : Finset ℕ) (H : ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d)) :
  ∃ k, k = 7 ∧ ∀ (M' : Finset ℕ), M'.card = k → ∀ (a b c d : ℕ), a ∈ M' → b ∈ M' → c ∈ M' → d ∈ M' → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d) :=
sorry

end smallest_k_l151_151399


namespace probability_of_selecting_A_l151_151803

noncomputable def total_students : ℕ := 4
noncomputable def selected_student_A : ℕ := 1

theorem probability_of_selecting_A : 
  (selected_student_A : ℝ) / (total_students : ℝ) = 1 / 4 :=
by
  sorry

end probability_of_selecting_A_l151_151803


namespace B_work_days_l151_151522

/-- 
  A and B undertake to do a piece of work for $500.
  A alone can do it in 5 days while B alone can do it in a certain number of days.
  With the help of C, they finish it in 2 days. C's share is $200.
  Prove B alone can do the work in 10 days.
-/
theorem B_work_days (x : ℕ) (h1 : (1/5 : ℝ) + (1/x : ℝ) = 3/10) : x = 10 := 
  sorry

end B_work_days_l151_151522


namespace balls_in_boxes_l151_151007

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l151_151007


namespace inequality_chain_l151_151161

theorem inequality_chain (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_chain_l151_151161


namespace price_correct_l151_151472

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end price_correct_l151_151472


namespace prob_goal_l151_151705

open MeasureTheory

variables {ξ : ℝ → Measure ℝ}
variables {μ σ : ℝ}

-- Assuming ξ follows a normal distribution N(μ, σ^2)
axiom normal_distribution : ∀ x, ξ x = Measure.normal_pdf μ σ x

-- Given conditions
axiom prob_1 : ξ {x | x < 1} = 0.5
axiom prob_2 : ξ {x | x > 2} = 0.4

-- Goal: Prove P(0 < ξ < 1) = 0.1
theorem prob_goal : ξ {x | 0 < x ∧ x < 1} = 0.1 :=
by
  sorry

end prob_goal_l151_151705


namespace smallest_angle_in_20_sided_polygon_is_143_l151_151885

theorem smallest_angle_in_20_sided_polygon_is_143
  (n : ℕ)
  (h_n : n = 20)
  (angles : ℕ → ℕ)
  (h_convex : ∀ i, 1 ≤ i → i ≤ n → angles i < 180)
  (h_arithmetic_seq : ∃ d : ℕ, ∀ i, 1 ≤ i → i < n → angles (i + 1) = angles i + d)
  (h_increasing : ∀ i, 1 ≤ i → i < n → angles (i + 1) > angles i)
  (h_sum : ∑ i in finset.range n, angles (i + 1) = (n - 2) * 180) :
  angles 1 = 143 :=
by
  sorry

end smallest_angle_in_20_sided_polygon_is_143_l151_151885


namespace colton_share_l151_151130

-- Definitions
def footToInch (foot : ℕ) : ℕ := 12 * foot -- 1 foot equals 12 inches

-- Problem conditions
def coltonBurgerLength := footToInch 1 -- Colton bought a foot long burger
def sharedBurger (length : ℕ) : ℕ := length / 2 -- shared half with his brother

-- Equivalent proof problem statement
theorem colton_share : sharedBurger coltonBurgerLength = 6 := 
by sorry

end colton_share_l151_151130


namespace umar_age_is_10_l151_151268

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l151_151268


namespace min_abc_value_l151_151591

noncomputable def minValue (a b c : ℝ) : ℝ := (a + b) / (a * b * c)

theorem min_abc_value (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  (minValue a b c) ≥ 16 :=
by
  sorry

end min_abc_value_l151_151591


namespace eval_32_pow_5_div_2_l151_151680

theorem eval_32_pow_5_div_2 :
  32^(5/2) = 4096 * Real.sqrt 2 :=
by
  sorry

end eval_32_pow_5_div_2_l151_151680


namespace find_a_l151_151704

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -1 - Real.log 2 := sorry

end find_a_l151_151704


namespace smallest_angle_in_20_sided_polygon_is_143_l151_151884

theorem smallest_angle_in_20_sided_polygon_is_143
  (n : ℕ)
  (h_n : n = 20)
  (angles : ℕ → ℕ)
  (h_convex : ∀ i, 1 ≤ i → i ≤ n → angles i < 180)
  (h_arithmetic_seq : ∃ d : ℕ, ∀ i, 1 ≤ i → i < n → angles (i + 1) = angles i + d)
  (h_increasing : ∀ i, 1 ≤ i → i < n → angles (i + 1) > angles i)
  (h_sum : ∑ i in finset.range n, angles (i + 1) = (n - 2) * 180) :
  angles 1 = 143 :=
by
  sorry

end smallest_angle_in_20_sided_polygon_is_143_l151_151884


namespace mnp_sum_correct_l151_151286

noncomputable def mnp_sum : ℕ :=
  let m := 1032
  let n := 40
  let p := 3
  m + n + p

theorem mnp_sum_correct : mnp_sum = 1075 := by
  -- Given the conditions, the established value for m, n, and p should sum to 1075
  sorry

end mnp_sum_correct_l151_151286


namespace paper_area_l151_151655

theorem paper_area (L W : ℝ) 
(h1 : 2 * L + W = 34) 
(h2 : L + 2 * W = 38) : 
L * W = 140 := by
  sorry

end paper_area_l151_151655


namespace solve_system_of_equations_solve_system_of_inequalities_l151_151745

-- For the system of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x + 4 * y = 2) (h2 : 2 * x - y = 5) : 
    x = 2 ∧ y = -1 :=
sorry

-- For the system of inequalities
theorem solve_system_of_inequalities (x : ℝ) 
    (h1 : x - 3 * (x - 1) < 7) 
    (h2 : x - 2 ≤ (2 * x - 3) / 3) :
    -2 < x ∧ x ≤ 3 :=
sorry

end solve_system_of_equations_solve_system_of_inequalities_l151_151745


namespace tan_diff_l151_151692

theorem tan_diff (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := 
by 
  sorry

end tan_diff_l151_151692


namespace square_diff_l151_151435

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l151_151435


namespace g_inv_eq_l151_151545

def g (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x - 5

theorem g_inv_eq (x : ℝ) (g_inv : ℝ → ℝ) (h_inv : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y) :
  (x = ( -1 + Real.sqrt 11 ) / 2) ∨ (x = ( -1 - Real.sqrt 11 ) / 2) :=
by
  -- proof omitted
  sorry

end g_inv_eq_l151_151545


namespace estimated_height_is_644_l151_151601

noncomputable def height_of_second_building : ℝ := 100
noncomputable def height_of_first_building : ℝ := 0.8 * height_of_second_building
noncomputable def height_of_third_building : ℝ := (height_of_first_building + height_of_second_building) - 20
noncomputable def height_of_fourth_building : ℝ := 1.15 * height_of_third_building
noncomputable def height_of_fifth_building : ℝ := 2 * |height_of_second_building - height_of_third_building|
noncomputable def total_estimated_height : ℝ := height_of_first_building + height_of_second_building + height_of_third_building + height_of_fourth_building + height_of_fifth_building

theorem estimated_height_is_644 : total_estimated_height = 644 := by
  sorry

end estimated_height_is_644_l151_151601


namespace smallest_angle_in_convex_20_gon_seq_l151_151883

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l151_151883


namespace find_a_of_normal_vector_l151_151758

theorem find_a_of_normal_vector (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y + 5 = 0) ∧ (∃ n : ℝ × ℝ, n = (a, a - 2)) → a = 6 := by
  sorry

end find_a_of_normal_vector_l151_151758


namespace total_muffins_correct_l151_151984

-- Define the conditions
def boys_count := 3
def muffins_per_boy := 12
def girls_count := 2
def muffins_per_girl := 20

-- Define the question and answer
def total_muffins_for_sale : Nat :=
  boys_count * muffins_per_boy + girls_count * muffins_per_girl

theorem total_muffins_correct :
  total_muffins_for_sale = 76 := by
  sorry

end total_muffins_correct_l151_151984


namespace floor_expression_correct_l151_151283

theorem floor_expression_correct :
  (∃ x : ℝ, x = 2007 ^ 3 / (2005 * 2006) - 2005 ^ 3 / (2006 * 2007) ∧ ⌊x⌋ = 8) := 
sorry

end floor_expression_correct_l151_151283


namespace cost_per_foot_of_fence_l151_151512

theorem cost_per_foot_of_fence 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h_area : area = 289) 
  (h_total_cost : total_cost = 4080) 
  : total_cost / (4 * (Real.sqrt area)) = 60 := 
by
  sorry

end cost_per_foot_of_fence_l151_151512


namespace distance_ratio_l151_151486

theorem distance_ratio (x : ℝ) (hx : abs x = 8) : abs (-4) / abs x = 1 / 2 :=
by {
  sorry
}

end distance_ratio_l151_151486


namespace pairs_nat_eq_l151_151148

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l151_151148


namespace largest_divisor_of_square_l151_151247

theorem largest_divisor_of_square (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n ^ 2) : 12 ∣ n := 
sorry

end largest_divisor_of_square_l151_151247


namespace square_distance_from_B_to_center_l151_151116

noncomputable def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

theorem square_distance_from_B_to_center :
  ∀ (a b : ℝ),
    (a^2 + (b + 8)^2 = 75) →
    ((a + 2)^2 + b^2 = 75) →
    distance_squared a b = 122 :=
by
  intros a b h1 h2
  sorry

end square_distance_from_B_to_center_l151_151116


namespace square_diff_l151_151437

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l151_151437


namespace maximize_a_n_l151_151315

-- Given sequence definition
noncomputable def a_n (n : ℕ) := (n + 2) * (7 / 8) ^ n

-- Prove that n = 5 or n = 6 maximizes the sequence
theorem maximize_a_n : ∃ n, (n = 5 ∨ n = 6) ∧ (∀ k, a_n k ≤ a_n n) :=
by
  sorry

end maximize_a_n_l151_151315


namespace parker_savings_l151_151931

-- Define the costs of individual items and meals
def burger_cost : ℝ := 5
def fries_cost : ℝ := 3
def drink_cost : ℝ := 3
def special_meal_cost : ℝ := 9.5
def kids_burger_cost : ℝ := 3
def kids_fries_cost : ℝ := 2
def kids_drink_cost : ℝ := 2
def kids_meal_cost : ℝ := 5

-- Define the number of meals Mr. Parker buys
def adult_meals : ℕ := 2
def kids_meals : ℕ := 2

-- Define the total cost of individual items for adults and children
def total_individual_cost_adults : ℝ :=
  adult_meals * (burger_cost + fries_cost + drink_cost)

def total_individual_cost_children : ℝ :=
  kids_meals * (kids_burger_cost + kids_fries_cost + kids_drink_cost)

-- Define the total cost of meal deals
def total_meals_cost : ℝ :=
  adult_meals * special_meal_cost + kids_meals * kids_meal_cost

-- Define the total cost of individual items for both adults and children
def total_individual_cost : ℝ :=
  total_individual_cost_adults + total_individual_cost_children

-- Define the savings
def savings : ℝ := total_individual_cost - total_meals_cost

theorem parker_savings : savings = 7 :=
by
  sorry

end parker_savings_l151_151931


namespace blocks_before_jess_turn_l151_151455

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_l151_151455


namespace tutors_schedule_l151_151817

theorem tutors_schedule :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_schedule_l151_151817


namespace sum_between_9p5_and_10_l151_151080

noncomputable def sumMixedNumbers : ℚ :=
  (29 / 9) + (11 / 4) + (81 / 20)

theorem sum_between_9p5_and_10 :
  9.5 < sumMixedNumbers ∧ sumMixedNumbers < 10 :=
by
  sorry

end sum_between_9p5_and_10_l151_151080


namespace Ann_end_blocks_l151_151668

-- Define blocks Ann initially has and finds
def initialBlocksAnn : ℕ := 9
def foundBlocksAnn : ℕ := 44

-- Define blocks Ann ends with
def finalBlocksAnn : ℕ := initialBlocksAnn + foundBlocksAnn

-- The proof goal
theorem Ann_end_blocks : finalBlocksAnn = 53 := by
  -- Use sorry to skip the proof
  sorry

end Ann_end_blocks_l151_151668


namespace domain_of_f_l151_151391

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / (Real.sqrt (x - 7))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = f x } = {x : ℝ | x > 7} :=
by
  sorry

end domain_of_f_l151_151391


namespace equation_of_circle_l151_151612

-- Defining the problem conditions directly
variables (a : ℝ) (x y: ℝ)

-- Assume a ≠ 0
variable (h : a ≠ 0)

-- Prove that the circle passing through the origin with center (a, a) has the equation (x - a)^2 + (y - a)^2 = 2a^2.
theorem equation_of_circle (h : a ≠ 0) :
  (x - a)^2 + (y - a)^2 = 2 * a^2 :=
sorry

end equation_of_circle_l151_151612


namespace find_x_l151_151771

def x : ℕ := 70

theorem find_x :
  x + (5 * 12) / (180 / 3) = 71 :=
by
  sorry

end find_x_l151_151771


namespace solve_for_y_l151_151627

theorem solve_for_y (y : ℝ) : (10 - y) ^ 2 = 4 * y ^ 2 → y = 10 / 3 ∨ y = -10 :=
by
  intro h
  -- The proof steps would go here, but we include sorry to allow for compilation.
  sorry

end solve_for_y_l151_151627


namespace max_min_M_l151_151961

noncomputable def M (x y : ℝ) : ℝ :=
  abs (x + y) + abs (y + 1) + abs (2 * y - x - 4)

theorem max_min_M (x y : ℝ) (hx : abs x ≤ 1) (hy : abs y ≤ 1) :
  3 ≤ M x y ∧ M x y ≤ 7 :=
sorry

end max_min_M_l151_151961


namespace part1_part2_l151_151966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 - (4 / (2 * a^x + a))

theorem part1 (h₁ : ∀ x, f a x = -f a (-x)) (h₂ : a > 0) (h₃ : a ≠ 1) : a = 2 :=
  sorry

theorem part2 (h₁ : a = 2) (x : ℝ) (hx : 0 < x ∧ x ≤ 1) (t : ℝ) :
  t * (f a x) ≥ 2^x - 2 ↔ t ≥ 0 :=
  sorry

end part1_part2_l151_151966


namespace cylinder_surface_area_l151_151913

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) 
  (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (circumference_formula : c = 2 * Real.pi * r) : 
  2 * (Real.pi * r^2) + (2 * Real.pi * r * h) = 6 * Real.pi := 
by
  sorry

end cylinder_surface_area_l151_151913


namespace expression_negativity_l151_151302

-- Given conditions: a, b, and c are lengths of the sides of a triangle
variables (a b c : ℝ)
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- To prove: (a - b)^2 - c^2 < 0
theorem expression_negativity (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a - b)^2 - c^2 < 0 :=
sorry

end expression_negativity_l151_151302


namespace geometric_sequence_property_l151_151451

open Classical

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property :
  ∃ (a : ℕ → ℝ) (q : ℝ), q < 0 ∧ geometric_sequence a q ∧
    a 1 = 1 - a 0 ∧ a 3 = 4 - a 2 ∧ a 3 + a 4 = -8 :=
by
  sorry

end geometric_sequence_property_l151_151451


namespace train_ride_duration_is_360_minutes_l151_151339

-- Define the conditions given in the problem
def arrived_at_station_at_8 (t : ℕ) : Prop := t = 8 * 60
def train_departed_at_835 (t_depart : ℕ) : Prop := t_depart = 8 * 60 + 35
def train_arrived_at_215 (t_arrive : ℕ) : Prop := t_arrive = 14 * 60 + 15
def exited_station_at_3 (t_exit : ℕ) : Prop := t_exit = 15 * 60

-- Define the problem statement
theorem train_ride_duration_is_360_minutes (boarding alighting : ℕ) :
  arrived_at_station_at_8 boarding ∧ 
  train_departed_at_835 boarding ∧ 
  train_arrived_at_215 alighting ∧ 
  exited_station_at_3 alighting → 
  alighting - boarding = 360 := 
by
  sorry

end train_ride_duration_is_360_minutes_l151_151339


namespace min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l151_151105

-- Define the first problem
theorem min_cuts_for_eleven_day_stay : 
  (∀ (chain_len num_days : ℕ), chain_len = 11 ∧ num_days = 11 
  → (∃ (cuts : ℕ), cuts = 2)) := 
sorry

-- Define the second problem
theorem max_days_with_n_cuts : 
  (∀ (n chain_len days : ℕ), chain_len = (n + 1) * 2 ^ n - 1 
  → days = (n + 1) * 2 ^ n - 1) := 
sorry

end min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l151_151105


namespace bond_value_after_8_years_l151_151721

theorem bond_value_after_8_years (r t1 t2 : ℕ) (A1 A2 P : ℚ) :
  r = 4 / 100 ∧ t1 = 3 ∧ t2 = 8 ∧ A1 = 560 ∧ A1 = P * (1 + r * t1) 
  → A2 = P * (1 + r * t2) ∧ A2 = 660 :=
by
  intro h
  obtain ⟨hr, ht1, ht2, hA1, hA1eq⟩ := h
  -- Proof needs to be filled in here
  sorry

end bond_value_after_8_years_l151_151721


namespace max_profit_price_range_for_minimum_profit_l151_151911

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-10 * x^2 + 2000 * x - 84000

theorem max_profit :
  ∃ x, (∀ x₀, x₀ ≠ x → functional_relationship x₀ < functional_relationship x) ∧
  functional_relationship x = 16000 := 
sorry

theorem price_range_for_minimum_profit :
  ∀ (x : ℝ), 
  -10 * (x - 100)^2 + 16000 - 1750 ≥ 12000 → 
  85 ≤ x ∧ x ≤ 115 :=
sorry

end max_profit_price_range_for_minimum_profit_l151_151911


namespace square_area_l151_151473

theorem square_area (x : ℝ) (h1 : BG = GH) (h2 : GH = HD) (h3 : BG = 20 * Real.sqrt 2) : x = 40 * Real.sqrt 2 → x^2 = 3200 :=
by
  sorry

end square_area_l151_151473


namespace triangle_balls_l151_151636

theorem triangle_balls (n : ℕ) (num_tri_balls : ℕ) (num_sq_balls : ℕ) :
  (∀ n : ℕ, num_tri_balls = n * (n + 1) / 2)
  ∧ (num_sq_balls = num_tri_balls + 424)
  ∧ (∀ s : ℕ, s = n - 8 → s * s = num_sq_balls)
  → num_tri_balls = 820 :=
by sorry

end triangle_balls_l151_151636


namespace calculation_correctness_l151_151808

theorem calculation_correctness : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end calculation_correctness_l151_151808


namespace min_possible_value_box_l151_151014

theorem min_possible_value_box :
  ∃ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15 ∧ a^2 + b^2 = 61) ∧
  ∀ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15) → (a^2 + b^2 ≥ 61) :=
by {
  sorry
}

end min_possible_value_box_l151_151014


namespace crayons_selection_l151_151493

theorem crayons_selection (n k : ℕ) (h_n : n = 15) (h_k : k = 5) :
  Nat.choose n k = 3003 := 
by
  rw [h_n, h_k]
  rfl

end crayons_selection_l151_151493


namespace distance_from_Q_to_EF_is_24_div_5_l151_151285

-- Define the configuration of the square and points
def E := (0, 8)
def F := (8, 8)
def G := (8, 0)
def H := (0, 0)
def N := (4, 0) -- Midpoint of GH
def r1 := 4 -- Radius of the circle centered at N
def r2 := 8 -- Radius of the circle centered at E

-- Definition of the first circle centered at N with radius r1
def circle1 (x y : ℝ) := (x - 4)^2 + y^2 = r1^2

-- Definition of the second circle centered at E with radius r2
def circle2 (x y : ℝ) := x^2 + (y - 8)^2 = r2^2

-- Define the intersection point Q, other than H
def Q := (32 / 5, 16 / 5) -- Found as an intersection point between circle1 and circle2

-- Define the distance from point Q to the line EF
def dist_to_EF := 8 - (Q.2) -- (Q.2 is the y-coordinate of Q)

-- The main statement to prove
theorem distance_from_Q_to_EF_is_24_div_5 : dist_to_EF = 24 / 5 := by
  sorry

end distance_from_Q_to_EF_is_24_div_5_l151_151285


namespace problem_1_problem_2_l151_151757

theorem problem_1 
  (h1 : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : 
  Int.floor (5 - Real.sqrt 2) = 3 :=
sorry

theorem problem_2 
  (h2 : Real.sqrt 3 > 1) : 
  abs (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1 :=
sorry

end problem_1_problem_2_l151_151757


namespace total_seeds_correct_l151_151475

def seeds_per_bed : ℕ := 6
def flower_beds : ℕ := 9
def total_seeds : ℕ := seeds_per_bed * flower_beds

theorem total_seeds_correct : total_seeds = 54 := by
  sorry

end total_seeds_correct_l151_151475


namespace geese_flew_away_l151_151357

theorem geese_flew_away (initial remaining flown_away : ℕ) (h_initial: initial = 51) (h_remaining: remaining = 23) : flown_away = 28 :=
by
  sorry

end geese_flew_away_l151_151357


namespace solution_set_l151_151389

def op (a b : ℝ) : ℝ := -2 * a + b

theorem solution_set (x : ℝ) : (op x 4 > 0) ↔ (x < 2) :=
by {
  -- proof required here
  sorry
}

end solution_set_l151_151389


namespace exam_maximum_marks_l151_151861

theorem exam_maximum_marks :
  (∃ M S E : ℕ, 
    (90 + 20 = 40 * M / 100) ∧ 
    (110 + 35 = 35 * S / 100) ∧ 
    (80 + 10 = 30 * E / 100) ∧ 
    M = 275 ∧ 
    S = 414 ∧ 
    E = 300) :=
by
  sorry

end exam_maximum_marks_l151_151861


namespace correct_operation_l151_151772

theorem correct_operation : (3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3) ∧ 
                            ¬(a^2 * a^3 = a^6) ∧ 
                            ¬(a^6 / a^2 = a^3) ∧ 
                            ¬((a^2)^3 = a^5) :=
by
  sorry

end correct_operation_l151_151772


namespace sqrt_subtraction_l151_151098

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l151_151098


namespace negation_proof_l151_151614

theorem negation_proof (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_proof_l151_151614


namespace find_term_number_l151_151616

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem find_term_number
  (a₁ : ℤ)
  (d : ℤ)
  (n : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 3)
  (h₂ : arithmetic_sequence a₁ d n = 2011) :
  n = 671 :=
  sorry

end find_term_number_l151_151616


namespace max_workers_l151_151792

variable {n : ℕ} -- number of workers on the smaller field
variable {S : ℕ} -- area of the smaller field
variable (a : ℕ) -- productivity of each worker

theorem max_workers 
  (h_area : ∀ large small : ℕ, large = 2 * small) 
  (h_workers : ∀ large small : ℕ, large = small + 4) 
  (h_inequality : ∀ (S : ℕ) (n a : ℕ), S / (a * n) > (2 * S) / (a * (n + 4))) :
  2 * n + 4 ≤ 10 :=
by
  -- h_area implies the area requirement
  -- h_workers implies the worker requirement
  -- h_inequality implies the time requirement
  sorry

end max_workers_l151_151792


namespace Bill_original_profit_percentage_l151_151273

theorem Bill_original_profit_percentage 
  (S : ℝ) 
  (h_S : S = 879.9999999999993) 
  (h_cond : ∀ (P : ℝ), 1.17 * P = S + 56) :
  ∃ (profit_percentage : ℝ), profit_percentage = 10 := 
by
  sorry

end Bill_original_profit_percentage_l151_151273


namespace digit_B_divisible_by_9_l151_151069

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l151_151069


namespace fraction_paint_used_second_week_l151_151456

noncomputable def total_paint : ℕ := 360
noncomputable def paint_used_first_week : ℕ := total_paint / 4
noncomputable def remaining_paint_after_first_week : ℕ := total_paint - paint_used_first_week
noncomputable def total_paint_used : ℕ := 135
noncomputable def paint_used_second_week : ℕ := total_paint_used - paint_used_first_week
noncomputable def remaining_paint_after_first_week_fraction : ℚ := paint_used_second_week / remaining_paint_after_first_week

theorem fraction_paint_used_second_week : remaining_paint_after_first_week_fraction = 1 / 6 := by
  sorry

end fraction_paint_used_second_week_l151_151456


namespace boat_speed_in_still_water_l151_151524

theorem boat_speed_in_still_water
  (speed_of_stream : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ)
  (h1: speed_of_stream = 5)
  (h2: time_downstream = 2)
  (h3: distance_downstream = 54)
  (h4: effective_speed = boat_speed + speed_of_stream)
  (h5: distance_downstream = effective_speed * time_downstream) :
  boat_speed = 22 := by
  sorry

end boat_speed_in_still_water_l151_151524


namespace exchanges_divisible_by_26_l151_151338

variables (p a d : ℕ) -- Define the variables for the number of exchanges

theorem exchanges_divisible_by_26 (t : ℕ) (h1 : p = 4 * a + d) (h2 : p = a + 5 * d) :
  ∃ k : ℕ, a + p + d = 26 * k :=
by {
  -- Replace these sorry placeholders with the actual proof where needed
  sorry
}

end exchanges_divisible_by_26_l151_151338


namespace a_and_b_are_kth_powers_l151_151726

theorem a_and_b_are_kth_powers (k : ℕ) (h_k : 1 < k) (a b : ℤ) (h_rel_prime : Int.gcd a b = 1)
  (c : ℤ) (h_ab_power : a * b = c^k) : ∃ (m n : ℤ), a = m^k ∧ b = n^k :=
by
  sorry

end a_and_b_are_kth_powers_l151_151726


namespace triangle_ab_value_l151_151165

theorem triangle_ab_value (a b c : ℝ) (A B C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by
  sorry

end triangle_ab_value_l151_151165


namespace cistern_empty_time_l151_151369

noncomputable def time_to_empty_cistern (fill_no_leak_time fill_with_leak_time : ℝ) (filled_cistern : ℝ) : ℝ :=
  let R := filled_cistern / fill_no_leak_time
  let L := (R - filled_cistern / fill_with_leak_time)
  filled_cistern / L

theorem cistern_empty_time :
  time_to_empty_cistern 12 14 1 = 84 :=
by
  unfold time_to_empty_cistern
  simp
  sorry

end cistern_empty_time_l151_151369


namespace problem_solution_l151_151412

noncomputable def coordinates_of_vertex_C (A : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  let t := -4 in
  (t, -t)

noncomputable def area_of_triangle (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) : ℝ :=
  let d := (|4 + 1 + 4|) / (Real.sqrt (4 + 1)) in
  let bc := (Real.sqrt ((-2 + 4)^2 + (0 - 4)^2)) in
  1 / 2 * d * 2 * bc

theorem problem_solution (A B C : ℝ × ℝ) :
  A = (2, 1) ∧ C = coordinates_of_vertex_C A (-1) ∧ area_of_triangle A B C = 9 := by
  sorry

end problem_solution_l151_151412


namespace distribute_balls_into_boxes_l151_151011

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l151_151011


namespace fraction_at_x_eq_4571_div_39_l151_151937

def numerator (x : ℕ) : ℕ := x^6 - 16 * x^3 + x^2 + 64
def denominator (x : ℕ) : ℕ := x^3 - 8

theorem fraction_at_x_eq_4571_div_39 : numerator 5 / denominator 5 = 4571 / 39 :=
by
  sorry

end fraction_at_x_eq_4571_div_39_l151_151937


namespace min_value_of_abc_l151_151443

noncomputable def minimum_value_abc (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ ((a + c) * (a + b) = 6 - 2 * Real.sqrt 5) → (2 * a + b + c ≥ 2 * Real.sqrt 5 - 2)

theorem min_value_of_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) : 
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by 
  sorry

end min_value_of_abc_l151_151443


namespace range_of_a_fixed_point_l151_151962

open Function

def f (x a : ℝ) := x^3 - a * x

theorem range_of_a (a : ℝ) (h1 : 0 < a) : 0 < a ∧ a ≤ 3 ↔ ∀ x ≥ 1, 3 * x^2 - a > 0 :=
sorry

theorem fixed_point (a x0 : ℝ) (h_a : 0 < a) (h_b : a ≤ 3)
  (h1 : x0 ≥ 1) (h2 : f x0 a ≥ 1) (h3 : f (f x0 a) a = x0) (strict_incr : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f x a < f y a) :
  f x0 a = x0 :=
sorry

end range_of_a_fixed_point_l151_151962


namespace margo_donation_l151_151596

variable (M J : ℤ)

theorem margo_donation (h1: J = 4700) (h2: (|J - M| / 2) = 200) : M = 4300 :=
sorry

end margo_donation_l151_151596


namespace problem_l151_151420

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l151_151420


namespace find_range_of_m_l151_151163

variable (m : ℝ)

-- Definition of p: There exists x in ℝ such that mx^2 - mx + 1 < 0
def p : Prop := ∃ x : ℝ, m * x ^ 2 - m * x + 1 < 0

-- Definition of q: The curve of the equation (x^2)/(m-1) + (y^2)/(3-m) = 1 is a hyperbola
def q : Prop := (m - 1) * (3 - m) < 0

-- Given conditions
def proposition_and : Prop := ¬ (p m ∧ q m)
def proposition_or : Prop := p m ∨ q m

-- Final theorem statement
theorem find_range_of_m : proposition_and m ∧ proposition_or m → (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4) :=
sorry

end find_range_of_m_l151_151163


namespace baseball_team_wins_more_than_three_times_losses_l151_151795

theorem baseball_team_wins_more_than_three_times_losses
    (total_games : ℕ)
    (wins : ℕ)
    (losses : ℕ)
    (h1 : total_games = 130)
    (h2 : wins = 101)
    (h3 : wins + losses = total_games) :
    wins - 3 * losses = 14 :=
by
    -- Proof goes here
    sorry

end baseball_team_wins_more_than_three_times_losses_l151_151795


namespace fixed_point_through_ellipse_l151_151172

-- Define the ellipse and the points
def C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def P2 : ℝ × ℝ := (0, 1)

-- Define the condition for a line not passing through P2 and intersecting the ellipse
def line_l_intersects_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ (x1 x2 b k : ℝ), l (x1, k * x1 + b) ∧ l (x2, k * x2 + b) ∧
  (C x1 (k * x1 + b)) ∧ (C x2 (k * x2 + b)) ∧
  ((x1, k * x1 + b) ≠ P2 ∧ (x2, k * x2 + b) ≠ P2) ∧
  ((k * x1 + b ≠ 1) ∧ (k * x2 + b ≠ 1)) ∧ 
  (∃ (kA kB : ℝ), kA = (k * x1 + b - 1) / x1 ∧ kB = (k * x2 + b - 1) / x2 ∧ kA + kB = -1)

-- Prove there exists a fixed point (2, -1) through which all such lines must pass
theorem fixed_point_through_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_l_intersects_ellipse A B l → l (2, -1) :=
sorry

end fixed_point_through_ellipse_l151_151172


namespace range_a_l151_151173

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_a (H : ∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → (f x1 a - f x2 a) / (x1 - x2) > 0) : a ≤ 2 := 
sorry

end range_a_l151_151173


namespace solve_abs_inequality_l151_151297

theorem solve_abs_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8) :=
sorry

end solve_abs_inequality_l151_151297


namespace eventB_is_not_random_l151_151513

def eventA := "The sun rises in the east and it rains in the west"
def eventB := "It's not cold when it snows but cold when it melts"
def eventC := "It rains continuously during the Qingming festival"
def eventD := "It's sunny every day when the plums turn yellow"

def is_random_event (event : String) : Prop :=
  event = eventA ∨ event = eventC ∨ event = eventD

theorem eventB_is_not_random : ¬ is_random_event eventB :=
by
  unfold is_random_event
  sorry

end eventB_is_not_random_l151_151513


namespace two_mathematicians_contemporaries_l151_151501

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l151_151501


namespace sqrt_subtraction_l151_151099

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l151_151099


namespace find_x_l151_151411

theorem find_x
    (x : ℝ)
    (l : ℝ := 4 * x)
    (w : ℝ := x + 8)
    (area_eq_twice_perimeter : l * w = 2 * (2 * l + 2 * w)) :
    x = 2 :=
by
  sorry

end find_x_l151_151411


namespace minimize_pollution_park_distance_l151_151925

noncomputable def pollution_index (x : ℝ) : ℝ :=
  (1 / x) + (4 / (30 - x))

theorem minimize_pollution_park_distance : ∃ x : ℝ, (0 < x ∧ x < 30) ∧ pollution_index x = 10 :=
by
  sorry

end minimize_pollution_park_distance_l151_151925


namespace speed_of_current_is_6_l151_151617

noncomputable def speed_of_current : ℝ :=
  let Vm := 18  -- speed in still water in kmph
  let distance_m := 100  -- distance covered in meters
  let time_s := 14.998800095992323  -- time taken in seconds
  let distance_km := distance_m / 1000  -- converting distance to kilometers
  let time_h := time_s / 3600  -- converting time to hours
  let Vd := distance_km / time_h  -- speed downstream in kmph
  Vd - Vm  -- speed of the current

theorem speed_of_current_is_6 :
  speed_of_current = 6 := by
  sorry -- proof is skipped

end speed_of_current_is_6_l151_151617


namespace total_apples_l151_151981

theorem total_apples (x : ℕ) : 
    (x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50) -> 
    x = 3360 :=
by
    sorry

end total_apples_l151_151981


namespace problem_statement_l151_151371

-- Define the repeating decimal 0.000272727... as x
noncomputable def repeatingDecimal : ℚ := 3 / 11000

-- Define the given condition for the question
def decimalRepeatsIndefinitely : Prop := 
  repeatingDecimal = 0.0002727272727272727  -- Representation for repeating decimal

-- Definitions of large powers of 10
def ten_pow_5 := 10^5
def ten_pow_3 := 10^3

-- The problem statement
theorem problem_statement : decimalRepeatsIndefinitely →
  (ten_pow_5 - ten_pow_3) * repeatingDecimal = 27 :=
sorry

end problem_statement_l151_151371


namespace solve_eq_l151_151938

theorem solve_eq (a b : ℕ) : a * a = b * (b + 7) ↔ (a, b) = (0, 0) ∨ (a, b) = (12, 9) :=
by
  sorry

end solve_eq_l151_151938


namespace red_paint_intensity_l151_151876

variable (I : ℝ) -- Intensity of the original paint
variable (P : ℝ) -- Volume of the original paint
variable (fraction_replaced : ℝ := 1) -- Fraction of original paint replaced
variable (new_intensity : ℝ := 20) -- New paint intensity
variable (replacement_intensity : ℝ := 20) -- Replacement paint intensity

theorem red_paint_intensity : new_intensity = replacement_intensity :=
by
  -- Placeholder for the actual proof
  sorry

end red_paint_intensity_l151_151876


namespace al_original_portion_l151_151922

variables (a b c d : ℝ)

theorem al_original_portion :
  a + b + c + d = 1200 →
  a - 150 + 2 * b + 2 * c + 3 * d = 1800 →
  a = 450 :=
by
  intros h1 h2
  sorry

end al_original_portion_l151_151922


namespace point_on_curve_iff_F_eq_zero_l151_151987

variable (F : ℝ → ℝ → ℝ)
variable (a b : ℝ)

theorem point_on_curve_iff_F_eq_zero :
  (F a b = 0) ↔ (∃ P : ℝ × ℝ, P = (a, b) ∧ F P.1 P.2 = 0) :=
by
  sorry

end point_on_curve_iff_F_eq_zero_l151_151987


namespace parallelogram_s_value_l151_151260

noncomputable def parallelogram_area (s : ℝ) : ℝ :=
  s * 2 * (s / Real.sqrt 2)

theorem parallelogram_s_value (s : ℝ) (h₀ : parallelogram_area s = 8 * Real.sqrt 2) : 
  s = 2 * Real.sqrt 2 :=
by
  sorry

end parallelogram_s_value_l151_151260


namespace car_r_speed_l151_151109

variable (v : ℝ)

theorem car_r_speed (h1 : (300 / v - 2 = 300 / (v + 10))) : v = 30 := 
sorry

end car_r_speed_l151_151109


namespace non_negative_integers_abs_less_than_3_l151_151351

theorem non_negative_integers_abs_less_than_3 :
  { x : ℕ | x < 3 } = {0, 1, 2} :=
by
  sorry

end non_negative_integers_abs_less_than_3_l151_151351


namespace Xiaoliang_catches_up_in_h_l151_151515

-- Define the speeds and head start
def speed_Xiaobin : ℝ := 4  -- Xiaobin's speed in km/h
def speed_Xiaoliang : ℝ := 12  -- Xiaoliang's speed in km/h
def head_start : ℝ := 6  -- Xiaobin's head start in hours

-- Define the additional distance Xiaoliang needs to cover
def additional_distance : ℝ := speed_Xiaobin * head_start

-- Define the hourly distance difference between them
def speed_difference : ℝ := speed_Xiaoliang - speed_Xiaobin

-- Prove that Xiaoliang will catch up with Xiaobin in exactly 3 hours
theorem Xiaoliang_catches_up_in_h : (additional_distance / speed_difference) = 3 :=
by
  sorry

end Xiaoliang_catches_up_in_h_l151_151515


namespace candy_difference_l151_151688

theorem candy_difference (frankie_candies : ℕ) (max_candies : ℕ) (h1 : frankie_candies = 74) (h2 : max_candies = 92) : max_candies - frankie_candies = 18 := by
  sorry

end candy_difference_l151_151688


namespace train_length_proof_l151_151804

/-- Given a train's speed of 45 km/hr, time to cross a bridge of 30 seconds, and the bridge length of 225 meters, prove that the length of the train is 150 meters. -/
theorem train_length_proof (speed_km_hr : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ)
    (h_speed : speed_km_hr = 45) (h_time : time_sec = 30) (h_bridge_length : bridge_length_m = 225) :
  train_length_m = 150 :=
by
  sorry

end train_length_proof_l151_151804


namespace exists_irreducible_fractions_sum_to_86_over_111_l151_151956

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l151_151956


namespace Dima_impossible_cut_l151_151519

theorem Dima_impossible_cut (n : ℕ) 
  (h1 : n % 5 = 0) 
  (h2 : n % 7 = 0) 
  (h3 : n ≤ 200) : ¬(n % 6 = 0) :=
sorry

end Dima_impossible_cut_l151_151519


namespace regular_21_gon_symmetry_calculation_l151_151811

theorem regular_21_gon_symmetry_calculation:
  let L := 21
  let R := 360 / 21
  L + R = 38 :=
by
  sorry

end regular_21_gon_symmetry_calculation_l151_151811


namespace count_valid_triples_l151_151159

def S (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c ∧ 
  (a + b + c = 2005) ∧ (S a + S b + S c = 61)

def number_of_valid_triples : ℕ := sorry

theorem count_valid_triples : number_of_valid_triples = 17160 :=
sorry

end count_valid_triples_l151_151159


namespace determine_a_l151_151402

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem determine_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3/4 :=
by
  sorry

end determine_a_l151_151402


namespace elise_saving_correct_l151_151818

-- Definitions based on the conditions
def initial_money : ℤ := 8
def spent_comic_book : ℤ := 2
def spent_puzzle : ℤ := 18
def final_money : ℤ := 1

-- The theorem to prove the amount saved
theorem elise_saving_correct (x : ℤ) : 
  initial_money + x - spent_comic_book - spent_puzzle = final_money → x = 13 :=
by
  sorry

end elise_saving_correct_l151_151818


namespace correct_survey_method_l151_151514

def service_life_of_light_tubes (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def viewership_rate_of_spring_festival_gala (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def crash_resistance_of_cars (survey_method : String) : Prop :=
  survey_method = "sample"

def fastest_student_for_sports_meeting (survey_method : String) : Prop :=
  survey_method = "sample"

theorem correct_survey_method :
  ¬(service_life_of_light_tubes "comprehensive") ∧
  ¬(viewership_rate_of_spring_festival_gala "comprehensive") ∧
  ¬(crash_resistance_of_cars "sample") ∧
  (fastest_student_for_sports_meeting "sample") :=
sorry

end correct_survey_method_l151_151514


namespace find_number_l151_151786

theorem find_number (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 :=
sorry

end find_number_l151_151786


namespace compound_interest_rate_l151_151923

theorem compound_interest_rate :
  ∃ r : ℝ, (1000 * (1 + r)^3 = 1331.0000000000005) ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l151_151923


namespace cylinder_volume_ratio_l151_151022

variable (h r : ℝ)

theorem cylinder_volume_ratio (h r : ℝ) :
  let V_original := π * r^2 * h
  let h_new := 2 * h
  let r_new := 4 * r
  let V_new := π * (r_new)^2 * h_new
  V_new = 32 * V_original :=
by
  sorry

end cylinder_volume_ratio_l151_151022


namespace pentomino_symmetry_count_l151_151180

def is_pentomino (shape : Type) : Prop :=
  -- Define the property of being a pentomino as composed of five squares edge to edge
  sorry

def has_reflectional_symmetry (shape : Type) : Prop :=
  -- Define the property of having at least one line of reflectional symmetry
  sorry

def has_rotational_symmetry_of_order_2 (shape : Type) : Prop :=
  -- Define the property of having rotational symmetry of order 2 (180 degrees rotation results in the same shape)
  sorry

noncomputable def count_valid_pentominoes : Nat :=
  -- Assume that we have a list of 18 pentominoes
  -- Count the number of pentominoes that meet both criteria
  sorry

theorem pentomino_symmetry_count :
  count_valid_pentominoes = 4 :=
sorry

end pentomino_symmetry_count_l151_151180


namespace factorize_a_cube_minus_nine_a_l151_151822

theorem factorize_a_cube_minus_nine_a (a : ℝ) : a^3 - 9 * a = a * (a + 3) * (a - 3) :=
by sorry

end factorize_a_cube_minus_nine_a_l151_151822


namespace mathematicians_contemporaries_probability_l151_151499

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l151_151499


namespace lowest_test_score_dropped_is_35_l151_151782

theorem lowest_test_score_dropped_is_35 
  (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : min A (min B (min C D)) = D)
  (h3 : (A + B + C) / 3 = 55) : 
  D = 35 := by
  sorry

end lowest_test_score_dropped_is_35_l151_151782


namespace ratio_of_constants_l151_151643

theorem ratio_of_constants (a b c: ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : c = b / a) : c = 1 / 16 :=
by sorry

end ratio_of_constants_l151_151643


namespace michael_card_count_l151_151869

variable (Lloyd Mark Michael : ℕ)
variable (L : ℕ)

-- Conditions from the problem
axiom condition1 : Mark = 3 * Lloyd
axiom condition2 : Mark + 10 = Michael
axiom condition3 : Lloyd + Mark + (Michael + 80) = 300

-- The correct answer we want to prove
theorem michael_card_count : Michael = 100 :=
by
  -- Proof will be here.
  sorry

end michael_card_count_l151_151869


namespace both_true_of_neg_and_false_l151_151023

variable (P Q : Prop)

theorem both_true_of_neg_and_false (h : ¬ (P ∧ Q) = False) : P ∧ Q :=
by
  -- Proof goes here
  sorry

end both_true_of_neg_and_false_l151_151023


namespace sum_digits_l151_151583

def distinct_digits (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d)

def valid_equation (Y E M T : ℕ) : Prop :=
  ∃ (YE ME TTT : ℕ),
    YE = Y * 10 + E ∧
    ME = M * 10 + E ∧
    TTT = T * 111 ∧
    YE < ME ∧
    YE * ME = TTT ∧
    distinct_digits Y E M T

theorem sum_digits (Y E M T : ℕ) :
  valid_equation Y E M T → Y + E + M + T = 21 := 
sorry

end sum_digits_l151_151583


namespace probability_diff_colors_l151_151495

def num_blue : ℕ := 5
def num_red : ℕ := 4
def num_yellow : ℕ := 3
def total_chips : ℕ := num_blue + num_red + num_yellow
def prob_diff_color : ℚ := (num_blue * (num_red + num_yellow) + num_red * (num_blue + num_yellow) + num_yellow * (num_blue + num_red)) / (total_chips * total_chips)

theorem probability_diff_colors : prob_diff_color = 47 / 72 := 
by 
  sorry

end probability_diff_colors_l151_151495


namespace true_proposition_l151_151731

def proposition_p := ∀ (x : ℤ), x^2 > x
def proposition_q := ∃ (x : ℝ) (hx : x > 0), x + (2 / x) > 4

theorem true_proposition :
  (¬ proposition_p) ∨ proposition_q :=
by
  sorry

end true_proposition_l151_151731


namespace lowest_positive_integer_divisible_by_primes_between_10_and_50_l151_151684

def primes_10_to_50 : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def lcm_list (lst : List ℕ) : ℕ :=
lst.foldr Nat.lcm 1

theorem lowest_positive_integer_divisible_by_primes_between_10_and_50 :
  lcm_list primes_10_to_50 = 614889782588491410 :=
by
  sorry

end lowest_positive_integer_divisible_by_primes_between_10_and_50_l151_151684


namespace sum_of_ages_l151_151494

theorem sum_of_ages (a b : ℕ) :
  let c1 := a
  let c2 := a + 2
  let c3 := a + 4
  let c4 := a + 6
  let coach1 := b
  let coach2 := b + 2
  c1^2 + c2^2 + c3^2 + c4^2 + coach1^2 + coach2^2 = 2796 →
  c1 + c2 + c3 + c4 + coach1 + coach2 = 106 :=
by
  intro h
  sorry

end sum_of_ages_l151_151494


namespace seating_arrangements_l151_151566

theorem seating_arrangements (n : ℕ) (max_capacity : ℕ) 
  (h_n : n = 6) (h_max : max_capacity = 4) :
  ∃ k : ℕ, k = 50 :=
by
  sorry

end seating_arrangements_l151_151566


namespace probability_exactly_3_common_l151_151870

open BigOperators

theorem probability_exactly_3_common (S : Finset ℕ) (hS : S.card = 12) :
  let books : Finset (Finset ℕ) := S.powerset.filter (λ s, s.card = 6)
  ∃ p : ℚ, p = 100 / 231 ∧ 
  ∑ H in books, ∑ B in books, if (H ∩ B).card = 3 then 1 else 0 = 
  p * ∑ H in books, ∑ B in books, 1 :=
by 
  sorry

end probability_exactly_3_common_l151_151870


namespace f_zero_f_odd_f_not_decreasing_f_increasing_l151_151257

noncomputable def f (x : ℝ) : ℝ := sorry -- The function definition is abstract.

-- Functional equation condition
axiom functional_eq (x y : ℝ) (h1 : -1 < x) (h2 : x < 1) (h3 : -1 < y) (h4 : y < 1) : 
  f x + f y = f ((x + y) / (1 + x * y))

-- Condition for negative interval
axiom neg_interval (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : f x < 0

-- Statements to prove

-- a): f(0) = 0
theorem f_zero : f 0 = 0 := 
by
  sorry

-- b): f(x) is an odd function
theorem f_odd (x : ℝ) (h1 : -1 < x) (h2 : x < 1) : f (-x) = -f x := 
by
  sorry

-- c): f(x) is not a decreasing function
theorem f_not_decreasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : ¬(f x1 > f x2) :=
by
  sorry

-- d): f(x) is an increasing function
theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : f x1 < f x2 :=
by
  sorry

end f_zero_f_odd_f_not_decreasing_f_increasing_l151_151257


namespace distinct_shell_arrangements_l151_151865

/--
John draws a regular five pointed star and places one of ten different sea shells at each of the 5 outward-pointing points and 5 inward-pointing points. 
Considering rotations and reflections of an arrangement as equivalent, prove that the number of ways he can place the shells is 362880.
-/
theorem distinct_shell_arrangements : 
  let total_arrangements := Nat.factorial 10
  let symmetries := 10
  total_arrangements / symmetries = 362880 :=
by
  sorry

end distinct_shell_arrangements_l151_151865


namespace find_divisor_l151_151085

theorem find_divisor (d : ℕ) : (55 / d) + 10 = 21 → d = 5 :=
by 
  sorry

end find_divisor_l151_151085


namespace min_2a_plus_3b_l151_151168

theorem min_2a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_parallel : (a * (b - 3) - 2 * b = 0)) :
  (2 * a + 3 * b) = 25 :=
by
  -- proof goes here
  sorry

end min_2a_plus_3b_l151_151168


namespace option_c_not_equivalent_l151_151548

theorem option_c_not_equivalent :
  ¬ (785 * 10^(-9) = 7.845 * 10^(-6)) :=
by
  sorry

end option_c_not_equivalent_l151_151548


namespace inv_matrix_eq_l151_151840

variable (a : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 1, a])
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![a, -3; -1, a])

theorem inv_matrix_eq : (A⁻¹ = A_inv) → (a = 2) := 
by 
  sorry

end inv_matrix_eq_l151_151840


namespace problem_statement_l151_151124

-- Define the set of numbers
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_multiple (a b : ℕ) : Prop := b ∣ a

-- Problem statement
theorem problem_statement (al bill cal : ℕ) (h_al : al ∈ num_set) (h_bill : bill ∈ num_set) (h_cal : cal ∈ num_set) (h_distinct: distinct al bill cal) : 
  (is_multiple al bill) ∧ (is_multiple bill cal) →
  ∃ (p : ℚ), p = 1 / 190 :=
sorry

end problem_statement_l151_151124


namespace desired_interest_rate_l151_151117

theorem desired_interest_rate 
  (F : ℝ) -- Face value of each share
  (D : ℝ) -- Dividend rate
  (M : ℝ) -- Market value of each share
  (annual_dividend : ℝ := (D / 100) * F) -- Annual dividend per share
  (desired_interest_rate : ℝ := (annual_dividend / M) * 100) -- Desired interest rate
  (F_eq : F = 44) -- Given Face value
  (D_eq : D = 9) -- Given Dividend rate
  (M_eq : M = 33) -- Given Market value
  : desired_interest_rate = 12 := 
by
  sorry

end desired_interest_rate_l151_151117


namespace find_x_l151_151029

def angle_sum_condition (x : ℝ) := 6 * x + 3 * x + x + x + 4 * x = 360

theorem find_x (x : ℝ) (h : angle_sum_condition x) : x = 24 := 
by {
  sorry
}

end find_x_l151_151029


namespace kids_played_on_Wednesday_l151_151329

def played_on_Monday : ℕ := 17
def played_on_Tuesday : ℕ := 15
def total_kids : ℕ := 34

theorem kids_played_on_Wednesday :
  total_kids - (played_on_Monday + played_on_Tuesday) = 2 :=
by sorry

end kids_played_on_Wednesday_l151_151329


namespace fixed_point_of_exponential_function_l151_151756

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ x : ℝ, (x, a^(x + 2)) = p → x = -2 ∧ a^(x + 2) = 1 :=
by
  sorry

end fixed_point_of_exponential_function_l151_151756


namespace find_a_l151_151414

variable (a : ℝ)

def augmented_matrix (a : ℝ) :=
  ([1, -1, -3], [a, 3, 4])

def solution := (-1, 2)

theorem find_a (hx : -1 - 2 = -3)
               (hy : a * (-1) + 3 * 2 = 4) :
               a = 2 :=
by
  sorry

end find_a_l151_151414


namespace intersection_complement_l151_151468

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement_l151_151468


namespace line_equation_under_transformation_l151_151700

noncomputable def T1_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

noncomputable def T2_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 3]
]

noncomputable def NM_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -2],
  ![3, 0]
]

theorem line_equation_under_transformation :
  ∀ x y : ℝ, (∃ x' y' : ℝ, NM_matrix.mulVec ![x, y] = ![x', y'] ∧ x' = y') → 3 * x + 2 * y = 0 :=
by sorry

end line_equation_under_transformation_l151_151700


namespace radius_of_circle_l151_151323

theorem radius_of_circle (d : ℝ) (h : d = 22) : (d / 2) = 11 := by
  sorry

end radius_of_circle_l151_151323


namespace tan_identity_l151_151695

theorem tan_identity
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 7)
  (h2 : Real.tan (β - Real.pi / 4) = -1 / 3)
  : Real.tan (α + Real.pi / 4) = 8 / 9 := by
  sorry

end tan_identity_l151_151695


namespace kite_diagonals_sum_l151_151740

theorem kite_diagonals_sum (a b e f : ℝ) (h₁ : a ≥ b) 
    (h₂ : e < 2 * a) (h₃ : f < a + b) : 
    e + f < 2 * a + b := by 
    sorry

end kite_diagonals_sum_l151_151740


namespace right_triangle_condition_l151_151344

theorem right_triangle_condition (a b c : ℝ) :
  (a^3 + b^3 + c^3 = a*b*(a + b) - b*c*(b + c) + a*c*(a + c)) ↔ (a^2 = b^2 + c^2) ∨ (b^2 = a^2 + c^2) ∨ (c^2 = a^2 + b^2) :=
by
  sorry

end right_triangle_condition_l151_151344


namespace opposite_of_abs_frac_l151_151219

theorem opposite_of_abs_frac (h : 0 < (1 : ℝ) / 2023) : -|((1 : ℝ) / 2023)| = -(1 / 2023) := by
  sorry

end opposite_of_abs_frac_l151_151219


namespace sum_series_eq_two_l151_151132

noncomputable def series_term (n : ℕ) : ℚ := (3 * n - 2) / (n * (n + 1) * (n + 2))

theorem sum_series_eq_two :
  ∑' n : ℕ, series_term (n + 1) = 2 :=
sorry

end sum_series_eq_two_l151_151132


namespace degrees_for_lemon_pie_l151_151325

theorem degrees_for_lemon_pie 
    (total_students : ℕ)
    (chocolate_lovers : ℕ)
    (apple_lovers : ℕ)
    (blueberry_lovers : ℕ)
    (remaining_students : ℕ)
    (lemon_pie_degrees : ℝ) :
    total_students = 42 →
    chocolate_lovers = 15 →
    apple_lovers = 9 →
    blueberry_lovers = 7 →
    remaining_students = total_students - (chocolate_lovers + apple_lovers + blueberry_lovers) →
    lemon_pie_degrees = (remaining_students / 2 / total_students * 360) →
    lemon_pie_degrees = 47.14 :=
by
  intros _ _ _ _ _ _
  sorry

end degrees_for_lemon_pie_l151_151325


namespace bones_weight_in_meat_l151_151649

theorem bones_weight_in_meat (cost_with_bones : ℝ) (cost_without_bones : ℝ) (cost_bones : ℝ) :
  cost_with_bones = 165 → cost_without_bones = 240 → cost_bones = 40 → 
  ∃ x : ℝ, (40 * x + 240 * (1 - x) = 165) ∧ (x * 1000 = 375) :=
by
  intros h1 h2 h3
  use 0.375
  split
  · calc
      40 * 0.375 + 240 * (1 - 0.375)
        = 15 + 240 * 0.625 : by rw [show 0.375 = 3 / 8, by norm_num]
        = 15 + 150 : by rw [show 240 * 0.625 = 150, by norm_num]
        = 165 : by norm_num
  · calc
      0.375 * 1000 = 375 : by norm_num

-- The complete proof is included to demonstrate correctness and ensure the validity of the statement.

end bones_weight_in_meat_l151_151649


namespace max_value_A_l151_151157

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end max_value_A_l151_151157


namespace same_solutions_a_value_l151_151445

theorem same_solutions_a_value (a x : ℝ) (h1 : 2 * x + 1 = 3) (h2 : 3 - (a - x) / 3 = 1) : a = 7 := by
  sorry

end same_solutions_a_value_l151_151445


namespace max_roses_l151_151054

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 7.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  ∃ n, n = 316 :=
by
  sorry

end max_roses_l151_151054


namespace number_is_46000050_l151_151652

-- Define the corresponding place values for the given digit placements
def ten_million (n : ℕ) : ℕ := n * 10000000
def hundred_thousand (n : ℕ) : ℕ := n * 100000
def hundred (n : ℕ) : ℕ := n * 100

-- Define the specific numbers given in the conditions.
def digit_4 : ℕ := ten_million 4
def digit_60 : ℕ := hundred_thousand 6
def digit_500 : ℕ := hundred 5

-- Combine these values to form the number
def combined_number : ℕ := digit_4 + digit_60 + digit_500

-- The theorem, stating the number equals 46000050
theorem number_is_46000050 : combined_number = 46000050 := by
  sorry

end number_is_46000050_l151_151652


namespace cube_volume_increase_l151_151444

variable (a : ℝ)

theorem cube_volume_increase (a : ℝ) : (2 * a)^3 - a^3 = 7 * a^3 :=
by
  sorry

end cube_volume_increase_l151_151444


namespace fifth_term_sequence_l151_151310

theorem fifth_term_sequence : 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := 
by 
  sorry

end fifth_term_sequence_l151_151310


namespace square_diff_l151_151436

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end square_diff_l151_151436


namespace fractional_uniform_independent_l151_151334

noncomputable theory

variables (Ω : Type) [MeasurableSpace Ω] (U V : Ω → ℝ)
variables [IsProbabilityMeasure (measure_theory.measure_space Ω)]

def uniform_on_01 (x : ℝ) : Prop := ∀ a b : ℝ, (0 ≤ a) → (a ≤ b) → (b ≤ 1) → 
  (measure_theory.probability_measure.has_pdf (λ x, 1)) (λ x, a ≤ x ∧ x ≤ b) = b - a

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

axiom U_uniform : uniform_on_01 U
axiom U_V_indep : measure_theory.Indep U V
axiom V_rv : measure_theory.ProbabilityMeasure (measure_theory.measure_space Ω V)

theorem fractional_uniform_independent : uniform_on_01 (fractional_part (U + V)) ∧
  measure_theory.Indep V (λ ω, fractional_part (U ω + V ω)) :=
sorry

end fractional_uniform_independent_l151_151334


namespace ratio_shorter_to_longer_l151_151909

-- Define the total length and the length of the shorter piece
def total_length : ℕ := 90
def shorter_length : ℕ := 20

-- Define the length of the longer piece
def longer_length : ℕ := total_length - shorter_length

-- Define the ratio of shorter piece to longer piece
def ratio := shorter_length / longer_length

-- The target statement to prove
theorem ratio_shorter_to_longer : ratio = 2 / 7 := by
  sorry

end ratio_shorter_to_longer_l151_151909


namespace correct_calculation_l151_151903

theorem correct_calculation (a : ℝ) :
  (¬ (a^2 + a^2 = a^4)) ∧ (¬ (a^2 * a^3 = a^6)) ∧ (¬ ((a + 1)^2 = a^2 + 1)) ∧ ((-a^2)^2 = a^4) :=
by
  sorry

end correct_calculation_l151_151903


namespace product_of_two_numbers_l151_151225

theorem product_of_two_numbers (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 3)
  : x * y = 154 := by
  sorry

end product_of_two_numbers_l151_151225


namespace total_arrangements_l151_151816

-- Definitions according to the conditions
def num_female_teachers := 2
def num_male_teachers := 4
def num_females_per_group := 1
def num_males_per_group := 2

-- The goal is to prove the total number of different arrangements
theorem total_arrangements : 
  (nat.choose num_female_teachers num_females_per_group) * 
  (nat.choose num_male_teachers (2 * num_males_per_group)) = 12 := 
by
  -- Calculation steps should go here, but we skip the proof with sorry
  sorry

end total_arrangements_l151_151816


namespace joan_gemstone_samples_l151_151032

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end joan_gemstone_samples_l151_151032


namespace parallel_vectors_t_eq_neg1_l151_151413

theorem parallel_vectors_t_eq_neg1 (t : ℝ) :
  let a := (1, -1)
  let b := (t, 1)
  (a.1 + b.1, a.2 + b.2) = (k * (a.1 - b.1), k * (a.2 - b.2)) -> t = -1 :=
by
  sorry

end parallel_vectors_t_eq_neg1_l151_151413


namespace cos_A_condition_is_isosceles_triangle_tan_sum_l151_151579

variable {A B C a b c : ℝ}

theorem cos_A_condition (h : (3 * b - c) * Real.cos A - a * Real.cos C = 0) :
  Real.cos A = 1 / 3 := sorry

theorem is_isosceles_triangle (ha : a = 2 * Real.sqrt 3)
  (hs : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2) :
  c = 3 ∧ b = 3 := sorry

theorem tan_sum (h_sin : Real.sin B * Real.sin C = 2 / 3)
  (h_cos : Real.cos A = 1 / 3) :
  Real.tan A + Real.tan B + Real.tan C = 4 * Real.sqrt 2 := sorry

end cos_A_condition_is_isosceles_triangle_tan_sum_l151_151579


namespace determine_k_values_parallel_lines_l151_151697

theorem determine_k_values_parallel_lines :
  ∀ k : ℝ, ((k - 3) * x + (4 - k) * y + 1 = 0 ∧ 2 * (k - 3) * x - 2 * y + 3 = 0)
  → k = 2 ∨ k = 3 ∨ k = 6 :=
by
  sorry

end determine_k_values_parallel_lines_l151_151697


namespace intersection_pairs_pass_through_B_l151_151679

universe u
variables (α : Type u) [metric_space α] [normed_group α] [normed_space ℝ α]

def Thales_circle (A B : α) : set α := { x | dist x ((A + B) / 2) = dist A ((A + B) / 2) }

theorem intersection_pairs_pass_through_B
  (A B C : α)
  (F : α := (A + C) / 2)
  (k : set α := { x | dist x F = arbitrary_radius})
  (k_A : set α := Thales_circle B C)
  (k_C : set α := Thales_circle B A)
  (P Q P' Q' : α)
  (hP : P ∈ k ∩ k_A) (hQ : Q ∈ k ∩ k_A)
  (hP' : P' ∈ k ∩ k_C) (hQ' : Q' ∈ k ∩ k_C) :
  ∃ (f : fin 4 → α), function.injective f ∧
  (f 0 = P ∧ f 1 = P' ∧ f 2 = Q ∧ f 3 = Q') ∧
  (∃ i j k l, f i ≠ f j ∧ f k ≠ f l ∧
    line_through f i f j = line_through f k f l ∧ vertex B ∈ line_through f i f j) :=
sorry

end intersection_pairs_pass_through_B_l151_151679


namespace fedya_deposit_l151_151497

theorem fedya_deposit (n : ℕ) (h1 : n < 30) (h2 : 847 * 100 % (100 - n) = 0) : 
  (847 * 100 / (100 - n) = 1100) :=
by
  sorry

end fedya_deposit_l151_151497


namespace ratio_of_areas_l151_151877

-- Definitions based on the conditions given
def square_side_length : ℕ := 48
def rectangle_width : ℕ := 56
def rectangle_height : ℕ := 63

-- Areas derived from the definitions
def square_area := square_side_length * square_side_length
def rectangle_area := rectangle_width * rectangle_height

-- Lean statement to prove the ratio of areas
theorem ratio_of_areas :
  (square_area : ℚ) / rectangle_area = 2 / 3 := 
sorry

end ratio_of_areas_l151_151877


namespace compound_interest_time_l151_151802

theorem compound_interest_time 
  (P : ℝ) (r : ℝ) (A₁ : ℝ) (A₂ : ℝ) (t₁ t₂ : ℕ)
  (h1 : r = 0.10)
  (h2 : A₁ = P * (1 + r) ^ t₁)
  (h3 : A₂ = P * (1 + r) ^ t₂)
  (h4 : A₁ = 2420)
  (h5 : A₂ = 2662)
  (h6 : t₂ = t₁ + 3) :
  t₁ = 3 := 
sorry

end compound_interest_time_l151_151802


namespace fraction_sum_equals_mixed_number_l151_151769

theorem fraction_sum_equals_mixed_number :
  (3 / 5 : ℚ) + (2 / 3) + (16 / 15) = (7 / 3) :=
by sorry

end fraction_sum_equals_mixed_number_l151_151769


namespace find_pairs_l151_151292

theorem find_pairs (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1)
  (h1 : (a^2 + b) % (b^2 - a) = 0) 
  (h2 : (b^2 + a) % (a^2 - b) = 0) :
  (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) := 
sorry

end find_pairs_l151_151292


namespace concyclic_iff_ratio_real_l151_151340

noncomputable def concyclic_condition (z1 z2 z3 z4 : ℂ) : Prop :=
  (∃ c : ℂ, c ≠ 0 ∧ ∀ (w : ℂ), (w - z1) * (w - z3) / ((w - z2) * (w - z4)) = c)

noncomputable def ratio_real (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ r : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = r

theorem concyclic_iff_ratio_real (z1 z2 z3 z4 : ℂ) :
  concyclic_condition z1 z2 z3 z4 ↔ ratio_real z1 z2 z3 z4 :=
sorry

end concyclic_iff_ratio_real_l151_151340


namespace total_number_of_possible_outcomes_l151_151242

-- Define the conditions
def num_faces_per_die : ℕ := 6
def num_dice : ℕ := 2

-- Define the question as a hypothesis and the answer as the conclusion
theorem total_number_of_possible_outcomes :
  (num_faces_per_die * num_faces_per_die) = 36 := 
by
  -- Provide a proof outline, this is used to skip the actual proof
  sorry

end total_number_of_possible_outcomes_l151_151242


namespace range_of_a_l151_151844

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end range_of_a_l151_151844


namespace linear_iff_m_eq_neg1_l151_151019

variables {m : ℝ} (x : ℝ)

def linear_form := (m - 1) * x ^ (abs m) + 2

theorem linear_iff_m_eq_neg1 : (∃ k b, linear_form x = k * x + b ∧ k ≠ 0) ↔ m = -1 := by sorry

end linear_iff_m_eq_neg1_l151_151019


namespace find_y_l151_151377

variable {R : Type} [Field R] (y : R)

-- The condition: y = (1/y) * (-y) + 3
def condition (y : R) : Prop :=
  y = (1 / y) * (-y) + 3

-- The theorem to prove: under the condition, y = 2
theorem find_y (y : R) (h : condition y) : y = 2 := 
sorry

end find_y_l151_151377


namespace common_root_and_param_l151_151689

theorem common_root_and_param :
  ∀ (x : ℤ) (P p : ℚ),
    (P = -((x^2 - x - 2) / (x - 1)) ∧ x ≠ 1) →
    (p = -((x^2 + 2*x - 1) / (x + 2)) ∧ x ≠ -2) →
    (-x + (2 / (x - 1)) = -x + (1 / (x + 2))) →
    x = -5 ∧ p = 14 / 3 :=
by
  intros x P p hP hp hroot
  sorry

end common_root_and_param_l151_151689


namespace find_r_l151_151388

theorem find_r (r s : ℝ)
  (h1 : ∀ α β : ℝ, (α + β = -r) ∧ (α * β = s) → 
         ∃ t : ℝ, (t^2 - (α^2 + β^2) * t + (α^2 * β^2) = 0) ∧ |α^2 - β^2| = 8)
  (h_sum : ∃ α β : ℝ, α + β = 10) :
  r = -10 := by
  sorry

end find_r_l151_151388


namespace fraction_power_l151_151807

theorem fraction_power (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : (↑a / ↑b)^3 = 27 / 64 :=
by
  rw [h_a, h_b]
  norm_num
  sorry

end fraction_power_l151_151807


namespace find_ordered_pair_l151_151556

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l151_151556


namespace latest_start_time_is_correct_l151_151196

noncomputable def doughComingToRoomTemp : ℕ := 1  -- 1 hour
noncomputable def shapingDough : ℕ := 15         -- 15 minutes
noncomputable def proofingDough : ℕ := 2         -- 2 hours
noncomputable def bakingBread : ℕ := 30          -- 30 minutes
noncomputable def coolingBread : ℕ := 15         -- 15 minutes
noncomputable def bakeryOpeningTime : ℕ := 6     -- 6:00 am

-- Total preparation time in minutes
noncomputable def totalPreparationTimeInMinutes : ℕ :=
  (doughComingToRoomTemp * 60) + shapingDough + (proofingDough * 60) + bakingBread + coolingBread

-- Total preparation time in hours
noncomputable def totalPreparationTimeInHours : ℕ :=
  totalPreparationTimeInMinutes / 60

-- Latest time the baker can start working
noncomputable def latestTimeBakerCanStart : ℕ :=
  if (bakeryOpeningTime - totalPreparationTimeInHours) < 0 then 24 + (bakeryOpeningTime - totalPreparationTimeInHours)
  else bakeryOpeningTime - totalPreparationTimeInHours

theorem latest_start_time_is_correct : latestTimeBakerCanStart = 2 := by
  sorry

end latest_start_time_is_correct_l151_151196


namespace solution_proof_l151_151661

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l151_151661


namespace find_B_value_l151_151071

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l151_151071


namespace squared_difference_l151_151430

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l151_151430


namespace mobius_total_trip_time_l151_151598

theorem mobius_total_trip_time :
  ∀ (d1 d2 v1 v2 : ℝ) (n r : ℕ),
  d1 = 143 → d2 = 143 → 
  v1 = 11 → v2 = 13 → 
  n = 4 → r = (30:ℝ)/60 →
  d1 / v1 + d2 / v2 + n * r = 26 :=
by
  intros d1 d2 v1 v2 n r h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end mobius_total_trip_time_l151_151598


namespace second_caterer_cheaper_l151_151207

theorem second_caterer_cheaper (x : ℕ) (h : x > 33) : 200 + 12 * x < 100 + 15 * x := 
by
  sorry

end second_caterer_cheaper_l151_151207


namespace union_of_subsets_l151_151707

open Set

variable (A B : Set ℕ)

theorem union_of_subsets (m : ℕ) (hA : A = {1, 3}) (hB : B = {1, 2, m}) (hSubset : A ⊆ B) :
    A ∪ B = {1, 2, 3} :=
  sorry

end union_of_subsets_l151_151707


namespace non_isosceles_triangle_has_equidistant_incenter_midpoints_l151_151719

structure Triangle (α : Type*) :=
(a b c : α)
(incenter : α)
(midpoint_a_b : α)
(midpoint_b_c : α)
(midpoint_c_a : α)
(equidistant : Bool)
(non_isosceles : Bool)

-- Define the triangle with the specified properties.
noncomputable def counterexample_triangle : Triangle ℝ :=
{ a := 3,
  b := 4,
  c := 5, 
  incenter := 1, -- incenter length for the right triangle.
  midpoint_a_b := 2.5,
  midpoint_b_c := 2,
  midpoint_c_a := 1.5,
  equidistant := true,    -- midpoints of two sides are equidistant from incenter
  non_isosceles := true } -- the triangle is not isosceles

theorem non_isosceles_triangle_has_equidistant_incenter_midpoints :
  ∃ (T : Triangle ℝ), T.equidistant ∧ T.non_isosceles := by
  use counterexample_triangle
  sorry

end non_isosceles_triangle_has_equidistant_incenter_midpoints_l151_151719


namespace largest_prime_factor_9801_l151_151554

theorem largest_prime_factor_9801 : ∃ p : ℕ, Prime p ∧ p ∣ 9801 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 9801 → q ≤ p :=
sorry

end largest_prime_factor_9801_l151_151554


namespace find_original_number_l151_151035

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l151_151035


namespace swimming_class_attendance_l151_151638

def total_students : ℕ := 1000
def chess_ratio : ℝ := 0.25
def swimming_ratio : ℝ := 0.50

def chess_students := chess_ratio * total_students
def swimming_students := swimming_ratio * chess_students

theorem swimming_class_attendance :
  swimming_students = 125 :=
by
  sorry

end swimming_class_attendance_l151_151638


namespace rotary_club_extra_omelets_l151_151753

theorem rotary_club_extra_omelets
  (small_children_tickets : ℕ)
  (older_children_tickets : ℕ)
  (adult_tickets : ℕ)
  (senior_tickets : ℕ)
  (eggs_total : ℕ)
  (omelet_for_small_child : ℝ)
  (omelet_for_older_child : ℝ)
  (omelet_for_adult : ℝ)
  (omelet_for_senior : ℝ)
  (eggs_per_omelet : ℕ)
  (extra_omelets : ℕ) :
  small_children_tickets = 53 →
  older_children_tickets = 35 →
  adult_tickets = 75 →
  senior_tickets = 37 →
  eggs_total = 584 →
  omelet_for_small_child = 0.5 →
  omelet_for_older_child = 1 →
  omelet_for_adult = 2 →
  omelet_for_senior = 1.5 →
  eggs_per_omelet = 2 →
  extra_omelets = (eggs_total - (2 * (small_children_tickets * omelet_for_small_child +
                                      older_children_tickets * omelet_for_older_child +
                                      adult_tickets * omelet_for_adult +
                                      senior_tickets * omelet_for_senior))) / eggs_per_omelet →
  extra_omelets = 25 :=
by
  intros hsmo_hold hsoc_hold hat_hold hsnt_hold htot_hold
        hosm_hold hocc_hold hact_hold hsen_hold hepom_hold hres_hold
  sorry

end rotary_club_extra_omelets_l151_151753


namespace equation_of_rotated_translated_line_l151_151603

theorem equation_of_rotated_translated_line (x y : ℝ) :
  (∀ x, y = 3 * x → y = x / -3 + 1 / -3) →
  (∀ x, y = -1/3 * (x - 1)) →
  y = -1/3 * x + 1/3 :=
sorry

end equation_of_rotated_translated_line_l151_151603


namespace unique_solution_a_eq_sqrt_three_l151_151701

theorem unique_solution_a_eq_sqrt_three {a : ℝ} (h1 : ∀ x y : ℝ, x^2 + a * abs x + a^2 - 3 = 0 ∧ y^2 + a * abs y + a^2 - 3 = 0 → x = y)
  (h2 : a > 0) : a = Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt_three_l151_151701


namespace hcl_formed_l151_151318

-- Define the balanced chemical equation as a relationship between reactants and products
def balanced_equation (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 + 4 * m_Cl2 = m_CCl4 + 6 * m_HCl

-- Define the problem-specific values
def reaction_given (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 = 3 ∧ m_Cl2 = 21 ∧ m_CCl4 = 6 ∧ balanced_equation m_C2H6 m_Cl2 m_CCl4 m_HCl

-- Prove the number of moles of HCl formed
theorem hcl_formed : ∃ (m_HCl : ℝ), reaction_given 3 21 6 m_HCl ∧ m_HCl = 18 :=
by
  sorry

end hcl_formed_l151_151318


namespace not_lengths_of_external_diagonals_l151_151288

theorem not_lengths_of_external_diagonals (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  (¬ (a = 5 ∧ b = 6 ∧ c = 9)) :=
by
  sorry

end not_lengths_of_external_diagonals_l151_151288


namespace problem_solution_l151_151227

theorem problem_solution :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ n = 58 :=
by
  -- Lean code to prove the statement
  sorry

end problem_solution_l151_151227


namespace commercials_time_l151_151600

theorem commercials_time (n : ℕ) (t : ℕ) (h : t = 30) (k : ℕ) (one_fourth : ℚ) (h_fraction : one_fourth = 1 / 4) :
  n = 6 → 
  k = 6 * t → 
  k / 4 = 45 :=
by
  intros h_n h_k
  rw [h, ← nat.cast_mul, ← nat.cast_div, h_fraction] at h_k
  norm_num at h_k
  assumption

end commercials_time_l151_151600


namespace lowest_degree_poly_meets_conditions_l151_151509

-- Define a predicate that checks if a polynomial P meets the conditions
def poly_meets_conditions (P : ℚ[X]) (b : ℚ) : Prop :=
  (∀ x, coeff P x ≠ b) ∧ 
  (∃ x y, coeff P x < b ∧ coeff P y > b)

-- Statement of the theorem we want to prove
theorem lowest_degree_poly_meets_conditions : ∀ (b : ℚ), 
  ∃ (P : ℚ[X]), poly_meets_conditions P b ∧ degree P = 4 :=
begin
  sorry
end

end lowest_degree_poly_meets_conditions_l151_151509


namespace solving_linear_equations_problems_l151_151342

theorem solving_linear_equations_problems (total_problems : ℕ) (perc_algebra : ℕ) :
  total_problems = 140 → perc_algebra = 40 → 
  let algebra_problems := (perc_algebra * total_problems) / 100 in
  let solving_linear := algebra_problems / 2 in
  solving_linear = 28 :=
by
  intros h_total h_perc
  let algebra_problems := (perc_algebra * total_problems) / 100
  let solving_linear := algebra_problems / 2
  have h1 : algebra_problems = 56 := by
    rw [h_total, h_perc]
    norm_num
  have h2 : solving_linear = algebra_problems / 2 := by rfl
  rw [h1, h2]
  norm_num
  simp only [Nat.div_eq_of_lt]
  norm_num
  sorry

end solving_linear_equations_problems_l151_151342


namespace days_passed_before_cows_ran_away_l151_151374

def initial_cows := 1000
def initial_days := 50
def cows_left := 800
def cows_run_away := initial_cows - cows_left
def total_food := initial_cows * initial_days
def remaining_food (x : ℕ) := total_food - initial_cows * x
def food_needed := cows_left * initial_days

theorem days_passed_before_cows_ran_away (x : ℕ) :
  (remaining_food x = food_needed) → (x = 10) :=
by
  sorry

end days_passed_before_cows_ran_away_l151_151374


namespace cody_spent_19_dollars_l151_151280

-- Given conditions
def initial_money : ℕ := 45
def birthday_gift : ℕ := 9
def remaining_money : ℕ := 35

-- Problem: Prove that the amount of money spent on the game is $19.
theorem cody_spent_19_dollars :
  (initial_money + birthday_gift - remaining_money) = 19 :=
by sorry

end cody_spent_19_dollars_l151_151280


namespace projectile_reaches_24_meters_l151_151067

theorem projectile_reaches_24_meters (h : ℝ) (t : ℝ) (v₀ : ℝ) :
  (h = -4.9 * t^2 + 19.6 * t) ∧ (h = 24) → t = 4 :=
by
  intros
  sorry

end projectile_reaches_24_meters_l151_151067


namespace cos_double_angle_l151_151167

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi - θ) = 1 / 3) : 
  Real.cos (2 * θ) = 7 / 9 :=
by 
  sorry

end cos_double_angle_l151_151167


namespace plane_speed_with_tailwind_l151_151654

theorem plane_speed_with_tailwind (V : ℝ) (tailwind_speed : ℝ) (ground_speed_against_tailwind : ℝ) 
  (H1 : tailwind_speed = 75) (H2 : ground_speed_against_tailwind = 310) (H3 : V - tailwind_speed = ground_speed_against_tailwind) :
  V + tailwind_speed = 460 :=
by
  sorry

end plane_speed_with_tailwind_l151_151654


namespace sum_of_squares_of_chords_in_sphere_l151_151308

-- Defining variables
variables (R PO : ℝ)

-- Define the problem statement
theorem sum_of_squares_of_chords_in_sphere
  (chord_lengths_squared : ℝ)
  (H_chord_lengths_squared : chord_lengths_squared = 3 * R^2 - 2 * PO^2) :
  chord_lengths_squared = 3 * R^2 - 2 * PO^2 :=
by
  sorry -- proof is omitted

end sum_of_squares_of_chords_in_sphere_l151_151308


namespace compute_x_l151_151386

theorem compute_x :
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = (∑' n : ℕ, 1 / (9^n)) →
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = 1 / (1 - (1 / 9)) →
  9 = 9 :=
by
  sorry

end compute_x_l151_151386


namespace central_angle_of_sector_l151_151841

theorem central_angle_of_sector
  (r : ℝ) (S_sector : ℝ) (alpha : ℝ) (h₁ : r = 2) (h₂ : S_sector = (2 / 5) * Real.pi)
  (h₃ : S_sector = (1 / 2) * alpha * r^2) : alpha = Real.pi / 5 :=
by
  sorry

end central_angle_of_sector_l151_151841


namespace repeated_mul_eq_pow_l151_151214

-- Define the repeated multiplication of 2, n times
def repeated_mul (n : ℕ) : ℕ :=
  (List.replicate n 2).prod

-- State the theorem to prove
theorem repeated_mul_eq_pow (n : ℕ) : repeated_mul n = 2 ^ n :=
by
  sorry

end repeated_mul_eq_pow_l151_151214


namespace multiply_exponents_l151_151384

theorem multiply_exponents (a : ℝ) : (6 * a^2) * (1/2 * a^3) = 3 * a^5 := by
  sorry

end multiply_exponents_l151_151384


namespace ducks_remaining_after_three_nights_l151_151765

def initial_ducks : ℕ := 320
def first_night_ducks_eaten (ducks : ℕ) : ℕ := ducks * 1 / 4
def after_first_night (ducks : ℕ) : ℕ := ducks - first_night_ducks_eaten ducks
def second_night_ducks_fly_away (ducks : ℕ) : ℕ := ducks * 1 / 6
def after_second_night (ducks : ℕ) : ℕ := ducks - second_night_ducks_fly_away ducks
def third_night_ducks_stolen (ducks : ℕ) : ℕ := ducks * 30 / 100
def after_third_night (ducks : ℕ) : ℕ := ducks - third_night_ducks_stolen ducks

theorem ducks_remaining_after_three_nights : after_third_night (after_second_night (after_first_night initial_ducks)) = 140 :=
by 
  -- replace the following sorry with the actual proof steps
  sorry

end ducks_remaining_after_three_nights_l151_151765


namespace part_I_equality_condition_part_II_l151_151702

-- Lean statement for Part (I)
theorem part_I (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) : 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ 5 :=
sorry

theorem equality_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  (2 * Real.sqrt x + Real.sqrt (5 - x) = 5) ↔ (x = 4) :=
sorry

-- Lean statement for Part (II)
theorem part_II (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ |m - 2|) →
  (m ≥ 7 ∨ m ≤ -3) :=
sorry

end part_I_equality_condition_part_II_l151_151702


namespace max_sum_first_n_terms_formula_sum_terms_abs_l151_151562

theorem max_sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  ∃ (n : ℕ), n = 15 ∧ S 15 = 225 := by
  sorry

theorem formula_sum_terms_abs (a : ℕ → ℤ) (S T : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  (∀ n, n ≤ 15 → T n = 30 * n - n * n) ∧
  (∀ n, n ≥ 16 → T n = n * n - 30 * n + 450) := by
  sorry

end max_sum_first_n_terms_formula_sum_terms_abs_l151_151562


namespace billy_soda_distribution_l151_151934

theorem billy_soda_distribution (sisters : ℕ) (brothers : ℕ) (total_sodas : ℕ) (total_siblings : ℕ)
  (h1 : total_sodas = 12)
  (h2 : sisters = 2)
  (h3 : brothers = 2 * sisters)
  (h4 : total_siblings = sisters + brothers) :
  total_sodas / total_siblings = 2 :=
by
  sorry

end billy_soda_distribution_l151_151934


namespace gcd_min_value_l151_151712

theorem gcd_min_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 :=
by
  sorry

end gcd_min_value_l151_151712


namespace kylie_beads_total_l151_151724

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end kylie_beads_total_l151_151724


namespace problem_l151_151421

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l151_151421


namespace valid_license_plates_l151_151327

-- Define the number of vowels and the total alphabet letters.
def num_vowels : ℕ := 5
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates in Eldoria.
theorem valid_license_plates : num_vowels * num_letters * num_digits^3 = 130000 := by
  sorry

end valid_license_plates_l151_151327


namespace find_angle_A_l151_151324

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end find_angle_A_l151_151324


namespace original_number_in_magician_game_l151_151716

theorem original_number_in_magician_game (a b c : ℕ) (habc : 100 * a + 10 * b + c = 332) (N : ℕ) (hN : N = 4332) :
    222 * (a + b + c) = 4332 → 100 * a + 10 * b + c = 332 :=
by 
  sorry

end original_number_in_magician_game_l151_151716


namespace audio_per_cd_l151_151806

theorem audio_per_cd (total_audio : ℕ) (max_per_cd : ℕ) (num_cds : ℕ) 
  (h1 : total_audio = 360) 
  (h2 : max_per_cd = 60) 
  (h3 : num_cds = total_audio / max_per_cd): 
  (total_audio / num_cds = max_per_cd) :=
by
  sorry

end audio_per_cd_l151_151806


namespace rational_solutions_k_values_l151_151139

theorem rational_solutions_k_values (k : ℕ) (h₁ : k > 0) 
    (h₂ : ∃ (m : ℤ), 900 - 4 * (k:ℤ)^2 = m^2) : k = 9 ∨ k = 15 := 
by
  sorry

end rational_solutions_k_values_l151_151139


namespace germination_rate_sunflower_l151_151848

variable (s_d s_s f_d f_s p : ℕ) (g_d g_f : ℚ)

-- Define the conditions
def conditions :=
  s_d = 25 ∧ s_s = 25 ∧ g_d = 0.60 ∧ g_f = 0.80 ∧ p = 28 ∧ f_d = 12 ∧ f_s = 16

-- Define the statement to be proved
theorem germination_rate_sunflower (h : conditions s_d s_s f_d f_s p g_d g_f) : 
  (f_s / (g_f * (s_s : ℚ))) > 0.0 ∧ (f_s / (g_f * (s_s : ℚ)) * 100) = 80 := 
by
  sorry

end germination_rate_sunflower_l151_151848


namespace homogeneous_variances_l151_151834

noncomputable def sample_sizes : (ℕ × ℕ × ℕ) := (9, 13, 15)
noncomputable def sample_variances : (ℝ × ℝ × ℝ) := (3.2, 3.8, 6.3)
noncomputable def significance_level : ℝ := 0.05
noncomputable def degrees_of_freedom : ℕ := 2
noncomputable def V : ℝ := 1.43
noncomputable def critical_value : ℝ := 6.0

theorem homogeneous_variances :
  V < critical_value :=
by
  sorry

end homogeneous_variances_l151_151834


namespace triangle_is_right_angled_l151_151489

noncomputable def median (a b c : ℝ) : ℝ := (1 / 2) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2))

theorem triangle_is_right_angled (a b c : ℝ) (ha : median a b c = 5) (hb : median b c a = Real.sqrt 52) (hc : median c a b = Real.sqrt 73) :
  a^2 = b^2 + c^2 :=
sorry

end triangle_is_right_angled_l151_151489


namespace present_worth_approx_l151_151912

noncomputable def amount_after_years (P : ℝ) : ℝ :=
  let A1 := P * (1 + 5 / 100)                      -- Amount after the first year.
  let A2 := A1 * (1 + 5 / 100)^2                   -- Amount after the second year.
  let A3 := A2 * (1 + 3 / 100)^4                   -- Amount after the third year.
  A3

noncomputable def banker's_gain (P : ℝ) : ℝ :=
  amount_after_years P - P

theorem present_worth_approx :
  ∃ P : ℝ, abs (P - 114.94) < 1 ∧ banker's_gain P = 36 :=
sorry

end present_worth_approx_l151_151912


namespace remainder_when_divided_by_29_l151_151528

theorem remainder_when_divided_by_29 (k N : ℤ) (h : N = 761 * k + 173) : N % 29 = 28 :=
by
  sorry

end remainder_when_divided_by_29_l151_151528


namespace soda_cost_l151_151211

theorem soda_cost (S P W : ℝ) (h1 : P = 3 * S) (h2 : W = 3 * P) (h3 : 3 * S + 2 * P + W = 18) : S = 1 :=
by
  sorry

end soda_cost_l151_151211


namespace max_pasture_area_maximization_l151_151262

noncomputable def max_side_length (fence_cost_per_foot : ℕ) (total_cost : ℕ) : ℕ :=
  let total_length := total_cost / fence_cost_per_foot
  let x := total_length / 4
  2 * x

theorem max_pasture_area_maximization :
  max_side_length 8 1920 = 120 :=
by
  sorry

end max_pasture_area_maximization_l151_151262


namespace sufficient_but_not_necessary_l151_151571

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem sufficient_but_not_necessary (m n : ℝ) :
  vectors_parallel (m, 1) (n, 1) ↔ (m = n) := sorry

end sufficient_but_not_necessary_l151_151571


namespace wilson_pays_total_l151_151245

def hamburger_price : ℝ := 5
def cola_price : ℝ := 2
def fries_price : ℝ := 3
def sundae_price : ℝ := 4
def discount_coupon : ℝ := 4
def loyalty_discount : ℝ := 0.10

def total_cost_before_discounts : ℝ :=
  2 * hamburger_price + 3 * cola_price + fries_price + sundae_price

def total_cost_after_coupon : ℝ :=
  total_cost_before_discounts - discount_coupon

def loyalty_discount_amount : ℝ :=
  loyalty_discount * total_cost_after_coupon

def total_cost_after_all_discounts : ℝ :=
  total_cost_after_coupon - loyalty_discount_amount

theorem wilson_pays_total : total_cost_after_all_discounts = 17.10 :=
  sorry

end wilson_pays_total_l151_151245


namespace bottles_have_200_mL_l151_151533

def liters_to_milliliters (liters : ℕ) : ℕ :=
  liters * 1000

def total_milliliters (liters : ℕ) : ℕ :=
  liters_to_milliliters liters

def milliliters_per_bottle (total_mL : ℕ) (num_bottles : ℕ) : ℕ :=
  total_mL / num_bottles

theorem bottles_have_200_mL (num_bottles : ℕ) (total_oil_liters : ℕ) (h1 : total_oil_liters = 4) (h2 : num_bottles = 20) :
  milliliters_per_bottle (total_milliliters total_oil_liters) num_bottles = 200 := 
by
  sorry

end bottles_have_200_mL_l151_151533


namespace product_of_abcd_l151_151904

noncomputable def a (c : ℚ) : ℚ := 33 * c + 16
noncomputable def b (c : ℚ) : ℚ := 8 * c + 4
noncomputable def d (c : ℚ) : ℚ := c + 1

theorem product_of_abcd :
  (2 * a c + 3 * b c + 5 * c + 8 * d c = 45) →
  (4 * (d c + c) = b c) →
  (4 * (b c) + c = a c) →
  (c + 1 = d c) →
  a c * b c * c * d c = ((1511 : ℚ) / 103) * ((332 : ℚ) / 103) * (-(7 : ℚ) / 103) * ((96 : ℚ) / 103) :=
by
  intros
  sorry

end product_of_abcd_l151_151904


namespace find_two_irreducible_fractions_l151_151955

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l151_151955


namespace parity_of_pq_l151_151169

theorem parity_of_pq (x y m n p q : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 0)
    (hx : x = p) (hy : y = q) (h1 : x - 1998 * y = n) (h2 : 1999 * x + 3 * y = m) :
    p % 2 = 0 ∧ q % 2 = 1 :=
by
  sorry

end parity_of_pq_l151_151169


namespace farthest_vertex_coordinates_l151_151878

noncomputable def image_vertex_coordinates_farthest_from_origin 
    (center_EFGH : ℝ × ℝ) (area_EFGH : ℝ) (dilation_center : ℝ × ℝ) 
    (scale_factor : ℝ) : ℝ × ℝ := sorry

theorem farthest_vertex_coordinates 
    (center_EFGH : ℝ × ℝ := (10, -6)) (area_EFGH : ℝ := 16) 
    (dilation_center : ℝ × ℝ := (2, 2)) (scale_factor : ℝ := 3) : 
    image_vertex_coordinates_farthest_from_origin center_EFGH area_EFGH dilation_center scale_factor = (32, -28) := 
sorry

end farthest_vertex_coordinates_l151_151878


namespace gcd_10010_20020_l151_151551

/-- 
Given that 10,010 can be written as 10 * 1001 and 20,020 can be written as 20 * 1001,
prove that the GCD of 10,010 and 20,020 is 10010.
-/
theorem gcd_10010_20020 : gcd 10010 20020 = 10010 := by
  sorry

end gcd_10010_20020_l151_151551


namespace find_difference_of_squares_l151_151424

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l151_151424


namespace sin_cos_identity_l151_151400

variable (α : Real)

theorem sin_cos_identity (h : Real.sin α - Real.cos α = -5/4) : Real.sin α * Real.cos α = -9/32 :=
by
  sorry

end sin_cos_identity_l151_151400


namespace x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l151_151574

theorem x_equals_neg_x_is_zero (x : ℝ) (h : x = -x) : x = 0 := sorry

theorem abs_x_equals_2_is_pm_2 (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := sorry

end x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l151_151574


namespace solution_l151_151592

variable (a : ℕ → ℝ)

noncomputable def pos_sequence (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0

noncomputable def recursive_relation (n : ℕ) : Prop :=
  ∀ n : ℕ, (n > 0) → (n+1) * a (n+1)^2 - n * a n^2 + a (n+1) * a n = 0

noncomputable def sequence_condition (n : ℕ) : Prop :=
  a 1 = 1 ∧ pos_sequence a n ∧ recursive_relation a n

theorem solution : ∀ n : ℕ, n > 0 → sequence_condition a n → a n = 1 / n :=
by
  intros n hn h
  sorry

end solution_l151_151592


namespace squared_difference_l151_151429

variable {x y : ℝ}

theorem squared_difference (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 :=
by
  sorry

end squared_difference_l151_151429


namespace chloe_treasures_first_level_l151_151385

def chloe_treasures_score (T : ℕ) (score_per_treasure : ℕ) (treasures_second_level : ℕ) (total_score : ℕ) :=
  T * score_per_treasure + treasures_second_level * score_per_treasure = total_score

theorem chloe_treasures_first_level :
  chloe_treasures_score T 9 3 81 → T = 6 :=
by
  intro h
  sorry

end chloe_treasures_first_level_l151_151385


namespace ratio_B_to_C_l151_151875

-- Definitions for conditions
def total_amount : ℕ := 1440
def B_amt : ℕ := 270
def A_amt := (1 / 3) * B_amt
def C_amt := total_amount - A_amt - B_amt

-- Theorem statement
theorem ratio_B_to_C : (B_amt : ℚ) / C_amt = 1 / 4 :=
  by
    sorry

end ratio_B_to_C_l151_151875


namespace ones_digit_of_power_l151_151830

theorem ones_digit_of_power (a b : ℕ) : (34^{34 * (17^{17})}) % 10 = 4 :=
by
  sorry

end ones_digit_of_power_l151_151830


namespace division_by_fraction_l151_151146

theorem division_by_fraction :
  5 / (8 / 13) = 65 / 8 :=
sorry

end division_by_fraction_l151_151146


namespace original_number_l151_151037

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l151_151037


namespace equivalent_annual_rate_correct_l151_151751

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100)^4

noncomputable def equivalent_annual_rate (annual_rate : ℝ) : ℝ :=
  (effective_annual_rate (quarterly_rate annual_rate) - 1) * 100

theorem equivalent_annual_rate_correct :
  equivalent_annual_rate 8 = 8.24 := 
by
  sorry

end equivalent_annual_rate_correct_l151_151751


namespace chosen_number_l151_151801

theorem chosen_number (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 :=
sorry

end chosen_number_l151_151801


namespace max_value_sum_product_l151_151304

noncomputable def maximum_dot_product {n : ℕ} (a b : Fin n → ℝ) : ℝ :=
  ∑ i, a i * b i

theorem max_value_sum_product (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∑ i, (a i)^2 = 4) (hb : ∑ i, (b i)^2 = 9) : 
  maximum_dot_product a b ≤ 6 :=
sorry

end max_value_sum_product_l151_151304


namespace court_cost_proof_l151_151469

-- Define all the given conditions
def base_fine : ℕ := 50
def penalty_rate : ℕ := 2
def mark_speed : ℕ := 75
def speed_limit : ℕ := 30
def school_zone_multiplier : ℕ := 2
def lawyer_fee_rate : ℕ := 80
def lawyer_hours : ℕ := 3
def total_owed : ℕ := 820

-- Define the calculation for the additional penalty
def additional_penalty : ℕ := (mark_speed - speed_limit) * penalty_rate

-- Define the calculation for the total fine
def total_fine : ℕ := (base_fine + additional_penalty) * school_zone_multiplier

-- Define the calculation for the lawyer's fee
def lawyer_fee : ℕ := lawyer_fee_rate * lawyer_hours

-- Define the calculation for the total of fine and lawyer's fee
def fine_and_lawyer_fee := total_fine + lawyer_fee

-- Prove the court costs
theorem court_cost_proof : total_owed - fine_and_lawyer_fee = 300 := by
  sorry

end court_cost_proof_l151_151469


namespace maximum_value_A_l151_151158

theorem maximum_value_A
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  let A := (x - y) * Real.sqrt (x ^ 2 + y ^ 2) +
           (y - z) * Real.sqrt (y ^ 2 + z ^ 2) +
           (z - x) * Real.sqrt (z ^ 2 + x ^ 2) +
           Real.sqrt 2,
      B := (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 + 2 in
  A / B ≤ 1 / Real.sqrt 2 :=
sorry

end maximum_value_A_l151_151158


namespace problem_l151_151423

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l151_151423


namespace sufficient_but_not_necessary_l151_151174

theorem sufficient_but_not_necessary (x : ℝ) : 
  (1 < x ∧ x < 2) → (x > 0) ∧ ¬((x > 0) → (1 < x ∧ x < 2)) := 
by 
  sorry

end sufficient_but_not_necessary_l151_151174


namespace rate_of_current_is_5_l151_151223

theorem rate_of_current_is_5 
  (speed_still_water : ℕ)
  (distance_travelled : ℕ)
  (time_travelled : ℚ) 
  (effective_speed_with_current : ℚ) : 
  speed_still_water = 20 ∧ distance_travelled = 5 ∧ time_travelled = 1/5 ∧ 
  effective_speed_with_current = (speed_still_water + 5) →
  effective_speed_with_current * time_travelled = distance_travelled :=
by
  sorry

end rate_of_current_is_5_l151_151223


namespace coefficient_of_x_l151_151140

theorem coefficient_of_x :
  let expr := (5 * (x - 6)) + (6 * (9 - 3 * x ^ 2 + 3 * x)) - (9 * (5 * x - 4))
  (expr : ℝ) → 
  let expr' := 5 * x - 30 + 54 - 18 * x ^ 2 + 18 * x - 45 * x + 36
  (expr' : ℝ) → 
  let coeff_x := 5 + 18 - 45
  coeff_x = -22 :=
by
  sorry

end coefficient_of_x_l151_151140


namespace find_number_l151_151393

theorem find_number (x : ℝ) (h : 4 * (3 * x / 5 - 220) = 320) : x = 500 :=
sorry

end find_number_l151_151393


namespace integer_solutions_eq_l151_151552

theorem integer_solutions_eq :
  { (x, y) : ℤ × ℤ | 2 * x ^ 4 - 4 * y ^ 4 - 7 * x ^ 2 * y ^ 2 - 27 * x ^ 2 + 63 * y ^ 2 + 85 = 0 }
  = { (3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3) } :=
by sorry

end integer_solutions_eq_l151_151552


namespace fraction_distance_traveled_by_bus_l151_151535

theorem fraction_distance_traveled_by_bus (D : ℝ) (hD : D = 105.00000000000003)
    (distance_by_foot : ℝ) (h_foot : distance_by_foot = (1 / 5) * D)
    (distance_by_car : ℝ) (h_car : distance_by_car = 14) :
    (D - (distance_by_foot + distance_by_car)) / D = 2 / 3 := by
  sorry

end fraction_distance_traveled_by_bus_l151_151535


namespace unique_solution_tan_eq_sin_cos_l151_151319

theorem unique_solution_tan_eq_sin_cos :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arccos 0.1 ∧ Real.tan x = Real.sin (Real.cos x) :=
sorry

end unique_solution_tan_eq_sin_cos_l151_151319


namespace probability_is_correct_l151_151667

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l151_151667


namespace range_of_m_length_of_chord_l151_151843

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

noncomputable def line_eq (x y m : ℝ) : Prop := y = x + m

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, ellipse_eq x y ∧ line_eq x y m) ↔ -real.sqrt 3 ≤ m ∧ m ≤ real.sqrt 3 :=
sorry

theorem length_of_chord (x1 y1 x2 y2 : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
(hline1 : line_eq x1 y1 (-1)) (hline2 : line_eq x2 y2 (-1)) : 
real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = (4 / 3) * real.sqrt 2 :=
sorry

end range_of_m_length_of_chord_l151_151843


namespace min_pairs_l151_151871

-- Define the types for knights and liars
inductive Residents
| Knight : Residents
| Liar : Residents

def total_residents : ℕ := 200
def knights : ℕ := 100
def liars : ℕ := 100

-- Additional conditions
def conditions (friend_claims_knights friend_claims_liars : ℕ) : Prop :=
  friend_claims_knights = 100 ∧
  friend_claims_liars = 100 ∧
  knights + liars = total_residents

-- Minimum number of knight-liar pairs to prove
def min_knight_liar_pairs : ℕ := 50

theorem min_pairs {friend_claims_knights friend_claims_liars : ℕ} (h : conditions friend_claims_knights friend_claims_liars) :
    min_knight_liar_pairs = 50 :=
sorry

end min_pairs_l151_151871


namespace min_distance_squared_l151_151253

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l151_151253


namespace solve_for_z_l151_151857

theorem solve_for_z (a b s z : ℝ) (h1 : z ≠ 0) (h2 : 1 - 6 * s ≠ 0) (h3 : z = a^3 * b^2 + 6 * z * s - 9 * s^2) :
  z = (a^3 * b^2 - 9 * s^2) / (1 - 6 * s) := 
 by
  sorry

end solve_for_z_l151_151857


namespace prism_volume_l151_151899

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) : a * b * c = 12 :=
by sorry

end prism_volume_l151_151899


namespace evaluate_expression_l151_151549

theorem evaluate_expression : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) + Int.ceil (4 / 5 : ℚ) + Int.floor (-4 / 5 : ℚ) = 0 :=
by
  sorry

end evaluate_expression_l151_151549


namespace smallest_palindrome_base2_base4_l151_151796

/-- A palindrome is a number that reads the same forward and backward -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = List.reverse digits

/-- The smallest 5-digit palindrome in base 2 can be expressed as a 3-digit palindrome in base 4 -/
theorem smallest_palindrome_base2_base4 :
  ∃ n : ℕ, is_palindrome n 2 ∧ n < 2^5 ∧ n ≥ 2^4 ∧ is_palindrome n 4 ∧ n = 17 :=
by
  exists 17
  sorry

end smallest_palindrome_base2_base4_l151_151796


namespace determine_b_l151_151212

noncomputable def f (x b : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, f (f_inv x) b = x) -> b = 3 :=
by
  intro h
  sorry

end determine_b_l151_151212


namespace steve_pencils_left_l151_151750

-- Conditions
def initial_pencils : Nat := 24
def pencils_given_to_Lauren : Nat := 6
def extra_pencils_given_to_Matt : Nat := 3

-- Question: How many pencils does Steve have left?
theorem steve_pencils_left :
  initial_pencils - (pencils_given_to_Lauren + (pencils_given_to_Lauren + extra_pencils_given_to_Matt)) = 9 := by
  -- You need to provide a proof here
  sorry

end steve_pencils_left_l151_151750


namespace find_y_in_terms_of_x_l151_151575

variable (x y : ℝ)

theorem find_y_in_terms_of_x (hx : x = 5) (hy : y = -4) (hp : ∃ k, y = k * (x - 3)) :
  y = -2 * x + 6 := by
sorry

end find_y_in_terms_of_x_l151_151575


namespace how_many_green_towels_l151_151597

-- Define the conditions
def initial_white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34
def towels_left_after_giving : ℕ := 22

-- Define the statement to prove
theorem how_many_green_towels (G : ℕ) (initial_white : ℕ) (given : ℕ) (left_after : ℕ) :
  initial_white = initial_white_towels →
  given = towels_given_to_mother →
  left_after = towels_left_after_giving →
  (G + initial_white) - given = left_after →
  G = 35 :=
by
  intros
  sorry

end how_many_green_towels_l151_151597


namespace ratio_of_probabilities_l151_151470

-- Define the total number of balls and bins
def balls : ℕ := 20
def bins : ℕ := 6

-- Define the sets A and B based on the given conditions
def A : ℕ := Nat.choose bins 1 * Nat.choose (bins - 1) 1 * (Nat.factorial balls / (Nat.factorial 2 * Nat.factorial 5 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def B : ℕ := Nat.choose bins 2 * (Nat.factorial balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))

-- Define the probabilities p and q
def p : ℚ := A / (Nat.factorial balls * Nat.factorial bins)
def q : ℚ := B / (Nat.factorial balls * Nat.factorial bins)

-- Prove the ratio of probabilities p and q equals 2
theorem ratio_of_probabilities : p / q = 2 := by sorry

end ratio_of_probabilities_l151_151470


namespace bert_initial_amount_l151_151516

theorem bert_initial_amount (n : ℝ) (h : (1 / 2) * (3 / 4 * n - 9) = 12) : n = 44 :=
sorry

end bert_initial_amount_l151_151516


namespace sum_arithmetic_sequence_l151_151128

theorem sum_arithmetic_sequence :
  let a : ℤ := -25
  let d : ℤ := 4
  let a_n : ℤ := 19
  let n : ℤ := (a_n - a) / d + 1
  let S : ℤ := n * (a + a_n) / 2
  S = -36 :=
by 
  let a := -25
  let d := 4
  let a_n := 19
  let n := (a_n - a) / d + 1
  let S := n * (a + a_n) / 2
  show S = -36
  sorry

end sum_arithmetic_sequence_l151_151128


namespace additional_men_joined_l151_151908

theorem additional_men_joined (men_initial : ℕ) (days_initial : ℕ)
  (days_new : ℕ) (additional_men : ℕ) :
  men_initial = 600 →
  days_initial = 20 →
  days_new = 15 →
  (men_initial * days_initial) = ((men_initial + additional_men) * days_new) →
  additional_men = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end additional_men_joined_l151_151908


namespace store_profit_is_20_percent_l151_151635

variable (C : ℝ)
variable (marked_up_price : ℝ := 1.20 * C)          -- First markup price
variable (new_year_price : ℝ := 1.50 * C)           -- Second markup price
variable (discounted_price : ℝ := 1.20 * C)         -- Discounted price in February
variable (profit : ℝ := discounted_price - C)       -- Profit on items sold in February

theorem store_profit_is_20_percent (C : ℝ) : profit = 0.20 * C := 
  sorry

end store_profit_is_20_percent_l151_151635


namespace ellipse_problem_l151_151568

theorem ellipse_problem :
  (∃ (k : ℝ) (a θ : ℝ), 
    (∀ x y : ℝ, y = k * (x + 3) → (x^2 / 25 + y^2 / 16 = 1)) ∧
    (a > -3) ∧
    (∃ x y : ℝ, (x = - (25 / 3) ∧ y = k * (x + 3)) ∧ 
                 (x = D_fst ∧ y = D_snd) ∧ -- Point D(a, θ)
                 (x = M_fst ∧ y = M_snd) ∧ -- Point M
                 (x = N_fst ∧ y = N_snd)) ∧ -- Point N
    (∃ x y : ℝ, (x = -3 ∧ y = 0))) → 
    a = 5 :=
sorry

end ellipse_problem_l151_151568


namespace limit_of_given_function_l151_151517

theorem limit_of_given_function :
  (∀ x ∈ ℝ, 
    tendsto (λ x, (9 - 2 * x) / 3) (𝓝 3) (𝓝 1) ∧ 
    tendsto (λ x, tan (π * x / 6)) (𝓝 3) at_top) → 
  tendsto (λ x, ((9 - 2 * x) / 3) ^ tan (π * x / 6)) (𝓝 3) (𝓝 (exp (4 / π))) :=
begin
  sorry
end

end limit_of_given_function_l151_151517


namespace sum_ages_of_brothers_l151_151718

theorem sum_ages_of_brothers (x : ℝ) (ages : List ℝ) 
  (h1 : ages = [x, x + 1.5, x + 3, x + 4.5, x + 6, x + 7.5, x + 9])
  (h2 : x + 9 = 4 * x) : 
    List.sum ages = 52.5 := 
  sorry

end sum_ages_of_brothers_l151_151718


namespace swimmers_pass_each_other_l151_151896

/-- Two swimmers in a 100-foot pool, one swimming at 4 feet per second, the other at 3 feet per second,
    continuously for 12 minutes, pass each other exactly 32 times. -/
theorem swimmers_pass_each_other 
  (pool_length : ℕ) 
  (time : ℕ) 
  (rate1 : ℕ)
  (rate2 : ℕ)
  (meet_times : ℕ)
  (hp : pool_length = 100) 
  (ht : time = 720) -- 12 minutes = 720 seconds
  (hr1 : rate1 = 4) 
  (hr2 : rate2 = 3)
  : meet_times = 32 := 
sorry

end swimmers_pass_each_other_l151_151896


namespace total_volume_tetrahedra_l151_151256

theorem total_volume_tetrahedra (side_length : ℝ) (x : ℝ) (sqrt_2 : ℝ := Real.sqrt 2) 
  (cube_to_octa_length : x = 2 * (sqrt_2 - 1)) 
  (volume_of_one_tetra : ℝ := ((6 - 4 * sqrt_2) * (3 - sqrt_2)) / 6) :
  side_length = 2 → 
  8 * volume_of_one_tetra = (104 - 72 * sqrt_2) / 3 :=
by
  intros
  sorry

end total_volume_tetrahedra_l151_151256


namespace ellipse_focal_distance_m_value_l151_151537

-- Define the given conditions 
def focal_distance := 2
def ellipse_equation (x y : ℝ) (m : ℝ) := (x^2 / m) + (y^2 / 4) = 1

-- The proof statement
theorem ellipse_focal_distance_m_value :
  ∀ (m : ℝ), 
    (∃ c : ℝ, (2 * c = focal_distance) ∧ (m = 4 + c^2)) →
      m = 5 := by
  sorry

end ellipse_focal_distance_m_value_l151_151537


namespace sum_of_roots_l151_151156

theorem sum_of_roots (r s t : ℝ) (hroots : 3 * (r^3 + s^3 + t^3) + 9 * (r^2 + s^2 + t^2) - 36 * (r + s + t) + 12 = 0) :
  r + s + t = -3 :=
sorry

end sum_of_roots_l151_151156


namespace area_OBEC_is_19_5_l151_151376

-- Definitions for the points and lines from the conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 0⟩
def B : Point := ⟨0, 15⟩
def C : Point := ⟨6, 0⟩
def E : Point := ⟨3, 6⟩

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * |(P1.x * P2.y + P2.x * P3.y + P3.x * P1.y) - (P1.y * P2.x + P2.y * P3.x + P3.y * P1.x)|

-- Definitions of the vertices of the quadrilateral
def O : Point := ⟨0, 0⟩

-- Calculating the area of triangles OCE and OBE
def OCE_area : ℝ := triangle_area O C E
def OBE_area : ℝ := triangle_area O B E

-- Total area of quadrilateral OBEC
def OBEC_area : ℝ := OCE_area + OBE_area

-- Proof statement: The area of quadrilateral OBEC is 19.5
theorem area_OBEC_is_19_5 : OBEC_area = 19.5 := sorry

end area_OBEC_is_19_5_l151_151376


namespace solve_base_6_addition_l151_151058

variables (X Y k : ℕ)

theorem solve_base_6_addition (h1 : Y + 3 = X) (h2 : ∃ k, X + 5 = 2 + 6 * k) : X + Y = 3 :=
sorry

end solve_base_6_addition_l151_151058


namespace min_sum_of_3_digit_numbers_l151_151505

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_3_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999

theorem min_sum_of_3_digit_numbers : 
  ∃ (a b c : ℕ), 
    a ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    b ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    c ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    a + b = c ∧ 
    a + b + c = 459 := 
sorry

end min_sum_of_3_digit_numbers_l151_151505


namespace result_m_plus_n_l151_151015

-- Definitions from the conditions
def like_terms (e1 e2 : ℕ × ℕ) : Prop :=
  e1.2 = e2.2

-- The main statement to prove
theorem result_m_plus_n (m n : ℕ)
  (h1 : like_terms (m, m + 1) (n - 1, 3)) :
  m + n = 5 :=
begin
  sorry
end

end result_m_plus_n_l151_151015


namespace smallest_and_largest_values_l151_151855

theorem smallest_and_largest_values (x : ℕ) (h : x < 100) :
  (x ≡ 2 [MOD 3]) ∧ (x ≡ 2 [MOD 4]) ∧ (x ≡ 2 [MOD 5]) ↔ (x = 2 ∨ x = 62) :=
by
  sorry

end smallest_and_largest_values_l151_151855


namespace central_angle_of_spherical_sector_l151_151361

theorem central_angle_of_spherical_sector (R α r m : ℝ) (h1 : R * Real.pi * r = 2 * R * Real.pi * m) (h2 : R^2 = r^2 + (R - m)^2) :
  α = 2 * Real.arccos (3 / 5) :=
by
  sorry

end central_angle_of_spherical_sector_l151_151361


namespace find_B_value_l151_151072

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l151_151072


namespace value_of_a_l151_151182

theorem value_of_a (m : ℝ) (f : ℝ → ℝ) (h : f = fun x => (1/3)^x + m - 1/3) 
  (h_m : ∀ x, f x ≥ 0 ↔ m ≥ -2/3) : m ≥ -2/3 :=
by
  sorry

end value_of_a_l151_151182


namespace number_of_three_digit_multiples_of_9_with_odd_digits_l151_151980

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def consists_only_of_odd_digits (n : ℕ) : Prop :=
  (∀ d ∈ (n.digits 10), d % 2 = 1)

theorem number_of_three_digit_multiples_of_9_with_odd_digits :
  ∃ t, t = 11 ∧
  (∀ n, is_three_digit_number n ∧ is_multiple_of_9 n ∧ consists_only_of_odd_digits n) → 1 ≤ t ∧ t ≤ 11 :=
sorry

end number_of_three_digit_multiples_of_9_with_odd_digits_l151_151980


namespace general_term_of_sequence_l151_151974

def S (n : ℕ) : ℕ := n^2 + 3 * n + 1

def a (n : ℕ) : ℕ := 
  if n = 1 then 5 
  else 2 * n + 2

theorem general_term_of_sequence (n : ℕ) : 
  a n = if n = 1 then 5 else (S n - S (n - 1)) := 
by 
  sorry

end general_term_of_sequence_l151_151974


namespace sqrt_product_simplifies_l151_151057

theorem sqrt_product_simplifies :
  real.sqrt 12 * real.sqrt 75 = 30 :=
by
  -- proof omitted
  sorry

end sqrt_product_simplifies_l151_151057


namespace lily_spent_amount_l151_151345

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l151_151345


namespace elf_distribution_finite_l151_151595

theorem elf_distribution_finite (infinite_rubies : ℕ → ℕ) (infinite_sapphires : ℕ → ℕ) :
  (∃ n : ℕ, ∀ i j : ℕ, i < n → j < n → (infinite_rubies i > infinite_rubies j → infinite_sapphires i < infinite_sapphires j) ∧
  (infinite_rubies i ≥ infinite_rubies j → infinite_sapphires i < infinite_sapphires j)) ↔
  ∃ k : ℕ, ∀ j : ℕ, j < k :=
sorry

end elf_distribution_finite_l151_151595


namespace jemma_grasshoppers_l151_151198

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l151_151198


namespace balls_in_boxes_l151_151008

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l151_151008


namespace cars_produced_total_l151_151910

theorem cars_produced_total :
  3884 + 2871 = 6755 :=
by
  sorry

end cars_produced_total_l151_151910


namespace obtuse_scalene_triangle_l151_151079

theorem obtuse_scalene_triangle {k : ℕ} (h1 : 13 < k + 17) (h2 : 17 < 13 + k)
  (h3 : 13 < k + 17) (h4 : k ≠ 13) (h5 : k ≠ 17) 
  (h6 : 17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2) 
  (h7 : (k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 22 ∨ 
        k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 ∨ k = 28 ∨ k = 29)) :
  ∃ n, n = 14 := 
by
  sorry

end obtuse_scalene_triangle_l151_151079


namespace number_of_valid_sets_l151_151593

open Finset

def M : Finset ℕ := (range 13).filter (λ n, 1 ≤ n)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def valid_subsets (S : Finset ℕ) : Finset (Finset ℕ) :=
  S.powerset.filter (λ A, A.card = 3 ∧ is_perfect_square (A.sum id))

theorem number_of_valid_sets : (valid_subsets M).card = 26 := sorry

end number_of_valid_sets_l151_151593


namespace point_in_fourth_quadrant_l151_151995

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l151_151995


namespace alicia_tax_deduction_l151_151669

theorem alicia_tax_deduction (earnings_per_hour_in_cents : ℕ) (tax_rate : ℚ) 
  (h1 : earnings_per_hour_in_cents = 2500) (h2 : tax_rate = 0.02) : 
  earnings_per_hour_in_cents * tax_rate = 50 := 
  sorry

end alicia_tax_deduction_l151_151669


namespace most_noteworthy_figure_is_mode_l151_151942

-- Define the types of possible statistics
inductive Statistic
| Median
| Mean
| Mode
| WeightedMean

-- Define a structure for survey data (details abstracted)
structure SurveyData where
  -- fields abstracted for this problem

-- Define the concept of the most noteworthy figure
def most_noteworthy_figure (data : SurveyData) : Statistic :=
  Statistic.Mode

-- Theorem to prove the most noteworthy figure in a survey's data is the mode
theorem most_noteworthy_figure_is_mode (data : SurveyData) :
  most_noteworthy_figure data = Statistic.Mode :=
by
  sorry

end most_noteworthy_figure_is_mode_l151_151942


namespace partA_l151_151780

theorem partA (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, a * x ^ 2 + b * x + c = k ^ 4) : a = 0 ∧ b = 0 := 
sorry

end partA_l151_151780


namespace highest_score_l151_151610

-- Definitions based on conditions
variable (H L : ℕ)

-- Condition (1): H - L = 150
def condition1 : Prop := H - L = 150

-- Condition (2): H + L = 208
def condition2 : Prop := H + L = 208

-- Condition (3): Total runs in 46 innings at an average of 60, excluding two innings averages to 58
def total_runs := 60 * 46
def excluded_runs := total_runs - 2552

theorem highest_score
  (cond1 : condition1 H L)
  (cond2 : condition2 H L)
  : H = 179 :=
by sorry

end highest_score_l151_151610


namespace trapezoid_problem_solution_l151_151051

noncomputable def greatestIntegerNotExceeding (x : ℝ) : ℕ :=
  ⌊x⌋.toNat

theorem trapezoid_problem_solution :
  ∀ (b h : ℝ), b = 37.5 → (h > 0) → 
  let x := 62.5 in
  greatestIntegerNotExceeding (x^2 / 50) = 78 := 
by
  intros b h hb h_pos x
  sorry

end trapezoid_problem_solution_l151_151051


namespace differentiated_roles_grouping_non_differentiated_roles_grouping_l151_151131

section soldier_grouping

variables (n : ℕ)

-- Case 1: Number of ways to form n groups of 2 with differentiated roles (main shooter and assistant shooter)
theorem differentiated_roles_grouping : 
  (nat.factorial (2 * n)) / (nat.factorial n) = (nat.choose (2 * n) n * nat.factorial n) :=
by sorry

-- Case 2: Number of ways to form n groups of 2 without differentiating roles
theorem non_differentiated_roles_grouping : 
  (nat.factorial (2 * n)) / ((2 ^ n) * nat.factorial n) = ((nat.factorial (2 * n)) / (nat.factorial (2 * n))) :=
by sorry

end soldier_grouping

end differentiated_roles_grouping_non_differentiated_roles_grouping_l151_151131


namespace ones_digit_34_pow_34_pow_17_pow_17_l151_151829

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end ones_digit_34_pow_34_pow_17_pow_17_l151_151829


namespace length_of_each_cut_section_xiao_hong_age_l151_151932

theorem length_of_each_cut_section (x : ℝ) (h : 60 - 2 * x = 10) : x = 25 := sorry

theorem xiao_hong_age (y : ℝ) (h : 2 * y + 10 = 30) : y = 10 := sorry

end length_of_each_cut_section_xiao_hong_age_l151_151932


namespace series_sum_is_correct_l151_151155

noncomputable def series_sum : ℝ := ∑' k, 5^((2 : ℕ)^k) / (25^((2 : ℕ)^k) - 1)

theorem series_sum_is_correct : series_sum = 1 / (Real.sqrt 5 - 1) := 
by
  sorry

end series_sum_is_correct_l151_151155


namespace pencil_case_probability_l151_151119

/-- A student has a pencil case with 6 different ballpoint pens: 3 black, 2 red, and 1 blue.
If 2 pens are randomly selected from the case, prove that the probability of both pens being black
is 1/5 and the probability of one pen being black and one pen being blue is also 1/5. -/
theorem pencil_case_probability :
  let total_pens := 6
  let total_combinations := nat.choose 6 2
  let black_pens := 3
  let red_pens := 2
  let blue_pens := 1
  (nat.choose black_pens 2) / total_combinations = 1 / 5 ∧
  (black_pens * blue_pens) / total_combinations = 1 / 5 :=
by
  intros
  have total_combinations := nat.choose 6 2
  have black_combinations := nat.choose 3 2
  have black_blue_combinations := 3 * 1
  split
  case left =>
    calc
      (black_combinations : ℚ) / total_combinations
          = 3 / 15 : by norm_num [total_combinations, black_combinations]
      ... = 1 / 5   : by norm_num
  case right =>
    calc
      (black_blue_combinations : ℚ) / total_combinations
          = 3 / 15 : by norm_num [total_combinations, black_blue_combinations]
      ... = 1 / 5   : by norm_num

end pencil_case_probability_l151_151119


namespace single_elimination_games_needed_l151_151920

theorem single_elimination_games_needed (teams : ℕ) (h : teams = 19) : 
∃ games, games = 18 ∧ (∀ (teams_left : ℕ), teams_left = teams - 1 → games = teams - 1) :=
by
  -- define the necessary parameters and properties here 
  sorry

end single_elimination_games_needed_l151_151920


namespace simplify_root_subtraction_l151_151478

axiom eight_cubed_root : 8^(1/3) = 2
axiom three_hundred_forty_three_cubed_root : 343^(1/3) = 7

theorem simplify_root_subtraction : 8^(1/3) - 343^(1/3) = -5 :=
by {
  rw [eight_cubed_root, three_hundred_forty_three_cubed_root],
  norm_num,
}

end simplify_root_subtraction_l151_151478


namespace a5_a6_val_l151_151031

variable (a : ℕ → ℝ)
variable (r : ℝ)

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a n > 0

axiom a1_a2 : a 1 + a 2 = 1
axiom a3_a4 : a 3 + a 4 = 9

theorem a5_a6_val :
  a 5 + a 6 = 81 :=
by
  sorry

end a5_a6_val_l151_151031


namespace mass_percentage_O_in_Al2_CO3_3_l151_151294

-- Define the atomic masses
def atomic_mass_Al : Float := 26.98
def atomic_mass_C : Float := 12.01
def atomic_mass_O : Float := 16.00

-- Define the formula of aluminum carbonate
def Al_count : Nat := 2
def C_count : Nat := 3
def O_count : Nat := 9

-- Define the molar mass calculation
def molar_mass_Al2_CO3_3 : Float :=
  (Al_count.toFloat * atomic_mass_Al) + 
  (C_count.toFloat * atomic_mass_C) + 
  (O_count.toFloat * atomic_mass_O)

-- Define the mass of oxygen in aluminum carbonate
def mass_O_in_Al2_CO3_3 : Float := O_count.toFloat * atomic_mass_O

-- Define the mass percentage of oxygen in aluminum carbonate
def mass_percentage_O : Float := (mass_O_in_Al2_CO3_3 / molar_mass_Al2_CO3_3) * 100

-- Proof statement
theorem mass_percentage_O_in_Al2_CO3_3 :
  mass_percentage_O = 61.54 := by
  sorry

end mass_percentage_O_in_Al2_CO3_3_l151_151294


namespace price_per_glass_on_second_day_correct_l151_151471

noncomputable def price_per_glass_on_second_day (O : ℝ) (volume_per_glass : ℝ) : ℝ :=
  let P := (0.60 * 2) / 3 in
  P

theorem price_per_glass_on_second_day_correct 
  (O : ℝ) (volume_per_glass : ℝ)
  (h1 : 0 < O)
  (h2 : 0 < volume_per_glass)
  (h3 : (0.60 * (2 * O / volume_per_glass)) = ((price_per_glass_on_second_day O volume_per_glass) * (3 * O / volume_per_glass))) :
  price_per_glass_on_second_day O volume_per_glass = 0.40 := by
  sorry

end price_per_glass_on_second_day_correct_l151_151471


namespace original_amount_of_milk_is_720_l151_151715

variable (M : ℝ) -- The original amount of milk in milliliters

theorem original_amount_of_milk_is_720 :
  ((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)) - ((2 / 3) * (((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)))) = 120 → 
  M = 720 := by
  sorry

end original_amount_of_milk_is_720_l151_151715


namespace lowest_degree_polynomial_l151_151510

-- Define the conditions
def polynomial_conditions (P : ℕ → ℤ) (b : ℤ): Prop :=
  (∃ c, c > b ∧ c ∈ set.range P) ∧ (∃ d, d < b ∧ d ∈ set.range P) ∧ (b ∉ set.range P)

-- The main statement
theorem lowest_degree_polynomial : ∃ P : ℕ → ℤ, polynomial_conditions P 4 ∧ (∀ Q : ℕ → ℤ, polynomial_conditions Q 4 → degree Q >= 4) :=
sorry

end lowest_degree_polynomial_l151_151510


namespace partition_diff_l151_151024

theorem partition_diff {A : Type} (S : Finset ℕ) (S_card : S.card = 67)
  (P : Finset (Finset ℕ)) (P_card : P.card = 4) :
  ∃ (U : Finset ℕ) (hU : U ∈ P), ∃ (a b c : ℕ) (ha : a ∈ U) (hb : b ∈ U) (hc : c ∈ U),
  a = b - c ∧ (1 ≤ a ∧ a ≤ 67) :=
by sorry

end partition_diff_l151_151024


namespace cannot_form_right_triangle_setA_l151_151365

def is_right_triangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem cannot_form_right_triangle_setA (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ¬ is_right_triangle a b c :=
by {
  sorry
}

end cannot_form_right_triangle_setA_l151_151365


namespace solve_for_x_l151_151628

theorem solve_for_x (x : ℝ) (h1 : 1 - x^2 = 0) (h2 : x ≠ 1) : x = -1 := 
by 
  sorry

end solve_for_x_l151_151628


namespace chord_length_of_circle_l151_151153

theorem chord_length_of_circle (x y : ℝ) (h1 : (x - 0)^2 + (y - 2)^2 = 4) (h2 : y = x) : 
  length_of_chord_intercepted_by_line_eq_2sqrt2 :=
sorry

end chord_length_of_circle_l151_151153


namespace surface_area_of_rectangular_prism_l151_151637

theorem surface_area_of_rectangular_prism :
  ∀ (length width height : ℝ), length = 8 → width = 4 → height = 2 → 
    2 * (length * width + length * height + width * height) = 112 :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  sorry

end surface_area_of_rectangular_prism_l151_151637


namespace find_difference_of_squares_l151_151425

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l151_151425


namespace point_above_line_l151_151893

theorem point_above_line (t : ℝ) : (∃ y : ℝ, y = (2 : ℝ)/3) → (t > (2 : ℝ)/3) :=
  by
  intro h
  sorry

end point_above_line_l151_151893


namespace isosceles_triangle_area_l151_151230

theorem isosceles_triangle_area {a b h : ℝ} (h1 : a = 13) (h2 : b = 13) (h3 : h = 10) :
  ∃ (A : ℝ), A = 60 ∧ A = (1 / 2) * h * 12 :=
by
  sorry

end isosceles_triangle_area_l151_151230


namespace problem_l151_151419

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l151_151419


namespace no_rational_roots_of_odd_coeffs_l151_151383

theorem no_rational_roots_of_odd_coeffs (a b c : ℤ) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) (h_c_odd : c % 2 = 1)
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (a * (p / q : ℚ)^2 + b * (p / q : ℚ) + c = 0)) : false :=
sorry

end no_rational_roots_of_odd_coeffs_l151_151383


namespace T_8_equals_546_l151_151706

-- Define the sum of the first n natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the squares of the first n natural numbers
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Define T_n based on the given formula
def T (n : ℕ) : ℕ := (sum_first_n n ^ 2 - sum_squares_first_n n) / 2

-- The proof statement we need to prove
theorem T_8_equals_546 : T 8 = 546 := sorry

end T_8_equals_546_l151_151706


namespace no_3_digit_even_sum_27_l151_151825

/-- Predicate for a 3-digit number -/
def is_3_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate for an even number -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function to compute the digit sum of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem: There are no 3-digit numbers with a digit sum of 27 that are even -/
theorem no_3_digit_even_sum_27 : 
  ∀ n : ℕ, is_3_digit n → digit_sum n = 27 → is_even n → false :=
by
  sorry

end no_3_digit_even_sum_27_l151_151825
