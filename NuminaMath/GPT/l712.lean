import Mathlib

namespace no_integer_solutions_l712_71264

theorem no_integer_solutions :
  ∀ x y : ℤ, x^3 + 4 * x^2 - 11 * x + 30 ≠ 8 * y^3 + 24 * y^2 + 18 * y + 7 :=
by sorry

end no_integer_solutions_l712_71264


namespace pyarelal_loss_l712_71251

/-
Problem statement:
Given the following conditions:
1. Ashok's capital is 1/9 of Pyarelal's.
2. Ashok experienced a loss of 12% on his investment.
3. Pyarelal's loss was 9% of his investment.
4. Their total combined loss is Rs. 2,100.

Prove that the loss incurred by Pyarelal is Rs. 1,829.32.
-/

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio : ℝ) (ashok_loss_percent : ℝ) (pyarelal_loss_percent : ℝ)
  (h1 : ashok_ratio = (1 : ℝ) / 9)
  (h2 : ashok_loss_percent = 0.12)
  (h3 : pyarelal_loss_percent = 0.09)
  (h4 : total_loss = 2100)
  (h5 : total_loss = ashok_loss_percent * (P * ashok_ratio) + pyarelal_loss_percent * P) :
  pyarelal_loss_percent * P = 1829.32 :=
by
  sorry

end pyarelal_loss_l712_71251


namespace count_squares_3x3_grid_count_squares_5x5_grid_l712_71214

/-- Define a mathematical problem: 
  Prove that the number of squares with all four vertices on the dots in a 3x3 grid is 4.
  Prove that the number of squares with all four vertices on the dots in a 5x5 grid is 50.
-/

def num_squares_3x3 : Nat := 4
def num_squares_5x5 : Nat := 50

theorem count_squares_3x3_grid : 
  ∀ (grid_size : Nat), grid_size = 3 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_3x3 = 4)) := 
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

theorem count_squares_5x5_grid : 
  ∀ (grid_size : Nat), grid_size = 5 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_5x5 = 50)) :=
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

end count_squares_3x3_grid_count_squares_5x5_grid_l712_71214


namespace joe_total_toy_cars_l712_71221

def joe_toy_cars (initial_cars additional_cars : ℕ) : ℕ :=
  initial_cars + additional_cars

theorem joe_total_toy_cars : joe_toy_cars 500 120 = 620 := by
  sorry

end joe_total_toy_cars_l712_71221


namespace infinite_squares_in_arithmetic_progression_l712_71296

theorem infinite_squares_in_arithmetic_progression
  (a d : ℕ) (hposd : 0 < d) (hpos : 0 < a) (k n : ℕ)
  (hk : a + k * d = n^2) :
  ∃ (t : ℕ), ∃ (m : ℕ), (a + (k + t) * d = m^2) := by
  sorry

end infinite_squares_in_arithmetic_progression_l712_71296


namespace fraction_irreducible_iff_l712_71203

-- Define the condition for natural number n
def is_natural (n : ℕ) : Prop :=
  True  -- All undergraduate natural numbers abide to True

-- Main theorem formalized in Lean 4
theorem fraction_irreducible_iff (n : ℕ) :
  (∃ (g : ℕ), g = 1 ∧ (∃ a b : ℕ, 2 * n * n + 11 * n - 18 = a * g ∧ n + 7 = b * g)) ↔ 
  (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end fraction_irreducible_iff_l712_71203


namespace minimum_rubles_to_reverse_chips_l712_71297

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end minimum_rubles_to_reverse_chips_l712_71297


namespace third_side_length_l712_71280

def is_odd (n : ℕ) := n % 2 = 1

theorem third_side_length (x : ℕ) (h1 : 2 + 5 > x) (h2 : x + 2 > 5) (h3 : is_odd x) : x = 5 :=
by
  sorry

end third_side_length_l712_71280


namespace no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l712_71250

-- Define the context for part (a)
theorem no_equilateral_triangle_on_grid (x1 y1 x2 y2 x3 y3 : ℤ) :
  ¬ (x1 = x2 ∧ y1 = y2) ∧ (x2 = x3 ∧ y2 = y3) ∧ (x3 = x1 ∧ y3 = y1) ∧ -- vertices must not be the same
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2) ∧ -- sides must be equal
  ((x3 - x1)^2 + (y3 - y1)^2 = (x2 - x1)^2 + (y2 - y1)^2) ->
  false := 
sorry

-- Define the context for part (b)
theorem regular_tetrahedron_on_grid (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ) :
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) ∧ -- first condition: edge lengths equal
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2 = (x4 - x3)^2 + (y4 - y3)^2 + (z4 - z3)^2) ∧ -- second condition: edge lengths equal
  ((x4 - x1)^2 + (y4 - y1)^2 + (z4 - z1)^2 = (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2) -> -- third condition: edge lengths equal
  true := 
sorry

end no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l712_71250


namespace water_consumed_is_correct_l712_71258

def water_consumed (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let camel_ounces := traveler_ounces * camel_multiplier
  let total_ounces := traveler_ounces + camel_ounces
  total_ounces / ounces_per_gallon

theorem water_consumed_is_correct :
  water_consumed 32 7 128 = 2 :=
by
  -- add proof here
  sorry

end water_consumed_is_correct_l712_71258


namespace isosceles_right_triangle_C_coordinates_l712_71209

theorem isosceles_right_triangle_C_coordinates :
  ∃ C : ℝ × ℝ, (let A : ℝ × ℝ := (1, 0)
                let B : ℝ × ℝ := (3, 1) 
                ∃ (x y: ℝ), C = (x, y) ∧ 
                ((x-1)^2 + y^2 = 10) ∧ 
                (((x-3)^2 + (y-1)^2 = 10))) ∨
                ((x = 2 ∧ y = 3) ∨ (x = 4 ∧ y = -1)) :=
by
  sorry

end isosceles_right_triangle_C_coordinates_l712_71209


namespace eel_jellyfish_ratio_l712_71257

noncomputable def combined_cost : ℝ := 200
noncomputable def eel_cost : ℝ := 180
noncomputable def jellyfish_cost : ℝ := combined_cost - eel_cost

theorem eel_jellyfish_ratio : eel_cost / jellyfish_cost = 9 :=
by
  sorry

end eel_jellyfish_ratio_l712_71257


namespace library_average_visitors_l712_71295

theorem library_average_visitors (V : ℝ) (h1 : (4 * 1000 + 26 * V = 750 * 30)) : V = 18500 / 26 := 
by 
  -- The actual proof is omitted and replaced by sorry.
  sorry

end library_average_visitors_l712_71295


namespace total_duration_of_running_l712_71239

-- Definition of conditions
def constant_speed_1 : ℝ := 18
def constant_time_1 : ℝ := 3
def next_distance : ℝ := 70
def average_speed_2 : ℝ := 14

-- Proof statement
theorem total_duration_of_running : 
    let distance_1 := constant_speed_1 * constant_time_1
    let time_2 := next_distance / average_speed_2
    distance_1 = 54 ∧ time_2 = 5 → (constant_time_1 + time_2 = 8) :=
sorry

end total_duration_of_running_l712_71239


namespace perimeter_of_star_is_160_l712_71241

-- Define the radius of the circles
def radius := 5 -- in cm

-- Define the diameter based on radius
def diameter := 2 * radius

-- Define the side length of the square
def side_length_square := 2 * diameter

-- Define the side length of each equilateral triangle
def side_length_triangle := side_length_square

-- Define the perimeter of the four-pointed star
def perimeter_star := 8 * side_length_triangle

-- Statement: The perimeter of the star is 160 cm
theorem perimeter_of_star_is_160 :
  perimeter_star = 160 := by
    sorry

end perimeter_of_star_is_160_l712_71241


namespace sum_digit_product_1001_to_2011_l712_71289

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc => d * acc) 1

theorem sum_digit_product_1001_to_2011 :
  (Finset.range 1011).sum (λ k => digit_product (1001 + k)) = 91125 :=
by
  sorry

end sum_digit_product_1001_to_2011_l712_71289


namespace volume_maximized_at_r_5_h_8_l712_71238

noncomputable def V (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- (1) Given that the total construction cost is 12000π yuan, 
express the volume V as a function of the radius r, and determine its domain. -/
def volume_function (r : ℝ) (h : ℝ) (cost : ℝ) : Prop :=
  cost = 12000 * Real.pi ∧
  h = 1 / (5 * r) * (300 - 4 * r^2) ∧
  V r = Real.pi * r^2 * h ∧
  0 < r ∧ r < 5 * Real.sqrt 3

/-- (2) Prove V(r) is maximized when r = 5 and h = 8 -/
theorem volume_maximized_at_r_5_h_8 :
  ∀ (r : ℝ) (h : ℝ) (cost : ℝ), volume_function r h cost → 
  ∃ (r_max : ℝ) (h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧ ∀ x, 0 < x → x < 5 * Real.sqrt 3 → V x ≤ V r_max :=
by
  intros r h cost hvolfunc
  sorry

end volume_maximized_at_r_5_h_8_l712_71238


namespace canteen_consumption_l712_71202

theorem canteen_consumption :
  ∀ (x : ℕ),
    (x + (500 - x) + (200 - x)) = 700 → 
    (500 - x) = 7 * (200 - x) →
    x = 150 :=
by
  sorry

end canteen_consumption_l712_71202


namespace f_sq_add_g_sq_eq_one_f_even_f_periodic_l712_71291

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x : ℝ, g (-x) = - g x
axiom f_0 : f 0 = 1
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_sq_add_g_sq_eq_one (x : ℝ) : f x ^ 2 + g x ^ 2 = 1 :=
sorry

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem f_periodic (a : ℝ) (ha : a ≠ 0) (hfa : f a = 1) : ∀ x : ℝ, f (x + a) = f x :=
sorry

end f_sq_add_g_sq_eq_one_f_even_f_periodic_l712_71291


namespace min_value_expr_l712_71212

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l712_71212


namespace jennifer_dogs_l712_71262

theorem jennifer_dogs (D : ℕ) (groom_time_per_dog : ℕ) (groom_days : ℕ) (total_groom_time : ℕ) :
  groom_time_per_dog = 20 →
  groom_days = 30 →
  total_groom_time = 1200 →
  groom_days * (groom_time_per_dog * D) = total_groom_time →
  D = 2 :=
by
  intro h1 h2 h3 h4
  sorry

end jennifer_dogs_l712_71262


namespace claudia_coins_l712_71245

theorem claudia_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 29 - x = 26) :
  y = 12 :=
by
  sorry

end claudia_coins_l712_71245


namespace isabella_non_yellow_houses_l712_71284

variable (Green Yellow Red Blue Pink : ℕ)

axiom h1 : 3 * Yellow = Green
axiom h2 : Red = Yellow + 40
axiom h3 : Green = 90
axiom h4 : Blue = (Green + Yellow) / 2
axiom h5 : Pink = (Red / 2) + 15

theorem isabella_non_yellow_houses : (Green + Red + Blue + Pink - Yellow) = 270 :=
by 
  sorry

end isabella_non_yellow_houses_l712_71284


namespace nine_possible_xs_l712_71207

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l712_71207


namespace relationship_among_values_l712_71259

-- Assume there exists a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing on (0, 3)
def increasing_on_0_to_3 : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f x < f y

-- Condition 2: f(x + 3) is an even function
def even_function_shifted : Prop :=
  ∀ x : ℝ, f (x + 3) = f (-(x + 3))

-- The theorem we need to prove
theorem relationship_among_values 
  (h1 : increasing_on_0_to_3 f)
  (h2 : even_function_shifted f) :
  f (9/2) < f 2 ∧ f 2 < f (7/2) :=
sorry

end relationship_among_values_l712_71259


namespace find_c_k_l712_71227

theorem find_c_k (a b : ℕ → ℕ) (c : ℕ → ℕ) (k : ℕ) (d r : ℕ) 
  (h1 : ∀ n, a n = 1 + (n-1)*d)
  (h2 : ∀ n, b n = r^(n-1))
  (h3 : ∀ n, c n = a n + b n)
  (h4 : c (k-1) = 80)
  (h5 : c (k+1) = 500) :
  c k = 167 := sorry

end find_c_k_l712_71227


namespace alvin_earns_14_dollars_l712_71222

noncomputable def total_earnings (total_marbles : ℕ) (percent_white percent_black : ℚ)
  (price_white price_black price_colored : ℚ) : ℚ :=
  let white_marbles := percent_white * total_marbles
  let black_marbles := percent_black * total_marbles
  let colored_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles * price_white) + (black_marbles * price_black) + (colored_marbles * price_colored)

theorem alvin_earns_14_dollars :
  total_earnings 100 (20/100) (30/100) 0.05 0.10 0.20 = 14 := by
  sorry

end alvin_earns_14_dollars_l712_71222


namespace jake_has_fewer_peaches_than_steven_l712_71299

theorem jake_has_fewer_peaches_than_steven :
  ∀ (jillPeaches jakePeaches stevenPeaches : ℕ),
    jillPeaches = 12 →
    jakePeaches = jillPeaches - 1 →
    stevenPeaches = jillPeaches + 15 →
    stevenPeaches - jakePeaches = 16 :=
  by
    intros jillPeaches jakePeaches stevenPeaches
    intro h_jill
    intro h_jake
    intro h_steven
    sorry

end jake_has_fewer_peaches_than_steven_l712_71299


namespace predict_HCl_formed_l712_71237

-- Define the initial conditions and chemical reaction constants
def initial_moles_CH4 : ℝ := 3
def initial_moles_Cl2 : ℝ := 6
def volume : ℝ := 2

-- Define the reaction stoichiometry constants
def stoich_CH4_to_HCl : ℝ := 2
def stoich_CH4 : ℝ := 1
def stoich_Cl2 : ℝ := 2

-- Declare the hypothesis that reaction goes to completion
axiom reaction_goes_to_completion : Prop

-- Define the function to calculate the moles of HCl formed
def moles_HCl_formed : ℝ :=
  initial_moles_CH4 * stoich_CH4_to_HCl

-- Prove the predicted amount of HCl formed is 6 moles under the given conditions
theorem predict_HCl_formed : reaction_goes_to_completion → moles_HCl_formed = 6 := by
  sorry

end predict_HCl_formed_l712_71237


namespace gcd_9247_4567_eq_1_l712_71204

theorem gcd_9247_4567_eq_1 : Int.gcd 9247 4567 = 1 := sorry

end gcd_9247_4567_eq_1_l712_71204


namespace draws_alternate_no_consecutive_same_color_l712_71231

-- Defining the total number of balls and the count of each color.
def total_balls : ℕ := 15
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 5

-- Defining the probability that the draws alternate in colors with no two consecutive balls of the same color.
def probability_no_consecutive_same_color : ℚ := 162 / 1001

theorem draws_alternate_no_consecutive_same_color :
  (white_balls + black_balls + red_balls = total_balls) →
  -- The resulting probability based on the given conditions.
  probability_no_consecutive_same_color = 162 / 1001 := by
  sorry

end draws_alternate_no_consecutive_same_color_l712_71231


namespace frequency_total_students_l712_71279

noncomputable def total_students (known : ℕ) (freq : ℝ) : ℝ :=
known / freq

theorem frequency_total_students (known : ℕ) (freq : ℝ) (h1 : known = 40) (h2 : freq = 0.8) :
  total_students known freq = 50 :=
by
  rw [total_students, h1, h2]
  norm_num

end frequency_total_students_l712_71279


namespace yellow_yellow_pairs_l712_71243

variable (students_total : ℕ := 150)
variable (blue_students : ℕ := 65)
variable (yellow_students : ℕ := 85)
variable (total_pairs : ℕ := 75)
variable (blue_blue_pairs : ℕ := 30)

theorem yellow_yellow_pairs : 
  (yellow_students - (blue_students - blue_blue_pairs * 2)) / 2 = 40 :=
by 
  -- proof goes here
  sorry

end yellow_yellow_pairs_l712_71243


namespace sin_C_l712_71229

variable {A B C : ℝ}

theorem sin_C (hA : A = 90) (hcosB : Real.cos B = 3/5) : Real.sin (90 - B) = 3/5 :=
by
  sorry

end sin_C_l712_71229


namespace number_of_dimes_l712_71236

theorem number_of_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 200) : d = 14 := 
sorry

end number_of_dimes_l712_71236


namespace flower_seedlings_pots_l712_71285

theorem flower_seedlings_pots (x y z : ℕ) :
  (1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) →
  (x + y + z = 16) →
  (2 * x + 4 * y + 10 * z = 50) →
  (x = 10 ∨ x = 13) :=
by
  intros h1 h2 h3
  sorry

end flower_seedlings_pots_l712_71285


namespace simplify_polynomials_l712_71263

theorem simplify_polynomials :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomials_l712_71263


namespace molly_age_l712_71220

theorem molly_age : 14 + 6 = 20 := by
  sorry

end molly_age_l712_71220


namespace parking_lot_perimeter_l712_71206

theorem parking_lot_perimeter (a b : ℝ) (h₁ : a^2 + b^2 = 625) (h₂ : a * b = 168) :
  2 * (a + b) = 62 :=
sorry

end parking_lot_perimeter_l712_71206


namespace find_a_l712_71201

-- Definitions of the conditions
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

-- The proof goal
theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 := 
by 
  sorry

end find_a_l712_71201


namespace remainder_of_n_div_11_is_1_l712_71277

def A : ℕ := 20072009
def n : ℕ := 100 * A

theorem remainder_of_n_div_11_is_1 :
  (n % 11) = 1 :=
sorry

end remainder_of_n_div_11_is_1_l712_71277


namespace reynald_soccer_balls_l712_71217

theorem reynald_soccer_balls (total_balls basketballs_more soccer tennis baseball more_baseballs volleyballs : ℕ) 
(h_total_balls: total_balls = 145) 
(h_basketballs_more: basketballs_more = 5)
(h_tennis: tennis = 2 * soccer)
(h_more_baseballs: more_baseballs = 10)
(h_volleyballs: volleyballs = 30) 
(sum_eq: soccer + (soccer + basketballs_more) + tennis + (soccer + more_baseballs) + volleyballs = total_balls) : soccer = 20 := 
by
  sorry

end reynald_soccer_balls_l712_71217


namespace least_subtracted_number_l712_71276

theorem least_subtracted_number (a b c d e : ℕ) 
  (h₁ : a = 2590) 
  (h₂ : b = 9) 
  (h₃ : c = 11) 
  (h₄ : d = 13) 
  (h₅ : e = 6) 
  : ∃ (x : ℕ), a - x % b = e ∧ a - x % c = e ∧ a - x % d = e := by
  sorry

end least_subtracted_number_l712_71276


namespace solve_abs_inequality_l712_71268

theorem solve_abs_inequality (x : ℝ) :
  (|x - 2| + |x - 4| > 6) ↔ (x < 0 ∨ 12 < x) :=
by
  sorry

end solve_abs_inequality_l712_71268


namespace sample_size_l712_71225

theorem sample_size (F n : ℕ) (FR : ℚ) (h1: F = 36) (h2: FR = 1/4) (h3: FR = F / n) : n = 144 :=
by 
  sorry

end sample_size_l712_71225


namespace sample_size_survey_l712_71261

theorem sample_size_survey (students_selected : ℕ) (h : students_selected = 200) : students_selected = 200 :=
by
  assumption

end sample_size_survey_l712_71261


namespace unique_real_solution_k_eq_35_over_4_l712_71260

theorem unique_real_solution_k_eq_35_over_4 :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ (k = 35 / 4) :=
by
  sorry

end unique_real_solution_k_eq_35_over_4_l712_71260


namespace find_petra_age_l712_71208

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

end find_petra_age_l712_71208


namespace words_per_page_l712_71254

theorem words_per_page (p : ℕ) (h1 : p ≤ 150) (h2 : 120 * p ≡ 172 [MOD 221]) : p = 114 := by
  sorry

end words_per_page_l712_71254


namespace inverse_of_congruence_implies_equal_area_l712_71210

-- Definitions to capture conditions and relationships
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with congruency of two triangles
  sorry

def equal_areas (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with equal areas of two triangles
  sorry

-- Statement to prove the inverse proposition
theorem inverse_of_congruence_implies_equal_area :
  (∀ T1 T2 : Triangle, congruent_triangles T1 T2 → equal_areas T1 T2) →
  (∀ T1 T2 : Triangle, equal_areas T1 T2 → congruent_triangles T1 T2) :=
  sorry

end inverse_of_congruence_implies_equal_area_l712_71210


namespace exists_positive_int_solutions_l712_71219

theorem exists_positive_int_solutions (a : ℕ) (ha : a > 2) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 :=
by
  sorry

end exists_positive_int_solutions_l712_71219


namespace arithmetic_sequence_ratio_l712_71278

theorem arithmetic_sequence_ratio
  (d : ℕ) (h₀ : d ≠ 0)
  (a : ℕ → ℕ)
  (h₁ : ∀ n, a (n + 1) = a n + d)
  (h₂ : (a 3)^2 = (a 1) * (a 9)) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
  sorry

end arithmetic_sequence_ratio_l712_71278


namespace ribbons_jane_uses_l712_71267

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l712_71267


namespace jason_stacked_bales_l712_71215

theorem jason_stacked_bales (initial_bales : ℕ) (final_bales : ℕ) (stored_bales : ℕ) 
  (h1 : initial_bales = 73) (h2 : final_bales = 96) : stored_bales = final_bales - initial_bales := 
by
  rw [h1, h2]
  sorry

end jason_stacked_bales_l712_71215


namespace solve_exp_eq_l712_71242

theorem solve_exp_eq (x : ℝ) (h : Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 2) : 
  x = 0 := 
sorry

end solve_exp_eq_l712_71242


namespace perpendicular_distance_is_8_cm_l712_71288

theorem perpendicular_distance_is_8_cm :
  ∀ (side_length distance_from_corner cut_angle : ℝ),
    side_length = 100 →
    distance_from_corner = 8 →
    cut_angle = 45 →
    (∃ h : ℝ, h = 8) :=
by
  intros side_length distance_from_corner cut_angle hms d8 a45
  sorry

end perpendicular_distance_is_8_cm_l712_71288


namespace find_number_l712_71275

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 := by
  sorry

end find_number_l712_71275


namespace cinematic_academy_members_l712_71290

theorem cinematic_academy_members (h1 : ∀ x, x / 4 ≥ 196.25 → x ≥ 785) : 
  ∃ n : ℝ, 1 / 4 * n = 196.25 ∧ n = 785 :=
by
  sorry

end cinematic_academy_members_l712_71290


namespace set_intersection_complement_l712_71292

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
noncomputable def M : Set ℕ := {2, 3, 4, 5}
noncomputable def N : Set ℕ := {1, 4, 5, 7}

theorem set_intersection_complement :
  M ∩ (U \ N) = {2, 3} :=
by
  sorry

end set_intersection_complement_l712_71292


namespace number_subtraction_l712_71234

theorem number_subtraction
  (x : ℕ) (y : ℕ)
  (h1 : x = 30)
  (h2 : 8 * x - y = 102) : y = 138 :=
by 
  sorry

end number_subtraction_l712_71234


namespace rectangle_area_l712_71287

theorem rectangle_area (r : ℝ) (L W : ℝ) (h₀ : r = 7) (h₁ : 2 * r = W) (h₂ : L / W = 3) : 
  L * W = 588 :=
by sorry

end rectangle_area_l712_71287


namespace condition_2_3_implies_f_x1_greater_f_x2_l712_71274

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem condition_2_3_implies_f_x1_greater_f_x2 
(x1 x2 : ℝ) (h1 : -2 * Real.pi / 3 ≤ x1 ∧ x1 ≤ 2 * Real.pi / 3) 
(h2 : -2 * Real.pi / 3 ≤ x2 ∧ x2 ≤ 2 * Real.pi / 3) 
(hx1_sq_gt_x2_sq : x1^2 > x2^2) (hx1_gt_abs_x2 : x1 > |x2|) : 
  f x1 > f x2 := 
sorry

end condition_2_3_implies_f_x1_greater_f_x2_l712_71274


namespace ann_total_fare_for_100_miles_l712_71270

-- Conditions
def base_fare : ℕ := 20
def fare_per_distance (distance : ℕ) : ℕ := 180 * distance / 80

-- Question: How much would Ann be charged if she traveled 100 miles?
def total_fare (distance : ℕ) : ℕ := (fare_per_distance distance) + base_fare

-- Prove that the total fare for 100 miles is 245 dollars
theorem ann_total_fare_for_100_miles : total_fare 100 = 245 :=
by
  -- Adding your proof here
  sorry

end ann_total_fare_for_100_miles_l712_71270


namespace kitten_length_doubling_l712_71255

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l712_71255


namespace archer_scores_distribution_l712_71266

structure ArcherScores where
  hits_40 : ℕ
  hits_39 : ℕ
  hits_24 : ℕ
  hits_23 : ℕ
  hits_17 : ℕ
  hits_16 : ℕ
  total_score : ℕ

theorem archer_scores_distribution
  (dora : ArcherScores)
  (reggie : ArcherScores)
  (finch : ArcherScores)
  (h1 : dora.total_score = 120)
  (h2 : reggie.total_score = 110)
  (h3 : finch.total_score = 100)
  (h4 : dora.hits_40 + dora.hits_39 + dora.hits_24 + dora.hits_23 + dora.hits_17 + dora.hits_16 = 6)
  (h5 : reggie.hits_40 + reggie.hits_39 + reggie.hits_24 + reggie.hits_23 + reggie.hits_17 + reggie.hits_16 = 6)
  (h6 : finch.hits_40 + finch.hits_39 + finch.hits_24 + finch.hits_23 + finch.hits_17 + finch.hits_16 = 6)
  (h7 : 40 * dora.hits_40 + 39 * dora.hits_39 + 24 * dora.hits_24 + 23 * dora.hits_23 + 17 * dora.hits_17 + 16 * dora.hits_16 = 120)
  (h8 : 40 * reggie.hits_40 + 39 * reggie.hits_39 + 24 * reggie.hits_24 + 23 * reggie.hits_23 + 17 * reggie.hits_17 + 16 * reggie.hits_16 = 110)
  (h9 : 40 * finch.hits_40 + 39 * finch.hits_39 + 24 * finch.hits_24 + 23 * finch.hits_23 + 17 * finch.hits_17 + 16 * finch.hits_16 = 100)
  (h10 : dora.hits_40 = 1)
  (h11 : dora.hits_39 = 0)
  (h12 : dora.hits_24 = 0) :
  dora.hits_40 = 1 ∧ dora.hits_16 = 5 ∧ 
  reggie.hits_23 = 2 ∧ reggie.hits_16 = 4 ∧ 
  finch.hits_17 = 4 ∧ finch.hits_16 = 2 :=
sorry

end archer_scores_distribution_l712_71266


namespace number_of_rows_is_ten_l712_71228

-- Definition of the arithmetic sequence
def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (3 * n + 1) / 2

-- The main theorem to prove
theorem number_of_rows_is_ten :
  (∃ n : ℕ, arithmetic_sequence_sum n = 145) ↔ n = 10 :=
by
  sorry

end number_of_rows_is_ten_l712_71228


namespace constant_term_in_expansion_l712_71248

-- Given conditions
def eq_half_n_minus_m_zero (n m : ℕ) : Prop := 1/2 * n = m
def eq_n_plus_m_ten (n m : ℕ) : Prop := n + m = 10
noncomputable def binom (n k : ℕ) : ℝ := Real.exp (Real.log (Nat.factorial n) - Real.log (Nat.factorial k) - Real.log (Nat.factorial (n - k)))

-- Main theorem
theorem constant_term_in_expansion : 
  ∃ (n m : ℕ), eq_half_n_minus_m_zero n m ∧ eq_n_plus_m_ten n m ∧ 
  binom 10 m * (3^4 : ℝ) = 17010 :=
by
  -- Definitions translation
  sorry

end constant_term_in_expansion_l712_71248


namespace line_through_two_points_l712_71233

theorem line_through_two_points (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 4)) :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ (k = 1) ∧ (b = 1) := 
by
  sorry

end line_through_two_points_l712_71233


namespace monotonically_increasing_interval_l712_71249

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l712_71249


namespace dividend_is_144_l712_71211

theorem dividend_is_144 
  (Q : ℕ) (D : ℕ) (M : ℕ)
  (h1 : M = 6 * D)
  (h2 : D = 4 * Q) 
  (Q_eq_6 : Q = 6) : 
  M = 144 := 
sorry

end dividend_is_144_l712_71211


namespace value_of_expression_l712_71281

theorem value_of_expression : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end value_of_expression_l712_71281


namespace snow_white_last_trip_l712_71218

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l712_71218


namespace simplify_expression_l712_71226

theorem simplify_expression :
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 :=
by sorry

end simplify_expression_l712_71226


namespace reflection_about_x_axis_l712_71256

theorem reflection_about_x_axis (a : ℝ) : 
  (A : ℝ × ℝ) = (3, a) → (B : ℝ × ℝ) = (3, 4) → A = (3, -4) → a = -4 :=
by
  intros A_eq B_eq reflection_eq
  sorry

end reflection_about_x_axis_l712_71256


namespace initial_bacteria_count_l712_71282

theorem initial_bacteria_count (doubling_interval : ℕ) (initial_count four_minutes_final_count : ℕ)
  (h1 : doubling_interval = 30)
  (h2 : four_minutes_final_count = 524288)
  (h3 : ∀ t : ℕ, initial_count * 2 ^ (t / doubling_interval) = four_minutes_final_count) :
  initial_count = 2048 :=
sorry

end initial_bacteria_count_l712_71282


namespace solution_l712_71230

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry
noncomputable def x7 : ℝ := sorry
noncomputable def x8 : ℝ := sorry

axiom cond1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 + 64 * x8 = 10
axiom cond2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 + 81 * x8 = 40
axiom cond3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 + 100 * x8 = 170

theorem solution : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 + 121 * x8 = 400 := 
by
  sorry

end solution_l712_71230


namespace initial_weight_of_cheese_l712_71252

theorem initial_weight_of_cheese :
  let initial_weight : Nat := 850
  -- final state after 3 bites
  let final_weight1 : Nat := 25
  let final_weight2 : Nat := 25
  -- third state
  let third_weight1 : Nat := final_weight1 + final_weight2
  let third_weight2 : Nat := final_weight1
  -- second state
  let second_weight1 : Nat := third_weight1 + third_weight2
  let second_weight2 : Nat := third_weight1
  -- first state
  let first_weight1 : Nat := second_weight1 + second_weight2
  let first_weight2 : Nat := second_weight1
  -- initial state
  let initial_weight1 : Nat := first_weight1 + first_weight2
  let initial_weight2 : Nat := first_weight1
  initial_weight = initial_weight1 + initial_weight2 :=
by
  sorry

end initial_weight_of_cheese_l712_71252


namespace sin_of_angle_in_first_quadrant_l712_71232

theorem sin_of_angle_in_first_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 3 / 4) : Real.sin α = 3 / 5 :=
by
  sorry

end sin_of_angle_in_first_quadrant_l712_71232


namespace sons_age_l712_71246

theorem sons_age (S M : ℕ) (h1 : M = 3 * S) (h2 : M + 12 = 2 * (S + 12)) : S = 12 :=
by 
  sorry

end sons_age_l712_71246


namespace yellow_tint_percentage_new_mixture_l712_71286

def original_volume : ℝ := 40
def yellow_tint_percentage : ℝ := 0.35
def additional_yellow_tint : ℝ := 10
def new_volume : ℝ := original_volume + additional_yellow_tint
def original_yellow_tint : ℝ := yellow_tint_percentage * original_volume
def new_yellow_tint : ℝ := original_yellow_tint + additional_yellow_tint

theorem yellow_tint_percentage_new_mixture : 
  (new_yellow_tint / new_volume) * 100 = 48 := 
by
  sorry

end yellow_tint_percentage_new_mixture_l712_71286


namespace jonathan_needs_more_money_l712_71265

def cost_dictionary : ℕ := 11
def cost_dinosaur_book : ℕ := 19
def cost_childrens_cookbook : ℕ := 7
def saved_money : ℕ := 8

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
def amount_needed : ℕ := total_cost - saved_money

theorem jonathan_needs_more_money : amount_needed = 29 := by
  have h1 : total_cost = 37 := by
    show 11 + 19 + 7 = 37
    sorry
  show 37 - 8 = 29
  sorry

end jonathan_needs_more_money_l712_71265


namespace steps_already_climbed_l712_71235

-- Definitions based on conditions
def total_stair_steps : ℕ := 96
def steps_left_to_climb : ℕ := 22

-- Theorem proving the number of steps already climbed
theorem steps_already_climbed : total_stair_steps - steps_left_to_climb = 74 := by
  sorry

end steps_already_climbed_l712_71235


namespace part1_part2_part3_l712_71253

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)
def g (x : ℝ) : ℝ := f x - abs (x - 2)

theorem part1 : ∀ x : ℝ, f x ≤ 8 ↔ (-11 ≤ x ∧ x ≤ 5) := by sorry

theorem part2 : ∃ x : ℝ, g x = 5 := by sorry

theorem part3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 5) : 
  1 / a + 9 / b = 16 / 5 := by sorry

end part1_part2_part3_l712_71253


namespace leo_third_part_time_l712_71273

-- Definitions to represent the conditions
def total_time : ℕ := 120
def first_part_time : ℕ := 25
def second_part_time : ℕ := 2 * first_part_time

-- Proposition to prove
theorem leo_third_part_time :
  total_time - (first_part_time + second_part_time) = 45 :=
by
  sorry

end leo_third_part_time_l712_71273


namespace water_needed_to_fill_glasses_l712_71244

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end water_needed_to_fill_glasses_l712_71244


namespace ratio_A_to_B_l712_71271

theorem ratio_A_to_B (A B C : ℕ) (h1 : A + B + C = 406) (h2 : C = 232) (h3 : B = C / 2) : A / gcd A B = 1 ∧ B / gcd A B = 2 := 
by sorry

end ratio_A_to_B_l712_71271


namespace inverse_of_g_l712_71213

theorem inverse_of_g : 
  ∀ (g g_inv : ℝ → ℝ) (p q r s : ℝ),
  (∀ x, g x = (3 * x - 2) / (x + 4)) →
  (∀ x, g_inv x = (p * x + q) / (r * x + s)) →
  (∀ x, g (g_inv x) = x) →
  q / s = 2 / 3 :=
by
  intros g g_inv p q r s h_g h_g_inv h_g_ginv
  sorry

end inverse_of_g_l712_71213


namespace min_stool_height_l712_71272

/-
Alice needs to reach a ceiling fan switch located 15 centimeters below a 3-meter-tall ceiling.
Alice is 160 centimeters tall and can reach 50 centimeters above her head. She uses a stack of books
12 centimeters tall to assist her reach. We aim to show that the minimum height of the stool she needs is 63 centimeters.
-/

def ceiling_height_cm : ℕ := 300
def alice_height_cm : ℕ := 160
def reach_above_head_cm : ℕ := 50
def books_height_cm : ℕ := 12
def switch_below_ceiling_cm : ℕ := 15

def total_reach_with_books := alice_height_cm + reach_above_head_cm + books_height_cm
def switch_height_from_floor := ceiling_height_cm - switch_below_ceiling_cm

theorem min_stool_height : total_reach_with_books + 63 = switch_height_from_floor := by
  unfold total_reach_with_books switch_height_from_floor
  sorry

end min_stool_height_l712_71272


namespace sum_of_first_2n_terms_l712_71216

-- Definitions based on conditions
variable (n : ℕ) (S : ℕ → ℝ)

-- Conditions
def condition1 : Prop := S n = 24
def condition2 : Prop := S (3 * n) = 42

-- Statement to be proved
theorem sum_of_first_2n_terms {n : ℕ} (S : ℕ → ℝ) 
    (h1 : S n = 24) (h2 : S (3 * n) = 42) : S (2 * n) = 36 := by
  sorry

end sum_of_first_2n_terms_l712_71216


namespace max_c_magnitude_l712_71247

variables {a b c : ℝ × ℝ}

-- Definitions of the given conditions
def unit_vector (v : ℝ × ℝ) : Prop := ‖v‖ = 1
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def satisfied_c (c a b : ℝ × ℝ) : Prop := ‖c - (a + b)‖ = 2

-- Main theorem to prove
theorem max_c_magnitude (ha : unit_vector a) (hb : unit_vector b) (hab : orthogonal a b) (hc : satisfied_c c a b) : ‖c‖ ≤ 2 + Real.sqrt 2 := 
sorry

end max_c_magnitude_l712_71247


namespace ratio_of_r_to_pq_l712_71298

theorem ratio_of_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 7000) (h₂ : r = 2800) :
  r / (p + q) = 2 / 3 :=
by sorry

end ratio_of_r_to_pq_l712_71298


namespace find_r_l712_71205

theorem find_r (k r : ℝ) (h1 : (5 = k * 3^r)) (h2 : (45 = k * 9^r)) : r = 2 :=
  sorry

end find_r_l712_71205


namespace angle_C_eq_pi_over_3_l712_71293

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end angle_C_eq_pi_over_3_l712_71293


namespace maximum_z_l712_71200

theorem maximum_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end maximum_z_l712_71200


namespace problem_statement_l712_71223

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end problem_statement_l712_71223


namespace fare_collected_from_I_class_l712_71224

theorem fare_collected_from_I_class (x y : ℕ) 
  (h_ratio_passengers : 4 * x = 4 * x) -- ratio of passengers 1:4
  (h_ratio_fare : 3 * y = 3 * y) -- ratio of fares 3:1
  (h_total_fare : 7 * 3 * x * y = 224000) -- total fare Rs. 224000
  : 3 * x * y = 96000 := 
by
  sorry

end fare_collected_from_I_class_l712_71224


namespace intersection_setA_setB_l712_71240

namespace Proof

def setA : Set ℝ := {x | ∃ y : ℝ, y = x + 1}
def setB : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem intersection_setA_setB : (setA ∩ setB) = {y | 0 < y} :=
by
  sorry

end Proof

end intersection_setA_setB_l712_71240


namespace a6_value_l712_71283

theorem a6_value
  (a : ℕ → ℤ)
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (geo : ∃ q : ℤ, ∀ n : ℕ, n > 0 → a (n + 1) = q^n * (a 1 + 1) - 1):
  a 6 = 63 :=
by
  sorry

end a6_value_l712_71283


namespace picture_area_l712_71294

theorem picture_area (x y : ℕ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (3*x + 3) * (y + 2) = 110) : x * y = 28 :=
by {
  sorry
}

end picture_area_l712_71294


namespace identify_stolen_bag_with_two_weighings_l712_71269

-- Definition of the weights of the nine bags
def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Statement of the problem: Using two weighings on a balance scale without weights,
-- prove that it is possible to identify the specific bag from which the treasure was stolen.
theorem identify_stolen_bag_with_two_weighings (stolen_bag : {n // n < 9}) :
  ∃ (group1 group2 : List ℕ), group1 ≠ group2 ∧ (group1.sum = 11 ∨ group1.sum = 15) ∧ (group2.sum = 11 ∨ group2.sum = 15) →
  ∃ (b1 b2 b3 : ℕ), b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ b1 + b2 + b3 = 6 ∧ (b1 + b2 = 11 ∨ b1 + b2 = 15) := sorry

end identify_stolen_bag_with_two_weighings_l712_71269
