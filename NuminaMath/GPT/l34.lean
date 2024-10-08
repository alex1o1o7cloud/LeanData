import Mathlib

namespace sum_of_interior_edges_l34_34773

def frame_width : ℝ := 1
def outer_length : ℝ := 5
def frame_area : ℝ := 18
def inner_length1 : ℝ := outer_length - 2 * frame_width

/-- Given conditions and required to prove:
1. The frame is made of one-inch-wide pieces of wood.
2. The area of just the frame is 18 square inches.
3. One of the outer edges of the frame is 5 inches long.
Prove: The sum of the lengths of the four interior edges is 14 inches.
-/
theorem sum_of_interior_edges (inner_length2 : ℝ) 
  (h1 : (outer_length * (inner_length2 + 2) - inner_length1 * inner_length2) = frame_area)
  (h2 : (inner_length2 - 2) / 2 = 1) : 
  inner_length1 + inner_length1 + inner_length2 + inner_length2 = 14 :=
by
  sorry

end sum_of_interior_edges_l34_34773


namespace exists_inequality_l34_34241

theorem exists_inequality (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (hx1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (h_n : 2 ≤ n) : 
  ∃ i : Fin n, x i * (1 - x (i + 1)) ≥ (1 / 4) * x 0 * (1 - x n) :=
sorry

end exists_inequality_l34_34241


namespace triangle_area_l34_34534

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_area :
  area_of_triangle 0 0 0 6 8 0 = 24 :=
by
  sorry

end triangle_area_l34_34534


namespace pumps_fill_time_l34_34148

-- Definitions for the rates and the time calculation
def small_pump_rate : ℚ := 1 / 3
def large_pump_rate : ℚ := 4
def third_pump_rate : ℚ := 1 / 2

def total_pump_rate : ℚ := small_pump_rate + large_pump_rate + third_pump_rate

theorem pumps_fill_time :
  1 / total_pump_rate = 6 / 29 :=
by
  -- Definition of the rates has already been given.
  -- Here we specify the calculation for the combined rate and filling time.
  sorry

end pumps_fill_time_l34_34148


namespace sale_price_tea_correct_l34_34594

noncomputable def sale_price_of_mixed_tea (weight1 weight2 price1 price2 profit_percentage : ℝ) : ℝ :=
let total_cost := weight1 * price1 + weight2 * price2
let total_weight := weight1 + weight2
let cost_price_per_kg := total_cost / total_weight
let profit_per_kg := profit_percentage * cost_price_per_kg
let sale_price_per_kg := cost_price_per_kg + profit_per_kg
sale_price_per_kg

theorem sale_price_tea_correct :
  sale_price_of_mixed_tea 80 20 15 20 0.20 = 19.2 :=
  by
  sorry

end sale_price_tea_correct_l34_34594


namespace fraction_to_decimal_l34_34073

theorem fraction_to_decimal : (22 / 8 : ℝ) = 2.75 := 
sorry

end fraction_to_decimal_l34_34073


namespace benedict_house_size_l34_34549

variable (K B : ℕ)

theorem benedict_house_size
    (h1 : K = 4 * B + 600)
    (h2 : K = 10000) : B = 2350 := by
sorry

end benedict_house_size_l34_34549


namespace point_on_circle_l34_34860

noncomputable def x_value_on_circle : ℝ :=
  let a := (-3 : ℝ)
  let b := (21 : ℝ)
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  let y := 12
  Cx

theorem point_on_circle (x y : ℝ) (a b : ℝ) (ha : a = -3) (hb : b = 21) (hy : y = 12) :
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  (x - Cx) ^ 2 + y ^ 2 = radius ^ 2 → x = x_value_on_circle :=
by
  intros
  sorry

end point_on_circle_l34_34860


namespace remainder_of_130_div_k_l34_34350

theorem remainder_of_130_div_k (k a : ℕ) (hk : 90 = a * k^2 + 18) : 130 % k = 4 :=
sorry

end remainder_of_130_div_k_l34_34350


namespace fraction_addition_l34_34482

theorem fraction_addition (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_addition_l34_34482


namespace tan_alpha_implies_fraction_l34_34756

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
sorry

end tan_alpha_implies_fraction_l34_34756


namespace power_identity_l34_34762

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l34_34762


namespace not_divisible_by_1998_l34_34239

theorem not_divisible_by_1998 (n : ℕ) :
  ∀ k : ℕ, ¬ (2^(k+1) * n + 2^k - 1) % 2 = 0 → ¬ (2^(k+1) * n + 2^k - 1) % 1998 = 0 :=
by
  intros _ _
  sorry

end not_divisible_by_1998_l34_34239


namespace parabola_focus_l34_34114

theorem parabola_focus (h : ∀ x y : ℝ, y ^ 2 = -12 * x → True) : (-3, 0) = (-3, 0) :=
  sorry

end parabola_focus_l34_34114


namespace bobs_sisters_mile_time_l34_34194

theorem bobs_sisters_mile_time (bobs_current_time_minutes : ℕ) (bobs_current_time_seconds : ℕ) (improvement_percentage : ℝ) :
  bobs_current_time_minutes = 10 → bobs_current_time_seconds = 40 → improvement_percentage = 9.062499999999996 →
  bobs_sisters_time_minutes = 9 ∧ bobs_sisters_time_seconds = 42 :=
by
  -- Definitions from conditions
  let bobs_time_in_seconds := bobs_current_time_minutes * 60 + bobs_current_time_seconds
  let improvement_in_seconds := bobs_time_in_seconds * improvement_percentage / 100
  let target_time_in_seconds := bobs_time_in_seconds - improvement_in_seconds
  let bobs_sisters_time_minutes := target_time_in_seconds / 60
  let bobs_sisters_time_seconds := target_time_in_seconds % 60
  
  sorry

end bobs_sisters_mile_time_l34_34194


namespace negation_of_universal_prop_l34_34045

theorem negation_of_universal_prop : 
  (¬ (∀ (x : ℝ), x ^ 2 ≥ 0)) ↔ (∃ (x : ℝ), x ^ 2 < 0) :=
by sorry

end negation_of_universal_prop_l34_34045


namespace gcd_hcf_of_36_and_84_l34_34906

theorem gcd_hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := sorry

end gcd_hcf_of_36_and_84_l34_34906


namespace johanna_loses_half_turtles_l34_34297

theorem johanna_loses_half_turtles
  (owen_turtles_initial : ℕ)
  (johanna_turtles_fewer : ℕ)
  (owen_turtles_after_month : ℕ)
  (owen_turtles_final : ℕ)
  (johanna_donates_rest_to_owen : ℚ → ℚ)
  (x : ℚ)
  (hx1 : owen_turtles_initial = 21)
  (hx2 : johanna_turtles_fewer = 5)
  (hx3 : owen_turtles_after_month = owen_turtles_initial * 2)
  (hx4 : owen_turtles_final = owen_turtles_after_month + johanna_donates_rest_to_owen (1 - x))
  (hx5 : owen_turtles_final = 50) :
  x = 1 / 2 :=
by
  sorry

end johanna_loses_half_turtles_l34_34297


namespace R_transformed_is_R_l34_34560

-- Define the initial coordinates of the rectangle PQRS
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (6, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (3, 1)

-- Define the reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the translation down by 2 units
def translate_down_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 - 2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the translation up by 2 units
def translate_up_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 2)

-- Define the transformation to find R''
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up_2 (reflect_y_neg_x (translate_down_2 (reflect_x p)))

-- Prove that the result of transforming R is (-3, -4)
theorem R_transformed_is_R'' : transform R = (-3, -4) :=
  by sorry

end R_transformed_is_R_l34_34560


namespace find_line_AB_l34_34987

noncomputable def equation_of_line_AB : Prop :=
  ∀ (x y : ℝ), ((x-2)^2 + (y-1)^2 = 10) ∧ ((x+6)^2 + (y+3)^2 = 50) → (2*x + y = 0)

theorem find_line_AB : equation_of_line_AB := by
  sorry

end find_line_AB_l34_34987


namespace no_valid_sum_seventeen_l34_34465

def std_die (n : ℕ) : Prop := n ∈ [1, 2, 3, 4, 5, 6]

def valid_dice (a b c d : ℕ) : Prop := std_die a ∧ std_die b ∧ std_die c ∧ std_die d

def sum_dice (a b c d : ℕ) : ℕ := a + b + c + d

def prod_dice (a b c d : ℕ) : ℕ := a * b * c * d

theorem no_valid_sum_seventeen (a b c d : ℕ) (h_valid : valid_dice a b c d) (h_prod : prod_dice a b c d = 360) : sum_dice a b c d ≠ 17 :=
sorry

end no_valid_sum_seventeen_l34_34465


namespace books_on_shelf_l34_34932

theorem books_on_shelf (total_books : ℕ) (sold_books : ℕ) (shelves : ℕ) (remaining_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 27 → sold_books = 6 → shelves = 3 → remaining_books = total_books - sold_books → books_per_shelf = remaining_books / shelves → books_per_shelf = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_on_shelf_l34_34932


namespace inequality_correctness_l34_34072

theorem inequality_correctness (a b c : ℝ) (h : c^2 > 0) : (a * c^2 > b * c^2) ↔ (a > b) := by 
sorry

end inequality_correctness_l34_34072


namespace farmer_plant_beds_l34_34058

theorem farmer_plant_beds :
  ∀ (bean_seedlings pumpkin_seeds radishes seedlings_per_row_pumpkin seedlings_per_row_radish radish_rows_per_bed : ℕ),
    bean_seedlings = 64 →
    seedlings_per_row_pumpkin = 7 →
    pumpkin_seeds = 84 →
    seedlings_per_row_radish = 6 →
    radish_rows_per_bed = 2 →
    (bean_seedlings / 8 + pumpkin_seeds / seedlings_per_row_pumpkin + radishes / seedlings_per_row_radish) / radish_rows_per_bed = 14 :=
by
  -- sorry to skip the proof
  sorry

end farmer_plant_beds_l34_34058


namespace hyperbola_eccentricity_l34_34993

theorem hyperbola_eccentricity (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : b ^ 2 = (5 / 4) * a ^ 2) 
  (h_c : c ^ 2 = a ^ 2 + b ^ 2) : 
  (3 / 2) = c / a :=
by sorry

end hyperbola_eccentricity_l34_34993


namespace relationship_between_first_and_third_numbers_l34_34226

variable (A B C : ℕ)

theorem relationship_between_first_and_third_numbers
  (h1 : A + B + C = 660)
  (h2 : A = 2 * B)
  (h3 : B = 180) :
  C = A - 240 :=
by
  sorry

end relationship_between_first_and_third_numbers_l34_34226


namespace basketball_team_girls_l34_34203

theorem basketball_team_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3) * G = 18) : 
  G = 18 :=
by
  have h3 : G - (1 / 3) * G = 30 - 18 := by sorry
  have h4 : (2 / 3) * G = 12 := by sorry
  have h5 : G = 12 * (3 / 2) := by sorry
  have h6 : G = 18 := by sorry
  exact h6

end basketball_team_girls_l34_34203


namespace part1_part2_part3_l34_34617

-- Given conditions and definitions
def A : ℝ := 1
def B : ℝ := 3
def y1 : ℝ := sorry  -- simply a placeholder value as y1 == y2
def y2 : ℝ := y1
def y (x m n : ℝ) : ℝ := x^2 + m * x + n

-- (1) Proof of m = -4
theorem part1 (n : ℝ) (h1 : y A m n = y1) (h2 : y B m n = y2) : m = -4 := sorry

-- (2) Proof of n = 4 when the parabola intersects the x-axis at one point
theorem part2 (h : ∃ n, ∀ x : ℝ, y x (-4) n = 0 → x = (x - 2)^2) : n = 4 := sorry

-- (3) Proof of the range of real number values for a
theorem part3 (a : ℝ) (b1 b2 : ℝ) (n : ℝ) (h1 : y a (-4) n = b1) 
  (h2 : y B (-4) n = b2) (h3 : b1 > b2) : a < 1 ∨ a > 3 := sorry

end part1_part2_part3_l34_34617


namespace find_candies_l34_34712

variable (e : ℝ)

-- Given conditions
def candies_sum (e : ℝ) : ℝ := e + 4 * e + 16 * e + 96 * e

theorem find_candies (h : candies_sum e = 876) : e = 7.5 :=
by
  -- proof omitted
  sorry

end find_candies_l34_34712


namespace charlyn_viewable_area_l34_34031

noncomputable def charlyn_sees_area (side_length viewing_distance : ℝ) : ℝ :=
  let inner_viewable_area := (side_length^2 - (side_length - 2 * viewing_distance)^2)
  let rectangular_area := 4 * (side_length * viewing_distance)
  let circular_corner_area := 4 * ((viewing_distance^2 * Real.pi) / 4)
  inner_viewable_area + rectangular_area + circular_corner_area

theorem charlyn_viewable_area :
  let side_length := 7
  let viewing_distance := 1.5
  charlyn_sees_area side_length viewing_distance = 82 := 
by
  sorry

end charlyn_viewable_area_l34_34031


namespace find_cupcakes_l34_34732

def total_students : ℕ := 20
def treats_per_student : ℕ := 4
def cookies : ℕ := 20
def brownies : ℕ := 35
def total_treats : ℕ := total_students * treats_per_student
def cupcakes : ℕ := total_treats - (cookies + brownies)

theorem find_cupcakes : cupcakes = 25 := by
  sorry

end find_cupcakes_l34_34732


namespace abc_is_772_l34_34725

noncomputable def find_abc (a b c : ℝ) : ℝ :=
if h₁ : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * (b + c) = 160 ∧ b * (c + a) = 168 ∧ c * (a + b) = 180
then 772 else 0

theorem abc_is_772 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
(h₄ : a * (b + c) = 160) (h₅ : b * (c + a) = 168) (h₆ : c * (a + b) = 180) :
  find_abc a b c = 772 := by
  sorry

end abc_is_772_l34_34725


namespace sum_of_decimals_l34_34367

theorem sum_of_decimals : (5.47 + 4.96) = 10.43 :=
by
  sorry

end sum_of_decimals_l34_34367


namespace avg_values_l34_34032

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l34_34032


namespace wall_length_is_800_l34_34053

def brick_volume : ℝ := 50 * 11.25 * 6
def total_brick_volume : ℝ := 3200 * brick_volume
def wall_volume (x : ℝ) : ℝ := x * 600 * 22.5

theorem wall_length_is_800 :
  ∀ (x : ℝ), total_brick_volume = wall_volume x → x = 800 :=
by
  intros x h
  sorry

end wall_length_is_800_l34_34053


namespace willie_currency_exchange_l34_34326

theorem willie_currency_exchange :
  let euro_amount := 70
  let pound_amount := 50
  let franc_amount := 30

  let euro_to_dollar := 1.2
  let pound_to_dollar := 1.5
  let franc_to_dollar := 1.1

  let airport_euro_rate := 5 / 7
  let airport_pound_rate := 3 / 4
  let airport_franc_rate := 9 / 10

  let flat_fee := 5

  let official_euro_dollars := euro_amount * euro_to_dollar
  let official_pound_dollars := pound_amount * pound_to_dollar
  let official_franc_dollars := franc_amount * franc_to_dollar

  let airport_euro_dollars := official_euro_dollars * airport_euro_rate
  let airport_pound_dollars := official_pound_dollars * airport_pound_rate
  let airport_franc_dollars := official_franc_dollars * airport_franc_rate

  let final_euro_dollars := airport_euro_dollars - flat_fee
  let final_pound_dollars := airport_pound_dollars - flat_fee
  let final_franc_dollars := airport_franc_dollars - flat_fee

  let total_dollars := final_euro_dollars + final_pound_dollars + final_franc_dollars

  total_dollars = 130.95 :=
by
  sorry

end willie_currency_exchange_l34_34326


namespace robot_distance_covered_l34_34880

theorem robot_distance_covered :
  let start1 := -3
  let end1 := -8
  let end2 := 6
  let distance1 := abs (end1 - start1)
  let distance2 := abs (end2 - end1)
  distance1 + distance2 = 19 := by
  sorry

end robot_distance_covered_l34_34880


namespace not_divisible_59_l34_34841

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l34_34841


namespace perimeter_rectangle_l34_34164

-- Defining the width and length of the rectangle based on the conditions
def width (a : ℝ) := a
def length (a : ℝ) := 2 * a + 1

-- Statement of the problem: proving the perimeter
theorem perimeter_rectangle (a : ℝ) :
  let W := width a
  let L := length a
  2 * W + 2 * L = 6 * a + 2 :=
by
  sorry

end perimeter_rectangle_l34_34164


namespace find_a_l34_34969

-- Definitions for the problem
def quadratic_distinct_roots (a : ℝ) : Prop :=
  let Δ := a^2 - 16
  Δ > 0

def satisfies_root_equation (x1 x2 : ℝ) : Prop :=
  (x1^2 - (20 / (3 * x2^3)) = x2^2 - (20 / (3 * x1^3)))

-- Main statement of the proof problem
theorem find_a (a x1 x2 : ℝ) (h_quadratic_roots : quadratic_distinct_roots a)
               (h_root_equation : satisfies_root_equation x1 x2)
               (h_vieta_sum : x1 + x2 = -a) (h_vieta_product : x1 * x2 = 4) :
  a = -10 :=
by
  sorry

end find_a_l34_34969


namespace school_bus_solution_l34_34349

-- Define the capacities
def bus_capacity : Prop := 
  ∃ x y : ℕ, x + y = 75 ∧ 3 * x + 2 * y = 180 ∧ x = 30 ∧ y = 45

-- Define the rental problem
def rental_plans : Prop :=
  ∃ a : ℕ, 6 ≤ a ∧ a ≤ 8 ∧ 
  (30 * a + 45 * (25 - a) ≥ 1000) ∧ 
  (320 * a + 400 * (25 - a) ≤ 9550) ∧ 
  3 = 3

-- The main theorem combines the two aspects
theorem school_bus_solution: bus_capacity ∧ rental_plans := 
  sorry -- Proof omitted

end school_bus_solution_l34_34349


namespace find_x_l34_34974

theorem find_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 :=
sorry

end find_x_l34_34974


namespace invalid_votes_l34_34603

theorem invalid_votes (W L total_polls : ℕ) 
  (h1 : total_polls = 90830) 
  (h2 : L = 9 * W / 11) 
  (h3 : W = L + 9000)
  (h4 : 100 * (W + L) = 90000) : 
  total_polls - (W + L) = 830 := 
sorry

end invalid_votes_l34_34603


namespace chef_initial_potatoes_l34_34422

theorem chef_initial_potatoes (fries_per_potato : ℕ) (total_fries_needed : ℕ) (leftover_potatoes : ℕ) 
  (H1 : fries_per_potato = 25) 
  (H2 : total_fries_needed = 200) 
  (H3 : leftover_potatoes = 7) : 
  (total_fries_needed / fries_per_potato + leftover_potatoes = 15) :=
by
  sorry

end chef_initial_potatoes_l34_34422


namespace investment_rate_l34_34759

theorem investment_rate (total : ℝ) (invested_at_3_percent : ℝ) (rate_3_percent : ℝ) 
                        (invested_at_5_percent : ℝ) (rate_5_percent : ℝ) 
                        (desired_income : ℝ) (remaining : ℝ) (additional_income : ℝ) (r : ℝ) : 
  total = 12000 ∧ 
  invested_at_3_percent = 5000 ∧ 
  rate_3_percent = 0.03 ∧ 
  invested_at_5_percent = 4000 ∧ 
  rate_5_percent = 0.05 ∧ 
  desired_income = 600 ∧ 
  remaining = total - invested_at_3_percent - invested_at_5_percent ∧ 
  additional_income = desired_income - (invested_at_3_percent * rate_3_percent + invested_at_5_percent * rate_5_percent) ∧ 
  r = (additional_income / remaining) * 100 → 
  r = 8.33 := 
by
  sorry

end investment_rate_l34_34759


namespace functional_equation_solution_l34_34608

theorem functional_equation_solution (f : ℚ → ℚ)
  (H : ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry

end functional_equation_solution_l34_34608


namespace a_profit_share_l34_34057

/-- Definitions for the shares of capital -/
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)
def total_profit : ℚ := 2415

/-- The profit share for A, given the conditions on capital subscriptions -/
theorem a_profit_share : a_share * total_profit = 805 := by
  sorry

end a_profit_share_l34_34057


namespace lockers_remaining_open_l34_34284

-- Define the number of lockers and students
def num_lockers : ℕ := 1000

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to count perfect squares up to a given number
def count_perfect_squares_up_to (n : ℕ) : ℕ :=
  Nat.sqrt n

-- Theorem statement
theorem lockers_remaining_open : 
  count_perfect_squares_up_to num_lockers = 31 :=
by
  -- Proof left out because it's not necessary to provide
  sorry

end lockers_remaining_open_l34_34284


namespace min_ab_l34_34844

theorem min_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 2 := by
  sorry

end min_ab_l34_34844


namespace emilia_strCartons_l34_34249

theorem emilia_strCartons (total_cartons_needed cartons_bought cartons_blueberries : ℕ) (h1 : total_cartons_needed = 42) (h2 : cartons_blueberries = 7) (h3 : cartons_bought = 33) :
  (total_cartons_needed - (cartons_bought + cartons_blueberries)) = 2 :=
by
  sorry

end emilia_strCartons_l34_34249


namespace volume_frustum_correct_l34_34613

noncomputable def volume_of_frustum 
  (base_edge_orig : ℝ) 
  (altitude_orig : ℝ) 
  (base_edge_small : ℝ) 
  (altitude_small : ℝ) : ℝ :=
  let volume_ratio := (base_edge_small / base_edge_orig) ^ 3
  let base_area_orig := (Real.sqrt 3 / 4) * base_edge_orig ^ 2
  let volume_orig := (1 / 3) * base_area_orig * altitude_orig
  let volume_small := volume_ratio * volume_orig
  let volume_frustum := volume_orig - volume_small
  volume_frustum

theorem volume_frustum_correct :
  volume_of_frustum 18 9 9 3 = 212.625 * Real.sqrt 3 :=
sorry

end volume_frustum_correct_l34_34613


namespace cone_volume_l34_34943

theorem cone_volume (V_cylinder : ℝ) (V_cone : ℝ) (h : V_cylinder = 81 * Real.pi) :
  V_cone = 27 * Real.pi :=
by
  sorry

end cone_volume_l34_34943


namespace evaluate_series_l34_34628

noncomputable def infinite_series :=
  ∑' n, (n^3 + 2*n^2 - 3) / (n+3).factorial

theorem evaluate_series : infinite_series = 1 / 4 :=
by
  sorry

end evaluate_series_l34_34628


namespace scientific_notation_of_935000000_l34_34435

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l34_34435


namespace ratio_alcohol_to_water_l34_34464

-- Definitions of volume fractions for alcohol and water
def alcohol_volume_fraction : ℚ := 1 / 7
def water_volume_fraction : ℚ := 2 / 7

-- The theorem stating the ratio of alcohol to water volumes
theorem ratio_alcohol_to_water : (alcohol_volume_fraction / water_volume_fraction) = 1 / 2 :=
by sorry

end ratio_alcohol_to_water_l34_34464


namespace eccentricity_of_ellipse_l34_34949

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : 1 / m + 2 / n = 1) (h2 : 0 < m) (h3 : 0 < n) (h4 : m * n = 8) :
  let a := n
  let b := m
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l34_34949


namespace solve_for_x_l34_34500

theorem solve_for_x :
  ∃ x : ℝ, (24 / 36) = Real.sqrt (x / 36) ∧ x = 16 :=
by
  use 16
  sorry

end solve_for_x_l34_34500


namespace minimum_width_for_fence_l34_34082

theorem minimum_width_for_fence (w : ℝ) (h : 0 ≤ 20) : 
  (w * (w + 20) ≥ 150) → w ≥ 10 :=
by
  sorry

end minimum_width_for_fence_l34_34082


namespace parametric_to_cartesian_l34_34545

variable (R t : ℝ)

theorem parametric_to_cartesian (x y : ℝ) (h1 : x = R * Real.cos t) (h2 : y = R * Real.sin t) : 
  x^2 + y^2 = R^2 := 
by
  sorry

end parametric_to_cartesian_l34_34545


namespace chocolate_bar_cost_l34_34761

-- Definitions based on the conditions given in the problem.
def total_bars : ℕ := 7
def remaining_bars : ℕ := 4
def total_money : ℚ := 9
def bars_sold : ℕ := total_bars - remaining_bars
def cost_per_bar := total_money / bars_sold

-- The theorem that needs to be proven.
theorem chocolate_bar_cost : cost_per_bar = 3 := by
  -- proof placeholder
  sorry

end chocolate_bar_cost_l34_34761


namespace quadratic_solution_l34_34561

theorem quadratic_solution : 
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ (x = 3 ∨ x = -1) :=
by {
  sorry
}

end quadratic_solution_l34_34561


namespace initial_girls_count_l34_34827

variable (p : ℕ) -- total number of people initially in the group
variable (girls_initial : ℕ) -- number of girls initially in the group
variable (girls_after : ℕ) -- number of girls after the change
variable (total_after : ℕ) -- total number of people after the change

/--
Initially, 50% of the group are girls. 
Later, five girls leave and five boys arrive, leading to 40% of the group now being girls.
--/
theorem initial_girls_count :
  (girls_initial = p / 2) →
  (total_after = p) →
  (girls_after = girls_initial - 5) →
  (girls_after = 2 * total_after / 5) →
  girls_initial = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_girls_count_l34_34827


namespace part1_part2_l34_34013

theorem part1 (m : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 ∧ y = x + m) → -3 - 3 * Real.sqrt 2 < m ∧ m < -3 + 3 * Real.sqrt 2) :=
sorry

theorem part2 (m x1 x2 y1 y2 : ℝ) (h1 : x1 + x2 = -(m + 1)) (h2 : x1 * x2 = (m^2 + 4 * m - 4) / 2) 
(h3 : (x - x1) * (x - x2) + (x1 + m) * (x2 + m) = 0) : 
  m = -4 ∨ m = 1 →
  (∀ x y : ℝ, y = x + m ↔ x - y - 4 = 0 ∨ x - y + 1 = 0) :=
sorry

end part1_part2_l34_34013


namespace nissa_grooming_time_correct_l34_34741

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l34_34741


namespace car_dealer_bmw_sales_l34_34121

theorem car_dealer_bmw_sales (total_cars : ℕ)
  (vw_percentage : ℝ)
  (toyota_percentage : ℝ)
  (acura_percentage : ℝ)
  (bmw_count : ℕ) :
  total_cars = 300 →
  vw_percentage = 0.10 →
  toyota_percentage = 0.25 →
  acura_percentage = 0.20 →
  bmw_count = total_cars * (1 - (vw_percentage + toyota_percentage + acura_percentage)) →
  bmw_count = 135 :=
by
  intros
  sorry

end car_dealer_bmw_sales_l34_34121


namespace cost_of_first_house_l34_34544

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l34_34544


namespace part1_part2_l34_34962

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def p (m : ℝ) : Prop :=
  let Δ := discriminant 1 m 1
  Δ > 0 ∧ -m / 2 < 0

def q (m : ℝ) : Prop :=
  let Δ := discriminant 4 (4 * (m - 2)) 1
  Δ < 0

theorem part1 (m : ℝ) (hp : p m) : m > 2 := 
sorry

theorem part2 (m : ℝ) (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m ≥ 3) ∨ (1 < m ∧ m ≤ 2) := 
sorry

end part1_part2_l34_34962


namespace probability_at_least_one_defective_is_correct_l34_34940

noncomputable def probability_at_least_one_defective : ℚ :=
  let total_bulbs := 23
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs
  let probability_neither_defective :=
    (non_defective_bulbs / total_bulbs) * ((non_defective_bulbs - 1) / (total_bulbs - 1))
  1 - probability_neither_defective

theorem probability_at_least_one_defective_is_correct :
  probability_at_least_one_defective = 164 / 506 :=
by
  sorry

end probability_at_least_one_defective_is_correct_l34_34940


namespace abs_eq_case_l34_34891

theorem abs_eq_case (x : ℝ) (h : |x - 3| = |x + 2|) : x = 1/2 :=
by
  sorry

end abs_eq_case_l34_34891


namespace margie_change_l34_34458

theorem margie_change (n_sold n_cost n_paid : ℕ) (h1 : n_sold = 3) (h2 : n_cost = 50) (h3 : n_paid = 500) : 
  n_paid - (n_sold * n_cost) = 350 := by
  sorry

end margie_change_l34_34458


namespace grasshopper_jump_distance_l34_34753

variable (F G M : ℕ) -- F for frog's jump, G for grasshopper's jump, M for mouse's jump

theorem grasshopper_jump_distance (h1 : F = G + 39) (h2 : M = F - 94) (h3 : F = 58) : G = 19 := 
by
  sorry

end grasshopper_jump_distance_l34_34753


namespace twelve_pow_six_mod_nine_eq_zero_l34_34606

theorem twelve_pow_six_mod_nine_eq_zero : (∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 12^6 ≡ n [MOD 9]) → 12^6 ≡ 0 [MOD 9] :=
by
  sorry

end twelve_pow_six_mod_nine_eq_zero_l34_34606


namespace johns_contribution_correct_l34_34001

noncomputable def average_contribution_before : Real := sorry
noncomputable def total_contributions_by_15 : Real := 15 * average_contribution_before
noncomputable def new_average_contribution : Real := 150
noncomputable def johns_contribution : Real := average_contribution_before * 15 + 1377.3

-- The theorem we want to prove
theorem johns_contribution_correct :
  (new_average_contribution = (total_contributions_by_15 + johns_contribution) / 16) ∧
  (new_average_contribution = 2.2 * average_contribution_before) :=
sorry

end johns_contribution_correct_l34_34001


namespace brianne_yard_length_l34_34540

theorem brianne_yard_length 
  (derrick_yard_length : ℝ)
  (h₁ : derrick_yard_length = 10)
  (alex_yard_length : ℝ)
  (h₂ : alex_yard_length = derrick_yard_length / 2)
  (brianne_yard_length : ℝ)
  (h₃ : brianne_yard_length = 6 * alex_yard_length) :
  brianne_yard_length = 30 :=
by sorry

end brianne_yard_length_l34_34540


namespace total_points_always_odd_l34_34666

theorem total_points_always_odd (n : ℕ) (h : n ≥ 1) :
  ∀ k : ℕ, ∃ m : ℕ, m = (2 ^ k * (n + 1) - 1) ∧ m % 2 = 1 :=
by
  sorry

end total_points_always_odd_l34_34666


namespace pow_mod_remainder_l34_34624

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l34_34624


namespace shortest_distance_between_circles_l34_34217

def circle_eq1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 15 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 + 12*y + 21 = 0

theorem shortest_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), circle_eq1 x1 y1 → circle_eq2 x2 y2 → 
  (abs ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) - (15^(1/2) + 82^(1/2))) =
  2 * 41^(1/2) - 97^(1/2) :=
by sorry

end shortest_distance_between_circles_l34_34217


namespace count_two_digit_numbers_with_digit_8_l34_34267

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l34_34267


namespace lucy_snowballs_eq_19_l34_34865

-- Define the conditions
def charlie_snowballs : ℕ := 50
def difference_charlie_lucy : ℕ := 31

-- Define what we want to prove, i.e., Lucy has 19 snowballs
theorem lucy_snowballs_eq_19 : (charlie_snowballs - difference_charlie_lucy = 19) :=
by
  -- We would provide the proof here, but it's not required for this prompt
  sorry

end lucy_snowballs_eq_19_l34_34865


namespace cos_angle_plus_pi_over_two_l34_34576

theorem cos_angle_plus_pi_over_two (α : ℝ) (h1 : Real.cos α = 1 / 5) (h2 : α ∈ Set.Icc (-2 * Real.pi) (-3 * Real.pi / 2) ∪ Set.Icc (0) (Real.pi / 2)) :
  Real.cos (α + Real.pi / 2) = 2 * Real.sqrt 6 / 5 :=
sorry

end cos_angle_plus_pi_over_two_l34_34576


namespace nina_walking_distance_l34_34951

def distance_walked_by_john : ℝ := 0.7
def distance_john_further_than_nina : ℝ := 0.3

def distance_walked_by_nina : ℝ := distance_walked_by_john - distance_john_further_than_nina

theorem nina_walking_distance :
  distance_walked_by_nina = 0.4 :=
by
  sorry

end nina_walking_distance_l34_34951


namespace train_cross_platform_time_l34_34656

def train_length : ℝ := 300
def platform_length : ℝ := 550
def signal_pole_time : ℝ := 18

theorem train_cross_platform_time :
  let speed : ℝ := train_length / signal_pole_time
  let total_distance : ℝ := train_length + platform_length
  let crossing_time : ℝ := total_distance / speed
  crossing_time = 51 :=
by
  sorry

end train_cross_platform_time_l34_34656


namespace problem1_problem2_l34_34035

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l34_34035


namespace erin_walks_less_l34_34324

variable (total_distance : ℕ)
variable (susan_distance : ℕ)

theorem erin_walks_less (h1 : total_distance = 15) (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 := by
  sorry

end erin_walks_less_l34_34324


namespace arithmetic_sequence_a2_value_l34_34654

theorem arithmetic_sequence_a2_value 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + 3)
  (h2 : S n = n * (a 1 + a n) / 2)
  (hS13 : S 13 = 156) :
  a 2 = -3 := 
    sorry

end arithmetic_sequence_a2_value_l34_34654


namespace sqrt_product_simplification_l34_34862

theorem sqrt_product_simplification (p : ℝ) : 
  (Real.sqrt (42 * p)) * (Real.sqrt (14 * p)) * (Real.sqrt (7 * p)) = 14 * p * (Real.sqrt (21 * p)) := 
  sorry

end sqrt_product_simplification_l34_34862


namespace mean_problem_l34_34358

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l34_34358


namespace min_operations_to_reach_goal_l34_34903

-- Define the initial and final configuration of the letters
structure Configuration where
  A : Char := 'A'
  B : Char := 'B'
  C : Char := 'C'
  D : Char := 'D'
  E : Char := 'E'
  F : Char := 'F'
  G : Char := 'G'

-- Define a valid rotation operation
inductive Rotation
| rotate_ABC : Rotation
| rotate_ABD : Rotation
| rotate_DEF : Rotation
| rotate_EFC : Rotation

-- Function representing a single rotation
def applyRotation : Configuration -> Rotation -> Configuration
| config, Rotation.rotate_ABC => 
  { A := config.C, B := config.A, C := config.B, D := config.D, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_ABD => 
  { A := config.B, B := config.D, D := config.A, C := config.C, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_DEF => 
  { D := config.E, E := config.F, F := config.D, A := config.A, B := config.B, C := config.C, G := config.G }
| config, Rotation.rotate_EFC => 
  { E := config.F, F := config.C, C := config.E, A := config.A, B := config.B, D := config.D, G := config.G }

-- Define the goal configuration
def goalConfiguration : Configuration := 
  { A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G' }

-- Function to apply multiple rotations
def applyRotations (config : Configuration) (rotations : List Rotation) : Configuration :=
  rotations.foldl applyRotation config

-- Main theorem statement 
theorem min_operations_to_reach_goal : 
  ∃ rotations : List Rotation, rotations.length = 3 ∧ applyRotations {A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G'} rotations = goalConfiguration :=
sorry

end min_operations_to_reach_goal_l34_34903


namespace number_of_blueberries_l34_34563

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end number_of_blueberries_l34_34563


namespace hypotenuse_length_l34_34149

theorem hypotenuse_length (a b c : ℝ) (h₁ : a + b + c = 40) (h₂ : 0.5 * a * b = 24) (h₃ : a^2 + b^2 = c^2) : c = 18.8 := sorry

end hypotenuse_length_l34_34149


namespace infinite_consecutive_pairs_l34_34821

-- Define the relation
def related (x y : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 (x + y)), d = 0 ∨ d = 1

-- Define sets A and B
variable (A B : Set ℕ)

-- Define the conditions
axiom cond1 : ∀ a ∈ A, ∀ b ∈ B, related a b
axiom cond2 : ∀ c, (∀ a ∈ A, related c a) → c ∈ B
axiom cond3 : ∀ c, (∀ b ∈ B, related c b) → c ∈ A

-- Prove that one of the sets contains infinitely many pairs of consecutive numbers
theorem infinite_consecutive_pairs :
  (∃ a ∈ A, ∀ n : ℕ, a + n ∈ A ∧ a + n + 1 ∈ A) ∨ (∃ b ∈ B, ∀ n : ℕ, b + n ∈ B ∧ b + n + 1 ∈ B) :=
sorry

end infinite_consecutive_pairs_l34_34821


namespace find_AD_length_l34_34936

noncomputable def triangle_AD (A B C : Type) (AB AC : ℝ) (ratio_BD_CD : ℝ) (AD : ℝ) : Prop :=
  AB = 13 ∧ AC = 20 ∧ ratio_BD_CD = 3 / 4 → AD = 8 * Real.sqrt 2

theorem find_AD_length {A B C : Type} :
  triangle_AD A B C 13 20 (3/4) (8 * Real.sqrt 2) :=
by
  sorry

end find_AD_length_l34_34936


namespace repeated_root_value_l34_34626

theorem repeated_root_value (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (2 / (x - 1) + 3 = m / (x - 1)) ∧ 
            ∀ y : ℝ, y ≠ 1 ∧ (2 / (y - 1) + 3 = m / (y - 1)) → y = x) →
  m = 2 :=
by
  sorry

end repeated_root_value_l34_34626


namespace integers_exist_for_eqns_l34_34346

theorem integers_exist_for_eqns (a b c : ℤ) :
  ∃ (p1 q1 r1 p2 q2 r2 : ℤ), 
    a = q1 * r2 - q2 * r1 ∧ 
    b = r1 * p2 - r2 * p1 ∧ 
    c = p1 * q2 - p2 * q1 :=
  sorry

end integers_exist_for_eqns_l34_34346


namespace total_spent_snacks_l34_34039

-- Define the costs and discounts
def cost_pizza : ℕ := 10
def boxes_robert_orders : ℕ := 5
def pizza_discount : ℝ := 0.15
def cost_soft_drink : ℝ := 1.50
def soft_drinks_robert : ℕ := 10
def cost_hamburger : ℕ := 3
def hamburgers_teddy_orders : ℕ := 6
def hamburger_discount : ℝ := 0.10
def soft_drinks_teddy : ℕ := 10

-- Calculate total costs
def total_cost_robert : ℝ := 
  let cost_pizza_total := (boxes_robert_orders * cost_pizza) * (1 - pizza_discount)
  let cost_soft_drinks_total := soft_drinks_robert * cost_soft_drink
  cost_pizza_total + cost_soft_drinks_total

def total_cost_teddy : ℝ :=
  let cost_hamburger_total := (hamburgers_teddy_orders * cost_hamburger) * (1 - hamburger_discount)
  let cost_soft_drinks_total := soft_drinks_teddy * cost_soft_drink
  cost_hamburger_total + cost_soft_drinks_total

-- The final theorem to prove the total spending
theorem total_spent_snacks : 
  total_cost_robert + total_cost_teddy = 88.70 := by
  sorry

end total_spent_snacks_l34_34039


namespace find_x_l34_34506

theorem find_x (x : ℤ) (h : (2 * x + 7) / 5 = 22) : x = 103 / 2 :=
by
  sorry

end find_x_l34_34506


namespace solve_for_sum_l34_34338

theorem solve_for_sum (x y : ℝ) (h : x^2 + y^2 = 18 * x - 10 * y + 22) : x + y = 4 + 2 * Real.sqrt 42 :=
sorry

end solve_for_sum_l34_34338


namespace set_C_is_pythagorean_triple_l34_34639

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem set_C_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
sorry

end set_C_is_pythagorean_triple_l34_34639


namespace Mary_younger_than_Albert_l34_34002

-- Define the basic entities and conditions
def Betty_age : ℕ := 11
def Albert_age : ℕ := 4 * Betty_age
def Mary_age : ℕ := Albert_age / 2

-- Define the property to prove
theorem Mary_younger_than_Albert : Albert_age - Mary_age = 22 :=
by 
  sorry

end Mary_younger_than_Albert_l34_34002


namespace decreasing_interval_l34_34791

def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative function
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0 :=
by
  intro x h
  have h1: x^2 < 1 := by
    sorry
  have h2: 3*x^2 < 3 := by
    sorry
  have h3: 3*x^2 - 3 < 0 := by
    sorry
  exact h3

end decreasing_interval_l34_34791


namespace c_value_difference_l34_34323

theorem c_value_difference (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  max c - min c = 34 / 3 :=
sorry

end c_value_difference_l34_34323


namespace six_sin6_cos6_l34_34209

theorem six_sin6_cos6 (A : ℝ) (h : Real.cos (2 * A) = - Real.sqrt 5 / 3) : 
  6 * Real.sin (A) ^ 6 + 6 * Real.cos (A) ^ 6 = 4 := 
sorry

end six_sin6_cos6_l34_34209


namespace tan_alpha_eq_neg_one_l34_34213

theorem tan_alpha_eq_neg_one (alpha : ℝ) (h1 : Real.tan alpha = -1) (h2 : 0 ≤ alpha ∧ alpha < Real.pi) :
  alpha = (3 * Real.pi) / 4 :=
sorry

end tan_alpha_eq_neg_one_l34_34213


namespace problem_conditions_l34_34456

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end problem_conditions_l34_34456


namespace intersection_M_complement_N_l34_34535

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def N : Set ℝ := {x | ∃ y : ℝ, y = 3*x^2 + 1 }

def complement_N : Set ℝ := {x | ¬ ∃ y : ℝ, y = 3*x^2 + 1}

theorem intersection_M_complement_N :
  (M ∩ complement_N) = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_l34_34535


namespace smallest_positive_integer_l34_34152

theorem smallest_positive_integer (n : ℕ) : 
  (∃ m : ℕ, (4410 * n = m^2)) → n = 10 := 
by
  sorry

end smallest_positive_integer_l34_34152


namespace range_of_dot_product_l34_34676

theorem range_of_dot_product
  (a b : ℝ)
  (h: ∃ (A B : ℝ × ℝ), (A ≠ B) ∧ ∃ m n : ℝ, A = (m, n) ∧ B = (-m, -n) ∧ m^2 + (n^2 / 9) = 1)
  : ∃ r : Set ℝ, r = (Set.Icc 41 49) :=
  sorry

end range_of_dot_product_l34_34676


namespace max_value_of_E_l34_34689

variable (a b c d : ℝ)

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  -5.5 ≤ a ∧ a ≤ 5.5 →
  -5.5 ≤ b ∧ b ≤ 5.5 →
  -5.5 ≤ c ∧ c ≤ 5.5 →
  -5.5 ≤ d ∧ d ≤ 5.5 →
  E a b c d ≤ 132 := by
  sorry

end max_value_of_E_l34_34689


namespace ratio_of_sum_of_terms_l34_34051

variable {α : Type*}
variable [Field α]

def geometric_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) := S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem ratio_of_sum_of_terms (a : ℕ → α) (S : ℕ → α)
  (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 :=
by
  sorry

end ratio_of_sum_of_terms_l34_34051


namespace area_on_map_correct_l34_34867

namespace FieldMap

-- Given conditions
def actual_length_m : ℕ := 200
def actual_width_m : ℕ := 100
def scale_factor : ℕ := 2000

-- Conversion from meters to centimeters
def length_cm := actual_length_m * 100
def width_cm := actual_width_m * 100

-- Dimensions on the map
def length_map_cm := length_cm / scale_factor
def width_map_cm := width_cm / scale_factor

-- Area on the map
def area_map_cm2 := length_map_cm * width_map_cm

-- Statement to prove
theorem area_on_map_correct : area_map_cm2 = 50 := by
  sorry

end FieldMap

end area_on_map_correct_l34_34867


namespace division_of_negatives_l34_34005

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end division_of_negatives_l34_34005


namespace find_x_squared_plus_y_squared_l34_34019

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l34_34019


namespace remainder_of_series_div_9_l34_34873

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l34_34873


namespace rectangle_square_area_ratio_eq_one_l34_34923

theorem rectangle_square_area_ratio_eq_one (r l w s: ℝ) (h1: l = 2 * w) (h2: r ^ 2 = (l / 2) ^ 2 + w ^ 2) (h3: s ^ 2 = 2 * r ^ 2) : 
  (l * w) / (s ^ 2) = 1 :=
by
sorry

end rectangle_square_area_ratio_eq_one_l34_34923


namespace total_amount_invested_l34_34531

theorem total_amount_invested (x y : ℝ) (hx : 0.06 * x = 0.05 * y + 160) (hy : 0.05 * y = 6000) :
  x + y = 222666.67 :=
by
  sorry

end total_amount_invested_l34_34531


namespace no_number_exists_decreasing_by_removing_digit_l34_34671

theorem no_number_exists_decreasing_by_removing_digit :
  ¬ ∃ (x y n : ℕ), x * 10^n + y = 58 * y :=
by
  sorry

end no_number_exists_decreasing_by_removing_digit_l34_34671


namespace dealer_profit_percentage_l34_34159

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℝ) (sp_total : ℝ) (sp_count : ℝ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  let profit_percentage := (profit_per_article / cp_per_article) * 100
  profit_percentage

theorem dealer_profit_percentage :
  profit_percentage 25 15 38 12 = 89.99 := by
  sorry

end dealer_profit_percentage_l34_34159


namespace geometric_sequence_ratio_l34_34359

theorem geometric_sequence_ratio (a1 q : ℝ) (h : (a1 * (1 - q^3) / (1 - q)) / (a1 * (1 - q^2) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 := by
  sorry

end geometric_sequence_ratio_l34_34359


namespace solve_for_x_l34_34360

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by 
  sorry

end solve_for_x_l34_34360


namespace max_red_balls_l34_34644

theorem max_red_balls (R B G : ℕ) (h1 : G = 12) (h2 : R + B + G = 28) (h3 : R + G < 24) : R ≤ 11 := 
by
  sorry

end max_red_balls_l34_34644


namespace advance_agency_fees_eq_8280_l34_34581

-- Conditions
variables (Commission GivenFees Incentive AdvanceAgencyFees : ℝ)
-- Given values
variables (h_comm : Commission = 25000) 
          (h_given : GivenFees = 18500) 
          (h_incent : Incentive = 1780)

-- The problem statement to prove
theorem advance_agency_fees_eq_8280 
    (h_comm : Commission = 25000) 
    (h_given : GivenFees = 18500) 
    (h_incent : Incentive = 1780)
    : AdvanceAgencyFees = 26780 - GivenFees :=
by
  sorry

end advance_agency_fees_eq_8280_l34_34581


namespace floor_sum_correct_l34_34451

def floor_sum_1_to_24 := 
  let sum := (3 * 1) + (5 * 2) + (7 * 3) + (9 * 4)
  sum

theorem floor_sum_correct : floor_sum_1_to_24 = 70 := by
  sorry

end floor_sum_correct_l34_34451


namespace probability_stopping_after_three_draws_l34_34180

def draws : List (List ℕ) := [
  [2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
  [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]
]

def favorable_sequences (seqs : List (List ℕ)) : List (List ℕ) :=
  seqs.filter (λ seq => 0 ∈ seq ∧ 1 ∈ seq)

def probability_of_drawing_zhong_hua (seqs : List (List ℕ)) : ℚ :=
  (favorable_sequences seqs).length / seqs.length

theorem probability_stopping_after_three_draws :
  probability_of_drawing_zhong_hua draws = 5 / 18 := by
sorry

end probability_stopping_after_three_draws_l34_34180


namespace frac_multiplication_l34_34577

theorem frac_multiplication : 
    ((2/3:ℚ)^4 * (1/5) * (3/4) = 4/135) :=
by
  sorry

end frac_multiplication_l34_34577


namespace initial_men_checking_exam_papers_l34_34772

theorem initial_men_checking_exam_papers :
  ∀ (M : ℕ),
  (M * 8 * 5 = (1/2 : ℝ) * (2 * 20 * 8)) → M = 4 :=
by
  sorry

end initial_men_checking_exam_papers_l34_34772


namespace samantha_score_l34_34869

variables (correct_answers geometry_correct_answers incorrect_answers unanswered_questions : ℕ)
          (points_per_correct : ℝ := 1) (additional_geometry_points : ℝ := 0.5)

def total_score (correct_answers geometry_correct_answers : ℕ) : ℝ :=
  correct_answers * points_per_correct + geometry_correct_answers * additional_geometry_points

theorem samantha_score 
  (Samantha_correct : correct_answers = 15)
  (Samantha_geometry : geometry_correct_answers = 4)
  (Samantha_incorrect : incorrect_answers = 5)
  (Samantha_unanswered : unanswered_questions = 5) :
  total_score correct_answers geometry_correct_answers = 17 := 
by
  sorry

end samantha_score_l34_34869


namespace sequence_a_n_l34_34179

-- Given conditions from the problem
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequence is given by S_n
axiom sum_Sn : ∀ n : ℕ, n > 0 → S n = 2 * n * n

-- Definition of a_n, the nth term of the sequence
def a_n (n : ℕ) : ℕ :=
  if n = 1 then
    S 1
  else
    S n - S (n - 1)

-- Prove that a_n = 4n - 2 for all n > 0.
theorem sequence_a_n (n : ℕ) (h : n > 0) : a_n S n = 4 * n - 2 :=
by
  sorry

end sequence_a_n_l34_34179


namespace factorization_proof_l34_34491

def factorization_problem (x : ℝ) : Prop := (x^2 - 1)^2 - 6 * (x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2

theorem factorization_proof (x : ℝ) : factorization_problem x :=
by
  -- The proof is omitted.
  sorry

end factorization_proof_l34_34491


namespace find_f_2021_l34_34902

variable (f : ℝ → ℝ)

axiom functional_equation : ∀ a b : ℝ, f ( (a + 2 * b) / 3) = (f a + 2 * f b) / 3
axiom f_one : f 1 = 1
axiom f_four : f 4 = 7

theorem find_f_2021 : f 2021 = 4041 := by
  sorry

end find_f_2021_l34_34902


namespace exists_x_in_interval_iff_m_lt_3_l34_34339

theorem exists_x_in_interval_iff_m_lt_3 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2 * x > m) ↔ m < 3 :=
by
  sorry

end exists_x_in_interval_iff_m_lt_3_l34_34339


namespace rectangle_area_l34_34344

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l34_34344


namespace other_root_correct_l34_34161

noncomputable def other_root (p : ℝ) : ℝ :=
  let a := 3
  let c := -2
  let root1 := -1
  (-c / a) / root1

theorem other_root_correct (p : ℝ) (h_eq : 3 * (-1) ^ 2 + p * (-1) = 2) : other_root p = 2 / 3 :=
  by
    unfold other_root
    sorry

end other_root_correct_l34_34161


namespace relationship_m_n_l34_34424

variables {a b : ℝ}

theorem relationship_m_n (h1 : |a| ≠ |b|) (m : ℝ) (n : ℝ)
  (hm : m = (|a| - |b|) / |a - b|)
  (hn : n = (|a| + |b|) / |a + b|) :
  m ≤ n :=
by sorry

end relationship_m_n_l34_34424


namespace faces_of_prism_with_24_edges_l34_34136

theorem faces_of_prism_with_24_edges (L : ℕ) (h1 : 3 * L = 24) : L + 2 = 10 := by
  sorry

end faces_of_prism_with_24_edges_l34_34136


namespace car_catches_truck_in_7_hours_l34_34206

-- Definitions based on the conditions
def initial_distance := 175 -- initial distance in kilometers
def truck_speed := 40 -- speed of the truck in km/h
def car_initial_speed := 50 -- initial speed of the car in km/h
def car_speed_increase := 5 -- speed increase per hour for the car in km/h

-- The main statement to prove
theorem car_catches_truck_in_7_hours :
  ∃ n : ℕ, (n ≥ 0) ∧ 
  (car_initial_speed - truck_speed) * n + (car_speed_increase * n * (n - 1) / 2) = initial_distance :=
by
  existsi 7
  -- Check the equation for n = 7
  -- Simplify: car initial extra speed + sum of increase terms should equal initial distance
  -- (50 - 40) * 7 + 5 * 7 * 6 / 2 = 175
  -- (10) * 7 + 35 * 3 / 2 = 175
  -- 70 + 105 = 175
  sorry

end car_catches_truck_in_7_hours_l34_34206


namespace geometric_sum_n_eq_3_l34_34622

theorem geometric_sum_n_eq_3 :
  (∃ n : ℕ, (1 / 2) * (1 - (1 / 3) ^ n) = 728 / 2187) ↔ n = 3 :=
by
  sorry

end geometric_sum_n_eq_3_l34_34622


namespace seq_a8_value_l34_34604

theorem seq_a8_value 
  (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) 
  (a7_eq : a 7 = 120) 
  : a 8 = 194 :=
sorry

end seq_a8_value_l34_34604


namespace winner_percentage_l34_34137

variable (votes_winner : ℕ) (win_by : ℕ)
variable (total_votes : ℕ)
variable (percentage_winner : ℕ)

-- Conditions
def conditions : Prop :=
  votes_winner = 930 ∧
  win_by = 360 ∧
  total_votes = votes_winner + (votes_winner - win_by) ∧
  percentage_winner = (votes_winner * 100) / total_votes

-- Theorem to prove
theorem winner_percentage (h : conditions votes_winner win_by total_votes percentage_winner) : percentage_winner = 62 :=
sorry

end winner_percentage_l34_34137


namespace sum_of_squares_inequality_l34_34801

theorem sum_of_squares_inequality (a b c : ℝ) (h : a + 2 * b + 3 * c = 4) : a^2 + b^2 + c^2 ≥ 8 / 7 := by
  sorry

end sum_of_squares_inequality_l34_34801


namespace circle_problem_l34_34752

theorem circle_problem 
  (x y : ℝ)
  (h : x^2 + 8*x - 10*y = 10 - y^2 + 6*x) :
  let a := -1
  let b := 5
  let r := 6
  a + b + r = 10 :=
by sorry

end circle_problem_l34_34752


namespace rate_2nd_and_3rd_hours_equals_10_l34_34708

-- Define the conditions as given in the problem
def total_gallons_after_5_hours := 34 
def rate_1st_hour := 8 
def rate_4th_hour := 14 
def water_lost_5th_hour := 8 

-- Problem statement: Prove the rate during 2nd and 3rd hours is 10 gallons/hour
theorem rate_2nd_and_3rd_hours_equals_10 (R : ℕ) :
  total_gallons_after_5_hours = rate_1st_hour + 2 * R + rate_4th_hour - water_lost_5th_hour →
  R = 10 :=
by sorry

end rate_2nd_and_3rd_hours_equals_10_l34_34708


namespace find_y_l34_34335

theorem find_y (y : ℝ) : 
  2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ y ∈ Set.Ioc (10 / 7) (8 / 5) := 
sorry

end find_y_l34_34335


namespace sum_of_N_values_eq_neg_one_l34_34468

theorem sum_of_N_values_eq_neg_one (R : ℝ) :
  ∀ (N : ℝ), N ≠ 0 ∧ (N + N^2 - 5 / N = R) →
  (∃ N₁ N₂ N₃ : ℝ, N₁ + N₂ + N₃ = -1 ∧ N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ N₃ ≠ 0) :=
by
  sorry

end sum_of_N_values_eq_neg_one_l34_34468


namespace minimize_cost_l34_34420

-- Define the unit prices of the soccer balls.
def price_A := 50
def price_B := 80

-- Define the condition for the total number of balls and cost function.
def total_balls := 80
def cost (a : ℕ) : ℕ := price_A * a + price_B * (total_balls - a)
def valid_a (a : ℕ) : Prop := 30 ≤ a ∧ a ≤ (3 * (total_balls - a))

-- Prove the number of brand A soccer balls to minimize the total cost.
theorem minimize_cost : ∃ a : ℕ, valid_a a ∧ ∀ b : ℕ, valid_a b → cost a ≤ cost b :=
sorry

end minimize_cost_l34_34420


namespace determine_k_value_l34_34182

theorem determine_k_value : (5 ^ 1002 + 6 ^ 1001) ^ 2 - (5 ^ 1002 - 6 ^ 1001) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end determine_k_value_l34_34182


namespace problem_solution_l34_34153

-- Definitions based on conditions
def valid_sequence (b : Fin 7 → Nat) : Prop :=
  (∀ i j : Fin 7, i ≤ j → b i ≥ b j) ∧ 
  (∀ i : Fin 7, b i ≤ 1500) ∧ 
  (∀ i : Fin 7, (b i + i) % 3 = 0)

-- The main theorem
theorem problem_solution :
  (∃ b : Fin 7 → Nat, valid_sequence b) →
  @Nat.choose 506 7 % 1000 = 506 :=
sorry

end problem_solution_l34_34153


namespace max_cards_mod3_l34_34989

theorem max_cards_mod3 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) : 
  ∃ t ⊆ s, t.card = 6 ∧ (t.prod id) % 3 = 1 := sorry

end max_cards_mod3_l34_34989


namespace sticks_difference_l34_34636

def sticks_picked_up : ℕ := 14
def sticks_left : ℕ := 4

theorem sticks_difference : (sticks_picked_up - sticks_left) = 10 := by
  sorry

end sticks_difference_l34_34636


namespace inequality_one_inequality_two_l34_34877

theorem inequality_one (x : ℝ) : 7 * x - 2 < 3 * (x + 2) → x < 2 :=
by
  sorry

theorem inequality_two (x : ℝ) : (x - 1) / 3 ≥ (x - 3) / 12 + 1 → x ≥ 13 / 3 :=
by
  sorry

end inequality_one_inequality_two_l34_34877


namespace spherical_to_rectangular_coordinates_l34_34397

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ),
    ρ = 15 →
    θ = 5 * Real.pi / 6 →
    φ = Real.pi / 3 →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    z = ρ * Real.cos φ →
    x = -45 / 4 ∧ y = -15 * Real.sqrt 3 / 4 ∧ z = 7.5 := 
by
  intro ρ θ φ x y z
  intro hρ hθ hφ hx hy hz
  rw [hρ, hθ, hφ] at *
  rw [hx, hy, hz]
  sorry

end spherical_to_rectangular_coordinates_l34_34397


namespace distance_covered_at_40_kmph_l34_34362

theorem distance_covered_at_40_kmph
   (total_distance : ℝ)
   (speed1 : ℝ)
   (speed2 : ℝ)
   (total_time : ℝ)
   (part_distance1 : ℝ) :
   total_distance = 250 ∧
   speed1 = 40 ∧
   speed2 = 60 ∧
   total_time = 6 ∧
   (part_distance1 / speed1 + (total_distance - part_distance1) / speed2 = total_time) →
   part_distance1 = 220 :=
by sorry

end distance_covered_at_40_kmph_l34_34362


namespace problem_l34_34591

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem problem (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∀ a b : ℝ, (0 ≤ a ∧ a ≤ 2 * π) → (0 ≤ b ∧ b ≤ 2 * π) → a ≤ b → f a ≤ f b := 
sorry

end problem_l34_34591


namespace find_a_l34_34851

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end find_a_l34_34851


namespace range_of_m_l34_34276

theorem range_of_m (x m : ℝ) (h1 : 2 * x - m ≤ 3) (h2 : -5 < x) (h3 : x < 4) :
  ∃ m, ∀ (x : ℝ), (-5 < x ∧ x < 4) → (2 * x - m ≤ 3) ↔ (m ≥ 5) :=
by sorry

end range_of_m_l34_34276


namespace probability_sequence_correct_l34_34251

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l34_34251


namespace ground_beef_per_package_l34_34683

-- Declare the given conditions and the expected result.
theorem ground_beef_per_package (num_people : ℕ) (weight_per_burger : ℕ) (total_packages : ℕ) 
    (h1 : num_people = 10) 
    (h2 : weight_per_burger = 2) 
    (h3 : total_packages = 4) : 
    (num_people * weight_per_burger) / total_packages = 5 := 
by 
  sorry

end ground_beef_per_package_l34_34683


namespace Carolina_mailed_five_letters_l34_34681

-- Definitions translating the given conditions into Lean
def cost_of_mail (cost_letters cost_packages : ℝ) (num_letters num_packages : ℕ) : ℝ :=
  cost_letters * num_letters + cost_packages * num_packages

-- The main theorem to prove the desired answer
theorem Carolina_mailed_five_letters (P L : ℕ)
  (h1 : L = P + 2)
  (h2 : cost_of_mail 0.37 0.88 L P = 4.49) :
  L = 5 := 
sorry

end Carolina_mailed_five_letters_l34_34681


namespace largest_cube_surface_area_l34_34917

theorem largest_cube_surface_area (width length height: ℕ) (h_w: width = 12) (h_l: length = 16) (h_h: height = 14) :
  (6 * (min width (min length height))^2) = 864 := by
  sorry

end largest_cube_surface_area_l34_34917


namespace greatest_value_of_x_is_20_l34_34411

noncomputable def greatest_multiple_of_4 (x : ℕ) : Prop :=
  (x % 4 = 0 ∧ x^2 < 500 ∧ ∀ y : ℕ, (y % 4 = 0 ∧ y^2 < 500) → y ≤ x)

theorem greatest_value_of_x_is_20 : greatest_multiple_of_4 20 :=
  by 
  sorry

end greatest_value_of_x_is_20_l34_34411


namespace remainder_when_expression_divided_l34_34256

theorem remainder_when_expression_divided 
  (x y u v : ℕ) 
  (h1 : x = u * y + v) 
  (h2 : 0 ≤ v) 
  (h3 : v < y) :
  (x - u * y + 3 * v) % y = (4 * v) % y :=
by
  sorry

end remainder_when_expression_divided_l34_34256


namespace find_eighth_number_l34_34498

def average_of_numbers (a b c d e f g h x : ℕ) : ℕ :=
  (a + b + c + d + e + f + g + h + x) / 9

theorem find_eighth_number (a b c d e f g h x : ℕ) (avg : ℕ) 
    (h_avg : average_of_numbers a b c d e f g h x = avg)
    (h_total_sum : a + b + c + d + e + f + g + h + x = 540)
    (h_x_val : x = 65) : a = 53 :=
by
  sorry

end find_eighth_number_l34_34498


namespace scale_model_height_l34_34580

theorem scale_model_height (real_height : ℕ) (scale_ratio : ℕ) (h_real : real_height = 1454) (h_scale : scale_ratio = 50) : 
⌊(real_height : ℝ) / scale_ratio + 0.5⌋ = 29 :=
by
  rw [h_real, h_scale]
  norm_num
  sorry

end scale_model_height_l34_34580


namespace percentage_of_sikh_boys_l34_34709

theorem percentage_of_sikh_boys (total_boys muslim_percentage hindu_percentage other_boys : ℕ) 
  (h₁ : total_boys = 300) 
  (h₂ : muslim_percentage = 44) 
  (h₃ : hindu_percentage = 28) 
  (h₄ : other_boys = 54) : 
  (10 : ℝ) = 
  (((total_boys - (muslim_percentage * total_boys / 100 + hindu_percentage * total_boys / 100 + other_boys)) * 100) / total_boys : ℝ) :=
by
  sorry

end percentage_of_sikh_boys_l34_34709


namespace isosceles_triangle_sides_l34_34119

theorem isosceles_triangle_sides (length_rope : ℝ) (one_side : ℝ) (a b : ℝ) :
  length_rope = 18 ∧ one_side = 5 ∧ a + a + one_side = length_rope ∧ b = one_side ∨ b + b + one_side = length_rope -> (a = 6.5 ∨ a = 5) ∧ (b = 6.5 ∨ b = 5) :=
by
  sorry

end isosceles_triangle_sides_l34_34119


namespace magic_island_red_parrots_l34_34372

noncomputable def total_parrots : ℕ := 120

noncomputable def green_parrots : ℕ := (5 * total_parrots) / 8

noncomputable def non_green_parrots : ℕ := total_parrots - green_parrots

noncomputable def red_parrots : ℕ := non_green_parrots / 3

theorem magic_island_red_parrots : red_parrots = 15 :=
by
  sorry

end magic_island_red_parrots_l34_34372


namespace area_of_trapezoid_l34_34589

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end area_of_trapezoid_l34_34589


namespace range_of_a_l34_34026

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x^2 + 2 * |x - a| ≥ a^2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l34_34026


namespace max_value_of_expression_l34_34186

theorem max_value_of_expression (x : Real) :
  (x^4 / (x^8 + 2 * x^6 - 3 * x^4 + 5 * x^3 + 8 * x^2 + 5 * x + 25)) ≤ (1 / 15) :=
sorry

end max_value_of_expression_l34_34186


namespace product_evaluation_l34_34798

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l34_34798


namespace positions_after_347_moves_l34_34830

-- Define the possible positions for the cat
inductive CatPosition
| top_vertex
| right_upper_vertex
| right_lower_vertex
| left_lower_vertex
| left_upper_vertex

-- Define the possible positions for the mouse
inductive MousePosition
| top_left_edge
| left_upper_vertex
| left_middle_edge
| left_lower_vertex
| bottom_edge
| right_lower_vertex
| right_middle_edge
| right_upper_vertex
| top_right_edge
| top_vertex

-- Define the movement function for the cat
def cat_position_after_moves (moves : Nat) : CatPosition :=
  match moves % 5 with
  | 0 => CatPosition.top_vertex
  | 1 => CatPosition.right_upper_vertex
  | 2 => CatPosition.right_lower_vertex
  | 3 => CatPosition.left_lower_vertex
  | 4 => CatPosition.left_upper_vertex
  | _ => CatPosition.top_vertex  -- This case is unreachable due to % 5

-- Define the movement function for the mouse
def mouse_position_after_moves (moves : Nat) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.top_left_edge
  | 1 => MousePosition.left_upper_vertex
  | 2 => MousePosition.left_middle_edge
  | 3 => MousePosition.left_lower_vertex
  | 4 => MousePosition.bottom_edge
  | 5 => MousePosition.right_lower_vertex
  | 6 => MousePosition.right_middle_edge
  | 7 => MousePosition.right_upper_vertex
  | 8 => MousePosition.top_right_edge
  | 9 => MousePosition.top_vertex
  | _ => MousePosition.top_left_edge  -- This case is unreachable due to % 10

-- Prove the positions after 347 moves
theorem positions_after_347_moves :
  cat_position_after_moves 347 = CatPosition.right_upper_vertex ∧
  mouse_position_after_moves 347 = MousePosition.right_middle_edge :=
by
  sorry

end positions_after_347_moves_l34_34830


namespace fido_area_reach_l34_34555

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l34_34555


namespace isosceles_trapezoid_fewest_axes_l34_34000

def equilateral_triangle_axes : Nat := 3
def isosceles_trapezoid_axes : Nat := 1
def rectangle_axes : Nat := 2
def regular_pentagon_axes : Nat := 5

theorem isosceles_trapezoid_fewest_axes :
  isosceles_trapezoid_axes < equilateral_triangle_axes ∧
  isosceles_trapezoid_axes < rectangle_axes ∧
  isosceles_trapezoid_axes < regular_pentagon_axes :=
by
  sorry

end isosceles_trapezoid_fewest_axes_l34_34000


namespace valid_three_digit_numbers_l34_34199

   noncomputable def three_digit_num_correct (A : ℕ) : Prop :=
     (100 ≤ A ∧ A < 1000) ∧ (1000000 + A = A * A)

   theorem valid_three_digit_numbers (A : ℕ) :
     three_digit_num_correct A → (A = 625 ∨ A = 376) :=
   by
     sorry
   
end valid_three_digit_numbers_l34_34199


namespace votes_cast_l34_34963

-- Define the conditions as given in the problem.
def total_votes (V : ℕ) := 35 * V / 100 + (35 * V / 100 + 2400) = V

-- The goal is to prove that the number of total votes V equals 8000.
theorem votes_cast : ∃ V : ℕ, total_votes V ∧ V = 8000 :=
by
  sorry -- The proof is not required, only the statement.

end votes_cast_l34_34963


namespace assignment_statement_correct_l34_34086

def meaning_of_assignment_statement (N : ℕ) := N + 1

theorem assignment_statement_correct :
  meaning_of_assignment_statement N = N + 1 :=
sorry

end assignment_statement_correct_l34_34086


namespace ratio_of_areas_l34_34539

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l34_34539


namespace value_of_6z_l34_34173

theorem value_of_6z (x y z : ℕ) (h1 : 6 * z = 2 * x) (h2 : x + y + z = 26) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 6 * z = 36 :=
by
  sorry

end value_of_6z_l34_34173


namespace algebraic_expression_value_l34_34532

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x - 2 = 0) :
  x * (x + 2) + (x + 1)^2 = 5 :=
by
  sorry

end algebraic_expression_value_l34_34532


namespace beads_counter_representation_l34_34788

-- Given conditions
variable (a : ℕ) -- a is a natural number representing the beads in the tens place.
variable (h : a ≥ 0) -- Ensure a is non-negative since the number of beads cannot be negative.

-- The main statement to prove
theorem beads_counter_representation (a : ℕ) (h : a ≥ 0) : 10 * a + 4 = (10 * a) + 4 :=
by sorry

end beads_counter_representation_l34_34788


namespace min_value_reciprocal_l34_34444

theorem min_value_reciprocal (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  3 ≤ (1/a) + (1/b) + (1/c) :=
sorry

end min_value_reciprocal_l34_34444


namespace greatest_integer_third_side_of_triangle_l34_34806

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l34_34806


namespace parallel_lines_l34_34673

-- Definitions of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

-- Definition of parallel lines: slopes are equal and the lines are not identical
def slopes_equal (m : ℝ) : Prop := -(3 + m) / 4 = -2 / (5 + m)
def not_identical_lines (m : ℝ) : Prop := l1 m ≠ l2 m

-- Theorem stating the given conditions
theorem parallel_lines (m : ℝ) (x y : ℝ) : slopes_equal m → not_identical_lines m → m = -7 := by
  sorry

end parallel_lines_l34_34673


namespace arithmetic_sequence_sum_l34_34107

variable {α : Type} [LinearOrderedField α]

noncomputable def a_n (a1 d n : α) := a1 + (n - 1) * d

theorem arithmetic_sequence_sum (a1 d : α) (h1 : a_n a1 d 3 * a_n a1 d 11 = 5)
  (h2 : a_n a1 d 3 + a_n a1 d 11 = 3) : a_n a1 d 5 + a_n a1 d 6 + a_n a1 d 10 = 9 / 2 :=
by
  sorry

end arithmetic_sequence_sum_l34_34107


namespace subtract_mult_equal_l34_34084

theorem subtract_mult_equal :
  2000000000000 - 1111111111111 * 1 = 888888888889 :=
by
  sorry

end subtract_mult_equal_l34_34084


namespace even_odd_product_zero_l34_34462

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_product_zero (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : is_even f) (hg : is_odd g) : ∀ x, f (-x) * g (-x) + f x * g x = 0 :=
by
  intro x
  have h₁ := hf x
  have h₂ := hg x
  sorry

end even_odd_product_zero_l34_34462


namespace variance_cows_l34_34218

-- Define the number of cows and incidence rate.
def n : ℕ := 10
def p : ℝ := 0.02

-- The variance of the binomial distribution, given n and p.
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Statement to prove
theorem variance_cows : variance n p = 0.196 :=
by
  sorry

end variance_cows_l34_34218


namespace number_of_cookies_first_friend_took_l34_34263

-- Definitions of given conditions:
def initial_cookies : ℕ := 22
def eaten_by_Kristy : ℕ := 2
def given_to_brother : ℕ := 1
def taken_by_second_friend : ℕ := 5
def taken_by_third_friend : ℕ := 5
def cookies_left : ℕ := 6

noncomputable def cookies_after_Kristy_ate_and_gave_away : ℕ :=
  initial_cookies - eaten_by_Kristy - given_to_brother

noncomputable def cookies_after_second_and_third_friends : ℕ :=
  taken_by_second_friend + taken_by_third_friend

noncomputable def cookies_before_second_and_third_friends_took : ℕ :=
  cookies_left + cookies_after_second_and_third_friends

theorem number_of_cookies_first_friend_took :
  cookies_after_Kristy_ate_and_gave_away - cookies_before_second_and_third_friends_took = 3 := by
  sorry

end number_of_cookies_first_friend_took_l34_34263


namespace sin_210_eq_neg_half_l34_34649

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 :=
by 
  sorry

end sin_210_eq_neg_half_l34_34649


namespace polygon_number_of_sides_l34_34384

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end polygon_number_of_sides_l34_34384


namespace no_real_roots_of_geom_seq_l34_34014

theorem no_real_roots_of_geom_seq (a b c : ℝ) (h_geom_seq : b^2 = a * c) : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  -- You can assume the steps of proving here
  sorry

end no_real_roots_of_geom_seq_l34_34014


namespace outfit_choices_l34_34680

/-- Given 8 shirts, 8 pairs of pants, and 8 hats, each in 8 colors,
only 6 colors have a matching shirt, pair of pants, and hat.
Each item in the outfit must be of a different color.
Prove that the number of valid outfits is 368. -/
theorem outfit_choices (shirts pants hats colors : ℕ)
  (matching_colors : ℕ)
  (h_shirts : shirts = 8)
  (h_pants : pants = 8)
  (h_hats : hats = 8)
  (h_colors : colors = 8)
  (h_matching_colors : matching_colors = 6) :
  (shirts * pants * hats) - 3 * (matching_colors * colors) = 368 := 
by {
  sorry
}

end outfit_choices_l34_34680


namespace posters_count_l34_34438

-- Define the regular price per poster
def regular_price : ℕ := 4

-- Jeremy can buy 24 posters at regular price
def posters_at_regular_price : ℕ := 24

-- Total money Jeremy has is equal to the money needed to buy 24 posters
def total_money : ℕ := posters_at_regular_price * regular_price

-- The special deal: buy one get the second at half price
def cost_of_two_posters : ℕ := regular_price + regular_price / 2

-- Number of pairs Jeremy can buy with his total money
def number_of_pairs : ℕ := total_money / cost_of_two_posters

-- Total number of posters Jeremy can buy under the sale
def total_posters := number_of_pairs * 2

-- Prove that the total posters is 32
theorem posters_count : total_posters = 32 := by
  sorry

end posters_count_l34_34438


namespace min_distance_squared_l34_34299

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l34_34299


namespace tan_inequality_l34_34694

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := Real.tan x

theorem tan_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < π / 2) (h3 : 0 < x2) (h4 : x2 < π / 2) (h5 : x1 ≠ x2) :
  (1/2) * (f x1 + f x2) > f ((x1 + x2) / 2) :=
  sorry

end tan_inequality_l34_34694


namespace Sarah_books_in_8_hours_l34_34064

theorem Sarah_books_in_8_hours (pages_per_hour: ℕ) (pages_per_book: ℕ) (hours_available: ℕ) 
  (h_pages_per_hour: pages_per_hour = 120) (h_pages_per_book: pages_per_book = 360) (h_hours_available: hours_available = 8) :
  hours_available * pages_per_hour / pages_per_book = 2 := by
  sorry

end Sarah_books_in_8_hours_l34_34064


namespace triangle_area_less_than_sqrt3_div_3_l34_34640

-- Definitions for a triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (area : ℝ)

def valid_triangle (Δ : Triangle) : Prop :=
  0 < Δ.a ∧ 0 < Δ.b ∧ 0 < Δ.c ∧ Δ.ha < 1 ∧ Δ.hb < 1 ∧ Δ.hc < 1

theorem triangle_area_less_than_sqrt3_div_3 (Δ : Triangle) (h : valid_triangle Δ) : Δ.area < (Real.sqrt 3) / 3 :=
sorry

end triangle_area_less_than_sqrt3_div_3_l34_34640


namespace julia_change_l34_34922

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end julia_change_l34_34922


namespace biscuits_initial_l34_34144

theorem biscuits_initial (F M A L B : ℕ) 
  (father_gave : F = 13) 
  (mother_gave : M = 15) 
  (brother_ate : A = 20) 
  (left_with : L = 40) 
  (remaining : B + F + M - A = L) :
  B = 32 := 
by 
  subst father_gave
  subst mother_gave
  subst brother_ate
  subst left_with
  simp at remaining
  linarith

end biscuits_initial_l34_34144


namespace P_subset_Q_l34_34840

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_subset_Q : P ⊂ Q := by
  sorry

end P_subset_Q_l34_34840


namespace tan_sum_formula_l34_34056

open Real

theorem tan_sum_formula (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_cos_2α : cos (2 * α) = -3 / 5) :
  tan (π / 4 + 2 * α) = -1 / 7 :=
by
  -- Insert the proof here
  sorry

end tan_sum_formula_l34_34056


namespace guests_accommodation_l34_34742

open Nat

theorem guests_accommodation :
  let guests := 15
  let rooms := 4
  (4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4 = 4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4) :=
by
  sorry

end guests_accommodation_l34_34742


namespace sequence_value_at_5_l34_34564

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 3 ∧ ∀ n, 1 < n → a n = (-1) ^ n * 2 * a (n - 1)

theorem sequence_value_at_5 (a : ℕ → ℚ) (h : seq a) : a 5 = 16 / 3 :=
by 
  sorry

end sequence_value_at_5_l34_34564


namespace nat_exponent_sum_eq_l34_34779

theorem nat_exponent_sum_eq (n p q : ℕ) : n^p + n^q = n^2010 ↔ (n = 2 ∧ p = 2009 ∧ q = 2009) :=
by
  sorry

end nat_exponent_sum_eq_l34_34779


namespace helen_made_56_pies_l34_34707

theorem helen_made_56_pies (pinky_pies total_pies : ℕ) (h_pinky : pinky_pies = 147) (h_total : total_pies = 203) :
  (total_pies - pinky_pies) = 56 :=
by
  sorry

end helen_made_56_pies_l34_34707


namespace original_combined_price_l34_34785

theorem original_combined_price (C S : ℝ) 
  (candy_box_increased : C * 1.25 = 18.75)
  (soda_can_increased : S * 1.50 = 9) : 
  C + S = 21 :=
by
  sorry

end original_combined_price_l34_34785


namespace simplify_and_evaluate_l34_34418

def a : Int := 1
def b : Int := -2

theorem simplify_and_evaluate :
  ((a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b)) = -8 := by
  sorry

end simplify_and_evaluate_l34_34418


namespace max_a_plus_b_min_a_squared_plus_b_squared_l34_34682

theorem max_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  a + b ≤ 2 := 
sorry

theorem min_a_squared_plus_b_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  2 ≤ a^2 + b^2 := 
sorry

end max_a_plus_b_min_a_squared_plus_b_squared_l34_34682


namespace tea_blend_gain_percent_l34_34918

theorem tea_blend_gain_percent :
  let cost_18 := 18
  let cost_20 := 20
  let ratio_5_to_3 := (5, 3)
  let selling_price := 21
  let total_cost := (ratio_5_to_3.1 * cost_18) + (ratio_5_to_3.2 * cost_20)
  let total_weight := ratio_5_to_3.1 + ratio_5_to_3.2
  let cost_price_per_kg := total_cost / total_weight
  let gain_percent := ((selling_price - cost_price_per_kg) / cost_price_per_kg) * 100
  gain_percent = 12 :=
by
  sorry

end tea_blend_gain_percent_l34_34918


namespace sequence_general_formula_l34_34232

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) 
    (h₂ : ∀ n : ℕ, 1 < n → a n = (n / (n - 1)) * a (n - 1)) : 
    ∀ n : ℕ, 1 ≤ n → a n = 3 * n :=
by
  -- Proof description here
  sorry

end sequence_general_formula_l34_34232


namespace ratio_u_v_l34_34156

variables {u v : ℝ}
variables (u_lt_v : u < v)
variables (h_triangle : triangle 15 12 9)
variables (inscribed_circle : is_inscribed_circle 15 12 9 u v)

theorem ratio_u_v : u / v = 1 / 2 :=
sorry

end ratio_u_v_l34_34156


namespace find_m_l34_34211

theorem find_m {x1 x2 m : ℝ} 
  (h_eqn : ∀ x, x^2 - (m+3)*x + (m+2) = 0) 
  (h_cond : x1 / (x1 + 1) + x2 / (x2 + 1) = 13 / 10) : 
  m = 2 := 
sorry

end find_m_l34_34211


namespace kim_gets_change_of_5_l34_34266

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def payment : ℝ := 20
noncomputable def total_cost_before_tip := meal_cost + drink_cost
noncomputable def tip := tip_rate * total_cost_before_tip
noncomputable def total_cost_with_tip := total_cost_before_tip + tip
noncomputable def change := payment - total_cost_with_tip

theorem kim_gets_change_of_5 : change = 5 := by
  sorry

end kim_gets_change_of_5_l34_34266


namespace tank_capacity_l34_34558

theorem tank_capacity (C : ℝ) :
  (3/4 * C - 0.4 * (3/4 * C) + 0.3 * (3/4 * C - 0.4 * (3/4 * C))) = 4680 → C = 8000 :=
by
  sorry

end tank_capacity_l34_34558


namespace value_of_expression_l34_34548

theorem value_of_expression (x y : ℝ) (h1 : x = -2) (h2 : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 :=
by
  sorry

end value_of_expression_l34_34548


namespace find_valid_ns_l34_34244

theorem find_valid_ns (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, k^2 = (n^2 + 7 * n + 136) / (n-1)) : n = 5 ∨ n = 37 :=
sorry

end find_valid_ns_l34_34244


namespace delta_max_success_ratio_l34_34165

theorem delta_max_success_ratio (y w x z : ℤ) (h1 : 360 + 240 = 600)
  (h2 : 0 < x ∧ x < y ∧ z < w)
  (h3 : y + w = 600)
  (h4 : (x : ℚ) / y < (200 : ℚ) / 360)
  (h5 : (z : ℚ) / w < (160 : ℚ) / 240)
  (h6 : (360 : ℚ) / 600 = 3 / 5)
  (h7 : (x + z) < 166) :
  (x + z : ℚ) / 600 ≤ 166 / 600 := 
sorry

end delta_max_success_ratio_l34_34165


namespace base9_add_subtract_l34_34878

theorem base9_add_subtract :
  let n1 := 3 * 9^2 + 5 * 9 + 1
  let n2 := 4 * 9^2 + 6 * 9 + 5
  let n3 := 1 * 9^2 + 3 * 9 + 2
  let n4 := 1 * 9^2 + 4 * 9 + 7
  (n1 + n2 + n3 - n4 = 8 * 9^2 + 4 * 9 + 7) :=
by
  sorry

end base9_add_subtract_l34_34878


namespace faith_change_l34_34667

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l34_34667


namespace correct_options_l34_34995

-- Definitions for lines l and n
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def line_n (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- The condition for lines to be parallel, equating the slopes
def parallel_lines (a : ℝ) : Prop := -(a + 2) / a = -(a - 2) / 3

-- The condition that line l passes through the point (1, -1)
def passes_through_point (a : ℝ) : Prop := line_l a 1 (-1)

-- The theorem statement
theorem correct_options (a : ℝ) :
  (parallel_lines a → a = 6 ∨ a = -1) ∧ (passes_through_point a) :=
by
  sorry

end correct_options_l34_34995


namespace center_of_circle_l34_34454

theorem center_of_circle : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → (1, 1) = (1, 1) :=
by
  intros x y h
  sorry

end center_of_circle_l34_34454


namespace ellipse_hyperbola_same_foci_l34_34520

theorem ellipse_hyperbola_same_foci (k : ℝ) (h1 : k > 0) :
  (∀ (x y : ℝ), (x^2 / 9 + y^2 / k^2 = 1) ↔ (x^2 / k - y^2 / 3 = 1)) → k = 2 :=
by
  sorry

end ellipse_hyperbola_same_foci_l34_34520


namespace sum_max_min_value_f_l34_34650

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + x) / (x ^ 2 + 1)

theorem sum_max_min_value_f : 
  let M := (⨆ x : ℝ, f x)
  let m := (⨅ x : ℝ, f x)
  M + m = 2 :=
by
-- Proof to be filled in
  sorry

end sum_max_min_value_f_l34_34650


namespace quadratic_decreases_after_vertex_l34_34924

theorem quadratic_decreases_after_vertex :
  ∀ x : ℝ, (x > 2) → (y = -(x - 2)^2 + 3) → ∃ k : ℝ, k < 0 :=
by
  sorry

end quadratic_decreases_after_vertex_l34_34924


namespace lowest_possible_students_l34_34541

theorem lowest_possible_students :
  ∃ n : ℕ, (n % 10 = 0 ∧ n % 24 = 0) ∧ n = 120 :=
by
  sorry

end lowest_possible_students_l34_34541


namespace no_polygon_with_half_parallel_diagonals_l34_34105

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end no_polygon_with_half_parallel_diagonals_l34_34105


namespace find_x_for_dot_product_l34_34195

theorem find_x_for_dot_product :
  let a : (ℝ × ℝ) := (1, -1)
  let b : (ℝ × ℝ) := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 1) ↔ x = 1 :=
by
  sorry

end find_x_for_dot_product_l34_34195


namespace find_B_value_l34_34221

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l34_34221


namespace coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l34_34965

section coexistent_rational_number_pairs

-- Definitions based on the problem conditions:
def coexistent_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Proof problem 1
theorem coexistent_pair_example : coexistent_pair 3 (1/2) :=
sorry

-- Proof problem 2
theorem coexistent_pair_neg (m n : ℚ) (h : coexistent_pair m n) :
  coexistent_pair (-n) (-m) :=
sorry

-- Proof problem 3
example : ∃ (p q : ℚ), coexistent_pair p q ∧ (p, q) ≠ (2, 1/3) ∧ (p, q) ≠ (5, 2/3) ∧ (p, q) ≠ (3, 1/2) :=
sorry

-- Proof problem 4
theorem coexistent_pair_find_a (a : ℚ) (h : coexistent_pair a 3) :
  a = -2 :=
sorry

end coexistent_rational_number_pairs

end coexistent_pair_example_coexistent_pair_neg_coexistent_pair_find_a_l34_34965


namespace arith_seq_sum_l34_34469

-- We start by defining what it means for a sequence to be arithmetic
def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- We are given that a_2 = 5 and a_6 = 33 for an arithmetic sequence
variable (a : ℕ → ℤ)
variable (h_arith : is_arith_seq a)
variable (h1 : a 2 = 5)
variable (h2 : a 6 = 33)

-- The statement we want to prove
theorem arith_seq_sum (a : ℕ → ℤ) (h_arith : is_arith_seq a) (h1 : a 2 = 5) (h2 : a 6 = 33) :
  (a 3 + a 5) = 38 :=
  sorry

end arith_seq_sum_l34_34469


namespace annual_avg_growth_rate_export_volume_2023_l34_34245

variable (V0 V2 V3 : ℕ) (r : ℝ)
variable (h1 : V0 = 200000) (h2 : V2 = 450000) (h3 : V3 = 675000)

-- Definition of the exponential growth equation
def growth_exponential (V0 Vn: ℕ) (n : ℕ) (r : ℝ) : Prop :=
  Vn = V0 * ((1 + r) ^ n)

-- The Lean statement to prove the annual average growth rate
theorem annual_avg_growth_rate (x : ℝ) (h : growth_exponential V0 V2 2 x) : 
  x = 0.5 :=
by
  sorry

-- The Lean statement to prove the export volume in 2023
theorem export_volume_2023 (h_growth : growth_exponential V2 V3 1 0.5) :
  V3 = 675000 :=
by
  sorry

end annual_avg_growth_rate_export_volume_2023_l34_34245


namespace CoastalAcademy_absent_percentage_l34_34175

theorem CoastalAcademy_absent_percentage :
  ∀ (total_students boys girls : ℕ) (absent_boys_ratio absent_girls_ratio : ℚ),
    total_students = 120 →
    boys = 70 →
    girls = 50 →
    absent_boys_ratio = 1/7 →
    absent_girls_ratio = 1/5 →
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    absent_percentage = 16.67 :=
  by
    intros total_students boys girls absent_boys_ratio absent_girls_ratio
           h1 h2 h3 h4 h5
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    sorry

end CoastalAcademy_absent_percentage_l34_34175


namespace circle_radius_order_l34_34453

theorem circle_radius_order 
  (rA: ℝ) (rA_condition: rA = 2)
  (CB: ℝ) (CB_condition: CB = 10 * Real.pi)
  (AC: ℝ) (AC_condition: AC = 16 * Real.pi) :
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  rA < rC ∧ rC < rB :=
by 
  sorry

end circle_radius_order_l34_34453


namespace cost_of_gas_per_gallon_l34_34081

-- Definitions based on the conditions
def hours_driven_1 : ℕ := 2
def speed_1 : ℕ := 60
def hours_driven_2 : ℕ := 3
def speed_2 : ℕ := 50
def mileage_per_gallon : ℕ := 30
def total_gas_cost : ℕ := 18

-- An assumption to simplify handling dollars and gallons
noncomputable def cost_per_gallon : ℕ := total_gas_cost / (speed_1 * hours_driven_1 + speed_2 * hours_driven_2) * mileage_per_gallon

theorem cost_of_gas_per_gallon :
  cost_per_gallon = 2 := by
sorry

end cost_of_gas_per_gallon_l34_34081


namespace total_students_l34_34621

-- Define the condition that the sum of boys (75) and girls (G) is the total number of students (T)
def sum_boys_girls (G T : ℕ) := 75 + G = T

-- Define the condition that the number of girls (G) equals 75% of the total number of students (T)
def girls_percentage (G T : ℕ) := G = Nat.div (3 * T) 4

-- State the theorem that given the above conditions, the total number of students (T) is 300
theorem total_students (G T : ℕ) (h1 : sum_boys_girls G T) (h2 : girls_percentage G T) : T = 300 := 
sorry

end total_students_l34_34621


namespace rose_joined_after_six_months_l34_34370

noncomputable def profit_shares (m : ℕ) : ℕ :=
  12000 * (12 - m) - 9000 * 8

theorem rose_joined_after_six_months :
  ∃ (m : ℕ), profit_shares m = 370 :=
by
  use 6
  unfold profit_shares
  norm_num
  sorry

end rose_joined_after_six_months_l34_34370


namespace lcm_is_multiple_of_230_l34_34925

theorem lcm_is_multiple_of_230 (d n : ℕ) (h1 : n = 230) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (2 ∣ d)) : ∃ m : ℕ, Nat.lcm d n = 230 * m :=
by
  exists 1 -- Placeholder for demonstration purposes
  sorry

end lcm_is_multiple_of_230_l34_34925


namespace correctness_of_statements_l34_34510

theorem correctness_of_statements (p q : Prop) (x y : ℝ) : 
  (¬ (p ∧ q) → (p ∨ q)) ∧
  ((xy = 0) → ¬(x^2 + y^2 = 0)) ∧
  ¬(∀ (L P : ℝ → ℝ), (∃ x, L x = P x) ↔ (∃ x, L x = P x ∧ ∀ x₁ x₂, x₁ ≠ x₂ → L x₁ ≠ P x₂)) →
  (0 + 1 + 0 = 1) :=
by
  sorry

end correctness_of_statements_l34_34510


namespace find_n_l34_34233

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end find_n_l34_34233


namespace cookies_left_l34_34021

theorem cookies_left (initial_cookies : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) :
  initial_cookies = 28 → cookies_eaten = 21 → cookies_left = initial_cookies - cookies_eaten → cookies_left = 7 :=
by
  intros h_initial h_eaten h_left
  rw [h_initial, h_eaten] at h_left
  exact h_left

end cookies_left_l34_34021


namespace equilateral_triangle_of_arith_geo_seq_l34_34881

def triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :=
  (α + β + γ = Real.pi) ∧
  (2 * β = α + γ) ∧
  (b^2 = a * c)

theorem equilateral_triangle_of_arith_geo_seq
  (A B C : ℝ) (a b c α β γ : ℝ)
  (h1 : triangle A B C a b c α β γ)
  : (a = c) ∧ (A = B) ∧ (B = C) ∧ (a = b) :=
  sorry

end equilateral_triangle_of_arith_geo_seq_l34_34881


namespace smallest_three_digit_multiple_of_17_l34_34901

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l34_34901


namespace price_after_discount_l34_34447

-- Define the original price and discount
def original_price : ℕ := 76
def discount : ℕ := 25

-- The main proof statement
theorem price_after_discount : original_price - discount = 51 := by
  sorry

end price_after_discount_l34_34447


namespace park_area_calculation_l34_34514

def scale := 300 -- miles per inch
def short_diagonal := 10 -- inches
def real_length := short_diagonal * scale -- miles
def park_area := (1/2) * real_length * real_length -- square miles

theorem park_area_calculation : park_area = 4500000 := by
  sorry

end park_area_calculation_l34_34514


namespace ellipse_area_l34_34461

-- Definitions based on the conditions
def cylinder_height : ℝ := 10
def cylinder_base_radius : ℝ := 1

-- Equivalent Proof Problem Statement
theorem ellipse_area
  (h : ℝ := cylinder_height)
  (r : ℝ := cylinder_base_radius)
  (ball_position_lower : ℝ := -4) -- derived from - (h / 2 - r)
  (ball_position_upper : ℝ := 4) -- derived from  (h / 2 - r)
  : (π * 4 * 2 = 16 * π) :=
by
  sorry

end ellipse_area_l34_34461


namespace minimum_frosting_time_l34_34478

def ann_time_per_cake := 8 -- Ann's time per cake in minutes
def bob_time_per_cake := 6 -- Bob's time per cake in minutes
def carol_time_per_cake := 10 -- Carol's time per cake in minutes
def passing_time := 1 -- time to pass a cake from one person to another in minutes
def total_cakes := 10 -- total number of cakes to be frosted

theorem minimum_frosting_time : 
  (ann_time_per_cake + passing_time + bob_time_per_cake + passing_time + carol_time_per_cake) + (total_cakes - 1) * carol_time_per_cake = 116 := 
by 
  sorry

end minimum_frosting_time_l34_34478


namespace margin_expression_l34_34584

variable (n : ℕ) (C S M : ℝ)

theorem margin_expression (H1 : M = (1 / n) * C) (H2 : C = S - M) : 
  M = (1 / (n + 1)) * S := 
by
  sorry

end margin_expression_l34_34584


namespace gas_pressure_in_final_container_l34_34200

variable (k : ℝ) (p_initial p_second p_final : ℝ) (v_initial v_second v_final v_half : ℝ)

theorem gas_pressure_in_final_container 
  (h1 : v_initial = 3.6)
  (h2 : p_initial = 6)
  (h3 : v_second = 7.2)
  (h4 : v_final = 3.6)
  (h5 : v_half = v_second / 2)
  (h6 : p_initial * v_initial = k)
  (h7 : p_second * v_second = k)
  (h8 : p_final * v_final = k) :
  p_final = 6 := 
sorry

end gas_pressure_in_final_container_l34_34200


namespace fraction_value_l34_34102

theorem fraction_value (x : ℝ) (h : 1 - 5 / x + 6 / x^3 = 0) : 3 / x = 3 / 2 :=
by
  sorry

end fraction_value_l34_34102


namespace square_area_proof_l34_34729

theorem square_area_proof
  (x s : ℝ)
  (h1 : x^2 = 3 * s)
  (h2 : 4 * x = s^2) :
  x^2 = 6 :=
  sorry

end square_area_proof_l34_34729


namespace neg_p_sufficient_not_necessary_for_neg_q_l34_34568

noncomputable def p (x : ℝ) : Prop := abs (x + 1) > 0
noncomputable def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l34_34568


namespace financed_amount_correct_l34_34340

-- Define the conditions
def monthly_payment : ℝ := 150.0
def years : ℝ := 5.0
def months_in_a_year : ℝ := 12.0

-- Define the total number of months
def total_months : ℝ := years * months_in_a_year

-- Define the amount financed
def total_financed : ℝ := monthly_payment * total_months

-- State the theorem
theorem financed_amount_correct :
  total_financed = 9000 :=
by
  -- Provide the proof here
  sorry

end financed_amount_correct_l34_34340


namespace sum_of_coefficients_l34_34319

noncomputable def u : ℕ → ℕ
| 0       => 5
| (n + 1) => u n + (3 + 4 * (n - 1))

theorem sum_of_coefficients :
  (2 + -3 + 6) = 5 :=
by {
  sorry
}

end sum_of_coefficients_l34_34319


namespace ratio_of_large_rooms_l34_34724

-- Definitions for the problem conditions
def total_classrooms : ℕ := 15
def total_students : ℕ := 400
def desks_in_large_room : ℕ := 30
def desks_in_small_room : ℕ := 25

-- Define x as the number of large (30-desk) rooms and y as the number of small (25-desk) rooms
variables (x y : ℕ)

-- Two conditions provided by the problem
def classrooms_condition := x + y = total_classrooms
def students_condition := desks_in_large_room * x + desks_in_small_room * y = total_students

-- Our main theorem to prove
theorem ratio_of_large_rooms :
  classrooms_condition x y →
  students_condition x y →
  (x : ℚ) / (total_classrooms : ℚ) = 1 / 3 :=
by
-- Here we would have our proof, but we leave it as "sorry" since the task only requires the statement.
sorry

end ratio_of_large_rooms_l34_34724


namespace max_extra_time_matches_l34_34168

theorem max_extra_time_matches (number_teams : ℕ) 
    (points_win : ℕ) (points_lose : ℕ) 
    (points_win_extra : ℕ) (points_lose_extra : ℕ) 
    (total_matches_2016 : number_teams = 2016)
    (pts_win_3 : points_win = 3)
    (pts_lose_0 : points_lose = 0)
    (pts_win_extra_2 : points_win_extra = 2)
    (pts_lose_extra_1 : points_lose_extra = 1) :
    ∃ N, N = 1512 := 
by {
  sorry
}

end max_extra_time_matches_l34_34168


namespace geometric_seq_a8_l34_34259

noncomputable def geometric_seq_term (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

noncomputable def geometric_seq_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_seq_a8
  (a₁ r : ℝ)
  (h1 : geometric_seq_sum a₁ r 3 = 7/4)
  (h2 : geometric_seq_sum a₁ r 6 = 63/4)
  (h3 : r ≠ 1) :
  geometric_seq_term a₁ r 8 = 32 :=
by
  sorry

end geometric_seq_a8_l34_34259


namespace relationship_among_abc_l34_34996

noncomputable def a : ℝ := Real.logb 11 10
noncomputable def b : ℝ := (Real.logb 11 9) ^ 2
noncomputable def c : ℝ := Real.logb 10 11

theorem relationship_among_abc : b < a ∧ a < c :=
  sorry

end relationship_among_abc_l34_34996


namespace smallest_possible_QNNN_l34_34909

theorem smallest_possible_QNNN :
  ∃ (Q N : ℕ), (N = 1 ∨ N = 5 ∨ N = 6) ∧ (NN = 10 * N + N) ∧ (Q * 1000 + NN * 10 + N = NN * N) ∧ (Q * 1000 + NN * 10 + N) = 275 :=
sorry

end smallest_possible_QNNN_l34_34909


namespace ab_minus_a_plus_b_eq_two_l34_34313

theorem ab_minus_a_plus_b_eq_two
  (a b : ℝ)
  (h1 : a + 1 ≠ 0)
  (h2 : b - 1 ≠ 0)
  (h3 : a + (1 / (a + 1)) = b + (1 / (b - 1)) - 2)
  (h4 : a - b + 2 ≠ 0)
: ab - a + b = 2 :=
sorry

end ab_minus_a_plus_b_eq_two_l34_34313


namespace profit_percentage_is_60_l34_34172

variable (SellingPrice CostPrice : ℝ)

noncomputable def Profit : ℝ := SellingPrice - CostPrice

noncomputable def ProfitPercentage : ℝ := (Profit SellingPrice CostPrice / CostPrice) * 100

theorem profit_percentage_is_60
  (h1 : SellingPrice = 400)
  (h2 : CostPrice = 250) :
  ProfitPercentage SellingPrice CostPrice = 60 := by
  sorry

end profit_percentage_is_60_l34_34172


namespace sum_of_three_consecutive_even_numbers_is_162_l34_34820

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l34_34820


namespace number_of_lines_with_negative_reciprocal_intercepts_l34_34282

-- Define the point (-2, 4)
def point : ℝ × ℝ := (-2, 4)

-- Define the condition that intercepts are negative reciprocals
def are_negative_reciprocals (a b : ℝ) : Prop :=
  a * b = -1

-- Define the proof problem: 
-- Number of lines through point (-2, 4) with intercepts negative reciprocals of each other
theorem number_of_lines_with_negative_reciprocal_intercepts :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ (a b : ℝ), are_negative_reciprocals a b →
  (∃ m k : ℝ, (k * (-2) + m = 4) ∧ ((m ⁻¹ = a ∧ k = b) ∨ (k = a ∧ m ⁻¹ = b))) :=
sorry

end number_of_lines_with_negative_reciprocal_intercepts_l34_34282


namespace best_sampling_method_l34_34398

theorem best_sampling_method :
  let elderly := 27
  let middle_aged := 54
  let young := 81
  let total_population := elderly + middle_aged + young
  let sample_size := 36
  let sampling_methods := ["simple random sampling", "systematic sampling", "stratified sampling"]
  stratified_sampling
:=
by
  sorry

end best_sampling_method_l34_34398


namespace initial_cards_eq_4_l34_34505

theorem initial_cards_eq_4 (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  sorry

end initial_cards_eq_4_l34_34505


namespace reversed_digits_sum_l34_34511

theorem reversed_digits_sum (a b n : ℕ) (x y : ℕ) (ha : a < 10) (hb : b < 10) 
(hx : x = 10 * a + b) (hy : y = 10 * b + a) (hsq : x^2 + y^2 = n^2) : 
  x + y + n = 264 :=
sorry

end reversed_digits_sum_l34_34511


namespace algebraic_expression_equality_l34_34177

variable {x : ℝ}

theorem algebraic_expression_equality (h : x^2 + 3*x + 8 = 7) : 3*x^2 + 9*x - 2 = -5 := 
by
  sorry

end algebraic_expression_equality_l34_34177


namespace extremum_of_f_l34_34416

def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

theorem extremum_of_f :
  ∃ (xmin xmax : ℝ) (x1 y1 : ℝ), f x1 y1 = xmin ∧ (x1 = 11 ∧ y1 = 2) ∧
  ∃ (xmax : ℝ) (x2 y2 : ℝ), f x2 y2 = xmax ∧ (x2 = 1 ∧ y2 = 4) ∧
  xmin = 10 ∧ xmax = 570 := 
by
  sorry

end extremum_of_f_l34_34416


namespace probability_of_experts_winning_l34_34269

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l34_34269


namespace find_p_l34_34710

noncomputable def binomial_parameter (n : ℕ) (p : ℚ) (E : ℚ) (D : ℚ) : Prop :=
  E = n * p ∧ D = n * p * (1 - p)

theorem find_p (n : ℕ) (p : ℚ) 
  (hE : n * p = 50)
  (hD : n * p * (1 - p) = 30)
  : p = 2 / 5 :=
sorry

end find_p_l34_34710


namespace calculate_three_times_neg_two_l34_34450

-- Define the multiplication of a positive and a negative number resulting in a negative number
def multiply_positive_negative (a b : Int) (ha : a > 0) (hb : b < 0) : Int :=
  a * b

-- Define the absolute value multiplication
def absolute_value_multiplication (a b : Int) : Int :=
  abs a * abs b

-- The theorem that verifies the calculation
theorem calculate_three_times_neg_two : 3 * (-2) = -6 :=
by
  -- Using the given conditions to conclude the result
  sorry

end calculate_three_times_neg_two_l34_34450


namespace total_cost_l34_34675

noncomputable def C1 : ℝ := 990 / 1.10
noncomputable def C2 : ℝ := 990 / 0.90

theorem total_cost (SP : ℝ) (profit_rate loss_rate : ℝ) : SP = 990 ∧ profit_rate = 0.10 ∧ loss_rate = 0.10 →
  C1 + C2 = 2000 :=
by
  intro h
  -- Show the sum of C1 and C2 equals 2000
  sorry

end total_cost_l34_34675


namespace resistor_problem_l34_34733

theorem resistor_problem (R : ℝ)
  (initial_resistance : ℝ := 3 * R)
  (parallel_resistance : ℝ := R / 3)
  (resistance_change : ℝ := initial_resistance - parallel_resistance)
  (condition : resistance_change = 10) : 
  R = 3.75 := by
  sorry

end resistor_problem_l34_34733


namespace michael_twice_jacob_l34_34850

variable {J M Y : ℕ}

theorem michael_twice_jacob :
  (J + 4 = 13) → (M = J + 12) → (M + Y = 2 * (J + Y)) → (Y = 3) := by
  sorry

end michael_twice_jacob_l34_34850


namespace rachel_minutes_before_bed_l34_34048

-- Define the conditions in the Lean Lean.
def minutes_spent_solving_before_bed (m : ℕ) : Prop :=
  let problems_solved_before_bed := 5 * m
  let problems_finished_at_lunch := 16
  let total_problems_solved := 76
  problems_solved_before_bed + problems_finished_at_lunch = total_problems_solved

-- The statement we want to prove
theorem rachel_minutes_before_bed : ∃ m : ℕ, minutes_spent_solving_before_bed m ∧ m = 12 :=
sorry

end rachel_minutes_before_bed_l34_34048


namespace same_sign_m_minus_n_opposite_sign_m_plus_n_l34_34699

-- Definitions and Conditions
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom abs_m_eq_4 : |m| = 4
axiom abs_n_eq_3 : |n| = 3

-- Part 1: Prove m - n when m and n have the same sign
theorem same_sign_m_minus_n :
  (m > 0 ∧ n > 0) ∨ (m < 0 ∧ n < 0) → (m - n = 1 ∨ m - n = -1) :=
by
  sorry

-- Part 2: Prove m + n when m and n have opposite signs
theorem opposite_sign_m_plus_n :
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = 1 ∨ m + n = -1) :=
by
  sorry

end same_sign_m_minus_n_opposite_sign_m_plus_n_l34_34699


namespace side_length_of_square_l34_34034

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l34_34034


namespace triangle_height_l34_34378

theorem triangle_height (area base height : ℝ) (h1 : area = 500) (h2 : base = 50) (h3 : area = (1 / 2) * base * height) : height = 20 :=
sorry

end triangle_height_l34_34378


namespace Masc_age_difference_l34_34997

theorem Masc_age_difference (masc_age sam_age : ℕ) (h1 : masc_age + sam_age = 27) (h2 : masc_age = 17) (h3 : sam_age = 10) : masc_age - sam_age = 7 :=
by {
  -- Proof would go here, but it's omitted as per instructions
  sorry
}

end Masc_age_difference_l34_34997


namespace total_ticket_count_is_59_l34_34223

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l34_34223


namespace sunil_total_amount_l34_34274

noncomputable def principal (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  CI / ((1 + R / 100) ^ T - 1)

noncomputable def total_amount (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  let P := principal CI R T
  P + CI

theorem sunil_total_amount (CI : ℝ) (R : ℝ) (T : ℕ) :
  CI = 420 → R = 10 → T = 2 → total_amount CI R T = 2420 := by
  intros hCI hR hT
  rw [hCI, hR, hT]
  sorry

end sunil_total_amount_l34_34274


namespace sarah_jim_ratio_l34_34071

theorem sarah_jim_ratio
  (Tim_toads : ℕ)
  (hTim : Tim_toads = 30)
  (Jim_toads : ℕ)
  (hJim : Jim_toads = Tim_toads + 20)
  (Sarah_toads : ℕ)
  (hSarah : Sarah_toads = 100) :
  Sarah_toads / Jim_toads = 2 :=
by
  sorry

end sarah_jim_ratio_l34_34071


namespace minimize_quadratic_function_l34_34509

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l34_34509


namespace g_is_correct_l34_34041

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2

axiom g_functional_eq : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2

theorem g_is_correct : ∀ x : ℝ, g x = 2 - 2 * x := 
by 
  sorry

end g_is_correct_l34_34041


namespace problem_statement_l34_34471

def g (x : ℝ) : ℝ := x ^ 3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem problem_statement : f (g 3) = 53 :=
by
  sorry

end problem_statement_l34_34471


namespace simplify_expression_l34_34704

theorem simplify_expression (x : ℤ) : 
  (12*x^10 + 5*x^9 + 3*x^8) + (2*x^12 + 9*x^10 + 4*x^8 + 6*x^4 + 7*x^2 + 10)
  = 2*x^12 + 21*x^10 + 5*x^9 + 7*x^8 + 6*x^4 + 7*x^2 + 10 :=
by sorry

end simplify_expression_l34_34704


namespace neighbors_receive_equal_mangoes_l34_34341

-- Definitions from conditions
def total_mangoes : ℕ := 560
def mangoes_sold : ℕ := total_mangoes / 2
def remaining_mangoes : ℕ := total_mangoes - mangoes_sold
def neighbors : ℕ := 8

-- The lean statement
theorem neighbors_receive_equal_mangoes :
  remaining_mangoes / neighbors = 35 :=
by
  -- This is where the proof would go, but we'll leave it with sorry for now.
  sorry

end neighbors_receive_equal_mangoes_l34_34341


namespace min_value_of_expression_l34_34565

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x = a^2 + b^2 + (1 / (a + b)^2) + (1 / (a * b)) ∧ x = Real.sqrt 10 :=
sorry

end min_value_of_expression_l34_34565


namespace domain_of_log_function_l34_34818

theorem domain_of_log_function : 
  { x : ℝ | x < 1 ∨ x > 2 } = { x : ℝ | 0 < x^2 - 3 * x + 2 } :=
by sorry

end domain_of_log_function_l34_34818


namespace total_students_in_class_l34_34401

-- No need for noncomputable def here as we're dealing with basic arithmetic

theorem total_students_in_class (jellybeans_total jellybeans_left boys_girls_diff : ℕ)
  (girls boys students : ℕ) :
  jellybeans_total = 450 →
  jellybeans_left = 10 →
  boys_girls_diff = 3 →
  boys = girls + boys_girls_diff →
  students = girls + boys →
  (girls * girls) + (boys * boys) = jellybeans_total - jellybeans_left →
  students = 29 := 
by
  intro h_total h_left h_diff h_boys h_students h_distribution
  sorry

end total_students_in_class_l34_34401


namespace fraction_of_students_l34_34332

theorem fraction_of_students {G B T : ℕ} (h1 : B = 2 * G) (h2 : T = G + B) (h3 : (1 / 2) * (G : ℝ) = (x : ℝ) * (T : ℝ)) : x = (1 / 6) :=
by sorry

end fraction_of_students_l34_34332


namespace dogs_not_eat_either_l34_34387

-- Let's define the conditions
variables (total_dogs : ℕ) (dogs_like_carrots : ℕ) (dogs_like_chicken : ℕ) (dogs_like_both : ℕ)

-- Given conditions
def conditions : Prop :=
  total_dogs = 85 ∧
  dogs_like_carrots = 12 ∧
  dogs_like_chicken = 62 ∧
  dogs_like_both = 8

-- Problem to solve
theorem dogs_not_eat_either (h : conditions total_dogs dogs_like_carrots dogs_like_chicken dogs_like_both) :
  (total_dogs - (dogs_like_carrots - dogs_like_both + dogs_like_chicken - dogs_like_both + dogs_like_both)) = 19 :=
by {
  sorry 
}

end dogs_not_eat_either_l34_34387


namespace meaningful_expression_range_l34_34823

theorem meaningful_expression_range (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 2 ≠ 0) → x < 2 :=
by
  sorry

end meaningful_expression_range_l34_34823


namespace largest_integer_odd_divides_expression_l34_34802

theorem largest_integer_odd_divides_expression (x : ℕ) (h_odd : x % 2 = 1) : 
    ∃ k, k = 384 ∧ ∀ m, m ∣ (8*x + 6) * (8*x + 10) * (4*x + 4) → m ≤ k :=
by {
  sorry
}

end largest_integer_odd_divides_expression_l34_34802


namespace infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l34_34294

noncomputable def grid_size := 10
noncomputable def initial_infected_count_1 := 9
noncomputable def initial_infected_count_2 := 10

def condition (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n → 
  infected + steps * (infected / 2) < grid_size * grid_size

def can_infect_entire_grid (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n ∧ (
  ∃ t : ℕ, infected + t * (infected / 2) = grid_size * grid_size)

theorem infection_does_not_spread_with_9_cells :
  ¬ can_infect_entire_grid initial_infected_count_1 :=
by
  sorry

theorem minimum_infected_cells_needed :
  condition initial_infected_count_2 :=
by
  sorry

end infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l34_34294


namespace value_of_f_at_neg_one_l34_34187

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (x : ℝ) (h : x ≠ 0) : ℝ := (2 - 3 * x^2) / x^2

theorem value_of_f_at_neg_one : f (-1) (by norm_num) = -1 := 
sorry

end value_of_f_at_neg_one_l34_34187


namespace weight_of_pecans_l34_34025

theorem weight_of_pecans (total_weight_of_nuts almonds_weight pecans_weight : ℝ)
  (h1 : total_weight_of_nuts = 0.52)
  (h2 : almonds_weight = 0.14)
  (h3 : pecans_weight = total_weight_of_nuts - almonds_weight) :
  pecans_weight = 0.38 :=
  by
    sorry

end weight_of_pecans_l34_34025


namespace Clarissa_photos_needed_l34_34747

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end Clarissa_photos_needed_l34_34747


namespace proof_g_2_l34_34660

def g (x : ℝ) : ℝ := 3 * x ^ 8 - 4 * x ^ 4 + 2 * x ^ 2 - 6

theorem proof_g_2 :
  g (-2) = 10 → g (2) = 1402 := by
  sorry

end proof_g_2_l34_34660


namespace inequality_abc_l34_34857

theorem inequality_abc (a b c : ℝ) (h1 : a ∈ Set.Icc (-1 : ℝ) 2) (h2 : b ∈ Set.Icc (-1 : ℝ) 2) (h3 : c ∈ Set.Icc (-1 : ℝ) 2) : 
  a * b * c + 4 ≥ a * b + b * c + c * a := 
sorry

end inequality_abc_l34_34857


namespace order_magnitudes_ln_subtraction_l34_34770

noncomputable def ln (x : ℝ) : ℝ := Real.log x -- Assuming the natural logarithm definition for real numbers

theorem order_magnitudes_ln_subtraction :
  (ln (3/2) - (3/2)) > (ln 3 - 3) ∧ 
  (ln 3 - 3) > (ln π - π) :=
sorry

end order_magnitudes_ln_subtraction_l34_34770


namespace segment_length_reflection_l34_34322

theorem segment_length_reflection (F : ℝ × ℝ) (F' : ℝ × ℝ)
  (hF : F = (-4, -2)) (hF' : F' = (4, -2)) :
  dist F F' = 8 :=
by
  sorry

end segment_length_reflection_l34_34322


namespace eq_b_minus_a_l34_34507

   -- Definition for rotating a point counterclockwise by 180° around another point
   def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
     (2 * h - x, 2 * k - y)

   -- Definition for reflecting a point about the line y = -x
   def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
     (-y, -x)

   -- Given point Q(a, b)
   variables (a b : ℝ)

   -- Image of Q after the transformations
   def Q_transformed :=
     (5, -1)

   -- Image of Q after reflection about y = -x
   def Q_reflected :=
     reflect_y_eq_neg_x (5) (-1)

   -- Image of Q after 180° rotation around (2,3)
   def Q_original :=
     rotate_180 (2) (3) a b

   -- Statement we want to prove:
   theorem eq_b_minus_a : b - a = 6 :=
   by
     -- Calculation steps
     sorry
   
end eq_b_minus_a_l34_34507


namespace wedge_volume_calculation_l34_34895

theorem wedge_volume_calculation :
  let r := 5 
  let h := 8 
  let V := (1 / 3) * (Real.pi * r^2 * h) 
  V = (200 * Real.pi) / 3 :=
by
  let r := 5
  let h := 8
  let V := (1 / 3) * (Real.pi * r^2 * h)
  -- Prove the equality step is omitted as per the prompt
  sorry

end wedge_volume_calculation_l34_34895


namespace largest_4_digit_congruent_to_15_mod_25_l34_34343

theorem largest_4_digit_congruent_to_15_mod_25 : 
  ∀ x : ℕ, (1000 ≤ x ∧ x < 10000 ∧ x % 25 = 15) → x = 9990 :=
by
  intros x h
  sorry

end largest_4_digit_congruent_to_15_mod_25_l34_34343


namespace binomial_10_2_equals_45_l34_34766

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l34_34766


namespace necessary_and_sufficient_condition_l34_34383

theorem necessary_and_sufficient_condition (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 4) :=
by
  sorry

end necessary_and_sufficient_condition_l34_34383


namespace C_gets_more_than_D_l34_34321

-- Define the conditions
def proportion_B := 3
def share_B : ℕ := 3000
def proportion_C := 5
def proportion_D := 4

-- Define the parts based on B's share
def part_value := share_B / proportion_B

-- Define the shares based on the proportions
def share_C := proportion_C * part_value
def share_D := proportion_D * part_value

-- Prove the final statement about the difference
theorem C_gets_more_than_D : share_C - share_D = 1000 :=
by
  -- Proof goes here
  sorry

end C_gets_more_than_D_l34_34321


namespace cylinder_volume_ratio_l34_34066

theorem cylinder_volume_ratio
  (h : ℝ)     -- height of cylinder B (radius of cylinder A)
  (r : ℝ)     -- radius of cylinder B (height of cylinder A)
  (VA : ℝ)    -- volume of cylinder A
  (VB : ℝ)    -- volume of cylinder B
  (cond1 : r = h / 3)
  (cond2 : VB = 3 * VA)
  (cond3 : VB = N * Real.pi * h^3) :
  N = 1 / 3 := 
sorry

end cylinder_volume_ratio_l34_34066


namespace book_vs_necklace_price_difference_l34_34303

-- Problem-specific definitions and conditions
def necklace_price : ℕ := 34
def limit_price : ℕ := 70
def overspent : ℕ := 3
def total_spent : ℕ := limit_price + overspent
def book_price : ℕ := total_spent - necklace_price

-- Lean statement to prove the correct answer
theorem book_vs_necklace_price_difference :
  book_price - necklace_price = 5 := by
  sorry

end book_vs_necklace_price_difference_l34_34303


namespace flute_player_count_l34_34480

-- Define the total number of people in the orchestra
def total_people : Nat := 21

-- Define the number of people in each section
def sebastian : Nat := 1
def brass : Nat := 4 + 2 + 1
def strings : Nat := 3 + 1 + 1
def woodwinds_excluding_flutes : Nat := 3
def maestro : Nat := 1

-- Calculate the number of accounted people
def accounted_people : Nat := sebastian + brass + strings + woodwinds_excluding_flutes + maestro

-- State the number of flute players
def flute_players : Nat := total_people - accounted_people

-- The theorem stating the number of flute players
theorem flute_player_count : flute_players = 4 := by
  unfold flute_players accounted_people total_people sebastian brass strings woodwinds_excluding_flutes maestro
  -- Need to evaluate the expressions step by step to reach the final number 4.
  -- (Or simply "sorry" since we are skipping the proof steps)
  sorry

end flute_player_count_l34_34480


namespace b_days_work_alone_l34_34261

theorem b_days_work_alone 
  (W_b : ℝ)  -- Work done by B in one day
  (W_a : ℝ)  -- Work done by A in one day
  (D_b : ℝ)  -- Number of days for B to complete the work alone
  (h1 : W_a = 2 * W_b)  -- A is twice as good a workman as B
  (h2 : 7 * (W_a + W_b) = D_b * W_b)  -- A and B took 7 days together to do the work
  : D_b = 21 :=
sorry

end b_days_work_alone_l34_34261


namespace union_A_B_l34_34016

def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem union_A_B : A ∪ B = {x | x > 0} :=
by
  sorry

end union_A_B_l34_34016


namespace range_of_a_l34_34983

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 :=
by
  sorry

end range_of_a_l34_34983


namespace least_integer_a_divisible_by_240_l34_34197

theorem least_integer_a_divisible_by_240 (a : ℤ) (h1 : 240 ∣ a^3) : a ≥ 60 := by
  sorry

end least_integer_a_divisible_by_240_l34_34197


namespace parabola_behavior_l34_34663

theorem parabola_behavior (x : ℝ) (h : x < 0) : ∃ y, y = 2*x^2 - 1 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0 → (2*x1^2 - 1) > (2*x2^2 - 1) :=
by
  sorry

end parabola_behavior_l34_34663


namespace todd_initial_gum_l34_34285

theorem todd_initial_gum (x : ℝ)
(h1 : 150 = 0.25 * x)
(h2 : x + 150 = 890) :
x = 712 :=
by
  -- Here "by" is used to denote the beginning of proof block
  sorry -- Proof will be filled in later.

end todd_initial_gum_l34_34285


namespace int_valued_fractions_l34_34642

theorem int_valued_fractions (a : ℤ) :
  ∃ k : ℤ, (a^2 - 21 * a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17 :=
by {
  sorry
}

end int_valued_fractions_l34_34642


namespace counterexample_conjecture_l34_34494

theorem counterexample_conjecture 
    (odd_gt_5 : ℕ → Prop) 
    (is_prime : ℕ → Prop) 
    (conjecture : ∀ n, odd_gt_5 n → ∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) : 
    ∃ n, odd_gt_5 n ∧ ¬ (∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) :=
sorry

end counterexample_conjecture_l34_34494


namespace tallest_is_Justina_l34_34279

variable (H G I J K : ℝ)

axiom height_conditions1 : H < G
axiom height_conditions2 : G < J
axiom height_conditions3 : K < I
axiom height_conditions4 : I < G

theorem tallest_is_Justina : J > G ∧ J > H ∧ J > I ∧ J > K :=
by
  sorry

end tallest_is_Justina_l34_34279


namespace parabola_directrix_l34_34382

theorem parabola_directrix (x y : ℝ) (h : y = 4 * (x - 1)^2 + 3) : y = 11 / 4 :=
sorry

end parabola_directrix_l34_34382


namespace total_cost_of_feeding_pets_for_one_week_l34_34986

-- Definitions based on conditions
def turtle_food_per_weight : ℚ := 1 / (1 / 2)
def turtle_weight : ℚ := 30
def turtle_food_qty_per_jar : ℚ := 15
def turtle_food_cost_per_jar : ℚ := 3

def bird_food_per_weight : ℚ := 2
def bird_weight : ℚ := 8
def bird_food_qty_per_bag : ℚ := 40
def bird_food_cost_per_bag : ℚ := 5

def hamster_food_per_weight : ℚ := 1.5 / (1 / 2)
def hamster_weight : ℚ := 3
def hamster_food_qty_per_box : ℚ := 20
def hamster_food_cost_per_box : ℚ := 4

-- Theorem stating the equivalent proof problem
theorem total_cost_of_feeding_pets_for_one_week :
  let turtle_food_needed := (turtle_weight * turtle_food_per_weight)
  let turtle_jars_needed := turtle_food_needed / turtle_food_qty_per_jar
  let turtle_cost := turtle_jars_needed * turtle_food_cost_per_jar
  let bird_food_needed := (bird_weight * bird_food_per_weight)
  let bird_bags_needed := bird_food_needed / bird_food_qty_per_bag
  let bird_cost := if bird_bags_needed < 1 then bird_food_cost_per_bag else bird_bags_needed * bird_food_cost_per_bag
  let hamster_food_needed := (hamster_weight * hamster_food_per_weight)
  let hamster_boxes_needed := hamster_food_needed / hamster_food_qty_per_box
  let hamster_cost := if hamster_boxes_needed < 1 then hamster_food_cost_per_box else hamster_boxes_needed * hamster_food_cost_per_box
  turtle_cost + bird_cost + hamster_cost = 21 :=
by
  sorry

end total_cost_of_feeding_pets_for_one_week_l34_34986


namespace function_zero_solution_l34_34599

def floor (x : ℝ) : ℤ := sorry -- Define floor function properly.

theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = (-1) ^ (floor y) * f x + (-1) ^ (floor x) * f y) →
  (∀ x : ℝ, f x = 0) := 
by
  -- Proof goes here
  sorry

end function_zero_solution_l34_34599


namespace tan_eq_sin3x_solutions_l34_34441

open Real

theorem tan_eq_sin3x_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ tan x = sin (3 * x)) ∧ s.card = 6 :=
sorry

end tan_eq_sin3x_solutions_l34_34441


namespace first_interest_rate_l34_34653

theorem first_interest_rate (r : ℝ) : 
  (70000:ℝ) = (60000:ℝ) + (10000:ℝ) →
  (8000:ℝ) = (60000 * r / 100) + (10000 * 20 / 100) →
  r = 10 :=
by
  intros h1 h2
  sorry

end first_interest_rate_l34_34653


namespace goods_train_length_l34_34890

theorem goods_train_length 
  (v_kmph : ℝ) (L_p : ℝ) (t : ℝ) (v_mps : ℝ) (d : ℝ) (L_t : ℝ) 
  (h1 : v_kmph = 96) 
  (h2 : L_p = 480) 
  (h3 : t = 36) 
  (h4 : v_mps = v_kmph * (5/18)) 
  (h5 : d = v_mps * t) : 
  L_t = d - L_p :=
sorry

end goods_train_length_l34_34890


namespace max_value_xyz_l34_34721

theorem max_value_xyz (x y z : ℝ) (h : x + y + 2 * z = 5) : 
  (∃ x y z : ℝ, x + y + 2 * z = 5 ∧ xy + xz + yz = 25/6) :=
sorry

end max_value_xyz_l34_34721


namespace athlete_last_finish_l34_34592

theorem athlete_last_finish (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  let T1 := 1 / v1 + 2 / v2 
  let T2 := 1 / v2 + 2 / v3
  let T3 := 1 / v3 + 2 / v1
  T2 > T1 ∧ T2 > T3 :=
by
  sorry

end athlete_last_finish_l34_34592


namespace find_a_l34_34645

noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

theorem find_a (x a : ℝ) (hx : f a = 4) (ha : a = 2 * x + 1) : a = 5 :=
by
  sorry

end find_a_l34_34645


namespace part1_l34_34559

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : 
  f x 1 ≥ 1 :=
sorry

end part1_l34_34559


namespace find_interest_rate_of_second_part_l34_34961

-- Definitions for the problem
def total_sum : ℚ := 2678
def P2 : ℚ := 1648
def P1 : ℚ := total_sum - P2
def r1 : ℚ := 0.03  -- 3% per annum
def t1 : ℚ := 8     -- 8 years
def I1 : ℚ := P1 * r1 * t1
def t2 : ℚ := 3     -- 3 years

-- Statement to prove
theorem find_interest_rate_of_second_part : ∃ r2 : ℚ, I1 = P2 * r2 * t2 ∧ r2 * 100 = 5 := by
  sorry

end find_interest_rate_of_second_part_l34_34961


namespace lead_points_l34_34079

-- Define final scores
def final_score_team : ℕ := 68
def final_score_green : ℕ := 39

-- Prove the lead
theorem lead_points : final_score_team - final_score_green = 29 :=
by
  sorry

end lead_points_l34_34079


namespace how_many_bottles_did_maria_drink_l34_34736

-- Define the conditions as variables and constants.
variable (x : ℕ)
def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def total_bottles_after_drinking_and_buying : ℕ := 51

-- The goal is to prove that Maria drank 8 bottles of water.
theorem how_many_bottles_did_maria_drink (h : initial_bottles - x + bought_bottles = total_bottles_after_drinking_and_buying) : x = 8 :=
by
  sorry

end how_many_bottles_did_maria_drink_l34_34736


namespace proof_problem_l34_34778

variables {a : ℕ → ℕ} -- sequence a_n is positive integers
variables {b : ℕ → ℕ} -- sequence b_n is integers
variables {q : ℕ} -- ratio for geometric sequence
variables {d : ℕ} -- difference for arithmetic sequence
variables {a1 b1 : ℕ} -- initial terms for the sequences

-- Additional conditions as per the problem statement
def geometric_seq (a : ℕ → ℕ) (a1 q : ℕ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n-1)

def arithmetic_seq (b : ℕ → ℕ) (b1 d : ℕ) : Prop :=
∀ n : ℕ, b n = b1 + (n-1) * d

-- Given conditions
variable (geometric : geometric_seq a a1 q)
variable (arithmetic : arithmetic_seq b b1 d)
variable (equal_term : a 6 = b 7)

-- The proof task
theorem proof_problem : a 3 + a 9 ≥ b 4 + b 10 :=
by sorry

end proof_problem_l34_34778


namespace no_such_natural_number_exists_l34_34312

theorem no_such_natural_number_exists :
  ¬ ∃ (n : ℕ), (∃ (m k : ℤ), 2 * n - 5 = 9 * m ∧ n - 2 = 15 * k) :=
by
  sorry

end no_such_natural_number_exists_l34_34312


namespace find_weeks_period_l34_34933

def weekly_addition : ℕ := 3
def bikes_sold : ℕ := 18
def bikes_in_stock : ℕ := 45
def initial_stock : ℕ := 51

theorem find_weeks_period (x : ℕ) :
  initial_stock + weekly_addition * x - bikes_sold = bikes_in_stock ↔ x = 4 := 
by 
  sorry

end find_weeks_period_l34_34933


namespace solve_linear_system_l34_34952

theorem solve_linear_system :
  ∃ x y : ℚ, (3 * x - y = 4) ∧ (6 * x - 3 * y = 10) ∧ (x = 2 / 3) ∧ (y = -2) :=
by
  sorry

end solve_linear_system_l34_34952


namespace mushroom_problem_l34_34318

variables (x1 x2 x3 x4 : ℕ)

theorem mushroom_problem
  (h1 : x1 + x2 = 6)
  (h2 : x1 + x3 = 7)
  (h3 : x2 + x3 = 9)
  (h4 : x2 + x4 = 11)
  (h5 : x3 + x4 = 12)
  (h6 : x1 + x4 = 9) :
  x1 = 2 ∧ x2 = 4 ∧ x3 = 5 ∧ x4 = 7 := 
  by
    sorry

end mushroom_problem_l34_34318


namespace binary_to_decimal_and_septal_l34_34967

theorem binary_to_decimal_and_septal :
  let bin : ℕ := 110101
  let dec : ℕ := 53
  let septal : ℕ := 104
  let convert_to_decimal (b : ℕ) : ℕ := 
    (b % 10) * 2^0 + ((b / 10) % 10) * 2^1 + ((b / 100) % 10) * 2^2 + 
    ((b / 1000) % 10) * 2^3 + ((b / 10000) % 10) * 2^4 + ((b / 100000) % 10) * 2^5
  let convert_to_septal (n : ℕ) : ℕ :=
    let rec aux (n : ℕ) (acc : ℕ) (place : ℕ) : ℕ :=
      if n = 0 then acc
      else aux (n / 7) (acc + (n % 7) * place) (place * 10)
    aux n 0 1
  convert_to_decimal bin = dec ∧ convert_to_septal dec = septal :=
by
  sorry

end binary_to_decimal_and_septal_l34_34967


namespace tickets_spent_on_beanie_l34_34484

-- Define the initial number of tickets Jerry had.
def initial_tickets : ℕ := 4

-- Define the number of tickets Jerry won later.
def won_tickets : ℕ := 47

-- Define the current number of tickets Jerry has.
def current_tickets : ℕ := 49

-- The statement of the problem to prove the tickets spent on the beanie.
theorem tickets_spent_on_beanie :
  initial_tickets + won_tickets - 2 = current_tickets := by
  sorry

end tickets_spent_on_beanie_l34_34484


namespace gcd_of_polynomial_l34_34407

theorem gcd_of_polynomial (a : ℤ) (h : 720 ∣ a) : Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := 
by 
  sorry

end gcd_of_polynomial_l34_34407


namespace distance_to_x_axis_l34_34518

theorem distance_to_x_axis (x y : ℤ) (h : (x, y) = (-3, 5)) : |y| = 5 := by
  -- coordinates of point A are (-3, 5)
  sorry

end distance_to_x_axis_l34_34518


namespace range_of_a_l34_34557

theorem range_of_a (a : ℝ) : (1 < a) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  (1 / (x1 + 2) = a * |x1| ∧ 1 / (x2 + 2) = a * |x2| ∧ 1 / (x3 + 2) = a * |x3|) :=
sorry

end range_of_a_l34_34557


namespace pencils_sold_l34_34408

theorem pencils_sold (C S : ℝ) (n : ℝ) 
  (h1 : 12 * C = n * S) (h2 : S = 1.5 * C) : n = 8 := by
  sorry

end pencils_sold_l34_34408


namespace greatest_N_consecutive_sum_50_l34_34310

theorem greatest_N_consecutive_sum_50 :
  ∃ N a : ℤ, (N > 0) ∧ (N * (2 * a + N - 1) = 100) ∧ (N = 100) :=
by
  sorry

end greatest_N_consecutive_sum_50_l34_34310


namespace unknown_sum_of_digits_l34_34037

theorem unknown_sum_of_digits 
  (A B C D : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h2 : D = 1)
  (h3 : (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D) : 
  A + B = 0 := 
sorry

end unknown_sum_of_digits_l34_34037


namespace sum_of_sequence_eq_six_seventeenth_l34_34406

noncomputable def cn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.cos (n * Real.arctan (2 / 3))
noncomputable def dn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.sin (n * Real.arctan (2 / 3))

theorem sum_of_sequence_eq_six_seventeenth : 
  (∑' n : ℕ, (cn n * dn n / 8^n)) = 6/17 := sorry

end sum_of_sequence_eq_six_seventeenth_l34_34406


namespace mod_product_l34_34440

theorem mod_product :
  (105 * 86 * 97) % 25 = 10 :=
by
  sorry

end mod_product_l34_34440


namespace three_digit_sum_seven_l34_34288

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l34_34288


namespace total_cups_of_ingredients_l34_34831

theorem total_cups_of_ingredients
  (ratio_butter : ℕ) (ratio_flour : ℕ) (ratio_sugar : ℕ)
  (flour_cups : ℕ)
  (h_ratio : ratio_butter = 2 ∧ ratio_flour = 3 ∧ ratio_sugar = 5)
  (h_flour : flour_cups = 6) :
  let part_cups := flour_cups / ratio_flour
  let butter_cups := ratio_butter * part_cups
  let sugar_cups := ratio_sugar * part_cups
  let total_cups := butter_cups + flour_cups + sugar_cups
  total_cups = 20 :=
by
  sorry

end total_cups_of_ingredients_l34_34831


namespace second_largest_geometric_sum_l34_34331

theorem second_largest_geometric_sum {a r : ℕ} (h_sum: a + a * r + a * r^2 + a * r^3 = 1417) (h_geometric: 1 + r + r^2 + r^3 ∣ 1417) : (a * r^2 = 272) :=
sorry

end second_largest_geometric_sum_l34_34331


namespace tangent_parallel_x_axis_monotonically_increasing_intervals_l34_34970

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem tangent_parallel_x_axis (m n : ℝ) (h : m ≠ 0) (h_tangent : 3 * m * (2:ℝ)^2 + 2 * n * (2:ℝ) = 0) :
  n = -3 * m :=
by
  sorry

theorem monotonically_increasing_intervals (m : ℝ) (h : m ≠ 0) : 
  (∀ x : ℝ, 3 * m * x * (x - (2 : ℝ)) > 0 ↔ 
    if m > 0 then x < 0 ∨ 2 < x else 0 < x ∧ x < 2) :=
by
  sorry

end tangent_parallel_x_axis_monotonically_increasing_intervals_l34_34970


namespace function_decreasing_on_interval_l34_34899

noncomputable def g (x : ℝ) := -(1 / 3) * Real.sin (4 * x - Real.pi / 3)
noncomputable def f (x : ℝ) := -(1 / 3) * Real.sin (4 * x)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 8) → (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 8) → x < y → f x > f y :=
sorry

end function_decreasing_on_interval_l34_34899


namespace meaningful_fraction_l34_34629

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 5 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end meaningful_fraction_l34_34629


namespace product_mod_7_l34_34388

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l34_34388


namespace speed_conversion_l34_34703

noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

theorem speed_conversion (h : kmh_to_ms 1 = 1000 / 3600) :
  kmh_to_ms 1.7 = 0.4722 :=
by sorry

end speed_conversion_l34_34703


namespace paving_stones_needed_l34_34648

variables (length_courtyard width_courtyard num_paving_stones length_paving_stone area_courtyard area_paving_stone : ℝ)
noncomputable def width_paving_stone := 2

theorem paving_stones_needed : 
  length_courtyard = 60 → 
  width_courtyard = 14 → 
  num_paving_stones = 140 →
  length_paving_stone = 3 →
  area_courtyard = length_courtyard * width_courtyard →
  area_paving_stone = length_paving_stone * width_paving_stone →
  num_paving_stones = area_courtyard / area_paving_stone :=
by
  intros h_length_courtyard h_width_courtyard h_num_paving_stones h_length_paving_stone h_area_courtyard h_area_paving_stone
  rw [h_length_courtyard, h_width_courtyard, h_length_paving_stone] at *
  simp at *
  sorry

end paving_stones_needed_l34_34648


namespace Geli_pushups_total_l34_34838

variable (x : ℕ)
variable (total_pushups : ℕ)

theorem Geli_pushups_total (h : 10 + (10 + x) + (10 + 2 * x) = 45) : x = 5 :=
by
  sorry

end Geli_pushups_total_l34_34838


namespace smallest_k_for_g_l34_34508

theorem smallest_k_for_g (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x + k = -3) ↔ k ≤ -3/4 := sorry

end smallest_k_for_g_l34_34508


namespace tangent_line_to_circle_l34_34845

theorem tangent_line_to_circle (c : ℝ) (h : 0 < c) : 
  (∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c) ↔ c = 4 :=
by sorry

end tangent_line_to_circle_l34_34845


namespace certain_number_exists_l34_34586

theorem certain_number_exists
  (N : ℕ) 
  (hN : ∀ x, x < N → x % 2 = 1 → ∃ k m, k = 5 * m ∧ x = k ∧ m % 2 = 1) :
  N = 76 := by
  sorry

end certain_number_exists_l34_34586


namespace cycle_original_cost_l34_34459

theorem cycle_original_cost (SP : ℝ) (gain : ℝ) (CP : ℝ) (h₁ : SP = 2000) (h₂ : gain = 1) (h₃ : SP = CP * (1 + gain)) : CP = 1000 :=
by
  sorry

end cycle_original_cost_l34_34459


namespace problem_statement_l34_34325

-- Defining the condition x^3 = 8
def condition1 (x : ℝ) : Prop := x^3 = 8

-- Defining the function f(x) = (x-1)(x+1)(x^2 + x + 1)
def f (x : ℝ) : ℝ := (x - 1) * (x + 1) * (x^2 + x + 1)

-- The theorem we want to prove: For any x satisfying the condition, the function value is 21
theorem problem_statement (x : ℝ) (h : condition1 x) : f x = 21 := 
by
  sorry

end problem_statement_l34_34325


namespace common_chord_properties_l34_34067

noncomputable def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 1 = 0

noncomputable def length_common_chord : ℝ := 2 * Real.sqrt 5

theorem common_chord_properties :
  (∀ x y : ℝ, 
    x^2 + y^2 + 2 * x + 8 * y - 8 = 0 ∧
    x^2 + y^2 - 4 * x - 4 * y - 2 = 0 →
    line_equation x y) ∧ 
  length_common_chord = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_properties_l34_34067


namespace sum_of_three_numbers_l34_34428

theorem sum_of_three_numbers (a b c : ℕ)
    (h1 : a + b = 35)
    (h2 : b + c = 40)
    (h3 : c + a = 45) :
    a + b + c = 60 := 
  by sorry

end sum_of_three_numbers_l34_34428


namespace average_speed_second_day_l34_34260

theorem average_speed_second_day
  (t v : ℤ)
  (h1 : 2 * t + 2 = 18)
  (h2 : (v + 5) * (t + 2) + v * t = 680) :
  v = 35 :=
by
  sorry

end average_speed_second_day_l34_34260


namespace lorie_total_bills_l34_34641

-- Definitions for the conditions
def initial_hundred_bills := 2
def hundred_to_fifty (bills : Nat) : Nat := bills * 2 / 100
def hundred_to_ten (bills : Nat) : Nat := (bills / 2) / 10
def hundred_to_five (bills : Nat) : Nat := (bills / 2) / 5

-- Statement of the problem
theorem lorie_total_bills : 
  let fifty_bills := hundred_to_fifty 100
  let ten_bills := hundred_to_ten 100
  let five_bills := hundred_to_five 100
  fifty_bills + ten_bills + five_bills = 2 + 5 + 10 :=
sorry

end lorie_total_bills_l34_34641


namespace probability_prime_sum_l34_34490

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end probability_prime_sum_l34_34490


namespace find_age_of_D_l34_34777

theorem find_age_of_D
(Eq1 : a + b + c + d = 108)
(Eq2 : a - b = 12)
(Eq3 : c - (a - 34) = 3 * (d - (a - 34)))
: d = 13 := 
sorry

end find_age_of_D_l34_34777


namespace sum_of_sequence_l34_34027

variable (S a b : ℝ)

theorem sum_of_sequence :
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 :=
by
  intros h1 h2 h3
  sorry

end sum_of_sequence_l34_34027


namespace total_number_of_shirts_l34_34678

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end total_number_of_shirts_l34_34678


namespace parabola_properties_l34_34395

theorem parabola_properties (m : ℝ) :
  (∀ P : ℝ × ℝ, P = (m, 1) ∧ (P.1 ^ 2 = 4 * P.2) →
    ((∃ y : ℝ, y = -1) ∧ (dist P (0, 1) = 2))) :=
by
  sorry

end parabola_properties_l34_34395


namespace prime_check_for_d1_prime_check_for_d2_l34_34050

-- Define d1 and d2
def d1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def d2 : ℕ := 9^4 - 9^2 + 1

-- Prime checking function
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Stating the conditions and proofs
theorem prime_check_for_d1 : ¬ is_prime d1 :=
by {
  -- condition: ten 8's in base nine is divisible by d1 (5905) is not used here directly
  sorry
}

theorem prime_check_for_d2 : is_prime d2 :=
by {
  -- condition: twelve 8's in base nine is divisible by d2 (6481) is not used here directly
  sorry
}

end prime_check_for_d1_prime_check_for_d2_l34_34050


namespace day_after_exponential_days_l34_34945

noncomputable def days_since_monday (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get! (n % 7)

theorem day_after_exponential_days :
  days_since_monday (2^20) = "Friday" :=
by
  sorry

end day_after_exponential_days_l34_34945


namespace move_point_right_l34_34501

theorem move_point_right (A B : ℤ) (hA : A = -3) (hAB : B = A + 4) : B = 1 :=
by {
  sorry
}

end move_point_right_l34_34501


namespace original_number_l34_34040

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l34_34040


namespace minimize_expression_l34_34235

theorem minimize_expression (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 = 18 :=
sorry

end minimize_expression_l34_34235


namespace merchant_product_quantities_l34_34946

theorem merchant_product_quantities
  (x p1 : ℝ)
  (h1 : 4000 = x * p1)
  (h2 : 8800 = 2 * x * (p1 + 4))
  (h3 : (8800 / (2 * x)) - (4000 / x) = 4):
  x = 100 ∧ 2 * x = 200 :=
by sorry

end merchant_product_quantities_l34_34946


namespace popsicle_sticks_l34_34552

theorem popsicle_sticks (total_sticks : ℕ) (gino_sticks : ℕ) (my_sticks : ℕ) 
  (h1 : total_sticks = 113) (h2 : gino_sticks = 63) (h3 : total_sticks = gino_sticks + my_sticks) : 
  my_sticks = 50 :=
  sorry

end popsicle_sticks_l34_34552


namespace solve_system_of_eq_l34_34100

noncomputable def system_of_eq (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2 * x) ∧
  z = y^3 * (3 - 2 * y) ∧
  x = z^3 * (3 - 2 * z)

theorem solve_system_of_eq (x y z : ℝ) :
  system_of_eq x y z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end solve_system_of_eq_l34_34100


namespace Toms_out_of_pocket_cost_l34_34355

theorem Toms_out_of_pocket_cost (visit_cost cast_cost insurance_percent : ℝ) 
  (h1 : visit_cost = 300) 
  (h2 : cast_cost = 200) 
  (h3 : insurance_percent = 0.6) : 
  (visit_cost + cast_cost) - ((visit_cost + cast_cost) * insurance_percent) = 200 :=
by
  sorry

end Toms_out_of_pocket_cost_l34_34355


namespace at_least_one_greater_than_one_l34_34957

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

end at_least_one_greater_than_one_l34_34957


namespace waiter_customers_l34_34826

theorem waiter_customers
    (initial_tables : ℝ)
    (left_tables : ℝ)
    (customers_per_table : ℝ)
    (remaining_tables : ℝ) 
    (total_customers : ℝ) 
    (h1 : initial_tables = 44.0)
    (h2 : left_tables = 12.0)
    (h3 : customers_per_table = 8.0)
    (remaining_tables_def : remaining_tables = initial_tables - left_tables)
    (total_customers_def : total_customers = remaining_tables * customers_per_table) :
    total_customers = 256.0 :=
by
  sorry

end waiter_customers_l34_34826


namespace remainder_product_l34_34515

theorem remainder_product (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := 
by 
  sorry

end remainder_product_l34_34515


namespace how_many_bigger_panda_bears_l34_34805

-- Definitions for the conditions
def four_small_panda_bears_eat_daily : ℕ := 25
def one_small_panda_bear_eats_daily : ℚ := 25 / 4
def each_bigger_panda_bear_eats_daily : ℚ := 40
def total_bamboo_eaten_weekly : ℕ := 2100
def total_bamboo_eaten_daily : ℚ := 2100 / 7

-- The theorem statement to prove
theorem how_many_bigger_panda_bears :
  ∃ B : ℚ, one_small_panda_bear_eats_daily * 4 + each_bigger_panda_bear_eats_daily * B = total_bamboo_eaten_daily := by
  sorry

end how_many_bigger_panda_bears_l34_34805


namespace vanessa_made_16_l34_34809

/-
Each chocolate bar in a box costs $4.
There are 11 bars in total in the box.
Vanessa sold all but 7 bars.
Prove that Vanessa made $16.
-/

def cost_per_bar : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost_per_bar

theorem vanessa_made_16 : money_made = 16 :=
by
  sorry

end vanessa_made_16_l34_34809


namespace proportional_function_range_l34_34980

theorem proportional_function_range (m : ℝ) (h : ∀ x : ℝ, (x < 0 → (1 - m) * x > 0) ∧ (x > 0 → (1 - m) * x < 0)) : m > 1 :=
by sorry

end proportional_function_range_l34_34980


namespace evaluate_expression_simplified_l34_34607

theorem evaluate_expression_simplified (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 3) ^ 2 + (x + 2) * (x - 2) - x * (x + 6) = 7 := by
  rw [h]
  sorry

end evaluate_expression_simplified_l34_34607


namespace find_multiple_of_q_l34_34255

variable (p q m : ℚ)

theorem find_multiple_of_q (h1 : p / q = 3 / 4) (h2 : 3 * p + m * q = 6.25) :
  m = 4 :=
sorry

end find_multiple_of_q_l34_34255


namespace sum_of_cubes_pattern_l34_34385

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 = 3^2) ->
  (1^3 + 2^3 + 3^3 = 6^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  intros h1 h2 h3
  -- Proof follows here
  sorry

end sum_of_cubes_pattern_l34_34385


namespace sum_first_4_terms_l34_34477

-- Define the sequence and its properties
def a (n : ℕ) : ℝ := sorry   -- The actual definition will be derived based on n, a_1, and q
def S (n : ℕ) : ℝ := sorry   -- The sum of the first n terms, also will be derived

-- Define the initial sequence properties based on the given conditions
axiom h1 : 0 < a 1  -- The sequence is positive
axiom h2 : a 4 * a 6 = 1 / 4
axiom h3 : a 7 = 1 / 8

-- The goal is to prove the sum of the first 4 terms equals 15
theorem sum_first_4_terms : S 4 = 15 := by
  sorry

end sum_first_4_terms_l34_34477


namespace bromine_atoms_in_compound_l34_34115

theorem bromine_atoms_in_compound
  (atomic_weight_H : ℕ := 1)
  (atomic_weight_Br : ℕ := 80)
  (atomic_weight_O : ℕ := 16)
  (total_molecular_weight : ℕ := 129) :
  ∃ (n : ℕ), total_molecular_weight = atomic_weight_H + n * atomic_weight_Br + 3 * atomic_weight_O ∧ n = 1 := 
by
  sorry

end bromine_atoms_in_compound_l34_34115


namespace range_of_a_l34_34419

theorem range_of_a (h : ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) : a > -1 :=
sorry

end range_of_a_l34_34419


namespace max_points_on_four_coplanar_circles_l34_34966

noncomputable def max_points_on_circles (num_circles : ℕ) (max_intersections : ℕ) : ℕ :=
num_circles * max_intersections

theorem max_points_on_four_coplanar_circles :
  max_points_on_circles 4 2 = 8 := 
sorry

end max_points_on_four_coplanar_circles_l34_34966


namespace least_of_consecutive_odds_l34_34023

noncomputable def average_of_consecutive_odds (n : ℕ) (start : ℤ) : ℤ :=
start + (2 * (n - 1))

theorem least_of_consecutive_odds
    (n : ℕ)
    (mean : ℤ)
    (h : n = 30 ∧ mean = 526) : 
    average_of_consecutive_odds 1 (mean * 2 - (n - 1)) = 497 :=
by
  sorry

end least_of_consecutive_odds_l34_34023


namespace marthas_bedroom_size_l34_34069

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l34_34069


namespace competition_participants_l34_34178

theorem competition_participants (n : ℕ) :
    (100 < n ∧ n < 200) ∧
    (n % 4 = 2) ∧
    (n % 5 = 2) ∧
    (n % 6 = 2)
    → (n = 122 ∨ n = 182) :=
by
  intro h
  sorry

end competition_participants_l34_34178


namespace cricketer_average_score_l34_34219

theorem cricketer_average_score
  (avg1 : ℕ)
  (matches1 : ℕ)
  (avg2 : ℕ)
  (matches2 : ℕ)
  (total_matches : ℕ)
  (total_avg : ℕ)
  (h1 : avg1 = 20)
  (h2 : matches1 = 2)
  (h3 : avg2 = 30)
  (h4 : matches2 = 3)
  (h5 : total_matches = 5)
  (h6 : total_avg = 26)
  (h_total_runs : total_avg * total_matches = avg1 * matches1 + avg2 * matches2) :
  total_avg = 26 := 
sorry

end cricketer_average_score_l34_34219


namespace combined_height_after_1_year_l34_34774

def initial_heights : ℕ := 200 + 150 + 250
def spring_and_summer_growth_A : ℕ := (6 * 4 / 2) * 50
def spring_and_summer_growth_B : ℕ := (6 * 4 / 3) * 70
def spring_and_summer_growth_C : ℕ := (6 * 4 / 4) * 90
def autumn_and_winter_growth_A : ℕ := (6 * 4 / 2) * 25
def autumn_and_winter_growth_B : ℕ := (6 * 4 / 3) * 35
def autumn_and_winter_growth_C : ℕ := (6 * 4 / 4) * 45

def total_growth_A : ℕ := spring_and_summer_growth_A + autumn_and_winter_growth_A
def total_growth_B : ℕ := spring_and_summer_growth_B + autumn_and_winter_growth_B
def total_growth_C : ℕ := spring_and_summer_growth_C + autumn_and_winter_growth_C

def total_growth : ℕ := total_growth_A + total_growth_B + total_growth_C

def combined_height : ℕ := initial_heights + total_growth

theorem combined_height_after_1_year : combined_height = 3150 := by
  sorry

end combined_height_after_1_year_l34_34774


namespace triangle_isosceles_or_right_l34_34108

theorem triangle_isosceles_or_right (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_side_constraint : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_condition: a^2 * c^2 - b^2 * c^2 = a^4 - b^4) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
by {
  sorry
}

end triangle_isosceles_or_right_l34_34108


namespace gab_score_ratio_l34_34853

theorem gab_score_ratio (S G C O : ℕ) (h1 : S = 20) (h2 : C = 2 * G) (h3 : O = 85) (h4 : S + G + C = O + 55) :
  G / S = 2 := 
by 
  sorry

end gab_score_ratio_l34_34853


namespace tara_had_more_l34_34190

theorem tara_had_more (M T X : ℕ) (h1 : T = 15) (h2 : M + T = 26) (h3 : T = M + X) : X = 4 :=
by 
  sorry

end tara_had_more_l34_34190


namespace number_of_4_letter_words_with_vowel_l34_34403

def is_vowel (c : Char) : Bool :=
c = 'A' ∨ c = 'E'

def count_4letter_words_with_vowels : Nat :=
  let total_words := 5^4
  let words_without_vowels := 3^4
  total_words - words_without_vowels

theorem number_of_4_letter_words_with_vowel :
  count_4letter_words_with_vowels = 544 :=
by
  -- proof goes here
  sorry

end number_of_4_letter_words_with_vowel_l34_34403


namespace initial_pounds_of_coffee_l34_34921

variable (x : ℝ) (h1 : 0.25 * x = d₀) (h2 : 0.60 * 100 = d₁) 
          (h3 : (d₀ + d₁) / (x + 100) = 0.32)

theorem initial_pounds_of_coffee (d₀ d₁ : ℝ) : 
  x = 400 :=
by
  -- Given conditions
  have h1 : d₀ = 0.25 * x := sorry
  have h2 : d₁ = 0.60 * 100 := sorry
  have h3 : 0.32 = (d₀ + d₁) / (x + 100) := sorry
  
  -- Additional steps to solve for x
  sorry

end initial_pounds_of_coffee_l34_34921


namespace p_sufficient_but_not_necessary_for_q_l34_34130

-- Definitions corresponding to conditions
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Theorem stating the relationship between p and q
theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l34_34130


namespace root_difference_l34_34273

theorem root_difference (p : ℝ) (r s : ℝ) :
  (r + s = p) ∧ (r * s = (p^2 - 1) / 4) ∧ (r ≥ s) → r - s = 1 :=
by
  intro h
  sorry

end root_difference_l34_34273


namespace inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l34_34738

variable (a : ℝ) (x : ℝ)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Part (1)
theorem inequality_solution (a : ℝ) (h1 : 0 < a ∧ a < 1) : (0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) → (f x a ≤ 1) :=
sorry

theorem inequality_solution_b (a : ℝ) (h2 : a ≥ 1) : (0 ≤ x) → (f x a ≤ 1) :=
sorry

-- Part (2)
theorem monotonic_increasing (a : ℝ) (h3 : a ≤ 0) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≤ f x2 a :=
sorry

theorem monotonic_decreasing (a : ℝ) (h4 : a ≥ 1) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≥ f x2 a :=
sorry

end inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l34_34738


namespace find_y_l34_34611

theorem find_y (AB BC : ℕ) (y x : ℕ) 
  (h1 : AB = 3 * y)
  (h2 : BC = 2 * x)
  (h3 : AB * BC = 2400) 
  (h4 : AB * BC = 6 * x * y) :
  y = 20 := by
  sorry

end find_y_l34_34611


namespace helium_balloon_height_l34_34815

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end helium_balloon_height_l34_34815


namespace height_on_hypotenuse_of_right_triangle_l34_34017

theorem height_on_hypotenuse_of_right_triangle (a b : ℝ) (h_a : a = 2) (h_b : b = 3) :
  ∃ h : ℝ, h = (6 * Real.sqrt 13) / 13 :=
by
  sorry

end height_on_hypotenuse_of_right_triangle_l34_34017


namespace Ganesh_avg_speed_l34_34647

theorem Ganesh_avg_speed (D : ℝ) : 
  (∃ (V : ℝ), (39.6 = (2 * D) / ((D / 44) + (D / V))) ∧ V = 36) :=
by
  sorry

end Ganesh_avg_speed_l34_34647


namespace probability_A_more_than_B_sum_m_n_l34_34533

noncomputable def prob_A_more_than_B : ℚ :=
  0.6 + 0.4 * (1 / 2) * (1 - (63 / 512))

theorem probability_A_more_than_B : prob_A_more_than_B = 779 / 1024 := sorry

theorem sum_m_n : 779 + 1024 = 1803 := sorry

end probability_A_more_than_B_sum_m_n_l34_34533


namespace find_number_l34_34215

theorem find_number (x : ℝ) (h : (x - 5) / 3 = 4) : x = 17 :=
by {
  sorry
}

end find_number_l34_34215


namespace power_identity_l34_34093

theorem power_identity (x y a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) : 10^(3*x + 2*y) = a^3 * b^2 := 
by 
  sorry

end power_identity_l34_34093


namespace number_of_pigs_l34_34479

theorem number_of_pigs (daily_feed_per_pig : ℕ) (weekly_feed_total : ℕ) (days_per_week : ℕ)
  (h1 : daily_feed_per_pig = 10) (h2 : weekly_feed_total = 140) (h3 : days_per_week = 7) : 
  (weekly_feed_total / days_per_week) / daily_feed_per_pig = 2 := by
  sorry

end number_of_pigs_l34_34479


namespace probability_of_winning_pair_l34_34981

-- Conditions: Define the deck composition and the winning pair.
inductive Color
| Red
| Green
| Blue

inductive Label
| A
| B
| C

structure Card :=
(color : Color)
(label : Label)

def deck : List Card :=
  [ {color := Color.Red, label := Label.A},
    {color := Color.Red, label := Label.B},
    {color := Color.Red, label := Label.C},
    {color := Color.Green, label := Label.A},
    {color := Color.Green, label := Label.B},
    {color := Color.Green, label := Label.C},
    {color := Color.Blue, label := Label.A},
    {color := Color.Blue, label := Label.B},
    {color := Color.Blue, label := Label.C} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

-- Question: Prove the probability of drawing a winning pair.
theorem probability_of_winning_pair :
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2 ∧ is_winning_pair c1 c2) →
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2) →
  (9 + 9) / 36 = 1 / 2 :=
sorry

end probability_of_winning_pair_l34_34981


namespace ralph_tv_hours_l34_34735

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l34_34735


namespace james_beats_per_week_l34_34170

def beats_per_minute := 200
def hours_per_day := 2
def days_per_week := 7

def beats_per_week (beats_per_minute: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  (beats_per_minute * hours_per_day * 60) * days_per_week

theorem james_beats_per_week : beats_per_week beats_per_minute hours_per_day days_per_week = 168000 := by
  sorry

end james_beats_per_week_l34_34170


namespace prime_gt_three_square_mod_twelve_l34_34979

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l34_34979


namespace evaluate_polynomial_at_three_l34_34113

def polynomial (x : ℕ) : ℕ :=
  x^6 + 2 * x^5 + 4 * x^3 + 5 * x^2 + 6 * x + 12

theorem evaluate_polynomial_at_three :
  polynomial 3 = 588 :=
by
  sorry

end evaluate_polynomial_at_three_l34_34113


namespace sum_A_B_C_l34_34975

noncomputable def number_B (A : ℕ) : ℕ := (A * 5) / 2
noncomputable def number_C (B : ℕ) : ℕ := (B * 7) / 4

theorem sum_A_B_C (A B C : ℕ) (h1 : A = 16) (h2 : A * 5 = B * 2) (h3 : B * 7 = C * 4) :
  A + B + C = 126 :=
by
  sorry

end sum_A_B_C_l34_34975


namespace range_of_a_l34_34728

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l34_34728


namespace average_snowfall_dec_1861_l34_34693

theorem average_snowfall_dec_1861 (snowfall : ℕ) (days_in_dec : ℕ) (hours_in_day : ℕ) 
  (time_period : ℕ) (Avg_inch_per_hour : ℚ) : 
  snowfall = 492 ∧ days_in_dec = 31 ∧ hours_in_day = 24 ∧ time_period = days_in_dec * hours_in_day ∧ 
  Avg_inch_per_hour = snowfall / time_period → 
  Avg_inch_per_hour = 492 / (31 * 24) :=
by sorry

end average_snowfall_dec_1861_l34_34693


namespace probability_at_least_one_alarm_on_time_l34_34887

noncomputable def P_alarm_A_on : ℝ := 0.80
noncomputable def P_alarm_B_on : ℝ := 0.90

theorem probability_at_least_one_alarm_on_time :
  (1 - (1 - P_alarm_A_on) * (1 - P_alarm_B_on)) = 0.98 :=
by
  sorry

end probability_at_least_one_alarm_on_time_l34_34887


namespace solve_abs_inequality_l34_34090

theorem solve_abs_inequality (x : ℝ) : abs ((7 - x) / 4) < 3 → 2 < x ∧ x < 19 :=
by 
  sorry

end solve_abs_inequality_l34_34090


namespace find_third_number_l34_34446

-- Given conditions
variable (A B C : ℕ)
variable (LCM HCF : ℕ)
variable (h1 : A = 36)
variable (h2 : B = 44)
variable (h3 : LCM = 792)
variable (h4 : HCF = 12)
variable (h5 : A * B * C = LCM * HCF)

-- Desired proof
theorem find_third_number : C = 6 :=
by
  sorry

end find_third_number_l34_34446


namespace first_prime_year_with_digit_sum_8_l34_34145

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem first_prime_year_with_digit_sum_8 :
  ∃ y : ℕ, y > 2015 ∧ sum_of_digits y = 8 ∧ is_prime y ∧
  ∀ z : ℕ, z > 2015 ∧ sum_of_digits z = 8 ∧ is_prime z → y ≤ z :=
sorry

end first_prime_year_with_digit_sum_8_l34_34145


namespace contradiction_method_assumption_l34_34496

-- Definitions for three consecutive positive integers
variables {a b c : ℕ}

-- Definitions for the proposition and its negation
def consecutive_integers (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1
def at_least_one_divisible_by_2 (a b c : ℕ) : Prop := a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0
def all_not_divisible_by_2 (a b c : ℕ) : Prop := a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem contradiction_method_assumption (a b c : ℕ) (h : consecutive_integers a b c) :
  (¬ at_least_one_divisible_by_2 a b c) ↔ all_not_divisible_by_2 a b c :=
by sorry

end contradiction_method_assumption_l34_34496


namespace remainder_97_pow_50_mod_100_l34_34356

theorem remainder_97_pow_50_mod_100 :
  (97 ^ 50) % 100 = 49 := 
by
  sorry

end remainder_97_pow_50_mod_100_l34_34356


namespace smallest_x_l34_34955

theorem smallest_x (x: ℕ) (hx: x > 0) (h: 11^2021 ∣ 5^(3*x) - 3^(4*x)) : 
  x = 11^2020 := sorry

end smallest_x_l34_34955


namespace least_positive_value_of_cubic_eq_l34_34885

theorem least_positive_value_of_cubic_eq (x y z w : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hz : Nat.Prime z) (hw : Nat.Prime w) 
  (sum_lt_50 : x + y + z + w < 50) : 
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 :=
by
  sorry

end least_positive_value_of_cubic_eq_l34_34885


namespace total_chips_is_90_l34_34760

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end total_chips_is_90_l34_34760


namespace find_S_l34_34080

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S) ^ 2) (h2 : S > 0) :
  S = 333332 := 
sorry

end find_S_l34_34080


namespace problem1_problem2_l34_34448

variable (a b : ℝ)

theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  1/a + 1/(b+1) ≥ 4/5 := by
  sorry

theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  4/(a*b) + a/b ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end problem1_problem2_l34_34448


namespace original_number_is_1200_l34_34875

theorem original_number_is_1200 (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by
  sorry

end original_number_is_1200_l34_34875


namespace cone_volume_and_surface_area_l34_34085

noncomputable def cone_volume (slant_height height : ℝ) : ℝ := 
  1 / 3 * Real.pi * (Real.sqrt (slant_height^2 - height^2))^2 * height

noncomputable def cone_surface_area (slant_height height : ℝ) : ℝ :=
  Real.pi * (Real.sqrt (slant_height^2 - height^2)) * (Real.sqrt (slant_height^2 - height^2) + slant_height)

theorem cone_volume_and_surface_area :
  (cone_volume 15 9 = 432 * Real.pi) ∧ (cone_surface_area 15 9 = 324 * Real.pi) :=
by
  sorry

end cone_volume_and_surface_area_l34_34085


namespace prove_a_eq_b_l34_34677

theorem prove_a_eq_b 
    (a b : ℕ) 
    (h_pos : a > 0 ∧ b > 0) 
    (h_multiple : ∃ k : ℤ, a^2 + a * b + 1 = k * (b^2 + b * a + 1)) : 
    a = b := 
sorry

end prove_a_eq_b_l34_34677


namespace B_takes_6_days_to_complete_work_alone_l34_34008

theorem B_takes_6_days_to_complete_work_alone 
    (work_duration_A : ℕ) 
    (work_payment : ℚ)
    (work_days_with_C : ℕ) 
    (payment_C : ℚ) 
    (combined_work_rate_A_B_C : ℚ)
    (amount_to_be_shared_A_B : ℚ) 
    (combined_daily_earning_A_B : ℚ) :
  work_duration_A = 6 ∧
  work_payment = 3360 ∧ 
  work_days_with_C = 3 ∧ 
  payment_C = 420.00000000000017 ∧ 
  combined_work_rate_A_B_C = 1 / 3 ∧ 
  amount_to_be_shared_A_B = 2940 ∧ 
  combined_daily_earning_A_B = 980 → 
  work_duration_A = 6 ∧
  (∃ (work_duration_B : ℕ), work_duration_B = 6) :=
by 
  sorry

end B_takes_6_days_to_complete_work_alone_l34_34008


namespace min_sum_a_b_l34_34863

theorem min_sum_a_b (a b : ℝ) (h1 : 4 * a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  a + b ≥ 16 :=
sorry

end min_sum_a_b_l34_34863


namespace find_root_and_m_l34_34912

theorem find_root_and_m {x : ℝ} {m : ℝ} (h : ∃ x1 x2 : ℝ, (x1 = 1) ∧ (x1 + x2 = -m) ∧ (x1 * x2 = 3)) :
  ∃ x2 : ℝ, (x2 = 3) ∧ (m = -4) :=
by
  obtain ⟨x1, x2, h1, h_sum, h_product⟩ := h
  have hx1 : x1 = 1 := h1
  rw [hx1] at h_product
  have hx2 : x2 = 3 := by linarith [h_product]
  have hm : m = -4 := by
    rw [hx1, hx2] at h_sum
    linarith
  exact ⟨x2, hx2, hm⟩

end find_root_and_m_l34_34912


namespace closest_ratio_adults_children_l34_34662

theorem closest_ratio_adults_children (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 50) 
  (h3 : c ≥ 20) : a = 50 ∧ c = 50 :=
by {
  sorry
}

end closest_ratio_adults_children_l34_34662


namespace number_of_mixed_vegetable_plates_l34_34991

theorem number_of_mixed_vegetable_plates :
  ∃ n : ℕ, n * 70 = 1051 - (16 * 6 + 5 * 45 + 6 * 40) ∧ n = 7 :=
by
  sorry

end number_of_mixed_vegetable_plates_l34_34991


namespace Winnie_lollipops_remain_l34_34317

theorem Winnie_lollipops_remain :
  let cherry_lollipops := 45
  let wintergreen_lollipops := 116
  let grape_lollipops := 4
  let shrimp_cocktail_lollipops := 229
  let total_lollipops := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops
  let friends := 11
  total_lollipops % friends = 9 :=
by
  sorry

end Winnie_lollipops_remain_l34_34317


namespace ceil_sqrt_225_l34_34542

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l34_34542


namespace inequalities_always_hold_l34_34883

theorem inequalities_always_hold (x y a b : ℝ) (hxy : x > y) (hab : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) :=
by
  sorry

end inequalities_always_hold_l34_34883


namespace cos_180_degree_l34_34942

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l34_34942


namespace sum_of_coefficients_of_expansion_l34_34982

-- Define a predicate for a term being constant
def is_constant_term (n : ℕ) (term : ℚ) : Prop := 
  term = 0

-- Define the sum of coefficients computation
noncomputable def sum_of_coefficients (n : ℕ) : ℚ := 
  (1 - 3)^n

-- The main statement of the problem in Lean
theorem sum_of_coefficients_of_expansion {n : ℕ} 
  (h : is_constant_term n (2 * n - 10)) : 
  sum_of_coefficients 5 = -32 := 
sorry

end sum_of_coefficients_of_expansion_l34_34982


namespace loss_percentage_second_venture_l34_34684

theorem loss_percentage_second_venture 
  (investment_total : ℝ)
  (investment_each : ℝ)
  (profit_percentage_first_venture : ℝ)
  (total_return_percentage : ℝ)
  (L : ℝ) 
  (H1 : investment_total = 25000) 
  (H2 : investment_each = 16250)
  (H3 : profit_percentage_first_venture = 0.15)
  (H4 : total_return_percentage = 0.08)
  (H5 : (investment_total * total_return_percentage) = ((investment_each * profit_percentage_first_venture) - (investment_each * L))) :
  L = 0.0269 := 
by
  sorry

end loss_percentage_second_venture_l34_34684


namespace evaluate_at_two_l34_34521

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem evaluate_at_two : f 2 = 15 :=
by
  sorry

end evaluate_at_two_l34_34521


namespace arithmetic_progression_probability_l34_34523

def is_arithmetic_progression (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

noncomputable def probability_arithmetic_progression_diff_two : ℚ :=
  have total_outcomes : ℚ := 6 * 6 * 6
  have favorable_outcomes : ℚ := 12
  favorable_outcomes / total_outcomes

theorem arithmetic_progression_probability (d : ℕ) (h : d = 2) :
  probability_arithmetic_progression_diff_two = 1 / 18 :=
by 
  sorry

end arithmetic_progression_probability_l34_34523


namespace evaluate_expression_at_neg3_l34_34425

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l34_34425


namespace cost_per_remaining_ticket_is_seven_l34_34092

def total_tickets : ℕ := 29
def nine_dollar_tickets : ℕ := 11
def total_cost : ℕ := 225
def nine_dollar_ticket_cost : ℕ := 9
def remaining_tickets : ℕ := total_tickets - nine_dollar_tickets

theorem cost_per_remaining_ticket_is_seven :
  (total_cost - nine_dollar_tickets * nine_dollar_ticket_cost) / remaining_tickets = 7 :=
  sorry

end cost_per_remaining_ticket_is_seven_l34_34092


namespace finite_solutions_to_equation_l34_34651

theorem finite_solutions_to_equation :
  ∃ (n : ℕ), ∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) = 1 / 1983) → 
  (a ≤ n ∧ b ≤ n ∧ c ≤ n) :=
sorry

end finite_solutions_to_equation_l34_34651


namespace tom_is_15_l34_34225

theorem tom_is_15 (T M : ℕ) (h1 : T + M = 21) (h2 : T + 3 = 2 * (M + 3)) : T = 15 :=
by {
  sorry
}

end tom_is_15_l34_34225


namespace smallest_number_of_people_l34_34614

theorem smallest_number_of_people (N : ℕ) :
  (∃ (N : ℕ), ∀ seats : ℕ, seats = 80 → N ≤ 80 → ∀ n : ℕ, n > N → (∃ m : ℕ, (m < N) ∧ ((seats + m) % 80 < seats))) → N = 20 :=
by
  sorry

end smallest_number_of_people_l34_34614


namespace smallest_positive_integer_n_l34_34794

theorem smallest_positive_integer_n (n : ℕ) (h : 527 * n ≡ 1083 * n [MOD 30]) : n = 2 :=
sorry

end smallest_positive_integer_n_l34_34794


namespace algebraic_expression_value_l34_34427

-- Define the conditions
variables (x y : ℝ)
-- Condition 1: x - y = 5
def cond1 : Prop := x - y = 5
-- Condition 2: xy = -3
def cond2 : Prop := x * y = -3

-- Define the statement to be proved
theorem algebraic_expression_value :
  cond1 x y → cond2 x y → x^2 * y - x * y^2 = -15 :=
by
  intros h1 h2
  sorry

end algebraic_expression_value_l34_34427


namespace find_f_g_2_l34_34930

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 - 6

theorem find_f_g_2 : f (g 2) = 1 := 
  by
  -- Proof goes here
  sorry

end find_f_g_2_l34_34930


namespace value_preserving_interval_of_g_l34_34928

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  x + m - Real.log x

theorem value_preserving_interval_of_g
  (m : ℝ)
  (h_increasing : ∀ x, x ∈ Set.Ici 2 → 1 - 1 / x > 0)
  (h_range : ∀ y, y ∈ Set.Ici 2): 
  (2 + m - Real.log 2 = 2) → 
  m = Real.log 2 :=
by 
  sorry

end value_preserving_interval_of_g_l34_34928


namespace solve_cos_theta_l34_34776

def cos_theta_proof (v1 v2 : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2)
  let norm_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  let cos_theta := dot_product / (norm_v1 * norm_v2)
  cos_theta = 43 / Real.sqrt 2173

theorem solve_cos_theta :
  cos_theta_proof (4, 5) (2, 7) (43 / Real.sqrt 2173) :=
by
  sorry

end solve_cos_theta_l34_34776


namespace correct_option_is_C_l34_34816

namespace ExponentProof

-- Definitions of conditions
def optionA (a : ℝ) : Prop := a^3 * a^4 = a^12
def optionB (a : ℝ) : Prop := a^3 + a^4 = a^7
def optionC (a : ℝ) : Prop := a^5 / a^3 = a^2
def optionD (a : ℝ) : Prop := (-2 * a)^3 = -6 * a^3

-- Proof problem stating that optionC is the only correct one
theorem correct_option_is_C : ∀ (a : ℝ), ¬ optionA a ∧ ¬ optionB a ∧ optionC a ∧ ¬ optionD a :=
by
  intro a
  sorry

end ExponentProof

end correct_option_is_C_l34_34816


namespace num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l34_34010

def total_students : ℕ := 800

def percentage_blue_shirts : ℕ := 45
def percentage_red_shirts : ℕ := 23
def percentage_green_shirts : ℕ := 15

def percentage_black_pants : ℕ := 30
def percentage_khaki_pants : ℕ := 25
def percentage_jeans_pants : ℕ := 10

def percentage_white_shoes : ℕ := 40
def percentage_black_shoes : ℕ := 20
def percentage_brown_shoes : ℕ := 15

def students_other_color_shirts : ℕ :=
  total_students * (100 - (percentage_blue_shirts + percentage_red_shirts + percentage_green_shirts)) / 100

def students_other_types_pants : ℕ :=
  total_students * (100 - (percentage_black_pants + percentage_khaki_pants + percentage_jeans_pants)) / 100

def students_other_color_shoes : ℕ :=
  total_students * (100 - (percentage_white_shoes + percentage_black_shoes + percentage_brown_shoes)) / 100

theorem num_students_other_color_shirts : students_other_color_shirts = 136 := by
  sorry

theorem num_students_other_types_pants : students_other_types_pants = 280 := by
  sorry

theorem num_students_other_color_shoes : students_other_color_shoes = 200 := by
  sorry

end num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l34_34010


namespace woman_weaves_amount_on_20th_day_l34_34900

theorem woman_weaves_amount_on_20th_day
  (a d : ℚ)
  (a2 : a + d = 17) -- second-day weaving in inches
  (S15 : 15 * a + 105 * d = 720) -- total for the first 15 days in inches
  : a + 19 * d = 108 := -- weaving on the twentieth day in inches (9 feet)
by
  sorry

end woman_weaves_amount_on_20th_day_l34_34900


namespace group_size_l34_34449

theorem group_size (n : ℕ) (T : ℕ) (h1 : T = 14 * n) (h2 : T + 32 = 16 * (n + 1)) : n = 8 :=
by
  -- We skip the proof steps
  sorry

end group_size_l34_34449


namespace circle_radius_zero_l34_34829

-- Theorem statement
theorem circle_radius_zero :
  ∀ (x y : ℝ), 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → 
  ∃ (c : ℝ) (r : ℝ), r = 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2 :=
by sorry

end circle_radius_zero_l34_34829


namespace find_number_l34_34483

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l34_34483


namespace num_positive_integers_le_500_l34_34196

-- Define a predicate to state that a number is a perfect square
def is_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- Define the main theorem
theorem num_positive_integers_le_500 (n : ℕ) :
  (∃ (ns : Finset ℕ), (∀ x ∈ ns, x ≤ 500 ∧ is_square (21 * x)) ∧ ns.card = 4) :=
by
  sorry

end num_positive_integers_le_500_l34_34196


namespace radius_of_circle_zero_l34_34074

theorem radius_of_circle_zero (x y : ℝ) :
    (x^2 + 4*x + y^2 - 2*y + 5 = 0) → 0 = 0 :=
by
  sorry

end radius_of_circle_zero_l34_34074


namespace total_employees_l34_34098

-- Definitions based on the conditions:
variables (N S : ℕ)
axiom condition1 : 75 % 100 * S = 75 / 100 * S
axiom condition2 : 65 % 100 * S = 65 / 100 * S
axiom condition3 : N - S = 40
axiom condition4 : 5 % 6 * N = 5 / 6 * N

-- The statement to be proven:
theorem total_employees (N S : ℕ)
    (h1 : 75 % 100 * S = 75 / 100 * S)
    (h2 : 65 % 100 * S = 65 / 100 * S)
    (h3 : N - S = 40)
    (h4 : 5 % 6 * N = 5 / 6 * N)
    : N = 240 :=
sorry

end total_employees_l34_34098


namespace expected_absolute_deviation_greater_in_10_tosses_l34_34062

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l34_34062


namespace abe_age_equation_l34_34103

theorem abe_age_equation (a : ℕ) (x : ℕ) (h1 : a = 19) (h2 : a + (a - x) = 31) : x = 7 :=
by
  sorry

end abe_age_equation_l34_34103


namespace linear_equation_value_l34_34767

-- Define the conditions of the equation
def equation_is_linear (m : ℝ) : Prop :=
  |m| = 1 ∧ m - 1 ≠ 0

-- Prove the equivalence statement
theorem linear_equation_value (m : ℝ) (h : equation_is_linear m) : m = -1 := 
sorry

end linear_equation_value_l34_34767


namespace quadrilateral_area_l34_34287

theorem quadrilateral_area (a b c d : ℝ) (horizontally_vertically_apart : a = b + 1 ∧ b = c + 1 ∧ c = d + 1 ∧ d = a + 1) : 
  area_of_quadrilateral = 6 :=
sorry

end quadrilateral_area_l34_34287


namespace books_borrowed_l34_34155

theorem books_borrowed (initial_books : ℕ) (additional_books : ℕ) (remaining_books : ℕ) : 
  initial_books = 300 → 
  additional_books = 10 * 5 → 
  remaining_books = 210 → 
  initial_books + additional_books - remaining_books = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end books_borrowed_l34_34155


namespace least_additional_squares_needed_for_symmetry_l34_34691

-- Conditions
def grid_size : ℕ := 5
def initial_shaded_squares : List (ℕ × ℕ) := [(1, 5), (3, 3), (5, 1)]

-- Goal statement
theorem least_additional_squares_needed_for_symmetry
  (grid_size : ℕ)
  (initial_shaded_squares : List (ℕ × ℕ)) : 
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℕ), (x, y) ∈ initial_shaded_squares ∨ (grid_size - x + 1, y) ∈ initial_shaded_squares ∨ (x, grid_size - y + 1) ∈ initial_shaded_squares ∨ (grid_size - x + 1, grid_size - y + 1) ∈ initial_shaded_squares) :=
sorry

end least_additional_squares_needed_for_symmetry_l34_34691


namespace union_sets_l34_34610

open Set

def setM : Set ℝ := {x : ℝ | x^2 < x}
def setN : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem union_sets : setM ∪ setN = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end union_sets_l34_34610


namespace original_deck_card_count_l34_34095

variable (r b : ℕ)

theorem original_deck_card_count (h1 : r / (r + b) = 1 / 4) (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
by
  -- The proof goes here
  sorry

end original_deck_card_count_l34_34095


namespace timber_logging_years_l34_34871

theorem timber_logging_years 
  (V0 : ℝ) (r : ℝ) (V : ℝ) (t : ℝ)
  (hV0 : V0 = 100000)
  (hr : r = 0.08)
  (hV : V = 400000)
  (hformula : V = V0 * (1 + r)^t)
  : t = (Real.log 4 / Real.log 1.08) :=
by
  sorry

end timber_logging_years_l34_34871


namespace pencils_per_child_l34_34405

-- Define the conditions
def totalPencils : ℕ := 18
def numberOfChildren : ℕ := 9

-- The proof problem
theorem pencils_per_child : totalPencils / numberOfChildren = 2 := 
by
  sorry

end pencils_per_child_l34_34405


namespace global_chess_tournament_total_games_global_chess_tournament_player_wins_l34_34154

theorem global_chess_tournament_total_games (num_players : ℕ) (h200 : num_players = 200) :
  (num_players * (num_players - 1)) / 2 = 19900 := by
  sorry

theorem global_chess_tournament_player_wins (num_players losses : ℕ) 
  (h200 : num_players = 200) (h30 : losses = 30) :
  (num_players - 1) - losses = 169 := by
  sorry

end global_chess_tournament_total_games_global_chess_tournament_player_wins_l34_34154


namespace goose_eggs_l34_34859

theorem goose_eggs (E : ℝ) :
  (E / 2 * 3 / 4 * 2 / 5 + (1 / 3 * (E / 2)) * 2 / 3 * 3 / 4 + (1 / 6 * (E / 2 + E / 6)) * 1 / 2 * 2 / 3 = 150) →
  E = 375 :=
by
  sorry

end goose_eggs_l34_34859


namespace polar_equation_is_circle_of_radius_five_l34_34463

theorem polar_equation_is_circle_of_radius_five :
  ∀ θ : ℝ, (3 * Real.sin θ + 4 * Real.cos θ) ^ 2 = 25 :=
by
  sorry

end polar_equation_is_circle_of_radius_five_l34_34463


namespace gcd_a_b_eq_one_l34_34296

def a : ℕ := 123^2 + 235^2 + 347^2
def b : ℕ := 122^2 + 234^2 + 348^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_a_b_eq_one_l34_34296


namespace prob1_prob2_l34_34123

-- Problem 1
theorem prob1 (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4 :=
sorry

-- Problem 2
theorem prob2 (x y : ℝ) : (5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2 :=
sorry

end prob1_prob2_l34_34123


namespace speed_of_first_boy_l34_34283

theorem speed_of_first_boy (x : ℝ) (h1 : 7.5 > 0) (h2 : 16 > 0) (h3 : 32 > 0) (h4 : 32 = 16 * (x - 7.5)) : x = 9.5 :=
by
  sorry

end speed_of_first_boy_l34_34283


namespace proposition_D_l34_34330

variable {A B C : Set α} (h1 : ∀ a (ha : a ∈ A), ∃ b ∈ B, a = b)
variable {A B C : Set α} (h2 : ∀ c (hc : c ∈ C), ∃ b ∈ B, b = c) 

theorem proposition_D (A B C : Set α) (h : A ∩ B = B ∪ C) : C ⊆ B :=
by 
  sorry

end proposition_D_l34_34330


namespace find_length_CD_m_plus_n_l34_34147

noncomputable def lengthAB : ℝ := 7
noncomputable def lengthBD : ℝ := 11
noncomputable def lengthBC : ℝ := 9

axiom angle_BAD_ADC : Prop
axiom angle_ABD_BCD : Prop

theorem find_length_CD_m_plus_n :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 67) :=
sorry  -- Proof would be provided here

end find_length_CD_m_plus_n_l34_34147


namespace election_total_polled_votes_l34_34036

theorem election_total_polled_votes (V : ℝ) (invalid_votes : ℝ) (candidate_votes : ℝ) (margin : ℝ)
  (h1 : candidate_votes = 0.3 * V)
  (h2 : margin = 5000)
  (h3 : V = 0.3 * V + (0.3 * V + margin))
  (h4 : invalid_votes = 100) :
  V + invalid_votes = 12600 :=
by
  sorry

end election_total_polled_votes_l34_34036


namespace overall_loss_is_450_l34_34588

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450_l34_34588


namespace last_digit_inverse_power_two_l34_34457

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end last_digit_inverse_power_two_l34_34457


namespace range_of_x_l34_34771

theorem range_of_x (a : ℕ → ℝ) (x : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_condition : ∀ n, a (n + 1)^2 + a n^2 < (5 / 2) * a (n + 1) * a n)
  (h_a2 : a 2 = 3 / 2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) : 2 < x ∧ x < 3 := by
  sorry

end range_of_x_l34_34771


namespace a_pow_5_mod_11_l34_34135

theorem a_pow_5_mod_11 (a : ℕ) : (a^5) % 11 = 0 ∨ (a^5) % 11 = 1 ∨ (a^5) % 11 = 10 :=
sorry

end a_pow_5_mod_11_l34_34135


namespace max_value_m_l34_34204

theorem max_value_m (m n : ℕ) (h : 8 * m + 9 * n = m * n + 6) : m ≤ 75 := 
sorry

end max_value_m_l34_34204


namespace chord_length_intercepted_by_line_on_circle_l34_34415

theorem chord_length_intercepted_by_line_on_circle :
  ∀ (ρ θ : ℝ), (ρ = 4) →
  (ρ * Real.sin (θ + (Real.pi / 4)) = 2) →
  (4 * Real.sqrt (16 - (2 ^ 2)) = 4 * Real.sqrt 3) :=
by
  intros ρ θ hρ hline_eq
  sorry

end chord_length_intercepted_by_line_on_circle_l34_34415


namespace steven_needs_more_seeds_l34_34953

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l34_34953


namespace quadratic_roots_l34_34087

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end quadratic_roots_l34_34087


namespace geometric_sequence_ninth_tenth_term_sum_l34_34049

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^n

theorem geometric_sequence_ninth_tenth_term_sum (a₁ q : ℝ)
  (h1 : a₁ + a₁ * q = 2)
  (h5 : a₁ * q^4 + a₁ * q^5 = 4) :
  geometric_sequence a₁ q 8 + geometric_sequence a₁ q 9 = 8 :=
by
  sorry

end geometric_sequence_ninth_tenth_term_sum_l34_34049


namespace andres_possibilities_10_dollars_l34_34718

theorem andres_possibilities_10_dollars : 
  (∃ (num_1_coins num_2_coins num_5_bills : ℕ),
    num_1_coins + 2 * num_2_coins + 5 * num_5_bills = 10) → 
  ∃ (ways : ℕ), ways = 10 :=
by
  -- The proof can be provided here, but we'll use sorry to skip it in this template.
  sorry

end andres_possibilities_10_dollars_l34_34718


namespace greatest_prime_factor_of_144_l34_34124

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l34_34124


namespace pie_eating_contest_l34_34445

theorem pie_eating_contest :
  let a := 5 / 6
  let b := 7 / 8
  let c := 2 / 3
  let max_pie := max a (max b c)
  let min_pie := min a (min b c)
  max_pie - min_pie = 5 / 24 :=
by
  sorry

end pie_eating_contest_l34_34445


namespace smallest_y_absolute_value_equation_l34_34171

theorem smallest_y_absolute_value_equation :
  ∃ y : ℚ, (|5 * y - 9| = 55) ∧ y = -46 / 5 :=
by
  sorry

end smallest_y_absolute_value_equation_l34_34171


namespace math_problem_l34_34745

theorem math_problem
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) :=
  sorry

end math_problem_l34_34745


namespace line_passes_through_fixed_point_minimal_triangle_area_eq_line_l34_34492

-- Part (1)
theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ M : ℝ × ℝ, M = (-1, -2) ∧
    (∀ m : ℝ, (2 + m) * (-1) + (1 - 2 * m) * (-2) + (4 - 3 * m) = 0) := by
  sorry

-- Part (2)
theorem minimal_triangle_area_eq_line :
  ∃ k : ℝ, k = -2 ∧ 
    (∀ x y : ℝ, y = k * (x + 1) - 2 ↔ y = 2 * x + 4) := by
  sorry

end line_passes_through_fixed_point_minimal_triangle_area_eq_line_l34_34492


namespace statement_bug_travel_direction_l34_34635

/-
  Theorem statement: On a plane with a grid formed by regular hexagons of side length 1,
  if a bug traveled from node A to node B along the shortest path of 100 units,
  then the bug traveled exactly 50 units in one direction.
-/
theorem bug_travel_direction (side_length : ℝ) (total_distance : ℝ) 
  (hexagonal_grid : Π (x y : ℝ), Prop) (A B : ℝ × ℝ) 
  (shortest_path : ℝ) :
  side_length = 1 ∧ shortest_path = 100 →
  ∃ (directional_travel : ℝ), directional_travel = 50 :=
by
  sorry

end statement_bug_travel_direction_l34_34635


namespace remainder_of_power_is_41_l34_34430

theorem remainder_of_power_is_41 : 
  ∀ (n k : ℕ), n = 2019 → k = 2018 → (n^k) % 100 = 41 :=
  by 
    intros n k hn hk 
    rw [hn, hk] 
    exact sorry

end remainder_of_power_is_41_l34_34430


namespace susan_betsy_ratio_l34_34375

theorem susan_betsy_ratio (betsy_wins : ℕ) (helen_wins : ℕ) (susan_wins : ℕ) (total_wins : ℕ)
  (h1 : betsy_wins = 5)
  (h2 : helen_wins = 2 * betsy_wins)
  (h3 : betsy_wins + helen_wins + susan_wins = total_wins)
  (h4 : total_wins = 30) :
  susan_wins / betsy_wins = 3 := by
  sorry

end susan_betsy_ratio_l34_34375


namespace linear_function_equality_l34_34044

theorem linear_function_equality (f : ℝ → ℝ) (hf : ∀ x, f (3 * (f x)⁻¹ + 5) = f x)
  (hf1 : f 1 = 5) : f 2 = 3 :=
sorry

end linear_function_equality_l34_34044


namespace find_distance_l34_34101

-- Definitions of given conditions
def speed : ℝ := 65 -- km/hr
def time  : ℝ := 3  -- hr

-- Statement: The distance is 195 km given the speed and time.
theorem find_distance (speed : ℝ) (time : ℝ) : (speed * time = 195) :=
by
  sorry

end find_distance_l34_34101


namespace abs_diff_squares_105_95_l34_34920

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l34_34920


namespace find_n_l34_34399

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n_l34_34399


namespace bicycles_in_garage_l34_34292

theorem bicycles_in_garage 
  (B : ℕ) 
  (h1 : 4 * 3 = 12) 
  (h2 : 7 * 1 = 7) 
  (h3 : 2 * B + 12 + 7 = 25) : 
  B = 3 := 
by
  sorry

end bicycles_in_garage_l34_34292


namespace C_is_20_years_younger_l34_34716

variable (A B C : ℕ)

-- Conditions from the problem
axiom age_condition : A + B = B + C + 20

-- Theorem representing the proof problem
theorem C_is_20_years_younger : A = C + 20 := sorry

end C_is_20_years_younger_l34_34716


namespace triangle_side_split_l34_34765

theorem triangle_side_split
  (PQ QR PR : ℝ)  -- Triangle sides
  (PS SR : ℝ)     -- Segments of PR divided by angle bisector
  (h_ratio : PQ / QR = 3 / 4)
  (h_sum : PR = 15)
  (h_PS_SR : PS / SR = 3 / 4)
  (h_PR_split : PS + SR = PR) :
  SR = 60 / 7 :=
by
  sorry

end triangle_side_split_l34_34765


namespace rowing_upstream_distance_l34_34701

theorem rowing_upstream_distance (b s d : ℝ) (h_stream_speed : s = 5)
    (h_downstream_distance : 60 = (b + s) * 3)
    (h_upstream_time : d = (b - s) * 3) : 
    d = 30 := by
  have h_b : b = 15 := by
    linarith [h_downstream_distance, h_stream_speed]
  rw [h_b, h_stream_speed] at h_upstream_time
  linarith [h_upstream_time]

end rowing_upstream_distance_l34_34701


namespace part1_expression_for_f_part2_three_solutions_l34_34516

noncomputable def f1 (x : ℝ) := x^2

noncomputable def f2 (x : ℝ) := 8 / x

noncomputable def f (x : ℝ) := f1 x + f2 x

theorem part1_expression_for_f : ∀ x:ℝ, f x = x^2 + 8 / x := by
  sorry  -- This is where the proof would go

theorem part2_three_solutions (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, f x1 = f a ∧ f x2 = f a ∧ f x3 = f a ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 := by
  sorry  -- This is where the proof would go

end part1_expression_for_f_part2_three_solutions_l34_34516


namespace opposite_of_expression_l34_34455

theorem opposite_of_expression : 
  let expr := 1 - (3 : ℝ)^(1/3)
  (-1 + (3 : ℝ)^(1/3)) = (3 : ℝ)^(1/3) - 1 :=
by 
  let expr := 1 - (3 : ℝ)^(1/3)
  sorry

end opposite_of_expression_l34_34455


namespace ratio_condition_equivalence_l34_34489

variable (a b c d : ℝ)

theorem ratio_condition_equivalence
  (h : (2 * a + 3 * b) / (b + 2 * c) = (3 * c + 2 * d) / (d + 2 * a)) :
  2 * a = 3 * c ∨ 2 * a + 3 * b + d + 2 * c = 0 :=
by
  sorry

end ratio_condition_equivalence_l34_34489


namespace injective_g_restricted_to_interval_l34_34938

def g (x : ℝ) : ℝ := (x + 3) ^ 2 - 10

theorem injective_g_restricted_to_interval :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (-3) → x2 ∈ Set.Ici (-3) → g x1 = g x2 → x1 = x2) :=
sorry

end injective_g_restricted_to_interval_l34_34938


namespace computation_correct_l34_34003

theorem computation_correct : 12 * ((216 / 3) + (36 / 6) + (16 / 8) + 2) = 984 := 
by 
  sorry

end computation_correct_l34_34003


namespace sequences_get_arbitrarily_close_l34_34306

noncomputable def a_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^n
noncomputable def b_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^(n + 1)

theorem sequences_get_arbitrarily_close (n : ℕ) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b_n n - a_n n| < ε :=
sorry

end sequences_get_arbitrarily_close_l34_34306


namespace revenue_from_full_price_tickets_l34_34630

-- Let's define our variables and assumptions
variables (f h p: ℕ)

-- Total number of tickets sold
def total_tickets (f h: ℕ) : Prop := f + h = 200

-- Total revenue from tickets
def total_revenue (f h p: ℕ) : Prop := f * p + h * (p / 3) = 2500

-- Statement to prove the revenue from full-price tickets
theorem revenue_from_full_price_tickets (f h p: ℕ) (hf: total_tickets f h) 
  (hr: total_revenue f h p): f * p = 1250 :=
sorry

end revenue_from_full_price_tickets_l34_34630


namespace problem1_l34_34487

theorem problem1 : (- (1 / 12) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = -21 :=
by
  sorry

end problem1_l34_34487


namespace halve_second_column_l34_34908

-- Definitions of given matrices
variable (f g h i : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := ![![f, g], ![h, i]])
variable (N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, (1/2)]])

-- Proof statement to be proved
theorem halve_second_column (hf : f ≠ 0) (hh : h ≠ 0) : N * A = ![![f, (1/2) * g], ![h, (1/2) * i]] := by
  sorry

end halve_second_column_l34_34908


namespace determine_x_y_l34_34254

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y_l34_34254


namespace total_disks_in_bag_l34_34734

/-- Given that the number of blue disks b, yellow disks y, and green disks g are in the ratio 3:7:8,
    and there are 30 more green disks than blue disks (g = b + 30),
    prove that the total number of disks is 108. -/
theorem total_disks_in_bag (b y g : ℕ) (h1 : 3 * y = 7 * b) (h2 : 8 * y = 7 * g) (h3 : g = b + 30) :
  b + y + g = 108 := by
  sorry

end total_disks_in_bag_l34_34734


namespace exists_function_l34_34556

theorem exists_function {n : ℕ} (hn : n ≥ 3) (S : Finset ℤ) (hS : S.card = n) :
  ∃ f : Fin (n) → S, 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i j k : Fin n, i < j ∧ j < k → 2 * (f j : ℤ) ≠ (f i : ℤ) + (f k : ℤ)) :=
by
  sorry

end exists_function_l34_34556


namespace sqrt_value_l34_34652

theorem sqrt_value (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := 
by
  sorry

end sqrt_value_l34_34652


namespace larger_sign_diameter_l34_34687

theorem larger_sign_diameter (d k : ℝ) 
  (h1 : ∀ d, d > 0) 
  (h2 : ∀ k, (π * (k * d / 2)^2 = 49 * π * (d / 2)^2)) : 
  k = 7 :=
by
sorry

end larger_sign_diameter_l34_34687


namespace total_renovation_cost_eq_l34_34661

-- Define the conditions
def hourly_rate_1 := 15
def hourly_rate_2 := 20
def hourly_rate_3 := 18
def hourly_rate_4 := 22
def hours_per_day := 8
def days := 10
def meal_cost_per_professional_per_day := 10
def material_cost := 2500
def plumbing_issue_cost := 750
def electrical_issue_cost := 500
def faulty_appliance_cost := 400

-- Define the calculated values based on the conditions
def daily_labor_cost_condition := 
  hourly_rate_1 * hours_per_day + 
  hourly_rate_2 * hours_per_day + 
  hourly_rate_3 * hours_per_day + 
  hourly_rate_4 * hours_per_day
def total_labor_cost := daily_labor_cost_condition * days

def daily_meal_cost := meal_cost_per_professional_per_day * 4
def total_meal_cost := daily_meal_cost * days

def unexpected_repair_costs := plumbing_issue_cost + electrical_issue_cost + faulty_appliance_cost

def total_cost := total_labor_cost + total_meal_cost + material_cost + unexpected_repair_costs

-- The theorem to prove that the total cost of the renovation is $10,550
theorem total_renovation_cost_eq : total_cost = 10550 := by
  sorry

end total_renovation_cost_eq_l34_34661


namespace simplified_sum_l34_34104

theorem simplified_sum :
  (-2^2003) + (2^2004) + (-2^2005) - (2^2006) = 5 * (2^2003) :=
by
  sorry

end simplified_sum_l34_34104


namespace area_of_rectangle_abcd_l34_34848

-- Definition of the problem's conditions and question
def small_square_side_length : ℝ := 1
def large_square_side_length : ℝ := 1.5
def area_rectangle_abc : ℝ := 4.5

-- Lean 4 statement: Prove the area of rectangle ABCD is 4.5 square inches
theorem area_of_rectangle_abcd :
  (3 * small_square_side_length) * large_square_side_length = area_rectangle_abc :=
by
  sorry

end area_of_rectangle_abcd_l34_34848


namespace range_of_independent_variable_l34_34554

theorem range_of_independent_variable (x : ℝ) : 
  (y = 3 / (x + 2)) → (x ≠ -2) :=
by
  -- suppose the function y = 3 / (x + 2) is given
  -- we need to prove x ≠ -2 for the function to be defined
  sorry

end range_of_independent_variable_l34_34554


namespace quadratic_expression_value_l34_34140

variables (α β : ℝ)
noncomputable def quadratic_root_sum (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) : Prop :=
  α + β = -2

theorem quadratic_expression_value (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) (h3 : α + β = -2) :
  α^2 + 3*α + β = -1 :=
sorry

end quadratic_expression_value_l34_34140


namespace count_3digit_numbers_div_by_13_l34_34404

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l34_34404


namespace percentage_of_original_price_l34_34476
-- Define the original price and current price in terms of real numbers
def original_price : ℝ := 25
def current_price : ℝ := 20

-- Lean statement to verify the correctness of the percentage calculation
theorem percentage_of_original_price :
  (current_price / original_price) * 100 = 80 := 
by
  sorry

end percentage_of_original_price_l34_34476


namespace flowers_given_l34_34711

theorem flowers_given (initial_flowers total_flowers flowers_given : ℝ)
  (h1 : initial_flowers = 67)
  (h2 : total_flowers = 157)
  (h3 : total_flowers = initial_flowers + flowers_given) :
  flowers_given = 90 :=
sorry

end flowers_given_l34_34711


namespace simplify_expression_l34_34396

theorem simplify_expression :
  (∃ (x : Real), x = 3 * (Real.sqrt 3 + Real.sqrt 7) / (4 * Real.sqrt (3 + Real.sqrt 5)) ∧ 
    x = Real.sqrt (224 - 22 * Real.sqrt 105) / 8) := sorry

end simplify_expression_l34_34396


namespace abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l34_34351

theorem abs_sqrt3_minus_1_sub_2_cos30_eq_neg1 :
  |(Real.sqrt 3) - 1| - 2 * Real.cos (Real.pi / 6) = -1 := by
  sorry

end abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l34_34351


namespace problem_statement_l34_34116

variables {Line Plane : Type}

-- Defining the perpendicular relationship between a line and a plane
def perp (a : Line) (α : Plane) : Prop := sorry

-- Defining the parallel relationship between two planes
def para (α β : Plane) : Prop := sorry

-- The main statement to prove
theorem problem_statement (a : Line) (α β : Plane) (h1 : perp a α) (h2 : perp a β) : para α β := 
sorry

end problem_statement_l34_34116


namespace election_total_votes_l34_34550

theorem election_total_votes (V : ℝ)
  (h_majority : ∃ O, 0.84 * V = O + 476)
  (h_total_votes : ∀ O, V = 0.84 * V + O) :
  V = 700 :=
sorry

end election_total_votes_l34_34550


namespace square_in_S_l34_34722

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

def S (n : ℕ) : Prop :=
  is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1)

theorem square_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end square_in_S_l34_34722


namespace negation_equivalence_l34_34698

-- Define the original proposition P
def proposition_P : Prop := ∀ x : ℝ, 0 ≤ x → x^3 + 2 * x ≥ 0

-- Define the negation of the proposition P
def negation_P : Prop := ∃ x : ℝ, 0 ≤ x ∧ x^3 + 2 * x < 0

-- The statement to be proven
theorem negation_equivalence : ¬ proposition_P ↔ negation_P := 
by sorry

end negation_equivalence_l34_34698


namespace find_d_l34_34065

theorem find_d (d x y : ℝ) (H1 : x - 2 * y = 5) (H2 : d * x + y = 6) (H3 : x > 0) (H4 : y > 0) :
  -1 / 2 < d ∧ d < 6 / 5 :=
by
  sorry

end find_d_l34_34065


namespace find_line_l_l34_34361

theorem find_line_l :
  ∃ l : ℝ × ℝ → Prop,
    (∀ (B : ℝ × ℝ), (2 * B.1 + B.2 - 8 = 0) → 
      (∀ A : ℝ × ℝ, (A.1 = -B.1 ∧ A.2 = 2 * B.1 - 6 ) → 
        (A.1 - 3 * A.2 + 10 = 0) → 
          B.1 = 4 ∧ B.2 = 0 ∧ ∀ p : ℝ × ℝ, B.1 * p.1 + 4 * p.2 - 4 = 0)) := 
  sorry

end find_line_l_l34_34361


namespace cost_price_correct_l34_34253

noncomputable def cost_price (selling_price marked_price_ratio cost_profit_ratio : ℝ) : ℝ :=
  (selling_price * marked_price_ratio) / cost_profit_ratio

theorem cost_price_correct : 
  abs (cost_price 63.16 0.94 1.25 - 50.56) < 0.01 :=
by 
  sorry

end cost_price_correct_l34_34253


namespace evaluate_expression_l34_34184

theorem evaluate_expression : 
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  a * b = 63 := 
by
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  sorry

end evaluate_expression_l34_34184


namespace find_subtracted_number_l34_34201

theorem find_subtracted_number (x y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  have hx : 2 * 129 - y = 110 := by
    rw [h1] at h2
    exact h2
  linarith

end find_subtracted_number_l34_34201


namespace eval_three_plus_three_cubed_l34_34371

theorem eval_three_plus_three_cubed : 3 + 3^3 = 30 := 
by 
  sorry

end eval_three_plus_three_cubed_l34_34371


namespace evaluate_expression_l34_34546

theorem evaluate_expression (x y : ℝ) (P Q : ℝ) 
  (hP : P = x^2 + y^2) 
  (hQ : Q = x - y) : 
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by 
  -- Insert proof here
  sorry

end evaluate_expression_l34_34546


namespace equivalence_of_expression_l34_34824

theorem equivalence_of_expression (x y : ℝ) :
  ( (x^2 + y^2 + xy) / (x^2 + y^2 - xy) ) - ( (x^2 + y^2 - xy) / (x^2 + y^2 + xy) ) =
  ( 4 * xy * (x^2 + y^2) ) / ( x^4 + y^4 ) :=
by sorry

end equivalence_of_expression_l34_34824


namespace sufficient_but_not_necessary_condition_l34_34128

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a ≠ 0) :
  (a > 2 ↔ |a - 1| > 1) ↔ (a > 2 → |a - 1| > 1) ∧ (a < 0 → |a - 1| > 1) ∧ (∃ x : ℝ, (|x - 1| > 1) ∧ x < 0 ∧ x ≠ a) :=
by
  sorry

end sufficient_but_not_necessary_condition_l34_34128


namespace ice_cubes_per_tray_l34_34569

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) (h1 : total_ice_cubes = 72) (h2 : number_of_trays = 8) : 
  total_ice_cubes / number_of_trays = 9 :=
by
  sorry

end ice_cubes_per_tray_l34_34569


namespace solution_to_fractional_equation_l34_34658

theorem solution_to_fractional_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 1) :
  (x / (x - 3) = (x + 1) / (x - 1)) ↔ (x = -3) :=
by
  sorry

end solution_to_fractional_equation_l34_34658


namespace problem1_problem2_l34_34976

-- Definitions based on the given conditions
def p (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a
def q (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0

-- Problem (1)
theorem problem1 (a x : ℝ) (h : a = 1) (hp : p a x) (hq : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : ∀ x, q x → p a x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end problem1_problem2_l34_34976


namespace speed_of_man_l34_34301

theorem speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ)
  (relative_speed_km_h : ℝ)
  (h_train_length : train_length = 440)
  (h_train_speed : train_speed_kmph = 60)
  (h_time : time_seconds = 24)
  (h_relative_speed : relative_speed_km_h = (train_length / time_seconds) * 3.6):
  (relative_speed_km_h - train_speed_kmph) = 6 :=
by sorry

end speed_of_man_l34_34301


namespace max_dn_eq_401_l34_34849

open BigOperators

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_dn_eq_401 : ∃ n, d n = 401 ∧ ∀ m, d m ≤ 401 := by
  -- Proof will be filled here
  sorry

end max_dn_eq_401_l34_34849


namespace tensor_identity_l34_34870

def tensor (a b : ℝ) : ℝ := a^3 - b

theorem tensor_identity (a : ℝ) : tensor a (tensor a (tensor a a)) = a^3 - a :=
by
  sorry

end tensor_identity_l34_34870


namespace arnaldo_bernaldo_distribute_toys_l34_34237

noncomputable def num_ways_toys_distributed (total_toys remaining_toys : ℕ) : ℕ :=
  if total_toys = 10 ∧ remaining_toys = 8 then 6561 - 256 else 0

theorem arnaldo_bernaldo_distribute_toys : num_ways_toys_distributed 10 8 = 6305 :=
by 
  -- Lean calculation for 3^8 = 6561 and 2^8 = 256 can be done as follows
  -- let three_power_eight := 3^8
  -- let two_power_eight := 2^8
  -- three_power_eight - two_power_eight = 6305
  sorry

end arnaldo_bernaldo_distribute_toys_l34_34237


namespace quadrilateral_area_l34_34380

theorem quadrilateral_area :
  let a1 := 9  -- adjacent side length
  let a2 := 6  -- other adjacent side length
  let d := 20  -- diagonal
  let θ1 := 35  -- first angle in degrees
  let θ2 := 110  -- second angle in degrees
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  area_triangle1 + area_triangle2 = 108.006 := 
by
  let a1 := 9
  let a2 := 6
  let d := 20
  let θ1 := 35
  let θ2 := 110
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  show area_triangle1 + area_triangle2 = 108.006
  sorry

end quadrilateral_area_l34_34380


namespace minimum_value_f_range_a_l34_34222

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, f x = -(1 / Real.exp 1) :=
sorry

theorem range_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ a ∈ Set.Iic 1 :=
sorry

end minimum_value_f_range_a_l34_34222


namespace geometric_sequence_sum_l34_34796

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l34_34796


namespace intersection_M_N_l34_34914

open Set

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

theorem intersection_M_N :
  M ∩ N = {-1, 3} := 
sorry

end intersection_M_N_l34_34914


namespace quadratic_roots_identity_l34_34109

variable (α β : ℝ)
variable (h1 : α^2 + 3*α - 7 = 0)
variable (h2 : β^2 + 3*β - 7 = 0)

-- The problem is to prove that α^2 + 4*α + β = 4
theorem quadratic_roots_identity :
  α^2 + 4*α + β = 4 :=
sorry

end quadratic_roots_identity_l34_34109


namespace bella_grazing_area_l34_34781

open Real

theorem bella_grazing_area:
  let leash_length := 5
  let barn_width := 4
  let barn_height := 6
  let sector_fraction := 3 / 4
  let area_circle := π * leash_length^2
  let grazed_area := sector_fraction * area_circle
  grazed_area = 75 / 4 * π := 
by
  sorry

end bella_grazing_area_l34_34781


namespace power_inequality_l34_34854

variable {a b : ℝ}

theorem power_inequality (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := 
by sorry

end power_inequality_l34_34854


namespace find_number_l34_34819

theorem find_number (x : ℤ) (h : (7 * (x + 10) / 5) - 5 = 44) : x = 25 :=
sorry

end find_number_l34_34819


namespace unique_solution_exists_l34_34336

theorem unique_solution_exists (k : ℝ) :
  (16 + 12 * k = 0) → ∃! x : ℝ, k * x^2 - 4 * x - 3 = 0 :=
by
  intro hk
  sorry

end unique_solution_exists_l34_34336


namespace triangle_perimeter_l34_34024

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 19)
  (ineq1 : a + b > c) (ineq2 : a + c > b) (ineq3 : b + c > a) : a + b + c = 44 :=
by
  -- Proof omitted
  sorry

end triangle_perimeter_l34_34024


namespace mailman_junk_mail_l34_34230

/-- 
  Given:
    - n = 640 : total number of pieces of junk mail for the block
    - h = 20 : number of houses in the block
  
  Prove:
    - The number of pieces of junk mail given to each house equals 32, when the total number of pieces of junk mail is divided by the number of houses.
--/
theorem mailman_junk_mail (n h : ℕ) (h_total : n = 640) (h_houses : h = 20) :
  n / h = 32 :=
by
  sorry

end mailman_junk_mail_l34_34230


namespace farmer_goats_l34_34038

theorem farmer_goats (G C P : ℕ) (h1 : P = 2 * C) (h2 : C = G + 4) (h3 : G + C + P = 56) : G = 11 :=
by
  sorry

end farmer_goats_l34_34038


namespace sum_of_fifth_powers_52070424_l34_34939

noncomputable def sum_of_fifth_powers (n : ℤ) : ℤ :=
  (n-1)^5 + n^5 + (n+1)^5

theorem sum_of_fifth_powers_52070424 :
  ∃ (n : ℤ), (n-1)^2 + n^2 + (n+1)^2 = 2450 ∧ sum_of_fifth_powers n = 52070424 :=
by
  sorry

end sum_of_fifth_powers_52070424_l34_34939


namespace ratio_tina_betsy_l34_34726

theorem ratio_tina_betsy :
  ∀ (t_cindy t_betsy t_tina : ℕ),
  t_cindy = 12 →
  t_betsy = t_cindy / 2 →
  t_tina = t_cindy + 6 →
  t_tina / t_betsy = 3 :=
by
  intros t_cindy t_betsy t_tina h_cindy h_betsy h_tina
  sorry

end ratio_tina_betsy_l34_34726


namespace product_of_roots_l34_34246

theorem product_of_roots (x : ℝ) (h : (x - 1) * (x + 4) = 22) : ∃ a b, (x^2 + 3*x - 26 = 0) ∧ a * b = -26 :=
by
  -- Given the equation (x - 1) * (x + 4) = 22,
  -- We want to show that the roots of the equation when simplified are such that
  -- their product is -26.
  sorry

end product_of_roots_l34_34246


namespace pencils_in_each_box_l34_34892

open Nat

theorem pencils_in_each_box (boxes pencils_given_to_Lauren pencils_left pencils_each_box more_pencils : ℕ)
  (h1 : boxes = 2)
  (h2 : pencils_given_to_Lauren = 6)
  (h3 : pencils_left = 9)
  (h4 : more_pencils = 3)
  (h5 : pencils_given_to_Matt = pencils_given_to_Lauren + more_pencils)
  (h6 : pencils_each_box = (pencils_given_to_Lauren + pencils_given_to_Matt + pencils_left) / boxes) :
  pencils_each_box = 12 := by
  sorry

end pencils_in_each_box_l34_34892


namespace value_of_fraction_l34_34089

variable (m n : ℚ)

theorem value_of_fraction (h₁ : 3 * m + 2 * n = 0) (h₂ : m ≠ 0 ∧ n ≠ 0) :
  (m / n - n / m) = 5 / 6 := 
sorry

end value_of_fraction_l34_34089


namespace ratio_john_maya_age_l34_34547

theorem ratio_john_maya_age :
  ∀ (john drew maya peter jacob : ℕ),
  -- Conditions:
  john = 30 ∧
  drew = maya + 5 ∧
  peter = drew + 4 ∧
  jacob = 11 ∧
  jacob + 2 = (peter + 2) / 2 →
  -- Conclusion:
  john / gcd john maya = 2 ∧ maya / gcd john maya = 1 :=
by
  sorry

end ratio_john_maya_age_l34_34547


namespace perpendicular_vector_l34_34412

-- Vectors a and b are given
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

-- Defining the vector addition and scalar multiplication for our context
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (m : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (m * v.1, m * v.2)

-- The vector a + m * b
def a_plus_m_b (m : ℝ) : ℝ × ℝ := vector_add a (scalar_mul m b)

-- The dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The statement that a is perpendicular to (a + m * b) when m = 5
theorem perpendicular_vector : dot_product a (a_plus_m_b 5) = 0 :=
sorry

end perpendicular_vector_l34_34412


namespace group1_calculation_group2_calculation_l34_34634

theorem group1_calculation : 9 / 3 * (9 - 1) = 24 := by
  sorry

theorem group2_calculation : 7 * (3 + 3 / 7) = 24 := by
  sorry

end group1_calculation_group2_calculation_l34_34634


namespace CaitlinAge_l34_34366

theorem CaitlinAge (age_AuntAnna : ℕ) (age_Brianna : ℕ) (age_Caitlin : ℕ)
  (h1 : age_AuntAnna = 42)
  (h2 : age_Brianna = age_AuntAnna / 2)
  (h3 : age_Caitlin = age_Brianna - 5) :
  age_Caitlin = 16 :=
by 
  sorry

end CaitlinAge_l34_34366


namespace probability_not_siblings_l34_34029

noncomputable def num_individuals : ℕ := 6
noncomputable def num_pairs : ℕ := num_individuals / 2
noncomputable def total_pairs : ℕ := num_individuals * (num_individuals - 1) / 2
noncomputable def sibling_pairs : ℕ := num_pairs
noncomputable def non_sibling_pairs : ℕ := total_pairs - sibling_pairs

theorem probability_not_siblings :
  (non_sibling_pairs : ℚ) / total_pairs = 4 / 5 := 
by sorry

end probability_not_siblings_l34_34029


namespace simplify_expression_l34_34573

theorem simplify_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 :=
sorry

end simplify_expression_l34_34573


namespace find_n_l34_34063

-- Definitions of the conditions
variables (x n : ℝ)
variable (h1 : (x / 4) * n + 10 - 12 = 48)
variable (h2 : x = 40)

-- Theorem statement
theorem find_n (x n : ℝ) (h1 : (x / 4) * n + 10 - 12 = 48) (h2 : x = 40) : n = 5 :=
by
  sorry

end find_n_l34_34063


namespace forty_percent_more_than_seventyfive_by_fifty_l34_34665

def number : ℝ := 312.5

theorem forty_percent_more_than_seventyfive_by_fifty 
    (x : ℝ) 
    (h : 0.40 * x = 0.75 * 100 + 50) : 
    x = number :=
by
  sorry

end forty_percent_more_than_seventyfive_by_fifty_l34_34665


namespace boys_in_biology_is_25_l34_34099

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l34_34099


namespace find_number_l34_34799

theorem find_number (x : ℕ) :
  ((4 * x) / 8 = 6) ∧ ((4 * x) % 8 = 4) → x = 13 :=
by
  sorry

end find_number_l34_34799


namespace find_square_tiles_l34_34078

theorem find_square_tiles (t s p : ℕ) (h1 : t + s + p = 35) (h2 : 3 * t + 4 * s + 5 * p = 140) (hp0 : p = 0) : s = 35 := by
  sorry

end find_square_tiles_l34_34078


namespace geometric_series_sum_l34_34127

theorem geometric_series_sum :
  let a := -2
  let r := 4
  let n := 10
  let S := (a * (r^n - 1)) / (r - 1)
  S = -699050 :=
by
  sorry

end geometric_series_sum_l34_34127


namespace quadrilateral_area_l34_34627

-- Define the dimensions of the rectangles
variables (AB BC EF FG : ℝ)
variables (AFCH_area : ℝ)

-- State the conditions explicitly
def conditions : Prop :=
  (AB = 9) ∧ 
  (BC = 5) ∧ 
  (EF = 3) ∧ 
  (FG = 10)

-- State the theorem to prove
theorem quadrilateral_area (h: conditions AB BC EF FG) : 
  AFCH_area = 52.5 := 
sorry

end quadrilateral_area_l34_34627


namespace highest_growth_rate_at_K_div_2_l34_34792

variable {K : ℝ}

-- Define the population growth rate as a function of the population size.
def population_growth_rate (N : ℝ) : ℝ := sorry

-- Define the S-shaped curve condition of population growth.
axiom s_shaped_curve : ∃ N : ℝ, population_growth_rate N = 0 ∧ population_growth_rate (N/2) > population_growth_rate N

theorem highest_growth_rate_at_K_div_2 (N : ℝ) (hN : N = K/2) :
  population_growth_rate N > population_growth_rate K :=
by
  sorry

end highest_growth_rate_at_K_div_2_l34_34792


namespace solve_quadratic_l34_34094

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l34_34094


namespace median_length_of_pieces_is_198_l34_34393

   -- Define the conditions
   variables (A B C D E : ℕ)
   variables (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E)
   variables (avg_length : (A + B + C + D + E) = 640)
   variables (h_A_max : A ≤ 110)

   -- Statement of the problem (proof stub)
   theorem median_length_of_pieces_is_198 :
     C = 198 :=
   by
   sorry
   
end median_length_of_pieces_is_198_l34_34393


namespace fixed_point_on_line_AC_l34_34364

-- Given definitions and conditions directly from a)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_P (x y : ℝ) : Prop := ∃ t : ℝ, x = t * y - 1
def reflection_across_x_axis (y : ℝ) : ℝ := -y

-- The final proof statement translating c)
theorem fixed_point_on_line_AC
  (A B C P : ℝ × ℝ)
  (hP : P = (-1, 0))
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hAB : ∃ t : ℝ, line_through_P A.1 A.2 ∧ line_through_P B.1 B.2)
  (hRef : C = (B.1, reflection_across_x_axis B.2)) :
  ∃ x y : ℝ, (x, y) = (1, 0) ∧ line_through_P x y := 
sorry

end fixed_point_on_line_AC_l34_34364


namespace algebraic_expression_value_l34_34620

theorem algebraic_expression_value (x : ℝ) (h : x = 4 * Real.sin (Real.pi / 4) - 2) :
  (1 / (x - 1) / (x + 2) / (x ^ 2 - 2 * x + 1) - x / (x + 2)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end algebraic_expression_value_l34_34620


namespace player1_winning_strategy_l34_34536

/--
Player 1 has a winning strategy if and only if N is not an odd power of 2,
under the game rules where players alternately subtract proper divisors
and a player loses when given a prime number or 1.
-/
theorem player1_winning_strategy (N: ℕ) : 
  ¬ (∃ k: ℕ, k % 2 = 1 ∧ N = 2^k) ↔ (∃ strategy: ℕ → ℕ, ∀ n ≠ 1, n ≠ prime → n - strategy n = m) :=
sorry

end player1_winning_strategy_l34_34536


namespace donation_ratio_l34_34769

theorem donation_ratio (D1 : ℝ) (D1_value : D1 = 10)
  (total_donation : D1 + D1 * 2 + D1 * 4 + D1 * 8 + D1 * 16 = 310) : 
  2 = 2 :=
by
  sorry

end donation_ratio_l34_34769


namespace accurate_place_24000_scientific_notation_46400000_l34_34006

namespace MathProof

def accurate_place (n : ℕ) : String :=
  if n = 24000 then "hundred's place" else "unknown"

def scientific_notation (n : ℕ) : String :=
  if n = 46400000 then "4.64 × 10^7" else "unknown"

theorem accurate_place_24000 : accurate_place 24000 = "hundred's place" :=
by
  sorry

theorem scientific_notation_46400000 : scientific_notation 46400000 = "4.64 × 10^7" :=
by
  sorry

end MathProof

end accurate_place_24000_scientific_notation_46400000_l34_34006


namespace distinct_values_of_fx_l34_34389

theorem distinct_values_of_fx :
  let f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋
  ∃ (s : Finset ℤ), (∀ x, 0 ≤ x ∧ x ≤ 10 → f x ∈ s) ∧ s.card = 61 :=
by
  sorry

end distinct_values_of_fx_l34_34389


namespace gcd_of_44_54_74_l34_34813

theorem gcd_of_44_54_74 : gcd (gcd 44 54) 74 = 2 :=
by
    sorry

end gcd_of_44_54_74_l34_34813


namespace distance_point_to_vertical_line_l34_34748

/-- The distance from a point to a vertical line equals the absolute difference in the x-coordinates. -/
theorem distance_point_to_vertical_line (x1 y1 x2 : ℝ) (h_line : x2 = -2) (h_point : (x1, y1) = (1, 2)) :
  abs (x1 - x2) = 3 :=
by
  -- Place proof here
  sorry

end distance_point_to_vertical_line_l34_34748


namespace leaked_before_fixing_l34_34228

def total_leaked_oil := 6206
def leaked_while_fixing := 3731

theorem leaked_before_fixing :
  total_leaked_oil - leaked_while_fixing = 2475 := by
  sorry

end leaked_before_fixing_l34_34228


namespace min_value_n_l34_34011

noncomputable def minN : ℕ :=
  5

theorem min_value_n :
  ∀ (S : Finset ℕ), (∀ n ∈ S, 1 ≤ n ∧ n ≤ 9) ∧ S.card = minN → 
    (∃ T ⊆ S, T ≠ ∅ ∧ 10 ∣ (T.sum id)) :=
by
  sorry

end min_value_n_l34_34011


namespace arithmetic_sequence_sum_10_l34_34503

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable def a_n (a1 d : α) (n : ℕ) : α :=
a1 + (n - 1) • d

def sequence_sum (a1 d : α) (n : ℕ) : α :=
n • a1 + (n • (n - 1) / 2) • d

theorem arithmetic_sequence_sum_10 
  (a1 d : ℤ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 4)
  (h2 : a_n a1 d 3 + a_n a1 d 5 = 10) :
  sequence_sum a1 d 10 = 95 :=
by
  sorry

end arithmetic_sequence_sum_10_l34_34503


namespace ratio_books_purchased_l34_34775

-- Definitions based on the conditions
def books_last_year : ℕ := 50
def books_before_purchase : ℕ := 100
def books_now : ℕ := 300

-- Let x be the multiple of the books purchased this year
def multiple_books_purchased_this_year (x : ℕ) : Prop :=
  books_now = books_before_purchase + books_last_year + books_last_year * x

-- Prove the ratio is 3:1
theorem ratio_books_purchased (x : ℕ) (h : multiple_books_purchased_this_year x) : x = 3 :=
  by sorry

end ratio_books_purchased_l34_34775


namespace ordered_triple_exists_l34_34308

theorem ordered_triple_exists (a b c : ℝ) (h₁ : 4 < a) (h₂ : 4 < b) (h₃ : 4 < c)
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) :=
sorry

end ordered_triple_exists_l34_34308


namespace xiangshan_port_investment_scientific_notation_l34_34258

-- Definition of scientific notation
def in_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

-- Theorem stating the equivalence of the investment in scientific notation
theorem xiangshan_port_investment_scientific_notation :
  in_scientific_notation 7.7 9 7.7e9 :=
by {
  sorry
}

end xiangshan_port_investment_scientific_notation_l34_34258


namespace evens_minus_odds_equal_40_l34_34176

-- Define the sum of even integers from 2 to 80
def sum_evens : ℕ := (List.range' 2 40).sum

-- Define the sum of odd integers from 1 to 79
def sum_odds : ℕ := (List.range' 1 40).sum

-- Define the main theorem to prove
theorem evens_minus_odds_equal_40 : sum_evens - sum_odds = 40 := by
  -- Proof will go here
  sorry

end evens_minus_odds_equal_40_l34_34176


namespace cost_of_each_soda_l34_34117

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l34_34117


namespace range_of_m_plus_n_l34_34208

theorem range_of_m_plus_n (m n : ℝ)
  (tangent_condition : (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1)) :
  m + n ∈ (Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2)) :=
sorry

end range_of_m_plus_n_l34_34208


namespace rose_bush_cost_correct_l34_34236

-- Definitions of the given conditions
def total_rose_bushes : ℕ := 20
def gardener_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def gardener_cost : ℕ := gardener_rate * gardener_hours_per_day * gardener_days
def soil_cubic_feet : ℕ := 100
def soil_cost_per_cubic_foot : ℕ := 5
def soil_cost : ℕ := soil_cubic_feet * soil_cost_per_cubic_foot
def total_cost : ℕ := 4100

-- Result computed given the conditions
def rose_bush_cost : ℕ := 150

-- The proof goal (statement only, no proof)
theorem rose_bush_cost_correct : 
  total_cost - gardener_cost - soil_cost = total_rose_bushes * rose_bush_cost :=
by
  sorry

end rose_bush_cost_correct_l34_34236


namespace find_line_equation_l34_34529

-- Definition of a line passing through a point
def passes_through (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := l p.1 p.2

-- Definition of intercepts being opposite
def opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 ∧ l 0 (-a)

-- The line passing through the point (7, 1)
def line_exists (l : ℝ → ℝ → Prop) : Prop :=
  passes_through l (7, 1) ∧ opposite_intercepts l

-- Main theorem to prove the equation of the line
theorem find_line_equation (l : ℝ → ℝ → Prop) :
  line_exists l ↔ (∀ x y, l x y ↔ x - 7 * y = 0) ∨ (∀ x y, l x y ↔ x - y - 6 = 0) :=
sorry

end find_line_equation_l34_34529


namespace range_of_function_l34_34876

theorem range_of_function : ∀ x : ℝ, 1 ≤ abs (Real.sin x) + 2 * abs (Real.cos x) ∧ abs (Real.sin x) + 2 * abs (Real.cos x) ≤ Real.sqrt 5 :=
by
  intro x
  sorry

end range_of_function_l34_34876


namespace negation_exists_ge_zero_l34_34502

theorem negation_exists_ge_zero (h : ∀ x > 0, x^2 - 3 * x + 2 < 0) :
  ∃ x > 0, x^2 - 3 * x + 2 ≥ 0 :=
sorry

end negation_exists_ge_zero_l34_34502


namespace find_f_4500_l34_34790

noncomputable def f : ℕ → ℕ
| 0 => 1
| (n + 3) => f n + 2 * n + 3
| n => sorry  -- This handles all other cases, but should not be called.

theorem find_f_4500 : f 4500 = 6750001 :=
by
  sorry

end find_f_4500_l34_34790


namespace distance_to_first_sign_l34_34055

-- Definitions based on conditions
def total_distance : ℕ := 1000
def after_second_sign : ℕ := 275
def between_signs : ℕ := 375

-- Problem statement
theorem distance_to_first_sign 
  (D : ℕ := total_distance) 
  (a : ℕ := after_second_sign) 
  (d : ℕ := between_signs) : 
  (D - a - d = 350) :=
by
  sorry

end distance_to_first_sign_l34_34055


namespace total_cards_given_away_l34_34609

-- Define the conditions in Lean
def Jim_initial_cards : ℕ := 365
def sets_given_to_brother : ℕ := 8
def sets_given_to_sister : ℕ := 5
def sets_given_to_friend : ℕ := 2
def cards_per_set : ℕ := 13

-- Define a theorem to prove the total number of cards given away
theorem total_cards_given_away : 
  sets_given_to_brother + sets_given_to_sister + sets_given_to_friend = 15 ∧
  15 * cards_per_set = 195 := 
by
  sorry

end total_cards_given_away_l34_34609


namespace tracey_initial_candies_l34_34443

theorem tracey_initial_candies (x : ℕ) :
  (x % 4 = 0) ∧ (104 ≤ x) ∧ (x ≤ 112) ∧
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ (x / 2 - 40 - k = 10)) →
  (x = 108 ∨ x = 112) :=
by
  sorry

end tracey_initial_candies_l34_34443


namespace triangle_right_angle_solution_l34_34142

def is_right_angle (a b : ℝ × ℝ) : Prop := (a.1 * b.1 + a.2 * b.2 = 0)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem triangle_right_angle_solution (x : ℝ) (h1 : (2, -1) = (2, -1)) (h2 : (x, 3) = (x, 3)) : 
  is_right_angle (2, -1) (x, 3) ∨ 
  is_right_angle (2, -1) (vector_sub (x, 3) (2, -1)) ∨ 
  is_right_angle (x, 3) (vector_sub (x, 3) (2, -1)) → 
  x = 3 / 2 ∨ x = 4 :=
sorry

end triangle_right_angle_solution_l34_34142


namespace xiaohua_amount_paid_l34_34125

def cost_per_bag : ℝ := 18
def discount_rate : ℝ := 0.1
def price_difference : ℝ := 36

theorem xiaohua_amount_paid (x : ℝ) 
  (h₁ : 18 * (x+1) * (1 - 0.1) = 18 * x - 36) :
  18 * (x + 1) * (1 - 0.1) = 486 := 
sorry

end xiaohua_amount_paid_l34_34125


namespace temperature_or_daytime_not_sufficiently_high_l34_34807

variable (T : ℝ) (Daytime Lively : Prop)
axiom h1 : (T ≥ 75 ∧ Daytime) → Lively
axiom h2 : ¬ Lively

theorem temperature_or_daytime_not_sufficiently_high : T < 75 ∨ ¬ Daytime :=
by
  -- proof steps
  sorry

end temperature_or_daytime_not_sufficiently_high_l34_34807


namespace determine_m_values_l34_34334

theorem determine_m_values (m : ℚ) :
  ((∃ x y : ℚ, x = -3 ∧ y = 0 ∧ (m^2 - 2 * m - 3) * x + (2 * m^2 + m - 1) * y = 2 * m - 6) ∨
  (∃ k : ℚ, k = -1 ∧ (m^2 - 2 * m - 3) + (2 * m^2 + m - 1) * k = 0)) →
  (m = -5/3 ∨ m = 4/3) :=
by
  sorry

end determine_m_values_l34_34334


namespace parabola_sum_vertex_point_l34_34047

theorem parabola_sum_vertex_point
  (a b c : ℝ)
  (h_vertex : ∀ y : ℝ, y = -6 → x = a * (y + 6)^2 + 8)
  (h_point : x = a * ((-4) + 6)^2 + 8)
  (ha : a = 0.5)
  (hb : b = 6)
  (hc : c = 26) :
  a + b + c = 32.5 :=
by
  sorry

end parabola_sum_vertex_point_l34_34047


namespace ellipse_perimeter_l34_34493

noncomputable def perimeter_of_triangle (a b : ℝ) (e : ℝ) : ℝ :=
  if (b = 4 ∧ e = 3 / 5 ∧ a = b / (1 - e^2) ^ (1 / 2))
  then 4 * a
  else 0

theorem ellipse_perimeter :
  let a : ℝ := 5
  let b : ℝ := 4
  let e : ℝ := 3 / 5
  4 * a = 20 :=
by
  sorry

end ellipse_perimeter_l34_34493


namespace smallest_four_digit_divisible_by_3_5_7_11_l34_34018

theorem smallest_four_digit_divisible_by_3_5_7_11 : 
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 
          n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 1155 :=
by
  sorry

end smallest_four_digit_divisible_by_3_5_7_11_l34_34018


namespace polynomial_value_l34_34537

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l34_34537


namespace holden_master_bath_size_l34_34985

theorem holden_master_bath_size (b n m : ℝ) (h_b : b = 309) (h_n : n = 918) (h : 2 * (b + m) = n) : m = 150 := by
  sorry

end holden_master_bath_size_l34_34985


namespace problem_proof_l34_34575

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- The conditions
def even_function_on_ℝ := ∀ x : ℝ, f x = f (-x)
def f_at_0_is_2 := f 0 = 2
def odd_after_translation := ∀ x : ℝ, f (x - 1) = -f (-x - 1)

-- Prove the required condition
theorem problem_proof (h1 : even_function_on_ℝ f) (h2 : f_at_0_is_2 f) (h3 : odd_after_translation f) :
    f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
by
  sorry

end problem_proof_l34_34575


namespace send_messages_ways_l34_34417

theorem send_messages_ways : (3^4 = 81) :=
by
  sorry

end send_messages_ways_l34_34417


namespace range_of_a_l34_34307

open Real

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 6) ∨ (a ≥ 5 ∨ a ≤ 1) ∧ ¬((0 < a ∧ a < 6) ∧ (a ≥ 5 ∨ a ≤ 1)) ↔ 
  (a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)) :=
by sorry

end range_of_a_l34_34307


namespace domain_f_l34_34198

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (x + 1) + (x - 2) ^ 0

theorem domain_f :
  (∃ x : ℝ, f x = f x) ↔ (∀ x, (x > -1 ∧ x ≠ 2) ↔ (x ∈ Ioo (-1 : ℝ) 2 ∨ x ∈ Ioi 2)) :=
by
  sorry

end domain_f_l34_34198


namespace marsha_remainder_l34_34268

-- Definitions based on problem conditions
def a (n : ℤ) : ℤ := 90 * n + 84
def b (m : ℤ) : ℤ := 120 * m + 114
def c (p : ℤ) : ℤ := 150 * p + 144

-- Proof statement
theorem marsha_remainder (n m p : ℤ) : ((a n + b m + c p) % 30) = 12 :=
by 
  -- Notice we need to add the proof steps here
  sorry 

end marsha_remainder_l34_34268


namespace relationship_of_arithmetic_progression_l34_34300

theorem relationship_of_arithmetic_progression (x y z d : ℝ) (h1 : x + (y - z) + d = y + (z - x))
    (h2 : y + (z - x) + d = z + (x - y))
    (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
    x = y + d / 2 ∧ z = y + d := by
  sorry

end relationship_of_arithmetic_progression_l34_34300


namespace find_r_l34_34429

variable (a b c r : ℝ)

theorem find_r (h1 : a * (b - c) / (b * (c - a)) = r)
               (h2 : b * (c - a) / (c * (b - a)) = r)
               (h3 : r > 0) : 
               r = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_r_l34_34429


namespace prob_B_independent_l34_34353

-- Definitions based on the problem's conditions
def prob_A := 0.7
def prob_A_union_B := 0.94

-- With these definitions established, we need to state the theorem.
-- The theorem should express that the probability of B solving the problem independently (prob_B) is 0.8.
theorem prob_B_independent : 
    (∃ (prob_B: ℝ), prob_A = 0.7 ∧ prob_A_union_B = 0.94 ∧ prob_B = 0.8) :=
by
    sorry

end prob_B_independent_l34_34353


namespace scientific_notation_representation_l34_34333

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l34_34333


namespace power_root_l34_34998

noncomputable def x : ℝ := 1024 ^ (1 / 5)

theorem power_root (h : 1024 = 2^10) : x = 4 :=
by
  sorry

end power_root_l34_34998


namespace sarahs_score_l34_34289

theorem sarahs_score (g s : ℕ) (h1 : s = g + 60) (h2 : s + g = 260) : s = 160 :=
sorry

end sarahs_score_l34_34289


namespace least_xy_value_l34_34731

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l34_34731


namespace volume_of_cube_l34_34354

theorem volume_of_cube (a : ℕ) (h : a^3 - (a^3 - 4 * a) = 12) : a^3 = 27 :=
by 
  sorry

end volume_of_cube_l34_34354


namespace rearrange_to_rectangle_l34_34452

-- Definition of a geometric figure and operations
structure Figure where
  parts : List (List (ℤ × ℤ)) -- List of parts represented by lists of coordinates

def is_cut_into_three_parts (fig : Figure) : Prop :=
  fig.parts.length = 3

def can_be_rearranged_to_form_rectangle (fig : Figure) : Prop := sorry

-- Initial given figure
variable (initial_figure : Figure)

-- Conditions
axiom figure_can_be_cut : is_cut_into_three_parts initial_figure
axiom cuts_not_along_grid_lines : True -- Replace with appropriate geometric operation when image is known
axiom parts_can_be_flipped : True -- Replace with operation allowing part flipping

-- Theorem to prove
theorem rearrange_to_rectangle : 
  is_cut_into_three_parts initial_figure →
  can_be_rearranged_to_form_rectangle initial_figure := 
sorry

end rearrange_to_rectangle_l34_34452


namespace parallelogram_base_length_l34_34112

theorem parallelogram_base_length
  (height : ℝ) (area : ℝ) (base : ℝ) 
  (h1 : height = 18) 
  (h2 : area = 576) 
  (h3 : area = base * height) : 
  base = 32 :=
by
  rw [h1, h2] at h3
  sorry

end parallelogram_base_length_l34_34112


namespace tangent_line_to_C1_and_C2_is_correct_l34_34374

def C1 (x : ℝ) : ℝ := x ^ 2
def C2 (x : ℝ) : ℝ := -(x - 2) ^ 2
def l (x : ℝ) : ℝ := -2 * x + 3

theorem tangent_line_to_C1_and_C2_is_correct :
  (∃ x1 : ℝ, C1 x1 = l x1 ∧ deriv C1 x1 = deriv l x1) ∧
  (∃ x2 : ℝ, C2 x2 = l x2 ∧ deriv C2 x2 = deriv l x2) :=
sorry

end tangent_line_to_C1_and_C2_is_correct_l34_34374


namespace sequence_sum_a_b_l34_34882

theorem sequence_sum_a_b (a b : ℕ) (a_seq : ℕ → ℕ) 
  (h1 : a_seq 1 = a)
  (h2 : a_seq 2 = b)
  (h3 : ∀ n ≥ 1, a_seq (n+2) = (a_seq n + 2018) / (a_seq (n+1) + 1)) :
  a + b = 1011 ∨ a + b = 2019 :=
sorry

end sequence_sum_a_b_l34_34882


namespace percentage_increase_l34_34068

variables {a b : ℝ} -- Assuming a and b are real numbers

-- Define the conditions explicitly
def initial_workers := a
def workers_left := b
def remaining_workers := a - b

-- Define the theorem for percentage increase in daily performance
theorem percentage_increase (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (100 * b) / (a - b) = (100 * a * b) / (a * (a - b)) :=
by
  sorry -- Proof will be filled in as needed

end percentage_increase_l34_34068


namespace Bruce_paid_l34_34129

noncomputable def total_paid : ℝ :=
  let grapes_price := 9 * 70 * (1 - 0.10)
  let mangoes_price := 7 * 55 * (1 - 0.05)
  let oranges_price := 5 * 45 * (1 - 0.15)
  let apples_price := 3 * 80 * (1 - 0.20)
  grapes_price + mangoes_price + oranges_price + apples_price

theorem Bruce_paid (h : total_paid = 1316.25) : true :=
by
  -- This is where the proof would be
  sorry

end Bruce_paid_l34_34129


namespace molecular_weight_calculation_l34_34262

-- Define the atomic weights of each element
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms of each element in the compound
def num_atoms_C : ℕ := 7
def num_atoms_H : ℕ := 6
def num_atoms_O : ℕ := 2

-- The molecular weight calculation
def molecular_weight : ℝ :=
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_H * atomic_weight_H) +
  (num_atoms_O * atomic_weight_O)

theorem molecular_weight_calculation : molecular_weight = 122.118 :=
by
  -- Proof
  sorry

end molecular_weight_calculation_l34_34262


namespace distinct_students_count_l34_34423

theorem distinct_students_count
  (algebra_students : ℕ)
  (calculus_students : ℕ)
  (statistics_students : ℕ)
  (algebra_statistics_overlap : ℕ)
  (no_other_overlaps : algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32) :
  algebra_students = 13 → calculus_students = 10 → statistics_students = 12 → algebra_statistics_overlap = 3 → 
  algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end distinct_students_count_l34_34423


namespace price_of_ice_cream_bar_is_correct_l34_34978

noncomputable def price_ice_cream_bar (n_ice_cream_bars n_sundaes total_price price_of_sundae price_ice_cream_bar : ℝ) : Prop :=
  n_ice_cream_bars = 125 ∧
  n_sundaes = 125 ∧
  total_price = 225 ∧
  price_of_sundae = 1.2 →
  price_ice_cream_bar = 0.6

theorem price_of_ice_cream_bar_is_correct :
  price_ice_cream_bar 125 125 225 1.2 0.6 :=
by
  sorry

end price_of_ice_cream_bar_is_correct_l34_34978


namespace neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l34_34409

-- (1) Prove -(-2) = 2
theorem neg_neg_two : -(-2) = 2 := 
sorry

-- (2) Prove -6 + 6 = 0
theorem neg_six_plus_six : -6 + 6 = 0 := 
sorry

-- (3) Prove (-3) * 5 = -15
theorem neg_three_times_five : (-3) * 5 = -15 := 
sorry

-- (4) Prove 2x - 3x = -x
theorem two_x_minus_three_x (x : ℝ) : 2 * x - 3 * x = - x := 
sorry

end neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l34_34409


namespace trig_identity_product_l34_34625

theorem trig_identity_product :
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * 
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 16 :=
by
  sorry

end trig_identity_product_l34_34625


namespace probability_of_selecting_same_gender_l34_34843

def number_of_ways_to_choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_of_selecting_same_gender (total_students male_students female_students : ℕ) (h1 : total_students = 10) (h2 : male_students = 2) (h3 : female_students = 8) : 
  let total_combinations := number_of_ways_to_choose_two total_students
  let male_combinations := number_of_ways_to_choose_two male_students
  let female_combinations := number_of_ways_to_choose_two female_students
  let favorable_combinations := male_combinations + female_combinations
  total_combinations = 45 ∧
  male_combinations = 1 ∧
  female_combinations = 28 ∧
  favorable_combinations = 29 ∧
  (favorable_combinations : ℚ) / total_combinations = 29 / 45 :=
by
  sorry

end probability_of_selecting_same_gender_l34_34843


namespace xiaoxia_exceeds_xiaoming_l34_34015

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  52 + 15 * n > 70 + 12 * n := 
sorry

end xiaoxia_exceeds_xiaoming_l34_34015


namespace no_integer_b_two_distinct_roots_l34_34394

theorem no_integer_b_two_distinct_roots :
  ∀ b : ℤ, ¬ ∃ x y : ℤ, x ≠ y ∧ (x^4 + 4 * x^3 + b * x^2 + 16 * x + 8 = 0) ∧ (y^4 + 4 * y^3 + b * y^2 + 16 * y + 8 = 0) :=
by
  sorry

end no_integer_b_two_distinct_roots_l34_34394


namespace sin_neg_045_unique_solution_l34_34538

theorem sin_neg_045_unique_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 180) (h: ℝ) :
  (h = Real.sin x → h = -0.45) → 
  ∃! x, 0 ≤ x ∧ x < 180 ∧ Real.sin x = -0.45 :=
by sorry

end sin_neg_045_unique_solution_l34_34538


namespace repeating_decimal_sum_in_lowest_terms_l34_34357

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l34_34357


namespace average_writing_speed_time_to_write_10000_words_l34_34780

-- Definitions based on the problem conditions
def total_words : ℕ := 60000
def total_hours : ℝ := 90.5
def writing_speed : ℝ := 663
def words_to_write : ℕ := 10000
def writing_time : ℝ := 15.08

-- Proposition that the average writing speed is 663 words per hour
theorem average_writing_speed :
  (total_words : ℝ) / total_hours = writing_speed :=
sorry

-- Proposition that the time to write 10,000 words at the given average speed is 15.08 hours
theorem time_to_write_10000_words :
  (words_to_write : ℝ) / writing_speed = writing_time :=
sorry

end average_writing_speed_time_to_write_10000_words_l34_34780


namespace fractional_expression_simplification_l34_34700

theorem fractional_expression_simplification (x : ℕ) (h : x - 3 < 0) : 
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / 3 :=
by {
  -- Typical proof steps would go here, adhering to the natural conditions.
  sorry
}

end fractional_expression_simplification_l34_34700


namespace no_point_on_line_y_eq_2x_l34_34866

theorem no_point_on_line_y_eq_2x
  (marked : Set (ℕ × ℕ))
  (initial_points : { p // p ∈ [(1, 1), (2, 3), (4, 5), (999, 111)] })
  (rule1 : ∀ a b, (a, b) ∈ marked → (b, a) ∈ marked ∧ (a - b, a + b) ∈ marked)
  (rule2 : ∀ a b c d, (a, b) ∈ marked ∧ (c, d) ∈ marked → (a * d + b * c, 4 * a * c - 4 * b * d) ∈ marked) :
  ∃ x, (x, 2 * x) ∈ marked → False := sorry

end no_point_on_line_y_eq_2x_l34_34866


namespace expand_expression_l34_34007

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 :=
by
  sorry

end expand_expression_l34_34007


namespace solve_problem_l34_34495

noncomputable def problem_statement : Prop :=
  ∀ (tons_to_pounds : ℕ) 
    (packet_weight_pounds : ℕ) 
    (packet_weight_ounces : ℕ)
    (num_packets : ℕ)
    (bag_capacity_tons : ℕ)
    (X : ℕ),
    tons_to_pounds = 2300 →
    packet_weight_pounds = 16 →
    packet_weight_ounces = 4 →
    num_packets = 1840 →
    bag_capacity_tons = 13 →
    X = (packet_weight_ounces * bag_capacity_tons * tons_to_pounds) / 
        ((bag_capacity_tons * tons_to_pounds) - (num_packets * packet_weight_pounds)) →
    X = 16

theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l34_34495


namespace problem1_problem2_l34_34783

theorem problem1 (x : ℝ) (h : 4 * x^2 - 9 = 0) : x = 3/2 ∨ x = -3/2 :=
by
  sorry

theorem problem2 (x : ℝ) (h : 64 * (x-2)^3 - 1 = 0) : x = 2 + 1/4 :=
by
  sorry

end problem1_problem2_l34_34783


namespace gravel_cost_calculation_l34_34948

def cubicYardToCubicFoot : ℕ := 27
def costPerCubicFoot : ℕ := 8
def volumeInCubicYards : ℕ := 8

theorem gravel_cost_calculation : 
  (volumeInCubicYards * cubicYardToCubicFoot * costPerCubicFoot) = 1728 := 
by
  -- This is just a placeholder to ensure the statement is syntactically correct.
  sorry

end gravel_cost_calculation_l34_34948


namespace function_passes_through_fixed_point_l34_34964

noncomputable def f (a : ℝ) (x : ℝ) := 4 + Real.log (x + 1) / Real.log a

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  f a 0 = 4 := 
by
  sorry

end function_passes_through_fixed_point_l34_34964


namespace evaluate_log_expression_l34_34959

noncomputable def evaluate_expression (x y : Real) : Real :=
  (Real.log x / Real.log (y ^ 8)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 7)) * 
  (Real.log (x ^ 7) / Real.log (y ^ 3)) * 
  (Real.log (y ^ 8) / Real.log (x ^ 2))

theorem evaluate_log_expression (x y : Real) : 
  evaluate_expression x y = (1 : Real) := sorry

end evaluate_log_expression_l34_34959


namespace work_efficiency_ratio_l34_34151

-- Define the problem conditions and the ratio we need to prove.
theorem work_efficiency_ratio :
  (∃ (a b : ℝ), b = 1 / 18 ∧ (a + b) = 1 / 12 ∧ (a / b) = 1 / 2) :=
by {
  -- Definitions and variables can be listed if necessary
  -- a : ℝ
  -- b : ℝ
  -- Assume conditions
  sorry
}

end work_efficiency_ratio_l34_34151


namespace simplify_root_power_l34_34786

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l34_34786


namespace triangle_angle_area_l34_34956

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x
variables {A B C : ℝ}
variables {BC : ℝ}
variables {S : ℝ}

theorem triangle_angle_area (hABC : A + B + C = π) (hBC : BC = 2) (h_fA : f A = 0) 
  (hA : A = π / 3) : S = Real.sqrt 3 :=
by
  -- Sorry, proof skipped
  sorry

end triangle_angle_area_l34_34956


namespace integer_solution_of_inequality_system_l34_34795

theorem integer_solution_of_inequality_system :
  ∃ x : ℤ, (2 * (x : ℝ) ≤ 1) ∧ ((x : ℝ) + 2 > 1) ∧ (x = 0) :=
by
  sorry

end integer_solution_of_inequality_system_l34_34795


namespace gcd_of_terms_l34_34977

theorem gcd_of_terms (m n : ℕ) : gcd (4 * m^3 * n) (9 * m * n^3) = m * n := 
sorry

end gcd_of_terms_l34_34977


namespace period_six_l34_34879

variable {R : Type} [LinearOrderedField R]

def symmetric1 (f : R → R) : Prop := ∀ x : R, f (2 + x) = f (2 - x)
def symmetric2 (f : R → R) : Prop := ∀ x : R, f (5 + x) = f (5 - x)

theorem period_six (f : R → R) (h1 : symmetric1 f) (h2 : symmetric2 f) : ∀ x : R, f (x + 6) = f x :=
sorry

end period_six_l34_34879


namespace probability_Z_l34_34616

theorem probability_Z (p_X p_Y p_Z : ℚ)
  (hX : p_X = 2 / 5)
  (hY : p_Y = 1 / 4)
  (hTotal : p_X + p_Y + p_Z = 1) :
  p_Z = 7 / 20 := by sorry

end probability_Z_l34_34616


namespace cassandra_collected_pennies_l34_34572

theorem cassandra_collected_pennies 
(C : ℕ) 
(h1 : ∀ J : ℕ,  J = C - 276) 
(h2 : ∀ J : ℕ, C + J = 9724) 
: C = 5000 := 
by
  sorry

end cassandra_collected_pennies_l34_34572


namespace index_card_area_l34_34252

theorem index_card_area :
  ∀ (length width : ℕ), length = 5 → width = 7 →
  (length - 2) * width = 21 →
  length * (width - 1) = 30 :=
by
  intros length width h_length h_width h_condition
  sorry

end index_card_area_l34_34252


namespace normal_price_of_article_l34_34793

theorem normal_price_of_article 
  (final_price : ℝ) 
  (d1 d2 d3 : ℝ) 
  (P : ℝ) 
  (h_final_price : final_price = 36) 
  (h_d1 : d1 = 0.15) 
  (h_d2 : d2 = 0.25) 
  (h_d3 : d3 = 0.20) 
  (h_eq : final_price = P * (1 - d1) * (1 - d2) * (1 - d3)) : 
  P = 70.59 := sorry

end normal_price_of_article_l34_34793


namespace quadrilateral_area_l34_34270

theorem quadrilateral_area (c d : ℤ) (h1 : 0 < d) (h2 : d < c) (h3 : 2 * ((c : ℝ) ^ 2 - (d : ℝ) ^ 2) = 18) : 
  c + d = 9 :=
by
  sorry

end quadrilateral_area_l34_34270


namespace quadratic_rewrite_sum_l34_34030

theorem quadratic_rewrite_sum :
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  a + b + c = 143.25 :=
by 
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  sorry

end quadratic_rewrite_sum_l34_34030


namespace smallest_num_is_1113805958_l34_34637

def smallest_num (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧ (n + 5) % 73 = 0 ∧ (n + 5) % 101 = 0 ∧ (n + 5) % 89 = 0

theorem smallest_num_is_1113805958 : ∃ n, smallest_num n ∧ n = 1113805958 :=
by
  use 1113805958
  unfold smallest_num
  simp
  sorry

end smallest_num_is_1113805958_l34_34637


namespace integer_part_of_shortest_distance_l34_34714

def cone_slant_height := 21
def cone_radius := 14
def ant_position := cone_slant_height / 2
def angle_opposite := 240
def cos_angle_opposite := -1 / 2

noncomputable def shortest_distance := 
  Real.sqrt ((ant_position ^ 2) + (ant_position ^ 2) + (2 * ant_position ^ 2 * cos_angle_opposite))

theorem integer_part_of_shortest_distance : Int.floor shortest_distance = 18 :=
by
  /- Proof steps go here -/
  sorry

end integer_part_of_shortest_distance_l34_34714


namespace mother_age_4times_daughter_l34_34250

-- Conditions
def Y := 12
def M := 42

-- Proof statement: Prove that 2 years ago, the mother's age was 4 times Yujeong's age.
theorem mother_age_4times_daughter (X : ℕ) (hY : Y = 12) (hM : M = 42) : (42 - X) = 4 * (12 - X) :=
by
  intros
  sorry

end mother_age_4times_daughter_l34_34250


namespace probability_of_event_B_l34_34096

def fair_dice := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

def event_B (x y : fair_dice) : Prop := x.val = y.val + 2

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 6

theorem probability_of_event_B : (favorable_outcomes : ℚ) / total_outcomes = 3/32 := by
  have h1 : (64 : ℚ) = 8 * 8 := by norm_num
  have h2 : (6 : ℚ) / 64 = 3 / 32 := by norm_num
  sorry

end probability_of_event_B_l34_34096


namespace correct_option_C_l34_34264

variable (x : ℝ)
variable (hx : 0 < x ∧ x < 1)

theorem correct_option_C : 0 < 1 - x^2 ∧ 1 - x^2 < 1 :=
by
  sorry

end correct_option_C_l34_34264


namespace product_divisible_by_10_l34_34295

noncomputable def probability_divisible_by_10 (n : ℕ) (h : n > 1) : ℝ :=
  1 - (8^n + 5^n - 4^n) / 9^n

theorem product_divisible_by_10 (n : ℕ) (h : n > 1) :
  probability_divisible_by_10 n h = 1 - (8^n + 5^n - 4^n)/(9^n) :=
by
  sorry

end product_divisible_by_10_l34_34295


namespace remainder_14_plus_x_mod_31_l34_34615

theorem remainder_14_plus_x_mod_31 (x : ℕ) (hx : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
sorry

end remainder_14_plus_x_mod_31_l34_34615


namespace father_l34_34585

theorem father's_age_at_middle_son_birth (a b c F : ℕ) 
  (h1 : a = b + c) 
  (h2 : F * a * b * c = 27090) : 
  F - b = 34 :=
by sorry

end father_l34_34585


namespace _l34_34763

-- Here we define our conditions

def parabola (x y : ℝ) := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

example (y_P : ℝ) (hP : parabola 4 y_P) :
  dist (point_on_parabola y_P) focus = 6 := by
  -- Since we only need the theorem statement, we finish with sorry
  sorry

end _l34_34763


namespace math_problem_l34_34619

noncomputable def f (x a : ℝ) : ℝ := -4 * (Real.cos x) ^ 2 + 4 * Real.sqrt 3 * a * (Real.sin x) * (Real.cos x) + 2

theorem math_problem (a : ℝ) :
  (∃ a, ∀ x, f x a = f (π/6 - x) a) →    -- Symmetry condition
  (a = 1 ∧
  ∀ k : ℤ, ∀ x, (x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π) → 
    x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π)) ∧  -- Decreasing intervals
  (∀ x, 2 * x - π / 6 ∈ Set.Icc (-2 * π / 3) (π / 6) → 
    f x a ∈ Set.Icc (-4 : ℝ) 2)) := -- Range on given interval
sorry

end math_problem_l34_34619


namespace find_fixed_point_l34_34814

theorem find_fixed_point (c d k : ℝ) 
(h : ∀ k : ℝ, d = 5 * c^2 + k * c - 3 * k) : (c, d) = (3, 45) :=
sorry

end find_fixed_point_l34_34814


namespace factor_evaluate_l34_34847

theorem factor_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -2) : 
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end factor_evaluate_l34_34847


namespace ticket_cost_is_correct_l34_34022

-- Conditions
def total_amount_raised : ℕ := 620
def number_of_tickets_sold : ℕ := 155

-- Definition of cost per ticket
def cost_per_ticket : ℕ := total_amount_raised / number_of_tickets_sold

-- The theorem to be proven
theorem ticket_cost_is_correct : cost_per_ticket = 4 :=
by
  sorry

end ticket_cost_is_correct_l34_34022


namespace complex_prod_eq_l34_34174

theorem complex_prod_eq (x y z : ℂ) (h1 : x * y + 6 * y = -24) (h2 : y * z + 6 * z = -24) (h3 : z * x + 6 * x = -24) :
  x * y * z = 144 :=
by
  sorry

end complex_prod_eq_l34_34174


namespace george_correct_possible_change_sum_l34_34954

noncomputable def george_possible_change_sum : ℕ :=
if h : ∃ (change : ℕ), change < 100 ∧
  ((change % 25 == 7) ∨ (change % 25 == 32) ∨ (change % 25 == 57) ∨ (change % 25 == 82)) ∧
  ((change % 10 == 2) ∨ (change % 10 == 12) ∨ (change % 10 == 22) ∨
   (change % 10 == 32) ∨ (change % 10 == 42) ∨ (change % 10 == 52) ∨
   (change % 10 == 62) ∨ (change % 10 == 72) ∨ (change % 10 == 82) ∨ (change % 10 == 92)) ∧
  ((change % 5 == 9) ∨ (change % 5 == 14) ∨ (change % 5 == 19) ∨
   (change % 5 == 24) ∨ (change % 5 == 29) ∨ (change % 5 == 34) ∨
   (change % 5 == 39) ∨ (change % 5 == 44) ∨ (change % 5 == 49) ∨
   (change % 5 == 54) ∨ (change % 5 == 59) ∨ (change % 5 == 64) ∨
   (change % 5 == 69) ∨ (change % 5 == 74) ∨ (change % 5 == 79) ∨
   (change % 5 == 84) ∨ (change % 5 == 89) ∨ (change % 5 == 94) ∨ (change % 5 == 99)) then
  114
else 0

theorem george_correct_possible_change_sum :
  george_possible_change_sum = 114 :=
by
  sorry

end george_correct_possible_change_sum_l34_34954


namespace total_weight_full_bucket_l34_34659

theorem total_weight_full_bucket (x y p q : ℝ)
  (h1 : x + (3 / 4) * y = p)
  (h2 : x + (1 / 3) * y = q) :
  x + y = (8 * p - 11 * q) / 5 :=
by
  sorry

end total_weight_full_bucket_l34_34659


namespace persimmons_in_Jungkook_house_l34_34672

-- Define the number of boxes and the number of persimmons per box
def num_boxes : ℕ := 4
def persimmons_per_box : ℕ := 5

-- Define the total number of persimmons calculation
def total_persimmons (boxes : ℕ) (per_box : ℕ) : ℕ := boxes * per_box

-- The main theorem statement proving the total number of persimmons
theorem persimmons_in_Jungkook_house : total_persimmons num_boxes persimmons_per_box = 20 := 
by 
  -- We should prove this, but we use 'sorry' to skip proof in this example.
  sorry

end persimmons_in_Jungkook_house_l34_34672


namespace find_number_of_clerks_l34_34242

-- Define the conditions 
def avg_salary_per_head_staff : ℝ := 90
def avg_salary_officers : ℝ := 600
def avg_salary_clerks : ℝ := 84
def number_of_officers : ℕ := 2

-- Define the variable C (number of clerks)
def number_of_clerks : ℕ := sorry   -- We will prove that this is 170

-- Define the total salary equations based on the conditions
def total_salary_officers := number_of_officers * avg_salary_officers
def total_salary_clerks := number_of_clerks * avg_salary_clerks
def total_number_of_staff := number_of_officers + number_of_clerks
def total_salary := total_salary_officers + total_salary_clerks

-- Define the average salary per head equation 
def avg_salary_eq : Prop := avg_salary_per_head_staff = total_salary / total_number_of_staff

theorem find_number_of_clerks (h : avg_salary_eq) : number_of_clerks = 170 :=
sorry

end find_number_of_clerks_l34_34242


namespace proper_subsets_count_l34_34472

theorem proper_subsets_count (A : Set (Fin 4)) (h : A = {1, 2, 3}) : 
  ∃ n : ℕ, n = 7 ∧ ∃ (S : Finset (Set (Fin 4))), S.card = n ∧ (∀ B, B ∈ S → B ⊂ A) := 
by {
  sorry
}

end proper_subsets_count_l34_34472


namespace max_band_members_l34_34571

theorem max_band_members (r x m : ℕ) (h1 : m < 150) (h2 : r * x + 3 = m) (h3 : (r - 3) * (x + 2) = m) : m = 147 := by
  sorry

end max_band_members_l34_34571


namespace sheets_per_pack_l34_34227

theorem sheets_per_pack (p d t : Nat) (total_sheets : Nat) (sheets_per_pack : Nat) 
  (h1 : p = 2) (h2 : d = 80) (h3 : t = 6) 
  (h4 : total_sheets = d * t)
  (h5 : sheets_per_pack = total_sheets / p) : sheets_per_pack = 240 := 
  by 
    sorry

end sheets_per_pack_l34_34227


namespace determine_valid_m_l34_34281

-- The function given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

-- The range of values for m
def valid_m (m : ℝ) : Prop := -1/4 ≤ m ∧ m ≤ 0

-- The condition that f is increasing on (-∞, 2)
def increasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < a → f x ≤ f y

-- The main statement we want to prove
theorem determine_valid_m (m : ℝ) :
  increasing_on_interval (f m) 2 ↔ valid_m m :=
sorry

end determine_valid_m_l34_34281


namespace amount_with_r_l34_34528

theorem amount_with_r (p q r : ℝ) (h₁ : p + q + r = 7000) (h₂ : r = (2 / 3) * (p + q)) : r = 2800 :=
  sorry

end amount_with_r_l34_34528


namespace geometric_sequence_expression_l34_34076

variable {a : ℕ → ℝ}

-- Define the geometric sequence property
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_expression :
  is_geometric a q →
  a 3 = 2 →
  a 6 = 16 →
  ∀ n, a n = 2^(n-2) := by
  intros h_geom h_a3 h_a6
  sorry

end geometric_sequence_expression_l34_34076


namespace necessary_but_not_sufficient_l34_34028

-- Define the necessary conditions
variables {a b c d : ℝ}

-- State the main theorem
theorem necessary_but_not_sufficient (h₁ : a > b) (h₂ : c > d) : (a + c > b + d) :=
by
  -- Placeholder for the proof (insufficient as per the context problem statement)
  sorry

end necessary_but_not_sufficient_l34_34028


namespace radius_of_larger_ball_l34_34695

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l34_34695


namespace jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l34_34247

theorem jake_first_week_sales :
  let initial_pieces := 80
  let monday_sales := 15
  let tuesday_sales := 2 * monday_sales
  let remaining_pieces := 7
  monday_sales + tuesday_sales + (initial_pieces - (monday_sales + tuesday_sales) - remaining_pieces) = 73 :=
by
  sorry

theorem jake_second_week_sales :
  let monday_sales := 12
  let tuesday_sales := 18
  let wednesday_sales := 20
  let thursday_sales := 11
  let friday_sales := 25
  monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales = 86 :=
by
  sorry

theorem jake_highest_third_week_sales :
  let highest_sales := 40
  highest_sales = 40 :=
by
  sorry

end jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l34_34247


namespace fraction_of_bag_spent_on_lunch_l34_34812

-- Definitions of conditions based on the problem
def initial_amount : ℕ := 158
def price_of_shoes : ℕ := 45
def price_of_bag : ℕ := price_of_shoes - 17
def amount_left : ℕ := 78
def money_before_lunch := amount_left + price_of_shoes + price_of_bag
def money_spent_on_lunch := initial_amount - money_before_lunch 

-- Statement of the problem in Lean
theorem fraction_of_bag_spent_on_lunch :
  (money_spent_on_lunch : ℚ) / price_of_bag = 1 / 4 :=
by
  -- Conditions decoded to match the solution provided
  have h1 : price_of_bag = 28 := by sorry
  have h2 : money_before_lunch = 151 := by sorry
  have h3 : money_spent_on_lunch = 7 := by sorry
  -- The main theorem statement
  exact sorry

end fraction_of_bag_spent_on_lunch_l34_34812


namespace domain_of_function_l34_34810

def domain_conditions (x : ℝ) : Prop :=
  (1 - x ≥ 0) ∧ (x + 2 > 0)

theorem domain_of_function :
  {x : ℝ | domain_conditions x} = {x : ℝ | -2 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l34_34810


namespace max_area_cross_section_rect_prism_l34_34052

/-- The maximum area of the cross-sectional cut of a rectangular prism 
having its vertical edges parallel to the z-axis, with cross-section 
rectangle of sides 8 and 12, whose bottom side lies in the xy-plane 
centered at the origin (0,0,0), cut by the plane 3x + 5y - 2z = 30 
is approximately 118.34. --/
theorem max_area_cross_section_rect_prism :
  ∃ A : ℝ, abs (A - 118.34) < 0.01 :=
sorry

end max_area_cross_section_rect_prism_l34_34052


namespace total_bales_stored_l34_34655

theorem total_bales_stored 
  (initial_bales : ℕ := 540) 
  (new_bales : ℕ := 2) : 
  initial_bales + new_bales = 542 :=
by
  sorry

end total_bales_stored_l34_34655


namespace true_compound_proposition_l34_34631

-- Define conditions and propositions in Lean
def proposition_p : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def proposition_q : Prop := ∀ (x : ℝ), 1 ≤ x → x ≤ 2 → x^2 - 1 ≥ 0

-- Define the compound proposition
def correct_proposition : Prop := ¬ proposition_p ∧ proposition_q

-- Prove the correct compound proposition
theorem true_compound_proposition : correct_proposition :=
by
  sorry

end true_compound_proposition_l34_34631


namespace percent_not_crust_l34_34212

-- Definitions as conditions
def pie_total_weight : ℕ := 200
def crust_weight : ℕ := 50

-- The theorem to be proven
theorem percent_not_crust : (pie_total_weight - crust_weight) / pie_total_weight * 100 = 75 := 
by
  sorry

end percent_not_crust_l34_34212


namespace problem_statement_l34_34754

noncomputable def alpha := 3 + Real.sqrt 8
noncomputable def beta := 3 - Real.sqrt 8
noncomputable def x := alpha ^ 1000
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem problem_statement : x * (1 - f) = 1 :=
by sorry

end problem_statement_l34_34754


namespace radius_of_O2_l34_34723

theorem radius_of_O2 (r_O1 r_dist r_O2 : ℝ) 
  (h1 : r_O1 = 3) 
  (h2 : r_dist = 7) 
  (h3 : (r_dist = r_O1 + r_O2 ∨ r_dist = |r_O2 - r_O1|)) :
  r_O2 = 4 ∨ r_O2 = 10 :=
by
  sorry

end radius_of_O2_l34_34723


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l34_34191

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l34_34191


namespace ned_initial_video_games_l34_34131

theorem ned_initial_video_games : ∀ (w t : ℕ), 7 * w = 63 ∧ t = w + 6 → t = 15 := by
  intro w t
  intro h
  sorry

end ned_initial_video_games_l34_34131


namespace solve_for_x_l34_34686

theorem solve_for_x : ∀ x : ℚ, 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l34_34686


namespace min_cost_for_boxes_l34_34342

def box_volume (l w h : ℕ) : ℕ := l * w * h
def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ := num_boxes * cost_per_box

theorem min_cost_for_boxes : 
  let l := 20
  let w := 20
  let h := 15
  let cost_per_box := (7 : ℚ) / 10
  let total_volume := 3060000
  let volume_box := box_volume l w h
  let num_boxes_needed := total_boxes_needed total_volume volume_box
  (num_boxes_needed = 510) → 
  (total_cost num_boxes_needed cost_per_box = 357) :=
by
  intros
  sorry

end min_cost_for_boxes_l34_34342


namespace ratio_roses_to_lilacs_l34_34012

theorem ratio_roses_to_lilacs
  (L: ℕ) -- number of lilacs sold
  (G: ℕ) -- number of gardenias sold
  (R: ℕ) -- number of roses sold
  (hL: L = 10) -- defining lilacs sold as 10
  (hG: G = L / 2) -- defining gardenias sold as half the lilacs
  (hTotal: R + L + G = 45) -- defining total flowers sold as 45
  : R / L = 3 :=
by {
  -- The actual proof would go here, but we skip it as per instructions
  sorry
}

end ratio_roses_to_lilacs_l34_34012


namespace find_period_l34_34526

variable (x : ℕ)
variable (theo_daily : ℕ := 8)
variable (mason_daily : ℕ := 7)
variable (roxy_daily : ℕ := 9)
variable (total_water : ℕ := 168)

theorem find_period (h : (theo_daily + mason_daily + roxy_daily) * x = total_water) : x = 7 :=
by
  sorry

end find_period_l34_34526


namespace solve_n_m_l34_34600

noncomputable def exponents_of_linear_equation (n m : ℕ) (x y : ℝ) : Prop :=
2 * x ^ (n - 3) - (1 / 3) * y ^ (2 * m + 1) = 0

theorem solve_n_m (n m : ℕ) (x y : ℝ) (h_linear : exponents_of_linear_equation n m x y) :
  n ^ m = 1 :=
sorry

end solve_n_m_l34_34600


namespace number_of_triangles_l34_34517

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 6) (h₂ : 8 > x) : ∃! t, t = 3 :=
by {
  sorry
}

end number_of_triangles_l34_34517


namespace determine_a_values_l34_34905

theorem determine_a_values (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔ a = 2 ∨ a = 8 :=
by
  sorry

end determine_a_values_l34_34905


namespace geometric_sequence_sum_l34_34927

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 0 = 3)
(h_sum : a 0 + a 1 + a 2 = 21) (hq : ∀ n, a (n + 1) = a n * q) : a 2 + a 3 + a 4 = 84 := by
  sorry

end geometric_sequence_sum_l34_34927


namespace gcf_60_90_150_l34_34782

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l34_34782


namespace nth_term_arithmetic_seq_l34_34719

theorem nth_term_arithmetic_seq (a b n t count : ℕ) (h1 : count = 25) (h2 : a = 3) (h3 : b = 75) (h4 : n = 8) :
    t = a + (n - 1) * ((b - a) / (count - 1)) → t = 24 :=
by
  intros
  sorry

end nth_term_arithmetic_seq_l34_34719


namespace part1_part2_l34_34566

def A (x : ℝ) : Prop := x < -2 ∨ x > 3
def B (a : ℝ) (x : ℝ) : Prop := 1 - a < x ∧ x < a + 3

theorem part1 (x : ℝ) : (¬A x ∨ B 1 x) ↔ -2 ≤ x ∧ x < 4 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, ¬(A x ∧ B a x)) ↔ a ≤ 0 :=
by
  sorry

end part1_part2_l34_34566


namespace tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l34_34692

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (x - 1) - 1 / 2 * Real.exp a * x^2

theorem tangent_line_at_origin (a : ℝ) (h : a < 0) : 
  let f₀ := f 0 a
  ∃ c : ℝ, (∀ x : ℝ,  f₀ + c * x = -1) := sorry

theorem local_minimum_at_zero (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, f 0 a ≤ f x a := sorry

theorem number_of_zeros (a : ℝ) (h : a < 0) :
  ∃! x : ℝ, f x a = 0 := sorry

end tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l34_34692


namespace min_xy_l34_34904

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : xy = x + 4 * y + 5) : xy ≥ 25 :=
sorry

end min_xy_l34_34904


namespace number_of_M_subsets_l34_34737

def P : Set ℤ := {0, 1, 2}
def Q : Set ℤ := {0, 2, 4}

theorem number_of_M_subsets (M : Set ℤ) (hP : M ⊆ P) (hQ : M ⊆ Q) : 
  ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_M_subsets_l34_34737


namespace max_cards_l34_34999

def card_cost : ℝ := 0.85
def budget : ℝ := 7.50

theorem max_cards (n : ℕ) : card_cost * n ≤ budget → n ≤ 8 :=
by sorry

end max_cards_l34_34999


namespace solve_rational_numbers_l34_34134

theorem solve_rational_numbers 
  (a b c d : ℚ)
  (h₁ : a + b + c = -1)
  (h₂ : a + b + d = -3)
  (h₃ : a + c + d = 2)
  (h₄ : b + c + d = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := 
by
  sorry

end solve_rational_numbers_l34_34134


namespace shaded_region_area_l34_34061

-- Definitions of known conditions
def grid_section_1_area : ℕ := 3 * 3
def grid_section_2_area : ℕ := 4 * 5
def grid_section_3_area : ℕ := 5 * 6

def total_grid_area : ℕ := grid_section_1_area + grid_section_2_area + grid_section_3_area

def base_of_unshaded_triangle : ℕ := 15
def height_of_unshaded_triangle : ℕ := 6

def unshaded_triangle_area : ℕ := (base_of_unshaded_triangle * height_of_unshaded_triangle) / 2

-- Statement of the problem
theorem shaded_region_area : (total_grid_area - unshaded_triangle_area) = 14 :=
by
  -- Placeholder for the proof
  sorry

end shaded_region_area_l34_34061


namespace johns_friends_count_l34_34150

-- Define the conditions
def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

-- Define the theorem to prove the number of friends John is going with
theorem johns_friends_count (total_cost cost_per_person : ℕ) (h1 : total_cost = 12100) (h2 : cost_per_person = 1100) : (total_cost / cost_per_person) - 1 = 10 := by
  -- Providing the proof is not required, so we use sorry to skip it
  sorry

end johns_friends_count_l34_34150


namespace polynomial_evaluation_l34_34657

-- Define operations using Lean syntax
def star (a b : ℚ) := a + b
def otimes (a b : ℚ) := a - b

-- Define a function to represent the polynomial expression
def expression (a b : ℚ) := star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b)

theorem polynomial_evaluation (a b : ℚ) (ha : a = 5) (hb : b = 3) : expression a b = 435 := by
  sorry

end polynomial_evaluation_l34_34657


namespace sum_first_8_terms_arithmetic_sequence_l34_34828

theorem sum_first_8_terms_arithmetic_sequence (a : ℕ → ℝ) (h : a 4 + a 5 = 12) :
    (8 * (a 1 + a 8)) / 2 = 48 :=
by
  sorry

end sum_first_8_terms_arithmetic_sequence_l34_34828


namespace negation_proposition_l34_34379

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) :=
by
  sorry

end negation_proposition_l34_34379


namespace points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l34_34265

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem points_in_first_quadrant (x y : ℝ) (h : x > 0 ∧ y > 0) : first_quadrant x y :=
by {
  sorry
}

theorem points_in_fourth_quadrant (x y : ℝ) (h : x > 0 ∧ y < 0) : fourth_quadrant x y :=
by {
  sorry
}

theorem points_in_second_quadrant (x y : ℝ) (h : x < 0 ∧ y > 0) : second_quadrant x y :=
by {
  sorry
}

theorem points_in_third_quadrant (x y : ℝ) (h : x < 0 ∧ y < 0) : third_quadrant x y :=
by {
  sorry
}

end points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l34_34265


namespace intersection_subset_complement_l34_34434

open Set

variable (U A B : Set ℕ)

theorem intersection_subset_complement (U : Set ℕ) (A B : Set ℕ) 
  (hU: U = {1, 2, 3, 4, 5, 6}) 
  (hA: A = {1, 3, 5}) 
  (hB: B = {2, 4, 5}) : 
  A ∩ (U \ B) = {1, 3} := 
by
  sorry

end intersection_subset_complement_l34_34434


namespace hunter_movies_count_l34_34670

theorem hunter_movies_count (H : ℕ) 
  (dalton_movies : ℕ := 7)
  (alex_movies : ℕ := 15)
  (together_movies : ℕ := 2)
  (total_movies : ℕ := 30)
  (all_different_movies : dalton_movies + alex_movies - together_movies + H = total_movies) :
  H = 8 :=
by
  -- The mathematical proof will go here
  sorry

end hunter_movies_count_l34_34670


namespace min_x_plus_2y_l34_34990

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : x + 2 * y ≥ (1 / 2) + Real.sqrt 3 :=
sorry

end min_x_plus_2y_l34_34990


namespace train_B_speed_l34_34224

theorem train_B_speed (V_B : ℝ) : 
  (∀ t meet_A meet_B, 
     meet_A = 9 ∧
     meet_B = 4 ∧
     t = 70 ∧
     (t * meet_A) = (V_B * meet_B)) →
     V_B = 157.5 :=
by
  intros h
  sorry

end train_B_speed_l34_34224


namespace find_x0_l34_34392

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 1
else if x < 0 then -x^2 + 1
else 0

theorem find_x0 :
  ∃ x0 : ℝ, f x0 = 1/2 ∧ x0 = -Real.sqrt 2 / 2 :=
by
  sorry

end find_x0_l34_34392


namespace solution_set_of_inequality_l34_34597

theorem solution_set_of_inequality
  (a b : ℝ)
  (x y : ℝ)
  (h1 : a * (-2) + b = 3)
  (h2 : a * (-1) + b = 2)
  :  -x + 1 < 0 ↔ x > 1 :=
by 
  -- Proof goes here
  sorry

end solution_set_of_inequality_l34_34597


namespace inequality_proof_l34_34893

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
by
  sorry

end inequality_proof_l34_34893


namespace less_money_than_Bob_l34_34706

noncomputable def Jennas_money (P: ℝ) : ℝ := 2 * P
noncomputable def Phils_money (B: ℝ) : ℝ := B / 3
noncomputable def Bobs_money : ℝ := 60
noncomputable def Johns_money (P: ℝ) : ℝ := P + 0.35 * P
noncomputable def average (x y: ℝ) : ℝ := (x + y) / 2

theorem less_money_than_Bob :
  ∀ (P Q J B : ℝ),
    P = Phils_money B →
    J = Jennas_money P →
    Q = Johns_money P →
    B = Bobs_money →
    average J Q = B - 0.25 * B →
    B - J = 20
  :=
by
  intros P Q J B hP hJ hQ hB h_avg
  -- Proof goes here
  sorry

end less_money_than_Bob_l34_34706


namespace num_best_friends_l34_34988

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l34_34988


namespace coefficients_sum_eq_four_l34_34513

noncomputable def simplified_coefficients_sum (y : ℚ → ℚ) : ℚ :=
  let A := 1
  let B := 3
  let C := 2
  let D := -2
  A + B + C + D

theorem coefficients_sum_eq_four : simplified_coefficients_sum (λ x => 
  (x^3 + 5*x^2 + 8*x + 4) / (x + 2)) = 4 := by
  sorry

end coefficients_sum_eq_four_l34_34513


namespace num_nonnegative_real_values_l34_34111

theorem num_nonnegative_real_values :
  ∃ n : ℕ, ∀ x : ℝ, (x ≥ 0) → (∃ k : ℕ, (169 - (x^(1/3))) = k^2) → n = 27 := 
sorry

end num_nonnegative_real_values_l34_34111


namespace actual_cost_before_decrease_l34_34004

theorem actual_cost_before_decrease (x : ℝ) (h : 0.76 * x = 1064) : x = 1400 :=
by
  sorry

end actual_cost_before_decrease_l34_34004


namespace cuboid_volume_l34_34864

theorem cuboid_volume (a b c : ℕ) (h_incr_by_2_becomes_cube : c + 2 = a)
  (surface_area_incr : 2*a*(a + a + c + 2) - 2*a*(c + a + b) = 56) : a * b * c = 245 :=
sorry

end cuboid_volume_l34_34864


namespace cherie_sparklers_count_l34_34958

-- Conditions
def koby_boxes : ℕ := 2
def koby_sparklers_per_box : ℕ := 3
def koby_whistlers_per_box : ℕ := 5
def cherie_boxes : ℕ := 1
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := 33

-- Total number of fireworks Koby has
def koby_total_fireworks : ℕ :=
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box)

-- Total number of fireworks Cherie has
def cherie_total_fireworks : ℕ :=
  total_fireworks - koby_total_fireworks

-- Number of sparklers in Cherie's box
def cherie_sparklers : ℕ :=
  cherie_total_fireworks - cherie_whistlers

-- Proof statement
theorem cherie_sparklers_count : cherie_sparklers = 8 := by
  sorry

end cherie_sparklers_count_l34_34958


namespace beta_speed_l34_34889

theorem beta_speed (d : ℕ) (S_s : ℕ) (t : ℕ) (S_b : ℕ) :
  d = 490 ∧ S_s = 37 ∧ t = 7 ∧ (S_s * t) + (S_b * t) = d → S_b = 33 := by
  sorry

end beta_speed_l34_34889


namespace remaining_black_cards_l34_34852

theorem remaining_black_cards 
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (cards_taken_out : ℕ)
  (h1 : total_cards = 52)
  (h2 : black_cards = 26)
  (h3 : red_cards = 26)
  (h4 : cards_taken_out = 5) :
  black_cards - cards_taken_out = 21 := 
by {
  sorry
}

end remaining_black_cards_l34_34852


namespace coin_grid_probability_l34_34668

/--
A square grid is given where the edge length of each smallest square is 6 cm.
A hard coin with a diameter of 2 cm is thrown onto this grid.
Prove that the probability that the coin, after landing, will have a common point with the grid lines is 5/9.
-/
theorem coin_grid_probability :
  let square_edge_cm := 6
  let coin_diameter_cm := 2
  let coin_radius_cm := coin_diameter_cm / 2
  let grid_center_edge_cm := square_edge_cm - coin_diameter_cm
  let non_intersect_area_ratio := (grid_center_edge_cm ^ 2) / (square_edge_cm ^ 2)
  1 - non_intersect_area_ratio = 5 / 9 :=
by
  sorry

end coin_grid_probability_l34_34668


namespace pow_log_sqrt_l34_34421

theorem pow_log_sqrt (a b c : ℝ) (h1 : a = 81) (h2 : b = 500) (h3 : c = 3) :
  ((a ^ (Real.log b / Real.log c)) ^ (1 / 2)) = 250000 :=
by
  sorry

end pow_log_sqrt_l34_34421


namespace remainder_of_large_number_l34_34042

noncomputable def X (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 32
  | 6 => 64
  | 7 => 128
  | 8 => 256
  | 9 => 512
  | 10 => 1024
  | 11 => 2048
  | 12 => 4096
  | 13 => 8192
  | _ => 0

noncomputable def concatenate_X (k : ℕ) : ℕ :=
  if k = 5 then 
    100020004000800160032
  else if k = 11 then 
    100020004000800160032006401280256051210242048
  else if k = 13 then 
    10002000400080016003200640128025605121024204840968192
  else 
    0

theorem remainder_of_large_number :
  (concatenate_X 13) % (concatenate_X 5) = 40968192 :=
by
  sorry

end remainder_of_large_number_l34_34042


namespace not_divisible_by_15_l34_34474

theorem not_divisible_by_15 (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) :=
by
  sorry

end not_divisible_by_15_l34_34474


namespace prove_N_value_l34_34842

theorem prove_N_value (x y N : ℝ) 
  (h1 : N = 4 * x + y) 
  (h2 : 3 * x - 4 * y = 5) 
  (h3 : 7 * x - 3 * y = 23) : 
  N = 86 / 3 := by
  sorry

end prove_N_value_l34_34842


namespace acute_triangle_iff_sum_of_squares_l34_34486

theorem acute_triangle_iff_sum_of_squares (a b c R : ℝ) 
  (hRpos : R > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∀ α β γ, (a = 2 * R * Real.sin α) ∧ (b = 2 * R * Real.sin β) ∧ (c = 2 * R * Real.sin γ) → 
   (α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2)) ↔ 
  (a^2 + b^2 + c^2 > 8 * R^2) :=
sorry

end acute_triangle_iff_sum_of_squares_l34_34486


namespace sector_area_l34_34858

theorem sector_area (theta : ℝ) (L : ℝ) (h_theta : theta = π / 3) (h_L : L = 4) :
  ∃ r : ℝ, (L = r * theta ∧ ∃ A : ℝ, A = 1/2 * r^2 * theta ∧ A = 24 / π) := by
  sorry

end sector_area_l34_34858


namespace college_girls_count_l34_34369

/-- Given conditions:
 1. The ratio of the numbers of boys to girls is 8:5.
 2. The total number of students in the college is 416.
 
 Prove: The number of girls in the college is 160.
 -/
theorem college_girls_count (B G : ℕ) (h1 : B = (8 * G) / 5) (h2 : B + G = 416) : G = 160 :=
by
  sorry

end college_girls_count_l34_34369


namespace line_passes_through_point_has_correct_equation_l34_34277

theorem line_passes_through_point_has_correct_equation :
  (∃ (L : ℝ × ℝ → Prop), (L (-2, 5)) ∧ (∃ m : ℝ, m = -3 / 4 ∧ ∀ (x y : ℝ), L (x, y) ↔ y - 5 = -3 / 4 * (x + 2))) →
  ∀ x y : ℝ, (3 * x + 4 * y - 14 = 0) ↔ (y - 5 = -3 / 4 * (x + 2)) :=
by
  intro h_L
  sorry

end line_passes_through_point_has_correct_equation_l34_34277


namespace assign_students_to_villages_l34_34633

theorem assign_students_to_villages (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  ∃ N : ℕ, N = 70 ∧ 
  (∃ (f : Fin n → Fin m), (∀ i j, f i = f j ↔ i = j) ∧ 
  (∀ x : Fin m, ∃ y : Fin n, f y = x)) :=
by
  sorry

end assign_students_to_villages_l34_34633


namespace dice_faces_l34_34043

theorem dice_faces (n : ℕ) (h : (1 / (n : ℝ)) ^ 5 = 0.0007716049382716049) : n = 10 := sorry

end dice_faces_l34_34043


namespace gcd_of_products_l34_34567

theorem gcd_of_products (a b a' b' d d' : ℕ) (h1 : Nat.gcd a b = d) (h2 : Nat.gcd a' b' = d') (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a') (hb' : 0 < b') :
  Nat.gcd (Nat.gcd (aa') (ab')) (Nat.gcd (ba') (bb')) = d * d' := 
sorry

end gcd_of_products_l34_34567


namespace largest_integer_less_than_100_with_remainder_7_divided_9_l34_34730

theorem largest_integer_less_than_100_with_remainder_7_divided_9 :
  ∃ x : ℕ, (∀ m : ℤ, x = 9 * m + 7 → 9 * m + 7 < 100) ∧ x = 97 :=
sorry

end largest_integer_less_than_100_with_remainder_7_divided_9_l34_34730


namespace Karen_baked_50_cookies_l34_34381

def Karen_kept_cookies : ℕ := 10
def Karen_grandparents_cookies : ℕ := 8
def people_in_class : ℕ := 16
def cookies_per_person : ℕ := 2

theorem Karen_baked_50_cookies :
  Karen_kept_cookies + Karen_grandparents_cookies + (people_in_class * cookies_per_person) = 50 :=
by 
  sorry

end Karen_baked_50_cookies_l34_34381


namespace value_x2012_l34_34083

def f (x : ℝ) : ℝ := sorry

noncomputable def x (n : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom increasing_f : ∀ x y : ℝ, x < y → f x < f y
axiom arithmetic_seq : ∀ n : ℕ, x (n) = x (1) + (n-1) * 2
axiom condition : f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

theorem value_x2012 : x 2012 = 4005 := 
by sorry

end value_x2012_l34_34083


namespace hyperbola_eccentricity_l34_34944

/--
Given a hyperbola with the following properties:
1. Point \( P \) is on the left branch of the hyperbola \( C \): \(\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1\), where \( a > 0 \) and \( b > 0 \).
2. \( F_2 \) is the right focus of the hyperbola.
3. One of the asymptotes of the hyperbola is perpendicular to the line segment \( PF_2 \).

Prove that the eccentricity \( e \) of the hyperbola is \( \sqrt{5} \).
-/
theorem hyperbola_eccentricity (a b e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P_on_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F2_is_focus : True) -- Placeholder for focus-related condition
  (asymptote_perpendicular : True) -- Placeholder for asymptote perpendicular condition
  : e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l34_34944


namespace jo_climb_stairs_ways_l34_34623

def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 3) => f (n + 2) + f (n + 1) + f n

theorem jo_climb_stairs_ways : f 8 = 81 :=
by
    sorry

end jo_climb_stairs_ways_l34_34623


namespace dodecagon_diagonals_l34_34758

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l34_34758


namespace vector_dot_product_result_l34_34368

variable {α : Type*} [Field α]

structure Vector2 (α : Type*) :=
(x : α)
(y : α)

def vector_add (a b : Vector2 α) : Vector2 α :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_sub (a b : Vector2 α) : Vector2 α :=
  ⟨a.x - b.x, a.y - b.y⟩

def dot_product (a b : Vector2 α) : α :=
  a.x * b.x + a.y * b.y

variable (a b : Vector2 ℝ)

theorem vector_dot_product_result
  (h1 : vector_add a b = ⟨1, -3⟩)
  (h2 : vector_sub a b = ⟨3, 7⟩) :
  dot_product a b = -12 :=
by
  sorry

end vector_dot_product_result_l34_34368


namespace simplify_and_evaluate_expression_l34_34984

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 2) : 
  (1 / (x - 3) / (1 / (x^2 - 9)) - x / (x + 1) * ((x^2 + x) / x^2)) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l34_34984


namespace tangent_product_l34_34327

-- Declarations for circles, points of tangency, and radii
variables (R r : ℝ) -- radii of the circles
variables (A B C : ℝ) -- distances related to the tangents

-- Conditions: Two circles, a common internal tangent intersecting at points A and B, tangent at point C
axiom tangent_conditions : A * B = R * r

-- Problem statement: Prove that A * C * C * B = R * r
theorem tangent_product (R r A B C : ℝ) (h : A * B = R * r) : A * C * C * B = R * r :=
by
  sorry

end tangent_product_l34_34327


namespace find_number_l34_34727

theorem find_number (n x : ℝ) (hx : x = 0.8999999999999999) (h : n / x = 0.01) : n = 0.008999999999999999 := by
  sorry

end find_number_l34_34727


namespace skyscraper_anniversary_l34_34207

theorem skyscraper_anniversary (current_year_event future_happens_year target_anniversary_year : ℕ) :
  current_year_event + future_happens_year = target_anniversary_year - 5 →
  target_anniversary_year > current_year_event →
  future_happens_year = 95 := 
by
  sorry

-- Definitions for conditions:
def current_year_event := 100
def future_happens_year := 95
def target_anniversary_year := 200

end skyscraper_anniversary_l34_34207


namespace parakeet_eats_2_grams_per_day_l34_34992

-- Define the conditions
def parrot_daily : ℕ := 14
def finch_daily (parakeet_daily : ℕ) : ℕ := parakeet_daily / 2
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def num_finches : ℕ := 4
def total_weekly_consumption : ℕ := 266

-- Define the daily consumption equation for all birds
def daily_consumption (parakeet_daily : ℕ) : ℕ :=
  num_parakeets * parakeet_daily + num_parrots * parrot_daily + num_finches * finch_daily parakeet_daily

-- Define the weekly consumption equation
def weekly_consumption (parakeet_daily : ℕ) : ℕ :=
  7 * daily_consumption parakeet_daily

-- State the theorem to prove that each parakeet eats 2 grams per day
theorem parakeet_eats_2_grams_per_day :
  (weekly_consumption 2) = total_weekly_consumption ↔ 2 = 2 :=
by
  sorry

end parakeet_eats_2_grams_per_day_l34_34992


namespace area_of_triangle_is_23_over_10_l34_34189

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_triangle_is_23_over_10 :
  let A : ℚ × ℚ := (3, 3)
  let B : ℚ × ℚ := (5, 3)
  let C : ℚ × ℚ := (21 / 5, 19 / 5)
  area_of_triangle A.1 A.2 B.1 B.2 C.1 C.2 = 23 / 10 :=
by
  sorry

end area_of_triangle_is_23_over_10_l34_34189


namespace sum_of_integers_is_19_l34_34950

theorem sum_of_integers_is_19
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : a - b = 5) 
  (h3 : a * b = 84) : 
  a + b = 19 :=
sorry

end sum_of_integers_is_19_l34_34950


namespace cost_of_water_l34_34309

theorem cost_of_water (total_cost sandwiches_cost : ℕ) (num_sandwiches sandwich_price water_price : ℕ) 
  (h1 : total_cost = 11) 
  (h2 : sandwiches_cost = num_sandwiches * sandwich_price) 
  (h3 : num_sandwiches = 3) 
  (h4 : sandwich_price = 3) 
  (h5 : total_cost = sandwiches_cost + water_price) : 
  water_price = 2 :=
by
  sorry

end cost_of_water_l34_34309


namespace sum_digits_B_of_4444_4444_l34_34643

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_digits_B_of_4444_4444 :
  let A : ℕ := sum_digits (4444 ^ 4444)
  let B : ℕ := sum_digits A
  sum_digits B = 7 :=
by
  sorry

end sum_digits_B_of_4444_4444_l34_34643


namespace hamburgers_second_day_l34_34097

theorem hamburgers_second_day (x H D : ℕ) (h1 : 3 * H + 4 * D = 10) (h2 : x * H + 3 * D = 7) (h3 : D = 1) (h4 : H = 2) :
  x = 2 :=
by
  sorry

end hamburgers_second_day_l34_34097


namespace part_a_part_b_part_c_l34_34519

theorem part_a (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b := 
sorry

theorem part_b (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) := 
sorry

theorem part_c (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) :
  ¬ (a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) → 
     a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :=
sorry

end part_a_part_b_part_c_l34_34519


namespace roses_and_orchids_difference_l34_34717

theorem roses_and_orchids_difference :
  let roses_now := 11
  let orchids_now := 20
  orchids_now - roses_now = 9 := 
by
  sorry

end roses_and_orchids_difference_l34_34717


namespace interest_earned_l34_34302

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) := P * (1 + r) ^ t

theorem interest_earned :
  let P := 2000
  let r := 0.05
  let t := 5
  let A := compound_interest P r t
  A - P = 552.56 :=
by
  sorry

end interest_earned_l34_34302


namespace minimum_pieces_for_K_1997_l34_34188

-- Definitions provided by the conditions in the problem.
def is_cube_shaped (n : ℕ) := ∃ (a : ℕ), n = a^3

def has_chocolate_coating (surface_area : ℕ) (n : ℕ) := 
  surface_area = 6 * n^2

def min_pieces (n K : ℕ) := n^3 / K

-- Expressing the proof problem in Lean 4.
theorem minimum_pieces_for_K_1997 {n : ℕ} (h_n : n = 1997) (H : ∀ (K : ℕ), K = 1997 ∧ K > 0) 
  (h_cube : is_cube_shaped n) (h_chocolate : has_chocolate_coating 6 n) :
  min_pieces 1997 1997 = 1997^3 :=
by
  sorry

end minimum_pieces_for_K_1997_l34_34188


namespace problem1_problem2_l34_34243

-- Definitions and assumptions
def p (m : ℝ) : Prop := ∀x y : ℝ, (x^2)/(4 - m) + (y^2)/m = 1 → ∃ c : ℝ, c^2 < (4 - m) ∧ c^2 < m
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ m ≥ 1 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l34_34243


namespace total_valid_votes_l34_34432

theorem total_valid_votes (V : ℕ) (h1 : 0.70 * (V: ℝ) - 0.30 * (V: ℝ) = 184) : V = 460 :=
by sorry

end total_valid_votes_l34_34432


namespace find_x_satisfying_condition_l34_34512

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem find_x_satisfying_condition : ∀ x : ℝ, (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) := by
  sorry

end find_x_satisfying_condition_l34_34512


namespace range_cos_2alpha_cos_2beta_l34_34972

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta_l34_34972


namespace rectangles_in_square_rectangles_in_three_squares_l34_34929

-- Given conditions as definitions
def positive_integer (n : ℕ) : Prop := n > 0

-- Part a
theorem rectangles_in_square (n : ℕ) (h : positive_integer n) :
  (n * (n + 1) / 2) ^ 2 = (n * (n + 1) / 2) ^ 2 :=
by sorry

-- Part b
theorem rectangles_in_three_squares (n : ℕ) (h : positive_integer n) :
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 = 
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 :=
by sorry

end rectangles_in_square_rectangles_in_three_squares_l34_34929


namespace bottles_recycled_l34_34181

theorem bottles_recycled (start_bottles : ℕ) (recycle_ratio : ℕ) (answer : ℕ)
  (h_start : start_bottles = 256) (h_recycle : recycle_ratio = 4) : answer = 85 :=
sorry

end bottles_recycled_l34_34181


namespace find_x_l34_34009

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end find_x_l34_34009


namespace gumballs_difference_l34_34593

theorem gumballs_difference :
  ∃ (x_min x_max : ℕ), 
    19 ≤ (16 + 12 + x_min) / 3 ∧ (16 + 12 + x_min) / 3 ≤ 25 ∧
    19 ≤ (16 + 12 + x_max) / 3 ∧ (16 + 12 + x_max) / 3 ≤ 25 ∧
    (x_max - x_min = 18) :=
by
  sorry

end gumballs_difference_l34_34593


namespace sales_revenue_nonnegative_l34_34789

def revenue (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

theorem sales_revenue_nonnegative (x : ℝ) (hx : x = 9 ∨ x = 11) : revenue x ≥ 15950 :=
by
  cases hx
  case inl h₁ =>
    sorry -- calculation for x = 9
  case inr h₂ =>
    sorry -- calculation for x = 11

end sales_revenue_nonnegative_l34_34789


namespace entree_cost_14_l34_34202

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l34_34202


namespace max_value_of_expression_l34_34784

noncomputable def maxExpression (x y : ℝ) :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  maxExpression x y ≤ (656^2 / 18) :=
by
  sorry

end max_value_of_expression_l34_34784


namespace smallest_n_l34_34757

theorem smallest_n (n : ℕ) (k : ℕ) (a m : ℕ) 
  (h1 : 0 ≤ k)
  (h2 : k < n)
  (h3 : a ≡ k [MOD n])
  (h4 : m > 0) :
  (∀ a m, (∃ k, a = n * k + 5) -> (a^2 - 3*a + 1) ∣ (a^m + 3^m) → false) 
  → n = 11 := sorry

end smallest_n_l34_34757


namespace solve_log_eq_l34_34543

theorem solve_log_eq : ∀ x : ℝ, (2 : ℝ) ^ (Real.log x / Real.log 3) = (1 / 4 : ℝ) → x = 1 / 9 :=
by
  intro x
  sorry

end solve_log_eq_l34_34543


namespace Andy_more_white_socks_than_black_l34_34941

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l34_34941


namespace oranges_for_juice_l34_34587

-- Define conditions
def total_oranges : ℝ := 7 -- in million tons
def export_percentage : ℝ := 0.25
def juice_percentage : ℝ := 0.60

-- Define the mathematical problem
theorem oranges_for_juice : 
  (total_oranges * (1 - export_percentage) * juice_percentage) = 3.2 :=
by
  sorry

end oranges_for_juice_l34_34587


namespace find_principal_sum_l34_34315

theorem find_principal_sum (P : ℝ) (r : ℝ) (A2 : ℝ) (A3 : ℝ) : 
  (A2 = 7000) → (A3 = 9261) → 
  (A2 = P * (1 + r)^2) → (A3 = P * (1 + r)^3) → 
  P = 4000 :=
by
  intro hA2 hA3 hA2_eq hA3_eq
  -- here, we assume the proof steps leading to P = 4000
  sorry

end find_principal_sum_l34_34315


namespace range_of_k_l34_34183

theorem range_of_k (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + k * x + 1 / 2 ≥ 0) → k ∈ Set.Ioc 0 4 := 
by 
  sorry

end range_of_k_l34_34183


namespace part1_part2_l34_34855

variable (a b c x : ℝ)

-- Condition: lengths of the sides of the triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Quadratic equation
def quadratic_eq (x : ℝ) : ℝ := (a + c) * x^2 - 2 * b * x + (a - c)

-- Proof problem 1: If x = 1 is a root, then triangle ABC is isosceles
theorem part1 (h : quadratic_eq a b c 1 = 0) : a = b :=
by
  sorry

-- Proof problem 2: If triangle ABC is equilateral, then roots of the quadratic equation are 0 and 1
theorem part2 (h_eq : a = b ∧ b = c) :
  (quadratic_eq a a a 0 = 0) ∧ (quadratic_eq a a a 1 = 0) :=
by
  sorry

end part1_part2_l34_34855


namespace division_of_polynomial_l34_34994

theorem division_of_polynomial (a : ℤ) : (-28 * a^3) / (7 * a) = -4 * a^2 := by
  sorry

end division_of_polynomial_l34_34994


namespace log_base_2_y_l34_34755

theorem log_base_2_y (y : ℝ) (h : y = (Real.log 3 / Real.log 9) ^ Real.log 27 / Real.log 3) : 
  Real.log y = -3 :=
by
  sorry

end log_base_2_y_l34_34755


namespace total_heads_l34_34271

def total_legs : ℕ := 45
def num_cats : ℕ := 7
def legs_per_cat : ℕ := 4
def captain_legs : ℕ := 1
def legs_humans := total_legs - (num_cats * legs_per_cat) - captain_legs
def num_humans := legs_humans / 2

theorem total_heads : (num_cats + (num_humans + 1)) = 15 := by
  sorry

end total_heads_l34_34271


namespace rajas_income_l34_34293

theorem rajas_income (I : ℝ) 
  (h1 : 0.60 * I + 0.10 * I + 0.10 * I + 5000 = I) : I = 25000 :=
by
  sorry

end rajas_income_l34_34293


namespace simplify_expression_l34_34185

theorem simplify_expression : 1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_expression_l34_34185


namespace mike_total_spent_on_toys_l34_34091

theorem mike_total_spent_on_toys :
  let marbles := 9.05
  let football := 4.95
  let baseball := 6.52
  marbles + football + baseball = 20.52 :=
by
  sorry

end mike_total_spent_on_toys_l34_34091


namespace part1_part2_l34_34298

noncomputable def f (x a : ℝ) : ℝ := |(x - a)| + |(x + 2)|

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≤ 7) : -4 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 * a + 1) : a ≤ 1 :=
by
  sorry

end part1_part2_l34_34298


namespace cos_210_eq_neg_sqrt3_div_2_l34_34551

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l34_34551


namespace largest_expr_is_expr1_l34_34475

def U : ℝ := 3 * 2005 ^ 2006
def V : ℝ := 2005 ^ 2006
def W : ℝ := 2004 * 2005 ^ 2005
def X : ℝ := 3 * 2005 ^ 2005
def Y : ℝ := 2005 ^ 2005
def Z : ℝ := 2005 ^ 2004

def expr1 : ℝ := U - V
def expr2 : ℝ := V - W
def expr3 : ℝ := W - X
def expr4 : ℝ := X - Y
def expr5 : ℝ := Y - Z

theorem largest_expr_is_expr1 : 
  max (max (max expr1 expr2) (max expr3 expr4)) expr5 = expr1 := 
sorry

end largest_expr_is_expr1_l34_34475


namespace amount_of_flour_already_put_in_l34_34414

theorem amount_of_flour_already_put_in 
  (total_flour_needed : ℕ) (flour_remaining : ℕ) (x : ℕ) 
  (h1 : total_flour_needed = 9) 
  (h2 : flour_remaining = 7) 
  (h3 : total_flour_needed - flour_remaining = x) : 
  x = 2 := 
sorry

end amount_of_flour_already_put_in_l34_34414


namespace regular_polygon_sides_l34_34231

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l34_34231


namespace find_a_f_greater_than_1_l34_34311

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) := x^2 * Real.exp x - a * Real.log x

-- Condition: Slope at x = 1 is 3e - 1
theorem find_a (a : ℝ) (h : deriv (fun x => f x a) 1 = 3 * Real.exp 1 - 1) : a = 1 := sorry

-- Given a = 1
theorem f_greater_than_1 (x : ℝ) (hx : x > 0) : f x 1 > 1 := sorry

end find_a_f_greater_than_1_l34_34311


namespace increased_speed_l34_34286

theorem increased_speed
  (d : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) 
  (h1 : d = 2) 
  (h2 : s1 = 2) 
  (h3 : t1 = 1)
  (h4 : t2 = 2 / 3)
  (h5 : s1 * t1 = d)
  (h6 : s2 * t2 = d) :
  s2 - s1 = 1 := 
sorry

end increased_speed_l34_34286


namespace find_n_l34_34913

theorem find_n (n : ℕ) (h : 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012) : n = 1005 :=
sorry

end find_n_l34_34913


namespace fruit_seller_original_apples_l34_34377

theorem fruit_seller_original_apples (x : ℝ) (h : 0.50 * x = 5000) : x = 10000 :=
sorry

end fruit_seller_original_apples_l34_34377


namespace compressor_stations_l34_34060

/-- 
Problem: Given three compressor stations connected by straight roads and not on the same line,
with distances satisfying:
1. x + y = 4z
2. x + z + y = x + a
3. z + y + x = 85

Prove:
- The range of values for 'a' such that the described configuration of compressor stations is 
  possible is 60.71 < a < 68.
- The distances between the compressor stations for a = 5 are x = 70, y = 0, z = 15.
--/
theorem compressor_stations (x y z a : ℝ) 
  (h1 : x + y = 4 * z)
  (h2 : x + z + y = x + a)
  (h3 : z + y + x = 85) :
  (60.71 < a ∧ a < 68) ∧ (a = 5 → x = 70 ∧ y = 0 ∧ z = 15) :=
  sorry

end compressor_stations_l34_34060


namespace students_participated_l34_34075

theorem students_participated (like_dislike_sum : 383 + 431 = 814) : 
  383 + 431 = 814 := 
by exact like_dislike_sum

end students_participated_l34_34075


namespace staff_discount_l34_34345

open Real

theorem staff_discount (d : ℝ) (h : d > 0) (final_price_eq : 0.14 * d = 0.35 * d * (1 - 0.6)) : 0.6 * 100 = 60 :=
by
  sorry

end staff_discount_l34_34345


namespace quadratic_polynomial_with_conditions_l34_34162

theorem quadratic_polynomial_with_conditions :
  ∃ (a b c : ℝ), 
  (∀ x : ℂ, x = -3 - 4 * Complex.I ∨ x = -3 + 4 * Complex.I → a * x^2 + b * x + c = 0)
  ∧ b = -10 
  ∧ a = -5/3 
  ∧ c = -125/3 := 
sorry

end quadratic_polynomial_with_conditions_l34_34162


namespace calculate_expression_l34_34390

theorem calculate_expression : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end calculate_expression_l34_34390


namespace eggs_distribution_l34_34751

theorem eggs_distribution
  (total_eggs : ℕ)
  (eggs_per_adult : ℕ)
  (num_adults : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (eggs_per_girl : ℕ)
  (total_eggs_def : total_eggs = 3 * 12)
  (eggs_per_adult_def : eggs_per_adult = 3)
  (num_adults_def : num_adults = 3)
  (num_girls_def : num_girls = 7)
  (num_boys_def : num_boys = 10)
  (eggs_per_girl_def : eggs_per_girl = 1) :
  ∃ eggs_per_boy : ℕ, eggs_per_boy - eggs_per_girl = 1 :=
by {
  sorry
}

end eggs_distribution_l34_34751


namespace cottonwood_fiber_scientific_notation_l34_34138

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l34_34138


namespace halt_duration_l34_34110

theorem halt_duration (avg_speed : ℝ) (distance : ℝ) (start_time end_time : ℝ) (halt_duration : ℝ) :
  avg_speed = 87 ∧ distance = 348 ∧ start_time = 9 ∧ end_time = 13.75 →
  halt_duration = (end_time - start_time) - (distance / avg_speed) → 
  halt_duration = 0.75 :=
by
  sorry

end halt_duration_l34_34110


namespace length_of_boat_l34_34118

-- Define Josie's jogging variables and problem conditions
variables (L J B : ℝ)
axiom eqn1 : 130 * J = L + 130 * B
axiom eqn2 : 70 * J = L - 70 * B

-- The theorem to prove that the length of the boat L equals 91 steps (i.e., 91 * J)
theorem length_of_boat : L = 91 * J :=
by
  sorry

end length_of_boat_l34_34118


namespace pebbles_difference_l34_34046

def candy_pebbles : Nat := 4
def lance_pebbles : Nat := 3 * candy_pebbles

theorem pebbles_difference {candy_pebbles lance_pebbles : Nat} (h1 : candy_pebbles = 4) (h2 : lance_pebbles = 3 * candy_pebbles) : lance_pebbles - candy_pebbles = 8 := by
  sorry

end pebbles_difference_l34_34046


namespace total_sum_step_l34_34874

-- Defining the conditions
def step_1_sum : ℕ := 2

-- Define the inductive process
def total_sum_labels (n : ℕ) : ℕ :=
  if n = 1 then step_1_sum
  else 2 * 3^(n - 1)

-- The theorem to prove
theorem total_sum_step (n : ℕ) : 
  total_sum_labels n = 2 * 3^(n - 1) :=
by
  sorry

end total_sum_step_l34_34874


namespace latte_cost_l34_34272

theorem latte_cost :
  ∃ (latte_cost : ℝ), 
    2 * 2.25 + 3.50 + 0.50 + 2 * 2.50 + 3.50 + 2 * latte_cost = 25.00 ∧ 
    latte_cost = 4.00 :=
by
  use 4.00
  simp
  sorry

end latte_cost_l34_34272


namespace raghu_investment_is_2200_l34_34688

noncomputable def RaghuInvestment : ℝ := 
  let R := 2200
  let T := 0.9 * R
  let V := 1.1 * T
  if R + T + V = 6358 then R else 0

theorem raghu_investment_is_2200 :
  RaghuInvestment = 2200 := by
  sorry

end raghu_investment_is_2200_l34_34688


namespace Mike_siblings_l34_34739

-- Define the types for EyeColor, HairColor and Sport
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define the Child structure
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define all the children based on the given conditions
def Lily : Child := { name := "Lily", eyeColor := EyeColor.Green, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Mike : Child := { name := "Mike", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Oliver : Child := { name := "Oliver", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Emma : Child := { name := "Emma", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Jacob : Child := { name := "Jacob", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }
def Sophia : Child := { name := "Sophia", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }

-- Siblings relation
def areSiblings (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.favoriteSport = c2.favoriteSport) ∧
  (c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor ∨ c1.favoriteSport = c3.favoriteSport) ∧
  (c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor ∨ c2.favoriteSport = c3.favoriteSport)

-- The proof statement
theorem Mike_siblings : areSiblings Mike Emma Jacob := by
  -- Proof must be provided here
  sorry

end Mike_siblings_l34_34739


namespace count_less_than_threshold_is_zero_l34_34499

def numbers := [0.8, 0.5, 0.9]
def threshold := 0.4

theorem count_less_than_threshold_is_zero :
  (numbers.filter (λ x => x < threshold)).length = 0 :=
by
  sorry

end count_less_than_threshold_is_zero_l34_34499


namespace least_of_10_consecutive_odd_integers_average_154_l34_34163

theorem least_of_10_consecutive_odd_integers_average_154 (x : ℤ)
  (h_avg : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18)) / 10 = 154) :
  x = 145 :=
by 
  sorry

end least_of_10_consecutive_odd_integers_average_154_l34_34163


namespace greatest_possible_sum_l34_34291

noncomputable def eight_products_sum_max : ℕ :=
  let a := 3
  let b := 4
  let c := 5
  let d := 8
  let e := 6
  let f := 7
  7 * (c + d) * (e + f)

theorem greatest_possible_sum (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) :
  eight_products_sum_max = 1183 :=
by
  sorry

end greatest_possible_sum_l34_34291


namespace cylinder_volume_in_sphere_l34_34166

theorem cylinder_volume_in_sphere 
  (h_c : ℝ) (d_s : ℝ) : 
  (h_c = 1) → (d_s = 2) → 
  (π * (d_s / 2)^2 * (h_c / 2) = π / 2) :=
by 
  intros h_c_eq h_s_eq
  sorry

end cylinder_volume_in_sphere_l34_34166


namespace abc_product_l34_34365

theorem abc_product :
  ∃ (a b c P : ℕ), 
    b + c = 3 ∧ 
    c + a = 6 ∧ 
    a + b = 7 ∧ 
    P = a * b * c ∧ 
    P = 10 :=
by sorry

end abc_product_l34_34365


namespace cubic_box_dimension_l34_34329

theorem cubic_box_dimension (a : ℤ) (h: 12 * a = 3 * (a^3)) : a = 2 :=
by
  sorry

end cubic_box_dimension_l34_34329


namespace quadratic_solution_l34_34910

theorem quadratic_solution
  (a c : ℝ) (h : a ≠ 0) (h_passes_through : ∃ b, b = c - 9 * a) :
  ∀ (x : ℝ), (ax^2 - 2 * a * x + c = 0) ↔ (x = -1) ∨ (x = 3) :=
by
  sorry

end quadratic_solution_l34_34910


namespace find_c_value_l34_34702

def projection_condition (v u : ℝ × ℝ) (c : ℝ) : Prop :=
  let v := (5, c)
  let u := (3, 2)
  let dot_product := (v.fst * u.fst + v.snd * u.snd)
  let norm_u_sq := (u.fst^2 + u.snd^2)
  (dot_product / norm_u_sq) * u.fst = -28 / 13 * u.fst

theorem find_c_value : ∃ c : ℝ, projection_condition (5, c) (3, 2) c :=
by
  use -43 / 2
  unfold projection_condition
  sorry

end find_c_value_l34_34702


namespace father_catches_up_l34_34834

noncomputable def min_steps_to_catch_up : Prop :=
  let x := 30
  let father_steps := 5
  let xiaoming_steps := 8
  let distance_ratio := 2 / 5
  let xiaoming_headstart := 27
  ((xiaoming_headstart + (xiaoming_steps / father_steps) * x) / distance_ratio) = x

theorem father_catches_up : min_steps_to_catch_up :=
  by
  sorry

end father_catches_up_l34_34834


namespace number_of_candies_in_a_packet_l34_34595

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l34_34595


namespace find_bc_l34_34070

theorem find_bc (A : ℝ) (a : ℝ) (area : ℝ) (b c : ℝ) :
  A = 60 * (π / 180) → a = Real.sqrt 7 → area = (3 * Real.sqrt 3) / 2 →
  ((b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3)) :=
by
  intros hA ha harea
  -- From the given area condition, derive bc = 6
  have h1 : b * c = 6 := sorry
  -- From the given conditions, derive b + c = 5
  have h2 : b + c = 5 := sorry
  -- Solve the system of equations to find possible values for b and c
  -- Using x² - S⋅x + P = 0 where x are roots, S = b + c, P = b⋅c
  have h3 : (b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3) := sorry
  exact h3

end find_bc_l34_34070


namespace class_average_l34_34160

theorem class_average (n : ℕ) (h₁ : n = 100) (h₂ : 25 ≤ n) 
  (h₃ : 50 ≤ n) (h₄ : 25 * 80 + 50 * 65 + (n - 75) * 90 = 7500) :
  (25 * 80 + 50 * 65 + (n - 75) * 90) / n = 75 := 
by
  sorry

end class_average_l34_34160


namespace max_area_of_triangle_l34_34907

noncomputable def max_triangle_area (v1 v2 v3 : ℝ) (S : ℝ) : Prop :=
  2 * S + Real.sqrt 3 * (v1 * v2 + v3) = 0 ∧ v3 = Real.sqrt 3 → S ≤ Real.sqrt 3 / 4

theorem max_area_of_triangle (v1 v2 v3 S : ℝ) :
  max_triangle_area v1 v2 v3 S :=
by
  sorry

end max_area_of_triangle_l34_34907


namespace flag_count_l34_34679

-- Definitions based on the conditions
def colors : ℕ := 3
def stripes : ℕ := 3

-- The main statement
theorem flag_count : colors ^ stripes = 27 :=
by
  sorry

end flag_count_l34_34679


namespace least_xy_l34_34470

noncomputable def condition (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 1 / 7)

theorem least_xy (x y : ℕ) (h : condition x y) : x * y = 98 :=
sorry

end least_xy_l34_34470


namespace find_y_l34_34363

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z1 (y : ℝ) : ℂ := 3 + y * imaginary_unit

noncomputable def z2 : ℂ := 2 - imaginary_unit

theorem find_y (y : ℝ) (h : z1 y / z2 = 1 + imaginary_unit) : y = 1 :=
by
  sorry

end find_y_l34_34363


namespace total_cost_smore_night_l34_34973

-- Define the costs per item
def cost_graham_cracker : ℝ := 0.10
def cost_marshmallow : ℝ := 0.15
def cost_chocolate : ℝ := 0.25
def cost_caramel_piece : ℝ := 0.20
def cost_toffee_piece : ℝ := 0.05

-- Calculate the cost for each ingredient per S'more
def cost_caramel : ℝ := 2 * cost_caramel_piece
def cost_toffee : ℝ := 4 * cost_toffee_piece

-- Total cost of one S'more
def cost_one_smore : ℝ :=
  cost_graham_cracker + cost_marshmallow + cost_chocolate + cost_caramel + cost_toffee

-- Number of people and S'mores per person
def num_people : ℕ := 8
def smores_per_person : ℕ := 3

-- Total number of S'mores
def total_smores : ℕ := num_people * smores_per_person

-- Total cost of all the S'mores
def total_cost : ℝ := total_smores * cost_one_smore

-- The final statement
theorem total_cost_smore_night : total_cost = 26.40 := 
  sorry

end total_cost_smore_night_l34_34973


namespace bob_distance_when_meet_l34_34846

-- Definitions of the variables and conditions
def distance_XY : ℝ := 40
def yolanda_rate : ℝ := 2  -- Yolanda's walking rate in miles per hour
def bob_rate : ℝ := 4      -- Bob's walking rate in miles per hour
def yolanda_start_time : ℝ := 1 -- Yolanda starts 1 hour earlier 

-- Prove that Bob has walked 25.33 miles when he meets Yolanda
theorem bob_distance_when_meet : 
  ∃ t : ℝ, 2 * (t + yolanda_start_time) + 4 * t = distance_XY ∧ (4 * t = 25.33) := 
by
  sorry

end bob_distance_when_meet_l34_34846


namespace max_sum_of_factors_of_48_l34_34713

theorem max_sum_of_factors_of_48 : ∃ (heartsuit clubsuit : ℕ), heartsuit * clubsuit = 48 ∧ heartsuit + clubsuit = 49 :=
by
  -- We insert sorry here to skip the actual proof construction.
  sorry

end max_sum_of_factors_of_48_l34_34713


namespace range_of_a_l34_34817

theorem range_of_a (a : ℝ) : 
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)) → a > 1 :=
by
  sorry

end range_of_a_l34_34817


namespace minimum_value_m_sq_plus_n_sq_l34_34596

theorem minimum_value_m_sq_plus_n_sq :
  ∃ (m n : ℝ), (m ≠ 0) ∧ (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ (m * x^2 + (2 * n + 1) * x - m - 2) = 0) ∧
  (m^2 + n^2) = 0.01 :=
by
  sorry

end minimum_value_m_sq_plus_n_sq_l34_34596


namespace rectangle_length_width_l34_34133

-- Given conditions
variables (L W : ℕ)

-- Condition 1: The area of the rectangular field is 300 square meters
def area_condition : Prop := L * W = 300

-- Condition 2: The perimeter of the rectangular field is 70 meters
def perimeter_condition : Prop := 2 * (L + W) = 70

-- Condition 3: One side of the rectangle is 20 meters
def side_condition : Prop := L = 20

-- Conclusion
def length_width_proof : Prop :=
  L = 20 ∧ W = 15

-- The final mathematical proof problem statement
theorem rectangle_length_width (L W : ℕ) 
  (h1 : area_condition L W) 
  (h2 : perimeter_condition L W) 
  (h3 : side_condition L) : 
  length_width_proof L W :=
sorry

end rectangle_length_width_l34_34133


namespace original_average_is_6_2_l34_34602

theorem original_average_is_6_2 (n : ℕ) (S : ℚ) (h1 : 6.2 = S / n) (h2 : 6.6 = (S + 4) / n) :
  6.2 = S / n :=
by
  sorry

end original_average_is_6_2_l34_34602


namespace additional_pairs_of_snakes_l34_34837

theorem additional_pairs_of_snakes (total_snakes breeding_balls snakes_per_ball additional_snakes_per_pair : ℕ)
  (h1 : total_snakes = 36) 
  (h2 : breeding_balls = 3)
  (h3 : snakes_per_ball = 8) 
  (h4 : additional_snakes_per_pair = 2) :
  (total_snakes - (breeding_balls * snakes_per_ball)) / additional_snakes_per_pair = 6 :=
by
  sorry

end additional_pairs_of_snakes_l34_34837


namespace find_n_value_l34_34054

theorem find_n_value : (15 * 25 + 20 * 5) = (10 * 25 + 45 * 5) := 
  sorry

end find_n_value_l34_34054


namespace snowdrift_ratio_l34_34750

theorem snowdrift_ratio
  (depth_first_day : ℕ := 20)
  (depth_second_day : ℕ)
  (h1 : depth_second_day + 24 = 34)
  (h2 : depth_second_day = 10) :
  depth_second_day / depth_first_day = 1 / 2 := by
  sorry

end snowdrift_ratio_l34_34750


namespace problem_part_1_problem_part_2_l34_34612

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := Real.log ((x + 2) / (x - 2))

theorem problem_part_1 :
  ∀ (x₁ x₂ : ℝ), 0 < x₂ ∧ x₂ < x₁ → Real.log x₁ + 2 * x₁ > Real.log x₂ + 2 * x₂ :=
sorry

theorem problem_part_2 :
  ∃ k : ℕ, ∀ (x₁ : ℝ), 0 < x₁ ∧ x₁ < 1 → (∃ (x₂ : ℝ), x₂ ∈ Set.Ioo (k : ℝ) (k + 1) ∧ Real.log x₁ + 2 * x₁ < Real.log ((x₂ + 2) / (x₂ - 2))) → k = 2 :=
sorry

end problem_part_1_problem_part_2_l34_34612


namespace zero_in_interval_l34_34803

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) → -- f(x) is increasing on (0, +∞)
  f 2 < 0 → -- f(2) < 0
  f 3 > 0 → -- f(3) > 0
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h_increasing h_f2_lt_0 h_f3_gt_0
  sorry

end zero_in_interval_l34_34803


namespace speed_of_boat_in_still_water_12_l34_34601

theorem speed_of_boat_in_still_water_12 (d b c : ℝ) (h1 : d = (b - c) * 5) (h2 : d = (b + c) * 3) (hb : b = 12) : b = 12 :=
by
  sorry

end speed_of_boat_in_still_water_12_l34_34601


namespace find_savings_l34_34705

def income : ℕ := 15000
def expenditure (I : ℕ) : ℕ := 4 * I / 5
def savings (I E : ℕ) : ℕ := I - E

theorem find_savings : savings income (expenditure income) = 3000 := 
by
  sorry

end find_savings_l34_34705


namespace inverse_g_neg138_l34_34120

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_g_neg138 :
  g (-3) = -138 :=
by
  sorry

end inverse_g_neg138_l34_34120


namespace triangle_area_ratio_l34_34618

theorem triangle_area_ratio
  (a b c : ℕ) (S_triangle : ℕ) -- assuming S_triangle represents the area of the original triangle
  (S_bisected_triangle : ℕ) -- assuming S_bisected_triangle represents the area of the bisected triangle
  (is_angle_bisector : ∀ x y z : ℕ, ∃ k, k = (2 * a * b * c * x) / ((a + b) * (a + c) * (b + c))) :
  S_bisected_triangle = (2 * a * b * c) / ((a + b) * (a + c) * (b + c)) * S_triangle :=
sorry

end triangle_area_ratio_l34_34618


namespace sqrt_sum_simplify_l34_34804

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l34_34804


namespace calc3aMinus4b_l34_34122

theorem calc3aMinus4b (a b : ℤ) (h1 : a * 1 - b * 2 = -1) (h2 : a * 1 + b * 2 = 7) : 3 * a - 4 * b = 1 :=
by
  /- Proof goes here -/
  sorry

end calc3aMinus4b_l34_34122


namespace part_a_part_b_l34_34158

-- Definition for the number of triangles when the n-gon is divided using non-intersecting diagonals
theorem part_a (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = n - 2 := 
sorry

-- Definition for the number of diagonals when the n-gon is divided using non-intersecting diagonals
theorem part_b (n : ℕ) (h : n ≥ 3) : 
  ∃ l, l = n - 3 := 
sorry

end part_a_part_b_l34_34158


namespace sum_first_12_terms_geom_seq_l34_34410

def geometric_sequence_periodic (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem sum_first_12_terms_geom_seq :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 2 ∧
    a 3 = 4 ∧
    geometric_sequence_periodic a 8 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_first_12_terms_geom_seq_l34_34410


namespace floor_S_value_l34_34574

theorem floor_S_value
  (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h_ab_squared : a^2 + b^2 = 1458)
  (h_cd_squared : c^2 + d^2 = 1458)
  (h_ac_product : a * c = 1156)
  (h_bd_product : b * d = 1156) :
  (⌊a + b + c + d⌋ = 77) := 
sorry

end floor_S_value_l34_34574


namespace geometric_seq_condition_l34_34590

/-- In a geometric sequence with common ratio q, sum of the first n terms S_n.
  Given q > 0, show that it is a necessary condition for {S_n} to be an increasing sequence,
  but not a sufficient condition. -/
theorem geometric_seq_condition (a1 q : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h1 : q > 0) : 
  (∀ n, S n < S (n + 1)) ↔ a1 > 0 :=
sorry

end geometric_seq_condition_l34_34590


namespace probability_at_least_one_coordinate_greater_l34_34894

theorem probability_at_least_one_coordinate_greater (p : ℝ) :
  (∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ (x > p ∨ y > p))) ↔ p = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end probability_at_least_one_coordinate_greater_l34_34894


namespace alyosha_possible_l34_34205

theorem alyosha_possible (current_date : ℕ) (day_before_yesterday_age current_year_age next_year_age : ℕ) : 
  (next_year_age = 12 ∧ day_before_yesterday_age = 9 ∧ current_year_age = 12 - 1)
  → (current_date = 1 ∧ current_year_age = 11 → (∃ bday : ℕ, bday = 31)) := 
by
  sorry

end alyosha_possible_l34_34205


namespace wall_length_eq_800_l34_34720

theorem wall_length_eq_800 
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ)
  (num_bricks : ℝ) 
  (brick_volume : ℝ) 
  (total_brick_volume : ℝ)
  (wall_volume : ℝ) :
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_width = 600 → 
  wall_height = 22.5 → 
  num_bricks = 6400 → 
  brick_volume = brick_length * brick_width * brick_height → 
  total_brick_volume = brick_volume * num_bricks → 
  total_brick_volume = wall_volume →
  wall_volume = (800 : ℝ) * wall_width * wall_height :=
by
  sorry

end wall_length_eq_800_l34_34720


namespace mark_total_theater_spending_l34_34433

def week1_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week2_cost : ℝ := (2.5 * 6 - 0.1 * (2.5 * 6)) + 3
def week3_cost : ℝ := 4 * 4 + 3
def week4_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week5_cost : ℝ := (2 * (3.5 * 6 - 0.1 * (3.5 * 6))) + 6
def week6_cost : ℝ := 2 * 7 + 3

def total_cost : ℝ := week1_cost + week2_cost + week3_cost + week4_cost + week5_cost + week6_cost

theorem mark_total_theater_spending : total_cost = 126.30 := sorry

end mark_total_theater_spending_l34_34433


namespace factor_expression_l34_34193

theorem factor_expression (x : ℝ) : 35 * x ^ 13 + 245 * x ^ 26 = 35 * x ^ 13 * (1 + 7 * x ^ 13) :=
by {
  sorry
}

end factor_expression_l34_34193


namespace find_number_l34_34143

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 :=
sorry

end find_number_l34_34143


namespace distance_from_point_A_l34_34337

theorem distance_from_point_A :
  ∀ (A : ℝ) (area : ℝ) (white_area : ℝ) (black_area : ℝ), area = 18 →
  (black_area = 2 * white_area) →
  A = (12 * Real.sqrt 2) / 5 := by
  intros A area white_area black_area h1 h2
  sorry

end distance_from_point_A_l34_34337


namespace relationship_between_m_and_n_l34_34530

theorem relationship_between_m_and_n
  (b m n : ℝ)
  (h₁ : m = 2 * (-1 / 2) + b)
  (h₂ : n = 2 * 2 + b) :
  m < n :=
by
  sorry

end relationship_between_m_and_n_l34_34530


namespace last_two_digits_of_product_squared_l34_34898

def mod_100 (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_product_squared :
  mod_100 ((301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2) = 76 := 
by
  sorry

end last_two_digits_of_product_squared_l34_34898


namespace max_sum_condition_l34_34896

theorem max_sum_condition (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : Nat.gcd a b = 6) : a + b ≤ 186 :=
sorry

end max_sum_condition_l34_34896


namespace ratio_eval_l34_34915

universe u

def a : ℕ := 121
def b : ℕ := 123
def c : ℕ := 122

theorem ratio_eval : (2 ^ a * 3 ^ b) / (6 ^ c) = (3 / 2) := by
  sorry

end ratio_eval_l34_34915


namespace monthly_expenses_last_month_l34_34437

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l34_34437


namespace initial_money_equals_26_l34_34911

def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5
def money_left : ℕ := 8

def total_cost_items : ℕ := cost_jumper + cost_tshirt + cost_heels

theorem initial_money_equals_26 : total_cost_items + money_left = 26 := by
  sorry

end initial_money_equals_26_l34_34911


namespace pyramid_coloring_ways_l34_34919

theorem pyramid_coloring_ways (colors : Fin 5) 
  (coloring_condition : ∀ (a b : Fin 5), a ≠ b) :
  ∃ (ways: Nat), ways = 420 :=
by
  -- Given:
  -- 1. There are 5 available colors
  -- 2. Each vertex of the pyramid is colored differently from the vertices connected by an edge
  -- Prove:
  -- There are 420 ways to color the pyramid's vertices
  sorry

end pyramid_coloring_ways_l34_34919


namespace train_crossing_time_l34_34210

-- Define the problem conditions in Lean 4
def train_length : ℕ := 130
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := (speed_kmph * 1000 / 3600)

-- The statement to prove
theorem train_crossing_time : (train_length + bridge_length) / speed_mps = 28 :=
by
  -- The proof starts here
  sorry

end train_crossing_time_l34_34210


namespace number_of_triangles_l34_34320

open Nat

/-- Each side of a square is divided into 8 equal parts, and using the divisions
as vertices (not including the vertices of the square), the number of different 
triangles that can be obtained is 3136. -/
theorem number_of_triangles (n : ℕ := 7) :
  (n * 4).choose 3 - 4 * n.choose 3 = 3136 := 
sorry

end number_of_triangles_l34_34320


namespace tangent_slope_of_cubic_l34_34426

theorem tangent_slope_of_cubic (P : ℝ × ℝ) (tangent_at_P : ℝ) (h1 : P.snd = P.fst ^ 3)
  (h2 : tangent_at_P = 3) : P = (1,1) ∨ P = (-1,-1) :=
by
  sorry

end tangent_slope_of_cubic_l34_34426


namespace dispatch_plans_count_l34_34192

theorem dispatch_plans_count:
  -- conditions
  let total_athletes := 9
  let basketball_players := 5
  let soccer_players := 6
  let both_players := 2
  let only_basketball := 3
  let only_soccer := 4
  -- proof
  (both_players.choose 2 + both_players * only_basketball + both_players * only_soccer + only_basketball * only_soccer) = 28 :=
by
  sorry

end dispatch_plans_count_l34_34192


namespace num_values_f100_eq_0_l34_34578

def f0 (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

def fn : ℕ → ℝ → ℝ
| 0, x   => f0 x
| (n+1), x => |fn n x| - 1

theorem num_values_f100_eq_0 : ∃ (xs : Finset ℝ), ∀ x ∈ xs, fn 100 x = 0 ∧ xs.card = 301 :=
by
  sorry

end num_values_f100_eq_0_l34_34578


namespace polar_circle_equation_l34_34822

theorem polar_circle_equation (ρ θ : ℝ) (O pole : ℝ) (eq_line : ρ * Real.cos θ + ρ * Real.sin θ = 2) :
  (∃ ρ, ρ = 2 * Real.cos θ) :=
sorry

end polar_circle_equation_l34_34822


namespace remaining_hours_needed_l34_34768

noncomputable
def hours_needed_to_finish (x : ℚ) : Prop :=
  (1/5 : ℚ) * (2 + x) + (1/8 : ℚ) * x = 1

theorem remaining_hours_needed :
  ∃ x : ℚ, hours_needed_to_finish x ∧ x = 24/13 :=
by
  use 24/13
  sorry

end remaining_hours_needed_l34_34768


namespace interval_of_increase_l34_34400

noncomputable def u (x : ℝ) : ℝ := x^2 - 5*x + 6

def increasing_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := Real.log (u x)

theorem interval_of_increase :
  increasing_interval f {x : ℝ | 3 < x} :=
sorry

end interval_of_increase_l34_34400


namespace derivative_at_zero_l34_34257
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem derivative_at_zero : (deriv f 0) = -120 :=
by
  -- The proof is omitted
  sorry

end derivative_at_zero_l34_34257


namespace k_value_for_z_perfect_square_l34_34126

theorem k_value_for_z_perfect_square (Z K : ℤ) (h1 : 500 < Z ∧ Z < 1000) (h2 : K > 1) (h3 : Z = K * K^2) :
  ∃ K : ℤ, Z = 729 ∧ K = 9 :=
by {
  sorry
}

end k_value_for_z_perfect_square_l34_34126


namespace total_balls_without_holes_l34_34764

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end total_balls_without_holes_l34_34764


namespace not_valid_base_five_l34_34553

theorem not_valid_base_five (k : ℕ) (h₁ : k = 5) : ¬(∀ d ∈ [3, 2, 5, 0, 1], d < k) :=
by
  sorry

end not_valid_base_five_l34_34553


namespace solve_for_x_l34_34872

theorem solve_for_x : ∃ x : ℝ, x^4 + 10 * x^3 + 9 * x^2 - 50 * x - 56 = 0 ↔ x = -2 :=
by
  sorry

end solve_for_x_l34_34872


namespace james_vegetable_consumption_l34_34402

def vegetable_consumption_weekdays (asparagus broccoli cauliflower spinach : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + spinach

def vegetable_consumption_weekend (asparagus broccoli cauliflower other_veg : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + other_veg

def total_vegetable_consumption (
  wd_asparagus wd_broccoli wd_cauliflower wd_spinach : ℚ)
  (sat_asparagus sat_broccoli sat_cauliflower sat_other : ℚ)
  (sun_asparagus sun_broccoli sun_cauliflower sun_other : ℚ) : ℚ :=
  5 * vegetable_consumption_weekdays wd_asparagus wd_broccoli wd_cauliflower wd_spinach +
  vegetable_consumption_weekend sat_asparagus sat_broccoli sat_cauliflower sat_other +
  vegetable_consumption_weekend sun_asparagus sun_broccoli sun_cauliflower sun_other

theorem james_vegetable_consumption :
  total_vegetable_consumption 0.5 0.75 0.875 0.5 0.3 0.4 0.6 1 0.3 0.4 0.6 0.5 = 17.225 :=
sorry

end james_vegetable_consumption_l34_34402


namespace students_tried_out_l34_34167

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
  (h1 : not_picked = 36) (h2 : groups = 4) (h3 : students_per_group = 7) :
  not_picked + groups * students_per_group = 64 :=
by
  sorry

end students_tried_out_l34_34167


namespace technician_round_trip_completion_percentage_l34_34139

theorem technician_round_trip_completion_percentage :
  ∀ (d total_d : ℝ),
  d = 1 + (0.75 * 1) + (0.5 * 1) + (0.25 * 1) →
  total_d = 4 * 2 →
  (d / total_d) * 100 = 31.25 :=
by
  intros d total_d h1 h2
  sorry

end technician_round_trip_completion_percentage_l34_34139


namespace SamLastPage_l34_34248

theorem SamLastPage (total_pages : ℕ) (Sam_read_time : ℕ) (Lily_read_time : ℕ) (last_page : ℕ) :
  total_pages = 920 ∧ Sam_read_time = 30 ∧ Lily_read_time = 50 → last_page = 575 :=
by
  intros h
  sorry

end SamLastPage_l34_34248


namespace cafeteria_pies_l34_34832

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h_initial : initial_apples = 50)
  (h_handed_out : handed_out_apples = 5)
  (h_apples_per_pie : apples_per_pie = 5) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := 
by
  sorry

end cafeteria_pies_l34_34832


namespace find_y_given_conditions_l34_34352

theorem find_y_given_conditions (a x y : ℝ) (h1 : y = a * x + (1 - a)) 
  (x_val : x = 3) (y_val : y = 7) (x_new : x = 8) :
  y = 22 := 
  sorry

end find_y_given_conditions_l34_34352


namespace problem1_problem2_problem3_l34_34229

noncomputable def a_n (n : ℕ) : ℕ := 3 * (2 ^ n) - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * a_n n - 3 * n

-- 1. Prove a_1 = 3 and a_2 = 9 given S_n = 2a_n - 3n
theorem problem1 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    a_n 1 = 3 ∧ a_n 2 = 9 :=
  sorry

-- 2. Prove that the sequence {a_n + 3} is a geometric sequence and find the general term formula for the sequence {a_n}.
theorem problem2 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    ∀ n, (a_n (n + 1) + 3) / (a_n n + 3) = 2 ∧ a_n n = 3 * (2 ^ n) - 3 :=
  sorry

-- 3. Prove {S_{n_k}} is not an arithmetic sequence given S_n = 2a_n - 3n and {n_k} is an arithmetic sequence
theorem problem3 (n_k : ℕ → ℕ) (h_arithmetic : ∃ d, ∀ k, n_k (k + 1) - n_k k = d) :
    ¬ ∃ d, ∀ k, S_n (n_k (k + 1)) - S_n (n_k k) = d :=
  sorry

end problem1_problem2_problem3_l34_34229


namespace total_rainfall_l34_34971

theorem total_rainfall (R1 R2 : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 15) : R1 + R2 = 25 := 
by
  sorry

end total_rainfall_l34_34971


namespace find_x_l34_34347

theorem find_x (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2 * x = A - 2 * x + 30)
  (h3 : (A + R / 2) + (V + R / 2) = 120)
  (h4 : S - 0.25 * S + 10 = 2 * (R / 2)) :
  x = 5 :=
  sorry

end find_x_l34_34347


namespace perpendicular_lines_condition_l34_34391

theorem perpendicular_lines_condition (a : ℝ) :
  (6 * a + 3 * 4 = 0) ↔ (a = -2) :=
sorry

end perpendicular_lines_condition_l34_34391


namespace Hillary_activities_LCM_l34_34632

theorem Hillary_activities_LCM :
  let swim := 6
  let run := 4
  let cycle := 16
  Nat.lcm (Nat.lcm swim run) cycle = 48 :=
by
  sorry

end Hillary_activities_LCM_l34_34632


namespace total_meters_built_l34_34234

/-- Define the length of the road -/
def road_length (L : ℕ) := L = 1000

/-- Define the average meters built per day -/
def average_meters_per_day (A : ℕ) := A = 120

/-- Define the number of days worked from July 29 to August 2 -/
def number_of_days_worked (D : ℕ) := D = 5

/-- The total meters built by the time they finished -/
theorem total_meters_built
  (L A D : ℕ)
  (h1 : road_length L)
  (h2 : average_meters_per_day A)
  (h3 : number_of_days_worked D)
  : L / D * A = 600 := by
  sorry

end total_meters_built_l34_34234


namespace final_value_of_A_l34_34935

theorem final_value_of_A (A : ℝ) (h1: A = 15) (h2: A = -A + 5) : A = -10 :=
sorry

end final_value_of_A_l34_34935


namespace number_of_cards_above_1999_l34_34664

def numberOfCardsAbove1999 (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if numberOfCardsAbove1999 (n-1) = n-2 then 1
  else numberOfCardsAbove1999 (n-1) + 2

theorem number_of_cards_above_1999 : numberOfCardsAbove1999 2000 = 927 := by
  sorry

end number_of_cards_above_1999_l34_34664


namespace point_P_trajectory_circle_l34_34638

noncomputable def trajectory_of_point_P (d h1 h2 : ℝ) (x y : ℝ) : Prop :=
  (x - d/2)^2 + y^2 = (h1^2 + h2^2) / (2 * (h2/h1)^(2/3))

theorem point_P_trajectory_circle :
  ∀ (d h1 h2 x y : ℝ),
  d = 20 →
  h1 = 15 →
  h2 = 10 →
  (∃ x y, trajectory_of_point_P d h1 h2 x y) →
  (∃ x y, (x - 16)^2 + y^2 = 24^2) :=
by
  intros d h1 h2 x y hd hh1 hh2 hxy
  sorry

end point_P_trajectory_circle_l34_34638


namespace friends_pay_6_22_l34_34582

noncomputable def cost_per_friend : ℕ :=
  let hamburgers := 5 * 3
  let fries := 4 * 120 / 100
  let soda := 5 * 50 / 100
  let spaghetti := 270 / 100
  let milkshakes := 3 * 250 / 100
  let nuggets := 2 * 350 / 100
  let total_bill := hamburgers + fries + soda + spaghetti + milkshakes + nuggets
  let discount := total_bill * 10 / 100
  let discounted_bill := total_bill - discount
  let birthday_friend := discounted_bill * 30 / 100
  let remaining_amount := discounted_bill - birthday_friend
  remaining_amount / 4

theorem friends_pay_6_22 : cost_per_friend = 622 / 100 :=
by
  sorry

end friends_pay_6_22_l34_34582


namespace quarters_addition_l34_34856

def original_quarters : ℝ := 783.0
def added_quarters : ℝ := 271.0
def total_quarters : ℝ := 1054.0

theorem quarters_addition :
  original_quarters + added_quarters = total_quarters :=
by
  sorry

end quarters_addition_l34_34856


namespace line_through_points_on_parabola_l34_34413

theorem line_through_points_on_parabola 
  (x1 y1 x2 y2 : ℝ)
  (h_parabola_A : y1^2 = 4 * x1)
  (h_parabola_B : y2^2 = 4 * x2)
  (h_midpoint : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  ∃ (m b : ℝ), m = 1 ∧ b = 2 ∧ (∀ x y : ℝ, y = m * x + b ↔ x - y = 0) :=
sorry

end line_through_points_on_parabola_l34_34413


namespace quadratic_vertex_problem_l34_34562

/-- 
    Given a quadratic equation y = ax^2 + bx + c, where (2, -3) 
    is the vertex of the parabola and it passes through (0, 1), 
    prove that a - b + c = 6. 
-/
theorem quadratic_vertex_problem 
    (a b c : ℤ)
    (h : ∀ x : ℝ, y = a * (x - 2)^2 - 3)
    (h_point : y = 1)
    (h_passes_through_origin : y = a * (0 - 2)^2 - 3) :
    a - b + c = 6 :=
sorry

end quadratic_vertex_problem_l34_34562


namespace celebration_women_count_l34_34373

theorem celebration_women_count (num_men : ℕ) (num_pairs : ℕ) (pairs_per_man : ℕ) (pairs_per_woman : ℕ) 
  (hm : num_men = 15) (hpm : pairs_per_man = 4) (hwp : pairs_per_woman = 3) (total_pairs : num_pairs = num_men * pairs_per_man) : 
  num_pairs / pairs_per_woman = 20 :=
by
  sorry

end celebration_women_count_l34_34373


namespace identical_digits_time_l34_34290

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l34_34290


namespace ken_paid_20_l34_34473

section
variable (pound_price : ℤ) (pounds_bought : ℤ) (change_received : ℤ)
variable (total_cost : ℤ) (amount_paid : ℤ)

-- Conditions
def price_per_pound := 7  -- A pound of steak costs $7
def pounds_bought_value := 2  -- Ken bought 2 pounds of steak
def change_received_value := 6  -- Ken received $6 back after paying

-- Intermediate Calculations
def total_cost_of_steak := pounds_bought_value * price_per_pound  -- Total cost of steak
def amount_paid_calculated := total_cost_of_steak + change_received_value  -- Amount paid based on total cost and change received

-- Problem Statement
theorem ken_paid_20 : (total_cost_of_steak = total_cost) ∧ (amount_paid_calculated = amount_paid) -> amount_paid = 20 :=
by
  intros h
  sorry
end

end ken_paid_20_l34_34473


namespace width_of_metallic_sheet_is_36_l34_34808

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end width_of_metallic_sheet_is_36_l34_34808


namespace standing_next_to_boris_l34_34579

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l34_34579


namespace additional_savings_if_purchase_together_l34_34220

theorem additional_savings_if_purchase_together :
  let price_per_window := 100
  let windows_each_offer := 4
  let free_each_offer := 1
  let dave_windows := 7
  let doug_windows := 8

  let cost_without_offer (windows : Nat) := windows * price_per_window
  let cost_with_offer (windows : Nat) := 
    if windows % (windows_each_offer + free_each_offer) = 0 then
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window
    else
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window 
      + (windows % (windows_each_offer + free_each_offer)) * price_per_window

  (cost_without_offer (dave_windows + doug_windows) 
  - cost_with_offer (dave_windows + doug_windows)) 
  - ((cost_without_offer dave_windows - cost_with_offer dave_windows)
  + (cost_without_offer doug_windows - cost_with_offer doug_windows)) = price_per_window := 
  sorry

end additional_savings_if_purchase_together_l34_34220


namespace inscribed_circles_radii_sum_l34_34132

noncomputable def sum_of_radii (d : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 + r2 = d / 2

theorem inscribed_circles_radii_sum (d : ℝ) (h : d = 23) (r1 r2 : ℝ) (h1 : r1 + r2 = d / 2) :
  r1 + r2 = 23 / 2 :=
by
  rw [h] at h1
  exact h1

end inscribed_circles_radii_sum_l34_34132


namespace bisection_method_next_interval_l34_34685

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x0 := (a + b) / 2
  (f a * f x0 < 0) ∨ (f x0 * f b < 0) →
  (x0 = 2.5) →
  f 2 * f 2.5 < 0 :=
by
  intros
  sorry

end bisection_method_next_interval_l34_34685


namespace difference_of_quarters_l34_34598

variables (n d q : ℕ)

theorem difference_of_quarters :
  (n + d + q = 150) ∧ (5 * n + 10 * d + 25 * q = 1425) →
  (∃ qmin qmax : ℕ, q = qmax - qmin ∧ qmax - qmin = 30) :=
by
  sorry

end difference_of_quarters_l34_34598


namespace solve_inequality_l34_34059

theorem solve_inequality (x : ℝ) (h : 3 - (1 / (3 * x + 4)) < 5) : 
  x ∈ { x : ℝ | x < -11/6 } ∨ x ∈ { x : ℝ | x > -4/3 } :=
by
  sorry

end solve_inequality_l34_34059


namespace mistaken_divisor_is_12_l34_34605

-- Definitions based on conditions
def correct_divisor : ℕ := 21
def correct_quotient : ℕ := 36
def mistaken_quotient : ℕ := 63

-- The mistaken divisor  is computed as:
def mistaken_divisor : ℕ := correct_quotient * correct_divisor / mistaken_quotient

-- The theorem to prove the mistaken divisor is 12
theorem mistaken_divisor_is_12 : mistaken_divisor = 12 := by
  sorry

end mistaken_divisor_is_12_l34_34605


namespace arithmetic_sequence_a2015_l34_34280

theorem arithmetic_sequence_a2015 :
  (∀ n : ℕ, n > 0 → (∃ a_n a_n1 : ℝ,
    a_n1 = a_n + 2 ∧ a_n + a_n1 = 4 * n - 58))
  → (∃ a_2015 : ℝ, a_2015 = 4000) :=
by
  intro h
  sorry

end arithmetic_sequence_a2015_l34_34280


namespace point_on_transformed_graph_l34_34968

variable (f : ℝ → ℝ)

theorem point_on_transformed_graph :
  (f 12 = 10) →
  3 * (19 / 9) = (f (3 * 4)) / 3 + 3 ∧ (4 + 19 / 9 = 55 / 9) :=
by
  sorry

end point_on_transformed_graph_l34_34968


namespace not_possible_linear_poly_conditions_l34_34583

theorem not_possible_linear_poly_conditions (a b : ℝ):
    ¬ (abs (b - 1) < 1 ∧ abs (a + b - 3) < 1 ∧ abs (2 * a + b - 9) < 1) := 
by
    sorry

end not_possible_linear_poly_conditions_l34_34583


namespace simplify_expression_l34_34439

theorem simplify_expression (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 :=
sorry

end simplify_expression_l34_34439


namespace average_population_increase_l34_34431

-- Conditions
def population_2000 : ℕ := 450000
def population_2005 : ℕ := 467000
def years : ℕ := 5

-- Theorem statement
theorem average_population_increase :
  (population_2005 - population_2000) / years = 3400 := by
  sorry

end average_population_increase_l34_34431


namespace eval_expression_eq_one_l34_34488

theorem eval_expression_eq_one (x : ℝ) (hx1 : x^3 + 1 = (x+1)*(x^2 - x + 1)) (hx2 : x^3 - 1 = (x-1)*(x^2 + x + 1)) :
  ( ((x+1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * ((x-1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 ) = 1 :=
by
  sorry

end eval_expression_eq_one_l34_34488


namespace decrease_percent_in_revenue_l34_34931

-- Definitions based on the conditions
def original_tax (T : ℝ) := T
def original_consumption (C : ℝ) := C
def new_tax (T : ℝ) := 0.70 * T
def new_consumption (C : ℝ) := 1.20 * C

-- Theorem statement for the decrease percent in revenue
theorem decrease_percent_in_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  100 * ((original_tax T * original_consumption C - new_tax T * new_consumption C) / (original_tax T * original_consumption C)) = 16 :=
by
  sorry

end decrease_percent_in_revenue_l34_34931


namespace students_bought_pencils_l34_34497

theorem students_bought_pencils (h1 : 2 * 2 + 6 * 3 + 2 * 1 = 24) : 
  2 + 6 + 2 = 10 := by
  sorry

end students_bought_pencils_l34_34497


namespace final_result_after_subtracting_15_l34_34481

theorem final_result_after_subtracting_15 :
  ∀ (n : ℕ) (r : ℕ) (f : ℕ),
  n = 120 → 
  r = n / 6 → 
  f = r - 15 → 
  f = 5 :=
by
  intros n r f hn hr hf
  have h1 : n = 120 := hn
  have h2 : r = n / 6 := hr
  have h3 : f = r - 15 := hf
  sorry

end final_result_after_subtracting_15_l34_34481


namespace nth_equation_l34_34916

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by
  sorry

end nth_equation_l34_34916


namespace number_of_routes_from_A_to_L_is_6_l34_34697

def A_to_B_or_E : Prop := True
def B_to_A_or_C_or_F : Prop := True
def C_to_B_or_D_or_G : Prop := True
def D_to_C_or_H : Prop := True
def E_to_A_or_F_or_I : Prop := True
def F_to_B_or_E_or_G_or_J : Prop := True
def G_to_C_or_F_or_H_or_K : Prop := True
def H_to_D_or_G_or_L : Prop := True
def I_to_E_or_J : Prop := True
def J_to_F_or_I_or_K : Prop := True
def K_to_G_or_J_or_L : Prop := True
def L_from_H_or_K : Prop := True

theorem number_of_routes_from_A_to_L_is_6 
  (h1 : A_to_B_or_E)
  (h2 : B_to_A_or_C_or_F)
  (h3 : C_to_B_or_D_or_G)
  (h4 : D_to_C_or_H)
  (h5 : E_to_A_or_F_or_I)
  (h6 : F_to_B_or_E_or_G_or_J)
  (h7 : G_to_C_or_F_or_H_or_K)
  (h8 : H_to_D_or_G_or_L)
  (h9 : I_to_E_or_J)
  (h10 : J_to_F_or_I_or_K)
  (h11 : K_to_G_or_J_or_L)
  (h12 : L_from_H_or_K) : 
  6 = 6 := 
by 
  sorry

end number_of_routes_from_A_to_L_is_6_l34_34697


namespace avg_highway_mpg_l34_34442

noncomputable def highway_mpg (total_distance : ℕ) (fuel : ℕ) : ℝ :=
  total_distance / fuel
  
theorem avg_highway_mpg :
  highway_mpg 305 25 = 12.2 :=
by
  sorry

end avg_highway_mpg_l34_34442


namespace general_formula_a_sum_bn_l34_34715

noncomputable section

open Nat

-- Define the sequence Sn
def S (n : ℕ) : ℕ := 2^n + n - 1

-- Define the sequence an
def a (n : ℕ) : ℕ := 1 + 2^(n-1)

-- Define the sequence bn
def b (n : ℕ) : ℕ := 2 * n * (a n - 1)

-- Define the sum Tn
def T (n : ℕ) : ℕ := n * 2^n

-- Proposition 1: General formula for an
theorem general_formula_a (n : ℕ) : a n = 1 + 2^(n-1) :=
by
  sorry

-- Proposition 2: Sum of first n terms of bn
theorem sum_bn (n : ℕ) : T n = 2 + (n - 1) * 2^(n+1) :=
by
  sorry

end general_formula_a_sum_bn_l34_34715


namespace final_score_is_correct_l34_34839

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l34_34839


namespace relationship_among_abc_l34_34926

noncomputable def a : ℝ := Real.sqrt 6 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 5 + Real.sqrt 8
def c : ℝ := 5

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l34_34926


namespace trig_proof_1_trig_proof_2_l34_34304

variables {α : ℝ}

-- Given condition
def tan_alpha (a : ℝ) := Real.tan a = -3

-- Proof problem statement
theorem trig_proof_1 (h : tan_alpha α) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 := sorry

theorem trig_proof_2 (h : tan_alpha α) :
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := sorry

end trig_proof_1_trig_proof_2_l34_34304


namespace find_points_and_min_ordinate_l34_34674

noncomputable def pi : Real := Real.pi
noncomputable def sin : Real → Real := Real.sin
noncomputable def cos : Real → Real := Real.cos

def within_square (x y : Real) : Prop :=
  -pi ≤ x ∧ x ≤ pi ∧ 0 ≤ y ∧ y ≤ 2 * pi

def satisfies_system (x y : Real) : Prop :=
  sin x + sin y = sin 2 ∧ cos x + cos y = cos 2

theorem find_points_and_min_ordinate :
  ∃ (points : List (Real × Real)), 
    (∀ (p : Real × Real), p ∈ points → within_square p.1 p.2 ∧ satisfies_system p.1 p.2) ∧
    points.length = 2 ∧
    ∃ (min_point : Real × Real), min_point ∈ points ∧ ∀ (p : Real × Real), p ∈ points → min_point.2 ≤ p.2 ∧ min_point = (2 + Real.pi / 3, 2 - Real.pi / 3) :=
by
  sorry

end find_points_and_min_ordinate_l34_34674


namespace polynomial_transformation_l34_34825

variable {x y : ℝ}

theorem polynomial_transformation
  (h : y = x + 1/x) 
  (poly_eq_0 : x^4 + x^3 - 5*x^2 + x + 1 = 0) :
  x^2 * (y^2 + y - 7) = 0 :=
sorry

end polynomial_transformation_l34_34825


namespace probability_all_from_same_tribe_l34_34787

-- Definitions based on the conditions of the problem
def total_people := 24
def tribe_count := 3
def people_per_tribe := 8
def quitters := 3

-- We assume each person has an equal chance of quitting and the quitters are chosen independently
-- The probability that all three people who quit belong to the same tribe

theorem probability_all_from_same_tribe :
  ((3 * (Nat.choose people_per_tribe quitters)) / (Nat.choose total_people quitters) : ℚ) = 1 / 12 := 
  by 
    sorry

end probability_all_from_same_tribe_l34_34787


namespace Kolya_Homework_Problem_l34_34570

-- Given conditions as definitions
def squaresToDigits (x : ℕ) (a b : ℕ) : Prop := x^2 = 10 * a + b
def doubledToDigits (x : ℕ) (a b : ℕ) : Prop := 2 * x = 10 * b + a

-- The main theorem statement
theorem Kolya_Homework_Problem :
  ∃ (x a b : ℕ), squaresToDigits x a b ∧ doubledToDigits x a b ∧ x = 9 ∧ x^2 = 81 :=
by
  -- proof skipped
  sorry

end Kolya_Homework_Problem_l34_34570


namespace negation_of_proposition_l34_34836

namespace NegationProp

theorem negation_of_proposition :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - x < 0) ↔
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ x0^2 - x0 ≥ 0) := by sorry

end NegationProp

end negation_of_proposition_l34_34836


namespace angle_B_value_triangle_perimeter_l34_34811

open Real

variables {A B C a b c : ℝ}

-- Statement 1
theorem angle_B_value (h1 : a = b * sin A + sqrt 3 * a * cos B) : B = π / 2 := by
  sorry

-- Statement 2
theorem triangle_perimeter 
  (h1 : B = π / 2)
  (h2 : b = 4)
  (h3 : (1 / 2) * a * c = 4) : 
  a + b + c = 4 + 4 * sqrt 2 := by
  sorry


end angle_B_value_triangle_perimeter_l34_34811


namespace max_value_200_max_value_attained_l34_34157

noncomputable def max_value (X Y Z : ℕ) : ℕ := 
  X * Y * Z + X * Y + Y * Z + Z * X

theorem max_value_200 (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  max_value X Y Z ≤ 200 :=
sorry

theorem max_value_attained (X Y Z : ℕ) (h : X = 5) (h1 : Y = 5) (h2 : Z = 5) : 
  max_value X Y Z = 200 :=
sorry

end max_value_200_max_value_attained_l34_34157


namespace negation_of_original_prop_l34_34169

variable (a : ℝ)
def original_prop (x : ℝ) : Prop := x^2 + a * x + 1 < 0

theorem negation_of_original_prop :
  ¬ (∃ x : ℝ, original_prop a x) ↔ ∀ x : ℝ, ¬ original_prop a x :=
by sorry

end negation_of_original_prop_l34_34169


namespace exists_square_all_invisible_l34_34696

open Nat

theorem exists_square_all_invisible (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → gcd (a + i) (b + j) > 1 := 
sorry

end exists_square_all_invisible_l34_34696


namespace number_of_albums_l34_34525

-- Definitions for the given conditions
def pictures_from_phone : ℕ := 7
def pictures_from_camera : ℕ := 13
def pictures_per_album : ℕ := 4

-- We compute the total number of pictures
def total_pictures : ℕ := pictures_from_phone + pictures_from_camera

-- Statement: Prove the number of albums is 5
theorem number_of_albums :
  total_pictures / pictures_per_album = 5 := by
  sorry

end number_of_albums_l34_34525


namespace mitchell_pizzas_l34_34897

def pizzas_bought (slices_per_goal goals_per_game games slices_per_pizza : ℕ) : ℕ :=
  (slices_per_goal * goals_per_game * games) / slices_per_pizza

theorem mitchell_pizzas : pizzas_bought 1 9 8 12 = 6 := by
  sorry

end mitchell_pizzas_l34_34897


namespace perfect_square_trinomial_l34_34316

theorem perfect_square_trinomial (x : ℝ) : 
  let a := x
  let b := 1 / 2
  2 * a * b = x :=
by
  sorry

end perfect_square_trinomial_l34_34316


namespace sum_f_positive_l34_34524

variable (a b c : ℝ)

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) :
  f a + f b + f c > 0 :=
sorry

end sum_f_positive_l34_34524


namespace lcm_condition_proof_l34_34937

theorem lcm_condition_proof (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2 * n)
  (h4 : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > n * 2 / 3 := 
sorry

end lcm_condition_proof_l34_34937


namespace determine_x_l34_34467

theorem determine_x (x : ℕ) 
  (hx1 : x % 6 = 0) 
  (hx2 : x^2 > 196) 
  (hx3 : x < 30) : 
  x = 18 ∨ x = 24 := 
sorry

end determine_x_l34_34467


namespace treasure_chest_total_value_l34_34743

def base7_to_base10 (n : Nat) : Nat :=
  let rec convert (n acc base : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * base) (base * 7)
  convert n 0 1

theorem treasure_chest_total_value :
  base7_to_base10 5346 + base7_to_base10 6521 + base7_to_base10 320 = 4305 :=
by
  sorry

end treasure_chest_total_value_l34_34743


namespace candy_mixture_l34_34436

theorem candy_mixture (x : ℝ) (h1 : x * 3 + 64 * 2 = (x + 64) * 2.2) : x + 64 = 80 :=
by sorry

end candy_mixture_l34_34436


namespace exist_c_l34_34466

theorem exist_c (p : ℕ) (r : ℤ) (a b : ℤ) [Fact (Nat.Prime p)]
  (hp1 : r^7 ≡ 1 [ZMOD p])
  (hp2 : r + 1 - a^2 ≡ 0 [ZMOD p])
  (hp3 : r^2 + 1 - b^2 ≡ 0 [ZMOD p]) :
  ∃ c : ℤ, (r^3 + 1 - c^2) ≡ 0 [ZMOD p] :=
by
  sorry

end exist_c_l34_34466


namespace Sara_Jim_equal_savings_l34_34740

theorem Sara_Jim_equal_savings:
  ∃ (w : ℕ), (∃ (sara_saved jim_saved : ℕ),
  sara_saved = 4100 + 10 * w ∧
  jim_saved = 15 * w ∧
  sara_saved = jim_saved) → w = 820 :=
by
  sorry

end Sara_Jim_equal_savings_l34_34740


namespace frog_jump_l34_34690

def coprime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem frog_jump (p q : ℕ) (h_coprime : coprime p q) :
  ∀ d : ℕ, d < p + q → (∃ m n : ℤ, m ≠ n ∧ (m - n = d ∨ n - m = d)) :=
by
  sorry

end frog_jump_l34_34690


namespace obtain_x_squared_obtain_xy_l34_34646

theorem obtain_x_squared (x y : ℝ) (hx : x ≠ 1) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x^2 :=
by
  sorry

theorem obtain_xy (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x * y :=
by
  sorry

end obtain_x_squared_obtain_xy_l34_34646


namespace Paula_initial_cans_l34_34020

theorem Paula_initial_cans :
  ∀ (cans rooms_lost : ℕ), rooms_lost = 10 → 
  (40 / (rooms_lost / 5) = cans + 5 → cans = 20) :=
by
  intros cans rooms_lost h_rooms_lost h_calculation
  sorry

end Paula_initial_cans_l34_34020


namespace no_integral_roots_l34_34460

theorem no_integral_roots :
  ¬(∃ (x : ℤ), 5 * x^2 + 3 = 40) ∧
  ¬(∃ (x : ℤ), (3 * x - 2)^3 = (x - 2)^3 - 27) ∧
  ¬(∃ (x : ℤ), x^2 - 4 = 3 * x - 4) :=
by sorry

end no_integral_roots_l34_34460


namespace square_of_99_l34_34835

theorem square_of_99 : 99 * 99 = 9801 :=
by sorry

end square_of_99_l34_34835


namespace find_d_minus_c_l34_34749

theorem find_d_minus_c (c d x : ℝ) (h : c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) : (d - c = 45) :=
  sorry

end find_d_minus_c_l34_34749


namespace solution_set_f_2_minus_x_l34_34278

def f (x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_f_2_minus_x (a b : ℝ) (h_even : b - 2 * a = 0)
  (h_mono : 0 < a) :
  {x : ℝ | f (2 - x) a b > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_2_minus_x_l34_34278


namespace johns_average_speed_l34_34833

def start_time := 8 * 60 + 15  -- 8:15 a.m. in minutes
def end_time := 14 * 60 + 45   -- 2:45 p.m. in minutes
def break_start := 12 * 60     -- 12:00 p.m. in minutes
def break_duration := 30       -- 30 minutes
def total_distance := 240      -- Total distance in miles

def total_driving_time : ℕ := 
  (break_start - start_time) + (end_time - (break_start + break_duration))

def average_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / (time / 60)  -- converting time from minutes to hours

theorem johns_average_speed :
  average_speed total_distance total_driving_time = 40 :=
by
  sorry

end johns_average_speed_l34_34833


namespace min_commission_deputies_l34_34216

theorem min_commission_deputies 
  (members : ℕ) 
  (brawls : ℕ) 
  (brawl_participants : brawls = 200) 
  (member_count : members = 200) :
  ∃ minimal_commission_members : ℕ, minimal_commission_members = 67 := 
sorry

end min_commission_deputies_l34_34216


namespace grassy_plot_width_l34_34868

noncomputable def gravel_cost (L w p : ℝ) : ℝ :=
  0.80 * ((L + 2 * p) * (w + 2 * p) - L * w)

theorem grassy_plot_width
  (L : ℝ) 
  (p : ℝ) 
  (cost : ℝ) 
  (hL : L = 110) 
  (hp : p = 2.5) 
  (hcost : cost = 680) :
  ∃ w : ℝ, gravel_cost L w p = cost ∧ w = 97.5 :=
by
  sorry

end grassy_plot_width_l34_34868


namespace zero_in_interval_l34_34106

theorem zero_in_interval {b : ℝ} (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * b * x - 3 * b + 1)
  (h₂ : b > 1/5)
  (h₃ : b < 1) :
  ∃ x, -1 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l34_34106


namespace graveling_cost_is_969_l34_34527

-- Definitions for lawn dimensions
def lawn_length : ℝ := 75
def lawn_breadth : ℝ := 45

-- Definitions for road widths and costs
def road1_width : ℝ := 6
def road1_cost_per_sq_meter : ℝ := 0.90

def road2_width : ℝ := 5
def road2_cost_per_sq_meter : ℝ := 0.85

def road3_width : ℝ := 4
def road3_cost_per_sq_meter : ℝ := 0.80

def road4_width : ℝ := 3
def road4_cost_per_sq_meter : ℝ := 0.75

-- Calculate the area of each road
def road1_area : ℝ := road1_width * lawn_length
def road2_area : ℝ := road2_width * lawn_length
def road3_area : ℝ := road3_width * lawn_breadth
def road4_area : ℝ := road4_width * lawn_breadth

-- Calculate the cost of graveling each road
def road1_graveling_cost : ℝ := road1_area * road1_cost_per_sq_meter
def road2_graveling_cost : ℝ := road2_area * road2_cost_per_sq_meter
def road3_graveling_cost : ℝ := road3_area * road3_cost_per_sq_meter
def road4_graveling_cost : ℝ := road4_area * road4_cost_per_sq_meter

-- Calculate the total cost
def total_graveling_cost : ℝ := 
  road1_graveling_cost + road2_graveling_cost + road3_graveling_cost + road4_graveling_cost

-- Statement to be proved
theorem graveling_cost_is_969 : total_graveling_cost = 969 := by
  sorry

end graveling_cost_is_969_l34_34527


namespace symmetric_points_origin_l34_34077

theorem symmetric_points_origin (a b : ℝ) 
  (h1 : (-2, b) = (-a, -3)) : a - b = 5 := 
by
  -- solution steps are not included in the statement
  sorry

end symmetric_points_origin_l34_34077


namespace rectangle_original_length_doubles_area_l34_34669

-- Let L and W denote the length and width of a rectangle respectively
-- Given the condition: (L + 2)W = 2LW
-- We need to prove that L = 2

theorem rectangle_original_length_doubles_area (L W : ℝ) (h : (L + 2) * W = 2 * L * W) : L = 2 :=
by 
  sorry

end rectangle_original_length_doubles_area_l34_34669


namespace infinitely_many_solutions_l34_34033

theorem infinitely_many_solutions (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := sorry

end infinitely_many_solutions_l34_34033


namespace batting_average_is_60_l34_34884

-- Definitions for conditions:
def highest_score : ℕ := 179
def difference_highest_lowest : ℕ := 150
def average_44_innings : ℕ := 58
def innings_excluding_highest_lowest : ℕ := 44
def total_innings : ℕ := 46

-- Lowest score
def lowest_score : ℕ := highest_score - difference_highest_lowest

-- Total runs in 44 innings
def total_runs_44 : ℕ := average_44_innings * innings_excluding_highest_lowest

-- Total runs in 46 innings
def total_runs_46 : ℕ := total_runs_44 + highest_score + lowest_score

-- Batting average in 46 innings
def batting_average_46 : ℕ := total_runs_46 / total_innings

-- The theorem to prove
theorem batting_average_is_60 :
  batting_average_46 = 60 :=
sorry

end batting_average_is_60_l34_34884


namespace will_buy_5_toys_l34_34238

theorem will_buy_5_toys (initial_money spent_money toy_cost money_left toys : ℕ) 
  (h1 : initial_money = 57) 
  (h2 : spent_money = 27) 
  (h3 : toy_cost = 6) 
  (h4 : money_left = initial_money - spent_money) 
  (h5 : toys = money_left / toy_cost) : 
  toys = 5 := 
by
  sorry

end will_buy_5_toys_l34_34238


namespace sum_of_coordinates_l34_34485

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 3) : 
  let x := 2 / 3
  let y := 2 * f (3 * x) + 4
  x + y = 32 / 3 :=
by
  sorry

end sum_of_coordinates_l34_34485


namespace slope_intercept_parallel_l34_34240

theorem slope_intercept_parallel (A : ℝ × ℝ) (x y : ℝ) (hA : A = (3, 2))
(hparallel : 4 * x + y - 2 = 0) :
  ∃ b : ℝ, y = -4 * x + b ∧ b = 14 :=
by
  sorry

end slope_intercept_parallel_l34_34240


namespace cubic_root_expression_l34_34746

theorem cubic_root_expression (p q r : ℝ)
  (h₁ : p + q + r = 8)
  (h₂ : p * q + p * r + q * r = 11)
  (h₃ : p * q * r = 3) :
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 32 / 15 :=
by 
  sorry

end cubic_root_expression_l34_34746


namespace m_value_l34_34886

open Polynomial

noncomputable def f (m : ℚ) : Polynomial ℚ := X^4 - 5*X^2 + 4*X - C m

theorem m_value (m : ℚ) : (2 * X + 1) ∣ f m ↔ m = -51/16 := by sorry

end m_value_l34_34886


namespace power_equality_l34_34141

theorem power_equality (n : ℝ) : (9:ℝ)^4 = (27:ℝ)^n → n = (8:ℝ) / 3 :=
by
  sorry

end power_equality_l34_34141


namespace total_study_time_is_60_l34_34934

-- Define the times Elizabeth studied for each test
def science_time : ℕ := 25
def math_time : ℕ := 35

-- Define the total study time
def total_study_time : ℕ := science_time + math_time

-- Proposition that the total study time equals 60 minutes
theorem total_study_time_is_60 : total_study_time = 60 := by
  /-
  Here we would provide the proof steps, but since the task is to write the statement only,
  we add 'sorry' to indicate the missing proof.
  -/
  sorry

end total_study_time_is_60_l34_34934


namespace total_earnings_to_afford_car_l34_34861

-- Define the earnings per month
def monthlyEarnings : ℕ := 4000

-- Define the savings per month
def monthlySavings : ℕ := 500

-- Define the total amount needed to buy the car
def totalNeeded : ℕ := 45000

-- Define the number of months needed to save enough money
def monthsToSave : ℕ := totalNeeded / monthlySavings

-- Theorem stating the total money earned before he saves enough to buy the car
theorem total_earnings_to_afford_car : monthsToSave * monthlyEarnings = 360000 := by
  sorry

end total_earnings_to_afford_car_l34_34861


namespace cannot_be_square_of_difference_formula_l34_34947

theorem cannot_be_square_of_difference_formula (x y c d a b m n : ℝ) :
  ¬ ((m - n) * (-m + n) = (x^2 - y^2) ∨ 
       (m - n) * (-m + n) = (c^2 - d^2) ∨ 
       (m - n) * (-m + n) = (a^2 - b^2)) :=
by sorry

end cannot_be_square_of_difference_formula_l34_34947


namespace avg_length_one_third_wires_l34_34088

theorem avg_length_one_third_wires (x : ℝ) (L1 L2 L3 L4 L5 L6 : ℝ) 
  (h_total_wires : L1 + L2 + L3 + L4 + L5 + L6 = 6 * 80) 
  (h_avg_other_wires : (L3 + L4 + L5 + L6) / 4 = 85) 
  (h_avg_all_wires : (L1 + L2 + L3 + L4 + L5 + L6) / 6 = 80) :
  (L1 + L2) / 2 = 70 :=
by
  sorry

end avg_length_one_third_wires_l34_34088


namespace perimeter_of_inner_polygon_le_outer_polygon_l34_34376

-- Definitions of polygons (for simplicity considered as list of points or sides)
structure Polygon where
  sides : List ℝ  -- assuming sides lengths are given as list of real numbers
  convex : Prop   -- a property stating that the polygon is convex

-- Definition of the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := p.sides.sum

-- Conditions from the problem
variable {P_in P_out : Polygon}
variable (h_convex_in : P_in.convex) (h_convex_out : P_out.convex)
variable (h_inside : ∀ s ∈ P_in.sides, s ∈ P_out.sides) -- simplifying the "inside" condition

-- The theorem statement
theorem perimeter_of_inner_polygon_le_outer_polygon :
  perimeter P_in ≤ perimeter P_out :=
by {
  sorry
}

end perimeter_of_inner_polygon_le_outer_polygon_l34_34376


namespace find_b_amount_l34_34328

theorem find_b_amount (A B : ℝ) (h1 : A + B = 100) (h2 : (3 / 10) * A = (1 / 5) * B) : B = 60 := 
by 
  sorry

end find_b_amount_l34_34328


namespace percentage_loss_l34_34386

theorem percentage_loss (CP SP : ℝ) (hCP : CP = 1400) (hSP : SP = 1148) : 
  (CP - SP) / CP * 100 = 18 := by 
  sorry

end percentage_loss_l34_34386


namespace amount_after_two_years_l34_34888

-- Definition of initial amount and the rate of increase
def initial_value : ℝ := 32000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

-- The compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- The proof problem: Prove that after 2 years the amount is 40500
theorem amount_after_two_years : compound_interest initial_value rate_of_increase time_period = 40500 :=
sorry

end amount_after_two_years_l34_34888


namespace correct_sampling_method_l34_34214

structure SchoolPopulation :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)

-- Define the school population
def school : SchoolPopulation :=
  { senior := 10, intermediate := 50, junior := 75 }

-- Define the condition for sampling method
def total_school_teachers (s : SchoolPopulation) : ℕ :=
  s.senior + s.intermediate + s.junior

-- The desired sample size
def sample_size : ℕ := 30

-- The correct sampling method based on the population strata
def stratified_sampling (s : SchoolPopulation) : Prop :=
  s.senior + s.intermediate + s.junior > 0

theorem correct_sampling_method : stratified_sampling school :=
by { sorry }

end correct_sampling_method_l34_34214


namespace simplify_fraction_l34_34275

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l34_34275


namespace fraction_unspent_is_correct_l34_34348

noncomputable def fraction_unspent (S : ℝ) : ℝ :=
  let after_tax := S - 0.15 * S
  let after_first_week := after_tax - 0.25 * after_tax
  let after_second_week := after_first_week - 0.3 * after_first_week
  let after_third_week := after_second_week - 0.2 * S
  let after_fourth_week := after_third_week - 0.1 * after_third_week
  after_fourth_week / S

theorem fraction_unspent_is_correct (S : ℝ) (hS : S > 0) : 
  fraction_unspent S = 0.221625 :=
by
  sorry

end fraction_unspent_is_correct_l34_34348


namespace total_students_in_class_l34_34960

-- Definitions of the conditions
def E : ℕ := 55
def T : ℕ := 85
def N : ℕ := 30
def B : ℕ := 20

-- Statement of the theorem to prove the total number of students
theorem total_students_in_class : (E + T - B) + N = 150 := by
  -- Proof is omitted
  sorry

end total_students_in_class_l34_34960


namespace min_value_expression_l34_34797

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (v : ℝ), (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ v) ∧ v = 30 :=
by
  sorry

end min_value_expression_l34_34797


namespace watch_arrangement_count_l34_34146

noncomputable def number_of_satisfying_watch_arrangements : Nat :=
  let dial_arrangements := Nat.factorial 2
  let strap_arrangements := Nat.factorial 3
  dial_arrangements * strap_arrangements

theorem watch_arrangement_count :
  number_of_satisfying_watch_arrangements = 12 :=
by
-- Proof omitted
sorry

end watch_arrangement_count_l34_34146


namespace function_machine_output_l34_34522

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_l34_34522


namespace smallest_y_square_factor_l34_34744

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l34_34744


namespace seokjin_paper_count_l34_34504

theorem seokjin_paper_count :
  ∀ (jimin_paper seokjin_paper : ℕ),
  jimin_paper = 41 →
  jimin_paper = seokjin_paper + 1 →
  seokjin_paper = 40 :=
by
  intros jimin_paper seokjin_paper h_jimin h_relation
  sorry

end seokjin_paper_count_l34_34504


namespace probability_females_not_less_than_males_l34_34800

noncomputable def prob_female_not_less_than_male : ℚ :=
  let total_students := 5
  let females := 2
  let males := 3
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose females 2 + females * males
  favorable_combinations / total_combinations

theorem probability_females_not_less_than_males (total_students females males : ℕ) :
  total_students = 5 → females = 2 → males = 3 →
  prob_female_not_less_than_male = 7 / 10 :=
by intros; sorry

end probability_females_not_less_than_males_l34_34800


namespace seating_arrangement_l34_34305

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem seating_arrangement : 
  let republicans := 6
  let democrats := 4
  (factorial (republicans - 1)) * (binom republicans democrats) * (factorial democrats) = 43200 :=
by
  sorry

end seating_arrangement_l34_34305


namespace straight_line_cannot_intersect_all_segments_l34_34314

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end straight_line_cannot_intersect_all_segments_l34_34314
